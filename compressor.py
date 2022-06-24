import os
import torch
import time
import datetime
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model import get_model

from utils.utils import to_var, write_print, write_to_file, save_plots, get_amp_gt_by_value
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, mean_squared_error, mean_absolute_error)
from utils.timer import Timer
import numpy as np
from utils.marunet_losses import cal_avg_ms_ssim
import torch.nn.functional as F

from flopco import FlopCo
from compression.musco.pytorch import CompressorVBMF
from compression.musco.pytorch.compressor.rank_selection.estimator import estimate_rank_for_compression_rate, estimate_vbmf_ranks
from collections import defaultdict
import copy
import shutil

from compression.skt_utils import AverageMeter, cal_para
from compression.skt_distillation import cosine_similarity, scale_process, cal_dense_fsp, upsample_process


class Compressor(object):

    DEFAULTS = {}

    def __init__(self, version, data_loaders, dataset_ids, config, output_txt, compile_txt):
        """
        Initializes a Solver object
        """

        # data loader
        self.__dict__.update(Compressor.DEFAULTS, **config)
        self.version = version
        self.data_loaders = data_loaders
        self.dataset_ids = dataset_ids
        self.output_txt = output_txt
        self.compile_txt = compile_txt

        self.dataset_info = self.dataset
        if (self.dataset == 'micc'):
            self.dataset_info = '{} {}'.format(self.dataset, self.dataset_subcategory)

        # self.lr = self.best_models[self.dataset][self.model]['lr']
        self.build_model()

        # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        rand_seed = 64678
        if rand_seed is not None:
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)

    def compress(self):

        if self.compression == 'skt':
            raise Exception("Not yet implemented")

        elif self.compression == 'musco':
            # self.build_model()
            self.musco()

        else:
            raise Exception("Compression technique ({}) not implemented".format(self.compression))

    def build_model(self):
        self.model_name = self.model
        self.model = get_model(self.model,
                               self.backbone_model,
                               self.imagenet_pretrain,
                               self.model_save_path,
                               self.input_channels,
                               self.class_count)

    def build_optimizer_loss(self, model):
        criterion = nn.MSELoss()

        optimizer = None
        if 'MARUNet' in self.model_name or self.compression == 'skt':
            optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=self.lr)

        return optimizer, criterion

    def print_network(self, model, name):
        """
        Prints the structure of the network and the total number of parameters
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        write_print(self.output_txt, name)
        write_print(self.output_txt, str(model))
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))

    def print_num_params(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))

    # def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """

        # if "MUSCO test" in self.pretrained_model:
        #     checkpoint = torch.load(os.path.join(
        #         '{}.pth.tar'.format(self.pretrained_model)))
        #     self.model.load_state_dict(checkpoint['musco_best_state_dict'])

        #     self.optimizer, self.criterion = self.build_optimizer_loss(self.model)
        #     self.optimizer.load_state_dict(checkpoint['musco_best_optim'])

        #     return

        # self.model.load_state_dict(torch.load(os.path.join(
        #     '{}.pth'.format(self.pretrained_model))))
        # write_print(self.output_txt,
        #             'loaded trained model {}'.format(self.pretrained_model))
    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """

        if ".pth.tar" not in self.pretrained_model:
            self.model.load_state_dict(torch.load(os.path.join(
                '{}.pth'.format(self.pretrained_model))))

        else:
            weights = torch.load('{}'.format(self.pretrained_model))
            self.model.load_state_dict(weights['state_dict'])

        write_print(self.output_txt,
                    'loaded trained model {}'.format(self.pretrained_model))

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        # torch.save(state, os.path.join(path, filename))

        torch.save(state, os.path.join(
            self.output_txt[:self.output_txt.rfind('\\')]),
            filename)
        
    def print_loss_log(self, start_time, iters_per_epoch, e, i, loss, num_epochs):
        """
        Prints the loss and elapsed time for each epoch
        """
        total_iter = num_epochs * iters_per_epoch
        cur_iter = e * iters_per_epoch + i

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)
        epoch_time = (iters_per_epoch - i) * elapsed / (cur_iter + 1)

        epoch_time = str(datetime.timedelta(seconds=epoch_time))
        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {}/{} -- {}, Epoch [{}/{}], Iter [{}/{}], " \
              "loss: {:.15f}".format(elapsed,
                                    epoch_time,
                                    total_time,
                                    e + 1,
                                    num_epochs,
                                    i + 1,
                                    iters_per_epoch,
                                    loss)

        write_print(self.output_txt, log)

    '''
        MUSCO IMPLEMENTATION
    '''

    def musco(self):
        if self.use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        img, _ = next(iter(self.data_loaders['train']))
        img_size = tuple(img.shape)


        self.model.to(device)
        model_stats = FlopCo(self.model, img_size = img_size, device = device)

        lnames_to_compress = [k for k in model_stats.flops.keys() if\
                      model_stats.ltypes[k]['type'] == nn.Conv2d and model_stats.ltypes[k]['kernel_size'] != (1, 1)
                      and 'out' not in k and 'back0.0' not in k and 'amp' not in k and 'brg' not in k
                      and not ('conv4' in k and 'SKT' in self.model_name)
                      ]

        if self.model_name == 'CSRNet':
            lnames_to_compress = [l for l in lnames_to_compress if\
                     'backend' not in l]

        print(lnames_to_compress)

        wf = 0.55
        nx = 1.

        max_ranks = defaultdict()

        if self.musco_filter_layers:
            lnames_compress_me = []
            for mname, m in self.model.named_modules():
                # break
                if mname in lnames_to_compress:
                    lname = mname
                    _, cin, _, _ = model_stats.input_shapes[lname][0]
                    _, cout, _, _ = model_stats.output_shapes[lname][0]
                    kernel_size = model_stats.ltypes[lname]['kernel_size']

                    tensor_shape = (cout, cin, *kernel_size)
                    r_pr = estimate_rank_for_compression_rate(tensor_shape[:2], rate = nx, key = 'svd')

                    r_vbmf = estimate_vbmf_ranks(m.weight.data[:, :, 0, 0], vbmf_weakenen_factor  = wf)

                    max_ranks[lname] = int(r_pr)

                    print('\n', lname, tensor_shape, r_pr, r_vbmf)
                    if r_pr > r_vbmf:
                        lnames_compress_me.append(lname)
                        print('===== COMPRESS ME ===== {} times\n'.format(r_pr/r_vbmf))
                    else:
                        print('===== DO NOT COMPRESS ME =====\n')
        else:
            lnames_compress_me = lnames_to_compress
        # if self.model_name == 'CSRNet':
        #     lnames_compress_me = lnames_compress_me[:-1]

        lnames_compress_me = lnames_to_compress[:13]
        noncompressing_lnames = {key: None for key in model_stats.flops.keys() if key not in lnames_compress_me}
        # noncompressing_lnames = {key: None for key in list(model_stats.flops.keys())[:-1]}

        write_print(self.output_txt, "LAYERS TO COMPRESS")
        write_print(self.output_txt, str(lnames_compress_me))

        compressor = CompressorVBMF(self.model,
                                    model_stats,
                                    ranks = noncompressing_lnames,
                                    ft_every=self.musco_ft_every,
                                    nglobal_compress_iters=self.musco_iters)
        # write_print('failed_layers.txt', '\n{} - {}'.format(self.model_name, self.dataset))

        self.compressor_step = 0
        while not compressor.done:
            self.compressor_step += 1

            # COMPRESSION
            write_print(self.output_txt, "\n Compress (STEP {})".format(self.compressor_step))
            compressor.compression_step()
            self.print_num_params(compressor.compressed_model)

            # FINE-TUNING
            write_print(self.output_txt, '\n Fine-tune')
            self.optimizer, self.criterion = self.build_optimizer_loss(compressor.compressed_model.to(device))

            self.musco_best_state_dict = None
            self.compression_save_path = self.output_txt[:self.output_txt.rfind('\\')] + "/compression step {}".format(self.compressor_step)
            
            self.musco_train(compressor.compressed_model.to(device), self.musco_ft_epochs, self.data_loaders['train'], eval_freq=self.musco_ft_checkpoint, checkpoint_file_name="compression step {}.pth.tar".format(self.compressor_step))
            
            write_print(self.output_txt, " ")

            # reload best performing epoch's weights into the compressed model
            if self.musco_best_state_dict is not None:
                compressor.compressed_model.load_state_dict(self.musco_best_state_dict)

                write_print(self.output_txt, "RELOADED BEST EPOCH")
                write_print(self.output_txt, "\t{}".format(self.musco_best_log))

            write_print(self.output_txt, "\n")

        # mae, mse, fps = self.eval(compressor.compressed_model.to(device), self.data_loaders['val'])
        # write_print(self.output_txt,  "MAE: {:.4f},  MSE: {:.4f},  FPS: {:.4f}".format(mae, mse, fps))

        stats_compressed = FlopCo(compressor.compressed_model.to(device), device = device)
        write_print(self.output_txt, str(1/(model_stats.total_flops / stats_compressed.total_flops)))
        self.print_network(compressor.compressed_model, "FINAL {}-COMPRESSED {} MODEL".format(self.compression.upper(), self.model_name.upper()))
        # self.save_model(compressor.compressed_model, -1)

    def model_step(self, model, images, labels, epoch):
        """
        A step for each iteration
        """

        # set model in training mode
        model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # print(images.dtype)
        # print(torch.cuda.memory_summary())
        # forward pass
        if self.model_name == 'MCNN':
            output = model(images, labels)
        else:
            images = images.float()
            output = model(images)

        if 'SKT' in self.model_name:
            output = output[-1]
            # print(output.size())

            # raise Exception


        if 'MARUNet' in self.model_name:
            output, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = output
            amp_gt = [get_amp_gt_by_value(l) for l in labels]
            amp_gt = torch.stack(amp_gt).cuda()

        # if self.save_output_plots:
        #     file_path = os.path.join(self.compile_txt[:self.compile_txt.rfind('COMPILED')], 'epoch ' + str(epoch +1))
        #     save_plots(file_path, output, labels)

        # compute loss
        if self.model_name == 'MCNN':
            model.loss.backward()
            loss = model.loss.item()
        elif 'MARUNet' in self.model_name:
            loss = 0
            outputs = [output, d0, d1, d2, d3, d4]

            target = 50 * labels[0].float().unsqueeze(1).cuda()
            for out in outputs:
                # out = out.cuda()
                # loss += self.criterion(out.squeeze(), labels.squeeze())
                loss += cal_avg_ms_ssim(out, target, 3)

            amp_outputs = [amp41, amp31, amp21, amp11, amp01]
            for amp in amp_outputs:
                # print(amp, loss)
                amp_gt_us = amp_gt[0].unsqueeze(0)
                amp = amp.cuda()
                if amp_gt_us.shape[2:]!=amp.shape[2:]:
                    amp_gt_us = F.interpolate(amp_gt_us, amp.shape[2:], mode='bilinear')
                cross_entropy = (amp_gt_us * torch.log(amp+1e-10) + (1 - amp_gt_us) * torch.log(1 - amp+1e-10)) * -1
                cross_entropy_loss = torch.mean(cross_entropy)
                loss = loss + cross_entropy_loss * 0.1

            loss.backward()
        else:
            loss = self.criterion(output.squeeze(), labels.squeeze())

            # compute gradients using back propagation
            loss.backward()

        # update parameters
        self.optimizer.step()

        # return loss
        return loss

    def musco_train(self, model, num_epochs, data_loader, eval_freq=None, checkpoint_file_name="checkpoint"):
        """
        Training process
        """
        self.losses = []
        iters_per_epoch = len(data_loader)
        sched = 0

        curr_best_mae = None
        curr_best_mse = None

        # start training
        write_print(self.output_txt, "TRAIN")
        start_time = time.time()
        for e in range(0, num_epochs):
            for i, (images, labels) in enumerate(tqdm(data_loader)):
                images = to_var(images, self.use_gpu)

                # labels = to_var(torch.tensor(labels), self.use_gpu)
                labels = [to_var(torch.Tensor(label), self.use_gpu) for label in labels]
                labels = torch.stack(labels)

                loss = self.model_step(model, images, labels, e)
                # print("\t{:.6f}".format(loss.item()))
                # break
            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time, iters_per_epoch, e, i, loss, num_epochs)
                self.losses.append((e, loss))

            # eval
            if (eval_freq is not None) and ((e + 1) % eval_freq == 0 or e + 1 == num_epochs):
                save_path = self.compression_save_path
                self.save_output_plots = (e + 1 == num_epochs)

                write_print(self.output_txt, "EVAL")
                mae, mse, fps = self.eval(model, self.data_loaders['val'], save_path=self.compression_save_path)
                log = "MAE: {:.4f},  MSE: {:.4f},  FPS: {:.4f}".format(mae, mse, fps)
                write_print(self.output_txt,  log)

                if curr_best_mae is None or (curr_best_mae + curr_best_mse > mae + mse):
                    curr_best_mae = mae
                    curr_best_mse = mse
                    self.musco_best_state_dict = copy.deepcopy(model.state_dict())
                    self.musco_best_optim = copy.deepcopy(self.optimizer.state_dict())
                    self.musco_best_loss = loss

                    self.musco_best_log = log

                    checkpoint = {
                        'architecture': model,
                        'state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                        'loss': loss,
                        'mae': mae,
                        'mse': mse
                    }

                    self.save_checkpoint(checkpoint,
                        # self.output_txt[:self.output_txt.rfind('\\')],
                        checkpoint_file_name)

                if (e+1) < num_epochs:
                    write_print(self.output_txt, "TRAIN")

    def eval(self, compressed_model, data_loader, save_path=None):
        """
        Returns the count of top 1 and top 5 predictions
        """

        # set the model to eval mode
        compressed_model.eval()

        y_true = to_var(torch.LongTensor([]), self.use_gpu)
        y_pred = to_var(torch.LongTensor([]), self.use_gpu)
        # out = []
        # sm = nn.Softmax()
        timer = Timer()
        elapsed = 0

        mae = 0
        mse = 0

        if self.dataset == 'mall':
            save_freq = 100
        else:
            save_freq = 30

        # if not self.save_output_plots:
        #     save_freq = int(len(data_loader) / 3)
        #     self.save_output_plots = True

        if save_path is None:
            save_path = self.model_test_path

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(data_loader)):
                images = to_var(images, self.use_gpu)

                labels = [to_var(torch.Tensor(label), self.use_gpu) for label in labels]
                labels = torch.stack(labels)

                images = images.float()
                timer.tic()
                output = compressed_model(images)
                elapsed += timer.toc(average=False)

                ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]

                if 'MARUNet' in self.model_name:
                    output = output[0] / 50

                if self.save_output_plots and i % save_freq == 0:
                    if save_path is None:
                        model = self.pretrained_model.split('/')
                        file_path = os.path.join(self.model_test_path, model[-2], self.dataset_info + ' epoch ' + model[-1])
                    else:
                        file_path = save_path
                    save_plots(file_path, output, labels, ids)

                y_true = torch.cat((y_true, labels))
                y_pred = torch.cat((y_pred, output))

                mae += abs(output.sum() - labels.sum()).item()
                mse += ((labels.sum() - output.sum())*(labels.sum() - output.sum())).item()
                # break


        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

        mae = mae / len(data_loader)
        mse = np.sqrt(mse / len(data_loader))
        fps = len(data_loader) / elapsed

        return mae, mse, fps

        

    '''
        SKT IMPLEMENTATION
    '''

    def skt(self):
        self.student_model = get_model('{}SKT'.format(self.model_name),
                               self.backbone_model,
                               self.imagenet_pretrain,
                               self.model_save_path,
                               self.input_channels,
                               self.class_count)

        # 
        cal_para(self.student_model)        # include 1x1 conv transform parameters
        self.model.regist_hook()            # use hook to get teacher's features
    
        if self.use_gpu:
            self.model.cuda()
            self.student_model.cuda()

        self.optimizer, _ = self.build_optimizer_loss()

        best_mae, best_mse = 1e3, 1e3
        for epoch in range(0, self.num_epochs):
            self.compression_save_path = self.output_txt[:self.output_txt.rfind('\\')] + "/compression step {}".format(epoch)
            
            print("TRAIN")
            self.skt_train(self.model, self.student_model, self.criterion)
            print("VAL")
            mae, mse = self.eval(self.student_model, self.data_loaders['val'])
            print("MAE: {.3f}, MSE: {.3f}  |  best_MAE: {.3f}, best_MSE: {.3f}".format(mae, mse, best_mae, best_mse))
            print()

            if self.skt_save_freq != 0 and (epoch+1) % self.skt_save_freq == 0:
                checkpoint = {
                    'state_dict': copy.deepcopy(self.student_model.state_dict()),
                    'mae': mae,
                    'mse': mse,
                    'best_mae': best_mae,   
                    'best_mse': best_mse
                }

                self.save_checkpoint(checkpoint,
                    "epoch_{}".format(epoch+1))
                continue

            if (mae < best_mae) or (mse < best_mse):
                best_mae = min(best_mae, mae)
                best_mse = min(best_mse, mse)

                checkpoint = {
                    'state_dict': copy.deepcopy(self.student_model.state_dict()),
                    'mae': mae,
                    'mse': mse,
                    'best_mae': best_mae,
                    'best_mse': best_mse
                }

                self.save_checkpoint(checkpoint,
                    "epoch_{}_mae_{}_mse_{}".format(
                        epoch+1,
                        "{:.3f}".format(mae).replace('.', '-'),
                        "{:.3f}".format(mse).replace('.', '-')
                    )
                )

    def skt_train(self, teacher, student, criterion):
        losses_h = AverageMeter()
        losses_s = AverageMeter()
        losses_fsp = AverageMeter()
        losses_cos = AverageMeter()
        losses_ssim = AverageMeter()

        data_loader = self.data_loaders['train']
        teacher.eval()
        student.train()
        for i, (images, labels) in enumerate(tqdm(data_loader)):
            images = to_var(images, self.use_gpu)

            labels = [to_var(torch.Tensor(label), self.use_gpu) for label in labels]
            labels = torch.stack(labels)

            # get teacher output
            with torch.no_grad():
                teacher_output = teacher(img)

                if self.model_name == 'MARUNet':
                    teacher.features.append(teacher_output[0])
                    teacher_fsp_features = [scale_process(teacher.features, scale = [4, 3, 2, 1, None, None, None, 2, 1, 4])]
                elif self.model_name == 'CSRNet':
                    teacher.features.append(teacher_output)
                    teacher_fsp_features = [scale_process(teacher.features, min_x=60)]
                teacher_fsp = cal_dense_fsp(teacher_fsp_features)

            # get student output
            if self.model_name == 'MARUNet':
                student_features, student_outputs = student(img)
            elif self.model_name == 'CSRNet':
                student_features = student(img)
            student_output = student_features[-1]

            # scale student output
            if self.model_name == 'MARUNet':
                student_fsp_features = [scale_process(student_features,  scale = [4, 3, 2, 1, None, None, None, 2, 1, 4])]
            elif self.model_name == 'CSRNet':
                student_fsp_features = [scale_process(student_features, min_x=None)]
            student_fsp = cal_dense_fsp(student_fsp_features)

            # get loss
            if self.model_name == 'MARUNet':
                loss_h = self.marunet_loss(student_outputs, target)# * args.lamb_h# * 1e6
                loss_s = self.marunet_loss(student_outputs, teacher_output)# * args.lamb_h# * 1e6
                divide_val = 50.
            elif self.model_name == 'CSRNet':
                loss_h = criterion(student_output, target)
                loss_s = criterion(student_output, teacher_output)
                divide_val = 1.

            loss_fsp = torch.tensor([0.], dtype=torch.float).cuda()
            if args.lamb_fsp:
                loss_f = []
                assert len(teacher_fsp) == len(student_fsp)
                for t in range(len(teacher_fsp)):
                    loss_f.append(criterion(teacher_fsp[t] / divide_val, student_fsp[t] / divide_val))
                loss_fsp = sum(loss_f) * self.skt_lamb_fsp

            loss_cos = torch.tensor([0.], dtype=torch.float).cuda()
            if args.lamb_cos:
                loss_c = []
                for t in range(len(student_features) - 1):
                    loss_c.append(cosine_similarity(student_features[t] / divide_val, teacher.features[t] / divide_val))
                loss_cos = sum(loss_c) * self.skt_lamb_cos

            # calculate total loss
            loss = loss_h + loss_s + loss_fsp + loss_cos # + loss_ssim

            losses_h.update(loss_h.item(), img.size(0))
            losses_s.update(loss_s.item(), img.size(0))
            losses_fsp.update(loss_fsp.item(), img.size(0))
            losses_cos.update(loss_cos.item(), img.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == self.skt_print_freq:
                print(
                      'Loss_h {loss_h.avg:.4f}  '
                      'Loss_s {loss_s.avg:.4f}  '
                      'Loss_fsp {loss_fsp.avg:.4f}  '
                      'Loss_cos {loss_kl.avg:.4f}  '
                    .format(
                    loss_h=losses_h, loss_s=losses_s,
                    loss_fsp=losses_fsp, loss_kl=losses_cos))

    def marunet_loss(self, student_output, target):
        loss = 0

        # type(target) == tuple means that the target is teacher_output
        # otherwise, target is the groundtruth
        
        if type(target) != tuple:
            amp_gt = [get_amp_gt_by_value(l) for l in target]
            amp_gt = torch.stack(amp_gt).cuda()
            amp_gt = amp_gt[0]
            target = 50 * target[0].float().unsqueeze(1).cuda()
        
        for i, out in enumerate(student_output[:6]):
            if type(target) != tuple:
                tar = target
            else:
                tar = student_output[i].cuda()

            loss += cal_avg_ms_ssim(out, tar, 3)

        for i, amp in enumerate(student_output[6:]):
            if type(target) != tuple:
                amp_gt_us = amp_gt.unsqueeze(0)
            else:
                amp_gt_us = student_output[i+6].cuda()

            amp = amp.cuda()
            if amp_gt_us.shape[2:]!=amp.shape[2:]:
                amp_gt_us = F2.interpolate(amp_gt_us, amp.shape[2:], mode='bilinear')
            cross_entropy = (amp_gt_us * torch.log(amp+1e-10) + (1 - amp_gt_us) * torch.log(1 - amp+1e-10)) * -1
            cross_entropy_loss = torch.mean(cross_entropy)
            loss = loss + cross_entropy_loss * 0.1

        return loss