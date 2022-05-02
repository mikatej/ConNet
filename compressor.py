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
            if self.musco_ft_only == True:
                self.musco_finetune()
            else:
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
        if 'MARUNet' in self.model_name:
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

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """

        if "MUSCO test" in self.pretrained_model:
            checkpoint = torch.load(os.path.join(
                '{}.pth.tar'.format(self.pretrained_model)))
            self.model.load_state_dict(checkpoint['musco_best_state_dict'])

            self.optimizer, self.criterion = self.build_optimizer_loss(self.model)
            self.optimizer.load_state_dict(checkpoint['musco_best_optim'])

            return

        self.model.load_state_dict(torch.load(os.path.join(
            '{}.pth'.format(self.pretrained_model))))
        write_print(self.output_txt,
                    'loaded trained model {}'.format(self.pretrained_model))

    def save_model(self, model, e):
        """
        Saves a model per e epoch
        """
        path = os.path.join(
            self.output_txt[:self.output_txt.rfind('\\')],
            '{}.pth'.format(self.version, e + 1)
        )

        torch.save(model.state_dict(), path)

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
            write_print(self.output_txt, "\n Compress (STEP {})".format(self.compressor_step))
            compressor.compression_step()
            self.print_num_params(compressor.compressed_model)
            # self.print_network(self.model, "original")
            # raise Exception
            # self.print_network(compressor.compressed_model, "{} MODEL MUSCO COMPRESSION STEP {}".format(self.model_name.upper(), self.compressor_step))

            write_print(self.output_txt, '\n Fine-tune')
            # self.optimizer, self.criterion = self.build_optimizer_loss(compressor.compressed_model.to(device))
            self.optimizer, self.criterion = self.build_optimizer_loss(compressor.compressed_model.to(device))

            self.musco_best_state_dict = None


            # try:
            # write_print(self.output_txt, "MUSCO COMPRESSION STEP " + str(self.compressor_step))
            self.compression_save_path = self.output_txt[:self.output_txt.rfind('\\')] + "/compression step {}".format(self.compressor_step)
            self.musco_train(compressor.compressed_model.to(device), self.musco_ft_epochs, self.data_loaders['train'], eval_freq=self.musco_ft_checkpoint, checkpoint_file_name="compression step {}.pth.tar".format(self.compressor_step))
            # mae, mse, fps = self.eval(compressor.compressed_model.to(device), self.data_loaders['val'])
            # continue

            write_print(self.output_txt, " ")

            if self.musco_best_state_dict is not None:
                compressor.compressed_model.load_state_dict(self.musco_best_state_dict)

                write_print(self.output_txt, "RELOADED BEST EPOCH")
                write_print(self.output_txt, "\t{}".format(self.musco_best_log))

            write_print(self.output_txt, "\n")
            # except Exception as e:
            #     write_print('failed_layers.txt', "FAILED {}".format(lnames_compress_me[self.compressor_step-1]))
            #     write_print('failed_layers.txt', '\t {}'.format(e))

        mae, mse, fps = self.eval(compressor.compressed_model.to(device), self.data_loaders['val'])
        write_print(self.output_txt,  "MAE: {:.4f},  MSE: {:.4f},  FPS: {:.4f}".format(mae, mse, fps))

        stats_compressed = FlopCo(compressor.compressed_model.to(device), device = device)
        write_print(self.output_txt, str(1/(model_stats.total_flops / stats_compressed.total_flops)))
        # print(best_mae)

        self.print_network(compressor.compressed_model, "FINAL {}-COMPRESSED {} MODEL".format(self.compression.upper(), self.model_name.upper()))
        self.save_model(compressor.compressed_model, -1)

    def musco_finetune(self):
        if self.use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        self.model.to(device)
        # self.optimizer, self.criterion = self.build_optimizer_loss(self.model)
        self.compression_save_path = self.output_txt[:self.output_txt.rfind('\\')] + "/training outputs"
        self.musco_train(self.model, self.musco_ft_epochs, self.data_loaders['train'], eval_freq=self.musco_ft_checkpoint)
        # for epoch in range(num_epochs):

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

    def save_checkpoint(self, state, path, filename='checkpoint.pth.tar'):
        torch.save(state, os.path.join(path, filename))
        # epoch = state['epoch']
        # if mae_is_best:
        #     shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch'+str(epoch)+'_best_mae.pth.tar'))
        # if mse_is_best:
        #     shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch'+str(epoch)+'_best_mse.pth.tar'))


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

    def musco_train(self, model, num_epochs, data_loader, eval_freq=None, checkpoint_file_name=None):
        """
        Training process
        """
        self.losses = []
        iters_per_epoch = len(data_loader)
        sched = 0

        curr_best_mae = None
        curr_best_mse = None

        # checkpoint_file_name = "compression step {}.pth.tar".format(self.compressor_step)

        # start with a trained model if exists
        # if self.pretrained_model:
        #     start = int(self.pretrained_model.split('/')[-1])

        #     for x in self.learning_sched:
        #         if start >= x:
        #             sched +=1
        #             self.lr /= 10
        #         else:
        #             break

        #     print("LEARNING RATE: ", self.lr, sched)
        # else:
        #     start = 0

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

                    if self.musco_ft_only is True:
                        try:
                            i += 1
                        except:
                            i = 0
                        checkpoint_file_name = "ft_epoch_{}_mae_{}_mse_{}.pth".format(e,
                            str(mae).replace(".","_"),
                            str(mse).replace(".","_"))
                        checkpoint = copy.deepcopy(model.state_dict())

                    else:
                        checkpoint = {
                            'architecture': model,
                            'state_dict': copy.deepcopy(model.state_dict()),
                            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                            'loss': loss,
                            'mae': mae,
                            'mse': mse
                        }

                    self.save_checkpoint(checkpoint,
                        self.output_txt[:self.output_txt.rfind('\\')],
                        checkpoint_file_name)

                if (e+1) < num_epochs:
                    write_print(self.output_txt, "TRAIN")

            # save model
            # if (e + 1) % self.model_save_step == 0:
            #     self.save_model(e)

            # num_sched = len(self.learning_sched)
            # if num_sched != 0 and sched < num_sched:
            #     # if (e + 1) == self.learning_sched[sched]:
            #     if (e + 1) in self.learning_sched:
            #         self.lr /= 10
            #         print('Learning rate reduced to', self.lr)
            #         sched += 1

        # print losses
        # write_print(self.output_txt, '\n--Losses--')
        # for e, loss in self.losses:
        #     write_print(self.output_txt, str(e) + ' {:.10f}'.format(loss))

        # # print top_1_acc
        # write_print(self.output_txt, '\n--Top 1 accuracy--')
        # for e, acc in self.top_1_acc:
        #     write_print(self.output_txt, str(e) + ' {:.4f}'.format(acc))

        # # print top_5_acc
        # write_print(self.output_txt, '\n--Top 5 accuracy--')
        # for e, acc in self.top_5_acc:
        #     write_print(self.output_txt, str(e) + ' {:.4f}'.format(acc))

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
            save_freq = 30
        else:
            save_freq = 10

        if not self.save_output_plots:
            save_freq = int(len(data_loader) / 3)
            self.save_output_plots = True

        if save_path is None:
            save_path = self.model_test_path

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(data_loader)):
                images = to_var(images, self.use_gpu)

                labels = [to_var(torch.Tensor(label), self.use_gpu) for label in labels]
                labels = torch.stack(labels)

                timer.tic()
                images = images.float()
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

                # _, top_1_output = torch.max(output.data, dim=1)
                # out.append(str(sm(output.data).tolist()))
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

    def calibrate(model, device, train_loader, max_iters = 1000,
                  freeze_lnames = None):

        model.to(device).train()
        for pname, p in model.named_parameters():

            if pname.strip('.weight').strip('.bias')  in freeze_lnames:
                p.requires_grad = False

        with torch.no_grad():
            for i, (data, _) in enumerate(loaders['train']):
                _ = model(data.to(device))

                if i%50 == 0:
                    print('hey')

                if i > max_iters:
                    break

                del data
                torch.cuda.empty_cache()

        model.eval()
        return model
