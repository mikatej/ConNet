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
import torch.nn.functional as F2


class Compressor(object):

    DEFAULTS = {}

    def __init__(self, version, data_loaders, dataset_ids, config, output_txt, compile_txt):
        """
        Initializes a Compressor object
        """

        self.__dict__.update(Compressor.DEFAULTS, **config)
        self.version = version
        self.data_loaders = data_loaders
        self.dataset_ids = dataset_ids
        self.output_txt = output_txt
        self.compile_txt = compile_txt

        self.dataset_info = self.dataset
        if (self.dataset == 'micc'):
            self.dataset_info = '{} {}'.format(self.dataset, self.dataset_subcategory)

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
        '''
        Begins compression depending on the technique specified
        '''

        if self.compression == 'skt':
            self.skt()

        elif self.compression == 'musco':
            self.musco()

        else:
            raise Exception("Compression technique ({}) not implemented".format(self.compression))

    def build_model(self):
        '''
        Initializes the model to be compressed. Also initializes the student model if the
        chosen compression technique is SKT
        '''

        self.model_name = self.model
        self.model = get_model(self.model,
                               self.imagenet_pretrain,
                               self.model_save_path,
                               self.input_channels)

        if self.compression == 'skt':            
            self.student_model = get_model('{}SKT'.format(self.model_name),
                                   self.imagenet_pretrain,
                                   self.model_save_path,
                                   self.input_channels)

    def build_optimizer_loss(self, model):
        '''
        Initializes the optimizer and loss criterion to be used
        '''

        # Instantiate the loss criterion
        criterion = nn.MSELoss()

        # Instantiate the optimizer
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

        write_print(self.output_txt, name)
        write_print(self.output_txt, str(model))
        self.print_num_params(model)

    def print_num_params(self, model):
        '''
        Prints the total number of parameters
        '''

        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth or .pth.tar file
        """

        # if pretrained model is a .pth file, load weights directly
        if ".pth.tar" not in self.pretrained_model:
            self.pretrained_model = self.pretrained_model.replace('.pth', '')
            self.model.load_state_dict(torch.load(os.path.join(
                '{}.pth'.format(self.pretrained_model))))

        # if pretrained model is a .pth.tar file, load weights stored in 'state_dict' key
        else:
            weights = torch.load('{}'.format(self.pretrained_model))
            self.model.load_state_dict(weights['state_dict'])

        write_print(self.output_txt,
                    'loaded trained model {}'.format(self.pretrained_model))

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        '''
        Saves the model weights as either a .pth file or a .pth.tar file
        '''

        torch.save(state, os.path.join(
            self.output_txt[:self.output_txt.rfind('\\')],
            filename))
        
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

    def eval(self, compressed_model, data_loader, save_path=None):
        """
        Performs evaluation of a given model to get the MAE, MSE, FPS performance
        """

        # set the model to eval mode
        compressed_model.eval()

        timer = Timer()
        elapsed = 0
        mae = 0
        mse = 0

        # predetermined save frequency of density maps on certain datasets
        if self.dataset == 'mall':
            save_freq = 100
        else:
            save_freq = 30

        if save_path is None:
            save_path = self.model_test_path

        # begin evaluating on the dataset
        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(data_loader)):
                # prepare the input images
                images = to_var(images, self.use_gpu)
                images = images.float()

                # prepare the groundtruth targets
                targets = [to_var(torch.Tensor(label), self.use_gpu) for target in targets]
                targets = torch.stack(targets)

                # generate output of model
                timer.tic()
                output = compressed_model(images)
                elapsed += timer.toc(average=False)

                # if model is MARUNet, divide output by 50 as designed by original proponents
                if 'MARUNet' in self.model_name:
                    output = output[0] / 50

                # generate copies of density maps as images
                if self.save_output_plots and i % save_freq == 0:
                    ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]
                    if save_path is None:
                        model = self.pretrained_model.split('/')
                        file_path = os.path.join(self.model_test_path, model[-2], self.dataset_info + ' epoch ' + model[-1])
                    else:
                        file_path = save_path
                    save_plots(file_path, output, targets, ids)

                # update MAE and MSE (summation part of the formula)
                mae += abs(output.sum() - targets.sum()).item()
                mse += ((targets.sum() - output.sum())*(targets.sum() - output.sum())).item()

        # compute for MAE, MSE and FPS
        mae = mae / len(data_loader)
        mse = np.sqrt(mse / len(data_loader))
        fps = len(data_loader) / elapsed

        return mae, mse, fps

    '''
        MUSCO IMPLEMENTATION
    '''

    def musco(self):
        """
        General implementation of the iterative compress-finetune MUSCO algorithm
        """

        if self.use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'
        self.model.to(device)

        # get the model_stats to pass to Compressor object
        img, _ = next(iter(self.data_loaders['train']))
        img_size = tuple(img.shape)
        model_stats = FlopCo(self.model, img_size = img_size, device = device)

        # get the ranks to pass to Compressor object
        noncompressing_lnames = self.musco_get_ranks()
        raise Exception()

        # create Compressor object
        compressor = CompressorVBMF(self.model,
                                    model_stats,
                                    ranks = noncompressing_lnames,
                                    ft_every=self.musco_ft_every,
                                    nglobal_compress_iters=self.musco_iters)

        self.compressor_step = 0
        save_path = self.output_txt[:self.output_txt.rfind('\\')]
        self.compression_save_path = os.path.join(save_path, "compression step %d")

        # begin MUSCO algorithm
        while not compressor.done:
            self.compressor_step += 1
            self.musco_best_state_dict = None

            # COMPRESSION
            write_print(self.output_txt, "\n Compress (STEP {})".format(self.compressor_step))
            compressor.compression_step()
            self.print_num_params(compressor.compressed_model)

            # FINE-TUNING
            write_print(self.output_txt, '\n Fine-tune')
            self.optimizer, self.criterion = self.build_optimizer_loss(compressor.compressed_model.to(device))
            
            # train and eval
            self.musco_train(compressor.compressed_model.to(device), self.musco_ft_epochs, eval_freq=self.musco_ft_checkpoint, checkpoint_file_name="compression step {}.pth.tar".format(self.compressor_step))
            write_print(self.output_txt, " ")

            # reload best performing epoch's weights into the compressed model
            if self.musco_best_state_dict is not None:
                compressor.compressed_model.load_state_dict(self.musco_best_state_dict)

                write_print(self.output_txt, "RELOADED BEST EPOCH")
                write_print(self.output_txt, "\t{}".format(self.musco_best_log))

            write_print(self.output_txt, "\n")

        self.print_network(compressor.compressed_model, "FINAL {}-COMPRESSED {} MODEL".format(self.compression.upper(), self.model_name.upper()))

    def musco_get_ranks(self):
        """
        Determines the layers to be compressed by the MUSCO algorithm
        """

        # if layers are not specified, all layers eligible for compression will be compressed
        if self.musco_layers_to_compress.strip() == '':
            lnames_to_compress = [k for k in model_stats.flops.keys() if\
                          model_stats.ltypes[k]['type'] == nn.Conv2d and model_stats.ltypes[k]['kernel_size'] != (1, 1)
                          and 'out' not in k and 'back0.0' not in k and 'amp' not in k and 'brg' not in k
                          and not ('conv4' in k and 'SKT' in self.model_name)
                          ]

            if self.model_name == 'CSRNet':
                lnames_to_compress = [l for l in lnames_to_compress if\
                         'backend' not in l]

        # if layers are specified, turn string to list
        else:
            lnames_to_compress = self.musco_layers_to_compress.split(',')

        write_print(self.output_txt, "LAYERS TO COMPRESS")
        write_print(self.output_txt, str(lnames_to_compress))

        # format results as expected by the Compressor object
        noncompressing_lnames = {key: None for key in model_stats.flops.keys() if key not in lnames_compress_me}
        
        return noncompressing_lnames

    def model_step(self, model, images, targets, epoch):
        """
        A step for each iteration
        """

        # set model in training mode
        model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # forward pass
        if self.model_name == 'MCNN':
            output = model(images, targets)
        else:
            images = images.float()
            output = model(images)

        if 'SKT' in self.model_name:
            output = output[-1]

        # if model is MARUNet, prepare groundtruth attention maps
        if 'MARUNet' in self.model_name:
            output, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = output
            amp_gt = [get_amp_gt_by_value(l) for l in targets]
            amp_gt = torch.stack(amp_gt).cuda()

        # compute loss
        if self.model_name == 'MCNN':
            model.loss.backward()
            loss = model.loss.item()

        # if model is MARUNet
        elif 'MARUNet' in self.model_name:
            loss = 0
            target = 50 * targets[0].float().unsqueeze(1).cuda()

            # get losses of density maps
            outputs = [output, d0, d1, d2, d3, d4]
            for out in outputs:
                loss += cal_avg_ms_ssim(out, target, 3)

            # get losses of attention maps
            amp_outputs = [amp41, amp31, amp21, amp11, amp01]
            for amp in amp_outputs:
                amp_gt_us = amp_gt[0].unsqueeze(0)
                amp = amp.cuda()
                if amp_gt_us.shape[2:]!=amp.shape[2:]:
                    amp_gt_us = F.interpolate(amp_gt_us, amp.shape[2:], mode='bilinear')
                
                cross_entropy = (amp_gt_us * torch.log(amp+1e-10) + (1 - amp_gt_us) * torch.log(1 - amp+1e-10)) * -1
                cross_entropy_loss = torch.mean(cross_entropy)
                loss = loss + cross_entropy_loss * 0.1

            # compute gradients using back propagation
            loss.backward()
        else:
            # compute loss using MSE loss function
            loss = self.criterion(output.squeeze(), targets.squeeze())

            # compute gradients using back propagation
            loss.backward()

        # update parameters
        self.optimizer.step()

        # return loss
        return loss

    def musco_train(self, model, num_epochs, eval_freq=None, checkpoint_file_name="checkpoint"):
        """
        Performs training process tailored specifically for the MUSCO algorithm
        """

        self.losses = []
        train_loader, val_loader = self.data_loaders

        curr_best_mae = None
        curr_best_mse = None

        # start training
        write_print(self.output_txt, "TRAIN")
        start_time = time.time()
        for e in range(0, num_epochs):
            for i, (images, targets) in enumerate(tqdm(train_loader)):
                # prepare input images
                images = to_var(images, self.use_gpu)

                # prepare groundtruth targets
                targets = [to_var(torch.Tensor(label), self.use_gpu) for label in targets]
                targets = torch.stack(targets)

                # train model and get loss
                loss = self.model_step(model, images, targets, e)

            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time, len(train_loader), e, i, loss, num_epochs)
                self.losses.append((e, loss))

            # eval
            if (eval_freq is not None) and ((e + 1) % eval_freq == 0 or e + 1 == num_epochs):
                save_path = self.compression_save_path % self.compressor_step
                self.save_output_plots = (e + 1 == num_epochs)

                write_print(self.output_txt, "EVAL")
                mae, mse, fps = self.eval(model, val_loader, save_path=save_path)
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
                        checkpoint_file_name)

                if (e+1) < num_epochs:
                    write_print(self.output_txt, "TRAIN")


    '''
        SKT IMPLEMENTATION
    '''

    def skt(self):
        """
        General implementation of the SKT algorithm
        """

        # prepare the models
        cal_para(self.student_model)        # include 1x1 conv transform parameters
        self.model.regist_hook()            # use hook to get teacher's features
    
        if self.use_gpu:
            self.model.cuda()
            self.student_model.cuda()

        # instantiate optimizer and loss criterion
        self.optimizer, self.criterion = self.build_optimizer_loss(self.student_model)
        path = self.output_txt[:self.output_txt.rfind('\\')]

        # load student weights if checkpoint exists
        start = 0
        if self.skt_student_ckpt:
            start = self.skt_load_student_ckpt()

        # begin algorithm
        best_mae, best_mse = 1e3, 1e3
        for epoch in range(start, self.num_epochs):
            self.compression_save_path = os.path.join(path, 'compression step {}'.format(epoch))
            
            # train
            print("TRAIN")
            self.skt_train(self.model, self.student_model, self.criterion, epoch, path)
            
            # val
            print("VAL")
            mae, mse, _ = self.eval(self.student_model, self.data_loaders['val'], save_path=os.path.join(path, "epoch {}".format(epoch+1)))
  
            # print and log mae/mse results
            write_print(
                os.path.join(path, 'mae mse.txt'),
                "[EPOCH {}]  MAE: {:.3f}, MSE: {:.3f}  |  best_MAE: {:.3f}, best_MSE: {:.3f}"
                    .format(epoch+1, mae, mse, min(mae, best_mae), min(mse, best_mse)))
            if (epoch+1) % 5 == 0:
                write_print(os.path.join(path, 'mae mse.txt'), " ")
            print()

            # if save frequency is specified and current epoch meets save frequency
            if self.skt_save_freq != 0 and (epoch+1) % self.skt_save_freq == 0:
                checkpoint = {
                    'state_dict': copy.deepcopy(self.student_model.state_dict()),
                    'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                    'mae': mae,
                    'mse': mse,
                    'best_mae': best_mae,   
                    'best_mse': best_mse
                }

                self.save_checkpoint(checkpoint,
                    "epoch_{}".format(epoch+1))
                continue

            # if save frequency is not specified and mae/mse is current best
            if self.skt_save_freq == 0 and ((mae < best_mae) or (mse < best_mse)):
                best_mae = min(best_mae, mae)
                best_mse = min(best_mse, mse)

                checkpoint = {
                    'state_dict': copy.deepcopy(self.student_model.state_dict()),
                    'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                    'mae': mae,
                    'mse': mse,
                    'best_mae': best_mae,
                    'best_mse': best_mse
                }

                self.save_checkpoint(checkpoint,
                    "epoch_{}_mae_{}_mse_{}.pth.tar".format(
                        epoch+1,
                        "{:.3f}".format(mae).replace('.', '-'),
                        "{:.3f}".format(mse).replace('.', '-')
                    )
                )

    def skt_load_student_ckpt(self):
        '''
        Loads the SKT student model's weights and the optimizer's weights
        '''

        # get last epoch number
        epoch = self.skt_student_ckpt.split('/')[-1].split('_')[1]
        epoch = int(epoch)

        # load weights
        weights = torch.load('{}'.format(self.skt_student_ckpt))
        self.student_model.load_state_dict(weights['state_dict'])
        self.optimizer.load_state_dict(weights['optimizer'])

        write_print(self.output_txt,
                    'loaded trained student model {}'.format(self.skt_student_ckpt))

        return epoch

    def skt_train(self, teacher, student, criterion, epoch=0, save_path=None):
        '''
        Performs training process tailored specifically for the SKT algorithm
        '''

        # instantiate objects for losses
        losses_h = AverageMeter()
        losses_s = AverageMeter()
        losses_fsp = AverageMeter()
        losses_cos = AverageMeter()
        losses_ssim = AverageMeter()

        # prepare models and data loader
        teacher.eval()
        student.train()
        data_loader = self.data_loaders['train']

        # begin train
        for i, (images, targets) in enumerate(data_loader):
            # prepare input images
            images = to_var(images, self.use_gpu)
            images = images.float()

            # prepare groundtruth targets
            targets = [to_var(torch.Tensor(target), self.use_gpu) for target in targets]
            targets = torch.stack(targets)

            # get teacher output
            with torch.no_grad():
                teacher_output = teacher(images)

                if self.model_name == 'MARUNet':
                    teacher.features.append(teacher_output[0])
                    teacher_fsp_features = [scale_process(teacher.features, scale = [4, 3, 2, 1, None, None, None, 2, 1, 4])]
                elif self.model_name == 'CSRNet':
                    teacher.features.append(teacher_output)
                    teacher_fsp_features = [scale_process(teacher.features, min_x=60)]
                teacher_fsp = cal_dense_fsp(teacher_fsp_features)

            # get student output
            if self.model_name == 'MARUNet':
                student_features, student_outputs = student(images)
            elif self.model_name == 'CSRNet':
                student_features = student(images)
            student_output = student_features[-1]

            # scale student output
            if self.model_name == 'MARUNet':
                student_fsp_features = [scale_process(student_features,  scale = [4, 3, 2, 1, None, None, None, 2, 1, 4])]
            elif self.model_name == 'CSRNet':
                student_fsp_features = [scale_process(student_features, min_x=None)]
            student_fsp = cal_dense_fsp(student_fsp_features)

            # get loss
            if self.model_name == 'MARUNet':
                loss_h = self.skt_marunet_loss(student_outputs, targets)
                loss_s = self.skt_marunet_loss(student_outputs, teacher_output)
                divide_val = 50.
            elif self.model_name == 'CSRNet':
                loss_h = criterion(student_output, targets)
                loss_s = criterion(student_output, teacher_output)
                divide_val = 1.

            loss_fsp = torch.tensor([0.], dtype=torch.float).cuda()
            if self.skt_lamb_fsp:
                loss_f = []
                assert len(teacher_fsp) == len(student_fsp)
                for t in range(len(teacher_fsp)):
                    loss_f.append(criterion(teacher_fsp[t] / divide_val, student_fsp[t] / divide_val))
                loss_fsp = sum(loss_f) * self.skt_lamb_fsp

            loss_cos = torch.tensor([0.], dtype=torch.float).cuda()
            if self.skt_lamb_cos:
                loss_c = []
                for t in range(len(student_features) - 1):
                    loss_c.append(cosine_similarity(student_features[t] / divide_val, teacher.features[t] / divide_val))
                loss_cos = sum(loss_c) * self.skt_lamb_cos

            # calculate total loss
            loss = loss_h + loss_s + loss_fsp + loss_cos

            losses_h.update(loss_h.item(), images.size(0))
            losses_s.update(loss_s.item(), images.size(0))
            losses_fsp.update(loss_fsp.item(), images.size(0))
            losses_cos.update(loss_cos.item(), images.size(0))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i+1) % self.skt_print_freq == 0:
                print(
                      'EPOCH {} [{}/{}] '
                      'Loss_h {loss_h.avg:.4f}  '
                      'Loss_s {loss_s.avg:.4f}  '
                      'Loss_fsp {loss_fsp.avg:.4f}  '
                      'Loss_cos {loss_kl.avg:.4f}  '
                    .format(
                    epoch, i+1, len(data_loader),
                    loss_h=losses_h, loss_s=losses_s,
                    loss_fsp=losses_fsp, loss_kl=losses_cos))
        print()
        write_to_file(os.path.join(save_path, 'losses.txt'),
                  '[Epoch {0}]  '
                  'Loss_h {loss_h.avg:.4f}  '
                  'Loss_s {loss_s.avg:.4f}  '
                  'Loss_fsp {loss_fsp.avg:.4f}  '
                  'Loss_cos {loss_kl.avg:.4f}'
                .format(
                epoch + 1,
                loss_h=losses_h, loss_s=losses_s,
                loss_fsp=losses_fsp, loss_kl=losses_cos))

    def skt_marunet_loss(self, student_output, target):
        '''
        Computes the loss of a MARUNet model, tailored specifically for the SKT algorithm.
        This accepts both groundtruth and teacher model's output as target.
        '''

        loss = 0

        # if target is a tuple, then the target passed is teacher_output
        # otherwise, target is the groundtruth density map        
        if type(target) != tuple:
            amp_gt = [get_amp_gt_by_value(l) for l in target]
            amp_gt = torch.stack(amp_gt).cuda()
            amp_gt = amp_gt[0]
            target = 50 * target[0].float().unsqueeze(1).cuda()
        
        # compute loss of density maps
        for i, out in enumerate(student_output[:6]):
            if type(target) != tuple:
                tar = target
            else:
                tar = student_output[i].cuda()

            loss += cal_avg_ms_ssim(out, tar, 3)

        # compute loss of attention maps
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
