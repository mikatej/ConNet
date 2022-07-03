import os
import torch
import time
import copy
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
import torch.nn.functional as F
from utils.marunet_losses import cal_avg_ms_ssim

class Solver(object):

    DEFAULTS = {}

    def __init__(self, version, data_loader, dataset_ids, config, output_txt, compile_txt):
        """
        Initializes a Solver object

        Arguments:
            version {str} -- version of the model based on the time
            data_loader {DataLoader} -- DataLoader of the dataset to be used
            dataset_ids {list} -- list of image IDs, used for naming the exported density maps
            config {dict} -- contains arguments and its values
            output_txt {str} -- file name for the text file where details are logged
            compile_txt {str} -- file name for the text file where performance is compiled (if val/test mode)
        """

        # data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.data_loader = data_loader
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


    def build_model(self):
        """
        Instantiates the model, loss criterion, and optimizer
        """

        # instantiate model
        self.model_name = self.model
        self.model = get_model(self.model,
                               self.imagenet_pretrain,
                               self.model_save_path,
                               self.input_channels)

        # instantiate loss criterion
        self.criterion = nn.MSELoss() 

        # instantiate optimizer
        if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr)

        # print network
        self.print_network(self.model, self.model_name)

        # use gpu if enabled
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

    def print_network(self, model, name):
        """
        Prints the structure of the network and the total number of parameters

        Arguments:
            model {Object} -- the model to be used
            name {str} -- name of the model
        """

        num_params = 0
        for name, param in model.named_parameters():
            if 'transform' in name:
                continue
            num_params += param.data.numel()
        write_print(self.output_txt, name)
        write_print(self.output_txt, str(model))
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
                self.model_save_path, '{}.pth'.format(self.pretrained_model))))
        
        # if pretrained model is a .pth.tar file, load weights stored in 'state_dict' and 'optimizer' keys
        else:
            weights = torch.load(os.path.join(
                self.model_save_path, '{}'.format(self.pretrained_model)))
            self.model.load_state_dict(weights['state_dict'])
            # self.optimizer.load_state_dict(weights['optimizer'])

        write_print(self.output_txt,
                    'loaded trained model {}'.format(self.pretrained_model))

    def print_loss_log(self, start_time, iters_per_epoch, e, i, loss):
        """
        Prints the loss and elapsed time for each epoch
        
        Arguments:
            start_time {float} -- time (milliseconds) at which training of an epoch began
            iters_per_epoch {int} -- number of iterations in an epoch
            e {int} -- current epoch
            i {int} -- current iteraion
            loss {float} -- loss value
        """

        total_iter = self.num_epochs * iters_per_epoch
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
                                    self.num_epochs,
                                    i + 1,
                                    iters_per_epoch,
                                    loss)

        write_print(self.output_txt, log)

    def save_model(self, e):
        """
        Saves the model and optimizer weights per e epoch

        Arguments:
            e {int} -- current epoch
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, e+1)
        )

        # torch.save(self.model.state_dict, path)
        torch.save({'state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())
            }, path)

    def model_step(self, images, targets, epoch):
        """
        A step for each iteration
        
        Arguments:
            images {torch.Tensor} -- input images
            targets {torch.Tensor} -- groundtruth density maps
            epoch {int} -- current epoch
        """

        # set model in training mode
        self.model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # forward pass
        if self.model_name == 'MCNN':
            output = self.model(images, targets)
        else:
            images = images.float()
            output = self.model(images)

        # if model is MARUNet or ConNet, prepare groundtruth attention maps
        if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
            if 'ConNet' in self.model_name:
                _, output = output
            output, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = output
            amp_gt = [get_amp_gt_by_value(l) for l in targets]
            amp_gt = torch.stack(amp_gt).cuda()
            
        # compute loss
        if self.model_name == 'MCNN':
            self.model.loss.backward()
            loss = self.model.loss.item()

        # if model is MARUNet or ConNet
        elif 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
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

    def train(self):
        """
        Performs training process
        """
        self.losses = []
        iters_per_epoch = len(self.data_loader)
        sched = 0

        # start with a trained model if exists
        if self.pretrained_model:
            try:
                start = int(self.pretrained_model.split('/')[-1].replace('.pth.tar', ''))
            except:
                start = 0

            for x in self.learning_sched:
                if start >= x:
                    sched +=1
                    self.lr /= 10
                else:
                    break

            print("LEARNING RATE: ", self.lr, sched, " | EPOCH:", start)
        else:
            start = 0

        # start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, targets) in enumerate(tqdm(self.data_loader)):
                # prepare input images
                images = to_var(images, self.use_gpu)

                # prepare groundtruth targets
                targets = [to_var(torch.Tensor(target), self.use_gpu) for target in targets]
                targets = torch.stack(targets)

                # train model and get loss
                loss = self.model_step(images, targets, e)

            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time, iters_per_epoch, e, i, loss)
                self.losses.append((e, loss))

            # save model
            if (e + 1) % self.model_save_step == 0:
                self.save_model(e)

            # update learning rate based on learning schedule
            num_sched = len(self.learning_sched)
            if num_sched != 0 and sched < num_sched:
                if (e + 1) in self.learning_sched:
                    self.lr /= 10
                    print('Learning rate reduced to', self.lr)
                    sched += 1

        # print losses
        write_print(self.output_txt, '\n--Losses--')
        for e, loss in self.losses:
            write_print(self.output_txt, str(e) + ' {:.10f}'.format(loss))

    def eval(self, data_loader):
        """
        Performs evaluation of a given model to get the MAE, MSE, FPS performance

        Arguments:
            data_loader {DataLoader} -- DataLoader of the dataset to be used
        """

        # set the model to eval mode
        self.model.eval()

        timer = Timer()
        elapsed = 0
        mae = 0
        mse = 0

        # predetermined save frequency of density maps on certain datasets
        if self.dataset == 'mall':
            save_freq = 100
        elif self.dataset == 'micc':
            save_freq = 50
        else:
            save_freq = 50

        # begin evaluating on the dataset
        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(data_loader)):
                # prepare the input images
                images = to_var(images, self.use_gpu)
                images = images.float()

                # prepare the groundtruth targets
                targets = [to_var(torch.Tensor(target), self.use_gpu) for target in targets]
                targets = torch.stack(targets)
                
                # generate output of model
                timer.tic()
                output = self.model(images)
                elapsed += timer.toc(average=False)

                # if model is MARUNet, divide output by 50 as designed by original proponents
                if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
                    output = output[0] / 50

                ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]
                model = self.pretrained_model.split('/')
                file_path = os.path.join(self.model_test_path, self.dataset_info + ' epoch ' + self.get_epoch_num())
                
                # generate copies of density maps as images 
                # if difference between predicted and actual counts are bigger than 1
                if self.fail_cases:
                    t = targets[0].cpu().detach().numpy()
                    o = output[0].cpu().detach().numpy()

                    gt_count = round(np.sum(t))
                    et_count = round(np.sum(o))

                    diff = abs(gt_count - et_count)

                    if (diff > 0):
                        save_plots(os.path.join(file_path, 'failure cases', str(diff)), output, targets, ids)
                
                # generate copies of density maps as images
                if self.save_output_plots and i % save_freq == 0:
                    save_plots(file_path, output, targets, ids)

                # update MAE and MSE (summation part of the formula)
                mae += abs(output.sum() - targets.sum()).item()
                mse += ((targets.sum() - output.sum())*(targets.sum() - output.sum())).item()

        # compute for MAE, MSE and FPS
        mae = mae / len(data_loader)
        mse = np.sqrt(mse / len(data_loader))
        fps = len(data_loader) / elapsed

        return mae, mse, fps

    def pred(self):

        # set the model to eval mode
        self.model.eval()
        data_loader = self.data_loader

        timer = Timer()
        elapsed = 0

        mae = 0
        mse = 0

        if self.dataset == 'mall':
            save_freq = 100
        elif self.dataset == 'micc':
            save_freq = 50
        else:
            save_freq = 1

        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(data_loader)):
                images = to_var(images, self.use_gpu)
                images = images.float()

                timer.tic()
                output = self.model(images)
                elapsed += timer.toc(average=False)

                ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]

                if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
                    output = output[0] / 50

                model = self.pretrained_model.split('/')
                file_path = os.path.join(self.model_test_path, '{} {} epoch {}'.format(self.model_name, self.dataset_info, self.get_epoch_num()))

                if self.save_output_plots and i % save_freq == 0:
                    save_plots(file_path, output, [], ids, pred=True)

    def test(self):
        """
        Evaluates the performance of the model using the test dataset
        """

        # evaluate the model
        out = self.eval(self.data_loader)

        # log the performance
        log = ('mae: {:.6f}, mse: {:.6f}, '
               'fps: {:.4f}')
        log = log.format(out[0], out[1], out[2])
        write_print(self.output_txt, log)

        epoch_num = self.get_epoch_num()
        write_to_file(self.compile_txt, 'epoch {} | {}'.format(epoch_num, log))

        try:
            if (int(epoch_num) % 5 == 0):
                write_to_file(self.compile_txt, '')
        except:
            pass

    def get_epoch_num(self):
        '''
        Gets the epoch number given the format of the pretrained model's file name
        '''

        if 'SKT experiments' in self.model_test_path:
            epoch_num = self.output_txt.split('/')[-1]
            epoch_num = epoch_num[epoch_num.rfind('epoch_')+6:epoch_num.rfind('_mae')]
        else:
            epoch_num = self.output_txt[self.output_txt.rfind('_')+1:-4].replace('.pth.tar', '')

        return epoch_num
