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
                               self.backbone_model,
                               self.imagenet_pretrain,
                               self.model_save_path,
                               self.input_channels,
                               self.class_count)

        # instantiate loss criterion
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss() #.cuda()

        # instantiate optimizer
        # self.optimizer = optim.SGD(self.model.parameters(),
        #                            lr=self.lr,
        #                            momentum=self.momentum,
        #                            weight_decay=self.weight_decay)

        # self.optimizer = optim.Adam(params=self.model.parameters(),
        #                             lr=self.lr)

        if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr)

        # print network
        # self.print_network(self.model, 'VGGNet')

        # use gpu if enabled
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

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

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """

        if ".pth.tar" not in self.pretrained_model:
            self.model.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}.pth'.format(self.pretrained_model))), strict=False)

        else:
            weights = torch.load(os.path.join(
                self.model_save_path, '{}'.format(self.pretrained_model)))
            self.model.load_state_dict(weights['state_dict'])
            self.optimizer.load_state_dict(weights['optimizer'])

        write_print(self.output_txt,
                    'loaded trained model {}'.format(self.pretrained_model))

    def print_loss_log(self, start_time, iters_per_epoch, e, i, loss):
        """
        Prints the loss and elapsed time for each epoch
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
        Saves a model per e epoch
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth.tar'.format(self.version, e + 1)
        )

        torch.save({'state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())}, path)

    def model_step(self, images, labels, epoch):
        """
        A step for each iteration
        """

        # set model in training mode
        self.model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # print(images.dtype)
        # print(torch.cuda.memory_summary())
        # forward pass
        if self.model_name == 'MCNN':
            output = self.model(images, labels)
        else:
            images = images.float()
            output = self.model(images)

        if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
            if 'ConNet' in self.model_name:
                _, output = output
            output, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = output
            amp_gt = [get_amp_gt_by_value(l) for l in labels]
            amp_gt = torch.stack(amp_gt).cuda()
            # labels = labels * 50

        # if self.save_output_plots:
        #     file_path = os.path.join(self.compile_txt[:self.compile_txt.rfind('COMPILED')], 'epoch ' + str(epoch +1))
        #     save_plots(file_path, output, labels)

        # compute loss
        if self.model_name == 'MCNN':
            self.model.loss.backward()
            loss = self.model.loss.item()
        elif 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
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

        if self.model_name == 'NLT':
            self.lr_scheduler.step()

        # return loss
        return loss

    def train(self):
        """
        Training process
        """
        self.losses = []
        self.top_1_acc = []
        self.top_5_acc = []

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
            for i, (images, labels) in enumerate(tqdm(self.data_loader)):
                images = to_var(images, self.use_gpu)

                # labels = to_var(torch.tensor(labels), self.use_gpu)
                labels = [to_var(torch.Tensor(label), self.use_gpu) for label in labels]
                labels = torch.stack(labels)

                loss = self.model_step(images, labels, e)
                # print("\t{:.6f}".format(loss.item()))

            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time, iters_per_epoch, e, i, loss)
                self.losses.append((e, loss))

            # save model
            if (e + 1) % self.model_save_step == 0:
                self.save_model(e)

            num_sched = len(self.learning_sched)
            if num_sched != 0 and sched < num_sched:
                # if (e + 1) == self.learning_sched[sched]:
                if (e + 1) in self.learning_sched:
                    self.lr /= 10
                    print('Learning rate reduced to', self.lr)
                    sched += 1

        # print losses
        write_print(self.output_txt, '\n--Losses--')
        for e, loss in self.losses:
            write_print(self.output_txt, str(e) + ' {:.10f}'.format(loss))

        # print top_1_acc
        write_print(self.output_txt, '\n--Top 1 accuracy--')
        for e, acc in self.top_1_acc:
            write_print(self.output_txt, str(e) + ' {:.4f}'.format(acc))

        # print top_5_acc
        write_print(self.output_txt, '\n--Top 5 accuracy--')
        for e, acc in self.top_5_acc:
            write_print(self.output_txt, str(e) + ' {:.4f}'.format(acc))

    def eval(self, data_loader):
        """
        Returns the count of top 1 and top 5 predictions
        """

        # set the model to eval mode
        self.model.eval()

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
        elif self.dataset == 'micc':
            save_freq = 50
        else:
            save_freq = 50

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(data_loader)):
                images = to_var(images, self.use_gpu)

                labels = [to_var(torch.Tensor(label), self.use_gpu) for label in labels]
                labels = torch.stack(labels)
                images = images.float()
                
                timer.tic()
                output = self.model(images)
                elapsed += timer.toc(average=False)

                ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]

                if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
                    output = output[0] / 50

                model = self.pretrained_model.split('/')
                file_path = os.path.join(self.model_test_path, model[0], self.dataset_info + ' epoch ' + model[1])
                if self.fail_cases:
                    l = labels[0].cpu().detach().numpy()
                    o = output[0].cpu().detach().numpy()

                    gt_count = round(np.sum(l))
                    et_count = round(np.sum(o))

                    diff = abs(gt_count - et_count)

                    if (diff > 0):
                        save_plots(os.path.join(file_path, 'failure cases', str(diff)), output, labels, ids)
                if self.save_output_plots and i % save_freq == 0:
                    save_plots(file_path, output, labels, ids)

                # _, top_1_output = torch.max(output.data, dim=1)
                # out.append(str(sm(output.data).tolist()))
                y_true = torch.cat((y_true, labels))
                y_pred = torch.cat((y_pred, output))

                mae += abs(output.sum() - labels.sum()).item()
                mse += ((labels.sum() - output.sum())*(labels.sum() - output.sum())).item()

        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

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
                file_path = os.path.join(self.model_test_path, model[0], '{} {} epoch {}'.format(self.model_name, self.dataset_info, model[1]))

                if self.save_output_plots and i % save_freq == 0:
                    save_plots(file_path, output, [], ids, pred=True)

    def test(self):
        """
        Evaluates the performance of the model using the test dataset
        """
        out = self.eval(self.data_loader)
        log = ('mae: {:.6f}, mse: {:.6f}, '
               'fps: {:.4f}')
        log = log.format(out[0], out[1], out[2])
        write_print(self.output_txt, log)

        epoch_num = self.output_txt[self.output_txt.rfind('_')+1:-4].replace('.pth.tar', '')
        write_to_file(self.compile_txt, 'epoch {} | {}'.format(epoch_num, log))

        if (int(epoch_num) % 5 == 0):
            write_to_file(self.compile_txt, '')
