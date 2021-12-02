import os
import torch
import time
import datetime
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model import get_model
from utils.utils import to_var, write_print
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from utils.timer import Timer
import numpy as np

class Solver(object):

    DEFAULTS = {}

    def __init__(self, version, data_loader, config, output_txt):
        """
        Initializes a Solver object
        """

        # data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.data_loader = data_loader
        self.output_txt = output_txt

        self.build_model()

        # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        """
        Instantiates the model, loss criterion, and optimizer
        """

        # instantiate model
        self.model = get_model(self.model,
                               self.imagenet_pretrain,
                               self.model_save_path,
                               self.input_channels,
                               self.class_count)

        # instantiate loss criterion
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss() #.cuda()

        # instantiate optimizer
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.lr,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        # self.optimizer = optim.Adam(params=self.model.parameters(),
        #                             lr=self.lr)

        # print network
        self.print_network(self.model, 'VGGNet')

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
        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.pretrained_model))))
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
              "loss: {:.4f}".format(elapsed,
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
            '{}/{}.pth'.format(self.version, e + 1)
        )

        torch.save(self.model.state_dict(), path)

    def model_step(self, images, labels):
        """
        A step for each iteration
        """

        # set model in training mode
        self.model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # forward pass
        output = self.model(images)

        # print(output.size(), "|", labels.size())
        # print(output[0])
        # print()
        # print(labels[0])

        # print()
        # print(output.shape, "|", labels.shape)
        # print(type(output))
        # print(type(output[0]))
        # print(output.squeeze()[0].size())
        # print((labels.squeeze()[0].size()))
        # print()

        # compute loss
        loss = self.criterion(output.squeeze(), labels.squeeze())

        # compute gradients using back propagation
        loss.backward()

        # update parameters
        self.optimizer.step()

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

        # start with a trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('/')[-1])
        else:
            start = 0

        sched = 0

        # start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, labels) in enumerate(tqdm(self.data_loader)):
                images = to_var(images, self.use_gpu)

                # labels = to_var(torch.tensor(labels), self.use_gpu)
                labels = [to_var(torch.Tensor(label), self.use_gpu) for label in labels]
                labels = torch.stack(labels)

                loss = self.model_step(images, labels)
                print("\t{:.6f}".format(loss.item()))

            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time, iters_per_epoch, e, i, loss)
                self.losses.append((e, loss))

            # save model
            if (e + 1) % self.model_save_step == 0:
                self.save_model(e)

            num_sched = len(self.learning_sched)
            if num_sched != 0 and sched < num_sched:
                if (e + 1) == self.learning_sched[sched]:
                    self.lr /= 10
                    print('Learning rate reduced to', self.lr)
                    sched += 1

        # print losses
        write_print(self.output_txt, '\n--Losses--')
        for e, loss in self.losses:
            write_print(self.output_txt, str(e) + ' {:.4f}'.format(loss))

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

        with torch.no_grad():
            for images, labels in tqdm(data_loader):

                images = to_var(images, self.use_gpu)
                labels = to_var(torch.LongTensor(labels), self.use_gpu)

                timer.tic()
                output = self.model(images)
                elapsed += timer.toc(average=False)

                _, top_1_output = torch.max(output.data, dim=1)
                # out.append(str(sm(output.data).tolist()))
                y_true = torch.cat((y_true, labels))
                y_pred = torch.cat((y_pred, top_1_output))

        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

        acc = accuracy_score(y_true, y_pred)
        b_acc = balanced_accuracy_score(y_true, y_pred)
        labels = [x for x in range(self.class_count)]

        f1_mi = f1_score(y_true, y_pred, labels=labels, average='micro')
        f1_ma = f1_score(y_true, y_pred, labels=labels, average='macro')
        f1_w = f1_score(y_true, y_pred, labels=labels, average='weighted')
        fps = y_true.shape[0] / elapsed

        return acc, b_acc, f1_mi, f1_ma, f1_w, fps

    def pred(self):

        # set the model to eval mode
        self.model.eval()
        data_loader = self.data_loader

        y_true = to_var(torch.LongTensor([]), self.use_gpu)
        y_pred = to_var(torch.LongTensor([]), self.use_gpu)
        out = to_var(torch.FloatTensor([]), self.use_gpu)
        sm = nn.Softmax()
        timer = Timer()
        elapsed = 0

        with torch.no_grad():
            for images, labels in tqdm(data_loader):

                images = to_var(images, self.use_gpu)
                labels = to_var(torch.LongTensor(labels), self.use_gpu)

                timer.tic()
                output = self.model(images)
                elapsed += timer.toc(average=False)

                _, top_1_output = torch.max(output.data, dim=1)
                top_1, _ = torch.max(sm(output.data), dim=1)
                out = torch.cat((out, top_1))
                y_true = torch.cat((y_true, labels))
                y_pred = torch.cat((y_pred, top_1_output))

        out = out.cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        print(out.shape, y_true.shape, y_pred.shape)

        for i, _ in enumerate(out):
            strOut = str(out[i]) + ',' + str(y_pred[i]) + ',' + str(y_true[i])
            write_print(self.output_txt, strOut)

    def test(self):
        """
        Evaluates the performance of the model using the test dataset
        """
        out = self.eval(self.data_loader)
        log = ('acc: {:.4f}, b_acc: {:.4f}, '
               'f1_micro: {:.4f}, f1_macro: {:.4f}, f1_weight: {:.4f}, '
               'fps: {:.4f}')
        log = log.format(out[0], out[1], out[2], out[3], out[4], out[5])
        write_print(self.output_txt, log)
