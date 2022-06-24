import os.path as osp
import torch
from torch.utils.data import TensorDataset
import cv2
import os 
import glob
import numpy as np
import csv
import h5py

from collections import Counter

class MallDataset(TensorDataset):

    def __init__(self,
                 data_path,
                 mode,
                 new_size,
                 density_sigma,
                 image_transform=None,
                 targets_resize=1,
                 fail_cases=False,
                 part=None):
        """Initializes the dataset

        Arguments:
            data_path {string} -- root path to the FER2013 dataset
            mode {string} -- current mode of the network
            new_size {int} -- rescaled size of the image
            image_transform {object} -- produces different dataset
            augmentation techniques
        """

        self.data_path = data_path
        self.mode = mode
        self.new_size = new_size
        self.image_transform = image_transform
        self.targets_resize = targets_resize

        self.ids = []
        self.targets = []

        if fail_cases == True:
            self.image_path = osp.join(self.data_path, 'test', '%s')
            self.list_path = osp.join(self.data_path, 'maru_mall_fail.txt')

            image_list = open(self.list_path, 'r').read()
            self.ids = image_list.split('\n')[829+84:]
            self.image_ids = self.ids
            self.targets = [i.replace('jpg', 'h5') for i in self.ids]
            self.target_path = osp.join(self.data_path, density_sigma, '%s')

            return

        if self.mode == 'train':
            self.image_path = osp.join(self.data_path, 'train', '%s')
        elif self.mode == 'val':
            self.image_path = osp.join(self.data_path, 'val', '%s')
        elif self.mode == 'test':
            self.image_path = osp.join(self.data_path, 'test', '%s')

        image_path = self.image_path % '*.jpg'
        images = glob.glob(image_path)

        self.ids = [img[img.rfind('\\') + 1:] for img in images]

        if part != None:
            self.ids = self.ids[(part-1)*400 : (part)*400]
        self.image_ids = self.ids
        self.targets = [i.replace('jpg', 'h5') for i in self.ids]
        self.target_path = osp.join(self.data_path, density_sigma, '%s')


    def __len__(self):
        """Returns number of data in the dataset

        Returns:
            int -- number of data in the dataset
        """
        return len(self.ids)

    def __getitem__(self, index):
        """Returns an image and its corresponding class from the dataset

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor, string -- Tensor representation of the pulled image
            and string representation of the class of the image
        """
        image, target, _, _ = self.pull_item(index)

        return image, target

    def pull_item(self, index):
        """Returns an image, its corresponding class, height, and width from
        the dataset

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor, string, int, int -- Tensor representation of the
            pulled image, string representation of the class of the image,
            height of the image, width of the image
        """
        image_id = self.ids[index]

        image = cv2.imread(self.image_path % image_id)
        target = self.pull_target(index)
        height, width, _ = image.shape

        # if self.mode == 'train':
        #    image, target = self.image_transform(image, target)
        #    image = image[:, :, (2, 1, 0)]

        if self.image_transform != None:
            image, target = self.image_transform(image, target)
        ht = image.shape[0]
        wd = image.shape[1]
        ht_1 = int((ht/4)*4)
        wd_1 = int((wd/4)*4)

        image = cv2.resize(image,(wd_1,ht_1))
        # out_size = (target.shape[1] // self.targets_resize, target.shape[0] // self.targets_resize)
        # target = cv2.resize(target, out_size)

        wd_1 = int(wd_1/self.targets_resize)
        ht_1 = int(ht_1/self.targets_resize)
        target = cv2.resize(target,(wd_1,ht_1))
        target = target * ((wd*ht)/(wd_1*ht_1))

        return torch.from_numpy(image).permute(2, 0, 1), torch.unsqueeze(torch.from_numpy(target), 0), height, width
        # return torch.from_numpy(image).permute(2, 0, 1), torch.unsqueeze(torch.from_numpy(target), 0).permute(2, 0, 1), height, width

    def pull_image(self, index):
        """Returns an image from the dataset represented as an ndarray

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            numpy.ndarray -- ndarray representation of the pulled image
        """

        image_id = self.ids[index]
        return cv2.imread(self.image_path % image_id, cv2.IMREAD_COLOR)

    def pull_target(self, index):
        """Returns a class corresponding to an image from the dataset

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            list -- list of x- and y-coordinates of the people in an image from
            the dataset
        """

        target = self.targets[index]
        target_path = self.target_path % target

        target = h5py.File(target_path, 'r')
        target = target['density']
        target = np.array(target)

        return target # torch.unsqueeze(torch.from_numpy(target), 0)

    def pull_tensor(self, index):
        """Returns an image from the dataset represented as a tensor

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor -- Tensor representation of the pulled image
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
