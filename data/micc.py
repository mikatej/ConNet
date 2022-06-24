import os.path as osp
import torch
from torch.utils.data import Dataset
import cv2
import os 
import glob
import numpy as np
import csv
import h5py

from collections import Counter

class MICC(Dataset):

    def __init__(self,
                 data_path,
                 dataset_subcategory,
                 mode,
                 density_sigma,
                 image_transform=None,
                 targets_resize=1):
        """Initializes the dataset

        Arguments:
            data_path {string} -- root path to the FER2013 dataset
            dataset_subcategory {string} -- determines whether flow, groups, 
                                    queue subset or all three combined is used
            mode {string} -- current mode of the network
            density_sigma {string} -- folder name where density maps are saved
            image_transform {object} -- applies augmentation to images
            targets_resize {int} -- resizes the target according to output size
                                    of a given model
        """

        self.data_path = data_path
        self.mode = mode
        self.image_transform = image_transform
        self.dataset_subcategory = dataset_subcategory
        self.targets_resize = targets_resize

        self.ids = []
        self.image_ids = []
        self.targets = []

        file_path = osp.join(self.data_path, '%s')

        if self.mode == 'train':
            list_path = osp.join(file_path, "%s_train.txt")
        elif self.mode == 'val':
            list_path = osp.join(file_path, "%s_validation.txt")
        elif self.mode == 'test':
            list_path = osp.join(file_path, "%s_testing.txt")

        categories = ['flow', 'groups', 'queue']
        if self.dataset_subcategory == 'all':
            # categories = [self.dataset_subcategory]

            self.image_path = osp.join(file_path, 'RGB', '%s')
            self.target_path = osp.join(file_path, density_sigma, '%s')
            for c in categories:
                list_file = list_path % (c, c)

                image_list = open(list_file, 'r').read()
                ids = image_list.split('\n')
                self.ids.extend([(c, image_id) for image_id in ids])
                self.image_ids.extend(['{}_{}'.format(c, image_id) for image_id in ids])
                self.targets.extend([(c, image_id.replace('.png', '.h5')) for image_id in ids])

        elif self.dataset_subcategory in categories:
            list_file = list_path % (self.dataset_subcategory, self.dataset_subcategory)
            image_list = open(list_file, 'r').read()

            self.ids = image_list.split('\n')
            self.image_ids = self.ids
            self.targets = [i.replace('.png', '.h5') for i in self.ids]

            file_path = file_path % self.dataset_subcategory
            self.image_path = osp.join(file_path, 'RGB', '%s')
            self.target_path = osp.join(file_path, density_sigma, '%s')

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
            torch.Tensor, torch.Tensor -- Tensor representation of the pulled image
            and Tensor representation of the class of the image
        """
        image, target, _, _ = self.pull_item(index)

        if self.mode == 'pred':
            return image,  None

        return image, target

    def pull_item(self, index):
        """Returns an image, its corresponding class, height, and width from
        the dataset

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor, torch.Tensor, int, int -- Tensor representation of the
            pulled image, Tensor representation of the class of the image,
            height of the image, width of the image
        """

        # get image and density map target
        image_id = self.ids[index]

        image = cv2.imread(self.image_path % image_id)
        target = self.pull_target(index)

        # apply augmentation
        if self.image_transform != None:
            image, target = self.image_transform(image, target)

        # get original height and width of image
        height, width, _ = image.shape
        ht = image.shape[0]
        wd = image.shape[1]
        ht_1 = int((ht/4)*4)
        wd_1 = int((wd/4)*4)

        image = cv2.resize(image,(wd_1,ht_1))

        # resize the target according to output size of the model
        wd_1 = int(wd_1/self.targets_resize)
        ht_1 = int(ht_1/self.targets_resize)
        target = cv2.resize(target,(wd_1,ht_1))                
        target = target * ((wd*ht)/(wd_1*ht_1))

        return torch.from_numpy(image).permute(2, 0, 1), torch.unsqueeze(torch.from_numpy(target), 0), height, width

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
            np.array -- np.array representation of the groundtruth density map
        """
        target = self.targets[index]
        target_path = self.target_path % target

        target = h5py.File(target_path, 'r')
        target = target['density']
        target = np.array(target)

        return target 

    def pull_tensor(self, index):
        """Returns an image from the dataset represented as a tensor

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor -- Tensor representation of the pulled image
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
