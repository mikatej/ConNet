import os.path as osp
import torch
from torch.utils.data import Dataset
import cv2
import os 
import glob
import numpy as np
import csv

from collections import Counter

class MallDataset(Dataset):

    def __init__(self,
                 data_path,
                 mode,
                 new_size,
                 image_transform=None):
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

        self.ids = []
        self.targets = []

        if self.mode == 'train':
            self.image_path = osp.join(self.data_path, 'train')
        elif self.mode == 'val':
            self.image_path = osp.join(self.data_path, 'val')
        elif self.mode == 'test':
            self.image_path = osp.join(self.data_path, 'test')

        image_path = osp.join(self.image_path, '*.jpg')
        images = glob.glob(image_path)

        for img in images:
            image_id = img[img.rfind('\\') + 1:]
            self.ids.append(image_id)

            csv_id = image_id[:image_id.find(".jpg")] + ".csv"
            csv_path = osp.join(self.image_path, csv_id)

            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                data = np.float_(list(reader)[1:])
                
                self.targets.append(data)

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
        target = self.targets[index]
        height, width, _ = image.shape

        if self.image_transform is not None:
            image, target = self.image_transform(image, target)
            image = image[:, :, (2, 1, 0)]

        return torch.from_numpy(image).permute(2, 0, 1), target, height, width

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
        return self.targets[index]

    def pull_tensor(self, index):
        """Returns an image from the dataset represented as a tensor

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor -- Tensor representation of the pulled image
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)