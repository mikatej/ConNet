import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np


def to_var(x, use_gpu, requires_grad=False):
    """Toggles the use of cuda of a Tensor variable

    Arguments:
        x {torch.Tensor} -- Tensor variable to toggle the CUDA of
        use_gpu {boolean} -- whether or not use of GPU is permitted

    Keyword Arguments:
        requires_grad {boolean} -- whether or not gradients must be computed

    Returns:
        torch.Tensor -- modified Tensor variable
    """
    if torch.cuda.is_available() and use_gpu:
        x = x.cuda()
    x.requires_grad = requires_grad
    return x


def mkdir(directory):
    """Creates the directory if not yet existing

    Arguments:
        directory {string} -- directory to be created
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_print(path, text):
    """Displays text in console and saves in text file

    Arguments:
        path {string} -- path to text file
        text {string} -- text to display and save
    """
    file = open(path, 'a')
    file.write(text + '\n')
    file.close()
    print(text)

def write_to_file(path, text):
    """Saves text in text file

    Arguments:
        path {string} -- path to text file
        text {string} -- text to save
    """
    file = open(path, 'a')
    file.write(text + '\n')
    file.close()

def save_plots(file_path, output, labels, ids, save_label=False):
    """Saves the density maps as images

    Arguments:
        file_path {string} -- path to save the images
        output {torch.Tensor} -- density map outputted by the model
        labels {torch.Tensor} -- groundtruth density map
        ids {list} -- list of the file names of the dataset images

    Keyword Arguments:
        save_label {boolean} -- toggles whether the labels should be saved as image
            {default: False}
    """

    # output folder
    dm_file_path = os.path.join(file_path, 'density maps')
    mkdir(dm_file_path)

    # file paths
    file_path = os.path.join(file_path , '%s')
    dm_file_path = os.path.join(dm_file_path, '%s')

    for i in range(0, len(ids)):

        # save density map outputted by the model
        file_name = dm_file_path % (ids[i])
        o = output[i].cpu().detach().numpy()
        et_count = np.sum(o)
        o = o.squeeze()
        plt.imsave(file_name, o)

        # prepare other file names
        file_name2 = file_path % (ids[i])
        file_name3 = dm_file_path % ("[gt] {}".format(ids[i]))

        # save the ground-truth density map
        l = labels[i].cpu().detach().numpy()
        gt_count = np.sum(l)
        l = l.squeeze()

        if save_label:
            plt.imsave(file_name3, l)

        # plot the two density maps in the same image
        plt.subplot(1, 2, 1)
        plt.imshow(l)
        text = plt.text(0, 0, 'actual: {} ({})\npredicted: {} ({})\n\n'.format(round(gt_count), str(gt_count), round(et_count), str(et_count)))
        
        plt.subplot(1, 2, 2)
        plt.imshow(o)
        plt.savefig(file_name2)

        text.set_visible(False)

def get_amp_gt_by_value(target, threshold=1e-5):
    """Creates the attention map groundtruth used by MARUNet

    Arguments:
        target {torch.Tensor} -- groundtruth density map

    Keyword Arguments:
        threshold {float} -- threshold value used for generating the attention map {default: 1e-5}
    """
    seg_map = (target>threshold).float().cuda()
    return seg_map
