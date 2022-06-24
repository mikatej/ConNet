import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np


def to_var(x, use_gpu, requires_grad=False):
    if torch.cuda.is_available() and use_gpu:
        x = x.cuda()
    x.requires_grad = requires_grad
    return x


def mkdir(directory):
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

def save_plots(file_path, output, labels, ids, pred=False):
    dm_file_path = os.path.join(file_path, 'density maps')

    if not os.path.exists(dm_file_path):
        # os.makedirs(file_path)
        os.makedirs(dm_file_path)

    file_path = os.path.join(file_path , '%s')
    dm_file_path = os.path.join(dm_file_path, '%s')


    # for i, o in enumerate(output):
    for i in range(0, len(ids), 5):
        # file_name = file_path % (str(time.time()) + '.png')
        file_name = dm_file_path % (ids[i])
        o = output[i].cpu().detach().numpy()
        et_count = np.sum(o)
        o = o.squeeze()
        plt.imsave(file_name, o)

        if pred:
            return

        file_name2 = file_path % (ids[i])
        file_name3 = dm_file_path % ("[gt] {}".format(ids[i]))

        l = labels[i].cpu().detach().numpy()
        gt_count = np.sum(l)
        l = l.squeeze()
        plt.imsave(file_name3, l)

        plt.subplot(1, 2, 1)
        plt.imshow(l)
        text = plt.text(0, 0, 'actual: {} ({})\npredicted: {} ({})\n\n'.format(round(gt_count), str(gt_count), round(et_count), str(et_count)))
        
        plt.subplot(1, 2, 2)
        plt.imshow(o)
        plt.savefig(file_name2)

        text.set_visible(False)
        # plt.imsave(file_name3, l)

        # np.savetxt(file_path % (ids[i].replace('.jpg', '.txt')), o)

def get_amp_gt_by_value(target, threshold=1e-5):
    seg_map = (target>threshold).float().cuda()
    return seg_map
