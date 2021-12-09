import os
import torch


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