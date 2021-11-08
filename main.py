import os
from utils.utils import write_print, mkdir
import argparse
from solver import Solver
from data.data_loader import get_loader
from torch.backends import cudnn
from datetime import datetime
import zipfile
import torch
import numpy as np


def zip_directory(path, zip_file):
    """Stores all py and cfg project files inside a zip file

    Arguments:
        path {string} -- current path
        zip_file {zipfile.ZipFile} -- zip file to contain the project files
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.py') or file.endswith('cfg'):
            zip_file.write(os.path.join(path, file))
            if file.endswith('cfg'):
                os.remove(file)


def save_config(path, version, config):
    """saves the configuration of the experiment

    Arguments:
        path {str} -- save path
        version {str} -- version of the model based on the time
        config {dict} -- contains argument and its value

    """
    cfg_name = '{}.{}'.format(version, 'cfg')

    with open(cfg_name, 'w') as f:
        for k, v in config.items():
            f.write('{}: {}\n'.format(str(k), str(v)))

    zip_name = '{}.{}'.format(version, 'zip')
    zip_name = os.path.join(path, zip_name)
    zip_file = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    zip_directory('.', zip_file)
    zip_file.close()


def string_to_boolean(v):
    """Converts string to boolean

    Arguments:
        v {string} -- string representation of a boolean values;
        must be true or false

    Returns:
        boolean -- boolean true or false
    """
    return v.lower() in ('true')


def main(version, config, output_txt):
    # for fast training
    cudnn.benchmark = True

    data_loader = get_loader(config)
    solver = Solver(version, data_loader, vars(config), output_txt)

    if config.mode == 'train':
        temp_save_path = os.path.join(config.model_save_path, version)
        mkdir(temp_save_path)
        solver.train()

    elif config.mode == 'val' or config.mode == 'test':
        solver.test()

    elif config.mode == 'pred':
        solver.pred()


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.nan)
    parser = argparse.ArgumentParser()

    # dataset info
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--class_count', type=int, default=102,
                        help='Number of classes in dataset')
    parser.add_argument('--dataset', type=str, default='ip102',
                        choices=['ip102', 'grocery_store', 'imagenet', 'sd',
                                 'chestxray8'],
                        help='Dataset to use')
    parser.add_argument('--new_size', type=int, default=224,
                        help='New height and width of input images')
    parser.add_argument('--means', type=tuple, default=(104, 117, 123),
                        help='Mean values of the dataset')
    parser.add_argument('--augment', type=string_to_boolean, default=True,
                        help='Toggles data augmentation')

    # training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=120,
                        help='Number of epochs')
    parser.add_argument('--learning_sched', type=list, default=[100, 110],
                        help='List of epochs to reduce the learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--model', type=str, default='ResNet50',
                        choices=['AlexNet', 'GoogleNet',
                                 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN',
                                 'ResNet18', 'ResNet34', 'ResNet50',
                                 'ResNet101', 'ResNet152',
                                 'DenseNet121', 'DenseNet169', 'DenseNet201'],
                        help='CNN model to use')
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='Pre-trained model')
    parser.add_argument('--init_weights', type=string_to_boolean, default=True,
                        help='Toggles weight initialization')
    parser.add_argument('--imagenet_pretrain', type=string_to_boolean,
                        default=True,
                        help='Toggles pretrained weights for vision models')

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test', 'pred'],
                        help='Mode of execution')
    parser.add_argument('--use_gpu', type=string_to_boolean, default=True,
                        help='Toggles the use of GPU')

    # ip102 dataset
    parser.add_argument('--ip102_data_path', type=str,
                        default='../../Datasets/ip102_v1.1/',
                        help='IP102 dataset path')

    # path
    parser.add_argument('--model_save_path', type=str, default='./weights',
                        help='Path for saving weights')
    parser.add_argument('--model_test_path', type=str, default='./tests',
                        help='Path for saving test results')

    # epoch step size
    parser.add_argument('--loss_log_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=10)

    config = parser.parse_args()

    args = vars(config)
    output_txt = ''

    if args['mode'] == 'train':
        version = str(datetime.now()).replace(':', '_')
        version = '{}_train'.format(version)
        path = args['model_save_path']
        path = os.path.join(path, version)
        output_txt = os.path.join(path, '{}.txt'.format(version))

    elif args['mode'] == 'val':
        model = args['pretrained_model'].split('/')
        version = '{}_val_{}'.format(model[0], model[1])
        path = args['model_test_path']
        path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))

    elif args['mode'] == 'test':
        model = args['pretrained_model'].split('/')
        version = '{}_test_{}'.format(model[0], model[1])
        path = args['model_test_path']
        path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))

    elif args['mode'] == 'pred':
        model = args['pretrained_model'].split('/')
        version = '{}_pred_{}'.format(model[0], model[1])
        path = args['model_test_path']
        path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))

    mkdir(path)
    save_config(path, version, args)

    write_print(output_txt, '------------ Options -------------')
    for k, v in args.items():
        write_print(output_txt, '{}: {}'.format(str(k), str(v)))
    write_print(output_txt, '-------------- End ----------------')

    main(version, config, output_txt)
