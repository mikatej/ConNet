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


def main(version, config, output_txt, compile_txt):
    # for fast training
    cudnn.benchmark = True

    data_loader, dataset_ids = get_loader(config)
    solver = Solver(version, data_loader, dataset_ids, vars(config), output_txt, compile_txt)

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
    parser.add_argument('--dataset', type=str, default='micc',
                        choices=['micc', 'mall'],
                        help='Dataset to use')
    parser.add_argument('--dataset_subcategory', type=str, default='all',
                        choices=['flow', 'groups', 'queue', 'all'],
                        help='(If MICC) dataset sequence to use')
    parser.add_argument('--density_sigma', type=str, default='h5py-5',
                        choices=['h5py-3', 'h5py-5'],
                        help='Sigma value for density maps')

    parser.add_argument('--new_size', type=int, default=224,
                        help='New height and width of input images')
    parser.add_argument('--means', type=tuple, default=(104, 117, 123),
                        help='Mean values of the dataset')
    parser.add_argument('--augment', type=string_to_boolean, default=False,
                        help='Toggles data augmentation')
    parser.add_argument('--base_transform', type=string_to_boolean, default=False,
                        help='Toggles base transformation (mean subtraction)')

    # training settings
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default= 0.0001,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--learning_sched', type=list, default=[],
                        help='List of epochs to reduce the learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--model', type=str, default='NLT',
                        choices=['CSRNet', 'MCNN', 'NLT', 'MARUNet', 'SirMCNN'],
                        help='CNN model to use')
    parser.add_argument('--backbone_model', type=str, default='vgg16',
                        choices=['vgg16', 'ResNet50'],
                        help='If NLT, which backbone model to use')
    parser.add_argument('--pretrained_model', type=str,
                        default='pre-trained NLT/0',
                        help='Pre-trained model')
    parser.add_argument('--save_output_plots', type=string_to_boolean, default=True)
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

    # mall dataset
    parser.add_argument('--mall_data_path', type=str,
                        default='../Datasets/mall_dataset/',
                        help='Mall dataset path')

    # micc dataset
    parser.add_argument('--micc_data_path', type=str,
                        default='../Datasets/MICC/',
                        help='MICC dataset path')

    # path
    parser.add_argument('--model_save_path', type=str, default='./weights',
                        help='Path for saving weights')
    parser.add_argument('--model_test_path', type=str, default='./tests',
                        help='Path for saving test results')
    parser.add_argument('--group_save_path', type=str, default=None)

    # epoch step size
    parser.add_argument('--loss_log_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=1)

    config = parser.parse_args()

    args = vars(config)
    output_txt = ''

    if args['group_save_path'] is not None:
        args['model_save_path'] = os.path.join(args['model_save_path'], args['group_save_path'])
        args['model_test_path'] = os.path.join(args['model_test_path'], args['group_save_path'])

    if args['mode'] == 'train':
        dataset = args['dataset']
        if dataset == 'micc':
            dataset = '{} {}'.format(dataset, args['dataset_subcategory'])

        version = str(datetime.now()).replace(':', '_')
        version = '{} {} {}_train'.format(args['model'], dataset, version)
        path = args['model_save_path']
        path = os.path.join(path, version)
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {}.txt'.format(args['model'], version))

    elif args['mode'] == 'val':
        model = args['pretrained_model'].split('/')
        version = '{}_val_{}'.format(model[0], model[1])
        path = args['model_test_path']
        path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {} {}.txt'.format(args['model'], args['mode'], model[0]))

    elif args['mode'] == 'test':
        model = args['pretrained_model'].split('/')
        version = '{}_test_{}'.format(model[0], model[1])
        path = args['model_test_path']
        path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {} {}.txt'.format(args['model'], args['mode'], model[0]))

    elif args['mode'] == 'pred':
        model = args['pretrained_model'].split('/')
        version = '{}_pred_{}'.format(model[0], model[1])
        path = args['model_test_path']
        path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {}.txt'.format(args['model'], model[0]))


    mkdir(path)
    save_config(path, version, args)

    write_print(output_txt, '------------ Options -------------')
    for k, v in args.items():
        write_print(output_txt, '{}: {}'.format(str(k), str(v)))
    write_print(output_txt, '-------------- End ----------------')

    main(version, config, output_txt, compile_txt)
