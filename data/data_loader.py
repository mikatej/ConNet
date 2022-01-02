import torch
from data.mall_dataset import MallDataset
from data.micc import MICC
from torch.utils.data import DataLoader
from data.augmentations import Augmentations, BaseTransform

def collate(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images, 0), targets


def get_loader(config):

    dataset = None
    loader = None
    targets_resize = 1

    if config.model == 'CSRNet' or config.model == 'CAN':
        targets_resize = 2 ** 3
    elif config.model == 'MCNN' or config.model == 'SirMCNN':
        targets_resize = 2 ** 2

    if config.dataset == 'micc':

        if config.mode == 'train':
            if config.augment is True:
                image_transform = Augmentations(config.means)
            elif config.base_transform:
                image_transform = BaseTransform(config.means)
            else:
                image_transform = None

            dataset = MICC(data_path=config.micc_data_path,
                            dataset_subcategory=config.dataset_subcategory,
                            mode='train',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'val':
            if config.base_transform:
                image_transform = BaseTransform(config.means)
            else:
                image_transform = None
            dataset = MICC(data_path=config.micc_data_path,
                            dataset_subcategory=config.dataset_subcategory,
                            mode='val',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'test' or config.mode == 'pred':
            if config.base_transform:
                image_transform = BaseTransform(config.means)
            else:
                image_transform = None
            dataset = MICC(data_path=config.micc_data_path,
                            dataset_subcategory=config.dataset_subcategory,
                            mode='test',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

    if config.dataset == 'mall':

        if config.mode == 'train':
            if config.augment is True:
                image_transform = Augmentations(config.means)
            elif config.base_transform:
                image_transform = BaseTransform(config.means)
            else:
                image_transform = None

            dataset = MallDataset(data_path=config.mall_data_path,
                            mode='train',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'val':
            if config.base_transform:
                image_transform = BaseTransform(config.means)
            else:
                image_transform = None
            dataset = MallDataset(data_path=config.mall_data_path,
                            mode='val',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'test' or config.mode == 'pred':
            if config.base_transform:
                image_transform = BaseTransform(config.means)
            else:
                image_transform = None
            dataset = MallDataset(data_path=config.mall_data_path,
                            mode='test',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

    if dataset is not None:
        if config.mode == 'train':
            loader = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                collate_fn=collate,
                                num_workers=4,
                                pin_memory=True)

        elif config.mode == 'val' or config.mode == 'test' or config.mode == 'pred':
            loader = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                collate_fn=collate,
                                num_workers=4,
                                pin_memory=True)

    return loader, dataset.image_ids
