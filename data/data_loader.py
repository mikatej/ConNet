import torch
from data.mall_dataset import MallDataset
from data.micc import MICC
from data.pets import PETS
from data.fdst import FDST
from torch.utils.data import DataLoader
from data.augmentations import Augmentations, BaseTransform

def collate(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])

        if sample[1] != None:
            targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images, 0), targets


def get_loader(config):

    dataset = None
    loader = None
    targets_resize = 1
    image_transform = None

    if 'CSRNet' in config.model:
        targets_resize = 2 ** 3
    elif config.model == 'MCNN':
        targets_resize = 2 ** 2


    if config.augment_exp:
        image_transform = Augmentations(brightness=config.brightness_change,
            scale=config.resolution_scale)

    if config.dataset == 'micc':

        if config.mode == 'train':

            dataset = MICC(data_path=config.micc_data_path,
                            dataset_subcategory=config.dataset_subcategory,
                            mode='train',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'val':
            dataset = MICC(data_path=config.micc_data_path,
                            dataset_subcategory=config.dataset_subcategory,
                            mode='val',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'test' or config.mode == 'pred':

            dataset = MICC(data_path=config.micc_data_path,
                            dataset_subcategory=config.dataset_subcategory,
                            mode='test',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

    if config.dataset == 'mall':

        if config.mode == 'train':
            dataset = MallDataset(data_path=config.mall_data_path,
                            mode='train',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'val':
            dataset = MallDataset(data_path=config.mall_data_path,
                            mode='val',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize)

        elif config.mode == 'test' or config.mode == 'pred':

            dataset = MallDataset(data_path=config.mall_data_path,
                            mode='test',
                            new_size=config.new_size,
                            density_sigma=config.density_sigma,
                            image_transform=image_transform,
                            targets_resize=targets_resize,
                            # part=config.part
                            )


    if config.dataset == 'pets':
        dataset = PETS(data_path=config.pets_data_path,
                        mode=config.mode,
                        new_size=config.new_size,
                        density_sigma=config.density_sigma,
                        image_transform=image_transform,
                        targets_resize=targets_resize)


    if config.dataset == 'fdst':
        dataset = FDST(data_path=config.fdst_data_path,
                        mode=config.mode,
                        new_size=config.new_size,
                        density_sigma=config.density_sigma,
                        image_transform=image_transform,
                        targets_resize=targets_resize,
                        outdoor=config.outdoor)

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
