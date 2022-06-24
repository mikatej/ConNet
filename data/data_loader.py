import torch
from data.mall_dataset import MallDataset
from data.micc import MICC
from data.fdst import FDST
from torch.utils.data import DataLoader
from data.augmentations import Augmentations, BaseTransform

def collate(batch):
    """Collate function used by the DataLoader"""

    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images, 0), targets


def get_loader(config):
    """Returns the data loader and dataset image ids

    Arguments:
        config {argparse.Namespace} -- contains information needed for the instantiation
            of the DataLoader object (e.g., data path, mode)

    Returns:
        DataLoader -- DataLoader object of the specified dataset to be used
        list -- list of image IDs in the dataset 
    """

    print(type(config))
    raise Exception()
    dataset = None
    loader = None
    targets_resize = 1
    image_transform = None

    # targets_resize refers to how much the dimensions of the
    # target density map must be downscaled to match the output
    # size of the model used
    if 'CSRNet' in config.model:
        targets_resize = 2 ** 3
    elif config.model == 'MCNN':
        targets_resize = 2 ** 2

    if config.augment_exp:
        image_transform = Augmentations(brightness=config.brightness_change,
            scale=config.resolution_scale)

    # get the Dataset object 
    if config.dataset == 'micc':
        dataset = MICC(data_path=config.micc_data_path,
                        dataset_subcategory=config.dataset_subcategory,
                        mode=config.mode,                            
                        density_sigma=config.density_sigma,
                        image_transform=image_transform,
                        targets_resize=targets_resize)

    if config.dataset == 'mall':        
        dataset = MallDataset(data_path=config.mall_data_path,
                        mode=config.mode,                            
                        density_sigma=config.density_sigma,
                        image_transform=image_transform,
                        targets_resize=targets_resize)

    if config.dataset == 'fdst':
        dataset = FDST(data_path=config.fdst_data_path,
                        mode=config.mode,                        
                        density_sigma=config.density_sigma,
                        image_transform=image_transform,
                        targets_resize=targets_resize,
                        outdoor=config.outdoor)

    # get the data loader
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
