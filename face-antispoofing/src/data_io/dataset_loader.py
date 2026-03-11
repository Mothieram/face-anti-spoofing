# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io import transform as trans


def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    trainset = DatasetFolderFT(root_path, train_transform,
                               None, conf.ft_width, conf.ft_height)
    num_workers = int(getattr(conf, "num_workers", 16))
    pin_memory = bool(getattr(conf, "pin_memory", True))
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None)
    return train_loader
