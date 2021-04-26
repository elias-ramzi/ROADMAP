import numpy as np
from torchvision import datasets

from .base_dataset import BaseDataset


class Cub200Dataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        dataset = datasets.ImageFolder(self.data_dir)
        paths = np.array([a for (a, b) in dataset.imgs])
        labels = np.array([b for (a, b) in dataset.imgs])

        sorted_lb = list(sorted(set(labels)))
        if mode == 'train':
            set_labels = set(sorted_lb[:len(sorted_lb) // 2])
        elif mode == 'test':
            set_labels = set(sorted_lb[len(sorted_lb) // 2:])

        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            if lb in set_labels:
                self.paths.append(pth)
                self.labels.append(lb)
