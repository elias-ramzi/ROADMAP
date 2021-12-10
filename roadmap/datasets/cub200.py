import os

import numpy as np
from torchvision import datasets

from .base_dataset import BaseDataset


class Cub200Dataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, load_super_labels=False, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.load_super_labels = load_super_labels

        dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'images'))
        paths = np.array([a for (a, b) in dataset.imgs])
        labels = np.array([b for (a, b) in dataset.imgs])

        sorted_lb = list(sorted(set(labels)))
        if mode == 'train':
            set_labels = set(sorted_lb[:len(sorted_lb) // 2])
        elif mode == 'test':
            set_labels = set(sorted_lb[len(sorted_lb) // 2:])
        elif mode == 'all':
            set_labels = sorted_lb

        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            if lb in set_labels:
                self.paths.append(pth)
                self.labels.append(lb)

        self.super_labels = None
        if self.load_super_labels:
            with open(os.path.join(self.data_dir, "classes.txt")) as f:
                lines = f.read().split("\n")
            lines.remove("")
            labels_id = list(map(lambda x: int(x.split(" ")[0])-1, lines))
            super_labels_name = list(map(lambda x: x.split(" ")[2], lines))
            slb_names_to_id = {x: i for i, x in enumerate(sorted(set(super_labels_name)))}
            super_labels = [slb_names_to_id[x] for x in super_labels_name]
            labels_to_super_labels = {lb: slb for lb, slb in zip(labels_id, super_labels)}
            self.super_labels = [labels_to_super_labels[x] for x in self.labels]

        self.get_instance_dict()
