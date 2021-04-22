import os

import numpy as np
import scipy.io as sio

from .base_dataset import BaseDataset


class Cars196(BaseDataset):

    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        img_data = sio.loadmat(os.path.join(self.data_dir, "cars_annos.mat"))
        labels = np.array([i[0, 0] - 1 for i in img_data["annotations"]["class"][0]])
        paths = [os.path.join(self.data_dir, i[0]) for i in img_data["annotations"]["relative_im_path"][0]]
        class_names = [i[0] for i in img_data["class_names"][0]]

        sorted_lb = list(sorted(set(labels)))
        if mode == 'train':
            set_labels = set(sorted_lb[:len(sorted_lb) // 2])
            self.labels_name = class_names[:len(sorted_lb) // 2]
        elif mode == 'test':
            set_labels = set(sorted_lb[len(sorted_lb) // 2:])
            self.labels_name = class_names[len(sorted_lb) // 2:]

        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            if lb in set_labels:
                self.paths.append(pth)
                self.labels.append(lb)
