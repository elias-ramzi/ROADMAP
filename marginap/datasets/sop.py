from os.path import join

import pandas as pd

from .base_dataset import BaseDataset


class SOPDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        gt = pd.read_csv(join(self.data_dir, f'Ebay_{mode}.txt'), sep=' ')
        self.paths = gt["path"].apply(lambda x: join(self.data_dir, x)).tolist()
        self.labels = (gt["class_id"] - 1).tolist()
        self.super_labels = (gt["super_class_id"] - 1).tolist()

        self.super_dict = {ct: {} for ct in set(self.super_labels)}
        for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
            try:
                self.super_dict[ct][cl].append(idx)
            except KeyError:
                self.super_dict[ct][cl] = [idx]

        self.instance_dict = {cl: [] for cl in set(self.labels)}
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)
