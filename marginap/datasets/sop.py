import os
import zipfile
from os.path import join

import pandas as pd
from torchvision.datasets.utils import download_url

import utils as lib
from .base_dataset import BaseDataset


class SOPDataset(BaseDataset):
    url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    filename = 'Stanford_Online_Products.zip'
    md5 = '7f73d41a2f44250d4779881525aea32e'

    def __init__(self, data_dir, mode, transform=None, download=False):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.download = download

        if self.download:
            self.root = self.data_dir.replace('Stanford_Online_Products', '')
            self.download_dataset()

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

    def download_dataset(self):
        download_url(self.url, self.root, filename=self.filename, md5=self.md5)
        with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
            zip_ref.extractall(self.root, members=lib.extract_progress(zip_ref))