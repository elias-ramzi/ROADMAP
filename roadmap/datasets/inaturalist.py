import json
from os.path import join

from .base_dataset import BaseDataset


class INaturalistDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        for splt in mode:
            with open(join(self.data_dir, f'Inat_dataset_splits/Inaturalist_{splt}_set1.txt')) as f:
                paths = f.read().split("\n")
                paths.remove("")
            self.paths.extend([join(self.data_dir, pth) for pth in paths])

        with open(join(self.data_dir, 'train2018.json')) as f:
            db = json.load(f)['categories']
            self.db = {}
            for x in db:
                _ = x.pop("name")
                id_ = x.pop("id")
                x["species"] = id_
                self.db[id_] = x

        self.labels_name = [int(x.split("/")[-2]) for x in self.paths]
        self.labels_to_id = {cl: i for i, cl in enumerate(sorted(set(self.labels_name)))}
        self.labels = [self.labels_to_id[x] for x in self.labels_name]

        self.hierarchy_name = {}
        for x in self.labels_name:
            for key, val in self.db[x].items():
                try:
                    self.hierarchy_name[key].append(val)
                except KeyError:
                    self.hierarchy_name[key] = [val]

        self.hierarchy_name_to_id = {}
        self.hierarchy_labels = {}
        for key, lst in self.hierarchy_name.items():
            self.hierarchy_name_to_id[key] = {cl: i for i, cl in enumerate(sorted(set(lst)))}
            self.hierarchy_labels[key] = [self.hierarchy_name_to_id[key][x] for x in lst]

        self.super_labels_name = [x.split("/")[-3] for x in self.paths]
        self.super_labels_to_id = {scl: i for i, scl in enumerate(sorted(set(self.super_labels_name)))}
        self.super_labels = [self.super_labels_to_id[x] for x in self.super_labels_name]

        self.get_instance_dict()
        self.get_super_dict()
