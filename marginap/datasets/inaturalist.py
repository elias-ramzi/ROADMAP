from os.path import join

from .base_dataset import BaseDataset


class INaturalistDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        with open(join(self.data_dir, f'Inat_dataset_splits/Inaturalist_{mode}_set1.txt')) as f:
            paths = f.read().split("\n")
            paths.remove("")
        self.paths = [join(self.data_dir, pth) for pth in paths]

        self.labels_name = [int(x.split("/")[-2]) for x in self.paths]
        self.labels_to_id = {cl: i for i, cl in enumerate(set(self.labels_name))}
        self.labels = [self.labels_to_id[x] for x in self.labels_name]

        self.super_labels_name = [x.split("/")[-3] for x in self.paths]
        self.super_labels_to_id = {scl: i for i, scl in enumerate(set(self.super_labels_name))}
        self.super_labels = [self.super_labels_to_id[x] for x in self.super_labels_name]

        self.instance_dict = {cl: [] for cl in set(self.labels)}
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)

        self.super_dict = {ct: {} for ct in set(self.super_labels)}
        for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
            try:
                self.super_dict[ct][cl].append(idx)
            except KeyError:
                self.super_dict[ct][cl] = [idx]
