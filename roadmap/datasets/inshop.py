import os

from .base_dataset import BaseDataset


class InShopDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, hierarchy_mode='all', **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "query", "gallery"], f"Mode : {mode} unknown"
        assert hierarchy_mode in ['1', '2', 'all'], f"Hierarchy mode : {hierarchy_mode} unknown"
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        with open(os.path.join(data_dir, "list_eval_partition.txt")) as f:
            db = f.read().split("\n")[2:-1]

        paths = []
        labels = []
        super_labels_name = []
        for line in db:
            line = line.split(" ")
            line = list(filter(lambda x: x, line))
            if line[2] == mode:
                paths.append(os.path.join(data_dir, line[0]))
                labels.append(int(line[1].split("_")[-1]))
                if hierarchy_mode == '2':
                    super_labels_name.append(line[0].split("/")[2])
                elif hierarchy_mode == '1':
                    super_labels_name.append(line[0].split("/")[1])
                elif hierarchy_mode == 'all':
                    super_labels_name.append('/'.join(line[0].split("/")[1:3]))

        self.paths = paths
        self.labels = labels

        slb_to_id = {slb: i for i, slb in enumerate(set(super_labels_name))}
        self.super_labels = [slb_to_id[slb] for slb in super_labels_name]
        self.super_labels_name = super_labels_name

        self.get_instance_dict()
        self.get_super_dict()
