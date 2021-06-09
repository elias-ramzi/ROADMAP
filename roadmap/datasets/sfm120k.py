import os
import pickle

from .base_dataset import BaseDataset


def cid2filename(cid, prefix):
    """
    https://github.com/filipradenovic/cnnimageretrieval-pytorch

    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    """
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


class SFM120kDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform):
        with open(os.path.join(data_dir, "retrieval-SfM-120k.pkl"), "rb") as f:
            db = pickle.load(f)[mode]

        self.paths = [cid2filename(cid, data_dir) for cid in db['cids']]
        self.labels = db["cluster"]

        self.instance_dict = {cl: [] for cl in set(self.labels)}
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)
