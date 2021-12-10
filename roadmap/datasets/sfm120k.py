from os.path import join

from scipy.io import loadmat

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
    return join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


class SfM120kDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        db = loadmat(join(self.data_dir, "retrieval-SfM-120k-imagenames-clusterids.mat"))

        cids = [x[0] for x in db['cids'][0]]
        self.paths = [cid2filename(x, join(self.data_dir, "ims")) for x in cids]
        self.labels = [int(x) for x in db['cluster'][0]]

        self.get_instance_dict()
