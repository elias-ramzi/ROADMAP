import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFilter


class BaseDataset(Dataset):

    def __init__(
        self,
        multi_crop=False,
        size_crops=[224, 96],
        nmb_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1., 0.14],
        size_dataset=-1,
        return_label='none',
    ):
        super().__init__()

        if not multi_crop:
            self.get_fn = self.simple_get
        else:
            # adapted from
            # https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
            self.get_fn = self.multiple_crop_get

            self.return_label = return_label
            assert self.return_label in ["none", "real", "hash"]

            color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
            mean = [0.485, 0.456, 0.406]
            std = [0.228, 0.224, 0.225]
            trans = []
            for i in range(len(size_crops)):
                randomresizedcrop = transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i]),
                )
                trans.extend([transforms.Compose([
                    randomresizedcrop,
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Compose(color_transform),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
                ] * nmb_crops[i])
            self.trans = trans

    def __len__(self,):
        return len(self.paths)

    @property
    def my_at_R(self,):
        if not hasattr(self, '_at_R'):
            self._at_R = max(Counter(self.labels).values())
        return self._at_R

    def get_instance_dict(self,):
        self.instance_dict = {cl: [] for cl in set(self.labels)}
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)

    def get_super_dict(self,):
        if hasattr(self, 'super_labels') and self.super_labels is not None:
            self.super_dict = {ct: {} for ct in set(self.super_labels)}
            for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
                try:
                    self.super_dict[ct][cl].append(idx)
                except KeyError:
                    self.super_dict[ct][cl] = [idx]

    def simple_get(self, idx):
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        label = torch.tensor([label])
        out = {"image": img, "label": label, "path": pth}

        if hasattr(self, 'super_labels') and self.super_labels is not None:
            super_label = self.super_labels[idx]
            super_label = torch.tensor([super_label])
            out['super_label'] = super_label

        return out

    def multiple_crop_get(self, idx):
        pth = self.paths[idx]
        image = Image.open(pth).convert('RGB')
        multi_crops = list(map(lambda trans: trans(image), self.trans))

        if self.return_label == 'real':
            label = self.labels[idx]
            labels = [label] * len(multi_crops)
            return {"image": multi_crops, "label": labels, "path": pth}

        if self.return_label == 'hash':
            label = abs(hash(pth))
            labels = [label] * len(multi_crops)
            return {"image": multi_crops, "label": labels, "path": pth}

        return {"image": multi_crops, "path": pth}

    def __getitem__(self, idx):
        return self.get_fn(idx)

    def __repr__(self,):
        return f"{self.__class__.__name__}(mode={self.mode}, len={len(self)})"


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
