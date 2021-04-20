import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):

    def __len__(self,):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        label = torch.tensor([label])
        out = {"image": img, "label": label, "path": pth}

        if hasattr(self, 'super_labels'):
            super_label = self.super_labels[idx]
            super_label = torch.tensor([super_label])
            out['super_label'] = super_label

        return out
