import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):

    def __len__(self,):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        label = self.labels[idx]
        super_label = self.super_labels[idx]

        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = torch.tensor([label])
        super_label = torch.tensor([super_label])

        out = {"image": img, "label": label, "super_label": super_label, "path": pth}
        return out
