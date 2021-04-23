from collections import deque

import torch
import torch.nn as nn


def get_mask(lst):
    return [i == lst.index(x) for i, x in enumerate(lst)]


class XBM(nn.Module):

    def __init__(self, size, weight=1.0, activate_after=-1, unique=True):
        super().__init__()
        self.size = tuple(size)
        self.unique = unique
        self.weight = weight
        self.activate_after = activate_after

        self.features_memory = nn.Parameter(torch.zeros(self.size, dtype=torch.float), requires_grad=False)
        self.labels_memory = nn.Parameter(torch.zeros((self.size[0], 1), dtype=torch.float), requires_grad=False)

        if self.unique:
            self.index = {'__init__': -1}
        else:
            self.index = deque()

    def add_without_keys(self, features, labels):
        bs = features.size(0)
        while len(self.index) + bs >= self.size[0]:
            self.index.popleft()

        available = list(set(range(self.size[0])) - set(self.index))
        self.index.extend(available[:bs])

    def add_with_keys(self, features, labels, keys):
        mask = get_mask(keys)
        keys = set(keys)
        for k in (keys - set(self.index.keys())):
            self.index[k] = max(self.index.values()) + 1

        indexes = [self.index[k] for k in keys]
        self.features_memory[indexes] = features[mask]
        self.labels_memory[indexes] = labels[mask].view(-1, 1).float()

    def get_occupied_storage(self,):
        if self.unique:
            occupied = list(self.index.values())
            occupied.remove(-1)
        else:
            occupied = list(self.index)

        return self.features_memory[occupied], self.labels_memory[occupied]

    def forward(self, features, labels, keys=None):
        if self.unique:
            assert keys is not None
            self.add_with_keys(features, labels, keys)
        else:
            self.add_without_keys(features, labels)

        return self.get_occupied_storage()

    def extra_repr(self,):
        return f"size={self.size}, unique={self.unique}"


if __name__ == '__main__':
    mem = XBM((300, 128), False)

    mem(torch.zeros(32, 128), torch.zeros(32,), torch.arange(32, 64))
    mem(torch.zeros(32, 128), torch.zeros(32,), torch.arange(32))

    print(mem.index)
