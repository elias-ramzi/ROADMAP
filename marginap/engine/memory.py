from collections import deque

import torch
import torch.nn as nn


class XBM(nn.Module):

    def __init__(self, size, unique=True):
        super().__init__()
        self.size = size
        self.unique = unique

        self.features_memory = nn.Parameter(torch.zeros(size, dtype=torch.float), requires_grad=False)
        self.labels_memory = nn.Parameter(torch.zeros((size[0], 1), dtype=torch.float), requires_grad=False)

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
        indexes = set(keys.view(-1).tolist())
        for k in (indexes - set(self.index.keys())):
            self.index[k] = max(self.index.values()) + 1

        indexes = [self.index[k] for k in keys.view(-1).tolist()]
        self.features_memory[indexes] = features
        self.labels_memory[indexes] = labels.view(-1, 1)

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
            assert len(set(keys.view(-1).tolist())) == features.size(0)
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
