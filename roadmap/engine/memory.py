from collections import deque

import torch
import torch.nn as nn


def get_mask(lst):
    return [i == lst.index(x) for i, x in enumerate(lst)]


class XBM(nn.Module):

    def __init__(self, size=None, weight=1.0, activate_after=-1, unique=True):
        super().__init__()
        self.size = size
        self.unique = unique
        self.weight = weight
        self.activate_after = activate_after

        if self.unique:
            self.features_memory = {}
            self.labels_memory = {}
        else:
            self.features_memory = deque()
            self.labels_memory = deque()

    def add_without_keys(self, features, labels):
        bs = features.size(0)
        while len(self.features_memory) + bs > self.size:
            self.features_memory.popleft()
            self.labels_memory.popleft()

        for feat, lb in zip(features, labels):
            self.features_memory.append(feat)
            self.labels_memory.append(lb)

    def add_with_keys(self, features, labels, keys):
        for k, feat, lb in zip(keys, features, labels):
            self.features_memory[k] = feat
            self.labels_memory[k] = lb

    def get_occupied_storage(self,):
        if not self.features_memory:
            return torch.tensor([]), torch.tensor([])

        if self.unique:
            return torch.stack(list(self.features_memory.values())), torch.stack(list(self.labels_memory.values()))

        return torch.stack(list(self.features_memory)), torch.stack(list(self.labels_memory))

    def forward(self, features, labels, keys=None):

        if self.unique:
            assert keys is not None
            self.add_with_keys(features, labels, keys)
        else:
            self.add_without_keys(features, labels)

        mem_features, mem_labels = self.get_occupied_storage()
        return mem_features, mem_labels

    def extra_repr(self,):
        return f"size={self.size}, unique={self.unique}"


if __name__ == '__main__':
    mem = XBM((56, 128), unique=False)

    mem(torch.ones(32, 128), torch.ones(32,), torch.arange(32, 64))
    mem(torch.ones(32, 128), torch.ones(32,), torch.arange(32))

    # mem(torch.ones(32, 128), torch.ones(32,))
    # features, labels = mem(torch.ones(32, 128), torch.ones(32,))
    print(mem.index)
