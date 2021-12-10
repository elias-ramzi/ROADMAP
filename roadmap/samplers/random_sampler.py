import numpy as np
from torch.utils.data import BatchSampler

import roadmap.utils as lib


class RandomSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
    ):
        self.batch_size = batch_size

        self.length = len(dataset)
        self.reshuffle()

    def __iter__(self,):
        self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return f"{self.__class__.__name__}(batch_size={self.batch_size})"

    def reshuffle(self):
        lib.LOGGER.info("Shuffling data")
        idxs = list(range(self.length))
        np.random.shuffle(idxs)
        self.batches = []
        for i in range(self.length // self.batch_size):
            self.batches.append(idxs[i*self.batch_size:(i+1)*self.batch_size])
