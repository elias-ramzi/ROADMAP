"""
adapted from :
https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/datasets.py
"""
import copy

import numpy as np


def flatten(list_):
    return [item for sublist in list_ for item in sublist]


class MPerClassSampler:
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class=4,
    ):
        """
        Args:
            image_dict: two-level dict, `super_dict[super_class_id][class_id]` gives the list of
                        image idxs having the same super-label and class label
        """
        assert samples_per_class > 1
        self.image_dict = dataset.instance_dict.copy()
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        self.reshuffle()

    def __iter__(self,):
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def reshuffle(self):
        image_dict = copy.deepcopy(self.image_dict)
        for sub in image_dict:
            np.random.shuffle(image_dict[sub])

        classes = [*image_dict]
        np.random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (len(batch) < self.batch_size/self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:]

            if len(batch) == self.batch_size/self.samples_per_class:
                batch = flatten(batch)
                np.random.shuffle(batch)
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        np.random.shuffle(total_batches)
        self.batches = total_batches
