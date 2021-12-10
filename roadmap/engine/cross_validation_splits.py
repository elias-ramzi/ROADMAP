import random

import numpy as np
from sklearn.model_selection import StratifiedKFold

import roadmap.utils as lib


@lib.get_set_random_state
def get_class_disjoint_splits(labels, kfold, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    unique_labels = list(set(labels))
    n = len(unique_labels)
    random.shuffle(unique_labels)

    classes = []
    for i in range(kfold):
        num_to_sample = n // kfold + (0 if i > n % kfold else 1)
        classes.append(set(unique_labels[:num_to_sample]))
        del unique_labels[:num_to_sample]

    indexes = [[] for _ in range(kfold)]
    for idx, cl in enumerate(labels):
        for i in range(kfold):
            if cl in classes[i]:
                indexes[i].append(idx)
                break

    splits = [{'train': [], 'val': []} for _ in range(kfold)]
    for i in range(kfold):
        tmp = indexes.copy()
        splits[i]['val'] = tmp[i]
        del tmp[i]
        splits[i]['train'] = [idx for sublist in tmp for idx in sublist]

    return splits


@lib.get_set_random_state
def get_hierarchical_class_disjoint_splits(labels, super_labels, kfold, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    unique_super_labels = sorted(set(super_labels))
    super_dict = {slb: set() for slb in unique_super_labels}
    for slb, lb in zip(super_labels, labels):
        super_dict[slb].add(lb)

    super_dict = {slb: sorted(lb) for slb, lb in super_dict.items()}
    for slb in unique_super_labels:
        _ = random.shuffle(super_dict[slb])
    classes = [[] for _ in range(kfold)]
    for slb in unique_super_labels:
        unique_labels = super_dict[slb].copy()
        n = len(unique_labels)
        for i in range(kfold):
            num_to_sample = n // kfold + (0 if i > n % kfold else 1)
            classes[i].extend(unique_labels[:num_to_sample])
            del unique_labels[:num_to_sample]

    indexes = [[] for _ in range(kfold)]
    for idx, cl in enumerate(labels):
        for i in range(kfold):
            if cl in classes[i]:
                indexes[i].append(idx)
                break

    splits = [{'train': [], 'val': []} for _ in range(kfold)]
    for i in range(kfold):
        tmp = indexes.copy()
        splits[i]['val'] = tmp[i]
        del tmp[i]
        splits[i]['train'] = [idx for sublist in tmp for idx in sublist]

    return splits


@lib.get_set_random_state
def get_closed_set_splits(labels, kfold, random_state=None):
    split_generator = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)

    splits = [{'train': [], 'val': []} for _ in range(kfold)]
    for i, (train_index, test_index) in enumerate(split_generator.split(np.zeros(len(labels)), labels)):
        splits[i]['train'] = train_index
        splits[i]['val'] = test_index

    return splits


def get_splits(labels, super_labels, kfold, random_state=None, with_super_labels=False, open_set=True):
    if open_set:
        if super_labels is None:
            return get_class_disjoint_splits(labels, kfold, random_state)
        elif with_super_labels:
            return get_class_disjoint_splits(super_labels, kfold, random_state)
        else:
            return get_hierarchical_class_disjoint_splits(labels, super_labels, kfold, random_state)
    else:
        return get_closed_set_splits(labels, kfold, random_state)


if __name__ == '__main__':
    random.seed(10)
    num_superlabels = 12
    dataset_size = 15023
    num_labels_per_super_labels = 17
    num_labels = num_superlabels * num_labels_per_super_labels
    kfold = 4
    labels = [random.randint(0, num_labels - 1) for _ in range(dataset_size)]
    splits = get_class_disjoint_splits(labels, kfold)

    for spl in splits:
        label_train = set([labels[x] for x in spl['train']])
        label_val = set([labels[x] for x in spl['val']])
        assert (len(set(spl['train'])) + len(set(spl['val']))) == dataset_size
        assert not label_train.intersection(label_val)

    num_superlabels = 12
    dataset_size = 15023
    num_labels_per_super_labels = 17
    num_labels = num_superlabels * num_labels_per_super_labels
    labels = [random.randint(0, num_labels - 1) for _ in range(dataset_size)]
    super_labels = [0] * len(labels)
    for slb in range(num_superlabels):
        mask = []
        for idx, lb in enumerate(labels):
            if (lb >= slb * num_labels_per_super_labels) & (lb < (slb + 1) * num_labels_per_super_labels):
                mask.append(idx)

        for idx in mask:
            super_labels[idx] = slb

    assert len(set(zip(super_labels, labels))) == num_labels

    h_splits_1 = get_hierarchical_class_disjoint_splits(labels, super_labels, kfold, random_state=1)
    h_splits_1_p = get_hierarchical_class_disjoint_splits(labels, super_labels, kfold, random_state=1)
    h_splits_2 = get_hierarchical_class_disjoint_splits(labels, super_labels, kfold, random_state=2)
    # import ipdb; ipdb.set_trace()
    for spl in h_splits_1:
        label_train = set([labels[x] for x in spl['train']])
        label_val = set([labels[x] for x in spl['val']])
        assert (len(set(spl['train'])) + len(set(spl['val']))) == dataset_size
        assert not label_train.intersection(label_val)

    for spl in h_splits_2:
        label_train = set([labels[x] for x in spl['train']])
        label_val = set([labels[x] for x in spl['val']])
        assert (len(set(spl['train'])) + len(set(spl['val']))) == dataset_size
        assert not label_train.intersection(label_val)

    for spl_1, spl_1_p, spl_2 in zip(h_splits_1, h_splits_1_p, h_splits_2):
        assert set(spl_1['train']) == set(spl_1_p['train'])
        assert set(spl_1['val']) == set(spl_1_p['val'])

        assert set(spl_1['train']) != set(spl_2['train'])
        assert set(spl_1['val']) != set(spl_2['val'])
