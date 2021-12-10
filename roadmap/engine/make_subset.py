from copy import deepcopy


def make_subset(dts, idxs, transform=None, mode=None):
    dts = deepcopy(dts)

    dts.paths = [dts.paths[x] for x in idxs]
    dts.labels = [dts.labels[x] for x in idxs]

    if hasattr(dts, 'super_labels') and dts.super_labels is not None:
        dts.super_labels = [dts.super_labels[x] for x in idxs]

    dts.get_instance_dict()
    dts.get_super_dict()

    if transform is not None:
        dts.transform = transform

    if mode is not None:
        dts.mode = mode

    return dts
