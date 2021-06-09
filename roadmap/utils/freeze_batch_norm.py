import torch.nn as nn


def freeze_batch_norm(model):
    for module in filter(lambda m: type(m) == nn.BatchNorm2d, model.modules()):
        module.eval()
        module.train = lambda _: None
    return model
