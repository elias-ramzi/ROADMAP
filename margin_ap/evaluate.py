from glob import glob

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from margin_ap.getter import Getter
import margin_ap.utils as lib
from margin_ap import losses as L
import margin_ap.engine as eng


state = torch.load(weights, map_location='cpu')
cfg = state["config"]

net = Getter().get_model(cfg.model)
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
net.cuda()

cfg.dataset.kwargs.data_dir = fix_path(cfg.dataset.kwargs.data_dir)

transform = Getter().get_transform(cfg.transform.test)
dts = Getter().get_dataset(transform, 'test', cfg.dataset)
sampler = Getter().get_sampler(dts, cfg.dataset.sampler)
loader = DataLoader(dts, batch_size=256, num_workers=16, pin_memory=True)
loader_train = DataLoader(dts, batch_sampler=sampler, num_workers=16, pin_memory=True)
