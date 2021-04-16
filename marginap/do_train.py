import os
from os.path import join
import sys
import logging
import random
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils as lib
import losses as lss
import engine as eng
from datasets import SOPDataset, INaturalistDataset
from samplers import HierarchicalSampler, MPerClassSampler
from models import RetrievalNet


def do_train(
    #  Experience
    log_dir,
    experiment_name,
    seed,
    max_iter,
    resume,
    #  Data
    data_dir,
    dataset_name,
    batch_size,
    batches_per_super_pair,
    val_bs,
    val_freq,
    num_workers,
    #  Network
    backbone_name,
    embed_dim,
    norm_features,
    without_fc,
    freeze_batch_norm,
    #  Optimizer
    optimizer,
    lr,
    wd,
    nlr,
    nwd,
    #  Scheduler
    scheduler,
    milestones,
    gamma,
    # Criterion
    criterion,
    temp,
    mu,
    tau,
):

    # """""""""""""""""" Handle Logging """"""""""""""""""""""""""
    log_dir = join(log_dir, experiment_name)
    log_dir = os.path.expandvars(log_dir)
    log_dir = os.path.expanduser(log_dir)
    if os.isdir(log_dir) and not resume:
        logging.warning(f"Existing {log_dir}, folder already exists")
        sys.exit()

    if not resume:
        restore_epoch = 0
        os.makedirs(join(log_dir, "logs"))
        os.makedirs(join(log_dir, "weights"))
    else:
        logging.info(f"Resuming from state : {resume}")
        state = torch.load(join(log_dir, resume), map_location='cpu')
        restore_epoch = state['epoch']

    writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    if not os.path.isfile(join(log_dir, "entry_point.txt")):
        command = 'python ' + ' '.join(sys.argv)
        with open(join(log_dir, "entry_point.txt"), 'w') as f:
            f.write(command)

    logging.info(f"Training with seed {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    if dataset_name == 'sop':
        data_dir = join(data_dir, 'Stanford_Online_Products')
        train_dts = SOPDataset(data_dir, 'train', eng.get_train_transform())
        test_dts = SOPDataset(data_dir, 'test', eng.get_test_transform())
        # ranks = [1, 10, 100, 1000]
        exclude_ranks = [4, 16, 32]
        sampler = HierarchicalSampler(train_dts, batch_size, 4, batches_per_super_pair)
        val_freq = val_freq if val_freq else 5
        milestones = milestones if milestones else [15, 30]
        gamma = gamma if gamma else 0.1

    if dataset_name == 'inaturalist':
        data_dir = join(data_dir, 'inaturalist')
        train_dts = INaturalistDataset(data_dir, 'train', eng.get_train_transform(with_resize=False))
        test_dts = INaturalistDataset(data_dir, 'test', eng.get_test_transform())
        # ranks = [1, 4, 16, 32]
        exclude_ranks = [10, 100, 1000]
        sampler = MPerClassSampler(train_dts, batch_size, 4)
        val_freq = val_freq if val_freq else 5
        milestones = milestones if milestones else [40, 70]
        gamma = gamma if gamma else 0.3

    def get_dataloader():
        return DataLoader(
            train_dts,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )

    tester = eng.get_tester(
        dataset=test_dts, exclude_ranks=exclude_ranks, batch_size=val_bs, num_workers=num_workers,
    )

    # """""""""""""""""" Create Network """"""""""""""""""""""""""
    net = RetrievalNet(
        backbone_name,
        embed_dim,
        norm_features,
        without_fc,
    )

    if freeze_batch_norm:
        net = lib.freeze_batch_norm(net)

    to_optim = []
    to_optim.append({'params': net.backbone.parameters(), 'lr': lr, 'weight_decay': wd})
    if hasattr(net, 'fc'):
        to_optim.append({'params': net.fc.parameters(), 'lr': nlr, 'weight_decay': nwd})

    # """""""""""""""""" Create Optimizer """"""""""""""""""""""""""
    if optimizer == 'Adam':
        optimizer = opt.Adam(to_optim)
    elif optimizer == 'AdamW':
        optimizer = opt.AdamW(to_optim)

    # """""""""""""""""" Create Scheduler """"""""""""""""""""""""""
    if scheduler == 'step':
        scheduler = opt.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    # """""""""""""""""" Create Criterion """"""""""""""""""""""""""
    if criterion == 'smoothap':
        criterion = lss.SmoothAP(temp=temp)
    elif criterion == 'marginap':
        criterion = lss.MarginAP(mu=mu, tau=tau)

    # """""""""""""""""" Handle Cuda """"""""""""""""""""""""""
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.cuda()
    criterion.cuda()

    # """""""""""""""""" Iter over epochs """"""""""""""""""""""""""
    logging.info(f"Training of model {log_dir}")

    for e in range(1 + restore_epoch, max_iter + 1 + restore_epoch):
        logs = None
        metrics = None

        logging.info(f"Training : @epoch #{e} for model {log_dir}")
        start_time = time()

        # """""""""""""""""" Training Loop """"""""""""""""""""""""""
        loader = get_dataloader()
        logs = eng.base_update(
            net=net,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            compute_similarity=True,
            create_label=True,
        )
        scheduler.step()
        end_train_time = time()

        # """""""""""""""""" Evaluate Model """"""""""""""""""""""""""
        if (e % val_freq == 0) or (e == max_iter):
            logging.info(f"Evaluation : @epoch #{e} for model {log_dir}")
            metrics = eng.evaluate(
                test_dts,
                net,
                epoch=e,
                tester=tester,
            )

        # """""""""""""""""" Checkpointing """"""""""""""""""""""""""
        eng.checkpoint(
            log_dir,
            (e % val_freq == 0),
            net,
            optimizer,
            scheduler,
            e,
            seed,
            sys.argv,
            writer,
        )

        # """""""""""""""""" Logging Step """"""""""""""""""""""""""
        for grp, v in lib.get_lr(optimizer):
            writer.add_scalar(f"LR/{grp}", v, e)

        for k, v in logs.items():
            writer.add_scalar(f"Train/{k}", v, e)

        if metrics is not None:
            for k, v in metrics.items():
                writer.add_scalar(f"Evaluation/{k}", v, e)

        end_time = time()

        elapsed_time = lib.format_time(end_time - start_time)
        elapsed_time_train = lib.format_time(end_train_time - start_time)

        logging.info(f"Epoch took : {elapsed_time}")
        logging.info(f"Training loop took : {elapsed_time_train}")

    return metrics
