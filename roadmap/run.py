import os
from os.path import join
import logging
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from ray import tune

import margin_ap.utils as lib
import margin_ap.engine as eng
from margin_ap.getter import Getter


def run(config, base_config=None, checkpoint_dir=None, splits=None):
    # """""""""""""""""" Handle Config """"""""""""""""""""""""""
    if base_config is not None:
        log_dir = None
        config = lib.override_config(
            hyperparameters=config,
            config=base_config,
        )

    else:
        log_dir = lib.expand_path(config.experience.log_dir)
        log_dir = join(log_dir, config.experience.experiment_name)
        os.makedirs(join(log_dir, 'logs'), exist_ok=True)
        os.makedirs(join(log_dir, 'weights'), exist_ok=True)

    # """""""""""""""""" Handle Logging """"""""""""""""""""""""""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    if checkpoint_dir is None:
        state = None
        restore_epoch = 0
    else:
        logging.info(f"Resuming from state : {checkpoint_dir}")
        state = torch.load(checkpoint_dir, map_location='cpu')
        restore_epoch = state['epoch']

    if log_dir is None:
        writer = SummaryWriter(join(tune.get_trial_dir(), "logs"), purge_step=restore_epoch)
    else:
        writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    logging.info(f"Training with seed {config.experience.seed}")
    random.seed(config.experience.seed)
    np.random.seed(config.experience.seed)
    torch.manual_seed(config.experience.seed)
    torch.cuda.manual_seed_all(config.experience.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    getter = Getter()

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    train_transform = getter.get_transform(config.transform.train)
    test_transform = getter.get_transform(config.transform.test)
    trainval_dts = getter.get_dataset(train_transform, 'train', config.dataset)
    if splits is not None:
        train_dts = Subset(deepcopy(trainval_dts), splits['train'])
        val_dts = Subset(deepcopy(trainval_dts), splits['val'])
        val_dts.dataset.transform = test_transform
        val_dts.dataset.mode = 'val'
        logging.info(val_dts.dataset)
    else:
        train_dts = trainval_dts
        val_dts = None

    test_dts = getter.get_dataset(test_transform, 'test', config.dataset)
    sampler = getter.get_sampler(train_dts, config.dataset.sampler)

    # """""""""""""""""" Create Network """"""""""""""""""""""""""
    net = getter.get_model(config.model)

    scaler = None
    if config.model.kwargs.with_autocast:
        scaler = torch.cuda.amp.GradScaler()
        if checkpoint_dir:
            scaler.load_state_dict(state['scaler_state'])

    if checkpoint_dir:
        net.load_state_dict(state['net_state'])
        net.cuda()

    # """""""""""""""""" Create Optimizer & Scheduler """"""""""""""""""""""""""
    optimizer, scheduler = getter.get_optimizer(net, config.optimizer)

    if checkpoint_dir:
        for key, opt in optimizer.items():
            opt.load_state_dict(state['optimizer_state'][key])

    if config.experience.force_lr is not None:
        _ = [lib.set_lr(opt, config.experience.force_lr) for opt in optimizer.values()]
        logging.info(optimizer)

    if checkpoint_dir:
        for key, schs in scheduler.items():
            for sch, sch_state in zip(schs, state[f'scheduler_{key}_state']):
                sch.load_state_dict(sch_state)

    # """""""""""""""""" Create Criterion """"""""""""""""""""""""""
    criterion = getter.get_loss(config.loss)

    # """""""""""""""""" Create Memory """"""""""""""""""""""""""
    memory = None
    if config.memory.name is not None:
        logging.info("Using cross batch memory")
        memory = getter.get_memory(config.memory)
        memory.cuda()

    # """""""""""""""""" Handle Cuda """"""""""""""""""""""""""
    if torch.cuda.device_count() > 1:
        logging.info("Model is parallelized")
        net = nn.DataParallel(net)

        if config.experience.parallel_crit:
            logging.info("Loss will be computed on multiple devices")
            criterion = [(nn.DataParallel(crit), w) for crit, w in criterion]

    net.cuda()
    _ = [crit.cuda() for crit, _ in criterion]

    # """""""""""""""""" Handle RANDOM_STATE """"""""""""""""""""""""""
    if state is not None:
        random.setstate(state["RANDOM_STATE"])
        np.random.set_state(state["NP_STATE"])
        torch.random.set_rng_state(state["TORCH_STATE"])
        torch.cuda.set_rng_state_all(state["TORCH_CUDA_STATE"])

    return eng.train(
        config=config,
        log_dir=log_dir,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        memory=memory,
        train_dts=train_dts,
        val_dts=val_dts,
        test_dts=test_dts,
        sampler=sampler,
        writer=writer,
        restore_epoch=restore_epoch,
    )


if __name__ == '__main__':
    run()