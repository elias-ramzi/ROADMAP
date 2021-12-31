import os
from os.path import join
import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import roadmap.utils as lib
import roadmap.engine as eng
from roadmap.getter import Getter


def run(config, base_config=None, checkpoint_dir=None, splits=None):
    """
    creates all objects required to launch a training
    """
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
        datefmt='%m/%d/%Y %I:%M:%S',
        level=logging.INFO,
    )

    if checkpoint_dir is None:
        state = None
        restore_epoch = 0
    else:
        lib.LOGGER.info(f"Resuming from state : {checkpoint_dir}")
        state = torch.load(checkpoint_dir, map_location='cpu')
        restore_epoch = state['epoch']

    if log_dir is None:
        from ray import tune
        writer = SummaryWriter(join(tune.get_trial_dir(), "logs"), purge_step=restore_epoch)
    else:
        writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    lib.LOGGER.info(f"Training with seed {config.experience.seed}")
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
    if config.experience.split is not None:
        assert isinstance(config.experience.split, int)
        dts = getter.get_dataset(None, 'all', config.dataset)
        splits = eng.get_splits(
            dts.labels, dts.super_labels,
            config.experience.kfold,
            random_state=config.experience.split_random_state,
            with_super_labels=config.experience.with_super_labels)
        train_dts = eng.make_subset(dts, splits[config.experience.split]['train'], train_transform, 'train')
        test_dts = eng.make_subset(dts, splits[config.experience.split]['val'], test_transform, 'test')
        val_dts = None
        lib.LOGGER.info(train_dts)
        lib.LOGGER.info(test_dts)
    else:
        train_dts = getter.get_dataset(train_transform, 'train', config.dataset)
        test_dts = getter.get_dataset(test_transform, 'test', config.dataset)
        val_dts = None

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
        lib.LOGGER.info(optimizer)

    if checkpoint_dir:
        for key, schs in scheduler.items():
            for sch, sch_state in zip(schs, state[f'scheduler_{key}_state']):
                sch.load_state_dict(sch_state)

    # """""""""""""""""" Create Criterion """"""""""""""""""""""""""
    criterion = getter.get_loss(config.loss)

    # """""""""""""""""" Create Memory """"""""""""""""""""""""""
    memory = None
    if config.memory.name is not None:
        lib.LOGGER.info("Using cross batch memory")
        memory = getter.get_memory(config.memory)
        memory.cuda()

    # """""""""""""""""" Handle Cuda """"""""""""""""""""""""""
    if torch.cuda.device_count() > 1:
        lib.LOGGER.info("Model is parallelized")
        net = nn.DataParallel(net)

    net.cuda()
    _ = [crit.cuda() for crit, _ in criterion]

    # """""""""""""""""" Handle RANDOM_STATE """"""""""""""""""""""""""
    if state is not None:
        # set random NumPy and Torch random states
        lib.set_random_state(state)

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
