import os
from os.path import join
import sys
import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import utils as lib
import engine as eng
from getter import Getter


def run(config, base_config=None):
    # """""""""""""""""" Handle Logging """"""""""""""""""""""""""
    if base_config is not None:
        config = lib.override_config(
            hyperparameters=config,
            config=base_config,
        )

    # """""""""""""""""" Handle Logging """"""""""""""""""""""""""
    log_dir = os.path.expandvars(config.experience.log_dir)
    log_dir = os.path.expanduser(log_dir)
    log_dir = join(log_dir, config.experience.experiment_name)

    if os.path.isdir(log_dir) and not config.experience.resume:
        logging.warning(f"Existing {log_dir}, folder already exists")
        return

    os.makedirs(join(log_dir, "logs"), exist_ok=True)
    os.makedirs(join(log_dir, "weights"), exist_ok=True)
    if not config.experience.resume:
        state = None
        restore_epoch = 0
    else:
        if not os.path.isfile(config.experience.resume):
            resume = join(log_dir, 'weights', config.experience.resume)
        else:
            resume = os.path.expandvars(config.experience.resume)
            resume = os.path.expanduser(resume)

        logging.info(f"Resuming from state : {resume}")
        state = torch.load(resume, map_location='cpu')
        restore_epoch = state['epoch']

    writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    if not os.path.isfile(join(log_dir, "entry_point.txt")):
        command = 'python ' + ' '.join(sys.argv)
        with open(join(log_dir, "entry_point.txt"), 'w') as f:
            f.write(command)

    logging.info(f"Training with seed {config.experience.seed}")
    torch.manual_seed(config.experience.seed)
    random.seed(config.experience.seed)
    np.random.seed(config.experience.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    getter = Getter()

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    train_transform = getter.get_transform(config.transform.train)
    test_transform = getter.get_transform(config.transform.test)
    train_dts = getter.get_dataset(train_transform, 'train', config.dataset)
    test_dts = getter.get_dataset(test_transform, 'test', config.dataset)
    sampler = getter.get_sampler(train_dts, config.dataset.sampler)

    tester = eng.get_tester(
        dataset=test_dts, exclude_ranks=None, batch_size=config.experience.val_bs, num_workers=config.experience.num_workers,
    )

    # """""""""""""""""" Create Network """"""""""""""""""""""""""
    net = getter.get_model(config.model)

    if config.experience.resume:
        net.load_state_dict(state['net_state'])
        net.cuda()

    # """""""""""""""""" Create Optimizer """"""""""""""""""""""""""
    optimizer = getter.get_optimizer(net, config.optimizer.optimizer)

    if config.experience.resume:
        for opt, opt_state in zip(optimizer, state['optimizer_state']):
            opt.load_state_dict(opt_state)

    # """""""""""""""""" Create Scheduler """"""""""""""""""""""""""
    scheduler = getter.get_scheduler(optimizer, config.optimizer.scheduler)

    if config.experience.resume:
        for key, schs in scheduler.items():
            for sch, sch_state in zip(optimizer, state[f'scheduler_{key}_state']):
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

    return eng.train(
        config=config,
        log_dir=log_dir,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        memory=memory,
        train_dts=train_dts,
        test_dts=test_dts,
        sampler=sampler,
        tester=tester,
        writer=writer,
        restore_epoch=restore_epoch,
    )


if __name__ == '__main__':
    run()
