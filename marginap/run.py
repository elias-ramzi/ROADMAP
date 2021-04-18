import os
from os.path import join
import sys
import logging
import random
import numpy as np
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra

import utils as lib
import engine as eng
from getter import Getter


@hydra.main(config_path='config', config_name='default')
def run(cfg):
    # """""""""""""""""" Handle Logging """"""""""""""""""""""""""
    log_dir = os.path.expandvars(cfg.experience.log_dir)
    log_dir = os.path.expanduser(log_dir)
    log_dir = join(log_dir, cfg.experience.experiment_name)
    if os.path.isdir(log_dir) and not cfg.experience.resume:
        logging.warning(f"Existing {log_dir}, folder already exists")
        sys.exit()

    if not cfg.experience.resume:
        restore_epoch = 0
        os.makedirs(join(log_dir, "logs"))
        os.makedirs(join(log_dir, "weights"))
    else:
        logging.info(f"Resuming from state : {cfg.experience.resume}")
        state = torch.load(join(log_dir, 'weights', cfg.experience.resume), map_location='cpu')
        restore_epoch = state['epoch']

    writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    if not os.path.isfile(join(log_dir, "entry_point.txt")):
        command = 'python ' + ' '.join(sys.argv)
        with open(join(log_dir, "entry_point.txt"), 'w') as f:
            f.write(command)

    logging.info(f"Training with seed {cfg.experience.seed}")
    torch.manual_seed(cfg.experience.seed)
    random.seed(cfg.experience.seed)
    np.random.seed(cfg.experience.seed)

    getter = Getter()

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    train_transform = getter.get_transform(cfg.transform.train)
    test_transform = getter.get_transform(cfg.transform.test)
    train_dts = getter.get_dataset(train_transform, 'train', cfg.dataset)
    test_dts = getter.get_dataset(test_transform, 'test', cfg.dataset)
    sampler = getter.get_sampler(train_dts, cfg.general.sampler)

    tester = eng.get_tester(
        dataset=test_dts, exclude_ranks=None, batch_size=cfg.general.val_bs, num_workers=cfg.general.num_workers,
    )

    # """""""""""""""""" Create Network """"""""""""""""""""""""""
    net = getter.get_model(cfg.model)

    if cfg.general.freeze_batch_norm:
        net = lib.freeze_batch_norm(net)

    if cfg.experience.resume:
        net.load_state_dict(state['net_state'])

    # """""""""""""""""" Create Optimizer """"""""""""""""""""""""""
    optimizer = getter.get_optimizer(net, cfg.optimizer.optimizer)

    if cfg.experience.resume:
        optimizer.load_state_dict(state['optimizer_state'])

    # """""""""""""""""" Create Scheduler """"""""""""""""""""""""""
    scheduler = getter.get_scheduler(optimizer, cfg.optimizer.scheduler)

    if cfg.experience.resume:
        scheduler.load_state_dict(state['scheduler_state'])

    # """""""""""""""""" Create Criterion """"""""""""""""""""""""""
    criterion = getter.get_loss(cfg.loss)

    # """""""""""""""""" Handle Cuda """"""""""""""""""""""""""
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.cuda()
    _ = [crit.cuda() for crit, _ in criterion]

    # """""""""""""""""" Iter over epochs """"""""""""""""""""""""""
    logging.info(f"Training of model {cfg.experience.experiment_name}")

    for e in range(1 + restore_epoch, cfg.general.max_iter + 1 + restore_epoch):
        logs = None
        metrics = None

        logging.info(f"Training : @epoch #{e} for model {cfg.experience.experiment_name}")
        start_time = time()

        # """""""""""""""""" Training Loop """"""""""""""""""""""""""
        loader = DataLoader(
            train_dts,
            batch_sampler=sampler,
            num_workers=cfg.general.num_workers,
            pin_memory=True
        )
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
        if (e % cfg.general.val_freq == 0) or (e == cfg.general.max_iter):
            logging.info(f"Evaluation : @epoch #{e} for model {cfg.experience.experiment_name}")
            torch.cuda.empty_cache()
            metrics = eng.evaluate(
                test_dts,
                net,
                epoch=e,
                tester=tester,
                batch_size=cfg.general.val_bs,
                num_workers=cfg.general.num_workers,
            )
            torch.cuda.empty_cache()

        # """""""""""""""""" Checkpointing """"""""""""""""""""""""""
        eng.checkpoint(
            log_dir=log_dir,
            save_checkpoint=(e % cfg.general.val_freq == 0),
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=e,
            seed=cfg.experience.seed,
            args=cfg,
            writer=writer,
        )

        # """""""""""""""""" Logging Step """"""""""""""""""""""""""
        for grp, v in lib.get_lr(optimizer).items():
            writer.add_scalar(f"LR/{grp}", v, e)

        for k, v in logs.items():
            writer.add_scalar(f"Train/{k}", v, e)

        if metrics is not None:
            for k, v in metrics['test'].items():
                if k == 'epoch':
                    continue
                writer.add_scalar(f"Evaluation/{k}", v, e)

        end_time = time()

        elapsed_time = lib.format_time(end_time - start_time)
        elapsed_time_train = lib.format_time(end_train_time - start_time)
        elapsed_time_eval = lib.format_time(end_time - end_train_time)

        logging.info(f"Epoch took : {elapsed_time}")
        logging.info(f"Training loop took : {elapsed_time_train}")
        if metrics is not None:
            logging.info(f"Training loop took : {elapsed_time_eval}")

        print()
        print()

    return metrics


if __name__ == '__main__':
    run()
