import logging
from time import time

import torch
from torch.utils.data import DataLoader

import utils as lib
from .base_update import base_update
from .evaluate import evaluate
from .checkpoint import checkpoint


def train(
    cfg,
    log_dir,
    net,
    criterion,
    optimizer,
    scheduler,
    memory,
    train_dts,
    test_dts,
    sampler,
    tester,
    writer,
    restore_epoch,
):
    # """""""""""""""""" Iter over epochs """"""""""""""""""""""""""
    logging.info(f"Training of model {cfg.experience.experiment_name}")
    best_score = 0.
    best_model = None

    metrics = None
    for e in range(1 + restore_epoch, cfg.experience.max_iter + 1):
        metrics = None

        logging.info(f"Training : @epoch #{e} for model {cfg.experience.experiment_name}")
        start_time = time()

        # """""""""""""""""" Training Loop """"""""""""""""""""""""""
        loader = DataLoader(
            train_dts,
            batch_sampler=sampler,
            num_workers=cfg.experience.num_workers,
            pin_memory=cfg.experience.pin_memory,
        )
        logs = base_update(
            net=net,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=e,
            memory=memory,
        )

        for sch in scheduler["on_epoch"]:
            scheduler.step()

        end_train_time = time()

        # """""""""""""""""" Evaluate Model """"""""""""""""""""""""""
        score = None
        if (e % cfg.experience.val_freq == 0) or (e == cfg.experience.max_iter):
            logging.info(f"Evaluation : @epoch #{e} for model {cfg.experience.experiment_name}")
            torch.cuda.empty_cache()
            metrics = evaluate(
                test_dts,
                net,
                epoch=e,
                tester=tester,
                batch_size=cfg.experience.val_bs,
                num_workers=cfg.experience.num_workers,
            )
            torch.cuda.empty_cache()
            score = metrics[cfg.experience.principal_metric]
            if score > best_score:
                best_model = "epoch_{e}"
                best_score = score

            for sch, key in scheduler["on_val"]:
                scheduler.step(metrics[key])

        # """""""""""""""""" Checkpointing """"""""""""""""""""""""""
        checkpoint(
            log_dir=log_dir,
            save_checkpoint=(e % cfg.experience.val_freq == 0),
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=e,
            seed=cfg.experience.seed,
            args=cfg,
            writer=writer,
            score=score,
            best_model=best_model,
            best_score=best_score,
        )

        # """""""""""""""""" Logging Step """"""""""""""""""""""""""
        for grp, opt in optimizer.items():
            writer.add_scalar(f"LR/{grp}", list(lib.get_lr(optimizer).values())[0], e)

        for k, v in logs.items():
            logging.info(f"{k} : {v:.4f}")
            writer.add_scalar(f"Train/{k}", v, e)

        if metrics is not None:
            for k, v in metrics['test'].items():
                if k == 'epoch':
                    continue
                logging.info(f"{k} : {v:.4f}")
                writer.add_scalar(f"Evaluation/{k}", v, e)

        end_time = time()

        elapsed_time = lib.format_time(end_time - start_time)
        elapsed_time_train = lib.format_time(end_train_time - start_time)
        elapsed_time_eval = lib.format_time(end_time - end_train_time)

        logging.info(f"Epoch took : {elapsed_time}")
        logging.info(f"Training loop took : {elapsed_time_train}")
        if metrics is not None:
            logging.info(f"Evaluation step took : {elapsed_time_eval}")

        print()
        print()

    return metrics
