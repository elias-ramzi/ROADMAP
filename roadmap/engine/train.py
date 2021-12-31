import random
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import roadmap.utils as lib
from .base_update import base_update
from .evaluate import evaluate
from .landmark_evaluation import landmark_evaluation
from . import checkpoint


def train(
    config,
    log_dir,
    net,
    criterion,
    optimizer,
    scheduler,
    scaler,
    memory,
    train_dts,
    val_dts,
    test_dts,
    sampler,
    writer,
    restore_epoch,
):
    # """""""""""""""""" Iter over epochs """"""""""""""""""""""""""
    lib.LOGGER.info(f"Training of model {config.experience.experiment_name}")
    best_score = 0.
    best_model = None

    metrics = None
    for e in range(1 + restore_epoch, config.experience.max_iter + 1):

        lib.LOGGER.info(f"Training : @epoch #{e} for model {config.experience.experiment_name}")
        start_time = time()

        # """""""""""""""""" Training Loop """"""""""""""""""""""""""
        sampler.reshuffle()
        loader = DataLoader(
            train_dts,
            batch_sampler=sampler,
            num_workers=config.experience.num_workers,
            pin_memory=config.experience.pin_memory,
        )
        logs = base_update(
            config=config,
            net=net,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=e,
            memory=memory,
        )

        for sch in scheduler["on_epoch"]:
            sch.step()

        end_train_time = time()

        dataset_dict = {}
        if (config.experience.train_eval_freq > -1) and ((e % config.experience.train_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["train_dataset"] = train_dts

        if (config.experience.val_eval_freq > -1) and ((e % config.experience.val_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["val_dataset"] = val_dts

        if (config.experience.test_eval_freq > -1) and ((e % config.experience.test_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["test_dataset"] = test_dts

        metrics = None
        if dataset_dict:
            RANDOM_STATE = random.getstate()
            NP_STATE = np.random.get_state()
            TORCH_STATE = torch.random.get_rng_state()
            TORCH_CUDA_STATE = torch.cuda.get_rng_state_all()

            lib.LOGGER.info(f"Evaluation : @epoch #{e} for model {config.experience.experiment_name}")
            torch.cuda.empty_cache()
            if config.experience.landmarks:
                metrics = landmark_evaluation(
                    net=net,
                    datasets=test_dts,
                    batch_size=config.experience.eval_bs,
                    num_workers=config.experience.num_workers,
                )
            else:
                metrics = evaluate(
                    net,
                    epoch=e,
                    batch_size=config.experience.eval_bs,
                    num_workers=config.experience.num_workers,
                    with_AP=config.experience.with_AP,
                    **dataset_dict,
                )
            torch.cuda.empty_cache()

            random.setstate(RANDOM_STATE)
            np.random.set_state(NP_STATE)
            torch.random.set_rng_state(TORCH_STATE)
            torch.cuda.set_rng_state_all(TORCH_CUDA_STATE)

        # """""""""""""""""" Evaluate Model """"""""""""""""""""""""""
        score = None
        if metrics is not None:
            score = metrics[config.experience.eval_split][config.experience.principal_metric]
            if score > best_score:
                best_model = f"epoch_{e}"
                best_score = score

            if log_dir is None:
                from ray import tune
                tune.report(**metrics[config.experience.eval_split])

            for sch, key in scheduler["on_val"]:
                sch.step(metrics[config.experience.eval_split][key])

        # """""""""""""""""" Logging Step """"""""""""""""""""""""""
        for grp, opt in optimizer.items():
            writer.add_scalar(f"LR/{grp}", list(lib.get_lr(opt).values())[0], e)

        for k, v in logs.items():
            lib.LOGGER.info(f"{k} : {v:.4f}")
            writer.add_scalar(f"Train/{k}", v, e)

        if metrics is not None:
            for split, mtrc in metrics.items():
                for k, v in mtrc.items():
                    if k == 'epoch':
                        continue
                    lib.LOGGER.info(f"{split} --> {k} : {np.around(v*100, decimals=2)}")
                    writer.add_scalar(f"{split.title()}/Evaluation/{k}", v, e)
                print()

        end_time = time()

        elapsed_time = lib.format_time(end_time - start_time)
        elapsed_time_train = lib.format_time(end_train_time - start_time)
        elapsed_time_eval = lib.format_time(end_time - end_train_time)

        lib.LOGGER.info(f"Epoch took : {elapsed_time}")
        lib.LOGGER.info(f"Training loop took : {elapsed_time_train}")
        if metrics is not None:
            lib.LOGGER.info(f"Evaluation step took : {elapsed_time_eval}")

        print()
        print()

        # """""""""""""""""" Checkpointing """"""""""""""""""""""""""
        checkpoint(
            log_dir=log_dir,
            save_checkpoint=(e % config.experience.save_model == 0),
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=e,
            seed=config.experience.seed,
            args=config,
            score=score,
            best_model=best_model,
            best_score=best_score,
        )

    return metrics
