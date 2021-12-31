from os.path import join

import torch

import roadmap.utils as lib


def checkpoint(
    log_dir,
    save_checkpoint,
    net,
    optimizer,
    scheduler,
    scaler,
    epoch,
    seed,
    args,
    score,
    best_model,
    best_score,
):
    state_dict = {}
    if torch.cuda.device_count() > 1:
        state_dict["net_state"] = net.module.state_dict()
    else:
        state_dict["net_state"] = net.state_dict()

    state_dict["optimizer_state"] = {key: opt.state_dict() for key, opt in optimizer.items()}

    state_dict["scheduler_on_epoch_state"] = [sch.state_dict() for sch in scheduler["on_epoch"]]
    state_dict["scheduler_on_step_state"] = [sch.state_dict() for sch in scheduler["on_step"]]
    state_dict["scheduler_on_val_state"] = [sch.state_dict() for sch, _ in scheduler["on_val"]]

    if scaler is not None:
        state_dict["scaler_state"] = scaler.state_dict()

    state_dict["epoch"] = epoch
    state_dict["seed"] = seed
    state_dict["config"] = args
    state_dict["score"] = score
    state_dict["best_score"] = best_score
    state_dict["best_model"] = f"{best_model}.ckpt"

    RANDOM_STATE = lib.get_random_state()
    state_dict.update(RANDOM_STATE)

    if log_dir is None:
        from ray import tune
        torch.save(state_dict, join(tune.get_trial_dir(), "rolling.ckpt"))
        if save_checkpoint:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                lib.LOGGER.info(f"Checkpoint of epoch {epoch} created")
                torch.save(state_dict, join(checkpoint_dir, f"epoch_{epoch}.ckpt"))

    else:
        torch.save(state_dict, join(log_dir, 'weights', "rolling.ckpt"))
        if save_checkpoint:
            lib.LOGGER.info(f"Checkpoint of epoch {epoch} created")
            torch.save(state_dict, join(log_dir, 'weights', f"epoch_{epoch}.ckpt"))
