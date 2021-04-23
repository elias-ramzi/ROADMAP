import logging
from os.path import join

import torch


def checkpoint(
    log_dir,
    save_checkpoint,
    net,
    optimizer,
    scheduler,
    epoch,
    seed,
    args,
    writer,
    score,
    best_model,
    best_score,
):
    state_dict = {}
    if torch.cuda.device_count() > 1:
        state_dict["net_state"] = net.module.state_dict()
    else:
        state_dict["net_state"] = net.state_dict()

    state_dict["optimizer_state"] = [opt.state_dict() for opt in optimizer]

    state_dict["scheduler_on_epoch_state"] = [sch.state_dict() for sch in scheduler["on_epoch"]]
    state_dict["scheduler_on_step_state"] = [sch.state_dict() for sch in scheduler["on_step"]]
    state_dict["scheduler_on_val_state"] = [sch.state_dict() for sch in scheduler["on_val"]]

    state_dict["epoch"] = epoch
    state_dict["seed"] = seed
    state_dict["config"] = args
    state_dict["score"] = score
    state_dict["best_score"] = best_score
    state_dict["best_model"] = join(log_dir, "weights", f"{best_model}.ckpt")

    torch.save(state_dict, join(log_dir, "weights", "rolling.ckpt"))
    if save_checkpoint:
        logging.info(f"Checkpoint of epoch {epoch} created")
        torch.save(state_dict, join(log_dir, "weights", f"epoch_{epoch}.ckpt"))
