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
):
    state_dict = {}
    if torch.cuda.device_count() > 1:
        state_dict["net_state"] = net.module.state_dict()
    else:
        state_dict["net_state"] = net.state_dict()
    state_dict["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state"] = scheduler.state_dict()
    state_dict["epoch"] = epoch
    state_dict["seed"] = seed
    state_dict["config"] = args

    torch.save(state_dict, join(log_dir, "weights", "rolling.ckpt"))
    if save_checkpoint:
        logging.info(f"Checkpoint of epoch {epoch} created")
        torch.save(state_dict, join(log_dir, "weights", f"epoch_{epoch}.ckpt"))
