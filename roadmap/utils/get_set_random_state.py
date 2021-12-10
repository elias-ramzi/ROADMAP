import random
from functools import wraps

import numpy as np
import torch

from .logger import LOGGER


def get_random_state():
    LOGGER.debug("Getting random state")
    RANDOM_STATE = {}
    RANDOM_STATE["RANDOM_STATE"] = random.getstate()
    RANDOM_STATE["NP_STATE"] = np.random.get_state()
    RANDOM_STATE["TORCH_STATE"] = torch.random.get_rng_state()
    RANDOM_STATE["TORCH_CUDA_STATE"] = torch.cuda.get_rng_state_all()
    return RANDOM_STATE


def set_random_state(RANDOM_STATE):
    LOGGER.debug("Setting random state")
    random.setstate(RANDOM_STATE["RANDOM_STATE"])
    np.random.set_state(RANDOM_STATE["NP_STATE"])
    torch.random.set_rng_state(RANDOM_STATE["TORCH_STATE"])
    torch.cuda.set_rng_state_all(RANDOM_STATE["TORCH_CUDA_STATE"])


def get_set_random_state(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        RANDOM_STATE = get_random_state()
        output = func(*args, **kwargs)
        set_random_state(RANDOM_STATE)
        return output
    return wrapper
