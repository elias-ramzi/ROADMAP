from functools import partial

import hydra
import tune
from ray.tune.schedulers import ASHAScheduler

import do_train


@hydra.main(config_path='config', config_name='default')
def grid_search(cfg):
    search_space = {}
    for item in cfg.grid_search:
        search_space[item["key"]] = getattr(tune, item["type"])(item["values"])

    analysis = tune.run(
        partial(do_train, cfg=cfg),
        config=search_space,
        scheduler=ASHAScheduler(metric=cfg.experience.principal_metric, mode="max", grace_period=3),
    )
