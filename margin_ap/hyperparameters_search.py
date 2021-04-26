from functools import partial

import hydra
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import utils as lib
import run


@hydra.main(config_path='config', config_name='default')
def hyperparameters_search(cfg):
    search_space = {}
    for item in cfg.grid_search.hyperparameters:
        try:
            search_space[item["key"]] = getattr(tune, item["type"])(item["values"])
        except TypeError:
            search_space[item["key"]] = getattr(tune, item["type"])(*item["values"])

    scheduler = ASHAScheduler(
        metric=cfg.experience.principal_metric,
        mode="max",
        grace_period=cfg.grid_search.grace_period,
        max_t=cfg.experience.max_iter,
    )

    result = tune.run(
        partial(run.run, base_config=cfg),
        config=search_space,
        num_samples=cfg.grid_search.num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": cfg.grid_search.num_cpus, "gpu": cfg.grid_search.num_gpus},
        local_dir=lib.expand_path(cfg.experience.log_dir),
        name=cfg.experience.experiment_name,
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == '__main__':
    hyperparameters_search()
