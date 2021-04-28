from functools import partial

import hydra
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import margin_ap.utils as lib
from margin_ap import run
# from margin_ap import engine as eng
# from margin_ap.getter import Getter


@hydra.main(config_path='config', config_name='default')
def hyperparameters_search(cfg):
    # train_dts = Getter().get_dataset(None, 'train', cfg.dataset)
    # super_labels = None
    # if hasattr(train_dts, 'super_labels'):
    #     super_labels = train_dts.super_labels
    # splits = eng.get_splits(
    #     train_dts.labels,
    #     super_labels,
    #     kfold=cfg.grid_search.kfold,
    #     random_state=0
    #         )

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

    best_trial = result.get_best_trial(cfg.experience.principal_metric, "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result[cfg.experience.principal_metric]))


if __name__ == '__main__':
    hyperparameters_search()
