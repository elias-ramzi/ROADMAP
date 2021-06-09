import os
import logging

import hydra

import roadmap.utils as lib
from roadmap import run


@hydra.main(config_path='config', config_name='default')
def single_experiment_runner(cfg):
    cfg.experience.log_dir = lib.expand_path(cfg.experience.log_dir)

    if cfg.experience.resume is not None:
        if os.path.isfile(lib.expand_path(cfg.experience.resume)):
            resume = lib.expand_path(cfg.experience.resume)
        else:
            resume = os.path.join(cfg.experience.log_dir, cfg.experience.experiment_name, 'weights', cfg.experience.resume)
    else:
        resume = None
        if os.path.isdir(os.path.join(cfg.experience.log_dir, cfg.experience.experiment_name, 'weights')):
            logging.warning(f"Exiting trial, experiment {cfg.experience.experiment_name} already exists")
            return

    metrics = run.run(
        config=cfg,
        base_config=None,
        checkpoint_dir=resume,
    )

    print(metrics)


if __name__ == '__main__':
    single_experiment_runner()
