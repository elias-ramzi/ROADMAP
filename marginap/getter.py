import logging

from torch import optim
import torchvision.transforms as transforms

import losses
import samplers
import datasets
import models
import engine


class Getter:

    def get(self, obj, *args, **kwargs):
        return getattr(self, f"get_{obj}")(*args, **kwargs)

    def get_transform(self, config):
        t_list = []
        for k, v in config.items():
            t_list.append(getattr(transforms, k)(**v))

        transform = transforms.Compose(t_list)
        logging.info(transform)
        return transform

    def get_optimizer(self, net, config):
        optimizers = {}
        schedulers = {
            "on_epoch": [],
            "on_step": [],
            "on_val": [],
        }
        for opt in config:
            optimizer = getattr(optim, opt.name)
            optimizer = optimizers(getattr(net, opt.params), **opt.kwargs)
            optimizers[opt.params] = optimizer
            logging.info(optimizer)
            if opt.scheduler_on_epoch is not None:
                schedulers["on_epoch"].append(self.get_scheduler(optimizer, opt.scheduler_on_epoch))
            if opt.scheduler_on_step is not None:
                schedulers["on_step"].append(self.get_scheduler(optimizer, opt.scheduler_on_step))
            if opt.scheduler_on_val is not None:
                schedulers["on_val"].append(
                    (self.get_scheduler(optimizer, opt.scheduler_on_val), opt.scheduler_on_val.key)
                )

        return optimizers, schedulers

    def get_scheduler(self, opt, config):
        sch = getattr(optim.lr_scheduler, config.name)(opt, **config.kwargs)
        logging.info(sch)
        return sch

    def get_loss(self, config):
        criterion = []
        for crit in config:
            loss = getattr(losses, crit.name)(**crit.kwargs)
            weight = crit.weight
            logging.info(f"{loss} with weight {weight}")
            criterion.append((loss, weight))
        return criterion

    def get_sampler(self, dataset, config):
        sampler = getattr(samplers, config.name)(dataset, **config.kwargs)
        logging.info(sampler)
        return sampler

    def get_dataset(self, transform, mode, config):
        dataset = getattr(datasets, config.name)(
            transform=transform,
            mode=mode,
            **config.kwargs,
        )
        logging.info(dataset)
        return dataset

    def get_model(self, config):
        net = getattr(models, config.name)(**config.kwargs)
        return net

    def get_memory(self, config):
        memory = getattr(engine, config.name)(**config.kwargs)
        logging.info(memory)
        return memory
