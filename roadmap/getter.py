import logging

from torch import optim
import torchvision.transforms as transforms

from roadmap import losses
from roadmap import samplers
from roadmap import datasets
from roadmap import models
from roadmap import engine
from roadmap import utils as lib


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
            if opt.params is not None:
                optimizer = optimizer(getattr(net, opt.params).parameters(), **opt.kwargs)
                optimizers[opt.params] = optimizer
            else:
                optimizer = optimizer(net.parameters(), **opt.kwargs)
                optimizers["net"] = optimizer
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
        if (config.name == "InShopDataset") and (mode == "test"):
            dataset = {
                "test": getattr(datasets, config.name)(transform=transform, mode="query", **config.kwargs),
                "gallery": getattr(datasets, config.name)(transform=transform, mode="gallery", **config.kwargs),
            }
            logging.info(dataset)
            return dataset
        else:
            dataset = getattr(datasets, config.name)(
                transform=transform,
                mode=mode,
                **config.kwargs,
            )
            logging.info(dataset)
            return dataset

    def get_model(self, config):
        net = getattr(models, config.name)(**config.kwargs)
        if config.freeze_batch_norm:
            logging.info("Freezing batch norm")
            net = lib.freeze_batch_norm(net)
        if config.freeze_pos_embedding:
            logging.info("Freezing pos embeddings")
            net.backbone = lib.freeze_pos_embedding(net.backbone)
        return net

    def get_memory(self, config):
        memory = getattr(engine, config.name)(**config.kwargs)
        logging.info(memory)
        return memory
