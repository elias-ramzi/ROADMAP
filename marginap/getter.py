import logging

from torch import optim
import torchvision.transforms as transforms

import losses
import samplers
import datasets
import models


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
        name = list(config.keys())[0]
        opt = getattr(optim, name)

        to_optim = []
        for param, kwargs in config[name].items():
            kwargs = dict(kwargs)
            kwargs.update({'params': getattr(net, param).parameters()})
            to_optim.append(kwargs)

        opt = opt(to_optim)
        logging.info(opt)
        return opt

    def get_scheduler(self, opt, config):
        name = list(config.keys())[0]
        sch = getattr(optim.lr_scheduler, name)(opt, **config[name])
        logging.info(sch)
        return sch

    def get_loss(self, config):
        criterion = []
        for name, kwargs in config.losses.items():
            loss = getattr(losses, name)(**kwargs)
            weight = config.weights.get(name, 1.)
            logging.info(f"{loss} with weight {weight}")
            criterion.append((loss, weight))
        return criterion

    def get_sampler(self, dataset, config):
        name = list(config.keys())[0]
        sampler = getattr(samplers, name)(dataset, **config[name])
        logging.info(sampler)
        return sampler

    def get_dataset(self, transform, mode, config):
        name = list(config.keys())[0]
        dataset = getattr(datasets, name)(
            transform=transform,
            mode=mode,
            **config[name],
        )
        logging.info(dataset)
        return dataset

    def get_model(self, config):
        name = list(config.keys())[0]
        net = getattr(models, name)(**config[name])
        return net
