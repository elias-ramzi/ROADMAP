from .base_update import base_update
from .large_batch_update import large_batch_update


def update(*args, **kwargs):
    if args:
        conf = args[0]
    else:
        conf = kwargs["config"]

    if conf.experience.update_type == 'large_batch_update':
        return large_batch_update(*args, **kwargs)

    else:
        return base_update(*args, **kwargs)
