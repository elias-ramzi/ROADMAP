def set_attribute(cfg, key, value):
    all_key = key.split('.')
    obj = cfg
    for k in all_key[:-1]:
        try:
            obj = obj[int(k)]
        except ValueError:
            obj = getattr(obj, k)
    setattr(obj, all_key[-1], value)
    return cfg


def override_config(hyperparameters, config):
    for k, v in hyperparameters.items():
        config = set_attribute(config, k, v)

    return config
