from omegaconf.listconfig import ListConfig


def list_or_tuple(lst):
    if isinstance(lst, (ListConfig, tuple, list)):
        return True

    return False
