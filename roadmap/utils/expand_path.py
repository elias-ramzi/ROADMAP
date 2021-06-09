import os


def expand_path(pth):
    pth = os.path.expandvars(pth)
    pth = os.path.expanduser(pth)
    return pth
