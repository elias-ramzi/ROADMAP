def get_lr(optimizer):
    all_lr = {}
    for i, param_group in enumerate(optimizer.param_groups):
        all_lr[i] = param_group['lr']
    return all_lr
