def set_initial_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
