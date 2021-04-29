def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
