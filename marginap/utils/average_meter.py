def _handle_types(value):
    if hasattr(value, "detach"):
        return value.detach().item()
    else:
        return value


class AverageMeter:
    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        val = _handle_types(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
