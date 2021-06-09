from pytorch_metric_learning import losses


class FastAP(losses.FastAPLoss):
    takes_embeddings = True
