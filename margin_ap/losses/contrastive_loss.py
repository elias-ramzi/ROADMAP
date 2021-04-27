from pytorch_metric_learning import losses, distances


class ContrastiveLoss(losses.ContrastiveLoss):
    takes_embeddings = True

    def get_default_distance(self):
        return distances.DotProductSimilarity()
