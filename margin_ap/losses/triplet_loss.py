from pytorch_metric_learning import losses, distances


class TripletLoss(losses.TripletMarginLoss):
    takes_embeddings = True

    def get_default_distance(self):
        return distances.DotProductSimilarity()
