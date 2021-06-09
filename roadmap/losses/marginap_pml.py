import torch

from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.distances import DotProductSimilarity


def tau_sigmoid(tensor, temp):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it
    through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = 1. + exponent.clamp(-50, 50).exp()
    return 1.0 / exponent


def heaviside_approx(tensor, temp=None, tau=None, offset=None, change_at=None, type="sigmo_sigmo"):
    h_neg, h_pos = type.split("_")
    pos_mask = tensor > 0
    neg_mask = ~pos_mask

    if h_neg == 'sigmo':
        tensor[neg_mask] = tau_sigmoid(tensor[neg_mask], temp)
    elif h_neg == 'linear':
        tensor[neg_mask] = torch.nn.relu(tensor[neg_mask] * temp + offset)

    if h_pos == 'linear':
        tensor[pos_mask] = tau * tensor[pos_mask] + offset

    elif h_pos == 'sigmo':
        if change_at is None:
            tensor[pos_mask] = tau_sigmoid(tensor[pos_mask], temp)

        else:
            change_mask = tensor > change_at
            tensor[pos_mask & ~change_mask] = tau_sigmoid(tensor[pos_mask & ~change_mask], temp)
            tensor[pos_mask & ~change_mask] = tau * tensor[pos_mask & ~change_mask] + offset

    return tensor


class MarginAP(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin=0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple):
        # indices_tuple = lmu.convert_to_triplets(
        #     indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        # )

        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(labels=labels)
        if len(anchor_idx) == 0:
            return self.zero_losses()

        mat = self.distance(embeddings)

        map = []
        for anchor in set(anchor_idx.view(-1).tolist()):
            all_pos = list(set(positive_idx[anchor_idx == anchor].view(-1).tolist()))
            pos_scores = mat[anchor, all_pos]
            pos_rank = pos_scores.argsort(-1, True).argsort()
            ap = 0.
            for i, p in enumerate(all_pos):
                # import ipdb; ipdb.set_trace()
                ap_dists = mat[anchor][p]
                an_dists = mat[anchor][negative_idx[(anchor_idx == anchor) & (positive_idx == p)]]
                current_margins = self.distance.margin(ap_dists, an_dists)
                ranks = tau_sigmoid(current_margins, self.margin)
                ap += pos_rank[i] / (pos_rank[i] + ranks.sum())

            map.append(ap / len(all_pos))

        map = torch.stack(map)
        loss = 1 - map
        return {
            "loss": {
                "losses": loss,
                "indices": torch.arange(0, loss.size(0)),
                "reduction_type": "element",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return DotProductSimilarity()

    def forward(self, embeddings, labels, ref_embeddings=None, ref_labels=None):
        if ref_embeddings is None:
            return super().forward(embeddings, labels)

        indices_tuple = self.create_indices_tuple(
            embeddings.size(0),
            embeddings,
            labels,
            ref_embeddings,
            ref_labels,
        )

        combined_embeddings = torch.cat([embeddings, ref_embeddings], dim=0)
        combined_labels = torch.cat([labels, ref_labels], dim=0)
        return super().forward(combined_embeddings, combined_labels, indices_tuple)

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        E_mem,
        L_mem,
    ):
        indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)
        return indices_tuple


if __name__ == '__main__':
    lss = MarginAP()
    labels = torch.randint(5, (100,))
    lss(torch.randn(100, 64), labels)
