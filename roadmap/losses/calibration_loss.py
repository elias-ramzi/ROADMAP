import torch
from pytorch_metric_learning import losses, distances
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class CalibrationLoss(losses.ContrastiveLoss):
    takes_embeddings = True

    def get_default_distance(self):
        return distances.DotProductSimilarity()

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
