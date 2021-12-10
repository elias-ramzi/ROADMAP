import math

import torch
import pytorch_metric_learning.utils.common_functions as c_f
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
    get_label_match_counts,
    get_lone_query_labels,
    try_getting_not_lone_labels,
    precision_at_k,
    mean_average_precision,
    mean_average_precision_at_r,
    r_precision,
)

import roadmap.utils as lib
from .get_knn import get_knn


EQUALITY = torch.eq


def add_axis(tens):
    if tens.ndim == 1:
        return tens[:, None]
    return tens


def maybe_index(tens, label_hierarchy_level=0, index=False):
    if not index:
        return tens

    if tens.ndim == 2:
        return tens[:, label_hierarchy_level]

    return tens[:, :, label_hierarchy_level]


def normalized_discounted_cumulative_gain(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    hierarchical_label_counts,
    label_comparison_fn,
    relevance_mask=None,
    at_r=False,
):
    device = gt_labels.device
    num_samples, num_k = knn_labels.shape[:2]

    # relevance_mask = (
    #     torch.ones((num_samples, num_k), dtype=torch.bool, device=device)
    #     if relevance_mask is None
    #     else relevance_mask
    # )

    hierarchy_level = gt_labels.size(-1)
    similarity_level = torch.zeros((num_samples, num_k), device=device, dtype=torch.float)
    for L in range(hierarchy_level):
        similarity_level += (knn_labels[:, :, L] == gt_labels[:, L:L+1]).float()

    k_idx = torch.arange(1, num_k + 1, device=device).repeat(num_samples, 1)
    dcg = ((2**similarity_level - 1) / (torch.log2(k_idx + 1))).sum(-1)

    if at_r:
        # We compute the IDCG on the whole dataset
        max_similarity_level = torch.zeros((num_samples, hierarchy_level), device=device, dtype=torch.long)
        for L in range(hierarchy_level-1, -1, -1):
            mask = gt_labels[:, L].unsqueeze(1) == hierarchical_label_counts[0][:, L]
            max_similarity_level[:, L] = (hierarchical_label_counts[1].unsqueeze(0) * mask).sum(-1)
            if L != hierarchy_level-1:
                max_similarity_level[:, L+1] -= max_similarity_level[:, L]

        if embeddings_come_from_same_source:
            max_similarity_level[:, 0] -= 1

        idcg = torch.zeros(num_samples, device=device, dtype=torch.float)
        for i, row in enumerate(max_similarity_level):
            rank = 0
            _idcg = 0.
            tmp = []
            for L in range(hierarchy_level):
                tmp += [i for i in range(rank+1, row[L]+rank+1)]
                _idcg += sum([(2**(L+1)-1) / math.log2(i+1) for i in range(rank+1, row[L]+rank+1)])
                rank += row[L]

            idcg[i] = _idcg
    else:
        # iDCG on the retrieved instances
        max_similarity_level = similarity_level.sort(-1, True)[0].float()
        idcg = ((2**max_similarity_level - 1) / (torch.log2(k_idx + 1))).sum(-1)

    return (dcg / idcg).mean().item()


class CustomCalculator(AccuracyCalculator):

    def __init__(
        self,
        *args,
        multi_level_labels=False,
        label_hierarchy_level=0,
        with_faiss=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.multi_level_labels = multi_level_labels
        self.label_hierarchy_level = label_hierarchy_level
        self.with_faiss = with_faiss

    def calculate_precision_at_1(
        self, knn_labels, query_labels, not_lone_query_mask, label_comparison_fn, **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return precision_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            1,
            self.avg_of_avgs,
            label_comparison_fn,
        )

    def calculate_mean_average_precision_at_r(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        label_comparison_fn,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return mean_average_precision_at_r(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            embeddings_come_from_same_source,
            label_counts,
            self.avg_of_avgs,
            label_comparison_fn,
        )

    def calculate_mean_average_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_comparison_fn,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0

        return mean_average_precision(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            embeddings_come_from_same_source,
            self.avg_of_avgs,
            label_comparison_fn,
        )

    def calculate_r_precision(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        embeddings_come_from_same_source,
        label_counts,
        label_comparison_fn,
        **kwargs
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return 0
        return r_precision(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            embeddings_come_from_same_source,
            label_counts,
            self.avg_of_avgs,
            label_comparison_fn,
        )

    def recall_at_k(self, knn_labels, query_labels, k, label_comparison_fn):
        recall = label_comparison_fn(query_labels, knn_labels[:, :k])
        return recall.any(1).float().mean().item()

    def calculate_recall_at_1(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            1,
            label_comparison_fn
        )

    def calculate_recall_at_2(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            2,
            label_comparison_fn
        )

    def calculate_recall_at_4(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            4,
            label_comparison_fn
        )

    def calculate_recall_at_8(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            8,
            label_comparison_fn
        )

    def calculate_recall_at_10(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            10,
            label_comparison_fn
        )

    def calculate_recall_at_16(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            16,
            label_comparison_fn
        )

    def calculate_recall_at_20(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            20,
            label_comparison_fn
        )

    def calculate_recall_at_30(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            30,
            label_comparison_fn
        )

    def calculate_recall_at_32(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            32,
            label_comparison_fn
        )

    def calculate_recall_at_100(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            100,
            label_comparison_fn
        )

    def calculate_recall_at_1000(self, knn_labels, query_labels, label_comparison_fn, **kwargs):
        return self.recall_at_k(
            maybe_index(knn_labels, self.label_hierarchy_level, self.multi_level_labels),
            add_axis(maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels)),
            1000,
            label_comparison_fn
        )

    def calculate_NDCG(
        self,
        knn_labels,
        query_labels,
        embeddings_come_from_same_source,
        hierarchical_label_counts,
        label_comparison_fn,
        at_r=False,
        **kwargs,
    ):
        return normalized_discounted_cumulative_gain(
            knn_labels,
            add_axis(query_labels),
            embeddings_come_from_same_source,
            hierarchical_label_counts,
            label_comparison_fn,
            at_r=at_r,
        )

    def calculate_NDCG_at_r(
        self,
        knn_labels,
        query_labels,
        embeddings_come_from_same_source,
        hierarchical_label_counts,
        label_comparison_fn,
        at_r=True,
        **kwargs,
    ):
        return normalized_discounted_cumulative_gain(
            knn_labels,
            add_axis(query_labels),
            embeddings_come_from_same_source,
            hierarchical_label_counts,
            label_comparison_fn,
            at_r=at_r,
        )

    def requires_knn(self):
        return super().requires_knn() + ["recall_classic"]

    def get_accuracy(
        self,
        query,
        reference,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source,
        include=(),
        exclude=(),
        return_indices=False,
    ):
        [query, reference, query_labels, reference_labels] = [
            c_f.numpy_to_torch(x)
            for x in [query, reference, query_labels, reference_labels]
        ]

        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {
            "query": query,
            "reference": reference,
            "query_labels": query_labels,
            "reference_labels": reference_labels,
            "embeddings_come_from_same_source": embeddings_come_from_same_source,
            "label_comparison_fn": self.label_comparison_fn,
        }

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts = get_label_match_counts(
                query_labels, reference_labels, self.label_comparison_fn,
            )

            if self.multi_level_labels:
                hierarchical_label_counts = (label_counts[0].detach(), label_counts[1])
                label_counts = (maybe_index(label_counts[0], self.label_hierarchy_level, self.multi_level_labels), label_counts[1])
                kwargs["hierarchical_label_counts"] = hierarchical_label_counts

            lone_query_labels, not_lone_query_mask = get_lone_query_labels(
                maybe_index(query_labels, self.label_hierarchy_level, self.multi_level_labels),
                label_counts,
                embeddings_come_from_same_source,
                self.label_comparison_fn,
            )

            num_k = self.determine_k(
                label_counts[1], len(reference), embeddings_come_from_same_source
            )

            # USE OUR OWN KNN SEARCH
            knn_indices, knn_distances = get_knn(
                reference, query, num_k, embeddings_come_from_same_source,
                with_faiss=self.with_faiss,
            )
            torch.cuda.empty_cache()

            knn_labels = reference_labels[knn_indices]
            if not any(not_lone_query_mask):
                lib.LOGGER.warning("None of the query labels are in the reference set.")
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels
            kwargs["knn_distances"] = knn_distances
            kwargs["lone_query_labels"] = lone_query_labels
            kwargs["not_lone_query_mask"] = not_lone_query_mask

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)

        if return_indices:
            # ADDED
            return knn_indices, self._get_accuracy(self.curr_function_dict, **kwargs)
        return self._get_accuracy(self.curr_function_dict, **kwargs)


def get_accuracy_calculator(
    exclude_ranks=None,
    k=2047,
    multi_level_labels=False,
    label_comparison_fn=None,
    with_AP=False,
    **kwargs,
):
    exclude = kwargs.pop('exclude', [])
    if with_AP:
        exclude.extend(['NMI', 'AMI'])
    else:
        exclude.extend(['NMI', 'AMI', 'mean_average_precision'])

    if exclude_ranks:
        for r in exclude_ranks:
            exclude.append(f'recall_at_{r}')

    if not multi_level_labels:
        exclude.extend(['NDCG', 'NDCG_at_r'])

    return CustomCalculator(
        exclude=exclude,
        k=k,
        multi_level_labels=multi_level_labels,
        label_comparison_fn=label_comparison_fn,
        **kwargs,
    )
