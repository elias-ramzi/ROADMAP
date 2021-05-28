import logging

import torch
import pytorch_metric_learning.utils.common_functions as c_f
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator, get_label_match_counts, get_lone_query_labels

from .get_knn import get_knn


class CustomCalculator(AccuracyCalculator):

    def recall_at_k(self, knn_labels, query_labels, k):
        recall = (query_labels == knn_labels[:, :k])
        return recall.any(1).float().mean().item()

    def calculate_recall_at_1(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 1)

    def calculate_recall_at_2(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 2)

    def calculate_recall_at_4(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 4)

    def calculate_recall_at_8(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 8)

    def calculate_recall_at_16(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 16)

    def calculate_recall_at_32(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 32)

    def calculate_recall_at_10(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 10)

    def calculate_recall_at_100(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 100)

    def calculate_recall_at_1000(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 1000)

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
                query_labels, reference_labels, self.label_comparison_fn
            )
            lone_query_labels, not_lone_query_mask = get_lone_query_labels(
                query_labels,
                label_counts,
                embeddings_come_from_same_source,
                self.label_comparison_fn,
            )

            num_k = self.determine_k(
                label_counts[1], len(reference), embeddings_come_from_same_source
            )

            knn_indices, knn_distances = get_knn(
                reference, query, num_k, embeddings_come_from_same_source
            )
            torch.cuda.empty_cache()

            knn_labels = reference_labels[knn_indices]
            if not any(not_lone_query_mask):
                logging.warning("None of the query labels are in the reference set.")
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels
            kwargs["knn_distances"] = knn_distances
            kwargs["lone_query_labels"] = lone_query_labels
            kwargs["not_lone_query_mask"] = not_lone_query_mask

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)

        if return_indices:
            return knn_indices, self._get_accuracy(self.curr_function_dict, **kwargs)
        return self._get_accuracy(self.curr_function_dict, **kwargs)


def get_accuracy_calculator(exclude_ranks=None, k=2047):
    exclude = ['NMI', 'AMI']
    if exclude_ranks:
        for r in exclude_ranks:
            exclude.append(f'recall_at_{r}')
    return CustomCalculator(exclude=exclude, k=k)
