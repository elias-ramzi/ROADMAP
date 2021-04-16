from pytorch_metric_learning.utils import accuracy_calculator


class CustomCalculator(accuracy_calculator.AccuracyCalculator):

    def recall_at_k(self, knn_labels, query_labels, k):
        recall = (query_labels == knn_labels[:, :k])
        return recall.any(1).mean()

    def calculate_recall_at_1(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 1)

    def calculate_recall_at_4(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(knn_labels, query_labels[:, None], 4)

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


def get_accuracy_calculator(exclude_ranks=None):
    exclude = ['NMI', 'AMI']
    if exclude_ranks:
        for r in exclude_ranks:
            exclude.append(f'recall_at_{r}')
    return CustomCalculator(exclude=exclude_ranks)
