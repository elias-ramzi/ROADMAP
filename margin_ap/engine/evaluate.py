import logging

import torch
from pytorch_metric_learning import testers
import pytorch_metric_learning.utils.common_functions as c_f

from .accuracy_calculator import get_accuracy_calculator


class GlobalEmbeddingSpaceTester(testers.GlobalEmbeddingSpaceTester):

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        s, e = 0, 0
        with torch.no_grad():
            logging.info("Computing embeddings")
            for i, data in enumerate(dataloader):
                img, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, "all", self.label_mapper)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(
                        len(dataloader.dataset),
                        label.size(1),
                        device=self.data_device,
                        dtype=label.dtype,
                    )
                    all_q = torch.zeros(
                        len(dataloader.dataset),
                        q.size(1),
                        device=self.data_device,
                        dtype=q.dtype,
                    )
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
        return all_q, labels


def get_tester(dataset, exclude_ranks=None, batch_size=64, num_workers=16):
    return GlobalEmbeddingSpaceTester(
        normalize_embeddings=False,
        data_and_label_getter=lambda batch: (batch["image"], batch["label"]),
        dataset_labels=dataset.labels,
        set_min_label_to_zero=True,
        batch_size=batch_size,
        dataloader_num_workers=num_workers,
        accuracy_calculator=get_accuracy_calculator(exclude_ranks),
        data_device=None,
    )


def evaluate(
    dataset,
    net,
    epoch=None,
    tester=None,
    exclude_ranks=None,
    batch_size=64,
    num_workers=16,
):
    if tester is None:
        tester = get_tester(
            dataset,
            exclude_ranks=exclude_ranks,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return tester.test(
        {"test": dataset},
        f"{epoch}",
        net,
    )['test']
