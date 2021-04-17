from pytorch_metric_learning import testers

from .accuracy_calculator import get_accuracy_calculator


def get_tester(dataset, exclude_ranks=None, batch_size=64, num_workers=16):
    return testers.GlobalEmbeddingSpaceTester(
        normalize_embeddings=False,
        data_and_label_getter=lambda batch: (batch["image"], batch["label"]),
        dataset_labels=dataset.labels,
        set_min_label_to_zero=True,
        batch_size=batch_size,
        dataloader_num_workers=num_workers,
        accuracy_calculator=get_accuracy_calculator(exclude_ranks),
        data_device='cuda',
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
    )
