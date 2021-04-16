import torch
from tqdm import tqdm

import utils as lib


def _batch_optimization(
    net,
    batch,
    criterion,
    compute_similarity,
    create_label,
):
    di = net(batch["image"].cuda())

    if compute_similarity:
        similarities = torch.mm(di, di.t())
    else:
        similarities = di

    if create_label:
        labels = lib.create_label_matrix(batch["label"]).cuda()
    else:
        labels = batch["label"].view(-1).cuda()

    if hasattr(criterion, "paths_needed"):
        loss = criterion(similarities, labels, batch["path"])
    else:
        loss = criterion(similarities, labels)

    loss.backward()
    loss.detach_()
    return {criterion.__class__.__name__: loss.item()}


def criterion_update(
    net,
    loader,
    criterion,
    optimizer,
    scheduler=None,
    compute_similarity=True,
    create_label=True,
):
    meter = lib.DictAverage()

    iterator = tqdm(loader, desc='Running epoch on loader')
    for i, batch in enumerate(iterator):
        logs = _batch_optimization(
            net,
            batch,
            criterion,
            compute_similarity,
            create_label,
        )

        optimizer.step()
        optimizer.zero_grad()
        criterion.zero_grad()

        if scheduler is not None:
            scheduler.step()

        meter.update(logs)
        iterator.set_postfix({k: f"{v:0.4f}" for k, v in meter.avg.items()})

    return meter.avg
