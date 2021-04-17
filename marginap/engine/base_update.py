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
    scores = torch.mm(di, di.t())
    label_matrix = lib.create_label_matrix(batch["label"]).cuda()

    logs = {}
    losses = []
    for crit, weight in criterion:
        if hasattr(crit, 'takes_embeddings'):
            loss = crit(di, batch["label"].view(-1).cuda())
        else:
            loss = crit(scores, label_matrix)

        logs[crit.__class__.__name__] = loss.item()
        losses.append(weight * loss)

    total_loss = sum(losses)
    total_loss.backward()
    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def base_update(
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
        _ = [crit.zero_grad() for crit, w in criterion]

        if scheduler is not None:
            scheduler.step()

        meter.update(logs)
        iterator.set_postfix({k: f"{v:0.4f}" for k, v in meter.avg.items()})

    return meter.avg
