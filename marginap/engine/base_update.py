import torch
from tqdm import tqdm

import utils as lib


def _batch_optimization(
    net,
    batch,
    criterion,
    memory
):
    di = net(batch["image"].cuda())
    labels = batch["label"].cuda()
    scores = torch.mm(di, di.t())
    label_matrix = lib.create_label_matrix(labels)

    if memory:
        memory_embeddings, memory_labels = memory(di.detach(), labels, batch["path"])
        memory_scores = torch.mm(di, memory_embeddings.t())
        memory_label_matrix = lib.create_label_matrix(labels, memory_labels)

    logs = {}
    losses = []
    for crit, weight in criterion:
        if hasattr(crit, 'takes_embeddings'):
            loss = crit(di, labels)
            logs[crit.__class__.__name__] = loss.item()
            if memory:
                mem_loss = crit(memory_embeddings, memory_labels)
                logs[f"memory_{crit.__class__.__name__}"] = mem_loss.item()
                loss += mem_loss
        else:
            loss = crit(scores, label_matrix)
            logs[crit.__class__.__name__] = loss.item()
            if memory:
                mem_loss = crit(memory_scores, memory_label_matrix)
                logs[f"memory_{crit.__class__.__name__}"] = mem_loss.item()
                loss += mem_loss

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
    memory=None,
):
    meter = lib.DictAverage()

    iterator = tqdm(loader, desc='Running epoch on loader')
    for i, batch in enumerate(iterator):
        logs = _batch_optimization(
            net,
            batch,
            criterion,
            memory,
        )

        optimizer.step()
        optimizer.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        if scheduler is not None:
            scheduler.step()

        meter.update(logs)
        iterator.set_postfix({k: f"{v:0.4f}" for k, v in meter.avg.items()})

    return meter.avg
