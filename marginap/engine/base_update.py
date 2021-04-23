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
            loss = crit(di, labels.view(-1))
            if memory:
                mem_loss = crit(di, labels.view(-1), memory_embeddings, memory_labels.view(-1))

        else:
            loss = crit(scores, label_matrix)
            if memory:
                mem_loss = crit(memory_scores, memory_label_matrix)

        loss = loss.mean()
        losses.append(weight * loss)
        logs[crit.__class__.__name__] = loss.item()
        if memory:
            mem_loss = mem_loss.mean()
            losses.append(weight * mem_loss)
            logs[f"memory_{crit.__class__.__name__}"] = mem_loss.item()

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
    scheduler,
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

        for opt in optimizer.values():
            optimizer.step()
            optimizer.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        for sch in scheduler["on_step"]:
            sch.step()

        meter.update(logs)
        iterator.set_postfix({k: f"{v:0.4f}" for k, v in meter.avg.items()})

    return meter.avg
