import logging

import torch

import margin_ap.utils as lib


def _batch_optimization(
    config,
    net,
    batch,
    criterion,
    optimizer,
    epoch,
    memory
):
    di = net(batch["image"].cuda())
    labels = batch["label"].cuda()
    scores = torch.mm(di, di.t())
    label_matrix = lib.create_label_matrix(labels)

    if memory:
        if memory.activate_after >= epoch:
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
            if memory.activate_after >= epoch:
                mem_loss = mem_loss.mean()
                losses.append(weight * memory.weight * mem_loss)
                logs[f"memory_{crit.__class__.__name__}"] = mem_loss.item()

    total_loss = sum(losses)
    if config.experience.apex:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        total_loss.backward()

    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def base_update(
    config,
    net,
    loader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    memory=None,
):
    meter = lib.DictAverage()

    for i, batch in enumerate(loader):
        logs = _batch_optimization(
            config,
            net,
            batch,
            criterion,
            optimizer,
            epoch,
            memory,
        )

        for key, opt in optimizer.items():
            if epoch < config.experience.warm_up and key != config.experience.warm_up_key:
                logging.info(f"Warming up @epoch {epoch}")
                continue
            opt.step()

        net.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        for sch in scheduler["on_step"]:
            sch.step()

        meter.update(logs)
        if (i + 1) % 50 == 0:
            logging.info(f'Iteration : {i}/{len(loader)}')
            for k, v in logs.items():
                logging.info(f'Loss: {k}: {v} ')

    return meter.avg
