import logging

import torch
from torch.utils.data import DataLoader, Subset

import margin_ap.utils as lib


def _compute_descriptors(net, loader,):
    descriptors = []
    labels = []
    for batch in loader:
        # computes descriptors without keeping activations for backpropagations
        with torch.no_grad():
            descriptors.append(net(batch["image"].cuda()))
        labels.extend(batch["label"].view(-1).tolist())

    descriptors = torch.cat(descriptors)
    labels = torch.tensor(labels).cuda()
    return descriptors, labels


def _batch_optimization(net, loader, grad):
    for i, batch in enumerate(loader):
        # computing activations needed for backpropagation for a subset of images
        di = net(batch["image"].cuda())

        # backpropagating the gradient obtained for the subset of images
        di.backward(grad[i * loader.batch_size: (i+1)*loader.batch_size])


def run_large_batch(net, loader, criterion, scaler):
    """
    Memory effective backpropagation algorithm for optimizing a
    loss function, as described in the paper
    https://arxiv.org/abs/1906.07589
    """
    descriptors, labels = _compute_descriptors(net, loader)
    # Computing gradient starting at the descriptor level
    descriptors.requires_grad_(True)
    descriptors.retain_grad()

    similarities = torch.mm(descriptors, descriptors.t())
    label_matrix = lib.create_label_matrix(labels)

    logs = {}
    losses = []
    for crit, weight in criterion:
        if hasattr(crit, 'takes_embeddings'):
            loss = crit(descriptors, labels.view(-1))
        else:
            loss = crit(similarities, label_matrix)

        loss = loss.mean()
        losses.append(weight * loss)
        logs[crit.__class__.__name__] = loss.item()

    total_loss = sum(losses)
    # This gives the gradients for the descriptors given by the criterion
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    logs["total_loss"] = total_loss.item()

    # we only need the values of the gradient and not the calcul graph
    grad = descriptors.grad.detach()
    descriptors.detach_()

    # backpropagating gradient for all images. Proceeding in subset batches
    _batch_optimization(net, loader, grad)
    return logs


def large_batch_update(
    config,
    net,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    memory=None,
):
    meter = lib.DictAverage()
    net.train()
    net.zero_grad()

    with torch.cuda.amp.autocast(enabled=(scaler is not None)):

        for i, batch_idx in enumerate(loader.batch_sampler.batches):
            dataset = Subset(loader.dataset, batch_idx)

            sub_loader = DataLoader(
                dataset,
                batch_size=config.experience.sub_batch,
                num_workers=config.experience.num_workers,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
            )

            logs = run_large_batch(
                net,
                sub_loader,
                criterion,
                scaler
            )

            for key, opt in optimizer.items():
                if scaler is None:
                    opt.step()
                else:
                    scaler.step(opt)

            net.zero_grad()
            _ = [crit.zero_grad() for crit, w in criterion]

            for sch in scheduler["on_step"]:
                sch.step()

            if scaler is not None:
                scaler.update()

            meter.update(logs)
            logging.info(f'Iteration : {i+1}/{len(loader)}')
            for k, v in logs.items():
                logging.info(f'Loss: {k}: {v} ')

        return meter.avg
