import torch


def get_gradient_norm(net):
    if torch.cuda.device_count() > 1:
        net = net.module

    if hasattr(net, 'fc'):
        final_layer = net.fc

    elif hasattr(net, 'blocks'):
        final_layer = net.blocks[-1].mlp.fc2

    return torch.norm(list(final_layer.parameters())[0].grad, 2)
