import torch.nn as nn

import roadmap.utils as lib


def create_projection_head(
    input_dimension=2048,
    layer_dim=512,
    normalization_layer='none',
):
    if isinstance(layer_dim, int):
        return nn.Linear(input_dimension, layer_dim)

    elif lib.list_or_tuple(layer_dim):
        layers = []
        prev_dim = input_dimension
        for i, dim in enumerate(layer_dim):
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
            if i < len(layer_dim) - 1:
                if normalization_layer == 'bn':
                    layers.append(nn.BatchNorm1d(dim))
                elif normalization_layer == 'ln':
                    layers.append(nn.LayerNorm(dim))
                elif normalization_layer == 'none':
                    pass
                else:
                    raise ValueError(f"Unknown normalization layer : {normalization_layer}")
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
