def freeze_pos_embedding(net):
    net.pos_embed.requires_grad_(False)
    return net
