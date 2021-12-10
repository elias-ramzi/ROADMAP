# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in
# https://github.com/msight-tech/research-xbm/blob/master/LICENSE
import torch
from torch import nn


class PairLoss(nn.Module):
    takes_embeddings = True

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def compute_loss(self, inputs_col, targets_col, inputs_row, target_row):

        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        neg_count = list()
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = 0

            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        return loss

    def forward(self, embeddings, labels, ref_embeddings=None, ref_labels=None):
        if ref_embeddings is None:
            return self.compute_loss(embeddings, labels, embeddings, labels)

        return self.compute_loss(embeddings, labels, ref_embeddings, ref_labels)

    def extra_repr(self,):
        return f"margin={self.margin}"
