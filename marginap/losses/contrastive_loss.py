import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self, pos_margin, neg_margin):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, scores, target):
        pos_mask = target.bool() & (scores < self.pos_margin)
        pos_loss = (self.pos_margin - torch.masked_select(scores, pos_mask)).sum()

        neg_mask = ~target.bool() & (scores > self.neg_margin)
        neg_loss = (torch.masked_select(scores, neg_mask) - self.neg_margin).sum()

        loss = (pos_loss + neg_loss) / scores.size(0)
        return loss

    def extra_repr(self,) -> str:
        return f"pos_margin={self.pos_margin}, neg_margin={self.neg_margin}"
