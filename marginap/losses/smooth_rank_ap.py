"""
adapted from
https://github.com/Andrew-Brown1/Smooth_AP

Changed so that the forward is called on a score and affinity matrixies
"""
from functools import partial

import torch
import torch.nn as nn


def tau_sigmoid(tensor, temp):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it
    through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = 1. + exponent.clamp(-50, 50).exp()
    return 1.0 / exponent


def rank_upper_bound(tens, theta, mu_n, tau_p):
    neg_mask = (tens < 0)
    pos_mask = ~neg_mask
    constrain_neg = (tens[neg_mask] > -mu_n).float()

    tens[neg_mask] = ((theta / (mu_n)) * tens[neg_mask] + theta) * constrain_neg
    tens[pos_mask] = tau_p * tens[pos_mask] + theta
    return tens


def piecewise_affine(tens, theta, mu_n, mu_p):
    neg_mask = (tens < 0)
    pos_mask = ~neg_mask
    constrain_neg = (tens[neg_mask] > -mu_n).float()
    constrain_pos = (tens[pos_mask] < mu_p)

    tens[neg_mask] = ((theta / mu_n) * tens[neg_mask] + theta) * constrain_neg
    tens[pos_mask] = (((1 - theta) / mu_p) * tens[pos_mask] + theta) * constrain_pos.float() + (~constrain_pos).float()
    return tens


class SmoothRankAP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores, target):
        batch_size = target.size(0)
        device = scores.device

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(batch_size, device=device).unsqueeze(0)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        # compute the difference matrix
        sim_diff = scores.unsqueeze(1) - scores.unsqueeze(1).permute(0, 2, 1)
        # pass through the sigmoid
        sim_diff_sigmoid = self.rank_approximation(sim_diff)

        sim_sg = sim_diff_sigmoid * mask
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        pos_mask = (target - torch.eye(batch_size).to(device))
        sim_pos_sg = sim_diff_sigmoid * pos_mask
        # compute the rankings of the positive set
        sim_pos_rk = (torch.sum(sim_pos_sg, dim=-1) + target) * target

        ap = ((sim_pos_rk / sim_all_rk).sum(1) * (1 / target.sum(1)))

        if self.return_type == 'AP':
            return ap
        if self.return_type == '1-AP':
            return 1 - ap
        else:
            loss = 1 - ap.mean()
            return loss


class SmoothAP(SmoothRankAP):

    def __init__(self, temp, return_type='1-mAP'):
        super().__init__()
        self.temp = temp
        assert return_type in ["1-mAP", "1-AP", "AP"]
        self.rank_approximation = partial(tau_sigmoid, temp=temp)
        self.return_type = return_type

    def extra_repr(self,):
        return f"temp={self.temp}"


class MarginAP(SmoothRankAP):

    def __init__(self, mu, tau, return_type='1-mAP'):
        super().__init__()
        self.mu = mu
        self.tau = tau
        assert return_type in ["1-mAP", "1-AP", "AP"]
        self.rank_approximation = partial(rank_upper_bound, theta=1.0, mu_n=mu, tau_p=tau)
        self.return_type = return_type

    def extra_repr(self,):
        return f"mu={self.mu}, tau={self.tau}"


class AffineAP(SmoothRankAP):

    def __init__(self, theta, mu_n, mu_p, return_type='1-mAP'):
        super().__init__()
        self.theta = theta
        self.mu_n = mu_n
        self.mu_p = mu_p
        assert return_type in ["1-mAP", "1-AP", "AP"]
        self.rank_approximation = partial(rank_upper_bound, theta=theta, mu_n=mu_n, mu_p=mu_p)
        self.return_type = return_type

    def extra_repr(self,):
        return f"theta={self.theta}, mu_n={self.mu_n}, mu_p={self.mu_p}"
