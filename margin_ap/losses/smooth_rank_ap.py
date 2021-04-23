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


def piecewise_affine(tens, theta, mu_n, mu_p, target=None, gamma=None):
    neg_mask = (tens < 0)
    pos_mask = ~neg_mask
    constrain_neg = (tens[neg_mask] > -mu_n).float()
    constrain_pos = (tens[pos_mask] < mu_p)

    tens[neg_mask] = ((theta / mu_n) * tens[neg_mask] + theta) * constrain_neg
    tens[pos_mask] = (((1 - theta) / mu_p) * tens[pos_mask] + theta) * constrain_pos.float() + (~constrain_pos).float()

    if gamma is not None:
        sto = (1 - target.unsqueeze(1)) * (1 + gamma * torch.randn(tens.size(0), tens.size(0), tens.size(1), device=tens.device).abs())
        sto += target
        tens *= sto
        tens.clamp_max_(1.)

    return tens


def change_regime(tens, theta, mu, tau):
    neg_mask = (tens < 0)
    constrain_neg = (tens[neg_mask] > -mu).float()

    pos_mask = ~neg_mask
    constrain_pos = tens < mu
    high = pos_mask & constrain_pos
    slope = pos_mask & (~constrain_pos)

    tens[neg_mask] = ((theta / mu) * tens[neg_mask] + theta) * constrain_neg
    tens[high] = (((1 - theta) / mu) * tens[high] + theta)
    tens[slope] = tens[slope] * tau + 1

    return tens


class SmoothRankAP(nn.Module):
    def __init__(self, rank_approximation, rank_needs_target=False, return_type='1-mAP'):
        super().__init__()
        self.rank_approximation = rank_approximation
        self.return_type = return_type
        self.rank_needs_target = rank_needs_target
        assert return_type in ["1-mAP", "1-AP", "AP", 'mAP']

    def general_forward(self, scores, target):
        batch_size = target.size(0)
        nb_instances = target.size(1)
        device = scores.device

        ap_score = []
        mask = (1 - torch.eye(nb_instances, device=device))
        for idx in range(batch_size):
            # shape M
            query = scores[idx]
            # shape M x M
            query = query.view(1, -1) - query.view(-1, 1)
            query = self.rank_approximation(query) * mask
            # shape M
            pos_mask = target[idx]
            pos_mask = pos_mask.view(1, -1)

            # shape M
            rk = torch.sum(query, -1) + 1

            # shape M x M
            pos_rk = query * pos_mask
            # shape M
            pos_rk = (torch.sum(pos_rk, -1) + target[idx]) * target[idx]

            # shape 1
            ap = (pos_rk / rk).sum(-1) / target[idx].sum(-1)
            ap_score.append(ap)

        # shape N
        ap_score = torch.stack(ap_score)
        return ap_score

    def quick_forward(self, scores, target):
        batch_size = target.size(0)
        device = scores.device

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(batch_size, device=device).unsqueeze(0)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        # compute the difference matrix
        sim_diff = scores.unsqueeze(1) - scores.unsqueeze(1).permute(0, 2, 1)
        # pass through the sigmoid
        if self.rank_needs_target:
            sim_diff_sigmoid = self.rank_approximation(sim_diff, target=target)
        else:
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
        return ap

    def forward(self, scores, target):
        assert scores.shape == target.shape
        assert len(scores.shape) == 2

        if scores.size(0) == scores.size(1):
            ap = self.quick_forward(scores, target)

        else:
            ap = self.general_forward(scores, target)

        if self.return_type == 'AP':
            return ap
        if self.return_type == 'mAP':
            return ap.mean()
        if self.return_type == '1-AP':
            return 1 - ap
        else:
            loss = 1 - ap.mean()
            return loss


class HeavisideAP(SmoothRankAP):
    """here for testing purposes"""

    def __init__(self, return_type='1-mAP'):
        rank_approximation = partial(torch.heaviside, values=torch.tensor(1.))
        super().__init__(rank_approximation, return_type=return_type)


class SmoothAP(SmoothRankAP):

    def __init__(self, temp, return_type='1-mAP'):
        rank_approximation = partial(tau_sigmoid, temp=temp)
        super().__init__(rank_approximation, return_type=return_type)
        self.temp = temp

    def extra_repr(self,):
        return f"temp={self.temp}"


class MarginAP(SmoothRankAP):

    def __init__(self, mu, tau, return_type='1-mAP'):
        rank_approximation = partial(rank_upper_bound, theta=1.0, mu_n=mu, tau_p=tau)
        super().__init__(rank_approximation, return_type=return_type)
        self.mu = mu
        self.tau = tau

    def extra_repr(self,):
        return f"mu={self.mu}, tau={self.tau}, return_type={self.return_type}"


class AffineAP(SmoothRankAP):

    def __init__(self, theta, mu_n, mu_p, gamma=None, return_type='1-mAP'):
        rank_approximation = partial(piecewise_affine, theta=theta, mu_n=mu_n, mu_p=mu_p, gamma=gamma)
        super().__init__(rank_approximation, rank_needs_target=(gamma is not None), return_type=return_type)
        self.theta = theta
        self.mu_n = mu_n
        self.mu_p = mu_p
        self.gamma = gamma

    def extra_repr(self,):
        repr = f"theta={self.theta}, mu_n={self.mu_n}, mu_p={self.mu_p}"
        if self.gamma is not None:
            repr += f", gamma={self.gamma}"
        return repr


class AdaptativeAP(SmoothRankAP):

    def __init__(self, theta, mu, tau, return_type='1-mAP'):
        rank_approximation = partial(change_regime, theta=theta, mu=mu, tau=tau)
        super().__init__(rank_approximation, return_type=return_type)
        self.mu = mu
        self.tau = tau

    def extra_repr(self,):
        return f"mu={self.mu}, tau={self.tau}, return_type={self.return_type}"
