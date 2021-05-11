"""
inspired from
https://github.com/Andrew-Brown1/Smooth_AP
"""
from functools import partial

import torch
import torch.nn as nn
from tqdm.auto import tqdm


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
    constrain_neg = (tens[neg_mask] > -mu_n).type(tens.dtype)

    tens[neg_mask] = ((theta / (mu_n)) * tens[neg_mask] + theta) * constrain_neg
    tens[pos_mask] = tau_p * tens[pos_mask] + theta
    return tens


def piecewise_affine(tens, theta, mu_n, mu_p):
    neg_mask = (tens < 0)
    pos_mask = ~neg_mask
    constrain_neg = (tens[neg_mask] > -mu_n).type(tens.dtype)
    constrain_pos = (tens[pos_mask] < mu_p)

    tens[neg_mask] = ((theta / mu_n) * tens[neg_mask] + theta) * constrain_neg
    tens[pos_mask] = (((1 - theta) / mu_p) * tens[pos_mask] + theta) * constrain_pos.type(tens.dtype) + (~constrain_pos).type(tens.dtype)

    return tens


def change_regime(tens, theta, mu, tau):
    neg_mask = (tens < 0)
    constrain_neg = (tens[neg_mask] > -mu).type(tens.dtype)

    pos_mask = ~neg_mask
    constrain_pos = tens < mu
    high = pos_mask & constrain_pos
    slope = pos_mask & (~constrain_pos)

    tens[neg_mask] = ((theta / mu) * tens[neg_mask] + theta) * constrain_neg
    tens[high] = (((1 - theta) / mu) * tens[high] + theta)
    tens[slope] = tens[slope] * tau + 1

    return tens


def ap_upper_bound(tens, mu, tau, target):
    target = target.unsqueeze(1).bool()

    tens[target] = piecewise_affine(tens, 0.5, mu, mu)
    tens[~target] = change_regime(tens, 0.5, mu, tau)

    return tens


class SmoothRankAP(nn.Module):
    def __init__(
        self,
        rank_approximation,
        return_type='1-mAP',
        gamma=None,
        with_true_rank=False
    ):
        super().__init__()
        self.rank_approximation = rank_approximation
        self.return_type = return_type
        self.gamma = gamma
        self.with_true_rank = with_true_rank
        assert return_type in ["1-mAP", "1-AP", "AP", 'mAP']

    def general_forward(self, scores, target, verbose=False):
        batch_size = target.size(0)
        nb_instances = target.size(1)
        device = scores.device
        target = target.type(scores.dtype)

        ap_score = []
        mask = (1 - torch.eye(nb_instances, device=device))
        iterator = range(batch_size)
        if verbose:
            iterator = tqdm(iterator, leave=None)
        for idx in iterator:
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
        sim_diff_sigmoid = self.rank_approximation(sim_diff)

        sim_sg = sim_diff_sigmoid * mask
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        if self.with_true_rank:
            sim_pos_rk = 1 + scores.detach().argsort(-1, True).argsort(-1)
            sim_pos_rk = sim_pos_rk.type(scores.dtype) * target
        else:
            pos_mask = (target - torch.eye(batch_size).to(device))
            sim_pos_sg = sim_diff_sigmoid * pos_mask
            sim_pos_rk = (torch.sum(sim_pos_sg, dim=-1) + target) * target
        # compute the rankings of the positive set

        ap = ((sim_pos_rk / sim_all_rk).sum(1) * (1 / target.sum(1)))
        return ap

    def forward(self, scores, target, force_general=False, verbose=False):
        assert scores.shape == target.shape
        assert len(scores.shape) == 2

        if self.gamma is not None:
            scores -= self.gamma * torch.randn_like(scores, device=scores.device, dtype=scores.dtype).abs() * (target - 0.5)

        if (scores.size(0) == scores.size(1)) and not force_general:
            ap = self.quick_forward(scores, target)

        else:
            ap = self.general_forward(scores, target, verbose=verbose)

        if self.return_type == 'AP':
            return ap
        elif self.return_type == 'mAP':
            return ap.mean()
        elif self.return_type == '1-AP':
            return 1 - ap
        elif self.return_type == '1-mAP':
            return 1 - ap.mean()

    @property
    def my_repr(self,):
        repr = f"return_type={self.return_type}"
        if self.gamma is not None:
            repr += f", gamma={self.gamma}"
        if self.with_true_rank is not None:
            repr += f", with_true_rank={self.with_true_rank}"

        return repr


class HeavisideAP(SmoothRankAP):
    """here for testing purposes"""

    def __init__(self, **kwargs):
        rank_approximation = partial(torch.heaviside, values=torch.tensor(0.))
        super().__init__(rank_approximation, **kwargs)

    def extra_repr(self,):
        repr = self.my_repr
        return repr


class SmoothAP(SmoothRankAP):

    def __init__(self, temp=0.01, **kwargs):
        rank_approximation = partial(tau_sigmoid, temp=temp)
        super().__init__(rank_approximation, **kwargs)
        self.temp = temp

    def extra_repr(self,):
        repr = f"temp={self.temp}, {self.my_repr}"
        return repr


class MarginAP(SmoothRankAP):

    def __init__(self, mu, tau, **kwargs):
        rank_approximation = partial(rank_upper_bound, theta=1.0, mu_n=mu, tau_p=tau)
        super().__init__(rank_approximation, **kwargs)
        self.mu = mu
        self.tau = tau

    def extra_repr(self,):
        repr = f"mu={self.mu}, tau={self.tau}, {self.my_repr}"
        return repr


class AffineAP(SmoothRankAP):

    def __init__(self, theta, mu_n, mu_p, **kwargs):
        rank_approximation = partial(piecewise_affine, theta=theta, mu_n=mu_n, mu_p=mu_p)
        super().__init__(rank_approximation, **kwargs)
        self.theta = theta
        self.mu_n = mu_n
        self.mu_p = mu_p

    def extra_repr(self,):
        repr = f"theta={self.theta}, mu_n={self.mu_n}, mu_p={self.mu_p}, {self.my_repr}"
        return repr


class AdaptativeAP(SmoothRankAP):

    def __init__(self, theta, mu, tau, **kwargs):
        rank_approximation = partial(change_regime, theta=theta, mu=mu, tau=tau)
        super().__init__(rank_approximation, **kwargs)
        self.mu = mu
        self.tau = tau

    def extra_repr(self,):
        repr = f"mu={self.mu}, tau={self.tau}, {self.my_repr}"
        return repr


class ScheduledSlopeAP(SmoothRankAP):

    def __init__(self, mu_n_range, mu_p_range, num_steps, scheduled_type='linear', **kwargs):
        super().__init__(self.scheduled_slope, **kwargs)
        self.scheduled_type = scheduled_type
        self.time = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.num_steps = nn.Parameter(torch.tensor(num_steps), requires_grad=False)

        if isinstance(mu_n_range, float):
            mu_n_range = [mu_n_range]
        if isinstance(mu_p_range, float):
            mu_p_range = [mu_p_range]

        if scheduled_type == 'linear':
            self.mu_n_range = nn.Parameter(torch.linspace(mu_n_range[0], mu_n_range[-1], num_steps), requires_grad=False)
            self.mu_p_range = nn.Parameter(torch.linspace(mu_p_range[0], mu_p_range[-1], num_steps), requires_grad=False)
        elif scheduled_type == 'step':
            self.mu_n_range = nn.Parameter(self.get_steps(mu_n_range), requires_grad=False)
            self.mu_p_range = nn.Parameter(self.get_steps(mu_p_range), requires_grad=False)

    def get_steps(self, mu_range):
        mu = [0] * self.num_steps
        all_steps = sorted(mu_range.keys(), reverse=True)
        for step in all_steps:
            for i in range(len(mu)):
                if i < step:
                    mu[i] = mu_range[step]
        return torch.tensor(mu)

    def scheduled_slope(self, tens):
        return piecewise_affine(
            tens,
            0.5,
            self.mu_n_range[min(self.time, self.num_steps - 1)],
            self.mu_p_range[min(self.time, self.num_steps - 1)],
        )

    def step(self,):
        self.time += 1

    def extra_repr(self,):
        repr = f"\tmu_n_range={self.mu_n_range},"
        repr += f"\n\tmu_p_range={self.mu_p_range},"
        repr += f"\n\tnum_steps={self.num_steps},"
        repr += f"\n\tscheduled_type={self.scheduled_type},"
        repr += f"\n\ttime={self.time},"
        repr += f"\n\t{self.my_repr}"
        return repr
