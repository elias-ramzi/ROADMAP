"""
inspired from
https://github.com/Andrew-Brown1/Smooth_AP
"""
from functools import partial

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import margin_ap.utils as lib


def tau_sigmoid(tensor, temp):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it
    through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = 1. + exponent.clamp(-50, 50).exp()
    return 1.0 / exponent


def piecewise_affine(tens, theta, mu_n, mu_p):
    neg_mask = (tens < 0)
    pos_mask = ~neg_mask
    constrain_neg = (tens[neg_mask] > -mu_n).type(tens.dtype)
    constrain_pos = (tens[pos_mask] < mu_p)

    if mu_n > 0.:
        tens[neg_mask] = ((theta / mu_n) * tens[neg_mask] + theta) * constrain_neg
    else:
        tens[neg_mask] *= 0.

    tens[pos_mask] = (((1 - theta) / mu_p) * tens[pos_mask] + theta) * constrain_pos.type(tens.dtype) + (~constrain_pos).type(tens.dtype)

    return tens


def step_rank(tens, temp, tau, offset, margin=None, target=None):
    target = target.squeeze()
    mask = target.unsqueeze(target.ndim - 1).bool()
    target = lib.create_label_matrix(target).bool() * mask
    pos_mask = (tens > 0).bool()
    neg_mask = ~pos_mask

    if not margin:
        tens[~target & pos_mask] = tau * tens[~target & pos_mask] + offset
    else:
        margin_mask = tens > margin
        tens[~target & pos_mask & ~margin_mask] = 0.5 + tau_sigmoid(tens[~target & pos_mask & ~margin_mask], temp).type(tens.dtype)
        tens[~target & pos_mask & margin_mask] = tau * tens[~target & pos_mask & margin_mask] + offset

    tens[~target & neg_mask] = tau_sigmoid(tens[~target & neg_mask], temp).type(tens.dtype)

    tens[target] = torch.heaviside(tens[target], values=torch.tensor(1., device=tens.device, dtype=tens.dtype))

    return tens


def no_saturarion_smoothap(tens, temp, tau, offset, target=None):
    target = target.squeeze()
    mask = target.unsqueeze(target.ndim - 1).bool()
    target = lib.create_label_matrix(target).bool() * mask
    pos_mask = (tens > 0).bool()
    neg_mask = ~pos_mask

    tens[~target & pos_mask] = tau * tens[~target & pos_mask] + offset

    tens[~target & neg_mask] = tau_sigmoid(tens[~target & neg_mask], temp).type(tens.dtype)

    tens[target] = tau_sigmoid(tens[target], temp).type(tens.dtype)

    return tens


def step_smoothap(tens, temp, ponder=None, target=None):
    target = target.squeeze()
    mask = target.unsqueeze(target.ndim - 1).bool()
    target = lib.create_label_matrix(target).bool() * mask

    if ponder is not None:
        pos_mask = (tens > 0).bool()
        neg_mask = ~pos_mask
        tens[~target & neg_mask] = tau_sigmoid(tens[~target & neg_mask], temp).type(tens.dtype)
        tens[~target & pos_mask] = ponder * tau_sigmoid(tens[~target & pos_mask], temp).type(tens.dtype)
    else:
        tens[~target] = tau_sigmoid(tens[~target], temp).type(tens.dtype)

    tens[target] = torch.heaviside(tens[target], values=torch.tensor(1., device=tens.device, dtype=tens.dtype))
    return tens


class SmoothRankAP(nn.Module):
    def __init__(
        self,
        rank_approximation,
        return_type='1-mAP',
        gamma=None,
        rank_approximation_supervised=False,
    ):
        super().__init__()
        self.rank_approximation = rank_approximation
        self.return_type = return_type
        self.gamma = gamma
        self.rank_approximation_supervised = rank_approximation_supervised
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
            if not self.rank_approximation_supervised:
                query = self.rank_approximation(query) * mask
            else:
                query = self.rank_approximation(query, target=target[idx]) * mask

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
        if not self.rank_approximation_supervised:
            sim_diff_sigmoid = self.rank_approximation(sim_diff)
        else:
            sim_diff_sigmoid = self.rank_approximation(sim_diff, target=target)

        sim_sg = sim_diff_sigmoid * mask
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
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


class StepSmoothAP(SmoothRankAP):

    def __init__(self, temp=0.01, ponder=None, **kwargs):
        rank_approximation = partial(step_smoothap, temp=temp, ponder=ponder)
        kwargs["rank_approximation_supervised"] = True
        super().__init__(rank_approximation, **kwargs)
        self.temp = temp

    def extra_repr(self,):
        repr = f"temp={self.temp}, {self.my_repr}"
        return repr


class NoSaturationSmoothAP(SmoothRankAP):

    def __init__(self, temp=0.01, tau=1.0, offset=1.0, **kwargs):
        rank_approximation = partial(no_saturarion_smoothap, temp=temp, tau=tau, offset=offset)
        kwargs["rank_approximation_supervised"] = True
        super().__init__(rank_approximation, **kwargs)
        self.temp = temp
        self.tau = tau
        self.offset = offset

    def extra_repr(self,):
        repr = f"temp={self.temp}, tau={self.tau}, offset={self.offset}, {self.my_repr}"
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


class MarginAP(SmoothRankAP):

    def __init__(self, temp, tau, offset, margin=None, **kwargs):
        rank_approximation = partial(step_rank, temp=temp, tau=tau, offset=offset, margin=margin)
        kwargs["rank_approximation_supervised"] = True
        super().__init__(rank_approximation, **kwargs)
        self.temp = temp
        self.tau = tau
        self.offset = offset
        self.margin = margin

    def extra_repr(self,):
        repr = f"temp={self.temp}, tau={self.tau}, offset={self.offset}, margin={self.margin}, {self.my_repr}"
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
