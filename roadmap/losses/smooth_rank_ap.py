"""
inspired from
https://github.com/Andrew-Brown1/Smooth_AP
"""
from functools import partial

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import roadmap.utils as lib


def heaviside(tens, val=1., target=None, general=None):
    return torch.heaviside(tens, values=torch.tensor(val, device=tens.device, dtype=tens.dtype))


def tau_sigmoid(tensor, tau, target=None, general=None):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it
    through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / tau
    # clamp the input tensor for stability
    exponent = 1. + exponent.clamp(-50, 50).exp()
    return 1.0 / exponent


def step_rank(tens, tau, rho, offset, delta=None, start=0.5, target=None, general=False):
    target = target.squeeze()
    if general:
        target = target.view(1, -1).repeat(tens.size(0), 1)
    else:
        mask = target.unsqueeze(target.ndim - 1).bool()
        target = lib.create_label_matrix(target).bool() * mask
    pos_mask = (tens > 0).bool()
    neg_mask = ~pos_mask

    if isinstance(tau, str):
        tau_n, tau_p = tau.split("_")
    else:
        tau_n = tau_p = tau

    if delta is None:
        tens[~target & pos_mask] = rho * tens[~target & pos_mask] + offset
    else:
        margin_mask = tens > delta
        tens[~target & pos_mask & ~margin_mask] = start + tau_sigmoid(tens[~target & pos_mask & ~margin_mask], tau_p).type(tens.dtype)
        if offset is None:
            offset = tau_sigmoid(torch.tensor([delta], device=tens.device), tau_p).type(tens.dtype) + start
        tens[~target & pos_mask & margin_mask] = rho * (tens[~target & pos_mask & margin_mask] - delta) + offset

    tens[~target & neg_mask] = tau_sigmoid(tens[~target & neg_mask], tau_n).type(tens.dtype)

    tens[target] = torch.heaviside(tens[target], values=torch.tensor(1., device=tens.device, dtype=tens.dtype))

    return tens


class SmoothRankAP(nn.Module):
    def __init__(
        self,
        rank_approximation,
        return_type='1-mAP',
    ):
        super().__init__()
        self.rank_approximation = rank_approximation
        self.return_type = return_type
        assert return_type in ["1-mAP", "1-AP", "AP", 'mAP']

    def general_forward(self, scores, target, verbose=False):
        batch_size = target.size(0)
        nb_instances = target.size(1)
        device = scores.device

        ap_score = []
        mask = (1 - torch.eye(nb_instances, device=device))
        iterator = range(batch_size)
        if verbose:
            iterator = tqdm(iterator, leave=None)
        for idx in iterator:
            # shape M
            query = scores[idx]
            pos_mask = target[idx].bool()

            # shape M x M
            query = query.view(1, -1) - query[pos_mask].view(-1, 1)
            query = self.rank_approximation(query, target=pos_mask, general=True) * mask[pos_mask]

            # shape M
            rk = 1 + query.sum(-1)

            # shape M
            pos_rk = 1 + (query * pos_mask.view(1, -1)).sum(-1)

            # shape 1
            ap = (pos_rk / rk).sum(-1) / pos_mask.sum()
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
        return repr


class HeavisideAP(SmoothRankAP):
    """here for testing purposes"""

    def __init__(self, **kwargs):
        rank_approximation = partial(heaviside)
        super().__init__(rank_approximation, **kwargs)

    def extra_repr(self,):
        repr = self.my_repr
        return repr


class SmoothAP(SmoothRankAP):

    def __init__(self, tau=0.01, **kwargs):
        rank_approximation = partial(tau_sigmoid, tau=tau)
        super().__init__(rank_approximation, **kwargs)
        self.tau = tau

    def extra_repr(self,):
        repr = f"tau={self.tau}, {self.my_repr}"
        return repr


class SupAP(SmoothRankAP):

    def __init__(self, tau=0.01, rho=100, offset=1.44, delta=0.05, start=0.5, **kwargs):
        rank_approximation = partial(step_rank, tau=tau, rho=rho, offset=offset, delta=delta, start=start)
        super().__init__(rank_approximation, **kwargs)
        self.tau = tau
        self.rho = rho
        self.offset = offset
        self.delta = delta

    def extra_repr(self,):
        repr = f"tau={self.tau}, rho={self.rho}, offset={self.offset}, delta={self.delta}, {self.my_repr}"
        return repr
