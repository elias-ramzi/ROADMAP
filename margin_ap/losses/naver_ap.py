from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class NaverAP (nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:

        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(
        self,
        nq: Optional[int] = 20,
        min: Optional[int] = -1,
        max: Optional[int] = 1,
        return_type='1-mAP',
    ):
        super().__init__()
        assert isinstance(nq, int) and 2 <= nq <= 100
        assert return_type in ["1-mAP", "AP", "1-AP", "debug"]
        self.nq = nq
        self.min = min
        self.max = max
        self.return_type = return_type

        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq-1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0)
        # with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0)
        # with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x: Tensor, label: Tensor, qw: Optional[Tensor] = None) -> Tensor:
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if self.return_type == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            loss = 1 - ap.mean()
            return loss
        elif self.return_type == 'AP':
            assert qw is None
            return ap
        elif self.return_type == 'mAP':
            assert qw is None
            return ap.mean()
        elif self.return_type == '1-AP':
            return 1 - ap
        elif self.return_type == 'debug':
            return prec, rec

    def extra_repr(self,):
        return f"nq={self.nq}, min={self.min}, max={self.max}, return_type={self.return_type}"
