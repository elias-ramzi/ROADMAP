import torch


class EntropyRegularization(torch.nn.Module):

    def forward(self, scores, target=None):
        eps = 1e-5
        mask = torch.ones_like(scores, dtype=scores.dtype, device=scores.device)

        distance = 2 * (1 - scores) + 5. * mask
        idx = distance.argmin(-1)
        koleo = torch.log(distance[idx] + eps)
        return -koleo.mean()
