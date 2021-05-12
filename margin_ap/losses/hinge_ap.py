from typing import Optional
import logging

import torch

try:
    import structured_ap_loss_cpp
except ImportError:
    logging.debug("Hinge AP is not available")


class HingeAP(torch.nn.Module):
    """
    Implement the structured Average Precision loss.
    """
    def __init__(self, return_type='1-mAP'):
        super().__init__()
        self.return_type = return_type
        assert return_type in ['1-mAP', '1-AP']

    def forward(
        self, prediction: torch.FloatTensor, target: torch.LongTensor, mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute the structured Average Precision loss.

        Args:
            prediction (``torch.FloatTensor`` of shape ``(batch_size, num_classes)``): Specifies the tensor with the
                predictions for each example.
            target (``torch.LongTensor`` of shape ``(batch_size, num_classes)``): Specifies the tensor with the
                targets for each example. The target of a positive (resp. negative) example should be 1 (resp. 0)
            mask (``torch.BoolTensor`` of shape ``(batch_size, num_classes)``, optional): Specifies a mask to ignore
                some examples. If the mask is True, the example is kept otherwise it is ignored.

        Returns:
            ``torch.FloatTensor``: The loss value. If there are several classes, the final loss is the average of the
                losses per class.
        """
        with torch.cuda.amp.autocast(enabled=False):

            prediction = prediction.t()
            target = target.t()

            if mask is None:
                mask = torch.ones_like(target)
            else:
                mask = mask.t()
            loss = StructuredMAPRankingLossFunction.apply(
                prediction.type(torch.float), target.type(torch.long), mask.type(torch.bool)
            )
            if self.return_type == '1-mAP':
                return loss.mean()
            elif self.return_type == '1-AP':
                return loss


class StructuredMAPRankingLossFunction(torch.autograd.Function):
    """
    You should not call directly this function, but the ``nn.Module`` ``StructuredMAPRankingLoss``.
    """

    @staticmethod
    def forward(
        ctx, prediction: torch.FloatTensor, target: torch.LongTensor, mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        with torch.cuda.amp.autocast(enabled=False):
            loss, ranking_lai = structured_ap_loss_cpp.forward(prediction, target, mask)
            ctx.save_for_backward(prediction, target, mask, ranking_lai)
            return loss

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        with torch.cuda.amp.autocast(enabled=False):
            prediction, target, mask, ranking_lai = ctx.saved_tensors
            grad_input = structured_ap_loss_cpp.backward(grad_output, prediction, target, mask, ranking_lai)
            return grad_input, None, None
