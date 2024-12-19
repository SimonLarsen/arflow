import torch
from torch.nn.functional import mse_loss
from einops import reduce


class FlowMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target=None):
        loss = mse_loss(
            input=output.flow_pred,
            target=output.flow_true,
            reduction="none",
        )
        if output.mask is not None:
            loss = reduce(loss, "b np ... -> b np", "mean")
            return (loss * output.mask).sum() / output.mask.sum()
        else:
            return loss.mean()
