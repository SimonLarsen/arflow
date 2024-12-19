import torch
from torch import nn
from einops import rearrange
from .norms import norm_to_2d


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        activation: nn.Module = nn.GELU,
        norm: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.channels = channels

        self.act = activation()

        self.emb = nn.Linear(emb_channels, 2 * channels)

        self.norm1 = norm_to_2d(norm(channels))
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.norm2 = norm_to_2d(norm(channels))
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        emb = rearrange(self.emb(emb), "... -> ... 1 1")
        scale, shift = emb.chunk(2, dim=1)

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)

        return h + x
