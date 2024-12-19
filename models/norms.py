from torch import nn
import torch
from einops import rearrange


def get_norm(name: str):
    name = name.lower()
    if name in ("layer", "layernorm"):
        return nn.LayerNorm
    elif name in ("rms", "rmsnorm"):
        return RMSNorm
    elif name in ("simplerms", "simplermsnorm"):
        return SimpleRMSNorm
    else:
        raise ValueError(f"Unsupported norm type '{name}'.")


class WrapNorm2d(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


def norm_to_2d(norm: nn.Module):
    if isinstance(norm, nn.GroupNorm):
        return norm
    return WrapNorm2d(norm)


class RMSNorm(nn.Module):
    def __init__(self, dim, unit_offset: bool = False):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim**0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1.0 - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return nn.functional.normalize(x, dim=-1) * self.scale * gamma


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x):
        return nn.functional.normalize(x, dim=-1) * self.scale
