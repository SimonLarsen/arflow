from functools import partial
from torch import nn


def get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return nn.GELU
    elif name == "fast_gelu":
        return partial(nn.GELU, approximate="tanh")
    elif name == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unsupported activation type '{name}'.")
