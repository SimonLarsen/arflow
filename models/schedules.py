from typing import Sequence
import torch


def get_schedule(name: str):
    name = name.lower()
    if name in ("cos", "cosine"):
        return schedule_cosine
    elif name == "linear":
        return schedule_linear
    else:
        raise ValueError(f"Unsupported schedule '{name}'.")


def get_schedule_sampler(name: str):
    name = name.lower()
    if name == "lognorm":
        return sample_lognorm
    elif name in ("cos", "cosine"):
        return sample_cosine
    else:
        raise ValueError(f"Unsupported schedule '{name}'.")


def schedule_cosine(x):
    return torch.cos(x * torch.pi / 2.0)


def schedule_linear(x):
    return x


def sample_lognorm(
    size: Sequence[int],
    generator: torch.Generator = None,
    m: float = 0.0,
    s: float = 1.0,
) -> torch.FloatTensor:
    device = generator.device if generator else None
    x = m + s * torch.randn(size=size, generator=generator, device=device)
    return 1.0 / (1.0 + torch.exp(-x))


def sample_cosine(
    size: Sequence[int],
    generator: torch.Generator = None,
) -> torch.FloatTensor:
    device = generator.device if generator else None
    x = torch.rand(size=(size,), generator=generator, device=device)
    return schedule_cosine(x)
