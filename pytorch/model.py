import torch
from torch import nn
import torch.nn.functional as F


def leaky_relu() -> nn.LeakyReLU:
    return nn.LeakyReLU(0.2, inplace=True)


kwds = {
    "kernel_size": 4,
    "stride": 2,
    "padding": 1
}


def get_critic() -> nn.Module:
    N = 32
    return nn.Sequential(
        nn.Conv2d(1, N, **kwds),    # (14,14)
        leaky_relu(),
        nn.Conv2d(N, N * 2, **kwds),    # (7,7)
        leaky_relu(),
        nn.Conv2d(N * 2, N * 4, kernel_size=2, stride=1, padding=0),  # (6,6)
        leaky_relu(),
        nn.Conv2d(N * 4, N * 8, **kwds),  # (3,3)
        leaky_relu(),
        nn.Conv2d(N * 8, 1, kernel_size=1, stride=1, padding=0),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )


def get_generator(z_dim: int) -> nn.Module:
    N = 32
    return nn.Sequential(
        nn.Linear(z_dim, 3 * 3 * N * 8, bias=False),
        nn.BatchNorm1d(3 * 3 * N * 8),
        leaky_relu(),
        Lambda(lambda x: x.reshape(-1, N * 8, 3, 3)),
        nn.ConvTranspose2d(N * 8, N * 4, bias=False, **kwds),
        nn.BatchNorm2d(N * 4),
        leaky_relu(),
        nn.ConvTranspose2d(N * 4, N * 2, kernel_size=2, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(N * 2),
        leaky_relu(),
        nn.ConvTranspose2d(N * 2, N, bias=False, **kwds),
        nn.BatchNorm2d(N),
        leaky_relu(),
        nn.ConvTranspose2d(N, 1, **kwds),
        nn.Sigmoid()
    )


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
