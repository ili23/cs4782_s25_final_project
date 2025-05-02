import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self, patch_dim, d_dim):
        super(Projection, self).__init__()
        self.proj = nn.Linear(patch_dim, d_dim)

    def forward(self, x):
        x = self.proj(x)
        return x
