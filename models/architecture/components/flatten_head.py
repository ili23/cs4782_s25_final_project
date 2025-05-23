import torch
import torch.nn as nn
from einops import rearrange

class FlattenHead(nn.Module):
    def __init__(self, num_features, target_window):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=-2)
        self.linear = torch.nn.Linear(num_features, target_window)

    def forward(self, x, b, f):
        x = rearrange(x, '(b f) p n -> b f p n', b=b, f=f)
        x = self.flatten(x)
        x = self.linear(x)
        return x
