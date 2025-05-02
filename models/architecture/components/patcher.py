import torch
import torch.nn as nn
import torch.nn.functional as F


class Patcher(nn.Module):
    def __init__(self):
        super(Patcher, self).__init__()

    def forward(self, x):
        # Input dimensions: [batch, 1, number_of_timesteps]
        # Output dimensions: [batch, patch_length, number_of_patches]
        patch_length = 32
        stride = 2
        # l = x.shape[1]
        # N = ((l - patch_length) // stride) + 2
        X_padded = F.pad(x, pad=(0, stride), mode='replicate')
        output = X_padded.unfold(dim=2, size=patch_length, step=stride)

        return output.squeeze(1)
