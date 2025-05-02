import torch
import torch.nn as nn
import torch.nn.functional as F

class Patcher(nn.Module):
    def __init__(self):
        super(Patcher, self).__init__()
    
    def forward(self, x):
        # Input dimensions: [batch, 1, number_of_timesteps]
        # Output dimensions: [batch, patch_length, number_of_patches]
        pass