import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VariableSplitter(nn.Module):
    def __init__(self):
        super(VariableSplitter, self).__init__()
    
    def forward(self, x):
        # Input dimensions: [batch, num_features, patch_length, number_of_patches]
        # Output dimensions: [batch * num_features, patch_length, number_of_patches] * num_features
        x = rearrange(x, 'b f p n -> (b f) p n')
        return x