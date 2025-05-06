import torch
import torch.nn as nn

class InstanceNormalizer(nn.Module):
    def __init__(self, eps=1e-5):
        super(InstanceNormalizer, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        # Input dimensions: [batch, num_features, number_of_timesteps]
        # Output dimensions: [batch, num_features, number_of_timesteps]
        # We want to get the mean and variance across the timestep dimension (dim=1)
        mean = torch.mean(x, dim=-1, keepdim=True)
        # var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        # std = torch.sqrt(var + self.eps)
        std = x.std(dim=-1, keepdim=True) + self.eps

        normalized = (x - mean) / std
        return normalized, mean, std
    
    def denormalize(self, normalized_x, mean, std):
        return normalized_x * std + mean