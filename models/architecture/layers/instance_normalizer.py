import torch
import torch.nn as nn

class InstanceNormalizer(nn.Module):
    def __init__(self, eps=1e-5):
        super(InstanceNormalizer, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        # Input dimensions: [batch, number_of_timesteps, num_features]
        # We want to get the mean and variance across the timestep dimension (dim=1)
        dims = list(range(1, x.dim()-1))
        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        normalized = (x - mean) / std
        return normalized, mean, std
    
    def denormalize(self, normalized_x, mean, std):
        return normalized_x * std + mean