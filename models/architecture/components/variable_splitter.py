import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableSplitter(nn.Module):
    def __init__(self):
        super(VariableSplitter, self).__init__()
    
    # def forward(self, x):
    #     # Input dimensions: [batch, num_features, number_of_timesteps]
    #     # Output dimensions: [batch, 1, number_of_timesteps] * num_features
    #     univariate_series = torch.split(x, 1, dim=1)
    #     return univariate_series
    
    def forward(self, x):
        # Input dimensions: [batch, num_features, number_of_timesteps]
        # Output dimensions: [batch, 1, number_of_timesteps] * num_features
        return x.permute(1, 0, 2, 3)