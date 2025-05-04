import torch
import torch.nn as nn
import torch.nn.functional as F

class Patcher(nn.Module):
    def __init__(self, patch_length = 16, stride = 8):
        super(Patcher, self).__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
    
    def forward(self, x):
        # Input dimensions: [batch, 1, number_of_timesteps]
        # Output dimensions: [batch, patch_length, number_of_patches]
        
        seq_len = x.shape[2]
        patch_num = int((seq_len - self.patch_length) / self.stride + 1)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
            patch_num += 1
        
        output = x.unfold(dimension=2, size=self.patch_length, step=self.stride)
        output = output.squeeze(1)                    
        output = output.permute(0, 2, 1)
        
        return output