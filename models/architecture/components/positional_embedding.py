import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, n_dim, d_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, N, D))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
    
    def forward(self, x):
        return x + self.pos_emb