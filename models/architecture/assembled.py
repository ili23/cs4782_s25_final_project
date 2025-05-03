import torch
import torch.nn as nn
from models.architecture.components.encoder import Encoder
from models.architecture.components.patcher import Patcher
from models.architecture.components.variable_splitter import VariableSplitter
from models.architecture.components.instance_normalizer import InstanceNormalizer
from models.architecture.components.embedding import PositionalEncoding
from models.architecture.components.projection import Projection

class AssembledModel(nn.Module):
    def __init__(self, 
                 input_dim=None, 
                 patch_size=16, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4.0):
        super(AssembledModel, self).__init__()
        
        # Initialize with proper parameters
        self.instance_normalizer = InstanceNormalizer()
        self.variable_splitter = VariableSplitter()
        self.patcher = Patcher(patch_size=patch_size)
        self.projection = Projection(in_features=patch_size**2, out_features=embed_dim)
        self.pos_embedder = PositionalEncoding(embed_dim=embed_dim)
        self.encoder = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_ratio=mlp_ratio)
        
    def forward(self, x):
        # Normalize input
        x, mean, std = self.instance_normalizer(x)
        x = self.variable_splitter(x)
        patches = self.patcher(x)
        embeddings = self.projection(patches)
        pos_embeddings = self.pos_embedder(embeddings)
        encoded = self.encoder(pos_embeddings)
        output = self.instance_normalizer.denormalize(encoded, mean, std)
        
        return output