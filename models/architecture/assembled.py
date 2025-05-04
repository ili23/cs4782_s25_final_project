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
                 patch_length=16, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4.0):
        super(AssembledModel, self).__init__()
        
        # Initialize with proper parameters
        self.instance_normalizer = InstanceNormalizer()
        self.variable_splitter = VariableSplitter()
        self.patcher = Patcher(patch_length, stride=8)
        self.projection = Projection(patch_dim=patch_length, d_dim=embed_dim)
        self.pos_embedder = PositionalEncoding(d_model=embed_dim)
        # d_ff is the hidden dimension size for the feed-forward network, change later
        self.encoder = Encoder(d_model=embed_dim, num_heads=num_heads, num_layers=depth, d_ff=embed_dim)
        
    def forward(self, x):
        # Normalize input
        x, mean, std = self.instance_normalizer(x)
        print("x shape after normalization:", x.shape)
        x = self.variable_splitter(x)
        print("x shape after variable splitting:", x.shape)
        patches = self.patcher(x)
        print("patches shape after patching:", patches.shape)
        embeddings = self.projection(patches)
        print("embeddings shape after projection:", embeddings.shape)
        pos_embeddings = self.pos_embedder(embeddings)
        print("pos_embeddings shape after positional encoding:", pos_embeddings.shape)
        encoded = self.encoder(pos_embeddings)
        print("encoded shape after encoding:", encoded.shape)
        output = self.instance_normalizer.denormalize(encoded, mean, std)
        print("output shape after denormalization:", output.shape)
        
        return output