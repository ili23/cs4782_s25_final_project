import torch
import torch.nn as nn
from models.architecture.components.encoder import Encoder
from models.architecture.components.patcher import Patcher
from models.architecture.components.variable_splitter import VariableSplitter
from models.architecture.components.instance_normalizer import InstanceNormalizer
from models.architecture.components.embedding import PositionalEncoding, LearnedEncoding
from models.architecture.components.projection import Projection
from models.architecture.components.flatten_head import FlattenHead

class AssembledModel(nn.Module):
    def __init__(self, 
                 seq_len,
                 input_dim=None, 
                 patch_length=16, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 target_window=24,
                 stride=8,
                 mlp_ratio=4.0):
        super(AssembledModel, self).__init__()
        
        # Initialize with proper parameters
        self.instance_normalizer = InstanceNormalizer()
        self.variable_splitter = VariableSplitter()
        self.patcher = Patcher(patch_length, stride=stride)
        self.projection = Projection(patch_dim=patch_length, d_dim=embed_dim)
        self.pos_embedder = PositionalEncoding(d_model=embed_dim)
        # self.pos_embedder = LearnedEncoding(patch_num=patch_length, d_model=embed_dim)
        # d_ff is the hidden dimension size for the feed-forward network, change later
        patch_num = int((seq_len - patch_length)/stride + 1) + 1
        self.flatten_head = FlattenHead(embed_dim * patch_num, target_window)
        self.encoder = Encoder(d_model=embed_dim, num_heads=num_heads, num_layers=depth, d_ff=embed_dim)

    def encode(self, x):
        print("x shape before projection:", len(x))
        embeddings = self.projection(x)
        # Tuple of num_features tensors size: [batch, 1, embedding_size, number_of_patches]
        print("embeddings shape after projection:", embeddings.shape)
        pos_embeddings = self.pos_embedder(embeddings)
        # Tuple of num_features tensors size: [batch, 1, embedding_size, number_of_patches]
        print("pos_embeddings shape after positional encoding:", pos_embeddings.shape)
        encoded = self.encoder(pos_embeddings)
        print("encoded shape after encoding:", encoded.shape)
        return encoded
        
    def forward(self, x):
        # Input dimensions: [batch, number_of_timesteps, num_features]
        # Rearranged dimensions: [batch, num_features, number_of_timesteps]
        b = x.shape[0]
        f = x.shape[2]
        x = x.permute(0,2,1)
        # Normalize input
        x, mean, std = self.instance_normalizer(x)
        print("x shape after normalization:", x.shape)
        patches = self.patcher(x)
        print("patches shape after patching:", patches.shape)
        # Shape should be: [batch, num_features, patch_length, number_of_patches]
        x = self.variable_splitter(patches)
        # Tuple of num_features tensors size: [batch, 1, patch_length, number_of_patches]
        print("x shape after variable splitting:", len(x))
        print("x[0] shape after variable splitting:", x[0].shape)

        # encoded = torch.vmap(self.encode, in_dims=(0), randomness="same")(x)
        encoded = self.encode(x)

        print("encoded shape after encoding:", encoded.shape)

        print("Mean shape after encoding:", mean.shape)
        print("Std shape after encoding:", std.shape)

        flattened = self.flatten_head.forward(encoded, b, f)

        output = self.instance_normalizer.denormalize(flattened, mean, std)
        print("output shape after denormalization:", output.shape)

        # x = x.permute(0,2,1)
        
        return output