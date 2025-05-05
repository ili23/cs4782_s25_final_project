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
                 depth=4, 
                 num_heads=12, 
                 pred_len=24,
                 stride=8,
                 mlp_ratio=4.0,
                 dataset=None,
                 output_dim=None):  # Make output_dim optional
        super().__init__()
        
        # Initialize with proper parameters
        self.instance_normalizer = InstanceNormalizer()
        self.variable_splitter = VariableSplitter()
        self.patcher = Patcher(patch_length, stride=stride)
        self.projection = Projection(patch_dim=patch_length, d_dim=embed_dim)
        self.pos_embedder = PositionalEncoding(d_model=embed_dim)
        # self.pos_embedder = LearnedEncoding(patch_num=patch_length, d_model=embed_dim)
        # d_ff is the hidden dimension size for the feed-forward network, change later
        patch_num = int((seq_len - patch_length)/stride + 1) + 1
        print("Number of patches: ", patch_num)
        self.flatten_head = FlattenHead(embed_dim * patch_num, pred_len)
        self.encoder = Encoder(d_model=embed_dim, num_heads=num_heads, num_layers=depth, d_ff=embed_dim)
        
        # Store the output dimension for later use
        self.output_dim = output_dim
        self.feature_projection = None  # Will be created in forward if needed
        
        self.set_dataset(dataset)  # Store reference to dataset for inverse scaling

    def set_dataset(self, dataset):
        """Set the dataset reference to access the scaler"""
        self.dataset = dataset
        # Move scaler parameters to the same device as the model
        if hasattr(dataset, 'scaler') and dataset.scaler is not None:
            if hasattr(dataset.scaler, 'mean') and dataset.scaler.mean is not None:
                device = next(self.parameters()).device
                dataset.scaler.mean = dataset.scaler.mean.to(device)
                dataset.scaler.std = dataset.scaler.std.to(device)
                dataset.scaler.train_device = device  # Update the train_device attribute

    def encode(self, x):
        # print("x shape before projection:", len(x))
        embeddings = self.projection(x)
        # Tuple of num_features tensors size: [batch, 1, embedding_size, number_of_patches]
        # print("embeddings shape after projection:", embeddings.shape)
        pos_embeddings = self.pos_embedder(embeddings)
        # Tuple of num_features tensors size: [batch, 1, embedding_size, number_of_patches]
        # print("pos_embeddings shape after positional encoding:", pos_embeddings.shape)
        encoded = self.encoder(pos_embeddings)
        # print("encoded shape after encoding:", encoded.shape)
        return encoded
        
    def forward(self, x):
        # Input dimensions: [batch, number_of_timesteps, num_features]
        # Rearranged dimensions: [batch, num_features, number_of_timesteps]
        b = x.shape[0]
        f = x.shape[2]  # Number of input features
        x = x.permute(0,2,1)
        # Normalize input
        x, mean, std = self.instance_normalizer(x)
        patches = self.patcher(x)
        x = self.variable_splitter(patches)
        encoded = self.encode(x)
        flattened = self.flatten_head.forward(encoded, b, f)
        output = self.instance_normalizer.denormalize(flattened, mean, std)
        
        # Permute output to match target shape [batch, pred_len, features]
        output = output.permute(0,2,1)
        
        # If output_dim is specified and different from input features, create a projection layer
        if self.output_dim is not None and self.output_dim != f:
            if self.feature_projection is None:
                # Create the projection layer on first forward pass when we know the input size
                self.feature_projection = nn.Linear(f, self.output_dim).to(output.device)
                print(f"Created projection layer from {f} to {self.output_dim} features")
            output = self.feature_projection(output)
        
        # Apply inverse transform from StandardScaler if dataset is available
        if self.dataset is not None and hasattr(self.dataset, 'scaler') and self.dataset.scaler is not None:
            device = output.device
            # Ensure device consistency
            if hasattr(self.dataset.scaler, 'mean') and self.dataset.scaler.mean is not None:
                if self.dataset.scaler.mean.device != device:
                    self.dataset.scaler.mean = self.dataset.scaler.mean.to(device)
                if self.dataset.scaler.std.device != device:
                    self.dataset.scaler.std = self.dataset.scaler.std.to(device)
            # We can directly apply inverse_transform on GPU tensors now
            output = self.dataset.inverse_transform(output)
        
        return output