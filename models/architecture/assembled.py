import torch
import torch.nn as nn
from models.architecture.components.encoder import Encoder
from models.architecture.components.patcher import Patcher
from models.architecture.components.variable_splitter import VariableSplitter
from models.architecture.components.instance_normalizer import InstanceNormalizer
from models.architecture.components.positional_embedding import PositionalEmbedding
from models.architecture.components.projection import Projection

class AssembledModel(nn.Module):
    def __init__(self):
        super(AssembledModel, self).__init__()
        self.encoder = Encoder()
        self.patcher = Patcher()
        self.variable_splitter = VariableSplitter()
        self.projection = Projection()
        self.instance_normalizer = InstanceNormalizer()
        self.pos_embedder = PositionalEmbedding()
        
    def forward(self, x):
        x, mean, std = self.instance_normalizer(x)
        x = self.variable_splitter(x)
        x = self.patcher(x) 
        x = self.projection(x)
        x = self.pos_embedder(x)
        x = self.encoder(x)
        x = self.patcher(x)
        x = self.instance_normalizer.denormalize(x, mean, std)
        return x