import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
class PositionalEncoding(nn.Module):
    # Code from A2
    def __init__(self, d_model, max_seq_length=5000):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        max_seq_length: Maximum length of sequences input into the transformer.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).reshape(max_seq_length, 1)
        div_term = torch.exp( 
              -1 * (torch.arange(0, d_model, 2).float()/d_model) * math.log(10000.0)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Adds the positional encoding to the model input x.
        """
        return x + self.pe[:, : x.size(1)]
    
class LearnedEncoding(nn.Module):
    def __init__(self, patch_num, d_model, max_seq_length=5000):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        max_seq_length: Maximum length of sequences input into the transformer.
        """
        super(LearnedEncoding, self).__init__()
        
        encoding = torch.empty((1, patch_num, d_model))
        torch.nn.init.normal_(encoding, mean=0.0, std=0.1)
        self.encoding = nn.Parameter(encoding, requires_grad=True)

    def forward(self, x):
        """
        Return the sum plus the learned encoding.
        """
        return x + self.encoding