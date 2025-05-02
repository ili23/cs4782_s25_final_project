import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def compute_attention(self, Q, K, V):
        """
        Returns single-headed attention between Q, K, and V.
        """
        attention = None

        num = torch.matmul(Q, K.transpose(-2, -1))
        denom = np.sqrt(self.d_k)
        attention = torch.matmul(F.softmax(num / denom, dim=-1), V)

        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x):

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        multihead = self.compute_attention(Q, K, V)
        multihead = self.combine_heads(multihead)
        x = self.W_o(multihead)

        return x



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        # TODO 11: define the encoder layer
        #################

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)
        

    def forward(self, x):

        ## TODO 11: implement the forward function based on the architecture described above
        #################

        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff):
        """
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        encoder_layers = []
        for _ in range(num_layers):
          encoder_layers.append(EncoderLayer(d_model, num_heads, d_ff, p))
        self.encoder_layers = nn.ModuleList(encoder_layers)

        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        pass

class Transformer(nn.Module):
    def __init__(
        self, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, p
    ):
        """
        Inputs:
        num_classes: Number of classes in the classification output.
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        max_seq_length: Maximum sequence length accepted by the transformer.
        p: Dropout probability.
        """
        super(Transformer, self).__init__()

        # TODO 12: define the transformer
        #################

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(p)

        


    def forward(self, x):

        ## TODO 12: implement the forward pass
        #################

        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
          x = layer(x)
        
        x = torch.mean(x, dim=1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
