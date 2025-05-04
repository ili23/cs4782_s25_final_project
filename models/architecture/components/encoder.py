# This code is based off of assignment 2 of our class
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

    def compute_attention(self, Q, K, V, mask=None):
        """
        Returns attention between Q, K, and V with optional masking.
        """
        # MatMul Q, K
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale
        scores = scores / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # SoftMax
        attention_weights = F.softmax(scores, dim=-1)
        
        # MatMul with V
        attention = torch.matmul(attention_weights, V)
        
        return attention, attention_weights

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x, mask=None):
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Compute attention
        multihead, attention_weights = self.compute_attention(Q, K, V, mask)
        
        # Combine heads
        multihead = self.combine_heads(multihead)
        
        # Final linear projection
        output = self.W_o(multihead)

        return output


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
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.BatchNorm1d(d_model, track_running_stats=True)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.BatchNorm1d(d_model, track_running_stats=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and normalization
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm1d expects [batch, channels, length]
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm1d expects [batch, channels, length]

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        """
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        dropout: Dropout rate
        """
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
