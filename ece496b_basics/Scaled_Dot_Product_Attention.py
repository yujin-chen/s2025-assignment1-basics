import torch
from torch.nn import Module, Dropout
from ece496b_basics import Softmax

class ScaledDotProductAttention(Module):
    def __init__(self, pdrop = None):  
        super().__init__()

        if pdrop is not None:
            self.dropout = Dropout(pdrop)

    #
    def forward(self, Q, K, V, mask=None):
        # Get dimensions
        d_k = Q.size(-1) 

        # Compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.to(dtype=torch.bool), -1e9) 

        # Apply softmax
        attn = Softmax.softmax(scores, dim=-1)

        return torch.matmul(attn, V)
