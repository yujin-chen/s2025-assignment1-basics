import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, weights=None):
        super().__init__()
        self.eps = eps

        # Create a parameter for the weight 
        self.weight = nn.Parameter(torch.ones(d_model))

        # Initialize weights
        if isinstance(weights, dict) and "weight" in weights and weights["weight"] is not None:
            with torch.no_grad():
                self.weight.copy_(weights["weight"])

    def forward(self, x):
        # Compute root mean square (RMS) of input
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize input by RMS
        return self.weight * (x / rms)
