import torch
from torch.nn import Module, Linear

# GELU
def gelu(x):
    return x * 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

class PositionwiseFeedForward(Module):
    def __init__(self, d_model, d_ff, weights=None):
        super().__init__()
        
        # Create Linear layers
        self.fc1 = Linear(d_model, d_ff, bias=False)
        self.fc2 = Linear(d_ff, d_model, bias=False)

        # Initialize weights
        if isinstance(weights, dict):
            if "w1.weight" in weights:
                with torch.no_grad():
                    self.fc1.weight.copy_(weights["w1.weight"])
            if "w2.weight" in weights:
                with torch.no_grad():
                    self.fc2.weight.copy_(weights["w2.weight"])
                    
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))
