import torch

def gradient_clipping(params, max_norm, eps=1e-6):
    # Compute total gradient norm across all parameters
    total_norm = torch.sqrt(sum((p.grad.data.norm(2) ** 2 for p in params if p.grad is not None)))

    #scale gradients
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + eps) 
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale_factor) 

