import torch

def softmax(x, dim):
    #Find the max value
    x_max = torch.max(x, dim=dim, keepdim=True)[0]  # Subtract max for numerical stability
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)