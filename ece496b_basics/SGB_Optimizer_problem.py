from collections.abc import Callable
from typing import Optional
import torch
import math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] 
            t = state.get("t", 0) 
            grad = p.grad.data 
            p.data -= lr / math.sqrt(t + 1) * grad # 
            state["t"] = t + 1 
        return loss
    
def test_optimizer(learning_rates):

    for lr in learning_rates:
        print(f"\n=== Testing Learning Rate: {lr} ===")
        
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)

        for t in range(10): 
            opt.zero_grad() 
            loss = (weights**2).mean()  
            print(f"Step {t}: Loss = {loss.item():.6f}")
            loss.backward()  
            opt.step()  


if __name__ == "__main__":
    learning_rates = [1e-3, 1e1, 1e2, 1e3]  
    test_optimizer(learning_rates)

'''
=== Testing Learning Rate: 0.001 ===
Step 0: Loss = 26.598076
Step 1: Loss = 26.597010
Step 2: Loss = 26.596258
Step 3: Loss = 26.595642
Step 4: Loss = 26.595112
Step 5: Loss = 26.594633
Step 6: Loss = 26.594200
Step 7: Loss = 26.593800
Step 8: Loss = 26.593422
Step 9: Loss = 26.593071

=== Testing Learning Rate: 10.0 ===
Step 0: Loss = 18.303894
Step 1: Loss = 11.714490
Step 2: Loss = 8.635423
Step 3: Loss = 6.756296
Step 4: Loss = 5.472600
Step 5: Loss = 4.537412
Step 6: Loss = 3.826705
Step 7: Loss = 3.270029
Step 8: Loss = 2.823927
Step 9: Loss = 2.459954

=== Testing Learning Rate: 100.0 ===
Step 0: Loss = 21.140680
Step 1: Loss = 21.140680
Step 2: Loss = 3.627167
Step 3: Loss = 0.086806
Step 4: Loss = 0.000000
Step 5: Loss = 0.000000
Step 6: Loss = 0.000000
Step 7: Loss = 0.000000
Step 8: Loss = 0.000000
Step 9: Loss = 0.000000

=== Testing Learning Rate: 1000.0 ===
Step 0: Loss = 31.537508
Step 1: Loss = 11385.040039
Step 2: Loss = 1966375.500000
Step 3: Loss = 218738272.000000
Step 4: Loss = 17717800960.000000
Step 5: Loss = 1118196072448.000000
Step 6: Loss = 57404536389632.000000
Step 7: Loss = 2469788462874624.000000
Step 8: Loss = 91031144643952640.000000
Step 9: Loss = 2923111036070395904.000000
    '''