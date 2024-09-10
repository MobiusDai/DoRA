import numpy as np 

# Debugging
step = 0
sigmoid_value = []

# Forward pass
for i in range(10):
    step += 1
    sigmoid_value.append(1/(1+np.exp(-i)))
    print(f"Step {step}: {sigmoid_value[-1]}")

import torch 

@torch.no_grad()
def test():
    # Forward process
    
    for i in range(10):
        step += 1
        sigmoid_value.append(1/(1+torch.exp(-i)))
        print(f"Step {step}: {sigmoid_value[-1]}")