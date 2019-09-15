import torch
import torch.nn.functional as F 
import time 
import tiramisu_compiler

def time_execution(f): 
    A = torch.randn(10000, 10000)
    t = time.time()
    rel  = torch.zeros(1)
    for _ in range(100):
        _ = f(A, rel)
    return time.time() - t 

@torch.jit.script
def test(A, rel):
    if rel : 
        return F.relu_(A)
    else : 
        return torch.randn(10000, 10000)

A = torch.randn(10000, 10000)
rel = torch.zeros(1)
print("PyTorch IR  \n", test.graph_for(A, rel ))
test(A, rel)

