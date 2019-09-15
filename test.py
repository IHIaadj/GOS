import torch
import torch.nn.functional as F 
import time 

def time_execution(f): 
    A = torch.randn(10000, 10000)
    t = time.time()
    for _ in range(100):
        _ = f(A)
    return time.time() - t 

@torch.jit.script
def test(t): 
    return F.relu_(t)

A = torch.randn(10000, 10000)
print("PyTorch IR  \n", test.graph_for(A))
print("Default version took {:.2f}ms".format(1000 * time_execution(test)))

import tiramisu_compiler

@torch.jit.script
def test_tiramisu(t): 
    return F.relu_(t)

print("Default version took {:.2f}ms".format(1000 * time_execution(test_tiramisu)))
    