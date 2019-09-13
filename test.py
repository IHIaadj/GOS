import torch

def time_execution(f): 
    A = torch.randn(10000, 10000, device=device)
    t = time.time()
    for _ in range(100):
        _ = f(A)
    return time.time() - t 

@torch.jit.script
def test(t): 
    return F.relu_(t)

print("PyTorch IR  \n", test.graph_for())
output = test(t)
print("Default version took {:.2f}ms".format(1000 * time_execution(test)))

import tiramisu_torch  

@torch.jit.script
def test_tiramisu(t): 
    return F.relu_(t)

output = test_tiramisu(t)
print("Default version took {:.2f}ms".format(1000 * time_execution(test_tiramisu)))
    