import torch
import computation_graph as cg 
import operator_search as os 
from utils import * 

def analyze_efficiency(model, data):
    # Implement layer efficiency analysis
    efficiency_list = []
    for name, module in model.named_modules():
        # Calculate efficiency metric for each module (layer)
        efficiency = utils.calculate_efficiency(module, data)
        efficiency_list.append((module, efficiency))
    return efficiency_list

# Example usage
if __name__ == "__main__":
    # Set up your model and data
    model = models.resnet18()
    data = torch.randn(1, 3, 224, 224)

    efficient_layers = analyze_efficiency(model, data)
    # Perform operator search
    for l in efficient_layers:
        optimized_model = os(l, data)

    print(optimized_model)
