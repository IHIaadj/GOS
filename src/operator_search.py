import torch 
import operator_search_layer  

def operator_search(model, data):
    # Analyze layer efficiency
    efficiency_list = analyze_efficiency(model)

    # Sort the efficiency list in ascending order
    sorted_efficiency = sorted(efficiency_list, key=lambda x: x[1])

    # Perform operator search for the least efficient layers
    for layer, efficiency in sorted_efficiency:
        optimized_layer = operator_search_layer(layer)
        # Replace the layer in the model with the optimized layer
        model = replace_layer(model, layer, optimized_layer)

    optimized_model = model
    return optimized_model

def replace_layer(model, old_layer, new_layer):
    # Helper function to replace a layer in the model with a new layer
    for name, module in model.named_modules():
        if module is old_layer:
            parent_module = get_parent_module(model, name)
            setattr(parent_module, name.split('.')[-1], new_layer)
            break
    return model

def get_parent_module(model, name):
    # Helper function to retrieve the parent module of a named module in the model
    parent_module = model
    for module_name in name.split('.')[:-1]:
        parent_module = getattr(parent_module, module_name)
    return parent_module
