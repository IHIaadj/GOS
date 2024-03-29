
#include "compiler.h"
#include <torch/csrc/jit/interpreter.h>

#include <cstdlib>
#include <chrono>
#include <string>
#include <time.h>
#include <iostream>
#include <stack>

using namespace torch::jit;

bool GOS::supported(const torch::jit::Node* node) {
  switch (node->kind()) {
    case aten::relu:
      return true;
    default:
      return false;
  }
  return false;
}

void GOS::run(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<IValue> inputs = last(stack, num_inputs);

  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};
  if (cache_.find(spec) == cache_.end()) {
      TORCH_CHECK(inputs.size(), "Need at least one input.");
        for (const auto& input : inputs) {
            TORCH_CHECK(input.isTensor(), "GOS only handles Tensor inputs.");
        }
        auto size = inputs[0].toTensor().numel();
        for (const auto& input : inputs) {
            TORCH_CHECK(
                input.toTensor().numel() == size,
                "GOS cannot handle broadcasting");
        } 

        std::cout << subgraph_->outputs()[0] << std::endl; 
  }

  // Run the compiled function!
  auto outputs = subgraph_->outputs(); 

  drop(stack, num_inputs);
  
}


