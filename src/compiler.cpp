
#include "compiler.h"
#include <torch/csrc/jit/interpreter.h>

#include "relu_layer_generator_tiramisu.o.h"
#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <chrono>
#include <string>
#include <time.h>
#include <iostream>
#include "configure.h"

#include <stack>

using namespace torch::jit;

bool TiramisuCompiler::supported(const torch::jit::Node* node) {
  switch (node->kind()) {
    case aten::relu:
      return true;
    default:
      return false;
  }
  return false;
}

void TiramisuCompiler::run(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<IValue> inputs = last(stack, num_inputs);

  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};
  if (cache_.find(spec) == cache_.end()) {
    cache_[spec] = compile(inputs);
  }

  // Run the compiled function!
  auto outputs = cache_[spec](inputs);

  drop(stack, num_inputs);
  for (auto& output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(IValue(var));
  }
}

void runOnFailure(torch::jit::Stack& stack) {
  torch::jit::InterpreterState(torch::jit::Code(subgraph_)).run(stack);
}

CompiledCode TiramisuCompiler::compile(
    at::ArrayRef<torch::jit::IValue>& inputs) {
        // Check the inputs 
        TORCH_CHECK(inputs.size(), "Need at least one input.");
        for (const auto& input : inputs) {
            TORCH_CHECK(input.isTensor(), "Compiler can only handle Tensor inputs.");
        }
        auto size = inputs[0].toTensor().numel();
        for (const auto& input : inputs) {
            TORCH_CHECK(
                input.toTensor().numel() == size,
                "Compiler can only handle pointwise operations without broadcasting.");
        } 
        



