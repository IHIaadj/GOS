
// All we need to understand PyTorch
#include <torch/csrc/jit/ir.h>
// CompleteArgumentSpec (useful for caching)
#include <torch/csrc/jit/argument_spec.h>


using CompiledCode = std::function<std::vector<torch::jit::IValue>(
    at::ArrayRef<torch::jit::IValue>&)>;

class TiramisuCompiler {
 public:
  PointwiseCompiler(const torch::jit::Node* node)
      : subgraph_(node->g(torch::jit::attr::Subgraph)) {}
  void run(torch::jit::Stack& stack);
  static bool supported(const torch::jit::Node* node);

};