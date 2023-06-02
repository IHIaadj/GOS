#include <pybind11/pybind11.h>

#include <torch/csrc/jit/custom_operator.h>


// Register a pass to convert the PyTorch IR to Tiramsu One 
#include <torch/csrc/jit/pass_manager.h>
// CustomFuseGraph is a helper to use simple whitelisting
#include <torch/csrc/jit/passes/graph_fuser.h>

#include "compiler.h"

namespace py = pybind11;
using namespace torch::jit;

PYBIND11_MODULE(gos, m) {

    auto options = c10::OperatorOptions();
    options.setAliasAnalysis(AliasAnalysisKind::PURE_FUNCTION);
    
    const auto gos_symbol =
      Symbol::fromQualString("pw::CompilationGroup");

    
    RegisterPass pass([gos_symbol](std::shared_ptr<Graph>& g) {
        CustomFuseGraph(g, TiramisuCompiler::supported, gos_symbol);
    });

    RegisterOperators op({Operator(
      gos_symbol,
      [](const Node* node) {
        auto compiler = std::make_shared<TiramisuCompiler>(node);
        return [compiler](Stack& stack) {
          compiler->run(stack);
          return 0;
        };
      },
      options)});

}


    