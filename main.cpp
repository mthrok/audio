#include <torch/script.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc !=4) {
    std::cerr << "Usage: " << argv[0] << " <JIT_OBJECT> <INPUT_FILE> <OUTPUT_FILE>" << std::endl;
    return -1;
  }

  const std::string module_path = argv[1];
  const std::string input_path = argv[2];
  const std::string output_path = argv[3];

  torch::jit::script::Module module;
  std::cout << "Loading module from: " << module_path << std::endl;
  try {
    module = torch::jit::load(module_path);
  } catch (const c10::Error &error) {
    std::cerr << "Failed to load the module:" << error.what() << std::endl;
    return -1;
  }

  std::cout << "Performing separation ..." << std::endl; 
  std::vector<torch::jit::IValue> args{c10::IValue(input_path), c10::IValue(output_path)};
  module.forward(args);
  std::cout << "Done." << std::endl; 
}
