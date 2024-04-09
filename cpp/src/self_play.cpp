#include <torch/script.h>

#include <iostream>
#include <memory>
#include <cassert>

int main() { 
    torch::Device device { torch::kCPU };

    try {
        std::string path = "./data/models/dragon/traced_dragon_iteration_80.pt";

        auto model = std::make_shared<torch::jit::script::Module>(torch::jit::load(path));
        model->to(device);

        std::cout << "Connect Four Network loaded successfully." << std::endl;

        auto input = torch::zeros({1, 2, 6, 7}).to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto output = model->forward(inputs).toTuple();

        auto policy_output = output->elements()[0].toTensor();
        auto value_output = output->elements()[1].toTensor();

        std::cout << "Policy output: " << policy_output << std::endl;
        std::cout << "Value output: " << value_output << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}