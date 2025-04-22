#include <torch/script.h>
#include <iostream>
#include <memory>

int main() {
    try {
        // Load the TorchScript model
        auto model = torch::jit::load("/OMNNSim.pt");
        model.eval();

        // Example input tensor: 1 photon with 7 features
        std::vector<float> photon_data = {400.0, 33.01, 65.313, 218.05, -0.047131, 0.019522, -0.9987};
        auto input_tensor = torch::from_blob(photon_data.data(), {1, 7}).clone();

        // Skip CUDA checks and force CPU
        std::cout << "Using CPU." << std::endl;
        input_tensor = input_tensor.to(torch::kCPU);
        model.to(torch::kCPU);

        // Run the model
        auto output = model.forward({input_tensor}).toTensor();
        std::cout << "Model output:\n" << output << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model or running inference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
