/**
 * @file diagnose_weight_layout.cpp
 * @brief Simple test to understand weight matrix layout expectations
 * @author David Sanftenberg
 * 
 * This program loads a weight matrix and tests different orientations
 * to definitively determine the correct layout.
 */
#include <iostream>
#include <vector>
#include <cmath>
#include "src/model_loader.h"

int main() {
    // Load model
    llaminar::ModelLoader loader("models/qwen2.5-0.5b-instruct-q4_0.gguf");
    if (!loader.load()) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    auto weights = loader.getWeights();
    
    // Find output projection weight for layer 0
    std::shared_ptr<llaminar::Tensor> wo_tensor;
    for (const auto& w : weights->tensors) {
        if (w.name == "blk.0.attn_output.weight") {
            wo_tensor = w.tensor;
            break;
        }
    }
    
    if (!wo_tensor) {
        std::cerr << "Could not find blk.0.attn_output.weight" << std::endl;
        return 1;
    }
    
    std::cout << "=============================================" << std::endl;
    std::cout << "OUTPUT PROJECTION WEIGHT ANALYSIS" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Tensor name: blk.0.attn_output.weight" << std::endl;
    std::cout << "Shape: [" << wo_tensor->shape()[0] << ", " << wo_tensor->shape()[1] << "]" << std::endl;
    std::cout << std::endl;
    
    // Print first few values
    const float* data = wo_tensor->data();
    std::cout << "First 5x5 corner:" << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << std::fixed << std::setprecision(6) << data[i * 896 + j] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Expected from PyTorch reference:" << std::endl;
    std::cout << "PyTorch o_proj.weight has shape [896, 896] (out_features, in_features)" << std::endl;
    std::cout << "PyTorch forward: y = x @ weight.T" << std::endl;
    std::cout << "So weight.T would be [896, 896] with transposed data" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Llaminar scalar code expects:" << std::endl;
    std::cout << "B indexed as B[k * d_model + j] = B[k][j]" << std::endl;
    std::cout << "So B should be shape [K, N] = [total_head_dim, d_model] = [896, 896]" << std::endl;
    std::cout << "To match PyTorch's x @ weight.T, we need B = weight.T" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Therefore:" << std::endl;
    std::cout << "- If GGUF stores PyTorch's weight directly [out, in] = [896, 896]" << std::endl;
    std::cout << "- Llaminar needs to transpose it OR use transpose_B=true in BLAS" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Dimensions after loader:" << std::endl;
    std::cout << "  shape[0] = " << wo_tensor->shape()[0] << " (should be total_head_dim=896 for no-transpose use)" << std::endl;
    std::cout << "  shape[1] = " << wo_tensor->shape()[1] << " (should be d_model=896 for no-transpose use)" << std::endl;
    
    return 0;
}
