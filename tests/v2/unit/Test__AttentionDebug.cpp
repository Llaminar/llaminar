#include <gtest/gtest.h>
#include "kernels/cpu/attention/CpuAttentionKernelT.h"
#include "tensors/Tensors.h"
#include <cmath>
#include <vector>
#include <fstream>

using namespace llaminar2;

// Simple NPY loader (FP32 only)
std::vector<float> loadNpy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    
    // Read magic
    char magic[6];
    file.read(magic, 6);
    
    // Read header length
    unsigned short header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    // Skip header
    file.seekg(10 + header_len, std::ios::beg);
    
    // Read rest as floats
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(10 + header_len, std::ios::beg);
    
    size_t data_size = file_size - (10 + header_len);
    size_t num_floats = data_size / sizeof(float);
    
    std::vector<float> data(num_floats);
    file.read(reinterpret_cast<char*>(data.data()), data_size);
    
    return data;
}

TEST(AttentionDebug, DirectKernelCall) {
    // Model params
    const int seq_len = 9;
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    
    // Load PyTorch data
    auto q_rope = loadNpy("pytorch_qwen2_snapshots/layer0_Q_ROPE.npy");
    auto k_rope = loadNpy("pytorch_qwen2_snapshots/layer0_K_ROPE.npy");
    auto v_proj = loadNpy("pytorch_qwen2_snapshots/layer0_V_PROJECTION.npy");
    auto expected = loadNpy("pytorch_qwen2_snapshots/layer0_ATTENTION_CONTEXT.npy");
    
    std::cout << "Q_ROPE size: " << q_rope.size() << " (expected: " << seq_len * n_heads * head_dim << ")" << std::endl;
    std::cout << "K_ROPE size: " << k_rope.size() << " (expected: " << seq_len * n_kv_heads * head_dim << ")" << std::endl;
    std::cout << "V_PROJ size: " << v_proj.size() << " (expected: " << seq_len * n_kv_heads * head_dim << ")" << std::endl;
    
    ASSERT_EQ(q_rope.size(), seq_len * n_heads * head_dim);
    ASSERT_EQ(k_rope.size(), seq_len * n_kv_heads * head_dim);
    ASSERT_EQ(v_proj.size(), seq_len * n_kv_heads * head_dim);
    
    // Create output buffer
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);
    
    // Create workspace buffers
    auto scores = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * seq_len)});
    
    // Create attention kernel
    CpuAttentionKernelT<FP32Tensor> kernel;
    
    // Call compute directly
    bool success = kernel.compute(
        q_rope.data(),
        k_rope.data(),
        v_proj.data(),
        output.data(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        false,  // causal
        -1,     // window_size
        scores.get(),
        nullptr,  // workspace_buffer
        nullptr,  // workspace_context
        nullptr,  // workspace_mask
        false,    // use_bf16
        nullptr,  // mpi_ctx
        -1        // device_idx
    );
    
    ASSERT_TRUE(success);
    
    // Compare output with expected
    float max_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    float sum_sq_expected = 0.0f;
    
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = std::abs(output[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        sum_sq_diff += (output[i] - expected[i]) * (output[i] - expected[i]);
        sum_sq_expected += expected[i] * expected[i];
    }
    
    float rel_l2 = std::sqrt(sum_sq_diff / sum_sq_expected);
    
    std::cout << "Max abs diff: " << max_diff << std::endl;
    std::cout << "Rel L2 norm: " << rel_l2 << std::endl;
    std::cout << "First 10 output: ";
    for (int i = 0; i < 10; ++i) std::cout << output[i] << " ";
    std::cout << std::endl;
    std::cout << "First 10 expected: ";
    for (int i = 0; i < 10; ++i) std::cout << expected[i] << " ";
    std::cout << std::endl;
    
    EXPECT_LT(rel_l2, 0.01f) << "Relative L2 norm too high";
}
