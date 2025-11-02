/**
 * @file Test__IQ4_NL_BF16_Decoder.cpp
 * @brief Unit test for IQ4_NL BF16 decoder correctness
 * 
 * @author David Sanftenberg
 * @date November 2, 2025
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../src/v2/kernels/cuda/IQ4_NL_BlockDecoder.h"
#include <vector>
#include <cmath>

using namespace llaminar2::cuda;

/**
 * @brief Test fixture for IQ4_NL BF16 decoder tests
 */
class Test__IQ4_NL_BF16_Decoder : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if BF16 is supported
        int device = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        if (prop.major < 8) {
            GTEST_SKIP() << "BF16 requires Compute Capability >= 8.0 (Ampere+), found SM " 
                         << prop.major << "." << prop.minor;
        }
    }
};

/**
 * @brief Simple kernel to test BF16 decoder on device
 */
__global__ void test_bf16_decoder_kernel(
    const IQ4_NLBlock* blocks,
    __nv_bfloat16* output_bf16,
    __half* output_fp16,
    float* output_fp32,
    int num_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_blocks) {
        IQ4_NL_Decoder<IQ4_NLBlock> decoder(blocks, num_blocks, 1);
        
        const IQ4_NLBlock* block = &blocks[tid];
        
        // Decode to BF16
        decoder.decode_block_bf16(block, &output_bf16[tid * 32]);
        
        // Decode to FP16 (reference)
        decoder.decode_block_fp16(block, &output_fp16[tid * 32]);
        
        // Decode to FP32 (ground truth)
        decoder.decode_block(block, &output_fp32[tid * 32]);
    }
}

/**
 * @brief Test that BF16 decoder produces reasonable results compared to FP32
 */
TEST_F(Test__IQ4_NL_BF16_Decoder, BasicCorrectness) {
    // Create a simple test block with known scale and quantized values
    IQ4_NLBlock h_block;
    
    // Set scale to 1.0 (as FP16)
    __half scale = __float2half(1.0f);
    h_block.d = *reinterpret_cast<uint16_t*>(&scale);
    
    // Fill with simple quantized values (0-15 pattern)
    for (int i = 0; i < 16; ++i) {
        h_block.qs[i] = (i & 0x0F) | ((i & 0x0F) << 4);  // Low and high nibbles same
    }
    
    // Allocate device memory
    IQ4_NLBlock* d_block;
    __nv_bfloat16* d_output_bf16;
    __half* d_output_fp16;
    float* d_output_fp32;
    
    cudaMalloc(&d_block, sizeof(IQ4_NLBlock));
    cudaMalloc(&d_output_bf16, 32 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_fp16, 32 * sizeof(__half));
    cudaMalloc(&d_output_fp32, 32 * sizeof(float));
    
    // Copy block to device
    cudaMemcpy(d_block, &h_block, sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    
    // Launch kernel
    test_bf16_decoder_kernel<<<1, 1>>>(d_block, d_output_bf16, d_output_fp16, d_output_fp32, 1);
    cudaDeviceSynchronize();
    
    // Copy results back
    std::vector<__nv_bfloat16> h_output_bf16(32);
    std::vector<__half> h_output_fp16(32);
    std::vector<float> h_output_fp32(32);
    
    cudaMemcpy(h_output_bf16.data(), d_output_bf16, 32 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fp16.data(), d_output_fp16, 32 * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fp32.data(), d_output_fp32, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify BF16 results match FP32 ground truth within tolerance
    for (int i = 0; i < 32; ++i) {
        float bf16_val = __bfloat162float(h_output_bf16[i]);
        float fp16_val = __half2float(h_output_fp16[i]);
        float fp32_val = h_output_fp32[i];
        
        // BF16 has ~7-bit mantissa, so tolerance should be ~1e-2
        float bf16_error = std::abs(bf16_val - fp32_val);
        float relative_bf16_error = bf16_error / (std::abs(fp32_val) + 1e-8f);
        
        EXPECT_LT(relative_bf16_error, 0.02f) 
            << "BF16 decoder output[" << i << "] = " << bf16_val 
            << " differs from FP32 ground truth " << fp32_val
            << " by " << (relative_bf16_error * 100.0f) << "%";
        
        // Verify FP16 is more precise than BF16
        float fp16_error = std::abs(fp16_val - fp32_val);
        EXPECT_LE(fp16_error, bf16_error) 
            << "FP16 should be more precise than BF16 (10-bit vs 7-bit mantissa)";
    }
    
    // Cleanup
    cudaFree(d_block);
    cudaFree(d_output_bf16);
    cudaFree(d_output_fp16);
    cudaFree(d_output_fp32);
}

/**
 * @brief Test BF16 decoder with various scale factors
 */
TEST_F(Test__IQ4_NL_BF16_Decoder, VariousScales) {
    std::vector<float> test_scales = {0.001f, 0.1f, 1.0f, 10.0f, 100.0f};
    
    for (float scale_val : test_scales) {
        IQ4_NLBlock h_block;
        
        // Set scale
        __half scale = __float2half(scale_val);
        h_block.d = *reinterpret_cast<uint16_t*>(&scale);
        
        // Fill with pattern
        for (int i = 0; i < 16; ++i) {
            h_block.qs[i] = i | (i << 4);
        }
        
        // Allocate device memory
        IQ4_NLBlock* d_block;
        __nv_bfloat16* d_output_bf16;
        float* d_output_fp32;
        
        cudaMalloc(&d_block, sizeof(IQ4_NLBlock));
        cudaMalloc(&d_output_bf16, 32 * sizeof(__nv_bfloat16));
        cudaMalloc(&d_output_fp32, 32 * sizeof(float));
        
        cudaMemcpy(d_block, &h_block, sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
        
        // Launch kernel
        test_bf16_decoder_kernel<<<1, 1>>>(d_block, d_output_bf16, nullptr, d_output_fp32, 1);
        cudaDeviceSynchronize();
        
        // Copy results
        std::vector<__nv_bfloat16> h_output_bf16(32);
        std::vector<float> h_output_fp32(32);
        
        cudaMemcpy(h_output_bf16.data(), d_output_bf16, 32 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_fp32.data(), d_output_fp32, 32 * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Verify
        for (int i = 0; i < 32; ++i) {
            float bf16_val = __bfloat162float(h_output_bf16[i]);
            float fp32_val = h_output_fp32[i];
            
            float relative_error = std::abs(bf16_val - fp32_val) / (std::abs(fp32_val) + 1e-8f);
            
            EXPECT_LT(relative_error, 0.02f)
                << "Scale=" << scale_val << ", output[" << i << "] BF16=" << bf16_val
                << " vs FP32=" << fp32_val;
        }
        
        cudaFree(d_block);
        cudaFree(d_output_bf16);
        cudaFree(d_output_fp32);
    }
}

/**
 * @brief Test that BF16 has wider dynamic range than FP16
 */
TEST_F(Test__IQ4_NL_BF16_Decoder, DynamicRange) {
    // Test large scale factor that might overflow FP16
    float large_scale = 50000.0f;  // Near FP16 max (~65504)
    
    IQ4_NLBlock h_block;
    __half scale = __float2half(large_scale);
    h_block.d = *reinterpret_cast<uint16_t*>(&scale);
    
    // Use small quantized values
    for (int i = 0; i < 16; ++i) {
        h_block.qs[i] = 1 | (1 << 4);  // Small values
    }
    
    IQ4_NLBlock* d_block;
    __nv_bfloat16* d_output_bf16;
    __half* d_output_fp16;
    
    cudaMalloc(&d_block, sizeof(IQ4_NLBlock));
    cudaMalloc(&d_output_bf16, 32 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_fp16, 32 * sizeof(__half));
    
    cudaMemcpy(d_block, &h_block, sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    
    test_bf16_decoder_kernel<<<1, 1>>>(d_block, d_output_bf16, d_output_fp16, nullptr, 1);
    cudaDeviceSynchronize();
    
    std::vector<__nv_bfloat16> h_output_bf16(32);
    std::vector<__half> h_output_fp16(32);
    
    cudaMemcpy(h_output_bf16.data(), d_output_bf16, 32 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fp16.data(), d_output_fp16, 32 * sizeof(__half), cudaMemcpyDeviceToHost);
    
    // Check for infinities or NaNs
    for (int i = 0; i < 32; ++i) {
        float bf16_val = __bfloat162float(h_output_bf16[i]);
        float fp16_val = __half2float(h_output_fp16[i]);
        
        // BF16 should handle this better (wider exponent range)
        EXPECT_FALSE(std::isinf(bf16_val)) << "BF16 output[" << i << "] is inf";
        EXPECT_FALSE(std::isnan(bf16_val)) << "BF16 output[" << i << "] is NaN";
        
        // FP16 might overflow (this is informational, not a failure)
        if (std::isinf(fp16_val)) {
            std::cout << "INFO: FP16 overflowed at output[" << i << "] "
                      << "while BF16=" << bf16_val << " is finite\n";
        }
    }
    
    cudaFree(d_block);
    cudaFree(d_output_bf16);
    cudaFree(d_output_fp16);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
