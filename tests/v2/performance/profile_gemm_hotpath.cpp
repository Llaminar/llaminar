/**
 * @file profile_gemm_hotpath.cpp
 * @brief Minimal profiling harness to identify V2 GEMM bottleneck
 *
 * Isolates key components to measure overhead:
 * 1. Decode performance (virtual vs direct)
 * 2. Dot product performance (SIMD efficiency)
 * 3. Memory access patterns (cache behavior)
 *
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "../../src/v2/tensors/IQ4_NLTensor.h"
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/tensors/IQ4_NLTensor.h"
#include "../../../src/v2/loaders/ModelLoader.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace llaminar2;

// Direct decode test (no virtual dispatch)
double benchmark_direct_decode(IQ4_NLTensor *tensor, int n, int k, int iters)
{
    const int num_k_blocks = (k + 31) / 32;
    alignas(64) float buffer[32];

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iters; ++iter)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int kb = 0; kb < num_k_blocks; ++kb)
            {
                tensor->decode_block_at(j, kb, buffer);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Virtual decode test (IBlockDecoder interface)
double benchmark_virtual_decode(const IBlockDecoder *decoder, int n, int k, int iters)
{
    const int num_k_blocks = (k + 31) / 32;
    alignas(64) float buffer[32];

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iters; ++iter)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int kb = 0; kb < num_k_blocks; ++kb)
            {
                decoder->decode_block_at(j, kb, buffer);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Dot product benchmark
double benchmark_dot_product(int size, int iters)
{
    std::vector<float> a(size), b(size);
    for (int i = 0; i < size; ++i)
    {
        a[i] = 0.01f * i;
        b[i] = 0.02f * i;
    }

    float sum = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iters; ++iter)
    {
        for (int i = 0; i < size; ++i)
        {
            sum += a[i] * b[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Prevent optimization
    if (sum < -1e10f)
        std::cout << sum;

    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main()
{
    std::cout << "=== V2 GEMM Hotpath Profiling ===\n\n";

    // Load model
    ModelLoader loader;
    if (!loader.loadModel("/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q4_0.gguf"))
    {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    // Get Q-projection weight (896x896)
    auto tensor = loader.loadTensor("blk.0.attn_q.weight");
    auto *iq4_tensor = dynamic_cast<IQ4_NLTensor *>(tensor.get());
    if (!iq4_tensor)
    {
        std::cerr << "Not an IQ4_NL tensor\n";
        return 1;
    }

    const int n = 896; // out_features
    const int k = 896; // in_features
    const int decode_iters = 100;
    const int dot_iters = 100000;

    std::cout << "Configuration:\n";
    std::cout << "  Matrix size: " << n << " x " << k << "\n";
    std::cout << "  Decode iterations: " << decode_iters << "\n";
    std::cout << "  Dot product iterations: " << dot_iters << "\n\n";

    // Test 1: Direct decode (concrete type)
    double direct_time = benchmark_direct_decode(iq4_tensor, n, k, decode_iters);
    double direct_mblocks_per_sec = (decode_iters * n * ((k + 31) / 32)) / (direct_time / 1000.0) / 1e6;

    std::cout << "Test 1: Direct Decode (IQ4_NLTensor*)\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << direct_time << " ms\n";
    std::cout << "  Throughput: " << direct_mblocks_per_sec << " Mblocks/sec\n\n";

    // Test 2: Virtual decode (IBlockDecoder interface)
    const IBlockDecoder *decoder = iq4_tensor;
    double virtual_time = benchmark_virtual_decode(decoder, n, k, decode_iters);
    double virtual_mblocks_per_sec = (decode_iters * n * ((k + 31) / 32)) / (virtual_time / 1000.0) / 1e6;

    std::cout << "Test 2: Virtual Decode (IBlockDecoder*)\n";
    std::cout << "  Time: " << virtual_time << " ms\n";
    std::cout << "  Throughput: " << virtual_mblocks_per_sec << " Mblocks/sec\n";
    std::cout << "  Overhead: " << std::setprecision(1) << (virtual_time / direct_time - 1.0) * 100.0 << "%\n\n";

    // Test 3: Dot product (SIMD efficiency)
    double dot_time = benchmark_dot_product(896, dot_iters);
    double dot_gflops = (dot_iters * 896 * 2) / (dot_time / 1000.0) / 1e9;

    std::cout << "Test 3: Dot Product (896 elements)\n";
    std::cout << "  Time: " << std::setprecision(2) << dot_time << " ms\n";
    std::cout << "  Throughput: " << dot_gflops << " GFLOPS\n\n";

    std::cout << "=== Analysis ===\n";
    std::cout << "Virtual dispatch overhead: ";
    if (virtual_time / direct_time > 1.1)
    {
        std::cout << "SIGNIFICANT (" << std::setprecision(0) << (virtual_time / direct_time - 1.0) * 100.0 << "%) - likely bottleneck\n";
    }
    else
    {
        std::cout << "NEGLIGIBLE (" << std::setprecision(1) << (virtual_time / direct_time - 1.0) * 100.0 << "%)\n";
    }

    return 0;
}
