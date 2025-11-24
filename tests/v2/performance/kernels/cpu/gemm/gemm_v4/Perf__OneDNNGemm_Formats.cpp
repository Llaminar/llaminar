/**
 * @file Perf__OneDNNGemm_Formats.cpp
 * @brief Benchmarks OneDNN GEMM performance across different formats (FP32, FP16, BF16, Q4_0xQ8_1).
 *
 * Target: Qwen 7B sized layers.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <memory>

#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;
using namespace std::chrono;

namespace
{

    // Qwen 7B Dimensions
    constexpr int D_MODEL = 4096;
    constexpr int INTERMEDIATE_SIZE = 11008;

    // Benchmark parameters
    constexpr int WARMUP_ITERS = 5;
    constexpr int BENCH_ITERS = 20;

    // Helper to generate random float data
    std::vector<float> generate_random_data(size_t size)
    {
        std::vector<float> data(size);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : data)
            v = dist(rng);
        return data;
    }

    // Helper to create FP32 Tensor
    std::unique_ptr<FP32Tensor> create_fp32_tensor(int rows, int cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto data = generate_random_data(rows * cols);
        std::memcpy(tensor->mutable_data(), data.data(), data.size() * sizeof(float));
        return tensor;
    }

    // Helper to create FP16 Tensor
    std::unique_ptr<FP16Tensor> create_fp16_tensor(int rows, int cols)
    {
        auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto fp32_data = generate_random_data(rows * cols);
        tensor->from_fp32(fp32_data.data(), fp32_data.size());
        return tensor;
    }

    // Helper to create BF16 Tensor
    std::unique_ptr<BF16Tensor> create_bf16_tensor(int rows, int cols)
    {
        auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto fp32_data = generate_random_data(rows * cols);
        tensor->from_fp32(fp32_data.data(), fp32_data.size());
        return tensor;
    }

    // Helper to create Q4_0 Tensor (Weights)
    std::unique_ptr<Q4_0Tensor> create_q4_0_tensor(int rows, int cols)
    {
        size_t num_blocks = (rows * cols) / Q4_0Block::BLOCK_SIZE;
        size_t raw_size = num_blocks * sizeof(Q4_0Block);
        std::vector<uint8_t> raw_data(raw_size);

        Q4_0Block *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist_d(0.1f, 1.0f);
        std::uniform_int_distribution<int> dist_q(0, 255);

        for (size_t i = 0; i < num_blocks; ++i)
        {
            blocks[i].d = dist_d(rng);
            for (int j = 0; j < 16; ++j)
            {
                blocks[i].qs[j] = static_cast<uint8_t>(dist_q(rng));
            }
        }

        return std::make_unique<Q4_0Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)}, raw_data);
    }

    // Helper to create Q8_1 Tensor (Activations)
    std::unique_ptr<Q8_1Tensor> create_q8_1_tensor(int rows, int cols)
    {
        size_t num_blocks = (rows * cols) / Q8_1Block::BLOCK_SIZE;
        size_t raw_size = num_blocks * sizeof(Q8_1Block);
        std::vector<uint8_t> raw_data(raw_size);

        Q8_1Block *blocks = reinterpret_cast<Q8_1Block *>(raw_data.data());
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist_d(0.001f, 0.1f);
        std::uniform_int_distribution<int> dist_q(-127, 127);

        for (size_t i = 0; i < num_blocks; ++i)
        {
            blocks[i].d = dist_d(rng);
            int sum = 0;
            for (int j = 0; j < 32; ++j)
            {
                blocks[i].qs[j] = static_cast<int8_t>(dist_q(rng));
                sum += blocks[i].qs[j];
            }
            blocks[i].sum_qs = static_cast<int16_t>(sum);
        }

        return std::make_unique<Q8_1Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)}, raw_data);
    }

    void print_header()
    {
        std::cout << std::left << std::setw(20) << "Format"
                  << std::setw(10) << "M"
                  << std::setw(10) << "N"
                  << std::setw(10) << "K"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "GFLOPS"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
    }

    void print_result(const std::string &format, int M, int N, int K, double avg_ms)
    {
        double gflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e9;
        std::cout << std::left << std::setw(20) << format
                  << std::setw(10) << M
                  << std::setw(10) << N
                  << std::setw(10) << K
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops
                  << std::endl;
    }
}

TEST(OneDNNGemmPerformance, Formats_Qwen7B)
{
    // Qwen 7B FFN Gate/Up Projection shape: [M, 11008] = [M, 4096] * [4096, 11008]^T
    // K = 4096, N = 11008
    const int K = D_MODEL;
    const int N = INTERMEDIATE_SIZE;

    std::vector<int> batch_sizes = {1, 32, 128, 512};

    print_header();

    for (int M : batch_sizes)
    {
        // 1. FP32 x FP32
        {
            auto A = create_fp32_tensor(M, K);
            auto B = create_fp32_tensor(N, K); // Transposed weights [N, K]
            auto C = create_fp32_tensor(M, N);

            OneDNNGemmKernel kernel(B.get());

            // Warmup
            for (int i = 0; i < WARMUP_ITERS; ++i)
            {
                kernel.multiply(A->data(), C->mutable_data(), M, N, K);
            }

            auto start = high_resolution_clock::now();
            for (int i = 0; i < BENCH_ITERS; ++i)
            {
                kernel.multiply(A->data(), C->mutable_data(), M, N, K);
            }
            auto end = high_resolution_clock::now();
            double avg_ms = duration_cast<microseconds>(end - start).count() / 1000.0 / BENCH_ITERS;
            print_result("FP32 x FP32", M, N, K, avg_ms);
        }

        // 2. FP16 x FP16
        {
            auto A = create_fp16_tensor(M, K);
            auto B = create_fp16_tensor(N, K); // Transposed weights
            auto C = create_fp32_tensor(M, N);

            OneDNNGemmKernel kernel(B.get());

            // Warmup
            for (int i = 0; i < WARMUP_ITERS; ++i)
            {
                kernel.multiply_activations_typed_impl(
                    A->data(), nullptr, C->mutable_data(),
                    M, N, K, true, 1.0f, 0.0f, nullptr, -1,
                    ActivationFormat::FP16, ActivationFormat::FP16);
            }

            auto start = high_resolution_clock::now();
            for (int i = 0; i < BENCH_ITERS; ++i)
            {
                kernel.multiply_activations_typed_impl(
                    A->data(), nullptr, C->mutable_data(),
                    M, N, K, true, 1.0f, 0.0f, nullptr, -1,
                    ActivationFormat::FP16, ActivationFormat::FP16);
            }
            auto end = high_resolution_clock::now();
            double avg_ms = duration_cast<microseconds>(end - start).count() / 1000.0 / BENCH_ITERS;
            print_result("FP16 x FP16", M, N, K, avg_ms);
        }

        // 3. BF16 x BF16
        {
            auto A = create_bf16_tensor(M, K);
            auto B = create_bf16_tensor(N, K); // Transposed weights
            auto C = create_fp32_tensor(M, N);

            OneDNNGemmKernel kernel(B.get());

            // Warmup
            for (int i = 0; i < WARMUP_ITERS; ++i)
            {
                kernel.multiply_activations_typed_impl(
                    A->data(), nullptr, C->mutable_data(),
                    M, N, K, true, 1.0f, 0.0f, nullptr, -1,
                    ActivationFormat::BF16, ActivationFormat::BF16);
            }

            auto start = high_resolution_clock::now();
            for (int i = 0; i < BENCH_ITERS; ++i)
            {
                kernel.multiply_activations_typed_impl(
                    A->data(), nullptr, C->mutable_data(),
                    M, N, K, true, 1.0f, 0.0f, nullptr, -1,
                    ActivationFormat::BF16, ActivationFormat::BF16);
            }
            auto end = high_resolution_clock::now();
            double avg_ms = duration_cast<microseconds>(end - start).count() / 1000.0 / BENCH_ITERS;
            print_result("BF16 x BF16", M, N, K, avg_ms);
        }

        // 4. Q4_0 x Q8_1
        {
            auto A = create_q8_1_tensor(M, K);
            auto B = create_q4_0_tensor(N, K); // Transposed weights
            auto C = create_fp32_tensor(M, N);

            OneDNNGemmKernel kernel(B.get());

            // Pre-pack weights (happens implicitly on first call, but let's do it explicitly to measure pure GEMM)
            // Actually, OneDNNGemmKernel handles packing internally and caches it.
            // So first call will be slow. We should do one warmup call to pack.

            // Get raw block pointer for A
            const void *A_ptr = A->get_raw_block_at(0, 0);

            // Warmup (includes packing on first iter)
            for (int i = 0; i < WARMUP_ITERS; ++i)
            {
                kernel.multiply_typed_activations(
                    A_ptr, TensorFormat::Q8_1, nullptr,
                    C->mutable_data(), M, N, K, true, 1.0f, 0.0f, nullptr, -1);
            }

            auto start = high_resolution_clock::now();
            for (int i = 0; i < BENCH_ITERS; ++i)
            {
                kernel.multiply_typed_activations(
                    A_ptr, TensorFormat::Q8_1, nullptr,
                    C->mutable_data(), M, N, K, true, 1.0f, 0.0f, nullptr, -1);
            }
            auto end = high_resolution_clock::now();
            double avg_ms = duration_cast<microseconds>(end - start).count() / 1000.0 / BENCH_ITERS;
            print_result("Q4_0 x Q8_1", M, N, K, avg_ms);
        }

        std::cout << std::string(80, '-') << std::endl;
    }
}
