#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

TEST(Perf__Q3K_Unpack, Benchmark)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX512 not supported";
    }

    // Setup data
    const int num_blocks = 10000;
    std::vector<Q3_KBlock> blocks(num_blocks);
    std::vector<int8_t> output(num_blocks * 256);
    std::vector<float> scales(num_blocks * 8);
    std::vector<float> mins(num_blocks * 8);

    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (auto &block : blocks)
    {
        for (int i = 0; i < 12; ++i)
            block.scales[i] = dist(gen);
        for (int i = 0; i < 256 / 4; ++i)
            block.qs[i] = dist(gen);
        for (int i = 0; i < 256 / 8; ++i)
            block.hmask[i] = dist(gen);
        block.d = 1.0f;
    }

    // Warmup
    for (int i = 0; i < 100; ++i)
    {
        simd::unpack_q3_k_superblock_to_int8_avx512(blocks[0], output.data(), scales.data(), mins.data());
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 1000;
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (int i = 0; i < num_blocks; ++i)
        {
            simd::unpack_q3_k_superblock_to_int8_avx512(blocks[i], output.data() + i * 256, scales.data() + i * 8, mins.data() + i * 8);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double duration_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    double total_bytes = (double)num_blocks * 256 * iterations; // Output bytes
    double gb_per_sec = (total_bytes / 1e9) / duration_sec;

    std::cout << "Q3_K Unpack Performance:" << std::endl;
    std::cout << "  Time: " << duration_sec << " s" << std::endl;
    std::cout << "  Throughput: " << gb_per_sec << " GB/s (output)" << std::endl;
#else
    GTEST_SKIP() << "AVX512 unpacker not compiled";
#endif
}
