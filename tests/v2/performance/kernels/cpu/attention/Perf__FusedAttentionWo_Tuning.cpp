/**
 * @file Perf__FusedAttentionWo_Tuning.cpp
 * @brief Performance tuning benchmark for Qwen 7B/14B/32B configurations
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

#include "tensors/Tensors.h"
#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "utils/Logger.h"

using namespace llaminar::v2::kernels::jit;
using namespace llaminar2;

class Perf__FusedAttentionWo_Tuning : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    void SetUp() override
    {
        rng_.seed(42);
    }

    void generate_q8_1_blocks(std::vector<Q8_1Block> &blocks, size_t num_elements)
    {
        size_t num_blocks = (num_elements + 31) / 32;
        blocks.resize(num_blocks);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            blocks[b].d = 1.0f;
            blocks[b].sum_qs = 0;
            for (int i = 0; i < 32; ++i)
                blocks[b].qs[i] = (int8_t)(dist(rng_) * 127);
        }
    }

    void run_benchmark(const std::string &name, int head_dim, int num_heads, int num_kv_heads, int seq_len, int kv_len)
    {
        std::cout << "\n[ " << name << " ]" << std::endl;
        std::cout << "Config: head_dim=" << head_dim << ", heads=" << num_heads
                  << ", kv_heads=" << num_kv_heads << ", seq=" << seq_len << ", kv=" << kv_len << std::endl;

        // Setup JIT Kernel
        JitAttentionConfig config;
        config.head_dim = head_dim;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.batch_size = seq_len; // For prefill, batch_size is seq_len
        config.wo_format = WoFormat::Q8_1;
        config.causal = true;

        JitFusedAttentionWo kernel(config);

        // Allocate data
        size_t q_elements = seq_len * num_heads * head_dim;
        size_t k_elements = kv_len * num_kv_heads * head_dim;
        size_t v_elements = kv_len * num_kv_heads * head_dim;
        size_t wo_elements = num_heads * head_dim * num_heads * head_dim; // Full Wo? No, Wo is (num_heads*head_dim) x (num_heads*head_dim)
        // But kernel takes Wo per head?
        // JitFusedAttentionWo signature:
        // void operator()(const void* Q, const void* K, const void* V, const void* Wo, void* output,
        //                 int kv_len, float scale, float theta_base, int pos_offset)

        // Q: [seq_len, num_heads, head_dim] (Q8_1)
        // K: [kv_len, num_kv_heads, head_dim] (Q8_1)
        // V: [kv_len, num_kv_heads, head_dim] (Q8_1)
        // Wo: [num_heads, head_dim, num_heads, head_dim] (Q8_1) - Wait, Wo layout?
        // Usually Wo is [d_model, d_model].
        // The kernel processes output projection per head.
        // It expects Wo to be accessible.

        std::vector<Q8_1Block> Q, K, V, Wo;
        generate_q8_1_blocks(Q, q_elements);
        generate_q8_1_blocks(K, k_elements);
        generate_q8_1_blocks(V, v_elements);
        generate_q8_1_blocks(Wo, num_heads * head_dim * num_heads * head_dim); // Approx size

        std::vector<float> output(seq_len * num_heads * head_dim);

        // Warmup
        for (int i = 0; i < 10; ++i)
        {
            kernel.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), seq_len, kv_len, 1.0f / std::sqrt(head_dim), 0);
        }

        // Benchmark
        int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            kernel.compute(Q.data(), K.data(), V.data(), Wo.data(), output.data(), seq_len, kv_len, 1.0f / std::sqrt(head_dim), 0);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = duration_ms / iterations;

        // Calculate GFLOPS (Approx)
        // Attention: 2 * seq * heads * kv_len * head_dim (Q*K) + 2 * seq * heads * kv_len * head_dim (Score*V)
        // Wo: 2 * seq * heads * head_dim * d_model
        double ops = 4.0 * seq_len * num_heads * kv_len * head_dim;           // Attention
        ops += 2.0 * seq_len * num_heads * head_dim * (num_heads * head_dim); // Wo

        double gflops = (ops / 1e9) / (avg_ms / 1000.0);

        std::cout << "Time: " << std::fixed << std::setprecision(2) << avg_ms << " ms" << std::endl;
        std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    }
};

TEST_F(Perf__FusedAttentionWo_Tuning, Qwen7B_Decode)
{
    // Qwen 7B: 28 heads, 4 KV heads, head_dim 128
    run_benchmark("Qwen 7B Decode (Seq=1, KV=4096)", 128, 28, 4, 1, 4096);
}

TEST_F(Perf__FusedAttentionWo_Tuning, Qwen14B_32B_Decode)
{
    // Qwen 14B/32B: 40 heads, 10 KV heads, head_dim 128
    run_benchmark("Qwen 14B/32B Decode (Seq=1, KV=4096)", 128, 40, 10, 1, 4096);
}

TEST_F(Perf__FusedAttentionWo_Tuning, Qwen7B_Prefill)
{
    // Qwen 7B Prefill: Seq=128, KV=128
    run_benchmark("Qwen 7B Prefill (Seq=128)", 128, 28, 4, 128, 128);
}

TEST_F(Perf__FusedAttentionWo_Tuning, Qwen14B_32B_Prefill)
{
    // Qwen 14B/32B Prefill: Seq=128, KV=128
    run_benchmark("Qwen 14B/32B Prefill (Seq=128)", 128, 40, 10, 128, 128);
}
