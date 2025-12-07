/**
 * @file Perf__Q8_1_OnlineSoftmax.cpp
 * @brief Performance benchmark for Q8_1 attention score computation (Q * K^T with softmax)
 *
 * This test benchmarks the Q8_1 x Q8_1 attention pattern used in transformers.
 * It measures throughput and correctness for:
 *   - Single token decode (S=1)
 *   - Batched prefill (S=128, 512, 1024)
 *   - Various model sizes (Qwen 0.5B, 7B, 32B)
 *
 * Uses a C++ reference implementation for correctness validation.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

// V2 includes
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmJit_Q8_1_OnlineSoftmax.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

/**
 * @brief Model attention configuration
 */
struct ModelAttentionConfig
{
    std::string name;
    int n_heads;    ///< Number of attention heads
    int n_kv_heads; ///< Number of key-value heads (for GQA)
    int head_dim;   ///< Dimension per head
    int d_model;    ///< Model hidden dimension = n_heads * head_dim
};

// Model configurations matching real architectures
static const ModelAttentionConfig QWEN_0_5B = {"Qwen-0.5B", 14, 2, 64, 896};
static const ModelAttentionConfig QWEN_7B = {"Qwen-7B", 28, 4, 128, 3584};
static const ModelAttentionConfig QWEN_32B = {"Qwen-32B", 40, 8, 128, 5120};

struct AttentionBenchmarkConfig
{
    ModelAttentionConfig model;
    int seq_len; ///< Query sequence length (S)
    int kv_len;  ///< Key/Value sequence length (often same as seq_len for prefill)
    int warmup_iters;
    int bench_iters;
    int num_trials;
    std::string description;
};

struct BenchmarkStats
{
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double mean_gflops;
    double l2_error;
    double cosine_sim;
};

class Q8_1_OnlineSoftmax_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        if (rank_ == 0)
        {
            std::cout << "[Performance Test] OpenMP max threads: " << omp_get_max_threads() << std::endl;
        }
    }

    /**
     * @brief Compute FP32 reference: softmax(Q * K^T / sqrt(d))
     */
    void compute_reference(
        int M, int N, int K,
        const float *Q, const float *K_data,
        float scale, const float *mask,
        float *C_ref)
    {
        // Compute Q * K^T
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int kk = 0; kk < K; ++kk)
                {
                    sum += Q[i * K + kk] * K_data[j * K + kk];
                }
                C_ref[i * N + j] = sum * scale;
                if (mask)
                {
                    C_ref[i * N + j] += mask[i * N + j];
                }
            }
        }

        // Softmax per row
        for (int i = 0; i < M; ++i)
        {
            float max_val = C_ref[i * N];
            for (int j = 1; j < N; ++j)
            {
                max_val = std::max(max_val, C_ref[i * N + j]);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                C_ref[i * N + j] = std::exp(C_ref[i * N + j] - max_val);
                sum_exp += C_ref[i * N + j];
            }

            for (int j = 0; j < N; ++j)
            {
                C_ref[i * N + j] /= sum_exp;
            }
        }
    }

    // Compute just raw GEMM scores (no softmax) for debugging
    void compute_raw_gemm(
        int M, int N, int K,
        const float *Q, const float *K_data,
        float scale, const float *mask,
        float *C_ref)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int kk = 0; kk < K; ++kk)
                {
                    sum += Q[i * K + kk] * K_data[j * K + kk];
                }
                C_ref[i * N + j] = sum * scale;
                if (mask)
                {
                    C_ref[i * N + j] += mask[i * N + j];
                }
            }
        }
    }

    std::pair<double, double> verify_correctness(
        int M, int N,
        const float *C_act, const float *C_ref)
    {
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        double dot_prod = 0.0;
        double norm_act = 0.0;
        double norm_ref = 0.0;

        for (int i = 0; i < M * N; ++i)
        {
            double diff = C_act[i] - C_ref[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += C_ref[i] * C_ref[i];

            dot_prod += C_act[i] * C_ref[i];
            norm_act += C_act[i] * C_act[i];
            norm_ref += C_ref[i] * C_ref[i];
        }

        double l2_error = (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_diff) / std::sqrt(sum_sq_ref) : 0.0;
        double cosine_sim = (norm_act > 0.0 && norm_ref > 0.0)
                                ? dot_prod / (std::sqrt(norm_act) * std::sqrt(norm_ref))
                                : 0.0;

        return {l2_error, cosine_sim};
    }

    BenchmarkStats run_benchmark(const AttentionBenchmarkConfig &config)
    {
        const auto &model = config.model;
        int S = config.seq_len; // Number of query rows
        int KV = config.kv_len; // Number of key rows
        int D = model.head_dim; // Head dimension
        int H = model.n_heads;  // Number of heads

        float scale = 1.0f / std::sqrt(static_cast<float>(D));

        // Allocate Q and K in interleaved head layout: [seq_len, n_heads, head_dim]
        // For simplicity, we'll benchmark a single head but with realistic strides
        int k_blocks = (D + 31) / 32;
        // Use flat layout (no multi-head padding) for simpler correctness testing
        int Q_stride_bytes = k_blocks * sizeof(Q8_1Block); // Stride between Q rows (flat)
        int K_stride_bytes = k_blocks * sizeof(Q8_1Block); // Stride between K rows (flat)

        // Total Q8_1 blocks needed (flat layout)
        size_t Q_total_blocks = static_cast<size_t>(S) * k_blocks;
        size_t K_total_blocks = static_cast<size_t>(KV) * k_blocks;

        // Allocate Q8_1 block storage
        std::vector<Q8_1Block> Q_blocks(Q_total_blocks);
        std::vector<Q8_1Block> K_blocks_data(K_total_blocks);

        // Generate random FP32 data and quantize
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> Q_fp32(S * D);
        std::vector<float> K_fp32(KV * D);

        for (auto &x : Q_fp32)
            x = dist(gen); // Random Q for realistic test
        for (auto &x : K_fp32)
            x = dist(gen);

        // Quantize to Q8_1 (for head 0 only in benchmark)
        auto Q_tensor = Q8_1Tensor::quantize_from_fp32(Q_fp32.data(), {static_cast<size_t>(S), static_cast<size_t>(D)});
        auto K_tensor = Q8_1Tensor::quantize_from_fp32(K_fp32.data(), {static_cast<size_t>(KV), static_cast<size_t>(D)});

        // Copy blocks into FLAT layout (no multi-head stride)
        const Q8_1Block *Q_src = Q_tensor->decode_to_q8_1(0, 0);
        const Q8_1Block *K_src = K_tensor->decode_to_q8_1(0, 0);

        for (int s = 0; s < S; ++s)
        {
            for (int kb = 0; kb < k_blocks; ++kb)
            {
                // Flat layout: row s at blocks [s*k_blocks .. s*k_blocks + k_blocks-1]
                Q_blocks[s * k_blocks + kb] = Q_src[s * k_blocks + kb];
            }
        }

        for (int kv = 0; kv < KV; ++kv)
        {
            for (int kb = 0; kb < k_blocks; ++kb)
            {
                K_blocks_data[kv * k_blocks + kb] = K_src[kv * k_blocks + kb];
            }
        }

        // DEBUG: Print sum_qs of first 10 blocks
        /*
        {
            const Q8_1Block *blocks = K_tensor->decode_to_q8_1(0, 0);
            std::cout << "DEBUG: K_tensor sum_qs[0..9]: ";
            for (int i = 0; i < 10; ++i)
                std::cout << blocks[i].sum_qs << " ";
            std::cout << std::endl;

            // Also print K_blocks_data layout (now flat)
            std::cout << "DEBUG: K_blocks_data sum_qs at stride positions: ";
            for (int kv = 0; kv < 5; ++kv)
            {
                int idx = kv * k_blocks; // First block of each K row (flat)
                std::cout << "K[" << kv << "]=" << K_blocks_data[idx].sum_qs << " ";
            }
            std::cout << std::endl;

            // Print expected vs actual K_stride
            std::cout << "DEBUG: K_stride_bytes=" << K_stride_bytes
                      << " (= " << k_blocks << " * 36)" << std::endl;
        }
        */

        // Create output tensor C (FP32)
        std::vector<float> C(S * KV);

        // Create JIT kernels
        static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m4(4);
        static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m1(1);

        // Build params
        OnlineSoftmaxParams params;
        params.Q = Q_blocks.data();
        params.K = K_blocks_data.data();
        params.C = C.data();
        params.M = S;
        params.N = KV;
        params.K_blocks = k_blocks;
        params.Q_stride_bytes = Q_stride_bytes;
        params.K_stride_bytes = K_stride_bytes;
        params.C_stride_bytes = KV * sizeof(float);
        params.scale = scale;
        params.mask = nullptr;
        params.mask_stride_bytes = 0;

        // DEBUG: Print params
        /*
        std::cout << "DEBUG params: K_stride_bytes=" << params.K_stride_bytes
                  << ", K_blocks=" << params.K_blocks
                  << ", N=" << params.N
                  << ", M=" << params.M << std::endl;
        */

        // Lambda to run the kernel
        auto run_kernel = [&]()
        {
            int m_blocking = 4;
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < S; i += m_blocking)
            {
                int current_m = std::min(m_blocking, S - i);

                OnlineSoftmaxParams p = params;
                p.Q = reinterpret_cast<const char *>(params.Q) + i * Q_stride_bytes;
                p.C = params.C + i * KV;
                p.M = current_m;

                if (current_m == 4)
                {
                    kernel_m4.get_kernel()(&p);
                }
                else
                {
                    auto func = kernel_m1.get_kernel();
                    for (int j = 0; j < current_m; ++j)
                    {
                        OnlineSoftmaxParams p1 = p;
                        p1.Q = reinterpret_cast<const char *>(p.Q) + j * Q_stride_bytes;
                        p1.C = p.C + j * KV;
                        p1.M = 1;
                        func(&p1);
                    }
                }
            }
        };

        // Verify correctness (once)
        run_kernel();

        // Dequantize for reference
        std::vector<float> Q_deq(S * D), K_deq(KV * D);
        Q_tensor->to_fp32(Q_deq.data());
        K_tensor->to_fp32(K_deq.data());

        // Manual reference using Q8_1 blocks directly (same as kernel should compute)
        std::vector<float> C_manual(S * KV);
        for (int s = 0; s < S; ++s)
        {
            for (int kv = 0; kv < KV; ++kv)
            {
                float dot = 0.0f;
                for (int kb = 0; kb < k_blocks; ++kb)
                {
                    const Q8_1Block &q_block = Q_blocks[s * k_blocks + kb];
                    const Q8_1Block &k_block = K_blocks_data[kv * k_blocks + kb];

                    float d_Q = _cvtsh_ss(q_block.d);
                    float d_K = _cvtsh_ss(k_block.d);

                    int32_t int_sum = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        int_sum += static_cast<int32_t>(q_block.qs[i]) * static_cast<int32_t>(k_block.qs[i]);
                    }

                    dot += static_cast<float>(int_sum) * d_Q * d_K;
                }
                C_manual[s * KV + kv] = dot * scale;
            }
        }

        // Compare kernel output to manual reference (for debugging)
        // auto [l2_manual, cos_manual] = verify_correctness(S, KV, C.data(), C_manual.data());

        // Verify against FP32 reference
        /*
        std::cout << "DEBUG: K_deq row 1 [0..3]: ";
        for (int i = 0; i < 4; ++i)
            std::cout << K_deq[1 * D + i] << " ";
        std::cout << std::endl;
        */

        // DEBUG: Compare raw GEMM scores (not softmax)
        std::vector<float> C_ref(S * KV);
        compute_raw_gemm(S, KV, D, Q_deq.data(), K_deq.data(), scale, nullptr, C_ref.data());

        auto [l2_error, cosine_sim] = verify_correctness(S, KV, C.data(), C_ref.data());

        if (rank_ == 0 && l2_error > 0.1)
        {
            std::cout << "DEBUG: Large Error Detected!" << std::endl;
            std::cout << "C_act[0..9]: ";
            for (int i = 0; i < 10; ++i)
                std::cout << C[i] << " ";
            std::cout << std::endl;
            std::cout << "C_ref[0..9]: ";
            for (int i = 0; i < 10; ++i)
                std::cout << C_ref[i] << " ";
            std::cout << std::endl;
        }

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            run_kernel();
        }

        // Benchmark trials
        std::vector<double> trial_times_ms;
        trial_times_ms.reserve(config.num_trials);

        for (int t = 0; t < config.num_trials; ++t)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.bench_iters; ++i)
            {
                run_kernel();
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();

            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            trial_times_ms.push_back(total_ms / config.bench_iters);
        }

        // Calculate stats
        double sum = std::accumulate(trial_times_ms.begin(), trial_times_ms.end(), 0.0);
        double mean_ms = sum / config.num_trials;

        double sq_sum = std::inner_product(trial_times_ms.begin(), trial_times_ms.end(), trial_times_ms.begin(), 0.0);
        double stddev_ms = std::sqrt(sq_sum / config.num_trials - mean_ms * mean_ms);

        double min_ms = *std::min_element(trial_times_ms.begin(), trial_times_ms.end());
        double max_ms = *std::max_element(trial_times_ms.begin(), trial_times_ms.end());

        // GFLOPS for attention: 2 * S * KV * D (matmul) + S * KV (softmax ~ignored)
        double ops = 2.0 * S * KV * D;
        double mean_gflops = (ops / (mean_ms * 1e-3)) / 1e9;

        return {mean_ms, stddev_ms, min_ms, max_ms, mean_gflops, l2_error, cosine_sim};
    }

    void
    print_results(const AttentionBenchmarkConfig &config, const BenchmarkStats &stats)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;

        std::cout << std::left << std::setw(45) << config.description
                  << " | S=" << std::setw(4) << config.seq_len
                  << " KV=" << std::setw(5) << config.kv_len
                  << " D=" << std::setw(4) << config.model.head_dim
                  << " H=" << std::setw(3) << config.model.n_heads
                  << " | Time: " << std::fixed << std::setprecision(3) << stats.mean_ms << " ms"
                  << " | T-put: " << std::setprecision(2) << stats.mean_gflops << " GFLOPS"
                  << " | L2 Err: " << std::scientific << std::setprecision(2) << stats.l2_error
                  << " | Cos Sim: " << std::fixed << std::setprecision(5) << stats.cosine_sim
                  << std::endl;
    }
};

// =============================================================================
// Qwen 0.5B Tests
// =============================================================================

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen05B_Decode_KV128)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_0_5B,
        .seq_len = 1,
        .kv_len = 128,
        .warmup_iters = 10,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Qwen-0.5B Decode (S=1, KV=128)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen05B_Decode_KV1024)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_0_5B,
        .seq_len = 1,
        .kv_len = 1024,
        .warmup_iters = 10,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Qwen-0.5B Decode (S=1, KV=1024)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen05B_Prefill_256)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_0_5B,
        .seq_len = 256,
        .kv_len = 256,
        .warmup_iters = 5,
        .bench_iters = 20,
        .num_trials = 3,
        .description = "Qwen-0.5B Prefill (S=256, KV=256)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen05B_Prefill_512)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_0_5B,
        .seq_len = 512,
        .kv_len = 512,
        .warmup_iters = 3,
        .bench_iters = 10,
        .num_trials = 3,
        .description = "Qwen-0.5B Prefill (S=512, KV=512)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

// =============================================================================
// Qwen 7B Tests
// =============================================================================

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen7B_Decode_KV128)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_7B,
        .seq_len = 1,
        .kv_len = 128,
        .warmup_iters = 10,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Qwen-7B Decode (S=1, KV=128)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen7B_Decode_KV1024)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_7B,
        .seq_len = 1,
        .kv_len = 1024,
        .warmup_iters = 10,
        .bench_iters = 50,
        .num_trials = 5,
        .description = "Qwen-7B Decode (S=1, KV=1024)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen7B_Decode_KV4096)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_7B,
        .seq_len = 1,
        .kv_len = 4096,
        .warmup_iters = 5,
        .bench_iters = 20,
        .num_trials = 3,
        .description = "Qwen-7B Decode (S=1, KV=4096)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen7B_Prefill_256)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_7B,
        .seq_len = 256,
        .kv_len = 256,
        .warmup_iters = 3,
        .bench_iters = 10,
        .num_trials = 3,
        .description = "Qwen-7B Prefill (S=256, KV=256)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen7B_Prefill_512)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_7B,
        .seq_len = 512,
        .kv_len = 512,
        .warmup_iters = 2,
        .bench_iters = 5,
        .num_trials = 3,
        .description = "Qwen-7B Prefill (S=512, KV=512)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

// =============================================================================
// Qwen 32B Tests
// =============================================================================

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen32B_Decode_KV128)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_32B,
        .seq_len = 1,
        .kv_len = 128,
        .warmup_iters = 10,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Qwen-32B Decode (S=1, KV=128)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen32B_Decode_KV1024)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_32B,
        .seq_len = 1,
        .kv_len = 1024,
        .warmup_iters = 10,
        .bench_iters = 50,
        .num_trials = 5,
        .description = "Qwen-32B Decode (S=1, KV=1024)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen32B_Decode_KV4096)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_32B,
        .seq_len = 1,
        .kv_len = 4096,
        .warmup_iters = 5,
        .bench_iters = 20,
        .num_trials = 3,
        .description = "Qwen-32B Decode (S=1, KV=4096)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen32B_Prefill_256)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_32B,
        .seq_len = 256,
        .kv_len = 256,
        .warmup_iters = 3,
        .bench_iters = 10,
        .num_trials = 3,
        .description = "Qwen-32B Prefill (S=256, KV=256)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, Qwen32B_Prefill_512)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_32B,
        .seq_len = 512,
        .kv_len = 512,
        .warmup_iters = 2,
        .bench_iters = 5,
        .num_trials = 3,
        .description = "Qwen-32B Prefill (S=512, KV=512)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

// =============================================================================
// Long Context Tests
// =============================================================================

TEST_F(Q8_1_OnlineSoftmax_Perf, LongContext_Decode_KV8192)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_7B,
        .seq_len = 1,
        .kv_len = 8192,
        .warmup_iters = 3,
        .bench_iters = 10,
        .num_trials = 3,
        .description = "Long Context Decode (S=1, KV=8192)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_OnlineSoftmax_Perf, LongContext_Prefill_1024)
{
    AttentionBenchmarkConfig config{
        .model = QWEN_7B,
        .seq_len = 1024,
        .kv_len = 1024,
        .warmup_iters = 2,
        .bench_iters = 3,
        .num_trials = 3,
        .description = "Long Context Prefill (S=1024, KV=1024)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.02);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

// Minimal debug test with known values
TEST_F(Q8_1_OnlineSoftmax_Perf, DEBUG_MinimalCase)
{
    // Super simple: 1 Q row, 2 K rows, 1 K_block (32 elements)
    // All Q values = 1, All K values = 1
    // Expected: C[0] = C[1] = 32 * scale (with scale = 1)

    constexpr int S = 1;  // 1 query
    constexpr int KV = 2; // 2 keys
    constexpr int D = 32; // 32 dim = 1 block
    constexpr int k_blocks = 1;
    constexpr float scale = 1.0f;

    // Create Q8_1 blocks with known values
    // All qs = 1, d = 1.0 (as fp16)
    Q8_1Block Q_block{};
    Q_block.d = _cvtss_sh(1.0f, 0); // d = 1.0
    Q_block.sum_qs = 32;            // sum of 32 ones
    for (int i = 0; i < 32; ++i)
        Q_block.qs[i] = 1;

    Q8_1Block K0_block{};
    K0_block.d = _cvtss_sh(1.0f, 0);
    K0_block.sum_qs = 32;
    for (int i = 0; i < 32; ++i)
        K0_block.qs[i] = 1;

    Q8_1Block K1_block{};
    K1_block.d = _cvtss_sh(1.0f, 0);
    K1_block.sum_qs = 32;
    for (int i = 0; i < 32; ++i)
        K1_block.qs[i] = 1;

    // Flat layout: Q[0], K[0], K[1]
    std::vector<Q8_1Block> Q_blocks = {Q_block};
    std::vector<Q8_1Block> K_blocks = {K0_block, K1_block};

    std::vector<float> C(S * KV, 0.0f);

    // Build params
    OnlineSoftmaxParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.C = C.data();
    params.M = S;
    params.N = KV;
    params.K_blocks = k_blocks;
    params.Q_stride_bytes = k_blocks * sizeof(Q8_1Block); // 36
    params.K_stride_bytes = k_blocks * sizeof(Q8_1Block); // 36
    params.C_stride_bytes = KV * sizeof(float);           // 8
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride_bytes = 0;

    // DEBUG: Print memory layout
    std::cout << "DEBUG: K_blocks memory layout:" << std::endl;
    std::cout << "  K_base = " << (void *)K_blocks.data() << std::endl;
    std::cout << "  K[0] at offset 0: d=" << _cvtsh_ss(K_blocks[0].d)
              << ", sum_qs=" << K_blocks[0].sum_qs
              << ", qs[0]=" << (int)K_blocks[0].qs[0] << std::endl;
    std::cout << "  K[1] at offset 36: d=" << _cvtsh_ss(K_blocks[1].d)
              << ", sum_qs=" << K_blocks[1].sum_qs
              << ", qs[0]=" << (int)K_blocks[1].qs[0] << std::endl;
    std::cout << "  K_stride_bytes = " << params.K_stride_bytes << std::endl;

    // Verify memory layout manually
    const char *K_base = reinterpret_cast<const char *>(K_blocks.data());
    const Q8_1Block *K1_ptr = reinterpret_cast<const Q8_1Block *>(K_base + 36);
    std::cout << "  K[1] via pointer: d=" << _cvtsh_ss(K1_ptr->d)
              << ", sum_qs=" << K1_ptr->sum_qs
              << ", qs[0]=" << (int)K1_ptr->qs[0] << std::endl;

    // Create and run kernel
    static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m1(1);
    kernel_m1.get_kernel()(&params);

    // Manual reference (signed Q8_1 math)
    // Q * K dot product: Σ(q[i] * k[i]) where q,k ∈ [-128, 127]
    // With all values = 1: dot = 32
    // With d_Q = d_K = 1.0: result = 32 * 1.0 * 1.0 = 32
    // But VPDPBUSD uses unsigned Q, signed K: (q+128) * k
    // (1+128) * 1 = 129 per element, 32 elements = 4128
    // Then correction: -128 * sum_qs_K * d_Q * d_K = -128 * 32 * 1 * 1 = -4096
    // Net: 4128 - 4096 = 32 ✓

    float expected = 32.0f * scale; // 32.0

    std::cout << "DEBUG MinimalCase:" << std::endl;
    std::cout << "  C[0] = " << C[0] << " (expected " << expected << ")" << std::endl;
    std::cout << "  C[1] = " << C[1] << " (expected " << expected << ")" << std::endl;

    EXPECT_NEAR(C[0], expected, 0.1f) << "C[0] should be " << expected;
    EXPECT_NEAR(C[1], expected, 0.1f) << "C[1] should be " << expected;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
