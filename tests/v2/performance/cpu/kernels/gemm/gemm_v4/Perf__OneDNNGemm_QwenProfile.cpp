/**
 * @file Perf__OneDNNGemm_QwenProfile.cpp
 * @brief Benchmarks the OneDNN-based gemm_v4 adapter on Qwen 0.5B FFN weights.
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

#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "kernels/cpu/gemm_v4/OneDNNGemmAdapter.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;
using namespace std::chrono;

namespace
{
    std::unique_ptr<ModelLoader> load_qwen_model()
    {
        const char *model_path = std::getenv("LLAMINAR_TEST_MODEL_PATH");
        if (!model_path)
        {
            model_path = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf";
        }

        auto loader = std::make_unique<ModelLoader>();
        if (!loader->loadModel(model_path))
        {
            std::cerr << "Failed to load model from: " << model_path << std::endl;
            return nullptr;
        }

        std::cout << "Loaded model: " << model_path << "\n\n";
        return loader;
    }

    std::unique_ptr<FP32Tensor> generate_random_activations(int M, int K)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> values(static_cast<size_t>(M) * static_cast<size_t>(K));
        for (auto &v : values)
        {
            v = dist(rng);
        }

        std::memcpy(tensor->mutable_data(), values.data(), values.size() * sizeof(float));
        return tensor;
    }
}

TEST(OneDNNGemmPerformance, IterativeDiagnostic_M8192_BestConfig)
{
    const int M = 8192;
    const int K = 896;
    const int N = 4864;
    const int warmup_iters = 10;
    const int bench_iters = 50;

    auto loader = load_qwen_model();
    ASSERT_NE(loader, nullptr) << "Failed to load Qwen model";

    auto B_tensor = loader->loadTensor("blk.0.ffn_gate.weight", 0);
    ASSERT_NE(B_tensor, nullptr) << "Failed to load FFN gate weights";
    auto *B_q8 = dynamic_cast<Q8_0Tensor *>(B_tensor.get());
    ASSERT_NE(B_q8, nullptr) << "FFN gate weights are not Q8_0";

    auto A = generate_random_activations(M, K);
    ASSERT_NE(A, nullptr);

    std::cout << "======================================================================================================\n";
    std::cout << "==                                          OneDNN GEMM Diagnostic                                   ==\n";
    std::cout << "======================================================================================================\n";
    std::cout << "Configuration: M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Warmup iterations: " << warmup_iters << ", Timed iterations: " << bench_iters << "\n\n";

    // Pack once so the benchmark measures OneDNN execution exclusively.
    auto activation_pack = A->to_int8_activation_pack(M, K);
    auto weight_pack = pack_weights_to_int8(*B_q8, K, N);
    std::vector<float> activation_fp32(static_cast<size_t>(M) * static_cast<size_t>(K));
    for (int m = 0; m < M; ++m)
    {
        const float scale = activation_pack.row_scales[static_cast<size_t>(m)];
        const int8_t *src_row = activation_pack.data.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
        float *dst_row = activation_fp32.data() + static_cast<size_t>(m) * static_cast<size_t>(K);
        for (int k = 0; k < K; ++k)
        {
            dst_row[static_cast<size_t>(k)] = static_cast<float>(src_row[static_cast<size_t>(k)]) * scale;
        }
    }

    std::vector<float> weight_fp32(static_cast<size_t>(K) * static_cast<size_t>(N));
    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            const int8_t q = weight_pack.data[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            weight_fp32[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                static_cast<float>(q) * weight_pack.col_scales[static_cast<size_t>(n)];
        }
    }

    std::vector<float> fp32_reference(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    ASSERT_TRUE(run_onednn_fp32_matmul(activation_fp32.data(),
                                       weight_fp32.data(),
                                       fp32_reference.data(),
                                       M,
                                       N,
                                       K))
        << "OneDNN FP32 GEMM failed";

    const std::vector<size_t> output_shape{
        static_cast<size_t>(M),
        static_cast<size_t>(N)};
    FP32Tensor packed_output_tensor(output_shape);

    auto run_kernel = [&]()
    {
        return onednn_gemm_from_packed(activation_pack,
                                       weight_pack,
                                       packed_output_tensor,
                                       M,
                                       N,
                                       K,
                                       nullptr);
    };

    for (int i = 0; i < warmup_iters; ++i)
    {
        ASSERT_TRUE(run_kernel());
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i)
    {
        ASSERT_TRUE(run_kernel());
    }
    auto end = high_resolution_clock::now();

    double total_ns = duration_cast<nanoseconds>(end - start).count();
    double avg_ms = (total_ns / 1e6) / static_cast<double>(bench_iters);
    double gops = (2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K)) / (avg_ms * 1e6);

    std::cout << "Step 1: OneDNN GEMM\n";
    std::cout << "  Average time:  " << std::fixed << std::setprecision(2) << avg_ms << " ms\n";
    std::cout << "  Throughput:    " << std::fixed << std::setprecision(2) << gops << " GOPS\n";
    std::cout << "  Efficiency vs 2240 GOPS peak: " << std::fixed << std::setprecision(2)
              << (gops / 2240.0 * 100.0) << "%\n";

    std::vector<float> output(static_cast<size_t>(M) * static_cast<size_t>(N));
    std::memcpy(output.data(), packed_output_tensor.data(), output.size() * sizeof(float));

    // Copy back into a row-major buffer with ldc = N to mimic pipeline usage.
    FP32Tensor pipeline_output_tensor(output_shape);
    ASSERT_TRUE(onednn_gemm_adapter(M, N, K, *A, *B_q8, pipeline_output_tensor));

    std::vector<float> pipeline_output(output.size());
    std::memcpy(pipeline_output.data(), pipeline_output_tensor.data(), pipeline_output.size() * sizeof(float));

    // Validate that adapter path matches cached-pack path.
    for (size_t idx = 0; idx < pipeline_output.size(); ++idx)
    {
        ASSERT_NEAR(pipeline_output[idx], output[idx], 1e-3f)
            << "Mismatch at element " << idx;
    }

    std::cout << "  Adapter verification: PASS (matches packed execution)\n";

    struct ErrorStats
    {
        double max_abs = 0.0;
        double rel_l2 = 0.0;
    };

    auto compute_error_stats = [](const std::vector<float> &reference, const std::vector<float> &candidate)
    {
        ErrorStats stats;
        double sum_sq = 0.0;
        double ref_sq = 0.0;
        for (size_t idx = 0; idx < reference.size(); ++idx)
        {
            const double ref = static_cast<double>(reference[idx]);
            const double diff = static_cast<double>(candidate[idx]) - ref;
            stats.max_abs = std::max(stats.max_abs, std::abs(diff));
            sum_sq += diff * diff;
            ref_sq += ref * ref;
        }

        const double denom = ref_sq > 0.0 ? ref_sq : 1.0;
        stats.rel_l2 = std::sqrt(sum_sq / denom);
        return stats;
    };

    const auto stats = compute_error_stats(fp32_reference, output);
    constexpr double kMaxAbsTolerance = 0.50; // Absolute error guard (int8 vs FP32)
    constexpr double kRelL2Tolerance = 0.05;  // Relative L2 guard
    std::cout << "Step 2: FP32 reference comparison\n";
    std::cout << "  Max abs diff:  " << std::fixed << std::setprecision(6) << stats.max_abs << "\n";
    std::cout << "  Relative L2:   " << std::fixed << std::setprecision(4) << (stats.rel_l2 * 100.0) << "%\n";
    ASSERT_LT(stats.max_abs, kMaxAbsTolerance)
        << "Int8 GEMM deviates from FP32 reference (max abs diff " << stats.max_abs << ")";
    ASSERT_LT(stats.rel_l2, kRelL2Tolerance)
        << "Int8 GEMM deviates from FP32 reference (relative L2 " << stats.rel_l2 << ")";

    std::cout << "  FP32 reference check: PASS\n";
    std::cout << "======================================================================================================\n";
}
