/**
 * @file Perf__OneDNNGemm_Overhead.cpp
 * @brief Benchmarks OneDNN overhead for single-token inference (M=1).
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
#include <omp.h>

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

TEST(OneDNNGemmPerformance, SingleTokenOverhead_M1)
{
    const int M = 1;
    const int K = 896;
    const int N = 4864;
    const int warmup_iters = 100;
    const int bench_iters = 1000;

    auto loader = load_qwen_model();
    ASSERT_NE(loader, nullptr) << "Failed to load Qwen model";

    auto B_tensor = loader->loadTensor("blk.0.ffn_gate.weight", 0);
    ASSERT_NE(B_tensor, nullptr) << "Failed to load FFN gate weights";
    auto *B_q8 = dynamic_cast<Q8_0Tensor *>(B_tensor.get());
    ASSERT_NE(B_q8, nullptr) << "FFN gate weights are not Q8_0";

    auto A = generate_random_activations(M, K);
    ASSERT_NE(A, nullptr);

    std::cout << "======================================================================================================\n";
    std::cout << "==                                OneDNN GEMM Single Token Overhead (M=1)                           ==\n";
    std::cout << "======================================================================================================\n";
    std::cout << "Configuration: M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Warmup iterations: " << warmup_iters << ", Timed iterations: " << bench_iters << "\n";
    int max_threads = omp_get_max_threads();
    std::cout << "Max OpenMP threads: " << max_threads << "\n\n";

    // Prepare data
    std::vector<float> activation_fp32(static_cast<size_t>(M) * static_cast<size_t>(K));
    // Copy from A tensor
    std::memcpy(activation_fp32.data(), A->data(), activation_fp32.size() * sizeof(float));

    // Generate random FP32 weights (simulating dequantized weights)
    std::vector<float> weight_fp32(static_cast<size_t>(K) * static_cast<size_t>(N));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : weight_fp32)
    {
        v = dist(rng);
    }

    std::vector<float> C(static_cast<size_t>(M) * static_cast<size_t>(N));

    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 28, 56, 112};

    std::cout << "------------------------------------------------------------------------------------------------------\n";
    std::cout << " Threads | Avg Latency (ms) | Creation (ms) | Execution (ms) | Throughput (GFLOPS)\n";
    std::cout << "------------------------------------------------------------------------------------------------------\n";

    for (int threads : thread_counts)
    {
        if (threads > max_threads)
            continue;

        omp_set_num_threads(threads);
        int current_max_threads = omp_get_max_threads();

        // Measure creation and execution separately
        double total_create_ms = 0;
        double total_exec_ms = 0;

        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;

        dnnl::memory::dims src_dims = {M, K};
        dnnl::memory::dims weight_dims = {K, N};
        dnnl::memory::dims dst_dims = {M, N};

        auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
        auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::ab);
        auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

        // Warmup
        for (int i = 0; i < warmup_iters; ++i)
        {
            dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);
            dnnl::memory src_mem(src_md, onednn_engine(), activation_fp32.data());
            dnnl::memory weight_mem(weight_md, onednn_engine(), weight_fp32.data());
            dnnl::memory dst_mem(dst_md, onednn_engine(), C.data());
            dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                            {{DNNL_ARG_SRC, src_mem},
                                             {DNNL_ARG_WEIGHTS, weight_mem},
                                             {DNNL_ARG_DST, dst_mem}});
            onednn_stream().wait();
        }

        // Benchmark
        for (int i = 0; i < bench_iters; ++i)
        {
            auto t0 = high_resolution_clock::now();
            dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);
            auto t1 = high_resolution_clock::now();

            dnnl::memory src_mem(src_md, onednn_engine(), activation_fp32.data());
            dnnl::memory weight_mem(weight_md, onednn_engine(), weight_fp32.data());
            dnnl::memory dst_mem(dst_md, onednn_engine(), C.data());

            dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                            {{DNNL_ARG_SRC, src_mem},
                                             {DNNL_ARG_WEIGHTS, weight_mem},
                                             {DNNL_ARG_DST, dst_mem}});
            onednn_stream().wait();
            auto t2 = high_resolution_clock::now();

            total_create_ms += duration_cast<nanoseconds>(t1 - t0).count() / 1e6;
            total_exec_ms += duration_cast<nanoseconds>(t2 - t1).count() / 1e6;
        }

        double avg_create_ms = total_create_ms / bench_iters;
        double avg_exec_ms = total_exec_ms / bench_iters;
        double avg_total_ms = avg_create_ms + avg_exec_ms;
        double gflops = (2.0 * M * N * K * 1e-9) / (avg_total_ms * 1e-3);

        std::cout << std::setw(8) << threads << " | "
                  << std::setw(16) << std::fixed << std::setprecision(4) << avg_total_ms << " | "
                  << std::setw(13) << std::fixed << std::setprecision(4) << avg_create_ms << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_exec_ms << " | "
                  << std::setw(19) << std::fixed << std::setprecision(2) << gflops << " | "
                  << "Max Threads: " << current_max_threads << "\n";
    }
    std::cout << "------------------------------------------------------------------------------------------------------\n";
}
