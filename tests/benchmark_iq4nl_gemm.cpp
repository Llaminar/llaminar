/**
 * @file benchmark_iq4nl_gemm.cpp
 * @brief Benchmark IQ4_NL quantized GEMM performance using MPILinearOperator_v2
 *
 * Measures throughput (GFLOPS) for various matrix sizes with IQ4_NL quantized weights.
 * Tests both FP32 and BF16 activation paths with distributed MPI execution.
 *
 * @author David Sanftenberg
 * @date 2025-10-22
 */

#include "operators/MPILinearOperator_v2.h"
#include "tensors/TensorFactory.h"
#include "tensors/BF16Tensor.h"
#include "tensors/IQ4_NLTensor.h"
#include "ModelLoader.h"
#include "Logger.h"
#include "utils/DebugEnv.h"

#include <mpi.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

using namespace llaminar;

struct BenchmarkConfig
{
    int seq_len;
    int in_features;
    int out_features;
    int warmup_iters;
    int bench_iters;
    std::string description;
};

class IQ4NLGemmBenchmark
{
private:
    int rank_;
    int world_size_;
    std::unique_ptr<ModelLoader> model_loader_;
    const std::string model_path_{"/workspaces/llaminar/models/Qwen2-0.5B.IQ4_NL.gguf"};

public:
    IQ4NLGemmBenchmark()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        if (world_size_ != 2)
        {
            if (rank_ == 0)
            {
                std::cerr << "Error: This benchmark requires exactly 2 MPI ranks\n";
                std::cerr << "Run with: mpirun -np 2 " << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Set quantization environment
        setenv("LLAMINAR_QUANT_ENABLE", "1", 1);
        setenv("LLAMINAR_LOAD_QUANTIZED", "1", 1);
        unsetenv("LLAMINAR_FORCE_FP32_WEIGHTS");
        refreshDebugEnv();

        // Load model
        if (rank_ == 0)
        {
            std::cout << "Loading model: " << model_path_ << std::endl;
        }

        model_loader_ = std::make_unique<ModelLoader>();
        if (!model_loader_->loadModel(model_path_))
        {
            if (rank_ == 0)
            {
                std::cerr << "Failed to load model!" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rank_ == 0)
        {
            std::cout << "Model loaded successfully\n"
                      << std::endl;
        }
    }

    std::shared_ptr<TensorBase> loadWeight(int layer_idx, const std::string &weight_type)
    {
        std::string tensor_name = "blk." + std::to_string(layer_idx) + "." + weight_type + ".weight";
        auto tensor = model_loader_->loadTensor(tensor_name);
        if (!tensor)
        {
            if (rank_ == 0)
            {
                std::cerr << "Failed to load tensor: " << tensor_name << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return tensor;
    }

    std::shared_ptr<TensorBase> createFP32Activation(int seq_len, int features)
    {
        auto activation = TensorFactory::create_simple({seq_len, features});
        float *data = activation->data();

        // Fill with realistic values
        for (int i = 0; i < seq_len * features; ++i)
        {
            data[i] = 0.01f * (i % 100 - 50); // Range: [-0.5, 0.49]
        }

        return activation;
    }

    std::shared_ptr<BF16Tensor> createBF16Activation(int seq_len, int features)
    {
        auto fp32_act = createFP32Activation(seq_len, features);
        auto bf16_act = std::make_shared<BF16Tensor>(std::vector<int>{seq_len, features});
        bf16_act->from_fp32(fp32_act->data(), seq_len * features);
        return bf16_act;
    }

    double benchmarkFP32(const BenchmarkConfig &config, std::shared_ptr<TensorBase> weight)
    {
        MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);
        auto activation = createFP32Activation(config.seq_len, config.in_features);
        auto output = TensorFactory::create_simple({config.seq_len, config.out_features});

        std::vector<std::shared_ptr<TensorBase>> inputs = {activation, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            if (!linear_op.execute(inputs, outputs))
            {
                throw std::runtime_error("Linear operator execute failed during warmup");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        // Benchmark
        for (int i = 0; i < config.bench_iters; ++i)
        {
            if (!linear_op.execute(inputs, outputs))
            {
                throw std::runtime_error("Linear operator execute failed during benchmark");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return elapsed_ms / config.bench_iters; // Average time per iteration
    }

    double benchmarkBF16(const BenchmarkConfig &config, std::shared_ptr<TensorBase> weight)
    {
        MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);
        auto activation = createBF16Activation(config.seq_len, config.in_features);
        auto output = std::make_shared<BF16Tensor>(std::vector<int>{config.seq_len, config.out_features});

        std::vector<std::shared_ptr<TensorBase>> inputs = {activation, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            if (!linear_op.execute(inputs, outputs))
            {
                throw std::runtime_error("Linear operator execute failed during warmup");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        // Benchmark
        for (int i = 0; i < config.bench_iters; ++i)
        {
            if (!linear_op.execute(inputs, outputs))
            {
                throw std::runtime_error("Linear operator execute failed during benchmark");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return elapsed_ms / config.bench_iters;
    }

    void printResults(const BenchmarkConfig &config, double fp32_time_ms, double bf16_time_ms)
    {
        if (rank_ != 0)
            return;

        // Calculate FLOPS: 2 * m * n * k (multiply-add)
        double flops = 2.0 * config.seq_len * config.out_features * config.in_features;
        double fp32_gflops = (flops / fp32_time_ms) / 1e6; // GFLOPS
        double bf16_gflops = (flops / bf16_time_ms) / 1e6;

        // Calculate effective memory bandwidth (compressed weight size)
        size_t weight_bytes_compressed = (config.out_features * config.in_features) / 2; // ~4 bits per element
        size_t activation_bytes = config.seq_len * config.in_features * sizeof(float);
        size_t output_bytes = config.seq_len * config.out_features * sizeof(float);
        double total_bytes = weight_bytes_compressed + activation_bytes + output_bytes;
        double fp32_bandwidth_gb = (total_bytes / fp32_time_ms) / 1e6; // GB/s
        double bf16_bandwidth_gb = (total_bytes / bf16_time_ms) / 1e6;

        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(62) << config.description << " ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Configuration:                                                 ║\n";
        std::cout << "║   Sequence Length:  " << std::setw(10) << config.seq_len
                  << "                                      ║\n";
        std::cout << "║   Input Features:   " << std::setw(10) << config.in_features
                  << "                                      ║\n";
        std::cout << "║   Output Features:  " << std::setw(10) << config.out_features
                  << "                                      ║\n";
        std::cout << "║   MPI Ranks:        " << std::setw(10) << world_size_
                  << "                                      ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ FP32 Activation Path:                                          ║\n";
        std::cout << "║   Time per iter:    " << std::setw(10) << std::fixed << std::setprecision(3)
                  << fp32_time_ms << " ms                                   ║\n";
        std::cout << "║   Throughput:       " << std::setw(10) << std::fixed << std::setprecision(2)
                  << fp32_gflops << " GFLOPS                               ║\n";
        std::cout << "║   Bandwidth:        " << std::setw(10) << std::fixed << std::setprecision(2)
                  << fp32_bandwidth_gb << " GB/s                                 ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ BF16 Activation Path:                                          ║\n";
        std::cout << "║   Time per iter:    " << std::setw(10) << std::fixed << std::setprecision(3)
                  << bf16_time_ms << " ms                                   ║\n";
        std::cout << "║   Throughput:       " << std::setw(10) << std::fixed << std::setprecision(2)
                  << bf16_gflops << " GFLOPS                               ║\n";
        std::cout << "║   Bandwidth:        " << std::setw(10) << std::fixed << std::setprecision(2)
                  << bf16_bandwidth_gb << " GB/s                                 ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Speedup (BF16 vs FP32): " << std::setw(10) << std::fixed << std::setprecision(2)
                  << (fp32_time_ms / bf16_time_ms) << "x                              ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << std::endl;
    }

    void runBenchmarks()
    {
        std::vector<BenchmarkConfig> configs = {
            // Single token decode baseline
            {1, 896, 896, 5, 100, "Single Token Decode (Q Projection, 896x896)"},

            // Batch decode scaling test - extended range
            {16, 896, 896, 3, 30, "Batch Decode (16 tokens)"},
            {64, 896, 896, 2, 20, "Batch Decode (64 tokens)"},
            {128, 896, 896, 2, 15, "Batch Decode (128 tokens)"},
            {256, 896, 896, 1, 10, "Batch Decode (256 tokens)"},

            // Prefill scenarios - find peak
            {512, 896, 896, 1, 10, "Prefill (512 tokens, Q Projection)"},
            {1024, 896, 896, 1, 8, "Prefill (1024 tokens, Q Projection)"},
            {2048, 896, 896, 1, 5, "Prefill (2048 tokens, Q Projection)"},
            {4096, 896, 896, 1, 3, "Prefill (4096 tokens, Q Projection)"},
            {8192, 896, 896, 1, 2, "Prefill (8192 tokens, Q Projection)"},

            // FFN layers - batch scaling
            {1, 896, 4864, 5, 50, "Single Token FFN Gate (896 → 4864)"},
            {16, 896, 4864, 2, 20, "Batch FFN Gate (16 tokens)"},
            {64, 896, 4864, 1, 15, "Batch FFN Gate (64 tokens)"},
            {128, 896, 4864, 1, 10, "Batch FFN Gate (128 tokens)"},
            {256, 896, 4864, 1, 8, "Batch FFN Gate (256 tokens)"},

            // FFN prefill - push to maximum
            {512, 896, 4864, 1, 8, "FFN Prefill (512 tokens)"},
            {1024, 896, 4864, 1, 5, "FFN Prefill (1024 tokens)"},
            {2048, 896, 4864, 1, 3, "FFN Prefill (2048 tokens)"},
            {4096, 896, 4864, 1, 2, "FFN Prefill (4096 tokens)"},
            {8192, 896, 4864, 1, 1, "FFN Prefill (8192 tokens)"},
        };

        // Load weights for different scenarios
        auto q_weight = loadWeight(0, "attn_q");   // 896x896
        auto ffn_gate = loadWeight(0, "ffn_gate"); // 4864x896

        if (rank_ == 0)
        {
            std::cout << "\n";
            std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║          IQ4_NL Quantized GEMM Performance Benchmark          ║\n";
            std::cout << "║              MPILinearOperator_v2 (Release Build)              ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
            std::cout << std::endl;
        }

        for (const auto &config : configs)
        {
            // Select appropriate weight
            auto weight = (config.out_features == 4864) ? ffn_gate : q_weight;

            if (rank_ == 0)
            {
                std::cout << "Running: " << config.description << "..." << std::endl;
            }

            double fp32_time = benchmarkFP32(config, weight);
            double bf16_time = benchmarkBF16(config, weight);

            printResults(config, fp32_time, bf16_time);
        }

        if (rank_ == 0)
        {
            std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                    Benchmark Complete                          ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        }
    }
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    try
    {
        IQ4NLGemmBenchmark benchmark;
        benchmark.runBenchmarks();
    }
    catch (const std::exception &e)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            std::cerr << "Benchmark failed with exception: " << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
