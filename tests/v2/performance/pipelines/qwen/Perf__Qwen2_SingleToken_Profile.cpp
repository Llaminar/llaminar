/**
 * @file Perf__Qwen2_SingleToken_Profile.cpp
 * @brief Performance profiling harness for Qwen2Pipeline single-token decode
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <fstream>

#include "pipelines/qwen/Qwen2Pipeline.h"
#include "loaders/ModelContext.h"
#include "utils/MPIContext.h"
#include "tensors/TensorFactory.h"

using namespace llaminar2;
using namespace std::chrono;

struct BenchmarkConfig
{
    std::string name;
    std::string model_path;
};

// Allow GTest to print the config
std::ostream &operator<<(std::ostream &os, const BenchmarkConfig &config)
{
    return os << config.name;
}

class Qwen2SingleTokenProfile : public ::testing::TestWithParam<BenchmarkConfig>
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<Qwen2Pipeline> pipeline_;

    // Configuration
    const int prefill_len_ = 32; // Small prefill to set up KV cache
    int warmup_iters_ = 2;
    int profile_iters_ = 10;

    void SetUp() override
    {
        const auto &config = GetParam();
        std::string model_path = config.model_path;

        // Create MPI context (single rank)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        // Check if file exists first
        std::ifstream f(model_path.c_str());
        if (!f.good())
        {
            std::cerr << "SKIPPING: Model file not found: " << model_path << "\n";
            GTEST_SKIP();
            return;
        }

        // Load model
        model_ctx_ = ModelContext::create(model_path, mpi_ctx_);
        if (!model_ctx_)
        {
            std::cerr << "SKIPPING: Failed to load model context from " << model_path << "\n";
            GTEST_SKIP();
            return;
        }

        // Configure pipeline
        PipelineConfig p_config;
        p_config.max_seq_len = 2048;
        p_config.activation_precision = ActivationPrecision::FP32;

        // Create pipeline
        pipeline_ = std::make_unique<Qwen2Pipeline>(
            model_ctx_,
            mpi_ctx_,
            -1,      // CPU
            nullptr, // No placement map
            p_config,
            1 // Batch size 1
        );
    }
};

TEST_P(Qwen2SingleTokenProfile, ProfileDecodeLoop)
{
    if (!pipeline_)
        return;

    const auto &config = GetParam();

    std::cout << "================================================================\n";
    std::cout << "  Qwen2 Single-Token Decode Profiling: " << config.name << "\n";
    std::cout << "  Model: " << config.model_path << "\n";
    std::cout << "================================================================\n";

    // 1. Prefill phase (warmup)
    std::vector<int> tokens(prefill_len_);
    std::iota(tokens.begin(), tokens.end(), 100); // Dummy tokens 100, 101...

    std::cout << "Running prefill (" << prefill_len_ << " tokens)...\n";
    pipeline_->forward(tokens.data(), prefill_len_);

    // 2. Decode warmup
    int next_token = 200;
    std::cout << "Running decode warmup (" << warmup_iters_ << " iters)...\n";
    for (int i = 0; i < warmup_iters_; ++i)
    {
        pipeline_->forward(&next_token, 1);
    }

    // 3. Profile loop
    std::cout << "Running profile loop (" << profile_iters_ << " iters)...\n";
    std::cout << "Attach 'perf record -p <pid>' now if running manually.\n";

    auto start = high_resolution_clock::now();

    for (int i = 0; i < profile_iters_; ++i)
    {
        pipeline_->forward(&next_token, 1);
    }

    auto end = high_resolution_clock::now();
    double total_ms = duration_cast<nanoseconds>(end - start).count() / 1e6;
    double avg_ms = total_ms / profile_iters_;
    double tok_s = 1000.0 / avg_ms;

    std::cout << "----------------------------------------------------------------\n";
    std::cout << "Results [" << config.name << "]:\n";
    std::cout << "  Total Time: " << total_ms << " ms\n";
    std::cout << "  Avg Latency: " << avg_ms << " ms/token\n";
    std::cout << "  Throughput: " << tok_s << " tokens/s\n";
    std::cout << "----------------------------------------------------------------\n";
}

INSTANTIATE_TEST_SUITE_P(
    ModelScaling,
    Qwen2SingleTokenProfile,
    ::testing::Values(
        BenchmarkConfig{"0.5B", "models/qwen2.5-0.5b-instruct-q4_0.gguf"},
        BenchmarkConfig{"7B", "models/Qwen2.5-7B-Instruct-Q4_0.gguf"}),
    [](const ::testing::TestParamInfo<BenchmarkConfig> &info)
    {
        std::string name = info.param.name;
        std::replace(name.begin(), name.end(), '.', '_');
        return name;
    });
