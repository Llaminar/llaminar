#include <gtest/gtest.h>
#include <mpi.h>
#include "v2/pipelines/attention/MpiAttentionOrchestrator.h"
#include "v2/tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace llaminar2;

class MpiAttentionOrchestratorMaskingTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            MPI_Init(nullptr, nullptr);
        }
    }

    static void TearDownTestSuite()
    {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized)
        {
            MPI_Finalize();
        }
    }

    void SetUp() override
    {
        // Common test dimensions
        seq_len_ = 4;
        n_heads_ = 2;
        n_kv_heads_ = 2;
        head_dim_ = 16;
        d_model_ = n_heads_ * head_dim_;
    }

    std::unique_ptr<FP32Tensor> createRandomTensor(const std::vector<size_t> &shape)
    {
        auto tensor = std::make_unique<FP32Tensor>(shape);
        float *data = tensor->mutable_data();
        size_t total = 1;
        for (auto dim : shape)
            total *= dim;
        for (size_t i = 0; i < total; ++i)
            data[i] = (float)rand() / RAND_MAX - 0.5f;
        return tensor;
    }

    std::unique_ptr<FP32Tensor> createFilledTensor(const std::vector<size_t> &shape, float value)
    {
        auto tensor = std::make_unique<FP32Tensor>(shape);
        float *data = tensor->mutable_data();
        size_t total = 1;
        for (auto dim : shape)
            total *= dim;
        std::fill(data, data + total, value);
        return tensor;
    }

    int seq_len_;
    int n_heads_;
    int n_kv_heads_;
    int head_dim_;
    int d_model_;
};

TEST_F(MpiAttentionOrchestratorMaskingTest, IgnoresGarbageMaskWhenNotNeeded)
{
    // Setup MPI context (rank 0, world size 1)
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

    // Batch size 2
    int batch_size = 2;
    int total_tokens = batch_size * seq_len_;

    // Create inputs
    auto Q = createRandomTensor({static_cast<size_t>(total_tokens), static_cast<size_t>(n_heads_ * head_dim_)});
    auto K = createRandomTensor({static_cast<size_t>(total_tokens), static_cast<size_t>(n_kv_heads_ * head_dim_)});
    auto V = createRandomTensor({static_cast<size_t>(total_tokens), static_cast<size_t>(n_kv_heads_ * head_dim_)});
    auto output = createFilledTensor({static_cast<size_t>(total_tokens), static_cast<size_t>(n_heads_ * head_dim_)}, 0.0f);

    // Create config
    MpiAttentionConfig config;
    config.mpi_ctx = mpi_ctx;
    config.mpi_strategy = MPIStrategy::TensorParallel;
    config.n_heads = n_heads_;
    config.n_kv_heads = n_kv_heads_;
    config.head_dim = head_dim_;
    config.causal = false; // No masking needed
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;

    // Create workspaces
    config.workspace_scores = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(batch_size * n_heads_ * seq_len_ * seq_len_)});
    config.workspace_qkv_buffer = nullptr; // Not needed for FP32
    config.workspace_context = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(total_tokens * n_heads_ * head_dim_)});

    // Create mask with GARBAGE values (large negative values that would affect softmax)
    config.workspace_mask = createFilledTensor({static_cast<size_t>(total_tokens * total_tokens)}, -10000.0f);

    // Orchestrator is static-only
    // MpiAttentionOrchestrator orchestrator;

    // Run with garbage mask
    // Since causal=false and no sequence lengths, needs_mask should be false.
    // The garbage mask should be IGNORED.
    bool success = MpiAttentionOrchestrator::compute_tensor_parallel(
        Q.get(), K.get(), V.get(), output.get(),
        config, batch_size, nullptr);

    ASSERT_TRUE(success);

    // Run again with CLEAN mask (zeros) to verify results match
    auto output_clean = createFilledTensor({static_cast<size_t>(total_tokens), static_cast<size_t>(n_heads_ * head_dim_)}, 0.0f);
    auto mask_clean = createFilledTensor({static_cast<size_t>(total_tokens * total_tokens)}, 0.0f);
    config.workspace_mask = std::move(mask_clean);

    success = MpiAttentionOrchestrator::compute_tensor_parallel(
        Q.get(), K.get(), V.get(), output_clean.get(),
        config, batch_size, nullptr);

    ASSERT_TRUE(success);

    // Compare outputs
    const float *out_garbage = output->data();
    const float *out_clean = output_clean->data();
    size_t total_elements = total_tokens * n_heads_ * head_dim_;

    for (size_t i = 0; i < total_elements; ++i)
    {
        EXPECT_NEAR(out_garbage[i], out_clean[i], 1e-4f) << "Mismatch at index " << i;
    }
}
