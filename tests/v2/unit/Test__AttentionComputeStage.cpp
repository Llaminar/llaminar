/**
 * @file Test__AttentionComputeStage.cpp
 * @brief Unit tests for AttentionComputeStage compute stage
 * @author David Sanftenberg
 *
 * Tests the new AttentionComputeStage which uses KernelFactory for
 * type-safe attention kernel dispatch. This is part of Phase 9 of
 * the multi-device architecture refactoring.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>

#include "backends/DeviceId.h"
#include "execution/compute_stages/ComputeStages.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/MPIContext.h"

using namespace llaminar2;

namespace
{
    constexpr float TOLERANCE = 1e-4f;

    class Test__AttentionComputeStage : public ::testing::Test
    {
    protected:
        MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};
        DeviceId device_id_ = DeviceId::cpu();

        void SetUp() override {}
        void TearDown() override {}
    };

    /**
     * @brief Test that AttentionComputeStage can be constructed with valid params
     */
    TEST_F(Test__AttentionComputeStage, Construction)
    {
        TensorFactory factory(mpi_ctx_);

        // Minimal dimensions
        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        // Create tensors
        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Construct stage
        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    }

    /**
     * @brief Test supportsBackend for CPU backends
     */
    TEST_F(Test__AttentionComputeStage, SupportsBackend_CPU)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        auto stage = std::make_unique<AttentionComputeStage>(params);

        // Should support CPU backends
        EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
        EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));

        // Unknown backends should not be supported
        EXPECT_FALSE(stage->supportsBackend(static_cast<ComputeBackendType>(999)));
    }

    /**
     * @brief Test getDumpInfo returns valid scalar info
     */
    TEST_F(Test__AttentionComputeStage, GetDumpInfo)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 8;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 16;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.window_size = -1;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        StageDumpInfo info = stage->getDumpInfo();

        // Check scalars captured key dimensions
        EXPECT_FALSE(info.scalars.empty());

        // Find specific scalars
        bool found_seq_len = false;
        bool found_n_heads = false;
        for (const auto &s : info.scalars)
        {
            if (std::string(s.name) == "seq_len")
            {
                found_seq_len = true;
                EXPECT_EQ(static_cast<int>(s.value), seq_len);
            }
            if (std::string(s.name) == "n_heads")
            {
                found_n_heads = true;
                EXPECT_EQ(static_cast<int>(s.value), n_heads);
            }
        }
        EXPECT_TRUE(found_seq_len);
        EXPECT_TRUE(found_n_heads);
    }

    /**
     * @brief Test estimatedFlops calculation
     */
    TEST_F(Test__AttentionComputeStage, EstimatedFlops)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 8;
        int kv_len = 16;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 16;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = kv_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        size_t flops = stage->estimatedFlops();

        // FLOPs should be:
        // Q@K^T: 2 * batch * n_heads * seq_len * kv_len * head_dim = 2*1*4*8*16*16 = 16384
        // softmax: ~4 * batch * n_heads * seq_len * kv_len = 4*1*4*8*16 = 2048
        // scores@V: 2 * batch * n_heads * seq_len * kv_len * head_dim = 16384
        // Total: 16384 + 2048 + 16384 = 34816
        size_t expected_qk = 2ULL * 1 * n_heads * seq_len * kv_len * head_dim;
        size_t expected_softmax = 4ULL * 1 * n_heads * seq_len * kv_len;
        size_t expected_sv = expected_qk;
        size_t expected_total = expected_qk + expected_softmax + expected_sv;

        EXPECT_EQ(flops, expected_total);
    }

    /**
     * @brief Test basic execute with FP32 tensors
     *
     * This test verifies that AttentionComputeStage::execute() can successfully
     * dispatch to the underlying attention kernel via KernelFactory.
     */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_Basic)
    {
        TensorFactory factory(mpi_ctx_);

        // Small attention problem
        int seq_len = 2;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 4;
        size_t hidden_size = n_heads * head_dim;

        // Create tensors
        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Initialize with simple values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();
        float *out_data = output->mutable_data();

        size_t total_q = seq_len * hidden_size;
        size_t total_kv = seq_len * n_kv_heads * head_dim;

        // Initialize Q, K, V with small values to avoid numerical issues
        for (size_t i = 0; i < total_q; ++i)
        {
            Q_data[i] = 0.1f * (i % 4);
        }
        for (size_t i = 0; i < total_kv; ++i)
        {
            K_data[i] = 0.1f * ((i + 1) % 4);
            V_data[i] = 0.1f * ((i + 2) % 4);
        }
        for (size_t i = 0; i < total_q; ++i)
        {
            out_data[i] = -999.0f; // Sentinel value
        }

        // Create workspace for attention scores
        auto workspace_scores = factory.createFP32(
            {static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)},
            device_id_);

        // Configure stage params
        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.window_size = -1;
        params.workspace_scores = workspace_scores.get();
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        // Create and execute stage
        auto stage = std::make_unique<AttentionComputeStage>(params);

        // Execute with no device context (CPU execution)
        bool success = stage->execute(nullptr);
        EXPECT_TRUE(success) << "AttentionComputeStage::execute() should succeed";

        // Verify output was modified (not sentinel values)
        bool output_modified = false;
        for (size_t i = 0; i < total_q; ++i)
        {
            if (std::abs(out_data[i] - (-999.0f)) > 1e-6f)
            {
                output_modified = true;
                break;
            }
        }
        EXPECT_TRUE(output_modified) << "Output should be modified after execution";

        // Verify output is finite
        for (size_t i = 0; i < total_q; ++i)
        {
            EXPECT_TRUE(std::isfinite(out_data[i])) << "Output[" << i << "] = " << out_data[i] << " should be finite";
        }
    }

    /**
     * @brief Test execute with GQA (n_kv_heads < n_heads)
     */
    TEST_F(Test__AttentionComputeStage, Execute_FP32_GQA)
    {
        TensorFactory factory(mpi_ctx_);

        // GQA configuration: 4 query heads, 2 KV heads (ratio 2:1)
        int seq_len = 4;
        int n_heads = 4;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        // Initialize with random-ish values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();

        for (size_t i = 0; i < Q->numel(); ++i)
            Q_data[i] = 0.05f * ((i * 7) % 11 - 5);
        for (size_t i = 0; i < K->numel(); ++i)
            K_data[i] = 0.05f * ((i * 13) % 11 - 5);
        for (size_t i = 0; i < V->numel(); ++i)
            V_data[i] = 0.05f * ((i * 17) % 11 - 5);

        auto workspace_scores = factory.createFP32(
            {static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)},
            device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.workspace_scores = workspace_scores.get();
        params.mpi_ctx = &mpi_ctx_;
        params.device_id = device_id_;

        auto stage = std::make_unique<AttentionComputeStage>(params);
        bool success = stage->execute(nullptr);
        EXPECT_TRUE(success) << "GQA attention should succeed";

        // Verify output is valid
        float *out_data = output->mutable_data();
        for (size_t i = 0; i < output->numel(); ++i)
        {
            EXPECT_TRUE(std::isfinite(out_data[i])) << "GQA output[" << i << "] should be finite";
        }
    }

    /**
     * @brief Test factory method createAttentionCompute
     */
    TEST_F(Test__AttentionComputeStage, FactoryMethod)
    {
        TensorFactory factory(mpi_ctx_);

        int seq_len = 4;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 8;
        size_t hidden_size = n_heads * head_dim;

        auto Q = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);
        auto K = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto V = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)}, device_id_);
        auto output = factory.createFP32({static_cast<size_t>(seq_len), hidden_size}, device_id_);

        AttentionComputeStage::Params params;
        params.Q = Q.get();
        params.K = K.get();
        params.V = V.get();
        params.output = output.get();
        params.batch_size = 1;
        params.seq_len = seq_len;
        params.kv_len = seq_len;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;

        // Use factory method
        auto stage = ComputeStageFactory::createAttentionCompute(params);
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->type(), ComputeStageType::ATTENTION);
    }

} // namespace
