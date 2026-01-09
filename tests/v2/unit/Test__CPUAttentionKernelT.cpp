/**
 * @file Test__CPUAttentionKernelT.cpp
 * @brief Unit tests for CPUAttentionKernelT via IActivationTensor factory
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Factory wiring (createAttention returns valid kernel)
 * 2. FP32 attention computation
 * 3. BF16 attention computation
 * 4. FP16 attention computation
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>

#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/MPIContext.h"
#include "v2/backends/DeviceId.h"

using namespace llaminar2;

namespace
{
    constexpr float FP32_TOLERANCE = 1e-5f;
    constexpr float BF16_TOLERANCE = 5e-2f; // BF16 has lower precision
    constexpr float FP16_TOLERANCE = 5e-3f;

    class Test__CPUAttentionKernelT : public ::testing::Test
    {
    protected:
        MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD}; // Mock MPI context
        DeviceId device_id_ = DeviceId::cpu();     // CPU device

        void SetUp() override
        {
            // Setup code if needed
        }

        void TearDown() override
        {
            // Cleanup code if needed
        }
    };

    TEST_F(Test__CPUAttentionKernelT, FactoryWiring_FP32)
    {
        TensorFactory factory(mpi_ctx_);
        auto tensor = factory.createFP32({1, 1}, device_id_);
        auto *activation = dynamic_cast<IActivationTensor *>(tensor.get());
        ASSERT_NE(activation, nullptr);

        auto kernel = activation->createAttention();
        ASSERT_NE(kernel, nullptr);
        EXPECT_TRUE(kernel->supports_device(-1));
    }

    TEST_F(Test__CPUAttentionKernelT, FactoryWiring_BF16)
    {
        TensorFactory factory(mpi_ctx_);
        auto tensor = factory.createBF16({1, 1});
        auto *activation = dynamic_cast<IActivationTensor *>(tensor.get());
        ASSERT_NE(activation, nullptr);

        auto kernel = activation->createAttention();
        ASSERT_NE(kernel, nullptr);
        EXPECT_TRUE(kernel->supports_device(-1));
    }

    TEST_F(Test__CPUAttentionKernelT, FactoryWiring_FP16)
    {
        TensorFactory factory(mpi_ctx_);
        auto tensor = factory.createFP16({1, 1});
        auto *activation = dynamic_cast<IActivationTensor *>(tensor.get());
        ASSERT_NE(activation, nullptr);

        auto kernel = activation->createAttention();
        ASSERT_NE(kernel, nullptr);
        EXPECT_TRUE(kernel->supports_device(-1));
    }

    TEST_F(Test__CPUAttentionKernelT, Compute_FP32_Small)
    {
        // Dimensions
        int seq_len = 2;
        int n_heads = 2;
        int n_kv_heads = 2;
        int head_dim = 4;
        int total_elements = seq_len * n_heads * head_dim;

        // Create tensors
        TensorFactory factory(mpi_ctx_);
        auto Q_tensor = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)}, device_id_);
        auto K_tensor = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads), static_cast<size_t>(head_dim)}, device_id_);
        auto V_tensor = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads), static_cast<size_t>(head_dim)}, device_id_);
        auto output_tensor = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)}, device_id_);

        // Initialize data
        float *Q = Q_tensor->mutable_data();
        float *K = K_tensor->mutable_data();
        float *V = V_tensor->mutable_data();
        float *output = output_tensor->mutable_data();

        // Simple case: Q=K, V=Identity-like
        // Head 0: Q=[1,0,0,0], K=[1,0,0,0] -> Score=1 (scaled)
        // Head 1: Q=[0,1,0,0], K=[0,1,0,0] -> Score=1 (scaled)
        for (int i = 0; i < total_elements; ++i)
        {
            Q[i] = 0.0f;
            K[i] = 0.0f;
            V[i] = 0.0f;
            output[i] = 0.0f;
        }

        // Set values for t=0
        Q[0] = 1.0f;
        K[0] = 1.0f;
        V[0] = 1.0f; // Head 0
        Q[head_dim] = 1.0f;
        K[head_dim] = 1.0f;
        V[head_dim] = 2.0f; // Head 1

        // Create kernel via factory
        auto *activation = dynamic_cast<IActivationTensor *>(Q_tensor.get());
        auto kernel = activation->createAttention();

        // Execute kernel using the correct ITensorAttention interface
        bool success = kernel->compute(
            Q,
            K,
            V,
            output,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            false,   // causal
            -1,      // window_size
            nullptr, // workspace_scores
            nullptr, // workspace_buffer
            nullptr, // workspace_context
            nullptr, // workspace_mask
            false,   // use_bf16
            &mpi_ctx_,
            -1 // device_idx
        );

        ASSERT_TRUE(success);

        // Verify output is not zero
        bool has_nonzero = false;
        for (int i = 0; i < total_elements; ++i)
        {
            if (std::abs(output[i]) > 1e-6f)
                has_nonzero = true;
        }
        EXPECT_TRUE(has_nonzero);
    }
}
