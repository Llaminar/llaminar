/**
 * @file MockPipeline.cpp
 * @brief Implementation of minimal mock pipeline for testing
 * @author David Sanftenberg
 */

#include "MockPipeline.h"
#include "../pipelines/PipelineConfig.h"
#include "../loaders/ModelContext.h"
#include "../utils/Logger.h"

namespace llaminar2
{
    namespace test
    {
        MockPipeline::MockPipeline(
            int n_heads,
            int n_kv_heads,
            int head_dim,
            std::shared_ptr<MPIContext> mpi_ctx)
            : PipelineBase(
                  ModelContext::createForTesting("mock_test.gguf"), // Test-only model context
                  mpi_ctx,                                          // mpi_ctx
                  -1,                                               // device_idx (CPU)
                  nullptr,                                          // placement_map
                  PipelineConfig{}),                                // config (defaults)
              n_heads_(n_heads),
              n_kv_heads_(n_kv_heads),
              head_dim_(head_dim)
        {
            // Set architecture parameters (required before initializeInfrastructure)
            n_layers_ = 1;
            n_heads_ = n_heads;
            n_kv_heads_ = n_kv_heads;
            head_dim_ = head_dim;
            d_model_ = n_heads * head_dim;

            // Initialize workspace buffers and infrastructure
            initializeInfrastructure();
        }

        bool MockPipeline::test_attention_gqa(
            TensorBase *Q,
            TensorBase *K,
            TensorBase *V,
            TensorBase *output,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size)
        {
            // Forward to protected method
            return attention_gqa(Q, K, V, output, n_heads, n_kv_heads, head_dim, causal, window_size);
        }

        bool MockPipeline::test_attention_gqa_mpi(
            TensorBase *Q,
            TensorBase *K,
            TensorBase *V,
            TensorBase *output,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size)
        {
            // Forward to protected method
            return attention_gqa_mpi(Q, K, V, output, n_heads, n_kv_heads, head_dim, causal, window_size);
        }

        bool MockPipeline::test_attention_gqa_tensor_parallel(
            TensorBase *Q,
            TensorBase *K,
            TensorBase *V,
            TensorBase *output,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size)
        {
            // Forward to protected method
            return attention_gqa_tensor_parallel(Q, K, V, output, n_heads, n_kv_heads, head_dim, causal, window_size);
        }

    } // namespace test
} // namespace llaminar2
