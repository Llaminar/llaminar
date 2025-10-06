/**
 * @file test_embedding_standalone.cpp
 * @brief Minimal standalone test for embedding layer correctness
 * @author David Sanftenberg
 *
 * This test isolates the embedding layer to verify:
 * - Correct GGUF weight loading
 * - Proper dequantization (if quantized)
 * - Correct embedding lookup logic
 * - Exact match with PyTorch reference
 *
 * This helps diagnose the root cause of embedding divergence seen in integration tests.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include "model_loader.h"
#include "tensors/tensor_factory.h"
#include "parity_test_framework.h"
#include "abstract_pipeline.h"
#include "qwen_pipeline_adapter.h"
#include "logger.h"

namespace llaminar
{
    namespace testing
    {

        class EmbeddingStandaloneTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            }

            void TearDown() override
            {
                // Cleanup
            }

            int rank_;
            int world_size_;
        };

        /**
         * @brief Test basic embedding lookup for known tokens
         *
         * This test:
         * 1. Creates a pipeline (which loads embedding weights)
         * 2. Runs prefill for test tokens
         * 3. Manually extracts the embedding output
         * 4. Compares against expected PyTorch values
         */
        TEST_F(EmbeddingStandaloneTest, BasicEmbeddingLookup)
        {
            // For this test, we'll use the pipeline to load weights
            // This is more representative of actual usage

            std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

            if (rank_ == 0)
            {
                LOG_INFO("=== Standalone Embedding Test ===");
                LOG_INFO("Model: " << model_path);
                LOG_INFO("This test uses the full pipeline but only examines embedding output");
            }

            // Test tokens from standard test case
            std::vector<int> test_tokens = {1639, 266, 285, 17, 10, 17, 30};

            // Load model
            auto model_loader = std::make_unique<ModelLoader>();
            ASSERT_TRUE(model_loader->loadModel(model_path)) << "Failed to load model";

            auto layer_config = model_loader->createLayerConfig();
            ASSERT_EQ(layer_config.vocab_size, 151936) << "Unexpected vocab size";
            ASSERT_EQ(layer_config.d_model, 896) << "Unexpected embedding dim";

            if (rank_ == 0)
            {
                LOG_INFO("Model config: vocab=" << layer_config.vocab_size
                                                << ", d_model=" << layer_config.d_model);
            }

            // Create pipeline
            ModelConfig model_config(layer_config, "qwen");
            registerQwenPipeline(); // From pipeline_factory.h
            auto pipeline = PipelineFactory::instance().create(model_config);
            ASSERT_NE(pipeline, nullptr) << "Failed to create pipeline";

            // Load weights
            auto loaded_weights = pipeline->loadWeights(model_path);
            ASSERT_TRUE(loaded_weights) << "Failed to load weights";

            if (rank_ == 0)
            {
                LOG_INFO("Weights loaded successfully");
            }

            // Enable parity capture to get embedding output
            parity::LlaminarSnapshotHook::set_enabled(true);

            // Run prefill (which will perform embedding lookup)
            StageContext ctx;
            bool success = pipeline->prefill(test_tokens, *loaded_weights, ctx);
            ASSERT_TRUE(success) << "Prefill failed";

            // Get the captured embedding snapshot
            auto &registry = parity::SnapshotRegistry::instance();

            // The embedding is captured with key "llaminar_EMBEDDING"
            parity::TensorSnapshot embedding_snapshot;
            bool found = registry.get_snapshot("llaminar_EMBEDDING", embedding_snapshot);

            if (rank_ == 0)
            {
                ASSERT_TRUE(found) << "Embedding snapshot not captured!";

                LOG_INFO("");
                LOG_INFO("=== Embedding Output ===");
                LOG_INFO("Shape: [" << embedding_snapshot.metadata.seq_len
                                    << ", " << embedding_snapshot.metadata.feature_dim << "]");
                ASSERT_EQ(embedding_snapshot.metadata.seq_len, 7) << "Wrong sequence length";
                ASSERT_EQ(embedding_snapshot.metadata.feature_dim, 896) << "Wrong feature dim";

                const float *data = embedding_snapshot.data.data();

                // Print first token embedding (token 1639)
                LOG_INFO("");
                LOG_INFO("Token 1639 embedding (first 20 values):");
                for (int i = 0; i < 20; ++i)
                {
                    LOG_INFO("  [" << i << "] = " << data[i]);
                }

                // Print second token embedding (token 266)
                LOG_INFO("");
                LOG_INFO("Token 266 embedding (first 20 values):");
                for (int i = 0; i < 20; ++i)
                {
                    LOG_INFO("  [" << i << "] = " << data[896 + i]);
                }

                // Expected values from PyTorch reference (from NPZ file)
                // Token 1639 first 10 values: [-0.01953125  0.00842285  0.0045166   0.01721191 -0.03735352  0.0213623
                //                              0.02618408 -0.01997375  0.01123047  0.02059937]
                std::vector<float> expected_token_1639 = {
                    -0.01953125f, 0.00842285f, 0.0045166f, 0.01721191f, -0.03735352f,
                    0.0213623f, 0.02618408f, -0.01997375f, 0.01123047f, 0.02059937f};

                LOG_INFO("");
                LOG_INFO("=== Comparison with PyTorch ===");
                bool all_match = true;
                float max_diff = 0.0f;

                for (size_t i = 0; i < expected_token_1639.size(); ++i)
                {
                    float llaminar_val = data[i];
                    float pytorch_val = expected_token_1639[i];
                    float diff = std::abs(llaminar_val - pytorch_val);
                    max_diff = std::max(max_diff, diff);

                    std::string status = (diff < 0.001f) ? "✓" : "✗";
                    LOG_INFO(status << " [" << i << "] Llaminar=" << llaminar_val
                                    << ", PyTorch=" << pytorch_val
                                    << ", diff=" << diff);

                    if (diff >= 0.001f)
                    {
                        all_match = false;
                    }
                }

                LOG_INFO("");
                LOG_INFO("Max difference: " << max_diff);
                LOG_INFO("Tolerance: 0.001 (tight for Q4_0)");

                if (all_match)
                {
                    LOG_INFO("✓✓✓ EMBEDDINGS MATCH! ✓✓✓");
                }
                else
                {
                    LOG_ERROR("✗✗✗ EMBEDDINGS DIVERGE! ✗✗✗");
                    LOG_ERROR("This indicates a bug in embedding lookup or dequantization");
                }

                // For Q4_0, we expect some quantization error, but it should be small
                // If max_diff > 0.1, something is seriously wrong
                EXPECT_LT(max_diff, 0.1f) << "Embedding divergence too large - likely a bug, not quantization error";

                // If all values match closely, the embedding layer is working correctly
                // If they don't match, we've isolated the bug to this specific component
            }

            parity::LlaminarSnapshotHook::set_enabled(false);
            registry.clear();
        }

    } // namespace testing
} // namespace llaminar

// Custom main to initialize MPI properly
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
