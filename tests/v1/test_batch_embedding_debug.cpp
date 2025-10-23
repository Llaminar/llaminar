/**
 * @file test_batch_embedding_debug.cpp
 * @brief Debug batch vs sequential at embedding level
 * @author David Sanftenberg
 * @date 2025-10-16
 */

#include <gtest/gtest.h>
#include "QwenPipelineAdapter.h"
#include "BatchQwenPipelineAdapter.h"
#include "BatchQwenPipeline.h"
#include "QwenPipeline.h"
#include "AbstractPipeline.h"
#include "ModelLoader.h"
#include "MpiContext.h"
#include "BatchPaddingUtils.h"
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>

using namespace llaminar;

TEST(BatchEmbeddingDebug, CompareEmbeddings)
{
    auto rank = MPIContext::capture().rank;
    std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // Disable COSMA
    setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);

    // Load model
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));

    TransformerLayerConfig layer_config = loader.createLayerConfig();
    ModelConfig batch_config(layer_config, "qwen_batch");
    ModelConfig seq_config(layer_config, "qwen");

    // Register pipelines
    registerQwenPipeline();
    registerBatchQwenPipeline();

    // Create simple test sequences
    std::vector<int> seq1 = {1, 2, 3, 4};
    std::vector<int> seq2 = {5, 6, 7, 8, 9};

    if (rank == 0)
    {
        std::cout << "\n=== Embedding Debug Test ===\n";
        std::cout << "Seq1: " << seq1.size() << " tokens\n";
        std::cout << "Seq2: " << seq2.size() << " tokens\n";
    }

    // ============================================
    // Get batch pipeline embedding
    // ============================================
    auto batch_pipeline_ptr = PipelineFactory::instance().create(batch_config);
    auto *batch_pipeline = dynamic_cast<BatchQwenPipeline *>(batch_pipeline_ptr.get());
    ASSERT_NE(batch_pipeline, nullptr);

    auto batch_weights_base = batch_pipeline->loadWeights(model_path);
    auto *batch_weights = dynamic_cast<BatchQwenWeights *>(batch_weights_base.get());
    ASSERT_NE(batch_weights, nullptr);

    // Call prepareEmbedding directly
    std::shared_ptr<TensorBase> batch_embedded;
    std::vector<std::vector<int>> batch_input = {seq1, seq2};
    ASSERT_TRUE(batch_pipeline->prepareEmbedding(batch_input, *batch_weights, batch_embedded));

    ASSERT_NE(batch_embedded, nullptr);
    auto batch_shape = batch_embedded->shape();
    ASSERT_EQ(batch_shape.size(), 3);
    int B = batch_shape[0];
    int T = batch_shape[1];
    int D = batch_shape[2];

    if (rank == 0)
    {
        std::cout << "Batch embedding shape: [" << B << ", " << T << ", " << D << "]\n";
    }

    // ============================================
    // Get sequential embeddings
    // ============================================
    std::vector<std::vector<float>> seq_embeddings(2);

    // Just reuse batch_weights - the embedding table is the same
    const auto &emb_weight = batch_weights->embedding();
    ASSERT_NE(emb_weight, nullptr);
    const float *emb_data = emb_weight->data();

    for (int i = 0; i < 2; ++i)
    {
        const auto &sequences = (i == 0) ? seq1 : seq2;
        int seq_len = sequences.size();
        seq_embeddings[i].resize(seq_len * D);

        for (int t = 0; t < seq_len; ++t)
        {
            int token_id = sequences[t];
            const float *src = emb_data + token_id * D;
            float *dst = seq_embeddings[i].data() + t * D;
            std::copy(src, src + D, dst);
        }

        if (rank == 0)
        {
            std::cout << "Sequential embedding " << i << " size: "
                      << seq_len << " x " << D << "\n";
        }
    }

    // ============================================
    // Compare embeddings
    // ============================================
    if (rank == 0)
    {
        std::cout << "\nComparing embeddings...\n";
    }

    const float *batch_data = batch_embedded->data();

    for (int b = 0; b < 2; ++b)
    {
        const auto &sequences = (b == 0) ? seq1 : seq2;
        int seq_len = sequences.size();

        size_t mismatches = 0;
        float max_diff = 0.0f;
        int max_diff_pos = -1;

        for (int t = 0; t < seq_len; ++t)
        {
            const float *batch_token = batch_data + (b * T + t) * D;
            const float *seq_token = seq_embeddings[b].data() + t * D;

            for (int d = 0; d < D; ++d)
            {
                float diff = std::abs(batch_token[d] - seq_token[d]);
                if (diff > max_diff)
                {
                    max_diff = diff;
                    max_diff_pos = t * D + d;
                }

                if (diff > 1e-6f)
                {
                    mismatches++;
                    if (mismatches <= 5 && rank == 0)
                    {
                        std::cout << "  Seq " << b << " token " << t << " dim " << d
                                  << ": batch=" << batch_token[d]
                                  << " seq=" << seq_token[d]
                                  << " diff=" << diff << "\n";
                    }
                }
            }
        }

        if (rank == 0)
        {
            if (mismatches == 0)
            {
                std::cout << "✓ Sequence " << b << " embeddings match perfectly\n";
            }
            else
            {
                std::cout << "✗ Sequence " << b << " has " << mismatches
                          << " mismatches (max diff: " << max_diff
                          << " at position " << max_diff_pos << ")\n";
            }
        }

        EXPECT_EQ(mismatches, 0) << "Sequence " << b << " embedding mismatch";
    }

    // ============================================
    // Verify padding is zero
    // ============================================
    if (rank == 0)
    {
        std::cout << "\nChecking padding regions...\n";
    }

    for (int b = 0; b < 2; ++b)
    {
        const auto &sequences = (b == 0) ? seq1 : seq2;
        int seq_len = sequences.size();

        size_t non_zero_padding = 0;

        for (int t = seq_len; t < T; ++t)
        {
            const float *padding_token = batch_data + (b * T + t) * D;

            for (int d = 0; d < D; ++d)
            {
                if (std::abs(padding_token[d]) > 1e-8f)
                {
                    non_zero_padding++;
                    if (non_zero_padding <= 3 && rank == 0)
                    {
                        std::cout << "  Padding at seq=" << b << " pos=" << t
                                  << " dim=" << d << " is " << padding_token[d] << "\n";
                    }
                }
            }
        }

        if (rank == 0)
        {
            if (non_zero_padding == 0)
            {
                std::cout << "✓ Sequence " << b << " padding is zero\n";
            }
            else
            {
                std::cout << "✗ Sequence " << b << " has " << non_zero_padding
                          << " non-zero padding values\n";
            }
        }

        EXPECT_EQ(non_zero_padding, 0) << "Sequence " << b << " padding not zero";
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
