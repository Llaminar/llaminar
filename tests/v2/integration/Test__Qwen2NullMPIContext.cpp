/**
 * @file Test__Qwen2NullMPIContext.cpp
 * @brief Unit tests for Qwen2Pipeline with null MPI context (single-rank mode)
 * @author David Sanftenberg
 *
 * Regression tests for scenarios where mpi_ctx_ is nullptr:
 * - Single-rank execution without MPI
 * - Debug logging that checks MPI rank
 * - RoPE application in non-MPI mode
 *
 * Background: Originally, code assumed MPI context was always available and
 * would crash when calling mpi_ctx_->rank() in single-rank mode.
 */

#include <gtest/gtest.h>
#include "v2/pipelines/qwen/Qwen2Pipeline.h"
#include "v2/loaders/ModelLoader.h"
#include "v2/loaders/ModelContext.h"
#include "v2/utils/MPIContext.h"
#include <memory>
#include <vector>
#include <fstream>

using namespace llaminar2;

/**
 * @brief Test fixture for Qwen2Pipeline null MPI context scenarios
 */
class Qwen2NullMPIContextTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Load a small model for testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        // Check if model exists
        std::ifstream f(model_path_);
        if (!f.good())
        {
            GTEST_SKIP() << "Model file not found: " << model_path_;
        }

        try
        {
            model_ctx_ = ModelContext::create(model_path_);
            if (!model_ctx_)
            {
                GTEST_SKIP() << "Failed to load model: " << model_path_;
            }
        }
        catch (const std::exception &e)
        {
            GTEST_SKIP() << "Exception loading model: " << e.what();
        }
    }

    std::string model_path_;
    std::shared_ptr<ModelContext> model_ctx_;
};

/**
 * @brief Test Qwen2Pipeline construction with null MPI context
 *
 * Regression test: Ensures pipeline can be created without MPI context
 */
TEST_F(Qwen2NullMPIContextTest, ConstructionWithNullMPIContext)
{
    ASSERT_NE(model_ctx_, nullptr);

    // Create pipeline with null MPI context (single-rank mode)
    std::shared_ptr<MPIContext> null_mpi_ctx = nullptr;

    EXPECT_NO_THROW({
        auto pipeline = std::make_unique<Qwen2Pipeline>(
            model_ctx_,
            null_mpi_ctx, // null MPI context
            -1,           // CPU device
            nullptr,      // no placement map
            PipelineConfig{},
            1 // batch_size = 1
        );

        EXPECT_NE(pipeline, nullptr);
    }) << "Pipeline construction should not crash with null MPI context";
}

/**
 * @brief Test single token inference without MPI context
 *
 * Regression test: Ensures forward pass works in single-rank mode
 * This previously crashed when logging code tried to call mpi_ctx_->rank()
 */
TEST_F(Qwen2NullMPIContextTest, SingleTokenInferenceNoMPI)
{
    ASSERT_NE(model_ctx_, nullptr);

    // Create pipeline without MPI
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_,
        nullptr, // null MPI context
        -1,
        nullptr,
        PipelineConfig{},
        1);

    ASSERT_NE(pipeline, nullptr);

    // Single token input (BOS token for Qwen 2.5)
    std::vector<int> tokens = {151644};

    // This should not crash (previously crashed in RoPE debug logging)
    bool success = false;
    EXPECT_NO_THROW({
        success = pipeline->forward(tokens.data(), tokens.size());
    }) << "Forward pass should not crash with null MPI context";

    EXPECT_TRUE(success) << "Forward pass should succeed";

    // Verify we can get logits
    const float *logits = pipeline->getLogits(0);
    EXPECT_NE(logits, nullptr) << "Should be able to get logits";
}

/**
 * @brief Test multiple token inference without MPI context
 */
TEST_F(Qwen2NullMPIContextTest, MultiTokenInferenceNoMPI)
{
    ASSERT_NE(model_ctx_, nullptr);

    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_,
        nullptr, // null MPI context
        -1,
        nullptr,
        PipelineConfig{},
        1);

    ASSERT_NE(pipeline, nullptr);

    // Multi-token input
    std::vector<int> tokens = {151644, 9707, 374, 264}; // "This is a"

    bool success = false;
    EXPECT_NO_THROW({
        success = pipeline->forward(tokens.data(), tokens.size());
    }) << "Multi-token forward should not crash with null MPI context";

    EXPECT_TRUE(success);
}

/**
 * @brief Test that RoPE debugging code handles null MPI context
 *
 * Regression test: The bug was specifically in RoPE debug logging at layer 0
 * which checked `mpi_ctx_->rank() == 0` without null check
 */
TEST_F(Qwen2NullMPIContextTest, RoPEDebugLoggingNoMPI)
{
    ASSERT_NE(model_ctx_, nullptr);

    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_,
        nullptr,
        -1,
        nullptr,
        PipelineConfig{},
        1);

    ASSERT_NE(pipeline, nullptr);

    // Single token - will hit layer 0 RoPE code path
    std::vector<int> tokens = {151644};

    // The critical code path is:
    // if (layer_idx == 0 && (!mpi_ctx_ || mpi_ctx_->rank() == 0))
    // Previously was: if (layer_idx == 0 && mpi_ctx_->rank() == 0)  // CRASH!

    EXPECT_NO_THROW({
        pipeline->forward(tokens.data(), tokens.size());
    }) << "RoPE debug logging in layer 0 should not crash with null MPI";
}

/**
 * @brief Test batch inference without MPI context
 */
TEST_F(Qwen2NullMPIContextTest, BatchInferenceNoMPI)
{
    ASSERT_NE(model_ctx_, nullptr);

    // Create pipeline with batch_size > 1
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_,
        nullptr,
        -1,
        nullptr,
        PipelineConfig{},
        2 // batch_size = 2
    );

    ASSERT_NE(pipeline, nullptr);

    // Batch input
    std::vector<std::vector<int>> batch_tokens = {
        {151644, 9707}, // Sequence 1
        {151644, 374}   // Sequence 2
    };

    bool success = false;
    EXPECT_NO_THROW({
        success = pipeline->forward_batch(batch_tokens);
    }) << "Batch forward should not crash with null MPI context";

    EXPECT_TRUE(success);
}

/**
 * @brief Test incremental decode without MPI context
 */
TEST_F(Qwen2NullMPIContextTest, IncrementalDecodeNoMPI)
{
    ASSERT_NE(model_ctx_, nullptr);

    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_,
        nullptr,
        -1,
        nullptr,
        PipelineConfig{},
        1);

    ASSERT_NE(pipeline, nullptr);

    // Prefill with multiple tokens
    std::vector<int> prefill_tokens = {151644, 9707, 374};
    bool success = pipeline->forward(prefill_tokens.data(), prefill_tokens.size());
    ASSERT_TRUE(success);

    // Incremental decode (single token at a time)
    for (int i = 0; i < 3; ++i)
    {
        std::vector<int> next_token = {264 + i}; // arbitrary token
        EXPECT_NO_THROW({
            success = pipeline->forward(next_token.data(), next_token.size());
        }) << "Incremental decode step "
           << i << " should not crash";
        EXPECT_TRUE(success);
    }
}

/**
 * @brief Verify MPI context check pattern is used consistently
 *
 * This is more of a code pattern test - verifying the fix is applied
 */
TEST_F(Qwen2NullMPIContextTest, MPIContextCheckPattern)
{
    // This test just documents the correct pattern for future code

    std::shared_ptr<MPIContext> mpi_ctx = nullptr;

    // WRONG (would crash):
    // if (mpi_ctx->rank() == 0) { ... }

    // CORRECT pattern:
    if (!mpi_ctx || mpi_ctx->rank() == 0)
    {
        // This executes in single-rank mode (null MPI)
        // OR in multi-rank mode on rank 0
        SUCCEED() << "Correct pattern demonstrated";
    }

    // Alternative correct pattern:
    if (mpi_ctx && mpi_ctx->rank() != 0)
    {
        // Only executes in multi-rank mode on non-zero ranks
        FAIL() << "Should not reach here with null MPI context";
    }
}
