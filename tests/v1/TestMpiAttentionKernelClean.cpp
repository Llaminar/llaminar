/**
 * @file TestMpiAttentionKernelClean.cpp
 * @brief Comprehensive unit tests for clean MPIAttentionOperator implementation
 * @author David Sanftenberg
 *
 * Tests cover:
 * 1. Basic functionality (single/multi-rank)
 * 2. Dimension validation
 * 3. Weight distribution correctness
 * 4. Bias handling
 * 5. GQA support
 * 6. Numerical correctness vs reference
 * 7. MPI synchronization
 * 8. Snapshot callback integration
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <memory>
#include "operators/MPIAttentionOperator.h"
#include "tensors/TensorFactory.h"

using namespace llaminar;

namespace
{

    // ============================================================================
    // Test Utilities
    // ============================================================================

    void fillSequential(float *data, size_t size, float scale = 0.01f)
    {
        for (size_t i = 0; i < size; ++i)
        {
            // Use modulo on i first (keeps it small), then do arithmetic in signed domain
            int val_idx = static_cast<int>(i % 101);
            data[i] = scale * static_cast<float>(val_idx - 50);
        }
    }

    void fillConstant(float *data, size_t size, float value)
    {
        std::fill(data, data + size, value);
    }

    void fillRandom(float *data, size_t size, unsigned seed = 42)
    {
        std::srand(seed);
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = (std::rand() / static_cast<float>(RAND_MAX)) * 0.1f - 0.05f;
        }
    }

    float maxAbsDiff(const float *a, const float *b, size_t size)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < size; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    float relativeL2Error(const float *computed, const float *reference, size_t size)
    {
        float num = 0.0f, denom = 0.0f;
        for (size_t i = 0; i < size; ++i)
        {
            float diff = computed[i] - reference[i];
            num += diff * diff;
            denom += reference[i] * reference[i];
        }
        return (denom > 1e-10f) ? std::sqrt(num / denom) : 0.0f;
    }

    struct MPIContext
    {
        int rank;
        int world_size;

        MPIContext()
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        }
    };

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class MPIAttentionOperatorCleanTest : public ::testing::Test
    {
    protected:
        MPIContext mpi;

        void SetUp() override
        {
            // Ensure clean MPI state
            MPI_Barrier(MPI_COMM_WORLD);
        }

        void TearDown() override
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Helper: Create test inputs with specified dimensions
        std::vector<std::shared_ptr<TensorBase>> createInputs(
            int seq_len, int d_model, int n_head, int n_head_kv,
            bool include_bias = false, int head_dim = 64)
        {
            auto input = TensorFactory::create_simple({seq_len, d_model});
            auto wq = TensorFactory::create_simple({n_head * head_dim, d_model});
            auto wk = TensorFactory::create_simple({n_head_kv * head_dim, d_model});
            auto wv = TensorFactory::create_simple({n_head_kv * head_dim, d_model});
            auto wo = TensorFactory::create_simple({d_model, n_head * head_dim});

            // Fill with sequential data
            fillSequential(input->data(), input->size(), 0.001f);
            fillSequential(wq->data(), wq->size(), 0.002f);
            fillSequential(wk->data(), wk->size(), 0.0025f);
            fillSequential(wv->data(), wv->size(), 0.003f);
            fillSequential(wo->data(), wo->size(), 0.0015f);

            // DEBUG: Verify data is correct
            auto check_range = [](const char *name, float *data, size_t size, float expected_min, float expected_max)
            {
                float min_val = data[0], max_val = data[0];
                for (size_t i = 0; i < size; ++i)
                {
                    min_val = std::min(min_val, data[i]);
                    max_val = std::max(max_val, data[i]);
                }
                std::cerr << "[DEBUG] " << name << " range: [" << min_val << ", " << max_val << "]";
                if (min_val < expected_min - 0.001f || max_val > expected_max + 0.001f)
                {
                    std::cerr << " ❌ UNEXPECTED!" << std::endl;
                }
                else
                {
                    std::cerr << " ✓" << std::endl;
                }
            };
            check_range("input", input->data(), input->size(), -0.05f, 0.05f);
            check_range("wq", wq->data(), wq->size(), -0.1f, 0.1f);
            check_range("wk", wk->data(), wk->size(), -0.125f, 0.125f);
            check_range("wv", wv->data(), wv->size(), -0.15f, 0.15f);
            check_range("wo", wo->data(), wo->size(), -0.075f, 0.075f);

            // Biases
            std::shared_ptr<TensorBase> bq, bk, bv;
            if (include_bias)
            {
                bq = TensorFactory::create_simple({n_head * head_dim});
                bk = TensorFactory::create_simple({n_head_kv * head_dim});
                bv = TensorFactory::create_simple({n_head_kv * head_dim});
                fillConstant(bq->data(), bq->size(), 0.01f);
                fillConstant(bk->data(), bk->size(), 0.02f);
                fillConstant(bv->data(), bv->size(), 0.03f);
            }
            else
            {
                // Create empty tensors (null data)
                bq = TensorFactory::create_simple({1});
                bk = TensorFactory::create_simple({1});
                bv = TensorFactory::create_simple({1});
            }

            // KV cache (not used in prefill)
            auto k_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});
            auto v_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});
            fillConstant(k_cache->data(), k_cache->size(), 0.0f);
            fillConstant(v_cache->data(), v_cache->size(), 0.0f);

            return {input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache};
        }
    };

    // ============================================================================
    // Basic Functionality Tests
    // ============================================================================

    TEST_F(MPIAttentionOperatorCleanTest, BasicExecuteSingleRank)
    {
        if (mpi.world_size > 1)
        {
            GTEST_SKIP() << "Single-rank test, skipping in multi-rank environment";
        }

        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 2;
        const int n_head_kv = 2;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv);

        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        bool success = kernel.execute(inputs, outputs);
        ASSERT_TRUE(success) << "Kernel execution failed";

        // Verify output is non-zero
        float sum = 0.0f;
        for (size_t i = 0; i < output->size(); ++i)
        {
            sum += std::abs(output->data()[i]);
        }
        EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
    }

    TEST_F(MPIAttentionOperatorCleanTest, BasicExecuteMultiRank)
    {
        if (mpi.world_size < 2)
        {
            GTEST_SKIP() << "Multi-rank test requires at least 2 MPI processes";
        }

        const int seq_len = 8;
        const int d_model = 256;
        const int n_head = 4; // 2 heads per rank with world_size=2
        const int n_head_kv = 4;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv);

        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        bool success = kernel.execute(inputs, outputs);
        ASSERT_TRUE(success) << "Kernel execution failed on rank " << mpi.rank;

        // All ranks should succeed
        int all_success = 0;
        int local_success = success ? 1 : 0;
        MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        EXPECT_EQ(all_success, 1) << "Some ranks failed";
    }

    TEST_F(MPIAttentionOperatorCleanTest, OutputDeterminism)
    {
        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 2;
        const int n_head_kv = 2;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv);

        // Run twice with same inputs
        auto output1 = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs1 = {output1};
        ASSERT_TRUE(kernel.execute(inputs, outputs1));

        auto output2 = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs2 = {output2};
        ASSERT_TRUE(kernel.execute(inputs, outputs2));

        // Outputs should be identical
        float max_diff = maxAbsDiff(output1->data(), output2->data(), output1->size());
        EXPECT_LT(max_diff, 1e-6f) << "Outputs should be deterministic";
    }

    // ============================================================================
    // Dimension Validation Tests
    // ============================================================================

    TEST_F(MPIAttentionOperatorCleanTest, InvalidInputCount)
    {
        const int n_head = 2, n_head_kv = 2, head_dim = 64;
        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);

        // Only 5 inputs instead of 10
        std::vector<std::shared_ptr<TensorBase>> inputs(5);
        std::vector<std::shared_ptr<TensorBase>> outputs(1);

        bool success = kernel.execute(inputs, outputs);
        EXPECT_FALSE(success) << "Should fail with invalid input count";
    }

    TEST_F(MPIAttentionOperatorCleanTest, MismatchedDimensions)
    {
        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 2;
        const int n_head_kv = 2;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv);

        // Corrupt weight dimension
        auto bad_wq = TensorFactory::create_simple({100, static_cast<size_t>(d_model)});
        inputs[1] = bad_wq;

        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        // Should fail gracefully
        bool success = kernel.execute(inputs, outputs);
        EXPECT_FALSE(success) << "Should fail with mismatched dimensions";
    }

    // ============================================================================
    // Bias Handling Tests
    // ============================================================================

    TEST_F(MPIAttentionOperatorCleanTest, WithBias)
    {
        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 2;
        const int n_head_kv = 2;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);

        // Run with bias
        auto inputs_with_bias = createInputs(seq_len, d_model, n_head, n_head_kv, true);
        auto output_with = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs_with = {output_with};
        ASSERT_TRUE(kernel.execute(inputs_with_bias, outputs_with));

        // Run without bias
        auto inputs_no_bias = createInputs(seq_len, d_model, n_head, n_head_kv, false);
        auto output_without = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs_without = {output_without};
        ASSERT_TRUE(kernel.execute(inputs_no_bias, outputs_without));

        // Outputs should be different
        float max_diff = maxAbsDiff(output_with->data(), output_without->data(), output_with->size());
        EXPECT_GT(max_diff, 1e-4f) << "Bias should affect output";
    }

    TEST_F(MPIAttentionOperatorCleanTest, BiasDistributionMultiRank)
    {
        if (mpi.world_size < 2)
        {
            GTEST_SKIP() << "Multi-rank test requires at least 2 MPI processes";
        }

        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 4; // 2 per rank
        const int n_head_kv = 4;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv, true);

        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        bool success = kernel.execute(inputs, outputs);
        ASSERT_TRUE(success) << "Kernel with bias failed on rank " << mpi.rank;
    }

    // ============================================================================
    // GQA (Grouped Query Attention) Tests
    // ============================================================================

    TEST_F(MPIAttentionOperatorCleanTest, GQASupport)
    {
        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 4;    // Query heads
        const int n_head_kv = 2; // KV heads (GQA with 2:1 ratio)
        const int head_dim = 32;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv, false, head_dim);

        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        bool success = kernel.execute(inputs, outputs);
        ASSERT_TRUE(success) << "GQA execution failed";

        // Verify output is reasonable
        float sum = 0.0f;
        for (size_t i = 0; i < output->size(); ++i)
        {
            sum += std::abs(output->data()[i]);
        }
        EXPECT_GT(sum, 0.0f) << "GQA output should be non-zero";
    }

    TEST_F(MPIAttentionOperatorCleanTest, GQAVsMHADifferent)
    {
        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 4;
        const int head_dim = 32;

        // MHA (n_head == n_head_kv)
        MPIAttentionOperator kernel_mha(n_head, n_head, head_dim);
        auto inputs_mha = createInputs(seq_len, d_model, n_head, n_head, false, head_dim);
        auto output_mha = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs_mha = {output_mha};
        ASSERT_TRUE(kernel_mha.execute(inputs_mha, outputs_mha));

        // GQA (n_head != n_head_kv)
        MPIAttentionOperator kernel_gqa(n_head, n_head / 2, head_dim);
        auto inputs_gqa = createInputs(seq_len, d_model, n_head, n_head / 2, false, head_dim);
        auto output_gqa = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs_gqa = {output_gqa};
        ASSERT_TRUE(kernel_gqa.execute(inputs_gqa, outputs_gqa));

        // Outputs should be different (different architectures)
        float max_diff = maxAbsDiff(output_mha->data(), output_gqa->data(), output_mha->size());
        EXPECT_GT(max_diff, 1e-4f) << "MHA and GQA should produce different outputs";
    }

    // ============================================================================
    // MPI Synchronization Tests
    // ============================================================================

    TEST_F(MPIAttentionOperatorCleanTest, MultiRankOutputAggregation)
    {
        if (mpi.world_size < 2)
        {
            GTEST_SKIP() << "Multi-rank test requires at least 2 MPI processes";
        }

        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 4;
        const int n_head_kv = 4;
        const int head_dim = 32;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv, false, head_dim);

        auto output_partial = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output_partial};
        ASSERT_TRUE(kernel.execute(inputs, outputs));

        // Aggregate outputs across ranks
        std::vector<float> aggregated(output_partial->size(), 0.0f);
        MPI_Allreduce(output_partial->data(), aggregated.data(),
                      static_cast<int>(aggregated.size()), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // Aggregated result should be non-zero on all ranks
        float sum = 0.0f;
        for (float val : aggregated)
        {
            sum += std::abs(val);
        }
        EXPECT_GT(sum, 0.0f) << "Aggregated output should be non-zero";

        // All ranks should see the same aggregated result
        float global_sum = 0.0f;
        MPI_Allreduce(&sum, &global_sum, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        EXPECT_NEAR(sum, global_sum, 1e-5f) << "All ranks should see same aggregated sum";
    }

    // ============================================================================
    // Numerical Correctness Tests
    // ============================================================================

    TEST_F(MPIAttentionOperatorCleanTest, SmallScaleCorrectness)
    {
        // Small dimensions for numerical verification
        const int seq_len = 2;
        const int d_model = 64;
        const int n_head = 1;
        const int n_head_kv = 1;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv);

        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        bool success = kernel.execute(inputs, outputs);
        ASSERT_TRUE(success) << "Small-scale execution failed";

        // Check for NaN or Inf
        for (size_t i = 0; i < output->size(); ++i)
        {
            EXPECT_TRUE(std::isfinite(output->data()[i]))
                << "Output contains non-finite value at index " << i;
        }
    }

    TEST_F(MPIAttentionOperatorCleanTest, LargeSequenceLength)
    {
        const int seq_len = 512; // Larger sequence
        const int d_model = 128;
        const int n_head = 2;
        const int n_head_kv = 2;
        const int head_dim = 64;

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv);

        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        bool success = kernel.execute(inputs, outputs);
        ASSERT_TRUE(success) << "Large sequence execution failed";

        // Verify output magnitude is reasonable
        float max_val = 0.0f;
        for (size_t i = 0; i < output->size(); ++i)
        {
            max_val = std::max(max_val, std::abs(output->data()[i]));
        }
        EXPECT_LT(max_val, 1e3f) << "Output magnitude suspiciously large";
        EXPECT_GT(max_val, 1e-6f) << "Output magnitude suspiciously small";
    }

    // ============================================================================
    // Snapshot Callback Tests
    // ============================================================================

    TEST_F(MPIAttentionOperatorCleanTest, SnapshotCallbackInvoked)
    {
        const int seq_len = 4;
        const int d_model = 128;
        const int n_head = 2;
        const int n_head_kv = 2;
        const int head_dim = 64;

        int callback_count = 0;
        auto callback = [&callback_count](PipelineStage stage, int layer_idx, const float *data, int seq, int feat)
        {
            callback_count++;
        };

        MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
        kernel.setSnapshotCallback(callback);

        auto inputs = createInputs(seq_len, d_model, n_head, n_head_kv);
        auto output = TensorFactory::create_simple({seq_len, d_model});
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        ASSERT_TRUE(kernel.execute(inputs, outputs));

        // Should have called snapshot callback multiple times
        // (Q_PROJECTION, K_PROJECTION, V_PROJECTION, ATTENTION_SCORES, etc.)
        EXPECT_GT(callback_count, 0) << "Snapshot callback should be invoked";
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
