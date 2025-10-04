#include "distributed_transformer_pipeline.h" // DistributedTransformerPipeline
#include "logger.h"
#include "test_timeout_guard.h"
#include "tensors/tensor_factory.h"
#include <gtest/gtest.h>
#include "test_mpi_utils.h"
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace llaminar;

class DistributedTransformerPipelineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Verify MPI is initialized
        int flag;
        MPI_Initialized(&flag);
        if (!flag)
        {
            throw std::runtime_error("MPI should be initialized before running tests");
        }

        // Set up test configuration for a small transformer model
        config_.n_head = 8;
        config_.n_head_kv = 8;
        config_.head_dim = 64;
        config_.d_model = 512; // 8 * 64
        config_.d_ff = 2048;   // 4 * d_model
        config_.vocab_size = 1000;
        config_.max_seq_len = 64;
        config_.n_layers = 2; // Small for testing
        config_.eps = 1e-6f;

        // Create transformer pipeline
        pipeline_ = createDistributedTransformerPipeline(config_);

        // Initialize random generator with fixed seed for reproducibility
        generator_.seed(42);
    }

    void TearDown() override
    {
        pipeline_.reset();
    }

    void fillRandomData(std::shared_ptr<TensorBase> &tensor, float min_val = -0.01f, float max_val = 0.01f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        auto *data = const_cast<float *>(tensor->data());
        size_t total_size = 1;
        for (int dim : tensor->shape())
        {
            total_size *= dim;
        }
        for (size_t i = 0; i < total_size; ++i)
        {
            data[i] = dist(generator_);
        }
    }

    DistributedTransformerPipeline::ModelWeights createTestWeights()
    {
        DistributedTransformerPipeline::ModelWeights weights;

        // Token embedding
        weights.token_embedding = llaminar::TensorFactory::create_simple(std::vector<int>{config_.vocab_size, config_.d_model});
        fillRandomData(weights.token_embedding);

        // Output layer weights
        weights.output_norm_weight = llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model});
        fillRandomData(weights.output_norm_weight, 0.8f, 1.2f); // Norm weights should be near 1

        weights.lm_head = llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.vocab_size});
        fillRandomData(weights.lm_head);

        // Per-layer weights
        for (int i = 0; i < config_.n_layers; ++i)
        {
            // Attention weights
            weights.attn_norm_weight.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model}));
            fillRandomData(weights.attn_norm_weight[i], 0.8f, 1.2f);

            weights.wq.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head * config_.head_dim}));
            fillRandomData(weights.wq[i]);

            weights.wk.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head_kv * config_.head_dim}));
            fillRandomData(weights.wk[i]);

            weights.wv.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head_kv * config_.head_dim}));
            fillRandomData(weights.wv[i]);

            weights.wo.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.n_head * config_.head_dim, config_.d_model}));
            fillRandomData(weights.wo[i]);

            // FFN weights
            weights.ffn_norm_weight.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model}));
            fillRandomData(weights.ffn_norm_weight[i], 0.8f, 1.2f);

            weights.w_gate.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.d_ff}));
            fillRandomData(weights.w_gate[i]);

            weights.w_up.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.d_ff}));
            fillRandomData(weights.w_up[i]);

            weights.w_down.push_back(llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_ff, config_.d_model}));
            fillRandomData(weights.w_down[i]);
        }

        return weights;
    }

    std::vector<int> createTestTokens(int seq_len)
    {
        std::vector<int> tokens;
        std::uniform_int_distribution<int> dist(0, config_.vocab_size - 1);

        for (int i = 0; i < seq_len; ++i)
        {
            tokens.push_back(dist(generator_));
        }

        return tokens;
    }

    void validateOutput(const std::shared_ptr<TensorBase> &output, int seq_len)
    {
        ASSERT_TRUE(output);
        ASSERT_EQ(output->shape().size(), 2);
        ASSERT_EQ(output->shape()[0], seq_len);
        ASSERT_EQ(output->shape()[1], config_.vocab_size);

        // Check output is not zero and has reasonable values
        bool has_nonzero = false;
        bool has_reasonable_values = true;
        float max_val = -std::numeric_limits<float>::infinity();
        float min_val = std::numeric_limits<float>::infinity();

        const auto *output_data = output->data();
        size_t output_size = output->shape()[0] * output->shape()[1];
        for (size_t i = 0; i < output_size; ++i)
        {
            const auto &val = output_data[i];
            if (std::abs(val) > 1e-6f)
            {
                has_nonzero = true;
            }
            if (std::abs(val) > 100.0f)
            { // Check for exploding values
                has_reasonable_values = false;
            }
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
        }

        EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";
        EXPECT_TRUE(has_reasonable_values) << "Output values should be reasonable (not exploding)";

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            LOG_INFO("Output validation: min=" << std::fixed << std::setprecision(6) << min_val
                                               << ", max=" << max_val << ", shape=[" << seq_len << ", " << config_.vocab_size << "]");
        }
    }

protected:
    std::unique_ptr<DistributedTransformerPipeline> pipeline_;
    DistributedTransformerPipeline::LayerConfig config_;
    std::mt19937 generator_;
};

// BasicFunctionality test removed (covered by ValidationTests + parity & sequence tests)

TEST_F(DistributedTransformerPipelineTest, ValidationTests)
{
    // Test weight validation
    auto valid_weights = createTestWeights();
    EXPECT_TRUE(pipeline_->validate(valid_weights));

    // Test invalid embedding shape
    auto invalid_weights = valid_weights;
    invalid_weights.token_embedding = llaminar::TensorFactory::create_simple(std::vector<int>{config_.vocab_size + 1, config_.d_model});
    EXPECT_FALSE(pipeline_->validate(invalid_weights));

    // Test missing layer weights
    invalid_weights = valid_weights;
    invalid_weights.wq.pop_back();
    EXPECT_FALSE(pipeline_->validate(invalid_weights));

    // Test invalid attention weight shape
    invalid_weights = valid_weights;
    invalid_weights.wq[0] = llaminar::TensorFactory::create_simple(std::vector<int>{config_.d_model, config_.n_head * config_.head_dim + 1});
    EXPECT_FALSE(pipeline_->validate(invalid_weights));
}

TEST_F(DistributedTransformerPipelineTest, DifferentSequenceLengths)
{
    // Test with various sequence lengths
    auto weights = createTestWeights();

    // Candidate lengths now include 1 to exercise replicated small-sequence fast path
    std::vector<int> candidate_lengths = {1, 2, 8, 16};

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    for (int seq_len : candidate_lengths)
    {
        // When seq_len < world_size the pipeline should activate replicated fast path.
        bool expect_replicated = seq_len < world_size;

        auto tokens = createTestTokens(seq_len);
        auto output = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});

        ASSERT_TRUE(pipeline_->execute(tokens, weights, output))
            << "Failed for sequence length " << seq_len;
        validateOutput(output, seq_len);

        // Simple cross-rank consistency check for replicated path: gather first token's first 5 logits
        if (expect_replicated)
        {
            // Collect first token's first 5 logits reference on rank 0 then broadcast
            std::array<float, 5> ref_vals{};
            int rank = -1;
            int rc_rank = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rc_rank != MPI_SUCCESS)
            {
                LOG_ERROR("[ReplicatedDiag] MPI_Comm_rank failed inside replicated check (rc=" << rc_rank << ")");
            }
            else
            {
                LOG_INFO("[ReplicatedDiag] Enter replicated block: seq_len=" << seq_len << ", rank=" << rank << ", world_size=" << world_size);
            }
            if (rank == 0)
            {
                // Diagnostic: log first 5 logits before capturing
                std::ostringstream pre_capture;
                pre_capture << "[ReplicatedDiag] Rank 0 output first 5 logits before capture (seq_len=" << seq_len << "): ";
                for (int i = 0; i < 5 && i < config_.vocab_size; ++i)
                {
                    pre_capture << output->data()[i] << (i + 1 < 5 ? ' ' : '\0');
                }
                LOG_INFO(pre_capture.str());
                for (int i = 0; i < 5 && i < config_.vocab_size; ++i)
                {
                    ref_vals[i] = output->data()[i];
                }
                std::ostringstream post_capture;
                post_capture << "[ReplicatedDiag] Rank 0 ref_vals after capture: ";
                for (int i = 0; i < 5; ++i)
                {
                    post_capture << ref_vals[i] << (i + 1 < 5 ? ' ' : '\0');
                }
                LOG_INFO(post_capture.str());
                // Also emit to stderr directly to bypass logger filtering (debug)
                std::cerr << "[ReplicatedDiag-STDERR] Rank 0 captured first logits: ";
                for (int i = 0; i < 5; ++i)
                    std::cerr << ref_vals[i] << (i + 1 < 5 ? ' ' : '\n');
            }
            MPI_Bcast(ref_vals.data(), 5, MPI_FLOAT, 0, MPI_COMM_WORLD);
            // Post-broadcast diagnostics on all ranks
            {
                std::ostringstream post_bcast;
                post_bcast << "[ReplicatedDiag] Rank " << rank << " ref_vals after broadcast: ";
                for (int i = 0; i < 5; ++i)
                {
                    post_bcast << ref_vals[i] << (i + 1 < 5 ? ' ' : '\0');
                }
                post_bcast << " | local output first 5: ";
                for (int i = 0; i < 5 && i < config_.vocab_size; ++i)
                {
                    post_bcast << output->data()[i] << (i + 1 < 5 ? ' ' : '\0');
                }
                LOG_INFO(post_bcast.str());
                std::cerr << "[ReplicatedDiag-STDERR] Rank " << rank << " broadcast ref_vals: ";
                for (int i = 0; i < 5; ++i)
                    std::cerr << ref_vals[i] << (i + 1 < 5 ? ' ' : '\n');
            }
            for (int i = 0; i < 5 && i < config_.vocab_size; ++i)
            {
                float local_val = output->data()[i];
                EXPECT_NEAR(local_val, ref_vals[i], 1e-6f)
                    << "Replicated fast path mismatch at logit index " << i << " for seq_len=" << seq_len;
            }
        }
    }
}

// Lightweight FNV-1a 64-bit hash for validating cross-rank identical outputs
static uint64_t fnv1a_hash(const float *data, size_t count)
{
    const uint64_t FNV_OFFSET = 1469598103934665603ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < count; ++i)
    {
        // Reinterpret float bits to stable uint32_t then mix
        uint32_t bits;
        std::memcpy(&bits, &data[i], sizeof(uint32_t));
        h ^= bits;
        h *= FNV_PRIME;
    }
    return h;
}

TEST_F(DistributedTransformerPipelineTest, SmallSequenceFastPath)
{
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2)
    {
        GTEST_SKIP() << "SmallSequenceFastPath test requires world_size >= 2 to trigger fast path reliably";
    }
    // Reset counter
    llaminar::DistributedTransformerPipeline::resetSmallSeqFastPathCount();

    // Choose seq_len=1 (< world_size) to force fast path
    int seq_len = 1;
    auto tokens = createTestTokens(seq_len);
    auto weights = createTestWeights();
    auto output = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});

    ASSERT_TRUE(pipeline_->execute(tokens, weights, output));
    validateOutput(output, seq_len);

    // Hash first row logits on all ranks and reduce via MPI_Allreduce (XOR) to ensure identical
    uint64_t local_hash = fnv1a_hash(output->data(), config_.vocab_size);
    uint64_t xor_hash = 0;
    MPI_Allreduce(&local_hash, &xor_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
    EXPECT_EQ(xor_hash, 0ULL) << "Ranks produced differing logits in small sequence fast path";

    // Counter should have incremented exactly once locally
    EXPECT_EQ(llaminar::DistributedTransformerPipeline::getSmallSeqFastPathCount(), (size_t)1);
}

// ConsistencyAcrossRuns test removed (redundant with parity/incremental validation)

// PerformanceBenchmark test removed (moved to dedicated benchmarking binaries)

TEST_F(DistributedTransformerPipelineTest, LoadBalancingAnalysis)
{
    // Test load balancing across MPI processes
    const int seq_len = 8;
    auto tokens = createTestTokens(seq_len);
    auto weights = createTestWeights();
    auto output = llaminar::TensorFactory::create_simple(std::vector<int>{seq_len, config_.vocab_size});

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(pipeline_->execute(tokens, weights, output));
    auto end = std::chrono::high_resolution_clock::now();

    double local_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Gather timing from all processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> all_times(size);
    MPI_Gather(&local_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double total_time = std::accumulate(all_times.begin(), all_times.end(), 0.0);
        double avg_time = total_time / size;
        double min_time = *std::min_element(all_times.begin(), all_times.end());
        double max_time = *std::max_element(all_times.begin(), all_times.end());
        double load_imbalance = (max_time - min_time) / avg_time * 100.0;

        LOG_INFO("Load balancing analysis:");
        LOG_INFO("  Process times: " << std::fixed << std::setprecision(2));
        for (int i = 0; i < size; ++i)
        {
            LOG_INFO("    Rank " << i << ": " << all_times[i] << "ms");
        }
        LOG_INFO("  Load imbalance: " << load_imbalance << "%");

        // Good load balancing should have less than 20% imbalance
        EXPECT_LT(load_imbalance, 20.0) << "Load imbalance too high";
    }

    validateOutput(output, seq_len);
}

// Main function for running MPI tests
LLAMINAR_DEFINE_GTEST_MPI_MAIN();