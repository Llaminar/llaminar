/**
 * @file Test__Q8_1_vs_FP32_Parity.cpp
 * @brief Integration test comparing Q8_1 and FP32 activation precision paths
 *
 * This is a lightweight integration test that validates Q8_1 activation precision
 * produces functionally equivalent results to FP32 by comparing final logits:
 *
 *   1. Top-1 token prediction (must match for greedy sampling equivalence)
 *   2. Top-5 token overlap (>= 80% required)
 *   3. Cosine similarity of logits (informational, should be >= 0.90)
 *   4. KL divergence (informational, should be < 2.0)
 *
 * Test scenarios:
 *   - Prefill: Multi-token prompt processing
 *   - Incremental decode: Single-token autoregressive generation
 *
 * For detailed layer-by-layer comparisons with snapshots, see the E2E test:
 *   tests/v2/e2e/qwen2/Test__Q8_1_LayerByLayer_Divergence.cpp
 *
 * @author David Sanftenberg
 * @date 2025-12-10
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <iomanip>

#include "pipelines/qwen/Qwen2Pipeline.h"
#include "pipelines/PipelineConfig.h"
#include "loaders/ModelContext.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Compute cosine similarity between two float arrays
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Get top-K token indices from logits (sorted by score descending)
     */
    std::vector<int> get_topk(const float *logits, size_t vocab_size, int k)
    {
        std::vector<std::pair<float, int>> indexed(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i)
        {
            indexed[i] = {logits[i], static_cast<int>(i)};
        }
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });
        std::vector<int> topk(k);
        for (int i = 0; i < k; ++i)
        {
            topk[i] = indexed[i].second;
        }
        return topk;
    }

    /**
     * @brief Count overlap between two top-K lists
     */
    int count_overlap(const std::vector<int> &a, const std::vector<int> &b)
    {
        std::set<int> set_a(a.begin(), a.end());
        int overlap = 0;
        for (int t : b)
        {
            if (set_a.count(t) > 0)
            {
                overlap++;
            }
        }
        return overlap;
    }

    /**
     * @brief Compute KL divergence D_KL(P || Q) from logits
     */
    double compute_kl_divergence(const float *logits_p, const float *logits_q, size_t vocab_size)
    {
        // Compute softmax for both distributions
        auto softmax = [vocab_size](const float *logits) -> std::vector<double>
        {
            float max_logit = *std::max_element(logits, logits + vocab_size);
            std::vector<double> probs(vocab_size);
            double sum = 0.0;
            for (size_t i = 0; i < vocab_size; ++i)
            {
                probs[i] = std::exp(static_cast<double>(logits[i] - max_logit));
                sum += probs[i];
            }
            for (size_t i = 0; i < vocab_size; ++i)
            {
                probs[i] /= sum;
            }
            return probs;
        };

        auto p = softmax(logits_p);
        auto q = softmax(logits_q);

        // Compute KL divergence with numerical stability
        double kl = 0.0;
        constexpr double epsilon = 1e-10;
        for (size_t i = 0; i < vocab_size; ++i)
        {
            if (p[i] > epsilon)
            {
                kl += p[i] * std::log((p[i] + epsilon) / (q[i] + epsilon));
            }
        }
        return kl;
    }

} // namespace

/**
 * @brief Test fixture for Q8_1 vs FP32 activation precision parity
 */
class Test__Q8_1_vs_FP32_Parity : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::string model_path_;
    int rank_ = 0;
    int world_size_ = 1;

    // Parity thresholds - STRICT requirements for Q8_1 vs FP32 parity
    // Top-1 match is REQUIRED (checked separately as a hard failure)
    // Top-5 overlap must be at least 80% (4/5 tokens)
    static constexpr double MIN_TOP5_OVERLAP_RATIO = 0.80; // 4/5 tokens must match
    static constexpr double MIN_COSINE_SIMILARITY = 0.90;  // Reasonable similarity
    static constexpr double MAX_KL_DIVERGENCE = 2.0;       // Relaxed threshold

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Use Qwen 0.5B Q4_0 for fast testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);
        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        if (rank_ == 0)
        {
            LOG_INFO("[Q8_1 vs FP32 Parity] Loaded model: " << model_path_);
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a pipeline with specified activation precision
     */
    std::unique_ptr<Qwen2Pipeline> createPipeline(ActivationPrecision precision)
    {
        PipelineConfig config;
        config.activation_precision = precision;
        config.max_seq_len = 512;

        return std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, -1, nullptr, config, 1);
    }

    /**
     * @brief Compare logits between FP32 and Q8_1 pipelines
     * @param fp32_logits Logits from FP32 pipeline
     * @param q8_1_logits Logits from Q8_1 pipeline
     * @param vocab_size Vocabulary size
     * @param label Test label for logging
     * @return True if parity checks pass
     */
    bool compareLogits(const float *fp32_logits, const float *q8_1_logits,
                       size_t vocab_size, const std::string &label)
    {
        // Get top-5 predictions
        auto fp32_top5 = get_topk(fp32_logits, vocab_size, 5);
        auto q8_1_top5 = get_topk(q8_1_logits, vocab_size, 5);

        // Compute metrics
        int top5_overlap = count_overlap(fp32_top5, q8_1_top5);
        double top5_ratio = top5_overlap / 5.0;
        bool top1_match = (fp32_top5[0] == q8_1_top5[0]);
        double cosine = cosine_similarity(fp32_logits, q8_1_logits, vocab_size);
        double kl_div = compute_kl_divergence(fp32_logits, q8_1_logits, vocab_size);

        // Log results
        if (rank_ == 0)
        {
            LOG_INFO("");
            LOG_INFO("=== " << label << " ===");
            LOG_INFO("  Top-1 match: " << (top1_match ? "YES ✓" : "NO ✗")
                                       << " (FP32=" << fp32_top5[0] << ", Q8_1=" << q8_1_top5[0] << ")");
            LOG_INFO("  Top-5 overlap: " << top5_overlap << "/5 (" << std::fixed << std::setprecision(0)
                                         << (top5_ratio * 100) << "%)");
            LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(4) << cosine);
            LOG_INFO("  KL divergence: " << std::fixed << std::setprecision(4) << kl_div);

            // Show top-5 tokens
            LOG_INFO("  FP32 top-5: [");
            for (int i = 0; i < 5; ++i)
            {
                LOG_INFO("    " << i + 1 << ". token " << fp32_top5[i]
                                << " (logit=" << std::fixed << std::setprecision(2) << fp32_logits[fp32_top5[i]] << ")");
            }
            LOG_INFO("  ]");
            LOG_INFO("  Q8_1 top-5: [");
            for (int i = 0; i < 5; ++i)
            {
                LOG_INFO("    " << i + 1 << ". token " << q8_1_top5[i]
                                << " (logit=" << std::fixed << std::setprecision(2) << q8_1_logits[q8_1_top5[i]] << ")");
            }
            LOG_INFO("  ]");
        }

        // Check pass/fail criteria
        bool pass = true;

        // Top-1 mismatch is a FAILURE - greedy sampling must produce same token
        if (!top1_match)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Top-1 mismatch (FP32=" << fp32_top5[0] << ", Q8_1=" << q8_1_top5[0] << ")");
            pass = false;
        }

        if (top5_ratio < MIN_TOP5_OVERLAP_RATIO)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Top-5 overlap " << (top5_ratio * 100) << "% < "
                                               << (MIN_TOP5_OVERLAP_RATIO * 100) << "% threshold");
            pass = false;
        }

        if (cosine < MIN_COSINE_SIMILARITY)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ Cosine similarity " << cosine << " < " << MIN_COSINE_SIMILARITY << " threshold");
            pass = false;
        }

        if (kl_div > MAX_KL_DIVERGENCE)
        {
            if (rank_ == 0)
                LOG_ERROR("  ✗ KL divergence " << kl_div << " > " << MAX_KL_DIVERGENCE << " threshold");
            pass = false;
        }

        return pass;
    }
};

/**
 * @brief Test prefill parity between Q8_1 and FP32
 *
 * Runs a multi-token prompt through both pipelines and compares final logits.
 */
TEST_F(Test__Q8_1_vs_FP32_Parity, PrefillParity)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Q8_1 vs FP32 Parity Test: Prefill                             ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // Test prompt: "The quick brown fox jumps over the lazy dog"
    // Token IDs from Qwen2.5 tokenizer (without special tokens)
    // Same prompt as E2E test for consistency
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

    // Create pipelines
    auto fp32_pipeline = createPipeline(ActivationPrecision::FP32);
    auto q8_1_pipeline = createPipeline(ActivationPrecision::Q8_1);

    // Run prefill
    bool fp32_ok = fp32_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    bool q8_1_ok = q8_1_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));

    ASSERT_TRUE(fp32_ok) << "FP32 pipeline forward failed";
    ASSERT_TRUE(q8_1_ok) << "Q8_1 pipeline forward failed";

    // Get logits - pipeline->logits() already returns the LAST token's logits
    // via getLastTokenLogits(0), so no additional offset needed
    size_t vocab_size = model_ctx_->model().vocab_size;
    const float *fp32_logits = fp32_pipeline->logits();
    const float *q8_1_logits = q8_1_pipeline->logits();

    // Compare
    bool pass = compareLogits(fp32_logits, q8_1_logits, vocab_size, "Prefill Logits");

    EXPECT_TRUE(pass) << "Prefill parity check failed";
}

/**
 * @brief Test incremental decode parity between Q8_1 and FP32
 *
 * After prefill, generates one token autoregressively and compares.
 */
TEST_F(Test__Q8_1_vs_FP32_Parity, IncrementalDecodeParity)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Q8_1 vs FP32 Parity Test: Incremental Decode                  ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // Initial prompt: "The quick brown fox jumps over the lazy dog"
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

    // Create pipelines
    auto fp32_pipeline = createPipeline(ActivationPrecision::FP32);
    auto q8_1_pipeline = createPipeline(ActivationPrecision::Q8_1);

    // Prefill
    bool fp32_ok = fp32_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    bool q8_1_ok = q8_1_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));

    ASSERT_TRUE(fp32_ok) << "FP32 prefill failed";
    ASSERT_TRUE(q8_1_ok) << "Q8_1 prefill failed";

    // Get next token (greedy) from FP32 as the "correct" next token
    // Note: pipeline->logits() already returns the LAST token's logits
    size_t vocab_size = model_ctx_->model().vocab_size;
    const float *fp32_prefill_logits = fp32_pipeline->logits();

    auto fp32_top1 = get_topk(fp32_prefill_logits, vocab_size, 1);
    int next_token = fp32_top1[0];

    if (rank_ == 0)
    {
        LOG_INFO("Next token from FP32 prefill: " << next_token);
    }

    // Incremental decode: feed next_token to both pipelines
    std::vector<int> decode_tokens = {next_token};
    fp32_ok = fp32_pipeline->forward(decode_tokens.data(), 1);
    q8_1_ok = q8_1_pipeline->forward(decode_tokens.data(), 1);

    ASSERT_TRUE(fp32_ok) << "FP32 decode failed";
    ASSERT_TRUE(q8_1_ok) << "Q8_1 decode failed";

    // Get decode logits (single row)
    const float *fp32_decode_logits = fp32_pipeline->logits();
    const float *q8_1_decode_logits = q8_1_pipeline->logits();

    // Compare decode logits
    bool pass = compareLogits(fp32_decode_logits, q8_1_decode_logits, vocab_size, "Decode Step 1 Logits");

    EXPECT_TRUE(pass) << "Incremental decode parity check failed";
}

/**
 * @brief Test greedy sampling sequence equivalence
 *
 * Generates multiple tokens and checks that the sequences match.
 */
TEST_F(Test__Q8_1_vs_FP32_Parity, GreedySamplingSequence)
{
    if (rank_ == 0)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Q8_1 vs FP32 Parity Test: Greedy Sampling Sequence            ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // Initial prompt: "The quick brown fox jumps over the lazy dog"
    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    const int num_decode_steps = 5;

    // Create pipelines
    auto fp32_pipeline = createPipeline(ActivationPrecision::FP32);
    auto q8_1_pipeline = createPipeline(ActivationPrecision::Q8_1);

    // Prefill both
    fp32_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));
    q8_1_pipeline->forward(tokens.data(), static_cast<int>(tokens.size()));

    size_t vocab_size = model_ctx_->model().vocab_size;
    std::vector<int> fp32_sequence;
    std::vector<int> q8_1_sequence;

    // Get first token from prefill - logits() already points to last token
    auto fp32_top1 = get_topk(fp32_pipeline->logits(), vocab_size, 1);
    auto q8_1_top1 = get_topk(q8_1_pipeline->logits(), vocab_size, 1);

    fp32_sequence.push_back(fp32_top1[0]);
    q8_1_sequence.push_back(q8_1_top1[0]);

    // Decode steps
    for (int step = 0; step < num_decode_steps - 1; ++step)
    {
        // Feed each pipeline its own predicted token
        std::vector<int> fp32_next = {fp32_sequence.back()};
        std::vector<int> q8_1_next = {q8_1_sequence.back()};

        fp32_pipeline->forward(fp32_next.data(), 1);
        q8_1_pipeline->forward(q8_1_next.data(), 1);

        // Get next predictions
        fp32_top1 = get_topk(fp32_pipeline->logits(), vocab_size, 1);
        q8_1_top1 = get_topk(q8_1_pipeline->logits(), vocab_size, 1);

        fp32_sequence.push_back(fp32_top1[0]);
        q8_1_sequence.push_back(q8_1_top1[0]);
    }

    // Report sequences
    if (rank_ == 0)
    {
        LOG_INFO("");
        LOG_INFO("Generated sequences (" << num_decode_steps << " tokens):");
        LOG_INFO("  FP32: [");
        for (int t : fp32_sequence)
        {
            LOG_INFO("    " << t);
        }
        LOG_INFO("  ]");
        LOG_INFO("  Q8_1: [");
        for (int t : q8_1_sequence)
        {
            LOG_INFO("    " << t);
        }
        LOG_INFO("  ]");
    }

    // Count matches
    int matches = 0;
    for (int i = 0; i < num_decode_steps; ++i)
    {
        if (fp32_sequence[i] == q8_1_sequence[i])
        {
            matches++;
        }
    }

    double match_ratio = static_cast<double>(matches) / num_decode_steps;

    if (rank_ == 0)
    {
        LOG_INFO("  Sequence match: " << matches << "/" << num_decode_steps
                                      << " (" << std::fixed << std::setprecision(0) << (match_ratio * 100) << "%)");
    }

    // With Q8_1 quantization, accumulated error causes sequence divergence.
    // We expect at least 40% of tokens to match (first 1-2 tokens typically match,
    // then paths diverge due to different top-1 selections)
    EXPECT_GE(match_ratio, 0.40) << "Greedy sampling sequence diverged too much";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
