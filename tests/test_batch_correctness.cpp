/**
 * @file test_batch_correctness.cpp
 * @brief Batch vs sequential correctness validation with real models
 * @author David Sanftenberg
 * @date 2025-01-16
 */

#include <gtest/gtest.h>
#include "QwenPipelineAdapter.h"
#include "BatchQwenPipelineAdapter.h"
#include "AbstractPipeline.h"
#include "ModelLoader.h"
#include "MpiContext.h"
#include "PipelineSnapshotManager.h"
#include "ParityTestFramework.h"
#include <memory>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace llaminar;

/**
 * @brief Fixture for batch correctness tests
 */
class BatchCorrectnessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use qwen2.5-0.5b-instruct-q4_0.gguf for testing
        model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        auto rank = MPIContext::capture().rank;

        // Check if model exists
        if (rank == 0)
        {
            std::ifstream model_file(model_path);
            if (!model_file.good())
            {
                GTEST_SKIP() << "Model file not found: " << model_path;
            }
        }

        // Register both pipelines
        registerQwenPipeline();      // For sequential execution
        registerBatchQwenPipeline(); // For batch execution

        // Disable COSMA for simpler testing
        setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);

        // Load model configuration
        ModelLoader loader;
        if (!loader.loadModel(model_path))
        {
            GTEST_SKIP() << "Failed to load model: " << model_path;
        }

        TransformerLayerConfig base_config = loader.createLayerConfig();
        batch_config = ModelConfig(base_config, "qwen_batch"); // For batch pipeline
        sequential_config = ModelConfig(base_config, "qwen");  // For sequential pipeline
    }

    std::string model_path;
    ModelConfig batch_config;
    ModelConfig sequential_config;
};

/**
 * @brief Helper to compare two tensors element-by-element
 */
void compareTensors(
    const std::shared_ptr<TensorBase> &a,
    const std::shared_ptr<TensorBase> &b,
    const std::string &name,
    float tolerance = 1e-4)
{
    ASSERT_NE(a, nullptr) << name << " tensor A is null";
    ASSERT_NE(b, nullptr) << name << " tensor B is null";

    const auto &shape_a = a->shape();
    const auto &shape_b = b->shape();

    ASSERT_EQ(shape_a.size(), shape_b.size())
        << name << " shape rank mismatch";

    for (size_t i = 0; i < shape_a.size(); ++i)
    {
        ASSERT_EQ(shape_a[i], shape_b[i])
            << name << " shape[" << i << "] mismatch";
    }

    size_t numel = 1;
    for (auto dim : shape_a)
        numel *= dim;

    const float *data_a = a->data();
    const float *data_b = b->data();

    size_t mismatches = 0;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;

    for (size_t i = 0; i < numel; ++i)
    {
        float diff = std::abs(data_a[i] - data_b[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
            max_diff_idx = i;
        }

        if (diff > tolerance)
        {
            mismatches++;
            if (mismatches <= 5) // Only print first 5 mismatches
            {
                ADD_FAILURE() << name << " mismatch at index " << i
                              << ": " << data_a[i] << " vs " << data_b[i]
                              << " (diff: " << diff << ")";
            }
        }
    }

    EXPECT_EQ(mismatches, 0)
        << name << " had " << mismatches << " mismatches out of " << numel
        << " elements (max diff: " << max_diff << " at index " << max_diff_idx << ")";
}

/**
 * @brief Test: Batch prefill produces same results as sequential prefill
 */
TEST_F(BatchCorrectnessTest, PrefillBatchVsSequential)
{
    auto rank = MPIContext::capture().rank;

    const int batch_size = 2;
    const std::vector<int> seq1 = {1, 2, 3, 4};
    const std::vector<int> seq2 = {5, 6, 7, 8, 9};

    if (rank == 0)
    {
        std::cout << "\n=== Testing Prefill: Batch vs Sequential ===\n";
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "Sequence 1: " << seq1.size() << " tokens\n";
        std::cout << "Sequence 2: " << seq2.size() << " tokens\n";
    }

    // ============================================
    // Run batch execution
    // ============================================
    auto batch_pipeline = PipelineFactory::instance().create(batch_config);
    ASSERT_NE(batch_pipeline, nullptr);

    auto batch_weights = batch_pipeline->loadWeights(model_path);
    ASSERT_NE(batch_weights, nullptr);

    std::vector<std::vector<int>> batch_input = {seq1, seq2};
    StageContext batch_ctx;
    std::shared_ptr<TensorBase> batch_logits;

    if (rank == 0)
        std::cout << "Running batch prefill...\n";
    ASSERT_TRUE(batch_pipeline->prefillBatch(batch_input, *batch_weights, batch_ctx, batch_logits));
    ASSERT_NE(batch_logits, nullptr);

    // Extract logits for each sequence
    const auto &batch_shape = batch_logits->shape();
    ASSERT_EQ(batch_shape.size(), 2);
    ASSERT_EQ(batch_shape[0], batch_size);

    int vocab_size = batch_shape[1];
    const float *batch_data = batch_logits->data();

    std::vector<std::vector<float>> batch_results(batch_size);
    for (int seq = 0; seq < batch_size; ++seq)
    {
        batch_results[seq].assign(
            batch_data + seq * vocab_size,
            batch_data + (seq + 1) * vocab_size);
    }

    // ============================================
    // Run sequential execution (reuse weights to avoid duplicate heavy loads)
    // ============================================
    std::vector<std::vector<float>> sequential_results(batch_size);
    std::vector<std::vector<int>> sequences = {seq1, seq2};

    // Load weights once using a temporary pipeline instance
    auto seq_weights_loader = PipelineFactory::instance().create(sequential_config);
    ASSERT_NE(seq_weights_loader, nullptr);
    auto shared_seq_weights = seq_weights_loader->loadWeights(model_path);
    ASSERT_NE(shared_seq_weights, nullptr);
    // Optionally release loader pipeline early to free its buffers
    seq_weights_loader.reset();

    for (int seq = 0; seq < batch_size; ++seq)
    {
        if (rank == 0)
            std::cout << "Running sequential prefill for sequence " << seq << " (reusing weights)...\n";

        auto seq_pipeline = PipelineFactory::instance().create(sequential_config);
        ASSERT_NE(seq_pipeline, nullptr);

        StageContext seq_ctx;
        ASSERT_TRUE(seq_pipeline->prefill(sequences[seq], *shared_seq_weights, seq_ctx));

        std::shared_ptr<TensorBase> seq_logits;
        ASSERT_TRUE(seq_pipeline->logits(seq_logits));
        ASSERT_NE(seq_logits, nullptr);

        const auto &seq_shape = seq_logits->shape();
        ASSERT_EQ(seq_shape.size(), 2);
        ASSERT_EQ(seq_shape[1], vocab_size);

        int rows = seq_shape[0];
        const float *seq_data = seq_logits->data();
        sequential_results[seq].assign(
            seq_data + (rows - 1) * vocab_size,
            seq_data + rows * vocab_size);
    }

    // ============================================
    // Compare results
    // ============================================
    if (rank == 0)
        std::cout << "Comparing results...\n";

    for (int seq = 0; seq < batch_size; ++seq)
    {
        if (rank == 0)
            std::cout << "Comparing sequence " << seq << "...\n";

        ASSERT_EQ(batch_results[seq].size(), sequential_results[seq].size());

        size_t mismatches = 0;
        float max_diff = 0.0f;

        for (size_t i = 0; i < batch_results[seq].size(); ++i)
        {
            float diff = std::abs(batch_results[seq][i] - sequential_results[seq][i]);
            max_diff = std::max(max_diff, diff);

            if (diff > 1e-4f)
            {
                mismatches++;
                if (mismatches <= 3)
                {
                    ADD_FAILURE() << "Sequence " << seq << " token " << i
                                  << " mismatch: batch=" << batch_results[seq][i]
                                  << " sequential=" << sequential_results[seq][i]
                                  << " diff=" << diff;
                }
            }
        }

        EXPECT_EQ(mismatches, 0)
            << "Sequence " << seq << " had " << mismatches
            << " mismatches (max diff: " << max_diff << ")";
    }

    if (rank == 0)
    {
        std::cout << "✓ Batch prefill matches sequential prefill\n";
    }
}

/**
 * @brief Test: Batch decode produces same results as sequential decode
 */
// Helper: Extract single sequence from batch snapshot
static parity::TensorSnapshot extractSequenceFromBatch(
    const parity::TensorSnapshot &batch_snap,
    int seq_idx,
    int padded_seq_len,
    int batch_size)
{
    // Batch snapshot has shape [batch_size * padded_seq_len, feature_dim]
    // Extract rows [seq_idx * padded_seq_len, (seq_idx+1) * padded_seq_len)
    int feature_dim = batch_snap.metadata.feature_dim;
    int total_rows = batch_snap.metadata.seq_len;

    if (total_rows != batch_size * padded_seq_len)
    {
        throw std::runtime_error("Batch snapshot shape mismatch: expected " +
                                 std::to_string(batch_size * padded_seq_len) + " rows, got " +
                                 std::to_string(total_rows));
    }

    parity::TensorSnapshot seq_snap;
    seq_snap.metadata = batch_snap.metadata;
    seq_snap.metadata.seq_len = padded_seq_len;
    seq_snap.metadata.source = batch_snap.metadata.source + "_seq" + std::to_string(seq_idx);

    size_t seq_size = static_cast<size_t>(padded_seq_len) * feature_dim;
    seq_snap.data.resize(seq_size);

    const float *src = batch_snap.data.data() + seq_idx * padded_seq_len * feature_dim;
    std::copy(src, src + seq_size, seq_snap.data.begin());

    return seq_snap;
}

TEST_F(BatchCorrectnessTest, DecodeBatchVsSequential)
{
    auto rank = MPIContext::capture().rank;

    const int batch_size = 2;
    const std::vector<int> seq1 = {1, 2, 3};
    const std::vector<int> seq2 = {4, 5, 6, 7};
    const int decode_steps = 3;

    if (rank == 0)
    {
        std::cout << "\n=== Testing Decode: Batch vs Sequential ===\n";
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "Decode steps: " << decode_steps << "\n";
    }

    // Force comparable semantics on ALL ranks: disable sequential incremental decode and enable batch replay
    setenv("LLAMINAR_DISABLE_INCREMENTAL_DECODE", "1", 1);
    setenv("LLAMINAR_BATCH_DECODE_REPLAY", "1", 1);
    setenv("LLAMINAR_PARITY_CAPTURE", "1", 1);                 // ensure snapshots if needed
    setenv("LLAMINAR_DECODE_STAGE_SNAPSHOTS", "1", 1);         // capture per-layer decode stage snapshots
    setenv("LLAMINAR_DECODE_STAGE_SNAPSHOTS_VERBOSE", "1", 1); // emit verbose decode snapshot logging
    if (rank == 0)
        std::cout << "[TEST_ENV] (all ranks) Set LLAMINAR_DISABLE_INCREMENTAL_DECODE=1 LLAMINAR_BATCH_DECODE_REPLAY=1" << std::endl;

    // Force a refresh of the debug environment snapshot since tests set env vars dynamically
    refreshDebugEnv();
    (void)debugEnv(); // rebuild under new environment

    // Explicitly (re)enable snapshot systems in case singleton instantiated before env vars
    PipelineSnapshotManager::instance().setEnabled(true);
    parity::LlaminarSnapshotHook::set_enabled(true);
    PipelineSnapshotManager::instance().clear();
    parity::SnapshotRegistry::instance().clear();
    if (rank == 0)
    {
        std::cout << "[SNAPSHOT_INIT] Enabled snapshot capture for decode parity test.\n";
        std::cout << "  PipelineSnapshotManager::isEnabled()=" << PipelineSnapshotManager::instance().isEnabled() << "\n";
        std::cout << "  LlaminarSnapshotHook::is_enabled()=" << parity::LlaminarSnapshotHook::is_enabled() << "\n";
    }

    // ============================================
    // Run batch execution
    // ============================================
    auto batch_pipeline = PipelineFactory::instance().create(batch_config);
    ASSERT_NE(batch_pipeline, nullptr);

    auto batch_weights = batch_pipeline->loadWeights(model_path);
    ASSERT_NE(batch_weights, nullptr);

    std::vector<std::vector<int>> batch_input = {seq1, seq2};
    StageContext batch_ctx;
    std::shared_ptr<TensorBase> batch_logits;

    // Prefill
    if (rank == 0)
        std::cout << "Batch prefill...\n";
    ASSERT_TRUE(batch_pipeline->prefillBatch(batch_input, *batch_weights, batch_ctx, batch_logits));

    // Decode steps
    std::vector<std::vector<std::vector<float>>> batch_decode_results(decode_steps);

    for (int step = 0; step < decode_steps; ++step)
    {
        if (rank == 0)
            std::cout << "Batch decode step " << (step + 1) << "...\n";

        // Use token ID 42 for all sequences (deterministic)
        std::vector<int> next_tokens(batch_size, 42);

        std::shared_ptr<TensorBase> decode_logits;
        ASSERT_TRUE(batch_pipeline->decodeBatch(next_tokens, *batch_weights, batch_ctx, decode_logits));
        ASSERT_NE(decode_logits, nullptr);

        const auto &shape = decode_logits->shape();
        ASSERT_EQ(shape.size(), 2);
        ASSERT_EQ(shape[0], batch_size);

        int vocab_size = shape[1];
        const float *data = decode_logits->data();

        batch_decode_results[step].resize(batch_size);
        for (int seq = 0; seq < batch_size; ++seq)
        {
            batch_decode_results[step][seq].assign(
                data + seq * vocab_size,
                data + (seq + 1) * vocab_size);
        }
    }

    // ============================================
    // Run sequential execution
    // ============================================
    std::vector<std::vector<std::vector<float>>> sequential_decode_results(decode_steps);
    for (auto &step_results : sequential_decode_results)
    {
        step_results.resize(batch_size);
    }

    std::vector<std::vector<int>> sequences = {seq1, seq2};

    for (int seq = 0; seq < batch_size; ++seq)
    {
        if (rank == 0)
            std::cout << "Sequential prefill+decode for sequence " << seq << "...\n";

        auto seq_pipeline = PipelineFactory::instance().create(sequential_config);
        ASSERT_NE(seq_pipeline, nullptr);

        auto seq_weights = seq_pipeline->loadWeights(model_path);
        ASSERT_NE(seq_weights, nullptr);

        StageContext seq_ctx;

        // Prefill
        ASSERT_TRUE(seq_pipeline->prefill(sequences[seq], *seq_weights, seq_ctx));

        // Decode steps
        for (int step = 0; step < decode_steps; ++step)
        {
            ASSERT_TRUE(seq_pipeline->decode(42, *seq_weights, seq_ctx));

            std::shared_ptr<TensorBase> seq_logits;
            ASSERT_TRUE(seq_pipeline->logits(seq_logits));
            ASSERT_NE(seq_logits, nullptr);

            const auto &shape = seq_logits->shape();
            ASSERT_EQ(shape.size(), 2);

            int vocab_size = shape[1];
            int rows = shape[0];
            const float *data = seq_logits->data();

            sequential_decode_results[step][seq].assign(
                data + (rows - 1) * vocab_size,
                data + rows * vocab_size);
        }
    }

    // ============================================
    // Compare results
    // ============================================
    if (rank == 0)
        std::cout << "Comparing decode results...\n";

    // --- Diagnostic: Stage-level snapshot parity for first decode step (step 0) ---
    // We compare batch replay vs sequential replay snapshots captured with decode step suffix _dec0
    if (rank == 0)
    {
        using namespace parity;
        auto &registry = SnapshotRegistry::instance();
        // Enumerate available keys with decode step 0 suffix for debugging
        {
            std::cout << "\n[DECODE_STAGE_PARITY] Registry keys containing 'dec0' before stage comparisons:" << std::endl;
            // There is no direct API to iterate keys; hack: attempt common stage keys
            PipelineStage probe_stages[] = {PipelineStage::EMBEDDING, PipelineStage::ATTENTION_NORM, PipelineStage::Q_PROJECTION,
                                            PipelineStage::K_PROJECTION, PipelineStage::V_PROJECTION, PipelineStage::ROPE_APPLICATION,
                                            PipelineStage::ATTENTION_CONTEXT, PipelineStage::ATTENTION_OUTPUT, PipelineStage::FINAL_NORM, PipelineStage::LM_HEAD};
            for (auto ps : probe_stages)
            {
                std::string bk = registry.make_key("batch_dec0", ps, ps == PipelineStage::EMBEDDING || ps == PipelineStage::FINAL_NORM || ps == PipelineStage::LM_HEAD ? -1 : 0);
                std::string sk = registry.make_key("llaminar_dec0", ps, ps == PipelineStage::EMBEDDING || ps == PipelineStage::FINAL_NORM || ps == PipelineStage::LM_HEAD ? -1 : 0);
                parity::TensorSnapshot tmp;
                bool have_b = registry.get_snapshot(bk, tmp);
                bool have_s = registry.get_snapshot(sk, tmp);
                std::cout << "  key_batch=" << bk << " present=" << have_b << " key_seq=" << sk << " present=" << have_s << std::endl;
            }
        }
        // Batch processes 2 sequences padded to 5 tokens each (seq1=3→5, seq2=4→5)
        // Sequential processes them one at a time: seq1 prefill (3) + decode (1) = 4 tokens
        //                                          seq2 prefill (4) + decode (1) = 5 tokens
        const int max_seq_len = 5;                         // Padded length in batch
        const std::vector<int> original_seq_lens = {3, 4}; // seq1, seq2 prefill lengths
        const std::vector<int> decode0_lens = {4, 5};      // After 1 decode token added

        // We'll compare batch sequence 1 (seq2) against the last sequential pipeline created (also seq2)
        // because the test creates sequential pipelines in order: seq=0 (seq1), seq=1 (seq2)
        const int compare_seq_idx = 1; // seq2

        auto compare_stage = [&](PipelineStage stage, int layer, const std::string &label)
        {
            std::string batch_key = registry.make_key("batch_dec0", stage, layer);
            std::string seq_key = registry.make_key("llaminar_dec0", stage, layer);
            TensorSnapshot batch_full, seq_snap;
            bool have_batch = registry.get_snapshot(batch_key, batch_full);
            bool have_seq = registry.get_snapshot(seq_key, seq_snap);
            if (!have_batch || !have_seq)
            {
                std::cout << "[DECODE_STAGE_PARITY] Missing snapshot(s) for " << label
                          << " batch_key=" << batch_key << " present=" << have_batch
                          << " seq_key=" << seq_key << " present=" << have_seq << "\n";
                return;
            }

            // DEBUG: Print raw batch snapshot before extraction
            if (label == "ROPE_APPLICATION_L0" && rank == 0)
            {
                std::cout << "[ROPE_DEBUG_TEST] Raw batch_full snapshot:\n";
                std::cout << "  seq_len=" << batch_full.metadata.seq_len
                          << " feature_dim=" << batch_full.metadata.feature_dim << "\n";
                std::cout << "  First 8 values: [";
                for (int i = 0; i < std::min<size_t>(8, batch_full.data.size()); ++i)
                {
                    std::cout << batch_full.data[i] << (i + 1 < std::min<size_t>(8, batch_full.data.size()) ? "," : "");
                }
                std::cout << "]\n";
                std::cout << "  Offset 5120 (seq2 start if 2 seqs × 5 tokens × 1024 dims / 2): [";
                for (int i = 0; i < 8 && (5120 + i) < batch_full.data.size(); ++i)
                {
                    std::cout << batch_full.data[5120 + i] << (i + 1 < 8 ? "," : "");
                }
                std::cout << "]\n";
            }

            // Extract the compare_seq_idx sequence from batch snapshot
            TensorSnapshot batch_seq;
            try
            {
                batch_seq = extractSequenceFromBatch(batch_full, compare_seq_idx, max_seq_len, batch_size);
            }
            catch (const std::exception &e)
            {
                std::cout << "[DECODE_STAGE_PARITY] Failed to extract sequence " << compare_seq_idx
                          << " from batch snapshot: " << e.what() << "\n";
                return;
            }

            // Now compare: batch_seq should have shape [max_seq_len, feature_dim]
            //              seq_snap should have shape [decode0_lens[compare_seq_idx], feature_dim]
            // We'll compare only the valid (non-padded) portion
            int valid_len = decode0_lens[compare_seq_idx]; // 5 for seq2
            if (batch_seq.metadata.seq_len < valid_len)
            {
                std::cout << "[DECODE_STAGE_PARITY] Batch sequence too short for " << label
                          << " batch_len=" << batch_seq.metadata.seq_len << " valid_len=" << valid_len << "\n";
                return;
            }
            if (seq_snap.metadata.seq_len != valid_len)
            {
                std::cout << "[DECODE_STAGE_PARITY] Sequential length mismatch for " << label
                          << " expected=" << valid_len << " got=" << seq_snap.metadata.seq_len << "\n";
                return;
            }
            if (batch_seq.metadata.feature_dim != seq_snap.metadata.feature_dim)
            {
                std::cout << "[DECODE_STAGE_PARITY] Feature dim mismatch for " << label
                          << " batch=" << batch_seq.metadata.feature_dim
                          << " seq=" << seq_snap.metadata.feature_dim << "\n";
                return;
            }
            // Compare only valid portion (non-padded rows)
            int feature_dim = seq_snap.metadata.feature_dim;
            size_t total = static_cast<size_t>(valid_len) * feature_dim;
            size_t mismatches = 0;
            float max_diff = 0.0f;
            double sum_sq_batch = 0.0;
            double sum_sq_seq = 0.0;

            for (int row = 0; row < valid_len; ++row)
            {
                for (int col = 0; col < feature_dim; ++col)
                {
                    size_t idx = row * feature_dim + col;
                    float batch_val = batch_seq.data[idx];
                    float seq_val = seq_snap.data[idx];
                    float diff = std::fabs(batch_val - seq_val);
                    if (diff > 1e-4f)
                    {
                        ++mismatches;
                        if (diff > max_diff)
                            max_diff = diff;
                    }
                    sum_sq_batch += batch_val * batch_val;
                    sum_sq_seq += seq_val * seq_val;
                }
            }

            double l2_batch = std::sqrt(sum_sq_batch / total);
            double l2_seq = std::sqrt(sum_sq_seq / total);
            std::cout << "[DECODE_STAGE_PARITY] " << label << " seq" << compare_seq_idx << " step0: "
                      << "mismatches=" << mismatches << "/" << total
                      << " max_diff=" << max_diff
                      << " L2_batch=" << l2_batch << " L2_seq=" << l2_seq;

            // Preview first 8 values
            std::cout << " first8_batch=[";
            for (int i = 0; i < std::min<size_t>(8, total); ++i)
            {
                std::cout << batch_seq.data[i] << (i + 1 < std::min<size_t>(8, total) ? "," : "");
            }
            std::cout << "] first8_seq=[";
            for (int i = 0; i < std::min<size_t>(8, total); ++i)
            {
                std::cout << seq_snap.data[i] << (i + 1 < std::min<size_t>(8, total) ? "," : "");
            }
            std::cout << "]\n";
        };
        std::cout << "\n=== Decode Step 0 Stage Parity Diagnostics ===\n";
        compare_stage(PipelineStage::EMBEDDING, -1, "EMBEDDING");
        compare_stage(PipelineStage::ATTENTION_NORM, 0, "ATTENTION_NORM_L0");
        compare_stage(PipelineStage::Q_PROJECTION, 0, "Q_PROJECTION_L0");
        compare_stage(PipelineStage::K_PROJECTION, 0, "K_PROJECTION_L0");
        compare_stage(PipelineStage::V_PROJECTION, 0, "V_PROJECTION_L0");
        compare_stage(PipelineStage::ROPE_APPLICATION, 0, "ROPE_APPLICATION_L0");
        compare_stage(PipelineStage::ATTENTION_CONTEXT, 0, "ATTENTION_CONTEXT_L0");
    }

    for (int step = 0; step < decode_steps; ++step)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            if (rank == 0)
                std::cout << "Comparing step " << (step + 1)
                          << " sequence " << seq << "...\n";

            const auto &batch_result = batch_decode_results[step][seq];
            const auto &seq_result = sequential_decode_results[step][seq];

            ASSERT_EQ(batch_result.size(), seq_result.size());

            size_t mismatches = 0;
            float max_diff = 0.0f;

            for (size_t i = 0; i < batch_result.size(); ++i)
            {
                float diff = std::abs(batch_result[i] - seq_result[i]);
                max_diff = std::max(max_diff, diff);

                if (diff > 1e-4f)
                {
                    mismatches++;
                    if (mismatches <= 3)
                    {
                        ADD_FAILURE() << "Step " << step << " seq " << seq
                                      << " token " << i << " mismatch: batch="
                                      << batch_result[i] << " sequential="
                                      << seq_result[i] << " diff=" << diff;
                    }
                }
            }

            EXPECT_EQ(mismatches, 0)
                << "Step " << step << " sequence " << seq << " had "
                << mismatches << " mismatches (max diff: " << max_diff << ")";
        }
    }

    if (rank == 0)
    {
        std::cout << "✓ Batch decode matches sequential decode\n";
    }
}

/**
 * @brief Test: Verify stage-by-stage parity between batch and sequential attention operators
 */
TEST_F(BatchCorrectnessTest, BatchedAttentionStagesParity)
{
    auto rank = MPIContext::capture().rank;

    const std::vector<int> tokens = {1, 2, 3, 4};

    if (rank == 0)
    {
        std::cout << "\n=== Stage-by-Stage Divergence Analysis ===\n";
        std::cout << "Tokens: " << tokens.size() << "\n";
    }

    // Clear previous snapshots from BOTH registries
    PipelineSnapshotManager::instance().clear();
    parity::SnapshotRegistry::instance().clear(); // Clear parity framework registry too!

    // Enable snapshot capture for BOTH mechanisms:
    // 1. PipelineSnapshotManager (used by AbstractPipeline::isParityEnabled())
    // 2. LlaminarSnapshotHook (used by parity framework)
    PipelineSnapshotManager::instance().setEnabled(true);
    parity::LlaminarSnapshotHook::set_enabled(true);

    if (rank == 0)
    {
        std::cout << "[DEBUG] About to check snapshot mechanisms...\n"
                  << std::flush;
        bool psm_enabled = PipelineSnapshotManager::instance().isEnabled();
        std::cout << "[DEBUG] PipelineSnapshotManager::isEnabled() = " << psm_enabled << "\n"
                  << std::flush;
        bool hook_enabled = parity::LlaminarSnapshotHook::is_enabled();
        std::cout << "[DEBUG] LlaminarSnapshotHook::is_enabled() = " << hook_enabled << "\n"
                  << std::flush;

        std::cout << "Snapshot mechanisms enabled:\n";
        std::cout << "  PipelineSnapshotManager: " << (psm_enabled ? "YES" : "NO") << "\n";
        std::cout << "  LlaminarSnapshotHook: " << (hook_enabled ? "YES" : "NO") << "\n";
    }

    // ============================================
    // Run sequential pipeline with snapshots
    // ============================================
    if (rank == 0)
        std::cout << "Running sequential pipeline...\n";

    auto seq_pipeline = PipelineFactory::instance().create(sequential_config);
    ASSERT_NE(seq_pipeline, nullptr);

    auto seq_weights = seq_pipeline->loadWeights(model_path);
    ASSERT_NE(seq_weights, nullptr);

    StageContext seq_ctx;
    ASSERT_TRUE(seq_pipeline->prefill(tokens, *seq_weights, seq_ctx));

    std::shared_ptr<TensorBase> seq_logits;
    ASSERT_TRUE(seq_pipeline->logits(seq_logits));
    ASSERT_NE(seq_logits, nullptr);

    // ============================================
    // Run batch pipeline with snapshots
    // ============================================
    if (rank == 0)
        std::cout << "Running batch pipeline...\n";

    auto batch_pipeline = PipelineFactory::instance().create(batch_config);
    ASSERT_NE(batch_pipeline, nullptr);

    auto batch_weights = batch_pipeline->loadWeights(model_path);
    ASSERT_NE(batch_weights, nullptr);

    std::vector<std::vector<int>> batch_input = {tokens};
    StageContext batch_ctx;
    std::shared_ptr<TensorBase> batch_logits;

    ASSERT_TRUE(batch_pipeline->prefillBatch(batch_input, *batch_weights, batch_ctx, batch_logits));
    ASSERT_NE(batch_logits, nullptr);

    // ============================================
    // Compare snapshots stage-by-stage
    // ============================================

    // Declare variables outside rank check so they're available for assertions
    int passed = 0, failed = 0, missing = 0;
    int seq_snapshot_count = 0, batch_snapshot_count = 0;

    if (rank == 0)
    {
        std::cout << "\n=== Comparing Snapshots Stage-by-Stage ===\n";

        using namespace parity;
        SnapshotRegistry &registry = SnapshotRegistry::instance();

        // List captured snapshots
        auto all_keys = registry.list_keys();
        std::cout << "Total snapshots captured: " << all_keys.size() << "\n";

        std::vector<std::string> seq_keys, batch_keys;
        for (const auto &key : all_keys)
        {
            // Accept both legacy provider source ("OpenBLAS_") and unified pipeline source ("llaminar_")
            if (key.find("OpenBLAS_") == 0 || key.find("llaminar_") == 0)
                seq_keys.push_back(key);
            else if (key.find("batch_") == 0)
                batch_keys.push_back(key);
        }

        std::cout << "Sequential snapshots: " << seq_keys.size() << "\n";
        std::cout << "Batch snapshots: " << batch_keys.size() << "\n";

        // Debug: Print EMBEDDING and layer 0 keys
        std::cout << "\nEMBEDDING keys:\n";
        for (const auto &key : all_keys)
        {
            if (key.find("EMBEDDING") != std::string::npos)
            {
                std::cout << "  " << key << "\n";
            }
        }
        std::cout << "\nLayer 0 keys:\n";
        for (const auto &key : all_keys)
        {
            if (key.find("layer_0") != std::string::npos)
            {
                std::cout << "  " << key << "\n";
            }
        }
        std::cout << "\n";

        // Expected snapshot count calculation (functional baseline, excluding optional internal attention diagnostics):
        // Per pipeline: EMBEDDING (1) + 24 layers × 14 functional stages + FINAL_NORM + LM_HEAD
        // Functional per-layer stages: ATTENTION_NORM, Q_PROJECTION, K_PROJECTION, V_PROJECTION,
        // ROPE_APPLICATION, ATTENTION_CONTEXT, ATTENTION_OUTPUT, ATTENTION_RESIDUAL,
        // FFN_NORM, FFN_GATE, FFN_UP, FFN_SWIGLU, FFN_DOWN, FFN_RESIDUAL
        // = 1 + 24×14 + 2 = 339 snapshots per pipeline
        const int expected_per_pipeline = 339;
        // NOTE: Internal stages ATTENTION_SCORES and ATTENTION_SOFTMAX are gated behind
        // LLAMINAR_ATTN_INTERNAL_DIFF (debugEnv().attention.internal_diff). When enabled,
        // sequential snapshot count will increase by 24×2 = 48 (and batch if attention operator also captures).
        seq_snapshot_count = seq_keys.size();
        batch_snapshot_count = batch_keys.size();

        // Define stages to compare (layer 0 for comprehensive validation)
        struct StageInfo
        {
            std::string name;
            int layer;
        };

        std::vector<StageInfo> stages = {
            // Input embedding
            {"EMBEDDING", -1},

            // Attention block (layer 0)
            {"ATTENTION_NORM", 0},
            {"Q_PROJECTION", 0},
            {"K_PROJECTION", 0},
            {"V_PROJECTION", 0},
            {"ROPE_APPLICATION", 0},
            {"ATTENTION_CONTEXT", 0},
            {"ATTENTION_OUTPUT", 0},
            {"ATTENTION_RESIDUAL", 0},

            // FFN block (layer 0)
            {"FFN_NORM", 0},
            {"FFN_GATE", 0},
            {"FFN_UP", 0},
            {"FFN_SWIGLU", 0},
            {"FFN_DOWN", 0},
            {"FFN_RESIDUAL", 0},

            // Output processing (after all layers)
            {"FINAL_NORM", -1},
            {"LM_HEAD", -1}};

        ComparisonTolerance tolerance(1e-4f, 1e-4);

        // Use the outer scope variables
        bool found_divergence = false;

        for (const auto &stage : stages)
        {
            // Use slightly relaxed tolerance for final stages that accumulate errors
            ComparisonTolerance stage_tolerance = tolerance;
            if (stage.name == "FINAL_NORM" || stage.name == "LM_HEAD")
            {
                // Final stages accumulate numerical errors from all previous operations
                // After switching from direct cblas to adaptiveMatMul wrapper, we see tiny
                // rounding differences accumulate (max_abs ~0.00033 on values ~63)
                stage_tolerance = ComparisonTolerance(5e-4f, 1e-3);
            }

            // Build keys using registry's make_key() for consistent formatting
            // Sequential uses "llaminar" (default snapshot_source_ from PipelineBase)
            std::string seq_key = registry.make_key("llaminar", stage.name, stage.layer);
            std::string batch_key = registry.make_key("batch", stage.name, stage.layer);

            // Get snapshots
            TensorSnapshot seq_snap, batch_snap;

            if (!registry.get_snapshot(seq_key, seq_snap))
            {
                std::cout << "⚠ MISSING: " << seq_key << "\n";
                missing++;
                continue;
            }

            if (!registry.get_snapshot(batch_key, batch_snap))
            {
                std::cout << "⚠ MISSING: " << batch_key << "\n";
                missing++;
                continue;
            }

            // Special handling for LM_HEAD: sequential captures all tokens, batch captures only last
            TensorSnapshot seq_snap_for_compare = seq_snap;
            if (stage.name == "LM_HEAD" && seq_snap.data.size() != batch_snap.data.size())
            {
                // Sequential has [seq_len, vocab], batch has [batch_size, vocab]
                // Extract last token from sequential to match batch
                size_t vocab_size = batch_snap.data.size(); // batch_size=1, so this is vocab_size
                size_t seq_len = seq_snap.data.size() / vocab_size;

                if (seq_snap.data.size() % vocab_size == 0 && seq_len > 0)
                {
                    // Extract last token's logits
                    seq_snap_for_compare.data.assign(
                        seq_snap.data.begin() + (seq_len - 1) * vocab_size,
                        seq_snap.data.end());

                    std::cout << "ℹ LM_HEAD: Extracted last token from sequential ["
                              << seq_len << ", " << vocab_size << "] -> [1, " << vocab_size << "]\n";
                }
            }

            // Compare
            auto result = SnapshotComparator::compare(seq_snap_for_compare, batch_snap, stage_tolerance);

            if (result.passed())
            {
                std::cout << "✓ " << stage.name;
                if (stage.layer >= 0)
                    std::cout << " layer " << stage.layer;
                std::cout << " (max_diff=" << result.metrics.max_abs_diff << ")\n";
                passed++;
            }
            else
            {
                std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
                std::cout << "║ ✗ FIRST DIVERGENCE DETECTED                                ║\n";
                std::cout << "╠════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ Stage: " << stage.name;
                if (stage.layer >= 0)
                    std::cout << " layer " << stage.layer;
                std::cout << "\n";

                // Debug: Show error message if present
                if (!result.error_message.empty())
                {
                    std::cout << "║ ERROR: " << result.error_message << "\n";
                }

                std::cout << "║ Snapshot sizes: seq=" << seq_snap.data.size()
                          << " batch=" << batch_snap.data.size() << "\n";
                std::cout << "║ Max absolute diff: " << result.metrics.max_abs_diff << "\n";
                std::cout << "║ Relative L2 error: " << result.metrics.rel_l2 << "\n";
                std::cout << "║ Worst index: " << result.metrics.worst_index << "\n";
                std::cout << "║   Sequential: " << result.metrics.worst_expected << "\n";
                std::cout << "║   Batch:      " << result.metrics.worst_actual << "\n";

                // Print stats
                auto compute_stats = [](const std::vector<float> &data)
                {
                    if (data.empty())
                        return std::make_tuple(0.0f, 0.0f, 0.0f);
                    float min_val = *std::min_element(data.begin(), data.end());
                    float max_val = *std::max_element(data.begin(), data.end());
                    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
                    return std::make_tuple(min_val, max_val, sum / data.size());
                };

                auto [seq_min, seq_max, seq_mean] = compute_stats(seq_snap.data);
                auto [batch_min, batch_max, batch_mean] = compute_stats(batch_snap.data);

                std::cout << "║ Sequential: min=" << seq_min << " max=" << seq_max
                          << " mean=" << seq_mean << "\n";
                std::cout << "║ Batch:      min=" << batch_min << " max=" << batch_max
                          << " mean=" << batch_mean << "\n";
                std::cout << "╚════════════════════════════════════════════════════════════╝\n";

                found_divergence = true;
                failed++;
                break; // Stop at first failure
            }
        }

        std::cout << "\n=== SUMMARY ===\n";
        std::cout << "Stages compared: " << stages.size() << "\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Missing: " << missing << "\n";
        std::cout << "\n=== SNAPSHOT CAPTURE STATS (All Layers) ===\n";
        std::cout << "Expected per pipeline: " << expected_per_pipeline << "\n";
        std::cout << "Sequential captured: " << seq_snapshot_count << " / " << expected_per_pipeline << "\n";
        std::cout << "Batch captured: " << batch_snapshot_count << " / " << expected_per_pipeline << "\n";

        if (!found_divergence && failed == 0 && missing == 0)
        {
            std::cout << "\n✓ ALL TESTED STAGES MATCH!\n";
        }
        else if (found_divergence)
        {
            std::cout << "\n✗ DIVERGENCE DETECTED - TEST WILL FAIL\n";
        }
    }

    // Synchronize counters across ranks (only rank 0 performs comparison)
    MPI_Bcast(&missing, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // CRITICAL FIX: Fail test if divergence detected OR snapshots missing
    ASSERT_EQ(missing, 0) << "Missing " << missing << " critical snapshot comparisons";
    ASSERT_EQ(failed, 0) << "Detected divergence in " << failed << " stage(s) - see output above for details";
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
