/**
 * @file test_batch_sequential_parity.cpp
 * @brief Pipeline-level parity test comparing batch vs sequential execution
 * @author David Sanftenberg
 *
 * This test captures snapshots at every pipeline stage and compares batch vs sequential
 * execution to identify where divergence occurs. Uses the existing PipelineSnapshotManager
 * infrastructure that was previously only used for PyTorch parity testing.
 */

#include <gtest/gtest.h>
#include "AbstractPipeline.h"
#include "ModelLoader.h"
#include "MpiContext.h"
#include "Logger.h"
#include "PipelineSnapshotManager.h"
#include "PipelineStages.h"
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace llaminar;

// Forward declare registration functions
extern void registerQwenPipeline();
extern void registerBatchQwenPipeline();

class BatchSequentialParityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        auto rank = MPIContext::capture().rank;

        // Check if model exists
        if (rank == 0)
        {
            std::ifstream model_file(model_path_);
            if (!model_file.good())
            {
                GTEST_SKIP() << "Model file not found: " << model_path_;
            }
        }

        // Register both pipelines
        registerQwenPipeline();
        registerBatchQwenPipeline();

        // Disable COSMA for deterministic comparison
        setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);

        // Enable snapshot capture
        setenv("LLAMINAR_PARITY_CAPTURE", "1", 1);

        // Load model configuration
        ModelLoader loader;
        if (!loader.loadModel(model_path_))
        {
            GTEST_SKIP() << "Failed to load model: " << model_path_;
        }

        TransformerLayerConfig base_config = loader.createLayerConfig();
        batch_config_ = ModelConfig(base_config, "qwen_batch");
        sequential_config_ = ModelConfig(base_config, "qwen");
    }

    void TearDown() override
    {
        // Clear snapshots after each test
        PipelineSnapshotManager::instance().clear();
    }

    std::string model_path_;
    ModelConfig batch_config_;
    ModelConfig sequential_config_;
};

/**
 * @brief Helper to compare two snapshots with detailed diagnostics
 */
void compareSnapshots(
    const std::string &stage_name,
    const PipelineSnapshotManager::SnapshotData &batch_snap,
    const PipelineSnapshotManager::SnapshotData &seq_snap,
    float tolerance = 1e-4)
{
    auto rank = MPIContext::capture().rank;

    // Shape comparison
    ASSERT_EQ(batch_snap.shape.size(), seq_snap.shape.size())
        << stage_name << ": Shape rank mismatch";

    // Note: Batch may have [B, T, D] while sequential has [T, D]
    // We'll compare per-sequence
    size_t batch_elements = 1;
    size_t seq_elements = 1;
    for (auto d : batch_snap.shape)
        batch_elements *= d;
    for (auto d : seq_snap.shape)
        seq_elements *= d;

    if (rank == 0)
    {
        std::cout << "\n=== " << stage_name << " ===" << std::endl;
        std::cout << "Batch shape: [";
        for (size_t i = 0; i < batch_snap.shape.size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << batch_snap.shape[i];
        }
        std::cout << "] (" << batch_elements << " elements)" << std::endl;

        std::cout << "Sequential shape: [";
        for (size_t i = 0; i < seq_snap.shape.size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << seq_snap.shape[i];
        }
        std::cout << "] (" << seq_elements << " elements)" << std::endl;

        // L2 norm comparison
        double batch_l2 = 0.0, seq_l2 = 0.0;
        for (size_t i = 0; i < batch_elements; ++i)
            batch_l2 += batch_snap.data[i] * batch_snap.data[i];
        for (size_t i = 0; i < seq_elements; ++i)
            seq_l2 += seq_snap.data[i] * seq_snap.data[i];

        batch_l2 = std::sqrt(batch_l2 / batch_elements);
        seq_l2 = std::sqrt(seq_l2 / seq_elements);

        std::cout << "Batch L2 norm: " << batch_l2 << std::endl;
        std::cout << "Sequential L2 norm: " << seq_l2 << std::endl;
        std::cout << "Ratio: " << (seq_l2 / batch_l2) << "x" << std::endl;

        // Sample values
        std::cout << "Batch first 5: [";
        for (int i = 0; i < std::min(5ul, batch_elements); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << batch_snap.data[i];
        }
        std::cout << "]" << std::endl;

        std::cout << "Sequential first 5: [";
        for (int i = 0; i < std::min(5ul, seq_elements); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << seq_snap.data[i];
        }
        std::cout << "]" << std::endl;
    }
}

/**
 * @brief Test: Compare batch vs sequential at all pipeline stages
 *
 * This test runs both pipelines with snapshot capture enabled and compares
 * the intermediate activations at every stage to pinpoint where divergence occurs.
 */
TEST_F(BatchSequentialParityTest, ComprehensivePipelineComparison)
{
    auto rank = MPIContext::capture().rank;

    // Use a simple sequence for reproducibility
    const std::vector<int> tokens = {1, 2, 3, 4};

    if (rank == 0)
    {
        std::cout << "\n=== Comprehensive Pipeline Parity Test ===" << std::endl;
        std::cout << "Sequence: " << tokens.size() << " tokens" << std::endl;
    }

    // Clear any previous snapshots
    PipelineSnapshotManager::instance().clear();

    // ============================================
    // Run Sequential Pipeline
    // ============================================
    if (rank == 0)
        std::cout << "\n--- Running Sequential Pipeline ---" << std::endl;

    auto seq_pipeline = PipelineFactory::instance().create(sequential_config_);
    ASSERT_NE(seq_pipeline, nullptr);

    auto seq_weights = seq_pipeline->loadWeights(model_path_);
    ASSERT_NE(seq_weights, nullptr);

    StageContext seq_ctx;
    ASSERT_TRUE(seq_pipeline->prefill(tokens, *seq_weights, seq_ctx));

    std::shared_ptr<TensorBase> seq_logits;
    ASSERT_TRUE(seq_pipeline->logits(seq_logits));
    ASSERT_NE(seq_logits, nullptr);

    // Save sequential snapshots
    auto &snapshot_mgr = PipelineSnapshotManager::instance();
    auto seq_snapshots = snapshot_mgr.getAllSnapshots();

    if (rank == 0)
    {
        std::cout << "Sequential captured " << seq_snapshots.size() << " snapshots" << std::endl;
    }

    // Clear for batch run
    snapshot_mgr.clear();

    // ============================================
    // Run Batch Pipeline
    // ============================================
    if (rank == 0)
        std::cout << "\n--- Running Batch Pipeline ---" << std::endl;

    auto batch_pipeline = PipelineFactory::instance().create(batch_config_);
    ASSERT_NE(batch_pipeline, nullptr);

    auto batch_weights = batch_pipeline->loadWeights(model_path_);
    ASSERT_NE(batch_weights, nullptr);

    std::vector<std::vector<int>> batch_input = {tokens}; // Single sequence batch
    StageContext batch_ctx;
    std::shared_ptr<TensorBase> batch_logits;

    ASSERT_TRUE(batch_pipeline->prefillBatch(batch_input, *batch_weights, batch_ctx, batch_logits));
    ASSERT_NE(batch_logits, nullptr);

    // Get batch snapshots
    auto batch_snapshots = snapshot_mgr.getAllSnapshots();

    if (rank == 0)
    {
        std::cout << "Batch captured " << batch_snapshots.size() << " snapshots" << std::endl;
    }

    // ============================================
    // Compare Snapshots Stage by Stage
    // ============================================
    if (rank == 0)
    {
        std::cout << "\n=== STAGE-BY-STAGE COMPARISON ===" << std::endl;
    }

    // Define stages to compare (in execution order)
    std::vector<std::pair<PipelineStage, std::string>> stages_to_compare = {
        {PipelineStage::EMBEDDING, "EMBEDDING"},
        {PipelineStage::ATTENTION_NORM, "ATTENTION_NORM_layer0"},
        {PipelineStage::Q_PROJECTION, "Q_PROJECTION_layer0"},
        {PipelineStage::K_PROJECTION, "K_PROJECTION_layer0"},
        {PipelineStage::V_PROJECTION, "V_PROJECTION_layer0"},
        {PipelineStage::ATTENTION_OUTPUT, "ATTENTION_OUTPUT_layer0"},
        {PipelineStage::ATTENTION_RESIDUAL, "ATTENTION_RESIDUAL_layer0"},
        {PipelineStage::FFN_NORM, "FFN_NORM_layer0"},
        {PipelineStage::FFN_GATE, "FFN_GATE_layer0"},
        {PipelineStage::FFN_UP, "FFN_UP_layer0"},
        {PipelineStage::FFN_SWIGLU, "FFN_SWIGLU_layer0"},
        {PipelineStage::FFN_DOWN, "FFN_DOWN_layer0"},
        {PipelineStage::FFN_RESIDUAL, "FFN_RESIDUAL_layer0"},
        {PipelineStage::FINAL_NORM, "FINAL_NORM"},
        {PipelineStage::LM_HEAD, "LM_HEAD"}};

    for (const auto &[stage, stage_name] : stages_to_compare)
    {
        // Find snapshots in both collections
        auto seq_it = std::find_if(seq_snapshots.begin(), seq_snapshots.end(),
                                   [&stage_name](const auto &pair)
                                   {
                                       return pair.first.find(stage_name) != std::string::npos;
                                   });

        auto batch_it = std::find_if(batch_snapshots.begin(), batch_snapshots.end(),
                                     [&stage_name](const auto &pair)
                                     {
                                         return pair.first.find(stage_name) != std::string::npos;
                                     });

        if (seq_it != seq_snapshots.end() && batch_it != batch_snapshots.end())
        {
            compareSnapshots(stage_name, batch_it->second, seq_it->second);
        }
        else
        {
            if (rank == 0)
            {
                std::cout << "\n=== " << stage_name << " ===" << std::endl;
                if (seq_it == seq_snapshots.end())
                    std::cout << "WARNING: Sequential snapshot not found" << std::endl;
                if (batch_it == batch_snapshots.end())
                    std::cout << "WARNING: Batch snapshot not found" << std::endl;
            }
        }
    }

    if (rank == 0)
    {
        std::cout << "\n=== TEST COMPLETE ===" << std::endl;
        std::cout << "Review the stage-by-stage comparison above to identify where divergence occurs." << std::endl;
    }
}

/**
 * @brief Test: Verify snapshot capture is working
 */
TEST_F(BatchSequentialParityTest, SnapshotCaptureEnabled)
{
    auto rank = MPIContext::capture().rank;

    const std::vector<int> tokens = {1, 2, 3};

    auto seq_pipeline = PipelineFactory::instance().create(sequential_config_);
    ASSERT_NE(seq_pipeline, nullptr);

    auto seq_weights = seq_pipeline->loadWeights(model_path_);
    ASSERT_NE(seq_weights, nullptr);

    StageContext ctx;
    ASSERT_TRUE(seq_pipeline->prefill(tokens, *seq_weights, ctx));

    auto &snapshot_mgr = PipelineSnapshotManager::instance();
    auto snapshots = snapshot_mgr.getAllSnapshots();

    if (rank == 0)
    {
        std::cout << "\nCaptured " << snapshots.size() << " snapshots:" << std::endl;
        for (const auto &[key, snap] : snapshots)
        {
            std::cout << "  - " << key << " (";
            for (size_t i = 0; i < snap.shape.size(); ++i)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << snap.shape[i];
            }
            std::cout << ")" << std::endl;
        }
    }

    // We should have captured at minimum: embedding, attention stages, FFN stages, final norm, lm_head
    EXPECT_GE(snapshots.size(), 10) << "Expected at least 10 snapshots captured";
}
