/**
 * @file Test__PPStageRunner.cpp
 * @brief Integration tests for createPPStageRunner factory function
 *
 * Tests the Pipeline Parallelism stage runner creation and configuration:
 * - First stage (embedding + initial layers)
 * - Middle stage (layers only, no embedding/LM head)
 * - Last stage (final layers + LM head)
 * - Invalid configuration rejection
 *
 * These tests require the test model at models/qwen2.5-0.5b-instruct-q4_0.gguf.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

#include "collective/BackendRouter.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "loaders/ModelContext.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;

namespace
{
    using namespace llaminar2::test;

    // Test model path (relative to workspace root, set as WORKING_DIRECTORY in CMake)
    const std::string TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    struct PrefixRunArtifacts
    {
        float hidden_state_cosine = 0.0f;
        std::map<std::string, std::vector<float>> cpu_snapshots;
        std::map<std::string, std::vector<float>> rocm_snapshots;
    };

    struct SnapshotComparison
    {
        std::string key;
        size_t size = 0;
        bool present_in_cpu = false;
        bool present_in_rocm = false;
        float cosine = 0.0f;
    };

    struct LayerLocalizationResult
    {
        int layer = -1;
        float hidden_state_cosine = 0.0f;
        std::string first_bad_key;
        std::vector<SnapshotComparison> comparisons;
    };

    std::string snapshotSuffix(const std::string &key)
    {
        const size_t pos = key.find('_');
        if (pos == std::string::npos || pos + 1 >= key.size())
        {
            return key;
        }
        return key.substr(pos + 1);
    }

    int snapshotStageOrder(const std::string &key)
    {
        static const std::vector<std::string> kStageOrder = {
            "ATTENTION_NORM",
            "Q_PROJECTION",
            "K_PROJECTION",
            "V_PROJECTION",
            "Q_ROPE",
            "K_ROPE",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "ATTENTION_RESIDUAL",
            "FFN_NORM",
            "FFN_GATE",
            "FFN_UP",
            "FFN_SWIGLU",
            "FFN_DOWN",
            "FFN_RESIDUAL"};

        const std::string suffix = snapshotSuffix(key);
        for (size_t i = 0; i < kStageOrder.size(); ++i)
        {
            if (suffix == kStageOrder[i])
            {
                return static_cast<int>(i);
            }
        }
        return static_cast<int>(kStageOrder.size());
    }

    bool snapshotKeyLess(const std::string &lhs, const std::string &rhs)
    {
        const int lhs_order = snapshotStageOrder(lhs);
        const int rhs_order = snapshotStageOrder(rhs);
        if (lhs_order != rhs_order)
        {
            return lhs_order < rhs_order;
        }
        return lhs < rhs;
    }

    float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
        {
            return 0.0f;
        }

        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        if (norm_a == 0.0 || norm_b == 0.0)
        {
            return 0.0f;
        }

        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    std::map<std::string, std::vector<float>> collectSnapshots(
        const std::unique_ptr<IInferenceRunner> &runner,
        const std::string &prefix)
    {
        std::map<std::string, std::vector<float>> snapshots;
        std::vector<std::string> keys = runner->getSnapshotKeys();
        std::sort(keys.begin(), keys.end(), snapshotKeyLess);

        for (const std::string &key : keys)
        {
            if (!prefix.empty() && key.rfind(prefix, 0) != 0)
            {
                continue;
            }

            size_t size = 0;
            const float *data = runner->getSnapshot(key, size);
            if (!data || size == 0)
            {
                continue;
            }

            snapshots.emplace(key, std::vector<float>(data, data + size));
        }

        return snapshots;
    }

    std::vector<SnapshotComparison> compareSnapshots(
        const std::map<std::string, std::vector<float>> &cpu_snapshots,
        const std::map<std::string, std::vector<float>> &rocm_snapshots)
    {
        std::vector<std::string> keys;
        keys.reserve(cpu_snapshots.size() + rocm_snapshots.size());

        for (const auto &[key, _] : cpu_snapshots)
        {
            keys.push_back(key);
        }
        for (const auto &[key, _] : rocm_snapshots)
        {
            if (std::find(keys.begin(), keys.end(), key) == keys.end())
            {
                keys.push_back(key);
            }
        }

        std::sort(keys.begin(), keys.end(), snapshotKeyLess);

        std::vector<SnapshotComparison> results;
        results.reserve(keys.size());

        for (const std::string &key : keys)
        {
            SnapshotComparison comparison;
            comparison.key = key;

            const auto cpu_it = cpu_snapshots.find(key);
            const auto rocm_it = rocm_snapshots.find(key);
            comparison.present_in_cpu = cpu_it != cpu_snapshots.end();
            comparison.present_in_rocm = rocm_it != rocm_snapshots.end();

            if (comparison.present_in_cpu)
            {
                comparison.size = cpu_it->second.size();
            }
            if (comparison.present_in_rocm)
            {
                comparison.size = std::max(comparison.size, rocm_it->second.size());
            }

            if (comparison.present_in_cpu && comparison.present_in_rocm &&
                cpu_it->second.size() == rocm_it->second.size())
            {
                comparison.cosine = cosineSimilarity(cpu_it->second, rocm_it->second);
            }

            results.push_back(std::move(comparison));
        }

        return results;
    }

    LayerLocalizationResult localizeLayerDivergence(
        const PrefixRunArtifacts &artifacts,
        int layer,
        float snapshot_threshold)
    {
        LayerLocalizationResult result;
        result.layer = layer;
        result.hidden_state_cosine = artifacts.hidden_state_cosine;
        result.comparisons = compareSnapshots(artifacts.cpu_snapshots, artifacts.rocm_snapshots);

        for (const SnapshotComparison &comparison : result.comparisons)
        {
            if (comparison.present_in_cpu &&
                comparison.present_in_rocm &&
                comparison.cosine < snapshot_threshold)
            {
                result.first_bad_key = comparison.key;
                break;
            }
        }

        return result;
    }

    void printLayerLocalizationDiagnostics(const LayerLocalizationResult &result)
    {
        std::cout << "\nLayer " << result.layer << " snapshot cosine diagnostics:" << std::endl;
        for (const SnapshotComparison &comparison : result.comparisons)
        {
            std::cout << "  " << comparison.key
                      << " cpu=" << (comparison.present_in_cpu ? "yes" : "no")
                      << " rocm=" << (comparison.present_in_rocm ? "yes" : "no")
                      << " size=" << comparison.size
                      << " cosine=" << comparison.cosine << std::endl;
        }

        if (!result.first_bad_key.empty())
        {
            std::cout << "First divergent layer-" << result.layer
                      << " snapshot: " << result.first_bad_key << std::endl;
        }
        else
        {
            std::cout << "No divergent layer-" << result.layer
                      << " snapshot found below threshold" << std::endl;
        }
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__PPStageRunner : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Skip if model doesn't exist
            std::ifstream f(TEST_MODEL_PATH);
            if (!f.good())
            {
                GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
            }
        }

        /**
         * @brief Load the test model into a ModelContext
         * @return Shared pointer to ModelContext, or nullptr on failure
         */
        std::shared_ptr<ModelContext> loadModel()
        {
            return ModelContext::create(TEST_MODEL_PATH);
        }

        /**
         * @brief Load a PP stage context with layer-partitioned weights
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding Whether this stage owns embedding
         * @param has_lm_head Whether this stage owns LM head
         * @return Shared pointer to ModelContext, or nullptr on failure
         */
        std::shared_ptr<ModelContext> loadPPStageContext(
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head)
        {
            return ModelContext::createForPPStage(
                TEST_MODEL_PATH,
                first_layer,
                last_layer,
                has_embedding,
                has_lm_head);
        }

        bool hasROCmDevice() const
        {
#ifdef HAVE_ROCM
            int device_count = 0;
            return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
#else
            return false;
#endif
        }

        PrefixRunArtifacts runPrefixParityThroughLayer(
            int last_layer_inclusive,
            int seq_len = 64,
            bool capture_snapshots = false)
        {
            PrefixRunArtifacts artifacts;
            auto cpu_ctx = loadPPStageContext(0, last_layer_inclusive + 1, true, false);
            auto rocm_ctx = loadPPStageContext(0, last_layer_inclusive + 1, true, false);
            EXPECT_NE(cpu_ctx, nullptr);
            EXPECT_NE(rocm_ctx, nullptr);
            if (!cpu_ctx || !rocm_ctx)
            {
                return artifacts;
            }

            FactoryPPStageConfig config;
            config.first_layer = 0;
            config.last_layer = last_layer_inclusive + 1;
            config.has_embedding = true;
            config.has_lm_head = false;
            EXPECT_TRUE(config.isValid());
            if (!config.isValid())
            {
                return artifacts;
            }

            auto cpu_runner = createPPStageRunner(cpu_ctx, DeviceId::cpu(), config);
            auto rocm_runner = createPPStageRunner(rocm_ctx, DeviceId::rocm(0), config);
            EXPECT_NE(cpu_runner, nullptr);
            EXPECT_NE(rocm_runner, nullptr);
            if (!cpu_runner || !rocm_runner)
            {
                return artifacts;
            }

            if (capture_snapshots)
            {
                cpu_runner->enableSnapshotCapture();
                rocm_runner->enableSnapshotCapture();
            }

            std::vector<int> tokens(seq_len);
            for (int i = 0; i < seq_len; ++i)
            {
                tokens[i] = i % 1024;
            }

            EXPECT_TRUE(cpu_runner->forward(tokens.data(), seq_len)) << "CPU prefix runner failed";
            EXPECT_TRUE(rocm_runner->forward(tokens.data(), seq_len)) << "ROCm prefix runner failed";

            TensorBase *cpu_hidden = cpu_runner->getHiddenState();
            TensorBase *rocm_hidden = rocm_runner->getHiddenState();
            EXPECT_NE(cpu_hidden, nullptr);
            EXPECT_NE(rocm_hidden, nullptr);
            if (!cpu_hidden || !rocm_hidden)
            {
                return artifacts;
            }

            EXPECT_EQ(cpu_hidden->numel(), rocm_hidden->numel());
            if (cpu_hidden->numel() != rocm_hidden->numel())
            {
                return artifacts;
            }

            const std::vector<float> cpu_values(cpu_hidden->data(), cpu_hidden->data() + cpu_hidden->numel());
            const std::vector<float> rocm_values(rocm_hidden->data(), rocm_hidden->data() + rocm_hidden->numel());
            artifacts.hidden_state_cosine = cosineSimilarity(cpu_values, rocm_values);

            if (capture_snapshots)
            {
                const std::string prefix = "layer" + std::to_string(last_layer_inclusive) + "_";
                artifacts.cpu_snapshots = collectSnapshots(cpu_runner, prefix);
                artifacts.rocm_snapshots = collectSnapshots(rocm_runner, prefix);
            }

            return artifacts;
        }
    };

    // =============================================================================
    // First Stage Tests (Embedding + Initial Layers)
    // =============================================================================

    /**
     * @test Create first PP stage runner (embedding + layers 0-11)
     *
     * Validates:
     * - Factory returns non-null runner
     * - Stage can be created with has_embedding=true
     */
    TEST_F(Test__PPStageRunner, CreateFirstStageRunner)
    {
        // Qwen2.5-0.5B has 24 layers
        auto stage_ctx = loadPPStageContext(0, 12, true, false);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = 12; // First 12 layers (out of 24)
        config.has_embedding = true;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), 12);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for first stage";
    }

    // =============================================================================
    // Middle Stage Tests (Layers Only)
    // =============================================================================

    /**
     * @test Create middle PP stage runner (layers 8-16, no embedding, no LM head)
     *
     * Validates:
     * - Factory returns non-null runner for middle stage
     * - Stage can be created without embedding or LM head
     */
    TEST_F(Test__PPStageRunner, CreateMiddleStageRunner)
    {
        // Create a middle stage (no embedding, no LM head)
        auto stage_ctx = loadPPStageContext(8, 16, false, false);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 8;
        config.last_layer = 16;
        config.has_embedding = false;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), 8);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for middle stage";
    }

    // =============================================================================
    // Last Stage Tests (Final Layers + LM Head)
    // =============================================================================

    /**
     * @test Create last PP stage runner (layers 12-24 + LM head)
     *
     * Validates:
     * - Factory returns non-null runner for last stage
     * - Stage can be created with has_lm_head=true
     */
    TEST_F(Test__PPStageRunner, CreateLastStageRunner)
    {
        // Get the full model to query layer count
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        int num_layers = full_ctx->blockCount();
        ASSERT_EQ(num_layers, 24) << "Qwen2.5-0.5B should have 24 layers";

        // Create last stage context (layers 12-24, has LM head)
        auto stage_ctx = loadPPStageContext(12, num_layers, false, true);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 12;
        config.last_layer = num_layers;
        config.has_embedding = false;
        config.has_lm_head = true;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), 12);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for last stage";
    }

    // =============================================================================
    // Full Model Single Stage Tests
    // =============================================================================

    /**
     * @test Create single-stage PP runner that covers all layers
     *
     * This is equivalent to non-PP mode: one stage has everything.
     */
    TEST_F(Test__PPStageRunner, CreateSingleStageRunnerAllLayers)
    {
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        int num_layers = full_ctx->blockCount();

        // Create a stage that covers all layers (degenerate PP with 1 stage)
        auto stage_ctx = loadPPStageContext(0, num_layers, true, true);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = num_layers;
        config.has_embedding = true;
        config.has_lm_head = true;

        ASSERT_TRUE(config.isValid()) << "PP config should be valid";
        EXPECT_EQ(config.layerCount(), num_layers);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr) << "createPPStageRunner returned nullptr for full model stage";
    }

    TEST_F(Test__PPStageRunner, PrefixThroughLayer20Parity_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        constexpr float COSINE_THRESHOLD = 0.995f;
        const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(20);

        EXPECT_GT(artifacts.hidden_state_cosine, COSINE_THRESHOLD)
            << "Layer-20 prefix ROCm hidden-state cosine too low";
    }

    TEST_F(Test__PPStageRunner, PrefixThroughLayer21Parity_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        constexpr float COSINE_THRESHOLD = 0.995f;
        const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(21);

        EXPECT_GT(artifacts.hidden_state_cosine, COSINE_THRESHOLD)
            << "Layer-21 prefix ROCm hidden-state cosine too low";
    }

    TEST_F(Test__PPStageRunner, PrefixThroughLayer21SnapshotLocalization_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        constexpr float HIDDEN_STATE_THRESHOLD = 0.995f;
        constexpr float SNAPSHOT_THRESHOLD = 0.995f;

        const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(21, 64, true);
        EXPECT_LT(artifacts.hidden_state_cosine, HIDDEN_STATE_THRESHOLD)
            << "Expected layer-21 reduced runner to reproduce the ROCm divergence";

        const std::vector<SnapshotComparison> comparisons =
            compareSnapshots(artifacts.cpu_snapshots, artifacts.rocm_snapshots);
        ASSERT_FALSE(comparisons.empty()) << "No layer-21 snapshots captured for comparison";

        std::string first_bad_key;
        std::cout << "\nLayer 21 snapshot cosine diagnostics:" << std::endl;
        for (const SnapshotComparison &comparison : comparisons)
        {
            std::cout << "  " << comparison.key
                      << " cpu=" << (comparison.present_in_cpu ? "yes" : "no")
                      << " rocm=" << (comparison.present_in_rocm ? "yes" : "no")
                      << " size=" << comparison.size
                      << " cosine=" << comparison.cosine << std::endl;

            if (first_bad_key.empty() &&
                comparison.present_in_cpu &&
                comparison.present_in_rocm &&
                comparison.cosine < SNAPSHOT_THRESHOLD)
            {
                first_bad_key = comparison.key;
            }
        }

        EXPECT_FALSE(first_bad_key.empty())
            << "No divergent layer-21 snapshot found despite hidden-state divergence";

        if (!first_bad_key.empty())
        {
            std::cout << "First divergent layer-21 snapshot: " << first_bad_key << std::endl;
        }
    }

    TEST_F(Test__PPStageRunner, PrefixEarliestSnapshotDivergence_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        constexpr float SNAPSHOT_THRESHOLD = 0.995f;
        constexpr int MAX_LAYER_TO_SCAN = 20;

        std::vector<LayerLocalizationResult> scan_results;
        scan_results.reserve(MAX_LAYER_TO_SCAN + 1);

        int earliest_bad_layer = -1;
        for (int layer = 0; layer <= MAX_LAYER_TO_SCAN; ++layer)
        {
            const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(layer, 64, true);
            LayerLocalizationResult result = localizeLayerDivergence(artifacts, layer, SNAPSHOT_THRESHOLD);
            std::cout << "Layer " << layer
                      << " hidden_state_cosine=" << result.hidden_state_cosine
                      << " first_bad=" << (result.first_bad_key.empty() ? "<none>" : result.first_bad_key)
                      << std::endl;
            scan_results.push_back(std::move(result));

            if (!scan_results.back().first_bad_key.empty())
            {
                earliest_bad_layer = layer;
                break;
            }
        }

        ASSERT_GE(earliest_bad_layer, 0)
            << "No divergent snapshot found up to layer " << MAX_LAYER_TO_SCAN;

        const LayerLocalizationResult &culprit = scan_results.back();
        printLayerLocalizationDiagnostics(culprit);

        std::cout << "Earliest layer with a sub-threshold snapshot: " << earliest_bad_layer << std::endl;
        EXPECT_FALSE(culprit.first_bad_key.empty());
    }

    TEST_F(Test__PPStageRunner, PrefixThroughLayer20SnapshotLocalization_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        constexpr float HIDDEN_STATE_THRESHOLD = 0.995f;
        constexpr float SNAPSHOT_THRESHOLD = 0.995f;

        const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(20, 64, true);
        EXPECT_GT(artifacts.hidden_state_cosine, HIDDEN_STATE_THRESHOLD)
            << "Expected layer-20 reduced runner to remain above the hidden-state threshold";

        const LayerLocalizationResult result = localizeLayerDivergence(artifacts, 20, SNAPSHOT_THRESHOLD);
        ASSERT_FALSE(result.comparisons.empty()) << "No layer-20 snapshots captured for comparison";
        printLayerLocalizationDiagnostics(result);

        EXPECT_FALSE(result.first_bad_key.empty())
            << "No divergent layer-20 snapshot found below the localization threshold";
    }

    TEST_F(Test__PPStageRunner, PrefixThroughLayer2SnapshotDiagnostics_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(2, 64, true);
        const std::vector<SnapshotComparison> comparisons =
            compareSnapshots(artifacts.cpu_snapshots, artifacts.rocm_snapshots);
        ASSERT_FALSE(comparisons.empty()) << "No layer-2 snapshots captured for comparison";

        std::cout << "\nLayer 2 hidden_state_cosine=" << artifacts.hidden_state_cosine << std::endl;

        std::string min_key;
        float min_cosine = 1.0f;
        for (const SnapshotComparison &comparison : comparisons)
        {
            std::cout << "  " << comparison.key
                      << " cpu=" << (comparison.present_in_cpu ? "yes" : "no")
                      << " rocm=" << (comparison.present_in_rocm ? "yes" : "no")
                      << " size=" << comparison.size
                      << " cosine=" << comparison.cosine << std::endl;

            if (comparison.present_in_cpu && comparison.present_in_rocm && comparison.cosine < min_cosine)
            {
                min_cosine = comparison.cosine;
                min_key = comparison.key;
            }
        }

        ASSERT_FALSE(min_key.empty()) << "Failed to compute minimum layer-2 snapshot cosine";
        std::cout << "Minimum layer-2 snapshot cosine: " << min_key
                  << " => " << min_cosine << std::endl;
    }

    TEST_F(Test__PPStageRunner, PrefixThroughLayer1SnapshotDiagnostics_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(1, 64, true);
        const std::vector<SnapshotComparison> comparisons =
            compareSnapshots(artifacts.cpu_snapshots, artifacts.rocm_snapshots);
        ASSERT_FALSE(comparisons.empty()) << "No layer-1 snapshots captured for comparison";

        std::cout << "\nLayer 1 hidden_state_cosine=" << artifacts.hidden_state_cosine << std::endl;

        std::string min_key;
        float min_cosine = 1.0f;
        for (const SnapshotComparison &comparison : comparisons)
        {
            std::cout << "  " << comparison.key
                      << " cpu=" << (comparison.present_in_cpu ? "yes" : "no")
                      << " rocm=" << (comparison.present_in_rocm ? "yes" : "no")
                      << " size=" << comparison.size
                      << " cosine=" << comparison.cosine << std::endl;

            if (comparison.present_in_cpu && comparison.present_in_rocm && comparison.cosine < min_cosine)
            {
                min_cosine = comparison.cosine;
                min_key = comparison.key;
            }
        }

        ASSERT_FALSE(min_key.empty()) << "Failed to compute minimum layer-1 snapshot cosine";
        std::cout << "Minimum layer-1 snapshot cosine: " << min_key
                  << " => " << min_cosine << std::endl;
    }

    TEST_F(Test__PPStageRunner, PrefixThroughLayer0SnapshotDiagnostics_CPUvsROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const PrefixRunArtifacts artifacts = runPrefixParityThroughLayer(0, 64, true);
        const std::vector<SnapshotComparison> comparisons =
            compareSnapshots(artifacts.cpu_snapshots, artifacts.rocm_snapshots);
        ASSERT_FALSE(comparisons.empty()) << "No layer-0 snapshots captured for comparison";

        std::cout << "\nLayer 0 hidden_state_cosine=" << artifacts.hidden_state_cosine << std::endl;

        std::string min_key;
        float min_cosine = 1.0f;
        for (const SnapshotComparison &comparison : comparisons)
        {
            std::cout << "  " << comparison.key
                      << " cpu=" << (comparison.present_in_cpu ? "yes" : "no")
                      << " rocm=" << (comparison.present_in_rocm ? "yes" : "no")
                      << " size=" << comparison.size
                      << " cosine=" << comparison.cosine << std::endl;

            if (comparison.present_in_cpu && comparison.present_in_rocm && comparison.cosine < min_cosine)
            {
                min_cosine = comparison.cosine;
                min_key = comparison.key;
            }
        }

        ASSERT_FALSE(min_key.empty()) << "Failed to compute minimum layer-0 snapshot cosine";
        std::cout << "Minimum layer-0 snapshot cosine: " << min_key
                  << " => " << min_cosine << std::endl;
    }

    // =============================================================================
    // Invalid Configuration Tests
    // =============================================================================

    /**
     * @test Reject invalid PP config where first_layer > last_layer
     *
     * Either the factory should return nullptr, or FactoryPPStageConfig::isValid() should fail.
     */
    TEST_F(Test__PPStageRunner, RejectsInvalidConfigFirstGreaterThanLast)
    {
        auto model_ctx = loadModel();
        ASSERT_NE(model_ctx, nullptr);

        FactoryPPStageConfig invalid_config;
        invalid_config.first_layer = 10;
        invalid_config.last_layer = 5; // Invalid: first > last
        invalid_config.has_embedding = false;
        invalid_config.has_lm_head = false;

        // Config validation should fail
        EXPECT_FALSE(invalid_config.isValid())
            << "Config with first_layer > last_layer should be invalid";

        // Factory should either return nullptr or throw for invalid config
        // We use a model context that wasn't layer-partitioned; the factory
        // should still validate the FactoryPPStageConfig before proceeding.
        auto runner = createPPStageRunner(model_ctx, DeviceId::cpu(), invalid_config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for invalid config";
    }

    /**
     * @test Reject invalid PP config where first_layer == last_layer (zero layers)
     */
    TEST_F(Test__PPStageRunner, RejectsInvalidConfigZeroLayers)
    {
        auto model_ctx = loadModel();
        ASSERT_NE(model_ctx, nullptr);

        FactoryPPStageConfig invalid_config;
        invalid_config.first_layer = 5;
        invalid_config.last_layer = 5; // Invalid: zero layers
        invalid_config.has_embedding = false;
        invalid_config.has_lm_head = false;

        EXPECT_FALSE(invalid_config.isValid())
            << "Config with first_layer == last_layer should be invalid";

        auto runner = createPPStageRunner(model_ctx, DeviceId::cpu(), invalid_config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for zero-layer config";
    }

    /**
     * @test Reject invalid PP config with negative first_layer
     */
    TEST_F(Test__PPStageRunner, RejectsInvalidConfigNegativeFirstLayer)
    {
        auto model_ctx = loadModel();
        ASSERT_NE(model_ctx, nullptr);

        FactoryPPStageConfig invalid_config;
        invalid_config.first_layer = -1;
        invalid_config.last_layer = 10;
        invalid_config.has_embedding = true;
        invalid_config.has_lm_head = false;

        EXPECT_FALSE(invalid_config.isValid())
            << "Config with negative first_layer should be invalid";

        auto runner = createPPStageRunner(model_ctx, DeviceId::cpu(), invalid_config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for negative first_layer";
    }

    // =============================================================================
    // Null Model Context Tests
    // =============================================================================

    /**
     * @test Reject null model context
     */
    TEST_F(Test__PPStageRunner, RejectsNullModelContext)
    {
        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = 12;
        config.has_embedding = true;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid());

        auto runner = createPPStageRunner(nullptr, DeviceId::cpu(), config);
        EXPECT_EQ(runner, nullptr)
            << "createPPStageRunner should return nullptr for null model context";
    }

    // =============================================================================
    // Layer Range Boundary Tests
    // =============================================================================

    /**
     * @test Create PP stage with single layer
     */
    TEST_F(Test__PPStageRunner, CreateSingleLayerStage)
    {
        // Create a minimal stage with just one layer (layer 5)
        auto stage_ctx = loadPPStageContext(5, 6, false, false);
        ASSERT_NE(stage_ctx, nullptr) << "Failed to create PP stage context";

        FactoryPPStageConfig config;
        config.first_layer = 5;
        config.last_layer = 6; // Single layer
        config.has_embedding = false;
        config.has_lm_head = false;

        ASSERT_TRUE(config.isValid());
        EXPECT_EQ(config.layerCount(), 1);

        auto runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr)
            << "createPPStageRunner returned nullptr for single-layer stage";
    }

    /**
     * @test Verify PP config layer count calculation
     */
    TEST_F(Test__PPStageRunner, LayerCountCalculation)
    {
        FactoryPPStageConfig config;

        // Two-stage PP: 24 layers total
        config.first_layer = 0;
        config.last_layer = 12;
        EXPECT_EQ(config.layerCount(), 12);

        config.first_layer = 12;
        config.last_layer = 24;
        EXPECT_EQ(config.layerCount(), 12);

        // Three-stage PP: 24 layers = 8 + 8 + 8
        config.first_layer = 0;
        config.last_layer = 8;
        EXPECT_EQ(config.layerCount(), 8);

        config.first_layer = 8;
        config.last_layer = 16;
        EXPECT_EQ(config.layerCount(), 8);

        config.first_layer = 16;
        config.last_layer = 24;
        EXPECT_EQ(config.layerCount(), 8);
    }

} // anonymous namespace
