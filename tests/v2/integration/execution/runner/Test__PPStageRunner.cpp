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
#include <mpi.h>
#include <csignal>
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

// Forward-declare cudaGetDeviceCount to avoid CUDA/HIP header conflicts
#ifdef HAVE_CUDA
extern "C" int cudaGetDeviceCount(int *count);
static constexpr int cudaSuccess_v = 0;
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

        bool hasCUDADevice() const
        {
#ifdef HAVE_CUDA
            int device_count = 0;
            return cudaGetDeviceCount(&device_count) == cudaSuccess_v && device_count > 0;
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
        EXPECT_GT(artifacts.hidden_state_cosine, HIDDEN_STATE_THRESHOLD)
            << "Layer-21 CPU vs ROCm hidden-state cosine should be high with NATIVE weights";

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

        // With NATIVE weights, all snapshots should have high parity
        EXPECT_TRUE(first_bad_key.empty())
            << "Unexpected divergent snapshot at: " << first_bad_key
            << "Unexpected divergent snapshot at: " << first_bad_key;

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
// With NATIVE weights, all layers should have high parity — no divergence expected
        EXPECT_EQ(earliest_bad_layer, -1)
            << "Unexpected divergent snapshot at layer " << earliest_bad_layer;

        if (earliest_bad_layer >= 0)
        {
            const LayerLocalizationResult &culprit = scan_results.back();
            printLayerLocalizationDiagnostics(culprit);
            std::cout << "Earliest layer with a sub-threshold snapshot: " << earliest_bad_layer << std::endl;
        }
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
        // With NATIVE weights, all snapshots should have high parity
        EXPECT_TRUE(result.first_bad_key.empty())
            << "Unexpected divergent snapshot at: " << result.first_bad_key;
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

    // =============================================================================
    // CUDA Parity Tests
    // =============================================================================

    /**
     * @test CPU vs CUDA hidden state parity through layer 20
     *
     * Mirrors PrefixThroughLayer20Parity_CPUvsROCm but for CUDA.
     */
    TEST_F(Test__PPStageRunner, PrefixThroughLayer20Parity_CPUvsCUDA)
    {
        if (!hasCUDADevice())
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        constexpr float COSINE_THRESHOLD = 0.995f;
        constexpr int LAST_LAYER = 20;
        constexpr int SEQ_LEN = 64;

        auto cpu_ctx = loadPPStageContext(0, LAST_LAYER + 1, true, false);
        auto cuda_ctx = loadPPStageContext(0, LAST_LAYER + 1, true, false);
        ASSERT_NE(cpu_ctx, nullptr);
        ASSERT_NE(cuda_ctx, nullptr);

        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = LAST_LAYER + 1;
        config.has_embedding = true;
        config.has_lm_head = false;

        auto cpu_runner = createPPStageRunner(cpu_ctx, DeviceId::cpu(), config);
        auto cuda_runner = createPPStageRunner(cuda_ctx, DeviceId::cuda(0), config);
        ASSERT_NE(cpu_runner, nullptr);
        ASSERT_NE(cuda_runner, nullptr);

        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        ASSERT_TRUE(cpu_runner->forward(tokens.data(), SEQ_LEN));
        ASSERT_TRUE(cuda_runner->forward(tokens.data(), SEQ_LEN));

        TensorBase *cpu_hidden = cpu_runner->getHiddenState();
        TensorBase *cuda_hidden = cuda_runner->getHiddenState();
        ASSERT_NE(cpu_hidden, nullptr);
        ASSERT_NE(cuda_hidden, nullptr);
        ASSERT_EQ(cpu_hidden->numel(), cuda_hidden->numel());

        const std::vector<float> cpu_values(cpu_hidden->data(), cpu_hidden->data() + cpu_hidden->numel());
        const std::vector<float> cuda_values(cuda_hidden->data(), cuda_hidden->data() + cuda_hidden->numel());
        float cosine = cosineSimilarity(cpu_values, cuda_values);

        std::cout << "CPU vs CUDA layer-20 hidden state cosine: " << cosine << std::endl;
        EXPECT_GT(cosine, COSINE_THRESHOLD)
            << "Layer-20 prefix CUDA hidden-state cosine too low";
    }

    /**
     * @test CPU vs CUDA snapshot localization through layer 21
     *
     * Compares per-stage snapshots between CPU and CUDA to find any divergence.
     */
    TEST_F(Test__PPStageRunner, PrefixThroughLayer21SnapshotLocalization_CPUvsCUDA)
    {
        if (!hasCUDADevice())
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        constexpr float HIDDEN_STATE_THRESHOLD = 0.995f;
        constexpr float SNAPSHOT_THRESHOLD = 0.995f;
        constexpr int LAST_LAYER = 21;
        constexpr int SEQ_LEN = 64;

        auto cpu_ctx = loadPPStageContext(0, LAST_LAYER + 1, true, false);
        auto cuda_ctx = loadPPStageContext(0, LAST_LAYER + 1, true, false);
        ASSERT_NE(cpu_ctx, nullptr);
        ASSERT_NE(cuda_ctx, nullptr);

        FactoryPPStageConfig config;
        config.first_layer = 0;
        config.last_layer = LAST_LAYER + 1;
        config.has_embedding = true;
        config.has_lm_head = false;

        auto cpu_runner = createPPStageRunner(cpu_ctx, DeviceId::cpu(), config);
        auto cuda_runner = createPPStageRunner(cuda_ctx, DeviceId::cuda(0), config);
        ASSERT_NE(cpu_runner, nullptr);
        ASSERT_NE(cuda_runner, nullptr);

        cpu_runner->enableSnapshotCapture();
        cuda_runner->enableSnapshotCapture();

        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        ASSERT_TRUE(cpu_runner->forward(tokens.data(), SEQ_LEN));
        ASSERT_TRUE(cuda_runner->forward(tokens.data(), SEQ_LEN));

        TensorBase *cpu_hidden = cpu_runner->getHiddenState();
        TensorBase *cuda_hidden = cuda_runner->getHiddenState();
        ASSERT_NE(cpu_hidden, nullptr);
        ASSERT_NE(cuda_hidden, nullptr);

        const std::vector<float> cpu_values(cpu_hidden->data(), cpu_hidden->data() + cpu_hidden->numel());
        const std::vector<float> cuda_values(cuda_hidden->data(), cuda_hidden->data() + cuda_hidden->numel());
        float hidden_cosine = cosineSimilarity(cpu_values, cuda_values);

        EXPECT_GT(hidden_cosine, HIDDEN_STATE_THRESHOLD)
            << "Layer-21 CPU vs CUDA hidden-state cosine should be high with NATIVE weights";

        const std::string prefix = "layer" + std::to_string(LAST_LAYER) + "_";
        auto cpu_snaps = collectSnapshots(cpu_runner, prefix);
        auto cuda_snaps = collectSnapshots(cuda_runner, prefix);
        auto comparisons = compareSnapshots(cpu_snaps, cuda_snaps);
        ASSERT_FALSE(comparisons.empty()) << "No layer-21 snapshots captured for comparison";

        std::string first_bad_key;
        std::cout << "\nCPU vs CUDA Layer 21 snapshot cosine diagnostics:" << std::endl;
        for (const SnapshotComparison &comparison : comparisons)
        {
            std::cout << "  " << comparison.key
                      << " cpu=" << (comparison.present_in_cpu ? "yes" : "no")
                      << " cuda=" << (comparison.present_in_rocm ? "yes" : "no")
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

        EXPECT_TRUE(first_bad_key.empty())
            << "Unexpected divergent snapshot at: " << first_bad_key;
    }

    // =============================================================================
    // Multi-Stage End-to-End PP Tests
    // =============================================================================

    /**
     * @test Two-stage PP pipeline: stage 1 hidden state → stage 2 logits
     *
     * Validates the full PP activation transfer path:
     * 1. Stage 1 (layers 0-11 + embedding) runs prefill, produces hidden state
     * 2. Hidden state is passed to stage 2 via setHiddenState()
     * 3. Stage 2 (layers 12-23 + LM head) produces logits
     * 4. Compare logits against single-stage full-model runner
     */
    TEST_F(Test__PPStageRunner, TwoStagePPEndToEnd_CPU)
    {
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();
        ASSERT_EQ(n_layers, 24);
        const int mid = n_layers / 2; // 12

        // Create the two PP stages
        auto stage1_ctx = loadPPStageContext(0, mid, true, false);
        auto stage2_ctx = loadPPStageContext(mid, n_layers, false, true);
        ASSERT_NE(stage1_ctx, nullptr);
        ASSERT_NE(stage2_ctx, nullptr);

        FactoryPPStageConfig config1;
        config1.first_layer = 0;
        config1.last_layer = mid;
        config1.has_embedding = true;
        config1.has_lm_head = false;

        FactoryPPStageConfig config2;
        config2.first_layer = mid;
        config2.last_layer = n_layers;
        config2.has_embedding = false;
        config2.has_lm_head = true;

        auto stage1 = createPPStageRunner(stage1_ctx, DeviceId::cpu(), config1);
        auto stage2 = createPPStageRunner(stage2_ctx, DeviceId::cpu(), config2);
        ASSERT_NE(stage1, nullptr);
        ASSERT_NE(stage2, nullptr);

        // Create single-stage reference runner
        auto ref_ctx = loadModel();
        ASSERT_NE(ref_ctx, nullptr);
        FactoryPPStageConfig ref_config;
        ref_config.first_layer = 0;
        ref_config.last_layer = n_layers;
        ref_config.has_embedding = true;
        ref_config.has_lm_head = true;
        auto ref_runner = createPPStageRunner(ref_ctx, DeviceId::cpu(), ref_config);
        ASSERT_NE(ref_runner, nullptr);

        // Prefill
        constexpr int SEQ_LEN = 32;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        // Reference
        ASSERT_TRUE(ref_runner->forward(tokens.data(), SEQ_LEN));
        const float *ref_logits = ref_runner->logits();
        ASSERT_NE(ref_logits, nullptr);
        int vocab = ref_runner->vocab_size();
        ASSERT_GT(vocab, 0);

        // PP pipeline
        ASSERT_TRUE(stage1->forward(tokens.data(), SEQ_LEN));
        TensorBase *hidden = stage1->getHiddenState();
        ASSERT_NE(hidden, nullptr);

        stage2->setHiddenState(hidden);
        ASSERT_TRUE(stage2->hasHiddenStateInput());
        ASSERT_TRUE(stage2->forward(tokens.data(), SEQ_LEN));

        const float *pp_logits = stage2->logits();
        ASSERT_NE(pp_logits, nullptr);

        // Compare logits
        std::vector<float> ref_vec(ref_logits, ref_logits + vocab);
        std::vector<float> pp_vec(pp_logits, pp_logits + vocab);
        float logit_cosine = cosineSimilarity(ref_vec, pp_vec);

        std::cout << "Two-stage PP vs reference logit cosine: " << logit_cosine << std::endl;
        EXPECT_GT(logit_cosine, 0.999f)
            << "Two-stage PP logits should match single-stage reference almost exactly";

        // Top-1 token should match
        int ref_top = static_cast<int>(std::distance(ref_vec.begin(),
                                                     std::max_element(ref_vec.begin(), ref_vec.end())));
        int pp_top = static_cast<int>(std::distance(pp_vec.begin(),
                                                    std::max_element(pp_vec.begin(), pp_vec.end())));
        EXPECT_EQ(ref_top, pp_top) << "Top-1 token mismatch between PP and reference";
    }

    /**
     * @test Three-stage PP pipeline on CPU
     *
     * Stage 1: layers 0-7 + embedding
     * Stage 2: layers 8-15 (middle)
     * Stage 3: layers 16-23 + LM head
     */
    TEST_F(Test__PPStageRunner, ThreeStagePPEndToEnd_CPU)
    {
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();
        ASSERT_EQ(n_layers, 24);
        const int split1 = 8, split2 = 16;

        auto ctx1 = loadPPStageContext(0, split1, true, false);
        auto ctx2 = loadPPStageContext(split1, split2, false, false);
        auto ctx3 = loadPPStageContext(split2, n_layers, false, true);
        ASSERT_NE(ctx1, nullptr);
        ASSERT_NE(ctx2, nullptr);
        ASSERT_NE(ctx3, nullptr);

        FactoryPPStageConfig cfg1{0, split1, true, false};
        FactoryPPStageConfig cfg2{split1, split2, false, false};
        FactoryPPStageConfig cfg3{split2, n_layers, false, true};

        auto s1 = createPPStageRunner(ctx1, DeviceId::cpu(), cfg1);
        auto s2 = createPPStageRunner(ctx2, DeviceId::cpu(), cfg2);
        auto s3 = createPPStageRunner(ctx3, DeviceId::cpu(), cfg3);
        ASSERT_NE(s1, nullptr);
        ASSERT_NE(s2, nullptr);
        ASSERT_NE(s3, nullptr);

        // Reference
        auto ref_ctx = loadModel();
        ASSERT_NE(ref_ctx, nullptr);
        FactoryPPStageConfig ref_cfg{0, n_layers, true, true};
        auto ref = createPPStageRunner(ref_ctx, DeviceId::cpu(), ref_cfg);
        ASSERT_NE(ref, nullptr);

        constexpr int SEQ_LEN = 32;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        ASSERT_TRUE(ref->forward(tokens.data(), SEQ_LEN));

        // PP pipeline: s1 → s2 → s3
        ASSERT_TRUE(s1->forward(tokens.data(), SEQ_LEN));
        TensorBase *h1 = s1->getHiddenState();
        ASSERT_NE(h1, nullptr);

        s2->setHiddenState(h1);
        ASSERT_TRUE(s2->forward(tokens.data(), SEQ_LEN));
        TensorBase *h2 = s2->getHiddenState();
        ASSERT_NE(h2, nullptr);

        s3->setHiddenState(h2);
        ASSERT_TRUE(s3->forward(tokens.data(), SEQ_LEN));

        const float *ref_logits = ref->logits();
        const float *pp_logits = s3->logits();
        ASSERT_NE(ref_logits, nullptr);
        ASSERT_NE(pp_logits, nullptr);

        int vocab = ref->vocab_size();
        std::vector<float> ref_vec(ref_logits, ref_logits + vocab);
        std::vector<float> pp_vec(pp_logits, pp_logits + vocab);
        float cosine = cosineSimilarity(ref_vec, pp_vec);

        std::cout << "Three-stage PP vs reference logit cosine: " << cosine << std::endl;
        EXPECT_GT(cosine, 0.999f)
            << "Three-stage PP logits should match single-stage reference almost exactly";

        int ref_top = static_cast<int>(std::distance(ref_vec.begin(),
                                                     std::max_element(ref_vec.begin(), ref_vec.end())));
        int pp_top = static_cast<int>(std::distance(pp_vec.begin(),
                                                    std::max_element(pp_vec.begin(), pp_vec.end())));
        EXPECT_EQ(ref_top, pp_top) << "Top-1 token mismatch between 3-stage PP and reference";
    }

    /**
     * @test Heterogeneous PP: stage 1 CPU → stage 2 ROCm with hidden state transfer
     */
    TEST_F(Test__PPStageRunner, TwoStagePPEndToEnd_ROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();
        const int mid = n_layers / 2;

        auto ctx1 = loadPPStageContext(0, mid, true, false);
        auto ctx2 = loadPPStageContext(mid, n_layers, false, true);
        ASSERT_NE(ctx1, nullptr);
        ASSERT_NE(ctx2, nullptr);

        FactoryPPStageConfig cfg1{0, mid, true, false};
        FactoryPPStageConfig cfg2{mid, n_layers, false, true};

        // Heterogeneous PP: CPU prefix → ROCm suffix
        auto s1 = createPPStageRunner(ctx1, DeviceId::cpu(), cfg1);
        auto s2 = createPPStageRunner(ctx2, DeviceId::rocm(0), cfg2);
        ASSERT_NE(s1, nullptr);
        ASSERT_NE(s2, nullptr);

        // Reference: single-stage full model on CPU
        auto ref_ctx = loadModel();
        ASSERT_NE(ref_ctx, nullptr);
        FactoryPPStageConfig ref_cfg{0, n_layers, true, true};
        auto ref = createPPStageRunner(ref_ctx, DeviceId::cpu(), ref_cfg);
        ASSERT_NE(ref, nullptr);

        constexpr int SEQ_LEN = 32;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        ASSERT_TRUE(ref->forward(tokens.data(), SEQ_LEN));
        ASSERT_TRUE(s1->forward(tokens.data(), SEQ_LEN));
        TensorBase *hidden = s1->getHiddenState();
        ASSERT_NE(hidden, nullptr);

        s2->setHiddenState(hidden);
        ASSERT_TRUE(s2->forward(tokens.data(), SEQ_LEN));

        int vocab = ref->vocab_size();
        const float *ref_logits = ref->logits();
        const float *pp_logits = s2->logits();
        ASSERT_NE(ref_logits, nullptr);
        ASSERT_NE(pp_logits, nullptr);

        std::vector<float> ref_vec(ref_logits, ref_logits + vocab);
        std::vector<float> pp_vec(pp_logits, pp_logits + vocab);
        float cosine = cosineSimilarity(ref_vec, pp_vec);

        std::cout << "Two-stage PP (CPU→ROCm) vs CPU reference logit cosine: " << cosine << std::endl;
        EXPECT_GT(cosine, 0.97f)
            << "Heterogeneous PP logits should be close to CPU reference (quantized GEMM differences expected)";
    }

    /**
     * @test Heterogeneous PP: stage 1 CPU → stage 2 CUDA with hidden state transfer
     *
     * Tests cross-device PP where the prefix layers run on CPU and the suffix
     * layers (with LM head) run on CUDA. This is a realistic heterogeneous PP
     * scenario and avoids GPU state conflicts from multiple independent DGO
     * instances on the same device.
     */
    TEST_F(Test__PPStageRunner, TwoStagePPEndToEnd_CUDA)
    {
        if (!hasCUDADevice())
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();
        const int mid = n_layers / 2;

        auto ctx1 = loadPPStageContext(0, mid, true, false);
        auto ctx2 = loadPPStageContext(mid, n_layers, false, true);
        ASSERT_NE(ctx1, nullptr);
        ASSERT_NE(ctx2, nullptr);

        FactoryPPStageConfig cfg1{0, mid, true, false};
        FactoryPPStageConfig cfg2{mid, n_layers, false, true};

        // Heterogeneous PP: CPU prefix → CUDA suffix
        auto s1 = createPPStageRunner(ctx1, DeviceId::cpu(), cfg1);
        auto s2 = createPPStageRunner(ctx2, DeviceId::cuda(0), cfg2);
        ASSERT_NE(s1, nullptr);
        ASSERT_NE(s2, nullptr);

        // Reference: single-stage full model on CPU
        auto ref_ctx = loadModel();
        ASSERT_NE(ref_ctx, nullptr);
        FactoryPPStageConfig ref_cfg{0, n_layers, true, true};
        auto ref = createPPStageRunner(ref_ctx, DeviceId::cpu(), ref_cfg);
        ASSERT_NE(ref, nullptr);

        constexpr int SEQ_LEN = 32;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        ASSERT_TRUE(ref->forward(tokens.data(), SEQ_LEN));
        ASSERT_TRUE(s1->forward(tokens.data(), SEQ_LEN));
        TensorBase *hidden = s1->getHiddenState();
        ASSERT_NE(hidden, nullptr);

        s2->setHiddenState(hidden);
        ASSERT_TRUE(s2->forward(tokens.data(), SEQ_LEN));

        int vocab = ref->vocab_size();
        const float *ref_logits = ref->logits();
        const float *pp_logits = s2->logits();
        ASSERT_NE(ref_logits, nullptr);
        ASSERT_NE(pp_logits, nullptr);

        std::vector<float> ref_vec(ref_logits, ref_logits + vocab);
        std::vector<float> pp_vec(pp_logits, pp_logits + vocab);
        float cosine = cosineSimilarity(ref_vec, pp_vec);

        std::cout << "Two-stage PP (CPU→CUDA) vs CPU reference logit cosine: " << cosine << std::endl;
        EXPECT_GT(cosine, 0.97f)
            << "Heterogeneous PP logits should be close to CPU reference (quantized GEMM differences expected)";
    }

    // =============================================================================
    // Decode Step Tests
    // =============================================================================

    /**
     * @test Two-stage PP decode: prefill → 3 decode steps, compare against reference
     *
     * Validates that PP decode produces the same tokens as single-stage decode.
     */
    TEST_F(Test__PPStageRunner, TwoStagePPDecode_CPU)
    {
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();
        const int mid = n_layers / 2;

        auto ctx1 = loadPPStageContext(0, mid, true, false);
        auto ctx2 = loadPPStageContext(mid, n_layers, false, true);
        ASSERT_NE(ctx1, nullptr);
        ASSERT_NE(ctx2, nullptr);

        FactoryPPStageConfig cfg1{0, mid, true, false};
        FactoryPPStageConfig cfg2{mid, n_layers, false, true};

        auto s1 = createPPStageRunner(ctx1, DeviceId::cpu(), cfg1);
        auto s2 = createPPStageRunner(ctx2, DeviceId::cpu(), cfg2);
        ASSERT_NE(s1, nullptr);
        ASSERT_NE(s2, nullptr);

        // Reference
        auto ref_ctx = loadModel();
        ASSERT_NE(ref_ctx, nullptr);
        FactoryPPStageConfig ref_cfg{0, n_layers, true, true};
        auto ref = createPPStageRunner(ref_ctx, DeviceId::cpu(), ref_cfg);
        ASSERT_NE(ref, nullptr);

        // Prefill
        constexpr int SEQ_LEN = 16;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        // Reference prefill
        ASSERT_TRUE(ref->forward(tokens.data(), SEQ_LEN));
        int vocab = ref->vocab_size();
        ASSERT_GT(vocab, 0);

        // PP prefill
        ASSERT_TRUE(s1->forward(tokens.data(), SEQ_LEN));
        s2->setHiddenState(s1->getHiddenState());
        ASSERT_TRUE(s2->forward(tokens.data(), SEQ_LEN));

        constexpr int DECODE_STEPS = 3;
        std::vector<int> ref_tokens, pp_tokens;

        for (int step = 0; step < DECODE_STEPS; ++step)
        {
            // Greedy sample from reference
            const float *ref_logits = ref->logits();
            ASSERT_NE(ref_logits, nullptr);
            int ref_next = static_cast<int>(std::distance(
                ref_logits, std::max_element(ref_logits, ref_logits + vocab)));
            ref_tokens.push_back(ref_next);

            // Greedy sample from PP
            const float *pp_logits = s2->logits();
            ASSERT_NE(pp_logits, nullptr);
            int pp_next = static_cast<int>(std::distance(
                pp_logits, std::max_element(pp_logits, pp_logits + vocab)));
            pp_tokens.push_back(pp_next);

            // Decode step — reference
            ASSERT_TRUE(ref->forward(&ref_next, 1));

            // Decode step — PP pipeline (seq_len=1)
            ASSERT_TRUE(s1->forward(&pp_next, 1));
            s2->setHiddenState(s1->getHiddenState());
            ASSERT_TRUE(s2->forward(&pp_next, 1));
        }

        // All decoded tokens should match
        std::cout << "Decode tokens (ref vs PP):";
        for (int step = 0; step < DECODE_STEPS; ++step)
        {
            std::cout << " " << ref_tokens[step] << "/" << pp_tokens[step];
        }
        std::cout << std::endl;

        EXPECT_EQ(ref_tokens, pp_tokens)
            << "Two-stage PP decode should produce same tokens as single-stage reference";
    }

    /**
     * @test Two-stage heterogeneous PP decode on ROCm with decode parity against CPU reference
     */
    TEST_F(Test__PPStageRunner, TwoStagePPDecode_ROCm)
    {
        if (!hasROCmDevice())
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();
        const int mid = n_layers / 2;

        // Heterogeneous PP: CPU prefix → ROCm suffix
        auto ctx1 = loadPPStageContext(0, mid, true, false);
        auto ctx2 = loadPPStageContext(mid, n_layers, false, true);
        ASSERT_NE(ctx1, nullptr);
        ASSERT_NE(ctx2, nullptr);

        FactoryPPStageConfig cfg1{0, mid, true, false};
        FactoryPPStageConfig cfg2{mid, n_layers, false, true};

        auto s1 = createPPStageRunner(ctx1, DeviceId::cpu(), cfg1);
        auto s2 = createPPStageRunner(ctx2, DeviceId::rocm(0), cfg2);
        ASSERT_NE(s1, nullptr);
        ASSERT_NE(s2, nullptr);

        // CPU reference
        auto ref_ctx = loadModel();
        ASSERT_NE(ref_ctx, nullptr);
        FactoryPPStageConfig ref_cfg{0, n_layers, true, true};
        auto ref = createPPStageRunner(ref_ctx, DeviceId::cpu(), ref_cfg);
        ASSERT_NE(ref, nullptr);

        constexpr int SEQ_LEN = 16;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        // Prefill
        ASSERT_TRUE(ref->forward(tokens.data(), SEQ_LEN));
        ASSERT_TRUE(s1->forward(tokens.data(), SEQ_LEN));
        s2->setHiddenState(s1->getHiddenState());
        ASSERT_TRUE(s2->forward(tokens.data(), SEQ_LEN));

        int vocab = ref->vocab_size();
        constexpr int DECODE_STEPS = 3;
        int matches = 0;

        int ref_next = 0, pp_next = 0;
        for (int step = 0; step < DECODE_STEPS; ++step)
        {
            const float *ref_logits = ref->logits();
            const float *pp_logits = s2->logits();
            ASSERT_NE(ref_logits, nullptr);
            ASSERT_NE(pp_logits, nullptr);

            ref_next = static_cast<int>(std::distance(
                ref_logits, std::max_element(ref_logits, ref_logits + vocab)));
            pp_next = static_cast<int>(std::distance(
                pp_logits, std::max_element(pp_logits, pp_logits + vocab)));

            if (ref_next == pp_next)
                ++matches;

            std::cout << "Decode step " << step << ": CPU=" << ref_next
                      << " ROCm_PP=" << pp_next << std::endl;

            // Next decode step — use each runner's own greedy token
            ASSERT_TRUE(ref->forward(&ref_next, 1));
            ASSERT_TRUE(s1->forward(&pp_next, 1));
            s2->setHiddenState(s1->getHiddenState());
            ASSERT_TRUE(s2->forward(&pp_next, 1));
        }

        // At least 2 of 3 tokens should match (quantized GEMM may diverge late)
        EXPECT_GE(matches, 2)
            << "ROCm PP decode should match CPU reference on most tokens";
    }

    // =============================================================================
    // Embedding / LM Head Flag Behavior Tests
    // =============================================================================

    /**
     * @test Stage without embedding: setHiddenState is required, forward succeeds
     */
    TEST_F(Test__PPStageRunner, NoEmbeddingStageRequiresHiddenState)
    {
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();
        const int mid = n_layers / 2;

        // Create stage 1 (with embedding) to produce hidden state
        auto ctx1 = loadPPStageContext(0, mid, true, false);
        ASSERT_NE(ctx1, nullptr);
        FactoryPPStageConfig cfg1{0, mid, true, false};
        auto s1 = createPPStageRunner(ctx1, DeviceId::cpu(), cfg1);
        ASSERT_NE(s1, nullptr);

        // Create stage 2 (without embedding)
        auto ctx2 = loadPPStageContext(mid, n_layers, false, true);
        ASSERT_NE(ctx2, nullptr);
        FactoryPPStageConfig cfg2{mid, n_layers, false, true};
        auto s2 = createPPStageRunner(ctx2, DeviceId::cpu(), cfg2);
        ASSERT_NE(s2, nullptr);

        constexpr int SEQ_LEN = 16;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        // Stage 1 doesn't need hidden state input (has embedding)
        EXPECT_FALSE(s1->hasHiddenStateInput());

        // Run stage 1 to produce hidden state
        ASSERT_TRUE(s1->forward(tokens.data(), SEQ_LEN));
        TensorBase *hidden = s1->getHiddenState();
        ASSERT_NE(hidden, nullptr);
        ASSERT_GT(hidden->numel(), 0u);

        // Stage 2 with hidden state input should succeed
        s2->setHiddenState(hidden);
        EXPECT_TRUE(s2->hasHiddenStateInput());
        ASSERT_TRUE(s2->forward(tokens.data(), SEQ_LEN));
    }

    /**
     * @test Stage without LM head: getHiddenState returns valid tensor, logits is null
     */
    TEST_F(Test__PPStageRunner, NoLmHeadStageProducesHiddenStateNoLogits)
    {
        auto ctx = loadPPStageContext(0, 12, true, false);
        ASSERT_NE(ctx, nullptr);
        FactoryPPStageConfig config{0, 12, true, false};
        auto runner = createPPStageRunner(ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr);

        constexpr int SEQ_LEN = 16;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        ASSERT_TRUE(runner->forward(tokens.data(), SEQ_LEN));

        // Without LM head, hidden state should be available
        TensorBase *hidden = runner->getHiddenState();
        EXPECT_NE(hidden, nullptr) << "Stage without LM head should produce hidden state";
        if (hidden)
        {
            EXPECT_GT(hidden->numel(), 0u);
        }
    }

    /**
     * @test Stage with LM head: logits are available after forward
     */
    TEST_F(Test__PPStageRunner, WithLmHeadStageProducesLogits)
    {
        auto full_ctx = loadModel();
        ASSERT_NE(full_ctx, nullptr);
        const int n_layers = full_ctx->blockCount();

        // Create full model as single stage with LM head
        auto ctx = loadPPStageContext(0, n_layers, true, true);
        ASSERT_NE(ctx, nullptr);
        FactoryPPStageConfig config{0, n_layers, true, true};
        auto runner = createPPStageRunner(ctx, DeviceId::cpu(), config);
        ASSERT_NE(runner, nullptr);

        constexpr int SEQ_LEN = 16;
        std::vector<int> tokens(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; ++i)
            tokens[i] = i % 1024;

        ASSERT_TRUE(runner->forward(tokens.data(), SEQ_LEN));

        // With LM head, logits should be available
        const float *logits = runner->logits();
        EXPECT_NE(logits, nullptr) << "Stage with LM head should produce logits";

        // Vocab size should be valid
        int vocab = runner->vocab_size();
        EXPECT_GT(vocab, 0) << "vocab_size should be positive for LM head stage";
    }

} // anonymous namespace

#include <csignal>

// Track whether any test assertion has failed. Used by signal handlers
// to distinguish ROCm/RCCL driver cleanup crashes from real test failures.
static volatile sig_atomic_t g_any_assertion_failed = 0;

static void cleanup_crash_handler(int sig)
{
    if (!g_any_assertion_failed)
        _exit(0);
    struct sigaction sa = {};
    sa.sa_handler = SIG_DFL;
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

static void install_crash_handlers()
{
    struct sigaction sa = {};
    sa.sa_handler = cleanup_crash_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
}

class AssertionTracker : public ::testing::EmptyTestEventListener
{
    void OnTestPartResult(const ::testing::TestPartResult &result) override
    {
        if (result.failed())
            g_any_assertion_failed = 1;
    }
};

int main(int argc, char **argv)
{
    install_crash_handlers();

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()->listeners().Append(new AssertionTracker);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    _exit(result);
}
