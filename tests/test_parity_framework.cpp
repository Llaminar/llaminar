/**
 * @file test_parity_framework.cpp
 * @brief Parity test framework for comparing Llaminar with llama.cpp
 * @author David Sanftenberg
 * 
 * This test validates the distributed attention pipeline by comparing intermediate
 * tensor snapshots from Llaminar against llama.cpp reference implementation.
 * 
 * The framework is designed to be extensible to other model architectures.
 */

#include "parity_test_framework.h"
#include "distributed_transformer_pipeline.h"
#include "model_loader.h"
#include "logger.h"
#include "test_timeout_guard.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

extern "C"
{
#include "llama.h"
}

using namespace llaminar;
using namespace llaminar::parity;

namespace
{
    constexpr const char* kParityCaptureEnv = "LLAMINAR_PARITY_CAPTURE";
    constexpr const char* kParityCompareEnv = "LLAMINAR_PARITY_COMPARE";

    struct MPIFinalizer
    {
        ~MPIFinalizer()
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (initialized)
            {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized)
                {
                    MPI_Finalize();
                }
            }
        }
    } mpi_finalizer;

    /**
     * @brief Find a suitable test model file
     */
    std::string find_test_model()
    {
        namespace fs = std::filesystem;
        fs::path models_dir{"models"};
        if (!fs::exists(models_dir))
        {
            return {};
        }

        const std::vector<std::string> preferred = {
            "qwen2.5-0.5b-instruct-q4_0.gguf",
            "qwen2.5-0.5b-instruct-fp16.gguf"
        };

        for (const auto& candidate : preferred)
        {
            fs::path path = models_dir / candidate;
            if (fs::exists(path))
            {
                return path.string();
            }
        }

        return {};
    }

    /**
     * @brief Helper to broadcast string across MPI ranks
     */
    void broadcast_string(std::string& value, int root, MPI_Comm comm)
    {
        int length = static_cast<int>(value.size());
        MPI_Bcast(&length, 1, MPI_INT, root, comm);
        
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank != root)
        {
            value.assign(length, '\0');
        }
        
        if (length > 0)
        {
            MPI_Bcast(value.data(), length, MPI_CHAR, root, comm);
        }
    }

    /**
     * @brief RAII guard for llama.cpp context
     */
    struct LlamaContextGuard
    {
        llama_model* model{nullptr};
        llama_context* ctx{nullptr};

        ~LlamaContextGuard()
        {
            if (ctx)
            {
                llama_free(ctx);
            }
            if (model)
            {
                llama_model_free(model);
            }
        }
    };

    /**
     * @brief Custom snapshot hook for Llaminar pipeline
     * 
     * This function can be called from within the pipeline to capture
     * intermediate states for parity testing.
     */
    class ParityTestHook
    {
    public:
        static void capture_embedding(int seq_len, int d_model, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::EMBEDDING, -1, data, seq_len, d_model);
        }

        static void capture_attention_norm(int layer, int seq_len, int d_model, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::ATTENTION_NORM, layer, data, seq_len, d_model);
        }

        static void capture_attention_output(int layer, int seq_len, int d_model, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::ATTENTION_OUTPUT, layer, data, seq_len, d_model);
        }

        static void capture_ffn_norm(int layer, int seq_len, int d_model, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::FFN_NORM, layer, data, seq_len, d_model);
        }

        static void capture_ffn_gate(int layer, int seq_len, int d_ff, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::FFN_GATE, layer, data, seq_len, d_ff);
        }

        static void capture_ffn_output(int layer, int seq_len, int d_model, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::FFN_DOWN, layer, data, seq_len, d_model);
        }

        static void capture_final_norm(int seq_len, int d_model, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::FINAL_NORM, -1, data, seq_len, d_model);
        }

        static void capture_logits(int seq_len, int vocab_size, const float* data)
        {
            if (!LlaminarSnapshotHook::is_enabled()) return;
            LlaminarSnapshotHook::capture(PipelineStage::LM_HEAD, -1, data, seq_len, vocab_size);
        }
    };

} // anonymous namespace

/**
 * @brief Test basic parity framework functionality
 */
TEST(ParityFramework, BasicSnapshotCapture)
{
    SnapshotRegistry& registry = SnapshotRegistry::instance();
    registry.clear();

    // Create a test snapshot
    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};
    
    SnapshotMetadata meta;
    meta.stage_name = "test_stage";
    meta.stage = PipelineStage::CUSTOM;
    meta.layer_index = 0;
    meta.seq_len = 2;
    meta.feature_dim = 2;
    meta.source = "test";

    TensorSnapshot snapshot(meta, test_data.data(), test_data.size());
    
    std::string key = registry.make_key("test", "test_stage", 0);
    registry.register_snapshot(key, snapshot);

    EXPECT_TRUE(registry.has_snapshot(key));
    
    TensorSnapshot retrieved;
    ASSERT_TRUE(registry.get_snapshot(key, retrieved));
    ASSERT_EQ(retrieved.data.size(), test_data.size());
    
    for (size_t i = 0; i < test_data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(retrieved.data[i], test_data[i]);
    }
}

/**
 * @brief Test snapshot comparison
 */
TEST(ParityFramework, SnapshotComparison)
{
    std::vector<float> reference = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> test_exact = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> test_close = {1.001f, 2.001f, 3.001f, 4.001f};
    std::vector<float> test_far = {1.1f, 2.1f, 3.1f, 4.1f};

    SnapshotMetadata meta;
    meta.stage_name = "test";
    meta.seq_len = 2;
    meta.feature_dim = 2;

    TensorSnapshot ref_snap(meta, reference.data(), reference.size());
    TensorSnapshot exact_snap(meta, test_exact.data(), test_exact.size());
    TensorSnapshot close_snap(meta, test_close.data(), test_close.size());
    TensorSnapshot far_snap(meta, test_far.data(), test_far.size());

    // Exact match
    {
        auto result = SnapshotComparator::compare(ref_snap, exact_snap, ComparisonTolerance(1e-3f, 1e-4));
        EXPECT_TRUE(result.passed());
        EXPECT_LT(result.metrics.max_abs_diff, 1e-6f);
    }

    // Close match (within tolerance)
    {
        auto result = SnapshotComparator::compare(ref_snap, close_snap, ComparisonTolerance(1e-2f, 1e-3));
        EXPECT_TRUE(result.passed());
    }

    // Far match (outside tolerance)
    {
        auto result = SnapshotComparator::compare(ref_snap, far_snap, ComparisonTolerance(1e-3f, 1e-4));
        EXPECT_FALSE(result.passed());
        EXPECT_GT(result.metrics.max_abs_diff, 0.09f);
    }
}

/**
 * @brief Main parity test comparing Llaminar pipeline with llama.cpp
 * 
 * This test:
 * 1. Runs llama.cpp inference to get reference outputs
 * 2. Runs Llaminar pipeline with snapshot hooks enabled
 * 3. Compares intermediate states and final logits
 */
TEST(ParityFramework, DistributedPipelineVsLlamaCpp)
{
    int world = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Find model file
    std::string model_path = find_test_model();
    if (rank == 0 && model_path.empty())
    {
        GTEST_SKIP() << "No test model found in models/ directory";
    }
    broadcast_string(model_path, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "[PARITY_TEST] Using model: " << model_path << std::endl;
    }

    // Load model configuration
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load GGUF model: " << model_path;
    TransformerLayerConfig base_config = loader.createLayerConfig();
    
    // Use a small test scenario
    const int test_seq_len = 8;
    const int test_layers = std::min(2, base_config.n_layers);
    
    TransformerLayerConfig config = base_config;
    config.n_layers = test_layers;
    config.max_seq_len = test_seq_len;

    // Test token sequence
    std::vector<int> token_ids(test_seq_len);
    for (int i = 0; i < test_seq_len; ++i)
    {
        token_ids[i] = 100 + i; // Simple test pattern
    }

    const int vocab = config.vocab_size;
    const int64_t total_logits = static_cast<int64_t>(test_seq_len) * static_cast<int64_t>(vocab);
    std::vector<float> llama_logits(total_logits, 0.0f);

    // ========== Run llama.cpp for reference ==========
    if (rank == 0)
    {
        std::cout << "[PARITY_TEST] Running llama.cpp reference..." << std::endl;
        
        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        mparams.use_mmap = false;

        LlamaContextGuard guard;
        guard.model = llama_model_load_from_file(model_path.c_str(), mparams);
        ASSERT_NE(guard.model, nullptr) << "Failed to load llama.cpp model";

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = test_seq_len;
        cparams.n_batch = test_seq_len;
        cparams.n_threads = 4;
        cparams.embeddings = true; // Enable embedding extraction

        guard.ctx = llama_init_from_model(guard.model, cparams);
        ASSERT_NE(guard.ctx, nullptr) << "Failed to initialize llama.cpp context";

        llama_batch batch = llama_batch_init(test_seq_len, 0, 1);
        for (int i = 0; i < test_seq_len; ++i)
        {
            batch.token[i] = token_ids[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        batch.n_tokens = test_seq_len;

        int32_t rc = llama_decode(guard.ctx, batch);
        ASSERT_EQ(rc, 0) << "llama_decode failed";
        llama_synchronize(guard.ctx);

        // Extract logits
        for (int i = 0; i < test_seq_len; ++i)
        {
            float* row = llama_get_logits_ith(guard.ctx, i);
            ASSERT_NE(row, nullptr);
            std::memcpy(llama_logits.data() + static_cast<int64_t>(i) * vocab,
                       row, sizeof(float) * static_cast<size_t>(vocab));
        }

        // Extract pre-LM hidden state as a reference snapshot
        std::vector<float> llama_final_hidden(static_cast<size_t>(test_seq_len) * config.d_model);
        for (int i = 0; i < test_seq_len; ++i)
        {
            float* emb_row = llama_get_embeddings_ith(guard.ctx, i);
            ASSERT_NE(emb_row, nullptr);
            std::memcpy(llama_final_hidden.data() + static_cast<size_t>(i) * config.d_model,
                       emb_row, sizeof(float) * static_cast<size_t>(config.d_model));
        }

        // Register llama.cpp reference snapshots
        SnapshotRegistry& registry = SnapshotRegistry::instance();
        
        SnapshotMetadata final_hidden_meta;
        final_hidden_meta.stage_name = "final_norm";
        final_hidden_meta.stage = PipelineStage::FINAL_NORM;
        final_hidden_meta.layer_index = -1;
        final_hidden_meta.seq_len = test_seq_len;
        final_hidden_meta.feature_dim = config.d_model;
        final_hidden_meta.source = "llama.cpp";
        
        TensorSnapshot final_hidden_snap(final_hidden_meta, llama_final_hidden.data(), llama_final_hidden.size());
        registry.register_snapshot(registry.make_key("llama.cpp", "final_norm", -1), final_hidden_snap);

        SnapshotMetadata logits_meta;
        logits_meta.stage_name = "lm_head";
        logits_meta.stage = PipelineStage::LM_HEAD;
        logits_meta.layer_index = -1;
        logits_meta.seq_len = test_seq_len;
        logits_meta.feature_dim = vocab;
        logits_meta.source = "llama.cpp";
        
        TensorSnapshot logits_snap(logits_meta, llama_logits.data(), llama_logits.size());
        registry.register_snapshot(registry.make_key("llama.cpp", "lm_head", -1), logits_snap);

        llama_batch_free(batch);
        llama_backend_free();
    }

    // Broadcast reference logits to all ranks
    const int broadcast_count = static_cast<int>(total_logits);
    MPI_Bcast(llama_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ========== Run Llaminar pipeline with snapshot capture ==========
    if (rank == 0)
    {
        std::cout << "[PARITY_TEST] Running Llaminar pipeline..." << std::endl;
    }

    // Enable snapshot capture (environment-gated for real tests)
    bool enable_capture = std::getenv(kParityCaptureEnv) != nullptr || rank == 0;
    LlaminarSnapshotHook::set_enabled(enable_capture);

    auto weights = loadModelWeights(loader, config);
    DistributedTransformerPipeline pipeline(config);

    // Enable pre-LM capture for comparison
    setenv("LLAMINAR_PIPELINE_CAPTURE_PRE_LM", "1", 1);

    std::shared_ptr<TensorBase> llaminar_output;
    ASSERT_TRUE(pipeline.execute(token_ids, weights, llaminar_output));

    std::vector<float> llaminar_logits(total_logits, 0.0f);
    if (llaminar_output && llaminar_output->data())
    {
        std::memcpy(llaminar_logits.data(), llaminar_output->data(),
                   sizeof(float) * static_cast<size_t>(total_logits));
    }
    MPI_Bcast(llaminar_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ========== Compare results ==========
    if (rank == 0)
    {
        std::cout << "[PARITY_TEST] Comparing results..." << std::endl;

        // Compare final logits
        auto logits_metrics = SnapshotComparator::compute_metrics(llama_logits, llaminar_logits);
        std::cout << "[PARITY_LOGITS] max_abs=" << logits_metrics.max_abs_diff
                  << " mean_abs=" << logits_metrics.mean_abs_diff
                  << " rel_l2=" << logits_metrics.rel_l2 << std::endl;

        // Tolerance from existing golden test
        constexpr float kMaxAbsTolerance = 2e-3f;
        constexpr double kRelL2Tolerance = 5e-4;

        EXPECT_LT(logits_metrics.max_abs_diff, kMaxAbsTolerance)
            << "Logits max_abs exceeds tolerance";
        EXPECT_LT(logits_metrics.rel_l2, kRelL2Tolerance)
            << "Logits rel_l2 exceeds tolerance";

        if (logits_metrics.max_abs_diff >= kMaxAbsTolerance || logits_metrics.rel_l2 >= kRelL2Tolerance)
        {
            SnapshotComparator::log_top_differences(llama_logits, llaminar_logits, vocab, 10, "logits");
        }

        // Compare pre-LM hidden state if available
        const auto& pre_lm_hidden = DistributedTransformerPipeline::getLastPreLMHidden();
        if (!pre_lm_hidden.empty())
        {
            SnapshotRegistry& registry = SnapshotRegistry::instance();
            TensorSnapshot llama_final_hidden;
            if (registry.get_snapshot(registry.make_key("llama.cpp", "final_norm", -1), llama_final_hidden))
            {
                auto hidden_metrics = SnapshotComparator::compute_metrics(llama_final_hidden.data, pre_lm_hidden);
                std::cout << "[PARITY_FINAL_HIDDEN] max_abs=" << hidden_metrics.max_abs_diff
                          << " mean_abs=" << hidden_metrics.mean_abs_diff
                          << " rel_l2=" << hidden_metrics.rel_l2 << std::endl;

                // Pre-LM hidden typically has tighter tolerances
                EXPECT_LT(hidden_metrics.max_abs_diff, kMaxAbsTolerance);
                EXPECT_LT(hidden_metrics.rel_l2, kRelL2Tolerance);
            }
        }

        std::cout << "[PARITY_TEST] Test complete" << std::endl;
    }
}

int main(int argc, char** argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // MPI cleanup handled by MPIFinalizer
    return result;
}
