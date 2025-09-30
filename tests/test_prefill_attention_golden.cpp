#include "mpi_transformer_pipeline.h"
#include "cosma_prefill_manager.h"
#include "tensors/tensor_factory.h"
#include "model_loader.h"
#include "logger.h"
#include "test_timeout_guard.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>

extern "C"
{
#include "llama.h"
}

using namespace llaminar;

namespace
{
    constexpr const char *kCaptureBaselineEnv = "LLAMINAR_PREFILL_CAPTURE_BASELINE";
    constexpr const char *kCompareBaselineEnv = "LLAMINAR_PREFILL_COMPARE_BASELINE";
    constexpr const char *kAllowSingleRankEnv = "LLAMINAR_PREFILL_ALLOW_SINGLE_RANK";

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
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "qwen2.5-0.5b-instruct-q5_0.gguf",
            "qwen2.5-0.5b-instruct-fp16.gguf"};

        for (const auto &candidate : preferred)
        {
            fs::path path = models_dir / candidate;
            if (fs::exists(path))
            {
                return path.string();
            }
        }

        for (const auto &entry : fs::directory_iterator(models_dir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            auto name = entry.path().filename().string();
            if (name.size() >= 5 && name.substr(name.size() - 5) == ".gguf")
            {
                return entry.path().string();
            }
        }

        return {};
    }

    void broadcast_string(std::string &value, int root, MPI_Comm comm)
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

    struct LlamaContextGuard
    {
        llama_model *model{nullptr};
        llama_context *ctx{nullptr};

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

    struct GoldenScenario
    {
        int seq_len;
        int max_layers;
        const char *name;
    };

    struct TokenPattern
    {
        const char *name;
        std::function<void(std::vector<int> &, int)> populate;
    };

    struct ComparisonMetrics
    {
        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        double rel_l2 = 0.0;
    };

    // Holds llama.cpp pre-LM hidden (embedding+layers+final norm) reference captured on rank 0
    static std::vector<float> g_llama_pre_lm_hidden_ref;

    std::string to_lower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return value;
    }

    void log_top_differences(const std::vector<float> &candidate,
                             const std::vector<float> &baseline,
                             int cols,
                             int top_k,
                             const char *label)
    {
        if (candidate.size() != baseline.size() || candidate.empty() || cols <= 0 || top_k <= 0 || !label)
        {
            return;
        }

        struct DiffEntry
        {
            size_t index;
            float diff;
            float cand_value;
            float base_value;
        };

        struct DiffCompare
        {
            bool operator()(const DiffEntry &lhs, const DiffEntry &rhs) const
            {
                // Min-heap: smallest diff on top so we can keep the largest entries in the container
                return lhs.diff > rhs.diff;
            }
        };

        std::priority_queue<DiffEntry, std::vector<DiffEntry>, DiffCompare> heap;
        for (size_t idx = 0; idx < candidate.size(); ++idx)
        {
            float diff = std::fabs(candidate[idx] - baseline[idx]);
            if (diff <= 0.0f && heap.size() >= static_cast<size_t>(top_k))
            {
                continue;
            }
            DiffEntry entry{idx, diff, candidate[idx], baseline[idx]};
            heap.push(entry);
            if (heap.size() > static_cast<size_t>(top_k))
            {
                heap.pop();
            }
        }

        if (heap.empty())
        {
            return;
        }

        std::vector<DiffEntry> results;
        results.reserve(heap.size());
        while (!heap.empty())
        {
            results.push_back(heap.top());
            heap.pop();
        }
        std::sort(results.begin(), results.end(), [](const DiffEntry &a, const DiffEntry &b)
                  { return a.diff > b.diff; });

        std::ostringstream oss;
        oss << "[DEBUG] top-diff " << label << " count=" << results.size();
        for (const auto &entry : results)
        {
            size_t row = entry.index / static_cast<size_t>(cols);
            size_t col = entry.index % static_cast<size_t>(cols);
            oss << " | idx=" << entry.index
                << " row=" << row
                << " col=" << col
                << " cand=" << entry.cand_value
                << " base=" << entry.base_value
                << " diff=" << entry.diff;
        }
        std::cout << oss.str() << std::endl;
    }

    ComparisonMetrics compute_metrics(const std::vector<float> &candidate, const std::vector<float> &baseline)
    {
        ComparisonMetrics metrics;
        if (candidate.size() != baseline.size() || candidate.empty())
        {
            return metrics;
        }

        double sum_abs = 0.0;
        double diff_sq = 0.0;
        double ref_sq = 0.0;

        for (size_t i = 0; i < candidate.size(); ++i)
        {
            double diff = static_cast<double>(candidate[i]) - static_cast<double>(baseline[i]);
            double ref = static_cast<double>(baseline[i]);
            double abs_diff = std::fabs(diff);

            metrics.max_abs = std::max(metrics.max_abs, static_cast<float>(abs_diff));
            sum_abs += abs_diff;
            diff_sq += diff * diff;
            ref_sq += ref * ref;
        }

        metrics.mean_abs = static_cast<float>(sum_abs / static_cast<double>(candidate.size()));
        metrics.rel_l2 = (ref_sq > 0.0) ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
        return metrics;
    }
} // namespace

TEST(PrefillAttentionGolden, PipelineMatchesLlamaBaseline)
{
    // Defensive: When test output is piped through utilities like `head`, the pipe
    // closes early and OpenMPI processes can receive SIGPIPE while still emitting
    // logs, leading to mpirun abort noise or a PMIx finalize crash. We ignore
    // SIGPIPE here so a truncated consumer doesn't cause the test to abort,
    // preserving the ability to inspect early-layer diagnostics.
    static bool sigpipe_ignored = []()
    {
        struct sigaction sa{};
        sa.sa_handler = SIG_IGN;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGPIPE, &sa, nullptr);
        return true;
    }();
    (void)sigpipe_ignored;
    static MPIFinalizer finalizer;

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }

    int world = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2 && std::getenv(kAllowSingleRankEnv) == nullptr)
    {
        GTEST_SKIP() << "Need >=2 MPI ranks to exercise distributed path";
    }

    const char *initial_capture_value_raw = std::getenv(kCaptureBaselineEnv);
    const char *initial_compare_value_raw = std::getenv(kCompareBaselineEnv);
    const bool baseline_capture_requested_initial = initial_capture_value_raw != nullptr;
    const bool baseline_compare_requested_initial = initial_compare_value_raw != nullptr;
    const std::string baseline_capture_initial_value =
        baseline_capture_requested_initial ? std::string(initial_capture_value_raw) : std::string{};
    const std::string baseline_compare_initial_value =
        baseline_compare_requested_initial ? std::string(initial_compare_value_raw) : std::string{};

    const char *scenario_filter_env = std::getenv("LLAMINAR_PREFILL_GOLDEN_SCENARIO");
    const char *pattern_filter_env = std::getenv("LLAMINAR_PREFILL_GOLDEN_PATTERN");
    const char *mode_filter_env = std::getenv("LLAMINAR_PREFILL_GOLDEN_COSMA_MODE");
    const std::string scenario_filter = scenario_filter_env ? to_lower(scenario_filter_env) : std::string{};
    const std::string pattern_filter = pattern_filter_env ? to_lower(pattern_filter_env) : std::string{};
    const std::string mode_filter = mode_filter_env ? to_lower(mode_filter_env) : std::string{};

    const char *early_break_env = std::getenv("LLAMINAR_GOLDEN_EARLY_BREAK");
    bool early_break = early_break_env != nullptr;
    float early_break_thresh = []()
    {
        if (const char *v = std::getenv("LLAMINAR_GOLDEN_EARLY_BREAK_MAX_ABS"))
        {
            return std::strtof(v, nullptr);
        }
        return 0.01f; // default threshold
    }();

    // New diagnostics: pre-LM hidden comparison & layerwise spike detection
    const bool compare_pre_lm = (std::getenv("LLAMINAR_GOLDEN_COMPARE_PRE_LM") != nullptr);
    const bool layerwise_scan = (std::getenv("LLAMINAR_GOLDEN_LAYERWISE_SCAN") != nullptr);
    double layerwise_spike_abs = 0.0;
    double layerwise_spike_rel = 0.0;
    if (const char *v = std::getenv("LLAMINAR_GOLDEN_LAYERWISE_SPIKE_ABS"))
    {
        layerwise_spike_abs = std::strtod(v, nullptr);
    }
    if (const char *v = std::getenv("LLAMINAR_GOLDEN_LAYERWISE_SPIKE_REL"))
    {
        layerwise_spike_rel = std::strtod(v, nullptr);
    }

    auto timeout = llaminar::test_util::TestTimeoutGuard::ResolveTimeout(
        {"LLAMINAR_TEST_TIMEOUT_MS", "LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS"},
        std::chrono::milliseconds(600000));
    llaminar::test_util::TestTimeoutGuard watchdog(
        "PrefillAttentionGolden.PipelineMatchesLlamaBaseline", timeout);

    std::string model_path;
    if (rank == 0)
    {
        model_path = find_test_model();
    }
    broadcast_string(model_path, 0, MPI_COMM_WORLD);
    if (model_path.empty())
    {
        watchdog.disarm();
        GTEST_SKIP() << "No GGUF model file found in models/ directory";
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load GGUF model: " << model_path;

    TransformerLayerConfig base_config = loader.createLayerConfig();

    const std::vector<GoldenScenario> scenarios = {
        {32, 2, "seq32_layers2"},
        {128, 4, "seq128_layers4"}};

    const std::vector<std::string> cosma_env_modes = {"direct", "replicated"};

    const std::vector<TokenPattern> token_patterns = {
        {"arithmetic_mod", [](std::vector<int> &tokens, int vocab)
         {
             const int safe_vocab = std::max(1, vocab);
             for (size_t i = 0; i < tokens.size(); ++i)
             {
                 tokens[i] = (static_cast<int>(i) * 7) % safe_vocab;
             }
         }},
        {"repeated_mid", [](std::vector<int> &tokens, int vocab)
         {
             const int safe_vocab = std::max(1, vocab);
             const int value = safe_vocab > 1 ? safe_vocab / 2 : 0;
             std::fill(tokens.begin(), tokens.end(), value);
         }},
        {"alternating_pair", [](std::vector<int> &tokens, int vocab)
         {
             const int safe_vocab = std::max(1, vocab);
             const int a = 0;
             const int b = safe_vocab > 1 ? 1 : 0;
             for (size_t i = 0; i < tokens.size(); ++i)
             {
                 tokens[i] = (i % 2 == 0) ? a : b;
             }
         }}};

    constexpr float kPointwiseTolerance = 2e-3f;
    constexpr double kRelL2Tolerance = 5e-4;

    CosmaPrefillManager &manager = CosmaPrefillManager::instance();

    bool abort_outer = false;
    for (const auto &scenario : scenarios)
    {
        if (abort_outer)
            break;
        if (!scenario_filter.empty())
        {
            std::string scenario_name_lower = to_lower(scenario.name);
            if (scenario_name_lower.find(scenario_filter) == std::string::npos)
            {
                continue;
            }
        }
        SCOPED_TRACE(::testing::Message() << "scenario=" << scenario.name);

        if (scenario.seq_len > base_config.max_seq_len)
        {
            ADD_FAILURE() << "Scenario sequence length " << scenario.seq_len
                          << " exceeds model context length " << base_config.max_seq_len;
            continue;
        }

        TransformerLayerConfig config = base_config;
        config.n_layers = std::min(config.n_layers, scenario.max_layers);
        config.max_seq_len = scenario.seq_len;

        if (config.n_layers <= 0)
        {
            ADD_FAILURE() << "No transformer layers available for scenario " << scenario.name;
            continue;
        }

        auto weights = loadModelWeights(loader, config);
        MPITransformerPipeline pipeline(config);
        const int vocab = config.vocab_size;

        for (const auto &pattern : token_patterns)
        {
            if (abort_outer)
                break;
            if (!pattern_filter.empty())
            {
                std::string pattern_name_lower = to_lower(pattern.name);
                if (pattern_name_lower.find(pattern_filter) == std::string::npos)
                {
                    continue;
                }
            }
            SCOPED_TRACE(::testing::Message() << "token_pattern=" << pattern.name);

            std::vector<int> token_ids(scenario.seq_len, 0);
            pattern.populate(token_ids, vocab);

            const int64_t total_logits = static_cast<int64_t>(scenario.seq_len) * static_cast<int64_t>(vocab);
            ASSERT_LE(total_logits, static_cast<int64_t>(std::numeric_limits<int>::max()))
                << "Total logits exceed MPI broadcast capacity";
            const int broadcast_count = static_cast<int>(total_logits);

            std::vector<float> llama_logits(total_logits, 0.0f);

            if (rank == 0)
            {
                llama_backend_init();

                llama_model_params mparams = llama_model_default_params();
                mparams.n_gpu_layers = 0;
                mparams.use_mmap = false;
                mparams.use_mlock = false;
                mparams.check_tensors = false;

                LlamaContextGuard guard;
                guard.model = llama_model_load_from_file(model_path.c_str(), mparams);
                ASSERT_NE(guard.model, nullptr) << "Failed to load llama.cpp model";

                llama_context_params cparams = llama_context_default_params();
                // Force minimal context equal to scenario sequence length to avoid
                // building unnecessarily large (default 512) graphs that stall decode.
                cparams.n_ctx = scenario.seq_len;
                cparams.n_batch = scenario.seq_len;
                cparams.n_ubatch = scenario.seq_len;
                // Using full hardware_concurrency (e.g. 112) inside the devcontainer caused
                // extremely long stalls before the first decode completed (watchdog timeouts).
                // Empirically a small thread count drastically reduces graph planning overhead
                // for these tiny golden scenarios. Allow override via env, else cap to 8.
                unsigned hw = std::thread::hardware_concurrency();
                unsigned desired = 8; // reasonable default
                if (const char *v = std::getenv("LLAMINAR_GOLDEN_BASELINE_THREADS"))
                {
                    long val = std::strtol(v, nullptr, 10);
                    if (val > 0 && val < 256)
                        desired = static_cast<unsigned>(val);
                }
                else
                {
                    // If container presents enormous CPU set, still limit to avoid oversubscription cost
                    if (hw > 0 && hw < desired)
                        desired = hw; // keep below hw if hw is already small
                }
                cparams.n_threads = desired;
                cparams.n_threads_batch = desired;
                cparams.offload_kqv = false;
                cparams.no_perf = true;
                // Enable embeddings if we want a pre-LM hidden reference
                cparams.embeddings = compare_pre_lm;

                guard.ctx = llama_init_from_model(guard.model, cparams);
                ASSERT_NE(guard.ctx, nullptr) << "Failed to initialize llama.cpp context";
                llama_set_n_threads(guard.ctx, cparams.n_threads, cparams.n_threads_batch);
                if (compare_pre_lm)
                {
                    std::cout << "[BASELINE_CFG] threads=" << cparams.n_threads
                              << " seq_len=" << scenario.seq_len
                              << " layers=" << config.n_layers
                              << " vocab=" << vocab << std::endl;
                }

                llama_batch batch = llama_batch_init(scenario.seq_len, 0, 1);
                for (int i = 0; i < scenario.seq_len; ++i)
                {
                    batch.token[i] = token_ids[i];
                    batch.pos[i] = i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = 1;
                }
                batch.n_tokens = scenario.seq_len;

                int32_t rc = llama_decode(guard.ctx, batch);
                ASSERT_EQ(rc, 0) << "llama_decode failed";
                llama_synchronize(guard.ctx);

                for (int i = 0; i < scenario.seq_len; ++i)
                {
                    float *row = llama_get_logits_ith(guard.ctx, i);
                    ASSERT_NE(row, nullptr) << "Missing logits for token index " << i;
                    std::memcpy(llama_logits.data() + static_cast<int64_t>(i) * vocab,
                                row,
                                sizeof(float) * static_cast<size_t>(vocab));
                }

                // Capture reference pre-LM hidden state (embeddings) on rank 0 only (no broadcast)
                if (compare_pre_lm && rank == 0)
                {
                    const int d_model = config.d_model;
                    g_llama_pre_lm_hidden_ref.resize(static_cast<size_t>(scenario.seq_len) * d_model);
                    for (int i = 0; i < scenario.seq_len; ++i)
                    {
                        float *emb_row = llama_get_embeddings_ith(guard.ctx, i);
                        ASSERT_NE(emb_row, nullptr) << "Missing embedding row for token index " << i;
                        std::memcpy(g_llama_pre_lm_hidden_ref.data() + static_cast<size_t>(i) * d_model,
                                    emb_row,
                                    sizeof(float) * static_cast<size_t>(d_model));
                    }
                }
                // Synchronize after optional capture to keep later collectives aligned
                if (compare_pre_lm)
                    MPI_Barrier(MPI_COMM_WORLD);

                llama_batch_free(batch);
                llama_backend_free();
            }

            MPI_Bcast(llama_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

            manager.reset_stats();
            manager.set_force_cosma(false);
            manager.set_threshold(std::numeric_limits<int>::max());

            std::shared_ptr<TensorBase> openblas_output;
            if (compare_pre_lm)
            {
                // Ensure the pipeline captures pre-LM hidden state
                setenv("LLAMINAR_PIPELINE_CAPTURE_PRE_LM", "1", 1);
            }
            ASSERT_TRUE(pipeline.execute(token_ids, weights, openblas_output));

            // Fetch layerwise stats for spike detection (must be after execute)
            if (layerwise_scan && rank == 0)
            {
                const auto &stats = MPITransformerPipeline::getLastLayerActivationStats();
                // Simple heuristic: compute per-layer deltas vs previous layer
                for (size_t i = 1; i < stats.size(); ++i)
                {
                    const auto &prev = stats[i - 1];
                    const auto &cur = stats[i];
                    double abs_delta = cur.max_abs - prev.max_abs;
                    double rel_delta = (prev.rms > 0.0) ? (cur.rms / prev.rms) - 1.0 : 0.0;
                    if ((layerwise_spike_abs > 0.0 && abs_delta > layerwise_spike_abs) || (layerwise_spike_rel > 0.0 && rel_delta > layerwise_spike_rel))
                    {
                        std::cout << "[LAYERWISE_SPIKE] layer=" << cur.layer << " prev_layer=" << prev.layer
                                  << " prev_rms=" << prev.rms << " cur_rms=" << cur.rms
                                  << " prev_max=" << prev.max_abs << " cur_max=" << cur.max_abs
                                  << " abs_delta=" << abs_delta << " rel_rms_delta=" << rel_delta << std::endl;
                        if (std::getenv("LLAMINAR_GOLDEN_LAYERWISE_SPIKE_FAIL"))
                        {
                            ADD_FAILURE() << "Layerwise spike detected at layer=" << cur.layer
                                          << " abs_delta=" << abs_delta << " rel_delta=" << rel_delta;
                            return; // Hard fail
                        }
                        break; // report first spike
                    }
                }
            }

            // True elementwise pre-LM hidden comparison (post final norm, before LM head) rank 0 only
            if (compare_pre_lm && rank == 0)
            {
                const auto &pre_hidden = MPITransformerPipeline::getLastPreLMHidden();
                if (pre_hidden.empty())
                {
                    std::cout << "[PRE_LM_HIDDEN] capture empty; skipping diff" << std::endl;
                }
                else if (g_llama_pre_lm_hidden_ref.empty())
                {
                    std::cout << "[PRE_LM_HIDDEN] reference empty (unexpected) size=" << pre_hidden.size() << std::endl;
                }
                else if (g_llama_pre_lm_hidden_ref.size() != pre_hidden.size())
                {
                    std::cout << "[PRE_LM_HIDDEN] size mismatch ours=" << pre_hidden.size() << " ref=" << g_llama_pre_lm_hidden_ref.size() << std::endl;
                }
                else
                {
                    int d_model = config.d_model;
                    auto pre_metrics = compute_metrics(pre_hidden, g_llama_pre_lm_hidden_ref);
                    std::cout << "[PRE_LM_DIFF] max_abs=" << pre_metrics.max_abs
                              << " mean_abs=" << pre_metrics.mean_abs
                              << " rel_l2=" << pre_metrics.rel_l2
                              << " elems=" << pre_hidden.size() << std::endl;
                    float pre_pointwise_tol = 2e-3f;
                    double pre_rel_l2_tol = 5e-4;
                    if (const char *v = std::getenv("LLAMINAR_GOLDEN_PRE_LM_MAX_ABS"))
                        pre_pointwise_tol = std::strtof(v, nullptr);
                    if (const char *v = std::getenv("LLAMINAR_GOLDEN_PRE_LM_REL_L2"))
                        pre_rel_l2_tol = std::strtod(v, nullptr);
                    bool pre_fail = false;
                    if (pre_metrics.max_abs > pre_pointwise_tol || pre_metrics.rel_l2 > pre_rel_l2_tol)
                    {
                        log_top_differences(pre_hidden, g_llama_pre_lm_hidden_ref, d_model, 10, "pre-lm-hidden-vs-llama");
                        if (std::getenv("LLAMINAR_GOLDEN_PRE_LM_FAIL"))
                            pre_fail = true;
                    }
                    if (pre_fail)
                    {
                        ADD_FAILURE() << "Pre-LM hidden divergence: max_abs=" << pre_metrics.max_abs
                                      << " rel_l2=" << pre_metrics.rel_l2
                                      << " tol_abs=" << pre_pointwise_tol << " tol_rel_l2=" << pre_rel_l2_tol;
                        return; // Hard fail early
                    }
                }
            }

            // Optional RMSNorm/QKV parity capture for first layer only (env gated)
            if (std::getenv("LLAMINAR_PREFILL_RMS_PARITY") != nullptr && rank == 0)
            {
                std::cout << "[GoldenRmsParity] ModelWeights structure does not expose per-layer gamma here; parity hook placeholder active." << std::endl;
            }

            std::vector<float> openblas_logits(total_logits, 0.0f);
            if (openblas_output && openblas_output->data())
            {
                std::memcpy(openblas_logits.data(), openblas_output->data(),
                            sizeof(float) * static_cast<size_t>(total_logits));
            }
            MPI_Bcast(openblas_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

            // Determine if we are in a forced fallback mode (skip strict assertions)
            const bool forced_fallback = std::getenv("LLAMINAR_COSMA_FORCE_FALLBACK") != nullptr ||
                                         std::getenv("ADAPTIVE_DISABLE_COSMA") != nullptr;

            const auto openblas_metrics = compute_metrics(openblas_logits, llama_logits);
            // Early-break (MPI coordinated) BEFORE assertions if enabled
            if (early_break)
            {
                int break_signal = 0;
                size_t offending_index = std::numeric_limits<size_t>::max();
                if (rank == 0 && openblas_metrics.max_abs > early_break_thresh)
                {
                    std::cout << "[EARLY_BREAK] Triggered after OpenBLAS path: max_abs=" << openblas_metrics.max_abs
                              << " rel_l2=" << openblas_metrics.rel_l2 << " threshold=" << early_break_thresh << std::endl;
                    for (size_t i = 0; i < llama_logits.size(); ++i)
                    {
                        float d = std::fabs(openblas_logits[i] - llama_logits[i]);
                        if (d > early_break_thresh)
                        {
                            offending_index = i;
                            break;
                        }
                    }
                    break_signal = 1;
                }
                MPI_Bcast(&break_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (break_signal)
                {
                    if (rank != 0)
                    {
                        // Rank 0 already computed offending_index; broadcast it
                        MPI_Bcast(&offending_index, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
                    }
                    else
                    {
                        MPI_Bcast(&offending_index, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
                        if (offending_index != std::numeric_limits<size_t>::max())
                        {
                            const int window = 8;
                            const size_t start = (offending_index > (size_t)window) ? offending_index - window : 0;
                            const size_t end = std::min(offending_index + window + 1, llama_logits.size());
                            auto stats = [&](const std::vector<float> &v)
                            {
                                long double sum=0, sum2=0; size_t n=end-start; for(size_t i=start;i<end;++i){ long double x=v[i]; sum+=x; sum2+=x*x; }
                                long double mean = n? sum/n : 0; long double var = n? (sum2/n - mean*mean) : 0; if (var < 0) var=0; return std::make_pair((double)mean, (double)std::sqrt(var)); };
                            auto ref_stats = stats(llama_logits);
                            auto ours_stats = stats(openblas_logits);
                            std::cout << "[EARLY_BREAK] offending_index=" << offending_index
                                      << " window_start=" << start << " window_end=" << end
                                      << " ref_mean=" << ref_stats.first << " ref_std=" << ref_stats.second
                                      << " ours_mean=" << ours_stats.first << " ours_std=" << ours_stats.second << std::endl;
                            std::cout << "[EARLY_BREAK] window_ref:";
                            for (size_t i = start; i < end; ++i)
                                std::cout << ' ' << llama_logits[i];
                            std::cout << "\n[EARLY_BREAK] window_ours:";
                            for (size_t i = start; i < end; ++i)
                                std::cout << ' ' << openblas_logits[i];
                            std::cout << "\n[EARLY_BREAK] window_abs_diff:";
                            for (size_t i = start; i < end; ++i)
                                std::cout << ' ' << std::fabs(openblas_logits[i] - llama_logits[i]);
                            std::cout << std::endl;
                            if (std::getenv("LLAMINAR_GOLDEN_EARLY_BREAK_DUMP") != nullptr)
                            {
                                std::ofstream jf("early_break_openblas.json");
                                if (jf.good())
                                {
                                    const size_t start = (offending_index > 8) ? offending_index - 8 : 0;
                                    const size_t end2 = std::min(offending_index + 9, llama_logits.size());
                                    jf << "{\n";
                                    jf << "  \"phase\": \"openblas\",\n";
                                    jf << "  \"offending_index\": " << offending_index << ",\n";
                                    jf << "  \"threshold\": " << early_break_thresh << ",\n";
                                    jf << "  \"max_abs\": " << openblas_metrics.max_abs << ",\n";
                                    jf << "  \"rel_l2\": " << openblas_metrics.rel_l2 << ",\n";
                                    jf << "  \"window_start\": " << start << ",\n";
                                    jf << "  \"window_end\": " << end2 << ",\n";
                                    jf << "  \"ref_values\": [";
                                    for (size_t i = start; i < end2; ++i)
                                    {
                                        jf << llama_logits[i];
                                        if (i + 1 < end2)
                                            jf << ',';
                                    }
                                    jf << "],\n";
                                    jf << "  \"ours_values\": [";
                                    for (size_t i = start; i < end2; ++i)
                                    {
                                        jf << openblas_logits[i];
                                        if (i + 1 < end2)
                                            jf << ',';
                                    }
                                    jf << "],\n";
                                    jf << "  \"abs_diff\": [";
                                    for (size_t i = start; i < end2; ++i)
                                    {
                                        jf << std::fabs(openblas_logits[i] - llama_logits[i]);
                                        if (i + 1 < end2)
                                            jf << ',';
                                    }
                                    jf << "]\n}";
                                }
                            }
                        }
                    }
                    ADD_FAILURE() << "Early break (OpenBLAS) triggered at index=" << offending_index
                                  << ": max_abs=" << openblas_metrics.max_abs
                                  << " rel_l2=" << openblas_metrics.rel_l2
                                  << " threshold=" << early_break_thresh;
                    return; // hard fail exit
                }
                else
                {
                    if (rank == 0 && (openblas_metrics.max_abs >= kPointwiseTolerance || openblas_metrics.rel_l2 >= kRelL2Tolerance))
                    {
                        log_top_differences(openblas_logits, llama_logits, vocab, 10, "openblas-vs-llama");
                    }
                    if (!forced_fallback)
                    {
                        EXPECT_LT(openblas_metrics.max_abs, kPointwiseTolerance) << "OpenBLAS max abs diff exceeds tolerance";
                        EXPECT_LT(openblas_metrics.rel_l2, kRelL2Tolerance) << "OpenBLAS relative L2 drift too large";
                    }
                }
            }
            else
            {
                if (rank == 0 && (openblas_metrics.max_abs >= kPointwiseTolerance || openblas_metrics.rel_l2 >= kRelL2Tolerance))
                {
                    log_top_differences(openblas_logits, llama_logits, vocab, 10, "openblas-vs-llama");
                }
                if (!forced_fallback)
                {
                    EXPECT_LT(openblas_metrics.max_abs, kPointwiseTolerance) << "OpenBLAS max abs diff exceeds tolerance";
                    EXPECT_LT(openblas_metrics.rel_l2, kRelL2Tolerance) << "OpenBLAS relative L2 drift too large";
                }
            }
            // When a global fallback/disable flag forces COSMA path off (e.g. LLAMINAR_COSMA_FORCE_FALLBACK or ADAPTIVE_DISABLE_COSMA),
            // we still want to ensure no COSMA usage; otherwise keep the strict assertion.
            if (!forced_fallback)
            {
                EXPECT_EQ(manager.stats().cosma_path_calls.load(), 0u)
                    << "COSMA path should not have been used during OpenBLAS comparison";
            }

            if (std::getenv(kCaptureBaselineEnv) != nullptr && std::getenv(kCompareBaselineEnv) == nullptr)
            {
                // Transition capture-only runs into compare mode for subsequent COSMA executions.
                MPI_Barrier(MPI_COMM_WORLD);
                unsetenv(kCaptureBaselineEnv);
                setenv(kCompareBaselineEnv, "1", 1);
                MPI_Barrier(MPI_COMM_WORLD);
            }

            for (const auto &mode : cosma_env_modes)
            {
                if (abort_outer)
                    break;
                if (!mode_filter.empty())
                {
                    std::string mode_lower = to_lower(mode);
                    if (mode_lower.find(mode_filter) == std::string::npos)
                    {
                        continue;
                    }
                }
                SCOPED_TRACE(::testing::Message() << "cosma_mode=" << mode);

                bool restore_direct = false;
                bool restore_replicated = false;
                std::string prev_force_direct;
                std::string prev_force_replicated;

                if (const char *existing = std::getenv("LLAMINAR_COSMA_FORCE_DIRECT"))
                {
                    restore_direct = true;
                    prev_force_direct = existing;
                }
                if (const char *existing = std::getenv("LLAMINAR_COSMA_FORCE_REPLICATED"))
                {
                    restore_replicated = true;
                    prev_force_replicated = existing;
                }

                if (mode == "direct")
                {
                    setenv("LLAMINAR_COSMA_FORCE_DIRECT", "1", 1);
                    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
                }
                else if (mode == "replicated")
                {
                    setenv("LLAMINAR_COSMA_FORCE_REPLICATED", "1", 1);
                    unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
                }

                manager.reset_stats();
                manager.set_threshold(1);
                manager.set_force_cosma(true);

                std::shared_ptr<TensorBase> cosma_output;
                ASSERT_TRUE(pipeline.execute(token_ids, weights, cosma_output));

                std::vector<float> cosma_logits(total_logits, 0.0f);
                if (cosma_output && cosma_output->data())
                {
                    std::memcpy(cosma_logits.data(), cosma_output->data(),
                                sizeof(float) * static_cast<size_t>(total_logits));
                }
                MPI_Bcast(cosma_logits.data(), broadcast_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

                const auto cosma_metrics = compute_metrics(cosma_logits, llama_logits);
                const auto cosma_vs_open_metrics = compute_metrics(cosma_logits, openblas_logits);
                if (rank == 0)
                {
                    std::cout << "[DEBUG] cosma vs llama max_abs=" << cosma_metrics.max_abs
                              << " rel_l2=" << cosma_metrics.rel_l2
                              << " | cosma vs open max_abs=" << cosma_vs_open_metrics.max_abs
                              << " rel_l2=" << cosma_vs_open_metrics.rel_l2 << std::endl;
                    if (cosma_metrics.max_abs >= kPointwiseTolerance ||
                        cosma_metrics.rel_l2 >= kRelL2Tolerance)
                    {
                        log_top_differences(cosma_logits, llama_logits, vocab, 10, "cosma-vs-llama");
                    }
                    if (cosma_vs_open_metrics.max_abs >= kPointwiseTolerance ||
                        cosma_vs_open_metrics.rel_l2 >= kRelL2Tolerance)
                    {
                        log_top_differences(cosma_logits, openblas_logits, vocab, 10, "cosma-vs-openblas");
                    }
                }
                if (!forced_fallback)
                {
                    EXPECT_LT(cosma_metrics.max_abs, kPointwiseTolerance)
                        << "COSMA max abs diff exceeds tolerance";
                    EXPECT_LT(cosma_metrics.rel_l2, kRelL2Tolerance)
                        << "COSMA relative L2 drift too large";
                }
                if (early_break)
                {
                    int break_signal = 0;
                    size_t offending_index = std::numeric_limits<size_t>::max();
                    if (rank == 0 && cosma_metrics.max_abs > early_break_thresh)
                    {
                        std::cout << "[EARLY_BREAK] Triggered after COSMA path: mode=" << mode
                                  << " max_abs=" << cosma_metrics.max_abs << " rel_l2=" << cosma_metrics.rel_l2
                                  << " threshold=" << early_break_thresh << std::endl;
                        for (size_t i = 0; i < llama_logits.size(); ++i)
                        {
                            float d = std::fabs(cosma_logits[i] - llama_logits[i]);
                            if (d > early_break_thresh)
                            {
                                offending_index = i;
                                break;
                            }
                        }
                        break_signal = 1;
                    }
                    MPI_Bcast(&break_signal, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (break_signal)
                    {
                        if (rank != 0)
                        {
                            MPI_Bcast(&offending_index, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
                        }
                        else
                        {
                            MPI_Bcast(&offending_index, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
                            if (offending_index != std::numeric_limits<size_t>::max())
                            {
                                const int window = 8;
                                const size_t start = (offending_index > (size_t)window) ? offending_index - window : 0;
                                const size_t end = std::min(offending_index + window + 1, llama_logits.size());
                                auto stats = [&](const std::vector<float> &v)
                                {
                                    long double sum=0, sum2=0; size_t n=end-start; for(size_t i=start;i<end;++i){ long double x=v[i]; sum+=x; sum2+=x*x; }
                                    long double mean = n? sum/n : 0; long double var = n? (sum2/n - mean*mean) : 0; if (var < 0) var=0; return std::make_pair((double)mean, (double)std::sqrt(var)); };
                                auto ref_stats = stats(llama_logits);
                                auto ours_stats = stats(cosma_logits);
                                std::cout << "[EARLY_BREAK] offending_index=" << offending_index
                                          << " window_start=" << start << " window_end=" << end
                                          << " ref_mean=" << ref_stats.first << " ref_std=" << ref_stats.second
                                          << " ours_mean=" << ours_stats.first << " ours_std=" << ours_stats.second << std::endl;
                                std::cout << "[EARLY_BREAK] window_ref:";
                                for (size_t i = start; i < end; ++i)
                                    std::cout << ' ' << llama_logits[i];
                                std::cout << "\n[EARLY_BREAK] window_ours:";
                                for (size_t i = start; i < end; ++i)
                                    std::cout << ' ' << cosma_logits[i];
                                std::cout << "\n[EARLY_BREAK] window_abs_diff:";
                                for (size_t i = start; i < end; ++i)
                                    std::cout << ' ' << std::fabs(cosma_logits[i] - llama_logits[i]);
                                std::cout << std::endl;
                                if (std::getenv("LLAMINAR_GOLDEN_EARLY_BREAK_DUMP") != nullptr)
                                {
                                    std::ofstream jf("early_break_cosma.json");
                                    if (jf.good())
                                    {
                                        const size_t start2 = (offending_index > 8) ? offending_index - 8 : 0;
                                        const size_t end2 = std::min(offending_index + 9, llama_logits.size());
                                        jf << "{\n";
                                        jf << "  \"phase\": \"cosma\",\n";
                                        jf << "  \"mode\": \"" << mode << "\",\n";
                                        jf << "  \"offending_index\": " << offending_index << ",\n";
                                        jf << "  \"threshold\": " << early_break_thresh << ",\n";
                                        jf << "  \"max_abs\": " << cosma_metrics.max_abs << ",\n";
                                        jf << "  \"rel_l2\": " << cosma_metrics.rel_l2 << ",\n";
                                        jf << "  \"window_start\": " << start2 << ",\n";
                                        jf << "  \"window_end\": " << end2 << ",\n";
                                        jf << "  \"ref_values\": [";
                                        for (size_t i = start2; i < end2; ++i)
                                        {
                                            jf << llama_logits[i];
                                            if (i + 1 < end2)
                                                jf << ',';
                                        }
                                        jf << "],\n";
                                        jf << "  \"ours_values\": [";
                                        for (size_t i = start2; i < end2; ++i)
                                        {
                                            jf << cosma_logits[i];
                                            if (i + 1 < end2)
                                                jf << ',';
                                        }
                                        jf << "],\n";
                                        jf << "  \"abs_diff\": [";
                                        for (size_t i = start2; i < end2; ++i)
                                        {
                                            jf << std::fabs(cosma_logits[i] - llama_logits[i]);
                                            if (i + 1 < end2)
                                                jf << ',';
                                        }
                                        jf << "]\n}";
                                    }
                                }
                            }
                        }
                        ADD_FAILURE() << "Early break (COSMA) triggered at index=" << offending_index
                                      << " mode=" << mode
                                      << ": max_abs=" << cosma_metrics.max_abs
                                      << " rel_l2=" << cosma_metrics.rel_l2
                                      << " threshold=" << early_break_thresh;
                        return; // hard fail exit
                    }
                }

                // Skip COSMA usage assertions entirely if we are in an environment explicitly forcing fallback
                // (enables golden test reuse for pipeline-local debugging where distributed GEMM is disabled).
                if (!forced_fallback)
                {
                    EXPECT_GT(manager.stats().cosma_path_calls.load(), 0u)
                        << "COSMA path should have been invoked";
                    EXPECT_EQ(manager.stats().fast_path_calls.load(), 0u)
                        << "Fast path should not trigger when COSMA forced";
                }

                if (restore_direct)
                {
                    setenv("LLAMINAR_COSMA_FORCE_DIRECT", prev_force_direct.c_str(), 1);
                }
                else
                {
                    unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
                }
                if (restore_replicated)
                {
                    setenv("LLAMINAR_COSMA_FORCE_REPLICATED", prev_force_replicated.c_str(), 1);
                }
                else
                {
                    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
                }
            }

            manager.set_force_cosma(false);
            manager.set_threshold(4096);
            manager.reset_stats();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (baseline_capture_requested_initial)
    {
        setenv(kCaptureBaselineEnv, baseline_capture_initial_value.c_str(), 1);
    }
    else
    {
        unsetenv(kCaptureBaselineEnv);
    }
    if (baseline_compare_requested_initial)
    {
        setenv(kCompareBaselineEnv, baseline_compare_initial_value.c_str(), 1);
    }
    else
    {
        unsetenv(kCompareBaselineEnv);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    watchdog.disarm();
}
