#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../ParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "kernels/KernelFactory.h"
#include "loaders/ModelContext.h"
#include "utils/Logger.h"

/**
 * @file Test__Qwen35MoE_LongContext_Parity.cpp
 * @brief Qwen3.5 35B MoE long-context CPU-vs-ROCm parity sweep.
 *
 * Exercises one CPU and one ROCm single-device graph runner over configurable
 * prefill lengths, then drives incremental decode with CPU greedy tokens so
 * both backends receive identical decode input. The test intentionally compares
 * Llaminar CPU against Llaminar ROCm directly rather than using PyTorch
 * snapshots; it is a numeric debugging harness for long-context ROCm drift.
 *
 * Lifecycle: each backend owns a separate ModelContext and IInferenceRunner so
 * prepared weights, KV caches, and device resources cannot interfere across the
 * CPU and ROCm paths. Global backend routers and GPU contexts are shut down
 * before MPI_Finalize in main().
 */

using namespace llaminar2;
using namespace llaminar2::test::parity;

namespace
{
    constexpr const char *kLengthsEnv = "LLAMINAR_QWEN35MOE_SWEEP_LENGTHS";
    constexpr const char *kDecodeStepsEnv = "LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS";
    constexpr const char *kContinueOnFailureEnv = "LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE";
    constexpr const char *kSnapshotGateEnv = "LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOTS";
    constexpr const char *kSnapshotStagesEnv = "LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOT_STAGES";
    constexpr const char *kSnapshotFailEnv = "LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOT_FAIL";
    constexpr const char *kModelPath = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";

    constexpr int kSeqLenMargin = 16;
    constexpr float kLogitsCosineThreshold = 0.98f;
    constexpr float kPrefillLogitsKlThreshold = 0.05f;
    constexpr float kDecodeLogitsKlThreshold = 0.06f;
    constexpr float kSnapshotCosineThreshold = 0.98f;
    constexpr float kSnapshotRelL2Threshold = 0.25f;
    constexpr float kRoutingOverlapThreshold = 1.0f;

    /**
     * @brief Runtime knobs for the sweep.
     */
    struct SweepConfig
    {
        std::vector<int> lengths;                        ///< Prefill lengths to execute in order.
        int decode_steps = 2;                            ///< Greedy CPU-driven decode steps per length.
        int max_seq_len = 4096;                          ///< Runner context size used for both backends.
        bool continue_on_failure = false;                ///< Keep sweeping after failed comparisons when explicitly requested.
        bool snapshots_enabled = false;                  ///< Capture selected decode snapshots only when explicitly requested.
        bool snapshot_fail = false;                      ///< Turn selected snapshot mismatches/missing keys into test failures.
        std::vector<std::string> snapshot_stage_filters; ///< Case-insensitive snapshot key substrings to compare.
    };

    /**
     * @brief Owns one backend's model context and inference runner.
     */
    struct BackendRunner
    {
        std::shared_ptr<ModelContext> model_ctx;
        std::unique_ptr<IInferenceRunner> runner;
    };

    /**
     * @brief One CSV row and assertion unit for a logits comparison.
     */
    struct LogitsComparisonRow
    {
        int length = 0;
        std::string phase;
        int decode_step = -1;
        int driver_token = -1;
        float logits_cosine = 0.0f;
        float logits_kl = 0.0f;
        int cpu_top1 = -1;
        int rocm_top1 = -1;
        float top5_overlap = 0.0f;
        bool cpu_top1_in_rocm_top5 = false;
        std::string worst_selected_stage = "LM_HEAD/logits";
        bool passed = false;
    };

    /**
     * @brief One CSV row for a selected CPU-vs-ROCm decode snapshot comparison.
     */
    struct SnapshotComparisonRow
    {
        int length = 0;
        std::string phase = "decode";
        int decode_step = -1;
        std::string stage_key;
        size_t rows = 0;
        size_t cols = 0;
        size_t elements = 0;
        std::string compare_mode;
        float cosine = std::numeric_limits<float>::quiet_NaN();
        float rel_l2 = std::numeric_limits<float>::quiet_NaN();
        float max_abs_diff = std::numeric_limits<float>::quiet_NaN();
        float rmse = std::numeric_limits<float>::quiet_NaN();
        size_t cpu_nan = 0;
        size_t cpu_inf = 0;
        size_t rocm_nan = 0;
        size_t rocm_inf = 0;
        float routing_overlap = std::numeric_limits<float>::quiet_NaN();
        float routing_top1_match = std::numeric_limits<float>::quiet_NaN();
        bool passed = false;
        std::string missing_reason;
    };

    /// @brief Returns true only for explicit boolean runtime knobs.
    bool envFlagEnabled(const char *name)
    {
        const char *value = std::getenv(name);
        return value != nullptr && std::string(value) == "1";
    }

    /// @brief Trim helper for comma-separated environment variables.
    std::string trim(std::string value)
    {
        auto first = std::find_if(value.begin(), value.end(), [](unsigned char ch)
                                  { return !std::isspace(ch); });
        auto last = std::find_if(value.rbegin(), value.rend(), [](unsigned char ch)
                                 { return !std::isspace(ch); })
                        .base();
        if (first >= last)
            return {};
        return std::string(first, last);
    }

    /**
     * @brief Parse a positive integer with a field-specific error message.
     */
    int parsePositiveInteger(const std::string &text, const std::string &field_name)
    {
        size_t consumed = 0;
        int value = 0;
        try
        {
            value = std::stoi(text, &consumed);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(field_name + " contains a non-integer value: '" + text + "'");
        }

        if (consumed != text.size() || value <= 0)
        {
            throw std::invalid_argument(field_name + " must contain positive integers only: '" + text + "'");
        }
        return value;
    }

    /**
     * @brief Parse the prefill length list from LLAMINAR_QWEN35MOE_SWEEP_LENGTHS.
     */
    std::vector<int> parseSweepLengths()
    {
        const char *value = std::getenv(kLengthsEnv);
        if (value == nullptr || std::string(value).empty())
        {
            return {1, 2, 4, 8, 16, 32, 64, 128};
        }

        std::vector<int> lengths;
        std::stringstream stream(value);
        std::string part;
        while (std::getline(stream, part, ','))
        {
            std::string token = trim(part);
            if (token.empty())
            {
                throw std::invalid_argument(std::string(kLengthsEnv) + " contains an empty entry");
            }
            lengths.push_back(parsePositiveInteger(token, kLengthsEnv));
        }

        if (lengths.empty())
        {
            throw std::invalid_argument(std::string(kLengthsEnv) + " did not provide any lengths");
        }
        return lengths;
    }

    /**
     * @brief Parse decode-step count, defaulting to a short two-step probe.
     */
    int parseDecodeSteps()
    {
        const char *value = std::getenv(kDecodeStepsEnv);
        if (value == nullptr || std::string(value).empty())
        {
            return 2;
        }
        return parsePositiveInteger(trim(value), kDecodeStepsEnv);
    }

    /// @brief Default decode snapshot key substrings used for long-context numeric localization.
    std::vector<std::string> defaultSnapshotStageFilters()
    {
        return {
            "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT",
            "ATTENTION_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FINAL_NORM",
            "LM_HEAD"};
    }

    /// @brief Returns the KL tolerance for a logits row phase.
    float logitsKlThresholdForPhase(const std::string &phase)
    {
        return phase == "decode" ? kDecodeLogitsKlThreshold : kPrefillLogitsKlThreshold;
    }

    /**
     * @brief Parse selected snapshot key substrings from the optional override.
     */
    std::vector<std::string> parseSnapshotStageFilters()
    {
        const char *value = std::getenv(kSnapshotStagesEnv);
        if (value == nullptr || std::string(value).empty())
        {
            return defaultSnapshotStageFilters();
        }

        std::vector<std::string> filters;
        std::stringstream stream(value);
        std::string part;
        while (std::getline(stream, part, ','))
        {
            std::string token = trim(part);
            if (token.empty())
            {
                throw std::invalid_argument(std::string(kSnapshotStagesEnv) + " contains an empty entry");
            }
            filters.push_back(token);
        }

        if (filters.empty())
        {
            throw std::invalid_argument(std::string(kSnapshotStagesEnv) + " did not provide any stage filters");
        }
        return filters;
    }

    /**
     * @brief Build the complete sweep configuration from environment overrides.
     */
    SweepConfig makeSweepConfig()
    {
        SweepConfig config;
        config.lengths = parseSweepLengths();
        config.decode_steps = parseDecodeSteps();
        config.continue_on_failure = envFlagEnabled(kContinueOnFailureEnv);

        // Size KV/cache capacity for the requested sweep instead of reserving
        // the model's full long-context ceiling. ROCm:0 needs the headroom for
        // 1024-token prefill scratch in this single-device 35B parity run.
        const int max_length = *std::max_element(config.lengths.begin(), config.lengths.end());
        config.max_seq_len = max_length + config.decode_steps + kSeqLenMargin;
        config.snapshots_enabled = envFlagEnabled(kSnapshotGateEnv);
        config.snapshot_fail = envFlagEnabled(kSnapshotFailEnv);
        if (config.snapshots_enabled)
        {
            config.snapshot_stage_filters = parseSnapshotStageFilters();
        }
        return config;
    }

    /// @brief Escape a single CSV field while keeping numeric cells unquoted by their callers.
    std::string csvEscape(const std::string &value)
    {
        if (value.find_first_of(",\"\n\r") == std::string::npos)
        {
            return value;
        }

        std::string escaped = "\"";
        for (char ch : value)
        {
            if (ch == '"')
            {
                escaped += "\"\"";
            }
            else
            {
                escaped.push_back(ch);
            }
        }
        escaped.push_back('"');
        return escaped;
    }

    /// @brief Uppercase helper for case-insensitive snapshot key substring matching.
    std::string uppercase(std::string value)
    {
        for (char &ch : value)
        {
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }
        return value;
    }

    /// @brief Return true if a snapshot key contains the configured substring, ignoring case.
    bool keyContainsFilter(const std::string &key, const std::string &filter)
    {
        return uppercase(key).find(uppercase(filter)) != std::string::npos;
    }

    /// @brief Returns true for categorical MoE top-k expert index snapshots.
    bool isRoutingIndicesKey(const std::string &key)
    {
        return keyContainsFilter(key, "MOE_ROUTING_INDICES");
    }

    /// @brief Return the abbreviated git hash used by existing parity CSV paths.
    std::string currentGitHash()
    {
        std::string hash = "unknown";
        FILE *pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
        if (pipe != nullptr)
        {
            char buffer[64];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            {
                hash = buffer;
                while (!hash.empty() && (hash.back() == '\n' || hash.back() == '\r'))
                    hash.pop_back();
            }
            pclose(pipe);
        }
        return hash;
    }

    /// @brief Sanitize a GTest name for use as one path component.
    std::string sanitizePathComponent(const std::string &value)
    {
        std::string sanitized;
        sanitized.reserve(value.size());
        for (char ch : value)
        {
            switch (ch)
            {
            case '/':
            case '\\':
            case ':':
            case '*':
            case '?':
            case '"':
            case '<':
            case '>':
            case '|':
                sanitized.push_back('_');
                break;
            default:
                sanitized.push_back(ch);
                break;
            }
        }
        return sanitized;
    }

    /**
     * @brief Create tests/v2/integration/parity/results/<git-hash>/<test-name>/.
     */
    std::filesystem::path ensureResultsDir()
    {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = "Qwen35MoE_LongContext_CPU_vs_ROCm";
        if (test_info != nullptr)
        {
            test_name = std::string(test_info->test_suite_name()) + "_" + test_info->name();
        }

        // __FILE__ is .../parity/qwen35moe/<file>; parent_path().parent_path()
        // returns the shared parity directory used by ParityTestBase.
        const std::filesystem::path parity_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
        const std::filesystem::path dir = parity_dir / "results" / currentGitHash() / sanitizePathComponent(test_name);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec)
        {
            LOG_WARN("[Qwen35MoE LongContext] Failed to create results directory: " << dir << " (" << ec.message() << ")");
        }
        return dir;
    }

    /**
     * @brief Generate deterministic valid token IDs without depending on Python tokenization.
     */
    std::vector<int> makeSyntheticPromptTokens(int length, int vocab_size)
    {
        if (vocab_size <= 1)
        {
            throw std::invalid_argument("vocab_size must be greater than one");
        }

        std::vector<int> tokens;
        tokens.reserve(static_cast<size_t>(length));
        const std::vector<int> qwen_anchor_tokens = {151644, 872, 198, 785, 264, 1273, 13, 151645, 198};

        uint64_t state = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(length);
        for (int i = 0; i < length; ++i)
        {
            int token = -1;
            if (i < static_cast<int>(qwen_anchor_tokens.size()) && qwen_anchor_tokens[i] < vocab_size)
            {
                token = qwen_anchor_tokens[i];
            }
            else
            {
                // LCG output gives a wide embedding sweep while staying inside
                // the model vocabulary; semantic quality is irrelevant here.
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                token = static_cast<int>((state >> 16) % static_cast<uint64_t>(vocab_size));
            }
            tokens.push_back(token);
        }
        return tokens;
    }

    /// @brief Copy the current host-visible logits row before another forward mutates it.
    std::vector<float> copyLogits(IInferenceRunner &runner, int vocab_size, const std::string &label)
    {
        const float *logits = runner.logits();
        if (logits == nullptr)
        {
            throw std::runtime_error("No logits available after " + label);
        }
        return std::vector<float>(logits, logits + vocab_size);
    }

    /// @brief Return the argmax token for a single logits row.
    int argmaxToken(const std::vector<float> &logits)
    {
        return static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    /**
     * @brief Compare CPU and ROCm logits using the same gates as the CSV output.
     */
    LogitsComparisonRow compareLogits(
        int length,
        const std::string &phase,
        int decode_step,
        int driver_token,
        const std::vector<float> &cpu_logits,
        const std::vector<float> &rocm_logits)
    {
        const size_t vocab_size = cpu_logits.size();
        LogitsComparisonRow row;
        row.length = length;
        row.phase = phase;
        row.decode_step = decode_step;
        row.driver_token = driver_token;

        // Treat CPU as the expected distribution and ROCm as the candidate.
        row.logits_cosine = computeCosineSimilarity(rocm_logits.data(), cpu_logits.data(), vocab_size);
        row.logits_kl = computeKLDivergence(rocm_logits.data(), cpu_logits.data(), vocab_size, vocab_size);
        row.top5_overlap = computeTopKOverlap(rocm_logits.data(), cpu_logits.data(), vocab_size, vocab_size, 5);
        row.cpu_top1 = argmaxToken(cpu_logits);
        row.rocm_top1 = argmaxToken(rocm_logits);

        const float cpu_top1_recall = pytorchTop1InLlaminarTopK(
            rocm_logits.data(), cpu_logits.data(), vocab_size, vocab_size, 5);
        row.cpu_top1_in_rocm_top5 = cpu_top1_recall >= 1.0f - 1e-6f;

        row.passed = std::isfinite(row.logits_cosine) &&
                     std::isfinite(row.logits_kl) &&
                     row.logits_cosine >= kLogitsCosineThreshold &&
                     row.logits_kl <= logitsKlThresholdForPhase(row.phase) &&
                     row.cpu_top1_in_rocm_top5;
        return row;
    }

    /// @brief Emit the CSV header expected by manual long-context triage scripts.
    void writeCsvHeader(std::ofstream &csv)
    {
        csv << "length,phase,decode_step,driver_token,logits_cosine,logits_kl,"
               "cpu_top1,rocm_top1,top5_overlap,cpu_top1_in_rocm_top5,"
               "worst_selected_stage,passed\n";
    }

    /// @brief Append one comparison row and flush so partial failures leave evidence.
    void writeCsvRow(std::ofstream &csv, const LogitsComparisonRow &row)
    {
        csv << row.length << ','
            << row.phase << ','
            << row.decode_step << ','
            << row.driver_token << ','
            << std::setprecision(9) << row.logits_cosine << ','
            << std::setprecision(9) << row.logits_kl << ','
            << row.cpu_top1 << ','
            << row.rocm_top1 << ','
            << std::setprecision(9) << row.top5_overlap << ','
            << (row.cpu_top1_in_rocm_top5 ? 1 : 0) << ','
            << csvEscape(row.worst_selected_stage) << ','
            << (row.passed ? 1 : 0) << '\n';
        csv.flush();
    }

    /// @brief Emit the decode-only snapshot diagnostic CSV header.
    void writeSnapshotCsvHeader(std::ofstream &csv)
    {
        csv << "length,phase,decode_step,stage_key,rows,cols,elements,compare_mode,"
               "cosine,rel_l2,max_abs_diff,rmse,cpu_nan,cpu_inf,rocm_nan,rocm_inf,"
               "routing_overlap,routing_top1_match,passed,missing_reason\n";
    }

    /// @brief Append one snapshot row and flush so interrupted diagnostic runs keep useful data.
    void writeSnapshotCsvRow(std::ofstream &csv, const SnapshotComparisonRow &row)
    {
        csv << row.length << ','
            << csvEscape(row.phase) << ','
            << row.decode_step << ','
            << csvEscape(row.stage_key) << ','
            << row.rows << ','
            << row.cols << ','
            << row.elements << ','
            << csvEscape(row.compare_mode) << ','
            << std::setprecision(9) << row.cosine << ','
            << std::setprecision(9) << row.rel_l2 << ','
            << std::setprecision(9) << row.max_abs_diff << ','
            << std::setprecision(9) << row.rmse << ','
            << row.cpu_nan << ','
            << row.cpu_inf << ','
            << row.rocm_nan << ','
            << row.rocm_inf << ','
            << std::setprecision(9) << row.routing_overlap << ','
            << std::setprecision(9) << row.routing_top1_match << ','
            << (row.passed ? 1 : 0) << ','
            << csvEscape(row.missing_reason) << '\n';
        csv.flush();
    }

    /// @brief Create a failed snapshot row for missing or incompatible selected keys.
    SnapshotComparisonRow makeMissingSnapshotRow(
        int length,
        int decode_step,
        const std::string &stage_key,
        const std::string &compare_mode,
        const std::string &missing_reason)
    {
        SnapshotComparisonRow row;
        row.length = length;
        row.decode_step = decode_step;
        row.stage_key = stage_key;
        row.compare_mode = compare_mode;
        row.missing_reason = missing_reason;
        row.passed = false;
        return row;
    }

    /// @brief Safely multiply rows and cols for snapshot shape validation.
    bool shapeElementCount(size_t rows, size_t cols, size_t &elements)
    {
        if (rows == 0 || cols == 0 || rows > std::numeric_limits<size_t>::max() / cols)
        {
            return false;
        }
        elements = rows * cols;
        return true;
    }

    /**
     * @brief Fill numeric metrics for one aligned CPU-vs-ROCm snapshot span.
     */
    void computeSnapshotMetrics(
        SnapshotComparisonRow &row,
        const float *cpu_data,
        const float *rocm_data,
        size_t count)
    {
        double dot = 0.0;
        double cpu_norm_sq = 0.0;
        double rocm_norm_sq = 0.0;
        double diff_sq = 0.0;
        float max_abs_diff = 0.0f;
        size_t finite_pairs = 0;

        for (size_t i = 0; i < count; ++i)
        {
            const float cpu_value = cpu_data[i];
            const float rocm_value = rocm_data[i];

            if (std::isnan(cpu_value))
                ++row.cpu_nan;
            if (std::isinf(cpu_value))
                ++row.cpu_inf;
            if (std::isnan(rocm_value))
                ++row.rocm_nan;
            if (std::isinf(rocm_value))
                ++row.rocm_inf;

            // Keep the metrics useful when a tensor has a few non-finite values,
            // but mark the row as failed below unless every compared pair is finite.
            if (!std::isfinite(cpu_value) || !std::isfinite(rocm_value))
            {
                continue;
            }

            const double cpu = static_cast<double>(cpu_value);
            const double rocm = static_cast<double>(rocm_value);
            const double diff = rocm - cpu;
            dot += rocm * cpu;
            cpu_norm_sq += cpu * cpu;
            rocm_norm_sq += rocm * rocm;
            diff_sq += diff * diff;
            max_abs_diff = std::max(max_abs_diff, std::abs(rocm_value - cpu_value));
            ++finite_pairs;
        }

        if (finite_pairs == 0)
        {
            row.max_abs_diff = std::numeric_limits<float>::quiet_NaN();
            return;
        }

        row.max_abs_diff = max_abs_diff;
        row.rmse = static_cast<float>(std::sqrt(diff_sq / static_cast<double>(finite_pairs)));

        const double norm_product = std::sqrt(cpu_norm_sq) * std::sqrt(rocm_norm_sq);
        if (norm_product > 1e-30)
        {
            row.cosine = static_cast<float>(dot / norm_product);
        }
        else
        {
            row.cosine = diff_sq <= 1e-30 ? 1.0f : 0.0f;
        }

        if (cpu_norm_sq > 1e-30)
        {
            row.rel_l2 = static_cast<float>(std::sqrt(diff_sq / cpu_norm_sq));
        }
        else
        {
            row.rel_l2 = diff_sq <= 1e-30 ? 0.0f : std::numeric_limits<float>::infinity();
        }

        row.passed = finite_pairs == count &&
                     row.cpu_nan == 0 && row.cpu_inf == 0 &&
                     row.rocm_nan == 0 && row.rocm_inf == 0 &&
                     std::isfinite(row.cosine) && row.cosine >= kSnapshotCosineThreshold &&
                     std::isfinite(row.rel_l2) && row.rel_l2 <= kSnapshotRelL2Threshold;
    }

    /**
     * @brief Compare MoE routing-index snapshots as categorical top-k expert sets.
     *
     * Expert IDs are stored in FP32 snapshot tensors for convenience, but their
     * numerical distance is meaningless. This comparison reports set overlap in
     * the cosine column for backward-compatible sorting and keeps the exact
     * top-1 match in the routing_top1_match column.
     */
    void computeRoutingIndexMetrics(
        SnapshotComparisonRow &row,
        const float *cpu_data,
        const float *rocm_data,
        size_t count)
    {
        std::set<int> cpu_experts;
        std::set<int> rocm_experts;
        float max_id_delta = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            const float cpu_value = cpu_data[i];
            const float rocm_value = rocm_data[i];

            if (std::isnan(cpu_value))
                ++row.cpu_nan;
            if (std::isinf(cpu_value))
                ++row.cpu_inf;
            if (std::isnan(rocm_value))
                ++row.rocm_nan;
            if (std::isinf(rocm_value))
                ++row.rocm_inf;
            if (!std::isfinite(cpu_value) || !std::isfinite(rocm_value))
                continue;

            const int cpu_expert = static_cast<int>(std::lround(cpu_value));
            const int rocm_expert = static_cast<int>(std::lround(rocm_value));
            cpu_experts.insert(cpu_expert);
            rocm_experts.insert(rocm_expert);
            max_id_delta = std::max(max_id_delta, std::abs(static_cast<float>(rocm_expert - cpu_expert)));
        }

        size_t intersection = 0;
        for (int expert : cpu_experts)
        {
            if (rocm_experts.find(expert) != rocm_experts.end())
                ++intersection;
        }

        const size_t denominator = std::max(cpu_experts.size(), rocm_experts.size());
        row.routing_overlap = denominator == 0 ? 0.0f : static_cast<float>(intersection) / static_cast<float>(denominator);
        row.routing_top1_match = count > 0 && std::isfinite(cpu_data[0]) && std::isfinite(rocm_data[0]) &&
                                         static_cast<int>(std::lround(cpu_data[0])) == static_cast<int>(std::lround(rocm_data[0]))
                                     ? 1.0f
                                     : 0.0f;

        row.cosine = row.routing_overlap;
        row.rel_l2 = 1.0f - row.routing_overlap;
        row.max_abs_diff = max_id_delta;
        row.rmse = std::numeric_limits<float>::quiet_NaN();
        row.passed = row.cpu_nan == 0 && row.cpu_inf == 0 && row.rocm_nan == 0 && row.rocm_inf == 0 &&
                     row.routing_overlap >= kRoutingOverlapThreshold && row.routing_top1_match >= 1.0f;
    }

    /**
     * @brief Compare one selected snapshot key, using a last-row fallback for decode shape skew.
     */
    SnapshotComparisonRow compareSnapshotInfo(
        int length,
        int decode_step,
        const std::string &stage_key,
        const SnapshotInfo &cpu_info,
        const SnapshotInfo &rocm_info)
    {
        if (!cpu_info)
        {
            return makeMissingSnapshotRow(length, decode_step, stage_key, "missing", "missing_cpu_snapshot");
        }
        if (!rocm_info)
        {
            return makeMissingSnapshotRow(length, decode_step, stage_key, "missing", "missing_rocm_snapshot");
        }

        size_t cpu_shape_elements = 0;
        size_t rocm_shape_elements = 0;
        const bool cpu_shape_ok = shapeElementCount(cpu_info.rows, cpu_info.cols, cpu_shape_elements) &&
                                  cpu_shape_elements == cpu_info.size;
        const bool rocm_shape_ok = shapeElementCount(rocm_info.rows, rocm_info.cols, rocm_shape_elements) &&
                                   rocm_shape_elements == rocm_info.size;
        if (!cpu_shape_ok || !rocm_shape_ok)
        {
            auto row = makeMissingSnapshotRow(length, decode_step, stage_key, "incompatible", "incompatible_shape");
            row.rows = cpu_info.rows;
            row.cols = cpu_info.cols;
            row.elements = cpu_info.size;
            return row;
        }

        SnapshotComparisonRow row;
        row.length = length;
        row.decode_step = decode_step;
        row.stage_key = stage_key;

        if (cpu_info.rows == rocm_info.rows && cpu_info.cols == rocm_info.cols && cpu_info.size == rocm_info.size)
        {
            row.rows = cpu_info.rows;
            row.cols = cpu_info.cols;
            row.elements = cpu_info.size;
            row.compare_mode = isRoutingIndicesKey(stage_key) ? "routing_indices" : "full";
            if (isRoutingIndicesKey(stage_key))
                computeRoutingIndexMetrics(row, cpu_info.data, rocm_info.data, row.elements);
            else
                computeSnapshotMetrics(row, cpu_info.data, rocm_info.data, row.elements);
            return row;
        }

        if (cpu_info.cols == rocm_info.cols && cpu_info.cols > 0)
        {
            row.rows = 1;
            row.cols = cpu_info.cols;
            row.elements = cpu_info.cols;
            row.compare_mode = isRoutingIndicesKey(stage_key) ? "last_row_routing_indices_shape_mismatch"
                                                              : "last_row_shape_mismatch";

            const float *cpu_last_row = cpu_info.data + ((cpu_info.rows - 1) * cpu_info.cols);
            const float *rocm_last_row = rocm_info.data + ((rocm_info.rows - 1) * rocm_info.cols);
            if (isRoutingIndicesKey(stage_key))
                computeRoutingIndexMetrics(row, cpu_last_row, rocm_last_row, row.elements);
            else
                computeSnapshotMetrics(row, cpu_last_row, rocm_last_row, row.elements);
            return row;
        }

        auto incompatible = makeMissingSnapshotRow(length, decode_step, stage_key, "incompatible", "incompatible_shape");
        incompatible.rows = cpu_info.rows;
        incompatible.cols = cpu_info.cols;
        incompatible.elements = cpu_info.size;
        return incompatible;
    }

    /**
     * @brief Compare all selected snapshot keys present in either backend.
     */
    std::vector<SnapshotComparisonRow> compareSelectedSnapshots(
        int length,
        int decode_step,
        IInferenceRunner &cpu_runner,
        IInferenceRunner &rocm_runner,
        const std::vector<std::string> &stage_filters)
    {
        const std::vector<std::string> cpu_keys = cpu_runner.getSnapshotKeys();
        const std::vector<std::string> rocm_keys = rocm_runner.getSnapshotKeys();
        const std::set<std::string> cpu_key_set(cpu_keys.begin(), cpu_keys.end());
        const std::set<std::string> rocm_key_set(rocm_keys.begin(), rocm_keys.end());

        std::vector<std::string> selected_keys;
        std::set<std::string> selected_key_set;
        std::vector<bool> filter_matched(stage_filters.size(), false);
        auto collect_selected = [&](const std::vector<std::string> &keys)
        {
            for (const std::string &key : keys)
            {
                for (size_t i = 0; i < stage_filters.size(); ++i)
                {
                    if (keyContainsFilter(key, stage_filters[i]))
                    {
                        if (selected_key_set.insert(key).second)
                            selected_keys.push_back(key);
                        filter_matched[i] = true;
                    }
                }
            }
        };
        collect_selected(cpu_keys);
        collect_selected(rocm_keys);

        std::vector<SnapshotComparisonRow> rows;
        rows.reserve(selected_keys.size() + stage_filters.size());
        for (const std::string &key : selected_keys)
        {
            const bool has_cpu = cpu_key_set.find(key) != cpu_key_set.end();
            const bool has_rocm = rocm_key_set.find(key) != rocm_key_set.end();
            if (!has_cpu || !has_rocm)
            {
                rows.push_back(makeMissingSnapshotRow(
                    length,
                    decode_step,
                    key,
                    "missing",
                    has_cpu ? "missing_rocm_snapshot" : "missing_cpu_snapshot"));
                continue;
            }

            rows.push_back(compareSnapshotInfo(
                length,
                decode_step,
                key,
                cpu_runner.getSnapshotWithShape(key),
                rocm_runner.getSnapshotWithShape(key)));
        }

        for (size_t i = 0; i < stage_filters.size(); ++i)
        {
            if (!filter_matched[i])
            {
                rows.push_back(makeMissingSnapshotRow(
                    length,
                    decode_step,
                    stage_filters[i],
                    "missing",
                    "no_matching_snapshot_key"));
            }
        }
        return rows;
    }

    /// @brief Convert a snapshot row into a sortable degradation score for compact summaries.
    float snapshotDegradationScore(const SnapshotComparisonRow &row)
    {
        if (!row.missing_reason.empty())
            return 2.0f;
        if (isRoutingIndicesKey(row.stage_key))
        {
            const float overlap_loss = std::isfinite(row.routing_overlap) ? 1.0f - row.routing_overlap : 1.5f;
            const float top1_loss = std::isfinite(row.routing_top1_match) ? 1.0f - row.routing_top1_match : 1.0f;
            return overlap_loss + top1_loss;
        }
        if (std::isfinite(row.cosine))
            return 1.0f - row.cosine;
        return 1.5f;
    }

    /// @brief Format one snapshot row for the logits CSV worst_selected_stage field.
    std::string compactSnapshotSummary(const SnapshotComparisonRow &row)
    {
        std::ostringstream oss;
        oss << row.stage_key << " mode=" << row.compare_mode;
        if (!row.missing_reason.empty())
        {
            oss << " missing=" << row.missing_reason;
            return oss.str();
        }

        if (isRoutingIndicesKey(row.stage_key))
        {
            oss << " routing_overlap=" << std::setprecision(6) << row.routing_overlap
                << " routing_top1=" << std::setprecision(6) << row.routing_top1_match;
        }
        else
        {
            oss << " cosine=" << std::setprecision(6) << row.cosine
                << " rel_l2=" << std::setprecision(6) << row.rel_l2;
        }
        return oss.str();
    }

    /**
     * @brief Summarize selected decode snapshots for the primary logits CSV.
     *
     * The snapshot CSV has the full row set. This compact summary makes the
     * first failing selected snapshot and the worst selected snapshot visible in
     * the main logits row, which is the file people usually open first after a
     * failing long-context sweep.
     */
    std::string summarizeSnapshotRows(const std::vector<SnapshotComparisonRow> &rows)
    {
        if (rows.empty())
            return "LM_HEAD/logits";

        const SnapshotComparisonRow *first_failing = nullptr;
        const SnapshotComparisonRow *worst = nullptr;
        float worst_score = -std::numeric_limits<float>::infinity();
        for (const SnapshotComparisonRow &row : rows)
        {
            if (!row.passed && first_failing == nullptr)
                first_failing = &row;

            const float score = snapshotDegradationScore(row);
            if (worst == nullptr || score > worst_score)
            {
                worst = &row;
                worst_score = score;
            }
        }

        if (first_failing && worst && first_failing != worst)
            return "first_fail=" + compactSnapshotSummary(*first_failing) + "; worst=" + compactSnapshotSummary(*worst);
        if (first_failing)
            return "first_fail=" + compactSnapshotSummary(*first_failing);
        return worst ? "worst=" + compactSnapshotSummary(*worst) : "LM_HEAD/logits";
    }

    /// @brief Format a compact assertion suffix for one failing phase/length.
    std::string describeRow(const LogitsComparisonRow &row)
    {
        std::ostringstream oss;
        oss << "length=" << row.length
            << " phase=" << row.phase
            << " decode_step=" << row.decode_step
            << " driver_token=" << row.driver_token
            << " cpu_top1=" << row.cpu_top1
            << " rocm_top1=" << row.rocm_top1
            << " cosine=" << row.logits_cosine
            << " kl=" << row.logits_kl
            << " top5_overlap=" << row.top5_overlap;
        return oss.str();
    }

    /// @brief Format a compact assertion suffix for one selected snapshot row.
    std::string describeSnapshotRow(const SnapshotComparisonRow &row)
    {
        std::ostringstream oss;
        oss << "length=" << row.length
            << " phase=" << row.phase
            << " decode_step=" << row.decode_step
            << " stage_key=" << row.stage_key
            << " compare_mode=" << row.compare_mode
            << " elements=" << row.elements
            << " cosine=" << row.cosine
            << " rel_l2=" << row.rel_l2
            << " max_abs_diff=" << row.max_abs_diff
            << " rmse=" << row.rmse
            << " cpu_nan=" << row.cpu_nan
            << " cpu_inf=" << row.cpu_inf
            << " rocm_nan=" << row.rocm_nan
            << " rocm_inf=" << row.rocm_inf
            << " routing_overlap=" << row.routing_overlap
            << " routing_top1_match=" << row.routing_top1_match
            << " missing_reason=" << row.missing_reason;
        return oss.str();
    }
}

/**
 * @brief CPU-vs-ROCm long-context parity fixture for Qwen3.5 MoE.
 */
class Qwen35MoELongContextParityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is not compiled in (HAVE_ROCM=OFF)";
#else
        if (!std::filesystem::exists(kModelPath))
        {
            GTEST_SKIP() << "Model file not found: " << kModelPath;
        }

        DeviceManager::instance().initialize(-1);
        if (DeviceManager::instance().rocm_device_count() <= 0 ||
            !DeviceManager::instance().deviceExists(DeviceId::rocm(0)))
        {
            GTEST_SKIP() << "No ROCm device available. Inventory: "
                         << DeviceManager::instance().availableDevicesString();
        }

        sweep_config_ = makeSweepConfig();
        results_dir_ = ensureResultsDir();
        Logger::getInstance().setLogFile((results_dir_ / "test_log.txt").string());

        // Start from a clean kernel registry so CPU and ROCm runner creation
        // cannot reuse stale prepared handles from a previous test process.
        llaminar::v2::kernels::KernelFactory::clearCache();
        loadBackendRunners();
#endif
    }

    void TearDown() override
    {
        cpu_.runner.reset();
        rocm_.runner.reset();
        cpu_.model_ctx.reset();
        rocm_.model_ctx.reset();

        llaminar::v2::kernels::KernelFactory::clearCache();
        GlobalBackendRouter::shutdown();
        GPUDeviceContextPool::instance().shutdown();
    }

    /**
     * @brief Load one isolated model context and runner for a target device.
     */
    BackendRunner makeBackendRunner(DeviceId device, const std::string &backend_name)
    {
        BackendRunner backend;
        backend.model_ctx = ModelContext::create(
            kModelPath,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        if (!backend.model_ctx)
        {
            throw std::runtime_error("Failed to load Qwen3.5 MoE model for " + backend_name);
        }

        backend.runner = makeRunnerForModel(backend.model_ctx, device, backend_name);
        return backend;
    }

    /// @brief Build the inference runner config shared by both parity backends.
    InferenceRunnerConfig makeRunnerConfig() const
    {
        InferenceRunnerConfig config;
        config.max_seq_len = sweep_config_.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        return config;
    }

    /// @brief Create a fresh runner around an already-loaded model context.
    std::unique_ptr<IInferenceRunner> makeRunnerForModel(
        const std::shared_ptr<ModelContext> &model_ctx,
        DeviceId device,
        const std::string &backend_name) const
    {
        auto runner = createInferenceRunner(model_ctx, nullptr, device, makeRunnerConfig());
        if (!runner)
        {
            throw std::runtime_error("Failed to create " + backend_name + " inference runner");
        }
        return runner;
    }

    /**
     * @brief Construct both backend runners after environment and device checks pass.
     */
    void loadBackendRunners()
    {
        cpu_ = makeBackendRunner(DeviceId::cpu(), "CPU");
        rocm_ = makeBackendRunner(DeviceId::rocm(0), "ROCm");

        ASSERT_GT(cpu_.runner->vocab_size(), 0) << "CPU runner returned an invalid vocab size";
        ASSERT_EQ(cpu_.runner->vocab_size(), rocm_.runner->vocab_size())
            << "CPU and ROCm runners disagree on vocab size";

        LOG_INFO("[Qwen35MoE LongContext] Loaded CPU and ROCm runners; max_seq_len="
                 << sweep_config_.max_seq_len
                 << " decode_steps=" << sweep_config_.decode_steps
                 << " continue_on_failure=" << (sweep_config_.continue_on_failure ? "true" : "false")
                 << " snapshots_enabled=" << (sweep_config_.snapshots_enabled ? "true" : "false")
                 << " snapshot_fail=" << (sweep_config_.snapshot_fail ? "true" : "false"));
    }

    /// @brief Clear recurrent state before each independent prefill length.
    void resetRunnersForLength()
    {
        // Each sweep length is an independent prompt. clear_cache() must fully
        // reset request state while preserving cached graphs and weight/kernel
        // infrastructure; recreating runners here would mask cache-reset bugs.
        cpu_.runner->disableSnapshotCapture();
        rocm_.runner->disableSnapshotCapture();
        cpu_.runner->clear_cache();
        rocm_.runner->clear_cache();
        cpu_.runner->clearSnapshots();
        rocm_.runner->clearSnapshots();
    }

    /// @brief Enable in-memory snapshot capture for decode after prefill is complete.
    void enableDecodeSnapshots()
    {
        cpu_.runner->enableSnapshotCapture();
        rocm_.runner->enableSnapshotCapture();
        cpu_.runner->clearSnapshots();
        rocm_.runner->clearSnapshots();
    }

    /// @brief Assert all numeric gates while preserving the full CSV sweep.
    void expectRowPasses(const LogitsComparisonRow &row)
    {
        EXPECT_TRUE(std::isfinite(row.logits_cosine)) << describeRow(row);
        EXPECT_TRUE(std::isfinite(row.logits_kl)) << describeRow(row);
        EXPECT_GE(row.logits_cosine, kLogitsCosineThreshold)
            << "Logits cosine below threshold; " << describeRow(row);
        EXPECT_LE(row.logits_kl, logitsKlThresholdForPhase(row.phase))
            << "Logits KL above threshold; " << describeRow(row);
        EXPECT_TRUE(row.cpu_top1_in_rocm_top5)
            << "CPU top-1 token is absent from ROCm top-5; " << describeRow(row);
    }

    /// @brief Write selected snapshot rows and optionally make mismatches visible to GTest.
    void writeAndMaybeExpectSnapshotRows(std::ofstream &csv, const std::vector<SnapshotComparisonRow> &rows)
    {
        for (const SnapshotComparisonRow &row : rows)
        {
            writeSnapshotCsvRow(csv, row);
            if (sweep_config_.snapshot_fail)
            {
                EXPECT_TRUE(row.passed) << describeSnapshotRow(row);
            }
        }
    }

    SweepConfig sweep_config_;
    std::filesystem::path results_dir_;
    BackendRunner cpu_;
    BackendRunner rocm_;
};

TEST_F(Qwen35MoELongContextParityTest, CPUVsROCmLogitsSweep)
{
    const int vocab_size = cpu_.runner->vocab_size();
    const auto csv_path = results_dir_ / "cpu_vs_rocm_long_context_sweep.csv";
    std::ofstream csv(csv_path);
    ASSERT_TRUE(csv.is_open()) << "Failed to open CSV results file: " << csv_path;
    writeCsvHeader(csv);

    const auto snapshot_csv_path = results_dir_ / "cpu_vs_rocm_long_context_snapshot_sweep.csv";
    std::ofstream snapshot_csv;
    if (sweep_config_.snapshots_enabled)
    {
        snapshot_csv.open(snapshot_csv_path);
        ASSERT_TRUE(snapshot_csv.is_open()) << "Failed to open snapshot CSV results file: " << snapshot_csv_path;
        writeSnapshotCsvHeader(snapshot_csv);
    }

    for (int length : sweep_config_.lengths)
    {
        SCOPED_TRACE("prefill length " + std::to_string(length));

        resetRunnersForLength();
        const std::vector<int> tokens = makeSyntheticPromptTokens(length, vocab_size);

        // Prefill both backends from identical token IDs and compare the final
        // LM_HEAD/logits row exposed by each runner.
        ASSERT_TRUE(cpu_.runner->forward(tokens.data(), static_cast<int>(tokens.size())))
            << "CPU prefill failed at length " << length;
        const std::vector<float> cpu_prefill_logits = copyLogits(*cpu_.runner, vocab_size, "CPU prefill");

        ASSERT_TRUE(rocm_.runner->forward(tokens.data(), static_cast<int>(tokens.size())))
            << "ROCm prefill failed at length " << length;
        const std::vector<float> rocm_prefill_logits = copyLogits(*rocm_.runner, vocab_size, "ROCm prefill");

        LogitsComparisonRow prefill_row = compareLogits(
            length, "prefill", -1, -1, cpu_prefill_logits, rocm_prefill_logits);
        writeCsvRow(csv, prefill_row);
        expectRowPasses(prefill_row);
        if (!sweep_config_.continue_on_failure)
        {
            ASSERT_TRUE(prefill_row.passed)
                << "Stopping sweep after first failing row; set " << kContinueOnFailureEnv
                << "=1 to continue collecting later lengths. " << describeRow(prefill_row);
        }

        if (sweep_config_.snapshots_enabled)
        {
            enableDecodeSnapshots();
        }

        // The CPU top-1 from the previous step is the only decode input source;
        // ROCm never gets to steer the token stream during this diagnostic.
        int driver_token = prefill_row.cpu_top1;
        for (int step = 0; step < sweep_config_.decode_steps; ++step)
        {
            SCOPED_TRACE("decode step " + std::to_string(step));
            const int token_for_step = driver_token;

            if (sweep_config_.snapshots_enabled)
            {
                cpu_.runner->clearSnapshots();
                rocm_.runner->clearSnapshots();
            }

            ASSERT_TRUE(cpu_.runner->forward(&token_for_step, 1))
                << "CPU decode failed at length " << length << ", step " << step;
            const std::vector<float> cpu_decode_logits = copyLogits(*cpu_.runner, vocab_size, "CPU decode");

            ASSERT_TRUE(rocm_.runner->forward(&token_for_step, 1))
                << "ROCm decode failed at length " << length << ", step " << step;
            const std::vector<float> rocm_decode_logits = copyLogits(*rocm_.runner, vocab_size, "ROCm decode");

            std::vector<SnapshotComparisonRow> snapshot_rows;
            if (sweep_config_.snapshots_enabled)
            {
                snapshot_rows = compareSelectedSnapshots(
                    length,
                    step,
                    *cpu_.runner,
                    *rocm_.runner,
                    sweep_config_.snapshot_stage_filters);
                writeAndMaybeExpectSnapshotRows(snapshot_csv, snapshot_rows);
            }

            LogitsComparisonRow decode_row = compareLogits(
                length, "decode", step, token_for_step, cpu_decode_logits, rocm_decode_logits);
            if (sweep_config_.snapshots_enabled)
            {
                decode_row.worst_selected_stage = summarizeSnapshotRows(snapshot_rows);
            }
            writeCsvRow(csv, decode_row);
            expectRowPasses(decode_row);
            if (!sweep_config_.continue_on_failure)
            {
                ASSERT_TRUE(decode_row.passed)
                    << "Stopping sweep after first failing row; set " << kContinueOnFailureEnv
                    << "=1 to continue collecting later lengths. " << describeRow(decode_row);
            }

            driver_token = decode_row.cpu_top1;
        }
    }

    LOG_INFO("[Qwen35MoE LongContext] CSV results: " << csv_path.string());
    if (sweep_config_.snapshots_enabled)
    {
        LOG_INFO("[Qwen35MoE LongContext] Snapshot CSV results: " << snapshot_csv_path.string());
    }
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();
    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
