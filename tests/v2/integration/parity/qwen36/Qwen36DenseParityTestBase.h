#pragma once

#include <cnpy.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "backends/ComputeBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/mtp/MTPDecodeCatchup.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "kernels/KernelFactory.h"
#include "loaders/ModelContext.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Sampler.h"
#include "utils/Tokenizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef HAVE_CUDA
extern "C"
{
    void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled);
    bool cudaNativeVNNIPrefill_getDeterministicMode();
}
#endif

namespace llaminar2::test::parity::qwen36
{
    enum class DensePrefixParityTopology
    {
        SingleDevice,
        LocalTP,
        LocalPP,
        NodeLocalTP,
    };

    enum class PrefixRestoreParityMode
    {
        FullHit,
        PartialHit,
    };

    struct DensePrefixRestoreParityCase
    {
        std::string name;
        DensePrefixParityTopology topology = DensePrefixParityTopology::SingleDevice;
        std::vector<GlobalDeviceAddress> devices;
        std::vector<std::string> model_envs;
        std::string default_model_path;
        std::vector<std::string> metadata_envs;
        std::string default_metadata_path;
        std::string prompt = "The quick brown fox jumps over the lazy dog";
        std::string kv_cache_precision = "auto";
        int decode_steps = 3;
        int max_seq_len = 96;
        int main_layers = 64;
        int mpi_ranks = 1;
        int required_cuda_devices = 0;
        int required_rocm_devices = 0;
    };

    inline std::string shellQuote(const std::string &value)
    {
        std::string quoted = "'";
        for (char ch : value)
        {
            if (ch == '\'')
            {
                quoted += "'\\''";
            }
            else
            {
                quoted += ch;
            }
        }
        quoted += "'";
        return quoted;
    }

    class ScopedEnvironmentValues
    {
    public:
        explicit ScopedEnvironmentValues(
            std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old_value = std::getenv(name))
                {
                    entry.had_old_value = true;
                    entry.old_value = old_value;
                }
                entries_.push_back(std::move(entry));
                setenv(name, value, 1);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedEnvironmentValues()
        {
            for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
            {
                if (it->had_old_value)
                {
                    setenv(it->name.c_str(), it->old_value.c_str(), 1);
                }
                else
                {
                    unsetenv(it->name.c_str());
                }
            }
            mutableDebugEnv().reload();
        }

        ScopedEnvironmentValues(const ScopedEnvironmentValues &) = delete;
        ScopedEnvironmentValues &operator=(const ScopedEnvironmentValues &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_old_value = false;
            std::string old_value;
        };

        std::vector<Entry> entries_;
    };

    inline bool densePhase138DirectCandidateEnabled()
    {
        const char *candidate =
            std::getenv("LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE");
        if (!candidate ||
            std::strcmp(candidate, "vllm_style_spec_decode") != 0)
        {
            return false;
        }
        return DebugEnv::isTruthyEnvValue(
                   std::getenv("LLAMINAR_MTP_PHASE138_EQUIVALENCE_CHECK")) ||
               DebugEnv::isTruthyEnvValue(
                   std::getenv("LLAMINAR_MTP_PHASE138_DIRECT_CANDIDATE"));
    }

    inline bool densePhase138PromotedTransactionExpected(
        const DensePrefixRestoreParityCase &test_case)
    {
        const char *candidate =
            std::getenv("LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE");
        if (candidate && *candidate)
        {
            return densePhase138DirectCandidateEnabled();
        }
        if (test_case.topology != DensePrefixParityTopology::SingleDevice ||
            test_case.devices.size() != 1)
        {
            return false;
        }
        return false;
    }

    inline void expectPhase138TransactionUsed(
        const DensePrefixRestoreParityCase &test_case,
        const PrefixRuntimeStateSnapshot &snapshot,
        const std::string &context,
        bool allow_transaction_rollbacks = false)
    {
        if (!densePhase138PromotedTransactionExpected(test_case))
        {
            return;
        }
        EXPECT_GT(snapshot.mtp_transaction_commits, 0u)
            << context << " did not commit any MTP transactions";
        if (!allow_transaction_rollbacks)
        {
            EXPECT_EQ(snapshot.mtp_transaction_rollbacks, 0u)
                << context << " should not roll back under the Phase 13.8 transaction";
        }
        EXPECT_EQ(snapshot.mtp_transaction_validation_failures, 0u)
            << context << " hit MTP transaction validation failures";
    }

    class ScopedDenseParityDeterministicMode
    {
    public:
        explicit ScopedDenseParityDeterministicMode(bool enabled)
            : enabled_(enabled)
        {
            if (!enabled_)
            {
                return;
            }

            if (const char *old_value = std::getenv("LLAMINAR_DETERMINISTIC"))
            {
                had_old_deterministic_env_ = true;
                old_deterministic_env_ = old_value;
            }

#ifdef HAVE_CUDA
            old_cuda_prefill_deterministic_ = cudaNativeVNNIPrefill_getDeterministicMode();
#endif

            setenv("LLAMINAR_DETERMINISTIC", "1", 1);
            mutableDebugEnv().reload();
#ifdef HAVE_CUDA
            cudaNativeVNNIPrefill_setDeterministicMode(true);
#endif
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ~ScopedDenseParityDeterministicMode()
        {
            if (!enabled_)
            {
                return;
            }

#ifdef HAVE_CUDA
            cudaNativeVNNIPrefill_setDeterministicMode(old_cuda_prefill_deterministic_);
#endif
            if (had_old_deterministic_env_)
            {
                setenv("LLAMINAR_DETERMINISTIC", old_deterministic_env_.c_str(), 1);
            }
            else
            {
                unsetenv("LLAMINAR_DETERMINISTIC");
            }
            mutableDebugEnv().reload();
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ScopedDenseParityDeterministicMode(const ScopedDenseParityDeterministicMode &) = delete;
        ScopedDenseParityDeterministicMode &operator=(const ScopedDenseParityDeterministicMode &) = delete;

    private:
        bool enabled_ = false;
        bool had_old_deterministic_env_ = false;
        std::string old_deterministic_env_;
#ifdef HAVE_CUDA
        bool old_cuda_prefill_deterministic_ = false;
#endif
    };

    inline bool shouldUseDenseParityDeterministicMode(
        const DensePrefixRestoreParityCase &test_case)
    {
        if (test_case.required_cuda_devices > 0)
        {
            return true;
        }
        if (test_case.required_rocm_devices > 0)
        {
            return true;
        }
        return std::any_of(
            test_case.devices.begin(),
            test_case.devices.end(),
            [](const GlobalDeviceAddress &device)
            {
                return device.device_type == DeviceType::CUDA ||
                       device.device_type == DeviceType::ROCm;
            });
    }

    inline std::string firstEnvOrDefault(
        const std::vector<std::string> &names,
        const std::string &fallback)
    {
        for (const auto &name : names)
        {
            const char *value = std::getenv(name.c_str());
            if (value && *value)
            {
                return value;
            }
        }
        return fallback;
    }

    inline std::string formatTokenWindow(
        const std::vector<int32_t> &tokens,
        size_t center,
        size_t context = 8)
    {
        if (tokens.empty())
        {
            return "[]";
        }

        const size_t begin = center > context ? center - context : 0;
        const size_t end = std::min(tokens.size(), center + context + 1);
        std::ostringstream oss;
        oss << "[";
        if (begin > 0)
        {
            oss << "... ";
        }
        for (size_t i = begin; i < end; ++i)
        {
            if (i != begin)
            {
                oss << ", ";
            }
            if (i == center)
            {
                oss << "{" << tokens[i] << "}";
            }
            else
            {
                oss << tokens[i];
            }
        }
        if (end < tokens.size())
        {
            oss << " ...";
        }
        oss << "]";
        return oss.str();
    }

    inline int denseArgmaxToken(const float *logits, int vocab_size)
    {
        if (!logits || vocab_size <= 0)
        {
            return -1;
        }
        return static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
    }

    inline std::string denseTopKSummary(const float *logits, int vocab_size, int k = 8)
    {
        if (!logits || vocab_size <= 0 || k <= 0)
        {
            return "<no logits>";
        }

        std::vector<int> indices(static_cast<size_t>(vocab_size));
        std::iota(indices.begin(), indices.end(), 0);
        const int limit = std::min(k, vocab_size);
        std::partial_sort(
            indices.begin(),
            indices.begin() + limit,
            indices.end(),
            [logits](int lhs, int rhs)
            {
                if (logits[lhs] == logits[rhs])
                {
                    return lhs < rhs;
                }
                return logits[lhs] > logits[rhs];
            });

        std::ostringstream oss;
        for (int i = 0; i < limit; ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            const int token = indices[static_cast<size_t>(i)];
            oss << token << ":" << logits[token];
        }
        return oss.str();
    }

    inline std::string denseJoinTokens(const std::vector<int32_t> &tokens)
    {
        std::ostringstream oss;
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (i > 0)
            {
                oss << ' ';
            }
            oss << tokens[i];
        }
        return oss.str();
    }

    inline ::testing::AssertionResult tokenSequencesMatch(
        const std::vector<int32_t> &actual,
        const std::vector<int32_t> &expected,
        const std::string &label)
    {
        if (actual == expected)
        {
            return ::testing::AssertionSuccess();
        }

        const size_t common = std::min(actual.size(), expected.size());
        size_t mismatch = common;
        for (size_t i = 0; i < common; ++i)
        {
            if (actual[i] != expected[i])
            {
                mismatch = i;
                break;
            }
        }

        std::ostringstream oss;
        oss << label << " token sequence mismatch";
        if (actual.size() != expected.size())
        {
            oss << " (actual size " << actual.size()
                << ", expected size " << expected.size() << ")";
        }

        if (mismatch < common)
        {
            oss << " at decode index " << mismatch
                << ": actual=" << actual[mismatch]
                << ", expected=" << expected[mismatch]
                << "\n  actual window:   "
                << formatTokenWindow(actual, mismatch)
                << "\n  expected window: "
                << formatTokenWindow(expected, mismatch);
        }
        else
        {
            oss << "; all " << common
                << " shared-prefix tokens match, extra tail differs"
                << "\n  actual tail:   "
                << formatTokenWindow(actual, common > 0 ? common - 1 : 0)
                << "\n  expected tail: "
                << formatTokenWindow(expected, common > 0 ? common - 1 : 0);
        }

        return ::testing::AssertionFailure() << oss.str();
    }

    inline ::testing::AssertionResult floatBytePayloadsNear(
        const std::vector<uint8_t> &actual,
        const std::vector<uint8_t> &expected,
        const std::string &label,
        float abs_tolerance = 1.0e-5f,
        float rel_tolerance = 1.0e-4f)
    {
        if (actual.size() != expected.size())
        {
            return ::testing::AssertionFailure()
                   << label << " payload size mismatch: actual="
                   << actual.size() << " expected=" << expected.size();
        }
        if (actual.empty())
        {
            return ::testing::AssertionSuccess();
        }
        if (actual.size() % sizeof(float) != 0)
        {
            if (actual == expected)
            {
                return ::testing::AssertionSuccess();
            }
            size_t first_mismatch = 0;
            while (first_mismatch < actual.size() &&
                   actual[first_mismatch] == expected[first_mismatch])
            {
                ++first_mismatch;
            }
            return ::testing::AssertionFailure()
                   << label << " byte payload mismatch at byte "
                   << first_mismatch << ": actual="
                   << static_cast<int>(actual[first_mismatch])
                   << " expected="
                   << static_cast<int>(expected[first_mismatch]);
        }

        const size_t float_count = actual.size() / sizeof(float);
        size_t mismatch_count = 0;
        size_t first_mismatch_index = std::numeric_limits<size_t>::max();
        float first_actual = 0.0f;
        float first_expected = 0.0f;
        float first_abs = 0.0f;
        float first_rel = 0.0f;
        size_t worst_index = 0;
        float worst_actual = 0.0f;
        float worst_expected = 0.0f;
        float worst_abs = 0.0f;
        float worst_rel = 0.0f;
        for (size_t i = 0; i < float_count; ++i)
        {
            float a = 0.0f;
            float e = 0.0f;
            std::memcpy(&a, actual.data() + i * sizeof(float), sizeof(float));
            std::memcpy(&e, expected.data() + i * sizeof(float), sizeof(float));
            const float abs_diff = std::fabs(a - e);
            const float scale = std::max(std::fabs(a), std::fabs(e));
            const float rel_diff = scale > 0.0f ? abs_diff / scale : 0.0f;
            const bool finite_match = std::isfinite(a) && std::isfinite(e);
            const bool within_tolerance =
                finite_match &&
                abs_diff <= abs_tolerance + rel_tolerance * scale;
            if (!within_tolerance)
            {
                ++mismatch_count;
                if (first_mismatch_index == std::numeric_limits<size_t>::max())
                {
                    first_mismatch_index = i;
                    first_actual = a;
                    first_expected = e;
                    first_abs = abs_diff;
                    first_rel = rel_diff;
                }
                if (abs_diff > worst_abs)
                {
                    worst_index = i;
                    worst_actual = a;
                    worst_expected = e;
                    worst_abs = abs_diff;
                    worst_rel = rel_diff;
                }
            }
        }

        if (mismatch_count == 0)
        {
            return ::testing::AssertionSuccess();
        }

        return ::testing::AssertionFailure()
               << label << " payload differs in " << mismatch_count
               << " / " << float_count << " floats; first index="
               << first_mismatch_index << " actual=" << first_actual
               << " expected=" << first_expected
               << " abs_diff=" << first_abs
               << " rel_diff=" << first_rel
               << "; worst index="
               << worst_index << " actual=" << worst_actual
               << " expected=" << worst_expected
               << " abs_diff=" << worst_abs
               << " rel_diff=" << worst_rel
               << " tolerances(abs=" << abs_tolerance
               << ", rel=" << rel_tolerance << ")";
    }

    inline std::vector<uint8_t> byteSlice(
        const std::vector<uint8_t> &bytes,
        size_t offset,
        size_t count)
    {
        if (offset > bytes.size() || count > bytes.size() - offset)
        {
            return {};
        }
        return std::vector<uint8_t>(
            bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + count));
    }

    inline ::testing::AssertionResult prefixSnapshotPayloadsNear(
        const PrefixStateSnapshot &actual,
        const PrefixStateSnapshot &expected,
        const std::string &label)
    {
        if (!actual.valid || !expected.valid)
        {
            return ::testing::AssertionFailure()
                   << label << " invalid snapshot: actual="
                   << actual.valid << " expected=" << expected.valid;
        }
        if (actual.cached_tokens != expected.cached_tokens)
        {
            return ::testing::AssertionFailure()
                   << label << " cached token mismatch: actual="
                   << actual.cached_tokens << " expected="
                   << expected.cached_tokens;
        }
        if (actual.blocks.empty() || expected.blocks.empty())
        {
            return ::testing::AssertionFailure()
                   << label << " missing main prefix block: actual="
                   << actual.blocks.size() << " expected="
                   << expected.blocks.size();
        }

        const PrefixBlockHandle &actual_block = actual.blocks.back();
        const PrefixBlockHandle &expected_block = expected.blocks.back();
        if (actual_block.layout.hybrid_host_state_bytes !=
                expected_block.layout.hybrid_host_state_bytes ||
            actual_block.layout.hybrid_device_state_bytes !=
                expected_block.layout.hybrid_device_state_bytes ||
            actual_block.layout.terminal_hidden_bytes !=
                expected_block.layout.terminal_hidden_bytes)
        {
            return ::testing::AssertionFailure()
                   << label << " layout mismatch"
                   << " hybrid_host actual="
                   << actual_block.layout.hybrid_host_state_bytes
                   << " expected="
                   << expected_block.layout.hybrid_host_state_bytes
                   << " hybrid_device actual="
                   << actual_block.layout.hybrid_device_state_bytes
                   << " expected="
                   << expected_block.layout.hybrid_device_state_bytes
                   << " terminal_hidden actual="
                   << actual_block.layout.terminal_hidden_bytes
                   << " expected="
                   << expected_block.layout.terminal_hidden_bytes;
        }

        if (actual_block.layout.hybrid_state_bytes > 0)
        {
            if (!actual_block.has_hybrid_state ||
                !expected_block.has_hybrid_state ||
                !actual_block.hybrid_storage ||
                !expected_block.hybrid_storage)
            {
                return ::testing::AssertionFailure()
                       << label << " missing hybrid state payload";
            }
            const size_t host_bytes =
                actual_block.layout.hybrid_host_state_bytes;
            const size_t device_bytes =
                actual_block.layout.hybrid_device_state_bytes;
            if (host_bytes > 0)
            {
                auto host_result = floatBytePayloadsNear(
                    byteSlice(*actual_block.hybrid_storage, 0, host_bytes),
                    byteSlice(*expected_block.hybrid_storage, 0, host_bytes),
                    label + " hybrid host");
                if (!host_result)
                {
                    return host_result;
                }
            }
            if (device_bytes > 0)
            {
                std::string device_label = label + " hybrid device";
                if (actual_block.layout.gdn_layers > 0 &&
                    device_bytes % static_cast<size_t>(actual_block.layout.gdn_layers) == 0)
                {
                    device_label += " (gdn_layers=" +
                                    std::to_string(actual_block.layout.gdn_layers) +
                                    ", per_gdn_layer_device_bytes=" +
                                    std::to_string(
                                        device_bytes /
                                        static_cast<size_t>(actual_block.layout.gdn_layers)) +
                                    ")";
                }
                auto device_result = floatBytePayloadsNear(
                    byteSlice(*actual_block.hybrid_storage, host_bytes, device_bytes),
                    byteSlice(*expected_block.hybrid_storage, host_bytes, device_bytes),
                    device_label);
                if (!device_result)
                {
                    return device_result;
                }
            }
        }

        if (actual_block.layout.terminal_hidden_bytes > 0)
        {
            if (!actual_block.has_terminal_hidden ||
                !expected_block.has_terminal_hidden ||
                !actual_block.terminal_hidden_storage ||
                !expected_block.terminal_hidden_storage)
            {
                return ::testing::AssertionFailure()
                       << label << " missing terminal-hidden payload";
            }
            auto hidden_result = floatBytePayloadsNear(
                *actual_block.terminal_hidden_storage,
                *expected_block.terminal_hidden_storage,
                label + " terminal hidden");
            if (!hidden_result)
            {
                return hidden_result;
            }
        }

        return ::testing::AssertionSuccess();
    }

    struct DenseStageSnapshot
    {
        std::string key;
        std::vector<float> data;
        size_t rows = 0;
        size_t cols = 0;
    };

    inline std::vector<float> loadDensePyTorchSnapshot(
        const std::filesystem::path &snapshot_dir,
        const std::string &key)
    {
        const auto npy_path = snapshot_dir / (key + ".npy");
        try
        {
            cnpy::NpyArray arr = cnpy::npy_load(npy_path.string());
            std::vector<float> data;
            if (arr.word_size == sizeof(float))
            {
                const float *ptr = arr.data<float>();
                data.assign(ptr, ptr + arr.num_vals);
                return data;
            }
            if (arr.word_size == sizeof(double))
            {
                const double *ptr = arr.data<double>();
                data.resize(arr.num_vals);
                for (size_t i = 0; i < arr.num_vals; ++i)
                {
                    data[i] = static_cast<float>(ptr[i]);
                }
                return data;
            }
        }
        catch (const std::exception &)
        {
            return {};
        }
        return {};
    }

    inline ::testing::AssertionResult denseFloatVectorsNear(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        const std::string &label,
        float cosine_threshold = 0.995f,
        float rel_l2_threshold = 0.25f)
    {
        if (actual.empty() || expected.empty())
        {
            return ::testing::AssertionFailure()
                   << label << " missing payload: actual=" << actual.size()
                   << " expected=" << expected.size();
        }

        const float *expected_data = expected.data();
        size_t expected_size = expected.size();
        std::vector<float> expected_tail;
        if (expected_size > actual.size() &&
            actual.size() > 0 &&
            expected_size % actual.size() == 0)
        {
            expected_tail.assign(
                expected.end() - static_cast<std::ptrdiff_t>(actual.size()),
                expected.end());
            expected_data = expected_tail.data();
            expected_size = expected_tail.size();
        }

        if (actual.size() != expected_size)
        {
            return ::testing::AssertionFailure()
                   << label << " size mismatch: actual=" << actual.size()
                   << " expected=" << expected_size
                   << " raw_expected=" << expected.size();
        }

        double dot = 0.0;
        double actual_norm = 0.0;
        double expected_norm = 0.0;
        double diff_norm = 0.0;
        float max_abs = 0.0f;
        size_t max_abs_index = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const double a = actual[i];
            const double e = expected_data[i];
            const double d = a - e;
            dot += a * e;
            actual_norm += a * a;
            expected_norm += e * e;
            diff_norm += d * d;
            const float abs_diff = static_cast<float>(std::fabs(d));
            if (abs_diff > max_abs)
            {
                max_abs = abs_diff;
                max_abs_index = i;
            }
        }

        const double denom = std::sqrt(actual_norm * expected_norm);
        const double cosine = denom > 0.0 ? dot / denom : 1.0;
        const double rel_l2 = expected_norm > 0.0
                                  ? std::sqrt(diff_norm / expected_norm)
                                  : std::sqrt(diff_norm);
        if (cosine >= cosine_threshold && rel_l2 <= rel_l2_threshold)
        {
            return ::testing::AssertionSuccess();
        }

        return ::testing::AssertionFailure()
               << label << " differs: cosine=" << cosine
               << " rel_l2=" << rel_l2
               << " max_abs=" << max_abs
               << " max_abs_index=" << max_abs_index
               << " actual_at_max=" << actual[max_abs_index]
               << " expected_at_max=" << expected_data[max_abs_index]
               << " thresholds(cosine>=" << cosine_threshold
               << ", rel_l2<=" << rel_l2_threshold << ")";
    }

    inline ::testing::AssertionResult denseDecodeStepSnapshotsNearPyTorch(
        const std::map<std::string, DenseStageSnapshot> &snapshots,
        const std::filesystem::path &snapshot_dir,
        int decode_step,
        const std::string &label)
    {
        static const std::vector<std::string> kOrderedKeys = {
            "EMBEDDING",
            "layer0_ATTENTION_NORM",
            "layer0_QKV_PROJECTION",
            "layer0_GDN_Z_PROJECTION",
            "layer0_GDN_ALPHA",
            "layer0_GDN_BETA",
            "layer0_GDN_CONV1D_OUTPUT",
            "layer0_GDN_DELTA_RULE_OUTPUT",
            "layer0_GDN_NORM_GATE_OUTPUT",
            "layer0_ATTENTION_OUTPUT",
            "layer0_ATTENTION_RESIDUAL",
            "layer0_FFN_NORM",
            "layer0_FFN_DOWN",
            "layer0_FFN_RESIDUAL",
            "layer1_ATTENTION_NORM",
            "layer1_QKV_PROJECTION",
            "layer1_GDN_Z_PROJECTION",
            "layer1_GDN_ALPHA",
            "layer1_GDN_BETA",
            "layer1_GDN_CONV1D_OUTPUT",
            "layer1_GDN_DELTA_RULE_OUTPUT",
            "layer1_GDN_NORM_GATE_OUTPUT",
            "layer1_ATTENTION_OUTPUT",
            "layer1_ATTENTION_RESIDUAL",
            "FINAL_NORM",
            "LM_HEAD",
        };

        const std::string prefix =
            "decode_step" + std::to_string(decode_step) + "_";
        size_t compared = 0;
        std::ostringstream errors;
        for (const auto &key : kOrderedKeys)
        {
            const auto actual_it = snapshots.find(key);
            if (actual_it == snapshots.end())
            {
                continue;
            }
            const std::vector<float> expected =
                loadDensePyTorchSnapshot(snapshot_dir, prefix + key);
            if (expected.empty())
            {
                continue;
            }
            ++compared;
            const auto match = denseFloatVectorsNear(
                actual_it->second.data,
                expected,
                label + " " + key);
            if (!match)
            {
                errors << "\nfirst divergent compared stage: " << key
                       << "\n" << match.message();
                break;
            }
        }

        if (compared == 0)
        {
            return ::testing::AssertionFailure()
                   << label << " found no comparable decode_step"
                   << decode_step << " snapshots in " << snapshot_dir;
        }
        if (!errors.str().empty())
        {
            return ::testing::AssertionFailure()
                   << label << " PyTorch stage parity failed after "
                   << compared << " compared stages"
                   << errors.str();
        }
        return ::testing::AssertionSuccess();
    }

    inline std::map<std::string, DenseStageSnapshot> captureDenseStageSnapshots(
        IInferenceRunner &runner)
    {
        std::map<std::string, DenseStageSnapshot> snapshots;
        auto keys = runner.getSnapshotKeys();
        std::sort(keys.begin(), keys.end());
        for (const auto &key : keys)
        {
            SnapshotInfo info = runner.getSnapshotWithShape(key);
            if (!info || info.rows == 0 || info.cols == 0 ||
                info.size != info.rows * info.cols)
            {
                continue;
            }
            DenseStageSnapshot snapshot;
            snapshot.key = key;
            snapshot.rows = info.rows;
            snapshot.cols = info.cols;
            snapshot.data.assign(info.data, info.data + info.size);
            snapshots.emplace(key, std::move(snapshot));
        }
        return snapshots;
    }

    inline ::testing::AssertionResult denseVerifierRowSnapshotsNear(
        const std::map<std::string, DenseStageSnapshot> &verifier_snapshots,
        const std::map<std::string, DenseStageSnapshot> &single_row_snapshots,
        const std::string &label,
        int verifier_rows,
        int verifier_row_index,
        float abs_tolerance = 1.0e-6f,
        float rel_tolerance = 1.0e-6f)
    {
        if (verifier_rows <= 0 ||
            verifier_row_index < 0 ||
            verifier_row_index >= verifier_rows)
        {
            return ::testing::AssertionFailure()
                   << label << " invalid verifier row index "
                   << verifier_row_index << " for verifier rows "
                   << verifier_rows;
        }

        struct Mismatch
        {
            std::string key;
            size_t col = 0;
            size_t mismatches = 0;
            float first_actual = 0.0f;
            float first_expected = 0.0f;
            float first_abs = 0.0f;
            float first_rel = 0.0f;
            float max_abs = 0.0f;
            float max_rel = 0.0f;
        };

        std::vector<Mismatch> mismatches;
        size_t comparable = 0;
        for (const auto &[key, verifier] : verifier_snapshots)
        {
            const auto single_it = single_row_snapshots.find(key);
            if (single_it == single_row_snapshots.end())
            {
                continue;
            }
            const DenseStageSnapshot &single = single_it->second;
            if (static_cast<int>(verifier.rows) != verifier_rows ||
                single.rows != 1 ||
                verifier.cols != single.cols)
            {
                continue;
            }
            ++comparable;
            Mismatch mismatch;
            mismatch.key = key;
            const size_t verifier_row_offset =
                static_cast<size_t>(verifier_row_index) * verifier.cols;
            for (size_t col = 0; col < verifier.cols; ++col)
            {
                const float a = verifier.data[verifier_row_offset + col];
                const float e = single.data[col];
                const float abs_diff = std::fabs(a - e);
                const float scale = std::max(std::fabs(a), std::fabs(e));
                const float rel_diff = scale > 0.0f ? abs_diff / scale : 0.0f;
                const bool within_tolerance =
                    std::isfinite(a) && std::isfinite(e) &&
                    abs_diff <= abs_tolerance + rel_tolerance * scale;
                if (!within_tolerance)
                {
                    if (mismatch.mismatches == 0)
                    {
                        mismatch.col = col;
                        mismatch.first_actual = a;
                        mismatch.first_expected = e;
                        mismatch.first_abs = abs_diff;
                        mismatch.first_rel = rel_diff;
                    }
                    ++mismatch.mismatches;
                    if (abs_diff > mismatch.max_abs)
                    {
                        mismatch.max_abs = abs_diff;
                        mismatch.max_rel = rel_diff;
                    }
                }
            }
            if (mismatch.mismatches > 0)
            {
                mismatches.push_back(std::move(mismatch));
            }
        }

        if (mismatches.empty())
        {
            if (comparable == 0)
            {
                return ::testing::AssertionFailure()
                       << label << " found no comparable verifier/single-row "
                       << "stage snapshots";
            }
            return ::testing::AssertionSuccess();
        }

        auto stage_sort_key = [](const std::string &key)
        {
            int layer = 100000;
            int stage = 100000;
            if (key.rfind("layer", 0) == 0)
            {
                size_t digit_end = 5;
                while (digit_end < key.size() &&
                       std::isdigit(static_cast<unsigned char>(key[digit_end])))
                {
                    ++digit_end;
                }
                try
                {
                    layer = std::stoi(key.substr(5, digit_end - 5));
                }
                catch (...)
                {
                    layer = 99999;
                }
            }

            static const std::vector<std::pair<std::string, int>> stage_order = {
                {"ATTENTION_NORM", 10},
                {"QKV_PROJECTION", 20},
                {"GDN_Z_PROJECTION", 25},
                {"Q_PROJECTION", 30},
                {"K_PROJECTION", 40},
                {"V_PROJECTION", 50},
                {"Q_NORM", 60},
                {"K_NORM", 70},
                {"Q_ROPE", 80},
                {"K_ROPE", 90},
                {"GDN_CONV1D_OUTPUT", 100},
                {"GDN_ALPHA", 105},
                {"GDN_BETA", 106},
                {"GDN_DELTA_RULE_OUTPUT", 110},
                {"GDN_NORM_GATE_OUTPUT", 120},
                {"ATTENTION_CONTEXT", 130},
                {"ATTENTION_CONTEXT_GATED", 140},
                {"ATTENTION_OUTPUT", 150},
                {"ATTENTION_RESIDUAL", 160},
                {"FFN_NORM", 170},
                {"FFN_GATE", 180},
                {"FFN_UP", 190},
                {"FFN_SWIGLU", 200},
                {"FFN_DOWN", 210},
                {"FFN_RESIDUAL", 220},
            };
            for (const auto &[needle, order] : stage_order)
            {
                if (key.find(needle) != std::string::npos)
                {
                    stage = order;
                    break;
                }
            }
            return std::tuple<int, int, std::string>(layer, stage, key);
        };
        std::sort(mismatches.begin(),
                  mismatches.end(),
                  [&](const Mismatch &a, const Mismatch &b)
                  {
                      return stage_sort_key(a.key) < stage_sort_key(b.key);
                  });

        std::ostringstream oss;
        oss << label << " stage snapshot row mismatch across "
            << mismatches.size() << " / " << comparable
            << " comparable stages";
        const size_t limit = std::min<size_t>(mismatches.size(), 24);
        for (size_t i = 0; i < limit; ++i)
        {
            const auto &m = mismatches[i];
            oss << "\n  " << m.key
                << ": mismatches=" << m.mismatches
                << " first_col=" << m.col
                << " actual=" << m.first_actual
                << " expected=" << m.first_expected
                << " abs=" << m.first_abs
                << " rel=" << m.first_rel
                << " max_abs=" << m.max_abs
                << " max_rel=" << m.max_rel;
        }
        return ::testing::AssertionFailure() << oss.str();
    }

    inline int mpiWorldSize()
    {
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        return world_size;
    }

    inline std::vector<int32_t> readTokenListFromMetadata(
        const std::filesystem::path &metadata_path,
        const std::string &key)
    {
        std::ifstream file(metadata_path);
        if (!file.is_open())
        {
            return {};
        }

        std::string line;
        const std::string prefix = key + ":";
        while (std::getline(file, line))
        {
            if (line.rfind(prefix, 0) != 0)
            {
                continue;
            }

            std::string tokens = line.substr(prefix.size());
            const size_t start = tokens.find_first_not_of(" \t");
            if (start != std::string::npos)
            {
                tokens = tokens.substr(start);
            }

            std::vector<int32_t> result;
            std::stringstream ss(tokens);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                const size_t token_start = token.find_first_not_of(" \t");
                const size_t token_end = token.find_last_not_of(" \t");
                if (token_start == std::string::npos || token_end == std::string::npos)
                {
                    continue;
                }
                result.push_back(std::stoi(token.substr(token_start, token_end - token_start + 1)));
            }
            return result;
        }

        return {};
    }

    inline std::optional<std::string> readStringFromMetadata(
        const std::filesystem::path &metadata_path,
        const std::string &key)
    {
        std::ifstream file(metadata_path);
        if (!file.is_open())
        {
            return std::nullopt;
        }

        const std::string prefix = key + ":";
        std::string line;
        while (std::getline(file, line))
        {
            if (line.rfind(prefix, 0) != 0)
            {
                continue;
            }

            std::string value = line.substr(prefix.size());
            const size_t start = value.find_first_not_of(" \t");
            if (start == std::string::npos)
            {
                return std::string{};
            }
            const size_t end = value.find_last_not_of(" \t\r\n");
            return value.substr(start, end - start + 1);
        }

        return std::nullopt;
    }

    inline bool metadataLooksUsable(
        const std::filesystem::path &metadata_path,
        const std::string &expected_prompt,
        int required_decode_steps)
    {
        constexpr int kRequiredQwen36DenseSnapshotVersion = 4;
        const auto version = readStringFromMetadata(metadata_path, "snapshot_version");
        const auto prompt = readStringFromMetadata(metadata_path, "prompt");
        const auto token_ids = readTokenListFromMetadata(metadata_path, "token_ids");
        const auto decode_tokens = readTokenListFromMetadata(metadata_path, "decode_tokens");
        int parsed_version = 0;
        if (version.has_value())
        {
            try
            {
                parsed_version = std::stoi(*version);
            }
            catch (...)
            {
                parsed_version = 0;
            }
        }
        return prompt.has_value() &&
               parsed_version >= kRequiredQwen36DenseSnapshotVersion &&
               *prompt == expected_prompt &&
               !token_ids.empty() &&
               decode_tokens.size() >= static_cast<size_t>(required_decode_steps);
    }

    inline bool regenerateQwen36Metadata(
        const std::string &model_path,
        const std::filesystem::path &metadata_path,
        const std::string &prompt,
        int decode_steps,
        std::string *output)
    {
        std::filesystem::create_directories(metadata_path.parent_path());

        std::string script =
            "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
            "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
            "[ -f /workspaces/llaminar/.venv/bin/activate ] && "
            "source /workspaces/llaminar/.venv/bin/activate; "
            "python3 python/reference/generate_qwen35_pipeline_snapshots.py";
        script += " --model " + shellQuote(model_path);
        script += " --prompt " + shellQuote(prompt);
        script += " --decode-steps " + std::to_string(decode_steps);
        script += " --output " + shellQuote(metadata_path.parent_path().string());
        script += " --metadata-only";

        const std::string command = "bash -c " + shellQuote(script) + " 2>&1";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            if (output)
            {
                *output = "failed to spawn python metadata generator";
            }
            return false;
        }

        char buffer[512];
        std::string captured;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            captured += buffer;
        }

        const int exit_code = pclose(pipe);
        if (output)
        {
            *output = std::move(captured);
        }
        return exit_code == 0;
    }

    inline bool qwen36DecodeSnapshotsLookUsable(
        const std::filesystem::path &snapshot_dir,
        int required_decode_steps)
    {
        if (required_decode_steps <= 0)
        {
            return true;
        }

        const std::vector<std::string> required_keys = {
            "layer0_QKV_PROJECTION",
            "layer0_GDN_Z_PROJECTION",
            "layer0_GDN_ALPHA",
            "layer0_GDN_BETA",
            "layer0_GDN_CONV1D_OUTPUT",
            "layer0_GDN_DELTA_RULE_OUTPUT",
        };

        for (int step = 0; step < required_decode_steps; ++step)
        {
            for (const auto &key : required_keys)
            {
                const auto path = snapshot_dir /
                                  ("decode_step" + std::to_string(step) + "_" + key + ".npy");
                if (!std::filesystem::exists(path))
                {
                    return false;
                }
            }
        }
        return true;
    }

    inline bool regenerateQwen36DecodeSnapshots(
        const std::string &model_path,
        const std::filesystem::path &metadata_path,
        const std::string &prompt,
        int decode_steps,
        std::string *output)
    {
        std::filesystem::create_directories(metadata_path.parent_path());

        std::string script =
            "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
            "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
            "[ -f /workspaces/llaminar/.venv/bin/activate ] && "
            "source /workspaces/llaminar/.venv/bin/activate; "
            "python3 python/reference/generate_qwen35_pipeline_snapshots.py";
        script += " --model " + shellQuote(model_path);
        script += " --prompt " + shellQuote(prompt);
        script += " --decode-steps " + std::to_string(decode_steps);
        script += " --output " + shellQuote(metadata_path.parent_path().string());
        script += " --decode-snapshots-only";

        const std::string command = "bash -c " + shellQuote(script) + " 2>&1";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            if (output)
            {
                *output = "failed to spawn python snapshot generator";
            }
            return false;
        }

        char buffer[512];
        std::string captured;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            captured += buffer;
        }

        const int exit_code = pclose(pipe);
        if (output)
        {
            *output = std::move(captured);
        }
        return exit_code == 0;
    }

    inline void ensurePyTorchDecodeSnapshots(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const std::filesystem::path &metadata_path)
    {
        const auto snapshot_dir = metadata_path.parent_path();
        if (qwen36DecodeSnapshotsLookUsable(snapshot_dir, test_case.decode_steps) &&
            metadataLooksUsable(metadata_path, test_case.prompt, test_case.decode_steps))
        {
            return;
        }

        std::string output;
        ASSERT_TRUE(regenerateQwen36DecodeSnapshots(
            model_path,
            metadata_path,
            test_case.prompt,
            test_case.decode_steps,
            &output))
            << test_case.name << " failed to regenerate PyTorch decode snapshots at "
            << snapshot_dir << "\n"
            << output;

        ASSERT_TRUE(metadataLooksUsable(metadata_path, test_case.prompt, test_case.decode_steps))
            << test_case.name << " regenerated metadata is incomplete at "
            << metadata_path << "\n"
            << output;
        ASSERT_TRUE(qwen36DecodeSnapshotsLookUsable(snapshot_dir, test_case.decode_steps))
            << test_case.name << " regenerated decode snapshots are incomplete at "
            << snapshot_dir << "\n"
            << output;
    }

    inline void ensurePyTorchMetadata(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const std::filesystem::path &metadata_path)
    {
        if (metadataLooksUsable(metadata_path, test_case.prompt, test_case.decode_steps))
        {
            return;
        }

        std::string output;
        ASSERT_TRUE(regenerateQwen36Metadata(
            model_path,
            metadata_path,
            test_case.prompt,
            test_case.decode_steps,
            &output))
            << test_case.name << " failed to regenerate PyTorch metadata at "
            << metadata_path << "\n"
            << output;

        ASSERT_TRUE(metadataLooksUsable(metadata_path, test_case.prompt, test_case.decode_steps))
            << test_case.name << " regenerated metadata is incomplete at "
            << metadata_path << "\n"
            << output;
    }

    inline std::optional<std::string> densePrefixParitySkipReason(
        const DensePrefixRestoreParityCase &test_case)
    {
        const int world_size = mpiWorldSize();
        if (test_case.topology == DensePrefixParityTopology::NodeLocalTP)
        {
            if (world_size != test_case.mpi_ranks)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires exactly "
                    << test_case.mpi_ranks << " MPI ranks (got "
                    << world_size << ")";
                return oss.str();
            }
        }
        else if (world_size != 1)
        {
            return test_case.name + " is a local topology test and must run with one MPI rank";
        }

        if (test_case.required_cuda_devices > 0 || test_case.required_rocm_devices > 0)
        {
            auto &dm = DeviceManager::instance();
            dm.initialize(-1, false);
            if (dm.cuda_device_count() < test_case.required_cuda_devices)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires "
                    << test_case.required_cuda_devices
                    << " CUDA device(s)";
                return oss.str();
            }
            if (dm.rocm_device_count() < test_case.required_rocm_devices)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires "
                    << test_case.required_rocm_devices
                    << " ROCm device(s)";
                return oss.str();
            }
        }

        return std::nullopt;
    }

    inline std::vector<PPStageDefinition> splitStages(
        int total_layers,
        const std::vector<GlobalDeviceAddress> &devices)
    {
        std::vector<PPStageDefinition> stages;
        const int stage_count = static_cast<int>(devices.size());
        if (stage_count <= 0 || total_layers <= 0)
        {
            return stages;
        }

        int first = 0;
        for (int stage = 0; stage < stage_count; ++stage)
        {
            const int next = ((stage + 1) * total_layers) / stage_count;
            const int last = std::max(first, next) - 1;
            stages.push_back(PPStageDefinition{
                stage,
                "stage" + std::to_string(stage),
                first,
                last,
            });
            first = last + 1;
        }
        return stages;
    }

    inline OrchestrationConfig makeDensePrefixRestoreConfig(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        bool enable_prefix_cache,
        int block_size,
        bool enable_mtp = false,
        int mtp_draft_tokens = 1,
        MTPDepthPolicyConfig depth_policy = {})
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = test_case.kv_cache_precision;
        config.prefix_cache.enabled = enable_prefix_cache;
        config.prefix_cache.storage_mode = enable_prefix_cache
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = block_size;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 1024ull * 1024ull * 1024ull;
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = mtp_draft_tokens;
        config.mtp.depth_policy = depth_policy;

        switch (test_case.topology)
        {
        case DensePrefixParityTopology::SingleDevice:
            config.tp_degree = 1;
            config.pp_degree = 1;
            config.device_for_this_rank = test_case.devices.empty()
                                              ? GlobalDeviceAddress::cpu()
                                              : test_case.devices.front();
            break;

        case DensePrefixParityTopology::LocalTP:
            config.tp_degree = static_cast<int>(test_case.devices.size());
            config.tp_scope = TPScope::LOCAL;
            config.tp_devices = test_case.devices;
            config.pp_degree = 1;
            config.default_backend = CollectiveBackendType::RCCL;
            break;

        case DensePrefixParityTopology::LocalPP:
            config.tp_degree = 1;
            config.pp_degree = static_cast<int>(test_case.devices.size());
            config.pp_split = PPSplitMode::MANUAL;
            config.domain_definitions.clear();
            config.pp_stage_definitions = splitStages(test_case.main_layers, test_case.devices);
            for (size_t i = 0; i < test_case.devices.size(); ++i)
            {
                DomainDefinition domain;
                domain.name = "stage" + std::to_string(i);
                domain.devices = {test_case.devices[i]};
                domain.scope = TPScope::LOCAL;
                domain.owner_rank = 0;
                domain.backend = CollectiveBackendType::AUTO;
                config.domain_definitions.push_back(std::move(domain));
            }
            break;

        case DensePrefixParityTopology::NodeLocalTP:
            config.tp_degree = test_case.mpi_ranks;
            config.tp_scope = TPScope::NODE_LOCAL;
            config.pp_degree = 1;
            config.default_backend = CollectiveBackendType::MPI;
            config.device_mode = DeviceAssignmentMode::EXPLICIT;
            config.device_map.clear();
            config.device_map_numa_explicit.clear();
            for (int rank = 0;
                 rank < test_case.mpi_ranks &&
                 rank < static_cast<int>(test_case.devices.size());
                 ++rank)
            {
                config.device_map.emplace_back(rank, test_case.devices[rank]);
                config.device_map_numa_explicit.emplace_back(
                    rank,
                    test_case.devices[rank].hasValidNuma());
            }
            break;
        }

        return config;
    }

    inline DensePrefixRestoreParityCase qwen36DensePrefixParityCase(
        const std::string &name,
        DensePrefixParityTopology topology)
    {
        DensePrefixRestoreParityCase test_case{
            .name = name,
            .topology = topology,
            .model_envs = {
                "LLAMINAR_QWEN36_DENSE_MODEL",
                "LLAMINAR_PARITY_DENSE_MODEL",
            },
            .default_model_path = "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
            .metadata_envs = {
                "LLAMINAR_QWEN36_PARITY_METADATA",
                "LLAMINAR_PARITY_DENSE_METADATA",
            },
            .default_metadata_path = "pytorch_qwen36_dense_snapshots/metadata.txt",
            .prompt = "The quick brown fox jumps over the lazy dog",
            .kv_cache_precision = "auto",
            .decode_steps = 3,
            .max_seq_len = 96,
            .main_layers = 64,
        };

        switch (topology)
        {
        case DensePrefixParityTopology::SingleDevice:
            test_case.devices = {GlobalDeviceAddress::rocm(0)};
            test_case.required_rocm_devices = 1;
            break;
        case DensePrefixParityTopology::LocalTP:
            test_case.devices = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
            };
            test_case.required_rocm_devices = 2;
            break;
        case DensePrefixParityTopology::LocalPP:
            test_case.devices = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
            };
            test_case.required_rocm_devices = 2;
            break;
        case DensePrefixParityTopology::NodeLocalTP:
            test_case.devices = {
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
            };
            test_case.mpi_ranks = 2;
            break;
        }

        return test_case;
    }

    inline std::string qwen36DefaultBenchmarkPrompt()
    {
        return "The following is a comprehensive analysis of machine learning systems "
               "and their applications in modern computing environments. "
               "We will explore the fundamental concepts, examine practical implementations, "
               "and discuss the future directions of this rapidly evolving field. "
               "Machine learning has transformed how we approach problem-solving across "
               "numerous domains, from natural language processing to computer vision, "
               "from autonomous vehicles to medical diagnosis. "
               "The key to understanding these systems lies in grasping the underlying "
               "mathematical foundations while also appreciating the engineering challenges "
               "involved in deploying them at scale. "
               "Let us begin our exploration with an overview of the main paradigms: "
               "supervised learning, unsupervised learning, and reinforcement learning. "
               "Each of these approaches has its own strengths and is suited to different "
               "types of problems. In supervised learning, we train models using labeled data, "
               "where the correct output is known for each input example. "
               "This approach is particularly effective for classification and regression tasks. "
               "Unsupervised learning, on the other hand, deals with finding patterns in data "
               "without explicit labels. Clustering, dimensionality reduction, and anomaly detection "
               "are common applications. Reinforcement learning takes a different approach, "
               "where agents learn optimal behaviors through interaction with an environment, "
               "receiving rewards or penalties based on their actions. "
               "Deep learning, a subset of machine learning, has revolutionized the field "
               "by enabling the training of neural networks with many layers. "
               "These deep neural networks can learn hierarchical representations of data, "
               "automatically extracting features at multiple levels of abstraction. "
               "Convolutional neural networks have become the standard for image processing, "
               "while recurrent neural networks and transformers excel at sequential data. "
               "The transformer architecture, introduced in 2017, has become particularly influential, "
               "forming the basis for large language models like GPT, BERT, and LLaMA. "
               "These models are trained on vast amounts of text data and can perform "
               "a wide range of natural language tasks with impressive accuracy. "
               "The training process involves optimizing millions or billions of parameters "
               "using gradient descent and backpropagation algorithms. "
               "Modern training infrastructure relies on specialized hardware like GPUs and TPUs, "
               "distributed computing frameworks, and sophisticated optimization techniques. "
               "Transfer learning has emerged as a powerful paradigm, allowing models "
               "pre-trained on large datasets to be fine-tuned for specific tasks "
               "with relatively little additional data. This approach has democratized "
               "access to state-of-the-art AI capabilities for researchers and practitioners "
               "who may not have the resources to train large models from scratch. "
               "As we look to the future, several exciting developments are on the horizon. "
               "Multimodal models that can process text, images, audio, and video together "
               "are becoming increasingly sophisticated. Federated learning enables "
               "training on distributed data while preserving privacy. "
               "Neural architecture search automates the design of optimal network structures. "
               "And new hardware accelerators promise to make AI more efficient and accessible. "
               "The ethical implications of these technologies cannot be overlooked. "
               "Issues of bias, fairness, transparency, and accountability must be addressed "
               "as AI systems become more prevalent in society. Responsible AI development "
               "requires collaboration between technologists, policymakers, and the public "
               "to ensure these powerful tools benefit humanity as a whole.";
    }

    inline void loadReferenceInputs(
        const DensePrefixRestoreParityCase &test_case,
        std::string *model_path,
        std::vector<int32_t> *prompt_tokens,
        std::vector<int32_t> *expected_tokens)
    {
        if (auto skip_reason = densePrefixParitySkipReason(test_case))
        {
            GTEST_SKIP() << *skip_reason;
        }

        *model_path = firstEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(*model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << *model_path;
        }

        const std::filesystem::path metadata_path = firstEnvOrDefault(
            test_case.metadata_envs,
            test_case.default_metadata_path);
        ensurePyTorchMetadata(test_case, *model_path, metadata_path);

        *prompt_tokens = readTokenListFromMetadata(metadata_path, "token_ids");
        const auto pytorch_decode_tokens = readTokenListFromMetadata(metadata_path, "decode_tokens");
        ASSERT_FALSE(prompt_tokens->empty());
        ASSERT_GE(pytorch_decode_tokens.size(), static_cast<size_t>(test_case.decode_steps));

        expected_tokens->assign(
            pytorch_decode_tokens.begin(),
            pytorch_decode_tokens.begin() + test_case.decode_steps);
    }

    inline void runDensePrefixRestoreParity(
        const DensePrefixRestoreParityCase &test_case,
        PrefixRestoreParityMode mode)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        const int block_size = mode == PrefixRestoreParityMode::FullHit
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 4;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, block_size));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_result.tokens, expected_tokens);
        EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);

        auto cached = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, true, block_size));
        ASSERT_NE(cached, nullptr);
        ASSERT_TRUE(cached->initialize()) << cached->lastError();

        std::vector<int32_t> first_prompt = prompt_tokens;
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            ASSERT_GT(prompt_tokens.size(), 4u);
            first_prompt.assign(prompt_tokens.begin(), prompt_tokens.begin() + 4);
        }

        auto first = cached->generate(first_prompt, test_case.decode_steps, greedy);
        const auto after_first = cached->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        if (mode == PrefixRestoreParityMode::FullHit)
        {
            ASSERT_EQ(first.tokens.size(), expected_tokens.size());
            EXPECT_EQ(first.tokens, expected_tokens);
        }

        auto second = cached->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_second = cached->prefixStateProbe();
        cached->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), expected_tokens.size());
        EXPECT_EQ(second.tokens, expected_tokens);
        EXPECT_EQ(second.tokens, baseline_result.tokens);
        EXPECT_TRUE(after_second.prefix_cache_ready);
        EXPECT_GE(after_second.prefix_cache_hits, 1u);

        if (mode == PrefixRestoreParityMode::FullHit)
        {
            EXPECT_TRUE(after_second.prefix_request.hit);
            EXPECT_FALSE(after_second.prefix_request.partial_hit);
            EXPECT_EQ(after_second.prefix_request.matched_tokens,
                      static_cast<int>(prompt_tokens.size()));
            EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
        }
        else
        {
            EXPECT_FALSE(after_second.prefix_request.hit);
            EXPECT_TRUE(after_second.prefix_request.partial_hit);
            EXPECT_EQ(after_second.prefix_request.matched_tokens, 4);
            EXPECT_FALSE(after_second.prefix_request.terminal_logits_restored);
        }
    }

    inline void runDenseSplitPrefillParity(
        const DensePrefixRestoreParityCase &test_case,
        int split_tokens)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GT(prompt_tokens.size(), static_cast<size_t>(split_tokens));

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, split_tokens));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens, expected_tokens);

        auto split = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, split_tokens));
        ASSERT_NE(split, nullptr);
        ASSERT_TRUE(split->initialize()) << split->lastError();
        split->setSamplingParams(greedy);

        const std::vector<int32_t> first_prompt(
            prompt_tokens.begin(),
            prompt_tokens.begin() + split_tokens);
        const std::vector<int32_t> suffix(
            prompt_tokens.begin() + split_tokens,
            prompt_tokens.end());

        ASSERT_TRUE(split->prefill(first_prompt)) << split->lastError();
        ASSERT_TRUE(split->prefill(suffix)) << split->lastError();
        EXPECT_EQ(split->currentPosition(), static_cast<int>(prompt_tokens.size()));
        EXPECT_TRUE(split->prefixStateProbe().prefill_logits_ready);

        std::vector<int32_t> split_tokens_out;
        for (int i = 0; i < test_case.decode_steps; ++i)
        {
            GenerationResult step = split->decodeStep();
            ASSERT_TRUE(step.error.empty()) << step.error;
            ASSERT_EQ(step.tokens.size(), 1u);
            split_tokens_out.push_back(step.tokens.front());
        }
        split->shutdown();

        EXPECT_EQ(split_tokens_out, expected_tokens);
        EXPECT_EQ(split_tokens_out, baseline_result.tokens);
    }

    inline void runDenseMTPParity(
        const DensePrefixRestoreParityCase &test_case,
        bool enable_prefix_cache,
        int mtp_draft_tokens = 1,
        MTPDepthPolicyConfig depth_policy = {})
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        const int block_size = enable_prefix_cache
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 2;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, block_size, false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_result.tokens, expected_tokens);
        EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);
        EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);

        auto mtp = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                enable_prefix_cache,
                block_size,
                true,
                mtp_draft_tokens,
                depth_policy));
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();

        auto first = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_first = mtp->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), expected_tokens.size());
        EXPECT_EQ(first.tokens, expected_tokens);
        EXPECT_EQ(first.tokens, baseline_result.tokens);
        EXPECT_FALSE(after_first.mtp_bypassed) << after_first.mtp_bypass_reason;
        const uint64_t expected_first_step_drafts = static_cast<uint64_t>(
            std::min(mtp_draft_tokens, std::max(0, test_case.decode_steps - 1)));
        EXPECT_GE(after_first.mtp_draft_steps, expected_first_step_drafts);
        if (expected_first_step_drafts > 0)
        {
            EXPECT_GE(after_first.mtp_verifier_runs, 1u);
            EXPECT_GE(after_first.mtp_verifier_token_count, expected_first_step_drafts + 1);
        }
        expectPhase138TransactionUsed(
            test_case,
            after_first,
            test_case.name + " first request");

        if (!enable_prefix_cache)
        {
            mtp->shutdown();
            return;
        }

        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);

        auto second = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_second = mtp->prefixStateProbe();
        mtp->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), expected_tokens.size());
        EXPECT_EQ(second.tokens, expected_tokens);
        EXPECT_EQ(second.tokens, baseline_result.tokens);
        EXPECT_TRUE(after_second.prefix_cache_ready);
        EXPECT_GE(after_second.prefix_cache_hits, 1u);
        EXPECT_TRUE(after_second.prefix_request.hit);
        EXPECT_EQ(after_second.prefix_request.matched_tokens,
                  static_cast<int>(prompt_tokens.size()));
        EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
        EXPECT_TRUE(after_second.prefix_request.terminal_hidden_restored);
        EXPECT_TRUE(after_second.prefix_request.mtp_state_restored);
        EXPECT_FALSE(after_second.mtp_bypassed) << after_second.mtp_bypass_reason;
        // MTP counters are request-local: prove the restored-prefix request
        // still ran the verifier instead of expecting cumulative growth.
        EXPECT_GE(after_second.mtp_draft_steps, expected_first_step_drafts);
        if (expected_first_step_drafts > 0)
        {
            EXPECT_GE(after_second.mtp_verifier_runs, 1u);
            EXPECT_GE(after_second.mtp_verifier_token_count, expected_first_step_drafts + 1);
        }
        expectPhase138TransactionUsed(
            test_case,
            after_second,
            test_case.name + " restored request");
    }

    inline void runDensePhase138VllmStyleCandidateEquivalence(
        DensePrefixRestoreParityCase test_case,
        int mtp_draft_tokens = 2)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);

        test_case.name += " Phase13.8 vllm-style candidate equivalence";
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        ScopedEnvironmentValues phase138_env({
            {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
            {"LLAMINAR_MTP_PHASE138_EQUIVALENCE_CHECK", "1"},
        });

        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/false,
                /*block_size=*/2,
                /*enable_mtp=*/true,
                mtp_draft_tokens));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        auto result = runner->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto snapshot = runner->prefixStateProbe();
        runner->shutdown();

        ASSERT_TRUE(result.error.empty()) << result.error;
        ASSERT_EQ(result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(result.tokens, expected_tokens);
        EXPECT_GT(snapshot.mtp_transaction_commits, 0u)
            << "Phase 13.8 candidate did not commit any MTP transactions";
        EXPECT_EQ(snapshot.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(snapshot.mtp_transaction_validation_failures, 0u);
    }

    inline void runDensePhase138VllmStyleCandidatePrefixRestoreEquivalence(
        DensePrefixRestoreParityCase test_case,
        int mtp_draft_tokens = 2)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);

        test_case.name += " Phase13.8 vllm-style candidate prefix restore equivalence";
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        ScopedEnvironmentValues phase138_env({
            {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
            {"LLAMINAR_MTP_PHASE138_EQUIVALENCE_CHECK", "1"},
        });

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/false,
                /*block_size=*/static_cast<int>(prompt_tokens.size()),
                /*enable_mtp=*/false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        baseline->shutdown();
        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_result.tokens, expected_tokens);

        auto runner = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/true,
                /*block_size=*/static_cast<int>(prompt_tokens.size()),
                /*enable_mtp=*/true,
                mtp_draft_tokens));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        auto first = runner->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_first = runner->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), expected_tokens.size());
        EXPECT_EQ(first.tokens, expected_tokens);
        EXPECT_EQ(first.tokens, baseline_result.tokens);
        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);
        expectPhase138TransactionUsed(
            test_case,
            after_first,
            test_case.name + " first request");

        auto second = runner->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_second = runner->prefixStateProbe();
        runner->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), expected_tokens.size());
        EXPECT_EQ(second.tokens, expected_tokens);
        EXPECT_EQ(second.tokens, baseline_result.tokens);
        EXPECT_TRUE(after_second.prefix_cache_ready);
        EXPECT_GE(after_second.prefix_cache_hits, 1u);
        EXPECT_TRUE(after_second.prefix_request.hit);
        EXPECT_EQ(after_second.prefix_request.matched_tokens,
                  static_cast<int>(prompt_tokens.size()));
        EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
        EXPECT_TRUE(after_second.prefix_request.terminal_hidden_restored);
        EXPECT_TRUE(after_second.prefix_request.mtp_state_restored);
        expectPhase138TransactionUsed(
            test_case,
            after_second,
            test_case.name + " restored request");
    }

    inline void runDensePhase138VllmStyleCandidateStopTokenEquivalence(
        DensePrefixRestoreParityCase test_case,
        int mtp_draft_tokens = 2,
        int stop_token_index = 1)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);
        ASSERT_GE(stop_token_index, 1);

        test_case.name += " Phase13.8 vllm-style candidate stop-token equivalence";
        test_case.decode_steps = std::max(test_case.decode_steps, stop_token_index + 2);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_stop_snapshots/metadata.txt";

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_LT(static_cast<size_t>(stop_token_index), expected_tokens.size());

        const std::vector<int32_t> stop_tokens = {
            expected_tokens[static_cast<size_t>(stop_token_index)],
        };

        ScopedEnvironmentValues phase138_env({
            {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
            {"LLAMINAR_MTP_PHASE138_EQUIVALENCE_CHECK", "1"},
        });

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/false,
                /*block_size=*/2,
                /*enable_mtp=*/false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        baseline->setStopTokens(stop_tokens);
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_TRUE(baseline_result.is_complete)
            << "baseline did not stop on the selected reference token";
        ASSERT_EQ(baseline_result.tokens.size(), static_cast<size_t>(stop_token_index + 1));
        EXPECT_EQ(baseline_result.tokens,
                  std::vector<int32_t>(
                      expected_tokens.begin(),
                      expected_tokens.begin() + stop_token_index + 1));

        auto runner = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/false,
                /*block_size=*/2,
                /*enable_mtp=*/true,
                mtp_draft_tokens));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        runner->setStopTokens(stop_tokens);

        auto result = runner->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto snapshot = runner->prefixStateProbe();
        runner->shutdown();

        ASSERT_TRUE(result.error.empty()) << result.error;
        EXPECT_TRUE(result.is_complete)
            << "Phase 13.8 candidate did not stop on the selected reference token";
        EXPECT_EQ(result.tokens, baseline_result.tokens);
        EXPECT_GT(snapshot.mtp_transaction_commits, 0u)
            << "Phase 13.8 stop-token candidate did not commit any MTP transactions";
        EXPECT_EQ(snapshot.mtp_transaction_validation_failures, 0u);
        expectPhase138TransactionUsed(
            test_case,
            snapshot,
            test_case.name + " stop-token request",
            /*allow_transaction_rollbacks=*/true);
    }

    inline void runDensePhase138VllmStyleCandidateContinuationEquivalence(
        DensePrefixRestoreParityCase test_case,
        int mtp_draft_tokens = 3,
        int decode_steps = 8)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);
        ASSERT_GE(decode_steps, mtp_draft_tokens + 2);

        test_case.name += " Phase13.8 vllm-style candidate continuation equivalence";
        test_case.decode_steps = std::max(test_case.decode_steps, decode_steps);
        test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        ScopedEnvironmentValues phase138_env({
            {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
            {"LLAMINAR_MTP_PHASE138_EQUIVALENCE_CHECK", "1"},
        });

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/false,
                /*block_size=*/2,
                /*enable_mtp=*/false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_result.tokens, expected_tokens);

        auto runner = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/false,
                /*block_size=*/2,
                /*enable_mtp=*/true,
                mtp_draft_tokens));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        auto result = runner->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto snapshot = runner->prefixStateProbe();
        runner->shutdown();

        ASSERT_TRUE(result.error.empty()) << result.error;
        EXPECT_EQ(result.tokens, baseline_result.tokens);
        EXPECT_EQ(result.tokens, expected_tokens);
        EXPECT_GT(snapshot.mtp_transaction_commits, 0u)
            << "Phase 13.8 continuation candidate did not commit any MTP transactions";
        EXPECT_GE(snapshot.mtp_verifier_runs, 1u);
        EXPECT_GE(snapshot.mtp_verifier_token_count,
                  static_cast<uint64_t>(std::min(mtp_draft_tokens, test_case.decode_steps - 1) + 1));
        EXPECT_EQ(snapshot.mtp_transaction_validation_failures, 0u);
        expectPhase138TransactionUsed(
            test_case,
            snapshot,
            test_case.name + " continuation request",
            /*allow_transaction_rollbacks=*/true);
    }

    inline void runDenseMTPShiftedRowCommitPreservesVerifierForward(
        DensePrefixRestoreParityCase test_case)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));

        test_case.name += " shifted-row commit preserves verifier forward";
        test_case.decode_steps = std::max(test_case.decode_steps, 8);
        test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GE(expected_tokens.size(), 7u);
        ASSERT_EQ(expected_tokens[0], 13);
        ASSERT_EQ(expected_tokens[1], 271);
        ASSERT_EQ(expected_tokens[2], 248068);
        ASSERT_EQ(expected_tokens[3], 198);
        ASSERT_EQ(expected_tokens[4], 8160);
        ASSERT_EQ(expected_tokens[5], 579);
        ASSERT_EQ(expected_tokens[6], 264);

        DeviceManager::instance().initialize(-1);
        auto model_ctx = ModelContext::create(
            model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 3;

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();
        auto runner = createInferenceRunner(
            model_ctx,
            nullptr,
            device,
            config);
        ASSERT_NE(runner, nullptr);
        runner->setSuppressTimeline(true);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())));
        EXPECT_EQ(runner->sampleGreedyOnDevice(), expected_tokens[0])
            << "Prefill sample drifted before shifted-row commit regression";

        for (int step = 0; step < 3; ++step)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(step)];
            const int position_before_forward = runner->get_position();
            ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
                token,
                /*already_appended_tokens=*/0,
                /*allow_speculative_discard=*/true,
                position_before_forward))
                << "Failed to maintain shifted MTP cache before setup token "
                << step << " token=" << token;
            ASSERT_TRUE(runner->forward(&token, 1))
                << "Failed to replay setup token " << step
                << " token=" << token;
            const int32_t sampled = runner->sampleGreedyOnDevice();
            ASSERT_EQ(sampled, expected_tokens[static_cast<size_t>(step + 1)])
                << "Setup replay drifted before the shifted-row isolation check"
                << "\nstep: " << step
                << "\ntoken: " << token;
        }

        const int base_position = runner->get_position();
        const PrefixStateSnapshot base_checkpoint = runner->captureLivePrefixState();
        ASSERT_TRUE(base_checkpoint.valid);
        const int32_t verifier_token = expected_tokens[3];
        const int32_t expected_next = expected_tokens[4];

        ASSERT_TRUE(runner->forward(&verifier_token, 1))
            << "Clean verifier forward failed";
        const int32_t clean_next = runner->sampleGreedyOnDevice();
        ASSERT_EQ(clean_next, expected_next)
            << "Clean verifier forward no longer matches PyTorch";

        ASSERT_TRUE(runner->restoreLivePrefixState(base_checkpoint));
        ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
            verifier_token,
            /*already_appended_tokens=*/0,
            /*allow_speculative_discard=*/true,
            base_position))
            << "Shifted-row commit failed at the verifier isolation point";
        ASSERT_TRUE(runner->forward(&verifier_token, 1))
            << "Verifier forward after shifted-row commit failed";
        const int32_t after_shifted_commit_next = runner->sampleGreedyOnDevice();

        EXPECT_EQ(after_shifted_commit_next, clean_next)
            << "MTP shifted-row commit must not alter the main verifier "
               "forward result"
            << "\nbase_position: " << base_position
            << "\nverifier token: " << verifier_token
            << "\nclean next: " << clean_next
            << "\nafter shifted commit next: "
            << after_shifted_commit_next;
        EXPECT_EQ(after_shifted_commit_next, expected_next)
            << "Verifier forward after shifted-row commit drifted from "
               "PyTorch continuation";

        ASSERT_TRUE(runner->restoreLivePrefixState(base_checkpoint));
        std::array<int32_t, 3> chained_sidecar_tokens = {-1, -1, -1};
        ASSERT_TRUE(runner->forwardMTPAndSampleGreedy(
            verifier_token,
            &chained_sidecar_tokens[0]))
            << "Initial chained sidecar draft failed";
        ASSERT_TRUE(runner->flushPendingMTPWork());
        ASSERT_EQ(chained_sidecar_tokens[0], expected_tokens[4]);
        for (int draft_idx = 1; draft_idx < 3; ++draft_idx)
        {
            ASSERT_TRUE(runner->forwardMTPFromLastDraftAndSampleGreedy(
                chained_sidecar_tokens[static_cast<size_t>(draft_idx - 1)],
                base_position + draft_idx,
                &chained_sidecar_tokens[static_cast<size_t>(draft_idx)]))
                << "Chained sidecar draft failed at depth " << draft_idx;
            ASSERT_TRUE(runner->flushPendingMTPWork());
            ASSERT_EQ(chained_sidecar_tokens[static_cast<size_t>(draft_idx)],
                      expected_tokens[static_cast<size_t>(4 + draft_idx)])
                << "Chained sidecar draft drifted at depth " << draft_idx;
        }

        ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
            verifier_token,
            /*already_appended_tokens=*/0,
            /*allow_speculative_discard=*/true,
            base_position))
            << "Shifted-row commit failed after chained sidecar drafts";
        ASSERT_TRUE(runner->forward(&verifier_token, 1))
            << "Verifier forward after chained sidecar shifted-row commit failed";
        const int32_t after_chained_sidecar_next = runner->sampleGreedyOnDevice();
        EXPECT_EQ(after_chained_sidecar_next, clean_next)
            << "Chained MTP sidecar drafts must not alter the main verifier "
               "forward result after a shifted-row commit"
            << "\nbase_position: " << base_position
            << "\nverifier token: " << verifier_token
            << "\nsidecar drafts: "
            << chained_sidecar_tokens[0] << ','
            << chained_sidecar_tokens[1] << ','
            << chained_sidecar_tokens[2]
            << "\nclean next: " << clean_next
            << "\nafter chained sidecar next: "
            << after_chained_sidecar_next;

        ASSERT_TRUE(runner->restoreLivePrefixState(base_checkpoint));
        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
    }

    inline void runDenseMTPFirstTransactionLeavesSequentialState(
        DensePrefixRestoreParityCase test_case)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));

        test_case.name += " first MTP transaction leaves sequential state";
        test_case.decode_steps = std::max(test_case.decode_steps, 8);
        test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GE(expected_tokens.size(), 5u);
        ASSERT_EQ(expected_tokens[0], 13);
        ASSERT_EQ(expected_tokens[1], 271);
        ASSERT_EQ(expected_tokens[2], 248068);
        ASSERT_EQ(expected_tokens[3], 198);
        ASSERT_EQ(expected_tokens[4], 8160);

        DeviceManager::instance().initialize(-1);
        auto model_ctx = ModelContext::create(
            model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 3;

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();
        auto runner = createInferenceRunner(
            model_ctx,
            nullptr,
            device,
            config);
        ASSERT_NE(runner, nullptr);
        runner->setSuppressTimeline(true);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())));
        ASSERT_EQ(runner->sampleGreedyOnDevice(), expected_tokens[0]);

        const int base_position = runner->get_position();
        const PrefixStateSnapshot base_checkpoint = runner->captureLivePrefixState();
        ASSERT_TRUE(base_checkpoint.valid);

        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {13, 271, 760, 3841};
        request.base_sidecar_position = base_position;
        request.allow_speculative_discard = true;
        request.verifier_path = "phase138_first_transaction_regression";
        request.verifier_base_checkpoint = &base_checkpoint;

        auto sample_after_forward = [&]() -> int32_t
        {
            return runner->sampleGreedyOnDevice();
        };

        MTPDecodeCatchupGreedyResult shared =
            runSharedStepwiseMTPDecodeCatchupGreedy(
                *runner,
                request,
                sample_after_forward);
        ASSERT_TRUE(shared.ok) << shared.error;
        EXPECT_EQ(shared.accepted_tokens,
                  (std::vector<int32_t>{13, 271, 248068}));
        EXPECT_EQ(shared.ready_token, expected_tokens[3]);
        ASSERT_TRUE(runner->forward(&expected_tokens[3], 1));
        const int32_t shared_next = runner->sampleGreedyOnDevice();
        EXPECT_EQ(shared_next, expected_tokens[4])
            << "Shared stepwise catch-up must leave the main runner in the "
               "same state as sequential decode after the first transaction";

        ASSERT_TRUE(runner->restoreLivePrefixState(base_checkpoint));
        PerfStatsCollector::reset();
        ASSERT_TRUE(PerfStatsCollector::isEnabled())
            << "Phase 13.8 fallback-counter regression requires perf stats";
        {
            ScopedEnvironmentValues candidate_env({
                {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
                {"LLAMINAR_MTP_PHASE138_DIRECT_CANDIDATE", "1"},
            });
            MTPDecodeCatchupGreedyResult candidate =
                runner->runOptimizedMTPDecodeCatchupGreedy(
                    request,
                    sample_after_forward);
            ASSERT_TRUE(candidate.ok) << candidate.error;
            EXPECT_EQ(candidate.accepted_tokens,
                      (std::vector<int32_t>{13, 271, 248068}));
            EXPECT_EQ(candidate.ready_token, expected_tokens[3]);
            ASSERT_TRUE(runner->forward(&expected_tokens[3], 1));
            const int32_t candidate_next = runner->sampleGreedyOnDevice();
            EXPECT_EQ(candidate_next, expected_tokens[4])
                << "vLLM-style candidate wrapper must leave the main runner "
                   "in the same state as sequential decode after publishing "
                   "accepted-count state and forwarding the rejected "
                   "correction suffix"
                << "\naccepted: "
                << candidate.accepted_tokens[0] << ','
                << candidate.accepted_tokens[1] << ','
                << candidate.accepted_tokens[2]
                << "\nready token: " << candidate.ready_token
                << "\nnext after ready: " << candidate_next;

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            auto has_counter = [&](const char *name,
                                   const char *tag_key = nullptr,
                                   const char *tag_value = nullptr) -> bool
            {
                return std::any_of(
                    records.begin(),
                    records.end(),
                    [&](const PerfStatRecord &record)
                    {
                        if (record.kind != PerfStatRecord::Kind::Counter ||
                            record.domain != "mtp" ||
                            record.name != name)
                        {
                            return false;
                        }
                        if (!tag_key)
                            return true;
                        auto it = record.tags.find(tag_key);
                        return it != record.tags.end() &&
                               it->second == tag_value;
                    });
            };
            EXPECT_TRUE(has_counter("phase138_vllm_style_correction_forward_tokens"))
                << "Rejected Phase 13.8 transactions must advance main state by "
                   "forwarding the correction suffix, not by deferring it to "
                   "the next outer decode step\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_TRUE(has_counter("phase138_vllm_style_spec_decode_runs",
                                    "suffix_replay_tokens",
                                    "1"))
                << PerfStatsCollector::summaryString({"mtp"});
        }

        ASSERT_TRUE(runner->restoreLivePrefixState(base_checkpoint));
        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
    }

    inline void runDenseNoMTPPhase138ContinuationMatchesPyTorch(
        DensePrefixRestoreParityCase test_case,
        int decode_steps = 8)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_GE(decode_steps, 1);

        test_case.name += " Phase13.8 no-MTP continuation parity";
        test_case.decode_steps = std::max(test_case.decode_steps, decode_steps);
        test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto run_manual_decode = [&]() -> GenerationResult
        {
            GenerationResult result;
            auto runner = factory->createFromOrchestrationConfig(
                makeDensePrefixRestoreConfig(
                    test_case,
                    model_path,
                    /*enable_prefix_cache=*/false,
                    /*block_size=*/2,
                    /*enable_mtp=*/false));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                result.error = "failed to create manual decode runner";
                return result;
            }
            if (!runner->initialize())
            {
                result.error = runner->lastError();
                return result;
            }
            runner->setSuppressTimeline(true);
            runner->setSamplingParams(greedy);
            if (!runner->prefill(prompt_tokens))
            {
                result.error = runner->lastError();
                runner->shutdown();
                return result;
            }
            while (static_cast<int>(result.tokens.size()) < test_case.decode_steps)
            {
                runner->setDecodeStepTokenBudget(
                    test_case.decode_steps - static_cast<int>(result.tokens.size()));
                GenerationResult step = runner->decodeStep();
                runner->setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    result.error = step.error;
                    break;
                }
                result.tokens.insert(result.tokens.end(), step.tokens.begin(), step.tokens.end());
                if (step.is_complete)
                {
                    result.is_complete = true;
                    break;
                }
            }
            runner->shutdown();
            return result;
        };

        auto run_generate = [&]() -> GenerationResult
        {
            auto runner = factory->createFromOrchestrationConfig(
                makeDensePrefixRestoreConfig(
                    test_case,
                    model_path,
                    /*enable_prefix_cache=*/false,
                    /*block_size=*/2,
                    /*enable_mtp=*/false));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                GenerationResult result;
                result.error = "failed to create generate runner";
                return result;
            }
            if (!runner->initialize())
            {
                GenerationResult result;
                result.error = runner->lastError();
                return result;
            }
            runner->setSuppressTimeline(true);
            auto result = runner->generate(prompt_tokens, test_case.decode_steps, greedy);
            runner->shutdown();
            return result;
        };

        const auto manual_result = run_manual_decode();
        ASSERT_TRUE(manual_result.error.empty()) << manual_result.error;
        ASSERT_EQ(manual_result.tokens.size(), expected_tokens.size());
        EXPECT_TRUE(tokenSequencesMatch(
            manual_result.tokens,
            expected_tokens,
            test_case.name + " manual decodeStep loop"))
            << "\nWrapper trace=manual_decodeStep_loop";

        const auto generate_result = run_generate();
        ASSERT_TRUE(generate_result.error.empty()) << generate_result.error;
        ASSERT_EQ(generate_result.tokens.size(), expected_tokens.size());
        EXPECT_TRUE(tokenSequencesMatch(
            generate_result.tokens,
            expected_tokens,
            test_case.name + " generate path"))
            << "\nWrapper trace=generate_path";
    }

    inline void runDenseNoMTPPhase138ThinkContinuationStageParity(
        DensePrefixRestoreParityCase test_case)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));

        test_case.name += " Phase13.8 no-MTP think continuation stage parity";
        test_case.decode_steps = std::max(test_case.decode_steps, 8);
        test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "0"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GE(expected_tokens.size(), 4u);
        ASSERT_EQ(expected_tokens[2], 248068)
            << "The Phase 13.8 continuation fixture no longer reaches the "
               "known <think> token at decode index 2; update this regression "
               "with the new first divergent row.";

        const std::filesystem::path metadata_path = firstEnvOrDefault(
            test_case.metadata_envs,
            test_case.default_metadata_path);
        const std::filesystem::path snapshot_dir = metadata_path.parent_path();
        ensurePyTorchDecodeSnapshots(test_case, model_path, metadata_path);

        DeviceManager::instance().initialize(-1);
        auto model_ctx = ModelContext::create(
            model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = false;

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();
        auto runner = createInferenceRunner(
            model_ctx,
            nullptr,
            device,
            config);
        ASSERT_NE(runner, nullptr);
        runner->setSuppressTimeline(true);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())));
        const int32_t prefill_sample = runner->sampleGreedyOnDevice();
        ASSERT_EQ(prefill_sample, expected_tokens[0])
            << "Prefill sample drifted before the Phase 13.8 continuation window";

        runner->enableSnapshotCapture();
        for (int step = 0; step < 3; ++step)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(step)];
            runner->clearSnapshots();
            ASSERT_TRUE(runner->forward(&token, 1))
                << "Decode forward failed at step " << step;
            const auto step_snapshots = captureDenseStageSnapshots(*runner);
            const int32_t sampled = runner->sampleGreedyOnDevice();
            const auto stage_match = denseDecodeStepSnapshotsNearPyTorch(
                step_snapshots,
                snapshot_dir,
                step,
                test_case.name);
            EXPECT_TRUE(stage_match)
                << stage_match.message()
                << "\nstep: " << step
                << "\ninput token: " << token
                << "\nactual sampled next: " << sampled
                << "\nexpected sampled next: " << expected_tokens[static_cast<size_t>(step + 1)];
            EXPECT_EQ(sampled, expected_tokens[static_cast<size_t>(step + 1)])
                << "Direct no-MTP runner diverged at Phase 13.8 continuation "
                   "decode step "
                << step;
        }

        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
    }

    inline void runDenseDynamicMTPParity(
        DensePrefixRestoreParityCase test_case,
        bool enable_prefix_cache = false)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        const int adaptive_max_depth = enable_prefix_cache ? 1 : 2;
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = adaptive_max_depth;
        depth_policy.initial_depth = adaptive_max_depth;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        const int block_size = enable_prefix_cache
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 2;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, block_size, false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_result.tokens, expected_tokens);

        auto mtp = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                enable_prefix_cache,
                block_size,
                true,
                adaptive_max_depth,
                depth_policy));
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();

        auto first = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_first = mtp->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), expected_tokens.size());
        EXPECT_EQ(first.tokens, expected_tokens);
        EXPECT_EQ(first.tokens, baseline_result.tokens);
        EXPECT_FALSE(after_first.mtp_bypassed) << after_first.mtp_bypass_reason;
        EXPECT_TRUE(after_first.mtp_request.adaptive_depth_enabled);
        EXPECT_EQ(after_first.mtp_request.depth_policy_mode, "dynamic");
        EXPECT_GE(after_first.mtp_depth_policy_windows, 1u);
        EXPECT_GE(after_first.mtp_min_depth, 1);
        EXPECT_EQ(after_first.mtp_max_depth, adaptive_max_depth);
        EXPECT_GE(after_first.mtp_current_depth, 1);
        EXPECT_LE(after_first.mtp_current_depth, adaptive_max_depth);

        if (!enable_prefix_cache)
        {
            mtp->shutdown();
            return;
        }

        auto second = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_second = mtp->prefixStateProbe();
        mtp->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), expected_tokens.size());
        EXPECT_EQ(second.tokens, expected_tokens);
        EXPECT_EQ(second.tokens, baseline_result.tokens);
        EXPECT_TRUE(after_second.prefix_request.hit);
        EXPECT_TRUE(after_second.prefix_request.mtp_state_restored);
        EXPECT_FALSE(after_second.mtp_bypassed) << after_second.mtp_bypass_reason;
        EXPECT_GE(after_second.mtp_depth_policy_windows, 1u);
    }

    inline void runDenseNoMTPBenchmarkStyleFreshRunnerDeterminism(
        DensePrefixRestoreParityCase test_case,
        int decode_token_budget = 16,
        int reused_cycle_count = 2)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_GT(decode_token_budget, 0);
        ASSERT_GT(reused_cycle_count, 0);

        test_case.name += " benchmark-style no-MTP fresh-runner determinism";
        test_case.prompt = qwen36DefaultBenchmarkPrompt();
        test_case.decode_steps = std::max(test_case.decode_steps, decode_token_budget);
        test_case.max_seq_len = 768;

        const char *gpu_graphs_override =
            std::getenv("LLAMINAR_QWEN36_DENSE_BENCHMARK_PARITY_GPU_GRAPHS");
        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS",
             (gpu_graphs_override && *gpu_graphs_override) ? gpu_graphs_override : "1"},
        });

        if (auto skip_reason = densePrefixParitySkipReason(test_case))
        {
            GTEST_SKIP() << *skip_reason;
        }

        const std::string model_path = firstEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << model_path;
        }

        auto model_context = ModelContext::create(model_path);
        ASSERT_NE(model_context, nullptr);
        auto tokenizer = createTokenizer(model_context);
        ASSERT_NE(tokenizer, nullptr);
        const std::vector<int> encoded_prompt =
            tokenizer->encode(test_case.prompt, /*add_bos=*/false, /*add_eos=*/false);
        ASSERT_FALSE(encoded_prompt.empty());
        std::vector<int32_t> prompt_tokens(encoded_prompt.begin(), encoded_prompt.end());
        ASSERT_LT(
            static_cast<int>(prompt_tokens.size()) + decode_token_budget,
            test_case.max_seq_len);

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        greedy.seed = 42;

        struct Trace
        {
            std::vector<int32_t> tokens;
            std::vector<int> gathered_argmax;
            std::vector<std::string> topk;
            std::string error;
        };

        auto trace_string = [](const Trace &trace) -> std::string
        {
            std::ostringstream oss;
            oss << "tokens={" << denseJoinTokens(trace.tokens) << "}";
            for (size_t i = 0; i < trace.topk.size(); ++i)
            {
                oss << "\n  step " << i
                    << " sampled="
                    << (i < trace.tokens.size() ? trace.tokens[i] : -1)
                    << " gathered_argmax="
                    << (i < trace.gathered_argmax.size() ? trace.gathered_argmax[i] : -1)
                    << " topk=[" << trace.topk[i] << "]";
            }
            return oss.str();
        };

        auto run_once = [&](int repetition, bool gather_logits) -> Trace
        {
            Trace trace;
            auto runner = factory->createFromOrchestrationConfig(
                makeDensePrefixRestoreConfig(test_case, model_path, false, 2, false));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                trace.error = "failed to create runner";
                return trace;
            }

            if (!runner->initialize())
            {
                trace.error = runner->lastError();
                return trace;
            }
            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherPrefill(true);
            runner->setSkipLogitsGatherDecode(true);

            if (!runner->prefill(prompt_tokens))
            {
                trace.error = runner->lastError();
                runner->shutdown();
                return trace;
            }

            const int vocab_size = runner->vocabSize();
            if (vocab_size <= 0)
            {
                trace.error = "invalid vocab size";
                runner->shutdown();
                return trace;
            }

            for (int produced = 0; produced < decode_token_budget; ++produced)
            {
                runner->setDecodeStepTokenBudget(decode_token_budget - produced);
                GenerationResult step = runner->decodeStep();
                runner->setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    trace.error = step.error;
                    break;
                }
                if (step.tokens.size() != 1u)
                {
                    std::ostringstream oss;
                    oss << "repetition " << repetition
                        << " produced " << step.tokens.size()
                        << " tokens for one no-MTP decode step";
                    trace.error = oss.str();
                    break;
                }
                const int32_t token = step.tokens.front();
                trace.tokens.push_back(token);

                if (gather_logits)
                {
                    const float *logits = runner->lastLogits();
                    if (!logits)
                    {
                        trace.error = "benchmark-style no-MTP decode produced no gathered logits";
                        break;
                    }
                    const int gathered_argmax = denseArgmaxToken(logits, vocab_size);
                    trace.gathered_argmax.push_back(gathered_argmax);
                    trace.topk.push_back(denseTopKSummary(logits, vocab_size));
                    if (token != gathered_argmax)
                    {
                        std::ostringstream oss;
                        oss << "GPU greedy sample does not match gathered logits argmax"
                            << " at repetition " << repetition
                            << " step " << produced
                            << ": sampled=" << token
                            << " gathered_argmax=" << gathered_argmax
                            << "\ntop-k: " << trace.topk.back();
                        trace.error = oss.str();
                        break;
                    }
                }
            }

            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return trace;
        };

        auto run_reused_cycles = [&]() -> std::vector<Trace>
        {
            std::vector<Trace> traces;
            auto runner = factory->createFromOrchestrationConfig(
                makeDensePrefixRestoreConfig(test_case, model_path, false, 2, false));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return traces;
            }

            if (!runner->initialize())
            {
                Trace trace;
                trace.error = runner->lastError();
                traces.push_back(std::move(trace));
                return traces;
            }

            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherPrefill(true);
            runner->setSkipLogitsGatherDecode(true);

            for (int cycle = 0; cycle < reused_cycle_count; ++cycle)
            {
                runner->clearCache();
                Trace trace;
                if (!runner->prefill(prompt_tokens))
                {
                    trace.error = runner->lastError();
                    traces.push_back(std::move(trace));
                    break;
                }

                while (static_cast<int>(trace.tokens.size()) < decode_token_budget)
                {
                    const int remaining = decode_token_budget - static_cast<int>(trace.tokens.size());
                    runner->setDecodeStepTokenBudget(remaining);
                    GenerationResult step = runner->decodeStep();
                    runner->setDecodeStepTokenBudget(0);
                    if (!step.error.empty())
                    {
                        trace.error = step.error;
                        break;
                    }
                    if (step.tokens.empty())
                    {
                        trace.error = "reused no-MTP production decode produced no tokens";
                        break;
                    }
                    if (step.tokens.size() > static_cast<size_t>(remaining))
                    {
                        std::ostringstream oss;
                        oss << "reused no-MTP production decode exceeded budget: "
                            << step.tokens.size() << " > " << remaining;
                        trace.error = oss.str();
                        break;
                    }
                    trace.tokens.insert(trace.tokens.end(), step.tokens.begin(), step.tokens.end());
                }
                traces.push_back(std::move(trace));
            }

            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return traces;
        };

        const Trace first = run_once(0, /*gather_logits=*/false);
        ASSERT_TRUE(first.error.empty()) << first.error << "\n" << trace_string(first);
        ASSERT_EQ(first.tokens.size(), static_cast<size_t>(decode_token_budget))
            << trace_string(first);

        const Trace second = run_once(1, /*gather_logits=*/false);
        ASSERT_TRUE(second.error.empty()) << second.error << "\n" << trace_string(second);
        ASSERT_EQ(second.tokens.size(), first.tokens.size())
            << "first:\n"
            << trace_string(first)
            << "\nsecond:\n"
            << trace_string(second);
        EXPECT_EQ(second.tokens, first.tokens)
            << "Qwen3.6 dense CUDA benchmark-style no-MTP decode must be "
            << "fresh-runner deterministic before MTP parity can be trusted."
            << "\nfirst:\n"
            << trace_string(first)
            << "\nsecond:\n"
            << trace_string(second);

        const Trace gathered = run_once(2, /*gather_logits=*/true);
        ASSERT_TRUE(gathered.error.empty()) << gathered.error << "\n" << trace_string(gathered);
        ASSERT_EQ(gathered.tokens.size(), first.tokens.size())
            << "first:\n"
            << trace_string(first)
            << "\ngathered:\n"
            << trace_string(gathered);
        EXPECT_EQ(gathered.tokens, first.tokens)
            << "Gathering logits for diagnostics must not change CUDA no-MTP decode tokens."
            << "\nfirst:\n"
            << trace_string(first)
            << "\ngathered:\n"
            << trace_string(gathered);

        const auto reused_cycles = run_reused_cycles();
        ASSERT_EQ(reused_cycles.size(), static_cast<size_t>(reused_cycle_count));
        for (size_t i = 0; i < reused_cycles.size(); ++i)
        {
            ASSERT_TRUE(reused_cycles[i].error.empty())
                << reused_cycles[i].error << "\n" << trace_string(reused_cycles[i]);
            ASSERT_EQ(reused_cycles[i].tokens.size(), first.tokens.size())
                << "first:\n"
                << trace_string(first)
                << "\ncycle:\n"
                << trace_string(reused_cycles[i]);
            EXPECT_EQ(reused_cycles[i].tokens, first.tokens)
                << "clearCache() must reset CUDA no-MTP production decode state "
                << "without relying on logits gather."
                << "\nfirst:\n"
                << trace_string(first)
                << "\ncycle " << i << ":\n"
                << trace_string(reused_cycles[i]);
        }
    }

    inline void runDenseBenchmarkStyleDynamicMTPParity(
        DensePrefixRestoreParityCase test_case)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        test_case.name += " benchmark-style dynamic MTP parity";
        test_case.prompt = qwen36DefaultBenchmarkPrompt();
        test_case.decode_steps = 128;
        test_case.max_seq_len = 768;

        const char *gpu_graphs_override =
            std::getenv("LLAMINAR_QWEN36_DENSE_BENCHMARK_PARITY_GPU_GRAPHS");
        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS",
             (gpu_graphs_override && *gpu_graphs_override) ? gpu_graphs_override : "1"},
        });

        std::string model_path;
        if (auto skip_reason = densePrefixParitySkipReason(test_case))
        {
            GTEST_SKIP() << *skip_reason;
        }

        model_path = firstEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << model_path;
        }

        auto model_context = ModelContext::create(model_path);
        ASSERT_NE(model_context, nullptr);
        auto tokenizer = createTokenizer(model_context);
        ASSERT_NE(tokenizer, nullptr);

        std::vector<int> encoded_prompt =
            tokenizer->encode(test_case.prompt, /*add_bos=*/false, /*add_eos=*/false);
        ASSERT_FALSE(encoded_prompt.empty());
        std::vector<int32_t> prompt_tokens(
            encoded_prompt.begin(),
            encoded_prompt.end());
        ASSERT_LT(
            static_cast<int>(prompt_tokens.size()) + test_case.decode_steps,
            test_case.max_seq_len);

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        greedy.seed = 42;

        auto run_decode = [&](bool enable_mtp,
                              int mtp_draft_tokens,
                              MTPDepthPolicyConfig depth_policy,
                              PrefixRuntimeStateSnapshot *out_state) -> std::vector<int32_t>
        {
            std::vector<int32_t> tokens;
            auto runner = factory->createFromOrchestrationConfig(
                makeDensePrefixRestoreConfig(
                    test_case,
                    model_path,
                    false,
                    2,
                    enable_mtp,
                    mtp_draft_tokens,
                    depth_policy));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return tokens;
            }

            if (!runner->initialize())
            {
                ADD_FAILURE() << runner->lastError();
                return tokens;
            }

            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherPrefill(true);
            runner->setSkipLogitsGatherDecode(true);

            if (!runner->prefill(prompt_tokens))
            {
                ADD_FAILURE() << runner->lastError();
                runner->shutdown();
                return tokens;
            }

            while (static_cast<int>(tokens.size()) < test_case.decode_steps)
            {
                const int remaining = test_case.decode_steps - static_cast<int>(tokens.size());
                runner->setDecodeStepTokenBudget(remaining);
                GenerationResult step = runner->decodeStep();
                runner->setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    ADD_FAILURE() << step.error;
                    break;
                }
                if (step.tokens.empty())
                {
                    ADD_FAILURE() << "benchmark-style decode produced no tokens";
                    break;
                }
                if (step.tokens.size() > static_cast<size_t>(remaining))
                {
                    ADD_FAILURE()
                        << "benchmark-style decode exceeded remaining token budget: "
                        << step.tokens.size() << " > " << remaining;
                    break;
                }
                tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
            }

            if (out_state)
            {
                *out_state = runner->prefixStateProbe();
            }
            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return tokens;
        };

        auto run_reused_decode_cycles = [&](bool enable_mtp,
                                            int mtp_draft_tokens,
                                            MTPDepthPolicyConfig depth_policy,
                                            int cycles,
                                            PrefixRuntimeStateSnapshot *out_state)
            -> std::vector<std::vector<int32_t>>
        {
            std::vector<std::vector<int32_t>> cycle_tokens;
            auto runner = factory->createFromOrchestrationConfig(
                makeDensePrefixRestoreConfig(
                    test_case,
                    model_path,
                    false,
                    2,
                    enable_mtp,
                    mtp_draft_tokens,
                    depth_policy));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return cycle_tokens;
            }

            if (!runner->initialize())
            {
                ADD_FAILURE() << runner->lastError();
                return cycle_tokens;
            }

            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherPrefill(true);
            runner->setSkipLogitsGatherDecode(true);

            for (int cycle = 0; cycle < cycles; ++cycle)
            {
                runner->clearCache();
                std::vector<int32_t> tokens;
                if (!runner->prefill(prompt_tokens))
                {
                    ADD_FAILURE() << "cycle " << cycle << ": " << runner->lastError();
                    break;
                }

                while (static_cast<int>(tokens.size()) < test_case.decode_steps)
                {
                    const int remaining = test_case.decode_steps - static_cast<int>(tokens.size());
                    runner->setDecodeStepTokenBudget(remaining);
                    GenerationResult step = runner->decodeStep();
                    runner->setDecodeStepTokenBudget(0);
                    if (!step.error.empty())
                    {
                        ADD_FAILURE() << "cycle " << cycle << ": " << step.error;
                        break;
                    }
                    if (step.tokens.empty())
                    {
                        ADD_FAILURE() << "cycle " << cycle
                                      << ": benchmark-style decode produced no tokens";
                        break;
                    }
                    if (step.tokens.size() > static_cast<size_t>(remaining))
                    {
                        ADD_FAILURE()
                            << "cycle " << cycle
                            << ": benchmark-style decode exceeded remaining token budget: "
                            << step.tokens.size() << " > " << remaining;
                        break;
                    }
                    tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
                }
                cycle_tokens.push_back(std::move(tokens));
            }

            if (out_state)
            {
                *out_state = runner->prefixStateProbe();
            }
            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return cycle_tokens;
        };

        PrefixRuntimeStateSnapshot baseline_state;
        const auto baseline_tokens =
            run_decode(false, 1, {}, &baseline_state);
        ASSERT_EQ(baseline_tokens.size(), static_cast<size_t>(test_case.decode_steps));
        EXPECT_EQ(baseline_state.mtp_draft_steps, 0u);

        PrefixRuntimeStateSnapshot baseline_repeat_state;
        const auto baseline_repeat_tokens =
            run_decode(false, 1, {}, &baseline_repeat_state);
        ASSERT_EQ(baseline_repeat_tokens.size(), baseline_tokens.size());
        EXPECT_TRUE(tokenSequencesMatch(
            baseline_repeat_tokens,
            baseline_tokens,
            "fresh no-MTP repeat"));
        EXPECT_EQ(baseline_repeat_state.mtp_draft_steps, 0u);

        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;

        PrefixRuntimeStateSnapshot mtp_state;
        const auto mtp_tokens =
            run_decode(true, 3, depth_policy, &mtp_state);
        ASSERT_EQ(mtp_tokens.size(), baseline_tokens.size());
        EXPECT_TRUE(tokenSequencesMatch(
            mtp_tokens,
            baseline_tokens,
            "fresh dynamic MTP"));
        EXPECT_FALSE(mtp_state.mtp_bypassed) << mtp_state.mtp_bypass_reason;
        EXPECT_TRUE(mtp_state.mtp_request.adaptive_depth_enabled);
        EXPECT_EQ(mtp_state.mtp_request.depth_policy_mode, "dynamic");

        PrefixRuntimeStateSnapshot fixed_depth_state;
        const auto fixed_depth_tokens =
            run_decode(true, 1, {}, &fixed_depth_state);
        ASSERT_EQ(fixed_depth_tokens.size(), baseline_tokens.size());
        EXPECT_TRUE(tokenSequencesMatch(
            fixed_depth_tokens,
            baseline_tokens,
            "fresh fixed-depth MTP"));
        EXPECT_FALSE(fixed_depth_state.mtp_bypassed) << fixed_depth_state.mtp_bypass_reason;

        PrefixRuntimeStateSnapshot reused_baseline_state;
        const auto reused_baseline_cycles =
            run_reused_decode_cycles(false, 1, {}, 4, &reused_baseline_state);
        ASSERT_EQ(reused_baseline_cycles.size(), 4u);
        for (size_t i = 0; i < reused_baseline_cycles.size(); ++i)
        {
            ASSERT_EQ(reused_baseline_cycles[i].size(), baseline_tokens.size())
                << "no-MTP reused-runner cycle " << i
                << " produced the wrong token count";
            EXPECT_TRUE(tokenSequencesMatch(
                reused_baseline_cycles[i],
                baseline_tokens,
                "no-MTP reused-runner cycle " + std::to_string(i)))
                << "no-MTP reused-runner cycle " << i
                << " diverged from the fresh no-MTP benchmark-style baseline";
        }
        EXPECT_EQ(reused_baseline_state.mtp_draft_steps, 0u);

        PrefixRuntimeStateSnapshot reused_dynamic_state;
        const auto reused_dynamic_cycles =
            run_reused_decode_cycles(true, 3, depth_policy, 4, &reused_dynamic_state);
        ASSERT_EQ(reused_dynamic_cycles.size(), 4u);
        for (size_t i = 0; i < reused_dynamic_cycles.size(); ++i)
        {
            ASSERT_EQ(reused_dynamic_cycles[i].size(), baseline_tokens.size())
                << "dynamic MTP reused-runner cycle " << i
                << " produced the wrong token count";
            EXPECT_TRUE(tokenSequencesMatch(
                reused_dynamic_cycles[i],
                baseline_tokens,
                "dynamic MTP reused-runner cycle " + std::to_string(i)))
                << "dynamic MTP reused-runner cycle " << i
                << " diverged from the no-MTP benchmark-style baseline";
        }
        EXPECT_FALSE(reused_dynamic_state.mtp_bypassed)
            << reused_dynamic_state.mtp_bypass_reason;
    }

    inline void runDenseBenchmarkStyleMTPParitySinglePass(
        DensePrefixRestoreParityCase test_case,
        int mtp_draft_tokens,
        MTPDepthPolicyConfig depth_policy,
        const std::string &label)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        test_case.name += " benchmark-style " + label + " MTP parity";
        test_case.prompt = qwen36DefaultBenchmarkPrompt();
        test_case.decode_steps = 128;
        test_case.max_seq_len = 768;
        test_case.metadata_envs = {
            "LLAMINAR_QWEN36_DENSE_BENCHMARK_PARITY_METADATA",
        };
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_benchmark_prompt_snapshots/metadata.txt";

        const char *gpu_graphs_override =
            std::getenv("LLAMINAR_QWEN36_DENSE_BENCHMARK_PARITY_GPU_GRAPHS");
        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS",
             (gpu_graphs_override && *gpu_graphs_override) ? gpu_graphs_override : "1"},
        });

        if (auto skip_reason = densePrefixParitySkipReason(test_case))
        {
            GTEST_SKIP() << *skip_reason;
        }

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        ASSERT_LT(
            static_cast<int>(prompt_tokens.size()) + test_case.decode_steps,
            test_case.max_seq_len);

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        greedy.seed = 42;

        auto run_decode = [&](bool enable_mtp,
                              int draft_tokens,
                              const MTPDepthPolicyConfig &policy,
                              PrefixRuntimeStateSnapshot *out_state) -> std::vector<int32_t>
        {
            std::vector<int32_t> tokens;
            auto runner = factory->createFromOrchestrationConfig(
                makeDensePrefixRestoreConfig(
                    test_case,
                    model_path,
                    false,
                    2,
                    enable_mtp,
                    draft_tokens,
                    policy));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return tokens;
            }

            if (!runner->initialize())
            {
                ADD_FAILURE() << runner->lastError();
                return tokens;
            }

            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherPrefill(true);
            runner->setSkipLogitsGatherDecode(true);

            if (!runner->prefill(prompt_tokens))
            {
                ADD_FAILURE() << runner->lastError();
                runner->shutdown();
                return tokens;
            }

            while (static_cast<int>(tokens.size()) < test_case.decode_steps)
            {
                const int remaining = test_case.decode_steps - static_cast<int>(tokens.size());
                runner->setDecodeStepTokenBudget(remaining);
                GenerationResult step = runner->decodeStep();
                runner->setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    ADD_FAILURE() << step.error;
                    break;
                }
                if (step.tokens.empty())
                {
                    ADD_FAILURE() << label << " benchmark-style decode produced no tokens";
                    break;
                }
                if (step.tokens.size() > static_cast<size_t>(remaining))
                {
                    ADD_FAILURE()
                        << label
                        << " benchmark-style decode exceeded remaining token budget: "
                        << step.tokens.size() << " > " << remaining;
                    break;
                }
                tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
            }

            if (out_state)
            {
                *out_state = runner->prefixStateProbe();
            }
            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return tokens;
        };

        PrefixRuntimeStateSnapshot baseline_state;
        const auto baseline_tokens =
            run_decode(false, 1, {}, &baseline_state);
        ASSERT_EQ(baseline_tokens.size(), static_cast<size_t>(test_case.decode_steps));
        EXPECT_TRUE(tokenSequencesMatch(
            baseline_tokens,
            expected_tokens,
            "fresh no-MTP benchmark-style baseline"));
        EXPECT_EQ(baseline_state.mtp_draft_steps, 0u);

        PrefixRuntimeStateSnapshot mtp_state;
        const auto mtp_tokens =
            run_decode(true, mtp_draft_tokens, depth_policy, &mtp_state);
        ASSERT_EQ(mtp_tokens.size(), expected_tokens.size());
        EXPECT_TRUE(tokenSequencesMatch(
            mtp_tokens,
            expected_tokens,
            "fresh " + label + " MTP"));
        EXPECT_FALSE(mtp_state.mtp_bypassed) << mtp_state.mtp_bypass_reason;
        expectPhase138TransactionUsed(
            test_case,
            mtp_state,
            test_case.name + " " + label);
        if (depth_policy.mode == MTPDepthPolicyMode::Dynamic)
        {
            EXPECT_TRUE(mtp_state.mtp_request.adaptive_depth_enabled);
            EXPECT_EQ(mtp_state.mtp_request.depth_policy_mode, "dynamic");
        }
    }

    inline void runDenseBenchmarkStyleFixedMTPParity(
        DensePrefixRestoreParityCase test_case,
        int mtp_draft_tokens)
    {
        runDenseBenchmarkStyleMTPParitySinglePass(
            std::move(test_case),
            mtp_draft_tokens,
            {},
            "fixed-depth-" + std::to_string(mtp_draft_tokens));
    }

    inline void runDenseBenchmarkStyleDynamicMTPParitySinglePass(
        DensePrefixRestoreParityCase test_case)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        runDenseBenchmarkStyleMTPParitySinglePass(
            std::move(test_case),
            3,
            depth_policy,
            "dynamic-depth");
    }

    inline void runDenseOneRowRestoreLongPrefixMatchesSequential(
        DensePrefixRestoreParityCase test_case)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        test_case.name += " dense long-prefix one-row restore parity";
        test_case.prompt = qwen36DefaultBenchmarkPrompt();
        test_case.decode_steps = 128;
        test_case.max_seq_len = 768;
        test_case.metadata_envs = {
            "LLAMINAR_QWEN36_DENSE_BENCHMARK_PARITY_METADATA",
        };
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_benchmark_prompt_snapshots/metadata.txt";

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "0"},
        });

        if (auto skip_reason = densePrefixParitySkipReason(test_case))
        {
            GTEST_SKIP() << *skip_reason;
        }

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        size_t first_token_index = expected_tokens.size();
        for (size_t i = 1; i + 3 < expected_tokens.size(); ++i)
        {
            if (expected_tokens[i - 1] == 258 &&
                expected_tokens[i] == 10608 &&
                expected_tokens[i + 1] == 20271 &&
                expected_tokens[i + 2] == 92217 &&
                expected_tokens[i + 3] == 48567)
            {
                first_token_index = i;
            }
        }
        ASSERT_LT(first_token_index + 1, expected_tokens.size())
            << "Benchmark metadata no longer contains the known long-prefix "
               "one-row restore window";
        ASSERT_LT(
            static_cast<int>(prompt_tokens.size() + first_token_index + 2),
            test_case.max_seq_len);

        DeviceManager::instance().initialize(-1);
        auto model_ctx = ModelContext::create(
            model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 3;

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();
        auto runner = createInferenceRunner(
            model_ctx,
            nullptr,
            device,
            config);
        ASSERT_NE(runner, nullptr);
        runner->setSuppressTimeline(true);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        ASSERT_TRUE(runner->forward(prompt_tokens.data(), static_cast<int>(prompt_tokens.size())));
        for (size_t i = 0; i < first_token_index; ++i)
        {
            const int32_t token = expected_tokens[i];
            ASSERT_TRUE(runner->forward(&token, 1))
                << "Failed to replay expected token at index " << i;
        }

        const PrefixStateSnapshot prefix_checkpoint = runner->captureLivePrefixState();
        ASSERT_TRUE(prefix_checkpoint.valid);
        runner->enableSnapshotCapture();

        const int32_t token = expected_tokens[first_token_index];
        const int32_t expected_next = expected_tokens[first_token_index + 1];

        runner->clearSnapshots();
        ASSERT_TRUE(runner->forward(&token, 1));
        const auto sequential_snapshots = captureDenseStageSnapshots(*runner);
        const int32_t sequential_next = runner->sampleGreedyOnDevice();
        ASSERT_EQ(sequential_next, expected_next)
            << "Sequential one-row decode no longer matches PyTorch at the "
               "known restore window";

        ASSERT_TRUE(runner->restoreLivePrefixState(prefix_checkpoint));
        runner->clearSnapshots();
        ASSERT_TRUE(runner->forward(&token, 1));
        const auto restored_snapshots = captureDenseStageSnapshots(*runner);
        const int32_t restored_next = runner->sampleGreedyOnDevice();

        const ::testing::AssertionResult stage_match =
            denseVerifierRowSnapshotsNear(
                restored_snapshots,
                sequential_snapshots,
                "Restored one-row decode",
                1,
                0,
                1.0e-5f,
                1.0e-5f);
        EXPECT_TRUE(stage_match)
            << stage_match.message();
        EXPECT_EQ(restored_next, sequential_next)
            << "Restored one-row decode must produce the same token as the "
               "original sequential decode"
            << "\ncondition token: " << expected_tokens[first_token_index - 1]
            << "\ninput token: " << token
            << "\nrestored next: " << restored_next
            << "\nsequential next: " << sequential_next
            << "\nstage diff: " << stage_match.message();

        ASSERT_TRUE(runner->restoreLivePrefixState(prefix_checkpoint));
        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
    }

    inline void runDenseMTPEnabledForwardOnlyMatchesNoMTP(
        DensePrefixRestoreParityCase test_case,
        int decode_steps = 128)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        test_case.name += " dense MTP-enabled forward-only parity";
        test_case.prompt = qwen36DefaultBenchmarkPrompt();
        test_case.decode_steps = decode_steps;
        test_case.max_seq_len = 768;

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
        });

        if (auto skip_reason = densePrefixParitySkipReason(test_case))
        {
            GTEST_SKIP() << *skip_reason;
        }

        const std::string model_path = firstEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << model_path;
        }

        DeviceManager::instance().initialize(-1);
        auto model_ctx = ModelContext::create(
            model_path,
            nullptr,
            nullptr,
            nullptr,
            WeightDistributionStrategy::REPLICATED);
        ASSERT_NE(model_ctx, nullptr);

        auto tokenizer = createTokenizer(model_ctx);
        ASSERT_NE(tokenizer, nullptr);
        const std::vector<int> encoded_prompt =
            tokenizer->encode(test_case.prompt, /*add_bos=*/false, /*add_eos=*/false);
        ASSERT_FALSE(encoded_prompt.empty());
        std::vector<int32_t> prompt_tokens(
            encoded_prompt.begin(),
            encoded_prompt.end());
        ASSERT_LT(static_cast<int>(prompt_tokens.size()) + decode_steps, test_case.max_seq_len);

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();

        auto make_config = [&](bool enable_mtp)
        {
            InferenceRunnerConfig config;
            config.max_seq_len = test_case.max_seq_len;
            config.batch_size = 1;
            config.force_graph = true;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
            config.use_mapped_memory = false;
            config.mtp.enabled = enable_mtp;
            config.mtp.draft_tokens = 1;
            return config;
        };

        auto run_forward_only = [&](bool enable_mtp,
                                    const std::vector<int32_t> *teacher_tokens)
            -> std::vector<int32_t>
        {
            std::vector<int32_t> generated;
            auto runner_model_ctx = ModelContext::create(
                model_path,
                nullptr,
                nullptr,
                nullptr,
                WeightDistributionStrategy::REPLICATED);
            EXPECT_NE(runner_model_ctx, nullptr);
            if (!runner_model_ctx)
            {
                return generated;
            }
            auto runner = createInferenceRunner(
                runner_model_ctx,
                nullptr,
                device,
                make_config(enable_mtp));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return generated;
            }
            if (auto *device_runner =
                    dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
            {
                device_runner->setHostResidentReleaseEnabled(false);
            }
            runner->setSuppressTimeline(true);
            runner->setSkipLogitsGatherPrefill(true);
            runner->setSkipLogitsGatherDecode(true);

            if (!runner->forward(prompt_tokens.data(), static_cast<int>(prompt_tokens.size())))
            {
                ADD_FAILURE() << "prefill failed with mtp=" << enable_mtp;
                return generated;
            }

            int32_t driver = prompt_tokens.back();
            for (int step = 0; step < decode_steps; ++step)
            {
                if (!runner->forward(&driver, 1))
                {
                    ADD_FAILURE() << "forward failed at step " << step
                                  << " with mtp=" << enable_mtp;
                    break;
                }
                const int32_t sampled = runner->sampleGreedyOnDevice();
                if (sampled < 0)
                {
                    ADD_FAILURE() << "sampling failed at step " << step
                                  << " with mtp=" << enable_mtp;
                    break;
                }
                generated.push_back(sampled);
                if (teacher_tokens && step < static_cast<int>(teacher_tokens->size()))
                {
                    driver = (*teacher_tokens)[static_cast<size_t>(step)];
                }
                else
                {
                    driver = sampled;
                }
            }

            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            return generated;
        };

        const std::vector<int32_t> baseline =
            run_forward_only(/*enable_mtp=*/false, nullptr);
        ASSERT_EQ(baseline.size(), static_cast<size_t>(decode_steps));
        const std::vector<int32_t> mtp_enabled =
            run_forward_only(/*enable_mtp=*/true, &baseline);
        ASSERT_EQ(mtp_enabled.size(), baseline.size());
        EXPECT_TRUE(tokenSequencesMatch(
            mtp_enabled,
            baseline,
            "MTP-enabled forward-only"));
    }

    inline void runDenseStochasticMTPVerifierParity(
        const DensePrefixRestoreParityCase &test_case)
    {
        ScopedDenseParityDeterministicMode deterministic_mode(
            shouldUseDenseParityDeterministicMode(test_case));
        ASSERT_EQ(test_case.topology, DensePrefixParityTopology::SingleDevice)
            << "Phase 13.7 stochastic MTP verifier parity is currently single-device only";

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "0"},
            {"LLAMINAR_MTP_PHASE138_CATCHUP_CANDIDATE", "vllm_style_spec_decode"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        constexpr int block_size = 2;
        constexpr int stochastic_decode_steps = 3;
        auto factory = createOrchestrationRunnerFactory();

        SamplingParams stochastic;
        stochastic.temperature = 0.6f;
        stochastic.top_k = 20;
        stochastic.top_p = 0.95f;
        stochastic.presence_penalty = 0.25f;
        stochastic.seed = 123;

        auto baseline_config =
            makeDensePrefixRestoreConfig(test_case, model_path, false, block_size, false);
        auto baseline = factory->createFromOrchestrationConfig(baseline_config);
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result =
            baseline->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), static_cast<size_t>(stochastic_decode_steps));
        EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);
        EXPECT_EQ(baseline_snapshot.mtp_stochastic_accept_tests, 0u);

        auto mtp_config =
            makeDensePrefixRestoreConfig(test_case, model_path, false, block_size, true, 1);
        mtp_config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

        auto mtp = factory->createFromOrchestrationConfig(mtp_config);
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();
        PerfStatsCollector::reset();
        ASSERT_TRUE(PerfStatsCollector::isEnabled())
            << "Phase 13.8 stochastic candidate parity requires perf stats";
        auto mtp_result = mtp->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        const auto after_mtp = mtp->prefixStateProbe();
        const auto phase138_records = PerfStatsCollector::snapshot({"mtp"});
        mtp->shutdown();

        ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
        ASSERT_EQ(mtp_result.tokens.size(), static_cast<size_t>(stochastic_decode_steps));
        EXPECT_FALSE(after_mtp.mtp_bypassed) << after_mtp.mtp_bypass_reason;
        EXPECT_EQ(after_mtp.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(after_mtp.mtp_request.stochastic_verify);
        expectPhase138TransactionUsed(
            test_case,
            after_mtp,
            test_case.name + " stochastic MTP",
            /*allow_transaction_rollbacks=*/true);
        EXPECT_GE(after_mtp.mtp_draft_steps, 1u);
        EXPECT_GE(after_mtp.mtp_verifier_runs, 1u);
        EXPECT_GE(after_mtp.mtp_verifier_token_count, 2u);
        EXPECT_GE(after_mtp.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(after_mtp.mtp_stochastic_accept_tests,
                  after_mtp.mtp_stochastic_accepts +
                      after_mtp.mtp_stochastic_residual_samples);
        EXPECT_GE(after_mtp.mtp_stochastic_residual_samples +
                      after_mtp.mtp_stochastic_terminal_samples,
                  1u);
        EXPECT_EQ(after_mtp.mtp_request.stochastic_accept_tests,
                  after_mtp.mtp_stochastic_accept_tests);
        EXPECT_EQ(after_mtp.mtp_request.stochastic_accepts,
                  after_mtp.mtp_stochastic_accepts);
        EXPECT_EQ(after_mtp.mtp_request.stochastic_residual_samples,
                  after_mtp.mtp_stochastic_residual_samples);
        EXPECT_EQ(after_mtp.mtp_request.stochastic_terminal_samples,
                  after_mtp.mtp_stochastic_terminal_samples);
        EXPECT_GE(after_mtp.mtp_request.stochastic_acceptance_rate, 0.0);
        EXPECT_LE(after_mtp.mtp_request.stochastic_acceptance_rate, 1.0);
        if (after_mtp.mtp_stochastic_accept_tests > 0)
        {
            const double expected_rate =
                static_cast<double>(after_mtp.mtp_stochastic_accepts) /
                static_cast<double>(after_mtp.mtp_stochastic_accept_tests);
            EXPECT_NEAR(after_mtp.mtp_request.stochastic_acceptance_rate,
                        expected_rate,
                        1e-12);
        }

        const bool used_phase138_stochastic_candidate =
            std::any_of(
                phase138_records.begin(),
                phase138_records.end(),
                [](const PerfStatRecord &record)
                {
                    return record.kind == PerfStatRecord::Kind::Counter &&
                           record.domain == "mtp" &&
                           record.name == "phase138_stochastic_spec_decode_runs";
                });
        EXPECT_TRUE(used_phase138_stochastic_candidate)
            << "Stochastic MTP parity must exercise the Phase 13.8 "
               "accepted-count candidate\n"
            << PerfStatsCollector::summaryString({"mtp"});
    }

} // namespace llaminar2::test::parity::qwen36
