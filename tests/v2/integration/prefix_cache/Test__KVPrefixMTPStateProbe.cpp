#include <gtest/gtest.h>

#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/IWorkerGPUContext.h"
#include "config/OrchestrationConfig.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/mtp/MTPStateTransaction.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "kernels/KernelFactory.h"
#include "loaders/PreparedWeightStore.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "loaders/ModelLoader.h"
#include "models/qwen/QwenStandardGraph.h"
#include "models/qwen35/Qwen35Graph.h"
#include "utils/MPIContext.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Sampler.h"
#include "utils/Sha256.h"
#include "utils/TestTensorFactory.h"
#include "utils/Tokenizer.h"
#include "utils/DebugEnv.h"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    constexpr const char *kDenseModelPath = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    /**
     * @brief Host materialization of one graph-stage diagnostic snapshot.
     *
     * Production GPU execution remains device-owned.  The integration-only
     * snapshot callback records point-in-time D2D copies inside graph capture
     * and materializes those copies after execution, so this diagnostic cannot
     * alter the ownership or ordering contract of the tensors being tested.
     */
    struct RequestBatchStageSnapshot
    {
        std::vector<float> data;  ///< Immutable post-execution stage contents.
    };

    /**
     * @brief Return semantic Qwen3.6 snapshot keys in forward execution order.
     *
     * The filter intentionally names every dense attention, GDN, and FFN
     * intermediate that SnapshotCapture can publish.  A model layer only emits
     * keys for its actual implementation, so one exhaustive list covers both
     * full-attention and GDN layers without teaching this regression the model's
     * layer schedule.  Generating more layer names than the loaded model owns is
     * harmless because the graph's snapshot-stage filter simply never matches
     * them.
     */
    std::vector<std::string> requestBatchPrefillSnapshotKeys()
    {
        static const std::vector<std::string> kLayerStageOrder = {
            "ATTENTION_NORM_RESIDUAL_OUT",
            "ATTENTION_NORM",
            "QKV_PROJECTION",
            "GDN_Z_PROJECTION",
            "GDN_ALPHA",
            "GDN_BETA",
            "GDN_CONV1D_OUTPUT",
            "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT",
            "Q_PROJECTION",
            "K_PROJECTION",
            "V_PROJECTION",
            "Q_NORM",
            "K_NORM",
            "Q_ROPE",
            "K_ROPE",
            "FA_GATE",
            "ATTENTION_EFFECTIVE_K",
            "ATTENTION_EFFECTIVE_V",
            "ATTENTION_CONTEXT",
            "ATTENTION_CONTEXT_GATED",
            "ATTENTION_OUTPUT",
            "ATTENTION_OUTPUT_ALLREDUCED",
            "ATTENTION_RESIDUAL",
            "FFN_NORM_RESIDUAL_OUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED",
            "MOE_EXPERT_OUTPUT",
            "MOE_EXPERT_OUTPUT_ALLREDUCED",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "MOE_COMBINED_OUTPUT_ALLREDUCED",
            "FFN_GATE",
            "FFN_UP",
            "FFN_SWIGLU",
            "FFN_DOWN",
            "FFN_DOWN_ALLREDUCED",
            "FFN_RESIDUAL",
        };

        constexpr int kMaximumDiagnosticLayers = 128;
        std::vector<std::string> keys;
        keys.reserve(
            3 + static_cast<size_t>(kMaximumDiagnosticLayers) *
                    kLayerStageOrder.size());
        keys.push_back("EMBEDDING");
        for (int layer = 0; layer < kMaximumDiagnosticLayers; ++layer)
        {
            const std::string prefix = "layer" + std::to_string(layer) + "_";
            for (const std::string &stage : kLayerStageOrder)
                keys.push_back(prefix + stage);
        }
        keys.push_back("FINAL_NORM");
        keys.push_back("LM_HEAD");
        return keys;
    }

    /**
     * @brief Copy all valid snapshots currently published by a runner.
     *
     * Keeping a private copy is important because clearSnapshots() reuses the
     * runner for the batched pass.  The public orchestration API intentionally
     * exposes a flat FP32 semantic view; the comparison helper below infers its
     * row geometry from the isolated and request-batched execution contracts.
     */
    std::map<std::string, RequestBatchStageSnapshot> captureRequestBatchSnapshots(
        const IOrchestrationRunner &runner,
        const std::vector<std::string> &semantic_keys,
        const std::string &context = {})
    {
        std::map<std::string, RequestBatchStageSnapshot> snapshots;
        const std::string context_prefix =
            context.empty() ? std::string{} : context + "_";
        for (const std::string &semantic_key : semantic_keys)
        {
            size_t size = 0;
            const float *data = runner.getSnapshot(
                context_prefix + semantic_key,
                size);
            if (!data || size == 0)
                continue;

            RequestBatchStageSnapshot snapshot;
            snapshot.data.assign(data, data + size);
            snapshots.emplace(semantic_key, std::move(snapshot));
        }
        return snapshots;
    }

    /**
     * @brief Select one transformer-layer window for prefill continuity diagnosis.
     *
     * The long-context state probe already identifies the first persistent
     * state slot that differs. Capturing the graph stages around that slot then
     * identifies the first arithmetic producer without materializing all
     * intermediates from the complete 41-layer model. Snapshot copies are
     * recorded on each stage's producer stream as graph nodes, so the captured
     * values preserve point-in-time semantics even when later stages reuse the
     * same arena tensor.
     *
     * @param first_layer Inclusive first transformer layer to capture.
     * @param last_layer Inclusive final transformer layer to capture.
     * @return Semantic snapshot keys in forward execution order.
     */
    std::vector<std::string> prefillReplayContinuitySnapshotKeys(
        int first_layer,
        int last_layer)
    {
        if (first_layer < 0 || last_layer < first_layer)
            throw std::invalid_argument(
                "prefill replay snapshot layer window is invalid");

        std::vector<std::string> selected;
        for (const std::string &key : requestBatchPrefillSnapshotKeys())
        {
            if (key.rfind("layer", 0) != 0)
                continue;

            const size_t separator = key.find('_', 5);
            if (separator == std::string::npos)
                continue;
            const int layer = std::stoi(key.substr(5, separator - 5));
            if (layer < first_layer || layer > last_layer)
                continue;

            /*
             * LLEP is allowed to move routed/shared expert work between LocalTP
             * participants. Participant-local MoE partials therefore change
             * even when the semantic result is exact. Compare the explicit
             * post-collective products instead; treating a valid placement
             * change as arithmetic drift would point diagnostics at the first
             * moved expert instead of the first observable model difference.
             */
            const bool local_moe_partial =
                (key.ends_with("_MOE_EXPERT_OUTPUT") ||
                 key.ends_with("_MOE_SHARED_EXPERT_OUTPUT") ||
                 key.ends_with("_MOE_COMBINED_OUTPUT"));
            if (!local_moe_partial)
                selected.push_back(key);
        }
        return selected;
    }

    /**
     * @brief Compare two captured suffix passes using raw FP32 bytes.
     *
     * Relaxed tensor metrics are intentionally unsuitable here: the verifier
     * and continuation paths require batch-invariant arithmetic, and one ULP
     * can later change routing or sampling.  The result names the first stage
     * in forward order and the first changed IEEE-754 word.
     */
    ::testing::AssertionResult prefillReplaySnapshotsByteIdentical(
        const std::map<std::string, RequestBatchStageSnapshot> &full_prefill,
        const std::map<std::string, RequestBatchStageSnapshot> &suffix_prefill,
        const std::vector<std::string> &ordered_keys,
        size_t full_rows,
        size_t suffix_row_offset,
        size_t suffix_rows)
    {
        if (full_rows == 0 || suffix_rows == 0 ||
            suffix_row_offset + suffix_rows != full_rows)
        {
            return ::testing::AssertionFailure()
                   << "invalid full/suffix snapshot geometry: full_rows="
                   << full_rows << " suffix_row_offset=" << suffix_row_offset
                   << " suffix_rows=" << suffix_rows;
        }

        size_t comparable = 0;
        for (const std::string &key : ordered_keys)
        {
            const auto full_it = full_prefill.find(key);
            const auto suffix_it = suffix_prefill.find(key);
            if (full_it == full_prefill.end() ||
                suffix_it == suffix_prefill.end())
                continue;

            ++comparable;
            const auto &full = full_it->second.data;
            const auto &suffix = suffix_it->second.data;
            const float *expected_suffix = nullptr;
            size_t compared_elements = 0;
            size_t diagnostic_row_width = 0;
            bool compares_terminal_graph_tile = false;

            if (full.size() == suffix.size())
            {
                /*
                 * Prefill graph execution tiles a long request through a
                 * graph-stable row bucket. SnapshotCapture retains the last
                 * callback for a semantic key, so equal-size captures are the
                 * terminal tile from each execution. The split point is itself
                 * graph-bucket aligned in this regression; therefore both
                 * terminal tiles describe the same absolute token rows and
                 * compare directly.
                 */
                expected_suffix = full.data();
                compared_elements = full.size();
                compares_terminal_graph_tile = true;
            }
            else
            {
                if (full.size() % full_rows != 0 ||
                    suffix.size() % suffix_rows != 0)
                {
                    return ::testing::AssertionFailure()
                           << "snapshot cannot be interpreted as token rows at "
                           << key << " full_elements=" << full.size()
                           << " full_rows=" << full_rows
                           << " suffix_elements=" << suffix.size()
                           << " suffix_rows=" << suffix_rows;
                }

                const size_t full_row_width = full.size() / full_rows;
                const size_t suffix_row_width = suffix.size() / suffix_rows;
                if (full_row_width == 0 || full_row_width != suffix_row_width)
                {
                    return ::testing::AssertionFailure()
                           << "snapshot row-width mismatch at " << key
                           << " full_width=" << full_row_width
                           << " suffix_width=" << suffix_row_width;
                }

                diagnostic_row_width = full_row_width;
                expected_suffix =
                    full.data() + suffix_row_offset * full_row_width;
                compared_elements = suffix_rows * suffix_row_width;
            }

            if (std::memcmp(
                    expected_suffix,
                    suffix.data(),
                    compared_elements * sizeof(float)) == 0)
                continue;

            for (size_t index = 0; index < compared_elements; ++index)
            {
                uint32_t full_bits = 0;
                uint32_t suffix_bits = 0;
                std::memcpy(
                    &full_bits,
                    &expected_suffix[index],
                    sizeof(full_bits));
                std::memcpy(
                    &suffix_bits,
                    &suffix[index],
                    sizeof(suffix_bits));
                if (full_bits == suffix_bits)
                    continue;

                auto firstIndexOfBits = [](const std::vector<float> &values,
                                           uint32_t wanted)
                    -> std::optional<size_t>
                {
                    for (size_t candidate = 0; candidate < values.size(); ++candidate)
                    {
                        uint32_t candidate_bits = 0;
                        std::memcpy(
                            &candidate_bits,
                            &values[candidate],
                            sizeof(candidate_bits));
                        if (candidate_bits == wanted)
                            return candidate;
                    }
                    return std::nullopt;
                };

                auto failure = ::testing::AssertionFailure();
                failure << "first snapshot byte mismatch at " << key
                        << " layout="
                        << (compares_terminal_graph_tile
                                ? "terminal_graph_tile"
                                : "whole_request_suffix")
                        << " element=" << index;
                if (diagnostic_row_width > 0)
                {
                    failure << " suffix_row="
                            << (index / diagnostic_row_width)
                            << " full_row="
                            << (suffix_row_offset +
                                index / diagnostic_row_width)
                            << " column=" << (index % diagnostic_row_width);
                }
                failure
                        << " full=" << expected_suffix[index]
                        << " suffix=" << suffix[index]
                        << " full_bits=0x" << std::hex << std::setw(8)
                        << std::setfill('0') << full_bits
                        << " suffix_bits=0x" << std::setw(8) << suffix_bits
                        << std::dec << std::setfill(' ');

                if (key.find("ATTENTION_EFFECTIVE_") != std::string::npos)
                {
                    const auto suffix_in_full =
                        firstIndexOfBits(full, suffix_bits);
                    const auto full_in_suffix =
                        firstIndexOfBits(suffix, full_bits);
                    failure << " suffix_value_in_full="
                            << (suffix_in_full
                                    ? std::to_string(*suffix_in_full)
                                    : std::string("absent"))
                            << " full_value_in_suffix="
                            << (full_in_suffix
                                    ? std::to_string(*full_in_suffix)
                                    : std::string("absent"));

                    const size_t layer_separator = key.find('_');
                    const std::string layer_prefix =
                        layer_separator == std::string::npos
                            ? std::string{}
                            : key.substr(0, layer_separator);
                    for (const char *suffix : {
                             "_K_PROJECTION",
                             "_K_NORM",
                             "_K_ROPE",
                         })
                    {
                        const std::string related_key = layer_prefix + suffix;
                        const auto full_related =
                            full_prefill.find(related_key);
                        const auto suffix_related =
                            suffix_prefill.find(related_key);
                        if (full_related == full_prefill.end() ||
                            suffix_related == suffix_prefill.end() ||
                            full_related->second.data.empty() ||
                            suffix_related->second.data.empty())
                        {
                            continue;
                        }
                        failure << ' ' << related_key
                                << "[0]=" << full_related->second.data[0]
                                << '/' << suffix_related->second.data[0];
                    }
                }
                return failure;
            }
        }

        if (comparable == 0)
            return ::testing::AssertionFailure() << "no comparable prefill stage snapshots";
        return ::testing::AssertionSuccess();
    }

    /**
     * @brief Compare one unequal-length request against its isolated prefill.
     *
     * Full-row stages use the scalar and flattened-batch terminal row indices.
     * Compact LM-head-style stages instead expose one row per request, so their
     * isolated row zero is compared with the selected request row.  The first
     * mismatch is returned in graph order with exact IEEE-754 bit patterns; this
     * makes the regression identify the responsible grouped operation instead
     * of merely reporting a later sampled-token difference.
     */
    ::testing::AssertionResult requestBatchTerminalRowsByteIdentical(
        const std::map<std::string, RequestBatchStageSnapshot> &scalar,
        const std::map<std::string, RequestBatchStageSnapshot> &batched,
        const std::vector<std::string> &ordered_keys,
        size_t scalar_total_rows,
        size_t batch_total_rows,
        size_t scalar_terminal_row,
        size_t batch_terminal_row,
        size_t request_index,
        size_t request_count,
        const std::string &label)
    {
        size_t comparable = 0;
        for (const std::string &key : ordered_keys)
        {
            const auto scalar_it = scalar.find(key);
            const auto batch_it = batched.find(key);
            if (scalar_it == scalar.end() || batch_it == batched.end())
                continue;

            const RequestBatchStageSnapshot &one = scalar_it->second;
            const RequestBatchStageSnapshot &many = batch_it->second;

            size_t one_row = scalar_terminal_row;
            size_t many_row = batch_terminal_row;
            size_t one_rows = 0;
            size_t many_rows = 0;
            size_t cols = 0;
            if (scalar_total_rows > 0 && batch_total_rows > 0 &&
                one.data.size() % scalar_total_rows == 0 &&
                many.data.size() % batch_total_rows == 0 &&
                one.data.size() / scalar_total_rows ==
                    many.data.size() / batch_total_rows)
            {
                one_rows = scalar_total_rows;
                many_rows = batch_total_rows;
                cols = one.data.size() / one_rows;
            }
            else if (many.data.size() == one.data.size() * request_count)
            {
                one_rows = 1;
                many_rows = request_count;
                cols = one.data.size();
                one_row = 0;
                many_row = request_index;
            }
            else
            {
                continue;
            }
            if (one_row >= one_rows || many_row >= many_rows || cols == 0)
                continue;

            ++comparable;
            const size_t one_offset = one_row * cols;
            const size_t many_offset = many_row * cols;
            size_t mismatch_count = 0;
            size_t first_col = 0;
            uint32_t scalar_bits = 0;
            uint32_t batch_bits = 0;
            float scalar_value = 0.0f;
            float batch_value = 0.0f;
            float max_abs = 0.0f;
            for (size_t col = 0; col < cols; ++col)
            {
                const float expected = one.data[one_offset + col];
                const float actual = many.data[many_offset + col];
                uint32_t expected_bits = 0;
                uint32_t actual_bits = 0;
                std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
                std::memcpy(&actual_bits, &actual, sizeof(actual_bits));
                if (expected_bits == actual_bits)
                    continue;

                if (mismatch_count == 0)
                {
                    first_col = col;
                    scalar_bits = expected_bits;
                    batch_bits = actual_bits;
                    scalar_value = expected;
                    batch_value = actual;
                }
                ++mismatch_count;
                max_abs = std::max(max_abs, std::fabs(actual - expected));
            }

            if (mismatch_count > 0)
            {
                return ::testing::AssertionFailure()
                       << label << " first divergent grouped stage " << key
                       << " after " << comparable << " comparable stages: "
                       << mismatch_count << "/" << cols
                       << " values differ; first_col=" << first_col
                       << " batch=" << batch_value
                       << " scalar=" << scalar_value
                       << " batch_bits=0x" << std::hex << batch_bits
                       << " scalar_bits=0x" << scalar_bits << std::dec
                       << " max_abs=" << max_abs
                       << " scalar_shape=[" << one_rows << "," << cols << "]"
                       << " batch_shape=[" << many_rows << "," << cols << "]";
            }
        }

        if (comparable == 0)
        {
            return ::testing::AssertionFailure()
                   << label << " captured no comparable scalar/batched stage rows";
        }
        return ::testing::AssertionSuccess();
    }

    /**
     * @brief Compare one active request prefix in a fixed-stride grouped KV view.
     *
     * Scalar attention snapshots contain exactly `scalar_kv_rows` rows.  A GPU
     * request-batch snapshot stores every bank with `grouped_kv_stride` rows,
     * zero-padding the suffix of shorter requests.  Comparing only the active
     * prefix isolates cache append/gather/RoPE correctness from the subsequent
     * softmax and value reduction.
     */
    ::testing::AssertionResult requestBatchEffectiveKVBankByteIdentical(
        const std::map<std::string, RequestBatchStageSnapshot> &scalar,
        const std::map<std::string, RequestBatchStageSnapshot> &grouped,
        const std::string &key,
        size_t scalar_kv_rows,
        size_t grouped_kv_stride,
        size_t request_index,
        const std::string &label)
    {
        const auto scalar_it = scalar.find(key);
        const auto grouped_it = grouped.find(key);
        if (scalar_it == scalar.end() || grouped_it == grouped.end())
        {
            return ::testing::AssertionFailure()
                   << label << " is missing effective-KV snapshot " << key
                   << " scalar_present=" << (scalar_it != scalar.end())
                   << " grouped_present=" << (grouped_it != grouped.end());
        }
        if (scalar_kv_rows == 0 || grouped_kv_stride < scalar_kv_rows ||
            scalar_it->second.data.size() % scalar_kv_rows != 0)
        {
            return ::testing::AssertionFailure()
                   << label << " has invalid KV row geometry for " << key
                   << " scalar_values=" << scalar_it->second.data.size()
                   << " scalar_rows=" << scalar_kv_rows
                   << " grouped_stride=" << grouped_kv_stride;
        }

        const size_t cols =
            scalar_it->second.data.size() / scalar_kv_rows;
        const size_t grouped_row_start = request_index * grouped_kv_stride;
        const size_t grouped_value_start = grouped_row_start * cols;
        const size_t compared_values = scalar_kv_rows * cols;
        if (cols == 0 ||
            grouped_it->second.data.size() <
                grouped_value_start + compared_values)
        {
            return ::testing::AssertionFailure()
                   << label << " grouped KV snapshot is undersized for " << key
                   << " grouped_values=" << grouped_it->second.data.size()
                   << " required_values="
                   << grouped_value_start + compared_values
                   << " cols=" << cols;
        }

        const float *expected = scalar_it->second.data.data();
        const float *actual =
            grouped_it->second.data.data() + grouped_value_start;
        if (std::memcmp(
                actual,
                expected,
                compared_values * sizeof(float)) == 0)
        {
            return ::testing::AssertionSuccess();
        }

        size_t first = 0;
        size_t mismatch_count = 0;
        double max_abs = 0.0;
        for (size_t i = 0; i < compared_values; ++i)
        {
            uint32_t expected_bits = 0;
            uint32_t actual_bits = 0;
            std::memcpy(&expected_bits, expected + i, sizeof(expected_bits));
            std::memcpy(&actual_bits, actual + i, sizeof(actual_bits));
            if (expected_bits == actual_bits)
                continue;
            if (mismatch_count == 0)
                first = i;
            ++mismatch_count;
            max_abs = std::max(
                max_abs,
                std::fabs(static_cast<double>(actual[i]) - expected[i]));
        }

        uint32_t expected_bits = 0;
        uint32_t actual_bits = 0;
        std::memcpy(&expected_bits, expected + first, sizeof(expected_bits));
        std::memcpy(&actual_bits, actual + first, sizeof(actual_bits));
        return ::testing::AssertionFailure()
               << label << " effective KV bank differs at " << key
               << " mismatches=" << mismatch_count << "/" << compared_values
               << " first_row=" << first / cols
               << " first_col=" << first % cols
               << " grouped=" << actual[first]
               << " scalar=" << expected[first]
               << " grouped_bits=0x" << std::hex << actual_bits
               << " scalar_bits=0x" << expected_bits << std::dec
               << " max_abs=" << max_abs;
    }

    class ScopedDebugEnv
    {
    public:
        explicit ScopedDebugEnv(std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old = std::getenv(name))
                    entry.old_value = old;
                entries_.push_back(std::move(entry));
                ::setenv(name, value, 1);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedDebugEnv()
        {
            for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
            {
                if (it->old_value.has_value())
                    ::setenv(it->name.c_str(), it->old_value->c_str(), 1);
                else
                    ::unsetenv(it->name.c_str());
            }
            mutableDebugEnv().reload();
        }

        ScopedDebugEnv(const ScopedDebugEnv &) = delete;
        ScopedDebugEnv &operator=(const ScopedDebugEnv &) = delete;

    private:
        struct Entry
        {
            std::string name;
            std::optional<std::string> old_value;
        };

        std::vector<Entry> entries_;
    };

    std::vector<int32_t> readTokenListFromMetadata(
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

    std::filesystem::path tempPrefixDiskDir();

    std::optional<std::string> firstGpuDeviceSpec()
    {
        auto &dm = DeviceManager::instance();
        dm.initialize(-1, false);
        if (dm.cuda_device_count() > 0)
        {
            return std::string("cuda:0");
        }
        if (dm.rocm_device_count() > 0)
        {
            return std::string("rocm:0");
        }
        return std::nullopt;
    }

    std::optional<std::string> explicitGpuDeviceSpec(DeviceType type)
    {
        auto &dm = DeviceManager::instance();
        dm.initialize(-1, false);
        if (type == DeviceType::CUDA && dm.cuda_device_count() > 0)
            return std::string("cuda:0");
        if (type == DeviceType::ROCm && dm.rocm_device_count() > 0)
            return std::string("rocm:0");
        return std::nullopt;
    }

    std::optional<DeviceId> firstGpuDeviceId()
    {
        auto &dm = DeviceManager::instance();
        dm.initialize(-1, false);
        if (dm.cuda_device_count() > 0)
        {
            return DeviceId::cuda(0);
        }
        if (dm.rocm_device_count() > 0)
        {
            return DeviceId::rocm(0);
        }
        return std::nullopt;
    }

    OrchestrationConfig makeSingleGpuConfig(const std::string &device_spec)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = kDenseModelPath;
        config.max_seq_len = 32;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "fp16";
        auto parsed = GlobalDeviceAddress::tryParse(device_spec);
        if (!parsed)
        {
            throw std::runtime_error("invalid GPU device spec: " + device_spec);
        }
        config.device_for_this_rank = *parsed;
        return config;
    }

    OrchestrationConfig makeSingleGpuPrefixCacheConfig(const std::string &device_spec)
    {
        OrchestrationConfig config = makeSingleGpuConfig(device_spec);
        config.prefix_cache.enabled = true;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
        config.prefix_cache.block_size = 2;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 64ull * 1024ull * 1024ull;
        return config;
    }

    OrchestrationConfig makeSingleGpuTieredPrefixCacheConfig(
        const std::string &device_spec,
        const std::filesystem::path &disk_dir)
    {
        OrchestrationConfig config = makeSingleGpuPrefixCacheConfig(device_spec);
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
        config.prefix_cache.ram_budget_bytes = 640ull * 1024ull;
        config.prefix_cache.device_budget_bytes = 0;
        config.prefix_cache.disk_budget_bytes = 32ull * 1024ull * 1024ull;
        config.prefix_cache.disk_dir = disk_dir.string();
        return config;
    }

    OrchestrationConfig makeSingleGpuDeviceHotPrefixCacheConfig(
        const std::string &device_spec,
        const std::filesystem::path &disk_dir)
    {
        OrchestrationConfig config = makeSingleGpuPrefixCacheConfig(device_spec);
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
        config.prefix_cache.ram_budget_bytes = 640ull * 1024ull;
        config.prefix_cache.device_budget_bytes = 32ull * 1024ull * 1024ull;
        config.prefix_cache.disk_budget_bytes = 32ull * 1024ull * 1024ull;
        config.prefix_cache.disk_dir = disk_dir.string();
        return config;
    }

    OrchestrationConfig makeSingleGpuConstrainedHotPrefixCacheConfig(
        const std::string &device_spec,
        const std::filesystem::path &disk_dir)
    {
        OrchestrationConfig config =
            makeSingleGpuDeviceHotPrefixCacheConfig(
                device_spec,
                disk_dir);
        /*
         * The dense test model's terminal two-token block is 632,320 bytes.
         * A 640 KiB budget admits that block but cannot retain it alongside the
         * smaller nonterminal block, forcing deterministic hot-tier LRU turns.
         */
        config.prefix_cache.device_budget_bytes =
            640ull * 1024ull;
        return config;
    }

    void verifyGpuDeviceHotTierRestoresDirectly(
        const std::string &device_spec)
    {
        const auto disk_dir = tempPrefixDiskDir();
        const auto cleanup = [&]() { std::filesystem::remove_all(disk_dir); };

        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(
            makeSingleGpuDeviceHotPrefixCacheConfig(device_spec, disk_dir));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        const std::vector<int32_t> prompt = {1, 2, 3, 4};

        auto first = runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), 1u);

        const auto after_first = runner->prefixStateProbe();
        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 2u);
        EXPECT_GE(after_first.prefix_cache_evictions, 1u);
        EXPECT_GT(after_first.prefix_cache_device_bytes, 0u);
        EXPECT_GT(after_first.prefix_cache_disk_bytes, 0u);
        EXPECT_GE(after_first.prefix_cache_promotions, 2u);

        auto second = runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), 1u);
        EXPECT_EQ(second.tokens.front(), first.tokens.front());

        const auto after_second = runner->prefixStateProbe();
        EXPECT_GE(after_second.prefix_cache_hits, 2u);
        EXPECT_GE(after_second.prefix_cache_device_hot_direct_hits, 2u)
            << "Every restored block should come directly from its VRAM replica";
        EXPECT_EQ(after_second.prefix_cache_disk_hydrations, 0u)
            << "A device-hot hit must not materialize an intermediate RAM copy";
        EXPECT_GT(after_second.prefix_cache_device_bytes, 0u);

        runner.reset();
        cleanup();
    }

    void verifyGpuTieredDiskCacheHydrates(
        const std::string &device_spec)
    {
        const auto disk_dir = tempPrefixDiskDir();
        const auto cleanup = [&]() { std::filesystem::remove_all(disk_dir); };

        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(
            makeSingleGpuTieredPrefixCacheConfig(device_spec, disk_dir));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        const std::vector<int32_t> prompt = {1, 2, 3, 4};

        auto first = runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), 1u);

        const auto after_first = runner->prefixStateProbe();
        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 2u);
        EXPECT_GE(after_first.prefix_cache_evictions, 1u);
        EXPECT_GT(after_first.prefix_cache_disk_bytes, 0u);

        auto second = runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), 1u);
        EXPECT_EQ(second.tokens.front(), first.tokens.front());

        const auto after_second = runner->prefixStateProbe();
        EXPECT_TRUE(after_second.prefix_cache_ready);
        EXPECT_GE(after_second.prefix_cache_hits, 2u);
        EXPECT_GE(after_second.prefix_cache_disk_hydrations, 2u)
            << "Both 2-token blocks should be served through disk hydration";
        EXPECT_GT(after_second.prefix_cache_disk_bytes, 0u);

        runner.reset();
        cleanup();
    }

    void verifyGpuDiskArchiveSurvivesRunnerRestart(
        const std::string &device_spec)
    {
        const auto disk_dir = tempPrefixDiskDir();
        const auto cleanup = [&]() { std::filesystem::remove_all(disk_dir); };
        const auto config =
            makeSingleGpuTieredPrefixCacheConfig(device_spec, disk_dir);
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        const std::vector<int32_t> prompt = {1, 2, 3, 4};
        std::vector<int32_t> first_tokens;

        {
            auto factory = createOrchestrationRunnerFactory();
            auto runner =
                factory->createFromOrchestrationConfig(config);
            ASSERT_NE(runner, nullptr);
            ASSERT_TRUE(runner->initialize()) << runner->lastError();
            auto first = runner->generate(prompt, 1, greedy);
            ASSERT_TRUE(first.error.empty()) << first.error;
            ASSERT_EQ(first.tokens.size(), 1u);
            first_tokens = first.tokens;
            EXPECT_GT(
                runner->prefixStateProbe().prefix_cache_disk_bytes,
                0u);
        }

        std::string digest_error;
        const auto model_digest =
            sha256FileHex(kDenseModelPath, &digest_error);
        ASSERT_TRUE(model_digest.has_value()) << digest_error;
        const auto archive_path =
            disk_dir / (*model_digest + ".kvcache");
        ASSERT_TRUE(std::filesystem::is_regular_file(archive_path));

        {
            auto factory = createOrchestrationRunnerFactory();
            auto runner =
                factory->createFromOrchestrationConfig(config);
            ASSERT_NE(runner, nullptr);
            ASSERT_TRUE(runner->initialize()) << runner->lastError();

            auto restored = runner->generate(prompt, 1, greedy);
            ASSERT_TRUE(restored.error.empty()) << restored.error;
            ASSERT_EQ(restored.tokens.size(), 1u);
            EXPECT_EQ(restored.tokens, first_tokens);
            const auto after_restore = runner->prefixStateProbe();
            EXPECT_GE(after_restore.prefix_cache_disk_hydrations, 1u);
            EXPECT_GE(after_restore.prefix_cache_matched_tokens, 2u);
            EXPECT_GT(after_restore.prefix_cache_disk_bytes, 0u)
                << "The first lookup lazily discovers and indexes committed "
                   "model archive records";
        }

        cleanup();
    }

    void verifyGpuDiskHitRepromotesToDeviceHot(
        const std::string &device_spec)
    {
        const auto disk_dir = tempPrefixDiskDir();
        const auto cleanup = [&]() { std::filesystem::remove_all(disk_dir); };
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        /*
         * Three two-token blocks with a one-block RAM budget force the first
         * two blocks into the durable archive. The terminal block remains in
         * RAM until runner destruction, so the restarted lookup is deliberately
         * a four-token partial hit and must continue through ordinary suffix
         * prefill after restoring the promoted state.
         */
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6};
        std::vector<int32_t> reference_tokens;
        {
            auto factory = createOrchestrationRunnerFactory();
            auto cold_runner =
                factory->createFromOrchestrationConfig(
                    makeSingleGpuTieredPrefixCacheConfig(
                        device_spec,
                        disk_dir));
            ASSERT_NE(cold_runner, nullptr);
            ASSERT_TRUE(cold_runner->initialize())
                << cold_runner->lastError();
            auto reference =
                cold_runner->generate(prompt, 1, greedy);
            ASSERT_TRUE(reference.error.empty()) << reference.error;
            ASSERT_EQ(reference.tokens.size(), 1u);
            reference_tokens = reference.tokens;
            EXPECT_GT(
                cold_runner->prefixStateProbe().prefix_cache_disk_bytes,
                0u)
                << "The cold runner should demote nonterminal dense blocks; "
                   "their archive layout intentionally omits terminal logits";
        }

        auto factory = createOrchestrationRunnerFactory();
        auto hot_runner =
            factory->createFromOrchestrationConfig(
                makeSingleGpuDeviceHotPrefixCacheConfig(
                    device_spec,
                    disk_dir));
        ASSERT_NE(hot_runner, nullptr);
        ASSERT_TRUE(hot_runner->initialize())
            << hot_runner->lastError();

        auto promoted = hot_runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(promoted.error.empty()) << promoted.error;
        ASSERT_EQ(promoted.tokens, reference_tokens);
        const auto after_promotion = hot_runner->prefixStateProbe();
        EXPECT_GE(after_promotion.prefix_cache_disk_hydrations, 2u);
        EXPECT_GE(after_promotion.prefix_cache_device_hot_repromotions, 2u)
            << "Each cold block eligible for VRAM must be promoted, not merely "
               "uploaded through transient per-layer staging";
        EXPECT_GT(after_promotion.prefix_cache_device_bytes, 0u);

        const uint64_t hydration_count =
            after_promotion.prefix_cache_disk_hydrations;
        auto direct = hot_runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(direct.error.empty()) << direct.error;
        ASSERT_EQ(direct.tokens, reference_tokens);
        const auto after_direct = hot_runner->prefixStateProbe();
        EXPECT_GE(after_direct.prefix_cache_device_hot_direct_hits, 2u)
            << "The request after cold promotion must consume persistent VRAM replicas";
        EXPECT_EQ(
            after_direct.prefix_cache_disk_hydrations,
            hydration_count)
            << "A promoted hot hit must not re-enter the disk/RAM hydration path";

        hot_runner.reset();
        cleanup();
    }

    void verifyGpuHotTierPressureCycles(
        const std::string &device_spec)
    {
        const auto disk_dir = tempPrefixDiskDir();
        const auto cleanup = [&]() { std::filesystem::remove_all(disk_dir); };
        auto factory = createOrchestrationRunnerFactory();
        auto runner =
            factory->createFromOrchestrationConfig(
                makeSingleGpuConstrainedHotPrefixCacheConfig(
                    device_spec,
                    disk_dir));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        const std::vector<int32_t> prompt = {1, 2, 3, 4};

        auto first = runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), 1u);
        const auto after_first = runner->prefixStateProbe();
        EXPECT_GE(after_first.prefix_cache_device_hot_promotions, 2u);
        EXPECT_GE(after_first.prefix_cache_device_hot_evictions, 1u)
            << "Publishing the terminal block must demote the older "
               "nonterminal hot replica";
        EXPECT_GE(after_first.prefix_cache_ram_to_disk_demotions, 1u);

        auto second = runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens, first.tokens);
        const auto after_second = runner->prefixStateProbe();
        EXPECT_GE(after_second.prefix_cache_disk_hydrations, 1u);
        EXPECT_GE(after_second.prefix_cache_device_hot_direct_hits, 1u);
        EXPECT_GE(after_second.prefix_cache_device_hot_repromotions, 1u);
        EXPECT_GE(after_second.prefix_cache_device_hot_evictions, 2u);

        /*
         * The previous request leaves the opposite block hot. A third request
         * therefore proves that LRU turnover is repeatable in both directions,
         * rather than merely exercising one startup transition.
         */
        auto third = runner->generate(prompt, 1, greedy);
        ASSERT_TRUE(third.error.empty()) << third.error;
        ASSERT_EQ(third.tokens, first.tokens);
        const auto after_third = runner->prefixStateProbe();
        EXPECT_GE(after_third.prefix_cache_device_hot_direct_hits, 2u);
        EXPECT_GE(after_third.prefix_cache_device_hot_repromotions, 2u);
        EXPECT_GE(after_third.prefix_cache_device_hot_evictions, 3u);
        EXPECT_GT(after_third.prefix_cache_device_bytes, 0u);
        EXPECT_GT(after_third.prefix_cache_disk_bytes, 0u);

        runner.reset();
        cleanup();
    }

    OrchestrationConfig makeSingleCpuConfig(bool prefix_cache_enabled)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = kDenseModelPath;
        config.max_seq_len = 16;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "q16_1";
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.prefix_cache.enabled = prefix_cache_enabled;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
        config.prefix_cache.block_size = 2;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 64ull * 1024ull * 1024ull;
        return config;
    }

    int maxLayerCachedTokens(const PrefixRuntimeStateSnapshot &snapshot)
    {
        int max_tokens = 0;
        for (const auto &cache : snapshot.kv_caches)
        {
            for (const auto &layer : cache.layers)
            {
                max_tokens = std::max(max_tokens, layer.cached_tokens);
            }
        }
        return max_tokens;
    }

    int maxCachedTokensIn(const std::vector<PrefixKVCacheProbe> &caches)
    {
        int max_tokens = 0;
        for (const auto &cache : caches)
        {
            for (const auto &layer : cache.layers)
            {
                max_tokens = std::max(max_tokens, layer.cached_tokens);
            }
        }
        return max_tokens;
    }

    void prepareDenseForwardWeights(
        const DeviceGraphOrchestrator &orchestrator,
        QwenStandardGraph &graph_builder,
        PreparedWeightStore &store,
        DeviceId device)
    {
        const FrozenModelWeightSet *frozen = orchestrator.frozenWeightSet();
        ASSERT_NE(frozen, nullptr);

        for (const auto &source_binding : frozen->bindings())
        {
            if (!source_binding.tensor ||
                source_binding.tensor->shape().size() != 2 ||
                source_binding.identity.role == WeightRole::Embedding)
            {
                continue;
            }

            WeightBinding binding = source_binding;
            binding.residency.home_device = device;
            binding.residency.resident_device = device;
            ASSERT_TRUE(binding.tensor->ensureOnDevice(device));
            store.prepareGemm(binding);
        }

        graph_builder.setPreparedWeightStore(&store);
    }

    bool allValuesZero(const std::vector<int> &values)
    {
        return std::all_of(values.begin(), values.end(), [](int value) { return value == 0; });
    }

    double findPerfCounterValue(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name,
        const std::string &phase,
        const PerfStatsCollector::Tags &tags)
    {
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == domain &&
                record.name == name &&
                record.phase == phase &&
                record.tags == tags)
            {
                return record.value;
            }
        }
        return 0.0;
    }

    /**
     * @brief Sum a counter across tag partitions for one exact domain and phase.
     *
     * Production ownership counters intentionally live beside the operation
     * they describe. Request admission is an `mtp/prefill` event, while grouped
     * terminal-state commits are `forward_graph/prefill` events. Keeping domain
     * and phase explicit prevents a decode-only helper from silently reporting
     * zero for a path that did execute.
     */
    double perfCounterValue(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name,
        const std::string &phase)
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == domain &&
                record.name == name &&
                record.phase == phase)
            {
                total += record.value;
            }
        }
        return total;
    }

    /**
     * @brief Prove that portable MoE placement restore ran on every GPU layer.
     *
     * The long-context LLEP probes execute two cache-hit regimes: an exact hit
     * whose first live compute is M=1 decode, and a partial hit whose first live
     * compute is suffix prefill. Each LocalTP participant owns a model-runtime
     * table and each table owns one transaction per routed MoE layer. These
     * counters therefore prove that both graph shapes consumed the dedicated
     * device-resident rehydration path instead of merely producing coincidentally
     * equal output through immutable-owner placement.
     */
    void expectPortableMoEDeviceRehydrationCoverage(
        const std::vector<PerfStatRecord> &records,
        const std::string &backend_name,
        int participant_count,
        int routed_layer_count)
    {
        SCOPED_TRACE(backend_name + " portable MoE device rehydration");
        ASSERT_GT(participant_count, 0);
        ASSERT_GT(routed_layer_count, 0);

        const double expected_transactions =
            static_cast<double>(participant_count * 2);
        EXPECT_GE(
            perfCounterValue(
                records,
                "prefix_cache",
                "moe_portable_runtime_state_restores",
                "prefix_cache"),
            expected_transactions)
            << "Both exact-hit decode and partial-hit suffix prefill must "
               "restore a portable runtime table on every participant.";
        EXPECT_GE(
            perfCounterValue(
                records,
                "prefix_cache",
                "moe_portable_runtime_device_rehydrations",
                "prefix_cache"),
            expected_transactions)
            << "Every adopted restore transaction must publish successful "
               "device completion before inference continues.";

        const double layer_transactions =
            perfCounterValue(
                records,
                "prefix_cache",
                "moe_layer_device_payload_rehydrations",
                "decode") +
            perfCounterValue(
                records,
                "prefix_cache",
                "moe_layer_device_payload_rehydrations",
                "prefill");
        EXPECT_GE(
            layer_transactions,
            static_cast<double>(
                participant_count * routed_layer_count * 2))
            << "Every routed layer on every participant must execute the "
               "rehydration transaction in both M=1 and suffix-prefill graphs.";
    }

    /**
     * @brief Prove that reusable GPU request-input banks close both event edges.
     *
     * Long-context LLEP executes several stable prefill transactions against one
     * persistent token/position/length arena bank. Correct outputs alone are not
     * sufficient evidence for its lifetime: an unprotected overwrite can pass
     * whenever graph execution happens to outrun the next H2D admission. Require
     * every LocalTP participant to observe writer-to-reader admission, publish
     * transitive last-reader completion, and wait for that release before at
     * least one subsequent bank write.
     */
    void expectRequestInputReuseCoverage(
        const std::vector<PerfStatRecord> &records,
        const std::string &backend_name,
        int participant_count)
    {
        SCOPED_TRACE(backend_name + " request-input bank lifetime");
        ASSERT_GT(participant_count, 0);
        const double expected_participants =
            static_cast<double>(participant_count);

        EXPECT_GE(
            perfCounterValue(
                records,
                "request_admission",
                "device_input_event_waits",
                "prefill"),
            expected_participants)
            << "Every participant must order its main graph after request admission.";
        EXPECT_GE(
            perfCounterValue(
                records,
                "request_admission",
                "device_input_reuse_publications",
                "prefill"),
            expected_participants)
            << "Every participant must publish completion of the complete input-reader chain.";
        EXPECT_GE(
            perfCounterValue(
                records,
                "request_admission",
                "device_input_reuse_waits",
                "prefill"),
            expected_participants)
            << "Stable/split prefill must reuse each input bank through the reader-completion event.";
    }

    /**
     * @brief Sum a decode graph lifecycle counter by execution context.
     *
     * Phase 6 MTP graph capture relies on named forward-graph contexts
     * (`main_decode`, `main_verifier`, sidecar contexts, and later TP variants).
     * The lifecycle counter is intentionally small and stable: each record says
     * which context ran and whether that step was warmup, capture, or replay.
     * Tests should assert the phase shape without depending on exact decode
     * token counts, since speculative acceptance can change the number of
     * iterations a prompt needs.
     */
    double decodeGraphPhaseCount(
        const std::vector<PerfStatRecord> &records,
        const std::string &context,
        const std::string &capture_phase)
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph" ||
                record.name != "decode_graph_phase" ||
                record.phase != "decode")
            {
                continue;
            }

            const auto context_it = record.tags.find("context");
            const auto phase_it = record.tags.find("phase");
            if (context_it == record.tags.end() ||
                phase_it == record.tags.end() ||
                phase_it->second != capture_phase)
            {
                continue;
            }

            const std::string &actual_context = context_it->second;
            const bool exact_match = actual_context == context;
            const bool family_match =
                actual_context.rfind(context + "_", 0) == 0;
            if (exact_match || family_match)
                total += record.value;
        }
        return total;
    }

    /**
     * @brief Sum a named MTP counter from an already-captured perf snapshot.
     */
    double mtpCounterValue(
        const std::vector<PerfStatRecord> &records,
        const std::string &name)
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == "mtp" &&
                record.name == name &&
                record.phase == "decode")
            {
                total += record.value;
            }
        }
        return total;
    }

    enum class MTPVerifierGraphPath
    {
        None,
        AllPositionStatePublication,
        DecodeEquivalent,
    };

    /**
     * @brief Identify which verifier contract executed for the current MTP run.
     *
     * Direct all-position state publication stays fail-closed until a
     * backend/model lane proves the stronger continuation contract.  The
     * baseline production lane is grouped decode-equivalent verification, for
     * both greedy and stochastic sampling.  That lane must still use graph-
     * captured main decode and catch-up contexts. This helper lets probes assert
     * the active contract instead of baking in an all-position-only expectation.
     */
    MTPVerifierGraphPath mtpVerifierGraphPath(
        const std::vector<PerfStatRecord> &records)
    {
        if (mtpCounterValue(records, "all_position_state_publication_verifier_runs") >= 1.0)
            return MTPVerifierGraphPath::AllPositionStatePublication;
        if (mtpCounterValue(records, "grouped_decode_equivalent_greedy_verifier_runs") >= 1.0 ||
            mtpCounterValue(records, "grouped_decode_equivalent_stochastic_verifier_runs") >= 1.0 ||
            mtpCounterValue(records, "grouped_outcome_device_resident_publication_uses") >= 1.0 ||
            mtpCounterValue(records, "grouped_outcome_host_publication_uses") >= 1.0)
            return MTPVerifierGraphPath::DecodeEquivalent;
        return MTPVerifierGraphPath::None;
    }

    /**
     * @brief Assert that a graph-captured MTP context reached the expected phases.
     */
    void expectSegmentedGraphLifecycle(
        const std::vector<PerfStatRecord> &records,
        const std::string &backend_name,
        const std::string &context,
        bool require_warmup_capture,
        bool require_replay)
    {
        SCOPED_TRACE(backend_name + " " + context);
        if (require_warmup_capture)
        {
            EXPECT_GE(decodeGraphPhaseCount(records, context, "warmup"), 1.0)
                << context << " must execute an explicit warmup before graph capture";
            EXPECT_GE(decodeGraphPhaseCount(records, context, "capture"), 1.0)
                << context << " must record a graph capture before replay";
        }
        if (require_replay)
        {
            EXPECT_GE(decodeGraphPhaseCount(records, context, "replay"), 1.0)
                << context << " must replay a previously captured graph";
        }
    }

    /**
     * @brief Assert graph replay for the active MTP verifier path.
     *
     * Direct all-position publication replays the `main_verifier` graph.  The
     * Phase 9.7-supported decode-equivalent lane replays ordinary `main_decode`
     * plus `mtp_decode_catchup` graphs while preserving the same verifier math
     * and accepted-state contract.
     */
    void expectMTPVerifierGraphLifecycle(
        const std::vector<PerfStatRecord> &records,
        const std::string &backend_name,
        bool require_warmup_capture,
        bool require_replay)
    {
        const MTPVerifierGraphPath path = mtpVerifierGraphPath(records);
        ASSERT_NE(path, MTPVerifierGraphPath::None)
            << backend_name << " MTP must execute a supported verifier path";

        if (path == MTPVerifierGraphPath::AllPositionStatePublication)
        {
            expectSegmentedGraphLifecycle(
                records,
                backend_name,
                "main_verifier",
                require_warmup_capture,
                require_replay);
            return;
        }

        expectSegmentedGraphLifecycle(
            records,
            backend_name,
            "main_decode",
            require_warmup_capture,
            require_replay);
        expectSegmentedGraphLifecycle(
            records,
            backend_name,
            "mtp_decode_catchup",
            require_warmup_capture,
            require_replay);
    }

    /**
     * @brief Assert that accepted MTP state was published through a fast path.
     *
     * Greedy and older host-planned paths may materialize accepted shifted rows
     * by replaying the `mtp_decode_catchup` graph. The vLLM-style stochastic
     * path can skip that graph entirely by publishing KV/GDN/hidden state from
     * compact device-resident verifier metadata. Both are valid fast paths; a
     * test failure here means the run fell back to neither.
     */
    void expectMTPAcceptedStateFastPublication(
        const std::vector<PerfStatRecord> &records,
        const std::string &backend_name)
    {
        const bool catchup_replayed =
            decodeGraphPhaseCount(records, "mtp_decode_catchup", "replay") >= 1.0;
        const bool resident_published =
            mtpCounterValue(records, "device_resident_state_publications") >= 1.0 &&
            mtpCounterValue(records, "device_resident_kv_sequence_state_publications") >= 1.0 &&
            mtpCounterValue(records, "spec_state_publications") >= 1.0;
        EXPECT_TRUE(catchup_replayed || resident_published)
            << backend_name << " MTP accepted-state publication must either replay "
            << "the catch-up graph or use direct device-resident publication";
    }

    std::string lowercase(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return value;
    }

    std::string firstEnvOrDefault(
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

    int firstIntEnvOrDefault(
        const std::vector<std::string> &names,
        int fallback)
    {
        for (const auto &name : names)
        {
            const char *value = std::getenv(name.c_str());
            if (!value || !*value)
            {
                continue;
            }

            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end != value && end && *end == '\0')
            {
                return static_cast<int>(parsed);
            }
        }
        return fallback;
    }

    int qwen36RocmSingleDeviceOrdinal()
    {
        return firstIntEnvOrDefault(
            {"LLAMINAR_QWEN36_ROCM_DEVICE", "LLAMINAR_TEST_ROCM_DEVICE"},
            0);
    }

    int qwen36CudaSingleDeviceOrdinal()
    {
        return firstIntEnvOrDefault(
            {"LLAMINAR_QWEN36_CUDA_DEVICE", "LLAMINAR_TEST_CUDA_DEVICE"},
            0);
    }

    RoutedExpertDomain qwen36MoELocalTPDomainForProbe(
        const std::string &name,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        RoutedExpertDomain domain;
        domain.name = name;
        domain.scope = ExecutionDomainScope::LOCAL;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.owner_rank = 0;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    RoutedExpertTier qwen36MoERoutedTierForProbe(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        size_t memory_budget_bytes)
    {
        RoutedExpertTier tier;
        tier.name = name;
        tier.domain = domain;
        tier.priority = priority;
        tier.max_experts_per_layer = max_experts_per_layer;
        tier.memory_budget_bytes = memory_budget_bytes;
        return tier;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> qwen36MoEOverlayRocm2TPHotOnlyForProbe(
        RoutedExpertAssignmentPolicy routed_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner)
    {
        constexpr const char *kRocmHotDomain = "qwen36_moe_rocm_hot";

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->continuation_domain = kRocmHotDomain;
        plan->shared_expert_domain = kRocmHotDomain;
        plan->domains = {
            qwen36MoELocalTPDomainForProbe(
                kRocmHotDomain,
                CollectiveBackendType::RCCL,
                {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
        };
        plan->domains.front().routed_assignment_policy = routed_assignment_policy;
        plan->routed_tiers = {
            qwen36MoERoutedTierForProbe(
                "hot",
                kRocmHotDomain,
                0,
                256,
                8ull * 1024ull * 1024ull * 1024ull),
        };
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
        return plan;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> qwen36MoEOverlayCuda2TPHotOnlyForProbe(
        RoutedExpertAssignmentPolicy routed_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner)
    {
        constexpr const char *kCudaHotDomain = "qwen36_moe_cuda_hot";

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->continuation_domain = kCudaHotDomain;
        plan->shared_expert_domain = kCudaHotDomain;
        plan->domains = {
            qwen36MoELocalTPDomainForProbe(
                kCudaHotDomain,
                CollectiveBackendType::NCCL,
                {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}),
        };
        plan->domains.front().routed_assignment_policy = routed_assignment_policy;
        plan->routed_tiers = {
            qwen36MoERoutedTierForProbe(
                "hot",
                kCudaHotDomain,
                0,
                256,
                8ull * 1024ull * 1024ull * 1024ull),
        };
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
        return plan;
    }

    /**
     * @brief Render the state owners needed to diagnose split-prefill drift.
     *
     * A full-attention KV mismatch is often only the first durable downstream
     * symptom of an earlier GDN carry-state error.  Keep both owners in the
     * same failure summary and include the prefill graph lifecycle so a test
     * failure says whether each exact shape ran eagerly, warmed a graph, or
     * replayed a previously captured executable.
     *
     * @param probe Request-boundary state snapshot to render.
     * @param layer_limit Maximum number of full-attention KV layers to print.
     * @return Compact single-line diagnostic suitable for a GoogleTest error.
     */
    std::string summarizeStateContinuityProbe(
        const PrefixRuntimeStateSnapshot &probe,
        size_t layer_limit = 4)
    {
        std::ostringstream oss;
        oss << "pos=" << probe.current_position
            << " seqs=";
        for (size_t i = 0; i < probe.sequence_lengths.size(); ++i)
        {
            if (i > 0)
                oss << ",";
            oss << probe.sequence_lengths[i];
        }
        oss << " kv_tokens=" << probe.totalCachedTokens()
            << " gdn_layers=" << probe.gdn_layers.size()
            << " prefix_hits=" << probe.prefix_cache_hits
            << " prefix_partial_hits=" << probe.prefix_cache_partial_hits
            << " matched_tokens=" << probe.prefix_cache_matched_tokens
            << " moe_epoch=" << probe.moe_runtime_movement_epoch
            << " live_epoch=" << probe.live_state_epoch
            << " live_mutations=" << probe.live_state_mutations;

        size_t emitted = 0;
        for (const auto &cache : probe.kv_caches)
        {
            for (const auto &layer : cache.layers)
            {
                if (emitted++ >= layer_limit)
                    break;
                oss << " KV" << layer.global_layer
                    << "/n=" << layer.cached_tokens
                    << "/k=" << layer.k_payload_hash
                    << "/v=" << layer.v_payload_hash;
                for (const auto &segment : layer.segments)
                {
                    oss << "/" << segment.name
                        << "=" << (segment.hash_available ? "h" : "na")
                        << ":" << segment.k_payload_hash
                        << ":" << segment.v_payload_hash;
                }
            }
            if (emitted >= layer_limit)
                break;
        }

        for (const auto &layer : probe.gdn_layers)
        {
            oss << " GDN" << layer.global_layer
                << "/rec_bytes=" << layer.recurrence_device_bytes
                << "/rec=" << layer.recurrence_device_hash
                << "/conv_bytes=" << layer.conv_device_bytes
                << "/conv=" << layer.conv_device_hash
                << "/local_rec=" << layer.recurrence_local_device_hash
                << "/local_conv=" << layer.conv_local_device_hash;
        }

        for (const auto &graph : probe.prefill_graphs)
        {
            oss << " GRAPH"
                << "/bucket=" << graph.bucket_seq_len
                << "/real=" << graph.real_token_start
                << ":" << graph.real_token_end
                << "/phase=" << graph.phase
                << "/capture=" << graph.capture_phase
                << "/warmups=" << graph.warmup_count
                << "/captures=" << graph.capture_count
                << "/replays=" << graph.replay_count
                << "/reason=" << graph.recapture_reason;
        }
        return oss.str();
    }

    /**
     * @brief List every recurrent and KV payload owner that differs.
     *
     * compareMTPRuntimeStateSnapshots() intentionally returns the first broken
     * invariant.  Long-context debugging needs the wider propagation pattern:
     * a GDN mismatch at layer 4 followed by a KV mismatch at layer 7 points to
     * a very different producer than an isolated layer-7 append error.  This
     * helper reports that pattern without copying model tensors to the host.
     *
     * @param oracle State produced by one monolithic prefill.
     * @param candidate State produced by split prefill or prefix restoration.
     * @return Compact mismatch inventory based on device-owned byte hashes.
     */
    std::string summarizeStateContinuityDifferences(
        const PrefixRuntimeStateSnapshot &oracle,
        const PrefixRuntimeStateSnapshot &candidate)
    {
        std::ostringstream oss;
        oss << "differences:";
        bool found = false;

        const size_t gdn_count = std::min(
            oracle.gdn_layers.size(),
            candidate.gdn_layers.size());
        for (size_t index = 0; index < gdn_count; ++index)
        {
            const auto &lhs = oracle.gdn_layers[index];
            const auto &rhs = candidate.gdn_layers[index];
            if (lhs.global_layer != rhs.global_layer ||
                lhs.recurrence_device_bytes != rhs.recurrence_device_bytes ||
                lhs.conv_device_bytes != rhs.conv_device_bytes ||
                lhs.recurrence_device_hash != rhs.recurrence_device_hash ||
                lhs.conv_device_hash != rhs.conv_device_hash ||
                lhs.recurrence_local_device_hash != rhs.recurrence_local_device_hash ||
                lhs.conv_local_device_hash != rhs.conv_local_device_hash)
            {
                found = true;
                oss << " GDN[" << lhs.global_layer << "/" << rhs.global_layer
                    << "] rec=" << lhs.recurrence_device_hash
                    << "/" << rhs.recurrence_device_hash
                    << " conv=" << lhs.conv_device_hash
                    << "/" << rhs.conv_device_hash
                    << " local_rec=" << lhs.recurrence_local_device_hash
                    << "/" << rhs.recurrence_local_device_hash
                    << " local_conv=" << lhs.conv_local_device_hash
                    << "/" << rhs.conv_local_device_hash;
            }
        }
        if (oracle.gdn_layers.size() != candidate.gdn_layers.size())
        {
            found = true;
            oss << " GDN_count=" << oracle.gdn_layers.size()
                << "/" << candidate.gdn_layers.size();
        }

        const size_t cache_count = std::min(
            oracle.kv_caches.size(),
            candidate.kv_caches.size());
        for (size_t cache_index = 0; cache_index < cache_count; ++cache_index)
        {
            const auto &lhs_cache = oracle.kv_caches[cache_index];
            const auto &rhs_cache = candidate.kv_caches[cache_index];
            const size_t layer_count = std::min(
                lhs_cache.layers.size(),
                rhs_cache.layers.size());
            for (size_t layer_index = 0; layer_index < layer_count; ++layer_index)
            {
                const auto &lhs = lhs_cache.layers[layer_index];
                const auto &rhs = rhs_cache.layers[layer_index];
                if (lhs.global_layer != rhs.global_layer ||
                    lhs.cached_tokens != rhs.cached_tokens ||
                    lhs.ring_head != rhs.ring_head ||
                    lhs.k_payload_hash != rhs.k_payload_hash ||
                    lhs.v_payload_hash != rhs.v_payload_hash)
                {
                    found = true;
                    oss << " KV" << cache_index
                        << "[" << lhs.global_layer << "/" << rhs.global_layer
                        << "] tokens=" << lhs.cached_tokens
                        << "/" << rhs.cached_tokens
                        << " ring=" << lhs.ring_head
                        << "/" << rhs.ring_head
                        << " k=" << lhs.k_payload_hash
                        << "/" << rhs.k_payload_hash
                        << " v=" << lhs.v_payload_hash
                        << "/" << rhs.v_payload_hash;
                }
            }
            if (lhs_cache.layers.size() != rhs_cache.layers.size())
            {
                found = true;
                oss << " KV" << cache_index << "_layer_count="
                    << lhs_cache.layers.size() << "/"
                    << rhs_cache.layers.size();
            }
        }
        if (oracle.kv_caches.size() != candidate.kv_caches.size())
        {
            found = true;
            oss << " KV_cache_count=" << oracle.kv_caches.size()
                << "/" << candidate.kv_caches.size();
        }

        if (!found)
            oss << " none in device GDN/KV hashes";
        return oss.str();
    }

    GenerationResult decodeGreedyTokens(
        IOrchestrationRunner &runner,
        int max_tokens,
        const char *label)
    {
        GenerationResult result;
        int remaining = max_tokens;
        while (remaining > 0)
        {
            runner.setDecodeStepTokenBudget(remaining);
            GenerationResult step = runner.decodeStep();
            runner.setDecodeStepTokenBudget(0);
            if (!step.error.empty())
            {
                result.error = std::string(label) + ": " + step.error;
                return result;
            }
            if (step.tokens.empty())
            {
                result.error = std::string(label) + ": decodeStep produced no tokens";
                return result;
            }
            if (static_cast<int>(step.tokens.size()) > remaining)
            {
                result.error = std::string(label) + ": decodeStep exceeded token budget";
                return result;
            }
            result.tokens.insert(result.tokens.end(), step.tokens.begin(), step.tokens.end());
            remaining -= static_cast<int>(step.tokens.size());
            if (step.is_complete)
                break;
        }
        return result;
    }

    std::string compactPreview(const std::string &text, size_t limit = 220)
    {
        std::ostringstream compact;
        bool previous_space = false;
        for (char ch : text)
        {
            if (std::isspace(static_cast<unsigned char>(ch)) != 0)
            {
                if (!previous_space)
                    compact << ' ';
                previous_space = true;
            }
            else
            {
                compact << ch;
                previous_space = false;
            }
        }
        std::string value = compact.str();
        if (value.size() <= limit)
            return value;
        if (limit <= 3)
            return value.substr(0, limit);
        return value.substr(0, limit - 3) + "...";
    }

    const std::vector<std::string> &longContextCodeWordsA()
    {
        static const std::vector<std::string> words = {
            "amber", "basil", "cedar", "delta", "ember", "fable",
            "garnet", "harbor", "iris", "juniper", "kelp", "laurel"};
        return words;
    }

    const std::vector<std::string> &longContextCodeWordsB()
    {
        static const std::vector<std::string> words = {
            "atlas", "beacon", "cobalt", "dawn", "elm", "fjord",
            "grove", "haven", "ion", "jasmine", "keystone", "lagoon"};
        return words;
    }

    const std::vector<std::string> &longContextCodeWordsC()
    {
        static const std::vector<std::string> words = {
            "north", "south", "east", "west", "upper", "lower",
            "inner", "outer", "prime", "quiet", "rapid", "steady"};
        return words;
    }

    std::string deterministicLongContextCode(
        const std::string &name_space,
        int index)
    {
        int seed = 0;
        for (char ch : name_space)
            seed += static_cast<unsigned char>(ch);

        const auto &a = longContextCodeWordsA();
        const auto &b = longContextCodeWordsB();
        const auto &c = longContextCodeWordsC();
        return a[(index + seed) % static_cast<int>(a.size())] + " " +
               b[((index / static_cast<int>(a.size())) + seed) %
                 static_cast<int>(b.size())] +
               " " +
               c[((index / static_cast<int>(a.size() * b.size())) + seed) %
                 static_cast<int>(c.size())];
    }

    int longContextRecallRecordCount(
        int min_prompt_tokens,
        int context_length,
        int max_tokens,
        const std::string &tier)
    {
        constexpr int kMinRecallRecords = 32;
        constexpr int kEstimatedTokensPerRecallRecord = 42;
        const int safety_margin = std::max(512, context_length / 8);
        const int prompt_budget =
            std::max(0, context_length - max_tokens - safety_margin);
        const int tier_extra_tokens = tier == "full" ? 900 : 360;
        const double context_bonus_rate = tier == "full" ? 0.18 : 0.10;
        const int context_bonus_tokens = static_cast<int>(
            std::min(std::max(0, context_length - 4096), 2048) *
            context_bonus_rate);
        const int target_prompt_tokens = std::min(
            prompt_budget,
            min_prompt_tokens + tier_extra_tokens + context_bonus_tokens);
        const int by_prompt_target =
            target_prompt_tokens / kEstimatedTokensPerRecallRecord;
        const int by_context_cap =
            prompt_budget / kEstimatedTokensPerRecallRecord;
        if (by_context_cap < kMinRecallRecords)
            return kMinRecallRecords;
        return std::max(
            kMinRecallRecords,
            std::min(by_prompt_target, by_context_cap));
    }

    std::string longContextAuditRecord(
        int index,
        const std::string &code,
        const std::string &name_space)
    {
        const int checksum =
            1000 + ((index * 37 + static_cast<int>(name_space.size()) * 97) % 9000);
        const char lane = static_cast<char>('A' + (index % 6));
        std::ostringstream oss;
        oss << "Ledger filler " << std::setw(4) << std::setfill('0') << index
            << ": distractor audit value \"" << code << "\". "
            << "Lane " << lane << "; checksum " << checksum
            << "; status normal; this is not the requested value.";
        return oss.str();
    }

    struct NeedlePromptForProbe
    {
        std::vector<ChatMessage> messages;
        std::string target_code;
        std::vector<std::string> distractors;
        int record_count = 0;
    };

    NeedlePromptForProbe buildNeedlePromptForProbe(
        const std::string &placement,
        int min_prompt_tokens,
        int context_length,
        int max_tokens,
        const std::string &tier)
    {
        const int count = longContextRecallRecordCount(
            min_prompt_tokens,
            context_length,
            max_tokens,
            tier);
        int target_index = 0;
        std::string target_key;
        std::string target_value;
        if (placement == "beginning")
        {
            target_index = std::min(3, count - 1);
            target_key = "alpha";
            target_value = "TUNDRA-84QX";
        }
        else if (placement == "middle")
        {
            target_index = count / 2;
            target_key = "middle";
            target_value = "COBALT-27LM";
        }
        else
        {
            target_index = std::max(0, count - 5);
            target_key = "omega";
            target_value = "RIVER-93RN";
        }

        std::string namespace_suffix = placement.substr(0, 3);
        std::transform(
            namespace_suffix.begin(),
            namespace_suffix.end(),
            namespace_suffix.begin(),
            [](unsigned char ch)
            { return static_cast<char>(std::toupper(ch)); });
        const std::string name_space = "LCN-" + namespace_suffix;

        std::vector<std::string> records;
        std::vector<std::string> codes;
        records.reserve(static_cast<size_t>(count));
        codes.reserve(static_cast<size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            const std::string code =
                index == target_index
                    ? target_value
                    : deterministicLongContextCode(name_space, index);
            codes.push_back(code);
            std::ostringstream row;
            if (index == target_index)
            {
                row << "Ledger item " << std::setw(4) << std::setfill('0') << index
                    << ": REQUIRED_JSON_FIELD " << target_key
                    << " has exact value " << code
                    << ". This is the only requested field.";
                records.push_back(row.str());
            }
            else
            {
                records.push_back(longContextAuditRecord(index, code, name_space));
            }
        }

        std::vector<std::string> distractors;
        for (int neighbor : {target_index - 2, target_index - 1,
                             target_index + 1, target_index + 2})
        {
            if (neighbor >= 0 && neighbor < count)
                distractors.push_back(codes[static_cast<size_t>(neighbor)]);
        }

        std::ostringstream user_prompt;
        user_prompt
            << "Task: read the ledger and return one minified JSON object.\n"
            << "The only allowed key is answer.\n"
            << "Only the REQUIRED_JSON_FIELD named " << target_key
            << " matters for the final answer.\n";
        for (const auto &record : records)
            user_prompt << record << "\n";
        user_prompt
            << "Return exactly one minified JSON object and no prose.\n"
            << "The object shape is {\"answer\":\"VALUE_FROM_LEDGER\"}.\n"
            << "Use the exact REQUIRED_JSON_FIELD value for " << target_key
            << " from the ledger.\n"
            << "Copy every letter, digit, and hyphen in the value; never shorten a value to a suffix.";

        NeedlePromptForProbe prompt;
        prompt.messages = {
            ChatMessage{
                "system",
                "<|think_off|>\nYou are a strict JSON renderer. Output JSON only."},
            ChatMessage{"user", user_prompt.str()},
        };
        prompt.target_code = codes[static_cast<size_t>(target_index)];
        prompt.distractors = std::move(distractors);
        prompt.record_count = count;
        return prompt;
    }

    struct NeedleRecallRunResult
    {
        std::string placement;
        std::string target_code;
        std::string content;
        std::string error;
        int prompt_tokens = 0;
        int generated_tokens = 0;

        bool containsTarget() const
        {
            return error.empty() &&
                   content.find(target_code) != std::string::npos;
        }
    };

    NeedleRecallRunResult runNeedleRecallPrompt(
        IOrchestrationRunner &runner,
        const std::string &placement,
        int context_length,
        int max_tokens)
    {
        NeedleRecallRunResult result;
        result.placement = placement;
        auto tokenizer = runner.tokenizer();
        if (!tokenizer)
        {
            result.error = "runner has no tokenizer";
            return result;
        }

        const NeedlePromptForProbe prompt =
            buildNeedlePromptForProbe(
                placement,
                900,
                context_length,
                max_tokens,
                "full");
        result.target_code = prompt.target_code;
        const std::vector<int> encoded =
            tokenizer->encodeChat(
                prompt.messages,
                /*add_generation_prompt=*/true,
                /*tools_json=*/"",
                /*enable_thinking=*/false);
        result.prompt_tokens = static_cast<int>(encoded.size());
        std::vector<int32_t> prompt_tokens(encoded.begin(), encoded.end());

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        GenerationResult generated;
        try
        {
            runner.clearCache();
            generated = runner.generate(prompt_tokens, max_tokens, greedy);
        }
        catch (const std::exception &e)
        {
            result.error = std::string("exception: ") + e.what();
            return result;
        }
        catch (...)
        {
            result.error = "unknown exception";
            return result;
        }
        if (!generated.error.empty())
        {
            result.error = generated.error;
            return result;
        }
        result.generated_tokens = static_cast<int>(generated.tokens.size());
        std::vector<int> output_tokens(
            generated.tokens.begin(),
            generated.tokens.end());
        result.content = tokenizer->decode(output_tokens, /*remove_special=*/true);
        return result;
    }

    std::string summarizeNeedleRecallResult(const NeedleRecallRunResult &result)
    {
        std::ostringstream oss;
        oss << result.placement
            << " target=" << result.target_code
            << " prompt_tokens=" << result.prompt_tokens
            << " generated_tokens=" << result.generated_tokens;
        if (!result.error.empty())
            oss << " error=" << result.error;
        else
            oss << " content=" << compactPreview(result.content, 220);
        return oss.str();
    }

    struct OverlayRecallProbeConfig
    {
        const char *label = "overlay";
        MoERebalanceRuntimeMode mode = MoERebalanceRuntimeMode::LLEP;
        RoutedExpertAssignmentPolicy routed_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        bool prefix_cache = true;
        int prefill_window_tokens = 0;
        bool require_transfer_backing = true;
    };

    std::string formatTokenWindow(
        const std::vector<int32_t> &tokens,
        size_t center,
        size_t radius = 6)
    {
        if (tokens.empty())
            return "[]";
        const size_t begin = center > radius ? center - radius : 0;
        const size_t end = std::min(tokens.size(), center + radius + 1);
        std::ostringstream oss;
        oss << "[";
        for (size_t i = begin; i < end; ++i)
        {
            if (i != begin)
                oss << ", ";
            if (i == center)
                oss << "*";
            oss << tokens[i];
        }
        oss << "]";
        return oss.str();
    }

    /**
     * @brief Build a deterministic prompt with at least the requested token count.
     *
     * The graph-stress tests need enough prompt history to exercise long-context
     * attention bucketing before MTP decode starts.  Using the real tokenizer
     * keeps the input model-shaped while the repeated numbered clauses make the
     * token stream stable across runs and easy to reproduce when a test fails.
     */
    std::vector<int32_t> buildDeterministicPromptTokens(
        ITokenizer &tokenizer,
        size_t requested_tokens)
    {
        if (requested_tokens == 0)
        {
            const auto encoded = tokenizer.encode(
                "The quick brown fox",
                /*add_bos=*/false,
                /*add_eos=*/false);
            return std::vector<int32_t>(encoded.begin(), encoded.end());
        }

        std::ostringstream prompt;
        for (size_t i = 0; i < requested_tokens; ++i)
        {
            prompt << "Section " << i
                   << ": The quick brown fox writes a deterministic CUDA and ROCm "
                   << "kernel note with repeated verifier state, graph capture, "
                   << "attention buckets, and speculative decoding evidence.\n";
        }

        auto encoded = tokenizer.encode(prompt.str(), /*add_bos=*/false, /*add_eos=*/false);
        if (encoded.size() < requested_tokens)
        {
            ADD_FAILURE()
                << "deterministic long prompt seed should encode to enough tokens";
            return {};
        }
        encoded.resize(requested_tokens);
        return std::vector<int32_t>(encoded.begin(), encoded.end());
    }

    /**
     * @brief Run one greedy MTP GPU-graph smoke on a concrete backend.
     *
     * This is intentionally symmetric with the stochastic helper below: Phase 6
     * acceptance depends on CUDA and ROCm proving the same captured verifier,
     * sidecar, and catch-up graph lifecycle for both greedy and stochastic MTP.
     */
    void runQwen36MTPGpuGraphsGreedyRealModelSmoke(
        GlobalDeviceAddress device,
        const std::string &backend_name,
        size_t prompt_token_count = 0,
        size_t decode_token_count = 8,
        int draft_tokens = 1)
    {
        ASSERT_GT(decode_token_count, 0u);
        ASSERT_GT(draft_tokens, 0);

        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_qwen36_greedy_mtp_graph_stats.json"},
            {"LLAMINAR_PERF_STATS_FILTER", "mtp,forward_graph"},
        });
        PerfStatsCollector::reset();

        const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
        if (!env_model)
            env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
        const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
        }

        const size_t planned_prompt_tokens =
            prompt_token_count == 0 ? 16 : prompt_token_count;
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = static_cast<int>(
            std::max<size_t>(128, planned_prompt_tokens + decode_token_count + 64));
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.device_for_this_rank = device;
        config.kv_cache_precision = "auto";
        config.mtp.enabled = true;
        config.mtp.draft_tokens = draft_tokens;

        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        auto tokenizer = runner->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        const std::vector<int32_t> prompt =
            buildDeterministicPromptTokens(*tokenizer, prompt_token_count);
        ASSERT_FALSE(prompt.empty());

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        runner->setSamplingParams(greedy);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        auto run_benchmark_style_cycle = [&](int cycle) -> std::vector<int32_t>
        {
            runner->clearCache();
            std::vector<int32_t> tokens;
            if (!runner->prefill(prompt))
            {
                ADD_FAILURE() << "cycle " << cycle << ": " << runner->lastError();
                return tokens;
            }
            while (tokens.size() < decode_token_count)
            {
                const int remaining = static_cast<int>(decode_token_count - tokens.size());
                runner->setDecodeStepTokenBudget(remaining);
                auto step = runner->decodeStep();
                runner->setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    ADD_FAILURE() << "cycle " << cycle << ": " << step.error;
                    return tokens;
                }
                if (step.tokens.empty())
                {
                    ADD_FAILURE() << "cycle " << cycle
                                  << ": greedy MTP benchmark-style decode produced no tokens";
                    return tokens;
                }
                if (step.tokens.size() > static_cast<size_t>(remaining))
                {
                    ADD_FAILURE() << "cycle " << cycle
                                  << ": greedy MTP decode exceeded remaining token budget";
                    return tokens;
                }
                tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
            }
            return tokens;
        };

        const auto warmup_tokens = run_benchmark_style_cycle(-1);
        ASSERT_EQ(warmup_tokens.size(), decode_token_count);
        const auto result_tokens = run_benchmark_style_cycle(0);
        const auto snapshot = runner->prefixStateProbe();
        const auto records = PerfStatsCollector::snapshot({"mtp", "forward_graph"});
        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
        runner->shutdown();

        ASSERT_EQ(result_tokens.size(), decode_token_count);
        EXPECT_TRUE(snapshot.mtp_config_enabled);
        EXPECT_FALSE(snapshot.mtp_bypassed) << snapshot.mtp_bypass_reason;
        EXPECT_GE(snapshot.mtp_draft_steps, 1u);
        EXPECT_GE(snapshot.mtp_verifier_runs, 1u);
        EXPECT_GE(snapshot.mtp_accepted_tokens + snapshot.mtp_rejected_tokens, 1u);

        expectMTPVerifierGraphLifecycle(
            records,
            backend_name,
            /*require_warmup_capture=*/true,
            /*require_replay=*/true);
        expectSegmentedGraphLifecycle(
            records,
            backend_name,
            "mtp_decode_sidecar",
            /*require_warmup_capture=*/true,
            /*require_replay=*/true);
        expectMTPAcceptedStateFastPublication(
            records,
            backend_name);
        PerfStatsCollector::reset();
    }

    void runQwen36MTPGpuGraphsStochasticRealModelSmoke(
        GlobalDeviceAddress device,
        const std::string &backend_name,
        size_t decode_token_count = 8,
        int repeat_cycles = 2,
        bool deterministic_repeatability = true,
        bool use_presence_penalty = false,
        size_t prompt_token_count = 0)
    {
        ASSERT_GT(decode_token_count, 0u);
        ASSERT_GE(repeat_cycles, 2);

        std::optional<ScopedDebugEnv> deterministic_env;
        if (deterministic_repeatability)
        {
            deterministic_env.emplace(std::initializer_list<std::pair<const char *, const char *>>{
                {"LLAMINAR_DETERMINISTIC", "1"},
            });
        }

        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_qwen36_stochastic_mtp_stats.json"},
            {"LLAMINAR_PERF_STATS_FILTER", "mtp,forward_graph"},
        });
        PerfStatsCollector::reset();

        const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
        if (!env_model)
            env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
        const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
        }

        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        const size_t planned_prompt_tokens =
            prompt_token_count == 0 ? 16 : prompt_token_count;
        config.max_seq_len = static_cast<int>(
            std::max<size_t>(128, planned_prompt_tokens + decode_token_count + 64));
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.device_for_this_rank = device;
        config.kv_cache_precision = "auto";
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 1;
        config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        auto tokenizer = runner->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        const std::vector<int32_t> prompt =
            buildDeterministicPromptTokens(*tokenizer, prompt_token_count);
        ASSERT_FALSE(prompt.empty());

        SamplingParams stochastic;
        stochastic.temperature = 0.6f;
        stochastic.top_k = 20;
        stochastic.top_p = 0.95f;
        stochastic.presence_penalty = use_presence_penalty ? 0.25f : 0.0f;
        stochastic.seed = 123;

        runner->setSamplingParams(stochastic);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        auto run_benchmark_style_cycle = [&](int cycle) -> std::vector<int32_t>
        {
            runner->clearCache();
            std::vector<int32_t> tokens;
            if (!runner->prefill(prompt))
            {
                ADD_FAILURE() << "cycle " << cycle << ": " << runner->lastError();
                return tokens;
            }
            while (tokens.size() < decode_token_count)
            {
                const int remaining = static_cast<int>(decode_token_count - tokens.size());
                runner->setDecodeStepTokenBudget(remaining);
                auto step = runner->decodeStep();
                runner->setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    ADD_FAILURE() << "cycle " << cycle << ": " << step.error;
                    return tokens;
                }
                if (step.tokens.empty())
                {
                    ADD_FAILURE() << "cycle " << cycle
                                  << ": stochastic MTP benchmark-style decode produced no tokens";
                    return tokens;
                }
                if (step.tokens.size() > static_cast<size_t>(remaining))
                {
                    ADD_FAILURE() << "cycle " << cycle
                                  << ": stochastic MTP decode exceeded remaining token budget";
                    return tokens;
                }
                tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
            }
            return tokens;
        };

        auto warmup_tokens = run_benchmark_style_cycle(-1);
        ASSERT_EQ(warmup_tokens.size(), decode_token_count)
            << "stochastic MTP graph replay warmup must complete before repeatability checks";

        auto first_tokens = run_benchmark_style_cycle(0);
        ASSERT_EQ(first_tokens.size(), decode_token_count);

        const auto graph_lifecycle_records = PerfStatsCollector::snapshot({"mtp", "forward_graph"});
        /*
         * vLLM-style d1 stochastic decode does not need an ordinary
         * `main_decode` forward after prefill: the target verifier consumes the
         * ready prefill/accepted logits, while draft/catch-up work runs through
         * MTP sidecar contexts.  Phase 6 therefore guards the graph-shaped
         * verifier and sidecar lanes directly.
         */
        expectMTPVerifierGraphLifecycle(
            graph_lifecycle_records,
            backend_name,
            /*require_warmup_capture=*/true,
            /*require_replay=*/true);
        expectSegmentedGraphLifecycle(
            graph_lifecycle_records,
            backend_name,
            "mtp_decode_sidecar",
            /*require_warmup_capture=*/true,
            /*require_replay=*/true);
        expectMTPAcceptedStateFastPublication(
            graph_lifecycle_records,
            backend_name);

        PerfStatsCollector::reset();
        std::vector<int32_t> result_tokens;
        for (int cycle = 1; cycle < repeat_cycles; ++cycle)
        {
            result_tokens = run_benchmark_style_cycle(cycle);
            ASSERT_EQ(result_tokens.size(), decode_token_count);
            auto mismatch = std::mismatch(
                result_tokens.begin(),
                result_tokens.end(),
                first_tokens.begin(),
                first_tokens.end());
            if (deterministic_repeatability &&
                (mismatch.first != result_tokens.end() ||
                 mismatch.second != first_tokens.end()))
            {
                const size_t index = static_cast<size_t>(
                    std::distance(result_tokens.begin(), mismatch.first));
                ADD_FAILURE()
                    << "Deterministic stochastic MTP graph replay must be reproducible "
                    << "under LLAMINAR_DETERMINISTIC after clearCache() with the same "
                    << "prompt and seed after graph warmup (cycle=" << cycle
                    << ", first_mismatch_index=" << index << ")\n"
                    << "result window: " << formatTokenWindow(result_tokens, index) << "\n"
                    << "first  window: " << formatTokenWindow(first_tokens, index) << "\n"
                    << "warmup window: " << formatTokenWindow(warmup_tokens, index) << "\n"
                    << "result_matches_warmup=" << (result_tokens == warmup_tokens ? "true" : "false")
                    << ", first_matches_warmup=" << (first_tokens == warmup_tokens ? "true" : "false");
            }
        }
        const auto snapshot = runner->prefixStateProbe();
        const auto records = PerfStatsCollector::snapshot({"mtp", "forward_graph"});
        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
        runner->shutdown();

        expectMTPVerifierGraphLifecycle(
            records,
            backend_name,
            /*require_warmup_capture=*/false,
            /*require_replay=*/true);
        expectSegmentedGraphLifecycle(
            records,
            backend_name,
            "mtp_decode_sidecar",
            /*require_warmup_capture=*/false,
            /*require_replay=*/true);
        expectMTPAcceptedStateFastPublication(
            records,
            backend_name);

        auto counter = [&](const std::string &name)
        {
            return mtpCounterValue(records, name);
        };

        EXPECT_TRUE(snapshot.mtp_config_enabled);
        EXPECT_FALSE(snapshot.mtp_bypassed) << snapshot.mtp_bypass_reason;
        EXPECT_GE(snapshot.mtp_draft_steps, 1u);
        EXPECT_GE(snapshot.mtp_verifier_runs, 1u);
        EXPECT_GE(snapshot.mtp_verifier_token_count, 2u);
        EXPECT_GE(snapshot.mtp_accepted_tokens + snapshot.mtp_rejected_tokens, 1u)
            << "Speculative-sampling mode must actually verify at least one draft token on "
            << backend_name;
        EXPECT_GT(snapshot.mtp_transaction_commits, 0u)
            << backend_name << " stochastic MTP must publish through Phase 13.8 transactions";
        EXPECT_EQ(snapshot.mtp_transaction_validation_failures, 0u)
            << backend_name << " stochastic MTP transaction validation failed";
        EXPECT_EQ(snapshot.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(snapshot.mtp_request.stochastic_verify);
        EXPECT_GE(snapshot.mtp_stochastic_accept_tests, 1u);
        EXPECT_GE(snapshot.mtp_request.stochastic_accept_tests, 1u);
        EXPECT_EQ(snapshot.mtp_request.stochastic_accept_tests, snapshot.mtp_stochastic_accept_tests);
        EXPECT_EQ(snapshot.mtp_request.stochastic_accepts, snapshot.mtp_stochastic_accepts);
        EXPECT_EQ(snapshot.mtp_request.stochastic_residual_samples,
                  snapshot.mtp_stochastic_residual_samples);
        EXPECT_EQ(snapshot.mtp_request.stochastic_terminal_samples,
                  snapshot.mtp_stochastic_terminal_samples);
        EXPECT_GE(snapshot.mtp_request.stochastic_acceptance_rate, 0.0);
        EXPECT_LE(snapshot.mtp_request.stochastic_acceptance_rate, 1.0);
        EXPECT_GE(counter("first_token_stochastic_device_samples"), 1.0);
        const MTPVerifierGraphPath verifier_path =
            mtpVerifierGraphPath(records);
        ASSERT_NE(verifier_path, MTPVerifierGraphPath::None)
            << backend_name << " stochastic MTP must execute a supported verifier path";
        if (verifier_path == MTPVerifierGraphPath::AllPositionStatePublication)
        {
            EXPECT_GE(counter("all_position_state_publication_verifier_runs"), 1.0)
                << backend_name << " stochastic MTP must publish accepted verifier rows "
                << "through all-position state publication when that capability is advertised";
            EXPECT_EQ(counter("grouped_decode_equivalent_stochastic_verifier_runs"), 0.0)
                << backend_name << " all-position stochastic MTP must not also "
                << "run the grouped decode-equivalent verifier";
        }
        else
        {
            EXPECT_GE(counter("grouped_decode_equivalent_stochastic_verifier_runs"), 1.0)
                << backend_name << " stochastic MTP should use grouped "
                << "decode-equivalent verification while direct all-position "
                << "publication remains fail-closed";
            EXPECT_EQ(counter("all_position_state_publication_verifier_runs"), 0.0)
                << backend_name << " decode-equivalent stochastic MTP must not "
                << "claim direct all-position state publication";
        }
        if (backend_name == "ROCm" || backend_name == "CUDA")
        {
            EXPECT_GE(counter("stochastic_draft_greedy_proposals"), 1.0)
                << backend_name << " vLLM-style stochastic MTP must draft with "
                << "device argmax/one-hot q rather than building full draft probabilities";
            EXPECT_GE(counter("stochastic_topk_smallk_scratch_distribution_builds"), 1.0)
                << backend_name << " stochastic MTP target sampling must use the "
                << "arena-declared small-k top-k scratch path";
            if (verifier_path == MTPVerifierGraphPath::AllPositionStatePublication &&
                !use_presence_penalty)
            {
                EXPECT_GE(counter("first_token_stochastic_deferred_host_reads"), 1.0)
                    << backend_name << " penalty-free stochastic first tokens should stay device-resident until summary";
                EXPECT_GE(counter("mtp_token_stochastic_deferred_host_reads"), 1.0)
                    << backend_name << " penalty-free stochastic drafts should stay device-resident";
                EXPECT_GE(counter("sample_stochastic_distribution_deferred_host_reads"), 1.0)
                    << backend_name << " deferred draft sampling should avoid immediate scalar D2H reads";
                EXPECT_GE(counter("stochastic_target_sample_ready_events"), 1.0)
                    << backend_name << " deferred target samples must record a stream dependency";
                EXPECT_GE(counter("stochastic_target_sample_ready_waits"), 1.0)
                    << backend_name << " sidecar/verifier consumers must wait on deferred target samples";
                EXPECT_GE(counter("stochastic_draft_sample_ready_events"), 1.0)
                    << backend_name << " deferred draft samples must record a stream dependency";
                EXPECT_GE(counter("stochastic_draft_sample_ready_waits"), 1.0)
                    << backend_name << " sidecar/verifier consumers must wait on deferred draft samples";
                EXPECT_GE(counter("verifier_device_token_input_prepares"), 1.0)
                    << backend_name << " verifier input tokens should be staged from device draft slots";
                EXPECT_GE(counter("stochastic_batch_summary_device_first_tokens"), 1.0)
                    << backend_name << " verifier summary should read the first token from device scratch";
                EXPECT_GE(counter("all_position_stochastic_device_batched_rows"), 1.0)
                    << backend_name << " penalty-free stochastic verification should use the batched device outcome";
                EXPECT_EQ(counter("mtp_token_stochastic_device_samples"), 0.0)
                    << backend_name << " deferred draft samples should not force a host-visible token sample";
            }
            else if (verifier_path == MTPVerifierGraphPath::AllPositionStatePublication)
            {
                EXPECT_GE(counter("mtp_token_stochastic_device_samples"), 1.0)
                    << backend_name << " penalty paths still need a host-visible sampled token for sampler history";
            }
            else
            {
                EXPECT_GE(counter("stochastic_verify_batch_rows"), 1.0)
                    << backend_name << " decode-equivalent stochastic verification "
                    << "must still use the shared device-side batch verifier";
                EXPECT_GE(counter("stochastic_request_batch_draft_token_stages"), 1.0)
                    << backend_name << " decode-equivalent stochastic verification "
                    << "must stage draft tokens through device-owned request buffers";
                EXPECT_GE(counter("mtp_token_stochastic_device_samples"), 1.0)
                    << backend_name << " decode-equivalent stochastic verification "
                    << "keeps host-visible token shadows for sampler history";
            }
        }
        EXPECT_GE(counter("stochastic_accept_tests"), 1.0);
        EXPECT_GE(counter("transaction_validation_passes"), 1.0)
            << backend_name << " stochastic MTP must validate at least one Phase 13.8 transaction";
        EXPECT_EQ(counter("first_token_stochastic_samples"), 0.0)
            << backend_name << " stochastic MTP must not sample first token from host full logits";
        EXPECT_EQ(counter("mtp_token_stochastic_samples"), 0.0)
            << backend_name << " stochastic MTP must not sample sidecar drafts from host full logits";
        EXPECT_EQ(counter("verifier_stochastic_distributions"), 0.0)
            << backend_name << " stochastic MTP must not build verifier distributions from host full logits";
        EXPECT_EQ(counter("phase138_stochastic_spec_decode_runs"), 0.0)
            << backend_name << " stochastic MTP must not use the retired accepted-count-only fallback";
    }

    /**
     * @brief Prove request-batched prefill and its first MTP transaction remain GPU-owned.
     *
     * This regression deliberately uses two prefixes with unequal lengths. A
     * row-indexing bug, stale host length, or batch-index-keyed RNG draw therefore
     * changes at least one first token. Scalar one-request generation supplies the
     * production decode oracle; the request-batched path must return the same
     * prefill token and the same first grouped-verifier continuation for each
     * request, then report every device-resident ownership transition.
     *
     * Only the immutable prompt-length row crosses H2D during admission. Compact
     * terminal logits, position-keyed threshold derivation, sampled token slots,
     * compact accept/reject outcomes, collective outcome publication, and live
     * state mutation remain on explicitly ordered GPU streams. The host-visible
     * response tokens are inspected only after resident state publication.
     */
    void runQwen36MTPGpuRequestBatchResidentPrefill(
        GlobalDeviceAddress device,
        const std::string &backend_name,
        const std::vector<GlobalDeviceAddress> &local_tp_devices = {})
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT", "1"},
            {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT_LAYER", "3"},
            {"LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_qwen36_request_batch_prefill_stats.json"},
            {"LLAMINAR_PERF_STATS_FILTER", "mtp,forward_graph"},
        });

        const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
        if (!env_model)
            env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
        const std::string model_path =
            env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << "Qwen3.6 dense request-batch model not found: "
                         << model_path;
        }

        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 128;
        config.batch_size = 2;
        config.pp_degree = 1;
        if (local_tp_devices.empty())
        {
            config.tp_degree = 1;
            config.device_for_this_rank = device;
        }
        else
        {
            config.tp_degree = static_cast<int>(local_tp_devices.size());
            config.tp_scope = TPScope::LOCAL;
            config.tp_devices = local_tp_devices;
            config.default_backend = device.isCUDA()
                                         ? CollectiveBackendType::NCCL
                                         : CollectiveBackendType::RCCL;
        }
        config.kv_cache_precision = "auto";
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 1;
        config.mtp.max_request_batch = 2;
        config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        ASSERT_TRUE(runner->supportsPrefillBatch(/*request_batch=*/2));

        auto tokenizer = runner->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        const std::vector<int32_t> long_prompt =
            buildDeterministicPromptTokens(*tokenizer, /*requested_tokens=*/16);
        ASSERT_EQ(long_prompt.size(), 16u);
        const std::vector<int32_t> short_prompt(
            long_prompt.begin(),
            long_prompt.begin() + 11);
        const double participant_count =
            static_cast<double>(
                local_tp_devices.empty() ? 1 : local_tp_devices.size());

        SamplingParams stochastic;
        stochastic.temperature = 0.6f;
        stochastic.top_k = 20;
        stochastic.top_p = 0.95f;
        stochastic.seed = 0x4D545031u;

        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        auto run_batch_invariance_case = [&](const SamplingParams &params,
                                             const char *sampling_mode)
        {
            SCOPED_TRACE(
                backend_name + " request-batched prefill " + sampling_mode);

            const bool capture_stage_diagnostics = params.temperature == 0.0f;
            const std::vector<std::string> stage_comparison_keys =
                capture_stage_diagnostics
                    ? requestBatchPrefillSnapshotKeys()
                    : std::vector<std::string>{};
            std::vector<std::string> diagnostic_keys = stage_comparison_keys;
            if (capture_stage_diagnostics)
            {
                diagnostic_keys.push_back("layer3_ATTENTION_EFFECTIVE_K");
                diagnostic_keys.push_back("layer3_ATTENTION_EFFECTIVE_V");
            }
            if (capture_stage_diagnostics)
            {
                runner->setSnapshotCaptureFilter(diagnostic_keys);
                runner->enableSnapshotCapture();
                runner->clearSnapshots();
            }

            runner->clearCache();
            runner->setSamplingParams(params);
            ASSERT_TRUE(runner->prefill(long_prompt)) << runner->lastError();
            runner->setDecodeStepTokenBudget(1);
            const GenerationResult scalar_long = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);
            ASSERT_TRUE(scalar_long.error.empty()) << scalar_long.error;
            ASSERT_EQ(scalar_long.tokens.size(), 1u);
            const GenerationResult scalar_long_mtp = runner->decodeStep();
            ASSERT_TRUE(scalar_long_mtp.error.empty()) << scalar_long_mtp.error;
            ASSERT_FALSE(scalar_long_mtp.tokens.empty())
                << "Scalar long-prompt oracle must execute one MTP verifier transaction";
            std::vector<int32_t> scalar_long_tokens = scalar_long.tokens;
            scalar_long_tokens.insert(
                scalar_long_tokens.end(),
                scalar_long_mtp.tokens.begin(),
                scalar_long_mtp.tokens.end());
            const auto scalar_long_condition_snapshots =
                capture_stage_diagnostics
                    ? captureRequestBatchSnapshots(
                          *runner,
                          diagnostic_keys,
                          "MTP_SCALAR_CONDITION")
                    : std::map<std::string, RequestBatchStageSnapshot>{};
            if (capture_stage_diagnostics)
                runner->clearSnapshots();

            runner->clearCache();
            runner->setSamplingParams(params);
            ASSERT_TRUE(runner->prefill(short_prompt)) << runner->lastError();
            /*
             * Copy the prefill snapshots before decodeStep() launches the MTP
             * sidecar.  SnapshotCapture intentionally publishes bare aliases
             * for sidecar stages, so waiting until after decode would replace
             * the M=11 prefill rows with the subsequent M=1 draft rows.
             */
            const auto scalar_short_snapshots =
                capture_stage_diagnostics
                    ? captureRequestBatchSnapshots(*runner, diagnostic_keys)
                    : std::map<std::string, RequestBatchStageSnapshot>{};
            runner->setDecodeStepTokenBudget(1);
            const GenerationResult scalar_short = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);
            ASSERT_TRUE(scalar_short.error.empty()) << scalar_short.error;
            ASSERT_EQ(scalar_short.tokens.size(), 1u);
            const GenerationResult scalar_short_mtp = runner->decodeStep();
            ASSERT_TRUE(scalar_short_mtp.error.empty()) << scalar_short_mtp.error;
            ASSERT_FALSE(scalar_short_mtp.tokens.empty())
                << "Scalar short-prompt oracle must execute one MTP verifier transaction";
            std::vector<int32_t> scalar_short_tokens = scalar_short.tokens;
            scalar_short_tokens.insert(
                scalar_short_tokens.end(),
                scalar_short_mtp.tokens.begin(),
                scalar_short_mtp.tokens.end());
            const auto scalar_short_condition_snapshots =
                capture_stage_diagnostics
                    ? captureRequestBatchSnapshots(
                          *runner,
                          diagnostic_keys,
                          "MTP_SCALAR_CONDITION")
                    : std::map<std::string, RequestBatchStageSnapshot>{};

            runner->clearCache();
            if (capture_stage_diagnostics)
                runner->clearSnapshots();
            runner->setSamplingParams(params);
            PerfStatsCollector::reset();
            ASSERT_TRUE(runner->prefillBatch({long_prompt, short_prompt}))
                << runner->lastError();
            const auto batch_snapshots =
                capture_stage_diagnostics
                    ? captureRequestBatchSnapshots(*runner, diagnostic_keys)
                    : std::map<std::string, RequestBatchStageSnapshot>{};
            ASSERT_TRUE(runner->supportsDecodeStepBatch(/*request_batch=*/2));
            runner->setDecodeStepTokenBudget(1);
            const GenerationBatchResult batched_prefill =
                runner->decodeStepBatch(/*request_batch=*/2);
            runner->setDecodeStepTokenBudget(0);
            ASSERT_TRUE(batched_prefill.error.empty()) << batched_prefill.error;
            ASSERT_EQ(batched_prefill.requests.size(), 2u);
            ASSERT_EQ(batched_prefill.requests[0].tokens.size(), 1u);
            ASSERT_EQ(batched_prefill.requests[1].tokens.size(), 1u);

            /*
             * The first call consumes terminal prefill logits only. The second
             * call is the proof that matters for resident MTP: it launches the
             * grouped sidecar/verifier, reduces child-local compact outcomes,
             * broadcasts one common NCCL/RCCL decision, and publishes every
             * child's KV/recurrent/logical state before returning response tokens.
             */
            ASSERT_TRUE(runner->supportsDecodeStepBatch(/*request_batch=*/2));
            const GenerationBatchResult batched_mtp =
                runner->decodeStepBatch(/*request_batch=*/2);
            ASSERT_TRUE(batched_mtp.error.empty()) << batched_mtp.error;
            ASSERT_EQ(batched_mtp.requests.size(), 2u);
            ASSERT_FALSE(batched_mtp.requests[0].tokens.empty());
            ASSERT_FALSE(batched_mtp.requests[1].tokens.empty());
            const auto grouped_condition_snapshots =
                capture_stage_diagnostics
                    ? captureRequestBatchSnapshots(
                          *runner,
                          diagnostic_keys,
                          "MTP_REQUEST_BATCH_CONDITION")
                    : std::map<std::string, RequestBatchStageSnapshot>{};

            std::vector<int32_t> batched_long_tokens =
                batched_prefill.requests[0].tokens;
            batched_long_tokens.insert(
                batched_long_tokens.end(),
                batched_mtp.requests[0].tokens.begin(),
                batched_mtp.requests[0].tokens.end());
            std::vector<int32_t> batched_short_tokens =
                batched_prefill.requests[1].tokens;
            batched_short_tokens.insert(
                batched_short_tokens.end(),
                batched_mtp.requests[1].tokens.begin(),
                batched_mtp.requests[1].tokens.end());

            if (capture_stage_diagnostics)
            {
                EXPECT_TRUE(requestBatchTerminalRowsByteIdentical(
                    scalar_short_snapshots,
                    batch_snapshots,
                    stage_comparison_keys,
                    /*scalar_total_rows=*/short_prompt.size(),
                    /*batch_total_rows=*/long_prompt.size() * 2,
                    /*scalar_terminal_row=*/short_prompt.size() - 1,
                    /*batch_terminal_row=*/long_prompt.size() + short_prompt.size() - 1,
                    /*request_index=*/1,
                    /*request_count=*/2,
                    backend_name + " unequal-length request-batch prefill"));

                EXPECT_TRUE(requestBatchTerminalRowsByteIdentical(
                    scalar_long_condition_snapshots,
                    grouped_condition_snapshots,
                    stage_comparison_keys,
                    /*scalar_total_rows=*/1,
                    /*batch_total_rows=*/2,
                    /*scalar_terminal_row=*/0,
                    /*batch_terminal_row=*/0,
                    /*request_index=*/0,
                    /*request_count=*/2,
                    backend_name +
                        " long-request grouped condition versus serial condition"));
                EXPECT_TRUE(requestBatchTerminalRowsByteIdentical(
                    scalar_short_condition_snapshots,
                    grouped_condition_snapshots,
                    stage_comparison_keys,
                    /*scalar_total_rows=*/1,
                    /*batch_total_rows=*/2,
                    /*scalar_terminal_row=*/0,
                    /*batch_terminal_row=*/1,
                    /*request_index=*/1,
                    /*request_count=*/2,
                    backend_name +
                        " short-request grouped condition versus serial condition"));

                for (const std::string &kv_key : {
                         std::string("layer3_ATTENTION_EFFECTIVE_K"),
                         std::string("layer3_ATTENTION_EFFECTIVE_V")})
                {
                    EXPECT_TRUE(requestBatchEffectiveKVBankByteIdentical(
                        scalar_long_condition_snapshots,
                        grouped_condition_snapshots,
                        kv_key,
                        /*scalar_kv_rows=*/long_prompt.size() + 1,
                        /*grouped_kv_stride=*/long_prompt.size() + 1,
                        /*request_index=*/0,
                        backend_name + " long-request grouped condition"));
                    EXPECT_TRUE(requestBatchEffectiveKVBankByteIdentical(
                        scalar_short_condition_snapshots,
                        grouped_condition_snapshots,
                        kv_key,
                        /*scalar_kv_rows=*/short_prompt.size() + 1,
                        /*grouped_kv_stride=*/long_prompt.size() + 1,
                        /*request_index=*/1,
                        backend_name + " short-request grouped condition"));
                }
                runner->disableSnapshotCapture();
            }

            EXPECT_EQ(batched_long_tokens, scalar_long_tokens)
                << backend_name << " " << sampling_mode
                << " request row zero must remain batch-invariant through its "
                   "first grouped MTP verifier transaction";
            EXPECT_EQ(batched_short_tokens, scalar_short_tokens)
                << backend_name << " " << sampling_mode
                << " request row one must use its own resident logical position "
                   "through its first grouped MTP verifier transaction";
        };

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        run_batch_invariance_case(greedy, "greedy");
        run_batch_invariance_case(stochastic, "stochastic");

        const auto records = PerfStatsCollector::snapshot({"mtp", "forward_graph"});
        auto mtp_decode_counter = [&](const std::string &name)
        {
            return mtpCounterValue(records, name);
        };
        EXPECT_EQ(
            perfCounterValue(
                records,
                "mtp",
                "request_batch_prefill_device_position_admissions",
                "prefill"),
            2.0 * participant_count)
            << backend_name << " must admit one device position per request";
        EXPECT_EQ(
            mtp_decode_counter("request_batch_prefill_resident_position_threshold_rows"),
            2.0 * participant_count)
            << backend_name << " must derive both first-token draws on device";
        EXPECT_EQ(
            mtp_decode_counter("request_batch_prefill_device_logical_state_publications"),
            participant_count)
            << backend_name << " must publish one complete request-batch mailbox";
        EXPECT_EQ(
            mtp_decode_counter(
                "request_batch_prefill_device_mtp_transaction_publications"),
            participant_count)
            << backend_name
            << " must establish one canonical request-batch shifted-KV transaction per participant";
        EXPECT_EQ(
            mtp_decode_counter("request_batched_verifier_transactions"),
            1.0)
            << backend_name << " must execute one grouped request-batch verifier transaction";
        EXPECT_EQ(
            mtp_decode_counter("request_batch_device_resident_state_publications"),
            1.0)
            << backend_name << " must publish accepted request state from compact device outcomes";
        EXPECT_EQ(
            mtp_decode_counter("request_batch_resident_plan_checks"),
            1.0)
            << backend_name << " must validate the resident logical-state mailbox after publication";
        if (!local_tp_devices.empty())
        {
            EXPECT_EQ(
                mtp_decode_counter(
                    "rank_mirrored_localtp_request_batch_prefill_samples"),
                2.0)
                << backend_name
                << " LocalTP must sample the batch through child-resident mirrored heads";
            EXPECT_EQ(
                mtp_decode_counter(
                    "rank_mirrored_localtp_resident_request_batch_sidecars"),
                1.0)
                << backend_name
                << " LocalTP must execute one true grouped child sidecar transaction";
            EXPECT_EQ(
                mtp_decode_counter(
                    "rank_mirrored_localtp_resident_request_batch_draft_slot_publications"),
                2.0)
                << backend_name
                << " LocalTP must collectively publish one resident draft slot per request";
            EXPECT_EQ(
                mtp_decode_counter(
                    "rank_mirrored_localtp_stochastic_resident_outcomes"),
                2.0)
                << backend_name
                << " LocalTP must reduce one child-resident stochastic outcome per request";
            EXPECT_EQ(
                mtp_decode_counter(
                    "rank_mirrored_localtp_common_outcome_broadcasts"),
                1.0)
                << backend_name
                << " LocalTP must publish one common compact outcome through NCCL/RCCL";
            EXPECT_EQ(
                mtp_decode_counter(
                    "rank_mirrored_localtp_stochastic_device_outcome_publications"),
                1.0)
                << backend_name
                << " LocalTP must publish the common outcome on every child device";
        }
        EXPECT_GE(mtp_decode_counter("device_resident_logical_state_mailboxes"), 1.0);
        EXPECT_GE(
            mtp_decode_counter("device_resident_logical_state_arena_publications"),
            participant_count)
            << backend_name
            << " must publish request-lifetime logical state from arena-owned rows";
        EXPECT_EQ(mtp_decode_counter("first_token_stochastic_samples"), 0.0)
            << backend_name << " must not route batched GPU prefill through host logits";
        EXPECT_GT(
            perfCounterValue(
                records,
                "forward_graph",
                "request_batched_terminal_state_direct_commits",
                "prefill"),
            0.0)
            << backend_name
            << " long padded prefill must commit GDN/short-conv request banks inside grouped kernels";

        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
        runner->shutdown();
        PerfStatsCollector::reset();
    }

    /**
     * @brief Prove the first stochastic token is stable across request resets.
     *
     * This isolates the ready-prefill-logits path used for token zero.  A
     * failure here means the first sampled distribution changed before later
     * MTP verifier/publication work can affect the request.
     */
    void runQwen36MTPGpuGraphsStochasticFirstTokenRepeatability(
        GlobalDeviceAddress device,
        const std::string &backend_name)
    {
        ScopedDebugEnv deterministic_env({
            {"LLAMINAR_DETERMINISTIC", "1"},
        });
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_qwen36_stochastic_first_token_stats.json"},
            {"LLAMINAR_PERF_STATS_FILTER", "mtp,forward_graph"},
        });
        PerfStatsCollector::reset();

        const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
        if (!env_model)
            env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
        const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
        }

        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 896;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.device_for_this_rank = device;
        config.kv_cache_precision = "auto";
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 1;
        config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();

        auto tokenizer = runner->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        const std::vector<int32_t> prompt =
            buildDeterministicPromptTokens(*tokenizer, /*prompt_token_count=*/768);
        ASSERT_FALSE(prompt.empty());

        SamplingParams stochastic;
        stochastic.temperature = 0.6f;
        stochastic.top_k = 20;
        stochastic.top_p = 0.95f;
        stochastic.presence_penalty = 0.25f;
        stochastic.seed = 123;

        runner->setSamplingParams(stochastic);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        auto run_cycle = [&](int cycle) -> std::vector<int32_t>
        {
            runner->clearCache();
            if (!runner->prefill(prompt))
            {
                ADD_FAILURE() << backend_name << " cycle " << cycle
                              << " prefill failed: " << runner->lastError();
                return {};
            }
            runner->setDecodeStepTokenBudget(1);
            auto step = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);
            if (!step.error.empty())
            {
                ADD_FAILURE() << backend_name << " cycle " << cycle
                              << " decode failed: " << step.error;
                return {};
            }
            return step.tokens;
        };

        const std::vector<int32_t> warmup = run_cycle(-1);
        ASSERT_EQ(warmup.size(), 1u);
        const std::vector<int32_t> expected = run_cycle(0);
        ASSERT_EQ(expected.size(), 1u);

        for (int cycle = 1; cycle <= 4; ++cycle)
        {
            const std::vector<int32_t> actual = run_cycle(cycle);
            ASSERT_EQ(actual.size(), 1u);
            EXPECT_EQ(actual, expected)
                << backend_name
                << " first stochastic token changed after clearCache/prefill at cycle "
                << cycle << "; warmup=" << formatTokenWindow(warmup, 0)
                << " expected=" << formatTokenWindow(expected, 0)
                << " actual=" << formatTokenWindow(actual, 0);
        }

        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
        runner->shutdown();
        PerfStatsCollector::reset();
    }

    int mpiWorldSize()
    {
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        return world_size;
    }

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
        std::string kv_cache_precision = "auto";
        int decode_steps = 3;
        int max_seq_len = 96;
        int main_layers = 0;
        int mpi_ranks = 1;
        int required_rocm_devices = 0;
    };

    std::optional<std::string> densePrefixParitySkipReason(
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

        if (test_case.required_rocm_devices > 0)
        {
            auto &dm = DeviceManager::instance();
            dm.initialize(-1, false);
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

    std::vector<PPStageDefinition> splitStages(
        int total_layers,
        const std::vector<GlobalDeviceAddress> &devices)
    {
        std::vector<PPStageDefinition> stages;
        const int stage_count = static_cast<int>(devices.size());
        if (stage_count <= 0 || total_layers <= 0)
            return stages;

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

    OrchestrationConfig makeDensePrefixRestoreConfig(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        bool enable_prefix_cache,
        int block_size,
        bool enable_mtp = false)
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
        config.mtp.draft_tokens = 1;

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
        {
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
        }

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
                config.device_map_numa_explicit.emplace_back(rank, test_case.devices[rank].hasValidNuma());
            }
            break;
        }

        return config;
    }

    void runDensePrefixRestoreParity(
        const DensePrefixRestoreParityCase &test_case,
        PrefixRestoreParityMode mode)
    {
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

        const std::filesystem::path metadata_path = firstEnvOrDefault(
            test_case.metadata_envs,
            test_case.default_metadata_path);
        if (!std::filesystem::exists(metadata_path))
        {
            GTEST_SKIP() << test_case.name
                         << " PyTorch metadata not found: " << metadata_path;
        }

        const auto prompt_tokens = readTokenListFromMetadata(metadata_path, "token_ids");
        const auto pytorch_decode_tokens = readTokenListFromMetadata(metadata_path, "decode_tokens");
        ASSERT_FALSE(prompt_tokens.empty());
        ASSERT_GE(pytorch_decode_tokens.size(), static_cast<size_t>(test_case.decode_steps));

        const std::vector<int32_t> expected_tokens(
            pytorch_decode_tokens.begin(),
            pytorch_decode_tokens.begin() + test_case.decode_steps);

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

    void runDenseSplitPrefillParity(
        const DensePrefixRestoreParityCase &test_case,
        int split_tokens)
    {
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

        const std::filesystem::path metadata_path = firstEnvOrDefault(
            test_case.metadata_envs,
            test_case.default_metadata_path);
        if (!std::filesystem::exists(metadata_path))
        {
            GTEST_SKIP() << test_case.name
                         << " PyTorch metadata not found: " << metadata_path;
        }

        const auto prompt_tokens = readTokenListFromMetadata(metadata_path, "token_ids");
        const auto pytorch_decode_tokens = readTokenListFromMetadata(metadata_path, "decode_tokens");
        ASSERT_GT(prompt_tokens.size(), static_cast<size_t>(split_tokens));
        ASSERT_GE(pytorch_decode_tokens.size(), static_cast<size_t>(test_case.decode_steps));

        const std::vector<int32_t> expected_tokens(
            pytorch_decode_tokens.begin(),
            pytorch_decode_tokens.begin() + test_case.decode_steps);

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

    DensePrefixRestoreParityCase qwen35CpuPrefixParityCase()
    {
        return DensePrefixRestoreParityCase{
            .name = "Qwen3.5 CPU prefix restore parity",
            .topology = DensePrefixParityTopology::SingleDevice,
            .devices = {GlobalDeviceAddress::cpu()},
            .model_envs = {"LLAMINAR_PREFIX_MTP_PARITY_MODEL"},
            .default_model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
            .metadata_envs = {"LLAMINAR_PREFIX_MTP_PARITY_METADATA"},
            .default_metadata_path = "pytorch_qwen35_snapshots/metadata.txt",
            .kv_cache_precision = "fp32",
            .decode_steps = 3,
            .max_seq_len = 64,
            .main_layers = 24,
        };
    }

    bool isMTPInventoryKey(const std::string &name)
    {
        const std::string lower = lowercase(name);
        return lower.find("mtp") != std::string::npos ||
               lower.find("nextn") != std::string::npos;
    }

    std::filesystem::path tempPrefixDiskDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               ("llaminar_prefix_cache_integration_" + std::to_string(stamp));
    }

    struct TinyQwenForwardFixture
    {
        struct LayerTensors
        {
            std::unique_ptr<FP32Tensor> attn_norm;
            std::unique_ptr<FP32Tensor> wq;
            std::unique_ptr<FP32Tensor> wk;
            std::unique_ptr<FP32Tensor> wv;
            std::unique_ptr<FP32Tensor> wo;
            std::unique_ptr<FP32Tensor> ffn_norm;
            std::unique_ptr<FP32Tensor> gate_proj;
            std::unique_ptr<FP32Tensor> up_proj;
            std::unique_ptr<FP32Tensor> down_proj;
        };

        GraphConfig config;
        std::shared_ptr<MPIContext> mpi = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        std::unique_ptr<FP32Tensor> embedding_table;
        std::unique_ptr<FP32Tensor> final_norm;
        std::unique_ptr<FP32Tensor> lm_head;
        std::vector<LayerTensors> layers;

        explicit TinyQwenForwardFixture(DeviceId device)
        {
            config.n_layers = 1;
            config.total_n_layers = 1;
            config.d_model = 64;
            config.n_heads = 4;
            config.n_kv_heads = 2;
            config.head_dim = 16;
            config.d_ff = 128;
            config.vocab_size = 128;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.default_device = device;
            config.max_seq_len = 8;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = KVCachePrecision::FP16;
            config.use_graph_buffer_management = true;
            config.mtp.enabled = true;
            config.mtp.draft_tokens = 1;

            const size_t d = static_cast<size_t>(config.d_model);
            const size_t q_dim = static_cast<size_t>(config.n_heads * config.head_dim);
            const size_t kv_dim = static_cast<size_t>(config.n_kv_heads * config.head_dim);
            const size_t ff = static_cast<size_t>(config.d_ff);
            const size_t vocab = static_cast<size_t>(config.vocab_size);

            embedding_table = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 201);
            final_norm = TestTensorFactory::createFP32Ones({d});
            lm_head = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 202);

            layers.resize(static_cast<size_t>(config.n_layers));
            for (int i = 0; i < config.n_layers; ++i)
            {
                auto &layer = layers[static_cast<size_t>(i)];
                layer.attn_norm = TestTensorFactory::createFP32Ones({d});
                layer.wq = TestTensorFactory::createFP32Random({q_dim, d}, -0.02f, 0.02f, 210 + i);
                layer.wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 220 + i);
                layer.wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 230 + i);
                layer.wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 240 + i);
                layer.ffn_norm = TestTensorFactory::createFP32Ones({d});
                layer.gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 250 + i);
                layer.up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 260 + i);
                layer.down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 270 + i);
            }
        }

        ModelWeights modelWeights()
        {
            ModelWeights weights;
            weights.embedding_table = embedding_table.get();
            weights.final_norm = final_norm.get();
            weights.lm_head = lm_head.get();
            weights.get_layer_weights = [this](int layer_idx)
            {
                const auto &src = layers.at(static_cast<size_t>(layer_idx));
                LayerWeights layer;
                layer.attn_norm = src.attn_norm.get();
                layer.wq = src.wq.get();
                layer.wk = src.wk.get();
                layer.wv = src.wv.get();
                layer.wo = src.wo.get();
                layer.ffn_norm = src.ffn_norm.get();
                return layer;
            };
            return weights;
        }
    };

    struct TinyMTPSidecarFixture
    {
        GraphConfig config;
        std::shared_ptr<MPIContext> mpi = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        std::unique_ptr<FP32Tensor> embedding_table;
        std::unique_ptr<FP32Tensor> lm_head;
        std::unique_ptr<FP32Tensor> fc;
        std::unique_ptr<FP32Tensor> pre_hidden_norm;
        std::unique_ptr<FP32Tensor> pre_embedding_norm;
        std::unique_ptr<FP32Tensor> final_norm;
        std::unique_ptr<FP32Tensor> attn_norm;
        std::unique_ptr<FP32Tensor> wq;
        std::unique_ptr<FP32Tensor> wk;
        std::unique_ptr<FP32Tensor> wv;
        std::unique_ptr<FP32Tensor> wo;
        std::unique_ptr<FP32Tensor> q_norm;
        std::unique_ptr<FP32Tensor> k_norm;
        std::unique_ptr<FP32Tensor> ffn_norm;
        std::unique_ptr<FP32Tensor> gate_proj;
        std::unique_ptr<FP32Tensor> up_proj;
        std::unique_ptr<FP32Tensor> down_proj;

        std::unique_ptr<FP32Tensor> terminal_hidden;
        std::unique_ptr<FP32Tensor> embedding;
        std::unique_ptr<FP32Tensor> norm_hidden;
        std::unique_ptr<FP32Tensor> norm_embedding;
        std::unique_ptr<FP32Tensor> concat;
        std::unique_ptr<FP32Tensor> projected;
        std::unique_ptr<FP32Tensor> hidden;
        std::unique_ptr<FP32Tensor> q;
        std::unique_ptr<FP32Tensor> k;
        std::unique_ptr<FP32Tensor> v;
        std::unique_ptr<FP32Tensor> q_raw;
        std::unique_ptr<FP32Tensor> q_gate;
        std::unique_ptr<FP32Tensor> attn_output;
        std::unique_ptr<FP32Tensor> attn_proj;
        std::unique_ptr<FP32Tensor> gate;
        std::unique_ptr<FP32Tensor> up;
        std::unique_ptr<FP32Tensor> ffn_output;
        std::unique_ptr<FP32Tensor> logits;

        std::vector<std::unique_ptr<WeightBinding>> binding_storage;
        ModelWeightBindings bindings;
        MTPDepthWeightBindings mtp_depth_bindings;
        uint64_t next_binding_id = 1;

        std::unique_ptr<IKVCache> kv_cache;
        int draft_token = 17;
        int position_id = 0;
        std::vector<int> sequence_lengths{0};
        DeviceId device;

        explicit TinyMTPSidecarFixture(DeviceId target_device)
            : device(target_device)
        {
            config.n_layers = 2;
            config.total_n_layers = 2;
            config.d_model = 64;
            config.n_heads = 4;
            config.n_kv_heads = 2;
            config.head_dim = 16;
            config.d_ff = 128;
            config.vocab_size = 128;
            config.rms_norm_eps = 1e-6f;
            config.rope_theta = 10000.0f;
            config.partial_rotary_factor = 1.0f;
            config.default_device = device;
            config.max_seq_len = 8;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = KVCachePrecision::FP16;
            config.layer_types = {"full_attention", "full_attention"};

            const size_t d = static_cast<size_t>(config.d_model);
            const size_t q_dim = static_cast<size_t>(config.n_heads * config.head_dim);
            const size_t kv_dim = static_cast<size_t>(config.n_kv_heads * config.head_dim);
            const size_t ff = static_cast<size_t>(config.d_ff);
            const size_t vocab = static_cast<size_t>(config.vocab_size);

            embedding_table = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 301);
            lm_head = TestTensorFactory::createFP32Random({vocab, d}, -0.02f, 0.02f, 302);
            fc = TestTensorFactory::createFP32Random({d, d * 2}, -0.02f, 0.02f, 303);
            pre_hidden_norm = TestTensorFactory::createFP32Ones({d});
            pre_embedding_norm = TestTensorFactory::createFP32Ones({d});
            final_norm = TestTensorFactory::createFP32Ones({d});
            attn_norm = TestTensorFactory::createFP32Ones({d});
            wq = TestTensorFactory::createFP32Random({q_dim * 2, d}, -0.02f, 0.02f, 304);
            wk = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 305);
            wv = TestTensorFactory::createFP32Random({kv_dim, d}, -0.02f, 0.02f, 306);
            wo = TestTensorFactory::createFP32Random({d, q_dim}, -0.02f, 0.02f, 307);
            q_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            k_norm = TestTensorFactory::createFP32Ones({static_cast<size_t>(config.head_dim)});
            ffn_norm = TestTensorFactory::createFP32Ones({d});
            gate_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 308);
            up_proj = TestTensorFactory::createFP32Random({ff, d}, -0.02f, 0.02f, 309);
            down_proj = TestTensorFactory::createFP32Random({d, ff}, -0.02f, 0.02f, 310);

            terminal_hidden = TestTensorFactory::createFP32Random({1, d}, -0.02f, 0.02f, 311);
            embedding = TestTensorFactory::createFP32({1, d});
            norm_hidden = TestTensorFactory::createFP32({1, d});
            norm_embedding = TestTensorFactory::createFP32({1, d});
            concat = TestTensorFactory::createFP32({1, d * 2});
            projected = TestTensorFactory::createFP32({1, d});
            hidden = TestTensorFactory::createFP32({1, d});
            q = TestTensorFactory::createFP32({1, q_dim});
            k = TestTensorFactory::createFP32({1, kv_dim});
            v = TestTensorFactory::createFP32({1, kv_dim});
            q_raw = TestTensorFactory::createFP32({1, q_dim * 2});
            q_gate = TestTensorFactory::createFP32({1, q_dim});
            attn_output = TestTensorFactory::createFP32({1, q_dim});
            attn_proj = TestTensorFactory::createFP32({1, d});
            gate = TestTensorFactory::createFP32({1, ff});
            up = TestTensorFactory::createFP32({1, ff});
            ffn_output = TestTensorFactory::createFP32({1, d});
            logits = TestTensorFactory::createFP32({1, vocab});

            llaminar::v2::kernels::KVCacheConfig kv_config;
            kv_config.precision = ActivationPrecision::FP32;
            kv_config.device = device;
            kv_config.num_layers = 1;
            kv_config.batch_size = 1;
            kv_config.max_seq_len = 8;
            kv_config.n_kv_heads = config.n_kv_heads;
            kv_config.head_dim = config.head_dim;
            kv_config.mpi_ctx = mpi.get();
            kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);

            buildBindings();
        }

        const WeightBinding *addBinding(TensorBase *tensor,
                                        std::string canonical_name,
                                        WeightRole role,
                                        int layer = -1)
        {
            auto binding = std::make_unique<WeightBinding>();
            binding->binding_id = next_binding_id++;
            binding->identity.canonical_name = std::move(canonical_name);
            binding->identity.logical_id = binding->binding_id;
            binding->identity.role = role;
            binding->identity.layer = layer;
            binding->residency.home_device = device;
            binding->residency.resident_device = device;
            binding->tensor = tensor;
            binding->immutable = true;
            const WeightBinding *ptr = binding.get();
            binding_storage.push_back(std::move(binding));
            return ptr;
        }

        void buildBindings()
        {
            bindings.embedding_table = addBinding(embedding_table.get(), "token_embd.weight", WeightRole::Embedding);
            bindings.lm_head = addBinding(lm_head.get(), "output.weight", WeightRole::LMHead);
            mtp_depth_bindings.depth_index = 0;
            mtp_depth_bindings.source_layer_index = 64;
            mtp_depth_bindings.nextn_block_layout = true;
            mtp_depth_bindings.fc = addBinding(fc.get(), "blk.64.nextn.eh_proj.weight", WeightRole::Other, 64);
            mtp_depth_bindings.pre_fc_norm_hidden = addBinding(pre_hidden_norm.get(), "blk.64.nextn.hnorm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.pre_fc_norm_embedding = addBinding(pre_embedding_norm.get(), "blk.64.nextn.enorm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.final_norm = addBinding(final_norm.get(), "blk.64.nextn.shared_head_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.attn_norm = addBinding(attn_norm.get(), "blk.64.attn_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.wq = addBinding(wq.get(), "blk.64.attn_q.weight", WeightRole::AttentionQ, 64);
            mtp_depth_bindings.fa_block.wk = addBinding(wk.get(), "blk.64.attn_k.weight", WeightRole::AttentionK, 64);
            mtp_depth_bindings.fa_block.wv = addBinding(wv.get(), "blk.64.attn_v.weight", WeightRole::AttentionV, 64);
            mtp_depth_bindings.fa_block.wo = addBinding(wo.get(), "blk.64.attn_output.weight", WeightRole::AttentionWO, 64);
            mtp_depth_bindings.fa_block.q_norm = addBinding(q_norm.get(), "blk.64.attn_q_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.k_norm = addBinding(k_norm.get(), "blk.64.attn_k_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.ffn_norm = addBinding(ffn_norm.get(), "blk.64.post_attention_norm.weight", WeightRole::Norm, 64);
            mtp_depth_bindings.fa_block.gate_proj = addBinding(gate_proj.get(), "blk.64.ffn_gate.weight", WeightRole::FFNGate, 64);
            mtp_depth_bindings.fa_block.up_proj = addBinding(up_proj.get(), "blk.64.ffn_up.weight", WeightRole::FFNUp, 64);
            mtp_depth_bindings.fa_block.down_proj = addBinding(down_proj.get(), "blk.64.ffn_down.weight", WeightRole::FFNDown, 64);
            bindings.mtp.depth = 1;
            bindings.mtp.depths.push_back(mtp_depth_bindings);
        }

        void prepareWeights(PreparedWeightStore &store)
        {
            for (const auto &owned : binding_storage)
            {
                ASSERT_NE(owned, nullptr);
                ASSERT_NE(owned->tensor, nullptr);
                ASSERT_TRUE(owned->tensor->ensureOnDevice(device));
                if (owned->tensor->shape().size() == 2 &&
                    owned->identity.role != WeightRole::Embedding)
                {
                    store.prepareGemm(*owned);
                }
            }
            ASSERT_TRUE(terminal_hidden->ensureOnDevice(device));

            const std::vector<TensorBase *> activation_outputs = {
                embedding.get(),
                norm_hidden.get(),
                norm_embedding.get(),
                concat.get(),
                projected.get(),
                hidden.get(),
                q.get(),
                k.get(),
                v.get(),
                q_raw.get(),
                q_gate.get(),
                attn_output.get(),
                attn_proj.get(),
                gate.get(),
                up.get(),
                ffn_output.get(),
                logits.get(),
            };
            for (TensorBase *tensor : activation_outputs)
            {
                ASSERT_NE(tensor, nullptr);
                ASSERT_TRUE(tensor->allocateOnDevice(device));
            }
        }

        MTPForwardInput input()
        {
            MTPForwardInput in;
            in.draft_token_ids = &draft_token;
            in.terminal_hidden = terminal_hidden.get();
            in.kv_cache = kv_cache.get();
            in.position_ids = &position_id;
            in.sequence_lengths = &sequence_lengths;
            in.batch_size = 1;
            in.seq_len = 1;
            in.device = device;
            return in;
        }

        MTPForwardOutput output()
        {
            MTPForwardOutput out;
            out.logits = logits.get();
            out.hidden = hidden.get();
            out.embedding = embedding.get();
            out.norm_hidden = norm_hidden.get();
            out.norm_embedding = norm_embedding.get();
            out.concat = concat.get();
            out.projected = projected.get();
            out.q = q.get();
            out.k = k.get();
            out.v = v.get();
            out.q_raw = q_raw.get();
            out.q_gate = q_gate.get();
            out.attn_output = attn_output.get();
            out.attn_proj = attn_proj.get();
            out.gate = gate.get();
            out.up = up.get();
            out.ffn_output = ffn_output.get();
            return out;
        }
    };
} // namespace

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_ResetStateInventory)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = firstGpuDeviceSpec();
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for prefix-cache state probe";
    }

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(makeSingleGpuConfig(*device_spec));
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    runner->setSamplingParams(greedy);

    const auto initial = runner->prefixStateProbe();
    EXPECT_TRUE(initial.initialized);
    EXPECT_FALSE(initial.prefill_logits_ready);
    EXPECT_EQ(initial.current_position, 0);
    EXPECT_EQ(maxLayerCachedTokens(initial), 0);

    const std::vector<int32_t> prefix_tokens = {1, 2, 3, 4};
    ASSERT_TRUE(runner->prefill(prefix_tokens)) << runner->lastError();

    const auto after_prefill = runner->prefixStateProbe();
    EXPECT_TRUE(after_prefill.prefill_logits_ready);
    EXPECT_EQ(after_prefill.current_position, static_cast<int>(prefix_tokens.size()));
    ASSERT_FALSE(after_prefill.kv_caches.empty());
    EXPECT_EQ(maxLayerCachedTokens(after_prefill), static_cast<int>(prefix_tokens.size()));
    ASSERT_FALSE(after_prefill.positions.empty());
    EXPECT_EQ(after_prefill.positions[0], static_cast<int>(prefix_tokens.size()));
    ASSERT_FALSE(after_prefill.sequence_lengths.empty());
    EXPECT_EQ(after_prefill.sequence_lengths[0], static_cast<int>(prefix_tokens.size()));

    auto first_decode = runner->decodeStep();
    ASSERT_TRUE(first_decode.error.empty()) << first_decode.error;

    const auto after_first_decode = runner->prefixStateProbe();
    EXPECT_FALSE(after_first_decode.prefill_logits_ready);
    EXPECT_EQ(after_first_decode.current_position, static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxLayerCachedTokens(after_first_decode), static_cast<int>(prefix_tokens.size()));

    auto second_decode = runner->decodeStep();
    ASSERT_TRUE(second_decode.error.empty()) << second_decode.error;

    const auto after_second_decode = runner->prefixStateProbe();
    EXPECT_EQ(after_second_decode.current_position, static_cast<int>(prefix_tokens.size()) + 1);
    EXPECT_EQ(maxLayerCachedTokens(after_second_decode), static_cast<int>(prefix_tokens.size()) + 1);

    runner->clearCache();
    const auto after_clear = runner->prefixStateProbe();
    EXPECT_FALSE(after_clear.prefill_logits_ready);
    EXPECT_EQ(after_clear.current_position, 0);
    EXPECT_EQ(maxLayerCachedTokens(after_clear), 0);
    EXPECT_TRUE(allValuesZero(after_clear.positions));
    EXPECT_TRUE(allValuesZero(after_clear.sequence_lengths));
    EXPECT_GT(after_clear.session_epoch, after_second_decode.session_epoch);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_GPUCacheFlagPreservesGreedyInference)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = firstGpuDeviceSpec();
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for prefix-cache GPU integration probe";
    }

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(makeSingleGpuPrefixCacheConfig(*device_spec));
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    SamplingParams greedy;
    greedy.temperature = 0.0f;

    const std::vector<int32_t> prompt = {1, 2, 3, 4};
    auto first = runner->generate(prompt, 2, greedy);
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 2u);

    const auto after_first = runner->prefixStateProbe();
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 2u)
        << "First GPU prompt should harvest both 2-token dense prefix blocks";

    auto second = runner->generate(prompt, 2, greedy);
    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 2u);
    EXPECT_EQ(second.tokens.front(), first.tokens.front())
        << "Full-hit terminal logits should preserve the first greedy token";

    const auto after_repeated_prompt = runner->prefixStateProbe();
    EXPECT_TRUE(after_repeated_prompt.initialized);
    EXPECT_TRUE(after_repeated_prompt.prefix_cache_ready);
    EXPECT_GE(after_repeated_prompt.prefix_cache_hits, 2u)
        << "Second GPU prompt should reuse both cached 2-token dense prefix blocks";
    EXPECT_GE(after_repeated_prompt.current_position, static_cast<int>(prompt.size()) + 1)
        << "Second decode step should advance over the restored/imported prefix KV state";
    EXPECT_GT(maxLayerCachedTokens(after_repeated_prompt), 0);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_CUDADeviceHotTierRestoresEvictedBlocksDirectly)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = explicitGpuDeviceSpec(DeviceType::CUDA);
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA GPU available for device-hot prefix-cache probe";
    }
    verifyGpuDeviceHotTierRestoresDirectly(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_ROCmDeviceHotTierRestoresEvictedBlocksDirectly)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::ROCm);
    if (!device_spec)
    {
        GTEST_SKIP() << "No ROCm GPU available for device-hot prefix-cache probe";
    }
    verifyGpuDeviceHotTierRestoresDirectly(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_CUDATieredDiskCacheHydratesEvictedBlocks)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    const auto device_spec = explicitGpuDeviceSpec(DeviceType::CUDA);
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA GPU available for tiered prefix-cache probe";
    }
    verifyGpuTieredDiskCacheHydrates(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_ROCmTieredDiskCacheHydratesEvictedBlocks)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::ROCm);
    if (!device_spec)
    {
        GTEST_SKIP() << "No ROCm GPU available for tiered prefix-cache probe";
    }
    verifyGpuTieredDiskCacheHydrates(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_CUDADiskArchiveSurvivesRunnerRestart)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::CUDA);
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA GPU available for disk restart probe";
    }
    verifyGpuDiskArchiveSurvivesRunnerRestart(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_ROCmDiskArchiveSurvivesRunnerRestart)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::ROCm);
    if (!device_spec)
    {
        GTEST_SKIP() << "No ROCm GPU available for disk restart probe";
    }
    verifyGpuDiskArchiveSurvivesRunnerRestart(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_CUDADiskHitRepromotesToDeviceHot)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::CUDA);
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA GPU available for disk-to-hot promotion probe";
    }
    verifyGpuDiskHitRepromotesToDeviceHot(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_ROCmDiskHitRepromotesToDeviceHot)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::ROCm);
    if (!device_spec)
    {
        GTEST_SKIP() << "No ROCm GPU available for disk-to-hot promotion probe";
    }
    verifyGpuDiskHitRepromotesToDeviceHot(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_CUDAHotTierPressureCycles)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::CUDA);
    if (!device_spec)
    {
        GTEST_SKIP() << "No CUDA GPU available for hot-tier pressure probe";
    }
    verifyGpuHotTierPressureCycles(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_ROCmHotTierPressureCycles)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }
    const auto device_spec = explicitGpuDeviceSpec(DeviceType::ROCm);
    if (!device_spec)
    {
        GTEST_SKIP() << "No ROCm GPU available for hot-tier pressure probe";
    }
    verifyGpuHotTierPressureCycles(*device_spec);
}

TEST(Test__KVPrefixMTPStateProbe, DenseQwen25_CPUPrefixCacheFullHitRecordsReuse)
{
    if (!std::filesystem::exists(kDenseModelPath))
    {
        GTEST_SKIP() << "Dense probe model not found: " << kDenseModelPath;
    }

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::vector<int32_t> prompt = {1, 2, 3, 4};

    auto factory = createOrchestrationRunnerFactory();
    auto baseline = factory->createFromOrchestrationConfig(makeSingleCpuConfig(false));
    ASSERT_NE(baseline, nullptr);
    ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
    auto baseline_result = baseline->generate(prompt, 1, greedy);
    ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
    ASSERT_EQ(baseline_result.tokens.size(), 1u);

    auto cached = factory->createFromOrchestrationConfig(makeSingleCpuConfig(true));
    ASSERT_NE(cached, nullptr);
    ASSERT_TRUE(cached->initialize()) << cached->lastError();

    auto first = cached->generate(prompt, 1, greedy);
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 1u);
    EXPECT_EQ(first.tokens[0], baseline_result.tokens[0]);

    const auto after_first = cached->prefixStateProbe();
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 2u);

    auto second = cached->generate(prompt, 1, greedy);
    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 1u);
    EXPECT_EQ(second.tokens[0], baseline_result.tokens[0]);

    const auto after_second = cached->prefixStateProbe();
    EXPECT_TRUE(after_second.prefix_cache_ready);
    EXPECT_GE(after_second.prefix_cache_hits, 2u)
        << "Second full prompt should reuse both cached 2-token dense prefix blocks";
    EXPECT_EQ(after_second.current_position, static_cast<int>(prompt.size()));
}

TEST(Test__KVPrefixMTPStateProbe, MTP_ModelInventoryWhenAvailable)
{
    const std::vector<std::string> model_paths = {
        "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
        "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
    };

    bool saw_model = false;
    for (const auto &path : model_paths)
    {
        if (!std::filesystem::exists(path))
        {
            continue;
        }
        saw_model = true;

        ModelLoader loader;
        loader.setUseMmap(false);
        ASSERT_TRUE(loader.loadModel(path)) << "failed to read GGUF metadata: " << path;

        size_t mtp_metadata = 0;
        uint64_t nextn_predict_layers = 0;
        for (const auto &[key, value] : loader.getModel().metadata)
        {
            if (isMTPInventoryKey(key))
            {
                ++mtp_metadata;
            }

            const std::string lower_key = lowercase(key);
            if (lower_key.find("nextn_predict_layers") != std::string::npos ||
                lower_key.find("mtp_num_hidden_layers") != std::string::npos ||
                lower_key.find("mtp.num_hidden_layers") != std::string::npos)
            {
                nextn_predict_layers = value.asUInt64();
            }
        }

        size_t mtp_tensors = 0;
        for (const auto &name : loader.tensorNames())
        {
            if (isMTPInventoryKey(name))
            {
                ++mtp_tensors;
            }
        }

        EXPECT_GT(mtp_metadata, 0u)
            << "expected MTP/nextn metadata in " << path;
        EXPECT_EQ(nextn_predict_layers, 1u)
            << "expected one MTP/nextn prediction layer in " << path;
        EXPECT_GT(mtp_tensors, 0u)
            << "expected MTP/nextn tensors in " << path;

        auto manifest = discoverMTPWeightManifest(
            loader,
            loader.architecture(),
            static_cast<int>(loader.blockCount()),
            /*explicit_mtp=*/true);
        ASSERT_TRUE(manifest.available) << manifest.diagnostic << " in " << path;
        ASSERT_EQ(manifest.depth, 1);
        ASSERT_EQ(manifest.depths.size(), 1u);
        EXPECT_TRUE(manifest.depths[0].nextn_block_layout);
        EXPECT_EQ(
            manifest.depths[0].source_layer_index,
            static_cast<int>(loader.blockCount() - nextn_predict_layers));
        if (lowercase(loader.architecture()).find("moe") != std::string::npos)
        {
            EXPECT_TRUE(manifest.depths[0].moe_ffn_layout);
        }
    }

    if (!saw_model)
    {
        GTEST_SKIP() << "Qwen3.6 MTP probe models are not available";
    }
}

TEST(Test__KVPrefixMTPStateProbe, Qwen35CPUPrefixCacheMatchesPyTorchDecodeTokens)
{
    runDensePrefixRestoreParity(
        qwen35CpuPrefixParityCase(),
        PrefixRestoreParityMode::FullHit);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen35CPUPartialPrefixCacheMatchesPyTorchDecodeTokens)
{
    runDensePrefixRestoreParity(
        qwen35CpuPrefixParityCase(),
        PrefixRestoreParityMode::PartialHit);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen35CPUSplitPrefillMatchesPyTorchDecodeTokens)
{
    runDenseSplitPrefillParity(qwen35CpuPrefixParityCase(), 4);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPRealModelSmoke)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "0"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
    });

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 MTP smoke";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";

    auto factory = createOrchestrationRunnerFactory();
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::string prompt_text = "The quick brown fox jumps over the lazy dog";

    auto make_config = [&](bool enable_mtp)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 32;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.device_for_this_rank = GlobalDeviceAddress::rocm(rocm_ordinal);
        config.kv_cache_precision = "auto";
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = 1;
        return config;
    };

    auto run_once = [&](bool enable_mtp,
                        GenerationResult *result,
                        PrefixRuntimeStateSnapshot *snapshot)
    {
        auto runner = factory->createFromOrchestrationConfig(make_config(enable_mtp));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        auto tokenizer = runner->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        const auto encoded = tokenizer->encode(prompt_text, /*add_bos=*/false, /*add_eos=*/false);
        ASSERT_FALSE(encoded.empty());
        const std::vector<int32_t> prompt(encoded.begin(), encoded.end());
        *result = runner->generate(prompt, 4, greedy);
        *snapshot = runner->prefixStateProbe();
        runner->shutdown();
    };

    GenerationResult baseline_result;
    PrefixRuntimeStateSnapshot baseline_snapshot;
    run_once(false, &baseline_result, &baseline_snapshot);

    GenerationResult mtp_result;
    PrefixRuntimeStateSnapshot mtp_snapshot;
    run_once(true, &mtp_result, &mtp_snapshot);

    ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
    ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
    ASSERT_EQ(baseline_result.tokens.size(), 4u);
    ASSERT_EQ(mtp_result.tokens.size(), 4u);
    EXPECT_EQ(mtp_result.tokens, baseline_result.tokens);
    EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);
    EXPECT_GE(mtp_snapshot.mtp_draft_steps, 2u);
    EXPECT_GE(mtp_snapshot.mtp_verifier_runs, 2u);
    EXPECT_GE(mtp_snapshot.mtp_accepted_tokens + mtp_snapshot.mtp_rejected_tokens, 2u);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPGpuGraphsRealModelSmoke)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 MTP GPU-graphs smoke";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsGreedyRealModelSmoke(
        GlobalDeviceAddress::rocm(rocm_ordinal),
        "ROCm");
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36CUDAMTPGpuGraphsRealModelSmoke)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() <= 0)
    {
        GTEST_SKIP() << "No CUDA device available for Qwen3.6 MTP GPU-graphs smoke";
    }
    const int cuda_ordinal = qwen36CudaSingleDeviceOrdinal();
    ASSERT_GE(cuda_ordinal, 0);
    ASSERT_LT(cuda_ordinal, dm.cuda_device_count())
        << "Selected CUDA device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsGreedyRealModelSmoke(
        GlobalDeviceAddress::cuda(cuda_ordinal),
        "CUDA");
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPGpuGraphsStochasticRealModelSmoke)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 stochastic MTP GPU-graphs smoke";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsStochasticRealModelSmoke(
        GlobalDeviceAddress::rocm(rocm_ordinal),
        "ROCm");
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPRequestBatchResidentPrefill)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 request-batch prefill";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count());
    runQwen36MTPGpuRequestBatchResidentPrefill(
        GlobalDeviceAddress::rocm(rocm_ordinal),
        "ROCm");
}

TEST(Test__KVPrefixMTPStateProbe,
     Qwen36ROCm2LocalTPMTPRequestBatchResidentPrefill)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() < 2)
    {
        GTEST_SKIP() << "Need two ROCm devices for Qwen3.6 LocalTP request-batch prefill";
    }
    runQwen36MTPGpuRequestBatchResidentPrefill(
        GlobalDeviceAddress::rocm(0),
        "ROCm2 LocalTP",
        {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPGpuGraphsStochasticClearCacheRepeatabilityLong)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 stochastic MTP GPU-graphs repeatability";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsStochasticRealModelSmoke(
        GlobalDeviceAddress::rocm(rocm_ordinal),
        "ROCm",
        /*decode_token_count=*/64,
        /*repeat_cycles=*/4,
        /*deterministic_repeatability=*/true,
        /*use_presence_penalty=*/true,
        /*prompt_token_count=*/768);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPGpuGraphsStochasticFirstTokenRepeatability)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 stochastic MTP first-token repeatability";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsStochasticFirstTokenRepeatability(
        GlobalDeviceAddress::rocm(rocm_ordinal),
        "ROCm");
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36CUDAMTPGpuGraphsStochasticRealModelSmoke)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() <= 0)
    {
        GTEST_SKIP() << "No CUDA device available for Qwen3.6 stochastic MTP GPU-graphs smoke";
    }
    const int cuda_ordinal = qwen36CudaSingleDeviceOrdinal();
    ASSERT_GE(cuda_ordinal, 0);
    ASSERT_LT(cuda_ordinal, dm.cuda_device_count())
        << "Selected CUDA device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsStochasticRealModelSmoke(
        GlobalDeviceAddress::cuda(cuda_ordinal),
        "CUDA");
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36CUDAMTPRequestBatchResidentPrefill)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() <= 0)
    {
        GTEST_SKIP() << "No CUDA device available for Qwen3.6 request-batch prefill";
    }
    const int cuda_ordinal = qwen36CudaSingleDeviceOrdinal();
    ASSERT_GE(cuda_ordinal, 0);
    ASSERT_LT(cuda_ordinal, dm.cuda_device_count());
    runQwen36MTPGpuRequestBatchResidentPrefill(
        GlobalDeviceAddress::cuda(cuda_ordinal),
        "CUDA");
}

TEST(Test__KVPrefixMTPStateProbe,
     Qwen36CUDA2LocalTPMTPRequestBatchResidentPrefill)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() < 2)
    {
        GTEST_SKIP() << "Need two CUDA devices for Qwen3.6 LocalTP request-batch prefill";
    }
    runQwen36MTPGpuRequestBatchResidentPrefill(
        GlobalDeviceAddress::cuda(0),
        "CUDA2 LocalTP",
        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36CUDAMTPGpuGraphsStochasticClearCacheRepeatabilityLong)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() <= 0)
    {
        GTEST_SKIP() << "No CUDA device available for Qwen3.6 stochastic MTP GPU-graphs repeatability";
    }
    const int cuda_ordinal = qwen36CudaSingleDeviceOrdinal();
    ASSERT_GE(cuda_ordinal, 0);
    ASSERT_LT(cuda_ordinal, dm.cuda_device_count())
        << "Selected CUDA device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsStochasticRealModelSmoke(
        GlobalDeviceAddress::cuda(cuda_ordinal),
        "CUDA",
        /*decode_token_count=*/64,
        /*repeat_cycles=*/4,
        /*deterministic_repeatability=*/true,
        /*use_presence_penalty=*/true,
        /*prompt_token_count=*/768);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36CUDAMTPGpuGraphsStochasticFirstTokenRepeatability)
{
    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() <= 0)
    {
        GTEST_SKIP() << "No CUDA device available for Qwen3.6 stochastic MTP first-token repeatability";
    }
    const int cuda_ordinal = qwen36CudaSingleDeviceOrdinal();
    ASSERT_GE(cuda_ordinal, 0);
    ASSERT_LT(cuda_ordinal, dm.cuda_device_count())
        << "Selected CUDA device ordinal is outside the available device range";
    runQwen36MTPGpuGraphsStochasticFirstTokenRepeatability(
        GlobalDeviceAddress::cuda(cuda_ordinal),
        "CUDA");
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPGpuGraphsChainedDraftRealModelSmoke)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
        {"LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_qwen36_chained_mtp_forward_graph_stats.json"},
        {"LLAMINAR_PERF_STATS_FILTER", "forward_graph"},
    });
    PerfStatsCollector::reset();

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 chained MTP GPU-graphs smoke";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";

    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.model_path = model_path;
    config.max_seq_len = 32;
    config.batch_size = 1;
    config.tp_degree = 1;
    config.pp_degree = 1;
    config.device_for_this_rank = GlobalDeviceAddress::rocm(rocm_ordinal);
    config.kv_cache_precision = "auto";
    config.mtp.enabled = true;
    config.mtp.draft_tokens = 3;

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(config);
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    auto tokenizer = runner->tokenizer();
    ASSERT_NE(tokenizer, nullptr);
    const auto encoded = tokenizer->encode("Paris is", /*add_bos=*/false, /*add_eos=*/false);
    ASSERT_FALSE(encoded.empty());
    const std::vector<int32_t> prompt(encoded.begin(), encoded.end());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    // Reproduce the verifier graph-cache lifetime pattern from the ROCm MTP
    // benchmark: full-depth draft, smaller tail draft, request reset, then
    // full-depth cache reuse. Phase 9.7 currently keeps direct all-position
    // verifier publication fail-closed, so the reusable graph cache is the
    // decode-equivalent main decode lane rather than `main_verifier`.
    auto warm_result = runner->generate(prompt, 6, greedy);
    ASSERT_TRUE(warm_result.error.empty()) << warm_result.error;
    runner->clearCache();
    auto result = runner->generate(prompt, 6, greedy);
    const auto snapshot = runner->prefixStateProbe();
    runner->shutdown();

    ASSERT_TRUE(result.error.empty()) << result.error;
    ASSERT_FALSE(result.tokens.empty());
    EXPECT_TRUE(snapshot.mtp_config_enabled);
    EXPECT_FALSE(snapshot.mtp_bypassed) << snapshot.mtp_bypass_reason;
    EXPECT_GE(snapshot.mtp_draft_steps, 2u);
    EXPECT_GE(snapshot.mtp_verifier_runs, 1u);
    EXPECT_GE(snapshot.mtp_verifier_token_count, 4u);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    const PerfStatsCollector::Tags miss_tags = {
        {"all_position_logit_rows", "0"},
        {"all_position_logits", "false"},
        {"context", "main_decode"},
        {"decode_has_history", "true"},
        {"moe_placement_epoch", "0"},
        {"result", "miss"},
        {"seq_len", "1"},
        {"uses_device_position_ids", "false"},
        {"uses_device_sequence_lengths", "false"},
        {"uses_device_token_ids", "false"},
    };
    const PerfStatsCollector::Tags hit_tags = {
        {"all_position_logit_rows", "0"},
        {"all_position_logits", "false"},
        {"context", "main_decode"},
        {"decode_has_history", "true"},
        {"moe_placement_epoch", "0"},
        {"result", "hit"},
        {"seq_len", "1"},
        {"uses_device_position_ids", "false"},
        {"uses_device_sequence_lengths", "false"},
        {"uses_device_token_ids", "false"},
    };
    EXPECT_GE(findPerfCounterValue(records, "forward_graph", "forward_cache_lookup", "decode", miss_tags), 1.0);
    EXPECT_GE(findPerfCounterValue(records, "forward_graph", "forward_cache_lookup", "decode", hit_tags), 1.0);
    PerfStatsCollector::reset();
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmMTPGpuGraphsBaselineThenMTPRegression)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
    });

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 MTP GPU-graphs regression";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";

    auto factory = createOrchestrationRunnerFactory();
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::string prompt_text = "The quick brown fox jumps over the lazy dog";

    auto make_config = [&](bool enable_mtp)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 64;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.device_for_this_rank = GlobalDeviceAddress::rocm(rocm_ordinal);
        config.kv_cache_precision = "auto";
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = 1;
        return config;
    };

    auto run_once = [&](bool enable_mtp,
                        GenerationResult *result,
                        PrefixRuntimeStateSnapshot *snapshot)
    {
        auto runner = factory->createFromOrchestrationConfig(make_config(enable_mtp));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        auto tokenizer = runner->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        const auto encoded = tokenizer->encode(prompt_text, /*add_bos=*/false, /*add_eos=*/false);
        ASSERT_FALSE(encoded.empty());
        const std::vector<int32_t> prompt(encoded.begin(), encoded.end());
        *result = runner->generate(prompt, 4, greedy);
        *snapshot = runner->prefixStateProbe();
        runner->shutdown();
    };

    GenerationResult baseline_result;
    PrefixRuntimeStateSnapshot baseline_snapshot;
    run_once(false, &baseline_result, &baseline_snapshot);

    GenerationResult mtp_result;
    PrefixRuntimeStateSnapshot mtp_snapshot;
    run_once(true, &mtp_result, &mtp_snapshot);

    ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
    ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
    ASSERT_EQ(baseline_result.tokens.size(), 4u);
    ASSERT_EQ(mtp_result.tokens.size(), 4u);
    EXPECT_EQ(mtp_result.tokens, baseline_result.tokens);
    EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);
    EXPECT_FALSE(mtp_snapshot.mtp_bypassed) << mtp_snapshot.mtp_bypass_reason;
    EXPECT_GE(mtp_snapshot.mtp_draft_steps, 2u);
    EXPECT_GE(mtp_snapshot.mtp_verifier_runs, 2u);
    EXPECT_GE(mtp_snapshot.mtp_accepted_tokens + mtp_snapshot.mtp_rejected_tokens, 2u);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmPaddedPrefillBucketGraphCaptureRegression)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "600"},
        {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
        {"LLAMINAR_PERF_STATS_JSON", "/tmp/llaminar_qwen36_padded_prefill_bucket_stats.json"},
        {"LLAMINAR_PERF_STATS_FILTER", "forward_graph"},
    });
    PerfStatsCollector::reset();

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 padded prefill bucket regression";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";

    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.model_path = model_path;
    config.max_seq_len = 1024;
    config.batch_size = 1;
    config.tp_degree = 1;
    config.pp_degree = 1;
    config.device_for_this_rank = GlobalDeviceAddress::rocm(rocm_ordinal);
    config.kv_cache_precision = "auto";

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(config);
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    auto tokenizer = runner->tokenizer();
    ASSERT_NE(tokenizer, nullptr);
    const auto encoded = tokenizer->encode(" the", /*add_bos=*/false, /*add_eos=*/false);
    ASSERT_FALSE(encoded.empty());
    const std::vector<int32_t> prompt(595, static_cast<int32_t>(encoded.front()));

    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();
    EXPECT_EQ(runner->currentPosition(), 595);
    runner->clearCache();

    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();
    EXPECT_EQ(runner->currentPosition(), 595);
    runner->clearCache();

    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();
    EXPECT_EQ(runner->currentPosition(), 595);

    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    runner->shutdown();

    auto lifecycle_count = [&](const std::string &capture_phase,
                               const std::string &cache_phase,
                               const std::string &recapture_reason) -> double
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph" ||
                record.name != "prefill_graph_lifecycle" ||
                record.phase != "prefill")
            {
                continue;
            }

            auto tag = [&](const std::string &key) -> std::string
            {
                const auto it = record.tags.find(key);
                return it == record.tags.end() ? std::string() : it->second;
            };

            if (tag("bucket_seq_len") == "600" &&
                tag("real_token_count") == "595" &&
                tag("capture_phase") == capture_phase &&
                tag("cache_phase") == cache_phase &&
                tag("recapture_reason") == recapture_reason)
            {
                total += record.value;
            }
        }
        return total;
    };

    auto forward_lookup_count = [&](const std::string &result) -> double
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph" ||
                record.name != "forward_cache_lookup" ||
                record.phase != "prefill")
            {
                continue;
            }
            const auto seq_it = record.tags.find("seq_len");
            const auto result_it = record.tags.find("result");
            if (seq_it != record.tags.end() && seq_it->second == "600" &&
                result_it != record.tags.end() && result_it->second == result)
            {
                total += record.value;
            }
        }
        return total;
    };

    EXPECT_GE(forward_lookup_count("miss"), 1.0);
    EXPECT_GE(forward_lookup_count("hit"), 1.0);
    /*
     * `clearCache()` now resets request-owned KV/GDN/short-conv state and
     * invalidates any prefill executable that could have captured pointers into
     * that state.  The padded bucket cache should still hit the host-side
     * forward graph, but each post-reset prefill must re-arm capture from Cold
     * instead of replaying or capturing an executable across request state.
     */
    EXPECT_GE(lifecycle_count("warmup", "warmup", "none"), 2.0);
    EXPECT_EQ(lifecycle_count("capture", "ready", "armed_warmup"), 0.0);
    PerfStatsCollector::reset();
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmPrefixCacheMTPRealModelSmoke)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "0"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
    });

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() <= 0)
    {
        GTEST_SKIP() << "No ROCm device available for Qwen3.6 prefix+MTP smoke";
    }
    const int rocm_ordinal = qwen36RocmSingleDeviceOrdinal();
    ASSERT_GE(rocm_ordinal, 0);
    ASSERT_LT(rocm_ordinal, dm.rocm_device_count())
        << "Selected ROCm device ordinal is outside the available device range";

    auto factory = createOrchestrationRunnerFactory();
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    const std::string prompt_text = "Paris is";

    auto make_config = [&](bool enable_prefix_cache, bool enable_mtp)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 32;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.device_for_this_rank = GlobalDeviceAddress::rocm(rocm_ordinal);
        config.kv_cache_precision = "auto";
        config.prefix_cache.enabled = enable_prefix_cache;
        config.prefix_cache.storage_mode = enable_prefix_cache
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = 4;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = 1;
        return config;
    };

    auto tokenize_prompt = [&](IOrchestrationRunner &runner)
    {
        auto tokenizer = runner.tokenizer();
        if (!tokenizer)
        {
            return std::vector<int32_t>{};
        }
        const auto encoded = tokenizer->encode(prompt_text, /*add_bos=*/false, /*add_eos=*/false);
        return std::vector<int32_t>(encoded.begin(), encoded.end());
    };

    auto baseline = factory->createFromOrchestrationConfig(make_config(false, false));
    ASSERT_NE(baseline, nullptr);
    ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
    const auto baseline_prompt = tokenize_prompt(*baseline);
    ASSERT_FALSE(baseline_prompt.empty());
    auto baseline_result = baseline->generate(baseline_prompt, 4, greedy);
    const auto baseline_snapshot = baseline->prefixStateProbe();
    baseline->shutdown();

    ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
    ASSERT_EQ(baseline_result.tokens.size(), 4u);
    EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);
    EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);

    auto cached = factory->createFromOrchestrationConfig(make_config(true, true));
    ASSERT_NE(cached, nullptr);
    ASSERT_TRUE(cached->initialize()) << cached->lastError();
    const auto cached_prompt = tokenize_prompt(*cached);
    ASSERT_FALSE(cached_prompt.empty());
    ASSERT_EQ(cached_prompt, baseline_prompt);

    auto first = cached->generate(cached_prompt, 4, greedy);
    const auto after_first = cached->prefixStateProbe();
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 4u);
    EXPECT_EQ(first.tokens, baseline_result.tokens);
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 1u);
    EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);
    EXPECT_GE(after_first.mtp_draft_steps, 2u);

    auto second = cached->generate(cached_prompt, 4, greedy);
    const auto after_second = cached->prefixStateProbe();
    cached->shutdown();

    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 4u);
    EXPECT_EQ(second.tokens, baseline_result.tokens);
    EXPECT_TRUE(after_second.prefix_cache_ready);
    EXPECT_GE(after_second.prefix_cache_hits, 1u);
    EXPECT_GE(after_second.prefix_cache_matched_tokens, static_cast<uint64_t>(cached_prompt.size()));
    EXPECT_TRUE(after_second.prefix_request.hit);
    EXPECT_EQ(after_second.prefix_request.matched_tokens,
              static_cast<int>(cached_prompt.size()));
    EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
    EXPECT_TRUE(after_second.prefix_request.terminal_hidden_restored);
    EXPECT_TRUE(after_second.prefix_request.mtp_state_restored);
    // MTP counters are request-local after each prefill/generate request.
    EXPECT_GE(after_second.mtp_draft_steps, 2u);
    EXPECT_GE(after_second.mtp_verifier_runs, 1u);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmLocalTPMTPRealModelSmoke)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
    });

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() < 2)
    {
        GTEST_SKIP() << "Need at least two ROCm devices for Qwen3.6 LocalTP MTP smoke";
    }

    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.model_path = model_path;
    config.max_seq_len = 32;
    config.batch_size = 1;
    config.tp_degree = 2;
    config.tp_scope = TPScope::LOCAL;
    config.tp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    config.pp_degree = 1;
    config.kv_cache_precision = "auto";
    config.mtp.enabled = true;
    config.mtp.draft_tokens = 1;

    auto factory = createOrchestrationRunnerFactory();
    auto runner = factory->createFromOrchestrationConfig(config);
    ASSERT_NE(runner, nullptr);
    ASSERT_TRUE(runner->initialize()) << runner->lastError();

    auto tokenizer = runner->tokenizer();
    ASSERT_NE(tokenizer, nullptr);
    const auto encoded = tokenizer->encode("Paris is", /*add_bos=*/false, /*add_eos=*/false);
    ASSERT_FALSE(encoded.empty());
    const std::vector<int32_t> prompt(encoded.begin(), encoded.end());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    auto result = runner->generate(prompt, 2, greedy);
    const auto snapshot = runner->prefixStateProbe();
    runner->shutdown();

    ASSERT_TRUE(result.error.empty()) << result.error;
    ASSERT_FALSE(result.tokens.empty());
    EXPECT_FALSE(snapshot.mtp_bypassed) << snapshot.mtp_bypass_reason;
    EXPECT_GE(snapshot.mtp_draft_steps, 1u);
    EXPECT_GE(snapshot.mtp_verifier_runs, 1u);
    EXPECT_GE(snapshot.mtp_accepted_tokens + snapshot.mtp_rejected_tokens, 1u);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36ROCmLocalTPPrefixCacheMTPRealModelSmoke)
{
    ScopedDebugEnv env({
        {"LLAMINAR_GPU_GRAPHS", "0"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
    });

    const char *env_model = std::getenv("LLAMINAR_QWEN36_DENSE_MODEL");
    if (!env_model)
        env_model = std::getenv("LLAMINAR_PARITY_DENSE_MODEL");
    const std::string model_path = env_model ? env_model : "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf";

    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 dense smoke model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() < 2)
    {
        GTEST_SKIP() << "Need at least two ROCm devices for Qwen3.6 LocalTP prefix+MTP smoke";
    }

    auto make_config = [&](bool enable_prefix_cache, bool enable_mtp)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = 32;
        config.batch_size = 1;
        config.tp_degree = 2;
        config.tp_scope = TPScope::LOCAL;
        config.tp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        config.pp_degree = 1;
        config.kv_cache_precision = "auto";
        config.prefix_cache.enabled = enable_prefix_cache;
        config.prefix_cache.storage_mode = enable_prefix_cache
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = 2;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 1024ull * 1024ull * 1024ull;
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = 1;
        return config;
    };

    auto factory = createOrchestrationRunnerFactory();
    SamplingParams greedy;
    greedy.temperature = 0.0f;

    auto baseline = factory->createFromOrchestrationConfig(make_config(false, false));
    ASSERT_NE(baseline, nullptr);
    ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
    auto baseline_tokenizer = baseline->tokenizer();
    ASSERT_NE(baseline_tokenizer, nullptr);
    const auto encoded = baseline_tokenizer->encode("Paris is", /*add_bos=*/false, /*add_eos=*/false);
    ASSERT_FALSE(encoded.empty());
    const std::vector<int32_t> prompt(encoded.begin(), encoded.end());

    auto baseline_result = baseline->generate(prompt, 4, greedy);
    const auto baseline_snapshot = baseline->prefixStateProbe();
    baseline->shutdown();

    ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
    ASSERT_EQ(baseline_result.tokens.size(), 4u);
    EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);
    EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);

    auto cached = factory->createFromOrchestrationConfig(make_config(true, true));
    ASSERT_NE(cached, nullptr);
    ASSERT_TRUE(cached->initialize()) << cached->lastError();

    auto first = cached->generate(prompt, 4, greedy);
    const auto after_first = cached->prefixStateProbe();
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_EQ(first.tokens.size(), 4u);
    EXPECT_EQ(first.tokens, baseline_result.tokens);
    EXPECT_TRUE(after_first.prefix_cache_ready);
    EXPECT_GE(after_first.prefix_cache_inserts, 2u);
    EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);
    EXPECT_FALSE(after_first.mtp_bypassed) << after_first.mtp_bypass_reason;
    EXPECT_GE(after_first.mtp_draft_steps, 1u);

    auto second = cached->generate(prompt, 4, greedy);
    const auto after_second = cached->prefixStateProbe();
    cached->shutdown();

    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_EQ(second.tokens.size(), 4u);
    EXPECT_EQ(second.tokens, baseline_result.tokens);
    EXPECT_TRUE(after_second.prefix_cache_ready);
    EXPECT_GE(after_second.prefix_cache_hits, 2u);
    EXPECT_GE(after_second.prefix_cache_matched_tokens, static_cast<uint64_t>(prompt.size() * 2));
    EXPECT_TRUE(after_second.prefix_request.hit);
    EXPECT_EQ(after_second.prefix_request.matched_tokens, static_cast<int>(prompt.size()));
    EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
    EXPECT_TRUE(after_second.prefix_request.terminal_hidden_restored);
    EXPECT_TRUE(after_second.prefix_request.mtp_state_restored);
    EXPECT_FALSE(after_second.mtp_bypassed) << after_second.mtp_bypass_reason;
    // MTP counters are request-local after each prefill/generate request.
    EXPECT_GE(after_second.mtp_draft_steps, 1u);
    EXPECT_GE(after_second.mtp_verifier_runs, 1u);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36MoEExpertOverlayROCm2TPLLEPLongContextStateContinuity)
{
    const int block_size = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_STATE_PROBE_BLOCK_TOKENS"},
        256);
    const int requested_prompt_tokens = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_STATE_PROBE_PROMPT_TOKENS"},
        1024);
    ASSERT_GT(block_size, 0);
    ASSERT_GT(requested_prompt_tokens, block_size);

    const int decode_steps = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_STATE_PROBE_DECODE_TOKENS"},
        2);
    ASSERT_GT(decode_steps, 0);

    const int boundary_start = std::max(0, block_size - 8);
    const int boundary_tokens = std::max(
        1,
        std::min(16, requested_prompt_tokens - boundary_start));
    const int tail_start = std::max(block_size, requested_prompt_tokens - 64);
    const int tail_tokens = requested_prompt_tokens - tail_start;
    std::ostringstream kv_segments;
    kv_segments << "prefix=0:" << block_size
                << ";boundary=" << boundary_start
                << ":" << boundary_tokens
                << ";tail=" << tail_start << ":" << tail_tokens;
    for (int start = 0; start < requested_prompt_tokens; start += 128)
    {
        const int tokens = std::min(128, requested_prompt_tokens - start);
        kv_segments << ";tile_" << std::setw(4) << std::setfill('0') << start
                    << std::setfill(' ') << "=" << start << ":" << tokens;
    }
    const bool stage_continuity_diagnostic_requested =
        DebugEnv::isTruthyEnv(
            "LLAMINAR_QWEN36_MOE_OVERLAY_STAGE_CONTINUITY_DIAG");

    ScopedDebugEnv env({
        {"LLAMINAR_LOG_LEVEL",
         stage_continuity_diagnostic_requested ? "INFO" : "WARN"},
        // Match the production LocalTP E2E lane: decode and collective work is
        // graph captured, while this topology deliberately uses unbucketed
        // prefill. Do not disable ROCm/CUDA production scheduling knobs here.
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
        {"LLAMINAR_PERF_STATS_JSON", "1"},
        {"LLAMINAR_PERF_STATS_FILTER", "prefix_cache,request_admission"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
        {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT", "1"},
        {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT_LAYER", "3"},
        {"LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT", std::to_string(block_size).c_str()},
        {"LLAMINAR_PREFIX_PROBE_KV_SEGMENTS", kv_segments.str().c_str()},
    });

    if (mpiWorldSize() != 1)
    {
        GTEST_SKIP() << "Qwen3.6 MoE ExpertOverlay state-continuity probe must run with one MPI rank";
    }

    const std::string model_path = firstEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_MODEL", "LLAMINAR_PARITY_MOE_MODEL"},
        "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf");
    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 MoE model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() < 2)
    {
        GTEST_SKIP() << "Need at least two ROCm devices for Qwen3.6 MoE ExpertOverlay state-continuity probe";
    }
    PerfStatsCollector::reset();

    const int max_seq_len = std::max(
        requested_prompt_tokens + decode_steps + 64,
        block_size * 2 + decode_steps + 64);
    auto make_config = [&](bool enable_prefix_cache)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = max_seq_len;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "auto";
        config.tp_allreduce_precision_override = "schema";
        config.prefix_cache.enabled = enable_prefix_cache;
        config.prefix_cache.storage_mode = enable_prefix_cache
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = block_size;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.moe_policy = PrefixCacheMoEPolicy::PlacementFingerprint;
        config.prefix_cache.ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
        config.moe_routed_expert_plan = qwen36MoEOverlayRocm2TPHotOnlyForProbe();
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::LLEP;
        config.moe_rebalance.window_size = 4;
        config.moe_rebalance.max_window_size = 4;
        config.moe_rebalance.window_growth_factor = 1.0f;
        config.moe_rebalance.prefill_window_tokens = block_size;
        config.moe_rebalance.dynamic_imbalance_threshold_per_mille = 0;
        config.moe_rebalance.dynamic_min_improvement_per_mille = 0;
        config.moe_rebalance.dynamic_max_swaps_per_layer = 20;
        config.moe_rebalance.dynamic_max_plan_entries_per_wave = 20;
        config.moe_rebalance.dynamic_min_window_activations = 0;
        config.moe_rebalance.device_min_load_spread_improvement = 0;
        config.moe_rebalance.device_min_load_spread_improvement_divisor = 0;
        config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_min_foreign_rows_per_transfer = 0;
        config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_max_post_wave_load_spread_per_mille = 1000;
        config.moe_rebalance.release_raw_expert_weights = true;
        return config;
    };

    auto factory = createOrchestrationRunnerFactory();
    SamplingParams greedy;
    greedy.temperature = 0.0f;

    PrefixRuntimeStateSnapshot full_prefill_probe;
    PrefixRuntimeStateSnapshot split_prefill_probe;
    PrefixRuntimeStateSnapshot restored_prefill_probe;
    GenerationResult full_decode;
    GenerationResult split_decode;
    GenerationResult restored_decode;
    std::vector<int32_t> prompt;
    std::vector<int32_t> seed_prompt;
    std::vector<int32_t> suffix_prompt;
    const bool capture_stage_continuity =
        stage_continuity_diagnostic_requested;
    const std::vector<std::string> stage_continuity_keys =
        capture_stage_continuity
            ? prefillReplayContinuitySnapshotKeys(
                  /*first_layer=*/31,
                  /*last_layer=*/39)
            : std::vector<std::string>{};
    std::map<std::string, RequestBatchStageSnapshot> full_prefill_stage_snapshots;
    std::map<std::string, RequestBatchStageSnapshot> split_prefill_stage_snapshots;
    std::map<std::string, RequestBatchStageSnapshot> restored_prefill_stage_snapshots;

    {
        auto baseline = factory->createFromOrchestrationConfig(make_config(false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto tokenizer = baseline->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        prompt = buildDeterministicPromptTokens(
            *tokenizer,
            static_cast<size_t>(requested_prompt_tokens));
        ASSERT_EQ(prompt.size(), static_cast<size_t>(requested_prompt_tokens));
        seed_prompt.assign(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(block_size));
        suffix_prompt.assign(
            prompt.begin() + static_cast<std::ptrdiff_t>(block_size),
            prompt.end());
        baseline->setSamplingParams(greedy);
        if (capture_stage_continuity)
        {
            baseline->setSnapshotCaptureFilter(stage_continuity_keys);
            baseline->enableSnapshotCapture();
            baseline->clearSnapshots();
        }

        ASSERT_TRUE(baseline->prefill(prompt))
            << baseline->lastError();
        full_prefill_probe = baseline->prefixStateProbe();
        if (capture_stage_continuity)
        {
            full_prefill_stage_snapshots = captureRequestBatchSnapshots(
                *baseline,
                stage_continuity_keys);
        }
        full_decode = decodeGreedyTokens(
            *baseline,
            decode_steps,
            "full-prefill baseline decode");
        ASSERT_TRUE(full_decode.error.empty()) << full_decode.error;
        ASSERT_EQ(full_decode.tokens.size(), static_cast<size_t>(decode_steps));

        baseline->clearCache();
        ASSERT_TRUE(baseline->prefill(seed_prompt))
            << baseline->lastError();
        if (capture_stage_continuity)
            baseline->clearSnapshots();
        ASSERT_TRUE(baseline->prefill(suffix_prompt))
            << baseline->lastError();
        split_prefill_probe = baseline->prefixStateProbe();
        if (capture_stage_continuity)
        {
            split_prefill_stage_snapshots = captureRequestBatchSnapshots(
                *baseline,
                stage_continuity_keys);
        }
        split_decode = decodeGreedyTokens(
            *baseline,
            decode_steps,
            "split-prefill baseline decode");
        ASSERT_TRUE(split_decode.error.empty()) << split_decode.error;
        ASSERT_EQ(split_decode.tokens.size(), static_cast<size_t>(decode_steps));
        if (capture_stage_continuity)
            baseline->disableSnapshotCapture();
        baseline->shutdown();
    }

    if (capture_stage_continuity)
    {
        ASSERT_TRUE(prefillReplaySnapshotsByteIdentical(
            full_prefill_stage_snapshots,
            split_prefill_stage_snapshots,
            stage_continuity_keys,
            static_cast<size_t>(requested_prompt_tokens),
            static_cast<size_t>(block_size),
            static_cast<size_t>(requested_prompt_tokens - block_size)));
    }

    {
        auto cached = factory->createFromOrchestrationConfig(make_config(true));
        ASSERT_NE(cached, nullptr);
        ASSERT_TRUE(cached->initialize()) << cached->lastError();
        cached->setSamplingParams(greedy);
        if (capture_stage_continuity)
        {
            cached->setSnapshotCaptureFilter(stage_continuity_keys);
            cached->enableSnapshotCapture();
            cached->clearSnapshots();
        }

        ASSERT_TRUE(cached->prefill(seed_prompt))
            << cached->lastError();
        const auto seed_probe = cached->prefixStateProbe();
        ASSERT_TRUE(seed_probe.prefix_cache_ready);
        EXPECT_GE(seed_probe.prefix_cache_inserts, 1u);

        const auto seed_miss_decode = decodeGreedyTokens(
            *cached,
            /*steps=*/2,
            "seed-prefix miss decode");
        ASSERT_TRUE(seed_miss_decode.error.empty())
            << seed_miss_decode.error;
        ASSERT_EQ(seed_miss_decode.tokens.size(), 2u);

        cached->clearCache();
        ASSERT_TRUE(cached->prefill(seed_prompt))
            << cached->lastError();
        const auto seed_full_hit_probe = cached->prefixStateProbe();
        EXPECT_GE(
            seed_full_hit_probe.prefix_cache_hits +
                seed_full_hit_probe.prefix_cache_partial_hits,
            1u);
        EXPECT_GE(
            seed_full_hit_probe.prefix_cache_matched_tokens,
            static_cast<uint64_t>(block_size));
        const auto seed_full_hit_decode = decodeGreedyTokens(
            *cached,
            /*steps=*/2,
            "seed-prefix full-hit decode");
        ASSERT_TRUE(seed_full_hit_decode.error.empty())
            << seed_full_hit_decode.error;
        EXPECT_EQ(seed_full_hit_decode.tokens, seed_miss_decode.tokens)
            << "Exact prefix hits must rehydrate live LLEP placement before "
               "the first M=1 decode graph.";

        cached->clearCache();
        if (capture_stage_continuity)
            cached->clearSnapshots();
        ASSERT_TRUE(cached->prefill(prompt))
            << cached->lastError();
        restored_prefill_probe = cached->prefixStateProbe();
        if (capture_stage_continuity)
        {
            restored_prefill_stage_snapshots = captureRequestBatchSnapshots(
                *cached,
                stage_continuity_keys);
        }
        restored_decode = decodeGreedyTokens(
            *cached,
            decode_steps,
            "prefix-restored decode");
        ASSERT_TRUE(restored_decode.error.empty()) << restored_decode.error;
        ASSERT_EQ(restored_decode.tokens.size(), static_cast<size_t>(decode_steps));
        if (capture_stage_continuity)
            cached->disableSnapshotCapture();
        cached->shutdown();
    }
    llaminar::v2::kernels::KernelFactory::clearCache();
    const auto rehydration_records =
        PerfStatsCollector::snapshot({"prefix_cache", "request_admission"});
    expectPortableMoEDeviceRehydrationCoverage(
        rehydration_records,
        "ROCm",
        /*participant_count=*/2,
        /*routed_layer_count=*/40);
    expectRequestInputReuseCoverage(
        rehydration_records,
        "ROCm",
        /*participant_count=*/2);
    PerfStatsCollector::reset();

    EXPECT_TRUE(restored_prefill_probe.prefix_cache_ready);
    EXPECT_GE(restored_prefill_probe.prefix_cache_hits +
                  restored_prefill_probe.prefix_cache_partial_hits,
              1u);
    EXPECT_GE(restored_prefill_probe.prefix_cache_matched_tokens,
              static_cast<uint64_t>(block_size));

    MTPRuntimeSnapshotComparisonOptions compare_options;
    compare_options.compare_main_kv_payload_hashes = true;
    compare_options.compare_shifted_mtp_kv = false;
    compare_options.compare_gdn_hashes = true;

    const MTPStateValidationResult full_vs_split =
        compareMTPRuntimeStateSnapshots(
            full_prefill_probe,
            split_prefill_probe,
            compare_options);
    ASSERT_TRUE(full_vs_split)
        << "Qwen3.6 MoE ExpertOverlay LLEP split-prefill state drifted from "
        << "single-request full prefill: " << full_vs_split.reason
        << "\n" << summarizeStateContinuityDifferences(
               full_prefill_probe,
               split_prefill_probe)
        << "\nfull: " << summarizeStateContinuityProbe(full_prefill_probe)
        << "\nsplit: " << summarizeStateContinuityProbe(split_prefill_probe);

    const MTPStateValidationResult full_vs_restored =
        compareMTPRuntimeStateSnapshots(
            full_prefill_probe,
            restored_prefill_probe,
            compare_options);
    if (capture_stage_continuity && !full_vs_restored)
    {
        ASSERT_TRUE(prefillReplaySnapshotsByteIdentical(
            split_prefill_stage_snapshots,
            restored_prefill_stage_snapshots,
            stage_continuity_keys,
            static_cast<size_t>(requested_prompt_tokens - block_size),
            /*suffix_row_offset=*/0,
            static_cast<size_t>(requested_prompt_tokens - block_size)));
    }
    ASSERT_TRUE(full_vs_restored)
        << "Qwen3.6 MoE ExpertOverlay LLEP prefix-restored state drifted from "
        << "single-request full prefill: " << full_vs_restored.reason
        << "\n" << summarizeStateContinuityDifferences(
               full_prefill_probe,
               restored_prefill_probe)
        << "\nfull: " << summarizeStateContinuityProbe(full_prefill_probe)
        << "\nrestored: " << summarizeStateContinuityProbe(restored_prefill_probe);

    EXPECT_EQ(split_decode.tokens, full_decode.tokens)
        << "split prefill continuation must match full prefill continuation";
    EXPECT_EQ(restored_decode.tokens, full_decode.tokens)
        << "prefix-restored continuation must match full prefill continuation";
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36MoEExpertOverlayCUDA2TPLLEPLongContextStateContinuity)
{
    const int block_size = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_STATE_PROBE_BLOCK_TOKENS"},
        256);
    const int requested_prompt_tokens = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_STATE_PROBE_PROMPT_TOKENS"},
        1024);
    ASSERT_GT(block_size, 0);
    ASSERT_GT(requested_prompt_tokens, block_size);

    const int decode_steps = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_STATE_PROBE_DECODE_TOKENS"},
        2);
    ASSERT_GT(decode_steps, 0);

    const int boundary_start = std::max(0, block_size - 8);
    const int boundary_tokens = std::max(
        1,
        std::min(16, requested_prompt_tokens - boundary_start));
    const int tail_start = std::max(block_size, requested_prompt_tokens - 64);
    const int tail_tokens = requested_prompt_tokens - tail_start;
    std::ostringstream kv_segments;
    kv_segments << "prefix=0:" << block_size
                << ";boundary=" << boundary_start
                << ":" << boundary_tokens
                << ";tail=" << tail_start << ":" << tail_tokens;
    for (int start = 0; start < requested_prompt_tokens; start += 128)
    {
        const int tokens = std::min(128, requested_prompt_tokens - start);
        kv_segments << ";tile_" << std::setw(4) << std::setfill('0') << start
                    << std::setfill(' ') << "=" << start << ":" << tokens;
    }
    const bool stage_continuity_diagnostic_requested =
        DebugEnv::isTruthyEnv(
            "LLAMINAR_QWEN36_MOE_OVERLAY_STAGE_CONTINUITY_DIAG");

    ScopedDebugEnv env({
        {"LLAMINAR_LOG_LEVEL",
         stage_continuity_diagnostic_requested ? "INFO" : "WARN"},
        // Match the production LocalTP E2E lane: decode and collective work is
        // graph captured, while this topology deliberately uses unbucketed
        // prefill. Do not disable ROCm/CUDA production scheduling knobs here.
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
        {"LLAMINAR_PERF_STATS_JSON", "1"},
        {"LLAMINAR_PERF_STATS_FILTER", "prefix_cache,request_admission"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
        {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT", "1"},
        {"LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT_LAYER", "3"},
        {"LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT", std::to_string(block_size).c_str()},
        {"LLAMINAR_PREFIX_PROBE_KV_SEGMENTS", kv_segments.str().c_str()},
    });

    if (mpiWorldSize() != 1)
    {
        GTEST_SKIP() << "Qwen3.6 MoE ExpertOverlay state-continuity probe must run with one MPI rank";
    }

    const std::string model_path = firstEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_MODEL", "LLAMINAR_PARITY_MOE_MODEL"},
        "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf");
    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 MoE model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() < 2)
    {
        GTEST_SKIP() << "Need at least two CUDA devices for Qwen3.6 MoE ExpertOverlay state-continuity probe";
    }
    PerfStatsCollector::reset();

    const int max_seq_len = std::max(
        requested_prompt_tokens + decode_steps + 64,
        block_size * 2 + decode_steps + 64);
    auto make_config = [&](bool enable_prefix_cache)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = max_seq_len;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "auto";
        config.tp_allreduce_precision_override = "schema";
        config.prefix_cache.enabled = enable_prefix_cache;
        config.prefix_cache.storage_mode = enable_prefix_cache
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = block_size;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.moe_policy = PrefixCacheMoEPolicy::PlacementFingerprint;
        config.prefix_cache.ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
        config.moe_routed_expert_plan = qwen36MoEOverlayCuda2TPHotOnlyForProbe();
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::LLEP;
        config.moe_rebalance.window_size = 4;
        config.moe_rebalance.max_window_size = 4;
        config.moe_rebalance.window_growth_factor = 1.0f;
        config.moe_rebalance.prefill_window_tokens = block_size;
        config.moe_rebalance.dynamic_imbalance_threshold_per_mille = 0;
        config.moe_rebalance.dynamic_min_improvement_per_mille = 0;
        config.moe_rebalance.dynamic_max_swaps_per_layer = 20;
        config.moe_rebalance.dynamic_max_plan_entries_per_wave = 20;
        config.moe_rebalance.dynamic_min_window_activations = 0;
        config.moe_rebalance.device_min_load_spread_improvement = 0;
        config.moe_rebalance.device_min_load_spread_improvement_divisor = 0;
        config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_min_foreign_rows_per_transfer = 0;
        config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_max_post_wave_load_spread_per_mille = 1000;
        config.moe_rebalance.release_raw_expert_weights = true;
        return config;
    };

    auto factory = createOrchestrationRunnerFactory();
    SamplingParams greedy;
    greedy.temperature = 0.0f;

    PrefixRuntimeStateSnapshot full_prefill_probe;
    PrefixRuntimeStateSnapshot split_prefill_probe;
    PrefixRuntimeStateSnapshot restored_prefill_probe;
    GenerationResult full_decode;
    GenerationResult split_decode;
    GenerationResult restored_decode;
    std::vector<int32_t> prompt;
    std::vector<int32_t> seed_prompt;
    std::vector<int32_t> suffix_prompt;
    const bool capture_stage_continuity =
        stage_continuity_diagnostic_requested;
    const std::vector<std::string> stage_continuity_keys =
        capture_stage_continuity
            ? prefillReplayContinuitySnapshotKeys(
                  /*first_layer=*/31,
                  /*last_layer=*/39)
            : std::vector<std::string>{};
    std::map<std::string, RequestBatchStageSnapshot> full_prefill_stage_snapshots;
    std::map<std::string, RequestBatchStageSnapshot> split_prefill_stage_snapshots;
    std::map<std::string, RequestBatchStageSnapshot> restored_prefill_stage_snapshots;

    {
        auto baseline = factory->createFromOrchestrationConfig(make_config(false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto tokenizer = baseline->tokenizer();
        ASSERT_NE(tokenizer, nullptr);
        prompt = buildDeterministicPromptTokens(
            *tokenizer,
            static_cast<size_t>(requested_prompt_tokens));
        ASSERT_EQ(prompt.size(), static_cast<size_t>(requested_prompt_tokens));
        seed_prompt.assign(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(block_size));
        suffix_prompt.assign(
            prompt.begin() + static_cast<std::ptrdiff_t>(block_size),
            prompt.end());
        baseline->setSamplingParams(greedy);
        if (capture_stage_continuity)
        {
            baseline->setSnapshotCaptureFilter(stage_continuity_keys);
            baseline->enableSnapshotCapture();
            baseline->clearSnapshots();
        }

        ASSERT_TRUE(baseline->prefill(prompt))
            << baseline->lastError();
        full_prefill_probe = baseline->prefixStateProbe();
        if (capture_stage_continuity)
        {
            full_prefill_stage_snapshots = captureRequestBatchSnapshots(
                *baseline,
                stage_continuity_keys);
        }
        full_decode = decodeGreedyTokens(
            *baseline,
            decode_steps,
            "full-prefill baseline decode");
        ASSERT_TRUE(full_decode.error.empty()) << full_decode.error;
        ASSERT_EQ(full_decode.tokens.size(), static_cast<size_t>(decode_steps));

        baseline->clearCache();
        ASSERT_TRUE(baseline->prefill(seed_prompt))
            << baseline->lastError();
        if (capture_stage_continuity)
            baseline->clearSnapshots();
        ASSERT_TRUE(baseline->prefill(suffix_prompt))
            << baseline->lastError();
        split_prefill_probe = baseline->prefixStateProbe();
        if (capture_stage_continuity)
        {
            split_prefill_stage_snapshots = captureRequestBatchSnapshots(
                *baseline,
                stage_continuity_keys);
        }
        split_decode = decodeGreedyTokens(
            *baseline,
            decode_steps,
            "split-prefill baseline decode");
        ASSERT_TRUE(split_decode.error.empty()) << split_decode.error;
        ASSERT_EQ(split_decode.tokens.size(), static_cast<size_t>(decode_steps));
        if (capture_stage_continuity)
            baseline->disableSnapshotCapture();
        baseline->shutdown();
    }

    if (capture_stage_continuity)
    {
        ASSERT_TRUE(prefillReplaySnapshotsByteIdentical(
            full_prefill_stage_snapshots,
            split_prefill_stage_snapshots,
            stage_continuity_keys,
            static_cast<size_t>(requested_prompt_tokens),
            static_cast<size_t>(block_size),
            static_cast<size_t>(requested_prompt_tokens - block_size)));
    }

    {
        auto cached = factory->createFromOrchestrationConfig(make_config(true));
        ASSERT_NE(cached, nullptr);
        ASSERT_TRUE(cached->initialize()) << cached->lastError();
        cached->setSamplingParams(greedy);
        if (capture_stage_continuity)
        {
            cached->setSnapshotCaptureFilter(stage_continuity_keys);
            cached->enableSnapshotCapture();
            cached->clearSnapshots();
        }

        ASSERT_TRUE(cached->prefill(seed_prompt))
            << cached->lastError();
        const auto seed_probe = cached->prefixStateProbe();
        ASSERT_TRUE(seed_probe.prefix_cache_ready);
        EXPECT_GE(seed_probe.prefix_cache_inserts, 1u);

        const auto seed_miss_decode = decodeGreedyTokens(
            *cached,
            /*steps=*/2,
            "seed-prefix miss decode");
        ASSERT_TRUE(seed_miss_decode.error.empty())
            << seed_miss_decode.error;
        ASSERT_EQ(seed_miss_decode.tokens.size(), 2u);

        cached->clearCache();
        ASSERT_TRUE(cached->prefill(seed_prompt))
            << cached->lastError();
        const auto seed_full_hit_probe = cached->prefixStateProbe();
        EXPECT_GE(
            seed_full_hit_probe.prefix_cache_hits +
                seed_full_hit_probe.prefix_cache_partial_hits,
            1u);
        EXPECT_GE(
            seed_full_hit_probe.prefix_cache_matched_tokens,
            static_cast<uint64_t>(block_size));
        const auto seed_full_hit_decode = decodeGreedyTokens(
            *cached,
            /*steps=*/2,
            "seed-prefix full-hit decode");
        ASSERT_TRUE(seed_full_hit_decode.error.empty())
            << seed_full_hit_decode.error;
        EXPECT_EQ(seed_full_hit_decode.tokens, seed_miss_decode.tokens)
            << "Exact prefix hits must rehydrate live LLEP placement before "
               "the first M=1 decode graph.";

        cached->clearCache();
        if (capture_stage_continuity)
            cached->clearSnapshots();
        ASSERT_TRUE(cached->prefill(prompt))
            << cached->lastError();
        restored_prefill_probe = cached->prefixStateProbe();
        if (capture_stage_continuity)
        {
            restored_prefill_stage_snapshots = captureRequestBatchSnapshots(
                *cached,
                stage_continuity_keys);
        }
        restored_decode = decodeGreedyTokens(
            *cached,
            decode_steps,
            "prefix-restored decode");
        ASSERT_TRUE(restored_decode.error.empty()) << restored_decode.error;
        ASSERT_EQ(restored_decode.tokens.size(), static_cast<size_t>(decode_steps));
        if (capture_stage_continuity)
            cached->disableSnapshotCapture();
        cached->shutdown();
    }
    llaminar::v2::kernels::KernelFactory::clearCache();
    const auto rehydration_records =
        PerfStatsCollector::snapshot({"prefix_cache", "request_admission"});
    expectPortableMoEDeviceRehydrationCoverage(
        rehydration_records,
        "CUDA",
        /*participant_count=*/2,
        /*routed_layer_count=*/40);
    expectRequestInputReuseCoverage(
        rehydration_records,
        "CUDA",
        /*participant_count=*/2);
    PerfStatsCollector::reset();

    EXPECT_TRUE(restored_prefill_probe.prefix_cache_ready);
    EXPECT_GE(restored_prefill_probe.prefix_cache_hits +
                  restored_prefill_probe.prefix_cache_partial_hits,
              1u);
    EXPECT_GE(restored_prefill_probe.prefix_cache_matched_tokens,
              static_cast<uint64_t>(block_size));

    MTPRuntimeSnapshotComparisonOptions compare_options;
    compare_options.compare_main_kv_payload_hashes = true;
    compare_options.compare_shifted_mtp_kv = false;
    compare_options.compare_gdn_hashes = true;

    const MTPStateValidationResult full_vs_split =
        compareMTPRuntimeStateSnapshots(
            full_prefill_probe,
            split_prefill_probe,
            compare_options);
    ASSERT_TRUE(full_vs_split)
        << "Qwen3.6 MoE ExpertOverlay CUDA LLEP split-prefill state drifted from "
        << "single-request full prefill: " << full_vs_split.reason
        << "\n" << summarizeStateContinuityDifferences(
               full_prefill_probe,
               split_prefill_probe)
        << "\nfull: " << summarizeStateContinuityProbe(full_prefill_probe)
        << "\nsplit: " << summarizeStateContinuityProbe(split_prefill_probe);

    const MTPStateValidationResult full_vs_restored =
        compareMTPRuntimeStateSnapshots(
            full_prefill_probe,
            restored_prefill_probe,
            compare_options);
    if (capture_stage_continuity && !full_vs_restored)
    {
        ASSERT_TRUE(prefillReplaySnapshotsByteIdentical(
            split_prefill_stage_snapshots,
            restored_prefill_stage_snapshots,
            stage_continuity_keys,
            static_cast<size_t>(requested_prompt_tokens - block_size),
            /*suffix_row_offset=*/0,
            static_cast<size_t>(requested_prompt_tokens - block_size)));
    }
    ASSERT_TRUE(full_vs_restored)
        << "Qwen3.6 MoE ExpertOverlay CUDA LLEP prefix-restored state drifted from "
        << "single-request full prefill: " << full_vs_restored.reason
        << "\n" << summarizeStateContinuityDifferences(
               full_prefill_probe,
               restored_prefill_probe)
        << "\nfull: " << summarizeStateContinuityProbe(full_prefill_probe)
        << "\nrestored: " << summarizeStateContinuityProbe(restored_prefill_probe);

    EXPECT_EQ(split_decode.tokens, full_decode.tokens)
        << "split prefill continuation must match full prefill continuation";
    EXPECT_EQ(restored_decode.tokens, full_decode.tokens)
        << "prefix-restored continuation must match full prefill continuation";
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36MoEExpertOverlayROCm2TPLLEPNeedleRecallMatchesROCm2TP)
{
    ScopedDebugEnv env({
        // Preserve the production graph/concurrency profile while keeping the
        // unsupported LocalTP prefill-bucket optimization disabled.
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
    });

    if (mpiWorldSize() != 1)
    {
        GTEST_SKIP() << "Qwen3.6 MoE needle recall probe must run with one MPI rank";
    }

    const std::string model_path = firstEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_MODEL", "LLAMINAR_PARITY_MOE_MODEL"},
        "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf");
    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 MoE model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.rocm_device_count() < 2)
    {
        GTEST_SKIP() << "Need at least two ROCm devices for Qwen3.6 MoE needle recall probe";
    }

    const int context_length = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_RECALL_CONTEXT_TOKENS"},
        2048);
    const int max_tokens = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_RECALL_MAX_TOKENS"},
        64);
    ASSERT_GT(context_length, max_tokens + 512);
    ASSERT_GT(max_tokens, 0);

    auto configure_prefix_cache =
        [](OrchestrationConfig &config, bool enabled)
    {
        config.prefix_cache.enabled = enabled;
        config.prefix_cache.storage_mode = enabled
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = 64;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.moe_policy = PrefixCacheMoEPolicy::PlacementFingerprint;
        config.prefix_cache.ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    };

    auto make_rocm2_tp_config = [&]()
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = context_length;
        config.batch_size = 1;
        config.tp_degree = 2;
        config.tp_scope = TPScope::LOCAL;
        config.tp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "auto";
        configure_prefix_cache(config, true);
        return config;
    };

    auto factory = createOrchestrationRunnerFactory();

    auto make_overlay_config = [&](const OverlayRecallProbeConfig &probe)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = context_length;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "auto";
        config.tp_allreduce_precision_override = "schema";
        configure_prefix_cache(config, probe.prefix_cache);
        config.moe_routed_expert_plan =
            qwen36MoEOverlayRocm2TPHotOnlyForProbe(probe.routed_assignment_policy);
        config.moe_rebalance.mode = probe.mode;
        config.moe_rebalance.window_size = 4;
        config.moe_rebalance.max_window_size = 4;
        config.moe_rebalance.window_growth_factor = 1.0f;
        config.moe_rebalance.prefill_window_tokens =
            firstIntEnvOrDefault(
                {"LLAMINAR_QWEN36_MOE_OVERLAY_RECALL_PREFILL_WINDOW_TOKENS"},
                probe.prefill_window_tokens);
        config.moe_rebalance.dynamic_imbalance_threshold_per_mille = 0;
        config.moe_rebalance.dynamic_min_improvement_per_mille = 0;
        config.moe_rebalance.dynamic_max_swaps_per_layer = 20;
        config.moe_rebalance.dynamic_max_plan_entries_per_wave = 20;
        config.moe_rebalance.dynamic_min_window_activations = 0;
        config.moe_rebalance.device_min_load_spread_improvement = 0;
        config.moe_rebalance.device_min_load_spread_improvement_divisor = 0;
        config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_min_foreign_rows_per_transfer = 0;
        config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_max_post_wave_load_spread_per_mille = 1000;
        config.moe_rebalance.release_raw_expert_weights = true;
        return config;
    };

    auto run_overlay_probe = [&](const OverlayRecallProbeConfig &probe)
    {
        auto runner = factory->createFromOrchestrationConfig(make_overlay_config(probe));
        EXPECT_NE(runner, nullptr);
        if (!runner)
        {
            NeedleRecallRunResult result;
            result.placement = probe.label;
            result.error = "failed to create runner";
            return result;
        }
        NeedleRecallRunResult result;
        if (runner->initialize())
        {
            result = runNeedleRecallPrompt(
                *runner,
                "beginning",
                context_length,
                max_tokens);
            result.placement = probe.label;
        }
        else
        {
            result.placement = probe.label;
            result.error = runner->lastError();
        }
        runner->shutdown();
        return result;
    };

    OverlayRecallProbeConfig static_probe;
    static_probe.label = "static-overlay";
    static_probe.mode = MoERebalanceRuntimeMode::Off;
    static_probe.routed_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    static_probe.prefix_cache = true;
    static_probe.prefill_window_tokens = 0;
    static_probe.require_transfer_backing = false;

    OverlayRecallProbeConfig llep_probe;
    llep_probe.label = "llep-overlay-prefix-window";
    llep_probe.mode = MoERebalanceRuntimeMode::LLEP;
    llep_probe.routed_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    llep_probe.prefix_cache = true;
    llep_probe.prefill_window_tokens = 0;
    llep_probe.require_transfer_backing = true;

    NeedleRecallRunResult rocm2_result;
    {
        auto runner = factory->createFromOrchestrationConfig(make_rocm2_tp_config());
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        rocm2_result = runNeedleRecallPrompt(
            *runner,
            "beginning",
            context_length,
            max_tokens);
        runner->shutdown();
    }
    if (!rocm2_result.error.empty())
    {
        FAIL()
            << "Ordinary ROCm2 TP failed before the overlay comparison. "
            << summarizeNeedleRecallResult(rocm2_result);
    }
    if (!rocm2_result.containsTarget())
    {
        GTEST_SKIP()
            << "Ordinary ROCm2 TP did not satisfy the guarded needle prompt; "
            << "not treating it as an overlay-specific regression. "
            << summarizeNeedleRecallResult(rocm2_result);
    }

    const NeedleRecallRunResult static_overlay_result =
        run_overlay_probe(static_probe);
    ASSERT_TRUE(static_overlay_result.containsTarget())
        << "Qwen3.6 MoE ExpertOverlay ROCm2TP static-owner guard failed; "
        << "the regression is not isolated to LLEP.\nstandard: "
        << summarizeNeedleRecallResult(rocm2_result)
        << "\nstatic-overlay: "
        << summarizeNeedleRecallResult(static_overlay_result);

    const NeedleRecallRunResult overlay_result =
        run_overlay_probe(llep_probe);
    llaminar::v2::kernels::KernelFactory::clearCache();

    EXPECT_TRUE(overlay_result.containsTarget())
        << "Qwen3.6 MoE ExpertOverlay ROCm2TP LLEP lost a long-context needle "
        << "that ordinary ROCm2 TP recalled.\nstandard: "
        << summarizeNeedleRecallResult(rocm2_result)
        << "\nstatic-overlay: "
        << summarizeNeedleRecallResult(static_overlay_result)
        << "\noverlay: "
        << summarizeNeedleRecallResult(overlay_result);
}

TEST(Test__KVPrefixMTPStateProbe, Qwen36MoEExpertOverlayCUDA2TPLLEPNeedleRecallMatchesCUDA2TP)
{
    ScopedDebugEnv env({
        // Preserve the production graph/concurrency profile while keeping the
        // unsupported LocalTP prefill-bucket optimization disabled.
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0"},
        {"LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full"},
    });

    if (mpiWorldSize() != 1)
    {
        GTEST_SKIP() << "Qwen3.6 MoE needle recall probe must run with one MPI rank";
    }

    const std::string model_path = firstEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_MODEL", "LLAMINAR_PARITY_MOE_MODEL"},
        "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf");
    if (!std::filesystem::exists(model_path))
    {
        GTEST_SKIP() << "Qwen3.6 MoE model not found: " << model_path;
    }

    auto &dm = DeviceManager::instance();
    dm.initialize(-1, false);
    if (dm.cuda_device_count() < 2)
    {
        GTEST_SKIP() << "Need at least two CUDA devices for Qwen3.6 MoE needle recall probe";
    }

    const int context_length = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_RECALL_CONTEXT_TOKENS"},
        2048);
    const int max_tokens = firstIntEnvOrDefault(
        {"LLAMINAR_QWEN36_MOE_OVERLAY_RECALL_MAX_TOKENS"},
        64);
    ASSERT_GT(context_length, max_tokens + 512);
    ASSERT_GT(max_tokens, 0);

    auto configure_prefix_cache =
        [](OrchestrationConfig &config, bool enabled)
    {
        config.prefix_cache.enabled = enabled;
        config.prefix_cache.storage_mode = enabled
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = 64;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.moe_policy = PrefixCacheMoEPolicy::PlacementFingerprint;
        config.prefix_cache.ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    };

    auto make_cuda2_tp_config = [&]()
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = context_length;
        config.batch_size = 1;
        config.tp_degree = 2;
        config.tp_scope = TPScope::LOCAL;
        config.tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.pp_degree = 1;
        config.default_backend = CollectiveBackendType::NCCL;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "auto";
        configure_prefix_cache(config, true);
        return config;
    };

    auto factory = createOrchestrationRunnerFactory();

    auto make_overlay_config = [&](const OverlayRecallProbeConfig &probe)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = context_length;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = "auto";
        config.tp_allreduce_precision_override = "schema";
        configure_prefix_cache(config, probe.prefix_cache);
        config.moe_routed_expert_plan =
            qwen36MoEOverlayCuda2TPHotOnlyForProbe(probe.routed_assignment_policy);
        config.moe_rebalance.mode = probe.mode;
        config.moe_rebalance.window_size = 4;
        config.moe_rebalance.max_window_size = 4;
        config.moe_rebalance.window_growth_factor = 1.0f;
        config.moe_rebalance.prefill_window_tokens =
            firstIntEnvOrDefault(
                {"LLAMINAR_QWEN36_MOE_OVERLAY_RECALL_PREFILL_WINDOW_TOKENS"},
                probe.prefill_window_tokens);
        config.moe_rebalance.dynamic_imbalance_threshold_per_mille = 0;
        config.moe_rebalance.dynamic_min_improvement_per_mille = 0;
        config.moe_rebalance.dynamic_max_swaps_per_layer = 20;
        config.moe_rebalance.dynamic_max_plan_entries_per_wave = 20;
        config.moe_rebalance.dynamic_min_window_activations = 0;
        config.moe_rebalance.device_min_load_spread_improvement = 0;
        config.moe_rebalance.device_min_load_spread_improvement_divisor = 0;
        config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_min_foreign_rows_per_transfer = 0;
        config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot = 0;
        config.moe_rebalance.device_max_post_wave_load_spread_per_mille = 1000;
        config.moe_rebalance.release_raw_expert_weights = true;
        return config;
    };

    auto run_overlay_probe = [&](const OverlayRecallProbeConfig &probe)
    {
        auto runner = factory->createFromOrchestrationConfig(make_overlay_config(probe));
        EXPECT_NE(runner, nullptr);
        if (!runner)
        {
            NeedleRecallRunResult result;
            result.placement = probe.label;
            result.error = "failed to create runner";
            return result;
        }
        NeedleRecallRunResult result;
        if (runner->initialize())
        {
            result = runNeedleRecallPrompt(
                *runner,
                "beginning",
                context_length,
                max_tokens);
            result.placement = probe.label;
        }
        else
        {
            result.placement = probe.label;
            result.error = runner->lastError();
        }
        runner->shutdown();
        return result;
    };

    OverlayRecallProbeConfig static_probe;
    static_probe.label = "static-overlay";
    static_probe.mode = MoERebalanceRuntimeMode::Off;
    static_probe.routed_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    static_probe.prefix_cache = true;
    static_probe.prefill_window_tokens = 0;
    static_probe.require_transfer_backing = false;

    OverlayRecallProbeConfig llep_probe;
    llep_probe.label = "llep-overlay-prefix-window";
    llep_probe.mode = MoERebalanceRuntimeMode::LLEP;
    llep_probe.routed_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    llep_probe.prefix_cache = true;
    llep_probe.prefill_window_tokens = 0;
    llep_probe.require_transfer_backing = true;

    NeedleRecallRunResult cuda2_result;
    {
        auto runner = factory->createFromOrchestrationConfig(make_cuda2_tp_config());
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        cuda2_result = runNeedleRecallPrompt(
            *runner,
            "beginning",
            context_length,
            max_tokens);
        runner->shutdown();
    }
    if (!cuda2_result.error.empty())
    {
        FAIL()
            << "Ordinary CUDA2 TP failed before the overlay comparison. "
            << summarizeNeedleRecallResult(cuda2_result);
    }
    if (!cuda2_result.containsTarget())
    {
        GTEST_SKIP()
            << "Ordinary CUDA2 TP did not satisfy the guarded needle prompt; "
            << "not treating it as an overlay-specific regression. "
            << summarizeNeedleRecallResult(cuda2_result);
    }

    const NeedleRecallRunResult static_overlay_result =
        run_overlay_probe(static_probe);
    ASSERT_TRUE(static_overlay_result.containsTarget())
        << "Qwen3.6 MoE ExpertOverlay CUDA2TP static-owner guard failed; "
        << "the regression is not isolated to LLEP.\nstandard: "
        << summarizeNeedleRecallResult(cuda2_result)
        << "\nstatic-overlay: "
        << summarizeNeedleRecallResult(static_overlay_result);

    const NeedleRecallRunResult overlay_result =
        run_overlay_probe(llep_probe);
    llaminar::v2::kernels::KernelFactory::clearCache();

    EXPECT_TRUE(overlay_result.containsTarget())
        << "Qwen3.6 MoE ExpertOverlay CUDA2TP LLEP lost a long-context needle "
        << "that ordinary CUDA2 TP recalled.\nstandard: "
        << summarizeNeedleRecallResult(cuda2_result)
        << "\nstatic-overlay: "
        << summarizeNeedleRecallResult(static_overlay_result)
        << "\noverlay: "
        << summarizeNeedleRecallResult(overlay_result);
}

TEST(Test__KVPrefixMTPStateProbe, MTP_ShiftedCacheCountProbeOnGPU)
{
    const auto device = firstGpuDeviceId();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for MTP shifted-cache probe";
    }

    TinyQwenForwardFixture fixture(*device);
    auto graph_builder = std::make_shared<QwenStandardGraph>(fixture.config, fixture.mpi);
    DeviceGraphOrchestrator orchestrator(graph_builder, fixture.mpi);

    ASSERT_TRUE(orchestrator.initializeInferenceStateFromArena(
        /*batch_size=*/1,
        fixture.config.max_seq_len,
        *device));
    orchestrator.setWeights(fixture.modelWeights());
    PreparedWeightStore prepared_store;
    ASSERT_NO_THROW(prepareDenseForwardWeights(orchestrator, *graph_builder, prepared_store, *device));

    const std::vector<int> prefix_tokens = {1, 2, 3, 4};
    ASSERT_NE(orchestrator.forward(prefix_tokens.data(), static_cast<int>(prefix_tokens.size()), 1), nullptr);

    const auto after_prefill = orchestrator.prefixStateProbe();
    EXPECT_TRUE(after_prefill.initialized);
    EXPECT_TRUE(after_prefill.primary_device.is_gpu());
    EXPECT_EQ(after_prefill.current_position, static_cast<int>(prefix_tokens.size()));
    EXPECT_EQ(maxLayerCachedTokens(after_prefill), static_cast<int>(prefix_tokens.size()));
    ASSERT_EQ(after_prefill.mtp_kv_caches.size(), 1u);
    EXPECT_EQ(after_prefill.mtp_kv_caches[0].owner, "mtp:0");
    EXPECT_EQ(after_prefill.mtp_kv_caches[0].n_layers, 1);
    EXPECT_EQ(maxCachedTokensIn(after_prefill.mtp_kv_caches), static_cast<int>(prefix_tokens.size()) - 1);
    EXPECT_EQ(after_prefill.totalMTPCachedTokens(), static_cast<int>(prefix_tokens.size()) - 1);

    orchestrator.clear_cache();
    const auto after_clear = orchestrator.prefixStateProbe();
    EXPECT_EQ(maxCachedTokensIn(after_clear.mtp_kv_caches), 0);
    EXPECT_EQ(after_clear.totalMTPCachedTokens(), 0);
}

TEST(Test__KVPrefixMTPStateProbe, MTP_SidecarOneTokenExecutesOnGPU)
{
    const auto device = firstGpuDeviceId();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA or ROCm GPU available for MTP sidecar smoke test";
    }

    TinyMTPSidecarFixture fixture(*device);
    ASSERT_NE(fixture.kv_cache, nullptr);

    PreparedWeightStore prepared_store;
    ASSERT_NO_THROW(fixture.prepareWeights(prepared_store));

    Qwen35Graph graph_builder(fixture.config, fixture.mpi);
    graph_builder.setWeightBindings(fixture.bindings);
    graph_builder.setWeights(toLegacyModelWeights(fixture.bindings));
    graph_builder.setPreparedWeightStore(&prepared_store);

    auto input = fixture.input();
    auto output = fixture.output();
    ComputeGraph graph = graph_builder.buildMTPGraph(
        /*depth_idx=*/0,
        fixture.mtp_depth_bindings,
        input,
        output);
    ASSERT_GT(graph.size(), 0u);
    EXPECT_EQ(graph.terminalNode(), "mtp0_lm_head");

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = nullptr;
    if (device->is_cuda())
    {
        ctx = &pool.getNvidiaContext(device->cuda_ordinal());
    }
    else if (device->is_rocm())
    {
        ctx = &pool.getAMDContext(device->rocm_ordinal());
    }
    ASSERT_NE(ctx, nullptr);
    auto device_ctx = IDeviceContext::create(*device, 1);
    ASSERT_NE(device_ctx, nullptr);

    WorkspaceSizingHints workspace_hints;
    workspace_hints.max_seq_len = input.seq_len;
    workspace_hints.n_heads = fixture.config.n_heads;
    workspace_hints.head_dim = fixture.config.head_dim;
    workspace_hints.d_model = fixture.config.d_model;
    workspace_hints.batch_size = input.batch_size;
    workspace_hints.vocab_size = fixture.config.vocab_size;
    WorkspaceAllocator workspace_allocator;
    ASSERT_TRUE(workspace_allocator.allocateForGraph(graph, workspace_hints));

    DeviceGraphExecutor executor;
    bool executed = false;
    ctx->submitAndWait([&]
                       { executed = executor.executeFastDecode(graph, device_ctx.get()); });
    ASSERT_TRUE(executed);

    ASSERT_TRUE(fixture.hidden->ensureOnHost());
    const float *hidden = fixture.hidden->fp32_data();
    ASSERT_NE(hidden, nullptr);
    float hidden_abs_sum = 0.0f;
    for (size_t i = 0; i < fixture.hidden->numel(); ++i)
    {
        ASSERT_TRUE(std::isfinite(hidden[i])) << "non-finite MTP hidden value at index " << i;
        hidden_abs_sum += std::abs(hidden[i]);
    }
    EXPECT_GT(hidden_abs_sum, 0.0f);

    ASSERT_TRUE(fixture.logits->ensureOnHost());
    const float *logits = fixture.logits->fp32_data();
    ASSERT_NE(logits, nullptr);
    float abs_sum = 0.0f;
    for (size_t i = 0; i < fixture.logits->numel(); ++i)
    {
        ASSERT_TRUE(std::isfinite(logits[i])) << "non-finite MTP logit at index " << i;
        abs_sum += std::abs(logits[i]);
    }
    EXPECT_GT(abs_sum, 0.0f);
    EXPECT_EQ(fixture.kv_cache->get_cached_tokens(/*layer=*/0, /*seq_idx=*/0), 1);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
