/**
 * @file Qwen36DenseParityTestBase.h
 * @brief Real-weight Qwen3.6 dense parity, prefix-cache, and MTP test support.
 *
 * The helpers in this file keep the live production runner as the system under
 * test while authenticating every reusable CPU/FP32 Hugging Face reference
 * pack against the declared GGUF descriptor, prompt, and tokenization.
 * Checkpoint campaigns
 * retain captured graph, device-state, depth-policy, and CSV evidence instead
 * of reducing correctness to final-token agreement.
 */

#pragma once

#include "../ParityTestBase.h"
#include "Qwen36MTPCheckpointSurface.h"

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
#include "execution/mtp/MTPSpecDecodeMetadata.h"
#include "execution/mtp/MTPStateTransaction.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/runner/ModelContextRetirement.h"
#include "kernels/KernelFactory.h"
#include "loaders/ModelContext.h"
#include "utils/DebugEnv.h"
#include "utils/MTPParitySnapshotContext.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Sampler.h"
#include "utils/Sha256.h"
#include "utils/Tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace llaminar2::test::parity::qwen36
{
    enum class DensePrefixParityTopology
    {
        SingleDevice,
        LocalTP,
        LocalPP,
        NodeTP,
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

    inline bool densePhase138PromotedTransactionExpected(
        const DensePrefixRestoreParityCase &test_case)
    {
        (void)test_case;
        return false;
    }

    inline bool denseCaseExpectsAllPositionSpecPublication(
        const DensePrefixRestoreParityCase &test_case)
    {
        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();
        /*
         * Qwen3.6 dense still carries hybrid/GDN live state. Phase 9.7 proves
         * the shared decode-equivalent verifier rows first; direct all-position
         * publication is a stronger live-state contract and stays fail-closed
         * until dense continuation-equivalence promotes a backend explicitly.
         */
        (void)device;
        return false;
    }

    /**
     * @brief Return whether dense MTP should use device-resident publication.
     *
     * Single-device GPU runners own every speculative slot locally.  GPU
     * LocalTP runners now reduce verifier outcomes at RankOrchestrator scope
     * and then fan the same compact accepted-state transaction to every child.
     * CPU runners and distributed runners without that mailbox remain on
     * explicit decode-equivalent replay until they have their own resident
     * publication proof.
     *
     * @param test_case Dense parity fixture under test.
     * @return true when the fixture should exercise device-resident MTP state
     *         publication.
     */
    inline bool denseCaseExpectsGroupedDevicePublication(
        const DensePrefixRestoreParityCase &test_case)
    {
        if (test_case.devices.empty())
            return false;

        const bool all_gpu =
            std::all_of(
                test_case.devices.begin(),
                test_case.devices.end(),
                [](const GlobalDeviceAddress &device)
                {
                    return device.isGPU();
                });
        return all_gpu &&
               (test_case.topology == DensePrefixParityTopology::SingleDevice ||
                test_case.topology == DensePrefixParityTopology::LocalTP);
    }

    inline bool denseHasPerfCounter(
        const std::vector<PerfStatRecord> &records,
        const char *domain,
        const char *name)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.kind == PerfStatRecord::Kind::Counter &&
                       record.domain == domain &&
                       record.name == name &&
                       record.value > 0.0;
            });
    }

    inline bool denseHasPerfRecordTag(
        const std::vector<PerfStatRecord> &records,
        const char *domain,
        const char *name,
        const char *tag_key,
        const char *tag_value)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                if (record.domain != domain || record.name != name)
                    return false;
                const auto it = record.tags.find(tag_key);
                return it != record.tags.end() && it->second == tag_value;
            });
    }

    /**
     * @brief Return whether a positive counter carries an exact metadata tag.
     *
     * Graph-capture assertions must distinguish evidence emitted by the main
     * decode graph from evidence emitted by the MTP sidecar graph. Merely
     * finding a tagged zero-valued planning record would not prove execution,
     * so this helper requires both the matching tag and positive counter value.
     */
    inline bool denseHasPositivePerfCounterTag(
        const std::vector<PerfStatRecord> &records,
        const char *domain,
        const char *name,
        const char *tag_key,
        const char *tag_value)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != domain ||
                    record.name != name ||
                    record.value <= 0.0)
                {
                    return false;
                }
                const auto it = record.tags.find(tag_key);
                return it != record.tags.end() && it->second == tag_value;
            });
    }

    /**
     * @brief Return whether a positive counter tag starts with a stable prefix.
     *
     * MTP sidecar graph contexts include resident-input policy suffixes in
     * their cache identity. Tests care that the sidecar graph family replayed,
     * while preserving those suffixes as useful diagnostics.
     */
    inline bool denseHasPositivePerfCounterTagPrefix(
        const std::vector<PerfStatRecord> &records,
        const char *domain,
        const char *name,
        const char *tag_key,
        const char *tag_prefix)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != domain ||
                    record.name != name ||
                    record.value <= 0.0)
                {
                    return false;
                }
                const auto it = record.tags.find(tag_key);
                return it != record.tags.end() &&
                       it->second.rfind(tag_prefix, 0) == 0;
            });
    }

    inline bool denseHasMTPPerfCounter(
        const std::vector<PerfStatRecord> &records,
        const char *name)
    {
        return denseHasPerfCounter(records, "mtp", name);
    }

    inline bool denseHasMTPPerfRecordTag(
        const std::vector<PerfStatRecord> &records,
        const char *name,
        const char *tag_key,
        const char *tag_value)
    {
        return denseHasPerfRecordTag(
            records,
            "mtp",
            name,
            tag_key,
            tag_value);
    }

    /**
     * @brief Format every tagged instance of one PerfStats counter.
     *
     * Stochastic MTP regressions need to compare the exact transaction chosen
     * during first capture with the transaction chosen during graph reuse.
     * PerfStats is deliberately reset between those requests, so formatting
     * the saved snapshots is more reliable than querying the live collector in
     * a later assertion message.
     */
    inline std::string densePerfCounterTagSummary(
        const std::vector<PerfStatRecord> &records,
        const char *domain,
        const char *name)
    {
        std::ostringstream out;
        bool found = false;
        for (const PerfStatRecord &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != domain ||
                record.name != name)
            {
                continue;
            }

            if (found)
                out << '\n';
            found = true;
            out << domain << '.' << name << '{';
            bool first_tag = true;
            for (const auto &[key, value] : record.tags)
            {
                if (!first_tag)
                    out << ',';
                first_tag = false;
                out << key << '=' << value;
            }
            out << '}';
        }
        return found ? out.str() : std::string("<no matching records>");
    }

    /**
     * @brief Return whether every participant uses one homogeneous GPU backend.
     *
     * CUDA-only and ROCm-only graphs can capture their backend-native
     * collectives. A mixed CUDA/ROCm topology is intentionally excluded because
     * its cross-backend handoff may require explicit execution segments.
     */
    inline bool denseCaseUsesHomogeneousGPU(
        const DensePrefixRestoreParityCase &test_case)
    {
        return classifyProductionParityExecutionTopology(test_case.devices) ==
               ProductionParityExecutionTopology::HomogeneousGPU;
    }

    /**
     * @brief Assert that a dense greedy MTP request used the expected verifier lane.
     *
     * Token equality proves the visible response, but not the performance path.
     * This guard makes the parity matrix fail if a GPU LocalTP, single-device,
     * CPU, or NodeTP request silently drifts away from the grouped
     * decode-equivalent verifier rows after those rows have already been proven
     * correct.
     *
     * @param test_case Dense parity fixture under test.
     * @param records Perfstats snapshot captured immediately after the request.
     * @param context Human-readable request label for assertion failures.
     */
    inline void expectDenseGreedyMTPPublicationPath(
        const DensePrefixRestoreParityCase &test_case,
        const std::vector<PerfStatRecord> &records,
        const std::string &context)
    {
        const bool used_retired_serial_replay =
            denseHasMTPPerfCounter(
                records,
                "decode_equivalent_sequential_verifier_runs");
        const bool used_grouped_host_publication =
            denseHasMTPPerfCounter(
                records,
                "grouped_outcome_host_publication_uses") &&
            denseHasMTPPerfCounter(
                records,
                "grouped_outcome_host_state_publications");
        const bool used_grouped_device_publication =
            denseHasMTPPerfCounter(
                records,
                "grouped_outcome_device_resident_publication_uses") &&
            (denseHasMTPPerfCounter(records, "spec_state_publications") ||
             denseHasMTPPerfCounter(records, "spec_state_batch_publications") ||
             denseHasMTPPerfCounter(
                 records,
                 "rank_grouped_decode_equivalent_spec_state_batch_publications"));
        const bool used_direct_all_position_publication =
            denseHasMTPPerfCounter(
                records,
                "all_position_state_publication_verifier_runs");
        const bool used_grouped_verifier =
            denseHasMTPPerfCounter(
                records,
                "grouped_decode_equivalent_greedy_verifier_runs");
        const bool used_rank_grouped_multi_stream_publication =
            denseHasPerfCounter(
                records,
                "tp_collective_runtime",
                "sideband_only_multi_stream_groups");

        if (denseCaseExpectsGroupedDevicePublication(test_case))
        {
            EXPECT_TRUE(used_grouped_device_publication)
                << context << " should publish grouped verifier rows through "
                   "the device-resident mailbox.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_TRUE(used_grouped_verifier)
                << context << " should run the grouped greedy verifier rows.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_retired_serial_replay)
                << context << " must not use row-serial verifier replay when "
                   "device-resident publication is available.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_direct_all_position_publication)
                << context << " must not promote dense direct all-position "
                   "publication without a continuation proof.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            if (test_case.topology == DensePrefixParityTopology::LocalTP)
            {
                EXPECT_TRUE(used_rank_grouped_multi_stream_publication)
                    << context << " must publish the compact mirrored outcome "
                       "through one rank-level NCCL/RCCL multi-stream group.\n"
                    << PerfStatsCollector::summaryString(
                           {"mtp", "tp_collective_runtime"});
                EXPECT_TRUE(denseHasPerfRecordTag(
                    records,
                    "tp_collective_runtime",
                    "sideband_only_multi_stream_groups",
                    "host_rendezvous",
                    "false"))
                    << context << " must not rendezvous host workers for compact "
                       "outcome publication.";
                EXPECT_TRUE(denseHasPerfRecordTag(
                    records,
                    "tp_collective_runtime",
                    "sideband_only_multi_stream_groups",
                    "device_completion_wait",
                    "false"))
                    << context << " must leave collective completion ordered by "
                       "the participant device streams.";
            }
            return;
        }

        EXPECT_TRUE(used_grouped_verifier)
            << context << " should run grouped decode-equivalent verifier rows.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_TRUE(used_grouped_device_publication || used_grouped_host_publication)
            << context << " should publish grouped verifier state through an "
               "explicit grouped publication path.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(used_retired_serial_replay)
            << context << " must not use the retired row-serial verifier replay.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(used_direct_all_position_publication)
            << context << " must not use unproven dense direct all-position "
                   "publication.\n"
                << PerfStatsCollector::summaryString({"mtp"});
    }

    /**
     * @brief Prove homogeneous GPU MTP executed whole captured graphs.
     *
     * CUDA-only and ROCm-only serving cells have graph-capturable collectives,
     * so both the ordinary decode graph and the MTP sidecar graph must replay as
     * one native graph each. Segmented plans, segmented capture executables, and
     * segmented replay are all architectural failures for these topologies.
     *
     * Heterogeneous device mixes are intentionally outside this assertion:
     * crossing backend boundaries can require explicit collective segments.
     *
     * @param test_case Dense parity fixture under test.
     * @param records Request-local PerfStats evidence.
     * @param context Human-readable request label for assertion failures.
     */
    inline void expectDenseHomogeneousGPUFullGraphReplay(
        const DensePrefixRestoreParityCase &test_case,
        const std::vector<PerfStatRecord> &records,
        const std::string &context)
    {
        if (!denseCaseUsesHomogeneousGPU(test_case))
        {
            return;
        }

        const bool planned_capturable_full_graph =
            denseHasPositivePerfCounterTag(
                records,
                "forward_graph",
                "full_graph_plan_graphs",
                "type",
                "capturable");
        const bool captured_full_graph = denseHasPerfCounter(
            records,
            "forward_graph",
            "full_graph_capture_executable_nodes");
        const bool replayed_full_graph = denseHasPerfCounter(
            records,
            "forward_graph",
            "full_graph_replay_calls");
        EXPECT_TRUE(
            planned_capturable_full_graph ||
            captured_full_graph ||
            replayed_full_graph)
            << context << " provided no proof of a whole-graph capture plan, "
                          "executable, or replay. A reused graph does not rerun "
                          "planning merely to recreate PerfStats evidence.\n"
            << PerfStatsCollector::summaryString({"forward_graph"});
        EXPECT_TRUE(captured_full_graph || replayed_full_graph)
            << context << " neither instantiated nor replayed a whole native graph.\n"
            << PerfStatsCollector::summaryString({"forward_graph"});
        const bool captured_full_sidecar =
            denseHasPositivePerfCounterTagPrefix(
                records,
                "forward_graph",
                "full_graph_capture_executable_nodes",
                "context",
                "mtp_decode_sidecar");
        const bool replayed_full_sidecar =
            denseHasPositivePerfCounterTagPrefix(
                records,
                "forward_graph",
                "full_graph_replay_calls",
                "context",
                "mtp_decode_sidecar");
        EXPECT_TRUE(captured_full_sidecar || replayed_full_sidecar)
            << context << " neither captured nor replayed the MTP sidecar as "
                          "one native graph.\n"
            << PerfStatsCollector::summaryString({"forward_graph"});

        EXPECT_FALSE(denseHasPerfCounter(
            records,
            "forward_graph",
            "segmented_plan_segments"))
            << context << " produced a segmented graph plan on homogeneous GPUs.\n"
            << PerfStatsCollector::summaryString({"forward_graph"});
        EXPECT_FALSE(denseHasPerfCounter(
            records,
            "forward_graph",
            "segmented_graph_capture_executable_nodes"))
            << context << " instantiated a segmented graph on homogeneous GPUs.\n"
            << PerfStatsCollector::summaryString({"forward_graph"});
        EXPECT_FALSE(denseHasPerfCounter(
            records,
            "forward_graph",
            "segmented_replay_segments"))
            << context << " executed segmented replay on homogeneous GPUs.\n"
            << PerfStatsCollector::summaryString({"forward_graph"});
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

    class ScopedDenseParityProductionMode
    {
    public:
        explicit ScopedDenseParityProductionMode(bool enabled)
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

            setenv("LLAMINAR_DETERMINISTIC", "0", 1);
            mutableDebugEnv().reload();
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ~ScopedDenseParityProductionMode()
        {
            if (!enabled_)
            {
                return;
            }

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

        ScopedDenseParityProductionMode(const ScopedDenseParityProductionMode &) = delete;
        ScopedDenseParityProductionMode &operator=(const ScopedDenseParityProductionMode &) = delete;

    private:
        bool enabled_ = false;
        bool had_old_deterministic_env_ = false;
        std::string old_deterministic_env_;
    };

    inline bool isDenseGpuParityCase(
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

    inline bool shouldForceDenseParityProductionMode(
        const DensePrefixRestoreParityCase &)
    {
        return true;
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

    /**
     * @brief Resolve the first configured model path through campaign tmpfs.
     * @param names Ordered environment-variable overrides for the model.
     * @param fallback Test-owned default GGUF path.
     * @return The staged path in an aggregate campaign, otherwise the selected
     *         source path.
     * @throws std::runtime_error when aggregate staging did not declare or
     *         publish the model selected by the concrete test.
     */
    inline std::string firstModelEnvOrDefault(
        const std::vector<std::string> &names,
        const std::string &fallback)
    {
        return productionParityResolvedModelPath(
            firstEnvOrDefault(names, fallback));
    }

    /**
     * @brief Return true when a parity case will stage weights for a GPU backend.
     *
     * The distinction matters before any runner is built: GPU model loads should
     * use the same demand-paged mmap policy as production so large GGUF files do
     * not block on a whole-file host prefault before device upload starts.
     */
    inline bool isQwen36GpuParityDevice(const DeviceId &device)
    {
        return device.is_gpu();
    }

    /**
     * @brief Create a Qwen 3.6 real-model parity context with backend-aware mmap.
     *
     * Most parity helpers build runners directly rather than through
     * OrchestrationRunner, so they must explicitly pass ModelContextConfig here.
     * This keeps GPU tests on the production GPU-target mmap path while leaving
     * CPU tests on the NUMA/eager-prefault path used for CPU decode.
     */
    inline std::shared_ptr<ModelContext> createQwen36ParityModelContext(
        const std::string &model_path,
        const DeviceId &device,
        WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED,
        WeightPrecision weight_precision = WeightPrecision::NATIVE)
    {
        ModelContextConfig model_config;
        model_config.strategy = strategy;
        model_config.weight_precision = weight_precision;
        model_config.use_mmap = true;
        model_config.payload_access_pattern =
            isQwen36GpuParityDevice(device)
                ? ModelPayloadAccessPattern::DeviceStaging
                : ModelPayloadAccessPattern::DenseCpuResident;
        return ModelContext::create(model_path, model_config);
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

    struct DenseVerifierLogitMetrics
    {
        double cosine = 1.0;
        double rel_l2 = 0.0;
        double max_abs_diff = 0.0;
        size_t max_abs_index = 0;
        double symmetric_kl = 0.0;
    };

    inline DenseVerifierLogitMetrics computeDenseVerifierLogitMetrics(
        const float *actual_logits,
        const float *serial_logits,
        int vocab_size)
    {
        DenseVerifierLogitMetrics metrics;
        if (!actual_logits || !serial_logits || vocab_size <= 0)
        {
            metrics.cosine = 0.0;
            metrics.rel_l2 = std::numeric_limits<double>::infinity();
            metrics.max_abs_diff = std::numeric_limits<double>::infinity();
            metrics.symmetric_kl = std::numeric_limits<double>::infinity();
            return metrics;
        }

        double dot = 0.0;
        double actual_norm = 0.0;
        double serial_norm = 0.0;
        double diff_norm = 0.0;
        float actual_max = actual_logits[0];
        float serial_max = serial_logits[0];
        for (int i = 0; i < vocab_size; ++i)
        {
            actual_max = std::max(actual_max, actual_logits[i]);
            serial_max = std::max(serial_max, serial_logits[i]);
            const double actual = static_cast<double>(actual_logits[i]);
            const double serial = static_cast<double>(serial_logits[i]);
            const double diff = actual - serial;
            dot += actual * serial;
            actual_norm += actual * actual;
            serial_norm += serial * serial;
            diff_norm += diff * diff;
            const double abs_diff = std::abs(diff);
            if (abs_diff > metrics.max_abs_diff)
            {
                metrics.max_abs_diff = abs_diff;
                metrics.max_abs_index = static_cast<size_t>(i);
            }
        }

        const double denom = std::sqrt(actual_norm * serial_norm);
        metrics.cosine = denom > 0.0 ? dot / denom : 1.0;
        metrics.rel_l2 = serial_norm > 0.0 ? std::sqrt(diff_norm / serial_norm)
                                           : std::sqrt(diff_norm);

        std::vector<double> actual_probs(static_cast<size_t>(vocab_size), 0.0);
        std::vector<double> serial_probs(static_cast<size_t>(vocab_size), 0.0);
        double actual_sum = 0.0;
        double serial_sum = 0.0;
        for (int i = 0; i < vocab_size; ++i)
        {
            const double actual =
                std::exp(static_cast<double>(actual_logits[i] - actual_max));
            const double serial =
                std::exp(static_cast<double>(serial_logits[i] - serial_max));
            actual_probs[static_cast<size_t>(i)] = actual;
            serial_probs[static_cast<size_t>(i)] = serial;
            actual_sum += actual;
            serial_sum += serial;
        }

        constexpr double probability_floor = 1.0e-300;
        double actual_to_serial = 0.0;
        double serial_to_actual = 0.0;
        for (int i = 0; i < vocab_size; ++i)
        {
            const double p = std::max(
                actual_probs[static_cast<size_t>(i)] / actual_sum,
                probability_floor);
            const double q = std::max(
                serial_probs[static_cast<size_t>(i)] / serial_sum,
                probability_floor);
            actual_to_serial += p * std::log(p / q);
            serial_to_actual += q * std::log(q / p);
        }
        metrics.symmetric_kl = 0.5 * (actual_to_serial + serial_to_actual);
        return metrics;
    }

    inline ::testing::AssertionResult denseVerifierLogitsNumericallyEquivalent(
        const float *actual_logits,
        const float *serial_logits,
        int vocab_size,
        const std::string &label,
        double min_cosine = 0.99995,
        double max_rel_l2 = 0.005,
        double max_symmetric_kl = 1.0e-4,
        double max_abs_diff = 0.25)
    {
        const DenseVerifierLogitMetrics metrics =
            computeDenseVerifierLogitMetrics(
                actual_logits,
                serial_logits,
                vocab_size);
        if (metrics.cosine >= min_cosine &&
            metrics.rel_l2 <= max_rel_l2 &&
            metrics.symmetric_kl <= max_symmetric_kl &&
            metrics.max_abs_diff <= max_abs_diff)
        {
            return ::testing::AssertionSuccess();
        }

        return ::testing::AssertionFailure()
               << label
               << " logit distribution drift: cosine=" << metrics.cosine
               << " rel_l2=" << metrics.rel_l2
               << " symmetric_kl=" << metrics.symmetric_kl
               << " max_abs_diff=" << metrics.max_abs_diff
               << " max_abs_index=" << metrics.max_abs_index
               << " thresholds(cosine>=" << min_cosine
               << ", rel_l2<=" << max_rel_l2
               << ", symmetric_kl<=" << max_symmetric_kl
               << ", max_abs_diff<=" << max_abs_diff << ")";
    }

    /**
     * @brief Require finite native logits identical to the ordinary serial row.
     * @param actual_logits Grouped verifier's selected vocabulary row.
     * @param serial_logits Matching M=1 production row.
     * @param vocab_size Positive full-row width.
     * @param label Checkpoint identity included in failure diagnostics.
     * @return Exact-byte success, or actionable finite/geometry/bit failure.
     */
    inline ::testing::AssertionResult denseVerifierLogitsByteIdentical(
        const float *actual_logits,
        const float *serial_logits,
        int vocab_size,
        const std::string &label)
    {
        if (!actual_logits || !serial_logits || vocab_size <= 0)
            return ::testing::AssertionFailure() << label << " missing native verifier row";
        const size_t count = static_cast<size_t>(vocab_size);
        const auto byte_evidence = compareNativeVerifierRow(
            {actual_logits, count}, {serial_logits, count});
        if (byte_evidence.passed())
        {
            return ::testing::AssertionSuccess();
        }

        const size_t first_mismatch = byte_evidence.first_mismatch;

        uint32_t actual_bits = 0;
        uint32_t serial_bits = 0;
        if (first_mismatch < count)
        {
            std::memcpy(&actual_bits,
                        actual_logits + first_mismatch,
                        sizeof(actual_bits));
            std::memcpy(&serial_bits,
                        serial_logits + first_mismatch,
                        sizeof(serial_bits));
        }

        const DenseVerifierLogitMetrics metrics =
            computeDenseVerifierLogitMetrics(
                actual_logits,
                serial_logits,
                vocab_size);
        return ::testing::AssertionFailure()
               << label
               << " must be byte-identical to serial decode"
               << " first_mismatch=" << first_mismatch
               << " actual=" << (first_mismatch < count ? actual_logits[first_mismatch] : 0.0f)
               << " serial=" << (first_mismatch < count ? serial_logits[first_mismatch] : 0.0f)
               << " actual_bits=0x" << std::hex << actual_bits
               << " serial_bits=0x" << serial_bits << std::dec
               << " cosine=" << metrics.cosine
               << " rel_l2=" << metrics.rel_l2
               << " symmetric_kl=" << metrics.symmetric_kl
               << " max_abs_diff=" << metrics.max_abs_diff
               << " max_abs_index=" << metrics.max_abs_index;
    }

    /**
     * @brief Prove grouped verifier recurrent state is byte-identical to serial decode.
     *
     * Logit equality is necessary but not sufficient for MTP publication:
     * accepted verifier rows also publish GDN recurrence and short-conv state
     * that the next ordinary decode step consumes.  CPU stores that state in
     * host vectors, while CUDA/ROCm own the production state in backend kernel
     * buffers.  This helper therefore treats GPU device hashes as mandatory
     * whenever either side exposes them, and only falls back to host-vector
     * hashes for pure CPU probes.
     */
    inline ::testing::AssertionResult verifierGDNStateByteIdentical(
        const PrefixRuntimeStateSnapshot &grouped,
        const PrefixRuntimeStateSnapshot &serial,
        const std::string &label)
    {
        if (grouped.gdn_layers.size() != serial.gdn_layers.size())
        {
            return ::testing::AssertionFailure()
                   << label << " GDN layer count mismatch: grouped="
                   << grouped.gdn_layers.size()
                   << " serial=" << serial.gdn_layers.size();
        }

        auto boolString = [](bool value) -> const char *
        {
            return value ? "true" : "false";
        };

        for (size_t i = 0; i < grouped.gdn_layers.size(); ++i)
        {
            const PrefixGDNLayerProbe &g = grouped.gdn_layers[i];
            const PrefixGDNLayerProbe &s = serial.gdn_layers[i];
            const bool compare_device =
                g.device_state_hash_available ||
                s.device_state_hash_available;
            const bool compare_local_device =
                g.local_device_state_hash_available ||
                s.local_device_state_hash_available;

            if (g.global_layer != s.global_layer)
            {
                return ::testing::AssertionFailure()
                       << label << " GDN layer index mismatch at ordinal " << i
                       << ": grouped=L" << g.global_layer
                       << " serial=L" << s.global_layer;
            }

            if (compare_device)
            {
                if (!g.device_state_hash_available ||
                    !s.device_state_hash_available ||
                    g.recurrence_device_bytes != s.recurrence_device_bytes ||
                    g.conv_device_bytes != s.conv_device_bytes ||
                    g.recurrence_device_hash != s.recurrence_device_hash ||
                    g.conv_device_hash != s.conv_device_hash)
                {
                    return ::testing::AssertionFailure()
                           << label << " device GDN state mismatch at L"
                           << g.global_layer
                           << ": grouped_available="
                           << boolString(g.device_state_hash_available)
                           << " serial_available="
                           << boolString(s.device_state_hash_available)
                           << " grouped_rec_bytes="
                           << g.recurrence_device_bytes
                           << " serial_rec_bytes="
                           << s.recurrence_device_bytes
                           << " grouped_conv_bytes=" << g.conv_device_bytes
                           << " serial_conv_bytes=" << s.conv_device_bytes
                           << " grouped_rec_hash="
                           << g.recurrence_device_hash
                           << " serial_rec_hash="
                           << s.recurrence_device_hash
                           << " grouped_conv_hash=" << g.conv_device_hash
                           << " serial_conv_hash=" << s.conv_device_hash;
                }
            }
            else if (g.recurrence_values != s.recurrence_values ||
                     g.conv_values != s.conv_values ||
                     g.recurrence_hash != s.recurrence_hash ||
                     g.conv_hash != s.conv_hash)
            {
                return ::testing::AssertionFailure()
                       << label << " host GDN state mismatch at L"
                       << g.global_layer
                       << ": grouped_rec_values=" << g.recurrence_values
                       << " serial_rec_values=" << s.recurrence_values
                       << " grouped_conv_values=" << g.conv_values
                       << " serial_conv_values=" << s.conv_values
                       << " grouped_rec_hash=" << g.recurrence_hash
                       << " serial_rec_hash=" << s.recurrence_hash
                       << " grouped_conv_hash=" << g.conv_hash
                       << " serial_conv_hash=" << s.conv_hash;
            }

            if (compare_local_device &&
                (!g.local_device_state_hash_available ||
                 !s.local_device_state_hash_available ||
                 g.recurrence_local_device_bytes !=
                     s.recurrence_local_device_bytes ||
                 g.conv_local_device_bytes != s.conv_local_device_bytes ||
                 g.recurrence_local_device_hash !=
                     s.recurrence_local_device_hash ||
                 g.conv_local_device_hash != s.conv_local_device_hash))
            {
                return ::testing::AssertionFailure()
                       << label << " local-device GDN state mismatch at L"
                       << g.global_layer
                       << ": grouped_available="
                       << boolString(g.local_device_state_hash_available)
                       << " serial_available="
                       << boolString(s.local_device_state_hash_available)
                       << " grouped_rec_bytes="
                       << g.recurrence_local_device_bytes
                       << " serial_rec_bytes="
                       << s.recurrence_local_device_bytes
                       << " grouped_conv_bytes="
                       << g.conv_local_device_bytes
                       << " serial_conv_bytes="
                       << s.conv_local_device_bytes
                       << " grouped_rec_hash="
                       << g.recurrence_local_device_hash
                       << " serial_rec_hash="
                       << s.recurrence_local_device_hash
                       << " grouped_conv_hash="
                       << g.conv_local_device_hash
                       << " serial_conv_hash="
                       << s.conv_local_device_hash;
            }
        }

        return ::testing::AssertionSuccess();
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

    /**
     * @brief Controls how the Phase 13.8 continuation diagnostic compares
     *        Llaminar stage snapshots to PyTorch reference snapshots.
     *
     * CPU decode uses the same FP32 scalar path as the reference closely enough
     * that all layers can stay strict.  GPU decode intentionally uses quantized
     * native GEMV kernels, so this policy lets the Phase 13.8 diagnostic focus
     * on early state/coherence regressions while the classic Qwen3.6 parity
     * suite owns full-output KLD, top-k, and all-layer math acceptance.
     */
    struct DenseDecodeSnapshotComparisonPolicy
    {
        int layer_count = 0;
        int max_layer_count = -1;
        bool compare_final_outputs = true;
        float cosine_threshold = 0.995f;
        float rel_l2_threshold = 0.25f;
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

    /**
     * @brief Semantic kind of one production checkpoint artifact row.
     */
    enum class Qwen36CheckpointKind
    {
        Numeric,
        RoutingIndices,
        RoutingWeights,
    };

    /**
     * @brief Collect rigorous checkpoint comparisons and emit standard parity CSVs.
     *
     * Prefix/MTP tests use production generation APIs rather than the classic
     * fixture's raw forward loop.  This recorder keeps that execution path but
     * feeds its checkpoints through the same metric and CSV authorities as the
     * established PyTorch parity suite.  A cell cannot finalize without both
     * prefill and decode stage evidence.
     */
    class Qwen36CheckpointArtifactRecorder final
    {
    public:
        Qwen36CheckpointArtifactRecorder(
            std::string backend,
            float prefill_cosine_threshold,
            float decode_cosine_threshold,
            float logit_kl_threshold,
            int routing_top_k = 8,
            int num_experts = 256)
            : backend_(std::move(backend)),
              prefill_cosine_threshold_(prefill_cosine_threshold),
              decode_cosine_threshold_(decode_cosine_threshold),
              logit_kl_threshold_(logit_kl_threshold),
              routing_top_k_(routing_top_k),
              num_experts_(num_experts)
        {
            if (routing_top_k_ <= 0 || num_experts_ < routing_top_k_)
                throw std::invalid_argument("invalid Qwen3.6 routing geometry");
        }

        /** @brief Record one prefill checkpoint against its PyTorch tensor. */
        void recordPrefill(
            int layer,
            const std::string &stage,
            const std::vector<float> &actual,
            const std::vector<float> &expected,
            Qwen36CheckpointKind kind = Qwen36CheckpointKind::Numeric)
        {
            auto comparison = compare(
                stage,
                actual,
                expected,
                kind,
                routingContext("prefill", -1, layer, stage),
                prefill_cosine_threshold_);
            if (!comparison)
                return;

            if (stage == "EMBEDDING")
            {
                prefill_.embedding_cosine = comparison->cosine_similarity;
                prefill_.embedding_passed = comparison->passed;
                saw_prefill_embedding_ = true;
                return;
            }
            if (stage == "LM_HEAD")
            {
                populatePrefillLogits(*comparison, actual, expected);
                saw_prefill_logits_ = true;
                return;
            }
            addStage(
                prefill_.layer_stats,
                layer,
                std::move(*comparison),
                prefill_cosine_threshold_);
            ++prefill_checkpoint_count_;
        }

        /** @brief Record the expected and emitted token for a decode row. */
        void recordDecodeToken(int step, int actual_token, int expected_token)
        {
            DecodeStepStats &stats = decodeStep(step);
            stats.llaminar_token = actual_token;
            stats.pytorch_token = expected_token;
            stats.token_match = actual_token == expected_token;
        }

        /** @brief Record one decode or MTP-sidecar checkpoint. */
        void recordDecode(
            int step,
            int layer,
            const std::string &stage,
            const std::vector<float> &actual,
            const std::vector<float> &expected,
            Qwen36CheckpointKind kind = Qwen36CheckpointKind::Numeric)
        {
            auto comparison = compare(
                stage,
                actual,
                expected,
                kind,
                routingContext("decode", step, layer, stage),
                decode_cosine_threshold_);
            if (!comparison)
                return;

            DecodeStepStats &stats = decodeStep(step);
            if (stage == "LM_HEAD")
            {
                populateDecodeLogits(stats, *comparison, actual, expected);
                ++decode_checkpoint_count_;
                return;
            }
            if (stage.ends_with("_LM_HEAD"))
                populateDecodeLogits(stats, *comparison, actual, expected);
            addStage(
                stats.layer_stats,
                layer,
                std::move(*comparison),
                decode_cosine_threshold_);
            ++decode_checkpoint_count_;
            observed_decode_stages_.insert(stage);
        }

        /**
         * @brief Require at least one recorded decode stage with this suffix.
         */
        void requireDecodeStageSuffix(std::string suffix)
        {
            required_decode_stage_suffixes_.insert(std::move(suffix));
        }

        /**
         * @brief Assert the numerical contract and write all standard CSVs.
         */
        void finalize()
        {
            finalizePrefillSummary();
            finalizeDecodeSummary();

            EXPECT_TRUE(saw_prefill_embedding_)
                << "Production checkpoint parity recorded no prefill EMBEDDING";
            EXPECT_TRUE(saw_prefill_logits_)
                << "Production checkpoint parity recorded no prefill LM_HEAD";
            EXPECT_GT(prefill_checkpoint_count_, 0u)
                << "Production checkpoint parity recorded no prefill layer stages";
            EXPECT_GT(decode_checkpoint_count_, 0u)
                << "Production checkpoint parity recorded no decode stages";

            for (const std::string &suffix : required_decode_stage_suffixes_)
            {
                const bool observed = std::any_of(
                    observed_decode_stages_.begin(),
                    observed_decode_stages_.end(),
                    [&suffix](const std::string &stage)
                    {
                        return stage.size() >= suffix.size() &&
                               stage.compare(
                                   stage.size() - suffix.size(),
                                   suffix.size(),
                                   suffix) == 0;
                    });
                EXPECT_TRUE(observed)
                    << "Production checkpoint parity missing required stage suffix "
                    << suffix;
            }

            EXPECT_GE(prefill_.early_layers_passed, min_early_layers_passed_)
                << "At least " << min_early_layers_passed_ << " of embedding plus the first "
                << early_layers_count_ << " layers must pass prefill parity";
            EXPECT_TRUE(prefill_.lm_head_passed)
                << "Prefill LM_HEAD failed KL/top-k parity";
            EXPECT_TRUE(decode_.overall_passed)
                << "Decode checkpoint/logit campaign failed established parity gates";

            const auto results_dir = ParityCSVArtifactWriter::resultsDir();
            EXPECT_TRUE(ParityCSVArtifactWriter::writePrefill(
                results_dir, backend_, prefill_))
                << "Failed to write standard prefill CSV artifacts";
            EXPECT_TRUE(ParityCSVArtifactWriter::writeDecode(
                results_dir, backend_, decode_))
                << "Failed to write standard decode CSV artifacts";
        }

    private:
        std::optional<StageComparisonResult> compare(
            const std::string &stage,
            const std::vector<float> &actual,
            const std::vector<float> &reference,
            Qwen36CheckpointKind kind,
            const std::string &routing_context,
            float cosine_threshold)
        {
            if (actual.empty() || reference.empty())
            {
                ADD_FAILURE() << stage << " has an empty production/reference payload";
                return std::nullopt;
            }

            const float *expected = reference.data();
            size_t expected_size = reference.size();
            if (reference.size() > actual.size() &&
                reference.size() % actual.size() == 0)
            {
                expected = reference.data() + (reference.size() - actual.size());
                expected_size = actual.size();
            }
            if (actual.size() != expected_size)
            {
                ADD_FAILURE()
                    << stage << " checkpoint size mismatch: production="
                    << actual.size() << " reference=" << reference.size();
                return std::nullopt;
            }

            if (kind == Qwen36CheckpointKind::RoutingIndices)
            {
                std::vector<float> aligned_expected(
                    expected,
                    expected + expected_size);
                routing_indices_[routing_context] = {
                    actual,
                    aligned_expected};
                return compareRoutingIndices(
                    stage,
                    actual,
                    aligned_expected,
                    cosine_threshold);
            }
            if (kind == Qwen36CheckpointKind::RoutingWeights)
            {
                const auto indices = routing_indices_.find(routing_context);
                if (indices == routing_indices_.end())
                {
                    ADD_FAILURE()
                        << stage << " has no paired routing-index checkpoint in "
                        << routing_context;
                    return std::nullopt;
                }
                return compareRoutingWeights(
                    stage,
                    actual,
                    std::vector<float>(expected, expected + expected_size),
                    indices->second.first,
                    indices->second.second,
                    cosine_threshold);
            }

            StageComparisonResult result = compareParityTensorData(
                actual.data(),
                expected,
                actual.size(),
                stage,
                cosine_threshold);
            return result;
        }

        static std::string routingContext(
            std::string_view phase,
            int step,
            int layer,
            const std::string &stage)
        {
            const size_t suffix = stage.find("MOE_ROUTING_");
            const std::string family =
                suffix == std::string::npos ? stage : stage.substr(0, suffix);
            return std::string(phase) + '/' + std::to_string(step) + '/' +
                   std::to_string(layer) + '/' + family;
        }

        StageComparisonResult compareRoutingIndices(
            const std::string &stage,
            const std::vector<float> &actual,
            const std::vector<float> &expected,
            float cosine_threshold) const
        {
            StageComparisonResult result;
            result.stage_name = stage;
            result.total_elements = actual.size();
            result.is_routing_stage = true;
            if (actual.size() != expected.size() ||
                actual.size() % static_cast<size_t>(routing_top_k_) != 0)
            {
                return result;
            }

            const size_t rows =
                actual.size() / static_cast<size_t>(routing_top_k_);
            double overlap_sum = 0.0;
            size_t top1_matches = 0;
            for (size_t row = 0; row < rows; ++row)
            {
                std::set<int> actual_experts;
                std::set<int> expected_experts;
                for (int route = 0; route < routing_top_k_; ++route)
                {
                    const size_t index =
                        row * static_cast<size_t>(routing_top_k_) +
                        static_cast<size_t>(route);
                    actual_experts.insert(static_cast<int>(actual[index]));
                    expected_experts.insert(static_cast<int>(expected[index]));
                }
                size_t intersection = 0;
                for (const int expert : actual_experts)
                    intersection += expected_experts.contains(expert) ? 1u : 0u;
                overlap_sum += static_cast<double>(intersection) /
                               static_cast<double>(routing_top_k_);
                top1_matches +=
                    static_cast<int>(actual[row * routing_top_k_]) ==
                            static_cast<int>(expected[row * routing_top_k_])
                        ? 1u
                        : 0u;
            }
            result.routing_overlap = static_cast<float>(
                overlap_sum / static_cast<double>(rows));
            result.routing_top1_match = static_cast<float>(
                static_cast<double>(top1_matches) / static_cast<double>(rows));
            result.cosine_similarity = result.routing_overlap;
            result.max_abs_diff = 1.0f - result.routing_overlap;
            result.passed = result.routing_overlap >= cosine_threshold;
            return result;
        }

        StageComparisonResult compareRoutingWeights(
            const std::string &stage,
            const std::vector<float> &actual_weights,
            const std::vector<float> &expected_weights,
            const std::vector<float> &actual_indices,
            const std::vector<float> &expected_indices,
            float cosine_threshold) const
        {
            StageComparisonResult result;
            result.stage_name = stage;
            result.total_elements = actual_weights.size();
            result.is_routing_stage = true;
            if (actual_weights.size() != expected_weights.size() ||
                actual_weights.size() != actual_indices.size() ||
                actual_indices.size() != expected_indices.size() ||
                actual_weights.size() % static_cast<size_t>(routing_top_k_) != 0)
            {
                return result;
            }

            const size_t rows =
                actual_weights.size() / static_cast<size_t>(routing_top_k_);
            double cosine_sum = 0.0;
            double l1_sum = 0.0;
            float max_difference = 0.0f;
            std::vector<float> actual_sparse(
                static_cast<size_t>(num_experts_));
            std::vector<float> expected_sparse(
                static_cast<size_t>(num_experts_));
            for (size_t row = 0; row < rows; ++row)
            {
                std::fill(actual_sparse.begin(), actual_sparse.end(), 0.0f);
                std::fill(expected_sparse.begin(), expected_sparse.end(), 0.0f);
                for (int route = 0; route < routing_top_k_; ++route)
                {
                    const size_t index =
                        row * static_cast<size_t>(routing_top_k_) +
                        static_cast<size_t>(route);
                    const int actual_expert =
                        static_cast<int>(actual_indices[index]);
                    const int expected_expert =
                        static_cast<int>(expected_indices[index]);
                    if (actual_expert >= 0 && actual_expert < num_experts_)
                        actual_sparse[static_cast<size_t>(actual_expert)] =
                            actual_weights[index];
                    if (expected_expert >= 0 && expected_expert < num_experts_)
                        expected_sparse[static_cast<size_t>(expected_expert)] =
                            expected_weights[index];
                }

                double dot = 0.0;
                double actual_norm = 0.0;
                double expected_norm = 0.0;
                double row_l1 = 0.0;
                for (int expert = 0; expert < num_experts_; ++expert)
                {
                    const double actual =
                        actual_sparse[static_cast<size_t>(expert)];
                    const double expected =
                        expected_sparse[static_cast<size_t>(expert)];
                    const double difference = std::abs(actual - expected);
                    dot += actual * expected;
                    actual_norm += actual * actual;
                    expected_norm += expected * expected;
                    row_l1 += difference;
                    max_difference = std::max(
                        max_difference,
                        static_cast<float>(difference));
                }
                const double denominator =
                    std::sqrt(actual_norm * expected_norm);
                cosine_sum += denominator > 1.0e-30 ? dot / denominator : 0.0;
                l1_sum += row_l1;
            }
            result.routing_overlap = static_cast<float>(
                cosine_sum / static_cast<double>(rows));
            result.routing_weight_l1 = static_cast<float>(
                l1_sum / static_cast<double>(rows));
            result.cosine_similarity = result.routing_overlap;
            result.max_abs_diff = max_difference;
            result.passed = result.routing_overlap >= cosine_threshold;
            return result;
        }

        static void addStage(
            std::vector<LayerStats> &layers,
            int layer_index,
            StageComparisonResult result,
            float cosine_threshold)
        {
            auto layer_it = std::find_if(
                layers.begin(),
                layers.end(),
                [layer_index](const LayerStats &layer)
                {
                    return layer.layer_idx == layer_index;
                });
            if (layer_it == layers.end())
            {
                layers.push_back(LayerStats{.layer_idx = layer_index});
                layer_it = std::prev(layers.end());
            }
            layer_it->stage_results.push_back(std::move(result));
            recomputeLayer(*layer_it, cosine_threshold);
            std::sort(
                layers.begin(),
                layers.end(),
                [](const LayerStats &left, const LayerStats &right)
                {
                    return left.layer_idx < right.layer_idx;
                });
        }

        static void recomputeLayer(
            LayerStats &layer,
            float cosine_threshold)
        {
            layer.avg_cosine_sim = 0.0f;
            layer.min_cosine_sim = 1.0f;
            layer.max_cosine_drop = 0.0f;
            layer.stages_compared = 0;
            layer.passed = false;
            layer.worst_stage.clear();
            layer.max_drop_stage.clear();
            layer.max_kurtosis = 0.0f;
            layer.max_kurtosis_stage.clear();

            float previous_cosine = 0.0f;
            bool have_previous = false;
            for (auto &stage : layer.stage_results)
            {
                if (stage.llaminar_stats.kurtosis > layer.max_kurtosis)
                {
                    layer.max_kurtosis = stage.llaminar_stats.kurtosis;
                    layer.max_kurtosis_stage = stage.stage_name;
                }
                if (stage.is_routing_stage)
                    continue;

                ++layer.stages_compared;
                layer.avg_cosine_sim += stage.cosine_similarity;
                if (stage.cosine_similarity < layer.min_cosine_sim)
                {
                    layer.min_cosine_sim = stage.cosine_similarity;
                    layer.worst_stage = stage.stage_name;
                }
                if (have_previous)
                {
                    stage.cosine_drop = previous_cosine - stage.cosine_similarity;
                    if (stage.cosine_drop > layer.max_cosine_drop)
                    {
                        layer.max_cosine_drop = stage.cosine_drop;
                        layer.max_drop_stage = stage.stage_name;
                    }
                }
                previous_cosine = stage.cosine_similarity;
                have_previous = true;
            }
            if (layer.stages_compared > 0)
            {
                layer.avg_cosine_sim /=
                    static_cast<float>(layer.stages_compared);
                layer.passed =
                    layer.avg_cosine_sim >= cosine_threshold;
            }
        }

        DecodeStepStats &decodeStep(int step)
        {
            auto it = std::find_if(
                decode_.step_stats.begin(),
                decode_.step_stats.end(),
                [step](const DecodeStepStats &stats)
                {
                    return stats.step_idx == step;
                });
            if (it == decode_.step_stats.end())
            {
                decode_.step_stats.push_back(DecodeStepStats{.step_idx = step});
                std::sort(
                    decode_.step_stats.begin(),
                    decode_.step_stats.end(),
                    [](const DecodeStepStats &left, const DecodeStepStats &right)
                    {
                        return left.step_idx < right.step_idx;
                    });
                it = std::find_if(
                    decode_.step_stats.begin(),
                    decode_.step_stats.end(),
                    [step](const DecodeStepStats &stats)
                    {
                        return stats.step_idx == step;
                    });
            }
            return *it;
        }

        void populatePrefillLogits(
            const StageComparisonResult &comparison,
            const std::vector<float> &actual,
            const std::vector<float> &reference)
        {
            const size_t size = std::min(actual.size(), reference.size());
            const float *expected = reference.data() + (reference.size() - size);
            const float *observed = actual.data() + (actual.size() - size);
            prefill_.lm_head_cosine = comparison.cosine_similarity;
            prefill_.lm_head_kl = computeKLDivergence(
                observed, expected, size, size);
            prefill_.lm_head_top1 = computeTopKOverlap(
                observed, expected, size, size, 1);
            prefill_.lm_head_top5 = computeTopKOverlap(
                observed, expected, size, size, 5);
            prefill_.lm_head_pytorch_top1_in_top3 =
                pytorchTop1InLlaminarTopK(
                    observed, expected, size, size, 3) >= 1.0f - 1.0e-6f;
            prefill_.lm_head_passed =
                prefill_.lm_head_kl < logit_kl_threshold_ &&
                prefill_.lm_head_top1 * 100.0f >= min_top1_accuracy_ &&
                prefill_.lm_head_top5 * 100.0f >= min_top5_accuracy_ &&
                prefill_.lm_head_pytorch_top1_in_top3;
        }

        void populateDecodeLogits(
            DecodeStepStats &stats,
            const StageComparisonResult &comparison,
            const std::vector<float> &actual,
            const std::vector<float> &reference)
        {
            stats.has_logit_data = true;
            const size_t size = std::min(actual.size(), reference.size());
            const float *expected = reference.data() + (reference.size() - size);
            const float *observed = actual.data() + (actual.size() - size);
            stats.cosine_similarity = comparison.cosine_similarity;
            stats.kl_divergence = computeKLDivergence(
                observed, expected, size, size);
            stats.top1_overlap = computeTopKOverlap(
                observed, expected, size, size, 1);
            stats.top5_overlap = computeTopKOverlap(
                observed, expected, size, size, 5);
            stats.top3_match = pytorchTop1InLlaminarTopK(
                                   observed, expected, size, size, 3) >=
                               1.0f - 1.0e-6f;
            stats.top5_match = pytorchTop1InLlaminarTopK(
                                   observed, expected, size, size, 5) >=
                               1.0f - 1.0e-6f;
            stats.passed =
                comparison.cosine_similarity >= decode_cosine_threshold_ ||
                stats.kl_divergence < logit_kl_threshold_;
        }

        void finalizePrefillSummary()
        {
            prefill_.early_layers_passed = prefill_.embedding_passed ? 1 : 0;
            prefill_.total_layers_passed = prefill_.embedding_passed ? 1 : 0;
            for (const auto &layer : prefill_.layer_stats)
            {
                if (layer.passed)
                {
                    ++prefill_.total_layers_passed;
                    if (layer.layer_idx >= 0 &&
                        layer.layer_idx < early_layers_count_)
                        ++prefill_.early_layers_passed;
                }
            }
            prefill_.overall_passed =
                prefill_.early_layers_passed >= min_early_layers_passed_ &&
                prefill_.lm_head_passed;
        }

        void finalizeDecodeSummary()
        {
            decode_.steps_total = static_cast<int>(decode_.step_stats.size());
            for (auto &step : decode_.step_stats)
            {
                if (step.passed)
                    ++decode_.steps_passed;
                if (step.token_match)
                    ++decode_.top1_matches;
                if (step.top3_match)
                    ++decode_.top3_matches;
                if (step.top5_match)
                    ++decode_.top5_matches;
                decode_.avg_cosine += step.cosine_similarity;
                decode_.avg_kl += step.kl_divergence;
            }
            if (decode_.steps_total > 0)
            {
                const float denominator =
                    static_cast<float>(decode_.steps_total);
                decode_.avg_cosine /= denominator;
                decode_.avg_kl /= denominator;
                decode_.top1_accuracy =
                    100.0f * static_cast<float>(decode_.top1_matches) /
                    denominator;
                decode_.top3_accuracy =
                    100.0f * static_cast<float>(decode_.top3_matches) /
                    denominator;
                decode_.top5_accuracy =
                    100.0f * static_cast<float>(decode_.top5_matches) /
                    denominator;
            }
            const int minimum_steps = static_cast<int>(
                static_cast<float>(decode_.steps_total) *
                minimum_decode_pass_rate_);
            decode_.overall_passed =
                decode_.steps_total > 0 &&
                decode_.steps_passed >= minimum_steps &&
                decode_.top5_accuracy >= min_top5_accuracy_ &&
                decode_.avg_cosine >= decode_cosine_threshold_ &&
                decode_.top3_matches == decode_.steps_total;
        }

        std::string backend_;
        float prefill_cosine_threshold_ = 0.96f;
        float decode_cosine_threshold_ = 0.98f;
        float logit_kl_threshold_ = 0.03f;
        int early_layers_count_ = 6;
        int min_early_layers_passed_ = 5;
        float min_top1_accuracy_ = 80.0f;
        float min_top5_accuracy_ = 60.0f;
        float minimum_decode_pass_rate_ = 0.8f;
        int routing_top_k_ = 8;
        int num_experts_ = 256;
        ParityTestSummary prefill_;
        DecodeParitySummary decode_;
        bool saw_prefill_embedding_ = false;
        bool saw_prefill_logits_ = false;
        size_t prefill_checkpoint_count_ = 0;
        size_t decode_checkpoint_count_ = 0;
        std::map<
            std::string,
            std::pair<std::vector<float>, std::vector<float>>>
            routing_indices_;
        std::set<std::string> observed_decode_stages_;
        std::set<std::string> required_decode_stage_suffixes_;
    };

    /** @brief Split a canonical snapshot key into its layer and stage fields. */
    inline std::pair<int, std::string> qwen36CheckpointLayerAndStage(
        const std::string &key)
    {
        if (key.rfind("layer", 0) != 0)
            return {-1, key};
        size_t digit_end = 5;
        while (digit_end < key.size() &&
               std::isdigit(static_cast<unsigned char>(key[digit_end])))
        {
            ++digit_end;
        }
        if (digit_end == 5 || digit_end >= key.size() || key[digit_end] != '_')
            return {-1, key};
        return {
            std::stoi(key.substr(5, digit_end - 5)),
            key.substr(digit_end + 1)};
    }

    /** @brief Select the semantically correct comparison for a stage. */
    inline Qwen36CheckpointKind qwen36CheckpointKind(const std::string &stage)
    {
        if (stage.ends_with("MOE_ROUTING_INDICES"))
            return Qwen36CheckpointKind::RoutingIndices;
        if (stage.ends_with("MOE_ROUTING_WEIGHTS"))
            return Qwen36CheckpointKind::RoutingWeights;
        return Qwen36CheckpointKind::Numeric;
    }

    /**
     * @brief Select the real prompt rows from a fixed-capacity captured tensor.
     *
     * GPU prefill graphs execute a bucket-sized physical matrix while PyTorch
     * snapshots contain only logical prompt rows.  The graph contract places
     * the logical prefix at the leading rows and masks the remaining capacity.
     * Project that explicit row interval before numerical comparison; treating
     * a shape mismatch as an absent checkpoint would silently erase nearly all
     * prefill coverage under graph capture.
     */
    inline std::optional<std::vector<float>> qwen36ProjectCapturedPrefillRows(
        const std::string &key,
        const float *actual,
        size_t actual_size,
        const std::vector<float> &expected,
        size_t logical_rows)
    {
        if (!actual || actual_size == 0 || expected.empty() || logical_rows == 0)
            return std::nullopt;
        if (actual_size == expected.size())
            return std::vector<float>(actual, actual + actual_size);

        if (expected.size() % logical_rows == 0)
        {
            const size_t columns = expected.size() / logical_rows;
            if (key == "LM_HEAD" && actual_size == columns)
            {
                return std::vector<float>(actual, actual + actual_size);
            }
            if (columns > 0 && actual_size % columns == 0)
            {
                const size_t physical_rows = actual_size / columns;
                if (physical_rows >= logical_rows)
                {
                    return std::vector<float>(
                        actual,
                        actual + logical_rows * columns);
                }
            }
        }

        /*
         * Terminal-only stages such as LM_HEAD may publish one selected row
         * even when their producer owns the full physical bucket.  Select the
         * last logical row, never the padded physical tail.
         */
        if (actual_size > expected.size() &&
            actual_size % expected.size() == 0)
        {
            const size_t physical_rows = actual_size / expected.size();
            if (physical_rows >= logical_rows)
            {
                const size_t row = logical_rows - 1;
                const float *begin = actual + row * expected.size();
                return std::vector<float>(begin, begin + expected.size());
            }
        }

        ADD_FAILURE()
            << key << " cannot project captured prefill shape: production="
            << actual_size << " reference=" << expected.size()
            << " logical_rows=" << logical_rows;
        return std::nullopt;
    }

    /**
     * @brief Compare every available production prefill checkpoint to PyTorch.
     */
    inline void recordQwen36PrefillArtifacts(
        Qwen36CheckpointArtifactRecorder &recorder,
        IOrchestrationRunner &runner,
        const std::filesystem::path &reference_dir,
        size_t logical_rows,
        const ParityGDNHeadConfig &gdn_config = {})
    {
        size_t comparable = 0;
        auto keys = runner.getSnapshotKeys();
        std::sort(keys.begin(), keys.end());
        for (const std::string &key : keys)
        {
            const std::vector<float> expected =
                loadDensePyTorchSnapshot(reference_dir, key);
            if (expected.empty())
                continue;
            size_t actual_size = 0;
            const float *actual_data = runner.getSnapshot(key, actual_size);
            if (!actual_data || actual_size == 0)
            {
                ADD_FAILURE() << "Production prefill advertised an empty checkpoint "
                              << key;
                continue;
            }
            auto projected = qwen36ProjectCapturedPrefillRows(
                key,
                actual_data,
                actual_size,
                expected,
                logical_rows);
            if (!projected)
                continue;
            const auto [layer, stage] = qwen36CheckpointLayerAndStage(key);
            auto permuted = applyParityGDNHeadPermutation(
                projected->data(),
                projected->size(),
                stage,
                gdn_config);
            const std::vector<float> &production =
                permuted.empty() ? *projected : permuted;
            recorder.recordPrefill(
                layer,
                stage,
                production,
                expected,
                qwen36CheckpointKind(stage));
            ++comparable;
        }
        EXPECT_GT(comparable, 0u)
            << "No production prefill checkpoints matched references in "
            << reference_dir;
    }

    inline std::vector<std::string> denseOrderedDecodeSnapshotKeys(
        const DenseDecodeSnapshotComparisonPolicy &policy)
    {
        /*
         * Keep this order aligned with the Qwen3.6 dense graph so a failed
         * diagnostic points at the first mismatching layer instead of the
         * accumulated final norm.  Missing expected snapshots are skipped by
         * denseDecodeStepSnapshotsNearPyTorch(), which lets the same helper
         * work for GDN and full-attention layers.
         */
        static const std::vector<std::string> kLayerStageOrder = {
            "ATTENTION_NORM",
            "QKV_PROJECTION",
            "GDN_Z_PROJECTION",
            "GDN_ALPHA",
            "GDN_BETA",
            "GDN_CONV1D_OUTPUT",
            "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT",
            "ATTENTION_OUTPUT",
            "ATTENTION_RESIDUAL",
            "FFN_NORM",
            "FFN_DOWN",
            "FFN_RESIDUAL",
        };

        std::vector<std::string> keys;
        const int requested_layer_count = std::max(policy.layer_count, 0);
        const int compared_layer_count = policy.max_layer_count >= 0
                                             ? std::min(requested_layer_count,
                                                        policy.max_layer_count)
                                             : requested_layer_count;
        keys.reserve(2 + static_cast<size_t>(compared_layer_count) *
                             kLayerStageOrder.size());
        keys.push_back("EMBEDDING");
        for (int layer = 0; layer < compared_layer_count; ++layer)
        {
            const std::string prefix = "layer" + std::to_string(layer) + "_";
            for (const std::string &stage : kLayerStageOrder)
            {
                keys.push_back(prefix + stage);
            }
        }
        if (policy.compare_final_outputs)
        {
            keys.push_back("FINAL_NORM");
            keys.push_back("LM_HEAD");
        }
        return keys;
    }

    inline const std::vector<std::string> &denseLegacyOrderedDecodeSnapshotKeys()
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
        return kOrderedKeys;
    }

    inline ::testing::AssertionResult denseDecodeStepSnapshotsNearPyTorch(
        const std::map<std::string, DenseStageSnapshot> &snapshots,
        const std::filesystem::path &snapshot_dir,
        int decode_step,
        const std::string &label,
        const DenseDecodeSnapshotComparisonPolicy &policy = {})
    {
        const std::string prefix =
            "decode_step" + std::to_string(decode_step) + "_";
        size_t compared = 0;
        std::ostringstream errors;
        const std::vector<std::string> dynamic_keys =
            policy.layer_count > 0 ? denseOrderedDecodeSnapshotKeys(policy)
                            : denseLegacyOrderedDecodeSnapshotKeys();
        for (const auto &key : dynamic_keys)
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
                label + " " + key,
                policy.cosine_threshold,
                policy.rel_l2_threshold);
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

    /**
     * @brief Copy orchestration-level snapshots without inventing tensor shape.
     *
     * The orchestration API intentionally exposes only authenticated element
     * cardinality because TP/PP reconstruction may have joined participant
     * shapes. Dense MTP checkpoint comparisons need the exact flat payload,
     * not a guessed row/column decomposition, so the diagnostic records one
     * logical row and the complete cardinality here.
     */
    inline std::map<std::string, DenseStageSnapshot> captureDenseStageSnapshots(
        IOrchestrationRunner &runner)
    {
        std::map<std::string, DenseStageSnapshot> snapshots;
        auto keys = runner.getSnapshotKeys();
        std::sort(keys.begin(), keys.end());
        for (const std::string &key : keys)
        {
            size_t size = 0;
            const float *data = runner.getSnapshot(key, size);
            if (!data || size == 0)
                continue;
            snapshots.emplace(
                key,
                DenseStageSnapshot{
                    .key = key,
                    .data = std::vector<float>(data, data + size),
                    .rows = 1,
                    .cols = size,
                });
        }
        return snapshots;
    }

    inline ::testing::AssertionResult denseVerifierRowSnapshotsByteIdentical(
        const std::map<std::string, DenseStageSnapshot> &verifier_snapshots,
        const std::map<std::string, DenseStageSnapshot> &single_row_snapshots,
        const std::string &label,
        int verifier_rows,
        int verifier_row_index)
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
            uint32_t first_actual_bits = 0;
            uint32_t first_expected_bits = 0;
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
                uint32_t actual_bits = 0;
                uint32_t expected_bits = 0;
                std::memcpy(&actual_bits, &a, sizeof(actual_bits));
                std::memcpy(&expected_bits, &e, sizeof(expected_bits));
                const float abs_diff = std::fabs(a - e);
                const float scale = std::max(std::fabs(a), std::fabs(e));
                const float rel_diff = scale > 0.0f ? abs_diff / scale : 0.0f;
                if (actual_bits != expected_bits)
                {
                    if (mismatch.mismatches == 0)
                    {
                        mismatch.col = col;
                        mismatch.first_actual = a;
                        mismatch.first_expected = e;
                        mismatch.first_actual_bits = actual_bits;
                        mismatch.first_expected_bits = expected_bits;
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
        oss << label << " stage snapshot row byte mismatch across "
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
                << " actual_bits=0x" << std::hex << m.first_actual_bits
                << " expected_bits=0x" << m.first_expected_bits << std::dec
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

    /**
     * @brief Read an exact multiline value from the line-oriented snapshot metadata format.
     *
     * The Python snapshot generators write prompts as human-readable text:
     * the first physical line begins with `prompt:`, while every remaining
     * prompt line is written verbatim until the following `token_ids:` field.
     * `readStringFromMetadata()` is intentionally a scalar-field reader and
     * therefore cannot authenticate such a prompt; using it here truncated the
     * expected value at the first newline and made every valid long-context
     * corpus appear stale.
     *
     * This parser preserves every embedded newline and every continuation-line
     * byte. It removes only a terminal carriage return so metadata written with
     * CRLF line endings compares identically on Linux. The field must be
     * terminated by the named next key. A missing terminator is treated as a
     * malformed corpus rather than accepting a partially written metadata file.
     *
     * @param metadata_path Metadata file produced by a parity snapshot generator.
     * @param key Name of the multiline field, without the trailing colon.
     * @param terminator_key Name of the scalar field immediately following it.
     * @return The exact logical field value, or `std::nullopt` when the file or
     *         requested field is incomplete.
     */
    inline std::optional<std::string> readMultilineStringFromMetadata(
        const std::filesystem::path &metadata_path,
        const std::string &key,
        const std::string &terminator_key)
    {
        std::ifstream file(metadata_path);
        if (!file.is_open())
        {
            return std::nullopt;
        }

        const std::string prefix = key + ":";
        const std::string terminator_prefix = terminator_key + ":";
        std::string line;
        while (std::getline(file, line))
        {
            if (line.rfind(prefix, 0) != 0)
            {
                continue;
            }

            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }

            std::string value = line.substr(prefix.size());
            const size_t first_content = value.find_first_not_of(" \t");
            value = first_content == std::string::npos
                        ? std::string{}
                        : value.substr(first_content);

            while (std::getline(file, line))
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                if (line.rfind(terminator_prefix, 0) == 0)
                {
                    return value;
                }

                // A physical continuation line represents an embedded newline,
                // including when the first prompt line itself was empty.
                value.push_back('\n');
                value += line;
            }

            return std::nullopt;
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
        const auto prompt = readMultilineStringFromMetadata(
            metadata_path,
            "prompt",
            "token_ids");
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

    /**
     * @brief Authenticate one CPU/FP32 PyTorch pack against live model inputs.
     *
     * Typed campaign registration binds the model and pack. The inexpensive
     * filename/size descriptor catches accidental fixture selection, while
     * prompt/token digests and full numerical checkpoints prove the actual
     * inference inputs without rescanning the GGUF payload.
     */
    inline bool qwen36ReferenceIdentityMatches(
        const std::filesystem::path &metadata_path,
        const std::string &model_path,
        const std::string &prompt,
        std::string *reason = nullptr)
    {
        const auto fail = [reason](std::string message)
        {
            if (reason)
                *reason = std::move(message);
            return false;
        };
        const auto expect = [&](const char *key, const char *value)
        {
            const auto observed = readStringFromMetadata(metadata_path, key);
            return observed.has_value() && *observed == value;
        };
        if (!expect("reference_identity_version", "1") ||
            !expect("reference_engine", "pytorch") ||
            !expect("reference_device", "cpu") ||
            !expect("reference_dtype", "float32"))
        {
            return fail("missing or incompatible CPU/FP32 PyTorch identity");
        }

        std::string digest_error;
        if (const auto model_error = productionParityModelDescriptorError(
                model_path,
                readStringFromMetadata(metadata_path, "model_filename"),
                readStringFromMetadata(metadata_path, "model_size_bytes")))
            return fail(*model_error);

        const auto prompt_digest = sha256BytesHex(prompt, &digest_error);
        const auto observed_prompt_digest =
            readStringFromMetadata(metadata_path, "prompt_sha256");
        if (!prompt_digest || !observed_prompt_digest ||
            *observed_prompt_digest != *prompt_digest)
        {
            return fail("prompt_sha256 does not match the configured prompt");
        }

        const auto token_ids =
            readTokenListFromMetadata(metadata_path, "token_ids");
        if (token_ids.empty())
            return fail("token_ids is missing or malformed");
        std::ostringstream canonical_tokens;
        for (size_t index = 0; index < token_ids.size(); ++index)
        {
            if (index != 0)
                canonical_tokens << ',';
            canonical_tokens << token_ids[index];
        }
        const auto token_digest =
            sha256BytesHex(canonical_tokens.str(), &digest_error);
        const auto observed_token_digest =
            readStringFromMetadata(metadata_path, "token_ids_sha256");
        if (!token_digest || !observed_token_digest ||
            *observed_token_digest != *token_digest)
        {
            return fail("token_ids_sha256 does not authenticate token_ids");
        }

        if (reason)
            reason->clear();
        return true;
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
        int required_decode_steps,
        bool require_mtp_sidecar_snapshots = false)
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

        if (!require_mtp_sidecar_snapshots)
            return true;

        /*
         * Schema 5 binds recursive depth N to the preceding predictor's
         * shared-head-normalized hidden result.  Older experimental packs
         * chained the pre-normalized decoder residual and cannot certify the
         * production MTP graph even when their filenames happen to match.
         */
        constexpr int kMTPSidecarSnapshotSchema = 5;
        std::ifstream schema_file(
            snapshot_dir / "mtp_sidecar_snapshot_schema.txt");
        int observed_schema = 0;
        if (!(schema_file >> observed_schema) ||
            observed_schema != kMTPSidecarSnapshotSchema)
        {
            return false;
        }

        const std::vector<std::string_view> sidecar_stage_suffixes = {
            "TERMINAL_HIDDEN_ROW_SELECT",
            "EMBEDDING",
            "NORM_HIDDEN",
            "NORM_EMBEDDING",
            "CONCAT",
            "FC",
            "ATTENTION_NORM",
            "Q_PROJECTION",
            "FA_GATE",
            "K_PROJECTION",
            "V_PROJECTION",
            "Q_NORM",
            "K_NORM",
            "ATTENTION_CONTEXT",
            "ATTENTION_CONTEXT_GATED",
            "ATTENTION_OUTPUT",
            "FFN_NORM",
            "FFN_GATE",
            "FFN_UP",
            "FFN_SWIGLU",
            "FFN_DOWN",
            "FFN_RESIDUAL",
            "FINAL_NORM",
            "LM_HEAD",
        };
        if (!std::filesystem::is_regular_file(
                snapshot_dir /
                "decode_step0_MTP_TERMINAL_HIDDEN_ROW_SELECT.npy"))
        {
            return false;
        }
        for (int depth = 0; depth <= 2; ++depth)
        {
            for (const std::string_view suffix : sidecar_stage_suffixes)
            {
                const auto path = snapshot_dir /
                    ("decode_step0_MTP" + std::to_string(depth) + "_" +
                     std::string(suffix) + ".npy");
                if (!std::filesystem::is_regular_file(path))
                    return false;
            }
        }
        return std::filesystem::is_regular_file(
            snapshot_dir /
            ("decode_step" + std::to_string(required_decode_steps - 1) +
             "_MTP2_LM_HEAD.npy"));
    }

    inline bool regenerateQwen36DecodeSnapshots(
        const std::string &model_path,
        const std::filesystem::path &metadata_path,
        const std::string &prompt,
        int decode_steps,
        bool require_mtp_sidecar_snapshots,
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
        if (require_mtp_sidecar_snapshots)
            script += " --mtp-sidecar-snapshots";

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
        const std::filesystem::path &metadata_path,
        bool require_mtp_sidecar_snapshots = false)
    {
        const auto snapshot_dir = metadata_path.parent_path();
        std::string identity_error;
        if (qwen36DecodeSnapshotsLookUsable(
                snapshot_dir,
                test_case.decode_steps,
                require_mtp_sidecar_snapshots) &&
            metadataLooksUsable(
                metadata_path,
                test_case.prompt,
                test_case.decode_steps) &&
            qwen36ReferenceIdentityMatches(
                metadata_path,
                model_path,
                test_case.prompt,
                &identity_error))
        {
            return;
        }

        std::string output;
        ASSERT_TRUE(regenerateQwen36DecodeSnapshots(
            model_path,
            metadata_path,
            test_case.prompt,
            test_case.decode_steps,
            require_mtp_sidecar_snapshots,
            &output))
            << test_case.name << " failed to regenerate PyTorch decode snapshots at "
            << snapshot_dir << "\n"
            << output;

        ASSERT_TRUE(
            metadataLooksUsable(
                metadata_path,
                test_case.prompt,
                test_case.decode_steps) &&
            qwen36ReferenceIdentityMatches(
                metadata_path,
                model_path,
                test_case.prompt,
                &identity_error))
            << test_case.name << " regenerated metadata is incomplete at "
            << metadata_path << "\n"
            << identity_error << "\n"
            << output;
        ASSERT_TRUE(qwen36DecodeSnapshotsLookUsable(
            snapshot_dir,
            test_case.decode_steps,
            require_mtp_sidecar_snapshots))
            << test_case.name << " regenerated decode snapshots are incomplete at "
            << snapshot_dir << "\n"
            << output;
    }

    inline void ensurePyTorchMetadata(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const std::filesystem::path &metadata_path)
    {
        std::string identity_error;
        if (metadataLooksUsable(
                metadata_path,
                test_case.prompt,
                test_case.decode_steps) &&
            qwen36ReferenceIdentityMatches(
                metadata_path,
                model_path,
                test_case.prompt,
                &identity_error))
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

        ASSERT_TRUE(
            metadataLooksUsable(
                metadata_path,
                test_case.prompt,
                test_case.decode_steps) &&
            qwen36ReferenceIdentityMatches(
                metadata_path,
                model_path,
                test_case.prompt,
                &identity_error))
            << test_case.name << " regenerated metadata is incomplete at "
            << metadata_path << "\n"
            << identity_error << "\n"
            << output;
    }

    inline std::optional<std::string> densePrefixParitySkipReason(
        const DensePrefixRestoreParityCase &test_case)
    {
        const int world_size = mpiWorldSize();
        if (test_case.topology == DensePrefixParityTopology::NodeTP)
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

    /**
     * @brief Select the collective backend for a homogeneous dense LocalTP case.
     *
     * The dense LocalTP parity helper used to assume ROCm because the original
     * generic suite only targeted ROCm devices.  Explicit CUDA and ROCm suites
     * make that assumption dangerous: the test name can say CUDA while the
     * config still asks LocalTPContext to create RCCL.  Keep backend selection
     * beside the topology translation so every explicit backend suite builds a
     * self-consistent TP domain.
     *
     * @param devices Devices participating in the LocalTP domain.
     * @return NCCL for homogeneous CUDA, RCCL for homogeneous ROCm, AUTO
     *         otherwise so production validation can reject unsupported mixes.
     */
    inline CollectiveBackendType denseLocalTPBackendForDevices(
        const std::vector<GlobalDeviceAddress> &devices)
    {
        if (devices.empty())
        {
            return CollectiveBackendType::AUTO;
        }

        const bool all_cuda =
            std::all_of(
                devices.begin(),
                devices.end(),
                [](const GlobalDeviceAddress &device)
                {
                    return device.isCUDA();
                });
        if (all_cuda)
        {
            return CollectiveBackendType::NCCL;
        }

        const bool all_rocm =
            std::all_of(
                devices.begin(),
                devices.end(),
                [](const GlobalDeviceAddress &device)
                {
                    return device.isROCm();
                });
        if (all_rocm)
        {
            return CollectiveBackendType::RCCL;
        }

        return CollectiveBackendType::AUTO;
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
            config.tp_scope = TPScope::RANK_LOCAL;
            config.tp_devices = test_case.devices;
            config.pp_degree = 1;
            config.default_backend = denseLocalTPBackendForDevices(test_case.devices);
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
                domain.scope = TPScope::RANK_LOCAL;
                domain.owner_rank = 0;
                domain.backend = CollectiveBackendType::AUTO;
                config.domain_definitions.push_back(std::move(domain));
            }
            break;

        case DensePrefixParityTopology::NodeTP:
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
        case DensePrefixParityTopology::NodeTP:
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

    /**
     * @brief Returns true for the ROCm single-device dense Qwen3.6 parity case.
     *
     * The benchmark prompt has backend-specific quantization near-ties. Keeping
     * this predicate local to the parity harness makes those expectations
     * explicit without weakening unrelated topologies.
     */
    inline bool isQwen36DenseROCmSingleDeviceCase(
        const DensePrefixRestoreParityCase &test_case)
    {
        return test_case.topology == DensePrefixParityTopology::SingleDevice &&
               !test_case.devices.empty() &&
               test_case.devices.front().isROCm();
    }

    /**
     * @brief Returns true for the CUDA single-device dense Qwen3.6 parity case.
     *
     * CUDA has its own stable PyTorch-token window for the default benchmark
     * prompt. Keeping this explicit prevents the MTP tests from masking a
     * backend/PyTorch quantized near-tie as a speculative decode failure.
     */
    inline bool isQwen36DenseCUDASingleDeviceCase(
        const DensePrefixRestoreParityCase &test_case)
    {
        return test_case.topology == DensePrefixParityTopology::SingleDevice &&
               !test_case.devices.empty() &&
               test_case.devices.front().isCUDA();
    }

    /**
     * @brief Exact-token PyTorch comparison window for the benchmark prompt.
     *
     * Longer benchmark-style MTP tests compare MTP against each backend's
     * no-MTP baseline. This helper only governs token-exact comparisons against
     * the FP32 PyTorch metadata, where quantized Llaminar backends can hit very
     * small top-token ties.
     */
    inline int qwen36BenchmarkPromptStableExactDecodeSteps(
        const DensePrefixRestoreParityCase &test_case)
    {
        if (isQwen36DenseROCmSingleDeviceCase(test_case))
        {
            // ROCm ranks token 4338 ahead of PyTorch token 1092 by only about
            // 0.009 logit at teacher-forced decode step 6. Keep token-exact
            // PyTorch checks on the stable prefix and let the diagnostic below
            // document the near-tie row.
            return 7;
        }

        if (isQwen36DenseCUDASingleDeviceCase(test_case))
        {
            // CUDA ranks token 1061 ahead of PyTorch token 15676 by about
            // 0.068 logit at decode index 48 on the benchmark prompt. The
            // no-MTP baseline and MTP path agree there, so exact PyTorch-token
            // tests stop at the stable prefix and the known-window diagnostic
            // documents the quantized boundary.
            return 48;
        }

        // CPU currently remains stable until the later quantized/PyTorch
        // FP32 near-tie at decode step 114.
        return 115;
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

        *model_path = firstModelEnvOrDefault(
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

    // The PyTorch decode token fixture is the no-MTP correctness oracle for
    // these helpers. Keep dedicated no-MTP/determinism tests separate instead
    // of adding a second large-model baseline runner to every Prefix/MTP cell.
    inline void runDensePrefixRestoreParity(
        const DensePrefixRestoreParityCase &test_case,
        PrefixRestoreParityMode mode)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        constexpr int kPartialPrefixBlockSize = 4;
        constexpr int kTerminalPartialPrefixTokens = 3;
        const int block_size = mode == PrefixRestoreParityMode::FullHit
                                   ? static_cast<int>(prompt_tokens.size())
                                   : kPartialPrefixBlockSize;
        std::unique_ptr<ScopedEnvironmentValues> partial_prefix_graph_env;
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            partial_prefix_graph_env =
                std::make_unique<ScopedEnvironmentValues>(
                    std::initializer_list<std::pair<const char *, const char *>>{
                        {"LLAMINAR_GPU_GRAPHS", "1"},
                        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
                    });
        }
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto cached = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, true, block_size));
        ASSERT_NE(cached, nullptr);
        ASSERT_TRUE(cached->initialize()) << cached->lastError();

        std::vector<int32_t> first_prompt = prompt_tokens;
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            ASSERT_GT(
                prompt_tokens.size(),
                static_cast<size_t>(kTerminalPartialPrefixTokens));
            first_prompt.assign(
                prompt_tokens.begin(),
                prompt_tokens.begin() + kTerminalPartialPrefixTokens);
            ASSERT_NE(first_prompt.size() % static_cast<size_t>(block_size), 0u)
                << "partial-prefix seed must terminate inside a cache block";
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
            EXPECT_EQ(
                after_second.prefix_request.matched_tokens,
                kTerminalPartialPrefixTokens);
            EXPECT_FALSE(after_second.prefix_request.terminal_logits_restored);
        }
    }

    inline void runDenseSplitPrefillParity(
        const DensePrefixRestoreParityCase &test_case,
        int split_tokens)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GT(prompt_tokens.size(), static_cast<size_t>(split_tokens));

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

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
    }

    /**
     * @brief Bounded immutable model authority shared by one dense MTP campaign.
     *
     * Depth changes controller and graph geometry but not the GGUF tensors or
     * their backend-specific prepared representation.  Each depth cell still
     * owns a fresh runner, arena, streams, graphs, controller, and request
     * state; this sole process-local slot retains only the model context and
     * the production plan that certified its prepared weights.
     */
    struct DenseMTPModelContextCampaignCache
    {
        std::mutex mutex;
        std::string key;
        std::optional<ModelContextReuseContract> contract;
    };

    /** @brief Return the one bounded prepared-weight cache for this process. */
    inline DenseMTPModelContextCampaignCache &
    denseMTPModelContextCampaignCache()
    {
        static DenseMTPModelContextCampaignCache cache;
        return cache;
    }

    /**
     * @brief Return whether the aggregate may amortize immutable dense weights.
     *
     * Focused GTest invocations retain fresh-load isolation.  The aggregate
     * driver groups only equivalent single-device depth cells and opts into
     * reuse explicitly through its process-campaign environment contract.
     */
    inline bool mayReuseDenseMTPModelContext(
        const DensePrefixRestoreParityCase &test_case)
    {
        return test_case.topology == DensePrefixParityTopology::SingleDevice &&
               DebugEnv::isTruthyEnv(
                   "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN");
    }

    /** @brief Build the physical identity of reusable prepared dense weights. */
    inline std::string denseMTPModelContextCampaignKey(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path)
    {
        std::ostringstream key;
        key << "model=" << model_path
            << "|topology=" << static_cast<int>(test_case.topology)
            << "|devices=";
        for (const auto &device : test_case.devices)
            key << device.toString() << ',';
        return key.str();
    }

    /**
     * @brief Find a plan-certified context retained by an earlier depth cell.
     *
     * A miss never constructs a guessed context.  The first production runner
     * is the sole authority for main-layer versus trailing-nextn ownership and
     * for the exact prepared-weight plan.
     */
    inline std::optional<ModelContextReuseContract>
    findDenseMTPModelContext(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        bool *cache_hit)
    {
        if (cache_hit)
            *cache_hit = false;
        if (!mayReuseDenseMTPModelContext(test_case))
            return std::nullopt;

        const std::string key =
            denseMTPModelContextCampaignKey(test_case, model_path);
        auto &cache = denseMTPModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.contract && cache.key == key)
        {
            if (cache_hit)
                *cache_hit = true;
            return cache.contract;
        }
        if (cache.contract)
        {
            llaminar::v2::kernels::KernelFactory::clearCache();
            cache.contract.reset();
            cache.key.clear();
        }
        return std::nullopt;
    }

    /**
     * @brief Publish the first initialized runner's reusable model authority.
     */
    inline bool publishDenseMTPModelContext(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const ModelContextReuseContract &contract,
        std::string *error)
    {
        if (!mayReuseDenseMTPModelContext(test_case) || !contract.context)
        {
            if (error)
                *error = "cannot publish an ineligible dense MTP model context";
            return false;
        }
        if (contract.context->path() != model_path)
        {
            if (error)
                *error = "production runner returned a different model authority";
            return false;
        }

        const std::string key =
            denseMTPModelContextCampaignKey(test_case, model_path);
        auto &cache = denseMTPModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.contract)
        {
            if (cache.key == key &&
                cache.contract->context == contract.context)
            {
                return true;
            }
            if (error)
                *error = "dense MTP campaign attempted to replace a live model authority";
            return false;
        }
        cache.key = key;
        cache.contract = contract;
        return true;
    }

    /**
     * @brief Retire the dense campaign's exclusive final prepared-model owner.
     *
     * Production campaign binaries terminate with `_exit` after explicitly
     * shutting down MPI and backend singletons, so process-static destructors
     * are intentionally bypassed.  This function creates the missing typed
     * final-owner edge before that shutdown: prepared weights and sealed
     * workspace backing are released, and every GPU runtime generation is
     * reset and certified through the same core API used by JIT eviction.
     *
     * @param error Receives the exact ownership or backend failure.
     * @return True when the slot was empty or retired completely.
     */
    inline bool releaseDenseMTPModelContextCampaignCache(
        std::string *error)
    {
        if (error)
            error->clear();
        auto &cache = denseMTPModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.contract)
        {
            cache.key.clear();
            return true;
        }

        try
        {
            llaminar::v2::kernels::KernelFactory::clearCache();
            (void)retireExclusiveModelContextReuseContract(
                *cache.contract);
            cache.contract.reset();
            cache.key.clear();
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
            {
                *error =
                    "dense MTP campaign final model retirement failed: " +
                    std::string(exception.what());
            }
            return false;
        }
    }

    /**
     * @brief Return the complete dense predictor checkpoint surface.
     *
     * ``FFN_SWIGLU`` is intentionally absent: production fuses that activation
     * into the down-projection kernel and exposes its gate/up inputs plus final
     * projection output.  Requiring a test-only materialization would alter the
     * optimized graph being certified.
     */
    inline const std::vector<std::string_view> &
    qwen36DenseMTPSidecarStageSuffixes()
    {
        static const std::vector<std::string_view> suffixes(
            kQwen36DenseMTPModelStageSuffixes.begin(),
            kQwen36DenseMTPModelStageSuffixes.end());
        return suffixes;
    }

    /**
     * @brief Run one real-weight production MTP checkpoint/CSV campaign cell.
     *
     * The runner executes its ordinary captured sidecar, verifier, sampling,
     * and device-state publication path.  The harness maps only the graph
     * context selected by the live depth controller to recursive MTP0..MTP2
     * CPU/FP32 Hugging Face checkpoints generated from the same GGUF nextn
     * weights.  Token parity, stage metrics, depth telemetry, full-capture
     * evidence, and all seven canonical CSVs are fail-closed requirements.
     */
    inline void runDenseMTPCheckpointProductionParity(
        DensePrefixRestoreParityCase test_case,
        int decode_token_budget,
        int mtp_draft_tokens,
        MTPDepthPolicyConfig depth_policy = {})
    {
        const auto campaign_started_at = std::chrono::steady_clock::now();
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });
        ASSERT_GT(decode_token_budget, 0);
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);

        test_case.decode_steps = std::max(
            test_case.decode_steps,
            std::max(decode_token_budget, 8));
        test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";

        if (auto skip_reason = densePrefixParitySkipReason(test_case))
            GTEST_SKIP() << *skip_reason;
        const std::string model_path = firstModelEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::is_regular_file(model_path))
            GTEST_SKIP() << test_case.name << " model not found: " << model_path;

        const std::filesystem::path metadata_path = firstEnvOrDefault(
            test_case.metadata_envs,
            test_case.default_metadata_path);
        ensurePyTorchDecodeSnapshots(
            test_case,
            model_path,
            metadata_path,
            /*require_mtp_sidecar_snapshots=*/true);
        const std::filesystem::path sidecar_reference_dir =
            metadata_path.parent_path();
        const std::vector<int32_t> prompt_tokens =
            readTokenListFromMetadata(metadata_path, "token_ids");
        const std::vector<int32_t> expected_tokens =
            readTokenListFromMetadata(metadata_path, "decode_tokens");
        ASSERT_FALSE(prompt_tokens.empty());
        ASSERT_GE(
            expected_tokens.size(),
            static_cast<size_t>(decode_token_budget));

        const std::filesystem::path full_reference_dir =
            "pytorch_qwen36_dense_singledevice_snapshots";
        const std::filesystem::path full_reference_metadata =
            full_reference_dir / "metadata.txt";
        ASSERT_TRUE(std::filesystem::is_regular_file(full_reference_metadata))
            << "Dense full-checkpoint reference is missing: "
            << full_reference_metadata;
        std::string full_identity_error;
        ASSERT_TRUE(qwen36ReferenceIdentityMatches(
            full_reference_metadata,
            model_path,
            test_case.prompt,
            &full_identity_error))
            << "Dense full-checkpoint reference is unauthenticated: "
            << full_identity_error;
        const std::vector<int32_t> artifact_prefill_tokens =
            readTokenListFromMetadata(full_reference_metadata, "token_ids");
        ASSERT_FALSE(artifact_prefill_tokens.empty());

        const ModelContextConfig metadata_context_config{
            .strategy = WeightDistributionStrategy::REPLICATED,
            .use_mmap = true,
            .payload_access_pattern =
                ModelPayloadAccessPattern::DeviceStaging,
        };
        auto metadata_context =
            ModelContext::create(model_path, metadata_context_config);
        ASSERT_NE(metadata_context, nullptr)
            << "Could not inspect dense GDN checkpoint geometry";
        const ParityGDNHeadConfig gdn_config =
            parityGDNHeadConfigFromModel(metadata_context.get());
        /*
         * Dense Qwen3.6 uses the reference head order directly.  The same
         * typed configuration is still passed to the common recorder so a
         * future dense checkpoint with unequal heads cannot accidentally
         * inherit guessed geometry from the test name.
         */
        EXPECT_FALSE(gdn_config.is_moe);
        metadata_context.reset();

        const bool dynamic_depth =
            depth_policy.mode == MTPDepthPolicyMode::Dynamic;
        const std::string artifact_backend =
            test_case.name +
            (dynamic_depth
                 ? "_MTP_DYNAMIC"
                 : "_MTP_DEPTH" + std::to_string(mtp_draft_tokens));
        Qwen36CheckpointArtifactRecorder artifact_recorder(
            artifact_backend,
            /*prefill_cosine_threshold=*/0.96f,
            /*decode_cosine_threshold=*/0.98f,
            /*logit_kl_threshold=*/0.08f);

        PerfStatsCollector::reset();
        auto factory = createOrchestrationRunnerFactory();
        const OrchestrationConfig config = makeDensePrefixRestoreConfig(
            test_case,
            model_path,
            /*enable_prefix_cache=*/false,
            /*block_size=*/2,
            /*enable_mtp=*/true,
            mtp_draft_tokens,
            depth_policy);

        const bool reuse_enabled =
            mayReuseDenseMTPModelContext(test_case);
        bool model_context_reused = false;
        std::string model_context_error;
        auto reuse_contract = findDenseMTPModelContext(
            test_case,
            model_path,
            &model_context_reused);
        auto mtp = reuse_contract
                       ? factory->createFromOrchestrationConfig(
                             config,
                             *reuse_contract)
                       : factory->createFromOrchestrationConfig(config);
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();
        if (reuse_enabled && !model_context_reused)
        {
            reuse_contract = mtp->modelContextReuseContract();
            ASSERT_TRUE(reuse_contract.has_value())
                << "Dense production runner exposed no prepared-weight reuse contract";
            ASSERT_TRUE(publishDenseMTPModelContext(
                test_case,
                model_path,
                *reuse_contract,
                &model_context_error))
                << model_context_error;
        }
        if (reuse_enabled)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                model_context_reused
                    ? "dense_parity_campaign_model_context_cache_hits"
                    : "dense_parity_campaign_model_context_cache_misses",
                1.0,
                "setup",
                mtp->primaryDeviceId().toString(),
                {{"depth", std::to_string(mtp_draft_tokens)},
                 {"policy", dynamic_depth ? "dynamic" : "fixed"}});
        }

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        mtp->setSamplingParams(greedy);
        mtp->setSkipLogitsGatherPrefill(false);
        mtp->setSkipLogitsGatherDecode(false);
        mtp->enableSnapshotCapture();

        ASSERT_TRUE(mtp->prefill(artifact_prefill_tokens)) << mtp->lastError();
        recordQwen36PrefillArtifacts(
            artifact_recorder,
            *mtp,
            full_reference_dir,
            artifact_prefill_tokens.size(),
            gdn_config);
        mtp->clearSnapshots();
        mtp->clearCache();

        ASSERT_TRUE(mtp->prefill(prompt_tokens)) << mtp->lastError();
        const bool host_owned_sidecar = mtp->primaryDeviceId().is_cpu();
        std::vector<int32_t> emitted_tokens;
        int speculative_transaction_count = 0;
        int observation_index = 0;

        struct ActiveCheckpointContext
        {
            std::string production_prefix;
            int reference_depth = 0;
        };

        while (static_cast<int>(emitted_tokens.size()) < decode_token_budget)
        {
            const int remaining =
                decode_token_budget - static_cast<int>(emitted_tokens.size());
            const auto before = mtp->prefixStateProbe();
            const int reference_step =
                mtpParityReferenceStepForConditionPosition(
                    before.mtp_next_condition_position,
                    static_cast<int>(prompt_tokens.size()));
            ASSERT_GE(reference_step, 0)
                << "Production MTP transaction began before the authenticated "
                   "prefill boundary: next_condition_position="
                << before.mtp_next_condition_position
                << " prompt_tokens=" << prompt_tokens.size();
            int selected_depth = before.mtp_current_depth;
            if (selected_depth <= 0)
                selected_depth = mtp_draft_tokens;
            selected_depth = std::clamp(
                selected_depth,
                1,
                mtp_draft_tokens);

            mtp->clearSnapshots();
            mtp->setDecodeStepTokenBudget(
                std::min(remaining, mtp_draft_tokens + 1));
            GenerationResult step = mtp->decodeStep();
            mtp->setDecodeStepTokenBudget(0);
            ASSERT_TRUE(step.error.empty()) << step.error;
            ASSERT_FALSE(step.tokens.empty());
            emitted_tokens.insert(
                emitted_tokens.end(),
                step.tokens.begin(),
                step.tokens.end());
            ASSERT_LE(
                emitted_tokens.size(),
                static_cast<size_t>(decode_token_budget));

            const auto after = mtp->prefixStateProbe();
            const MTPParityTransactionCounters before_counters{
                .draft_steps = before.mtp_draft_steps,
                .verifier_runs = before.mtp_verifier_runs,
            };
            const MTPParityTransactionCounters after_counters{
                .draft_steps = after.mtp_draft_steps,
                .verifier_runs = after.mtp_verifier_runs,
            };
            const auto activity = classifyMTPParityTransactionActivity(
                before_counters,
                after_counters);
            ASSERT_NE(activity, MTPParityTransactionActivity::Inconsistent)
                << "Dense MTP draft/verifier counters disagree around decodeStep";

            std::vector<ActiveCheckpointContext> active_contexts;
            if (activity == MTPParityTransactionActivity::Speculative)
            {
                const uint64_t transaction_delta =
                    mtpParityExecutedTransactionCount(
                        before_counters,
                        after_counters);
                const uint64_t draft_delta =
                    mtpParityAttemptedDraftTokenCount(
                        before_counters,
                        after_counters);
                ASSERT_GE(transaction_delta, 1u);
                ASSERT_GE(draft_delta, transaction_delta);
                ASSERT_LE(
                    draft_delta,
                    transaction_delta *
                        static_cast<uint64_t>(mtp_draft_tokens));

                const int last_depth =
                    after.mtp_last_transaction_draft_depth;
                ASSERT_GE(last_depth, 1);
                ASSERT_LE(last_depth, mtp_draft_tokens);
                int captured_depth = selected_depth;
                if (host_owned_sidecar)
                {
                    ASSERT_EQ(transaction_delta, 1u)
                        << "CPU snapshot observation must own one transaction";
                    captured_depth = last_depth;
                }
                else if (!dynamic_depth)
                {
                    ASSERT_EQ(last_depth, selected_depth)
                        << "Fixed-depth device controller changed depth in-call";
                }
                speculative_transaction_count +=
                    static_cast<int>(transaction_delta);

                active_contexts.push_back({
                    .production_prefix = std::string(
                        host_owned_sidecar
                            ? mtpParityCheckpointContextPrefix(
                                  MTPParityCheckpointContext::
                                      HostConditionTokenLivePosition)
                            : reference_step == 0
                                  ? mtpParityCheckpointContextPrefix(
                                        MTPParityCheckpointContext::
                                            DeviceTargetTokenLivePosition)
                                  : mtpParityCheckpointContextPrefix(
                                        MTPParityCheckpointContext::
                                            DeviceResidentLogicalState)),
                    .reference_depth = 0,
                });
                if (captured_depth > 1)
                {
                    active_contexts.push_back({
                        .production_prefix = std::string(
                            host_owned_sidecar
                                ? mtpParityCheckpointContextPrefix(
                                      MTPParityCheckpointContext::
                                          HostChainedDraftLivePosition)
                                : mtpParityCheckpointContextPrefix(
                                      MTPParityCheckpointContext::
                                          DeviceChainedTokenLivePosition)),
                        .reference_depth = captured_depth - 1,
                    });
                }
            }

            const auto snapshots = captureDenseStageSnapshots(*mtp);
            for (const auto &context : active_contexts)
            {
                SCOPED_TRACE(
                    "dense MTP observation=" +
                    std::to_string(observation_index) +
                    " context=" + context.production_prefix +
                    " reference_depth=" +
                    std::to_string(context.reference_depth));
                for (const std::string_view suffix :
                     qwen36DenseMTPSidecarStageSuffixes())
                {
                    const std::string production_key =
                        context.production_prefix + "MTP0_" +
                        std::string(suffix);
                    const auto production = snapshots.find(production_key);
                    ASSERT_NE(production, snapshots.end())
                        << "Selected production sidecar omitted "
                        << production_key;

                    const std::string reference_stage =
                        "MTP" + std::to_string(context.reference_depth) +
                        "_" + std::string(suffix);
                    const std::string reference_key =
                        "decode_step" + std::to_string(reference_step) +
                        "_" + reference_stage;
                    const std::vector<float> reference =
                        loadDensePyTorchSnapshot(
                            sidecar_reference_dir,
                            reference_key);
                    ASSERT_FALSE(reference.empty())
                        << "Authenticated dense MTP pack omitted "
                        << reference_key;
                    ASSERT_EQ(production->second.data.size(), reference.size())
                        << production_key << " shape differs from "
                        << reference_key;

                    const StageComparisonResult stage_result =
                        compareParityTensorData(
                            production->second.data.data(),
                            reference.data(),
                            reference.size(),
                            reference_stage,
                            /*cosine_threshold=*/0.98f);
                    EXPECT_TRUE(stage_result.passed)
                        << production_key << " failed dense MTP checkpoint "
                        << "parity against " << reference_key
                        << " cosine=" << stage_result.cosine_similarity
                        << " max_abs=" << stage_result.max_abs_diff;

                    artifact_recorder.requireDecodeStageSuffix(reference_stage);
                    artifact_recorder.recordDecode(
                        reference_step,
                        /*layer=*/-1,
                        reference_stage,
                        production->second.data,
                        reference);
                    if (suffix == "LM_HEAD")
                    {
                        artifact_recorder.recordDecodeToken(
                            reference_step,
                            denseArgmaxToken(
                                production->second.data.data(),
                                static_cast<int>(
                                    production->second.data.size())),
                            denseArgmaxToken(
                                reference.data(),
                                static_cast<int>(reference.size())));
                    }
                }
            }
            ++observation_index;
        }

        const auto mtp_state = mtp->prefixStateProbe();
        const bool graph_execution = mtp->executorStats() != nullptr;
        const std::string production_device =
            mtp->primaryDeviceId().toString();
        const auto production_records = PerfStatsCollector::snapshot(
            {"forward_graph", "mtp", "weight_loading"});
        mtp->disableSnapshotCapture();
        mtp->shutdown();

        ASSERT_EQ(
            emitted_tokens,
            std::vector<int32_t>(
                expected_tokens.begin(),
                expected_tokens.begin() + decode_token_budget));
        EXPECT_FALSE(mtp_state.mtp_bypassed) << mtp_state.mtp_bypass_reason;
        EXPECT_GE(mtp_state.mtp_verifier_runs, 1u);
        EXPECT_GE(speculative_transaction_count, 1);
        if (dynamic_depth)
        {
            EXPECT_TRUE(mtp_state.mtp_request.adaptive_depth_enabled);
            EXPECT_EQ(mtp_state.mtp_request.depth_policy_mode, "dynamic");
            EXPECT_GE(mtp_state.mtp_depth_policy_windows, 1u);
            EXPECT_GE(mtp_state.mtp_depth_policy_updates, 1u);
            EXPECT_GE(mtp_state.mtp_depth_policy_demotions, 1u);
            EXPECT_EQ(mtp_state.mtp_min_depth, depth_policy.min_depth);
            EXPECT_EQ(mtp_state.mtp_max_depth, depth_policy.max_depth);
            EXPECT_GE(mtp_state.mtp_current_depth, depth_policy.min_depth);
            EXPECT_LE(mtp_state.mtp_current_depth, depth_policy.max_depth);
        }
        else
        {
            EXPECT_FALSE(mtp_state.mtp_request.adaptive_depth_enabled);
            EXPECT_EQ(mtp_state.mtp_request.depth_policy_mode, "fixed");
            EXPECT_EQ(mtp_state.mtp_current_depth, mtp_draft_tokens);
            EXPECT_EQ(mtp_state.mtp_max_depth, mtp_draft_tokens);
            EXPECT_EQ(mtp_state.mtp_depth_policy_updates, 0u);
        }

        if (reuse_enabled)
        {
            const auto counter_sum = [&](std::string_view name)
            {
                double total = 0.0;
                for (const auto &record : production_records)
                {
                    if (record.kind == PerfStatRecord::Kind::Counter &&
                        record.domain == "mtp" && record.name == name)
                        total += record.value;
                }
                return total;
            };
            EXPECT_EQ(
                counter_sum(
                    "dense_parity_campaign_model_context_cache_hits"),
                model_context_reused ? 1.0 : 0.0);
            EXPECT_EQ(
                counter_sum(
                    "dense_parity_campaign_model_context_cache_misses"),
                model_context_reused ? 0.0 : 1.0);
        }

        artifact_recorder.finalize();

        const ProductionParityExecutionTopology execution_topology =
            classifyProductionParityExecutionTopology(test_case.devices);
        const ProductionParityEvidence production_evidence =
            collectProductionParityEvidence(
                production_records,
                graph_execution,
                execution_topology,
                resolveProductionParityGraphContract(
                    execution_topology,
                    /*is_campaign_authority=*/true,
                    execution_topology !=
                        ProductionParityExecutionTopology::CPUOnly),
                model_context_reused,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - campaign_started_at)
                    .count());
        const ProductionParityGraphCertification graph_certification =
            certifyProductionParityGraphExecution(production_evidence);
        EXPECT_TRUE(
            graph_certification ==
            ProductionParityGraphCertification::Certified)
            << "Dense MTP graph certification failed: topology='"
            << productionParityExecutionTopologyName(execution_topology)
            << "' result='"
            << productionParityGraphCertificationName(graph_certification)
            << "'.\n"
            << PerfStatsCollector::summaryString({"forward_graph", "mtp"});
        if (production_evidence.usesHomogeneousGPU())
        {
            EXPECT_TRUE(production_evidence.device_generation_controller)
                << "Dense MTP parity published no device-generation controller evidence";
            EXPECT_TRUE(production_evidence.generation_loop_certified)
                << "Dense MTP parity used outer-loop policy '"
                << productionDeviceGenerationPolicyName(
                       production_evidence.generation_execution_policy)
                << "' without satisfying its backend-specific authority and dispatch proof: "
                << production_evidence.generation_certification_detail;
            EXPECT_TRUE(
                productionParityHasRequiredGenerationGraph(
                    production_evidence))
                << "Dense MTP parity did not certify the captured graph body required by its generation policy";
            expectDenseHomogeneousGPUFullGraphReplay(
                test_case,
                production_records,
                "real-weight dense MTP checkpoint campaign");
            EXPECT_TRUE(
                production_evidence.decode_graph_capture ||
                production_evidence.decode_graph_replay)
                << "Dense MTP campaign produced no native capture/replay evidence";
        }
        EXPECT_TRUE(ParityCSVArtifactWriter::writeProductionPath(
            ParityCSVArtifactWriter::resultsDir(),
            artifact_backend,
            production_device,
            production_evidence))
            << "Failed to write dense MTP production_path.csv";
        PerfStatsCollector::reset();
    }

    inline void runDenseMTPParity(
        DensePrefixRestoreParityCase test_case,
        bool enable_prefix_cache,
        int mtp_draft_tokens = 1,
        MTPDepthPolicyConfig depth_policy = {},
        int terminal_partial_prefix_tokens = 0)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        if (denseCaseUsesHomogeneousGPU(test_case))
        {
            // Short requests can finish before a repeated verifier geometry
            // crosses graph warmup. Use the established continuation fixture so
            // every homogeneous-GPU MTP cell proves an executable full graph,
            // rather than stopping at the weaker "capturable plan" assertion.
            test_case.decode_steps = std::max(test_case.decode_steps, 8);
            test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
            test_case.default_metadata_path =
                "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";
        }
        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);
        ASSERT_GE(terminal_partial_prefix_tokens, 0);
        ASSERT_TRUE(
            terminal_partial_prefix_tokens == 0 || enable_prefix_cache)
            << "terminal-partial MTP restore requires prefix cache";

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        const bool terminal_partial_restore =
            terminal_partial_prefix_tokens > 0;
        if (terminal_partial_restore)
        {
            ASSERT_LT(
                terminal_partial_prefix_tokens,
                static_cast<int>(prompt_tokens.size()));
        }
        const int block_size = terminal_partial_restore
                                   ? terminal_partial_prefix_tokens + 1
                                   : (enable_prefix_cache
                                          ? static_cast<int>(prompt_tokens.size())
                                          : 2);
        std::unique_ptr<ScopedEnvironmentValues> partial_prefix_graph_env;
        if (terminal_partial_restore)
        {
            partial_prefix_graph_env =
                std::make_unique<ScopedEnvironmentValues>(
                    std::initializer_list<std::pair<const char *, const char *>>{
                        {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
                    });
        }
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

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

        const std::vector<int32_t> first_prompt = terminal_partial_restore
                                                      ? std::vector<int32_t>(
                                                            prompt_tokens.begin(),
                                                            prompt_tokens.begin() +
                                                                terminal_partial_prefix_tokens)
                                                      : prompt_tokens;
        if (terminal_partial_restore)
        {
            ASSERT_NE(first_prompt.size() % static_cast<size_t>(block_size), 0u)
                << "MTP prefix seed must terminate inside its cache block";
        }

        PerfStatsCollector::reset();
        auto first = mtp->generate(first_prompt, test_case.decode_steps, greedy);
        const auto after_first = mtp->prefixStateProbe();
        const auto first_records =
            PerfStatsCollector::snapshot(
                {"mtp", "tp_collective_runtime", "forward_graph"});
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_FALSE(first.tokens.empty());
        if (!terminal_partial_restore)
        {
            ASSERT_EQ(first.tokens.size(), expected_tokens.size());
            EXPECT_EQ(first.tokens, expected_tokens);
        }
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
        expectDenseGreedyMTPPublicationPath(
            test_case,
            first_records,
            test_case.name + " first request");
        expectDenseHomogeneousGPUFullGraphReplay(
            test_case,
            first_records,
            test_case.name + " first request");

        if (!enable_prefix_cache)
        {
            mtp->shutdown();
            PerfStatsCollector::reset();
            return;
        }

        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);

        PerfStatsCollector::reset();
        auto second = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_second = mtp->prefixStateProbe();
        const auto second_records =
            PerfStatsCollector::snapshot(
                {"mtp", "prefix_cache", "tp_collective_runtime", "forward_graph"});
        mtp->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), expected_tokens.size());
        EXPECT_EQ(second.tokens, expected_tokens);
        EXPECT_TRUE(after_second.prefix_cache_ready);
        EXPECT_GE(after_second.prefix_cache_hits, 1u);
        if (terminal_partial_restore)
        {
            EXPECT_FALSE(after_second.prefix_request.hit);
            EXPECT_TRUE(after_second.prefix_request.partial_hit);
            EXPECT_EQ(
                after_second.prefix_request.matched_tokens,
                terminal_partial_prefix_tokens);
            EXPECT_FALSE(after_second.prefix_request.terminal_logits_restored);
            EXPECT_FALSE(after_second.prefix_request.terminal_hidden_restored);
            EXPECT_TRUE(denseHasPerfCounter(
                second_records,
                "prefix_cache",
                "terminal_partial_block_prefix_hits"));
        }
        else
        {
            EXPECT_TRUE(after_second.prefix_request.hit);
            EXPECT_FALSE(after_second.prefix_request.partial_hit);
            EXPECT_EQ(after_second.prefix_request.matched_tokens,
                      static_cast<int>(prompt_tokens.size()));
            EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
            EXPECT_TRUE(after_second.prefix_request.terminal_hidden_restored);
        }
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
        expectDenseGreedyMTPPublicationPath(
            test_case,
            second_records,
            test_case.name + " restored request");
        expectDenseHomogeneousGPUFullGraphReplay(
            test_case,
            second_records,
            test_case.name + " restored request");
        PerfStatsCollector::reset();
    }

    /**
     * @brief Prove an in-block terminal prefix restores shifted MTP KV exactly.
     *
     * The seed request ends at token three of a four-token cache block. The
     * second request extends that same block, forcing production lookup to use
     * the shorter terminal key, import recurrent and shifted-MTP state, and
     * prefill only the suffix before grouped verification resumes.
     */
    inline void runDenseMTPPartialTerminalPrefixRestoreParity(
        DensePrefixRestoreParityCase test_case)
    {
        runDenseMTPParity(
            std::move(test_case),
            /*enable_prefix_cache=*/true,
            /*mtp_draft_tokens=*/3,
            /*depth_policy=*/{},
            /*terminal_partial_prefix_tokens=*/3);
    }

    /**
     * @brief Prove temperature-zero penalties stay on a safe dense MTP path.
     *
     * This is intentionally a one-runner model-level regression.  Unit tests
     * own the exact no-MTP/MTP policy mechanics; this large-model row proves
     * that non-zero repetition penalties do not bypass MTP and that dense
     * Qwen3.6 still records the correct fail-closed verifier contract.  The
     * penalties are deliberately small so the existing PyTorch greedy fixture
     * remains a stable token oracle without doubling test time through a second
     * no-MTP large-model runner.
     */
    inline void runDensePenaltyGreedyMTPMatchesPyTorch(
        DensePrefixRestoreParityCase test_case,
        int mtp_draft_tokens = 3)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);

        test_case.name += " penalty-greedy MTP parity";

        ScopedEnvironmentValues perf_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_LT(
            static_cast<int>(prompt_tokens.size()) + test_case.decode_steps,
            test_case.max_seq_len);

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams penalty_greedy;
        penalty_greedy.temperature = 0.0f;
        penalty_greedy.presence_penalty = 0.01f;
        penalty_greedy.frequency_penalty = 0.005f;
        penalty_greedy.seed = 42;
        ASSERT_TRUE(penalty_greedy.has_penalties());

        auto mtp = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                false,
                2,
                true,
                mtp_draft_tokens));
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();

        PerfStatsCollector::reset();
        auto mtp_result = mtp->generate(
            prompt_tokens,
            test_case.decode_steps,
            penalty_greedy);
        const auto mtp_state = mtp->prefixStateProbe();
        const auto mtp_records = PerfStatsCollector::snapshot({"mtp"});
        mtp->shutdown();
        PerfStatsCollector::reset();

        ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
        ASSERT_EQ(mtp_result.tokens.size(), expected_tokens.size());
        EXPECT_TRUE(tokenSequencesMatch(
            mtp_result.tokens,
            expected_tokens,
            test_case.name + " MTP versus PyTorch fixture"))
            << "The small penalty values should preserve the existing greedy "
               "fixture while still exercising the penalty verifier policy.";
        EXPECT_FALSE(mtp_state.mtp_bypassed) << mtp_state.mtp_bypass_reason;
        EXPECT_GE(mtp_state.mtp_draft_steps, 1u);
        EXPECT_GE(mtp_state.mtp_verifier_runs, 1u);
        EXPECT_GE(mtp_state.mtp_verifier_token_count, 2u);

        const bool used_grouped_decode_equivalent_greedy_verifier =
            denseHasMTPPerfCounter(
                mtp_records,
                "grouped_decode_equivalent_greedy_verifier_runs");
        const bool used_retired_serial_replay =
            denseHasMTPPerfCounter(
                mtp_records,
                "decode_equivalent_sequential_verifier_runs");
        const bool used_all_position_publication =
            denseHasMTPPerfCounter(
                mtp_records,
                "all_position_state_publication_verifier_runs") &&
            denseHasMTPPerfCounter(mtp_records, "spec_state_publications");
        const bool selected_penalty_shared_verifier =
            denseHasMTPPerfRecordTag(
                mtp_records,
                "verifier_policy_selections",
                "reason",
                "greedy_penalties_use_grouped_decode_equivalent_outcome");

        if (denseCaseExpectsAllPositionSpecPublication(test_case))
        {
            EXPECT_TRUE(used_all_position_publication)
                << "Dense penalty-greedy MTP advertised all-position "
                   "publication support but did not use it.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_grouped_decode_equivalent_greedy_verifier)
                << "Dense penalty-greedy MTP should not use the shared "
                   "verifier once direct publication is proven.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_retired_serial_replay)
                << "Dense penalty-greedy MTP must not use retired row-serial "
                   "verifier replay.\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }
        else
        {
            EXPECT_TRUE(used_grouped_decode_equivalent_greedy_verifier)
                << "Dense penalty-greedy MTP must use the grouped "
                   "decode-equivalent verifier while direct publication is "
                   "not advertised.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_TRUE(selected_penalty_shared_verifier)
                << "Penalty-greedy policy selection did not record the "
                   "expected fail-closed verifier reason.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_all_position_publication)
                << "Dense penalty-greedy MTP must not publish from an "
                   "unproven all-position verifier.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_retired_serial_replay)
                << "Dense penalty-greedy MTP must not use retired row-serial "
                   "verifier replay.\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }
    }

    inline void runDenseMTPFirstTransactionLeavesSequentialState(
        DensePrefixRestoreParityCase test_case)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));

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

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        auto model_ctx = createQwen36ParityModelContext(model_path, device);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 3;

        auto runner = createInferenceRunner(
            model_ctx,
            nullptr,
            device,
            config);
        ASSERT_NE(runner, nullptr);
        runner->setSuppressTimeline(true);
        runner->setSkipLogitsGatherPrefill(true);
        runner->setSkipLogitsGatherDecode(true);

        constexpr int kCatchupTargetSampleSlot = 0;
        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())));
        int32_t prefill_sample = -1;
        ASSERT_TRUE(runner->sampleGreedyFromMainLogitsToDeviceTargetSlot(
            kCatchupTargetSampleSlot,
            &prefill_sample));
        ASSERT_EQ(prefill_sample, expected_tokens[0]);

        const int base_position = runner->get_position();
        const PrefixStateSnapshot base_checkpoint = runner->captureLivePrefixState();
        ASSERT_TRUE(base_checkpoint.valid);

        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {13, 271, 760, 3841};
        request.base_sidecar_position = base_position;
        request.allow_speculative_discard = true;
        request.verifier_path = "phase138_first_transaction_regression";
        request.verifier_base_checkpoint = &base_checkpoint;
        request.device_target_sample_slot = kCatchupTargetSampleSlot;

        auto sample_after_forward = [&](int32_t) -> int32_t
        {
            int32_t sampled = -1;
            return runner->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                       kCatchupTargetSampleSlot,
                       &sampled)
                       ? sampled
                       : -1;
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
        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
    }

    inline void runDenseNoMTPPhase138ContinuationMatchesPyTorch(
        DensePrefixRestoreParityCase test_case,
        int decode_steps = 8)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
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
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));

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

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        auto model_ctx = createQwen36ParityModelContext(model_path, device);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.mtp.enabled = false;

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
        DenseDecodeSnapshotComparisonPolicy snapshot_policy;
        snapshot_policy.layer_count = test_case.main_layers;
        if (isDenseGpuParityCase(test_case))
        {
            /*
             * This regression is about no-MTP continuation state around the
             * Phase 13.8 verifier window.  GPU native decode is quantized and,
             * at the <think> continuation row in this fixture, both CUDA and
             * ROCm diverge from the FP32 PyTorch hidden-vector trajectory at
             * layer 3 while still producing the exact greedy token stream.  Keep
             * this diagnostic strict for the graph prefix before that known
             * quantized-drift point; the classic Qwen3.6 parity suite owns the
             * full GPU math acceptance with KLD/cosine/top-k checks.
             */
            snapshot_policy.max_layer_count = 3;
            snapshot_policy.compare_final_outputs = false;
        }
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
                test_case.name,
                snapshot_policy);
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
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        // A single longer request is required here. It exercises repeated
        // dynamic-depth verifier shapes after graph warmup without resetting
        // runner state between requests and accidentally testing lifecycle
        // behavior instead of the production dynamic-depth path.
        test_case.decode_steps = std::max(test_case.decode_steps, 8);
        test_case.max_seq_len = std::max(test_case.max_seq_len, 128);
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_phase138_continuation_snapshots/metadata.txt";
        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });
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

        PerfStatsCollector::reset();
        auto first = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_first = mtp->prefixStateProbe();
        const auto first_records =
            PerfStatsCollector::snapshot(
                {"mtp", "tp_collective_runtime", "forward_graph"});
        mtp->shutdown();
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), expected_tokens.size());
        EXPECT_EQ(first.tokens, expected_tokens);
        EXPECT_FALSE(after_first.mtp_bypassed) << after_first.mtp_bypass_reason;
        EXPECT_TRUE(after_first.mtp_request.adaptive_depth_enabled);
        EXPECT_EQ(after_first.mtp_request.depth_policy_mode, "dynamic");
        EXPECT_GE(after_first.mtp_depth_policy_windows, 1u);
        EXPECT_GE(after_first.mtp_min_depth, 1);
        EXPECT_EQ(after_first.mtp_max_depth, adaptive_max_depth);
        EXPECT_GE(after_first.mtp_current_depth, 1);
        EXPECT_LE(after_first.mtp_current_depth, adaptive_max_depth);
        expectDenseGreedyMTPPublicationPath(
            test_case,
            first_records,
            test_case.name + " dynamic-depth first request");
        expectDenseHomogeneousGPUFullGraphReplay(
            test_case,
            first_records,
            test_case.name + " dynamic-depth long request");
        PerfStatsCollector::reset();
    }

    inline void runDenseNoMTPBenchmarkStyleFreshRunnerDeterminism(
        DensePrefixRestoreParityCase test_case,
        int decode_token_budget = 16,
        int reused_cycle_count = 2)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
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

        const std::string model_path = firstModelEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << model_path;
        }

        const DeviceId tokenizer_device = test_case.devices.empty()
                                              ? DeviceId::cpu()
                                              : test_case.devices.front().toLocalDeviceId();
        auto model_context = createQwen36ParityModelContext(
            model_path,
            tokenizer_device);
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
            << "Qwen3.6 dense benchmark-style no-MTP decode must be "
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
            << "Gathering logits for diagnostics must not change no-MTP decode tokens."
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
                << "clearCache() must reset no-MTP production decode state "
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
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
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

        model_path = firstModelEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << model_path;
        }

        const DeviceId tokenizer_device = test_case.devices.empty()
                                              ? DeviceId::cpu()
                                              : test_case.devices.front().toLocalDeviceId();
        auto model_context = createQwen36ParityModelContext(
            model_path,
            tokenizer_device);
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
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        test_case.name += " benchmark-style " + label + " MTP parity";
        test_case.prompt = qwen36DefaultBenchmarkPrompt();
        test_case.decode_steps = qwen36BenchmarkPromptStableExactDecodeSteps(test_case);
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
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
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

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        auto model_ctx = createQwen36ParityModelContext(model_path, device);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 3;

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
            denseVerifierRowSnapshotsByteIdentical(
                restored_snapshots,
                sequential_snapshots,
                "Restored one-row decode",
                1,
                0);
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

    inline void runDenseBenchmarkPromptKnownWindowPyTorchTokenParity(
        DensePrefixRestoreParityCase test_case)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        test_case.name += " dense benchmark-prompt known-window PyTorch token parity";
        test_case.prompt = qwen36DefaultBenchmarkPrompt();
        test_case.decode_steps = 128;
        test_case.max_seq_len = 768;
        test_case.metadata_envs = {
            "LLAMINAR_QWEN36_DENSE_BENCHMARK_PARITY_METADATA",
        };
        test_case.default_metadata_path =
            "pytorch_qwen36_dense_benchmark_prompt_snapshots/metadata.txt";

        // This is a teacher-forced long-decode diagnostic. GPU graph capture is
        // irrelevant here and only adds noise to the failing-row token
        // comparison; the production graph-captured decode path is covered by
        // the benchmark-style parity tests.
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

        // Stop just before the known backend-specific near-tie. With quantized
        // GEMM plus Q16_1 KV, token-exact parity should validate the stable
        // prefix and then document the near-tie boundary separately.
        const bool rocm_near_tie = isQwen36DenseROCmSingleDeviceCase(test_case);
        const bool cuda_near_tie = isQwen36DenseCUDASingleDeviceCase(test_case);
        const int kTargetDecodeStep = rocm_near_tie ? 5 : (cuda_near_tie ? 46 : 113);
        const int kExpectedTokenIndex = kTargetDecodeStep + 1;
        ASSERT_GT(static_cast<int>(expected_tokens.size()), kExpectedTokenIndex);
        if (rocm_near_tie)
        {
            ASSERT_EQ(expected_tokens[5], 3294)
                << "Benchmark prompt fixture changed; update ROCm near-tie diagnostic";
            ASSERT_EQ(expected_tokens[6], 11)
                << "Benchmark prompt fixture changed; update ROCm near-tie diagnostic";
            ASSERT_EQ(expected_tokens[7], 1092)
                << "Benchmark prompt fixture changed; update ROCm near-tie diagnostic";
        }
        else if (cuda_near_tie)
        {
            ASSERT_EQ(expected_tokens[45], 75318)
                << "Benchmark prompt fixture changed; update CUDA near-tie diagnostic";
            ASSERT_EQ(expected_tokens[46], 20271)
                << "Benchmark prompt fixture changed; update CUDA near-tie diagnostic";
            ASSERT_EQ(expected_tokens[47], 92217)
                << "Benchmark prompt fixture changed; update CUDA near-tie diagnostic";
            ASSERT_EQ(expected_tokens[48], 15676)
                << "Benchmark prompt fixture changed; update CUDA near-tie diagnostic";
            ASSERT_EQ(expected_tokens[49], 3983)
                << "Benchmark prompt fixture changed; update CUDA near-tie diagnostic";
        }
        else
        {
            ASSERT_EQ(expected_tokens[111], 258)
                << "Benchmark prompt fixture changed; update known-window diagnostic";
            ASSERT_EQ(expected_tokens[112], 10608)
                << "Benchmark prompt fixture changed; update known-window diagnostic";
            ASSERT_EQ(expected_tokens[113], 20271)
                << "Benchmark prompt fixture changed; update known-window diagnostic";
            ASSERT_EQ(expected_tokens[114], 92217)
                << "Benchmark prompt fixture changed; update known-window diagnostic";
            ASSERT_EQ(expected_tokens[115], 48567)
                << "Benchmark prompt fixture changed; update known near-tie diagnostic";
        }

        ASSERT_LT(
            static_cast<int>(prompt_tokens.size()) + kExpectedTokenIndex + 1,
            test_case.max_seq_len);

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        auto model_ctx = createQwen36ParityModelContext(model_path, device);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.mtp.enabled = false;

        auto runner = createInferenceRunner(
            model_ctx,
            nullptr,
            device,
            config);
        ASSERT_NE(runner, nullptr);
        runner->setSuppressTimeline(true);
        runner->setSkipLogitsGatherPrefill(false);
        runner->setSkipLogitsGatherDecode(false);

        std::vector<int32_t> actual_tokens;
        actual_tokens.reserve(static_cast<size_t>(kExpectedTokenIndex + 1));

        ASSERT_TRUE(runner->forward(prompt_tokens.data(), static_cast<int>(prompt_tokens.size())));
        const int32_t prefill_sample = runner->sampleGreedyOnDevice();
        actual_tokens.push_back(prefill_sample);
        ASSERT_EQ(prefill_sample, expected_tokens[0])
            << "Prefill token already diverged from PyTorch; top-k="
            << denseTopKSummary(runner->logits(), runner->vocab_size());

        for (int step = 0; step <= kTargetDecodeStep; ++step)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(step)];
            ASSERT_TRUE(runner->forward(&token, 1))
                << "Failed to teacher-force benchmark prompt token at decode step "
                << step;
            const int32_t sampled = runner->sampleGreedyOnDevice();
            actual_tokens.push_back(sampled);
            const size_t token_index = static_cast<size_t>(step + 1);
            ASSERT_LT(token_index, expected_tokens.size());
            ASSERT_EQ(sampled, expected_tokens[static_cast<size_t>(step + 1)])
                << "Teacher-forced dense decode diverged from PyTorch token metadata"
                << "\nstep=" << step
                << "\ninput_token=" << token
                << "\nactual_next=" << sampled
                << "\nexpected_next=" << expected_tokens[static_cast<size_t>(step + 1)]
                << "\nactual window:   "
                << formatTokenWindow(actual_tokens, actual_tokens.size() - 1)
                << "\nexpected window: "
                << formatTokenWindow(expected_tokens, token_index)
                << "\ntop-k=" << denseTopKSummary(runner->logits(), runner->vocab_size());
        }

        EXPECT_TRUE(tokenSequencesMatch(
            actual_tokens,
            std::vector<int32_t>(
                expected_tokens.begin(),
                expected_tokens.begin() + actual_tokens.size()),
            "dense benchmark prompt known-window"));

        if (rocm_near_tie || cuda_near_tie)
        {
            const int near_tie_step = rocm_near_tie ? 6 : 47;
            const int32_t input_token = expected_tokens[near_tie_step];
            const int32_t pytorch_token = expected_tokens[near_tie_step + 1];
            const int32_t observed_backend_token =
                rocm_near_tie ? 4338 : 1061;
            ASSERT_TRUE(runner->forward(&input_token, 1))
                << "Failed to teacher-force backend near-tie row";

            const int32_t sampled = runner->sampleGreedyOnDevice();
            const float *logits = runner->logits();
            ASSERT_NE(logits, nullptr);
            ASSERT_TRUE(sampled == pytorch_token || sampled == observed_backend_token)
                << "Backend near-tie row changed to an unexpected token"
                << "\nstep=" << near_tie_step
                << "\ninput_token=" << input_token
                << "\nsampled=" << sampled
                << "\nexpected_pytorch=" << pytorch_token
                << "\nobserved_backend=" << observed_backend_token
                << "\ntop-k=" << denseTopKSummary(logits, runner->vocab_size());
            const float max_known_boundary_gap = rocm_near_tie ? 0.05f : 0.10f;
            EXPECT_LT(
                std::abs(logits[observed_backend_token] - logits[pytorch_token]),
                max_known_boundary_gap)
                << "Benchmark prompt row is no longer a documented quantized/PyTorch boundary"
                << "\nstep=" << near_tie_step
                << "\nbackend_token_logit=" << logits[observed_backend_token]
                << "\npytorch_token_logit=" << logits[pytorch_token]
                << "\nmax_known_boundary_gap=" << max_known_boundary_gap
                << "\ntop-k=" << denseTopKSummary(logits, runner->vocab_size());
        }

        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
    }

    inline void runDenseMTPEnabledForwardOnlyMatchesNoMTP(
        DensePrefixRestoreParityCase test_case,
        int decode_steps = 128)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
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

        const std::string model_path = firstModelEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << model_path;
        }

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cuda(0)
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        auto model_ctx = createQwen36ParityModelContext(model_path, device);
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

        auto make_config = [&](bool enable_mtp)
        {
            InferenceRunnerConfig config;
            config.max_seq_len = test_case.max_seq_len;
            config.batch_size = 1;
            config.force_graph = true;
            config.activation_precision = ActivationPrecision::FP32;
            config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
            config.mtp.enabled = enable_mtp;
            config.mtp.draft_tokens = 1;
            return config;
        };

        auto run_forward_only = [&](bool enable_mtp,
                                    const std::vector<int32_t> *teacher_tokens,
                                    const std::vector<int32_t> *expected_outputs = nullptr)
            -> std::vector<int32_t>
        {
            std::vector<int32_t> generated;
            auto runner_model_ctx = createQwen36ParityModelContext(
                model_path,
                device);
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
                if (expected_outputs && step < static_cast<int>(expected_outputs->size()))
                {
                    const int32_t expected =
                        (*expected_outputs)[static_cast<size_t>(step)];
                    if (sampled != expected)
                    {
                        ADD_FAILURE()
                            << "MTP-enabled forward-only diverged during teacher-forced decode"
                            << "\nstep=" << step
                            << "\ninput_token=" << driver
                            << "\nactual_next=" << sampled
                            << "\nexpected_next=" << expected
                            << "\nactual window:   "
                            << formatTokenWindow(generated, generated.size() - 1)
                            << "\nexpected window: "
                            << formatTokenWindow(*expected_outputs, static_cast<size_t>(step))
                            << "\ntop-k=" << denseTopKSummary(runner->logits(), runner->vocab_size());
                        break;
                    }
                }
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
            run_forward_only(/*enable_mtp=*/true, &baseline, &baseline);
        ASSERT_EQ(mtp_enabled.size(), baseline.size());
        EXPECT_TRUE(tokenSequencesMatch(
            mtp_enabled,
            baseline,
            "MTP-enabled forward-only"));
    }

    inline void runDenseMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        const DensePrefixRestoreParityCase &test_case,
        int verifier_row_count)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        ASSERT_GE(verifier_row_count, 1)
            << "decode-equivalent verifier-row proof must exercise at least one row";
        ASSERT_LE(verifier_row_count, 4)
            << "Phase 9.7 production proof currently targets M=1..4";

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GE(expected_tokens.size(), 2u)
            << "decode-equivalent verifier row proof needs two setup tokens";

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        auto model_ctx = createQwen36ParityModelContext(model_path, device);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.mtp.enabled = true;
        config.mtp.draft_tokens = verifier_row_count;

        auto runner = createInferenceRunner(model_ctx, nullptr, device, config);
        ASSERT_NE(runner, nullptr);
        ASSERT_GT(runner->vocab_size(), 0);

        constexpr int kSetupTargetSampleSlot = 0;
        constexpr int kCatchupTargetSampleSlot = 1;
        auto sample_current_to_slot =
            [&](const char *label, int target_sample_slot) -> int32_t
        {
            int32_t sampled = -1;
            const bool sampled_ok =
                device.is_gpu()
                    ? runner->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                          target_sample_slot,
                          &sampled)
                    : ((sampled = runner->sampleGreedyOnDevice()) >= 0);
            if (!sampled_ok || sampled < 0)
            {
                ADD_FAILURE()
                    << "device-owned greedy sampling failed after " << label;
                return -1;
            }
            return sampled;
        };
        auto sample_current = [&](const char *label) -> int32_t
        {
            return sample_current_to_slot(label, kSetupTargetSampleSlot);
        };

        runner->setSuppressTimeline(true);
        runner->enableSnapshotCapture();

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())))
            << "prefill forward failed";
        EXPECT_EQ(sample_current("prefill"), expected_tokens[0]);

        int32_t token_after_setup = -1;
        /*
         * Match the production shifted-MTP ordering exactly. The sampled token
         * remains in the persistent target slot while the sidecar consumes it;
         * only after that event-published append do we advance the main graph.
         * Delaying both sidecar appends until after serial setup would lose the
         * token-to-terminal-hidden relationship and would require a host replay.
         */
        for (int i = 0; i < 2; ++i)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(i)];
            const int setup_sidecar_position = runner->get_position();
            const bool shifted_row_committed =
                device.is_gpu()
                    ? runner->commitMTPShiftedRowFromDeviceTargetSample(
                          kSetupTargetSampleSlot,
                          /*already_appended_tokens=*/0,
                          /*allow_speculative_discard=*/true)
                    : runner->commitMTPShiftedRowFromCurrentTerminalHidden(
                          token,
                          /*already_appended_tokens=*/0,
                          /*allow_speculative_discard=*/true,
                          setup_sidecar_position);
            ASSERT_TRUE(shifted_row_committed)
                << "failed to publish shifted MTP setup row " << i
                << " from the production condition-token owner";
            ASSERT_TRUE(runner->forward(&token, 1))
                << "serial setup forward failed at token index " << i;
            token_after_setup = sample_current("serial setup");
        }
        ASSERT_GE(token_after_setup, 0)
            << "serial setup must produce the first verifier input token";
        if (device.is_gpu())
        {
            ASSERT_TRUE(runner->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                kCatchupTargetSampleSlot,
                /*out_token=*/nullptr))
                << "the verifier-base condition token must be checkpointed in "
                   "a device-owned consumer slot before diagnostic serial rows "
                   "overwrite the ordinary sampling slot";
        }

        const PrefixStateSnapshot verifier_base = runner->captureLivePrefixState();
        ASSERT_TRUE(verifier_base.valid);

        std::vector<int32_t> verifier_tokens;
        verifier_tokens.reserve(static_cast<size_t>(verifier_row_count));
        int32_t next_verifier_token = token_after_setup;
        for (int i = 0; i < verifier_row_count; ++i)
        {
            verifier_tokens.push_back(next_verifier_token);
            ASSERT_TRUE(runner->forward(&next_verifier_token, 1))
                << "serial verifier-token extension failed at row " << i;
            next_verifier_token =
                sample_current("serial verifier-token extension");
            ASSERT_GE(next_verifier_token, 0)
                << "serial verifier-token extension must produce row "
                << (i + 1);
        }
        const int32_t expected_ready_token = next_verifier_token;

        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base))
            << "decode-equivalent row proof must restore the verifier base "
               "before running the shared production catch-up helper";
        if (device.is_gpu())
        {
            ASSERT_TRUE(runner
                            ->deviceStochasticTargetSampleProducerSlot(
                                kCatchupTargetSampleSlot)
                            .valid())
                << "prefix restore must preserve the explicitly verifier-owned "
                   "condition-token event edge";
        }

        const int vocab = runner->vocab_size();
        const int base_sidecar_position = runner->get_position();
        std::vector<std::vector<float>> catchup_logits_by_row;
        std::vector<int32_t> catchup_samples_by_row;
        catchup_logits_by_row.reserve(static_cast<size_t>(verifier_row_count));
        catchup_samples_by_row.reserve(static_cast<size_t>(verifier_row_count));

        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = verifier_tokens;
        request.base_sidecar_position = base_sidecar_position;
        request.allow_speculative_discard = true;
        request.verifier_path = "phase97_dense_decode_equivalent_row_proof";
        request.implementation_name = "shared_stepwise";
        request.verifier_base_checkpoint = &verifier_base;
        if (device.is_gpu())
        {
            request.device_target_sample_slot = kCatchupTargetSampleSlot;
        }

        auto sample_after_forward = [&](int32_t) -> int32_t
        {
            const int32_t sampled =
                sample_current_to_slot(
                    "dense decode-equivalent catch-up row",
                    kCatchupTargetSampleSlot);
            const float *logits = runner->logits();
            if (!logits)
            {
                ADD_FAILURE()
                    << "dense decode-equivalent catch-up row did not expose logits";
                return sampled;
            }
            catchup_samples_by_row.push_back(sampled);
            catchup_logits_by_row.emplace_back(
                logits,
                logits + static_cast<size_t>(vocab));
            return sampled;
        };

        MTPDecodeCatchupGreedyResult catchup =
            runSharedStepwiseMTPDecodeCatchupGreedy(
                *runner,
                request,
                sample_after_forward);
        ASSERT_TRUE(catchup.ok) << catchup.error;
        EXPECT_EQ(catchup.accepted_tokens, verifier_tokens)
            << "The proof fixture builds verifier draft tokens from the serial "
               "oracle, so the shared stepwise verifier should accept every "
               "row before producing the ready token.";
        EXPECT_TRUE(catchup.all_speculative_accepted);
        EXPECT_EQ(catchup.ready_token, expected_ready_token);
        ASSERT_EQ(catchup_logits_by_row.size(),
                  static_cast<size_t>(verifier_row_count));
        ASSERT_EQ(catchup_samples_by_row.size(),
                  static_cast<size_t>(verifier_row_count));

        auto generate_continuation =
            [&](const std::string &label,
                int32_t input_token,
                int count,
                std::vector<int32_t> *out) -> bool
        {
            out->clear();
            int32_t next_input = input_token;
            for (int i = 0; i < count; ++i)
            {
                if (!runner->forward(&next_input, 1))
                {
                    ADD_FAILURE() << label
                                  << " continuation forward failed at step "
                                  << i;
                    return false;
                }
                const std::string sample_label =
                    label + " continuation step " + std::to_string(i);
                const int32_t sampled =
                    sample_current(sample_label.c_str());
                out->push_back(sampled);
                next_input = sampled;
            }
            return true;
        };

        std::vector<int32_t> catchup_continuation;
        constexpr int continuation_tokens = 4;
        ASSERT_TRUE(generate_continuation(
            "dense decode-equivalent catch-up state",
            catchup.ready_token,
            continuation_tokens,
            &catchup_continuation));

        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
        for (int32_t token : verifier_tokens)
        {
            ASSERT_TRUE(runner->forward(&token, 1))
                << "dense serial continuation reference failed while replaying "
                   "verifier token "
                << token;
        }
        const int32_t serial_ready_token =
            sample_current("dense serial continuation reference");
        EXPECT_EQ(serial_ready_token, catchup.ready_token);
        std::vector<int32_t> serial_continuation;
        ASSERT_TRUE(generate_continuation(
            "dense serial verifier state",
            serial_ready_token,
            continuation_tokens,
            &serial_continuation));
        EXPECT_EQ(catchup_continuation, serial_continuation)
            << "dense decode-equivalent catch-up state must continue exactly "
               "like serial decode"
            << "\nverifier_tokens=" << denseJoinTokens(verifier_tokens)
            << "\ncatchup_continuation="
            << denseJoinTokens(catchup_continuation)
            << "\nserial_continuation="
            << denseJoinTokens(serial_continuation);

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
            for (size_t token_idx = 0; token_idx <= row; ++token_idx)
            {
                const int32_t token = verifier_tokens[token_idx];
                ASSERT_TRUE(runner->forward(&token, 1))
                    << "dense serial row " << row
                    << " verifier forward failed at token index "
                    << token_idx;
            }
            const std::string sample_label =
                "dense serial verifier row " + std::to_string(row);
            const int32_t serial_sample =
                sample_current(sample_label.c_str());
            const float *serial_logits = runner->logits();
            ASSERT_NE(serial_logits, nullptr)
                << "dense serial verifier row " << row
                << " must expose logits for numeric equivalence metrics";

            EXPECT_TRUE(denseVerifierLogitsNumericallyEquivalent(
                catchup_logits_by_row[row].data(),
                serial_logits,
                vocab,
                "dense decode-equivalent catch-up row " +
                    std::to_string(row) +
                    " vs serial prefix " + std::to_string(row + 1)))
                << "\ncondition_prefix_tokens="
                << denseJoinTokens({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << denseJoinTokens(verifier_tokens)
                << "\nrow catch-up top5=["
                << denseTopKSummary(catchup_logits_by_row[row].data(), vocab, 5)
                << "]\nrow serial top5=["
                << denseTopKSummary(serial_logits, vocab, 5)
                << "]";
            EXPECT_EQ(catchup_samples_by_row[row], serial_sample)
                << "dense decode-equivalent row " << row
                << " must sample the same token as serial replay"
                << "\nverifier_tokens="
                << denseJoinTokens(verifier_tokens);
        }

        runner->disableSnapshotCapture();
    }

    /**
     * @brief Prove dense grouped verifier rows match serial decode rows.
     *
     * Phase 9.7 proved the shared stepwise verifier replay path.  Phase 9.8
     * needs the stronger vLLM-style grouped verifier to be numerically safe
     * before it can be promoted for performance.  This helper runs verifier
     * tokens as one all-position graph forward, asks the runner for compact
     * row-indexed logits, and compares every row against the serial oracle.
     *
     * This helper intentionally proves row logits and sampled rows only.  Dense
     * live-state publication is a separate Phase 9.8 gate: verifier graphs may
     * write recurrent state into speculative capture slots, and production must
     * explicitly publish the accepted row before ordinary decode can continue.
     */
    inline void runDenseMainVerifierGroupedRowsMatchSerialDecode(
        const DensePrefixRestoreParityCase &test_case,
        int verifier_row_count)
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        ASSERT_GE(verifier_row_count, 1)
            << "grouped verifier-row proof must exercise at least one row";
        ASSERT_LE(verifier_row_count, 4)
            << "Phase 9.8 production proof currently targets M=1..4";

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GE(expected_tokens.size(), 2u)
            << "grouped verifier proof needs two setup tokens";

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();

        DeviceManager::instance().initialize(-1);
        auto model_ctx = createQwen36ParityModelContext(model_path, device);
        ASSERT_NE(model_ctx, nullptr);

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.mtp.enabled = true;
        config.mtp.draft_tokens = verifier_row_count;

        auto runner = createInferenceRunner(model_ctx, nullptr, device, config);
        ASSERT_NE(runner, nullptr);
        ASSERT_GT(runner->vocab_size(), 0);
        const bool verifier_snapshot_diagnostic =
            std::getenv("LLAMINAR_DENSE_VERIFIER_SNAPSHOT_DIAGNOSTIC") != nullptr;
        runner->setSuppressTimeline(!verifier_snapshot_diagnostic);
        if (verifier_snapshot_diagnostic)
        {
            runner->enableSnapshotCapture();
        }

        constexpr int kSetupTargetSampleSlot = 0;
        auto sample_current = [&](const char *label) -> int32_t
        {
            int32_t sampled = -1;
            const bool sampled_ok =
                device.is_gpu()
                    ? runner->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                          kSetupTargetSampleSlot,
                          &sampled)
                    : ((sampled = runner->sampleGreedyOnDevice()) >= 0);
            if (!sampled_ok || sampled < 0)
            {
                ADD_FAILURE()
                    << "device-owned greedy sampling failed after " << label;
                return -1;
            }
            return sampled;
        };

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())))
            << "prefill forward failed";
        EXPECT_EQ(sample_current("prefill"), expected_tokens[0]);

        int32_t token_after_setup = -1;
        for (int i = 0; i < 2; ++i)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(i)];
            ASSERT_TRUE(runner->forward(&token, 1))
                << "serial setup forward failed at token index " << i;
            token_after_setup = sample_current("serial setup");
        }
        ASSERT_GE(token_after_setup, 0)
            << "serial setup must produce the first grouped verifier input token";

        const PrefixStateSnapshot verifier_base = runner->captureLivePrefixState();
        ASSERT_TRUE(verifier_base.valid);

        std::vector<int32_t> verifier_tokens;
        verifier_tokens.reserve(static_cast<size_t>(verifier_row_count));
        int32_t next_verifier_token = token_after_setup;
        for (int i = 0; i < verifier_row_count; ++i)
        {
            verifier_tokens.push_back(next_verifier_token);
            ASSERT_TRUE(runner->forward(&next_verifier_token, 1))
                << "serial verifier-token extension failed at row " << i;
            next_verifier_token =
                sample_current("serial verifier-token extension");
            ASSERT_GE(next_verifier_token, 0)
                << "serial verifier-token extension must produce row "
                << (i + 1);
        }
        const int32_t expected_ready_token = next_verifier_token;

        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base))
            << "grouped verifier proof must restore the verifier base before "
               "running the all-position candidate";

        MTPSpecDecodeMetadataShape shape;
        shape.max_requests = 1;
        shape.max_draft_tokens = static_cast<int>(verifier_tokens.size());
        MTPSpecDecodeVerifierDraftRequest verifier_request;
        verifier_request.request_id = 0;
        verifier_request.draft_tokens.assign(
            verifier_tokens.begin(),
            verifier_tokens.end());
        const MTPSpecDecodeVerifierInputPlan row_plan =
            buildMTPSpecDecodeVerifierInputPlan(shape, {verifier_request});
        ASSERT_TRUE(row_plan.ok) << row_plan.error;

        ASSERT_TRUE(runner->setMTPSpecVerifierInputPlan(row_plan));
        ASSERT_TRUE(runner->setComputeRowIndexedAllPositionLogits(
            true,
            row_plan.compact_logit_row_count));
        ASSERT_TRUE(runner->setComputeAllPositionLogits(true));
        bool grouped_forward_ok = false;
        if (device.is_gpu())
        {
            const void *verifier_tokens_device =
                runner->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                    verifier_tokens.data(),
                    static_cast<int>(verifier_tokens.size()),
                    static_cast<int>(verifier_tokens.size() - 1));
            ASSERT_NE(verifier_tokens_device, nullptr)
                << "dense GPU grouped verifier must bind its arena-owned token row";
            grouped_forward_ok =
                runner->forwardGroupedMTPVerifierWithDeviceTokenIds(
                verifier_tokens.data(),
                verifier_tokens_device,
                static_cast<int>(verifier_tokens.size()));
        }
        else
        {
            grouped_forward_ok = runner->forward(
                verifier_tokens.data(),
                static_cast<int>(verifier_tokens.size()));
        }
        ASSERT_TRUE(grouped_forward_ok)
            << "dense grouped all-position verifier forward failed";

        std::map<std::string, DenseStageSnapshot> grouped_snapshots;
        if (verifier_snapshot_diagnostic)
        {
            grouped_snapshots = captureDenseStageSnapshots(*runner);
        }

        std::vector<int32_t> grouped_rows(verifier_tokens.size(), -1);
        ASSERT_TRUE(runner->sampleGreedyFromAllPositionLogitsOnDeviceRows(
            0,
            static_cast<int>(grouped_rows.size()),
            grouped_rows.data()));

        const int vocab = runner->vocab_size();
        const float *grouped_logits = runner->getAllPositionLogits();
        ASSERT_NE(grouped_logits, nullptr);
        std::vector<float> grouped_logits_copy(
            grouped_logits,
            grouped_logits + static_cast<size_t>(grouped_rows.size()) *
                                 static_cast<size_t>(vocab));
        EXPECT_EQ(grouped_rows.back(), expected_ready_token)
            << "final grouped verifier row must expose the same ready token as "
               "serial verifier replay";

        /*
         * The verifier flags only describe how the just-completed grouped
         * forward materializes row logits.  Normal continuation must run with
         * those flags disarmed; otherwise the diagnostic would keep asking the
         * graph to behave like a verifier instead of ordinary decode.
         */
        ASSERT_TRUE(runner->setComputeAllPositionLogits(false));
        ASSERT_TRUE(runner->setComputeRowIndexedAllPositionLogits(false, 0));
        runner->clearMTPSpecVerifierInputPlan();

        std::vector<int32_t> serial_rows(verifier_tokens.size(), -1);
        std::vector<std::vector<float>> serial_logits_by_row;
        serial_logits_by_row.reserve(verifier_tokens.size());
        std::vector<std::string> serial_top5_by_row;
        serial_top5_by_row.reserve(verifier_tokens.size());
        std::vector<std::map<std::string, DenseStageSnapshot>> serial_snapshots_by_row;
        if (verifier_snapshot_diagnostic)
        {
            serial_snapshots_by_row.reserve(verifier_tokens.size());
        }

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
            for (size_t token_idx = 0; token_idx <= row; ++token_idx)
            {
                const int32_t token = verifier_tokens[token_idx];
                ASSERT_TRUE(runner->forward(&token, 1))
                    << "dense serial row " << row
                    << " verifier forward failed at token index "
                    << token_idx;
            }
            const std::string sample_label =
                "dense serial grouped verifier row " + std::to_string(row);
            serial_rows[row] = sample_current(sample_label.c_str());
            const float *serial_logits = runner->logits();
            ASSERT_NE(serial_logits, nullptr)
                << "dense serial verifier row " << row
                << " must expose logits for grouped numeric equivalence metrics";
            serial_logits_by_row.emplace_back(
                serial_logits,
                serial_logits + static_cast<size_t>(vocab));
            serial_top5_by_row.push_back(
                denseTopKSummary(serial_logits, vocab, 5));
            if (verifier_snapshot_diagnostic)
            {
                serial_snapshots_by_row.push_back(captureDenseStageSnapshots(*runner));
            }
        }

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            const float *grouped_row_logits =
                grouped_logits_copy.data() + row * static_cast<size_t>(vocab);
            ::testing::AssertionResult snapshot_result =
                ::testing::AssertionSuccess();
            if (verifier_snapshot_diagnostic)
            {
                snapshot_result = denseVerifierRowSnapshotsByteIdentical(
                    grouped_snapshots,
                    serial_snapshots_by_row[row],
                    "dense grouped verifier diagnostic row " +
                        std::to_string(row),
                    static_cast<int>(verifier_tokens.size()),
                    static_cast<int>(row));
            }
            EXPECT_TRUE(denseVerifierLogitsByteIdentical(
                grouped_row_logits,
                serial_logits_by_row[row].data(),
                vocab,
                "dense grouped all-position row " + std::to_string(row) +
                    " vs serial prefix " + std::to_string(row + 1)))
                << "\ncondition_prefix_tokens="
                << denseJoinTokens({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << denseJoinTokens(verifier_tokens)
                << "\nrow grouped top5=["
                << denseTopKSummary(grouped_row_logits, vocab, 5)
                << "]\nrow serial top5=["
                << serial_top5_by_row[row] << "]"
                << (snapshot_result
                        ? std::string()
                        : (std::string("\n") + snapshot_result.message()));
            EXPECT_EQ(grouped_rows[row], serial_rows[row])
                << "dense grouped row " << row
                << " must sample the same token as serial replay"
                << "\nverifier_tokens="
                << denseJoinTokens(verifier_tokens)
                << "\ngrouped_rows="
                << denseJoinTokens(grouped_rows)
                << "\nserial_rows="
                << denseJoinTokens(serial_rows);
        }
    }

    /**
     * @brief Construct the dynamic-depth policy used by stochastic MTP parity.
     *
     * The stochastic parity matrix intentionally uses an eager one-sample
     * controller window so the test proves the dynamic controller is active in
     * short integration runs instead of silently behaving like fixed-depth MTP.
     */
    inline MTPDepthPolicyConfig qwen36DenseStochasticDynamicDepthPolicy(
        int max_depth = 3)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = std::max(1, max_depth);
        depth_policy.initial_depth = depth_policy.max_depth;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;
        return depth_policy;
    }

    inline void runDenseStochasticMTPVerifierParity(
        const DensePrefixRestoreParityCase &test_case,
        int draft_depth = 1,
        MTPDepthPolicyConfig depth_policy = {})
    {
        ScopedDenseParityProductionMode production_mode(
            shouldForceDenseParityProductionMode(test_case));
        ASSERT_TRUE(test_case.topology == DensePrefixParityTopology::SingleDevice ||
                    test_case.topology == DensePrefixParityTopology::LocalPP ||
                    test_case.topology == DensePrefixParityTopology::LocalTP)
            << "Stochastic MTP verifier parity requires either a full-logit "
               "owner or a rank-owned LocalTP compact reducer";

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        const int requested_draft_depth = std::max(1, draft_depth);
        const bool dynamic_depth =
            depth_policy.mode == MTPDepthPolicyMode::Dynamic;
        constexpr int block_size = 2;
        /*
         * One decode-step output slot belongs to the already-sampled first
         * token.  The remaining budget must still fit every requested draft
         * token, otherwise a nominal depth-3 cell is silently clamped to depth
         * 2 and dynamic policy observations are intentionally discarded as
         * budget-limited.  Keep three outputs for the shallow cells and add the
         * fourth output required to exercise a complete depth-3 verifier.
         */
        const int stochastic_decode_steps =
            std::max(3, requested_draft_depth + 1);
        auto factory = createOrchestrationRunnerFactory();

        SamplingParams stochastic;
        stochastic.temperature = 0.6f;
        stochastic.top_k = 20;
        stochastic.top_p = 0.95f;
        stochastic.presence_penalty = 0.25f;
        stochastic.seed = 123;

        /*
         * Same-seed graph-lifecycle parity is necessary but not sufficient:
         * two MTP executions can agree on the same stale recurrent/KV state.
         * Establish the production serial stochastic result first so every
         * cold, warmup, capture, and replay result has an unambiguous oracle.
         * This mirrors the MoE stochastic suite and keeps the dense GDN path
         * honest when graph reuse changes execution ordering.
         */
        auto baseline_config =
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/false,
                block_size,
                /*enable_mtp=*/false);
        auto baseline =
            factory->createFromOrchestrationConfig(baseline_config);
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result =
            baseline->generate(
                prompt_tokens,
                stochastic_decode_steps,
                stochastic);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty())
            << baseline_result.error;
        ASSERT_EQ(
            baseline_result.tokens.size(),
            static_cast<size_t>(stochastic_decode_steps));
        EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);
        EXPECT_EQ(baseline_snapshot.mtp_stochastic_accept_tests, 0u);

        auto mtp_config =
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                false,
                block_size,
                true,
                requested_draft_depth,
                depth_policy);
        mtp_config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

        auto mtp = factory->createFromOrchestrationConfig(mtp_config);
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();
        PerfStatsCollector::reset();
        ASSERT_TRUE(PerfStatsCollector::isEnabled())
            << "Phase 13.8 stochastic candidate parity requires perf stats";
        auto mtp_result = mtp->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
        ASSERT_EQ(mtp_result.tokens.size(), static_cast<size_t>(stochastic_decode_steps));
        const PrefixRuntimeStateSnapshot initial_mtp_snapshot =
            mtp->prefixStateProbe();
        EXPECT_EQ(mtp_result.tokens, baseline_result.tokens)
            << "Dense stochastic MTP must match serial stochastic decode for "
               "the same seed on the first request";
        MTPRuntimeSnapshotComparisonOptions state_comparison;
        state_comparison.compare_shifted_mtp_kv = false;
        const MTPStateValidationResult initial_state_match =
            compareMTPRuntimeStateSnapshots(
                baseline_snapshot,
                initial_mtp_snapshot,
                state_comparison);
        EXPECT_TRUE(initial_state_match)
            << "Dense stochastic MTP must leave main KV payload, GDN "
               "recurrence, short-conv, terminal state, and logical metadata "
               "serial-row-equivalent after its first graph lifecycle: "
            << initial_state_match.reason;
        const auto initial_phase138_records =
            PerfStatsCollector::snapshot(
                {"mtp", "tp_collective_runtime", "forward_graph"});
        expectDenseHomogeneousGPUFullGraphReplay(
            test_case,
            initial_phase138_records,
            test_case.name + " stochastic MTP during first capture");

        mtp->clearCache();
        PerfStatsCollector::reset();
        auto reused_mtp_result = mtp->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        const auto after_reused_mtp = mtp->prefixStateProbe();
        const auto phase138_records =
            PerfStatsCollector::snapshot(
                {"mtp", "tp_collective_runtime", "forward_graph"});
        mtp->shutdown();

        ASSERT_TRUE(reused_mtp_result.error.empty()) << reused_mtp_result.error;
        ASSERT_EQ(reused_mtp_result.tokens.size(), mtp_result.tokens.size());
        EXPECT_EQ(reused_mtp_result.tokens, baseline_result.tokens)
            << "Dense stochastic MTP after clearCache() must match serial "
               "stochastic decode for the same seed";
        const MTPStateValidationResult reused_state_match =
            compareMTPRuntimeStateSnapshots(
                baseline_snapshot,
                after_reused_mtp,
                state_comparison);
        EXPECT_TRUE(reused_state_match)
            << "Dense stochastic MTP graph replay must preserve serial-row "
               "equivalence for main KV payload, GDN recurrence, short-conv, "
               "terminal state, and logical metadata: "
            << reused_state_match.reason;
        EXPECT_EQ(reused_mtp_result.tokens, mtp_result.tokens)
            << "Stochastic MTP with the same seed must be reproducible after "
               "clearCache().\nfirst-capture acceptance trace:\n"
            << densePerfCounterTagSummary(
                   initial_phase138_records,
                   "mtp",
                   "acceptance_trace")
            << "\nreused-graph acceptance trace:\n"
            << densePerfCounterTagSummary(
                   phase138_records,
                   "mtp",
                   "acceptance_trace")
            << "\nfirst-request graph phases:\n"
            << densePerfCounterTagSummary(
                   initial_phase138_records,
                   "forward_graph",
                   "decode_graph_phase")
            << "\nreused-request graph phases:\n"
            << densePerfCounterTagSummary(
                   phase138_records,
                   "forward_graph",
                   "decode_graph_phase");
        EXPECT_FALSE(after_reused_mtp.mtp_bypassed) << after_reused_mtp.mtp_bypass_reason;
        EXPECT_EQ(after_reused_mtp.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(after_reused_mtp.mtp_request.stochastic_verify);
        if (dynamic_depth)
        {
            EXPECT_TRUE(after_reused_mtp.mtp_request.adaptive_depth_enabled);
            EXPECT_EQ(after_reused_mtp.mtp_request.depth_policy_mode, "dynamic");
            EXPECT_GE(after_reused_mtp.mtp_depth_policy_windows, 1u);
            EXPECT_GE(after_reused_mtp.mtp_min_depth, depth_policy.min_depth);
            EXPECT_EQ(after_reused_mtp.mtp_max_depth, depth_policy.max_depth);
            EXPECT_GE(after_reused_mtp.mtp_current_depth, depth_policy.min_depth);
            EXPECT_LE(after_reused_mtp.mtp_current_depth, depth_policy.max_depth);
        }
        else
        {
            EXPECT_FALSE(after_reused_mtp.mtp_request.adaptive_depth_enabled);
            EXPECT_EQ(after_reused_mtp.mtp_max_depth, requested_draft_depth);
        }
        expectPhase138TransactionUsed(
            test_case,
            after_reused_mtp,
            test_case.name + " stochastic MTP",
            /*allow_transaction_rollbacks=*/true);
        EXPECT_GE(after_reused_mtp.mtp_draft_steps, 1u);
        EXPECT_GE(after_reused_mtp.mtp_verifier_runs, 1u);
        EXPECT_GE(
            after_reused_mtp.mtp_verifier_token_count,
            static_cast<uint64_t>(requested_draft_depth + 1))
            << "The stochastic depth cell must execute one complete verifier "
               "at its requested depth instead of passing through a "
               "budget-clamped shallower draft.";
        EXPECT_GE(after_reused_mtp.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(after_reused_mtp.mtp_stochastic_accept_tests,
                  after_reused_mtp.mtp_stochastic_accepts +
                      after_reused_mtp.mtp_stochastic_residual_samples);
        EXPECT_GE(after_reused_mtp.mtp_stochastic_residual_samples +
                      after_reused_mtp.mtp_stochastic_terminal_samples,
                  1u);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_accept_tests,
                  after_reused_mtp.mtp_stochastic_accept_tests);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_accepts,
                  after_reused_mtp.mtp_stochastic_accepts);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_residual_samples,
                  after_reused_mtp.mtp_stochastic_residual_samples);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_terminal_samples,
                  after_reused_mtp.mtp_stochastic_terminal_samples);
        EXPECT_GE(after_reused_mtp.mtp_request.stochastic_acceptance_rate, 0.0);
        EXPECT_LE(after_reused_mtp.mtp_request.stochastic_acceptance_rate, 1.0);
        if (after_reused_mtp.mtp_stochastic_accept_tests > 0)
        {
            const double expected_rate =
                static_cast<double>(after_reused_mtp.mtp_stochastic_accepts) /
                static_cast<double>(after_reused_mtp.mtp_stochastic_accept_tests);
            EXPECT_NEAR(after_reused_mtp.mtp_request.stochastic_acceptance_rate,
                        expected_rate,
                        1e-12);
        }

        const bool used_grouped_stochastic_verifier =
            denseHasMTPPerfCounter(
                phase138_records,
                "grouped_decode_equivalent_stochastic_verifier_runs");
        const bool used_all_position_publication =
            denseHasMTPPerfCounter(
                phase138_records,
                "all_position_state_publication_verifier_runs") &&
            denseHasMTPPerfCounter(phase138_records, "spec_state_publications");
        const bool used_grouped_device_publication =
            denseHasMTPPerfCounter(
                phase138_records,
                "grouped_decode_equivalent_stochastic_verifier_runs") &&
            denseHasMTPPerfCounter(
                phase138_records,
                "grouped_outcome_device_resident_publication_uses") &&
            denseHasMTPPerfCounter(
                phase138_records,
                "grouped_outcome_device_resident_state_publications");
        if (denseCaseExpectsAllPositionSpecPublication(test_case))
        {
            EXPECT_TRUE(used_all_position_publication)
                << "GPU Qwen3.6 stochastic MTP must exercise vLLM-style "
                   "all-position state publication\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_grouped_stochastic_verifier)
                << "GPU Qwen3.6 stochastic MTP must not also run grouped "
                   "decode-equivalent stochastic verification once direct "
                   "all-position publication is available\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }
        else if (denseCaseExpectsGroupedDevicePublication(test_case))
        {
            EXPECT_TRUE(used_grouped_device_publication)
                << "GPU LocalTP Qwen3.6 stochastic MTP must exercise the "
                   "rank-owned grouped verifier outcome and device-resident "
                   "publication path\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_all_position_publication)
                << "GPU LocalTP Qwen3.6 stochastic MTP must not promote to a "
                   "single-owner all-position publication path\n"
                << PerfStatsCollector::summaryString({"mtp"});
            const bool used_rank_grouped_multi_stream_publication =
                denseHasPerfCounter(
                    phase138_records,
                    "tp_collective_runtime",
                    "sideband_only_multi_stream_groups");
            if (test_case.topology == DensePrefixParityTopology::LocalTP)
            {
                EXPECT_TRUE(used_rank_grouped_multi_stream_publication)
                    << "GPU LocalTP stochastic MTP must publish each compact "
                       "mirrored outcome through the rank-level NCCL/RCCL "
                       "multi-stream primitive\n"
                    << PerfStatsCollector::summaryString(
                           {"mtp", "tp_collective_runtime"});
                EXPECT_TRUE(denseHasPerfRecordTag(
                    phase138_records,
                    "tp_collective_runtime",
                    "sideband_only_multi_stream_groups",
                    "host_rendezvous",
                    "false"));
                EXPECT_TRUE(denseHasPerfRecordTag(
                    phase138_records,
                    "tp_collective_runtime",
                    "sideband_only_multi_stream_groups",
                    "device_completion_wait",
                    "false"));
            }
            else
            {
                EXPECT_FALSE(used_rank_grouped_multi_stream_publication)
                    << "SingleDevice stochastic MTP must publish locally and "
                       "must not enter a rank-level collective.";
            }
        }
        else
        {
            EXPECT_TRUE(used_grouped_stochastic_verifier)
                << "CPU Qwen3.6 stochastic MTP must use grouped "
                   "decode-equivalent verification while direct all-position "
                   "publication is not advertised\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_all_position_publication)
                << "CPU Qwen3.6 stochastic MTP must not publish from an "
                   "unproven multi-row all-position verifier\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }

        const bool used_retired_phase138_stochastic_candidate =
            denseHasMTPPerfCounter(
                phase138_records,
                "phase138_stochastic_spec_decode_runs");
        EXPECT_FALSE(used_retired_phase138_stochastic_candidate)
            << "Stateful Qwen3.6 stochastic MTP must not use the retired "
               "accepted-count publication candidate\n"
            << PerfStatsCollector::summaryString({"mtp"});
        expectDenseHomogeneousGPUFullGraphReplay(
            test_case,
            phase138_records,
            test_case.name + " stochastic MTP after clearCache");
        PerfStatsCollector::reset();
    }

} // namespace llaminar2::test::parity::qwen36
