/**
 * @file Test__Qwen35MoE_NodeTP_Parity.cpp
 * @brief Qwen3.5 MoE NodeTP parity tests across MPI ranks on one node.
 *
 * These tests validate NodeTP infrastructure for Qwen3.5 MoE, where tensor
 * parallelism spans multiple MPI ranks on the same physical node.
 *
 * ExpertOverlay strategy for MoE:
 *   - One homogeneous CPU tier spans both NUMA participants and remains the
 *     sole authority for Static, Dynamic, and current-batch LLEP execution.
 *   - Sparse routed rows return to one logical continuation root, which
 *     broadcasts the complete active output before dense execution resumes.
 *   - Attention and shared-expert row-parallel outputs use the ordinary
 *     production NodeTP collectives.
 *   - Parity reads explicit post-collective graph snapshots; it never inserts
 *     evidence-only MPI operations into the coordinated production protocol.
 *
 * NOTE: CPU TP MUST use NodeTP (MPI), not LocalTP, because
 * DeviceId::cpu() is a singleton. Each MPI rank gets its own process
 * with distinct WeightManager.
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - Model at /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35MoEParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "utils/PerfStatsCollector.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

// =============================================================================
// Common Excluded Stages for NodeTP MoE
// =============================================================================

// For NodeTP, intermediate activations are SHARDED across ranks. Only the final
// LM_HEAD output (after allgather) can be compared against PyTorch reference.
// Includes GDN-specific stages (same as dense Qwen3.5) plus MoE-specific partials.
static const std::vector<std::string> kNodeTPMoEExcludedStages = {
    // --- Standard attention projections (FA layers) — column-parallel slices ---
    "Q_PROJECTION",
    "K_PROJECTION",
    "V_PROJECTION",
    "Q_NORM",
    "K_NORM",
    "Q_ROPE",
    "K_ROPE",
    "ATTENTION_CONTEXT",
    "FA_GATE",
    "ATTENTION_CONTEXT_GATED",
    // --- FFN sharded intermediates (shared expert TP) ---
    "FFN_GATE",
    "FFN_UP",
    "FFN_SWIGLU",
    // --- GDN-specific: column-parallel slices ---
    "QKV_PROJECTION",
    "GDN_CONV1D_OUTPUT",
    "GDN_Z_PROJECTION",
    "GDN_DELTA_RULE_OUTPUT",
    "GDN_NORM_GATE_OUTPUT",
};

// Semantic stages completed by an explicit production collective. The fixture
// requires their post-collective snapshots rather than summing rank partials.
static const std::vector<std::string> kNodeTPMoECollectiveStages = {
    "ATTENTION_OUTPUT",
    "MOE_EXPERT_OUTPUT",
    "MOE_SHARED_EXPERT_OUTPUT",
};

// =============================================================================
// Test Configuration Definitions
// =============================================================================

namespace
{
    enum class CPUExpertPolicyScenario
    {
        StaticOwnership,
        DynamicResidencyMaintenance,
        CurrentBatchLLEP,
    };

    BackendThresholds nodeTPMoEThresholds()
    {
        return {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.03f,
            .excluded_stages = kNodeTPMoEExcludedStages,
            .allreduce_stages = kNodeTPMoECollectiveStages,
            .min_top1_accuracy = 0.80f,
            .min_top5_accuracy = 0.80f,
            .pytorch_top1_in_topk = 4,
        };
    }

    /**
     * @brief Build the one-tier CPU NodeTP ExpertOverlay authority.
     *
     * The typed participants express CPU NUMA intent while production hardware
     * binding chooses their MPI owners. No rank or socket is privileged by the
     * fixture. Static-owner prefill is used by durable Dynamic maintenance;
     * least-loaded-resident prefill selects the graph-owned current-batch LLEP
     * transaction without changing the dense NodeTP continuation graph.
     */
    std::shared_ptr<MoERoutedExpertPlacementPlan> cpuNodeTPExpertPlan(
        RoutedExpertOwnerOrder owner_order,
        RoutedExpertAssignmentPolicy prefill_assignment_policy)
    {
        constexpr const char *kDomain = "cpu_node_tp";

        RoutedExpertDomain routed_domain;
        routed_domain.name = kDomain;
        routed_domain.scope = ExecutionDomainScope::NODE_LOCAL;
        routed_domain.backend = CollectiveBackendType::MPI;
        routed_domain.participants = {
            GlobalDeviceAddress::cpu(0),
            GlobalDeviceAddress::cpu(1),
        };
        routed_domain.routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        routed_domain.routed_phase_policy =
            RoutedExpertPhasePolicy::Uniform;
        routed_domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        routed_domain.routed_prefill_assignment_policy =
            prefill_assignment_policy;

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::SingleDomain;
        plan->continuation_domain = kDomain;
        plan->base_model_domain = kDomain;
        plan->shared_expert_domain = kDomain;
        plan->continuation_domain_spec.domain = kDomain;
        plan->continuation_domain_spec.logical_root_participant = 0;
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::TensorParallel);
        plan->continuation_domain_spec.hidden_layout =
            MoEContinuationActivationLayout::ReplicatedHidden;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = owner_order;
        plan->dense_domains = {
            routed_domain.toExecutionDomainDefinition(),
        };
        plan->domains = {routed_domain};
        plan->routed_tiers = {{
            .name = "all_routed_experts",
            .domain = kDomain,
            .priority = 0,
            .max_experts_per_layer = 0,
            .memory_budget_bytes = 0,
            .fallback = true,
        }};

        const auto validation = validateMoERoutedExpertPlacementPlan(*plan);
        if (!validation.ok())
        {
            std::string error =
                "invalid CPU NodeTP routed-expert parity plan";
            for (const auto &entry : validation.errors)
                error += "\n - " + entry;
            throw std::logic_error(error);
        }
        return plan;
    }

    TestConfig nodeTPMoEConfig(
        CPUExpertPolicyScenario scenario,
        RoutedExpertOwnerOrder owner_order)
    {
        const std::string policy_name =
            scenario == CPUExpertPolicyScenario::StaticOwnership
                ? "Static"
                : scenario ==
                          CPUExpertPolicyScenario::DynamicResidencyMaintenance
                      ? "DynamicMaintenance"
                      : "CurrentBatchLLEP";
        const std::string order_name =
            owner_order == RoutedExpertOwnerOrder::Ordinal
                ? "OrdinalOwners"
                : "RandomOwners";

        TestConfig config;
        config.name = "NodeTP_2xMPI_CPU_35B_MoE_" + policy_name +
                      "_" + order_name + "_FP32_FP16KV";
        config.devices = {
            ParityDeviceType::CPU,
            ParityDeviceType::CPU,
        };
        config.parallelism = Parallelism::NodeTP;
        config.collective = Collective::MPI;
        config.thresholds = nodeTPMoEThresholds();
        config.collective_evidence_source =
            ParityCollectiveEvidenceSource::PostCollectiveSnapshot;
        config.mpi_ranks = 2;
        config.model_path =
            "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";
        config.snapshot_dir = "pytorch_qwen35_moe_snapshots";
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        /*
         * Three decode observations include the terminal-prefill sample and
         * real incremental forwards. Movement policies use a one-row
         * production window below, so five repeated full-model CPU forwards
         * add runtime without adding a distinct correctness boundary.
         */
        config.decode_steps = 3;
        config.routed_expert_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        config.routed_expert_owner_order = owner_order;
        config.moe_rebalance.mode =
            scenario ==
                    CPUExpertPolicyScenario::DynamicResidencyMaintenance
                ? MoERebalanceRuntimeMode::Dynamic
                : MoERebalanceRuntimeMode::Off;
        config.moe_rebalance.release_raw_expert_weights = false;
        config.moe_routed_expert_plan = cpuNodeTPExpertPlan(
            owner_order,
            scenario == CPUExpertPolicyScenario::CurrentBatchLLEP
                ? RoutedExpertAssignmentPolicy::LeastLoadedResident
                : RoutedExpertAssignmentPolicy::StaticOwner);

        if (scenario ==
            CPUExpertPolicyScenario::DynamicResidencyMaintenance)
        {
            config.moe_rebalance.window_size = 1;
            config.moe_rebalance.max_window_size = 1;
            config.moe_rebalance.window_growth_factor = 1.0f;
            /*
             * A max/min load ratio cannot be below 1.0.  Use the most
             * aggressive coherent trigger so this proof cell proposes every
             * strictly imbalanced same-priority layout without bypassing the
             * production policy validator.
             */
            config.moe_rebalance.dynamic_imbalance_threshold_per_mille = 1000;
            config.moe_rebalance.dynamic_min_improvement_per_mille = 0;
            config.moe_rebalance.dynamic_max_swaps_per_layer = 20;
            config.moe_rebalance.dynamic_max_plan_entries_per_wave = 20;
            config.moe_rebalance.dynamic_min_window_activations = 0;
            /*
             * This is a persistent model-server residency proof, not a
             * short-lived batch. A complete cross-socket pair-swap wave costs
             * tens of milliseconds on this topology, so the ordinary
             * 2k-token default correctly rejects the first stationary windows.
             * Declare the same 64k routed-token lifetime used by heterogeneous
             * convergence campaigns: the production economy gate remains
             * authoritative and must still prove positive net benefit from its
             * measured profiles. No fixture constant substitutes for the live
             * transfer and interference certificate.
             */
            config.moe_rebalance.migration_payoff_horizon_tokens = 65'536;
            config.moe_rebalance_exercise = {
                .enabled = true,
                .require_production_overlay_authority = true,
                .request_after_prefill = false,
                .request_every_decode_steps = 1,
                .min_decode_steps = 2,
                /*
                 * This fixture completes production economy calibration and a
                 * committed migration before collecting parity checkpoints.
                 * The generic three-token parity epilogue must not demand a
                 * second, unrelated residency epoch; the policy-specific
                 * PerfStats gate below proves the already-completed movement.
                 */
                .require_movement_epoch_advance = false,
                .min_movement_epoch_delta = 1,
            };
        }

        if (scenario == CPUExpertPolicyScenario::CurrentBatchLLEP)
        {
            config.moe_routed_prefill =
                RoutedExpertPrefillRuntimeConfig{
                    .assignment_window_tokens = 0,
                    .least_loaded_min_routed_rows = 0,
                    .llep_alpha_numerator = 9,
                    .llep_alpha_denominator = 10,
                    .llep_lambda_numerator = 13,
                    .llep_lambda_denominator = 10,
                    .llep_enable_balanced_skip = false,
            };
        }
        return config;
    }

    std::vector<TestConfig> makeNodeTPMoEConfigs()
    {
        std::vector<TestConfig> configs;
        for (const auto owner_order : {
                 RoutedExpertOwnerOrder::Ordinal,
                 RoutedExpertOwnerOrder::Random})
        {
            configs.push_back(nodeTPMoEConfig(
                CPUExpertPolicyScenario::StaticOwnership,
                owner_order));
            configs.push_back(nodeTPMoEConfig(
                CPUExpertPolicyScenario::DynamicResidencyMaintenance,
                owner_order));
            configs.push_back(nodeTPMoEConfig(
                CPUExpertPolicyScenario::CurrentBatchLLEP,
                owner_order));
        }
        return configs;
    }

    const std::vector<TestConfig> kNodeTPMoETestConfigs =
        makeNodeTPMoEConfigs();

    double perfCounterTotal(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name,
        const std::optional<std::pair<std::string, std::string>> &required_tag =
            std::nullopt)
    {
        return std::accumulate(
            records.begin(), records.end(), 0.0,
            [&](double total, const PerfStatRecord &record)
            {
                const bool tag_matches = !required_tag ||
                    (record.tags.contains(required_tag->first) &&
                     record.tags.at(required_tag->first) ==
                         required_tag->second);
                return total +
                       (record.kind == PerfStatRecord::Kind::Counter &&
                                record.domain == domain &&
                                record.name == name && tag_matches
                            ? record.value
                            : 0.0);
            });
    }

    /** @brief Complete identity of one production economy-calibration probe. */
    using CalibrationProbeIdentity = std::array<std::string, 6>;

    /** @brief Ordinary inference phase currently requested by calibration. */
    struct CalibrationProbeDemand
    {
        CalibrationProbeIdentity identity;
        std::uint64_t sequence = 0;
        ExpertHistogramSource source =
            ExpertHistogramSource::SyntheticTest;
        bool concurrent = false;
    };

    /**
     * @brief Reconstruct the correlation key published by a probe event.
     * @param record One production arm or consumed-sample counter.
     * @return Exact immutable identity, or no value for malformed evidence.
     */
    std::optional<CalibrationProbeIdentity> calibrationProbeIdentity(
        const PerfStatRecord &record)
    {
        constexpr std::array<const char *, 6> kIdentityTags{
            "calibration_sequence",
            "mode",
            "source",
            "layer",
            "source_participant",
            "destination_participant",
        };
        CalibrationProbeIdentity identity;
        for (std::size_t index = 0; index < kIdentityTags.size(); ++index)
        {
            const auto tag = record.tags.find(kIdentityTags[index]);
            if (tag == record.tags.end())
                return std::nullopt;
            identity[index] = tag->second;
        }
        return identity;
    }

    /**
     * @brief Find the newest production probe that has not consumed a sample.
     * @param records One process-local residency evidence snapshot.
     * @return Phase-directed inference demand, or no outstanding request.
     *
     * The test uses this only to admit matching ordinary model traffic. The
     * maintenance controller remains the sole owner of calibration timing,
     * transfer overlap, evidence acceptance, and economy certification.
     */
    std::optional<CalibrationProbeDemand> outstandingCalibrationProbe(
        const std::vector<PerfStatRecord> &records)
    {
        std::set<CalibrationProbeIdentity> sampled;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_residency" ||
                record.name != "economy_calibration_probe_samples")
            {
                continue;
            }
            if (const auto identity = calibrationProbeIdentity(record))
                sampled.insert(*identity);
        }

        std::optional<CalibrationProbeDemand> newest;
        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_residency" ||
                record.name != "economy_calibration_probe_arms")
            {
                continue;
            }

            const auto identity = calibrationProbeIdentity(record);
            if (!identity || sampled.contains(*identity))
                continue;

            const auto sequence_tag = record.tags.find("calibration_sequence");
            const auto source_tag = record.tags.find("source");
            const auto mode_tag = record.tags.find("mode");
            if (sequence_tag == record.tags.end() ||
                source_tag == record.tags.end() ||
                mode_tag == record.tags.end())
            {
                continue;
            }

            std::uint64_t sequence = 0;
            const char *const begin = sequence_tag->second.data();
            const char *const end = begin + sequence_tag->second.size();
            const auto parsed = std::from_chars(begin, end, sequence);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
                continue;

            std::optional<ExpertHistogramSource> source;
            if (source_tag->second == "prefill")
                source = ExpertHistogramSource::PrefillChunk;
            else if (source_tag->second == "decode")
                source = ExpertHistogramSource::DecodeToken;
            else if (source_tag->second == "grouped_verifier")
                source = ExpertHistogramSource::GroupedVerifier;
            if (!source)
                continue;

            const bool concurrent =
                mode_tag->second == "concurrent_movement";
            if (!concurrent && mode_tag->second != "baseline")
                continue;
            if (newest &&
                (sequence < newest->sequence ||
                 (sequence == newest->sequence &&
                  !concurrent && newest->concurrent)))
            {
                continue;
            }
            newest = CalibrationProbeDemand{
                .identity = *identity,
                .sequence = sequence,
                .source = *source,
                .concurrent = concurrent,
            };
        }
        return newest;
    }

    /**
     * @brief Sum one counter over every rank after the worker loop is closed.
     *
     * PerfStats is deliberately process-local. Expert movement can originate
     * on either NUMA participant, so a root-only snapshot is not proof of the
     * distributed production path. Every caller must invoke this helper in the
     * same order after SHUTDOWN, when no production command can compete with
     * this evidence-only collective.
     */
    double globalPerfCounterTotal(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name,
        const std::optional<std::pair<std::string, std::string>> &required_tag =
            std::nullopt)
    {
        const double local =
            perfCounterTotal(records, domain, name, required_tag);
        double global = 0.0;
        MPI_Allreduce(
            &local,
            &global,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);
        return global;
    }
} // namespace

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35MoENodeTPParityTest : public Qwen35MoEConfigDrivenParityTest<Qwen35MoENodeTPParityTest>,
                                       public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }

protected:
    void SetUp() override
    {
        /*
         * Path evidence is part of these policy cells, not an optional launch
         * decoration.  Keep collection narrowly scoped to the two MoE domains
         * and restore the caller's environment in ParityTestBase::TearDown().
         */
        setScopedParityEnvOverride(
            "LLAMINAR_PERF_STATS_SUMMARY",
            "1");
        setScopedParityEnvOverride(
            "LLAMINAR_PERF_STATS_FILTER",
            "forward_graph,moe_rebalance,moe_placement,moe_overlay_residency");
        Qwen35MoEConfigDrivenParityTest<
            Qwen35MoENodeTPParityTest>::SetUp();
    }

    /**
     * @brief Return the inventory-bound continuation authority after setup.
     *
     * Reference preparation still defaults to rank zero before a runner
     * exists. Once hardware binding completes, the same rank that owns dense
     * logits also owns comparisons and CSV artifacts.
     */
    int parityArtifactAuthorityRank() const override
    {
        return orch_runner_ ? orch_runner_->coordinatedRootRank() : 0;
    }

    /** @return Whether this process owns coordinated inference and artifacts. */
    bool isRootParityRank() const
    {
        return !mpi_ctx_ ||
               mpi_ctx_->rank() == parityArtifactAuthorityRank();
    }

    /** @brief Fold one setup result across the complete MPI application. */
    bool synchronizeRanksOk(bool local_ok) const
    {
        int ok = local_ok ? 1 : 0;
        MPI_Allreduce(
            MPI_IN_PLACE,
            &ok,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD);
        return ok == 1;
    }

    /**
     * @brief Terminate a world whose production collective order is unknown.
     *
     * An exception during an active root-issued forward can leave a worker in
     * a sparse or dense collective. Broadcasting SHUTDOWN in that state is not
     * safe, so fail the whole MPI application with the original diagnostic.
     */
    [[noreturn]] void abortProductionWorld(
        const char *phase,
        const std::string &detail) const
    {
        LOG_ERROR("[Qwen3.5 MoE CPU ExpertOverlay] " << phase
                  << " failed: " << detail
                  << "; aborting MPI world because collective order is indeterminate");
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }

    /** @brief Snapshot and sum one process-local residency counter. */
    double localResidencyCounter(const std::string &name) const
    {
        return perfCounterTotal(
            PerfStatsCollector::snapshot({"moe_overlay_residency"}),
            "moe_overlay_residency",
            name);
    }

    /**
     * @brief Drive measured economy certification and one durable CPU migration.
     * @return True after production publishes a same-tier residency epoch.
     *
     * A newly started dynamic authority first measures real packed-weight
     * transfers and their overlap with every reachable inference phase. The
     * test therefore follows the outstanding PerfStats probe with an ordinary
     * prefill or decode call. It never supplies a timing, histogram, proposal,
     * placement, or completion value. Once the controller has certified its
     * measured economy, a stationary request stream lets the production
     * histogram policy decide and commit a profitable same-priority move.
     */
    bool driveDynamicEconomyAndMovement()
    {
        if (cfg().moe_rebalance.mode != MoERebalanceRuntimeMode::Dynamic)
            return true;

        constexpr auto kProgressTimeout = std::chrono::seconds(30);
        constexpr auto kPollPeriod = std::chrono::milliseconds(2);
        constexpr int kDecodeStepsPerHistogramRequest = 4;
        constexpr int kMaximumHistogramRequests = 24;

        const double expected_pairs_value =
            localResidencyCounter("economy_calibration_expected_pairs");
        if (expected_pairs_value <= 0.0 ||
            expected_pairs_value > static_cast<double>(
                                       std::numeric_limits<std::uint64_t>::max()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Invalid economy calibration pair count: "
                << expected_pairs_value);
            return false;
        }
        const auto expected_pairs =
            static_cast<std::uint64_t>(expected_pairs_value);
        if (static_cast<double>(expected_pairs) != expected_pairs_value ||
            expected_pairs >
                (std::numeric_limits<std::uint64_t>::max() - 32u) / 8u)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Non-integral or overflowing economy calibration plan: "
                << expected_pairs_value);
            return false;
        }
        const std::uint64_t maximum_calibration_forwards =
            expected_pairs * 8u + 32u;

        /*
         * Calibration must time the serving graph, not thousands of diagnostic
         * checkpoint copies. A deliberately unmatched semantic key keeps
         * snapshot infrastructure enabled but rejects every calibration stage.
         * runPrefillParity() restores the complete evidence filter before the
         * post-migration request, so every authenticated CSV checkpoint remains
         * unchanged while calibration measures the production economy.
         */
        activeSetSnapshotCaptureFilter(
            {"__EXPERT_OVERLAY_ECONOMY_CALIBRATION_NO_SNAPSHOT__"});

        const std::vector<int32_t> reference_prompt(
            config_.token_ids.begin(), config_.token_ids.end());
        if (reference_prompt.empty())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Economy calibration requires the authenticated parity prompt");
            return false;
        }

        const int vocabulary_size = orch_runner_->vocabSize();
        if (vocabulary_size <= 4'096)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Invalid vocabulary for calibration service coverage: "
                << vocabulary_size);
            return false;
        }

        const auto routing_corpus_prompt = [&](int request_index)
        {
            if (request_index == 0)
                return reference_prompt;

            /*
             * These are deterministic valid embedding rows, not injected
             * routing data. Broad ordinary traffic ensures both CPU
             * participants naturally publish service samples for every layer.
             */
            std::uint64_t state =
                0x9e3779b97f4a7c15ULL ^
                (static_cast<std::uint64_t>(request_index) *
                 0xbf58476d1ce4e5b9ULL);
            const auto usable_vocabulary =
                static_cast<std::uint64_t>(vocabulary_size - 2'048);
            std::vector<int32_t> prompt(reference_prompt.size(), 0);
            for (auto &token : prompt)
            {
                state += 0x9e3779b97f4a7c15ULL;
                std::uint64_t mixed = state;
                mixed = (mixed ^ (mixed >> 30u)) *
                        0xbf58476d1ce4e5b9ULL;
                mixed = (mixed ^ (mixed >> 27u)) *
                        0x94d049bb133111ebULL;
                mixed ^= mixed >> 31u;
                token = static_cast<int32_t>(
                    256u + mixed % usable_vocabulary);
            }
            return prompt;
        };

        bool request_active = false;
        const auto wake_maintenance = [&](const char *phase)
        {
            if (orch_runner_->maybeApplyMoERebalance())
                return true;
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Maintenance wake failed after "
                << phase << ": " << orch_runner_->lastError());
            return false;
        };
        const auto run_prefill = [&](const std::vector<int32_t> &prompt)
        {
            activeClearSnapshots();
            activeClearCache();
            if (!orch_runner_->prefill(prompt))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE CPU ExpertOverlay] Calibration prefill failed: "
                    << orch_runner_->lastError());
                return false;
            }
            request_active = true;
            return wake_maintenance("prefill");
        };
        const auto run_decode = [&]() -> std::optional<bool>
        {
            const GenerationResult generated = orch_runner_->decodeStep();
            if (!generated.success() || generated.tokens.empty())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE CPU ExpertOverlay] Calibration decode failed: "
                    << generated.error);
                return std::nullopt;
            }
            if (!wake_maintenance("decode"))
                return std::nullopt;
            request_active = !generated.is_complete;
            return generated.is_complete;
        };

        std::uint64_t calibration_forwards = 0;
        int service_coverage_request = 1;
        int service_coverage_decode_steps =
            kDecodeStepsPerHistogramRequest;
        std::optional<CalibrationProbeIdentity> last_served_probe;
        std::optional<CalibrationProbeIdentity> last_observed_probe;
        double last_accepted_pairs =
            localResidencyCounter("economy_calibration_pairs_accepted");
        double last_rejected_attempts = localResidencyCounter(
            "economy_calibration_attempt_rejections");
        auto last_progress = std::chrono::steady_clock::now();

        /* Admit one real request while the background worker publishes its arm. */
        if (!run_prefill(reference_prompt))
            return false;
        ++calibration_forwards;

        while (calibration_forwards < maximum_calibration_forwards)
        {
            const auto records =
                PerfStatsCollector::snapshot({"moe_overlay_residency"});
            if (perfCounterTotal(
                    records,
                    "moe_overlay_residency",
                    "economy_certification_complete") > 0.0)
            {
                break;
            }

            const auto requested_probe =
                outstandingCalibrationProbe(records);
            const double accepted_pairs = perfCounterTotal(
                records,
                "moe_overlay_residency",
                "economy_calibration_pairs_accepted");
            const double rejected_attempts = perfCounterTotal(
                records,
                "moe_overlay_residency",
                "economy_calibration_attempt_rejections");
            const bool movement_calibration_complete = perfCounterTotal(
                records,
                "moe_overlay_residency",
                "economy_calibration_complete") > 0.0;
            const bool probe_changed =
                requested_probe.has_value() !=
                    last_observed_probe.has_value() ||
                (requested_probe &&
                 requested_probe->identity != *last_observed_probe);
            if (probe_changed || accepted_pairs != last_accepted_pairs)
            {
                if (accepted_pairs != last_accepted_pairs)
                {
                    LOG_INFO(
                        "[Qwen3.5 MoE CPU ExpertOverlay] Economy calibration accepted "
                        << accepted_pairs << "/" << expected_pairs
                        << " robust pairs after " << calibration_forwards
                        << " forwards");
                }
                last_observed_probe =
                    requested_probe
                        ? std::optional<CalibrationProbeIdentity>{
                              requested_probe->identity}
                        : std::nullopt;
                last_accepted_pairs = accepted_pairs;
                last_progress = std::chrono::steady_clock::now();
            }
            if (rejected_attempts != last_rejected_attempts)
            {
                std::string newest_reason = "unknown";
                for (auto record = records.rbegin();
                     record != records.rend(); ++record)
                {
                    if (record->kind != PerfStatRecord::Kind::Counter ||
                        record->domain != "moe_overlay_residency" ||
                        record->name !=
                            "economy_calibration_attempt_rejections")
                    {
                        continue;
                    }
                    if (const auto reason = record->tags.find("reason");
                        reason != record->tags.end())
                    {
                        newest_reason = reason->second;
                    }
                    break;
                }
                LOG_INFO(
                    "[Qwen3.5 MoE CPU ExpertOverlay] Economy calibration rejected attempt "
                    << rejected_attempts << " after "
                    << calibration_forwards << " forwards; reason="
                    << newest_reason);
                last_rejected_attempts = rejected_attempts;
            }

            if (!requested_probe && movement_calibration_complete)
            {
                /*
                 * Migration timing is complete. Continue real, broad traffic
                 * until every participant/layer/phase service coordinate has
                 * contributed to the all-rank certification vote.
                 */
                if (!request_active ||
                    service_coverage_decode_steps >=
                        kDecodeStepsPerHistogramRequest)
                {
                    if (!run_prefill(
                            routing_corpus_prompt(service_coverage_request++)))
                    {
                        return false;
                    }
                    service_coverage_decode_steps = 0;
                }
                else
                {
                    const auto complete = run_decode();
                    if (!complete)
                        return false;
                    ++service_coverage_decode_steps;
                    if (*complete)
                    {
                        service_coverage_decode_steps =
                            kDecodeStepsPerHistogramRequest;
                    }
                }
                ++calibration_forwards;
                if ((calibration_forwards % 8u) == 0u)
                {
                    LOG_INFO(
                        "[Qwen3.5 MoE CPU ExpertOverlay] Movement calibration complete; collecting measured service coverage at forward "
                        << calibration_forwards);
                }
                last_progress = std::chrono::steady_clock::now();
                continue;
            }

            if (!requested_probe ||
                (last_served_probe &&
                 requested_probe->identity == *last_served_probe))
            {
                if (!wake_maintenance("calibration progress"))
                    return false;
                if (std::chrono::steady_clock::now() - last_progress >
                    kProgressTimeout)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE CPU ExpertOverlay] Economy calibration made no progress for 30 seconds after "
                        << calibration_forwards << " production forwards\n"
                        << PerfStatsCollector::summaryString(
                               {"moe_overlay_residency"}));
                    return false;
                }
                std::this_thread::sleep_for(kPollPeriod);
                continue;
            }

            if (requested_probe->source ==
                ExpertHistogramSource::PrefillChunk)
            {
                if (!run_prefill(reference_prompt))
                    return false;
                last_served_probe = requested_probe->identity;
                ++calibration_forwards;
                continue;
            }
            if (requested_probe->source ==
                ExpertHistogramSource::GroupedVerifier)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE CPU ExpertOverlay] Non-MTP cell received a grouped-verifier calibration probe");
                return false;
            }
            if (!request_active)
            {
                if (!run_prefill(reference_prompt))
                    return false;
                ++calibration_forwards;
                continue;
            }

            /*
             * The first decodeStep consumes prefill logits without executing a
             * DecodeToken graph. Leave the arm outstanding until the next call
             * actually traverses the model.
             */
            const bool executes_decode_phase =
                !activePrefixStateProbe().prefill_logits_ready;
            const auto complete = run_decode();
            if (!complete)
                return false;
            if (executes_decode_phase)
                last_served_probe = requested_probe->identity;
            ++calibration_forwards;
        }

        if (localResidencyCounter("economy_certification_complete") == 0.0)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Economy calibration did not certify after "
                << calibration_forwards << " production forwards\n"
                << PerfStatsCollector::summaryString(
                       {"moe_overlay_residency"}));
            return false;
        }
        LOG_INFO(
            "[Qwen3.5 MoE CPU ExpertOverlay] Measured economy certified after "
            << calibration_forwards << " production forwards");

        /*
         * Keep the post-certification workload stationary. A changed corpus
         * would optimize different traffic each epoch and could not prove that
         * histogram-driven placement improves one stable request population.
         */
        for (int request = 0;
             request < kMaximumHistogramRequests;
             ++request)
        {
            if (!run_prefill(reference_prompt))
                return false;
            for (int step = 0;
                 step < kDecodeStepsPerHistogramRequest;
                 ++step)
            {
                const auto complete = run_decode();
                if (!complete)
                    return false;
                if (*complete)
                    break;
            }

            if (localResidencyCounter("committed_expert_migrations") > 0.0 &&
                localResidencyCounter("same_priority_moves") > 0.0)
            {
                LOG_INFO(
                    "[Qwen3.5 MoE CPU ExpertOverlay] Certified economy and committed same-tier movement after "
                    << calibration_forwards << " calibration forwards and "
                    << (request + 1) << " stationary histogram requests");
                return true;
            }

            LOG_INFO(
                "[Qwen3.5 MoE CPU ExpertOverlay] Stationary histogram request "
                << (request + 1)
                << " completed without a committed same-tier move; proposals="
                << localResidencyCounter("economy_proposals"));
        }

        LOG_ERROR(
            "[Qwen3.5 MoE CPU ExpertOverlay] Certified dynamic policy committed no same-priority migration\n"
            << PerfStatsCollector::summaryString(
                   {"moe_overlay_residency"}));
        return false;
    }

    /**
     * @brief Prove policy-specific movement after production workers stop.
     *
     * All ranks execute the identical reduction sequence. Static must publish
     * an explicit no-movement check and no migration edge. Dynamic must commit
     * a same-priority participant migration and transfer real packed bytes.
     * LLEP remains a current-batch transaction and must move a non-owner expert
     * before restoring the durable owner map.
     */
    void assertPolicyMovementEvidenceAfterWorkerShutdown()
    {
        const auto records = PerfStatsCollector::snapshot(
            {"moe_rebalance", "moe_placement", "moe_overlay_residency"});
        const bool current_batch_llep =
            cfg().moe_routed_expert_plan &&
            std::any_of(
                cfg().moe_routed_expert_plan->domains.begin(),
                cfg().moe_routed_expert_plan->domains.end(),
                [](const RoutedExpertDomain &domain)
                {
                    return domain.routed_prefill_assignment_policy ==
                           RoutedExpertAssignmentPolicy::LeastLoadedResident;
                });
        const bool dynamic_maintenance =
            cfg().moe_rebalance.mode ==
            MoERebalanceRuntimeMode::Dynamic;

        const double llep_begin = globalPerfCounterTotal(
            records, "moe_rebalance", "cpu_llep_plan_calls");
        const double llep_restore = globalPerfCounterTotal(
            records, "moe_rebalance", "cpu_llep_restore_calls");
        const double llep_transfers = globalPerfCounterTotal(
            records, "moe_rebalance", "cpu_llep_weight_transfers");
        const double llep_bytes = globalPerfCounterTotal(
                                      records,
                                      "moe_rebalance",
                                      "cpu_llep_weight_transfer_outgoing_bytes") +
                                  globalPerfCounterTotal(
                                      records,
                                      "moe_rebalance",
                                      "cpu_llep_weight_transfer_incoming_bytes");
        const double llep_non_owner_rows = globalPerfCounterTotal(
            records, "moe_rebalance", "cpu_llep_non_owner_rows");

        const double static_checks = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "static_no_movement_checks");
        const double committed_migrations = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "committed_expert_migrations");
        const double migration_edges = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "expert_migration_edges");
        const double same_priority_moves = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "same_priority_moves");
        const double committed_waves = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "maintenance_waves_committed");
        const double calibration_complete = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_calibration_complete");
        const double certification_complete = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_certification_complete");
        const double calibration_wave_ns = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_calibration_wave_wall_ns");
        const double calibration_baseline_ns = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_calibration_inference_baseline_ns");
        const double calibration_concurrent_ns = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_calibration_inference_concurrent_ns");
        const double local_copied_bytes = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "cpu_native_copy_bytes");
        const double remote_placement_bytes = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "remote_projection_payload_bytes_completed",
            std::pair<std::string, std::string>{
                "purpose", "placement_change"});
        const double copied_bytes =
            local_copied_bytes + remote_placement_bytes;
        const double published_epoch = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "published_epoch");

        if (isRootParityRank())
        {
            if (current_batch_llep)
            {
                EXPECT_GT(llep_begin, 0.0)
                    << "CPU LLEP parity did not execute the production begin stage";
                EXPECT_EQ(llep_restore, llep_begin)
                    << "Every CPU LLEP transaction must restore durable residency";
                EXPECT_GT(llep_transfers, 0.0)
                    << "Movement-positive CPU LLEP parity moved no packed expert";
                EXPECT_GT(llep_bytes, 0.0)
                    << "Movement-positive CPU LLEP parity transferred no bytes";
                EXPECT_GT(llep_non_owner_rows, 0.0)
                    << "Movement-positive CPU LLEP parity assigned no non-owner rows";
                EXPECT_EQ(committed_migrations, 0.0)
                    << "Current-batch LLEP must not mutate durable residency";
            }
            else
            {
                EXPECT_EQ(llep_begin, 0.0)
                    << "Static/Dynamic cells must not silently enter LLEP";
            }

            if (dynamic_maintenance)
            {
                EXPECT_GT(calibration_complete, 0.0)
                    << "Dynamic ExpertOverlay did not complete real movement calibration";
                EXPECT_GT(certification_complete, 0.0)
                    << "Dynamic ExpertOverlay did not install measured economy profiles";
                EXPECT_GT(calibration_wave_ns, 0.0)
                    << "Dynamic ExpertOverlay did not retain accepted wave timing evidence";
                EXPECT_GT(calibration_baseline_ns, 0.0)
                    << "Dynamic ExpertOverlay did not retain baseline inference timing";
                EXPECT_GT(calibration_concurrent_ns, 0.0)
                    << "Dynamic ExpertOverlay did not retain concurrent inference timing";
                EXPECT_GT(committed_waves, 0.0)
                    << "Dynamic ExpertOverlay committed no background wave";
                EXPECT_GT(committed_migrations, 0.0)
                    << "Dynamic ExpertOverlay committed no expert migration";
                EXPECT_GT(migration_edges, 0.0)
                    << "Dynamic ExpertOverlay published no migration edge";
                EXPECT_GT(same_priority_moves, 0.0)
                    << "One-tier Dynamic must rebalance between same-priority participants";
                EXPECT_GT(copied_bytes, 0.0)
                    << "Dynamic ExpertOverlay copied no real packed expert bytes";
                EXPECT_GT(published_epoch, 0.0)
                    << "Dynamic ExpertOverlay did not publish a later epoch";
            }
            else if (!current_batch_llep)
            {
                EXPECT_GT(static_checks, 0.0)
                    << "Static ExpertOverlay did not publish its immobility proof";
                EXPECT_EQ(committed_migrations, 0.0)
                    << "Static ExpertOverlay unexpectedly migrated an expert";
                EXPECT_EQ(migration_edges, 0.0)
                    << "Static ExpertOverlay unexpectedly published a movement edge";
                EXPECT_EQ(copied_bytes, 0.0)
                    << "Static ExpertOverlay unexpectedly copied migration bytes";
                EXPECT_EQ(published_epoch, 0.0)
                    << "Static ExpertOverlay unexpectedly advanced residency epoch";
            }
        }

        finishProductionParityEvidence();
    }

    /**
     * @brief Run numerical parity through the real coordinated MPI application.
     *
     * Only the inventory-selected continuation authority issues request
     * commands. The other MPI instance remains inside the production worker
     * loop and executes its real participant graph until SHUTDOWN.
     */
    void runExpertOverlayProductionParityCampaign()
    {
        beginProductionParityEvidence();
        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(synchronizeRanksOk(setup_ok))
            << "ExpertOverlay pipeline setup failed on one or more ranks";
        ASSERT_NE(orch_runner_, nullptr);

        orch_runner_->setMPICoordinatedMode(true);
        MPI_Barrier(MPI_COMM_WORLD);
        if (!isRootParityRank())
        {
            orch_runner_->runMPIWorkerLoop();
            assertPolicyMovementEvidenceAfterWorkerShutdown();
            return;
        }

        struct WorkerShutdownGuard
        {
            IOrchestrationRunner *runner = nullptr;
            ~WorkerShutdownGuard()
            {
                if (runner)
                    runner->shutdownMPIWorkers();
            }
        } worker_shutdown{orch_runner_.get()};

        try
        {
            if (!driveDynamicEconomyAndMovement())
            {
                throw std::runtime_error(
                    "dynamic ExpertOverlay economy/movement proof did not converge");
            }

            /*
             * Calibration traffic is deliberately not reference evidence.
             * Reset request data and diagnostic buffers, retain the migrated
             * residency epoch, then compare the complete production graph at
             * every authenticated checkpoint after movement.
             */
            activeClearSnapshots();
            activeClearCache();

            const auto prefill = runPrefillParity();
            assertParity(prefill);
            assertProductionParitySnapshotInfrastructure();

            // This is a request-data reset. The ExpertOverlay authority,
            // prepared banks, graph topology, streams, and residency epoch are
            // model-lifetime state and deliberately survive it.
            activeClearSnapshots();
            activeClearCache();

            const auto decode = runDecodeParity();
            if (decode.steps_total == 0)
            {
                ADD_FAILURE()
                    << "Production parity requires authenticated incremental-decode references";
            }
            else
            {
                assertDecodeParity(decode);
            }
        }
        catch (const std::exception &error)
        {
            abortProductionWorld("root parity execution", error.what());
        }
        catch (...)
        {
            abortProductionWorld(
                "root parity execution",
                "non-standard exception");
        }

        // Close the production protocol before the evidence-only allreduces.
        orch_runner_->shutdownMPIWorkers();
        worker_shutdown.runner = nullptr;
        assertPolicyMovementEvidenceAfterWorkerShutdown();
    }
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Verify NodeTP infrastructure initialization for MoE
 */
TEST_P(Qwen35MoENodeTPParityTest, NodeTPContextInitialization)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    ASSERT_NE(mpi_ctx_, nullptr) << "MPI context should be initialized";
    EXPECT_GE(mpi_ctx_->world_size(), cfg().mpi_ranks)
        << "World size should be at least " << cfg().mpi_ranks;

    if (hasOrchestrationRunner())
    {
        ASSERT_NE(orch_runner_, nullptr);
        const auto &plan = orch_runner_->executionPlan();
        EXPECT_TRUE(plan.usesGlobalTP());
        EXPECT_EQ(plan.tp_scope, TPScope::NODE_LOCAL);
        EXPECT_EQ(plan.global_tp_domain_size, mpi_ctx_->world_size());
        EXPECT_EQ(plan.global_tp_rank_in_domain, mpi_ctx_->rank());
        EXPECT_TRUE(plan.primary_device.isCPU());
        EXPECT_EQ(plan.primary_device.numa_node, mpi_ctx_->rank());
        EXPECT_TRUE(plan.primary_device_numa_explicit);
        EXPECT_EQ(
            plan.runtime.activation_precision,
            cfg().activation_precision);
        EXPECT_EQ(
            plan.runtime.kv_cache_precision,
            cfg().kv_cache_precision)
            << "The production execution plan must consume the precision named by the parity cell";
    }
    else
    {
        ASSERT_NE(global_tp_ctx_, nullptr)
            << "NodeTP context should be created";
        EXPECT_EQ(global_tp_ctx_->degree(), mpi_ctx_->world_size())
            << "TP degree should match world size";
        EXPECT_EQ(global_tp_ctx_->myIndex(), mpi_ctx_->rank())
            << "TP index should match MPI rank";
    }

    LOG_INFO("[NodeTP Qwen3.5 MoE] Rank " << mpi_ctx_->rank() << " verified TPContext");
}

/**
 * @brief Full real-weight production parity for cross-rank MoE tensor parallelism.
 */
TEST_P(Qwen35MoENodeTPParityTest, ProductionParity)
{
    runExpertOverlayProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoENodeTP,
    Qwen35MoENodeTPParityTest,
    ::testing::ValuesIn(kNodeTPMoETestConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0)
    {
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║     QWEN3.5 MoE NODE TP PARITY TEST SUITE                        ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  MPI world size: " << world_size << " ranks" << std::string(42 - std::to_string(world_size).length(), ' ') << "║\n";
        std::cout << "║  Thread support: " << (provided >= MPI_THREAD_MULTIPLE ? "MPI_THREAD_MULTIPLE" : "limited") << std::string(26, ' ') << "║\n";
        std::cout << "║  TP strategy:   static routed rows + Megatron shared expert" << std::string(2, ' ') << "║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    int global_result;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(global_result);
}
