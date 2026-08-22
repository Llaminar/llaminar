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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
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
                 * This fixture completes production economy certification and
                 * a committed migration before collecting parity checkpoints.
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
     * A newly started Dynamic authority profiles a finite set of reversible
     * production transfer waves in the background. Ordinary prefill/decode
     * traffic independently supplies prepared-expert service telemetry. The
     * fixture knows neither controller state nor requested workloads: it uses
     * the same public request surface as the server and observes only outcome
     * evidence. Once production certifies economy, a stationary request stream
     * lets the histogram policy commit a profitable same-priority move.
     */
    bool driveDynamicEconomyAndMovement()
    {
        if (cfg().moe_rebalance.mode != MoERebalanceRuntimeMode::Dynamic)
            return true;

        constexpr int kDecodeStepsPerRequest = 4;
        constexpr int kMaximumServiceProfileRequests = 24;
        constexpr int kMaximumHistogramRequests = 24;

        /*
         * Pre-certification service traffic is ordinary production inference,
         * but it is not part of the authenticated Hugging Face comparison. A
         * deliberately unmatched semantic key avoids diagnostic checkpoint
         * copies while that traffic fills the service telemetry ledger.
         * runPrefillParity() restores the complete evidence filter before the
         * post-migration comparison, preserving every canonical CSV artifact.
         */
        activeSetSnapshotCaptureFilter(
            {"__EXPERT_OVERLAY_SERVICE_PROFILE_NO_SNAPSHOT__"});

        const std::vector<int32_t> reference_prompt(
            config_.token_ids.begin(), config_.token_ids.end());
        if (reference_prompt.empty())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Economy certification requires the authenticated parity prompt");
            return false;
        }

        const int vocabulary_size = orch_runner_->vocabSize();
        if (vocabulary_size <= 4'096)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Invalid vocabulary for service coverage: "
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

        const auto wake_maintenance = [&](const char *phase,
                                          uint64_t committed_tokens)
        {
            if (orch_runner_->maybeApplyMoERebalance(committed_tokens))
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
                    "[Qwen3.5 MoE CPU ExpertOverlay] Service-profile prefill failed: "
                    << orch_runner_->lastError());
                return false;
            }
            return true;
        };
        const auto run_decode = [&]() -> std::optional<bool>
        {
            const GenerationResult generated = orch_runner_->decodeStep();
            if (!generated.success() || generated.tokens.empty())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE CPU ExpertOverlay] Service-profile decode failed: "
                    << generated.error);
                return std::nullopt;
            }
            if (!wake_maintenance("decode", generated.tokens.size()))
                return std::nullopt;
            return generated.is_complete;
        };

        std::uint64_t service_profile_forwards = 0;
        for (int request = 0;
             request < kMaximumServiceProfileRequests &&
             localResidencyCounter("economy_certification_complete") == 0.0;
             ++request)
        {
            /*
             * This is the same public request surface used by the server. The
             * finite transport profiler advances independently; these broad,
             * valid-token requests only populate prepared-expert service
             * telemetry. No controller arm, timing, or private state is read.
             */
            if (!run_prefill(routing_corpus_prompt(request)))
                return false;
            ++service_profile_forwards;

            for (int step = 0;
                 step < kDecodeStepsPerRequest;
                 ++step)
            {
                const auto complete = run_decode();
                if (!complete)
                    return false;
                ++service_profile_forwards;
                if (*complete)
                    break;
            }
        }

        if (localResidencyCounter("economy_certification_complete") == 0.0)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE CPU ExpertOverlay] Economy profile did not certify after "
                << service_profile_forwards << " ordinary production forwards\n"
                << PerfStatsCollector::summaryString(
                       {"moe_overlay_residency"}));
            return false;
        }
        LOG_INFO(
            "[Qwen3.5 MoE CPU ExpertOverlay] Measured economy certified after "
            << service_profile_forwards << " ordinary production forwards");

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
                 step < kDecodeStepsPerRequest;
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
                    << service_profile_forwards << " service-profile forwards and "
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
        const double transport_profile_complete = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_transport_profile_complete");
        const double non_synthetic_transport_profiles =
            globalPerfCounterTotal(
                records,
                "moe_overlay_residency",
                "economy_transport_profile_complete",
                std::pair<std::string, std::string>{
                    "synthetic_inference", "false"});
        const double non_publishing_transport_profiles =
            globalPerfCounterTotal(
                records,
                "moe_overlay_residency",
                "economy_transport_profile_complete",
                std::pair<std::string, std::string>{
                    "publish_residency", "false"});
        const double transport_profile_waves = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_transport_profile_waves_completed");
        const double certification_complete = globalPerfCounterTotal(
            records,
            "moe_overlay_residency",
            "economy_certification_complete");
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
                EXPECT_GT(transport_profile_complete, 0.0)
                    << "Dynamic ExpertOverlay did not complete its finite real-transfer profile";
                EXPECT_EQ(
                    non_synthetic_transport_profiles,
                    transport_profile_complete)
                    << "Dynamic transport profiling must not request synthetic inference";
                EXPECT_EQ(
                    non_publishing_transport_profiles,
                    transport_profile_complete)
                    << "Dynamic transport profiling must not publish residency";
                EXPECT_GT(transport_profile_waves, 0.0)
                    << "Dynamic ExpertOverlay retained no completed transfer-profile waves";
                EXPECT_GT(certification_complete, 0.0)
                    << "Dynamic ExpertOverlay did not install measured economy profiles";
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
        EXPECT_TRUE(plan.hasResolvedPrimaryDeviceNuma());
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
