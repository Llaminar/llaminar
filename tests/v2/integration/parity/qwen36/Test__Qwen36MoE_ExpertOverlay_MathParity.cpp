/**
 * @file Test__Qwen36MoE_ExpertOverlay_MathParity.cpp
 * @brief PyTorch snapshot parity for Qwen3.6 MoE homogeneous multi-GPU expert overlays.
 *
 * These tests are the quality gate for the GPU MoE rebalancing sprint target:
 * one LocalTP routed expert domain with two CUDA/NCCL or ROCm/RCCL
 * participants in the current fixtures, tensor-parallel prefill and replicated
 * decode dense execution, apportioned routed experts, explicit assignment
 * policies, and schema-selected TP transport. Every model-bearing case enters
 * through the production orchestration factory so the ExpertOverlay residency
 * authority, LocalTP graph, and movement controller have their serving
 * lifetimes rather than test-owned substitutes.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35moe/Qwen35MoEParityTestBase.h"
#include "Qwen36MoEParityTestBase.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "execution/config/RuntimeConfig.h"
#include "utils/Logger.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    using PlanFactory = std::shared_ptr<MoERoutedExpertPlacementPlan> (*)();

    const std::vector<std::string> kOverlayExcludedStages = {
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
        "FFN_GATE",
        "FFN_UP",
        "FFN_SWIGLU",
        "QKV_PROJECTION",
        "GDN_CONV1D_OUTPUT",
        "GDN_Z_PROJECTION",
        "GDN_DELTA_RULE_OUTPUT",
        "GDN_NORM_GATE_OUTPUT",
        "MOE_EXPERT_OUTPUT",
        "MOE_SHARED_EXPERT_OUTPUT",
        "MOE_SHARED_GATE_OUTPUT",
    };

    BackendThresholds qwen36MoEOverlayThresholds()
    {
        return {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .excluded_stages = kOverlayExcludedStages,
            /*
             * The graph reduces routed and shared branches independently,
             * then publishes MOE_COMBINED_OUTPUT from their final local add.
             * That boundary is already replicated; asking the harness to find
             * a second combined-output collective would test the retired DAG.
             */
            .allreduce_stages = {},
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 60.0f,
            .pytorch_top1_in_topk = 4,
        };
    }

    enum class ExpertOverlayPolicyScenario
    {
        StaticOwnership,
        DynamicResidencyMaintenance,
        CurrentBatchLLEP,
    };

    struct ExpertOverlayParityConfig : TestConfig
    {
        ExpertOverlayPolicyScenario policy_scenario =
            ExpertOverlayPolicyScenario::StaticOwnership;
        /** Physical deterministic owner order used by loading and graph maps. */
        RoutedExpertOwnerOrder owner_order = RoutedExpertOwnerOrder::Ordinal;
        /** Typed replica policy supplied to the production runner. */
        MoEHotExpertCacheConfig moe_hot_expert_cache;
        RoutedExpertPrefillRuntimeConfig moe_routed_prefill;
        MoERebalanceRuntimeConfig moe_rebalance;
        int max_seq_len = 4096;
    };

    /**
     * @brief Require parity to exercise its named movement-positive policy.
     *
     * Token parity alone cannot distinguish a healthy Dynamic-maintenance or
     * current-batch LLEP lane from a graph that never planned its named work.
     * These assertions consume request-local PerfStats after the generic parity
     * epilogue has published final device status. They deliberately validate
     * the policy-specific planner plus shared transport/apply health so a future
     * failure identifies the missing axis.
     *
     * @param config Concrete backend and MoE runtime policy under test.
     * @param records Request-local overlay records collected during the
     *        authenticated prefill and incremental decode transaction.
     */
    void expectMovementPositiveExpertOverlayPerfPath(
        const ExpertOverlayParityConfig &config,
        const std::vector<PerfStatRecord> &records)
    {
        const std::string context =
            config.name + " movement-positive expert overlay";

        if (config.policy_scenario ==
            ExpertOverlayPolicyScenario::CurrentBatchLLEP)
        {
            expectLLEPPrefillWorkRedistributionPositive(records, context);
            expectLLEPExpertPayloadMovementPositive(records, context);
            expectMoEExpertMovementPositive(records, context);
            for (const char *error_counter : {
                     "device_rebalance_copy_missing_source_descriptors",
                     "device_rebalance_apply_missing_source_descriptors",
                     "device_rebalance_copy_missing_destination_slots",
                     "device_rebalance_apply_missing_destination_slots",
                     "device_rebalance_copy_descriptor_mismatches",
                     "device_rebalance_apply_descriptor_mismatches",
                     "device_rebalance_copy_invalid_plan_entries",
                     "device_rebalance_apply_invalid_plan_entries",
                     "device_rebalance_controller_last_error_code"})
            {
                expectPerfCounterZero(
                    records,
                    "moe_rebalance",
                    error_counter,
                    context);
            }
            return;
        }

        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_maintenance_graph_launches",
            context);
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_maintenance_graph_diagnostic_exports",
            context);
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_rebalance_controller_maintenance_launches",
            context);
        /*
         * Planner status is intentionally per wave: the last maintenance tick
         * may reject an uneconomical transfer after an earlier wave moved
         * experts successfully. Controller/apply counters retain request-level
         * evidence and therefore certify end-to-end movement without requiring
         * the final wave itself to be non-empty.
         */
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_rebalance_controller_decode_apply_hits",
            context);
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_rebalance_wave_copied_arrivals_total",
            context);
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_rebalance_wave_applied_arrivals_total",
            context);
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_rebalance_wave_applied_layer_count_total",
            context);

        expectDynamicRebalancePlacementPositive(records, context);
        expectMoEExpertMovementPositive(records, context);

        for (const char *error_counter : {
                 "device_rebalance_copy_missing_source_descriptors",
                 "device_rebalance_apply_missing_source_descriptors",
                 "device_rebalance_copy_missing_destination_slots",
                 "device_rebalance_apply_missing_destination_slots",
                 "device_rebalance_copy_descriptor_mismatches",
                 "device_rebalance_apply_descriptor_mismatches",
                 "device_rebalance_copy_invalid_plan_entries",
                 "device_rebalance_apply_invalid_plan_entries",
                 "device_rebalance_controller_last_error_code"})
        {
            expectPerfCounterZero(
                records,
                "moe_rebalance",
                error_counter,
                context);
        }
    }

    /**
     * @brief Prove that real routed weights were loaded in the requested order.
     *
     * Graph parity alone can miss a configuration axis if both the loader and
     * graph accidentally retain the same default.  This assertion consumes the
     * loader's setup-only evidence and requires both physical LocalTP
     * participants to report the selected order and its characteristic packed
     * layout.  The subsequent checkpoint comparison independently proves that
     * the graph owner map agrees with those physical selections.
     */
    void expectRoutedExpertOwnerSelectionPerfPath(
        const ExpertOverlayParityConfig &config,
        const std::vector<PerfStatRecord> &records,
        size_t expected_layer_count)
    {
        const std::string requested_order =
            routedExpertOwnerOrderToString(config.owner_order);
        const std::string requested_layout =
            config.owner_order == RoutedExpertOwnerOrder::Random
                ? "noncontiguous"
                : "contiguous";
        std::set<std::string> participants;
        std::set<std::string> layers;
        size_t matching_records = 0;

        for (const auto &record : records)
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_placement" ||
                record.name != "routed_expert_weight_selection")
            {
                continue;
            }

            const auto order = record.tags.find("owner_order");
            ASSERT_NE(order, record.tags.end());
            EXPECT_EQ(order->second, requested_order)
                << "Physical routed-weight loading used the wrong owner order";
            if (order->second != requested_order)
                continue;

            const auto layout = record.tags.find("selection_layout");
            ASSERT_NE(layout, record.tags.end());
            EXPECT_EQ(layout->second, requested_layout)
                << "Physical routed-weight IDs do not match the requested "
                   "ownership policy";

            const auto participant = record.tags.find("participant");
            const auto layer = record.tags.find("layer");
            ASSERT_NE(participant, record.tags.end());
            ASSERT_NE(layer, record.tags.end());
            participants.insert(participant->second);
            layers.insert(layer->second);
            EXPECT_GT(record.count, 0u);
            EXPECT_GT(record.value, 0.0);
            ++matching_records;
        }

        EXPECT_GT(matching_records, 0u)
            << "No physical routed-expert weight-selection evidence was published\n"
            << PerfStatsCollector::summaryString({"moe_placement"});
        EXPECT_EQ(participants.size(), config.devices.size())
            << "Weight-selection evidence did not cover every LocalTP participant";
        EXPECT_EQ(layers.size(), expected_layer_count)
            << "Weight-selection evidence did not cover every routed MoE layer";
    }

    ExpertOverlayParityConfig baseConfig(
        const std::string &name,
        std::vector<ParityDeviceType> devices,
        Collective backend,
        const std::string &snapshot_dir,
        ExpertOverlayPolicyScenario policy_scenario,
        RoutedExpertOwnerOrder owner_order = RoutedExpertOwnerOrder::Ordinal)
    {
        ExpertOverlayParityConfig config;
        config.name = name;
        config.devices = std::move(devices);
        config.parallelism = Parallelism::LocalTP;
        config.collective = backend;
        config.thresholds = qwen36MoEOverlayThresholds();
        config.model_path = "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf";
        config.snapshot_dir = snapshot_dir;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = KVCachePrecision::FP16;
        config.decode_steps = 3;
        config.policy_scenario = policy_scenario;
        config.owner_order = owner_order;
        config.moe_hot_expert_cache.kind =
            MoEHotExpertCacheConfig::Kind::Off;
        config.moe_rebalance.mode =
            policy_scenario ==
                    ExpertOverlayPolicyScenario::DynamicResidencyMaintenance
                ? MoERebalanceRuntimeMode::Dynamic
                : MoERebalanceRuntimeMode::Off;
        if (config.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
        {
            /*
             * The campaign is a path proof, not a throughput soak. A one-row
             * production routing window makes the first committed decode row
             * eligible for maintenance, preserving real planner/copy/apply
             * arithmetic while avoiding a 32-token diagnostic tail.
             */
            config.moe_rebalance.window_size = 1;
            config.moe_rebalance.max_window_size = 1;
            config.moe_rebalance.window_growth_factor = 1.0f;
            config.moe_rebalance.dynamic_imbalance_threshold_per_mille = 0;
            config.moe_rebalance.dynamic_min_improvement_per_mille = 0;
            config.moe_rebalance.dynamic_max_swaps_per_layer = 20;
            config.moe_rebalance.dynamic_max_plan_entries_per_wave = 20;
            config.moe_rebalance.dynamic_min_window_activations = 0;
            config.moe_rebalance.device_min_load_spread_improvement = 0;
            config.moe_rebalance.device_min_load_spread_improvement_divisor = 0;
            config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot = 0;
            config.moe_rebalance
                .device_min_foreign_rows_per_critical_path_payload_slot = 0;
            config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot = 0;
            config.moe_rebalance.device_max_post_wave_load_spread_per_mille = 1000;
            config.moe_rebalance.device_maintenance_slack_tokens = 0;
            config.moe_rebalance.device_min_maintenance_period_tokens = 1;
            config.moe_rebalance.device_initial_maintenance_period_tokens = 1;
            config.moe_rebalance_exercise = {
                .enabled = true,
                .require_production_overlay_authority = true,
                .request_every_decode_steps = 1,
                .min_decode_steps = 2,
                .require_movement_epoch_advance = true,
                .min_movement_epoch_delta = 1,
            };
        }
        if (policy_scenario == ExpertOverlayPolicyScenario::CurrentBatchLLEP)
        {
            config.moe_routed_prefill = RoutedExpertPrefillRuntimeConfig{
                .assignment_window_tokens = 0,
                .least_loaded_min_routed_rows = 0,
                .llep_alpha_numerator = 9,
                .llep_alpha_denominator = 10,
                .llep_lambda_numerator = 13,
                .llep_lambda_denominator = 10,
                .llep_enable_balanced_skip = false,
            };
        }
        config.moe_rebalance.release_raw_expert_weights = true;
        config.graph_snapshot_policy.enabled = true;
        config.graph_snapshot_policy.require_graph_execution_on_gpu = true;
        config.graph_snapshot_policy.require_snapshot_publication = true;
        config.graph_snapshot_policy.require_prefill_graph_capture_on_gpu = true;
        config.graph_snapshot_policy.retry_prefill_after_warmup_for_capture = true;
        config.graph_snapshot_policy.required_prefill_snapshot_keys = {
            "LM_HEAD",
            "layer0_MOE_COMBINED_OUTPUT",
        };
        config.graph_snapshot_policy.required_decode_snapshot_keys = {
            "LM_HEAD",
            "layer0_MOE_COMBINED_OUTPUT",
        };
        return config;
    }

    /**
     * @brief Build the complete GPU policy-by-owner-order parity matrix.
     *
     * Every cell compares all prefill/decode checkpoints and requires its
     * named movement policy to become physically observable. The production
     * windows are deliberately minimal so the complete cross-backend matrix
     * proves the path without retaining the old long-context soak runtime.
     */
    std::vector<ExpertOverlayParityConfig> makeExpertOverlayConfigs()
    {
        std::vector<ExpertOverlayParityConfig> configs;
        struct BackendFixture
        {
            std::string name;
            std::vector<ParityDeviceType> devices;
            Collective collective;
            std::string short_snapshots;
        };
        const std::array<BackendFixture, 2> backends = {{
            {
                "CUDA2TP",
                {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
                Collective::NCCL,
                "pytorch_qwen36_moe_singledevice_cuda_snapshots",
            },
            {
                "ROCm2TP",
                {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
                Collective::RCCL,
                "pytorch_qwen36_moe_singledevice_rocm_snapshots",
            },
        }};

        for (const auto &backend : backends)
        {
            for (const auto owner_order : {
                     RoutedExpertOwnerOrder::Ordinal,
                     RoutedExpertOwnerOrder::Random})
            {
                const std::string order_name =
                    owner_order == RoutedExpertOwnerOrder::Ordinal
                        ? "OrdinalOwners"
                        : "RandomOwners";
                const auto add_short = [&](const std::string &policy_name,
                                           ExpertOverlayPolicyScenario scenario)
                {
                    configs.push_back(baseConfig(
                        "Qwen36MoE_ExpertOverlay_" + backend.name + "_" +
                            policy_name + "_" + order_name +
                            "_DenseTP_FP16Transport",
                        backend.devices,
                        backend.collective,
                        backend.short_snapshots,
                        scenario,
                        owner_order));
                };

                add_short("Static", ExpertOverlayPolicyScenario::StaticOwnership);
                add_short(
                    "DynamicMaintenance",
                    ExpertOverlayPolicyScenario::DynamicResidencyMaintenance);
                add_short(
                    "CurrentBatchLLEP",
                    ExpertOverlayPolicyScenario::CurrentBatchLLEP);
            }
        }
        return configs;
    }

    const std::vector<ExpertOverlayParityConfig> kQwen36MoEExpertOverlayConfigs =
        makeExpertOverlayConfigs();

    PlanFactory planFactoryForConfig(const TestConfig &config)
    {
        if (config.name.find("CUDA2TP") != std::string::npos)
            return qwen36MoEOverlayPlanCuda2TPHotOnly;
        if (config.name.find("ROCm2TP") != std::string::npos)
            return qwen36MoEOverlayPlanRocm2TPHotOnly;
        throw std::invalid_argument(
            "no Qwen3.6 MoE expert overlay plan factory for config '" +
            config.name + "'");
    }

    std::optional<std::string> expertOverlayHardwareBlocker(const TestConfig &config)
    {
        if (auto blocker = checkHardwareAvailability(config))
            return blocker;
#ifndef HAVE_NCCL
        if (config.collective == Collective::NCCL)
            return "NCCL not available";
#endif
#ifndef HAVE_RCCL
        if (config.collective == Collective::RCCL)
            return "RCCL not available";
#endif
        return std::nullopt;
    }

    /**
     * @brief Build only the declarative placement request consumed by production.
     *
     * Model dimensions, capacity admission, concrete per-layer ownership, and
     * the immutable initial epoch are deliberately absent here. The production
     * `OrchestrationRunner` resolves those values from the live GGUF before it
     * creates the sole ExpertOverlay residency authority.
     *
     * @param config Policy and physical-owner axis for this campaign cell.
     * @return Declarative routed-expert plan for production orchestration.
     */
    std::shared_ptr<MoERoutedExpertPlacementPlan> makeRequestedOverlayPlan(
        const ExpertOverlayParityConfig &config)
    {
        auto requested = planFactoryForConfig(config)();
        if (!requested)
            throw std::invalid_argument("overlay parity plan factory returned null");
        requested->owner_order = config.owner_order;

        for (auto &domain : requested->domains)
        {
            domain.routed_decode_assignment_policy =
                RoutedExpertAssignmentPolicy::StaticOwner;
            domain.routed_prefill_assignment_policy =
                config.policy_scenario ==
                        ExpertOverlayPolicyScenario::CurrentBatchLLEP
                    ? RoutedExpertAssignmentPolicy::LeastLoadedResident
                    : RoutedExpertAssignmentPolicy::StaticOwner;
        }
        return requested;
    }

} // namespace

class Qwen36MoEExpertOverlayParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen36MoEExpertOverlayParityTest>,
      public ::testing::WithParamInterface<ExpertOverlayParityConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen36MoEExpertOverlayParityTest>;

    void SetUp() override
    {
        if (auto blocker = expertOverlayHardwareBlocker(GetParam()))
        {
            if (productionParityCampaignEnabled())
            {
                FAIL() << "Production Qwen3.6 ExpertOverlay prerequisite failed: "
                       << *blocker;
            }
            GTEST_SKIP() << GetParam().name << " " << *blocker;
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size != 1)
        {
            if (productionParityCampaignEnabled())
            {
                FAIL() << "Production Qwen3.6 homogeneous LocalTP ExpertOverlay parity "
                          "requires -np 1 (got "
                       << world_size << ")";
            }
            GTEST_SKIP() << "Qwen3.6 homogeneous LocalTP expert overlay parity "
                         << "must run with -np 1 (got " << world_size << ")";
        }
        if (GetParam().policy_scenario ==
            ExpertOverlayPolicyScenario::CurrentBatchLLEP)
        {
            setScopedParityEnvOverride(
                "LLAMINAR_MOE_GPU_DIRECT_TRANSFER_WAVE_EXPERTS",
                "4");
            setScopedParityEnvOverride(
                "LLAMINAR_MOE_DEVICE_REBALANCE_COMPACT_PAYLOAD_SLOTS",
                "4");
        }
        setScopedParityEnvOverride(
            "LLAMINAR_MOE_GPU_CACHE_EXPERTS_PER_LAYER",
            "0");
        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        Base::SetUp();
    }

    ParityGraphSnapshotPolicy parityGraphSnapshotPolicy(
        ParityForwardPhase phase) const override
    {
        auto policy = Base::parityGraphSnapshotPolicy(phase);
        if (phase == ParityForwardPhase::Prefill)
        {
            /*
             * Final MoE publication is the only routed/shared arithmetic
             * boundary this focused fixture compares.  Requiring every layer
             * here prevents a graph rewrite from silently preserving layer 0
             * while dropping later captured publications.
             */
            for (int layer = 0; layer < parityLayerCount(); ++layer)
            {
                const std::string key =
                    "layer" + std::to_string(layer) +
                    "_MOE_COMBINED_OUTPUT";
                if (std::find(
                        policy.required_prefill_snapshot_keys.begin(),
                        policy.required_prefill_snapshot_keys.end(),
                        key) == policy.required_prefill_snapshot_keys.end())
                {
                    policy.required_prefill_snapshot_keys.push_back(key);
                }
            }
        }
        return policy;
    }

    bool productionParityRequiresPrefillSnapshots() const override
    {
        return true;
    }

    /**
     * @brief Construct this parity cell through the live production lifecycle.
     *
     * ExpertOverlay authority, model-aware capacity admission, initial epoch,
     * participant prepared banks, LocalTP contexts, graph lowering, and capture
     * are all owned by `OrchestrationRunner`. The fixture supplies only the same
     * declarative policy a CLI/config caller would supply, then enables the
     * ordinary diagnostic snapshot boundary after successful initialization.
     *
     * @return `true` after one complete production runner is initialized.
     */
    bool setupProductionOverlayRunner()
    {
        try
        {
            overlay_plan_ = makeRequestedOverlayPlan(GetParam());
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[Qwen3.6 MoE ExpertOverlay MathParity] " << error.what());
            return false;
        }

        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = config_.model_path;
        config.max_seq_len = GetParam().max_seq_len;
        config.batch_size = 1;
        config.tp_degree = 1;
        config.pp_degree = 1;
        config.activation_precision =
            Base::orchestrationActivationPrecisionValue(
                GetParam().activation_precision);
        config.kv_cache_precision =
            Base::orchestrationKVCachePrecisionValue(
                GetParam().kv_cache_precision);
        config.tp_allreduce_precision_override = "schema";
        config.routed_expert_owner_order = GetParam().owner_order;
        config.moe_hot_expert_cache = GetParam().moe_hot_expert_cache;
        config.moe_routed_prefill = GetParam().moe_routed_prefill;
        config.moe_rebalance = GetParam().moe_rebalance;
        config.moe_routed_expert_plan = overlay_plan_;

        // Retire every legacy owner before installing the sole production
        // runner. A model context may only be reached through that runner.
        runner_.reset();
        borrowed_runner_ = nullptr;
        model_ctx_.reset();
        orch_runner_.reset();

        auto factory = createOrchestrationRunnerFactory();
        orch_runner_ = factory->createFromOrchestrationConfig(
            std::move(config));
        if (!orch_runner_)
        {
            LOG_ERROR(
                "[Qwen3.6 MoE ExpertOverlay MathParity] Production factory "
                "did not create an OrchestrationRunner");
            return false;
        }
        if (!orch_runner_->initialize())
        {
            LOG_ERROR(
                "[Qwen3.6 MoE ExpertOverlay MathParity] Production runner "
                "initialization failed: "
                << orch_runner_->lastError());
            orch_runner_.reset();
            return false;
        }

        /*
         * The authenticated Hugging Face pack advances decode with greedy
         * tokens.  Declare that serving policy on the production runner so the
         * prefill-boundary sample and every subsequent decode transaction are
         * the exact operations certified by the reference.  Leaving the model
         * recommendation active would ask the GPU stochastic sampler to prove
         * a different token trajectory (and Qwen3.6's open-ended top-k=0
         * recommendation is intentionally not interpreted as greedy).
         */
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        orch_runner_->setSamplingParams(greedy);
        orch_runner_->enableSnapshotCapture();
        return true;
    }

    bool decodeWorkAvailable()
    {
        return !loadPyTorchSnapshot("decode_step0_LM_HEAD").empty() &&
               !readDecodeTokensFromMetadata().empty();
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> overlay_plan_;
};

TEST(Qwen36MoEExpertOverlayPerfStats, DynamicMovementAcceptsEitherProductionDecisionForm)
{
    const PerfStatRecord accepted_replica{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "moe_rebalance",
        .name = "device_rebalance_selected_replicas",
        .value = 2.0,
    };
    const PerfStatRecord accepted_ownership_swap{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "moe_rebalance",
        .name = "device_rebalance_dynamic_ownership_swap_accepts",
        .value = 1.0,
    };
    const PerfStatRecord rejected_attempt{
        .kind = PerfStatRecord::Kind::Counter,
        .domain = "moe_rebalance",
        .name = "device_rebalance_dynamic_ownership_swap_attempts",
        .value = 7.0,
    };

    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount({accepted_replica}),
        2.0);
    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount({accepted_ownership_swap}),
        1.0);
    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount(
            {accepted_replica, accepted_ownership_swap}),
        3.0);
    EXPECT_EQ(
        dynamicRebalanceAcceptedPlacementCount({rejected_attempt}),
        0.0);
}

TEST_P(Qwen36MoEExpertOverlayParityTest, ProductionParity)
{
    beginProductionParityEvidence();
    {
        auto scope =
            profileParityScope("expert_overlay.setup_production_runner");
        ASSERT_TRUE(setupProductionOverlayRunner())
            << "Production ExpertOverlay runner setup failed";
    }

    ASSERT_TRUE(decodeWorkAvailable())
        << "Production parity requires incremental-decode snapshots and metadata";

    ParityTestSummary prefill;
    {
        auto scope = profileParityScope("expert_overlay.run_prefill_parity");
        prefill = runPrefillParity();
    }
    {
        auto scope = profileParityScope("expert_overlay.assert_prefill_parity");
        assertParity(prefill);
    }
    assertProductionParitySnapshotInfrastructure();
    activeClearSnapshots();
    activeClearCache();

    ASSERT_TRUE(PerfStatsCollector::isEnabled())
        << "ExpertOverlay production parity requires structured graph evidence";
    DecodeParitySummary decode;
    {
        auto scope = profileParityScope("expert_overlay.run_decode_parity");
        decode = runDecodeParity();
    }
    expectRoutedExpertOwnerSelectionPerfPath(
        GetParam(),
        PerfStatsCollector::snapshot({"moe_placement"}),
        static_cast<size_t>(parityLayerCount()));
    const auto rebalance_records =
        PerfStatsCollector::snapshot({"moe_rebalance"});
    if (GetParam().policy_scenario ==
        ExpertOverlayPolicyScenario::StaticOwnership)
    {
        expectNoMoEExpertMovement(
            rebalance_records,
            GetParam().name + " static expert overlay");
    }
    if (GetParam().policy_scenario !=
        ExpertOverlayPolicyScenario::StaticOwnership)
    {
        expectMovementPositiveExpertOverlayPerfPath(
            GetParam(),
            rebalance_records);
    }
    {
        auto scope = profileParityScope("expert_overlay.assert_decode_parity");
        assertDecodeParity(decode);
    }
    finishProductionParityEvidence();
}

/**
 * @brief Proves that a valid multiline long-context corpus is reusable.
 *
 * Long-context prompts are persisted over several physical metadata lines.
 * This regression deliberately uses the same representation as the Python
 * oracle generator and verifies both the positive authentication case and a
 * prompt-drift rejection. It performs no model load or GPU work.
 */
TEST(Qwen36MoEExpertOverlayMetadata, MultilinePromptAuthenticatesWithoutRegeneration)
{
    const std::filesystem::path metadata_path =
        std::filesystem::temp_directory_path() /
        ("llaminar_qwen36_multiline_metadata_" +
         std::to_string(static_cast<long long>(::getpid())) +
         ".txt");
    const std::string prompt =
        "Read the ledger exactly.\n"
        "Ledger item 0001: alpha.\n"
        "Return one JSON object.";

    {
        std::ofstream metadata(metadata_path, std::ios::trunc);
        ASSERT_TRUE(metadata.is_open()) << "failed to create " << metadata_path;
        metadata << "snapshot_version: 4\n"
                 << "prompt: Read the ledger exactly.\n"
                 << "Ledger item 0001: alpha.\n"
                 << "Return one JSON object.\n"
                 << "token_ids: 1,2,3\n"
                 << "decode_steps: 2\n"
                 << "decode_tokens: 4,5\n";
    }

    EXPECT_EQ(
        readMultilineStringFromMetadata(metadata_path, "prompt", "token_ids"),
        std::optional<std::string>{prompt});
    EXPECT_TRUE(metadataLooksUsable(metadata_path, prompt, 2));
    EXPECT_FALSE(metadataLooksUsable(metadata_path, prompt + "\nchanged", 2));

    std::error_code remove_error;
    std::filesystem::remove(metadata_path, remove_error);
    EXPECT_FALSE(remove_error) << "failed to remove " << metadata_path
                               << ": " << remove_error.message();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen36MoEExpertOverlay,
    Qwen36MoEExpertOverlayParityTest,
    ::testing::ValuesIn(kQwen36MoEExpertOverlayConfigs),
    [](const ::testing::TestParamInfo<ExpertOverlayParityConfig> &info)
    {
        return info.param.name;
    });

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    initializeLogging();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
