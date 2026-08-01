/**
 * @file Test__Qwen36MoE_ExpertOverlay_MathParity.cpp
 * @brief PyTorch snapshot parity for Qwen3.6 MoE homogeneous multi-GPU expert overlays.
 *
 * These tests are the quality gate for the GPU MoE rebalancing sprint target:
 * one LocalTP routed expert domain with two CUDA/NCCL or ROCm/RCCL
 * participants in the current fixtures, phase-split dense policy, and fp16 TP
 * allreduce transport. The runner path is generic for LocalTP domains with two
 * or more participants.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../qwen35moe/Qwen35MoEParityTestBase.h"
#include "Qwen36MoEParityTestBase.h"
#include "backends/DeviceAddressAdapter.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "collective/LocalTPContext.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
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

    const std::vector<std::string> kOverlayAllreduceStages = {
        "MOE_COMBINED_OUTPUT",
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
            .allreduce_stages = kOverlayAllreduceStages,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 60.0f,
            .pytorch_top1_in_topk = 4,
        };
    }

    struct ExpertOverlayParityConfig : TestConfig
    {
        MoERebalanceRuntimeConfig moe_rebalance;
        int max_seq_len = 4096;
        bool decode_snapshots_only = false;
        bool require_prompt_metadata_match = false;
    };

    /**
     * @brief Require long-context parity to exercise its named rebalance path.
     *
     * Token parity alone cannot distinguish a healthy Dynamic/LLEP lane from a
     * graph that never planned or applied expert movement. These assertions
     * consume request-local PerfStats after the generic parity epilogue has
     * published the final device status. They deliberately validate planner,
     * transport/apply, and health counters independently so a future failure
     * identifies the missing phase.
     *
     * @param config Concrete backend and MoE runtime policy under test.
     * @param records Request-local maintenance records collected during the
     *        prefill plus 32 committed decode steps.
     */
    void expectLongContextMoERebalancePerfPath(
        const ExpertOverlayParityConfig &config,
        const std::vector<PerfStatRecord> &records)
    {
        const std::string context =
            config.name + " long-context rebalance";

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

        if (config.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
        {
            expectDynamicRebalancePlacementPositive(records, context);
        }
        else if (config.moe_rebalance.mode == MoERebalanceRuntimeMode::LLEP)
        {
            expectLLEPAppliedPrefillMovementPositive(records, context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_planned_arrivals",
                context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_copy_copied_arrivals",
                context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_apply_applied_arrivals",
                context);
        }

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

    ExpertOverlayParityConfig baseConfig(
        const std::string &name,
        std::vector<ParityDeviceType> devices,
        Collective backend,
        const std::string &snapshot_dir,
        MoERebalanceRuntimeMode rebalance_mode)
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
        config.moe_rebalance.mode = rebalance_mode;
        config.moe_rebalance.window_size = 4;
        config.moe_rebalance.max_window_size = 4;
        config.moe_rebalance.window_growth_factor = 1.0f;
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
        config.moe_rebalance.device_maintenance_slack_tokens = 0;
        config.moe_rebalance.device_min_maintenance_period_tokens = 4;
        config.moe_rebalance.device_initial_maintenance_period_tokens = 4;
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

    ExpertOverlayParityConfig longContextConfig(
        const ExpertOverlayParityConfig &base,
        const std::string &name,
        const std::string &snapshot_dir)
    {
        ExpertOverlayParityConfig config = base;
        config.name = name;
        config.prompt = qwen36MoELongNeedleParityPrompt();
        config.snapshot_dir = snapshot_dir;
        config.decode_steps = 32;
        config.max_seq_len = 4096;
        config.decode_snapshots_only = true;
        config.require_prompt_metadata_match = true;
        const auto rebalance_mode = config.moe_rebalance.mode;
        if (rebalance_mode != MoERebalanceRuntimeMode::Dynamic &&
            rebalance_mode != MoERebalanceRuntimeMode::LLEP)
        {
            throw std::invalid_argument(
                "long-context expert-overlay parity requires Dynamic or LLEP rebalance mode");
        }

        /*
         * Both planners are maintained at the same committed decode boundary.
         * Keeping this exercise mode-independent is important: a token-parity
         * pass cannot prove that either planner launched, transferred payloads,
         * or published its device-owned result.
         */
        config.moe_rebalance_exercise.enabled = true;
        config.moe_rebalance_exercise.require_device_side_controller = true;
        config.moe_rebalance_exercise.request_every_decode_steps = 4;
        config.moe_rebalance_exercise.min_decode_steps = 8;
        config.moe_rebalance_exercise.require_movement_epoch_advance = true;
        config.moe_rebalance_exercise.min_movement_epoch_delta = 1;
        return config;
    }

    const std::vector<ExpertOverlayParityConfig> kQwen36MoEExpertOverlayConfigs = {
        baseConfig(
            "Qwen36MoE_ExpertOverlay_CUDA2TP_Dynamic_PhaseSplit_FP16Transport",
            {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
            Collective::NCCL,
            "pytorch_qwen36_moe_singledevice_cuda_snapshots",
            MoERebalanceRuntimeMode::Dynamic),
        baseConfig(
            "Qwen36MoE_ExpertOverlay_ROCm2TP_Dynamic_PhaseSplit_FP16Transport",
            {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
            Collective::RCCL,
            "pytorch_qwen36_moe_singledevice_rocm_snapshots",
            MoERebalanceRuntimeMode::Dynamic),
        baseConfig(
            "Qwen36MoE_ExpertOverlay_CUDA2TP_LLEP_PhaseSplit_FP16Transport",
            {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
            Collective::NCCL,
            "pytorch_qwen36_moe_singledevice_cuda_snapshots",
            MoERebalanceRuntimeMode::LLEP),
        baseConfig(
            "Qwen36MoE_ExpertOverlay_ROCm2TP_LLEP_PhaseSplit_FP16Transport",
            {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
            Collective::RCCL,
            "pytorch_qwen36_moe_singledevice_rocm_snapshots",
            MoERebalanceRuntimeMode::LLEP),
        longContextConfig(
            baseConfig(
                "Qwen36MoE_ExpertOverlay_CUDA2TP_Dynamic_PhaseSplit_LongDecode_FP16Transport",
                {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
                Collective::NCCL,
                "pytorch_qwen36_moe_expert_overlay_cuda2_dynamic_phase_split_long_decode_snapshots",
                MoERebalanceRuntimeMode::Dynamic),
            "Qwen36MoE_ExpertOverlay_CUDA2TP_Dynamic_PhaseSplit_LongDecode_FP16Transport",
            "pytorch_qwen36_moe_expert_overlay_cuda2_dynamic_phase_split_long_decode_snapshots"),
        longContextConfig(
            baseConfig(
                "Qwen36MoE_ExpertOverlay_ROCm2TP_Dynamic_PhaseSplit_LongDecode_FP16Transport",
                {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
                Collective::RCCL,
                "pytorch_qwen36_moe_expert_overlay_rocm2_dynamic_phase_split_long_decode_snapshots",
                MoERebalanceRuntimeMode::Dynamic),
            "Qwen36MoE_ExpertOverlay_ROCm2TP_Dynamic_PhaseSplit_LongDecode_FP16Transport",
            "pytorch_qwen36_moe_expert_overlay_rocm2_dynamic_phase_split_long_decode_snapshots"),
        longContextConfig(
            baseConfig(
                "Qwen36MoE_ExpertOverlay_CUDA2TP_LLEP_PhaseSplit_LongDecode_FP16Transport",
                {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
                Collective::NCCL,
                "pytorch_qwen36_moe_expert_overlay_cuda2_llep_phase_split_long_decode_snapshots",
                MoERebalanceRuntimeMode::LLEP),
            "Qwen36MoE_ExpertOverlay_CUDA2TP_LLEP_PhaseSplit_LongDecode_FP16Transport",
            "pytorch_qwen36_moe_expert_overlay_cuda2_llep_phase_split_long_decode_snapshots"),
        longContextConfig(
            baseConfig(
                "Qwen36MoE_ExpertOverlay_ROCm2TP_LLEP_PhaseSplit_LongDecode_FP16Transport",
                {ParityDeviceType::ROCm, ParityDeviceType::ROCm},
                Collective::RCCL,
                "pytorch_qwen36_moe_expert_overlay_rocm2_llep_phase_split_long_decode_snapshots",
                MoERebalanceRuntimeMode::LLEP),
            "Qwen36MoE_ExpertOverlay_ROCm2TP_LLEP_PhaseSplit_LongDecode_FP16Transport",
            "pytorch_qwen36_moe_expert_overlay_rocm2_llep_phase_split_long_decode_snapshots"),
    };

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

    MoERoutedExpertModelMetadata metadataFromModel(const ModelContext &ctx)
    {
        const auto &loader = ctx.concreteLoader();
        const std::string &arch = ctx.architecture();

        MoERoutedExpertModelMetadata metadata;
        metadata.num_layers = mainLayerCountExcludingMTP(
            loader,
            arch,
            ctx.blockCount());
        metadata.num_experts = loader.getInt(arch + ".expert_count", 0);
        metadata.d_model = ctx.embeddingLength();
        metadata.routed_intermediate_size =
            loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (metadata.routed_intermediate_size == 0)
            metadata.routed_intermediate_size = ctx.feedForwardLength();
        metadata.has_shared_expert = loader.getInt(arch + ".expert_shared_count", 0) > 0;
        metadata.shared_intermediate_size = metadata.has_shared_expert
                                                ? metadata.routed_intermediate_size
                                                : 0;
        metadata.routed_quant_type = "IQ3";
        metadata.shared_quant_type = "IQ3";
        return metadata;
    }

    std::string validationErrors(const MoERoutedExpertPlacementValidationResult &validation)
    {
        std::ostringstream message;
        for (const auto &error : validation.errors)
            message << "\n - " << error;
        return message.str();
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> makePlannedOverlayPlan(
        const TestConfig &config,
        const ModelContext &ctx)
    {
        const auto metadata = metadataFromModel(ctx);
        auto requested = planFactoryForConfig(config)();
        if (!requested)
            throw std::invalid_argument("overlay parity plan factory returned null");

        auto planned = MoERoutedExpertPlacementPlanner::plan(*requested, metadata).planned_plan;

        MoERoutedExpertPlacementValidationOptions options;
        options.layer_count = metadata.num_layers;
        options.routed_expert_count = metadata.num_experts;
        auto validation = validateMoERoutedExpertPlacementPlan(planned, options);
        if (!validation.ok())
        {
            throw std::invalid_argument(
                "invalid planned Qwen3.6 MoE expert overlay:" +
                validationErrors(validation));
        }

        return std::make_shared<MoERoutedExpertPlacementPlan>(std::move(planned));
    }

    DeviceId continuationRootDevice(const MoERoutedExpertPlacementPlan &plan)
    {
        for (const auto &domain : plan.domains)
        {
            if (domain.name != plan.continuation_domain)
                continue;
            if (domain.participants.empty())
            {
                throw std::runtime_error(
                    "continuation domain '" + plan.continuation_domain +
                    "' has no participants");
            }

            const int root = std::clamp(
                plan.continuation_domain_spec.logical_root_participant,
                0,
                static_cast<int>(domain.participants.size()) - 1);
            return DeviceAddressAdapter::toDeviceId(
                domain.participants[static_cast<size_t>(root)]);
        }

        throw std::runtime_error(
            "continuation domain '" + plan.continuation_domain +
            "' was not found in overlay plan");
    }

    const RoutedExpertDomain &continuationDomain(const MoERoutedExpertPlacementPlan &plan)
    {
        for (const auto &domain : plan.domains)
        {
            if (domain.name == plan.continuation_domain)
                return domain;
        }

        throw std::runtime_error(
            "continuation domain '" + plan.continuation_domain +
            "' was not found in overlay plan");
    }

    std::vector<float> equalWeights(size_t count)
    {
        if (count == 0)
            return {};
        return std::vector<float>(count, 1.0f / static_cast<float>(count));
    }

    struct CachedExpertOverlayPipeline
    {
        std::shared_ptr<ModelContext> model_ctx;
        std::shared_ptr<MoERoutedExpertPlacementPlan> overlay_plan;
        std::unique_ptr<IInferenceRunner> runner;
        std::string key;
    };

    std::mutex &overlayPipelineCacheMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    std::map<std::string, std::shared_ptr<ModelContext>> &overlayModelContextCache()
    {
        static std::map<std::string, std::shared_ptr<ModelContext>> cache;
        return cache;
    }

    std::map<std::string, std::unique_ptr<CachedExpertOverlayPipeline>> &overlayPipelineCache()
    {
        static std::map<std::string, std::unique_ptr<CachedExpertOverlayPipeline>> cache;
        return cache;
    }

    int overlayPipelineCacheMaxEntries()
    {
        const char *raw = std::getenv("LLAMINAR_PARITY_PIPELINE_CACHE_MAX");
        if (!raw || !*raw)
            return 0;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || parsed < 0)
            return 0;
        return static_cast<int>(parsed);
    }

    bool overlayModelContextCanSeedNewRunner(
        const ExpertOverlayParityConfig &config)
    {
        return !config.moe_rebalance.release_raw_expert_weights;
    }

    bool overlayPipelineCanKeepDistinctRunnerEntries(
        const ExpertOverlayParityConfig &config)
    {
        /*
         * A release-raw expert-overlay runner owns the only valid prepared expert
         * payloads and the resident multi-participant Qwen3.6 overlay footprint
         * is too large to keep side-by-side variants alive while constructing the
         * next runner.
         * Cache hits may reuse the exact runner; cache misses must evict first.
         */
        return overlayModelContextCanSeedNewRunner(config);
    }

    std::string graphCaptureBucketKeyFragment()
    {
        const auto &exec = debugEnv().execution;
        std::ostringstream key;
        key << "|gpu_graphs=" << (exec.gpu_graphs ? 1 : 0)
            << "|prefill_min=" << exec.prefill_graph_min_seq
            << "|prefill_buckets=" << (exec.prefill_graph_buckets ? 1 : 0)
            << "|prefill_bucket_sizes=";
        for (size_t i = 0; i < exec.prefill_graph_bucket_sizes.size(); ++i)
        {
            if (i > 0)
                key << ',';
            key << exec.prefill_graph_bucket_sizes[i];
        }
        return key.str();
    }

    std::string overlayPipelineCacheKey(
        const ExpertOverlayParityConfig &config,
        const ParityConfig &resolved_config)
    {
        std::ostringstream key;
        key << config.name
            << "|model=" << config.model_path
            << "|max_seq=" << config.max_seq_len
            << "|decode_steps=" << config.decode_steps
            << "|resolved_snapshot_dir=" << resolved_config.snapshot_dir
            << "|resolved_prefill_tokens=" << resolved_config.token_ids.size()
            << "|rebalance=" << static_cast<int>(config.moe_rebalance.mode)
            << "|backend=" << static_cast<int>(config.collective)
            << "|activation=" << static_cast<int>(config.activation_precision)
            << "|kv=" << static_cast<int>(config.kv_cache_precision)
            << graphCaptureBucketKeyFragment();
        return key.str();
    }

    void clearOverlayPipelineRunnerCache()
    {
        auto &pipeline_cache = overlayPipelineCache();
        for (auto &[key, entry] : pipeline_cache)
        {
            (void)key;
            if (entry && entry->runner)
            {
                entry->runner->clearSnapshots();
                entry->runner->clear_cache();
                entry->runner.reset();
            }
        }
        llaminar::v2::kernels::KernelFactory::clearCache();
        pipeline_cache.clear();

        /*
         * Expert-overlay parity enables release_raw_expert_weights to match the
         * production GPU memory profile. Once a runner has consumed and released
         * those raw payloads, the owning ModelContext is no longer a valid source
         * for constructing a different runner shape. Drop it with the runner cache
         * rather than preserving a context whose expert payloads may now be empty.
         */
        overlayModelContextCache().clear();
    }

    std::shared_ptr<ModelContext> getOrCreateOverlayModelContext(
        const ExpertOverlayParityConfig &config,
        const std::shared_ptr<IMPIContext> &mpi_ctx,
        const std::function<void(const std::shared_ptr<ModelContext> &)> &configure)
    {
        auto &cache = overlayModelContextCache();
        const bool can_seed_new_runner =
            overlayModelContextCanSeedNewRunner(config);
        if (can_seed_new_runner)
        {
            if (auto it = cache.find(config.model_path); it != cache.end())
                return it->second;
        }
        else if (!cache.empty())
        {
            cache.clear();
        }

        if (cache.empty() && overlayPipelineCache().empty())
        {
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        const bool gpu_only = std::all_of(
            config.devices.begin(),
            config.devices.end(),
            [](ParityDeviceType device)
            {
                return device == ParityDeviceType::CUDA ||
                       device == ParityDeviceType::ROCm;
            });
        if (!gpu_only)
        {
            throw std::invalid_argument(
                "expert-overlay parity model context requires GPU-only participants");
        }

        /*
         * The legacy ModelContext::create overload defaults to a CPU target. That
         * makes ModelLoader NUMA-bind and first-touch every page in the GGUF
         * before the bounded GPU upload ring can consume its first weight. Use
         * the explicit configuration so GPU-only parity follows production's
         * demand-paged mmap path and keeps host staging within its configured
         * budget.
         */
        const ModelContextConfig model_config{
            .mpi_ctx = mpi_ctx,
            .strategy = WeightDistributionStrategy::SHARDED,
            .use_mmap = true,
            .target_is_gpu = true,
        };
        auto model_ctx = ModelContext::create(config.model_path, model_config);
        if (model_ctx)
        {
            configure(model_ctx);
            if (can_seed_new_runner)
                cache.emplace(config.model_path, model_ctx);
        }
        return model_ctx;
    }

    void evictOverlayPipelineCacheIfNeeded(const std::string &protected_key)
    {
        const int max_entries = overlayPipelineCacheMaxEntries();
        auto &cache = overlayPipelineCache();
        if (max_entries <= 0)
        {
            cache.clear();
            return;
        }
        while (static_cast<int>(cache.size()) > max_entries)
        {
            auto victim = std::find_if(
                cache.begin(),
                cache.end(),
                [&protected_key](const auto &entry)
                {
                    return entry.first != protected_key;
                });
            if (victim == cache.end())
                break;
            cache.erase(victim);
        }
    }

    void clearExpertOverlayPipelineCache()
    {
        std::lock_guard<std::mutex> lock(overlayPipelineCacheMutex());
        clearOverlayPipelineRunnerCache();
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
            GTEST_SKIP() << GetParam().name << " " << *blocker;

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size != 1)
        {
            GTEST_SKIP() << "Qwen3.6 homogeneous LocalTP expert overlay parity "
                         << "must run with -np 1 (got " << world_size << ")";
        }
        if (GetParam().moe_rebalance.mode == MoERebalanceRuntimeMode::LLEP)
        {
            setScopedParityEnvOverride("LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE", "full");
            setScopedParityEnvOverride("LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS", "0");
            setScopedParityEnvOverride("LLAMINAR_MOE_GPU_DIRECT_TRANSFER_WAVE_EXPERTS", "32");
            setScopedParityEnvOverride("LLAMINAR_MOE_DEVICE_REBALANCE_COMPACT_PAYLOAD_SLOTS", "32");
        }
        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        Base::SetUp();
    }

    bool preserveParityPipelineCachesBetweenTests() const override
    {
        return overlayPipelineCacheMaxEntries() > 0;
    }

    ParityGraphSnapshotPolicy parityGraphSnapshotPolicy(
        ParityForwardPhase phase) const override
    {
        auto policy = Base::parityGraphSnapshotPolicy(phase);
        if (GetParam().decode_snapshots_only && phase == ParityForwardPhase::Prefill)
        {
            policy.required_prefill_snapshot_keys = {
                "layer0_MOE_COMBINED_OUTPUT",
            };
            policy.prefill_snapshot_capture_filter = policy.required_prefill_snapshot_keys;
        }
        return policy;
    }

    void applyModelOverrides() override
    {
        Base::applyModelOverrides();

        if (!GetParam().require_prompt_metadata_match)
            return;

        const auto metadata_path =
            std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        if (metadataLooksUsable(
                metadata_path,
                config_.prompt,
                config_.decode_steps))
        {
            return;
        }

        if (!std::filesystem::exists(config_.model_path))
            return;

        LOG_INFO("[Qwen3.6 MoE ExpertOverlay MathParity] Regenerating snapshots for prompt/decode-matched metadata: "
                 << config_.snapshot_dir);
        if (!regeneratePyTorchSnapshots())
        {
            ADD_FAILURE() << "Qwen3.6 MoE expert overlay snapshot regeneration failed";
            return;
        }

        auto prefill_tokens = readPrefillTokensFromMetadata();
        if (!prefill_tokens.empty())
        {
            config_.token_ids = std::move(prefill_tokens);
            LOG_INFO("[Qwen3.6 MoE ExpertOverlay MathParity] Loaded "
                     << config_.token_ids.size()
                     << " prompt token IDs from regenerated metadata");
        }
    }

    bool regeneratePyTorchSnapshots() override
    {
        if (!GetParam().decode_snapshots_only)
            return Base::regeneratePyTorchSnapshots();

        std::string output;
        const auto metadata_path =
            std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const bool ok = regenerateQwen36MoEDecodeSnapshots(
            config_.model_path,
            metadata_path,
            config_.prompt,
            config_.decode_steps,
            false,
            &output);
        if (!ok)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] decode-only PyTorch snapshot generation failed:\n"
                      << output);
        }
        return ok;
    }

    bool setupPipeline()
    {
        const std::string cache_key = overlayPipelineCacheKey(GetParam(), config_);
        {
            auto scope = profileParityScope("expert_overlay.setup.pipeline_cache_lookup");
            std::lock_guard<std::mutex> lock(overlayPipelineCacheMutex());
            auto &cache = overlayPipelineCache();
            if (auto it = cache.find(cache_key); it != cache.end() &&
                it->second && it->second->runner)
            {
                model_ctx_ = it->second->model_ctx;
                overlay_plan_ = it->second->overlay_plan;
                borrowParityPipeline(model_ctx_, it->second->runner.get());
                activeClearSnapshots();
                activeClearCache();
                return true;
            }
            const int max_entries = overlayPipelineCacheMaxEntries();
            if (max_entries <= 0 ||
                static_cast<int>(cache.size()) >= max_entries ||
                (!cache.empty() &&
                 !overlayPipelineCanKeepDistinctRunnerEntries(GetParam())))
            {
                clearOverlayPipelineRunnerCache();
            }
        }

        {
            auto scope = profileParityScope("expert_overlay.setup.device_manager_initialize");
            DeviceManager::instance().initialize(-1);
        }

        {
            auto scope = profileParityScope("expert_overlay.setup.model_context_create");
            std::lock_guard<std::mutex> lock(overlayPipelineCacheMutex());
            model_ctx_ = getOrCreateOverlayModelContext(
                GetParam(),
                mpi_ctx_,
                [this](const std::shared_ptr<ModelContext> &ctx)
                {
                    configureModel(ctx);
                });
        }
        if (!model_ctx_)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] Failed to load model");
            return false;
        }

        try
        {
            auto scope = profileParityScope("expert_overlay.setup.plan_overlay");
            overlay_plan_ = makePlannedOverlayPlan(GetParam(), *model_ctx_);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] " << e.what());
            return false;
        }

        InferenceRunnerConfig inf_config;
        inf_config.max_seq_len = GetParam().max_seq_len;
        inf_config.batch_size = 1;
        inf_config.force_graph = true;
        inf_config.activation_precision = GetParam().activation_precision;
        inf_config.kv_cache_precision = GetParam().kv_cache_precision;
        // Preserve Qwen35Graph's hybrid schema policy: early GDN layers and all
        // FA layers use FP32 allreduce, later GDN layers use FP16 transport.
        inf_config.tp_allreduce_precision_override = "schema";
        inf_config.use_mapped_memory = true;
        inf_config.moe_routed_expert_plan = overlay_plan_;
        inf_config.moe_expert_overlay_mpi_ctx = mpi_ctx_;
        inf_config.moe_rebalance = GetParam().moe_rebalance;

        const RoutedExpertDomain *domain = nullptr;
        try
        {
            (void)continuationRootDevice(*overlay_plan_);
            domain = &continuationDomain(*overlay_plan_);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] " << e.what());
            return false;
        }

        if (!domain || domain->scope != ExecutionDomainScope::LOCAL || domain->participants.size() < 2)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] continuation domain must be a multi-device LocalTP domain");
            return false;
        }

        const std::vector<float> weights =
            domain->weights.empty() ? equalWeights(domain->participants.size()) : domain->weights;
        std::unique_ptr<ILocalTPContext> tp_ctx;
        {
            auto scope = profileParityScope("expert_overlay.setup.create_local_tp_context");
            tp_ctx = createLocalTPContext(
                domain->participants,
                weights,
                domain->backend);
        }
        if (!tp_ctx)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] Failed to create LocalTP context");
            return false;
        }

        RankOrchestrator::Config rank_config;
        rank_config.devices = domain->participants;
        rank_config.weights = weights;
        rank_config.backend = domain->backend;
        rank_config.max_seq_len = inf_config.max_seq_len;
        rank_config.batch_size = inf_config.batch_size;
        rank_config.activation_precision = inf_config.activation_precision;
        rank_config.kv_cache_precision = inf_config.kv_cache_precision;
        rank_config.tp_allreduce_precision_override = inf_config.tp_allreduce_precision_override;
        rank_config.use_mapped_memory = inf_config.use_mapped_memory;
        rank_config.moe_routed_expert_plan = overlay_plan_;
        rank_config.moe_expert_overlay_mpi_ctx = mpi_ctx_;
        rank_config.moe_rebalance = GetParam().moe_rebalance;

        {
            auto scope = profileParityScope("expert_overlay.setup.create_rank_orchestrator");
            runner_ = createRankOrchestrator(
                model_ctx_,
                std::move(tp_ctx),
                rank_config);
        }
        if (!runner_)
        {
            LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] Failed to create runner");
            return false;
        }

        {
            auto scope = profileParityScope("expert_overlay.setup.enable_snapshot_capture");
            runner_->enableSnapshotCapture();
        }

        if (overlayPipelineCacheMaxEntries() <= 0)
            return true;

        {
            auto scope = profileParityScope("expert_overlay.setup.pipeline_cache_store");
            auto entry = std::make_unique<CachedExpertOverlayPipeline>();
            entry->model_ctx = model_ctx_;
            entry->overlay_plan = overlay_plan_;
            entry->runner = std::move(runner_);
            entry->key = cache_key;
            IInferenceRunner *borrowed = entry->runner.get();

            std::lock_guard<std::mutex> lock(overlayPipelineCacheMutex());
            auto &cache = overlayPipelineCache();
            cache[cache_key] = std::move(entry);
            evictOverlayPipelineCacheIfNeeded(cache_key);
            auto it = cache.find(cache_key);
            if (it == cache.end() || !it->second || !it->second->runner)
            {
                LOG_ERROR("[Qwen3.6 MoE ExpertOverlay MathParity] Cached pipeline was evicted immediately; "
                          "increase LLAMINAR_PARITY_PIPELINE_CACHE_MAX or set it to at least 1");
                return false;
            }
            borrowed = it->second->runner.get();
            borrowParityPipeline(model_ctx_, borrowed);
        }
        activeClearSnapshots();
        activeClearCache();
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

TEST(Qwen36MoEExpertOverlayPipelineCacheKey, IncludesResolvedPrefillShapeAndGraphBucketPolicy)
{
    auto config = baseConfig(
        "Qwen36MoE_ExpertOverlay_CUDA2TP_Dynamic_PhaseSplit_FP16Transport",
        {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        Collective::NCCL,
        "snapshots_a",
        MoERebalanceRuntimeMode::Dynamic);

    ParityConfig short_prompt;
    short_prompt.snapshot_dir = "snapshots_a";
    short_prompt.token_ids.assign(9, 1);

    ParityConfig long_prompt = short_prompt;
    long_prompt.token_ids.assign(64, 1);

    const std::string short_key = overlayPipelineCacheKey(config, short_prompt);
    const std::string long_key = overlayPipelineCacheKey(config, long_prompt);
    EXPECT_NE(short_key, long_key)
        << "A borrowed parity runner with request-shaped activation graph state "
           "must not survive into a different prefill shape.";

    auto save_env = [](const char *name) -> std::optional<std::string>
    {
        const char *value = std::getenv(name);
        if (!value)
            return std::nullopt;
        return std::string(value);
    };
    auto restore_env = [](const char *name, const std::optional<std::string> &value)
    {
        if (value)
            setenv(name, value->c_str(), 1);
        else
            unsetenv(name);
    };

    const auto old_min_seq = save_env("LLAMINAR_PREFILL_GRAPH_MIN_SEQ");
    const auto old_buckets_enabled = save_env("LLAMINAR_PREFILL_GRAPH_BUCKETS");
    const auto old_bucket_sizes = save_env("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES");

    setenv("LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "9", 1);
    setenv("LLAMINAR_PREFILL_GRAPH_BUCKETS", "1", 1);
    setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "9", 1);
    mutableDebugEnv().reload();
    const std::string exact_short_bucket_key = overlayPipelineCacheKey(config, short_prompt);

    setenv("LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "64", 1);
    setenv("LLAMINAR_PREFILL_GRAPH_BUCKETS", "1", 1);
    setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64", 1);
    mutableDebugEnv().reload();
    const std::string exact_long_bucket_key = overlayPipelineCacheKey(config, short_prompt);

    restore_env("LLAMINAR_PREFILL_GRAPH_MIN_SEQ", old_min_seq);
    restore_env("LLAMINAR_PREFILL_GRAPH_BUCKETS", old_buckets_enabled);
    restore_env("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", old_bucket_sizes);
    mutableDebugEnv().reload();

    EXPECT_NE(exact_short_bucket_key, exact_long_bucket_key)
        << "The borrowed pipeline key must include the graph-capture bucket "
           "contract because it controls activation arena sizing.";
}

TEST(Qwen36MoEExpertOverlayPipelineCacheKey, RunnerCacheClearDropsPotentiallyStrippedModelContext)
{
    {
        std::lock_guard<std::mutex> lock(overlayPipelineCacheMutex());
        overlayModelContextCache()["dummy-model"] = std::shared_ptr<ModelContext>{};
        ASSERT_FALSE(overlayModelContextCache().empty());
        clearOverlayPipelineRunnerCache();
        EXPECT_TRUE(overlayModelContextCache().empty())
            << "release_raw_expert_weights can strip raw expert payloads from "
               "the cached ModelContext, so runner-cache eviction must drop "
               "the model context as well.";
    }
}

TEST(Qwen36MoEExpertOverlayPipelineCacheKey, RawExpertReleaseForbidsModelContextReuseForNewRunner)
{
    auto config = baseConfig(
        "Qwen36MoE_ExpertOverlay_CUDA2TP_Dynamic_PhaseSplit_FP16Transport",
        {ParityDeviceType::CUDA, ParityDeviceType::CUDA},
        Collective::NCCL,
        "snapshots_a",
        MoERebalanceRuntimeMode::Dynamic);

    config.moe_rebalance.release_raw_expert_weights = true;
    EXPECT_FALSE(overlayModelContextCanSeedNewRunner(config))
        << "A consumed ModelContext cannot seed another runner after raw "
           "expert payload release.";
    EXPECT_FALSE(overlayPipelineCanKeepDistinctRunnerEntries(config))
        << "Release-raw GPU overlay runners may only be reused on exact cache "
           "hits; cache misses must evict before constructing another runner.";

    config.moe_rebalance.release_raw_expert_weights = false;
    EXPECT_TRUE(overlayModelContextCanSeedNewRunner(config))
        << "ModelContext reuse is only valid when raw expert payloads remain "
           "available for subsequent runner construction.";
    EXPECT_TRUE(overlayPipelineCanKeepDistinctRunnerEntries(config));
}

TEST_P(Qwen36MoEExpertOverlayParityTest, PrefillParity)
{
    if (GetParam().decode_snapshots_only)
        GTEST_SKIP() << "decode-only long-context config";
    {
        auto scope = profileParityScope("expert_overlay.setup_pipeline");
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    }
    ParityTestSummary summary;
    {
        auto scope = profileParityScope("expert_overlay.run_prefill_parity");
        summary = runPrefillParity();
    }
    {
        auto scope = profileParityScope("expert_overlay.assert_prefill_parity");
        assertParity(summary);
    }
}

TEST_P(Qwen36MoEExpertOverlayParityTest, DecodeParity)
{
    if (GetParam().decode_snapshots_only)
        GTEST_SKIP() << "decode-only long-context config";
    {
        auto scope = profileParityScope("expert_overlay.setup_pipeline");
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    }
    if (!decodeWorkAvailable())
        GTEST_SKIP() << "Decode snapshots or decode token metadata are unavailable";
    DecodeParitySummary summary;
    {
        auto scope = profileParityScope("expert_overlay.run_decode_parity");
        summary = runDecodeParity();
    }
    {
        auto scope = profileParityScope("expert_overlay.assert_decode_parity");
        assertDecodeParity(summary);
    }
}

TEST_P(Qwen36MoEExpertOverlayParityTest, LongContextDecodeParity)
{
    if (!GetParam().decode_snapshots_only)
        GTEST_SKIP() << "short-context config";
    {
        auto scope = profileParityScope("expert_overlay.setup_pipeline");
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    }
    if (!decodeWorkAvailable())
        GTEST_SKIP() << "Decode snapshots or decode token metadata are unavailable";
    ASSERT_TRUE(PerfStatsCollector::isEnabled())
        << "Long-context ExpertOverlay parity requires PerfStats counter "
           "collection so Dynamic/LLEP planner, movement, and health paths "
           "are certified.";
    PerfStatsCollector::reset();
    DecodeParitySummary summary;
    {
        auto scope = profileParityScope("expert_overlay.run_decode_parity");
        summary = runDecodeParity();
    }
    const auto rebalance_records =
        PerfStatsCollector::snapshot({"moe_rebalance"});
    expectLongContextMoERebalancePerfPath(
        GetParam(),
        rebalance_records);
    {
        auto scope = profileParityScope("expert_overlay.assert_decode_parity");
        assertDecodeParity(summary);
    }
}

TEST_P(Qwen36MoEExpertOverlayParityTest, SnapshotInfrastructure)
{
    if (GetParam().decode_snapshots_only)
        GTEST_SKIP() << "decode-only long-context config";
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    auto embedding = loadPyTorchSnapshot("EMBEDDING");
    ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";

    auto policy = parityGraphSnapshotPolicy(ParityForwardPhase::Prefill);
    const int terminal_layer = std::max(0, parityLayerCount() - 1);
    policy.required_prefill_snapshot_keys = {
        "EMBEDDING",
        "LM_HEAD",
        "layer" + std::to_string(terminal_layer) + "_FFN_RESIDUAL",
    };
    policy.prefill_snapshot_capture_filter = policy.required_prefill_snapshot_keys;

    std::vector<int> snapshot_tokens;
    try
    {
        snapshot_tokens = makeGraphCaptureEligiblePrefillTokens(
            config_.token_ids,
            GetParam().max_seq_len);
    }
    catch (const std::exception &e)
    {
        FAIL() << e.what();
    }
    ASSERT_FALSE(snapshot_tokens.empty());

    ASSERT_TRUE(runParityForwardWithPolicy(
        ParityForwardPhase::Prefill,
        snapshot_tokens.data(),
        static_cast<int>(snapshot_tokens.size()),
        policy));

    auto keys = activeSnapshotKeys();
    EXPECT_GT(keys.size(), 0) << "No snapshots captured";

    EXPECT_NE(std::find(keys.begin(), keys.end(), "EMBEDDING"), keys.end())
        << "Missing EMBEDDING snapshot";
    EXPECT_NE(std::find(keys.begin(), keys.end(), "LM_HEAD"), keys.end())
        << "Missing LM_HEAD snapshot";

    bool has_ffn_residual = false;
    for (const auto &key : keys)
    {
        if (key.find("FFN_RESIDUAL") != std::string::npos)
        {
            has_ffn_residual = true;
            break;
        }
    }
    EXPECT_TRUE(has_ffn_residual) << "Missing FFN_RESIDUAL snapshot";
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

    clearExpertOverlayPipelineCache();
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
