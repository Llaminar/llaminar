/**
 * @file RankOrchestrator.cpp
 * @brief Multi-device orchestrator implementation for LOCAL tensor parallelism
 * @author David Sanftenberg
 * @date January 2026
 *
 * Implements coordination of multiple DeviceGraphOrchestrator instances for LOCAL
 * tensor parallelism across multiple devices within a single MPI rank.
 *
 * Key features:
 * - Parallel forward pass execution across devices via TPWorkerPool (TP) / std::async (PP)
 * - AllGather for combining partial logits from column-parallel LM head
 * - Unified snapshot/profiling API across all device runners
 */

#include "RankOrchestrator.h"
#include "LogitsGatherer.h"
#include "DeviceSampler.h"
#include "DeviceGraphOrchestrator.h"
#include "../../factory/InferenceRunnerFactory.h"
#include "../../prefix_cache/PrefixCacheCoordinator.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../collective/ILocalPPContext.h"
#include "../../../config/TensorParallelConfig.h"
#include "../../../interfaces/IModelContext.h"
#include "../../../loaders/ModelContext.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../loaders/WeightManager.h"
#include "../graph/SchemaFactoryRegistry.h" // Model-agnostic sharding config access
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../backends/BackendManager.h"       // getBackendFor() for partial D2H in gatherLogits
#include "../../../backends/GPUDeviceContextPool.h" // Compute stream registration for event-based collective sync
#include "../../../utils/Logger.h"
#include "../../../utils/Sampler.h"            // SamplingParams for sampleOnDevice()
#include "../../../utils/KernelProfiler.h"     // Phase propagation to worker threads
#include "../../../utils/ROCmKernelProfiler.h" // Phase propagation to worker threads
#include "../../../utils/CUDAKernelProfiler.h" // Phase propagation to worker threads
#include "../../../utils/KVCacheProfiler.h"    // Phase propagation to worker threads
#include "../../../utils/DebugEnv.h"
#include "../../mpi_orchestration/RankExecutionPlan.h" // For Config::fromPlan()
#include "../../../collective/PPActivationContract.h"  // PPActivationContract for forwardPP
#include "fort.hpp"                                    // libfort for TP profiling summary table
#include <algorithm>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#ifdef __linux__
#include <malloc.h> // malloc_trim for deferred host weight release
#endif

namespace llaminar2
{

    // =========================================================================
    // Config Implementation
    // =========================================================================

    bool RankOrchestrator::PPStageConfig::validate() const
    {
        // Layer range must be valid
        if (last_layer <= first_layer)
        {
            LOG_ERROR("PPStageConfig: Invalid layer range [" << first_layer << ", " << last_layer << ")");
            return false;
        }

        // Must have at least one device
        if (stage_devices.empty())
        {
            LOG_ERROR("PPStageConfig: No stage devices specified");
            return false;
        }

        // If TP weights are provided, must match device count
        if (!tp_weights.empty() && tp_weights.size() != stage_devices.size())
        {
            LOG_ERROR("PPStageConfig: TP weights count (" << tp_weights.size()
                                                          << ") doesn't match device count (" << stage_devices.size() << ")");
            return false;
        }

        // If TP weights are provided, must sum to approximately 1.0
        if (!tp_weights.empty())
        {
            float sum = std::accumulate(tp_weights.begin(), tp_weights.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.01f)
            {
                LOG_ERROR("PPStageConfig: TP weights sum to " << sum << ", expected 1.0");
                return false;
            }
        }

        return true;
    }

    RankOrchestrator::ParallelismMode
    RankOrchestrator::Config::detectMode() const
    {
        if (pp_stages.empty())
        {
            // No PP stages - pure TP mode
            return ParallelismMode::TP;
        }

        // Check if any PP stage is a TP domain
        bool has_tp_stages = std::any_of(pp_stages.begin(), pp_stages.end(),
                                         [](const PPStageConfig &stage)
                                         { return stage.isTPDomain(); });

        return has_tp_stages ? ParallelismMode::TP_PP : ParallelismMode::PP;
    }

    std::vector<int> RankOrchestrator::Config::buildLayerBoundaries() const
    {
        std::vector<int> boundaries;
        if (pp_stages.empty())
        {
            return boundaries;
        }

        boundaries.push_back(0);
        for (const auto &stage : pp_stages)
        {
            boundaries.push_back(stage.last_layer);
        }
        return boundaries;
    }

    bool RankOrchestrator::Config::validate() const
    {
        ParallelismMode effective = effectiveMode();

        if (effective == ParallelismMode::TP)
        {
            // TP mode validation
            if (devices.empty())
            {
                LOG_ERROR("RankOrchestrator::Config: No devices specified for TP mode");
                return false;
            }

            // If weights are provided, must match device count
            if (!weights.empty() && weights.size() != devices.size())
            {
                LOG_ERROR("RankOrchestrator::Config: Weights count (" << weights.size()
                                                                      << ") doesn't match device count (" << devices.size() << ")");
                return false;
            }

            // If weights are provided, must sum to approximately 1.0
            if (!weights.empty())
            {
                float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
                if (std::abs(sum - 1.0f) > 0.01f)
                {
                    LOG_ERROR("RankOrchestrator::Config: Weights sum to " << sum << ", expected 1.0");
                    return false;
                }
            }
        }
        else
        {
            // PP or TP_PP mode validation
            if (pp_stages.empty())
            {
                LOG_ERROR("RankOrchestrator::Config: No PP stages specified for PP mode");
                return false;
            }

            // Validate each stage
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (!pp_stages[i].validate())
                {
                    LOG_ERROR("RankOrchestrator::Config: PP stage " << i << " validation failed");
                    return false;
                }
            }

            // Check layer continuity (no gaps)
            int expected_first = 0;
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (pp_stages[i].first_layer != expected_first)
                {
                    LOG_ERROR("RankOrchestrator::Config: PP stage " << i
                                                                    << " first_layer=" << pp_stages[i].first_layer
                                                                    << " but expected " << expected_first << " (gap in layers)");
                    return false;
                }
                expected_first = pp_stages[i].last_layer;
            }

            // First stage should have embedding, last should have LM head
            if (!pp_stages.front().has_embedding)
            {
                LOG_WARN("RankOrchestrator::Config: First PP stage doesn't have embedding flag set");
            }
            if (!pp_stages.back().has_lm_head)
            {
                LOG_WARN("RankOrchestrator::Config: Last PP stage doesn't have lm_head flag set");
            }
        }

        return true;
    }

    std::vector<float> RankOrchestrator::Config::getNormalizedWeights() const
    {
        if (weights.empty() || weights.size() != devices.size())
        {
            // Equal distribution
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        // Normalize to ensure sum is exactly 1.0
        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= 0.0f)
        {
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        std::vector<float> normalized(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            normalized[i] = weights[i] / sum;
        }
        return normalized;
    }

    // =========================================================================
    // Config::fromPlan — Canonical translation from RankExecutionPlan
    // =========================================================================

    RankOrchestrator::Config
    RankOrchestrator::Config::fromPlan(const RankExecutionPlan &plan)
    {
        Config config;

        // Runtime fields from pre-parsed RuntimeConfig
        config.max_seq_len = plan.runtime.max_seq_len;
        config.batch_size = plan.runtime.batch_size;
        config.activation_precision = plan.runtime.activation_precision;
        config.kv_cache_precision = plan.runtime.kv_cache_precision;
        config.prefix_cache = plan.runtime.prefix_cache;
        config.mtp = plan.runtime.mtp;
        config.moe_expert_mode = plan.runtime.moe_expert_mode;
        config.moe_hot_expert_cache = plan.runtime.moe_hot_expert_cache;
        config.moe_rebalance = plan.runtime.moe_rebalance;

        if (plan.usesLocalPP())
        {
            // PP mode: build stage configs from plan boundaries
            config.mode = ParallelismMode::PP;

            const auto &pp_devices = plan.local_pp_devices;
            const auto &boundaries = plan.local_pp_layer_boundaries;
            const auto &stage_tp_info = plan.local_pp_stage_tp_info;

            for (size_t i = 0; i < pp_devices.size(); ++i)
            {
                PPStageConfig stage_cfg;
                stage_cfg.first_layer = boundaries[i];
                stage_cfg.last_layer = boundaries[i + 1]; // exclusive
                stage_cfg.has_embedding = (i == 0);
                stage_cfg.has_lm_head = (i == pp_devices.size() - 1);

                // Use per-stage TP info if available (TP-in-PP composition)
                if (i < stage_tp_info.size() && stage_tp_info[i].devices.size() > 1)
                {
                    stage_cfg.stage_devices = stage_tp_info[i].devices;
                    stage_cfg.tp_weights = stage_tp_info[i].tp_weights;
                    stage_cfg.tp_backend = stage_tp_info[i].tp_backend;
                }
                else
                {
                    stage_cfg.stage_devices = {pp_devices[i]};
                }

                // Cross-vendor detection for host-staged hidden state transfer
                // Compare primary devices of adjacent stages
                auto primaryDeviceType = [&](size_t idx) -> DeviceType
                {
                    if (idx < stage_tp_info.size() && !stage_tp_info[idx].devices.empty())
                        return stage_tp_info[idx].devices[0].device_type;
                    if (idx < pp_devices.size())
                        return pp_devices[idx].device_type;
                    return DeviceType::CPU;
                };
                if (i + 1 < pp_devices.size() &&
                    primaryDeviceType(i) != primaryDeviceType(i + 1))
                {
                    // Cross-vendor PP stage boundary detected
                }

                config.pp_stages.push_back(std::move(stage_cfg));
            }
        }
        else
        {
            // TP mode: copy devices, weights, backend
            config.devices = plan.local_tp_devices;
            if (!plan.local_tp_weights.empty())
            {
                config.weights = plan.local_tp_weights;
            }
            config.backend = plan.local_tp_backend;
        }

        return config;
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    std::unique_ptr<RankOrchestrator> RankOrchestrator::createForTest(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
    {
        // Use the private constructor
        return std::unique_ptr<RankOrchestrator>(
            new RankOrchestrator(
                std::move(model_ctx),
                std::move(device_runners),
                std::move(tp_ctx),
                config));
    }

    std::unique_ptr<RankOrchestrator> RankOrchestrator::createForTestWithPipelineStages(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> pp_stage_runners,
        const Config &config)
    {
        auto orchestrator = createForTest(
            std::move(model_ctx),
            {},
            nullptr,
            config);
        orchestrator->mode_ = ParallelismMode::PP;
        orchestrator->pp_stage_runners_ = std::move(pp_stage_runners);
        return orchestrator;
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    RankOrchestrator::RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        const Config &config,
        std::unique_ptr<ILocalTPContext> tp_ctx)
        : model_ctx_(std::move(model_ctx)), config_(config)
    {
        if (!config_.validate())
        {
            throw std::invalid_argument("Invalid RankOrchestrator configuration");
        }

        // Initialize stage sharding map from model architecture
        stage_sharding_map_ = SchemaFactoryRegistry::getStageShardingConfig(model_ctx_->architecture());

        if (tp_ctx)
        {
            // Pre-existing TP context provided — force TP mode
            tp_ctx_ = std::move(tp_ctx);
            mode_ = ParallelismMode::TP;

            if (tp_ctx_->degree() < 2)
            {
                LOG_WARN("RankOrchestrator: TP degree is " << tp_ctx_->degree()
                                                           << ", multi-device orchestration may not be beneficial");
            }

            LOG_DEBUG("RankOrchestrator: Creating with pre-existing TP context, "
                      << tp_ctx_->degree() << " devices");

            initializeDeviceRunners();

            LOG_DEBUG("RankOrchestrator: Initialized with " << device_runners_.size() << " device runners");
        }
        else
        {
            // Auto-detect mode from config
            mode_ = config_.effectiveMode();

            LOG_DEBUG("RankOrchestrator: Creating with mode="
                      << (mode_ == ParallelismMode::TP ? "TP" : mode_ == ParallelismMode::PP ? "PP"
                                                                                             : "TP_PP")
                      << ", backend=" << static_cast<int>(config_.backend));

            if (mode_ == ParallelismMode::TP)
            {
                // Pure TP mode - create LOCAL TP context from config
                tp_ctx_ = createLocalTPContext(
                    config_.devices,
                    config_.getNormalizedWeights(),
                    config_.backend);

                if (!tp_ctx_)
                {
                    throw std::runtime_error("Failed to create LOCAL TP context");
                }

                if (tp_ctx_->degree() < 2)
                {
                    LOG_WARN("RankOrchestrator: TP degree is " << tp_ctx_->degree()
                                                               << ", multi-device orchestration may not be beneficial");
                }

                initializeDeviceRunners();

                LOG_DEBUG("RankOrchestrator: Initialized TP mode with " << device_runners_.size() << " device runners");
            }
            else
            {
                // PP or TP_PP mode - create PP context and stage runners
                initializePPDeviceRunners();
                initializePPContext();

                LOG_DEBUG("RankOrchestrator: Initialized PP mode with " << config_.pp_stages.size() << " stages");
            }
        }
    }

    // Private constructor for createForTest
    RankOrchestrator::RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)),
          tp_ctx_(std::move(tp_ctx)),
          mode_(ParallelismMode::TP), // Test factory currently only supports TP mode
          device_runners_(std::move(device_runners)),
          config_(config)
    {
        // Initialize stage sharding map from model architecture (if registered)
        const auto arch = model_ctx_->architecture();
        if (SchemaFactoryRegistry::isSupported(arch))
        {
            stage_sharding_map_ = SchemaFactoryRegistry::getStageShardingConfig(arch);

            // GQA-aware override: replicate K/V snapshot modes when n_kv_heads < tp_degree
            if (tp_ctx_ && model_ctx_->headCountKV() < tp_ctx_->degree())
            {
                stage_sharding_map_["K_PROJECTION"] = SnapshotShardingMode::REPLICATED;
                stage_sharding_map_["V_PROJECTION"] = SnapshotShardingMode::REPLICATED;
                stage_sharding_map_["K_ROPE"] = SnapshotShardingMode::REPLICATED;
            }
        }

        LOG_DEBUG("RankOrchestrator: Created via createForTest with "
                  << device_runners_.size() << " injected device runners");
    }

    RankOrchestrator::~RankOrchestrator() = default;

    // Move operations
    RankOrchestrator::RankOrchestrator(RankOrchestrator &&) noexcept = default;
    RankOrchestrator &RankOrchestrator::operator=(RankOrchestrator &&) noexcept = default;

    // =========================================================================
    // Private Methods
    // =========================================================================

    void RankOrchestrator::initializeDeviceRunners()
    {
        if (!tp_ctx_)
        {
            throw std::runtime_error("Cannot initialize device runners: tp_ctx_ is null");
        }

        const auto &devices = tp_ctx_->devices();
        device_runners_.reserve(devices.size());

        LOG_DEBUG("RankOrchestrator: Initializing " << devices.size() << " device runners");

        // =====================================================================
        // BUILD WEIGHTMANAGERCONFIG FOR LOCAL TP WEIGHT SHARDING
        // =====================================================================
        // Configure WeightManager with TP config, model dimensions, and sharding
        // in a single configure() call. This replaces the previous multi-setter chain.
        // =====================================================================
        {
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr && tp_ctx_)
            {
                // Get model dimensions
                int n_heads = model_ctx_->headCount();
                int n_kv_heads = model_ctx_->headCountKV();
                int d_ff = model_ctx_->feedForwardLength();
                int vocab_size = model_ctx_->vocabSize();

                // feedForwardLength must be available from model metadata for correct TP sharding
                if (d_ff <= 0)
                {
                    throw std::runtime_error(
                        "RankOrchestrator: feedForwardLength() unavailable from model context. "
                        "This value is required for correct tensor parallel weight sharding. "
                        "Ensure the GGUF model contains the feed_forward_length metadata field.");
                }

                auto tp_config = std::make_shared<TensorParallelConfig>(
                    TensorParallelConfig::fromLocalTPContext(
                        *tp_ctx_, n_heads, n_kv_heads, d_ff, vocab_size));

                // Build unified WeightManagerConfig
                WeightManagerConfig wm_config;
                wm_config.tp_config = tp_config;
                wm_config.sharding = SchemaFactoryRegistry::getWeightShardingConfig(model_ctx_->architecture());

                // Model head dimensions for FusedQKV sub-block slicing
                const int embed_len = model_ctx_->embeddingLength();
                const int head_dim = (n_heads > 0) ? (embed_len / n_heads) : 0;
                if (n_heads > 0 && head_dim > 0)
                {
                    wm_config.dimensions.n_heads = n_heads;
                    wm_config.dimensions.n_kv_heads = n_kv_heads;
                    wm_config.dimensions.head_dim = head_dim;
                }

                // GDN dimensions from GGUF metadata for asymmetric FusedQKV
                auto *concrete_ctx = dynamic_cast<ModelContext *>(model_ctx_.get());
                if (concrete_ctx)
                {
                    const std::string arch_prefix = model_ctx_->architecture();
                    ModelLoader &loader = concrete_ctx->concreteLoader();
                    const int gdn_group_count = loader.getInt(arch_prefix + ".ssm.group_count", 0);
                    const int gdn_time_step_rank = loader.getInt(arch_prefix + ".ssm.time_step_rank", 0);
                    const int gdn_state_size = loader.getInt(arch_prefix + ".ssm.state_size", 0);

                    if (gdn_group_count > 0 && gdn_time_step_rank > 0 && gdn_state_size > 0)
                    {
                        wm_config.dimensions.gdn_n_k_heads = gdn_group_count;
                        wm_config.dimensions.gdn_n_v_heads = gdn_time_step_rank;
                        wm_config.dimensions.gdn_d_state = gdn_state_size;
                        LOG_DEBUG("RankOrchestrator: GDN dimensions for FusedQKV slicing"
                                  << " (group_count=" << gdn_group_count
                                  << " time_step_rank=" << gdn_time_step_rank
                                  << " state_size=" << gdn_state_size << ")");
                    }
                }

                // Apply all configuration in a single call
                weight_mgr->configure(wm_config);

                LOG_DEBUG("RankOrchestrator: Configured WeightManager for LOCAL TP ("
                          << tp_ctx_->degree() << " devices, "
                          << "heads=" << n_heads << ", kv_heads=" << n_kv_heads
                          << ", d_ff=" << d_ff << ", vocab=" << vocab_size << ")");

                // Print per-device assignments for debugging TP parity issues
                for (int dev_idx = 0; dev_idx < tp_ctx_->degree(); ++dev_idx)
                {
                    const auto &addr = tp_ctx_->devices()[dev_idx];
                    DeviceId dev_id = addr.toLocalDeviceId();
                    try
                    {
                        const auto &assignment = tp_config->forDevice(dev_id);
                        LOG_DEBUG("RankOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") assignment:"
                                                              << " head_start=" << assignment.head_start
                                                              << " head_count=" << assignment.head_count
                                                              << " kv_head_start=" << assignment.kv_head_start
                                                              << " kv_head_count=" << assignment.kv_head_count
                                                              << " d_ff_start=" << assignment.d_ff_start
                                                              << " d_ff_count=" << assignment.d_ff_count);
                    }
                    catch (const std::out_of_range &e)
                    {
                        LOG_WARN("RankOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") NOT in TensorParallelConfig!");
                    }
                }

                // GQA-aware stage sharding override: when n_kv_heads < tp_degree,
                // K/V are replicated (not column-parallel). Override the snapshot
                // sharding map so parity tests compare K/V outputs correctly.
                if (n_kv_heads < tp_ctx_->degree())
                {
                    stage_sharding_map_["K_PROJECTION"] = SnapshotShardingMode::REPLICATED;
                    stage_sharding_map_["V_PROJECTION"] = SnapshotShardingMode::REPLICATED;
                    stage_sharding_map_["K_ROPE"] = SnapshotShardingMode::REPLICATED;
                    LOG_DEBUG("RankOrchestrator: GQA override: K/V stages set to REPLICATED "
                              "(n_kv_heads="
                              << n_kv_heads << " < tp_degree=" << tp_ctx_->degree() << ")");
                }
            }
        }

        // =====================================================================
        // PRE-RESERVE COLLECTIVE TEMP BUFFER
        // =====================================================================
        // Pre-allocate temp buffer for allreduce operations based on model dimensions
        // and activation precision. This avoids allocation in the hot path.
        // Buffer uses grow-only semantics (never shrinks during inference).
        // =====================================================================
        {
            size_t hidden_size = model_ctx_->embeddingLength();
            size_t max_elements = config_.max_seq_len * hidden_size;

            // Calculate buffer bytes based on activation precision (handles block quantization alignment)
            size_t buffer_bytes = activationPrecisionBufferBytes(max_elements, config_.activation_precision);

            // Add 10% margin for safety
            size_t buffer_with_margin = static_cast<size_t>(buffer_bytes * 1.1);

            if (tp_ctx_->reserveTempBufferBytes(buffer_with_margin))
            {
                LOG_DEBUG("RankOrchestrator: Reserved collective temp buffer: "
                          << buffer_with_margin << " bytes ("
                          << "max_seq_len=" << config_.max_seq_len
                          << ", hidden_size=" << hidden_size
                          << ", precision=" << activationPrecisionToString(config_.activation_precision) << ")");
            }
            else
            {
                LOG_WARN("RankOrchestrator: Failed to reserve collective temp buffer");
            }
        }

        // =====================================================================
        // PRE-LOAD WEIGHTS FOR ALL DEVICES
        // =====================================================================
        // This is critical for multi-device operation:
        // - Creates device-specific clones of shared tensors (embedding, norms)
        // - Uploads each clone to its target device BEFORE parallel execution
        // - Avoids race condition where multiple devices try to upload same tensor
        //
        // The WeightManager now handles all device-aware weight management centrally.
        // =====================================================================
        {
            // Collect device IDs for preloading
            std::vector<DeviceId> device_ids;
            device_ids.reserve(devices.size());
            for (const auto &device_addr : devices)
            {
                device_ids.push_back(device_addr.toLocalDeviceId());
            }

            // Finalize weights for all devices: clone, upload, and prepare GEMM
            // weights. Host data release is deferred until after device runners are
            // created because graph construction resolves prepared weight bindings.
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr)
            {
                if (auto concrete_weight_mgr = std::dynamic_pointer_cast<WeightManager>(weight_mgr))
                {
                    if (config_.prepared_weight_store)
                    {
                        concrete_weight_mgr->setPreparedWeightStore(config_.prepared_weight_store);
                    }
                    else
                    {
                        config_.prepared_weight_store = std::make_shared<PreparedWeightStore>(
                            ModelContextId{reinterpret_cast<uint64_t>(model_ctx_.get())});
                        concrete_weight_mgr->setPreparedWeightStore(config_.prepared_weight_store);
                    }
                }
                if (config_.nested_pp_stage_config.has_value())
                {
                    LOG_DEBUG("RankOrchestrator: Preloading weights for "
                              << device_ids.size() << " nested TP-in-PP devices"
                              << " (binding-driven preparation deferred to runner materialization)");
                    if (!weight_mgr->preloadForDevices(device_ids))
                    {
                        LOG_WARN("RankOrchestrator: Weight preload failed; runner materialization may fail");
                    }
                }
                else
                {
                    LOG_DEBUG("RankOrchestrator: Finalizing weights for "
                              << device_ids.size() << " devices"
                              << " (release_host_data=false, deferred until after graph build)");
                    if (!weight_mgr->finalizeForDevices(device_ids, /*release_host_data=*/false))
                    {
                        LOG_WARN("RankOrchestrator: Weight finalization failed; prepared kernels may be unavailable");
                    }
                }
            }
        }

        // =====================================================================
        // Create device runners in parallel
        // =====================================================================
        // After preloadForDevices(), each runner creation is independent:
        //   - Different device_id, different device_idx
        //   - WeightManager cache hits are mutex-protected
        //   - Graph construction reads immutable model config
        // Parallelizing this cuts ~50% off the MDO init time for 2+ devices.
        // =====================================================================

        struct RunnerResult
        {
            std::unique_ptr<IInferenceRunner> runner;
            int device_idx;
            std::string error;
        };

        const int num_devices = static_cast<int>(devices.size());
        std::vector<std::future<RunnerResult>> futures;
        futures.reserve(num_devices);

        for (int device_idx = 0; device_idx < num_devices; ++device_idx)
        {
            const auto &device_addr = devices[device_idx];
            DeviceId device_id = device_addr.toLocalDeviceId();

            futures.push_back(std::async(std::launch::async,
                                         [this, device_idx, device_id]() -> RunnerResult
                                         {
                                             RunnerResult result;
                                             result.device_idx = device_idx;
                                             try
                                             {
                                                 LOG_DEBUG("RankOrchestrator: Creating runner for device " << device_idx
                                                                                                           << " (" << device_id.toString() << ")");

                                                 // Build InferenceRunnerConfig for LOCAL TP
                                                 InferenceRunnerConfig runner_config;
                                                 runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
                                                 runner_config.batch_size = config_.batch_size;
                                                 runner_config.activation_precision = config_.activation_precision;
                                                 runner_config.kv_cache_scale_k = config_.kv_cache_scale_k;
                                                 runner_config.kv_cache_scale_v = config_.kv_cache_scale_v;
                                                 runner_config.kv_cache_precision = config_.kv_cache_precision;
                                                 runner_config.prefix_cache = config_.prefix_cache;
                                                 runner_config.mtp = config_.mtp;
                                                 runner_config.moe_expert_mode = config_.moe_expert_mode;
                                                 runner_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
                                                 runner_config.moe_rebalance = config_.moe_rebalance;
                                                 runner_config.use_mapped_memory = config_.use_mapped_memory;
                                                 runner_config.prepared_weight_store = config_.prepared_weight_store;
                                                 runner_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
                                                 runner_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
                                                 runner_config.cancellation_requested = [this]()
                                                 {
                                                     return tp_ctx_ && tp_ctx_->isAbortRequested();
                                                 };
                                                 runner_config.stage_failure_callback = [this, device_idx](const std::string &stage_name, const std::string &reason)
                                                 {
                                                     LOG_WARN("RankOrchestrator: device " << device_idx
                                                                                          << " stage failure triggers TP abort"
                                                                                          << " stage='" << stage_name << "' reason='" << reason << "'");
                                                     if (tp_ctx_)
                                                         tp_ctx_->requestAbort();
                                                 };

                                                 // Set TP parameters (LOCAL TP context here)
                                                 runner_config.tp_ctx = tp_ctx_.get();
                                                 runner_config.tp_device_index = device_idx;

                                                 // Pass PP stage config for nested TP-in-PP
                                                 if (config_.nested_pp_stage_config.has_value())
                                                 {
                                                     runner_config.pp_stage_config = config_.nested_pp_stage_config;
                                                 }

                                                 // Create the inference runner
                                                 result.runner = createTestableInferenceRunner(model_ctx_, device_id, runner_config);
                                                 if (!result.runner)
                                                 {
                                                     result.error = "Failed to create inference runner for device " +
                                                                    std::to_string(device_idx);
                                                 }
                                             }
                                             catch (const std::exception &e)
                                             {
                                                 result.error = std::string("Exception creating runner for device ") +
                                                                std::to_string(device_idx) + ": " + e.what();
                                             }
                                             return result;
                                         }));
        }

        // Collect results in device order
        for (auto &fut : futures)
        {
            auto result = fut.get();
            if (!result.error.empty())
            {
                throw std::runtime_error(result.error);
            }

            // Cast to DeviceGraphOrchestrator for setPPStageConfig (construction-time only)
            if (config_.nested_pp_stage_config.has_value())
            {
                auto *device_orchestrator = dynamic_cast<DeviceGraphOrchestrator *>(result.runner.get());
                if (device_orchestrator)
                {
                    device_orchestrator->setHostResidentReleaseEnabled(false);
                    device_orchestrator->setPPStageConfig(config_.nested_pp_stage_config.value());
                    LOG_DEBUG("RankOrchestrator: Set PP stage config on device " << result.device_idx
                                                                                 << " (layers " << config_.nested_pp_stage_config->first_layer
                                                                                 << "-" << config_.nested_pp_stage_config->last_layer
                                                                                 << " has_lm_head=" << config_.nested_pp_stage_config->has_lm_head << ")");
                }
                else if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(result.runner.get()))
                {
                    dgo->setHostResidentReleaseEnabled(false);
                }
            }
            else if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(result.runner.get()))
            {
                dgo->setHostResidentReleaseEnabled(false);
            }

            // Store as IInferenceRunner (decoupled from concrete DGO type)
            device_runners_.push_back(std::move(result.runner));

            LOG_DEBUG("RankOrchestrator: Successfully created runner for device " << result.device_idx);
        }

        // =====================================================================
        // SYNCHRONOUS HOST WEIGHT RELEASE (Phase 9 final)
        // =====================================================================
        // All expert GEMM preparation is resolved before host release: GPU paths
        // consume the unified ExpertGemmRegistry and CPU paths prepare inline.
        // There is no lazy graph building path.
        // Dynamic MoE rebalancing uses serialized transfer blobs, not raw host data.
        //
        // Mark graph_materialization_complete and release host data immediately.
        // =====================================================================
        {
            const bool is_nested_tp_in_pp = config_.nested_pp_stage_config.has_value();
            if (!is_nested_tp_in_pp)
            {
                auto weight_mgr = model_ctx_->weightManager();
                if (weight_mgr)
                {
                    weight_mgr->markGraphMaterializationComplete();

                    size_t released = weight_mgr->releaseAllHostWeightData();
                    if (released > 0)
                    {
                        LOG_DEBUG("RankOrchestrator: Host weight release after graph materialization: "
                                  << released << " tensors released");
#ifdef __linux__
                        ::malloc_trim(0);
#endif
                    }
                }
            }
            else
            {
                LOG_DEBUG("RankOrchestrator: Retaining host weight data "
                          "(nested TP-in-PP; outer caller will release)");
            }
        }

        // Create logits gatherer for combined logits buffer management
        if (model_ctx_ && device_runners_.size() > 0)
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                size_t max_tokens = static_cast<size_t>(config_.batch_size) *
                                    static_cast<size_t>(config_.max_seq_len);
                logits_gatherer_ = std::make_unique<LogitsGatherer>(vocab, max_tokens);

                // Pin the logits buffer for faster D2H DMA
                if (device_runners_.size() > 1)
                {
                    DeviceId primary_dev = device_runners_[0]->primaryDeviceId();
                    if (primary_dev.is_gpu())
                    {
                        logits_gatherer_->pinForDevice(primary_dev);
                    }
                }
            }
        }

        // =====================================================================
        // REGISTER COMPUTE STREAMS WITH COLLECTIVE BACKEND
        // =====================================================================
        // Enables event-based pre-synchronization in RCCL/NCCL coordinators instead
        // of hipDeviceSynchronize/cudaDeviceSynchronize. Each device's compute stream
        // is passed so the coordinator can do hipEventRecord(compute_stream) +
        // hipStreamWaitEvent(rccl_stream) before collectives — zero host stall.
        // =====================================================================
        if (tp_ctx_ && tp_ctx_->degree() > 1)
        {
            const auto &devices = tp_ctx_->devices();

            auto &pool = GPUDeviceContextPool::instance();
            std::vector<void *> compute_streams;
            compute_streams.reserve(devices.size());

            for (const auto &dev : devices)
            {
                if (dev.device_type == DeviceType::ROCm || dev.device_type == DeviceType::CUDA)
                {
                    compute_streams.push_back(
                        pool.getContext(DeviceId(dev.device_type, dev.device_ordinal)).defaultStream());
                }
                else
                {
                    LOG_WARN("RankOrchestrator: Skipping compute stream registration for non-GPU device "
                             << dev.toString());
                    compute_streams.clear();
                    break;
                }
            }

            if (!compute_streams.empty())
            {
                tp_ctx_->setComputeStreams(compute_streams);
                LOG_DEBUG("RankOrchestrator: Registered " << compute_streams.size()
                                                          << " compute streams for event-based collective sync");
            }
        }
    }

    // =========================================================================
    // PP Mode Initialization
    // =========================================================================

    void RankOrchestrator::initializePPDeviceRunners()
    {
        if (config_.pp_stages.empty())
        {
            throw std::runtime_error("Cannot initialize PP device runners: no PP stages configured");
        }

        LOG_DEBUG("RankOrchestrator: Initializing " << config_.pp_stages.size() << " PP stage runners");

        // Cast to concrete ModelContext — needed by createPPStageRunner for concreteLoader() access
        auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx_);
        if (!concrete_model_ctx)
        {
            throw std::runtime_error("RankOrchestrator: model_ctx_ must be a concrete ModelContext for PP mode");
        }

        const size_t num_stages = config_.pp_stages.size();

        // =========================================================================
        // Cross-Vendor PP Detection
        // =========================================================================
        // Check if any PP transfer is cross-vendor (ROCm→CUDA or CUDA→ROCm).
        // If so, the source stage's hidden state tensor needs host-staged allocation
        // to enable cross-vendor transfers via HOST backend.
        // =========================================================================
        auto isCrossVendorTransfer = [](const PPStageConfig &src, const PPStageConfig &dst) -> bool
        {
            if (src.stage_devices.empty() || dst.stage_devices.empty())
            {
                return false;
            }
            DeviceId src_dev = src.stage_devices[0].toLocalDeviceId();
            DeviceId dst_dev = dst.stage_devices[0].toLocalDeviceId();

            bool src_cuda = src_dev.is_cuda();
            bool src_rocm = src_dev.is_rocm();
            bool dst_cuda = dst_dev.is_cuda();
            bool dst_rocm = dst_dev.is_rocm();

            return (src_cuda && dst_rocm) || (src_rocm && dst_cuda);
        };

        // Check for cross-vendor transfers for logging purposes
        for (size_t i = 0; i + 1 < num_stages; ++i)
        {
            if (isCrossVendorTransfer(config_.pp_stages[i], config_.pp_stages[i + 1]))
            {
                LOG_DEBUG("RankOrchestrator: PP stage " << i
                                                        << " outputs to cross-vendor stage " << (i + 1)
                                                        << " - will use host-staged transfer");
            }
        }

        pp_stage_runners_.reserve(num_stages);

        for (size_t stage_idx = 0; stage_idx < num_stages; ++stage_idx)
        {
            const auto &stage_config = config_.pp_stages[stage_idx];

            LOG_DEBUG("RankOrchestrator: Creating PP stage " << stage_idx
                                                             << " [layers " << stage_config.first_layer
                                                             << "-" << stage_config.last_layer << ")"
                                                             << " has_embedding=" << stage_config.has_embedding
                                                             << " has_lm_head=" << stage_config.has_lm_head);

            // Validate stage has at least one device
            if (stage_config.stage_devices.empty())
            {
                throw std::runtime_error("PP stage " + std::to_string(stage_idx) + " has no devices configured");
            }

            // Get primary device for this stage
            DeviceId primary_device = stage_config.stage_devices[0].toLocalDeviceId();

            // =====================================================================
            // Build InferenceRunnerConfig for this stage
            // =====================================================================
            InferenceRunnerConfig runner_config;
            runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
            runner_config.batch_size = config_.batch_size;
            runner_config.activation_precision = config_.activation_precision;
            runner_config.kv_cache_scale_k = config_.kv_cache_scale_k;
            runner_config.kv_cache_scale_v = config_.kv_cache_scale_v;
            runner_config.kv_cache_precision = config_.kv_cache_precision;
            runner_config.prefix_cache = config_.prefix_cache;
            runner_config.mtp = config_.mtp;
            runner_config.moe_expert_mode = config_.moe_expert_mode;
            runner_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
            runner_config.moe_rebalance = config_.moe_rebalance;
            runner_config.use_mapped_memory = config_.use_mapped_memory;
            runner_config.prepared_weight_store = config_.prepared_weight_store;
            runner_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
            runner_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
            // =====================================================================
            // Build FactoryPPStageConfig for the createPPStageRunner factory
            // =====================================================================
            FactoryPPStageConfig factory_pp_config;
            factory_pp_config.first_layer = stage_config.first_layer;
            factory_pp_config.last_layer = stage_config.last_layer;
            factory_pp_config.has_embedding = stage_config.has_embedding;
            factory_pp_config.has_lm_head = stage_config.has_lm_head;

            // =====================================================================
            // Handle single-device vs TP-domain stages
            // =====================================================================
            if (stage_config.isTPDomain())
            {
                // =====================================================================
                // TP Domain Stage: Create nested RankOrchestrator in TP mode
                // =====================================================================
                LOG_DEBUG("RankOrchestrator: PP stage " << stage_idx
                                                        << " is a TP domain with " << stage_config.stage_devices.size() << " devices");

                // Build TP configuration for the nested orchestrator
                Config nested_config;
                nested_config.mode = ParallelismMode::TP;
                nested_config.devices = stage_config.stage_devices;
                nested_config.weights = stage_config.tp_weights;
                nested_config.backend = stage_config.tp_backend;
                nested_config.max_seq_len = config_.max_seq_len;
                nested_config.batch_size = config_.batch_size;
                nested_config.activation_precision = config_.activation_precision;
                nested_config.kv_cache_scale_k = config_.kv_cache_scale_k;
                nested_config.kv_cache_scale_v = config_.kv_cache_scale_v;
                nested_config.kv_cache_precision = config_.kv_cache_precision;
                nested_config.prefix_cache = config_.prefix_cache;
                nested_config.mtp = config_.mtp;
                nested_config.moe_expert_mode = config_.moe_expert_mode;
                nested_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
                nested_config.moe_rebalance = config_.moe_rebalance;
                nested_config.use_mapped_memory = config_.use_mapped_memory;
                nested_config.prepared_weight_store = config_.prepared_weight_store;
                nested_config.moe_expert_parallel_plan = config_.moe_expert_parallel_plan;
                nested_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
                // CRITICAL: Pass PP stage config to nested TP MDO so its DeviceGraphOrchestrators
                // build partial graphs instead of full graphs. Without this, the TP devices would
                // build LM_HEAD stages even though this PP stage doesn't own LM_HEAD.
                nested_config.nested_pp_stage_config = factory_pp_config;

                // Create the nested RankOrchestrator
                // Note: model_ctx_ is the shared ModelContext — the nested MDO's
                // createPPStageRunner uses the shared WeightManager from ModelContext
                auto nested_mdo = std::make_unique<RankOrchestrator>(concrete_model_ctx, nested_config);

                if (!nested_mdo)
                {
                    throw std::runtime_error("Failed to create nested RankOrchestrator for PP stage " +
                                             std::to_string(stage_idx));
                }

                pp_stage_runners_.push_back(std::move(nested_mdo));

                LOG_DEBUG("RankOrchestrator: Created TP domain PP stage " << stage_idx
                                                                          << " with " << stage_config.stage_devices.size() << " devices"
                                                                          << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
            else
            {
                // =====================================================================
                // Single Device Stage: Use existing factory path
                // =====================================================================
                auto runner = createPPStageRunner(concrete_model_ctx, primary_device, factory_pp_config, runner_config);

                if (!runner)
                {
                    throw std::runtime_error("Failed to create PP stage runner for stage " +
                                             std::to_string(stage_idx) + " on device " +
                                             primary_device.to_string());
                }

                if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                    dgo->setHostResidentReleaseEnabled(false);

                pp_stage_runners_.push_back(std::move(runner));

                LOG_DEBUG("RankOrchestrator: Created PP stage " << stage_idx
                                                                << " runner on device " << primary_device.to_string()
                                                                << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
        }

        LOG_DEBUG("RankOrchestrator: Successfully initialized " << pp_stage_runners_.size() << " PP stage runners");

        // =====================================================================
        // Release host weight copies now that all PP stages are prepared.
        // With single-WM architecture, host copies are retained across stages
        // and only released after ALL stage preparation is complete.
        //
        // CPU safety: If any PP stage runs on CPU, retain host weight data —
        // CPU stages read weights directly from host memory.
        // =====================================================================
        bool has_cpu_stage = false;
        for (const auto &stage_config : config_.pp_stages)
        {
            if (!stage_config.stage_devices.empty() &&
                stage_config.stage_devices[0].toLocalDeviceId().is_cpu())
            {
                has_cpu_stage = true;
                break;
            }
        }

        if (has_cpu_stage)
        {
            LOG_DEBUG("RankOrchestrator: Retaining host weight data (CPU PP stages present)");
        }
        else if (auto *concrete_wm = dynamic_cast<WeightManager *>(model_ctx_->weightManager().get()))
        {
            size_t released = concrete_wm->releaseAllHostWeightData();
            LOG_DEBUG("RankOrchestrator: Released " << released << " host weight copies after PP preparation");
        }

        // Build PPActivationContract describing inter-stage data transfers
        if (config_.pp_stages.size() > 1)
        {
            pp_activation_contract_ = std::make_unique<PPActivationContract>();

            const size_t embedding_dim = model_ctx_->embeddingLength();
            const size_t max_seq_len = model_ctx_->contextLength();

            for (size_t i = 0; i + 1 < config_.pp_stages.size(); ++i)
            {
                const auto &src = config_.pp_stages[i];
                const auto &dst = config_.pp_stages[i + 1];

                PPStageTransferContract xfer;
                xfer.source_stage = static_cast<int>(i);
                xfer.target_stage = static_cast<int>(i + 1);
                xfer.source_device = src.stage_devices[0].toLocalDeviceId();
                xfer.target_device = dst.stage_devices[0].toLocalDeviceId();
                xfer.embedding_dim = embedding_dim;
                xfer.dtype = ActivationDType::FP32;
                xfer.max_seq_len = max_seq_len;

                pp_activation_contract_->addTransfer(std::move(xfer));
            }

            auto validation_err = pp_activation_contract_->validate();
            if (!validation_err.empty())
            {
                LOG_ERROR("RankOrchestrator: PPActivationContract validation failed: " << validation_err);
                pp_activation_contract_.reset();
            }
            else
            {
                LOG_DEBUG("RankOrchestrator: PPActivationContract built with "
                          << pp_activation_contract_->numTransfers() << " transfers"
                          << " (embedding=" << embedding_dim << ", max_seq=" << max_seq_len << ")");
            }
        }

        // Note: PP context initialization is done by the caller (constructor)
        // after this method returns, to avoid double-initialization
    }

    void RankOrchestrator::initializePPContext()
    {
        if (pp_stage_runners_.empty())
        {
            // PP stage runners not initialized - this is expected for Phase 1
            LOG_DEBUG("RankOrchestrator::initializePPContext: No PP stage runners yet");
            return;
        }

        // Build LocalPPConfig for simple single-device stages
        // TODO: Phase 5 - use HierarchicalPPConfig for TP domain stages
        LocalPPConfig pp_config;
        pp_config.stage_devices.reserve(config_.pp_stages.size());
        pp_config.layer_boundaries.reserve(config_.pp_stages.size() + 1);

        // Add first boundary (start of first stage)
        pp_config.layer_boundaries.push_back(config_.pp_stages[0].first_layer);

        for (size_t i = 0; i < config_.pp_stages.size(); ++i)
        {
            const auto &stage_config = config_.pp_stages[i];
            // Use first device in stage_devices for single-device stages
            if (stage_config.stage_devices.empty())
            {
                throw std::runtime_error("PP stage " + std::to_string(i) + " has no devices");
            }
            pp_config.stage_devices.push_back(stage_config.stage_devices[0]);
            // Each boundary is the exclusive end = last_layer
            pp_config.layer_boundaries.push_back(stage_config.last_layer);
        }

        // Validate config
        if (!pp_config.isValid())
        {
            throw std::runtime_error("Generated LocalPPConfig is invalid");
        }

        LOG_DEBUG("RankOrchestrator::initializePPContext: Creating LocalPPContext with "
                  << pp_config.numStages() << " stages");

        // Create the LocalPPContext using the factory function
        pp_ctx_ = createLocalPPContext(pp_config);

        if (!pp_ctx_)
        {
            throw std::runtime_error("Failed to create LocalPPContext");
        }

        LOG_DEBUG("RankOrchestrator: Initialized LocalPPContext with " << pp_config.numStages() << " stages");
    }

    void RankOrchestrator::aggregateStats() const
    {
        if (!stats_dirty_ || device_runners_.empty())
        {
            return;
        }

        // Reset or create aggregated stats
        if (!aggregated_stats_)
        {
            aggregated_stats_ = std::make_unique<GraphExecutorStats>();
        }
        aggregated_stats_->reset();

        // Aggregate from all device runners
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                const auto *stats = runner->executorStats();
                if (stats)
                {
                    aggregated_stats_->total_stages_executed += stats->total_stages_executed;
                    aggregated_stats_->total_flops += stats->total_flops;
                    aggregated_stats_->total_time_ms += stats->total_time_ms;
                    aggregated_stats_->total_execute_ms += stats->total_execute_ms;
                    aggregated_stats_->total_collective_ms += stats->total_collective_ms;
                    aggregated_stats_->total_collective_calls += stats->total_collective_calls;

                    // Merge stage times
                    for (const auto &[stage_name, time_ms] : stats->stage_times_ms)
                    {
                        aggregated_stats_->stage_times_ms[stage_name] += time_ms;
                    }

                    // Merge stage type execution times and counts
                    for (const auto &[type_name, time_ms] : stats->stage_type_execute_ms)
                    {
                        aggregated_stats_->stage_type_execute_ms[type_name] += time_ms;
                    }
                    for (const auto &[type_name, count] : stats->stage_type_counts)
                    {
                        aggregated_stats_->stage_type_counts[type_name] += count;
                    }

                    // Merge overhead breakdown
                    aggregated_stats_->overhead += stats->overhead;

                    // Merge phase-split stats (prefill / decode)
                    auto mergePhase = [](PhaseStats &dst, const PhaseStats &src)
                    {
                        dst.total_execute_ms += src.total_execute_ms;
                        dst.total_stages_executed += src.total_stages_executed;
                        dst.total_collective_ms += src.total_collective_ms;
                        dst.total_collective_calls += src.total_collective_calls;
                        for (const auto &[type_name, time_ms] : src.stage_type_execute_ms)
                            dst.stage_type_execute_ms[type_name] += time_ms;
                        for (const auto &[type_name, cnt] : src.stage_type_counts)
                            dst.stage_type_counts[type_name] += cnt;
                    };
                    mergePhase(aggregated_stats_->prefill, stats->prefill);
                    mergePhase(aggregated_stats_->decode, stats->decode);
                }
            }
        }

        // Average the times (since devices run in parallel)
        if (!device_runners_.empty())
        {
            size_t count = device_runners_.size();
            double dcount = static_cast<double>(count);
            aggregated_stats_->total_time_ms /= dcount;
            aggregated_stats_->total_execute_ms /= dcount;
            aggregated_stats_->total_collective_ms /= dcount;
            aggregated_stats_->total_collective_calls /= count;
            for (auto &[stage_name, time_ms] : aggregated_stats_->stage_times_ms)
            {
                time_ms /= dcount;
            }
            for (auto &[type_name, time_ms] : aggregated_stats_->stage_type_execute_ms)
            {
                time_ms /= dcount;
            }
            for (auto &[type_name, cnt] : aggregated_stats_->stage_type_counts)
            {
                cnt /= count;
            }

            // Average overhead (each device incurs its own overhead in parallel)
            aggregated_stats_->overhead.input_cohere_ms /= dcount;
            aggregated_stats_->overhead.weight_cohere_ms /= dcount;
            aggregated_stats_->overhead.output_alloc_ms /= dcount;
            aggregated_stats_->overhead.mark_dirty_ms /= dcount;
            aggregated_stats_->overhead.dump_input_ms /= dcount;
            aggregated_stats_->overhead.dump_output_ms /= dcount;
            aggregated_stats_->overhead.verify_ms /= dcount;
            aggregated_stats_->overhead.callback_ms /= dcount;
            aggregated_stats_->overhead.get_dump_info_ms /= dcount;

            // Average phase stats (devices run in parallel)
            auto avgPhase = [dcount, count](PhaseStats &ps)
            {
                ps.total_execute_ms /= dcount;
                ps.total_stages_executed /= count;
                ps.total_collective_ms /= dcount;
                ps.total_collective_calls /= count;
                for (auto &[_, ms] : ps.stage_type_execute_ms)
                    ms /= dcount;
                for (auto &[_, cnt] : ps.stage_type_counts)
                    cnt /= count;
            };
            avgPhase(aggregated_stats_->prefill);
            avgPhase(aggregated_stats_->decode);
        }

        stats_dirty_ = false;
    }

    // =========================================================================
    // IInferenceRunner Interface Implementation
    // =========================================================================

    bool RankOrchestrator::forward(const int *tokens, int seq_len)
    {
        bool success = false;

        // Dispatch to appropriate implementation based on parallelism mode
        switch (mode_)
        {
        case ParallelismMode::TP:
            success = forwardTP(tokens, seq_len);
            break;
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            // PP and TP_PP both use sequential stage execution
            // The difference is that TP_PP stages may be nested MDOs (TP domains)
            // but forwardPP() works through IInferenceRunner interface regardless
            success = forwardPP(tokens, seq_len);
            break;
        default:
            LOG_ERROR("RankOrchestrator::forward: Unknown parallelism mode");
            return false;
        }

        return success;
    }

    // =========================================================================
    // TP Mode Forward Implementation (existing parallel execution)
    // =========================================================================
    bool RankOrchestrator::forwardTP(const int *tokens, int seq_len)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forwardTP: No device runners available");
            return false;
        }

        // TP timing diagnostic — enabled via LLAMINAR_TP_TIMING=1 or LLAMINAR_PROFILING=1
        const bool tp_timing = debugEnv().tp_timing;
        const bool tp_profiling = debugEnv().execution.executor_profiling;
        const bool collect_timing = tp_timing || tp_profiling;
        auto tp_t0 = collect_timing ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};

        LOG_DEBUG("RankOrchestrator::forwardTP: seq_len=" << seq_len
                                                          << ", devices=" << device_runners_.size());

        // Decode latency breakdown timing
        const bool decode_breakdown = collect_timing && seq_len == 1;
        auto launch_t0 = decode_breakdown ? std::chrono::high_resolution_clock::now()
                                          : std::chrono::high_resolution_clock::time_point{};
        auto launch_t1 = launch_t0; // Set properly after dispatch in parallel mode

        // DIAGNOSTIC: Run forward passes SEQUENTIALLY to test if concurrency causes crash.
        // If sequential execution works but parallel crashes, it's a concurrent HIP issue.
        const bool serialize_devices = debugEnv().runtime_debug.serialize_tp_forward;

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;

        if (serialize_devices)
        {
            // ----- SERIAL MODE (diagnostic fallback) -----
            for (size_t i = 0; i < device_runners_.size(); ++i)
            {
                auto &runner = device_runners_[i];
                if (!runner)
                    continue;
                // Cast to IInferenceRunner* to disambiguate forward() overloads
                IInferenceRunner *runner_iface = runner.get();
                LOG_WARN("RankOrchestrator::forwardTP: SERIAL mode - device "
                         << i << " running synchronously");
                try
                {
                    if (!runner_iface->forward(tokens, seq_len))
                    {
                        LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                  << i << " forward failed");
                        all_success = false;
                    }
                }
                catch (...)
                {
                    all_success = false;
                    if (!first_exception)
                    {
                        first_exception = std::current_exception();
                        first_exception_device = i;
                    }
                }
            }
        }
        else
        {
            // ----- PARALLEL MODE: persistent thread pool -----
            // Lazy-initialize on first TP forward call. Workers persist for the
            // lifetime of the orchestrator, eliminating per-step thread overhead.
            if (!tp_worker_pool_)
            {
                tp_worker_pool_ = std::make_unique<TPWorkerPool>(device_runners_.size());

                // Wire abort callback: when one worker fails (exception or false return),
                // abort the collective backend to unblock any workers stuck in NCCL/RCCL
                // collective calls. Without this, a failed worker exits the forward pass
                // while other workers block forever in ncclAllReduce waiting for all ranks.
                if (tp_ctx_)
                {
                    tp_worker_pool_->setFailureCallback([this]()
                                                        {
                        LOG_WARN("[TPWorkerPool] Worker failure detected — aborting collective backend");
                        tp_ctx_->requestAbort(); });
                }

                LOG_DEBUG("[TPWorkerPool] Created " << device_runners_.size()
                                                    << " persistent worker threads");
            }

            // Dispatch parallel forward passes to persistent worker threads.
            // dispatch() wakes all workers via condition_variable and returns
            // immediately — no thread creation, no pthread_create syscall.
            //
            // Capture profiler phases from the caller thread and propagate to
            // workers. All profilers use thread-local phase storage, so worker
            // threads must explicitly set their phase to match the caller.
            auto kernel_phase = KernelProfiler::getCurrentPhase();
            auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            auto kv_phase = KVCacheProfiler::getCurrentPhase();
            auto executor_phase = GraphExecutorStats::currentPhase();

            tp_worker_pool_->dispatch(
                [this, tokens, seq_len, kernel_phase, rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    // Propagate profiler phases from caller thread
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    // Set per-device profiler context so ROCm/CUDA profilers
                    // can attribute kernel times to specific GPUs
                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    IInferenceRunner *runner_iface = device_runners_[i].get();
                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " runner=" << static_cast<void *>(runner_iface)
                                  << " seq_len=" << seq_len);
                    }
                    try
                    {
                        const bool ok = runner_iface->forward(tokens, seq_len);
                        if (debugEnv().tp_collective_contract_trace)
                        {
                            LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_leave"
                                      << " worker=" << i
                                      << " device=" << device_id.toString()
                                      << " success=" << (ok ? 1 : 0));
                        }
                        return ok;
                    }
                    catch (...)
                    {
                        if (debugEnv().tp_collective_contract_trace)
                        {
                            LOG_ERROR("[TP_WORKER_CONTRACT] event=forward_exception"
                                      << " worker=" << i
                                      << " device=" << device_id.toString());
                        }
                        throw;
                    }
                });

            launch_t1 = decode_breakdown ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::high_resolution_clock::time_point{};

            // Collect results from all workers. Debug/Integration builds default to
            // a 30s safety net, while Release keeps unlimited waits unless
            // LLAMINAR_TP_COLLECT_TIMEOUT_MS is set explicitly.
            auto results = tp_worker_pool_->collectAll(debugEnv().tp_collect_timeout_ms);

            // Process results with fault-tolerant exception handling.
            // IMPORTANT: Store the FIRST substantive exception. When one device
            // throws (e.g., VerificationFailure), it can cause CUDA/HIP context
            // destruction, making other devices fail with misleading "context is
            // destroyed" errors. We want to surface the original root cause.
            for (auto &r : results)
            {
                if (!r.completed)
                {
                    LOG_ERROR("RankOrchestrator::forwardTP: Device "
                              << r.worker_index << " did not complete (stuck)");
                    if (tp_ctx_)
                    {
                        LOG_WARN("RankOrchestrator::forwardTP: requesting TP context abort after worker timeout");
                        tp_ctx_->requestAbort();
                    }
                    all_success = false;
                    continue;
                }

                if (r.exception)
                {
                    all_success = false;
                    try
                    {
                        std::rethrow_exception(r.exception);
                    }
                    catch (const std::exception &e)
                    {
                        std::string error_msg = e.what();
                        bool is_context_destroyed =
                            (error_msg.find("context is destroyed") != std::string::npos ||
                             error_msg.find("context destroyed") != std::string::npos ||
                             error_msg.find("error 709") != std::string::npos);

                        if (!first_exception)
                        {
                            first_exception = r.exception;
                            first_exception_device = r.worker_index;
                            LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                      << r.worker_index
                                      << " threw PRIMARY exception: " << error_msg);
                        }
                        else if (!is_context_destroyed)
                        {
                            // Substantive error — replace if first was a context error
                            try
                            {
                                std::rethrow_exception(first_exception);
                            }
                            catch (const std::exception &first_e)
                            {
                                std::string first_msg = first_e.what();
                                bool first_is_ctx =
                                    (first_msg.find("context is destroyed") != std::string::npos ||
                                     first_msg.find("context destroyed") != std::string::npos ||
                                     first_msg.find("error 709") != std::string::npos);
                                if (first_is_ctx)
                                {
                                    LOG_WARN("RankOrchestrator::forwardTP: Replacing "
                                             "secondary context error with primary error from device "
                                             << r.worker_index);
                                    first_exception = r.exception;
                                    first_exception_device = r.worker_index;
                                }
                            }
                            LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                      << r.worker_index
                                      << " threw exception: " << error_msg);
                        }
                        else
                        {
                            LOG_WARN("RankOrchestrator::forwardTP: Device "
                                     << r.worker_index
                                     << " threw SECONDARY exception (likely due to primary failure): "
                                     << error_msg);
                        }
                    }
                }
                else if (!r.success)
                {
                    LOG_ERROR("RankOrchestrator::forwardTP: Device "
                              << r.worker_index << " forward failed");
                    all_success = false;
                }
            }

            // Capture timing for decode breakdown (only in parallel mode)
            if (decode_breakdown)
            {
                auto collect_t1 = std::chrono::high_resolution_clock::now();
                double launch_us = std::chrono::duration<double, std::micro>(
                                       launch_t1 - launch_t0)
                                       .count();
                double wait_us = std::chrono::duration<double, std::micro>(
                                     collect_t1 - launch_t1)
                                     .count();
                double total_us = std::chrono::duration<double, std::micro>(
                                      collect_t1 - launch_t0)
                                      .count();
                if (tp_timing)
                {
                    LOG_DEBUG("[DECODE_BREAKDOWN] launch=" << std::fixed << std::setprecision(1)
                                                           << launch_us << "us"
                                                           << " wait=" << wait_us << "us"
                                                           << " total=" << total_us << "us");
                }
            }
        }

        // If we captured an exception, re-throw it with full context
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardTP: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        auto tp_t1 = collect_timing ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};

        if (all_success)
        {
            // Gather logits from all devices (delegates to LogitsGatherer)
            bool need_gather = logits_gatherer_ && logits_gatherer_->needsGather(seq_len);

            if (!need_gather && seq_len > 1)
            {
                static bool logged_prefill_skip = false;
                if (!logged_prefill_skip)
                {
                    LOG_DEBUG("[forwardTP] Skipping prefill logits gather (seq_len="
                              << seq_len << ") — prefill logits are not consumed");
                    logged_prefill_skip = true;
                }
            }

            // During prefill the LM head computes only 1 row of logits (the
            // last-token position) written to row 0 of logits_local.  We must
            // gather exactly 1 row; gathering seq_len rows would include
            // uninitialised data in rows 1..seq_len-1.
            size_t gather_rows = (seq_len > 1) ? 1 : static_cast<size_t>(seq_len);
            if (need_gather && !logits_gatherer_->gather(device_runners_, gather_rows, vocab_size()))
            {
                LOG_ERROR("RankOrchestrator::forwardTP: Failed to gather logits");
                all_success = false;
            }

            // Update position tracking
            current_position_ += seq_len;
            current_padded_seq_len_ = seq_len;
            stats_dirty_ = true;

            // After first prefill, release host-resident weight data.
            // GPU kernels (e.g., embedding repack) have now uploaded their own
            // device copies, and later MoE migration must use prepared/packed
            // weight transfer rather than falling back to raw GGUF bytes.
            if (!host_resident_released_ && seq_len > 1 && model_ctx_)
            {
                host_resident_released_ = true;
                if (auto wm = model_ctx_->weightManager())
                {
                    wm->releaseHostResidentWeightData();
                }
            }
        }

        if (collect_timing)
        {
            auto tp_t2 = std::chrono::high_resolution_clock::now();
            double forward_ms = std::chrono::duration<double, std::milli>(tp_t1 - tp_t0).count();
            double gather_ms = std::chrono::duration<double, std::milli>(tp_t2 - tp_t1).count();
            double total_ms = std::chrono::duration<double, std::milli>(tp_t2 - tp_t0).count();

            if (tp_timing)
            {
                LOG_DEBUG("[TP_TIMING] seq_len=" << seq_len
                                                 << " forward=" << std::fixed << std::setprecision(3) << forward_ms << "ms"
                                                 << " gather=" << std::fixed << std::setprecision(3) << gather_ms << "ms"
                                                 << " total=" << std::fixed << std::setprecision(3) << total_ms << "ms");
            }

            // Accumulate TP decode stats for profiling summary at benchmark end.
            // dispatch_ms = time to wake workers, wait_ms = time blocked on collect.
            // These are computed from DECODE_BREAKDOWN timestamps which are only
            // available in parallel (non-serialized) mode for decode (seq_len==1).
            if (tp_profiling && seq_len == 1)
            {
                double dispatch_ms = 0, wait_ms = 0;
                if (decode_breakdown && !serialize_devices)
                {
                    dispatch_ms = std::chrono::duration<double, std::milli>(launch_t1 - launch_t0).count();
                    wait_ms = std::chrono::duration<double, std::milli>(tp_t1 - launch_t1).count();
                }
                else
                {
                    // Serial mode or timing unavailable: entire forward time is wait
                    wait_ms = forward_ms;
                }
                tp_decode_stats_.record(total_ms, dispatch_ms, wait_ms, gather_ms);
            }
        }

        return all_success;
    }

    // =========================================================================
    // PP Mode Forward Implementation (sequential pipeline execution)
    // =========================================================================
    bool RankOrchestrator::forwardPP(const int *tokens, int seq_len)
    {
        if (pp_stage_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forwardPP: No PP stage runners available");
            return false;
        }

        if (!pp_ctx_)
        {
            LOG_ERROR("RankOrchestrator::forwardPP: No LocalPPContext available for transfers");
            return false;
        }

        const size_t num_stages = pp_stage_runners_.size();

        LOG_DEBUG("RankOrchestrator::forwardPP: seq_len=" << seq_len
                                                          << " num_stages=" << num_stages);

        // =====================================================================
        // Stage 0: Embedding + first layers (receives tokens as input)
        // =====================================================================
        auto &stage0_runner = pp_stage_runners_[0];
        if (!stage0_runner)
        {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage 0 runner is null");
            return false;
        }

        // PP device coherence: After PP transfer in the previous iteration, the
        // hidden state tensor's gpu_device_ may point to the *destination* device
        // (e.g., ROCm:1) instead of this stage's device (ROCm:0). Promote the
        // secondary buffer back to primary so kernels and transitionToWithEvent
        // operate on the correct device.
        if (pp_ctx_ && num_stages > 1)
        {
            TensorBase *hidden = stage0_runner->getHiddenState();
            if (hidden)
            {
                DeviceId stage0_device = pp_ctx_->deviceForStage(0).toLocalDeviceId();
                auto cur = hidden->current_device();
                if (cur.has_value() && *cur != stage0_device)
                {
                    LOG_DEBUG("RankOrchestrator::forwardPP: Re-asserting hidden state device from "
                              << cur->toString() << " to " << stage0_device.toString());
                    hidden->allocateOnDevice(stage0_device);
                }
            }
        }

        LOG_DEBUG("RankOrchestrator::forwardPP: Executing stage 0 (embedding)");
        if (!stage0_runner->forward(tokens, seq_len))
        {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage 0 forward failed");
            return false;
        }

        // =====================================================================
        // Intermediate stages: Transfer activations and continue execution
        // =====================================================================
        for (size_t stage_idx = 1; stage_idx < num_stages; ++stage_idx)
        {
            auto &prev_runner = pp_stage_runners_[stage_idx - 1];
            auto &curr_runner = pp_stage_runners_[stage_idx];

            if (!curr_runner)
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " runner is null");
                return false;
            }

            // Get hidden state from previous stage
            // DeviceGraphOrchestrator stores hidden state in InferenceState
            TensorBase *hidden_state = prev_runner->getHiddenState();
            if (!hidden_state)
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << (stage_idx - 1)
                                                                << " has no hidden state to transfer");
                return false;
            }

            // Transfer activations from previous stage to current stage
            // Only transfer the active region (seq_len tokens), not the full buffer
            const size_t active_bytes = pp_activation_contract_
                                            ? pp_activation_contract_->transfer(static_cast<int>(stage_idx - 1)).activeBytes(seq_len)
                                            : static_cast<size_t>(seq_len) * model_ctx_->embeddingLength() * sizeof(float);

            LOG_DEBUG("RankOrchestrator::forwardPP: Transferring hidden state from stage "
                      << (stage_idx - 1) << " to stage " << stage_idx
                      << " (" << active_bytes << " bytes for " << seq_len << " tokens)");

            if (!pp_ctx_->transfer(hidden_state, static_cast<int>(stage_idx - 1),
                                   static_cast<int>(stage_idx), active_bytes))
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Transfer from stage "
                          << (stage_idx - 1) << " to stage " << stage_idx << " failed");
                return false;
            }

            // Set the hidden state as input for current stage
            curr_runner->setHiddenState(hidden_state);

            LOG_DEBUG("RankOrchestrator::forwardPP: Executing stage " << stage_idx);

            // Call forward with nullptr tokens - the stage will use setHiddenState input
            // and skip embedding since has_embedding=false for non-first stages
            if (!curr_runner->forward(nullptr, seq_len))
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " forward failed");
                return false;
            }

            // Clear hidden state input for clean state on next forward
            curr_runner->clearHiddenStateInput();
        }

        // =====================================================================
        // Copy logits from last stage to combined buffer
        // =====================================================================
        {
            int last_stage = static_cast<int>(num_stages - 1);
            if (last_stage >= 0 && static_cast<size_t>(last_stage) < pp_stage_runners_.size() && pp_stage_runners_[last_stage])
            {
                if (!logits_gatherer_)
                    logits_gatherer_ = std::make_unique<LogitsGatherer>(0, 0);
                logits_gatherer_->copyFromStage(*pp_stage_runners_[last_stage],
                                                0, config_.batch_size, config_.max_seq_len);
            }
        }

        // =====================================================================
        // Update position tracking for PP mode
        // Each PP stage runner updates its own position internally, but we also
        // need to update current_position_ for consistency with TP mode and
        // get_position() queries.
        // =====================================================================
        current_position_ += seq_len;

        LOG_DEBUG("RankOrchestrator::forwardPP: Complete, all " << num_stages << " stages executed"
                                                                << ", position now " << current_position_);
        return true;
    }

    int RankOrchestrator::sampleGreedyOnDevice()
    {
        if (mode_ != ParallelismMode::TP || device_runners_.size() < 2)
            return -1;
        return DeviceSampler::sampleGreedy(device_runners_);
    }

    int RankOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        if (params.is_greedy())
            return sampleGreedyOnDevice();
        if (mode_ != ParallelismMode::TP || device_runners_.size() < 2)
            return -1;
        return DeviceSampler::sample(device_runners_, params);
    }

    const float *RankOrchestrator::logits() const
    {
        // For PP mode: return combined logits (copied from final stage)
        if (mode_ == ParallelismMode::PP || mode_ == ParallelismMode::TP_PP)
        {
            if (logits_gatherer_ && logits_gatherer_->isAllocated())
            {
                return logits_gatherer_->data();
            }
            // Fallback: try to get from final PP stage
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->logits();
            }
            return nullptr;
        }

        // For TP mode: return combined logits if available (multi-device)
        if (logits_gatherer_ && logits_gatherer_->isAllocated() && device_runners_.size() > 1)
        {
            return logits_gatherer_->data();
        }

        // For single device, return primary device's logits
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->logits();
        }

        return nullptr;
    }

    bool RankOrchestrator::forwardMTP(int32_t draft_condition_token)
    {
        if (device_runners_.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : device_runners_)
        {
            if (!runner || !runner->forwardMTP(draft_condition_token))
            {
                all_success = false;
            }
        }
        return all_success;
    }

    const float *RankOrchestrator::mtpLogits() const
    {
        if (device_runners_.empty())
        {
            return nullptr;
        }

        const int compare_vocab = device_runners_[0] ? device_runners_[0]->vocab_size() : 0;
        if (compare_vocab <= 0)
        {
            return nullptr;
        }

        const float *primary_logits = nullptr;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            if (runner->vocab_size() != compare_vocab)
                return nullptr;

            const float *child_logits = runner->mtpLogits();
            if (!child_logits)
                return nullptr;

            if (!primary_logits)
            {
                primary_logits = child_logits;
                continue;
            }

            if (!std::equal(primary_logits, primary_logits + compare_vocab, child_logits))
            {
                LOG_WARN("RankOrchestrator::mtpLogits: child MTP logits diverged in replicated TP path");
                return nullptr;
            }
        }
        return primary_logits;
    }

    bool RankOrchestrator::setComputeAllPositionLogits(bool enabled)
    {
        if (device_runners_.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : device_runners_)
        {
            if (!runner || !runner->setComputeAllPositionLogits(enabled))
            {
                all_success = false;
            }
        }
        return all_success;
    }

    const float *RankOrchestrator::getAllPositionLogits() const
    {
        if (device_runners_.empty())
        {
            return nullptr;
        }

        const int compare_vocab = device_runners_[0] ? device_runners_[0]->vocab_size() : 0;
        if (compare_vocab <= 0)
        {
            return nullptr;
        }

        const size_t rows = std::max<size_t>(1, static_cast<size_t>(std::max(0, current_padded_seq_len_)));
        const size_t count = rows * static_cast<size_t>(compare_vocab);

        const float *primary_logits = nullptr;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            if (runner->vocab_size() != compare_vocab)
                return nullptr;

            const float *child_logits = runner->getAllPositionLogits();
            if (!child_logits)
                return nullptr;

            if (!primary_logits)
            {
                primary_logits = child_logits;
                continue;
            }

            if (!std::equal(primary_logits, primary_logits + count, child_logits))
            {
                LOG_WARN("RankOrchestrator::getAllPositionLogits: child verifier logits diverged in replicated TP path");
                return nullptr;
            }
        }
        return primary_logits;
    }

    std::string RankOrchestrator::mtpDecodeUnsupportedReason() const
    {
        if (!pp_stage_runners_.empty())
        {
            return "MTP decode is not enabled for PP topologies";
        }
        if (device_runners_.size() > 1)
        {
            for (const auto &runner : device_runners_)
            {
                if (!runner)
                    return "MTP decode requires every TP participant to be available";

                const std::string child_reason = runner->mtpDecodeUnsupportedReason();
                if (!child_reason.empty())
                    return child_reason;
            }
        }
        return {};
    }

    void RankOrchestrator::setSkipLogitsGatherDecode(bool skip)
    {
        if (logits_gatherer_)
            logits_gatherer_->setSkipDecode(skip);
    }

    void RankOrchestrator::setSkipLogitsGatherPrefill(bool skip)
    {
        if (logits_gatherer_)
            logits_gatherer_->setSkipPrefill(skip);
    }

    bool RankOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forward_batch: No device runners available");
            return false;
        }

        LOG_DEBUG("RankOrchestrator::forward_batch: batch_size=" << token_batches.size()
                                                                 << ", devices=" << device_runners_.size());

        // Launch parallel batch forward passes on all devices
        std::vector<std::future<bool>> futures;
        futures.reserve(device_runners_.size());

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            auto &runner = device_runners_[i];
            if (runner)
            {
                futures.push_back(std::async(std::launch::async,
                                             [&runner, &token_batches]()
                                             {
                                                 return runner->forward_batch(token_batches);
                                             }));
            }
        }

        // Wait for all to complete - using same exception capture pattern as forward()
        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;

        for (size_t i = 0; i < futures.size(); ++i)
        {
            try
            {
                bool success = futures[i].get();
                if (!success)
                {
                    LOG_ERROR("RankOrchestrator::forward_batch: Device " << i << " forward_batch failed");
                    all_success = false;
                }
            }
            catch (const std::exception &e)
            {
                all_success = false;
                std::string error_msg = e.what();
                bool is_context_destroyed = (error_msg.find("context is destroyed") != std::string::npos ||
                                             error_msg.find("error 709") != std::string::npos);

                if (!first_exception)
                {
                    first_exception = std::current_exception();
                    first_exception_device = i;
                    LOG_ERROR("RankOrchestrator::forward_batch: Device " << i
                                                                         << " threw PRIMARY exception: " << error_msg);
                }
                else if (is_context_destroyed)
                {
                    LOG_WARN("RankOrchestrator::forward_batch: Device " << i
                                                                        << " threw SECONDARY exception (context destroyed): " << error_msg);
                }
                else
                {
                    LOG_ERROR("RankOrchestrator::forward_batch: Device " << i
                                                                         << " threw exception: " << error_msg);
                }
            }
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forward_batch: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (all_success)
        {
            // Update batch tracking from primary device
            if (!device_runners_.empty() && device_runners_[0])
            {
                current_batch_size_ = device_runners_[0]->batch_size();
                current_padded_seq_len_ = device_runners_[0]->padded_seq_len();
                current_sequence_lengths_ = device_runners_[0]->sequence_lengths();
            }
            stats_dirty_ = true;
        }

        return all_success;
    }

    const float *RankOrchestrator::getLogits(int seq_idx) const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getLogits(seq_idx);
        }
        return nullptr;
    }

    int RankOrchestrator::batch_size() const
    {
        return current_batch_size_;
    }

    int RankOrchestrator::padded_seq_len() const
    {
        return current_padded_seq_len_;
    }

    const std::vector<int> &RankOrchestrator::sequence_lengths() const
    {
        return current_sequence_lengths_;
    }

    int RankOrchestrator::vocab_size() const
    {
        // For tensor-parallel setups, the LM head may be column-sharded across devices.
        // In that case, each device has logits for vocab_size/tp_degree tokens.
        // We should return the FULL vocab size (sum of all devices), not just device 0's.
        //
        // The model_ctx_ always has the true total vocab size from the model metadata.
        if (model_ctx_)
        {
            return static_cast<int>(model_ctx_->vocabSize());
        }

        // Fallback: if no model context, use device 0's vocab (may be wrong for TP)
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->vocab_size();
        }
        return 0;
    }

    void RankOrchestrator::clear_cache()
    {
        LOG_DEBUG("RankOrchestrator::clear_cache: Clearing cache on all "
                  << device_runners_.size() << " TP devices and "
                  << pp_stage_runners_.size() << " PP stages");
        last_device_prefix_hits_.clear();
        last_pp_prefix_hits_.clear();

        // Clear TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->clear_cache();
            }
        }

        // Clear PP stage runners (each has its own KV cache)
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->clear_cache();
            }
        }

        current_position_ = 0;
        stats_dirty_ = true;
    }

    int RankOrchestrator::get_position() const
    {
        // PP mode: return position from first PP stage runner
        // All PP stage runners should have synchronized positions
        if (!pp_stage_runners_.empty() && pp_stage_runners_[0])
        {
            return pp_stage_runners_[0]->get_position();
        }

        // TP mode: return position from primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->get_position();
        }

        return current_position_;
    }

    ExecutionPath RankOrchestrator::executionPath() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->executionPath();
        }
        return ExecutionPath::GRAPH;
    }

    const char *RankOrchestrator::architecture() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->architecture();
        }
        return "Unknown";
    }

    PrefixLookupResult RankOrchestrator::lookupPrefix(const std::vector<int32_t> &tokens)
    {
        PrefixLookupResult aggregate;
        last_device_prefix_hits_.clear();
        last_pp_prefix_hits_.clear();

        std::vector<PrefixParticipantLookup> participants;
        int block_size = 0;
        int participant_id = 0;

        auto query_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                 std::vector<PrefixLookupResult> &hits)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                PrefixLookupResult hit = runner->lookupPrefix(tokens);
                participants.push_back(makePrefixParticipantLookup(
                    participant_id++,
                    runner->primaryDeviceId(),
                    hit));
                hits.push_back(hit);
                if (block_size <= 0 && hit.block_size > 0)
                    block_size = hit.block_size;
            }
        };

        query_runners(device_runners_, last_device_prefix_hits_);
        query_runners(pp_stage_runners_, last_pp_prefix_hits_);

        if (participants.empty())
            return aggregate;

        return makePrefixLookupResult(coordinatePrefixLookups(std::move(participants)), block_size);
    }

    bool RankOrchestrator::populatePrefix(const PrefixLookupResult &hit, int seq_idx)
    {
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0)
            return true;

        auto populate_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                    const std::vector<PrefixLookupResult> &hits)
        {
            size_t hit_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                if (hit_index >= hits.size())
                    return false;

                PrefixLookupResult child_hit = hits[hit_index++].clampedTo(common_tokens);
                if (!runner->populatePrefix(child_hit, seq_idx))
                    return false;
            }
            return hit_index == hits.size();
        };

        const bool populated =
            populate_runners(device_runners_, last_device_prefix_hits_) &&
            populate_runners(pp_stage_runners_, last_pp_prefix_hits_);
        if (!populated)
        {
            clear_cache();
        }
        return populated;
    }

    bool RankOrchestrator::harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count)
    {
        bool saw_runner = false;
        bool ok = true;
        auto harvest_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                saw_runner = true;
                ok = runner->harvestPrefix(tokens, prompt_token_count) && ok;
            }
        };

        harvest_runners(device_runners_);
        harvest_runners(pp_stage_runners_);
        return saw_runner && ok;
    }

    bool RankOrchestrator::restorePrefixTerminalState(const PrefixLookupResult &hit)
    {
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0)
            return false;

        auto restore_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                   const std::vector<PrefixLookupResult> &hits)
        {
            size_t hit_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                PrefixLookupResult child_hit =
                    hit_index < hits.size() ? hits[hit_index++].clampedTo(common_tokens)
                                            : hit.clampedTo(common_tokens);
                if (!runner->restorePrefixTerminalState(child_hit))
                    return false;
            }
            return true;
        };

        return restore_runners(device_runners_, last_device_prefix_hits_) &&
               restore_runners(pp_stage_runners_, last_pp_prefix_hits_);
    }

    PrefixStateSnapshot RankOrchestrator::captureLivePrefixState(int seq_idx) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        int common_tokens = 0;

        auto capture_runners = [&](const std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                PrefixStateSnapshot child = runner->captureLivePrefixState(seq_idx);
                if (!child.valid)
                    return false;

                if (!have_common_tokens)
                {
                    common_tokens = child.cached_tokens;
                    have_common_tokens = true;
                }
                else if (child.cached_tokens != common_tokens)
                {
                    return false;
                }

                aggregate.participant_snapshots.push_back(std::move(child));
            }
            return true;
        };

        if (!capture_runners(device_runners_) ||
            !capture_runners(pp_stage_runners_) ||
            !saw_runner ||
            !have_common_tokens)
        {
            return {};
        }

        aggregate.valid = true;
        aggregate.cached_tokens = common_tokens;
        return aggregate;
    }

    bool RankOrchestrator::restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        if (!snapshot.valid || snapshot.participant_snapshots.empty())
            return false;

        size_t participant_index = 0;
        bool ok = true;
        bool saw_runner = false;
        auto restore_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                if (participant_index >= snapshot.participant_snapshots.size())
                {
                    ok = false;
                    continue;
                }
                ok = runner->restoreLivePrefixState(
                         snapshot.participant_snapshots[participant_index++],
                         seq_idx) &&
                     ok;
            }
        };

        restore_runners(device_runners_);
        restore_runners(pp_stage_runners_);
        return saw_runner && ok && participant_index == snapshot.participant_snapshots.size();
    }

    bool RankOrchestrator::truncateLivePrefixState(int cached_tokens, int seq_idx)
    {
        bool saw_runner = false;
        bool ok = true;
        auto truncate_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                saw_runner = true;
                ok = runner->truncateLivePrefixState(cached_tokens, seq_idx) && ok;
            }
        };

        truncate_runners(device_runners_);
        truncate_runners(pp_stage_runners_);
        return saw_runner && ok;
    }

    PrefixRuntimeStateSnapshot RankOrchestrator::prefixStateProbe() const
    {
        PrefixRuntimeStateSnapshot snapshot;
        snapshot.initialized = true;
        snapshot.architecture = architecture();
        snapshot.execution_path = "rank-orchestrator";
        snapshot.current_position = get_position();
        snapshot.positions = {snapshot.current_position};
        snapshot.sequence_lengths = current_sequence_lengths_;

        auto merge_child = [&snapshot](const PrefixRuntimeStateSnapshot &child)
        {
            snapshot.initialized = snapshot.initialized && child.initialized;
            snapshot.has_hidden = snapshot.has_hidden || child.has_hidden;
            snapshot.has_logits = snapshot.has_logits || child.has_logits;
            snapshot.session_epoch = std::max(snapshot.session_epoch, child.session_epoch);
            snapshot.prefix_cache_config_enabled =
                snapshot.prefix_cache_config_enabled || child.prefix_cache_config_enabled;
            snapshot.prefix_cache_ready = snapshot.prefix_cache_ready || child.prefix_cache_ready;
            snapshot.prefix_cache_bypassed =
                snapshot.prefix_cache_bypassed || child.prefix_cache_bypassed;
            if (snapshot.prefix_cache_bypass_reason.empty() &&
                !child.prefix_cache_bypass_reason.empty())
            {
                snapshot.prefix_cache_bypass_reason = child.prefix_cache_bypass_reason;
            }
            snapshot.prefix_cache_lookups += child.prefix_cache_lookups;
            snapshot.prefix_cache_hits += child.prefix_cache_hits;
            snapshot.prefix_cache_partial_hits += child.prefix_cache_partial_hits;
            snapshot.prefix_cache_misses += child.prefix_cache_misses;
            snapshot.prefix_cache_matched_blocks += child.prefix_cache_matched_blocks;
            snapshot.prefix_cache_matched_tokens += child.prefix_cache_matched_tokens;
            snapshot.prefix_cache_stores += child.prefix_cache_stores;
            snapshot.prefix_cache_inserts += child.prefix_cache_inserts;
            snapshot.prefix_cache_evictions += child.prefix_cache_evictions;
            snapshot.prefix_cache_promotions += child.prefix_cache_promotions;
            snapshot.prefix_cache_disk_hydrations += child.prefix_cache_disk_hydrations;
            snapshot.prefix_cache_terminal_state_hits += child.prefix_cache_terminal_state_hits;
            snapshot.prefix_cache_ram_bytes += child.prefix_cache_ram_bytes;
            snapshot.prefix_cache_device_bytes += child.prefix_cache_device_bytes;
            snapshot.prefix_cache_disk_bytes += child.prefix_cache_disk_bytes;
            snapshot.prefix_cache_hybrid_state_bytes += child.prefix_cache_hybrid_state_bytes;
            snapshot.prefix_cache_mtp_state_bytes += child.prefix_cache_mtp_state_bytes;
            snapshot.prefix_cache_bypasses += child.prefix_cache_bypasses;
            snapshot.prefix_cache_unsupported_backend_bypasses +=
                child.prefix_cache_unsupported_backend_bypasses;
            snapshot.prefix_cache_fingerprint_bypasses +=
                child.prefix_cache_fingerprint_bypasses;
            snapshot.prefix_cache_terminal_state_bypasses +=
                child.prefix_cache_terminal_state_bypasses;
            snapshot.mtp_config_enabled = snapshot.mtp_config_enabled || child.mtp_config_enabled;
            snapshot.mtp_bypassed = snapshot.mtp_bypassed || child.mtp_bypassed;
            if (snapshot.mtp_bypass_reason.empty() && !child.mtp_bypass_reason.empty())
            {
                snapshot.mtp_bypass_reason = child.mtp_bypass_reason;
            }
            snapshot.mtp_draft_steps += child.mtp_draft_steps;
            snapshot.mtp_accepted_tokens += child.mtp_accepted_tokens;
            snapshot.mtp_rejected_tokens += child.mtp_rejected_tokens;
            snapshot.mtp_rollbacks += child.mtp_rollbacks;
            snapshot.mtp_bypasses += child.mtp_bypasses;
            snapshot.mtp_verifier_runs += child.mtp_verifier_runs;
            snapshot.mtp_verifier_token_count += child.mtp_verifier_token_count;
            if (snapshot.primary_device.is_cpu() && !child.primary_device.is_cpu())
            {
                snapshot.primary_device = child.primary_device;
            }
            if (snapshot.positions.empty() && !child.positions.empty())
            {
                snapshot.positions = child.positions;
            }
            if (snapshot.sequence_lengths.empty() && !child.sequence_lengths.empty())
            {
                snapshot.sequence_lengths = child.sequence_lengths;
            }
            snapshot.kv_caches.insert(snapshot.kv_caches.end(), child.kv_caches.begin(), child.kv_caches.end());
            snapshot.mtp_kv_caches.insert(snapshot.mtp_kv_caches.end(),
                                          child.mtp_kv_caches.begin(),
                                          child.mtp_kv_caches.end());
            snapshot.gdn_layers.insert(snapshot.gdn_layers.end(), child.gdn_layers.begin(), child.gdn_layers.end());
        };

        bool saw_child = false;
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                merge_child(runner->prefixStateProbe());
                saw_child = true;
            }
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                merge_child(runner->prefixStateProbe());
                saw_child = true;
            }
        }

        if (!saw_child)
        {
            snapshot.initialized = false;
        }
        if (snapshot.prefix_cache_bypassed)
        {
            snapshot.prefix_cache_ready = false;
        }

        return snapshot;
    }

    // =========================================================================
    // Hidden State API (for Pipeline Parallelism nesting)
    // =========================================================================

    TensorBase *RankOrchestrator::getHiddenState()
    {
        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, all devices have same hidden state after allreduce
            // Delegate to primary device runner
            if (!device_runners_.empty() && device_runners_[0])
            {
                return device_runners_[0]->getHiddenState();
            }
        }
        else
        {
            // PP or TP_PP mode - last stage has the final hidden state
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->getHiddenState();
            }
        }
        return nullptr;
    }

    const TensorBase *RankOrchestrator::getHiddenState() const
    {
        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, all devices have same hidden state after allreduce
            // Delegate to primary device runner
            if (!device_runners_.empty() && device_runners_[0])
            {
                return device_runners_[0]->getHiddenState();
            }
        }
        else
        {
            // PP or TP_PP mode - last stage has the final hidden state
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->getHiddenState();
            }
        }
        return nullptr;
    }

    void RankOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        hidden_state_input_ = hidden_state;

        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, set on ALL device runners (they all need the same input)
            for (auto &runner : device_runners_)
            {
                if (runner)
                {
                    runner->setHiddenState(hidden_state);
                }
            }
        }
        else
        {
            // PP or TP_PP mode - set on first stage runner (stage 0 receives external input)
            if (!pp_stage_runners_.empty() && pp_stage_runners_.front())
            {
                pp_stage_runners_.front()->setHiddenState(hidden_state);
            }
        }
    }

    bool RankOrchestrator::hasHiddenStateInput() const
    {
        return hidden_state_input_ != nullptr;
    }

    void RankOrchestrator::clearHiddenStateInput()
    {
        hidden_state_input_ = nullptr;

        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, clear on all device runners
            for (auto &runner : device_runners_)
            {
                if (runner)
                {
                    runner->clearHiddenStateInput();
                }
            }
        }
        else
        {
            // PP or TP_PP mode - clear on first stage runner
            if (!pp_stage_runners_.empty() && pp_stage_runners_.front())
            {
                pp_stage_runners_.front()->clearHiddenStateInput();
            }
        }
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void RankOrchestrator::enableSnapshotCapture(const std::string &output_dir)
    {
        LOG_DEBUG("RankOrchestrator::enableSnapshotCapture: Enabling on all devices");

        // Enable on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->enableSnapshotCapture(output_dir);
            }
        }

        // Enable on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->enableSnapshotCapture(output_dir);
            }
        }
    }

    void RankOrchestrator::disableSnapshotCapture()
    {
        LOG_DEBUG("RankOrchestrator::disableSnapshotCapture: Disabling on all devices");

        // Disable on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->disableSnapshotCapture();
            }
        }

        // Disable on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->disableSnapshotCapture();
            }
        }
    }

    void RankOrchestrator::clearSnapshots()
    {
        LOG_DEBUG("RankOrchestrator::clearSnapshots: Clearing on all devices");

        // Clear on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->clearSnapshots();
            }
        }

        // Clear on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->clearSnapshots();
            }
        }
    }

    const float *RankOrchestrator::getSnapshot(const std::string &key, size_t &out_size) const
    {
        // For GATHERED stages (e.g., LM_HEAD) with multi-device TP, return the gathered combined_logits.
        // This is necessary because each device only has logits_local with vocab_local entries,
        // but tests expect the full vocab_size logits.
        //
        // CRITICAL: For hybrid PP+TP, only the stage that actually HAS the LM head should
        // return combined_logits. Otherwise, a nested TP stage (like PP stage 0) that has
        // logits_local buffers allocated but never computes LM_HEAD would return stale data.
        if (getStageShardingMode(key, stage_sharding_map_) == SnapshotShardingMode::GATHERED &&
            device_runners_.size() > 1 && logits_gatherer_ && logits_gatherer_->isAllocated() && tp_ctx_)
        {
            bool owns_lm_head = true;
            if (config_.nested_pp_stage_config.has_value())
            {
                owns_lm_head = config_.nested_pp_stage_config->has_lm_head;
            }
            else
            {
                auto wm = model_ctx_->weightManager();
                owns_lm_head = wm && wm->hasLMHead();
            }

            if (!owns_lm_head)
            {
                LOG_DEBUG("RankOrchestrator::getSnapshot LM_HEAD: this stage doesn't own LM_HEAD, "
                          << "falling through to PP stage search"
                          << " (nested_pp_config=" << config_.nested_pp_stage_config.has_value()
                          << " has_lm_head=" << (config_.nested_pp_stage_config.has_value() ? config_.nested_pp_stage_config->has_lm_head : false) << ")");
            }
            else
            {
                bool has_column_parallel_lm_head = false;
                for (const auto &runner : device_runners_)
                {
                    if (runner && runner->hasLogitsLocal())
                    {
                        has_column_parallel_lm_head = true;
                        break;
                    }
                }

                if (has_column_parallel_lm_head)
                {
                    out_size = logits_gatherer_->lastGatheredSize();
                    const float *ptr = logits_gatherer_->data();
                    LOG_DEBUG("RankOrchestrator::getSnapshot LM_HEAD returning combined_logits with "
                              << out_size << " elements (column-parallel gathering), ptr=" << (void *)ptr
                              << " first_element=" << (ptr ? ptr[0] : -999999.0f));
                    return ptr;
                }
            }
        }

        // PP mode: search across all PP stage runners
        if (!pp_stage_runners_.empty())
        {
            for (const auto &runner : pp_stage_runners_)
            {
                if (runner)
                {
                    const float *result = runner->getSnapshot(key, out_size);
                    if (result != nullptr)
                    {
                        return result;
                    }
                }
            }
            out_size = 0;
            return nullptr;
        }

        // Default (TP mode): get from primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getSnapshot(key, out_size);
        }
        out_size = 0;
        return nullptr;
    }

    std::vector<std::string> RankOrchestrator::getSnapshotKeys() const
    {
        // Merge keys from all devices (use set to deduplicate)
        std::set<std::string> all_keys;

        // Collect from TP device runners
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                auto keys = runner->getSnapshotKeys();
                all_keys.insert(keys.begin(), keys.end());
            }
        }

        // Collect from PP stage runners
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                auto keys = runner->getSnapshotKeys();
                all_keys.insert(keys.begin(), keys.end());
            }
        }

        return std::vector<std::string>(all_keys.begin(), all_keys.end());
    }

    SnapshotInfo RankOrchestrator::getSnapshotWithShape(const std::string &key) const
    {
        // PP mode: search across all PP stage runners
        if (!pp_stage_runners_.empty())
        {
            for (const auto &runner : pp_stage_runners_)
            {
                if (runner)
                {
                    auto snap = runner->getSnapshotWithShape(key);
                    if (snap)
                        return snap;
                }
            }
            return {};
        }

        // Default (TP mode): get from primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getSnapshotWithShape(key);
        }
        return {};
    }

    TPSnapshot RankOrchestrator::getTPSnapshot(const std::string &key) const
    {
        TPSnapshot result;
        result.key = key;
        result.mode = getStageShardingMode(key, stage_sharding_map_);
        result.tp_degree = static_cast<int>(device_runners_.size());

        LOG_DEBUG("RankOrchestrator::getTPSnapshot: key=" << key
                                                          << " mode=" << shardingModeToString(result.mode)
                                                          << " tp_degree=" << result.tp_degree);

        // =========================================================================
        // PP Mode: Delegate to appropriate PP stage runner
        // For PP+TP hybrid, each PP stage runner may be an MDO for TP
        // =========================================================================
        if (!pp_stage_runners_.empty())
        {
            // Find which PP stage owns this snapshot key
            for (size_t stage_idx = 0; stage_idx < pp_stage_runners_.size(); ++stage_idx)
            {
                if (!pp_stage_runners_[stage_idx])
                    continue;

                // Check if this stage has the snapshot (with shape metadata)
                auto snap = pp_stage_runners_[stage_idx]->getSnapshotWithShape(key);
                if (!snap)
                    continue;

                // Found the owning stage - check if it's a TP domain (RankOrchestrator)
                auto *inner_mdo = dynamic_cast<const RankOrchestrator *>(
                    pp_stage_runners_[stage_idx].get());

                if (inner_mdo)
                {
                    // This PP stage is a TP domain - delegate to its getTPSnapshot
                    LOG_DEBUG("RankOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                           << " is TP domain, delegating getTPSnapshot for key=" << key);
                    return inner_mdo->getTPSnapshot(key);
                }
                else
                {
                    // Single-device PP stage - wrap the snapshot as non-TP
                    LOG_DEBUG("RankOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                           << " is single device for key=" << key);

                    result.tp_degree = 1;
                    result.mode = SnapshotShardingMode::REPLICATED;

                    DeviceSnapshotData dev_data;
                    dev_data.device_index = 0;
                    dev_data.rows = snap.rows;
                    dev_data.cols = snap.cols;
                    dev_data.global_start_col = 0;
                    dev_data.global_total_cols = snap.cols;
                    dev_data.data.assign(snap.data, snap.data + snap.size);

                    result.device_data.push_back(std::move(dev_data));
                    result.combined_valid = true;
                    result.combined_data = result.device_data[0].data;
                    result.combined_rows = snap.rows;
                    result.combined_cols = snap.cols;

                    return result;
                }
            }

            // Key not found in any PP stage
            LOG_DEBUG("RankOrchestrator::getTPSnapshot: key=" << key
                                                              << " not found in any PP stage");
            return result;
        }

        // =========================================================================
        // TP Mode (non-PP): Collect from device_runners_
        // =========================================================================

        // Special case: GATHERED stages (e.g., LM_HEAD) with combined logits already gathered
        if (getStageShardingMode(key, stage_sharding_map_) == SnapshotShardingMode::GATHERED &&
            device_runners_.size() > 1 && logits_gatherer_ && logits_gatherer_->isAllocated() && tp_ctx_)
        {
            bool has_column_parallel_lm_head = false;
            for (const auto &runner : device_runners_)
            {
                if (runner && runner->hasLogitsLocal())
                {
                    has_column_parallel_lm_head = true;
                    break;
                }
            }

            if (has_column_parallel_lm_head)
            {
                size_t gathered_size = logits_gatherer_->lastGatheredSize();
                const float *gathered_ptr = logits_gatherer_->data();
                DeviceSnapshotData gathered;
                gathered.device_id = GlobalDeviceId::gpu(0, 0, config_.devices[0].device_type);
                gathered.device_index = 0;
                gathered.rows = 1;
                gathered.cols = gathered_size;
                gathered.global_start_col = 0;
                gathered.global_total_cols = gathered_size;
                gathered.data.assign(gathered_ptr, gathered_ptr + gathered_size);

                result.device_data.push_back(std::move(gathered));
                result.combined_valid = true;
                result.combined_data = result.device_data[0].data;
                result.combined_rows = result.device_data[0].rows;
                result.combined_cols = result.device_data[0].cols;

                LOG_DEBUG("RankOrchestrator::getTPSnapshot: LM_HEAD using combined_logits "
                          << "size=" << gathered_size);
                return result;
            }
        }

        // Collect snapshots from all device runners
        size_t global_col_offset = 0;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                continue;

            // Use shape-aware snapshot retrieval — the stage itself reported
            // its output rows/cols via getDumpInfo() at capture time, so we
            // don't need model-specific dimension calculations here.
            auto snap = device_runners_[i]->getSnapshotWithShape(key);

            if (!snap)
            {
                LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                     << " has no data for key=" << key);
                continue;
            }

            // Debug: log data pointer and first 4 values to verify each device has different data
            LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                 << " key=" << key << " ptr=" << static_cast<const void *>(snap.data)
                                                                 << " size=" << snap.size
                                                                 << " shape=[" << snap.rows << "x" << snap.cols << "]"
                                                                 << " val[0-3]=" << snap.data[0] << "," << snap.data[1] << "," << snap.data[2] << "," << snap.data[3]);

            DeviceSnapshotData dev_data;
            // Use device type from config if available, otherwise default to CUDA
            DeviceType dev_type = DeviceType::CUDA;
            if (i < config_.devices.size())
            {
                dev_type = config_.devices[i].device_type;
            }
            dev_data.device_id = GlobalDeviceId::gpu(0, static_cast<int>(i), dev_type);
            dev_data.device_index = static_cast<int>(i);
            dev_data.data.assign(snap.data, snap.data + snap.size);

            // Use shape metadata from the stage's getDumpInfo() output.
            // This is model-agnostic: stages report their own dimensions
            // (e.g., KV projections report kv_dim cols, FFN reports d_ff cols).
            if (result.mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                dev_data.rows = snap.rows;
                dev_data.cols = snap.cols;
                dev_data.global_start_col = global_col_offset;
                global_col_offset += snap.cols;
            }
            else
            {
                // Replicated or row-parallel - each device has full output
                dev_data.rows = snap.rows;
                dev_data.cols = snap.cols;
                dev_data.global_start_col = 0;
                dev_data.global_total_cols = snap.cols;
            }

            LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                 << " size=" << snap.size
                                                                 << " cols=" << dev_data.cols
                                                                 << " start_col=" << dev_data.global_start_col);

            result.device_data.push_back(std::move(dev_data));
        }

        // Set global_total_cols for column-parallel stages
        if (result.mode == SnapshotShardingMode::COLUMN_PARALLEL && !result.device_data.empty())
        {
            size_t total_cols = global_col_offset;
            for (auto &dev : result.device_data)
            {
                dev.global_total_cols = total_cols;
            }
        }

        return result;
    }

    std::vector<std::pair<std::string, SnapshotShardingMode>>
    RankOrchestrator::getSnapshotKeysWithSharding() const
    {
        auto keys = getSnapshotKeys();
        std::vector<std::pair<std::string, SnapshotShardingMode>> result;
        result.reserve(keys.size());

        for (const auto &key : keys)
        {
            result.emplace_back(key, getStageShardingMode(key, stage_sharding_map_));
        }

        return result;
    }

    // =========================================================================
    // Profiling API
    // =========================================================================

    const GraphExecutorStats *RankOrchestrator::executorStats() const
    {
        aggregateStats();
        return aggregated_stats_.get();
    }

    void RankOrchestrator::resetExecutorStats()
    {
        LOG_DEBUG("RankOrchestrator::resetExecutorStats: Resetting on all devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->resetExecutorStats();
            }
        }

        if (aggregated_stats_)
        {
            aggregated_stats_->reset();
        }
        tp_decode_stats_.reset();
        stats_dirty_ = true;
    }

    // =========================================================================
    // Orchestration API
    // =========================================================================

    bool RankOrchestrator::hasPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->hasPlacementPlan();
        }
        return false;
    }

    const PlacementPlan &RankOrchestrator::getPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getPlacementPlan();
        }
        throw std::runtime_error("No placement plan available: no device runners");
    }

    // =========================================================================
    // IRankOrchestrator Interface Implementation
    // =========================================================================

    int RankOrchestrator::device_count() const
    {
        return static_cast<int>(device_runners_.size());
    }

    IInferenceRunner *RankOrchestrator::deviceRunner(int device_idx)
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    const IInferenceRunner *RankOrchestrator::deviceRunner(int device_idx) const
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    ILocalTPContext *RankOrchestrator::localTPContext()
    {
        return tp_ctx_.get();
    }

    const ILocalTPContext *RankOrchestrator::localTPContext() const
    {
        return tp_ctx_.get();
    }

    bool RankOrchestrator::allDevicesReady() const
    {
        if (device_runners_.empty())
        {
            return false;
        }

        for (const auto &runner : device_runners_)
        {
            if (!runner)
            {
                return false;
            }
            // Check if runner is ready (has vocab_size > 0 indicates initialization)
            if (runner->vocab_size() <= 0)
            {
                return false;
            }
        }

        return true;
    }

    void RankOrchestrator::synchronizeDevices()
    {
        LOG_DEBUG("RankOrchestrator::synchronizeDevices: Synchronizing all devices");

        if (tp_ctx_)
        {
            tp_ctx_->synchronize();
        }
    }

    MoERebalanceController *RankOrchestrator::moeRebalanceController() const
    {
        for (const auto &runner : device_runners_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
            {
                if (auto *controller = dgo->moeRebalanceController())
                    return controller;
            }
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
            {
                if (auto *controller = rank->moeRebalanceController())
                    return controller;
            }
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
            {
                if (auto *controller = dgo->moeRebalanceController())
                    return controller;
            }
        }
        return nullptr;
    }

    void RankOrchestrator::applyMoEExpertMasksForAllDevices(const MoERebalanceController &controller)
    {
        const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
        if (gpu_cache_experts > 0)
        {
            auto masks_by_socket = controller.computeGpuCacheExpertMasks(gpu_cache_experts);
            applyMoEExpertMasksForAllDevices(masks_by_socket);
            return;
        }

        std::vector<std::vector<std::vector<bool>>> masks_by_socket;
        masks_by_socket.reserve(device_runners_.size());
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
            masks_by_socket.push_back(controller.computeExpertMasks(static_cast<int>(device_idx)));

        applyMoEExpertMasksForAllDevices(masks_by_socket);
    }

    void RankOrchestrator::applyMoEExpertMasksForAllDevices(
        const std::vector<std::vector<std::vector<bool>>> &masks_by_socket)
    {
        int applied = 0;
        std::vector<DeviceGraphOrchestrator *> local_dgos;
        local_dgos.reserve(device_runners_.size());
        for (auto &runner : device_runners_)
            local_dgos.push_back(dynamic_cast<DeviceGraphOrchestrator *>(runner.get()));

        auto collect_local_transfers = [&](size_t destination_idx,
                                           const std::vector<std::vector<bool>> &destination_masks)
        {
            ReceivedWeightsMap merged;
            for (size_t source_idx = 0; source_idx < local_dgos.size(); ++source_idx)
            {
                if (source_idx == destination_idx || !local_dgos[source_idx])
                    continue;

                auto source_blobs = local_dgos[source_idx]->collectExpertWeightsForMasks(destination_masks);
                for (auto &layer_entry : source_blobs)
                {
                    auto &dst_layer = merged[layer_entry.first];
                    for (auto &expert_entry : layer_entry.second)
                    {
                        if (dst_layer.find(expert_entry.first) == dst_layer.end())
                            dst_layer.emplace(expert_entry.first, std::move(expert_entry.second));
                    }
                }
            }
            return merged;
        };

        std::vector<ReceivedWeightsMap> received_by_device(device_runners_.size());
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx < masks_by_socket.size() && local_dgos[device_idx])
                received_by_device[device_idx] = collect_local_transfers(device_idx, masks_by_socket[device_idx]);
        }

        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx >= masks_by_socket.size())
                continue;

            auto *dgo = local_dgos[device_idx];
            if (!dgo)
                continue;

            dgo->applyExpertMasks(masks_by_socket[device_idx], received_by_device[device_idx]);
            ++applied;
        }

        if (!pp_stage_runners_.empty())
        {
            for (auto &runner : pp_stage_runners_)
            {
                if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
                {
                    rank->applyMoEExpertMasksForAllDevices(masks_by_socket);
                    ++applied;
                    continue;
                }
                if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                {
                    if (!masks_by_socket.empty())
                    {
                        dgo->applyExpertMasks(masks_by_socket.front(), ReceivedWeightsMap{});
                        ++applied;
                    }
                }
            }
        }
        LOG_DEBUG("[RankOrchestrator] Applied LocalTP MoE expert masks to "
                  << applied << "/" << device_runners_.size() << " device runners");
    }

    void RankOrchestrator::setExpertReplicaSetForAllDevices(const ExpertReplicaSet &replicas)
    {
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(device_runners_[device_idx].get());
            if (dgo)
                dgo->setExpertReplicaSet(replicas, static_cast<int>(device_idx));
        }
    }

    void RankOrchestrator::setSuppressTimeline(bool suppress)
    {
        for (auto &runner : device_runners_)
            runner->setSuppressTimeline(suppress);
    }

    void RankOrchestrator::setAccumulatePrefill(bool accumulate)
    {
        for (auto &runner : device_runners_)
            runner->setAccumulatePrefill(accumulate);
    }

    void RankOrchestrator::flushStageTimeline()
    {
        // Print per-device GPU stage timelines
        for (auto &runner : device_runners_)
            runner->flushStageTimeline();

        // Print TP orchestrator decode summary if profiling data was collected
        if (tp_decode_stats_.iterations > 0)
        {
            const size_t n = tp_decode_stats_.iterations;
            const double avg_wall = tp_decode_stats_.total_wall_ms / n;
            const double avg_dispatch = tp_decode_stats_.total_dispatch_ms / n;
            const double avg_wait = tp_decode_stats_.total_wait_ms / n;
            const double avg_gather = tp_decode_stats_.total_gather_ms / n;
            // "Other" captures orchestrator overhead not attributed to
            // dispatch/wait/gather: exception checking, position tracking,
            // condition evaluation, etc.
            const double avg_other = avg_wall - avg_dispatch - avg_wait - avg_gather;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            {
                std::ostringstream title;
                title << "TP DECODE ORCHESTRATOR PROFILING"
                      << " (avg of " << n << " decode steps, "
                      << device_runners_.size() << " devices)";
                table << title.str() << "" << "" << fort::endr;
                table[0][0].set_cell_span(3);
                table[0][0].set_cell_text_align(fort::text_align::center);
            }

            // Summary
            {
                std::ostringstream info;
                info << std::fixed << std::setprecision(2);
                info << "Wall Avg: " << avg_wall << " ms/tok";
                if (avg_wall > 0)
                    info << "  |  " << std::setprecision(1) << (1000.0 / avg_wall) << " tok/s";
                table << info.str() << "" << "" << fort::endr;
                table[1][0].set_cell_span(3);
            }

            // Header
            table << fort::header << "PHASE" << "AVG (ms)" << "%" << fort::endr;
            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);

            auto fmt_row = [&](const char *name, double ms)
            {
                std::ostringstream ms_str, pct_str;
                ms_str << std::fixed << std::setprecision(3) << ms;
                double pct = avg_wall > 0 ? (ms / avg_wall) * 100.0 : 0.0;
                pct_str << std::fixed << std::setprecision(1) << pct << "%";
                table << name << ms_str.str() << pct_str.str() << fort::endr;
            };

            fmt_row("Dispatch (wake workers)", avg_dispatch);
            fmt_row("Wait (device forward)", avg_wait);
            fmt_row("Gather logits", avg_gather);
            if (avg_other > 0.001)
                fmt_row("Other overhead", avg_other);

            // Separator + total
            table << fort::separator;
            {
                std::ostringstream ms_str;
                ms_str << std::fixed << std::setprecision(3) << avg_wall;
                table << "TOTAL" << ms_str.str() << "100.0%" << fort::endr;
            }

            std::cout << table.to_string() << std::flush;
            tp_decode_stats_.reset();
        }
    }

} // namespace llaminar2
