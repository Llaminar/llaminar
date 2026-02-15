/**
 * @file MultiDeviceOrchestrator.cpp
 * @brief Multi-device orchestrator implementation for LOCAL tensor parallelism
 * @author David Sanftenberg
 * @date January 2026
 *
 * Implements coordination of multiple DeviceGraphOrchestrator instances for LOCAL
 * tensor parallelism across multiple devices within a single MPI rank.
 *
 * Key features:
 * - Parallel forward pass execution across devices via std::async
 * - AllGather for combining partial logits from column-parallel LM head
 * - Unified snapshot/profiling API across all device runners
 */

#include "MultiDeviceOrchestrator.h"
#include "DeviceGraphOrchestrator.h"
#include "../../factory/InferenceRunnerFactory.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../collective/ILocalPPContext.h"
#include "../../../config/TensorParallelConfig.h"
#include "../../../interfaces/IModelContext.h"
#include "../../../loaders/ModelContext.h"
#include "../graph/SchemaFactoryRegistry.h" // Model-agnostic sharding config access
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../backends/p2p/DirectP2P.h" // DirectP2PEngine for BAR pre-init in cross-vendor PP
#include "../../../utils/Logger.h"
#include <algorithm>
#include <future>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Config Implementation
    // =========================================================================

    bool MultiDeviceOrchestrator::PPStageConfig::validate() const
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

    MultiDeviceOrchestrator::ParallelismMode
    MultiDeviceOrchestrator::Config::detectMode() const
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

    std::vector<int> MultiDeviceOrchestrator::Config::buildLayerBoundaries() const
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

    bool MultiDeviceOrchestrator::Config::validate() const
    {
        ParallelismMode effective = effectiveMode();

        if (effective == ParallelismMode::TP)
        {
            // TP mode validation
            if (devices.empty())
            {
                LOG_ERROR("MultiDeviceOrchestrator::Config: No devices specified for TP mode");
                return false;
            }

            // If weights are provided, must match device count
            if (!weights.empty() && weights.size() != devices.size())
            {
                LOG_ERROR("MultiDeviceOrchestrator::Config: Weights count (" << weights.size()
                                                                             << ") doesn't match device count (" << devices.size() << ")");
                return false;
            }

            // If weights are provided, must sum to approximately 1.0
            if (!weights.empty())
            {
                float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
                if (std::abs(sum - 1.0f) > 0.01f)
                {
                    LOG_ERROR("MultiDeviceOrchestrator::Config: Weights sum to " << sum << ", expected 1.0");
                    return false;
                }
            }
        }
        else
        {
            // PP or TP_PP mode validation
            if (pp_stages.empty())
            {
                LOG_ERROR("MultiDeviceOrchestrator::Config: No PP stages specified for PP mode");
                return false;
            }

            // Validate each stage
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (!pp_stages[i].validate())
                {
                    LOG_ERROR("MultiDeviceOrchestrator::Config: PP stage " << i << " validation failed");
                    return false;
                }
            }

            // Check layer continuity (no gaps)
            int expected_first = 0;
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (pp_stages[i].first_layer != expected_first)
                {
                    LOG_ERROR("MultiDeviceOrchestrator::Config: PP stage " << i
                                                                           << " first_layer=" << pp_stages[i].first_layer
                                                                           << " but expected " << expected_first << " (gap in layers)");
                    return false;
                }
                expected_first = pp_stages[i].last_layer;
            }

            // First stage should have embedding, last should have LM head
            if (!pp_stages.front().has_embedding)
            {
                LOG_WARN("MultiDeviceOrchestrator::Config: First PP stage doesn't have embedding flag set");
            }
            if (!pp_stages.back().has_lm_head)
            {
                LOG_WARN("MultiDeviceOrchestrator::Config: Last PP stage doesn't have lm_head flag set");
            }
        }

        return true;
    }

    std::vector<float> MultiDeviceOrchestrator::Config::getNormalizedWeights() const
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
    // Factory Methods
    // =========================================================================

    std::unique_ptr<MultiDeviceOrchestrator> MultiDeviceOrchestrator::createForTest(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
    {
        // Use the private constructor
        return std::unique_ptr<MultiDeviceOrchestrator>(
            new MultiDeviceOrchestrator(
                std::move(model_ctx),
                std::move(device_runners),
                std::move(tp_ctx),
                config));
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    MultiDeviceOrchestrator::MultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)), config_(config)
    {
        if (!config_.validate())
        {
            throw std::invalid_argument("Invalid MultiDeviceOrchestrator configuration");
        }

        // Determine effective parallelism mode
        mode_ = config_.effectiveMode();

        LOG_INFO("MultiDeviceOrchestrator: Creating with mode="
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

            // Validate TP degree
            if (tp_ctx_->degree() < 2)
            {
                LOG_WARN("MultiDeviceOrchestrator: TP degree is " << tp_ctx_->degree()
                                                                  << ", multi-device orchestration may not be beneficial");
            }

            // Initialize device runners for TP mode
            initializeDeviceRunners();

            LOG_INFO("MultiDeviceOrchestrator: Initialized TP mode with " << device_runners_.size() << " device runners");
        }
        else
        {
            // PP or TP_PP mode - create PP context and stage runners
            initializePPDeviceRunners();
            initializePPContext();

            LOG_INFO("MultiDeviceOrchestrator: Initialized PP mode with " << config_.pp_stages.size() << " stages");
        }
    }

    MultiDeviceOrchestrator::MultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)), tp_ctx_(std::move(tp_ctx)), config_(config),
          mode_(ParallelismMode::TP) // Pre-existing TP context implies TP mode
    {
        if (!tp_ctx_)
        {
            throw std::invalid_argument("tp_ctx cannot be null");
        }

        // Validate TP degree
        if (tp_ctx_->degree() < 2)
        {
            LOG_WARN("MultiDeviceOrchestrator: TP degree is " << tp_ctx_->degree()
                                                              << ", multi-device orchestration may not be beneficial");
        }

        LOG_INFO("MultiDeviceOrchestrator: Creating with pre-existing TP context, "
                 << tp_ctx_->degree() << " devices");

        // Initialize device runners
        initializeDeviceRunners();

        LOG_INFO("MultiDeviceOrchestrator: Initialized with " << device_runners_.size() << " device runners");
    }

    // Private constructor for createForTest
    MultiDeviceOrchestrator::MultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)),
          tp_ctx_(std::move(tp_ctx)),
          mode_(ParallelismMode::TP), // Test factory currently only supports TP mode
          device_runners_(std::move(device_runners)),
          config_(config)
    {
        LOG_DEBUG("MultiDeviceOrchestrator: Created via createForTest with "
                  << device_runners_.size() << " injected device runners");
    }

    MultiDeviceOrchestrator::~MultiDeviceOrchestrator() = default;

    // Move operations
    MultiDeviceOrchestrator::MultiDeviceOrchestrator(MultiDeviceOrchestrator &&) noexcept = default;
    MultiDeviceOrchestrator &MultiDeviceOrchestrator::operator=(MultiDeviceOrchestrator &&) noexcept = default;

    // =========================================================================
    // Private Methods
    // =========================================================================

    void MultiDeviceOrchestrator::initializeDeviceRunners()
    {
        if (!tp_ctx_)
        {
            throw std::runtime_error("Cannot initialize device runners: tp_ctx_ is null");
        }

        const auto &devices = tp_ctx_->devices();
        device_runners_.reserve(devices.size());

        LOG_DEBUG("MultiDeviceOrchestrator: Initializing " << devices.size() << " device runners");

        // =====================================================================
        // BUILD TENSORPARALLELCONFIG FOR LOCAL TP WEIGHT SHARDING
        // =====================================================================
        // This enables WeightManager to slice weights by DeviceId instead of
        // falling back to REPLICATED mode for world_size==1.
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

                // Estimate d_ff if not available (common SwiGLU ratio)
                if (d_ff <= 0)
                {
                    d_ff = model_ctx_->embeddingLength() * 4;
                    LOG_WARN("MultiDeviceOrchestrator: feedForwardLength() unavailable, using estimate: " << d_ff);
                }

                auto tp_config = std::make_shared<TensorParallelConfig>(
                    TensorParallelConfig::fromLocalTPContext(
                        *tp_ctx_, n_heads, n_kv_heads, d_ff, vocab_size));

                weight_mgr->setTensorParallelConfig(tp_config);

                // =====================================================================
                // SET WEIGHT SHARDING CONFIG FOR TP SLICING MODE DETECTION
                // =====================================================================
                // WeightManager needs the sharding config to determine which weights
                // should be column-parallel vs row-parallel vs replicated.
                // Without this, determineShardingMode() throws an exception.
                // Use SchemaFactoryRegistry for model-agnostic architecture lookup.
                // =====================================================================
                const std::string arch = model_ctx_->architecture();
                auto sharding_config = SchemaFactoryRegistry::getWeightShardingConfig(arch);
                weight_mgr->setWeightShardingConfig(sharding_config);
                LOG_DEBUG("MultiDeviceOrchestrator: Set WeightShardingConfig for TP slicing mode detection"
                          << " (architecture=" << arch << ")");

                LOG_INFO("MultiDeviceOrchestrator: Set TensorParallelConfig for LOCAL TP ("
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
                        LOG_INFO("MultiDeviceOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") assignment:"
                                                                    << " head_start=" << assignment.head_start
                                                                    << " head_count=" << assignment.head_count
                                                                    << " kv_head_start=" << assignment.kv_head_start
                                                                    << " kv_head_count=" << assignment.kv_head_count
                                                                    << " d_ff_start=" << assignment.d_ff_start
                                                                    << " d_ff_count=" << assignment.d_ff_count);
                    }
                    catch (const std::out_of_range &e)
                    {
                        LOG_WARN("MultiDeviceOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") NOT in TensorParallelConfig!");
                    }
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
                LOG_INFO("MultiDeviceOrchestrator: Reserved collective temp buffer: "
                         << buffer_with_margin << " bytes ("
                         << "max_seq_len=" << config_.max_seq_len
                         << ", hidden_size=" << hidden_size
                         << ", precision=" << activationPrecisionToString(config_.activation_precision) << ")");
            }
            else
            {
                LOG_WARN("MultiDeviceOrchestrator: Failed to reserve collective temp buffer");
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

            // Pre-load all weights for all devices
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr)
            {
                LOG_INFO("MultiDeviceOrchestrator: Pre-loading weights for " << device_ids.size() << " devices");
                if (!weight_mgr->preloadForDevices(device_ids))
                {
                    LOG_WARN("MultiDeviceOrchestrator: Weight preloading failed, may encounter race conditions");
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
                                                 LOG_DEBUG("MultiDeviceOrchestrator: Creating runner for device " << device_idx
                                                                                                                  << " (" << device_id.toString() << ")");

                                                 // Build InferenceRunnerConfig for LOCAL TP
                                                 InferenceRunnerConfig runner_config;
                                                 runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
                                                 runner_config.batch_size = config_.batch_size;
                                                 runner_config.activation_precision = config_.activation_precision;
                                                 runner_config.kv_cache_scale = config_.kv_cache_scale;
                                                 runner_config.kv_cache_precision = config_.kv_cache_precision;
                                                 runner_config.use_mapped_memory = config_.use_mapped_memory;
                                                 runner_config.use_bar_backed_hidden = config_.use_bar_backed_hidden;

                                                 // Set LOCAL TP parameters
                                                 runner_config.local_tp_ctx = tp_ctx_.get();
                                                 runner_config.local_tp_device_index = device_idx;

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

            // Cast to DeviceGraphOrchestrator
            auto *device_orchestrator = dynamic_cast<DeviceGraphOrchestrator *>(result.runner.get());
            if (!device_orchestrator)
            {
                throw std::runtime_error("Inference runner is not a DeviceGraphOrchestrator for device " +
                                         std::to_string(result.device_idx));
            }

            // CRITICAL: For nested TP-in-PP, set PP stage config on the DeviceGraphOrchestrator
            // so it builds a partial graph instead of a full graph. Without this, the TP devices
            // would include LM_HEAD even when this PP stage doesn't own it.
            if (config_.nested_pp_stage_config.has_value())
            {
                device_orchestrator->setPPStageConfig(config_.nested_pp_stage_config.value());
                LOG_DEBUG("MultiDeviceOrchestrator: Set PP stage config on device " << result.device_idx
                                                                                    << " (layers " << config_.nested_pp_stage_config->first_layer
                                                                                    << "-" << config_.nested_pp_stage_config->last_layer
                                                                                    << " has_lm_head=" << config_.nested_pp_stage_config->has_lm_head << ")");
            }

            // Transfer ownership
            result.runner.release();
            device_runners_.push_back(std::unique_ptr<DeviceGraphOrchestrator>(device_orchestrator));

            LOG_DEBUG("MultiDeviceOrchestrator: Successfully created runner for device " << result.device_idx);
        }

        // Allocate combined logits buffer if we have vocab size info
        // For column-parallel LM head, this needs to be [batch_size * max_seq_len, vocab_size]
        // to hold the gathered logits from all devices
        if (model_ctx_ && device_runners_.size() > 0)
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                // Calculate max tokens = batch_size * max_seq_len
                size_t max_tokens = static_cast<size_t>(config_.batch_size) *
                                    static_cast<size_t>(config_.max_seq_len);
                combined_logits_ = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{max_tokens, static_cast<size_t>(vocab)});
                LOG_DEBUG("MultiDeviceOrchestrator: Allocated combined logits buffer ["
                          << max_tokens << ", " << vocab << "]");
            }
        }
    }

    // =========================================================================
    // PP Mode Initialization
    // =========================================================================

    void MultiDeviceOrchestrator::initializePPDeviceRunners()
    {
        if (config_.pp_stages.empty())
        {
            throw std::runtime_error("Cannot initialize PP device runners: no PP stages configured");
        }

        LOG_DEBUG("MultiDeviceOrchestrator: Initializing " << config_.pp_stages.size() << " PP stage runners");

        // Get model path for creating stage-specific model contexts
        const std::string &model_path = model_ctx_->path();
        const size_t num_stages = config_.pp_stages.size();

        // =========================================================================
        // Cross-Vendor PP Detection
        // =========================================================================
        // Check if any PP transfer is cross-vendor (ROCm→CUDA or CUDA→ROCm).
        // If so, the source stage's hidden state tensor needs BAR-backed allocation
        // to enable zero-copy PCIe BAR transfers.
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

        // Pre-compute which stages need BAR-backed hidden state
        std::vector<bool> needs_bar_backed(num_stages, false);
        for (size_t i = 0; i + 1 < num_stages; ++i)
        {
            if (isCrossVendorTransfer(config_.pp_stages[i], config_.pp_stages[i + 1]))
            {
                // Source stage outputs to cross-vendor, needs BAR-backed hidden
                needs_bar_backed[i] = true;
                LOG_INFO("MultiDeviceOrchestrator: PP stage " << i
                                                              << " outputs to cross-vendor stage " << (i + 1)
                                                              << " - will use BAR-backed hidden state");
            }
        }

        // =========================================================================
        // Pre-initialize PCIe BAR for Cross-Vendor PP
        // =========================================================================
        // If any stage needs BAR-backed hidden state, we must initialize the
        // DirectP2PEngine's BAR mapping NOW, before creating device runners.
        // Otherwise, when DeviceGraphOrchestrator calls initializeInferenceState(),
        // isPCIeBarActive() will return false and it will fall back to standard
        // allocation (which will fail during cross-vendor transfer).
        // =========================================================================
        for (size_t i = 0; i < num_stages; ++i)
        {
            if (needs_bar_backed[i] && i + 1 < num_stages)
            {
                const auto &src_stage = config_.pp_stages[i];
                const auto &dst_stage = config_.pp_stages[i + 1];

                if (!src_stage.stage_devices.empty() && !dst_stage.stage_devices.empty())
                {
                    DeviceId src_dev = src_stage.stage_devices[0].toLocalDeviceId();
                    DeviceId dst_dev = dst_stage.stage_devices[0].toLocalDeviceId();

                    DeviceId cuda_dev = src_dev.is_cuda() ? src_dev : dst_dev;
                    DeviceId rocm_dev = src_dev.is_rocm() ? src_dev : dst_dev;

                    auto p2p = DirectP2PEngine::getSharedInstance();
                    if (!p2p->isPCIeBarActive())
                    {
                        LOG_INFO("MultiDeviceOrchestrator: Pre-initializing PCIe BAR for cross-vendor PP "
                                 << "(CUDA:" << cuda_dev.cuda_ordinal() << " <-> ROCm:" << rocm_dev.rocm_ordinal() << ")");

                        constexpr size_t bar_map_size = 1024 * 1024 * 1024; // 1GB BAR region
                        if (!p2p->initializePCIeBar(cuda_dev, rocm_dev, 0, bar_map_size))
                        {
                            LOG_ERROR("MultiDeviceOrchestrator: Failed to pre-initialize PCIe BAR. "
                                      "Cross-vendor PP transfer will fail.");
                            throw std::runtime_error("Failed to initialize PCIe BAR for cross-vendor PP");
                        }

                        LOG_INFO("MultiDeviceOrchestrator: PCIe BAR pre-initialized successfully");
                        break; // Only need to initialize once
                    }
                }
            }
        }

        pp_stage_runners_.reserve(num_stages);

        for (size_t stage_idx = 0; stage_idx < num_stages; ++stage_idx)
        {
            const auto &stage_config = config_.pp_stages[stage_idx];

            LOG_DEBUG("MultiDeviceOrchestrator: Creating PP stage " << stage_idx
                                                                    << " [layers " << stage_config.first_layer
                                                                    << "-" << stage_config.last_layer << ")"
                                                                    << " has_embedding=" << stage_config.has_embedding
                                                                    << " has_lm_head=" << stage_config.has_lm_head
                                                                    << " needs_bar_backed=" << needs_bar_backed[stage_idx]);

            // Validate stage has at least one device
            if (stage_config.stage_devices.empty())
            {
                throw std::runtime_error("PP stage " + std::to_string(stage_idx) + " has no devices configured");
            }

            // Get primary device for this stage
            DeviceId primary_device = stage_config.stage_devices[0].toLocalDeviceId();

            // =====================================================================
            // Create stage-specific ModelContext with layer-partitioned weights
            // This only loads weights for this stage's layer range, reducing memory
            // =====================================================================
            auto stage_ctx = ModelContext::createForPPStage(
                model_path,
                stage_config.first_layer,
                stage_config.last_layer,
                stage_config.has_embedding,
                stage_config.has_lm_head);

            if (!stage_ctx)
            {
                throw std::runtime_error("Failed to create ModelContext for PP stage " +
                                         std::to_string(stage_idx));
            }

            // =====================================================================
            // Build InferenceRunnerConfig for this stage
            // =====================================================================
            InferenceRunnerConfig runner_config;
            runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
            runner_config.batch_size = config_.batch_size;
            runner_config.activation_precision = config_.activation_precision;
            runner_config.kv_cache_scale = config_.kv_cache_scale;
            runner_config.kv_cache_precision = config_.kv_cache_precision;
            runner_config.use_mapped_memory = config_.use_mapped_memory;

            // =====================================================================
            // Build FactoryPPStageConfig for the createPPStageRunner factory
            // =====================================================================
            FactoryPPStageConfig factory_pp_config;
            factory_pp_config.first_layer = stage_config.first_layer;
            factory_pp_config.last_layer = stage_config.last_layer;
            factory_pp_config.has_embedding = stage_config.has_embedding;
            factory_pp_config.has_lm_head = stage_config.has_lm_head;
            factory_pp_config.use_bar_backed_hidden = needs_bar_backed[stage_idx] || stage_config.requires_bar_backed_hidden;

            // =====================================================================
            // Handle single-device vs TP-domain stages
            // =====================================================================
            if (stage_config.isTPDomain())
            {
                // =====================================================================
                // TP Domain Stage: Create nested MultiDeviceOrchestrator in TP mode
                // =====================================================================
                LOG_INFO("MultiDeviceOrchestrator: PP stage " << stage_idx
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
                nested_config.kv_cache_scale = config_.kv_cache_scale;
                nested_config.kv_cache_precision = config_.kv_cache_precision;
                nested_config.use_mapped_memory = config_.use_mapped_memory;
                nested_config.use_bar_backed_hidden = needs_bar_backed[stage_idx] || stage_config.requires_bar_backed_hidden;

                // CRITICAL: Pass PP stage config to nested TP MDO so its DeviceGraphOrchestrators
                // build partial graphs instead of full graphs. Without this, the TP devices would
                // build LM_HEAD stages even though this PP stage doesn't own LM_HEAD.
                nested_config.nested_pp_stage_config = factory_pp_config;

                // Create the nested MultiDeviceOrchestrator
                // Note: stage_ctx contains weights only for this stage's layers
                auto nested_mdo = std::make_unique<MultiDeviceOrchestrator>(stage_ctx, nested_config);

                if (!nested_mdo)
                {
                    throw std::runtime_error("Failed to create nested MultiDeviceOrchestrator for PP stage " +
                                             std::to_string(stage_idx));
                }

                pp_stage_runners_.push_back(std::move(nested_mdo));

                LOG_INFO("MultiDeviceOrchestrator: Created TP domain PP stage " << stage_idx
                                                                                << " with " << stage_config.stage_devices.size() << " devices"
                                                                                << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
            else
            {
                // =====================================================================
                // Single Device Stage: Use existing factory path
                // =====================================================================
                auto runner = createPPStageRunner(stage_ctx, primary_device, factory_pp_config, runner_config);

                if (!runner)
                {
                    throw std::runtime_error("Failed to create PP stage runner for stage " +
                                             std::to_string(stage_idx) + " on device " +
                                             primary_device.to_string());
                }

                pp_stage_runners_.push_back(std::move(runner));

                LOG_INFO("MultiDeviceOrchestrator: Created PP stage " << stage_idx
                                                                      << " runner on device " << primary_device.to_string()
                                                                      << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
        }

        LOG_INFO("MultiDeviceOrchestrator: Successfully initialized " << pp_stage_runners_.size() << " PP stage runners");

        // Note: PP context initialization is done by the caller (constructor)
        // after this method returns, to avoid double-initialization
    }

    void MultiDeviceOrchestrator::initializePPContext()
    {
        if (pp_stage_runners_.empty())
        {
            // PP stage runners not initialized - this is expected for Phase 1
            LOG_DEBUG("MultiDeviceOrchestrator::initializePPContext: No PP stage runners yet");
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

        LOG_DEBUG("MultiDeviceOrchestrator::initializePPContext: Creating LocalPPContext with "
                  << pp_config.numStages() << " stages");

        // Create the LocalPPContext using the factory function
        pp_ctx_ = createLocalPPContext(pp_config);

        if (!pp_ctx_)
        {
            throw std::runtime_error("Failed to create LocalPPContext");
        }

        LOG_INFO("MultiDeviceOrchestrator: Initialized LocalPPContext with " << pp_config.numStages() << " stages");
    }

    bool MultiDeviceOrchestrator::gatherLogits(size_t seq_len)
    {
        if (!combined_logits_ || device_runners_.empty())
        {
            return false;
        }

        // Single device or no TP context - just copy from primary device
        if (!tp_ctx_ || device_runners_.size() == 1)
        {
            const float *primary_logits = device_runners_[0]->logits();
            if (primary_logits)
            {
                int vocab = vocab_size();
                // For decode, seq_len=1 so we copy vocab elements
                // For prefill, seq_len * vocab elements
                size_t copy_size = seq_len * static_cast<size_t>(vocab);
                std::memcpy(combined_logits_->mutable_data(), primary_logits,
                            copy_size * sizeof(float));
                last_gathered_logits_size_ = copy_size;
            }
            return true;
        }

        // Check if column-parallel LM head is enabled by checking if any device
        // has logits_local allocated
        bool has_column_parallel_lm_head = false;
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                const auto &state = runner->inferenceState();
                if (state.logits_local)
                {
                    has_column_parallel_lm_head = true;
                    break;
                }
            }
        }

        if (!has_column_parallel_lm_head)
        {
            // LM head is replicated, not sharded - use primary device's full logits
            const float *primary_logits = device_runners_[0]->logits();
            if (primary_logits)
            {
                int vocab = vocab_size();
                size_t copy_size = seq_len * static_cast<size_t>(vocab);
                std::memcpy(combined_logits_->mutable_data(), primary_logits,
                            copy_size * sizeof(float));
                last_gathered_logits_size_ = copy_size;
            }
            return true;
        }

        // Column-parallel LM head: each device has logits_local [seq_len, vocab_local]
        // We need to gather along the vocab dimension (axis=1), producing [seq_len, vocab_total]
        //
        // For each row (token position), concatenate vocab_local slices from all devices:
        //   output[row, 0:vocab_local_0] = device_0_logits[row, :]
        //   output[row, vocab_local_0:vocab_local_0+vocab_local_1] = device_1_logits[row, :]
        //   etc.

        // Collect info from all devices
        std::vector<const float *> device_data;
        std::vector<size_t> vocab_locals;

        for (const auto &runner : device_runners_)
        {
            if (!runner)
            {
                LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: null device runner");
                return false;
            }

            const auto &state = runner->inferenceState();
            if (!state.logits_local)
            {
                LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: device missing logits_local");
                return false;
            }

            const auto &shape = state.logits_local->shape();
            if (shape.size() < 2)
            {
                LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: logits_local must be 2D");
                return false;
            }

            // Use the passed-in seq_len instead of reading from tensor shape
            // (tensor shape is pre-allocated for max_seq_len, but for decode we only want 1 row)
            size_t this_vocab_local = shape[1];

            device_data.push_back(state.logits_local->data());
            vocab_locals.push_back(this_vocab_local);
        }

        // Calculate total vocab and validate output buffer
        size_t total_vocab = 0;
        for (size_t vl : vocab_locals)
        {
            total_vocab += vl;
        }

        size_t expected_output_size = seq_len * total_vocab;
        if (combined_logits_->numel() < expected_output_size)
        {
            LOG_ERROR("MultiDeviceOrchestrator::gatherLogits: output buffer too small. "
                      << "Need " << expected_output_size << ", have " << combined_logits_->numel());
            return false;
        }

        // Perform row-by-row column gathering
        float *output = combined_logits_->mutable_data();

        // Debug: Log first elements from each device for decode debugging
        if (seq_len == 1 && device_data.size() >= 2)
        {
            LOG_DEBUG("MultiDeviceOrchestrator::gatherLogits: DECODE DEBUG" << " device0_first=" << device_data[0][0] << " device0_max_idx=" << [&]()
                      {
                          size_t max_idx = 0;
                          float max_val = device_data[0][0];
                          for (size_t i = 1; i < vocab_locals[0]; ++i) {
                              if (device_data[0][i] > max_val) {
                                  max_val = device_data[0][i];
                                  max_idx = i;
                              }
                          }
                          return std::to_string(max_idx) + " (val=" + std::to_string(max_val) + ")"; }() << " device1_first=" << device_data[1][0] << " device1_max_idx=" << [&]()
                      {
                          size_t max_idx = 0;
                          float max_val = device_data[1][0];
                          for (size_t i = 1; i < vocab_locals[1]; ++i) {
                              if (device_data[1][i] > max_val) {
                                  max_val = device_data[1][i];
                                  max_idx = i;
                              }
                          }
                          return std::to_string(max_idx) + " (val=" + std::to_string(max_val) + ")"; }());
        }

        for (size_t row = 0; row < seq_len; ++row)
        {
            size_t col_offset = 0;
            for (size_t dev = 0; dev < device_data.size(); ++dev)
            {
                const float *src = device_data[dev] + row * vocab_locals[dev];
                float *dst = output + row * total_vocab + col_offset;
                std::memcpy(dst, src, vocab_locals[dev] * sizeof(float));
                col_offset += vocab_locals[dev];
            }
        }

        // Store the actual gathered size for getSnapshot()
        last_gathered_logits_size_ = expected_output_size;

        LOG_DEBUG("MultiDeviceOrchestrator::gatherLogits: gathered column-parallel logits "
                  << "[" << seq_len << ", " << total_vocab << "] from " << device_data.size() << " devices");

        return true;
    }

    void MultiDeviceOrchestrator::aggregateStats() const
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

                    // Merge stage times
                    for (const auto &[stage_name, time_ms] : stats->stage_times_ms)
                    {
                        aggregated_stats_->stage_times_ms[stage_name] += time_ms;
                    }
                }
            }
        }

        // Average the times (since devices run in parallel)
        if (!device_runners_.empty())
        {
            size_t count = device_runners_.size();
            aggregated_stats_->total_time_ms /= static_cast<double>(count);
            aggregated_stats_->total_execute_ms /= static_cast<double>(count);
            for (auto &[stage_name, time_ms] : aggregated_stats_->stage_times_ms)
            {
                time_ms /= static_cast<double>(count);
            }
        }

        stats_dirty_ = false;
    }

    // =========================================================================
    // IInferenceRunner Interface Implementation
    // =========================================================================

    bool MultiDeviceOrchestrator::forward(const int *tokens, int seq_len)
    {
        // Dispatch to appropriate implementation based on parallelism mode
        switch (mode_)
        {
        case ParallelismMode::TP:
            return forwardTP(tokens, seq_len);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            // PP and TP_PP both use sequential stage execution
            // The difference is that TP_PP stages may be nested MDOs (TP domains)
            // but forwardPP() works through IInferenceRunner interface regardless
            return forwardPP(tokens, seq_len);
        default:
            LOG_ERROR("MultiDeviceOrchestrator::forward: Unknown parallelism mode");
            return false;
        }
    }

    // =========================================================================
    // TP Mode Forward Implementation (existing parallel execution)
    // =========================================================================
    bool MultiDeviceOrchestrator::forwardTP(const int *tokens, int seq_len)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("MultiDeviceOrchestrator::forwardTP: No device runners available");
            return false;
        }

        LOG_DEBUG("MultiDeviceOrchestrator::forwardTP: seq_len=" << seq_len
                                                                 << ", devices=" << device_runners_.size());

        // Launch parallel forward passes on all devices
        std::vector<std::future<bool>> futures;
        futures.reserve(device_runners_.size());

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            auto &runner = device_runners_[i];
            if (runner)
            {
                // Cast to IInferenceRunner* to disambiguate the forward() call
                // DeviceGraphOrchestrator has both forward(tokens, seq_len) -> bool
                // and forward(tokens, seq_len, batch_size=1) -> const float*
                IInferenceRunner *runner_iface = runner.get();
                futures.push_back(std::async(std::launch::async,
                                             [runner_iface, tokens, seq_len]() -> bool
                                             {
                                                 return runner_iface->forward(tokens, seq_len);
                                             }));
            }
        }

        // Wait for all to complete and check results
        // IMPORTANT: Store the FIRST exception so we can re-throw it with the real error message.
        // When one device throws (e.g., VerificationFailure with NaN/Inf), it can cause CUDA
        // context destruction, which then makes other devices fail with misleading "context is
        // destroyed" errors. We want to surface the original root cause exception.
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
                    LOG_ERROR("MultiDeviceOrchestrator::forwardTP: Device " << i << " forward failed");
                    all_success = false;
                }
            }
            catch (const std::exception &e)
            {
                all_success = false;

                // Check if this is a secondary "context destroyed" error vs the real root cause
                std::string error_msg = e.what();
                bool is_context_destroyed = (error_msg.find("context is destroyed") != std::string::npos ||
                                             error_msg.find("context destroyed") != std::string::npos ||
                                             error_msg.find("error 709") != std::string::npos);

                if (!first_exception)
                {
                    // This is the first exception - store it
                    first_exception = std::current_exception();
                    first_exception_device = i;
                    LOG_ERROR("MultiDeviceOrchestrator::forwardTP: Device " << i
                                                                            << " threw PRIMARY exception: " << error_msg);
                }
                else if (!is_context_destroyed)
                {
                    // This is a substantive error (not just context cleanup failure)
                    // Replace the stored exception if the first one was a context error
                    try
                    {
                        std::rethrow_exception(first_exception);
                    }
                    catch (const std::exception &first_e)
                    {
                        std::string first_msg = first_e.what();
                        bool first_is_context_destroyed =
                            (first_msg.find("context is destroyed") != std::string::npos ||
                             first_msg.find("context destroyed") != std::string::npos ||
                             first_msg.find("error 709") != std::string::npos);

                        if (first_is_context_destroyed)
                        {
                            // Replace context error with the real error
                            LOG_WARN("MultiDeviceOrchestrator::forwardTP: Replacing secondary context error "
                                     "with primary error from device "
                                     << i);
                            first_exception = std::current_exception();
                            first_exception_device = i;
                        }
                    }
                    LOG_ERROR("MultiDeviceOrchestrator::forwardTP: Device " << i
                                                                            << " threw exception: " << error_msg);
                }
                else
                {
                    // Secondary context error - log but don't replace the primary exception
                    LOG_WARN("MultiDeviceOrchestrator::forwardTP: Device " << i
                                                                           << " threw SECONDARY exception (likely due to primary failure): "
                                                                           << error_msg);
                }
            }
        }

        // If we captured an exception, re-throw it with full context
        if (first_exception)
        {
            LOG_ERROR("MultiDeviceOrchestrator::forwardTP: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (all_success)
        {
            // Gather logits from all devices
            // Pass seq_len so gatherLogits knows how many rows to gather
            // (logits_local buffer is pre-allocated for max_seq_len)
            if (!gatherLogits(static_cast<size_t>(seq_len)))
            {
                LOG_ERROR("MultiDeviceOrchestrator::forwardTP: Failed to gather logits");
                all_success = false;
            }

            // Update position tracking
            current_position_ += seq_len;
            current_padded_seq_len_ = seq_len;
            stats_dirty_ = true;
        }

        return all_success;
    }

    // =========================================================================
    // PP Mode Forward Implementation (sequential pipeline execution)
    // =========================================================================
    bool MultiDeviceOrchestrator::forwardPP(const int *tokens, int seq_len)
    {
        if (pp_stage_runners_.empty())
        {
            LOG_ERROR("MultiDeviceOrchestrator::forwardPP: No PP stage runners available");
            return false;
        }

        if (!pp_ctx_)
        {
            LOG_ERROR("MultiDeviceOrchestrator::forwardPP: No LocalPPContext available for transfers");
            return false;
        }

        const size_t num_stages = pp_stage_runners_.size();

        LOG_DEBUG("MultiDeviceOrchestrator::forwardPP: seq_len=" << seq_len
                                                                 << " num_stages=" << num_stages);

        // =====================================================================
        // Stage 0: Embedding + first layers (receives tokens as input)
        // =====================================================================
        auto &stage0_runner = pp_stage_runners_[0];
        if (!stage0_runner)
        {
            LOG_ERROR("MultiDeviceOrchestrator::forwardPP: Stage 0 runner is null");
            return false;
        }

        LOG_DEBUG("MultiDeviceOrchestrator::forwardPP: Executing stage 0 (embedding)");
        if (!stage0_runner->forward(tokens, seq_len))
        {
            LOG_ERROR("MultiDeviceOrchestrator::forwardPP: Stage 0 forward failed");
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
                LOG_ERROR("MultiDeviceOrchestrator::forwardPP: Stage " << stage_idx << " runner is null");
                return false;
            }

            // Get hidden state from previous stage
            // DeviceGraphOrchestrator stores hidden state in InferenceState
            TensorBase *hidden_state = prev_runner->getHiddenState();
            if (!hidden_state)
            {
                LOG_ERROR("MultiDeviceOrchestrator::forwardPP: Stage " << (stage_idx - 1)
                                                                       << " has no hidden state to transfer");
                return false;
            }

            // Transfer activations from previous stage to current stage
            LOG_DEBUG("MultiDeviceOrchestrator::forwardPP: Transferring hidden state from stage "
                      << (stage_idx - 1) << " to stage " << stage_idx);

            if (!pp_ctx_->transfer(hidden_state, static_cast<int>(stage_idx - 1),
                                   static_cast<int>(stage_idx)))
            {
                LOG_ERROR("MultiDeviceOrchestrator::forwardPP: Transfer from stage "
                          << (stage_idx - 1) << " to stage " << stage_idx << " failed");
                return false;
            }

            // Set the hidden state as input for current stage
            // This makes the stage runner skip embedding and use the transferred hidden state
            curr_runner->setHiddenState(hidden_state);

            LOG_DEBUG("MultiDeviceOrchestrator::forwardPP: Executing stage " << stage_idx);

            // Call forward with nullptr tokens - the stage will use setHiddenState input
            // and skip embedding since has_embedding=false for non-first stages
            if (!curr_runner->forward(nullptr, seq_len))
            {
                LOG_ERROR("MultiDeviceOrchestrator::forwardPP: Stage " << stage_idx << " forward failed");
                return false;
            }

            // Clear hidden state input for clean state on next forward
            curr_runner->clearHiddenStateInput();
        }

        // =====================================================================
        // Copy logits from last stage to combined buffer
        // =====================================================================
        copyLogitsFromStage(static_cast<int>(num_stages - 1));

        // =====================================================================
        // Update position tracking for PP mode
        // Each PP stage runner updates its own position internally, but we also
        // need to update current_position_ for consistency with TP mode and
        // get_position() queries.
        // =====================================================================
        current_position_ += seq_len;

        LOG_DEBUG("MultiDeviceOrchestrator::forwardPP: Complete, all " << num_stages << " stages executed"
                                                                       << ", position now " << current_position_);
        return true;
    }

    void MultiDeviceOrchestrator::copyLogitsFromStage(int stage_idx)
    {
        if (stage_idx < 0 || static_cast<size_t>(stage_idx) >= pp_stage_runners_.size())
        {
            LOG_ERROR("MultiDeviceOrchestrator::copyLogitsFromStage: Invalid stage index " << stage_idx);
            return;
        }

        const auto &stage_runner = pp_stage_runners_[stage_idx];
        if (!stage_runner)
        {
            LOG_ERROR("MultiDeviceOrchestrator::copyLogitsFromStage: Stage " << stage_idx << " runner is null");
            return;
        }

        const float *stage_logits = stage_runner->logits();
        if (!stage_logits)
        {
            LOG_DEBUG("MultiDeviceOrchestrator::copyLogitsFromStage: Stage " << stage_idx
                                                                             << " has no logits (may not have LM head)");
            return;
        }

        // Get logits shape from stage runner
        int vocab = stage_runner->vocab_size();
        if (vocab <= 0)
        {
            LOG_ERROR("MultiDeviceOrchestrator::copyLogitsFromStage: Invalid vocab_size from stage " << stage_idx);
            return;
        }

        // Ensure combined_logits_ is allocated
        if (!combined_logits_)
        {
            // Allocate based on config and vocab
            size_t max_tokens = static_cast<size_t>(config_.batch_size) * static_cast<size_t>(config_.max_seq_len);
            combined_logits_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{max_tokens, static_cast<size_t>(vocab)});
            LOG_DEBUG("MultiDeviceOrchestrator::copyLogitsFromStage: Allocated combined logits buffer ["
                      << max_tokens << ", " << vocab << "]");
        }

        // Copy logits from stage runner
        // For decode mode (seq_len=1), copy vocab elements
        // For prefill mode, copy seq_len * vocab elements
        size_t copy_elements = last_gathered_logits_size_ > 0 ? last_gathered_logits_size_ : static_cast<size_t>(vocab);
        std::memcpy(combined_logits_->mutable_data(), stage_logits, copy_elements * sizeof(float));

        LOG_DEBUG("MultiDeviceOrchestrator::copyLogitsFromStage: Copied " << copy_elements
                                                                          << " elements from stage " << stage_idx);
    }

    const float *MultiDeviceOrchestrator::logits() const
    {
        // For PP mode: return combined logits (copied from final stage)
        if (mode_ == ParallelismMode::PP || mode_ == ParallelismMode::TP_PP)
        {
            if (combined_logits_)
            {
                return combined_logits_->data();
            }
            // Fallback: try to get from final PP stage
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->logits();
            }
            return nullptr;
        }

        // For TP mode: return combined logits if available (multi-device)
        if (combined_logits_ && device_runners_.size() > 1)
        {
            return combined_logits_->data();
        }

        // For single device, return primary device's logits
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->logits();
        }

        return nullptr;
    }

    bool MultiDeviceOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("MultiDeviceOrchestrator::forward_batch: No device runners available");
            return false;
        }

        LOG_DEBUG("MultiDeviceOrchestrator::forward_batch: batch_size=" << token_batches.size()
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
                    LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Device " << i << " forward_batch failed");
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
                    LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Device " << i
                                                                                << " threw PRIMARY exception: " << error_msg);
                }
                else if (is_context_destroyed)
                {
                    LOG_WARN("MultiDeviceOrchestrator::forward_batch: Device " << i
                                                                               << " threw SECONDARY exception (context destroyed): " << error_msg);
                }
                else
                {
                    LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Device " << i
                                                                                << " threw exception: " << error_msg);
                }
            }
        }

        if (first_exception)
        {
            LOG_ERROR("MultiDeviceOrchestrator::forward_batch: Re-throwing primary exception from device "
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

    const float *MultiDeviceOrchestrator::getLogits(int seq_idx) const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getLogits(seq_idx);
        }
        return nullptr;
    }

    int MultiDeviceOrchestrator::batch_size() const
    {
        return current_batch_size_;
    }

    int MultiDeviceOrchestrator::padded_seq_len() const
    {
        return current_padded_seq_len_;
    }

    const std::vector<int> &MultiDeviceOrchestrator::sequence_lengths() const
    {
        return current_sequence_lengths_;
    }

    int MultiDeviceOrchestrator::vocab_size() const
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

    void MultiDeviceOrchestrator::clear_cache()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::clear_cache: Clearing cache on all "
                  << device_runners_.size() << " TP devices and "
                  << pp_stage_runners_.size() << " PP stages");

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

    int MultiDeviceOrchestrator::get_position() const
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

    ExecutionPath MultiDeviceOrchestrator::executionPath() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->executionPath();
        }
        return ExecutionPath::GRAPH;
    }

    const char *MultiDeviceOrchestrator::architecture() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->architecture();
        }
        return "Unknown";
    }

    // =========================================================================
    // Hidden State API (for Pipeline Parallelism nesting)
    // =========================================================================

    TensorBase *MultiDeviceOrchestrator::getHiddenState()
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

    const TensorBase *MultiDeviceOrchestrator::getHiddenState() const
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

    void MultiDeviceOrchestrator::setHiddenState(TensorBase *hidden_state)
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

    bool MultiDeviceOrchestrator::hasHiddenStateInput() const
    {
        return hidden_state_input_ != nullptr;
    }

    void MultiDeviceOrchestrator::clearHiddenStateInput()
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

    void MultiDeviceOrchestrator::enableSnapshotCapture(const std::string &output_dir)
    {
        LOG_DEBUG("MultiDeviceOrchestrator::enableSnapshotCapture: Enabling on all devices");

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

    void MultiDeviceOrchestrator::disableSnapshotCapture()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::disableSnapshotCapture: Disabling on all devices");

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

    void MultiDeviceOrchestrator::clearSnapshots()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::clearSnapshots: Clearing on all devices");

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

    const float *MultiDeviceOrchestrator::getSnapshot(const std::string &key, size_t &out_size) const
    {
        // For LM_HEAD with multi-device TP, return the gathered combined_logits
        // This is necessary because each device only has logits_local with vocab_local entries,
        // but tests expect the full vocab_size logits.
        //
        // CRITICAL: For hybrid PP+TP, only the stage that actually HAS the LM head should
        // return combined_logits. Otherwise, a nested TP stage (like PP stage 0) that has
        // logits_local buffers allocated but never computes LM_HEAD would return stale data.
        if (key == "LM_HEAD" && device_runners_.size() > 1 && combined_logits_ && tp_ctx_)
        {
            // First, verify this stage actually owns the LM_HEAD
            auto wm = model_ctx_->weightManager();
            if (!wm || !wm->hasLMHead())
            {
                LOG_DEBUG("MultiDeviceOrchestrator::getSnapshot LM_HEAD: this stage doesn't own LM_HEAD, "
                          << "falling through to PP stage search (hasLMHead=" << (wm ? wm->hasLMHead() : false) << ")");
                // Fall through to PP stage search or return nullptr
            }
            else
            {
                // Check if we have column-parallel LM head
                bool has_column_parallel_lm_head = false;
                for (const auto &runner : device_runners_)
                {
                    if (runner)
                    {
                        const auto &state = runner->inferenceState();
                        if (state.logits_local)
                        {
                            has_column_parallel_lm_head = true;
                            break;
                        }
                    }
                }

                if (has_column_parallel_lm_head)
                {
                    // Return the combined logits which have full vocab_size
                    // Use the actual gathered size from last gatherLogits() call,
                    // NOT the buffer capacity (which is pre-allocated for max_seq_len)
                    out_size = last_gathered_logits_size_;
                    const float *ptr = combined_logits_->data();
                    LOG_DEBUG("MultiDeviceOrchestrator::getSnapshot LM_HEAD returning combined_logits with "
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

    std::vector<std::string> MultiDeviceOrchestrator::getSnapshotKeys() const
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

    TPSnapshot MultiDeviceOrchestrator::getTPSnapshot(const std::string &key) const
    {
        TPSnapshot result;
        result.key = key;
        result.mode = getStageShardingMode(key);
        result.tp_degree = static_cast<int>(device_runners_.size());

        LOG_DEBUG("MultiDeviceOrchestrator::getTPSnapshot: key=" << key
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

                // Check if this stage has the snapshot
                size_t temp_size = 0;
                const float *temp_data = pp_stage_runners_[stage_idx]->getSnapshot(key, temp_size);
                if (!temp_data || temp_size == 0)
                    continue;

                // Found the owning stage - check if it's a TP domain (MultiDeviceOrchestrator)
                auto *inner_mdo = dynamic_cast<const MultiDeviceOrchestrator *>(
                    pp_stage_runners_[stage_idx].get());

                if (inner_mdo)
                {
                    // This PP stage is a TP domain - delegate to its getTPSnapshot
                    LOG_DEBUG("MultiDeviceOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                                  << " is TP domain, delegating getTPSnapshot for key=" << key);
                    return inner_mdo->getTPSnapshot(key);
                }
                else
                {
                    // Single-device PP stage - wrap the snapshot as non-TP
                    LOG_DEBUG("MultiDeviceOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                                  << " is single device for key=" << key);

                    result.tp_degree = 1;
                    result.mode = SnapshotShardingMode::REPLICATED;

                    DeviceSnapshotData dev_data;
                    dev_data.device_index = 0;
                    dev_data.rows = 1;
                    dev_data.cols = temp_size;
                    dev_data.global_start_col = 0;
                    dev_data.global_total_cols = temp_size;
                    dev_data.data.assign(temp_data, temp_data + temp_size);

                    result.device_data.push_back(std::move(dev_data));
                    result.combined_valid = true;
                    result.combined_data = result.device_data[0].data;
                    result.combined_rows = 1;
                    result.combined_cols = temp_size;

                    return result;
                }
            }

            // Key not found in any PP stage
            LOG_DEBUG("MultiDeviceOrchestrator::getTPSnapshot: key=" << key
                                                                     << " not found in any PP stage");
            return result;
        }

        // =========================================================================
        // TP Mode (non-PP): Collect from device_runners_
        // =========================================================================

        // Special case: LM_HEAD with combined_logits_ already gathered
        if (key == "LM_HEAD" && device_runners_.size() > 1 && combined_logits_ && tp_ctx_)
        {
            bool has_column_parallel_lm_head = false;
            for (const auto &runner : device_runners_)
            {
                if (runner)
                {
                    const auto &state = runner->inferenceState();
                    if (state.logits_local)
                    {
                        has_column_parallel_lm_head = true;
                        break;
                    }
                }
            }

            if (has_column_parallel_lm_head)
            {
                // Return the already-gathered combined logits as a single "device"
                DeviceSnapshotData gathered;
                gathered.device_id = GlobalDeviceId::gpu(0, 0, config_.devices[0].device_type);
                gathered.device_index = 0;
                gathered.rows = 1; // Single position for decode
                gathered.cols = last_gathered_logits_size_;
                gathered.global_start_col = 0;
                gathered.global_total_cols = gathered.cols;
                gathered.data.assign(combined_logits_->data(),
                                     combined_logits_->data() + last_gathered_logits_size_);

                result.device_data.push_back(std::move(gathered));
                result.combined_valid = true;
                result.combined_data = result.device_data[0].data;
                result.combined_rows = result.device_data[0].rows;
                result.combined_cols = result.device_data[0].cols;

                LOG_DEBUG("MultiDeviceOrchestrator::getTPSnapshot: LM_HEAD using combined_logits "
                          << "size=" << last_gathered_logits_size_);
                return result;
            }
        }

        // Collect snapshots from all device runners
        size_t global_col_offset = 0;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                continue;

            size_t size = 0;
            const float *data = device_runners_[i]->getSnapshot(key, size);

            if (!data || size == 0)
            {
                LOG_DEBUG("MultiDeviceOrchestrator::getTPSnapshot: device " << i
                                                                            << " has no data for key=" << key);
                continue;
            }

            // Debug: log data pointer and first 4 values to verify each device has different data
            LOG_INFO("MultiDeviceOrchestrator::getTPSnapshot: device " << i
                                                                       << " key=" << key << " ptr=" << static_cast<const void *>(data)
                                                                       << " size=" << size
                                                                       << " val[0-3]=" << data[0] << "," << data[1] << "," << data[2] << "," << data[3]);

            DeviceSnapshotData dev_data;
            // Use device type from config if available, otherwise default to CUDA
            DeviceType dev_type = DeviceType::CUDA;
            if (i < config_.devices.size())
            {
                dev_type = config_.devices[i].device_type;
            }
            dev_data.device_id = GlobalDeviceId::gpu(0, static_cast<int>(i), dev_type);
            dev_data.device_index = static_cast<int>(i);
            dev_data.data.assign(data, data + size);

            // Infer rows and cols from size and TP configuration
            // For column-parallel stages, we need to compute actual row/col dimensions
            if (result.mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                // For column-parallel, each device has shape [seq_len, local_cols]
                // local_cols = global_cols / tp_degree
                // Different stages have different global widths:
                //   - ATTENTION_CONTEXT: hidden_size (896 for Qwen2.5-0.5B)
                //   - FFN_SWIGLU: d_ff (4864 for Qwen2.5-0.5B)
                //   - FFN_RESIDUAL: hidden_size after allreduce (back to 896)
                size_t local_cols = 0;
                if (model_ctx_)
                {
                    // Check if this is an FFN stage (contains "FFN" but NOT "RESIDUAL")
                    // FFN_SWIGLU output is [seq_len, d_ff_local] where d_ff_local = d_ff / tp_degree
                    bool is_ffn_stage =
                        (key.find("FFN") != std::string::npos && key.find("RESIDUAL") == std::string::npos);
                    if (is_ffn_stage)
                    {
                        size_t d_ff = static_cast<size_t>(model_ctx_->feedForwardLength());
                        local_cols = d_ff / static_cast<size_t>(result.tp_degree);
                    }
                    else
                    {
                        size_t hidden_size = model_ctx_->embeddingLength();
                        local_cols = hidden_size / static_cast<size_t>(result.tp_degree);
                    }
                }

                // If we couldn't get from model config, estimate from data size
                // Assume square-ish data or use global_col_offset pattern
                if (local_cols == 0 && result.tp_degree > 0)
                {
                    // Fallback: assume all devices have equal cols
                    // This will be validated when we have multiple devices
                    local_cols = size; // Will be rows=1 in worst case
                }

                // Compute rows from size and cols
                size_t rows = (local_cols > 0) ? (size / local_cols) : 1;
                if (rows * local_cols != size && local_cols > 0)
                {
                    // Size doesn't divide evenly - log warning and fallback
                    LOG_WARN("MultiDeviceOrchestrator::getTPSnapshot: size=" << size
                                                                             << " doesn't divide evenly by local_cols=" << local_cols
                                                                             << " for key=" << key);
                    rows = 1;
                    local_cols = size;
                }

                dev_data.rows = rows;
                dev_data.cols = local_cols;
                dev_data.global_start_col = global_col_offset;
                global_col_offset += local_cols;
            }
            else
            {
                // Replicated or row-parallel - each device has full output
                dev_data.rows = 1;
                dev_data.cols = size;
                dev_data.global_start_col = 0;
                dev_data.global_total_cols = size;
            }

            LOG_DEBUG("MultiDeviceOrchestrator::getTPSnapshot: device " << i
                                                                        << " size=" << size
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
    MultiDeviceOrchestrator::getSnapshotKeysWithSharding() const
    {
        auto keys = getSnapshotKeys();
        std::vector<std::pair<std::string, SnapshotShardingMode>> result;
        result.reserve(keys.size());

        for (const auto &key : keys)
        {
            result.emplace_back(key, getStageShardingMode(key));
        }

        return result;
    }

    // =========================================================================
    // Profiling API
    // =========================================================================

    const GraphExecutorStats *MultiDeviceOrchestrator::executorStats() const
    {
        aggregateStats();
        return aggregated_stats_.get();
    }

    void MultiDeviceOrchestrator::resetExecutorStats()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::resetExecutorStats: Resetting on all devices");

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
        stats_dirty_ = true;
    }

    // =========================================================================
    // Orchestration API
    // =========================================================================

    bool MultiDeviceOrchestrator::hasPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->hasPlacementPlan();
        }
        return false;
    }

    const PlacementPlan &MultiDeviceOrchestrator::getPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getPlacementPlan();
        }
        throw std::runtime_error("No placement plan available: no device runners");
    }

    // =========================================================================
    // IMultiDeviceOrchestrator Interface Implementation
    // =========================================================================

    int MultiDeviceOrchestrator::device_count() const
    {
        return static_cast<int>(device_runners_.size());
    }

    IInferenceRunner *MultiDeviceOrchestrator::deviceRunner(int device_idx)
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    const IInferenceRunner *MultiDeviceOrchestrator::deviceRunner(int device_idx) const
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    ILocalTPContext *MultiDeviceOrchestrator::localTPContext()
    {
        return tp_ctx_.get();
    }

    const ILocalTPContext *MultiDeviceOrchestrator::localTPContext() const
    {
        return tp_ctx_.get();
    }

    bool MultiDeviceOrchestrator::allDevicesReady() const
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

    void MultiDeviceOrchestrator::synchronizeDevices()
    {
        LOG_DEBUG("MultiDeviceOrchestrator::synchronizeDevices: Synchronizing all devices");

        if (tp_ctx_)
        {
            tp_ctx_->synchronize();
        }
    }

} // namespace llaminar2
