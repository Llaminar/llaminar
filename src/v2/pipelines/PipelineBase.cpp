/**
 * @file PipelineBase.cpp
 * @brief Base pipeline implementation
 * @author David Sanftenberg
 */

#include "../utils/Logger.h"
#include "../utils/DebugAssert.h"
#include "../utils/DebugEnv.h"
#include "../utils/KernelProfiler.h"
#include "../utils/OpenMPUtils.h"
#include "PipelineBase.h"
#include "attention/MpiAttentionOrchestrator.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorSlice.h"
#include "../tensors/SIMDHelpers.h"
#include "../kernels/cpu/attention/CpuAttentionKernelT.h"
#include "../kernels/cpu/attention/CPUAttentionKernelTyped.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <fstream>
#include <sys/stat.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace llaminar2
{

    PipelineBase::PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                               std::shared_ptr<MPIContext> mpi_ctx,
                               int device_idx,
                               std::shared_ptr<WeightPlacementMap> placement_map,
                               const PipelineConfig &config)
        : model_ctx_(model_ctx), mpi_ctx_(mpi_ctx), device_idx_(device_idx), config_(config), placement_map_(placement_map)
    {
        if (!model_ctx_)
        {
            throw std::runtime_error("PipelineBase: model_ctx cannot be null");
        }

        model_path_ = model_ctx_->path();

        // Initialize tensor factory for NUMA-aware allocation
        // If MPI context available, use it for NUMA awareness; otherwise create default single-rank context
        if (mpi_ctx_)
        {
            tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
        }
        else
        {
            // Create a default single-rank MPI context for TensorFactory
            // This allows pipelines to work without explicit MPI initialization
            default_mpi_ctx_ = std::make_unique<MPIContext>(0, 1, MPI_COMM_SELF); // rank 0, world_size 1, self comm
            tensor_factory_ = std::make_unique<TensorFactory>(*default_mpi_ctx_);
        }

        LOG_DEBUG("[PipelineBase] Initializing with model: " << model_path_);
        LOG_DEBUG("[PipelineBase] Runtime config: max_seq_len=" << config_.max_seq_len
                                                                << ", n_threads=" << config_.n_threads << ", batch_size=" << config_.batch_size);

        if (mpi_ctx_)
        {
            LOG_DEBUG("[PipelineBase] MPI context provided, rank "
                      << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size());
        }

        if (device_idx_ >= 0)
        {
            LOG_DEBUG("[PipelineBase] Device index: " << device_idx_ << " (GPU)\n");
            // TODO Phase 4: GPU tensor support
        }
        else
        {
            LOG_DEBUG("[PipelineBase] Device index: " << device_idx_ << " (CPU)\n");
        }

        // Create default placement map if not provided (all weights on device_idx_)
        if (!placement_map_)
        {
            LOG_DEBUG("[PipelineBase] No placement map provided, creating default (all on device " << device_idx_ << ")");
            placement_map_ = std::make_shared<WeightPlacementMap>(device_idx_);
        }

        // Generic pipeline initialization (derived classes have already set architecture params)
        // Derived constructors must set n_layers_, n_heads_, n_kv_heads_, head_dim_, d_model_
        // BEFORE calling initializeInfrastructure()
    }

    void PipelineBase::initializeInfrastructure()
    {
        // Use max_seq_len from runtime configuration
        int max_seq_len = config_.max_seq_len;

        // Phase 4.1: Device infrastructure (device discovery, buffer allocation)
        // Default batch_size=1 for backward compatibility
        initializeDeviceInfrastructure(max_seq_len, 1);

        // Phase 2: MPI strategy configuration (auto-select or validate)
        configureMPIStrategy();

        // Phase 3: KV cache initialization (uses attention device placement)
        initializeKVCache(max_seq_len);

        LOG_DEBUG("Pipeline infrastructure initialized (max_seq_len=" << max_seq_len << ")");
    }
    const float *PipelineBase::logits() const
    {
        DEBUG_ASSERT_NOT_NULL(logits_.get(), "logits() called before forward()");
        return logits_->data();
    }

    bool PipelineBase::attention_gqa(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        int batch_size, const std::vector<int> *sequence_lengths)
    {
        // Delegate to MpiAttentionOrchestrator static method
        MpiAttentionConfig config;
        config.n_heads = n_heads;
        config.n_kv_heads = n_kv_heads;
        config.head_dim = head_dim;
        config.causal = causal;
        config.window_size = window_size;
        config.precision = config_.activation_precision; // Use pipeline's activation precision setting
        config.mpi_ctx = mpi_ctx_;
        config.mpi_strategy = MPIStrategy::None; // Single-rank mode
        config.verbose_logging = mpi_config_.verbose_logging;

        // Provide workspace buffers (zero-allocation hot path)
        config.workspace_scores = attention_workspace_scores_;
        config.workspace_qkv_buffer = attention_workspace_qkv_buffer_;
        config.workspace_context = attention_workspace_context_;
        config.workspace_mask = attention_workspace_mask_;

        return MpiAttentionOrchestrator::compute(Q, K, V, output, config, batch_size, sequence_lengths);
    }

    bool PipelineBase::attention_gqa_batch(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const std::vector<int> &actual_lengths,
        int batch_size, int seq_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size)
    {
        // Delegate to MpiAttentionOrchestrator static method
        MpiAttentionConfig config;
        config.n_heads = n_heads;
        config.n_kv_heads = n_kv_heads;
        config.head_dim = head_dim;
        config.causal = causal;
        config.window_size = window_size;
        config.precision = config_.activation_precision; // Use pipeline's activation precision setting
        config.mpi_ctx = mpi_ctx_;
        config.mpi_strategy = MPIStrategy::None; // Single-rank mode
        config.verbose_logging = mpi_config_.verbose_logging;

        // Provide workspace buffers (zero-allocation hot path)
        config.workspace_scores = attention_workspace_scores_;
        config.workspace_qkv_buffer = attention_workspace_qkv_buffer_;
        config.workspace_context = attention_workspace_context_;
        config.workspace_mask = attention_workspace_mask_;

        return MpiAttentionOrchestrator::compute_batch(Q, K, V, output, actual_lengths, batch_size, seq_len, config);
    }

    // =============================================================================
    // Multi-Device Infrastructure (Phase 4)
    // =============================================================================

    std::vector<int> PipelineBase::discoverActiveDevices()
    {
        std::set<int> device_set;

        // Get all weight names from derived class (architecture-specific)
        std::vector<std::string> weight_names = getAllWeightNames();

        // Query placement map for each weight
        for (const auto &weight_name : weight_names)
        {
            // Try to extract layer index from weight name (e.g., "blk.5.attn_q.weight" -> 5)
            // This is a heuristic - some weights don't have layer indices
            int layer_idx = -1;
            size_t blk_pos = weight_name.find("blk.");
            if (blk_pos != std::string::npos)
            {
                size_t dot_pos = weight_name.find('.', blk_pos + 4);
                if (dot_pos != std::string::npos)
                {
                    std::string layer_str = weight_name.substr(blk_pos + 4, dot_pos - (blk_pos + 4));
                    try
                    {
                        layer_idx = std::stoi(layer_str);
                    }
                    catch (...)
                    {
                        // Not a valid layer index, keep -1
                    }
                }
            }

            int device = placement_map_->getDeviceForWeight(weight_name, layer_idx);
            device_set.insert(device);
        }

        // Convert set to sorted vector
        std::vector<int> devices(device_set.begin(), device_set.end());
        std::sort(devices.begin(), devices.end());

        return devices;
    }

    // =============================================================================
    // Phase 3: MoE Device Placement Helpers
    // =============================================================================

    std::vector<int> PipelineBase::detectAttentionDevices(int n_layers) const
    {
        std::vector<int> attention_devices(n_layers);

        for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
        {
            // Query placement map for attention block device
            // Uses Phase 2 block-level method (getAttentionDevice)
            attention_devices[layer_idx] = placement_map_->getAttentionDevice(layer_idx);
        }

        return attention_devices;
    }

    std::vector<int> PipelineBase::detectFFNDevices(int n_layers) const
    {
        std::vector<int> ffn_devices(n_layers);

        for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
        {
            // Query placement map for FFN block device
            // Uses Phase 2 block-level method (getFFNDevice)
            ffn_devices[layer_idx] = placement_map_->getFFNDevice(layer_idx);
        }

        return ffn_devices;
    }

    ActivationBuffers &PipelineBase::getBuffersForDevice(int device_idx)
    {
        // Check if we already have buffers for this device
        auto it = buffers_per_device_.find(device_idx);
        if (it != buffers_per_device_.end())
        {
            return it->second;
        }

        // Lazy allocation: create buffers for this device
        LOG_DEBUG("[PipelineBase] Lazy allocating buffers for device " << device_idx);

        // Determine max_seq_len from existing buffers (or use default)
        int max_seq_len = 2048; // Default
        if (!buffers_per_device_.empty())
        {
            max_seq_len = buffers_per_device_.begin()->second.max_seq_len;
        }

        // Call derived class to create buffers with architecture-specific dimensions
        ActivationBuffers buffers = createBuffersForDevice(device_idx, max_seq_len);

        // Insert into map
        auto [inserted_it, success] = buffers_per_device_.emplace(device_idx, std::move(buffers));
        if (!success)
        {
            throw std::runtime_error("Failed to insert buffers for device " + std::to_string(device_idx));
        }

        return inserted_it->second;
    }

    int PipelineBase::getWeightDevice(const std::string &weight_name, int layer_idx) const
    {
        return placement_map_->getDeviceForWeight(weight_name, layer_idx);
    }

    TensorBase *PipelineBase::prepareActivationForDevice(TensorBase *activation, int target_device, const std::string &context)
    {
        if (!activation)
        {
            LOG_ERROR("[PipelineBase] prepareActivationForDevice: null activation for " << context);
            return nullptr;
        }

        int current_device = activation->device_index();

        // Fast path: already on target device
        if (current_device == target_device)
        {
            return activation;
        }

        // Transfer required
        LOG_DEBUG("[PipelineBase] [" << context << "] Transferring activation from device "
                                     << current_device << " to device " << target_device);

        // Get target device's buffers
        ActivationBuffers &target_buffers = const_cast<PipelineBase *>(this)->getBuffersForDevice(target_device);

        // Use residual buffer as staging area
        TensorBase *staging = target_buffers.residual.get();

        // Compute element counts from shapes
        auto compute_element_count = [](const std::vector<size_t> &shape) -> size_t
        {
            size_t count = 1;
            for (auto dim : shape)
                count *= dim;
            return count;
        };

        size_t staging_count = compute_element_count(staging->shape());
        size_t activation_count = compute_element_count(activation->shape());

        // Validate staging buffer size
        if (staging_count < activation_count)
        {
            LOG_ERROR("[PipelineBase] Staging buffer too small: " << staging_count << " < " << activation_count);
            return nullptr;
        }

        // Perform transfer
        if (!staging->copyFrom(activation))
        {
            LOG_ERROR("[PipelineBase] Failed to transfer activation to device " << target_device);
            return nullptr;
        }

        // Update staging buffer's device index
        staging->set_device(target_device);

        return staging;
    }

    bool PipelineBase::attention_gqa_mpi(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        int batch_size, const std::vector<int> *sequence_lengths)
    {
        // Delegate to MpiAttentionOrchestrator static method
        MpiAttentionConfig config;
        config.n_heads = n_heads;
        config.n_kv_heads = n_kv_heads;
        config.head_dim = head_dim;
        config.causal = causal;
        config.window_size = window_size;
        config.precision = config_.activation_precision; // Use pipeline's activation precision setting
        config.mpi_ctx = mpi_ctx_;
        config.mpi_strategy = mpi_strategy_;
        config.verbose_logging = mpi_config_.verbose_logging;

        // Provide workspace buffers (zero-allocation hot path)
        config.workspace_scores = attention_workspace_scores_;
        config.workspace_qkv_buffer = attention_workspace_qkv_buffer_;
        config.workspace_context = attention_workspace_context_;
        config.workspace_mask = attention_workspace_mask_;

        return MpiAttentionOrchestrator::compute_mpi(Q, K, V, output, config, batch_size, sequence_lengths);
    }

    bool PipelineBase::attention_gqa_tensor_parallel(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        int batch_size, const std::vector<int> *sequence_lengths)
    {
        // Delegate to MpiAttentionOrchestrator static method
        MpiAttentionConfig config;
        config.n_heads = n_heads;
        config.n_kv_heads = n_kv_heads;
        config.head_dim = head_dim;
        config.causal = causal;
        config.window_size = window_size;
        config.precision = config_.activation_precision; // Use pipeline's activation precision setting

        // Provide workspace buffers (zero-allocation hot path)
        config.workspace_scores = attention_workspace_scores_;
        config.workspace_qkv_buffer = attention_workspace_qkv_buffer_;
        config.workspace_context = attention_workspace_context_;
        config.workspace_mask = attention_workspace_mask_;
        config.mpi_ctx = mpi_ctx_;
        config.mpi_strategy = MPIStrategy::TensorParallel; // Force tensor-parallel
        config.verbose_logging = mpi_config_.verbose_logging;

        return MpiAttentionOrchestrator::compute_tensor_parallel(Q, K, V, output, config, batch_size, sequence_lengths);
    }

    // =============================================================================
    // MPI Strategy Management
    // =============================================================================

    MPIStrategy PipelineBase::selectOptimalStrategy()
    {
        // No MPI or single rank
        if (!mpi_ctx_ || mpi_ctx_->world_size() == 1)
        {
            return MPIStrategy::None;
        }

        int world_size = mpi_ctx_->world_size();
        int rank = mpi_ctx_->rank();

        // Try tensor-parallel first (most common, best performance)
        if (validateStrategy(MPIStrategy::TensorParallel))
        {
            if (rank == 0)
            {
                LOG_DEBUG("[MPI Strategy] Selected TensorParallel (n_heads=" << n_heads_
                                                                             << " divisible by world_size=" << world_size << ")");
            }
            return MPIStrategy::TensorParallel;
        }

        // Fallback to pipeline-parallel
        if (validateStrategy(MPIStrategy::PipelineParallel))
        {
            if (rank == 0)
            {
                LOG_DEBUG("[MPI Strategy] Selected PipelineParallel (n_layers=" << n_layers_
                                                                                << " divisible by world_size=" << world_size << ")");
            }
            return MPIStrategy::PipelineParallel;
        }

        // No valid strategy found
        if (rank == 0)
        {
            LOG_WARN("[MPI Strategy] No valid strategy found (n_heads=" << n_heads_
                                                                        << ", n_layers=" << n_layers_
                                                                        << ", world_size=" << world_size
                                                                        << "). Using single-rank execution.");
        }

        return MPIStrategy::None;
    }

    bool PipelineBase::validateStrategy(MPIStrategy strategy)
    {
        if (!mpi_ctx_)
            return false;

        int world_size = mpi_ctx_->world_size();

        switch (strategy)
        {
        case MPIStrategy::None:
            return true; // Always valid

        case MPIStrategy::TensorParallel:
            // Requires n_heads divisible by world_size
            if (n_heads_ == 0)
            {
                LOG_WARN("[MPI Validation] n_heads not yet set (called too early?)");
                return false;
            }
            if (n_heads_ % world_size != 0)
            {
                LOG_WARN("[MPI Validation] TensorParallel requires n_heads (" << n_heads_
                                                                              << ") divisible by world_size (" << world_size << ")");
                return false;
            }
            return true;

        case MPIStrategy::PipelineParallel:
            // Requires n_layers divisible by world_size
            if (n_layers_ == 0)
            {
                LOG_WARN("[MPI Validation] n_layers not yet set (called too early?)");
                return false;
            }
            if (n_layers_ % world_size != 0)
            {
                LOG_WARN("[MPI Validation] PipelineParallel requires n_layers (" << n_layers_
                                                                                 << ") divisible by world_size (" << world_size << ")");
                return false;
            }
            return true;

        case MPIStrategy::SequenceParallel:
            // Always valid (can split any sequence length)
            return true;

        case MPIStrategy::Hybrid:
            // TODO: Implement hybrid validation (Phase 6)
            LOG_WARN("[MPI Validation] Hybrid strategy not yet implemented");
            return false;

        default:
            LOG_ERROR("[MPI Validation] Unknown strategy: " << static_cast<int>(strategy));
            return false;
        }
    }

    // =============================================================================
    // MPI Distribution Helpers
    // =============================================================================

    std::pair<size_t, size_t> PipelineBase::getHeadDistribution(int n_heads)
    {
        if (!mpi_ctx_)
        {
            return {0, static_cast<size_t>(n_heads)};
        }

        return mpi_ctx_->get_local_slice(static_cast<size_t>(n_heads));
    }

    std::pair<size_t, size_t> PipelineBase::getLayerDistribution(int n_layers)
    {
        if (!mpi_ctx_)
        {
            return {0, static_cast<size_t>(n_layers)};
        }

        return mpi_ctx_->get_local_slice(static_cast<size_t>(n_layers));
    }

    std::pair<size_t, size_t> PipelineBase::getTokenDistribution(int seq_len)
    {
        if (!mpi_ctx_)
        {
            return {0, static_cast<size_t>(seq_len)};
        }

        return mpi_ctx_->get_local_slice(static_cast<size_t>(seq_len));
    }

    // =============================================================================
    // Weight Device Orchestration (Phase 5)
    // =============================================================================

    bool PipelineBase::ensureWeightsOnDevice(
        const std::vector<std::shared_ptr<TensorBase>> &weights,
        int target_device,
        const std::string &context)
    {
        // CPU path: no transfer needed
        if (target_device < 0)
        {
            return true;
        }

        // Lazy transfer all weights to GPU
        // ensureOnDevice() is a no-op if already on target device
        bool success = true;
        int transfer_count = 0;

        for (const auto &weight : weights)
        {
            if (weight && !weight->ensureOnDevice(target_device))
            {
                LOG_ERROR("Failed to transfer weight to device " << target_device
                                                                 << (context.empty() ? "" : " (context: " + context + ")"));
                success = false;
            }
            else if (weight)
            {
                ++transfer_count;
            }
        }

        if (success && transfer_count > 0)
        {
            LOG_DEBUG("Ensured " << transfer_count << " weights on device " << target_device
                                 << (context.empty() ? "" : " (" + context + ")"));
        }

        return success;
    }

    bool PipelineBase::ensureWeightOnDevice(
        const std::shared_ptr<TensorBase> &weight,
        int target_device,
        const std::string &weight_name)
    {
        // CPU path or null weight: no transfer needed
        if (target_device < 0 || !weight)
        {
            return true;
        }

        if (!weight->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer " << (weight_name.empty() ? "weight" : weight_name)
                                            << " to device " << target_device);
            return false;
        }

        return true;
    }

    // =============================================================================
    // Generic Initialization (extracted from Qwen2Pipeline)
    // =============================================================================

    void PipelineBase::initializeDeviceInfrastructure(int max_seq_len, int batch_size)
    {
        // Phase 4.1: Discover which devices are used by this rank
        active_devices_ = discoverActiveDevices();
        std::stringstream devices_str;
        devices_str << "Active devices for this rank: [";
        for (size_t i = 0; i < active_devices_.size(); ++i)
        {
            if (i > 0)
                devices_str << ", ";
            devices_str << active_devices_[i];
        }
        devices_str << "]";
        LOG_DEBUG(devices_str.str());

        if (active_devices_.size() == 1 && active_devices_[0] == device_idx_)
        {
            // Single-device mode: use legacy path for backward compat
            LOG_DEBUG("Single-device mode (device " << device_idx_ << ")");
            // Derived class must allocate buffers via createBuffersForDevice
            activation_buffers_ = createBuffersForDevice(device_idx_, max_seq_len);
        }
        else
        {
            // Multi-device mode: allocate buffer pool per device
            LOG_DEBUG("Multi-device mode: allocating buffers for "
                      << active_devices_.size() << " devices");
            for (int dev_idx : active_devices_)
            {
                // Lazy allocation happens in getBuffersForDevice(), just ensure they exist
                ActivationBuffers &buffers = getBuffersForDevice(dev_idx);
                // Log memory usage (estimated, architecture-specific)
                // Derived class can override for accurate calculation
                (void)buffers; // Suppress unused warning
            }
            // For backward compat, point activation_buffers_ to primary device
            activation_buffers_ = buffers_per_device_[device_idx_];
        }

        // Phase 4.2: Allocate attention workspace buffers (zero-allocation hot path)
        // These buffers are reused across all attention calls, eliminating per-call allocations
        const int max_threads = omp_get_max_threads();
        const int total_len = batch_size * max_seq_len; // Total sequence length across batch
        LOG_DEBUG("Allocating attention workspace buffers (max_seq_len=" << max_seq_len
                                                                         << ", batch_size=" << batch_size
                                                                         << ", total_len=" << total_len
                                                                         << ", max_threads=" << max_threads << ")");
        ;

        // Scores buffer: [n_heads * total_len, total_len] where total_len = batch_size * max_seq_len
        // For batched attention, scores computed between all tokens across all sequences
        // Sized for worst case (all heads, full batch)
        // Use TensorFactory for NUMA-aware allocation
        attention_workspace_scores_ = std::shared_ptr<FP32Tensor>(
            tensor_factory_->createFP32(
                               {static_cast<size_t>(n_heads_ * total_len), static_cast<size_t>(total_len)},
                               device_idx_)
                .release());

        // QKV extraction buffer: [max_threads * total_len * head_dim * 3]
        // 3x: Q, K, V extraction buffers per thread, sized for full batch
        attention_workspace_qkv_buffer_ = std::shared_ptr<FP32Tensor>(
            tensor_factory_->createFP32(
                               {static_cast<size_t>(max_threads * total_len * head_dim_ * 3)},
                               device_idx_)
                .release());

        // Context buffer: [max_threads * total_len * head_dim]
        // Thread-local context accumulation, sized for full batch
        attention_workspace_context_ = std::shared_ptr<FP32Tensor>(
            tensor_factory_->createFP32(
                               {static_cast<size_t>(max_threads * total_len * head_dim_)},
                               device_idx_)
                .release());

        // Mask buffer: [total_len * total_len]
        // Causal/padding mask (reused across heads)
        // Must be square to support batched attention (attention between all tokens across all batches)
        attention_workspace_mask_ = std::shared_ptr<FP32Tensor>(
            tensor_factory_->createFP32(
                               {static_cast<size_t>(total_len * total_len)},
                               device_idx_)
                .release());

        LOG_DEBUG("Attention workspace buffers allocated (batch_size=" << batch_size << "): "
                                                                       << "scores=" << (n_heads_ * max_seq_len * max_seq_len * sizeof(float) / 1024 / 1024) << "MB, "
                                                                       << "qkv=" << (max_threads * max_seq_len * head_dim_ * 3 * sizeof(float) / 1024 / 1024) << "MB, "
                                                                       << "context=" << (max_threads * max_seq_len * head_dim_ * sizeof(float) / 1024 / 1024) << "MB, "
                                                                       << "mask=" << (total_len * total_len * sizeof(float) / 1024 / 1024) << "MB");
    }

    void PipelineBase::configureMPIStrategy()
    {
        // Configure MPI strategy if using multi-rank execution
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            // Use default MPIConfig (TensorParallel with auto_select=true)
            mpi_config_ = defaultMPIConfig();

            if (mpi_config_.auto_select)
            {
                mpi_strategy_ = selectOptimalStrategy();
            }
            else
            {
                // User specified a strategy - validate it
                mpi_strategy_ = mpi_config_.strategy;
                if (!validateStrategy(mpi_strategy_))
                {
                    LOG_WARN("[PipelineBase] User-specified strategy '"
                             << strategyName(mpi_strategy_) << "' invalid, using fallback");
                    mpi_strategy_ = mpi_config_.fallback_strategy;
                }
            }

            if (mpi_ctx_->rank() == 0)
            {
                LOG_DEBUG("[PipelineBase] MPI Strategy: " << strategyName(mpi_strategy_)
                                                          << " (rank " << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size() << ")");
                ;

                // Log strategy-specific info (virtual, can be overridden)
                logMPIStrategyInfo();
            }
        }
        else
        {
            mpi_strategy_ = MPIStrategy::None;
            if (mpi_ctx_)
            {
                LOG_DEBUG("[PipelineBase] Single-rank MPI execution (world_size=1)");
            }
        }
    }

    void PipelineBase::logMPIStrategyInfo()
    {
        // Default implementation: log tensor-parallel head distribution
        if (mpi_strategy_ == MPIStrategy::TensorParallel && n_heads_ > 0)
        {
            auto [start_head, local_n_heads] = getHeadDistribution(n_heads_);
            LOG_DEBUG("[PipelineBase] Tensor-parallel: " << local_n_heads
                                                         << " heads per rank (total: " << n_heads_ << ")");
        }
    }

    void PipelineBase::initializeKVCache(int max_seq_len, int batch_size)
    {
        DEBUG_ASSERT(n_layers_ > 0, "n_layers_ must be set before calling initializeKVCache");
        DEBUG_ASSERT(n_kv_heads_ > 0, "n_kv_heads_ must be set before calling initializeKVCache");
        DEBUG_ASSERT(head_dim_ > 0, "head_dim_ must be set before calling initializeKVCache");

        // Use effective MPI context (user-provided or default single-rank)
        const MPIContext &effective_mpi_ctx = mpi_ctx_ ? *mpi_ctx_ : *default_mpi_ctx_;

        // Phase 3: Use placement map to detect attention devices per layer
        std::vector<int> attention_devices = detectAttentionDevices(n_layers_);

        // Create unified KV cache matching pipeline's activation precision
        kv_cache_ = createUnifiedKVCache(config_.activation_precision, effective_mpi_ctx,
                                         n_layers_, batch_size, max_seq_len, n_kv_heads_, head_dim_,
                                         attention_devices);
        current_positions_.clear(); // Will be resized to batch_size in forward_batch()

        LOG_DEBUG("Initialized unified KV cache: " << n_layers_ << " layers, "
                                                   << "batch_size=" << batch_size << ", "
                                                   << max_seq_len << " max_seq_len, "
                                                   << n_kv_heads_ << " KV heads, " << head_dim_ << " head_dim"
                                                   << ", precision: " << static_cast<int>(config_.activation_precision));
    }

    // ===== Snapshot Capture Implementation =====
    // Only compiled when ENABLE_PIPELINE_SNAPSHOTS is defined
    // In release builds, these functions don't exist (callers get compile errors if used)

#ifdef ENABLE_PIPELINE_SNAPSHOTS

    void PipelineBase::enableSnapshotCapture(const std::string &output_dir)
    {
        snapshot_capture_enabled_ = true;
        snapshot_output_dir_ = output_dir;
        snapshots_.clear();
        LOG_DEBUG("[PipelineBase] Snapshot capture ENABLED" << (output_dir.empty() ? " (memory only)" : " (output: " + output_dir + ")"));
    }

    void PipelineBase::disableSnapshotCapture()
    {
        snapshot_capture_enabled_ = false;
        snapshots_.clear();
        LOG_DEBUG("[PipelineBase] Snapshot capture DISABLED");
    }

    void PipelineBase::clearSnapshots()
    {
        snapshots_.clear();
        LOG_DEBUG("[PipelineBase] Snapshots cleared (capture still " << (snapshot_capture_enabled_ ? "ENABLED" : "DISABLED") << ")");
    }

    const float *PipelineBase::getSnapshot(const std::string &key, size_t &out_size) const
    {
        auto it = snapshots_.find(key);
        if (it == snapshots_.end())
        {
            out_size = 0;
            return nullptr;
        }
        out_size = it->second.size();
        return it->second.data();
    }

    std::vector<std::string> PipelineBase::getSnapshotKeys() const
    {
        std::vector<std::string> keys;
        keys.reserve(snapshots_.size());
        for (const auto &pair : snapshots_)
        {
            keys.push_back(pair.first);
        }
        return keys;
    }

    void PipelineBase::dumpTensorToDisk(const std::string &key, TensorBase *tensor, int rows, int cols)
    {
        const auto &snap_cfg = debugEnv().snapshot;
        if (!snap_cfg.tensor_dump_enabled)
            return;

        // Check MPI rank filter
        int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        if (!snap_cfg.shouldDumpRank(rank))
            return;

        // Create output directory if needed
        struct stat st;
        if (stat(snap_cfg.dump_dir.c_str(), &st) != 0)
        {
            mkdir(snap_cfg.dump_dir.c_str(), 0755);
        }

        std::string base_path = snap_cfg.dump_dir + "/" + key + "_rank" + std::to_string(rank);

        // Get shape info
        size_t effective_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);

        // Dump FP32 dequantized data
        auto view = tensor->create_view({static_cast<size_t>(rows), static_cast<size_t>(cols)});
        if (view)
        {
            const float *fp32_data = view->fp32_data();
            if (fp32_data)
            {
                std::ofstream fp32_file(base_path + "_fp32.bin", std::ios::binary);
                if (fp32_file)
                {
                    fp32_file.write(reinterpret_cast<const char *>(fp32_data),
                                    effective_elements * sizeof(float));
                    LOG_DEBUG("[TensorDump] Wrote FP32 data to " << base_path << "_fp32.bin"
                                                                 << " (" << effective_elements << " elements)");
                }
            }
        }

        // Check if tensor is Q8_1 and dump native blocks
        auto *q8_1_tensor = dynamic_cast<Q8_1Tensor *>(tensor);
        if (q8_1_tensor)
        {
            size_t blocks_per_row = (cols + 31) / 32;
            size_t total_blocks = rows * blocks_per_row;
            const Q8_1Block *blocks = q8_1_tensor->q8_1_blocks();

            std::ofstream blocks_file(base_path + "_q8_1_blocks.bin", std::ios::binary);
            if (blocks_file)
            {
                blocks_file.write(reinterpret_cast<const char *>(blocks),
                                  total_blocks * sizeof(Q8_1Block));
                LOG_DEBUG("[TensorDump] Wrote Q8_1 blocks to " << base_path << "_q8_1_blocks.bin"
                                                               << " (" << total_blocks << " blocks)");
            }
        }

        // Write metadata
        std::ofstream meta_file(base_path + "_metadata.txt");
        if (meta_file)
        {
            meta_file << "key=" << key << "\n";
            meta_file << "rows=" << rows << "\n";
            meta_file << "cols=" << cols << "\n";
            meta_file << "elements=" << effective_elements << "\n";
            meta_file << "tensor_type=" << static_cast<int>(tensor->native_type()) << "\n";
            meta_file << "rank=" << rank << "\n";

            if (q8_1_tensor)
            {
                size_t blocks_per_row = (cols + 31) / 32;
                meta_file << "q8_1_blocks_per_row=" << blocks_per_row << "\n";
                meta_file << "q8_1_total_blocks=" << rows * blocks_per_row << "\n";

                // Sample first few block scales
                const Q8_1Block *blocks = q8_1_tensor->q8_1_blocks();
                meta_file << "sample_scales=";
                for (int i = 0; i < std::min(5, rows); ++i)
                {
                    uint16_t d_bits = blocks[i * blocks_per_row].d;
                    float scale;
                    std::memcpy(&scale, &d_bits, sizeof(uint16_t));
                    // FP16 to FP32 conversion (manual)
                    uint32_t sign = (d_bits >> 15) & 0x1;
                    uint32_t exp = (d_bits >> 10) & 0x1F;
                    uint32_t mant = d_bits & 0x3FF;
                    uint32_t fp32_bits;
                    if (exp == 0)
                    {
                        fp32_bits = sign << 31;
                    }
                    else if (exp == 31)
                    {
                        fp32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
                    }
                    else
                    {
                        fp32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                    }
                    std::memcpy(&scale, &fp32_bits, sizeof(float));
                    meta_file << scale;
                    if (i < std::min(5, rows) - 1)
                        meta_file << ",";
                }
                meta_file << "\n";
            }

            LOG_DEBUG("[TensorDump] Wrote metadata to " << base_path << "_metadata.txt");
        }
    }

#endif // ENABLE_PIPELINE_SNAPSHOTS

    // ===== Tensor Dump (always available, controlled by debugEnv) =====

    void PipelineBase::maybeDumpTensor(const std::string &key, TensorBase *tensor, int rows, int cols)
    {
        const auto &snap_cfg = debugEnv().snapshot;

        // Quick exit if tensor dump not enabled
        if (!snap_cfg.tensor_dump_enabled)
            return;

        // Check if this key matches the configured filters
        if (!snap_cfg.shouldDumpKey(key))
            return;

        // Check MPI rank filter
        int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        if (!snap_cfg.shouldDumpRank(rank))
            return;

        // Create output directory if needed
        struct stat st;
        if (stat(snap_cfg.dump_dir.c_str(), &st) != 0)
        {
            mkdir(snap_cfg.dump_dir.c_str(), 0755);
        }

        // Prefix with pipeline activation precision so we can distinguish FP32 vs Q8_1 runs
        std::string precision_prefix;
        switch (config_.activation_precision)
        {
        case ActivationPrecision::FP32:
            precision_prefix = "fp32_pipeline_";
            break;
        case ActivationPrecision::Q8_1:
            precision_prefix = "q8_1_pipeline_";
            break;
        default:
            precision_prefix = "unknown_";
            break;
        }

        std::string base_path = snap_cfg.dump_dir + "/" + precision_prefix + key + "_rank" + std::to_string(rank);

        // Get shape info
        size_t effective_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);

        // Dump FP32 dequantized data
        auto view = tensor->create_view({static_cast<size_t>(rows), static_cast<size_t>(cols)});
        if (view)
        {
            const float *fp32_data = view->fp32_data();
            if (fp32_data)
            {
                std::ofstream fp32_file(base_path + "_fp32.bin", std::ios::binary);
                if (fp32_file)
                {
                    fp32_file.write(reinterpret_cast<const char *>(fp32_data),
                                    effective_elements * sizeof(float));
                    LOG_DEBUG("[TensorDump] Wrote " << key << " FP32 data (" << rows << "x" << cols
                                                    << " = " << effective_elements << " elements) to " << base_path << "_fp32.bin");
                }
            }
        }

        // Check if tensor is Q8_1 and dump native blocks
        auto *q8_1_tensor = dynamic_cast<Q8_1Tensor *>(tensor);
        if (q8_1_tensor)
        {
            size_t blocks_per_row = (cols + 31) / 32;
            size_t total_blocks = rows * blocks_per_row;
            const Q8_1Block *blocks = q8_1_tensor->q8_1_blocks();

            std::ofstream blocks_file(base_path + "_q8_1_blocks.bin", std::ios::binary);
            if (blocks_file)
            {
                blocks_file.write(reinterpret_cast<const char *>(blocks),
                                  total_blocks * sizeof(Q8_1Block));
                LOG_DEBUG("[TensorDump] Wrote " << key << " Q8_1 blocks (" << total_blocks << " blocks) to "
                                                << base_path << "_q8_1_blocks.bin");
            }
        }

        // Write metadata
        std::ofstream meta_file(base_path + "_metadata.txt");
        if (meta_file)
        {
            meta_file << "key=" << key << "\n";
            meta_file << "rows=" << rows << "\n";
            meta_file << "cols=" << cols << "\n";
            meta_file << "elements=" << effective_elements << "\n";
            meta_file << "tensor_type=" << static_cast<int>(tensor->native_type()) << "\n";
            int rank_val = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            meta_file << "rank=" << rank_val << "\n";

            if (q8_1_tensor)
            {
                size_t blocks_per_row = (cols + 31) / 32;
                meta_file << "q8_1_blocks_per_row=" << blocks_per_row << "\n";
                meta_file << "q8_1_total_blocks=" << rows * blocks_per_row << "\n";

                // Sample first few block scales for sanity check
                const Q8_1Block *blocks = q8_1_tensor->q8_1_blocks();
                meta_file << "sample_scales=";
                for (int i = 0; i < std::min(5, rows); ++i)
                {
                    uint16_t d_bits = blocks[i * blocks_per_row].d;
                    // FP16 to FP32 conversion
                    uint32_t sign = (d_bits >> 15) & 0x1;
                    uint32_t exp = (d_bits >> 10) & 0x1F;
                    uint32_t mant = d_bits & 0x3FF;
                    uint32_t fp32_bits;
                    if (exp == 0)
                    {
                        fp32_bits = sign << 31;
                    }
                    else if (exp == 31)
                    {
                        fp32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
                    }
                    else
                    {
                        fp32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                    }
                    float scale;
                    std::memcpy(&scale, &fp32_bits, sizeof(float));
                    meta_file << scale;
                    if (i < std::min(5, rows) - 1)
                        meta_file << ",";
                }
                meta_file << "\n";
            }
        }
    }

    // =============================================================================
    // Numerical Health Checks (Debug builds only)
    // =============================================================================

#ifndef NDEBUG
    float check_numerical_health_impl(const char *stage_name,
                                      const float *data,
                                      size_t len)
    {
        if (!data || len == 0)
        {
            return 0.0f;
        }

        // Vectorized computation of max_abs and sum_abs
        float max_val = 0.0f;
        float sum_abs = 0.0f;

#if defined(__AVX512F__)
        // AVX512: Process 16 floats per iteration
        __m512 vmax = _mm512_setzero_ps();
        __m512 vsum = _mm512_setzero_ps();

        size_t i = 0;
        for (; i + 16 <= len; i += 16)
        {
            __m512 v = _mm512_loadu_ps(&data[i]);
            __m512 vabs = _mm512_abs_ps(v); // Absolute value (flip sign bit)
            vmax = _mm512_max_ps(vmax, vabs);
            vsum = _mm512_add_ps(vsum, vabs);
        }

        // Horizontal reduction
        float max_array[16], sum_array[16];
        _mm512_storeu_ps(max_array, vmax);
        _mm512_storeu_ps(sum_array, vsum);

        for (int j = 0; j < 16; ++j)
        {
            max_val = std::max(max_val, max_array[j]);
            sum_abs += sum_array[j];
        }

        // Scalar tail
        for (; i < len; ++i)
        {
            float abs_val = std::fabs(data[i]);
            max_val = std::max(max_val, abs_val);
            sum_abs += abs_val;
        }

#elif defined(__AVX2__)
        // AVX2: Process 8 floats per iteration
        __m256 vmax = _mm256_setzero_ps();
        __m256 vsum = _mm256_setzero_ps();
        const __m256 sign_mask = _mm256_set1_ps(-0.0f); // Sign bit mask

        size_t i = 0;
        for (; i + 8 <= len; i += 8)
        {
            __m256 v = _mm256_loadu_ps(&data[i]);
            __m256 vabs = _mm256_andnot_ps(sign_mask, v); // Clear sign bit
            vmax = _mm256_max_ps(vmax, vabs);
            vsum = _mm256_add_ps(vsum, vabs);
        }

        // Horizontal reduction
        float max_array[8], sum_array[8];
        _mm256_storeu_ps(max_array, vmax);
        _mm256_storeu_ps(sum_array, vsum);

        for (int j = 0; j < 8; ++j)
        {
            max_val = std::max(max_val, max_array[j]);
            sum_abs += sum_array[j];
        }

        // Scalar tail
        for (; i < len; ++i)
        {
            float abs_val = std::fabs(data[i]);
            max_val = std::max(max_val, abs_val);
            sum_abs += abs_val;
        }

#elif defined(__SSE2__)
        // SSE2: Process 4 floats per iteration
        __m128 vmax = _mm_setzero_ps();
        __m128 vsum = _mm_setzero_ps();
        const __m128 sign_mask = _mm_set1_ps(-0.0f);

        size_t i = 0;
        for (; i + 4 <= len; i += 4)
        {
            __m128 v = _mm_loadu_ps(&data[i]);
            __m128 vabs = _mm_andnot_ps(sign_mask, v);
            vmax = _mm_max_ps(vmax, vabs);
            vsum = _mm_add_ps(vsum, vabs);
        }

        // Horizontal reduction
        float max_array[4], sum_array[4];
        _mm_storeu_ps(max_array, vmax);
        _mm_storeu_ps(sum_array, vsum);

        for (int j = 0; j < 4; ++j)
        {
            max_val = std::max(max_val, max_array[j]);
            sum_abs += sum_array[j];
        }

        // Scalar tail
        for (; i < len; ++i)
        {
            float abs_val = std::fabs(data[i]);
            max_val = std::max(max_val, abs_val);
            sum_abs += abs_val;
        }

#else
        // Scalar fallback
        for (size_t i = 0; i < len; ++i)
        {
            float abs_val = std::fabs(data[i]);
            max_val = std::max(max_val, abs_val);
            sum_abs += abs_val;
        }
#endif

        // Compute statistics
        float mean = sum_abs / static_cast<float>(len);
        float dynamic_range = max_val / (mean + 1e-10f);

        // Check for numerical issues
        constexpr float EXPLODING_THRESHOLD = 1e3f;
        constexpr float COLLAPSING_THRESHOLD = 1e-5f;

        if (max_val > EXPLODING_THRESHOLD)
        {
            LOG_WARN("[NumHealth] Exploding activations at " << stage_name
                                                             << ": max=" << max_val << ", mean=" << mean
                                                             << ", dynamic_range=" << dynamic_range);
            return 0.0f; // Signal unhealthy
        }

        if (mean < COLLAPSING_THRESHOLD)
        {
            LOG_WARN("[NumHealth] Collapsing distribution at " << stage_name
                                                               << ": max=" << max_val << ", mean=" << mean
                                                               << ", dynamic_range=" << dynamic_range);
            return 0.0f; // Signal unhealthy
        }

        // Optional: Warn on poor dynamic range (may indicate quantization issues)
        constexpr float POOR_DYNAMIC_RANGE_THRESHOLD = 1e4f;
        if (dynamic_range > POOR_DYNAMIC_RANGE_THRESHOLD)
        {
            LOG_DEBUG("[NumHealth] High dynamic range at " << stage_name
                                                           << ": max=" << max_val << ", mean=" << mean
                                                           << ", dynamic_range=" << dynamic_range);
        }

        return dynamic_range;
    }
#endif // NDEBUG

    // =============================================================================
    // Declarative Compute Graph Operations
    // =============================================================================

    bool PipelineBase::rms_norm(
        TensorBase *input, const TensorBase *gamma, TensorBase *output,
        int rows, int cols, float eps,
        const std::string &snapshot_key, int device)
    {
        KERNEL_PROFILE_SCOPE(KernelType::RMS_NORM);

        int target_device = (device >= 0) ? device : device_idx_;

        // Check if PipelineExecutor should handle this operation
        // (based on config flags and whether this is FFN or attention norm)
        bool is_ffn_norm = snapshot_key.find("FFN_NORM") != std::string::npos;
        bool is_attn_norm = snapshot_key.find("ATTENTION_NORM") != std::string::npos;
        bool use_executor = pipeline_executor_ &&
                            ((is_ffn_norm && config_.executor_ffn_norm) ||
                             (is_attn_norm && config_.executor_attn_norm));

        if (use_executor)
        {
            // Use new multi-device executor framework
            if (!pipeline_executor_->executeRMSNorm(input, gamma, output, rows, cols, eps))
            {
                return false;
            }
        }
        else
        {
            // Use existing typed op
            if (!typed_rmsnorm_op_ || !typed_rmsnorm_op_->execute(
                                          input, gamma, output, rows, cols, eps,
                                          mpi_ctx_.get(), target_device))
            {
                return false;
            }
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT_VIEW(snapshot_key.c_str(), output, rows, cols);
        return true;
    }

    bool PipelineBase::project(
        const TensorBase *input, TensorBase *weight, TensorBase *output,
        int m, int n, int k,
        const std::string &snapshot_key, int device)
    {
        // Profile with appropriate kernel type based on operation
        KernelType profile_type = (snapshot_key == "LM_HEAD") ? KernelType::LM_HEAD : (snapshot_key.find("FFN_GATE") != std::string::npos) ? KernelType::FFN_GATE
                                                                                  : (snapshot_key.find("FFN_UP") != std::string::npos)     ? KernelType::FFN_UP
                                                                                  : (snapshot_key.find("FFN_DOWN") != std::string::npos)   ? KernelType::FFN_DOWN
                                                                                                                                           : KernelType::GEMM_Q8; // Default to Q8 GEMM (most common)
        KERNEL_PROFILE_SCOPE(profile_type);

        int target_device = (device >= 0) ? device : device_idx_;

        // Use typed_gemm_op_ to support Q8_1 output tensors
        if (!typed_gemm_op_->execute(input, weight, output, m, n, k,
                                     mpi_ctx_.get(), target_device))
        {
            return false;
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT_VIEW(snapshot_key.c_str(), output, m, n);
        return true;
    }

    bool PipelineBase::project_row_parallel(
        const TensorBase *input, TensorBase *weight, TensorBase *output,
        int m, int n, int k,
        const std::string &snapshot_key, int device)
    {
        // Row-parallel projection: GEMM with output dimension partitioned across ranks
        //
        // For TensorSlice weights (ROW_PARALLEL mode):
        //   - Weight is TensorSlice wrapping full [n, k] tensor
        //   - TensorSlice::createGemm() creates kernel for rows [row_start, row_end)
        //   - n_local = row_end - row_start (output dimension is sliced)
        //   - Each rank computes: C_local = A @ W_local^T  (shape: [m, n_local])
        //   - Concatenate C_local from all ranks to get C_global [m, n]
        //
        // For old FP32 sliced weights (K dimension sliced):
        //   - Weight is [n, k_local] where k_local = k / world_size
        //   - Input is [m, k_local] (the local slice)
        //   - Each rank computes: C_local = A_local @ W_local^T
        //   - Allreduce-sum to get: C_global = sum(C_local)

        KernelType profile_type = (snapshot_key.find("FFN_DOWN") != std::string::npos) ? KernelType::FFN_DOWN
                                                                                       : KernelType::GEMM_Q8;
        KERNEL_PROFILE_SCOPE(profile_type);

        int target_device = (device >= 0) ? device : device_idx_;

        // Check if weight is a TensorSlice with row-parallel mode
        auto *tensor_slice = dynamic_cast<TensorSlice *>(weight);
        if (tensor_slice && tensor_slice->is_row_parallel() && mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            // ROW-PARALLEL PATH: TensorSlice with row-parallel mode
            // Each rank has a slice of the output columns (rows in transposed weight matrix)
            // The TensorSlice::createGemm() returns a kernel for the sliced rows
            const auto &meta = tensor_slice->metadata();
            const int n_local = static_cast<int>(meta.slice_size());
            const int n_start = static_cast<int>(meta.slice_start);
            const int world_size = mpi_ctx_->world_size();
            const int rank = mpi_ctx_->rank();

            // Detect output tensor type
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);
            auto *output_q8_1 = dynamic_cast<Q8_1Tensor *>(output);
            auto *output_bf16 = dynamic_cast<BF16Tensor *>(output);
            auto *output_fp16 = dynamic_cast<FP16Tensor *>(output);

            // Check if Q8_1 native path is viable (requires block-aligned slices)
            // Block boundaries must align with slice boundaries for clean allgather
            constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE; // 32
            const bool n_start_aligned = (n_start % BLOCK_SIZE) == 0;
            const bool n_local_aligned = (n_local % BLOCK_SIZE) == 0;
            const bool q8_block_aligned = n_start_aligned && n_local_aligned;

            // =====================================================================
            // Q8_1 NATIVE PATH: Direct Q8_1 GEMM + block-level allgatherv
            // Avoids FP32 intermediate when slices are block-aligned
            // =====================================================================
            if (output_q8_1 && q8_block_aligned)
            {
                // Calculate block counts
                const size_t blocks_per_row_full = n / BLOCK_SIZE;
                const size_t blocks_per_row_local = n_local / BLOCK_SIZE;
                const size_t block_start = n_start / BLOCK_SIZE;

                // Create Q8_1 temporary for local GEMM result [m, n_local]
                std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(n_local)};
                Q8_1Tensor output_local(local_shape);

                // GEMM with sliced kernel: [m, k] @ [n_local, k]^T = [m, n_local] in Q8_1
                if (!typed_gemm_op_->execute(input, weight, &output_local, m, n_local, k,
                                             mpi_ctx_.get(), target_device))
                {
                    LOG_ERROR("[TP GEMM] Row-parallel TensorSlice Q8_1 GEMM failed for " << snapshot_key);
                    return false;
                }

                // Get Q8_1 block pointers
                const Q8_1Block *local_blocks = output_local.q8_1_blocks();
                Q8_1Block *output_blocks = output_q8_1->mutable_q8_1_blocks();

                // Allgather Q8_1 blocks from all ranks
                // Each rank contributes blocks for its column slice
                // We gather row-by-row to handle the strided layout

                // For each row, gather blocks from all ranks into correct positions
                // Use allgatherv with displacement arrays
                std::vector<int> recv_counts(world_size);
                std::vector<int> displs(world_size);

                // Calculate counts and displacements for one row
                // Each rank sends blocks_per_row_local blocks per row
                for (int r = 0; r < world_size; ++r)
                {
                    size_t r_n_local = meta.original_cols / world_size;
                    if (r == world_size - 1)
                    {
                        r_n_local = meta.original_cols - r_n_local * (world_size - 1);
                    }
                    size_t r_blocks_local = r_n_local / BLOCK_SIZE;
                    size_t r_block_start = (r * (meta.original_cols / world_size)) / BLOCK_SIZE;

                    recv_counts[r] = static_cast<int>(r_blocks_local * sizeof(Q8_1Block));
                    displs[r] = static_cast<int>(r_block_start * sizeof(Q8_1Block));
                }

                // Gather blocks row by row (strided pattern)
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                const int local_row_bytes = static_cast<int>(blocks_per_row_local * sizeof(Q8_1Block));

                OMP_SINGLE([&]()
                           {
                    for (int row = 0; row < m; ++row)
                    {
                        const Q8_1Block *src_row = local_blocks + row * blocks_per_row_local;
                        Q8_1Block *dst_row = output_blocks + row * blocks_per_row_full;

                        mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes,
                                                   dst_row, recv_counts.data(), displs.data());
                    } });

                LOG_TRACE("[TP GEMM] Row-parallel TensorSlice Q8_1-native completed for " << snapshot_key
                                                                                          << " (m=" << m << ", n_local=" << n_local << "/" << n
                                                                                          << ", blocks_local=" << blocks_per_row_local << ")");
                return true;
            }

            // =====================================================================
            // BF16 NATIVE PATH: Direct BF16 GEMM + element-level allgatherv
            // =====================================================================
            if (output_bf16)
            {
                // Create BF16 temporary for local GEMM result [m, n_local]
                std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(n_local)};
                BF16Tensor output_local(local_shape);

                // GEMM with sliced kernel
                if (!typed_gemm_op_->execute(input, weight, &output_local, m, n_local, k,
                                             mpi_ctx_.get(), target_device))
                {
                    LOG_ERROR("[TP GEMM] Row-parallel TensorSlice BF16 GEMM failed for " << snapshot_key);
                    return false;
                }

                // Get BF16 data pointers
                const uint16_t *local_data = output_local.bf16_data();
                uint16_t *output_data = output_bf16->mutable_bf16_data();

                // Calculate counts and displacements for allgatherv
                std::vector<int> recv_counts(world_size);
                std::vector<int> displs(world_size);

                for (int r = 0; r < world_size; ++r)
                {
                    size_t r_n_local = meta.original_cols / world_size;
                    if (r == world_size - 1)
                    {
                        r_n_local = meta.original_cols - r_n_local * (world_size - 1);
                    }
                    size_t r_start = r * (meta.original_cols / world_size);

                    recv_counts[r] = static_cast<int>(r_n_local * sizeof(uint16_t));
                    displs[r] = static_cast<int>(r_start * sizeof(uint16_t));
                }

                // Gather elements row by row
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                const int local_row_bytes = static_cast<int>(n_local * sizeof(uint16_t));

                OMP_SINGLE([&]()
                           {
                    for (int row = 0; row < m; ++row)
                    {
                        const uint16_t *src_row = local_data + row * n_local;
                        uint16_t *dst_row = output_data + row * n;

                        mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes,
                                                   dst_row, recv_counts.data(), displs.data());
                    } });

                LOG_TRACE("[TP GEMM] Row-parallel TensorSlice BF16-native completed for " << snapshot_key);
                return true;
            }

            // =====================================================================
            // FP16 NATIVE PATH: Direct FP16 GEMM + element-level allgatherv
            // =====================================================================
            if (output_fp16)
            {
                // Create FP16 temporary for local GEMM result [m, n_local]
                std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(n_local)};
                FP16Tensor output_local(local_shape);

                // GEMM with sliced kernel
                if (!typed_gemm_op_->execute(input, weight, &output_local, m, n_local, k,
                                             mpi_ctx_.get(), target_device))
                {
                    LOG_ERROR("[TP GEMM] Row-parallel TensorSlice FP16 GEMM failed for " << snapshot_key);
                    return false;
                }

                // Get FP16 data pointers
                const uint16_t *local_data = output_local.fp16_data();
                uint16_t *output_data = output_fp16->mutable_fp16_data();

                // Calculate counts and displacements for allgatherv
                std::vector<int> recv_counts(world_size);
                std::vector<int> displs(world_size);

                for (int r = 0; r < world_size; ++r)
                {
                    size_t r_n_local = meta.original_cols / world_size;
                    if (r == world_size - 1)
                    {
                        r_n_local = meta.original_cols - r_n_local * (world_size - 1);
                    }
                    size_t r_start = r * (meta.original_cols / world_size);

                    recv_counts[r] = static_cast<int>(r_n_local * sizeof(uint16_t));
                    displs[r] = static_cast<int>(r_start * sizeof(uint16_t));
                }

                // Gather elements row by row
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                const int local_row_bytes = static_cast<int>(n_local * sizeof(uint16_t));

                OMP_SINGLE([&]()
                           {
                    for (int row = 0; row < m; ++row)
                    {
                        const uint16_t *src_row = local_data + row * n_local;
                        uint16_t *dst_row = output_data + row * n;

                        mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes,
                                                   dst_row, recv_counts.data(), displs.data());
                    } });

                LOG_TRACE("[TP GEMM] Row-parallel TensorSlice FP16-native completed for " << snapshot_key);
                return true;
            }

            // =====================================================================
            // FP32 PATH (default) or Q8_1 FALLBACK (non-block-aligned)
            // Uses FP32 for communication, quantizes to Q8_1 at end if needed
            // =====================================================================
            if (!output_fp32 && !output_q8_1)
            {
                LOG_ERROR("[TP GEMM] Row-parallel TensorSlice requires FP32, Q8_1, BF16, or FP16 output");
                return false;
            }

            // Log if Q8_1 fallback due to alignment
            if (output_q8_1 && !q8_block_aligned)
            {
                LOG_DEBUG("[TP GEMM] Q8_1 output using FP32 fallback (n_start=" << n_start
                                                                                << " aligned=" << n_start_aligned
                                                                                << ", n_local=" << n_local
                                                                                << " aligned=" << n_local_aligned << ")");
            }

            // Create FP32 buffer for communication
            std::unique_ptr<FP32Tensor> output_fp32_temp;
            FP32Tensor *output_for_comm = nullptr;

            if (output_fp32)
            {
                output_for_comm = output_fp32;
            }
            else
            {
                // Q8_1 output with non-aligned slices: use FP32 temporary
                output_fp32_temp = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});
                output_for_comm = output_fp32_temp.get();
            }

            // Create FP32 temporary for local GEMM result [m, n_local]
            std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(n_local)};
            FP32Tensor output_local(local_shape);

            // GEMM with sliced kernel: [m, k] @ [n_local, k]^T = [m, n_local]
            if (!typed_gemm_op_->execute(input, weight, &output_local, m, n_local, k,
                                         mpi_ctx_.get(), target_device))
            {
                LOG_ERROR("[TP GEMM] Row-parallel TensorSlice GEMM failed for " << snapshot_key);
                return false;
            }

            // Gather results using allgatherv (more efficient than zero+allreduce)
            float *out_data = output_for_comm->mutable_data();
            const float *local_data = output_local.data();

            // Calculate counts and displacements for allgatherv
            std::vector<int> recv_counts(world_size);
            std::vector<int> displs(world_size);

            for (int r = 0; r < world_size; ++r)
            {
                size_t r_n_local = meta.original_cols / world_size;
                if (r == world_size - 1)
                {
                    r_n_local = meta.original_cols - r_n_local * (world_size - 1);
                }
                size_t r_start = r * (meta.original_cols / world_size);

                recv_counts[r] = static_cast<int>(r_n_local * sizeof(float));
                displs[r] = static_cast<int>(r_start * sizeof(float));
            }

            // Gather elements row by row
            // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
            const int local_row_bytes = static_cast<int>(n_local * sizeof(float));

            OMP_SINGLE([&]()
                       {
                for (int row = 0; row < m; ++row)
                {
                    const float *src_row = local_data + row * n_local;
                    float *dst_row = out_data + row * n;

                    mpi_ctx_->allgatherv_bytes(src_row, local_row_bytes,
                                               dst_row, recv_counts.data(), displs.data());
                } });

            // If output is Q8_1, quantize FP32 result into Q8_1 blocks
            if (output_q8_1)
            {
                Q8_1Block *q8_blocks = output_q8_1->mutable_q8_1_blocks();
                const float *fp32_data = output_for_comm->data();

                const size_t blocks_per_row = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

                auto quantize_work = [&]()
                {
#pragma omp for schedule(static)
                    for (int row = 0; row < m; ++row)
                    {
                        const float *src_row = fp32_data + row * n;
                        Q8_1Block *dst_blocks = q8_blocks + row * blocks_per_row;

                        const size_t n_full_blocks = n / BLOCK_SIZE;
                        simd::quantize_fp32_to_q8_1_blocks(src_row, dst_blocks, n_full_blocks * BLOCK_SIZE);

                        // Handle tail elements if any
                        const size_t tail_start = n_full_blocks * BLOCK_SIZE;
                        if (tail_start < static_cast<size_t>(n))
                        {
                            const size_t tail_count = n - tail_start;
                            Q8_1Block &tail_block = dst_blocks[n_full_blocks];

                            float tail_buf[BLOCK_SIZE] = {0};
                            std::memcpy(tail_buf, src_row + tail_start, tail_count * sizeof(float));
                            simd::quantize_fp32_to_q8_1_blocks(tail_buf, &tail_block, BLOCK_SIZE);
                        }
                    }
                };
                OMP_WORKSHARE_REGION(quantize_work);
            }

            LOG_TRACE("[TP GEMM] Row-parallel TensorSlice completed for " << snapshot_key
                                                                          << " (m=" << m << ", n_local=" << n_local << "/" << n << ", k=" << k << ")");
            return true;
        }

        // K-SLICED ROW-PARALLEL PATH: Weight K dimension is partitioned
        // Weight is [n, k_local], input slice is [m, k_local]
        // Each rank computes partial product, allreduce-sum combines results
        const auto &weight_shape = weight->shape();
        const int weight_k = static_cast<int>(weight_shape.size() == 2 ? weight_shape[1] : 0);
        const bool is_k_sliced = (weight_k > 0 && weight_k < k); // k_local < k_full means sharded

        if (is_k_sliced && mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            const int world_size = mpi_ctx_->world_size();
            const int rank = mpi_ctx_->rank();
            const int k_local = weight_k;
            const int k_start = k_local * rank;

            // Detect input/output tensor types
            const auto *input_fp32 = dynamic_cast<const FP32Tensor *>(input);
            auto *input_q8_1_mutable = const_cast<Q8_1Tensor *>(dynamic_cast<const Q8_1Tensor *>(input));
            const auto *input_bf16 = dynamic_cast<const BF16Tensor *>(input);
            const auto *input_fp16 = dynamic_cast<const FP16Tensor *>(input);

            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);
            auto *output_q8_1 = dynamic_cast<Q8_1Tensor *>(output);
            auto *output_bf16 = dynamic_cast<BF16Tensor *>(output);
            auto *output_fp16 = dynamic_cast<FP16Tensor *>(output);

            // Check Q8_1 block alignment for native k-slicing
            constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE; // 32
            const bool k_start_aligned = Q8_1Tensor::is_k_aligned(k_start);
            const bool k_local_aligned = Q8_1Tensor::is_k_aligned(k_local);
            const bool q8_k_aligned = k_start_aligned && k_local_aligned;

            // =====================================================================
            // Q8_1 NATIVE PATH: Q8_1 input + Q8_1 output + aligned k-slicing
            // GEMM produces Q8_1 output, native Q8_1 allreduce
            // =====================================================================
            if (input_q8_1_mutable && output_q8_1 && q8_k_aligned)
            {
                // Native Q8_1 k-slicing for input
                auto input_q8_1_sliced = input_q8_1_mutable->slice_k_blocks(k_start, k_local);
                if (!input_q8_1_sliced)
                {
                    LOG_WARN("[TP GEMM] Q8_1 slice_k_blocks() failed, falling back to FP32 path");
                    goto k_sliced_fp32_fallback;
                }

                // GEMM: [m, k_local] (Q8_1) @ [n, k_local]^T → [m, n] (Q8_1)
                if (!typed_gemm_op_->execute(input_q8_1_sliced.get(), weight, output_q8_1, m, n, k_local,
                                             mpi_ctx_.get(), target_device))
                {
                    LOG_ERROR("[TP GEMM] K-sliced Q8_1-native GEMM failed for " << snapshot_key);
                    return false;
                }

                // Native Q8_1 allreduce-sum (AVX512 dequant→sum→requant)
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                Q8_1Block *blocks = output_q8_1->mutable_q8_1_blocks();
                const size_t output_size = static_cast<size_t>(m) * n;
                const size_t n_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                OMP_SINGLE([&]()
                           { mpi_ctx_->allreduce_q8_1_inplace(blocks, n_blocks); });

                LOG_TRACE("[TP GEMM] K-sliced Q8_1-native completed for " << snapshot_key
                                                                          << " (m=" << m << ", n=" << n
                                                                          << ", k_local=" << k_local << "/" << k << ")");
                return true;
            }

            // =====================================================================
            // BF16 NATIVE PATH: BF16 input + BF16 output
            // GEMM produces BF16 output, native BF16 allreduce
            // =====================================================================
            if (input_bf16 && output_bf16)
            {
                // Create BF16 k-sliced input view
                std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(k_local)};
                BF16Tensor input_local(local_shape);

                const uint16_t *src = input_bf16->bf16_data();
                uint16_t *dst = input_local.mutable_bf16_data();

                auto bf16_slice_work = [&]()
                {
#pragma omp for schedule(static)
                    for (int row = 0; row < m; ++row)
                    {
                        const uint16_t *src_row = src + row * k + k_start;
                        uint16_t *dst_row = dst + row * k_local;
                        std::memcpy(dst_row, src_row, k_local * sizeof(uint16_t));
                    }
                };
                OMP_WORKSHARE_REGION(bf16_slice_work);

                // GEMM: [m, k_local] (BF16) @ [n, k_local]^T → [m, n] (BF16)
                if (!typed_gemm_op_->execute(&input_local, weight, output_bf16, m, n, k_local,
                                             mpi_ctx_.get(), target_device))
                {
                    LOG_ERROR("[TP GEMM] K-sliced BF16-native GEMM failed for " << snapshot_key);
                    return false;
                }

                // Native BF16 allreduce-sum
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                uint16_t *out_data = output_bf16->mutable_bf16_data();
                const size_t output_size = static_cast<size_t>(m) * n;
                OMP_SINGLE([&]()
                           { mpi_ctx_->allreduce_bf16_inplace(out_data, output_size); });

                LOG_TRACE("[TP GEMM] K-sliced BF16-native completed for " << snapshot_key);
                return true;
            }

            // =====================================================================
            // FP16 NATIVE PATH: FP16 input + FP16 output
            // GEMM produces FP16 output, native FP16 allreduce
            // =====================================================================
            if (input_fp16 && output_fp16)
            {
                // Create FP16 k-sliced input view
                std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(k_local)};
                FP16Tensor input_local(local_shape);

                const uint16_t *src = input_fp16->fp16_data();
                uint16_t *dst = input_local.mutable_fp16_data();

                auto fp16_slice_work = [&]()
                {
#pragma omp for schedule(static)
                    for (int row = 0; row < m; ++row)
                    {
                        const uint16_t *src_row = src + row * k + k_start;
                        uint16_t *dst_row = dst + row * k_local;
                        std::memcpy(dst_row, src_row, k_local * sizeof(uint16_t));
                    }
                };
                OMP_WORKSHARE_REGION(fp16_slice_work);

                // GEMM: [m, k_local] (FP16) @ [n, k_local]^T → [m, n] (FP16)
                if (!typed_gemm_op_->execute(&input_local, weight, output_fp16, m, n, k_local,
                                             mpi_ctx_.get(), target_device))
                {
                    LOG_ERROR("[TP GEMM] K-sliced FP16-native GEMM failed for " << snapshot_key);
                    return false;
                }

                // Native FP16 allreduce-sum
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                uint16_t *out_data = output_fp16->mutable_fp16_data();
                const size_t output_size = static_cast<size_t>(m) * n;
                OMP_SINGLE([&]()
                           { mpi_ctx_->allreduce_fp16_inplace(out_data, output_size); });

                LOG_TRACE("[TP GEMM] K-sliced FP16-native completed for " << snapshot_key);
                return true;
            }

        // =====================================================================
        // FP32 PATH (default) or Q8_1 FALLBACK (unaligned k)
        // Uses FP32 for allreduce, requantizes to Q8_1 if needed
        // =====================================================================
        k_sliced_fp32_fallback:

            // Prepare input for GEMM
            std::shared_ptr<Q8_1Tensor> input_q8_1_sliced;
            std::unique_ptr<FP32Tensor> input_local_tensor;
            TensorBase *input_for_gemm = nullptr;

            // Try Q8_1 native k-slicing if aligned (even if output will be FP32)
            if (input_q8_1_mutable && q8_k_aligned)
            {
                input_q8_1_sliced = input_q8_1_mutable->slice_k_blocks(k_start, k_local);
                if (input_q8_1_sliced)
                {
                    input_for_gemm = input_q8_1_sliced.get();
                    LOG_DEBUG("[TP GEMM] Using Q8_1 k-slicing with FP32 output for " << snapshot_key);
                }
            }

            // Fallback: dequantize Q8_1 or use FP32/BF16/FP16 directly with FP32 slice
            if (!input_for_gemm)
            {
                const float *src = nullptr;
                if (input_fp32)
                {
                    src = input_fp32->data();
                }
                else if (input_q8_1_mutable)
                {
                    src = input_q8_1_mutable->fp32_data(); // Dequantizes on-demand
                }
                else if (input_bf16)
                {
                    // BF16 input with non-BF16 output: dequantize to FP32
                    LOG_DEBUG("[TP GEMM] BF16 input with non-BF16 output, using FP32 fallback");
                    std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(k_local)};
                    input_local_tensor = std::make_unique<FP32Tensor>(local_shape);
                    float *dst = input_local_tensor->mutable_data();
                    const uint16_t *bf16_src = input_bf16->bf16_data();

                    auto bf16_dequant_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (int row = 0; row < m; ++row)
                        {
                            for (int col = 0; col < k_local; ++col)
                            {
                                uint16_t bf16_val = bf16_src[row * k + k_start + col];
                                dst[row * k_local + col] = simd::bf16_to_fp32(bf16_val);
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(bf16_dequant_work);
                    input_for_gemm = input_local_tensor.get();
                }
                else if (input_fp16)
                {
                    // FP16 input with non-FP16 output: dequantize to FP32
                    LOG_DEBUG("[TP GEMM] FP16 input with non-FP16 output, using FP32 fallback");
                    std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(k_local)};
                    input_local_tensor = std::make_unique<FP32Tensor>(local_shape);
                    float *dst = input_local_tensor->mutable_data();
                    const uint16_t *fp16_src = input_fp16->fp16_data();

                    auto fp16_dequant_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (int row = 0; row < m; ++row)
                        {
                            for (int col = 0; col < k_local; ++col)
                            {
                                uint16_t fp16_val = fp16_src[row * k + k_start + col];
                                dst[row * k_local + col] = simd::fp16_to_fp32(fp16_val);
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(fp16_dequant_work);
                    input_for_gemm = input_local_tensor.get();
                }
                else
                {
                    LOG_ERROR("[TP GEMM] K-sliced row-parallel requires FP32, Q8_1, BF16, or FP16 input");
                    return false;
                }

                // Create FP32 slice if needed (FP32 or dequantized Q8_1 source)
                if (src && !input_for_gemm)
                {
                    std::vector<float> input_local(static_cast<size_t>(m) * k_local);

                    auto fp32_slice_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (int row = 0; row < m; ++row)
                        {
                            const float *src_row = src + row * k + k_start;
                            float *dst_row = input_local.data() + row * k_local;
                            std::memcpy(dst_row, src_row, k_local * sizeof(float));
                        }
                    };
                    OMP_WORKSHARE_REGION(fp32_slice_work);

                    std::vector<size_t> local_shape = {static_cast<size_t>(m), static_cast<size_t>(k_local)};
                    input_local_tensor = std::make_unique<FP32Tensor>(local_shape);
                    std::memcpy(input_local_tensor->mutable_data(), input_local.data(), input_local.size() * sizeof(float));
                    input_for_gemm = input_local_tensor.get();
                }
            }

            // Validate we have an input
            if (!input_for_gemm)
            {
                LOG_ERROR("[TP GEMM] Failed to prepare input for k-sliced GEMM");
                return false;
            }

            // Prepare FP32 output buffer for allreduce
            std::unique_ptr<FP32Tensor> output_fp32_temp;
            FP32Tensor *output_for_allreduce = nullptr;

            if (output_fp32)
            {
                output_for_allreduce = output_fp32;
            }
            else
            {
                // Non-FP32 output: use FP32 temporary for allreduce
                output_fp32_temp = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});
                output_for_allreduce = output_fp32_temp.get();
            }

            // GEMM: [m, k_local] @ [n, k_local]^T → [m, n]
            if (!typed_gemm_op_->execute(input_for_gemm, weight, output_for_allreduce, m, n, k_local,
                                         mpi_ctx_.get(), target_device))
            {
                LOG_ERROR("[TP GEMM] K-sliced row-parallel GEMM failed for " << snapshot_key);
                return false;
            }

            // FP32 allreduce-sum
            // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
            float *out_data = output_for_allreduce->mutable_data();
            const size_t output_size = static_cast<size_t>(m) * n;
            OMP_SINGLE([&]()
                       { mpi_ctx_->allreduce_sum_inplace(out_data, output_size); });

            // Convert FP32 result to output tensor type if needed
            if (output_q8_1)
            {
                Q8_1Block *q8_blocks = output_q8_1->mutable_q8_1_blocks();
                const float *fp32_data = output_for_allreduce->data();
                const size_t blocks_per_row = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

                auto quantize_work = [&]()
                {
#pragma omp for schedule(static)
                    for (int row = 0; row < m; ++row)
                    {
                        const float *src_row = fp32_data + row * n;
                        Q8_1Block *dst_blocks = q8_blocks + row * blocks_per_row;

                        const size_t n_full_blocks = n / BLOCK_SIZE;
                        simd::quantize_fp32_to_q8_1_blocks(src_row, dst_blocks, n_full_blocks * BLOCK_SIZE);

                        const size_t tail_start = n_full_blocks * BLOCK_SIZE;
                        if (tail_start < static_cast<size_t>(n))
                        {
                            const size_t tail_count = n - tail_start;
                            Q8_1Block &tail_block = dst_blocks[n_full_blocks];

                            float tail_buf[BLOCK_SIZE] = {0};
                            std::memcpy(tail_buf, src_row + tail_start, tail_count * sizeof(float));
                            simd::quantize_fp32_to_q8_1_blocks(tail_buf, &tail_block, BLOCK_SIZE);
                        }
                    }
                };
                OMP_WORKSHARE_REGION(quantize_work);
            }
            else if (output_bf16)
            {
                const float *fp32_data = output_for_allreduce->data();
                uint16_t *bf16_data = output_bf16->mutable_bf16_data();

                auto bf16_convert_work = [&]()
                {
#pragma omp for simd schedule(static)
                    for (size_t i = 0; i < output_size; ++i)
                    {
                        bf16_data[i] = simd::fp32_to_bf16(fp32_data[i]);
                    }
                };
                OMP_WORKSHARE_REGION(bf16_convert_work);
            }
            else if (output_fp16)
            {
                const float *fp32_data = output_for_allreduce->data();
                uint16_t *fp16_data = output_fp16->mutable_fp16_data();

                auto fp16_convert_work = [&]()
                {
#pragma omp for simd schedule(static)
                    for (size_t i = 0; i < output_size; ++i)
                    {
                        fp16_data[i] = simd::fp32_to_fp16(fp32_data[i]);
                    }
                };
                OMP_WORKSHARE_REGION(fp16_convert_work);
            }

            LOG_TRACE("[TP GEMM] K-sliced row-parallel (FP32 path) completed for " << snapshot_key
                                                                                   << " (m=" << m << ", n=" << n << ", k_local=" << k_local << "/" << k << ")");
            return true;
        }
        else
        {
            // FALLBACK: Full GEMM with workaround allreduce
            // Perform local GEMM with full weights
            // Use typed_gemm_op_ to support Q8_1 output tensors
            if (!typed_gemm_op_->execute(input, weight, output, m, n, k,
                                         mpi_ctx_.get(), target_device))
            {
                return false;
            }

            // For replicated weights: scale by 1/world_size then allreduce
            // This produces the same result as a single-rank computation
            // Dispatch by tensor type to use the appropriate native allreduce
            if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
            {
                const size_t output_size = static_cast<size_t>(m) * n;
                const int world_size = mpi_ctx_->world_size();
                const float scale = 1.0f / world_size;

                if (auto *output_fp32 = dynamic_cast<FP32Tensor *>(output))
                {
                    // FP32: Scale then allreduce
                    float *out_data = output_fp32->mutable_data();

                    auto scale_work = [&]()
                    {
#pragma omp for simd schedule(static)
                        for (size_t i = 0; i < output_size; ++i)
                        {
                            out_data[i] *= scale;
                        }
                    };
                    OMP_WORKSHARE_REGION(scale_work);

                    // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                    OMP_SINGLE([&]()
                               { mpi_ctx_->allreduce_sum_inplace(out_data, output_size); });

                    LOG_TRACE("[TP GEMM] Row-parallel (replicated) FP32 allreduce for " << snapshot_key
                                                                                        << " (m=" << m << ", n=" << n << ")");
                }
                else if (auto *output_q8 = dynamic_cast<Q8_1Tensor *>(output))
                {
                    // Q8_1: Scale blocks then native Q8_1 allreduce
                    Q8_1Block *blocks = output_q8->mutable_q8_1_blocks();
                    if (!blocks)
                    {
                        LOG_ERROR("[TP GEMM] Row-parallel (replicated) failed to get mutable Q8_1 blocks");
                        return false;
                    }

                    // Scale Q8_1 blocks by 1/world_size (scale the 'd' field)
                    const size_t n_blocks = (output_size + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                    auto block_scale_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (size_t b = 0; b < n_blocks; ++b)
                        {
                            float d = fp16_to_fp32(blocks[b].d);
                            blocks[b].d = fp32_to_fp16(d * scale);
                        }
                    };
                    OMP_WORKSHARE_REGION(block_scale_work);

                    // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                    OMP_SINGLE([&]()
                               { mpi_ctx_->allreduce_q8_1_inplace(blocks, n_blocks); });

                    LOG_TRACE("[TP GEMM] Row-parallel (replicated) Q8_1-native allreduce for " << snapshot_key
                                                                                               << " (m=" << m << ", n=" << n << ", n_blocks=" << n_blocks << ")");
                }
                else if (auto *output_fp16 = dynamic_cast<FP16Tensor *>(output))
                {
                    // FP16: Scale then native FP16 allreduce
                    uint16_t *fp16_data = output_fp16->mutable_fp16_data();
                    if (!fp16_data)
                    {
                        LOG_ERROR("[TP GEMM] Row-parallel (replicated) failed to get mutable FP16 data");
                        return false;
                    }

                    // Scale FP16 values by 1/world_size
                    simd::fp16_scale_inplace(fp16_data, output_size, scale);

                    // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                    OMP_SINGLE([&]()
                               { mpi_ctx_->allreduce_fp16_inplace(fp16_data, output_size); });

                    LOG_TRACE("[TP GEMM] Row-parallel (replicated) FP16-native allreduce for " << snapshot_key
                                                                                               << " (m=" << m << ", n=" << n << ")");
                }
                else if (auto *output_bf16 = dynamic_cast<BF16Tensor *>(output))
                {
                    // BF16: Scale then native BF16 allreduce
                    uint16_t *bf16_data = output_bf16->mutable_bf16_data();
                    if (!bf16_data)
                    {
                        LOG_ERROR("[TP GEMM] Row-parallel (replicated) failed to get mutable BF16 data");
                        return false;
                    }

                    // Scale BF16 values by 1/world_size
                    simd::bf16_scale_inplace(bf16_data, output_size, scale);

                    // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                    OMP_SINGLE([&]()
                               { mpi_ctx_->allreduce_bf16_inplace(bf16_data, output_size); });

                    LOG_TRACE("[TP GEMM] Row-parallel (replicated) BF16-native allreduce for " << snapshot_key
                                                                                               << " (m=" << m << ", n=" << n << ")");
                }
                else
                {
                    LOG_WARN("[TP GEMM] Row-parallel (replicated) allreduce unsupported for output tensor type");
                }
            }
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT_VIEW(snapshot_key.c_str(), output, m, n);
        return true;
    }

    bool PipelineBase::project_column_parallel(
        const TensorBase *input, TensorBase *weight, TensorBase *output,
        int m, int n, int k_local,
        const std::string &snapshot_key, int device)
    {
        // Column-parallel projection for cascaded tensor parallelism
        // Input: [m, k_local] (local slice from previous column-parallel layer)
        // Weight: [n, k_local] (column-parallel slice)
        // Output: [m, n] (partial sum, needs allreduce)

        KernelType profile_type = (snapshot_key.find("FFN_DOWN") != std::string::npos) ? KernelType::FFN_DOWN
                                                                                       : KernelType::GEMM_Q8;
        KERNEL_PROFILE_SCOPE(profile_type);

        int target_device = (device >= 0) ? device : device_idx_;

        // Perform local GEMM: [m, k_local] @ [n, k_local]^T = [m, n]
        // Use typed_gemm_op_ to support Q8_1 output tensors
        if (!typed_gemm_op_->execute(input, weight, output, m, n, k_local,
                                     mpi_ctx_.get(), target_device))
        {
            LOG_ERROR("[TP GEMM] Column-parallel GEMM failed for " << snapshot_key);
            return false;
        }

        // Allreduce-sum to combine partial results from all ranks
        // NOTE: Required for all activation precisions when using column-parallel sharding.
        // Dispatch by tensor type to use the appropriate native allreduce.
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            const size_t output_size = static_cast<size_t>(m) * n;

            if (auto *output_fp32 = dynamic_cast<FP32Tensor *>(output))
            {
                // FP32: Use native MPI_FLOAT allreduce
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                float *out_data = output_fp32->mutable_data();
                OMP_SINGLE([&]()
                           { mpi_ctx_->allreduce_sum_inplace(out_data, output_size); });

                LOG_TRACE("[TP GEMM] Column-parallel FP32 allreduce completed for " << snapshot_key
                                                                                    << " (m=" << m << ", n=" << n << ", k_local=" << k_local << ")");
            }
            else if (auto *output_q8 = dynamic_cast<Q8_1Tensor *>(output))
            {
                // Q8_1: Use Q8_1-native allreduce (AVX512 vectorized)
                // Each block is dequantized, summed, and requantized in SIMD registers.
                Q8_1Block *blocks = output_q8->mutable_q8_1_blocks();
                if (!blocks)
                {
                    LOG_ERROR("[TP GEMM] Column-parallel failed to get mutable Q8_1 blocks for allreduce");
                    return false;
                }

                // Calculate number of blocks: ceil(elements / 32)
                const size_t n_blocks = (output_size + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                OMP_SINGLE([&]()
                           { mpi_ctx_->allreduce_q8_1_inplace(blocks, n_blocks); });

                LOG_TRACE("[TP GEMM] Column-parallel Q8_1-native allreduce completed for " << snapshot_key
                                                                                           << " (m=" << m << ", n=" << n << ", k_local=" << k_local
                                                                                           << ", n_blocks=" << n_blocks << ")");
            }
            else if (auto *output_fp16 = dynamic_cast<FP16Tensor *>(output))
            {
                // FP16: Use FP16-native allreduce (AVX512/AVX2 F16C vectorized)
                uint16_t *fp16_data = output_fp16->mutable_fp16_data();
                if (!fp16_data)
                {
                    LOG_ERROR("[TP GEMM] Column-parallel failed to get mutable FP16 data for allreduce");
                    return false;
                }

                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                OMP_SINGLE([&]()
                           { mpi_ctx_->allreduce_fp16_inplace(fp16_data, output_size); });

                LOG_TRACE("[TP GEMM] Column-parallel FP16-native allreduce completed for " << snapshot_key
                                                                                           << " (m=" << m << ", n=" << n << ", k_local=" << k_local << ")");
            }
            else if (auto *output_bf16 = dynamic_cast<BF16Tensor *>(output))
            {
                // BF16: Use BF16-native allreduce (AVX512 vectorized)
                uint16_t *bf16_data = output_bf16->mutable_bf16_data();
                if (!bf16_data)
                {
                    LOG_ERROR("[TP GEMM] Column-parallel failed to get mutable BF16 data for allreduce");
                    return false;
                }

                // OMP_SINGLE ensures only one thread calls MPI in layer-level parallel region
                OMP_SINGLE([&]()
                           { mpi_ctx_->allreduce_bf16_inplace(bf16_data, output_size); });

                LOG_TRACE("[TP GEMM] Column-parallel BF16-native allreduce completed for " << snapshot_key
                                                                                           << " (m=" << m << ", n=" << n << ", k_local=" << k_local << ")");
            }
            else
            {
                LOG_WARN("[TP GEMM] Column-parallel allreduce unsupported for output tensor type");
            }
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT_VIEW(snapshot_key.c_str(), output, m, n);
        return true;
    }

    bool PipelineBase::add_residual(
        const TensorBase *residual, const TensorBase *input, TensorBase *output,
        int batch_size, int seq_len, int hidden_dim,
        const std::vector<int> &sequence_lengths,
        const std::string &snapshot_key)
    {
        KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);

        // Check if PipelineExecutor should handle this operation
        // (based on config flags and whether this is FFN or attention residual)
        bool is_ffn_residual = snapshot_key.find("FFN_RESIDUAL") != std::string::npos;
        bool is_attn_residual = snapshot_key.find("ATTENTION_RESIDUAL") != std::string::npos;
        bool use_executor = pipeline_executor_ &&
                            ((is_ffn_residual && config_.executor_ffn_residual) ||
                             (is_attn_residual && config_.executor_attn_residual));

        if (use_executor)
        {
            // Use new multi-device executor framework
            // Note: PipelineExecutor::executeResidualAdd expects simple element count
            int total_elements = batch_size * seq_len * hidden_dim;
            if (!pipeline_executor_->executeResidualAdd(residual, input, output, total_elements))
            {
                return false;
            }
        }
        else
        {
            // Use typed residual op (supports Q8_1/BF16/FP16 activation precision)
            if (!typed_residual_op_->batched(
                    const_cast<TensorBase *>(residual),
                    const_cast<TensorBase *>(input),
                    output,
                    batch_size, sequence_lengths.data(), seq_len, hidden_dim))
            {
                return false;
            }
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT(snapshot_key.c_str(), output);
        return true;
    }

    bool PipelineBase::swiglu(
        TensorBase *gate, TensorBase *up, TensorBase *output,
        int rows, int cols,
        const std::string &snapshot_key, int device)
    {
        KERNEL_PROFILE_SCOPE(KernelType::SWIGLU);

        int target_device = (device >= 0) ? device : device_idx_;

        // Check if PipelineExecutor should handle this operation
        bool use_executor = pipeline_executor_ && config_.executor_ffn_swiglu;

        if (use_executor)
        {
            // Use new multi-device executor framework
            if (!pipeline_executor_->executeSwiGLU(gate, up, output, rows, cols))
            {
                return false;
            }
        }
        else
        {
            // Use typed_swiglu_op_ to support Q8_1 tensors
            if (!typed_swiglu_op_->execute(gate, up, output, rows, cols,
                                           mpi_ctx_.get(), target_device))
            {
                return false;
            }
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT_VIEW(snapshot_key.c_str(), output, rows, cols);
        return true;
    }

    bool PipelineBase::apply_rope(
        TensorBase *Q, TensorBase *K,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float theta,
        const std::string &snapshot_prefix, int device)
    {
        KERNEL_PROFILE_SCOPE(KernelType::ROPE);

        int target_device = (device >= 0) ? device : device_idx_;

        // Check if PipelineExecutor should handle this operation
        bool use_executor = pipeline_executor_ && config_.executor_rope;

        if (use_executor)
        {
            // Use new multi-device executor framework
            if (!pipeline_executor_->executeRoPE(Q, K, position_ids, seq_len, n_heads, n_kv_heads, head_dim, theta, target_device))
            {
                return false;
            }
        }
        else
        {
            // Use typed RoPE op (supports Q8_1/BF16/FP16 activation precision)
            if (!typed_rope_op_->apply(Q, K, position_ids, seq_len, n_heads, n_kv_heads, head_dim, theta,
                                       mpi_ctx_.get(), target_device))
            {
                return false;
            }
        }

        // Capture snapshots for Q and K after RoPE
        CAPTURE_SNAPSHOT_VIEW((snapshot_prefix + "_Q_ROPE").c_str(), Q, seq_len, n_heads * head_dim);
        CAPTURE_SNAPSHOT_VIEW((snapshot_prefix + "_K_ROPE").c_str(), K, seq_len, n_kv_heads * head_dim);
        return true;
    }

    bool PipelineBase::copy_tensor(const TensorBase *src, TensorBase *dst, size_t num_elements)
    {
        if (!src || !dst)
        {
            LOG_ERROR("copy_tensor: null tensor");
            return false;
        }

        // Handle Q8_1 tensor copies (must copy blocks, not FP32 data)
        auto *src_q8_1 = dynamic_cast<const Q8_1Tensor *>(src);
        auto *dst_q8_1 = dynamic_cast<Q8_1Tensor *>(dst);

        if (dst_q8_1)
        {
            // Destination is Q8_1: must use block-based copy
            if (src_q8_1)
            {
                // Q8_1 -> Q8_1: Copy blocks directly
                const Q8_1Block *src_blocks = src_q8_1->q8_1_blocks();
                Q8_1Block *dst_blocks = dst_q8_1->mutable_q8_1_blocks();
                if (!src_blocks || !dst_blocks)
                {
                    LOG_ERROR("copy_tensor: null Q8_1 block pointer");
                    return false;
                }

                // If we're copying a full 2D tensor, copy full padded rows (rows * blocks_per_row),
                // not ceil(total_elements/32). This avoids leaving tail blocks uninitialized when
                // cols is not a multiple of 32.
                size_t blocks_to_copy = 0;
                if (src_q8_1->shape().size() == 2 && dst_q8_1->shape().size() == 2)
                {
                    const size_t src_elems = src_q8_1->shape()[0] * src_q8_1->shape()[1];
                    const size_t dst_elems = dst_q8_1->shape()[0] * dst_q8_1->shape()[1];
                    if (num_elements == src_elems && num_elements <= dst_elems && src_q8_1->shape()[1] == dst_q8_1->shape()[1])
                    {
                        blocks_to_copy = src_q8_1->total_blocks();
                    }
                }

                if (blocks_to_copy == 0)
                {
                    blocks_to_copy = (num_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                }

                std::memcpy(dst_blocks, src_blocks, blocks_to_copy * sizeof(Q8_1Block));
            }
            else
            {
                // FP32 -> Q8_1: Quantize source to destination
                const float *src_data = src->data();
                Q8_1Block *dst_blocks = dst_q8_1->mutable_q8_1_blocks();
                if (!src_data || !dst_blocks)
                {
                    LOG_ERROR("copy_tensor: null pointer for FP32->Q8_1 copy");
                    return false;
                }

                // Quantizer requires count to be a multiple of 32. For partial/tail elements,
                // pad with zeros to ensure deterministic output.
                size_t padded_count = 0;
                if (dst_q8_1->shape().size() == 2)
                {
                    const size_t dst_elems = dst_q8_1->shape()[0] * dst_q8_1->shape()[1];
                    if (num_elements == dst_elems)
                    {
                        padded_count = dst_q8_1->total_blocks() * Q8_1Block::BLOCK_SIZE;
                    }
                }
                if (padded_count == 0)
                {
                    const size_t n_blocks = (num_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                    padded_count = n_blocks * Q8_1Block::BLOCK_SIZE;
                }

                if (padded_count == num_elements)
                {
                    simd::quantize_fp32_to_q8_1_blocks(src_data, dst_blocks, padded_count);
                }
                else
                {
                    std::vector<float> tmp(padded_count, 0.0f);
                    std::memcpy(tmp.data(), src_data, num_elements * sizeof(float));
                    simd::quantize_fp32_to_q8_1_blocks(tmp.data(), dst_blocks, padded_count);
                }
            }
            return true;
        }

        // Non-Q8_1 destination: use FP32 path
        // Source can be Q8_1 (data() dequantizes) or any other type
        if (!src->data())
        {
            LOG_ERROR("copy_tensor: null source data pointer");
            return false;
        }

        float *dst_data = dst->mutable_data();
        if (!dst_data)
        {
            LOG_ERROR("copy_tensor: null destination data pointer");
            return false;
        }

        std::memcpy(dst_data, src->data(), num_elements * sizeof(float));
        return true;
    }

    bool PipelineBase::save_residual(const TensorBase *input, TensorBase *residual_buffer, int seq_len, int hidden_dim)
    {
        return copy_tensor(input, residual_buffer, static_cast<size_t>(seq_len) * hidden_dim);
    }

    bool PipelineBase::compute_attention(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        int batch_size, const std::vector<int> &sequence_lengths, int padded_seq_len,
        bool causal, const std::string &snapshot_key)
    {
        LOG_TRACE("[PipelineBase::compute_attention] seq_len=" << seq_len << ", precision=" << static_cast<int>(config_.activation_precision));
        KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

        // Create views with actual effective_seq_len
        auto Q_view = Q->create_view({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
        auto K_view = K->create_view({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
        auto V_view = V->create_view({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
        auto out_view = output->create_view({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});

        if (!Q_view || !K_view || !V_view || !out_view)
        {
            LOG_ERROR("compute_attention: failed to create tensor views");
            return false;
        }

        // Determine if we need padding mask
        const std::vector<int> *seq_lens_ptr = nullptr;
        if (batch_size > 1)
        {
            bool has_padding = false;
            for (int b = 0; b < batch_size; ++b)
            {
                if (sequence_lengths[b] < padded_seq_len)
                {
                    has_padding = true;
                    break;
                }
            }
            if (has_padding)
            {
                seq_lens_ptr = &sequence_lengths;
            }
        }

        // Execute attention
        if (!attention_gqa_mpi(
                Q_view.get(), K_view.get(), V_view.get(), out_view.get(),
                n_heads, n_kv_heads, head_dim,
                causal, /*window_size=*/-1,
                batch_size, seq_lens_ptr))
        {
            LOG_ERROR("compute_attention: attention_gqa_mpi failed");
            return false;
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT_VIEW(snapshot_key.c_str(), output, seq_len, n_heads * head_dim);
        return true;
    }

    bool PipelineBase::compute_attention_with_kv_cache(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int q_seq_len, int kv_seq_len, int n_heads, int n_kv_heads, int head_dim,
        int batch_size, const std::vector<int> &sequence_lengths,
        bool causal, const std::string &snapshot_key)
    {
        KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

        // For now, when q_seq_len == kv_seq_len (prefill), use existing path
        if (q_seq_len == kv_seq_len)
        {
            // Note: compute_attention already has its own profiling, but it's fine to nest
            return compute_attention(Q, K, V, output, q_seq_len, n_heads, n_kv_heads, head_dim,
                                     batch_size, sequence_lengths, q_seq_len, causal, snapshot_key);
        }

        // Decode path: Q has fewer tokens than K/V (incremental decode with KV cache)
        // Create views with correct shapes
        auto Q_view = Q->create_view({static_cast<size_t>(q_seq_len), static_cast<size_t>(n_heads * head_dim)});
        auto K_view = K->create_view({static_cast<size_t>(kv_seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
        auto V_view = V->create_view({static_cast<size_t>(kv_seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
        auto out_view = output->create_view({static_cast<size_t>(q_seq_len), static_cast<size_t>(n_heads * head_dim)});

        if (!Q_view || !K_view || !V_view || !out_view)
        {
            LOG_ERROR("compute_attention_with_kv_cache: failed to create tensor views");
            return false;
        }

        // Determine output handling based on input type:
        // - Q8_1 input: Q8_1 native kernel outputs Q8_1 blocks directly, no conversion needed
        // - Other inputs: kernel outputs FP32, may need post-quantization
        std::shared_ptr<FP32Tensor> fp32_output_temp;
        TensorBase *effective_output = out_view.get();
        const TensorType input_type = Q->native_type();
        const TensorType output_type = output->native_type();

        // For Q8_1 input with Q8_1 output: native path writes Q8_1 directly
        // For Q8_1 input with non-Q8_1 output: would need dequant (not implemented)
        // For non-Q8_1 input with Q8_1 output: need FP32 temp + quantize
        bool needs_quantize = false;
        if (input_type == TensorType::Q8_1)
        {
            // Q8_1 native attention writes Q8_1 blocks directly to output
            // No temp buffer needed - just use the output tensor
            effective_output = out_view.get();
            needs_quantize = false; // Q8_1 kernel handles quantization internally
        }
        else if (output_type == TensorType::Q8_1)
        {
            // Non-Q8_1 input (FP32/BF16/FP16) with Q8_1 output
            // Need FP32 temp buffer, then quantize
            // Use TensorFactory for NUMA-aware allocation
            fp32_output_temp = std::shared_ptr<FP32Tensor>(
                tensor_factory_->createFP32(
                                   {static_cast<size_t>(q_seq_len), static_cast<size_t>(n_heads * head_dim)},
                                   device_idx_)
                    .release());
            effective_output = fp32_output_temp.get();
            needs_quantize = true;
        }

        // Get attention kernel from effective output tensor (FP32 for Q8_1 path)
        auto *activation_output = dynamic_cast<IActivationTensor *>(effective_output);
        if (!activation_output)
        {
            LOG_ERROR("compute_attention_with_kv_cache: output tensor does not implement IActivationTensor");
            return false;
        }

        auto attention_kernel = activation_output->createAttention();
        if (!attention_kernel)
        {
            LOG_ERROR("compute_attention_with_kv_cache: failed to create attention kernel");
            return false;
        }

        // Use the attention kernel's compute_decode method (supports asymmetric lengths)
        // For now, we need to access the underlying kernel directly since ITensorAttention
        // doesn't expose compute_decode. We'll use the raw compute with proper workspace setup.

        // Allocate workspace for attention scores [n_heads, q_seq_len, kv_seq_len]
        // Use TensorFactory for NUMA-aware allocation
        auto scores_workspace = std::shared_ptr<FP32Tensor>(
            tensor_factory_->createFP32(
                               {static_cast<size_t>(n_heads * q_seq_len * kv_seq_len)},
                               device_idx_)
                .release());

        // Build causal mask for decode: [q_seq_len, kv_seq_len]
        // For decode, we mask future positions. Since Q is at the end of the sequence,
        // all K/V positions up to and including current position should be unmasked.
        std::shared_ptr<FP32Tensor> mask_tensor = nullptr;
        if (causal)
        {
            // Use TensorFactory for NUMA-aware allocation
            mask_tensor = std::shared_ptr<FP32Tensor>(
                tensor_factory_->createFP32(
                                   {static_cast<size_t>(q_seq_len * kv_seq_len)},
                                   device_idx_)
                    .release());
            float *mask_data = mask_tensor->mutable_data();

            // For decode with KV cache:
            // Q position is at kv_seq_len - 1 (the last position)
            // All K/V positions [0, kv_seq_len-1] should be visible (mask = 0)
            // No future positions to mask since Q is at the end
            for (int q = 0; q < q_seq_len; ++q)
            {
                int q_pos = (kv_seq_len - q_seq_len) + q; // Q position in the full sequence
                for (int k = 0; k < kv_seq_len; ++k)
                {
                    // Causal: Q at position q_pos can attend to K at positions [0, q_pos]
                    float mask_val = (k <= q_pos) ? 0.0f : -std::numeric_limits<float>::infinity();
                    mask_data[q * kv_seq_len + k] = mask_val;
                }
            }
        }

        // Use CPUAttentionKernelTyped's compute_decode which handles asymmetric lengths
        // Since ITensorAttention doesn't expose compute_decode, we need to cast to the concrete type

        // IMPORTANT: Dispatch based on INPUT tensor type, not kernel type
        // This ensures we use native precision operations without unnecessary dequantization

        bool success = false;
        // input_type already defined above for output handling

        if (input_type == TensorType::Q8_1)
        {
            // Q8_1 input path: Use Q8_1 kernel with native Q8_1 blocks
            auto *q8_1_kernel = dynamic_cast<CPUAttentionKernelTyped<ActivationPrecision::Q8_1> *>(attention_kernel.get());
            if (!q8_1_kernel)
            {
                // Create Q8_1 kernel if we got a different type
                static CPUAttentionKernelTyped<ActivationPrecision::Q8_1> q8_1_kernel_static;
                q8_1_kernel = &q8_1_kernel_static;
            }

            // Cast to Q8_1 blocks - the kernel expects Q8_1Block* cast to float*
            auto *Q_q8_1 = dynamic_cast<Q8_1Tensor *>(Q_view.get());
            auto *K_q8_1 = dynamic_cast<Q8_1Tensor *>(K_view.get());
            auto *V_q8_1 = dynamic_cast<Q8_1Tensor *>(V_view.get());

            if (!Q_q8_1 || !K_q8_1 || !V_q8_1)
            {
                LOG_ERROR("compute_attention_with_kv_cache: Q8_1 input type but failed to cast views");
                return false;
            }

            // Output must also be Q8_1 for native path (kernel writes Q8_1 blocks)
            auto *out_q8_1 = dynamic_cast<Q8_1Tensor *>(effective_output);
            if (!out_q8_1)
            {
                LOG_ERROR("compute_attention_with_kv_cache: Q8_1 input requires Q8_1 output tensor");
                return false;
            }

            success = q8_1_kernel->compute_decode(
                reinterpret_cast<const float *>(Q_q8_1->q8_1_blocks()),
                reinterpret_cast<const float *>(K_q8_1->q8_1_blocks()),
                reinterpret_cast<const float *>(V_q8_1->q8_1_blocks()),
                reinterpret_cast<float *>(out_q8_1->mutable_q8_1_blocks()),
                q_seq_len, kv_seq_len, n_heads, n_kv_heads, head_dim,
                causal, /*window_size=*/-1,
                scores_workspace.get(),
                nullptr, // workspace_buffer
                nullptr, // workspace_context
                mask_tensor.get(),
                false, // use_bf16
                mpi_ctx_.get(),
                -1); // CPU device index for kernel

            if (!success)
            {
                LOG_ERROR("compute_attention_with_kv_cache: compute_decode failed (Q8_1 native)");
                return false;
            }
        }
        else if (input_type == TensorType::BF16)
        {
            // BF16 input path: Use BF16 kernel with native BF16 data
            auto *bf16_kernel = dynamic_cast<CPUAttentionKernelTyped<ActivationPrecision::BF16> *>(attention_kernel.get());
            if (!bf16_kernel)
            {
                static CPUAttentionKernelTyped<ActivationPrecision::BF16> bf16_kernel_static;
                bf16_kernel = &bf16_kernel_static;
            }

            auto *Q_bf16 = dynamic_cast<BF16Tensor *>(Q_view.get());
            auto *K_bf16 = dynamic_cast<BF16Tensor *>(K_view.get());
            auto *V_bf16 = dynamic_cast<BF16Tensor *>(V_view.get());

            if (!Q_bf16 || !K_bf16 || !V_bf16)
            {
                LOG_ERROR("compute_attention_with_kv_cache: BF16 input type but failed to cast views");
                return false;
            }

            success = bf16_kernel->compute_decode(
                reinterpret_cast<const float *>(Q_bf16->bf16_data()),
                reinterpret_cast<const float *>(K_bf16->bf16_data()),
                reinterpret_cast<const float *>(V_bf16->bf16_data()),
                effective_output->mutable_data(),
                q_seq_len, kv_seq_len, n_heads, n_kv_heads, head_dim,
                causal, /*window_size=*/-1,
                scores_workspace.get(),
                nullptr, // workspace_buffer
                nullptr, // workspace_context
                mask_tensor.get(),
                true, // use_bf16
                mpi_ctx_.get(),
                -1);

            if (!success)
            {
                LOG_ERROR("compute_attention_with_kv_cache: compute_decode failed (BF16 native)");
                return false;
            }
        }
        else if (input_type == TensorType::FP16)
        {
            // FP16 input path: Use FP16 kernel with native FP16 data
            auto *fp16_kernel = dynamic_cast<CPUAttentionKernelTyped<ActivationPrecision::FP16> *>(attention_kernel.get());
            if (!fp16_kernel)
            {
                static CPUAttentionKernelTyped<ActivationPrecision::FP16> fp16_kernel_static;
                fp16_kernel = &fp16_kernel_static;
            }

            auto *Q_fp16 = dynamic_cast<FP16Tensor *>(Q_view.get());
            auto *K_fp16 = dynamic_cast<FP16Tensor *>(K_view.get());
            auto *V_fp16 = dynamic_cast<FP16Tensor *>(V_view.get());

            if (!Q_fp16 || !K_fp16 || !V_fp16)
            {
                LOG_ERROR("compute_attention_with_kv_cache: FP16 input type but failed to cast views");
                return false;
            }

            success = fp16_kernel->compute_decode(
                reinterpret_cast<const float *>(Q_fp16->fp16_data()),
                reinterpret_cast<const float *>(K_fp16->fp16_data()),
                reinterpret_cast<const float *>(V_fp16->fp16_data()),
                effective_output->mutable_data(),
                q_seq_len, kv_seq_len, n_heads, n_kv_heads, head_dim,
                causal, /*window_size=*/-1,
                scores_workspace.get(),
                nullptr, // workspace_buffer
                nullptr, // workspace_context
                mask_tensor.get(),
                false, // use_bf16
                mpi_ctx_.get(),
                -1);

            if (!success)
            {
                LOG_ERROR("compute_attention_with_kv_cache: compute_decode failed (FP16 native)");
                return false;
            }
        }
        else
        {
            // FP32 input path (default): Use FP32 kernel with native FP32 data
            auto *fp32_kernel = dynamic_cast<CPUAttentionKernelTyped<ActivationPrecision::FP32> *>(attention_kernel.get());
            if (!fp32_kernel)
            {
                // Try legacy kernel for backwards compatibility
                auto *legacy_kernel = dynamic_cast<CpuAttentionKernelT<FP32Tensor> *>(attention_kernel.get());
                if (legacy_kernel)
                {
                    success = legacy_kernel->compute_decode(
                        Q_view->data(),
                        K_view->data(),
                        V_view->data(),
                        effective_output->mutable_data(),
                        q_seq_len, kv_seq_len, n_heads, n_kv_heads, head_dim,
                        causal, /*window_size=*/-1,
                        scores_workspace.get(),
                        nullptr,
                        nullptr,
                        mask_tensor.get(),
                        false,
                        mpi_ctx_.get(),
                        -1);
                }
                else
                {
                    static CPUAttentionKernelTyped<ActivationPrecision::FP32> fp32_kernel_static;
                    fp32_kernel = &fp32_kernel_static;
                }
            }

            if (fp32_kernel)
            {
                success = fp32_kernel->compute_decode(
                    Q_view->data(),
                    K_view->data(),
                    V_view->data(),
                    effective_output->mutable_data(),
                    q_seq_len, kv_seq_len, n_heads, n_kv_heads, head_dim,
                    causal, /*window_size=*/-1,
                    scores_workspace.get(),
                    nullptr,
                    nullptr,
                    mask_tensor.get(),
                    false,
                    mpi_ctx_.get(),
                    -1);
            }

            if (!success)
            {
                LOG_ERROR("compute_attention_with_kv_cache: compute_decode failed (FP32)");
                return false;
            }
        }

        // If we computed to a temporary FP32 buffer, quantize to Q8_1 output
        if (needs_quantize && fp32_output_temp)
        {
            auto *q8_1_output = dynamic_cast<Q8_1Tensor *>(out_view.get());
            if (!q8_1_output)
            {
                LOG_ERROR("compute_attention_with_kv_cache: expected Q8_1 output but cast failed");
                return false;
            }

            // Quantize FP32 to Q8_1
            const size_t n_elements = static_cast<size_t>(q_seq_len) * static_cast<size_t>(n_heads * head_dim);
            simd::quantize_fp32_to_q8_1_blocks(
                fp32_output_temp->data(),
                q8_1_output->mutable_q8_1_blocks(),
                n_elements);
        }

        // Capture snapshot
        CAPTURE_SNAPSHOT_VIEW(snapshot_key.c_str(), output, q_seq_len, n_heads * head_dim);
        return true;
    }

    void PipelineBase::capture_snapshot(const std::string &key, TensorBase *tensor, int rows, int cols)
    {
        CAPTURE_SNAPSHOT_VIEW(key.c_str(), tensor, rows, cols);
        maybeDumpTensor(key, tensor, rows, cols);
    }

} // namespace llaminar2
