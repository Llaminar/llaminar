/**
 * @file WeightManager.cpp
 * @brief Weight distribution and caching implementation
 * @author David Sanftenberg
 */

#include "WeightManager.h"
#include "MmapRegion.h"
#include "PreparedWeightStore.h"
#include "../execution/moe/ExpertWeightPayloadProvider.h"
#include "../utils/Logger.h"
#include "../utils/WeightLoadingProfiler.h"
#include "../utils/DebugEnv.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/TensorSlice.h"
#include "../kernels/KernelFactory.h"
#include "../backends/BackendManager.h"

// GPU weight loading pipeline
#include "gpu_pipeline/LoadOrchestrator.h"
#include "gpu_pipeline/WeightVRAMPool.h"
#include "gpu_pipeline/RepackFormat.h"

// Backend-specific GEMM kernels (for GPU pipeline kernel creation)
#ifdef HAVE_ROCM
#include "../kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "../kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h"
#endif
#ifdef HAVE_CUDA
#include "../kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "../kernels/cuda/gemm/CUDAFloatingPointGemmKernel.h"
#endif
#include "../tensors/BlockStructures.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <iomanip>
#include <thread>
#include <future>
#include <unordered_map>
#include <vector>
#ifdef __linux__
#include <malloc.h> // malloc_trim
#endif

namespace llaminar2
{

    namespace
    {
        uint64_t stableMaterializedBindingId(
            const WeightRequirement &requirement,
            const InferenceStrategy &strategy)
        {
            uint64_t hash = 1469598103934665603ull;
            auto mixByte = [&](uint8_t byte) {
                hash ^= static_cast<uint64_t>(byte);
                hash *= 1099511628211ull;
            };
            auto mixString = [&](const std::string &value) {
                for (unsigned char ch : value)
                    mixByte(ch);
                mixByte(0xffu);
            };
            auto mixInt = [&](int64_t value) {
                uint64_t bits = static_cast<uint64_t>(value);
                for (int i = 0; i < 8; ++i)
                    mixByte(static_cast<uint8_t>((bits >> (i * 8)) & 0xffu));
            };
            auto mixSize = [&](size_t value) {
                mixInt(static_cast<int64_t>(value));
            };

            mixInt(static_cast<int64_t>(strategy.model_id.value));
            mixString(requirement.canonical_name);
            mixString(requirement.source_name);
            mixInt(static_cast<int>(requirement.role));
            mixInt(static_cast<int>(requirement.derivation));
            mixInt(requirement.layer);
            mixInt(requirement.expert);
            mixInt(requirement.pp_stage);
            mixInt(requirement.tp_domain);
            mixInt(requirement.tp_rank_or_device_index);
            mixInt(static_cast<int>(requirement.target_device.type));
            mixInt(requirement.target_device.ordinal);
            mixSize(requirement.slice.row_start);
            mixSize(requirement.slice.row_count);
            mixSize(requirement.slice.col_start);
            mixSize(requirement.slice.col_count);
            mixSize(requirement.slice.expert_start);
            mixSize(requirement.slice.expert_count);

            return hash == 0 ? 1 : hash;
        }

    }

    const char *WeightManager::weightPrepStateName(WeightPrepState state)
    {
        switch (state)
        {
        case WeightPrepState::UNKNOWN:
            return "UNKNOWN";
        case WeightPrepState::LOADED_HOST:
            return "LOADED_HOST";
        case WeightPrepState::PACKED_HOST:
            return "PACKED_HOST";
        case WeightPrepState::UPLOADED_DEVICE:
            return "UPLOADED_DEVICE";
        case WeightPrepState::READY:
            return "READY";
        case WeightPrepState::FAILED:
            return "FAILED";
        default:
            return "INVALID";
        }
    }

    void WeightManager::registerExpectedDeviceForWeight(const std::string &name, DeviceId device)
    {
        std::lock_guard<std::mutex> lock(prep_ticket_mutex_);
        expected_devices_by_weight_[name].insert(device.to_string());
    }

    void WeightManager::markPrepState(const std::string &name,
                                      DeviceId device,
                                      WeightPrepState state,
                                      bool is_gemm,
                                      const std::string &detail)
    {
        std::lock_guard<std::mutex> lock(prep_ticket_mutex_);

        const std::string device_key = device.to_string();
        expected_devices_by_weight_[name].insert(device_key);

        auto &ticket = prep_tickets_[name][device_key];
        ticket.state = state;
        ticket.is_gemm = is_gemm;
        ticket.detail = detail;
    }

    void WeightManager::evaluateReclaimEligibility(const std::string &name, bool is_gemm)
    {
        if (!is_gemm)
        {
            return;
        }

        bool should_attempt_reclaim = false;

        {
            std::lock_guard<std::mutex> lock(prep_ticket_mutex_);

            auto exp_it = expected_devices_by_weight_.find(name);
            auto ticket_it = prep_tickets_.find(name);
            if (exp_it == expected_devices_by_weight_.end() || ticket_it == prep_tickets_.end())
            {
                return;
            }

            const auto &expected = exp_it->second;
            const auto &tickets = ticket_it->second;

            if (expected.empty())
            {
                return;
            }

            const bool all_ready = std::all_of(expected.begin(), expected.end(), [&](const std::string &device_key)
                                               {
                auto it = tickets.find(device_key);
                return it != tickets.end() && it->second.state == WeightPrepState::READY; });

            if (!all_ready)
            {
                return;
            }

            reclaim_ready_weights_.insert(name);
            if (reclaim_applied_weights_.find(name) == reclaim_applied_weights_.end())
            {
                should_attempt_reclaim = true;
            }
        }

        if (!should_attempt_reclaim)
        {
            return;
        }

        if (tryReleaseReclaimHostRawData(name))
        {
            std::lock_guard<std::mutex> lock(prep_ticket_mutex_);
            reclaim_applied_weights_.insert(name);
            LOG_DEBUG("[WeightManager][Phase2] Reclaimed GEMM host raw data: " << name);
        }
    }

    bool WeightManager::tryReleaseReclaimHostRawData(const std::string &name)
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        std::unordered_set<TensorBase *> released;
        bool found_any = false;

        // Helper: safely release a tensor's host data only if it has a device
        // copy (GPU upload or kernel-managed packed data). Floating-point CPU
        // GEMM kernels (oneDNN) read weight_tensor_->data() live at every
        // inference call, so host data must be retained for CPU-only weights.
        auto safe_release = [&](TensorBase *ptr, const std::string &key)
        {
            if (!ptr || !released.insert(ptr).second)
                return;

            if (ptr->is_raw_data_released())
                return; // Already released

            // Same safety check as releaseAllHostWeightData: only release if
            // the tensor has a valid GPU copy OR kernel-managed device data.
            if (!ptr->deviceValid())
            {
                bool has_kernel_device_data = ptr->hasPreparedDeviceState();

                if (!has_kernel_device_data)
                {
                    LOG_DEBUG("[WeightManager][Phase2] Skipped reclaim for host-only weight: " << key
                                                                                               << " (no device copy)");
                    return;
                }
            }

            ptr->release_host_weight_data();
            found_any = true;
        };

        auto base_it = cache_.find(name);
        if (base_it != cache_.end() && base_it->second)
        {
            safe_release(base_it->second.get(), name);
        }

        const std::string suffix = ":" + name;
        for (const auto &[key, tensor] : per_device_cache_)
        {
            if (!tensor)
            {
                continue;
            }
            if (key.size() >= suffix.size() &&
                key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0)
            {
                safe_release(tensor.get(), key);
            }
        }

        if (!found_any)
        {
            LOG_DEBUG("[WeightManager][Phase2] Reclaim requested but no reclaimable tensors for: " << name);
        }

        return found_any;
    }

    WeightManager::WeightManager(IModelLoader &loader,
                                 std::shared_ptr<IMPIContext> mpi_ctx,
                                 std::shared_ptr<WeightPlacementMap> placement_map,
                                 WeightDistributionStrategy strategy,
                                 WeightPrecision weight_precision)
        : loader_(loader), mpi_ctx_(mpi_ctx), placement_map_(placement_map),
                    strategy_(strategy), weight_precision_(weight_precision),
                    weight_metadata_(std::make_shared<WeightMetadataRegistry>())
    {
        int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;

        if (rank == 0)
        {
            std::string strategy_name;
            switch (strategy_)
            {
            case WeightDistributionStrategy::REPLICATED:
                strategy_name = "REPLICATED (full copy per rank)";
                break;
            case WeightDistributionStrategy::SHARDED:
                strategy_name = "SHARDED (partitioned across ranks)";
                break;
            case WeightDistributionStrategy::INTERLEAVED:
                strategy_name = "INTERLEAVED (NUMA-aware global)";
                break;
            case WeightDistributionStrategy::LAYER_PARTITIONED:
                strategy_name = "LAYER_PARTITIONED (PP: only assigned layers)";
                break;
            }
            LOG_DEBUG("[WeightManager] Initialized with strategy: " << strategy_name);

            // Log weight precision mode
            const char *precision_name = "UNKNOWN";
            switch (weight_precision_)
            {
            case WeightPrecision::NATIVE:
                precision_name = "NATIVE (weights in original GGUF format, dequantize on-the-fly)";
                break;
            case WeightPrecision::CONVERT_TO_FP32:
                precision_name = "CONVERT_TO_FP32 (all weights dequantized to FP32 at load)";
                break;
            case WeightPrecision::CONVERT_TO_BF16:
                precision_name = "CONVERT_TO_BF16 (all weights dequantized to BF16 at load)";
                break;
            case WeightPrecision::CONVERT_TO_FP16:
                precision_name = "CONVERT_TO_FP16 (all weights dequantized to FP16 at load)";
                break;
            case WeightPrecision::CONVERT_TO_INT8:
                precision_name = "CONVERT_TO_INT8 (all weights dequantized to INT8 at load)";
                break;
            }
            LOG_DEBUG("[WeightManager] Weight precision: " << precision_name);

            if (strategy_ == WeightDistributionStrategy::SHARDED && mpi_ctx_)
            {
                LOG_DEBUG("[WeightManager] Sharding enabled with " << mpi_ctx_->world_size() << " ranks");
                LOG_DEBUG("[WeightManager] Column-parallel: Gate/Up (split output dim, local output)");
                LOG_DEBUG("[WeightManager] Input-parallel: Down (split input dim to match Gate/Up, allreduce after)");
                LOG_DEBUG("[WeightManager] Input-parallel: Wo (split input dim to match column-parallel QKV, allreduce after)");
                LOG_DEBUG("[WeightManager] Replicated: QKV (column-parallel via head sharding), norms, biases, embeddings, LM head");
            }
        }
    }

    WeightSliceSpec WeightManager::fullSliceSpec(const TensorBase &tensor) const
    {
        WeightSliceSpec spec;
        const auto &shape = tensor.shape();
        spec.source_rows = shape.empty() ? 0 : shape[0];
        spec.source_cols = shape.size() > 1 ? shape[1] : 1;
        spec.row_start = 0;
        spec.row_count = spec.source_rows;
        spec.col_start = 0;
        spec.col_count = spec.source_cols;
        if (shape.size() > 2)
        {
            spec.expert_start = 0;
            spec.expert_count = shape[2];
        }
        return spec;
    }

    void WeightManager::registerSourceMetadata(
        const std::string &name,
        const std::shared_ptr<TensorBase> &tensor,
        DeviceId device)
    {
        if (!tensor || !weight_metadata_)
            return;

        if (!weight_metadata_->has(tensor.get()))
            weight_metadata_->registerSource(tensor.get(), name, device);

        WeightLifecycleTrace::record(
            WeightLifecycleEventType::SourceLoad,
            name,
            inferWeightRole(name),
            inferWeightLayer(name),
            device,
            "source tensor loaded");
    }

    void WeightManager::registerDerivedMetadata(
        const std::string &name,
        const std::shared_ptr<TensorBase> &tensor,
        WeightDerivationKind derivation,
        WeightSliceSpec slice,
        DeviceId device)
    {
        if (!tensor || !weight_metadata_)
            return;

        WeightIdentity identity = makeSourceWeightIdentity(name);
        identity.derivation = derivation;
        weight_metadata_->registerWeight(tensor.get(), std::move(identity), slice, WeightResidency{device, device});

        WeightLifecycleTrace::record(
            derivation == WeightDerivationKind::DeviceClone ? WeightLifecycleEventType::Clone : WeightLifecycleEventType::Slice,
            name,
            inferWeightRole(name),
            inferWeightLayer(name),
            device,
            toString(derivation));
    }

    void WeightManager::registerCloneMetadata(
        const std::string &name,
        const std::shared_ptr<TensorBase> &source,
        const std::shared_ptr<TensorBase> &clone,
        DeviceId device)
    {
        if (!source || !clone || !weight_metadata_)
            return;

        if (!weight_metadata_->has(source.get()))
            registerSourceMetadata(name, source, DeviceId::cpu());

        if (!weight_metadata_->registerDerived(
                clone.get(), source.get(), WeightDerivationKind::DeviceClone, fullSliceSpec(*source), device))
        {
            registerDerivedMetadata(name, clone, WeightDerivationKind::DeviceClone, fullSliceSpec(*source), device);
        }

        WeightLifecycleTrace::record(
            WeightLifecycleEventType::Clone,
            name,
            inferWeightRole(name),
            inferWeightLayer(name),
            device,
            "device clone");
    }

    bool WeightManager::isGemmWeight(const std::string &name) const
    {
        if (!has_sharding_config_)
        {
            throw std::runtime_error(
                "[WeightManager] isGemmWeight called without sharding config. "
                "Call setWeightShardingConfig() with model schema first.");
        }
        return !sharding_config_.isNonGemmWeight(name);
    }

    std::shared_ptr<TensorBase> WeightManager::getReplicatedWeight(const std::string &name, DeviceId device)
    {
        // Phase 1: Simple replication - each rank loads independently
        // No MPI coordination needed

        auto tensor = loader_.loadTensor(name, device, weight_precision_);
        if (!tensor)
        {
            // LOG_DEBUG, not LOG_ERROR: returning nullptr is a valid result for optional weights.
            // The caller (InferenceRunnerFactory) decides severity based on schema context.
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            LOG_DEBUG("[WeightManager] Rank " << rank << " weight not found: " << name);
            return nullptr;
        }

        registerSourceMetadata(name, tensor, device);
        return tensor;
    }

    ShardingMode WeightManager::toShardingMode(WeightShardingMode mode)
    {
        switch (mode)
        {
        case WeightShardingMode::ColumnParallel:
            return ShardingMode::COLUMN_PARALLEL;
        case WeightShardingMode::RowParallel:
            return ShardingMode::ROW_PARALLEL;
        case WeightShardingMode::InputParallel:
            return ShardingMode::INPUT_PARALLEL;
        case WeightShardingMode::ExpertParallel:
            return ShardingMode::EXPERT_PARALLEL;
        case WeightShardingMode::Replicate:
        default:
            return ShardingMode::REPLICATE;
        }
    }

    ShardingMode WeightManager::determineShardingMode(const std::string &name) const
    {
        if (!has_sharding_config_)
        {
            throw std::runtime_error(
                "[WeightManager] determineShardingMode called without sharding config. "
                "Call setWeightShardingConfig() with model schema first.");
        }
        WeightShardingMode mode = sharding_config_.getMode(name);
        return toShardingMode(mode);
    }

    ShardingMode WeightManager::getShardingMode(const std::string &name) const
    {
        std::lock_guard<std::mutex> lock(sharding_mode_cache_mutex_);

        // Check cache first
        auto it = sharding_mode_cache_.find(name);
        if (it != sharding_mode_cache_.end())
        {
            return it->second;
        }

        // Determine and cache (cache is mutable)
        ShardingMode mode = determineShardingMode(name);
        sharding_mode_cache_[name] = mode;
        return mode;
    }

    bool WeightManager::isWeightSharded(const std::string &name) const
    {
        if (strategy_ != WeightDistributionStrategy::SHARDED)
        {
            return false;
        }
        ShardingMode mode = getShardingMode(name);
        return mode != ShardingMode::REPLICATE;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceColumns(
        const std::shared_ptr<TensorBase> &full_tensor,
        int rank, int world_size)
    {
        // For weight matrix [out_dim, in_dim], extract [out_local, in_dim]
        // where out_local = out_dim / world_size for this rank

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Column slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];
        size_t out_local = out_dim / world_size;
        size_t out_start = out_local * rank;

        // Handle remainder: last rank gets extra rows if not evenly divisible
        if (rank == world_size - 1)
        {
            out_local = out_dim - out_start;
        }

        LOG_DEBUG("[WeightManager] Column slicing: [" << out_dim << ", " << in_dim
                                                      << "] -> rank " << rank << " gets rows [" << out_start << ", "
                                                      << (out_start + out_local) << ")");

        // Currently only FP32 slicing is supported
        // For quantized tensors, we dequantize first then slice
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Column slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for sharded strategy.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_local, in_dim};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy data rows
        const float *src = fp32_tensor->data() + out_start * in_dim;
        float *dst = sliced->mutable_data();
        std::memcpy(dst, src, out_local * in_dim * sizeof(float));

        return sliced;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceRows(
        const std::shared_ptr<TensorBase> &full_tensor,
        int rank, int world_size)
    {
        // For weight matrix [out_dim, in_dim], extract [out_dim, in_local]
        // where in_local = in_dim / world_size for this rank

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Row slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];
        size_t in_local = in_dim / world_size;
        size_t in_start = in_local * rank;

        // Handle remainder: last rank gets extra columns if not evenly divisible
        if (rank == world_size - 1)
        {
            in_local = in_dim - in_start;
        }

        LOG_DEBUG("[WeightManager] Row slicing: [" << out_dim << ", " << in_dim
                                                   << "] -> rank " << rank << " gets cols [" << in_start << ", "
                                                   << (in_start + in_local) << ")");

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Row slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for sharded strategy.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_dim, in_local};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy data - need to extract columns, which is non-contiguous
        const float *src = fp32_tensor->data();
        float *dst = sliced->mutable_data();

#pragma omp parallel for
        for (size_t row = 0; row < out_dim; ++row)
        {
            const float *src_row = src + row * in_dim + in_start;
            float *dst_row = dst + row * in_local;
            std::memcpy(dst_row, src_row, in_local * sizeof(float));
        }

        return sliced;
    }

    bool WeightManager::hostDataRequired(const TensorBase *tensor, const std::string &key) const
    {
        // Phase 9 final: All expert GEMM preparation happens upfront at graph-build
        // time. No lazy host-data fallback exists. Expert weights are always
        // releasable after device preparation completes. Dynamic rebalancing uses
        // serialized transfer blobs, never raw host tensor data.
        //
        // The only case where host data must be retained is explicit CPU execution
        // (RequiredForCPUExecution policy in metadata).

        if (weight_metadata_ && weight_metadata_->has(tensor))
        {
            auto res = weight_metadata_->residency(tensor);
            if (res)
            {
                switch (res->host_policy)
                {
                case WeightHostPolicy::RequiredForCPUExecution:
                    return true;
                case WeightHostPolicy::Released:
                case WeightHostPolicy::ReleasableAfterPreparation:
                case WeightHostPolicy::RequiredUntilPreparedOrTransferred:
                case WeightHostPolicy::RequiredUntilGraphMaterialized:
                    return false;
                }
            }
        }

        // No metadata or no explicit policy: host data is not required.
        // All preparation (GEMM packing, embedding prep, GPU upload) completes
        // before releaseAllHostWeightData() is called.
        return false;
    }

    namespace
    {

        /**
         * @brief Create a typed tensor from raw bytes (generic for all tensor types)
         *
         * Handles FP32, FP16, BF16, and all quantized tensor types.
         * Raw bytes are interpreted according to the tensor type's native storage format.
         */
        std::unique_ptr<TensorBase> createTensorFromRawData(
            TensorType type,
            const std::vector<size_t> &shape,
            std::vector<uint8_t> raw_data)
        {
            switch (type)
            {
            case TensorType::FP32:
            {
                auto tensor = std::make_unique<FP32Tensor>(shape);
                std::memcpy(tensor->mutable_data(), raw_data.data(), raw_data.size());
                return tensor;
            }
            case TensorType::Q4_0:
                return std::make_unique<Q4_0Tensor>(shape, raw_data);
            case TensorType::Q8_0:
                return std::make_unique<Q8_0Tensor>(shape, raw_data);
            case TensorType::Q8_1:
                return std::make_unique<Q8_1Tensor>(shape, raw_data);
            case TensorType::Q4_1:
                return std::make_unique<Q4_1Tensor>(shape, raw_data);
            case TensorType::Q5_0:
                return std::make_unique<Q5_0Tensor>(shape, raw_data);
            case TensorType::Q5_1:
                return std::make_unique<Q5_1Tensor>(shape, raw_data);
            case TensorType::Q6_K:
                return std::make_unique<Q6_KTensor>(shape, raw_data);
            case TensorType::Q2_K:
                return std::make_unique<Q2_KTensor>(shape, raw_data);
            case TensorType::Q5_K:
                return std::make_unique<Q5_KTensor>(shape, raw_data);
            case TensorType::Q3_K:
                return std::make_unique<Q3_KTensor>(shape, raw_data);
            case TensorType::Q4_K:
                return std::make_unique<Q4_KTensor>(shape, raw_data);
            case TensorType::Q8_K:
                return std::make_unique<Q8_KTensor>(shape, raw_data);
            case TensorType::IQ4_NL:
                return std::make_unique<IQ4_NLTensor>(shape, raw_data);
            case TensorType::IQ4_XS:
                return std::make_unique<IQ4_XSTensor>(shape, raw_data);
            case TensorType::IQ2_XXS:
                return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
            case TensorType::IQ2_XS:
                return std::make_unique<IQ2_XSTensor>(shape, raw_data);
            case TensorType::IQ3_XXS:
                return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
            case TensorType::IQ2_S:
                return std::make_unique<IQ2_STensor>(shape, raw_data);
            case TensorType::IQ3_S:
                return std::make_unique<IQ3_STensor>(shape, raw_data);
            case TensorType::IQ1_S:
                return std::make_unique<IQ1_STensor>(shape, raw_data);
            case TensorType::IQ1_M:
                return std::make_unique<IQ1_MTensor>(shape, raw_data);
            case TensorType::BF16:
            {
                std::vector<uint16_t> bf16_data(raw_data.size() / 2);
                std::memcpy(bf16_data.data(), raw_data.data(), raw_data.size());
                return std::make_unique<BF16Tensor>(shape, bf16_data);
            }
            case TensorType::FP16:
            {
                std::vector<uint16_t> fp16_data(raw_data.size() / 2);
                std::memcpy(fp16_data.data(), raw_data.data(), raw_data.size());
                return std::make_unique<FP16Tensor>(shape, fp16_data);
            }
            default:
                throw std::runtime_error("[WeightManager] Unsupported tensor type for createTensorFromRawData: " +
                                         std::to_string(static_cast<int>(type)));
            }
        }
    } // anonymous namespace

    std::shared_ptr<TensorBase> WeightManager::getShardedWeight(const std::string &name, DeviceId device)
    {
        // For SHARDED strategy:
        // 1. Determine sharding mode based on weight name
        // 2. Load full tensor in native format (quantized)
        // 3. Wrap in TensorSlice with sharding metadata
        // 4. Slicing happens during kernel creation (preserves quantization)

        if (!mpi_ctx_ || mpi_ctx_->world_size() == 1)
        {
            // Single rank: no sharding needed
            return getReplicatedWeight(name, device);
        }

        ShardingMode mode = getShardingMode(name);
        int rank = mpi_ctx_->rank();
        int world_size = mpi_ctx_->world_size();

        if (mode == ShardingMode::REPLICATE)
        {
            // This weight should not be sharded (norms, biases, etc.)
            return getReplicatedWeight(name, device);
        }

        if (mode == ShardingMode::ROW_PARALLEL)
        {
            // Row-parallel: Load ONLY the slice from GGUF (memory efficient!)
            // Get tensor dimensions
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for row-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " invalid tensor for row-parallel (expected 2D): " + name);
            }
            const auto &dims = *dims_opt;

            size_t total_rows = dims[0];
            size_t cols = dims[1];

            // Calculate row range for this rank
            // Use proportional slicing if TensorParallelConfig is set, otherwise equal split
            size_t row_start, row_count;
            if (tp_config_)
            {
                // Row-parallel actually splits ROWS (for Wo, which has shape [d_model, n_heads*head_dim])
                // But the input dimension to match is from the column-parallel output
                // For Wo: we split rows based on head assignment since input comes from Q output
                auto [start, count] = calculateProportionalRowSlice(name, total_rows);
                row_start = start;
                row_count = count;
            }
            else
            {
                size_t rows_per_rank = total_rows / world_size;
                row_start = rows_per_rank * rank;
                row_count = (rank == world_size - 1) ? (total_rows - row_start) : rows_per_rank;
            }
            size_t row_end = row_start + row_count;

            // Load ONLY the slice from GGUF file (memory efficient!)
            auto slice_tensor = loader_.loadTensorRowSlice(
                name, row_start, row_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load row slice for: " + name);
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " row-parallel " << name
                                              << " [" << total_rows << ", " << cols
                                              << "] -> loaded ONLY rows [" << row_start << ", " << row_end
                                              << ") = " << row_count << " rows"
                                              << (tp_config_ ? " (proportional)" : " (equal)"));

            // Wrap in TensorSlice with metadata (inner_is_presliced=true)
            // The inner tensor already contains only the slice data
            auto meta = SliceMetadata::forRowParallel(
                total_rows, cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
            WeightSliceSpec row_slice;
            row_slice.source_rows = total_rows;
            row_slice.source_cols = cols;
            row_slice.row_start = row_start;
            row_slice.row_count = row_count;
            row_slice.col_start = 0;
            row_slice.col_count = cols;
            row_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice, WeightDerivationKind::RowSlice, row_slice, device);
            return slice;
        }
        else if (mode == ShardingMode::COLUMN_PARALLEL)
        {
            // Column-parallel: split the OUTPUT dimension of the weight matrix
            //
            // Terminology clarification:
            // - "Column-parallel" in tensor parallelism means each rank produces independent
            //   columns of the OUTPUT tensor (Y = X @ W^T)
            // - For weight matrix W of shape [N, K] (N=output features, K=input features):
            //   * Column-parallel splits N (the rows of W), NOT K (the columns)
            //   * Each rank loads rows [row_start, row_end) of W
            //   * GEMM output is [M, N/world_size], then concatenated via AllGather
            //
            // For FFN Gate/Up weights [d_ff=4864, d_model=896]:
            //   * Each rank loads [d_ff/2=2432, d_model=896]
            //   * GEMM input: [seq, d_model=896]
            //   * GEMM output: [seq, d_ff/2=2432]
            //
            // For 1D bias tensors [N] (e.g., Q/K/V biases):
            //   * Each rank loads elements [start, end) where N is split across ranks

            int rank = mpi_ctx_->rank();
            int world_size = mpi_ctx_->world_size();

            // Get tensor dimensions
            // For tied embeddings: if output.weight is missing, substitute token_embd.weight
            std::string load_name = name;
            auto dims_opt = loader_.getTensorShape(name);
            if ((!dims_opt || dims_opt->empty()) && name == "output.weight")
            {
                auto embd_dims = loader_.getTensorShape("token_embd.weight");
                if (embd_dims && embd_dims->size() == 2)
                {
                    LOG_INFO("[WeightManager] Rank " << rank
                                                     << " output.weight not in GGUF — using tied embedding "
                                                     << "token_embd.weight as column-parallel LM head");
                    dims_opt = embd_dims;
                    load_name = "token_embd.weight";
                }
            }
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for column-parallel: " << name);
                return nullptr;
            }
            const auto &dims = *dims_opt;

            // Handle 1D tensors (biases)
            // For biases like Q/K/V biases [n_heads * head_dim], we load full then slice in memory.
            // This is acceptable since biases are tiny (e.g., 896 floats = 3.5KB).
            if (dims.size() == 1)
            {
                size_t total_elements = dims[0];

                // Calculate slice range - use proportional if config set
                size_t start, slice_elements;
                if (tp_config_)
                {
                    // Use column slice calculation (1D bias treated as single-row 2D)
                    auto [s, count] = calculateProportionalColumnSlice(name, total_elements);
                    start = s;
                    slice_elements = count;
                }
                else
                {
                    size_t elements_per_rank = total_elements / world_size;
                    start = elements_per_rank * rank;
                    slice_elements = (rank == world_size - 1) ? (total_elements - start) : elements_per_rank;
                }
                size_t end = start + slice_elements;

                // Load full 1D tensor first (biases are small, ~3KB)
                auto full_tensor = loader_.loadTensor(name, device, WeightPrecision::NATIVE);
                if (!full_tensor)
                {
                    throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                             " failed to load 1D tensor for: " + name);
                }

                // Cast to FP32 (biases are stored as FP32)
                auto *fp32_full = dynamic_cast<FP32Tensor *>(full_tensor.get());
                if (!fp32_full)
                {
                    throw std::runtime_error("[WeightManager] 1D tensor is not FP32 for: " + name);
                }

                // Create sliced 1D tensor
                auto slice_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{slice_elements});
                std::memcpy(slice_tensor->mutable_data(),
                            fp32_full->data() + start,
                            slice_elements * sizeof(float));

                LOG_TRACE("[WeightManager] Rank " << rank << " column-parallel 1D " << name
                                                  << " [" << total_elements
                                                  << "] -> sliced elements [" << start << ", " << end
                                                  << ") = " << slice_elements << " elements"
                                                  << (tp_config_ ? " (proportional)" : " (equal)"));

                // For 1D tensors, use row-parallel metadata with cols=1
                auto meta = SliceMetadata::forRowParallel(
                    total_elements, 1, rank, world_size,
                    true /* inner_is_presliced */);

                auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
                return slice;
            }

            // Handle 2D tensors (weight matrices)
            if (dims.size() != 2)
            {
                throw std::runtime_error("[WeightManager] Column-parallel requires 1D or 2D tensor, got " +
                                         std::to_string(dims.size()) + "D for: " + name);
            }

            size_t total_rows = dims[0]; // N = output features
            size_t cols = dims[1];       // K = input features

            // FusedQKVHeads requires special sub-block slicing: the weight is
            // [Q_all | K_all | V_all] vertically. A simple contiguous row slice
            // crosses sub-block boundaries. Instead, split each sub-block independently.
            //
            // GQA support: Q, K, V sub-blocks may have DIFFERENT sizes when
            // n_kv_heads < n_heads. We compute actual sub-block sizes from
            // model dimensions when available.
            //
            // GDN (non-QKV) weights tagged FusedQKVHeads may have total_rows
            // that doesn't match the expected Q+K+V size. These fall back to
            // simple equal row splitting.
            if (has_sharding_config_ &&
                sharding_config_.getDimensionType(name) == WeightDimensionType::FusedQKVHeads)
            {
                // Compute expected Q, K, V row counts from model dimensions
                bool use_sub_block_slicing = false;
                size_t q_rows = 0, kv_rows = 0, v_rows = 0;

                if (has_model_dimensions_ && model_n_heads_ > 0 && model_head_dim_ > 0)
                {
                    q_rows = static_cast<size_t>(model_n_heads_) * model_head_dim_;
                    kv_rows = static_cast<size_t>(model_n_kv_heads_) * model_head_dim_;
                    const size_t expected_qkv = q_rows + 2 * kv_rows;

                    if (total_rows == expected_qkv)
                    {
                        // True GQA layout: [Q(n_heads*hd) | K(n_kv_heads*hd) | V(n_kv_heads*hd)]
                        v_rows = kv_rows;
                        use_sub_block_slicing = true;
                        LOG_TRACE("[WeightManager] FusedQKV " << name
                                                              << " Q=" << q_rows << " K=" << kv_rows << " V=" << v_rows
                                                              << " (GQA: n_heads=" << model_n_heads_
                                                              << " n_kv_heads=" << model_n_kv_heads_ << ")");
                    }
                    else if (total_rows % 3 == 0)
                    {
                        // Some models (e.g. Qwen3.5) store K/V at full n_heads size in the
                        // fused weight, applying GQA only at attention time. Fall back to
                        // 3 equal sub-blocks.
                        q_rows = total_rows / 3;
                        kv_rows = total_rows / 3;
                        v_rows = total_rows / 3;
                        use_sub_block_slicing = true;
                        LOG_TRACE("[WeightManager] FusedQKV " << name
                                                              << " total_rows=" << total_rows
                                                              << " != expected GQA=" << expected_qkv
                                                              << " but divisible by 3 — using 3 equal sub-blocks"
                                                              << " (" << q_rows << " each)");
                    }
                    else
                    {
                        // Try GDN layout: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
                        // GDN has asymmetric Q/K/V when n_k_heads != n_v_heads
                        if (has_gdn_dimensions_ && gdn_n_k_heads_ > 0 && gdn_d_state_ > 0)
                        {
                            const size_t gdn_key_dim = static_cast<size_t>(gdn_n_k_heads_) * gdn_d_state_;
                            const size_t gdn_value_dim = static_cast<size_t>(gdn_n_v_heads_) * gdn_d_state_;
                            const size_t expected_gdn_qkv = 2 * gdn_key_dim + gdn_value_dim;

                            if (total_rows == expected_gdn_qkv)
                            {
                                q_rows = gdn_key_dim;   // Q = n_k_heads * d_state
                                kv_rows = gdn_key_dim;  // K = n_k_heads * d_state (same as Q)
                                v_rows = gdn_value_dim; // V = n_v_heads * d_state (may differ from K)
                                use_sub_block_slicing = true;

                                LOG_TRACE("[WeightManager] FusedQKV " << name
                                                                      << " matched GDN layout: Q=" << q_rows
                                                                      << " K=" << kv_rows << " V=" << gdn_value_dim
                                                                      << " (n_k=" << gdn_n_k_heads_
                                                                      << " n_v=" << gdn_n_v_heads_
                                                                      << " d=" << gdn_d_state_ << ")");
                            }
                            else
                            {
                                LOG_DEBUG("[WeightManager] FusedQKV " << name
                                                                      << " total_rows=" << total_rows
                                                                      << " != expected Q+K+V=" << expected_qkv
                                                                      << " and not GDN=" << expected_gdn_qkv
                                                                      << " and not divisible by 3"
                                                                      << " — using simple equal row split");
                            }
                        }
                        else
                        {
                            LOG_DEBUG("[WeightManager] FusedQKV " << name
                                                                  << " total_rows=" << total_rows
                                                                  << " != expected Q+K+V=" << expected_qkv
                                                                  << " and not divisible by 3"
                                                                  << " — using simple equal row split (non-standard fused weight)");
                        }
                    }
                }
                else
                {
                    // No model dimensions: try 3 equal sub-blocks (non-GQA fallback)
                    if (total_rows % 3 == 0)
                    {
                        q_rows = total_rows / 3;
                        kv_rows = total_rows / 3;
                        v_rows = total_rows / 3;
                        use_sub_block_slicing = true;
                    }
                    else
                    {
                        LOG_DEBUG("[WeightManager] FusedQKV " << name
                                                              << " total_rows=" << total_rows
                                                              << " not divisible by 3 and no model dimensions"
                                                              << " — using simple equal row split");
                    }
                }

                if (use_sub_block_slicing)
                {
                    // Sub-block sizes: [Q_rows, K_rows, V_rows]
                    // For GDN layers, V may differ from K (n_v_heads != n_k_heads)
                    const size_t sub_block_sizes[3] = {q_rows, kv_rows, v_rows > 0 ? v_rows : kv_rows};

                    // GDN modular repeat (repeat_type=1): v_head j uses k_head j%n_k.
                    // With contiguous V sharding, v_heads on one rank may need k_heads
                    // from another rank. Fix: replicate Q and K sub-blocks on every rank,
                    // only shard V. This ensures all k_heads are locally available.
                    const bool gdn_replicate_qk = has_gdn_dimensions_ && gdn_n_k_heads_ > 0 && gdn_n_v_heads_ > gdn_n_k_heads_ && (total_rows == 2 * static_cast<size_t>(gdn_n_k_heads_) * gdn_d_state_ + static_cast<size_t>(gdn_n_v_heads_) * gdn_d_state_);

                    // Compute per-rank slice within each sub-block
                    // For GDN with replicated Q/K: Q and K are loaded in full, only V is sharded.
                    auto compute_slice = [rank, world_size, gdn_replicate_qk](size_t block_rows, int sub_block_idx) -> std::pair<size_t, size_t>
                    {
                        // Sub-blocks 0 (Q) and 1 (K): replicate for GDN, shard otherwise
                        if (gdn_replicate_qk && sub_block_idx < 2)
                            return {0, block_rows}; // Full sub-block (replicated)

                        size_t rows_per_rank = block_rows / world_size;
                        size_t start = rows_per_rank * rank;
                        size_t count = (rank == world_size - 1)
                                           ? (block_rows - start)
                                           : rows_per_rank;
                        return {start, count};
                    };

                    // Validate sub-block divisibility BEFORE loading slices.
                    // If any sharded sub-block has fewer rows than the world_size,
                    // the model is too small for this TP degree — fail early and clearly.
                    {
                        static constexpr const char *sub_names[3] = {"Q", "K", "V"};
                        for (size_t s = 0; s < 3; s++)
                        {
                            // Replicated sub-blocks (GDN Q/K) are always valid
                            if (gdn_replicate_qk && s < 2)
                                continue;

                            if (sub_block_sizes[s] % static_cast<size_t>(world_size) != 0)
                            {
                                std::ostringstream err;
                                err << "[WeightManager] Cannot shard FusedQKV weight '" << name
                                    << "': sub-block " << sub_names[s]
                                    << " has " << sub_block_sizes[s]
                                    << " rows, which is not evenly divisible by TP degree "
                                    << world_size << ". ";
                                if (gdn_replicate_qk)
                                    err << "GDN modular repeat replicates Q/K, but V ("
                                        << sub_block_sizes[2] << " rows = "
                                        << gdn_n_v_heads_ << " heads * " << gdn_d_state_
                                        << " d_state) must be divisible by " << world_size << ". ";
                                err << "Reduce the TP degree or use a larger model.";
                                LOG_ERROR(err.str());
                                throw std::invalid_argument(err.str());
                            }

                            if (sub_block_sizes[s] / static_cast<size_t>(world_size) == 0)
                            {
                                std::ostringstream err;
                                err << "[WeightManager] Cannot shard FusedQKV weight '" << name
                                    << "': sub-block " << sub_names[s]
                                    << " has " << sub_block_sizes[s]
                                    << " rows but TP degree is " << world_size
                                    << " — each rank would get 0 rows. "
                                    << "Reduce the TP degree or use a larger model.";
                                LOG_ERROR(err.str());
                                throw std::invalid_argument(err.str());
                            }
                        }
                    }

                    // Load each sub-block's slice from GGUF
                    std::shared_ptr<TensorBase> slices[3];
                    size_t total_out_rows = 0;
                    size_t abs_offset = 0; // Running offset into the weight

                    for (size_t s = 0; s < 3; s++)
                    {
                        auto [local_start, local_count] = compute_slice(sub_block_sizes[s], static_cast<int>(s));
                        const size_t abs_row_start = abs_offset + local_start;
                        const size_t abs_row_end = abs_row_start + local_count;

                        slices[s] = loader_.loadTensorRowSlice(
                            load_name, abs_row_start, abs_row_end, device, WeightPrecision::NATIVE);

                        if (!slices[s])
                        {
                            throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                                     " failed to load fused-QKV sub-block " + std::to_string(s) +
                                                     " for: " + name);
                        }

                        total_out_rows += local_count;
                        abs_offset += sub_block_sizes[s];
                    }

                    // Concatenate raw bytes from the 3 sub-block slices
                    size_t total_bytes = 0;
                    for (size_t s = 0; s < 3; s++)
                        total_bytes += slices[s]->size_bytes();

                    std::vector<uint8_t> combined_raw(total_bytes);
                    size_t byte_offset = 0;
                    for (size_t s = 0; s < 3; s++)
                    {
                        const void *src = slices[s]->raw_data();
                        if (!src)
                        {
                            throw std::runtime_error("[WeightManager] Null raw_data for fused-QKV sub-block " +
                                                     std::to_string(s) + ": " + name);
                        }
                        std::memcpy(combined_raw.data() + byte_offset, src, slices[s]->size_bytes());
                        byte_offset += slices[s]->size_bytes();
                    }

                    std::vector<size_t> out_shape = {total_out_rows, cols};
                    TensorType native_type = slices[0]->native_type();

                    auto result_tensor = createTensorFromRawData(
                        native_type, out_shape, std::move(combined_raw));

                    if (!result_tensor)
                    {
                        throw std::runtime_error("[WeightManager] Failed to create fused-QKV tensor for: " + name);
                    }

                    auto meta = SliceMetadata::forColumnParallel(
                        total_rows, cols, rank, world_size,
                        true /* inner_is_presliced */);

                    auto result = std::make_shared<TensorSlice>(
                        std::shared_ptr<TensorBase>(std::move(result_tensor)), meta);

                    WeightSliceSpec fused_slice;
                    fused_slice.source_rows = total_rows;
                    fused_slice.source_cols = cols;
                    fused_slice.row_start = 0;
                    fused_slice.row_count = total_out_rows;
                    fused_slice.col_start = 0;
                    fused_slice.col_count = cols;
                    fused_slice.inner_is_presliced = true;
                    registerDerivedMetadata(name, result, WeightDerivationKind::FusedSubblockConcat, fused_slice, device);

                    LOG_DEBUG("[WeightManager] Rank " << rank
                                                      << " fused-QKV column-parallel " << name
                                                      << " [" << total_rows << ", " << cols << "]"
                                                      << " -> [" << total_out_rows << ", " << cols << "]"
                                                      << " (Q=" << sub_block_sizes[0]
                                                      << " K=" << sub_block_sizes[1]
                                                      << " V=" << sub_block_sizes[2] << ")");

                    return result;
                }
                // Fall through to simple equal row split below
            }

            // Calculate this rank's ROW range (splitting output dimension)
            // Use proportional slicing if TensorParallelConfig is set, otherwise equal split
            size_t row_start, row_count;
            if (tp_config_)
            {
                auto [start, count] = calculateProportionalColumnSlice(name, total_rows);
                row_start = start;
                row_count = count;
            }
            else
            {
                size_t rows_per_rank = total_rows / world_size;
                row_start = rows_per_rank * rank;
                row_count = (rank == world_size - 1) ? (total_rows - row_start) : rows_per_rank;
            }
            size_t row_end = row_start + row_count;

            // Load only this rank's row slice (preserves quantized format)
            auto slice_tensor = loader_.loadTensorRowSlice(
                load_name, row_start, row_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load row slice for: " + name);
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " column-parallel " << name
                                              << " [" << total_rows << ", " << cols
                                              << "] -> loaded ONLY rows [" << row_start << ", " << row_end
                                              << ") = " << row_count << " rows"
                                              << (tp_config_ ? " (proportional)" : " (equal)"));

            // Wrap in TensorSlice with metadata (inner_is_presliced=true)
            // Note: forColumnParallel stores original dimensions for allgather reconstruction
            auto meta = SliceMetadata::forColumnParallel(
                total_rows, cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
            WeightSliceSpec column_slice;
            column_slice.source_rows = total_rows;
            column_slice.source_cols = cols;
            column_slice.row_start = row_start;
            column_slice.row_count = row_count;
            column_slice.col_start = 0;
            column_slice.col_count = cols;
            column_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice, WeightDerivationKind::RowSlice, column_slice, device);
            return slice;
        }
        else if (mode == ShardingMode::INPUT_PARALLEL)
        {
            // Input-parallel: split the INPUT dimension of the weight matrix
            //
            // This is used for FFN Down when Gate/Up are column-parallel:
            // - Gate/Up output: [seq, d_ff_local] where d_ff_local = d_ff / world_size
            // - Down weight: [d_model, d_ff] → each rank needs [d_model, d_ff_local]
            // - Down output: [seq, d_model], needs allreduce-sum across ranks
            //
            // For weight matrix W of shape [N, K] (N=output, K=input):
            //   * Input-parallel splits K (the columns of W)
            //   * Each rank loads cols [col_start, col_end) of W
            //   * GEMM input is [M, K_local], output is [M, N]
            //   * Outputs are allreduce-summed

            int rank = mpi_ctx_->rank();
            int world_size = mpi_ctx_->world_size();

            // Get tensor dimensions
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for input-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for input-parallel (expected 2D): " + name);
            }
            const auto &dims = *dims_opt;

            size_t rows = dims[0];       // N = output features (d_model)
            size_t total_cols = dims[1]; // K = input features (d_ff)

            // Calculate this rank's COLUMN range (splitting input dimension)
            // Use proportional slicing if TensorParallelConfig is set, otherwise equal split
            size_t col_start, col_count;
            if (tp_config_)
            {
                auto [start, count] = calculateProportionalRowSlice(name, total_cols);
                col_start = start;
                col_count = count;
            }
            else
            {
                size_t cols_per_rank = total_cols / world_size;
                col_start = cols_per_rank * rank;
                col_count = (rank == world_size - 1) ? (total_cols - col_start) : cols_per_rank;
            }
            size_t col_end = col_start + col_count;

            // Load only this rank's column slice (preserves quantized format)
            auto slice_tensor = loader_.loadTensorColumnSlice(
                name, col_start, col_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load column slice for: " + name);
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " input-parallel " << name
                                              << " [" << rows << ", " << total_cols
                                              << "] -> loaded ONLY cols [" << col_start << ", " << col_end
                                              << ") = " << col_count << " cols"
                                              << (tp_config_ ? " (proportional)" : " (equal)"));

            // Wrap in TensorSlice with row-parallel metadata
            // (input-parallel is mathematically like row-parallel but with column slicing)
            auto meta = SliceMetadata::forRowParallel(
                rows, total_cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
            WeightSliceSpec input_slice;
            input_slice.source_rows = rows;
            input_slice.source_cols = total_cols;
            input_slice.row_start = 0;
            input_slice.row_count = rows;
            input_slice.col_start = col_start;
            input_slice.col_count = col_count;
            input_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice, WeightDerivationKind::ColumnSlice, input_slice, device);
            return slice;
        }
        else if (mode == ShardingMode::EXPERT_PARALLEL)
        {
            // Dynamic MoE rebalancing: load full (replicated) expert tensor so that
            // any expert can be assigned to any rank at runtime.
            // mmap-backed, so no extra physical RAM — pages fault in on demand.
            const auto& env = debugEnv();
            if (env.moe_rebalance.mode == "dynamic") {
                LOG_DEBUG("[WeightManager] MoE rebalance=dynamic: loading FULL expert tensor "
                         << name << " (replicated mmap, all experts accessible)");
                return getReplicatedWeight(name, device);
            }

            // Expert-parallel: split the EXPERT dimension of 3D MoE weight tensors
            //
            // Expert weights are stored as 3D tensors [ne0, ne1, num_experts] where:
            //   ne0 = columns (fastest varying)
            //   ne1 = rows per expert
            //   ne2 = num_experts (slowest varying)
            //
            // Each rank loads only its assigned expert subset:
            //   expert_start = rank * experts_per_rank
            //   expert_count = experts_per_rank (or remainder for last rank)
            //
            // The resulting tensor has shape [ne0, ne1, local_count] and contains
            // only the local expert data. extractExpertViews() must account for
            // the reduced expert count when creating 2D views.

            int rank = mpi_ctx_->rank();
            int world_size = mpi_ctx_->world_size();

            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for expert-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 3)
            {
                throw std::runtime_error("[WeightManager] Expert-parallel requires 3D tensor, got " +
                                         std::to_string(dims_opt->size()) + "D for: " + name);
            }
            const auto &dims = *dims_opt;
            size_t ne0 = dims[0]; // cols
            size_t ne1 = dims[1]; // rows per expert
            size_t ne2 = dims[2]; // num_experts

            // Equal split of experts across ranks
            size_t experts_per_rank = ne2 / world_size;
            size_t expert_start = experts_per_rank * rank;
            size_t expert_count = (rank == world_size - 1) ? (ne2 - expert_start) : experts_per_rank;
            size_t expert_end = expert_start + expert_count;

            if (experts_per_rank == 0)
            {
                throw std::runtime_error("[WeightManager] Cannot shard " + std::to_string(ne2) +
                                         " experts across " + std::to_string(world_size) +
                                         " ranks for: " + name);
            }

            auto slice_tensor = loader_.loadTensorExpertSlice(
                name, expert_start, expert_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                throw std::runtime_error("[WeightManager] Rank " + std::to_string(rank) +
                                         " failed to load expert slice for: " + name);
            }

            LOG_INFO("[WeightManager] Rank " << rank << " expert-parallel " << name
                                             << " [" << ne0 << ", " << ne1 << ", " << ne2
                                             << "] -> loaded ONLY experts [" << expert_start << ", " << expert_end
                                             << ") = " << expert_count << "/" << ne2 << " experts");

            // Return the sliced 3D tensor directly (no TensorSlice wrapping needed —
            // expert parallelism uses allreduce on the MoE output, not on weights)
            WeightSliceSpec expert_slice;
            expert_slice.source_rows = ne0;
            expert_slice.source_cols = ne1;
            expert_slice.row_start = 0;
            expert_slice.row_count = ne0;
            expert_slice.col_start = 0;
            expert_slice.col_count = ne1;
            expert_slice.expert_start = expert_start;
            expert_slice.expert_count = expert_count;
            expert_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, slice_tensor, WeightDerivationKind::ExpertSlice, expert_slice, device);
            return slice_tensor;
        }

        throw std::runtime_error("[WeightManager] Unknown sharding mode for: " + name);
    }

    std::shared_ptr<TensorBase> WeightManager::getInterleavedWeight(const std::string &name, DeviceId device)
    {
        // Phase 3: Interleaved strategy (not yet implemented)
        // TODO: Implement NUMA-aware allocation with page interleaving
        //
        // For shared-memory multi-socket systems:
        // 1. Allocate with NUMA interleave policy (mbind)
        // 2. Memory distributed across NUMA nodes
        // 3. All ranks can access, but with varying latency
        // 4. Good for systems with fast interconnect (NVLink, etc.)

        LOG_ERROR("[WeightManager] INTERLEAVED strategy not yet implemented, falling back to REPLICATED");
        return getReplicatedWeight(name, device);
    }

    void WeightManager::forEachWeight(std::function<void(const std::string &, TensorBase *)> visitor) const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (const auto &[name, tensor] : cache_)
        {
            if (tensor)
                visitor(name, tensor.get());
        }
    }

    void WeightManager::forEachPreparedWeight(std::function<void(const std::string &, TensorBase *)> visitor) const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        // Primary cache (original tensors, used in single-device mode)
        for (const auto &[name, tensor] : cache_)
        {
            if (tensor)
                visitor(name, tensor.get());
        }
        // Per-device cache (TP-sliced tensors, used in LOCAL TP mode)
        // Keys are "device:name", extract the weight name after the first ':'
        for (const auto &[cache_key, tensor] : per_device_cache_)
        {
            if (!tensor) continue;
            auto colon_pos = cache_key.find(':');
            if (colon_pos == std::string::npos) continue;
            std::string name = cache_key.substr(colon_pos + 1);
            visitor(name, tensor.get());
        }
    }

    std::shared_ptr<PreparedWeightStore> WeightManager::preparedWeightStore()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (!prepared_weight_store_)
            prepared_weight_store_ = std::make_shared<PreparedWeightStore>();
        return prepared_weight_store_;
    }

    std::shared_ptr<PreparedWeightStore> WeightManager::preparedWeightStoreIfInitialized() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return prepared_weight_store_;
    }

    void WeightManager::setPreparedWeightStore(std::shared_ptr<PreparedWeightStore> store)
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        prepared_weight_store_ = std::move(store);
    }

    FrozenModelWeightSet WeightManager::materialize(const WeightPlan &plan)
    {
        ModelWeightSetBuilder builder(plan.strategy());

        for (const auto &requirement : plan.requirements())
        {
            const std::string &load_name = requirement.source_name.empty()
                                               ? requirement.canonical_name
                                               : requirement.source_name;
            const DeviceId lookup_device = requirement.lookup_device.value_or(requirement.target_device);
            std::shared_ptr<TensorBase> tensor;
            if (tp_config_ && requirement.tp_domain >= 0 && requirement.tp_rank_or_device_index >= 0)
            {
                try
                {
                    const auto &assignment = tp_config_->forRank(requirement.tp_rank_or_device_index);
                    tensor = getShardedWeightForAssignment(
                        load_name,
                        lookup_device,
                        assignment,
                        requirement.layer);
                }
                catch (const std::out_of_range &)
                {
                    LOG_WARN("[WeightManager] TP rank " << requirement.tp_rank_or_device_index
                                                         << " not in TensorParallelConfig while materializing "
                                                         << requirement.canonical_name
                                                         << "; falling back to device lookup");
                }
            }
            if (!tensor)
            {
                tensor = getWeightForDevice(
                    load_name,
                    lookup_device,
                    requirement.layer);
            }

            if (!tensor)
            {
                if (requirement.required)
                {
                    throw std::runtime_error("[WeightManager] Required weight missing during materialization: " +
                                             requirement.canonical_name +
                                             (load_name == requirement.canonical_name ? "" : " (source: " + load_name + ")"));
                }
                continue;
            }

            WeightBinding binding;
            binding.binding_id = stableMaterializedBindingId(requirement, plan.strategy());
            binding.identity = makeSourceWeightIdentity(
                requirement.canonical_name,
                plan.strategy().model_id,
                binding.binding_id);
            binding.identity.derivation = requirement.derivation;
            binding.identity.role = requirement.role == WeightRole::Other
                                        ? inferWeightRole(requirement.canonical_name)
                                        : requirement.role;
            binding.identity.layer = requirement.layer >= 0
                                         ? requirement.layer
                                         : inferWeightLayer(requirement.canonical_name);
            binding.identity.expert = requirement.expert >= 0
                                          ? requirement.expert
                                          : inferWeightExpert(requirement.canonical_name);
            binding.identity.pp_stage = requirement.pp_stage;
            binding.identity.tp_domain = requirement.tp_domain;
            binding.identity.tp_rank_or_device_index = requirement.tp_rank_or_device_index;

            binding.slice = requirement.slice;
            binding.residency.home_device = requirement.target_device;
            binding.residency.resident_device = requirement.target_device;
            binding.residency.host_policy = requirement.host_policy;
            binding.tensor = tensor.get();

            if (weight_metadata_)
            {
                if (auto metadata = weight_metadata_->metadata(tensor.get()))
                {
                    binding.identity = metadata->identity;
                    binding.identity.canonical_name = requirement.canonical_name;
                    binding.identity.model_id = plan.strategy().model_id;
                    binding.identity.role = requirement.role == WeightRole::Other
                                                ? metadata->identity.role
                                                : requirement.role;
                    binding.identity.derivation = requirement.derivation;
                    binding.identity.layer = requirement.layer >= 0
                                                 ? requirement.layer
                                                 : metadata->identity.layer;
                    binding.identity.expert = requirement.expert >= 0
                                                  ? requirement.expert
                                                  : metadata->identity.expert;
                    binding.identity.pp_stage = requirement.pp_stage;
                    binding.identity.tp_domain = requirement.tp_domain;
                    binding.identity.tp_rank_or_device_index = requirement.tp_rank_or_device_index;
                    binding.slice = metadata->slice;
                    binding.residency = metadata->residency;
                    binding.residency.home_device = requirement.target_device;
                    binding.residency.resident_device = requirement.target_device;
                    binding.residency.host_policy = requirement.host_policy;
                }
            }

            if (requirement.slice.source_rows != 0 || requirement.slice.source_cols != 0 ||
                requirement.slice.row_count != 0 || requirement.slice.col_count != 0 ||
                requirement.slice.expert_count != 0)
            {
                binding.slice = requirement.slice;
            }

            if (binding.slice.source_rows == 0 && tensor)
                binding.slice = fullSliceSpec(*tensor);

            binding.identity.instance_id = binding.binding_id;

            if (requirement.expected_prepared_kind != PreparedWeightKind::None)
            {
                binding.prepared = PreparedWeightRef{
                    plan.strategy().model_id,
                    0,
                    requirement.expected_prepared_kind,
                    requirement.target_device};
            }

            auto &added = builder.addBinding(std::move(binding));
            if (added.prepared)
                added.prepared->binding_id = added.binding_id;

            if (added.prepared.has_value())
            {
                if (auto store = preparedWeightStoreIfInitialized())
                {
                    if (store->adoptPreparedGemmForBinding(added, requirement.target_device))
                    {
                        WeightLifecycleTrace::record(
                            WeightLifecycleEventType::RegisterPrepared,
                            requirement.canonical_name,
                            added.identity.role,
                            added.identity.layer,
                            requirement.target_device,
                            "adopted prepared handle for materialized binding");
                    }
                }
            }

            WeightLifecycleTrace::record(
                WeightLifecycleEventType::GraphBind,
                requirement.canonical_name,
                added.identity.role,
                added.identity.layer,
                requirement.target_device,
                "materialized binding " + std::to_string(added.binding_id));
        }

        FrozenModelWeightSet frozen(plan.strategy(), builder.freezeBindings());
        frozen.validateForGraph();
        return frozen;
    }

    // =========================================================================
    // Decode Weight Shard Support (CPU Decode Participation - Option A)
    // =========================================================================

    std::shared_ptr<TensorBase> WeightManager::getDecodeWeight(
        const std::string &name,
        DeviceId decode_device,
        float fraction,
        int layer_idx)
    {
        // Validate fraction
        if (fraction <= 0.0f || fraction > 1.0f)
        {
            throw std::runtime_error("[WeightManager] Invalid decode fraction: " +
                                     std::to_string(fraction) + " (must be 0 < fraction <= 1)");
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Generate cache key (includes fraction to allow different shard sizes)
        std::string cache_key = name + "@decode_" + std::to_string(fraction);

        // Check decode cache first
        auto it = decode_cache_.find(cache_key);
        if (it != decode_cache_.end())
        {
            return it->second;
        }

        // Get the full weight tensor from cache or load fresh
        // NOTE: We cannot call getWeightForDevice() here because we already hold cache_mutex_
        // Instead, check cache directly and load via getReplicatedWeight if needed
        std::shared_ptr<TensorBase> full_tensor;
        auto cache_it = cache_.find(name);
        if (cache_it != cache_.end())
        {
            full_tensor = cache_it->second;
        }
        else
        {
            // Load fresh - use replicated loading (decode shards are always on CPU)
            full_tensor = getReplicatedWeight(name, DeviceId::cpu());
            if (full_tensor)
            {
                cache_[name] = full_tensor;
            }
        }

        if (!full_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load full weight for decode shard: " + name);
        }

        // Determine sharding mode for this weight
        ShardingMode mode = ShardingMode::REPLICATE;
        if (has_sharding_config_)
        {
            mode = getShardingMode(name);
        }

        std::shared_ptr<TensorBase> decode_shard;

        switch (mode)
        {
        case ShardingMode::REPLICATE:
        {
            // Replicated weights (norms, biases) - return full copy
            // These are small, so no need to slice
            if (fraction >= 1.0f)
            {
                // Full copy requested - just return the tensor
                decode_shard = full_tensor;
            }
            else
            {
                // Still return full tensor for replicated weights
                // Slicing doesn't make sense for norms/biases
                LOG_DEBUG("[WeightManager] Decode shard for REPLICATE weight " << name
                                                                               << " - returning full copy (fraction ignored)");
                decode_shard = full_tensor;
            }
            break;
        }

        case ShardingMode::COLUMN_PARALLEL:
        {
            // Column-parallel weights (Q, K, V, Gate, Up) - slice tail rows
            // These weights have shape [out_dim, in_dim]
            // We slice the output dimension (rows) for decode
            decode_shard = sliceTailRows(full_tensor, fraction);
            if (!decode_shard)
            {
                throw std::runtime_error("[WeightManager] Failed to slice tail rows for decode: " + name);
            }
            LOG_DEBUG("[WeightManager] Decode shard for COLUMN_PARALLEL " << name
                                                                          << ": tail " << (fraction * 100) << "% rows -> ["
                                                                          << decode_shard->shape()[0] << ", "
                                                                          << decode_shard->shape()[1] << "]");
            break;
        }

        case ShardingMode::ROW_PARALLEL:
        case ShardingMode::INPUT_PARALLEL:
        {
            // Row/Input-parallel weights (Wo, Down) - slice tail columns
            // These weights have shape [out_dim, in_dim]
            // We slice the input dimension (columns) for decode
            decode_shard = sliceTailColumns(full_tensor, fraction);
            if (!decode_shard)
            {
                throw std::runtime_error("[WeightManager] Failed to slice tail columns for decode: " + name);
            }
            LOG_DEBUG("[WeightManager] Decode shard for " << (mode == ShardingMode::ROW_PARALLEL ? "ROW_PARALLEL" : "INPUT_PARALLEL")
                                                          << " " << name
                                                          << ": tail " << (fraction * 100) << "% cols -> ["
                                                          << decode_shard->shape()[0] << ", "
                                                          << decode_shard->shape()[1] << "]");
            break;
        }

        case ShardingMode::EXPERT_PARALLEL:
        {
            // Expert-parallel weights are 3D MoE tensors — decode sharding
            // not applicable. Return the full tensor as-is.
            decode_shard = full_tensor;
            break;
        }

        default:
            throw std::runtime_error("[WeightManager] Unknown sharding mode for decode weight: " + name);
        }

        // Cache the decode shard
        if (decode_shard)
        {
            if (decode_shard.get() != full_tensor.get())
            {
                WeightSliceSpec decode_slice = fullSliceSpec(*full_tensor);
                decode_slice.inner_is_presliced = true;
                if (mode == ShardingMode::COLUMN_PARALLEL)
                {
                    decode_slice.row_count = decode_shard->shape()[0];
                    decode_slice.row_start = decode_slice.source_rows - decode_slice.row_count;
                    decode_slice.col_start = 0;
                    decode_slice.col_count = decode_slice.source_cols;
                }
                else
                {
                    decode_slice.row_start = 0;
                    decode_slice.row_count = decode_slice.source_rows;
                    decode_slice.col_count = decode_shard->shape().size() > 1 ? decode_shard->shape()[1] : 1;
                    decode_slice.col_start = decode_slice.source_cols - decode_slice.col_count;
                }
                registerDerivedMetadata(name, decode_shard, WeightDerivationKind::DecodeShard, decode_slice, decode_device);
            }
            decode_cache_[cache_key] = decode_shard;
        }

        return decode_shard;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceTailRows(
        const std::shared_ptr<TensorBase> &full_tensor,
        float fraction)
    {
        // For weight matrix [out_dim, in_dim], extract tail rows
        // Result: [out_local, in_dim] where out_local = out_dim * fraction

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Tail row slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];

        // Calculate slice dimensions
        size_t out_local = static_cast<size_t>(std::ceil(out_dim * fraction));
        if (out_local == 0)
        {
            out_local = 1; // At least one row
        }
        if (out_local > out_dim)
        {
            out_local = out_dim;
        }

        size_t out_start = out_dim - out_local; // Tail portion

        LOG_TRACE("[WeightManager] Tail row slicing: [" << out_dim << ", " << in_dim
                                                        << "] -> rows [" << out_start << ", " << out_dim
                                                        << ") = [" << out_local << ", " << in_dim << "]");

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Tail row slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for decode shards.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_local, in_dim};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy tail rows
        const float *src = fp32_tensor->data() + out_start * in_dim;
        float *dst = sliced->mutable_data();
        std::memcpy(dst, src, out_local * in_dim * sizeof(float));

        return sliced;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceTailColumns(
        const std::shared_ptr<TensorBase> &full_tensor,
        float fraction)
    {
        // For weight matrix [out_dim, in_dim], extract tail columns
        // Result: [out_dim, in_local] where in_local = in_dim * fraction

        const auto &shape = full_tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] Tail column slicing requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];

        // Calculate slice dimensions
        size_t in_local = static_cast<size_t>(std::ceil(in_dim * fraction));
        if (in_local == 0)
        {
            in_local = 1; // At least one column
        }
        if (in_local > in_dim)
        {
            in_local = in_dim;
        }

        size_t in_start = in_dim - in_local; // Tail portion

        LOG_TRACE("[WeightManager] Tail column slicing: [" << out_dim << ", " << in_dim
                                                           << "] -> cols [" << in_start << ", " << in_dim
                                                           << ") = [" << out_dim << ", " << in_local << "]");

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] Tail column slicing currently requires FP32 tensor. "
                                     "Use CONVERT_TO_FP32 weight precision for decode shards.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {out_dim, in_local};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy tail columns (non-contiguous in memory)
        const float *src = fp32_tensor->data();
        float *dst = sliced->mutable_data();

#pragma omp parallel for
        for (size_t row = 0; row < out_dim; ++row)
        {
            const float *src_row = src + row * in_dim + in_start;
            float *dst_row = dst + row * in_local;
            std::memcpy(dst_row, src_row, in_local * sizeof(float));
        }

        return sliced;
    }

    // =========================================================================
    // Proportional Slicing (TensorParallelConfig support)
    // =========================================================================

    WeightManager::WeightCategory WeightManager::categorizeWeight(const std::string &name) const
    {
        // Attention Q/K/V projections (column-parallel by heads)
        if (name.find("attn_q.") != std::string::npos ||
            name.find("attn_k.") != std::string::npos ||
            name.find("attn_v.") != std::string::npos ||
            name.find("attn_qkv.") != std::string::npos)
        {
            return WeightCategory::ATTENTION_QKV;
        }

        // Attention output projection (row-parallel)
        if (name.find("attn_output.") != std::string::npos)
        {
            return WeightCategory::ATTENTION_WO;
        }

        // FFN gate and up projections (column-parallel by d_ff)
        if (name.find("ffn_gate.") != std::string::npos ||
            name.find("ffn_up.") != std::string::npos ||
            name.find("ffn_gate_up.") != std::string::npos)
        {
            return WeightCategory::FFN_GATE_UP;
        }

        // FFN down projection (input-parallel)
        if (name.find("ffn_down.") != std::string::npos)
        {
            return WeightCategory::FFN_DOWN;
        }

        // LM head (column-parallel by vocab)
        if (name.find("output.weight") != std::string::npos)
        {
            return WeightCategory::LM_HEAD;
        }

        // Everything else is replicated
        return WeightCategory::REPLICATE;
    }

    std::pair<size_t, size_t> WeightManager::calculateProportionalColumnSlice(
        const std::string &name, size_t total_rows) const
    {
        if (!tp_config_ || !mpi_ctx_)
        {
            // Fallback to equal split
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            int world_size = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
            size_t rows_per_rank = total_rows / world_size;
            size_t start = rows_per_rank * rank;
            size_t count = (rank == world_size - 1) ? (total_rows - start) : rows_per_rank;
            return {start, count};
        }

        int rank = mpi_ctx_->rank();
        const auto &assignment = tp_config_->forRank(rank);
        WeightCategory category = categorizeWeight(name);

        switch (category)
        {
        case WeightCategory::ATTENTION_QKV:
        {
            // Column-parallel by heads
            // For Q: full head_dim per head, for K/V: use kv_head mapping
            // Q weight shape: [n_heads * head_dim, d_model]
            // K/V weight shape: [n_kv_heads * head_dim, d_model]
            // Each rank gets [local_head_count * head_dim, d_model]
            const bool is_kv_weight = (name.find("attn_k.") != std::string::npos ||
                                       name.find("attn_v.") != std::string::npos);

            // CRITICAL: K/V weights use totalKVHeads(), not totalHeads()!
            const int total_heads_for_weight = is_kv_weight
                                                   ? tp_config_->totalKVHeads()
                                                   : tp_config_->totalHeads();
            if (total_heads_for_weight <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads/total_kv_heads in TensorParallelConfig");
            }
            const size_t head_dim = total_rows / static_cast<size_t>(total_heads_for_weight);

            if (is_kv_weight)
            {
                // K/V use KV heads
                size_t start = assignment.kv_head_start * head_dim;
                size_t count = assignment.kv_head_count * head_dim;
                LOG_TRACE("[WeightManager] Proportional KV slice for " << name
                                                                       << ": total_kv_heads=" << total_heads_for_weight
                                                                       << " head_dim=" << head_dim
                                                                       << " kv_heads [" << assignment.kv_head_start << ", "
                                                                       << (assignment.kv_head_start + assignment.kv_head_count) << ")"
                                                                       << " -> rows [" << start << ", " << (start + count) << ")");
                return {start, count};
            }
            else
            {
                // Q uses full Q heads
                size_t start = assignment.head_start * head_dim;
                size_t count = assignment.head_count * head_dim;
                LOG_TRACE("[WeightManager] Proportional Q slice for " << name
                                                                      << ": total_heads=" << total_heads_for_weight
                                                                      << " head_dim=" << head_dim
                                                                      << " heads [" << assignment.head_start << ", "
                                                                      << (assignment.head_start + assignment.head_count) << ")"
                                                                      << " -> rows [" << start << ", " << (start + count) << ")");
                return {start, count};
            }
        }

        case WeightCategory::FFN_GATE_UP:
        {
            // Column-parallel by d_ff
            // Weight shape: [d_ff, d_model]
            // Each rank gets [d_ff_count, d_model]
            size_t start = assignment.d_ff_start;
            size_t count = assignment.d_ff_count;
            LOG_TRACE("[WeightManager] Proportional FFN gate/up slice for " << name
                                                                            << ": d_ff [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        case WeightCategory::LM_HEAD:
        {
            // Column-parallel by vocab
            // Weight shape: [vocab_size, d_model]
            // Each rank gets [vocab_count, d_model]
            size_t start = assignment.vocab_start;
            size_t count = assignment.vocab_count;
            LOG_TRACE("[WeightManager] Proportional LM head slice for " << name
                                                                        << ": vocab [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        default:
        {
            // Check schema-based dimension type for weights not covered by WeightCategory
            // (e.g. GDN ssm_alpha/beta/dt.bias/ssm_a which are ProportionalHeads)
            if (has_sharding_config_)
            {
                WeightDimensionType dim_type = sharding_config_.getDimensionType(name);
                if (dim_type == WeightDimensionType::ProportionalHeads)
                {
                    const int total_heads = tp_config_->totalHeads();
                    if (total_heads > 0)
                    {
                        size_t start = total_rows * static_cast<size_t>(assignment.head_start) / static_cast<size_t>(total_heads);
                        size_t end = total_rows * static_cast<size_t>(assignment.head_start + assignment.head_count) / static_cast<size_t>(total_heads);
                        LOG_TRACE("[WeightManager] Proportional heads slice for " << name
                                                                                  << ": total_heads=" << total_heads
                                                                                  << " weight_rows=" << total_rows
                                                                                  << " -> [" << start << ", " << end << ")");
                        return {start, end - start};
                    }
                }
                else if (dim_type == WeightDimensionType::Heads)
                {
                    const int total_heads = tp_config_->totalHeads();
                    if (total_heads > 0)
                    {
                        const size_t head_dim = total_rows / static_cast<size_t>(total_heads);
                        size_t start = static_cast<size_t>(assignment.head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.head_count) * head_dim;
                        LOG_TRACE("[WeightManager] Schema heads slice for " << name
                                                                            << ": total_heads=" << total_heads
                                                                            << " head_dim=" << head_dim
                                                                            << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
                else if (dim_type == WeightDimensionType::KVHeads)
                {
                    const int total_kv_heads = tp_config_->totalKVHeads();
                    if (total_kv_heads > 0)
                    {
                        const size_t head_dim = total_rows / static_cast<size_t>(total_kv_heads);
                        size_t start = static_cast<size_t>(assignment.kv_head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.kv_head_count) * head_dim;
                        LOG_TRACE("[WeightManager] Schema KV heads slice for " << name
                                                                               << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
            }
            LOG_WARN("[WeightManager] calculateProportionalColumnSlice called for non-column weight: " << name);
            return {0, total_rows};
        }
        } // end switch
        return {0, total_rows}; // unreachable but silences -Wreturn-type
    }

    std::pair<size_t, size_t> WeightManager::calculateProportionalRowSlice(
        const std::string &name, size_t total_cols) const
    {
        if (!tp_config_ || !mpi_ctx_)
        {
            // Fallback to equal split
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            int world_size = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
            size_t cols_per_rank = total_cols / world_size;
            size_t start = cols_per_rank * rank;
            size_t count = (rank == world_size - 1) ? (total_cols - start) : cols_per_rank;
            return {start, count};
        }

        int rank = mpi_ctx_->rank();
        const auto &assignment = tp_config_->forRank(rank);
        WeightCategory category = categorizeWeight(name);

        switch (category)
        {
        case WeightCategory::ATTENTION_WO:
        {
            // Row-parallel: input dimension matches column-parallel Q output
            // Weight shape: [d_model, n_heads * head_dim]
            // Each rank needs columns [head_start * head_dim, (head_start + head_count) * head_dim)
            int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads in TensorParallelConfig");
            }
            size_t head_dim = total_cols / total_heads;
            size_t start = assignment.head_start * head_dim;
            size_t count = assignment.head_count * head_dim;
            LOG_TRACE("[WeightManager] Proportional Wo slice for " << name
                                                                   << ": cols [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        case WeightCategory::FFN_DOWN:
        {
            // Input-parallel: input dimension matches column-parallel gate/up output
            // Weight shape: [d_model, d_ff]
            // Each rank needs columns [d_ff_start, d_ff_start + d_ff_count)
            size_t start = assignment.d_ff_start;
            size_t count = assignment.d_ff_count;
            LOG_TRACE("[WeightManager] Proportional FFN down slice for " << name
                                                                         << ": cols [" << start << ", " << (start + count) << ")");
            return {start, count};
        }

        default:
        {
            // Check schema-based dimension type for weights not covered by WeightCategory
            // (e.g. GDN ssm_out which is InputParallel with ProportionalHeads or Heads)
            if (has_sharding_config_)
            {
                WeightDimensionType dim_type = sharding_config_.getDimensionType(name);
                if (dim_type == WeightDimensionType::ProportionalHeads)
                {
                    const int total_heads = tp_config_->totalHeads();
                    if (total_heads > 0)
                    {
                        size_t start = total_cols * static_cast<size_t>(assignment.head_start) / static_cast<size_t>(total_heads);
                        size_t end = total_cols * static_cast<size_t>(assignment.head_start + assignment.head_count) / static_cast<size_t>(total_heads);
                        LOG_TRACE("[WeightManager] Proportional heads row-slice for " << name
                                                                                      << ": total_heads=" << total_heads
                                                                                      << " weight_cols=" << total_cols
                                                                                      << " -> [" << start << ", " << end << ")");
                        return {start, end - start};
                    }
                }
                else if (dim_type == WeightDimensionType::Heads)
                {
                    const int total_heads = tp_config_->totalHeads();
                    if (total_heads > 0)
                    {
                        const size_t head_dim = total_cols / static_cast<size_t>(total_heads);
                        size_t start = static_cast<size_t>(assignment.head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.head_count) * head_dim;
                        LOG_WARN("[WeightManager] Schema heads row-slice for " << name
                                                                               << " total_cols=" << total_cols
                                                                               << " total_heads=" << total_heads
                                                                               << " head_dim=" << head_dim
                                                                               << " assignment.head_start=" << assignment.head_start
                                                                               << " head_count=" << assignment.head_count
                                                                               << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
                else if (dim_type == WeightDimensionType::KVHeads)
                {
                    const int total_kv_heads = tp_config_->totalKVHeads();
                    if (total_kv_heads > 0)
                    {
                        const size_t head_dim = total_cols / static_cast<size_t>(total_kv_heads);
                        size_t start = static_cast<size_t>(assignment.kv_head_start) * head_dim;
                        size_t count = static_cast<size_t>(assignment.kv_head_count) * head_dim;
                        LOG_TRACE("[WeightManager] Schema KV heads row-slice for " << name
                                                                                   << " -> [" << start << ", " << (start + count) << ")");
                        return {start, count};
                    }
                }
            }
            LOG_WARN("[WeightManager] calculateProportionalRowSlice called for non-row weight: " << name);
            return {0, total_cols};
        }
        } // end switch
        return {0, total_cols}; // unreachable but silences -Wreturn-type
    }

    // =========================================================================
    // Thread-safe cache accessors
    // =========================================================================

    size_t WeightManager::cacheSize() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_.size();
    }

    void WeightManager::clearCache()
    {
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_.clear();
            per_device_cache_.clear();
        }

        {
            std::lock_guard<std::mutex> lock(prep_ticket_mutex_);
            prep_tickets_.clear();
            expected_devices_by_weight_.clear();
            reclaim_ready_weights_.clear();
            reclaim_applied_weights_.clear();
        }
    }

    size_t WeightManager::decodeCacheSize() const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return decode_cache_.size();
    }

    void WeightManager::clearDecodeCache()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        decode_cache_.clear();
    }

    // =========================================================================
    // Device-aware weight access (for multi-device LOCAL TP)
    // =========================================================================

    std::shared_ptr<TensorBase> WeightManager::cloneTensorForDevice(
        const std::string &name,
        const std::shared_ptr<TensorBase> &original,
        DeviceId target_device)
    {
        if (!original)
        {
            return nullptr;
        }

        TensorType tensor_type = original->native_type();
        size_t byte_count = original->size_bytes();
        const void *raw_ptr = original->raw_data();

        if (!raw_ptr || byte_count == 0)
        {
            throw std::runtime_error("[WeightManager] Cannot clone tensor with no data: " + name);
        }

        // Copy raw bytes
        std::vector<uint8_t> raw_copy(byte_count);
        std::memcpy(raw_copy.data(), raw_ptr, byte_count);

        auto unique_clone = createTensorFromRawData(
            tensor_type, original->shape(), std::move(raw_copy));
        if (!unique_clone)
        {
            throw std::runtime_error("[WeightManager] Failed to create clone for tensor type " +
                                     std::to_string(static_cast<int>(tensor_type)));
        }
        std::shared_ptr<TensorBase> clone = std::shared_ptr<TensorBase>(std::move(unique_clone));

        clone->setDebugName(name + "@" + target_device.to_string());

        LOG_DEBUG("[WeightManager] Cloned tensor for " << target_device.to_string()
                                                       << ": " << name << " ["
                                                       << original->shape()[0] << "x"
                                                       << (original->shape().size() > 1 ? original->shape()[1] : 1) << "]"
                                                       << " type=" << static_cast<int>(tensor_type));

        // Enhanced logging for non-GEMM weights (biases, norms) to debug multi-GPU pointer issues
        if (name.find("bias") != std::string::npos || name.find("norm") != std::string::npos)
        {
            LOG_DEBUG("[WeightManager] NON-GEMM WEIGHT CLONE: " << name
                                                                << " original_ptr=" << static_cast<const void *>(original->raw_data())
                                                                << " clone_ptr=" << static_cast<const void *>(clone->raw_data())
                                                                << " target_device=" << target_device.to_string()
                                                                << " size_bytes=" << clone->size_bytes());
        }

        registerCloneMetadata(name, original, clone, target_device);
        return clone;
    }

    std::shared_ptr<TensorBase> WeightManager::getWeightForDevice(
        const std::string &name,
        DeviceId device,
        int layer_idx)
    {
        std::unique_lock<std::mutex> lock(cache_mutex_);

        // For LAYER_PARTITIONED strategy, filter out weights not in our layer range
        if (strategy_ == WeightDistributionStrategy::LAYER_PARTITIONED && has_layer_range_)
        {
            if (!isWeightInLayerRange(name))
            {
                LOG_DEBUG("[WeightManager] Layer filter: skipping " << name << " (not in range ["
                                                                    << layer_first_ << ", " << layer_last_ << "))");
                return nullptr;
            }
        }

        // =======================================================================
        // LOCAL TP support: Check TensorParallelConfig FIRST for device-specific slicing
        // =======================================================================
        // When tp_config_ is set, each device gets a DIFFERENT slice of the weights
        // based on its DeviceShardingAssignment. This applies regardless of the
        // distribution strategy (SHARDED, REPLICATED, or LAYER_PARTITIONED).
        //
        // For hybrid PP+TP mode:
        // - The PP stage context uses LAYER_PARTITIONED strategy for layer filtering
        // - But the nested MDO still needs TP column-parallel slicing within those layers
        // - So we check tp_config_ alone, not strategy_
        if (tp_config_)
        {
            try
            {
                const auto &assignment = tp_config_->forDevice(device);
                return getShardedWeightForAssignment(name, device, assignment, layer_idx);
            }
            catch (const std::out_of_range &)
            {
                // Device not in config - fall through to standard path
                LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                    << " not in TensorParallelConfig, using standard path for: " << name);
            }
        }

        // Track first device - original tensors stay on this device
        if (!first_device_.has_value())
        {
            first_device_ = device;
            LOG_DEBUG("[WeightManager] First device for weights: " << device.to_string());
        }

        // Helper lambda to load weight if not in cache (reuses getWeight logic without re-locking)
        auto ensureWeightLoaded = [this, &name, &device, layer_idx, &lock]() -> std::shared_ptr<TensorBase>
        {
            auto it = cache_.find(name);
            if (it != cache_.end())
            {
                return it->second;
            }

            // Weight not in cache - load it now
            LOG_DEBUG("[WeightManager] getWeightForDevice: loading " << name << " (not in cache)");

            // Determine device from placement map if not explicitly provided
            DeviceId target_device = device;
            if (!target_device.is_valid() && placement_map_)
            {
                target_device = placement_map_->getDeviceForWeight(name, layer_idx);
            }
            if (!target_device.is_valid())
            {
                target_device = DeviceId::cpu();
            }

            // Load based on strategy (same logic as getWeight)
            std::shared_ptr<TensorBase> tensor;
            lock.unlock();
            switch (strategy_)
            {
            case WeightDistributionStrategy::REPLICATED:
            case WeightDistributionStrategy::LAYER_PARTITIONED:
                // LAYER_PARTITIONED uses same loading as REPLICATED, but getWeightForDevice()
                // filters which weights are allowed based on layer range
                tensor = getReplicatedWeight(name, target_device);
                break;
            case WeightDistributionStrategy::SHARDED:
                tensor = getShardedWeight(name, target_device);
                break;
            case WeightDistributionStrategy::INTERLEAVED:
                tensor = getInterleavedWeight(name, target_device);
                break;
            default:
                throw std::runtime_error("[WeightManager] Unknown strategy: " +
                                         std::to_string(static_cast<int>(strategy_)));
            }

            lock.lock();

            // Double-check cache after loading in case another thread won the race.
            auto cached_it = cache_.find(name);
            if (cached_it != cache_.end())
            {
                return cached_it->second;
            }

            if (tensor)
            {
                cache_[name] = tensor;
                LOG_DEBUG("[WeightManager] Cached tensor via getWeightForDevice: " << name);
            }
            return tensor;
        };

        // For the first device, return the original tensor
        if (first_device_.value() == device)
        {
            auto result = ensureWeightLoaded();
            // Log non-GEMM weights (biases, norms) for multi-GPU debugging
            if (result && (name.find("bias") != std::string::npos || name.find("norm") != std::string::npos))
            {
                LOG_DEBUG("[WeightManager] NON-GEMM WEIGHT (first device): " << name
                                                                             << " ptr=" << static_cast<const void *>(result->raw_data())
                                                                             << " device=" << device.to_string()
                                                                             << " size_bytes=" << result->size_bytes());
            }
            return result;
        }

        // Check per-device cache for subsequent devices
        std::string cache_key = device.to_string() + ":" + name;
        auto it = per_device_cache_.find(cache_key);
        if (it != per_device_cache_.end())
        {
            // Log non-GEMM weights (biases, norms) for multi-GPU debugging
            if (name.find("bias") != std::string::npos || name.find("norm") != std::string::npos)
            {
                LOG_DEBUG("[WeightManager] NON-GEMM WEIGHT (cached clone): " << name
                                                                             << " ptr=" << static_cast<const void *>(it->second->raw_data())
                                                                             << " device=" << device.to_string()
                                                                             << " size_bytes=" << it->second->size_bytes());
            }
            return it->second;
        }

        // Need to create a clone for this device
        // First, ensure the original is loaded
        auto original = ensureWeightLoaded();
        if (!original)
        {
            // LOG_DEBUG, not LOG_ERROR: returning nullptr is valid for optional weights.
            // The caller decides severity based on schema context.
            LOG_DEBUG("[WeightManager] getWeightForDevice: weight not found: " << name);
            return nullptr;
        }

        auto clone = cloneTensorForDevice(name, original, device);
        if (!clone)
        {
            throw std::runtime_error("[WeightManager] getWeightForDevice: failed to clone " + name);
        }

        per_device_cache_[cache_key] = clone;
        return clone;
    }

    bool WeightManager::preloadForDevices(const std::vector<DeviceId> &devices)
    {
        if (devices.empty())
        {
            LOG_WARN("[WeightManager] preloadForDevices called with empty device list");
            return true;
        }

        LOG_INFO("[WeightManager] Pre-loading weights for " << devices.size() << " devices");

        std::vector<std::string> weight_names;
        bool seeded_from_loader = false;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Set first device if not already set
            if (!first_device_.has_value())
            {
                first_device_ = devices[0];
                LOG_DEBUG("[WeightManager] First device set to: " << devices[0].to_string());
            }

            // Prefer already-cached names (fast path)
            weight_names.reserve(cache_.size());
            for (const auto &[name, _] : cache_)
            {
                weight_names.push_back(name);
            }
        }

        // If cache is empty, seed from model tensor list and let getWeightForDevice()
        // materialize/cache per-device tensors on demand.
        if (weight_names.empty())
        {
            weight_names = loader_.tensorNames();
            seeded_from_loader = true;

            if (weight_names.empty())
            {
                LOG_WARN("[WeightManager] No tensor names available for preloading");
                return true;
            }

            LOG_INFO("[WeightManager] Cache empty; seeding preload from model tensor list ("
                     << weight_names.size() << " tensors)");
        }

        // Add virtual weights that exist in the sharding config but not in
        // the GGUF file. Example: "output.weight" when tied to
        // "token_embd.weight" — the GGUF has no "output.weight" tensor, but
        // getShardedWeightForAssignment() knows how to synthesize one.
        // Without this, packGemmWeightsViaPipeline() never sees the LM head,
        // and the later buildWeights() call creates a different tensor object
        // that misses the prepared-GEMM registry.
        if (has_sharding_config_)
        {
            std::unordered_set<std::string> name_set(weight_names.begin(), weight_names.end());
            for (const auto &[name, mode] : sharding_config_.exact_matches)
            {
                if (name_set.find(name) == name_set.end())
                {
                    weight_names.push_back(name);
                    LOG_DEBUG("[WeightManager] Adding virtual weight to preload: " << name);
                }
            }
        }

        size_t loaded_tensors = 0;
        size_t total_uploads = 0;
        size_t load_failures = 0;

        for (const auto &device : devices)
        {
            LOG_DEBUG("[WeightManager] Preloading weights for device " << device.to_string());

            for (const auto &name : weight_names)
            {
                // Respect PP layer filtering semantics during seeded preload.
                if (strategy_ == WeightDistributionStrategy::LAYER_PARTITIONED && has_layer_range_)
                {
                    if (!isWeightInLayerRange(name))
                    {
                        continue;
                    }
                }

                auto tensor = getWeightForDevice(name, device);
                const bool is_gemm_weight = isGemmWeight(name);
                registerExpectedDeviceForWeight(name, device);
                if (!tensor)
                {
                    markPrepState(name, device, WeightPrepState::FAILED, is_gemm_weight, "getWeightForDevice failed during preload");
                    ++load_failures;
                    continue;
                }

                markPrepState(name, device, WeightPrepState::LOADED_HOST, is_gemm_weight, "weight loaded for device");

                // On GPU, mark tensors HOST_RESIDENT when the TransferEngine
                // upload would produce a dead copy:
                //
                // - token_embd.weight: The embedding kernel reads host data
                //   to repack into a device workspace; never read on GPU.
                //
                // - GEMM weights (attn_q/k/v/output, ffn_gate/up/down, etc.):
                //   ROCmQuantisedGemmKernel uploads its own VNNI-repacked copy
                //   and never reads the raw Q8_0 gpu_data_ptr_. Without this,
                //   both the raw Q8_0 AND the VNNI copy sit on GPU — ~1.9 GB
                //   wasted per device.
                //
                // - MoE 3D expert tensors (_exps.weight): MoEExpertWeightService
                //   reads host views and batch-packs into VNNI slabs with its own
                //   GPU upload.  Uploading the raw 3D tensor here wastes ~8 GB/device
                //   and fragments VRAM, making subsequent hipMalloc calls very slow.
                const bool is_moe_expert = (name.find("_exps.weight") != std::string::npos);
                if (device.is_gpu() && (name == "token_embd.weight" || is_gemm_weight || is_moe_expert))
                {
                    tensor->setHostResident();
                }

                ++loaded_tensors;

                if (device.type != DeviceType::CPU)
                {
                    if (tensor->ensureOnDevice(device))
                    {
                        markPrepState(name, device, WeightPrepState::UPLOADED_DEVICE, is_gemm_weight, "preload ensureOnDevice completed");
                        if (!is_gemm_weight)
                        {
                            markPrepState(name, device, WeightPrepState::READY, false, "preload non-GEMM ready");
                        }
                        ++total_uploads;
                    }
                    else
                    {
                        markPrepState(name, device, WeightPrepState::FAILED, is_gemm_weight, "preload ensureOnDevice failed");
                        ++load_failures;
                        LOG_WARN("[WeightManager] Failed to upload preloaded tensor "
                                 << name << " to " << device.to_string());
                    }
                }
            }
        }

        LOG_INFO("[WeightManager] Preload complete: loaded=" << loaded_tensors
                                                             << ", uploads=" << total_uploads
                                                             << ", failures=" << load_failures
                                                             << (seeded_from_loader ? " (seeded from loader names)" : ""));
        return true;
    }

    // =========================================================================
    // Weight Preparation (repack + upload, NO host release)
    // =========================================================================

    bool WeightManager::prepareWeightsForDevice(DeviceId device)
    {
        return prepareWeightsForDeviceImpl(device, nullptr);
    }

    bool WeightManager::prepareWeightsForDevice(
        const FrozenModelWeightSet &frozen_weights,
        DeviceId device,
        bool include_expert_jobs)
    {
        LOG_INFO("[WeightManager] prepareWeightsForDevice(" << device.to_string()
                                                            << ", frozen_bindings=" << frozen_weights.bindings().size() << ")");
        if (!lifecycle_gates_.materialization_complete)
            markMaterializationComplete();

        if (device.is_cpu())
        {
            auto store = preparedWeightStore();
            int registered = 0;
            for (const auto &binding : frozen_weights.bindings())
            {
                if (!binding.prepared.has_value() || binding.prepared->device != device)
                    continue;
                if (!binding.tensor || binding.tensor->shape().size() != 2)
                    continue;
                if (binding.identity.role == WeightRole::Embedding ||
                    binding.identity.role == WeightRole::Norm ||
                    binding.identity.role == WeightRole::Bias)
                {
                    continue;
                }
                if (!store->preparedRefForBinding(binding.binding_id, device).has_value())
                {
                    store->prepareGemm(binding);
                    ++registered;
                }
            }

            if (!lifecycle_gates_.device_preparation_complete)
                markDevicePreparationComplete();
            LOG_INFO("[WeightManager] Binding-driven CPU preparation registered "
                     << registered << " GEMM bindings for " << device.to_string());
            return true;
        }

        const bool ok = prepareWeightsForDeviceImpl(device, nullptr, &frozen_weights, include_expert_jobs);
        if (ok && !lifecycle_gates_.device_preparation_complete)
            markDevicePreparationComplete();
        return ok;
    }

    bool WeightManager::prepareWeightsForDevice(
        DeviceId device,
        int first_layer, int last_layer,
        bool has_embedding, bool has_lm_head)
    {
        auto layer_filter = [this, first_layer, last_layer, has_embedding, has_lm_head](
                                const std::string &name) -> bool
        {
            return isWeightInLayerRange(name, first_layer, last_layer, has_embedding, has_lm_head);
        };

        LOG_INFO("[WeightManager] prepareWeightsForDevice(" << device.to_string()
                                                            << " layers=[" << first_layer << ", " << last_layer << ")"
                                                            << " embed=" << has_embedding << " lm_head=" << has_lm_head << ")");

        return prepareWeightsForDeviceImpl(device, layer_filter);
    }

    bool WeightManager::prepareWeightsForDeviceImpl(
        DeviceId device,
        std::function<bool(const std::string &)> layer_filter,
        const FrozenModelWeightSet *frozen_weights,
        bool include_expert_jobs)
    {
        const bool is_gpu = device.is_gpu();
        const char *device_name = device.is_rocm() ? "ROCm" : (device.is_cuda() ? "CUDA" : "CPU");

        // Step 1: Pack GEMM weights (async on GPU for overlap with step 2)
        // GPU devices use the GPU-native pipeline (LoadOrchestrator) which does
        // single VRAM alloc + pipelined H2D + GPU repack kernels.
        // CPU devices use the CPU-side VNNI packing path.
        std::future<bool> gemm_future;
        if (is_gpu)
        {
            LOG_INFO("[WeightManager] Using GPU weight loading pipeline for " << device_name);
            gemm_future = std::async(std::launch::async, [this, device, &layer_filter, frozen_weights, include_expert_jobs]()
                                     { return packGemmWeightsViaPipeline(device, layer_filter, frozen_weights, include_expert_jobs); });
        }

        // Step 2: Upload non-GEMM weights (norms, embeddings) to device
        bool non_gemm_ok = true;
        if (is_gpu)
        {
            non_gemm_ok = uploadNonGemmWeights(device, layer_filter);
            if (!non_gemm_ok)
            {
                LOG_ERROR("[WeightManager] Non-GEMM weight upload failed for " << device_name);
            }
        }

        // Step 3: Wait for GEMM pack (or do sync pack for CPU)
        bool gemm_ok = true;
        if (is_gpu)
        {
            gemm_ok = gemm_future.get();
        }
        else
        {
            gemm_ok = packGemmWeights(device, nullptr, /*release_raw_data=*/false, layer_filter);
        }

        if (!gemm_ok)
        {
            LOG_WARN("[WeightManager] GEMM weight packing failed for " << device_name
                                                                       << "; required prepared kernels may be unavailable");
        }

        return gemm_ok && non_gemm_ok;
    }

    // =========================================================================
    // Weight Lifecycle (convenience entry points)
    // =========================================================================

    bool WeightManager::finalizeForDevice(DeviceId device)
    {
        const bool is_gpu = device.is_gpu();
        const char *device_name = device.is_rocm() ? "ROCm" : (device.is_cuda() ? "CUDA" : "CPU");

        // Phase 9: Mark materialization complete before preparation
        if (!lifecycle_gates_.materialization_complete)
        {
            markMaterializationComplete();
        }

        // Phase 1: Prepare (pack + upload, no release)
        bool ok = prepareWeightsForDevice(device);

        // Phase 9: Mark device preparation complete after packing/upload
        if (ok && !lifecycle_gates_.device_preparation_complete)
        {
            markDevicePreparationComplete();
        }

        // Phase 2: Release host copies (only for GPU, since CPU reads from host)
        if (is_gpu && ok)
        {
            size_t released = releaseAllHostWeightData();
            LOG_INFO("[WeightManager] finalizeForDevice(" << device_name
                                                          << "): released " << released << " host tensors");
        }

        return ok;
    }

    bool WeightManager::finalizeForDevices(const std::vector<DeviceId> &devices,
                                           bool release_host_data)
    {
        if (devices.empty())
        {
            LOG_WARN("[WeightManager] finalizeForDevices called with empty device list");
            return true;
        }

        // Step 1: Clone and upload all weights to all devices
        LOG_INFO("[WeightManager] finalizeForDevices: pre-loading weights for "
                 << devices.size() << " devices");
        if (!preloadForDevices(devices))
        {
            LOG_ERROR("[WeightManager] finalizeForDevices: preloadForDevices failed");
            return false;
        }

        // Step 2: Pack GEMM weights per device.
        // GPU devices use the GPU-native pipeline (LoadOrchestrator) with
        // pipelined H2D + GPU repack kernels. CPU devices use CPU-side packing.
        // Each GPU gets its own LoadOrchestrator, VRAM pool, and pinned ring
        // buffer — fully independent, safe to parallelize.
        bool all_ok = true;

        const auto gemm_wall_start = std::chrono::high_resolution_clock::now();

        {
            std::vector<std::future<bool>> gemm_futures;
            std::vector<std::chrono::high_resolution_clock::time_point> per_device_start(devices.size());
            gemm_futures.reserve(devices.size());

            for (size_t i = 0; i < devices.size(); ++i)
            {
                const auto &dev = devices[i];
                per_device_start[i] = std::chrono::high_resolution_clock::now();

                if (dev.is_gpu())
                {
                    LOG_INFO("[WeightManager] finalizeForDevices: launching GPU pipeline for "
                             << dev.toString());
                    gemm_futures.push_back(std::async(std::launch::async,
                        [this, dev]() { return packGemmWeightsViaPipeline(dev, nullptr); }));
                }
                else
                {
                    gemm_futures.push_back(std::async(std::launch::async,
                        [this, dev]() { return packGemmWeights(dev, nullptr, /*release_raw_data=*/false); }));
                }
            }

            for (size_t i = 0; i < devices.size(); ++i)
            {
                const bool ok = gemm_futures[i].get();
                const auto pack_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - per_device_start[i]).count();
                if (!ok)
                {
                    LOG_WARN("[WeightManager] GEMM packing failed for device " << devices[i].toString());
                    all_ok = false;
                }
                const char *mode = devices[i].is_gpu() ? "GPU pipeline" : "CPU repack";
                LOG_INFO("[WeightManager] packGemmWeights for " << devices[i].toString()
                         << " completed in " << pack_ms << " ms (" << mode << ")");
            }
        }

        const auto gemm_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - gemm_wall_start).count();
        LOG_INFO("[WeightManager] All " << devices.size() << " devices GEMM packed in "
                 << gemm_wall_ms << " ms wall-clock"
                 << (devices.size() > 1 ? " (parallel)" : ""));

        // Step 3: Release all host weight data (unless caller opts out — e.g.
        // nested TP-in-PP, where a later PP stage on a different device still
        // needs host copies to clone from).
        if (release_host_data)
        {
            size_t released = releaseAllHostWeightData();
            LOG_INFO("[WeightManager] finalizeForDevices: released " << released
                                                                     << " host tensors across " << devices.size() << " devices");
        }
        else
        {
            LOG_INFO("[WeightManager] finalizeForDevices: retaining host weight data "
                     "(nested TP-in-PP; outer caller will release)");
        }

        // Step 4: Return freed memory to the OS.
        // glibc malloc keeps freed blocks in its arena and only returns memory
        // above the brk/mmap threshold.  After releasing hundreds of weight
        // tensors (several GB), the arena is heavily fragmented.  malloc_trim
        // forces glibc to release free pages back to the kernel via madvise.
#ifdef __linux__
        if (release_host_data)
        {
            auto report_rss = [](const char *label)
            {
                std::ifstream status("/proc/self/status");
                std::string line;
                while (std::getline(status, line))
                {
                    if (line.compare(0, 6, "VmRSS:") == 0 ||
                        line.compare(0, 8, "RssAnon:") == 0)
                    {
                        LOG_INFO("[WeightManager] " << label << " " << line);
                    }
                }
            };
            report_rss("Pre-trim");
            ::malloc_trim(0);
            report_rss("Post-trim");
        }
#endif

        return all_ok;
    }

    // =========================================================================
    // Weight Packing and Preloading (folded from WeightPreloader)
    // =========================================================================

    bool WeightManager::packGemmWeights(
        DeviceId target_device,
        PreloadProgressCallback progress_cb,
        bool release_raw_data,
        std::function<bool(const std::string &)> layer_filter)
    {
        using namespace llaminar::v2::kernels;
        using Clock = std::chrono::high_resolution_clock;

        const bool detail_enabled = WeightLoadingProfiler::isEnabled();
        auto classify_weight = [](const std::string &name) -> std::string
        {
            if (name.find("attn_q.weight") != std::string::npos)
                return "attn_q";
            if (name.find("attn_k.weight") != std::string::npos)
                return "attn_k";
            if (name.find("attn_v.weight") != std::string::npos)
                return "attn_v";
            if (name.find("attn_output.weight") != std::string::npos)
                return "attn_output";
            if (name.find("ffn_gate.weight") != std::string::npos)
                return "ffn_gate";
            if (name.find("ffn_up.weight") != std::string::npos)
                return "ffn_up";
            if (name.find("ffn_down.weight") != std::string::npos)
                return "ffn_down";
            if (name == "output.weight")
                return "lm_head";
            if (name == "token_embd.weight")
                return "embedding";
            return "other";
        };

        std::unordered_map<std::string, double> bucket_ms;
        std::vector<std::pair<std::string, double>> per_weight_ms;
        std::mutex detail_mutex;
        auto pack_loop_start = Clock::now();

        // Collect GEMM tensors under lock, then pack outside lock.
        // This allows overlap with other preload operations (e.g., non-GEMM uploads).
        //
        // For multi-device LOCAL TP: check per_device_cache_ first for device-specific
        // sharded tensors. Fall back to cache_ for single-device scenarios.
        std::vector<std::pair<std::string, std::shared_ptr<TensorBase>>> gemm_weights;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            const std::string device_prefix = target_device.to_string() + ":";
            bool has_per_device_entries = false;

            // Check if we have device-specific entries in per_device_cache_
            for (const auto &[key, tensor] : per_device_cache_)
            {
                if (key.compare(0, device_prefix.size(), device_prefix) == 0)
                {
                    has_per_device_entries = true;
                    break;
                }
            }

            if (has_per_device_entries)
            {
                // LOCAL TP: iterate per_device_cache_ entries for this device
                for (const auto &[key, tensor] : per_device_cache_)
                {
                    if (key.compare(0, device_prefix.size(), device_prefix) != 0)
                        continue;
                    // Extract original weight name from key (strip "rocm:0:" prefix)
                    const std::string name = key.substr(device_prefix.size());
                    if (isGemmWeight(name) && tensor)
                    {
                        if (layer_filter && !layer_filter(name))
                            continue;
                        gemm_weights.emplace_back(name, tensor);
                    }
                }
            }

            // Fall back to main cache if per_device_cache had no GEMM weights
            // (e.g. only non-GEMM entries exist in per_device_cache for this device)
            if (gemm_weights.empty())
            {
                gemm_weights.reserve(cache_.size());
                for (const auto &[name, tensor] : cache_)
                {
                    if (isGemmWeight(name) && tensor)
                    {
                        if (layer_filter && !layer_filter(name))
                            continue;
                        gemm_weights.emplace_back(name, tensor);
                    }
                }
            }
        }

        if (gemm_weights.empty())
        {
            LOG_DEBUG("[WeightManager] No GEMM weights to pack");
            return true;
        }

        // Apply weight preprocessor (e.g., activation rotation) before packing.
        // This runs once per weight and replaces the cache entry, so that
        // subsequent getWeightForDevice() returns the preprocessed version.
        if (weight_preprocessor_)
        {
            const auto preproc_start = Clock::now();
            size_t preprocessed_count = 0;

            for (auto &[name, tensor] : gemm_weights)
            {
                auto result = weight_preprocessor_(name, tensor);
                if (result && result != tensor)
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    cache_[name] = result;
                    tensor = result;
                    ++preprocessed_count;
                }
            }

            const auto preproc_end = Clock::now();
            const auto preproc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        preproc_end - preproc_start)
                                        .count();

            if (preprocessed_count > 0)
            {
                LOG_INFO("[WeightManager] Preprocessed " << preprocessed_count << "/" << gemm_weights.size()
                                                         << " weights in " << preproc_ms << " ms");
            }
        }

        LOG_DEBUG("[WeightManager] Packing " << gemm_weights.size() << " GEMM weights for "
                                             << target_device.to_string());

        const size_t total = gemm_weights.size();

        // CPU-only path: single producer, single consumer.
        // GPU devices use packGemmWeightsViaPipeline() instead.
        constexpr unsigned prepare_workers = 1;
        constexpr unsigned upload_workers = 1;
        constexpr size_t queue_capacity = 4;

        struct PreparedJob
        {
            std::string name;
            std::shared_ptr<TensorBase> tensor;
            std::shared_ptr<KernelFactory::PreparedGemmHandle> prepared;
            bool preparation_ok = false;
        };

        std::deque<PreparedJob> job_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_not_empty;
        std::condition_variable queue_not_full;

        std::atomic<size_t> next_prepare_index{0};
        std::atomic<size_t> completed_count{0};
        std::atomic<bool> producer_done{false};
        std::atomic<bool> all_success{true};
        std::atomic<size_t> local_cpu_packed{0};

        auto prepare_job = [&](const std::pair<std::string, std::shared_ptr<TensorBase>> &entry) -> PreparedJob
        {
            PreparedJob job;
            job.name = entry.first;
            job.tensor = entry.second;

            try
            {
                job.prepared = KernelFactory::prepareGemmHandleLocal(job.tensor.get(), target_device);
                job.preparation_ok = (job.prepared != nullptr);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[WeightManager] Prepared GEMM creation failed for " << job.name << ": " << e.what());
                job.preparation_ok = false;
            }

            return job;
        };

        auto upload_job = [&](PreparedJob &job) -> bool
        {
            if (!job.preparation_ok || !job.prepared)
            {
                return false;
            }

            try
            {
                auto *kernel = KernelFactory::getOrCreateGemmEngine(job.prepared.get());
                if (!kernel)
                {
                    return false;
                }
                kernel->prepareWeights();
                if (!kernel->weights_converted())
                {
                    LOG_ERROR("[WeightManager] prepareWeights() did not complete upload for: "
                              << job.name << " on " << target_device.to_string());
                    return false;
                }
                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[WeightManager] GEMM upload failed for " << job.name << ": " << e.what());
                return false;
            }
        };

        auto producer_worker = [&]()
        {
            while (true)
            {
                const size_t index = next_prepare_index.fetch_add(1, std::memory_order_relaxed);
                if (index >= total)
                {
                    return;
                }

                const auto prep_start = Clock::now();
                PreparedJob job = prepare_job(gemm_weights[index]);
                const auto prep_end = Clock::now();

                if (detail_enabled)
                {
                    const double prep_ms = std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
                    WeightLoadingProfiler::addDetail("weights.gemm_pack.stage_prepare", prep_ms);
                }

                if (!job.preparation_ok)
                {
                    markPrepState(job.name, target_device, WeightPrepState::FAILED, true, "prepared GEMM creation failed");
                    all_success.store(false, std::memory_order_relaxed);
                    continue;
                }

                std::unique_lock<std::mutex> queue_lock(queue_mutex);
                const auto slot_wait_start = Clock::now();
                queue_not_full.wait(queue_lock, [&]()
                                    { return job_queue.size() < queue_capacity; });
                if (detail_enabled)
                {
                    const auto slot_wait_end = Clock::now();
                    const double slot_wait_ms = std::chrono::duration<double, std::milli>(slot_wait_end - slot_wait_start).count();
                    WeightLoadingProfiler::addDetail("weights.gemm_pack.slot_wait_time", slot_wait_ms);
                }
                job_queue.push_back(std::move(job));
                queue_lock.unlock();
                queue_not_empty.notify_one();
            }
        };

        auto consumer_worker = [&]()
        {
            while (true)
            {
                PreparedJob job;
                {
                    std::unique_lock<std::mutex> queue_lock(queue_mutex);
                    const auto bubble_wait_start = Clock::now();
                    queue_not_empty.wait(queue_lock, [&]()
                                         { return !job_queue.empty() || producer_done.load(std::memory_order_relaxed); });
                    if (detail_enabled)
                    {
                        const auto bubble_wait_end = Clock::now();
                        const double bubble_wait_ms = std::chrono::duration<double, std::milli>(bubble_wait_end - bubble_wait_start).count();
                        WeightLoadingProfiler::addDetail("weights.gemm_pack.pipeline_bubble_time", bubble_wait_ms);
                    }

                    if (job_queue.empty())
                    {
                        return;
                    }

                    job = std::move(job_queue.front());
                    job_queue.pop_front();
                }

                queue_not_full.notify_one();

                const auto upload_start = Clock::now();
                const bool upload_ok = upload_job(job);
                const auto upload_end = Clock::now();

                if (detail_enabled)
                {
                    const double upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
                    {
                        std::lock_guard<std::mutex> detail_lock(detail_mutex);
                        bucket_ms[classify_weight(job.name)] += upload_ms;
                        per_weight_ms.emplace_back(job.name, upload_ms);
                    }
                    WeightLoadingProfiler::addDetail("weights.gemm_pack.stage_upload", upload_ms);
                }

                if (!upload_ok)
                {
                    markPrepState(job.name, target_device, WeightPrepState::FAILED, true, "prepareWeights failed");
                    LOG_ERROR("[WeightManager] Failed to pack weight: " << job.name);
                    all_success.store(false, std::memory_order_relaxed);
                }
                else
                {
                    markPrepState(job.name, target_device, WeightPrepState::PACKED_HOST, true, "GEMM packed on CPU");
                    local_cpu_packed.fetch_add(1, std::memory_order_relaxed);
                    // Only release host data for quantized tensors on CPU.
                    // Quantized GEMM kernels (VNNI) prepack data into their own buffers,
                    // so the original host data is no longer needed.
                    // Floating-point GEMM (oneDNN) reads weight_tensor_->data() live
                    // at every inference call — releasing would cause a null dereference.
                    if (release_raw_data && job.tensor &&
                        dynamic_cast<IINT8Unpackable *>(job.tensor.get()))
                    {
                        job.tensor->release_host_weight_data();
                    }

                    markPrepState(job.name, target_device, WeightPrepState::READY, true, "GEMM weight ready");
                    evaluateReclaimEligibility(job.name, true);

                    // Release mmap pages for this weight now that the GEMM engine
                    // has its own copy (VNNI-packed on CPU).
                    // is_mmap_data() checks the tensor's data is actually in an
                    // mmap'd region. is_view() is NOT safe here — TensorSlice
                    // (from TP sharding) returns is_view()=true but has heap data.
                    // MADV_DONTNEED on heap pages zeroes them, causing corruption.
                    if (job.tensor && job.tensor->is_mmap_data())
                    {
                        MmapRegion::adviseDontneedRange(
                            job.tensor->raw_data(), job.tensor->size_bytes());
                    }
                }

                const size_t finished = completed_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (progress_cb)
                {
                    progress_cb(finished, total, job.name);
                }
            }
        };

        std::vector<std::future<void>> producer_futures;
        producer_futures.reserve(prepare_workers);
        for (unsigned i = 0; i < prepare_workers; ++i)
        {
            producer_futures.emplace_back(std::async(std::launch::async, producer_worker));
        }

        std::vector<std::future<void>> consumer_futures;
        consumer_futures.reserve(upload_workers);
        for (unsigned i = 0; i < upload_workers; ++i)
        {
            consumer_futures.emplace_back(std::async(std::launch::async, consumer_worker));
        }

        for (auto &future : producer_futures)
        {
            future.get();
        }

        producer_done.store(true, std::memory_order_relaxed);
        queue_not_empty.notify_all();

        for (auto &future : consumer_futures)
        {
            future.get();
        }

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            num_cpu_packed_ += local_cpu_packed.load(std::memory_order_relaxed);
        }

        if (detail_enabled)
        {
            const auto pack_loop_end = Clock::now();
            const double total_pack_ms = std::chrono::duration<double, std::milli>(pack_loop_end - pack_loop_start).count();
            WeightLoadingProfiler::addDetail("weights.gemm_pack.loop_total", total_pack_ms);

            {
                std::lock_guard<std::mutex> detail_lock(detail_mutex);
                for (const auto &[bucket, ms] : bucket_ms)
                {
                    WeightLoadingProfiler::addDetail(std::string("weights.gemm_pack.bucket.") + bucket, ms);
                }

                std::sort(per_weight_ms.begin(), per_weight_ms.end(), [](const auto &a, const auto &b)
                          { return a.second > b.second; });

                const size_t top_n = std::min<size_t>(8, per_weight_ms.size());
                for (size_t i = 0; i < top_n; ++i)
                {
                    WeightLoadingProfiler::addDetail(std::string("weights.gemm_pack.weight_top.") + per_weight_ms[i].first,
                                                     per_weight_ms[i].second);
                }
            }
        }

        return all_success.load(std::memory_order_relaxed);
    }

    // -------------------------------------------------------------------------
    // Helper: compute raw byte size for a 2D quantized view whose size_bytes()
    // may report 0 (views alias parent storage without tracking their own size).
    // Duplicated from MoEExpertWeightService.cpp — kept as a static helper here
    // to avoid cross-module dependencies.
    // -------------------------------------------------------------------------
    static size_t quantizedViewRawBytes(const TensorBase &tensor)
    {
        const size_t reported = tensor.size_bytes();
        if (reported > 0)
            return reported;

        const auto &shape = tensor.shape();
        if (shape.size() != 2)
            return 0;

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        auto bytes_for = [rows, cols](size_t block_size, size_t block_bytes) -> size_t
        {
            const size_t blocks_per_row = (cols + block_size - 1) / block_size;
            return rows * blocks_per_row * block_bytes;
        };

        switch (tensor.native_type())
        {
        case TensorType::IQ4_NL: return bytes_for(IQ4_NLBlock::BLOCK_SIZE, sizeof(IQ4_NLBlock));
        case TensorType::IQ4_XS: return bytes_for(IQ4_XSBlock::BLOCK_SIZE, sizeof(IQ4_XSBlock));
        case TensorType::Q8_0: return bytes_for(Q8_0Block::BLOCK_SIZE, sizeof(Q8_0Block));
        case TensorType::Q4_0: return bytes_for(Q4_0Block::BLOCK_SIZE, sizeof(Q4_0Block));
        case TensorType::Q4_1: return bytes_for(Q4_1Block::BLOCK_SIZE, sizeof(Q4_1Block));
        case TensorType::Q5_0: return bytes_for(Q5_0Block::BLOCK_SIZE, sizeof(Q5_0Block));
        case TensorType::Q5_1: return bytes_for(Q5_1Block::BLOCK_SIZE, sizeof(Q5_1Block));
        case TensorType::Q2_K: return bytes_for(Q2_KBlock::BLOCK_SIZE, sizeof(Q2_KBlock));
        case TensorType::Q3_K: return bytes_for(Q3_KBlock::BLOCK_SIZE, sizeof(Q3_KBlock));
        case TensorType::Q4_K: return bytes_for(Q4_KBlock::BLOCK_SIZE, sizeof(Q4_KBlock));
        case TensorType::Q5_K: return bytes_for(Q5_KBlock::BLOCK_SIZE, sizeof(Q5_KBlock));
        case TensorType::Q6_K: return bytes_for(Q6_KBlock::BLOCK_SIZE, sizeof(Q6_KBlock));
        case TensorType::Q8_K: return bytes_for(Q8_KBlock::BLOCK_SIZE, sizeof(Q8_KBlock));
        case TensorType::IQ2_XXS: return bytes_for(IQ2_XXSBlock::BLOCK_SIZE, sizeof(IQ2_XXSBlock));
        case TensorType::IQ2_XS: return bytes_for(IQ2_XSBlock::BLOCK_SIZE, sizeof(IQ2_XSBlock));
        case TensorType::IQ2_S: return bytes_for(IQ2_SBlock::BLOCK_SIZE, sizeof(IQ2_SBlock));
        case TensorType::IQ3_XXS: return bytes_for(IQ3_XXSBlock::BLOCK_SIZE, sizeof(IQ3_XXSBlock));
        case TensorType::IQ3_S: return bytes_for(IQ3_SBlock::BLOCK_SIZE, sizeof(IQ3_SBlock));
        case TensorType::IQ1_S: return bytes_for(IQ1_SBlock::BLOCK_SIZE, sizeof(IQ1_SBlock));
        case TensorType::IQ1_M: return bytes_for(IQ1_MBlock::BLOCK_SIZE, sizeof(IQ1_MBlock));
        default: return 0;
        }
    }

    // -------------------------------------------------------------------------
    // Helper: parse layer index from "blk.N.ffn_gate_exps.weight" etc.
    // Returns -1 on parse failure.
    // -------------------------------------------------------------------------
    static int parseMoELayerIndex(const std::string &name)
    {
        // Expect "blk.N.<suffix>"
        if (name.compare(0, 4, "blk.") != 0)
            return -1;
        auto dot2 = name.find('.', 4);
        if (dot2 == std::string::npos)
            return -1;
        try
        {
            return std::stoi(name.substr(4, dot2 - 4));
        }
        catch (...)
        {
            return -1;
        }
    }

    static const char *moeWeightRoleName(ExpertGemmRegistry::WeightRole role)
    {
        switch (role)
        {
        case ExpertGemmRegistry::WeightRole::GATE: return "gate";
        case ExpertGemmRegistry::WeightRole::UP: return "up";
        case ExpertGemmRegistry::WeightRole::DOWN: return "down";
        default: return "unknown";
        }
    }

    static bool parseMoEExpertParentName(const std::string &name,
                                         int &layer_idx,
                                         ExpertGemmRegistry::WeightRole &role)
    {
        layer_idx = parseMoELayerIndex(name);
        if (layer_idx < 0)
            return false;

        if (name.find("ffn_gate_exps.weight") != std::string::npos)
        {
            role = ExpertGemmRegistry::WeightRole::GATE;
            return true;
        }
        if (name.find("ffn_up_exps.weight") != std::string::npos)
        {
            role = ExpertGemmRegistry::WeightRole::UP;
            return true;
        }
        if (name.find("ffn_down_exps.weight") != std::string::npos)
        {
            role = ExpertGemmRegistry::WeightRole::DOWN;
            return true;
        }

        return false;
    }

    static int moeExpertCountFromParentTensor(const TensorBase &tensor)
    {
        const auto &shape = tensor.shape();
        if (shape.size() != 3)
            return 0;
        return static_cast<int>(shape[2]);
    }

    static std::string formatMiB(size_t bytes)
    {
        return std::to_string(bytes / (1024 * 1024)) + " MiB";
    }

    bool WeightManager::packGemmWeightsViaPipeline(
        DeviceId target_device,
        std::function<bool(const std::string &)> layer_filter,
        const FrozenModelWeightSet *frozen_weights,
        bool include_expert_jobs)
    {
        using namespace llaminar::v2::kernels;
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();

        if (!target_device.is_gpu())
        {
            LOG_ERROR("[WeightManager] GPU pipeline only supports GPU devices");
            return false;
        }

        // ------------------------------------------------------------------
        // Step 1: Collect GEMM weights (same logic as packGemmWeights)
        // ------------------------------------------------------------------
        struct DenseGemmJob
        {
            std::string name;
            TensorBase *tensor = nullptr;
            std::shared_ptr<TensorBase> owner;
            std::optional<WeightBinding> binding;
        };

        std::vector<DenseGemmJob> gemm_weights;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            if (frozen_weights)
            {
                for (const auto &binding : frozen_weights->bindings())
                {
                    if (!binding.prepared.has_value() || binding.prepared->device != target_device)
                        continue;
                    if (!binding.tensor || binding.tensor->shape().size() != 2)
                        continue;
                    if (binding.identity.role == WeightRole::Embedding ||
                        binding.identity.role == WeightRole::Norm ||
                        binding.identity.role == WeightRole::Bias)
                    {
                        continue;
                    }

                    gemm_weights.push_back(DenseGemmJob{
                        binding.identity.canonical_name,
                        binding.tensor,
                        nullptr,
                        binding});
                }
            }
            else
            {
                const std::string device_prefix = target_device.to_string() + ":";
                bool has_per_device_entries = false;
                std::unordered_set<std::string> collected_names;

                for (const auto &[key, tensor] : per_device_cache_)
                {
                    if (key.compare(0, device_prefix.size(), device_prefix) == 0)
                    {
                        has_per_device_entries = true;
                        break;
                    }
                }

                if (has_per_device_entries)
                {
                    for (const auto &[key, tensor] : per_device_cache_)
                    {
                        if (key.compare(0, device_prefix.size(), device_prefix) != 0)
                            continue;
                        const std::string name = key.substr(device_prefix.size());
                        if (isGemmWeight(name) && tensor)
                        {
                            if (layer_filter && !layer_filter(name))
                                continue;
                            gemm_weights.push_back(DenseGemmJob{name, tensor.get(), tensor, std::nullopt});
                            collected_names.insert(name);
                        }
                    }
                }

                // Supplement from the main cache for weights that do not have a
                // device-specific clone. Hybrid PP/TP can populate per_device_cache_
                // partially; treating any per-device entry as complete skips dense
                // PP-stage weights that still live in cache_.
                {
                    for (const auto &[name, tensor] : cache_)
                    {
                        if (collected_names.count(name) != 0)
                            continue;
                        if (isGemmWeight(name) && tensor)
                        {
                            if (layer_filter && !layer_filter(name))
                                continue;
                            gemm_weights.push_back(DenseGemmJob{name, tensor.get(), tensor, std::nullopt});
                            collected_names.insert(name);
                        }
                    }
                }

                // Tied LM head: if this stage owns lm_head (layer_filter passes
                // output.weight) but output.weight wasn't collected, the model
                // ties token_embd.weight as the LM projection weight. Include
                // it in the pipeline so it gets GPU-repacked with the rest of the
                // resident GEMM weights.
                if (!layer_filter || layer_filter("output.weight"))
                {
                    bool has_output_weight = false;
                    for (const auto &job : gemm_weights)
                    {
                        if (job.name == "output.weight") { has_output_weight = true; break; }
                    }
                    if (!has_output_weight)
                    {
                        std::shared_ptr<TensorBase> embed_tensor;
                        const std::string device_key = target_device.to_string() + ":token_embd.weight";
                        auto pdit = per_device_cache_.find(device_key);
                        if (pdit != per_device_cache_.end() && pdit->second)
                            embed_tensor = pdit->second;
                        else
                        {
                            auto it = cache_.find("token_embd.weight");
                            if (it != cache_.end() && it->second)
                                embed_tensor = it->second;
                        }
                        if (embed_tensor)
                        {
                            // The tensor data comes from token_embd.weight, but the
                            // graph owns and resolves the LM projection through the
                            // canonical output.weight binding. Registering the
                            // prepared handle under output.weight keeps tied-embedding
                            // models on the same binding-first path as untied models.
                            gemm_weights.push_back(DenseGemmJob{"output.weight", embed_tensor.get(), embed_tensor, std::nullopt});
                            collected_names.insert("output.weight");
                            LOG_INFO("[WeightManager] GPU pipeline: including tied token_embd.weight as canonical output.weight LM head GEMM");
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 1b: Collect MoE expert weights (3D _exps.weight tensors)
        //
        // These are filtered OUT by isGemmWeight() because isNonGemmWeight()
        // returns true for _exps.weight. We scan for them separately, create
        // 2D views for each expert, and include them in the same orchestrator.
        // ------------------------------------------------------------------
        struct MoEExpertJob
        {
            int layer_idx;
            int expert_idx;
            ExpertGemmRegistry::WeightRole role;
            std::string slot_name;               // e.g., "moe_L0_gate_e5"
            std::shared_ptr<TensorBase> view;     // 2D expert view (keeps parent alive)
        };
        std::vector<MoEExpertJob> moe_jobs;

        // Track 3D parent tensors by name so release readiness is tensor-specific.
        std::vector<std::pair<std::string, std::shared_ptr<TensorBase>>> moe_parent_tensors;

        if (include_expert_jobs)
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Group 3D expert tensors by layer: layer_idx -> {gate, up, down}
            struct MoELayerTensors
            {
                std::shared_ptr<TensorBase> gate;
                std::shared_ptr<TensorBase> up;
                std::shared_ptr<TensorBase> down;
                std::string gate_name;
                std::string up_name;
                std::string down_name;
            };
            std::unordered_map<int, MoELayerTensors> moe_layers;

            for (const auto &[name, tensor] : cache_)
            {
                if (!tensor || name.find("_exps.weight") == std::string::npos)
                    continue;
                if (layer_filter && !layer_filter(name))
                    continue;

                int layer_idx = parseMoELayerIndex(name);
                if (layer_idx < 0)
                    continue;

                if (name.find("ffn_gate_exps.weight") != std::string::npos)
                {
                    moe_layers[layer_idx].gate = tensor;
                    moe_layers[layer_idx].gate_name = name;
                }
                else if (name.find("ffn_up_exps.weight") != std::string::npos)
                {
                    moe_layers[layer_idx].up = tensor;
                    moe_layers[layer_idx].up_name = name;
                }
                else if (name.find("ffn_down_exps.weight") != std::string::npos)
                {
                    moe_layers[layer_idx].down = tensor;
                    moe_layers[layer_idx].down_name = name;
                }
            }

            // Create 2D expert views for each complete layer
            for (auto &[layer_idx, tensors] : moe_layers)
            {
                if (!tensors.gate || !tensors.up || !tensors.down)
                {
                    LOG_WARN("[WeightManager] GPU pipeline: incomplete MoE expert tensors for layer "
                             << layer_idx << " — skipping");
                    continue;
                }

                // GGUF 3D: shape[0]=cols (K, fastest), shape[1]=rows_per_expert, shape[2]=num_experts
                const auto &gate_shape = tensors.gate->shape();
                if (gate_shape.size() != 3)
                {
                    LOG_WARN("[WeightManager] GPU pipeline: MoE gate tensor for layer "
                             << layer_idx << " is not 3D — skipping");
                    continue;
                }

                const int num_experts = static_cast<int>(gate_shape[2]);

                moe_parent_tensors.emplace_back(tensors.gate_name, tensors.gate);
                moe_parent_tensors.emplace_back(tensors.up_name, tensors.up);
                moe_parent_tensors.emplace_back(tensors.down_name, tensors.down);

                struct RoleTensor
                {
                    ExpertGemmRegistry::WeightRole role;
                    const char *tag;
                    TensorBase *tensor_3d;
                };
                RoleTensor roles[] = {
                    {ExpertGemmRegistry::WeightRole::GATE, "gate", tensors.gate.get()},
                    {ExpertGemmRegistry::WeightRole::UP, "up", tensors.up.get()},
                    {ExpertGemmRegistry::WeightRole::DOWN, "down", tensors.down.get()},
                };

                for (const auto &rt : roles)
                {
                    // Each role tensor may have different dimensions
                    // (e.g., down is [intermediate, d_model, num_experts] while gate/up are [d_model, intermediate, num_experts])
                    const auto &role_shape = rt.tensor_3d->shape();
                    const size_t role_cols = role_shape[0];
                    const size_t role_rows_per_expert = role_shape[1];
                    const size_t role_elements_per_expert = role_rows_per_expert * role_cols;

                    for (int e = 0; e < num_experts; ++e)
                    {
                        const size_t element_offset = static_cast<size_t>(e) * role_elements_per_expert;
                        std::vector<size_t> view_shape = {role_rows_per_expert, role_cols};
                        auto view = rt.tensor_3d->create_view(view_shape, element_offset);
                        if (!view)
                        {
                            throw std::runtime_error(
                                "[WeightManager] GPU pipeline: failed to create expert view for layer "
                                + std::to_string(layer_idx) + " " + rt.tag + " expert " + std::to_string(e)
                                + " (shape=[" + std::to_string(role_rows_per_expert) + "," + std::to_string(role_cols)
                                + "], offset=" + std::to_string(element_offset) + ")");
                        }

                        std::string slot_name = "moe_L" + std::to_string(layer_idx) + "_"
                                                + rt.tag + "_e" + std::to_string(e);

                        moe_jobs.push_back({layer_idx, e, rt.role, std::move(slot_name), std::move(view)});
                    }
                }

                LOG_DEBUG("[WeightManager] GPU pipeline: collected " << num_experts
                          << " experts × 3 roles for MoE layer " << layer_idx);
            }
        }

        if (gemm_weights.empty() && moe_jobs.empty())
        {
            LOG_DEBUG("[WeightManager] GPU pipeline: no weights to load");
            return true;
        }

        LOG_INFO("[WeightManager] GPU pipeline: loading " << gemm_weights.size()
                 << " GEMM weights + " << moe_jobs.size()
                 << " MoE expert slots for " << target_device.to_string());

        // ------------------------------------------------------------------
        // Step 2: Get backend
        // ------------------------------------------------------------------
        IBackend *backend = target_device.is_rocm() ? getROCmBackend()
                            : target_device.is_cuda() ? getCUDABackend()
                                                      : nullptr;
        if (!backend)
        {
            LOG_ERROR("[WeightManager] GPU pipeline: no backend for " << target_device.to_string());
            return false;
        }

        // ------------------------------------------------------------------
        // Step 3: Create orchestrator and plan weights
        // ------------------------------------------------------------------
        auto orchestrator = std::make_shared<LoadOrchestrator>(backend);
        orchestrator->addDevice(target_device.ordinal);

        size_t max_raw_bytes = 0;
        size_t planned_count = 0;
        std::vector<std::pair<std::string, const NativeVnniFormatInfo *>> weight_formats;
        weight_formats.reserve(gemm_weights.size());

        for (const auto &job : gemm_weights)
        {
            const std::string &name = job.name;
            TensorBase *tensor = job.tensor;
            if (!tensor)
                continue;

            auto *unpackable = dynamic_cast<IINT8Unpackable *>(tensor);
            const NativeVnniFormatInfo *vnni = unpackable ? unpackable->vnniFormatInfo() : nullptr;
            if (!vnni)
            {
                // Floating-point types (FP32, FP16, BF16) don't use VNNI repack —
                // they go through the pipeline as raw H2D copies (no GPU repack kernel).
                const auto type = tensor->native_type();
                if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
                {
                    const int N = static_cast<int>(tensor->rows());
                    const int K = static_cast<int>(tensor->cols());
                    const size_t raw_bytes = tensor->size_bytes();

                    orchestrator->planRawWeight(target_device.ordinal, name, N, K, raw_bytes);
                    max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
                    ++planned_count;
                    weight_formats.emplace_back(name, nullptr);
                    continue;
                }
                // Any quantized type that doesn't implement IINT8Unpackable is a bug.
                throw std::runtime_error(
                    "[WeightManager] GPU pipeline: weight does not implement "
                    "IINT8Unpackable (unsupported quantized type for GPU repack): " + name);
            }

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());
            const size_t raw_bytes = tensor->size_bytes();

            orchestrator->planWeight(target_device.ordinal, name, N, K,
                                     vnni->payload_bytes, vnni->is_asymmetric,
                                     vnni->has_emins, raw_bytes);
            max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
            ++planned_count;
            weight_formats.emplace_back(name, vnni);
        }

        // Plan MoE expert weights into the same orchestrator
        std::vector<const NativeVnniFormatInfo *> moe_vnni_infos;
        moe_vnni_infos.reserve(moe_jobs.size());
        for (const auto &mj : moe_jobs)
        {
            auto *unpackable = dynamic_cast<IINT8Unpackable *>(mj.view.get());
            const NativeVnniFormatInfo *vnni = unpackable ? unpackable->vnniFormatInfo() : nullptr;
            if (!vnni)
            {
                // Floating-point types (FP32, FP16, BF16) don't use VNNI repack —
                // they go through the pipeline as raw H2D copies.
                const auto type = mj.view->native_type();
                if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
                {
                    const int N = static_cast<int>(mj.view->rows());
                    const int K = static_cast<int>(mj.view->cols());
                    const size_t raw_bytes = mj.view->size_bytes();

                    orchestrator->planRawWeight(target_device.ordinal, mj.slot_name, N, K, raw_bytes);
                    max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
                    ++planned_count;
                    moe_vnni_infos.push_back(nullptr);
                    continue;
                }
                // Any quantized type that doesn't implement IINT8Unpackable is a bug.
                throw std::runtime_error(
                    "[WeightManager] GPU pipeline: MoE expert view does not implement "
                    "IINT8Unpackable (unsupported quantized type for GPU repack): " + mj.slot_name);
            }

            // 2D view: rows()=rows_per_expert, cols()=cols
            const int N = static_cast<int>(mj.view->rows());
            const int K = static_cast<int>(mj.view->cols());
            const size_t raw_bytes = quantizedViewRawBytes(*mj.view);

            orchestrator->planWeight(target_device.ordinal, mj.slot_name, N, K,
                                     vnni->payload_bytes, vnni->is_asymmetric,
                                     vnni->has_emins, raw_bytes);
            max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
            ++planned_count;
            moe_vnni_infos.push_back(vnni);
        }

        if (planned_count == 0)
        {
            LOG_WARN("[WeightManager] GPU pipeline: no weights to load");
            return false;
        }

        // ------------------------------------------------------------------
        // Step 4: Allocate VRAM pool + pinned ring buffer
        // ------------------------------------------------------------------
        const auto &rocm_cfg = debugEnv().rocm;
        const int repack_streams = rocm_cfg.repack_streams;
        const auto *planned_pool = orchestrator->getPool(target_device.ordinal);
        const size_t planned_weight_bytes = planned_pool ? planned_pool->totalPlannedBytes() : 0;
        const size_t staging_bytes = max_raw_bytes * static_cast<size_t>(std::max(0, repack_streams));
        const size_t required_vram_bytes = planned_weight_bytes + staging_bytes;
        const size_t free_vram_bytes = backend->deviceMemoryFree(target_device.ordinal);
        const size_t total_vram_bytes = backend->deviceMemoryTotal(target_device.ordinal);
        const size_t safety_margin_bytes = std::max<size_t>(512ULL * 1024ULL * 1024ULL,
                                                            total_vram_bytes / 20ULL); // 5%, at least 512 MiB

        if (free_vram_bytes > 0 && required_vram_bytes + safety_margin_bytes > free_vram_bytes)
        {
            LOG_ERROR("[WeightManager] GPU pipeline VRAM preflight failed for "
                      << target_device.to_string()
                      << ": required=" << formatMiB(required_vram_bytes)
                      << " available_after_margin="
                      << formatMiB(free_vram_bytes > safety_margin_bytes ? free_vram_bytes - safety_margin_bytes : 0)
                      << " free=" << formatMiB(free_vram_bytes)
                      << " total=" << formatMiB(total_vram_bytes)
                      << " planned_weights=" << formatMiB(planned_weight_bytes)
                      << " staging=" << formatMiB(staging_bytes)
                      << " safety_margin=" << formatMiB(safety_margin_bytes)
                      << ". Mitigations: set LLAMINAR_WEIGHT_STREAMING=1, use a smaller model, "
                      << "reduce context/KV cache pressure, or reduce resident experts.");
            return false;
        }

        LOG_DEBUG("[WeightManager] GPU pipeline VRAM preflight for " << target_device.to_string()
                  << ": required=" << formatMiB(required_vram_bytes)
                  << " planned_weights=" << formatMiB(planned_weight_bytes)
                  << " staging=" << formatMiB(staging_bytes)
                  << " free=" << formatMiB(free_vram_bytes)
                  << " safety_margin=" << formatMiB(safety_margin_bytes));

        orchestrator->allocate(max_raw_bytes, rocm_cfg.repack_streams);

        // ------------------------------------------------------------------
        // Step 5: Create weight jobs
        // ------------------------------------------------------------------
        for (size_t i = 0; i < gemm_weights.size(); ++i)
        {
            const auto &dense_job = gemm_weights[i];
            const std::string &name = dense_job.name;
            TensorBase *tensor = dense_job.tensor;
            const auto *vnni = weight_formats[i].second;
            if (!tensor)
                continue;
            if (!vnni)
            {
                // Floating-point weight: submit as RAW_FP passthrough job
                const auto type = tensor->native_type();
                if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
                {
                    WeightJob job;
                    job.name = name;
                    job.host_raw_data = tensor->raw_data();
                    job.raw_bytes = tensor->size_bytes();
                    job.format = RepackFormat::RAW_FP;
                    job.N = static_cast<int>(tensor->rows());
                    job.K = static_cast<int>(tensor->cols());
                    job.is_asymmetric = false;
                    orchestrator->addWeightJob(target_device.ordinal, job);
                }
                continue;
            }

            auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
            if (!repack_fmt)
            {
                LOG_WARN("[WeightManager] GPU pipeline: unsupported format for " << name
                         << " (codebook=" << static_cast<int>(vnni->codebook_id)
                         << ", superblock=" << vnni->is_superblock << ")");
                continue;
            }

            WeightJob job;
            job.name = name;
            job.host_raw_data = tensor->raw_data();
            job.raw_bytes = tensor->size_bytes();
            job.format = *repack_fmt;
            job.N = static_cast<int>(tensor->rows());
            job.K = static_cast<int>(tensor->cols());
            job.is_asymmetric = vnni->is_asymmetric;

            orchestrator->addWeightJob(target_device.ordinal, job);
        }

        // Add MoE expert weight jobs
        for (size_t i = 0; i < moe_jobs.size(); ++i)
        {
            const auto *vnni = moe_vnni_infos[i];
            if (!vnni)
            {
                // Floating-point MoE expert: submit as RAW_FP passthrough job
                const auto &mj = moe_jobs[i];
                const auto type = mj.view->native_type();
                if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
                {
                    WeightJob job;
                    job.name = mj.slot_name;
                    job.host_raw_data = mj.view->raw_data();
                    job.raw_bytes = mj.view->size_bytes();
                    job.format = RepackFormat::RAW_FP;
                    job.N = static_cast<int>(mj.view->rows());
                    job.K = static_cast<int>(mj.view->cols());
                    job.is_asymmetric = false;
                    orchestrator->addWeightJob(target_device.ordinal, job);
                }
                continue;
            }

            auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
            if (!repack_fmt)
            {
                LOG_WARN("[WeightManager] GPU pipeline: unsupported MoE format for "
                         << moe_jobs[i].slot_name);
                continue;
            }

            WeightJob job;
            job.name = moe_jobs[i].slot_name;
            job.host_raw_data = moe_jobs[i].view->raw_data();
            job.raw_bytes = quantizedViewRawBytes(*moe_jobs[i].view);
            job.format = *repack_fmt;
            job.N = static_cast<int>(moe_jobs[i].view->rows());
            job.K = static_cast<int>(moe_jobs[i].view->cols());
            job.is_asymmetric = vnni->is_asymmetric;

            orchestrator->addWeightJob(target_device.ordinal, job);
        }

        // ------------------------------------------------------------------
        // Step 6: Execute pipeline (H2D + GPU repack)
        // ------------------------------------------------------------------
        orchestrator->load();

        // ------------------------------------------------------------------
        // Step 7: Create GEMM kernels from pool slots and register in KernelFactory
        // ------------------------------------------------------------------
        auto *pool = orchestrator->getPool(target_device.ordinal);
        if (!pool)
        {
            LOG_ERROR("[WeightManager] GPU pipeline: pool not found after load");
            return false;
        }

        size_t registered = 0;
        auto prepared_kind_for_device = [](DeviceId device) {
            if (device.is_cuda()) return PreparedWeightKind::CudaInt8PackedGemm;
            if (device.is_rocm()) return PreparedWeightKind::RocmInt8PackedGemm;
            return PreparedWeightKind::CpuPackedGemm;
        };

        auto register_pipeline_gemm = [&](const DenseGemmJob &dense_job,
                                          std::unique_ptr<ITensorGemm> kernel) -> bool
        {
            const std::string &name = dense_job.name;
            TensorBase *tensor = dense_job.tensor;
            if (!tensor || !kernel)
                return false;

            const bool quantized = dynamic_cast<IINT8Unpackable *>(tensor) != nullptr;
            KernelFactory::GemmPreparationKind resolved_kind = quantized
                ? (target_device.is_cuda() ? KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED
                   : target_device.is_rocm() ? KernelFactory::GemmPreparationKind::ROCM_INT8_PACKED
                                             : KernelFactory::GemmPreparationKind::CPU_PACKED)
                : KernelFactory::GemmPreparationKind::FLOATING_POINT;

            auto owned_kernel = std::shared_ptr<ITensorGemm>(std::move(kernel));
            auto prepared_weights = std::make_shared<KernelFactory::PreparedGemmWeights>();
            prepared_weights->kind = resolved_kind;
            prepared_weights->owned_kernel = owned_kernel;
            prepared_weights->kernel = owned_kernel.get();

            auto handle = std::make_shared<KernelFactory::PreparedGemmHandle>();
            handle->tensor = tensor;
            handle->device_id = target_device;
            handle->kind = resolved_kind;
            handle->variant = static_cast<int>(tensor->native_type());
            handle->prepared_weights = std::move(prepared_weights);

            WeightBinding binding;
            if (dense_job.binding.has_value())
            {
                binding = *dense_job.binding;
            }
            else
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                binding.binding_id = next_pipeline_prepared_binding_id_++;
                binding.identity = makeSourceWeightIdentity(name, {}, binding.binding_id);
            }
            binding.tensor = tensor;
            if (binding.slice.source_rows == 0 && binding.slice.source_cols == 0 &&
                binding.slice.row_count == 0 && binding.slice.col_count == 0)
            {
                binding.slice = fullSliceSpec(*tensor);
            }
            binding.residency.home_device = target_device;
            binding.residency.resident_device = target_device;
            binding.residency.host_policy = WeightHostPolicy::ReleasableAfterPreparation;
            binding.immutable = true;

            try
            {
                preparedWeightStore()->registerPreparedGemmHandle(
                    binding,
                    prepared_kind_for_device(target_device),
                    target_device,
                    std::move(handle));
                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[WeightManager] GPU pipeline: failed to register prepared store entry for "
                          << name << ": " << e.what());
                return false;
            }
        };

        for (size_t i = 0; i < gemm_weights.size(); ++i)
        {
            const auto &dense_job = gemm_weights[i];
            const std::string &name = dense_job.name;
            TensorBase *tensor = dense_job.tensor;
            const auto *vnni = weight_formats[i].second;
            if (!tensor)
                continue;

            auto slot = pool->getSlot(name);
            if (!slot)
            {
                // Slot missing means this weight wasn't planned (shouldn't happen)
                if (vnni)
                    LOG_WARN("[WeightManager] GPU pipeline: no slot for " << name);
                continue;
            }

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());

            std::unique_ptr<ITensorGemm> kernel;

            if (!vnni)
            {
                // Floating-point weight: create FP GEMM kernel from raw device pointer
                const auto type = tensor->native_type();
#ifdef HAVE_ROCM
                if (target_device.is_rocm())
                {
                    auto precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP32;
                    if (type == TensorType::FP16)
                        precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP16;
                    else if (type == TensorType::BF16)
                        precision = rocm::ROCmFloatingPointGemmKernel::Precision::BF16;
                    kernel = std::make_unique<llaminar2::rocm::ROCmFloatingPointGemmKernel>(
                        slot->d_native_vnni_payload, N, K,
                        target_device.ordinal, precision, orchestrator);
                }
#endif
#ifdef HAVE_CUDA
                if (target_device.is_cuda())
                {
                    auto precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP32;
                    if (type == TensorType::FP16)
                        precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP16;
                    else if (type == TensorType::BF16)
                        precision = cuda::CUDAFloatingPointGemmKernel::Precision::BF16;
                    kernel = std::make_unique<llaminar2::cuda::CUDAFloatingPointGemmKernel>(
                        slot->d_native_vnni_payload, N, K,
                        target_device.ordinal, precision, orchestrator);
                }
#endif
            }
            else
            {
            const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);

#ifdef HAVE_ROCM
            if (target_device.is_rocm())
            {
                kernel = std::make_unique<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    N, K, target_device.ordinal,
                    slot->d_native_vnni_payload,
                    slot->d_native_vnni_scales,
                    slot->d_native_vnni_mins,
                    slot->d_native_vnni_emins,
                    vnni->codebook_id, blocks_per_row,
                    orchestrator); // lifetime owner: keeps VRAM pool alive
            }
#endif

#ifdef HAVE_CUDA
            if (target_device.is_cuda())
            {
                kernel = std::make_unique<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    N, K, target_device.ordinal,
                    slot->d_native_vnni_payload,
                    static_cast<uint16_t *>(slot->d_native_vnni_scales),
                    static_cast<uint16_t *>(slot->d_native_vnni_mins),
                    static_cast<uint32_t *>(slot->d_native_vnni_emins),
                    vnni->codebook_id, blocks_per_row,
                    orchestrator); // lifetime owner: keeps VRAM pool alive
            }
#endif
            } // end else (quantized path)

            if (kernel)
            {
                if (register_pipeline_gemm(dense_job, std::move(kernel)))
                {
                    markPrepState(name, target_device, WeightPrepState::UPLOADED_DEVICE, true,
                                  "GPU pipeline: loaded + registered");
                    markPrepState(name, target_device, WeightPrepState::READY, true,
                                  "GPU pipeline: GEMM weight ready");
                    ++registered;

                    // Release host weight data — the GEMM kernel now owns device copies
                    // in the WeightVRAMPool. PreparedWeightStore registration marks
                    // has_prepared_device_state_, so releaseAllHostWeightData() would
                    // also catch these, but releasing inline saves peak host memory.
                    // Skip for token_embd.weight: it's also used for embedding lookup,
                    // which runs concurrently (Steps 2/2b). Its host data is freed
                    // later by releaseAllHostWeightData().
                    const bool tied_alias = dense_job.binding.has_value() &&
                                            dense_job.binding->identity.derivation == WeightDerivationKind::TiedAlias;
                    if (name != "token_embd.weight" && !tied_alias)
                    {
                        try
                        {
                            tensor->release_host_weight_data();
                        }
                        catch (const std::exception &e)
                        {
                            LOG_DEBUG("[WeightManager] GPU pipeline: failed to release host data for "
                                      << name << ": " << e.what());
                        }
                    }
                }
                else
                {
                    LOG_ERROR("[WeightManager] GPU pipeline: failed to register kernel for " << name);
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 7b: Create MoE expert GEMM kernels and register in ExpertGemmRegistry
        // ------------------------------------------------------------------
        size_t moe_registered = 0;
        for (size_t i = 0; i < moe_jobs.size(); ++i)
        {
            const auto *vnni = moe_vnni_infos[i];
            const auto &mj = moe_jobs[i];

            auto slot = pool->getSlot(mj.slot_name);
            if (!slot)
            {
                if (vnni)
                    LOG_WARN("[WeightManager] GPU pipeline: no slot for MoE expert " << mj.slot_name);
                continue;
            }

            const int N = static_cast<int>(mj.view->rows());
            const int K = static_cast<int>(mj.view->cols());

            std::shared_ptr<ITensorGemm> kernel;

            if (!vnni)
            {
                // Floating-point MoE expert: create FP GEMM kernel
                const auto type = mj.view->native_type();
#ifdef HAVE_ROCM
                if (target_device.is_rocm())
                {
                    auto precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP32;
                    if (type == TensorType::FP16)
                        precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP16;
                    else if (type == TensorType::BF16)
                        precision = rocm::ROCmFloatingPointGemmKernel::Precision::BF16;
                    kernel = std::make_shared<llaminar2::rocm::ROCmFloatingPointGemmKernel>(
                        slot->d_native_vnni_payload, N, K,
                        target_device.ordinal, precision, orchestrator);
                }
#endif
#ifdef HAVE_CUDA
                if (target_device.is_cuda())
                {
                    auto precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP32;
                    if (type == TensorType::FP16)
                        precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP16;
                    else if (type == TensorType::BF16)
                        precision = cuda::CUDAFloatingPointGemmKernel::Precision::BF16;
                    kernel = std::make_shared<llaminar2::cuda::CUDAFloatingPointGemmKernel>(
                        slot->d_native_vnni_payload, N, K,
                        target_device.ordinal, precision, orchestrator);
                }
#endif
            }
            else
            {
            const uint32_t blocks_per_row = static_cast<uint32_t>(K / 32);

#ifdef HAVE_ROCM
            if (target_device.is_rocm())
            {
                kernel = std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    N, K, target_device.ordinal,
                    slot->d_native_vnni_payload,
                    slot->d_native_vnni_scales,
                    slot->d_native_vnni_mins,
                    slot->d_native_vnni_emins,
                    vnni->codebook_id, blocks_per_row,
                    orchestrator);
            }
#endif

#ifdef HAVE_CUDA
            if (target_device.is_cuda())
            {
                kernel = std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    N, K, target_device.ordinal,
                    slot->d_native_vnni_payload,
                    static_cast<uint16_t *>(slot->d_native_vnni_scales),
                    static_cast<uint16_t *>(slot->d_native_vnni_mins),
                    static_cast<uint32_t *>(slot->d_native_vnni_emins),
                    vnni->codebook_id, blocks_per_row,
                    orchestrator);
            }
#endif
            } // end else (quantized path)

            if (kernel)
            {
                ITensorGemm *raw_ptr = kernel.get();
                expert_gemm_registry_.registerEngine(
                    target_device, mj.layer_idx, mj.expert_idx, mj.role,
                    raw_ptr, std::move(kernel));
                ++moe_registered;
            }
        }

        // Mark 3D MoE parent tensors ready only after their exact layer/role has
        // all expert engines registered for this device. Host release happens in
        // releaseAllHostWeightData(), after all expected device tickets are ready.
        for (const auto &[name, parent] : moe_parent_tensors)
        {
            if (!parent)
                continue;

            int layer_idx = -1;
            ExpertGemmRegistry::WeightRole role = ExpertGemmRegistry::WeightRole::GATE;
            const int num_experts = moeExpertCountFromParentTensor(*parent);
            if (!parseMoEExpertParentName(name, layer_idx, role) || num_experts <= 0)
            {
                markPrepState(name, target_device, WeightPrepState::FAILED, true,
                              "GPU pipeline: invalid MoE parent tensor metadata");
                continue;
            }

            if (expert_gemm_registry_.hasCompleteRole(target_device, layer_idx, num_experts, role))
            {
                markPrepState(name, target_device, WeightPrepState::READY, true,
                              std::string("GPU pipeline: MoE ") + moeWeightRoleName(role) +
                                  " expert parent ready");
            }
            else
            {
                markPrepState(name, target_device, WeightPrepState::FAILED, true,
                              std::string("GPU pipeline: incomplete MoE ") + moeWeightRoleName(role) +
                                  " expert registry entries");
            }
        }

        if (moe_registered > 0)
        {
            LOG_INFO("[WeightManager] GPU pipeline: registered " << moe_registered
                     << " MoE expert GEMM kernels in ExpertGemmRegistry");
        }

        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        LOG_INFO("[WeightManager] GPU pipeline: loaded " << registered << "/" << gemm_weights.size()
                 << " dense + " << moe_registered << "/" << moe_jobs.size()
                 << " MoE expert weights in " << std::fixed << std::setprecision(1) << elapsed << " ms");

        // Release orchestrator staging resources now that all weights are loaded.
        // The VRAM pool is kept alive by the GEMM kernels' lifetime_owner_ shared_ptrs,
        // but the staging buffers (pinned ring) are no longer needed.
        orchestrator->finalize();

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            num_gpu_packed_ += registered;
        }

        const size_t total_registered = registered + moe_registered;
        return total_registered == planned_count;
    }

    bool WeightManager::uploadNonGemmWeights(
        DeviceId target_device,
        std::function<bool(const std::string &)> layer_filter)
    {
        if (!target_device.is_gpu())
        {
            LOG_DEBUG("[WeightManager] uploadNonGemmWeights skipped for CPU target");
            return true;
        }

        size_t uploaded_count = 0;

        LOG_DEBUG("[WeightManager] Uploading non-GEMM weights to " << target_device.to_string() << "...");

        // Snapshot non-GEMM names under lock, then process each tensor with
        // lock held only for cache access/clone bookkeeping.
        std::vector<std::string> non_gemm_names;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            non_gemm_names.reserve(cache_.size());
            for (const auto &[name, tensor] : cache_)
            {
                if (!isGemmWeight(name) && tensor)
                {
                    // MoE 3D expert parent tensors are prepared through per-expert
                    // views in the unified GPU pipeline. Uploading the raw parent
                    // as a non-GEMM tensor duplicates several GB of data and can
                    // exhaust VRAM before graph execution.
                    if (name.find("_exps.weight") != std::string::npos)
                        continue;
                    if (layer_filter && !layer_filter(name))
                        continue;
                    non_gemm_names.push_back(name);
                }
            }
        }

        for (const auto &name : non_gemm_names)
        {
            std::shared_ptr<TensorBase> tensor;
            std::shared_ptr<TensorBase> device_tensor_holder;

            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto base_it = cache_.find(name);
                if (base_it == cache_.end() || !base_it->second)
                {
                    throw std::runtime_error(
                        "[WeightManager] Non-GEMM weight missing from cache: " + name);
                }
                tensor = base_it->second;

                // Get the device-specific tensor if multi-GPU, otherwise use original
                if (first_device_.has_value() && first_device_.value() == target_device)
                {
                    // First device uses original tensor
                    device_tensor_holder = tensor;
                }
                else if (first_device_.has_value())
                {
                    // Subsequent devices need clones - check per-device cache
                    std::string cache_key = target_device.to_string() + ":" + name;
                    auto it = per_device_cache_.find(cache_key);
                    if (it != per_device_cache_.end())
                    {
                        device_tensor_holder = it->second;
                    }
                    else
                    {
                        // Create clone for this device
                        auto clone = cloneTensorForDevice(name, tensor, target_device);
                        if (clone)
                        {
                            device_tensor_holder = clone;
                            per_device_cache_[cache_key] = clone;
                        }
                    }
                }
                else
                {
                    // First device not set yet, set it now
                    first_device_ = target_device;
                    device_tensor_holder = tensor;
                }
            }

            if (!device_tensor_holder)
            {
                throw std::runtime_error(
                    "[WeightManager] Failed to get device tensor for: " + name);
            }

            TensorBase *device_tensor = device_tensor_holder.get();

            // Set debug name so transfers can be traced
            device_tensor->setDebugName(name);

            // Upload to GPU using ensureOnDevice (no GEMM packing needed)
            if (device_tensor->ensureOnDevice(target_device))
            {
                size_t rows = device_tensor->shape()[0];
                size_t cols = device_tensor->shape().size() > 1 ? device_tensor->shape()[1] : 1;
                size_t bytes = rows * cols * sizeof(float);
                markPrepState(name, target_device, WeightPrepState::UPLOADED_DEVICE, false, "non-GEMM ensureOnDevice completed");
                markPrepState(name, target_device, WeightPrepState::READY, false, "non-GEMM ready");
                LOG_TRACE("[WeightManager] Uploaded non-GEMM weight: " << name
                                                                       << " [" << rows << "x" << cols << "] = " << bytes << " bytes");
                uploaded_count++;
            }
            else
            {
                markPrepState(name, target_device, WeightPrepState::FAILED, false, "non-GEMM ensureOnDevice failed");
                LOG_WARN("[WeightManager] Failed to upload non-GEMM weight: " << name);
            }
        }

        LOG_INFO("[WeightManager] Uploaded " << uploaded_count << " non-GEMM weights to "
                                             << target_device.to_string());
        return true;
    }

    size_t WeightManager::releaseAllHostWeightData()
    {
        // Phase 9: Hard gate — do NOT release host data until all lifecycle gates
        // are complete. This replaces the old ad-hoc deferred_host_release_pending_
        // flag in RankOrchestrator. The caller must mark graph_materialization_complete
        // after all graphs have resolved their prepared weight bindings.
        if (!lifecycle_gates_.canReleaseHostData())
        {
            LOG_INFO("[WeightManager] releaseAllHostWeightData() blocked by lifecycle gate "
                     "(state=" << toString(lifecycle_gates_.currentState())
                     << "). Host data retained until all gates complete.");
            return 0;
        }

        // Mark host_release_allowed now that we're proceeding
        lifecycle_gates_.host_release_allowed = true;

        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t released_count = 0;
        size_t skipped_count = 0;
        size_t error_count = 0;
        size_t released_bytes = 0;
        size_t retained_bytes = 0;
        size_t retained_count = 0;
        std::unordered_set<TensorBase *> visited_ptrs;

        auto moe_parent_ready_for_release = [&](const TensorBase &tensor, const std::string &key) -> bool
        {
            int layer_idx = -1;
            ExpertGemmRegistry::WeightRole role = ExpertGemmRegistry::WeightRole::GATE;
            if (!parseMoEExpertParentName(key, layer_idx, role))
                return false;

            const int num_experts = moeExpertCountFromParentTensor(tensor);
            if (num_experts <= 0)
                return false;

            std::lock_guard<std::mutex> prep_lock(prep_ticket_mutex_);
            auto expected_it = expected_devices_by_weight_.find(key);
            auto tickets_it = prep_tickets_.find(key);
            if (expected_it == expected_devices_by_weight_.end() ||
                tickets_it == prep_tickets_.end() ||
                expected_it->second.empty())
            {
                return false;
            }

            const auto &tickets = tickets_it->second;
            return std::all_of(expected_it->second.begin(), expected_it->second.end(),
                               [&](const std::string &device_key)
                               {
                                   auto ticket_it = tickets.find(device_key);
                                   return ticket_it != tickets.end() &&
                                          ticket_it->second.state == WeightPrepState::READY;
                               });
        };

        auto try_release = [&](TensorBase *ptr, const std::string &key)
        {
            if (!ptr || !visited_ptrs.insert(ptr).second)
                return; // null or already visited

            const size_t tensor_bytes = ptr->is_raw_data_released() ? 0 : ptr->size_bytes();

            // Skip tensors that are already released
            if (ptr->is_raw_data_released())
            {
                skipped_count++;
                return;
            }

            // Phase 9: hostDataRequired() handles policy-based retention. The
            // lifecycle gate ensures graph construction has consumed prepared
            // bindings before raw host data can be reclaimed.
            if (hostDataRequired(ptr, key))
            {
                skipped_count++;
                retained_bytes += tensor_bytes;
                retained_count++;
                LOG_DEBUG("[WeightManager] RETAINED host data for " << key
                                                                    << " (" << (tensor_bytes / 1024) << " KB)"
                                                                    << " (host policy requires retention)");
                return;
            }

            // Release host data for tensors that have valid GPU data OR
            // kernel-managed device data (GEMM packed weights, prepared embedding).
            // GEMM kernels upload pre-packed representations to their own device
            // buffers and never read the raw TensorBase data, so the host copy can
            // be freed. Similarly, PreparedEmbeddingWeights hold their own GPU copy.
            if (!ptr->deviceValid())
            {
                bool has_kernel_device_data = ptr->hasPreparedDeviceState();

                // 3D MoE expert tensors (_exps.weight) are safe to release only
                // after this exact parent layer/role has complete expert engines
                // for every expected device. A non-empty global registry is not
                // sufficient proof for an unrelated parent tensor.
                if (!has_kernel_device_data && key.find("_exps.weight") != std::string::npos)
                {
                    has_kernel_device_data = moe_parent_ready_for_release(*ptr, key);
                }

                if (!has_kernel_device_data)
                {
                    skipped_count++;
                    retained_bytes += tensor_bytes;
                    retained_count++;
                    LOG_DEBUG("[WeightManager] RETAINED host data for " << key
                                                                        << " (" << (tensor_bytes / 1024) << " KB)"
                                                                        << " deviceValid=" << ptr->deviceValid()
                                                                        << " hasPreparedDeviceState=" << ptr->hasPreparedDeviceState()
                                                                        << " hostResident=" << ptr->isHostResident()
                                                                        << " type=" << static_cast<int>(ptr->native_type()));
                    return;
                }
            }

            try
            {
                released_bytes += tensor_bytes;
                ptr->release_host_weight_data();
                released_count++;
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[WeightManager] Failed to release host data for tensor "
                         << ptr->debugName() << " (type=" << static_cast<int>(ptr->native_type())
                         << "): " << e.what());
                error_count++;
            }
        };

        // Sweep main cache
        for (auto &[name, tensor] : cache_)
        {
            try_release(tensor.get(), name);
        }

        // Sweep per-device cache (clones for multi-GPU)
        for (auto &[key, tensor] : per_device_cache_)
        {
            try_release(tensor.get(), key);
        }

        LOG_INFO("[WeightManager] Released host weight data: " << released_count
                                                               << " tensors (" << (released_bytes / (1024 * 1024)) << " MB) released, "
                                                               << retained_count << " tensors (" << (retained_bytes / (1024 * 1024)) << " MB) retained, "
                                                               << skipped_count << " already released, " << error_count << " errors"
                                                               << " | cache=" << cache_.size()
                                                               << " per_device=" << per_device_cache_.size()
                                                               << " decode=" << decode_cache_.size());
        return released_count;
    }

    size_t WeightManager::releaseHostResidentWeightData()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t released_count = 0;
        size_t released_bytes = 0;
        std::unordered_set<TensorBase *> visited_ptrs;

        auto try_release = [&](TensorBase *ptr, const std::string &key)
        {
            if (!ptr || !visited_ptrs.insert(ptr).second)
                return;
            if (ptr->is_raw_data_released())
                return;
            if (!ptr->isHostResident())
                return;
            if (hostDataRequired(ptr, key))
            {
                LOG_DEBUG("[WeightManager] RETAINED host-resident weight for " << key
                                                                               << " (host policy requires retention)");
                return;
            }

            size_t tensor_bytes = ptr->size_bytes();
            ptr->release_host_weight_data();
            released_bytes += tensor_bytes;
            released_count++;
            LOG_INFO("[WeightManager] Released host-resident weight: " << key
                                                                       << " (" << (tensor_bytes / (1024 * 1024)) << " MB)");
        };

        for (auto &[name, tensor] : cache_)
        {
            try_release(tensor.get(), name);
        }

        for (auto &[key, tensor] : per_device_cache_)
        {
            try_release(tensor.get(), key);
        }

        if (released_count > 0)
        {
            LOG_INFO("[WeightManager] Post-upload host-resident release: "
                     << released_count << " tensors (" << (released_bytes / (1024 * 1024)) << " MB) freed");

#if defined(__GLIBC__)
            ::malloc_trim(0);
#endif
        }

        // Advise the OS to reclaim mmap physical pages. All GEMM weight data
        // has been packed into interleaved format (owned allocations). Small
        // FP32 weights (norms, biases ~0.7 MB) will transparently re-fault
        // from the page cache on next access.
        loader_.adviseMmapDontneed();

        return released_count;
    }

    // =============================================================================
    // Device-aware weight slicing for LOCAL TP
    // =============================================================================

    std::shared_ptr<TensorBase> WeightManager::sliceRowRange(
        const std::shared_ptr<TensorBase> &tensor,
        size_t row_start,
        size_t row_count)
    {
        if (!tensor)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange: null tensor");
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];

        if (row_start + row_count > out_dim)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange: row_start=" + std::to_string(row_start) +
                                     " + row_count=" + std::to_string(row_count) +
                                     " > out_dim=" + std::to_string(out_dim));
        }

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange currently requires FP32 tensor. "
                                     "For quantized weights, use GGUF loadTensorRowSlice instead.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {row_count, in_dim};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy the specified row range
        const float *src = fp32_tensor->data() + row_start * in_dim;
        float *dst = sliced->mutable_data();
        std::memcpy(dst, src, row_count * in_dim * sizeof(float));

        LOG_TRACE("[WeightManager] sliceRowRange: [" << out_dim << ", " << in_dim
                                                     << "] -> rows [" << row_start << ", " << (row_start + row_count)
                                                     << ") = [" << row_count << ", " << in_dim << "]");

        return sliced;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceColumnRange(
        const std::shared_ptr<TensorBase> &tensor,
        size_t col_start,
        size_t col_count)
    {
        if (!tensor)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange: null tensor");
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t rows = shape[0];
        size_t cols = shape[1];

        if (col_start + col_count > cols)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange: col_start=" + std::to_string(col_start) +
                                     " + col_count=" + std::to_string(col_count) +
                                     " > cols=" + std::to_string(cols));
        }

        if (col_count == 0)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange: col_count cannot be 0");
        }

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange currently requires FP32 tensor. "
                                     "For quantized weights, use GGUF loadTensorColumnSlice instead.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {rows, col_count};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy column range row by row
        const float *src = fp32_tensor->data();
        float *dst = sliced->mutable_data();
        for (size_t row = 0; row < rows; ++row)
        {
            std::memcpy(dst + row * col_count, src + row * cols + col_start, col_count * sizeof(float));
        }

        LOG_TRACE("[WeightManager] sliceColumnRange: [" << rows << ", " << cols
                                                        << "] -> cols [" << col_start << ", " << (col_start + col_count)
                                                        << ") = [" << rows << ", " << col_count << "]");

        return sliced;
    }

    // ========================================================================
    // Sharding helper methods
    // ========================================================================

    // ========================================================================
    // Helper: Compute slice boundaries based on dimension type from config
    // ========================================================================

    bool WeightManager::computeSliceBoundaries(
        const std::string &name,
        size_t total_size,
        const DeviceShardingAssignment &assignment,
        size_t &out_start,
        size_t &out_count) const
    {
        if (!has_sharding_config_)
        {
            throw std::runtime_error("[WeightManager] No sharding config set - cannot compute slice boundaries");
        }

        WeightDimensionType dim_type = sharding_config_.getDimensionType(name);

        switch (dim_type)
        {
        case WeightDimensionType::Heads:
        {
            const int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads in TensorParallelConfig");
            }
            const size_t head_dim = total_size / static_cast<size_t>(total_heads);
            out_start = assignment.head_start * head_dim;
            out_count = assignment.head_count * head_dim;
            LOG_TRACE("[WeightManager] " << name << " (Heads): total_heads=" << total_heads
                                         << " head_dim=" << head_dim
                                         << " -> [" << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::KVHeads:
        {
            const int total_kv_heads = tp_config_->totalKVHeads();
            if (total_kv_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_kv_heads in TensorParallelConfig");
            }
            const size_t head_dim = total_size / static_cast<size_t>(total_kv_heads);
            out_start = assignment.kv_head_start * head_dim;
            out_count = assignment.kv_head_count * head_dim;
            LOG_TRACE("[WeightManager] " << name << " (KVHeads): total_kv_heads=" << total_kv_heads
                                         << " head_dim=" << head_dim
                                         << " -> [" << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::FFNHidden:
        {
            out_start = assignment.d_ff_start;
            out_count = assignment.d_ff_count;
            LOG_TRACE("[WeightManager] " << name << " (FFNHidden): d_ff=["
                                         << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::Vocab:
        {
            out_start = assignment.vocab_start;
            out_count = assignment.vocab_count;
            LOG_TRACE("[WeightManager] " << name << " (Vocab): vocab=["
                                         << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::ProportionalHeads:
        {
            const int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads for ProportionalHeads slicing");
            }
            out_start = total_size * static_cast<size_t>(assignment.head_start) / static_cast<size_t>(total_heads);
            const size_t end = total_size * static_cast<size_t>(assignment.head_start + assignment.head_count) / static_cast<size_t>(total_heads);
            out_count = end - out_start;
            LOG_TRACE("[WeightManager] " << name << " (ProportionalHeads): total_heads=" << total_heads
                                         << " weight_size=" << total_size
                                         << " -> [" << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::Bias1D:
        case WeightDimensionType::None:
        default:
            throw std::runtime_error("[WeightManager] Cannot compute slice boundaries for dimension type " +
                                     std::to_string(static_cast<int>(dim_type)) + " on weight: " + name);
        } // switch
    }

    std::shared_ptr<TensorBase> WeightManager::loadColumnParallel1DBias(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        // 1D bias tensor - slice along single dimension
        size_t total_size = dimensions[0];
        size_t slice_start = 0;
        size_t slice_count = 0;

        // Use config to determine dimension type for slicing
        if (!computeSliceBoundaries(name, total_size, assignment, slice_start, slice_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for 1D tensor: " + name);
        }

        // Load full tensor and slice in memory (biases are small)
        auto full_tensor = loader_.loadTensor(name, device, WeightPrecision::NATIVE);
        if (!full_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load 1D tensor: " + name);
        }
        registerSourceMetadata(name, full_tensor, device);

        // Create sliced tensor
        auto *fp32_full = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_full)
        {
            throw std::runtime_error("[WeightManager] 1D bias must be FP32: " + name);
        }

        std::vector<size_t> slice_shape = {slice_count};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);
        std::memcpy(sliced->mutable_data(),
                    fp32_full->data() + slice_start,
                    slice_count * sizeof(float));

        LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config_->worldSize() << ")"
                                            << " column-parallel 1D " << name
                                            << " [" << total_size << "]"
                                            << " -> [" << slice_start << ", " << (slice_start + slice_count) << ")"
                                            << " = " << slice_count << " elements");

        WeightSliceSpec bias_slice;
        bias_slice.source_rows = total_size;
        bias_slice.source_cols = 1;
        bias_slice.row_start = slice_start;
        bias_slice.row_count = slice_count;
        bias_slice.col_start = 0;
        bias_slice.col_count = 1;
        bias_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, sliced, WeightDerivationKind::RowSlice, bias_slice, device);

        return sliced;
    }

    std::shared_ptr<TensorBase> WeightManager::loadColumnParallel2DWeight(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        size_t total_rows = dimensions[0];
        size_t cols = dimensions[1];

        // Fused QKV weights need special handling: 3 concatenated sub-blocks
        // each split independently by heads
        if (has_sharding_config_ &&
            sharding_config_.getDimensionType(name) == WeightDimensionType::FusedQKVHeads)
        {
            return loadFusedQKVColumnParallel(name, device, assignment, dimensions);
        }

        size_t row_start = 0;
        size_t row_count = 0;

        // Use config-based dimension type to determine slicing
        if (!computeSliceBoundaries(name, total_rows, assignment, row_start, row_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for 2D column-parallel: " + name);
        }

        // Load only the slice from GGUF file (memory efficient, preserves quantization)
        auto slice_tensor = loader_.loadTensorRowSlice(
            name, row_start, row_start + row_count, device, WeightPrecision::NATIVE);

        if (!slice_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load row slice for: " + name);
        }

        // Wrap in TensorSlice with metadata
        auto meta = SliceMetadata::forColumnParallel(
            total_rows, cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        WeightSliceSpec column_slice;
        column_slice.source_rows = total_rows;
        column_slice.source_cols = cols;
        column_slice.row_start = row_start;
        column_slice.row_count = row_count;
        column_slice.col_start = 0;
        column_slice.col_count = cols;
        column_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::RowSlice, column_slice, device);

        LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config_->worldSize() << ")"
                                            << " column-parallel " << name
                                            << " [" << total_rows << ", " << cols << "]"
                                            << " -> rows [" << row_start << ", " << (row_start + row_count) << ")"
                                            << " = " << row_count << " rows");

        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadFusedQKVColumnParallel(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        const size_t total_rows = dimensions[0];
        const size_t cols = dimensions[1];
        const int world_size = tp_config_->worldSize();
        const int rank = assignment.local_rank;

        // Fused QKV weights have 3 concatenated sub-blocks: [Q | K | V]
        // Sub-blocks may be equal (standard FA) or asymmetric (GDN with n_v != n_k).
        // We load each sub-block's local rows directly from GGUF in native
        // quantized format, then concatenate the raw bytes into a single tensor.

        size_t sub_block_sizes[3] = {0, 0, 0};
        bool gdn_replicate_qk = false;

        // Try GDN asymmetric layout: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
        if (has_gdn_dimensions_ && gdn_n_k_heads_ > 0 && gdn_d_state_ > 0)
        {
            const size_t gdn_key_dim = static_cast<size_t>(gdn_n_k_heads_) * gdn_d_state_;
            const size_t gdn_value_dim = static_cast<size_t>(gdn_n_v_heads_) * gdn_d_state_;
            const size_t expected_gdn_qkv = 2 * gdn_key_dim + gdn_value_dim;

            if (total_rows == expected_gdn_qkv)
            {
                sub_block_sizes[0] = gdn_key_dim;   // Q
                sub_block_sizes[1] = gdn_key_dim;   // K
                sub_block_sizes[2] = gdn_value_dim; // V (may differ)

                // GDN modular repeat: v_head j uses k_head j%n_k.
                // Replicate Q and K on every rank, only shard V.
                gdn_replicate_qk = (gdn_n_v_heads_ > gdn_n_k_heads_);

                LOG_TRACE("[WeightManager] FusedQKV " << name
                                                      << " matched GDN layout: Q=" << gdn_key_dim
                                                      << " K=" << gdn_key_dim << " V=" << gdn_value_dim
                                                      << " (n_k=" << gdn_n_k_heads_
                                                      << " n_v=" << gdn_n_v_heads_
                                                      << " d=" << gdn_d_state_
                                                      << " replicate_qk=" << gdn_replicate_qk << ")");
            }
        }

        // Fall back to 3 equal sub-blocks (standard FA)
        if (sub_block_sizes[0] == 0)
        {
            if (total_rows % 3 != 0)
            {
                throw std::runtime_error("[WeightManager] FusedQKVHeads: total_rows " + std::to_string(total_rows) +
                                         " not divisible by 3 and no GDN layout match for weight: " + name);
            }
            const size_t equal_rows = total_rows / 3;
            sub_block_sizes[0] = equal_rows;
            sub_block_sizes[1] = equal_rows;
            sub_block_sizes[2] = equal_rows;
        }

        // Compute per-sub-block slice for this rank
        auto compute_slice = [rank, world_size, gdn_replicate_qk](
                                 size_t block_rows, int sub_block_idx) -> std::pair<size_t, size_t>
        {
            // Sub-blocks 0 (Q) and 1 (K): replicate for GDN, shard otherwise
            if (gdn_replicate_qk && sub_block_idx < 2)
                return {0, block_rows}; // Full sub-block (replicated)

            size_t rows_per_rank = block_rows / static_cast<size_t>(world_size);
            size_t start = rows_per_rank * static_cast<size_t>(rank);
            size_t count = (rank == world_size - 1)
                               ? (block_rows - start)
                               : rows_per_rank;
            return {start, count};
        };

        // Validate sub-block divisibility
        {
            static constexpr const char *sub_names[3] = {"Q", "K", "V"};
            for (size_t s = 0; s < 3; s++)
            {
                if (gdn_replicate_qk && s < 2)
                    continue; // Replicated sub-blocks are always valid
                if (sub_block_sizes[s] % static_cast<size_t>(world_size) != 0)
                {
                    throw std::runtime_error(std::string("[WeightManager] Cannot shard FusedQKV weight '") + name +
                                             "': sub-block " + sub_names[s] +
                                             " has " + std::to_string(sub_block_sizes[s]) +
                                             " rows, not divisible by TP degree " + std::to_string(world_size));
                }
            }
        }

        // Load each sub-block's slice from GGUF
        std::shared_ptr<TensorBase> slices[3];
        size_t total_out_rows = 0;
        size_t abs_offset = 0;

        for (size_t s = 0; s < 3; s++)
        {
            auto [local_start, local_count] = compute_slice(sub_block_sizes[s], static_cast<int>(s));
            const size_t abs_row_start = abs_offset + local_start;
            const size_t abs_row_end = abs_row_start + local_count;

            slices[s] = loader_.loadTensorRowSlice(
                name, abs_row_start, abs_row_end, device, WeightPrecision::NATIVE);

            if (!slices[s])
            {
                throw std::runtime_error("[WeightManager] Failed to load fused-QKV sub-block " +
                                         std::to_string(s) + " for: " + name);
            }

            total_out_rows += local_count;
            abs_offset += sub_block_sizes[s];
        }

        // Concatenate raw bytes from the 3 slices into a single native tensor
        size_t total_bytes = 0;
        for (size_t s = 0; s < 3; s++)
            total_bytes += slices[s]->size_bytes();

        std::vector<uint8_t> combined_raw(total_bytes);
        size_t byte_offset = 0;

        for (size_t s = 0; s < 3; s++)
        {
            const void *src = slices[s]->raw_data();
            if (!src)
            {
                throw std::runtime_error("[WeightManager] Null raw_data for fused-QKV sub-block " +
                                         std::to_string(s) + ": " + name);
            }
            std::memcpy(combined_raw.data() + byte_offset, src, slices[s]->size_bytes());
            byte_offset += slices[s]->size_bytes();
        }

        // Create a new tensor from the concatenated raw bytes
        std::vector<size_t> out_shape = {total_out_rows, cols};
        TensorType native_type = slices[0]->native_type();

        auto result_tensor = createTensorFromRawData(
            native_type, out_shape, std::move(combined_raw));

        if (!result_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to create fused-QKV tensor of type " +
                                     std::to_string(static_cast<int>(native_type)) + " for: " + name);
        }

        // Wrap in TensorSlice with column-parallel metadata so downstream TP
        // logic (allreduce detection, etc.) works correctly
        auto meta = SliceMetadata::forColumnParallel(
            total_rows, cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(
            std::shared_ptr<TensorBase>(std::move(result_tensor)), meta);

        WeightSliceSpec fused_slice;
        fused_slice.source_rows = total_rows;
        fused_slice.source_cols = cols;
        fused_slice.row_start = 0;
        fused_slice.row_count = total_out_rows;
        fused_slice.col_start = 0;
        fused_slice.col_count = cols;
        fused_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::FusedSubblockConcat, fused_slice, device);

        LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                            << " (rank " << rank << "/" << world_size << ")"
                                            << " fused-QKV column-parallel " << name
                                            << " [" << total_rows << ", " << cols << "]"
                                            << " -> [" << total_out_rows << ", " << cols << "]"
                                            << " (" << static_cast<int>(native_type) << " native)"
                                            << " sub-blocks [" << sub_block_sizes[0] << "," << sub_block_sizes[1] << "," << sub_block_sizes[2] << "]"
                                            << (gdn_replicate_qk ? " (GDN: Q/K replicated, V sharded)" : ""));

        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadRowParallelWeight(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        size_t total_rows = dimensions[0];
        size_t cols = dimensions[1];
        size_t row_start = 0;
        size_t row_count = 0;

        // Use config-based dimension type to determine row slicing
        if (!computeSliceBoundaries(name, total_rows, assignment, row_start, row_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for row-parallel: " + name);
        }

        auto slice_tensor = loader_.loadTensorRowSlice(
            name, row_start, row_start + row_count, device, WeightPrecision::NATIVE);

        if (!slice_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load row slice for row-parallel: " + name);
        }

        auto meta = SliceMetadata::forRowParallel(
            total_rows, cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        WeightSliceSpec row_slice;
        row_slice.source_rows = total_rows;
        row_slice.source_cols = cols;
        row_slice.row_start = row_start;
        row_slice.row_count = row_count;
        row_slice.col_start = 0;
        row_slice.col_count = cols;
        row_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::RowSlice, row_slice, device);

        LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config_->worldSize() << ")"
                                            << " row-parallel " << name
                                            << " [" << total_rows << ", " << cols << "]"
                                            << " -> rows [" << row_start << ", " << (row_start + row_count) << ")"
                                            << " = " << row_count << " rows (needs allreduce)");

        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadInputParallelWeight(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        size_t rows = dimensions[0];
        size_t total_cols = dimensions[1];

        // Use config-based dimension type to determine column slicing
        size_t col_start = 0;
        size_t col_count = 0;

        if (!computeSliceBoundaries(name, total_cols, assignment, col_start, col_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for input-parallel: " + name);
        }

        auto slice_tensor = loader_.loadTensorColumnSlice(
            name, col_start, col_start + col_count, device, WeightPrecision::NATIVE);

        if (!slice_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load column slice for input-parallel: " + name);
        }

        // Input-parallel uses row-parallel metadata (mathematically similar but with column slicing)
        auto meta = SliceMetadata::forRowParallel(
            rows, total_cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        WeightSliceSpec input_slice;
        input_slice.source_rows = rows;
        input_slice.source_cols = total_cols;
        input_slice.row_start = 0;
        input_slice.row_count = rows;
        input_slice.col_start = col_start;
        input_slice.col_count = col_count;
        input_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::ColumnSlice, input_slice, device);

        LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config_->worldSize() << ")"
                                            << " input-parallel " << name
                                            << " [" << rows << ", " << total_cols << "]"
                                            << " -> cols [" << col_start << ", " << (col_start + col_count) << ")"
                                            << " = " << col_count << " cols (needs allreduce)");

        return result;
    }

    // ========================================================================
    // Main sharding entry point
    // ========================================================================

    std::shared_ptr<TensorBase> WeightManager::getShardedWeightForAssignment(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        int layer_idx)
    {
        // Get sharding mode for this weight before consulting the per-device
        // cache. Older/preload paths may have populated the cache with a plain
        // device clone; for TP-sharded weights graph builders must see the
        // TensorSlice wrapper so row/column-parallel metadata is preserved.
        ShardingMode mode = getShardingMode(name);

        // Check per-device cache first.
        // WeightManager is rank-local, so device string uniquely identifies the
        // cache entry. LOCAL TP devices have distinct DeviceIds (e.g. cuda:0, cuda:1).
        std::string cache_key = device.to_string() + ":" + name;
        auto it = per_device_cache_.find(cache_key);
        if (it != per_device_cache_.end())
        {
            TensorSlice *slice = dynamic_cast<TensorSlice *>(it->second.get());
            const bool cache_matches_mode =
                (mode == ShardingMode::REPLICATE) ||
                (mode == ShardingMode::ROW_PARALLEL && slice && slice->is_row_parallel()) ||
                (mode == ShardingMode::INPUT_PARALLEL && slice && slice->is_row_parallel()) ||
                (mode == ShardingMode::COLUMN_PARALLEL && slice && slice->is_column_parallel()) ||
                (mode == ShardingMode::EXPERT_PARALLEL);

            if (cache_matches_mode)
            {
                return it->second;
            }

            LOG_DEBUG("[WeightManager] Ignoring stale per-device cache entry without TP slice metadata: "
                      << cache_key << " mode=" << static_cast<int>(mode)
                      << " tensor=" << it->second.get());
            per_device_cache_.erase(it);
        }

        LOG_TRACE("[WeightManager] getShardedWeightForAssignment: " << name
                                                                    << " device=" << device.to_string()
                                                                    << " local_rank=" << assignment.local_rank
                                                                    << " mode=" << static_cast<int>(mode));

        std::shared_ptr<TensorBase> result;

        switch (mode)
        {
        case ShardingMode::REPLICATE:
        {
            // For host-resident REPLICATE weights (e.g., token_embd.weight),
            // all devices share identical host data — no clone needed.
            // Non-host-resident REPLICATE weights (norms, biases) need per-device
            // clones because each device calls ensureOnDevice() which allocates
            // separate GPU memory and releaseAllHostWeightData() may free host data
            // after the first device uploads (breaking the second device's upload).
            auto cached_it = cache_.find(name);
            if (cached_it == cache_.end() || !cached_it->second)
            {
                auto original = getReplicatedWeight(name, DeviceId::cpu());
                if (original)
                {
                    cache_[name] = original;
                    cached_it = cache_.find(name);
                }
            }

            if (cached_it != cache_.end() && cached_it->second && cached_it->second->isHostResident())
            {
                result = cached_it->second;
                LOG_DEBUG("[WeightManager] Sharing host-resident REPLICATE weight: " << name
                                                                                     << " for " << device.to_string()
                                                                                     << " (" << (result->size_bytes() / (1024 * 1024)) << " MB)");
            }
            else if (cached_it != cache_.end() && cached_it->second)
            {
                result = cloneTensorForDevice(name, cached_it->second, device);
                if (result)
                    per_device_cache_[cache_key] = result;
            }
            if (result)
            {
                LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                    << " gets REPLICATED weight: " << name);
            }
            break;
        }

        case ShardingMode::COLUMN_PARALLEL:
        {
            // For tied embeddings: if output.weight is missing, substitute token_embd.weight
            auto dims_opt = loader_.getTensorShape(name);
            if ((!dims_opt || dims_opt->empty()) && name == "output.weight")
            {
                auto embd_dims = loader_.getTensorShape("token_embd.weight");
                if (embd_dims && embd_dims->size() == 2)
                {
                    LOG_INFO("[WeightManager] Device " << device.to_string()
                                                       << " output.weight not in GGUF — using tied embedding "
                                                       << "token_embd.weight as column-parallel LM head");

                    // Vocab-dimension slicing: use assignment's vocab_start/vocab_count
                    size_t total_rows = (*embd_dims)[0];
                    size_t cols = (*embd_dims)[1];
                    size_t row_start = assignment.vocab_start;
                    size_t row_count = assignment.vocab_count;

                    auto slice_tensor = loader_.loadTensorRowSlice(
                        "token_embd.weight", row_start, row_start + row_count, device, WeightPrecision::NATIVE);
                    if (!slice_tensor)
                    {
                        throw std::runtime_error("[WeightManager] Failed to load tied embedding row slice");
                    }

                    auto meta = SliceMetadata::forColumnParallel(
                        total_rows, cols, assignment.local_rank, tp_config_->worldSize(),
                        true /* inner_is_presliced */);
                    result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

                    WeightSliceSpec tied_slice;
                    tied_slice.source_rows = total_rows;
                    tied_slice.source_cols = cols;
                    tied_slice.row_start = row_start;
                    tied_slice.row_count = row_count;
                    tied_slice.col_start = 0;
                    tied_slice.col_count = cols;
                    tied_slice.inner_is_presliced = true;
                    registerDerivedMetadata(name, result, WeightDerivationKind::TiedAlias, tied_slice, device);

                    LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                                        << " tied embedding LM head"
                                                        << " [" << total_rows << ", " << cols << "]"
                                                        << " -> rows [" << row_start << ", " << (row_start + row_count) << ")"
                                                        << " = " << row_count << " rows");
                    break;
                }
            }
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for column-parallel: " << name);
                return nullptr;
            }
            const auto &dims = *dims_opt;

            if (dims.size() == 1)
            {
                result = loadColumnParallel1DBias(name, device, assignment, dims);
            }
            else if (dims.size() == 2)
            {
                result = loadColumnParallel2DWeight(name, device, assignment, dims);
            }
            else
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for column-parallel (expected 1D or 2D): " + name);
            }
            break;
        }

        case ShardingMode::ROW_PARALLEL:
        {
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for row-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for row-parallel (expected 2D): " + name);
            }
            result = loadRowParallelWeight(name, device, assignment, *dims_opt);
            break;
        }

        case ShardingMode::INPUT_PARALLEL:
        {
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for input-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for input-parallel (expected 2D): " + name);
            }
            result = loadInputParallelWeight(name, device, assignment, *dims_opt);
            break;
        }

        case ShardingMode::EXPERT_PARALLEL:
        {
            // Expert-parallel for LOCAL TP: split 3D expert tensors across local devices
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for expert-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 3)
            {
                throw std::runtime_error("[WeightManager] Expert-parallel requires 3D tensor, got " +
                                         std::to_string(dims_opt->size()) + "D for: " + name);
            }
            const auto &dims = *dims_opt;
            size_t ne2 = dims[2]; // num_experts
            int local_ws = tp_config_->worldSize();
            size_t experts_per_rank = ne2 / local_ws;
            size_t expert_start = experts_per_rank * assignment.local_rank;
            size_t expert_count = (assignment.local_rank == local_ws - 1)
                                      ? (ne2 - expert_start)
                                      : experts_per_rank;

            if (experts_per_rank == 0)
            {
                throw std::runtime_error("[WeightManager] Cannot shard " + std::to_string(ne2) +
                                         " experts across " + std::to_string(local_ws) +
                                         " devices for: " + name);
            }

            result = loader_.loadTensorExpertSlice(
                name, expert_start, expert_start + expert_count, device, WeightPrecision::NATIVE);

            if (!result)
            {
                throw std::runtime_error("[WeightManager] Failed to load expert slice for: " + name);
            }

            WeightSliceSpec expert_slice;
            expert_slice.source_rows = dims[0];
            expert_slice.source_cols = dims[1];
            expert_slice.row_start = 0;
            expert_slice.row_count = dims[0];
            expert_slice.col_start = 0;
            expert_slice.col_count = dims[1];
            expert_slice.expert_start = expert_start;
            expert_slice.expert_count = expert_count;
            expert_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, result, WeightDerivationKind::ExpertSlice, expert_slice, device);

            LOG_INFO("[WeightManager] Device " << device.to_string()
                                               << " expert-parallel " << name
                                               << " [" << dims[0] << ", " << dims[1] << ", " << ne2
                                               << "] -> experts [" << expert_start << ", "
                                               << (expert_start + expert_count) << ") = "
                                               << expert_count << "/" << ne2);
            break;
        }

        default:
            throw std::runtime_error("[WeightManager] Unknown sharding mode for: " + name);
        }

        // Cache the result for subsequent requests
        if (result)
        {
            per_device_cache_[cache_key] = result;
            result->setDebugName(name);
        }

        return result;
    }

    // =========================================================================
    // Layer Range Filtering for Pipeline Parallelism
    // =========================================================================

    void WeightManager::setHasEmbedding(bool has_embedding)
    {
        has_embedding_ = has_embedding;
        LOG_DEBUG("[WeightManager] has_embedding = " << (has_embedding ? "true" : "false"));
    }

    void WeightManager::setHasLmHead(bool has_lm_head)
    {
        has_lm_head_ = has_lm_head;
        LOG_DEBUG("[WeightManager] has_lm_head = " << (has_lm_head ? "true" : "false"));
    }

    bool WeightManager::isWeightInLayerRange(const std::string &name) const
    {
        // If no layer range is set, include all weights
        if (!has_layer_range_)
        {
            return true;
        }
        return isWeightInLayerRange(name, layer_first_, layer_last_, has_embedding_, has_lm_head_);
    }

    bool WeightManager::isWeightInLayerRange(const std::string &name,
                                             int first_layer, int last_layer,
                                             bool has_embedding, bool has_lm_head) const
    {
        // Handle special weights (embedding, output norm, LM head)
        if (name == "token_embd.weight")
        {
            if (has_embedding)
            {
                return true;
            }
            // Allow embedding through for the LM head stage ONLY if the model
            // uses tied embeddings (no separate output.weight in GGUF).
            if (has_lm_head)
            {
                auto output_shape = loader_.getTensorShape("output.weight");
                bool tied = !output_shape || output_shape->empty();
                return tied;
            }
            return false;
        }
        if (name == "output_norm.weight" || name == "output.weight")
        {
            return has_lm_head;
        }

        // Extract layer index from "blk.N.xxx" pattern
        // Pattern: blk.{layer_idx}.{component}.weight
        static const std::regex layer_pattern(R"(blk\.(\d+)\.)");
        std::smatch match;
        if (std::regex_search(name, match, layer_pattern))
        {
            int layer_idx = std::stoi(match[1].str());
            // Layer range is [first, last) - first inclusive, last exclusive
            return layer_idx >= first_layer && layer_idx < last_layer;
        }

        // Unknown weight pattern - include by default (e.g., custom weights)
        LOG_DEBUG("[WeightManager] Unknown weight pattern, including: " << name);
        return true;
    }

} // namespace llaminar2
