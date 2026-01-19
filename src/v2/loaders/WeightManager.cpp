/**
 * @file WeightManager.cpp
 * @brief Weight distribution and caching implementation
 * @author David Sanftenberg
 */

#include "WeightManager.h"
#include "../utils/Logger.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/TensorSlice.h"
#include "../kernels/KernelFactory.h"
#include <iostream>
#include <cstring>
#include <regex>

namespace llaminar2
{

    WeightManager::WeightManager(ModelLoader &loader,
                                 std::shared_ptr<MPIContext> mpi_ctx,
                                 std::shared_ptr<WeightPlacementMap> placement_map,
                                 WeightDistributionStrategy strategy,
                                 WeightPrecision weight_precision)
        : loader_(loader), mpi_ctx_(mpi_ctx), placement_map_(placement_map),
          strategy_(strategy), weight_precision_(weight_precision)
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

    std::shared_ptr<TensorBase> WeightManager::getWeight(const std::string &name, DeviceId device, int layer_idx)
    {
        // Check cache first
        auto it = cache_.find(name);
        if (it != cache_.end())
        {
            LOG_INFO("[WeightManager] Cache hit: " << name << " ptr=" << (void *)it->second.get());
            return it->second;
        }
        LOG_INFO("[WeightManager] Cache miss: " << name << " (loading now)");

        // Determine device from placement map if not explicitly provided
        DeviceId target_device = device;
        if (!target_device.is_valid() && placement_map_)
        {
            target_device = placement_map_->getDeviceForWeight(name, layer_idx);
        }
        if (!target_device.is_valid())
        {
            target_device = DeviceId::cpu(); // Default to CPU
        }

        // Load based on strategy
        std::shared_ptr<TensorBase> tensor;

        switch (strategy_)
        {
        case WeightDistributionStrategy::REPLICATED:
            tensor = getReplicatedWeight(name, target_device);
            break;

        case WeightDistributionStrategy::SHARDED:
            tensor = getShardedWeight(name, target_device);
            break;

        case WeightDistributionStrategy::INTERLEAVED:
            tensor = getInterleavedWeight(name, target_device);
            break;

        default:
            LOG_ERROR("[WeightManager] Unknown strategy: " << static_cast<int>(strategy_));
            return nullptr;
        }

        // NOTE: Weight packing is now handled by WeightPreloader, NOT here.
        // WeightPreloader.preloadAll() or preloadForDevice() should be called
        // after all weights are loaded to pack them for the target device.
        // This separation ensures:
        // 1. WeightManager only handles loading and distribution
        // 2. WeightPreloader handles device-specific packing
        // 3. Raw data isn't released until we know the target device

        // Cache the loaded tensor
        if (tensor)
        {
            cache_[name] = tensor;
            LOG_DEBUG("[WeightManager] Cached NEW tensor: " << name << " -> " << (void *)tensor.get());
        }

        return tensor;
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
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            LOG_ERROR("[WeightManager] Rank " << rank << " failed to load: " << name);
            return nullptr;
        }

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
            LOG_ERROR("[WeightManager] Column slicing requires 2D tensor, got " << shape.size() << "D");
            return nullptr;
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
            LOG_ERROR("[WeightManager] Column slicing currently requires FP32 tensor. "
                      "Use CONVERT_TO_FP32 weight precision for sharded strategy.");
            return nullptr;
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
            LOG_ERROR("[WeightManager] Row slicing requires 2D tensor, got " << shape.size() << "D");
            return nullptr;
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
            LOG_ERROR("[WeightManager] Row slicing currently requires FP32 tensor. "
                      "Use CONVERT_TO_FP32 weight precision for sharded strategy.");
            return nullptr;
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
            // Get tensor info to determine dimensions
            const GGUFTensorInfo *info = loader_.getModel().findTensor(name);
            if (!info || info->dimensions.size() != 2)
            {
                LOG_ERROR("[WeightManager] Rank " << rank << " invalid tensor for row-parallel: " << name);
                return nullptr;
            }

            size_t total_rows = info->dimensions[0];
            size_t cols = info->dimensions[1];

            // Calculate row range for this rank
            size_t rows_per_rank = total_rows / world_size;
            size_t row_start = rows_per_rank * rank;
            size_t row_end = (rank == world_size - 1) ? total_rows : row_start + rows_per_rank;

            // Load ONLY the slice from GGUF file (memory efficient!)
            auto slice_tensor = loader_.loadTensorRowSlice(
                name, row_start, row_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                LOG_ERROR("[WeightManager] Rank " << rank << " failed to load row slice for: " << name);
                return nullptr;
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " row-parallel " << name
                                              << " [" << total_rows << ", " << cols
                                              << "] -> loaded ONLY rows [" << row_start << ", " << row_end
                                              << ") = " << (row_end - row_start) << " rows (memory efficient)");

            // Wrap in TensorSlice with metadata (inner_is_presliced=true)
            // The inner tensor already contains only the slice data
            auto meta = SliceMetadata::forRowParallel(
                total_rows, cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
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

            // Get tensor dimensions via the model's findTensor
            const auto *info = loader_.getModel().findTensor(name);
            if (!info || info->dimensions.empty())
            {
                LOG_ERROR("[WeightManager] Cannot find tensor for column-parallel: " << name);
                return nullptr;
            }

            // Handle 1D tensors (biases)
            // For biases like Q/K/V biases [n_heads * head_dim], we load full then slice in memory.
            // This is acceptable since biases are tiny (e.g., 896 floats = 3.5KB).
            if (info->dimensions.size() == 1)
            {
                size_t total_elements = info->dimensions[0];
                size_t elements_per_rank = total_elements / world_size;
                size_t start = elements_per_rank * rank;
                size_t end = (rank == world_size - 1) ? total_elements : start + elements_per_rank;
                size_t slice_elements = end - start;

                // Load full 1D tensor first (biases are small, ~3KB)
                auto full_tensor = loader_.loadTensor(name, device, WeightPrecision::NATIVE);
                if (!full_tensor)
                {
                    LOG_ERROR("[WeightManager] Rank " << rank << " failed to load 1D tensor for: " << name);
                    return nullptr;
                }

                // Cast to FP32 (biases are stored as FP32)
                auto *fp32_full = dynamic_cast<FP32Tensor *>(full_tensor.get());
                if (!fp32_full)
                {
                    LOG_ERROR("[WeightManager] 1D tensor is not FP32 for: " << name);
                    return nullptr;
                }

                // Create sliced 1D tensor
                auto slice_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{slice_elements});
                std::memcpy(slice_tensor->mutable_data(),
                            fp32_full->data() + start,
                            slice_elements * sizeof(float));

                LOG_TRACE("[WeightManager] Rank " << rank << " column-parallel 1D " << name
                                                  << " [" << total_elements
                                                  << "] -> sliced elements [" << start << ", " << end
                                                  << ") = " << slice_elements << " elements");

                // For 1D tensors, use row-parallel metadata with cols=1
                auto meta = SliceMetadata::forRowParallel(
                    total_elements, 1, rank, world_size,
                    true /* inner_is_presliced */);

                auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
                return slice;
            }

            // Handle 2D tensors (weight matrices)
            if (info->dimensions.size() != 2)
            {
                LOG_ERROR("[WeightManager] Column-parallel requires 1D or 2D tensor, got "
                          << info->dimensions.size() << "D for: " << name);
                return nullptr;
            }

            size_t total_rows = info->dimensions[0]; // N = output features
            size_t cols = info->dimensions[1];       // K = input features
            size_t rows_per_rank = total_rows / world_size;

            // Calculate this rank's ROW range (splitting output dimension)
            size_t row_start = rows_per_rank * rank;
            size_t row_end = (rank == world_size - 1) ? total_rows : row_start + rows_per_rank;

            // Load only this rank's row slice (preserves quantized format)
            auto slice_tensor = loader_.loadTensorRowSlice(
                name, row_start, row_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                LOG_ERROR("[WeightManager] Rank " << rank << " failed to load row slice for: " << name);
                return nullptr;
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " column-parallel " << name
                                              << " [" << total_rows << ", " << cols
                                              << "] -> loaded ONLY rows [" << row_start << ", " << row_end
                                              << ") = " << (row_end - row_start) << " rows (memory efficient, quantized)");

            // Wrap in TensorSlice with metadata (inner_is_presliced=true)
            // Note: forColumnParallel stores original dimensions for allgather reconstruction
            auto meta = SliceMetadata::forColumnParallel(
                total_rows, cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
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

            // Get tensor dimensions via the model's findTensor
            const auto *info = loader_.getModel().findTensor(name);
            if (!info || info->dimensions.size() != 2)
            {
                LOG_ERROR("[WeightManager] Cannot find 2D tensor for input-parallel: " << name);
                return nullptr;
            }

            size_t rows = info->dimensions[0];       // N = output features (d_model)
            size_t total_cols = info->dimensions[1]; // K = input features (d_ff)
            size_t cols_per_rank = total_cols / world_size;

            // Calculate this rank's COLUMN range (splitting input dimension)
            size_t col_start = cols_per_rank * rank;
            size_t col_end = (rank == world_size - 1) ? total_cols : col_start + cols_per_rank;

            // Load only this rank's column slice (preserves quantized format)
            auto slice_tensor = loader_.loadTensorColumnSlice(
                name, col_start, col_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                LOG_ERROR("[WeightManager] Rank " << rank << " failed to load column slice for: " << name);
                return nullptr;
            }

            LOG_TRACE("[WeightManager] Rank " << rank << " input-parallel " << name
                                              << " [" << rows << ", " << total_cols
                                              << "] -> loaded ONLY cols [" << col_start << ", " << col_end
                                              << ") = " << (col_end - col_start) << " cols (memory efficient, quantized)");

            // Wrap in TensorSlice with row-parallel metadata
            // (input-parallel is mathematically like row-parallel but with column slicing)
            auto meta = SliceMetadata::forRowParallel(
                rows, total_cols, rank, world_size,
                true /* inner_is_presliced=true - slice already loaded */);

            auto slice = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);
            return slice;
        }

        LOG_ERROR("[WeightManager] Unknown sharding mode for: " << name);
        return nullptr;
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
            LOG_ERROR("[WeightManager] Invalid decode fraction: " << fraction
                                                                  << " (must be 0 < fraction <= 1)");
            return nullptr;
        }

        // Generate cache key (includes fraction to allow different shard sizes)
        std::string cache_key = name + "@decode_" + std::to_string(fraction);

        // Check decode cache first
        auto it = decode_cache_.find(cache_key);
        if (it != decode_cache_.end())
        {
            return it->second;
        }

        // Get the full weight tensor (from prefill cache or load fresh)
        // We need to load to CPU first for slicing, even if decode_device differs
        auto full_tensor = getWeight(name, DeviceId::cpu(), layer_idx);
        if (!full_tensor)
        {
            LOG_ERROR("[WeightManager] Failed to load full weight for decode shard: " << name);
            return nullptr;
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
                LOG_ERROR("[WeightManager] Failed to slice tail rows for decode: " << name);
                return nullptr;
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
                LOG_ERROR("[WeightManager] Failed to slice tail columns for decode: " << name);
                return nullptr;
            }
            LOG_DEBUG("[WeightManager] Decode shard for " << (mode == ShardingMode::ROW_PARALLEL ? "ROW_PARALLEL" : "INPUT_PARALLEL")
                                                          << " " << name
                                                          << ": tail " << (fraction * 100) << "% cols -> ["
                                                          << decode_shard->shape()[0] << ", "
                                                          << decode_shard->shape()[1] << "]");
            break;
        }

        default:
            LOG_ERROR("[WeightManager] Unknown sharding mode for decode weight: " << name);
            return nullptr;
        }

        // Cache the decode shard
        if (decode_shard)
        {
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
            LOG_ERROR("[WeightManager] Tail row slicing requires 2D tensor, got " << shape.size() << "D");
            return nullptr;
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
            LOG_ERROR("[WeightManager] Tail row slicing currently requires FP32 tensor. "
                      "Use CONVERT_TO_FP32 weight precision for decode shards.");
            return nullptr;
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
            LOG_ERROR("[WeightManager] Tail column slicing requires 2D tensor, got " << shape.size() << "D");
            return nullptr;
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
            LOG_ERROR("[WeightManager] Tail column slicing currently requires FP32 tensor. "
                      "Use CONVERT_TO_FP32 weight precision for decode shards.");
            return nullptr;
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

} // namespace llaminar2
