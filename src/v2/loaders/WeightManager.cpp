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

    WeightManager::WeightManager(IModelLoader &loader,
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
        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check cache first
        auto it = cache_.find(name);
        if (it != cache_.end())
        {
            LOG_TRACE("[WeightManager] Cache hit: " << name << " ptr=" << (void *)it->second.get());
            return it->second;
        }
        LOG_DEBUG("[WeightManager] Cache miss: " << name << " (loading now)");

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
            // Get tensor dimensions
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->size() != 2)
            {
                LOG_ERROR("[WeightManager] Rank " << rank << " invalid tensor for row-parallel: " << name);
                return nullptr;
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
                LOG_ERROR("[WeightManager] Rank " << rank << " failed to load row slice for: " << name);
                return nullptr;
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
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_ERROR("[WeightManager] Cannot find tensor for column-parallel: " << name);
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
                LOG_ERROR("[WeightManager] Column-parallel requires 1D or 2D tensor, got "
                          << dims.size() << "D for: " << name);
                return nullptr;
            }

            size_t total_rows = dims[0]; // N = output features
            size_t cols = dims[1];       // K = input features

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
                name, row_start, row_end, device, WeightPrecision::NATIVE);

            if (!slice_tensor)
            {
                LOG_ERROR("[WeightManager] Rank " << rank << " failed to load row slice for: " << name);
                return nullptr;
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
            if (!dims_opt || dims_opt->size() != 2)
            {
                LOG_ERROR("[WeightManager] Cannot find 2D tensor for input-parallel: " << name);
                return nullptr;
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
                LOG_ERROR("[WeightManager] Rank " << rank << " failed to load column slice for: " << name);
                return nullptr;
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
        // NOTE: We cannot call getWeight() here because we already hold cache_mutex_
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
                LOG_ERROR("[WeightManager] Invalid total_heads/total_kv_heads in TensorParallelConfig");
                return {0, total_rows};
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
            // Shouldn't reach here for column-parallel weights
            LOG_WARN("[WeightManager] calculateProportionalColumnSlice called for non-column weight: " << name);
            return {0, total_rows};
        }
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
                LOG_ERROR("[WeightManager] Invalid total_heads in TensorParallelConfig");
                return {0, total_cols};
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
            // Shouldn't reach here for row-parallel weights
            LOG_WARN("[WeightManager] calculateProportionalRowSlice called for non-row weight: " << name);
            return {0, total_cols};
        }
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
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
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

    namespace
    {
        /**
         * @brief Create a quantized tensor from raw bytes (generic for all tensor types)
         *
         * This mirrors TensorFactory::createQuantized() but doesn't require an instance.
         * Handles all 27+ quantized tensor types in a single function.
         *
         * @param type Tensor type enum
         * @param shape Tensor dimensions
         * @param raw_data Raw bytes (moved into tensor)
         * @return New tensor of the specified type, or nullptr on failure
         */
        std::unique_ptr<TensorBase> createQuantizedTensorFromRawData(
            TensorType type,
            const std::vector<size_t> &shape,
            std::vector<uint8_t> raw_data)
        {
            switch (type)
            {
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
                LOG_ERROR("[WeightManager] Unsupported tensor type for cloning: "
                          << static_cast<int>(type));
                return nullptr;
            }
        }
    } // anonymous namespace

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
            LOG_WARN("[WeightManager] Cannot clone tensor with no data: " << name);
            return nullptr;
        }

        // Copy raw bytes
        std::vector<uint8_t> raw_copy(byte_count);
        std::memcpy(raw_copy.data(), raw_ptr, byte_count);

        std::shared_ptr<TensorBase> clone;

        // Special case for FP32 (most common for norms/biases) - uses different constructor
        if (tensor_type == TensorType::FP32)
        {
            auto fp32_clone = std::make_shared<FP32Tensor>(original->shape());
            std::memcpy(fp32_clone->mutable_data(), raw_ptr, byte_count);
            clone = std::move(fp32_clone);
        }
        else
        {
            // All quantized types: use the common (shape, raw_data) constructor pattern
            auto unique_clone = createQuantizedTensorFromRawData(
                tensor_type, original->shape(), std::move(raw_copy));
            if (!unique_clone)
            {
                LOG_WARN("[WeightManager] Failed to create clone for tensor type "
                         << static_cast<int>(tensor_type));
                return nullptr;
            }
            clone = std::shared_ptr<TensorBase>(std::move(unique_clone));
        }

        clone->setDebugName(name + "@" + target_device.to_string());

        LOG_DEBUG("[WeightManager] Cloned tensor for " << target_device.to_string()
                                                       << ": " << name << " ["
                                                       << original->shape()[0] << "x"
                                                       << (original->shape().size() > 1 ? original->shape()[1] : 1) << "]"
                                                       << " type=" << static_cast<int>(tensor_type));

        return clone;
    }

    std::shared_ptr<TensorBase> WeightManager::getWeightForDevice(
        const std::string &name,
        DeviceId device,
        int layer_idx)
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        // =======================================================================
        // LOCAL TP support: Check TensorParallelConfig FIRST for device-specific slicing
        // =======================================================================
        // When tp_config_ is set and we're using SHARDED strategy, each device gets
        // a DIFFERENT slice of the weights based on its DeviceShardingAssignment.
        // This is different from the original code path which gave all devices the
        // same sharded weight (bug for LOCAL TP with world_size == 1).
        if (tp_config_ && strategy_ == WeightDistributionStrategy::SHARDED)
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
        auto ensureWeightLoaded = [this, &name, &device, layer_idx]() -> std::shared_ptr<TensorBase>
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
            return ensureWeightLoaded();
        }

        // Check per-device cache for subsequent devices
        std::string cache_key = device.to_string() + ":" + name;
        auto it = per_device_cache_.find(cache_key);
        if (it != per_device_cache_.end())
        {
            return it->second;
        }

        // Need to create a clone for this device
        // First, ensure the original is loaded
        auto original = ensureWeightLoaded();
        if (!original)
        {
            LOG_ERROR("[WeightManager] getWeightForDevice: failed to load weight: " << name);
            return nullptr;
        }

        auto clone = cloneTensorForDevice(name, original, device);
        if (!clone)
        {
            LOG_ERROR("[WeightManager] getWeightForDevice: failed to clone " << name);
            return nullptr;
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

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Set first device if not already set
        if (!first_device_.has_value())
        {
            first_device_ = devices[0];
            LOG_DEBUG("[WeightManager] First device set to: " << devices[0].to_string());
        }

        // Get all weight names from cache
        std::vector<std::string> weight_names;
        weight_names.reserve(cache_.size());
        for (const auto &[name, _] : cache_)
        {
            weight_names.push_back(name);
        }

        if (weight_names.empty())
        {
            LOG_WARN("[WeightManager] No weights in cache - call getWeight() first to load weights");
            return true;
        }

        size_t total_clones = 0;
        size_t total_uploads = 0;

        // For each device (except first), create clones and upload
        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx)
        {
            const DeviceId &device = devices[dev_idx];

            // First device uses original tensors, just upload them
            if (device == first_device_.value())
            {
                LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                                    << " is first device, uploading original tensors");
                for (const auto &name : weight_names)
                {
                    auto &tensor = cache_[name];
                    if (tensor && device.type != DeviceType::CPU)
                    {
                        if (tensor->ensureOnDevice(device))
                        {
                            ++total_uploads;
                        }
                    }
                }
                continue;
            }

            // Subsequent devices need clones
            LOG_DEBUG("[WeightManager] Creating clones for device " << device.to_string());
            for (const auto &name : weight_names)
            {
                std::string cache_key = device.to_string() + ":" + name;

                // Skip if already cloned
                if (per_device_cache_.find(cache_key) != per_device_cache_.end())
                {
                    continue;
                }

                auto &original = cache_[name];
                auto clone = cloneTensorForDevice(name, original, device);
                if (clone)
                {
                    // Upload to device
                    if (device.type != DeviceType::CPU)
                    {
                        if (clone->ensureOnDevice(device))
                        {
                            ++total_uploads;
                        }
                    }
                    per_device_cache_[cache_key] = clone;
                    ++total_clones;
                }
            }
        }

        LOG_INFO("[WeightManager] Pre-loaded " << total_clones << " clones, "
                                               << total_uploads << " uploads for "
                                               << devices.size() << " devices");
        return true;
    }

    // =========================================================================
    // Weight Packing and Preloading (folded from WeightPreloader)
    // =========================================================================

    bool WeightManager::packGemmWeights(
        DeviceId target_device,
        PreloadProgressCallback progress_cb,
        bool release_raw_data)
    {
        using namespace llaminar::v2::kernels;

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Collect all GEMM weight names
        std::vector<std::string> gemm_weights;
        for (const auto &[name, tensor] : cache_)
        {
            if (isGemmWeight(name))
            {
                gemm_weights.push_back(name);
            }
        }

        if (gemm_weights.empty())
        {
            LOG_DEBUG("[WeightManager] No GEMM weights to pack");
            return true;
        }

        LOG_DEBUG("[WeightManager] Packing " << gemm_weights.size() << " GEMM weights for "
                                             << target_device.to_string());

        size_t total = gemm_weights.size();
        size_t current = 0;
        bool all_success = true;

        // Set up device ordinal guards for GPU packing
        // Lambda that performs the actual packing
        auto do_packing = [&]() -> bool
        {
            for (const auto &name : gemm_weights)
            {
                current++;
                if (progress_cb)
                {
                    progress_cb(current, total, name);
                }

                auto &tensor = cache_[name];
                if (!tensor)
                {
                    LOG_WARN("[WeightManager] Weight not found in cache: " << name);
                    continue;
                }

                if (!packWeight(tensor.get(), target_device, release_raw_data))
                {
                    LOG_ERROR("[WeightManager] Failed to pack weight: " << name);
                    all_success = false;
                }
            }
            return all_success;
        };

        // Use ordinal guards for GPU devices
        if (target_device.is_rocm())
        {
            LOG_DEBUG("[WeightManager] Setting ROCm ordinal " << target_device.ordinal
                                                              << " for weight packing");
            KernelFactory::ROCmOrdinalGuard guard(target_device.ordinal);
            return do_packing();
        }
        else if (target_device.is_cuda())
        {
            LOG_DEBUG("[WeightManager] Setting CUDA ordinal " << target_device.ordinal
                                                              << " for weight packing");
            KernelFactory::CUDAOrdinalGuard guard(target_device.ordinal);
            return do_packing();
        }
        else
        {
            return do_packing();
        }
    }

    bool WeightManager::packWeight(
        TensorBase *tensor,
        DeviceId target_device,
        bool release_raw_data)
    {
        if (!tensor)
        {
            return false;
        }

        using namespace llaminar::v2::kernels;

        // Create kernel for target device (this creates the kernel with packed weights)
        auto *kernel = KernelFactory::getOrCreateGemm(tensor, target_device);
        if (!kernel)
        {
            LOG_ERROR("[WeightManager] Failed to create GEMM kernel for weight packing");
            return false;
        }

        // Call prepareWeights() to upload to GPU if needed
        // For CPU: this is a no-op
        // For GPU: this calls ensureWeightsConverted() which uploads INT8 data
        kernel->prepareWeights();

        if (target_device.is_cpu())
        {
            num_cpu_packed_++;

            // For CPU, we can release raw data since packed weights are in cache
            if (release_raw_data)
            {
                tensor->release_raw_data();
                LOG_TRACE("[WeightManager] Released raw data for: " << tensor->shape()[0]
                                                                    << "x" << tensor->shape()[1]);
            }
        }
        else
        {
            num_gpu_packed_++;
            // For GPU, do NOT release raw data!
            // The tensor coherence system (ensureOnDevice) still needs the host data
            // to upload to GPU buffers during stage execution.
            LOG_TRACE("[WeightManager] GPU weight packed (keeping raw data): "
                      << tensor->shape()[0] << "x" << tensor->shape()[1]);
        }

        return true;
    }

    bool WeightManager::uploadNonGemmWeights(DeviceId target_device)
    {
        if (!target_device.is_gpu())
        {
            LOG_DEBUG("[WeightManager] uploadNonGemmWeights skipped for CPU target");
            return true;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t uploaded_count = 0;

        LOG_DEBUG("[WeightManager] Uploading non-GEMM weights to " << target_device.to_string() << "...");

        for (const auto &[name, tensor] : cache_)
        {
            // Only process non-GEMM weights (norms, embeddings, biases)
            if (isGemmWeight(name))
            {
                continue;
            }

            if (!tensor)
            {
                LOG_WARN("[WeightManager] Non-GEMM weight is null: " << name);
                continue;
            }

            // Get the device-specific tensor if multi-GPU, otherwise use original
            TensorBase *device_tensor = nullptr;

            if (first_device_.has_value() && first_device_.value() == target_device)
            {
                // First device uses original tensor
                device_tensor = tensor.get();
            }
            else if (first_device_.has_value())
            {
                // Subsequent devices need clones - check per-device cache
                std::string cache_key = target_device.to_string() + ":" + name;
                auto it = per_device_cache_.find(cache_key);
                if (it != per_device_cache_.end())
                {
                    device_tensor = it->second.get();
                }
                else
                {
                    // Create clone for this device
                    auto clone = cloneTensorForDevice(name, tensor, target_device);
                    if (clone)
                    {
                        device_tensor = clone.get();
                        per_device_cache_[cache_key] = clone;
                    }
                }
            }
            else
            {
                // First device not set yet, set it now
                first_device_ = target_device;
                device_tensor = tensor.get();
            }

            if (!device_tensor)
            {
                LOG_WARN("[WeightManager] Failed to get device tensor for: " << name);
                continue;
            }

            // Set debug name so transfers can be traced
            device_tensor->setDebugName(name);

            // Upload to GPU using ensureOnDevice (no GEMM packing needed)
            if (device_tensor->ensureOnDevice(target_device))
            {
                size_t rows = device_tensor->shape()[0];
                size_t cols = device_tensor->shape().size() > 1 ? device_tensor->shape()[1] : 1;
                size_t bytes = rows * cols * sizeof(float);
                LOG_TRACE("[WeightManager] Uploaded non-GEMM weight: " << name
                                                                       << " [" << rows << "x" << cols << "] = " << bytes << " bytes");
                uploaded_count++;
            }
            else
            {
                LOG_WARN("[WeightManager] Failed to upload non-GEMM weight: " << name);
            }
        }

        LOG_INFO("[WeightManager] Uploaded " << uploaded_count << " non-GEMM weights to "
                                             << target_device.to_string());
        return true;
    }

    // =============================================================================
    // Device-aware weight slicing for LOCAL TP (Phase 1)
    // =============================================================================

    bool WeightManager::isQKVWeight(const std::string &name)
    {
        return name.find("attn_q.weight") != std::string::npos ||
               name.find("attn_k.weight") != std::string::npos ||
               name.find("attn_v.weight") != std::string::npos ||
               name.find("attn_qkv.weight") != std::string::npos;
    }

    bool WeightManager::isQKVBias(const std::string &name)
    {
        return name.find("attn_q.bias") != std::string::npos ||
               name.find("attn_k.bias") != std::string::npos ||
               name.find("attn_v.bias") != std::string::npos;
    }

    bool WeightManager::isFFNGateUpWeight(const std::string &name)
    {
        return name.find("ffn_gate.weight") != std::string::npos ||
               name.find("ffn_up.weight") != std::string::npos ||
               name.find("ffn_gate_up.weight") != std::string::npos;
    }

    bool WeightManager::isFFNDownWeight(const std::string &name)
    {
        return name.find("ffn_down.weight") != std::string::npos;
    }

    bool WeightManager::isLMHeadWeight(const std::string &name)
    {
        return name == "output.weight";
    }

    bool WeightManager::isWoWeight(const std::string &name)
    {
        return name.find("attn_output.weight") != std::string::npos;
    }

    bool WeightManager::isEmbeddingWeight(const std::string &name)
    {
        return name == "token_embd.weight";
    }

    bool WeightManager::isOutputNormWeight(const std::string &name)
    {
        return name == "output_norm.weight";
    }

    std::shared_ptr<TensorBase> WeightManager::sliceRowRange(
        const std::shared_ptr<TensorBase> &tensor,
        size_t row_start,
        size_t row_count)
    {
        if (!tensor)
        {
            LOG_ERROR("[WeightManager] sliceRowRange: null tensor");
            return nullptr;
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            LOG_ERROR("[WeightManager] sliceRowRange requires 2D tensor, got " << shape.size() << "D");
            return nullptr;
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];

        if (row_start + row_count > out_dim)
        {
            LOG_ERROR("[WeightManager] sliceRowRange: row_start=" << row_start
                                                                  << " + row_count=" << row_count << " > out_dim=" << out_dim);
            return nullptr;
        }

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32_tensor)
        {
            LOG_ERROR("[WeightManager] sliceRowRange currently requires FP32 tensor. "
                      "For quantized weights, use GGUF loadTensorRowSlice instead.");
            return nullptr;
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
            LOG_ERROR("[WeightManager] sliceColumnRange: null tensor");
            return nullptr;
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            LOG_ERROR("[WeightManager] sliceColumnRange requires 2D tensor, got " << shape.size() << "D");
            return nullptr;
        }

        size_t rows = shape[0];
        size_t cols = shape[1];

        if (col_start + col_count > cols)
        {
            LOG_ERROR("[WeightManager] sliceColumnRange: col_start=" << col_start
                                                                     << " + col_count=" << col_count << " > cols=" << cols);
            return nullptr;
        }

        if (col_count == 0)
        {
            LOG_ERROR("[WeightManager] sliceColumnRange: col_count cannot be 0");
            return nullptr;
        }

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32_tensor)
        {
            LOG_ERROR("[WeightManager] sliceColumnRange currently requires FP32 tensor. "
                      "For quantized weights, use GGUF loadTensorColumnSlice instead.");
            return nullptr;
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

        // For QKV biases, use head-based slicing (same as corresponding weight)
        if (!isQKVBias(name))
        {
            LOG_ERROR("[WeightManager] Unknown 1D column-parallel tensor: " << name);
            return nullptr;
        }

        const bool is_kv_bias = (name.find("attn_k.") != std::string::npos ||
                                 name.find("attn_v.") != std::string::npos);

        const int total_heads_for_bias = is_kv_bias
                                             ? tp_config_->totalKVHeads()
                                             : tp_config_->totalHeads();
        if (total_heads_for_bias <= 0)
        {
            LOG_ERROR("[WeightManager] Invalid total_heads/total_kv_heads for bias slicing");
            return nullptr;
        }
        const size_t head_dim = total_size / static_cast<size_t>(total_heads_for_bias);

        if (is_kv_bias)
        {
            slice_start = assignment.kv_head_start * head_dim;
            slice_count = assignment.kv_head_count * head_dim;
        }
        else
        {
            slice_start = assignment.head_start * head_dim;
            slice_count = assignment.head_count * head_dim;
        }

        // Load full tensor and slice in memory (biases are small)
        auto full_tensor = loader_.loadTensor(name, device, WeightPrecision::NATIVE);
        if (!full_tensor)
        {
            LOG_ERROR("[WeightManager] Failed to load 1D tensor: " << name);
            return nullptr;
        }

        // Create sliced tensor
        auto *fp32_full = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_full)
        {
            LOG_ERROR("[WeightManager] 1D bias must be FP32: " << name);
            return nullptr;
        }

        std::vector<size_t> slice_shape = {slice_count};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);
        std::memcpy(sliced->mutable_data(),
                    fp32_full->data() + slice_start,
                    slice_count * sizeof(float));

        LOG_INFO("[WeightManager] Device " << device.to_string()
                                           << " (rank " << assignment.local_rank << "/" << tp_config_->worldSize() << ")"
                                           << " column-parallel 1D " << name
                                           << " [" << total_size << "]"
                                           << " -> [" << slice_start << ", " << (slice_start + slice_count) << ")"
                                           << " = " << slice_count << " elements");

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
        size_t row_start = 0;
        size_t row_count = 0;

        // Determine row slice based on weight type
        if (isQKVWeight(name))
        {
            const bool is_kv_weight = (name.find("attn_k.") != std::string::npos ||
                                       name.find("attn_v.") != std::string::npos);

            // Q/K/V: slice by head ranges
            // CRITICAL: K/V weights use totalKVHeads(), not totalHeads()!
            const int total_heads_for_weight = is_kv_weight
                                                   ? tp_config_->totalKVHeads()
                                                   : tp_config_->totalHeads();
            if (total_heads_for_weight <= 0)
            {
                LOG_ERROR("[WeightManager] Invalid total_heads/total_kv_heads in TensorParallelConfig");
                return nullptr;
            }
            const size_t head_dim = total_rows / static_cast<size_t>(total_heads_for_weight);

            if (is_kv_weight)
            {
                row_start = assignment.kv_head_start * head_dim;
                row_count = assignment.kv_head_count * head_dim;
                LOG_TRACE("[WeightManager] KV weight " << name
                                                       << " total_kv_heads=" << total_heads_for_weight
                                                       << " head_dim=" << head_dim
                                                       << " kv_heads=[" << assignment.kv_head_start << ", "
                                                       << (assignment.kv_head_start + assignment.kv_head_count) << ")"
                                                       << " -> rows [" << row_start << ", " << (row_start + row_count) << ")");
            }
            else
            {
                row_start = assignment.head_start * head_dim;
                row_count = assignment.head_count * head_dim;
                LOG_TRACE("[WeightManager] Q weight " << name
                                                      << " total_heads=" << total_heads_for_weight
                                                      << " head_dim=" << head_dim
                                                      << " heads=[" << assignment.head_start << ", "
                                                      << (assignment.head_start + assignment.head_count) << ")"
                                                      << " -> rows [" << row_start << ", " << (row_start + row_count) << ")");
            }
        }
        else if (isFFNGateUpWeight(name))
        {
            row_start = assignment.d_ff_start;
            row_count = assignment.d_ff_count;
            LOG_TRACE("[WeightManager] FFN Gate/Up weight " << name
                                                            << " d_ff=[" << row_start << ", " << (row_start + row_count) << ")");
        }
        else if (isLMHeadWeight(name))
        {
            row_start = assignment.vocab_start;
            row_count = assignment.vocab_count;
            LOG_TRACE("[WeightManager] LM head weight " << name
                                                        << " vocab=[" << row_start << ", " << (row_start + row_count) << ")");
        }
        else
        {
            LOG_ERROR("[WeightManager] Unknown column-parallel weight type: " << name);
            return nullptr;
        }

        // Load only the slice from GGUF file (memory efficient, preserves quantization)
        auto slice_tensor = loader_.loadTensorRowSlice(
            name, row_start, row_start + row_count, device, WeightPrecision::NATIVE);

        if (!slice_tensor)
        {
            LOG_ERROR("[WeightManager] Failed to load row slice for: " << name);
            return nullptr;
        }

        // Wrap in TensorSlice with metadata
        auto meta = SliceMetadata::forColumnParallel(
            total_rows, cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        LOG_INFO("[WeightManager] Device " << device.to_string()
                                           << " (rank " << assignment.local_rank << "/" << tp_config_->worldSize() << ")"
                                           << " column-parallel " << name
                                           << " [" << total_rows << ", " << cols << "]"
                                           << " -> rows [" << row_start << ", " << (row_start + row_count) << ")"
                                           << " = " << row_count << " rows");

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

        // For Wo, split rows based on head assignment
        int total_heads = tp_config_->totalHeads();
        if (total_heads <= 0)
        {
            LOG_ERROR("[WeightManager] Invalid total_heads for row-parallel: " << name);
            return nullptr;
        }
        size_t head_dim = total_rows / total_heads;

        size_t row_start = assignment.head_start * head_dim;
        size_t row_count = assignment.head_count * head_dim;

        LOG_TRACE("[WeightManager] Wo weight " << name
                                               << " heads=[" << assignment.head_start << ", "
                                               << (assignment.head_start + assignment.head_count) << ")"
                                               << " -> rows [" << row_start << ", " << (row_start + row_count) << ")");

        auto slice_tensor = loader_.loadTensorRowSlice(
            name, row_start, row_start + row_count, device, WeightPrecision::NATIVE);

        if (!slice_tensor)
        {
            LOG_ERROR("[WeightManager] Failed to load row slice for row-parallel: " << name);
            return nullptr;
        }

        auto meta = SliceMetadata::forRowParallel(
            total_rows, cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        LOG_INFO("[WeightManager] Device " << device.to_string()
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

        // Determine column slice based on weight type
        size_t col_start = 0;
        size_t col_count = 0;

        if (isWoWeight(name))
        {
            // Wo (attn_output): slice by heads to match column-parallel Q/K/V output
            const int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
            {
                LOG_ERROR("[WeightManager] Invalid total_heads for input-parallel Wo: " << name);
                return nullptr;
            }
            const size_t head_dim = total_cols / static_cast<size_t>(total_heads);
            col_start = assignment.head_start * head_dim;
            col_count = assignment.head_count * head_dim;

            LOG_TRACE("[WeightManager] Wo weight " << name
                                                   << " heads=[" << assignment.head_start << ", "
                                                   << (assignment.head_start + assignment.head_count) << ")"
                                                   << " cols=[" << col_start << ", " << (col_start + col_count) << ")");
        }
        else
        {
            // FFN Down: slice by d_ff to match column-parallel Gate/Up output
            col_start = assignment.d_ff_start;
            col_count = assignment.d_ff_count;

            LOG_TRACE("[WeightManager] FFN Down weight " << name
                                                         << " d_ff=[" << col_start << ", " << (col_start + col_count) << ")");
        }

        auto slice_tensor = loader_.loadTensorColumnSlice(
            name, col_start, col_start + col_count, device, WeightPrecision::NATIVE);

        if (!slice_tensor)
        {
            LOG_ERROR("[WeightManager] Failed to load column slice for input-parallel: " << name);
            return nullptr;
        }

        // Input-parallel uses row-parallel metadata (mathematically similar but with column slicing)
        auto meta = SliceMetadata::forRowParallel(
            rows, total_cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        LOG_INFO("[WeightManager] Device " << device.to_string()
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
        // Check per-device cache first
        std::string cache_key = device.to_string() + ":" + name;
        auto it = per_device_cache_.find(cache_key);
        if (it != per_device_cache_.end())
        {
            return it->second;
        }

        // Get sharding mode for this weight
        ShardingMode mode = getShardingMode(name);

        LOG_TRACE("[WeightManager] getShardedWeightForAssignment: " << name
                                                                    << " device=" << device.to_string()
                                                                    << " local_rank=" << assignment.local_rank
                                                                    << " mode=" << static_cast<int>(mode));

        std::shared_ptr<TensorBase> result;

        switch (mode)
        {
        case ShardingMode::REPLICATE:
            result = getReplicatedWeight(name, device);
            if (result)
            {
                LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                    << " gets REPLICATED weight: " << name);
            }
            break;

        case ShardingMode::COLUMN_PARALLEL:
        {
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt)
            {
                LOG_ERROR("[WeightManager] Tensor not found for column-parallel: " << name);
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
                LOG_ERROR("[WeightManager] Invalid tensor for column-parallel (expected 1D or 2D): " << name);
                return nullptr;
            }
            break;
        }

        case ShardingMode::ROW_PARALLEL:
        {
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->size() != 2)
            {
                LOG_ERROR("[WeightManager] Invalid tensor for row-parallel: " << name);
                return nullptr;
            }
            result = loadRowParallelWeight(name, device, assignment, *dims_opt);
            break;
        }

        case ShardingMode::INPUT_PARALLEL:
        {
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->size() != 2)
            {
                LOG_ERROR("[WeightManager] Invalid tensor for input-parallel: " << name);
                return nullptr;
            }
            result = loadInputParallelWeight(name, device, assignment, *dims_opt);
            break;
        }

        default:
            LOG_ERROR("[WeightManager] Unknown sharding mode for: " << name);
            return nullptr;
        }

        // Cache the result for subsequent requests
        if (result)
        {
            per_device_cache_[cache_key] = result;
            result->setDebugName(name);
        }

        return result;
    }

} // namespace llaminar2
