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
                LOG_DEBUG("[WeightManager] Row-parallel: Wo (split output dim, allreduce after)");
                LOG_DEBUG("[WeightManager] Replicated: QKV, norms, biases, embeddings, LM head");
            }
        }
    }

    std::shared_ptr<TensorBase> WeightManager::getWeight(const std::string &name, int device_idx, int layer_idx)
    {
        // Check cache first
        auto it = cache_.find(name);
        if (it != cache_.end())
        {
            return it->second;
        }

        // Determine device from placement map if not explicitly provided
        int target_device = device_idx;
        if (target_device < 0 && placement_map_)
        {
            target_device = placement_map_->getDeviceForWeight(name, layer_idx);
        }
        if (target_device < 0)
        {
            target_device = 0; // Default to device 0
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

        // For GEMM weight matrices, immediately pack to INT8 and release raw data
        // This prevents peak memory from growing as we load all weights
        if (tensor && isGemmWeight(name))
        {
            // Pack weights into INT8 VNNI format (creates cached kernel via KernelFactory)
            auto *gemm_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(tensor.get());
            if (gemm_kernel)
            {
                // Release original quantized data - kernel has packed copy
                tensor->release_raw_data();
            }
        }

        // Cache the loaded tensor
        if (tensor)
        {
            cache_[name] = tensor;
        }

        return tensor;
    }

    bool WeightManager::isGemmWeight(const std::string &name)
    {
        // GEMM weights are 2D matrices used for matmul operations
        // These should have their raw data released after GEMM packing
        //
        // GEMM weights:
        //   - attn_q.weight, attn_k.weight, attn_v.weight, attn_output.weight
        //   - ffn_gate.weight, ffn_up.weight, ffn_down.weight
        //   - output.weight (LM head)
        //
        // NOT GEMM weights (keep raw data):
        //   - *_norm.weight (RMSNorm, 1D)
        //   - token_embd.weight (embeddings, used directly)
        //   - *.bias (1D biases)

        // Exclude norms (1D tensors)
        if (name.find("_norm.weight") != std::string::npos)
        {
            return false;
        }

        // Exclude biases (1D tensors)
        if (name.find(".bias") != std::string::npos)
        {
            return false;
        }

        // Exclude embeddings (used directly, not via GEMM)
        if (name.find("token_embd") != std::string::npos)
        {
            return false;
        }

        // All other .weight tensors are GEMM weights
        return name.find(".weight") != std::string::npos;
    }

    std::shared_ptr<TensorBase> WeightManager::getReplicatedWeight(const std::string &name, int device_idx)
    {
        // Phase 1: Simple replication - each rank loads independently
        // No MPI coordination needed

        auto tensor = loader_.loadTensor(name, device_idx, weight_precision_);
        if (!tensor)
        {
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            LOG_ERROR("[WeightManager] Rank " << rank << " failed to load: " << name);
            return nullptr;
        }

        return tensor;
    }

    ShardingMode WeightManager::determineShardingMode(const std::string &name)
    {
        // =========================================================================
        // Weight Sharding Strategy for Tensor Parallelism
        // =========================================================================
        //
        // FFN Tensor Parallelism Pattern (Phase 4b-1):
        // =============================================
        // 1. Gate/Up are COLUMN-PARALLEL (split output d_ff dimension)
        //    - Weight: [d_ff, d_model] → [d_ff_local, d_model] per rank
        //    - Output: [seq, d_ff_local] (each rank has different neurons)
        //    - No allreduce needed
        //
        // 2. Down is COLUMN-PARALLEL (split input d_ff dimension to match Gate/Up output)
        //    - Weight: [d_model, d_ff] → [d_model, d_ff_local] per rank
        //    - Input: [seq, d_ff_local] (matches Gate/Up output)
        //    - Output: [seq, d_model] partial sum
        //    - Allreduce needed to sum partial results
        //
        // Attention Tensor Parallelism (Wo only for Phase 4b-1):
        // ======================================================
        // Wo is ROW-PARALLEL (split output d_model dimension, allreduce after)
        // Q/K/V are replicated (attention handles head splitting internally)
        //
        // Pattern matching for GGUF weight names:
        // - Row-parallel: attn_output.weight (Wo)
        // - Column-parallel: ffn_gate, ffn_up (split output d_ff dimension)
        // - Input-parallel: ffn_down (split input d_ff dimension to match Gate/Up output)
        // - Replicated: attn_q, attn_k, attn_v, norms, biases, embeddings, lm_head
        // =========================================================================

        // Row-parallel weights: Split output dimension, requires allgather
        // Only Wo uses row-parallel in Phase 4b-1
        if (name.find("attn_output.weight") != std::string::npos)
        {
            return ShardingMode::ROW_PARALLEL;
        }

        // Column-parallel weights: FFN Gate/Up (Phase 4b-1)
        // Gate/Up split output dimension (d_ff rows of weight matrix)
        if (name.find("ffn_gate.weight") != std::string::npos ||
            name.find("ffn_up.weight") != std::string::npos)
        {
            return ShardingMode::COLUMN_PARALLEL;
        }

        // Input-parallel weights: FFN Down (Phase 4b-1)
        // Down splits input dimension (d_ff columns of weight matrix) to match Gate/Up output
        if (name.find("ffn_down.weight") != std::string::npos)
        {
            return ShardingMode::INPUT_PARALLEL;
        }

        // TODO: Enable when attention infrastructure supports local-only Q/K/V
        // if (name.find("attn_q.weight") != std::string::npos ||
        //     name.find("attn_k.weight") != std::string::npos ||
        //     name.find("attn_v.weight") != std::string::npos)
        // {
        //     return ShardingMode::COLUMN_PARALLEL;
        // }

        // Everything else is replicated (including Q/K/V for now)
        return ShardingMode::REPLICATE;
    }

    ShardingMode WeightManager::getShardingMode(const std::string &name) const
    {
        // Check cache first
        auto it = sharding_mode_cache_.find(name);
        if (it != sharding_mode_cache_.end())
        {
            return it->second;
        }

        // Determine and cache
        ShardingMode mode = determineShardingMode(name);
        const_cast<WeightManager *>(this)->sharding_mode_cache_[name] = mode;
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

    std::shared_ptr<TensorBase> WeightManager::getShardedWeight(const std::string &name, int device_idx)
    {
        // For SHARDED strategy:
        // 1. Determine sharding mode based on weight name
        // 2. Load full tensor in native format (quantized)
        // 3. Wrap in TensorSlice with sharding metadata
        // 4. Slicing happens during kernel creation (preserves quantization)

        if (!mpi_ctx_ || mpi_ctx_->world_size() == 1)
        {
            // Single rank: no sharding needed
            return getReplicatedWeight(name, device_idx);
        }

        ShardingMode mode = getShardingMode(name);
        int rank = mpi_ctx_->rank();
        int world_size = mpi_ctx_->world_size();

        if (mode == ShardingMode::REPLICATE)
        {
            // This weight should not be sharded (norms, biases, etc.)
            return getReplicatedWeight(name, device_idx);
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
                name, row_start, row_end, device_idx, WeightPrecision::NATIVE);

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

            int rank = mpi_ctx_->rank();
            int world_size = mpi_ctx_->world_size();

            // Get tensor dimensions via the model's findTensor
            const auto *info = loader_.getModel().findTensor(name);
            if (!info || info->dimensions.size() != 2)
            {
                LOG_ERROR("[WeightManager] Cannot find 2D tensor for column-parallel: " << name);
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
                name, row_start, row_end, device_idx, WeightPrecision::NATIVE);

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
                name, col_start, col_end, device_idx, WeightPrecision::NATIVE);

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

    std::shared_ptr<TensorBase> WeightManager::getInterleavedWeight(const std::string &name, int device_idx)
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
        return getReplicatedWeight(name, device_idx);
    }

} // namespace llaminar2
