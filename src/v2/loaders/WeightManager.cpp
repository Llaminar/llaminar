/**
 * @file WeightManager.cpp
 * @brief Weight distribution and caching implementation
 * @author David Sanftenberg
 */

#include "WeightManager.h"
#include "../utils/Logger.h"
#include "../utils/WeightLoadingProfiler.h"
#include "../utils/DebugEnv.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/TensorSlice.h"
#include "../kernels/KernelFactory.h"
#include "../backends/BackendManager.h"
#include <iostream>
#include <cstring>
#include <regex>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <thread>
#include <future>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

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

        auto base_it = cache_.find(name);
        if (base_it != cache_.end() && base_it->second)
        {
            TensorBase *ptr = base_it->second.get();
            if (released.insert(ptr).second)
            {
                ptr->release_host_weight_data();
                found_any = true;
            }
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
                TensorBase *ptr = tensor.get();
                if (released.insert(ptr).second)
                {
                    ptr->release_host_weight_data();
                    found_any = true;
                }
            }
        }

        if (!found_any)
        {
            LOG_WARN("[WeightManager][Phase2] Reclaim requested but no tensors found for: " << name);
        }

        return found_any;
    }

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
                LOG_ERROR("[WeightManager] Unsupported tensor type for createTensorFromRawData: "
                          << static_cast<int>(type));
                return nullptr;
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
                LOG_ERROR("[WeightManager] Rank " << rank << " invalid tensor for row-parallel (expected 2D): " << name);
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

                    // Compute per-rank slice within each sub-block
                    // Q sliced by n_heads, K and V sliced by n_kv_heads
                    auto compute_slice = [rank, world_size](size_t block_rows) -> std::pair<size_t, size_t>
                    {
                        size_t rows_per_rank = block_rows / world_size;
                        size_t start = rows_per_rank * rank;
                        size_t count = (rank == world_size - 1)
                                           ? (block_rows - start)
                                           : rows_per_rank;
                        return {start, count};
                    };

                    // Load each sub-block's slice from GGUF
                    std::shared_ptr<TensorBase> slices[3];
                    size_t total_out_rows = 0;
                    size_t abs_offset = 0; // Running offset into the weight

                    for (size_t s = 0; s < 3; s++)
                    {
                        auto [local_start, local_count] = compute_slice(sub_block_sizes[s]);
                        const size_t abs_row_start = abs_offset + local_start;
                        const size_t abs_row_end = abs_row_start + local_count;

                        slices[s] = loader_.loadTensorRowSlice(
                            load_name, abs_row_start, abs_row_end, device, WeightPrecision::NATIVE);

                        if (!slices[s])
                        {
                            LOG_ERROR("[WeightManager] Rank " << rank
                                                              << " failed to load fused-QKV sub-block " << s
                                                              << " rows [" << abs_row_start << ", " << abs_row_end << ")"
                                                              << " for: " << name);
                            return nullptr;
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
                            LOG_ERROR("[WeightManager] Null raw_data for fused-QKV sub-block " << s << ": " << name);
                            return nullptr;
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
                        LOG_ERROR("[WeightManager] Failed to create fused-QKV tensor for: " << name);
                        return nullptr;
                    }

                    auto meta = SliceMetadata::forColumnParallel(
                        total_rows, cols, rank, world_size,
                        true /* inner_is_presliced */);

                    auto result = std::make_shared<TensorSlice>(
                        std::shared_ptr<TensorBase>(std::move(result_tensor)), meta);

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
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for input-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                LOG_ERROR("[WeightManager] Invalid tensor for input-parallel (expected 2D): " << name);
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
            LOG_WARN("[WeightManager] Cannot clone tensor with no data: " << name);
            return nullptr;
        }

        // Copy raw bytes
        std::vector<uint8_t> raw_copy(byte_count);
        std::memcpy(raw_copy.data(), raw_ptr, byte_count);

        auto unique_clone = createTensorFromRawData(
            tensor_type, original->shape(), std::move(raw_copy));
        if (!unique_clone)
        {
            LOG_WARN("[WeightManager] Failed to create clone for tensor type "
                     << static_cast<int>(tensor_type));
            return nullptr;
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
                LOG_ERROR("[WeightManager] Unknown strategy: " << static_cast<int>(strategy_));
                lock.lock();
                return nullptr;
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
    // Weight Packing and Preloading (folded from WeightPreloader)
    // =========================================================================

    bool WeightManager::packGemmWeights(
        DeviceId target_device,
        PreloadProgressCallback progress_cb,
        bool release_raw_data)
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
        std::vector<std::pair<std::string, std::shared_ptr<TensorBase>>> gemm_weights;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            gemm_weights.reserve(cache_.size());
            for (const auto &[name, tensor] : cache_)
            {
                if (isGemmWeight(name) && tensor)
                {
                    gemm_weights.emplace_back(name, tensor);
                }
            }
        }

        if (gemm_weights.empty())
        {
            LOG_DEBUG("[WeightManager] No GEMM weights to pack");
            return true;
        }

        LOG_DEBUG("[WeightManager] Packing " << gemm_weights.size() << " GEMM weights for "
                                             << target_device.to_string());

        const size_t total = gemm_weights.size();
        const unsigned hardware_threads = std::max(1u, std::thread::hardware_concurrency());

        unsigned prepare_workers = target_device.is_gpu()
                                       ? std::min<unsigned>(8u, hardware_threads)
                                       : 1u;
        unsigned upload_workers = target_device.is_gpu()
                                      ? std::min<unsigned>(2u, hardware_threads)
                                      : 1u;
        size_t queue_capacity = std::max<size_t>(4, static_cast<size_t>(upload_workers) * 4);

        // Phase 4 Step 2 pilot (control-plane only): historical worker/slot shaping
        // for ROCm startup CK row-major repack pipeline.
        //
        // Architectural direction is now VNNI-only startup preparation. CK row-major
        // startup repack is disabled, so we intentionally keep default producer/
        // consumer topology to avoid injecting control-plane overhead that no longer
        // has corresponding GPU overlap benefits.
        if (target_device.is_rocm() && debugEnv().rocm.startup_gpu_repack)
        {
            const auto &rocm_cfg = debugEnv().rocm;

            const size_t requested_budget_mb = static_cast<size_t>(std::max(128, rocm_cfg.repack_budget_mb));
            size_t effective_budget_mb = requested_budget_mb;
            if (auto *rocm_backend = getROCmBackend())
            {
                const size_t free_mb = rocm_backend->deviceMemoryFree(target_device.ordinal) / (1024ull * 1024ull);
                if (free_mb > 512)
                {
                    effective_budget_mb = std::min(effective_budget_mb, free_mb - 512);
                }
            }

            LOG_INFO("[WeightManager][Phase4Pilot] ROCm startup GPU repack control enabled: "
                     << "prepare_workers=" << prepare_workers
                     << " upload_workers=" << upload_workers
                     << " queue_slots=" << queue_capacity
                     << " budget_mb=" << requested_budget_mb
                     << " effective_budget_mb=" << effective_budget_mb);

            if (detail_enabled)
            {
                WeightLoadingProfiler::addDetail("weights.gemm_pack.phase4.prepare_workers", static_cast<double>(prepare_workers));
                WeightLoadingProfiler::addDetail("weights.gemm_pack.phase4.upload_workers", static_cast<double>(upload_workers));
                WeightLoadingProfiler::addDetail("weights.gemm_pack.phase4.queue_slots", static_cast<double>(queue_capacity));
                WeightLoadingProfiler::addDetail("weights.gemm_pack.phase4.effective_budget_mb", static_cast<double>(effective_budget_mb));
            }
        }

        struct PreparedJob
        {
            std::string name;
            std::shared_ptr<TensorBase> tensor;
            const KernelFactory::PreparedGemmHandle *prepared = nullptr;
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
        std::atomic<size_t> local_gpu_packed{0};

        auto prepare_job = [&](const std::pair<std::string, std::shared_ptr<TensorBase>> &entry) -> PreparedJob
        {
            PreparedJob job;
            job.name = entry.first;
            job.tensor = entry.second;

            try
            {
                job.prepared = KernelFactory::getOrCreatePreparedGemmWeights(job.tensor.get(), target_device);
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

            auto run_upload = [&]() -> bool
            {
                auto *kernel = KernelFactory::getOrCreateGemmEngine(job.prepared);
                if (!kernel)
                {
                    return false;
                }
                kernel->prepareWeights();
                return true;
            };

            try
            {
                if (target_device.is_rocm())
                {
                    KernelFactory::ROCmOrdinalGuard guard(target_device.ordinal);
                    return run_upload();
                }
                if (target_device.is_cuda())
                {
                    KernelFactory::CUDAOrdinalGuard guard(target_device.ordinal);
                    return run_upload();
                }
                return run_upload();
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
                    if (target_device.is_gpu())
                    {
                        markPrepState(job.name, target_device, WeightPrepState::UPLOADED_DEVICE, true, "GEMM packed + uploaded");
                        local_gpu_packed.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        markPrepState(job.name, target_device, WeightPrepState::PACKED_HOST, true, "GEMM packed on CPU");
                        local_cpu_packed.fetch_add(1, std::memory_order_relaxed);
                        if (release_raw_data && job.tensor)
                        {
                            job.tensor->release_host_weight_data();
                        }
                    }

                    markPrepState(job.name, target_device, WeightPrepState::READY, true, "GEMM weight ready");
                    evaluateReclaimEligibility(job.name, true);
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
            num_gpu_packed_ += local_gpu_packed.load(std::memory_order_relaxed);
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
        auto *prepared = KernelFactory::getOrCreatePreparedGemmWeights(tensor, target_device);
        auto *kernel = KernelFactory::getOrCreateGemmEngine(prepared);
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
                    LOG_WARN("[WeightManager] Non-GEMM weight missing from cache: " << name);
                    continue;
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
                LOG_WARN("[WeightManager] Failed to get device tensor for: " << name);
                continue;
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
        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t released_count = 0;
        size_t skipped_count = 0;
        size_t error_count = 0;
        std::unordered_set<TensorBase *> visited_ptrs;

        auto try_release = [&](TensorBase *ptr)
        {
            if (!ptr || !visited_ptrs.insert(ptr).second)
                return; // null or already visited

            // Skip tensors that are already released
            if (ptr->is_raw_data_released())
            {
                skipped_count++;
                return;
            }

            // Release host data for tensors that have valid GPU data OR
            // kernel-managed device data (GEMM packed weights). GEMM kernels
            // upload pre-packed representations to their own device buffers and
            // never read the raw TensorBase data, so the host copy can be freed.
            if (!ptr->deviceValid())
            {
                // Check if kernel has its own device copy (CUDA/ROCm packed weights)
                bool has_kernel_device_data =
                    ptr->hasCachedDeviceData(DeviceType::CUDA) ||
                    ptr->hasCachedDeviceData(DeviceType::ROCm);
                if (!has_kernel_device_data)
                {
                    skipped_count++;
                    return;
                }
            }

            try
            {
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
            try_release(tensor.get());
        }

        // Sweep per-device cache (clones for multi-GPU)
        for (auto &[key, tensor] : per_device_cache_)
        {
            try_release(tensor.get());
        }

        LOG_INFO("[WeightManager] Released host weight data: " << released_count
                                                               << " tensors released, " << skipped_count
                                                               << " already released, " << error_count << " errors");
        return released_count;
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
            LOG_ERROR("[WeightManager] No sharding config set - cannot compute slice boundaries");
            return false;
        }

        WeightDimensionType dim_type = sharding_config_.getDimensionType(name);

        switch (dim_type)
        {
        case WeightDimensionType::Heads:
        {
            const int total_heads = tp_config_->totalHeads();
            if (total_heads <= 0)
            {
                LOG_ERROR("[WeightManager] Invalid total_heads in TensorParallelConfig");
                return false;
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
                LOG_ERROR("[WeightManager] Invalid total_kv_heads in TensorParallelConfig");
                return false;
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

        case WeightDimensionType::Bias1D:
        case WeightDimensionType::None:
        default:
            LOG_ERROR("[WeightManager] Cannot compute slice boundaries for dimension type "
                      << static_cast<int>(dim_type) << " on weight: " << name);
            return false;
        } // switch
        return false; // unreachable but silences -Wreturn-type
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
            LOG_ERROR("[WeightManager] Failed to compute slice boundaries for 1D tensor: " << name);
            return nullptr;
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

        LOG_DEBUG("[WeightManager] Device " << device.to_string()
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
            LOG_ERROR("[WeightManager] Failed to compute slice boundaries for 2D column-parallel: " << name);
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

        // Fused QKV weights have 3 equal concatenated sub-blocks: [Q_all | K_all | V_all]
        // A simple contiguous row slice would cross Q/K/V boundaries.
        // Instead, load each sub-block's local rows directly from GGUF in native
        // quantized format, then concatenate the raw bytes into a single tensor.
        constexpr size_t N_SUB_BLOCKS = 3;

        if (total_rows % N_SUB_BLOCKS != 0)
        {
            LOG_ERROR("[WeightManager] FusedQKVHeads: total_rows " << total_rows
                                                                   << " not divisible by " << N_SUB_BLOCKS
                                                                   << " for weight: " << name);
            return nullptr;
        }

        const size_t sub_block_rows = total_rows / N_SUB_BLOCKS;

        // Compute per-sub-block head slice
        const int total_heads = tp_config_->totalHeads();
        if (total_heads <= 0)
        {
            LOG_ERROR("[WeightManager] Invalid total_heads for FusedQKVHeads: " << name);
            return nullptr;
        }

        const size_t head_dim = sub_block_rows / static_cast<size_t>(total_heads);
        const size_t local_start = assignment.head_start * head_dim;
        const size_t local_count = assignment.head_count * head_dim;

        // Load 3 row slices from GGUF in native quantized format (no dequantization)
        std::shared_ptr<TensorBase> slices[N_SUB_BLOCKS];
        for (size_t s = 0; s < N_SUB_BLOCKS; s++)
        {
            const size_t abs_row_start = s * sub_block_rows + local_start;
            const size_t abs_row_end = abs_row_start + local_count;

            slices[s] = loader_.loadTensorRowSlice(
                name, abs_row_start, abs_row_end, device, WeightPrecision::NATIVE);

            if (!slices[s])
            {
                LOG_ERROR("[WeightManager] Failed to load fused-QKV sub-block " << s
                                                                                << " rows [" << abs_row_start << ", " << abs_row_end << ")"
                                                                                << " for: " << name);
                return nullptr;
            }
        }

        // Concatenate raw bytes from the 3 slices into a single native tensor
        const size_t bytes_per_slice = slices[0]->size_bytes();
        const size_t total_bytes = bytes_per_slice * N_SUB_BLOCKS;
        std::vector<uint8_t> combined_raw(total_bytes);

        for (size_t s = 0; s < N_SUB_BLOCKS; s++)
        {
            const void *src = slices[s]->raw_data();
            if (!src)
            {
                LOG_ERROR("[WeightManager] Null raw_data for fused-QKV sub-block " << s << ": " << name);
                return nullptr;
            }
            std::memcpy(combined_raw.data() + s * bytes_per_slice, src, bytes_per_slice);
        }

        // Create a new tensor from the concatenated raw bytes
        const size_t out_rows = local_count * N_SUB_BLOCKS;
        std::vector<size_t> out_shape = {out_rows, cols};
        TensorType native_type = slices[0]->native_type();

        auto result_tensor = createTensorFromRawData(
            native_type, out_shape, std::move(combined_raw));

        if (!result_tensor)
        {
            LOG_ERROR("[WeightManager] Failed to create fused-QKV tensor of type "
                      << static_cast<int>(native_type) << " for: " << name);
            return nullptr;
        }

        // Wrap in TensorSlice with column-parallel metadata so downstream TP
        // logic (allreduce detection, etc.) works correctly
        auto meta = SliceMetadata::forColumnParallel(
            total_rows, cols, assignment.local_rank, tp_config_->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(
            std::shared_ptr<TensorBase>(std::move(result_tensor)), meta);

        LOG_DEBUG("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config_->worldSize() << ")"
                                            << " fused-QKV column-parallel " << name
                                            << " [" << total_rows << ", " << cols << "]"
                                            << " -> [" << out_rows << ", " << cols << "]"
                                            << " (" << static_cast<int>(native_type) << " native)"
                                            << " (3 sub-blocks x rows [" << local_start << ", " << (local_start + local_count) << ") each)");

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
            LOG_ERROR("[WeightManager] Failed to compute slice boundaries for row-parallel: " << name);
            return nullptr;
        }

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
            LOG_ERROR("[WeightManager] Failed to compute slice boundaries for input-parallel: " << name);
            return nullptr;
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
        // Check per-device cache first.
        // WeightManager is rank-local, so device string uniquely identifies the
        // cache entry. LOCAL TP devices have distinct DeviceIds (e.g. cuda:0, cuda:1).
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
                LOG_ERROR("[WeightManager] Invalid tensor for column-parallel (expected 1D or 2D): " << name);
                return nullptr;
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
                LOG_ERROR("[WeightManager] Invalid tensor for row-parallel (expected 2D): " << name);
                return nullptr;
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
                LOG_ERROR("[WeightManager] Invalid tensor for input-parallel (expected 2D): " << name);
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

        // Handle special weights (embedding, output norm, LM head)
        if (name == "token_embd.weight")
        {
            return has_embedding_;
        }
        if (name == "output_norm.weight" || name == "output.weight")
        {
            return has_lm_head_;
        }

        // Extract layer index from "blk.N.xxx" pattern
        // Pattern: blk.{layer_idx}.{component}.weight
        static const std::regex layer_pattern(R"(blk\.(\d+)\.)");
        std::smatch match;
        if (std::regex_search(name, match, layer_pattern))
        {
            int layer_idx = std::stoi(match[1].str());
            // Layer range is [first, last) - first inclusive, last exclusive
            return layer_idx >= layer_first_ && layer_idx < layer_last_;
        }

        // Unknown weight pattern - include by default (e.g., custom weights)
        LOG_DEBUG("[WeightManager] Unknown weight pattern, including: " << name);
        return true;
    }

} // namespace llaminar2
