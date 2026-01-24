/**
 * @file KVCacheGatherStage.cpp
 * @brief Implementation of KVCacheGatherStage
 */

#include "KVCacheGatherStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/cpu/CPUKVCache.h"

namespace llaminar2
{

    // =============================================================================
    // KVCacheGatherStage Implementation
    // =============================================================================

    KVCacheGatherStage::KVCacheGatherStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool KVCacheGatherStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.kv_cache)
        {
            LOG_ERROR("[KVCacheGatherStage] No KV cache provided");
            return false;
        }

        if (!params_.out_K || !params_.out_V)
        {
            LOG_ERROR("[KVCacheGatherStage] Invalid output K/V tensors");
            return false;
        }

        if (params_.batch_size <= 0)
        {
            LOG_ERROR("[KVCacheGatherStage] Invalid batch_size=" << params_.batch_size);
            return false;
        }

        // Cast ITensor* to TensorBase* for CPU KV cache operations
        // TODO: Update ICPUKVCache interface to use ITensor* for device-agnostic support
        auto *out_K_base = dynamic_cast<TensorBase *>(params_.out_K);
        auto *out_V_base = dynamic_cast<TensorBase *>(params_.out_V);
        if (!out_K_base || !out_V_base)
        {
            LOG_ERROR("[KVCacheGatherStage] Output K/V tensors must be CPU TensorBase (GPU not yet supported)");
            return false;
        }

        // Call the unified gather method
        int max_kv_len = params_.kv_cache->gather_kv_batched(
            params_.layer_idx,
            params_.batch_size,
            out_K_base,
            out_V_base,
            last_per_seq_kv_lens_);

        if (max_kv_len < 0)
        {
            LOG_ERROR("[KVCacheGatherStage] gather_kv_batched failed for layer " << params_.layer_idx);
            return false;
        }

        last_max_kv_len_ = max_kv_len;

        // Write outputs if requested
        if (params_.out_max_kv_len)
        {
            *params_.out_max_kv_len = max_kv_len;
        }
        if (params_.out_per_seq_kv_lens)
        {
            *params_.out_per_seq_kv_lens = last_per_seq_kv_lens_;
        }

        LOG_DEBUG("[KVCacheGatherStage] Gathered " << params_.batch_size
                                                   << " sequences from layer " << params_.layer_idx
                                                   << ", max_kv_len=" << max_kv_len);

        return true;
    }

    StageBufferRequirements KVCacheGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Output: K (gathered from cache)
        if (params_.out_K)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.out_K->native_type());
            reqs.addOutput("gathered_K", params_.out_K->shape(), buf_type);
        }

        // Output: V (gathered from cache)
        if (params_.out_V)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.out_V->native_type());
            reqs.addOutput("gathered_V", params_.out_V->shape(), buf_type);
        }

        // Note: KV cache itself is external state, not a buffer managed by this stage

        return reqs;
    }

    StageDumpInfo KVCacheGatherStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Output: gathered K from cache
        if (params_.out_K)
        {
            info.addOutput("gathered_K", params_.out_K, params_.out_K->rows(), params_.out_K->cols());
        }

        // Output: gathered V from cache
        if (params_.out_V)
        {
            info.addOutput("gathered_V", params_.out_V, params_.out_V->rows(), params_.out_V->cols());
        }

        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("last_max_kv_len", last_max_kv_len_);

        return info;
    }

} // namespace llaminar2
