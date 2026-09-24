/**
 * @file PrefillChunkMaterializationStage.cpp
 * @brief Device-only implementation of captured long-prefill chunk publication.
 */

#include "PrefillChunkMaterializationStage.h"

#include "../../../backends/IBackend.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
    PrefillChunkMaterializationStage::PrefillChunkMaterializationStage(
        Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool PrefillChunkMaterializationStage::validate() const
    {
        if (!params_.device_id.is_gpu() || !params_.backend)
        {
            LOG_ERROR("[PrefillChunkMaterializationStage] An explicit GPU backend is required");
            return false;
        }
        if (!params_.request_token_ids_device ||
            !params_.request_position_ids_device ||
            !params_.request_total_rows_device ||
            !params_.cached_tokens_device ||
            !params_.chunk_token_ids_device ||
            !params_.chunk_position_ids_device ||
            !params_.chunk_real_rows_device ||
            !params_.chunk_row_stride_device)
        {
            LOG_ERROR("[PrefillChunkMaterializationStage] Device ownership bindings are incomplete");
            return false;
        }
        if (params_.request_row_capacity <= 0 ||
            params_.bucket_seq_len <= 0 ||
            params_.bucket_seq_len > params_.request_row_capacity ||
            params_.capture_identity == 0)
        {
            LOG_ERROR("[PrefillChunkMaterializationStage] Capture geometry or identity is invalid");
            return false;
        }
        return true;
    }

    bool PrefillChunkMaterializationStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "PrefillChunkMaterializationStage") ||
            !validate())
        {
            return false;
        }

        void *const stream = requireGPUStream();
        if (!stream)
            return false;

        if (!params_.backend->enqueuePreparePrefillChunkView(
                params_.request_token_ids_device,
                params_.request_position_ids_device,
                params_.request_total_rows_device,
                params_.cached_tokens_device,
                params_.request_row_capacity,
                params_.bucket_seq_len,
                params_.pad_token_id,
                params_.device_id.gpu_ordinal(),
                stream,
                params_.chunk_token_ids_device,
                params_.chunk_position_ids_device,
                params_.chunk_real_rows_device,
                params_.chunk_row_stride_device))
        {
            LOG_ERROR("[PrefillChunkMaterializationStage] Backend rejected the captured chunk publication");
            return false;
        }
        return true;
    }

    size_t PrefillChunkMaterializationStage::estimatedFlops() const
    {
        return static_cast<size_t>(std::max(0, params_.bucket_seq_len)) * 3U;
    }

    size_t PrefillChunkMaterializationStage::estimatedMemoryBytes() const
    {
        const size_t rows =
            static_cast<size_t>(std::max(0, params_.bucket_seq_len));
        return rows * sizeof(int32_t) * 4U + sizeof(int32_t) * 5U;
    }

    bool PrefillChunkMaterializationStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo PrefillChunkMaterializationStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("request_row_capacity", params_.request_row_capacity);
        info.addScalarInt("bucket_seq_len", params_.bucket_seq_len);
        info.addScalarInt("pad_token_id", params_.pad_token_id);
        return info;
    }

    StageBufferContract PrefillChunkMaterializationStage::bufferContract() const
    {
        StageBufferContract contract;
        contract.inputs.reserve(3);
        contract.outputs.reserve(3);
        contract.addInput(BufferId::REQUEST_TOKEN_IDS);
        contract.addInput(BufferId::REQUEST_POSITION_IDS);
        contract.addInput(BufferId::REQUEST_BATCH_GEOMETRY);
        contract.addOutput(BufferId::PREFILL_CHUNK_TOKEN_IDS);
        contract.addOutput(BufferId::PREFILL_CHUNK_POSITION_IDS);
        contract.addOutput(BufferId::PREFILL_CHUNK_GEOMETRY);
        return contract;
    }
} // namespace llaminar2
