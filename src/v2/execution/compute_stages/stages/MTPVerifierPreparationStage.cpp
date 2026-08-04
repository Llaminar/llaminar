/**
 * @file MTPVerifierPreparationStage.cpp
 * @brief Device-only implementation of captured grouped-verifier preparation.
 */

#include "MTPVerifierPreparationStage.h"

#include "../../../backends/IBackend.h"
#include "../../../kernels/IKVCache.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace llaminar2
{
    MTPVerifierPreparationStage::MTPVerifierPreparationStage(Params params)
        : IComputeStage(params.device_id),
          owned_token_rows_(
              params.token_rows.begin(),
              params.token_rows.end()),
          owned_main_kv_checkpoints_(
              params.main_kv_checkpoints.begin(),
              params.main_kv_checkpoints.end()),
          params_(std::move(params))
    {
        /*
         * Params enters through non-owning spans so callers can compare graph
         * identities with preallocated scratch and no decode-time allocation.
         * Rebind the retained Params to stage-owned immutable copies before the
         * caller's scratch is reused for another geometry.
         */
        params_.token_rows = owned_token_rows_;
        params_.main_kv_checkpoints = owned_main_kv_checkpoints_;
    }

    bool MTPVerifierPreparationStage::validate() const
    {
        const int total_rows = params_.request_count * params_.padded_seq_len;
        const bool has_generation_control =
            params_.generation_control_device != nullptr;
        const bool has_any_maintenance_boundary =
            params_.maintenance_rows_remaining_device != nullptr ||
            params_.maintenance_due_device != nullptr ||
            params_.decode_boundary_advanced_device != nullptr;
        const bool has_complete_maintenance_boundary =
            params_.maintenance_rows_remaining_device != nullptr &&
            params_.maintenance_due_device != nullptr &&
            params_.decode_boundary_advanced_device != nullptr;
        if (!params_.device_id.is_gpu() || !params_.backend)
        {
            LOG_ERROR("[MTPVerifierPreparationStage] An explicit GPU backend is required");
            return false;
        }
        if (params_.request_count <= 0 || params_.padded_seq_len <= 0 ||
            total_rows <= 0 ||
            static_cast<int>(params_.token_rows.size()) !=
                params_.request_count ||
            !std::all_of(
                params_.token_rows.begin(),
                params_.token_rows.end(),
                [&](const TokenRowBinding &row)
                {
                    return row.valid(params_.padded_seq_len) &&
                           (!has_generation_control ||
                            row.draft_tokens_device != nullptr);
                }))
        {
            LOG_ERROR("[MTPVerifierPreparationStage] Token-row cardinality or geometry is invalid");
            return false;
        }
        if (!params_.base_cached_tokens_device ||
            !params_.position_ids_device ||
            !params_.request_lengths_device ||
            !params_.base_cached_tokens_snapshot_device ||
            params_.valid_graph_row_count <= 0 ||
            params_.valid_graph_row_count > total_rows ||
            (!params_.valid_graph_rows_device &&
             params_.valid_graph_row_count % params_.request_count != 0))
        {
            LOG_ERROR("[MTPVerifierPreparationStage] Device-owned verifier geometry bindings are incomplete");
            return false;
        }
        if (has_generation_control !=
                (params_.generation_control_stride > 0) ||
            (has_generation_control &&
             params_.generation_control_stride <
                 sampling_math::kDeviceGenerationControlCount) ||
            (has_any_maintenance_boundary &&
             (!has_generation_control ||
              !has_complete_maintenance_boundary)))
        {
            LOG_ERROR("[MTPVerifierPreparationStage] Device-generation controller or maintenance-boundary geometry is incomplete");
            return false;
        }
        if (static_cast<int>(params_.main_kv_checkpoints.size()) !=
                params_.request_count ||
            !std::all_of(
                params_.main_kv_checkpoints.begin(),
                params_.main_kv_checkpoints.end(),
                [](const MainKVCheckpointBinding &binding)
                {
                    return binding.valid();
                }))
        {
            LOG_ERROR("[MTPVerifierPreparationStage] Main-KV checkpoint cardinality does not match verifier requests");
            return false;
        }
        return true;
    }

    bool MTPVerifierPreparationStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MTPVerifierPreparationStage") || !validate())
            return false;

        void *const stream = requireGPUStream();
        if (!stream)
            return false;
        const int device_ordinal = params_.device_id.gpu_ordinal();

        /*
         * Padding is semantically inactive, but embedding still dereferences its
         * token id before later masks suppress the row. Zero only rows that have
         * padding, then overwrite the valid prefix from authoritative device
         * sources. Rectangular scalar verification therefore pays no memset.
         */
        const bool device_generation_controlled =
            params_.generation_control_device != nullptr;
        if (device_generation_controlled &&
            !params_.backend->enqueuePrepareDeviceGenerationTransactionBudget(
                params_.generation_control_device,
                params_.generation_control_stride,
                params_.request_count,
                params_.padded_seq_len,
                params_.maintenance_rows_remaining_device,
                params_.maintenance_due_device,
                params_.decode_boundary_advanced_device,
                device_ordinal,
                stream))
        {
            LOG_ERROR("[MTPVerifierPreparationStage] Failed to publish the resident verifier transaction budget");
            return false;
        }
        if (!device_generation_controlled)
        {
            for (const TokenRowBinding &row : params_.token_rows)
            {
                const int valid_tokens = row.draft_token_count + 1;
                if (valid_tokens < params_.padded_seq_len &&
                    !params_.backend->memset(
                        row.destination_device,
                        0,
                        static_cast<size_t>(params_.padded_seq_len) *
                            sizeof(int32_t),
                        device_ordinal,
                        stream))
                {
                    LOG_ERROR("[MTPVerifierPreparationStage] Failed to clear an inactive verifier token suffix");
                    return false;
                }
                if (!params_.backend->deviceCopyAsync(
                        row.destination_device,
                        row.first_token_device,
                        sizeof(int32_t),
                        device_ordinal,
                        stream))
                {
                    LOG_ERROR("[MTPVerifierPreparationStage] Failed to publish a resident verifier condition token");
                    return false;
                }
                if (row.draft_token_count > 0 &&
                    !params_.backend->deviceCopyAsync(
                        row.destination_device + 1,
                        row.draft_tokens_device,
                        static_cast<size_t>(row.draft_token_count) *
                            sizeof(int32_t),
                        device_ordinal,
                        stream))
                {
                    LOG_ERROR("[MTPVerifierPreparationStage] Failed to publish resident verifier draft tokens");
                    return false;
                }
            }
        }

        /*
         * Capture every request's complete ring-head/count bank before verifier
         * append kernels can mutate it. Publication later restores accepted state
         * relative to these opaque bytes, which is strictly stronger than keeping
         * only a layer-zero count scalar.
         */
        for (const MainKVCheckpointBinding &binding :
             params_.main_kv_checkpoints)
        {
            std::string error;
            if (!binding.cache->captureDeviceSequenceStateCheckpoint(
                    binding.sequence_index,
                    binding.checkpoint_device,
                    binding.checkpoint_bytes,
                    stream,
                    &error))
            {
                LOG_ERROR("[MTPVerifierPreparationStage] Main-KV checkpoint capture failed for sequence "
                          << binding.sequence_index << ": " << error);
                return false;
            }
        }

        if (device_generation_controlled)
        {
            for (int request = 0; request < params_.request_count; ++request)
            {
                const TokenRowBinding &row =
                    params_.token_rows[static_cast<size_t>(request)];
                if (!params_.backend->enqueuePrepareMTPVerifierControlledRow(
                        row.first_token_device,
                        row.draft_tokens_device,
                        params_.base_cached_tokens_device + request,
                        params_.generation_control_device +
                            static_cast<size_t>(request) *
                                static_cast<size_t>(
                                    params_.generation_control_stride),
                        params_.generation_control_stride,
                        params_.padded_seq_len,
                        device_ordinal,
                        stream,
                        row.destination_device,
                        params_.position_ids_device +
                            static_cast<size_t>(request) *
                                static_cast<size_t>(params_.padded_seq_len),
                        params_.request_lengths_device + request,
                        params_.base_cached_tokens_snapshot_device + request))
                {
                    LOG_ERROR("[MTPVerifierPreparationStage] Controlled verifier transaction publication failed for request "
                              << request);
                    return false;
                }
            }
        }
        else
        {
            if (!params_.backend->enqueuePrepareMTPVerifierGeometry(
                    params_.base_cached_tokens_device,
                    params_.valid_graph_rows_device,
                    params_.valid_graph_row_count,
                    /*generation_control_device=*/nullptr,
                    /*generation_control_stride=*/0,
                    params_.request_count,
                    params_.padded_seq_len,
                    device_ordinal,
                    stream,
                    params_.position_ids_device,
                    params_.request_lengths_device))
            {
                LOG_ERROR("[MTPVerifierPreparationStage] Device verifier geometry publication failed");
                return false;
            }
            if (!params_.backend->deviceCopyAsync(
                    params_.base_cached_tokens_snapshot_device,
                    params_.base_cached_tokens_device,
                    static_cast<size_t>(params_.request_count) *
                        sizeof(int32_t),
                    device_ordinal,
                    stream))
            {
                LOG_ERROR("[MTPVerifierPreparationStage] Device verifier base-count snapshot failed");
                return false;
            }
        }
        return true;
    }

    size_t MTPVerifierPreparationStage::estimatedFlops() const
    {
        return static_cast<size_t>(std::max(0, params_.request_count)) *
               static_cast<size_t>(std::max(0, params_.padded_seq_len)) * 3U;
    }

    size_t MTPVerifierPreparationStage::estimatedMemoryBytes() const
    {
        const size_t token_rows =
            static_cast<size_t>(std::max(0, params_.request_count)) *
            static_cast<size_t>(std::max(0, params_.padded_seq_len));
        size_t checkpoint_bytes = 0;
        for (const MainKVCheckpointBinding &binding :
             params_.main_kv_checkpoints)
        {
            checkpoint_bytes += binding.checkpoint_bytes;
        }
        return token_rows * sizeof(int32_t) * 2U + checkpoint_bytes +
               static_cast<size_t>(std::max(0, params_.request_count)) *
                   sizeof(int32_t) * 3U;
    }

    bool MTPVerifierPreparationStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo MTPVerifierPreparationStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("request_count", params_.request_count);
        info.addScalarInt("padded_seq_len", params_.padded_seq_len);
        info.addScalarInt(
            "valid_graph_row_count",
            params_.valid_graph_row_count);
        info.addScalarBool(
            "ragged_geometry",
            params_.valid_graph_rows_device != nullptr);
        info.addScalarBool(
            "device_generation_controlled_geometry",
            params_.generation_control_device != nullptr);
        return info;
    }

    StageBufferContract MTPVerifierPreparationStage::bufferContract() const
    {
        StageBufferContract contract;
        contract.addInput(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS);
        contract.addInput(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS);
        contract.addOutput(BufferId::MTP_VERIFIER_INPUT_TOKENS);
        contract.addOutput(BufferId::MTP_VERIFIER_POSITION_IDS);
        contract.addOutput(BufferId::MTP_VERIFIER_REQUEST_LENGTHS);
        if (params_.generation_control_device)
        {
            contract.addPreallocatedInOut(
                BufferId::MTP_GENERATION_CONTROL,
                "INT32");
        }
        return contract;
    }

    bool MTPVerifierPreparationStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        const bool device_generation_controlled =
            params_.generation_control_device != nullptr;
        const bool other_device_generation_controlled =
            other.generation_control_device != nullptr;
        if (device_generation_controlled !=
            other_device_generation_controlled)
        {
            return false;
        }
        const auto token_row_identity_equal =
            [device_generation_controlled](
                const TokenRowBinding &left,
                const TokenRowBinding &right)
        {
            return left.first_token_device == right.first_token_device &&
                   left.draft_tokens_device == right.draft_tokens_device &&
                   left.destination_device == right.destination_device &&
                   (device_generation_controlled ||
                    left.draft_token_count == right.draft_token_count);
        };
        return params_.device_id == other.device_id &&
               params_.backend == other.backend &&
               params_.token_rows.size() == other.token_rows.size() &&
               std::equal(
                   params_.token_rows.begin(),
                   params_.token_rows.end(),
                   other.token_rows.begin(),
                   token_row_identity_equal) &&
               params_.request_count == other.request_count &&
               params_.padded_seq_len == other.padded_seq_len &&
               params_.base_cached_tokens_device ==
                   other.base_cached_tokens_device &&
               (device_generation_controlled ||
                (params_.valid_graph_rows_device ==
                     other.valid_graph_rows_device &&
                 params_.valid_graph_row_count ==
                     other.valid_graph_row_count)) &&
               params_.generation_control_device ==
                   other.generation_control_device &&
               params_.generation_control_stride ==
                   other.generation_control_stride &&
               params_.maintenance_rows_remaining_device ==
                   other.maintenance_rows_remaining_device &&
               params_.maintenance_due_device ==
                   other.maintenance_due_device &&
               params_.decode_boundary_advanced_device ==
                   other.decode_boundary_advanced_device &&
               params_.position_ids_device == other.position_ids_device &&
               params_.request_lengths_device ==
                   other.request_lengths_device &&
               params_.base_cached_tokens_snapshot_device ==
                   other.base_cached_tokens_snapshot_device &&
               params_.main_kv_checkpoints.size() ==
                   other.main_kv_checkpoints.size() &&
               std::equal(
                   params_.main_kv_checkpoints.begin(),
                   params_.main_kv_checkpoints.end(),
                   other.main_kv_checkpoints.begin()) &&
               params_.stage_name == other.stage_name;
    }

    std::string MTPVerifierPreparationStage::describeCaptureIdentity(
        const Params &params)
    {
        std::ostringstream out;
        out << "device=" << params.device_id.toString()
            << ",backend=" << params.backend
            << ",requests=" << params.request_count
            << ",padded_seq_len=" << params.padded_seq_len
            << ",base_cached_tokens="
            << static_cast<const void *>(params.base_cached_tokens_device)
            << ",valid_graph_rows="
            << static_cast<const void *>(params.valid_graph_rows_device)
            << ",valid_graph_row_count=" << params.valid_graph_row_count
            << ",generation_control="
            << static_cast<const void *>(params.generation_control_device)
            << ",generation_control_stride="
            << params.generation_control_stride
            << ",maintenance_rows_remaining="
            << static_cast<const void *>(
                   params.maintenance_rows_remaining_device)
            << ",maintenance_due="
            << static_cast<const void *>(params.maintenance_due_device)
            << ",decode_boundary_advanced="
            << static_cast<const void *>(
                   params.decode_boundary_advanced_device)
            << ",position_ids="
            << static_cast<const void *>(params.position_ids_device)
            << ",request_lengths="
            << static_cast<const void *>(params.request_lengths_device)
            << ",base_snapshot="
            << static_cast<const void *>(
                   params.base_cached_tokens_snapshot_device)
            << ",stage_name=" << params.stage_name
            << ",token_rows=[";

        for (size_t index = 0; index < params.token_rows.size(); ++index)
        {
            if (index > 0)
                out << ';';
            const TokenRowBinding &row = params.token_rows[index];
            out << index
                << ":first="
                << static_cast<const void *>(row.first_token_device)
                << ",draft="
                << static_cast<const void *>(row.draft_tokens_device)
                << ",destination="
                << static_cast<const void *>(row.destination_device)
                << ",draft_count=" << row.draft_token_count;
        }
        out << "],checkpoints=[";
        for (size_t index = 0;
             index < params.main_kv_checkpoints.size();
             ++index)
        {
            if (index > 0)
                out << ';';
            const MainKVCheckpointBinding &checkpoint =
                params.main_kv_checkpoints[index];
            out << index
                << ":cache=" << checkpoint.cache
                << ",sequence=" << checkpoint.sequence_index
                << ",destination=" << checkpoint.checkpoint_device
                << ",bytes=" << checkpoint.checkpoint_bytes;
        }
        out << ']';
        return out.str();
    }
} // namespace llaminar2
