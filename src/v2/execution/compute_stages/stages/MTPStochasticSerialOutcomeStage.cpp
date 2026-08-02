/**
 * @file MTPStochasticSerialOutcomeStage.cpp
 * @brief Implementation of captured serial-equivalent stochastic MTP reduction.
 */

#include "MTPStochasticSerialOutcomeStage.h"

#include "../../../backends/IBackend.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
    MTPStochasticSerialOutcomeStage::MTPStochasticSerialOutcomeStage(
        Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MTPStochasticSerialOutcomeStage::validate() const
    {
        using namespace sampling_math;

        constexpr int kMaxComparisonRows = 15;
        if (!params_.device_id.is_gpu() || !params_.backend)
        {
            LOG_ERROR("[MTPStochasticSerialOutcomeStage] An explicit GPU backend is required");
            return false;
        }
        if (params_.request_count <= 0 ||
            params_.comparison_rows_per_request <= 0 ||
            params_.comparison_rows_per_request > kMaxComparisonRows ||
            params_.verifier_row_capacity <
                params_.comparison_rows_per_request ||
            static_cast<int>(params_.threshold_seeds.size()) !=
                params_.request_count ||
            std::any_of(
                params_.threshold_seeds.begin(),
                params_.threshold_seeds.end(),
                [](uint64_t seed) { return seed == 0; }))
        {
            LOG_ERROR("[MTPStochasticSerialOutcomeStage] Invalid request, depth, or seed geometry");
            return false;
        }
        if (!params_.target_token_ids_device ||
            !params_.target_probs_device ||
            params_.target_distribution_row_stride <= 0 ||
            params_.top_k <= 0 ||
            params_.top_k > params_.target_distribution_row_stride ||
            !params_.verifier_input_tokens_device ||
            params_.verifier_input_token_stride <
                params_.comparison_rows_per_request + 1 ||
            !params_.stop_tokens_device ||
            (params_.stop_token_stride != 0 &&
             params_.stop_token_stride < kSpeculativeBatchMaxStopTokens) ||
            !params_.threshold_base_positions_device ||
            !params_.generation_control_device ||
            params_.generation_control_stride <= 0)
        {
            LOG_ERROR("[MTPStochasticSerialOutcomeStage] Incomplete resident input binding");
            return false;
        }
        if (!params_.sampled_target_tokens_device ||
            params_.sampled_target_token_stride <
                params_.comparison_rows_per_request + 1 ||
            !params_.output_tokens_device ||
            params_.output_token_stride <
                params_.comparison_rows_per_request + 1 ||
            !params_.output_meta_device ||
            params_.output_meta_stride < kSpeculativeBatchMetaCount)
        {
            LOG_ERROR("[MTPStochasticSerialOutcomeStage] Incomplete compact output binding");
            return false;
        }
        if (params_.advance_maintenance_boundary &&
            (!params_.decode_rounds_committed_device ||
             !params_.decode_rounds_until_maintenance_device ||
             !params_.maintenance_due_device ||
             !params_.decode_boundary_advanced_device))
        {
            LOG_ERROR("[MTPStochasticSerialOutcomeStage] Maintenance publication is missing controller fields");
            return false;
        }
        return true;
    }

    bool MTPStochasticSerialOutcomeStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MTPStochasticSerialOutcomeStage") ||
            !validate())
        {
            return false;
        }
        void *const stream = requireGPUStream();
        if (!stream)
            return false;

        /*
         * The generation controller is the sole authority for transaction
         * clipping.  Preparing its budget inside this captured stage prevents a
         * host depth shadow from selecting a different commit boundary than the
         * fused reducer that consumes it.
         */
        if (!params_.backend->enqueuePrepareDeviceGenerationTransactionBudget(
                params_.generation_control_device,
                params_.generation_control_stride,
                params_.request_count,
                params_.verifier_row_capacity,
                params_.maintenance_rows_remaining_device,
                params_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[MTPStochasticSerialOutcomeStage] Transaction budget launch failed");
            return false;
        }

        const int target_rows_per_request =
            params_.comparison_rows_per_request + 1;
        for (int request = 0; request < params_.request_count; ++request)
        {
            const size_t target_row_offset =
                static_cast<size_t>(request) *
                static_cast<size_t>(target_rows_per_request) *
                static_cast<size_t>(params_.target_distribution_row_stride);
            if (!params_.backend
                     ->enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
                         params_.target_token_ids_device + target_row_offset,
                         params_.target_probs_device + target_row_offset,
                         params_.target_distribution_row_stride,
                         params_.top_k,
                         params_.comparison_rows_per_request,
                         params_.threshold_seeds[static_cast<size_t>(request)],
                         params_.threshold_base_positions_device + request,
                         params_.threshold_position_offset,
                         params_.verifier_input_tokens_device +
                             static_cast<size_t>(request) *
                                 params_.verifier_input_token_stride,
                         params_.stop_tokens_device +
                             static_cast<size_t>(request) *
                                 params_.stop_token_stride,
                         params_.generation_control_device +
                             static_cast<size_t>(request) *
                                 params_.generation_control_stride,
                         params_.device_id.gpu_ordinal(),
                         stream,
                         params_.output_token_stride,
                         params_.sampled_target_tokens_device +
                             static_cast<size_t>(request) *
                                 params_.sampled_target_token_stride,
                         params_.output_tokens_device +
                             static_cast<size_t>(request) *
                                 params_.output_token_stride,
                         params_.output_meta_device +
                             static_cast<size_t>(request) *
                                 params_.output_meta_stride))
            {
                LOG_ERROR("[MTPStochasticSerialOutcomeStage] Fused request outcome launch failed for request "
                          << request);
                return false;
            }
        }

        if (params_.advance_maintenance_boundary &&
            !params_.backend->enqueueAdvanceSpeculativeCommitBoundary(
                params_.output_meta_device,
                params_.request_count,
                params_.output_meta_stride,
                params_.decode_rounds_committed_device,
                params_.decode_rounds_until_maintenance_device,
                params_.maintenance_due_device,
                params_.decode_boundary_advanced_device,
                params_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[MTPStochasticSerialOutcomeStage] Maintenance-boundary advancement failed");
            return false;
        }
        return true;
    }

    size_t MTPStochasticSerialOutcomeStage::estimatedMemoryBytes() const
    {
        const size_t target_rows =
            static_cast<size_t>(params_.request_count) *
            static_cast<size_t>(params_.comparison_rows_per_request + 1);
        return target_rows *
                   static_cast<size_t>(params_.target_distribution_row_stride) *
                   (sizeof(int32_t) + sizeof(float)) +
               static_cast<size_t>(params_.request_count) *
                   static_cast<size_t>(params_.output_token_stride +
                                       params_.output_meta_stride) *
                   sizeof(int32_t);
    }

    bool MTPStochasticSerialOutcomeStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo MTPStochasticSerialOutcomeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("request_count", params_.request_count);
        info.addScalarInt(
            "comparison_rows_per_request",
            params_.comparison_rows_per_request);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt(
            "target_distribution_row_stride",
            params_.target_distribution_row_stride);
        info.addScalarInt(
            "threshold_position_offset",
            params_.threshold_position_offset);
        info.addScalarBool(
            "advance_maintenance_boundary",
            params_.advance_maintenance_boundary);
        return info;
    }

    StageBufferContract MTPStochasticSerialOutcomeStage::bufferContract() const
    {
        StageBufferContract contract;
        contract.addInput(BufferId::STOCHASTIC_TARGET_TOKEN_IDS);
        contract.addInput(BufferId::STOCHASTIC_TARGET_PROBS);
        contract.addInput(BufferId::MTP_VERIFIER_INPUT_TOKENS);
        contract.addInput(BufferId::MTP_VERIFIER_STOP_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_VERIFY_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_META);
        return contract;
    }

    bool MTPStochasticSerialOutcomeStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        return params_.device_id == other.device_id &&
               params_.backend == other.backend &&
               params_.target_token_ids_device ==
                   other.target_token_ids_device &&
               params_.target_probs_device == other.target_probs_device &&
               params_.target_distribution_row_stride ==
                   other.target_distribution_row_stride &&
               params_.top_k == other.top_k &&
               params_.verifier_input_tokens_device ==
                   other.verifier_input_tokens_device &&
               params_.verifier_input_token_stride ==
                   other.verifier_input_token_stride &&
               params_.stop_tokens_device == other.stop_tokens_device &&
               params_.stop_token_stride == other.stop_token_stride &&
               params_.threshold_base_positions_device ==
                   other.threshold_base_positions_device &&
               params_.threshold_position_offset ==
                   other.threshold_position_offset &&
               params_.threshold_seeds == other.threshold_seeds &&
               params_.generation_control_device ==
                   other.generation_control_device &&
               params_.generation_control_stride ==
                   other.generation_control_stride &&
               params_.maintenance_rows_remaining_device ==
                   other.maintenance_rows_remaining_device &&
               params_.verifier_row_capacity ==
                   other.verifier_row_capacity &&
               params_.sampled_target_tokens_device ==
                   other.sampled_target_tokens_device &&
               params_.sampled_target_token_stride ==
                   other.sampled_target_token_stride &&
               params_.output_tokens_device == other.output_tokens_device &&
               params_.output_token_stride == other.output_token_stride &&
               params_.output_meta_device == other.output_meta_device &&
               params_.output_meta_stride == other.output_meta_stride &&
               params_.request_count == other.request_count &&
               params_.comparison_rows_per_request ==
                   other.comparison_rows_per_request &&
               params_.advance_maintenance_boundary ==
                   other.advance_maintenance_boundary &&
               params_.decode_rounds_committed_device ==
                   other.decode_rounds_committed_device &&
               params_.decode_rounds_until_maintenance_device ==
                   other.decode_rounds_until_maintenance_device &&
               params_.maintenance_due_device ==
                   other.maintenance_due_device &&
               params_.decode_boundary_advanced_device ==
                   other.decode_boundary_advanced_device &&
               params_.stage_name == other.stage_name;
    }

} // namespace llaminar2
