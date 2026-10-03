/**
 * @file MTPStochasticOutcomeStage.cpp
 * @brief Complete captured stochastic outcomes for both request sampling laws.
 *
 * The seeded law keeps the existing fused serial-equivalent kernel. Probability
 * rejection captures the established verifier, bonus sample and summary on the
 * same stream, with all draws derived from immutable request entropy and the
 * device-owned verifier position. Neither law reads back intermediate state.
 */

#include "MTPStochasticOutcomeStage.h"

#include "../../../backends/IBackend.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
    MTPStochasticOutcomeStage::MTPStochasticOutcomeStage(
        Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MTPStochasticOutcomeStage::validate() const
    {
        using namespace sampling_math;

        constexpr int kMaxComparisonRows = 15;
        if (!params_.device_id.is_gpu() || !params_.backend)
        {
            LOG_ERROR("[MTPStochasticOutcomeStage] An explicit GPU backend is required");
            return false;
        }
        if (params_.request_count <= 0 ||
            params_.comparison_rows_per_request <= 0 ||
            params_.comparison_rows_per_request > kMaxComparisonRows ||
            params_.verifier_row_capacity <
                params_.comparison_rows_per_request + 1 ||
            static_cast<int>(params_.threshold_seeds.size()) !=
                params_.request_count ||
            std::any_of(
                params_.threshold_seeds.begin(),
                params_.threshold_seeds.end(),
                [](uint64_t seed) { return seed == 0; }))
        {
            LOG_ERROR("[MTPStochasticOutcomeStage] Invalid request, depth, or seed geometry");
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
            LOG_ERROR("[MTPStochasticOutcomeStage] Incomplete resident input binding");
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
            LOG_ERROR("[MTPStochasticOutcomeStage] Incomplete compact output binding");
            return false;
        }
        if (params_.advance_maintenance_boundary &&
            (!params_.decode_rounds_committed_device ||
             !params_.decode_rounds_until_maintenance_device ||
             !params_.maintenance_due_device ||
             !params_.decode_boundary_advanced_device))
        {
            LOG_ERROR("[MTPStochasticOutcomeStage] Maintenance publication is missing controller fields");
            return false;
        }
        switch (params_.verification)
        {
        case Verification::SerialEquivalent:
            break;
        case Verification::OneHotProbabilityRejection:
            if (!params_.accepted_rows_device ||
                params_.accepted_row_stride < params_.comparison_rows_per_request ||
                params_.vocabulary_size < params_.top_k)
            {
                LOG_ERROR("[MTPStochasticOutcomeStage] Rejection requires resident decisions and full vocabulary geometry");
                return false;
            }
            break;
        default:
            return false;
        }
        return true;
    }

    bool MTPStochasticOutcomeStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MTPStochasticOutcomeStage") ||
            !validate())
        {
            return false;
        }
        void *const stream = requireGPUStream();
        if (!stream)
            return false;

        const int target_rows_per_request =
            params_.comparison_rows_per_request + 1;
        for (int request = 0; request < params_.request_count; ++request)
        {
            const size_t target_row_offset =
                static_cast<size_t>(request) *
                static_cast<size_t>(target_rows_per_request) *
                static_cast<size_t>(params_.target_distribution_row_stride);
            if (params_.verification == Verification::OneHotProbabilityRejection)
            {
                // The materialized verifier row owns both the condition token
                // and its drafts. Never resurrect a host shadow or a stale
                // pre-publication sampler slot as the first-token authority.
                const auto *tokens = params_.verifier_input_tokens_device +
                    static_cast<size_t>(request) * params_.verifier_input_token_stride;
                auto *sampled = params_.sampled_target_tokens_device +
                    static_cast<size_t>(request) * params_.sampled_target_token_stride;
                auto *accepted = params_.accepted_rows_device +
                    static_cast<size_t>(request) * params_.accepted_row_stride;
                const auto *position = params_.threshold_base_positions_device + request;
                const uint64_t seed = params_.threshold_seeds[static_cast<size_t>(request)];
                const auto bonus_offset = target_row_offset +
                    static_cast<size_t>(params_.comparison_rows_per_request) *
                        params_.target_distribution_row_stride;
                if (!params_.backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                        params_.target_token_ids_device + target_row_offset,
                        params_.target_probs_device + target_row_offset,
                        nullptr, nullptr, params_.top_k,
                        params_.target_distribution_row_stride, tokens + 1,
                        nullptr, nullptr, params_.comparison_rows_per_request,
                        params_.device_id.gpu_ordinal(), stream, sampled, accepted,
                        nullptr, nullptr, nullptr, seed, -1,
                        params_.vocabulary_size, position, params_.threshold_position_offset) ||
                    !params_.backend->enqueueSampleDistributionF32Device(
                        params_.target_token_ids_device + bonus_offset,
                        params_.target_probs_device + bonus_offset, params_.top_k,
                        0.0f, params_.device_id.gpu_ordinal(), stream,
                        sampled + params_.comparison_rows_per_request, nullptr,
                        seed, position,
                        params_.threshold_position_offset + params_.comparison_rows_per_request) ||
                    !params_.backend->enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
                        sampled, accepted, nullptr, params_.comparison_rows_per_request,
                        tokens, params_.stop_tokens_device +
                            static_cast<size_t>(request) * params_.stop_token_stride,
                        sampled + params_.comparison_rows_per_request, true,
                        params_.generation_control_device +
                            static_cast<size_t>(request) * params_.generation_control_stride,
                        params_.device_id.gpu_ordinal(), stream, params_.output_token_stride,
                        params_.output_tokens_device +
                            static_cast<size_t>(request) * params_.output_token_stride,
                        params_.output_meta_device +
                            static_cast<size_t>(request) * params_.output_meta_stride))
                {
                    LOG_ERROR("[MTPStochasticOutcomeStage] Captured rejection launch failed for request " << request);
                    return false;
                }
            }
            else if (!params_.backend
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
                                 params_.output_meta_stride,
                         request == 0
                             ? params_.first_transaction_diagnostic_device
                             : nullptr))
            {
                LOG_ERROR("[MTPStochasticOutcomeStage] Fused request outcome launch failed for request "
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
            LOG_ERROR("[MTPStochasticOutcomeStage] Maintenance-boundary advancement failed");
            return false;
        }
        return true;
    }

    size_t MTPStochasticOutcomeStage::estimatedMemoryBytes() const
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
                   sizeof(int32_t) +
               (params_.verification == Verification::OneHotProbabilityRejection
                    ? target_rows * sizeof(int32_t)
                    : 0u) +
               (params_.first_transaction_diagnostic_device
                    ? sizeof(
                          sampling_math::MTPFirstTransactionDiagnosticRecord)
                    : 0u);
    }

    bool MTPStochasticOutcomeStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo MTPStochasticOutcomeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("request_count", params_.request_count);
        info.addScalarInt("verification", static_cast<int>(params_.verification));
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

    StageBufferContract MTPStochasticOutcomeStage::bufferContract() const
    {
        StageBufferContract contract;
        contract.addInput(BufferId::STOCHASTIC_TARGET_TOKEN_IDS);
        contract.addInput(BufferId::STOCHASTIC_TARGET_PROBS);
        contract.addInput(BufferId::MTP_VERIFIER_INPUT_TOKENS);
        contract.addInput(BufferId::MTP_VERIFIER_STOP_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_VERIFY_TOKENS);
        if (params_.verification == Verification::OneHotProbabilityRejection)
            contract.addOutput(BufferId::STOCHASTIC_VERIFY_ACCEPTED);
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_META);
        if (params_.first_transaction_diagnostic_device)
        {
            contract.addOutput(
                BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC);
        }
        return contract;
    }

    bool MTPStochasticOutcomeStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        return params_.device_id == other.device_id &&
               params_.backend == other.backend &&
               params_.verification == other.verification &&
               params_.vocabulary_size == other.vocabulary_size &&
               params_.accepted_rows_device == other.accepted_rows_device &&
               params_.accepted_row_stride == other.accepted_row_stride &&
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
               params_.first_transaction_diagnostic_device ==
                   other.first_transaction_diagnostic_device &&
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
