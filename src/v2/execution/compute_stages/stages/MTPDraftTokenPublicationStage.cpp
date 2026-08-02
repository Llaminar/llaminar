/**
 * @file MTPDraftTokenPublicationStage.cpp
 * @brief Device-only implementation of captured MTP draft publication.
 */

#include "MTPDraftTokenPublicationStage.h"

#include "../../../backends/IBackend.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
    MTPDraftTokenPublicationStage::MTPDraftTokenPublicationStage(
        Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MTPDraftTokenPublicationStage::validate() const
    {
        if (!params_.device_id.is_gpu() || !params_.backend)
        {
            LOG_ERROR("[MTPDraftTokenPublicationStage] An explicit GPU backend is required");
            return false;
        }
        if (!params_.logits_row_device || params_.logits_row < 0 ||
            params_.vocab_size <= 0 || !params_.draft_values_device ||
            !params_.draft_tokens_device || params_.destination_slot < 0 ||
            !params_.argmax_partial_values_device ||
            !params_.argmax_partial_indices_device ||
            params_.argmax_partial_capacity <= 0)
        {
            LOG_ERROR("[MTPDraftTokenPublicationStage] Invalid logit, destination, or argmax workspace binding");
            return false;
        }
        if (params_.prior_draft_count < 0)
        {
            LOG_ERROR("[MTPDraftTokenPublicationStage] Prior draft count cannot be negative");
            return false;
        }
        if (params_.apply_penalties &&
            (!params_.first_condition_token_device ||
             (params_.prior_draft_count > 0 &&
              !params_.prior_draft_tokens_device) ||
             !params_.generated_token_counts_device ||
             !params_.penalty_policy_device))
        {
            LOG_ERROR("[MTPDraftTokenPublicationStage] Penalty-enabled proposal has incomplete resident branch history");
            return false;
        }
        return true;
    }

    bool MTPDraftTokenPublicationStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MTPDraftTokenPublicationStage") ||
            !validate())
        {
            return false;
        }
        void *const stream = requireGPUStream();
        if (!stream)
            return false;

        if (params_.apply_penalties)
        {
            /*
             * Policy publication and branch scoring share the captured stream.
             * This makes the fragment self-contained: cloning it into a parent
             * loop cannot accidentally consume a stale host-authored policy.
             */
            if (!params_.backend->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                    params_.penalty_policy_device,
                    params_.presence_penalty,
                    params_.frequency_penalty,
                    params_.first_token_already_in_history,
                    params_.device_id.gpu_ordinal(),
                    stream) ||
                !params_.backend->enqueueApplyMTPBranchPenaltiesToF32RowDevice(
                    params_.logits_row_device,
                    params_.vocab_size,
                    params_.first_condition_token_device,
                    params_.prior_draft_tokens_device,
                    params_.prior_draft_count,
                    params_.generated_token_counts_device,
                    params_.penalty_policy_device,
                    params_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[MTPDraftTokenPublicationStage] Serial-equivalent branch penalty transform failed");
                return false;
            }
        }

        if (!params_.backend->enqueueArgmaxF32BatchedRowsDevice(
                params_.logits_row_device,
                /*rows=*/1,
                params_.vocab_size,
                params_.device_id.gpu_ordinal(),
                stream,
                params_.draft_values_device + params_.destination_slot,
                params_.draft_tokens_device + params_.destination_slot,
                params_.argmax_partial_values_device,
                params_.argmax_partial_indices_device,
                params_.argmax_partial_capacity,
                /*output_stride=*/1))
        {
            LOG_ERROR("[MTPDraftTokenPublicationStage] Device argmax publication failed");
            return false;
        }
        return true;
    }

    size_t MTPDraftTokenPublicationStage::estimatedFlops() const
    {
        const size_t cols =
            static_cast<size_t>(std::max(0, params_.vocab_size));
        return cols * (params_.apply_penalties ? 8U : 1U);
    }

    size_t MTPDraftTokenPublicationStage::estimatedMemoryBytes() const
    {
        const size_t cols =
            static_cast<size_t>(std::max(0, params_.vocab_size));
        const size_t logit_passes = params_.apply_penalties ? 2U : 1U;
        return cols * sizeof(float) * logit_passes +
               (sizeof(float) + sizeof(int32_t));
    }

    bool MTPDraftTokenPublicationStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo MTPDraftTokenPublicationStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("logits_row", params_.logits_row);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarInt("destination_slot", params_.destination_slot);
        info.addScalarInt("prior_draft_count", params_.prior_draft_count);
        info.addScalarBool("apply_penalties", params_.apply_penalties);
        info.addScalar("presence_penalty", params_.presence_penalty);
        info.addScalar("frequency_penalty", params_.frequency_penalty);
        info.addScalarBool(
            "first_token_already_in_history",
            params_.first_token_already_in_history);
        return info;
    }

    StageBufferContract MTPDraftTokenPublicationStage::bufferContract() const
    {
        StageBufferContract contract;
        contract.inputs.reserve(params_.apply_penalties ? 5U : 1U);
        contract.outputs.reserve(params_.apply_penalties ? 6U : 4U);
        contract.addInput(BufferId::MTP_LOGITS);
        if (params_.apply_penalties)
        {
            contract.addInput(BufferId::MTP_CONDITION_TOKEN);
            contract.addInput(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS);
            contract.addInput(BufferId::MTP_GENERATED_TOKEN_COUNTS);
            contract.addInput(BufferId::MTP_GREEDY_PENALTY_POLICY);
            contract.addOutput(BufferId::MTP_LOGITS);
            contract.addOutput(BufferId::MTP_GREEDY_PENALTY_POLICY);
        }
        contract.addOutput(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS);
        contract.addOutput(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS);
        contract.addOutput(BufferId::ARGMAX_PARTIAL_VALS);
        contract.addOutput(BufferId::ARGMAX_PARTIAL_IDXS);
        return contract;
    }

    bool MTPDraftTokenPublicationStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        return params_.device_id == other.device_id &&
               params_.backend == other.backend &&
               params_.logits_row_device == other.logits_row_device &&
               params_.logits_row == other.logits_row &&
               params_.vocab_size == other.vocab_size &&
               params_.apply_penalties == other.apply_penalties &&
               params_.first_condition_token_device ==
                   other.first_condition_token_device &&
               params_.prior_draft_tokens_device ==
                   other.prior_draft_tokens_device &&
               params_.prior_draft_count == other.prior_draft_count &&
               params_.generated_token_counts_device ==
                   other.generated_token_counts_device &&
               params_.penalty_policy_device == other.penalty_policy_device &&
               params_.presence_penalty == other.presence_penalty &&
               params_.frequency_penalty == other.frequency_penalty &&
               params_.first_token_already_in_history ==
                   other.first_token_already_in_history &&
               params_.draft_values_device == other.draft_values_device &&
               params_.draft_tokens_device == other.draft_tokens_device &&
               params_.destination_slot == other.destination_slot &&
               params_.argmax_partial_values_device ==
                   other.argmax_partial_values_device &&
               params_.argmax_partial_indices_device ==
                   other.argmax_partial_indices_device &&
               params_.argmax_partial_capacity ==
                   other.argmax_partial_capacity &&
               params_.stage_name == other.stage_name;
    }
} // namespace llaminar2
