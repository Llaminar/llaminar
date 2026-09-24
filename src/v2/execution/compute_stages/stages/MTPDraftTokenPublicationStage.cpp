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
            !params_.next_chain_condition_token_device ||
            !params_.next_chain_position_id_device ||
            params_.chain_position_increment <= 0 ||
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

        const bool has_diagnostic_bindings =
            params_.first_transaction_diagnostic_device != nullptr ||
            params_.diagnostic_condition_token_device != nullptr ||
            params_.diagnostic_position_id_device != nullptr ||
            params_.diagnostic_generation_control_device != nullptr ||
            params_.diagnostic_generation_control_stride != 0 ||
            std::any_of(
                params_.diagnostic_boundary_words_device.begin(),
                params_.diagnostic_boundary_words_device.end(),
                [](const void *pointer) { return pointer != nullptr; }) ||
            std::any_of(
                params_.diagnostic_boundary_word_counts.begin(),
                params_.diagnostic_boundary_word_counts.end(),
                [](int count) { return count != 0; });
        if (has_diagnostic_bindings)
        {
            if (!params_.first_transaction_diagnostic_device ||
                !params_.diagnostic_condition_token_device ||
                !params_.diagnostic_position_id_device ||
                !params_.diagnostic_generation_control_device ||
                params_.diagnostic_generation_control_stride <
                    sampling_math::kDeviceGenerationControlCount)
            {
                LOG_ERROR("[MTPDraftTokenPublicationStage] Transaction-zero diagnostics require one complete resident controller boundary");
                return false;
            }
            for (size_t boundary = 0;
                 boundary < params_.diagnostic_boundary_word_counts.size();
                 ++boundary)
            {
                const int word_count =
                    params_.diagnostic_boundary_word_counts[boundary];
                if (word_count < 0 ||
                    (word_count > 0 &&
                     !params_.diagnostic_boundary_words_device[boundary]))
                {
                    LOG_ERROR("[MTPDraftTokenPublicationStage] Transaction-zero diagnostic boundary has invalid word geometry");
                    return false;
                }
            }
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
             * Request admission and accepted-state publication own policy
             * mutation. This fragment only consumes the graph-stable resident
             * policy, including the device-produced history predicate.
             */
            if (!params_.backend->enqueueApplyMTPBranchPenaltiesToF32RowDevice(
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

        if (params_.first_transaction_diagnostic_device)
        {
            /*
             * The sidecar activation workspace is overwritten by the next
             * proposal fragment.  Retain every named boundary now, on this
             * exact stream, after optional scoring transforms and before the
             * argmax publishes the next chained-sidecar inputs.  The backend
             * kernel itself admits only controller transaction zero.
             */
            for (size_t boundary = 0;
                 boundary < params_.diagnostic_boundary_word_counts.size();
                 ++boundary)
            {
                if (!params_.backend
                         ->enqueueRetainMTPFirstTransactionDraftBoundaryDevice(
                             params_.diagnostic_boundary_words_device[boundary],
                             params_.diagnostic_boundary_word_counts[boundary],
                             static_cast<int>(boundary),
                             params_.destination_slot,
                             params_.diagnostic_condition_token_device,
                             params_.diagnostic_position_id_device,
                             params_.diagnostic_generation_control_device,
                             params_.diagnostic_generation_control_stride,
                             params_.first_transaction_diagnostic_device,
                             params_.device_id.gpu_ordinal(),
                             stream))
                {
                    LOG_ERROR("[MTPDraftTokenPublicationStage] Transaction-zero sidecar boundary retention failed");
                    return false;
                }
            }
        }

        if (!params_.backend
                 ->enqueueArgmaxF32BatchedRowsAndPublishMTPChainDevice(
                params_.logits_row_device,
                /*rows=*/1,
                params_.vocab_size,
                params_.device_id.gpu_ordinal(),
                stream,
                params_.draft_values_device + params_.destination_slot,
                params_.draft_tokens_device + params_.destination_slot,
                params_.next_chain_condition_token_device,
                params_.next_chain_position_id_device,
                params_.chain_position_increment,
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
        size_t diagnostic_words = 0;
        if (params_.first_transaction_diagnostic_device)
        {
            for (int word_count : params_.diagnostic_boundary_word_counts)
                diagnostic_words += static_cast<size_t>(std::max(0, word_count));
        }
        return cols * (params_.apply_penalties ? 8U : 1U) +
               diagnostic_words * 3U;
    }

    size_t MTPDraftTokenPublicationStage::estimatedMemoryBytes() const
    {
        const size_t cols =
            static_cast<size_t>(std::max(0, params_.vocab_size));
        const size_t logit_passes = params_.apply_penalties ? 2U : 1U;
        size_t diagnostic_bytes = 0;
        if (params_.first_transaction_diagnostic_device)
        {
            for (int word_count : params_.diagnostic_boundary_word_counts)
            {
                diagnostic_bytes +=
                    static_cast<size_t>(std::max(0, word_count)) *
                    sizeof(uint32_t);
            }
        }
        return cols * sizeof(float) * logit_passes + diagnostic_bytes +
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
        info.addScalarInt(
            "chain_position_increment",
            params_.chain_position_increment);
        info.addScalarInt("prior_draft_count", params_.prior_draft_count);
        info.addScalarBool("apply_penalties", params_.apply_penalties);
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
        }
        contract.addOutput(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS);
        contract.addOutput(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS);
        contract.addOutput(BufferId::MTP_CONDITION_TOKEN);
        contract.addOutput(BufferId::MTP_POSITION_IDS);
        contract.addOutput(BufferId::ARGMAX_PARTIAL_VALS);
        contract.addOutput(BufferId::ARGMAX_PARTIAL_IDXS);
        if (params_.first_transaction_diagnostic_device)
        {
            contract.addInput(BufferId::MTP_GENERATION_CONTROL);
            contract.addInput(BufferId::MTP_CONDITION_TOKEN);
            contract.addInput(BufferId::MTP_POSITION_IDS);
            contract.addOutput(BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC);
        }
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
               params_.draft_values_device == other.draft_values_device &&
               params_.draft_tokens_device == other.draft_tokens_device &&
               params_.destination_slot == other.destination_slot &&
               params_.next_chain_condition_token_device ==
                   other.next_chain_condition_token_device &&
               params_.next_chain_position_id_device ==
                   other.next_chain_position_id_device &&
               params_.chain_position_increment ==
                   other.chain_position_increment &&
               params_.argmax_partial_values_device ==
                   other.argmax_partial_values_device &&
               params_.argmax_partial_indices_device ==
                   other.argmax_partial_indices_device &&
               params_.argmax_partial_capacity ==
                   other.argmax_partial_capacity &&
               params_.diagnostic_condition_token_device ==
                   other.diagnostic_condition_token_device &&
               params_.diagnostic_position_id_device ==
                   other.diagnostic_position_id_device &&
               params_.diagnostic_generation_control_device ==
                   other.diagnostic_generation_control_device &&
               params_.diagnostic_generation_control_stride ==
                   other.diagnostic_generation_control_stride &&
               params_.first_transaction_diagnostic_device ==
                   other.first_transaction_diagnostic_device &&
               params_.diagnostic_boundary_words_device ==
                   other.diagnostic_boundary_words_device &&
               params_.diagnostic_boundary_word_counts ==
                   other.diagnostic_boundary_word_counts &&
               params_.stage_name == other.stage_name;
    }
} // namespace llaminar2
