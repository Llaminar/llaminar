/**
 * @file MTPStochasticTargetDistributionStage.cpp
 * @brief Device-only implementation of captured stochastic target preparation.
 */

#include "MTPStochasticTargetDistributionStage.h"

#include "../../../backends/IBackend.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
    MTPStochasticTargetDistributionStage::
        MTPStochasticTargetDistributionStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MTPStochasticTargetDistributionStage::validate() const
    {
        if (!params_.device_id.is_gpu() || !params_.backend)
        {
            LOG_ERROR("[MTPStochasticTargetDistributionStage] An explicit GPU backend is required");
            return false;
        }
        if (!params_.logits_device || params_.first_logit_row < 0 ||
            params_.row_count <= 0 ||
            params_.vocab_size <= 0 ||
            params_.logits_row_stride < params_.vocab_size)
        {
            LOG_ERROR("[MTPStochasticTargetDistributionStage] Invalid verifier-logit geometry or binding");
            return false;
        }
        if (params_.top_k <= 0 ||
            params_.top_k > params_.target_row_stride ||
            params_.top_p <= 0.0F || params_.top_p > 1.0F ||
            params_.temperature <= 0.0F ||
            !params_.target_token_ids_device ||
            !params_.target_probs_device ||
            params_.first_target_slot < 0 ||
            params_.target_row_stride <= 0 ||
            !params_.topk_partial_values_device ||
            !params_.topk_partial_indices_device ||
            params_.topk_partial_capacity == 0)
        {
            LOG_ERROR("[MTPStochasticTargetDistributionStage] Invalid compact-distribution policy or workspace");
            return false;
        }
        if (params_.apply_penalties &&
            (!params_.verifier_input_tokens_device ||
             !params_.generated_token_counts_device ||
             !params_.penalty_policy_device))
        {
            LOG_ERROR("[MTPStochasticTargetDistributionStage] Penalty-enabled capture has incomplete resident history bindings");
            return false;
        }
        return true;
    }

    bool MTPStochasticTargetDistributionStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(
                ctx,
                "MTPStochasticTargetDistributionStage") ||
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
             * The graph consumes request-admitted policy and the history bit
             * published by the previous accepted-state transaction. Mutation is
             * forbidden here: this stage owns logits/distribution bytes only.
             */
            if (!params_.backend->enqueueApplyMTPPenaltiesToF32RowsDevice(
                    params_.logits_device,
                    params_.row_count,
                    params_.vocab_size,
                    params_.logits_row_stride,
                    params_.verifier_input_tokens_device,
                    params_.generated_token_counts_device,
                    params_.penalty_policy_device,
                    params_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[MTPStochasticTargetDistributionStage] Device-history penalty transform failed");
                return false;
            }
        }

        if (!params_.backend->enqueueBuildTopKTopPDistributionsF32Device(
                params_.logits_device,
                params_.row_count,
                params_.vocab_size,
                params_.logits_row_stride,
                params_.top_k,
                params_.top_p,
                params_.temperature,
                params_.device_id.gpu_ordinal(),
                stream,
                params_.target_token_ids_device,
                params_.target_row_stride,
                params_.target_probs_device,
                params_.topk_partial_values_device,
                params_.topk_partial_indices_device,
                params_.topk_partial_capacity))
        {
            LOG_ERROR("[MTPStochasticTargetDistributionStage] Compact target-row construction failed");
            return false;
        }
        return true;
    }

    size_t MTPStochasticTargetDistributionStage::estimatedFlops() const
    {
        const size_t rows = static_cast<size_t>(std::max(0, params_.row_count));
        const size_t cols = static_cast<size_t>(std::max(0, params_.vocab_size));
        return rows * cols * (params_.apply_penalties ? 8U : 6U);
    }

    size_t MTPStochasticTargetDistributionStage::estimatedMemoryBytes() const
    {
        const size_t rows = static_cast<size_t>(std::max(0, params_.row_count));
        const size_t cols = static_cast<size_t>(std::max(0, params_.vocab_size));
        const size_t compact_stride =
            static_cast<size_t>(std::max(0, params_.target_row_stride));
        return rows * cols * sizeof(float) +
               rows * compact_stride * (sizeof(int32_t) + sizeof(float));
    }

    bool MTPStochasticTargetDistributionStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo
    MTPStochasticTargetDistributionStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("first_logit_row", params_.first_logit_row);
        info.addScalarInt("row_count", params_.row_count);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarInt("first_target_slot", params_.first_target_slot);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalar("top_p", params_.top_p);
        info.addScalar("temperature", params_.temperature);
        info.addScalarBool("apply_penalties", params_.apply_penalties);
        return info;
    }

    StageBufferContract
    MTPStochasticTargetDistributionStage::bufferContract() const
    {
        StageBufferContract contract;
        contract.inputs.reserve(params_.apply_penalties ? 4U : 1U);
        contract.outputs.reserve(params_.apply_penalties ? 5U : 4U);
        contract.addInput(BufferId::ALL_POSITION_LOGITS);
        if (params_.apply_penalties)
        {
            contract.addInput(BufferId::MTP_VERIFIER_INPUT_TOKENS);
            contract.addInput(BufferId::MTP_GENERATED_TOKEN_COUNTS);
            contract.addInput(BufferId::MTP_GREEDY_PENALTY_POLICY);
        }
        contract.addOutput(BufferId::ALL_POSITION_LOGITS);
        contract.addOutput(BufferId::STOCHASTIC_TARGET_TOKEN_IDS);
        contract.addOutput(BufferId::STOCHASTIC_TARGET_PROBS);
        contract.addOutput(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS);
        contract.addOutput(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS);
        return contract;
    }

    bool MTPStochasticTargetDistributionStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        return params_.device_id == other.device_id &&
               params_.backend == other.backend &&
               params_.logits_device == other.logits_device &&
               params_.first_logit_row == other.first_logit_row &&
               params_.row_count == other.row_count &&
               params_.vocab_size == other.vocab_size &&
               params_.logits_row_stride == other.logits_row_stride &&
               params_.apply_penalties == other.apply_penalties &&
               params_.verifier_input_tokens_device ==
                   other.verifier_input_tokens_device &&
               params_.generated_token_counts_device ==
                   other.generated_token_counts_device &&
               params_.penalty_policy_device == other.penalty_policy_device &&
               params_.top_k == other.top_k &&
               params_.top_p == other.top_p &&
               params_.temperature == other.temperature &&
               params_.target_token_ids_device ==
                   other.target_token_ids_device &&
               params_.target_probs_device == other.target_probs_device &&
               params_.first_target_slot == other.first_target_slot &&
               params_.target_row_stride == other.target_row_stride &&
               params_.topk_partial_values_device ==
                   other.topk_partial_values_device &&
               params_.topk_partial_indices_device ==
                   other.topk_partial_indices_device &&
               params_.topk_partial_capacity ==
                   other.topk_partial_capacity &&
               params_.stage_name == other.stage_name;
    }

} // namespace llaminar2
