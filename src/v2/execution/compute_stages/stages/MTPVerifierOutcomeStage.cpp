/**
 * @file MTPVerifierOutcomeStage.cpp
 * @brief Implementation of graph-owned MTP verifier outcome publication.
 */

#include "MTPVerifierOutcomeStage.h"

#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <cstddef>
#include <utility>

namespace llaminar2
{
    MTPVerifierOutcomeStage::MTPVerifierOutcomeStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MTPVerifierOutcomeStage::validate() const
    {
        if (!params_.device_id.is_gpu())
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] The graph-owned outcome path is GPU-only");
            return false;
        }
        if (!params_.logits || !params_.binding.validForGreedy())
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] Missing logits or persistent greedy bindings");
            return false;
        }
        if (params_.mode != MTPVerifierOutcomeGraphMode::Greedy)
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] Unsupported graph outcome mode="
                      << static_cast<int>(params_.mode));
            return false;
        }

        if (params_.ownership_policy !=
            MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal)
        {
            LOG_ERROR(
                "[MTPVerifierOutcomeStage] Terminal verifier reduction has no "
                "supported declarative ownership policy");
            return false;
        }
        if (params_.verifier_row_count <= 0 ||
            params_.vocab_size <= 0 ||
            params_.binding.output_token_capacity <
                params_.verifier_row_count ||
            params_.binding.output_meta_capacity <
                sampling_math::kSpeculativeBatchMetaCount)
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] Invalid verifier/output geometry"
                      << " rows=" << params_.verifier_row_count
                      << " vocab=" << params_.vocab_size
                      << " token_capacity="
                      << params_.binding.output_token_capacity
                      << " meta_capacity="
                      << params_.binding.output_meta_capacity);
            return false;
        }

        return true;
    }

    bool MTPVerifierOutcomeStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MTPVerifierOutcomeStage") || !validate())
            return false;

        void *stream = gpuStream();
        if (!stream)
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] An explicit non-null graph stream is required");
            return false;
        }

        /*
         * ParticipantLocal ownership makes this terminal transaction complete
         * on every graph participant. Mirrored TP guarantees that logits,
         * verifier inputs, stop controls, and penalty history are byte-identical
         * at this point, so identical deterministic kernels produce identical
         * compact mailboxes without a rank authority transfer.
         */
        const auto logits_device = params_.logits->current_device();
        const auto &shape = params_.logits->shape();
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols =
            shape.size() >= 2 ? shape[1]
                              : (shape.empty() ? 0 : shape[0]);
        auto *logits = static_cast<const float *>(
            params_.logits->gpu_data_ptr());

        /*
         * The LM-head node and this consumer execute inside one captured graph.
         * Their dependency and the ALL_POSITION_LOGITS buffer contract are the
         * temporal proof that the producer runs first. A host coherence flag
         * cannot represent an in-graph write: consulting deviceValid() here
         * observes capture-time host metadata and incorrectly rejects the
         * freshly produced device row. Validate only immutable capture identity
         * here: storage device, element type, geometry, and persistent pointer.
         */
        if (!logits_device.has_value() ||
            *logits_device != params_.device_id ||
            params_.logits->native_type() != TensorType::FP32 ||
            rows < static_cast<size_t>(params_.verifier_row_count) ||
            cols != static_cast<size_t>(params_.vocab_size) ||
            !logits)
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] Participant-local verifier logits do not "
                      "match the captured full-vocabulary contract"
                      << " rows=" << rows
                      << " cols=" << cols
                      << " expected_rows=" << params_.verifier_row_count
                      << " expected_cols=" << params_.vocab_size
                      << " dtype=" << params_.logits->dtype_name()
                      << " current_device="
                      << (logits_device.has_value()
                              ? logits_device->toString()
                              : "none")
                      << " expected_device=" << params_.device_id.toString()
                      << " device_ptr=" << static_cast<const void *>(logits));
            return false;
        }

        IBackend *backend = getBackendFor(params_.device_id);
        if (!backend)
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] No backend for "
                      << params_.device_id.toString());
            return false;
        }

        if (!backend->enqueueArgmaxF32BatchedRowsWithMTPPenaltiesDevice(
                logits,
                params_.verifier_row_count,
                params_.vocab_size,
                params_.binding.verifier_input_tokens_device,
                params_.binding.generated_token_counts_device,
                params_.binding.penalty_policy_device,
                params_.binding.active_verifier_row_count_device,
                params_.device_id.gpu_ordinal(),
                stream,
                params_.binding.argmax_values_device,
                params_.binding.verifier_tokens_device,
                params_.binding.argmax_partial_values_device,
                params_.binding.argmax_partial_indices_device,
                params_.binding.argmax_partial_capacity))
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] Batched verifier argmax launch failed");
            return false;
        }

        if (!backend->enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
                params_.binding.verifier_tokens_device,
                params_.binding.verifier_input_tokens_device,
                params_.verifier_row_count - 1,
                params_.binding.active_verifier_row_count_device,
                params_.binding.stop_tokens_device,
                params_.device_id.gpu_ordinal(),
                stream,
                params_.binding.output_token_capacity,
                params_.binding.output_tokens_device,
                params_.binding.output_meta_device,
                params_.binding.transaction_commit_budget_device,
                params_.binding.next_leading_committed_output_count_device))
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] Greedy compact outcome reduction failed");
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "graph_owned_greedy_outcome_stage_enqueues",
            1.0,
            "decode",
            params_.device_id.toString(),
            {{"rows", std::to_string(params_.verifier_row_count)},
             {"participant_full_vocabulary",
              params_.participant_full_vocabulary ? "true" : "false"},
             {"ownership", "participant_local"},
             {"collective", "none"}});
        return true;
    }

    size_t MTPVerifierOutcomeStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(params_.verifier_row_count) *
                   static_cast<size_t>(params_.vocab_size) * sizeof(float) +
               static_cast<size_t>(params_.vocab_size) * sizeof(int32_t) +
               static_cast<size_t>(params_.binding.output_token_capacity +
                                   params_.binding.output_meta_capacity) *
                   sizeof(int32_t);
    }

    bool MTPVerifierOutcomeStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo MTPVerifierOutcomeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addInput(
            "verifier_logits",
            params_.logits,
            static_cast<size_t>(params_.verifier_row_count),
            static_cast<size_t>(params_.vocab_size));
        info.addScalarInt("verifier_rows", params_.verifier_row_count);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarBool(
            "participant_full_vocabulary",
            params_.participant_full_vocabulary);
        return info;
    }

    StageBufferContract MTPVerifierOutcomeStage::bufferContract() const
    {
        StageBufferContract contract;

        contract.addInput(BufferId::ALL_POSITION_LOGITS);
        contract.addInput(BufferId::MTP_VERIFIER_INPUT_TOKENS);
        contract.addInput(BufferId::MTP_VERIFIER_REQUEST_LENGTHS, "INT32");
        contract.addInput(BufferId::MTP_GENERATION_CONTROL, "INT32");
        contract.addInput(BufferId::MTP_VERIFIER_STOP_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_VERIFY_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS);
        contract.addInput(BufferId::MTP_GREEDY_PENALTY_POLICY);
        contract.addInput(
            BufferId::MTP_GENERATED_TOKEN_COUNTS,
            "INT32");
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_META);
        return contract;
    }

} // namespace llaminar2
