/**
 * @file MTPVerifierOutcomeStage.cpp
 * @brief Implementation of graph-owned MTP verifier outcome publication.
 */

#include "MTPVerifierOutcomeStage.h"

#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <cstddef>
#include <span>
#include <utility>

namespace llaminar2
{
    MTPVerifierOutcomeStage::MTPVerifierOutcomeStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
        const bool is_root = isRootParticipant();
        auto &tokens = mirrored_outcome_sidebands_[0];
        tokens.kind = LocalTPCollectiveSidebandKind::Broadcast;
        tokens.send_buffer =
            is_root ? params_.binding.output_tokens_device : nullptr;
        tokens.recv_buffer = params_.binding.output_tokens_device;
        tokens.element_count =
            static_cast<size_t>(params_.binding.output_token_capacity);
        tokens.dtype = CollectiveDataType::INT32;
        tokens.root_device_index = params_.local_tp_root_device_index;
        tokens.name = "mtp_graph_outcome_tokens";

        auto &meta = mirrored_outcome_sidebands_[1];
        meta.kind = LocalTPCollectiveSidebandKind::Broadcast;
        meta.send_buffer =
            is_root ? params_.binding.output_meta_device : nullptr;
        meta.recv_buffer = params_.binding.output_meta_device;
        meta.element_count =
            static_cast<size_t>(params_.binding.output_meta_capacity);
        meta.dtype = CollectiveDataType::INT32;
        meta.root_device_index = params_.local_tp_root_device_index;
        meta.name = "mtp_graph_outcome_meta";
    }

    bool MTPVerifierOutcomeStage::isRootParticipant() const noexcept
    {
        return !params_.publish_mirrored_local_tp ||
               params_.local_tp_device_index ==
                   params_.local_tp_root_device_index;
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

        if (!params_.publish_mirrored_local_tp)
            return true;
        if (!params_.local_tp_ctx ||
            params_.local_tp_ctx->degree() <= 1 ||
            params_.local_tp_device_index < 0 ||
            params_.local_tp_device_index >= params_.local_tp_ctx->degree() ||
            params_.local_tp_root_device_index < 0 ||
            params_.local_tp_root_device_index >= params_.local_tp_ctx->degree())
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] Invalid mirrored LocalTP publication contract");
            return false;
        }
        if (!params_.local_tp_ctx
                 ->supportsCollectiveSidebandOnStreamGraphCapture())
        {
            LOG_ERROR("[MTPVerifierOutcomeStage] LocalTP backend cannot capture compact outcome broadcasts");
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
         * Only the root performs the expensive vocabulary reduction.  Every
         * participant still enters the same two collectives below, so CUDA/HIP
         * graph replay retains a symmetric collective order.
         */
        if (isRootParticipant())
        {
            /*
             * Non-root participants never read their duplicate verifier logits:
             * their only semantic outputs are the persistent token/meta receive
             * buffers written by the collective below. Keeping tensor coherence
             * access inside this branch makes receiver-side logits ownership
             * irrelevant and prevents a host-authoritative duplicate row from
             * creating a false H2D requirement.
             */
            const auto logits_device = params_.logits->current_device();
            const auto &shape = params_.logits->shape();
            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols =
                shape.size() >= 2 ? shape[1]
                                  : (shape.empty() ? 0 : shape[0]);
            if (!params_.logits->deviceValid() ||
                !logits_device.has_value() ||
                *logits_device != params_.device_id ||
                rows < static_cast<size_t>(params_.verifier_row_count) ||
                cols != static_cast<size_t>(params_.vocab_size))
            {
                LOG_ERROR("[MTPVerifierOutcomeStage] Root verifier logits do not "
                          "match the captured full-vocabulary contract"
                          << " rows=" << rows
                          << " cols=" << cols
                          << " expected_rows=" << params_.verifier_row_count
                          << " expected_cols=" << params_.vocab_size);
                return false;
            }
            auto *logits = static_cast<const float *>(
                params_.logits->gpu_data_ptr());
            if (!logits)
            {
                LOG_ERROR("[MTPVerifierOutcomeStage] Root verifier logits have "
                          "no prepared device storage");
                return false;
            }

            IBackend *backend = getBackendFor(params_.device_id);
            if (!backend)
            {
                LOG_ERROR("[MTPVerifierOutcomeStage] No backend for "
                          << params_.device_id.toString());
                return false;
            }

            if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                    logits,
                    params_.verifier_row_count,
                    params_.vocab_size,
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

            if (!backend
                     ->enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
                         params_.binding.verifier_tokens_device,
                         params_.binding.verifier_input_tokens_device,
                         params_.verifier_row_count - 1,
                         params_.binding.stop_tokens_device,
                         params_.device_id.gpu_ordinal(),
                         stream,
                         params_.binding.output_token_capacity,
                         params_.binding.output_tokens_device,
                         params_.binding.output_meta_device))
            {
                LOG_ERROR("[MTPVerifierOutcomeStage] Greedy compact outcome reduction failed");
                return false;
            }
        }

        if (params_.publish_mirrored_local_tp)
        {
            if (!params_.local_tp_ctx->collectiveSidebandSpanOnStream(
                    std::span<const LocalTPCollectiveSidebandBuffer>(
                        mirrored_outcome_sidebands_),
                    params_.local_tp_device_index,
                    stream,
                    params_.stage_name))
            {
                LOG_ERROR("[MTPVerifierOutcomeStage] Captured LocalTP compact outcome publication failed");
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "graph_owned_greedy_outcome_stage_enqueues",
            1.0,
            "decode",
            params_.device_id.toString(),
            {{"rows", std::to_string(params_.verifier_row_count)},
             {"root", isRootParticipant() ? "true" : "false"},
             {"mirrored_local_tp",
              params_.publish_mirrored_local_tp ? "true" : "false"}});
        return true;
    }

    size_t MTPVerifierOutcomeStage::estimatedMemoryBytes() const
    {
        return static_cast<size_t>(params_.verifier_row_count) *
                   static_cast<size_t>(params_.vocab_size) * sizeof(float) +
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
            "mirrored_local_tp",
            params_.publish_mirrored_local_tp);
        return info;
    }

    StageBufferContract MTPVerifierOutcomeStage::bufferContract() const
    {
        StageBufferContract contract;

        /*
         * The contract must describe what this participant actually touches,
         * not the union of the root and receiver algorithms. DeviceGraphExecutor
         * establishes coherence from this declaration before execute() runs.
         * Advertising root-only logits on a receiver would therefore demand a
         * needless H2D transfer before the receiver enters the collective.
         */
        if (isRootParticipant())
        {
            contract.addInput(BufferId::ALL_POSITION_LOGITS);
            contract.addInput(BufferId::MTP_VERIFIER_INPUT_TOKENS);
            contract.addInput(BufferId::MTP_VERIFIER_STOP_TOKENS);
            contract.addOutput(BufferId::STOCHASTIC_VERIFY_TOKENS);
            contract.addOutput(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS);
        }
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS);
        contract.addOutput(BufferId::STOCHASTIC_BATCH_OUTPUT_META);
        return contract;
    }

} // namespace llaminar2
