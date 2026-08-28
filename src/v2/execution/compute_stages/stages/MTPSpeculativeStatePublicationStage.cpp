/**
 * @file MTPSpeculativeStatePublicationStage.cpp
 * @brief Implementation of captured device-owned MTP accepted-state publication.
 */

#include "MTPSpeculativeStatePublicationStage.h"

#include "../../../backends/IBackend.h"
#include "../../../execution/mtp/MTPSpecStatePublisher.h"
#include "../../../execution/moe/IMoEGroupedVerifierHistogramPublisher.h"
#include "../../../kernels/IKVCache.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../memory/BufferId.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
    MTPSpeculativeStatePublicationStage::
        MTPSpeculativeStatePublicationStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params))
    {
    }

    bool MTPSpeculativeStatePublicationStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR(
                "[MTPSpeculativeStatePublicationStage] Graph preparation "
                "requires an exact non-null producer stream");
            return false;
        }
        if (!params_.moe_histogram_publishers.empty() &&
            stream != params_.moe_histogram_publication_stream)
        {
            LOG_ERROR(
                "[MTPSpeculativeStatePublicationStage] Graph preparation "
                "received a foreign histogram publication stream"
                << " stream=" << stream
                << " expected="
                << params_.moe_histogram_publication_stream);
            return false;
        }
        setGPUStream(stream);
        for (IMoEGroupedVerifierHistogramPublisher *publisher :
            params_.moe_histogram_publishers)
        {
            if (!publisher ||
                publisher->groupedVerifierHistogramRole() !=
                    MoEGroupedVerifierHistogramRole::DeferredAcceptedRows ||
                !publisher->prepareGroupedVerifierHistogramProducer(stream))
            {
                LOG_ERROR(
                    "[MTPSpeculativeStatePublicationStage] Failed to prepare "
                    "a deferred MoE histogram producer"
                    << " publisher="
                    << (publisher
                            ? publisher->groupedVerifierHistogramPublisherName()
                            : std::string_view{"null"})
                    << " layer="
                    << (publisher
                            ? publisher->groupedVerifierHistogramLayerIndex()
                            : -1));
                return false;
            }
        }
        return true;
    }

    bool MTPSpeculativeStatePublicationStage::validate() const
    {
        if (!params_.device_id.is_gpu() || !params_.backend)
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Publication requires an explicit GPU backend");
            return false;
        }
        if (params_.request_count <= 0 ||
            params_.verifier_rows_per_request <= 0 ||
            params_.max_state_commit_rows < 0 ||
            params_.max_state_commit_rows >
                params_.verifier_rows_per_request ||
            !params_.outcome_tokens_device ||
            !params_.outcome_meta_device ||
            params_.outcome_token_stride <= 0 ||
            params_.outcome_meta_stride <= 0 ||
            !params_.base_cached_tokens_device ||
            !params_.accepted_restore_rows_device ||
            !params_.target_cached_tokens_device ||
            !params_.accepted_state_counts_device ||
            !params_.publication_ok_flags_device ||
            !params_.next_condition_tokens_device ||
            !params_.all_drafts_accepted_flags_device ||
            !params_.stopped_flags_device ||
            !params_.next_verifier_condition_tokens_device)
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Incomplete compact-outcome or publication metadata binding");
            return false;
        }
        if (static_cast<int>(params_.main_kv_bindings.size()) !=
                params_.request_count ||
            !std::all_of(
                params_.main_kv_bindings.begin(),
                params_.main_kv_bindings.end(),
                [](const MainKVBinding &binding)
                {
                    return binding.valid();
                }))
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Main-KV checkpoint cardinality does not match request geometry");
            return false;
        }
        if (params_.generation_controller_owned &&
            (!params_.generation_response_tokens_device ||
             params_.generation_response_token_stride <= 0 ||
             !params_.generation_control_device ||
             params_.generation_control_stride <= 0 ||
             !params_.verifier_input_tokens_device ||
             params_.verifier_input_token_stride <
                 params_.verifier_rows_per_request ||
             !params_.committed_verifier_identity_device ||
             !params_.next_sidecar_condition_tokens_device ||
             !params_.next_sidecar_position_ids_device))
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Controller-owned publication has no persistent response ledger or next-sidecar mailbox");
            return false;
        }
        if (params_.publish_shifted_kv &&
            (params_.shifted_kv_caches.empty() ||
             !params_.shifted_target_cached_tokens_device ||
             !params_.shifted_accepted_state_counts_device ||
             !std::all_of(
                 params_.shifted_kv_caches.begin(),
                 params_.shifted_kv_caches.end(),
                 [](const IKVCache *cache)
                 {
                     return cache != nullptr;
                 })))
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Shifted-KV publication has incomplete persistent bindings");
            return false;
        }
        if (params_.request_count == 1 &&
            (!params_.penalty_policy_device ||
             !params_.generated_token_counts_device ||
             params_.vocab_size <= 0))
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Penalty-history publication has invalid scalar geometry or storage");
            return false;
        }
        if (!std::all_of(
                params_.moe_histogram_publishers.begin(),
                params_.moe_histogram_publishers.end(),
                [](const IMoEGroupedVerifierHistogramPublisher *publisher)
                {
                    return publisher != nullptr;
                }) ||
            !std::all_of(
                params_.verifier_state_stages.begin(),
                params_.verifier_state_stages.end(),
                [](const IComputeStage *stage)
                {
                    return stage != nullptr;
                }))
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Publication graph contains a null verifier stage binding");
            return false;
        }
        if (params_.moe_histogram_publishers.empty() !=
            (params_.moe_histogram_publication_stream == nullptr))
        {
            LOG_ERROR(
                "[MTPSpeculativeStatePublicationStage] Deferred MoE publishers and their table-owned stream must be bound together");
            return false;
        }
        return true;
    }

    bool MTPSpeculativeStatePublicationStage::publishMoEHistograms(
        void *stream) const
    {
        for (IMoEGroupedVerifierHistogramPublisher *publisher :
             params_.moe_histogram_publishers)
        {
            if (publisher->groupedVerifierHistogramRole() !=
                    MoEGroupedVerifierHistogramRole::DeferredAcceptedRows ||
                !publisher->enqueueCommittedGroupedVerifierHistograms(
                    params_.accepted_state_counts_device,
                    params_.publication_ok_flags_device,
                    params_.request_count,
                    params_.verifier_rows_per_request,
                    stream))
            {
                LOG_ERROR("[MTPSpeculativeStatePublicationStage] Failed to enqueue committed MoE verifier history for "
                          << publisher->groupedVerifierHistogramPublisherName()
                          << " layer="
                          << publisher->groupedVerifierHistogramLayerIndex());
                return false;
            }
        }
        return true;
    }

    bool MTPSpeculativeStatePublicationStage::publishMainKV(
        void *stream) const
    {
        for (int request_index = 0;
             request_index < params_.request_count;
             ++request_index)
        {
            const MainKVBinding &binding =
                params_.main_kv_bindings[
                    static_cast<size_t>(request_index)];
            IKVCache::DeviceSequenceStatePublicationRequest request{
                .request_count = 1,
                .first_seq_idx = binding.first_sequence_index,
                .target_cached_tokens_device =
                    params_.target_cached_tokens_device + request_index,
                .accepted_state_counts_device =
                    params_.accepted_state_counts_device + request_index,
                .publication_ok_flags_device =
                    params_.publication_ok_flags_device + request_index,
                .basis =
                    IKVCache::DeviceSequenceStatePublicationBasis::
                        CapturedBase,
                .base_sequence_state_checkpoint_device =
                    binding.base_checkpoint_device,
                .base_sequence_state_checkpoint_bytes =
                    binding.base_checkpoint_bytes,
                .stream = stream,
            };
            std::string error;
            if (!binding.cache->publishSequenceStateFromDeviceMetadata(
                    request,
                    &error))
            {
                LOG_ERROR("[MTPSpeculativeStatePublicationStage] Main-KV publication failed for request "
                          << request_index << ": " << error);
                return false;
            }
        }
        return true;
    }

    bool MTPSpeculativeStatePublicationStage::publishShiftedKV(
        void *stream) const
    {
        if (!params_.publish_shifted_kv)
            return true;

        for (size_t depth = 0;
             depth < params_.shifted_kv_caches.size();
             ++depth)
        {
            if (!params_.backend
                     ->enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary(
                         params_.base_cached_tokens_device,
                         params_.target_cached_tokens_device,
                         params_.publication_ok_flags_device,
                         params_.request_count,
                         static_cast<int>(depth),
                         params_.device_id.gpu_ordinal(),
                         stream,
                         params_.shifted_target_cached_tokens_device,
                         params_.shifted_accepted_state_counts_device,
                         params_.publication_ok_flags_device))
            {
                LOG_ERROR("[MTPSpeculativeStatePublicationStage] Shifted publication metadata derivation failed at depth "
                          << depth);
                return false;
            }

            IKVCache::DeviceSequenceStatePublicationRequest request{
                .request_count = params_.request_count,
                .first_seq_idx = 0,
                .target_cached_tokens_device =
                    params_.shifted_target_cached_tokens_device,
                .accepted_state_counts_device =
                    params_.shifted_accepted_state_counts_device,
                .publication_ok_flags_device =
                    params_.publication_ok_flags_device,
                .basis =
                    IKVCache::DeviceSequenceStatePublicationBasis::
                        CurrentVisibleWindow,
                .base_sequence_state_checkpoint_device = nullptr,
                .base_sequence_state_checkpoint_bytes = 0,
                .stream = stream,
            };
            std::string error;
            if (!params_.shifted_kv_caches[depth]
                     ->publishSequenceStateFromDeviceMetadata(
                         request,
                         &error))
            {
                LOG_ERROR("[MTPSpeculativeStatePublicationStage] Shifted-KV publication failed at depth "
                          << depth << ": " << error);
                return false;
            }
        }
        return true;
    }

    bool MTPSpeculativeStatePublicationStage::publishPenaltyHistory(
        void *stream) const
    {
        /*
         * Request-batched publication has no shared token-history owner. Its
         * geometry is already a distinct capture identity. Every canonical
         * single-request graph, however, contains this node regardless of
         * whether penalties are enabled. The resident policy makes disabled
         * publication a deterministic device no-op and keeps policy values out
         * of graph topology.
         */
        if (params_.request_count != 1)
            return true;

        if (!params_.backend->enqueueCommitMTPGreedyPenaltyHistoryDevice(
                params_.outcome_tokens_device,
                params_.outcome_meta_device,
                params_.penalty_policy_device,
                params_.accepted_state_counts_device,
                params_.stopped_flags_device,
                params_.outcome_token_stride,
                params_.vocab_size,
                params_.generated_token_counts_device,
                params_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Device-owned penalty-history commit failed");
            return false;
        }
        return true;
    }

    bool MTPSpeculativeStatePublicationStage::publishVerifierState(
        void *stream) const
    {
        MTPDeviceVerifierStatePublicationShape shape{
            .request_count = params_.request_count,
            .target_rows = params_.verifier_rows_per_request,
            .request_id = 0,
        };
        MTPSpecStatePublicationResult result;
        if (params_.request_count == 1)
        {
            result = publishAcceptedMTPSpecStateFromDeviceVerifierRow(
                shape,
                params_.accepted_restore_rows_device,
                params_.verifier_state_stages,
                params_.device_id,
                stream,
                params_.require_captured_verifier_state);
        }
        else
        {
            result = publishAcceptedMTPSpecStateFromDeviceVerifierRows(
                shape,
                params_.accepted_restore_rows_device,
                /*row_index_stride=*/1,
                params_.verifier_state_stages,
                params_.device_id,
                stream,
                params_.require_captured_verifier_state);
        }
        if (!result.ok)
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Accepted verifier-state publication failed: "
                      << result.error);
            return false;
        }
        return true;
    }

    bool MTPSpeculativeStatePublicationStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(
                ctx,
                "MTPSpeculativeStatePublicationStage") ||
            !validate())
        {
            return false;
        }
        void *const stream = requireGPUStream();

        /*
         * The first kernel joins response visibility and state visibility.  A
         * controller-owned stochastic transaction is not allowed to mutate KV
         * or recurrent state unless its emitted bytes were accepted by the same
         * generation budget.  Greedy graph outcomes use the equivalent compact
         * metadata derivation because their response commit is already inside
         * the verifier graph.
         */
        const bool metadata_enqueued =
            params_.generation_controller_owned
                ? params_.backend
                      ->enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
                          params_.outcome_tokens_device,
                          params_.outcome_token_stride,
                          params_.outcome_meta_device,
                          params_.outcome_meta_stride,
                          params_.base_cached_tokens_device,
                          params_.request_count,
                          params_.verifier_rows_per_request,
                          params_.generation_response_tokens_device,
                          params_.generation_response_token_stride,
                          params_.generation_control_device,
                          params_.generation_control_stride,
                          params_.device_id.gpu_ordinal(),
                          stream,
                          params_.accepted_restore_rows_device,
                          params_.target_cached_tokens_device,
                          params_.accepted_state_counts_device,
                          params_.publication_ok_flags_device,
                          params_.next_condition_tokens_device,
                          params_.all_drafts_accepted_flags_device,
                          params_.stopped_flags_device,
                          params_.next_sidecar_condition_tokens_device,
                          params_.next_sidecar_position_ids_device,
                          params_.next_verifier_condition_tokens_device,
                          params_.verifier_input_tokens_device,
                          params_.verifier_input_token_stride,
                          params_.committed_verifier_identity_device)
                : params_.backend
                      ->enqueueDeriveSpeculativePublicationMetadata(
                          params_.outcome_meta_device,
                          params_.outcome_meta_stride,
                          params_.base_cached_tokens_device,
                          params_.request_count,
                          params_.verifier_rows_per_request,
                          params_.max_state_commit_rows,
                          params_.device_id.gpu_ordinal(),
                          stream,
                          params_.accepted_restore_rows_device,
                          params_.target_cached_tokens_device,
                          params_.accepted_state_counts_device,
                          params_.publication_ok_flags_device,
                          params_.next_condition_tokens_device,
                          params_.outcome_tokens_device,
                          params_.outcome_token_stride,
                          params_.all_drafts_accepted_flags_device,
                          params_.stopped_flags_device,
                          params_.next_verifier_condition_tokens_device);
        if (!metadata_enqueued)
        {
            LOG_ERROR("[MTPSpeculativeStatePublicationStage] Response/state metadata transaction failed to enqueue");
            return false;
        }

        if (!publishMoEHistograms(stream) ||
            !publishMainKV(stream) ||
            !publishShiftedKV(stream) ||
            !publishPenaltyHistory(stream) ||
            !publishVerifierState(stream))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "captured_speculative_state_publication_stage_enqueues",
            1.0,
            "decode",
            params_.device_id.toString(),
            {{"requests", std::to_string(params_.request_count)},
             {"verifier_rows",
              std::to_string(params_.verifier_rows_per_request)},
             {"shifted_depths",
              std::to_string(params_.shifted_kv_caches.size())},
             {"moe_histogram_publishers",
              std::to_string(params_.moe_histogram_publishers.size())},
             {"controller_owned",
              params_.generation_controller_owned ? "true" : "false"}});
        return true;
    }

    size_t MTPSpeculativeStatePublicationStage::estimatedMemoryBytes() const
    {
        const size_t request_rows =
            static_cast<size_t>(std::max(0, params_.request_count));
        const size_t metadata_words =
            request_rows * 8U * sizeof(int32_t);
        const size_t shifted_words =
            params_.publish_shifted_kv
                ? request_rows * 3U *
                      params_.shifted_kv_caches.size() * sizeof(int32_t)
                : 0U;
        const size_t committed_identity_bytes =
            params_.generation_controller_owned
                ? request_rows *
                      sizeof(
                          sampling_math::
                              MTPCommittedVerifierIdentityRecord)
                : 0U;
        return metadata_words + shifted_words + committed_identity_bytes;
    }

    bool MTPSpeculativeStatePublicationStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo
    MTPSpeculativeStatePublicationStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("request_count", params_.request_count);
        info.addScalarInt(
            "verifier_rows_per_request",
            params_.verifier_rows_per_request);
        info.addScalarInt(
            "max_state_commit_rows",
            params_.max_state_commit_rows);
        info.addScalarBool(
            "generation_controller_owned",
            params_.generation_controller_owned);
        info.addScalarBool(
            "publish_shifted_kv",
            params_.publish_shifted_kv);
        return info;
    }

    StageBufferContract
    MTPSpeculativeStatePublicationStage::bufferContract() const
    {
        StageBufferContract contract;
        contract.addInput(
            BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS,
            "INT32");
        contract.addInOut(
            BufferId::STOCHASTIC_BATCH_OUTPUT_META,
            "INT32");
        contract.addPreallocatedOutput(
            BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS,
            "INT32");
        if (params_.generation_controller_owned)
        {
            contract.addPreallocatedInOut(
                BufferId::MTP_GENERATION_RESPONSE_TOKENS,
                "INT32");
            contract.addPreallocatedInOut(
                BufferId::MTP_GENERATION_CONTROL,
                "INT32");
            contract.addInput(
                BufferId::MTP_VERIFIER_INPUT_TOKENS,
                "INT32");
            contract.addPreallocatedOutput(
                BufferId::MTP_COMMITTED_VERIFIER_IDENTITY,
                "INT32");
            contract.addPreallocatedOutput(
                BufferId::MTP_CONDITION_TOKEN,
                "INT32");
            contract.addPreallocatedOutput(
                BufferId::MTP_POSITION_IDS,
                "INT32");
        }
        if (params_.request_count == 1)
        {
            contract.addPreallocatedInOut(
                BufferId::MTP_GREEDY_PENALTY_POLICY,
                "INT32");
            contract.addPreallocatedInOut(
                BufferId::MTP_GENERATED_TOKEN_COUNTS,
                "INT32");
        }
        return contract;
    }

    bool MTPSpeculativeStatePublicationStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        const Params &self = params_;
        return self.device_id == other.device_id &&
               self.backend == other.backend &&
               self.outcome_tokens_device ==
                   other.outcome_tokens_device &&
               self.outcome_meta_device == other.outcome_meta_device &&
               self.outcome_token_stride == other.outcome_token_stride &&
               self.outcome_meta_stride == other.outcome_meta_stride &&
               self.base_cached_tokens_device ==
                   other.base_cached_tokens_device &&
               self.accepted_restore_rows_device ==
                   other.accepted_restore_rows_device &&
               self.target_cached_tokens_device ==
                   other.target_cached_tokens_device &&
               self.accepted_state_counts_device ==
                   other.accepted_state_counts_device &&
               self.publication_ok_flags_device ==
                   other.publication_ok_flags_device &&
               self.next_condition_tokens_device ==
                   other.next_condition_tokens_device &&
               self.all_drafts_accepted_flags_device ==
                   other.all_drafts_accepted_flags_device &&
               self.stopped_flags_device == other.stopped_flags_device &&
               self.next_verifier_condition_tokens_device ==
                   other.next_verifier_condition_tokens_device &&
               self.next_sidecar_condition_tokens_device ==
                   other.next_sidecar_condition_tokens_device &&
               self.next_sidecar_position_ids_device ==
                   other.next_sidecar_position_ids_device &&
               self.generation_controller_owned ==
                   other.generation_controller_owned &&
               self.generation_response_tokens_device ==
                   other.generation_response_tokens_device &&
               self.generation_response_token_stride ==
                   other.generation_response_token_stride &&
               self.generation_control_device ==
                   other.generation_control_device &&
               self.generation_control_stride ==
                   other.generation_control_stride &&
               self.verifier_input_tokens_device ==
                   other.verifier_input_tokens_device &&
               self.verifier_input_token_stride ==
                   other.verifier_input_token_stride &&
               self.committed_verifier_identity_device ==
                   other.committed_verifier_identity_device &&
               self.request_count == other.request_count &&
               self.verifier_rows_per_request ==
                   other.verifier_rows_per_request &&
               self.max_state_commit_rows ==
                   other.max_state_commit_rows &&
               self.moe_histogram_publishers ==
                   other.moe_histogram_publishers &&
               self.moe_histogram_publication_stream ==
                   other.moe_histogram_publication_stream &&
               self.main_kv_bindings == other.main_kv_bindings &&
               self.publish_shifted_kv == other.publish_shifted_kv &&
               self.shifted_kv_caches == other.shifted_kv_caches &&
               self.shifted_target_cached_tokens_device ==
                   other.shifted_target_cached_tokens_device &&
               self.shifted_accepted_state_counts_device ==
                   other.shifted_accepted_state_counts_device &&
               self.penalty_policy_device == other.penalty_policy_device &&
               self.generated_token_counts_device ==
                   other.generated_token_counts_device &&
               self.vocab_size == other.vocab_size &&
               self.verifier_state_stages ==
                   other.verifier_state_stages &&
               self.require_captured_verifier_state ==
                   other.require_captured_verifier_state &&
               self.stage_name == other.stage_name;
    }

} // namespace llaminar2
