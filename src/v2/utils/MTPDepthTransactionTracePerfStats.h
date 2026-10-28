/**
 * @file MTPDepthTransactionTracePerfStats.h
 * @brief Exact finite controller-input evidence from actual runtime owners.
 *
 * GPU callers publish only after their existing terminal event; CPU callers
 * own their real host observations. One unwarmed measured request per collector
 * epoch bounds raw diagnostic keys. These records are never timing evidence.
 */
#pragma once
#include "kernels/common/MTPDepthTransactionTrace.h"
#include "execution/mtp/MTPDepthReplayBuildIdentity.h"
#include "PerfStatsCollector.h"
#include <span>
#include <string>
namespace llaminar2
{
/** @return Source hashes embedded in this compiled runtime, never restamped by a reader. */
inline PerfStatsCollector::Tags mtpDepthReplayCompiledIdentityTags()
{
    PerfStatsCollector::Tags tags{{"compiled_identity_schema", std::string(mtp_depth_replay_identity::kSchema)}};
    for (const auto &[name, hash] : mtp_depth_replay_identity::kSourceHashes)
        tags.emplace("compiled_source_" + std::string(name), std::string(hash));
    return tags;
}
/** @brief Export the actual source-authored pre-first-append controller image.
 * @param trace Complete native trace after terminal observation.
 * @param device Actual producer participant. @param identity Existing lifecycle identity.
 * @param authority Must describe the actual resident owner, never a model-state replay.
 */
inline void publishMTPDepthTransactionInitial(const sampling_math::MTPDepthTransactionTrace &trace,
    const std::string &device, const std::string &identity, const char *authority)
{
    auto tags = mtpDepthReplayCompiledIdentityTags();
    tags.emplace("diagnostic", "untimed_depth_transaction_trace_v3");
    tags.emplace("authority", authority); tags.emplace("observation_identity", identity);
    tags.emplace("phase", "validated_pre_first_response_append");
    tags.emplace("capacity", std::to_string(trace.capacity)); tags.emplace("record_words", std::to_string(trace.record_words));
    tags.emplace("schema_version", std::to_string(trace.version));
    for (int word = 0; word < sampling_math::kDeviceGenerationControlCount; ++word)
        tags.emplace("control_word_" + std::to_string(word), std::to_string(trace.initial_control[word]));
    PerfStatsCollector::addCounter("mtp", "depth_transaction_trace.initial", 1.0, "decode", device, std::move(tags));
}
/** @brief Export one packed source observation with its independently checkable decode.
 * @param record Actual immutable three-word observation.
 * @param transaction_count One-based controller ordinal, never a parallel cursor.
 * @param device Actual producer participant. @param identity Existing lifecycle identity.
 * @param authority Native resident or actual host-controller provenance.
 */
inline void publishMTPDepthTransactionRecord(const sampling_math::MTPDepthTransactionTraceRecord &record,
    int transaction_count, const std::string &device, const std::string &identity, const char *authority)
{
    sampling_math::MTPDepthTransactionTraceValues v;
    if (transaction_count <= 0 || transaction_count > sampling_math::kMTPDepthTransactionTraceCapacity ||
        !sampling_math::unpackMTPDepthTransactionTraceRecord(record, &v))
        throw std::invalid_argument("MTP transaction trace record is malformed");
    PerfStatsCollector::addCounter("mtp", "depth_transaction_trace.record", 1.0, "decode", device,
        {{"diagnostic", "untimed_depth_transaction_trace_v3"}, {"authority", authority}, {"observation_identity", identity},
         {"transaction_count", std::to_string(transaction_count)}, {"packed", std::to_string(record.packed)},
         {"pre_depth", std::to_string(v.pre_depth)}, {"accepted_prefix", std::to_string(v.accepted_prefix)},
         {"post_depth", std::to_string(v.post_depth)}, {"recommended_depth", std::to_string(v.recommended_depth)},
         {"rollback", v.rollback ? "true" : "false"}, {"budget_limited", v.budget_limited ? "true" : "false"},
         {"incoming_debt", v.incoming_debt ? "true" : "false"},
         {"outgoing_debt", v.outgoing_debt ? "true" : "false"},
         {"model_stopped", v.model_stopped ? "true" : "false"},
         {"host_observation", v.host_observation ? "true" : "false"},
         {"requested_depth", std::to_string(v.requested_depth)}, {"effective_depth", std::to_string(v.effective_depth)},
         {"token_witness", std::to_string(v.token_witness)},
         {"token_witness_scope", v.host_observation ? "cpu_transaction_emitted" : "native_committed_response"},
         {"newly_emitted_tokens", v.host_observation ? std::to_string(v.token_witness) : "-1"},
         {"newly_emitted_tokens_valid", v.host_observation ? "true" : "false"},
         {"response_token_count", v.host_observation ? "-1" : std::to_string(v.token_witness)},
         {"response_token_count_valid", v.host_observation ? "false" : "true"},
         {"response_token_count_scope", v.host_observation ? "unavailable_cpu_host" : "native_committed_response"},
         {"evaluated_windows", std::to_string(v.evaluated_windows)}});
}
/** @brief Export the existing actual terminal image, without another GPU read.
 * @param control Already authenticated terminal control from the existing D2H.
 * @param device Actual producer participant. @param identity Existing lifecycle identity.
 * @param authority Actual resident owner provenance.
 */
inline void publishMTPDepthTransactionFinal(std::span<const int, sampling_math::kDeviceGenerationControlCount> control,
    const std::string &device, const std::string &identity, const char *authority)
{
    PerfStatsCollector::Tags tags{{"diagnostic", "untimed_depth_transaction_trace_v3"},
        {"authority", authority}, {"observation_identity", identity},
        {"transactions", std::to_string(control[sampling_math::kDeviceGenerationControlTransactionCount])},
        {"published_state_tokens", std::to_string(control[sampling_math::kDeviceGenerationControlPublishedStateCommitCount])},
        {"emitted_tokens", std::to_string(control[sampling_math::kDeviceGenerationControlResponseTokenCount])},
        {"model_stopped", control[sampling_math::kDeviceGenerationControlModelStopped] != 0 ? "true" : "false"},
        {"replay_projection", "controller_owned_fields_only"}};
    for (int word = 0; word < sampling_math::kDeviceGenerationControlCount; ++word)
        tags.emplace("control_word_" + std::to_string(word), std::to_string(control[word]));
    PerfStatsCollector::addCounter("mtp", "depth_transaction_trace.final", 1.0, "decode", device, std::move(tags));
}
} // namespace llaminar2
