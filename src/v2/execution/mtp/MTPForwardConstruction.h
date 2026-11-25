/**
 * @file MTPForwardConstruction.h
 * @brief Borrow original MTP construction inputs through the shared forward scope.
 *
 * This projection owns no tensor, stream, cache or request state. Its stack
 * lifetime is bounded by graph construction and each stage retains its original
 * immutable inputs. Device token/position owners exclude diagnostic host views.
 */
#pragma once
#include "models/GraphTypes.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "kernels/IKVCache.h"
#include <algorithm>

namespace llaminar2
{
/**
 * @brief Validate explicit host-owned sidecar slots before cache lookup or construction.
 * @param input Original row geometry and authoritative cache owner.
 * @throws std::invalid_argument For foreign, aliased, or out-of-capacity slots.
 */
inline void validateMTPRequestKVSlots(const MTPForwardInput &input)
{
    if (!input.request_kv_slots)
        return;
    if (!input.device.is_cpu() || !input.kv_cache || input.first_sequence_index != 0 ||
        input.batch_size <= 0 || input.request_kv_slots->size() != static_cast<std::size_t>(input.batch_size))
        throw std::invalid_argument("Explicit MTP KV slots require CPU cache ownership and exact row geometry");
    const int capacity = input.kv_cache->request_capacity();
    for (auto it = input.request_kv_slots->begin(); it != input.request_kv_slots->end(); ++it)
        if (*it < 0 || *it >= capacity || std::find(input.request_kv_slots->begin(), it, *it) != it)
            throw std::invalid_argument("Explicit MTP KV slots must be distinct admitted cache slots");
}

/**
 * @brief Project the original sidecar input for existing reusable graph builders.
 * @param input Exact MTP pointer, geometry and construction-purpose declaration.
 * @return Borrowed forward input for one scoped construction interval.
 * @throws std::invalid_argument For an unknown construction purpose or inconsistent sidecar cache role.
 */
inline ForwardInput mtpForwardConstructionInput(const MTPForwardInput &input)
{
    if (input.graph_submission_intent != ForwardGraphSubmissionIntent::WorkspaceDeclaration &&
        !executableForwardSubmissionIntent(input.graph_submission_intent))
        throw std::invalid_argument("Unknown MTP graph construction intent");
    switch (input.capture_role)
    {
    case MTPSidecarCaptureRole::Full:
    case MTPSidecarCaptureRole::Chained:
        break;
    case MTPSidecarCaptureRole::KVOnly:
        if (!input.kv_cache_only)
            throw std::invalid_argument("KV-only sidecar cache role requires its original KV-only graph purpose");
        break;
    default:
        throw std::invalid_argument("Unknown original sidecar cache role");
    }
    validateMTPRequestKVSlots(input);
    ForwardInput result;
    result.request_kv_slots = input.request_kv_slots;
    result.mtp_sidecar_capture_role = input.kv_cache_only
        ? MTPSidecarCaptureRole::KVOnly : input.capture_role;
    result.graph_submission_intent = input.graph_submission_intent;
    result.token_ids = input.draft_token_ids_device ? nullptr : input.draft_token_ids;
    result.token_ids_device = input.draft_token_ids_device;
    result.position_ids = input.position_ids_device ? nullptr : input.position_ids;
    result.position_ids_device = input.position_ids_device;
    result.batch_size = input.batch_size;
    result.seq_len = input.seq_len;
    result.device = input.device;
    result.kv_cache = input.kv_cache;
    result.sequence_lengths = input.sequence_lengths;
    result.sequence_lengths_device = input.sequence_lengths_device;
    result.device_state_publication_stream = input.device_state_publication_stream;
    result.execution_phase = input.kv_cache_only
        ? ForwardExecutionPhase::Prefill : ForwardExecutionPhase::Decode;
    return result;
}
}
