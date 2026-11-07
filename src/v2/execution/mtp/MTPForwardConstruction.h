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

namespace llaminar2
{
/**
 * @brief Project the original sidecar input for existing reusable graph builders.
 * @param input Exact MTP pointer, geometry and construction-purpose declaration.
 * @return Borrowed forward input for one scoped construction interval.
 * @throws std::invalid_argument If the original construction purpose is unknown.
 */
inline ForwardInput mtpForwardConstructionInput(const MTPForwardInput &input)
{
    if (input.graph_submission_intent != ForwardGraphSubmissionIntent::WorkspaceDeclaration &&
        !executableForwardSubmissionIntent(input.graph_submission_intent))
        throw std::invalid_argument("Unknown MTP graph construction intent");
    ForwardInput result;
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
