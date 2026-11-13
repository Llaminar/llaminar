/**
 * @file PrefixArchiveSourceContract.h
 * @brief Test-only source contract for typed prefix publication callback ordering.
 *
 * A callback's lexical location is not its execution location. The structural
 * guard authenticates the publication inside the once-only scope and the
 * explicit normal commit after import/mutation, retaining the existing ordering
 * contract while the runtime failure-injection tests prove unwind behavior.
 */
#pragma once
#include <cctype>
#include <string>
#include <string_view>
namespace llaminar2::test
{
/**
 * @brief Require the callback-local event edge and its normal post-mutation invocation.
 * @param body One populatePrefix method, without the following methods.
 * @return True only for one scope, callback publication and final commit ordered around imports.
 */
inline bool hasOrderedPrefixArchivePublication(std::string_view body)
{
    constexpr std::string_view scope_name = "PrefixArchivePublicationScope restore_completion(";
    constexpr std::string_view publication = "recordLivePrefixMutationReady(";
    constexpr std::string_view commit = "restore_completion.commit();";
    const auto scope = body.find(scope_name);
    if (scope == std::string_view::npos)
        return false;
    const auto callback_end = body.find("\n        });", scope);
    const auto record = body.find(publication, scope);
    const auto imports = body.find("importHybridPrefixPayload");
    const auto mutation = body.find("handleLivePrefixReplayStateAfterMutation(");
    const auto invocation = body.find(commit);
    return callback_end != std::string_view::npos &&
        record != std::string_view::npos && imports != std::string_view::npos &&
        mutation != std::string_view::npos && invocation != std::string_view::npos &&
        scope < record && record < callback_end && callback_end < imports &&
        imports < mutation && mutation < invocation &&
        body.find(scope_name, scope + scope_name.size()) == std::string_view::npos &&
        body.find(publication, record + publication.size()) == std::string_view::npos &&
        body.find(commit, invocation + commit.size()) == std::string_view::npos;
}

/**
 * @brief Compact one source slice while preserving operation labels and statement order.
 * @param body Source text containing the ordered statements to inspect.
 * @return Source text with ASCII whitespace removed.
 */
inline std::string compactPrefixArchiveSource(std::string_view body)
{
    std::string result;
    for (unsigned char byte : body)
        if (!std::isspace(byte)) result.push_back(static_cast<char>(byte));
    return result;
}
/**
 * @brief Borrow the sole restore callback containing the sole event publication, or reject it.
 * @param body Complete restore method containing the typed completion scope.
 * @return Callback source slice, or an empty view for missing, detached or duplicate publication.
 */
inline std::string_view prefixCheckpointRestoreCallback(std::string_view body)
{
    constexpr std::string_view scope = "PrefixArchivePublicationScope restore_completion(";
    constexpr std::string_view publication = "recordLivePrefixMutationReady(";
    const auto begin = body.find(scope);
    if (begin == std::string_view::npos || body.find(scope, begin + scope.size()) != std::string_view::npos)
        return {};
    const auto end = body.find("\n        });", begin);
    const auto record = body.find(publication);
    if (end == std::string_view::npos || record == std::string_view::npos ||
        record < begin || record >= end ||
        body.find(publication, record + publication.size()) != std::string_view::npos)
        return {};
    return body.substr(begin, end - begin);
}
/**
 * @brief Require main, shifted and opaque device source leases in the completion callback.
 * @param body Complete restore method containing all borrowed source owners.
 * @return True only when every owner reaches the callback-returned readiness publication.
 */
inline bool hasPrefixCheckpointRestoreSourceRetention(std::string_view body)
{
    const auto callback = compactPrefixArchiveSource(prefixCheckpointRestoreCallback(body));
    const auto record = callback.find("returnrecordLivePrefixMutationReady(stream,producer,std::move(sources),std::move(device_sources))");
    if (record == std::string::npos) return false;
    for (const auto lease : {
        "for(constauto&source:snapshot.blocks)sources.push_back(PrefixPayloadReadLease::wholeArchive(source));",
        "for(constauto&source:snapshot.mtp_blocks)sources.push_back(PrefixPayloadReadLease::wholeArchive(source));",
        "for(constauto&source:snapshot.device_sequence_state_checkpoints)device_sources.push_back(source.storage);"})
    {
        const auto found = callback.find(lease);
        if (found == std::string::npos || found >= record) return false;
    }
    return true;
}
/**
 * @brief Follow all three successful restore commits through the one typed callback.
 * @param body Complete restoreLivePrefixState method, without subsequent methods.
 * @return True only when callback creation precedes writes and each branch mutates,
 * commits its event, then publishes the corresponding shifted-cache transaction.
 */
inline bool hasOrderedPrefixCheckpointRestorePublication(std::string_view body)
{
    const auto callback = prefixCheckpointRestoreCallback(body);
    if (callback.empty()) return false;
    const auto compact = compactPrefixArchiveSource(body);
    const auto callback_text = compactPrefixArchiveSource(callback);
    if (callback_text.find("returnrecordLivePrefixMutationReady(") == std::string::npos) return false;
    if (callback_text.find("snapshot.logical_checkpoint?\"restore_logical_checkpoint\":snapshot.cached_tokens==0?\"restore_payload_checkpoint_zero\":\"restore_payload_checkpoint\"") == std::string::npos)
        return false;
    const auto scope = compact.find(callback_text);
    const auto first_write = compact.find("state_.kv_cache->truncateSequence(");
    if (first_write == std::string::npos || scope + callback_text.size() >= first_write)
        return false;
    constexpr std::string_view commit = "restore_completion.commit();";
    size_t commits = 0;
    for (auto pos = compact.find(commit); pos != std::string::npos; pos = compact.find(commit, pos + commit.size()))
        ++commits;
    if (commits != 3) return false;
    for (const char *reason : {"restore_logical_checkpoint", "restore_payload_checkpoint_zero", "restore_payload_checkpoint"})
    {
        const auto mutation = compact.find(std::string("handleLivePrefixReplayStateAfterMutation(LivePrefixMutationReason::PrefixRestore,\"") + reason + "\");");
        const auto transaction = compact.find(std::string("recordRestoredDeviceResidentMTPTransaction(stream,\"") + reason + "_shifted_kv\")");
        if (mutation == std::string::npos || transaction == std::string::npos) return false;
        const auto invocation = compact.find(commit, mutation);
        if (invocation == std::string::npos || !(first_write < mutation && mutation < invocation && invocation < transaction))
            return false;
    }
    return true;
}
} // namespace llaminar2::test
