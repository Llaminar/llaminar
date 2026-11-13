/**
 * @file Test__PrefixArchiveSourceContract.cpp
 * @brief Negative controls for callback-aware archive publication source guards.
 *
 * These fast host-only tests reject early, missing, duplicated or detached
 * completion edges. They supplement real readiness/arena failure injection;
 * source structure alone cannot certify a native asynchronous copy.
 */
#include "PrefixArchiveSourceContract.h"
#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <sstream>
namespace llaminar2::test
{
namespace
{
/** @return Minimal valid ordering with the same typed callback and normal commit as production. */
std::string orderedPublication()
{
    return R"(PrefixArchivePublicationScope restore_completion([&](PrefixArchiveCompletion) {
            return recordLivePrefixMutationReady(stream, "populate_prefix_cache", owners);
        });
importHybridPrefixPayload(cache, handle);
handleLivePrefixReplayStateAfterMutation(reason);
restore_completion.commit();
)";
}
/** @brief Replace exactly one known control token, rejecting an invalid test setup. */
void replaceOnce(std::string &source, const std::string &before, const std::string &after)
{
    const auto pos = source.find(before);
    ASSERT_NE(pos, std::string::npos);
    source.replace(pos, before.size(), after);
}
} // namespace
/** @test Typed callback definition precedes imports, while its sole invocation follows mutation. */
TEST(PrefixArchiveSourceContract, AcceptsTypedCallbackAndPostMutationCommit)
{
    EXPECT_TRUE(hasOrderedPrefixArchivePublication(orderedPublication()));
}
/** @test The real populate method satisfies the callback/commit contract used by the broader DGO guard. */
TEST(PrefixArchiveSourceContract, ProductionPopulateRetainsPublicationOrder)
{
    std::ifstream input("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    ASSERT_TRUE(input.good());
    std::ostringstream content;content << input.rdbuf();const auto source=content.str();
    const auto begin=source.find("bool DeviceGraphOrchestrator::populatePrefix(");
    ASSERT_NE(begin,std::string::npos);
    const auto end=source.find("bool DeviceGraphOrchestrator::restorePrefixTerminalState",begin);
    ASSERT_NE(end,std::string::npos);
    EXPECT_TRUE(hasOrderedPrefixArchivePublication(std::string_view(source).substr(begin,end-begin)));
}
/** @test A normal completion moved before the mutation boundary cannot certify readiness. */
TEST(PrefixArchiveSourceContract, RejectsEarlyCommit)
{
    auto source = orderedPublication();
    replaceOnce(source, "restore_completion.commit();", "");
    replaceOnce(source, "handleLivePrefixReplayStateAfterMutation", "restore_completion.commit();\nhandleLivePrefixReplayStateAfterMutation");
    EXPECT_FALSE(hasOrderedPrefixArchivePublication(source));
}
/** @test Unwind protection does not replace the explicit successful publication edge. */
TEST(PrefixArchiveSourceContract, RejectsMissingCommit)
{
    auto source = orderedPublication();
    replaceOnce(source, "restore_completion.commit();", "");
    EXPECT_FALSE(hasOrderedPrefixArchivePublication(source));
}
/** @test An empty scope with a detached direct publication has no exception-safe ownership proof. */
TEST(PrefixArchiveSourceContract, RejectsPublicationOutsideCallback)
{
    auto source = orderedPublication();
    replaceOnce(source, "return recordLivePrefixMutationReady(stream, \"populate_prefix_cache\", owners);", "return true;");
    replaceOnce(source, "restore_completion.commit();", "recordLivePrefixMutationReady(stream, name, owners);\nrestore_completion.commit();");
    EXPECT_FALSE(hasOrderedPrefixArchivePublication(source));
}
/** @test A scope created after submission leaves the first import outside unwind protection. */
TEST(PrefixArchiveSourceContract, RejectsScopeAfterImports)
{
    auto source = orderedPublication();
    replaceOnce(source, "importHybridPrefixPayload(cache, handle);", "");
    source.insert(0, "importHybridPrefixPayload(cache, handle);\n");
    EXPECT_FALSE(hasOrderedPrefixArchivePublication(source));
}
/** @test A duplicate commit or publication cannot silently replace the once-only lifecycle. */
TEST(PrefixArchiveSourceContract, RejectsDuplicateCompletionEdges)
{
    auto source = orderedPublication();source += "restore_completion.commit();\n";
    EXPECT_FALSE(hasOrderedPrefixArchivePublication(source));
    source = orderedPublication();source += "recordLivePrefixMutationReady(stream, name, owners);\n";
    EXPECT_FALSE(hasOrderedPrefixArchivePublication(source));
}

namespace
{
/** @return The complete three-branch callback/lease contract, without native execution. */
std::string checkpointRestorePublication()
{
    std::string source = R"(PrefixArchivePublicationScope restore_completion([&](PrefixArchiveCompletion) {
            for (const auto &source : snapshot.blocks)
                sources.push_back(PrefixPayloadReadLease::wholeArchive(source));
            for (const auto &source : snapshot.mtp_blocks)
                sources.push_back(PrefixPayloadReadLease::wholeArchive(source));
            for (const auto &source : snapshot.device_sequence_state_checkpoints)
                device_sources.push_back(source.storage);
            const char *producer = snapshot.logical_checkpoint ? "restore_logical_checkpoint"
                : snapshot.cached_tokens == 0 ? "restore_payload_checkpoint_zero" : "restore_payload_checkpoint";
            return recordLivePrefixMutationReady(stream, producer, std::move(sources), std::move(device_sources));
        });
state_.kv_cache->truncateSequence(seq_idx, 0, stream);
)";
    for (const char *reason : {"restore_logical_checkpoint", "restore_payload_checkpoint_zero", "restore_payload_checkpoint"})
    {
        source += std::string("handleLivePrefixReplayStateAfterMutation(LivePrefixMutationReason::PrefixRestore, \"") + reason + "\");\n";
        source += "restore_completion.commit();\n";
        source += std::string("recordRestoredDeviceResidentMTPTransaction(stream, \"") + reason + "_shifted_kv\");\n";
    }
    return source;
}
}
/** @test Each successful branch commits after its mutation and before its shifted-cache transaction. */
TEST(PrefixArchiveSourceContract, AcceptsAllCheckpointRestoreBranchesAndLeases)
{
    const auto source = checkpointRestorePublication();
    EXPECT_TRUE(hasPrefixCheckpointRestoreSourceRetention(source));
    EXPECT_TRUE(hasOrderedPrefixCheckpointRestorePublication(source));
}
/** @test Removing any exact source owner or moving it outside the callback is rejected. */
TEST(PrefixArchiveSourceContract, RejectsMissingOrDetachedCheckpointSourceLease)
{
    for (const char *owner : {"snapshot.blocks", "snapshot.mtp_blocks", "snapshot.device_sequence_state_checkpoints"})
    {
        auto source = checkpointRestorePublication();
        replaceOnce(source, owner, "unrelated_sources");
        EXPECT_FALSE(hasPrefixCheckpointRestoreSourceRetention(source)) << owner;
    }
    auto source = checkpointRestorePublication();
    replaceOnce(source, "std::move(device_sources)", "std::vector<std::shared_ptr<void>>{}");
    EXPECT_FALSE(hasPrefixCheckpointRestoreSourceRetention(source));
    source = checkpointRestorePublication();
    replaceOnce(source, "device_sources.push_back(source.storage);", "");
    source += "device_sources.push_back(source.storage);\n";
    EXPECT_FALSE(hasPrefixCheckpointRestoreSourceRetention(source));
}
/** @test Every branch independently rejects publication before its live-state mutation. */
TEST(PrefixArchiveSourceContract, RejectsEarlyCheckpointRestoreCommitInEveryBranch)
{
    for (const char *reason : {"restore_logical_checkpoint", "restore_payload_checkpoint_zero", "restore_payload_checkpoint"})
    {
        auto source = checkpointRestorePublication();
        const auto mutation = std::string("handleLivePrefixReplayStateAfterMutation(LivePrefixMutationReason::PrefixRestore, \"") + reason + "\");\n";
        replaceOnce(source, mutation + "restore_completion.commit();", "restore_completion.commit();\n" + mutation);
        EXPECT_FALSE(hasOrderedPrefixCheckpointRestorePublication(source)) << reason;
    }
}
/** @test A missing or duplicated normal commit cannot satisfy the three-branch contract. */
TEST(PrefixArchiveSourceContract, RejectsMissingOrDuplicateCheckpointRestoreCommit)
{
    auto source = checkpointRestorePublication();
    replaceOnce(source, "restore_completion.commit();", "");
    EXPECT_FALSE(hasOrderedPrefixCheckpointRestorePublication(source));
    source = checkpointRestorePublication();
    source += "restore_completion.commit();\n";
    EXPECT_FALSE(hasOrderedPrefixCheckpointRestorePublication(source));
}
/** @test A detached event record or an event-scope installed after writes loses unwind ownership. */
TEST(PrefixArchiveSourceContract, RejectsDetachedCheckpointRestorePublication)
{
    auto source = checkpointRestorePublication();
    replaceOnce(source, "recordLivePrefixMutationReady(", "unrelatedPublication(");
    source += "recordLivePrefixMutationReady(stream, producer, owners);\n";
    EXPECT_FALSE(hasOrderedPrefixCheckpointRestorePublication(source));
    EXPECT_FALSE(hasPrefixCheckpointRestoreSourceRetention(source));
    source = checkpointRestorePublication();
    replaceOnce(source, "return recordLivePrefixMutationReady(", "recordLivePrefixMutationReady(");
    EXPECT_FALSE(hasOrderedPrefixCheckpointRestorePublication(source));
    EXPECT_FALSE(hasPrefixCheckpointRestoreSourceRetention(source));
    source = checkpointRestorePublication();
    source.insert(0, "state_.kv_cache->truncateSequence(seq_idx, 0, stream);\n");
    EXPECT_FALSE(hasOrderedPrefixCheckpointRestorePublication(source));
}
/** @test The actual production restore method retains every source and all three ordered commits. */
TEST(PrefixArchiveSourceContract, ProductionCheckpointRestoreRetainsPublicationOrder)
{
    std::ifstream input("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    ASSERT_TRUE(input.good());
    std::ostringstream content;
    content << input.rdbuf();
    const auto source = content.str();
    const auto begin = source.find("bool DeviceGraphOrchestrator::restoreLivePrefixState(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("bool DeviceGraphOrchestrator::truncateLivePrefixState(", begin);
    ASSERT_NE(end, std::string::npos);
    const auto body = std::string_view(source).substr(begin, end - begin);
    EXPECT_TRUE(hasPrefixCheckpointRestoreSourceRetention(body));
    EXPECT_TRUE(hasOrderedPrefixCheckpointRestorePublication(body));
}
} // namespace llaminar2::test
