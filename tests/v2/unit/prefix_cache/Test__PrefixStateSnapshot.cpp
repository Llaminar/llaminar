/**
 * @file Test__PrefixStateSnapshot.cpp
 * @brief Device-free proofs of restorable archive boundaries and checkpoint ownership.
 *
 * Payload metadata must describe the selected frontier, not a later terminal
 * record. Boundary planning never invents a rewind of recurrent model state.
 */
#include <gtest/gtest.h>
#include <limits>

#include "execution/prefix_cache/PrefixStateSnapshot.h"
#include "execution/prefix_cache/PrefixTerminalLogitsSlice.h"

namespace llaminar2
{
namespace
{

    /** @brief Construct a small owned archive with explicit state availability. */
    PrefixBlockHandle makeBlock(int index,
                                int start,
                                int count,
                                bool includes_hybrid_payload,
                                bool hybrid_shape,
                                bool has_hybrid,
                                bool has_terminal)
    {
        PrefixBlockHandle handle;
        handle.key.fingerprint = 0x1234;
        handle.key.block_index = index;
        handle.key.token_start = start;
        handle.key.token_count = count;
        handle.layout.block_size = 4;
        handle.layout.fa_layers = 1;
        handle.layout.bytes_per_fa_layer_k = 16;
        handle.layout.bytes_per_fa_layer_v = 16;
        handle.layout.includes_hybrid_state = includes_hybrid_payload;
        handle.layout.hybrid_state_bytes = hybrid_shape ? 8 : 0;
        handle.has_hybrid_state = has_hybrid;
        handle.has_terminal_logits = has_terminal;
        handle.has_terminal_hidden = has_terminal;
        handle.total_bytes = handle.layout.totalBytes();
        handle.kv_storage = std::make_shared<std::vector<uint8_t>>(handle.layout.faKVBytes());
        handle.kv_payload = handle.kv_storage->data();
        if (includes_hybrid_payload)
        {
            handle.hybrid_storage = std::make_shared<std::vector<uint8_t>>(handle.layout.hybrid_state_bytes);
            handle.hybrid_payload = handle.hybrid_storage->data();
        }
        return handle;
    }

} // namespace

/** @brief Full and local producers select identical, possibly uneven TP intervals. */
TEST(PrefixTerminalLogitsSlice, FullAndLocalSourcesAgreeThroughTP8)
{
    for (size_t degree = 1; degree <= 8; ++degree)
        for (const size_t vocabulary : {size_t{257}, size_t{248320}})
            for (size_t rank = 0; rank < degree; ++rank)
            {
                const size_t begin = vocabulary * rank / degree;
                const size_t end = vocabulary * (rank + 1) / degree;
                const PrefixLogitsVocabularyRange shard{begin, end - begin};
                const auto full = PrefixTerminalLogitsSlice::resolve(
                    {0, vocabulary}, shard, vocabulary * sizeof(float));
                const auto local = PrefixTerminalLogitsSlice::resolve(
                    shard, shard, (shard.token_count + 16) * sizeof(float));
                EXPECT_EQ(full.byteOffset(), begin * sizeof(float));
                EXPECT_EQ(local.byteOffset(), 0u);
                EXPECT_EQ(full.byteCount(), shard.token_count * sizeof(float));
                EXPECT_EQ(local.byteCount(), full.byteCount());
            }
}

/** @brief An archive can select a strict subinterval of a nonzero-origin producer. */
TEST(PrefixTerminalLogitsSlice, NestedVocabularyIntervalExcludesPadding)
{
    const auto slice = PrefixTerminalLogitsSlice::resolve({37, 29}, {43, 11}, 64 * sizeof(float));
    EXPECT_EQ(slice.byteOffset(), 6 * sizeof(float));
    EXPECT_EQ(slice.byteCount(), 11 * sizeof(float));
}

/** @brief Misowned, missing, overflowing or physically truncated rows fail closed. */
TEST(PrefixTerminalLogitsSlice, InvalidVocabularyOrStorageIsRejected)
{
    const auto reject = [](PrefixLogitsVocabularyRange source, PrefixLogitsVocabularyRange archive,
                           size_t bytes)
    { EXPECT_THROW(PrefixTerminalLogitsSlice::resolve(source, archive, bytes), std::invalid_argument); };
    reject({0, 0}, {0, 1}, 4);
    reject({0, 4}, {0, 0}, 16);
    reject({4, 4}, {3, 1}, 16);
    reject({4, 4}, {8, 1}, 16);
    reject({4, 4}, {6, 3}, 16);
    reject({0, 4}, {0, 4}, 15);
    reject({0, 4}, {0, 4}, 12);
    constexpr size_t max = std::numeric_limits<size_t>::max();
    reject({max, 1}, {max, 1}, 4);
    reject({0, 8}, {max, 2}, 32);
    reject({0, max / sizeof(float) + 1}, {0, 1}, max - max % sizeof(float));
}

/** @brief One pre-tail frontier preserves cache/routing alignment without overflow. */
TEST(Test__PrefixStateSnapshot, ReusableCheckpointRespectsLiveAndStableBoundaries)
{
    PrefixLookupResult admission;
    admission.supported = admission.cache_enabled = true;
    admission.block_size = 64;
    admission.checkpoint_policy = PrefixCheckpointPolicy::ReusableBoundary;
    EXPECT_EQ(admission.reusablePrefillCheckpoint(365, 0), 320);
    EXPECT_EQ(admission.reusablePrefillCheckpoint(128, 0), 64);
    EXPECT_EQ(admission.reusablePrefillCheckpoint(129, 64), 128);
    EXPECT_EQ(admission.reusablePrefillCheckpoint(365, 0, 128), 256);
    EXPECT_EQ(admission.reusablePrefillCheckpoint(365, 0, 96), 192);
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(365, 320));
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(365, 365));
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(64, 0));
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(0, 0));
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(365, 0, 512));
    admission.block_size = std::numeric_limits<int>::max() - 1;
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(
        std::numeric_limits<int>::max(), 0, std::numeric_limits<int>::max()));
}

/** @brief Disabled, unsupported, and attention-only caches retain one terminal harvest. */
TEST(Test__PrefixStateSnapshot, ReusableCheckpointRequiresRecurrentCacheAdmission)
{
    PrefixLookupResult admission;
    admission.block_size = 64;
    admission.checkpoint_policy = PrefixCheckpointPolicy::ReusableBoundary;
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(365, 0));
    admission.supported = true;
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(365, 0));
    admission.cache_enabled = true;
    ASSERT_TRUE(admission.reusablePrefillCheckpoint(365, 0));
    admission.checkpoint_policy = PrefixCheckpointPolicy::TerminalOnly;
    EXPECT_FALSE(admission.reusablePrefillCheckpoint(365, 0));
}

/** @brief An earlier recurrent image owns its own terminal row when ranks clamp. */
TEST(Test__PrefixStateSnapshot, ClampedToRetainsEarlierCheckpointTerminalState)
{
    PrefixLookupResult hit;
    hit.supported = hit.cache_enabled = true;
    hit.block_size = 4;
    hit.cached_tokens = 9;
    hit.has_terminal_hidden = hit.has_terminal_logits = true;
    hit.checkpoint_policy = PrefixCheckpointPolicy::ReusableBoundary;
    hit.blocks.push_back(makeBlock(0, 0, 4, false, true, false, false));
    hit.blocks.push_back(makeBlock(1, 4, 4, true, true, true, true));
    hit.blocks.push_back(makeBlock(2, 8, 1, true, true, true, true));
    const auto clamped = hit.clampedTo(8);
    ASSERT_EQ(clamped.cached_tokens, 8);
    EXPECT_TRUE(clamped.has_terminal_hidden);
    EXPECT_TRUE(clamped.has_terminal_logits);
    EXPECT_EQ(clamped.checkpoint_policy, PrefixCheckpointPolicy::ReusableBoundary);
}

TEST(Test__PrefixStateSnapshot, ClampedToKeepsTerminalPartialBlock)
{
    PrefixLookupResult hit;
    hit.supported = true;
    hit.cache_enabled = true;
    hit.cached_tokens = 9;
    hit.block_size = 4;
    hit.has_terminal_hidden = true;
    hit.has_terminal_logits = true;
    hit.blocks.push_back(makeBlock(0, 0, 4, false, false, false, false));
    hit.blocks.push_back(makeBlock(1, 4, 4, false, false, false, false));
    hit.blocks.push_back(makeBlock(2, 8, 1, false, false, false, true));

    PrefixLookupResult clamped = hit.clampedTo(9);

    EXPECT_EQ(clamped.cached_tokens, 9);
    ASSERT_EQ(clamped.blocks.size(), 3u);
    EXPECT_EQ(clamped.blocks.back().key.token_count, 1);
    EXPECT_TRUE(clamped.has_terminal_hidden);
    EXPECT_TRUE(clamped.has_terminal_logits);
}

TEST(Test__PrefixStateSnapshot, ClampedToPreservesModelRuntimeRestorePolicy)
{
    PrefixLookupResult hit;
    hit.supported = true;
    hit.cache_enabled = true;
    hit.cached_tokens = 8;
    hit.block_size = 4;
    hit.restore_model_runtime_state = false;
    hit.restore_hybrid_state_for_suffix_prefill = true;
    hit.blocks.push_back(makeBlock(0, 0, 4, false, false, false, false));
    hit.blocks.push_back(makeBlock(1, 4, 4, false, false, false, true));

    PrefixLookupResult clamped = hit.clampedTo(4);

    EXPECT_EQ(clamped.cached_tokens, 4);
    EXPECT_FALSE(clamped.restore_model_runtime_state)
        << "Partial prefix continuations must be able to restore KV without "
           "also restoring optimization-only MoE placement runtime state.";
    EXPECT_TRUE(clamped.restore_hybrid_state_for_suffix_prefill)
        << "Partial prefix continuations must preserve the request to restore "
           "GDN state before suffix prefill.";
}

TEST(Test__PrefixStateSnapshot, ClampedToTrimsHybridBlocksWithoutRestorableState)
{
    PrefixLookupResult hit;
    hit.supported = true;
    hit.cache_enabled = true;
    hit.cached_tokens = 9;
    hit.block_size = 4;
    hit.has_terminal_hidden = true;
    hit.has_terminal_logits = true;
    hit.blocks.push_back(makeBlock(0, 0, 4, false, true, false, false));
    hit.blocks.push_back(makeBlock(1, 4, 4, false, true, false, false));
    hit.blocks.push_back(makeBlock(2, 8, 1, true, true, true, true));

    PrefixLookupResult full = hit.clampedTo(9);
    EXPECT_EQ(full.cached_tokens, 9);
    ASSERT_EQ(full.blocks.size(), 3u);
    EXPECT_TRUE(full.blocks.back().has_hybrid_state);

    PrefixLookupResult block_boundary = hit.clampedTo(8);
    EXPECT_EQ(block_boundary.cached_tokens, 0);
    EXPECT_TRUE(block_boundary.blocks.empty());
    EXPECT_FALSE(block_boundary.has_terminal_hidden);
    EXPECT_FALSE(block_boundary.has_terminal_logits);
}

/**
 * @brief Exact full hits reuse their immutable terminal archive at harvest.
 *
 * Replacing this record while the lookup still owns it can transiently spend
 * two physical RAM blocks and was observed to abort a production MPI request
 * after otherwise successful long generation. The typed decision must remain
 * conservative for partial or incomplete terminal records.
 */
TEST(Test__PrefixStateSnapshot, ExactFullHitReusesCompleteTerminalArchive)
{
    PrefixLookupResult hit;
    hit.supported = true;
    hit.cache_enabled = true;
    hit.cached_tokens = 9;
    hit.block_size = 4;
    hit.fingerprint_key = 0x1234;
    hit.requires_terminal_hidden = true;
    hit.requires_terminal_logits = true;
    hit.has_terminal_hidden = true;
    hit.has_terminal_logits = true;
    hit.blocks.push_back(makeBlock(0, 0, 4, false, false, false, false));
    hit.blocks.push_back(makeBlock(1, 4, 4, false, false, false, false));
    hit.blocks.push_back(makeBlock(2, 8, 1, true, true, true, true));

    const PrefixCacheKey terminal_key = hit.blocks.back().key;
    EXPECT_EQ(
        hit.terminalHarvestDisposition(terminal_key, 9),
        PrefixTerminalHarvestDisposition::ReuseAdmittedArchive);

    auto partial = hit;
    partial.cached_tokens = 8;
    EXPECT_EQ(
        partial.terminalHarvestDisposition(terminal_key, 9),
        PrefixTerminalHarvestDisposition::ArchiveLiveState);

    auto missing_terminal = hit;
    missing_terminal.has_terminal_logits = false;
    missing_terminal.blocks.back().has_terminal_logits = false;
    EXPECT_EQ(
        missing_terminal.terminalHarvestDisposition(terminal_key, 9),
        PrefixTerminalHarvestDisposition::ArchiveLiveState);

    auto missing_hybrid = hit;
    missing_hybrid.blocks.back().has_hybrid_state = false;
    EXPECT_EQ(
        missing_hybrid.terminalHarvestDisposition(terminal_key, 9),
        PrefixTerminalHarvestDisposition::ArchiveLiveState);

    auto foreign_key = terminal_key;
    ++foreign_key.token_hash;
    EXPECT_EQ(
        hit.terminalHarvestDisposition(foreign_key, 9),
        PrefixTerminalHarvestDisposition::ArchiveLiveState);
}

TEST(Test__PrefixStateSnapshot, ProvenanceMarksOnlyReplaySafeStatesDecodeEquivalent)
{
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::PayloadCheckpoint));
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::LogicalCheckpoint));
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::DecodeEquivalent));
    EXPECT_TRUE(isDecodeEquivalent(PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent));

    EXPECT_FALSE(isDecodeEquivalent(PrefixStateProvenance::Unknown));
    EXPECT_FALSE(isDecodeEquivalent(PrefixStateProvenance::VerifierPrefillRows));
    EXPECT_FALSE(isDecodeEquivalent(PrefixStateProvenance::SidecarDraftOnly));

    PrefixStateSnapshot snapshot;
    snapshot.valid = true;
    snapshot.provenance = PrefixStateProvenance::VerifierPrefillRows;
    EXPECT_FALSE(snapshot.decodeEquivalent());

    snapshot.provenance = PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent;
    EXPECT_TRUE(snapshot.decodeEquivalent());
}

TEST(Test__PrefixStateSnapshot, MoveLeavesSourceEmptyForNestedPayloadHandles)
{
    PrefixStateSnapshot nested;
    nested.valid = true;
    nested.logical_checkpoint = true;
    nested.provenance = PrefixStateProvenance::LogicalCheckpoint;
    nested.cached_tokens = 4;
    nested.mtp_cached_tokens = {3};
    nested.blocks.push_back(makeBlock(0, 0, 4, true, true, true, true));
    nested.mtp_blocks.push_back(makeBlock(1, 0, 3, false, false, false, false));

    PrefixStateSnapshot source;
    source.valid = true;
    source.logical_checkpoint = false;
    source.provenance = PrefixStateProvenance::PayloadCheckpoint;
    source.cached_tokens = 8;
    source.mtp_cached_tokens = {7, 6};
    DeviceKVSequenceStateCheckpoint main_sequence_state;
    main_sequence_state.cache_depth = -1;
    main_sequence_state.sequence_index = 0;
    main_sequence_state.metadata_layer_count = 2;
    main_sequence_state.bytes = 16;
    main_sequence_state.device = DeviceId::cuda(0);
    main_sequence_state.storage =
        std::make_shared<std::vector<uint8_t>>(main_sequence_state.bytes);
    main_sequence_state.ready_event =
        std::make_shared<std::vector<uint8_t>>(1);
    source.device_sequence_state_checkpoints.push_back(main_sequence_state);
    source.blocks.push_back(makeBlock(0, 0, 4, true, true, true, false));
    source.blocks.push_back(makeBlock(1, 4, 4, true, true, true, true));
    source.mtp_blocks.push_back(makeBlock(2, 0, 7, false, false, false, false));
    source.participant_snapshots.push_back(std::move(nested));

    const void *source_block_payload = source.blocks.back().hybrid_payload;
    ASSERT_NE(source_block_payload, nullptr);

    PrefixStateSnapshot moved(std::move(source));

    EXPECT_FALSE(source.valid);
    EXPECT_FALSE(source.logical_checkpoint);
    EXPECT_EQ(source.provenance, PrefixStateProvenance::Unknown);
    EXPECT_EQ(source.cached_tokens, 0);
    EXPECT_TRUE(source.mtp_cached_tokens.empty());
    EXPECT_TRUE(source.device_sequence_state_checkpoints.empty());
    EXPECT_TRUE(source.blocks.empty());
    EXPECT_TRUE(source.mtp_blocks.empty());
    EXPECT_TRUE(source.participant_snapshots.empty());

    EXPECT_TRUE(moved.valid);
    EXPECT_EQ(moved.provenance, PrefixStateProvenance::PayloadCheckpoint);
    EXPECT_EQ(moved.cached_tokens, 8);
    ASSERT_EQ(moved.device_sequence_state_checkpoints.size(), 1u);
    EXPECT_EQ(moved.device_sequence_state_checkpoints.front().cache_depth, -1);
    EXPECT_EQ(moved.device_sequence_state_checkpoints.front().sequence_index, 0);
    EXPECT_EQ(moved.device_sequence_state_checkpoints.front().metadata_layer_count, 2);
    EXPECT_EQ(moved.device_sequence_state_checkpoints.front().bytes, 16u);
    EXPECT_TRUE(moved.device_sequence_state_checkpoints.front().valid());
    EXPECT_EQ(moved.device_sequence_state_checkpoints.front().storage,
              main_sequence_state.storage);
    EXPECT_EQ(moved.device_sequence_state_checkpoints.front().ready_event,
              main_sequence_state.ready_event);
    ASSERT_EQ(moved.blocks.size(), 2u);
    EXPECT_EQ(moved.blocks.back().hybrid_payload, source_block_payload);
    ASSERT_EQ(moved.participant_snapshots.size(), 1u);
    EXPECT_TRUE(moved.participant_snapshots.front().logical_checkpoint);

    PrefixStateSnapshot assigned;
    assigned = std::move(moved);

    EXPECT_FALSE(moved.valid);
    EXPECT_TRUE(moved.device_sequence_state_checkpoints.empty());
    EXPECT_TRUE(moved.blocks.empty());
    EXPECT_TRUE(moved.participant_snapshots.empty());
    EXPECT_TRUE(assigned.valid);
    EXPECT_EQ(assigned.cached_tokens, 8);
    ASSERT_EQ(assigned.device_sequence_state_checkpoints.size(), 1u);
    EXPECT_TRUE(assigned.device_sequence_state_checkpoints.front().valid());
    ASSERT_EQ(assigned.blocks.size(), 2u);
    EXPECT_EQ(assigned.blocks.back().hybrid_payload, source_block_payload);
}

} // namespace llaminar2
