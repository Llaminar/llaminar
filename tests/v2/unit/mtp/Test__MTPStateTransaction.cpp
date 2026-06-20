#include <gtest/gtest.h>

#include "execution/mtp/MTPStateTransaction.h"

namespace llaminar2
{
namespace
{

    MTPDecodeStateStamp makeState(
        int logical_tokens,
        PrefixStateProvenance provenance = PrefixStateProvenance::DecodeEquivalent)
    {
        MTPDecodeStateStamp state;
        state.valid = true;
        state.logical_tokens = logical_tokens;
        state.main_kv_tokens = logical_tokens;
        state.shifted_mtp_kv_tokens = expectedShiftedMTPTokens(logical_tokens);
        state.position = logical_tokens;
        state.has_terminal_hidden = true;
        state.has_terminal_logits = true;
        state.has_ready_token = true;
        state.provenance = provenance;
        return state;
    }

    PrefixRuntimeStateSnapshot makeRuntimeSnapshot(int logical_tokens)
    {
        PrefixRuntimeStateSnapshot snapshot;
        snapshot.initialized = true;
        snapshot.current_position = logical_tokens;
        snapshot.has_hidden = true;
        snapshot.has_logits = true;
        snapshot.positions = {logical_tokens};
        snapshot.sequence_lengths = {logical_tokens};

        PrefixKVCacheProbe main_kv;
        main_kv.owner = "main";
        PrefixKVLayerProbe main_layer;
        main_layer.cache_layer = 0;
        main_layer.global_layer = 0;
        main_layer.seq_idx = 0;
        main_layer.cached_tokens = logical_tokens;
        main_layer.ring_head = 0;
        main_layer.payload_hash_available = true;
        main_layer.k_payload_bytes = 128;
        main_layer.v_payload_bytes = 128;
        main_layer.k_payload_hash = 0xaaaa;
        main_layer.v_payload_hash = 0xbbbb;
        main_kv.layers.push_back(main_layer);
        snapshot.kv_caches.push_back(main_kv);

        PrefixKVCacheProbe mtp_kv;
        mtp_kv.owner = "mtp";
        PrefixKVLayerProbe mtp_layer;
        mtp_layer.cache_layer = 0;
        mtp_layer.global_layer = 0;
        mtp_layer.seq_idx = 0;
        mtp_layer.cached_tokens = expectedShiftedMTPTokens(logical_tokens);
        mtp_layer.ring_head = 0;
        mtp_layer.payload_hash_available = true;
        mtp_layer.k_payload_bytes = 64;
        mtp_layer.v_payload_bytes = 64;
        mtp_layer.k_payload_hash = 0xcccc;
        mtp_layer.v_payload_hash = 0xdddd;
        mtp_kv.layers.push_back(mtp_layer);
        snapshot.mtp_kv_caches.push_back(mtp_kv);

        PrefixGDNLayerProbe gdn;
        gdn.global_layer = 4;
        gdn.recurrence_values = 64;
        gdn.conv_values = 12;
        gdn.recurrence_hash = 0x1234;
        gdn.conv_hash = 0x5678;
        gdn.recurrence_all_zero = false;
        gdn.conv_all_zero = false;
        gdn.recurrence_sample_values = {1.0f, 2.0f, 3.0f};
        gdn.conv_sample_values = {4.0f, 5.0f};
        snapshot.gdn_layers.push_back(gdn);
        return snapshot;
    }

} // namespace

TEST(Test__MTPStateTransaction, ExpectedShiftedTokensLagMainByOne)
{
    EXPECT_EQ(expectedShiftedMTPTokens(0), 0);
    EXPECT_EQ(expectedShiftedMTPTokens(1), 0);
    EXPECT_EQ(expectedShiftedMTPTokens(2), 1);
    EXPECT_EQ(expectedShiftedMTPTokens(9), 8);
}

TEST(Test__MTPStateTransaction, LogicalVerifierBaseSnapshotCarriesDecodeEquivalentTokenCounts)
{
    const PrefixStateSnapshot snapshot =
        makeLogicalMTPVerifierBaseSnapshot(/*cached_tokens=*/7);

    EXPECT_TRUE(snapshot.valid);
    EXPECT_TRUE(snapshot.logical_checkpoint);
    EXPECT_EQ(snapshot.provenance, PrefixStateProvenance::LogicalCheckpoint);
    EXPECT_EQ(snapshot.cached_tokens, 7);
    ASSERT_EQ(snapshot.mtp_cached_tokens.size(), 1u);
    EXPECT_EQ(snapshot.mtp_cached_tokens.front(), 6);
    EXPECT_TRUE(snapshot.blocks.empty())
        << "Logical verifier-base snapshots must not smuggle payload blocks "
           "back into the steady MTP path.";
    EXPECT_TRUE(snapshot.mtp_blocks.empty());
    EXPECT_TRUE(isDecodeEquivalent(snapshot.provenance));
}

TEST(Test__MTPStateTransaction, LogicalVerifierBaseSnapshotRejectsNegativePositions)
{
    const PrefixStateSnapshot snapshot =
        makeLogicalMTPVerifierBaseSnapshot(/*cached_tokens=*/-1);

    EXPECT_FALSE(snapshot.valid);
    EXPECT_TRUE(snapshot.logical_checkpoint);
    EXPECT_EQ(snapshot.provenance, PrefixStateProvenance::LogicalCheckpoint);
}

TEST(Test__MTPStateTransaction, CommittedDecodeStateRequiresConsistentCounts)
{
    MTPDecodeStateStamp state = makeState(5);
    EXPECT_TRUE(validateCommittedMTPDecodeState(state));

    state.main_kv_tokens = 4;
    auto result = validateCommittedMTPDecodeState(state);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("main KV"), std::string::npos);

    state = makeState(5);
    state.shifted_mtp_kv_tokens = 5;
    result = validateCommittedMTPDecodeState(state);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("shifted MTP KV"), std::string::npos);

    state = makeState(5);
    state.position = 4;
    result = validateCommittedMTPDecodeState(state);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("position"), std::string::npos);
}

TEST(Test__MTPStateTransaction, UnsafeVerifierPrefillRowsCannotCommit)
{
    MTPDecodeStateStamp base = makeState(5);
    MTPDecodeStateStamp committed = makeState(7);

    auto result = validateAtomicMTPCommit(
        base,
        committed,
        /*emitted_tokens=*/2,
        PrefixStateProvenance::VerifierPrefillRows);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("verifier source"), std::string::npos);

    result = validateAtomicMTPCommit(
        base,
        committed,
        /*emitted_tokens=*/2,
        PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent);
    EXPECT_TRUE(result) << result.reason;
}

TEST(Test__MTPStateTransaction, AtomicCommitRequiresBasePlusEmittedTokens)
{
    MTPDecodeStateStamp base = makeState(5);
    MTPDecodeStateStamp committed = makeState(8);

    auto result = validateAtomicMTPCommit(
        base,
        committed,
        /*emitted_tokens=*/2,
        PrefixStateProvenance::DecodeEquivalent);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("base plus emitted"), std::string::npos);
}

TEST(Test__MTPStateTransaction, AtomicCommitCanAllowOnlyBaseShiftedKVLag)
{
    MTPDecodeStateStamp base = makeState(5);
    base.shifted_mtp_kv_tokens = 0;
    MTPDecodeStateStamp committed = makeState(6);

    MTPCommitValidationOptions options;
    options.require_base_shifted_mtp_kv = false;
    options.require_committed_shifted_mtp_kv = true;

    auto result = validateAtomicMTPCommit(
        base,
        committed,
        /*emitted_tokens=*/1,
        PrefixStateProvenance::DecodeEquivalent,
        options);
    EXPECT_TRUE(result) << result.reason;

    committed.shifted_mtp_kv_tokens = 0;
    result = validateAtomicMTPCommit(
        base,
        committed,
        /*emitted_tokens=*/1,
        PrefixStateProvenance::DecodeEquivalent,
        options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("shifted MTP KV"), std::string::npos);
}

TEST(Test__MTPStateTransaction, AtomicCommitRequiresTerminalReadyState)
{
    MTPDecodeStateStamp base = makeState(5);
    MTPDecodeStateStamp committed = makeState(6);
    committed.has_terminal_hidden = false;

    auto result = validateAtomicMTPCommit(
        base,
        committed,
        /*emitted_tokens=*/1,
        PrefixStateProvenance::DecodeEquivalent);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("terminal hidden"), std::string::npos);
}

TEST(Test__MTPStateTransaction, RuntimeSnapshotEquivalenceAcceptsMatchingState)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    EXPECT_TRUE(result) << result.reason;
}

TEST(Test__MTPStateTransaction, RuntimeSnapshotEquivalenceRejectsShiftedKVDrift)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    candidate.mtp_kv_caches.front().layers.front().cached_tokens -= 1;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("shifted MTP"), std::string::npos);
}

TEST(Test__MTPStateTransaction, RuntimeSnapshotSerialOracleCanIgnoreShiftedMTPKV)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    candidate.mtp_kv_caches.front().layers.front().cached_tokens += 2;

    MTPRuntimeSnapshotComparisonOptions options;
    options.compare_shifted_mtp_kv = false;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    EXPECT_TRUE(result) << result.reason;
}

TEST(Test__MTPStateTransaction, RuntimeSnapshotSerialOracleCanIgnoreMainKVPayloadHashes)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    candidate.kv_caches.front().layers.front().k_payload_hash ^= 0x1;
    candidate.kv_caches.front().layers.front().v_payload_hash ^= 0x2;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("main KV payload hash"), std::string::npos);

    MTPRuntimeSnapshotComparisonOptions options;
    options.compare_main_kv_payload_hashes = false;
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    EXPECT_TRUE(result) << result.reason;

    candidate.kv_caches.front().layers.front().cached_tokens -= 1;
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("main KV cached token count"), std::string::npos);
}

TEST(Test__MTPStateTransaction, RuntimeSnapshotEquivalenceRejectsGDNHashDrift)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    candidate.gdn_layers.front().recurrence_hash ^= 0x1;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("GDN recurrence hash"), std::string::npos);
}

TEST(Test__MTPStateTransaction, RuntimeSnapshotPrefersDeviceOwnedGDNHashesWhenAvailable)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;

    auto &oracle_gdn = oracle.gdn_layers.front();
    auto &candidate_gdn = candidate.gdn_layers.front();
    oracle_gdn.device_state_hash_available = true;
    candidate_gdn.device_state_hash_available = true;
    oracle_gdn.recurrence_device_bytes = 256;
    candidate_gdn.recurrence_device_bytes = 256;
    oracle_gdn.conv_device_bytes = 64;
    candidate_gdn.conv_device_bytes = 64;
    oracle_gdn.recurrence_device_hash = 0xaaaa1111;
    candidate_gdn.recurrence_device_hash = 0xaaaa1111;
    oracle_gdn.conv_device_hash = 0xbbbb2222;
    candidate_gdn.conv_device_hash = 0xbbbb2222;

    /*
     * GPU GDN/short-conv publication is device-owned.  The hybrid cache host
     * mirror can legitimately lag the device state, so stale host hashes and
     * zero flags must not make a device-owned snapshot look invalid.
     */
    candidate_gdn.recurrence_hash ^= 0x1;
    candidate_gdn.conv_hash ^= 0x2;
    candidate_gdn.recurrence_all_zero = !oracle_gdn.recurrence_all_zero;
    candidate_gdn.conv_all_zero = !oracle_gdn.conv_all_zero;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    EXPECT_TRUE(result) << result.reason;

    candidate_gdn.recurrence_device_hash ^= 0x4;
    result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("GDN recurrence device hash"), std::string::npos);
}

TEST(Test__MTPStateTransaction, RuntimeSnapshotCanUseToleranceAwareGDNValues)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    candidate.gdn_layers.front().recurrence_hash ^= 0x1;
    candidate.gdn_layers.front().recurrence_sample_values[1] += 1e-7f;

    MTPRuntimeSnapshotComparisonOptions options;
    options.compare_gdn_hashes = false;
    options.compare_gdn_values_if_available = true;
    options.gdn_relative_l2_tolerance = 1e-5;
    options.gdn_max_abs_tolerance = 1e-5;
    options.gdn_min_cosine = 0.999999;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    EXPECT_TRUE(result) << result.reason;

    candidate.gdn_layers.front().recurrence_sample_values[1] += 1e-2f;
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("GDN recurrence value mismatch"), std::string::npos);
}

} // namespace llaminar2
