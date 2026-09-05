/**
 * @file Test__MTPStateTransaction.cpp
 * @brief Adversarial unit proofs for MTP and prefix-replay state transactions.
 *
 * These tests keep the persistent-state comparison rules independent of a
 * model or accelerator.  In particular, they prove which bytes must remain
 * exact across prefix restore and which bounded floating-point payload may use
 * numerical equivalence after an authenticated ExpertOverlay placement change.
 */

#include <gtest/gtest.h>

#include "execution/mtp/MTPStateTransaction.h"
#include "tensors/FP16Utils.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

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
        snapshot.terminal_hidden_hash_available = true;
        snapshot.terminal_hidden_bytes = 256;
        snapshot.terminal_hidden_hash = 0x11112222;
        snapshot.terminal_logits_hash_available = true;
        snapshot.terminal_logits_bytes = 512;
        snapshot.terminal_logits_hash = 0x33334444;
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
        gdn.recurrence_sample_values.resize(gdn.recurrence_values);
        gdn.conv_sample_values.resize(gdn.conv_values);
        for (size_t index = 0;
             index < gdn.recurrence_sample_values.size();
             ++index)
        {
            gdn.recurrence_sample_values[index] =
                1.0f + static_cast<float>(index) * 0.01f;
        }
        for (size_t index = 0;
             index < gdn.conv_sample_values.size();
             ++index)
        {
            gdn.conv_sample_values[index] =
                4.0f + static_cast<float>(index) * 0.01f;
        }
        snapshot.gdn_layers.push_back(gdn);
        return snapshot;
    }

    /**
     * @brief Encode logical floating values exactly as one KV precision stores them.
     * @param values Source values used by the state-machine proof.
     * @param precision Native KV payload precision.
     * @return Canonical byte sequence consumed by the production comparator.
     */
    std::vector<uint8_t> encodeKVValues(
        std::initializer_list<float> values,
        ActivationPrecision precision)
    {
        const size_t element_bytes =
            precision == ActivationPrecision::FP32 ? sizeof(float)
                                                   : sizeof(uint16_t);
        std::vector<uint8_t> payload(values.size() * element_bytes);
        size_t index = 0;
        for (float value : values)
        {
            if (precision == ActivationPrecision::FP32)
            {
                std::memcpy(
                    payload.data() + index * sizeof(float),
                    &value,
                    sizeof(value));
            }
            else
            {
                uint16_t word = 0;
                if (precision == ActivationPrecision::FP16)
                {
                    word = fp32_to_fp16(value);
                }
                else
                {
                    uint32_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    word = static_cast<uint16_t>(bits >> 16u);
                }
                std::memcpy(
                    payload.data() + index * sizeof(uint16_t),
                    &word,
                    sizeof(word));
            }
            ++index;
        }
        return payload;
    }

    /**
     * @brief Install exact-prefix and retained-suffix evidence on a snapshot.
     * @param snapshot Snapshot whose one main-KV layer receives the evidence.
     * @param precision Native K/V precision under test.
     * @param k_payload Complete logical values for the one-token K suffix.
     * @param v_payload Complete logical values for the one-token V suffix.
     */
    void installPartialPrefixKVEvidence(
        PrefixRuntimeStateSnapshot *snapshot,
        ActivationPrecision precision,
        std::vector<uint8_t> k_payload,
        std::vector<uint8_t> v_payload)
    {
        ASSERT_NE(snapshot, nullptr);
        PrefixKVCacheProbe &cache = snapshot->kv_caches.front();
        PrefixKVLayerProbe &layer = cache.layers.front();
        cache.k_precision = precision;
        cache.v_precision = precision;

        const size_t row_k_bytes = k_payload.size();
        const size_t row_v_bytes = v_payload.size();
        layer.k_payload_bytes = 7u * row_k_bytes;
        layer.v_payload_bytes = 7u * row_v_bytes;
        layer.leading_segment_hash_available = true;
        layer.leading_segment_tokens = 6;
        layer.leading_k_payload_bytes = 6u * row_k_bytes;
        layer.leading_v_payload_bytes = 6u * row_v_bytes;
        layer.leading_k_payload_hash = 0xabc001;
        layer.leading_v_payload_hash = 0xabc002;

        PrefixKVSegmentProbe suffix;
        suffix.name = "recomputed_suffix";
        suffix.token_start = 6;
        suffix.token_count = 1;
        suffix.hash_available = true;
        suffix.k_payload_bytes = row_k_bytes;
        suffix.v_payload_bytes = row_v_bytes;
        suffix.k_payload_hash = 0xdef001;
        suffix.v_payload_hash = 0xdef002;
        suffix.k_payload = std::move(k_payload);
        suffix.v_payload = std::move(v_payload);
        layer.segments = {std::move(suffix)};
    }

} // namespace

TEST(Test__MTPStateTransaction, ExpectedShiftedTokensLagMainByOne)
{
    EXPECT_EQ(expectedShiftedMTPTokens(0), 0);
    EXPECT_EQ(expectedShiftedMTPTokens(1), 0);
    EXPECT_EQ(expectedShiftedMTPTokens(2), 1);
    EXPECT_EQ(expectedShiftedMTPTokens(9), 8);
}

TEST(Test__MTPStateTransaction,
     ShiftedReplayReusesVerifierBaseRowThenAppendsAfterMainAdvances)
{
    const MTPShiftedReplayRowPlan base =
        planMTPShiftedReplayRow(
            /*main_cached_tokens=*/596,
            /*shifted_cached_tokens=*/596);
    ASSERT_TRUE(base) << base.reason;
    EXPECT_EQ(
        base.action,
        MTPShiftedReplayRowAction::ReuseResidentRow);
    EXPECT_EQ(base.append_position_offset, -1);

    const MTPShiftedReplayRowPlan after_main_advance =
        planMTPShiftedReplayRow(
            /*main_cached_tokens=*/597,
            /*shifted_cached_tokens=*/596);
    ASSERT_TRUE(after_main_advance) << after_main_advance.reason;
    EXPECT_EQ(
        after_main_advance.action,
        MTPShiftedReplayRowAction::AppendRow);
    EXPECT_EQ(after_main_advance.append_position_offset, 597);
}

TEST(Test__MTPStateTransaction,
     ShiftedReplayRejectsAmbiguousOrCorruptLifecycleShapes)
{
    for (const auto [main_tokens, shifted_tokens] :
         std::vector<std::pair<int, int>>{
             {-1, 0},
             {0, -1},
             {598, 596},
             {596, 597}})
    {
        const MTPShiftedReplayRowPlan plan =
            planMTPShiftedReplayRow(main_tokens, shifted_tokens);
        EXPECT_FALSE(plan)
            << "main=" << main_tokens
            << " shifted=" << shifted_tokens;
        EXPECT_FALSE(plan.reason.empty());
    }
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

TEST(Test__MTPStateTransaction, RuntimeSnapshotEquivalenceRejectsTerminalPayloadDrift)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    candidate.terminal_logits_hash ^= 0x1;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("terminal logits payload"), std::string::npos);

    candidate = oracle;
    candidate.terminal_hidden_bytes += 4;
    result = compareMTPRuntimeStateSnapshots(oracle, candidate);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("terminal hidden payload"), std::string::npos);
}

TEST(Test__MTPStateTransaction,
     RuntimeSnapshotPlacementAwareTerminalHiddenStillRequiresExactSameEpoch)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 4;
    oracle.terminal_hidden_values.assign(64, 1.0f);
    candidate.terminal_hidden_values = oracle.terminal_hidden_values;
    candidate.terminal_hidden_values.front() += 1.0e-4f;
    candidate.terminal_hidden_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.terminal_hidden_policy =
        MTPTerminalPayloadComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_FALSE(result.terminal_hidden_numerical.compared);
    EXPECT_NE(result.reason.find("moe_movement_epoch=4/4"), std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     RuntimeSnapshotPlacementAwareTerminalHiddenAcceptsFullNumericalEvidence)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 5;
    oracle.terminal_hidden_values.assign(64, 1.0f);
    candidate.terminal_hidden_values = oracle.terminal_hidden_values;
    candidate.terminal_hidden_values.front() += 1.0e-3f;
    candidate.terminal_hidden_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.terminal_hidden_policy =
        MTPTerminalPayloadComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    options.terminal_hidden_min_cosine =
        kDefaultPlacementAwareTerminalPayloadMinimumCosine;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_TRUE(result) << result.reason;
    EXPECT_TRUE(result.terminal_hidden_numerical.compared);
    EXPECT_TRUE(result.terminal_hidden_numerical.passed);
    EXPECT_EQ(result.terminal_hidden_numerical.elements, 64u);
    EXPECT_GE(
        result.terminal_hidden_numerical.cosine,
        kDefaultPlacementAwareTerminalPayloadMinimumCosine);
    EXPECT_GT(result.terminal_hidden_numerical.max_abs, 0.0);
}

TEST(Test__MTPStateTransaction,
     RuntimeSnapshotPlacementAwareTerminalHiddenRejectsMissingOrBadEvidence)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 5;
    candidate.terminal_hidden_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.terminal_hidden_policy =
        MTPTerminalPayloadComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    options.terminal_hidden_min_cosine =
        kDefaultPlacementAwareTerminalPayloadMinimumCosine;
    auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("complete retained values"), std::string::npos)
        << result.reason;

    oracle.terminal_hidden_values.assign(64, 1.0f);
    candidate.terminal_hidden_values.assign(64, -1.0f);
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.terminal_hidden_numerical.compared);
    EXPECT_FALSE(result.terminal_hidden_numerical.passed);
    EXPECT_NE(result.reason.find("numerical mismatch"), std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     RuntimeSnapshotPlacementAwareTerminalLogitsStillRequiresExactSameEpoch)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 9;
    candidate.moe_runtime_movement_epoch = 9;
    oracle.terminal_logits_values.assign(128, 1.0f);
    candidate.terminal_logits_values = oracle.terminal_logits_values;
    candidate.terminal_logits_values.front() += 1.0e-4f;
    candidate.terminal_logits_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.terminal_logits_policy =
        MTPTerminalPayloadComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_FALSE(result.terminal_logits_numerical.compared);
    EXPECT_NE(result.reason.find("moe_movement_epoch=9/9"), std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     RuntimeSnapshotPlacementAwareTerminalLogitsAcceptsFullNumericalEvidence)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 9;
    candidate.moe_runtime_movement_epoch = 10;
    oracle.terminal_logits_values.assign(128, 1.0f);
    candidate.terminal_logits_values = oracle.terminal_logits_values;
    candidate.terminal_logits_values.front() += 1.0e-3f;
    candidate.terminal_logits_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.terminal_logits_policy =
        MTPTerminalPayloadComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    options.terminal_logits_min_cosine =
        kDefaultPlacementAwareTerminalPayloadMinimumCosine;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_TRUE(result) << result.reason;
    EXPECT_TRUE(result.terminal_logits_numerical.compared);
    EXPECT_TRUE(result.terminal_logits_numerical.passed);
    EXPECT_EQ(result.terminal_logits_numerical.elements, 128u);
    EXPECT_GE(
        result.terminal_logits_numerical.cosine,
        kDefaultPlacementAwareTerminalPayloadMinimumCosine);
    EXPECT_GT(result.terminal_logits_numerical.max_abs, 0.0);
}

TEST(Test__MTPStateTransaction,
     RuntimeSnapshotPlacementAwareTerminalLogitsRejectsMissingOrBadEvidence)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 9;
    candidate.moe_runtime_movement_epoch = 10;
    candidate.terminal_logits_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.terminal_logits_policy =
        MTPTerminalPayloadComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    options.terminal_logits_min_cosine =
        kDefaultPlacementAwareTerminalPayloadMinimumCosine;
    auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("complete retained values"), std::string::npos)
        << result.reason;

    oracle.terminal_logits_values.assign(128, 1.0f);
    candidate.terminal_logits_values.assign(128, -1.0f);
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.terminal_logits_numerical.compared);
    EXPECT_FALSE(result.terminal_logits_numerical.passed);
    EXPECT_NE(result.reason.find("numerical mismatch"), std::string::npos)
        << result.reason;
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
    EXPECT_NE(
        result.reason.find("main KV layer metadata mismatch"),
        std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     PlacementAwareMainKVStillRequiresExactBytesWithoutMovement)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    installPartialPrefixKVEvidence(
        &oracle,
        ActivationPrecision::FP32,
        encodeKVValues({1.0f, 2.0f, 3.0f, 4.0f}, ActivationPrecision::FP32),
        encodeKVValues({5.0f, 6.0f, 7.0f, 8.0f}, ActivationPrecision::FP32));
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 4;
    candidate.kv_caches.front().layers.front().k_payload_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.main_kv_payload_policy =
        MTPMainKVPayloadComparisonPolicy::
            ExactPrefixNumericalSuffixAfterMoEPlacementChange;
    options.main_kv_exact_prefix_tokens = 6;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);

    ASSERT_FALSE(result);
    EXPECT_FALSE(result.main_kv_numerical.compared);
    EXPECT_NE(result.reason.find("main KV payload hash mismatch"), std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     PlacementAwareMainKVAcceptsExactPrefixAndNumericalSuffixForAllFloatingFormats)
{
    for (const ActivationPrecision precision : {
             ActivationPrecision::FP16,
             ActivationPrecision::BF16,
             ActivationPrecision::FP32})
    {
        PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
        installPartialPrefixKVEvidence(
            &oracle,
            precision,
            encodeKVValues({1.0f, 2.0f, 3.0f, 4.0f}, precision),
            encodeKVValues({5.0f, 6.0f, 7.0f, 8.0f}, precision));
        PrefixRuntimeStateSnapshot candidate = oracle;
        installPartialPrefixKVEvidence(
            &candidate,
            precision,
            encodeKVValues({1.1f, 2.0f, 3.0f, 4.0f}, precision),
            encodeKVValues({5.1f, 6.0f, 7.0f, 8.0f}, precision));
        oracle.moe_runtime_movement_epoch = 4;
        candidate.moe_runtime_movement_epoch = 5;
        candidate.kv_caches.front().layers.front().k_payload_hash ^= 0x1;
        candidate.kv_caches.front().layers.front().v_payload_hash ^= 0x2;
        candidate.kv_caches.front().layers.front()
            .segments.front().k_payload_hash ^= 0x1;
        candidate.kv_caches.front().layers.front()
            .segments.front().v_payload_hash ^= 0x2;

        MTPRuntimeSnapshotComparisonOptions options;
        options.main_kv_payload_policy =
            MTPMainKVPayloadComparisonPolicy::
                ExactPrefixNumericalSuffixAfterMoEPlacementChange;
        options.main_kv_exact_prefix_tokens = 6;
        options.main_kv_suffix_min_cosine = 0.99;
        const auto result =
            compareMTPRuntimeStateSnapshots(oracle, candidate, options);

        ASSERT_TRUE(result)
            << "precision=" << activationPrecisionToString(precision)
            << " reason=" << result.reason;
        EXPECT_TRUE(result.main_kv_numerical.compared);
        EXPECT_TRUE(result.main_kv_numerical.passed);
        EXPECT_EQ(result.main_kv_numerical.exact_prefix_segments, 1u);
        EXPECT_EQ(result.main_kv_numerical.numerical_suffix_payloads, 2u);
        EXPECT_EQ(result.main_kv_numerical.elements, 8u);
        EXPECT_GE(result.main_kv_numerical.minimum_cosine, 0.99);
        EXPECT_GT(result.main_kv_numerical.maximum_abs, 0.0);
    }
}

TEST(Test__MTPStateTransaction,
     PlacementAwareMainKVRejectsAnyCachedPrefixByteDrift)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    installPartialPrefixKVEvidence(
        &oracle,
        ActivationPrecision::FP32,
        encodeKVValues({1.0f, 2.0f, 3.0f, 4.0f}, ActivationPrecision::FP32),
        encodeKVValues({5.0f, 6.0f, 7.0f, 8.0f}, ActivationPrecision::FP32));
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 5;
    candidate.kv_caches.front().layers.front().k_payload_hash ^= 0x1;
    candidate.kv_caches.front().layers.front().leading_k_payload_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.main_kv_payload_policy =
        MTPMainKVPayloadComparisonPolicy::
            ExactPrefixNumericalSuffixAfterMoEPlacementChange;
    options.main_kv_exact_prefix_tokens = 6;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);

    ASSERT_FALSE(result);
    EXPECT_FALSE(result.main_kv_numerical.compared);
    EXPECT_NE(result.reason.find("cached-prefix bytes changed"), std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     PlacementAwareMainKVRejectsMissingOrNumericallyBadSuffixEvidence)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    installPartialPrefixKVEvidence(
        &oracle,
        ActivationPrecision::FP32,
        encodeKVValues({1.0f, 2.0f, 3.0f, 4.0f}, ActivationPrecision::FP32),
        encodeKVValues({5.0f, 6.0f, 7.0f, 8.0f}, ActivationPrecision::FP32));
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 5;
    candidate.kv_caches.front().layers.front().k_payload_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.main_kv_payload_policy =
        MTPMainKVPayloadComparisonPolicy::
            ExactPrefixNumericalSuffixAfterMoEPlacementChange;
    options.main_kv_exact_prefix_tokens = 6;
    options.main_kv_suffix_min_cosine = 0.99;

    candidate.kv_caches.front().layers.front()
        .segments.front().k_payload.clear();
    auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.reason.find("complete retained suffix bytes"), std::string::npos)
        << result.reason;

    candidate = oracle;
    candidate.moe_runtime_movement_epoch = 5;
    candidate.kv_caches.front().layers.front().k_payload_hash ^= 0x1;
    PrefixKVSegmentProbe &bad_suffix =
        candidate.kv_caches.front().layers.front().segments.front();
    bad_suffix.k_payload = encodeKVValues(
        {-1.0f, -2.0f, -3.0f, -4.0f},
        ActivationPrecision::FP32);
    bad_suffix.k_payload_hash ^= 0x1;
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.main_kv_numerical.compared);
    EXPECT_FALSE(result.main_kv_numerical.passed);
    EXPECT_NE(result.reason.find("suffix numerical mismatch"), std::string::npos)
        << result.reason;
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
    options.gdn_state_policy =
        MTPGDNStateComparisonPolicy::NumericalValues;
    options.gdn_relative_l2_tolerance = 1e-5;
    options.gdn_max_abs_tolerance = 1e-5;
    options.gdn_min_cosine = 0.999999;

    auto result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    EXPECT_TRUE(result) << result.reason;

    candidate.gdn_layers.front().recurrence_sample_values[1] += 1e-2f;
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_NE(
        result.reason.find("GDN recurrence numerical mismatch"),
        std::string::npos);
}

TEST(Test__MTPStateTransaction,
     PlacementAwareGDNRequiresExactBytesWithoutMovement)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 4;
    candidate.gdn_layers.front().recurrence_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.gdn_state_policy =
        MTPGDNStateComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);

    ASSERT_FALSE(result);
    EXPECT_FALSE(result.gdn_numerical.compared);
    EXPECT_NE(result.reason.find("GDN recurrence hash mismatch"), std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     PlacementAwareGDNAcceptsCompleteNumericalStateAfterMovement)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 5;
    candidate.gdn_layers.front().recurrence_hash ^= 0x1;
    candidate.gdn_layers.front().conv_hash ^= 0x2;
    candidate.gdn_layers.front().recurrence_sample_values[1] += 1e-4f;
    candidate.gdn_layers.front().conv_sample_values[2] += 1e-4f;

    MTPRuntimeSnapshotComparisonOptions options;
    options.gdn_state_policy =
        MTPGDNStateComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    options.gdn_relative_l2_tolerance = 1e-3;
    options.gdn_max_abs_tolerance = 1e-3;
    options.gdn_min_cosine = 0.999999;
    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);

    ASSERT_TRUE(result) << result.reason;
    EXPECT_TRUE(result.gdn_numerical.compared);
    EXPECT_TRUE(result.gdn_numerical.passed);
    EXPECT_EQ(result.gdn_numerical.payloads, 2u);
    EXPECT_EQ(result.gdn_numerical.elements, 76u);
    EXPECT_GT(result.gdn_numerical.maximum_abs, 0.0);
}

TEST(Test__MTPStateTransaction,
     PlacementAwareGDNRejectsMissingOrNumericallyBadState)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.moe_runtime_movement_epoch = 4;
    candidate.moe_runtime_movement_epoch = 5;
    candidate.gdn_layers.front().recurrence_hash ^= 0x1;

    MTPRuntimeSnapshotComparisonOptions options;
    options.gdn_state_policy =
        MTPGDNStateComparisonPolicy::
            ExactUnlessMoEPlacementChanged;
    options.gdn_relative_l2_tolerance = 0.05;
    options.gdn_max_abs_tolerance.reset();
    options.gdn_min_cosine = 0.99;

    candidate.gdn_layers.front().recurrence_sample_values.clear();
    auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_NE(
        result.reason.find("complete retained values"),
        std::string::npos)
        << result.reason;

    candidate = oracle;
    candidate.moe_runtime_movement_epoch = 5;
    candidate.gdn_layers.front().recurrence_hash ^= 0x1;
    std::fill(
        candidate.gdn_layers.front().recurrence_sample_values.begin(),
        candidate.gdn_layers.front().recurrence_sample_values.end(),
        -10.0f);
    result = compareMTPRuntimeStateSnapshots(oracle, candidate, options);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.gdn_numerical.compared);
    EXPECT_FALSE(result.gdn_numerical.passed);
    EXPECT_NE(
        result.reason.find("GDN recurrence numerical mismatch"),
        std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction,
     RuntimeSnapshotRejectsAsymmetricGDNDeviceAuthority)
{
    PrefixRuntimeStateSnapshot oracle = makeRuntimeSnapshot(7);
    PrefixRuntimeStateSnapshot candidate = oracle;
    oracle.gdn_layers.front().device_state_hash_available = true;

    const auto result =
        compareMTPRuntimeStateSnapshots(oracle, candidate);

    ASSERT_FALSE(result);
    EXPECT_NE(
        result.reason.find("device-state hash availability mismatch"),
        std::string::npos)
        << result.reason;
}

TEST(Test__MTPStateTransaction, VisibleCommitPlanIsTotalAcrossMTPDepthAndResponseBudget)
{
    /*
     * Exercise every production MTP verifier width (depths 1..15 produce
     * verifier widths 2..16), both ordinary and previously-emitted condition
     * rows, and budgets on both sides of the transaction width.  This is the
     * pure state-machine oracle used by CUDA, ROCm, and CPU publication.
     */
    for (int verifier_rows = 1; verifier_rows <= 16; ++verifier_rows)
    {
        for (int emitted_start = 0; emitted_start <= 1; ++emitted_start)
        {
            if (emitted_start > verifier_rows)
                continue;

            const int maximum_new_outputs = verifier_rows - emitted_start;
            for (int budget = 0; budget <= 18; ++budget)
            {
                const MTPVisibleStateCommitPlan plan =
                    planMTPVisibleStateCommit(
                        verifier_rows,
                        emitted_start,
                        budget);
                ASSERT_TRUE(plan)
                    << "rows=" << verifier_rows
                    << " emitted_start=" << emitted_start
                    << " budget=" << budget
                    << " reason=" << plan.reason;

                const bool reaches_response_boundary =
                    budget > 0 && budget <= maximum_new_outputs;
                const int expected_commit_rows =
                    reaches_response_boundary
                        ? std::clamp(
                              budget + emitted_start - 1,
                              0,
                              verifier_rows)
                        : verifier_rows;
                EXPECT_EQ(
                    plan.max_state_commit_rows,
                    expected_commit_rows)
                    << "rows=" << verifier_rows
                    << " emitted_start=" << emitted_start
                    << " budget=" << budget;
                EXPECT_EQ(
                    plan.response_boundary_clipped,
                    expected_commit_rows < verifier_rows);
            }
        }
    }
}

TEST(Test__MTPStateTransaction, VisibleCommitPlanRejectsAmbiguousLifecycleInputs)
{
    EXPECT_FALSE(planMTPVisibleStateCommit(0, 0, 1));
    EXPECT_FALSE(planMTPVisibleStateCommit(2, -1, 1));
    EXPECT_FALSE(planMTPVisibleStateCommit(2, 3, 1));
    EXPECT_FALSE(planMTPVisibleStateCommit(2, 0, -1));
}

} // namespace llaminar2
