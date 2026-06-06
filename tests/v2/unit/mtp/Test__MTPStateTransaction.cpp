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

} // namespace

TEST(Test__MTPStateTransaction, ExpectedShiftedTokensLagMainByOne)
{
    EXPECT_EQ(expectedShiftedMTPTokens(0), 0);
    EXPECT_EQ(expectedShiftedMTPTokens(1), 0);
    EXPECT_EQ(expectedShiftedMTPTokens(2), 1);
    EXPECT_EQ(expectedShiftedMTPTokens(9), 8);
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

} // namespace llaminar2
