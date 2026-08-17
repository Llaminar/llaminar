/**
 * @file Test__MTPParitySnapshotContext.cpp
 * @brief Device-free regressions for MTP parity snapshot context selection.
 */

#include <gtest/gtest.h>

#include "../../utils/MTPParitySnapshotContext.h"

#include <array>
#include <string>

namespace llaminar2::test::parity
{
    TEST(Test__MTPParitySnapshotContext, EveryTypedContextRoundTrips)
    {
        constexpr std::array contexts = {
            MTPParityCheckpointContext::HostConditionTokenLivePosition,
            MTPParityCheckpointContext::HostChainedDraftLivePosition,
            MTPParityCheckpointContext::DeviceTargetTokenLivePosition,
            MTPParityCheckpointContext::DeviceResidentLogicalState,
            MTPParityCheckpointContext::DeviceChainedTokenLivePosition,
        };

        for (const auto context : contexts)
        {
            const std::string prefix(
                mtpParityCheckpointContextPrefix(context));
            const std::string key = prefix + "MTP2_MOE_ROUTER_OUTPUT";
            const auto parsed =
                mtpParityModelCheckpointContextPrefix(key);
            ASSERT_TRUE(parsed.has_value()) << key;
            EXPECT_EQ(*parsed, prefix) << key;
            EXPECT_TRUE(isMTPParityModelCheckpointSnapshotKey(key));
        }
    }

    TEST(Test__MTPParitySnapshotContext, HostPrimaryAndChainAreDisjoint)
    {
        const std::string primary_key =
            std::string(mtpParityCheckpointContextPrefix(
                MTPParityCheckpointContext::HostConditionTokenLivePosition)) +
            "MTP0_EMBEDDING";
        const std::string chain_key =
            std::string(mtpParityCheckpointContextPrefix(
                MTPParityCheckpointContext::HostChainedDraftLivePosition)) +
            "MTP0_EMBEDDING";

        const auto primary =
            mtpParityModelCheckpointContextPrefix(primary_key);
        const auto chain =
            mtpParityModelCheckpointContextPrefix(chain_key);
        ASSERT_TRUE(primary.has_value());
        ASSERT_TRUE(chain.has_value());
        EXPECT_NE(*primary, *chain);
    }

    TEST(Test__MTPParitySnapshotContext, OperationalAndMalformedKeysFailClosed)
    {
        constexpr std::array<std::string_view, 7> keys = {
            "MTP_DECODE_SIDECAR_DEVICE_TARGET_TOKEN_LIVE_POSITION_",
            "MTP_DECODE_SIDECAR_RESIDENT_LOGICAL_STATE_",
            "MTP_DECODE_SIDECAR_CHAIN_DEVICE_TOKEN_LIVE_POSITION_",
            "MTP_DECODE_SIDECAR_MTP_EMBEDDING",
            "MTP_DECODE_SIDECAR_MTP0_",
            "MTP0_EMBEDDING",
            "MTP_TERMINAL_HIDDEN_ROW_SELECT",
        };

        for (const auto key : keys)
        {
            EXPECT_FALSE(isMTPParityModelCheckpointSnapshotKey(key))
                << key;
            EXPECT_FALSE(
                mtpParityModelCheckpointContextPrefix(key).has_value())
                << key;
        }
    }

    TEST(Test__MTPParitySnapshotContext, TransactionActivityRequiresDraftAndVerifier)
    {
        EXPECT_EQ(
            classifyMTPParityTransactionActivity({2, 2}, {2, 2}),
            MTPParityTransactionActivity::None);
        EXPECT_EQ(
            classifyMTPParityTransactionActivity({1, 4}, {3, 5}),
            MTPParityTransactionActivity::Speculative);
        EXPECT_EQ(
            mtpParityAttemptedDraftTokenCount({1, 4}, {3, 5}),
            2u)
            << "A controller-selected depth three may execute only two drafts "
               "when the response budget truncates the transaction";
        EXPECT_EQ(
            mtpParityExecutedTransactionCount({1, 4}, {3, 5}),
            1u);
        EXPECT_EQ(
            classifyMTPParityTransactionActivity({1, 4}, {3, 6}),
            MTPParityTransactionActivity::Speculative)
            << "A resident GPU decodeStep may retire two depth-one transactions";
        EXPECT_EQ(
            mtpParityAttemptedDraftTokenCount({1, 4}, {3, 6}),
            2u);
        EXPECT_EQ(
            mtpParityExecutedTransactionCount({1, 4}, {3, 6}),
            2u);
        EXPECT_EQ(
            classifyMTPParityTransactionActivity({1, 4}, {2, 4}),
            MTPParityTransactionActivity::Inconsistent);
        EXPECT_EQ(
            classifyMTPParityTransactionActivity({1, 4}, {1, 5}),
            MTPParityTransactionActivity::Inconsistent);
        EXPECT_EQ(
            classifyMTPParityTransactionActivity({2, 4}, {1, 4}),
            MTPParityTransactionActivity::Inconsistent);
    }

    TEST(Test__MTPParitySnapshotContext, ReferenceStepUsesLiveConditionPosition)
    {
        constexpr int prompt_tokens = 9;
        EXPECT_EQ(
            mtpParityReferenceStepForConditionPosition(9, prompt_tokens),
            0);
        EXPECT_EQ(
            mtpParityReferenceStepForConditionPosition(11, prompt_tokens),
            2)
            << "A fully accepted depth-one transaction advances the live "
               "condition by two serial rows; the same position also maps a "
               "depth-two rejection that returned three response tokens but "
               "left serial condition row two pending";
        EXPECT_EQ(
            mtpParityReferenceStepForConditionPosition(16, prompt_tokens),
            7);
        EXPECT_EQ(
            mtpParityReferenceStepForConditionPosition(8, prompt_tokens),
            -1);
        EXPECT_EQ(
            mtpParityReferenceStepForConditionPosition(9, -1),
            -1);
    }
} // namespace llaminar2::test::parity
