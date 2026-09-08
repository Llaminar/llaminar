/**
 * @file Test__MTPParitySnapshotContext.cpp
 * @brief Device-free regressions for MTP parity snapshot context selection.
 */

#include <gtest/gtest.h>

#include "../../utils/MTPParitySnapshotContext.h"

#include <array>
#include <cstdint>
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

    TEST(Test__MTPParitySnapshotContext, CheckpointPlanUsesExecutedDepthNotResponseLimit)
    {
        constexpr auto plan = makeMTPParityCheckpointPlan(
            /*cpu_owned=*/false,
            /*execution_draft_depth=*/3);
        static_assert(plan.valid());
        static_assert(plan.count == 2);

        EXPECT_EQ(
            plan.checkpoints[0].role,
            MTPParityCheckpointRole::Primary);
        EXPECT_EQ(
            plan.checkpoints[0].context,
            MTPParityCheckpointContext::DeviceTargetTokenLivePosition);
        EXPECT_EQ(plan.checkpoints[0].reference_depth, 0);
        EXPECT_EQ(
            plan.checkpoints[1].role,
            MTPParityCheckpointRole::TerminalChained);
        EXPECT_EQ(
            plan.checkpoints[1].context,
            MTPParityCheckpointContext::DeviceChainedTokenLivePosition);
        EXPECT_EQ(plan.checkpoints[1].reference_depth, 2)
            << "A depth-three retained transaction leaves MTP2 in its chained "
               "snapshot bank even when the response ledger exposes fewer rows";
    }

    TEST(Test__MTPParitySnapshotContext, DepthOneHasOnlyItsPrimaryCheckpoint)
    {
        constexpr auto host_plan = makeMTPParityCheckpointPlan(
            /*cpu_owned=*/true,
            /*execution_draft_depth=*/1);
        static_assert(host_plan.valid());
        EXPECT_EQ(host_plan.count, 1u);
        EXPECT_EQ(
            host_plan.checkpoints[0].context,
            MTPParityCheckpointContext::HostConditionTokenLivePosition);
        EXPECT_EQ(host_plan.checkpoints[0].reference_depth, 0);

        constexpr auto invalid = makeMTPParityCheckpointPlan(
            /*cpu_owned=*/false,
            /*execution_draft_depth=*/0);
        static_assert(!invalid.valid());
        EXPECT_EQ(invalid.count, 0u);
    }

    TEST(Test__MTPParitySnapshotContext,
         ComparedCheckpointCarriesExactSidecarNamespaces)
    {
        const ComparedMTPParityCheckpoint checkpoint{
            .reference_step = 0,
            .model_layer = 48,
            .identity = {
                .role = MTPParityCheckpointRole::Primary,
                .context = MTPParityCheckpointContext::
                    DeviceTargetTokenLivePosition,
                .reference_depth = 0,
            },
            .production_stage_prefix =
                "MTP_DECODE_SIDECAR_DEVICE_TARGET_TOKEN_LIVE_POSITION_MTP0_",
            .reference_stage_prefix = "decode_step0_MTP0_",
        };

        ASSERT_TRUE(checkpoint.valid());
        EXPECT_EQ(
            checkpoint.productionKey("MOE_ROUTING_INDICES"),
            "MTP_DECODE_SIDECAR_DEVICE_TARGET_TOKEN_LIVE_POSITION_MTP0_"
            "MOE_ROUTING_INDICES");
        EXPECT_EQ(
            checkpoint.referenceKey("MOE_ROUTE_CONTRIBUTIONS"),
            "decode_step0_MTP0_MOE_ROUTE_CONTRIBUTIONS");
        EXPECT_NE(
            checkpoint.productionKey("MOE_ROUTING_INDICES"),
            "layer48_MOE_ROUTING_INDICES")
            << "A synthetic CSV layer is not a snapshot namespace authority";

        auto malformed = checkpoint;
        malformed.production_stage_prefix.pop_back();
        EXPECT_FALSE(malformed.valid());
    }

    TEST(Test__MTPParitySnapshotContext, BranchPrefixOwnsOnlyConsumedTokens)
    {
        constexpr std::array<int32_t, 3> consumed = {198, 760, 3841};
        EXPECT_EQ(
            mtpParityBranchReferencePrefix(0, 3, consumed),
            "decode_step0_BRANCH_198_760_3841_MTP3_")
            << "The output sampled by MTP3 is not one of its input condition "
               "tokens and must not enter the reference identity";
    }

    TEST(Test__MTPParitySnapshotContext, InvalidBranchIdentityFailsClosed)
    {
        constexpr std::array<int32_t, 2> too_short = {13, 17};
        EXPECT_THROW(
            (void)mtpParityBranchReferencePrefix(0, 3, too_short),
            std::invalid_argument);

        constexpr std::array<int32_t, 2> negative = {13, -1};
        EXPECT_THROW(
            (void)mtpParityBranchReferencePrefix(0, 2, negative),
            std::invalid_argument);
        EXPECT_THROW(
            (void)mtpParityBranchReferencePrefix(-1, 2, too_short),
            std::invalid_argument);
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

    TEST(Test__MTPParitySnapshotContext,
         PlacementTrajectoryRequiresOneEpochForCompleteHistory)
    {
        MTPParityPlacementEpochTrajectory stable{.epoch = 7u};
        stable.observe(7u, 7u, 7u);
        EXPECT_TRUE(stable.matches(7u));
        EXPECT_FALSE(stable.matches(8u));

        MTPParityPlacementEpochTrajectory crossed{.epoch = 7u};
        crossed.observe(7u, 8u, 7u);
        EXPECT_FALSE(crossed.epoch.has_value());
        crossed.observe(8u, 8u, 8u);
        EXPECT_FALSE(crossed.epoch.has_value())
            << "A stable suffix cannot erase recurrent state inherited from "
               "the earlier placement";

        MTPParityPlacementEpochTrajectory execution_mismatch{.epoch = 9u};
        execution_mismatch.observe(9u, 9u, 8u);
        EXPECT_FALSE(execution_mismatch.epoch.has_value());

        MTPParityPlacementEpochTrajectory missing_authority{.epoch = 3u};
        missing_authority.observe(3u, 3u, 0u);
        EXPECT_FALSE(missing_authority.epoch.has_value());

        MTPParityPlacementEpochTrajectory unknown_restore{.epoch = 11u};
        unknown_restore.invalidate();
        EXPECT_FALSE(unknown_restore.matches(11u));
    }

    TEST(Test__MTPParitySnapshotContext,
         FullWidthPolicyWitnessBudgetLeavesOnePendingResponseSlot)
    {
        EXPECT_EQ(mtpParityFullWidthPolicyWitnessBudget(1), 3);
        EXPECT_EQ(mtpParityFullWidthPolicyWitnessBudget(15), 17);
        EXPECT_EQ(mtpParityFullWidthPolicyWitnessBudget(0), 0);
        EXPECT_EQ(
            mtpParityFullWidthPolicyWitnessBudget(
                std::numeric_limits<int>::max()),
            0);
    }

    TEST(Test__MTPParitySnapshotContext, GroupedTokensUseSerialNotHuggingFaceOracle)
    {
        constexpr std::array<int32_t, 3> serial = {13, 271, 760};
        constexpr std::array<int32_t, 3> grouped = {13, 271, 760};
        constexpr std::array<int32_t, 3> hugging_face = {13, 271, 71093};

        static_assert(serial != hugging_face);
        const auto comparison =
            compareMTPGroupedTokensToSerialOracle(serial, grouped);
        EXPECT_TRUE(comparison.exact)
            << "A quantized Hugging Face near-tie must not replace the exact "
               "serial Llaminar oracle";
        EXPECT_EQ(comparison.compared_tokens, grouped.size());
    }

    TEST(Test__MTPParitySnapshotContext,
         AcceptedDraftReferenceSelectsTheEarliestMatchingSerialEdge)
    {
        constexpr std::array<int32_t, 5> serial = {
            13, 271, 760, 3841, 13477};
        constexpr std::array<int32_t, 4> mtp0 = {
            561, 760, 3841, 13477};
        const auto selected =
            firstMTPParityAcceptedDraftReference(serial, mtp0);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(selected->reference_step, 1u);
        EXPECT_EQ(selected->base_token, 271);
        EXPECT_EQ(selected->first_draft_token, 760);

        constexpr std::array<int32_t, 2> rejected = {7, 8};
        EXPECT_FALSE(
            firstMTPParityAcceptedDraftReference(serial, rejected)
                .has_value());
    }

    TEST(Test__MTPParitySnapshotContext,
         TeacherForcedRowsCertifyAFreeRunningSerialTrajectoryByInduction)
    {
        constexpr std::array rows = {
            MTPParityCertifiedDecodeRow{
                .reference_step = 0u,
                .committed_token = 13,
                .predicted_successor_token = 271,
            },
            MTPParityCertifiedDecodeRow{
                .reference_step = 1u,
                .committed_token = 271,
                .predicted_successor_token = 760,
            },
            MTPParityCertifiedDecodeRow{
                .reference_step = 2u,
                .committed_token = 760,
                .predicted_successor_token = 3841,
            },
        };

        const auto certification =
            certifyMTPParitySerialTrajectory(13, rows, 4u);
        ASSERT_TRUE(certification.complete(4u));
        EXPECT_EQ(
            certification.tokens,
            (std::vector<int32_t>{13, 271, 760, 3841}));
    }

    TEST(Test__MTPParitySnapshotContext,
         SerialCertificationFailsClosedAtTheFirstTeacherForcedDrift)
    {
        constexpr std::array rows = {
            MTPParityCertifiedDecodeRow{
                .reference_step = 0u,
                .committed_token = 13,
                .predicted_successor_token = 271,
            },
            MTPParityCertifiedDecodeRow{
                .reference_step = 1u,
                .committed_token = 71093,
                .predicted_successor_token = 760,
            },
        };

        const auto certification =
            certifyMTPParitySerialTrajectory(13, rows, 3u);
        EXPECT_FALSE(certification.complete(3u));
        EXPECT_EQ(
            certification.failure,
            MTPParitySerialCertificationFailure::DiscontinuousTokenEdge);
        EXPECT_EQ(certification.failure_row, 1u);
        EXPECT_EQ(certification.expected_token, 271);
        EXPECT_EQ(certification.observed_token, 71093);
    }

    TEST(Test__MTPParitySnapshotContext,
         SerialCertificationDistinguishesShortEvidenceFromBrokenEvidence)
    {
        constexpr std::array rows = {
            MTPParityCertifiedDecodeRow{
                .reference_step = 0u,
                .committed_token = 13,
                .predicted_successor_token = 271,
            },
        };

        const auto certification =
            certifyMTPParitySerialTrajectory(13, rows, 3u);
        EXPECT_FALSE(certification.complete(3u));
        EXPECT_EQ(
            certification.failure,
            MTPParitySerialCertificationFailure::MissingDecodeRow);
        EXPECT_EQ(certification.tokens, (std::vector<int32_t>{13, 271}));
        EXPECT_EQ(certification.failure_row, 1u);
    }

    TEST(Test__MTPParitySnapshotContext,
         DeviceCheckpointBudgetDoesNotNarrowCapturedDepth)
    {
        for (int depth = 1; depth <= 15; ++depth)
        {
            const auto plan = makeMTPParityCheckpointTransactionPlan(
                /*device_controller_owned=*/true,
                depth);
            ASSERT_TRUE(plan.valid()) << "depth=" << depth;
            EXPECT_EQ(plan.execution_draft_depth, depth);
            EXPECT_EQ(plan.response_token_budget, 2);
            EXPECT_TRUE(plan.device_commit_boundary);
        }

        const auto cpu_plan = makeMTPParityCheckpointTransactionPlan(
            /*device_controller_owned=*/false,
            /*execution_draft_depth=*/15);
        ASSERT_TRUE(cpu_plan.valid());
        EXPECT_EQ(cpu_plan.execution_draft_depth, 15);
        EXPECT_EQ(cpu_plan.response_token_budget, 16);
        EXPECT_FALSE(cpu_plan.device_commit_boundary);

        EXPECT_FALSE(
            makeMTPParityCheckpointTransactionPlan(true, 0).valid());
        EXPECT_FALSE(
            makeMTPParityCheckpointTransactionPlan(
                false,
                std::numeric_limits<int>::max())
                .valid());
    }

    TEST(Test__MTPParitySnapshotContext, GroupedTokenMismatchReportsFirstEdge)
    {
        constexpr std::array<int32_t, 4> serial = {13, 271, 760, 42};
        constexpr std::array<int32_t, 4> grouped = {13, 271, 71093, 42};
        const auto mismatch =
            compareMTPGroupedTokensToSerialOracle(serial, grouped);
        EXPECT_FALSE(mismatch.exact);
        EXPECT_EQ(mismatch.compared_tokens, 3u);
        EXPECT_EQ(mismatch.mismatch_index, 2u);
        EXPECT_EQ(mismatch.serial_token, 760);
        EXPECT_EQ(mismatch.grouped_token, 71093);

        constexpr std::array<int32_t, 2> short_oracle = {13, 271};
        const auto missing =
            compareMTPGroupedTokensToSerialOracle(short_oracle, grouped);
        EXPECT_FALSE(missing.exact);
        EXPECT_EQ(missing.mismatch_index, short_oracle.size());
        EXPECT_EQ(missing.serial_token, -1);
        EXPECT_EQ(missing.grouped_token, 71093);
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
