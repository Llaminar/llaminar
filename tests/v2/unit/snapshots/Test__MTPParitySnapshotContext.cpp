/**
 * @file Test__MTPParitySnapshotContext.cpp
 * @brief Device-free regressions for MTP checkpoint and serial-oracle authority.
 *
 * These tests keep tolerant HF tensor comparisons separate from exact native
 * token induction, and reject malformed lifecycle evidence before inference.
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

    TEST(Test__MTPParitySnapshotContext, InitialConditionOwnsDepthZeroBranchIdentity)
    {
        EXPECT_EQ(mtpParityBranchReferencePrefix(2, 0, {}, 760),
                  "decode_step2_CONDITION_760_MTP0_");
        constexpr std::array<int32_t, 2> drafts{13, 17};
        EXPECT_EQ(mtpParityBranchReferencePrefix(2, 2, drafts, 760),
                  "decode_step2_CONDITION_760_BRANCH_13_17_MTP2_");
        EXPECT_NE(mtpParityBranchReferencePrefix(2, 2, drafts, 760),
                  mtpParityBranchReferencePrefix(2, 2, drafts));
        EXPECT_THROW((void)mtpParityBranchReferencePrefix(2, 0, {}), std::invalid_argument);
        EXPECT_THROW((void)mtpParityBranchReferencePrefix(2, 0, {}, -1), std::invalid_argument);
        constexpr std::array<int32_t, 14> deepest{};
        EXPECT_NO_THROW((void)mtpParityBranchReferencePrefix(2, 14, deepest, 760));
        constexpr std::array<int32_t, 15> too_deep{};
        EXPECT_THROW((void)mtpParityBranchReferencePrefix(2, 15, too_deep, 760), std::invalid_argument);
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
        constexpr std::array<MTPParityDraftPrediction, 4> mtp0 = {{
            {561}, {760}, {3841}, {13477}}};
        constexpr std::array rows = {
            MTPParityCertifiedDecodeRow{0u, 13, 271},
            MTPParityCertifiedDecodeRow{1u, 271, 760},
            MTPParityCertifiedDecodeRow{2u, 760, 3841},
            MTPParityCertifiedDecodeRow{3u, 3841, 13477}};
        const auto selected =
            selectMTPParityCheckpointAcceptedDraftReference(13, serial, rows, mtp0);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(selected->reference_step, 1u);
        EXPECT_EQ(selected->base_token, 271);
        EXPECT_EQ(selected->first_draft_token, 760);

        constexpr std::array<MTPParityDraftPrediction, 2> rejected = {{{7}, {8}}};
        EXPECT_FALSE(
            selectMTPParityCheckpointAcceptedDraftReference(13, serial, rows, rejected)
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
         AcceptedDraftUsesNativeSuccessorAtHuggingFaceNearTie)
    {
        // The 122B reference forces 271 after row zero; native M=1 and the
        // predictor both choose 561. Row zero is an accepting native edge,
        // whereas row one's forced input cannot extend the native trajectory.
        constexpr std::array rows = {
            MTPParityCertifiedDecodeRow{0u, 13, 561},
            MTPParityCertifiedDecodeRow{1u, 271, 760},
            MTPParityCertifiedDecodeRow{2u, 760, 3841},
        };
        constexpr std::array<int32_t, 3> reference = {13, 271, 760};
        constexpr std::array<MTPParityDraftPrediction, 3> drafts = {{{561}, {760}, {3841}}};
        const auto selected = selectMTPParityCheckpointAcceptedDraftReference(
            13, reference, rows, drafts);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(selected->reference_step, 0u);
        EXPECT_EQ(selected->base_token, 13);
        EXPECT_EQ(selected->first_draft_token, 561);

        const auto short_proof = certifyMTPParitySerialTrajectory(13, rows, 2u);
        EXPECT_EQ(selectMTPParitySerialOracleSource(short_proof, 2u, true),
                  MTPParitySerialOracleSource::ComparedDecodeRows);
        EXPECT_EQ(selectMTPParitySerialOracleSource(short_proof, 2u, false),
                  MTPParitySerialOracleSource::ProductionRequest);
        const auto longer_proof = certifyMTPParitySerialTrajectory(13, rows, 4u);
        EXPECT_FALSE(longer_proof.complete(4u));
        EXPECT_EQ(selectMTPParitySerialOracleSource(longer_proof, 4u, true),
                  MTPParitySerialOracleSource::ProductionRequest);

        constexpr std::array<MTPParityDraftPrediction, 3> rejected = {{{7}, {760}, {3841}}};
        const auto later_checkpoint = selectMTPParityCheckpointAcceptedDraftReference(
            13, reference, rows, rejected);
        ASSERT_TRUE(later_checkpoint);
        EXPECT_EQ(later_checkpoint->reference_step, 2u)
            << "Only a new checkpoint request can use the later matching edge";
        EXPECT_FALSE(longer_proof.complete(4u))
            << "Checkpoint selection must not reclassify forced rows as free-running";
    }

    /** The dual-CUDA near-tie rejects row one but permits a new row-two request. */
    TEST(Test__MTPParitySnapshotContext, CheckpointRequestDoesNotInheritDivergedNativePrefix)
    {
        constexpr std::array<int32_t, 3> prompt = {760, 3841, 13477};
        constexpr std::array<int32_t, 5> reference = {13, 271, 760, 3841, 13477};
        constexpr std::array rows = {
            MTPParityCertifiedDecodeRow{0u, 13, 198},
            MTPParityCertifiedDecodeRow{1u, 271, 760},
            MTPParityCertifiedDecodeRow{2u, 760, 3841},
            MTPParityCertifiedDecodeRow{3u, 3841, 13477}};
        constexpr std::array<MTPParityDraftPrediction, 4> drafts = {{{561}, {760}, {3841}, {13477}}};
        const auto selected = selectMTPParityCheckpointAcceptedDraftReference(
            13, reference, rows, drafts);
        ASSERT_TRUE(selected);
        EXPECT_EQ(selected->reference_step, 2u);
        EXPECT_EQ(selected->base_token, 760);
        EXPECT_EQ(selected->first_draft_token, 3841);
        EXPECT_EQ(mtpParityCheckpointPrompt(prompt, reference, *selected),
                  (std::vector<int32_t>{760, 3841, 13477, 13, 271}));
        const auto serial = certifyMTPParitySerialTrajectory(13, rows, 5u);
        EXPECT_EQ(serial.tokens, (std::vector<int32_t>{13, 198}));
        EXPECT_FALSE(serial.complete(5u));
        auto wrong_condition = *selected;
        wrong_condition.base_token = 198;
        EXPECT_THROW((void)mtpParityCheckpointPrompt(prompt, reference, wrong_condition),
                     std::invalid_argument);
        wrong_condition.reference_step = reference.size();
        EXPECT_THROW((void)mtpParityCheckpointPrompt(prompt, reference, wrong_condition),
                     std::invalid_argument);
        EXPECT_THROW((void)mtpParityCheckpointPrompt({}, reference, *selected), std::invalid_argument);

        // The concrete row-two draft was another near-tie (3841/11316).
        // Prefer the pack's decisive row-three prediction; do not relax the
        // runtime requirement that the nominated draft is actually accepted.
        auto margins = drafts;
        margins[2].margin = 0.01f;
        margins[3].margin = 5.95f;
        const auto strongest = selectMTPParityCheckpointAcceptedDraftReference(
            13, reference, rows, margins);
        ASSERT_TRUE(strongest);
        EXPECT_EQ(strongest->reference_step, 3u);
        EXPECT_EQ(mtpParityCheckpointPrompt(prompt, reference, *strongest),
                  (std::vector<int32_t>{760, 3841, 13477, 13, 271, 760}));
    }

    /** Validate the entire pack before a locally accepting edge can be selected. */
    TEST(Test__MTPParitySnapshotContext, CheckpointSelectionRejectsMalformedRowsAfterEarlyAcceptance)
    {
        constexpr std::array<int32_t, 3> reference = {13, 271, 760};
        constexpr std::array<MTPParityDraftPrediction, 3> drafts = {{{561}, {760}, {3841}}};
        const std::array rows = {
            MTPParityCertifiedDecodeRow{0u, 13, 561},
            MTPParityCertifiedDecodeRow{1u, 271, 760},
            MTPParityCertifiedDecodeRow{2u, 760, 3841}};
        for (int defect = 0; defect != 3; ++defect)
        {
            auto malformed = rows;
            if (defect == 0) malformed[2].reference_step = 3u;
            if (defect == 1) malformed[2].committed_token = 198;
            if (defect == 2) malformed[2].predicted_successor_token = -1;
            EXPECT_FALSE(selectMTPParityCheckpointAcceptedDraftReference(
                13, reference, malformed, drafts));
        }
        EXPECT_FALSE(selectMTPParityCheckpointAcceptedDraftReference(-1, reference, rows, drafts));
        EXPECT_FALSE(selectMTPParityCheckpointAcceptedDraftReference(13, {}, rows, drafts));
        auto malformed_drafts = drafts;
        malformed_drafts.back().token = -1;
        EXPECT_FALSE(selectMTPParityCheckpointAcceptedDraftReference(
            13, reference, rows, malformed_drafts));
        malformed_drafts = drafts;
        malformed_drafts.back().margin = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(selectMTPParityCheckpointAcceptedDraftReference(
            13, reference, rows, malformed_drafts));
    }

    /** A reset-row numerical oracle cannot also certify a running epoch clock. */
    TEST(Test__MTPParitySnapshotContext, MovementOracleRequiresContinuousRequest)
    {
        constexpr std::array rows = {
            MTPParityCertifiedDecodeRow{0u, 13, 271},
            MTPParityCertifiedDecodeRow{1u, 271, 760}};
        const auto proof = certifyMTPParitySerialTrajectory(13, rows, 3u);
        ASSERT_TRUE(proof.complete(3u));
        EXPECT_EQ(selectMTPParitySerialOracleSource(
                      proof, 3u, true, MTPParitySerialOracleContinuity::ComparedRows),
                  MTPParitySerialOracleSource::ComparedDecodeRows);
        EXPECT_EQ(selectMTPParitySerialOracleSource(
                      proof, 3u, true, MTPParitySerialOracleContinuity::ContinuousRequest),
                  MTPParitySerialOracleSource::ProductionRequest);
        EXPECT_EQ(selectMTPParitySerialOracleSource(
                      {}, 3u, true, MTPParitySerialOracleContinuity::ContinuousRequest),
                  MTPParitySerialOracleSource::InvalidEvidence);
        EXPECT_EQ(selectMTPParitySerialOracleSource(
                      proof, 3u, true, static_cast<MTPParitySerialOracleContinuity>(99)),
                  MTPParitySerialOracleSource::InvalidEvidence);
    }

    TEST(Test__MTPParitySnapshotContext,
         MalformedSerialEvidenceCannotSelectProductionRequest)
    {
        for (const auto failure : {
                 MTPParitySerialCertificationFailure::InvalidRequestedHorizon,
                 MTPParitySerialCertificationFailure::InvalidPrefillPrediction,
                 MTPParitySerialCertificationFailure::NonContiguousDecodeRow,
                 MTPParitySerialCertificationFailure::InvalidSuccessorPrediction})
        {
            const MTPParitySerialTrajectoryCertification malformed{
                .tokens = {13, 561}, .failure = failure};
            EXPECT_EQ(selectMTPParitySerialOracleSource(malformed, 2u, true),
                      MTPParitySerialOracleSource::InvalidEvidence);
            EXPECT_EQ(selectMTPParitySerialOracleSource(
                          malformed, 2u, true,
                          MTPParitySerialOracleContinuity::ContinuousRequest),
                      MTPParitySerialOracleSource::InvalidEvidence);
        }
        EXPECT_EQ(selectMTPParitySerialOracleSource({}, 2u, true),
                  MTPParitySerialOracleSource::InvalidEvidence);
        constexpr std::array malformed_rows = {
            MTPParityCertifiedDecodeRow{1u, 13, 561}};
        constexpr std::array<MTPParityDraftPrediction, 1> drafts = {{{561}}};
        constexpr std::array<int32_t, 1> reference = {13};
        EXPECT_FALSE(selectMTPParityCheckpointAcceptedDraftReference(
            13, reference, malformed_rows, drafts).has_value());
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
            EXPECT_FALSE(plan.fitsMaintenanceWindow(1));
            EXPECT_TRUE(plan.fitsMaintenanceWindow(2));
            EXPECT_TRUE(plan.fitsMaintenanceWindow(depth + 1));
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

    /** Cadence, oracle offset, and adaptive budget must describe one request. */
    TEST(Test__MTPParitySnapshotContext, AdaptiveWitnessRetiresDeclaredInitialCadence)
    {
        for (int depth = 1; depth <= 15; ++depth)
        {
            for (int cadence : {1, 2, 16})
            {
                const auto plan = makeMTPParityAdaptiveWitnessPlan(depth, cadence);
                ASSERT_TRUE(plan.valid());
                EXPECT_EQ(plan.warmup_tokens, cadence);
                EXPECT_EQ(plan.response_tokens, depth + 2);
                EXPECT_EQ(plan.serial_oracle_tokens, cadence + depth + 2);
            }
            const auto no_device_clock = makeMTPParityAdaptiveWitnessPlan(depth);
            ASSERT_TRUE(no_device_clock.valid());
            EXPECT_EQ(no_device_clock.warmup_tokens, 1);
        }
        EXPECT_FALSE(makeMTPParityAdaptiveWitnessPlan(0, 2).valid());
        EXPECT_FALSE(makeMTPParityAdaptiveWitnessPlan(15, 0).valid());
        EXPECT_FALSE(makeMTPParityAdaptiveWitnessPlan(15, -1).valid());
        EXPECT_FALSE(makeMTPParityAdaptiveWitnessPlan(
            15, std::numeric_limits<int>::max()).valid());
        EXPECT_FALSE(makeMTPParityAdaptiveWitnessPlan(
            std::numeric_limits<int>::max(), 2).valid());
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
