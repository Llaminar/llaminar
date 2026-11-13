/**
 * @file Test__PipelineMoECommitEvidence.cpp
 * @brief Actual committed-round schema and capture-identity replay for pipeline maintenance.
 *
 * Tail compact outcomes and accepted KV rows have different contracts. This
 * device-free replay exercises production projection and arithmetic, preserving
 * original metadata identities without claiming a GPU or executable sampler.
 */
#include "execution/local_execution/orchestrators/PipelineForwardGraphEdges.h"
#include "execution/local_execution/orchestrators/PipelineMetadataExchange.h"
#include "execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.h"
#include "kernels/common/SamplingMath.h"
#include <gtest/gtest.h>
#include <array>
#include <set>
using namespace llaminar2;

/** @test Only three original compact fields join the four accepted-state fields. */
TEST(PipelineMoECommitEvidence, ExactOriginalFieldsAndSemanticWireExtentReplay)
{
    using Stage = MTPSpeculativeStatePublicationStage;
    std::array<int32_t, 4> accepted{};
    std::array<int32_t, sampling_math::kSpeculativeBatchMetaCount> original{};
    Stage::Params params;
    params.accepted_restore_rows_device = std::span(accepted).first(1);
    params.target_cached_tokens_device = accepted.data() + 1;
    params.accepted_state_counts_device = accepted.data() + 2;
    params.publication_ok_flags_device = accepted.data() + 3;
    const auto static_banks = PipelineForwardGraphEdges::PublicationBanks::from(params);
    ASSERT_TRUE(static_banks.valid());
    EXPECT_EQ(static_banks.committed_round_evidence, (std::array<int32_t *, 3>{}));
    params.pipeline_moe_commit_evidence = Stage::PipelineMoECommitEvidence{original.data(), int(original.size())};
    const auto banks = PipelineForwardGraphEdges::PublicationBanks::from(params);
    ASSERT_TRUE(banks.valid());
    EXPECT_EQ(banks.committed_round_evidence, (std::array<int32_t *, 3>{
        original.data() + sampling_math::kSpecBatchMetaOk,
        original.data() + sampling_math::kSpecBatchMetaOutputCount,
        original.data() + sampling_math::kSpecBatchMetaLeadingCommittedOutputCount}));
    auto incomplete = banks; incomplete.committed_round_evidence[1] = nullptr;
    EXPECT_FALSE(incomplete.valid());
    auto aliased = banks; aliased.committed_round_evidence[2] = accepted.data();
    EXPECT_FALSE(aliased.valid());
    const PipelineMetadataLayout layout(PipelineMetadataLayout::Kind::CommittedStateWithMoERounds, 1);
    ASSERT_EQ(layout.fieldCount(), 7u);
    std::set<std::uint64_t> keys;
    std::size_t actual_bytes = 0;
    for (std::size_t field = 0; field < layout.fieldCount(); ++field) {
        const auto message = layout.message(field);
        EXPECT_EQ(message.bytes, sizeof(int32_t)); actual_bytes += message.bytes;
        EXPECT_TRUE(keys.insert(message.key).second);
        EXPECT_NE(message.key, PipelineMetadataLayout(PipelineMetadataLayout::Kind::CommittedState, 1).message(0).key);
    }
    EXPECT_EQ(actual_bytes, 7 * sizeof(int32_t));
    EXPECT_THROW(layout.message(7), std::out_of_range);
}

/** @test Follower cadence evidence is an explicit read dependency and complete capture identity. */
TEST(PipelineMoECommitEvidence, FollowerOwnClockCannotAcquireOutcomeOrResponseAuthority)
{
    using Stage = MTPSpeculativeStatePublicationStage;
    std::array<int32_t, 12> original{}, foreign{};
    std::array<uint32_t, 4> clock{};
    Stage::Params params;
    params.authority = Stage::Authority::PipelineFollower;
    params.pipeline_moe_commit_evidence = Stage::PipelineMoECommitEvidence{original.data(), 12};
    params.dynamic_moe_commit_boundary = Stage::DynamicMoECommitBoundaryBinding{
        &clock[0], &clock[1], &clock[2], &clock[3]};
    Stage stage(params); // Contract projection only; no device ownership is fabricated.
    EXPECT_TRUE(stage.hasSameCaptureIdentity(params));
    const auto contract = stage.bufferContract();
    ASSERT_EQ(contract.inputs.size(), 1u);
    EXPECT_EQ(contract.inputs.front().id, BufferId::STOCHASTIC_BATCH_OUTPUT_META);
    EXPECT_TRUE(contract.inouts.empty()); EXPECT_TRUE(contract.outputs.empty());
    auto changed = params; changed.pipeline_moe_commit_evidence->meta_device = foreign.data();
    EXPECT_FALSE(stage.hasSameCaptureIdentity(changed));
    changed = params; changed.pipeline_moe_commit_evidence->meta_stride = 13;
    EXPECT_FALSE(stage.hasSameCaptureIdentity(changed));
    changed = params; changed.dynamic_moe_commit_boundary->maintenance_due_device = &clock[0];
    EXPECT_FALSE(stage.hasSameCaptureIdentity(changed));
    params.pipeline_moe_commit_evidence->meta_stride = 11;
    EXPECT_FALSE(params.pipeline_moe_commit_evidence->valid());
    params.pipeline_moe_commit_evidence->meta_stride = 12;
    params.pipeline_moe_commit_evidence->meta_device = nullptr;
    EXPECT_FALSE(params.pipeline_moe_commit_evidence->valid());
}

/** @test Actual publication clipping and carry arithmetic cannot substitute accepted KV rows for rounds. */
TEST(PipelineMoECommitEvidence, CarryAndClippedStateRowsAreDistinctFromNewCommittedRoundsReplay)
{
    using namespace sampling_math;
    for (int replay = 0; replay < 20; ++replay) {
        std::array<int, kSpeculativeBatchMetaCount> original{};
        original[kSpecBatchMetaOk] = 1;
        original[kSpecBatchMetaOutputCount] = 4;
        original[kSpecBatchMetaTargetVerifierStateCommitCount] = 4;
        original[kSpecBatchMetaLeadingCommittedOutputCount] = replay % 2;
        int restore = -1, cached = -1, accepted = -1, healthy = 0;
        const int state_limit = replay % 3 + 1;
        derive_speculative_publication_metadata(original.data(), original.size(), 0, 4, 32, state_limit,
            &restore, &cached, &accepted, &healthy);
        ASSERT_EQ(healthy, 1); EXPECT_EQ(accepted, state_limit); EXPECT_EQ(cached, 32 + accepted);
        const int rounds = speculative_new_commit_count(original[kSpecBatchMetaOutputCount],
            original[kSpecBatchMetaLeadingCommittedOutputCount]);
        EXPECT_EQ(rounds, 4 - replay % 2);
        if (state_limit == 1) EXPECT_NE(rounds, accepted);
    }
    EXPECT_EQ(speculative_new_commit_count(1, 1), 0); // A carried row alone advances no new round.
    EXPECT_EQ(speculative_new_commit_count(1, 2), -1);
    EXPECT_EQ(speculative_new_commit_count(0, 0), -1);
}
