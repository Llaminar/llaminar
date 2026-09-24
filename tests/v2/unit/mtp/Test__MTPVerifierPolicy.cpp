/**
 * @file Test__MTPVerifierPolicy.cpp
 * @brief Device-free proof of verifier policy and captured/live row separation.
 *
 * These tests exercise the metadata arithmetic used by captured GPU kernels,
 * including shrinking, growing, empty, and tiled logical extents. No device
 * count is read by host policy and no accelerator is required.
 */
#include <gtest/gtest.h>

#include "execution/mtp/MTPVerifierPolicy.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/DeviceRowWorkGrid.h"
#include "tensors/TensorKernels.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace llaminar2
{

/** @brief Response clipping and checkpoint sizing share one executed width. */
TEST(Test__MTPVerifierPolicy, TransactionDraftAdmissionSeparatesCommitBudgetAndGraphWidth)
{
    for (int capacity = 1; capacity <= 31; ++capacity)
        for (int budget = 0; budget <= 33; ++budget)
            for (int leading_cost : {0, 1})
                for (const auto authority : {MTPCommitBudgetAuthority::Host,
                                             MTPCommitBudgetAuthority::Device})
                {
                    const int remaining = std::max(0, budget - leading_cost);
                    const int expected = budget == 0 ? capacity :
                        remaining == 0 ? 0 :
                        authority == MTPCommitBudgetAuthority::Device ? capacity :
                        std::min(capacity, remaining);
                    EXPECT_EQ(mtpTransactionDraftCount(
                        capacity, budget, leading_cost, authority), expected);
                }
    EXPECT_EQ(mtpTransactionDraftCount(-1, 1, 1, MTPCommitBudgetAuthority::Host), -1);
    EXPECT_EQ(mtpTransactionDraftCount(15, -1, 1, MTPCommitBudgetAuthority::Device), -1);
    EXPECT_EQ(mtpTransactionDraftCount(15, 1, 2, MTPCommitBudgetAuthority::Device), -1);
}

/** @brief Request depth must not shrink the immutable setup capture envelope. */
TEST(Test__MTPVerifierPolicy, ServingCaptureUsesRetainedCapacityAcrossRequests)
{
    for (const int retained_depth : {1, 2, 3, 7, 15})
        for (const auto mode : {MTPDepthPolicyMode::Fixed,
                                MTPDepthPolicyMode::Observe,
                                MTPDepthPolicyMode::Dynamic})
            for (int selected_depth = 1; selected_depth <= retained_depth;
                 ++selected_depth)
            {
                SCOPED_TRACE(::testing::Message()
                    << "retained=" << retained_depth
                    << " selected=" << selected_depth
                    << " mode=" << static_cast<int>(mode));
                MTPRuntimeConfig mtp;
                mtp.enabled = true;
                mtp.draft_tokens = selected_depth;
                mtp.graph_capacity_draft_tokens = retained_depth;
                mtp.depth_policy.mode = mode;
                mtp.depth_policy.min_depth = 1;
                mtp.depth_policy.initial_depth = selected_depth;
                mtp.depth_policy.max_depth = selected_depth;
                const auto geometry = resolveMTPServingForwardCaptureGeometry(
                    mtp, retained_depth + 1);
                ASSERT_TRUE(geometry.enabled);
                ASSERT_TRUE(geometry.valid());
                EXPECT_EQ(geometry.draft_depth, retained_depth);
                EXPECT_EQ(geometry.verifier_rows, retained_depth + 1);

                // The wider capture envelope must never authorize deeper
                // execution: only request admission owns the active policy.
                const auto policy = resolveMTPDeviceGenerationDepthPolicy(mtp);
                EXPECT_EQ(policy.initial_depth, selected_depth);
                EXPECT_EQ(policy.maximum_depth, selected_depth);
            }
}

/** @brief Largest-first ordering includes a later verifier, not just request one. */
TEST(Test__MTPVerifierPolicy, RetainedVerifierWinsSharedArenaCaptureOrdering)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 1;
    mtp.graph_capacity_draft_tokens = 15;
    const auto geometry = resolveMTPServingForwardCaptureGeometry(mtp, 16);
    for (const int prefill_rows : {1, 2, 4, 8, 16})
        EXPECT_TRUE(materializeMTPWideCheckpointLaneFirst(
            true, prefill_rows, geometry));
    EXPECT_FALSE(materializeMTPWideCheckpointLaneFirst(true, 32, geometry));
    EXPECT_FALSE(materializeMTPWideCheckpointLaneFirst(false, 8, geometry));
    EXPECT_FALSE(materializeMTPWideCheckpointLaneFirst(true, 0, geometry));
    const auto undersized = resolveMTPServingForwardCaptureGeometry(mtp, 8);
    EXPECT_FALSE(undersized.valid());
    EXPECT_FALSE(materializeMTPWideCheckpointLaneFirst(true, 8, undersized));

    // An ordinary non-MTP runner has no verifier to put first.
    const auto disabled = resolveMTPServingForwardCaptureGeometry({}, 0);
    EXPECT_FALSE(disabled.enabled);
    EXPECT_TRUE(disabled.valid());
    EXPECT_FALSE(materializeMTPWideCheckpointLaneFirst(true, 8, disabled));
}

/** @brief Bounded grids cover every live row exactly once without consulting its owner. */
TEST(Test__MTPVerifierPolicy, CountedWorkerGridCoversEveryLiveTile)
{
    const std::int32_t owner = -1; // An invalid value proves planning does not read it.
    for (int capacity : {1, 3, 16, 17, 31, 65})
        for (int tile_width : {1, 2, 4, 8})
            for (int independent : {1, 13, 64, 816, 13056})
            {
                const auto rows = DeviceRowRange::deviceCounted(capacity, &owner);
                const int workers = deviceRowWorkerGroups(rows, tile_width, independent, 720);
                const int physical = 1 + (capacity - 1) / tile_width;
                ASSERT_GE(workers, 1);
                ASSERT_LE(workers, physical);
                EXPECT_GE(workers * independent, std::min(720, physical * independent));
                for (int live = 0; live <= capacity; ++live)
                {
                    std::vector<int> visits(capacity);
                    for (int worker = 0; worker < workers; ++worker)
                        for (int first = worker * tile_width; first < live; first += workers * tile_width)
                            for (int row = first; row < std::min(first + tile_width, live); ++row)
                                ++visits[row];
                    for (int row = 0; row < capacity; ++row)
                        ASSERT_EQ(visits[row], row < live ? 1 : 0);
                }
                EXPECT_EQ(deviceRowWorkerGroups(DeviceRowRange::fullyActive(capacity),
                    tile_width, independent, 720), physical);
            }
}

/** @brief Rejected and extreme metadata cannot produce an empty or overflowing launch. */
TEST(Test__MTPVerifierPolicy, CountedWorkerGridValidatesGeometry)
{
    const std::int32_t owner = 0;
    const auto rows = DeviceRowRange::deviceCounted(16, &owner);
    EXPECT_EQ(deviceRowWorkerGroups(rows, 4, 13056, 720), 1);
    EXPECT_EQ(deviceRowWorkerGroups(rows, 4, 1, 720), 4);
    EXPECT_THROW((void)deviceRowWorkerGroups(rows, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW((void)deviceRowWorkerGroups(rows, 1, 0, 1), std::invalid_argument);
    EXPECT_THROW((void)deviceRowWorkerGroups(rows, 1, 1, -1), std::invalid_argument);
    const auto maximum = std::numeric_limits<std::int64_t>::max();
    EXPECT_EQ(deviceRowWorkerGroups(rows, 1, maximum, maximum), 1);
    EXPECT_EQ(deviceRowWorkerGroups(rows, 1, 1, maximum), 16);
}

TEST(Test__MTPVerifierPolicy, VerifierRecordingScopesRestoreGeometry)
{
    using Scope = ITensorGemm::VerifierKernelModeScope;
    const std::int32_t owner = 0; // Borrow address only, never a host live count.
    EXPECT_EQ(Scope::rowsFor(16), nullptr);
    {
        Scope outer(DeviceRowRange::deviceCounted(16, &owner));
        EXPECT_EQ(Scope::rowsFor(16)->countOwner(), &owner);
        EXPECT_THROW((void)Scope::rowsFor(3), std::logic_error);
        {
            Scope inherited;
            EXPECT_EQ(Scope::rowsFor(16)->countOwner(), &owner);
        }
        try
        {
            Scope independent(DeviceRowRange::fullyActive(3));
            EXPECT_EQ(Scope::rowsFor(3)->countOwner(), nullptr);
            throw std::runtime_error("adversarial launch failure");
        }
        catch (const std::runtime_error &) {}
        EXPECT_EQ(Scope::rowsFor(16)->countOwner(), &owner);
    }
    EXPECT_EQ(Scope::rowsFor(3), nullptr);
}

TEST(Test__MTPVerifierPolicy, DeviceRowsRetainCapacityAcrossDepthChanges)
{
    const std::int32_t owner = 0; // Address identity only: never dereferenced.
    const auto rows = DeviceRowRange::deviceCounted(16, &owner);
    for (int active : {3, 2, 16, 1, 0, 15, 3})
    {
        EXPECT_EQ(rows.activeRowsFor(active), active);
        EXPECT_EQ(rows.physicalRows(), 16);
        EXPECT_EQ(rows.capacity(), 16);
        EXPECT_EQ(rows.countOwner(), &owner);
    }
}

TEST(Test__MTPVerifierPolicy, DeviceRowSlicesPreserveScratchStrideAndCountOwner)
{
    const std::int32_t owner = 0;
    for (int capacity : {1, 2, 3, 15, 16, 17, 31, 64})
    {
        const auto matrix = DeviceRowRange::deviceCounted(capacity, &owner);
        for (int tile_width : {1, 2, 4, 8, 16})
        {
            for (int active = 0; active <= capacity; ++active)
            {
                int covered = 0;
                for (int first = 0; first < capacity; first += tile_width)
                {
                    const int width = std::min(tile_width, capacity - first);
                    const auto tile = matrix.slice(first, width);
                    const int expected = std::clamp(active - first, 0, width);
                    EXPECT_EQ(tile.activeRowsFor(active), expected);
                    EXPECT_EQ(tile.physicalRows(), width);
                    EXPECT_EQ(tile.capacity(), capacity);
                    EXPECT_EQ(tile.countOwner(), &owner);
                    covered += tile.activeRowsFor(active);
                    // A nested slice keeps coordinates in the original matrix.
                    EXPECT_EQ(tile.slice(width - 1, 1).activeRowsFor(active),
                              active > first + width - 1 ? 1 : 0);
                }
                EXPECT_EQ(covered, active);
            }
        }
    }
}

TEST(Test__MTPVerifierPolicy, DeviceRowsRejectInvalidGeometryAndPublication)
{
    EXPECT_THROW((void)DeviceRowRange::fullyActive(0), std::invalid_argument);
    EXPECT_THROW((void)DeviceRowRange::fullyActive(-1), std::invalid_argument);
    EXPECT_THROW((void)DeviceRowRange::deviceCounted(16, nullptr), std::invalid_argument);
    const auto rows = DeviceRowRange::fullyActive(16);
    EXPECT_EQ(rows.countOwner(), nullptr);
    EXPECT_EQ(rows.activeRowsFor(-1), -1);
    EXPECT_EQ(rows.activeRowsFor(17), -1);
    EXPECT_THROW((void)rows.slice(-1, 1), std::out_of_range);
    EXPECT_THROW((void)rows.slice(0, 0), std::out_of_range);
    EXPECT_THROW((void)rows.slice(16, 1), std::out_of_range);
    EXPECT_THROW((void)rows.slice(1, std::numeric_limits<int>::max()), std::out_of_range);
}

TEST(Test__MTPVerifierPolicy, PhysicalRowBucketsBoundTheCapturedGraphFamily)
{
    constexpr int max_rows = 16;
    constexpr int expected[] = {
        1, 2, 4, 4, 8, 8, 8, 8,
        16, 16, 16, 16, 16, 16, 16, 16,
    };

    for (int logical_rows = 1; logical_rows <= max_rows; ++logical_rows)
    {
        EXPECT_EQ(
            mtpVerifierPhysicalRowBucket(logical_rows, max_rows),
            expected[logical_rows - 1])
            << "logical_rows=" << logical_rows;
    }
}

TEST(Test__MTPVerifierPolicy, PhysicalRowBucketsRemainTotalForNonPowerOfTwoCaps)
{
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(1, 15), 1);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(3, 15), 4);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(8, 15), 8);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(9, 15), 15);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(15, 15), 15);
}

TEST(Test__MTPVerifierPolicy, PhysicalRowBucketsRejectInvalidGeometry)
{
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(0, 16), 0);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(-1, 16), 0);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(1, 0), 0);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(17, 16), 0);
}

TEST(Test__MTPVerifierPolicy,
     PhysicalPaddedWidthBucketsOnlyTheScalarGpuLane)
{
    constexpr auto bucket =
        MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 5, 15, bucket), 8);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 9, 15, bucket), 15);

    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(2, 5, 15, bucket), 5);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(8, 9, 15, bucket), 9);

    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(0, 5, 15, bucket), 0);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 0, 15, bucket), 0);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 16, 15, bucket), 0);
}

TEST(Test__MTPVerifierPolicy,
     DynamicDeviceEnvelopeIsStableAcrossEveryScalarLogicalWidth)
{
    constexpr int max_rows = 16;
    constexpr auto envelope =
        MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope;
    for (int logical_rows = 1; logical_rows <= max_rows; ++logical_rows)
    {
        EXPECT_EQ(
            mtpVerifierPhysicalPaddedSeqLen(
                /*request_count=*/1,
                logical_rows,
                max_rows,
                envelope),
            max_rows)
            << "logical_rows=" << logical_rows;
    }

    EXPECT_EQ(
        mtpVerifierPhysicalPaddedSeqLen(2, 5, max_rows, envelope),
        5)
        << "Request-batched metadata retains its explicit row stride.";
}

TEST(Test__MTPVerifierPolicy, GreedyUsesGroupedDecodeEquivalentOutcomeByDefault)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_grouped_decode_equivalent_outcome");
}

TEST(Test__MTPVerifierPolicy, StochasticUsesGroupedDecodeEquivalentOutcomeByDefault)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .stochastic_verify = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_STREQ(
        decision.reason,
        "stochastic_uses_grouped_decode_equivalent_outcome");
}

TEST(Test__MTPVerifierPolicy, PenaltiesWithoutRowLocalSupportAreUnsupported)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
            });

    EXPECT_EQ(decision.path, MTPVerifierExecutionPath::Unsupported);
    EXPECT_STREQ(
        decision.reason,
        "row_local_penalty_application_required_for_grouped_verifier");
}

TEST(Test__MTPVerifierPolicy, PenaltiesUseGroupedOutcomeWithRowLocalSupport)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
                .supports_row_local_penalty_application = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_grouped_decode_equivalent_outcome");
}

TEST(Test__MTPVerifierPolicy, NonGreedyWithoutStochasticVerifierIsUnsupported)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{});

    EXPECT_EQ(decision.path, MTPVerifierExecutionPath::Unsupported);
    EXPECT_STREQ(
        decision.reason,
        "sampling_mode_not_supported_by_grouped_verifier");
}

} // namespace llaminar2
