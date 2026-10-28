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
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include "execution/mtp/MTPMainForwardPolicy.h"
#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/FixedHalfProjectionGroup.h"
#include "kernels/common/FixedFloatingProjectionGroup.h"
#include "kernels/common/DeviceRowWorkGrid.h"
#include "tensors/TensorKernels.h"
#include <algorithm>
#include <limits>
#include <thread>
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

/** @test Every admissible request width was captured and charged before request admission. */
TEST(Test__MTPVerifierPolicy, RetainedVerifierFamilyCoversEveryRequestWidth)
{
    for (int depth = 1; depth <= kMaximumSupportedMTPDraftDepth; ++depth)
    for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe, MTPDepthPolicyMode::Dynamic})
    for (int selected = 1; selected <= depth; ++selected)
    {
        SCOPED_TRACE(::testing::Message() << depth << '/' << int(mode) << '/' << selected);
        MTPRuntimeConfig mtp;
        mtp.enabled = true;
        mtp.draft_tokens = selected;
        mtp.graph_capacity_draft_tokens = depth;
        mtp.depth_policy.mode = mode;
        mtp.depth_policy.initial_depth = selected;
        mtp.depth_policy.max_depth = selected;
        const auto family = resolveMTPServingForwardCaptureFamily(mtp, depth + 1);
        ASSERT_FALSE(family.empty());
        EXPECT_EQ(family.front().verifier_rows, depth + 1);
        EXPECT_EQ(family.back().verifier_rows, 2);
        EXPECT_EQ(resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp, MTPVerifierGraphFamily::BoundedBuckets),
            kMTPConditionForwardPurposes.size() + 2 * family.size());
        int previous = depth + 2;
        for (const auto &geometry : family)
        {
            EXPECT_LT(geometry.verifier_rows, previous);
            previous = geometry.verifier_rows;
            EXPECT_EQ(geometry.draft_depth + 1, geometry.verifier_rows);
            for (const auto outcome : {MTPVerifierOutcomeGraphMode::Greedy, MTPVerifierOutcomeGraphMode::Disabled})
            {
                const MTPVerifierForwardPolicy policy(outcome, geometry.verifier_rows, family.front());
                EXPECT_EQ(policy.physicalRows(), geometry.verifier_rows);
                EXPECT_EQ(policy.outcome(), outcome);
            }
        }
        const auto width_policy = mode == MTPDepthPolicyMode::Dynamic
            ? MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope : MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
        const int rows = mtpVerifierPhysicalPaddedSeqLen(1, selected + 1, depth + 1, width_policy);
        EXPECT_EQ(std::count_if(family.begin(), family.end(), [&](const auto &geometry) {
            return geometry.verifier_rows == rows;
        }), 1) << "A live request must select exactly one pre-existing graph";
    }
}

/** @test Fixed depth seven retains four widths; malformed members cannot enter a replay. */
TEST(Test__MTPVerifierPolicy, FixedDepthOvercapacityRetainsExactVerifierMembers)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 7;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;
    mtp.graph_capacity_draft_tokens = 15;
    const auto family = resolveMTPServingForwardCaptureFamily(mtp, 16);
    std::vector<int> widths;
    for (const auto &geometry : family) widths.push_back(geometry.verifier_rows);
    EXPECT_EQ(widths, (std::vector<int>{16, 8, 4, 2}));
    EXPECT_EQ(resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp, MTPVerifierGraphFamily::BoundedBuckets), 10u);
    EXPECT_EQ(resolveMTPDeviceGenerationDepthPolicy(mtp).maximum_depth, 7);
    for (const int invalid : {-1, 0, 1, 3, 5, 7, 9, 15, 17, std::numeric_limits<int>::max()})
        EXPECT_THROW((void)MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode::Disabled, invalid, family.front()), std::invalid_argument);
    EXPECT_THROW((void)MTPVerifierForwardPolicy(static_cast<MTPVerifierOutcomeGraphMode>(99), 8, family.front()), std::invalid_argument);
    EXPECT_THROW((void)MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode::Disabled, 8, {}), std::invalid_argument);
    for (const int mismatched : {0, 8, 32})
        EXPECT_THROW((void)resolveMTPServingForwardCaptureFamily(mtp, mismatched), std::invalid_argument);
    EXPECT_TRUE(resolveMTPServingForwardCaptureFamily({}, 0).empty());
}

/** @test Hosted tickets select exactly one retained child at every depth and request mode. */
TEST(Test__MTPVerifierPolicy, ExactHostedFamilyCoversEveryDepthWithoutPadding)
{
    const auto family_kind = resolveMTPVerifierGraphFamily(MTPVerifierExecutionRole::HostedCapturedTransactions);
    ASSERT_EQ(family_kind, MTPVerifierGraphFamily::ExactRows);
    for (int depth = 1; depth <= kMaximumSupportedMTPDraftDepth; ++depth)
    for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe, MTPDepthPolicyMode::Dynamic})
    {
        MTPRuntimeConfig mtp;
        mtp.enabled = true;
        mtp.draft_tokens = 1;
        mtp.graph_capacity_draft_tokens = depth;
        mtp.depth_policy.mode = mode;
        mtp.depth_policy.min_depth = 1;
        mtp.depth_policy.initial_depth = 1;
        mtp.depth_policy.max_depth = depth;
        const auto family = resolveMTPServingForwardCaptureFamily(mtp, depth + 1, family_kind);
        ASSERT_EQ(family.size(), static_cast<std::size_t>(depth));
        ASSERT_EQ(resolveMTPRetainedVerifierWidthCount(mtp, family_kind), family.size());
        EXPECT_EQ(resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp, family_kind),
            kMTPConditionForwardPurposes.size() + 2 * family.size());
        const auto physical_policy = resolveMTPVerifierPhysicalWidthPolicy(
            MTPVerifierExecutionRole::HostedCapturedTransactions, mode);
        ASSERT_EQ(physical_policy, MTPVerifierPhysicalWidthPolicy::ExactLogicalRows);
        for (int selected_depth = depth; selected_depth >= 1; --selected_depth)
        {
            const auto index = static_cast<std::size_t>(depth - selected_depth);
            const auto &geometry = family[index];
            EXPECT_EQ(geometry.family, family_kind);
            EXPECT_EQ(geometry.verifier_rows, selected_depth + 1);
            EXPECT_EQ(geometry.draft_depth, selected_depth);
            EXPECT_EQ(mtpVerifierGraphFamilyWidthAt(index, depth + 1, family_kind), geometry.verifier_rows);
            EXPECT_TRUE(mtpVerifierGraphFamilyContains(geometry.verifier_rows, depth + 1, family_kind));
            for (int requests : {1, 2, 4})
                EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(requests, selected_depth + 1, depth + 1, physical_policy),
                    selected_depth + 1);
            for (auto outcome : {MTPVerifierOutcomeGraphMode::Greedy, MTPVerifierOutcomeGraphMode::Disabled})
                EXPECT_EQ(MTPVerifierForwardPolicy(outcome, selected_depth + 1, family.front()).physicalRows(),
                    selected_depth + 1);
        }
        EXPECT_EQ(mtpVerifierGraphFamilyWidthAt(family.size(), depth + 1, family_kind), 0);
        EXPECT_FALSE(mtpVerifierGraphFamilyContains(1, depth + 1, family_kind));
        EXPECT_FALSE(mtpVerifierGraphFamilyContains(depth + 2, depth + 1, family_kind));
        EXPECT_THROW((void)MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode::Disabled, 1, family.front()), std::invalid_argument);
    }
}

/** @test Native conditional dynamic parents keep their envelope and standalone roles keep buckets. */
TEST(Test__MTPVerifierPolicy, VerifierExecutionRolesPreserveUnchangedFamilies)
{
    for (auto role : {MTPVerifierExecutionRole::Standalone, MTPVerifierExecutionRole::NativeConditionalParent})
    for (auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe, MTPDepthPolicyMode::Dynamic})
    {
        EXPECT_EQ(resolveMTPVerifierGraphFamily(role), MTPVerifierGraphFamily::BoundedBuckets);
        const auto policy = resolveMTPVerifierPhysicalWidthPolicy(role, mode);
        const bool native_dynamic = role == MTPVerifierExecutionRole::NativeConditionalParent && mode == MTPDepthPolicyMode::Dynamic;
        EXPECT_EQ(policy, native_dynamic ? MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope :
            MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket);
        for (int rows = 2; rows <= 16; ++rows)
            EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, rows, 16, policy),
                native_dynamic ? 16 : mtpVerifierPhysicalRowBucket(rows, 16));
    }
}

/** @test Exact retention is setup-owned even when request one disables MTP; invalid policy cannot underprice it. */
TEST(Test__MTPVerifierPolicy, ExactHostedFamilyAdmissionIsIndependentOfActiveDepth)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = false;
    mtp.graph_capacity_draft_tokens = 15;
    EXPECT_EQ(resolveMTPRetainedVerifierWidthCount(mtp, MTPVerifierGraphFamily::ExactRows), 15u);
    EXPECT_EQ(resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp, MTPVerifierGraphFamily::ExactRows), 32u);
    EXPECT_EQ(resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp, MTPVerifierGraphFamily::BoundedBuckets), 10u);
    EXPECT_EQ(resolveMTPRetainedServingForwardModelGraphIdentityCount({}, MTPVerifierGraphFamily::ExactRows), 0u);
    // Disabled execution builds no live verifier; admission still reserves the complete family above.
    EXPECT_TRUE(resolveMTPServingForwardCaptureFamily(mtp, 16, MTPVerifierGraphFamily::ExactRows).empty());
    mtp.enabled = true;
    const auto exact = resolveMTPServingForwardCaptureFamily(mtp, 16, MTPVerifierGraphFamily::ExactRows);
    const auto bounded = resolveMTPServingForwardCaptureFamily(mtp, 16, MTPVerifierGraphFamily::BoundedBuckets);
    EXPECT_NO_THROW((void)MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode::Disabled, 3, exact.front()));
    EXPECT_THROW((void)MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode::Disabled, 3, bounded.front()), std::invalid_argument);
    for (int wrong_capacity : {0, 8, 32})
        EXPECT_THROW((void)resolveMTPServingForwardCaptureFamily(mtp, wrong_capacity, MTPVerifierGraphFamily::ExactRows), std::invalid_argument);
    const auto unknown_family = static_cast<MTPVerifierGraphFamily>(99);
    EXPECT_FALSE(mtpVerifierGraphFamilyContains(3, 16, unknown_family));
    EXPECT_THROW((void)mtpVerifierGraphFamilyWidthCount(16, unknown_family), std::invalid_argument);
    EXPECT_THROW((void)resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp, unknown_family), std::invalid_argument);
    EXPECT_THROW((void)resolveMTPVerifierGraphFamily(static_cast<MTPVerifierExecutionRole>(99)), std::invalid_argument);
    EXPECT_THROW((void)resolveMTPVerifierPhysicalWidthPolicy(static_cast<MTPVerifierExecutionRole>(99), MTPDepthPolicyMode::Dynamic), std::invalid_argument);
    EXPECT_THROW((void)resolveMTPVerifierPhysicalWidthPolicy(MTPVerifierExecutionRole::HostedCapturedTransactions, static_cast<MTPDepthPolicyMode>(99)), std::invalid_argument);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 4, 16, static_cast<MTPVerifierPhysicalWidthPolicy>(99)), 0);
    for (int bad_rows : {-1, 0, 17, std::numeric_limits<int>::max()})
        EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, bad_rows, 16, MTPVerifierPhysicalWidthPolicy::ExactLogicalRows), 0);
    for (auto family : {MTPVerifierGraphFamily::BoundedBuckets, MTPVerifierGraphFamily::ExactRows})
        for (int invalid_max : {-1, 0, 1})
        {
            EXPECT_EQ(mtpVerifierGraphFamilyWidthCount(invalid_max, family), 0u);
            EXPECT_EQ(mtpVerifierGraphFamilyWidthAt(0, invalid_max, family), 0);
        }
    mtp.graph_capacity_draft_tokens = std::numeric_limits<int>::max();
    EXPECT_THROW((void)resolveMTPRetainedVerifierWidthCount(mtp, MTPVerifierGraphFamily::ExactRows), std::overflow_error);
}

/** @test Declared complete-parent ownership determines admission and width before capture exists. */
TEST(Test__MTPVerifierPolicy, ExactHostedFamilyExecutionDeclaration)
{
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    for (auto domain : {DeviceGenerationExecutionDomain::NativeDomain, DeviceGenerationExecutionDomain::HostedBoundary})
    {
        const bool native = type == DeviceType::CUDA && domain == DeviceGenerationExecutionDomain::NativeDomain;
        const auto execution = declaredDeviceGenerationExecutionPolicy(type, domain);
        EXPECT_EQ(execution, native ? DeviceGenerationExecutionPolicy::NativeConditionalGraph :
            DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions);
        const auto role = mtpVerifierExecutionRole(execution);
        EXPECT_EQ(role, native ? MTPVerifierExecutionRole::NativeConditionalParent :
            MTPVerifierExecutionRole::HostedCapturedTransactions);
        const auto family = resolveMTPVerifierGraphFamily(role);
        EXPECT_EQ(family, native ? MTPVerifierGraphFamily::BoundedBuckets : MTPVerifierGraphFamily::ExactRows);
        MTPRuntimeConfig retained;
        retained.enabled = false;
        retained.graph_capacity_draft_tokens = 15;
        EXPECT_EQ(resolveMTPRetainedServingForwardModelGraphIdentityCount(retained, family), native ? 10u : 32u);
        for (auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe, MTPDepthPolicyMode::Dynamic})
        {
            const auto width_policy = resolveMTPVerifierPhysicalWidthPolicy(role, mode);
            EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 3, 16, width_policy),
                native ? (mode == MTPDepthPolicyMode::Dynamic ? 16 : 4) : 3);
        }
    }
}

/** @test Unknown declarations cannot become a supported policy or quietly underprice a family. */
TEST(Test__MTPVerifierPolicy, ExactHostedFamilyExecutionDeclarationRejectsUnsupportedInputs)
{
    for (auto domain : {DeviceGenerationExecutionDomain::NativeDomain, DeviceGenerationExecutionDomain::HostedBoundary})
        for (auto type : {DeviceType::CPU, DeviceType::Vulkan, DeviceType::Metal, static_cast<DeviceType>(99)})
        {
            const auto policy = declaredDeviceGenerationExecutionPolicy(type, domain);
            EXPECT_EQ(policy, DeviceGenerationExecutionPolicy::Unsupported);
            EXPECT_THROW((void)mtpVerifierExecutionRole(policy), std::invalid_argument);
        }
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto policy = declaredDeviceGenerationExecutionPolicy(type, static_cast<DeviceGenerationExecutionDomain>(255));
        EXPECT_EQ(policy, DeviceGenerationExecutionPolicy::Unsupported);
        EXPECT_THROW((void)mtpVerifierExecutionRole(policy), std::invalid_argument);
    }
    EXPECT_THROW((void)mtpVerifierExecutionRole(static_cast<DeviceGenerationExecutionPolicy>(255)), std::invalid_argument);
}

/** @test Member-type union is order independent and an authored hosted boundary is sticky. */
TEST(Test__MTPVerifierPolicy, ExactHostedFamilyExecutionDeclarationUnionsActualMembers)
{
    const auto native = DeviceGenerationExecutionDomain::NativeDomain;
    const auto hosted = DeviceGenerationExecutionDomain::HostedBoundary;
    EXPECT_EQ(DeviceGenerationDomainDeclaration{}.domain(), native);
    for (auto first : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm, DeviceType::Vulkan, DeviceType::Metal})
    for (auto second : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm, DeviceType::Vulkan, DeviceType::Metal})
    for (auto third : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm, DeviceType::Vulkan, DeviceType::Metal})
    {
        const auto expected = first == second && second == third ? native : hosted;
        DeviceGenerationDomainDeclaration forward, reverse, explicit_boundary;
        forward.observe(first); forward.observe(second); forward.observe(third);
        reverse.observe(third); reverse.observe(second); reverse.observe(first);
        EXPECT_EQ(forward.domain(), expected);
        EXPECT_EQ(reverse.domain(), expected);
        explicit_boundary.requireHostedBoundary();
        explicit_boundary.observe(first); explicit_boundary.observe(second); explicit_boundary.observe(third);
        EXPECT_EQ(explicit_boundary.domain(), hosted);
        forward.requireHostedBoundary();
        forward.observe(first);
        EXPECT_EQ(forward.domain(), hosted);
    }
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

/** @brief Fully active verifier scopes retain arithmetic across nesting and exceptions. */
TEST(Test__MTPVerifierPolicy, VerifierArithmeticScopeIsIndependentOfRowCounts)
{
    using Scope = ITensorGemm::VerifierKernelModeScope;
    EXPECT_FALSE(Scope::active());
    {
        Scope outer;
        EXPECT_TRUE(Scope::active());
        EXPECT_EQ(Scope::rowsFor(16), nullptr);
        try
        {
            Scope inner(DeviceRowRange::fullyActive(3));
            EXPECT_TRUE(Scope::active());
            EXPECT_NE(Scope::rowsFor(3), nullptr);
            throw std::runtime_error("recording failure");
        }
        catch (const std::runtime_error &) {}
        EXPECT_TRUE(Scope::active());
        EXPECT_EQ(Scope::rowsFor(16), nullptr);
    }
    EXPECT_FALSE(Scope::active());
    EXPECT_EQ(Scope::rowsFor(16), nullptr);
}

/** @brief Independent graph recorders never inherit another thread's verifier contract. */
TEST(Test__MTPVerifierPolicy, VerifierArithmeticScopeIsThreadLocal)
{
    using Scope = ITensorGemm::VerifierKernelModeScope;
    Scope parent(DeviceRowRange::fullyActive(3));
    std::thread other([] {
        EXPECT_FALSE(Scope::active());
        EXPECT_EQ(Scope::rowsFor(16), nullptr);
        {
            Scope inner;
            EXPECT_TRUE(Scope::active());
        }
        EXPECT_FALSE(Scope::active());
    });
    other.join();
    EXPECT_TRUE(Scope::active());
    EXPECT_NE(Scope::rowsFor(3), nullptr);
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

/** @test Capture metadata rejects incomplete owners, capacity overflow and aliases before launch. */
TEST(Test__MTPVerifierPolicy, ShortHalfProjectionBindingsAreBoundedAndNonempty)
{
    using llaminar2::FixedHalfProjectionGroup;
    std::uint16_t input{}, weights[9]{}; float outputs[9]{};
    EXPECT_THROW((void)FixedHalfProjectionGroup::single(nullptr,weights,outputs,1),std::invalid_argument);
    EXPECT_THROW((void)FixedHalfProjectionGroup::single(&input,nullptr,outputs,1),std::invalid_argument);
    EXPECT_THROW((void)FixedHalfProjectionGroup::single(&input,weights,nullptr,1),std::invalid_argument);
    EXPECT_THROW((void)FixedHalfProjectionGroup::single(&input,weights,outputs,0),std::invalid_argument);
    auto group=FixedHalfProjectionGroup::single(&input,weights,outputs,1);
    const auto retained=group;
    EXPECT_THROW(group.append(weights+1,outputs,3),std::invalid_argument);
    EXPECT_THROW(group.append(nullptr,outputs+1,3),std::invalid_argument);
    EXPECT_THROW(group.append(weights+1,outputs+1,-1),std::invalid_argument);
    int total=1;
    for(int i=1;i<8;++i) {
        group.append(weights+i,outputs+i,i*3); total+=i*3;
        EXPECT_EQ(group.size(),i+1); EXPECT_EQ(group.totalColumns(),total);
        EXPECT_EQ(group.columns(i),i*3); EXPECT_EQ(group.columnEnd(i),total);
        EXPECT_EQ(group.weight(i),weights+i); EXPECT_EQ(group.output(i),outputs+i);
    }
    EXPECT_THROW(group.append(weights+8,outputs+8,1),std::length_error);
    EXPECT_EQ(group.size(),8); EXPECT_EQ(group.totalColumns(),total);
    EXPECT_EQ(retained.size(),1); EXPECT_EQ(retained.totalColumns(),1); EXPECT_EQ(retained.input(),&input);
    auto limit=FixedHalfProjectionGroup::single(&input,weights,outputs,std::numeric_limits<int>::max());
    EXPECT_THROW(limit.append(weights+1,outputs+1,1),std::overflow_error);
    EXPECT_EQ(limit.totalColumns(),std::numeric_limits<int>::max()); EXPECT_EQ(limit.size(),1);
}

/** @test Custom floating arguments retain exact typed addresses without device tables or host lifetimes. */
TEST(Test__MTPVerifierPolicy, FixedFloatingProjectionBindingsAreTypedBoundedValueSnapshots)
{
    using llaminar2::FixedFP32ProjectionGroup;
    using llaminar2::FixedFP32x16ProjectionGroup;
    static_assert(!std::is_same_v<FixedFP32ProjectionGroup,FixedFP32x16ProjectionGroup>);
    static_assert(!std::is_convertible_v<FixedFP32ProjectionGroup,FixedFP32x16ProjectionGroup>);
    float inputs[9]{},weights[9]{},outputs[9]{};
    std::uint16_t native[9]{};
    EXPECT_THROW((void)FixedFP32ProjectionGroup::single(nullptr,weights,outputs),std::invalid_argument);
    EXPECT_THROW((void)FixedFP32ProjectionGroup::single(inputs,nullptr,outputs),std::invalid_argument);
    EXPECT_THROW((void)FixedFP32ProjectionGroup::single(inputs,weights,nullptr),std::invalid_argument);
    auto wide=FixedFP32ProjectionGroup::single(inputs,weights,outputs);
    auto narrow=FixedFP32x16ProjectionGroup::single(inputs,native,outputs);
    const auto frozen=wide;
    for (int i=1;i<8;++i) {
        wide.append(inputs+i,weights+i,outputs+i);
        narrow.append(inputs+i,native+i,outputs+i);
    }
    for (int i=0;i<8;++i) {
        EXPECT_EQ(wide.input(i),inputs+i);
        EXPECT_EQ(wide.weight(i),weights+i);
        EXPECT_EQ(wide.output(i),outputs+i);
        EXPECT_EQ(narrow.input(i),inputs+i);
        EXPECT_EQ(narrow.weight(i),native+i);
        EXPECT_EQ(narrow.output(i),outputs+i);
    }
    EXPECT_EQ(wide.size(),8);
    EXPECT_EQ(narrow.size(),8);
    EXPECT_EQ(frozen.size(),1);
    EXPECT_EQ(frozen.weight(0),weights);
    EXPECT_THROW(wide.append(inputs+8,weights+8,outputs+8),std::length_error);
    EXPECT_THROW(narrow.append(inputs+8,native+8,outputs+8),std::length_error);
    auto partial=FixedFP32ProjectionGroup::single(inputs,weights,outputs);
    EXPECT_THROW(partial.append(nullptr,weights,outputs),std::invalid_argument);
    EXPECT_THROW(partial.append(inputs,nullptr,outputs),std::invalid_argument);
    EXPECT_THROW(partial.append(inputs,weights,nullptr),std::invalid_argument);
    EXPECT_EQ(partial.size(),1);
}
