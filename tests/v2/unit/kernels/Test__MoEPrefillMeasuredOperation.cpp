/**
 * @file Test__MoEPrefillMeasuredOperation.cpp
 * @brief Device-free regression for complete-operation MoE measurement identity.
 *
 * The literal installed tables remain unchanged. These tests use their actual
 * keys to prove that a complete FFN measurement cannot certify a split phase,
 * compact column interval, canonical-route publication or selected ownership.
 * The same immutable tag survives ROCm live-route admission without reading
 * counts. This is dispatch provenance coverage, not GPU byte or timing evidence.
 */
#include <gtest/gtest.h>

#include "kernels/common/MoEPrefillProjectionExecution.h"
#include "kernels/rocm/gemm/ROCmMoEProductionPrefillPolicy.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPrefillRoutePolicy.h"
#include "kernels/cuda/gemm/CUDAMoEProductionPrefillPolicy.h"
#include "kernels/cuda/gemm/CUDAMoEGroupedPrefillKernels.h"

#include <array>
#include <vector>

namespace
{
    using Execution = llaminar2::MoEPrefillProjectionExecution;
    using Layout = llaminar2::MoEPrefillOutputLayout;
    using Scope = llaminar2::MoEPrefillRouteScope;
    using Operation = llaminar2::MoEPrefillMeasuredOperation;
    constexpr auto kComplete = Operation::CompleteFullWidthTokenRowsAllRoutes;

    /**
     * @brief Enumerate operations sharing the old table key but not its meaning.
     * @param hidden Original model width used by every arithmetic policy.
     * @return Gate/up, whole down, every TP2/4/8 down interval, canonical and
     *         selected/unproven complete publications, without reading routes.
     */
    std::vector<Operation> unmeasuredOperations(int hidden)
    {
        std::vector<Operation> operations{
            Execution::gateUp(hidden).measuredOperation(Layout::TokenRows, Scope::AllRoutes),
            Execution::gateUp(hidden).measuredOperation(Layout::TokenRows, Scope::SelectedRoutes),
            Execution::down(hidden, 0, hidden).measuredOperation(Layout::TokenRows, Scope::AllRoutes),
            Execution::complete(hidden).measuredOperation(Layout::OriginalRouteRows, Scope::AllRoutes),
            Execution::complete(hidden).measuredOperation(Layout::TokenRows, Scope::SelectedRoutes),
            Execution::complete(hidden).measuredOperation(Layout::TokenRows, Scope::Unspecified)};
        for (int degree : {2, 4, 8})
            for (int ordinal = 0; ordinal < degree; ++ordinal)
                for (auto layout : {Layout::TokenRows, Layout::OriginalRouteRows})
                    operations.push_back(Execution::down(hidden, ordinal * (hidden / degree), hidden / degree)
                                             .measuredOperation(layout, Scope::AllRoutes));
        return operations;
    }
}

/** The actual producer metadata revokes old evidence before failed setup. */
TEST(MoEPrefillMeasuredOperation, GroupingPreparationFailureAndResetRevokePriorOrigin)
{
    llaminar2::MoEPrefillGroupingProvenance provenance;
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    {
        auto publication = provenance.beginAllRoutes();
        EXPECT_EQ(provenance.scope(), Scope::Unspecified);
        std::move(publication).publish();
        EXPECT_EQ(provenance.scope(), Scope::AllRoutes);
        EXPECT_THROW(std::move(publication).publish(), std::logic_error);
    }
    {
        // A production early return abandons this same typed transaction.
        auto failed = provenance.beginSelectedRoutes();
        EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    }
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    try
    {
        auto failed = provenance.beginAllRoutes();
        throw std::runtime_error("injected setup failure before publication");
    }
    catch (const std::runtime_error &) {}
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    {
        auto publication = provenance.beginAllRoutes();
        std::move(publication).publish();
    }
    provenance.invalidate(); // Same operation used by reset/raw/workspace paths.
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::TokenRows, provenance.scope()),
              Operation::Unmeasured);
}

/** Masked-to-unmasked replacement publishes the origin of the new producer. */
TEST(MoEPrefillMeasuredOperation, GroupingReplacementPublishesOnlySuccessfulNewOrigin)
{
    llaminar2::MoEPrefillGroupingProvenance provenance;
    {
        auto selected = provenance.beginSelectedRoutes();
        std::move(selected).publish();
    }
    EXPECT_EQ(provenance.scope(), Scope::SelectedRoutes);
    EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::TokenRows, provenance.scope()),
              Operation::Unmeasured);
    {
        auto all = provenance.beginAllRoutes();
        // The raw count/scan/scatter helper invalidates any stale private map.
        provenance.invalidate();
        std::move(all).publish(); // Its enclosing complete producer succeeded.
    }
    EXPECT_EQ(provenance.scope(), Scope::AllRoutes);
    EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::TokenRows, provenance.scope()), kComplete);
    {
        auto failed = provenance.beginSelectedRoutes();
    }
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
}

/** Overlapping or consumed tokens cannot restore an obsolete all-route origin. */
TEST(MoEPrefillMeasuredOperation, GroupingPreparationRejectsOverlapAndConsumedReuse)
{
    llaminar2::MoEPrefillGroupingProvenance provenance;
    auto all = provenance.beginAllRoutes();
    EXPECT_THROW((void)provenance.beginSelectedRoutes(), std::logic_error);
    EXPECT_THROW((void)provenance.beginAllRoutes(), std::logic_error);
    provenance.invalidate(); // The raw helper may invalidate within its producer.
    EXPECT_THROW((void)provenance.beginSelectedRoutes(), std::logic_error);
    std::move(all).publish(); // Rejected overlap did not steal this authority.
    EXPECT_EQ(provenance.scope(), Scope::AllRoutes);
    {
        auto selected = provenance.beginSelectedRoutes();
        EXPECT_THROW(std::move(all).publish(), std::logic_error);
        EXPECT_THROW((void)provenance.beginAllRoutes(), std::logic_error);
        std::move(selected).publish(); // A consumed token did not release B.
        EXPECT_EQ(provenance.scope(), Scope::SelectedRoutes);
    }
    {
        auto abandoned = provenance.beginAllRoutes();
        EXPECT_THROW((void)provenance.beginSelectedRoutes(), std::logic_error);
    }
    auto after_failure = provenance.beginSelectedRoutes();
    std::move(after_failure).publish(); // Failure released the active authority.
    EXPECT_EQ(provenance.scope(), Scope::SelectedRoutes);
}

/** Existing complete rows require every static operation fact simultaneously. */
TEST(MoEPrefillMeasuredOperation, QualifiesOnlyCompleteFullWidthTokenRowsAllRoutes)
{
    for (int hidden : {2048, 2560, 3072})
    {
        EXPECT_EQ(Execution::complete(hidden).measuredOperation(Layout::TokenRows, Scope::AllRoutes), kComplete);
        for (auto operation : unmeasuredOperations(hidden))
            EXPECT_EQ(operation, Operation::Unmeasured);
    }
    EXPECT_THROW((void)Execution::complete(2048).measuredOperation(
                     static_cast<Layout>(255), Scope::AllRoutes), std::invalid_argument);
    EXPECT_THROW((void)Execution::complete(2048).measuredOperation(
                     Layout::TokenRows, static_cast<Scope>(255)), std::invalid_argument);
}

/** Every ROCm installed config survives complete selection byte-for-field. */
TEST(MoEPrefillMeasuredOperation, ROCmInstalledCompleteRowsRetainEveryConfig)
{
    using namespace llaminar2::rocm;
    for (const auto &entry : generated::kROCmMoEProductionPrefillOverlayEntries)
    {
        generated::ROCmMoEProductionPrefillOverlayConfig selected{};
        ASSERT_TRUE(selectROCmMoEProductionPrefillOverlayForOperation(
            kComplete, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
            entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
        EXPECT_EQ(selected.gateup_tile_m, entry.config.gateup_tile_m);
        EXPECT_EQ(selected.gateup_tile_n, entry.config.gateup_tile_n);
        EXPECT_EQ(selected.down_tile_m, entry.config.down_tile_m);
        EXPECT_EQ(selected.down_tile_n, entry.config.down_tile_n);
    }
}

/** A failed exact lookup must not contaminate the caller's generic selection. */
TEST(MoEPrefillMeasuredOperation, ROCmSplitAndSelectedOperationsCannotInheritExactRows)
{
    using namespace llaminar2::rocm;
    for (const auto &entry : generated::kROCmMoEProductionPrefillOverlayEntries)
        for (auto operation : unmeasuredOperations(entry.hidden_size))
        {
            generated::ROCmMoEProductionPrefillOverlayConfig selected{3, 17, 5, 19};
            EXPECT_FALSE(selectROCmMoEProductionPrefillOverlayForOperation(
                operation, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
                entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
            EXPECT_EQ(selected.gateup_tile_m, 3);
            EXPECT_EQ(selected.gateup_tile_n, 17);
            EXPECT_EQ(selected.down_tile_m, 5);
            EXPECT_EQ(selected.down_tile_n, 19);
        }
}

/** Every CUDA ordered exact width, including non-generic widths, is retained. */
TEST(MoEPrefillMeasuredOperation, CUDAInstalledCompleteRowsRetainEveryConfig)
{
    using namespace llaminar2::cuda;
    for (const auto &entry : generated::kCUDAMoEProductionPrefillOverlayEntries)
    {
        int selected = -71;
        ASSERT_TRUE(selectCUDAMoEProductionPrefillOverlayForOperation(
            kComplete, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
            entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
        EXPECT_EQ(selected, entry.gateup_tile_n);
    }
}

/** Same old key with a different operation is no longer measured CUDA evidence. */
TEST(MoEPrefillMeasuredOperation, CUDASplitAndSelectedOperationsCannotInheritExactRows)
{
    using namespace llaminar2::cuda;
    for (const auto &entry : generated::kCUDAMoEProductionPrefillOverlayEntries)
        for (auto operation : unmeasuredOperations(entry.hidden_size))
        {
            int selected = -71;
            EXPECT_FALSE(selectCUDAMoEProductionPrefillOverlayForOperation(
                operation, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
                entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
            EXPECT_EQ(selected, -71);
        }
}

/** The separate inline IMMA regimes obey the same qualification as ordered rows. */
TEST(MoEPrefillMeasuredOperation, CUDAInlineImmaExceptionsRequireCompleteOperation)
{
    using namespace llaminar2::cuda::moe;
    for (const auto &regime : kQwen35BGroupedImmaCodebookRegimes)
        for (int rows : {64, 512, 768, 1024, 4096})
            for (int down : {regime.down_codebook,
                             regime.down_codebook == kQwenIQ4ExecutionCodebook ? 4 : regime.down_codebook})
            {
                const auto complete = selectGroupedImmaLaunchPolicy(
                    kComplete, regime.gate_up_codebook, down, rows, 2048, 512, 256, 8);
                EXPECT_TRUE(complete.exact_overlay);
                EXPECT_EQ(complete.gate_up_columns, rows <= regime.columns32_max_rows
                                                       ? GroupedImmaColumns::Columns32
                                                       : GroupedImmaColumns::Columns64);
                for (auto operation : unmeasuredOperations(2048))
                {
                    const auto selected = selectGroupedImmaLaunchPolicy(
                        operation, regime.gate_up_codebook, down, rows, 2048, 512, 256, 8);
                    EXPECT_FALSE(selected.exact_overlay);
                    EXPECT_EQ(selected.gate_up_columns, GroupedImmaColumns::Columns32);
                    EXPECT_EQ(selected.down_columns, GroupedImmaColumns::Columns32);
                    EXPECT_EQ(selected.gate_up_schedule, GroupedImmaGateUpSchedule::PairedProjections);
                }
            }
}

/** Sparse endpoint exceptions belong to the measured operation, not H/F/M alone. */
TEST(MoEPrefillMeasuredOperation, ROCmRouteExceptionsRequireCompleteOperation)
{
    using namespace llaminar2::rocm;
    for (const auto &entry : kROCmMoEGroupedPrefillExactRoutePolicies)
    {
        EXPECT_TRUE(selectROCmMoEGroupedPrefillRouteStrategy(entry.key).exact);
        EXPECT_EQ(selectROCmMoEGroupedPrefillRouteStrategy(entry.key).strategy, entry.strategy);
        for (auto operation : unmeasuredOperations(entry.key.hidden_size))
        {
            auto key = entry.key;
            key.operation = operation;
            const auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
            EXPECT_FALSE(decision.exact);
            EXPECT_EQ(decision.strategy, ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
        }
    }
}

/** Device live-row projection preserves qualification rather than minting it. */
TEST(MoEPrefillMeasuredOperation, ROCmLiveAdmissionCannotResurrectExactProvenance)
{
    using namespace llaminar2::rocm;
    int unread_device_word = -919;
    auto key = kROCmMoEGroupedPrefillExactRoutePolicies.front().key;
    key.rows = 64;
    const ROCmMoEGroupedRouteAdmission complete(key, &unread_device_word, &unread_device_word);
    EXPECT_EQ(complete.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned), 32);
    for (auto operation : unmeasuredOperations(key.hidden_size))
    {
        key.operation = operation;
        const ROCmMoEGroupedRouteAdmission selected(key, &unread_device_word, &unread_device_word);
        EXPECT_TRUE(selected.maySelect(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned));
        EXPECT_TRUE(selected.maySelect(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled));
        EXPECT_EQ(selected.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned), 8);
        EXPECT_EQ(selected.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled), 64);
        for (int slots = 0; slots <= 64; ++slots)
            EXPECT_EQ(selected.strategyForSlots(slots), slots <= 8
                          ? ROCmMoEGroupedPrefillRouteStrategy::RouteOwned
                          : ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    }
    EXPECT_EQ(unread_device_word, -919);
}

/** Unseen complete geometries and source families keep their existing exact miss. */
TEST(MoEPrefillMeasuredOperation, UnseenShapesAndFamiliesRemainExactMisses)
{
    llaminar2::rocm::generated::ROCmMoEProductionPrefillOverlayConfig rocm{3, 17, 5, 19};
    int cuda = -71;
    for (auto operation : {kComplete, Operation::Unmeasured})
    {
        EXPECT_FALSE(llaminar2::rocm::selectROCmMoEProductionPrefillOverlayForOperation(
            operation, 5, 19, 2560, 640, 512, 10, 512, rocm));
        EXPECT_FALSE(llaminar2::cuda::selectCUDAMoEProductionPrefillOverlayForOperation(
            operation, 5, 19, 2560, 640, 512, 10, 512, cuda));
        EXPECT_FALSE(llaminar2::rocm::selectROCmMoEProductionPrefillOverlayForOperation(
            operation, 23, 23, 2048, 512, 256, 8, 512, rocm));
        EXPECT_FALSE(llaminar2::cuda::selectCUDAMoEProductionPrefillOverlayForOperation(
            operation, 23, 23, 2048, 512, 256, 8, 512, cuda));
    }
    EXPECT_EQ(cuda, -71);
    EXPECT_EQ(rocm.gateup_tile_n, 17);
}
