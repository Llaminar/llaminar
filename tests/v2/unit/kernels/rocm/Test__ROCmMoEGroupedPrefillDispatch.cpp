/**
 * @file Test__ROCmMoEGroupedPrefillDispatch.cpp
 * @brief CPU-only structural tests for generated ROCm grouped-MoE prefill policy.
 *
 * These tests deliberately perform no GPU work. They validate the generated
 * decision surface consumed when a ROCm graph is captured: every advertised
 * execution codebook, projection role, aspect-ratio bucket, and M anchor must
 * resolve to a compiled tile, while unknown inputs must fail instead of falling
 * back to a legacy default.
 * Whole-expert dot admission additionally proves exclusive Dot4/WMMA writers
 * across unequal physical tiles, source aliases, CPU-promoted payloads and
 * sliced down projections without dereferencing device storage.
 */

#include <gtest/gtest.h>
#include "kernels/rocm/gemm/ROCmMoEGroupedPopulationPublication.h"
#include "kernels/common/MoEPrefillProjectionExecution.h"

/** @brief Output sharding keeps the full matrix's immutable arithmetic key. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, ProjectionTransactionsKeepCanonicalGeometry)
{
    using Execution = llaminar2::MoEPrefillProjectionExecution;
    const auto complete = Execution::complete(2048);
    EXPECT_TRUE(complete.executesGateUp());
    EXPECT_TRUE(complete.executesDown());
    EXPECT_EQ(complete.columnCount(), 2048);
    const auto producer = Execution::gateUp(2048);
    EXPECT_TRUE(producer.executesGateUp());
    EXPECT_FALSE(producer.executesDown());
    for (const int degree : {1, 2, 4, 8})
        for (int shard = 0; shard < degree; ++shard)
        {
            const auto consumer = Execution::down(2048, shard * (2048 / degree), 2048 / degree);
            EXPECT_FALSE(consumer.executesGateUp());
            EXPECT_TRUE(consumer.executesDown());
            EXPECT_EQ(consumer.modelColumns(), 2048);
            EXPECT_EQ(consumer.firstColumn(), shard * (2048 / degree));
            EXPECT_EQ(consumer.columnCount(), 2048 / degree);
        }
    EXPECT_THROW((void)Execution::complete(0), std::invalid_argument);
    // Floating matrices support tails; only quantized launch admission owns
    // block alignment. The phase contract must not restrict FP32 geometry.
    EXPECT_EQ(Execution::gateUp(33).modelColumns(), 33);
    EXPECT_EQ(Execution::down(33, 16, 17).columnCount(), 17);
    EXPECT_THROW((void)Execution::gateUp(-1), std::invalid_argument);
    EXPECT_THROW((void)Execution::down(2048, -1, 128), std::invalid_argument);
    EXPECT_THROW((void)Execution::down(2048, 2048, 1), std::invalid_argument);
    EXPECT_THROW((void)Execution::down(2048, 1024, 0), std::invalid_argument);
    EXPECT_THROW((void)Execution::down(2048, 1024, 1025), std::invalid_argument);
    EXPECT_THROW((void)Execution::down(2048, 1, 2147483647), std::invalid_argument);
}

#include "kernels/rocm/gemm/ROCmMoEGroupedPrefillDispatchGenerated.inc"
#include "kernels/rocm/gemm/ROCmMoEGroupedPrefillRoutePolicy.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedDotAdmission.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{
    using llaminar2::rocm::generated::ROCmMoEGroupedPrefillConfig;
    using llaminar2::rocm::generated::kROCmMoEGroupedPrefillEntries;
    using llaminar2::rocm::generated::kROCmMoEGroupedPrefillMAnchors;
    using llaminar2::rocm::generated::rocmMoEGroupedPrefillMAnchor;
    using llaminar2::rocm::generated::rocmMoEGroupedPrefillRatioExponent;
    using llaminar2::rocm::generated::selectROCmMoEGroupedPrefillGenerated;

    constexpr std::array<uint8_t, 16> kExecutionCodebooks = {
        0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19};

    /** Representative N:K dimensions for ratio exponents -3 through +3. */
    constexpr std::array<std::pair<int, int>, 7> kRatioShapes = {{
        {256, 2048},
        {256, 1024},
        {256, 512},
        {256, 256},
        {512, 256},
        {1024, 256},
        {2048, 256},
    }};

    bool isCompiledTile(const ROCmMoEGroupedPrefillConfig &config)
    {
        const bool valid_m =
            config.tile_m == 4 || config.tile_m == 8 ||
            config.tile_m == 12 || config.tile_m == 16;
        const bool valid_n =
            config.tile_n == 64 || config.tile_n == 128 ||
            config.tile_n == 256;
        return valid_m && valid_n;
    }
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, CompletePolicyCartesianProductResolves)
{
    constexpr size_t expected_entries =
        kExecutionCodebooks.size() * 2u * kRatioShapes.size() *
        (sizeof(kROCmMoEGroupedPrefillMAnchors) /
         sizeof(kROCmMoEGroupedPrefillMAnchors[0]));
    static_assert(
        sizeof(kROCmMoEGroupedPrefillEntries) /
                sizeof(kROCmMoEGroupedPrefillEntries[0]) ==
            expected_entries);

    for (uint8_t codebook : kExecutionCodebooks)
    {
        for (uint8_t role : {uint8_t{0}, uint8_t{1}})
        {
            for (size_t ratio_index = 0; ratio_index < kRatioShapes.size(); ++ratio_index)
            {
                const auto [n, k] = kRatioShapes[ratio_index];
                ASSERT_EQ(
                    rocmMoEGroupedPrefillRatioExponent(n, k),
                    static_cast<int>(ratio_index) - 3);
                for (uint16_t m : kROCmMoEGroupedPrefillMAnchors)
                {
                    ROCmMoEGroupedPrefillConfig config{};
                    ASSERT_TRUE(selectROCmMoEGroupedPrefillGenerated(
                        codebook, role, m, n, k, config))
                        << "codebook=" << static_cast<int>(codebook)
                        << " role=" << static_cast<int>(role)
                        << " ratio=" << static_cast<int>(ratio_index) - 3
                        << " M=" << m;
                    EXPECT_TRUE(isCompiledTile(config));
                }
            }
        }
    }
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, RatioBucketsUseNearestLog2Boundaries)
{
    // sqrt(2), sqrt(8), and sqrt(32) are the exact boundaries between
    // neighboring powers of two. Integer dimensions are checked through their
    // squares, avoiding floating-point rounding in graph-capture policy code.
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(141, 100), 0);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(142, 100), 1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(282, 100), 1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(283, 100), 2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(565, 100), 2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(566, 100), 3);

    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 141), 0);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 142), -1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 282), -1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 283), -2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 565), -2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 566), -3);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, NearestMAnchorTiesPreferSmallerBucket)
{
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(1), 12);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(14), 12);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(20), 16);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(28), 24);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(192), 128);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(4096), 256);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, UnsupportedPolicyHardFailsWithZeroConfig)
{
    ROCmMoEGroupedPrefillConfig unknown_codebook{};
    EXPECT_FALSE(selectROCmMoEGroupedPrefillGenerated(
        /*codebook=*/1,
        /*role=*/0,
        /*m=*/32,
        /*n=*/512,
        /*k=*/2048,
        unknown_codebook));
    EXPECT_EQ(unknown_codebook.tile_m, 0);
    EXPECT_EQ(unknown_codebook.tile_n, 0);

    ROCmMoEGroupedPrefillConfig invalid_role{};
    EXPECT_FALSE(selectROCmMoEGroupedPrefillGenerated(
        /*codebook=*/13,
        /*role=*/2,
        /*m=*/32,
        /*n=*/512,
        /*k=*/2048,
        invalid_role));
    EXPECT_EQ(invalid_role.tile_m, 0);
    EXPECT_EQ(invalid_role.tile_n, 0);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch,
     Qwen35_122BSparseEndpointUsesMeasuredRouteOwnedBuckets)
{
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteKey;
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteStrategy;
    using llaminar2::rocm::selectROCmMoEGroupedPrefillRouteStrategy;

    ROCmMoEGroupedPrefillRouteKey key{
        .gateup_codebook = 19, // NativeQuantised Q8_0 execution id.
        .down_codebook = 19,
        .hidden_size = 3072,
        .expert_width = 1024,
        .expert_count = 256,
        .top_k = 1,
        .rows = 16,
        .operation = llaminar2::MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes,
    };
    for (const int rows : {16, 32})
    {
        key.rows = rows;
        const auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
        EXPECT_EQ(
            decision.strategy,
            ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
        EXPECT_TRUE(decision.exact);
    }

    // Adjacent untrained identities must not inherit the measured exception.
    key.rows = 64;
    auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    EXPECT_FALSE(decision.exact);

    key.rows = 16;
    key.top_k = 2;
    decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    EXPECT_FALSE(decision.exact);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch,
     GenericRoutePolicyIsTotalAndRejectsMalformedGeometry)
{
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteKey;
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteStrategy;
    using llaminar2::rocm::selectROCmMoEGroupedPrefillRouteStrategy;

    ROCmMoEGroupedPrefillRouteKey key{
        .gateup_codebook = 13,
        .down_codebook = 4,
        .hidden_size = 2048,
        .expert_width = 512,
        .expert_count = 256,
        .top_k = 8,
        .rows = 8,
    };
    auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
    EXPECT_FALSE(decision.exact);

    key.rows = 9;
    decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    EXPECT_FALSE(decision.exact);

    key.rows = 0;
    decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::Invalid);
    EXPECT_FALSE(decision.valid());
}

/** A retained envelope must use its live route publication, not its capacity. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, LiveRouteAdmissionCoversEveryFormatAndSlot)
{
    using namespace llaminar2::rocm;
    int unread_device_storage = -999;
    for (uint8_t gateup : kExecutionCodebooks)
    {
        for (uint8_t down : kExecutionCodebooks)
        {
            for (int top_k : {1, 2, 8, 16})
            {
                const ROCmMoEGroupedPrefillRouteKey key{
                    .gateup_codebook = gateup, .down_codebook = down,
                    .hidden_size = 2048, .expert_width = 512,
                    .expert_count = 256, .top_k = top_k, .rows = 32};
                const ROCmMoEGroupedRouteAdmission admission(
                    key, &unread_device_storage, &unread_device_storage);
                EXPECT_TRUE(admission.maySelect(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned));
                EXPECT_TRUE(admission.maySelect(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled));
                EXPECT_EQ(admission.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned),
                          8 * top_k);
                EXPECT_EQ(admission.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled),
                          key.rows * top_k);
                for (int slots = 0; slots <= key.rows * top_k; ++slots)
                {
                    auto live = key;
                    live.rows = slots == 0 ? 1 : 1 + (slots - 1) / top_k;
                    EXPECT_EQ(admission.strategyForSlots(slots),
                              selectROCmMoEGroupedPrefillRouteStrategy(live).strategy);
                    EXPECT_LE(slots, admission.maximumSlotsFor(admission.strategyForSlots(slots)));
                }
                EXPECT_EQ(admission.strategyForSlots(-1), ROCmMoEGroupedPrefillRouteStrategy::Invalid);
                EXPECT_EQ(admission.strategyForSlots(key.rows * top_k + 1),
                          ROCmMoEGroupedPrefillRouteStrategy::Invalid);
            }
        }
    }
    EXPECT_EQ(unread_device_storage, -999)
        << "capture-time admission must not read or rewrite the device publication";
}

/** Exact exceptions remain exact even when a wider graph contains their work. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, LiveAdmissionPreservesExactKeysAndRejectsMissingOwners)
{
    using namespace llaminar2::rocm;
    int storage = 0;
    ROCmMoEGroupedPrefillRouteKey key{
        .gateup_codebook = 19, .down_codebook = 19,
        .hidden_size = 3072, .expert_width = 1024,
        .expert_count = 256, .top_k = 1, .rows = 64,
        .operation = llaminar2::MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes};
    const ROCmMoEGroupedRouteAdmission admission(key, &storage, &storage);
    EXPECT_EQ(admission.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned), 32);
    EXPECT_EQ(admission.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled), 64);
    EXPECT_EQ(admission.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::Invalid), 0);
    for (int rows : {16, 32})
        EXPECT_EQ(admission.strategyForSlots(rows), ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
    for (int rows : {9, 15, 17, 31, 33, 64})
        EXPECT_EQ(admission.strategyForSlots(rows), ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    EXPECT_THROW((ROCmMoEGroupedRouteAdmission(key, nullptr, &storage)), std::invalid_argument);
    EXPECT_THROW((ROCmMoEGroupedRouteAdmission(key, &storage, nullptr)), std::invalid_argument);
    key.rows = INT32_MAX;
    key.top_k = 8;
    EXPECT_THROW((ROCmMoEGroupedRouteAdmission(key, &storage, &storage)), std::invalid_argument);
    key.rows = 0;
    EXPECT_THROW((ROCmMoEGroupedRouteAdmission(key, &storage, &storage)), std::invalid_argument);
}

namespace
{
    /**
     * @brief Author metadata-only projection identity; admission must not read payload storage.
     * @param physical Actual prepared decoder, independent of arithmetic provenance.
     * @param source Original source codebook, authoritative when present is true.
     * @param n Physical output columns. @param k Contraction width.
     * @param present Whether source provenance belongs to this live descriptor.
     * @return A descriptor containing only the fields owned by dot admission.
     */
    constexpr llaminar2::DeviceNativeQuantisedMatrixDesc dotAdmissionDescriptor(
        uint8_t physical = 19, uint8_t source = 19, int n = 96, int k = 256, bool present = true)
    {
        return {.n = n, .k = k, .blocks_per_row = static_cast<uint32_t>(k / 32),
            .codebook_id = physical, .source_codebook_id = source,
            .source_identity_present = static_cast<uint8_t>(present)};
    }

    /** @param codebook Reviewed mask coordinate. @return Its exact unsigned mask bit. */
    constexpr uint32_t dotAdmissionBit(uint8_t codebook) { return uint32_t{1} << codebook; }
}

/** @test Default, forced-native and format-restricted envelopes retain every reachable producer. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, DotAdmissionRetainsExactlyTheReachableFamilies)
{
    using namespace llaminar2::rocm;
    const auto desc = dotAdmissionDescriptor();
    const MoEGroupedDotAdmission ordered;
    EXPECT_FALSE(ordered.mayUseNative());
    EXPECT_TRUE(ordered.mayUseOrdered());
    EXPECT_FALSE(ordered.binds(MoEPrefillProjectionRole::GateUp,96,256));
    EXPECT_EQ(ordered.select(desc,31),MoEGroupedDotProducer::OrderedDot4);

    const auto native = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::EveryExpert);
    EXPECT_TRUE(native.mayUseNative());
    EXPECT_FALSE(native.mayUseOrdered()) << "Canonical Q8 alias normalization must preserve a total source mask";
    for (const int rows : {1,4,8,12,16,17,31})
        EXPECT_EQ(native.select(desc,rows),MoEGroupedDotProducer::Wmma);

    const auto physical_only = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::EveryExpert,dotAdmissionBit(19));
    const auto source_only = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::EveryExpert,UINT32_MAX,dotAdmissionBit(19));
    EXPECT_TRUE(physical_only.mayUseOrdered());
    EXPECT_TRUE(source_only.mayUseOrdered());
    EXPECT_EQ(physical_only.select(desc,31),MoEGroupedDotProducer::Wmma);
    EXPECT_EQ(source_only.select(desc,31),MoEGroupedDotProducer::Wmma);
    EXPECT_EQ(physical_only.select(dotAdmissionDescriptor(7,19),31),MoEGroupedDotProducer::OrderedDot4);
    EXPECT_EQ(source_only.select(dotAdmissionDescriptor(19,7),31),MoEGroupedDotProducer::OrderedDot4);
}

/** @test Whole-count boundaries give unequal Dot4/WMMA tiles one writer for every output cell. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, WholeExpertFloorsHaveNoTileBoundaryHolesOrDoubleWriters)
{
    using namespace llaminar2::rocm;
    struct FloorCase { MoEGroupedWmmaRowFloor floor; int threshold; };
    constexpr std::array<FloorCase,4> floors{{
        {MoEGroupedWmmaRowFloor::Four,4}, {MoEGroupedWmmaRowFloor::Eight,8},
        {MoEGroupedWmmaRowFloor::Twelve,12}, {MoEGroupedWmmaRowFloor::Sixteen,16}}};
    constexpr int columns = 96, maximum_rows = 31;
    const auto desc = dotAdmissionDescriptor();
    for (const auto [floor,threshold] : floors)
    {
        const auto admission = MoEGroupedDotAdmission::nativeCandidate(
            MoEPrefillProjectionRole::GateUp,columns,columns,256,floor);
        ASSERT_TRUE(admission.mayUseNative());
        ASSERT_TRUE(admission.mayUseOrdered());
        for (const int rows : {1,threshold-1,threshold,threshold+1,17,31})
            for (const int ordered_m : {4,8,12,16})
                for (const int ordered_n : {64,128,256})
                    for (const int native_n : {32,64})
                    {
                        SCOPED_TRACE(::testing::Message() << "floor=" << threshold << " whole_rows=" << rows
                            << " Dot4=" << ordered_m << 'x' << ordered_n << " WMMA=16x" << native_n);
                        const auto expected = rows >= threshold ? MoEGroupedDotProducer::Wmma : MoEGroupedDotProducer::OrderedDot4;
                        ASSERT_EQ(admission.select(desc,rows),expected);
                        std::array<uint8_t,maximum_rows*columns> writers{};
                        for (const bool native : {false,true})
                        {
                            // Both independently tiled families borrow the captured value
                            // and the SAME whole-expert count, including their final tail.
                            const auto captured = admission;
                            const int tile_m = native ? 16 : ordered_m;
                            const int tile_n = native ? native_n : ordered_n;
                            for (int first_row = 0; first_row < rows; first_row += tile_m)
                                for (int first_column = 0; first_column < columns; first_column += tile_n)
                                {
                                    const auto producer = captured.select(desc,rows);
                                    ASSERT_EQ(producer,expected);
                                    if ((producer == MoEGroupedDotProducer::Wmma) != native) continue;
                                    for (int row = first_row; row < rows && row < first_row+tile_m; ++row)
                                        for (int column = first_column; column < columns && column < first_column+tile_n; ++column)
                                            ++writers[row*columns+column];
                                }
                        }
                        for (int row = 0; row < maximum_rows; ++row)
                            for (int column = 0; column < columns; ++column)
                                ASSERT_EQ(writers[row*columns+column],row < rows ? 1u : 0u)
                                    << "row=" << row << " column=" << column;
                    }
        // Empty/corrupt live counts are rejected before either writer is selected;
        // restoring the complete count must recover the same immutable decision.
        EXPECT_EQ(admission.select(desc,0),MoEGroupedDotProducer::Invalid);
        EXPECT_EQ(admission.select(desc,-1),MoEGroupedDotProducer::Invalid);
        EXPECT_EQ(admission.select(desc,31),MoEGroupedDotProducer::Wmma);
    }
}

/** @test Q8 source aliases share arithmetic eligibility without changing the physical mask. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, DotAdmissionNormalizesOnlyQ8SourceAliases)
{
    using namespace llaminar2::rocm;
    for (const uint8_t source_mask : {uint8_t{19},uint8_t{20},uint8_t{21}})
    {
        const auto admission = MoEGroupedDotAdmission::nativeCandidate(
            MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::EveryExpert,
            dotAdmissionBit(19),dotAdmissionBit(source_mask));
        EXPECT_TRUE(admission.mayUseOrdered());
        for (const uint8_t source : {uint8_t{19},uint8_t{20},uint8_t{21}})
            EXPECT_EQ(admission.select(dotAdmissionDescriptor(19,source),17),MoEGroupedDotProducer::Wmma);
        // An absent provenance flag uses the actual canonical physical codebook.
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(19,255,96,256,false),17),MoEGroupedDotProducer::Wmma);
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(20,20),17),MoEGroupedDotProducer::OrderedDot4);
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(21,21),17),MoEGroupedDotProducer::OrderedDot4);
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(19,7),17),MoEGroupedDotProducer::OrderedDot4);
    }
    const auto mixed = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::EveryExpert,
        dotAdmissionBit(19)|dotAdmissionBit(23),dotAdmissionBit(20)|dotAdmissionBit(21)|dotAdmissionBit(7));
    EXPECT_EQ(mixed.select(dotAdmissionDescriptor(19,21),31),MoEGroupedDotProducer::Wmma);
    EXPECT_EQ(mixed.select(dotAdmissionDescriptor(23,7),31),MoEGroupedDotProducer::Wmma);
}

/** @test CPU-promoted physical23 eligibility requires both decoder and compact source provenance. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, DotAdmissionKeepsPromotedPhysicalAndSourceMasksSeparate)
{
    using namespace llaminar2::rocm;
    const auto admission = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::Down,2048,512,512,MoEGroupedWmmaRowFloor::Eight,
        dotAdmissionBit(23),dotAdmissionBit(7));
    const auto promoted = dotAdmissionDescriptor(23,7,512,512);
    EXPECT_EQ(admission.select(promoted,7),MoEGroupedDotProducer::OrderedDot4);
    EXPECT_EQ(admission.select(promoted,8),MoEGroupedDotProducer::Wmma);
    EXPECT_EQ(admission.select(promoted,31),MoEGroupedDotProducer::Wmma);
    EXPECT_EQ(admission.select(dotAdmissionDescriptor(7,7,512,512),31),MoEGroupedDotProducer::OrderedDot4);
    EXPECT_EQ(admission.select(dotAdmissionDescriptor(23,5,512,512),31),MoEGroupedDotProducer::OrderedDot4);
    EXPECT_EQ(admission.select(dotAdmissionDescriptor(23,7,512,512,false),31),MoEGroupedDotProducer::OrderedDot4);
    EXPECT_EQ(admission.select(dotAdmissionDescriptor(23,23,512,512),31),MoEGroupedDotProducer::OrderedDot4);
}

/** @test IQ4_NL and source-scaled IQ4_XS keep distinct physical writers despite shared source4. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, DotAdmissionNeverAliasesPhysicalIQ4Decoders)
{
    using namespace llaminar2::rocm;
    const auto nl = dotAdmissionDescriptor(4,4);
    const auto xs = dotAdmissionDescriptor(24,4);
    for (const uint8_t physical : {uint8_t{4},uint8_t{24}})
    {
        const auto admission = MoEGroupedDotAdmission::nativeCandidate(
            MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::EveryExpert,
            dotAdmissionBit(physical),dotAdmissionBit(4));
        EXPECT_EQ(admission.select(nl,17),physical == 4 ? MoEGroupedDotProducer::Wmma : MoEGroupedDotProducer::OrderedDot4);
        EXPECT_EQ(admission.select(xs,17),physical == 24 ? MoEGroupedDotProducer::Wmma : MoEGroupedDotProducer::OrderedDot4);
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(24,4,96,256,false),17),MoEGroupedDotProducer::OrderedDot4);
    }
    const auto both = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::EveryExpert,
        dotAdmissionBit(4)|dotAdmissionBit(24),dotAdmissionBit(4));
    EXPECT_EQ(both.select(nl,17),MoEGroupedDotProducer::Wmma);
    EXPECT_EQ(both.select(xs,17),MoEGroupedDotProducer::Wmma);
}

/** @test Gate/up and sliced down bind exact physical N/K while retaining the original full width. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, DotAdmissionBindsExactRoleAndSlicedDownGeometry)
{
    using namespace llaminar2::rocm;
    using Execution = llaminar2::MoEPrefillProjectionExecution;
    const auto gate_up = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::GateUp,512,512,2048,MoEGroupedWmmaRowFloor::Four);
    EXPECT_TRUE(gate_up.binds(MoEPrefillProjectionRole::GateUp,512,2048));
    EXPECT_FALSE(gate_up.binds(MoEPrefillProjectionRole::Down,512,2048));
    EXPECT_FALSE(gate_up.binds(MoEPrefillProjectionRole::GateUp,256,2048));
    EXPECT_FALSE(gate_up.binds(MoEPrefillProjectionRole::GateUp,512,512));
    EXPECT_EQ(gate_up.sourceOutputColumns(),512);
    for (const int first_column : {0,512,1024,1536})
    {
        const auto execution = Execution::down(2048,first_column,512);
        const auto down = MoEGroupedDotAdmission::nativeCandidate(
            MoEPrefillProjectionRole::Down,execution.modelColumns(),execution.columnCount(),512,
            MoEGroupedWmmaRowFloor::Four);
        EXPECT_EQ(down.sourceOutputColumns(),2048);
        EXPECT_TRUE(down.binds(MoEPrefillProjectionRole::Down,512,512));
        EXPECT_FALSE(down.binds(MoEPrefillProjectionRole::GateUp,512,512));
        EXPECT_FALSE(down.binds(MoEPrefillProjectionRole::Down,2048,512));
        EXPECT_FALSE(down.binds(MoEPrefillProjectionRole::Down,512,2048));
        EXPECT_EQ(down.select(dotAdmissionDescriptor(19,19,512,512),17),MoEGroupedDotProducer::Wmma);
        EXPECT_EQ(down.select(dotAdmissionDescriptor(19,19,2048,512),17),MoEGroupedDotProducer::Invalid);
    }
}

/** @test Incomplete capture bindings fail before they can advertise either native producer. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, DotAdmissionRejectsMalformedCaptureGeometryMasksAndFloors)
{
    using namespace llaminar2::rocm;
    using Role = MoEPrefillProjectionRole;
    using Floor = MoEGroupedWmmaRowFloor;
    for (const auto role : {Role::GateUp,Role::Down})
    {
        for (const int source_n : {0,-1})
            EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(role,source_n,96,256,Floor::Four),std::invalid_argument);
        for (const int physical_n : {0,-1,97})
            EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(role,96,physical_n,256,Floor::Four),std::invalid_argument);
        for (const int k : {0,-32,1,31,33,255})
            EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(role,96,96,k,Floor::Four),std::invalid_argument);
        EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(role,96,96,256,Floor::Four,0),std::invalid_argument);
        EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(role,96,96,256,Floor::Four,UINT32_MAX,0),std::invalid_argument);
        for (const uint8_t floor : {uint8_t{1},uint8_t{3},uint8_t{5},uint8_t{17},uint8_t{255}})
        {
            EXPECT_FALSE(validMoEGroupedWmmaRowFloor(static_cast<Floor>(floor)));
            EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(role,96,96,256,static_cast<Floor>(floor)),std::invalid_argument);
        }
    }
    EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(static_cast<Role>(255),96,96,256,Floor::Four),std::invalid_argument);
    EXPECT_THROW((void)MoEGroupedDotAdmission::nativeCandidate(Role::GateUp,2048,512,256,Floor::Four),std::invalid_argument);
}

/** @test Corrupt live metadata is Invalid even when a floor or mask would otherwise choose Dot4. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, DotAdmissionRejectsInvalidLiveMetadataBeforeChoosingAWriter)
{
    using namespace llaminar2::rocm;
    const auto admission = MoEGroupedDotAdmission::nativeCandidate(
        MoEPrefillProjectionRole::GateUp,96,96,256,MoEGroupedWmmaRowFloor::Sixteen,
        dotAdmissionBit(19),dotAdmissionBit(19));
    const auto valid = dotAdmissionDescriptor();
    for (const int rows : {0,-1,-2147483647})
        EXPECT_EQ(admission.select(valid,rows),MoEGroupedDotProducer::Invalid);
    for (const int n : {0,-1,95,97})
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(19,19,n),1),MoEGroupedDotProducer::Invalid);
    for (const int k : {0,-32,224,288})
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(19,19,96,k),1),MoEGroupedDotProducer::Invalid);
    for (const uint8_t invalid : {uint8_t{32},uint8_t{255}})
    {
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(invalid,19),1),MoEGroupedDotProducer::Invalid);
        EXPECT_EQ(admission.select(dotAdmissionDescriptor(19,invalid),1),MoEGroupedDotProducer::Invalid);
    }
    EXPECT_EQ(admission.select(valid,1),MoEGroupedDotProducer::OrderedDot4);
    EXPECT_EQ(admission.select(valid,16),MoEGroupedDotProducer::Wmma);
}

/** @test A sealed row plan gives cold second tiles and native tails exactly one writer. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, WideRowPlanCoversThresholdsAndAliasesExactlyOnce)
{
    using namespace llaminar2::rocm;
    using Role = MoEPrefillProjectionRole;
    using Floor = MoEGroupedWmmaRowFloor;
    for (const auto role : {Role::GateUp,Role::Down})
        for (const auto floor : {Floor::EveryExpert,Floor::Four,Floor::Eight,Floor::Twelve,Floor::Sixteen})
        {
            const auto admission = MoEGroupedDotAdmission::wideCandidate(role,96,96,256,floor);
            const auto plan = admission.rowPlan();
            ASSERT_TRUE(plan.isWide());
            ASSERT_EQ(plan.role(),role);
            ASSERT_EQ(plan.floor(),floor);
            EXPECT_EQ(plan.rows(0),0);
            EXPECT_EQ(plan.rows(-1),0);
            for (int rows = 1; rows <= 257; ++rows)
            {
                const bool native = rows >= static_cast<int>(floor);
                const int expected_step = native || role == Role::Down ? 16 : 12;
                ASSERT_EQ(plan.rows(rows),expected_step);
                ASSERT_LE(plan.minimumRows(),expected_step);
                // Gate/up may use different prepared codebooks but must share
                // one count-only directory. Include Q8 aliases and promotion.
                for (uint8_t codebook : {uint8_t{0},uint8_t{4},uint8_t{7},uint8_t{19},uint8_t{20},uint8_t{21},uint8_t{24}})
                    for (uint8_t physical : {codebook,uint8_t{23}})
                        ASSERT_EQ(admission.select(dotAdmissionDescriptor(physical,codebook),rows),
                            native ? MoEGroupedDotProducer::Wmma : MoEGroupedDotProducer::OrderedDot4);
                std::vector<int> writers(rows);
                for (int first = 0; first < rows; first += plan.rows(rows))
                {
                    const int live = std::min(plan.rows(rows),rows-first);
                    for (int local = 0; local < live; ++local) ++writers[first+local];
                }
                for (int count : writers) ASSERT_EQ(count,1);
            }
            if (role == Role::GateUp && floor == Floor::Sixteen)
            {
                EXPECT_EQ(plan.rows(13),12);
                EXPECT_EQ(plan.rows(15),12);
                EXPECT_EQ(plan.rows(16),16);
                EXPECT_EQ(plan.minimumRows(),12);
            }
            if (floor == Floor::EveryExpert) EXPECT_EQ(plan.minimumRows(),16);
        }
}

/** @test Directory cache identity distinguishes equal-width but different role/floor plans. */
TEST(Test__ROCmMoEGroupedPrefillDispatch, RowPlanRejectsInvalidGeometryAndSeparatesCaptureIdentity)
{
    using namespace llaminar2::rocm;
    using Role = MoEPrefillProjectionRole;
    using Floor = MoEGroupedWmmaRowFloor;
    const auto wide = MoEGroupedRowPlan::wide(Role::Down,Floor::Sixteen);
    EXPECT_EQ(wide,MoEGroupedRowPlan::wide(Role::Down,Floor::Sixteen));
    EXPECT_NE(wide,MoEGroupedRowPlan::wide(Role::Down,Floor::Eight));
    EXPECT_NE(wide,MoEGroupedRowPlan::wide(Role::GateUp,Floor::Sixteen));
    EXPECT_NE(wide,MoEGroupedRowPlan::nativeMicro(Role::Down,Floor::Sixteen));
    EXPECT_NE(wide,MoEGroupedRowPlan::fixed(16));
    for (int step : {4,8,12,16})
    {
        const auto fixed = MoEGroupedRowPlan::fixed(step);
        EXPECT_FALSE(fixed.isWide());
        EXPECT_EQ(fixed.minimumRows(),step);
        for (int count : {1,13,15,16,17,31,257}) EXPECT_EQ(fixed.rows(count),step);
    }
    for (int step : {-1,0,1,3,5,17,256})
        EXPECT_THROW((void)MoEGroupedRowPlan::fixed(step),std::invalid_argument);
    EXPECT_THROW((void)MoEGroupedRowPlan::wide(static_cast<Role>(255),Floor::Sixteen),std::invalid_argument);
    EXPECT_THROW((void)MoEGroupedRowPlan::wide(Role::GateUp,static_cast<Floor>(15)),std::invalid_argument);
    for (auto role : {Role::GateUp,Role::Down})
        for (auto floor : {Floor::EveryExpert,Floor::Four,Floor::Eight,Floor::Twelve,Floor::Sixteen})
            for (int count : {1,13,15,16,17,31,257})
                EXPECT_EQ(MoEGroupedRowPlan::nativeMicro(role,floor).rows(count),16);
}

#include "kernels/rocm/gemm/ROCmMoEGroupedPopulationPolicy.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedWmma.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPopulationDecision.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPopulationWork.h"
#include "execution/moe/MoEGroupedWorkspaceGeometry.h"

/** @test Exact operation admission excludes each independent neighboring capture identity. */
TEST(Test__ROCmMoEGroupedPopulation, ImmutableOperationReplay)
{
    using namespace llaminar2;
    using namespace llaminar2::rocm;
    const MoEGroupedPopulationKey qwen{
        .operation=MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes,
        .input=MoEGroupedPopulationInput::OriginalTokens,
        .preparation=MoEGroupedPopulationPreparation::PublishedQ8,
        .rows=512,.hidden=2048,.intermediate=512,.experts=256,.top_k=8,
        .gateup_physical=1u<<13,.down_physical=1u<<24,.gateup_source=1u<<13,.down_source=1u<<4};
    EXPECT_EQ(groupedPopulationDomain(qwen),MoEGroupedPopulationDomain::Qwen);
    for (auto member : {&MoEGroupedPopulationKey::rows,&MoEGroupedPopulationKey::hidden,
                       &MoEGroupedPopulationKey::intermediate,&MoEGroupedPopulationKey::experts,
                       &MoEGroupedPopulationKey::top_k})
        for (int difference : {-1,1})
        {
            auto changed=qwen; changed.*member+=difference;
            EXPECT_EQ(groupedPopulationDomain(changed),MoEGroupedPopulationDomain::Unmeasured);
        }
    for (auto member : {&MoEGroupedPopulationKey::gateup_physical,&MoEGroupedPopulationKey::down_physical,
                       &MoEGroupedPopulationKey::gateup_source,&MoEGroupedPopulationKey::down_source})
        for (uint32_t mask : {0u,1u,1u<<4,1u<<13,1u<<24,(1u<<4)|(1u<<24)})
        {
            if (mask==qwen.*member) continue;
            auto changed=qwen; changed.*member=mask;
            EXPECT_EQ(groupedPopulationDomain(changed),MoEGroupedPopulationDomain::Unmeasured);
        }
    auto changed=qwen;changed.operation=MoEPrefillMeasuredOperation::Unmeasured;
    EXPECT_EQ(groupedPopulationDomain(changed),MoEGroupedPopulationDomain::Unmeasured);
    changed=qwen;changed.input=MoEGroupedPopulationInput::OriginalRouteSlots;
    EXPECT_EQ(groupedPopulationDomain(changed),MoEGroupedPopulationDomain::Unmeasured);
    changed=qwen;changed.preparation=MoEGroupedPopulationPreparation::Internal;
    EXPECT_EQ(groupedPopulationDomain(changed),MoEGroupedPopulationDomain::Unmeasured);
    for (uint32_t mask : {1u,1u<<13})
    {
        auto compact=qwen;compact.hidden=512;compact.intermediate=256;compact.experts=8;
        compact.gateup_physical=compact.down_physical=compact.gateup_source=compact.down_source=mask;
        EXPECT_EQ(groupedPopulationDomain(compact),MoEGroupedPopulationDomain::Compact);
    }
    for (auto execution : {MoEPrefillProjectionExecution::gateUp(2048),
                           MoEPrefillProjectionExecution::down(2048,0,2048),
                           MoEPrefillProjectionExecution::down(2048,0,1024)})
        EXPECT_EQ(execution.measuredOperation(MoEPrefillOutputLayout::TokenRows,MoEPrefillRouteScope::AllRoutes),
                  MoEPrefillMeasuredOperation::Unmeasured);
    const auto complete=MoEPrefillProjectionExecution::complete(2048);
    EXPECT_EQ(complete.measuredOperation(MoEPrefillOutputLayout::OriginalRouteRows,MoEPrefillRouteScope::AllRoutes),
              MoEPrefillMeasuredOperation::Unmeasured);
    EXPECT_EQ(complete.measuredOperation(MoEPrefillOutputLayout::TokenRows,MoEPrefillRouteScope::SelectedRoutes),
              MoEPrefillMeasuredOperation::Unmeasured);
}

/** @test Replay measured population identities and adversarial threshold neighbors without a device. */
TEST(Test__ROCmMoEGroupedPopulation, PopulationThresholdReplay)
{
    using namespace llaminar2::rocm;
    using enum MoEGroupedPopulationFamily;
    struct Trace { MoEGroupedPopulation population; MoEGroupedPopulationFamily family; };
    constexpr Trace qwen[] = {
        {{0,INT_MAX,0},Dot4}, {{256,1,1},Dot4}, {{256,8,8},Dot4},
        {{256,11,11},Dot4}, {{256,12,12},WideFloor16}, {{256,13,13},Dot4},
        {{256,12,13},Dot4}, {{256,15,15},Dot4}, {{256,15,16},WideFloor16},
        {{256,16,16},N32}, {{256,17,17},N32},
        {{31,7,8},N32}, {{32,7,8},N32}, {{33,7,8},Dot4},
        {{32,6,8},Dot4}, {{32,8,8},N32}, {{32,11,13},N32},
        {{68,1,4},Dot4}, {{68,1,5},WideFloor16}, {{68,1,6},WideFloor16},
        {{32,16,18},N32}, {{138,1,13},WideFloor16}};
    for (int replay=0;replay<3;++replay)
        for (const auto &trace:qwen)
            EXPECT_EQ(groupedPopulationFamily(MoEGroupedPopulationDomain::Qwen,trace.population),trace.family);
    for (auto trace : {Trace{{8,11,11},Dot4},Trace{{8,12,12},N32},Trace{{8,13,13},N32},
                       Trace{{8,12,13},Dot4},Trace{{7,16,16},Dot4},Trace{{8,17,17},N32}})
        EXPECT_EQ(groupedPopulationFamily(MoEGroupedPopulationDomain::Compact,trace.population),trace.family);
    for (auto population : {MoEGroupedPopulation{-1,1,1},MoEGroupedPopulation{257,1,1},
                            MoEGroupedPopulation{1,0,1},MoEGroupedPopulation{1,2,1},
                            MoEGroupedPopulation{1,1,4097},MoEGroupedPopulation{0,1,1}})
        EXPECT_EQ(groupedPopulationFamily(MoEGroupedPopulationDomain::Qwen,population),Invalid);
    EXPECT_EQ(groupedPopulationFamily(MoEGroupedPopulationDomain::Unmeasured,{8,16,16}),Invalid);
    // Every physically possible positive minimum and a representative tail set
    // has a total decision, independent of unmeasured exact histogram ordering.
    for (int active=1;active<=256;++active)
        for (int minimum=1;minimum<=4096/active;++minimum)
            for (int maximum : {minimum,minimum+1,2*minimum})
                if (maximum<=4096-(active-1)*minimum)
                    EXPECT_NE(groupedPopulationFamily(MoEGroupedPopulationDomain::Qwen,{active,minimum,maximum}),Invalid);
}

/** @test A projection gate cannot advertise another writer, and submitted geometry remains explicit. */
TEST(Test__ROCmMoEGroupedPopulation, BindingAndSubmissionReplay)
{
    using namespace llaminar2::rocm;
    using enum MoEGroupedPopulationFamily;
    uint32_t opaque_storage=0xa5a5a5a5; // Factory never reads this device-address stand-in.
    const MoEGroupedPopulationBinding binding(&opaque_storage);
    EXPECT_EQ(binding.address(),&opaque_storage);
    EXPECT_THROW((void)MoEGroupedPopulationBinding(nullptr),std::invalid_argument);
    const MoEGroupedPopulationPublisher publisher(&opaque_storage);
    EXPECT_EQ(publisher.address(),binding.address());
    EXPECT_EQ(opaque_storage,0xa5a5a5a5u);
    EXPECT_THROW((void)MoEGroupedPopulationPublisher(nullptr),std::invalid_argument);
    MoEGroupedPrefillSubmission receipt;
    EXPECT_EQ(receipt.state,MoEGroupedPrefillSubmission::State::Empty);
    const MoEGroupedPrefillTileSubmission tile{MoEGroupedPrefillEngine::WmmaM16N32,
        MoEGroupedWmmaRowFloor::EveryExpert,16,32,128,0,1};
    receipt.gateup.record(tile);receipt.gateup.record(tile);
    EXPECT_EQ(receipt.gateup.size,1u); EXPECT_EQ(receipt.gateup.leaves(),2u);
    EXPECT_EQ(receipt.gateup.tiles[0].threads,128);
    receipt.gateup.record({MoEGroupedPrefillEngine::WideDot4Wmma,
        MoEGroupedWmmaRowFloor::Sixteen,16,64,256,12,1,256});
    EXPECT_EQ(receipt.gateup.size,2u); EXPECT_EQ(receipt.gateup.leaves(),3u);
    receipt={};
    EXPECT_EQ(receipt.gateup.leaves(),0u);EXPECT_EQ(receipt.down.leaves(),0u);
    EXPECT_EQ(receipt.state,MoEGroupedPrefillSubmission::State::Empty);
}

/** @test Producer and consumer agree on concentrated native rows and exact ordered Qwen rows. */
TEST(Test__ROCmMoEGroupedPopulation, FusedRowOwnershipReplay)
{
    using namespace llaminar2::rocm;
    using enum MoEGroupedPopulationFamily;
    struct Trace { MoEGroupedPopulationFamily family; int count; bool native; int rows; };
    for (const auto trace : {Trace{Dot4,1,false,12},Trace{Dot4,12,false,12},Trace{Dot4,15,false,12},
             Trace{N32,7,true,16},Trace{N32,12,true,16},Trace{N32,13,true,16},Trace{N32,15,true,16},
             Trace{N32,16,true,16},Trace{N32,17,true,16},Trace{N32,4096,true,16},
             Trace{WideFloor16,12,false,12},Trace{WideFloor16,13,false,12},Trace{WideFloor16,15,false,12},
             Trace{WideFloor16,16,true,16},Trace{WideFloor16,17,true,16},Trace{WideFloor16,4096,true,16}})
    {
        EXPECT_EQ(populationProjectionProducer(trace.family,trace.count),trace.native
            ? MoEGroupedDotProducer::Wmma : MoEGroupedDotProducer::OrderedDot4);
        const int gu = populationProjectionRows(MoEPrefillProjectionRole::GateUp,trace.family,trace.count);
        ASSERT_EQ(gu,trace.rows);
        EXPECT_EQ(populationProjectionRows(MoEPrefillProjectionRole::Down,trace.family,trace.count),16);
        int next = 0;
        for (int first=0;first<trace.count;first+=gu)
            for (int row=first;row<std::min(first+gu,trace.count);++row) EXPECT_EQ(row,next++);
        EXPECT_EQ(next,trace.count);
    }
    EXPECT_EQ(populationProjectionRows(MoEPrefillProjectionRole::GateUp,N32,0),0);
    EXPECT_EQ(populationProjectionRows(MoEPrefillProjectionRole::Down,Invalid,1),0);
    EXPECT_EQ(populationProjectionRows(static_cast<MoEPrefillProjectionRole>(255),Dot4,1),0);
}

/** @test The live publication has a distinct, bounded word in the existing arena bank. */
TEST(Test__ROCmMoEGroupedPopulation, PersistentDirectoryStorageReplay)
{
    using namespace llaminar2::rocm;
    for (int capacity : {1,16,512,598})
    {
        const MoEGroupedPopulationDirectoryLayout layout(capacity);
        EXPECT_EQ(layout.liveCountIndex(),capacity);
        EXPECT_EQ(layout.words(),static_cast<std::size_t>(capacity)+1);
        EXPECT_LE(layout.words(),llaminar2::MoEWorkspaceBuffers::rocmAdaptivePrefillDirectoryWords(4096,256));
    }
    EXPECT_THROW((void)MoEGroupedPopulationDirectoryLayout(0),std::invalid_argument);
    EXPECT_THROW((void)MoEGroupedPopulationDirectoryLayout(-1),std::invalid_argument);
}

/** @test Route-publication identity is complete and independent of live device values. */
TEST(Test__ROCmMoEGroupedPopulation, RoutePublicationBindingReplay)
{
    using namespace llaminar2;
    using namespace llaminar2::rocm;
    int counts[8]{}, offsets[8]{}, foreign[8]{};
    const ROCmMoEGroupedPrefillRouteKey key{
        .gateup_codebook=0,.down_codebook=0,.hidden_size=512,.expert_width=256,
        .expert_count=8,.top_k=8,.rows=512,
        .operation=MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes};
    const ROCmMoEGroupedRouteAdmission admission(key,counts,offsets);
    for (int replay=0;replay<3;++replay)
    {
        counts[0]=replay; offsets[7]=replay*8;
        EXPECT_TRUE(admission.binds(key,counts,offsets));
        EXPECT_FALSE(admission.binds(key,foreign,offsets));
        EXPECT_FALSE(admission.binds(key,counts,foreign));
        EXPECT_FALSE(admission.binds(key,nullptr,offsets));
        EXPECT_FALSE(admission.binds(key,counts,nullptr));
        for (auto field : {&ROCmMoEGroupedPrefillRouteKey::hidden_size,&ROCmMoEGroupedPrefillRouteKey::expert_width,
                           &ROCmMoEGroupedPrefillRouteKey::expert_count,&ROCmMoEGroupedPrefillRouteKey::top_k,
                           &ROCmMoEGroupedPrefillRouteKey::rows})
        {
            auto changed=key; ++(changed.*field);
            EXPECT_FALSE(admission.binds(changed,counts,offsets));
        }
        auto changed=key; changed.gateup_codebook=13;
        EXPECT_FALSE(admission.binds(changed,counts,offsets));
        changed=key; changed.down_codebook=13;
        EXPECT_FALSE(admission.binds(changed,counts,offsets));
        changed=key; changed.operation=MoEPrefillMeasuredOperation::Unmeasured;
        EXPECT_FALSE(admission.binds(changed,counts,offsets));
    }
}
