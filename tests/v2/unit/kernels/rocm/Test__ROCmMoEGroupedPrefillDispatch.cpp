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
#include <cstddef>
#include <cstdint>
#include <utility>

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
