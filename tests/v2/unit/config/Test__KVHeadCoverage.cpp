/**
 * @file Test__KVHeadCoverage.cpp
 * @brief Device-free original-source KV coverage and subset-replica assembly proofs.
 *
 * These tests exercise opt-in typed geometry and diagnostic byte ownership only.
 * No schema/planner/cache/kernel consumer is switched to the candidate policy.
 */
#include <gtest/gtest.h>
#include "config/TensorParallelConfig.h"
#include "config/GDNHeadAssignment.h"
#include "execution/debug/KVHeadSnapshotColumnGroups.h"
#include <bit>
#include <cstdint>
#include <limits>

namespace llaminar2::test
{
namespace
{
/** @brief Produce exact opt-in Q24/KV2 geometry using existing ordinary split axes. */
TensorParallelConfig covered(int degree)
{
    return TensorParallelConfig::equalSplit(degree, 24, 2, 10240, 248320,
        std::nullopt, KVHeadCoveragePolicy::QueryInterval);
}
/** @brief Derive actual local row widths and deterministic full-source head values. */
TPSnapshot snapshot(const TensorParallelConfig &assignment, size_t rows = 3, size_t width = 4)
{
    TPSnapshot result;
    result.key = "K_PROJECTION";
    result.mode = SnapshotShardingMode::PACKED_COLUMN_PARALLEL;
    result.tp_degree = assignment.worldSize();
    std::vector<size_t> widths;
    for (size_t rank = 0; rank < assignment.assignments().size(); ++rank)
    {
        const auto &a = assignment.assignments()[rank];
        DeviceSnapshotData data;
        data.device_index = static_cast<int>(rank);
        data.rows = rows;
        data.cols = static_cast<size_t>(a.kv_head_count) * width;
        for (size_t row = 0; row < rows; ++row)
            for (int head = a.kv_head_start; head < a.kvHeadEnd(); ++head)
                for (size_t column = 0; column < width; ++column)
                    data.data.push_back(row == 0 && head == 0 && column == 0 ? -0.f :
                        static_cast<float>(1000 * row + 100 * head + column));
        widths.push_back(data.cols);
        result.device_data.push_back(std::move(data));
    }
    result.column_groups = resolveKVHeadSnapshotColumnGroups(assignment, widths, width);
    return result;
}
/** @brief Check original head order and unchanged sign bit; no replica sum is permitted. */
void expectOriginal(TPSnapshot value, size_t rows = 3, size_t width = 4, size_t heads = 2)
{
    ASSERT_TRUE(value.computeCombined());
    ASSERT_EQ(value.combined_rows, rows);
    ASSERT_EQ(value.combined_cols, heads * width);
    for (size_t row = 0; row < rows; ++row)
        for (size_t head = 0; head < heads; ++head)
            for (size_t column = 0; column < width; ++column)
            {
                const float expected = row == 0 && head == 0 && column == 0 ? -0.f :
                    static_cast<float>(1000 * row + 100 * head + column);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(value.combined_data[row * heads * width + head * width + column]),
                          std::bit_cast<std::uint32_t>(expected));
            }
}
} // namespace

/** @test Equal coverage follows actual Q intervals, including TP3 straddling and TP8 subgroups. */
TEST(KVHeadCoverage, ExactOriginalIntervalsAndLogicalTotals)
{
    for (int degree = 1; degree <= 8; ++degree)
    {
        const auto actual = covered(degree);
        const auto legacy = TensorParallelConfig::equalSplit(degree, 24, 2, 10240, 248320);
        ASSERT_TRUE(actual.validate()) << actual.validationError();
        ASSERT_EQ(actual.totalHeads(), 24);
        ASSERT_EQ(actual.totalKVHeads(), 2);
        EXPECT_EQ(actual.kvHeadCoveragePolicy(), KVHeadCoveragePolicy::QueryInterval);
        EXPECT_EQ(legacy.kvHeadCoveragePolicy(), KVHeadCoveragePolicy::Legacy);
        EXPECT_FALSE(legacy.kvHeadCoverageDeclaration());
        int physical_copies = 0;
        for (int rank = 0; rank < degree; ++rank)
        {
            const auto &a = actual.forRank(rank), &b = legacy.forRank(rank);
            EXPECT_EQ(a.head_start, b.head_start); EXPECT_EQ(a.head_count, b.head_count);
            EXPECT_EQ(a.d_ff_start, b.d_ff_start); EXPECT_EQ(a.d_ff_count, b.d_ff_count);
            EXPECT_EQ(a.vocab_start, b.vocab_start); EXPECT_EQ(a.vocab_count, b.vocab_count);
            // Independent literal global-GQA endpoints, rather than invoking the resolver as oracle.
            EXPECT_EQ(a.kv_head_start, a.head_start / 12);
            EXPECT_EQ(a.kvHeadEnd(), (a.headEnd() - 1) / 12 + 1);
            physical_copies += a.kv_head_count;
            if (degree > 2) { EXPECT_EQ(b.kv_head_start, 0); EXPECT_EQ(b.kv_head_count, 2); }
        }
        if (degree == 4)
        {
            EXPECT_EQ(physical_copies, 4);
            for (int rank = 0; rank < degree; ++rank)
            { EXPECT_EQ(actual.forRank(rank).kv_head_start, rank / 2); EXPECT_EQ(actual.forRank(rank).kv_head_count, 1); }
        }
        expectOriginal(snapshot(actual));
    }
}

/** @test Every degree1..8 retains exact local/global GQA/MHA mapping and minimal required work. */
TEST(KVHeadCoverage, EveryDegreeSupportsRepresentativeOriginalHeadGeometry)
{
    for (const auto &[queries, kv] : {std::pair{24, 2}, std::pair{28, 4}, std::pair{8, 8}, std::pair{32, 1}})
        for (int degree = 1; degree <= 8; ++degree)
        {
            SCOPED_TRACE(::testing::Message() << queries << "/" << kv << "/TP" << degree);
            const auto actual = TensorParallelConfig::equalSplit(degree, queries, kv, 10240, 248320,
                std::nullopt, KVHeadCoveragePolicy::QueryInterval);
            ASSERT_TRUE(actual.validate());
            const int ratio = queries / kv;
            std::vector<int> original_query_owners(queries, 0), original_kv_owners(kv, 0);
            for (const auto &a : actual.assignments())
            {
                for (int query = a.head_start; query < a.headEnd(); ++query)
                {
                    ++original_query_owners[query];
                    // This is the exact offset-aware sparse-consumer formula.
                    const int local_kv = query / ratio - a.kv_head_start;
                    EXPECT_GE(local_kv, 0); EXPECT_LT(local_kv, a.kv_head_count);
                    EXPECT_EQ(a.kv_head_start + local_kv, query / ratio);
                }
                for (int head = 0; head < kv; ++head)
                {
                    const bool needed = head * ratio < a.headEnd() && a.head_start < (head + 1) * ratio;
                    const bool assigned = head >= a.kv_head_start && head < a.kvHeadEnd();
                    EXPECT_EQ(assigned, needed); // No irrelevant KV projection/cache head.
                    if (assigned) ++original_kv_owners[head];
                }
            }
            EXPECT_TRUE(std::all_of(original_query_owners.begin(), original_query_owners.end(), [](int n) { return n == 1; }));
            EXPECT_TRUE(std::all_of(original_kv_owners.begin(), original_kv_owners.end(), [](int n) { return n > 0; }));
            EXPECT_EQ(actual.totalKVHeads(), kv);
            expectOriginal(snapshot(actual), 3, 4, kv);
        }
    EXPECT_THROW(TensorParallelConfig::equalSplit(5, 4, 1, 10240, 248320,
        std::nullopt, KVHeadCoveragePolicy::QueryInterval), std::invalid_argument);
    EXPECT_THROW(TensorParallelConfig::equalSplit(8, 24, 2, 32, 248320,
        std::nullopt, KVHeadCoveragePolicy::QueryInterval), std::invalid_argument);
    EXPECT_THROW(TensorParallelConfig::equalSplit(8, 24, 2, 10240, 1,
        std::nullopt, KVHeadCoveragePolicy::QueryInterval), std::invalid_argument);
}

/** @test Existing equal-Q GDN limits stay visible; authored aligned geometry supports all1..8. */
TEST(KVHeadCoverage, GDNBoundaryLimitationIsExplicitRatherThanSkipped)
{
    for (int degree = 1; degree <= 8; ++degree)
    {
        auto assignments = covered(degree).assignments();
        bool original_gdn_representable = true;
        for (const auto &a : assignments)
            original_gdn_representable &= GDNHeadAssignment::hasIntegralKeyHeadBoundaries(
                16, 48, a.head_start, a.head_count, 24);
        EXPECT_EQ(original_gdn_representable, degree == 1 || degree == 2 || degree == 4 || degree == 8);
        // An explicit legal query partition exists: split eight3-head units.
        // This is NOT installed as another implicit factory rounding rule.
        int start = 0;
        for (int rank = 0; rank < degree; ++rank)
        {
            auto &a = assignments[rank];
            a.head_start = start; a.head_count = 3 * (8 / degree + (rank < 8 % degree ? 1 : 0));
            start += a.head_count;
            a.kv_head_start = a.head_start / 12;
            a.kv_head_count = (a.headEnd() - 1) / 12 + 1 - a.kv_head_start;
            EXPECT_TRUE(GDNHeadAssignment::hasIntegralKeyHeadBoundaries(16, 48, a.head_start, a.head_count, 24));
        }
        const TensorParallelConfig aligned(std::move(assignments), KVHeadCoverageDeclaration{24, 2});
        ASSERT_TRUE(aligned.validate());
        expectOriginal(snapshot(aligned));
    }
}

/** @test Unequal straddling intervals and reversed participant order retain original identities. */
TEST(KVHeadCoverage, UnequalStraddlingAndReorderedDevices)
{
    const auto actual = TensorParallelConfig::proportionalSplit(
        {DeviceId::rocm(3), DeviceId::rocm(1), DeviceId::rocm(2), DeviceId::rocm(0)},
        {1.f, 5.f, 1.f, 1.f}, 24, 2, 10240, 248320, KVHeadCoveragePolicy::QueryInterval);
    EXPECT_EQ(actual.forRank(0).head_count, 3);
    EXPECT_EQ(actual.forRank(1).head_count, 15);
    EXPECT_EQ(actual.forRank(1).kv_head_start, 0);
    EXPECT_EQ(actual.forRank(1).kv_head_count, 2);
    expectOriginal(snapshot(actual));
    auto reordered = actual.assignments();
    std::reverse(reordered.begin(), reordered.end());
    for (size_t rank = 0; rank < reordered.size(); ++rank) reordered[rank].local_rank = static_cast<int>(rank);
    const TensorParallelConfig reverse(reordered, KVHeadCoverageDeclaration{24, 2});
    EXPECT_EQ(reverse.forDevice(DeviceId::rocm(1)).head_start, 3);
    expectOriginal(snapshot(reverse));
}

/** @test Original bounds and exact consumer overlap cannot be inferred from a smaller union. */
TEST(KVHeadCoverage, RejectsMissingForeignEmptyAndOverflowGeometry)
{
    const auto good = covered(4).assignments();
    auto bad = good; bad[1].local_rank = 0;
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; bad[1].local_rank = 4;
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; std::reverse(bad.begin(), bad.end());
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; bad.pop_back();
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; bad[1].kv_head_start = 1;
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; bad[0].kv_head_count = 2;
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; bad[0].head_count = 0;
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; bad[1].head_start = 5;
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    bad = good; bad[0].d_ff_start = std::numeric_limits<int>::max();
    EXPECT_THROW(TensorParallelConfig(bad, KVHeadCoverageDeclaration{24, 2}), std::invalid_argument);
    EXPECT_THROW(TensorParallelConfig(good, KVHeadCoverageDeclaration{24, 3}), std::invalid_argument);
    EXPECT_THROW(resolveQueryKVHeadCoverage({25, 2}, 0, 1), std::invalid_argument);
    EXPECT_THROW(resolveQueryKVHeadCoverage({24, 2}, -1, 1), std::invalid_argument);
    EXPECT_THROW(resolveQueryKVHeadCoverage({24, 2}, 0, 0), std::invalid_argument);
    EXPECT_THROW(resolveQueryKVHeadCoverage({24, 2}, 23, 2), std::invalid_argument);
    EXPECT_THROW(resolveQueryKVHeadCoverage({std::numeric_limits<int>::max(), 1},
        std::numeric_limits<int>::max() - 1, 2), std::invalid_argument);
    EXPECT_EQ(resolveQueryKVHeadCoverage({std::numeric_limits<int>::max(), 1},
        std::numeric_limits<int>::max() - 1, 1), (KVHeadInterval{0, 1}));
    EXPECT_THROW(TensorParallelConfig::equalSplit(4, 24, 2, 10240, 248320, std::nullopt,
        static_cast<KVHeadCoveragePolicy>(99)), std::invalid_argument);
    EXPECT_THROW(TensorParallelConfig::proportionalSplit({DeviceId::rocm(0), DeviceId::rocm(1)},
        {std::numeric_limits<float>::quiet_NaN(), 1.f}, 24, 2, 128, 128, KVHeadCoveragePolicy::QueryInterval),
        std::invalid_argument);
}

/** @test Subset ownership verifies every actual replica, its shape and complete membership. */
TEST(KVHeadCoverage, SnapshotRejectsMissingAndDivergentReplicas)
{
    const auto exact_head_width = snapshot(covered(4), 1, 256);
    EXPECT_EQ(exact_head_width.column_groups[0].participant_cols, (std::vector<size_t>{256, 256, 0, 0}));
    EXPECT_EQ(exact_head_width.column_groups[1].participant_cols, (std::vector<size_t>{0, 0, 256, 256}));
    expectOriginal(exact_head_width, 1, 256);
    expectOriginal(snapshot(covered(4), 0), 0);
    auto original = snapshot(covered(4));
    ASSERT_EQ(original.column_groups[0].participant_cols, (std::vector<size_t>{4, 4, 0, 0}));
    ASSERT_EQ(original.column_groups[1].participant_cols, (std::vector<size_t>{0, 0, 4, 4}));
    for (const int participant : {1, 3})
    {
        auto bad = original; bad.device_data[participant].data[1] += 1.f;
        EXPECT_FALSE(bad.computeCombined());
        bad = original; bad.device_data[participant].data[1] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(bad.computeCombined());
    }
    auto bad = original; bad.device_data[2].data[0] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(bad.computeCombined()); // First real owner of original head1 is not participant0.
    bad = original; bad.device_data.pop_back(); EXPECT_FALSE(bad.computeCombined());
    bad = original; bad.device_data.back().device_index = 0; EXPECT_FALSE(bad.computeCombined());
    bad = original; ++bad.device_data.back().rows; EXPECT_FALSE(bad.computeCombined());
    bad = original; bad.column_groups[0].participant_cols.assign(4, 0); EXPECT_FALSE(bad.computeCombined());
    bad = original; bad.column_groups[0].participant_cols[1] = 3; EXPECT_FALSE(bad.computeCombined());
    bad = original; bad.column_groups[0].mode = static_cast<SnapshotColumnGroupMode>(99); EXPECT_FALSE(bad.computeCombined());
    EXPECT_THROW(resolveKVHeadSnapshotColumnGroups(covered(4), std::vector<size_t>{4, 4, 4}, 4), std::invalid_argument);
    EXPECT_THROW(resolveKVHeadSnapshotColumnGroups(covered(4), std::vector<size_t>{4, 8, 4, 4}, 4), std::invalid_argument);
    EXPECT_THROW(resolveKVHeadSnapshotColumnGroups(TensorParallelConfig::equalSplit(4, 24, 2, 128, 128),
        std::vector<size_t>{8, 8, 8, 8}, 4), std::invalid_argument);
    EXPECT_THROW(resolveKVHeadSnapshotColumnGroups(covered(4), std::vector<size_t>{4, 4, 4, 4},
        std::numeric_limits<size_t>::max()), std::invalid_argument);
}

/** @test Existing all-participant and disjoint group modes retain their original valid assembly. */
TEST(KVHeadCoverage, LegacyGroupModesStayDistinct)
{
    TPSnapshot value;
    value.mode = SnapshotShardingMode::PACKED_COLUMN_PARALLEL; value.tp_degree = 2;
    for (int rank = 0; rank < 2; ++rank)
    {
        DeviceSnapshotData data; data.device_index = rank; data.rows = 1; data.cols = 2;
        data.data = {float(rank + 1), 9.f}; value.device_data.push_back(std::move(data));
    }
    value.column_groups = {{"Q", 2, SnapshotColumnGroupMode::PARTITIONED, {1, 1}},
                           {"K", 1, SnapshotColumnGroupMode::REPLICATED, {1, 1}}};
    ASSERT_TRUE(value.computeCombined()); EXPECT_EQ(value.combined_data, (std::vector<float>{1, 2, 9}));
    auto legacy = TensorParallelConfig::equalSplit(2, 24, 2, 128, 128);
    EXPECT_TRUE(legacy.validate()); EXPECT_EQ(legacy.forRank(1).kv_head_start, 1);
    EXPECT_EQ(legacy.forRank(1).kv_head_count, 1);
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/** @test Canonical aligned distributor resolves every QFN degree1..8 without changing1/2/4/8. */
TEST(KVHeadCoverage, CanonicalConsumerGranularitySupportsEveryDegree)
{
    const auto declaration = QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {{16, 48}});
    ASSERT_EQ(declaration.granularity(), 3);
    ASSERT_EQ(declaration.originalQueryHeads(), 24);
    const std::vector<std::vector<int>> expected = {
        {24}, {12, 12}, {9, 9, 6}, {6, 6, 6, 6}, {6, 6, 6, 3, 3},
        {6, 6, 3, 3, 3, 3}, {6, 3, 3, 3, 3, 3, 3}, {3, 3, 3, 3, 3, 3, 3, 3}};
    for (int degree = 1; degree <= 8; ++degree)
    {
        SCOPED_TRACE(degree);
        const auto actual = TensorParallelConfig::equalSplit(degree, 24, 2, 10240, 248320,
            std::nullopt, KVHeadCoveragePolicy::QueryInterval, declaration);
        const auto original = TensorParallelConfig::equalSplit(degree, 24, 2, 10240, 248320);
        ASSERT_TRUE(actual.validate()); ASSERT_TRUE(actual.queryHeadPartitionDeclaration());
        EXPECT_EQ(actual.queryHeadGranularity(), 3);
        std::vector<int> counts;
        std::vector<int> query_owners(24, 0), kv_owners(2, 0);
        for (int rank = 0; rank < degree; ++rank)
        {
            const auto &a = actual.forRank(rank);
            counts.push_back(a.head_count);
            EXPECT_TRUE(GDNHeadAssignment::hasIntegralKeyHeadBoundaries(16, 48, a.head_start, a.head_count, 24));
            for (int query = a.head_start; query < a.headEnd(); ++query)
            {
                ++query_owners[query];
                const int local = query / 12 - a.kv_head_start;
                EXPECT_GE(local, 0); EXPECT_LT(local, a.kv_head_count);
            }
            for (int head = 0; head < 2; ++head)
            {
                const bool needed = head * 12 < a.headEnd() && a.head_start < (head + 1) * 12;
                const bool assigned = head >= a.kv_head_start && head < a.kvHeadEnd();
                EXPECT_EQ(assigned, needed);
                if (assigned) ++kv_owners[head];
            }
            if (degree == 1 || degree == 2 || degree == 4 || degree == 8)
            {
                EXPECT_EQ(a.head_start, original.forRank(rank).head_start);
                EXPECT_EQ(a.head_count, original.forRank(rank).head_count);
            }
            EXPECT_EQ(a.d_ff_start, original.forRank(rank).d_ff_start);
            EXPECT_EQ(a.d_ff_count, original.forRank(rank).d_ff_count);
            EXPECT_EQ(a.vocab_start, original.forRank(rank).vocab_start);
            EXPECT_EQ(a.vocab_count, original.forRank(rank).vocab_count);
        }
        EXPECT_EQ(counts, expected[degree - 1]);
        EXPECT_TRUE(std::all_of(query_owners.begin(), query_owners.end(), [](int n) { return n == 1; }));
        EXPECT_GT(kv_owners[0], 0); EXPECT_GT(kv_owners[1], 0);
        expectOriginal(snapshot(actual));
        // Aligned query demand is independent from the legacy KV policy.
        const auto full_bank = TensorParallelConfig::equalSplit(degree, 24, 2, 10240, 248320,
            std::nullopt, KVHeadCoveragePolicy::Legacy, declaration);
        EXPECT_EQ(full_bank.queryHeadGranularity(), 3);
        if (degree > 2) for (const auto &a : full_bank.assignments())
        { EXPECT_EQ(a.kv_head_start, 0); EXPECT_EQ(a.kv_head_count, 2); }
        EXPECT_EQ(full_bank.totalKVHeads(), 2);
    }
    const auto devices = std::vector<DeviceId>{DeviceId::rocm(3), DeviceId::rocm(1), DeviceId::rocm(0)};
    const auto proportional = TensorParallelConfig::proportionalSplit(devices, {1.f, 1.f, 1.f},
        24, 2, 10240, 248320, KVHeadCoveragePolicy::QueryInterval, declaration);
    EXPECT_EQ(proportional.forRank(0).head_count, 9);
    EXPECT_EQ(proportional.forRank(1).head_count, 9);
    EXPECT_EQ(proportional.forRank(2).head_count, 6);
    expectOriginal(snapshot(proportional));
}

/** @test Original consumers seal minimal LCM geometry; neither forged scalar nor stale raw cuts are accepted. */
TEST(KVHeadCoverage, ConsumerDeclarationRejectsForeignCutsAndPreservesNonGDNDefault)
{
    const auto source = QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {{16, 48}});
    const auto both = QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {{16, 48}, {12, 24}});
    EXPECT_EQ(both.granularity(), 6);
    for (int boundary = 0; boundary <= 24; ++boundary)
    {
        const bool original_integral = (16 * boundary) % 24 == 0 && (48 * boundary) % 24 == 0 &&
                                       (12 * boundary) % 24 == 0 && (24 * boundary) % 24 == 0;
        EXPECT_EQ(boundary % both.granularity() == 0, original_integral);
    }
    const auto unconstrained = QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {});
    EXPECT_EQ(unconstrained.granularity(), 1);
    for (int degree = 1; degree <= 8; ++degree)
    {
        const auto ordinary = covered(degree);
        const auto explicit_non_gdn = TensorParallelConfig::equalSplit(degree, 24, 2, 10240, 248320,
            std::nullopt, KVHeadCoveragePolicy::QueryInterval, unconstrained);
        for (int rank = 0; rank < degree; ++rank)
        { EXPECT_EQ(explicit_non_gdn.forRank(rank).head_start, ordinary.forRank(rank).head_start);
          EXPECT_EQ(explicit_non_gdn.forRank(rank).head_count, ordinary.forRank(rank).head_count); }
    }
    EXPECT_THROW(QueryHeadPartitionDeclaration::fromOriginalConsumers(0, {}), std::invalid_argument);
    EXPECT_THROW(QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {{0, 48}}), std::invalid_argument);
    EXPECT_THROW(QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {{16, 47}}), std::invalid_argument);
    EXPECT_THROW(TensorParallelConfig::equalSplit(9, 24, 2, 10240, 248320,
        std::nullopt, KVHeadCoveragePolicy::QueryInterval, source), std::invalid_argument);
    EXPECT_THROW(TensorParallelConfig::equalSplit(4, 24, 2, 10240, 248320,
        std::nullopt, KVHeadCoveragePolicy::QueryInterval,
        QueryHeadPartitionDeclaration::fromOriginalConsumers(30, {{10, 20}})), std::invalid_argument);
    const auto mha = QueryHeadPartitionDeclaration::fromOriginalConsumers(8, {{4, 8}});
    const auto exact_mha = TensorParallelConfig::equalSplit(3, 8, 8, 10240, 248320,
        std::nullopt, KVHeadCoveragePolicy::QueryInterval, mha);
    ASSERT_TRUE(exact_mha.validate());
    EXPECT_EQ(exact_mha.forRank(0).head_count, 4);
    EXPECT_EQ(exact_mha.forRank(0).kv_head_count, 4);
    EXPECT_THROW(TensorParallelConfig::equalSplit(3, 8, 8, 10240, 248320,
        std::nullopt, KVHeadCoveragePolicy::Legacy, mha), std::invalid_argument);
    const auto unaligned = covered(3).assignments();
    EXPECT_THROW(TensorParallelConfig(unaligned, KVHeadCoverageDeclaration{24, 2}, source), std::invalid_argument);
    EXPECT_THROW(TensorParallelConfig(unaligned, source), std::invalid_argument);
    auto aligned = covered(4).assignments();
    EXPECT_NO_THROW(TensorParallelConfig(aligned, KVHeadCoverageDeclaration{24, 2}, source));
    EXPECT_THROW(TensorParallelConfig(aligned, KVHeadCoverageDeclaration{24, 2},
        QueryHeadPartitionDeclaration::fromOriginalConsumers(30, {})), std::invalid_argument);
    aligned[0].local_rank = 3;
    EXPECT_THROW(TensorParallelConfig(aligned, source), std::invalid_argument);
}
} // namespace llaminar2::test
