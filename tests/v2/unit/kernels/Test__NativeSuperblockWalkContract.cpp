/**
 * @file Test__NativeSuperblockWalkContract.cpp
 * @brief Device-free admission, exact-boundary and fatal checks for native CPU
 *        superblock walks and ordered K-partition plans.
 *
 * The NEON and x86 ordered-partition kernels keep their hot int expressions
 * `first = begin + p * width` and `last = min(first + width, end)` unchanged;
 * CPUNativeOrderedPartitionPlan.h admits each plan once, in cold code, so that
 * none of those expressions can overflow. These tests call the production
 * predicates and the production producer requireOrderedPartitionWidth() (the
 * NEON tile constructor's and packed prefill's width authority) directly, prove the exact representability boundary (INT_MAX
 * admitted, INT_MAX + 1 rejected) and the fatal rejections, including the
 * original stuck range first=8, last=9, blocks=8. On x86 the actual
 * single-scale kernel entry is called with that range as a production-path
 * fatal control. The NEON tile's plan rejection cannot be reached through an
 * admitted view (blocks_per_row = ceil(K / 32) <= 2^26 with at most sixteen
 * partitions), so its producer is exercised directly.
 *
 * Ordered K tiles handed to a kernel entry come from orderedKTileBegin(): the
 * sixteen-tile GPU-aligned rows of 35, 37, 17 and 24 blocks, whose unclamped
 * trailing starts reached the x86 single-scale entries as reversed walks, now
 * hand over [blocks, blocks); the reversed ranges themselves stay fatal.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include "kernels/cpu/gemm/CPUNativeOrderedKTile.h"
#include "kernels/cpu/gemm/CPUNativeOrderedPartitionPlan.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedPreparedFootprint.h"
#if defined(__AVX2__)
#include "kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h"
#endif
using namespace llaminar2::cpu::native_quantised;

namespace {
/** @brief One canonical ordered plan over [begin, end) of a row with `blocks` blocks. */
struct Plan {
    int begin;
    int end;
    int blocks;
    int partitions;
};

/** @brief Singleton, odd, 37 and overpartitioned plans at zero and nonzero begin, up to INT_MAX. */
std::vector<Plan> admittedPlans() {
    std::vector<Plan> result;
    for (int blocks : {0, 1, 7, 8, 9, 31, 4096})
        for (int begin : {0, 1, 5, blocks / 2, blocks})
            for (int partitions : {1, 3, 16, 17, 37}) {
                if (begin > blocks) continue;
                result.push_back({begin, blocks, blocks, partitions});
                if (blocks - begin > 2) result.push_back({begin, blocks - 1, blocks, partitions});
            }
    // INT_MAX extents whose canonical width keeps partitions * width <= INT_MAX - begin.
    result.push_back({0, INT_MAX, INT_MAX, 1});
    result.push_back({INT_MAX - 37, INT_MAX, INT_MAX, 37});
    result.push_back({INT_MAX - 3, INT_MAX, INT_MAX, 3});
    result.push_back({INT_MAX, INT_MAX, INT_MAX, 37});
    return result;
}
} // namespace

TEST(NativeSuperblockWalkContract, AdmittedPlansKeepTheUnchangedHotIntExpressionsInRange) {
    for (const Plan &plan : admittedPlans())
        for (int phase = 0; phase < 8; ++phase) {
            SCOPED_TRACE(::testing::Message() << "begin=" << plan.begin << " end=" << plan.end << " blocks="
                                              << plan.blocks << " partitions=" << plan.partitions << " phase=" << phase);
            const int width = requireOrderedPartitionWidth(plan.begin, plan.end, std::size_t(plan.blocks),
                                                           plan.partitions, std::size_t(phase));
            ASSERT_EQ(width, (std::int64_t(plan.end) - plan.begin + plan.partitions - 1) / plan.partitions);
            ASSERT_TRUE(admitsOrderedPartitionPlan(plan.begin, plan.end, std::size_t(plan.blocks), plan.partitions,
                                                   width, std::size_t(phase)));
            const CPUNativeQuantisedBlockGeometry geometry{.blocks_per_row = std::size_t(plan.blocks),
                                                          .phase = std::size_t(phase)};
            std::int64_t covered = plan.begin;
            for (int partition = 0; partition < plan.partitions; ++partition) {
                // The kernels' widest int intermediates, evaluated exactly: both must fit int.
                const std::int64_t first = std::int64_t(plan.begin) + std::int64_t(partition) * width;
                const std::int64_t past = first + width;
                ASSERT_LE(past, INT_MAX) << "partition " << partition;
                const std::int64_t last = std::min<std::int64_t>(past, plan.end);
                if (first >= last) {
                    // An excess partition is empty (possibly reversed): `block < last` loads nothing.
                    EXPECT_GE(first, plan.end);
                    continue;
                }
                // A nonempty interval is ordered, contiguous and an admitted walk.
                ASSERT_EQ(first, covered);
                ASSERT_TRUE(admitsSuperblockWalk(int(first), int(last), std::size_t(plan.blocks), std::size_t(phase)));
                requireSuperblockWalk(int(first), int(last), std::size_t(plan.blocks), std::size_t(phase));
                // Sample both ends, including a ragged final superblock, without walking INT_MAX blocks.
                for (std::int64_t block : {first, last - 1}) {
                    EXPECT_GT(std::min<std::size_t>(std::size_t(last), geometry.superblockEnd(std::size_t(block))),
                              std::size_t(block));
                }
                covered = last;
            }
            EXPECT_EQ(covered, plan.end) << "the partitions cover [begin, end)";
        }
}

TEST(NativeSuperblockWalkContract, ExactIntRepresentabilityBoundary) {
    // NEON (begin 0): partitions * width <= INT_MAX.
    EXPECT_TRUE(admitsOrderedPartitionPlan(0, INT_MAX, INT_MAX, 1, INT_MAX));
    EXPECT_FALSE(admitsOrderedPartitionPlan(0, INT_MAX, INT_MAX, 2, std::int64_t(1) << 30)); // product 2^31
    EXPECT_TRUE(admitsOrderedPartitionPlan(0, INT_MAX - 1, INT_MAX, 2, (std::int64_t(1) << 30) - 1)); // 2^31 - 2
    // x86 (begin > 0): begin + partitions * width <= INT_MAX; MAX passes, MAX + 1 fails.
    EXPECT_TRUE(admitsOrderedPartitionArithmetic(1, INT_MAX, 1, INT_MAX - 1));
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(1, INT_MAX, 1, INT_MAX));
    EXPECT_TRUE(admitsOrderedPartitionArithmetic(7, 7 + 37, 37, (INT_MAX - 7) / 37));
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(7, 7 + 37, 37, (INT_MAX - 7) / 37 + 1));
    // Excess partitions near INT_MAX: the hot start begin + p * width of a trailing
    // empty partition must itself fit int, so these plans are genuinely unrepresentable.
    EXPECT_TRUE(admitsOrderedPartitionArithmetic(INT_MAX - 37, INT_MAX, 37, 1));
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(INT_MAX - 36, INT_MAX, 37, 1));
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(INT_MAX - 1, INT_MAX, 3, 1));
    // Empty rows and excess partitions.
    EXPECT_EQ(requireOrderedPartitionWidth(0, 0, 0, 3), 0);
    EXPECT_EQ(requireOrderedPartitionWidth(0, 8, 8, 37), 1);
    EXPECT_EQ(requireOrderedPartitionWidth(5, 13, 16, 37), 1);
    EXPECT_EQ(requireOrderedPartitionWidth(0, INT_MAX, INT_MAX, 1), INT_MAX);
    // Coverage and sign are part of the plan.
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(0, 9, 2, 4));
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(0, 8, 0, 8));
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(0, 8, 2, -1));
    EXPECT_FALSE(admitsOrderedPartitionArithmetic(-1, 8, 2, 8));
    // Walk ranges and phases.
    EXPECT_FALSE(admitsSuperblockWalk(8, 9, 8));
    EXPECT_FALSE(admitsSuperblockWalk(-1, 1, 8));
    EXPECT_FALSE(admitsSuperblockWalk(2, 1, 8));
    EXPECT_FALSE(admitsSuperblockWalk(0, 1, std::size_t(INT_MAX) + 1));
    EXPECT_FALSE(admitsSuperblockWalk(0, 1, 8, 8));
    EXPECT_TRUE(admitsSuperblockWalk(INT_MAX, INT_MAX, INT_MAX, 7));
}

TEST(NativeSuperblockWalkContract, RejectsOriginalStuckRange) {
    EXPECT_DEATH(requireSuperblockWalk(8, 9, 8), "Invalid CPU superblock walk range: first=8 last=9 blocks=8");
    EXPECT_DEATH(requireSuperblockWalk(-1, 1, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(2, 1, 8), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(0, 1, std::size_t(INT_MAX) + 1), "Invalid CPU superblock walk range");
    EXPECT_DEATH(requireSuperblockWalk(0, 1, 8, 8), "Invalid CPU superblock walk range");
}

TEST(NativeSuperblockWalkContract, ProducerRejectsUnrepresentablePlans) {
    // The tile constructor's own producer: the original stuck range and a plan
    // whose canonical width cannot fit the hot int arithmetic are fatal.
    EXPECT_DEATH(static_cast<void>(requireOrderedPartitionWidth(8, 9, 8, 1)),
                 "Invalid CPU superblock walk range: first=8 last=9 blocks=8");
    EXPECT_DEATH(static_cast<void>(requireOrderedPartitionWidth(0, INT_MAX, INT_MAX, 2)),
                 "ordered partition geometry exceeds int partition arithmetic.*partitions=2 width=1073741824 "
                 "partitions\\*width=2147483648");
    EXPECT_DEATH(static_cast<void>(requireOrderedPartitionWidth(0, 8, 8, 0)), "Invalid CPU ordered partition plan");
    EXPECT_DEATH(static_cast<void>(requireOrderedPartitionWidth(0, 8, 8, 1, 8)), "Invalid CPU superblock walk range");
}

#if defined(__AVX2__)
namespace {
/** @brief Minimal packed publication: the kernel entry reads only these before its walk admission. */
struct StuckRangePublication {
    int blocks_per_row = 8;
    [[nodiscard]] const std::uint8_t *interleavedBase() const noexcept { return nullptr; }
};
} // namespace

TEST(NativeSuperblockWalkContract, X86SingleScaleKernelEntryRejectsOriginalStuckRange) {
    // The production kernel entry (not requireSingleScaleTile's recoverable
    // pre-check) admits its walk once per call; the original range is fatal
    // before any weight byte is addressed.
    const StuckRangePublication packed;
    float row[64]{};
    const std::array<const llaminar2::Q8_1Block *, 1> activation{nullptr};
    const std::array<float *, 1> output{row};
    EXPECT_DEATH((multi_scale::x86::singleScalePolicy<7, 8, 1, multi_scale::Reduction::Contiguous,
                      llaminar2::CPUProjectionNumericalPolicy::BackendNative>(packed, activation, output, 0, 8, 9,
                                                                               false, 1, 0)),
                 "Invalid CPU superblock walk range: first=8 last=9 blocks=8");
}
#endif

namespace {
/** @brief One GPU-aligned 16-tile ordered row the x86 CI run (6528b623b) reported as a reversed walk. */
struct ReversedTileCase {
    int blocks;
    int tile;        ///< First tile whose unclamped start passed the row.
    int unclamped;   ///< tile * width, the reported first.
};
// GPU-aligned trees always form sixteen tiles: 35 blocks (K=1120) and 37, 17 and 24 blocks leave empty trailing
// tiles whose unclamped start produced "first=36 last=35", "first=39 last=37", "first=18 last=17" and
// "first=26 last=24" at the x86 single-scale kernel entries.
constexpr int kGPUAlignedTiles = 16;
constexpr std::array<ReversedTileCase, 4> kReportedReversedTiles{{{35, 12, 36}, {37, 13, 39}, {17, 9, 18}, {24, 13, 26}}};

/** @brief Check that the ordered K tiles of one row are admitted walks covering [0, blocks) in order. */
void expectCanonicalTiles(int blocks, int tiles) {
    SCOPED_TRACE(::testing::Message() << "blocks=" << blocks << " tiles=" << tiles);
    const int width = blocks == 0 ? 0 : (blocks + tiles - 1) / tiles;
    int covered = 0;
    for (int tile = 0; tile < tiles; ++tile) {
        // The production producers' exact expressions.
        const int begin = orderedKTileBegin(tile, width, blocks);
        const int end = std::min(begin + width, blocks);
        ASSERT_TRUE(admitsSuperblockWalk(begin, end, std::size_t(blocks))) << "tile " << tile << ": [" << begin
                                                                           << ", " << end << ")";
        requireSuperblockWalk(begin, end, std::size_t(blocks));
        if (begin == end) {
            // An empty tile is exactly the empty walk at the end of the row.
            EXPECT_EQ(begin, blocks) << "tile " << tile;
            continue;
        }
        // Nonempty tiles keep their blocks: increasing, contiguous and unchanged by the clamp.
        ASSERT_EQ(begin, covered) << "tile " << tile;
        ASSERT_EQ(begin, tile * width) << "tile " << tile;
        covered = end;
    }
    EXPECT_EQ(covered, blocks) << "the tiles cover [0, blocks)";
}
} // namespace

TEST(NativeSuperblockWalkContract, OrderedKTileProducerNeverHandsOverAReversedRange) {
    for (const ReversedTileCase &reported : kReportedReversedTiles) {
        const int width = (reported.blocks + kGPUAlignedTiles - 1) / kGPUAlignedTiles;
        // The reported range is what the unclamped producer formed, and it is (still) inadmissible.
        ASSERT_EQ(reported.tile * width, reported.unclamped);
        ASSERT_GT(reported.unclamped, reported.blocks);
        EXPECT_FALSE(admitsSuperblockWalk(reported.unclamped, reported.blocks, std::size_t(reported.blocks)));
        EXPECT_EQ(orderedKTileBegin(reported.tile, width, reported.blocks), reported.blocks);
        expectCanonicalTiles(reported.blocks, kGPUAlignedTiles);
    }
    // Every short row, every tile count up to and beyond the row, including empty rows.
    for (int blocks = 0; blocks <= 160; ++blocks)
        for (int tiles = 1; tiles <= 37; ++tiles)
            expectCanonicalTiles(blocks, tiles);
}

TEST(NativeSuperblockWalkContract, ReportedReversedTileRangesStayFatal) {
    // The producer, not the admission, changed: the reversed ranges themselves remain fatal.
    EXPECT_DEATH(requireSuperblockWalk(36, 35, 35), "Invalid CPU superblock walk range: first=36 last=35 blocks=35");
    EXPECT_DEATH(requireSuperblockWalk(39, 37, 37), "Invalid CPU superblock walk range: first=39 last=37 blocks=37");
    EXPECT_DEATH(requireSuperblockWalk(18, 17, 17), "Invalid CPU superblock walk range: first=18 last=17 blocks=17");
    EXPECT_DEATH(requireSuperblockWalk(26, 24, 24), "Invalid CPU superblock walk range: first=26 last=24 blocks=24");
}

#if defined(__AVX2__)
namespace {
/** @brief Minimal admitted publication of one 64-column chunk: an empty walk addresses no weight byte. */
struct EmptyTilePublication {
    int blocks_per_row = 0;
    int N_padded = 64;
    [[nodiscard]] const std::uint8_t *interleavedBase() const noexcept { return nullptr; }
};
} // namespace

TEST(NativeSuperblockWalkContract, X86SingleScaleEntryAdmitsTheCanonicalEmptyTrailingTile) {
    // The x86 Q5_K leaf receives the producer's empty trailing tile [blocks, blocks) of each reported row. It
    // publishes positive zero, or retains the output when accumulating, exactly as an ordered partial of +0.
    for (const ReversedTileCase &reported : kReportedReversedTiles) {
        SCOPED_TRACE(::testing::Message() << "blocks=" << reported.blocks);
        const EmptyTilePublication packed{.blocks_per_row = reported.blocks};
        const int width = (reported.blocks + kGPUAlignedTiles - 1) / kGPUAlignedTiles;
        const int begin = orderedKTileBegin(reported.tile, width, reported.blocks);
        const int end = std::min(begin + width, reported.blocks);
        ASSERT_EQ(begin, reported.blocks);
        ASSERT_EQ(end, reported.blocks);
        const std::array<const llaminar2::Q8_1Block *, 1> activation{nullptr};
        for (const bool accumulate : {false, true}) {
            float row[64];
            std::fill(std::begin(row), std::end(row), -3.5f);
            const std::array<float *, 1> output{row};
            multi_scale::x86::singleScalePolicy<7, 8, 1, multi_scale::Reduction::Contiguous,
                                                llaminar2::CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                packed, activation, output, 0, begin, end, accumulate, 1, 0);
            for (const float value : row) {
                std::uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                EXPECT_EQ(bits, accumulate ? 0xc0600000u : 0u) << "accumulate=" << accumulate;
            }
        }
        // The recoverable tile pre-check admits the empty tile and rejects the reversed one before the kernel.
        EXPECT_NO_THROW((multi_scale::x86::requireSingleScaleTile<multi_scale::Reduction::Contiguous>(
            packed, 0, begin, end, 1, 0)));
        EXPECT_THROW((multi_scale::x86::requireSingleScaleTile<multi_scale::Reduction::Contiguous>(
                         packed, 0, reported.unclamped, reported.blocks, 1, 0)),
                     std::invalid_argument);
        // The kernel entry itself still terminates on the reversed range.
        float row[64]{};
        const std::array<float *, 1> output{row};
        EXPECT_DEATH((multi_scale::x86::singleScalePolicy<7, 8, 1, multi_scale::Reduction::Contiguous,
                          llaminar2::CPUProjectionNumericalPolicy::GPUAlignedExpert>(
                         packed, activation, output, 0, reported.unclamped, reported.blocks, false, 1, 0)),
                     "Invalid CPU superblock walk range");
    }
}
#endif
