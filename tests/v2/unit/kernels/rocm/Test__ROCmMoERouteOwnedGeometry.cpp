/**
 * @file Test__ROCmMoERouteOwnedGeometry.cpp
 * @brief Device-free closed geometry and capability ownership contracts.
 *
 * Native symbol inspection belongs to the explicit ROCm integration test.
 * These tests instantiate no backend, initialize no device and cannot forge a
 * resource admission merely to exercise a scoped selector on a CPU.
 */
#include "kernels/rocm/gemm/ROCmMoERouteOwnedTuning.h"
#include <gtest/gtest.h>
#include <set>
#include <type_traits>

using namespace llaminar2::rocm;

/** @brief Each declared candidate is a distinct exact existing one-lane width. */
TEST(ROCmMoERouteOwnedGeometry, ClosedIndependentColumnInventory)
{
    std::set<int> columns;
    for (const auto geometry : kMoERouteOwnedGateUpGeometries)
        EXPECT_TRUE(columns.insert(moeRouteOwnedGateUpColumns(geometry)).second);
    EXPECT_EQ(columns, (std::set<int>{64, 128, 256}));
    EXPECT_THROW((void)moeRouteOwnedGateUpColumns(
        static_cast<MoERouteOwnedGateUpGeometry>(255)), std::invalid_argument);
}

/** @brief Resource capabilities and lexical scopes cannot be fabricated or moved across owners. */
TEST(ROCmMoERouteOwnedGeometry, NativeAdmissionAndScopeHaveOneAuthority)
{
    static_assert(!std::is_default_constructible_v<MoERouteOwnedGateUpAdmission>);
    static_assert(!std::is_constructible_v<MoERouteOwnedGateUpAdmission,
        MoERouteOwnedGateUpGeometry, std::uint8_t, int, MoERouteOwnedKernelResources>);
    static_assert(std::is_copy_constructible_v<MoERouteOwnedGateUpAdmission>);
    static_assert(!std::is_copy_constructible_v<ScopedMoERouteOwnedGateUpGeometry>);
    static_assert(!std::is_move_constructible_v<ScopedMoERouteOwnedGateUpGeometry>);
    SUCCEED();
}

/** @brief Validate exact format axes without constructing a device capability or probing hardware. */
TEST(ROCmMoERouteOwnedGeometry, OriginalEnvelopeAndLaunchFamilyAreDistinct)
{
    using namespace llaminar2;
    const auto bit = [](std::uint8_t cb) { return nativeVnniCodebookMaskBit(cb); };
    EXPECT_FALSE(moeRouteOwnedGateUpEnvelopeValid({}));
    EXPECT_FALSE(moeRouteOwnedGateUpEnvelopeValid({bit(19), 0}));
    EXPECT_FALSE(moeRouteOwnedGateUpEnvelopeValid({0, bit(19)}));
    EXPECT_FALSE(moeRouteOwnedGateUpEnvelopeValid({bit(1), bit(19)}));
    EXPECT_FALSE(moeRouteOwnedGateUpEnvelopeValid({bit(19), bit(1)}));
    EXPECT_FALSE(moeRouteOwnedGateUpEnvelopeValid({bit(19), bit(kNativeVnniExpandedInt8MinCodebook)}));
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        NativeVnniExecutionFormatEnvelope envelope;
        const NativeVnniSourceIdentity original{source.metadata->codebook_id, source.metadata->is_superblock, true};
        ASSERT_TRUE(addNativeVnniExecutionFormat(envelope,
            canonicalDeviceVnniFormat(*source.metadata).codebook_id, original, true));
        EXPECT_TRUE(moeRouteOwnedGateUpEnvelopeValid(envelope)) << source.quant_type;
        EXPECT_EQ(envelope.execution_codebook_mask,
            bit(canonicalDeviceVnniFormat(*source.metadata).codebook_id) |
            bit(migrationStableDeviceVnniFormat(*source.metadata).codebook_id));
        const auto launch = moeRouteOwnedGateUpLaunchCodebookMask(envelope.execution_codebook_mask);
        EXPECT_NE(launch, 0u);
        EXPECT_EQ(moeRouteOwnedGateUpLaunchCodebookMask(launch), launch);
    }
    const auto iq4 = bit(4) | bit(kNativeVnniIQ4XSSourceScaleCodebook);
    EXPECT_EQ(moeRouteOwnedGateUpLaunchCodebookMask(iq4), bit(kNativeVnniIQ4XSSourceScaleCodebook));
    EXPECT_EQ(moeRouteOwnedGateUpLaunchCodebookMask(iq4 | bit(19)),
        bit(kNativeVnniIQ4XSSourceScaleCodebook) | bit(19));
    static_assert(std::tuple_size_v<decltype(MoERouteOwnedGateUpLaunchSet::leaves)> == 32);
    static_assert(!std::is_constructible_v<MoERouteOwnedGateUpAdmission,
        MoERouteOwnedGateUpGeometry, NativeVnniExecutionFormatEnvelope, int,
        std::array<MoERouteOwnedKernelResources, 32>>);
}

/** @brief Named shape and exact source arrays size the bank without inventing a KB policy. */
TEST(ROCmMoERouteOwnedGeometry, ParallelSourcePartitionContract)
{
    using namespace llaminar2;
    using P = MoERouteOwnedGateUpPartitionExecution;
    const auto bit = [](std::uint8_t cb) { return nativeVnniCodebookMaskBit(cb); };
    const NativeVnniExecutionFormatEnvelope corrected{bit(7) | bit(kNativeVnniExpandedInt8MinCodebook), bit(7)};
    EXPECT_TRUE(moeRouteOwnedGateUpPartitionExecutionValid(P::Serial, corrected, {}));
    EXPECT_FALSE(moeRouteOwnedGateUpPartitionExecutionValid(P::FourWorkerPlanes, corrected, {}));
    for (const int k : {32, 64, 96, 256, 640, 768, 2560})
    {
        const MoERouteOwnedProjectionShape shape{.output_columns = 96, .reduction_columns = k};
        EXPECT_TRUE(moeRouteOwnedGateUpPartitionExecutionValid(P::Serial, corrected, shape));
        EXPECT_TRUE(moeRouteOwnedGateUpPartitionExecutionValid(P::FourWorkerPlanes, corrected, shape));
    }
    for (const auto shape : {MoERouteOwnedProjectionShape{96, 0}, {0, 2560}, {-1, 2560}, {96, -32}, {96, 31}})
        EXPECT_FALSE(moeRouteOwnedGateUpPartitionExecutionValid(P::FourWorkerPlanes, corrected, shape));
    EXPECT_FALSE(moeRouteOwnedGateUpPartitionExecutionValid(P::Serial, corrected, {96, 0}));
    EXPECT_FALSE(moeRouteOwnedGateUpPartitionExecutionValid(P::Serial, corrected, {0, 2560}));
    EXPECT_FALSE(moeRouteOwnedGateUpPartitionExecutionValid(static_cast<P>(255), corrected, {96, 2560}));
    EXPECT_FALSE(moeRouteOwnedGateUpPartitionExecutionValid(P::FourWorkerPlanes, {}, {96, 2560}));
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        NativeVnniExecutionFormatEnvelope envelope;
        ASSERT_TRUE(addNativeVnniExecutionFormat(envelope,
            canonicalDeviceVnniFormat(*source.metadata).codebook_id,
            {source.metadata->codebook_id, source.metadata->is_superblock, true}, true));
        EXPECT_TRUE(moeRouteOwnedGateUpPartitionExecutionValid(P::FourWorkerPlanes,
            envelope, {.output_columns = 96, .reduction_columns = 2560})) << source.quant_type;
        for (std::uint8_t cb = 0; cb < 32; ++cb)
            if (moeRouteOwnedGateUpLaunchCodebookMask(envelope.execution_codebook_mask) & bit(cb))
                EXPECT_TRUE(moeRouteOwnedGateUpPartitionCodebook(cb));
    }
    for (const auto cb : {1, 2, 3, 18, 20, 21, 22, 25, 31, 255})
        EXPECT_FALSE(moeRouteOwnedGateUpPartitionCodebook(cb));
    std::array<std::uint8_t, 32> partitions{};
    EXPECT_EQ(moeRouteOwnedGateUpMaximumKPartitions(corrected, partitions, 2560), 0);
    for (int count = 1; count <= NativeVNNIGroupedDecodePolicy::maximum_k_partitions; ++count)
    {
        partitions[7] = count;
        EXPECT_EQ(moeRouteOwnedGateUpMaximumKPartitions(corrected, partitions, 2560), count);
        EXPECT_EQ(moeRouteOwnedGateUpPartitionSharedBytes(count), 64u * count * sizeof(float));
    }
    partitions[7] = 65;
    EXPECT_EQ(moeRouteOwnedGateUpMaximumKPartitions(corrected, partitions, 2560), 0);
    partitions[7] = 2;
    EXPECT_EQ(moeRouteOwnedGateUpMaximumKPartitions(corrected, partitions, 32), 0);
    partitions[7] = 1;
    partitions[5] = 1; // Nonowned policy entries cannot become an anonymous bank reserve.
    EXPECT_EQ(moeRouteOwnedGateUpMaximumKPartitions(corrected, partitions, 2560), 0);
    const NativeVnniExecutionFormatEnvelope mixed{bit(5) | bit(7), bit(5) | bit(7)};
    partitions[5] = 3; partitions[7] = 16;
    EXPECT_EQ(moeRouteOwnedGateUpMaximumKPartitions(mixed, partitions, 2560), 16);
    for (int bad : {-1, 0, 65})
        EXPECT_THROW((void)moeRouteOwnedGateUpPartitionSharedBytes(bad), std::invalid_argument);
    EXPECT_EQ(kMoERouteOwnedGateUpPartitionWorkers * kMoERouteOwnedGateUpPartitionColumns, 256);
    static_assert(!std::is_constructible_v<MoERouteOwnedGateUpAdmission,
        MoERouteOwnedGateUpGeometry, NativeVnniExecutionFormatEnvelope, int,
        std::array<MoERouteOwnedKernelResources, 32>, P, MoERouteOwnedProjectionShape,
        std::array<std::uint8_t, 32>, int>);
}

/** @brief The immutable IQ2_S grid and concurrently written partials never alias. */
TEST(ROCmMoERouteOwnedGeometry, AllFormatSharedLayoutHasDisjointGridAndPartials)
{
    using namespace llaminar2;
    using P = MoERouteOwnedGateUpPartitionExecution;
    std::set<std::uint8_t> seen;
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto cb : {canonicalDeviceVnniFormat(*source.metadata).codebook_id,
                              migrationStableDeviceVnniFormat(*source.metadata).codebook_id})
        {
            if (!seen.insert(cb).second) continue;
            const auto serial = moeRouteOwnedGateUpK1SharedLayout(cb, P::Serial);
            const auto grid = cb == 13 ? 1024u * sizeof(std::uint64_t) : 0u;
            EXPECT_EQ(serial.codebook_bytes, grid);
            EXPECT_EQ(serial.partial_offset, grid);
            EXPECT_EQ(serial.partial_bytes, 0u);
            EXPECT_EQ(serial.total_bytes, grid);
            for (int count = 1; count <= NativeVNNIGroupedDecodePolicy::maximum_k_partitions; ++count)
            {
                const auto parallel = moeRouteOwnedGateUpK1SharedLayout(cb, P::FourWorkerPlanes, count);
                EXPECT_EQ(parallel.codebook_bytes, serial.codebook_bytes);
                EXPECT_GE(parallel.partial_offset, parallel.codebook_bytes);
                EXPECT_EQ(parallel.partial_offset % alignof(float), 0u);
                EXPECT_EQ(parallel.partial_bytes, moeRouteOwnedGateUpPartitionSharedBytes(count));
                EXPECT_EQ(parallel.total_bytes, parallel.partial_offset + parallel.partial_bytes);
            }
            EXPECT_THROW((void)moeRouteOwnedGateUpK1SharedLayout(cb, P::FourWorkerPlanes, 0), std::invalid_argument);
            EXPECT_THROW((void)moeRouteOwnedGateUpK1SharedLayout(cb, static_cast<P>(255), 16), std::invalid_argument);
        }
    EXPECT_EQ(seen.size(), 18u);
    EXPECT_THROW((void)moeRouteOwnedGateUpK1SharedLayout(255, P::Serial), std::invalid_argument);
}

/** @brief Complete logical intent is device-free, distinct from native admission and changes on every launch axis. */
TEST(ROCmMoERouteOwnedGeometry, ImmutableOwnerPlanIdentityIsComplete)
{
    using namespace llaminar2;
    using P = MoERouteOwnedGateUpPartitionExecution;
    const NativeVnniExecutionFormatEnvelope envelope{(1u << 5) | (1u << 23), 1u << 5};
    const MoERouteOwnedProjectionShape shape{.output_columns = 96, .reduction_columns = 256};
    std::array<std::uint8_t, 32> partitions{};
    partitions[5] = 8;
    const auto identity = [&](auto geometry, auto formats, int device, auto execution, auto axes, auto source_partitions) {
        return moeRouteOwnedGateUpPlanIdentity(geometry, formats, device, execution, axes, source_partitions);
    };
    const auto original = identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes, shape, partitions);
    ASSERT_NE(original, 0u);
    EXPECT_EQ(original, identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes, shape, partitions));
    EXPECT_NE(original, identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 1, P::FourWorkerPlanes, shape, partitions));
    EXPECT_NE(original, identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::Serial, shape, partitions));
    EXPECT_NE(original, identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes,
        MoERouteOwnedProjectionShape{.output_columns=64, .reduction_columns=256}, partitions));
    EXPECT_NE(original, identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes,
        MoERouteOwnedProjectionShape{.output_columns=96, .reduction_columns=512}, partitions));
    auto different = partitions; different[5] = 4;
    EXPECT_NE(original, identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes, shape, different));
    EXPECT_NE(original, identity(MoERouteOwnedGateUpGeometry::TN64,
        NativeVnniExecutionFormatEnvelope{1u<<5, 1u<<5}, 0, P::FourWorkerPlanes, shape, partitions));
    const auto serial = identity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::Serial, shape, partitions);
    EXPECT_NE(serial, identity(MoERouteOwnedGateUpGeometry::TN128, envelope, 0, P::Serial, shape, partitions));
    static_assert(!std::is_default_constructible_v<MoERouteOwnedGateUpPlan>);
    static_assert(!std::is_constructible_v<MoERouteOwnedGateUpPlan,
        MoERouteOwnedGateUpGeometry, NativeVnniExecutionFormatEnvelope, int, P,
        MoERouteOwnedProjectionShape, std::array<std::uint8_t,32>, int>);
}

/** @brief Malformed logical metadata cannot be relabeled as a complete intent or native capability. */
TEST(ROCmMoERouteOwnedGeometry, ImmutableOwnerPlanRejectsContradictoryAxes)
{
    using namespace llaminar2;
    using P = MoERouteOwnedGateUpPartitionExecution;
    const NativeVnniExecutionFormatEnvelope envelope{1u<<5, 1u<<5};
    const MoERouteOwnedProjectionShape shape{.output_columns=96, .reduction_columns=256};
    std::array<std::uint8_t,32> partitions{}; partitions[5] = 8;
    const auto reject = [&](auto geometry, auto formats, int device, auto execution, auto axes, auto source_partitions) {
        EXPECT_THROW((void)moeRouteOwnedGateUpPlanIdentity(geometry, formats, device, execution, axes, source_partitions), std::invalid_argument);
    };
    reject(MoERouteOwnedGateUpGeometry::TN64, envelope, -1, P::FourWorkerPlanes, shape, partitions);
    reject(MoERouteOwnedGateUpGeometry::TN128, envelope, 0, P::FourWorkerPlanes, shape, partitions);
    reject(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes, MoERouteOwnedProjectionShape{}, partitions);
    reject(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::Serial, MoERouteOwnedProjectionShape{}, partitions);
    reject(MoERouteOwnedGateUpGeometry::TN64, NativeVnniExecutionFormatEnvelope{}, 0, P::Serial, shape, partitions);
    auto missing = partitions; missing[5]=0;
    reject(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes, shape, missing);
    auto foreign = partitions; foreign[7]=8;
    reject(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes, shape, foreign);
    auto oversize = partitions; oversize[5]=9;
    reject(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::FourWorkerPlanes, shape, oversize);
    EXPECT_NE(moeRouteOwnedGateUpPlanIdentity(MoERouteOwnedGateUpGeometry::TN64, envelope, 0, P::Serial, {}, {}), 0u);
}
