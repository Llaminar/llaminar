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
