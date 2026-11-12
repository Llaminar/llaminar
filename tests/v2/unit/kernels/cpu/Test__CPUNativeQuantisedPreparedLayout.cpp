/**
 * @file Test__CPUNativeQuantisedPreparedLayout.cpp
 * @brief Layout-spine contract for the prepared NativeQuantised unit tables.
 *
 * Phase 0 of the compact AArch64 prepared layout (design:
 * docs/v2/projects/2026-11/arm-perf-review/08-arm-prepared-weight-layout-design.md)
 * introduces the layout enum, the host selector, the PreparedUnitSections
 * authority and the layout-aware stride overloads without changing one
 * prepared byte. This file proves that table contract itself: the canonical
 * arm of every layout-aware query must reproduce the historical stride tables
 * and section anatomy exactly, every default and production selector must
 * name the canonical layout, and a not-yet-implemented layout must fail
 * closed instead of silently substituting canonical geometry. Byte equality
 * of real packing through the spine is owned by the admission/packing suites;
 * the packed end-to-end checks here only pin packer-to-view layout threading.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedEncoding.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedPackedWeights.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightView.h"
#include "utils/NativeQuantisedPreparedByteFixture.h"

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <string>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_quantised;

/** @brief Resolve the prepared encoding and asymmetry exactly as the packer does. */
struct PreparedTuple
{
    CPUNativeQuantisedEncoding encoding;
    bool is_asymmetric;
    const NativeQuantisedFormatInfo &source;

    PreparedTuple(const NativeQuantisedFormatInfo &source, bool rotation)
        : encoding(preparedEncodingForSource(source, rotation)),
          is_asymmetric(rotation ? false : source.is_asymmetric),
          source(source)
    {
    }
};

/** @brief Six representative sources covering every physical encoding family. */
const std::array<const NativeQuantisedFormatInfo *, 6> kEncodingRepresentatives{{
    &native_quantised_formats::Q4_0,   // NibbleLUT
    &native_quantised_formats::Q8_0,   // ExpandedInt8
    &native_quantised_formats::Q6_K,   // Q6KNativeDualScale
    &native_quantised_formats::Q2_K,   // CompactMultiScale
    &native_quantised_formats::IQ4_XS, // NibbleLUTSourceScale
    &native_quantised_formats::IQ3_S,  // IQ3SExpandedSourceScale
}};

/** @brief Name a source format for SCOPED_TRACE without a catalog wrapper. */
std::string traceSource(const NativeQuantisedFormatInfo &source)
{
    return "codebook " + std::to_string(source.codebook_id) +
        (source.is_superblock ? "/superblock" : "/block");
}
} // namespace

/** @test The layout enum, host selector and every ABI default name the canonical layout. */
TEST(CPUNativeQuantisedPreparedLayout, HostSelectorAndDefaultsAreCanonical)
{
    EXPECT_EQ(kHostPreparedLayout, CPUNativeQuantisedLayout::InterleavedCanonical);
    EXPECT_TRUE(isKnownPreparedLayout(CPUNativeQuantisedLayout::InterleavedCanonical));
    EXPECT_TRUE(isKnownPreparedLayout(CPUNativeQuantisedLayout::InterleavedCompactA64));
    EXPECT_FALSE(isKnownPreparedLayout(static_cast<CPUNativeQuantisedLayout>(2)));

    const CPUNativeQuantisedWeightDescriptor descriptor{};
    EXPECT_EQ(descriptor.layout, CPUNativeQuantisedLayout::InterleavedCanonical);

    // A freshly constructed owner (the archive-load default) is canonical:
    // every existing archive carries canonical bytes.
    const CPUNativeQuantisedPackedWeights owner{};
    EXPECT_EQ(owner.layout, CPUNativeQuantisedLayout::InterleavedCanonical);
    EXPECT_EQ(owner.descriptor().layout, CPUNativeQuantisedLayout::InterleavedCanonical);
    EXPECT_EQ(owner.preparedFootprint().layout, CPUNativeQuantisedLayout::InterleavedCanonical);
}

/** @test Layout-aware strides and sections reproduce the canonical tables for every source. */
TEST(CPUNativeQuantisedPreparedLayout, CanonicalArmReproducesStrideTablesForAllSources)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        for (const bool rotation : {false, true})
        {
            const PreparedTuple tuple(*source.metadata, rotation);
            SCOPED_TRACE(std::string(source.quant_type) + (rotation ? "/rotated" : "/native"));
            const PreparedUnitSections sections = preparedUnitSections(
                CPUNativeQuantisedLayout::InterleavedCanonical, tuple.encoding,
                tuple.is_asymmetric, tuple.source);
            ASSERT_TRUE(sections.isValid());
            EXPECT_EQ(sections.payload_stride, preparedDataStride(tuple.encoding));
            EXPECT_EQ(sections.block_stride,
                preparedInterleavedBlockStride(tuple.encoding, tuple.is_asymmetric));
            EXPECT_EQ(preparedDataStride(CPUNativeQuantisedLayout::InterleavedCanonical,
                      tuple.encoding, tuple.is_asymmetric, tuple.source),
                sections.payload_stride);
            EXPECT_EQ(preparedInterleavedBlockStride(CPUNativeQuantisedLayout::InterleavedCanonical,
                      tuple.encoding, tuple.is_asymmetric, tuple.source),
                sections.block_stride);
        }
}

/** @test The canonical section anatomy holds for every encoding family. */
TEST(CPUNativeQuantisedPreparedLayout, CanonicalSectionAnatomyMatchesEveryEncoding)
{
    for (const NativeQuantisedFormatInfo *source : kEncodingRepresentatives)
    {
        const PreparedTuple tuple(*source, false);
        SCOPED_TRACE(traceSource(*source));
        const PreparedUnitSections sections = preparedUnitSections(
            CPUNativeQuantisedLayout::InterleavedCanonical, tuple.encoding,
            tuple.is_asymmetric, tuple.source);

        const bool compensation = tuple.encoding != CPUNativeQuantisedEncoding::Q6KNativeDualScale &&
            tuple.encoding != CPUNativeQuantisedEncoding::CompactMultiScale;
        EXPECT_EQ(sections.has_compensation, compensation);
        if (compensation)
            EXPECT_EQ(sections.compensation_offset, sections.payload_stride);

        EXPECT_EQ(sections.scales_offset,
            sections.payload_stride + (compensation ? 128u : 0u));

        const bool secondary = tuple.is_asymmetric ||
            tuple.encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
            tuple.encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale ||
            tuple.encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
            tuple.encoding == CPUNativeQuantisedEncoding::CompactMultiScale;
        EXPECT_EQ(sections.has_secondary, secondary);
        if (secondary)
            EXPECT_EQ(sections.mins_offset, sections.scales_offset + 128u);

        const bool emins = tuple.encoding == CPUNativeQuantisedEncoding::CompactMultiScale;
        EXPECT_EQ(sections.has_emins, emins);
        if (emins)
            EXPECT_EQ(sections.emins_offset, sections.payload_stride + 256u);

        // The unit covers every advertised section end exactly.
        EXPECT_GE(sections.block_stride, sections.scales_offset + 128u);
        if (secondary)
            EXPECT_GE(sections.block_stride, sections.mins_offset + 128u);
        if (emins)
            EXPECT_GE(sections.block_stride, sections.emins_offset + 256u);
    }
}

/** @test The pinned canonical unit strides stay exactly the historical values. */
TEST(CPUNativeQuantisedPreparedLayout, CanonicalUnitStridesStayPinned)
{
    struct Pinned
    {
        std::uint8_t codebook_id;
        bool is_superblock;
        std::uint32_t data_stride;
        std::uint32_t block_stride;
    };
    constexpr Pinned pinned[] = {
        {0, false, 1024, 1280},  // Q4_0 symmetric nibble
        {5, false, 1024, 1408},  // Q4_1 asymmetric nibble
        {5, true, 1024, 1408},   // Q4_K asymmetric nibble superblock
        {19, false, 2048, 2304}, // Q8_0 symmetric expanded
        {7, false, 2048, 2432},  // Q5_1 asymmetric expanded
        {7, true, 2048, 2432},   // Q5_K asymmetric expanded superblock
        {8, true, 1536, 1792},   // Q6_K
        {10, true, 1024, 1536},  // Q2_K multi-scale
        {4, true, 1024, 1408},   // IQ4_XS source-scale nibble
        {11, true, 2048, 2432},  // IQ3_S expanded source-scale
        {21, true, 2048, 2304},  // Q8_K symmetric expanded superblock
    };
    for (const Pinned &entry : pinned)
    {
        const NativeQuantisedFormatInfo *source =
            native_quantised_formats::forSourceIdentity(entry.codebook_id, entry.is_superblock);
        ASSERT_NE(source, nullptr);
        SCOPED_TRACE(traceSource(*source));
        const PreparedTuple tuple(*source, false);
        const PreparedUnitSections sections = preparedUnitSections(
            CPUNativeQuantisedLayout::InterleavedCanonical, tuple.encoding,
            tuple.is_asymmetric, tuple.source);
        ASSERT_TRUE(sections.isValid());
        EXPECT_EQ(sections.payload_stride, entry.data_stride);
        EXPECT_EQ(sections.block_stride, entry.block_stride);
    }
}

/** @test An unimplemented layout fails closed everywhere instead of aliasing canonical geometry. */
TEST(CPUNativeQuantisedPreparedLayout, UnimplementedLayoutFailsClosed)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        for (const bool rotation : {false, true})
        {
            const PreparedTuple tuple(*source.metadata, rotation);
            SCOPED_TRACE(std::string(source.quant_type) + (rotation ? "/rotated" : "/native"));
            const PreparedUnitSections sections = preparedUnitSections(
                CPUNativeQuantisedLayout::InterleavedCompactA64, tuple.encoding,
                tuple.is_asymmetric, tuple.source);
            EXPECT_FALSE(sections.isValid());
            EXPECT_EQ(sections.payload_stride, 0u);
            EXPECT_EQ(sections.block_stride, 0u);
            EXPECT_EQ(preparedDataStride(CPUNativeQuantisedLayout::InterleavedCompactA64,
                      tuple.encoding, tuple.is_asymmetric, tuple.source), 0u);
            EXPECT_EQ(preparedInterleavedBlockStride(CPUNativeQuantisedLayout::InterleavedCompactA64,
                      tuple.encoding, tuple.is_asymmetric, tuple.source), 0u);

            // Footprint pricing fails closed the same way: an invalid
            // footprint can never be mistaken for a canonical estimate.
            const NativeQuantisedPreparedFootprint footprint =
                preparedFootprintForFormat(*source.metadata, rotation,
                    CPUNativeQuantisedLayout::InterleavedCompactA64);
            EXPECT_FALSE(footprint.isValid());
            EXPECT_EQ(footprint.weight_bytes_per_n_chunk_k_block, 0u);
        }
}

/** @test The production estimator carries the layout and prices canonical bytes exactly. */
TEST(CPUNativeQuantisedPreparedLayout, FootprintCarriesLayoutAndCanonicalBytes)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        for (const bool rotation : {false, true})
        {
            const PreparedTuple tuple(*source.metadata, rotation);
            SCOPED_TRACE(std::string(source.quant_type) + (rotation ? "/rotated" : "/native"));
            const NativeQuantisedPreparedFootprint footprint =
                preparedFootprintForFormat(*source.metadata, rotation);
            ASSERT_TRUE(footprint.isValid());
            EXPECT_EQ(footprint.layout, CPUNativeQuantisedLayout::InterleavedCanonical);
            EXPECT_EQ(footprint.layout, kHostPreparedLayout);
            EXPECT_EQ(footprint.encoding, tuple.encoding);
            EXPECT_EQ(footprint.is_asymmetric, tuple.is_asymmetric);
            EXPECT_EQ(footprint.weight_bytes_per_n_chunk_k_block,
                preparedInterleavedBlockStride(tuple.encoding, tuple.is_asymmetric));
        }
}

/** @test A real packed stream threads the canonical layout from packer to sealed view. */
TEST(CPUNativeQuantisedPreparedLayout, PackedStreamThreadsCanonicalLayoutIntoView)
{
    for (const NativeQuantisedFormatInfo *source : kEncodingRepresentatives)
    {
        SCOPED_TRACE(traceSource(*source));
        const NativeQuantisedPreparedByteFixture fixture(*source);
        const CPUNativeQuantisedWeightView view = fixture.view();
        EXPECT_EQ(view.layout, kHostPreparedLayout);
        // The hot metadata delta is exactly the canonical compensation
        // presence the encoding family implies.
        const PreparedTuple tuple(*source, false);
        const PreparedUnitSections sections = preparedUnitSections(
            kHostPreparedLayout, tuple.encoding, tuple.is_asymmetric, tuple.source);
        EXPECT_EQ(view.usesInlineCompensation(), sections.has_compensation);
        // The view's published coordinates are the section authority's.
        EXPECT_EQ(view.data_stride, static_cast<int>(sections.payload_stride));
        EXPECT_EQ(view.interleaved_block_stride, static_cast<int>(sections.block_stride));
    }
}
} // namespace llaminar2::test
