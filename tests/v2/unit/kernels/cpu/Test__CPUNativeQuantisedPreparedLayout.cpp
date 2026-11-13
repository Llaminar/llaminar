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
#include "execution/moe/ExpertTierWeightDeviceLayout.h"
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

/** @test View and seal admission reject an unimplemented layout before any division (S1).
 * The zero-stride sentinel of an unimplemented layout's section tables must
 * surface as a precise admission rejection — never as a division operand in
 * the extent overflow guard. The first negative exercises the guard itself:
 * MetadataSeal::validate's structural admission call. The second negative is
 * the published-view structural entry (fromPublished) and proves sealed
 * byte-range identity: the layout mismatch is rejected at seal.admits, not
 * by a second pass through the admission guard. */
TEST(CPUNativeQuantisedPreparedLayout, UnimplementedLayoutIsRejectedByViewAndSealAdmission)
{
    for (const NativeQuantisedFormatInfo *source : kEncodingRepresentatives)
    {
        SCOPED_TRACE(traceSource(*source));
        const NativeQuantisedPreparedByteFixture fixture(*source);
        // The fixture owns canonical bytes; asking the actual seal admission
        // to interpret them under a compact layout must fail closed.
        CPUNativeQuantisedWeightDescriptor compact = fixture.descriptor;
        compact.layout = CPUNativeQuantisedLayout::InterleavedCompactA64;
        EXPECT_THROW(static_cast<void>(CPUNativeQuantisedMetadataSeal::validate(
                         compact, fixture.bytes)),
                     std::invalid_argument);
        // A canonical seal must not admit a compact descriptor either: the
        // layout is part of the sealed byte-range identity.
        const CPUNativeQuantisedMetadataSeal canonical_seal =
            CPUNativeQuantisedMetadataSeal::validate(fixture.descriptor, fixture.bytes);
        EXPECT_THROW(static_cast<void>(CPUNativeQuantisedWeightView::fromPublished(
                         compact, fixture.bytes, canonical_seal)),
                     std::invalid_argument);
    }
}

/** @test Section presence and offsets equal the historical family predicates for all 21 formats.
 * Every producer, copier and accessor now consumes PreparedUnitSections; this
 * pins the tables against the legacy encoding-derived conditions so the
 * presence guards are provably byte-identical for every canonical stream. */
TEST(CPUNativeQuantisedPreparedLayout, SectionPresenceMatchesHistoricalFamilyPredicates)
{
    for (const NativeQuantisedSourceFormat &entry : native_quantised_formats::kAllSourceFormats)
    {
        const NativeQuantisedFormatInfo &source = *entry.metadata;
        for (const bool rotation : {false, true})
        {
            SCOPED_TRACE(std::string{entry.quant_type} + (rotation ? "/rotated" : ""));
            const PreparedTuple tuple(source, rotation);
            const PreparedUnitSections sections = preparedUnitSections(
                kHostPreparedLayout, tuple.encoding, tuple.is_asymmetric, source);
            ASSERT_TRUE(sections.isValid());
            // Presence equals the historical family conditions exactly.
            EXPECT_EQ(sections.has_compensation,
                tuple.encoding != CPUNativeQuantisedEncoding::Q6KNativeDualScale &&
                tuple.encoding != CPUNativeQuantisedEncoding::CompactMultiScale);
            EXPECT_EQ(sections.has_secondary, tuple.is_asymmetric ||
                tuple.encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
                tuple.encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale ||
                tuple.encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
                tuple.encoding == CPUNativeQuantisedEncoding::CompactMultiScale);
            EXPECT_EQ(sections.has_emins,
                tuple.encoding == CPUNativeQuantisedEncoding::CompactMultiScale);
            // Offsets equal the historical unit arithmetic exactly.
            EXPECT_EQ(sections.payload_stride, preparedDataStride(tuple.encoding));
            EXPECT_EQ(sections.block_stride,
                preparedInterleavedBlockStride(tuple.encoding, tuple.is_asymmetric));
            EXPECT_EQ(sections.scales_offset,
                sections.payload_stride + (sections.has_compensation ? 128u : 0u));
            EXPECT_EQ(sections.mins_offset,
                sections.has_secondary ? sections.scales_offset + 128u : 0u);
            EXPECT_EQ(sections.emins_offset,
                sections.has_emins ? sections.payload_stride + 256u : 0u);
        }
    }
}

/** @test Section presence is a source-data contract, not an encoding presence table.
 *
 * The endpoint validators must admit layouts that drop derived aids and
 * optional planes — no compensation, and no embedded minima for super-block
 * sources that do not need them — while still rejecting real data loss: a
 * missing secondary plane, a Q2_K layout without embedded minima, or an
 * invented emins plane outside the super-block family. Compensation presence
 * is owned by the layout tables and never consulted here. */
TEST(CPUNativeQuantisedPreparedLayout, SectionPresenceIsADataRequirementNotAnEncodingTable)
{
    // A minimal geometrically consistent table skeleton: the contract
    // consults presence flags, the geometry only needs to stay orderable.
    const auto syntheticSections = [](bool secondary, bool emins) {
        PreparedUnitSections sections;
        sections.payload_stride = 1024;
        sections.scales_offset = 1024;
        sections.block_stride = 1152 + (secondary ? 128u : 0u) + (emins ? 256u : 0u);
        sections.has_secondary = secondary;
        sections.has_emins = emins;
        return sections;
    };

    // Q6_K without compensation and without emins: the derived x86 aid and
    // the family-forbidden plane are both dropped, and the contract holds.
    {
        const NativeQuantisedFormatInfo &source = native_quantised_formats::Q6_K;
        const auto sections = syntheticSections(true, false);
        EXPECT_TRUE(sectionPresenceSatisfiesSourceContract(
            sections, CPUNativeQuantisedEncoding::Q6KNativeDualScale, true, source));
    }
    // A super-block source without embedded minima may drop the emins plane…
    {
        const NativeQuantisedFormatInfo &source = native_quantised_formats::Q3_K;
        const auto dropped = syntheticSections(true, false);
        EXPECT_TRUE(sectionPresenceSatisfiesSourceContract(
            dropped, CPUNativeQuantisedEncoding::CompactMultiScale, true, source));
        // …and may equally retain the canonical zero-filled plane.
        const auto zeroFilled = syntheticSections(true, true);
        EXPECT_TRUE(sectionPresenceSatisfiesSourceContract(
            zeroFilled, CPUNativeQuantisedEncoding::CompactMultiScale, true, source));
    }
    // Q2_K carries embedded minima: dropping them is real data loss.
    {
        const NativeQuantisedFormatInfo &source = native_quantised_formats::Q2_K;
        ASSERT_TRUE(source.has_emins);
        const auto sections = syntheticSections(true, false);
        EXPECT_FALSE(sectionPresenceSatisfiesSourceContract(
            sections, CPUNativeQuantisedEncoding::CompactMultiScale, true, source));
    }
    // Secondary words are source metadata: Q6_K without the plane loses data.
    {
        const NativeQuantisedFormatInfo &source = native_quantised_formats::Q6_K;
        const auto sections = syntheticSections(false, false);
        EXPECT_FALSE(sectionPresenceSatisfiesSourceContract(
            sections, CPUNativeQuantisedEncoding::Q6KNativeDualScale, true, source));
    }
    // No encoding outside the super-block family may invent embedded minima.
    {
        const NativeQuantisedFormatInfo &source = native_quantised_formats::Q8_0;
        const auto sections = syntheticSections(false, true);
        EXPECT_FALSE(sectionPresenceSatisfiesSourceContract(
            sections, CPUNativeQuantisedEncoding::ExpandedInt8, false, source));
    }
    // Canonical tables satisfy the contract tautologically for all 21 formats.
    for (const NativeQuantisedSourceFormat &entry : native_quantised_formats::kAllSourceFormats)
    {
        const NativeQuantisedFormatInfo &source = *entry.metadata;
        for (const bool rotation : {false, true})
        {
            SCOPED_TRACE(traceSource(source) + (rotation ? "/rotated" : ""));
            const PreparedTuple tuple(source, rotation);
            const PreparedUnitSections sections = preparedUnitSections(
                kHostPreparedLayout, tuple.encoding, tuple.is_asymmetric, source);
            ASSERT_TRUE(sections.isValid());
            EXPECT_TRUE(sectionPresenceSatisfiesSourceContract(
                sections, tuple.encoding, tuple.is_asymmetric, source));
        }
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
        EXPECT_EQ(view.layout(), kHostPreparedLayout);
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

/**
 * @test Device-layout validity is section-authoritative in both directions.
 *
 * The zero-stride sentinel of an unimplemented layout is never a value a
 * stride comparison may match: an otherwise-valid layout that declares the
 * compact CPU layout with both strides zero must be rejected by the
 * device-layout validator exactly as by the storage-layout validator, in
 * both conversion directions. For every cataloged format and both prepared
 * layouts, DeviceLayout::valid() is exactly section completeness (when all
 * other contract fields are consistent) and agrees with cpuStorage().valid()
 * — one authority, the A0/S1 principle applied to the device-layout
 * validator. CUDA/ROCm host chunk admission trusts this validator, so the
 * regression covers both launch directions.
 */
TEST(CPUNativeQuantisedPreparedLayout,
     DeviceLayoutValidityRequiresSectionsAndAgreesWithStorageAuthority)
{
    using Direction = ExpertTierWeightConversionDirection;
    using native_quantised_formats::kAllSourceFormats;

    // Build the complete production field contract for one format, prepared
    // layout, and conversion direction. Demotion mirrors the canonical VNNI
    // source representation; promotion mirrors the per-encoding destination
    // normalization the kernel contract requires.
    const auto build = [](const NativeQuantisedFormatInfo &source,
                          CPUNativeQuantisedLayout layout,
                          Direction direction) {
        const PreparedTuple tuple{source, false};
        const PreparedUnitSections sections =
            preparedUnitSections(layout, tuple.encoding, tuple.is_asymmetric, source);
        const NativeQuantisedMigrationStableDeviceFormat gpu_format =
            canonicalDeviceVnniFormat(source);
        ExpertTierWeightDeviceLayout candidate{};
        candidate.direction = direction;
        candidate.N = 64;
        candidate.K = 32;
        candidate.N_padded = 64;
        candidate.blocks_per_row = 1;
        candidate.cpu_N = 64;
        candidate.cpu_K = 32;
        candidate.cpu_first_row = 0;
        candidate.cpu_first_column = 0;
        candidate.gpu_N = 64;
        candidate.gpu_K = 32;
        candidate.gpu_first_row = 0;
        candidate.gpu_first_column = 0;
        candidate.cpu_codebook_id = source.codebook_id;
        candidate.gpu_source_codebook_id = source.codebook_id;
        candidate.gpu_source_is_superblock = source.is_superblock ? 1u : 0u;
        candidate.gpu_codebook_id = gpu_format.codebook_id;
        candidate.gpu_payload_bytes_per_block =
            gpu_format.payload_bytes_per_block;
        candidate.cpu_is_asymmetric = tuple.is_asymmetric ? 1u : 0u;
        candidate.cpu_is_superblock = source.is_superblock ? 1u : 0u;
        candidate.gpu_is_asymmetric = gpu_format.is_asymmetric ? 1u : 0u;
        candidate.gpu_has_emins = gpu_format.has_emins ? 1u : 0u;
        candidate.cpu_encoding = tuple.encoding;
        candidate.cpu_layout = layout;
        candidate.cpu_data_stride = sections.payload_stride;
        candidate.cpu_block_stride = sections.block_stride;
        candidate.unit_count = 1;
        candidate.maximum_units_per_chunk = 1;
        // The wire-plane contract derives the endpoint planes from the same
        // section tables; unimplemented layouts leave the zero sentinel,
        // which validity rejects exactly like the zero strides.
        if (!candidate.deriveCpuEndpointWirePlanes())
        {
            EXPECT_FALSE(sections.isValid());
        }
        if (direction == Direction::CpuToGpu)
        {
            switch (tuple.encoding)
            {
            case CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale:
                candidate.gpu_payload_bytes_per_block = 13;
                candidate.gpu_is_asymmetric = 1;
                candidate.gpu_has_emins = 0;
                break;
            case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
                candidate.gpu_codebook_id =
                    kNativeQuantisedIQ4XSSourceScaleCodebook;
                candidate.gpu_payload_bytes_per_block = 16;
                candidate.gpu_is_asymmetric = 1;
                candidate.gpu_has_emins = 0;
                break;
            case CPUNativeQuantisedEncoding::NibbleLUT:
                candidate.gpu_payload_bytes_per_block = 16;
                candidate.gpu_is_asymmetric = candidate.cpu_is_asymmetric;
                candidate.gpu_has_emins = 0;
                break;
            case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
                candidate.gpu_payload_bytes_per_block = 24;
                candidate.gpu_is_asymmetric = 1;
                candidate.gpu_has_emins = 0;
                break;
            case CPUNativeQuantisedEncoding::CompactMultiScale:
                candidate.gpu_is_asymmetric = 1;
                candidate.gpu_has_emins = source.has_emins ? 1u : 0u;
                break;
            case CPUNativeQuantisedEncoding::ExpandedInt8:
                candidate.gpu_payload_bytes_per_block = 32;
                candidate.gpu_codebook_id =
                    candidate.cpu_is_asymmetric
                        ? kNativeQuantisedExpandedInt8MinCodebook
                        : static_cast<std::uint8_t>(19);
                candidate.gpu_is_asymmetric = candidate.cpu_is_asymmetric;
                candidate.gpu_has_emins = 0;
                break;
            }
        }
        return candidate;
    };

    for (const auto &entry : kAllSourceFormats)
    {
        const NativeQuantisedFormatInfo &source = *entry.metadata;
        const PreparedTuple tuple{source, false};
        // Every cataloged format promotes: CompactMultiScale sources are
        // asymmetric by construction, and symmetric ExpandedInt8 sources
        // (IQ2_XXS, IQ3_XXS) normalize to codebook 19 exactly as the
        // promotion contract specifies.
        for (const Direction direction :
             {Direction::GpuToCpu, Direction::CpuToGpu})
        {
            for (const CPUNativeQuantisedLayout layout :
                 {CPUNativeQuantisedLayout::InterleavedCanonical,
                  CPUNativeQuantisedLayout::InterleavedCompactA64})
            {
                SCOPED_TRACE(traceSource(source) + " " +
                             (direction == Direction::GpuToCpu ? "demotion"
                                                               : "promotion") +
                             " layout=" +
                             std::to_string(static_cast<int>(layout)));
                const auto candidate = build(source, layout, direction);
                const PreparedUnitSections sections = preparedUnitSections(
                    layout, tuple.encoding, tuple.is_asymmetric, source);
                EXPECT_EQ(candidate.valid(), sections.isValid());
                EXPECT_EQ(candidate.valid(), candidate.cpuStorage().valid());
                auto zeroed = candidate;
                zeroed.cpu_data_stride = 0;
                zeroed.cpu_block_stride = 0;
                EXPECT_FALSE(zeroed.valid());
                EXPECT_FALSE(zeroed.cpuStorage().valid());
            }
        }
    }

    // The named regression: an otherwise-valid Q8_0 demotion that declares
    // the compact CPU layout with zero strides must fail both authorities —
    // the sentinel is not a value a stride comparison may match.
    const auto attack = build(native_quantised_formats::Q8_0,
                              CPUNativeQuantisedLayout::InterleavedCompactA64,
                              Direction::GpuToCpu);
    ASSERT_EQ(attack.cpu_encoding, CPUNativeQuantisedEncoding::ExpandedInt8);
    auto sentinel = attack;
    sentinel.cpu_data_stride = 0;
    sentinel.cpu_block_stride = 0;
    EXPECT_FALSE(sentinel.valid());
    EXPECT_FALSE(sentinel.cpuStorage().valid());
    // Canonical positive control: implemented tables with their exact
    // strides validate on both authorities.
    const auto canonical = build(native_quantised_formats::Q8_0,
                                 CPUNativeQuantisedLayout::InterleavedCanonical,
                                 Direction::GpuToCpu);
    EXPECT_TRUE(canonical.valid());
    EXPECT_TRUE(canonical.cpuStorage().valid());
}
} // namespace llaminar2::test
