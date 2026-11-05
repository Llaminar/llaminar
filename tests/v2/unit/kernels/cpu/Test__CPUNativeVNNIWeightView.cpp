/**
 * @file Test__CPUNativeVNNIWeightView.cpp
 * @brief Admission, layout and original source-factor proof for native CPU read views.
 *
 * Small byte fixtures own one canonical interleaved stream, not a shadow tensor
 * or a second preparation implementation. Tests inject metadata at independently
 * derived layout offsets and compare factor words with memory-rounded arithmetic.
 * Malformed geometry, source identity, spans and coordinates must fail closed.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIWeightView.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include "utils/NativeVNNIPreparedByteFixture.h"

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;

using PreparedBytes = NativeVNNIPreparedByteFixture;

/** @brief Store a complete metadata word without alignment/aliasing assumptions. */
void storeHalf(std::span<std::uint8_t> bytes, std::size_t offset, std::uint16_t word)
{ std::memcpy(bytes.data() + offset, &word, sizeof(word)); }

/** @brief Compare exact FP32 publication words with a precise source-factor trace. */
void expectFactor(float actual, float expected, const char *which)
{
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), std::bit_cast<std::uint32_t>(expected)) << which;
}
} // namespace

/** @test Every source identity admits its exact canonical byte extent and physical strides. */
TEST(CPUNativeVNNIWeightView, AllSourcesKeepTheirCanonicalRootAndMetadataCoordinates)
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(std::string(source.quant_type));
        PreparedBytes fixture(*source.metadata);
        const auto view = fixture.view();
        EXPECT_EQ(view.N, 65);
        EXPECT_EQ(view.N_padded, 128);
        EXPECT_EQ(view.K, 65);
        EXPECT_EQ(view.blocks_per_row, 3);
        EXPECT_EQ(view.codebook_id, source.metadata->codebook_id);
        EXPECT_EQ(view.source_payload_bytes, source.metadata->payload_bytes);
        EXPECT_EQ(view.payload_bytes, view.usesExpandedInt8() ? 32 : source.metadata->payload_bytes);
        EXPECT_EQ(view.is_superblock, source.metadata->is_superblock);
        EXPECT_EQ(view.interleavedBase(), fixture.bytes.data());
        EXPECT_EQ(view.bytes().size(), fixture.bytes.size());
        const auto *unit = fixture.bytes.data() + 5 * view.interleaved_block_stride;
        const std::size_t metadata = view.usesInlineCompensation() ? 128 : 0;
        EXPECT_EQ(reinterpret_cast<const std::uint8_t *>(view.chunkScales(1, 2)), unit + view.data_stride + metadata);
        if (view.is_asymmetric || view.usesNibbleLUTSourceScale() || view.usesIQ3SExpandedSourceScale() ||
            view.usesQ6KNativeDualScale() || view.usesCompactMultiScale())
            EXPECT_EQ(reinterpret_cast<const std::uint8_t *>(view.chunkMins(1, 2)), unit + view.data_stride + metadata + 128);
        else
            EXPECT_THROW((void)view.chunkMins(1, 2), std::logic_error);
        EXPECT_EQ(view.interleavedB(1, 2, 0, 3), unit + 192);
        EXPECT_THROW((void)view.interleavedB(2, 0, 0, 0), std::out_of_range);
        EXPECT_THROW((void)view.interleavedB(0, 3, 0, 0), std::out_of_range);
        EXPECT_THROW((void)view.interleavedB(0, 0, view.data_stride / 256, 0), std::out_of_range);
        EXPECT_THROW((void)view.interleavedB(0, 0, 0, 4), std::out_of_range);
    }
}

/** @test Raw source coefficients reproduce original half-factor arithmetic without new narrowing. */
TEST(CPUNativeVNNIWeightView, OriginalSourceFactorsRetainEveryRoundingBoundary)
{
    const std::array<const NativeVnniFormatInfo *, 4> sources{
        &native_vnni_formats::Q6_K, &native_vnni_formats::IQ2_S,
        &native_vnni_formats::IQ4_XS, &native_vnni_formats::IQ3_S};
    for (const auto *source : sources)
    {
        PreparedBytes fixture(*source, 64, 32);
        const auto view = fixture.view();
        const std::size_t primary = view.data_stride + (view.usesInlineCompensation() ? 128 : 0);
        const std::size_t secondary = primary + 128;
        constexpr std::uint16_t base_word = 0x1235;
        const float base = fp16_to_fp32(base_word);
        for (int word = 0; word <= 0xffff; ++word)
        {
            if (source->codebook_id == 13 && word > 255) break;
            if (source->codebook_id == 4 && word > 63) break;
            if (source->codebook_id == 11 && word > 15) break;
            storeHalf(fixture.bytes, primary, base_word);
            storeHalf(fixture.bytes, secondary, static_cast<std::uint16_t>(word));
            float first = 0, second = 0;
            if (source->codebook_id == 8)
            {
                const int low = (word & 255) - ((word & 128) ? 256 : 0);
                const int high = (word >> 8) - ((word & 32768) ? 256 : 0);
                first = memoryRoundedMultiply(base, static_cast<float>(low));
                second = memoryRoundedMultiply(base, static_cast<float>(high));
            }
            else if (source->codebook_id == 13)
            {
                first = memoryRoundedMultiply(memoryRoundedMultiply(base, 0.5f + (word & 15)), 0.25f);
                second = memoryRoundedMultiply(memoryRoundedMultiply(base, 0.5f + ((word >> 4) & 15)), 0.25f);
            }
            else if (source->codebook_id == 4)
                first = memoryRoundedMultiply(base, static_cast<float>(word - 32));
            else
                first = memoryRoundedMultiply(base, static_cast<float>(2 * word + 1));
            expectFactor(view.blockScale(0, 0, 0), first, "primary");
            expectFactor(view.blockMin(0, 0, 0), second, "secondary");
        }
        if (source->codebook_id != 8)
        {
            const auto invalid = source->codebook_id == 13 ? 256 : source->codebook_id == 4 ? 64 : 16;
            storeHalf(fixture.bytes, secondary, static_cast<std::uint16_t>(invalid));
            EXPECT_THROW((void)view.blockScale(0, 0, 0), std::invalid_argument);
        }
        EXPECT_THROW((void)view.blockScale(0, 0, 64), std::out_of_range);
    }
}

/** @test Missing secondary vectors mean an exact zero, never a speculative metadata read. */
TEST(CPUNativeVNNIWeightView, EverySingleFactorAndRotatedSourceHasBoundedZeroMinimum)
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(std::string(source.quant_type));
        PreparedBytes fixture(*source.metadata, 64, 32);
        const auto view = fixture.view();
        const bool zero_minimum = view.usesNibbleLUTSourceScale() || view.usesIQ3SExpandedSourceScale() ||
            (!view.is_asymmetric && !view.usesQ6KNativeDualScale() && !view.usesCompactMultiScale());
        if (zero_minimum)
            expectFactor(view.blockMin(0, 0, 0), 0.0f, "single-factor zero");
        EXPECT_THROW((void)view.blockMin(-1, 0, 0), std::out_of_range);
        EXPECT_THROW((void)view.blockMin(0, -1, 0), std::out_of_range);
        EXPECT_THROW((void)view.blockMin(1, 0, 0), std::out_of_range);
        EXPECT_THROW((void)view.blockMin(0, 1, 0), std::out_of_range);

        // Rotation's canonical symmetric ExpandedInt8 stream has the same
        // physical span for every original source; provenance stays original.
        PreparedBytes rotated(native_vnni_formats::Q8_0, 64, 32);
        rotated.descriptor.source = {
            .codebook_id = source.metadata->codebook_id,
            .is_superblock = source.metadata->is_superblock, .present = true};
        const auto rotated_view = rotated.view();
        EXPECT_EQ(rotated_view.payload_bytes, 32);
        EXPECT_EQ(rotated_view.source_payload_bytes, source.metadata->payload_bytes);
        expectFactor(rotated_view.blockMin(0, 0, 63), 0.0f, "rotated zero");
        EXPECT_THROW((void)rotated_view.chunkMins(0, 0), std::logic_error);
        EXPECT_THROW((void)rotated_view.blockMin(-1, -1, 0), std::out_of_range);
    }
}

/** @test Invalid input descriptors and borrowed extents never create a readable matrix. */
TEST(CPUNativeVNNIWeightView, RejectsIncompleteForeignAndOverflowingPublications)
{
    PreparedBytes fixture(native_vnni_formats::Q4_0);
    auto descriptor = fixture.descriptor;
    descriptor.source.present = false;
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(descriptor, fixture.bytes), std::invalid_argument);
    descriptor = fixture.descriptor;
    descriptor.source.codebook_id = 255;
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(descriptor, fixture.bytes), std::invalid_argument);
    descriptor = fixture.descriptor;
    descriptor.encoding = CPUNativeVNNIEncoding::Q6KNativeDualScale;
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(descriptor, fixture.bytes), std::invalid_argument);
    descriptor = fixture.descriptor;
    descriptor.numerical_policy = static_cast<CPUProjectionNumericalPolicy>(255);
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(descriptor, fixture.bytes), std::invalid_argument);
    descriptor = fixture.descriptor;
    descriptor.is_asymmetric = true;
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(descriptor, fixture.bytes), std::invalid_argument);
    descriptor = fixture.descriptor;
    descriptor.columns = 0;
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(descriptor, fixture.bytes), std::invalid_argument);
    descriptor = fixture.descriptor;
    descriptor.columns = std::numeric_limits<int>::max();
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(descriptor, fixture.bytes), std::overflow_error);
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(fixture.descriptor, fixture.bytes.first(1)), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(fixture.descriptor, {}), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeVNNIWeightView::fromPublished(fixture.descriptor,
        std::span<const std::uint8_t>(fixture.bytes.data() + 1, fixture.bytes.size())), std::invalid_argument);
}
} // namespace llaminar2::test
