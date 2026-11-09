/**
 * @file Test__CPUNativeQuantisedMetadataSeal.cpp
 * @brief Device-free contract and lifecycle tests of prepared CPU metadata admission.
 *
 * The seal replaced per-K-block reserved-bit tests inside native worker teams.
 * These tests prove that it covers exactly the words those checks covered, for
 * every cataloged source, padding lanes included, and that the owner of the
 * bytes withdraws admission on every mutation: rewrites, storage replacement,
 * resizing and moves. Slot reuse at an unchanged address is a rewrite.
 * Copies allocate through the physical-memory authority, so their seal
 * inheritance is tested in the core-linked expert slot suite instead.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedPackedWeights.h"
#include "utils/NativeQuantisedPreparedByteFixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
using namespace llaminar2::cpu::native_quantised;

namespace
{
/// Whether arbitrary code can move a seal onto other bytes (it must not).
template <class Seal> concept CanRebindSeal = requires(const Seal &seal, std::span<const std::uint8_t> bytes) {
    seal.rebound(bytes);
};
struct PublicRebindControl { PublicRebindControl rebound(std::span<const std::uint8_t>) const; };
static_assert(CanRebindSeal<PublicRebindControl>, "positive control: the concept detects a public rebind");
static_assert(!CanRebindSeal<CPUNativeQuantisedMetadataSeal>,
    "only CPUNativeQuantisedPublishedBytes may rebind a seal, while deep-copying bytes it owns");

/** @return Byte offset of a unit's secondary metadata word for one column. */
std::size_t secondaryOffset(const CPUNativeQuantisedWeightView &view, int chunk, int block, int column)
{
    const std::size_t unit = (std::size_t(chunk) * view.blocks_per_row + block) * view.interleaved_block_stride;
    return unit + view.data_stride + (view.usesInlineCompensation() ? 256 : 128) + std::size_t(column) * 2;
}

/** @return Whether the encoding owns a secondary metadata word. */
bool hasSecondary(const CPUNativeQuantisedWeightView &view)
{
    return view.is_asymmetric || view.usesNibbleLUTSourceScale() || view.usesIQ3SExpandedSourceScale() ||
        view.usesQ6KNativeDualScale() || view.usesCompactMultiScale();
}

/** @return Reserved-bit mask the former in-loop check enforced, or zero where none existed. */
std::uint16_t reservedMask(const CPUNativeQuantisedWeightView &view)
{
    if (view.usesIQ3SExpandedSourceScale()) return 0xfff0u;
    if (view.usesNibbleLUTSourceScale()) return 0xffc0u;
    if (view.usesCompactMultiScale() && view.codebook_id == 13) return 0xff00u;
    return 0;
}

/** @brief Store one word without alignment assumptions. */
void store(std::span<std::uint8_t> bytes, std::size_t offset, std::uint16_t word)
{ std::memcpy(bytes.data() + offset, &word, sizeof(word)); }

/** @brief Build a sealed owner whose bytes copy one seeded fixture. */
CPUNativeQuantisedPackedWeights owner(const NativeQuantisedPreparedByteFixture &fixture)
{
    const auto view = fixture.view();
    CPUNativeQuantisedPackedWeights result;
    result.N = view.N; result.K = view.K; result.N_padded = view.N_padded;
    result.blocks_per_row = view.blocks_per_row;
    result.payload_bytes = view.source_payload_bytes;
    result.codebook_id = view.codebook_id; result.is_superblock = view.is_superblock;
    result.is_asymmetric = view.is_asymmetric; result.encoding = view.encoding;
    result.numerical_policy = view.numerical_policy; result.data_stride = view.data_stride;
    result.interleaved_block_stride = view.interleaved_block_stride;
    result.native_interleaved.resize(fixture.bytes.size());
    const auto bytes = result.native_interleaved.beginRewrite();
    std::copy(fixture.bytes.begin(), fixture.bytes.end(), bytes.begin());
    result.sealPublication();
    return result;
}
} // namespace

/**
 * @test The seal enforces exactly the reserved-bit rules execution used to test
 * per block, on every lane of every unit (padding columns included), and nothing more.
 */
TEST(CPUNativeQuantisedMetadataSeal, CoversExactlyTheFormerInLoopRulesOnEveryLane)
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(std::string(source.quant_type));
        NativeQuantisedPreparedByteFixture fixture(*source.metadata, 65, 65);
        fixture.seedPayloadAndMetadata();
        const auto view = fixture.view();
        if (!hasSecondary(view))
            continue;
        const std::uint16_t mask = reservedMask(view);
        // Column 63 of chunk 1 is padding (N=65): wider native loads read it.
        for (const auto [chunk, block, column] : {std::array{0, 0, 0}, std::array{1, 2, 63}})
        {
            const auto offset = secondaryOffset(view, chunk, block, column);
            std::uint16_t original = 0;
            std::memcpy(&original, fixture.bytes.data() + offset, 2);
            store(fixture.bytes, offset, static_cast<std::uint16_t>(original | 0x8000u));
            if (mask)
                EXPECT_THROW((void)CPUNativeQuantisedMetadataSeal::validate(fixture.descriptor, fixture.bytes),
                    std::invalid_argument) << chunk << "/" << block << "/" << column;
            else
                EXPECT_NO_THROW((void)CPUNativeQuantisedMetadataSeal::validate(fixture.descriptor, fixture.bytes));
            store(fixture.bytes, offset, original);
        }
        if (mask)
        {
            // The largest permitted value is still admitted.
            const auto offset = secondaryOffset(view, 0, 1, 7);
            store(fixture.bytes, offset, static_cast<std::uint16_t>(~mask));
            EXPECT_NO_THROW((void)CPUNativeQuantisedMetadataSeal::validate(fixture.descriptor, fixture.bytes));
        }
    }
}

/**
 * @test Raw poisoned padding in a partial final chunk is rejected at
 * publication; the same arrival after producer normalization (benign padding
 * written in the owned generation) is accepted, with live words untouched.
 */
TEST(CPUNativeQuantisedMetadataSeal, PaddingNormalizationSeparatesRawFromNormalizedArrivals)
{
    for (const auto *source : {&native_quantised_formats::IQ3_S, &native_quantised_formats::IQ4_XS, &native_quantised_formats::IQ2_S,
                               &native_quantised_formats::Q6_K, &native_quantised_formats::Q4_K})
        for (const int n : {1, 33, 63, 65, 97, 127})
        {
            SCOPED_TRACE(::testing::Message() << "codebook=" << int(source->codebook_id) << " n=" << n);
            NativeQuantisedPreparedByteFixture fixture(*source, n, 96);
            fixture.seedPayloadAndMetadata();
            auto packed = owner(fixture);
            const auto view = packed.readView();
            const bool rules = reservedMask(view) != 0;
            const int last = view.N_padded / 64 - 1, live = n % 64;
            // An arrival writes live columns only; the slot's padding holds garbage.
            const auto written = packed.native_interleaved.beginRewrite();
            for (int block = 0; block < view.blocks_per_row; ++block)
                for (int column = live; column < 64; ++column)
                    store(written, secondaryOffset(view, last, block, column), 0xffffu);
            const std::vector<std::uint8_t> raw(packed.native_interleaved.begin(), packed.native_interleaved.end());
            if (rules)
                EXPECT_THROW(packed.sealPublication(), std::invalid_argument);
            packed.normalizePaddingMetadata();
            EXPECT_NO_THROW(packed.sealPublication());
            for (std::size_t byte = 0; byte < raw.size(); ++byte)
            {
                const std::size_t unit = byte / std::size_t(view.interleaved_block_stride);
                const std::size_t offset = byte % std::size_t(view.interleaved_block_stride);
                const std::size_t secondary = std::size_t(view.data_stride) + (view.usesInlineCompensation() ? 256 : 128);
                const bool padding_word = int(unit / std::size_t(view.blocks_per_row)) == last &&
                    offset >= secondary + std::size_t(live) * 2 && offset < secondary + 128;
                ASSERT_EQ(packed.native_interleaved[byte], padding_word ? 0 : raw[byte]) << "byte " << byte;
            }
            EXPECT_DEATH(packed.normalizePaddingMetadata(), "sealed generation");
        }
}

/** @test Normalization trusts only descriptor-owned geometry, never a raw byte size. */
TEST(CPUNativeQuantisedMetadataSeal, PaddingNormalizationRejectsForeignGeometry)
{
    NativeQuantisedPreparedByteFixture fixture(native_quantised_formats::IQ3_S, 33, 64);
    fixture.seedPayloadAndMetadata();
    auto packed = owner(fixture);
    (void)packed.native_interleaved.beginRewrite();
    auto foreign = packed.blocks_per_row;
    packed.blocks_per_row = foreign + 1;
    EXPECT_THROW(packed.normalizePaddingMetadata(), std::logic_error);
    packed.blocks_per_row = foreign;
    packed.N_padded += 64;
    EXPECT_THROW(packed.normalizePaddingMetadata(), std::logic_error);
    packed.N_padded -= 64;
    EXPECT_NO_THROW(packed.normalizePaddingMetadata());
}

/** @test A view needs a seal of exactly its bytes and interpretation; arithmetic policy is not part of it. */
TEST(CPUNativeQuantisedMetadataSeal, ViewRequiresTheSealOfItsExactBytesAndInterpretation)
{
    NativeQuantisedPreparedByteFixture first(native_quantised_formats::IQ3_S, 64, 64), second(native_quantised_formats::IQ3_S, 64, 64);
    first.seedPayloadAndMetadata();
    second.seedPayloadAndMetadata();
    const auto seal = CPUNativeQuantisedMetadataSeal::validate(first.descriptor, first.bytes);
    EXPECT_NO_THROW((void)CPUNativeQuantisedWeightView::fromPublished(first.descriptor, first.bytes, seal));
    EXPECT_THROW((void)CPUNativeQuantisedWeightView::fromPublished(second.descriptor, second.bytes, seal), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeQuantisedWeightView::fromPublished(first.descriptor, first.bytes.first(first.bytes.size() - 1), seal),
        std::invalid_argument);
    auto other = first.descriptor;
    other.columns = 63;
    EXPECT_THROW((void)CPUNativeQuantisedWeightView::fromPublished(other, first.bytes, seal), std::invalid_argument);
    other = first.descriptor;
    other.numerical_policy = CPUProjectionNumericalPolicy::BackendNative;
    EXPECT_NO_THROW((void)CPUNativeQuantisedWeightView::fromPublished(other, first.bytes, seal));
    EXPECT_TRUE(CPUNativeQuantisedWeightView::metadata_admitted);
}

/** @test Every mutation of owned bytes withdraws admission until a producer reseals. */
TEST(CPUNativeQuantisedMetadataSeal, EveryMutationWithdrawsAdmission)
{
    NativeQuantisedPreparedByteFixture fixture(native_quantised_formats::IQ4_XS, 65, 96);
    fixture.seedPayloadAndMetadata();
    auto packed = owner(fixture);
    const auto descriptor = packed.descriptor();
    ASSERT_TRUE(packed.publicationSealed());
    EXPECT_NO_THROW((void)packed.readView());

    const auto generation = packed.native_interleaved.generation();
    (void)packed.native_interleaved.beginRewrite();
    EXPECT_GT(packed.native_interleaved.generation(), generation);
    EXPECT_FALSE(packed.publicationSealed());
    packed.sealPublication();
    EXPECT_TRUE(packed.publicationSealed());

    packed.native_interleaved.resize(packed.native_interleaved.size());
    EXPECT_FALSE(packed.publicationSealed());
    packed.sealPublication();

    AlignedVector<std::uint8_t> replacement(packed.native_interleaved.size());
    std::copy(packed.native_interleaved.begin(), packed.native_interleaved.end(), replacement.data());
    packed.native_interleaved.replace(std::move(replacement));
    EXPECT_FALSE(packed.publicationSealed());
    packed.sealPublication();

    // Copies (which allocate through the physical-memory authority) are
    // covered by the core-linked CPUNativeQuantisedPublicationCopy regression.
    // A move keeps the root and admission; the source is empty and unadmitted.
    const auto *root = packed.native_interleaved.data();
    auto moved = std::move(packed);
    EXPECT_EQ(moved.native_interleaved.data(), root);
    EXPECT_TRUE(moved.publicationSealed());
    EXPECT_TRUE(packed.native_interleaved.empty());
    EXPECT_FALSE(packed.native_interleaved.sealedFor(descriptor));

    moved.native_interleaved.release();
    EXPECT_FALSE(moved.publicationSealed());
}

/** @test Reserved bits written by a rewrite fail the reseal and leave the publication unadmitted. */
TEST(CPUNativeQuantisedMetadataSeal, RejectedResealLeavesThePublicationUnadmitted)
{
    NativeQuantisedPreparedByteFixture fixture(native_quantised_formats::IQ3_S, 64, 64);
    fixture.seedPayloadAndMetadata();
    auto packed = owner(fixture);
    const auto view = packed.readView();
    const auto bytes = packed.native_interleaved.beginRewrite();
    store(bytes, secondaryOffset(view, 0, 1, 5), 0x0010u);
    EXPECT_THROW(packed.sealPublication(), std::invalid_argument);
    EXPECT_FALSE(packed.publicationSealed());
    EXPECT_DEATH((void)packed.readView(), "not admitted");
}

/** @test Execution over unsealed or stale bytes is fatal at binding, never a retry. */
TEST(CPUNativeQuantisedMetadataSeal, UnsealedAndStalePublicationsAreFatal)
{
    NativeQuantisedPreparedByteFixture fixture(native_quantised_formats::Q4_K, 64, 256);
    fixture.seedPayloadAndMetadata();
    auto packed = owner(fixture);
    CPUNativeQuantisedPackedWeights never;
    never.N = packed.N; never.K = packed.K; never.N_padded = packed.N_padded;
    never.blocks_per_row = packed.blocks_per_row; never.payload_bytes = packed.payload_bytes;
    never.codebook_id = packed.codebook_id; never.is_superblock = packed.is_superblock;
    never.encoding = packed.encoding; never.data_stride = packed.data_stride;
    never.interleaved_block_stride = packed.interleaved_block_stride;
    never.native_interleaved.resize(packed.native_interleaved.size());
    EXPECT_DEATH((void)never.readView(), "never sealed");
    (void)packed.native_interleaved.beginRewrite();
    EXPECT_DEATH((void)packed.readView(), "stale generation");
}

/**
 * @test Slot reuse: rewriting a sealed slot at the same address, as an overlay
 * acquisition does, is unadmitted until the arrival reseals it.
 */
TEST(CPUNativeQuantisedMetadataSeal, SlotRewriteAtTheSameAddressNeedsAReseal)
{
    NativeQuantisedPreparedByteFixture first(native_quantised_formats::IQ2_S, 64, 256), second(native_quantised_formats::IQ2_S, 64, 256);
    first.seedPayloadAndMetadata();
    second.seedPayloadAndMetadata();
    auto slot = owner(first);
    const auto root = slot.native_interleaved.retainedSlotRoot();
    EXPECT_TRUE(slot.publicationSealed()); // Retaining the destination identity is not a rewrite.

    const auto destination = slot.native_interleaved.beginRewrite(); // Acquisition for a new epoch.
    ASSERT_EQ(destination.data(), root.data());
    std::copy(second.bytes.begin(), second.bytes.end(), destination.begin());
    EXPECT_DEATH((void)slot.readView(), "stale generation");
    slot.sealPublication(); // Arrival admits the complete rewrite.
    EXPECT_EQ(slot.readView().interleavedBase(), root.data());
}
} // namespace llaminar2::test
