/**
 * @file Test__PackedWeightsArchiveLayoutIdentity.cpp
 * @brief Versioned archive identity: the prepared layout is sealed, hashed
 *        by version and re-derived before any payload is adopted.
 *
 * A3 of the compact AArch64 prepared layout (design:
 * docs/v2/projects/2026-11/arm-perf-review/08-arm-prepared-weight-layout-design.md)
 * promotes the CPU endpoint layout into every versioned identity before any
 * compact section table can become valid. For the packed-weight archive
 * family this file proves: version 8 seals the layout byte into the header
 * and the receiver re-derives both strides from that layout's section
 * tables; exactly version 7 remains admissible as an explicitly canonical
 * archive (a versioned statement, never a guess); version 6 and earlier
 * reject globally, not per-format; an unknown layout byte, a tampered
 * stride, and the compact layout byte with canonical strides all reject
 * before a single payload byte is adopted. The compact byte rejection is
 * the sequencing rule itself: compact tables stay unadmitted until the
 * identity binding that carries them is complete.
 */
#include "kernels/PackedWeightsSerialization.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedEncoding.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedPackedWeights.h"
#include "tensors/NativeQuantisedFormatInfo.h"

#include <gtest/gtest.h>
#include <cstring>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace ::llaminar2;
using PackedWeightsHeader = ::llaminar2::PackedWeightsHeader;
using packed_weights_serialization::PackedWeightsTransferDescriptor;
using packed_weights_serialization::PackedWeightsConstSectionViews;
using CPUNativeQuantisedPackedWeights =
    cpu::native_quantised::CPUNativeQuantisedPackedWeights;
using CPUPackedWeights = cpu::native_quantised::CPUPackedWeights;

/** @brief One real canonical Q4_0 owner whose strides the tables reproduce. */
CPUNativeQuantisedPackedWeights canonicalQ4_0Owner()
{
    const auto &source = native_quantised_formats::Q4_0;
    const auto sections = cpu::native_quantised::preparedUnitSections(
        cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical,
        cpu::native_quantised::preparedEncodingForSource(source),
        source.is_asymmetric, source);
    CPUNativeQuantisedPackedWeights packed;
    packed.N = 128;
    packed.K = 256;
    packed.N_padded = 128;
    packed.blocks_per_row = 8;
    packed.codebook_id = source.codebook_id;
    packed.payload_bytes = 16;
    packed.is_asymmetric = source.is_asymmetric;
    packed.encoding = cpu::native_quantised::preparedEncodingForSource(source);
    packed.is_superblock = source.is_superblock;
    packed.data_stride = static_cast<int>(sections.payload_stride);
    packed.interleaved_block_stride = static_cast<int>(sections.block_stride);
    packed.native_interleaved.resize_uninitialized(
        static_cast<size_t>(packed.N_padded / 64) * packed.blocks_per_row *
        sections.block_stride);
    auto bytes = packed.native_interleaved.unsealedGenerationBytes();
    for (size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<uint8_t>((index * 29u + 7u) & 0xffu);
    packed.payload.resize(static_cast<size_t>(packed.blocks_per_row) * 64u * 16u);
    for (size_t index = 0; index < packed.payload.size(); ++index)
        packed.payload[index] = static_cast<uint8_t>((index * 13u + 5u) & 0xffu);
    return packed;
}

/** @brief Serialize one owner into a standard byte vector. */
std::vector<uint8_t> blobOf(
    const CPUNativeQuantisedPackedWeights &packed)
{
    std::vector<uint8_t> blob;
    if (!packed_weights_serialization::serializeInto(packed, blob))
        blob.clear();
    return blob;
}

} // namespace

/** @test v8 seals the canonical layout byte and the round trip restores it. */
TEST(PackedWeightsArchiveLayoutIdentity, V8SealsLayoutAndRoundTripsCanonical)
{
    auto packed = canonicalQ4_0Owner();
    auto blob = blobOf(packed);
    ASSERT_FALSE(blob.empty());
    // The sealed identity: canonical is byte zero, and the header carries it.
    PackedWeightsHeader header{};
    std::memcpy(&header, blob.data(), sizeof(header));
    EXPECT_EQ(header.version, PACKED_WEIGHTS_VERSION);
    EXPECT_EQ(header.prepared_layout,
        static_cast<uint8_t>(
            cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical));

    auto restored = packed_weights_serialization::deserialize(
        blob.data(), blob.size());
    ASSERT_NE(restored, nullptr);
    const auto *const cpu = dynamic_cast<const CPUPackedWeights *>(
        restored.get());
    ASSERT_NE(cpu, nullptr);
    EXPECT_EQ(cpu->packed().layout,
        cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical);
    EXPECT_EQ(cpu->packed().data_stride, packed.data_stride);
    EXPECT_EQ(cpu->packed().interleaved_block_stride,
        packed.interleaved_block_stride);
    EXPECT_EQ(std::memcmp(cpu->packed().native_interleaved.data(),
        packed.native_interleaved.data(),
        packed.native_interleaved.size()), 0);
}

/** @test Exactly v7 is explicit canonical; v6 and earlier reject globally. */
TEST(PackedWeightsArchiveLayoutIdentity, ExactlyV7ExplicitCanonicalEarlierRejectGlobally)
{
    auto packed = canonicalQ4_0Owner();
    auto blob = blobOf(packed);
    ASSERT_FALSE(blob.empty());

    // Exactly version 7: the implied layout is the canonical selector by
    // statement — the byte did not exist on that wire, so v7 admission
    // names canonical explicitly rather than guessing from the strides.
    auto v7 = blob;
    const uint32_t v7_version = 7;
    std::memcpy(v7.data() + offsetof(PackedWeightsHeader, version),
        &v7_version, sizeof(v7_version));
    const uint8_t zero_layout = 0;
    std::memcpy(v7.data() + offsetof(PackedWeightsHeader, prepared_layout),
        &zero_layout, sizeof(zero_layout));
    auto restored = packed_weights_serialization::deserialize(
        v7.data(), v7.size());
    ASSERT_NE(restored, nullptr);
    const auto *const cpu = dynamic_cast<const CPUPackedWeights *>(
        restored.get());
    ASSERT_NE(cpu, nullptr);
    EXPECT_EQ(cpu->packed().layout,
        cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical);

    // Every earlier version rejects globally — never per-format.
    for (const uint32_t early : {1u, 2u, 3u, 4u, 5u, 6u})
    {
        auto legacy = blob;
        std::memcpy(legacy.data() + offsetof(PackedWeightsHeader, version),
            &early, sizeof(early));
        EXPECT_EQ(packed_weights_serialization::deserialize(
                      legacy.data(), legacy.size()), nullptr)
            << "version " << early << " must reject globally";
    }
}

/**
 * @test Unknown bytes, tampered strides and the compact byte reject before
 *        any payload byte is adopted.
 *
 * The compact-byte control is the sequencing rule: version 8 carries the
 * layout, but compact section tables are not implemented yet, so a compact
 * header with canonical strides fails the re-derivation — it can never
 * smuggle itself in by matching the canonical stride numbers.
 */
TEST(PackedWeightsArchiveLayoutIdentity, UnknownByteStrideTamperAndCompactRejectBeforeAdoption)
{
    auto packed = canonicalQ4_0Owner();
    auto blob = blobOf(packed);
    ASSERT_FALSE(blob.empty());

    // Unknown layout byte: a typed rejection before payload adoption.
    auto unknown = blob;
    const uint8_t unknown_byte = 7;
    std::memcpy(unknown.data() + offsetof(PackedWeightsHeader, prepared_layout),
        &unknown_byte, sizeof(unknown_byte));
    EXPECT_EQ(packed_weights_serialization::deserialize(
                  unknown.data(), unknown.size()), nullptr);

    // Tampered stride: the receiver re-derives from the sealed layout, so
    // the header's own stride value is never trusted on its own.
    auto tampered = blob;
    const int32_t bad_stride =
        packed.data_stride + 1;
    std::memcpy(tampered.data() + offsetof(PackedWeightsHeader, data_stride),
        &bad_stride, sizeof(bad_stride));
    EXPECT_EQ(packed_weights_serialization::deserialize(
                  tampered.data(), tampered.size()), nullptr);

    // Compact byte with canonical strides: compact tables are unadmitted,
    // so their (invalid) section tables fail the stride re-derivation.
    auto compact = blob;
    const uint8_t compact_byte = static_cast<uint8_t>(
        cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCompactA64);
    std::memcpy(compact.data() + offsetof(PackedWeightsHeader, prepared_layout),
        &compact_byte, sizeof(compact_byte));
    EXPECT_EQ(packed_weights_serialization::deserialize(
                  compact.data(), compact.size()), nullptr);
}

/** @test The direct-transfer descriptor seals and re-derives the same way. */
TEST(PackedWeightsArchiveLayoutIdentity, DirectTransferDescriptorSealsAndReDerives)
{
    auto packed = canonicalQ4_0Owner();
    PackedWeightsTransferDescriptor descriptor;
    PackedWeightsConstSectionViews views;
    ASSERT_TRUE(packed_weights_serialization::describeTransfer(
        packed, descriptor, views));
    EXPECT_EQ(descriptor.header.version,
        PACKED_WEIGHTS_VERSION);
    EXPECT_EQ(descriptor.header.prepared_layout,
        static_cast<uint8_t>(
            cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical));

    // Unknown layout byte on the direct path: typed rejection before any
    // receive buffer is adopted.
    auto unknown = descriptor;
    unknown.header.prepared_layout = 7;
    EXPECT_THROW(static_cast<void>(
        packed_weights_serialization::allocateTransferTarget(unknown)),
        std::invalid_argument);

    // Compact byte with canonical strides on the direct path: the compact
    // tables are unadmitted, so the re-derivation fails closed.
    auto compact = descriptor;
    compact.header.prepared_layout = static_cast<uint8_t>(
        cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCompactA64);
    EXPECT_THROW(static_cast<void>(
        packed_weights_serialization::allocateTransferTarget(compact)),
        std::invalid_argument);

    // Tampered stride on the direct path.
    auto tampered = descriptor;
    tampered.header.data_stride += 1;
    EXPECT_THROW(static_cast<void>(
        packed_weights_serialization::allocateTransferTarget(tampered)),
        std::invalid_argument);

    // Positive: the authentic canonical descriptor allocates a real receive
    // target — non-null weights and every advertised section sized.
    const auto target =
        packed_weights_serialization::allocateTransferTarget(descriptor);
    ASSERT_NE(target.weights, nullptr);
    for (size_t section = 0; section < packed_weights_serialization::
             PACKED_WEIGHT_SECTION_COUNT; ++section)
    {
        const auto size = views.sizes[section];
        ASSERT_EQ(target.sizes[section], size);
        if (size != 0)
            ASSERT_NE(target.data[section], nullptr);
    }

    // Direct path versions: a same-epoch descriptor carries only the
    // current version. Archive-compat v7, older v6 and newer v9 all reject
    // before admission — none is a legitimate direct descriptor.
    static_assert(PACKED_WEIGHTS_VERSION == 8u &&
        PACKED_WEIGHTS_CANONICAL_VERSION == 7u,
        "the direct-version matrix follows the archive version pair");
    for (const uint32_t version : {7u, 6u, 9u})
    {
        auto versioned = descriptor;
        versioned.header.version = version;
        try
        {
            static_cast<void>(
                packed_weights_serialization::allocateTransferTarget(versioned));
            FAIL() << "non-current direct version must be refused";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_NE(std::string(error.what()).find("current archive version"),
                      std::string::npos)
                << "version " << version;
        }
    }
}


/** @test Receive admission rejects a table-valid foreign layout before allocation.
 *
 * Both archive receive surfaces are execution-bound: `deserialize` adopts
 * bytes into CPU-executable packed weights and `allocateTransferTarget`
 * allocates the final direct-receive storage. Once the compact AArch64 unit
 * has valid section tables (Stage B), a known-layout archive would pass
 * stride re-derivation and be adopted by a host that can never execute it.
 * The shared admission boundary takes the section resolver as a parameter:
 * production passes the section-table authority only, and this regression
 * injects the compact Q4_0 sections (design §2.1: payload 1024, unit 1152,
 * scales inline, compensation dropped) so the strides DO validate, proving
 * the executability gate — not a stride mismatch — rejects the foreign
 * layout before allocation. Admission is a pure function of its inputs;
 * there is no process-global resolver state to leak between tests.
 */
TEST(PackedWeightsArchiveLayoutIdentity, ExecutableReceiveRejectsTableValidForeignLayoutBeforeAllocation)
{
    using Layout = cpu::native_quantised::CPUNativeQuantisedLayout;
    // Faithful Stage-B-world resolution: canonical keeps the production
    // tables; compact returns the design unit for the fixture's Q4_0 source.
    const packed_weights_serialization::PreparedSectionsResolver compact_sections =
        [](Layout layout,
           cpu::native_quantised::CPUNativeQuantisedEncoding encoding,
           bool is_asymmetric,
           const NativeQuantisedFormatInfo &source)
        -> cpu::native_quantised::PreparedUnitSections {
        if (layout == Layout::InterleavedCanonical)
            return cpu::native_quantised::preparedUnitSections(
                layout, encoding, is_asymmetric, source);
        if (layout != Layout::InterleavedCompactA64)
            return {};
        cpu::native_quantised::PreparedUnitSections sections{};
        sections.payload_stride = 1024;
        sections.block_stride = 1152;
        sections.has_compensation = false;
        sections.has_secondary = false;
        sections.has_emins = false;
        sections.scales_offset = 1024;
        sections.mins_offset = 0;
        sections.emins_offset = 0;
        sections.compensation_offset = 0;
        return sections;
    };

    const auto packed = canonicalQ4_0Owner();
    const auto blob = blobOf(packed);
    ASSERT_FALSE(blob.empty());

    // Compact header whose strides the injected tables reproduce exactly:
    // the shared admission must reject it by the executability gate.
    PackedWeightsHeader compact_header;
    std::memcpy(&compact_header, blob.data(), sizeof(compact_header));
    compact_header.prepared_layout =
        static_cast<uint8_t>(Layout::InterleavedCompactA64);
    compact_header.data_stride = 1024;
    compact_header.interleaved_block_stride = 1152;

    // Canonical control through the SAME injected resolver: admission is
    // unchanged for the layout this build executes.
    PackedWeightsHeader canonical_header;
    std::memcpy(&canonical_header, blob.data(), sizeof(canonical_header));
    EXPECT_EQ(packed_weights_serialization::detail::admitReceiveLayout(
                  canonical_header, compact_sections),
              Layout::InterleavedCanonical);

    // The table-valid compact header is rejected with the typed
    // executability diagnostic before any allocation decision exists.
    try
    {
        static_cast<void>(packed_weights_serialization::detail::admitReceiveLayout(
            compact_header, compact_sections));
        FAIL() << "table-valid compact admission must be refused";
    }
    catch (const std::invalid_argument &error)
    {
        EXPECT_NE(std::string(error.what()).find("CPU execution"),
                  std::string::npos);
        EXPECT_NE(std::string(error.what()).find("interleaved-compact-a64"),
                  std::string::npos);
        EXPECT_NE(std::string(error.what()).find("interleaved-canonical"),
                  std::string::npos);
    }

    // The public entrypoints pass the section-table authority only: the same
    // compact header fails closed at stride re-derivation there (unimplemented
    // compact tables), and the canonical archive still round-trips.
    auto compact = blob;
    std::memcpy(compact.data() + offsetof(PackedWeightsHeader, prepared_layout),
        &compact_header.prepared_layout, sizeof(compact_header.prepared_layout));
    const int32_t compact_data_stride = 1024;
    const int32_t compact_block_stride = 1152;
    std::memcpy(compact.data() + offsetof(PackedWeightsHeader, data_stride),
        &compact_data_stride, sizeof(compact_data_stride));
    std::memcpy(compact.data() + offsetof(PackedWeightsHeader, interleaved_block_stride),
        &compact_block_stride, sizeof(compact_block_stride));
    EXPECT_EQ(packed_weights_serialization::deserialize(
                  compact.data(), compact.size()),
              nullptr);
    EXPECT_NE(packed_weights_serialization::deserialize(
                  blob.data(), blob.size()),
              nullptr);
}

} // namespace llaminar2::test
