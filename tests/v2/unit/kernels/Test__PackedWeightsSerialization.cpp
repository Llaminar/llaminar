/**
 * @file Test__PackedWeightsSerialization.cpp
 * @brief Unit tests for packed weights serialization/deserialization.
 *
 * Tests round-trip integrity for CPUPackedWeights and CPUPackedWeightsWithNativeBlocks,
 * plus validation of error paths (bad magic, truncated buffers, etc.).
 * IQ2_S source-factor archives preserve raw words and reject derived-half V4.
 */

#include "tensors/NativeVnniIQ2SScale.h"
#include "tensors/NativeVnniIQ4XSScale.h"
#include "loaders/gpu_pipeline/GpuPackedWeightsFormat.h"
#include <gtest/gtest.h>

#include "kernels/PackedWeightsSerialization.h"
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "kernels/cpu/gemm/CPUPackedWeights.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::packed_weights_serialization;

namespace {

/// Fill a buffer with a deterministic pattern.
void fillPattern(uint8_t* data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
}

void fillPatternSigned(int8_t* data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<int8_t>(((i * 7 + 13) & 0xFF) - 128);
}

/// Build a CPUNativeVNNIPackedWeights with given metadata and deterministic data.
CPUNativeVNNIPackedWeights buildTestPacked(
    int N, int K, uint8_t codebook_id, int blocks_per_row,
    int payload_bytes_val, bool is_nibble_lut, bool is_asymmetric,
    bool is_superblock, int data_stride, int interleaved_block_stride,
    size_t interleaved_sz, size_t payload_sz, size_t int8_flat_sz)
{
    CPUNativeVNNIPackedWeights packed;
    packed.N                        = N;
    packed.K                        = K;
    packed.N_padded                 = ((N + 63) / 64) * 64;
    packed.blocks_per_row           = blocks_per_row;
    packed.codebook_id              = codebook_id;
    packed.payload_bytes            = payload_bytes_val;
    packed.encoding = is_nibble_lut
                          ? CPUNativeVNNIEncoding::NibbleLUT
                          : CPUNativeVNNIEncoding::ExpandedInt8;
    packed.is_asymmetric            = is_asymmetric;
    packed.is_superblock            = is_superblock;
    packed.data_stride              = data_stride;
    packed.interleaved_block_stride = interleaved_block_stride;

    if (interleaved_sz > 0)
    {
        packed.native_interleaved.resize_uninitialized(interleaved_sz);
        fillPattern(packed.native_interleaved.data(), interleaved_sz);
    }
    if (payload_sz > 0)
    {
        packed.payload.resize(payload_sz);
        fillPattern(packed.payload.data(), payload_sz);
    }
    if (int8_flat_sz > 0)
    {
        packed.int8_flat.resize(int8_flat_sz);
        fillPatternSigned(packed.int8_flat.data(), int8_flat_sz);
    }

    return packed;
}

} // anonymous namespace

// ─── Round-trip tests ──────────────────────────────────────────────

TEST(Test__PackedWeightsSerialization, SerializeDeserialize_BasicQ4_0)
{
    auto packed = buildTestPacked(
        /*N=*/128, /*K=*/256, /*codebook_id=*/0, /*blocks_per_row=*/8,
        /*payload_bytes=*/16, /*is_nibble_lut=*/true, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/1024, /*interleaved_block_stride=*/1280,
        /*interleaved_sz=*/2 * 8 * 1280, /*payload_sz=*/2 * 8 * 64 * 16, /*int8_flat_sz=*/0);

    CPUPackedWeights weights(std::move(packed));

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    auto result = deserialize(blob.data(), blob.size());
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->format(), PackedWeightsFormat::CPU_NATIVE_VNNI);

    const auto* cpu_result = dynamic_cast<const CPUPackedWeights*>(result.get());
    ASSERT_NE(cpu_result, nullptr);

    const auto& p = cpu_result->packed();
    EXPECT_EQ(p.N, 128);
    EXPECT_EQ(p.K, 256);
    EXPECT_EQ(p.N_padded, 128);
    EXPECT_EQ(p.blocks_per_row, 8);
    EXPECT_EQ(p.codebook_id, 0);
    EXPECT_EQ(p.payload_bytes, 16);
    EXPECT_TRUE(p.usesNibbleLUT());
    EXPECT_FALSE(p.is_asymmetric);
    EXPECT_FALSE(p.is_superblock);
    EXPECT_EQ(p.data_stride, 1024);
    EXPECT_EQ(p.interleaved_block_stride, 1280);
    EXPECT_EQ(p.workspace_data_, nullptr);

    // Verify data is bit-identical.
    ASSERT_EQ(p.native_interleaved.size(), weights.packed().native_interleaved.size());
    EXPECT_EQ(std::memcmp(p.native_interleaved.data(),
                          weights.packed().native_interleaved.data(),
                          p.native_interleaved.size()), 0);

    // weights was moved-from, so check the original payload via the round-tripped copy
    ASSERT_EQ(p.payload.size(), 2u * 8 * 64 * 16);
    // Verify payload pattern
    for (size_t i = 0; i < p.payload.size(); ++i)
        EXPECT_EQ(p.payload[i], static_cast<uint8_t>((i * 7 + 13) & 0xFF)) << "payload mismatch at " << i;

    EXPECT_TRUE(p.int8_flat.empty());

    // Should NOT be a CPUPackedWeightsWithNativeBlocks
    EXPECT_EQ(dynamic_cast<const CPUPackedWeightsWithNativeBlocks*>(result.get()), nullptr);
}

TEST(Test__PackedWeightsSerialization, CompactMultiScalePreservesEveryNativeMetadataBit)
{
    for (const auto &entry : native_vnni_formats::kAllSourceFormats)
    {
        const auto &format = *entry.metadata;
        if (!hasCompactMultiScaleVnniPayload(format.codebook_id))
            continue;
        SCOPED_TRACE(std::string(entry.quant_type));
        auto packed = buildTestPacked(
            67, 544, format.codebook_id, 17, format.payload_bytes,
            false, true, true, 1024, 1536, 2 * 17 * 1536, 0, 0);
        packed.encoding = CPUNativeVNNIEncoding::CompactMultiScale;
        CPUPackedWeights owner(std::move(packed));
        const auto blob = serialize(owner);
        ASSERT_FALSE(blob.empty());
        auto result = deserialize(blob.data(), blob.size());
        ASSERT_NE(result, nullptr);
        const auto *native = dynamic_cast<const CPUPackedWeights *>(result.get());
        ASSERT_NE(native, nullptr);
        const auto &received = native->packed();
        ASSERT_TRUE(received.usesCompactMultiScale());
        EXPECT_FALSE(received.usesInlineCompensation());
        EXPECT_EQ(received.payload_bytes, format.payload_bytes);
        ASSERT_EQ(received.native_interleaved.size(), owner.packed().native_interleaved.size());
        EXPECT_EQ(std::memcmp(received.native_interleaved.data(),
                              owner.packed().native_interleaved.data(),
                              received.native_interleaved.size()), 0);
        EXPECT_TRUE(received.int8_flat.empty());

        // Older producers encoded these same source identities using a lossy
        // single-scale representation. The archive version is an arithmetic
        // contract too, not merely a check that the struct fields fit.
        auto legacy_blob = blob;
        const uint32_t legacy_version = 2;
        std::memcpy(legacy_blob.data() + offsetof(PackedWeightsHeader, version),
                    &legacy_version, sizeof(legacy_version));
        EXPECT_EQ(deserialize(legacy_blob.data(), legacy_blob.size()), nullptr);
    }
}

TEST(Test__PackedWeightsSerialization, SerializeDeserialize_AsymmetricQ4_1)
{
    auto packed = buildTestPacked(
        /*N=*/64, /*K=*/128, /*codebook_id=*/5, /*blocks_per_row=*/4,
        /*payload_bytes=*/20, /*is_nibble_lut=*/true, /*is_asymmetric=*/true,
        /*is_superblock=*/false, /*data_stride=*/1024, /*interleaved_block_stride=*/1408,
        /*interleaved_sz=*/1 * 4 * 1408, /*payload_sz=*/1 * 4 * 64 * 20, /*int8_flat_sz=*/0);

    CPUPackedWeights weights(std::move(packed));

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    auto result = deserialize(blob.data(), blob.size());
    ASSERT_NE(result, nullptr);

    const auto* cpu_result = dynamic_cast<const CPUPackedWeights*>(result.get());
    ASSERT_NE(cpu_result, nullptr);

    const auto& p = cpu_result->packed();
    EXPECT_EQ(p.N, 64);
    EXPECT_EQ(p.K, 128);
    EXPECT_EQ(p.codebook_id, 5);
    EXPECT_EQ(p.payload_bytes, 20);
    EXPECT_TRUE(p.is_asymmetric);
    EXPECT_TRUE(p.usesNibbleLUT());
    EXPECT_FALSE(p.is_superblock);
    EXPECT_EQ(p.interleaved_block_stride, 1408);
}

TEST(Test__PackedWeightsSerialization, SerializeDeserialize_WithNativeBlocks)
{
    auto packed = buildTestPacked(
        /*N=*/128, /*K=*/256, /*codebook_id=*/0, /*blocks_per_row=*/8,
        /*payload_bytes=*/16, /*is_nibble_lut=*/true, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/1024, /*interleaved_block_stride=*/1280,
        /*interleaved_sz=*/2 * 8 * 1280, /*payload_sz=*/2 * 8 * 64 * 16, /*int8_flat_sz=*/0);

    // Create native blocks data.
    const size_t nb_size = 128 * 8 * 18; // N * blocks_per_row * block_bytes(Q4_0)
    std::vector<uint8_t> native_blocks(nb_size);
    fillPattern(native_blocks.data(), nb_size);

    CPUPackedWeightsWithNativeBlocks weights(
        std::move(packed), std::move(native_blocks), /*native_block_size=*/18);

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    auto result = deserialize(blob.data(), blob.size());
    ASSERT_NE(result, nullptr);

    // Must be the WithNativeBlocks subclass.
    const auto* with_nb = dynamic_cast<const CPUPackedWeightsWithNativeBlocks*>(result.get());
    ASSERT_NE(with_nb, nullptr);

    EXPECT_EQ(with_nb->nativeBlockSize(), 18u);
    ASSERT_EQ(with_nb->nativeBlocks().size(), nb_size);

    // Verify native blocks are bit-identical.
    for (size_t i = 0; i < nb_size; ++i)
        EXPECT_EQ(with_nb->nativeBlocks()[i],
                  static_cast<uint8_t>((i * 7 + 13) & 0xFF)) << "native_blocks mismatch at " << i;

    // Verify packed metadata survived.
    const auto& p = with_nb->packed();
    EXPECT_EQ(p.N, 128);
    EXPECT_EQ(p.K, 256);
}

TEST(Test__PackedWeightsSerialization, SerializeDeserialize_EmptySections)
{
    // Common case: payload and int8_flat released after packing.
    auto packed = buildTestPacked(
        /*N=*/64, /*K=*/64, /*codebook_id=*/0, /*blocks_per_row=*/2,
        /*payload_bytes=*/16, /*is_nibble_lut=*/true, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/1024, /*interleaved_block_stride=*/1280,
        /*interleaved_sz=*/1 * 2 * 1280, /*payload_sz=*/0, /*int8_flat_sz=*/0);

    CPUPackedWeights weights(std::move(packed));

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    auto result = deserialize(blob.data(), blob.size());
    ASSERT_NE(result, nullptr);

    const auto* cpu_result = dynamic_cast<const CPUPackedWeights*>(result.get());
    ASSERT_NE(cpu_result, nullptr);

    const auto& p = cpu_result->packed();
    EXPECT_TRUE(p.payload.empty());
    EXPECT_TRUE(p.int8_flat.empty());
    EXPECT_EQ(p.native_interleaved.size(), static_cast<size_t>(1 * 2 * 1280));
}

// ─── Error path tests ──────────────────────────────────────────────

TEST(Test__PackedWeightsSerialization, Deserialize_InvalidMagic)
{
    // Build a valid blob then corrupt the magic.
    auto packed = buildTestPacked(
        32, 64, 0, 2, 16, true, false, false, 1024, 1280,
        1 * 2 * 1280, 0, 0);
    CPUPackedWeights weights(std::move(packed));
    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    // Corrupt magic bytes.
    blob[0] = 0xFF;
    blob[1] = 0xFF;

    auto result = deserialize(blob.data(), blob.size());
    EXPECT_EQ(result, nullptr);
}

TEST(Test__PackedWeightsSerialization, Deserialize_InvalidVersion)
{
    auto packed = buildTestPacked(
        32, 64, 0, 2, 16, true, false, false, 1024, 1280,
        1 * 2 * 1280, 0, 0);
    CPUPackedWeights weights(std::move(packed));
    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    // Set version to 99.
    uint32_t bad_version = 99;
    std::memcpy(blob.data() + 4, &bad_version, sizeof(bad_version));

    auto result = deserialize(blob.data(), blob.size());
    EXPECT_EQ(result, nullptr);
}

TEST(Test__PackedWeightsSerialization, Deserialize_TruncatedBuffer)
{
    auto packed = buildTestPacked(
        128, 256, 0, 8, 16, true, false, false, 1024, 1280,
        2 * 8 * 1280, 2 * 8 * 64 * 16, 0);
    CPUPackedWeights weights(std::move(packed));
    auto blob = serialize(weights);
    ASSERT_GT(blob.size(), 96u + 100u);

    // Truncate to just past the section table — missing data sections.
    auto result = deserialize(blob.data(), 97);
    EXPECT_EQ(result, nullptr);
}

TEST(Test__PackedWeightsSerialization, Deserialize_ZeroSize)
{
    auto result = deserialize(nullptr, 0);
    EXPECT_EQ(result, nullptr);

    uint8_t dummy = 0;
    result = deserialize(&dummy, 0);
    EXPECT_EQ(result, nullptr);
}

TEST(Test__PackedWeightsSerialization, SerializeDeserialize_LargeWeights)
{
    // Realistic MoE expert dimensions.
    const int N = 4096;
    const int K = 2048;
    const int bpr = (K + 31) / 32; // 64
    const int N_chunks = (N + 63) / 64; // 64

    auto packed = buildTestPacked(
        N, K, /*codebook_id=*/4, bpr,
        /*payload_bytes=*/16, /*is_nibble_lut=*/true, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/1024, /*interleaved_block_stride=*/1280,
        /*interleaved_sz=*/static_cast<size_t>(N_chunks) * bpr * 1280,
        /*payload_sz=*/static_cast<size_t>(N_chunks) * bpr * 64 * 16,
        /*int8_flat_sz=*/0);

    CPUPackedWeights weights(std::move(packed));

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    auto result = deserialize(blob.data(), blob.size());
    ASSERT_NE(result, nullptr);

    const auto* cpu_result = dynamic_cast<const CPUPackedWeights*>(result.get());
    ASSERT_NE(cpu_result, nullptr);

    const auto& p = cpu_result->packed();
    EXPECT_EQ(p.N, N);
    EXPECT_EQ(p.K, K);
    EXPECT_EQ(p.codebook_id, 4);

    // Verify interleaved data integrity.
    const size_t expected_il_size = static_cast<size_t>(N_chunks) * bpr * 1280;
    ASSERT_EQ(p.native_interleaved.size(), expected_il_size);
    for (size_t i = 0; i < expected_il_size; ++i)
        EXPECT_EQ(p.native_interleaved[i],
                  static_cast<uint8_t>((i * 7 + 13) & 0xFF)) << "interleaved mismatch at " << i;
}

TEST(Test__PackedWeightsSerialization, SerializeSize_MatchesExpected)
{
    const size_t il_size = 5120;
    const size_t pay_size = 16384;
    const size_t i8_size = 0;

    auto packed = buildTestPacked(
        128, 256, 0, 8, 16, true, false, false, 1024, 1280,
        il_size, pay_size, i8_size);
    CPUPackedWeights weights(std::move(packed));

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    const size_t expected = sizeof(PackedWeightsHeader) + sizeof(PackedWeightsSectionTable)
                          + il_size + pay_size + i8_size;
    EXPECT_EQ(blob.size(), expected);
}

// ─── IPackedWeights::serialize() default returns empty ─────────────

TEST(Test__PackedWeightsSerialization, IPackedWeights_DefaultSerializeReturnsEmpty)
{
    // Verify that the default virtual serialize() returns empty.
    // We can test this by calling it on a CPUPackedWeights (which doesn't override).
    auto packed = buildTestPacked(
        32, 64, 0, 2, 16, true, false, false, 1024, 1280,
        2560, 0, 0);
    CPUPackedWeights weights(std::move(packed));

    // Call the virtual method (default impl returns empty).
    auto default_result = weights.serialize();
    EXPECT_TRUE(default_result.empty());
}

// ─── Additional edge cases ─────────────────────────────────────────

TEST(Test__PackedWeightsSerialization, SerializeDeserialize_WithInt8Flat)
{
    // Test with all three data sections populated.
    auto packed = buildTestPacked(
        64, 128, /*codebook_id=*/19, /*blocks_per_row=*/4,
        /*payload_bytes=*/34, /*is_nibble_lut=*/false, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/2048, /*interleaved_block_stride=*/2304,
        /*interleaved_sz=*/1 * 4 * 2304, /*payload_sz=*/1 * 4 * 64 * 34,
        /*int8_flat_sz=*/1 * 4 * 64 * 32);

    CPUPackedWeights weights(std::move(packed));

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    auto result = deserialize(blob.data(), blob.size());
    ASSERT_NE(result, nullptr);

    const auto* cpu_result = dynamic_cast<const CPUPackedWeights*>(result.get());
    ASSERT_NE(cpu_result, nullptr);

    const auto& p = cpu_result->packed();
    EXPECT_EQ(p.codebook_id, 19);
    EXPECT_EQ(p.data_stride, 2048);
    EXPECT_EQ(p.interleaved_block_stride, 2304);
    EXPECT_TRUE(p.usesExpandedInt8());

    ASSERT_EQ(p.int8_flat.size(), static_cast<size_t>(1 * 4 * 64 * 32));
    for (size_t i = 0; i < p.int8_flat.size(); ++i)
        EXPECT_EQ(p.int8_flat[i],
                  static_cast<int8_t>(((i * 7 + 13) & 0xFF) - 128)) << "int8_flat mismatch at " << i;
}

TEST(Test__PackedWeightsSerialization, SerializeDeserialize_SuperblockFormat)
{
    // Q6_K superblock format.
    auto packed = buildTestPacked(
        256, 512, /*codebook_id=*/10, /*blocks_per_row=*/2,
        /*payload_bytes=*/24, /*is_nibble_lut=*/false, /*is_asymmetric=*/false,
        /*is_superblock=*/true, /*data_stride=*/2048, /*interleaved_block_stride=*/2304,
        /*interleaved_sz=*/4 * 2 * 2304, /*payload_sz=*/4 * 2 * 64 * 24,
        /*int8_flat_sz=*/4 * 2 * 64 * 32);

    CPUPackedWeights weights(std::move(packed));

    auto blob = serialize(weights);
    ASSERT_FALSE(blob.empty());

    auto result = deserialize(blob.data(), blob.size());
    ASSERT_NE(result, nullptr);

    const auto* cpu_result = dynamic_cast<const CPUPackedWeights*>(result.get());
    ASSERT_NE(cpu_result, nullptr);

    const auto& p = cpu_result->packed();
    EXPECT_EQ(p.codebook_id, 10);
    EXPECT_TRUE(p.is_superblock);
    EXPECT_EQ(p.N, 256);
    EXPECT_EQ(p.K, 512);
}

/** @brief Same-sized old Q6 metadata must not be adopted as source coefficients. */
TEST(Test__PackedWeightsSerialization, Q6SourceFactorsRoundTripAndRejectVersion3)
{
    auto packed = buildTestPacked(67, 288, 8, 9, 24, false, true, true,
        1536, 1792, 2 * 9 * 1792, 0, 0);
    packed.encoding = CPUNativeVNNIEncoding::Q6KNativeDualScale;
    CPUPackedWeights owner(std::move(packed));
    auto blob = serialize(owner);
    auto restored = deserialize(blob.data(), blob.size());
    ASSERT_NE(restored, nullptr);
    const auto *native = dynamic_cast<const CPUPackedWeights *>(restored.get());
    ASSERT_NE(native, nullptr);
    ASSERT_TRUE(native->packed().usesQ6KNativeDualScale());
    ASSERT_EQ(native->packed().native_interleaved.size(), owner.packed().native_interleaved.size());
    EXPECT_EQ(std::memcmp(native->packed().native_interleaved.data(),
        owner.packed().native_interleaved.data(), owner.packed().native_interleaved.size()), 0);
    const uint32_t legacy_version = 3;
    std::memcpy(blob.data() + offsetof(PackedWeightsHeader, version), &legacy_version, sizeof(legacy_version));
    EXPECT_EQ(deserialize(blob.data(), blob.size()), nullptr);
}

/** @brief The canonical IQ2_S raw secondary word preserves all256 source bytes. */
TEST(Test__PackedWeightsSerialization, IQ2SRawCoefficientWordsPreserveAllBytes)
{
    for(unsigned value=0;value<256;++value)
    {
        const auto word=llaminar2::native_vnni_iq2s_scale::pack(static_cast<uint8_t>(value));
        EXPECT_EQ(word,value);
        EXPECT_TRUE(llaminar2::native_vnni_iq2s_scale::valid(word));
        EXPECT_EQ(llaminar2::native_vnni_iq2s_scale::low(word),int(value&15u));
        EXPECT_EQ(llaminar2::native_vnni_iq2s_scale::high(word),int(value>>4));
        EXPECT_FALSE(llaminar2::native_vnni_iq2s_scale::valid(word|0x0100u));
    }
}

/** @brief Source-word archives roundtrip exactly and reject derived-half V4 first. */
TEST(Test__PackedWeightsSerialization, IQ2SSourceFactorsRoundTripAndRejectVersion4)
{
    auto packed=buildTestPacked(67,288,13,9,9,false,true,true,
        1024,1536,2*9*1536,0,0);
    packed.encoding=CPUNativeVNNIEncoding::CompactMultiScale;
    for(int c=0;c<2;++c) for(int kb=0;kb<9;++kb) for(int n=0;n<64;++n)
    {
        const unsigned ordinal=unsigned(c*9*64+kb*64+n);
        const_cast<uint16_t*>(packed.chunkScales(c,kb))[n]=
            static_cast<uint16_t>((ordinal&1u)?0x8001u:0x3c01u);
        const_cast<uint16_t*>(packed.chunkMins(c,kb))[n]=static_cast<uint8_t>(ordinal);
    }
    CPUPackedWeights owner(std::move(packed));
    auto blob=serialize(owner);
    ASSERT_FALSE(blob.empty());
    auto restored=deserialize(blob.data(),blob.size());
    ASSERT_NE(restored,nullptr);
    const auto *native=dynamic_cast<const CPUPackedWeights*>(restored.get());
    ASSERT_NE(native,nullptr);
    ASSERT_TRUE(native->packed().usesCompactMultiScale());
    ASSERT_EQ(native->packed().native_interleaved.size(),owner.packed().native_interleaved.size());
    EXPECT_EQ(std::memcmp(native->packed().native_interleaved.data(),
        owner.packed().native_interleaved.data(),owner.packed().native_interleaved.size()),0);
    const uint32_t legacy_version=4;
    std::memcpy(blob.data()+offsetof(PackedWeightsHeader,version),&legacy_version,sizeof(legacy_version));
    EXPECT_EQ(deserialize(blob.data(),blob.size()),nullptr);
}

/** @brief GPU wire records preserve source words and reject derived-half V1. */
TEST(Test__PackedWeightsSerialization, IQ2SGpuWireSourceFactorsRoundTripAndRejectVersion1)
{
    using namespace llaminar2;
    const auto header=buildGpuPackedHeader(RepackFormat::IQ2_S,13,9,true,true,false,67,288);
    std::vector<uint8_t> blob(gpuPackedTotalSize(header),0u);
    std::memcpy(blob.data(),&header,sizeof(header));
    const size_t words=static_cast<size_t>(header.N)*header.blocks_per_row;
    const size_t primary=sizeof(header)+header.payload_size;
    const size_t secondary=primary+header.scales_size;
    for(size_t index=0;index<words;++index)
    {
        const uint16_t d=(index&1u)?0x8001u:0x3c01u;
        const uint16_t coefficient=static_cast<uint8_t>(index);
        std::memcpy(blob.data()+primary+index*sizeof(uint16_t),&d,sizeof(d));
        std::memcpy(blob.data()+secondary+index*sizeof(uint16_t),&coefficient,sizeof(coefficient));
    }
    GpuPackedWeightsView view{};
    ASSERT_TRUE(parseGpuPackedBuffer(blob.data(),blob.size(),view));
    ASSERT_NE(view.scales,nullptr); ASSERT_NE(view.mins,nullptr);
    for(size_t index=0;index<words;++index)
    {
        // The compact wire layout has no alignment padding for an odd N.
        // Inspect its raw words with memcpy rather than an unaligned host load.
        uint16_t d=0,coefficient=0;
        std::memcpy(&d,reinterpret_cast<const uint8_t*>(view.scales)+index*sizeof(uint16_t),sizeof(d));
        std::memcpy(&coefficient,reinterpret_cast<const uint8_t*>(view.mins)+index*sizeof(uint16_t),sizeof(coefficient));
        EXPECT_EQ(d,(index&1u)?0x8001u:0x3c01u);
        EXPECT_EQ(coefficient,static_cast<uint8_t>(index));
    }
    const uint32_t old_version=1;
    std::memcpy(blob.data()+offsetof(GpuPackedWeightsHeader,version),&old_version,sizeof(old_version));
    GpuPackedWeightsView rejected{};
    EXPECT_FALSE(parseGpuPackedBuffer(blob.data(),blob.size(),rejected));
    EXPECT_EQ(rejected.payload,nullptr);
    EXPECT_EQ(rejected.scales,nullptr);
    EXPECT_EQ(rejected.mins,nullptr);
}

/** @brief Raw IQ4_XS tuples survive CPU serialization and reject derived-half V5. */
TEST(Test__PackedWeightsSerialization, IQ4XSSourceTupleRoundTripsAndRejectsVersion5)
{
    auto packed = buildTestPacked(67, 288, 4, 9, 16, false, false, true,
        1024, 1408, 2 * 9 * 1408, 0, 0);
    packed.encoding = CPUNativeVNNIEncoding::NibbleLUTSourceScale;
    for (int c = 0; c < 2; ++c) for (int kb = 0; kb < 9; ++kb) for (int n = 0; n < 64; ++n)
    {
        const unsigned ordinal = unsigned(c * 9 * 64 + kb * 64 + n);
        const_cast<uint16_t *>(packed.chunkScales(c, kb))[n] = ordinal & 1u ? 0x8001u : 0x3c01u;
        const_cast<uint16_t *>(packed.chunkMins(c, kb))[n] = ordinal % 64;
    }
    CPUPackedWeights owner(std::move(packed));
    auto blob = serialize(owner);
    auto restored = deserialize(blob.data(), blob.size());
    ASSERT_NE(restored, nullptr);
    const auto *native = dynamic_cast<const CPUPackedWeights *>(restored.get());
    ASSERT_NE(native, nullptr); ASSERT_TRUE(native->packed().usesNibbleLUTSourceScale());
    ASSERT_EQ(native->packed().native_interleaved.size(), owner.packed().native_interleaved.size());
    EXPECT_EQ(std::memcmp(native->packed().native_interleaved.data(),
        owner.packed().native_interleaved.data(), owner.packed().native_interleaved.size()), 0);
    const uint32_t old_version = 5;
    std::memcpy(blob.data() + offsetof(PackedWeightsHeader, version), &old_version, sizeof(old_version));
    EXPECT_EQ(deserialize(blob.data(), blob.size()), nullptr);
    PackedWeightsTransferDescriptor descriptor{};
    std::memcpy(&descriptor.header, blob.data(), sizeof(descriptor.header));
    std::memcpy(&descriptor.sections, blob.data() + sizeof(descriptor.header), sizeof(descriptor.sections));
    EXPECT_THROW((void)allocateTransferTarget(descriptor), std::invalid_argument);
}

/** @brief GPU tuple wire has20 bytes/block, authentic source4 and a distinct physical ID. */
TEST(Test__PackedWeightsSerialization, IQ4XSGpuWireTupleRejectsVersion2AndSourceSubstitution)
{
    using namespace llaminar2;
    const auto header = buildGpuPackedHeader(RepackFormat::IQ4_XS, 24, 16, true, true, false, 17, 288);
    EXPECT_EQ(header.payload_size + header.scales_size + header.mins_size, 17u * 9u * 20u);
    std::vector<uint8_t> blob(gpuPackedTotalSize(header), 0);
    std::memcpy(blob.data(), &header, sizeof(header));
    GpuPackedWeightsView view{};
    ASSERT_TRUE(parseGpuPackedBuffer(blob.data(), blob.size(), view));
    ASSERT_NE(view.mins, nullptr);
    auto wrong = header; wrong.version = 2;
    EXPECT_FALSE(validateGpuPackedHeader(wrong, blob.size()));
    wrong = header; wrong.codebook_id = 4;
    EXPECT_FALSE(validateGpuPackedHeader(wrong, blob.size()));
    wrong = header; wrong.format = static_cast<uint8_t>(RepackFormat::IQ4_NL);
    EXPECT_FALSE(validateGpuPackedHeader(wrong, blob.size()));
    wrong = header; wrong.is_asymmetric = 0;
    EXPECT_FALSE(validateGpuPackedHeader(wrong, blob.size()));
    wrong = header; wrong.mins_size = 0;
    EXPECT_FALSE(validateGpuPackedHeader(wrong, blob.size()));
    const auto nl = buildGpuPackedHeader(RepackFormat::IQ4_NL, 4, 16, false, false, false, 17, 288);
    EXPECT_TRUE(validateGpuPackedHeader(nl, gpuPackedTotalSize(nl)));
}
