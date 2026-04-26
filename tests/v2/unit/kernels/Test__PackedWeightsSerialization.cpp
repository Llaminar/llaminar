/**
 * @file Test__PackedWeightsSerialization.cpp
 * @brief Unit tests for packed weights serialization/deserialization.
 *
 * Tests round-trip integrity for CPUPackedWeights and CPUPackedWeightsWithNativeBlocks,
 * plus validation of error paths (bad magic, truncated buffers, etc.).
 */

#include <gtest/gtest.h>

#include "kernels/PackedWeightsSerialization.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h"
#include "kernels/cpu/native_vnni/CPUPackedWeights.h"

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
    packed.is_nibble_lut            = is_nibble_lut;
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
    EXPECT_TRUE(p.is_nibble_lut);
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
    EXPECT_TRUE(p.is_nibble_lut);
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
    EXPECT_FALSE(p.is_nibble_lut);

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
