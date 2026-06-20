/**
 * @file Test__ExpertWeightTransfer.cpp
 * @brief Unit tests for ExpertWeightTransfer manifest building and tag helpers.
 *
 * Tests pure-logic helpers (no MPI required) plus serialization round-trip
 * through ExpertWeightBlobs to simulate the full transfer path.
 */

#include <gtest/gtest.h>

#include "execution/moe/ExpertWeightTransfer.h"
#include "kernels/PackedWeightsSerialization.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h"
#include "kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "utils/MPITags.h"

#include <cstring>
#include <set>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::packed_weights_serialization;
using namespace llaminar2::mpi_tags;

namespace {

/// Fill a buffer with a deterministic pattern (matches Test__PackedWeightsSerialization).
void fillPattern(uint8_t* data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
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
        for (size_t i = 0; i < int8_flat_sz; ++i)
            packed.int8_flat[i] = static_cast<int8_t>(((i * 7 + 13) & 0xFF) - 128);
    }

    return packed;
}

} // anonymous namespace

// ─── Manifest Building ────────────────────────────────────────────

TEST(Test__ExpertWeightTransfer, BuildManifest_NoChanges)
{
    std::vector<int> placement = {0, 0, 1, 1};
    auto manifest = ExpertWeightTransfer::buildManifest(placement, placement);
    EXPECT_TRUE(manifest.empty());
}

TEST(Test__ExpertWeightTransfer, BuildManifest_AllChange)
{
    std::vector<int> old_p = {0, 0, 0, 0};
    std::vector<int> new_p = {1, 1, 1, 1};
    auto manifest = ExpertWeightTransfer::buildManifest(old_p, new_p);
    ASSERT_EQ(manifest.size(), 4u);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(manifest[i].expert_id, i);
        EXPECT_EQ(manifest[i].src_rank, 0);
        EXPECT_EQ(manifest[i].dst_rank, 1);
    }
}

TEST(Test__ExpertWeightTransfer, BuildManifest_PartialChange)
{
    // 8 experts, 2 sockets: experts 2,5 swap ranks
    std::vector<int> old_p = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int> new_p = {0, 0, 1, 0, 1, 0, 1, 1};
    auto manifest = ExpertWeightTransfer::buildManifest(old_p, new_p);
    ASSERT_EQ(manifest.size(), 2u);

    EXPECT_EQ(manifest[0].expert_id, 2);
    EXPECT_EQ(manifest[0].src_rank, 0);
    EXPECT_EQ(manifest[0].dst_rank, 1);

    EXPECT_EQ(manifest[1].expert_id, 5);
    EXPECT_EQ(manifest[1].src_rank, 1);
    EXPECT_EQ(manifest[1].dst_rank, 0);
}

TEST(Test__ExpertWeightTransfer, BuildManifest_DifferentSizes)
{
    // If vectors differ in size, only processes up to the shorter one
    std::vector<int> old_p = {0, 1, 0};
    std::vector<int> new_p = {0, 0};
    auto manifest = ExpertWeightTransfer::buildManifest(old_p, new_p);
    ASSERT_EQ(manifest.size(), 1u);
    EXPECT_EQ(manifest[0].expert_id, 1);
    EXPECT_EQ(manifest[0].src_rank, 1);
    EXPECT_EQ(manifest[0].dst_rank, 0);
}

// ─── Departing/Arriving Experts ───────────────────────────────────

TEST(Test__ExpertWeightTransfer, DepartingExperts_CorrectForRank)
{
    std::vector<ExpertMigration> manifest = {
        {2, 0, 1},  // expert 2: rank 0 → rank 1
        {5, 1, 0},  // expert 5: rank 1 → rank 0
        {7, 0, 1},  // expert 7: rank 0 → rank 1
    };

    auto departing_r0 = ExpertWeightTransfer::departingExperts(manifest, 0);
    ASSERT_EQ(departing_r0.size(), 2u);
    EXPECT_EQ(departing_r0[0], 2);
    EXPECT_EQ(departing_r0[1], 7);

    auto departing_r1 = ExpertWeightTransfer::departingExperts(manifest, 1);
    ASSERT_EQ(departing_r1.size(), 1u);
    EXPECT_EQ(departing_r1[0], 5);
}

TEST(Test__ExpertWeightTransfer, ArrivingExperts_CorrectForRank)
{
    std::vector<ExpertMigration> manifest = {
        {2, 0, 1},
        {5, 1, 0},
        {7, 0, 1},
    };

    auto arriving_r0 = ExpertWeightTransfer::arrivingExperts(manifest, 0);
    ASSERT_EQ(arriving_r0.size(), 1u);
    EXPECT_EQ(arriving_r0[0], 5);

    auto arriving_r1 = ExpertWeightTransfer::arrivingExperts(manifest, 1);
    ASSERT_EQ(arriving_r1.size(), 2u);
    EXPECT_EQ(arriving_r1[0], 2);
    EXPECT_EQ(arriving_r1[1], 7);
}

TEST(Test__ExpertWeightTransfer, DepartingExperts_EmptyManifest)
{
    std::vector<ExpertMigration> manifest;
    auto departing = ExpertWeightTransfer::departingExperts(manifest, 0);
    EXPECT_TRUE(departing.empty());
}

TEST(Test__ExpertWeightTransfer, ArrivingExperts_EmptyManifest)
{
    std::vector<ExpertMigration> manifest;
    auto arriving = ExpertWeightTransfer::arrivingExperts(manifest, 0);
    EXPECT_TRUE(arriving.empty());
}

// ─── Tag Computation ──────────────────────────────────────────────

TEST(Test__ExpertWeightTransfer, TagComputation_Unique)
{
    // Verify all size tags are unique for a range of inputs
    std::set<int> size_tags;
    for (int layer = 0; layer < 40; ++layer)
    {
        for (int expert = 0; expert < 256; ++expert)
        {
            int tag = weightTransferSizeTag(layer, expert);
            ASSERT_TRUE(size_tags.insert(tag).second)
                << "Duplicate size tag " << tag
                << " at layer=" << layer << " expert=" << expert;
        }
    }

    // Verify all data tags are unique for a subset (full 40×256×3 = 30720)
    std::set<int> data_tags;
    for (int layer = 0; layer < 40; ++layer)
    {
        for (int expert = 0; expert < 256; ++expert)
        {
            for (int proj = 0; proj < 3; ++proj)
            {
                int tag = weightTransferDataTag(layer, expert, proj);
                ASSERT_TRUE(data_tags.insert(tag).second)
                    << "Duplicate data tag " << tag
                    << " at layer=" << layer << " expert=" << expert << " proj=" << proj;
            }
        }
    }
}

TEST(Test__ExpertWeightTransfer, TagComputation_NoCollision)
{
    // Verify size tags and data tags never collide
    std::set<int> all_tags;

    // Insert all size tags for 40 layers × 256 experts
    for (int layer = 0; layer < 40; ++layer)
    {
        for (int expert = 0; expert < 256; ++expert)
        {
            all_tags.insert(weightTransferSizeTag(layer, expert));
        }
    }
    size_t size_tag_count = all_tags.size();
    EXPECT_EQ(size_tag_count, 40u * 256u);

    // Insert all data tags — none should collide with size tags
    for (int layer = 0; layer < 40; ++layer)
    {
        for (int expert = 0; expert < 256; ++expert)
        {
            for (int proj = 0; proj < 3; ++proj)
            {
                bool inserted = all_tags.insert(weightTransferDataTag(layer, expert, proj)).second;
                ASSERT_TRUE(inserted)
                    << "Data tag collides at layer=" << layer
                    << " expert=" << expert << " proj=" << proj;
            }
        }
    }
    // Total: 40*256 size + 40*256*3 data = 10240 + 30720 = 40960
    EXPECT_EQ(all_tags.size(), 40u * 256u + 40u * 256u * 3u);
}

TEST(Test__ExpertWeightTransfer, TagComputation_RangeCheck)
{
    // Size tags should start at WEIGHT_TRANSFER_SIZE
    EXPECT_EQ(weightTransferSizeTag(0, 0), WEIGHT_TRANSFER_SIZE);

    // Data tags should start at WEIGHT_TRANSFER_DATA
    EXPECT_EQ(weightTransferDataTag(0, 0, 0), WEIGHT_TRANSFER_DATA);

    // Max size tag: layer=63, expert=255
    int max_size = weightTransferSizeTag(63, 255);
    EXPECT_LT(max_size, WEIGHT_TRANSFER_DATA); // No overlap with data range

    // Max data tag: layer=63, expert=255, proj=2
    int max_data = weightTransferDataTag(63, 255, 2);
    EXPECT_LT(max_data, 200000); // Well within MPI_TAG_UB
}

// ─── Serialization Round-Trip ─────────────────────────────────────

TEST(Test__ExpertWeightTransfer, RoundTrip_SerializeTransferDeserialize)
{
    // Simulate the full transfer path: serialize → put in ExpertWeightBlobs → deserialize
    // This tests that data survives the blob-based transfer without MPI.

    // Build 3 packed weights (gate, up, down) with different dimensions
    auto gate_packed = buildTestPacked(
        /*N=*/128, /*K=*/256, /*codebook_id=*/0, /*blocks_per_row=*/8,
        /*payload_bytes=*/16, /*is_nibble_lut=*/true, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/1024, /*interleaved_block_stride=*/1280,
        /*interleaved_sz=*/2 * 8 * 1280, /*payload_sz=*/2 * 8 * 64 * 16, /*int8_flat_sz=*/0);
    CPUPackedWeights gate_weights(std::move(gate_packed));

    auto up_packed = buildTestPacked(
        /*N=*/128, /*K=*/256, /*codebook_id=*/0, /*blocks_per_row=*/8,
        /*payload_bytes=*/16, /*is_nibble_lut=*/true, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/1024, /*interleaved_block_stride=*/1280,
        /*interleaved_sz=*/2 * 8 * 1280, /*payload_sz=*/2 * 8 * 64 * 16, /*int8_flat_sz=*/0);
    CPUPackedWeights up_weights(std::move(up_packed));

    auto down_packed = buildTestPacked(
        /*N=*/256, /*K=*/128, /*codebook_id=*/0, /*blocks_per_row=*/4,
        /*payload_bytes=*/16, /*is_nibble_lut=*/true, /*is_asymmetric=*/false,
        /*is_superblock=*/false, /*data_stride=*/512, /*interleaved_block_stride=*/640,
        /*interleaved_sz=*/4 * 4 * 640, /*payload_sz=*/4 * 4 * 64 * 16, /*int8_flat_sz=*/0);
    CPUPackedWeights down_weights(std::move(down_packed));

    // Serialize into blobs (as the sender would)
    ExpertWeightBlobs blobs;
    blobs.gate = serialize(gate_weights);
    blobs.up   = serialize(up_weights);
    blobs.down = serialize(down_weights);

    ASSERT_FALSE(blobs.empty());
    EXPECT_GT(blobs.gate.size(), 96u);  // at least header + section table
    EXPECT_GT(blobs.up.size(), 96u);
    EXPECT_GT(blobs.down.size(), 96u);
    EXPECT_GT(blobs.totalBytes(), 0u);

    // Deserialize (as the receiver would)
    auto gate_result = deserialize(blobs.gate.data(), blobs.gate.size());
    auto up_result   = deserialize(blobs.up.data(),   blobs.up.size());
    auto down_result = deserialize(blobs.down.data(), blobs.down.size());

    ASSERT_NE(gate_result, nullptr);
    ASSERT_NE(up_result, nullptr);
    ASSERT_NE(down_result, nullptr);

    // Verify gate
    EXPECT_EQ(gate_result->format(), PackedWeightsFormat::CPU_NATIVE_VNNI);
    EXPECT_EQ(gate_result->N(), 128);
    EXPECT_EQ(gate_result->K(), 256);

    // Verify up
    EXPECT_EQ(up_result->N(), 128);
    EXPECT_EQ(up_result->K(), 256);

    // Verify down (different dimensions)
    EXPECT_EQ(down_result->N(), 256);
    EXPECT_EQ(down_result->K(), 128);

    // Verify data integrity — gate interleaved data
    const auto* gate_cpu = dynamic_cast<const CPUPackedWeights*>(gate_result.get());
    ASSERT_NE(gate_cpu, nullptr);
    const auto& gp = gate_cpu->packed();
    ASSERT_EQ(gp.native_interleaved.size(), 2u * 8 * 1280);
    ASSERT_EQ(gp.payload.size(), 2u * 8 * 64 * 16);

    // Verify payload pattern is intact
    for (size_t i = 0; i < std::min<size_t>(gp.payload.size(), 100); ++i)
    {
        EXPECT_EQ(gp.payload[i], static_cast<uint8_t>((i * 7 + 13) & 0xFF))
            << "Gate payload mismatch at byte " << i;
    }
}

TEST(Test__ExpertWeightTransfer, TransferAllLayers_EmptyManifest)
{
    // Empty manifest should return empty map without touching MPI
    auto result = ExpertWeightTransfer::transferAllLayers(
        {}, 40,
        [](int, int) -> ExpertWeightBlobs { ADD_FAILURE() << "Should not be called"; return {}; },
        0, MPI_COMM_NULL);
    EXPECT_TRUE(result.empty());
}

TEST(Test__ExpertWeightTransfer, TransferAllLayers_ZeroLayers)
{
    std::vector<ExpertMigration> manifest = {{0, 0, 1}};
    auto result = ExpertWeightTransfer::transferAllLayers(
        manifest, 0,
        [](int, int) -> ExpertWeightBlobs { ADD_FAILURE() << "Should not be called"; return {}; },
        0, MPI_COMM_NULL);
    EXPECT_TRUE(result.empty());
}
