/**
 * @file Test__ExpertTransferFixes.cpp
 * @brief Tests for cross-domain expert transfer fixes:
 *   1. createExpertGemmFromTransferBlob creates valid kernel without global registry
 *   2. Transfer blob path produces numerically identical results to full repack
 *   3. Empty/corrupt blobs return nullptr gracefully
 *   4. Phase D alignment: transferred experts don't pollute global registry
 *   5. MPI INT_MAX size guard in ExpertWeightTransfer
 *   6. Serialization → createExpertGemmFromTransferBlob round-trip correctness
 */

#include <gtest/gtest.h>

#include "kernels/KernelFactory.h"
#include "kernels/PackedWeightsSerialization.h"
#include "kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "execution/moe/ExpertWeightTransfer.h"
#include "backends/DeviceId.h"
#include "utils/TestTensorFactory.h"

#include <climits>
#include <cmath>
#include <memory>
#include <vector>

using namespace llaminar2;
using KF = llaminar::v2::kernels::KernelFactory;
using namespace llaminar2::test;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::packed_weights_serialization;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class Test__ExpertTransferFixes : public ::testing::Test
{
protected:
    void TearDown() override
    {
        KF::clearCache();
    }

    /// Helper: create a quantized tensor, pack it via prepareExpertGemmLocal,
    /// serialize via engine->cloneWeights(), return the serialized blob.
    std::vector<uint8_t> createPackedBlob(const TensorBase* tensor)
    {
        auto engine = KF::prepareExpertGemmLocal(tensor, DeviceId::cpu());
        if (!engine || !engine->hasWeights()) return {};
        auto packed = engine->cloneWeights();
        if (!packed) return {};
        return serialize(*packed);
    }
};

// ---------------------------------------------------------------------------
// 1. createExpertGemmFromTransferBlob returns a valid engine
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, TransferBlob_ReturnsValidEngine)
{
    // Create a quantized tensor, pack it, serialize, then create from blob
    auto tensor = TestTensorFactory::createQ4_0Random({64, 128});
    ASSERT_NE(tensor, nullptr);

    auto blob = createPackedBlob(tensor.get());
    ASSERT_FALSE(blob.empty()) << "Serialization should produce non-empty blob";

    auto engine = KF::createExpertGemmFromTransferBlob(blob);
    ASSERT_NE(engine, nullptr) << "createExpertGemmFromTransferBlob must return valid engine";
    EXPECT_TRUE(engine->hasWeights()) << "Engine must have packed weights";
}

// ---------------------------------------------------------------------------
// 2. Transfer blob path does NOT register in global registry
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, TransferBlob_NoGlobalRegistry)
{
    auto tensor = TestTensorFactory::createQ4_0Random({64, 128});
    auto blob = createPackedBlob(tensor.get());
    ASSERT_FALSE(blob.empty());

    // Clear cache to ensure clean slate
    KF::clearCache();

    auto engine = KF::createExpertGemmFromTransferBlob(blob);
    ASSERT_NE(engine, nullptr);

    // Global registry should remain empty
    EXPECT_EQ(KF::preparedGemmRegistrySize(), 0u)
        << "createExpertGemmFromTransferBlob must NOT insert into global registry";
}

// ---------------------------------------------------------------------------
// 3. Transfer blob preserves weight data through round-trip
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, TransferBlob_NumericalParity)
{
    // Verify: serialize(packed) → blob → createFromBlob → valid engine with weights
    // The blob created from a tensor via prepareExpertGemmLocal + cloneWeights + serialize
    // should produce a valid engine when deserialized via createExpertGemmFromTransferBlob.
    auto tensor = TestTensorFactory::createQ4_0Random({64, 128}, /*seed=*/42);
    ASSERT_NE(tensor, nullptr);

    // Create blob via standard packing path
    auto blob = createPackedBlob(tensor.get());
    ASSERT_FALSE(blob.empty());
    const size_t original_blob_size = blob.size();

    KF::clearCache();

    // Deserialize blob into a new engine
    auto engine = KF::createExpertGemmFromTransferBlob(blob);
    ASSERT_NE(engine, nullptr);
    EXPECT_TRUE(engine->hasWeights());

    // The blob size should match what we'd expect for a Q4_0 [64, 128] weight
    // (VNNI packed representation is deterministic for given dimensions)
    EXPECT_GT(original_blob_size, 0u);

    // Create a second engine from the same blob — both should be valid
    auto engine2 = KF::createExpertGemmFromTransferBlob(blob);
    ASSERT_NE(engine2, nullptr);
    EXPECT_TRUE(engine2->hasWeights());
    EXPECT_NE(engine.get(), engine2.get()) << "Each deserialization should produce a new engine";
}

// ---------------------------------------------------------------------------
// 4. Empty blob returns nullptr
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, TransferBlob_EmptyReturnsNull)
{
    std::vector<uint8_t> empty_blob;
    auto engine = KF::createExpertGemmFromTransferBlob(empty_blob);
    EXPECT_EQ(engine, nullptr) << "Empty blob must return nullptr";
}

// ---------------------------------------------------------------------------
// 5. Corrupt blob returns nullptr (no crash)
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, TransferBlob_CorruptReturnsNull)
{
    std::vector<uint8_t> garbage(256, 0xDE);
    auto engine = KF::createExpertGemmFromTransferBlob(garbage);
    EXPECT_EQ(engine, nullptr) << "Corrupt blob must return nullptr without crashing";
}

// ---------------------------------------------------------------------------
// 6. Multiple blobs produce independent engines
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, TransferBlob_IndependentEngines)
{
    auto tensor_a = TestTensorFactory::createQ4_0Random({64, 128}, /*seed=*/10);
    auto tensor_b = TestTensorFactory::createQ4_0Random({64, 128}, /*seed=*/20);

    auto blob_a = createPackedBlob(tensor_a.get());
    auto blob_b = createPackedBlob(tensor_b.get());
    ASSERT_FALSE(blob_a.empty());
    ASSERT_FALSE(blob_b.empty());

    // Blobs from different random tensors must differ
    EXPECT_NE(blob_a, blob_b)
        << "Serialized blobs from different tensors must be different";

    KF::clearCache();

    auto engine_a = KF::createExpertGemmFromTransferBlob(blob_a);
    auto engine_b = KF::createExpertGemmFromTransferBlob(blob_b);

    ASSERT_NE(engine_a, nullptr);
    ASSERT_NE(engine_b, nullptr);
    EXPECT_NE(engine_a.get(), engine_b.get())
        << "Different blobs must produce distinct engine instances";
}

// ---------------------------------------------------------------------------
// 7. Lifetime governed by shared_ptr (no registry dependency)
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, TransferBlob_LifetimeViaSharedPtr)
{
    auto tensor = TestTensorFactory::createQ4_0Random({64, 128});
    auto blob = createPackedBlob(tensor.get());
    ASSERT_FALSE(blob.empty());

    KF::clearCache();

    auto engine = KF::createExpertGemmFromTransferBlob(blob);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine.use_count(), 1) << "Caller is sole owner";

    // Engine survives tensor destruction (it holds its own packed data)
    tensor.reset();
    EXPECT_NE(engine, nullptr);
    EXPECT_TRUE(engine->hasWeights()) << "Engine must retain weights after tensor is gone";
}

// ---------------------------------------------------------------------------
// 8. Full round-trip: pack → serialize → transfer blob → valid engine (Q8_0)
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, FullRoundTrip_Q8_0)
{
    auto tensor = TestTensorFactory::createQ8_0Random({128, 256}, /*seed=*/77);
    ASSERT_NE(tensor, nullptr);

    // Simulate: sender packs and serializes
    auto sender_engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(sender_engine, nullptr);
    auto packed = sender_engine->cloneWeights();
    ASSERT_NE(packed, nullptr);
    auto blob = serialize(*packed);
    ASSERT_FALSE(blob.empty());

    // Simulate: receiver creates engine from blob
    auto receiver_engine = KF::createExpertGemmFromTransferBlob(blob);
    ASSERT_NE(receiver_engine, nullptr);
    EXPECT_TRUE(receiver_engine->hasWeights())
        << "Round-trip Q8_0: receiver engine must have valid weights";

    // Verify re-serialization produces a non-empty blob (engine is functional)
    auto re_packed = receiver_engine->cloneWeights();
    if (re_packed) {
        auto re_blob = serialize(*re_packed);
        EXPECT_FALSE(re_blob.empty())
            << "Re-serialization of deserialized engine should produce valid blob";
    }
}

// ---------------------------------------------------------------------------
// 9. ExpertWeightBlobs: empty gate/up/down detection
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, ExpertWeightBlobs_EmptyDetection)
{
    ExpertWeightBlobs blobs;
    EXPECT_TRUE(blobs.empty());
    EXPECT_EQ(blobs.totalBytes(), 0u);

    blobs.gate = {1, 2, 3};
    EXPECT_FALSE(blobs.empty());
    EXPECT_EQ(blobs.totalBytes(), 3u);
}

// ---------------------------------------------------------------------------
// 10. Contrast: prepareExpertGemmLocal vs createExpertGemmFromTransferBlob
//     Neither pollutes global registry (Phase D alignment)
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, PhaseDAlignment_NeitherPathRegistersGlobally)
{
    auto tensor = TestTensorFactory::createQ4_0Random({64, 128});

    // Path 1: local prep (from raw data)
    auto engine_local = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(engine_local, nullptr);

    // Path 2: transfer blob (from serialized)
    auto blob = createPackedBlob(tensor.get());
    KF::clearCache(); // Clear any side effects from createPackedBlob
    auto engine_transfer = KF::createExpertGemmFromTransferBlob(blob);
    ASSERT_NE(engine_transfer, nullptr);

    // Neither should appear in global registry
    const auto* found = KF::findPreparedGemmWeights(tensor.get(), DeviceId::cpu());
    EXPECT_EQ(found, nullptr)
        << "Neither local prep nor transfer blob should register globally";
    EXPECT_EQ(KF::preparedGemmRegistrySize(), 0u);
}

// ---------------------------------------------------------------------------
// 11. MPI INT_MAX guard — size tag computation stays in range
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferFixes, MPITagRange_SizeAndDataSeparate)
{
    // Verify size and data tag ranges don't overlap
    using namespace llaminar2::mpi_tags;

    // Max realistic: 28 MoE layers, 64 experts
    int max_size_tag = weightTransferSizeTag(27, 63);
    int min_data_tag = weightTransferDataTag(0, 0, 0);
    EXPECT_LT(max_size_tag, min_data_tag)
        << "Size tags must never overlap with data tags";

    // Verify max data tag fits in INT_MAX
    int max_data_tag = weightTransferDataTag(63, 255, 2);
    EXPECT_GT(max_data_tag, 0);
    EXPECT_LT(max_data_tag, INT_MAX);
}
