/**
 * @file Test__MoEWeightTransferAttach.cpp
 * @brief Unit tests for Phase 3: KernelFactory transfer registration and
 *        packed weight round-trip (detach → serialize → deserialize → attach).
 *
 * Tests the bypass path where pre-packed GEMM kernels are registered into
 * KernelFactory's prepared registry WITHOUT triggering the full VNNI packing
 * pipeline. This is the core mechanism enabling cross-rank weight transfer.
 */

#include <gtest/gtest.h>

#include "kernels/KernelFactory.h"
#include "kernels/PackedWeightsSerialization.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h"
#include "utils/TestTensorFactory.h"

#include <cstring>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::packed_weights_serialization;
using KernelFactory = llaminar::v2::kernels::KernelFactory;
using PreparedGemmHandle = KernelFactory::PreparedGemmHandle;

namespace {

/// Fill a buffer with a deterministic pattern.
void fillPattern(uint8_t* data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
}

/// Build a CPUNativeVNNIPackedWeights with given dimensions and deterministic data.
CPUNativeVNNIPackedWeights buildTestPacked(int N, int K)
{
    CPUNativeVNNIPackedWeights packed;
    packed.N = N;
    packed.K = K;
    packed.N_padded = ((N + 63) / 64) * 64;
    packed.blocks_per_row = K / 32;
    packed.codebook_id = 2;  // Q4_0
    packed.payload_bytes = 16;
    packed.is_nibble_lut = false;
    packed.is_asymmetric = false;
    packed.is_superblock = false;
    packed.data_stride = 0;
    packed.interleaved_block_stride = 0;

    // Create interleaved data of reasonable size
    size_t interleaved_sz = static_cast<size_t>(packed.N_padded) * packed.blocks_per_row * 64;
    packed.native_interleaved.resize_uninitialized(interleaved_sz);
    fillPattern(packed.native_interleaved.data(), interleaved_sz);

    // Create payload data
    size_t payload_sz = static_cast<size_t>(packed.N_padded) * packed.blocks_per_row * packed.payload_bytes;
    packed.payload.resize(payload_sz);
    fillPattern(packed.payload.data(), payload_sz);

    return packed;
}

} // anonymous namespace

// ─── Test 1: Register creates a valid handle ─────────────────────

TEST(Test__MoEWeightTransferAttach, RegisterPreparedGemmFromTransfer_CreatesValidHandle)
{
    // Build packed weights and create kernel via pre-packed constructor
    auto packed = buildTestPacked(128, 256);
    auto kernel = std::make_unique<CPUNativeVNNIGemmKernel>(std::move(packed));
    ASSERT_TRUE(kernel->isValid());

    // Create a simple FP32 tensor to serve as the cache key (shape determines key)
    auto tensor = llaminar2::test::TestTensorFactory::createFP32Random({128, 256});
    ASSERT_NE(tensor, nullptr);

    DeviceId cpu_device{DeviceType::CPU, 0};

    // Register the pre-packed kernel
    const auto* handle = KernelFactory::registerPreparedGemmFromTransfer(
        tensor.get(), cpu_device, std::move(kernel));

    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle->tensor, tensor.get());
    EXPECT_EQ(handle->device_id.type, DeviceType::CPU);
    EXPECT_EQ(handle->device_id.ordinal, 0);
    EXPECT_NE(handle->prepared_weights, nullptr);
    EXPECT_NE(handle->prepared_weights->kernel, nullptr);

    // Verify engine can be resolved
    auto* engine = KernelFactory::getOrCreateGemmEngine(handle);
    EXPECT_NE(engine, nullptr);

    // Cleanup
    KernelFactory::clearPreparedGemmWeightsFor(tensor.get());
}

// ─── Test 2: Register then resolve hits cache ────────────────────

TEST(Test__MoEWeightTransferAttach, RegisterThenResolve_SkipsPacking)
{
    auto packed = buildTestPacked(64, 128);
    auto kernel = std::make_unique<CPUNativeVNNIGemmKernel>(std::move(packed));
    ASSERT_TRUE(kernel->isValid());

    auto* raw_kernel = kernel.get();
    auto tensor = llaminar2::test::TestTensorFactory::createFP32Random({64, 128});
    DeviceId cpu_device{DeviceType::CPU, 0};

    // Register
    const auto* handle = KernelFactory::registerPreparedGemmFromTransfer(
        tensor.get(), cpu_device, std::move(kernel));
    ASSERT_NE(handle, nullptr);

    // Resolve engine — should get the same kernel (cache hit, no new packing)
    auto* engine = KernelFactory::getOrCreateGemmEngine(handle);
    EXPECT_EQ(engine, raw_kernel);

    // Cleanup
    KernelFactory::clearPreparedGemmWeightsFor(tensor.get());
}

// ─── Test 3: Register then clear removes entry ───────────────────

TEST(Test__MoEWeightTransferAttach, RegisterThenClear_CleanupWorks)
{
    auto packed = buildTestPacked(32, 64);
    auto kernel = std::make_unique<CPUNativeVNNIGemmKernel>(std::move(packed));
    auto tensor = llaminar2::test::TestTensorFactory::createFP32Random({32, 64});
    DeviceId cpu_device{DeviceType::CPU, 0};

    // Track initial registry size
    size_t before = KernelFactory::preparedGemmRegistrySize();

    // Register — adds primary + fallback key entries (2 per tensor with non-null raw_data)
    const auto* handle = KernelFactory::registerPreparedGemmFromTransfer(
        tensor.get(), cpu_device, std::move(kernel));
    ASSERT_NE(handle, nullptr);
    EXPECT_GT(KernelFactory::preparedGemmRegistrySize(), before);

    // Clear — should remove all entries for this tensor (both primary and fallback keys)
    KernelFactory::clearPreparedGemmWeightsFor(tensor.get());
    EXPECT_EQ(KernelFactory::preparedGemmRegistrySize(), before);
}

// ─── Test 4: Serialize → Deserialize → PrePacked constructor ─────

TEST(Test__MoEWeightTransferAttach, SerializeDeserializeRoundTrip_KernelValid)
{
    // Build original packed weights
    auto original_packed = buildTestPacked(128, 256);
    auto original_pw = std::make_unique<CPUPackedWeights>(std::move(original_packed));

    // Serialize
    auto blob = serialize(*original_pw);
    EXPECT_GT(blob.size(), 96u);  // header + section table = 96 bytes minimum

    // Deserialize
    auto deserialized = deserialize(blob.data(), blob.size());
    ASSERT_NE(deserialized, nullptr);
    EXPECT_EQ(deserialized->format(), PackedWeightsFormat::CPU_NATIVE_VNNI);
    EXPECT_EQ(deserialized->N(), 128);
    EXPECT_EQ(deserialized->K(), 256);

    // Create kernel from deserialized packed weights via pre-packed constructor
    auto* cpu_pw = dynamic_cast<CPUPackedWeights*>(deserialized.get());
    ASSERT_NE(cpu_pw, nullptr);

    auto kernel = std::make_unique<CPUNativeVNNIGemmKernel>(cpu_pw->takePacked());
    EXPECT_TRUE(kernel->isValid());
    EXPECT_TRUE(kernel->hasWeights());
}

// ─── Test 5: Detach → Serialize → Deserialize → Attach ──────────

TEST(Test__MoEWeightTransferAttach, DetachSerializeDeserializeAttach_RoundTrip)
{
    // Build original packed weights and create kernel
    auto packed = buildTestPacked(64, 128);
    auto kernel = std::make_unique<CPUNativeVNNIGemmKernel>(std::move(packed));
    ASSERT_TRUE(kernel->isValid());
    ASSERT_TRUE(kernel->hasWeights());

    // Detach weights from kernel
    auto detached = kernel->detachWeights();
    ASSERT_NE(detached, nullptr);
    EXPECT_FALSE(kernel->hasWeights());  // kernel invalidated after detach

    // Serialize the detached weights
    auto blob = serialize(*detached);
    EXPECT_GT(blob.size(), 96u);

    // Deserialize
    auto deserialized = deserialize(blob.data(), blob.size());
    ASSERT_NE(deserialized, nullptr);

    // Create a new kernel from the deserialized weights
    auto* cpu_pw = dynamic_cast<CPUPackedWeights*>(deserialized.get());
    ASSERT_NE(cpu_pw, nullptr);
    auto new_kernel = std::make_unique<CPUNativeVNNIGemmKernel>(cpu_pw->takePacked());
    EXPECT_TRUE(new_kernel->isValid());
    EXPECT_TRUE(new_kernel->hasWeights());

    // Verify dimensions match
    EXPECT_EQ(new_kernel->get_n(), 64);
}

// ─── Test 6: Attach after detach restores kernel ─────────────────

TEST(Test__MoEWeightTransferAttach, AttachAfterDetach_KernelValid)
{
    auto packed = buildTestPacked(32, 64);
    auto kernel = std::make_unique<CPUNativeVNNIGemmKernel>(std::move(packed));
    ASSERT_TRUE(kernel->hasWeights());

    // Detach
    auto detached = kernel->detachWeights();
    ASSERT_NE(detached, nullptr);
    EXPECT_FALSE(kernel->hasWeights());

    // Serialize → deserialize round trip
    auto blob = serialize(*detached);
    auto restored = deserialize(blob.data(), blob.size());
    ASSERT_NE(restored, nullptr);

    // Attach restored weights to the original kernel
    EXPECT_TRUE(kernel->attachWeights(std::move(restored)));
    EXPECT_TRUE(kernel->hasWeights());
    EXPECT_TRUE(kernel->isValid());
}

// ─── Test 7: Null inputs return nullptr ──────────────────────────

TEST(Test__MoEWeightTransferAttach, RegisterNull_ReturnsNullptr)
{
    DeviceId cpu_device{DeviceType::CPU, 0};

    // Null tensor
    EXPECT_EQ(KernelFactory::registerPreparedGemmFromTransfer(
        nullptr, cpu_device, std::make_unique<CPUNativeVNNIGemmKernel>(buildTestPacked(32, 64))),
        nullptr);

    // Null kernel
    auto tensor = llaminar2::test::TestTensorFactory::createFP32Random({32, 64});
    EXPECT_EQ(KernelFactory::registerPreparedGemmFromTransfer(
        tensor.get(), cpu_device, nullptr),
        nullptr);
}
