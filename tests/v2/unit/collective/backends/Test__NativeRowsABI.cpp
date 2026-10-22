/**
 * @file Test__NativeRowsABI.cpp
 * @brief Exercise actual NCCL/RCCL fixed-row dispatch at its native call ABI.
 *
 * The integration vocabulary fixture proves real captured output and poisoned
 * tails; this complementary device-free test records the exact send count at
 * the last project-owned loader boundary. Explicit fixture loading, dladdr and
 * ordinary target linkage follow the existing RCCL dependency-admission tests.
 * No device, native communicator, transport or byte-receipt producer is created.
 */
#include "NativeRowsABIFixture.h"
#include <gtest/gtest.h>
#include <array>
#include <dlfcn.h>

namespace llaminar2::test
{
namespace
{
    /** @brief Isolate one actual loader and retire it even after assertion failure. */
    class NativeRowsABI : public ::testing::Test
    {
    protected:
        /** @brief Admit the exact linked host-only fixture without runtime initialization. */
        void SetUp() override
        {
            ASSERT_FALSE(native_rows_abi::isLoaded());
            llaminarNativeRowsABIAnchor();
            Dl_info owner{};
            ASSERT_NE(dladdr(reinterpret_cast<void *>(&llaminarNativeRowsABIAnchor), &owner), 0);
            ASSERT_NE(owner.dli_fname, nullptr);
            ASSERT_TRUE(native_rows_abi::load(owner.dli_fname)) << native_rows_abi::getLastError();
            llaminarNativeRowsABIReset();
        }
        /** @brief Withdraw all callbacks before the linked fixture owner retires. */
        void TearDown() override { native_rows_abi::unload(); }

        // Inert host addresses supply identity only. The spy never reads the
        // 62080-element payload; it records the actual native argument instead.
        std::array<std::byte, 64> source_{}, destination_{};
        std::array<std::byte, 16> communicator_{}, stream_{};
    };

    /** @test A real fixed M1 dispatch reaches ncclAllGather as62080 FP32 elements. */
    TEST_F(NativeRowsABI, FixedM1ForwardsExactVocabularyExtentAndOwnerIdentity)
    {
        const NativeCollectiveRows rows(DeviceRowRange::fullyActive(1), 62080);
        ASSERT_EQ(native_rows_abi::nativeRows(NativeRowCollective::AllGather,
            source_.data(), destination_.data(), rows, native_rows_abi::ncclFloat32,
            native_rows_abi::ncclSum, nullptr, communicator_.data(), stream_.data()),
            native_rows_abi::ncclSuccess);
        const auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.fixed_calls, 1u);
        EXPECT_EQ(call.counted_calls, 0u);
        EXPECT_EQ(call.unexpected_calls, 0u);
        EXPECT_EQ(call.elements, 62080u);
        EXPECT_EQ(call.elements * sizeof(float), 248320u);
        EXPECT_EQ(call.datatype, static_cast<int>(native_rows_abi::ncclFloat32));
        EXPECT_EQ(call.send, source_.data());
        EXPECT_EQ(call.receive, destination_.data());
        EXPECT_EQ(call.communicator, communicator_.data());
        EXPECT_EQ(call.stream, stream_.data());
        EXPECT_EQ(call.payload_receipt, nullptr);
    }

    /** @test Fixed native rows reject passive counters without calling either ABI. */
    TEST_F(NativeRowsABI, FixedM1CannotFabricateADevicePayloadReceipt)
    {
        const NativeCollectiveRows rows(DeviceRowRange::fullyActive(1), 62080);
        alignas(8) unsigned long long counter = 0x13579bdf2468ace0ULL;
        EXPECT_EQ(native_rows_abi::nativeRows(NativeRowCollective::AllGather,
            source_.data(), destination_.data(), rows, native_rows_abi::ncclFloat32,
            native_rows_abi::ncclSum, &counter, communicator_.data(), stream_.data()),
            native_rows_abi::ncclInvalidArgument);
        const auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.fixed_calls, 0u);
        EXPECT_EQ(call.counted_calls, 0u);
        EXPECT_EQ(call.unexpected_calls, 0u);
        EXPECT_EQ(counter, 0x13579bdf2468ace0ULL);
    }

    /** @test The counted-call trap is live and keeps its distinct byte-width ABI. */
    TEST_F(NativeRowsABI, CountedControlUsesOnlyTheCountedExtension)
    {
        std::int32_t count = 1;
        alignas(8) unsigned long long counter = 0x2468ace013579bdfULL;
        const NativeCollectiveRows rows(DeviceRowRange::deviceCounted(16, &count), 62080);
        ASSERT_EQ(native_rows_abi::nativeRows(NativeRowCollective::AllGather,
            source_.data(), destination_.data(), rows, native_rows_abi::ncclFloat32,
            native_rows_abi::ncclSum, &counter, communicator_.data(), stream_.data()),
            native_rows_abi::ncclSuccess);
        const auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.fixed_calls, 0u);
        EXPECT_EQ(call.counted_calls, 1u);
        EXPECT_EQ(call.unexpected_calls, 0u);
        EXPECT_EQ(call.live_rows, &count);
        EXPECT_EQ(call.capacity_rows, 16u);
        EXPECT_EQ(call.row_bytes, 248320u);
        EXPECT_EQ(call.send, source_.data());
        EXPECT_EQ(call.receive, destination_.data());
        EXPECT_EQ(call.communicator, communicator_.data());
        EXPECT_EQ(call.stream, stream_.data());
        EXPECT_EQ(call.payload_receipt, &counter);
        EXPECT_EQ(counter, 0x2468ace013579bdfULL);
    }
}
} // namespace llaminar2::test
