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

    /** @test Header-compiled MIN/MAX ordinals reach the actual isolated loader ABI unchanged. */
    TEST_F(NativeRowsABI, ReductionOpcodesMatchConfiguredHeaderAndNativeDispatch)
    {
        ASSERT_TRUE(native_rows_abi::configuredReductionABIContract());
        for (const auto [operation, expected] : std::array{
            std::pair{native_rows_abi::ncclMin, 3}, std::pair{native_rows_abi::ncclMax, 2}})
        {
            llaminarNativeRowsABIReset();
            ASSERT_EQ(native_rows_abi::ncclAllReduce(source_.data(), destination_.data(), 1,
                native_rows_abi::ncclInt32, operation, communicator_.data(), stream_.data()), native_rows_abi::ncclSuccess);
            const auto call=llaminarNativeRowsABIObservation();
            EXPECT_EQ(call.fixed_calls, 1u); EXPECT_EQ(call.unexpected_calls, 0u);
            EXPECT_EQ(call.elements, 1u); EXPECT_EQ(call.reduction, expected);
            EXPECT_EQ(call.datatype, int(native_rows_abi::ncclInt32));
            EXPECT_EQ(call.send, source_.data()); EXPECT_EQ(call.receive, destination_.data());
            EXPECT_EQ(call.communicator, communicator_.data()); EXPECT_EQ(call.stream, stream_.data());
        }
    }

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
    /** @test Known fixed headers/ACKs use only the ordinary native API. */
    TEST_F(NativeRowsABI, NativeBytesFixedHasNoDynamicHeaderOrCountBinding)
    {
        const auto bytes = NativeCollectiveBytes::fixed(64);
        ASSERT_EQ(native_rows_abi::nativeBroadcastBytes(source_.data(), destination_.data(),
            bytes, 1, communicator_.data(), stream_.data()), native_rows_abi::ncclSuccess);
        const auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.fixed_calls, 1u);
        EXPECT_EQ(call.byte_prefix_calls, 0u);
        EXPECT_EQ(call.byte_slice_calls, 0u);
        EXPECT_EQ(call.elements, 64u);
        EXPECT_EQ(call.datatype, static_cast<int>(native_rows_abi::ncclInt8));
        EXPECT_EQ(call.root, 1);
        EXPECT_EQ(call.live_bytes, nullptr);
        EXPECT_EQ(call.communicator, communicator_.data());
        EXPECT_EQ(call.stream, stream_.data());
    }

    /** @test An inaccessible device word is retained exactly, never truncated or host-read. */
    TEST_F(NativeRowsABI, NativeBytesPrefixPreservesOriginalCountReceiptAndStream)
    {
        auto* extent = reinterpret_cast<std::uint64_t*>(std::uintptr_t{0x1000});
        auto* completion = reinterpret_cast<const ncclLlaminarCompletionOperand*>(std::uintptr_t{0x6000});
        const auto bytes = NativeCollectiveBytes::prefix(extent, 737280, completion);
        ASSERT_EQ(native_rows_abi::nativeBroadcastBytes(source_.data(), destination_.data(),
            bytes, 1, communicator_.data(), stream_.data()), native_rows_abi::ncclSuccess);
        const auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.fixed_calls, 0u);
        EXPECT_EQ(call.byte_prefix_calls, 1u);
        EXPECT_EQ(call.byte_slice_calls, 0u);
        EXPECT_EQ(call.live_bytes, extent);
        EXPECT_EQ(call.capacity_bytes, 737280u);
        EXPECT_EQ(call.completion, completion);
        EXPECT_EQ(call.send, source_.data());
        EXPECT_EQ(call.receive, destination_.data());
        EXPECT_EQ(call.communicator, communicator_.data());
        EXPECT_EQ(call.stream, stream_.data());
    }

    /** @test Source and receiver operands keep distinct original metadata roles. */
    TEST_F(NativeRowsABI, NativeBytesSliceUsesRootDescriptorAndSeparateReceiverExtent)
    {
        auto* slice = reinterpret_cast<NativeByteSourceSlice*>(std::uintptr_t{0x2000});
        const auto root_bytes = NativeCollectiveBytes::sourceSlice(slice, 8192, 4099,
            reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000), 0x1234567800000000ULL);
        ASSERT_EQ(native_rows_abi::nativeBroadcastBytes(source_.data(), destination_.data(),
            root_bytes, 1, communicator_.data(), stream_.data()), native_rows_abi::ncclSuccess);
        auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.byte_slice_calls, 1u);
        EXPECT_EQ(call.fixed_calls, 0u);
        EXPECT_EQ(call.byte_prefix_calls, 0u);
        EXPECT_EQ(call.source_slice, slice);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(call.live_bytes), 0x2008u);
        EXPECT_EQ(call.source_capacity_bytes, 8192u);
        EXPECT_EQ(call.source_offset_origin, 0x1234567800000000ULL);
        EXPECT_EQ(call.capacity_bytes, 4099u);
        EXPECT_EQ(call.root, 1);
        llaminarNativeRowsABIReset();
        auto* receiver = reinterpret_cast<std::uint64_t*>(std::uintptr_t{0x3000});
        const auto receive_bytes = NativeCollectiveBytes::receiveSlice(receiver, 4099, reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000));
        ASSERT_EQ(native_rows_abi::nativeBroadcastBytes(source_.data(), destination_.data(),
            receive_bytes, 1, communicator_.data(), stream_.data()), native_rows_abi::ncclSuccess);
        call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.source_slice, nullptr);
        EXPECT_EQ(call.live_bytes, receiver);
        EXPECT_EQ(call.source_capacity_bytes, 0u);
        EXPECT_EQ(call.source_offset_origin, 0u);
        EXPECT_EQ(call.capacity_bytes, 4099u);
        EXPECT_EQ(call.root, 1);
    }

    /** @test Only the actual communicator root can retain a source-relative descriptor. */
    TEST_F(NativeRowsABI, NativeBytesRolesUseOriginalCommunicatorCoordinates)
    {
        auto* slice = reinterpret_cast<NativeByteSourceSlice*>(std::uintptr_t{0x2000});
        auto* extent = reinterpret_cast<std::uint64_t*>(std::uintptr_t{0x3000});
        const auto source = NativeCollectiveBytes::sourceSlice(slice, 8192, 4099, reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000));
        const auto receive = NativeCollectiveBytes::receiveSlice(extent, 4099, reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000));
        for (int degree : {2, 4, 8})
            for (int participant = 0; participant < degree; ++participant)
            {
                EXPECT_EQ(source.participantValid(degree, participant, 1), participant == 1);
                EXPECT_EQ(receive.participantValid(degree, participant, 1), participant != 1);
            }
        EXPECT_FALSE(source.participantValid(1, 0, 0));
        EXPECT_FALSE(source.participantValid(2, -1, 1));
        EXPECT_FALSE(source.participantValid(2, 1, 2));
    }

    /** @test Actual addressed loader dispatch preserves exact peer, role and original device metadata identity. */
    TEST_F(NativeRowsABI, AddressedBytesUseOnlyActualOriginalEndpointsAndExactOperands) {
        auto* extent=reinterpret_cast<std::uint64_t*>(0x3000);
        const auto* completion=reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000);
        const NativeByteEdge edge(3,6);const auto prefix=NativeCollectiveBytes::prefix(extent,4099,completion);
        ASSERT_EQ(native_rows_abi::nativeTransferBytes(source_.data(),prefix,edge,3,communicator_.data(),stream_.data()),native_rows_abi::ncclSuccess);
        auto call=llaminarNativeRowsABIObservation();EXPECT_EQ(call.send,source_.data());EXPECT_EQ(call.receive,nullptr);EXPECT_EQ(call.root,6);
        EXPECT_EQ(call.live_bytes,extent);EXPECT_EQ(call.capacity_bytes,4099u);EXPECT_EQ(call.completion,completion);
        llaminarNativeRowsABIReset();
        ASSERT_EQ(native_rows_abi::nativeTransferBytes(destination_.data(),prefix,edge,6,communicator_.data(),stream_.data()),native_rows_abi::ncclSuccess);
        call=llaminarNativeRowsABIObservation();EXPECT_EQ(call.send,nullptr);EXPECT_EQ(call.receive,destination_.data());EXPECT_EQ(call.root,3);
        llaminarNativeRowsABIReset();
        EXPECT_EQ(native_rows_abi::nativeTransferBytes(destination_.data(),prefix,edge,2,communicator_.data(),stream_.data()),native_rows_abi::ncclInvalidArgument);
        EXPECT_EQ(llaminarNativeRowsABIObservation().byte_prefix_calls,0u);
        auto* slice=reinterpret_cast<NativeByteSourceSlice*>(0x2000);
        const auto sliced=NativeCollectiveBytes::sourceSlice(slice,8192,4099,completion,0x12340000);
        ASSERT_EQ(native_rows_abi::nativeTransferBytes(source_.data(),sliced,edge,3,communicator_.data(),stream_.data()),native_rows_abi::ncclSuccess);
        call=llaminarNativeRowsABIObservation();EXPECT_EQ(call.source_slice,slice);EXPECT_EQ(call.source_offset_origin,0x12340000u);
        EXPECT_EQ(call.source_capacity_bytes,8192u);EXPECT_EQ(call.capacity_bytes,4099u);EXPECT_EQ(call.root,6);
        llaminarNativeRowsABIReset();
        EXPECT_EQ(native_rows_abi::nativeTransferBytes(destination_.data(),sliced,edge,6,communicator_.data(),stream_.data()),native_rows_abi::ncclInvalidArgument);
        const auto receive=NativeCollectiveBytes::receiveSlice(extent,4099,completion);
        EXPECT_EQ(native_rows_abi::nativeTransferBytes(source_.data(),receive,edge,3,communicator_.data(),stream_.data()),native_rows_abi::ncclInvalidArgument);
        EXPECT_EQ(llaminarNativeRowsABIObservation().byte_prefix_calls,0u);EXPECT_EQ(llaminarNativeRowsABIObservation().byte_slice_calls,0u);
    }

    /** @test Invalid geometry fails without reading any device publication. */
    TEST_F(NativeRowsABI, NativeBytesRejectInvalidStaticGeometryAndDefaultStreams)
    {
        EXPECT_THROW(NativeCollectiveBytes::fixed(0), std::invalid_argument);
        EXPECT_THROW(NativeCollectiveBytes::prefix(nullptr, 8, reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000)), std::invalid_argument);
        EXPECT_THROW(NativeCollectiveBytes::sourceSlice(nullptr, 16, 8, reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000)), std::invalid_argument);
        EXPECT_THROW(NativeCollectiveBytes::sourceSlice(reinterpret_cast<NativeByteSourceSlice*>(0x2000), 4, 8, reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000)), std::invalid_argument);
        EXPECT_THROW(NativeCollectiveBytes::prefix(reinterpret_cast<std::uint64_t*>(0x1001), 8, reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6000)), std::invalid_argument);
        EXPECT_THROW(NativeCollectiveBytes::prefix(reinterpret_cast<std::uint64_t*>(0x1000), 8, nullptr), std::invalid_argument);
        EXPECT_THROW(NativeCollectiveBytes::prefix(reinterpret_cast<std::uint64_t*>(0x1000), 8,
            reinterpret_cast<const ncclLlaminarCompletionOperand*>(0x6001)), std::invalid_argument);
        for (std::uintptr_t stream : {0, 1, 2})
            EXPECT_EQ(native_rows_abi::nativeBroadcastBytes(source_.data(), destination_.data(),
                NativeCollectiveBytes::fixed(8), 0, communicator_.data(), reinterpret_cast<void*>(stream)),
                native_rows_abi::ncclInvalidArgument);
        const auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.fixed_calls, 0u);
        EXPECT_EQ(call.byte_prefix_calls, 0u);
        EXPECT_EQ(call.byte_slice_calls, 0u);
    }

    /** @test Original native communicator capability is metadata-only and never guessed from ordinal. */
    TEST_F(NativeRowsABI, NativeBytesOriginalCompletionAuthorityPreservesExactPointerIdentity)
    {
        ncclLlaminarCompletionAuthority original{};
        ASSERT_EQ(native_rows_abi::nativeCompletionAuthority(communicator_.data(), &original), native_rows_abi::ncclSuccess);
        const auto call = llaminarNativeRowsABIObservation();
        EXPECT_EQ(call.authority_calls, 1u);
        EXPECT_EQ(call.communicator, communicator_.data());
        EXPECT_EQ(original.abortFlag, reinterpret_cast<const std::uint32_t*>(0x4000));
        EXPECT_EQ(original.deviceCommunicator, reinterpret_cast<const void*>(0x5000));
        EXPECT_EQ(native_rows_abi::nativeCompletionAuthority(nullptr, &original), native_rows_abi::ncclInvalidArgument);
        EXPECT_EQ(native_rows_abi::nativeCompletionAuthority(communicator_.data(), nullptr), native_rows_abi::ncclInvalidArgument);
        EXPECT_EQ(native_rows_abi::signalNativeCompletionAbort(communicator_.data()), native_rows_abi::ncclSuccess);
        EXPECT_EQ(llaminarNativeRowsABIObservation().abort_signal_calls, 1u);
        EXPECT_EQ(native_rows_abi::signalNativeCompletionAbort(nullptr), native_rows_abi::ncclInvalidArgument);
    }

    /** @test Zero bytes require exact fresh native success rather than a passive count. */
    TEST_F(NativeRowsABI, NativeBytesCompletionRejectsEmptyStaleForeignAndCancelledEvidence)
    {
        std::uint32_t abort = 0;
        const ncclLlaminarCompletionAuthority original{&abort, reinterpret_cast<void*>(0x5000)};
        ncclLlaminarByteCompletion receipt{17, 23, 31, 0, &abort, original.deviceCommunicator,
            51, ncclLlaminarCompletionPhase::Completed, 0};
        const auto accepted = [&](const auto& observed, auto authority, std::uint32_t cancellation, int first_error) {
            return ncclLlaminarCompletionMatches(observed, 17, 23, 31, 0, 51, authority, cancellation, first_error);
        };
        ASSERT_TRUE(accepted(receipt, original, 0, 0));
        for (int fault = 0; fault < 8; ++fault)
        {
            auto observed = receipt;
            auto authority = original;
            std::uint32_t cancellation = 0;
            int first_error = 0;
            switch (fault)
            {
            case 0: observed.phase = ncclLlaminarCompletionPhase::Pending; break;
            case 1: observed.admission++; break;
            case 2: observed.sequence++; break;
            case 3: observed.message++; break;
            case 4: observed.bytes++; break;
            case 5: authority.deviceCommunicator = reinterpret_cast<void*>(0x7000); break;
            case 6: cancellation = 1; break;
            case 7: first_error = 37; break;
            }
            EXPECT_FALSE(accepted(observed, authority, cancellation, first_error)) << fault;
        }
    }

}
} // namespace llaminar2::test
