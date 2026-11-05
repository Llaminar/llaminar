/**
 * @file NativeRowsABIDependency.cpp
 * @brief Explicit device-free dependency for native fixed/counting ABI tests.
 *
 * The real loader resolves this DSO through its ordinary explicit-path API.
 * AllGather, counted-gather and fixed/prefix/slice Broadcast are callable witnesses;
 * other symbols satisfy lookup and reject accidental use. No function reads
 * device pointers, initializes a runtime, performs transport or writes receipts.
 */
#include "NativeRowsABIFixture.h"

namespace
{
    NativeRowsABICall observed;
}

extern "C" void llaminarNativeRowsABIAnchor() {}
extern "C" void llaminarNativeRowsABIReset() { observed = {}; }
extern "C" NativeRowsABICall llaminarNativeRowsABIObservation() { return observed; }

/**
 * @brief Record the native API's actual scalar-element count without execution.
 * @param send Original local source pointer, never accessed.
 * @param receive Original complete-output pointer, never accessed.
 * @param elements Actual scalar count reaching ncclAllGather, not byte count.
 * @param datatype Actual native datatype enum.
 * @param communicator Original opaque participant communicator.
 * @param stream Original exact non-null stream.
 * @return Success so the production wrapper's dispatch result is testable.
 */
extern "C" native_rows_abi::ncclResult_t ncclAllGather(
    const void *send, void *receive, std::size_t elements,
    native_rows_abi::ncclDataType_t datatype, native_rows_abi::ncclComm_t communicator,
    void *stream)
{
    ++observed.fixed_calls;
    observed.send = send;
    observed.receive = receive;
    observed.elements = elements;
    observed.datatype = static_cast<int>(datatype);
    observed.communicator = communicator;
    observed.stream = stream;
    return native_rows_abi::ncclSuccess;
}

/**
 * @brief Observe an erroneous fixed-to-counted dispatch or an explicit control.
 * @param send Borrowed source pointer, never accessed.
 * @param receive Borrowed destination pointer, never accessed.
 * @param rows Borrowed count authority, never accessed.
 * @param capacity Actual row-capacity ABI argument.
 * @param row_bytes Actual byte width supplied to the counted extension.
 * @param receipt Borrowed passive-counter pointer, deliberately never written.
 * @param communicator Original opaque communicator.
 * @param stream Original exact stream.
 * @return Success without simulating native payload traffic.
 */
extern "C" native_rows_abi::ncclResult_t llaminarNcclAllGatherDeviceRows(
    const void *send, void *receive, const std::int32_t *rows,
    std::size_t capacity, std::size_t row_bytes, unsigned long long *receipt,
    native_rows_abi::ncclComm_t communicator, void *stream)
{
    ++observed.counted_calls;
    observed.send = send;
    observed.receive = receive;
    observed.live_rows = rows;
    observed.capacity_rows = capacity;
    observed.row_bytes = row_bytes;
    observed.payload_receipt = receipt;
    observed.communicator = communicator;
    observed.stream = stream;
    return native_rows_abi::ncclSuccess;
}

/**
 * @def LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY
 * @brief Supply lookup-only exports; an unintended invocation fails the test.
 *
 * Like RCCLPartialDependency, these are symbol-admission stubs, never callable
 * collective implementations. Every observed transport call above uses its
 * actual public loader signature. Incrementing unexpected_calls detects misuse.
 */
#define LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(name) \
    extern "C" int name() { ++observed.unexpected_calls; return 5; }
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclGetVersion)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclGetUniqueId)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommInitRank)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommInitRankConfig)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommInitAll)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommDestroy)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommAbort)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommFinalize)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommCount)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommCuDevice)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclCommUserRank)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclGetErrorString)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclAllReduce)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclReduce)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclReduceScatter)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(llaminarNcclAllReduceDeviceRows)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(llaminarNcclReduceScatterDeviceRows)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(llaminarRcclHostTransportStorage)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclSend)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclRecv)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclGroupStart)
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclGroupEnd)
#undef LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY

/**
 * @brief Record ordinary fixed-byte Broadcast without synthesizing a dynamic header.
 * @param send Original source identity, never accessed.
 * @param receive Original destination identity, never accessed.
 * @param count Exact native fixed byte count.
 * @param datatype Actual native wire scalar enum.
 * @param root Original communicator root, independent of device ordinal.
 * @param communicator Original opaque native communicator.
 * @param stream Exact original stream identity.
 * @return Success without native execution or a simulated payload receipt.
 */
extern "C" native_rows_abi::ncclResult_t ncclBroadcast(const void* send, void* receive,
    std::size_t count, native_rows_abi::ncclDataType_t datatype, int root,
    native_rows_abi::ncclComm_t communicator, void* stream)
{
    ++observed.fixed_calls;
    observed.send = send;
    observed.receive = receive;
    observed.elements = count;
    observed.datatype = static_cast<int>(datatype);
    observed.root = root;
    observed.communicator = communicator;
    observed.stream = stream;
    return native_rows_abi::ncclSuccess;
}

/** @brief Record signal dispatch without a native runtime or fabricated device flag.
 * @param communicator Original opaque communicator identity.
 * @return Success for the exact host ABI call only.
 */
extern "C" native_rows_abi::ncclResult_t llaminarNcclSignalCompletionAbort(native_rows_abi::ncclComm_t communicator)
{
    if (!communicator) return native_rows_abi::ncclInvalidArgument;
    ++observed.abort_signal_calls;
    observed.communicator = communicator;
    return native_rows_abi::ncclSuccess;
}

/** @brief Project inert original communicator metadata without any runtime/device read.
 * @param communicator Original opaque test communicator identity.
 * @param original Host output for immutable pointer identity only.
 * @return Success, not transport or native completion evidence.
 */
extern "C" native_rows_abi::ncclResult_t llaminarNcclCompletionAuthority(
    native_rows_abi::ncclComm_t communicator, ncclLlaminarCompletionAuthority* original)
{
    if (!communicator || !original) return native_rows_abi::ncclInvalidArgument;
    ++observed.authority_calls;
    observed.communicator = communicator;
    *original = {reinterpret_cast<const std::uint32_t*>(0x4000), reinterpret_cast<void*>(0x5000)};
    return native_rows_abi::ncclSuccess;
}

/**
 * @brief Observe the exact UINT64 prefix ABI; no count or receipt is accessed.
 * @param send Original source identity.
 * @param receive Original destination identity.
 * @param extent Original device count address, never read.
 * @param capacity Original admitted wire progress bound.
 * @param root Original communicator coordinate.
 * @param completion Original device metadata identity, never read or written.
 * @param communicator Original native communicator identity.
 * @param stream Exact original stream identity.
 * @return Success without native initialization, transport or count simulation.
 */
extern "C" native_rows_abi::ncclResult_t llaminarNcclBroadcastCompletedBytes(const void* send, void* receive,
    std::uint64_t* extent, std::size_t capacity, int root, const ncclLlaminarCompletionOperand* completion,
    native_rows_abi::ncclComm_t communicator, void* stream)
{
    ++observed.byte_prefix_calls;
    observed.send = send;
    observed.receive = receive;
    observed.live_bytes = extent;
    observed.capacity_bytes = capacity;
    observed.root = root;
    observed.completion = completion;
    observed.communicator = communicator;
    observed.stream = stream;
    return native_rows_abi::ncclSuccess;
}

/**
 * @brief Observe source-only descriptor plus receiver metadata without a host dereference.
 * @param send Original root source identity.
 * @param receive Original destination identity.
 * @param slice Original root descriptor or null on a receiver.
 * @param extent Original root/receiver UINT64 field, never read.
 * @param source_capacity Exact admitted root subspan bytes, zero on receivers.
 * @param source_offset_origin Immutable absolute subspan lower bound, zero on receivers.
 * @param capacity Original wire progress bound, never predicted useful bytes.
 * @param root Original communicator coordinate.
 * @param completion Original device metadata identity, never read or written.
 * @param communicator Original native communicator identity.
 * @param stream Exact original stream identity.
 * @return Success without native execution or fabricated completion evidence.
 */
extern "C" native_rows_abi::ncclResult_t llaminarNcclBroadcastCompletedSliceOrigin(const void* send, void* receive,
    const llaminar2::NativeByteSourceSlice* slice, std::uint64_t* extent,
    std::size_t source_capacity, std::size_t capacity, int root, std::uint64_t source_offset_origin,
    const ncclLlaminarCompletionOperand* completion,
    native_rows_abi::ncclComm_t communicator, void* stream)
{
    ++observed.byte_slice_calls;
    observed.send = send;
    observed.receive = receive;
    observed.source_slice = slice;
    observed.live_bytes = extent;
    observed.source_capacity_bytes = source_capacity;
    observed.source_offset_origin = source_offset_origin;
    observed.capacity_bytes = capacity;
    observed.root = root;
    observed.completion = completion;
    observed.communicator = communicator;
    observed.stream = stream;
    return native_rows_abi::ncclSuccess;
}

/** @brief Recorded DSO fixture implements the same semantic marker without native runtime work. */
extern "C" std::uint32_t llaminarNcclCompletedProtocolVersion() { return 3; }

/** @brief Observe addressed source identity without executing a native transport or publishing a receipt. */
extern "C" native_rows_abi::ncclResult_t llaminarNcclSendCompletedBytes(const void* payload,std::uint64_t* extent,
    std::size_t capacity,int peer,const ncclLlaminarCompletionOperand* completion,native_rows_abi::ncclComm_t comm,void* stream) {
    return llaminarNcclBroadcastCompletedBytes(payload,nullptr,extent,capacity,peer,completion,comm,stream);
}
/** @brief Observe addressed destination identity; unrelated communicator members contribute no call. */
extern "C" native_rows_abi::ncclResult_t llaminarNcclRecvCompletedBytes(void* payload,std::uint64_t* extent,
    std::size_t capacity,int peer,const ncclLlaminarCompletionOperand* completion,native_rows_abi::ncclComm_t comm,void* stream) {
    return llaminarNcclBroadcastCompletedBytes(nullptr,payload,extent,capacity,peer,completion,comm,stream);
}
/** @brief Observe the exact original addressed source slice; its count and bytes remain unobserved. */
extern "C" native_rows_abi::ncclResult_t llaminarNcclSendCompletedSliceOrigin(const void* payload,
    const llaminar2::NativeByteSourceSlice* slice,std::size_t sourceCapacity,std::size_t capacity,int peer,std::uint64_t origin,
    const ncclLlaminarCompletionOperand* completion,native_rows_abi::ncclComm_t comm,void* stream) {
    return llaminarNcclBroadcastCompletedSliceOrigin(payload,nullptr,slice,const_cast<std::uint64_t*>(&slice->live_bytes),
        sourceCapacity,capacity,peer,origin,completion,comm,stream);
}
