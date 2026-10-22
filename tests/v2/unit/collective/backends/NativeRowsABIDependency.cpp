/**
 * @file NativeRowsABIDependency.cpp
 * @brief Explicit device-free dependency for native fixed/counting ABI tests.
 *
 * The real loader resolves this DSO through its ordinary explicit-path API.
 * Only ncclAllGather and the counted-gather extension are callable witnesses;
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
LLAMINAR_NATIVE_ROWS_LOOKUP_ONLY(ncclBroadcast)
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
