/**
 * @file RCCLPartialDependency.cpp
 * @brief Device-free, deliberately incomplete DSO for loader publication tests.
 *
 * Required symbols before the final group-end symbol can resolve, but admission
 * must fail and withdraw every callback. No stub initializes a device. The
 * unique anchor makes this DSO an ordinary executable dependency, so installed
 * test runners retain it through normal library-closure staging rather than
 * an embedded build path or a compiler dependency at test time.
 */
#include "collective/backends/RCCLHostTransportStorage.h"

namespace
{
    unsigned storage_calls = 0;
}

/** @brief Link anchor and address owner for the fixture's actual installed DSO. */
extern "C" void llaminarPartialDependencyAnchor() {}

/** @return Calls reaching a callback that failed admission should have withdrawn. */
extern "C" unsigned llaminarPartialHostStorageCalls() { return storage_calls; }

/**
 * @def LLAMINAR_PARTIAL_RCCL_SYMBOL
 * @brief Define a lookup-only inert stub; no such operation runs in this fixture.
 *
 * Only symbol resolution is exercised. Actual communicator and collective
 * signatures belong to the native library, never these device-free stubs.
 */
#define LLAMINAR_PARTIAL_RCCL_SYMBOL(name) extern "C" int name() { return 0; }
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclGetUniqueId)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommInitRank)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommInitAll)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommDestroy)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommAbort)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommFinalize)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommCount)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommCuDevice)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclCommUserRank)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclGetErrorString)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclAllReduce)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclBroadcast)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclReduce)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclAllGather)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclReduceScatter)
LLAMINAR_PARTIAL_RCCL_SYMBOL(llaminarNcclAllGatherDeviceRows)
LLAMINAR_PARTIAL_RCCL_SYMBOL(llaminarNcclAllReduceDeviceRows)
LLAMINAR_PARTIAL_RCCL_SYMBOL(llaminarNcclReduceScatterDeviceRows)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclSend)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclRecv)
LLAMINAR_PARTIAL_RCCL_SYMBOL(ncclGroupStart)
// ncclGroupEnd is intentionally absent: admission fails after resolving storage.
#undef LLAMINAR_PARTIAL_RCCL_SYMBOL

/**
 * @brief Count an illicit use of a resolved-but-unpublished diagnostic callback.
 * @param comm Unused opaque fixture address; never a real communicator.
 * @param receipt Valid caller-owned record, left unchanged.
 * @return Success so stale callback use becomes a functional test failure.
 */
extern "C" int llaminarRcclHostTransportStorage(
    void* comm, llaminar2::RCCLHostTransportStorage* receipt)
{
    (void)comm;
    (void)receipt;
    ++storage_calls;
    return 0;
}
