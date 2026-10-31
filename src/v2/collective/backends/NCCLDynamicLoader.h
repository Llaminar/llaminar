/**
 * @file NCCLDynamicLoader.h
 * @brief Dynamic loader for NCCL library using dlopen/dlsym
 *
 * This module loads the exact CMake-selected dependency, or an explicit path,
 * at runtime using dlopen with RTLD_LOCAL to isolate
 * NCCL symbols from RCCL. Both libraries export identical symbol names
 * (ncclAllReduce, ncclCommInitRank, etc.), and using RTLD_LOCAL prevents
 * symbol conflicts in the global namespace.
 *
 * Without dynamic loading, whichever library loads first would shadow the
 * other's symbols, causing AMD code to call NVIDIA's NCCL or vice versa.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <cstddef>
#include "../NativeCollectiveRows.h"
#include "NCCLNetworkPolicy.h"
#include <string>

namespace llaminar2
{
    namespace nccl_dynamic
    {
        /**
         * @brief Load the exact configured NCCL dependency with isolated symbols.
         * @param library_path Explicit nonempty library path, or nullptr for the
         *        LLAMINAR_NCCL_LIBRARY/NCCL_LIBRARY path selected by CMake.
         * @return True if the same path is already loaded or all mandatory
         *         entrypoints resolve; false for invalid/different selection or failure.
         * Empty explicit input always rejects. A failed selection never searches
         * another path or bare SONAME. RTLD_NOW resolves symbols immediately;
         * RTLD_LOCAL keeps the addresses separate from RCCL. Repeated valid calls
         * with the identical path retain the dependency; a different path rejects
         * until unload() retires both its selection identity and entrypoints.
         */
        bool load(const char *library_path = nullptr);

        /**
         * @brief Retire the selected library and withdraw all borrowed entrypoints.
         * Resolved callbacks are cleared before dlclose, including after partial
         * load failure. Later wrappers return unavailable instead of calling an
         * address whose library lifetime has ended. The existing mutex protects
         * selection/retirement; native operations must retire before this call.
         */
        void unload();

        /**
         * @brief Check if NCCL is loaded
         */
        bool isLoaded();

        /**
         * @brief Get error message from last failed operation
         */
        const char *getLastError();

        // =========================================================================
        // NCCL Type Definitions (matching nccl.h)
        // =========================================================================

        // Opaque types - we don't need the actual structure, just pointers
        using ncclComm_t = void *;

        // ncclUniqueId is a 128-byte structure
        struct ncclUniqueId
        {
            char internal[128];
        };

        // Result codes
        enum ncclResult_t
        {
            ncclSuccess = 0,
            ncclUnhandledCudaError = 1,
            ncclSystemError = 2,
            ncclInternalError = 3,
            ncclInvalidArgument = 4,
            ncclInvalidUsage = 5,
            ncclRemoteError = 6,
            ncclInProgress = 7,
            ncclNumResults = 8
        };

        // Data types
        enum ncclDataType_t
        {
            ncclInt8 = 0,
            ncclChar = 0,
            ncclUint8 = 1,
            ncclInt32 = 2,
            ncclInt = 2,
            ncclUint32 = 3,
            ncclInt64 = 4,
            ncclUint64 = 5,
            ncclFloat16 = 6,
            ncclHalf = 6,
            ncclFloat32 = 7,
            ncclFloat = 7,
            ncclFloat64 = 8,
            ncclDouble = 8,
            ncclBfloat16 = 9,
            ncclFloat8e4m3 = 10,
            ncclFloat8e5m2 = 11,
            ncclNumTypes = 12
        };

        // Reduction operations (values must match NCCL ABI)
        enum ncclRedOp_t
        {
            ncclSum = 0,
            ncclProd = 1,
            ncclMax = 2,
            ncclMin = 3,
            ncclAvg = 4,
            ncclNumOps = 5
        };

        // =========================================================================
        // NCCL API Function Wrappers
        // =========================================================================

        // Unique ID management
        ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId);

        // Communicator management
        ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank);

        /**
         * @brief Initialize one rank with a frozen topology/economy policy.
         *
         * Network and CTA limits belong to this communicator, without mutation
         * of process-wide NCCL variables. The measured profile binds the exact
         * canonical dependency ABI; native algorithms still adapt to message size.
         *
         * @param comm Receives the initialized communicator.
         * @param nranks Number of ranks in the communicator.
         * @param comm_id Unique communicator identifier shared by every rank.
         * @param rank Rank represented by the calling thread and CUDA device.
         * @param policy Typed network module and measured channel profile.
         * @return NCCL status from communicator construction.
         */
        ncclResult_t ncclCommInitRankWithPolicy(
            ncclComm_t *comm,
            int nranks,
            ncclUniqueId comm_id,
            int rank,
            NCCLCommunicatorPolicy policy);

        ncclResult_t ncclCommDestroy(ncclComm_t comm);
        ncclResult_t ncclCommAbort(ncclComm_t comm);
        ncclResult_t ncclCommCount(const ncclComm_t comm, int *count);
        ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device);
        ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank);

        // Error handling
        const char *ncclGetErrorString(ncclResult_t result);

        // Collective operations
        ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                   void *stream);

        ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, int root, ncclComm_t comm,
                                   void *stream);

        ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op, int root,
                                ncclComm_t comm, void *stream);

        ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                                   ncclDataType_t datatype, ncclComm_t comm, void *stream);

        ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                       void *stream);

        /**
         * @brief Record exact live rows through the canonical native dependency ABI.
         * @param operation Native collective, never an emulated transport.
         * @param send Input bank; only allreduce may alias the output.
         * @param receive Output bank with immutable capacity-sized rank strides.
         * @param rows Borrowed ordered device count or an exact fully-live range.
         * @param datatype Native scalar/wire representation.
         * @param op Native reduction operation (ignored for gather).
         * @param payload_bytes Optional persistent passive device payload counter.
         * @param comm Existing initialized node-local communicator.
         * @param stream Exact non-null CUDA/HIP stream.
         * @return Native enqueue result; a missing live-extent ABI fails closed.
         */
        ncclResult_t nativeRows(
            NativeRowCollective operation, const void *send, void *receive,
            const NativeCollectiveRows &rows, ncclDataType_t datatype, ncclRedOp_t op,
            unsigned long long *payload_bytes, ncclComm_t comm, void *stream);

        // Point-to-point operations
        ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                              int peer, ncclComm_t comm, void *stream);

        ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                              int peer, ncclComm_t comm, void *stream);

        // Group operations
        ncclResult_t ncclGroupStart();
        ncclResult_t ncclGroupEnd();

    } // namespace nccl_dynamic
} // namespace llaminar2
