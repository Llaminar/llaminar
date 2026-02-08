/**
 * @file RCCLDynamicLoader.h
 * @brief Dynamic loader for RCCL library using dlopen/dlsym
 *
 * This module loads RCCL at runtime using dlopen with RTLD_LOCAL to isolate
 * RCCL symbols from NCCL. Both libraries export identical symbol names
 * (ncclAllReduce, ncclCommInitRank, etc.), and using RTLD_LOCAL prevents
 * symbol conflicts in the global namespace.
 *
 * Without dynamic loading, whichever library loads first would shadow the
 * other's symbols, causing NVIDIA code to call AMD's RCCL or vice versa.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <cstddef>
#include <string>

namespace llaminar2
{
    namespace rccl_dynamic
    {
        /**
         * @brief Load RCCL library dynamically with isolated symbols
         * @param library_path Optional path to librccl.so (nullptr for default search)
         * @return true if loaded successfully
         *
         * The library is loaded with RTLD_NOW | RTLD_LOCAL to:
         * - RTLD_NOW: Resolve all symbols immediately (fail fast on missing symbols)
         * - RTLD_LOCAL: Keep symbols in local namespace (avoid conflicts with NCCL)
         */
        bool load(const char *library_path = nullptr);

        /**
         * @brief Unload RCCL library
         */
        void unload();

        /**
         * @brief Check if RCCL is loaded
         */
        bool isLoaded();

        /**
         * @brief Get error message from last failed operation
         */
        const char *getLastError();

        // =========================================================================
        // RCCL Type Definitions (matching rccl.h / nccl.h API)
        // =========================================================================

        // Opaque types - we don't need the actual structure, just pointers
        using ncclComm_t = void *;

        // ncclUniqueId is a 128-byte structure (same as NCCL)
        struct ncclUniqueId
        {
            char internal[128];
        };

        // Result codes (same as NCCL)
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

        // Data types (same as NCCL)
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
            ncclNumTypes = 10
        };

        // Reduction operations (same as NCCL)
        enum ncclRedOp_t
        {
            ncclSum = 0,
            ncclProd = 1,
            ncclMin = 2,
            ncclMax = 3,
            ncclAvg = 4,
            ncclNumOps = 5
        };

        // =========================================================================
        // RCCL API Function Wrappers
        // =========================================================================

        // Unique ID management
        ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId);

        // Communicator management
        ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank);
        ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist);
        ncclResult_t ncclCommDestroy(ncclComm_t comm);

        /**
         * @brief Abort and destroy a communicator without coordinated shutdown
         *
         * Unlike ncclCommDestroy, ncclCommAbort does not require coordination
         * with other ranks and safely handles partially-initialized or unused
         * communicators. This avoids null-pointer dereferences in the ROCm CLR
         * runtime when destroying communicators that never performed operations.
         *
         * @param comm Communicator to abort (may be unused/partially initialized)
         * @return ncclSuccess on success
         * @note Optional symbol - may not be available in older RCCL versions.
         *       Use isCommAbortAvailable() to check before calling.
         */
        ncclResult_t ncclCommAbort(ncclComm_t comm);

        /**
         * @brief Check if ncclCommAbort is available in the loaded RCCL library
         */
        bool isCommAbortAvailable();

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

        // Point-to-point operations
        ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                              int peer, ncclComm_t comm, void *stream);

        ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                              int peer, ncclComm_t comm, void *stream);

        // Group operations
        ncclResult_t ncclGroupStart();
        ncclResult_t ncclGroupEnd();

    } // namespace rccl_dynamic
} // namespace llaminar2
