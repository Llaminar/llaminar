/**
 * @file NCCLDynamicLoader.h
 * @brief Dynamic loader for NCCL library using dlopen/dlsym
 *
 * This module loads NCCL at runtime using dlopen with RTLD_LOCAL to isolate
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
#include <string>

namespace llaminar2
{
    namespace nccl_dynamic
    {
        /**
         * @brief Load NCCL library dynamically with isolated symbols
         * @param library_path Exact explicit library, or nullptr for the
         *        installed libllaminar_nccl.so.2 capture-reentry implementation.
         *        A failed load never selects a different library.
         * @return true if loaded successfully
         *
         * The library is loaded with RTLD_NOW | RTLD_LOCAL to:
         * - RTLD_NOW: Resolve all symbols immediately (fail fast on missing symbols)
         * - RTLD_LOCAL: Keep symbols in local namespace (avoid conflicts with RCCL)
         */
        bool load(const char *library_path = nullptr);

        /**
         * @brief Unload NCCL library
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
         * @brief Initialize one rank with an optional per-communicator network module.
         *
         * A null or empty module name uses ncclCommInitRank and preserves NCCL's
         * automatic network selection. A non-empty name uses
         * ncclCommInitRankConfig and writes only ncclConfig_t::netName, leaving
         * every other configuration field at NCCL's documented default.
         *
         * @param comm Receives the initialized communicator.
         * @param nranks Number of ranks in the communicator.
         * @param comm_id Unique communicator identifier shared by every rank.
         * @param rank Rank represented by the calling thread and CUDA device.
         * @param network_module Exact NCCL network module name, or null for auto.
         * @return NCCL status from communicator construction.
         */
        ncclResult_t ncclCommInitRankWithNetwork(
            ncclComm_t *comm,
            int nranks,
            ncclUniqueId comm_id,
            int rank,
            const char *network_module);

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
