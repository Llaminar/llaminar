/**
 * @file RCCLDynamicLoader.cpp
 * @brief Implementation of dynamic RCCL loader using dlopen/dlsym
 *
 * Uses dlopen with RTLD_LOCAL to load RCCL into an isolated symbol namespace,
 * preventing conflicts with NCCL which exports identical symbol names.
 * CMake's RCCL_LIBRARY is the sole dependency-selection authority. Missing or
 * incompatible dependencies fail; the loader never substitutes a distribution
 * library or guesses a checkout path inside a relocated container.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "RCCLDynamicLoader.h"
#include "utils/Logger.h"

#include <dlfcn.h>
#include <mutex>
#include <string>

namespace llaminar2
{
    namespace rccl_dynamic
    {
        // =========================================================================
        // Internal State
        // =========================================================================

        namespace
        {
            std::mutex g_mutex;
            void *g_library_handle = nullptr;
            std::string g_last_error;

            // Function pointer types (same signatures as NCCL)
            using ncclGetUniqueId_t = ncclResult_t (*)(ncclUniqueId *);
            using ncclCommInitRank_t = ncclResult_t (*)(ncclComm_t *, int, ncclUniqueId, int);
            using ncclCommInitAll_t = ncclResult_t (*)(ncclComm_t *, int, const int *);
            using ncclCommDestroy_t = ncclResult_t (*)(ncclComm_t);
            using ncclCommAbort_t = ncclResult_t (*)(ncclComm_t);
            using ncclCommFinalize_t = ncclResult_t (*)(ncclComm_t);
            using ncclCommCount_t = ncclResult_t (*)(const ncclComm_t, int *);
            using ncclCommCuDevice_t = ncclResult_t (*)(const ncclComm_t, int *);
            using ncclCommUserRank_t = ncclResult_t (*)(const ncclComm_t, int *);
            using ncclGetErrorString_t = const char *(*)(ncclResult_t);

            using ncclAllReduce_t = ncclResult_t (*)(const void *, void *, size_t,
                                                     ncclDataType_t, ncclRedOp_t, ncclComm_t, void *);
            using ncclBroadcast_t = ncclResult_t (*)(const void *, void *, size_t,
                                                     ncclDataType_t, int, ncclComm_t, void *);
            using ncclReduce_t = ncclResult_t (*)(const void *, void *, size_t,
                                                  ncclDataType_t, ncclRedOp_t, int, ncclComm_t, void *);
            using ncclAllGather_t = ncclResult_t (*)(const void *, void *, size_t,
                                                     ncclDataType_t, ncclComm_t, void *);
            using ncclReduceScatter_t = ncclResult_t (*)(const void *, void *, size_t,
                                                         ncclDataType_t, ncclRedOp_t, ncclComm_t, void *);
            using ncclSend_t = ncclResult_t (*)(const void *, size_t, ncclDataType_t,
                                                int, ncclComm_t, void *);
            using ncclRecv_t = ncclResult_t (*)(void *, size_t, ncclDataType_t,
                                                int, ncclComm_t, void *);
            using ncclGroupStart_t = ncclResult_t (*)();
            using ncclGroupEnd_t = ncclResult_t (*)();

            // The extension retains the native schedule/transport while clipping
            // only useful payload bytes. Resolve it once, never during capture.
            using deviceRowsGather_t = ncclResult_t (*)(const void *, void *,
                const std::int32_t *, size_t, size_t, unsigned long long *, ncclComm_t, void *);
            using deviceRowsReduction_t = ncclResult_t (*)(const void *, void *,
                const std::int32_t *, size_t, size_t, ncclDataType_t, ncclRedOp_t,
                unsigned long long *, ncclComm_t, void *);
            deviceRowsGather_t fp_device_rows_gather = nullptr;
            deviceRowsReduction_t fp_device_rows_reduce = nullptr;
            deviceRowsReduction_t fp_device_rows_scatter = nullptr;

            // Read native descriptor ownership only at a quiescent diagnostic
            // boundary. This is not a live allocator or an inference decision.
            using hostTransportStorage_t = ncclResult_t (*)(ncclComm_t, RCCLHostTransportStorage *);
            hostTransportStorage_t fp_host_transport_storage = nullptr;

            // Function pointers
            ncclGetUniqueId_t fp_ncclGetUniqueId = nullptr;
            ncclCommInitRank_t fp_ncclCommInitRank = nullptr;
            ncclCommInitAll_t fp_ncclCommInitAll = nullptr;
            ncclCommDestroy_t fp_ncclCommDestroy = nullptr;
            ncclCommAbort_t fp_ncclCommAbort = nullptr;       // Optional - may not exist in older RCCL
            ncclCommFinalize_t fp_ncclCommFinalize = nullptr; // Optional - RCCL 2.14+
            ncclCommCount_t fp_ncclCommCount = nullptr;
            ncclCommCuDevice_t fp_ncclCommCuDevice = nullptr;
            ncclCommUserRank_t fp_ncclCommUserRank = nullptr;
            ncclGetErrorString_t fp_ncclGetErrorString = nullptr;
            ncclAllReduce_t fp_ncclAllReduce = nullptr;
            ncclBroadcast_t fp_ncclBroadcast = nullptr;
            ncclReduce_t fp_ncclReduce = nullptr;
            ncclAllGather_t fp_ncclAllGather = nullptr;
            ncclReduceScatter_t fp_ncclReduceScatter = nullptr;
            ncclSend_t fp_ncclSend = nullptr;
            ncclRecv_t fp_ncclRecv = nullptr;
            ncclGroupStart_t fp_ncclGroupStart = nullptr;
            ncclGroupEnd_t fp_ncclGroupEnd = nullptr;

            /**
             * @brief Withdraw the entire resolved table before retiring its DSO.
             *
             * Both failed partial admission and ordinary unload use this one
             * path. A callback resolved before a later missing symbol must not
             * outlive dlclose or masquerade as available diagnostic evidence.
             * The caller already owns g_mutex; this helper never takes a lock.
             */
            void clearResolvedSymbols() noexcept
            {
                fp_ncclGetUniqueId = nullptr;
                fp_ncclCommInitRank = nullptr;
                fp_ncclCommInitAll = nullptr;
                fp_ncclCommDestroy = nullptr;
                fp_ncclCommAbort = nullptr;
                fp_ncclCommFinalize = nullptr;
                fp_ncclCommCount = nullptr;
                fp_ncclCommCuDevice = nullptr;
                fp_ncclCommUserRank = nullptr;
                fp_ncclGetErrorString = nullptr;
                fp_ncclAllReduce = nullptr;
                fp_ncclBroadcast = nullptr;
                fp_ncclReduce = nullptr;
                fp_ncclAllGather = nullptr;
                fp_ncclReduceScatter = nullptr;
                fp_device_rows_gather = nullptr;
                fp_device_rows_reduce = nullptr;
                fp_device_rows_scatter = nullptr;
                fp_host_transport_storage = nullptr;
                fp_ncclSend = nullptr;
                fp_ncclRecv = nullptr;
                fp_ncclGroupStart = nullptr;
                fp_ncclGroupEnd = nullptr;
            }

            // Helper to load a symbol
            template <typename T>
            bool loadSymbol(T &func_ptr, const char *name)
            {
                func_ptr = reinterpret_cast<T>(dlsym(g_library_handle, name));
                if (!func_ptr)
                {
                    g_last_error = std::string("Failed to load symbol '") + name + "': " + dlerror();
                    return false;
                }
                return true;
            }

        } // anonymous namespace

        // =========================================================================
        // Public API Implementation
        // =========================================================================

        /**
         * @brief Resolve the configured RCCL dependency once without substitution.
         * @param library_path Explicit diagnostic dependency, or null to use
         *        CMake's selected RCCL_LIBRARY. An invalid path fails as given.
         * @return True only when the selected DSO and required symbols load.
         */
        bool load(const char *library_path)
        {
            std::lock_guard<std::mutex> lock(g_mutex);

            if (g_library_handle)
            {
                // Already loaded
                return true;
            }

            // A search list can silently select packaged GPU code that differs
            // from the dependency admitted by the build. In particular, moving
            // a checkout into /src must not change which RCCL executes.
            const char *path = library_path;
#ifdef LLAMINAR_RCCL_LIBRARY_PATH
            if (!path)
                path = LLAMINAR_RCCL_LIBRARY_PATH;
#endif
            if (!path || !*path)
            {
                g_last_error = "No RCCL library configured; set RCCL_LIBRARY when configuring the build";
                LOG_ERROR("RCCL Dynamic Loader: " << g_last_error);
                return false;
            }

            // Clear any previous dlerror
            dlerror();

            LOG_DEBUG("RCCL Dynamic Loader: Loading configured dependency '" << path << "'");
            // Resolve symbols immediately and keep them local so CUDA's NCCL
            // and ROCm's RCCL can coexist without sharing function addresses.
            g_library_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (!g_library_handle)
            {
                g_last_error = std::string("Failed to load configured RCCL library '") + path + "': " + dlerror();
                LOG_ERROR("RCCL Dynamic Loader: " << g_last_error);
                return false;
            }
            LOG_INFO("RCCL Dynamic Loader: Successfully loaded '" << path << "'");

            // Load all required symbols
            bool success = true;
            success = success && loadSymbol(fp_ncclGetUniqueId, "ncclGetUniqueId");
            success = success && loadSymbol(fp_ncclCommInitRank, "ncclCommInitRank");
            success = success && loadSymbol(fp_ncclCommInitAll, "ncclCommInitAll");
            success = success && loadSymbol(fp_ncclCommDestroy, "ncclCommDestroy");

            // ncclCommAbort is optional - don't fail if not found
            // It's available in RCCL 2.13+ and provides safer cleanup for unused communicators
            if (!loadSymbol(fp_ncclCommAbort, "ncclCommAbort"))
            {
                LOG_DEBUG("RCCL Dynamic Loader: ncclCommAbort not available (optional, RCCL 2.13+)");
                fp_ncclCommAbort = nullptr;
            }
            // ncclCommFinalize is optional - available in RCCL 2.14+
            if (!loadSymbol(fp_ncclCommFinalize, "ncclCommFinalize"))
            {
                LOG_DEBUG("RCCL Dynamic Loader: ncclCommFinalize not available (optional, RCCL 2.14+)");
                fp_ncclCommFinalize = nullptr;
            }
            success = success && loadSymbol(fp_ncclCommCount, "ncclCommCount");
            success = success && loadSymbol(fp_ncclCommCuDevice, "ncclCommCuDevice");
            success = success && loadSymbol(fp_ncclCommUserRank, "ncclCommUserRank");
            success = success && loadSymbol(fp_ncclGetErrorString, "ncclGetErrorString");
            success = success && loadSymbol(fp_ncclAllReduce, "ncclAllReduce");
            success = success && loadSymbol(fp_ncclBroadcast, "ncclBroadcast");
            success = success && loadSymbol(fp_ncclReduce, "ncclReduce");
            success = success && loadSymbol(fp_ncclAllGather, "ncclAllGather");
            success = success && loadSymbol(fp_ncclReduceScatter, "ncclReduceScatter");
            success = success && loadSymbol(fp_device_rows_gather, "llaminarNcclAllGatherDeviceRows");
            success = success && loadSymbol(fp_device_rows_reduce, "llaminarNcclAllReduceDeviceRows");
            success = success && loadSymbol(fp_device_rows_scatter, "llaminarNcclReduceScatterDeviceRows");
            success = success && loadSymbol(fp_host_transport_storage, "llaminarRcclHostTransportStorage");
            success = success && loadSymbol(fp_ncclSend, "ncclSend");
            success = success && loadSymbol(fp_ncclRecv, "ncclRecv");
            success = success && loadSymbol(fp_ncclGroupStart, "ncclGroupStart");
            success = success && loadSymbol(fp_ncclGroupEnd, "ncclGroupEnd");

            if (!success)
            {
                LOG_ERROR("RCCL Dynamic Loader: " << g_last_error);
                clearResolvedSymbols();
                dlclose(g_library_handle);
                g_library_handle = nullptr;
                return false;
            }

            LOG_INFO("RCCL Dynamic Loader: All required symbols loaded successfully");
            return true;
        }

        void unload()
        {
            std::lock_guard<std::mutex> lock(g_mutex);

            if (g_library_handle)
            {
                clearResolvedSymbols();
                dlclose(g_library_handle);
                g_library_handle = nullptr;
                LOG_DEBUG("RCCL Dynamic Loader: Library unloaded");
            }
        }

        bool isLoaded()
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            return g_library_handle != nullptr;
        }

        const char *getLastError()
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            return g_last_error.c_str();
        }

        // =========================================================================
        // RCCL API Wrapper Implementations
        // =========================================================================

        ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId)
        {
            if (!fp_ncclGetUniqueId)
            {
                LOG_ERROR("RCCL not loaded: ncclGetUniqueId");
                return ncclInternalError;
            }
            return fp_ncclGetUniqueId(uniqueId);
        }

        ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank)
        {
            if (!fp_ncclCommInitRank)
            {
                LOG_ERROR("RCCL not loaded: ncclCommInitRank");
                return ncclInternalError;
            }
            return fp_ncclCommInitRank(comm, nranks, commId, rank);
        }

        ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist)
        {
            if (!fp_ncclCommInitAll)
            {
                LOG_ERROR("RCCL not loaded: ncclCommInitAll");
                return ncclInternalError;
            }
            return fp_ncclCommInitAll(comms, ndev, devlist);
        }

        ncclResult_t ncclCommDestroy(ncclComm_t comm)
        {
            if (!fp_ncclCommDestroy)
            {
                LOG_ERROR("RCCL not loaded: ncclCommDestroy");
                return ncclInternalError;
            }
            return fp_ncclCommDestroy(comm);
        }

        ncclResult_t ncclCommAbort(ncclComm_t comm)
        {
            if (!fp_ncclCommAbort)
            {
                LOG_WARN("RCCL: ncclCommAbort not available, falling back to ncclCommDestroy");
                return ncclCommDestroy(comm);
            }
            return fp_ncclCommAbort(comm);
        }

        bool isCommAbortAvailable()
        {
            return fp_ncclCommAbort != nullptr;
        }

        ncclResult_t ncclCommFinalize(ncclComm_t comm)
        {
            if (!fp_ncclCommFinalize)
            {
                // Not available - caller should fall back to direct destroy
                return ncclInternalError;
            }
            return fp_ncclCommFinalize(comm);
        }

        bool isCommFinalizeAvailable()
        {
            return fp_ncclCommFinalize != nullptr;
        }

        ncclResult_t ncclCommCount(const ncclComm_t comm, int *count)
        {
            if (!fp_ncclCommCount)
            {
                LOG_ERROR("RCCL not loaded: ncclCommCount");
                return ncclInternalError;
            }
            return fp_ncclCommCount(comm, count);
        }

        ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device)
        {
            if (!fp_ncclCommCuDevice)
            {
                LOG_ERROR("RCCL not loaded: ncclCommCuDevice");
                return ncclInternalError;
            }
            return fp_ncclCommCuDevice(comm, device);
        }

        ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank)
        {
            if (!fp_ncclCommUserRank)
            {
                LOG_ERROR("RCCL not loaded: ncclCommUserRank");
                return ncclInternalError;
            }
            return fp_ncclCommUserRank(comm, rank);
        }

        /** @copydoc hostTransportStorage(ncclComm_t,RCCLHostTransportStorage&) */
        ncclResult_t hostTransportStorage(ncclComm_t comm, RCCLHostTransportStorage &receipt)
        {
            if (!comm || !receipt.valid())
                return ncclInvalidArgument;
            if (!fp_host_transport_storage)
                return ncclInvalidUsage;
            const auto result = fp_host_transport_storage(comm, &receipt);
            return result == ncclSuccess && !receipt.valid() ? ncclInvalidUsage : result;
        }

        const char *ncclGetErrorString(ncclResult_t result)
        {
            if (!fp_ncclGetErrorString)
            {
                return "RCCL not loaded";
            }
            return fp_ncclGetErrorString(result);
        }

        ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                   void *stream)
        {
            if (!fp_ncclAllReduce)
            {
                LOG_ERROR("RCCL not loaded: ncclAllReduce");
                return ncclInternalError;
            }
            return fp_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
        }

        ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                                   ncclDataType_t datatype, int root, ncclComm_t comm,
                                   void *stream)
        {
            if (!fp_ncclBroadcast)
            {
                LOG_ERROR("RCCL not loaded: ncclBroadcast");
                return ncclInternalError;
            }
            return fp_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
        }

        ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op, int root,
                                ncclComm_t comm, void *stream)
        {
            if (!fp_ncclReduce)
            {
                LOG_ERROR("RCCL not loaded: ncclReduce");
                return ncclInternalError;
            }
            return fp_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
        }

        ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                                   ncclDataType_t datatype, ncclComm_t comm, void *stream)
        {
            if (!fp_ncclAllGather)
            {
                LOG_ERROR("RCCL not loaded: ncclAllGather");
                return ncclInternalError;
            }
            return fp_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        }

        /** @copydoc nativeRows(NativeRowCollective,const void*,void*,const NativeCollectiveRows&,ncclDataType_t,ncclRedOp_t,unsigned long long*,ncclComm_t,void*) */
        ncclResult_t nativeRows(NativeRowCollective operation, const void *send, void *receive,
            const NativeCollectiveRows &rows, ncclDataType_t datatype, ncclRedOp_t op,
            unsigned long long *payload_bytes, ncclComm_t comm, void *stream)
        {
            if (!send || !receive || !comm || !stream) return ncclInvalidArgument;
            if (payload_bytes && rows.payloadReceipt() && payload_bytes != rows.payloadReceipt())
                return ncclInvalidArgument;
            if (rows.payloadReceipt()) payload_bytes = rows.payloadReceipt();
            // Fully-live rows retain exactly the ordinary native API. A passive
            // receipt is available only for the device-counted implementation.
            if (!rows.rows().countOwner())
            {
                if (payload_bytes) return ncclInvalidArgument;
                switch (operation)
                {
                case NativeRowCollective::AllGather:
                    return ncclAllGather(send, receive, rows.bankElements(), datatype, comm, stream);
                case NativeRowCollective::AllReduce:
                    return ncclAllReduce(send, receive, rows.bankElements(), datatype, op, comm, stream);
                case NativeRowCollective::ReduceScatter:
                    return ncclReduceScatter(send, receive, rows.bankElements(), datatype, op, comm, stream);
                default: return ncclInvalidArgument;
                }
            }
            const auto *count = rows.rows().countOwner();
            const auto capacity = static_cast<size_t>(rows.rows().capacity());
            const auto width = rows.elementsPerRow();
            if (operation == NativeRowCollective::AllGather)
            {
                size_t bytes = 0;
                switch (datatype)
                {
                case ncclInt8: bytes = 1; break;
                case ncclFloat16: case ncclBfloat16: bytes = 2; break;
                case ncclInt32: case ncclFloat32: bytes = 4; break;
                default: return ncclInvalidArgument;
                }
                if (!fp_device_rows_gather) return ncclInvalidUsage;
                if (!rows.byteGeometryValid(bytes, 1)) return ncclInvalidArgument;
                return fp_device_rows_gather(send, receive, count, capacity, width * bytes,
                    payload_bytes, comm, stream);
            }
            const auto function = operation == NativeRowCollective::AllReduce ? fp_device_rows_reduce :
                operation == NativeRowCollective::ReduceScatter ? fp_device_rows_scatter : nullptr;
            if (!function) return ncclInvalidUsage;
            return function(send, receive, count, capacity, width, datatype, op, payload_bytes, comm, stream);
        }

        ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                       void *stream)
        {
            if (!fp_ncclReduceScatter)
            {
                LOG_ERROR("RCCL not loaded: ncclReduceScatter");
                return ncclInternalError;
            }
            return fp_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
        }

        ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                              int peer, ncclComm_t comm, void *stream)
        {
            if (!fp_ncclSend)
            {
                LOG_ERROR("RCCL not loaded: ncclSend");
                return ncclInternalError;
            }
            return fp_ncclSend(sendbuff, count, datatype, peer, comm, stream);
        }

        ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                              int peer, ncclComm_t comm, void *stream)
        {
            if (!fp_ncclRecv)
            {
                LOG_ERROR("RCCL not loaded: ncclRecv");
                return ncclInternalError;
            }
            return fp_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
        }

        ncclResult_t ncclGroupStart()
        {
            if (!fp_ncclGroupStart)
            {
                LOG_ERROR("RCCL not loaded: ncclGroupStart");
                return ncclInternalError;
            }
            return fp_ncclGroupStart();
        }

        ncclResult_t ncclGroupEnd()
        {
            if (!fp_ncclGroupEnd)
            {
                LOG_ERROR("RCCL not loaded: ncclGroupEnd");
                return ncclInternalError;
            }
            return fp_ncclGroupEnd();
        }

    } // namespace rccl_dynamic
} // namespace llaminar2
