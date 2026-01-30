/**
 * @file RCCLBackendHIP.cpp
 * @brief HIP and RCCL-specific helper functions for RCCLBackend
 *
 * Isolated HIP runtime and RCCL API calls in separate compilation unit.
 * Uses dynamic loader for RCCL to avoid symbol conflicts with NCCL.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "RCCLDynamicLoader.h"
#include "utils/Logger.h"

#include <hip/hip_runtime.h>
#include <string>
#include <cstring>
#include <cstdlib> // for setenv

// Use the dynamically loaded RCCL types and functions
namespace rccl = llaminar2::rccl_dynamic;

namespace llaminar2
{
    namespace rccl_backend_detail
    {

        // =========================================================================
        // Device Management
        // =========================================================================

        bool hipSetDeviceOrdinal(int device_ordinal)
        {
            hipError_t err = hipSetDevice(device_ordinal);
            return (err == hipSuccess);
        }

        bool hipGetDeviceCountWrapper(int *count)
        {
            hipError_t err = hipGetDeviceCount(count);
            return (err == hipSuccess);
        }

        bool hipCheckP2PAvailable(const std::vector<int> &device_ordinals)
        {
            // Check if P2P is available between ALL device pairs
            // Returns true only if bidirectional P2P works for every pair
            const int n = static_cast<int>(device_ordinals.size());
            if (n < 2)
            {
                return true; // Single device, P2P not needed
            }

            for (int i = 0; i < n; ++i)
            {
                int from_dev = device_ordinals[i];
                for (int j = 0; j < n; ++j)
                {
                    if (i == j)
                        continue;

                    int to_dev = device_ordinals[j];
                    int can_access = 0;
                    hipError_t err = hipDeviceCanAccessPeer(&can_access, from_dev, to_dev);

                    if (err != hipSuccess || !can_access)
                    {
                        LOG_DEBUG("[P2P Check] P2P NOT available: GPU " << from_dev << " -> GPU " << to_dev
                                                                        << " (err=" << (err == hipSuccess ? "ok" : hipGetErrorString(err))
                                                                        << ", can_access=" << can_access << ")");
                        return false;
                    }
                }
            }

            LOG_INFO("[P2P Check] P2P available between all " << n << " devices");
            return true;
        }

        bool hipEnablePeerAccessForDevices(const std::vector<int> &device_ordinals, std::string &error_out)
        {
            // Enable bidirectional peer access between all devices in the list
            // This must be called BEFORE RCCL initialization to ensure proper memory access
            const int n = static_cast<int>(device_ordinals.size());
            if (n < 2)
            {
                return true; // Nothing to do for single device
            }

            LOG_INFO("[PeerAccess] Enabling peer access for " << n << " devices: ");
            for (int i = 0; i < n; ++i)
            {
                LOG_INFO("  device_ordinals[" << i << "] = GPU " << device_ordinals[i]);
            }

            for (int i = 0; i < n; ++i)
            {
                int from_dev = device_ordinals[i];
                hipError_t err = hipSetDevice(from_dev);
                if (err != hipSuccess)
                {
                    error_out = "hipSetDevice(" + std::to_string(from_dev) + ") failed: " + hipGetErrorString(err);
                    LOG_ERROR(error_out);
                    return false;
                }

                for (int j = 0; j < n; ++j)
                {
                    if (i == j)
                        continue;

                    int to_dev = device_ordinals[j];

                    // Check if peer access is possible
                    int can_access = 0;
                    err = hipDeviceCanAccessPeer(&can_access, from_dev, to_dev);
                    if (err != hipSuccess)
                    {
                        LOG_WARN("[PeerAccess] hipDeviceCanAccessPeer(from=" << from_dev << ", to=" << to_dev
                                                                             << ") failed: " << hipGetErrorString(err));
                        continue;
                    }

                    if (!can_access)
                    {
                        LOG_WARN("[PeerAccess] P2P not available from GPU " << from_dev << " to GPU " << to_dev);
                        continue;
                    }

                    // Enable peer access
                    err = hipDeviceEnablePeerAccess(to_dev, 0);
                    if (err == hipSuccess)
                    {
                        LOG_INFO("[PeerAccess] Enabled P2P access: GPU " << from_dev << " -> GPU " << to_dev);
                    }
                    else if (err == hipErrorPeerAccessAlreadyEnabled)
                    {
                        LOG_INFO("[PeerAccess] P2P already enabled: GPU " << from_dev << " -> GPU " << to_dev);
                    }
                    else
                    {
                        LOG_WARN("[PeerAccess] hipDeviceEnablePeerAccess(from=" << from_dev << ", to=" << to_dev
                                                                                << ") failed: " << hipGetErrorString(err));
                    }
                }
            }

            return true;
        }

        // =========================================================================
        // Stream Management
        // =========================================================================

        bool hipCreateStream(void **stream_ptr)
        {
            hipStream_t stream;
            hipError_t err = hipStreamCreate(&stream);
            if (err == hipSuccess)
            {
                *stream_ptr = static_cast<void *>(stream);
                return true;
            }
            *stream_ptr = nullptr;
            return false;
        }

        bool hipDestroyStream(void *stream)
        {
            if (stream)
            {
                hipError_t err = hipStreamDestroy(static_cast<hipStream_t>(stream));
                return (err == hipSuccess);
            }
            return true;
        }

        bool hipSynchronizeStream(void *stream)
        {
            if (!stream)
            {
                return false;
            }
            hipError_t err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
            return (err == hipSuccess);
        }

        // =========================================================================
        // Error Handling
        // =========================================================================

        std::string hipGetLastErrorString()
        {
            hipError_t err = hipGetLastError();
            return std::string(hipGetErrorString(err));
        }

        std::string hipErrorToString(int error_code)
        {
            return std::string(hipGetErrorString(static_cast<hipError_t>(error_code)));
        }

        // =========================================================================
        // RCCL Pre-Initialization (P2P Check and Environment Setup)
        // =========================================================================

        /**
         * @brief Pre-initialize RCCL by checking P2P and setting environment variables.
         *
         * This MUST be called before any RCCL function that might load the library.
         * It checks P2P availability between devices and sets NCCL_* environment
         * variables BEFORE RCCL is loaded (RCCL reads env vars at load time).
         *
         * @param device_ordinals List of GPU device ordinals to use
         * @param p2p_available_out Set to true if P2P is available between all devices
         * @return true if pre-initialization succeeded
         */
        bool rcclPreInitialize(const std::vector<int> &device_ordinals, bool &p2p_available_out)
        {
            // Check P2P availability BEFORE loading RCCL
            p2p_available_out = hipCheckP2PAvailable(device_ordinals);

            if (!p2p_available_out)
            {
                // P2P not available - set environment variables to force host staging
                // These MUST be set before RCCL is loaded (it reads env vars at load time)
                LOG_WARN("[RCCLBackend] P2P not available between all devices - forcing host staging mode");
                LOG_WARN("[RCCLBackend] Setting NCCL_P2P_DISABLE=1, NCCL_P2P_LEVEL=LOC, NCCL_SHM_DISABLE=1");
                setenv("NCCL_P2P_DISABLE", "1", 1);   // Disable P2P transfers
                setenv("NCCL_P2P_LEVEL", "LOC", 1);   // Only local device P2P (effectively none)
                setenv("NCCL_SHM_DISABLE", "1", 1);   // Disable shared memory transport
                setenv("NCCL_NET_GDR_LEVEL", "0", 1); // Disable GPU Direct RDMA
            }
            else
            {
                LOG_INFO("[RCCLBackend] P2P available between all devices - using direct GPU transfers");
            }

            // Now load RCCL (it will see the environment variables we just set)
            if (!rccl::isLoaded() && !rccl::load())
            {
                LOG_ERROR("[RCCLBackend] Failed to load RCCL library: " << rccl::getLastError());
                return false;
            }

            // If P2P is available, enable peer access for optimal performance
            if (p2p_available_out && device_ordinals.size() >= 2)
            {
                std::string peer_error;
                if (!hipEnablePeerAccessForDevices(device_ordinals, peer_error))
                {
                    LOG_WARN("[RCCLBackend] Failed to enable peer access: " << peer_error << " (continuing anyway)");
                }
            }

            return true;
        }

        // =========================================================================
        // RCCL Unique ID Management
        // =========================================================================

        // Size of ncclUniqueId for serialization
        size_t rcclUniqueIdSize()
        {
            return sizeof(rccl::ncclUniqueId);
        }

        bool rcclGetUniqueIdWrapper(void *id_out)
        {
            // Ensure RCCL is loaded
            if (!rccl::isLoaded() && !rccl::load())
            {
                return false;
            }
            rccl::ncclUniqueId *id = static_cast<rccl::ncclUniqueId *>(id_out);
            rccl::ncclResult_t r = rccl::ncclGetUniqueId(id);
            return (r == rccl::ncclSuccess);
        }

        // =========================================================================
        // RCCL Communicator Management
        // =========================================================================

        bool rcclCommInitRankWrapper(void **comm_out, int nranks, void *unique_id, int rank, std::string &error_out)
        {
            // Ensure RCCL is loaded
            if (!rccl::isLoaded() && !rccl::load())
            {
                error_out = rccl::getLastError();
                *comm_out = nullptr;
                return false;
            }
            rccl::ncclComm_t comm;
            rccl::ncclResult_t r = rccl::ncclCommInitRank(&comm, nranks, *static_cast<rccl::ncclUniqueId *>(unique_id), rank);
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                *comm_out = nullptr;
                return false;
            }
            *comm_out = static_cast<void *>(comm);
            return true;
        }

        bool rcclCommInitAllWrapper(void **comms_out, int ndevs, const int *devlist, std::string &error_out)
        {
            // Ensure RCCL is loaded
            if (!rccl::isLoaded() && !rccl::load())
            {
                error_out = rccl::getLastError();
                *comms_out = nullptr;
                return false;
            }
            rccl::ncclComm_t *comms = new rccl::ncclComm_t[ndevs];
            rccl::ncclResult_t r = rccl::ncclCommInitAll(comms, ndevs, devlist);
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                delete[] comms;
                *comms_out = nullptr;
                return false;
            }
            // For single device, just return the first comm
            *comms_out = static_cast<void *>(comms[0]);
            delete[] comms;
            return true;
        }

        void rcclCommDestroyWrapper(void *comm)
        {
            if (comm && rccl::isLoaded())
            {
                rccl::ncclCommDestroy(static_cast<rccl::ncclComm_t>(comm));
            }
        }

        // =========================================================================
        // RCCL Data Type Conversion
        // =========================================================================

        rccl::ncclDataType_t toRcclDataType(int dtype_int)
        {
            switch (dtype_int)
            {
            case 0: // FLOAT32
                return rccl::ncclFloat;
            case 1: // FLOAT16
                return rccl::ncclHalf;
            case 2: // BFLOAT16
                return rccl::ncclBfloat16;
            case 3: // INT32
                return rccl::ncclInt32;
            case 4: // INT8
                return rccl::ncclInt8;
            default:
                return rccl::ncclFloat;
            }
        }

        rccl::ncclRedOp_t toRcclRedOp(int op_int)
        {
            switch (op_int)
            {
            case 0: // SUM
                return rccl::ncclSum;
            case 1: // PROD
                return rccl::ncclProd;
            case 2: // MIN
                return rccl::ncclMin;
            case 3: // MAX
                return rccl::ncclMax;
            default:
                return rccl::ncclSum;
            }
        }

        // =========================================================================
        // RCCL Collective Operations
        // =========================================================================

        bool rcclAllReduceWrapper(void *sendbuff, void *recvbuff, size_t count,
                                  int dtype_int, int op_int, void *comm, void *stream,
                                  std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclAllReduce(sendbuff, recvbuff, count,
                                                       toRcclDataType(dtype_int), toRcclRedOp(op_int),
                                                       static_cast<rccl::ncclComm_t>(comm),
                                                       static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool rcclAllGatherWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                  int dtype_int, void *comm, void *stream,
                                  std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclAllGather(sendbuff, recvbuff, sendcount,
                                                       toRcclDataType(dtype_int),
                                                       static_cast<rccl::ncclComm_t>(comm),
                                                       static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool rcclBroadcastWrapper(void *buff, size_t count, int dtype_int, int root,
                                  void *comm, void *stream, std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclBroadcast(buff, buff, count, toRcclDataType(dtype_int), root,
                                                       static_cast<rccl::ncclComm_t>(comm),
                                                       static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool rcclReduceScatterWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                      int dtype_int, int op_int, void *comm, void *stream,
                                      std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                                           toRcclDataType(dtype_int), toRcclRedOp(op_int),
                                                           static_cast<rccl::ncclComm_t>(comm),
                                                           static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // =========================================================================
        // RCCL Group Operations
        // =========================================================================

        bool rcclGroupStartWrapper(std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclGroupStart();
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool rcclGroupEndWrapper(std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclGroupEnd();
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool rcclReduceInGroupWrapper(const void *sendbuff, void *recvbuff, size_t count,
                                      int dtype_int, int op_int, int root,
                                      void *comm, void *stream, std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclReduce(sendbuff, recvbuff, count,
                                                    toRcclDataType(dtype_int), toRcclRedOp(op_int),
                                                    root,
                                                    static_cast<rccl::ncclComm_t>(comm),
                                                    static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool rcclReduceScatterInGroupWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                             int dtype_int, int op_int, void *comm, void *stream,
                                             std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                                           toRcclDataType(dtype_int), toRcclRedOp(op_int),
                                                           static_cast<rccl::ncclComm_t>(comm),
                                                           static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // =========================================================================
        // Point-to-Point Operations
        // =========================================================================

        bool rcclSendWrapper(const void *sendbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclSend(sendbuff, count, toRcclDataType(dtype_int), peer,
                                                  static_cast<rccl::ncclComm_t>(comm),
                                                  static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool rcclRecvWrapper(void *recvbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out)
        {
            rccl::ncclResult_t r = rccl::ncclRecv(recvbuff, count, toRcclDataType(dtype_int), peer,
                                                  static_cast<rccl::ncclComm_t>(comm),
                                                  static_cast<hipStream_t>(stream));
            if (r != rccl::ncclSuccess)
            {
                error_out = rccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

    } // namespace rccl_backend_detail
} // namespace llaminar2
