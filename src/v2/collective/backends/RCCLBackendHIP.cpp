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
