/**
 * @file NCCLBackendCUDA.cu
 * @brief CUDA and NCCL-specific helper functions for NCCLBackend
 *
 * Isolated CUDA runtime and NCCL API calls in separate compilation unit to avoid
 * conflicts with HIP headers when building with both CUDA and ROCm support.
 *
 * IMPORTANT: Uses dynamic loader (dlopen/dlsym) for NCCL to avoid symbol conflicts
 * with RCCL, which exports identical symbol names. Both libraries can now coexist
 * in the same process with isolated symbol namespaces.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <cuda_runtime.h>
#include "NCCLDynamicLoader.h"
#include <string>
#include <cstring>

// Use the dynamically loaded NCCL types and functions
namespace nccl = llaminar2::nccl_dynamic;

namespace llaminar2
{
    namespace nccl_backend_detail
    {

        // =========================================================================
        // Device Management
        // =========================================================================

        bool cudaSetDeviceOrdinal(int device_ordinal)
        {
            cudaError_t err = cudaSetDevice(device_ordinal);
            return (err == cudaSuccess);
        }

        bool cudaGetDeviceCountWrapper(int *count)
        {
            cudaError_t err = cudaGetDeviceCount(count);
            return (err == cudaSuccess);
        }

        // =========================================================================
        // Stream Management
        // =========================================================================

        bool cudaCreateStream(void **stream_ptr)
        {
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err == cudaSuccess)
            {
                *stream_ptr = static_cast<void *>(stream);
                return true;
            }
            *stream_ptr = nullptr;
            return false;
        }

        bool cudaDestroyStream(void *stream)
        {
            if (stream)
            {
                cudaError_t err = cudaStreamDestroy(static_cast<cudaStream_t>(stream));
                return (err == cudaSuccess);
            }
            return true;
        }

        bool cudaSynchronizeStream(void *stream)
        {
            if (!stream)
            {
                return false;
            }
            cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            return (err == cudaSuccess);
        }

        // =========================================================================
        // Error Handling
        // =========================================================================

        std::string cudaGetLastErrorString()
        {
            cudaError_t err = cudaGetLastError();
            return std::string(cudaGetErrorString(err));
        }

        // =========================================================================
        // NCCL Unique ID Management
        // =========================================================================

        // Size of ncclUniqueId for serialization
        size_t ncclUniqueIdSize()
        {
            return sizeof(nccl::ncclUniqueId);
        }

        bool ncclGetUniqueIdWrapper(void *id_out)
        {
            // Ensure NCCL is loaded
            if (!nccl::isLoaded() && !nccl::load())
            {
                return false;
            }
            nccl::ncclUniqueId *id = static_cast<nccl::ncclUniqueId *>(id_out);
            nccl::ncclResult_t r = nccl::ncclGetUniqueId(id);
            return (r == nccl::ncclSuccess);
        }

        // =========================================================================
        // NCCL Communicator Management
        // =========================================================================

        bool ncclCommInitRankWrapper(void **comm_out, int nranks, void *unique_id, int rank, std::string &error_out)
        {
            // Ensure NCCL is loaded
            if (!nccl::isLoaded() && !nccl::load())
            {
                error_out = nccl::getLastError();
                *comm_out = nullptr;
                return false;
            }
            nccl::ncclComm_t comm;
            nccl::ncclResult_t r = nccl::ncclCommInitRank(&comm, nranks, *static_cast<nccl::ncclUniqueId *>(unique_id), rank);
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                *comm_out = nullptr;
                return false;
            }
            *comm_out = static_cast<void *>(comm);
            return true;
        }

        bool ncclCommInitAllWrapper(void **comms_out, int ndevs, const int *devlist, std::string &error_out)
        {
            // Ensure NCCL is loaded
            if (!nccl::isLoaded() && !nccl::load())
            {
                error_out = nccl::getLastError();
                *comms_out = nullptr;
                return false;
            }
            nccl::ncclComm_t *comms = new nccl::ncclComm_t[ndevs];
            nccl::ncclResult_t r = nccl::ncclCommInitAll(comms, ndevs, devlist);
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                delete[] comms;
                *comms_out = nullptr;
                return false;
            }
            // For single device, just return the first comm
            *comms_out = static_cast<void *>(comms[0]);
            delete[] comms;
            return true;
        }

        void ncclCommDestroyWrapper(void *comm)
        {
            if (comm && nccl::isLoaded())
            {
                nccl::ncclCommDestroy(static_cast<nccl::ncclComm_t>(comm));
            }
        }

        // =========================================================================
        // NCCL Data Type Conversion
        // =========================================================================

        // Internal conversion from our enum values to NCCL types
        // We use integers to avoid exposing NCCL types in the header
        nccl::ncclDataType_t toNcclDataType(int dtype_int)
        {
            // These values must match CollectiveDataType enum
            switch (dtype_int)
            {
            case 0: // FLOAT32
                return nccl::ncclFloat;
            case 1: // FLOAT16
                return nccl::ncclHalf;
            case 2: // BFLOAT16
                return nccl::ncclBfloat16;
            case 3: // INT32
                return nccl::ncclInt32;
            case 4: // INT8
                return nccl::ncclInt8;
            default:
                return nccl::ncclFloat;
            }
        }

        nccl::ncclRedOp_t toNcclRedOp(int op_int)
        {
            // These values must match CollectiveOp enum
            switch (op_int)
            {
            case 0: // SUM
                return nccl::ncclSum;
            case 1: // PROD
                return nccl::ncclProd;
            case 2: // MIN
                return nccl::ncclMin;
            case 3: // MAX
                return nccl::ncclMax;
            default:
                return nccl::ncclSum;
            }
        }

        // =========================================================================
        // NCCL Collective Operations
        // =========================================================================

        bool ncclAllReduceWrapper(void *sendbuff, void *recvbuff, size_t count,
                                  int dtype_int, int op_int, void *comm, void *stream,
                                  std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclAllReduce(sendbuff, recvbuff, count,
                                                       toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                                       static_cast<nccl::ncclComm_t>(comm),
                                                       static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclAllGatherWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                  int dtype_int, void *comm, void *stream,
                                  std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclAllGather(sendbuff, recvbuff, sendcount,
                                                       toNcclDataType(dtype_int),
                                                       static_cast<nccl::ncclComm_t>(comm),
                                                       static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclBroadcastWrapper(void *buff, size_t count, int dtype_int, int root,
                                  void *comm, void *stream, std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclBroadcast(buff, buff, count, toNcclDataType(dtype_int), root,
                                                       static_cast<nccl::ncclComm_t>(comm),
                                                       static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclReduceScatterWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                      int dtype_int, int op_int, void *comm, void *stream,
                                      std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                                           toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                                           static_cast<nccl::ncclComm_t>(comm),
                                                           static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // =========================================================================
        // NCCL Group Operations (for multi-GPU single process)
        // =========================================================================

        bool ncclGroupStartWrapper(std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclGroupStart();
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclGroupEndWrapper(std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclGroupEnd();
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // Batched operations within a group
        bool ncclAllReduceInGroupWrapper(void *sendbuff, void *recvbuff, size_t count,
                                         int dtype_int, int op_int, void *comm, void *stream,
                                         std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclAllReduce(sendbuff, recvbuff, count,
                                                       toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                                       static_cast<nccl::ncclComm_t>(comm),
                                                       static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclAllGatherInGroupWrapper(const void *sendbuff, void *recvbuff, size_t sendcount,
                                         int dtype_int, void *comm, void *stream,
                                         std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclAllGather(sendbuff, recvbuff, sendcount,
                                                       toNcclDataType(dtype_int),
                                                       static_cast<nccl::ncclComm_t>(comm),
                                                       static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclBroadcastInGroupWrapper(void *buff, size_t count, int dtype_int, int root,
                                         void *comm, void *stream, std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclBroadcast(buff, buff, count, toNcclDataType(dtype_int), root,
                                                       static_cast<nccl::ncclComm_t>(comm),
                                                       static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclReduceInGroupWrapper(const void *sendbuff, void *recvbuff, size_t count,
                                      int dtype_int, int op_int, int root,
                                      void *comm, void *stream, std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclReduce(sendbuff, recvbuff, count,
                                                    toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                                    root,
                                                    static_cast<nccl::ncclComm_t>(comm),
                                                    static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclReduceScatterInGroupWrapper(const void *sendbuff, void *recvbuff, size_t recvcount,
                                             int dtype_int, int op_int, void *comm, void *stream,
                                             std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                                           toNcclDataType(dtype_int), toNcclRedOp(op_int),
                                                           static_cast<nccl::ncclComm_t>(comm),
                                                           static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // =========================================================================
        // Point-to-Point Operations (for allgatherv emulation)
        // =========================================================================

        bool ncclSendWrapper(const void *sendbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclSend(sendbuff, count, toNcclDataType(dtype_int), peer,
                                                  static_cast<nccl::ncclComm_t>(comm),
                                                  static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        bool ncclRecvWrapper(void *recvbuff, size_t count, int dtype_int, int peer,
                             void *comm, void *stream, std::string &error_out)
        {
            nccl::ncclResult_t r = nccl::ncclRecv(recvbuff, count, toNcclDataType(dtype_int), peer,
                                                  static_cast<nccl::ncclComm_t>(comm),
                                                  static_cast<cudaStream_t>(stream));
            if (r != nccl::ncclSuccess)
            {
                error_out = nccl::ncclGetErrorString(r);
                return false;
            }
            return true;
        }

        // =========================================================================
        // Strided Deinterleave Kernel (for column-parallel AllGather)
        // =========================================================================
        //
        // After NCCL AllGather, data is laid out contiguously by rank:
        //   [rank0_row0, rank0_row1, ..., rank1_row0, rank1_row1, ...]
        //
        // Column-parallel LM head needs strided layout where each row has all ranks' data interleaved:
        //   row0: [rank0_slice, rank1_slice, ...]
        //   row1: [rank0_slice, rank1_slice, ...]
        //
        // This kernel deinterleaves from contiguous to strided layout.
        //
        // Input (contiguous):  [world_size * seq_len, local_dim]
        // Output (strided):    [seq_len, local_dim * world_size]
        //
        // Element mapping:
        //   output[row][rank * local_dim + col] = input[rank * seq_len + row][col]
        //
        // Grid: one thread per output element
        // Block: 256 threads
        // =========================================================================

        __global__ void deinterleaveKernel(
            const float *__restrict__ input, // Contiguous: [world_size * seq_len, local_dim]
            float *__restrict__ output,      // Strided: [seq_len, full_dim]
            int seq_len,
            int local_dim,
            int world_size)
        {
            // Total elements in output: seq_len * full_dim = seq_len * local_dim * world_size
            const int full_dim = local_dim * world_size;
            const int total_elements = seq_len * full_dim;

            for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < total_elements;
                 idx += gridDim.x * blockDim.x)
            {
                // Output coordinates
                const int row = idx / full_dim; // Which row in output [0, seq_len)
                const int col = idx % full_dim; // Which col in output [0, full_dim)

                // Which rank does this column belong to?
                const int rank = col / local_dim;      // [0, world_size)
                const int local_col = col % local_dim; // [0, local_dim)

                // Input coordinates: rank's data is at rows [rank*seq_len, (rank+1)*seq_len)
                const int input_row = rank * seq_len + row;
                const int input_col = local_col;
                const int input_idx = input_row * local_dim + input_col;

                output[idx] = input[input_idx];
            }
        }

        bool launchDeinterleaveKernel(
            const void *input,
            void *output,
            int seq_len,
            int local_dim,
            int world_size,
            void *stream)
        {
            const int full_dim = local_dim * world_size;
            const int total_elements = seq_len * full_dim;

            const int block_size = 256;
            const int grid_size = (total_elements + block_size - 1) / block_size;
            // Cap grid size to avoid excessive kernel launch overhead for small workloads
            const int max_blocks = 65535;
            const int actual_grid = (grid_size < max_blocks) ? grid_size : max_blocks;

            deinterleaveKernel<<<actual_grid, block_size, 0, static_cast<cudaStream_t>(stream)>>>(
                static_cast<const float *>(input),
                static_cast<float *>(output),
                seq_len,
                local_dim,
                world_size);

            cudaError_t err = cudaGetLastError();
            return (err == cudaSuccess);
        }

        // =========================================================================
        // Temporary Buffer Allocation (for strided AllGather)
        // =========================================================================

        bool cudaAllocTempBuffer(void **ptr, size_t bytes)
        {
            cudaError_t err = cudaMalloc(ptr, bytes);
            return (err == cudaSuccess);
        }

        void cudaFreeTempBuffer(void *ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
            }
        }

    } // namespace nccl_backend_detail
} // namespace llaminar2
