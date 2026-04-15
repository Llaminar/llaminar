/**
 * @file HostBackendCUDA.cu
 * @brief CUDA-specific helper functions for HostBackend
 *
 * Isolated CUDA runtime calls in separate compilation unit to avoid
 * conflicts with HIP headers.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <cuda_runtime.h>

namespace llaminar2
{
    namespace host_backend_detail
    {

        bool cudaCopyToHost(void *host_dst, const void *device_src, int device_ordinal, size_t bytes, void *stream)
        {
            cudaError_t err = cudaSetDevice(device_ordinal);
            if (err != cudaSuccess)
            {
                return false;
            }

            cudaStream_t s = static_cast<cudaStream_t>(stream);
            err = cudaMemcpyAsync(host_dst, device_src, bytes, cudaMemcpyDeviceToHost, s);
            if (err != cudaSuccess)
                return false;
            err = cudaStreamSynchronize(s);
            return (err == cudaSuccess);
        }

        bool cudaCopyFromHost(void *device_dst, const void *host_src, int device_ordinal, size_t bytes, void *stream)
        {
            cudaError_t err = cudaSetDevice(device_ordinal);
            if (err != cudaSuccess)
            {
                return false;
            }

            cudaStream_t s = static_cast<cudaStream_t>(stream);
            err = cudaMemcpyAsync(device_dst, host_src, bytes, cudaMemcpyHostToDevice, s);
            if (err != cudaSuccess)
                return false;
            err = cudaStreamSynchronize(s);
            return (err == cudaSuccess);
        }

        bool cudaHostRegisterBuffer(void *ptr, size_t size)
        {
            cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
            if (err != cudaSuccess)
            {
                // CRITICAL: Clear the sticky CUDA error so downstream kernel launches
                // don't pick up a stale "invalid argument" from failed host pinning.
                // Host pinning is optional (mmap'd buffers typically can't be pinned),
                // so the error must not propagate.
                cudaGetLastError();
                return false;
            }
            return true;
        }

        void cudaHostUnregisterBuffer(void *ptr)
        {
            cudaHostUnregister(ptr);
        }

    } // namespace host_backend_detail
} // namespace llaminar2
