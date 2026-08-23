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
#include <exception>

namespace llaminar2
{
    namespace host_backend_detail
    {

        namespace
        {
            /**
             * @brief Execute a host-registration operation in its owning CUDA context.
             *
             * Host registration is context-sensitive even when the portable
             * flag makes the resulting mapping visible from peer contexts.
             * Multi-device initialization and teardown may leave another CUDA
             * device current on the calling thread, so registration and
             * retirement must bind the declared owner explicitly.
             */
            bool selectRegistrationDevice(int device_ordinal, int *previous_device)
            {
                if (!previous_device || device_ordinal < 0)
                    return false;
                if (cudaGetDevice(previous_device) != cudaSuccess)
                {
                    (void)cudaGetLastError();
                    return false;
                }
                if (*previous_device == device_ordinal)
                    return true;
                if (cudaSetDevice(device_ordinal) == cudaSuccess)
                    return true;
                (void)cudaGetLastError();
                return false;
            }

            /**
             * @brief Restore the CUDA device selected before registration work.
             *
             * Losing the caller's exact device context makes all subsequent
             * stream and event ownership ambiguous. That state is not
             * recoverable by a host-staging fallback, so fail immediately.
             */
            void restoreRegistrationDevice(int previous_device, int selected_device)
            {
                if (previous_device != selected_device &&
                    cudaSetDevice(previous_device) != cudaSuccess)
                {
                    (void)cudaGetLastError();
                    std::terminate();
                }
            }
        } // namespace

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

        /**
         * @brief Register pageable storage for portable CUDA DMA.
         * @param ptr First byte of the caller-owned allocation.
         * @param size Number of live bytes in the allocation.
         * @param device_ordinal CUDA context that owns the registration lifecycle.
         */
        bool cudaHostRegisterBuffer(void *ptr, size_t size, int device_ordinal)
        {
            int previous_device = -1;
            if (!ptr || size == 0 ||
                !selectRegistrationDevice(device_ordinal, &previous_device))
                return false;

            cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
            if (err != cudaSuccess)
            {
                // CRITICAL: Clear the sticky CUDA error so downstream kernel launches
                // don't pick up a stale "invalid argument" from failed host pinning.
                // Host pinning is optional (mmap'd buffers typically can't be pinned),
                // so the error must not propagate.
                cudaGetLastError();
                restoreRegistrationDevice(previous_device, device_ordinal);
                return false;
            }

            restoreRegistrationDevice(previous_device, device_ordinal);
            return true;
        }

        /**
         * @brief Retire a host registration in the context that created it.
         * @return true only when CUDA confirms that the mapping was removed.
         */
        bool cudaHostUnregisterBuffer(void *ptr, int device_ordinal)
        {
            int previous_device = -1;
            if (!ptr || !selectRegistrationDevice(device_ordinal, &previous_device))
                return false;

            const cudaError_t error = cudaHostUnregister(ptr);
            if (error != cudaSuccess)
            {
                (void)cudaGetLastError();
            }
            restoreRegistrationDevice(previous_device, device_ordinal);
            return error == cudaSuccess;
        }

    } // namespace host_backend_detail
} // namespace llaminar2
