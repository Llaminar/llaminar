/**
 * @file CUDAWorkerImpl.cu
 * @brief CUDA implementation of cuda_worker namespace functions
 *
 * These functions are called by GPUDeviceWorker to initialize and cleanup
 * CUDA contexts on worker threads used by NCCLBackend for collective operations.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include "../../utils/Logger.h"

namespace llaminar2
{
    namespace cuda_worker
    {

        /**
         * @brief Initialize CUDA context on the current worker thread
         *
         * Creates a CUDA stream and retains the primary context for the device.
         * This is called once when a GPUDeviceWorker starts its thread.
         *
         * @param ordinal GPU ordinal (0, 1, 2, ...)
         * @param out_stream Output: created CUDA stream (as void*)
         * @param out_context Output: retained CUDA context (as void*, CUcontext)
         * @return true on success, false on failure
         */
        bool initializeContext(int ordinal, void **out_stream, void **out_context)
        {
            // Set the device
            cudaError_t err = cudaSetDevice(ordinal);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[cuda_worker] cudaSetDevice(" << ordinal << ") failed: "
                                                         << cudaGetErrorString(err));
                return false;
            }

            // Retain the primary context using driver API
            CUcontext ctx;
            CUdevice dev;
            CUresult res = cuDeviceGet(&dev, ordinal);
            if (res != CUDA_SUCCESS)
            {
                LOG_ERROR("[cuda_worker] cuDeviceGet failed: " << res);
                return false;
            }

            res = cuDevicePrimaryCtxRetain(&ctx, dev);
            if (res != CUDA_SUCCESS)
            {
                LOG_ERROR("[cuda_worker] cuDevicePrimaryCtxRetain failed: " << res);
                return false;
            }

            // Set the context as current
            res = cuCtxSetCurrent(ctx);
            if (res != CUDA_SUCCESS)
            {
                LOG_ERROR("[cuda_worker] cuCtxSetCurrent failed: " << res);
                cuDevicePrimaryCtxRelease(dev);
                return false;
            }

            // Create a stream
            cudaStream_t stream;
            err = cudaStreamCreate(&stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[cuda_worker] cudaStreamCreate failed: " << cudaGetErrorString(err));
                cuDevicePrimaryCtxRelease(dev);
                return false;
            }

            *out_stream = static_cast<void *>(stream);
            *out_context = static_cast<void *>(ctx);

            LOG_DEBUG("[cuda_worker] Initialized context for CUDA device " << ordinal);
            return true;
        }

        /**
         * @brief Cleanup CUDA context on worker thread shutdown
         *
         * Destroys the stream and releases the primary context.
         *
         * @param ordinal GPU ordinal (0, 1, 2, ...)
         * @param stream CUDA stream to destroy (as void*)
         * @param context CUDA context to release (as void*, CUcontext)
         */
        void cleanupContext(int ordinal, void *stream, void *context)
        {
            if (stream)
            {
                cudaStreamDestroy(static_cast<cudaStream_t>(stream));
            }

            if (context)
            {
                CUdevice dev;
                CUresult res = cuDeviceGet(&dev, ordinal);
                if (res == CUDA_SUCCESS)
                {
                    cuDevicePrimaryCtxRelease(dev);
                }
            }

            LOG_DEBUG("[cuda_worker] Cleaned up context for CUDA device " << ordinal);
        }

    } // namespace cuda_worker
} // namespace llaminar2
