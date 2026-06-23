/**
 * @file ROCmWorkerImpl.cpp
 * @brief ROCm/HIP implementation of rocm_worker namespace functions
 *
 * These functions are called by GPUDeviceWorker to initialize and cleanup
 * HIP contexts on worker threads used by RCCLBackend for collective operations.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <hip/hip_runtime.h>
#include "../../utils/Logger.h"

namespace llaminar2
{
    namespace rocm_worker
    {

        /**
         * @brief Initialize HIP context on the current worker thread
         *
         * Creates a HIP stream and sets the device for the current thread.
         *
         * @param ordinal GPU ordinal (0, 1, 2, ...)
         * @param out_stream Output: created HIP stream (as void*)
         * @param out_context Output: not used for HIP (set to nullptr)
         * @return true on success, false on failure
         */
        bool initializeContext(int ordinal, void **out_stream, void **out_context)
        {
            // Set the device
            hipError_t err = hipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                LOG_ERROR("[rocm_worker] hipSetDevice(" << ordinal << ") failed: "
                                                        << hipGetErrorString(err));
                return false;
            }

            // Create a stream
            hipStream_t stream;
            err = hipStreamCreate(&stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[rocm_worker] hipStreamCreate failed: " << hipGetErrorString(err));
                return false;
            }

            *out_stream = static_cast<void *>(stream);
            *out_context = nullptr; // HIP doesn't use explicit contexts like CUDA driver API

            LOG_DEBUG("[rocm_worker] Initialized context for ROCm device " << ordinal);
            return true;
        }

        /**
         * @brief Cleanup HIP context on worker thread shutdown
         *
         * Destroys the stream.
         *
         * @param ordinal GPU ordinal (0, 1, 2, ...) - unused
         * @param stream HIP stream to destroy (as void*)
         * @param context Unused (HIP doesn't have explicit contexts)
         */
        void cleanupContext(int ordinal, void *stream, void *context)
        {
            (void)context; // Unused - HIP doesn't have explicit contexts

            if (stream)
            {
                (void)hipStreamDestroy(static_cast<hipStream_t>(stream));
            }

            LOG_DEBUG("[rocm_worker] Cleaned up context for ROCm device " << ordinal);
        }

    } // namespace rocm_worker
} // namespace llaminar2
