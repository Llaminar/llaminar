/**
 * @file GPUBlasSubmission.h
 * @brief Scoped host submission access to context-owned BLAS library handles.
 *
 * A projection's weight lifetime is not a device-library lifetime. Worker
 * contexts retain handles until device quiescence; short-lived projections
 * borrow them only while binding their exact stream/workspace and submitting
 * a complete library call. The lock orders host API mutation, never GPU work.
 * Captured replay does not acquire it. Concurrent streams must still use
 * independent arena scratch because their submitted kernels may overlap.
 */
#pragma once

#include <mutex>
#include <stdexcept>

namespace llaminar2
{
    class IWorkerGPUContext;

    /** @brief Non-copyable scope protecting one complete BLAS host submission. */
    class GPUBlasSubmission final
    {
    public:
        GPUBlasSubmission(const GPUBlasSubmission &) = delete;
        GPUBlasSubmission &operator=(const GPUBlasSubmission &) = delete;
        GPUBlasSubmission(GPUBlasSubmission &&) = delete;
        GPUBlasSubmission &operator=(GPUBlasSubmission &&) = delete;

        /** @return Context-owned BLAS handle, borrowed only inside this scope. */
        [[nodiscard]] void *handle() const noexcept { return handle_; }
        /** @return Context-owned BLASLt handle, borrowed only inside this scope. */
        [[nodiscard]] void *ltHandle() const noexcept { return lt_handle_; }

    private:
        friend class IWorkerGPUContext;

        /** @brief Lock the context's API state and reject incomplete initialization. */
        GPUBlasSubmission(std::mutex &mutex, void *handle, void *lt_handle)
            : lock_(mutex), handle_(handle), lt_handle_(lt_handle)
        {
            if (!handle_ || !lt_handle_)
                throw std::logic_error("GPU BLAS submission requires initialized context-owned handles");
        }

        std::unique_lock<std::mutex> lock_;
        void *const handle_;
        void *const lt_handle_;
    };
}
