/**
 * @file ExplicitGPUStream.h
 * @brief Non-null, backend-opaque GPU execution stream binding.
 *
 * GPU execution ordering depends on the exact producer stream. Passing raw
 * nullable pointers through kernel interfaces made two very different actions
 * indistinguishable: binding a stream for execution and clearing a stale
 * lifecycle binding. This value type represents only the first action.
 *
 * Kernel interfaces accept ExplicitGPUStream while exposing a separately named
 * clearGPUStreamBinding() operation for teardown. Consequently,
 * setGPUStream(nullptr) is no longer a valid expression, and a nullable stream
 * variable must cross an explicit runtime validation boundary before it can be
 * installed on a kernel.
 */

#pragma once

#include <stdexcept>
#include <string>

namespace llaminar2
{
    /**
     * @brief Validated opaque handle for one CUDA or HIP execution stream.
     *
     * The type deliberately does not infer or own the backend stream. Ownership
     * remains with the executor or worker device context, and the kernel merely
     * borrows the handle for the duration of its explicit binding. The absence
     * of a default constructor prevents an "empty but apparently valid" value.
     */
    class ExplicitGPUStream final
    {
    public:
        ExplicitGPUStream() = delete;

        /**
         * @brief Validate and wrap an opaque CUDA or HIP stream handle.
         *
         * @param stream Non-null cudaStream_t or hipStream_t cast to void*.
         * @throws std::invalid_argument when @p stream is null.
         */
        explicit ExplicitGPUStream(void *stream)
            : stream_(stream)
        {
            if (!stream_)
            {
                throw std::invalid_argument(
                    "ExplicitGPUStream requires a non-null CUDA or HIP stream");
            }
        }

        /**
         * @brief Return the validated backend-opaque stream handle.
         */
        [[nodiscard]] void *get() const noexcept { return stream_; }

    private:
        void *stream_;
    };

    /**
     * @brief Require an already-bound stream at a GPU kernel launch boundary.
     *
     * Lightweight kernels that do not inherit CUDAKernelBase or ROCmKernelBase
     * use this helper from getStream(). Device-context ownership is deliberately
     * irrelevant here: the current execution transaction must have explicitly
     * bound its producer stream before a launch is legal.
     *
     * @param stream Borrowed stream currently stored by the kernel.
     * @param component Human-readable kernel family for the fatal diagnostic.
     * @return The unchanged, non-null stream handle.
     * @throws std::runtime_error when no explicit binding exists.
     */
    [[nodiscard]] inline void *requireExplicitGPUStreamBinding(
        void *stream,
        const char *component)
    {
        if (stream)
            return stream;

        throw std::runtime_error(
            std::string(component) +
            " requires an explicit non-null GPU stream binding before launch");
    }
} // namespace llaminar2
