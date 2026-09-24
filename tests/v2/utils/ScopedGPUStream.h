/**
 * @file ScopedGPUStream.h
 * @brief Backend-neutral ownership for explicit CUDA or ROCm test streams.
 *
 * GPU integration tests must exercise the same stream contract as production:
 * every asynchronous producer names a non-null stream, and coherence
 * publication records completion on that exact stream. This helper keeps the
 * stream lifetime local to a fixture without teaching backend-neutral tests
 * about `cudaStream_t` or `hipStream_t`.
 */

#pragma once

#include <stdexcept>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/DeviceId.h"

namespace llaminar2::test
{
    /**
     * @brief Own one non-default GPU stream for an integration-test device.
     *
     * Construction fails when the requested device is not a GPU, its backend
     * is unavailable, or the backend cannot create a real stream. This fail-fast
     * behavior prevents a test from silently changing into a stream-zero test
     * when device setup is incomplete.
     */
    class ScopedGPUStream final
    {
    public:
        /**
         * @brief Create a non-default stream on @p device.
         *
         * @param device CUDA or ROCm device whose backend owns the stream.
         * @throws std::invalid_argument when @p device is not a GPU.
         * @throws std::runtime_error when no backend or stream is available.
         */
        explicit ScopedGPUStream(DeviceId device)
            : device_(device),
              backend_(getBackendFor(device))
        {
            if (!device_.is_gpu())
            {
                throw std::invalid_argument(
                    "ScopedGPUStream requires a CUDA or ROCm device");
            }
            if (!backend_)
            {
                throw std::runtime_error(
                    "ScopedGPUStream could not resolve the device backend");
            }
            stream_ = backend_->createStream(device_.ordinal);
            if (!stream_)
            {
                throw std::runtime_error(
                    "ScopedGPUStream backend returned a null/default stream");
            }
        }

        ~ScopedGPUStream()
        {
            if (stream_)
                backend_->destroyStream(stream_, device_.ordinal);
        }

        ScopedGPUStream(const ScopedGPUStream &) = delete;
        ScopedGPUStream &operator=(const ScopedGPUStream &) = delete;
        ScopedGPUStream(ScopedGPUStream &&) = delete;
        ScopedGPUStream &operator=(ScopedGPUStream &&) = delete;

        /**
         * @brief Return the exact opaque stream passed to launches/publication.
         */
        [[nodiscard]] void *get() const noexcept
        {
            return stream_;
        }

        /**
         * @brief Return the device on which this stream was created.
         */
        [[nodiscard]] DeviceId device() const noexcept
        {
            return device_;
        }

    private:
        DeviceId device_ = DeviceId::cpu();
        IBackend *backend_ = nullptr;
        void *stream_ = nullptr;
    };
}
