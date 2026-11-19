/**
 * @file ScopedGPUCompletionEvent.h
 * @brief Reusable exact-stream terminal observation for native GPU fixtures.
 *
 * The event is allocated before capture or timing. Each host observation waits
 * only for work preceding its record on the named producer stream. Production
 * ordering remains device-side; this helper serves diagnostic readback and
 * retirement only, never a per-stage host join inside an inference graph.
 */
#pragma once
#include "backends/IBackend.h"
#include "backends/DeviceId.h"
#include <cstdint>
#include <exception>
#include <stdexcept>

namespace llaminar2::test {
/** @brief Own one completion event until all borrowed fixture work has retired. */
class ScopedGPUCompletionEvent final {
public:
    /**
     * @brief Allocate one reusable event on the fixture's exact endpoint.
     * @param backend Original backend retained through this owner's lifetime.
     * @param device Original CUDA or ROCm device.
     * @param stream Explicit producer stream; default-stream sentinels are rejected.
     * @throws std::invalid_argument For an incomplete endpoint.
     * @throws std::runtime_error If event creation fails.
     */
    ScopedGPUCompletionEvent(IBackend* backend, DeviceId device, void* stream)
        : backend_(backend), device_(device), stream_(stream) {
        if (!backend_ || !device_.is_gpu() || reinterpret_cast<std::uintptr_t>(stream_) <= 2)
            throw std::invalid_argument("Fixture completion requires an exact GPU endpoint and stream");
        event_ = backend_->createEvent(device_.ordinal);
        if (!event_) throw std::runtime_error("Fixture completion event creation failed");
    }
    /** @brief Retire preceding work before releasing the event, including failed assertions. */
    ~ScopedGPUCompletionEvent() {
        if (!join()) std::terminate();
        backend_->destroyEvent(event_, device_.ordinal);
    }
    ScopedGPUCompletionEvent(const ScopedGPUCompletionEvent&) = delete;
    ScopedGPUCompletionEvent& operator=(const ScopedGPUCompletionEvent&) = delete;
    ScopedGPUCompletionEvent(ScopedGPUCompletionEvent&&) = delete;
    ScopedGPUCompletionEvent& operator=(ScopedGPUCompletionEvent&&) = delete;
    /**
     * @brief Record and observe this stream's current submission prefix.
     * @return Whether the exact completion record and host observation succeeded.
     * @note Call outside capture, only at explicit diagnostic observation boundaries.
     */
    [[nodiscard]] bool join() {
        return backend_->recordEvent(event_, device_.ordinal, stream_) &&
               backend_->waitForEvent(event_, device_.ordinal);
    }
private:
    IBackend* backend_;
    DeviceId device_;
    void* stream_;
    void* event_ = nullptr;
};
} // namespace llaminar2::test
