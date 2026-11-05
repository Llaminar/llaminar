/**
 * @file KVCacheMetadataInitialization.h
 * @brief Cold exact-event completion before raw GPU sequence metadata is published.
 *
 * CUDA/ROCm cache constructors enqueue initial zeroes on their original setup
 * stream. Later raw-pointer consumers may use any independent explicit stream;
 * completing this cold producer event makes those pointers safe without adding
 * request-time waits, mutable host mirrors or another readiness authority.
 */
#pragma once
#include "backends/IBackend.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <memory>
#include <stdexcept>
namespace llaminar2::detail {
/**
 * @brief Complete only the original cold metadata producer through backend event ownership.
 * @param backend Original allocation backend; owns native event creation/destruction.
 * @param device Exact device owning both metadata rows and producer stream.
 * @param producer_stream Exact non-null stream that received the initial zeroes.
 * @throws std::logic_error During graph capture, before any native event creation.
 * @throws std::invalid_argument For missing stream or invalid device.
 * @throws std::runtime_error If event creation, record or completion fails.
 * @note This setup-only host event wait precedes publication of raw pointers.
 * The RAII event lives through completion or error unwind, then retires through
 * the SAME backend creator boundary used by synchronous setup transfers.
 */
inline void completeColdKVMetadataInitialization(
    IBackend& backend, int device, void* producer_stream)
{
    if (isGraphCaptureActive())
        throw std::logic_error("KV metadata initialization cannot complete during graph capture");
    if (device < 0 || !producer_stream)
        throw std::invalid_argument("KV metadata initialization requires its exact device and producer stream");
    const auto destroy = [&backend, device](void* event) {
        backend.destroyEvent(event, device);
    };
    std::unique_ptr<void, decltype(destroy)> completion(backend.createEvent(device), destroy);
    if (!completion ||
        !backend.recordEvent(completion.get(), device, producer_stream) ||
        !backend.waitForEvent(completion.get(), device))
    {
        throw std::runtime_error("Failed to complete original device sequence metadata initialization");
    }
}
}
