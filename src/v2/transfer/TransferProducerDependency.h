/**
 * @file TransferProducerDependency.h
 * @brief Immutable source-ordering contract for background transfer admission.
 *
 * A published, retained source is immediately readable. Freshly prepared bytes
 * instead retain their exact producer event until the transfer completes. The
 * maintenance authority observes that event without blocking; a graph service
 * must never infer readiness from the unrelated stream on which a client lives.
 */
#pragma once

#include <stdexcept>

namespace llaminar2
{
    /** Factory-validated dependency carried with one immutable copy command. */
    class TransferProducerDependency final
    {
    public:
        /** @return A source whose publication already proved producer completion. */
        [[nodiscard]] static TransferProducerDependency published() noexcept
        { return TransferProducerDependency(nullptr); }

        /**
         * @brief Retain the exact producer event as a non-blocking admission edge.
         * @param event Non-null event; the caller retains it until copy completion.
         * @throws std::invalid_argument for a missing producer event.
         */
        [[nodiscard]] static TransferProducerDependency afterEvent(void *event)
        {
            if (!event)
                throw std::invalid_argument("Transfer producer dependency requires an exact event");
            return TransferProducerDependency(event);
        }

        /** @return Exact event, or null only for an already-published source. */
        [[nodiscard]] void *event() const noexcept { return event_; }

    private:
        /** Store a factory-validated dependency without allocating any storage. */
        explicit TransferProducerDependency(void *event) noexcept : event_(event) {}
        void *event_;
    };
}
