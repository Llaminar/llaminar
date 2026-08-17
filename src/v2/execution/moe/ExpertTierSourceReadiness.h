/**
 * @file ExpertTierSourceReadiness.h
 * @brief Typed ordering proof for immutable ExpertOverlay source weights.
 *
 * A freshly prepared device projection needs an exact producer event before a
 * background transfer stream may read it.  A projection reached through an
 * installed immutable residency bank is different: installation happened only
 * after its producer event completed, and the retained epoch prevents mutation
 * or reclamation.  Encoding those two cases explicitly prevents callers from
 * inventing a dummy event or silently omitting a required dependency.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Exact authority proving when a transfer may read source weights.
     *
     * Instances are created only through the two named factories.  This keeps
     * an event-backed producer edge distinct from an already-published RCU bank
     * and makes a null event or epoch zero unrepresentable.
     */
    class ExpertTierSourceReadiness final
    {
    public:
        /** @brief Supported source-ordering authorities. */
        enum class Kind : std::uint8_t
        {
            ProducerEvent,
            PublishedQuiescentResidencyBank,
        };

        /**
         * @brief Bind the exact completion event recorded by a source producer.
         * @param event Non-null CUDA or HIP event owned beyond transfer start.
         * @throws std::invalid_argument When @p event is null.
         */
        [[nodiscard]] static ExpertTierSourceReadiness producerEvent(
            void *event)
        {
            if (!event)
                throw std::invalid_argument(
                    "Expert tier source producer event cannot be null");
            return ExpertTierSourceReadiness(
                Kind::ProducerEvent, event, 0);
        }

        /**
         * @brief Certify an immutable source retained by an installed RCU bank.
         * @param epoch Positive residency epoch that owns the source lifetime.
         * @throws std::invalid_argument When @p epoch is zero.
         */
        [[nodiscard]] static ExpertTierSourceReadiness
        publishedResidencyBank(std::uint64_t epoch)
        {
            if (epoch == 0)
                throw std::invalid_argument(
                    "Expert tier published source epoch must be positive");
            return ExpertTierSourceReadiness(
                Kind::PublishedQuiescentResidencyBank, nullptr, epoch);
        }

        /** @return Which ordering authority this value represents. */
        [[nodiscard]] Kind kind() const noexcept { return kind_; }

        /** @return Exact producer event, or null for a published RCU bank. */
        [[nodiscard]] void *event() const noexcept { return event_; }

        /** @return Retained source epoch, or zero for an event-backed source. */
        [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

        /** @return Whether a transfer stream must join @ref event(). */
        [[nodiscard]] bool requiresProducerWait() const noexcept
        {
            return kind_ == Kind::ProducerEvent;
        }

    private:
        /** Store only factory-validated authority fields. */
        ExpertTierSourceReadiness(
            Kind kind,
            void *event,
            std::uint64_t epoch) noexcept
            : kind_(kind), event_(event), epoch_(epoch)
        {
        }

        Kind kind_;
        void *event_ = nullptr;
        std::uint64_t epoch_ = 0;
    };
} // namespace llaminar2
