/**
 * @file MultiDeviceIdentity.h
 * @brief Multi-device slot identity and participant agreement substrate (design §12).
 *
 * One serving slot identity/generation maps to every TP shard, PP stage and
 * expert tier; only the declared continuation owns sampling/control state.
 * Slot-qualified MPI commands carry ordered serving/lifecycle epochs.
 * Participation agreement uses a DEVICE-AUTHORED digest compared across
 * participants — the host never validates or selects a participation set
 * (no host-selected-set oracle). This header makes the rules typed: the
 * digest provenance is part of the type, and continuation-only sampling
 * authority is enforced by a typed guard.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief One slot identity shared by every participant (design §12.1). */
    struct MultiDeviceSlotIdentity
    {
        std::uint32_t slot = 0;
        std::uint64_t generation = 0;

        [[nodiscard]] bool sameIdentity(const MultiDeviceSlotIdentity &other) const noexcept
        {
            return slot == other.slot && generation == other.generation;
        }
    };

    /** @brief Ordered command eras for slot-qualified MPI commands (design §12.2). */
    struct SlotQualifiedEpochs
    {
        std::uint64_t serving_epoch = 0;
        std::uint64_t lifecycle_epoch = 0;

        [[nodiscard]] bool ordersBefore(const SlotQualifiedEpochs &other) const noexcept
        {
            return serving_epoch < other.serving_epoch ||
                   (serving_epoch == other.serving_epoch && lifecycle_epoch < other.lifecycle_epoch);
        }
    };

    /** @brief Continuation-ownership guard: ONLY the declared continuation samples. */
    struct ContinuationGuard
    {
        bool is_declared_continuation = false;

        void assertMaySample() const
        {
            if (!is_declared_continuation)
                throw std::logic_error(
                    "Only the declared continuation may own sampling/control state");
        }
    };

    /** @brief Device-authored participation digest (never a host-selected oracle). */
    struct ParticipationDigest
    {
        std::uint64_t value = 0;
        bool device_authored = true; ///< Provenance: authored on device, compared only.

        [[nodiscard]] bool agreesWith(const ParticipationDigest &other) const noexcept
        {
            if (!device_authored || !other.device_authored)
                return false; // a host-authored value is not a participation digest
            return value == other.value;
        }
    };
} // namespace llaminar2
