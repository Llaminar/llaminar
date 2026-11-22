/**
 * @file WindowGrant.h
 * @brief One-window execution grant authority and saturating serving age (design §5.2, §11).
 *
 * The runner's host-owned destination reservation is the sole authority to
 * issue a WindowGrant: it binds serving instance, slot/generation, a
 * non-reused grant ID, the immutable window policy and an opaque destination
 * reservation. At most one grant is outstanding per binding; a grant permits
 * at most one complete request-local output window and is never resized.
 * Resolution returns a typed receipt exactly once; cancellation returns the
 * unused grant. Age is a device-owned, saturating 64-bit count of completed
 * serving operations (admission-order tie-break is a host-stable ordinal),
 * never host wall-clock priority.
 */

#pragma once

#include "execution/serving/RequestIntrinsicCoordinates.h"

#include <cstdint>
#include <set>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Outstanding/resolved/cancelled grant state. */
    enum class GrantState : std::uint8_t
    {
        Outstanding,
        Resolved,
        Cancelled,
    };

    /** @brief Host-issued one-window execution grant value (design §5.2). */
    struct WindowGrant
    {
        std::uint64_t grant_id = 0;        ///< Non-reused grant identity (0 = unissued/invalid).
        std::uint32_t slot = 0;
        std::uint64_t generation = 0;
        IntrinsicWindowPlan window{};      ///< Immutable window policy (never resized).
        std::uint64_t destination_reservation = 0; ///< Opaque host-owned destination reservation.
        std::uint64_t serving_instance = 0;         ///< Serving-instance identity (binding).
        GrantState state = GrantState::Outstanding;
    };

    /**
     * @brief Sole grant authority: at most one outstanding grant per binding.
     *
     * @throws std::logic_error For a second outstanding grant on the same
     *         slot/generation, a reused grant ID, or a non-positive quantum.
     */
    inline void issueGrant(WindowGrant &authority, WindowGrant new_grant)
    {
        if (new_grant.grant_id == 0)
            throw std::logic_error("Grant ID must be nonzero (non-reused identity)");
        if (new_grant.window.quantum == 0)
            throw std::logic_error("Window grant quantum must be positive");
        if (authority.grant_id != 0 && authority.state == GrantState::Outstanding &&
            authority.slot == new_grant.slot && authority.generation == new_grant.generation)
            throw std::logic_error("At most one grant may be outstanding per binding");
        authority = new_grant;
    }

    /** @brief Typed receipt for a completed window (exactly once). */
    inline void resolveGrant(WindowGrant &grant)
    {
        if (grant.grant_id == 0)
            throw std::logic_error("An unissued (default) grant cannot be resolved");
        if (grant.state != GrantState::Outstanding)
            throw std::logic_error("Grant already resolved or cancelled");
        grant.state = GrantState::Resolved;
    }

    /** @brief Return an unused grant (backpressure/cancellation path). */
    inline void cancelGrant(WindowGrant &grant)
    {
        if (grant.grant_id == 0)
            throw std::logic_error("An unissued (default) grant cannot be cancelled");
        if (grant.state != GrantState::Outstanding)
            throw std::logic_error("Grant already resolved or cancelled");
        grant.state = GrantState::Cancelled;
    }

    /** @brief Whether a grant may execute (issued, nontrivial, outstanding, usable window). */
    [[nodiscard]] inline bool grantMayExecute(const WindowGrant &grant) noexcept
    {
        return grant.grant_id != 0 && grant.state == GrantState::Outstanding &&
               grant.window.quantum > 0 && grant.destination_reservation != 0 &&
               grant.serving_instance != 0;
    }

    /** @brief Non-copyable one-window grant authority (sole per-binding owner). */
    class WindowGrantAuthority final
    {
    public:
        explicit WindowGrantAuthority(std::uint64_t serving_instance = 0,
                                      std::size_t max_grants = kDefaultMaxGrants)
            : serving_instance_(serving_instance), max_grants_(max_grants) {}
        [[nodiscard]] std::size_t maxGrants() const noexcept { return max_grants_; }
        // NO copies (sole authority), but legitimate ownership transfer moves
        // are allowed (the source becomes a fresh empty authority).
        WindowGrantAuthority(const WindowGrantAuthority &) = delete;
        WindowGrantAuthority &operator=(const WindowGrantAuthority &) = delete;
        // Sole ownership transfer: the grant (which itself carries the immutable
        // serving-instance identity) and the never-reused ID ledger move together;
        // no per-authority instance state exists to lose or mismatch.
        WindowGrantAuthority(WindowGrantAuthority &&other)
            : grant_(std::move(other.grant_)), used_ids_(std::move(other.used_ids_)),
              serving_instance_(other.serving_instance_), max_grants_(other.max_grants_)
        {
            other.grant_ = WindowGrant{};
        }
        WindowGrantAuthority &operator=(WindowGrantAuthority &&other)
        {
            if (this != &other)
            {
                if (grantMayExecute(grant_))
                    throw std::logic_error(
                        "Cannot move over an outstanding grant (retire first)");
                // A FRESH (unbound) destination may adopt the source's identity;
                // an already-BOUND authority may never change instances.
                if (serving_instance_ != 0 && serving_instance_ != other.serving_instance_)
                    throw std::logic_error("Cross-instance authority move is refused");
                if (serving_instance_ == 0)
                {
                    serving_instance_ = other.serving_instance_;
                    max_grants_ = other.max_grants_;
                }
                else if (serving_instance_ == other.serving_instance_)
                {
                    // Same-instance bound move: the incoming request-derived bound
                    // replaces the destination's stale default bound.
                    max_grants_ = other.max_grants_;
                }
                grant_ = std::move(other.grant_);
                used_ids_ = std::move(other.used_ids_);
                other.grant_ = WindowGrant{};
            }
            return *this;
        }


        [[nodiscard]] const WindowGrant &current() const noexcept { return grant_; }
        [[nodiscard]] bool mayExecute() const noexcept { return grantMayExecute(grant_); }
        /// @brief Bounded never-reused grant identity ledger (default ceiling).
        static constexpr std::size_t kDefaultMaxGrants = 4096;

        /** @brief Issue the sole outstanding grant (replace only after resolution). */
        void issue(WindowGrant new_grant)
        {
            if (new_grant.grant_id == 0)
                throw std::logic_error("Grant ID must be nonzero (non-reused identity)");
            if (new_grant.window.quantum == 0)
                throw std::logic_error("Window grant quantum must be positive");
            // A grant must be BACKED by an owned destination reservation and bound
            // to this authority's serving instance (no unbacked issuance).
            if (new_grant.destination_reservation == 0)
                throw std::logic_error("Window grant must be backed by a destination reservation");
            if (new_grant.serving_instance == 0)
                throw std::logic_error("Window grant must carry the serving-instance identity");
            if (serving_instance_ == 0)
                throw std::logic_error(
                    "A wildcard (unbound) authority cannot issue grants");
            if (serving_instance_ != 0 && new_grant.serving_instance != serving_instance_)
                throw std::logic_error(
                    "Grant serving instance does not match THIS authority (authenticated)");
            if (used_ids_.size() >= max_grants_)
                throw std::logic_error("Authority grant ledger is bounded (request capacity reached)");
            if (grant_.grant_id != 0 && grant_.state == GrantState::Outstanding)
                throw std::logic_error(
                    "At most one grant may be outstanding on the sole authority "
                    "(resolve/cancel the current one first; grant IDs are never reused)");
            if (!used_ids_.insert(new_grant.grant_id).second)
                throw std::logic_error("Grant IDs are never reused (already issued)");
            grant_ = new_grant;
        }

        void resolve() { resolveGrant(grant_); }
        void cancel() { cancelGrant(grant_); }

    private:
        WindowGrant grant_;
        std::uint64_t serving_instance_ = 0;
        std::size_t max_grants_ = kDefaultMaxGrants;
        std::set<std::uint64_t> used_ids_; ///< Bounded never-reused grant identity ledger.
    };


    /** @brief Saturating 64-bit serving age (design §11): never wraps to zero. */
    struct ServingAge
    {
        std::uint64_t value = 0;

        void tick() noexcept
        {
            if (value != std::numeric_limits<std::uint64_t>::max())
                ++value;
        }
    };
} // namespace llaminar2
