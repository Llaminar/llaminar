/**
 * @file ServingCapacity.h
 * @brief One typed serving-capacity input and its qualified resolution.
 *
 * This is the adopted §9.4 serving-capacity contract for the continuous
 * batching work. It owns no bytes and is not a second physical-memory ledger:
 * residents, response leases, and queue/preparing capacity are ownership
 * counts. `PhysicalMemoryAuthority` and the typed BOM continue to own the
 * actual backing allocations behind each admission.
 *
 * Semantics:
 * - `ServingCapacityRequest` carries authored intent (resident slots S,
 *   response leases R, queue/preparing capacity Q) or an explicit automatic
 *   intent. An authored positive capacity is exact: it is never silently
 *   reduced or clamped.
 * - `ServingCapacityCapability` names what the implementation is admitted to
 *   serve. Only `capacityOnePreset()` is public and it is explicitly a
 *   self-consistent minimal preset used by the A-1 contract tests: S = 1
 *   (this branch's resident-one bound), R = 1, Q = 0. It is NOT the current
 *   serving admission envelope. The current HTTP accepted-response pool
 *   (`kDefaultCapacity = 32`, accepted responses including active and queued)
 *   remains owned by the existing pool and is bound through this type in
 *   Phase A-2 by the actual qualification-policy owner; no R/Q value is
 *   guessed here and no native/gate certification is asserted by this file.
 * - `resolveServingCapacity` validates the invariant envelope until
 *   `ResolvedServingCapacity` is immutable and every consumer borrows the same
 *   typed result. Constraints: S >= 1, Q >= 0, R >= S + Q.
 * - The runtime hard total (preparing-or-queued + execution-attached +
 *   detached-delivering responses <= R) is an admission-time invariant checked
 *   by the lifecycle authority against one resolved capacity. `admits()`
 *   checks every axis explicitly: queue capacity Q bounds preparing-or-queued,
 *   resident slots S bound execution-attached, and response leases R bound the
 *   total including detached delivery.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief Canonical preserved serving response pool (accepted including active and queued). */
    inline constexpr int kDefaultServingResponsePool = 32;

    /** @brief Shared installed serving response-pool ceiling (executor and qualification agree). */
    inline constexpr int kServingResponsePoolCeiling = 1024;

    /** @brief Typed rejection of a structurally valid but unqualified capacity request. */
    class ServingCapacityUnqualified : public std::invalid_argument
    {
    public:
        /** @brief Preserve the incompatible axis and requested/qualified values. */
        explicit ServingCapacityUnqualified(const std::string &message)
            : std::invalid_argument(message) {}
    };

    /** @brief Authoring mode for a serving-capacity intent. */
    enum class ServingCapacityIntentKind
    {
        /** @brief Explicit automatic intent; resolution derives from the quality envelope. */
        Automatic,
        /** @brief Exact authored values; never silently reduced by resolution. */
        Authored,
    };

    /**
     * @brief Immutable intent for resident slots, response leases, and queue capacity.
     *
     * Values are interpreted as exact ownership counts. Zero is a valid queue
     * capacity (no additional preparing/queued requests) but never a valid
     * resident or response count. The intent cannot be reassigned: it is
     * constructed once through a checked factory and borrowed thereafter.
     */
    class ServingCapacityRequest final
    {
    public:
        /** @brief Explicit automatic intent: the qualified ceiling is resolved later. */
        [[nodiscard]] static ServingCapacityRequest automatic() noexcept
        {
            return ServingCapacityRequest(ServingCapacityIntentKind::Automatic, 0, 0, 0);
        }

        /**
         * @brief Author exact ownership counts.
         * @param resident_slots Positive resident execution slots.
         * @param response_leases Positive bounded response/output leases.
         * @param queue_capacity Nonnegative preparing-or-queued capacity.
         * @return Immutable exact request.
         * @throws std::invalid_argument For a nonpositive resident/response count or
         *         a negative queue capacity.
         */
        [[nodiscard]] static ServingCapacityRequest authored(int resident_slots, int response_leases,
                                                             int queue_capacity)
        {
            if (resident_slots <= 0)
                throw std::invalid_argument("Serving capacity resident slots must be positive");
            if (response_leases <= 0)
                throw std::invalid_argument("Serving capacity response leases must be positive");
            if (queue_capacity < 0)
                throw std::invalid_argument("Serving capacity queue capacity must be nonnegative");
            return ServingCapacityRequest(ServingCapacityIntentKind::Authored, resident_slots,
                                          response_leases, queue_capacity);
        }

        /** @return Whether this is exact authored intent rather than automatic. */
        [[nodiscard]] constexpr bool isAutomatic() const noexcept
        {
            return kind_ == ServingCapacityIntentKind::Automatic;
        }

        /** @return Exact authored resident slots; zero with automatic intent. */
        [[nodiscard]] constexpr int residentSlots() const noexcept { return resident_slots_; }

        /** @return Exact authored response leases; zero with automatic intent. */
        [[nodiscard]] constexpr int responseLeases() const noexcept { return response_leases_; }

        /** @return Exact authored queue capacity; zero with automatic intent. */
        [[nodiscard]] constexpr int queueCapacity() const noexcept { return queue_capacity_; }

        friend constexpr bool operator==(const ServingCapacityRequest &, const ServingCapacityRequest &) noexcept = default;

        /** @brief Intent is immutable after construction (value copies only). */
        ServingCapacityRequest(const ServingCapacityRequest &) = default;
        ServingCapacityRequest(ServingCapacityRequest &&) = default;
        ServingCapacityRequest &operator=(const ServingCapacityRequest &) = delete;
        ServingCapacityRequest &operator=(ServingCapacityRequest &&) = delete;

    private:
        /** @brief Only the checked factories may construct intent. */
        constexpr ServingCapacityRequest(ServingCapacityIntentKind kind, int resident_slots, int response_leases,
                                         int queue_capacity) noexcept
            : kind_(kind), resident_slots_(resident_slots), response_leases_(response_leases),
              queue_capacity_(queue_capacity)
        {
        }

        ServingCapacityIntentKind kind_;
        int resident_slots_;
        int response_leases_;
        int queue_capacity_;
    };

    /**
     * @brief Qualified envelope the implementation is admitted to serve.
     *
     * Only the qualification owner may mint a production envelope: a general
     * public arithmetic factory is deliberately absent. `capacityOnePreset()`
     * is a minimal self-consistent fixture (S = 1, R = 1, Q = 0) used by the
     * A-1 contract tests; the installed production envelope is issued through
     * `ServingCapacityQualificationOwner`. This file asserts no native/gate
     * certification.
     */
    class ServingCapacityCapability final
    {
    public:
        /** @brief Minimal self-consistent capacity-one preset for the A-1 contract. */
        [[nodiscard]] static constexpr ServingCapacityCapability capacityOnePreset() noexcept
        {
            return ServingCapacityCapability(1, 1, 0);
        }

        /** @return Largest admitted resident slot count. */
        [[nodiscard]] constexpr int maxResidentSlots() const noexcept { return max_resident_slots_; }

        /** @return Largest admitted response-lease count. */
        [[nodiscard]] constexpr int maxResponseLeases() const noexcept { return max_response_leases_; }

        /** @return Largest admitted queue capacity. */
        [[nodiscard]] constexpr int maxQueueCapacity() const noexcept { return max_queue_capacity_; }

        /** @return Whether the envelope is self-consistent (R >= S + Q). */
        [[nodiscard]] constexpr bool isSelfConsistent() const noexcept
        {
            return max_resident_slots_ >= 1 && max_response_leases_ >= 1 && max_queue_capacity_ >= 0 &&
                   max_response_leases_ >= checkedSum(max_resident_slots_, max_queue_capacity_);
        }

        friend constexpr bool operator==(const ServingCapacityCapability &, const ServingCapacityCapability &) noexcept = default;

        /** @brief The envelope is immutable after construction (value copies only). */
        ServingCapacityCapability(const ServingCapacityCapability &) = default;
        ServingCapacityCapability(ServingCapacityCapability &&) = default;
        ServingCapacityCapability &operator=(const ServingCapacityCapability &) = delete;
        ServingCapacityCapability &operator=(ServingCapacityCapability &&) = delete;

    private:
        /** @brief Only the preset and the qualification owner may create an envelope. */
        constexpr explicit ServingCapacityCapability(int max_resident_slots, int max_response_leases,
                                                     int max_queue_capacity) noexcept
            : max_resident_slots_(max_resident_slots), max_response_leases_(max_response_leases),
              max_queue_capacity_(max_queue_capacity)
        {
        }

        friend struct ServingCapacityQualificationOwner;

        /** @brief Overflow-checked S + Q for the self-consistency invariant. */
        [[nodiscard]] static constexpr std::int64_t checkedSum(int left, int right) noexcept
        {
            return static_cast<std::int64_t>(left) + static_cast<std::int64_t>(right);
        }

        int max_resident_slots_;
        int max_response_leases_;
        int max_queue_capacity_;
    };

    /**
     * @brief Sole owner of the installed qualified serving envelope (Phase A).
     *
     * Issuance of a production envelope is restricted to this qualification
     * owner: the envelope encodes the current implementation's resident-one
     * bound together with a preserved accepted-response pool (active + queued
     * total), i.e. the existing HTTP admission semantics. No other general
     * arithmetic factory exists.
     */
    struct ServingCapacityQualificationOwner final
    {
        /**
         * @brief Install the Phase-A envelope for the preserved response pool.
         *
         * No arithmetic factory exists: the installed envelope is derived only
         * from the canonical preserved pool constant (active + queued total of
         * the existing HTTP executor), so an ordinary caller cannot mint an
         * envelope for a pool that was never served.
         * @return S = 1, R = kDefaultServingResponsePool, Q = R - S.
         */
        [[nodiscard]] static ServingCapacityCapability installedEnvelope() noexcept
        {
            return ServingCapacityCapability(1, kDefaultServingResponsePool,
                                             kDefaultServingResponsePool - 1);
        }
    };

    /**
     * @brief Immutable resolved serving capacity borrowed by every consumer.
     *
     * The resolved object is the single ownership-count contract for planning,
     * PMA BOM consumers, admission, and runtime diagnostics. It never owns
     * bytes, never grows or shrinks behind a consumer, and cannot be
     * reassigned: consumers may only read it or hold a copy of its value.
     */
    class ResolvedServingCapacity final
    {
    public:
        /** @brief Resolved resident execution slots (S). Always >= 1. */
        [[nodiscard]] constexpr int residentSlots() const noexcept { return resident_slots_; }

        /** @return Resolved response leases (R). Always >= S + Q. */
        [[nodiscard]] constexpr int responseLeases() const noexcept { return response_leases_; }

        /** @return Resolved queue/preparing capacity (Q). Always >= 0. */
        [[nodiscard]] constexpr int queueCapacity() const noexcept { return queue_capacity_; }

        /** @return Whether this is the minimal capacity-one preset (S=1,R=1,Q=0). */
        [[nodiscard]] constexpr bool isCapacityOnePreset() const noexcept
        {
            return resident_slots_ == 1 && queue_capacity_ == 0 && response_leases_ == 1;
        }

        /** @return Whether the preparing-or-queued count fits the Q axis. */
        [[nodiscard]] bool fitsPreparingOrQueued(int preparing_or_queued) const
        {
            if (preparing_or_queued < 0)
                throw std::invalid_argument("Serving capacity preparing-or-queued counter must be nonnegative");
            return preparing_or_queued <= queue_capacity_;
        }

        /** @return Whether the execution-attached count fits the S axis. */
        [[nodiscard]] bool fitsExecutionAttached(int execution_attached) const
        {
            if (execution_attached < 0)
                throw std::invalid_argument("Serving capacity execution-attached counter must be nonnegative");
            return execution_attached <= resident_slots_;
        }

        /** @return Whether the total response ownership fits the R axis. */
        [[nodiscard]] bool fitsResponseLeaseTotal(std::int64_t total) const
        {
            if (total < 0)
                throw std::invalid_argument("Serving capacity response total must be nonnegative");
            return total <= static_cast<std::int64_t>(response_leases_);
        }

        /**
         * @brief Check the hard admission-time ownership invariant on every axis.
         * @param preparing_or_queued Nonnegative preparing-or-queued response count.
         * @param execution_attached Nonnegative execution-attached response count.
         * @param detached_delivering Nonnegative detached-delivering response count.
         * @return True while the Q axis, the S axis, and the R total all fit.
         * @throws std::invalid_argument For a negative counter.
         */
        [[nodiscard]] bool admits(int preparing_or_queued, int execution_attached, int detached_delivering) const
        {
            if (preparing_or_queued < 0 || execution_attached < 0 || detached_delivering < 0)
                throw std::invalid_argument("Serving capacity admission counters must be nonnegative");
            return fitsPreparingOrQueued(preparing_or_queued) && fitsExecutionAttached(execution_attached) &&
                   fitsResponseLeaseTotal(static_cast<std::int64_t>(preparing_or_queued) +
                                          static_cast<std::int64_t>(execution_attached) +
                                          static_cast<std::int64_t>(detached_delivering));
        }

        friend constexpr bool operator==(const ResolvedServingCapacity &, const ResolvedServingCapacity &) noexcept = default;

        /** @brief The resolved value is immutable after construction (value copies only). */
        ResolvedServingCapacity(const ResolvedServingCapacity &) = default;
        ResolvedServingCapacity(ResolvedServingCapacity &&) = default;
        ResolvedServingCapacity &operator=(const ResolvedServingCapacity &) = delete;
        ResolvedServingCapacity &operator=(ResolvedServingCapacity &&) = delete;

    private:
        /** @brief Only resolveServingCapacity may construct a resolved result. */
        constexpr explicit ResolvedServingCapacity(int resident_slots, int response_leases, int queue_capacity) noexcept
            : resident_slots_(resident_slots), response_leases_(response_leases), queue_capacity_(queue_capacity)
        {
        }

        friend ResolvedServingCapacity resolveServingCapacity(const ServingCapacityRequest &,
                                                              const ServingCapacityCapability &);

        int resident_slots_;
        int response_leases_;
        int queue_capacity_;
    };

    /**
     * @brief Resolve one immutable serving capacity from intent and the quality envelope.
     *
     * Automatic intent resolves to the qualified envelope. Authored intent is
     * exact: it is validated against the invariant envelope and the capability,
     * and is rejected with `ServingCapacityUnqualified` when the envelope
     * cannot serve it — never silently reduced.
     *
     * @param request Immutable authored or automatic intent.
     * @param capability Qualified envelope of the installed implementation.
     * @return Immutable resolved ownership counts.
     * @throws std::invalid_argument For a structurally invalid, overflowed request.
     * @throws ServingCapacityUnqualified For valid intent the envelope cannot serve.
     */
    inline ResolvedServingCapacity resolveServingCapacity(const ServingCapacityRequest &request,
                                                          const ServingCapacityCapability &capability)
    {
        if (!capability.isSelfConsistent())
            throw std::logic_error("Serving capacity capability envelope is self-inconsistent");

        int resident = request.isAutomatic() ? capability.maxResidentSlots() : request.residentSlots();
        int responses = request.isAutomatic() ? capability.maxResponseLeases() : request.responseLeases();
        int queue = request.isAutomatic() ? capability.maxQueueCapacity() : request.queueCapacity();

        if (resident <= 0)
            throw std::invalid_argument("Resolved serving capacity resident slots must be positive");
        if (responses <= 0)
            throw std::invalid_argument("Resolved serving capacity response leases must be positive");
        if (queue < 0)
            throw std::invalid_argument("Resolved serving capacity queue capacity must be nonnegative");
        if (static_cast<std::int64_t>(responses) <
            static_cast<std::int64_t>(resident) + static_cast<std::int64_t>(queue))
            throw std::invalid_argument(
                "Resolved serving capacity response leases must cover resident slots plus queue capacity");

        if (resident > capability.maxResidentSlots())
            throw ServingCapacityUnqualified("Serving capacity resident slots " + std::to_string(resident) +
                                             " exceed the qualified envelope (" +
                                             std::to_string(capability.maxResidentSlots()) + ")");
        if (queue > capability.maxQueueCapacity())
            throw ServingCapacityUnqualified("Serving capacity queue capacity " + std::to_string(queue) +
                                             " exceeds the qualified envelope (" +
                                             std::to_string(capability.maxQueueCapacity()) + ")");
        if (responses > capability.maxResponseLeases())
            throw ServingCapacityUnqualified("Serving capacity response leases " + std::to_string(responses) +
                                             " exceed the qualified envelope (" +
                                             std::to_string(capability.maxResponseLeases()) + ")");

        return ResolvedServingCapacity(resident, responses, queue);
    }
    /**
     * @brief Resolve intent against the installed envelope for a preserved pool.
     *
     * This is the single enforcement entry: automatic intent resolves to the
     * installed envelope (behaviour-preserving), authored intent is exact and
     * is rejected with `ServingCapacityUnqualified` when the installed
     * envelope cannot serve it (never clamped).
     *
     * @param request Immutable authored or automatic intent.
     * @param http_response_pool Positive preserved accepted-response pool.
     * @return Immutable resolved ownership counts.
     * @throws std::invalid_argument For structurally invalid intent or a
     *         nonpositive pool.
     * @throws ServingCapacityUnqualified For valid intent outside the envelope.
     */
    inline ResolvedServingCapacity resolveInstalledServingCapacity(const ServingCapacityRequest &request)
    {
        return resolveServingCapacity(request, ServingCapacityQualificationOwner::installedEnvelope());
    }
} // namespace llaminar2
