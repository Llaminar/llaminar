/**
 * @file ServingLifetimes.h
 * @brief Typed request/slot identity and response/execution/publication lifetimes.
 *
 * Adopted design §3.1 and §4. These are host-owned ownership machines; they
 * contain no GPU phase, position, learner or scheduler state. Invalid state
 * combinations are unrepresentable through named transitions, and the lease
 * machines are single-owner: copy and move construction are deleted so no
 * second handle can independently advance the same lifetime.
 *
 * - `ServingRequestHandle` identifies an external RequestId and exists before
 *   execution admission.
 * - `ServingSlotBinding` = (serving instance, slot, 64-bit generation); one
 *   binding is replicated across participants; generations are never reused
 *   and exhaustion fails closed.
 * - `ResponseLease` (Preparing -> Queued -> Attached -> Delivering -> Closed)
 *   may outlive the execution lease.
 * - `ExecutionLease` (Unbound -> Admitting -> Resident -> TerminalObserved ->
 *   Retiring -> Released) waits for its state/event borrowers, not the socket.
 * - `PublicationLease` (Reserved -> InFlight -> Detached -> Drained) owns one
 *   immutable request output and is released by its consumer/framer.
 */
#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /** @brief Immutable external request identity, created before execution admission. */
    class ServingRequestHandle final
    {
    public:
        /** @brief Create the external handle from the caller's RequestId. */
        [[nodiscard]] static ServingRequestHandle fromExternalId(std::uint64_t external_id) noexcept
        {
            return ServingRequestHandle(external_id);
        }

        /** @return The external RequestId, stable for the request lifetime. */
        [[nodiscard]] constexpr std::uint64_t externalId() const noexcept { return external_id_; }

        friend constexpr bool operator==(const ServingRequestHandle &, const ServingRequestHandle &) noexcept = default;

        ServingRequestHandle(const ServingRequestHandle &) = default;
        ServingRequestHandle(ServingRequestHandle &&) = default;
        ServingRequestHandle &operator=(const ServingRequestHandle &) = delete;
        ServingRequestHandle &operator=(ServingRequestHandle &&) = delete;

    private:
        explicit constexpr ServingRequestHandle(std::uint64_t external_id) noexcept : external_id_(external_id) {}
        std::uint64_t external_id_;
    };

    /**
     * @brief Stable (serving instance, slot, generation) binding.
     *
     * One binding is replicated across all participants; per-domain physical
     * addresses are derived from it, and domains must never mint independent
     * generations for the same request. Generations are at least 64-bit,
     * monotonic per slot, and never reused.
     */
    class ServingSlotBinding final
    {
    public:
        /**
         * @brief Author the binding.
         * @param serving_instance Process/runtime instance identity.
         * @param slot Nonnegative physical serving slot.
         * @param generation Positive monotonic slot generation (0 = never opened).
         * @throws std::invalid_argument For a negative slot or zero generation.
         */
        [[nodiscard]] static ServingSlotBinding from(std::uint64_t serving_instance, int slot,
                                                     std::uint64_t generation)
        {
            if (slot < 0)
                throw std::invalid_argument("Serving slot must be nonnegative");
            if (generation == 0)
                throw std::invalid_argument("Serving slot generation must be positive");
            return ServingSlotBinding(serving_instance, slot, generation);
        }

        /** @return The serving-instance identity that mints this binding. */
        [[nodiscard]] constexpr std::uint64_t servingInstance() const noexcept { return serving_instance_; }

        /** @return The physical serving slot. */
        [[nodiscard]] constexpr int slot() const noexcept { return slot_; }

        /** @return The monotonic, never-reused slot generation. */
        [[nodiscard]] constexpr std::uint64_t generation() const noexcept { return generation_; }

        friend constexpr bool operator==(const ServingSlotBinding &, const ServingSlotBinding &) noexcept = default;

        ServingSlotBinding(const ServingSlotBinding &) = default;
        ServingSlotBinding(ServingSlotBinding &&) = default;
        ServingSlotBinding &operator=(const ServingSlotBinding &) = delete;
        ServingSlotBinding &operator=(ServingSlotBinding &&) = delete;

    private:
        constexpr ServingSlotBinding(std::uint64_t serving_instance, int slot,
                                     std::uint64_t generation) noexcept
            : serving_instance_(serving_instance), slot_(slot), generation_(generation) {}

        std::uint64_t serving_instance_;
        int slot_;
        std::uint64_t generation_;
    };

    /**
     * @brief Monotonic per-slot generation bank; exhaustion fails closed.
     *
     * The bank is the sole mint/retire authority for its serving instance.
     * `next` never wraps: at the 64-bit ceiling it throws instead of reusing
     * a generation. Each minted generation stays open until `retire` retires
     * it; consecutive mints without retirement are refused, and a retired
     * generation is stale (`isCurrent` rejects it) while the slot itself
     * remains safely reusable for the next mint.
     */
    class ServingSlotGenerationBank final
    {
    public:
        /**
         * @param serving_instance The process/runtime instance that owns this bank.
         * @param max_slots Positive admitted slot count; the bank never grows
         *        beyond it (out-of-range slots are rejected). All storage is
         *        allocated here, never in `next`.
         * @param ceiling Exclusive monotonic ceiling; the production default is
         *        the full 64-bit range, so exhaustion fails closed before wrap.
         * @throws std::invalid_argument When the admitted slot count is zero.
         */
        explicit ServingSlotGenerationBank(std::uint64_t serving_instance,
                                           std::size_t max_slots,
                                           std::uint64_t ceiling = UINT64_MAX)
            : serving_instance_(serving_instance), max_slots_(max_slots), ceiling_(ceiling)
        {
            if (max_slots_ == 0)
                throw std::invalid_argument("Serving slot bank requires a positive admitted slot count");
            generations_.assign(max_slots_, 0);
            open_.assign(max_slots_, false);
        }

        ServingSlotGenerationBank(const ServingSlotGenerationBank &) = delete;
        ServingSlotGenerationBank &operator=(const ServingSlotGenerationBank &) = delete;
        ServingSlotGenerationBank(ServingSlotGenerationBank &&) = delete;
        ServingSlotGenerationBank &operator=(ServingSlotGenerationBank &&) = delete;

        /** @return The owning serving-instance identity (sole mint authority). */
        [[nodiscard]] constexpr std::uint64_t servingInstance() const noexcept { return serving_instance_; }

        /** @return The current generation for a slot; 0 when never opened. */
        [[nodiscard]] std::uint64_t current(int slot) const
        {
            checkSlot(slot);
            return generations_[static_cast<std::size_t>(slot)];
        }

        /** @return Whether the slot currently has an open (unretired) generation. */
        [[nodiscard]] bool hasOpenGeneration(int slot) const
        {
            checkSlot(slot);
            return open_[static_cast<std::size_t>(slot)];
        }

        /**
         * @brief Retire the binding's generation for safe slot reuse.
         *
         * The binding authenticates exactly which generation is retired: a
         * stale or foreign binding is refused, so a late retire cannot close a
         * newer request's generation. This models request-slot retirement only;
         * process/executor shutdown is a separate lifecycle owned elsewhere.
         * @throws std::invalid_argument For a foreign serving instance.
         * @throws std::logic_error When the binding is not the slot's current
         *         open generation (stale or already retired).
         */
        void retire(const ServingSlotBinding &binding)
        {
            if (binding.servingInstance() != serving_instance_)
                throw std::invalid_argument("Serving slot binding belongs to a foreign instance");
            checkSlot(binding.slot());
            const auto index = static_cast<std::size_t>(binding.slot());
            if (!open_[index] || generations_[index] != binding.generation())
                throw std::logic_error("Serving slot binding is stale or already retired");
            open_[index] = false;
        }

        /**
         * @brief Open the next generation for a slot (never allocates).
         * @throws std::logic_error When the previous generation is still open;
         *         retire it before minting the next.
         * @throws std::overflow_error When the monotonic ceiling would wrap.
         */
        [[nodiscard]] std::uint64_t next(int slot)
        {
            checkSlot(slot);
            const auto index = static_cast<std::size_t>(slot);
            if (open_[index])
                throw std::logic_error("Serving slot generation is still open; retire before minting the next");
            if (generations_[index] >= ceiling_)
                throw std::overflow_error("Serving slot generation exhausted; refusal before wrap");
            ++generations_[index];
            open_[index] = true;
            return generations_[index];
        }

        /** @return Whether the binding belongs to this instance and is the slot's current OPEN generation. */
        [[nodiscard]] bool isCurrent(const ServingSlotBinding &binding) const
        {
            if (binding.servingInstance() != serving_instance_)
                return false;
            checkSlot(binding.slot());
            const auto index = static_cast<std::size_t>(binding.slot());
            return open_[index] && generations_[index] == binding.generation();
        }

    private:
        void checkSlot(int slot) const
        {
            if (slot < 0)
                throw std::invalid_argument("Serving slot must be nonnegative");
            if (static_cast<std::size_t>(slot) >= max_slots_)
                throw std::out_of_range("Serving slot exceeds the admitted slot count");
        }

        std::vector<std::uint64_t> generations_;
        std::vector<bool> open_;
        std::uint64_t serving_instance_;
        std::size_t max_slots_;
        std::uint64_t ceiling_;
    };

    /** @brief Host-owned response lease state machine (design §4). */
    class ResponseLease final
    {
    public:
        enum class State
        {
            Preparing,
            Queued,
            Attached,
            Delivering,
            Closed,
            Cancelled,
        };

        /** @brief Start a response lease in preparation. */
        [[nodiscard]] static ResponseLease preparing() noexcept { return ResponseLease(State::Preparing); }

        /** @return The current typed state. */
        [[nodiscard]] constexpr State state() const noexcept { return state_; }

        /** @brief Named transition: Preparing -> Queued. */
        void transitionToQueued() { transition(State::Preparing, State::Queued); }
        /** @brief Pre-execution cancellation: Preparing -> Cancelled. */
        void cancel() { transition(State::Preparing, State::Cancelled); }
        /** @brief Abandoned queued submission: Queued -> Cancelled. */
        void abandon() { transition(State::Queued, State::Cancelled); }
        /** @brief Terminal cancelled state factory (release path). */
        [[nodiscard]] static ResponseLease cancelled() noexcept { return ResponseLease(State::Cancelled); }
        /** @brief Named transition: Queued -> Attached. */
        void transitionToAttached() { transition(State::Queued, State::Attached); }
        /** @brief Named transition: Attached -> Delivering. */
        void transitionToDelivering() { transition(State::Attached, State::Delivering); }
        /** @brief Terminal transition: Delivering -> Closed. */
        void close() { transition(State::Delivering, State::Closed); }

        ResponseLease(const ResponseLease &) = delete;
        ResponseLease(ResponseLease &&) = delete;
        ResponseLease &operator=(const ResponseLease &) = delete;
        ResponseLease &operator=(ResponseLease &&) = delete;

    private:
        explicit constexpr ResponseLease(State state) noexcept : state_(state) {}

        void transition(State from, State to)
        {
            if (state_ != from)
                throw std::logic_error("Invalid response-lease transition from " + std::to_string(static_cast<int>(state_)));
            state_ = to;
        }

        State state_;
    };

    /** @brief Host-owned execution lease state machine (design §4). */
    class ExecutionLease final
    {
    public:
        enum class State
        {
            Unbound,
            Admitting,
            Resident,
            TerminalObserved,
            Retiring,
            Released,
        };

        [[nodiscard]] static ExecutionLease unbound() noexcept { return ExecutionLease(State::Unbound); }
        [[nodiscard]] constexpr State state() const noexcept { return state_; }

        void admit() { transition(State::Unbound, State::Admitting); }
        void install() { transition(State::Admitting, State::Resident); }
        void observeTerminal() { transition(State::Resident, State::TerminalObserved); }
        void beginRetire() { transition(State::TerminalObserved, State::Retiring); }
        void release() { transition(State::Retiring, State::Released); }

        ExecutionLease(const ExecutionLease &) = delete;
        ExecutionLease(ExecutionLease &&) = delete;
        ExecutionLease &operator=(const ExecutionLease &) = delete;
        ExecutionLease &operator=(ExecutionLease &&) = delete;

    private:
        explicit constexpr ExecutionLease(State state) noexcept : state_(state) {}

        void transition(State from, State to)
        {
            if (state_ != from)
                throw std::logic_error("Invalid execution-lease transition from " + std::to_string(static_cast<int>(state_)));
            state_ = to;
        }

        State state_;
    };

    /** @brief Host-owned publication lease state machine (design §4). */
    class PublicationLease final
    {
    public:
        enum class State
        {
            Reserved,
            InFlight,
            Detached,
            Drained,
        };

        [[nodiscard]] static PublicationLease reserved() noexcept { return PublicationLease(State::Reserved); }
        [[nodiscard]] constexpr State state() const noexcept { return state_; }

        void begin() { transition(State::Reserved, State::InFlight); }
        void detach() { transition(State::InFlight, State::Detached); }
        void drain() { transition(State::Detached, State::Drained); }

        PublicationLease(const PublicationLease &) = delete;
        PublicationLease(PublicationLease &&) = delete;
        PublicationLease &operator=(const PublicationLease &) = delete;
        PublicationLease &operator=(PublicationLease &&) = delete;

    private:
        explicit constexpr PublicationLease(State state) noexcept : state_(state) {}

        void transition(State from, State to)
        {
            if (state_ != from)
                throw std::logic_error("Invalid publication-lease transition from " + std::to_string(static_cast<int>(state_)));
            state_ = to;
        }

        State state_;
    };

    /**
     * @brief Valid composite host state (design §4 example).
     *
     * (response = Delivering, execution = Released, publications still draining)
     * is valid; (response = Closed, execution = Resident) is not. A terminal
     * response state (Closed or Cancelled) requires a fully retired execution
     * lease and drained publications; the one exception is pre-execution
     * cancellation, where nothing was ever bound (Unbound execution, no
     * publications) and therefore nothing needs releasing or draining. The
     * composite observes the leases without owning them: it stores an
     * immutable state snapshot taken at construction.
     */
    class RespondingComposite final
    {
    public:
        explicit RespondingComposite(const ResponseLease &response, const ExecutionLease &execution,
                                     std::span<const PublicationLease> publications = {})
            : response_state_(response.state()), execution_state_(execution.state())
        {
            publication_states_.reserve(publications.size());
            for (const auto &publication : publications)
                publication_states_.push_back(publication.state());
            if (!isValid())
                throw std::logic_error("Invalid responding composite: a terminal (closed or cancelled) "
                                       "response requires a released execution lease and drained "
                                       "publications");
        }

        /** @return Whether the three ownership machines may coexist. */
        [[nodiscard]] bool isValid() const noexcept
        {
            const bool terminal = response_state_ == ResponseLease::State::Closed ||
                                  response_state_ == ResponseLease::State::Cancelled;
            if (!terminal)
                return true;
            // Pre-execution cancellation: nothing was ever bound, so nothing
            // must be released or drained. Holding a reserved publication here
            // is rejected: it could never be drained from Reserved.
            if (execution_state_ == ExecutionLease::State::Unbound && publication_states_.empty())
                return true;
            if (execution_state_ != ExecutionLease::State::Released)
                return false;
            for (const auto state : publication_states_)
            {
                if (state != PublicationLease::State::Drained)
                    return false;
            }
            return true;
        }

    private:
        ResponseLease::State response_state_;
        ExecutionLease::State execution_state_;
        std::vector<PublicationLease::State> publication_states_;
    };
} // namespace llaminar2
