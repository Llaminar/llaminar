/**
 * @file SerializedInferenceExecutor.h
 * @brief One stable inference worker with bounded HTTP admission and independent readers.
 *
 * Only admitted inference work enters the stable worker, preserving model/KV
 * ownership and OpenMP locality. HTTP has more workers than admitted inference
 * responses, so queued generations cannot occupy every control-plane worker.
 * Reservations also cover SSE responses before their content provider starts.
 * Shutdown drains submitted work and rejects new admission; it never cancels
 * a GPU transaction or substitutes an inference deadline.
 *
 * Admission models the adopted §9.4 axes explicitly: the response-lease total
 * bound R (capacity), the preparing-or-queued bound Q (waiting responses, not
 * the currently executing slot occupant), and the single execution slot S
 * (one worker). Reservations carry the typed ResponseLease
 * (Preparing -> Queued -> Attached -> Delivering -> Closed/Cancelled); the
 * stable worker owns every completion transition, so slot release never
 * depends on caller scheduling and a throwing occupant cannot pin the idle
 * slot until response destruction. Retained delivery consumes only response
 * leases, never queue capacity.
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include "execution/config/ServingCapacity.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/config/ServingPoolMemoryDemand.h" // kServingResponsePoolCeiling
#include "execution/config/ServingLifetimes.h" // typed response lease

namespace llaminar2
{
    /** @brief Bound network waiters without allowing concurrent model execution. */
    class SerializedInferenceExecutor
    {
    public:
        class Reservation; // owns one admitted response; defined below

    private:
        /** @brief Type-erased worker job: uniquely owns its callable and result state. */
        struct JobBase
        {
            virtual void invoke() = 0;
            virtual ~JobBase() = default;
            JobBase() = default;
            JobBase(const JobBase &) = delete;
            JobBase &operator=(const JobBase &) = delete;
            JobBase(JobBase &&) = delete;
            JobBase &operator=(JobBase &&) = delete;
        };

        /**
         * @brief Request-bound completion receipt for one submission.
         *
         * Only the job that owns this ticket sets it, and only at the end of
         * its own completion hook (after the slot is idle). The waiter holds
         * its own shared_ptr, so the receipt survives the job and the
         * Reservation: a waiter can never be released by another job's
         * progress, and its wait never dereferences a destroyed Reservation.
         */
        struct JobTicket
        {
            std::mutex mutex;
            std::condition_variable cv;
            bool done = false;
        };

        /** @brief Host request ownership only; no model or device state is stored. */
        struct State
        {
            /** @brief Fix admission bounds before exposing the worker. */
            explicit State(ResolvedServingCapacity serving,
                           std::size_t queue_limit,
                           int headroom)
                : serving_capacity(serving), queue_capacity(queue_limit),
                  control_headroom(headroom) {}
            /** @brief The admitted typed serving capacity (R/S/Q authority). */
            ResolvedServingCapacity serving_capacity;
            /** @brief PMA-backed serving pool CAPACITY reservation (single
             *  typed charge). Only the capacity is committed here: the pool's
             *  typed record/staging layout materialization (suballocation
             *  claim + real backing) belongs to the real serving-engine slice,
             *  NOT to a synthetic blob. The reservation lives on the SHARED
             *  state so it follows final physical ownership (released only
             *  when the LAST reservation/worker drops the state) — never a
             *  ledger number on the executor that dies first. */
            PhysicalMemoryOwnerReservation pool_reservation;
            /** @brief Derived: reservation validity IS the reservation state. */
            [[nodiscard]] bool poolPmaReserved() const noexcept
            {
                return pool_reservation.valid();
            }

            /** @brief Commit the serving pool CAPACITY (single typed charge).
             *  @throws std::invalid_argument when the admitted plan has NO
             *          ServingResponsePool owner line for this rank (the
             *          memory plan must include addServingPoolDemand).
             *  @throws std::logic_error (canonical reserve diagnostic) when a
             *          POSITIVE line is smaller than the typed demand — the
             *          reserve enforces the admitted owner's incremental
             *          allowance with its own message; this installer does not
             *          pre-intercept it. Aggregate physical fit is certified
             *          by the admission itself. */
            void installPoolPma(const std::shared_ptr<PhysicalMemoryAuthority> &authority,
                                const ResolvedServingCapacity &serving,
                                const ServingPoolPerResponseBudget &budget)
            {
                const std::uint64_t demand = servingPoolMemoryDemand(serving, budget);
                // Declared owner line check FIRST: a zero line means the admitted
                // plan did not include the serving pool (a precise diagnostics
                // path, never a deep reserve failure).
                if (!authority->contains(DeviceId::cpu()) ||
                    authority->plannedBytes(DeviceId::cpu(),
                                            PhysicalMemoryOwner::ServingResponsePool) == 0)
                {
                    throw std::invalid_argument(
                        "Serving pool owner line is not admitted by the physical-memory plan; "
                        "the plan must include addServingPoolDemand before service startup");
                }
                pool_reservation = authority->reserveNewAllocations(
                    DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, demand);

            }
            /** @brief Maximum accepted responses that are preparing or queued (Q). */
            const std::size_t queue_capacity;
            /** @brief Typed control-plane headroom above R (single pool authority). */
            const int control_headroom;
            std::mutex mutex;
            std::condition_variable ready;
            /** @brief Wakes a reservation whose worker-side completion ran. */
            std::deque<std::unique_ptr<JobBase>> jobs; ///< Solely worker-owned execution jobs.
            std::thread::id worker_id;
            std::size_t reserved = 0;       ///< Total accepted response leases (R bound).
            std::size_t preparing = 0;      ///< Accepted, not yet submitted (waiting).
            std::size_t queued_waiting = 0; ///< Submitted waiting responses not yet executing (Q).
            std::size_t delivering = 0;     ///< Production finished; transport retained.
            std::uint64_t drained_publications = 0; ///< Drained via the transport terminal.
            /** @brief The exempt slot holder (S = 1): the executing request once
             *         started, or the first pending admission while the worker
             *         is idle. Never dangles: cleared under the state mutex by
             *         the holder's completion or destruction. */
            Reservation *holder_ = nullptr;
            bool active = false;        ///< The worker is executing a submission.
            bool stopping = false;
        };

    public:
        /** @brief Maximum accepted generation responses, including queued clients. */
        static constexpr std::size_t kDefaultCapacity = kDefaultServingResponsePool;
        /**
         * @brief Default maximum preparing-or-queued accepted responses (Q).
         *
         * The qualified capacity-one envelope resolves to Q = R - S: with one
         * execution-attached response, at most R - S others may be preparing or
         * queued, preserving the existing "accepted including active and queued
         * <= R" admission semantics.
         */
        static constexpr std::size_t kDefaultQueueCapacity = kDefaultCapacity - 1;

        /** @brief Coherent request-executor counters, independent of GPU execution state. */
        struct Snapshot
        {
            std::size_t capacity;       ///< Response-lease total bound R.
            std::size_t queue_capacity; ///< Preparing-or-queued bound Q.
            std::size_t admitted;       ///< Accepted response leases (R usage).
            std::size_t queued;         ///< Submitted waiting responses not yet executing.
            std::size_t preparing;      ///< Accepted, not yet submitted.
            std::size_t delivering;     ///< Production finished; transport retained.
            std::size_t pending;        ///< Submissions enqueued, not yet picked up by the worker.
            bool active;                ///< The single worker is executing a response.
        };

        /** @brief Own one admitted HTTP response and at most one inference submission. */
        class Reservation
        {
            /** @brief Only the executor can construct a charged response lifetime. */
            struct ConstructionKey {};

        public:
            /** @brief Construct under admission ownership after shared storage is allocated. */
            Reservation(ConstructionKey, std::shared_ptr<State> state) noexcept
                : state_(std::move(state)) {}
            /**
             * @brief Serve started publishing this response (Reserved -> InFlight).
             */
            void publicationInFlight()
            {
                std::lock_guard lock(state_->mutex);
                if (lease_.state() != ResponseLease::State::Delivering)
                    throw std::logic_error(
                        "Publication may only start after production (run) delivered the response");
                publication_.begin();
            }

            /**
             * @brief The response transport handle was released (InFlight -> Detached).
             */
            void publicationDetached()
            {
                std::lock_guard lock(state_->mutex);
                if (lease_.state() != ResponseLease::State::Delivering)
                    throw std::logic_error(
                        "Publication detachment requires a delivering response");
                publication_.detach();
            }

            /**
             * @brief Exact transport-completion edge (Detached -> Drained).
             *
             * Called once when the server finished delivering the response.
             * This is the single moment the retained-delivery count is
             * released for a normally completed publication; the response
             * lease is closed by destructor afterwards. Exactly-once: a second
             * call (or a call before Detached) is refused.
             */
            void publicationFinished()
            {
                std::lock_guard lock(state_->mutex);
                if (lease_.state() != ResponseLease::State::Delivering)
                    throw std::logic_error(
                        "Publication completion requires a delivering response (run completed)");
                if (publication_.state() != PublicationLease::State::Detached)
                    throw std::logic_error("Publication completion requires a detached publication");
                if (delivery_released_)
                    throw std::logic_error("Publication delivery already released (exactly once)");
                publication_.drain();
                --state_->delivering;
                delivery_released_ = true;
                ++state_->drained_publications;
            }

            /** @brief TRANSPORT TERMINAL RECEIPT (S2): authenticates BOTH the native
             *         transport verdict AND the producer lifecycle in ONE pass. This
             *         binding emits publication transitions only here (the provider
             *         writes inside run(); the releaser owns the terminal edge), so on
             *         a SUCCESSFUL transport the missing InFlight/Detached transitions
             *         are performed by THIS receipt and the response is DRAINED (counter
             *         increments). NON-THROWING SCOPE: the authenticated combinations
             *         (native failure; unfinished Producer/Preparing; already-closed
             *         delivery; Reserved->Detached->Drained performed internally) return
             *         normally, so a response-teardown caller never sees a throw from
             *         those paths. A genuine state-machine VIOLATION (e.g. a
             *         publication already InFlight at the terminal) still throws
             *         (fatal, never silently kept). */
            void publicationTerminal(bool native_success)
            {
                std::lock_guard lock(state_->mutex);
                if (!native_success)
                    return; // abandoned path (destructor releases delivering once)
                if (lease_.state() != ResponseLease::State::Delivering)
                    return; // producer never finished: the destructor owns the lease
                if (delivery_released_)
                    return; // exactly-once (already drained or released)
                if (publication_.state() == PublicationLease::State::Reserved)
                {
                    // The transport completed; the publication was never explicitly
                    // opened by this binding -> the terminal receipt performs the
                    // authenticated transitions: InFlight -> Detached -> Drained.
                    publication_.begin();
                    publication_.detach();
                }
                if (publication_.state() != PublicationLease::State::Detached)
                    throw std::logic_error(
                        "Publication terminal requires a detached publication "
                        "(state-machine violation is fatal, never silently kept)");
                publication_.drain();
                --state_->delivering;
                delivery_released_ = true;
                ++state_->drained_publications;
            }

            /**
             * @brief Release the response lease and any retained delivery.
             *
             * An SSE client leaving before publication still releases capacity;
             * a reservation that never submitted releases its preparing slot
             * through ResponseLease cancellation; a delivering response releases
             * the lease only.
             */
            ~Reservation()
            {
                std::lock_guard lock(state_->mutex);
                --state_->reserved;
                const auto state = lease_.state();
                if (state == ResponseLease::State::Preparing)
                {
                    if (occupant_)
                        state_->holder_ = nullptr; // pending holder leaves the exemption
                    else
                        --state_->preparing;
                    lease_.cancel();
                }
                else if (state == ResponseLease::State::Delivering)
                {
                    if (!delivery_released_)
                    {
                        // Client left before transport completion: release the
                        // retained-delivery count honestly (abandoned path).
                        --state_->delivering;
                        delivery_released_ = true;
                    }
                    lease_.close();
                }
                else
                {
                    // Unreachable: run() blocks until the worker advanced this
                    // lease to Delivering, and only run()/this destructor
                    // transition it. An impossible ownership state is fatal,
                    // never silently swallowed.
                    std::terminate();
                }
            }
            Reservation(const Reservation &) = delete;
            Reservation &operator=(const Reservation &) = delete;

            /**
             * @brief Execute once on the stable worker and deliver its result/exception.
             * @param work Synchronous inference transaction; captured references outlive this call.
             * @return The transaction's exact result, with no retry or alternate path.
             *
             * run() returns only after the worker completed THIS reservation's
             * completion hook (slot released, lease Delivering): release never
             * depends on caller scheduling. A successor may already be
             * executing at that point — the guarantee is per reservation, not
             * global worker idleness, and it always holds even if the worker
             * retires the job (and the final Reservation reference) before the
             * caller wakes.
             */
            template <class Work>
            auto run(Work &&work) -> std::invoke_result_t<Work>
            {
                using Result = std::invoke_result_t<Work>;
                using WorkT = std::decay_t<Work>;
                // Retain the State authority and a request-bound completion
                // ticket BEFORE any handoff: after the job is enqueued the
                // worker can complete this reservation's bookkeeping and
                // destroy the job — and with it the LAST Reservation reference
                // (see TypedJob) — so run() must never dereference `this`
                // again. The local shared_ptr keeps State alive and the
                // ticket is set only by THIS job's completion hook.
                std::shared_ptr<State> state = state_;
                auto ticket = std::make_shared<JobTicket>();
                // The callable lives INSIDE the worker-typed job; the caller's
                // std::future shares only the promise's ready result state,
                // never the callable (a packaged_task's task state would share
                // the callable with the future, so caller-side retention could
                // delay or relocate final-owner cleanup). The worker therefore
                // owns the sole retirement point for every captured owner
                // reference (see TypedJob): no caller-side retention can delay
                // or relocate final-owner cleanup.
                auto job = std::make_unique<TypedJob<Result, WorkT>>(this, state, std::forward<Work>(work),
                                                                      ticket);
                auto result = job->future();
                {
                    std::lock_guard lock(state->mutex);
                    if (lease_.state() != ResponseLease::State::Preparing || state->stopping ||
                        state->worker_id == std::this_thread::get_id())
                        throw std::logic_error("HTTP inference reservation is retired, submitted or reentrant");
                    // Enqueue before mutating any counter: an allocation failure
                    // must leave admission state untouched. The worker owns both
                    // bookkeeping edges of this submission:
                    // - start hook: a waiting submission leaves the waiting room
                    //   when it starts EXECUTING (not at completion); the S
                    //   exemption TRANSFERS to this sole actual execution owner,
                    //   recharging any displaced pending holder into the waiting
                    //   room; the lease advances Queued -> Attached here;
                    // - completion hook: production end (normal or thrown) frees
                    //   the slot, publishes the idle worker and advances the
                    //   lease Attached -> Delivering before any caller observes
                    //   the result, so release never depends on caller
                    //   scheduling and a throwing occupant cannot pin the idle
                    //   slot until response destruction.
                    state->jobs.emplace_back(std::move(job));
                    if (!occupant_)
                    {
                        --state->preparing;
                        ++state->queued_waiting; // waiting submissions count against Q, never the occupant
                    }
                    lease_.transitionToQueued();
                }
                state->ready.notify_one();
                // Wait only on THIS submission's request-bound ticket: the
                // worker sets it at the end of this job's own completion hook
                // (after the slot is released), so run() is never released by
                // another job's progress and never returns while this
                // reservation is still attached. The predicate reads only the
                // ticket (and the local State shared_ptr), never a member of
                // `this`, which the worker may already have destroyed.
                {
                    std::unique_lock lock(ticket->mutex);
                    ticket->cv.wait(lock, [&] { return ticket->done; });
                }
                if constexpr (std::is_void_v<Result>)
                {
                    result.get();
                    return;
                }
                else if constexpr (std::is_reference_v<Result>)
                {
                    // References stay references: the exact object the job
                    // produced is returned, never a decaying local copy.
                    return result.get();
                }
                else
                {
                    // Move the produced value out of the future so a move-only
                    // result type is supported; the caller-visible value is the
                    // job's own result, not a copy.
                    Result value = std::move(result.get());
                    return value;
                }
            }

        private:
            /**
             * @brief Solely worker-owned execution job (the "captured owner").
             *
             * The user callable lives INSIDE this object, uniquely owned by
             * the worker: the worker moves the job out of the queue, invokes
             * it, then destroys it outside the queue mutex. That destruction
             * is the only point where the callable (and any final Reservation
             * / lease references captured inside it) can retire. The caller
             * holds only a std::future sharing the promise's result state —
             * never the callable — so caller-side retention can neither delay
             * nor relocate final-owner cleanup: destroying the job on the
             * worker is provably the final-owner retirement. A std::promise is
             * used instead of a packaged_task because the task state shares
             * the callable with the future, which would let the caller keep
             * the callable alive.
             */
            template <class Result, class Work>
            class TypedJob final : public JobBase
            {
            public:
                TypedJob(Reservation *owner, std::shared_ptr<State> state, Work work,
                         std::shared_ptr<JobTicket> ticket)
                    : owner_(owner), state_(std::move(state)), fn_(std::move(work)),
                      ticket_(std::move(ticket))
                {
                }
                TypedJob(const TypedJob &) = delete;
                TypedJob &operator=(const TypedJob &) = delete;

                /** @brief Share only the ready result state with the caller. */
                std::future<Result> future() { return promise_.get_future(); }

                void invoke() override
                {
                    {
                        std::lock_guard start_lock(state_->mutex);
                        if (!owner_->occupant_)
                        {
                            --state_->queued_waiting; // executing now: no longer waiting
                            if (state_->holder_ != nullptr && state_->holder_ != owner_)
                            {
                                // Displaced pending holder: recharge it into
                                // the waiting room according to its progress.
                                auto *const displaced = state_->holder_;
                                displaced->occupant_ = false;
                                if (displaced->lease_.state() == ResponseLease::State::Preparing)
                                    ++state_->preparing;      // pending: enters the waiting room
                                else
                                    ++state_->queued_waiting; // submitted: back into the queue count
                            }
                            owner_->occupant_ = true;
                            state_->holder_ = owner_;
                        }
                        owner_->lease_.transitionToAttached(); // Queued -> Attached: executing on the worker
                    }
                    try
                    {
                        if constexpr (std::is_void_v<Result>)
                        {
                            std::invoke(fn_);
                            promise_.set_value();
                        }
                        else
                        {
                            promise_.set_value(std::invoke(fn_));
                        }
                    }
                    catch (...)
                    {
                        // An inference exception cannot kill or replace this
                        // worker: it is delivered to the original waiter.
                        promise_.set_exception(std::current_exception());
                    }
                    std::lock_guard completion_lock(state_->mutex);
                    if (owner_->occupant_)
                        state_->holder_ = nullptr; // production done: slot free, lease retained
                    state_->active = false;       // publish the idle slot before waking waiters
                    owner_->lease_.transitionToDelivering(); // Attached -> Delivering
                    ++state_->delivering;
                    // Publish THIS job's own receipt after the slot is idle:
                    // the waiter is released only by its own request, so it
                    // can never return while its lease is still Attached and
                    // never observes another job's progress.
                    {
                        std::lock_guard ticket_lock(ticket_->mutex);
                        ticket_->done = true;
                    }
                    ticket_->cv.notify_all();
                }

            private:
                Reservation *owner_;
                std::shared_ptr<State> state_;
                Work fn_;
                std::promise<Result> promise_;
                std::shared_ptr<JobTicket> ticket_;
            };

            friend class SerializedInferenceExecutor;
            std::shared_ptr<State> state_;
            ResponseLease lease_ = ResponseLease::preparing();
            PublicationLease publication_ = PublicationLease::reserved();
            bool delivery_released_ = false; ///< Delivering count released exactly once.
            bool occupant_ = false; ///< Owns the single execution slot.
        };

        /** @brief Committed serving-pool CAPACITY bytes (admitted reservation;
         *  no bytes are materialized at construction — typed-layout
         *  materialization belongs to the real serving-engine slice). */
        [[nodiscard]] std::size_t servingPoolCommittedBytes() const noexcept
        {
            return state_->pool_reservation.valid() ? state_->pool_reservation.capacityBytes() : 0u;
        }

        /** @brief Whether the serving pool capacity is committed through the
         *  admitted PMA reservation (reservation only; not a materialized
         *  claim — `servingPoolCommittedBytes()` reports the capacity). */
        [[nodiscard]] bool servingPoolPmaReserved() const noexcept
        {
            return state_->poolPmaReserved();
        }

        /**
         * @brief Start the sole inference thread with explicit bounded admission.
         * @param capacity Number of accepted generation responses (R), including active and queued.
         * @param queue_capacity Maximum accepted responses that are preparing or
         *        queued (Q); zero allows only the response occupying the worker.
         */
        explicit SerializedInferenceExecutor(
            ResolvedServingCapacity serving_capacity,
            int control_headroom = kDefaultServingHttpControlHeadroom)
            : state_(std::make_shared<State>(
                  serving_capacity,
                  checkedQueueCapacity(static_cast<std::size_t>(serving_capacity.queueCapacity())),
                  checkedControlHeadroom(serving_capacity, control_headroom))),
              worker_([state = state_] { serve(state); }) {}

        /**
         * @brief Start the sole inference thread with a PMA-RESERVED serving
         *  pool CAPACITY. REQUIRED for production serving: the authority must
         *  be non-null (ServerMode fails fast otherwise) and the admitted plan
         *  must carry a sufficient ServingResponsePool owner line (the exact
         *  typed demand; addServingPoolDemand). The aggregate physical fit is
         *  certified by the admission itself; this constructor commits the
         *  owner allowance and NEVER materializes pool storage (typed-layout
         *  materialization is the real serving-engine slice). Device-free
         *  tests use the plain capacity constructor; there is NO test-only
         *  null-authority shortcut on this path. */
        explicit SerializedInferenceExecutor(
            ResolvedServingCapacity serving_capacity,
            std::shared_ptr<PhysicalMemoryAuthority> authority,
            ServingPoolPerResponseBudget pool_budget,
            int control_headroom = kDefaultServingHttpControlHeadroom)
            : SerializedInferenceExecutor(serving_capacity, control_headroom)
        {
            if (!authority)
            {
                throw std::invalid_argument(
                    "PMA-backed serving pool requires a physical-memory authority "
                    "(production server refuses unbacked serving; device-free tests use "
                    "the plain capacity constructor)");
            }
            state_->installPoolPma(authority, serving_capacity, pool_budget);
        }

        /** @brief Device-free binding (friend-minted through the checked private
         *         path; never an ad-hoc capacity and never a public mint). */
        explicit SerializedInferenceExecutor(std::size_t limit = kDefaultCapacity,
                                             std::size_t queue_capacity = kDefaultQueueCapacity)
            : SerializedInferenceExecutor(ResolvedServingCapacity::fromComponentsChecked(
                  1, static_cast<int>(checkedCapacity(limit)),
                  static_cast<int>(checkedQueueCapacity(queue_capacity))))
{}
        /** @brief Drain accepted submissions before model ownership can be released. */
        ~SerializedInferenceExecutor() { shutdown(); }
        SerializedInferenceExecutor(const SerializedInferenceExecutor &) = delete;
        SerializedInferenceExecutor &operator=(const SerializedInferenceExecutor &) = delete;

        /**
         * @brief Reserve one network waiter without blocking an HTTP worker.
         *
         * Admission enforces the typed axes separately: the response-lease
         * total (R) bounds all accepted leases; the preparing-or-queued bound
         * (Q) covers only waiting responses, never the single execution-slot
         * occupant and never retained delivery. The first admission always gets
         * the execution slot even when Q is zero.
         * @return Its shared response lifetime, or null for an explicit overload/shutdown response.
         */
        [[nodiscard]] std::shared_ptr<Reservation> tryReserve()
        {
            std::lock_guard lock(state_->mutex);
            // The execution slot (S = 1) is admitted separately from the
            // preparing-or-queued bound (Q) and from retained delivery (R only).
            if (state_->stopping || state_->reserved >= static_cast<std::size_t>(state_->serving_capacity.responseLeases()))
                return {};
            const bool slot_busy = state_->holder_ != nullptr || state_->active;
            // Q bounds waiting responses only: the executing request and the
            // single exempt holder are never waiting-room work.
            if (slot_busy && state_->preparing + state_->queued_waiting >= state_->queue_capacity)
                return {};
            auto result = std::make_shared<Reservation>(Reservation::ConstructionKey{}, state_);
            ++state_->reserved;
            if (!slot_busy)
            {
                state_->holder_ = result.get(); // first pending admission takes the exemption
                result->occupant_ = true;
            }
            else
            {
                ++state_->preparing;
            }
            return result;
        }

        /** @brief Return the HTTP pool width that leaves capacity for control requests. */
        [[nodiscard]] std::uint64_t drainedPublications() const noexcept
        {
            std::lock_guard lock(state_->mutex);
            return state_->drained_publications;
        }

        [[nodiscard]] std::size_t httpWorkerCount() const noexcept
        {
            return static_cast<std::size_t>(
                servingHttpWorkerCount(state_->serving_capacity, state_->control_headroom));
        }

        /** @brief Read host queue ownership without waiting for a generation. */
        [[nodiscard]] Snapshot snapshot() const
        {
            std::lock_guard lock(state_->mutex);
            return {static_cast<std::size_t>(state_->serving_capacity.responseLeases()),
                    state_->queue_capacity, state_->reserved, state_->queued_waiting,
                    state_->preparing, state_->delivering, state_->jobs.size(), state_->active};
        }

        /** @brief Reject new submissions and join all previously submitted work. */
        void shutdown()
        {
            {
                std::lock_guard lock(state_->mutex);
                state_->stopping = true;
            }
            state_->ready.notify_one();
            if (worker_.joinable())
                worker_.join();
        }

    private:
        /** @brief Reject invalid capacity before constructing any worker or pool. */
        static int checkedControlHeadroom(const ResolvedServingCapacity &serving,
                                          int headroom)
        {
            // Validated BEFORE worker creation: an overflowing width must never
            // reach the noexcept httpWorkerCount() helper (which would terminate).
            if (headroom <= 0)
                throw std::invalid_argument("Serving HTTP control headroom must be positive (got " +
                                            std::to_string(headroom) + ", R=" +
                                            std::to_string(serving.responseLeases()) + ")");
            if (headroom > std::numeric_limits<int>::max() - serving.responseLeases())
                throw std::invalid_argument("Serving HTTP pool width overflows int");
            return headroom;
        }

        static std::size_t checkedCapacity(std::size_t capacity)
        {
            if (capacity == 0 || capacity > static_cast<std::size_t>(kServingResponsePoolCeiling))
                throw std::invalid_argument(
                    "HTTP inference capacity must be between 1 and the installed ceiling (1024)");
            return capacity;
        }

        /** @brief Reject an invalid queue bound before constructing the worker. */
        static std::size_t checkedQueueCapacity(std::size_t queue_capacity)
        {
            if (queue_capacity > static_cast<std::size_t>(kServingResponsePoolCeiling))
                throw std::invalid_argument(
                    "HTTP inference queue capacity must be at most the installed ceiling (1024)");
            return queue_capacity;
        }

        /** @brief Execute FIFO jobs on one thread, retaining ownership through exceptions. */
        static void serve(const std::shared_ptr<State> &state)
        {
            std::unique_lock lock(state->mutex);
            state->worker_id = std::this_thread::get_id();
            for (;;)
            {
                state->ready.wait(lock, [&] { return state->stopping || !state->jobs.empty(); });
                if (state->jobs.empty())
                    return;
                auto job = std::move(state->jobs.front());
                state->jobs.pop_front();
                state->active = true;
                lock.unlock();
                // An inference exception is delivered to the original waiter
                // through the job's promise; it can never kill or replace this
                // worker.
                job->invoke();
                // Retire the job BEFORE reacquiring the queue mutex. The job
                // uniquely owns the user callable (see Reservation::TypedJob), so this destruction is the sole
                // final-owner retirement point: releasing the final owner
                // references (captured work, Reservation handles) runs here,
                // outside the mutex. Retiring under the mutex would deadlock
                // the worker and shutdown.
                job = nullptr;
                lock.lock();
                state->active = false;
            }
        }

        std::shared_ptr<State> state_;
        std::thread worker_;
    };
}
