/**
 * @file ServingResponseLeasePool.h
 * @brief PMA-backed response leases of continuous-batching HTTP serving.
 *
 * The single authority for batched response storage (§9.4): the pool commits
 * the ServingResponsePool owner line of the admitted memory plan — R
 * per-response budgets (mailbox, SSE staging) plus one retained decision-
 * logits row per resident slot — and admits requests against both typed axes:
 *
 *  - a response lease (at most R) lives from HTTP admission until the
 *    response finished publishing (detached delivery included);
 *  - an execution claim (at most S + Q: resident or waiting for a slot) is
 *    released as soon as the request's serving row finished, so a response
 *    still publishing never blocks a new request from executing.
 *
 * S = 1 serving keeps SerializedInferenceExecutor, which owns the same owner
 * line for its single execution slot.
 */
#pragma once

#include "execution/config/ServingCapacity.h"
#include "execution/config/ServingPoolMemoryDemand.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace llaminar2
{
    class ServingResponseLeasePool
    {
        struct State
        {
            explicit State(ResolvedServingCapacity serving) : capacity(serving) {}
            ResolvedServingCapacity capacity;
            PhysicalMemoryOwnerReservation reservation;
            std::mutex mutex;
            std::size_t responses = 0; ///< Admitted response leases (<= R).
            std::size_t executing = 0; ///< Resident or waiting requests (<= S + Q).
            bool closed = false;
        };

    public:
        /** @brief One admitted response; RAII over both typed axes. */
        class Lease
        {
            struct Key {};

        public:
            Lease(Key, std::shared_ptr<State> state) noexcept : state_(std::move(state)) {}
            Lease(const Lease &) = delete;
            Lease &operator=(const Lease &) = delete;

            /** @brief The request's serving row finished: free its execution claim (idempotent). */
            void executionFinished() noexcept
            {
                std::lock_guard lock(state_->mutex);
                if (executing_)
                {
                    executing_ = false;
                    --state_->executing;
                }
            }

            /** @brief Publication complete: release the response lease (and any claim). */
            ~Lease()
            {
                std::lock_guard lock(state_->mutex);
                if (executing_)
                    --state_->executing;
                --state_->responses;
            }

        private:
            friend class ServingResponseLeasePool;
            std::shared_ptr<State> state_;
            bool executing_ = true;
        };

        /** @brief Coherent counters for /stats. */
        struct Snapshot
        {
            std::size_t response_leases;     ///< R.
            std::size_t execution_capacity;  ///< S + Q.
            std::size_t responses;           ///< Admitted response leases.
            std::size_t executing;           ///< Resident or waiting requests.
            std::uint64_t committed_bytes;   ///< PMA ServingResponsePool commitment.
            std::uint64_t materialized_bytes; ///< Bytes claimed inside the commitment.
        };

        /**
         * @brief Commit the admitted ServingResponsePool owner line.
         * @throws std::invalid_argument When the plan has no such owner line
         *         or the authority is missing (production never serves unbacked).
         * @throws std::logic_error When the line is smaller than the demand.
         */
        ServingResponseLeasePool(ResolvedServingCapacity capacity,
                                 const std::shared_ptr<PhysicalMemoryAuthority> &authority,
                                 const ServingPoolPerResponseBudget &budget,
                                 const ServingSlotDecisionGeometry &slots)
            : state_(std::make_shared<State>(capacity))
        {
            if (!authority)
                throw std::invalid_argument("Batched serving requires a physical-memory authority");
            if (!authority->contains(DeviceId::cpu()) ||
                authority->plannedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool) == 0)
                throw std::invalid_argument(
                    "Serving pool owner line is not admitted by the physical-memory plan; "
                    "the plan must include the serving pool demand before service startup");
            state_->reservation = authority->reserveNewAllocations(
                DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool,
                servingPoolMemoryDemand(capacity, budget, slots));
        }

        /** @brief Device-free pool (tests): admission only, no PMA commitment. */
        explicit ServingResponseLeasePool(ResolvedServingCapacity capacity)
            : state_(std::make_shared<State>(capacity))
        {
        }

        /** @return A lease, or null when R or S + Q is exhausted or the pool closed. */
        [[nodiscard]] std::shared_ptr<Lease> tryAcquire()
        {
            std::lock_guard lock(state_->mutex);
            const auto responses = static_cast<std::size_t>(state_->capacity.responseLeases());
            const auto execution = static_cast<std::size_t>(state_->capacity.residentSlots()) +
                                   static_cast<std::size_t>(state_->capacity.queueCapacity());
            if (state_->closed || state_->responses >= responses || state_->executing >= execution)
                return {};
            ++state_->responses;
            ++state_->executing;
            return std::make_shared<Lease>(Lease::Key{}, state_);
        }

        /** @brief Refuse new admissions (shutdown). */
        void close()
        {
            std::lock_guard lock(state_->mutex);
            state_->closed = true;
        }

        [[nodiscard]] Snapshot snapshot() const
        {
            std::lock_guard lock(state_->mutex);
            return {static_cast<std::size_t>(state_->capacity.responseLeases()),
                    static_cast<std::size_t>(state_->capacity.residentSlots() + state_->capacity.queueCapacity()),
                    state_->responses, state_->executing,
                    state_->reservation.valid() ? state_->reservation.capacityBytes() : 0u,
                    state_->reservation.valid() ? state_->reservation.materializedBytes() : 0u};
        }

        [[nodiscard]] const ResolvedServingCapacity &capacity() const noexcept { return state_->capacity; }

        /**
         * @brief Claim the session's decision store from the committed pool.
         * @return Null for a device-free pool without a PMA reservation.
         * @throws std::logic_error When the claim would exceed the reservation.
         */
        [[nodiscard]] std::shared_ptr<PhysicalMemorySuballocationLease> claimDecisionStorage(std::size_t bytes) const
        {
            if (!state_->reservation.valid())
                return nullptr;
            return std::make_shared<PhysicalMemorySuballocationLease>(state_->reservation.claimAllocation(bytes));
        }

    private:
        std::shared_ptr<State> state_;
    };
} // namespace llaminar2
