/**
 * @file ParityCellLifecycle.h
 * @brief Typed, fixture-owned MPI channels for one model-parity cell.
 *
 * Production runners may retain their own MPI communicator and immutable model
 * state across generated parity cells. Test evidence and teardown rendezvous
 * must therefore use fresh communicator contexts so a fast rank cannot match a
 * collective from an adjacent cell. This file owns that test-only protocol in
 * one state machine shared by every model and topology fixture.
 */

#pragma once

#include "utils/MPIContext.h"

#include <mpi.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace llaminar2::test::parity
{
    /** @brief Explicit lifetime states for one parity cell's test channels. */
    enum class ParityCellLifecycleState : std::uint8_t
    {
        NotEntered,
        Active,
        Retired,
    };

    /**
     * @brief RAII owner for one communicator duplicated from a cell rank set.
     *
     * MPIContext is a non-owning facade. This owner therefore outlives every
     * facade that refers to its communicator and releases the handle before
     * MPI_Finalize.
     */
    class ParityCellMPIChannel final
    {
    public:
        /**
         * @brief Collectively duplicate a communicator for one protocol lane.
         * @param source Live communicator containing every rank in the cell.
         * @throws std::invalid_argument when source is null.
         * @throws std::runtime_error when duplication or identity lookup fails.
         */
        explicit ParityCellMPIChannel(MPI_Comm source)
        {
            if (source == MPI_COMM_NULL)
            {
                throw std::invalid_argument(
                    "Parity cell channel requires a live MPI communicator");
            }
            if (MPI_Comm_dup(source, &communicator_) != MPI_SUCCESS ||
                communicator_ == MPI_COMM_NULL)
            {
                throw std::runtime_error(
                    "Could not duplicate a parity cell communicator");
            }
            if (MPI_Comm_rank(communicator_, &rank_) != MPI_SUCCESS ||
                MPI_Comm_size(communicator_, &world_size_) != MPI_SUCCESS)
            {
                MPI_Comm_free(&communicator_);
                throw std::runtime_error(
                    "Could not resolve a parity cell channel identity");
            }
        }

        /** @brief Release the channel while the MPI runtime is still active. */
        ~ParityCellMPIChannel()
        {
            if (communicator_ == MPI_COMM_NULL)
                return;

            int finalized = 0;
            if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
                MPI_Comm_free(&communicator_) != MPI_SUCCESS)
            {
                // A destructor cannot report an MPI ownership violation safely.
                std::terminate();
            }
        }

        ParityCellMPIChannel(const ParityCellMPIChannel &) = delete;
        ParityCellMPIChannel &operator=(const ParityCellMPIChannel &) = delete;

        /** @return The owned communicator handle. */
        [[nodiscard]] MPI_Comm communicator() const noexcept
        {
            return communicator_;
        }

        /** @return This process's rank within the duplicated communicator. */
        [[nodiscard]] int rank() const noexcept
        {
            return rank_;
        }

        /** @return Number of ranks participating in the channel. */
        [[nodiscard]] int worldSize() const noexcept
        {
            return world_size_;
        }

    private:
        MPI_Comm communicator_ = MPI_COMM_NULL;
        int rank_ = 0;
        int world_size_ = 1;
    };

    /**
     * @brief Single authority for all test-only MPI state owned by one cell.
     *
     * Entering constructs both protocol lanes atomically from the caller's
     * perspective. The control lane carries evidence and setup coordination;
     * the teardown lane carries exactly the entry and exit teardown barriers.
     * Retirement releases both only after production execution state is gone.
     */
    class ParityCellLifecycle final
    {
    public:
        ParityCellLifecycle() = default;
        ParityCellLifecycle(const ParityCellLifecycle &) = delete;
        ParityCellLifecycle &operator=(const ParityCellLifecycle &) = delete;

        /**
         * @brief Enter the cell exactly once and create both isolated lanes.
         * @param source Communicator containing the cell's complete rank set.
         * @throws std::logic_error on re-entry.
         * @throws std::runtime_error when channel construction fails.
         */
        void enter(MPI_Comm source)
        {
            if (state_ != ParityCellLifecycleState::NotEntered || channels_)
            {
                throw std::logic_error(
                    "Parity cell lifecycle may be entered exactly once");
            }

            auto channels = std::make_unique<ActiveChannels>(source);
            channels_ = std::move(channels);
            state_ = ParityCellLifecycleState::Active;
        }

        /** @return Current typed lifecycle state. */
        [[nodiscard]] ParityCellLifecycleState state() const noexcept
        {
            return state_;
        }

        /** @return Whether both isolated channels are currently active. */
        [[nodiscard]] bool active() const noexcept
        {
            return state_ == ParityCellLifecycleState::Active &&
                   channels_ != nullptr;
        }

        /**
         * @return Non-owning MPI facade for evidence and test coordination.
         * @throws std::logic_error outside the active state.
         */
        [[nodiscard]] const std::shared_ptr<IMPIContext> &
        controlContext() const
        {
            requireActive("Parity control requires an active cell lifecycle");
            return channels_->control_context;
        }

        /**
         * @brief Rendezvous on the teardown-only lane.
         * @throws std::logic_error outside the active state.
         */
        void teardownBarrier() const
        {
            requireActive(
                "Parity teardown requires an active isolated lifecycle");
            channels_->teardown_context->barrier();
        }

        /**
         * @brief Release both lanes and make the lifecycle permanently retired.
         * @throws std::logic_error unless the lifecycle is active.
         */
        void retire()
        {
            requireActive("Only an active parity cell lifecycle may retire");
            channels_.reset();
            state_ = ParityCellLifecycleState::Retired;
        }

    private:
        /**
         * @brief The only representable active channel set.
         *
         * Channel owners precede their MPIContext facades so reverse member
         * destruction drops the facades before freeing MPI handles.
         */
        struct ActiveChannels final
        {
            /** @brief Construct and authenticate both distinct protocol lanes. */
            explicit ActiveChannels(MPI_Comm source)
                : control_channel(
                      std::make_unique<ParityCellMPIChannel>(source)),
                  teardown_channel(
                      std::make_unique<ParityCellMPIChannel>(source)),
                  control_context(std::make_shared<MPIContext>(
                      control_channel->rank(),
                      control_channel->worldSize(),
                      control_channel->communicator())),
                  teardown_context(std::make_shared<MPIContext>(
                      teardown_channel->rank(),
                      teardown_channel->worldSize(),
                      teardown_channel->communicator()))
            {
                int relationship = MPI_UNEQUAL;
                if (MPI_Comm_compare(
                        control_channel->communicator(),
                        teardown_channel->communicator(),
                        &relationship) != MPI_SUCCESS ||
                    relationship != MPI_CONGRUENT)
                {
                    throw std::runtime_error(
                        "Parity control and teardown lanes are not isolated congruent communicators");
                }
            }

            std::unique_ptr<ParityCellMPIChannel> control_channel;
            std::unique_ptr<ParityCellMPIChannel> teardown_channel;
            std::shared_ptr<IMPIContext> control_context;
            std::shared_ptr<IMPIContext> teardown_context;
        };

        /** @brief Reject every operation that is invalid in the current state. */
        void requireActive(const char *message) const
        {
            if (!active())
                throw std::logic_error(message);
        }

        std::unique_ptr<ActiveChannels> channels_;
        ParityCellLifecycleState state_ =
            ParityCellLifecycleState::NotEntered;
    };
} // namespace llaminar2::test::parity
