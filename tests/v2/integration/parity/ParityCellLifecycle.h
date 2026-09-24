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
        TeardownEntered,
        OutcomeConverged,
        Retired,
    };

    /** @brief This rank's GoogleTest result before parity-cell retirement. */
    enum class ParityCellLocalOutcome : std::uint8_t
    {
        Passed,
        Failed,
    };

    /** @brief Rank-wide result agreed on the isolated teardown channel. */
    enum class ParityCellAggregateOutcome : std::uint8_t
    {
        AllRanksPassed,
        AtLeastOneRankFailed,
    };

    /**
     * @brief Immutable receipt proving every rank agreed on the cell outcome.
     *
     * `originating_failed_ranks` counts only ranks that were already failed
     * when consensus began. Passing peers use this receipt to publish the same
     * failure into GoogleTest before fail-fast is allowed to select another
     * generated cell.
     */
    struct ParityCellOutcomeConsensus final
    {
        ParityCellAggregateOutcome outcome =
            ParityCellAggregateOutcome::AllRanksPassed;
        int originating_failed_ranks = 0;
        int participant_count = 1;

        /** @return Whether every participating rank entered as passing. */
        [[nodiscard]] bool allRanksPassed() const noexcept
        {
            return outcome == ParityCellAggregateOutcome::AllRanksPassed;
        }
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
     * perspective. The control lane carries evidence and setup coordination.
     * The teardown lane owns one typed sequence: entry rendezvous, rank-wide
     * outcome consensus, exit rendezvous, retirement. This sequence prevents
     * rank-local GoogleTest fail-fast from advancing only part of an MPI cell.
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

        /** @return Whether the cell still admits setup/evidence operations. */
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
         * @brief Close cell work and rendezvous before resource retirement.
         *
         * No rank may inspect fail-fast or begin another generated cell after
         * this transition. The following legal operation is exactly one call
         * to `convergeOutcome()`.
         *
         * @throws std::logic_error outside the active state.
         */
        void beginTeardown()
        {
            requireActive(
                "Parity teardown may begin only from an active lifecycle");
            channels_->teardown_context->barrier();
            state_ = ParityCellLifecycleState::TeardownEntered;
        }

        /**
         * @brief Agree whether any rank failed before teardown began.
         * @param local_outcome This rank's typed GoogleTest outcome.
         * @return Rank-wide immutable outcome receipt.
         * @throws std::logic_error unless teardown has begun exactly once.
         * @throws std::invalid_argument for a corrupted enum value.
         * @throws std::runtime_error when MPI cannot complete the consensus.
         */
        [[nodiscard]] ParityCellOutcomeConsensus convergeOutcome(
            ParityCellLocalOutcome local_outcome)
        {
            requireState(
                ParityCellLifecycleState::TeardownEntered,
                "Parity outcome consensus requires the teardown-entered state");

            int local_failed = 0;
            switch (local_outcome)
            {
            case ParityCellLocalOutcome::Passed:
                break;
            case ParityCellLocalOutcome::Failed:
                local_failed = 1;
                break;
            default:
                throw std::invalid_argument(
                    "Parity cell received an invalid local outcome");
            }

            int failed_ranks = 0;
            if (MPI_Allreduce(
                    &local_failed,
                    &failed_ranks,
                    1,
                    MPI_INT,
                    MPI_SUM,
                    channels_->teardown_channel->communicator()) !=
                MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Parity cell could not converge its rank outcomes");
            }
            if (failed_ranks < 0 ||
                failed_ranks > channels_->teardown_channel->worldSize())
            {
                throw std::runtime_error(
                    "Parity cell outcome consensus returned an impossible failure count");
            }

            state_ = ParityCellLifecycleState::OutcomeConverged;
            return {
                .outcome = failed_ranks == 0
                               ? ParityCellAggregateOutcome::AllRanksPassed
                               : ParityCellAggregateOutcome::AtLeastOneRankFailed,
                .originating_failed_ranks = failed_ranks,
                .participant_count =
                    channels_->teardown_channel->worldSize(),
            };
        }

        /**
         * @brief Rendezvous after cleanup, release both lanes, and retire.
         *
         * Every rank has already imported the aggregate result into its local
         * test state before this barrier. Consequently GoogleTest can apply
         * fail-fast only after all participants have left the same cell.
         *
         * @throws std::logic_error unless outcome consensus completed.
         */
        void finishTeardown()
        {
            requireState(
                ParityCellLifecycleState::OutcomeConverged,
                "Parity teardown may finish only after outcome consensus");
            channels_->teardown_context->barrier();
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

        /** @brief Require one exact non-active lifecycle transition state. */
        void requireState(
            ParityCellLifecycleState required,
            const char *message) const
        {
            if (state_ != required || !channels_)
                throw std::logic_error(message);
        }

        std::unique_ptr<ActiveChannels> channels_;
        ParityCellLifecycleState state_ =
            ParityCellLifecycleState::NotEntered;
    };
} // namespace llaminar2::test::parity
