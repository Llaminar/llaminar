/**
 * @file MoEOverlayCpuPreparationGate.h
 * @brief Startup dependency for a shared, rank-local CPU ExpertOverlay bank.
 *
 * LocalTP GPU runners may materialize their private GPU weights concurrently.
 * A colocated CPU expert bank, however, is prepared exactly once by the
 * continuation-root runner and is subsequently read by every participant's
 * graph builder. This gate joins that one producer to its graph consumers
 * before capture; it is never consulted during inference or maintenance.
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /** @brief One-shot publication of a shared CPU expert bank's setup result. */
    class MoEOverlayCpuPreparationGate final
    {
    public:
        /** @brief Publish the completed bank after every local CPU expert is registered. */
        void publishPrepared()
        {
            {
                std::lock_guard lock(mutex_);
                if (state_ != State::Pending)
                    throw std::logic_error(
                        "MoE overlay CPU preparation was published more than once");
                state_ = State::Prepared;
            }
            ready_.notify_all();
        }

        /**
         * @brief Wake dependent graph builders when the producer cannot prepare.
         * @param detail Precise producer failure propagated to every dependent.
         *
         * A later runner-construction error may arrive after preparation was
         * already published; it cannot retract the prepared bank, so this
         * operation deliberately leaves a completed gate unchanged.
         */
        void publishFailure(std::string detail)
        {
            {
                std::lock_guard lock(mutex_);
                if (state_ != State::Pending)
                    return;
                failure_ = std::move(detail);
                state_ = State::Failed;
            }
            ready_.notify_all();
        }

        /**
         * @brief Join the producer before a dependent graph reads CPU engines.
         * @throws std::runtime_error with the exact producer failure.
         */
        void awaitPrepared() const
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return state_ != State::Pending; });
            if (state_ == State::Failed)
                throw std::runtime_error(
                    "MoE overlay CPU expert preparation failed: " + failure_);
        }

    private:
        enum class State : std::uint8_t
        {
            Pending,
            Prepared,
            Failed,
        };

        mutable std::mutex mutex_;
        mutable std::condition_variable ready_;
        State state_ = State::Pending;
        std::string failure_;
    };

    /**
     * @brief Typed role for one child runner in the rank-local setup edge.
     *
     * The participation object always contains a valid gate and one of two
     * roles. A child cannot accidentally publish as a dependent or wait on
     * its own unpublished bank.
     */
    class MoEOverlayCpuPreparationParticipation final
    {
    public:
        /** @brief Create the sole continuation-root publication role. */
        static MoEOverlayCpuPreparationParticipation publisher(
            std::shared_ptr<MoEOverlayCpuPreparationGate> gate)
        {
            return {std::move(gate), Role::Publisher};
        }

        /** @brief Create a child graph dependency on the producer's bank. */
        static MoEOverlayCpuPreparationParticipation dependent(
            std::shared_ptr<MoEOverlayCpuPreparationGate> gate)
        {
            return {std::move(gate), Role::Dependent};
        }

        /** @brief Publish or join after this child finished its own weight preparation. */
        void afterLocalPreparation() const
        {
            if (role_ == Role::Publisher)
                gate_->publishPrepared();
            else
                gate_->awaitPrepared();
        }

        /**
         * @brief Retire a producer failure so dependent futures cannot hang.
         * @param detail Failure from the child runner's outer construction scope.
         */
        void publishFailureIfProducer(std::string detail) const
        {
            if (role_ == Role::Publisher)
                gate_->publishFailure(std::move(detail));
        }

    private:
        enum class Role : std::uint8_t
        {
            Publisher,
            Dependent,
        };

        /** @brief Require a live gate for every published participation. */
        MoEOverlayCpuPreparationParticipation(
            std::shared_ptr<MoEOverlayCpuPreparationGate> gate,
            Role role)
            : gate_(std::move(gate)), role_(role)
        {
            if (!gate_)
                throw std::invalid_argument(
                    "MoE overlay CPU preparation participation requires a gate");
        }

        std::shared_ptr<MoEOverlayCpuPreparationGate> gate_;
        Role role_;
    };
} // namespace llaminar2
