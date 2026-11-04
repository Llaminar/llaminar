/**
 * @file ScopedMoEParityCoordinatedSession.h
 * @brief Existing production command-loop lifecycle shared by MoE parity fixtures.
 *
 * Only the original continuation invokes inference. Every other original rank
 * stays in its production worker loop until the same runner publishes shutdown.
 * This test owner retires local graphs before its final retirement rendezvous.
 */
#pragma once
#include "execution/runner/IOrchestrationRunner.h"
#include <mpi.h>
#include <iostream>
#include <stdexcept>
namespace llaminar2::test::parity {
    /**
     * @brief Own one production-shaped coordinated MPI inference session.
     *
     * A heterogeneous ExpertOverlay has exactly one frontend: the
     * inventory-resolved continuation authority.  Every other MPI rank must
     * remain in @ref IOrchestrationRunner::runMPIWorkerLoop while that
     * authority issues public inference calls.  Having every parity process
     * call `generate()` directly creates two frontends and bypasses the
     * retained transaction-follower protocol that production actually uses.
     *
     * This scope makes that role split explicit and owns the terminal
     * `SHUTDOWN` command.  Both sides retire their local runner before the
     * final test-only barrier, so a following runner cannot enter setup while
     * a peer still owns graphs or backend resources from this session.
     */
    class ScopedMoEParityCoordinatedSession
    {
    public:
        /**
         * @brief Admit all ranks to one coordinated serving session.
         *
         * @param runner Fully initialized production orchestration runner.
         * @throws std::runtime_error when MPI is unavailable or the resolved
         *         continuation authority is outside the active world.
         */
        explicit ScopedMoEParityCoordinatedSession(
            IOrchestrationRunner &runner)
            : runner_(runner)
        {
            int initialized = 0;
            if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized)
            {
                throw std::runtime_error(
                    "Coordinated MoE parity requires an initialized MPI world");
            }
            if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_) != MPI_SUCCESS ||
                MPI_Comm_size(MPI_COMM_WORLD, &world_size_) != MPI_SUCCESS ||
                world_size_ <= 1)
            {
                throw std::runtime_error(
                    "Coordinated MoE parity requires more than one MPI rank");
            }

            authority_rank_ = runner_.coordinatedRootRank();
            if (authority_rank_ < 0 || authority_rank_ >= world_size_)
            {
                throw std::runtime_error(
                    "Coordinated MoE parity resolved an invalid continuation authority rank");
            }

            runner_.setMPICoordinatedMode(true);
            if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
            {
                runner_.setMPICoordinatedMode(false);
                throw std::runtime_error(
                    "Coordinated MoE parity admission barrier failed");
            }
            active_ = true;
        }

        ScopedMoEParityCoordinatedSession(
            const ScopedMoEParityCoordinatedSession &) = delete;
        ScopedMoEParityCoordinatedSession &operator=(
            const ScopedMoEParityCoordinatedSession &) = delete;

        /** @return Whether this process owns continuation and public requests. */
        [[nodiscard]] bool isAuthority() const noexcept
        {
            return world_rank_ == authority_rank_;
        }

        /** @return Inventory-resolved authority used by evidence broadcasts. */
        [[nodiscard]] int authorityRank() const noexcept
        {
            return authority_rank_;
        }

        /**
         * @brief Serve production commands until the authority closes the lane.
         *
         * Only a non-authority rank may enter this method.  It returns after
         * observing the typed terminal command, retires local resources, and
         * joins the same session-retirement barrier as the authority.
         */
        void serveFollowerAndFinish()
        {
            if (!active_ || isAuthority())
            {
                throw std::logic_error(
                    "Only an active non-authority parity rank may serve the worker loop");
            }
            runner_.runMPIWorkerLoop();
            finishLocal(/*publish_shutdown=*/false);
        }

        /**
         * @brief Publish the terminal command and retire the authority runner.
         */
        void finishAuthority()
        {
            if (!active_ || !isAuthority())
            {
                throw std::logic_error(
                    "Only an active parity authority may close the worker loop");
            }
            finishLocal(/*publish_shutdown=*/true);
        }

        /**
         * @brief Ensure fatal GoogleTest exits still release every follower.
         *
         * Normal code calls @ref finishAuthority explicitly.  Assertions can
         * return from the enclosing helper, so the authority destructor owns
         * the same terminal transition as a last-resort test lifecycle guard.
         */
        ~ScopedMoEParityCoordinatedSession() noexcept
        {
            if (!active_ || !isAuthority())
                return;
            try
            {
                finishLocal(/*publish_shutdown=*/true);
            }
            catch (const std::exception &error)
            {
                std::cerr
                    << "Failed to retire coordinated MoE parity session: "
                    << error.what() << '\n';
            }
        }

    private:
        /**
         * @brief Retire process-local state and rendezvous after graph teardown.
         *
         * @param publish_shutdown True only on the continuation authority.
         */
        void finishLocal(bool publish_shutdown)
        {
            if (publish_shutdown)
                runner_.shutdownMPIWorkers();
            runner_.setMPICoordinatedMode(false);
            runner_.shutdown();
            if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Coordinated MoE parity retirement barrier failed");
            }
            active_ = false;
        }

        IOrchestrationRunner &runner_;
        int world_rank_ = -1;
        int world_size_ = 0;
        int authority_rank_ = -1;
        bool active_ = false;
    };

}
