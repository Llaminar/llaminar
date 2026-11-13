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
#include "interfaces/IMPIContext.h"
#include <mpi.h>
#include <iostream>
#include <functional>
#include <exception>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>
namespace llaminar2::test::parity {
    /**
     * @brief Agree exact original audit layer IDs before any variable-size coverage sums.
     * @param context Same original coordination context used by the audits.
     * @param layers Actual prepared main/predictor layer inventory, possibly empty when inapplicable.
     * @throws std::logic_error If any rank has invalid/different counts or IDs.
     * @note Only passive integer metadata is exchanged; no source hashes,
     * payloads, owner reconstruction or new communicator are involved.
     */
    inline void requireCoordinatedPassiveEvidenceLayers(const IMPIContext& context,std::span<const int> layers)
    {
        struct Header { std::uint64_t count=0; std::uint64_t valid=0; };
        const int population=context.world_size();
        if(population<=0)throw std::logic_error("Passive layer audit has no original rank population");
        const Header local{layers.size(),
            layers.size()<=std::size_t(std::numeric_limits<int>::max())/sizeof(int) &&
            std::is_sorted(layers.begin(),layers.end()) &&
            std::adjacent_find(layers.begin(),layers.end())==layers.end() &&
            (layers.empty() || layers.front()>=0)?1u:0u};
        std::vector<Header> headers(static_cast<std::size_t>(population));
        context.allgather_bytes(&local,headers.data(),sizeof(Header));
        for(const auto& header:headers)
            if(!header.valid || header.count!=headers.front().count)
                throw std::logic_error("Original ranks disagree on passive audit layer count or validity");
        const auto count=static_cast<std::size_t>(headers.front().count);
        if(count==0)return;
        if(count>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(population))
            throw std::logic_error("Passive audit layer inventory extent overflows");
        std::vector<int> original_ids(count*static_cast<std::size_t>(population));
        context.allgather_bytes(layers.data(),original_ids.data(),count*sizeof(int));
        for(int rank=1;rank<population;++rank)
            if(!std::equal(original_ids.begin(),original_ids.begin()+count,
                original_ids.begin()+static_cast<std::size_t>(rank)*count))
                throw std::logic_error("Original ranks disagree on exact passive audit layer IDs");
    }

    /**
     * @brief One scoped obligation to defer collective evidence until worker-loop closure.
     * @note This records test lifecycle only, never inference state or readiness.
     */
    class CoordinatedPassiveEvidenceObligation
    {
    public:
        /** @brief Keep one stable address for the session callback's lifecycle owner. */
        CoordinatedPassiveEvidenceObligation() = default;
        CoordinatedPassiveEvidenceObligation(const CoordinatedPassiveEvidenceObligation &) = delete;
        CoordinatedPassiveEvidenceObligation &operator=(const CoordinatedPassiveEvidenceObligation &) = delete;
        CoordinatedPassiveEvidenceObligation(CoordinatedPassiveEvidenceObligation &&) = delete;
        CoordinatedPassiveEvidenceObligation &operator=(CoordinatedPassiveEvidenceObligation &&) = delete;

        /** @brief The only legal progression of the deferred passive audit. */
        enum class Phase { Unbound, Serving, Auditing, Complete };
        /**
         * @brief Bind one nonempty audit to the existing coordinated session.
         * @param audit Same ordered collective audit on every original rank.
         * @return Session-owned callback, callable exactly once after command closure.
         * @throws std::logic_error For empty or repeated binding/callback execution.
         */
        std::function<void()> bind(std::function<void()> audit)
        {
            if (!audit || phase_ != Phase::Unbound)
                throw std::logic_error("Passive evidence needs one original session binding");
            phase_=Phase::Serving;
            return [this,audit=std::move(audit)] {
                requireServing();
                phase_=Phase::Auditing;
                try { audit(); }
                catch (...) { phase_=Phase::Complete; throw; }
                phase_=Phase::Complete;
            };
        }
        /** @brief Reject an unbound, premature or already completed deferral. */
        void requireServing() const
        {
            if(phase_!=Phase::Serving)
                throw std::logic_error("Collective evidence has no live deferred session obligation");
        }
        /** @brief Reject teardown with an audit bound but never completed. */
        void requireCompletedIfBound() const
        {
            if(phase_!=Phase::Unbound && phase_!=Phase::Complete)
                throw std::logic_error("Coordinated passive evidence was omitted before retirement");
        }
    private:
        Phase phase_=Phase::Unbound; ///< This fixture's single passive-audit lifecycle.
    };

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
        /** @brief Passive all-rank audit after command closure, while original diagnostics remain live. */
        using PassiveEvidenceAudit = std::function<void()>;
        /**
         * @brief Admit all ranks to one coordinated serving session.
         *
         * @param runner Fully initialized production orchestration runner.
         * @param passive_evidence Optional same-order audit on every original rank,
         * after the worker loop closes and before original runner teardown.
         * @throws std::runtime_error when MPI is unavailable or the resolved
         *         continuation authority is outside the active world.
         */
        explicit ScopedMoEParityCoordinatedSession(
            IOrchestrationRunner &runner,
            PassiveEvidenceAudit passive_evidence = {})
            : runner_(runner), passive_evidence_(std::move(passive_evidence))
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
            // No peer remains in the inference command receive now. Consume
            // this obligation once, before either original model is destroyed.
            // An audit failure must not strand teardown or replay its collectives.
            std::exception_ptr audit_failure;
            auto audit = std::exchange(passive_evidence_, PassiveEvidenceAudit{});
            try { if (audit) audit(); }
            catch (...) { audit_failure = std::current_exception(); }
            if(audit)
            {
                // Every callback completes the same collective audit sequence.
                // Join terminal errors before peers enter the retirement fence.
                int failed=audit_failure?1:0, any_failed=0;
                if(MPI_Allreduce(&failed,&any_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD)!=MPI_SUCCESS)
                    throw std::runtime_error("Coordinated passive evidence failure consensus failed");
                if(any_failed && !audit_failure)
                    audit_failure=std::make_exception_ptr(std::runtime_error(
                        "Original peer failed coordinated passive evidence"));
            }
            runner_.setMPICoordinatedMode(false);
            std::exception_ptr retirement_failure;
            try { runner_.shutdown(); }
            catch (...) { retirement_failure=std::current_exception(); }
            // A retirement attempt is consumed even when it throws. Never let
            // the authority destructor retry owner destruction or reenter MPI.
            active_=false;
            int local_retirement_failed=retirement_failure?1:0, any_retirement_failed=0;
            if(MPI_Allreduce(&local_retirement_failed,&any_retirement_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD)!=MPI_SUCCESS)
                throw std::runtime_error("Coordinated parity retirement failure consensus failed");
            if(any_retirement_failed && !retirement_failure)
                retirement_failure=std::make_exception_ptr(std::runtime_error(
                    "Original peer failed its single coordinated retirement attempt"));
            if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Coordinated MoE parity retirement barrier failed");
            }
            if (retirement_failure) std::rethrow_exception(retirement_failure);
            if (audit_failure) std::rethrow_exception(audit_failure);
        }

        IOrchestrationRunner &runner_;
        PassiveEvidenceAudit passive_evidence_; ///< Consumed exactly once at original command closure.
        int world_rank_ = -1;
        int world_size_ = 0;
        int authority_rank_ = -1;
        bool active_ = false;
    };

}
