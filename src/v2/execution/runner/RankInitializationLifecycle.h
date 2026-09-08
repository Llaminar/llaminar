/**
 * @file RankInitializationLifecycle.h
 * @brief Typed, exception-safe lifecycle for rank-synchronous initialization.
 *
 * Distributed initialization must give every phase exactly one collective
 * exit. A rank-local return or exception is data for that collective; it is
 * never permission to unwind while a peer is still waiting. This file owns
 * that rule independently of model loading and GPU setup so it can be tested
 * without a model or accelerator.
 *
 * @author David Sanftenberg
 * @date August 2026
 */

#pragma once

#include <mpi.h>

#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace llaminar2
{
    /** @brief Rank-local result produced before phase consensus. */
    enum class RankInitializationLocalOutcome : std::uint8_t
    {
        Succeeded,
        ReturnedFailure,
        ThrewException,
    };

    /** @brief Result of authenticating one phase across its communicator. */
    enum class RankInitializationConsensusOutcome : std::uint8_t
    {
        AllRanksSucceeded,
        AtLeastOneRankFailed,
        PhaseIdentityMismatch,
        TransportFailure,
    };

    /**
     * @brief Stable identity of one ordered initialization phase.
     *
     * The ordinal detects reordered or skipped phases while the name makes the
     * resulting diagnostic useful. Both values are authenticated by the MPI
     * consensus implementation before a phase can advance.
     */
    struct RankInitializationPhaseIdentity
    {
        std::uint32_t ordinal{0};
        std::string_view name;
    };

    /** @brief Typed result returned by a phase-consensus transport. */
    struct RankInitializationConsensusResult
    {
        RankInitializationConsensusOutcome outcome{
            RankInitializationConsensusOutcome::TransportFailure};
        std::string detail;
    };

    /** @brief Terminal state of one rank-synchronous initialization phase. */
    enum class RankInitializationPhaseStatus : std::uint8_t
    {
        Succeeded,
        LocalStepReturnedFailure,
        LocalStepThrewException,
        PeerStepFailed,
        PhaseIdentityMismatch,
        ConsensusTransportFailed,
        InvalidConsensusResult,
    };

    /** @brief Complete, immutable outcome of one initialization phase. */
    struct RankInitializationPhaseResult
    {
        RankInitializationPhaseIdentity identity;
        RankInitializationLocalOutcome local_outcome{
            RankInitializationLocalOutcome::ReturnedFailure};
        RankInitializationPhaseStatus status{
            RankInitializationPhaseStatus::ConsensusTransportFailed};
        std::string detail;

        /** @return True only when every rank completed this exact phase. */
        [[nodiscard]] bool succeeded() const noexcept
        {
            return status == RankInitializationPhaseStatus::Succeeded;
        }

        /**
         * @brief Format a stable operator-facing failure diagnostic.
         * @param workflow Name of the enclosing workflow, such as initialization.
         * @return A diagnostic naming the phase and exact failure class.
         */
        [[nodiscard]] std::string diagnostic(
            std::string_view workflow) const;
    };

    /**
     * @brief Execute one local phase and always submit its outcome to consensus.
     *
     * `Step` must return a bool-like value. `Consensus` receives the immutable
     * phase identity and typed local outcome, then returns a
     * `RankInitializationConsensusResult`. Exceptions from either callable are
     * converted into typed terminal results; no rank-local step exception can
     * escape before the consensus callback has been invoked.
     */
    class RankInitializationLifecycle final
    {
    public:
        RankInitializationLifecycle() = delete;

        /**
         * @brief Run and authenticate one ordered initialization phase.
         * @tparam Step Callable returning success for the rank-local phase.
         * @tparam Consensus Callable authenticating identity and outcome globally.
         * @param identity Stable ordinal and name shared by every rank.
         * @param step Rank-local initialization work.
         * @param consensus Rank-wide terminal consensus operation.
         * @return Typed terminal phase result.
         */
        template <typename Step, typename Consensus>
        [[nodiscard]] static RankInitializationPhaseResult execute(
            RankInitializationPhaseIdentity identity,
            Step &&step,
            Consensus &&consensus)
        {
            static_assert(
                std::is_invocable_r_v<bool, Step>,
                "Rank initialization steps must return bool");
            static_assert(
                std::is_invocable_r_v<
                    RankInitializationConsensusResult,
                    Consensus,
                    RankInitializationPhaseIdentity,
                    RankInitializationLocalOutcome>,
                "Rank initialization consensus has the wrong signature");

            RankInitializationLocalOutcome local_outcome =
                RankInitializationLocalOutcome::ReturnedFailure;
            std::string local_detail;
            try
            {
                local_outcome = std::invoke(std::forward<Step>(step))
                                    ? RankInitializationLocalOutcome::Succeeded
                                    : RankInitializationLocalOutcome::ReturnedFailure;
            }
            catch (const std::exception &error)
            {
                local_outcome =
                    RankInitializationLocalOutcome::ThrewException;
                local_detail = error.what();
            }
            catch (...)
            {
                local_outcome =
                    RankInitializationLocalOutcome::ThrewException;
                local_detail = "non-standard exception";
            }

            RankInitializationConsensusResult consensus_result;
            try
            {
                /*
                 * This call is deliberately unconditional. It is the single
                 * lifecycle edge that prevents a local failure from escaping
                 * into teardown while another rank waits in initialization.
                 */
                consensus_result = std::invoke(
                    std::forward<Consensus>(consensus),
                    identity,
                    local_outcome);
            }
            catch (const std::exception &error)
            {
                consensus_result = {
                    .outcome =
                        RankInitializationConsensusOutcome::TransportFailure,
                    .detail = error.what(),
                };
            }
            catch (...)
            {
                consensus_result = {
                    .outcome =
                        RankInitializationConsensusOutcome::TransportFailure,
                    .detail = "non-standard consensus exception",
                };
            }

            RankInitializationPhaseResult result{
                .identity = identity,
                .local_outcome = local_outcome,
                .status =
                    RankInitializationPhaseStatus::ConsensusTransportFailed,
                .detail = std::move(consensus_result.detail),
            };

            switch (consensus_result.outcome)
            {
            case RankInitializationConsensusOutcome::AllRanksSucceeded:
                if (local_outcome !=
                    RankInitializationLocalOutcome::Succeeded)
                {
                    result.status =
                        RankInitializationPhaseStatus::InvalidConsensusResult;
                    result.detail =
                        "consensus reported success for a failed local phase";
                }
                else
                {
                    result.status =
                        RankInitializationPhaseStatus::Succeeded;
                }
                break;

            case RankInitializationConsensusOutcome::AtLeastOneRankFailed:
                if (local_outcome ==
                    RankInitializationLocalOutcome::Succeeded)
                {
                    result.status =
                        RankInitializationPhaseStatus::PeerStepFailed;
                }
                else if (local_outcome ==
                         RankInitializationLocalOutcome::ThrewException)
                {
                    result.status =
                        RankInitializationPhaseStatus::LocalStepThrewException;
                    result.detail = std::move(local_detail);
                }
                else
                {
                    result.status =
                        RankInitializationPhaseStatus::LocalStepReturnedFailure;
                }
                break;

            case RankInitializationConsensusOutcome::PhaseIdentityMismatch:
                result.status =
                    RankInitializationPhaseStatus::PhaseIdentityMismatch;
                break;

            case RankInitializationConsensusOutcome::TransportFailure:
                result.status =
                    RankInitializationPhaseStatus::ConsensusTransportFailed;
                break;
            }

            return result;
        }
    };

    /**
     * @brief Authenticate one initialization phase over an MPI communicator.
     *
     * One fixed-size all-gather carries the phase ordinal, exact phase name,
     * and typed local outcome from every rank. This both computes the global
     * success result and rejects peers that skipped or reordered a phase. The
     * operation owns no communicator and performs no barrier or teardown.
     */
    class MPIRankInitializationConsensus final
    {
    public:
        MPIRankInitializationConsensus() = delete;

        /**
         * @brief Reach one exact phase terminal across `communicator`.
         * @param communicator Live communicator shared by every participant.
         * @param identity Ordered phase identity expected on every rank.
         * @param local_outcome Typed result of this rank's local work.
         * @return Global success, failure, identity mismatch, or transport error.
         */
        [[nodiscard]] static RankInitializationConsensusResult reach(
            MPI_Comm communicator,
            RankInitializationPhaseIdentity identity,
            RankInitializationLocalOutcome local_outcome);
    };

} // namespace llaminar2
