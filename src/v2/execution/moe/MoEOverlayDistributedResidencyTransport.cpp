/**
 * @file MoEOverlayDistributedResidencyTransport.cpp
 * @brief Non-blocking global agreement for heterogeneous residency waves.
 */

#include "MoEOverlayDistributedResidencyTransport.h"

#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    /** @brief Atomic counters shared by facade and independently owned waves. */
    struct MoEOverlayDistributedResidencyTransportSharedStats
    {
        std::atomic<std::uint64_t> waves_started{0};
        std::atomic<std::uint64_t> local_stage_started{0};
        std::atomic<std::uint64_t> local_stage_deferred{0};
        std::atomic<std::uint64_t> local_stage_failed{0};
        std::atomic<std::uint64_t> reservation_consensus_started{0};
        std::atomic<std::uint64_t> reservation_consensus_ready{0};
        std::atomic<std::uint64_t> reservation_consensus_deferred{0};
        std::atomic<std::uint64_t> reservation_consensus_failed{0};
        std::atomic<std::uint64_t> stage_consensus_started{0};
        std::atomic<std::uint64_t> stage_consensus_ready{0};
        std::atomic<std::uint64_t> stage_consensus_deferred{0};
        std::atomic<std::uint64_t> stage_consensus_failed{0};
        std::atomic<std::uint64_t> preparation_consensus_started{0};
        std::atomic<std::uint64_t> preparation_consensus_ready{0};
        std::atomic<std::uint64_t> preparation_consensus_failed{0};
        std::atomic<std::uint64_t> local_preparation_begin_failed{0};
        std::atomic<std::uint64_t> publication_consensus_started{0};
        std::atomic<std::uint64_t> publication_consensus_ready{0};
        std::atomic<std::uint64_t> publication_consensus_failed{0};
        std::atomic<std::uint64_t> local_publication_begin_failed{0};
        std::atomic<std::uint64_t> retirement_admission_consensus_started{0};
        std::atomic<std::uint64_t> retirement_admission_consensus_waiting{0};
        std::atomic<std::uint64_t> retirement_admission_consensus_ready{0};
        std::atomic<std::uint64_t> retirement_admission_consensus_failed{0};
        std::atomic<std::uint64_t> retirement_consensus_started{0};
        std::atomic<std::uint64_t> retirement_consensus_waiting{0};
        std::atomic<std::uint64_t> retirement_consensus_ready{0};
        std::atomic<std::uint64_t> retirement_consensus_failed{0};
        std::atomic<std::uint64_t> waves_published{0};
        std::atomic<std::uint64_t> waves_aborted{0};
        std::atomic<std::uint64_t> abort_cleanups_completed{0};
        /** Candidate epoch whose first local capacity deferral was diagnosed. */
        std::atomic<std::uint64_t> last_logged_local_deferred_candidate_epoch{0};
    };

    namespace
    {
        /** Stable local failure codes carried by the fixed-layout vote ABI. */
        enum class DistributedResidencyError : int
        {
            LocalReservation = 2101,
            LocalStagePoll = 2102,
            LocalPreparationBegin = 2103,
            LocalPreparationPoll = 2104,
            LocalPublicationBegin = 2105,
            LocalPublicationPoll = 2106,
        };

        /** @brief Assign an optional caller diagnostic. */
        void setDistributedError(std::string *error, const std::string &message)
        {
            if (error)
                *error = message;
        }

        /**
         * @brief One process-local wave fenced by two all-rank consensuses.
         *
         * The local physical wave may be absent when its start deferred or
         * failed. That is intentional: the wrapper still owns a protocol wave
         * and contributes the corresponding stage vote. Consensus polling is
         * performed only by the maintenance worker and never waits.
         */
        class DistributedResidencyWave final
            : public IMoEOverlayResidencyWave
        {
        public:
            /**
             * @brief Take local work and bind it to one complete identity.
             * @param transaction Immutable transaction shared by all ranks.
             * @param local_start Process-local start result, including cleanup.
             * @param consensus Model-lifetime asynchronous all-rank lane.
             * @param stats Lifetime-safe process-local evidence counters.
             * @param perf_device Stable evidence topology/model label.
             */
            DistributedResidencyWave(
                const MoEOverlayResidencyTransaction &transaction,
                MoEOverlayResidencyStageStart local_start,
                std::shared_ptr<IMoEOverlayResidencyConsensusLane> consensus,
                std::shared_ptr<
                    MoEOverlayDistributedResidencyTransportSharedStats> stats,
                std::string perf_device)
                : local_status_(local_start.status),
                  local_wave_(std::move(local_start.wave)),
                  local_cleanup_wave_(std::move(local_start.cleanup_wave)),
                  local_start_error_(std::move(local_start.error)),
                  consensus_(std::move(consensus)),
                  protocol_({
                      .identity =
                          makeMoEOverlayDistributedResidencyWaveIdentity(
                              transaction),
                      .local_world_rank = consensus_->worldRank(),
                      .world_size = consensus_->worldSize(),
                  }),
                  stats_(std::move(stats)),
                  perf_device_(std::move(perf_device)),
                  expected_epoch_(transaction.expected_epoch),
                  candidate_epoch_(transaction.candidate->epoch)
            {
                /* Normalize malformed local ownership into a global failure vote. */
                const bool started_shape =
                    local_status_ ==
                        MoEOverlayResidencyStageStartStatus::Started &&
                    local_wave_ != nullptr && local_cleanup_wave_ == nullptr;
                const bool deferred_shape =
                    local_status_ ==
                        MoEOverlayResidencyStageStartStatus::Deferred &&
                    local_wave_ == nullptr && local_cleanup_wave_ == nullptr;
                const bool failed_shape =
                    local_status_ ==
                        MoEOverlayResidencyStageStartStatus::Failed &&
                    local_wave_ == nullptr;
                if (!started_shape && !deferred_shape && !failed_shape)
                {
                    local_status_ =
                        MoEOverlayResidencyStageStartStatus::Failed;
                    if (local_start_error_.empty())
                    {
                        local_start_error_ =
                            "Local ExpertOverlay transport returned invalid wave ownership";
                    }
                }
                if (local_status_ ==
                        MoEOverlayResidencyStageStartStatus::Failed &&
                    local_start_error_.empty())
                {
                    local_start_error_ =
                        "Local ExpertOverlay transport failed to start staging";
                }
                if (local_status_ ==
                    MoEOverlayResidencyStageStartStatus::Deferred)
                {
                    /*
                     * Global backpressure is expected, but an indefinitely
                     * retained slot must name the process-local cause. Emit
                     * once per candidate epoch so a retry loop remains quiet
                     * while its exact participant/layer diagnostic survives.
                     */
                    const auto previous =
                        stats_->last_logged_local_deferred_candidate_epoch
                            .exchange(
                                candidate_epoch_,
                                std::memory_order_acq_rel);
                    if (previous != candidate_epoch_)
                    {
                        LOG_INFO(
                            "[ExpertOverlay][Residency] Local reservation deferred"
                            << " world_rank=" << consensus_->worldRank()
                            << " expected_epoch=" << expected_epoch_
                            << " candidate_epoch=" << candidate_epoch_
                            << " detail="
                            << (local_start_error_.empty()
                                    ? "unspecified local capacity pressure"
                                    : local_start_error_));
                    }
                }
            }

            /** @brief Prove global admission, then poll and vote physical staging. */
            MoEOverlayResidencyWaveProgress pollStage(
                std::string *error) noexcept override
            {
                if (aborted_ || published_)
                {
                    setDistributedError(
                        error,
                        "Distributed residency staging was polled after its terminal lifecycle edge");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (stage_ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (stage_deferred_)
                    return MoEOverlayResidencyWaveProgress::Deferred;
                if (stage_failed_)
                {
                    setDistributedError(error, terminal_error_);
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                if (!stage_dispatch_started_)
                {
                    /*
                     * Global reservation consensus is part of staging.  Start
                     * its interval when the maintenance worker first releases
                     * the prepared protocol, rather than while calibration is
                     * waiting for a matching live inference ticket.
                     */
                    wave_started_at_ = std::chrono::steady_clock::now();
                    stage_dispatch_started_ = true;
                }

                /*
                 * No transfer operation may be polled before every rank proves
                 * complete slot, source-pin, and lane admission. Otherwise a
                 * source send could wait forever on a destination that deferred.
                 */
                if (!reservation_ready_)
                {
                    if (!reservation_consensus_started_)
                    {
                        auto decision =
                            MoEOverlayDistributedResidencyVoteDecision::Ready;
                        int error_code = 0;
                        std::string diagnostic;
                        if (local_status_ ==
                            MoEOverlayResidencyStageStartStatus::Deferred)
                        {
                            decision =
                                MoEOverlayDistributedResidencyVoteDecision::Deferred;
                        }
                        else if (local_status_ ==
                                 MoEOverlayResidencyStageStartStatus::Failed)
                        {
                            decision =
                                MoEOverlayDistributedResidencyVoteDecision::Failed;
                            error_code = static_cast<int>(
                                DistributedResidencyError::LocalReservation);
                            diagnostic = local_start_error_;
                        }

                        try
                        {
                            const auto vote = protocol_.makeLocalVote(
                                decision,
                                error_code,
                                diagnostic);
                            if (!consensus_->begin(vote, &terminal_error_))
                            {
                                stage_failed_ = true;
                                if (terminal_error_.empty())
                                {
                                    terminal_error_ =
                                        "Failed to begin distributed ExpertOverlay reservation consensus";
                                }
                                stats_->reservation_consensus_failed.fetch_add(
                                    1,
                                    std::memory_order_relaxed);
                                setDistributedError(error, terminal_error_);
                                return MoEOverlayResidencyWaveProgress::Failed;
                            }
                        }
                        catch (const std::exception &exception)
                        {
                            stage_failed_ = true;
                            terminal_error_ = exception.what();
                            stats_->reservation_consensus_failed.fetch_add(
                                1,
                                std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayResidencyWaveProgress::Failed;
                        }
                        reservation_consensus_started_ = true;
                        stats_->reservation_consensus_started.fetch_add(
                            1,
                            std::memory_order_relaxed);
                    }

                    std::vector<MoEOverlayDistributedResidencyVote> votes;
                    const auto consensus_progress =
                        consensus_->poll(&votes, &terminal_error_);
                    if (consensus_progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        return consensus_progress;
                    }
                    if (consensus_progress ==
                        MoEOverlayResidencyWaveProgress::Failed)
                    {
                        stage_failed_ = true;
                        stats_->reservation_consensus_failed.fetch_add(
                            1,
                            std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return consensus_progress;
                    }

                    std::string protocol_error;
                    if (!protocol_.acceptConsensus(votes, &protocol_error))
                    {
                        if (protocol_.state() ==
                            MoEOverlayDistributedResidencyProtocolState::Deferred)
                        {
                            stage_deferred_ = true;
                            stats_->reservation_consensus_deferred.fetch_add(
                                1,
                                std::memory_order_relaxed);
                            if (error)
                                error->clear();
                            recordCounter("distributed_reservation_deferred");
                            return MoEOverlayResidencyWaveProgress::Deferred;
                        }
                        stage_failed_ = true;
                        terminal_error_ = protocol_error.empty()
                                              ? "Distributed ExpertOverlay reservation consensus failed"
                                              : std::move(protocol_error);
                        /*
                         * The wire vote deliberately carries only an
                         * authenticated fingerprint.  Preserve the originating
                         * rank's process-local diagnostic as well so an
                         * operator can identify the failed pool, endpoint, or
                         * lane without weakening the fixed-layout protocol.
                         */
                        appendOriginatingLocalDiagnostic(
                            "reservation", local_start_error_);
                        stats_->reservation_consensus_failed.fetch_add(
                            1,
                            std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        recordCounter("distributed_reservation_failed");
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }

                    reservation_ready_ = true;
                    stats_->reservation_consensus_ready.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    recordCounter("distributed_reservation_ready");
                }

                if (!stage_consensus_started_)
                {
                    auto decision =
                        MoEOverlayDistributedResidencyVoteDecision::Ready;
                    int error_code = 0;
                    std::string diagnostic;
                    std::string local_error;
                    const auto progress = local_wave_->pollStage(&local_error);
                    if (progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        return progress;
                    }
                    if (progress ==
                        MoEOverlayResidencyWaveProgress::Deferred)
                    {
                        decision =
                            MoEOverlayDistributedResidencyVoteDecision::Deferred;
                    }
                    else if (progress ==
                             MoEOverlayResidencyWaveProgress::Failed)
                    {
                        decision =
                            MoEOverlayDistributedResidencyVoteDecision::Failed;
                        error_code = static_cast<int>(
                            DistributedResidencyError::LocalStagePoll);
                        local_stage_error_ = local_error.empty()
                                                 ? "Local ExpertOverlay staging failed"
                                                 : std::move(local_error);
                        diagnostic = local_stage_error_;
                    }

                    try
                    {
                        const auto vote = protocol_.makeLocalVote(
                            decision,
                            error_code,
                            diagnostic);
                        if (!consensus_->begin(vote, &terminal_error_))
                        {
                            stage_failed_ = true;
                            if (terminal_error_.empty())
                            {
                                terminal_error_ =
                                    "Failed to begin distributed ExpertOverlay stage consensus";
                            }
                            stats_->stage_consensus_failed.fetch_add(
                                1,
                                std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayResidencyWaveProgress::Failed;
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        stage_failed_ = true;
                        terminal_error_ = exception.what();
                        stats_->stage_consensus_failed.fetch_add(
                            1,
                            std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    stage_consensus_started_ = true;
                    stats_->stage_consensus_started.fetch_add(
                        1,
                        std::memory_order_relaxed);
                }

                std::vector<MoEOverlayDistributedResidencyVote> votes;
                const auto consensus_progress =
                    consensus_->poll(&votes, &terminal_error_);
                if (consensus_progress ==
                    MoEOverlayResidencyWaveProgress::Pending)
                {
                    return consensus_progress;
                }
                if (consensus_progress ==
                    MoEOverlayResidencyWaveProgress::Failed)
                {
                    stage_failed_ = true;
                    stats_->stage_consensus_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    return consensus_progress;
                }

                std::string protocol_error;
                if (!protocol_.acceptConsensus(votes, &protocol_error))
                {
                    if (protocol_.state() ==
                        MoEOverlayDistributedResidencyProtocolState::Deferred)
                    {
                        stage_deferred_ = true;
                        stats_->stage_consensus_deferred.fetch_add(
                            1,
                            std::memory_order_relaxed);
                        if (error)
                            error->clear();
                        recordCounter("distributed_stage_deferred");
                        return MoEOverlayResidencyWaveProgress::Deferred;
                    }
                    stage_failed_ = true;
                    terminal_error_ = protocol_error.empty()
                                          ? "Distributed ExpertOverlay stage consensus failed"
                                          : std::move(protocol_error);
                    appendOriginatingLocalDiagnostic(
                        "staging", local_stage_error_);
                    stats_->stage_consensus_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    recordCounter("distributed_stage_failed");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                stage_ready_ = true;
                const auto finished_at = std::chrono::steady_clock::now();
                const auto begin_count =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        wave_started_at_.time_since_epoch()).count();
                const auto end_count =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        finished_at.time_since_epoch()).count();
                stage_interval_ = {
                    .begin_steady_nanoseconds =
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(1, begin_count)),
                    .end_steady_nanoseconds =
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(
                                begin_count + 1,
                                end_count)),
                };
                stats_->stage_consensus_ready.fetch_add(
                    1,
                    std::memory_order_relaxed);
                recordCounter("distributed_stage_ready");
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return this rank's begin-to-global-ready interval. */
            [[nodiscard]] std::optional<MoEOverlayResidencyWaveInterval>
            completedStageInterval() const noexcept override
            {
                if (!stage_ready_ || !stage_interval_.valid())
                    return std::nullopt;
                return stage_interval_;
            }

            /** @brief Start local preparation but defer every outcome to its vote. */
            bool beginPrepare(std::string *error) noexcept override
            {
                if (!stage_ready_ || prepare_started_ || aborted_ || published_)
                {
                    setDistributedError(
                        error,
                        "Distributed residency preparation has an invalid lifecycle state");
                    return false;
                }

                prepare_started_ = true;
                std::string local_error;
                if (!local_wave_->beginPrepare(&local_error))
                {
                    local_prepare_begin_failed_ = true;
                    local_prepare_error_ = local_error.empty()
                                              ? "Local inactive-bank preparation failed to start"
                                              : std::move(local_error);
                    stats_->local_preparation_begin_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                }
                /*
                 * Returning true is essential: a local enqueue failure must be
                 * exchanged as a Failed vote while every peer enters the same
                 * preparation collective. Returning false here would strand peers.
                 */
                if (error)
                    error->clear();
                return true;
            }

            /** @brief Poll local preparation, then the all-rank preparation vote. */
            MoEOverlayResidencyWaveProgress pollPrepare(
                std::string *error) noexcept override
            {
                if (!prepare_started_ || aborted_ || published_)
                {
                    setDistributedError(
                        error,
                        "Distributed residency preparation was polled in an invalid lifecycle state");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (prepare_ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (prepare_failed_)
                {
                    setDistributedError(error, terminal_error_);
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                if (!prepare_consensus_started_)
                {
                    auto decision =
                        MoEOverlayDistributedResidencyVoteDecision::Ready;
                    int error_code = 0;
                    std::string diagnostic;
                    if (local_prepare_begin_failed_)
                    {
                        decision =
                            MoEOverlayDistributedResidencyVoteDecision::Failed;
                        error_code = static_cast<int>(
                            DistributedResidencyError::LocalPreparationBegin);
                        diagnostic = local_prepare_error_;
                    }
                    else
                    {
                        std::string local_error;
                        const auto progress = local_wave_->pollPrepare(&local_error);
                        if (progress ==
                            MoEOverlayResidencyWaveProgress::Pending)
                        {
                            return progress;
                        }
                        if (progress !=
                            MoEOverlayResidencyWaveProgress::Ready)
                        {
                            decision =
                                MoEOverlayDistributedResidencyVoteDecision::Failed;
                            error_code = static_cast<int>(
                                DistributedResidencyError::LocalPreparationPoll);
                            local_prepare_error_ = local_error.empty()
                                                       ? "Local inactive-bank preparation failed"
                                                       : std::move(local_error);
                            diagnostic = local_prepare_error_;
                        }
                    }

                    try
                    {
                        const auto vote = protocol_.makeLocalVote(
                            decision,
                            error_code,
                            diagnostic);
                        if (!consensus_->begin(vote, &terminal_error_))
                        {
                            prepare_failed_ = true;
                            if (terminal_error_.empty())
                            {
                                terminal_error_ =
                                    "Failed to begin distributed ExpertOverlay preparation consensus";
                            }
                            stats_->preparation_consensus_failed.fetch_add(
                                1,
                                std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayResidencyWaveProgress::Failed;
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        prepare_failed_ = true;
                        terminal_error_ = exception.what();
                        stats_->preparation_consensus_failed.fetch_add(
                            1,
                            std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    prepare_consensus_started_ = true;
                    stats_->preparation_consensus_started.fetch_add(
                        1,
                        std::memory_order_relaxed);
                }

                std::vector<MoEOverlayDistributedResidencyVote> votes;
                const auto consensus_progress =
                    consensus_->poll(&votes, &terminal_error_);
                if (consensus_progress ==
                    MoEOverlayResidencyWaveProgress::Pending)
                {
                    return consensus_progress;
                }
                if (consensus_progress ==
                    MoEOverlayResidencyWaveProgress::Failed)
                {
                    prepare_failed_ = true;
                    stats_->preparation_consensus_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    return consensus_progress;
                }

                std::string protocol_error;
                if (!protocol_.acceptConsensus(votes, &protocol_error))
                {
                    prepare_failed_ = true;
                    terminal_error_ = protocol_error.empty()
                                          ? "Distributed ExpertOverlay preparation consensus failed"
                                          : std::move(protocol_error);
                    appendOriginatingLocalDiagnostic(
                        "inactive-bank preparation",
                        local_prepare_error_);
                    stats_->preparation_consensus_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    recordCounter("distributed_preparation_failed");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                prepare_ready_ = true;
                stats_->preparation_consensus_ready.fetch_add(
                    1,
                    std::memory_order_relaxed);
                recordCounter("distributed_preparation_ready");
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Start local selector publication and retain its global vote. */
            bool beginPublication(std::string *error) noexcept override
            {
                if (!prepare_ready_ || publication_started_ || aborted_ ||
                    published_)
                {
                    setDistributedError(
                        error,
                        "Distributed residency publication has an invalid lifecycle state");
                    return false;
                }

                publication_started_ = true;
                std::string local_error;
                if (!local_wave_->beginPublication(&local_error))
                {
                    local_publication_begin_failed_ = true;
                    local_publication_error_ = local_error.empty()
                                                   ? "Local runtime publication failed to start"
                                                   : std::move(local_error);
                    stats_->local_publication_begin_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                }
                /* Every rank must still enter the publication vote. */
                if (error)
                    error->clear();
                return true;
            }

            /** @brief Poll local selectors, then prove publication on all ranks. */
            MoEOverlayResidencyWaveProgress pollPublication(
                std::string *error) noexcept override
            {
                if (!publication_started_ || aborted_ || published_)
                {
                    setDistributedError(
                        error,
                        "Distributed residency publication was polled in an invalid lifecycle state");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (publication_ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (publication_failed_)
                {
                    setDistributedError(error, terminal_error_);
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                if (!publication_consensus_started_)
                {
                    auto decision =
                        MoEOverlayDistributedResidencyVoteDecision::Ready;
                    int error_code = 0;
                    std::string diagnostic;
                    if (local_publication_begin_failed_)
                    {
                        decision =
                            MoEOverlayDistributedResidencyVoteDecision::Failed;
                        error_code = static_cast<int>(
                            DistributedResidencyError::LocalPublicationBegin);
                        diagnostic = local_publication_error_;
                    }
                    else
                    {
                        std::string local_error;
                        const auto progress =
                            local_wave_->pollPublication(&local_error);
                        if (progress == MoEOverlayResidencyWaveProgress::Pending)
                            return progress;
                        if (progress != MoEOverlayResidencyWaveProgress::Ready)
                        {
                            decision =
                                MoEOverlayDistributedResidencyVoteDecision::Failed;
                            error_code = static_cast<int>(
                                DistributedResidencyError::LocalPublicationPoll);
                            local_publication_error_ = local_error.empty()
                                                           ? "Local runtime publication failed"
                                                           : std::move(local_error);
                            diagnostic = local_publication_error_;
                        }
                    }

                    try
                    {
                        const auto vote = protocol_.makeLocalVote(
                            decision, error_code, diagnostic);
                        if (!consensus_->begin(vote, &terminal_error_))
                        {
                            publication_failed_ = true;
                            if (terminal_error_.empty())
                            {
                                terminal_error_ =
                                    "Failed to begin distributed ExpertOverlay publication consensus";
                            }
                            stats_->publication_consensus_failed.fetch_add(
                                1, std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayResidencyWaveProgress::Failed;
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        publication_failed_ = true;
                        terminal_error_ = exception.what();
                        stats_->publication_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    publication_consensus_started_ = true;
                    stats_->publication_consensus_started.fetch_add(
                        1, std::memory_order_relaxed);
                }

                std::vector<MoEOverlayDistributedResidencyVote> votes;
                const auto progress =
                    consensus_->poll(&votes, &terminal_error_);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                    return progress;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                {
                    publication_failed_ = true;
                    stats_->publication_consensus_failed.fetch_add(
                        1, std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    return progress;
                }

                std::string protocol_error;
                if (!protocol_.acceptConsensus(votes, &protocol_error))
                {
                    publication_failed_ = true;
                    terminal_error_ = protocol_error.empty()
                                          ? "Distributed ExpertOverlay publication consensus failed"
                                          : std::move(protocol_error);
                    appendOriginatingLocalDiagnostic(
                        "runtime publication",
                        local_publication_error_);
                    stats_->publication_consensus_failed.fetch_add(
                        1, std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    recordCounter("distributed_publication_failed");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                publication_ready_ = true;
                stats_->publication_consensus_ready.fetch_add(
                    1, std::memory_order_relaxed);
                recordCounter("distributed_publication_ready");
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Abort all local work and retain an in-flight vote to drain. */
            void abortStaged() noexcept override
            {
                if (aborted_ || published_)
                    return;
                aborted_ = true;
                protocol_.abort();
                if (local_wave_)
                    local_wave_->abortStaged();
                if (local_cleanup_wave_)
                    local_cleanup_wave_->abortStaged();
                stats_->waves_aborted.fetch_add(1, std::memory_order_relaxed);
                recordCounter("distributed_waves_aborted");
            }

            /** @brief Drain the vote request and every local abort event. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (!aborted_)
                {
                    setDistributedError(
                        error,
                        "Distributed residency abort was polled before abortStaged()");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (abort_ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;

                bool pending = false;
                if (!consensus_->idle())
                {
                    std::vector<MoEOverlayDistributedResidencyVote> discarded;
                    std::string consensus_error;
                    const auto progress =
                        consensus_->poll(&discarded, &consensus_error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    {
                        setDistributedError(error, consensus_error);
                        return progress;
                    }
                    pending = progress == MoEOverlayResidencyWaveProgress::Pending;
                }

                if (local_wave_ && !local_abort_ready_)
                {
                    std::string local_error;
                    const auto progress = local_wave_->pollAbort(&local_error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    {
                        setDistributedError(error, local_error);
                        return progress;
                    }
                    local_abort_ready_ =
                        progress == MoEOverlayResidencyWaveProgress::Ready;
                    pending = pending || !local_abort_ready_;
                }
                else if (!local_wave_)
                {
                    local_abort_ready_ = true;
                }

                if (local_cleanup_wave_ && !cleanup_abort_ready_)
                {
                    std::string cleanup_error;
                    const auto progress =
                        local_cleanup_wave_->pollAbort(&cleanup_error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    {
                        setDistributedError(error, cleanup_error);
                        return progress;
                    }
                    cleanup_abort_ready_ =
                        progress == MoEOverlayResidencyWaveProgress::Ready;
                    pending = pending || !cleanup_abort_ready_;
                }
                else if (!local_cleanup_wave_)
                {
                    cleanup_abort_ready_ = true;
                }

                abort_ready_ = consensus_->idle() && local_abort_ready_ &&
                               cleanup_abort_ready_;
                if (!abort_ready_ || pending)
                    return MoEOverlayResidencyWaveProgress::Pending;

                stats_->abort_cleanups_completed.fetch_add(
                    1,
                    std::memory_order_relaxed);
                recordCounter("distributed_abort_cleanup_ready");
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Record the authority's exact successful publication edge. */
            void markAuthorityPublished() noexcept override
            {
                if (published_ || aborted_ || !publication_ready_ ||
                    protocol_.state() !=
                        MoEOverlayDistributedResidencyProtocolState::
                            ReadyForAuthorityPublication)
                {
                    LOG_ERROR(
                        "[DistributedResidencyWave] Invalid markPublished lifecycle");
                    std::terminate();
                }
                try
                {
                    protocol_.markAuthorityPublished();
                }
                catch (...)
                {
                    std::terminate();
                }
                published_ = true;
                stats_->waves_published.fetch_add(1, std::memory_order_relaxed);
                recordCounter("distributed_waves_published");
            }

            /** @brief Exchange both ordered old-epoch grace-period barriers. */
            MoEOverlayRetirementFenceProgress pollRetirementFence(
                MoEOverlayLocalRetirementState local_state,
                std::string *error) noexcept override
            {
                if (retired_ || aborted_ || !published_)
                {
                    setDistributedError(
                        error,
                        "Distributed residency retirement fence has an invalid lifecycle state");
                    return MoEOverlayRetirementFenceProgress::Failed;
                }
                if (retirement_ready_)
                {
                    if (error)
                        error->clear();
                    return MoEOverlayRetirementFenceProgress::ReadyToRetire;
                }
                if (retirement_failed_)
                {
                    setDistributedError(error, terminal_error_);
                    return MoEOverlayRetirementFenceProgress::Failed;
                }

                /*
                 * Phase 5 is entered by every rank immediately after
                 * publication. A Waiting vote keeps the collective sequence
                 * aligned while current host readers or device-side producers
                 * drain. Unanimous readiness permits, but does not itself
                 * perform, the source-owned exact-admission close.
                 */
                if (local_state.admission ==
                    MoEOverlayRetirementAdmissionState::Open)
                {
                    if (retirement_admission_ready_)
                    {
                        if (error)
                            error->clear();
                        return MoEOverlayRetirementFenceProgress::
                            ReadyToCloseAdmission;
                    }

                    if (!retirement_admission_consensus_started_)
                    {
                        std::string local_error;
                        const auto local_progress =
                            local_wave_->pollRetirementFence(
                                local_state, &local_error);
                        if (local_progress ==
                            MoEOverlayRetirementFenceProgress::Failed)
                        {
                            retirement_failed_ = true;
                            terminal_error_ = local_error.empty()
                                                  ? "Local retirement-admission fence failed"
                                                  : std::move(local_error);
                            stats_->retirement_admission_consensus_failed
                                .fetch_add(1, std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayRetirementFenceProgress::Failed;
                        }
                        if (local_progress ==
                            MoEOverlayRetirementFenceProgress::ReadyToRetire)
                        {
                            retirement_failed_ = true;
                            terminal_error_ =
                                "Local retirement wave reported final readiness while exact admission remained open";
                            stats_->retirement_admission_consensus_failed
                                .fetch_add(1, std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayRetirementFenceProgress::Failed;
                        }

                        try
                        {
                            const auto vote = protocol_.makeLocalVote(
                                local_progress ==
                                        MoEOverlayRetirementFenceProgress::
                                            ReadyToCloseAdmission
                                    ? MoEOverlayDistributedResidencyVoteDecision::
                                          Ready
                                    : MoEOverlayDistributedResidencyVoteDecision::
                                          Waiting);
                            if (!consensus_->begin(vote, &terminal_error_))
                            {
                                retirement_failed_ = true;
                                if (terminal_error_.empty())
                                {
                                    terminal_error_ =
                                        "Failed to begin distributed ExpertOverlay retirement-admission consensus";
                                }
                                stats_->retirement_admission_consensus_failed
                                    .fetch_add(1, std::memory_order_relaxed);
                                setDistributedError(error, terminal_error_);
                                return MoEOverlayRetirementFenceProgress::Failed;
                            }
                        }
                        catch (const std::exception &exception)
                        {
                            retirement_failed_ = true;
                            terminal_error_ = exception.what();
                            stats_->retirement_admission_consensus_failed
                                .fetch_add(1, std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayRetirementFenceProgress::Failed;
                        }
                        retirement_admission_consensus_started_ = true;
                        stats_->retirement_admission_consensus_started.fetch_add(
                            1, std::memory_order_relaxed);
                        recordCounter(
                            "distributed_retirement_admission_consensus_started");
                    }

                    std::vector<MoEOverlayDistributedResidencyVote> votes;
                    const auto consensus_progress =
                        consensus_->poll(&votes, &terminal_error_);
                    if (consensus_progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        return MoEOverlayRetirementFenceProgress::Pending;
                    }
                    if (consensus_progress ==
                        MoEOverlayResidencyWaveProgress::Failed)
                    {
                        retirement_failed_ = true;
                        stats_->retirement_admission_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayRetirementFenceProgress::Failed;
                    }

                    std::string protocol_error;
                    if (!protocol_.acceptConsensus(votes, &protocol_error))
                    {
                        if (protocol_.state() ==
                            MoEOverlayDistributedResidencyProtocolState::Published)
                        {
                            retirement_admission_consensus_started_ = false;
                            stats_->retirement_admission_consensus_waiting
                                .fetch_add(
                                1,
                                std::memory_order_relaxed);
                            recordCounter(
                                "distributed_retirement_admission_consensus_waiting");
                            if (error)
                                error->clear();
                            return MoEOverlayRetirementFenceProgress::Pending;
                        }
                        retirement_failed_ = true;
                        terminal_error_ = protocol_error.empty()
                                              ? "Distributed ExpertOverlay retirement-admission consensus failed"
                                              : std::move(protocol_error);
                        stats_->retirement_admission_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        recordCounter(
                            "distributed_retirement_admission_consensus_failed");
                        return MoEOverlayRetirementFenceProgress::Failed;
                    }
                    if (protocol_.state() !=
                        MoEOverlayDistributedResidencyProtocolState::
                            ReadyToCloseRetirementAdmission)
                    {
                        retirement_failed_ = true;
                        terminal_error_ =
                            "Distributed ExpertOverlay retirement-admission consensus advanced to an invalid lifecycle state";
                        stats_->retirement_admission_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayRetirementFenceProgress::Failed;
                    }

                    retirement_admission_ready_ = true;
                    stats_->retirement_admission_consensus_ready.fetch_add(
                        1, std::memory_order_relaxed);
                    recordCounter(
                        "distributed_retirement_admission_consensus_ready");
                    if (error)
                        error->clear();
                    return MoEOverlayRetirementFenceProgress::
                        ReadyToCloseAdmission;
                }

                /*
                 * Phase 6 begins only after the authority has closed local
                 * exact admission. A reader that raced the close contributes
                 * Waiting until it releases; no new reader can enter, so a
                 * Ready vote is stable for the remainder of the wave.
                 */
                if (!retirement_admission_ready_)
                {
                    retirement_failed_ = true;
                    terminal_error_ =
                        "Distributed retirement admission closed before unanimous quiescence";
                    stats_->retirement_consensus_failed.fetch_add(
                        1, std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    return MoEOverlayRetirementFenceProgress::Failed;
                }
                if (!retirement_admission_marked_closed_)
                {
                    try
                    {
                        protocol_.markRetirementAdmissionClosed();
                    }
                    catch (const std::exception &exception)
                    {
                        retirement_failed_ = true;
                        terminal_error_ = exception.what();
                        stats_->retirement_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayRetirementFenceProgress::Failed;
                    }
                    retirement_admission_marked_closed_ = true;
                }

                if (!retirement_consensus_started_)
                {
                    std::string local_error;
                    const auto local_progress =
                        local_wave_->pollRetirementFence(
                            local_state, &local_error);
                    if (local_progress ==
                        MoEOverlayRetirementFenceProgress::Failed)
                    {
                        retirement_failed_ = true;
                        terminal_error_ = local_error.empty()
                                              ? "Local closed-admission retirement fence failed"
                                              : std::move(local_error);
                        stats_->retirement_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayRetirementFenceProgress::Failed;
                    }
                    if (local_progress ==
                        MoEOverlayRetirementFenceProgress::ReadyToCloseAdmission)
                    {
                        retirement_failed_ = true;
                        terminal_error_ =
                            "Local retirement wave requested an admission close after admission was already closed";
                        stats_->retirement_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayRetirementFenceProgress::Failed;
                    }

                    try
                    {
                        const auto vote = protocol_.makeLocalVote(
                            local_progress ==
                                    MoEOverlayRetirementFenceProgress::
                                        ReadyToRetire
                                ? MoEOverlayDistributedResidencyVoteDecision::
                                      Ready
                                : MoEOverlayDistributedResidencyVoteDecision::
                                      Waiting);
                        if (!consensus_->begin(vote, &terminal_error_))
                        {
                            retirement_failed_ = true;
                            if (terminal_error_.empty())
                            {
                                terminal_error_ =
                                    "Failed to begin distributed ExpertOverlay final retirement consensus";
                            }
                            stats_->retirement_consensus_failed.fetch_add(
                                1, std::memory_order_relaxed);
                            setDistributedError(error, terminal_error_);
                            return MoEOverlayRetirementFenceProgress::Failed;
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        retirement_failed_ = true;
                        terminal_error_ = exception.what();
                        stats_->retirement_consensus_failed.fetch_add(
                            1, std::memory_order_relaxed);
                        setDistributedError(error, terminal_error_);
                        return MoEOverlayRetirementFenceProgress::Failed;
                    }
                    retirement_consensus_started_ = true;
                    stats_->retirement_consensus_started.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    recordCounter("distributed_retirement_consensus_started");
                }

                std::vector<MoEOverlayDistributedResidencyVote> votes;
                const auto consensus_progress =
                    consensus_->poll(&votes, &terminal_error_);
                if (consensus_progress ==
                    MoEOverlayResidencyWaveProgress::Pending)
                {
                    return MoEOverlayRetirementFenceProgress::Pending;
                }
                if (consensus_progress ==
                    MoEOverlayResidencyWaveProgress::Failed)
                {
                    retirement_failed_ = true;
                    stats_->retirement_consensus_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    return MoEOverlayRetirementFenceProgress::Failed;
                }

                std::string protocol_error;
                if (!protocol_.acceptConsensus(votes, &protocol_error))
                {
                    if (protocol_.state() ==
                        MoEOverlayDistributedResidencyProtocolState::
                            RetirementAdmissionClosed)
                    {
                        retirement_consensus_started_ = false;
                        stats_->retirement_consensus_waiting.fetch_add(
                            1,
                            std::memory_order_relaxed);
                        recordCounter(
                            "distributed_retirement_consensus_waiting");
                        if (error)
                            error->clear();
                        return MoEOverlayRetirementFenceProgress::Pending;
                    }
                    retirement_failed_ = true;
                    terminal_error_ = protocol_error.empty()
                                          ? "Distributed ExpertOverlay retirement consensus failed"
                                          : std::move(protocol_error);
                    stats_->retirement_consensus_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    recordCounter("distributed_retirement_consensus_failed");
                    return MoEOverlayRetirementFenceProgress::Failed;
                }
                if (protocol_.state() !=
                    MoEOverlayDistributedResidencyProtocolState::ReadyToRetire)
                {
                    retirement_failed_ = true;
                    terminal_error_ =
                        "Distributed ExpertOverlay retirement consensus advanced to an invalid lifecycle state";
                    stats_->retirement_consensus_failed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    setDistributedError(error, terminal_error_);
                    recordCounter("distributed_retirement_consensus_failed");
                    return MoEOverlayRetirementFenceProgress::Failed;
                }

                retirement_ready_ = true;
                stats_->retirement_consensus_ready.fetch_add(
                    1,
                    std::memory_order_relaxed);
                recordCounter("distributed_retirement_consensus_ready");
                if (error)
                    error->clear();
                return MoEOverlayRetirementFenceProgress::ReadyToRetire;
            }

            /** @brief Retire only the local bank after all-rank lease drainage. */
            void retirePrevious() noexcept override
            {
                if (retired_ || aborted_ || !published_)
                    return;
                if (!retirement_ready_ ||
                    protocol_.state() !=
                        MoEOverlayDistributedResidencyProtocolState::
                            ReadyToRetire)
                {
                    LOG_ERROR(
                        "[DistributedResidencyWave] Old bank retirement preceded all-rank lease drainage");
                    std::terminate();
                }
                local_wave_->retirePrevious();
                try
                {
                    protocol_.markRetired();
                }
                catch (...)
                {
                    std::terminate();
                }
                retired_ = true;
            }

        private:
            /**
             * @brief Attach the full local diagnostic only on its owning rank.
             *
             * Votes deliberately carry fixed-size codes and fingerprints. The
             * rank which authored the first failed vote still owns the exact
             * backend diagnostic and must retain it in its terminal message;
             * peers can correlate that message through the shared fingerprint
             * without putting variable strings into the collective ABI.
             */
            void appendOriginatingLocalDiagnostic(
                const char *phase,
                const std::string &diagnostic)
            {
                const auto &failure = protocol_.failure();
                if (!failure ||
                    failure->world_rank != consensus_->worldRank() ||
                    diagnostic.empty())
                {
                    return;
                }
                terminal_error_ += "; local ";
                terminal_error_ += phase;
                terminal_error_ += " detail: ";
                terminal_error_ += diagnostic;
            }

            /** @brief Export one rare wave lifecycle fact to PerfStats. */
            void recordCounter(const char *name) const
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    name,
                    1.0,
                    "maintenance",
                    perf_device_,
                    {{"world_rank", std::to_string(consensus_->worldRank())},
                     {"world_size", std::to_string(consensus_->worldSize())},
                     {"expected_epoch", std::to_string(expected_epoch_)},
                     {"candidate_epoch", std::to_string(candidate_epoch_)},
                     {"background", "true"},
                     {"blocking", "false"}});
            }

            MoEOverlayResidencyStageStartStatus local_status_ =
                MoEOverlayResidencyStageStartStatus::Failed;
            std::unique_ptr<IMoEOverlayResidencyWave> local_wave_;
            std::unique_ptr<IMoEOverlayResidencyWave> local_cleanup_wave_;
            std::string local_start_error_;
            std::shared_ptr<IMoEOverlayResidencyConsensusLane> consensus_;
            MoEOverlayDistributedResidencyProtocol protocol_;
            std::shared_ptr<
                MoEOverlayDistributedResidencyTransportSharedStats> stats_;
            std::string perf_device_;
            std::uint64_t expected_epoch_ = 0;
            std::uint64_t candidate_epoch_ = 0;
            std::chrono::steady_clock::time_point wave_started_at_{};
            MoEOverlayResidencyWaveInterval stage_interval_{};
            std::string local_stage_error_;
            std::string local_prepare_error_;
            std::string local_publication_error_;
            std::string terminal_error_;
            bool stage_dispatch_started_ = false;
            bool reservation_consensus_started_ = false;
            bool reservation_ready_ = false;
            bool stage_consensus_started_ = false;
            bool stage_ready_ = false;
            bool stage_deferred_ = false;
            bool stage_failed_ = false;
            bool prepare_started_ = false;
            bool local_prepare_begin_failed_ = false;
            bool prepare_consensus_started_ = false;
            bool prepare_ready_ = false;
            bool prepare_failed_ = false;
            bool publication_started_ = false;
            bool local_publication_begin_failed_ = false;
            bool publication_consensus_started_ = false;
            bool publication_ready_ = false;
            bool publication_failed_ = false;
            bool retirement_admission_consensus_started_ = false;
            bool retirement_admission_ready_ = false;
            bool retirement_admission_marked_closed_ = false;
            bool retirement_consensus_started_ = false;
            bool retirement_ready_ = false;
            bool retirement_failed_ = false;
            bool aborted_ = false;
            bool local_abort_ready_ = false;
            bool cleanup_abort_ready_ = false;
            bool abort_ready_ = false;
            bool published_ = false;
            bool retired_ = false;
        };
    } // namespace

    MoEOverlayDistributedResidencyTransport::
        MoEOverlayDistributedResidencyTransport(Config config)
        : config_(std::move(config)),
          stats_(std::make_shared<
                 MoEOverlayDistributedResidencyTransportSharedStats>())
    {
        if (!config_.local_transport)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay transport requires a local transport");
        }
        if (!config_.consensus || config_.consensus->worldSize() < 2 ||
            config_.consensus->worldRank() < 0 ||
            config_.consensus->worldRank() >= config_.consensus->worldSize())
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay transport requires valid multi-rank consensus membership");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay_distributed";
    }

    MoEOverlayResidencyStageStart
    MoEOverlayDistributedResidencyTransport::beginStage(
        const MoEOverlayResidencyTransaction &transaction)
    {
        if (!transaction.valid() || transaction.empty())
        {
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "Distributed ExpertOverlay transport requires a valid non-empty transaction",
            };
        }
        if (!config_.consensus->idle())
        {
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "Distributed ExpertOverlay consensus lane was not drained before a new wave",
            };
        }

        MoEOverlayResidencyStageStart local_start;
        try
        {
            local_start = config_.local_transport->beginStage(transaction);
        }
        catch (const std::exception &exception)
        {
            local_start.status = MoEOverlayResidencyStageStartStatus::Failed;
            local_start.error = exception.what();
        }
        catch (...)
        {
            local_start.status = MoEOverlayResidencyStageStartStatus::Failed;
            local_start.error =
                "Local ExpertOverlay transport threw a non-standard exception";
        }

        switch (local_start.status)
        {
        case MoEOverlayResidencyStageStartStatus::Started:
            stats_->local_stage_started.fetch_add(1, std::memory_order_relaxed);
            break;
        case MoEOverlayResidencyStageStartStatus::Deferred:
            stats_->local_stage_deferred.fetch_add(1, std::memory_order_relaxed);
            break;
        case MoEOverlayResidencyStageStartStatus::Failed:
            stats_->local_stage_failed.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        auto wave = std::make_unique<DistributedResidencyWave>(
            transaction,
            std::move(local_start),
            config_.consensus,
            stats_,
            config_.perf_device);
        stats_->waves_started.fetch_add(1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "distributed_waves_started",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"world_rank", std::to_string(config_.consensus->worldRank())},
             {"world_size", std::to_string(config_.consensus->worldSize())},
             {"migrations", std::to_string(transaction.migrations.size())},
             {"background", "true"},
             {"blocking", "false"}});
        return {
            .status = MoEOverlayResidencyStageStartStatus::Started,
            .wave = std::move(wave),
        };
    }

    MoEOverlayDistributedResidencyTransportStats
    MoEOverlayDistributedResidencyTransport::stats() const noexcept
    {
        return {
            .waves_started =
                stats_->waves_started.load(std::memory_order_relaxed),
            .local_stage_started =
                stats_->local_stage_started.load(std::memory_order_relaxed),
            .local_stage_deferred =
                stats_->local_stage_deferred.load(std::memory_order_relaxed),
            .local_stage_failed =
                stats_->local_stage_failed.load(std::memory_order_relaxed),
            .reservation_consensus_started =
                stats_->reservation_consensus_started.load(
                    std::memory_order_relaxed),
            .reservation_consensus_ready =
                stats_->reservation_consensus_ready.load(
                    std::memory_order_relaxed),
            .reservation_consensus_deferred =
                stats_->reservation_consensus_deferred.load(
                    std::memory_order_relaxed),
            .reservation_consensus_failed =
                stats_->reservation_consensus_failed.load(
                    std::memory_order_relaxed),
            .stage_consensus_started =
                stats_->stage_consensus_started.load(std::memory_order_relaxed),
            .stage_consensus_ready =
                stats_->stage_consensus_ready.load(std::memory_order_relaxed),
            .stage_consensus_deferred =
                stats_->stage_consensus_deferred.load(std::memory_order_relaxed),
            .stage_consensus_failed =
                stats_->stage_consensus_failed.load(std::memory_order_relaxed),
            .preparation_consensus_started =
                stats_->preparation_consensus_started.load(
                    std::memory_order_relaxed),
            .preparation_consensus_ready =
                stats_->preparation_consensus_ready.load(
                    std::memory_order_relaxed),
            .preparation_consensus_failed =
                stats_->preparation_consensus_failed.load(
                    std::memory_order_relaxed),
            .local_preparation_begin_failed =
                stats_->local_preparation_begin_failed.load(
                    std::memory_order_relaxed),
            .publication_consensus_started =
                stats_->publication_consensus_started.load(
                    std::memory_order_relaxed),
            .publication_consensus_ready =
                stats_->publication_consensus_ready.load(
                    std::memory_order_relaxed),
            .publication_consensus_failed =
                stats_->publication_consensus_failed.load(
                    std::memory_order_relaxed),
            .local_publication_begin_failed =
                stats_->local_publication_begin_failed.load(
                    std::memory_order_relaxed),
            .retirement_admission_consensus_started =
                stats_->retirement_admission_consensus_started.load(
                    std::memory_order_relaxed),
            .retirement_admission_consensus_waiting =
                stats_->retirement_admission_consensus_waiting.load(
                    std::memory_order_relaxed),
            .retirement_admission_consensus_ready =
                stats_->retirement_admission_consensus_ready.load(
                    std::memory_order_relaxed),
            .retirement_admission_consensus_failed =
                stats_->retirement_admission_consensus_failed.load(
                    std::memory_order_relaxed),
            .retirement_consensus_started =
                stats_->retirement_consensus_started.load(
                    std::memory_order_relaxed),
            .retirement_consensus_waiting =
                stats_->retirement_consensus_waiting.load(
                    std::memory_order_relaxed),
            .retirement_consensus_ready =
                stats_->retirement_consensus_ready.load(
                    std::memory_order_relaxed),
            .retirement_consensus_failed =
                stats_->retirement_consensus_failed.load(
                    std::memory_order_relaxed),
            .waves_published =
                stats_->waves_published.load(std::memory_order_relaxed),
            .waves_aborted =
                stats_->waves_aborted.load(std::memory_order_relaxed),
            .abort_cleanups_completed =
                stats_->abort_cleanups_completed.load(
                    std::memory_order_relaxed),
            .inference_thread_waits = 0,
            .blocking_synchronizations = 0,
        };
    }

    std::uint64_t
    MoEOverlayDistributedResidencyTransport::completedPlacementPayloadBytes()
        const noexcept
    {
        return config_.local_transport
                   ? config_.local_transport->completedPlacementPayloadBytes()
                   : 0u;
    }
} // namespace llaminar2
