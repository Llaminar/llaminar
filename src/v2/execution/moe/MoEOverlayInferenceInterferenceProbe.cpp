/**
 * @file MoEOverlayInferenceInterferenceProbe.cpp
 * @brief Lock-free live-inference interval publication for overlay calibration.
 */

#include "MoEOverlayInferenceInterferenceProbe.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <limits>

namespace llaminar2
{
    namespace
    {
        /** @brief Reject the ingestion-only alias at a production boundary. */
        bool isProductionPhase(ExpertHistogramSource source) noexcept
        {
            return static_cast<std::size_t>(source) <
                   kExpertHistogramProductionSourceCount;
        }

        /** @brief Mix one fixed-width scalar in byte-stable FNV-1a order. */
        std::uint64_t mixWorkloadScalar(
            std::uint64_t hash,
            std::uint64_t value) noexcept
        {
            constexpr std::uint64_t kPrime = 1099511628211ull;
            for (std::size_t byte = 0; byte < sizeof(value); ++byte)
            {
                hash ^= value & 0xffu;
                hash *= kPrime;
                value >>= 8u;
            }
            return hash;
        }
    } // namespace

    MoEOverlayInferenceWorkloadIdentity
    makeMoEOverlayInferenceWorkloadIdentity(
        ExpertHistogramSource source,
        int real_rows,
        int execution_rows,
        int transaction_count,
        int speculative_depth,
        std::uint64_t schedule_fingerprint) noexcept
    {
        if (!isProductionPhase(source) || real_rows <= 0 ||
            execution_rows < real_rows || transaction_count <= 0 ||
            speculative_depth < 0 ||
            (source == ExpertHistogramSource::GroupedVerifier
                 ? speculative_depth <= 0
                 : speculative_depth != 0))
        {
            return {};
        }

        std::uint64_t fingerprint = schedule_fingerprint;
        if (fingerprint == 0)
        {
            fingerprint = 14695981039346656037ull;
            for (const std::uint64_t value : {
                     static_cast<std::uint64_t>(source),
                     static_cast<std::uint64_t>(real_rows),
                     static_cast<std::uint64_t>(execution_rows),
                     static_cast<std::uint64_t>(transaction_count),
                     static_cast<std::uint64_t>(speculative_depth)})
            {
                fingerprint = mixWorkloadScalar(fingerprint, value);
            }
        }
        return {
            .source = source,
            .real_rows = real_rows,
            .execution_rows = execution_rows,
            .transaction_count = transaction_count,
            .speculative_depth = speculative_depth,
            .schedule_fingerprint = fingerprint,
        };
    }

    bool MoEOverlayInferenceWorkloadIdentity::valid() const noexcept
    {
        if (!isProductionPhase(source) || real_rows <= 0 ||
            execution_rows < real_rows || transaction_count <= 0 ||
            speculative_depth < 0 || schedule_fingerprint == 0)
        {
            return false;
        }
        return source == ExpertHistogramSource::GroupedVerifier
                   ? speculative_depth > 0
                   : speculative_depth == 0;
    }

    bool MoEOverlayInterferenceProbeRequest::valid() const noexcept
    {
        return coordinate.valid() && isProductionPhase(source) &&
               calibration_sequence != 0 &&
               (!require_exact_workload ||
                (required_workload.valid() &&
                 required_workload.source == source));
    }

    bool MoEOverlayInterferenceProbeTicket::valid() const noexcept
    {
        return isProductionPhase(source) && workload.valid() &&
               workload.source == source && probe_generation != 0 &&
               calibration_sequence != 0 &&
               begin_steady_nanoseconds != 0;
    }

    std::uint64_t
    MoEOverlayInterferenceProbeSample::durationNanoseconds() const noexcept
    {
        if (end_steady_nanoseconds <= begin_steady_nanoseconds)
            return 0;
        return end_steady_nanoseconds - begin_steady_nanoseconds;
    }

    bool MoEOverlayInterferenceProbeSample::whollyContainedBy(
        std::uint64_t wave_begin_steady_nanoseconds,
        std::uint64_t wave_end_steady_nanoseconds) const noexcept
    {
        return valid() && wave_begin_steady_nanoseconds != 0 &&
               wave_end_steady_nanoseconds > wave_begin_steady_nanoseconds &&
               begin_steady_nanoseconds >= wave_begin_steady_nanoseconds &&
               end_steady_nanoseconds <= wave_end_steady_nanoseconds;
    }

    bool MoEOverlayInterferenceProbeSample::whollyContains(
        std::uint64_t wave_begin_steady_nanoseconds,
        std::uint64_t wave_end_steady_nanoseconds) const noexcept
    {
        return valid() && wave_begin_steady_nanoseconds != 0 &&
               wave_end_steady_nanoseconds > wave_begin_steady_nanoseconds &&
               wave_begin_steady_nanoseconds >= begin_steady_nanoseconds &&
               wave_end_steady_nanoseconds <= end_steady_nanoseconds;
    }

    bool MoEOverlayInterferenceProbeSample::valid() const noexcept
    {
        return request.valid() && workload.valid() &&
               workload.source == request.source &&
               (!request.require_exact_workload ||
                request.required_workload == workload) &&
               begin_steady_nanoseconds != 0 &&
               end_steady_nanoseconds > begin_steady_nanoseconds;
    }

    std::uint64_t
    MoEOverlayInferenceInterferenceProbe::steadyNanoseconds() noexcept
    {
        const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return value > 0 ? static_cast<std::uint64_t>(value) : 1u;
    }

    MoEOverlayInferenceInterferenceProbe::State
    MoEOverlayInferenceInterferenceProbe::armedState(
        MoEOverlayInterferenceProbeMode mode) noexcept
    {
        return mode == MoEOverlayInterferenceProbeMode::Baseline
                   ? State::ArmedBaseline
                   : State::ArmedConcurrent;
    }

    MoEOverlayInferenceInterferenceProbe::State
    MoEOverlayInferenceInterferenceProbe::runningState(
        MoEOverlayInterferenceProbeMode mode) noexcept
    {
        return mode == MoEOverlayInterferenceProbeMode::Baseline
                   ? State::RunningBaseline
                   : State::RunningConcurrent;
    }

    MoEOverlayInferenceInterferenceProbe::State
    MoEOverlayInferenceInterferenceProbe::completedState(
        MoEOverlayInterferenceProbeMode mode) noexcept
    {
        return mode == MoEOverlayInterferenceProbeMode::Baseline
                   ? State::CompletedBaseline
                   : State::CompletedConcurrent;
    }

    std::uint64_t MoEOverlayInferenceInterferenceProbe::stateWord(
        State state,
        ExpertHistogramSource source,
        std::uint64_t generation) noexcept
    {
        return (generation << 16u) |
               (static_cast<std::uint64_t>(source) << 8u) |
               static_cast<std::uint64_t>(state);
    }

    MoEOverlayInferenceInterferenceProbe::State
    MoEOverlayInferenceInterferenceProbe::stateFromWord(
        std::uint64_t word) noexcept
    {
        return static_cast<State>(word & 0xffu);
    }

    ExpertHistogramSource
    MoEOverlayInferenceInterferenceProbe::sourceFromWord(
        std::uint64_t word) noexcept
    {
        return static_cast<ExpertHistogramSource>((word >> 8u) & 0xffu);
    }

    std::uint64_t
    MoEOverlayInferenceInterferenceProbe::generationFromWord(
        std::uint64_t word) noexcept
    {
        return word >> 16u;
    }

    bool MoEOverlayInferenceInterferenceProbe::arm(
        MoEOverlayInterferenceProbeRequest request) noexcept
    {
        const std::uint64_t observed =
            state_word_.load(std::memory_order_acquire);
        if (!request.valid() || stateFromWord(observed) != State::Idle ||
            next_probe_generation_ ==
                (std::numeric_limits<std::uint64_t>::max() >> 16u))
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        /*
         * The maintenance producer owns these plain fields while idle.  The
         * release publication makes the complete request visible before an
         * inference caller can claim its armed state.
         */
        request_ = request;
        completed_sample_ = {};
        ++next_probe_generation_;
        state_word_.store(
            stateWord(
                armedState(request.mode),
                request.source,
                next_probe_generation_),
            std::memory_order_release);
        arms_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    MoEOverlayInterferenceProbeTicket
    MoEOverlayInferenceInterferenceProbe::beginSample(
        MoEOverlayInferenceWorkloadIdentity workload) noexcept
    {
        if (!workload.valid())
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
        std::uint64_t observed =
            state_word_.load(std::memory_order_acquire);
        const State observed_state = stateFromWord(observed);
        MoEOverlayInterferenceProbeMode mode;
        if (observed_state == State::ArmedBaseline)
            mode = MoEOverlayInterferenceProbeMode::Baseline;
        else if (observed_state == State::ArmedConcurrent)
            mode = MoEOverlayInterferenceProbeMode::ConcurrentMovement;
        else
            return {};

        const ExpertHistogramSource armed_source = sourceFromWord(observed);
        if (armed_source != workload.source)
        {
            phase_misses_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
        const std::uint64_t generation = generationFromWord(observed);
        if (!state_word_.compare_exchange_strong(
                observed,
                stateWord(runningState(mode), armed_source, generation),
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
            return {};
        }
        if (request_.require_exact_workload &&
            request_.required_workload != workload)
        {
            state_word_.store(
                stateWord(armedState(mode), armed_source, generation),
                std::memory_order_release);
            workload_misses_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }

        claims_.fetch_add(1, std::memory_order_relaxed);
        return {
            .source = armed_source,
            .workload = workload,
            .mode = mode,
            .probe_generation = generation,
            .calibration_sequence = request_.calibration_sequence,
            .begin_steady_nanoseconds = steadyNanoseconds(),
        };
    }

    bool MoEOverlayInferenceInterferenceProbe::finishSample(
        const MoEOverlayInterferenceProbeTicket &ticket) noexcept
    {
        if (deferred_completion_generation_.load(
                std::memory_order_acquire) != 0)
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        std::uint64_t end = steadyNanoseconds();
        if (end <= ticket.begin_steady_nanoseconds)
            end = ticket.begin_steady_nanoseconds + 1u;
        return completeSample(ticket, end);
    }

    bool MoEOverlayInferenceInterferenceProbe::completeSample(
        const MoEOverlayInterferenceProbeTicket &ticket,
        std::uint64_t end_steady_nanoseconds) noexcept
    {
        if (!ticket.valid() || request_.source != ticket.source ||
            request_.mode != ticket.mode ||
            request_.calibration_sequence != ticket.calibration_sequence ||
            end_steady_nanoseconds <= ticket.begin_steady_nanoseconds ||
            state_word_.load(std::memory_order_acquire) !=
                stateWord(
                    runningState(ticket.mode),
                    ticket.source,
                    ticket.probe_generation))
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        completed_sample_ = {
            .request = request_,
            .workload = ticket.workload,
            .begin_steady_nanoseconds = ticket.begin_steady_nanoseconds,
            .end_steady_nanoseconds = end_steady_nanoseconds,
        };
        /* Publish the complete result after every plain field is initialized. */
        state_word_.store(
            stateWord(
                completedState(ticket.mode),
                ticket.source,
                ticket.probe_generation),
            std::memory_order_release);
        completions_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool MoEOverlayInferenceInterferenceProbe::deferSampleCompletion(
        const MoEOverlayInterferenceProbeTicket &ticket,
        std::shared_ptr<IMoEOverlayInferenceCompletionFence> fence,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        const std::uint64_t expected_state = stateWord(
            runningState(ticket.mode),
            ticket.source,
            ticket.probe_generation);
        if (!ticket.valid() || !fence ||
            request_.source != ticket.source ||
            request_.mode != ticket.mode ||
            request_.calibration_sequence != ticket.calibration_sequence ||
            state_word_.load(std::memory_order_acquire) != expected_state ||
            deferred_completion_generation_.load(
                std::memory_order_acquire) != 0)
        {
            if (error)
                *error =
                    "ExpertOverlay device completion does not own the exact running sample";
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        /*
         * Every participant event is already recorded before these plain
         * fields become visible. The maintenance producer's acquire load
         * therefore cannot query an uninitialized fence or a prior sample.
         */
        deferred_completion_ticket_ = ticket;
        deferred_completion_fence_ = std::move(fence);
        deferred_completion_generation_.store(
            ticket.probe_generation,
            std::memory_order_release);
        return true;
    }

    void MoEOverlayInferenceInterferenceProbe::pollDeferredCompletion()
        noexcept
    {
        const std::uint64_t generation =
            deferred_completion_generation_.load(std::memory_order_acquire);
        if (generation == 0)
            return;

        const auto &ticket = deferred_completion_ticket_;
        const auto &fence = deferred_completion_fence_;
        if (!fence || !ticket.valid() ||
            ticket.probe_generation != generation ||
            state_word_.load(std::memory_order_acquire) !=
                stateWord(
                    runningState(ticket.mode),
                    ticket.source,
                    ticket.probe_generation))
        {
            std::fputs(
                "ExpertOverlay calibration lost its deferred device-completion ownership\n",
                stderr);
            std::terminate();
        }

        std::string error;
        const auto progress = fence->poll(&error);
        if (progress ==
            MoEOverlayInferenceCompletionFenceProgress::Pending)
        {
            return;
        }
        if (progress ==
            MoEOverlayInferenceCompletionFenceProgress::Failed)
        {
            std::fprintf(
                stderr,
                "ExpertOverlay calibration device-completion query failed: %s\n",
                error.empty() ? "backend event query failed" : error.c_str());
            std::terminate();
        }

        std::uint64_t end = steadyNanoseconds();
        if (end <= ticket.begin_steady_nanoseconds)
            end = ticket.begin_steady_nanoseconds + 1u;
        if (!completeSample(ticket, end))
        {
            std::fputs(
                "ExpertOverlay calibration could not publish its device-complete sample\n",
                stderr);
            std::terminate();
        }

        deferred_completion_generation_.store(0, std::memory_order_release);
        deferred_completion_fence_.reset();
        deferred_completion_ticket_ = {};
    }

    bool MoEOverlayInferenceInterferenceProbe::discardSample(
        const MoEOverlayInterferenceProbeTicket &ticket) noexcept
    {
        if (!ticket.valid() || request_.source != ticket.source ||
            request_.mode != ticket.mode ||
            request_.calibration_sequence != ticket.calibration_sequence ||
            deferred_completion_generation_.load(
                std::memory_order_acquire) != 0)
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        std::uint64_t expected = stateWord(
            runningState(ticket.mode),
            ticket.source,
            ticket.probe_generation);
        if (!state_word_.compare_exchange_strong(
                expected,
                stateWord(
                    State::Idle,
                    ExpertHistogramSource::SyntheticTest,
                    ticket.probe_generation),
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        running_discards_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool MoEOverlayInferenceInterferenceProbe::consume(
        MoEOverlayInterferenceProbeSample *sample) noexcept
    {
        if (!sample)
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        pollDeferredCompletion();
        const std::uint64_t observed =
            state_word_.load(std::memory_order_acquire);
        const State observed_state = stateFromWord(observed);
        if (observed_state != State::CompletedBaseline &&
            observed_state != State::CompletedConcurrent)
        {
            return false;
        }

        *sample = completed_sample_;
        if (!sample->valid())
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        state_word_.store(
            stateWord(
                State::Idle,
                ExpertHistogramSource::SyntheticTest,
                generationFromWord(observed)),
            std::memory_order_release);
        consumptions_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool MoEOverlayInferenceInterferenceProbe::cancelArmed() noexcept
    {
        std::uint64_t observed =
            state_word_.load(std::memory_order_acquire);
        const State observed_state = stateFromWord(observed);
        if (observed_state != State::ArmedBaseline &&
            observed_state != State::ArmedConcurrent)
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!state_word_.compare_exchange_strong(
                observed,
                stateWord(
                    State::Idle,
                    ExpertHistogramSource::SyntheticTest,
                    generationFromWord(observed)),
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
            rejected_operations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        cancellations_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool MoEOverlayInferenceInterferenceProbe::discardAvailable() noexcept
    {
        pollDeferredCompletion();
        std::uint64_t observed =
            state_word_.load(std::memory_order_acquire);
        for (;;)
        {
            const State observed_state = stateFromWord(observed);
            if (observed_state == State::Idle)
                return true;
            if (observed_state == State::RunningBaseline ||
                observed_state == State::RunningConcurrent)
            {
                return false;
            }
            if (observed_state != State::ArmedBaseline &&
                observed_state != State::ArmedConcurrent &&
                observed_state != State::CompletedBaseline &&
                observed_state != State::CompletedConcurrent)
            {
                rejected_operations_.fetch_add(
                    1, std::memory_order_relaxed);
                return false;
            }

            const bool completed =
                observed_state == State::CompletedBaseline ||
                observed_state == State::CompletedConcurrent;
            if (state_word_.compare_exchange_weak(
                    observed,
                    stateWord(
                        State::Idle,
                        ExpertHistogramSource::SyntheticTest,
                        generationFromWord(observed)),
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                if (completed)
                {
                    completed_discards_.fetch_add(
                        1, std::memory_order_relaxed);
                }
                else
                {
                    cancellations_.fetch_add(
                        1, std::memory_order_relaxed);
                }
                return true;
            }
        }
    }

    MoEOverlayInterferenceProbeProgress
    MoEOverlayInferenceInterferenceProbe::progress(
        const MoEOverlayInterferenceProbeRequest &request) noexcept
    {
        if (!request.valid())
            return MoEOverlayInterferenceProbeProgress::Missing;

        pollDeferredCompletion();

        const std::uint64_t observed =
            state_word_.load(std::memory_order_acquire);
        const State observed_state = stateFromWord(observed);
        if (observed_state == State::Idle)
            return MoEOverlayInterferenceProbeProgress::Missing;

        /*
         * request_ is immutable from the release publication in arm() until
         * this generation returns to Idle.  Reading it after the acquire above
         * is therefore race-free even while inference changes only state_word_.
         */
        if (request_ != request)
            return MoEOverlayInterferenceProbeProgress::Missing;

        switch (observed_state)
        {
        case State::ArmedBaseline:
        case State::ArmedConcurrent:
            return MoEOverlayInterferenceProbeProgress::Armed;
        case State::RunningBaseline:
        case State::RunningConcurrent:
            return MoEOverlayInterferenceProbeProgress::Running;
        case State::CompletedBaseline:
        case State::CompletedConcurrent:
            return MoEOverlayInterferenceProbeProgress::Completed;
        case State::Idle:
            break;
        }
        return MoEOverlayInterferenceProbeProgress::Missing;
    }

    bool MoEOverlayInferenceInterferenceProbe::idle() const noexcept
    {
        return stateFromWord(
                   state_word_.load(std::memory_order_acquire)) == State::Idle;
    }

    MoEOverlayInferenceInterferenceProbeStats
    MoEOverlayInferenceInterferenceProbe::stats() const noexcept
    {
        return {
            .arms = arms_.load(std::memory_order_relaxed),
            .claims = claims_.load(std::memory_order_relaxed),
            .completions = completions_.load(std::memory_order_relaxed),
            .consumptions = consumptions_.load(std::memory_order_relaxed),
            .cancellations = cancellations_.load(std::memory_order_relaxed),
            .completed_discards = completed_discards_.load(
                std::memory_order_relaxed),
            .running_discards = running_discards_.load(
                std::memory_order_relaxed),
            .phase_misses = phase_misses_.load(std::memory_order_relaxed),
            .workload_misses = workload_misses_.load(
                std::memory_order_relaxed),
            .rejected_operations = rejected_operations_.load(
                std::memory_order_relaxed),
        };
    }

    MoEOverlayInferenceInterferenceScope::
        MoEOverlayInferenceInterferenceScope(
            MoEOverlayInferenceInterferenceProbe *probe,
            MoEOverlayInferenceWorkloadIdentity workload) noexcept
        : probe_(probe)
    {
        if (probe_)
            ticket_ = probe_->beginSample(workload);
    }

    MoEOverlayInferenceInterferenceScope::
        ~MoEOverlayInferenceInterferenceScope()
    {
        if (probe_ && ticket_.valid())
            (void)probe_->finishSample(ticket_);
    }

    MoEOverlayInferenceInterferenceScope::
        MoEOverlayInferenceInterferenceScope(
            MoEOverlayInferenceInterferenceScope &&other) noexcept
        : probe_(other.probe_), ticket_(other.ticket_)
    {
        other.probe_ = nullptr;
        other.ticket_ = {};
    }

    void MoEOverlayInferenceInterferenceScope::discard() noexcept
    {
        if (probe_ && ticket_.valid())
            (void)probe_->discardSample(ticket_);
        probe_ = nullptr;
        ticket_ = {};
    }
} // namespace llaminar2
