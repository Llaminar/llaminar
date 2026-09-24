/**
 * @file MoEOverlayInferenceInterferenceProbe.h
 * @brief Lock-free timing handshake between live inference and overlay calibration.
 *
 * The maintenance thread arms one exact calibration request.  The next live
 * production invocation of the requested phase claims it with one atomic CAS,
 * records a host monotonic interval, and publishes a fixed-size result.  The
 * ordinary disabled path is one acquire load and performs no allocation,
 * waiting, logging, or device synchronization.
 *
 * This class deliberately does not decide whether movement overlapped the
 * sample.  The calibration owner must compare the returned absolute interval
 * with the physical wave interval and reject partial overlap.  Keeping that
 * proof outside the inference path prevents a transport poll or host lock from
 * leaking into captured decode, prefill, or grouped-verifier execution.
 */

#pragma once

#include "MoEOverlayMigrationMeasurementLedger.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace llaminar2
{
    /** @brief Whether an armed sample is an inference-only or movement run. */
    enum class MoEOverlayInterferenceProbeMode : std::uint8_t
    {
        Baseline,
        ConcurrentMovement,
    };

    /**
     * @brief Maintenance-side view of one exact probe request lifecycle.
     *
     * `Missing` deliberately combines idle and foreign generations.  A caller
     * must supply the complete request it armed, so it can never mistake an old
     * or unrelated inference ticket for authority to dispatch background work.
     */
    enum class MoEOverlayInterferenceProbeProgress : std::uint8_t
    {
        Missing,
        Armed,
        Running,
        Completed,
    };

    /** @brief Result of one non-blocking device-terminal event observation. */
    enum class MoEOverlayInferenceCompletionFenceProgress : std::uint8_t
    {
        Pending, ///< Recorded device work has not reached the event yet.
        Ready,   ///< Every operation preceding the event has completed.
        Failed,  ///< The backend rejected the non-blocking event query.
    };

    /**
     * @brief Stable setup-owned device event used to close one live sample.
     *
     * Implementations expose an already-armed completion condition to the
     * maintenance producer, which later calls `poll()` without blocking. A
     * fence may aggregate several participant-local backend events; recording
     * those exact producer-stream events remains the topology coordinator's
     * responsibility.
     */
    class IMoEOverlayInferenceCompletionFence
    {
    public:
        virtual ~IMoEOverlayInferenceCompletionFence() = default;

        /**
         * @brief Query the recorded event without waiting on the host.
         * @param error Optional precise backend/lifecycle diagnostic.
         * @return Pending, Ready, or Failed; never blocks.
         */
        [[nodiscard]] virtual MoEOverlayInferenceCompletionFenceProgress poll(
            std::string *error = nullptr) noexcept = 0;
    };

    /**
     * @brief One reusable participant-local event with an exact producer stream.
     *
     * Graph setup allocates an implementation for every GPU continuation
     * participant. The rank coordinator records each event at that participant's
     * post-launch terminal and aggregates them behind one probe fence.
     */
    class IMoEOverlayInferenceCompletionEvent
        : public IMoEOverlayInferenceCompletionFence
    {
    public:
        /**
         * @brief Record the persistent event after one graph terminal.
         * @param producer_stream Exact non-null graph producer stream.
         * @param error Optional precise backend/lifecycle diagnostic.
         * @return True when the event now represents this sample.
         */
        virtual bool record(
            void *producer_stream,
            std::string *error = nullptr) noexcept = 0;
    };

    /**
     * @brief Allocation-free identity of the production graph workload timed.
     *
     * Paired baseline/concurrent observations must compare this value exactly.
     * A prefill signature covers its complete bucket/chunk schedule; grouped
     * verification includes the selected draft depth. Token contents are not
     * retained because the calibration path never copies request data.
     */
    struct MoEOverlayInferenceWorkloadIdentity
    {
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        int real_rows = 0;
        int execution_rows = 0;
        int transaction_count = 0;
        int speculative_depth = 0;
        std::uint64_t schedule_fingerprint = 0;

        /** @return Whether phase and graph geometry form a complete identity. */
        [[nodiscard]] bool valid() const noexcept;

        /** @brief Compare every phase and execution-geometry field. */
        bool operator==(
            const MoEOverlayInferenceWorkloadIdentity &) const = default;
    };

    /**
     * @brief Build the canonical identity for one live production transaction.
     *
     * Every calibration participant names the same semantic workload even when
     * its local graph has a different implementation, such as a padded dense
     * continuation graph paired with a live-row-only remote expert graph.
     * Centralizing the fingerprint arithmetic prevents rank-local orchestration
     * layers from publishing subtly incompatible evidence identities.
     *
     * @param source Production phase represented by the transaction.
     * @param real_rows Logical rows consumed by the transaction.
     * @param execution_rows Planner-declared physical row geometry.
     * @param transaction_count Number of graph transactions in the interval.
     * @param speculative_depth MTP draft depth, or zero outside MTP.
     * @param schedule_fingerprint Optional composed schedule identity; zero
     *        requests the canonical scalar-derived fingerprint.
     * @return A complete identity, or an invalid value for invalid geometry.
     */
    [[nodiscard]] MoEOverlayInferenceWorkloadIdentity
    makeMoEOverlayInferenceWorkloadIdentity(
        ExpertHistogramSource source,
        int real_rows,
        int execution_rows,
        int transaction_count,
        int speculative_depth,
        std::uint64_t schedule_fingerprint = 0) noexcept;

    /** @brief Maintenance-owned identity for one requested live-path sample. */
    struct MoEOverlayInterferenceProbeRequest
    {
        MoEOverlayMigrationMeasurementCoordinate coordinate;
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        /** When true, only this exact paired workload may claim the request. */
        bool require_exact_workload = false;
        MoEOverlayInferenceWorkloadIdentity required_workload;
        MoEOverlayInterferenceProbeMode mode =
            MoEOverlayInterferenceProbeMode::Baseline;
        std::uint64_t calibration_sequence = 0;

        /** @return Whether this names one production phase and calibration. */
        [[nodiscard]] bool valid() const noexcept;

        /** @brief Compare the complete calibration and workload identity. */
        bool operator==(
            const MoEOverlayInterferenceProbeRequest &) const = default;
    };

    /** @brief Inference-thread token proving ownership of one claimed sample. */
    struct MoEOverlayInterferenceProbeTicket
    {
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        MoEOverlayInferenceWorkloadIdentity workload;
        MoEOverlayInterferenceProbeMode mode =
            MoEOverlayInterferenceProbeMode::Baseline;
        /** Internal publication generation that prevents an armed-state ABA. */
        std::uint64_t probe_generation = 0;
        std::uint64_t calibration_sequence = 0;
        std::uint64_t begin_steady_nanoseconds = 0;

        /** @return Whether this token can close a claimed sample. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Immutable absolute interval consumed by the maintenance owner. */
    struct MoEOverlayInterferenceProbeSample
    {
        MoEOverlayInterferenceProbeRequest request;
        MoEOverlayInferenceWorkloadIdentity workload;
        std::uint64_t begin_steady_nanoseconds = 0;
        std::uint64_t end_steady_nanoseconds = 0;

        /** @return Positive elapsed time, or zero for an invalid interval. */
        [[nodiscard]] std::uint64_t durationNanoseconds() const noexcept;

        /** @return Whether this complete sample is inside the supplied wave. */
        [[nodiscard]] bool whollyContainedBy(
            std::uint64_t wave_begin_steady_nanoseconds,
            std::uint64_t wave_end_steady_nanoseconds) const noexcept;

        /** @return Whether the supplied complete wave is inside this sample. */
        [[nodiscard]] bool whollyContains(
            std::uint64_t wave_begin_steady_nanoseconds,
            std::uint64_t wave_end_steady_nanoseconds) const noexcept;

        /** @return Whether request identity and monotonic interval are valid. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Race-safe proof and rejection counters for one probe lifetime. */
    struct MoEOverlayInferenceInterferenceProbeStats
    {
        std::uint64_t arms = 0;
        std::uint64_t claims = 0;
        std::uint64_t completions = 0;
        std::uint64_t consumptions = 0;
        std::uint64_t cancellations = 0;
        std::uint64_t completed_discards = 0;
        std::uint64_t running_discards = 0;
        std::uint64_t phase_misses = 0;
        std::uint64_t workload_misses = 0;
        std::uint64_t rejected_operations = 0;
    };

    /**
     * @brief Single-slot allocation-free live-inference interval publisher.
     *
     * One maintenance producer owns `arm`, `cancelArmed`, and `consume`.
     * Arbitrarily many inference callers may attempt `beginSample`; exactly one
     * wins the atomic claim.  Only the winning caller may call `finishSample`.
     * The maintenance producer cannot cancel a running sample, which prevents
     * request storage from being republished while the inference thread still
     * holds its ticket.
     */
    class MoEOverlayInferenceInterferenceProbe final
    {
    public:
        /**
         * @brief Arm the next matching live production phase.
         * @return False if the request is invalid or the slot is not idle.
         */
        bool arm(MoEOverlayInterferenceProbeRequest request) noexcept;

        /**
         * @brief Claim an armed request from the matching inference phase.
         * @return A valid timing ticket for the sole winner, otherwise empty.
         */
        [[nodiscard]] MoEOverlayInterferenceProbeTicket beginSample(
            MoEOverlayInferenceWorkloadIdentity workload) noexcept;

        /**
         * @brief Publish the end time for the exact winning ticket.
         * @return False for a stale, foreign, or otherwise invalid ticket.
         */
        bool finishSample(
            const MoEOverlayInterferenceProbeTicket &ticket) noexcept;

        /**
         * @brief Close a running sample from an exact device-terminal event.
         *
         * The already-armed fence is release-published to the maintenance
         * thread. `consume()` and `progress()` then poll it
         * non-blockingly and publish the sample only after device completion.
         * This path prevents graph submission from masquerading as execution
         * completion while adding no host wait or callback to inference.
         *
         * @param ticket Exact currently running sample ticket.
         * @param fence Setup-owned, already-armed terminal implementation.
         * @param error Optional precise validation/backend diagnostic.
         * @return True when device completion now owns the sample terminal.
         */
        bool deferSampleCompletion(
            const MoEOverlayInterferenceProbeTicket &ticket,
            std::shared_ptr<IMoEOverlayInferenceCompletionFence> fence,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Release a claimed sample that did not execute its named phase.
         * @return True only for the exact currently running ticket.
         */
        bool discardSample(
            const MoEOverlayInterferenceProbeTicket &ticket) noexcept;

        /**
         * @brief Consume one completed sample and return the slot to idle.
         * @param sample Receives the complete immutable interval on success.
         * @return True only when a completed sample was available.
         */
        bool consume(MoEOverlayInterferenceProbeSample *sample) noexcept;

        /**
         * @brief Cancel a request that has not been claimed by inference.
         * @return True only when an armed request returned directly to idle.
         */
        bool cancelArmed() noexcept;

        /**
         * @brief Quiesce an unused armed or completed maintenance request.
         * @return True when the slot is idle on return; false only while an
         *         inference caller still owns a running ticket.
         *
         * Fatal calibration cleanup uses this operation instead of guessing
         * whether `cancelArmed()` or `consume()` matches the current state. It
         * never invalidates a running inference ticket.
         */
        bool discardAvailable() noexcept;

        /**
         * @brief Observe one exact armed request without claiming or consuming it.
         * @param request Complete maintenance-owned request identity.
         * @return Its current lifecycle, or `Missing` for idle/foreign state.
         *
         * This acquire-only query is the launch handshake used by economy
         * calibration.  The maintenance worker may dispatch a prepared transfer
         * only after the matching inference caller has published `Running`.
         */
        [[nodiscard]] MoEOverlayInterferenceProbeProgress progress(
            const MoEOverlayInterferenceProbeRequest &request) noexcept;

        /** @return Whether no request, sample, or ticket currently owns the slot. */
        [[nodiscard]] bool idle() const noexcept;

        /** @return Race-safe copy of proof and rejection counters. */
        [[nodiscard]] MoEOverlayInferenceInterferenceProbeStats stats()
            const noexcept;

    private:
        /** @brief Complete single-slot lifecycle encoded in one atomic word. */
        enum class State : std::uint8_t
        {
            Idle,
            ArmedBaseline,
            ArmedConcurrent,
            RunningBaseline,
            RunningConcurrent,
            CompletedBaseline,
            CompletedConcurrent,
        };

        /** @brief Read Linux/C++ monotonic time without allocation or waiting. */
        [[nodiscard]] static std::uint64_t steadyNanoseconds() noexcept;

        /** @brief Map one public mode to its armed lifecycle state. */
        [[nodiscard]] static State armedState(
            MoEOverlayInterferenceProbeMode mode) noexcept;

        /** @brief Map one public mode to its running lifecycle state. */
        [[nodiscard]] static State runningState(
            MoEOverlayInterferenceProbeMode mode) noexcept;

        /** @brief Map one public mode to its completed lifecycle state. */
        [[nodiscard]] static State completedState(
            MoEOverlayInterferenceProbeMode mode) noexcept;

        /** @brief Pack one lifecycle state with its ABA-resistant generation. */
        [[nodiscard]] static std::uint64_t stateWord(
            State state,
            ExpertHistogramSource source,
            std::uint64_t generation) noexcept;

        /** @brief Decode the lifecycle state from the atomic publication word. */
        [[nodiscard]] static State stateFromWord(std::uint64_t word) noexcept;

        /** @brief Decode the armed/running phase from the publication word. */
        [[nodiscard]] static ExpertHistogramSource sourceFromWord(
            std::uint64_t word) noexcept;

        /** @brief Decode the publication generation from the atomic word. */
        [[nodiscard]] static std::uint64_t generationFromWord(
            std::uint64_t word) noexcept;

        /**
         * @brief Advance a deferred device terminal without blocking.
         *
         * Only the maintenance producer calls this helper. Backend query
         * failure is fatal because retaining a Running sample would permit an
         * inference/migration ordering defect to degrade into a timeout.
         */
        void pollDeferredCompletion() noexcept;

        /** @brief Publish a complete interval using the supplied end time. */
        bool completeSample(
            const MoEOverlayInterferenceProbeTicket &ticket,
            std::uint64_t end_steady_nanoseconds) noexcept;

        std::atomic<std::uint64_t> state_word_{0};
        /** Sole-maintenance-producer generation, embedded in state_word_. */
        std::uint64_t next_probe_generation_ = 0;
        MoEOverlayInterferenceProbeRequest request_;
        MoEOverlayInterferenceProbeSample completed_sample_;

        /**
         * Release-published generation owning the plain deferred fields below.
         * Zero means host completion still owns the running ticket.
         */
        std::atomic<std::uint64_t> deferred_completion_generation_{0};
        MoEOverlayInterferenceProbeTicket deferred_completion_ticket_;
        std::shared_ptr<IMoEOverlayInferenceCompletionFence>
            deferred_completion_fence_;

        std::atomic<std::uint64_t> arms_{0};
        std::atomic<std::uint64_t> claims_{0};
        std::atomic<std::uint64_t> completions_{0};
        std::atomic<std::uint64_t> consumptions_{0};
        std::atomic<std::uint64_t> cancellations_{0};
        std::atomic<std::uint64_t> completed_discards_{0};
        std::atomic<std::uint64_t> running_discards_{0};
        std::atomic<std::uint64_t> phase_misses_{0};
        std::atomic<std::uint64_t> workload_misses_{0};
        std::atomic<std::uint64_t> rejected_operations_{0};
    };

    /**
     * @brief Stack-only RAII timing guard for a real production inference call.
     *
     * Construction attempts the probe's single CAS claim. Destruction closes
     * only a successfully claimed ticket, so early error returns cannot strand
     * the calibration slot in Running state. The guard neither owns nor extends
     * the probe lifetime; the runner must destroy all scopes before its probe.
     */
    class MoEOverlayInferenceInterferenceScope final
    {
    public:
        /** @brief Claim the next armed sample for this exact live workload. */
        MoEOverlayInferenceInterferenceScope(
            MoEOverlayInferenceInterferenceProbe *probe,
            MoEOverlayInferenceWorkloadIdentity workload) noexcept;

        /** @brief Close a claimed ticket at the lexical inference boundary. */
        ~MoEOverlayInferenceInterferenceScope();

        MoEOverlayInferenceInterferenceScope(
            const MoEOverlayInferenceInterferenceScope &) = delete;
        MoEOverlayInferenceInterferenceScope &operator=(
            const MoEOverlayInferenceInterferenceScope &) = delete;
        /** @brief Transfer one stack guard without duplicating ticket ownership. */
        MoEOverlayInferenceInterferenceScope(
            MoEOverlayInferenceInterferenceScope &&other) noexcept;
        MoEOverlayInferenceInterferenceScope &operator=(
            MoEOverlayInferenceInterferenceScope &&) = delete;

        /** @return Whether this scope owns the currently running sample. */
        [[nodiscard]] bool active() const noexcept
        {
            return ticket_.valid();
        }

        /** @brief Discard a claimed interval whose semantic phase did not run. */
        void discard() noexcept;

    private:
        MoEOverlayInferenceInterferenceProbe *probe_ = nullptr;
        MoEOverlayInterferenceProbeTicket ticket_;
    };
} // namespace llaminar2
