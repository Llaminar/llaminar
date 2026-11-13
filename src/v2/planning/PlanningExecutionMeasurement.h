/**
 * @file PlanningExecutionMeasurement.h
 * @brief Bounded completed-work observations for automatic orchestration setup.
 *
 * Prepared kernels, captured graphs and their admitted storage remain owned by
 * the caller. This boundary only measures execution: it never discovers a
 * device, loads a model, changes dispatch, or consults PerfStats. GPU samples
 * retain exact-stream execution; prepared startup bodies use an owned fixed-clock
 * ordered parent, while standalone graph APIs retain native-event timing;
 * CPU samples include the completed production workshare's wall time. The
 * split GPU owner retains its sealed graphs until creating-thread retirement.
 * Compute uses contiguous warmup/timed work; explicitly published host links
 * refresh host-owned bytes before each separately clock-bracketed invocation.
 */
#pragma once

#include "backends/DeviceId.h"
#include "backends/ExplicitGPUStream.h"
#include "planning/OrchestrationPerformanceEvidence.h"
#include "planning/PlanningGPUClockBuffer.h"
#include <functional>
#include <memory>
#include <vector>
#include <span>
#include <string>

namespace llaminar2
{
    class IBackend;
    class IGPUGraphCapture;
    class RetainedParticipantGraphExecutable;
    class IWorkerGPUContext;

    /**
     * @brief Immutable work declaration for one bounded production sample.
     *
     * Provenance names the physical device, format, geometry, worker/ISA and
     * cache regime at the producer. It is not permission to extrapolate this
     * observation to an unrelated operation or a different working set.
     */
    class PlanningMeasurementWork final
    {
    public:
        /**
         * @brief Validate work before any device launch or CPU invocation.
         * @param unit Bytes serviced or arithmetic operations actually issued.
         * @param per_invocation Positive finite work per complete invocation.
         * @param provenance Nonempty identity/method description from the producer.
         * @throws std::invalid_argument for invalid units, work or provenance.
         */
        PlanningMeasurementWork(PlanningWorkUnit unit, double per_invocation, std::string provenance);
        /** @return Unit of completed work; never inferred from the operation name. */
        PlanningWorkUnit unit() const noexcept { return unit_; }
        /** @return Declared work in one complete invocation, not allocation capacity. */
        double perInvocation() const noexcept { return per_invocation_; }
        /** @return Immutable producer-supplied execution identity. */
        const std::string &provenance() const noexcept { return provenance_; }
    private:
        PlanningWorkUnit unit_;
        double per_invocation_;
        std::string provenance_;
    };

    /**
     * @brief Worker-owned device-clock windows with one external ordering completion.
     *
     * Device stamp writes belong inside the retained graph, never around host
     * submission. The caller retains the original PMA clock buffer through graph
     * retirement. Only one Ordering event publishes compact readback completion.
     */
    class PlanningCapturedGPUClockSampler final
    {
    public:
        /** @brief Bind positive even clock slots and one original-worker Ordering event. */
        PlanningCapturedGPUClockSampler(IWorkerGPUContext &worker, ExplicitGPUStream stream, PlanningGPUClockBuffer &clock);
        /** @brief Retire the original worker's Ordering event after captured graph borrowers retire. */
        ~PlanningCapturedGPUClockSampler();
        PlanningCapturedGPUClockSampler(const PlanningCapturedGPUClockSampler &) = delete;
        PlanningCapturedGPUClockSampler &operator=(const PlanningCapturedGPUClockSampler &) = delete;
        /** @brief Record the next exact window's device-clock start inside native capture. */
        void recordStart(size_t window);
        /** @brief Record the matching stop inside capture; skipped/overlapping windows are rejected. */
        void recordStop(size_t window);
        /** @brief Bind the fully recorded window family to this exact instantiated graph. */
        void seal(const IGPUGraphCapture &graph);
        /** @return Immutable number of captured timing windows; independent of launch repetitions. */
        size_t windowCount() const noexcept;
        /**
         * @brief Warm up once, replay once, then read only the compact clock bank after completion.
         * @param work One exact positive work declaration per sealed captured window.
         * @param graph Original sealed executable with exact stream and captured timing bindings.
         * @return Positive per-device tick deltas converted by the admitted fixed clock frequency.
         * @throws std::exception for foreign/order/incomplete state, event errors or standard30s timeout.
         */
        std::vector<PlanningServiceObservation> samples(std::span<const PlanningMeasurementWork> work,
            const IGPUGraphCapture &graph);
    private:
        friend class PlanningPreparedGPUObservation;
        /** @brief Launch a sealed parent's clock-bracketed work once, then read its compact clock bank. */
        std::vector<PlanningServiceObservation> samplesFromPreparedParent(
            std::span<const PlanningMeasurementWork> work, const IGPUGraphCapture &graph);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /** @brief One original-worker clock transaction; staged preparation seals an ordered warmup and timed transaction. */
    class PlanningPreparedGPUObservation final
    {
    public:
        ~PlanningPreparedGPUObservation();
        PlanningPreparedGPUObservation(PlanningPreparedGPUObservation&&);
        PlanningPreparedGPUObservation& operator=(PlanningPreparedGPUObservation&&);
        PlanningPreparedGPUObservation(const PlanningPreparedGPUObservation&)=delete;
        /** @brief Consume the sealed owner using its explicit contiguous or freshly published protocol.
         * @return Three measured invocations; one warmup and every host publication are outside the clock interval.
         * @throws std::logic_error For foreign-thread, consumed, moved or invalid executable state.
         */
        PlanningServiceObservation observe();
        /** @brief Instantiate and seal the original four-graph family without executing its body.
         * @param memory Canonical authority admitting every graph runtime context.
         * @throws std::logic_error For repeated, foreign or incomplete preparation.
         */
        void prepare(const std::shared_ptr<PhysicalMemoryAuthority> &memory);
        /** @return Resident bytes of the four live graphs, for the caller's family audit. */
        size_t residentMemoryBytes() const;
        /** @return Original production-body node count, excluding clock instrumentation. */
        size_t graphNodeCount() const;
    private:
        friend class PlanningExecutionMeasurement;
        struct Impl;
        explicit PlanningPreparedGPUObservation(std::unique_ptr<Impl>);
        std::unique_ptr<Impl> impl_;
    };

    /** @brief One setup-only warmup followed by a fixed, short observation batch. */
    class PlanningExecutionMeasurement final
    {
    public:
        /**
         * @brief Own an ended production body and record two unique fixed-clock marker fragments.
         * @param work Exact single-body completed-work geometry and immutable provenance.
         * @param worker Original worker owning the body and every new graph on its exact stream.
         * @param clock Exclusive two-slot clock bank, admitted/materialized before any cohort capture.
         * @param body Ended original graph; the token owns its lifetime and all four ordered parent imports.
         * @return Recorded token; prepare(memory) must complete after the cohort capture fence.
         */
        static PlanningPreparedGPUObservation recordGPU(const PlanningMeasurementWork &work,
            IWorkerGPUContext &worker,std::unique_ptr<PlanningGPUClockBuffer> clock,
            std::unique_ptr<IGPUGraphCapture> body);
        /**
         * @brief Record the same fixed-clock owner for fresh host publication before every invocation.
         * @param work Exact single-body completed-work geometry and source/NUMA provenance.
         * @param worker Original GPU worker for every lifecycle operation.
         * @param clock Exclusive two-slot bank admitted before capture.
         * @param body Ended, uninstantiated production copy body.
         * @param publish Setup-only host write, invoked after the previous parent/readback retires.
         * @return Token retaining a three-step [start,body,stop] parent. Observation
         * performs one warmup and three measured replays, each after fresh publication;
         * compact fixed-clock readback excludes all host publication/submission gaps.
         */
        static PlanningPreparedGPUObservation recordPublishedGPU(const PlanningMeasurementWork &work,
            IWorkerGPUContext &worker,std::unique_ptr<PlanningGPUClockBuffer> clock,
            std::unique_ptr<IGPUGraphCapture> body,std::function<void()> publish);
        static constexpr size_t kPreparedGraphCount = 4; ///< Body, two unique markers and one ordered parent.
        /** @brief Declare one simultaneously live timer family, excluding caller-owned body payloads.
         * @param host Original same-rank CPU clock backing owner.
         * @param execution Original GPU owner for clock slots and four native graph reservations.
         * @param builder Canonical PMA declaration receiving exact concurrent storage.
         */
        static void contributeGPUObservationMemory(PhysicalMemoryResource host,PhysicalMemoryResource execution,
            PhysicalMemoryPlanBuilder &builder);
        static constexpr int kTimedInvocations = 3; ///< No unbounded startup calibration loop.
        /**
         * @brief Measure a prepared CPU operation including all workshare completion.
         * @param work Exact work/identity shared by warmup and timed invocations.
         * @param execute Production operation; false or exceptions are fatal.
         * @return Completed work and wall time, excluding one untimed warmup.
         * @throws std::invalid_argument for an absent operation.
         * @throws std::runtime_error if the operation fails.
         */
        static PlanningServiceObservation cpu(const PlanningMeasurementWork &work,
            const std::function<bool()> &execute);

        /**
         * @brief Measure a retained production graph with exact-stream native events.
         * @param work Exact operation/working-set identity, already prepared by the owner.
         * @param backend Canonical backend matching the explicit device.
         * @param device Observed physical CUDA or ROCm device; never a rank ordinal.
         * @param graph Complete prepared executable; not recorded or changed here.
         * @return Completed device work and GPU time, excluding preparation/warmup.
         * @throws std::invalid_argument for mismatched backend/device or an unready graph.
         * @throws std::runtime_error for event, submission, asynchronous execution or timeout failures.
         *
         * The owner must retain the graph, stream and all admitted allocations
         * until this setup transaction returns. Failure is not permission to
         * retry eager execution, substitute host timing, or assign a zero rate.
         */
        static PlanningServiceObservation gpu(const PlanningMeasurementWork &work,
            IBackend &backend, DeviceId device, const IGPUGraphCapture &graph);

        /**
         * @brief Time original-owner submissions without launching its sealed recording facade.
         * @param work Exact operation/working-set identity.
         * @param backend Canonical backend matching the original owner.
         * @param device Exact original GPU.
         * @param graph Original General owner, already recorded and instantiated under the same PMA.
         * @return Three completed invocations in one native event interval, excluding warmup.
         * @throws std::exception For foreign ownership, failed submission or unproved completion.
         * @note Only original owner submit/terminal APIs advance lifecycle; no raw capture launch occurs.
         */
        static PlanningServiceObservation gpu(const PlanningMeasurementWork &work,
            IBackend &backend, DeviceId device, RetainedParticipantGraphExecutable &graph);


    };
}
