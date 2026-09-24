/**
 * @file PlanningExecutionMeasurement.h
 * @brief Bounded completed-work observations for automatic orchestration setup.
 *
 * Prepared kernels, captured graphs and their admitted storage remain owned by
 * the caller. This boundary only measures execution: it never discovers a
 * device, loads a model, changes dispatch, or consults PerfStats. GPU samples
 * replay the existing executable on its exact stream and use native events;
 * CPU samples include the completed production workshare's wall time.
 */
#pragma once

#include "backends/DeviceId.h"
#include "backends/ExplicitGPUStream.h"
#include "planning/OrchestrationPerformanceEvidence.h"
#include <functional>
#include <string>

namespace llaminar2
{
    class IBackend;
    class IGPUGraphCapture;

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

    /** @brief One setup-only warmup followed by a fixed, short observation batch. */
    class PlanningExecutionMeasurement final
    {
    public:
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
         * @brief Observe retained GPU work with a fresh host publication before every invocation.
         * @param work Exact completed work and operation identity.
         * @param backend Canonical exact-device backend.
         * @param device GPU owning the executable and timing events.
         * @param graph Prepared immutable graph, reused without recapture.
         * @param publish Untimed setup-only host write; called only after the preceding invocation retires.
         * @return Sum of three native event intervals, excluding publications and warmup.
         *
         * Mapped-input link samples must not repeatedly price unchanged cached
         * bytes. This explicit publication boundary changes only host-owned
         * sample storage, never inference state. It is not a host callback node,
         * hot-path synchronization or permission to mutate a live graph input.
         */
        static PlanningServiceObservation gpuPublished(const PlanningMeasurementWork &work,
            IBackend &backend, DeviceId device, const IGPUGraphCapture &graph,
            const std::function<void()> &publish);
    };
}
