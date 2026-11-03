/**
 * @file PlanningExecutionMeasurement.cpp
 * @brief Exact-stream, fail-closed startup sampling without inference warmup.
 *
 * Event storage lives across the entire GPU observation. Warmup retires before
 * the start event, and the stop event retires before elapsed time is read.
 * These are setup/observation boundaries, never per-token synchronization.
 * The caller owns production work and its physical-memory claims; this module
 * does not manufacture proxy tensors, alternative kernels or live ledgers.
 */
#include "planning/PlanningExecutionMeasurement.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    PlanningMeasurementWork::PlanningMeasurementWork(PlanningWorkUnit unit,
        double per_invocation, std::string provenance)
        : unit_(unit), per_invocation_(per_invocation), provenance_(std::move(provenance))
    {
        if ((unit != PlanningWorkUnit::Bytes && unit != PlanningWorkUnit::ArithmeticOperations) ||
            !std::isfinite(per_invocation) || per_invocation <= 0 ||
            !std::isfinite(per_invocation * PlanningExecutionMeasurement::kTimedInvocations) ||
            provenance_.find_first_not_of(" \t\n\r\f\v") == std::string::npos)
            throw std::invalid_argument("Planning measurement requires valid work units, positive bounded work and provenance");
    }

    namespace
    {
        /** @brief Own one native timing event through successful or failed setup. */
        class TimingEvent final
        {
        public:
            /** @brief Create on the exact backend device; missing timing support is fatal. */
            TimingEvent(IBackend &backend, int ordinal,
                std::function<void(const std::function<void()> &)> dispatch = [](const auto &operation) { operation(); },
                void* borrowed = nullptr)
                : backend_(backend), ordinal_(ordinal), dispatch_(std::move(dispatch)), borrowed_(borrowed!=nullptr)
            {
                if(borrowed)event_=borrowed;
                else dispatch_([&] { event_ = backend_.createTimingEvent(ordinal_); });
                if (!event_) throw std::runtime_error("Planning measurement could not create a timing event");
            }
            /** @brief Retire only this owned event; never synchronize an entire stream. */
            ~TimingEvent() { if(!borrowed_)dispatch_([&] { backend_.destroyEvent(event_, ordinal_); }); }
            TimingEvent(const TimingEvent &) = delete;
            TimingEvent &operator=(const TimingEvent &) = delete;
            /** @return Borrowed handle for elapsed-time calculation after completion. */
            void *get() const noexcept { return event_; }
            /** @brief Record on the producer's exact stream, rejecting submission errors. */
            void record(ExplicitGPUStream stream)
            {
                bool recorded = false;
                dispatch_([&] { recorded = backend_.recordEvent(event_, ordinal_, stream.get()); });
                if (!recorded)
                    throw std::runtime_error("Planning measurement event record failed");
            }
            /**
             * @brief Observe completion with the standard bounded setup wait.
             * @throws std::runtime_error on asynchronous errors or a 30-second timeout.
             *
             * A failed query is not 'not ready'. Keeping those outcomes distinct
             * prevents asynchronous HIP/CUDA errors from becoming an endless poll.
             */
            void awaitCompletion()
            {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                for (;;)
                {
                    bool ready = false;
                    bool queried = false;
                    dispatch_([&] { queried = backend_.queryEvent(event_, ordinal_, &ready); });
                    if (!queried)
                        throw std::runtime_error("Planning measurement asynchronous event query failed");
                    if (ready) return;
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("Planning measurement event exceeded the 30-second setup timeout");
                    std::this_thread::yield();
                }
            }
        private:
            IBackend &backend_;
            int ordinal_;
            std::function<void(const std::function<void()> &)> dispatch_;
            bool borrowed_=false;
            void *event_ = nullptr;
        };
    }

    PlanningServiceObservation PlanningExecutionMeasurement::cpu(
        const PlanningMeasurementWork &work, const std::function<bool()> &execute)
    {
        if (!execute) throw std::invalid_argument("Planning CPU measurement requires a production operation");
        if (!execute()) throw std::runtime_error("Planning CPU measurement warmup failed");
        const auto start = std::chrono::steady_clock::now();
        for (int sample = 0; sample < kTimedInvocations; ++sample)
            if (!execute()) throw std::runtime_error("Planning CPU measurement execution failed");
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return {work.unit(), work.perInvocation() * kTimedInvocations, seconds,
            work.provenance() + "; completed CPU workshare; warmup excluded; timed_invocations=" +
            std::to_string(kTimedInvocations)};
    }

    namespace
    {
        /**
         * @brief Shared native-event loop; the original caller owns the only launch capability.
         * @param work Complete original operation identity and issued work.
         * @param backend Matching native backend.
         * @param device Exact GPU owner.
         * @param stream Exact original graph stream.
         * @param submit Closed host enqueue capability, never a device callback node.
         * @param dispatch Original worker dispatch for native timing operations; legacy callers already own it.
         * @param borrowed Original calibration-owner timing pair, or empty for legacy owned events.
         * @return Completed native GPU service excluding warmup and all preparation.
         * @throws std::exception For failed submission/event/timing; no fallback rate is returned.
         */
        PlanningServiceObservation timeGPU(const PlanningMeasurementWork &work, IBackend &backend,
            DeviceId device, ExplicitGPUStream stream, const std::function<bool()> &submit,
            const std::function<void(const std::function<void()> &)> &dispatch = [](const auto &operation) { operation(); },
            std::array<void*,2> borrowed = {})
        {
            TimingEvent start(backend, device.gpu_ordinal(), dispatch, borrowed[0]);
            TimingEvent stop(backend, device.gpu_ordinal(), dispatch, borrowed[1]);
            dispatch([&] {
                if (!submit())
                    throw std::runtime_error("Planning GPU measurement warmup graph submission failed");
                stop.record(stream);
            });
            stop.awaitCompletion();
            // The same executable, stream and pointer bindings are reused. Model
            // loading, compilation, graph construction and the first launch are not
            // included in the steady service observation.
            // One original-worker closure brackets all timed native launches.
            // Control-thread queue round trips must not inflate device service.
            dispatch([&] {
                start.record(stream);
                for (int sample = 0; sample < PlanningExecutionMeasurement::kTimedInvocations; ++sample)
                    if (!submit())
                        throw std::runtime_error("Planning GPU measurement graph submission failed");
                stop.record(stream);
            });
            stop.awaitCompletion();
            float milliseconds = 0;
            bool elapsed = false;
            dispatch([&] { elapsed = backend.eventElapsedTimeMs(start.get(), stop.get(), device.gpu_ordinal(), &milliseconds); });
            if (!elapsed)
                throw std::runtime_error("Planning GPU measurement elapsed-time query failed");
            return {work.unit(), work.perInvocation() * PlanningExecutionMeasurement::kTimedInvocations,
                static_cast<double>(milliseconds) * 0.001,
                work.provenance() + "; retained graph native events; warmup excluded; timed_invocations=" +
                std::to_string(PlanningExecutionMeasurement::kTimedInvocations)};
        }
    }

    PlanningServiceObservation PlanningExecutionMeasurement::gpu(const PlanningMeasurementWork &work,
        IBackend &backend, DeviceId device, const IGPUGraphCapture &graph)
    {
        if (!device.is_gpu() || backend.backendDeviceType() != device.type || !graph.hasExecutable())
            throw std::invalid_argument("Planning GPU measurement requires a matching device/backend and retained executable");
        return timeGPU(work, backend, device, ExplicitGPUStream(graph.executionStream()),
            [&] { return graph.launchOnStream(graph.executionStream()); });
    }

    PlanningServiceObservation PlanningExecutionMeasurement::gpu(const PlanningMeasurementWork &work,
        IBackend &backend, DeviceId device, RetainedParticipantGraphExecutable &graph)
    {
        if (!device.is_gpu() || device != graph.device_ || backend.backendDeviceType() != device.type ||
            !graph.readyForSubmission() || graph.worker_->ownsCurrentThread())
            throw std::invalid_argument("Prepared GPU measurement requires its original off-worker graph owner");
        graph.requireLiveOriginal();
        if(graph.purpose_!=RetainedParticipantGraphExecutable::DefinitionPurpose::PreparedServiceCalibration ||
            !graph.calibrationTimingEvents_[0] || !graph.calibrationTimingEvents_[1])
            throw std::logic_error("Prepared measurement requires its original owner-held timing pair");
        const auto measured = timeGPU(work, backend, device, ExplicitGPUStream(graph.stream_),
            [&] { return graph.submit(graph.lastSubmission_ + 1); },
            [&](const auto &operation) {
                if(graph.worker_->ownsCurrentThread())operation();
                else graph.worker_->submitAndWait(operation);
            }, graph.calibrationTimingEvents_);
        if (!graph.observeTerminal())
            throw std::runtime_error("Prepared GPU measurement lost original terminal completion");
        return measured;
    }

    PlanningServiceObservation PlanningExecutionMeasurement::gpuPublished(const PlanningMeasurementWork &work,
        IBackend &backend, DeviceId device, const IGPUGraphCapture &graph, const std::function<void()> &publish)
    {
        if (!publish || !device.is_gpu() || backend.backendDeviceType() != device.type || !graph.hasExecutable())
            throw std::invalid_argument("Published GPU measurement requires a host publisher and matching retained graph");
        const ExplicitGPUStream stream(graph.executionStream());
        TimingEvent start(backend, device.gpu_ordinal()), stop(backend, device.gpu_ordinal());
        double seconds = 0;
        for (int sample = 0; sample <= kTimedInvocations; ++sample)
        {
            // The preceding stop event has retired before another host write.
            // Each timed interval contains GPU work only, not publication cost.
            publish();
            if (sample) start.record(stream);
            if (!graph.launchOnStream(stream.get()))
                throw std::runtime_error("Published GPU measurement graph submission failed");
            stop.record(stream);
            stop.awaitCompletion();
            if (sample)
            {
                float milliseconds = 0;
                if (!backend.eventElapsedTimeMs(start.get(), stop.get(), device.gpu_ordinal(), &milliseconds) ||
                    !std::isfinite(milliseconds) || milliseconds <= 0)
                    throw std::runtime_error("Published GPU measurement has no positive completed interval");
                seconds += static_cast<double>(milliseconds) * .001;
            }
        }
        return {work.unit(), work.perInvocation() * kTimedInvocations, seconds,
            work.provenance() + "; retained graph native events; fresh host publication excluded; timed_invocations=" +
            std::to_string(kTimedInvocations)};
    }
}
