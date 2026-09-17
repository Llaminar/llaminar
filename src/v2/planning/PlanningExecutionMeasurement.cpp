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
            TimingEvent(IBackend &backend, int ordinal) : backend_(backend), ordinal_(ordinal),
                event_(backend.createTimingEvent(ordinal))
            {
                if (!event_) throw std::runtime_error("Planning measurement could not create a timing event");
            }
            /** @brief Retire only this owned event; never synchronize an entire stream. */
            ~TimingEvent() { backend_.destroyEvent(event_, ordinal_); }
            TimingEvent(const TimingEvent &) = delete;
            TimingEvent &operator=(const TimingEvent &) = delete;
            /** @return Borrowed handle for elapsed-time calculation after completion. */
            void *get() const noexcept { return event_; }
            /** @brief Record on the producer's exact stream, rejecting submission errors. */
            void record(ExplicitGPUStream stream)
            {
                if (!backend_.recordEvent(event_, ordinal_, stream.get()))
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
                    if (!backend_.queryEvent(event_, ordinal_, &ready))
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
            void *event_;
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

    PlanningServiceObservation PlanningExecutionMeasurement::gpu(const PlanningMeasurementWork &work,
        IBackend &backend, DeviceId device, const IGPUGraphCapture &graph)
    {
        if (!device.is_gpu() || backend.backendDeviceType() != device.type || !graph.hasExecutable())
            throw std::invalid_argument("Planning GPU measurement requires a matching device/backend and retained executable");
        // Stream identity comes from the graph owner, not a second caller hint.
        const ExplicitGPUStream stream(graph.executionStream());
        TimingEvent start(backend, device.gpu_ordinal());
        TimingEvent stop(backend, device.gpu_ordinal());
        if (!graph.launchOnStream(stream.get()))
            throw std::runtime_error("Planning GPU measurement warmup graph submission failed");
        stop.record(stream);
        stop.awaitCompletion();
        // The same executable, stream and pointer bindings are reused. Model
        // loading, compilation, graph construction and the first launch are not
        // included in the steady service observation.
        start.record(stream);
        for (int sample = 0; sample < kTimedInvocations; ++sample)
            if (!graph.launchOnStream(stream.get()))
                throw std::runtime_error("Planning GPU measurement graph submission failed");
        stop.record(stream);
        stop.awaitCompletion();
        float milliseconds = 0;
        if (!backend.eventElapsedTimeMs(start.get(), stop.get(), device.gpu_ordinal(), &milliseconds))
            throw std::runtime_error("Planning GPU measurement elapsed-time query failed");
        return {work.unit(), work.perInvocation() * kTimedInvocations,
            static_cast<double>(milliseconds) * 0.001,
            work.provenance() + "; retained graph native events; warmup excluded; timed_invocations=" +
            std::to_string(kTimedInvocations)};
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
