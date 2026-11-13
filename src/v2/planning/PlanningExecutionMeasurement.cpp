/**
 * @file PlanningExecutionMeasurement.cpp
 * @brief Exact-stream, fail-closed startup sampling without inference warmup.
 *
 * Prepared startup observations bracket three original body imports with fixed
 * device-clock markers in one parent. Its sole original-body warmup precedes
 * the start marker in that same ordered parent, without an intervening host gap;
 * only the final compact clock bank is read after an Ordering completion.
 * Explicitly published host links instead refresh bytes before each fixed-clock
 * replay. Standalone graph APIs retain their original external native-event timing.
 * These are setup/observation boundaries, never per-token synchronization.
 * The caller owns production work and its physical-memory claims; this module
 * does not manufacture proxy tensors, alternative kernels or live ledgers.
 */
#include "planning/PlanningExecutionMeasurement.h"
#include "backends/IBackend.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <limits>
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

    /** @brief The original clock bank and one Ordering event; no native timestamp events are created. */
    struct PlanningCapturedGPUClockSampler::Impl
    {
        IWorkerGPUContext &worker;
        ExplicitGPUStream stream;
        PlanningGPUClockBuffer &clock;
        size_t windows;
        std::uint64_t frequency;
        void *completion = nullptr;
        size_t next = 0;
        bool open = false;
        const IGPUGraphCapture *sealed = nullptr;
        /** @brief Create only the original worker's small completion authority. */
        Impl(IWorkerGPUContext &worker_, ExplicitGPUStream stream_, PlanningGPUClockBuffer &clock_)
            : worker(worker_), stream(stream_), clock(clock_), windows(clock_.slotCount() / 2),
              frequency(clock_.ticksPerSecond())
        {
            completion = worker.createEvent(GPUEventPurpose::Ordering);
            if (!completion) throw std::runtime_error("Captured clock completion event creation failed");
        }
        /** @brief Graph and device-clock storage are owned outside and must outlive their submissions. */
        ~Impl() { worker.destroyEvent(completion); }
        /** @brief Publish original stream completion with checked nonblocking polling. */
        void awaitGraphCompletion()
        {
            if (!worker.recordEventChecked(completion, stream.get()))
                throw std::runtime_error("Captured clock ordering publication failed");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            for (;;)
            {
                bool ready = false;
                if (!worker.queryEventChecked(completion, ready))
                    throw std::runtime_error("Captured clock asynchronous completion query failed");
                if (ready) return;
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("Captured clock exceeded the30-second setup timeout");
                std::this_thread::yield();
            }
        }
        /** @brief Geometry and event operations remain bound to their original worker and immutable bank. */
        void requireOriginal() const
        {
            if (!worker.ownsCurrentThread() || clock.slotCount() / 2 != windows || clock.slotCount() % 2 ||
                clock.ticksPerSecond() != frequency)
                throw std::invalid_argument("Captured clock requires its original worker and fixed slot/frequency geometry");
        }
    };

    PlanningCapturedGPUClockSampler::PlanningCapturedGPUClockSampler(IWorkerGPUContext &worker,
        ExplicitGPUStream stream, PlanningGPUClockBuffer &clock)
    {
        if (!worker.ownsCurrentThread() || !clock.slotCount() || clock.slotCount() % 2 || !clock.ticksPerSecond())
            throw std::invalid_argument("Captured clock requires positive even slots and a fixed clock on its original worker");
        impl_ = std::make_unique<Impl>(worker, stream, clock);
    }

    PlanningCapturedGPUClockSampler::~PlanningCapturedGPUClockSampler() = default;

    size_t PlanningCapturedGPUClockSampler::windowCount() const noexcept { return impl_->windows; }

    void PlanningCapturedGPUClockSampler::recordStart(size_t window)
    {
        auto &owner = *impl_;
        owner.requireOriginal();
        if (window == 0 && owner.next == windowCount() && !owner.open)
        { owner.next = 0; owner.sealed = nullptr; }
        if (owner.open || window != owner.next || window >= windowCount())
            throw std::invalid_argument("Captured clock start has foreign/overlapping window order");
        owner.clock.record(window * 2, owner.stream);
        owner.open = true;
    }

    void PlanningCapturedGPUClockSampler::recordStop(size_t window)
    {
        auto &owner = *impl_;
        owner.requireOriginal();
        if (!owner.open || window != owner.next || window >= windowCount())
            throw std::invalid_argument("Captured clock stop has foreign/incomplete window order");
        owner.clock.record(window * 2 + 1, owner.stream);
        owner.open = false;
        ++owner.next;
    }

    void PlanningCapturedGPUClockSampler::seal(const IGPUGraphCapture &graph)
    {
        auto &owner = *impl_;
        owner.requireOriginal();
        if ((owner.sealed && owner.sealed != &graph) || owner.open || owner.next != windowCount() || !graph.hasExecutable() ||
            graph.executionStream() != owner.stream.get() || graph.nodeCount() < 2 * windowCount())
            throw std::invalid_argument("Captured clock graph lacks its complete original window binding");
        owner.sealed = &graph;
    }

    std::vector<PlanningServiceObservation> PlanningCapturedGPUClockSampler::samples(
        std::span<const PlanningMeasurementWork> work, const IGPUGraphCapture &graph)
    {
        auto &owner = *impl_;
        owner.requireOriginal();
        if (owner.sealed != &graph || work.size() != windowCount() || !graph.hasExecutable() ||
            graph.executionStream() != owner.stream.get())
            throw std::invalid_argument("Captured clock requires its original sealed graph and window geometry");
        if (!graph.launchOnStream(owner.stream.get()))
            throw std::runtime_error("Captured clock warmup graph submission failed");
        owner.awaitGraphCompletion();
        return samplesFromPreparedParent(work, graph);
    }

    std::vector<PlanningServiceObservation> PlanningCapturedGPUClockSampler::samplesFromPreparedParent(
        std::span<const PlanningMeasurementWork> work,const IGPUGraphCapture &graph)
    {
        auto &owner=*impl_;owner.requireOriginal();
        if(owner.sealed!=&graph || work.size()!=windowCount() || !graph.hasExecutable() ||
            graph.executionStream()!=owner.stream.get())
            throw std::invalid_argument("Captured clock timed transaction lost its original seal");
        if (!graph.launchOnStream(owner.stream.get()))
            throw std::runtime_error("Captured clock timed graph submission failed");
        owner.clock.enqueueReadback(owner.stream);
        owner.awaitGraphCompletion();
        const auto ticks = owner.clock.ticks();
        if (ticks.size() != 2 * windowCount())
            throw std::runtime_error("Captured clock returned an incomplete timestamp bank");
        std::vector<PlanningServiceObservation> observations;
        observations.reserve(work.size());
        for (size_t window = 0; window < work.size(); ++window)
        {
            const auto start = ticks[2 * window], stop = ticks[2 * window + 1];
            if (!start || stop <= start) throw std::runtime_error("Captured clock interval is unwritten, zero, reversed or wrapped");
            observations.emplace_back(work[window].unit(), work[window].perInvocation(),
                double(stop - start) / double(owner.frequency),
                work[window].provenance() + "; captured fixed-device-clock window=" + std::to_string(window) +
                "; timed graph replay; final timestamp-bank readback only");
        }
        return observations;
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
        /** @brief Native preparation completion is separate from the isolated timing interval. */
        void warmGPU(ExplicitGPUStream stream,const std::function<bool()> &submit,
            const std::function<void(const std::function<void()> &)> &dispatch,TimingEvent &stop)
        {
            dispatch([&] {
                if (!submit())
                    throw std::runtime_error("Planning GPU measurement warmup graph submission failed");
                stop.record(stream);
            });
            stop.awaitCompletion();
        }
        /** @brief Shared original timing body; no public caller can manufacture a warmup token. */
        PlanningServiceObservation observeGPU(const PlanningMeasurementWork &work,IBackend &backend,DeviceId device,
            ExplicitGPUStream stream,const std::function<bool()> &submit,
            const std::function<void(const std::function<void()> &)> &dispatch,TimingEvent &start,TimingEvent &stop)
        {
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
            TimingEvent start(backend,device.gpu_ordinal(),dispatch,borrowed[0]);
            TimingEvent stop(backend,device.gpu_ordinal(),dispatch,borrowed[1]);
            warmGPU(stream,submit,dispatch,stop);
            return observeGPU(work,backend,device,stream,submit,dispatch,start,stop);
        }

    }

    /** @brief Clock backing outlives marker fragments; parent retires before all imported children. */
    struct PlanningPreparedGPUObservation::Impl
    {
        PlanningMeasurementWork work;
        std::function<void()> publish; ///< Setup-only host ownership; absent for contiguous GPU compute.
        IWorkerGPUContext &worker;
        ExplicitGPUStream stream;
        const std::thread::id owner_thread=std::this_thread::get_id();
        std::unique_ptr<PlanningGPUClockBuffer> clock;
        std::unique_ptr<PlanningCapturedGPUClockSampler> timer;
        std::unique_ptr<IGPUGraphCapture> body,start,stop,parent;
        enum class State { Recorded, Failed, Prepared, Consumed };
        State state=State::Recorded;
        /** @brief Record only unique marker fragments; original body capture is already ended. */
        Impl(const PlanningMeasurementWork &w,IWorkerGPUContext &worker_,
            std::unique_ptr<PlanningGPUClockBuffer> c,std::unique_ptr<IGPUGraphCapture> graph)
            :work(w),worker(worker_),stream(graph->executionStream()),clock(std::move(c)),body(std::move(graph))
        {
            timer=std::make_unique<PlanningCapturedGPUClockSampler>(worker,stream,*clock);
            start=worker.createGraphCapture(stream.get());stop=worker.createGraphCapture(stream.get());
            parent=worker.createGraphCapture(stream.get());
            if(!start || !stop || !parent)throw std::runtime_error("Prepared clock graph creation failed");
            {ScopedBackendGraphCapture capture(worker,*start,"planning clock start");
             if(!capture.begin())throw std::runtime_error("Prepared clock start capture failed");
             timer->recordStart(0);capture.finish();}
            {ScopedBackendGraphCapture capture(worker,*stop,"planning clock stop");
             if(!capture.begin())throw std::runtime_error("Prepared clock stop capture failed");
             timer->recordStop(0);capture.finish();}
        }
        /** @brief Reject foreign lifecycle calls before touching native state. */
        void requireOwner() const
        {
            if(owner_thread!=std::this_thread::get_id() || !worker.ownsCurrentThread() ||
               !body || body->executionStream()!=stream.get())
                throw std::logic_error("Prepared GPU observation requires its original worker/body/stream");
        }
    };
    PlanningPreparedGPUObservation::PlanningPreparedGPUObservation(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
    PlanningPreparedGPUObservation::~PlanningPreparedGPUObservation()
    { if(impl_ && impl_->owner_thread!=std::this_thread::get_id())std::terminate(); }
    PlanningPreparedGPUObservation::PlanningPreparedGPUObservation(PlanningPreparedGPUObservation&& other)
    {
        if(other.impl_)other.impl_->requireOwner();
        impl_=std::move(other.impl_);
    }
    PlanningPreparedGPUObservation& PlanningPreparedGPUObservation::operator=(PlanningPreparedGPUObservation&& other)
    {
        if(impl_)impl_->requireOwner();
        if(other.impl_)other.impl_->requireOwner();
        impl_=std::move(other.impl_);return *this;
    }
    void PlanningPreparedGPUObservation::prepare(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if(!impl_)throw std::logic_error("Prepared GPU observation was moved");
        auto &owner=*impl_;owner.requireOwner();
        if(owner.state!=Impl::State::Recorded)throw std::logic_error("Prepared GPU observation preparation repeated");
        if(!memory)throw std::invalid_argument("Prepared GPU observation requires its physical memory authority");
        owner.state=Impl::State::Failed;
        // Both explicit protocols own the same four graphs. GPU compute embeds
        // one contiguous warmup; host publication uses one clock-bracketed body
        // per replay so CPU-owned mapped input can be refreshed after completion.
        for(auto *graph:{owner.body.get(),owner.start.get(),owner.stop.get()})
            if(!graph->prepareRuntimeContextStorage(memory) || !graph->instantiate())
                throw std::runtime_error("Prepared GPU clock fragment instantiation failed");
        bool built=false;
        if(owner.publish) {
            const std::array<GPUOrderedTimelineStep,3> steps{{
                {.name="clock-start",.capture=owner.start.get()},
                {.name="published-body",.capture=owner.body.get()},
                {.name="clock-stop",.capture=owner.stop.get()}}};
            built=owner.parent->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::Disabled);
        } else {
            const std::array<GPUOrderedTimelineStep,6> steps{{
                {.name="body-warmup",.capture=owner.body.get()},
                {.name="clock-start",.capture=owner.start.get()},
                {.name="body-0",.capture=owner.body.get()},
                {.name="body-1",.capture=owner.body.get()},
                {.name="body-2",.capture=owner.body.get()},
                {.name="clock-stop",.capture=owner.stop.get()}}};
            built=owner.parent->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::Disabled);
        }
        if(!built || !owner.parent->prepareRuntimeContextStorage(memory) || !owner.parent->instantiate())
            throw std::runtime_error("Prepared GPU clock parent instantiation failed");
        owner.timer->seal(*owner.parent);
        owner.state=Impl::State::Prepared;
    }
    size_t PlanningPreparedGPUObservation::residentMemoryBytes() const
    {
        if(!impl_)throw std::logic_error("Prepared GPU observation was moved");
        impl_->requireOwner();size_t bytes=0;
        for(auto *graph:{impl_->body.get(),impl_->start.get(),impl_->stop.get(),impl_->parent.get()}) {
            const auto n=graph->residentMemoryBytes();
            if(n>std::numeric_limits<size_t>::max()-bytes)throw std::overflow_error("Prepared graph byte sum overflow");
            bytes+=n;
        }
        return bytes;
    }
    size_t PlanningPreparedGPUObservation::graphNodeCount() const
    {
        if(!impl_)throw std::logic_error("Prepared GPU observation was moved");
        impl_->requireOwner();return impl_->body->nodeCount();
    }
    PlanningServiceObservation PlanningPreparedGPUObservation::observe()
    {
        if(!impl_)throw std::logic_error("Prepared GPU observation was moved");
        auto &owner=*impl_;owner.requireOwner();
        if(owner.state!=Impl::State::Prepared)throw std::logic_error("Prepared GPU observation is incomplete or consumed");
        owner.state=Impl::State::Consumed;
        if(owner.publish) {
            double seconds=0;
            const std::array work{owner.work};
            for(int sample=0;sample<=PlanningExecutionMeasurement::kTimedInvocations;++sample) {
                // The previous compact readback and Ordering completion retire
                // before CPU writes these same mapped pages again. No host
                // submission gap can enter the captured timestamp interval.
                owner.publish();
                const auto observed=owner.timer->samplesFromPreparedParent(work,*owner.parent).front();
                if(sample)seconds+=observed.elapsedSeconds();
            }
            return {owner.work.unit(),owner.work.perInvocation()*PlanningExecutionMeasurement::kTimedInvocations,seconds,
                owner.work.provenance()+"; fresh host publication before every invocation; one warmup excluded; "
                "timed_invocations=3; sum of captured fixed-device-clock intervals; compact readback after every replay"};
        }
        const PlanningMeasurementWork batch(owner.work.unit(),owner.work.perInvocation()*PlanningExecutionMeasurement::kTimedInvocations,
            owner.work.provenance()+"; one contiguous original body warmup before clock start; timed_invocations=3; one ordered parent launch");
        const std::array work{batch};
        return owner.timer->samplesFromPreparedParent(work,*owner.parent).front();
    }
    void PlanningExecutionMeasurement::contributeGPUObservationMemory(PhysicalMemoryResource host,
        PhysicalMemoryResource execution,PhysicalMemoryPlanBuilder &builder)
    {
        if(!host.device.is_cpu() || !execution.device.is_gpu() || host.world_rank!=execution.world_rank)
            throw std::invalid_argument("Prepared GPU timer requires same-rank CPU/GPU resources");
        builder.add(host,PhysicalMemoryOwner::ExecutionWorkspace,PlanningNativeGPUClockBuffer::hostAllocationBytes(2));
        builder.add(execution,PhysicalMemoryOwner::ExecutionWorkspace,PlanningNativeGPUClockBuffer::deviceAllocationBytes(2));
        builder.add(execution,PhysicalMemoryOwner::NativeGraphExecutable,
            kPreparedGraphCount*GPUGraphMemoryContract::reservationBytesPerExecutable(execution.device));
    }
    PlanningPreparedGPUObservation PlanningExecutionMeasurement::recordGPU(const PlanningMeasurementWork &work,
        IWorkerGPUContext &worker,std::unique_ptr<PlanningGPUClockBuffer> clock,std::unique_ptr<IGPUGraphCapture> body)
    {
        if(!worker.isInitialized() || !worker.ownsCurrentThread() || !clock || clock->slotCount()!=2 ||
           !body || !body->nodeCount() || body->hasExecutable() ||
           worker.isDeviceGraphCaptureActive())
            throw std::invalid_argument("Prepared GPU clock requires initialized original worker, ended uninstantiated body and two clock slots");
        return PlanningPreparedGPUObservation(std::make_unique<PlanningPreparedGPUObservation::Impl>(work,worker,std::move(clock),std::move(body)));
    }

    PlanningPreparedGPUObservation PlanningExecutionMeasurement::recordPublishedGPU(const PlanningMeasurementWork &work,
        IWorkerGPUContext &worker,std::unique_ptr<PlanningGPUClockBuffer> clock,
        std::unique_ptr<IGPUGraphCapture> body,std::function<void()> publish)
    {
        if(!publish)throw std::invalid_argument("Published GPU clock observation requires its original host publisher");
        auto token=recordGPU(work,worker,std::move(clock),std::move(body));
        token.impl_->publish=std::move(publish);
        return token;
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

}
