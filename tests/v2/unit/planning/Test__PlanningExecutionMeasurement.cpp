/**
 * @file Test__PlanningExecutionMeasurement.cpp
 * @brief Device-free failure and ordering proofs for startup measurements.
 *
 * Native-event doubles expose exact stream, device, warmup and retirement
 * ordering. They never allocate GPU storage or claim throughput. Real kernel
 * execution belongs to the corresponding integration preflight tests.
 */
#include "planning/PlanningExecutionMeasurement.h"
#include "collective/OriginalNativeGraphCaptureGroup.h"
#include "planning/PlanningGPUClockBuffer.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "backends/IGPUGraphCapture.h"
#include "../../mocks/MockBackend.h"
#include "../../mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <thread>

using namespace llaminar2;

namespace
{
    /** @brief Distinct native failure sites must not become successful observations. */
    enum class Failure { None, FirstEvent, SecondEvent, Record, Query, Elapsed, ZeroTime };

    /** @brief Observe only the native event API, never real device state. */
    class EventBackend final : public test::MockBackend
    {
    public:
        /** @brief Select the backend spelling without loading a device driver. */
        explicit EventBackend(DeviceType type) : MockBackend(type) {}
        Failure failure = Failure::None;
        int creates = 0, destroys = 0, records = 0, queries = 0;
        int ordinal = 7;
        int elapsed_calls = 0;
        void *expected_stream = nullptr;
        std::array<int, 2> storage{};
        /** @brief Give each successful event stable independent host-only identity. */
        void *createTimingEvent(int device) override
        {
            EXPECT_EQ(device, ordinal);
            ++creates;
            if ((failure == Failure::FirstEvent && creates == 1) ||
                (failure == Failure::SecondEvent && creates == 2)) return nullptr;
            return &storage.at(static_cast<size_t>(creates - 1));
        }
        /** @brief Count event retirement on all success and exception paths. */
        void destroyEvent(void *event, int device) override
        {
            EXPECT_EQ(device, ordinal);
            bool known = false;
            for (auto &slot : storage) known |= event == &slot;
            EXPECT_TRUE(known);
            ++destroys;
        }
        /** @brief Authenticate the graph-owned stream and explicit device. */
        bool recordEvent(void *, int device, void *stream) override
        {
            EXPECT_EQ(device, ordinal);
            EXPECT_EQ(stream, expected_stream);
            ++records;
            return failure != Failure::Record;
        }
        /** @brief Distinguish one pending poll from an asynchronous failure. */
        bool queryEvent(void *, int device, bool *ready) override
        {
            EXPECT_EQ(device, ordinal);
            ++queries;
            *ready = queries != 1;
            return failure != Failure::Query;
        }
        /** @brief Publish milliseconds only after both observation edges retired. */
        bool eventElapsedTimeMs(void *start, void *stop, int device, float *ms) override
        {
            EXPECT_EQ(start, &storage[0]);
            EXPECT_EQ(stop, &storage[1]);
            EXPECT_EQ(device, ordinal);
            EXPECT_EQ(records, 3);
            EXPECT_GE(queries, 3);
            ++elapsed_calls;
            *ms = failure == Failure::ZeroTime ? 0.0f : 6.0f;
            return failure != Failure::Elapsed;
        }
        /** @brief Stream-wide waits are never part of the sampler protocol. */
        bool synchronizeStream(void *, int) override { ADD_FAILURE(); return false; }
        /** @brief Device-wide waits are never part of the sampler protocol. */
        bool synchronize(int) override { ADD_FAILURE(); return false; }
        /** @brief Native blocking event waits must not hide asynchronous errors/timeouts. */
        bool waitForEvent(void *, int) override { ADD_FAILURE(); return false; }
    };

    /** @brief Retained executable double that rejects recapture and stream substitution. */
    class SampleGraph : public IGPUGraphCapture
    {
    public:
        int stream_storage = 0;
        void *stream = &stream_storage;
        bool executable = true;
        bool allow_capture = false, capturing = false;
        size_t nodes = 1;
        std::function<void()> before_launch;
        mutable int launches = 0;
        int fail_launch = -1;
        std::function<void()> on_destroy;
        ~SampleGraph() override {if(on_destroy)on_destroy();}
        /** @brief Sampling cannot record a replacement graph. */
        bool beginCapture() override
        {
            if (!allow_capture) { ADD_FAILURE(); return false; }
            if (capturing) return false;
            capturing = true; executable = false; return true;
        }
        /** @brief Sampling cannot record a replacement graph. */
        bool endCapture() override
        {
            if (!allow_capture) { ADD_FAILURE(); return false; }
            if (!capturing) return false;
            capturing = false; return true;
        }
        /** @brief Construction must already be complete before observation. */
        bool instantiate() override
        {
            if (!allow_capture) { ADD_FAILURE(); return false; }
            executable = true; return true;
        }
        /** @brief The sampler must explicitly name the graph-owned replay stream. */
        bool launch() override { ADD_FAILURE(); return false; }
        /** @brief Count complete retained submissions and inject warmup/timed failures. */
        bool launchOnStream(void *actual) const override
        {
            EXPECT_EQ(actual, stream);
            ++launches;
            if (before_launch) before_launch();
            return launches != fail_launch;
        }
        /** @return Exact producer stream, including the invalid-null sentinel test. */
        void *executionStream() const noexcept override { return stream; }
        /** @brief Sampling never updates graph topology. */
        GraphUpdateResult tryUpdate() override { ADD_FAILURE(); return GraphUpdateResult::Failed; }
        /** @return No update API is needed by a retained sample. */
        bool supportsExecutableUpdate() const noexcept override { return false; }
        /** @return Explicit readiness state used by admission tests. */
        bool hasExecutable() const override { return executable; }
        /** @return No device memory is owned by this unit-test double. */
        size_t residentMemoryBytes() const noexcept override { return 0; }
        /** @return One immutable sample operation. */
        size_t nodeCount() const override { return nodes; }
        /** @brief Sampling may not reset the owner's executable. */
        void reset() override
        {
            if (!allow_capture) { ADD_FAILURE(); return; }
            executable = false; capturing = false;
        }
        /** @return Diagnostic identity, never a performance score. */
        const char *backendName() const override { return "measurement-unit-double"; }
    };

    /** @return Explicit synthetic work declaration, not real throughput evidence. */
    PlanningMeasurementWork work()
    {
        return {PlanningWorkUnit::ArithmeticOperations, 1024, "unit ordering oracle"};
    }

    /** @brief Only external Ordering events are permitted around the compact clock readback. */
    class CapturedEventWorker final : public llaminar2::testing::MockWorkerGPUContext
    {
    public:
        CapturedEventWorker() : MockWorkerGPUContext(7) {}
        SampleGraph *graph = nullptr;
        Failure failure = Failure::None;
        int creates = 0, destroys = 0, ordering_records = 0, queries = 0;
        int fail_create_at = -1, token = 0;
        bool pending = false, completed = false;
        /** @brief A single non-timing event is minted on the exact original worker. */
        void *createEvent(GPUEventPurpose purpose) override
        {
            EXPECT_TRUE(ownsCurrentThread()); EXPECT_EQ(deviceOrdinal(), 7);
            EXPECT_EQ(purpose, GPUEventPurpose::Ordering);
            return ++creates == fail_create_at ? nullptr : &token;
        }
        /** @brief Every successful event handle retires exactly once on its owner. */
        void destroyEvent(void *event) override
        {
            EXPECT_TRUE(ownsCurrentThread()); EXPECT_EQ(event, &token); EXPECT_EQ(destroys, 0); ++destroys;
        }
        /** @brief Ordinary events delimit completion outside capture, never timestamp native transfers. */
        bool recordEventChecked(void *event, void *stream) override
        {
            EXPECT_TRUE(ownsCurrentThread()); EXPECT_EQ(event, &token); EXPECT_NE(graph, nullptr);
            EXPECT_EQ(stream, graph->stream); EXPECT_FALSE(graph->capturing);
            if (failure == Failure::Record) return false;
            ++ordering_records; EXPECT_EQ(graph->launches, ordering_records);
            pending = true; completed = false; return true;
        }
        /** @brief The sampler polls only its external completion, with no blocking wait substitution. */
        bool queryEventChecked(void *event, bool &ready) override
        {
            EXPECT_TRUE(ownsCurrentThread()); EXPECT_EQ(event, &token); EXPECT_FALSE(graph->capturing);
            ++queries; if (failure == Failure::Query) return false;
            ready = !pending; pending = false; if (ready) completed = true; return true;
        }
        /** @brief Clock buffers replace native event timestamps entirely. */
        float eventElapsedTime(void *, void *) override { ADD_FAILURE(); return 0; }
        /** @brief No blocking event wait belongs to this setup owner. */
        void synchronizeEvent(void *) override { ADD_FAILURE(); }
        /** @brief No stream-wide wait belongs to this setup owner. */
        void synchronizeStream(void *) override { ADD_FAILURE(); }
        /** @brief No device-wide wait belongs to this setup owner. */
        void synchronize() override { ADD_FAILURE(); }
    };

    /** @brief Device-free clock owner authenticates captured slots, live bytes and terminal visibility. */
    class CapturedClockBuffer final : public PlanningGPUClockBuffer
    {
    public:
        /** @brief Bind fixed storage to the fixture's exact worker and stream. */
        explicit CapturedClockBuffer(CapturedEventWorker &owner, size_t slots = 6)
            : worker(owner), slots_(slots), values(slots)
        {
            for (size_t index = 0; index < slots; ++index) values[index] = 100 + (index / 2) * 100 + (index % 2) * 6;
        }
        CapturedEventWorker &worker;
        size_t slots_;
        uint64_t frequency = 1000;
        std::vector<uint64_t> values;
        std::vector<size_t> recorded_slots;
        int readbacks = 0;
        mutable int reads = 0;
        bool fail_record = false, fail_readback = false;
        /** @return Exact admitted timestamp capacity, never payload capacity. */
        size_t slotCount() const noexcept override { return slots_; }
        /** @return Fixed wall clock, independent of shader DVFS. */
        uint64_t ticksPerSecond() const noexcept override { return frequency; }
        /** @brief Clock writes must be recorded inside capture on the original stream. */
        void record(size_t slot, ExplicitGPUStream stream) override
        {
            EXPECT_TRUE(worker.ownsCurrentThread()); EXPECT_TRUE(worker.graph->capturing);
            EXPECT_EQ(stream.get(), worker.graph->stream); EXPECT_LT(slot, slots_);
            if (fail_record) throw std::runtime_error("recorded clock marker failure");
            EXPECT_EQ(slot, recorded_slots.size()); recorded_slots.push_back(slot);
        }
        /** @brief One compact readback follows the measured replay, before its external completion. */
        void enqueueReadback(ExplicitGPUStream stream) override
        {
            EXPECT_TRUE(worker.ownsCurrentThread()); EXPECT_FALSE(worker.graph->capturing);
            EXPECT_EQ(stream.get(), worker.graph->stream);
            EXPECT_EQ(worker.graph->launches % 2, 0);
            EXPECT_EQ(worker.ordering_records, worker.graph->launches - 1);
            if (fail_readback) throw std::runtime_error("recorded clock readback failure");
            ++readbacks;
        }
        /** @brief Host ticks are unreadable until the measured replay and compact readback complete. */
        std::span<const uint64_t> ticks() const override
        {
            if (!worker.completed || worker.graph->launches % 2 != 0 ||
                worker.ordering_records != worker.graph->launches || readbacks != worker.graph->launches / 2)
                throw std::logic_error("Clock readback observed before exact terminal completion");
            EXPECT_TRUE(worker.ownsCurrentThread()); ++reads; return values;
        }
    };

    /** @brief Record one successor graph, keeping captured clock slots separate from replay. */
    void captureClockWindows(PlanningCapturedGPUClockSampler &sampler, SampleGraph &graph, CapturedClockBuffer &clock)
    {
        graph.reset(); clock.recorded_slots.clear();
        ASSERT_TRUE(graph.beginCapture());
        for (size_t window = 0; window < sampler.windowCount(); ++window)
        { sampler.recordStart(window); sampler.recordStop(window); }
        ASSERT_TRUE(graph.endCapture()); ASSERT_TRUE(graph.instantiate()); sampler.seal(graph);
    }

    /** @return Actual immutable PMA certificate for the mock graph family's declared owners. */
    std::shared_ptr<PhysicalMemoryAuthority> projectionClockMemory()
    {
        const PhysicalMemoryResource host{0,DeviceId::cpu(),1ull<<40,1ull<<40};
        const PhysicalMemoryResource gpu{0,DeviceId::rocm(7),1ull<<40,1ull<<40};
        PhysicalMemoryPlanBuilder builder;
        PlanningExecutionMeasurement::contributeGPUObservationMemory(host,gpu,builder);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
    }

    /** @brief One deterministic device-clock transaction; host time never supplies service. */
    struct ProjectionClockState
    {
        void *stream = nullptr;
        int graphs_live = 0, graphs_retired = 0, body_launches = 0, parent_launches = 0;
        int event_creates = 0, event_destroys = 0, ordering = 0, readbacks = 0, reads = 0, parent_bodies = 4;
        int fail_create = -1, fail_launch = -1;
        bool fail_instantiate = false, fail_readback = false, fail_query = false, completed = false;
        double host_enqueue_gap = 0, host_elapsed = 0;
        std::vector<const IGPUGraphCapture *> steps;
        std::vector<size_t> slots;
        std::vector<uint64_t> ticks{100, 118};
    };

    /** @brief Fake native parent preserves three copies of the SAME original body. */
    class ProjectionClockGraph final : public SampleGraph
    {
    public:
        ProjectionClockState &state;
        bool body;
        /** @brief Track each independently owned fragment and parent, without native allocation. */
        ProjectionClockGraph(ProjectionClockState &s, bool original) : state(s), body(original)
        { ++state.graphs_live; stream=state.stream; executable=false; allow_capture=true; }
        /** @brief Preserve clock-backing retirement after all graph borrowers. */
        ~ProjectionClockGraph() override { --state.graphs_live; ++state.graphs_retired; }
        /** @brief Graph context preparation does not allocate device storage in a Unit double. */
        bool prepareRuntimeContextStorage(const std::shared_ptr<PhysicalMemoryAuthority> &) override { return true; }
        /** @brief Inject instance failure without permitting an eager substitute. */
        bool instantiate() override
        { if(state.fail_instantiate)return false; executable=true; return true; }
        /** @brief Authenticate ordinary six-step or freshly published three-step parent ordering. */
        bool buildOrderedTimelineTransaction(std::span<const GPUOrderedTimelineStep> ordered,
            GPUOrderedTimelineInstrumentation = GPUOrderedTimelineInstrumentation::Disabled) override
        {
            if(ordered.size()==3){
                EXPECT_NE(ordered[0].capture,ordered[1].capture);EXPECT_NE(ordered[0].capture,ordered[2].capture);
                state.parent_bodies=1;
            }else{
                EXPECT_EQ(ordered.size(),6u); if(ordered.size()!=6)return false;
                EXPECT_EQ(ordered[0].capture,ordered[2].capture);
                EXPECT_EQ(ordered[2].capture,ordered[3].capture);EXPECT_EQ(ordered[3].capture,ordered[4].capture);
                EXPECT_NE(ordered[1].capture,ordered[5].capture);EXPECT_NE(ordered[1].capture,ordered[0].capture);
            }
            for(const auto &step:ordered){EXPECT_TRUE(step.valid());state.steps.push_back(step.capture);}
            nodes=ordered.size();return true;
        }
        /** @brief Host enqueue gaps advance only the host oracle, not the device clock bank. */
        bool launchOnStream(void *actual) const override
        {
            EXPECT_TRUE(executable);EXPECT_EQ(actual,state.stream);state.host_elapsed+=state.host_enqueue_gap;
            if(body)++state.body_launches;
            else {++state.parent_launches;state.body_launches+=state.parent_bodies;}
            ++launches;
            return (body?state.body_launches:state.parent_launches)!=state.fail_launch;
        }
        /** @return Distinct fixed mock bytes, summed across all four native graph owners. */
        size_t residentMemoryBytes() const noexcept override { return 17; }
    };

    /** @brief Exact original worker provides Ordering completion only, never event timing. */
    class ProjectionClockWorker final : public llaminar2::testing::MockWorkerGPUContext
    {
    public:
        ProjectionClockState &state;
        int token=0, created_graphs=0;
        /** @brief Bind the whole family to one original worker stream. */
        explicit ProjectionClockWorker(ProjectionClockState &s):MockWorkerGPUContext(7),state(s)
        {state.stream=defaultStream();}
        /** @brief Create only the two marker fragments and one parent on their exact worker. */
        std::unique_ptr<IGPUGraphCapture> createGraphCapture(void *stream) override
        {
            EXPECT_TRUE(ownsCurrentThread());EXPECT_EQ(stream,state.stream);
            if(++created_graphs==state.fail_create)throw std::runtime_error("recorded graph creation failure");
            return std::make_unique<ProjectionClockGraph>(state,false);
        }
        /** @brief Timing events are forbidden; only one completion event is retained. */
        void *createEvent(GPUEventPurpose purpose) override
        {EXPECT_EQ(purpose,GPUEventPurpose::Ordering);++state.event_creates;return &token;}
        /** @brief Retire the completion handle on its original worker. */
        void destroyEvent(void *event) override
        {EXPECT_TRUE(ownsCurrentThread());EXPECT_EQ(event,&token);++state.event_destroys;}
        /** @brief Only the complete parent and compact readback receive an external ordering edge. */
        bool recordEventChecked(void *event,void *stream) override
        {EXPECT_TRUE(ownsCurrentThread());EXPECT_FALSE(isDeviceGraphCaptureActive());EXPECT_EQ(event,&token);EXPECT_EQ(stream,state.stream);++state.ordering;state.completed=false;return true;}
        /** @brief Publish terminal visibility without blocking device synchronization. */
        bool queryEventChecked(void *event,bool &ready) override
        {EXPECT_EQ(event,&token);if(state.fail_query)return false;ready=true;state.completed=true;return true;}
        /** @brief Reject the former host-enqueue-inflated timing source. */
        float eventElapsedTime(void *,void *) override {ADD_FAILURE()<<"Host/event interval cannot price compute";return 999999;}
    };

    /** @brief Owned compact clock bank remains unreadable until measured-parent completion. */
    class ProjectionClockBuffer final : public PlanningGPUClockBuffer
    {
    public:
        ProjectionClockWorker &worker;
        /** @brief Retain the original worker through its compact timestamp lifetime. */
        explicit ProjectionClockBuffer(ProjectionClockWorker &w):worker(w){}
        /** @brief Embedded marker pointers cannot outlive their backing storage. */
        ~ProjectionClockBuffer() override {if(!worker.state.slots.empty())EXPECT_EQ(worker.state.graphs_live,0);}
        /** @return One exact start/stop pair for the whole three-body batch. */
        size_t slotCount() const noexcept override {return 2;}
        /** @return Fixed synthetic clock; host gaps and event intervals cannot change it. */
        uint64_t ticksPerSecond() const noexcept override {return 1000;}
        /** @brief Marker slots are recorded only during exact-worker native capture. */
        void record(size_t slot,ExplicitGPUStream stream) override
        {EXPECT_TRUE(worker.ownsCurrentThread());EXPECT_TRUE(worker.isDeviceGraphCaptureActive());EXPECT_EQ(stream.get(),worker.state.stream);worker.state.slots.push_back(slot);}
        /** @brief A single compact readback follows the measured parent outside capture. */
        void enqueueReadback(ExplicitGPUStream stream) override
        {EXPECT_EQ(stream.get(),worker.state.stream);if(worker.state.fail_readback)throw std::runtime_error("clock readback failure");++worker.state.readbacks;worker.state.completed=false;}
        /** @brief Refuse clock observation before the ordering completion. */
        std::span<const uint64_t> ticks() const override
        {if(!worker.state.completed||worker.state.readbacks!=worker.state.parent_launches)throw std::logic_error("premature clock bank");++worker.state.reads;return worker.state.ticks;}
    };

}

TEST(PlanningExecutionMeasurement, RejectsInvalidWorkBeforeExecution)
{
    for (double value : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max()})
        EXPECT_THROW(PlanningMeasurementWork(PlanningWorkUnit::Bytes, value, "fixture"), std::invalid_argument);
    EXPECT_THROW(PlanningMeasurementWork(static_cast<PlanningWorkUnit>(200), 1, "fixture"), std::invalid_argument);
    EXPECT_THROW(PlanningMeasurementWork(PlanningWorkUnit::Bytes, 1, " \t\n"), std::invalid_argument);
    EXPECT_THROW(PlanningExecutionMeasurement::cpu(work(), {}), std::invalid_argument);
}

/** @test Fresh host bytes precede every original invocation and follow the prior exact completion. */
TEST(PlanningExecutionMeasurement, PublishedHostInputsChangeOnlyBetweenRetiredInvocations)
{
    for(int replay=0;replay<2;++replay){ProjectionClockState state;state.host_enqueue_gap=1e9;ProjectionClockWorker worker(state);
        worker.submitAndWait([&]{int publications=0;
            auto token=PlanningExecutionMeasurement::recordPublishedGPU(work(),worker,
                std::make_unique<ProjectionClockBuffer>(worker),std::make_unique<ProjectionClockGraph>(state,true),[&]{
                    EXPECT_EQ(state.body_launches,publications);EXPECT_EQ(state.ordering,publications);
                    EXPECT_EQ(state.readbacks,publications);if(publications)EXPECT_TRUE(state.completed);
                    ++publications;state.ticks={uint64_t(100*publications),uint64_t(100*publications+18)};
                });
            token.prepare(projectionClockMemory());EXPECT_EQ(publications,0);EXPECT_EQ(state.body_launches,0);
            const auto observation=token.observe();EXPECT_EQ(publications,4);EXPECT_EQ(state.parent_launches,4);
            EXPECT_EQ(state.body_launches,4);EXPECT_EQ(state.ordering,4);EXPECT_EQ(state.readbacks,4);EXPECT_EQ(state.reads,4);
            EXPECT_EQ(observation.completedWork(),3*work().perInvocation());EXPECT_DOUBLE_EQ(observation.elapsedSeconds(),.054);
            EXPECT_THROW(token.observe(),std::logic_error);
        });
        EXPECT_EQ(state.graphs_live,0);EXPECT_EQ(state.graphs_retired,4);EXPECT_EQ(state.event_destroys,1);
    }
}

/** @test A failed fresh publication or original parent cannot retry or publish partial compute service. */
TEST(PlanningExecutionMeasurement, PublishedObservationPropagatesNativeAndPublicationFailuresWithoutRetry)
{
    for(int fail_at:{1,2,3,4}){ProjectionClockState state;ProjectionClockWorker worker(state);int publications=0;
        worker.submitAndWait([&]{auto token=PlanningExecutionMeasurement::recordPublishedGPU(work(),worker,
            std::make_unique<ProjectionClockBuffer>(worker),std::make_unique<ProjectionClockGraph>(state,true),[&]{
                if(++publications==fail_at)throw std::runtime_error("fresh publication failure");});
            token.prepare(projectionClockMemory());
            EXPECT_THROW(token.observe(),std::runtime_error);
            EXPECT_THROW(token.observe(),std::logic_error);
        });
        EXPECT_EQ(publications,fail_at);EXPECT_EQ(state.body_launches,fail_at-1);EXPECT_EQ(state.parent_launches,fail_at-1);
        EXPECT_EQ(state.graphs_live,0);EXPECT_EQ(state.event_destroys,1);
    }
    for(int failure=0;failure<3;++failure){ProjectionClockState state;ProjectionClockWorker worker(state);
        worker.submitAndWait([&]{auto token=PlanningExecutionMeasurement::recordPublishedGPU(work(),worker,
            std::make_unique<ProjectionClockBuffer>(worker),std::make_unique<ProjectionClockGraph>(state,true),[]{});
            token.prepare(projectionClockMemory());if(failure==0)state.fail_launch=2;if(failure==1)state.fail_query=true;if(failure==2)state.fail_readback=true;
            EXPECT_THROW(token.observe(),std::exception);
            EXPECT_THROW(token.observe(),std::logic_error);
        });EXPECT_EQ(state.graphs_live,0);EXPECT_EQ(state.event_destroys,1);
    }
    ProjectionClockState state;ProjectionClockWorker worker(state);
    worker.submitAndWait([&]{
        EXPECT_THROW(PlanningExecutionMeasurement::recordPublishedGPU(work(),worker,
            std::make_unique<ProjectionClockBuffer>(worker),std::make_unique<ProjectionClockGraph>(state,true),{}),std::invalid_argument);
    });
}

TEST(PlanningExecutionMeasurement, CpuExcludesWarmupAndNeverRetriesFailedWork)
{
    int calls = 0;
    const auto observation = PlanningExecutionMeasurement::cpu(work(), [&] { ++calls; return true; });
    EXPECT_EQ(calls, 1 + PlanningExecutionMeasurement::kTimedInvocations);
    EXPECT_EQ(observation.completedWork(), 1024 * PlanningExecutionMeasurement::kTimedInvocations);
    EXPECT_GT(observation.elapsedSeconds(), 0);
    for (const int fail : {1, 2, 3, 4})
    {
        calls = 0;
        EXPECT_THROW(PlanningExecutionMeasurement::cpu(work(), [&] { return ++calls != fail; }), std::runtime_error);
        EXPECT_EQ(calls, fail);
    }
}

TEST(PlanningExecutionMeasurement, NativeTimingAndExactStreamOrderingAreBackendSymmetric)
{
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        EventBackend backend(type);
        SampleGraph graph;
        backend.expected_stream = graph.stream;
        const auto observation = PlanningExecutionMeasurement::gpu(work(), backend, DeviceId(type, 7), graph);
        EXPECT_EQ(graph.launches, 1 + PlanningExecutionMeasurement::kTimedInvocations);
        EXPECT_EQ(backend.creates, 2);
        EXPECT_EQ(backend.destroys, 2);
        EXPECT_EQ(observation.completedWork(), 3072);
        EXPECT_DOUBLE_EQ(observation.elapsedSeconds(), .006);
    }
}

TEST(PlanningExecutionMeasurement, RejectsUnreadyOrMismatchedGpuBeforeAnyNativeWork)
{
    for (int fault = 0; fault < 4; ++fault)
    {
        EventBackend backend(DeviceType::CUDA);
        SampleGraph graph;
        if (fault == 0) graph.executable = false;
        if (fault == 1) graph.stream = nullptr;
        const auto device = fault == 2 ? DeviceId::rocm(7) :
                            fault == 3 ? DeviceId::cpu() : DeviceId::cuda(7);
        EXPECT_THROW(PlanningExecutionMeasurement::gpu(work(), backend, device, graph), std::invalid_argument);
        EXPECT_EQ(backend.creates, 0);
        EXPECT_EQ(graph.launches, 0);
    }
}

TEST(PlanningExecutionMeasurement, EveryNativeFailureRetiresEventsWithoutInventingAnObservation)
{
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    for (auto failure : {Failure::FirstEvent, Failure::SecondEvent, Failure::Record,
                         Failure::Query, Failure::Elapsed, Failure::ZeroTime})
    {
        EventBackend backend(type);
        backend.failure = failure;
        SampleGraph graph;
        backend.expected_stream = graph.stream;
        EXPECT_THROW(PlanningExecutionMeasurement::gpu(work(), backend, DeviceId(type, 7), graph), std::exception);
        EXPECT_EQ(backend.destroys, backend.creates -
            (failure == Failure::FirstEvent || failure == Failure::SecondEvent ? 1 : 0));
    }
    for (const int fail : {1, 2, 3, 4})
    {
        EventBackend backend(DeviceType::ROCm);
        SampleGraph graph;
        graph.fail_launch = fail;
        backend.expected_stream = graph.stream;
        EXPECT_THROW(PlanningExecutionMeasurement::gpu(work(), backend, DeviceId::rocm(7), graph), std::runtime_error);
        EXPECT_EQ(graph.launches, fail);
        EXPECT_EQ(backend.destroys, 2);
    }
}

/** @test Device-clock windows retain fixed storage and one Ordering event across replay/reset. */
TEST(PlanningExecutionMeasurement, RetainedPairClockFamilyReusesStorageAndIndividualIntervals)
{
    CapturedEventWorker worker; SampleGraph graph; graph.allow_capture = true; graph.nodes = 18;
    worker.graph = &graph; CapturedClockBuffer clock(worker, 18);
    worker.submitAndWait([&] {
        {
            PlanningCapturedGPUClockSampler sampler(worker, ExplicitGPUStream(graph.stream), clock);
            EXPECT_EQ(sampler.windowCount(), 9u);
            const std::vector<PlanningMeasurementWork> windows(9, work());
            for (int pair = 0; pair < 3; ++pair)
            {
                captureClockWindows(sampler, graph, clock);
                EXPECT_EQ(clock.recorded_slots.size(), 18u);
                for (int replay = 0; replay < 2; ++replay)
                {
                    const auto observations = sampler.samples(windows, graph);
                    ASSERT_EQ(observations.size(), 9u); // Three raw samples for each of three anchors.
                    EXPECT_EQ(worker.creates, 1); EXPECT_EQ(worker.destroys, 0);
                    EXPECT_EQ(graph.launches, (pair * 2 + replay + 1) * 2);
                    EXPECT_EQ(worker.ordering_records, graph.launches);
                    EXPECT_EQ(worker.queries, 2 * graph.launches);
                    EXPECT_EQ(clock.readbacks, graph.launches / 2); EXPECT_EQ(clock.reads, clock.readbacks);
                    for (const auto &observation : observations)
                    { EXPECT_EQ(observation.completedWork(), 1024); EXPECT_DOUBLE_EQ(observation.elapsedSeconds(), .006); }
                }
            }
            graph.reset(); // Retire captured clock borrowers before sampler/storage ownership.
        }
        EXPECT_EQ(worker.destroys, 1);
    });
}

/** @test Event creation, native submissions, clock markers and completion failures remain fatal. */
TEST(PlanningExecutionMeasurement, RetainedPairClockFamilyRejectsFailuresAndForeignStreams)
{
    {
        CapturedEventWorker worker; worker.fail_create_at = 1;
        SampleGraph graph; worker.graph = &graph; CapturedClockBuffer clock(worker);
        worker.submitAndWait([&] {
            EXPECT_THROW(PlanningCapturedGPUClockSampler(worker, ExplicitGPUStream(graph.stream), clock), std::runtime_error);
            EXPECT_EQ(worker.creates, 1); EXPECT_EQ(worker.destroys, 0); EXPECT_EQ(graph.launches, 0);
        });
    }
    for (const auto failure : {Failure::Record, Failure::Query})
    {
        CapturedEventWorker worker; SampleGraph graph; graph.allow_capture = true; graph.nodes = 6;
        worker.graph = &graph; CapturedClockBuffer clock(worker);
        worker.submitAndWait([&] {
            {
                PlanningCapturedGPUClockSampler sampler(worker, ExplicitGPUStream(graph.stream), clock);
                captureClockWindows(sampler, graph, clock); worker.failure = failure;
                const std::vector<PlanningMeasurementWork> windows(3, work());
                EXPECT_THROW(sampler.samples(windows, graph), std::runtime_error);
                EXPECT_EQ(graph.launches, 1); EXPECT_EQ(clock.readbacks, 0); EXPECT_EQ(clock.reads, 0); graph.reset();
            }
            EXPECT_EQ(worker.destroys, 1);
        });
    }
    for (int failed_launch = 1; failed_launch <= 2; ++failed_launch)
    {
        CapturedEventWorker worker; SampleGraph graph; graph.allow_capture = true; graph.nodes = 6;
        graph.fail_launch = failed_launch; worker.graph = &graph; CapturedClockBuffer clock(worker);
        worker.submitAndWait([&] {
            {
                PlanningCapturedGPUClockSampler sampler(worker, ExplicitGPUStream(graph.stream), clock);
                captureClockWindows(sampler, graph, clock);
                const std::vector<PlanningMeasurementWork> windows(3, work());
                EXPECT_THROW(sampler.samples(windows, graph), std::runtime_error);
                EXPECT_EQ(graph.launches, failed_launch); EXPECT_EQ(clock.readbacks, 0); graph.reset();
            }
            EXPECT_EQ(worker.destroys, 1);
        });
    }
    CapturedEventWorker worker; SampleGraph graph; graph.allow_capture = true; graph.nodes = 6;
    worker.graph = &graph; CapturedClockBuffer clock(worker);
    worker.submitAndWait([&] {
        PlanningCapturedGPUClockSampler sampler(worker, ExplicitGPUStream(graph.stream), clock);
        ASSERT_TRUE(graph.beginCapture()); clock.fail_record = true;
        EXPECT_THROW(sampler.recordStart(0), std::runtime_error); clock.fail_record = false;
        graph.reset(); captureClockWindows(sampler, graph, clock); clock.fail_readback = true;
        const std::vector<PlanningMeasurementWork> windows(3, work());
        EXPECT_THROW(sampler.samples(windows, graph), std::runtime_error);
        EXPECT_EQ(graph.launches, 2); EXPECT_EQ(clock.reads, 0); graph.reset();
    });
}

/** @test Incomplete capture, foreign worker/graph/stream and invalid order cannot publish observations. */
TEST(PlanningExecutionMeasurement, RetainedPairClockFamilyRejectsWindowAndCaptureOrdering)
{
    CapturedEventWorker worker; SampleGraph graph; graph.allow_capture = true; graph.nodes = 6;
    worker.graph = &graph; CapturedClockBuffer clock(worker);
    EXPECT_THROW(PlanningCapturedGPUClockSampler(worker, ExplicitGPUStream(graph.stream), clock), std::invalid_argument);
    worker.submitAndWait([&] {
        PlanningCapturedGPUClockSampler sampler(worker, ExplicitGPUStream(graph.stream), clock);
        const std::vector<PlanningMeasurementWork> windows(3, work());
        EXPECT_THROW(sampler.samples(windows, graph), std::invalid_argument);
        EXPECT_THROW(sampler.recordStart(1), std::invalid_argument);
        EXPECT_THROW(sampler.recordStop(0), std::invalid_argument);
        ASSERT_TRUE(graph.beginCapture()); sampler.recordStart(0);
        EXPECT_THROW(sampler.recordStart(0), std::invalid_argument);
        EXPECT_THROW(sampler.recordStop(1), std::invalid_argument);
        EXPECT_THROW(sampler.seal(graph), std::invalid_argument); sampler.recordStop(0);
        sampler.recordStart(1); sampler.recordStop(1); sampler.recordStart(2); sampler.recordStop(2);
        EXPECT_THROW(sampler.recordStart(3), std::invalid_argument);
        ASSERT_TRUE(graph.endCapture()); ASSERT_TRUE(graph.instantiate());
        graph.nodes = 5;
        EXPECT_THROW(sampler.seal(graph), std::invalid_argument); graph.nodes = 6;
        graph.executable = false;
        EXPECT_THROW(sampler.seal(graph), std::invalid_argument); graph.executable = true;
        int other_stream; auto *original_stream = graph.stream; graph.stream = &other_stream;
        EXPECT_THROW(sampler.seal(graph), std::invalid_argument); graph.stream = original_stream; sampler.seal(graph);
        SampleGraph foreign; foreign.stream = graph.stream; foreign.nodes = 6;
        EXPECT_THROW(sampler.seal(foreign), std::invalid_argument);
        EXPECT_THROW(sampler.samples(windows, foreign), std::invalid_argument);
        EXPECT_THROW(sampler.samples(std::span<const PlanningMeasurementWork>(windows).first(2), graph), std::invalid_argument);
        graph.stream = &other_stream;
        EXPECT_THROW(sampler.samples(windows, graph), std::invalid_argument); graph.stream = original_stream;
        EXPECT_THROW(clock.ticks(), std::logic_error); // Mock rejects an early host observation.
        EXPECT_EQ(sampler.samples(windows, graph).size(), 3u);
        graph.reset(); clock.recorded_slots.clear(); ASSERT_TRUE(graph.beginCapture()); sampler.recordStart(0);
        EXPECT_THROW(sampler.samples(windows, graph), std::invalid_argument);
        clock.fail_record = true;
        EXPECT_THROW(sampler.recordStop(0), std::runtime_error); clock.fail_record = false;
        sampler.recordStop(0); sampler.recordStart(1); sampler.recordStop(1); sampler.recordStart(2); sampler.recordStop(2);
        ASSERT_TRUE(graph.endCapture()); ASSERT_TRUE(graph.instantiate()); sampler.seal(graph);
        EXPECT_EQ(sampler.samples(windows, graph).size(), 3u); graph.reset();
    });
    EXPECT_EQ(worker.destroys, 1);
    CapturedEventWorker original; SampleGraph retained; original.graph = &retained; CapturedClockBuffer buffer(original);
    std::unique_ptr<PlanningCapturedGPUClockSampler> owned;
    original.submitAndWait([&] { owned = std::make_unique<PlanningCapturedGPUClockSampler>(original, ExplicitGPUStream(retained.stream), buffer); });
    const std::vector<PlanningMeasurementWork> windows(3, work());
    EXPECT_THROW(owned->recordStart(0), std::invalid_argument);
    EXPECT_THROW(owned->recordStop(0), std::invalid_argument);
    EXPECT_THROW(owned->seal(retained), std::invalid_argument);
    EXPECT_THROW(owned->samples(windows, retained), std::invalid_argument);
    original.submitAndWait([&] { owned.reset(); }); EXPECT_EQ(original.destroys, 1);
}

/** @test Missing, zero or nonmonotonic timestamps and invalid physical clock geometry fail closed. */
TEST(PlanningExecutionMeasurement, RetainedPairClockFamilyRejectsInvalidClockReadbacks)
{
    for (int defect = 0; defect < 3; ++defect)
    {
        CapturedEventWorker worker; SampleGraph graph; worker.graph = &graph; CapturedClockBuffer clock(worker);
        if (defect == 0) clock.slots_ = 0;
        if (defect == 1) clock.slots_ = 5;
        if (defect == 2) clock.frequency = 0;
        worker.submitAndWait([&] {
            EXPECT_THROW(PlanningCapturedGPUClockSampler(worker, ExplicitGPUStream(graph.stream), clock), std::invalid_argument);
            EXPECT_EQ(worker.creates, 0);
        });
    }
    for (int defect = 0; defect < 4; ++defect)
    {
        SCOPED_TRACE(defect); CapturedEventWorker worker; SampleGraph graph; graph.allow_capture = true; graph.nodes = 6;
        worker.graph = &graph; CapturedClockBuffer clock(worker);
        if (defect == 0) clock.values.pop_back();
        if (defect == 1) clock.values[0] = 0;
        if (defect == 2) clock.values[1] = clock.values[0];
        if (defect == 3) clock.values[1] = clock.values[0] - 1;
        worker.submitAndWait([&] {
            PlanningCapturedGPUClockSampler sampler(worker, ExplicitGPUStream(graph.stream), clock);
            captureClockWindows(sampler, graph, clock);
            const std::vector<PlanningMeasurementWork> windows(3, work());
            EXPECT_THROW(sampler.samples(windows, graph), std::exception);
            EXPECT_EQ(clock.readbacks, 1); EXPECT_EQ(worker.ordering_records, 2); graph.reset();
        });
    }
}

/** @test Preparation never executes; each parent retains one warmup before three measured original bodies. */
TEST(PlanningExecutionMeasurement, PreparedGPUObservationWarmupAndIsolationReplay)
{
    ProjectionClockState first,second;ProjectionClockWorker a(first),b(second);
    std::unique_ptr<PlanningPreparedGPUObservation> one,two;
    a.submitAndWait([&]{one=std::make_unique<PlanningPreparedGPUObservation>(PlanningExecutionMeasurement::recordGPU(work(),a,
        std::make_unique<ProjectionClockBuffer>(a),std::make_unique<ProjectionClockGraph>(first,true)));one->prepare(projectionClockMemory());});
    b.submitAndWait([&]{two=std::make_unique<PlanningPreparedGPUObservation>(PlanningExecutionMeasurement::recordGPU(work(),b,
        std::make_unique<ProjectionClockBuffer>(b),std::make_unique<ProjectionClockGraph>(second,true)));two->prepare(projectionClockMemory());});
    EXPECT_EQ(first.body_launches,0);EXPECT_EQ(second.body_launches,0);
    a.submitAndWait([&]{EXPECT_EQ(one->observe().completedWork(),3*work().perInvocation());
        EXPECT_EQ(one->residentMemoryBytes(),68u);EXPECT_EQ(one->graphNodeCount(),1u);
        EXPECT_THROW(one->observe(),std::logic_error);one.reset();});
    EXPECT_EQ(first.body_launches,4);EXPECT_EQ(first.parent_launches,1);EXPECT_EQ(second.parent_launches,0);
    b.submitAndWait([&]{EXPECT_DOUBLE_EQ(two->observe().elapsedSeconds(),.018);two.reset();});
    for(const auto *s:{&first,&second}){EXPECT_EQ(s->graphs_live,0);EXPECT_EQ(s->graphs_retired,4);
        EXPECT_EQ(s->event_creates,1);EXPECT_EQ(s->event_destroys,1);EXPECT_EQ(s->readbacks,1);EXPECT_EQ(s->reads,1);}
}

/** @test Host enqueue delays cannot enter the fixed device interval, including fresh reset/replay owners. */
TEST(PlanningExecutionMeasurement, PreparedGPUClockBatchExcludesHostGapsAndPreservesThreeBodies)
{
    for(const double gap:{0.,1000.,1.e9}){
        ProjectionClockState state;state.host_enqueue_gap=gap;ProjectionClockWorker worker(state);
        worker.submitAndWait([&]{
            auto token=PlanningExecutionMeasurement::recordGPU(work(),worker,std::make_unique<ProjectionClockBuffer>(worker),
                std::make_unique<ProjectionClockGraph>(state,true));
            EXPECT_THROW(token.observe(),std::logic_error);
            EXPECT_THROW(token.prepare({}),std::invalid_argument);
            EXPECT_EQ(state.body_launches,0);EXPECT_EQ(state.parent_launches,0);
            token.prepare(projectionClockMemory());
            EXPECT_EQ(state.body_launches,0);EXPECT_EQ(state.ordering,0);
            EXPECT_THROW(token.prepare(projectionClockMemory()),std::logic_error);
            const auto measured=token.observe();EXPECT_DOUBLE_EQ(measured.elapsedSeconds(),.018);
            EXPECT_DOUBLE_EQ(measured.completedWork(),3*work().perInvocation());
            EXPECT_EQ(state.body_launches,4);EXPECT_EQ(state.parent_launches,1);
            EXPECT_EQ(state.ordering,1);EXPECT_EQ(state.readbacks,1);EXPECT_EQ(state.reads,1);
            EXPECT_DOUBLE_EQ(state.host_elapsed,gap);EXPECT_EQ(state.slots,(std::vector<size_t>{0,1}));
            EXPECT_THROW(token.observe(),std::logic_error);
        });
        EXPECT_EQ(state.graphs_live,0);EXPECT_EQ(state.graphs_retired,4);EXPECT_EQ(state.event_destroys,1);
    }
}

/** @test Original worker/stream, moved-from and consumed state all remain strict boundaries. */
TEST(PlanningExecutionMeasurement, PreparedGPUObservationRejectsForeignMovedAndConsumed)
{
    // Already materialized bodies must not enter this recorded-only owner.
    // Rejection precedes capture, native preparation and any execution.
    CapturedEventWorker rejected_worker;
    rejected_worker.submitAndWait([&]{
        auto graph=std::make_unique<SampleGraph>();
        EXPECT_TRUE(graph->hasExecutable());
        EXPECT_THROW(PlanningExecutionMeasurement::recordGPU(work(),rejected_worker,
            std::make_unique<CapturedClockBuffer>(rejected_worker,2),std::move(graph)),std::invalid_argument);
    });
    EXPECT_EQ(rejected_worker.creates,0);
    ProjectionClockState state;ProjectionClockWorker worker(state);std::unique_ptr<PlanningPreparedGPUObservation> original;
    ProjectionClockGraph *body=nullptr;
    worker.submitAndWait([&]{auto graph=std::make_unique<ProjectionClockGraph>(state,true);body=graph.get();
        original=std::make_unique<PlanningPreparedGPUObservation>(PlanningExecutionMeasurement::recordGPU(work(),worker,
            std::make_unique<ProjectionClockBuffer>(worker),std::move(graph)));original->prepare(projectionClockMemory());});
    std::thread foreign([&]{EXPECT_THROW(original->observe(),std::logic_error);
        EXPECT_THROW((void)PlanningPreparedGPUObservation(std::move(*original)),std::logic_error);});foreign.join();
    EXPECT_THROW(original->observe(),std::logic_error);
    worker.submitAndWait([&]{int alien;body->stream=&alien;
        EXPECT_THROW(original->observe(),std::logic_error);body->stream=state.stream;
        auto retained=std::move(*original);
        EXPECT_THROW(original->observe(),std::logic_error);
        EXPECT_DOUBLE_EQ(retained.observe().elapsedSeconds(),.018);
        EXPECT_THROW(retained.observe(),std::logic_error);original.reset();});
    EXPECT_EQ(state.graphs_live,0);EXPECT_EQ(state.body_launches,4);EXPECT_EQ(state.event_destroys,1);
}

/** @test Construction, instance, parent, readback and completion failures retire every original owner. */
TEST(PlanningExecutionMeasurement, PreparedGPUObservationRetiresOnPreparationAndTimingFailure)
{
    for(int failure=0;failure<6;++failure){SCOPED_TRACE(failure);
        ProjectionClockState state;ProjectionClockWorker worker(state);
        worker.submitAndWait([&]{
            if(failure==0)state.fail_create=2;
            if(failure==1)state.fail_instantiate=true;
            if(failure==0){EXPECT_THROW(PlanningExecutionMeasurement::recordGPU(work(),worker,
                std::make_unique<ProjectionClockBuffer>(worker),std::make_unique<ProjectionClockGraph>(state,true)),std::runtime_error);return;}
            auto token=PlanningExecutionMeasurement::recordGPU(work(),worker,std::make_unique<ProjectionClockBuffer>(worker),
                std::make_unique<ProjectionClockGraph>(state,true));
            if(failure==1){EXPECT_THROW(token.prepare(projectionClockMemory()),std::runtime_error);
                EXPECT_THROW(token.observe(),std::logic_error);return;}
            token.prepare(projectionClockMemory());
            EXPECT_EQ(state.body_launches,0);EXPECT_EQ(state.parent_launches,0);
            if(failure==2)state.fail_launch=1;
            if(failure==3)state.fail_readback=true;
            if(failure==4)state.fail_query=true;
            if(failure==5)state.ticks={100,100};
            EXPECT_THROW(token.observe(),std::exception);
            EXPECT_THROW(token.observe(),std::logic_error);
        });
        EXPECT_EQ(state.graphs_live,0);EXPECT_EQ(state.event_creates,state.event_destroys);
    }
}

/** @test Existing failure-atomic join drains started jobs, including a late completion after partial submission. */
TEST(PlanningExecutionMeasurement, PreparedGPUPreparationJoinsAllOnFailure)
{
    std::promise<void> failed,late;std::vector<std::future<void>> pending;
    pending.push_back(failed.get_future());pending.push_back(late.get_future());
    failed.set_exception(std::make_exception_ptr(std::runtime_error("preparation job failed")));
    std::exception_ptr failure;
    joinOriginalNativeWorkerSubmissions(pending,[&](std::exception_ptr error){if(!failure)failure=error;late.set_value();});
    for(auto &job:pending)EXPECT_FALSE(job.valid());ASSERT_TRUE(failure);
    // A second submission failure still requires retirement of the first real worker receipt.
    llaminar2::testing::MockWorkerGPUContext first(0),second(1);
    second.before_submit=[]{throw std::runtime_error("partial submission failed");};
    pending.clear();pending.reserve(2);failure=nullptr;int completed=0;
    try{pending.push_back(first.submitAsync([&]{++completed;}));pending.push_back(second.submitAsync([&]{++completed;}));}
    catch(...){failure=std::current_exception();}
    joinOriginalNativeWorkerSubmissions(pending,[&](std::exception_ptr error){if(!failure)failure=error;});
    ASSERT_EQ(pending.size(),1u);EXPECT_FALSE(pending[0].valid());EXPECT_EQ(completed,1);ASSERT_TRUE(failure);
}
