/**
 * @file Test__PlanningExecutionMeasurement.cpp
 * @brief Device-free failure and ordering proofs for startup measurements.
 *
 * Native-event doubles expose exact stream, device, warmup and retirement
 * ordering. They never allocate GPU storage or claim throughput. Real kernel
 * execution belongs to the corresponding integration preflight tests.
 */
#include "planning/PlanningExecutionMeasurement.h"
#include "backends/IGPUGraphCapture.h"
#include "../../mocks/MockBackend.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>

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
        bool published_intervals = false;
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
            EXPECT_TRUE(event == &storage[0] || event == &storage[1]);
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
            EXPECT_EQ(records, published_intervals ? 3 + 2 * elapsed_calls : 3);
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
    class SampleGraph final : public IGPUGraphCapture
    {
    public:
        int stream_storage = 0;
        void *stream = &stream_storage;
        bool executable = true;
        mutable int launches = 0;
        int fail_launch = -1;
        /** @brief Sampling cannot record a replacement graph. */
        bool beginCapture() override { ADD_FAILURE(); return false; }
        /** @brief Sampling cannot record a replacement graph. */
        bool endCapture() override { ADD_FAILURE(); return false; }
        /** @brief Construction must already be complete before observation. */
        bool instantiate() override { ADD_FAILURE(); return false; }
        /** @brief The sampler must explicitly name the graph-owned replay stream. */
        bool launch() override { ADD_FAILURE(); return false; }
        /** @brief Count complete retained submissions and inject warmup/timed failures. */
        bool launchOnStream(void *actual) const override
        {
            EXPECT_EQ(actual, stream);
            return ++launches != fail_launch;
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
        size_t nodeCount() const override { return 1; }
        /** @brief Sampling may not reset the owner's executable. */
        void reset() override { ADD_FAILURE(); }
        /** @return Diagnostic identity, never a performance score. */
        const char *backendName() const override { return "measurement-unit-double"; }
    };

    /** @return Explicit synthetic work declaration, not real throughput evidence. */
    PlanningMeasurementWork work()
    {
        return {PlanningWorkUnit::ArithmeticOperations, 1024, "unit ordering oracle"};
    }
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

TEST(PlanningExecutionMeasurement, PublishedHostInputsChangeOnlyBetweenRetiredInvocations)
{
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        EventBackend backend(type);
        SampleGraph graph;
        backend.expected_stream = graph.stream;
        backend.published_intervals = true;
        int publications = 0;
        const auto observation = PlanningExecutionMeasurement::gpuPublished(work(), backend, {type, 7}, graph, [&] {
            EXPECT_EQ(graph.launches, publications);
            EXPECT_EQ(backend.queries, publications ? publications + 1 : 0);
            ++publications;
        });
        EXPECT_EQ(publications, 4);
        EXPECT_EQ(graph.launches, 4);
        EXPECT_EQ(backend.records, 7);
        EXPECT_EQ(backend.elapsed_calls, 3);
        EXPECT_EQ(backend.destroys, 2);
        EXPECT_EQ(observation.completedWork(), 3 * 1024);
        EXPECT_DOUBLE_EQ(observation.elapsedSeconds(), .018);
    }
}

TEST(PlanningExecutionMeasurement, PublishedObservationPropagatesNativeAndPublicationFailuresWithoutRetry)
{
    for (auto failure : {Failure::FirstEvent, Failure::SecondEvent, Failure::Record, Failure::Query,
                         Failure::Elapsed, Failure::ZeroTime})
    {
        EventBackend backend(DeviceType::CUDA);
        SampleGraph graph;
        backend.expected_stream = graph.stream;
        backend.published_intervals = true;
        backend.failure = failure;
        EXPECT_THROW(PlanningExecutionMeasurement::gpuPublished(work(), backend, DeviceId::cuda(7), graph, [] {}), std::exception);
        EXPECT_LE(graph.launches, 2);
    }
    for (int fail_at : {1, 2, 3, 4})
    {
        EventBackend backend(DeviceType::ROCm);
        SampleGraph graph;
        backend.expected_stream = graph.stream;
        backend.published_intervals = true;
        int calls = 0;
        EXPECT_THROW(PlanningExecutionMeasurement::gpuPublished(work(), backend, DeviceId::rocm(7), graph, [&] {
            if (++calls == fail_at) throw std::runtime_error("publication failure");
        }), std::runtime_error);
        EXPECT_EQ(calls, fail_at);
        EXPECT_EQ(graph.launches, fail_at - 1);
        EXPECT_EQ(backend.destroys, 2);
    }
    EventBackend backend(DeviceType::CUDA);
    SampleGraph graph;
    EXPECT_THROW(PlanningExecutionMeasurement::gpuPublished(work(), backend, DeviceId::cuda(7), graph, {}), std::invalid_argument);
    EXPECT_EQ(backend.creates, 0);
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
