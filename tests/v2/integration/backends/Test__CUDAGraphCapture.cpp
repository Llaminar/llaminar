/**
 * @file Test__CUDAGraphCapture.cpp
 * @brief Integration tests for CUDA Graph Capture/Replay on NVIDIA GPUs
 *
 * Tests the IGPUGraphCapture interface via the CUDAGraphCapture implementation:
 * - Factory creation via IWorkerGPUContext::createGraphCapture()
 * - Capture lifecycle: beginCapture → endCapture → instantiate → launch
 * - Reset/clear state
 * - Error paths (launch without instantiate, instantiate without capture)
 * - tryUpdate() behavior
 * - Ownership transfer via unique_ptr
 * - Independent controller progress while distinct retained graphs await peers
 *
 * Graph ownership lives on the device context worker. Concurrency regressions
 * additionally use explicit device-bound host submitters to exercise the same
 * HTTP/archive versus maintenance-worker ordering as production.
 */

#include <gtest/gtest.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "backends/BackendManager.h"
#include "MTPTerminalScratchCaptureProof.h"
#include "MTPMainForwardReadRetirementProof.h"
#include "GPUGraphMemoryContractProof.h"
#include "kernels/cuda/kvcache/CUDARingKVCache.h"

#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

using namespace llaminar2;

// ===========================================================================
// Test Fixture
// ===========================================================================

/** @brief Run graph lifecycle checks on the owning CUDA worker. */
class Test__CUDAGraphCapture : public ::testing::Test
{
protected:
    /** @brief Register CUDA and skip when the backend is unavailable. */
    void SetUp() override
    {
        ensureNvidiaFactoryRegistered();
        if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
            GTEST_SKIP() << "CUDA not available";
    }

    /** @return Device-zero worker that owns every graph in this fixture. */
    IWorkerGPUContext &ctx()
    {
        return GPUDeviceContextPool::instance().getNvidiaContext(0);
    }
};

/** @test Scratch invalidation preserves in-flight readers and accepted bytes. */
TEST_F(Test__CUDAGraphCapture, MTPCatchupScratchRetiresBeforeAcceptedPublication)
{
    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);
    ctx().submitAndWait([&] { test::proveMTPTerminalScratchCapture(ctx(), *backend); });
}

/** @test Every main-forward role protects a pending sidecar's terminal-hidden read. */
TEST_F(Test__CUDAGraphCapture, MTPMainForwardWaitsForSidecarReadRetirement)
{
    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);
    ctx().submitAndWait([&] { test::proveMTPMainForwardReadRetirement(ctx(), *backend, DeviceId::cuda(0)); });
}

/** @test Forced tokens retain the current forward's completion boundary. */
TEST_F(Test__CUDAGraphCapture, MTPForcedTokenWaitsForCurrentForward)
{
    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);
    ctx().submitAndWait([&] { test::proveMTPForcedTokenForwardBoundary(ctx(), *backend, DeviceId::cuda(0)); });
}

/** @test General control metadata must not receive a bounded helper charge. */
TEST_F(Test__CUDAGraphCapture, BoundedHelperRejectsEventNode)
{
    ctx().submitAndWait([&] {
        test::proveBoundedHelperRejectsEventGraph(ctx(), [](void *event, void *stream) {
            return cudaEventRecordWithFlags(static_cast<cudaEvent_t>(event),
                       static_cast<cudaStream_t>(stream), cudaEventRecordExternal) ==
                   cudaSuccess;
        });
    });
}

/**
 * @test Pending peer waits must not starve an independent captured controller.
 *
 * Two distinct retained transactions each expose five independent peer waits.
 * Native batch-memory wait nodes can consume CUDA's finite scheduling channels,
 * including when the graphs are queued on the same stream. A controller that
 * supplies the eventual peer work must still execute, as must unrelated DMA.
 * This tests both stream arrangements through the production capture helper.
 * The cleanup guard releases the artificial peer gate before draining any
 * stream, including on an assertion failure, so the regression cannot deadlock.
 */
TEST_F(Test__CUDAGraphCapture, PendingPeerGraphsAllowIndependentControllerProgress)
{
    ctx().submitAndWait([&] {
        for (const bool shared_submission_stream : {true, false})
        {
            SCOPED_TRACE(shared_submission_stream);
            std::array<cudaStream_t, 4> streams{};
            std::array<cudaEvent_t, 2> terminals{};
            std::array<std::unique_ptr<IGPUGraphCapture>, 3> captures;
            std::uint64_t *host_words = nullptr;
            std::uint64_t *mapped_words = nullptr;
            void *device_bytes = nullptr;
            auto cleanup = [&](void *) {
                if (host_words)
                    std::atomic_ref(host_words[0]).store(1u, std::memory_order_release);
                for (const auto stream : streams)
                    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                for (auto &capture : captures) capture.reset();
                for (const auto event : terminals)
                    if (event) EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
                if (device_bytes) EXPECT_EQ(cudaFree(device_bytes), cudaSuccess);
                if (host_words) EXPECT_EQ(cudaFreeHost(host_words), cudaSuccess);
                for (const auto stream : streams)
                    if (stream) EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
            };
            std::unique_ptr<void, decltype(cleanup)> guard(&streams, cleanup);
            for (auto &stream : streams)
                ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            for (auto &event : terminals)
                ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), cudaSuccess);
            ASSERT_EQ(cudaHostAlloc(&host_words, 4096u, cudaHostAllocMapped), cudaSuccess);
            host_words[0] = 0u;
            host_words[1] = 0u;
            ASSERT_EQ(cudaHostGetDevicePointer(&mapped_words, host_words, 0u), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&device_bytes, 4096u), cudaSuccess);

            for (std::size_t graph_index = 0u; graph_index < 2u; ++graph_index)
            {
                captures[graph_index] = ctx().createGraphCapture();
                auto &capture = *captures[graph_index];
                ASSERT_TRUE(capture.beginCapture());
                const auto stream = static_cast<cudaStream_t>(ctx().defaultStream());
                std::array<cudaGraphNode_t, 5> wait_frontiers{};
                for (auto &frontier : wait_frontiers)
                {
                    // Each peer wait is a root; the final frontier joins all peers.
                    ASSERT_EQ(cudaStreamUpdateCaptureDependencies(
                        stream, nullptr, nullptr, 0u, cudaStreamSetCaptureDependencies), cudaSuccess);
                    ASSERT_TRUE(appendCUDAActiveCaptureTimelineWait64(stream, mapped_words, 1u));
                    cudaStreamCaptureStatus status{};
                    const cudaGraphNode_t *dependencies = nullptr;
                    std::size_t count = 0u;
                    ASSERT_EQ(cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr,
                        &dependencies, nullptr, &count), cudaSuccess);
                    ASSERT_EQ(count, 1u);
                    frontier = dependencies[0];
                }
                ASSERT_EQ(cudaStreamUpdateCaptureDependencies(stream, wait_frontiers.data(),
                    nullptr, wait_frontiers.size(), cudaStreamSetCaptureDependencies), cudaSuccess);
                ASSERT_TRUE(capture.endCapture());
                ASSERT_TRUE(capture.instantiate());
            }
            captures[2] = ctx().createGraphCapture();
            const std::array<GPUOrderedTimelineStep, 1> publish{{{
                .name = "independent_controller_publication",
                .kind = GPUOrderedTimelineStepKind::PublishValue64,
                .signal = mapped_words + 1u,
                .value = 1u,
            }}};
            ASSERT_TRUE(captures[2]->buildOrderedTimelineTransaction(publish));
            ASSERT_TRUE(captures[2]->instantiate());
            ASSERT_TRUE(captures[0]->launchOnStream(streams[0]));
            ASSERT_TRUE(captures[1]->launchOnStream(streams[shared_submission_stream ? 0u : 1u]));
            // Let the peer waits become pending before submitting maintenance.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ASSERT_EQ(cudaMemcpyAsync(device_bytes, host_words, 4096u,
                cudaMemcpyHostToDevice, streams[2]), cudaSuccess);
            ASSERT_EQ(cudaEventRecord(terminals[0], streams[2]), cudaSuccess);
            ASSERT_TRUE(captures[2]->launchOnStream(streams[3]));
            ASSERT_EQ(cudaEventRecord(terminals[1], streams[3]), cudaSuccess);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            std::array<bool, 2> ready{};
            do
            {
                for (std::size_t index = 0u; index < terminals.size(); ++index)
                {
                    const auto result = cudaEventQuery(terminals[index]);
                    ASSERT_TRUE(result == cudaSuccess || result == cudaErrorNotReady)
                        << cudaGetErrorString(result);
                    ready[index] = result == cudaSuccess;
                }
                if (ready[0] && ready[1]) break;
                std::this_thread::yield();
            } while (std::chrono::steady_clock::now() < deadline);
            EXPECT_TRUE(ready[0]) << "Pending peer graphs starved independent DMA";
            EXPECT_TRUE(ready[1]) << "Pending peer graphs starved the captured controller";
            EXPECT_EQ(std::atomic_ref(host_words[1]).load(std::memory_order_acquire), 1u);
        }
    });
}

// ===========================================================================
// Factory Tests
// ===========================================================================

namespace
{
/**
 * @brief Prove archive and maintenance progress for one retained producer shape.
 *
 * The archive's producer is deliberately waiting on a retained controller graph.
 * A separate host submitter issues the real ring-KV gather and archive copies,
 * reproducing the HTTP-thread versus controller-worker split. The rescue word
 * exists only for failing-test cleanup; success requires the captured controller
 * to release the producer before rescue, despite native queue backpressure.
 * @param context Device-zero worker owning the retained graphs.
 * @param conditional Whether the blocked producer lives inside a native IF body.
 */
void proveArchiveControllerProgress(IWorkerGPUContext &context, bool conditional)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr int kRows = 64;
    constexpr int kColumns = 128;
    constexpr std::size_t kBytes = kRows * kColumns * sizeof(float);
    std::array<cudaStream_t, 4> streams{};
    cudaEvent_t producer_ready = nullptr;
    std::unique_ptr<IGPUGraphCapture> producer;
    std::unique_ptr<IGPUGraphCapture> controller;
    std::unique_ptr<CUDARingKVCacheFP32> cache;
    std::uint64_t *host_words = nullptr;
    std::uint64_t *device_words = nullptr;
    float *staging = nullptr;
    float *archive = nullptr;
    std::future<void> publication;
    auto cleanup = [&](void *) {
        // Unblock every accepted operation before retiring its referenced bytes.
        if (host_words)
            std::atomic_ref(host_words[0]).store(1u, std::memory_order_release);
        if (publication.valid())
            publication.wait();
        for (const auto stream : streams)
            if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        controller.reset();
        producer.reset();
        cache.reset();
        if (producer_ready) EXPECT_EQ(cudaEventDestroy(producer_ready), cudaSuccess);
        if (staging) EXPECT_EQ(cudaFree(staging), cudaSuccess);
        if (archive) EXPECT_EQ(cudaFreeHost(archive), cudaSuccess);
        if (host_words) EXPECT_EQ(cudaFreeHost(host_words), cudaSuccess);
        for (const auto stream : streams)
            if (stream) EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    };
    std::unique_ptr<void, decltype(cleanup)> guard(&streams, cleanup);
    for (auto &stream : streams)
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&producer_ready, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaHostAlloc(&host_words, 4096u, cudaHostAllocMapped), cudaSuccess);
    host_words[0] = host_words[1] = 0u;
    ASSERT_EQ(cudaHostGetDevicePointer(&device_words, host_words, 0u), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&staging, 2u * kBytes), cudaSuccess);
    ASSERT_EQ(cudaHostAlloc(&archive, 2u * kBytes, cudaHostAllocDefault), cudaSuccess);
    cache = std::make_unique<CUDARingKVCacheFP32>(1, 1, kRows, 1, kColumns, 0);
    ASSERT_EQ(cudaMemsetAsync(staging, 0, 2u * kBytes, streams[0]), cudaSuccess);
    ASSERT_TRUE(cache->append(0, 0, staging, staging + kRows * kColumns, kRows, streams[0]));
    IKVCache::KVCacheLogicalBlockDescriptor descriptor{0, 0, 0, kRows, streams[1]};
    descriptor.payload_domain = IKVCache::KVCacheLogicalBlockPayloadDomain::Device;
    // Warm the exact gather symbol before pressure; lazy module loading is not
    // the lifecycle under test and must not become an implicit device fence.
    ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);
    ASSERT_TRUE(cache->exportLogicalBlock(descriptor, staging, staging + kRows * kColumns));
    ASSERT_EQ(cudaStreamSynchronize(streams[1]), cudaSuccess);
    context.submitAndWait([&] {
        producer = context.createGraphCapture(streams[0]);
        ASSERT_TRUE(producer->beginCapture());
        if (conditional)
        {
            // The conditional handle belongs to the final parent. Record its
            // body directly, avoiding graph cloning or a test-only eager path.
            cudaStreamCaptureStatus status{};
            cudaGraph_t graph = nullptr;
            ASSERT_EQ(cudaStreamGetCaptureInfo(streams[0], &status, nullptr,
                &graph, nullptr, nullptr, nullptr), cudaSuccess);
            cudaGraphConditionalHandle condition{};
            ASSERT_EQ(cudaGraphConditionalHandleCreate(&condition, graph,
                1u, cudaGraphCondAssignDefault), cudaSuccess);
            cudaGraphNodeParams params{};
            params.type = cudaGraphNodeTypeConditional;
            params.conditional.handle = condition;
            params.conditional.type = cudaGraphCondTypeIf;
            params.conditional.size = 1u;
            cudaGraphNode_t branch = nullptr;
            ASSERT_EQ(cudaGraphAddNode(&branch, graph, nullptr, nullptr,
                0u, &params), cudaSuccess);
            ASSERT_EQ(cudaStreamBeginCaptureToGraph(streams[3],
                params.conditional.phGraph_out[0], nullptr, nullptr, 0u,
                cudaStreamCaptureModeRelaxed), cudaSuccess);
            ASSERT_TRUE(appendCUDAActiveCaptureTimelineWait64(
                streams[3], device_words, 1u));
            cudaGraph_t body = nullptr;
            ASSERT_EQ(cudaStreamEndCapture(streams[3], &body), cudaSuccess);
            ASSERT_EQ(body, params.conditional.phGraph_out[0]);
            ASSERT_EQ(cudaStreamUpdateCaptureDependencies(streams[0],
                &branch, nullptr, 1u, cudaStreamSetCaptureDependencies), cudaSuccess);
        }
        else
        {
            ASSERT_TRUE(appendCUDAActiveCaptureTimelineWait64(
                streams[0], device_words, 1u));
        }
        ASSERT_TRUE(producer->endCapture());
        ASSERT_TRUE(producer->instantiate());
        controller = context.createGraphCapture(streams[2]);
        const std::array<GPUOrderedTimelineStep, 2> publish{{
            {.name = "release_archive_producer", .kind = GPUOrderedTimelineStepKind::PublishValue64,
             .signal = device_words, .value = 1u},
            {.name = "controller_completed", .kind = GPUOrderedTimelineStepKind::PublishValue64,
             .signal = device_words + 1u, .value = 1u},
        }};
        ASSERT_TRUE(controller->buildOrderedTimelineTransaction(publish));
        ASSERT_TRUE(controller->instantiate());
    });
    ASSERT_TRUE(producer && producer->hasExecutable());
    ASSERT_TRUE(controller && controller->hasExecutable());
    ASSERT_TRUE(producer->launchOnStream(streams[0]));
    // Prefix export observes the producer on a different stream through its
    // exact event; it does not submit archive kernels onto the producer stream.
    ASSERT_EQ(cudaEventRecord(producer_ready, streams[0]), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(streams[1], producer_ready, 0u), cudaSuccess);
    std::atomic<bool> rescued{false};
    // This bounded test-only rescue cannot satisfy the controller receipt.
    std::jthread rescue([&](std::stop_token stop) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!stop.stop_requested())
        {
            rescued.store(true, std::memory_order_release);
            std::atomic_ref(host_words[0]).store(1u, std::memory_order_release);
        }
    });
    publication = context.submitAsync([&] {
        // Admit archive work first, then prove another worker can submit the
        // finite graph that releases it while that archive queue is pressured.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ASSERT_TRUE(controller->launchOnStream(streams[2]));
    });
    for (int operation = 0; operation < 2048; ++operation)
    {
        ASSERT_TRUE(cache->exportLogicalBlock(descriptor, staging, staging + kRows * kColumns));
        ASSERT_EQ(cudaMemcpyAsync(archive, staging, 2u * kBytes,
                      cudaMemcpyDeviceToHost, streams[1]), cudaSuccess);
    }
    publication.get();
    ASSERT_EQ(cudaStreamSynchronize(streams[1]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[2]), cudaSuccess);
    rescue.request_stop();
    rescue.join();
    EXPECT_FALSE(rescued.load(std::memory_order_acquire));
    EXPECT_EQ(std::atomic_ref(host_words[1]).load(std::memory_order_acquire), 1u);
    EXPECT_TRUE(std::all_of(archive, archive + 2u * kRows * kColumns,
        [](float value) { return value == 0.0f; }));
}
} // namespace

/** @test Flat captured peer waits retain independent archive/controller progress. */
TEST_F(Test__CUDAGraphCapture, PrefixArchiveQueuePressureAllowsControllerProgress)
{
    proveArchiveControllerProgress(ctx(), false);
}

/** @test Native conditional parents retain independent archive/controller progress. */
TEST_F(Test__CUDAGraphCapture, ConditionalPrefixArchiveAllowsControllerProgress)
{
    proveArchiveControllerProgress(ctx(), true);
}

TEST_F(Test__CUDAGraphCapture, BackendNameIsCUDA)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_STREQ(capture->backendName(), "CUDA");
    });
}

TEST_F(Test__CUDAGraphCapture, CreateReturnsNonNull)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
    });
}

TEST_F(Test__CUDAGraphCapture, CreateViaWorkerThread)
{
    std::unique_ptr<IGPUGraphCapture> capture_out;
    ctx().submitAndWait([&] {
        capture_out = ctx().createGraphCapture();
        ASSERT_NE(capture_out, nullptr);
        EXPECT_STREQ(capture_out->backendName(), "CUDA");
    });
    EXPECT_NE(capture_out, nullptr);
}

// ===========================================================================
// Initial State Tests
// ===========================================================================

TEST_F(Test__CUDAGraphCapture, InitialState_NoExecutable)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_FALSE(capture->hasExecutable()) << "Fresh capture should have no executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "Fresh capture should have 0 nodes";
    });
}

// ===========================================================================
// Lifecycle Tests
// ===========================================================================

TEST_F(Test__CUDAGraphCapture, EmptyCaptureInstantiate)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        EXPECT_TRUE(capture->beginCapture());
        EXPECT_TRUE(capture->endCapture());
        EXPECT_TRUE(capture->instantiate());
        EXPECT_TRUE(capture->hasExecutable());
        EXPECT_TRUE(capture->launch());
    });
}

TEST_F(Test__CUDAGraphCapture, ResetClearsState)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        capture->reset();
        EXPECT_FALSE(capture->hasExecutable()) << "reset() should clear executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "reset() should clear node count";
    });
}

TEST_F(Test__CUDAGraphCapture, DoubleResetIsSafe)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());

        capture->reset();
        capture->reset();

        EXPECT_FALSE(capture->hasExecutable());
        EXPECT_EQ(capture->nodeCount(), 0u);
    });
}

/**
 * @test Ordered-timeline event instrumentation measures the retained child.
 *
 * The terminal stream synchronization is test-only observation. Production
 * collection uses the same event query through the non-blocking snapshot API.
 */
TEST_F(Test__CUDAGraphCapture, OrderedTimelinePerStepTimingIsConsumable)
{
    ctx().submitAndWait([&] {
        auto *const stream =
            static_cast<cudaStream_t>(ctx().defaultStream());
        ASSERT_NE(stream, nullptr);

        void *device_bytes = nullptr;
        ASSERT_EQ(cudaMalloc(&device_bytes, 4096u), cudaSuccess);

        auto child = ctx().createGraphCapture();
        ASSERT_NE(child, nullptr);
        ASSERT_TRUE(child->beginCapture());
        ASSERT_EQ(
            cudaMemsetAsync(device_bytes, 0x5a, 4096u, stream),
            cudaSuccess);
        ASSERT_TRUE(child->endCapture());
        ASSERT_GT(child->nodeCount(), 0u);

        auto parent = ctx().createGraphCapture();
        ASSERT_NE(parent, nullptr);
        const std::array<GPUOrderedTimelineStep, 1> steps{{{
            .name = "timed_memset",
            .kind = GPUOrderedTimelineStepKind::CapturedFragment,
            .capture = child.get(),
        }}};
        ASSERT_TRUE(parent->buildOrderedTimelineTransaction(
            steps,
            GPUOrderedTimelineInstrumentation::PerStepEvents));
        EXPECT_EQ(parent->nodeCount(), 3u);
        EXPECT_EQ(
            parent->consumeOrderedTimelineTiming().state,
            GPUOrderedTimelineTimingState::AwaitingLaunch);

        ASSERT_TRUE(parent->instantiate());
        ASSERT_TRUE(parent->launch());
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const auto snapshot = parent->consumeOrderedTimelineTiming();
        ASSERT_EQ(snapshot.state, GPUOrderedTimelineTimingState::Complete);
        ASSERT_EQ(snapshot.samples.size(), 1u);
        EXPECT_EQ(snapshot.samples.front().name, "timed_memset");
        EXPECT_EQ(
            snapshot.samples.front().kind,
            GPUOrderedTimelineStepKind::CapturedFragment);
        EXPECT_GE(snapshot.samples.front().elapsed_ms, 0.0);
        EXPECT_EQ(
            parent->consumeOrderedTimelineTiming().state,
            GPUOrderedTimelineTimingState::AwaitingLaunch);

        EXPECT_EQ(cudaFree(device_bytes), cudaSuccess);
    });
}

// ===========================================================================
// Error Path Tests
// ===========================================================================

/**
 * @test Conditional fragments retain their native handles in the final owner.
 *
 * Record in the opposite order to execution and reuse the executable with
 * poisoned output. Both conditional outcomes, timeline edges, optional timing,
 * final-owner parallel attachment and fragment-before-parent retirement must
 * preserve the same result. In particular, decoration must not clone a graph
 * that owns native conditional handles. No model
 * loading is needed to reproduce the previously rejected prefill graph shape.
 */
TEST_F(Test__CUDAGraphCapture, OrderedTimelineOwnsConditionalFragments)
{
    ctx().submitAndWait([&] {
        auto stream = static_cast<cudaStream_t>(ctx().defaultStream());
        unsigned char *bytes = nullptr;
        ASSERT_EQ(cudaMalloc(&bytes, 64u), cudaSuccess);
        std::unique_ptr<unsigned char, decltype(&cudaFree)> storage(bytes, cudaFree);
        for (const unsigned condition : {0u, 1u})
        for (const auto instrumentation : {
                 GPUOrderedTimelineInstrumentation::Disabled,
                 GPUOrderedTimelineInstrumentation::PerStepEvents})
        for (const bool auxiliary : {false, true})
        {
            auto parent = ctx().createGraphCapture();
            auto final = parent->createOrderedTimelineFragment();
            auto first = parent->createOrderedTimelineFragment();
            ASSERT_NE(first, nullptr);
            ASSERT_NE(final, nullptr);
            const auto record = [&](IGPUGraphCapture &fragment, unsigned value) {
                ASSERT_TRUE(fragment.beginCapture());
                ASSERT_EQ(cudaMemsetAsync(bytes, 0x7f, 32u, stream), cudaSuccess);
                cudaStreamCaptureStatus status{};
                cudaGraph_t graph = nullptr;
                const cudaGraphNode_t *dependencies = nullptr;
                size_t count = 0u;
                ASSERT_EQ(cudaStreamGetCaptureInfo(stream, &status, nullptr,
                    &graph, &dependencies, nullptr, &count), cudaSuccess);
                cudaGraphConditionalHandle handle{};
                ASSERT_EQ(cudaGraphConditionalHandleCreate(&handle, graph,
                    condition, cudaGraphCondAssignDefault), cudaSuccess);
                cudaGraphNodeParams params{};
                params.type = cudaGraphNodeTypeConditional;
                params.conditional.handle = handle;
                params.conditional.type = cudaGraphCondTypeIf;
                params.conditional.size = 1u;
                cudaGraphNode_t branch = nullptr;
                ASSERT_EQ(cudaGraphAddNode(&branch, graph, dependencies,
                    nullptr, count, &params), cudaSuccess);
                cudaMemsetParams fill{};
                fill.dst = bytes;
                fill.value = value;
                fill.elementSize = 1u;
                fill.width = 32u;
                fill.height = 1u;
                cudaGraphNode_t body = nullptr;
                ASSERT_EQ(cudaGraphAddMemsetNode(&body,
                    params.conditional.phGraph_out[0], nullptr, 0u, &fill), cudaSuccess);
                ASSERT_EQ(cudaStreamUpdateCaptureDependencies(stream, &branch,
                    nullptr, 1u, cudaStreamSetCaptureDependencies), cudaSuccess);
                ASSERT_TRUE(fragment.endCapture());
                EXPECT_FALSE(fragment.instantiate());
                EXPECT_FALSE(fragment.beginCapture());
            };
            record(*final, 0x35u);
            record(*first, 0x12u);
            ASSERT_GT(first->nodeCount(), 0u);
            ASSERT_GT(final->nodeCount(), 0u);
            const std::array<GPUOrderedTimelineStep, 4> steps{{
                {.name = "first", .kind = GPUOrderedTimelineStepKind::CapturedFragment,
                 .capture = first.get()},
                {.name = "publish", .kind = GPUOrderedTimelineStepKind::PublishValue64,
                 .signal = bytes + 32u, .value = 1u},
                {.name = "wait", .kind = GPUOrderedTimelineStepKind::WaitValue64,
                 .signal = bytes + 32u, .value = 1u},
                {.name = "final", .kind = GPUOrderedTimelineStepKind::CapturedFragment,
                 .capture = final.get()},
            }};
            ASSERT_TRUE(parent->buildOrderedTimelineTransaction(steps, instrumentation));
            EXPECT_FALSE(parent->supportsExecutableUpdate());
            EXPECT_EQ(parent->tryUpdate(), GraphUpdateResult::Failed);
            EXPECT_EQ(parent->createOrderedTimelineFragment(), nullptr);
            EXPECT_FALSE(parent->buildOrderedTimelineTransaction(steps, instrumentation));
            std::array<std::unique_ptr<IGPUGraphCapture>, 3> parallel_fragments;
            if (auxiliary)
            {
                // Only these tiny sources may be cloned. The conditional
                // handles above must retain this exact final graph owner.
                for (std::size_t i = 0u; i < parallel_fragments.size(); ++i)
                {
                    parallel_fragments[i] = ctx().createGraphCapture(stream);
                    ASSERT_NE(parallel_fragments[i], nullptr);
                    ASSERT_TRUE(parallel_fragments[i]->beginCapture());
                    ASSERT_EQ(cudaMemsetAsync(bytes + 40u + i, 0x25, 1u, stream), cudaSuccess);
                    ASSERT_TRUE(parallel_fragments[i]->endCapture());
                }
                ASSERT_TRUE(parent->appendParallelBranch({
                    *parallel_fragments[0], *parallel_fragments[1], *parallel_fragments[2]}));
            }
            ASSERT_TRUE(parent->instantiate());
            EXPECT_EQ(parent->tryUpdate(), GraphUpdateResult::Failed);
            first.reset();
            final.reset();
            for (auto &fragment : parallel_fragments) fragment.reset();
            for (int replay = 0; replay < 5; ++replay)
            {
                ASSERT_EQ(cudaMemsetAsync(bytes, 0xee, 64u, stream), cudaSuccess);
                ASSERT_TRUE(parent->launch());
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                std::array<unsigned char, 32> actual{};
                ASSERT_EQ(cudaMemcpy(actual.data(), bytes, actual.size(),
                    cudaMemcpyDeviceToHost), cudaSuccess);
                for (const auto byte : actual)
                    EXPECT_EQ(byte, condition ? 0x35u : 0x7fu);
                std::uint64_t published = 0u;
                ASSERT_EQ(cudaMemcpy(&published, bytes + 32u, sizeof(published),
                    cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_EQ(published, 1u);
                if (auxiliary)
                {
                    std::array<unsigned char, 3> markers{};
                    ASSERT_EQ(cudaMemcpy(markers.data(), bytes + 40u, markers.size(),
                        cudaMemcpyDeviceToHost), cudaSuccess);
                    for (const auto marker : markers) EXPECT_EQ(marker, 0x25u);
                }
                if (instrumentation == GPUOrderedTimelineInstrumentation::PerStepEvents)
                {
                    const auto timing = parent->consumeOrderedTimelineTiming();
                    EXPECT_EQ(timing.state, GPUOrderedTimelineTimingState::Complete);
                    EXPECT_EQ(timing.samples.size(), 4u);
                }
            }
        }
    });
}

/** @test Incomplete, duplicate, foreign, and revoked views cannot become executable. */
TEST_F(Test__CUDAGraphCapture, OrderedTimelineRejectsInvalidFragmentOwnership)
{
    ctx().submitAndWait([&] {
        auto parent = ctx().createGraphCapture();
        auto fragment = parent->createOrderedTimelineFragment();
        ASSERT_NE(fragment, nullptr);
        EXPECT_FALSE(parent->instantiate());
        EXPECT_FALSE(parent->beginCapture());
        ASSERT_TRUE(fragment->beginCapture());
        ASSERT_TRUE(fragment->endCapture());
        const GPUOrderedTimelineStep step{
            .name = "fragment", .kind = GPUOrderedTimelineStepKind::CapturedFragment,
            .capture = fragment.get()};
        const std::array<GPUOrderedTimelineStep, 2> duplicate{step, step};
        EXPECT_FALSE(parent->buildOrderedTimelineTransaction(duplicate));
        auto missing = parent->createOrderedTimelineFragment();
        EXPECT_FALSE(parent->buildOrderedTimelineTransaction(std::span(&step, 1u)));
        auto foreign_parent = ctx().createGraphCapture();
        auto foreign = foreign_parent->createOrderedTimelineFragment();
        EXPECT_FALSE(foreign_parent->buildOrderedTimelineTransaction(std::span(&step, 1u)));
        parent.reset();
        EXPECT_FALSE(fragment->instantiate());
        EXPECT_FALSE(missing->beginCapture());
    });
}

TEST_F(Test__CUDAGraphCapture, LaunchWithoutInstantiateFails)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_FALSE(capture->launch()) << "launch() without instantiate should fail";
    });
}

TEST_F(Test__CUDAGraphCapture, InstantiateWithoutCaptureFails)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_FALSE(capture->instantiate()) << "instantiate() without capture should fail";
    });
}

TEST_F(Test__CUDAGraphCapture, TryUpdateWithoutExecutableFails)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        auto result = capture->tryUpdate();
        EXPECT_EQ(result, GraphUpdateResult::Failed)
            << "tryUpdate() without executable should return Failed";
    });
}

// ===========================================================================
// Ownership Transfer Tests
// ===========================================================================

TEST_F(Test__CUDAGraphCapture, OwnershipTransferViaMove)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        std::unique_ptr<IGPUGraphCapture> dest = std::move(capture);
        EXPECT_EQ(capture, nullptr);
        ASSERT_NE(dest, nullptr);

        EXPECT_TRUE(dest->hasExecutable())
            << "Moved-to unique_ptr should own the executable";
        EXPECT_TRUE(dest->launch());
        EXPECT_STREQ(dest->backendName(), "CUDA");
    });
}
