/**
 * @file Test__OrdinaryDeviceGeneration.cpp
 * @brief Captured CUDA/ROCm proof of the ordinary-generation publication ABI.
 *
 * This model-free regression executes the production backend entrypoints with
 * persistent buffers. It covers prefill versus continuation frontiers, EOS,
 * independent request rows, padded strides and twenty resets/replays of each
 * captured transaction. It proves the controller, not whole-model generation;
 * model parity must separately authenticate the complete generation parent.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "../../utils/GraphArenaTestHarness.h"
#include "kernels/common/SamplingMath.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::sampling_math;

/** @brief Backend-parameterized publication proof with no real model dependency. */
class OrdinaryDeviceGeneration : public ::testing::TestWithParam<std::string>
{
};

/**
 * @brief Persistent logical-state fixture matching the production arena's row layout.
 *
 * Three inactive request entries guard every field. Setup/readback are outside
 * capture; the graph keeps one immutable view across every reset and replay.
 * Declare this owner before graph objects so embedded addresses retire last.
 */
class OrdinaryFrontierFixture final
{
public:
    /** @brief Allocate four logical rows on the fixture's already-owned GPU. */
    OrdinaryFrontierFixture(IBackend &backend, void *stream, int requests)
        : backend_(backend), stream_(stream), requests_(requests), capacity_(requests + 3),
          host_(static_cast<size_t>(capacity_) * 4, -77),
          device_(static_cast<int32_t *>(backend.allocate(host_.size() * sizeof(int32_t), 0))) {}
    /** @brief Graph owners and terminal readbacks have retired before storage. */
    ~OrdinaryFrontierFixture() { if (device_) backend_.free(device_, 0); }
    OrdinaryFrontierFixture(const OrdinaryFrontierFixture &) = delete;
    OrdinaryFrontierFixture &operator=(const OrdinaryFrontierFixture &) = delete;
    /** @return Complete immutable publication view; no allocation or state observation. */
    OrdinaryGenerationFrontier view() const
    {
        if (!device_) return {};
        return {.cached_tokens = device_, .next_condition_tokens = device_ + capacity_,
            .stopped_flags = device_ + 2 * capacity_, .publication_ok_flags = device_ + 3 * capacity_,
            .request_capacity = capacity_, .context_capacity = 8192};
    }
    /** @brief Republish independent initial positions/conditions, retaining every address. */
    bool reset()
    {
        std::fill(host_.begin(), host_.end(), -77);
        for (int request = 0; request < requests_; ++request) {
            host_[request] = 7 + request;
            host_[capacity_ + request] = 19;
            host_[2 * capacity_ + request] = 0;
            host_[3 * capacity_ + request] = 1;
        }
        return device_ && backend_.hostToDevice(device_, host_.data(), host_.size() * sizeof(int32_t), 0, stream_);
    }
    /** @brief Materialize only the final test oracle, after captured publication. */
    bool observe()
    { return device_ && backend_.deviceToHost(host_.data(), device_, host_.size() * sizeof(int32_t), 0, stream_); }
    /** @brief Check both live outputs and all inactive row guards. */
    void expect(int request, int consumed, int next, int stopped, int ok) const
    {
        EXPECT_EQ(host_[request], 7 + request + consumed);
        EXPECT_EQ(host_[capacity_ + request], next);
        EXPECT_EQ(host_[2 * capacity_ + request], stopped);
        EXPECT_EQ(host_[3 * capacity_ + request], ok);
        for (int field = 0; field < 4; ++field)
            for (int guard = requests_; guard < capacity_; ++guard)
                EXPECT_EQ(host_[field * capacity_ + guard], -77);
    }
private:
    IBackend &backend_; ///< Borrowed backend outlives the exact context task.
    void *const stream_; ///< Explicit setup/readback stream, joined to capture by the caller.
    const int requests_, capacity_; ///< Active and physical row geometry.
    std::vector<int32_t> host_; ///< Diagnostic-only initial/terminal values.
    int32_t *const device_; ///< Fixture-owned storage, never allocated inside capture.
};

/**
 * @brief Test-only forward marker followed by the production ordinary publisher.
 *
 * The immutable descriptor borrows fixture-owned device storage. Going through
 * a declarative stage exercises the real retained-parent compiler, source-only
 * capture ownership, admission ordering and replay; it is not model parity.
 */
class OrdinaryPublicationProbeStage final : public IComputeStage
{
public:
    /** @brief Bind fixed production publication addresses and one observable forward marker. */
    OrdinaryPublicationProbeStage(IBackend &backend,
        OrdinaryGenerationPublication publication, int32_t *marker)
        : IComputeStage(DeviceId::cuda(0)), backend_(backend),
          publication_(publication), marker_(marker) {}

    /** @brief Record the marker and publication on the executor's exact stream. */
    bool execute(IDeviceContext *) override
    {
        void *const stream = requireGPUStream();
        return backend_.enqueuePrepareMTPVerifierPositionIds(
                   publication_.frontier.cached_tokens,
                   1, 1, 0, stream, marker_) &&
               backend_.enqueuePublishOrdinaryGenerationSample(publication_, 0, stream);
    }

    /** @return This model-free stage consumes the live model position before publication. */
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    /** @return This test's native conditional graph is a CUDA-owned program. */
    bool supportsBackend(ComputeBackendType backend) const override
    { return backend == ComputeBackendType::GPU_CUDA; }
protected:
    /** @return No tensor snapshot payload: this probe owns resident control rows. */
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }
private:
    IBackend &backend_; ///< Same production backend used by admission and terminal observation.
    const OrdinaryGenerationPublication publication_; ///< Immutable captured pointer/stride identity.
    int32_t *const marker_; ///< Fixture-owned address, released only after graph retirement.
};

TEST_P(OrdinaryDeviceGeneration, CapturedPublicationMatchesSerialLedgerAcrossReset)
{
    IBackend *const backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = GetParam() == "CUDA" ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    context.submitAndWait([&]
    {
        void *const stream = context.defaultStream();
        ASSERT_NE(stream, nullptr);
        // Keep graph owners inside this allocation scope. Even failed assertions
        // retire capture objects before their embedded buffer addresses vanish.
        const auto release = [backend](int32_t *p) { if (p) backend->free(p, 0); };
        const auto allocate = [backend, &release](size_t words)
        {
            return std::unique_ptr<int32_t, decltype(release)>(
                static_cast<int32_t *>(backend->allocate(words * sizeof(int32_t), 0)), release);
        };
        for (int requests : {1, 2, 31, 32, 33, 63, 64, 65, 129})
        for (int budget : {1, 2, 17})
        for (auto leading : {DeviceGenerationLeadingRowDisposition::PendingResponse,
                             DeviceGenerationLeadingRowDisposition::AlreadyEmitted})
        {
            SCOPED_TRACE(::testing::Message() << GetParam() << " requests=" << requests
                << " budget=" << budget << " leading=" << static_cast<int>(leading));
            const int response_stride = budget + 3;
            const int control_stride = kDeviceGenerationControlCount + 5;
            std::vector<int32_t> samples((budget + 1) * requests);
            std::vector<int32_t> stops(samples.size());
            std::vector<int32_t> response(requests * response_stride, -77);
            std::vector<int32_t> control(requests * control_stride, -77);
            for (int row = 0; row <= budget; ++row)
            for (int request = 0; request < requests; ++request)
            {
                const int index = row * requests + request;
                samples[index] = 1000 + index;
                // Alternate full-budget requests and independent early EOS.
                stops[index] = request % 2 && row == request % budget ? 1 : 0;
            }
            auto d_samples = allocate(samples.size());
            auto d_stops = allocate(stops.size());
            auto d_response = allocate(response.size());
            auto d_control = allocate(control.size());
            OrdinaryFrontierFixture frontier(*backend, stream, requests);
            ASSERT_TRUE(d_samples && d_stops && d_response && d_control);
            ASSERT_TRUE(frontier.reset());
            ASSERT_TRUE(backend->hostToDevice(d_samples.get(), samples.data(), samples.size() * 4, 0, stream));
            ASSERT_TRUE(backend->hostToDevice(d_stops.get(), stops.data(), stops.size() * 4, 0, stream));
            ASSERT_TRUE(backend->hostToDevice(d_response.get(), response.data(), response.size() * 4, 0, stream));
            ASSERT_TRUE(backend->hostToDevice(d_control.get(), control.data(), control.size() * 4, 0, stream));

            OrdinaryGenerationPublication publication{
                .request_count = requests, .sampled_tokens = d_samples.get(),
                .stopped_flags = d_stops.get(), .source = OrdinaryGenerationSampleSource::PrefillLogits,
                .response_tokens = d_response.get(), .response_token_stride = response_stride,
                .control = d_control.get(), .control_stride = control_stride,
                .frontier = frontier.view()};
            EXPECT_FALSE(backend->enqueuePublishOrdinaryGenerationSample(publication, 0, nullptr));
            auto malformed = publication;
            malformed.control_stride = kDeviceGenerationControlCount - 1;
            EXPECT_FALSE(backend->enqueuePublishOrdinaryGenerationSample(malformed, 0, stream));

            auto graph = context.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            ASSERT_TRUE(graph->beginCapture());
            ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(
                requests, budget, DeviceGenerationPolicy::ordinary(), leading,
                response_stride, d_response.get(), control_stride, d_control.get(), 0, stream));
            for (int row = 0; row <= budget; ++row)
            {
                publication.sampled_tokens = d_samples.get() + row * requests;
                publication.stopped_flags = d_stops.get() + row * requests;
                publication.source = row == 0 && leading == DeviceGenerationLeadingRowDisposition::PendingResponse
                    ? OrdinaryGenerationSampleSource::PrefillLogits : OrdinaryGenerationSampleSource::DecodeLogits;
                ASSERT_TRUE(backend->enqueuePublishOrdinaryGenerationSample(publication, 0, stream));
            }
            ASSERT_TRUE(graph->endCapture());
            ASSERT_TRUE(graph->instantiate());
            const DeviceGenerationAdmissionRequest admission{
                .request_count = requests, .max_new_tokens = budget,
                .depth_policy = DeviceGenerationPolicy::ordinary(),
                .initial_leading_row_disposition = leading};
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                SCOPED_TRACE(::testing::Message() << "repeat=" << repeat);
                ASSERT_TRUE(frontier.reset());
                ASSERT_TRUE(graph->launch());
                // Observe every replay, not just the final one: reset could erase
                // a transient bad ledger before a last-replay-only check sees it.
                ASSERT_TRUE(backend->deviceToHost(response.data(), d_response.get(), response.size() * 4, 0, stream));
                ASSERT_TRUE(backend->deviceToHost(control.data(), d_control.get(), control.size() * 4, 0, stream));
                ASSERT_TRUE(frontier.observe());
                for (int request = 0; request < requests; ++request)
                {
                    const int emitted = request % 2 ? request % budget + 1 : budget;
                    const int published = emitted -
                        (leading == DeviceGenerationLeadingRowDisposition::PendingResponse ? 1 : 0);
                    const int *const row = control.data() + request * control_stride;
                    EXPECT_EQ(validateDeviceGenerationTerminal(
                        std::span<const int, kDeviceGenerationControlCount>(row, kDeviceGenerationControlCount),
                        admission, response_stride, request, 0), DeviceGenerationTerminalError::None);
                    EXPECT_EQ(row[kDeviceGenerationControlOk], 1);
                    EXPECT_EQ(row[kDeviceGenerationControlRequestComplete], 1);
                    EXPECT_EQ(row[kDeviceGenerationControlResponseTokenCount], emitted);
                    EXPECT_EQ(row[kDeviceGenerationControlRemainingTokenCount], budget - emitted);
                    EXPECT_EQ(row[kDeviceGenerationControlTransactionCount], emitted);
                    EXPECT_EQ(row[kDeviceGenerationControlPublishedStateCommitCount], published);
                    EXPECT_EQ(row[kDeviceGenerationControlModelStopped], request % 2);
                    EXPECT_EQ(row[kDeviceGenerationControlAttemptedDraftTokenCount], 0);
                    frontier.expect(request, published,
                        1000 + (emitted - 1) * requests + request, request % 2, 1);
                    for (int i = kDeviceGenerationControlCount; i < control_stride; ++i)
                        EXPECT_EQ(row[i], -77);
                    for (int i = 0; i < response_stride; ++i)
                        EXPECT_EQ(response[request * response_stride + i],
                            i < emitted ? 1000 + i * requests + request : -77);
                }
            }
            if (requests == 33 && budget == 17 &&
                leading == DeviceGenerationLeadingRowDisposition::PendingResponse)
            {
                // Reuse the exact executable with poisoned producer bytes.
                // Independent rows must fail locally and retain their first
                // diagnostic through all later captured publication nodes.
                stops[0] = 2;
                samples[1] = -1;
                ASSERT_TRUE(backend->hostToDevice(d_stops.get(), stops.data(), stops.size() * 4, 0, stream));
                ASSERT_TRUE(backend->hostToDevice(d_samples.get(), samples.data(), samples.size() * 4, 0, stream));
                ASSERT_TRUE(frontier.reset());
                ASSERT_TRUE(graph->launch());
                ASSERT_TRUE(backend->deviceToHost(control.data(), d_control.get(), control.size() * 4, 0, stream));
                ASSERT_TRUE(frontier.observe());
                for (int request : {0, 1})
                {
                    const int *const row = control.data() + request * control_stride;
                    EXPECT_EQ(row[kDeviceGenerationControlOk], 0);
                    EXPECT_EQ(row[kDeviceGenerationControlRequestComplete], 1);
                    EXPECT_EQ(row[kDeviceGenerationControlResponseTokenCount], 0);
                    EXPECT_EQ(row[kDeviceGenerationControlPublishedStateCommitCount], 0);
                    EXPECT_EQ(row[kDeviceGenerationControlErrorCode],
                        static_cast<int>(DeviceGenerationError::InvalidOrdinarySample));
                    frontier.expect(request, 0, 19, 0, 0);
                }
                EXPECT_EQ(control[2 * control_stride + kDeviceGenerationControlOk], 1);
                EXPECT_EQ(control[2 * control_stride + kDeviceGenerationControlResponseTokenCount], budget);
            }
        }
    });
}

/** @test The real CUDA/HIP publication ignores sampler poison on forward-only calls. */
TEST_P(OrdinaryDeviceGeneration, CapturedForwardOnlyDoesNotReadOrEmitSamples)
{
    IBackend *const backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = GetParam() == "CUDA" ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    context.submitAndWait([&]
    {
        void *const stream = context.defaultStream();
        const auto release = [backend](int32_t *p) { if (p) backend->free(p, 0); };
        const auto allocate = [backend, &release](size_t words)
        {
            return std::unique_ptr<int32_t, decltype(release)>(
                static_cast<int32_t *>(backend->allocate(words * sizeof(int32_t), 0)), release);
        };
        for (int requests : {1, 2, 31, 32, 33, 63, 64, 65, 129})
        {
            constexpr int response_stride = 4;
            constexpr int control_stride = kDeviceGenerationControlCount + 3;
            auto control = allocate(requests * control_stride);
            auto response = allocate(requests * response_stride);
            auto poison = allocate(requests);
            OrdinaryFrontierFixture frontier(*backend, stream, requests);
            ASSERT_TRUE(control && response && poison);
            ASSERT_TRUE(frontier.reset());
            ASSERT_TRUE(backend->memset(control.get(), 0xff, requests * control_stride * 4, 0, stream));
            ASSERT_TRUE(backend->memset(response.get(), 0xff, requests * response_stride * 4, 0, stream));
            ASSERT_TRUE(backend->memset(poison.get(), 0xff, requests * 4, 0, stream));
            const DeviceGenerationAdmissionRequest admission{
                .request_count = requests, .max_new_tokens = 0,
                .depth_policy = DeviceGenerationPolicy::forwardOnly(),
                .initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::AlreadyEmitted};
            OrdinaryGenerationPublication publication{
                .request_count = requests, .sampled_tokens = poison.get(), .stopped_flags = poison.get(),
                .source = OrdinaryGenerationSampleSource::DecodeLogits,
                .response_tokens = response.get(), .response_token_stride = response_stride,
                .control = control.get(), .control_stride = control_stride,
                .frontier = frontier.view()};
            auto graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph && graph->beginCapture());
            ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(
                requests, 0, admission.depth_policy, admission.initial_leading_row_disposition,
                response_stride, response.get(), control_stride, control.get(), 0, stream));
            ASSERT_TRUE(backend->enqueuePublishOrdinaryGenerationSample(publication, 0, stream));
            ASSERT_TRUE(backend->enqueuePublishOrdinaryGenerationSample(publication, 0, stream));
            ASSERT_TRUE(graph->endCapture() && graph->instantiate());
            std::vector<int32_t> actual_control(requests * control_stride);
            std::vector<int32_t> actual_response(requests * response_stride);
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                ASSERT_TRUE(frontier.reset());
                ASSERT_TRUE(graph->launch());
                ASSERT_TRUE(backend->deviceToHost(actual_control.data(), control.get(), actual_control.size() * 4, 0, stream));
                ASSERT_TRUE(backend->deviceToHost(actual_response.data(), response.get(), actual_response.size() * 4, 0, stream));
                ASSERT_TRUE(frontier.observe());
                for (int request = 0; request < requests; ++request)
                {
                    const auto row = std::span<const int, kDeviceGenerationControlCount>(
                        actual_control.data() + request * control_stride, kDeviceGenerationControlCount);
                    EXPECT_EQ(validateDeviceGenerationTerminal(row, admission, response_stride, request, 0),
                        DeviceGenerationTerminalError::None);
                    frontier.expect(request, 1, -1, 0, 1);
                    for (int i = kDeviceGenerationControlCount; i < control_stride; ++i)
                        EXPECT_EQ(actual_control[request * control_stride + i], -1);
                }
                for (int token : actual_response) EXPECT_EQ(token, -1);
            }
        }
    });
}

/**
 * @test Captured producers may reuse the next-token row without a staging allocation.
 *
 * The copy kernel is deliberately a model-free sampler stand-in: its output
 * depends on the preceding publication's live cache position. Every subsequent
 * graph therefore consumes device-published state, not a preloaded sample list.
 * Whole-model sampling and arithmetic remain the parity campaign's obligation.
 */
TEST_P(OrdinaryDeviceGeneration, CapturedAliasedSamplerConsumesPublishedFrontier)
{
    IBackend *const backend = GetParam() == "CUDA" ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    auto &pool = GPUDeviceContextPool::instance();
    auto &context = GetParam() == "CUDA" ? pool.getNvidiaContext(0) : pool.getAMDContext(0);
    context.submitAndWait([&]
    {
        constexpr int budget = 17;
        constexpr int response_stride = budget + 3;
        void *const stream = context.defaultStream();
        ASSERT_NE(stream, nullptr);
        const auto release = [backend](int32_t *p) { if (p) backend->free(p, 0); };
        const auto allocate = [backend, &release](size_t words)
        {
            return std::unique_ptr<int32_t, decltype(release)>(
                static_cast<int32_t *>(backend->allocate(words * sizeof(int32_t), 0)), release);
        };
        for (int requests : {1, 33, 65})
        {
            SCOPED_TRACE(::testing::Message() << GetParam() << " requests=" << requests);
            auto response = allocate(requests * response_stride);
            auto control = allocate(requests * kDeviceGenerationControlCount);
            OrdinaryFrontierFixture frontier(*backend, stream, requests);
            ASSERT_TRUE(response && control && frontier.reset());
            const auto live = frontier.view();
            const OrdinaryGenerationPublication publication{
                .request_count = requests, .sampled_tokens = live.next_condition_tokens,
                .stopped_flags = live.stopped_flags,
                .source = OrdinaryGenerationSampleSource::DecodeLogits,
                .response_tokens = response.get(), .response_token_stride = response_stride,
                .control = control.get(), .control_stride = kDeviceGenerationControlCount,
                .frontier = live};
            auto graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph && graph->beginCapture());
            ASSERT_TRUE(backend->enqueuePrepareMTPVerifierPositionIds(
                live.cached_tokens, requests, 1, 0, stream, live.next_condition_tokens));
            ASSERT_TRUE(backend->enqueuePublishOrdinaryGenerationSample(publication, 0, stream));
            ASSERT_TRUE(graph->endCapture() && graph->instantiate());
            const DeviceGenerationAdmissionRequest admission{
                .request_count = requests, .max_new_tokens = budget,
                .depth_policy = DeviceGenerationPolicy::ordinary(),
                .initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::AlreadyEmitted};
            std::vector<int32_t> actual_response(requests * response_stride);
            std::vector<int32_t> actual_control(requests * kDeviceGenerationControlCount);
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                SCOPED_TRACE(::testing::Message() << "repeat=" << repeat);
                ASSERT_TRUE(frontier.reset());
                ASSERT_TRUE(backend->memset(response.get(), 0xff, actual_response.size() * 4, 0, stream));
                ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(
                    requests, budget, admission.depth_policy, admission.initial_leading_row_disposition,
                    response_stride, response.get(), kDeviceGenerationControlCount, control.get(), 0, stream));
                for (int row = 0; row < budget; ++row) ASSERT_TRUE(graph->launch());
                ASSERT_TRUE(frontier.observe());
                ASSERT_TRUE(backend->deviceToHost(actual_response.data(), response.get(), actual_response.size() * 4, 0, stream));
                ASSERT_TRUE(backend->deviceToHost(actual_control.data(), control.get(), actual_control.size() * 4, 0, stream));
                for (int request = 0; request < requests; ++request)
                {
                    frontier.expect(request, budget, 7 + request + budget - 1, 0, 1);
                    EXPECT_EQ(validateDeviceGenerationTerminal(
                        std::span<const int, kDeviceGenerationControlCount>(
                            actual_control.data() + request * kDeviceGenerationControlCount,
                            kDeviceGenerationControlCount),
                        admission, response_stride, request, 0), DeviceGenerationTerminalError::None);
                    for (int row = 0; row < response_stride; ++row)
                        EXPECT_EQ(actual_response[request * response_stride + row],
                            row < budget ? 7 + request + row : -1);
                }
            }
        }
    });
}

INSTANTIATE_TEST_SUITE_P(Backends, OrdinaryDeviceGeneration,
    ::testing::Values("CUDA", "ROCm"),
    [](const auto &info) { return info.param; });

/**
 * @test One native parent samples prefill once and admits only necessary forwards.
 *
 * The model-free body consumes the live cache position as an observable
 * forward marker before publishing a decode sample. Budget one and EOS in the
 * prologue must leave that marker untouched. All budgets and stop policies
 * reuse one executable; every replay is checked, not just the last reset.
 * Continued requests use the same graph and bypass prefill from the canonical
 * device-owned leading-row field, including when stale prefill samples are bad.
 * CUDA's WHILE policy is tested here; HIP uses its separate ticket policy.
 */
TEST(OrdinaryDeviceGeneration, NativeWhileSharesPrefillForwardAndGenerationCUDA)
{
    IBackend *const backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);
    auto &context = GPUDeviceContextPool::instance().getNvidiaContext(0);
    context.submitAndWait([&]
    {
        constexpr int capacity = 256;
        void *const stream = context.defaultStream();
        ASSERT_NE(stream, nullptr);
        const auto release = [backend](int32_t *p) { if (p) backend->free(p, 0); };
        const auto allocate = [backend, &release](size_t words)
        {
            return std::unique_ptr<int32_t, decltype(release)>(
                static_cast<int32_t *>(backend->allocate(words * sizeof(int32_t), 0)), release);
        };
        auto control = allocate(kDeviceGenerationControlCount);
        auto response = allocate(capacity);
        auto samples = allocate(2);
        auto stops = allocate(2);
        auto forward_marker = allocate(1);
        auto prefill_sample_marker = allocate(1);
        OrdinaryFrontierFixture frontier(*backend, stream, 1);
        ASSERT_TRUE(control && response && samples && stops && forward_marker && prefill_sample_marker);
        ASSERT_TRUE(frontier.reset());

        OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = samples.get(),
            .stopped_flags = stops.get(), .source = OrdinaryGenerationSampleSource::PrefillLogits,
            .response_tokens = response.get(), .response_token_stride = capacity,
            .control = control.get(), .control_stride = kDeviceGenerationControlCount,
            .frontier = frontier.view()};
        auto prologue = context.createGraphCapture(stream);
        ASSERT_TRUE(prologue);
        ASSERT_TRUE(prologue->beginCapture());
        ASSERT_TRUE(backend->enqueuePrepareMTPVerifierPositionIds(
            control.get() + kDeviceGenerationControlTransactionCount,
            1, 1, 0, stream, prefill_sample_marker.get()));
        ASSERT_TRUE(backend->enqueuePublishOrdinaryGenerationSample(publication, 0, stream));
        ASSERT_TRUE(prologue->endCapture());
        publication.sampled_tokens = samples.get() + 1;
        publication.stopped_flags = stops.get() + 1;
        publication.source = OrdinaryGenerationSampleSource::DecodeLogits;
        auto device_context = IDeviceContext::create(DeviceId::cuda(0), 1);
        ASSERT_TRUE(device_context);
        test::GraphArenaTestHarness arena;
        GraphExecutorConfig executor_config;
        executor_config.enable_validation = false;
        DeviceGraphExecutor executor(executor_config);
        arena.bindExecutor(executor);
        ComputeGraph graph;
        graph.addNode("ordinary_forward_and_publication",
            std::make_unique<OrdinaryPublicationProbeStage>(
                *backend, publication, forward_marker.get()), DeviceId::cuda(0));
        graph.setTerminalNode("ordinary_forward_and_publication");
        DeviceGraphExecutor::GraphSegmentCache cache;
        const DeviceControlledLoopPredicate predicate{
            .control_rows_device = control.get(),
            .control_stride = kDeviceGenerationControlCount, .request_count = 1,
            .healthy_index = kDeviceGenerationControlOk,
            .complete_index = kDeviceGenerationControlRequestComplete};
        int compositions = 0;
        const DeviceGraphExecutor::RetainedParentCompositionHook composer =
            [&](IGPUGraphCapture &parent, const ComputeGraph &,
                std::span<const DeviceGraphExecutor::GraphSegmentCache::
                    RetainedCaptureUnitTemplateView> units)
        {
            ++compositions;
            if (units.size() != 1 || !units.front().capture ||
                units.front().capture->hasExecutable())
                return false;
            const DeviceControlledLoopFragment body_fragments[] = {
                {.name = "sample terminal prefill logits", .capture = prologue.get(),
                 .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero,
                 .condition_word_device = reinterpret_cast<const uint32_t *>(
                     control.get() + kDeviceGenerationControlNextLeadingCommittedOutputCount)},
                {.name = "consume condition and sample decode logits",
                 .capture = units.front().capture,
                 .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero,
                 .condition_word_device = reinterpret_cast<const uint32_t *>(
                     control.get() + kDeviceGenerationControlRequestComplete)}};
            return parent.buildDeviceControlledWhileLoop(body_fragments, predicate);
        };
        const auto execute = [&](DeviceGraphExecutor::GraphInitialSubmissionPolicy submission)
        {
            graph.reset();
            return executor.executeWithCachedGraphReplay(
                graph, device_context.get(), cache, stream, &context,
                nullptr, false, false, true, {},
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireCloneableParentComposition,
                {}, {}, composer, submission);
        };
        ASSERT_TRUE(execute(DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch));
        ASSERT_TRUE(cache.retained_parent_capture);
        EXPECT_TRUE(cache.retained_parent_capture->hasExecutable());
        ASSERT_EQ(cache.segments.size(), 1u);
        ASSERT_TRUE(cache.segments.front().capture);
        // The production executor, not a test-only composer, owns the sole
        // executable. Source captures have definitions but no resident executable.
        EXPECT_FALSE(cache.segments.front().capture->hasExecutable());
        EXPECT_FALSE(prologue->hasExecutable());
        EXPECT_EQ(compositions, 1);

        for (auto leading : {DeviceGenerationLeadingRowDisposition::PendingResponse,
                             DeviceGenerationLeadingRowDisposition::AlreadyEmitted})
        for (int budget : {1, 2, 17, capacity})
        for (int stop_policy : {0, 1, 2, 3})
        for (int repeat = 0; repeat < 20; ++repeat)
        {
            SCOPED_TRACE(::testing::Message() << "budget=" << budget
                << " stop_policy=" << stop_policy << " repeat=" << repeat
                << " leading=" << static_cast<int>(leading));
            const int32_t host_samples[] = {stop_policy == 3 ? -1 : 19, 23};
            const int32_t host_stops[] = {stop_policy == 1, stop_policy == 2};
            const int32_t sentinel = -77;
            ASSERT_TRUE(backend->hostToDevice(samples.get(), host_samples, sizeof(host_samples), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(stops.get(), host_stops, sizeof(host_stops), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(forward_marker.get(), &sentinel, sizeof(sentinel), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(prefill_sample_marker.get(), &sentinel, sizeof(sentinel), 0, stream));
            ASSERT_TRUE(backend->memset(response.get(), 0xff, capacity * sizeof(int32_t), 0, stream));
            ASSERT_TRUE(frontier.reset());
            ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(
                1, budget, DeviceGenerationPolicy::ordinary(),
                leading,
                capacity, response.get(), kDeviceGenerationControlCount, control.get(), 0, stream));
            ASSERT_TRUE(cache.orderCaptureStreamAfter(&context, stream));
            ASSERT_TRUE(execute(DeviceGraphExecutor::GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch));
            // A terminal or failed request must not execute the body on replay.
            ASSERT_TRUE(execute(DeviceGraphExecutor::GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch));
            ASSERT_TRUE(cache.orderStreamAfterCapture(&context, stream));
            EXPECT_EQ(compositions, 1) << "Request reset must not recapture or recompose topology";
            std::array<int32_t, kDeviceGenerationControlCount> actual_control{};
            std::array<int32_t, capacity> actual_response{};
            int32_t actual_marker = 0;
            ASSERT_TRUE(backend->deviceToHost(actual_control.data(), control.get(), sizeof(actual_control), 0, stream));
            ASSERT_TRUE(backend->deviceToHost(actual_response.data(), response.get(), sizeof(actual_response), 0, stream));
            ASSERT_TRUE(backend->deviceToHost(&actual_marker, forward_marker.get(), sizeof(actual_marker), 0, stream));
            std::array<int, kDeviceGenerationControlCount> expected_control{};
            std::array<int32_t, capacity> expected_response;
            expected_response.fill(-1);
            ASSERT_TRUE(initialize_device_generation_control(
                budget, capacity, DeviceGenerationPolicy::ordinary(),
                expected_control.data(), leading));
            if (leading == DeviceGenerationLeadingRowDisposition::PendingResponse)
                (void)append_ordinary_sample_to_device_generation(
                    host_samples[0], host_stops[0], OrdinaryGenerationSampleSource::PrefillLogits,
                    expected_response.data(), capacity, expected_control.data());
            int expected_marker = sentinel;
            // This host oracle is test-only. The production graph makes each
            // admission decision and advances each row entirely on the device.
            for (int row = 0; row < budget &&
                 !expected_control[kDeviceGenerationControlRequestComplete]; ++row)
            {
                expected_marker = 7 + expected_control[kDeviceGenerationControlPublishedStateCommitCount];
                ASSERT_TRUE(append_ordinary_sample_to_device_generation(
                    host_samples[1], host_stops[1], OrdinaryGenerationSampleSource::DecodeLogits,
                    expected_response.data(), capacity, expected_control.data()));
            }
            EXPECT_EQ(actual_control, expected_control);
            EXPECT_EQ(actual_response, expected_response);
            EXPECT_EQ(actual_marker, expected_marker);
            ASSERT_TRUE(frontier.observe());
            const int emitted = expected_control[kDeviceGenerationControlResponseTokenCount];
            frontier.expect(0, expected_control[kDeviceGenerationControlPublishedStateCommitCount],
                emitted > 0 ? expected_response[emitted - 1] : 19,
                expected_control[kDeviceGenerationControlModelStopped],
                expected_control[kDeviceGenerationControlOk]);
            ASSERT_TRUE(backend->deviceToHost(&actual_marker, prefill_sample_marker.get(), sizeof(actual_marker), 0, stream));
            EXPECT_EQ(actual_marker, leading == DeviceGenerationLeadingRowDisposition::PendingResponse ? 0 : sentinel);
        }

        // The very same model executable must also serve forward-only prefix
        // continuation. Poison both sampling sources and prove exactly one body
        // execution, zero response writes and absorbing replay, without recapture.
        for (int repeat = 0; repeat < 20; ++repeat)
        {
            ASSERT_TRUE(frontier.reset());
            ASSERT_TRUE(backend->memset(samples.get(), 0xff, 2 * sizeof(int32_t), 0, stream));
            ASSERT_TRUE(backend->memset(stops.get(), 0xff, 2 * sizeof(int32_t), 0, stream));
            ASSERT_TRUE(backend->memset(response.get(), 0xff, capacity * sizeof(int32_t), 0, stream));
            ASSERT_TRUE(backend->memset(forward_marker.get(), 0xff, sizeof(int32_t), 0, stream));
            ASSERT_TRUE(backend->memset(prefill_sample_marker.get(), 0xff, sizeof(int32_t), 0, stream));
            const DeviceGenerationAdmissionRequest admission{
                .request_count = 1, .max_new_tokens = 0,
                .depth_policy = DeviceGenerationPolicy::forwardOnly(),
                .initial_leading_row_disposition = DeviceGenerationLeadingRowDisposition::AlreadyEmitted};
            ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(
                1, 0, admission.depth_policy, admission.initial_leading_row_disposition,
                capacity, response.get(), kDeviceGenerationControlCount, control.get(), 0, stream));
            ASSERT_TRUE(cache.orderCaptureStreamAfter(&context, stream));
            ASSERT_TRUE(execute(DeviceGraphExecutor::GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch));
            ASSERT_TRUE(execute(DeviceGraphExecutor::GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch));
            ASSERT_TRUE(cache.orderStreamAfterCapture(&context, stream));
            std::array<int32_t, kDeviceGenerationControlCount> actual_control{};
            std::array<int32_t, capacity> actual_response{};
            int32_t marker = -1;
            ASSERT_TRUE(backend->deviceToHost(actual_control.data(), control.get(), sizeof(actual_control), 0, stream));
            ASSERT_TRUE(backend->deviceToHost(actual_response.data(), response.get(), sizeof(actual_response), 0, stream));
            ASSERT_TRUE(backend->deviceToHost(&marker, forward_marker.get(), sizeof(marker), 0, stream));
            EXPECT_EQ(validateDeviceGenerationTerminal(actual_control, admission, capacity, 0, 0),
                DeviceGenerationTerminalError::None);
            ASSERT_TRUE(frontier.observe());
            frontier.expect(0, 1, -1, 0, 1);
            EXPECT_EQ(marker, 7) << "A second forward would consume the already advanced cache position";
            ASSERT_TRUE(backend->deviceToHost(&marker, prefill_sample_marker.get(), sizeof(marker), 0, stream));
            EXPECT_EQ(marker, -1) << "Forward-only, including terminal replay, must not run the prefill sampler";
            for (int token : actual_response) EXPECT_EQ(token, -1);
            EXPECT_EQ(compositions, 1);
        }
    });
}
}
