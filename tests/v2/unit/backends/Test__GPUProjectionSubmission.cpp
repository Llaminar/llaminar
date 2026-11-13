/**
 * @file Test__GPUProjectionSubmission.cpp
 * @brief Replay the production projection owner without a runtime or device.
 *
 * Inject failures at every native creation boundary and retain whole-capture
 * ownership across separated projections. Fake handles represent obligations,
 * not physical memory accounting or an alternative production execution path.
 */
#include "backends/GPUProjectionSubmission.h"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <future>
#include <set>
using namespace llaminar2;
namespace
{
    /** @brief Exact fake handle inventory with deterministic failure injection. */
    class Operations final : public GPUProjectionResourceOperations
    {
    public:
        int fail_at = -1, calls = 0, released = 0;
        std::set<void *> live;
        /** @brief Publish a unique obligation or fail before creating one. */
        void *create()
        {
            const int call = calls++;
            if (call == fail_at) return nullptr;
            auto *handle = reinterpret_cast<void *>(static_cast<uintptr_t>(call + 1));
            EXPECT_TRUE(live.insert(handle).second); return handle;
        }
        void *createProjectionStream() override { return create(); }
        void *createProjectionEvent() override { return create(); }
        /** @brief Require every destruction to match exactly one created handle. */
        bool release(void *handle) noexcept
        { ++released; return live.erase(handle) == 1; }
        bool destroyProjectionStream(void *p) noexcept override { return release(p); }
        bool destroyProjectionEvent(void *p) noexcept override { return release(p); }
    };
    void *root() { return reinterpret_cast<void *>(1000); }
}

/** @test Every partial construction rolls back and remains reusable, unpublished. */
TEST(GPUProjectionSubmission, EveryPartialConstructionBoundary)
{
    for (auto population : {GPUProjectionStreamCount::Four, GPUProjectionStreamCount::Eight})
    {
        const int count = static_cast<int>(population);
        SCOPED_TRACE(static_cast<int>(population));
        for (int boundary = 0; boundary < 2 * count + 1; ++boundary)
        {
            SCOPED_TRACE(boundary);
            Operations ops; GPUProjectionResources owner(nullptr, population);
            ops.fail_at = boundary;
            EXPECT_THROW(owner.prepare(ops), std::runtime_error);
            EXPECT_TRUE(ops.live.empty()); EXPECT_EQ(ops.released, boundary);
            EXPECT_THROW((void)owner.acquire(root(), GPUProjectionSubmissionKind::Eager), std::logic_error);
            ops.fail_at = -1; owner.prepare(ops);
            { auto lease = owner.acquire(root(), GPUProjectionSubmissionKind::Eager);
              EXPECT_EQ(lease.resources().count, count); EXPECT_NE(lease.resources().quant_ready, nullptr);
              EXPECT_EQ(ops.live.size(), static_cast<size_t>(2 * count + 1));
              for (int slot = count; slot < GPUProjectionResourceView::MAX_STREAMS; ++slot)
              { EXPECT_EQ(lease.resources().streams[slot], nullptr); EXPECT_EQ(lease.resources().completion[slot], nullptr); } }
            owner.retire(ops); EXPECT_TRUE(ops.live.empty());
        }
    }
}

/** @test A whole capture retains exclusivity between separated projections. */
TEST(GPUProjectionSubmission, WholeCaptureAndExactRootOwnership)
{
    for (auto population : {GPUProjectionStreamCount::Four, GPUProjectionStreamCount::Eight})
    {
        const int count = static_cast<int>(population);
        SCOPED_TRACE(static_cast<int>(population));
        Operations ops; GPUProjectionResources owner(nullptr, population); owner.prepare(ops);
        {
            auto capture = owner.acquire(root(), GPUProjectionSubmissionKind::Capture);
            EXPECT_THROW(owner.retire(ops), std::logic_error);
            EXPECT_THROW(owner.beginRetirement(), std::logic_error);
            { auto projection = owner.acquire(root(), GPUProjectionSubmissionKind::Eager, true);
              EXPECT_NE(projection.resources().streams[count - 1], nullptr); }
            EXPECT_THROW(owner.assertRetirable(), std::logic_error);
            EXPECT_THROW((void)owner.acquire(reinterpret_cast<void *>(2000), GPUProjectionSubmissionKind::Eager, true), std::logic_error);
            EXPECT_THROW((void)owner.acquire(root(), GPUProjectionSubmissionKind::Capture), std::logic_error);
            { auto projection = owner.acquire(root(), GPUProjectionSubmissionKind::Eager, true);
              EXPECT_NE(projection.resources().completion[count - 1], nullptr); }
        }
        owner.assertRetirable(); owner.retire(ops);
    }
}

/** @test Standalone native capture and incomplete preparation cannot borrow handles. */
TEST(GPUProjectionSubmission, StandaloneCaptureAndNullRootRejected)
{
    for (auto population : {GPUProjectionStreamCount::Four, GPUProjectionStreamCount::Eight})
    {
        const int count = static_cast<int>(population);
        SCOPED_TRACE(static_cast<int>(population));
        Operations ops; GPUProjectionResources owner(nullptr, population);
        EXPECT_THROW((void)owner.acquire(nullptr, GPUProjectionSubmissionKind::Capture), std::invalid_argument);
        { auto capture = owner.acquire(root(), GPUProjectionSubmissionKind::Capture);
          EXPECT_THROW((void)capture.resources(), std::logic_error);
          EXPECT_THROW(owner.prepare(ops), std::logic_error); }
        owner.prepare(ops);
        EXPECT_THROW((void)owner.acquire(root(), GPUProjectionSubmissionKind::Eager, true), std::logic_error);
        const int created = ops.calls; owner.prepare(ops); EXPECT_EQ(ops.calls, created);
        owner.retire(ops); owner.prepare(ops); EXPECT_EQ(ops.calls, created + 2 * count + 1);
        owner.beginRetirement();
        EXPECT_THROW((void)owner.acquire(root(), GPUProjectionSubmissionKind::Eager), std::logic_error);
        EXPECT_THROW(owner.prepare(ops), std::logic_error);
        owner.retire(ops);
    }
}

/** @test Another submitter cannot enter between capture projection stages. */
TEST(GPUProjectionSubmission, CompetingSubmissionWaitsForNativeEnd)
{
    for (auto population : {GPUProjectionStreamCount::Four, GPUProjectionStreamCount::Eight})
    {
        SCOPED_TRACE(static_cast<int>(population));
        Operations ops; GPUProjectionResources owner(nullptr, population); owner.prepare(ops);
        std::promise<void> entered, release; auto released = release.get_future();
        auto capturing = std::async(std::launch::async, [&] {
            auto capture = owner.acquire(root(), GPUProjectionSubmissionKind::Capture);
            { auto stage = owner.acquire(root(), GPUProjectionSubmissionKind::Eager, true); }
            entered.set_value(); released.wait();
            { auto stage = owner.acquire(root(), GPUProjectionSubmissionKind::Eager, true); }
        });
        entered.get_future().wait();
        EXPECT_THROW(owner.assertRetirable(), std::logic_error);
        std::promise<void> attempted; std::atomic<bool> acquired{false};
        auto eager = std::async(std::launch::async, [&] {
            attempted.set_value(); auto lease = owner.acquire(root(), GPUProjectionSubmissionKind::Eager);
            acquired.store(true);
        });
        attempted.get_future().wait(); EXPECT_FALSE(acquired.load());
        release.set_value(); capturing.get(); eager.get(); EXPECT_TRUE(acquired.load());
        owner.retire(ops); EXPECT_TRUE(ops.live.empty());
    }
}

/** @test Native destroy failure cannot be reported as successful retirement. */
TEST(GPUProjectionSubmission, IncompleteRetirementIsFatal)
{
    for (auto population : {GPUProjectionStreamCount::Four, GPUProjectionStreamCount::Eight})
    {
        SCOPED_TRACE(static_cast<int>(population));
        class FailedDestroy final : public GPUProjectionResourceOperations
        {
            void *createProjectionStream() override { return reinterpret_cast<void *>(1); }
            void *createProjectionEvent() override { return reinterpret_cast<void *>(2); }
            bool destroyProjectionStream(void *) noexcept override { return false; }
            bool destroyProjectionEvent(void *) noexcept override { return false; }
        };
        EXPECT_DEATH({ FailedDestroy ops; GPUProjectionResources owner(nullptr, population); owner.prepare(ops); owner.retire(ops); }, "");
    }
}

/** @test Invalid populations cannot materialize or alias another backend's pool. */
TEST(GPUProjectionSubmission, InvalidPopulationRejected)
{
    for (int count : {0, 1, 3, 5, 7, 9, -1})
    {
        EXPECT_THROW((void)GPUProjectionResources(nullptr,
            static_cast<GPUProjectionStreamCount>(count)), std::invalid_argument);
    }
}

/** @test Named auxiliary maintenance preserves fixed handles and excludes capture begin. */
TEST(GPUProjectionSubmission, IdleOperationPreservesContextLifetimeResources)
{
    for (auto population : {GPUProjectionStreamCount::Four, GPUProjectionStreamCount::Eight})
    {
        Operations ops;
        GPUProjectionResources owner(nullptr, population);
        owner.prepare(ops);
        const auto expected = [&] {
            auto submission = owner.acquire(root(), GPUProjectionSubmissionKind::Eager);
            return submission.resources();
        }();
        const int created = ops.calls;
        int maintenance_calls = 0;
        owner.withIdleOperation([&] {
            ++maintenance_calls;
            EXPECT_THROW((void)owner.acquire(root(), GPUProjectionSubmissionKind::Capture), std::logic_error);
            EXPECT_THROW(owner.retire(ops), std::logic_error);
        });
        EXPECT_THROW(owner.withIdleOperation([&] { throw std::runtime_error("maintenance failure"); }), std::runtime_error);
        {
            auto submission = owner.acquire(root(), GPUProjectionSubmissionKind::Eager);
            EXPECT_EQ(submission.resources().count, expected.count);
            EXPECT_EQ(submission.resources().streams, expected.streams);
            EXPECT_EQ(submission.resources().completion, expected.completion);
            EXPECT_EQ(submission.resources().quant_ready, expected.quant_ready);
            EXPECT_THROW(owner.withIdleOperation([&] { ++maintenance_calls; }), std::logic_error);
        }
        {
            auto capture = owner.acquire(root(), GPUProjectionSubmissionKind::Capture);
            EXPECT_THROW(owner.withIdleOperation([&] { ++maintenance_calls; }), std::logic_error);
        }
        EXPECT_EQ(maintenance_calls, 1);
        EXPECT_EQ(ops.calls, created);
        EXPECT_EQ(ops.released, 0);
        owner.retire(ops);
        EXPECT_EQ(ops.released, created);
        EXPECT_TRUE(ops.live.empty());
    }
}
