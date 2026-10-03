/**
 * @file NativeDependencyReductionProof.h
 * @brief Symmetric captured fork/join ordering proof for native DAG reduction.
 *
 * Two independent streams copy disjoint halves of a producer's publication.
 * The consumer must acquire both. Capture also retains the original stream's
 * redundant producer-to-final edge; the real owner removes it at instantiate.
 * Twenty exact
 * changing-input replays then exercise reset, alternate launch streams and a
 * retained parent after source capture retirement. D2H and terminal joins are
 * test observations only, never part of the captured execution.
 */
#pragma once
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    /** @brief Ensure native failure unwinds storage rather than continuing capture. */
    inline void requireNativeDependencyProof(bool success, const char *operation)
    {
        if (!success) throw std::runtime_error(operation);
    }

    /**
     * @brief Prove production reduction and exact retained replay on one GPU.
     * @param context Exact device worker; caller must execute on that worker.
     * @param backend Matching allocation and asynchronous copy implementation.
     * @param edge_count Lossless native edge-count query for the supplied owner.
     */
    template <typename EdgeCount>
    void proveNativeDependencyReduction(IWorkerGPUContext &context, IBackend &backend,
                                        EdgeCount edge_count)
    {
        const int ordinal = context.deviceOrdinal();
        constexpr std::size_t extent = 256 * 1024;
        constexpr std::size_t half = extent / 2;
        const auto release = [&](void *pointer) { if (pointer) backend.free(pointer, ordinal); };
        std::unique_ptr<void, decltype(release)> storage(backend.allocate(4 * extent, ordinal), release);
        ASSERT_TRUE(storage);
        auto *input = static_cast<unsigned char *>(storage.get());
        auto *producer = input + extent;
        auto *joined = producer + extent;
        auto *output = joined + extent;
        void *const stream = context.defaultStream();
        void *const left = context.getOrCreateAuxiliaryStream("dependency_reduction_left");
        void *const right = context.getOrCreateAuxiliaryStream("dependency_reduction_right");
        void *const alternate = context.getOrCreateAuxiliaryStream("dependency_reduction_replay");
        ASSERT_NE(stream, nullptr); ASSERT_NE(left, nullptr); ASSERT_NE(right, nullptr);
        ASSERT_NE(alternate, nullptr); ASSERT_NE(left, right);
        const auto destroy_event = [&](void *event) { context.destroyEvent(event); };
        std::unique_ptr<void, decltype(destroy_event)> fork(context.createEvent(), destroy_event);
        std::unique_ptr<void, decltype(destroy_event)> left_done(context.createEvent(), destroy_event);
        std::unique_ptr<void, decltype(destroy_event)> right_done(context.createEvent(), destroy_event);
        auto capture = context.createGraphCapture(stream);
        std::unique_ptr<IGPUGraphCapture> parent;
        /** @brief All captured readers retire before their test allocation. */
        struct Retire
        {
            IWorkerGPUContext &context;
            std::array<void *, 4> streams;
            /** @brief Test-only terminal ordering also covers assertion unwinding. */
            ~Retire()
            {
                for (auto *stream : streams) EXPECT_TRUE(context.synchronizeStreamChecked(stream));
            }
        } retire{context, {stream, left, right, alternate}};
        const auto copy = [&](void *to, const void *from, std::size_t bytes, void *on) {
            requireNativeDependencyProof(backend.deviceCopyAsync(to, from, bytes, ordinal, on), "dependency proof D2D");
        };
        ASSERT_TRUE(capture->beginCapture());
        copy(producer, input, extent, stream);
        context.recordEvent(fork.get(), stream);
        context.waitEvent(fork.get(), left);
        context.waitEvent(fork.get(), right);
        copy(joined, producer, half, left);
        copy(joined + half, producer + half, half, right);
        context.recordEvent(left_done.get(), left);
        context.recordEvent(right_done.get(), right);
        context.waitEvent(left_done.get(), stream);
        context.waitEvent(right_done.get(), stream);
        copy(output, joined, extent, stream);
        ASSERT_TRUE(capture->endCapture());
        ASSERT_EQ(capture->nodeCount(), 4u);
        // The original stream frontier survives the branch joins, so capture
        // produces five edges for a diamond needing only four. This reproduces
        // the real collective-stream excess without manufacturing graph nodes.
        ASSERT_EQ(edge_count(*capture), 5u);
        ASSERT_TRUE(capture->instantiate());
        EXPECT_EQ(edge_count(*capture), 4u);

        // The complete owner is also composable. Its source can retire while
        // the parent's copied graph retains the exact same byte dependencies.
        parent = context.createGraphCapture(stream);
        const GPUOrderedTimelineStep step{.name = "reduced fork join", .capture = capture.get()};
        ASSERT_TRUE(parent->buildOrderedTimelineTransaction(std::span(&step, 1)));
        ASSERT_TRUE(parent->instantiate());
        capture.reset();
        std::vector<unsigned char> expected(extent), actual(extent);
        for (int replay = 0; replay < 20; ++replay)
        {
            SCOPED_TRACE(replay);
            void *const submit = replay % 2 == 0 ? stream : alternate;
            for (std::size_t i = 0; i < extent; ++i)
                expected[i] = static_cast<unsigned char>((i * 31 + replay * 17) % 251);
            ASSERT_TRUE(backend.memset(producer, 0xff, 3 * extent, ordinal, submit));
            ASSERT_TRUE(backend.hostToDeviceOnStream(input, expected.data(), extent, ordinal, submit));
            ASSERT_TRUE(parent->launchOnStream(submit));
            ASSERT_TRUE(backend.deviceToHostOnStream(actual.data(), output, extent, ordinal, submit));
            ASSERT_TRUE(context.synchronizeStreamChecked(submit));
            ASSERT_EQ(actual, expected);
        }
    }
}
