/**
 * @file NativeTimelineImportProof.h
 * @brief Prove retained native imports compose with newly recorded local work.
 *
 * One unchanged captured D2D transfer is reused between two in-place recording
 * fragments and then embedded in another complete native parent. Destination
 * bytes prove ordering; destroying all source views proves independent parent
 * lifetime. The CUDA/ROCm preflight groups run the identical implementation.
 */
#pragma once
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
#include <memory>

namespace llaminar2::test
{
/** @brief Exercise mixed native composition on the caller's exact GPU worker.
 * @param context Owns all native captures and the explicit stream.
 * @param backend Matching GPU allocation/copy implementation.
 * Terminal D2H is test-only; no host action occurs between captured operations. */
inline void proveNativeTimelineImports(IWorkerGPUContext &context, IBackend &backend)
{
    const int ordinal = context.deviceOrdinal();
    void *const stream = context.defaultStream();
    ASSERT_NE(stream, nullptr);
    const auto release = [&](void *bytes) { if (bytes) backend.free(bytes, ordinal); };
    std::unique_ptr<void, decltype(release)> allocation(backend.allocate(64, ordinal), release);
    ASSERT_TRUE(allocation);
    auto *bytes = static_cast<unsigned char *>(allocation.get());
    auto transfer = context.createGraphCapture(stream);
    ASSERT_TRUE(transfer->beginCapture());
    // This infrastructure proof records the asynchronous copy primitive; the
    // compatibility copy wrapper waits for completion and cannot be captured.
    ASSERT_TRUE(backend.deviceCopyAsync(bytes + 32, bytes, 32, ordinal, stream));
    ASSERT_TRUE(transfer->endCapture());
    ASSERT_TRUE(transfer->instantiate());
    for (int policy = 0; policy < 3; ++policy)
    {
        auto parent = context.createGraphCapture(stream);
        auto first = parent->createOrderedTimelineFragment();
        auto last = parent->createOrderedTimelineFragment();
        ASSERT_NE(first, nullptr);
        ASSERT_NE(last, nullptr);
        // Record in reverse order; only the declared timeline determines execution.
        ASSERT_TRUE(last->beginCapture());
        ASSERT_TRUE(backend.memset(bytes, 0x55, 32, ordinal, stream));
        ASSERT_TRUE(last->endCapture());
        ASSERT_TRUE(first->beginCapture());
        ASSERT_TRUE(backend.memset(bytes, 0x20 + policy, 32, ordinal, stream));
        ASSERT_TRUE(first->endCapture());
        const std::array steps{
            GPUOrderedTimelineStep{.name = "local producer", .capture = first.get()},
            GPUOrderedTimelineStep{.name = "unchanged native transfer", .capture = transfer.get()},
            GPUOrderedTimelineStep{.name = "local bank reuse", .capture = last.get()}};
        ASSERT_TRUE(parent->buildOrderedTimelineTransaction(steps));
        ASSERT_FALSE(parent->hasExecutable());
        // A complete nonconditional owner is also a valid import into a later parent.
        auto outer = context.createGraphCapture(stream);
        const GPUOrderedTimelineStep complete{.name = "complete transaction", .capture = parent.get()};
        ASSERT_TRUE(outer->buildOrderedTimelineTransaction(std::span(&complete, 1)));
        ASSERT_TRUE(outer->instantiate());
        parent.reset();
        first.reset();
        last.reset();
        for (int replay = 0; replay < 20; ++replay)
        {
            ASSERT_TRUE(backend.memset(bytes, 0xee, 64, ordinal, stream));
            ASSERT_TRUE(outer->launch());
            std::array<unsigned char, 64> actual{};
            ASSERT_TRUE(backend.deviceToHost(actual.data(), bytes, actual.size(), ordinal, stream));
            for (size_t index = 0; index < actual.size(); ++index)
                EXPECT_EQ(actual[index], index < 32 ? 0x55 : 0x20 + policy);
        }
    }
}
}
