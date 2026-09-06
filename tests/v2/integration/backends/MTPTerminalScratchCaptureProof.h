/**
 * @file MTPTerminalScratchCaptureProof.h
 * @brief Symmetric CUDA/HIP proof of MTP scratch retirement and mailbox reuse.
 *
 * Retiring a typed publication prevents new readers, but does not complete an
 * already-submitted GPU read. Two explicit streams and retained copy graphs
 * prove the distinct semantic and physical lifetimes. Accepted publication
 * waits on the sidecar's exact event, with no prefill lengths or host row repair.
 * This is a model-free protocol regression, not a model-parity substitute.
 */
#pragma once

#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

namespace llaminar2::test
{
    /**
     * @brief Replay initial-only and initial-plus-suffix catch-up on one device.
     * @param context Exact worker context; call on its owning worker thread.
     * @param backend Backend for the same device, used only for raw test storage.
     *
     * All buffers, streams, events and graphs are retained across twenty rounds.
     * Setup uploads and final test observation are the only host data boundaries.
     */
    inline void proveMTPTerminalScratchCapture(
        IWorkerGPUContext &context, IBackend &backend)
    {
        constexpr std::size_t row_bytes = 4096;
        constexpr std::size_t rows = 7;
        const int ordinal = context.deviceOrdinal();
        std::array<void *, 2> streams{};
        std::array<void *, 2> events{};
        std::array<std::unique_ptr<IGPUGraphCapture>, 6> graphs;
        auto *storage = static_cast<std::uint8_t *>(backend.allocate(rows * row_bytes, ordinal));
        ASSERT_NE(storage, nullptr);
        // Even failed assertions must drain readers before dropping graph-bound
        // storage. These waits are test teardown, never part of the transaction.
        auto cleanup = [&](void *) {
            for (auto stream : streams)
                if (stream) context.synchronizeStream(stream);
            for (auto &graph : graphs) graph.reset();
            for (auto event : events)
                if (event) context.destroyEvent(event);
            for (auto stream : streams)
                if (stream) context.destroyStream(stream);
            backend.free(storage, ordinal);
        };
        std::unique_ptr<void, decltype(cleanup)> lifetime(storage, cleanup);
        for (auto &stream : streams)
        {
            stream = context.createStream();
            ASSERT_NE(stream, nullptr);
        }
        for (auto &event : events)
        {
            event = context.createEvent();
            ASSERT_NE(event, nullptr);
        }
        std::array<std::uint8_t, rows * row_bytes> expected{};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t byte = 0; byte < row_bytes; ++byte)
                expected[row * row_bytes + byte] =
                    static_cast<std::uint8_t>(17 + row * 53 + byte % 127);
        ASSERT_TRUE(backend.hostToDevice(storage, expected.data(), expected.size(),
                                        ordinal, streams[0]));

        // Sources 0/1/2 are checkpoint/suffix/accepted; row 3 is the shared
        // mailbox; rows 4/5/6 are immutable observations made by its consumers.
        for (std::size_t phase = 0; phase < 3; ++phase)
        {
            for (std::size_t side = 0; side < 2; ++side)
            {
                auto &graph = graphs[phase * 2 + side];
                graph = context.createGraphCapture(streams[side]);
                ASSERT_NE(graph, nullptr);
                ASSERT_TRUE(graph->beginCapture());
                const auto src = side == 0 ? phase : 3;
                const auto dst = side == 0 ? 3 : 4 + phase;
                const bool copied = backend.deviceCopyAsync(
                    storage + dst * row_bytes, storage + src * row_bytes,
                    row_bytes, ordinal, streams[side]);
                const bool ended = graph->endCapture();
                ASSERT_TRUE(copied);
                ASSERT_TRUE(ended);
                ASSERT_TRUE(graph->instantiate());
            }
        }

        MTPTerminalHiddenPublication publication;
        for (int round = 0; round < 20; ++round)
        {
            SCOPED_TRACE(round);
            for (const bool has_suffix : {false, true})
            {
                SCOPED_TRACE(has_suffix);
                for (std::size_t phase = 0; phase < 3; ++phase)
                {
                    if (phase == 1 && !has_suffix) continue;
                    // A prior read must retire physically before its mailbox
                    // is overwritten, even after its semantic lease is invalid.
                    if (round > 0 || has_suffix || phase > 0)
                        ASSERT_TRUE(context.waitEventChecked(events[1], streams[0]));
                    ASSERT_TRUE(graphs[phase * 2]->launch());
                    ASSERT_TRUE(context.recordEventChecked(events[0], streams[0]));
                    if (phase == 0) publication.publishCheckpointRestore();
                    else if (phase == 1) publication.publishMainForward();
                    else publication.publishAcceptedVerifier();
                    const auto lease = publication.acquireReadLease();
                    ASSERT_TRUE(lease.has_value());
                    ASSERT_TRUE(context.waitEventChecked(events[0], streams[1]));
                    ASSERT_TRUE(graphs[phase * 2 + 1]->launch());
                    ASSERT_TRUE(context.recordEventChecked(events[1], streams[1]));
                    ASSERT_TRUE(publication.stillOwns(*lease));
                    if (phase != 2)
                    {
                        publication.invalidate();
                        EXPECT_FALSE(publication.stillOwns(*lease));
                        EXPECT_FALSE(publication.acquireReadLease().has_value());
                    }
                }
                EXPECT_EQ(publication.source(),
                          MTPTerminalHiddenPublication::Source::AcceptedVerifier);
                std::array<std::uint8_t, 3 * row_bytes> observed{};
                // Observation is outside the replay protocol and waits only
                // for its exact consumer stream's result.
                ASSERT_TRUE(backend.deviceToHost(observed.data(), storage + 4 * row_bytes,
                                                observed.size(), ordinal, streams[1]));
                for (std::size_t phase = 0; phase < 3; ++phase)
                {
                    if (phase == 1 && !has_suffix) continue;
                    EXPECT_TRUE(std::equal(
                        expected.begin() + phase * row_bytes,
                        expected.begin() + (phase + 1) * row_bytes,
                        observed.begin() + phase * row_bytes));
                }
            }
        }
    }
}
