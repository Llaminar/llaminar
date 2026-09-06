/**
 * @file MTPMainForwardReadRetirementProof.h
 * @brief Exercise the real main-forward prelude against a held sidecar reader.
 *
 * A narrow peer supplies a device and the production shifted-KV publication,
 * without loading model weights. Retained copy/scalar-write graphs stand in for
 * hidden producers/readers; prepareLiveStateForForwardGraphExecution owns
 * the ordering edge. A timeline gate makes the race deterministic and proves
 * that submission returns while the reader is still pending on another stream.
 */
#pragma once

#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "models/qwen/QwenStandardGraph.h"
#include "memory/BufferArena.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    /** @brief Inject only the device/publication needed by the real prelude. */
    struct DeviceGraphOrchestratorLiveStateTestAccess
    {
        /** @brief Bind one real backend and preallocate its production event. */
        static bool initialize(DeviceGraphOrchestrator &runner, DeviceId device)
        {
            runner.state_.device_id = device;
            return runner.initializeShiftedMTPKVReadyEvent();
        }

        /** @brief Publish the exact sidecar stream after its captured read. */
        static bool publishRead(DeviceGraphOrchestrator &runner, void *stream)
        {
            return runner.recordShiftedMTPKVReady(stream, "held_sidecar_read_test");
        }

        /** @brief Enter the installed prelude; never recreate its waits in the test. */
        static bool prepareMain(DeviceGraphOrchestrator &runner, void *stream,
                                DeviceId device, ForwardExecutionRole role)
        {
            ForwardInput input;
            input.seq_len = 1;
            input.batch_size = 1;
            input.execution_role = role;
            return runner.prepareLiveStateForForwardGraphExecution(input, stream, device);
        }

        /** @brief Prove a main writer did not consume the next sidecar's handoff. */
        static bool publicationRetained(const DeviceGraphOrchestrator &runner)
        {
            return runner.shifted_mtp_kv_ready_.valid;
        }

        /** @brief Bind one canonical arena slot and its preallocated producer event. */
        static bool initializeTargetSlot(DeviceGraphOrchestrator &runner,
                                         IBackend &backend, DeviceId device)
        {
            runner.state_.device_id = device;
            runner.arena_ = std::make_unique<BufferArena>();
            constexpr auto id = BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS;
            if (!runner.arena_->registerBuffer(id, 1, 1, "INT32", device) ||
                !runner.arena_->allocate() ||
                !runner.arena_->allocateDeviceStorage(id, device) ||
                !runner.initializeForwardGraphOutputReadyEvent())
                return false;
            runner.stochastic_target_row_capacity_ = 1;
            runner.stochastic_target_sample_tokens_dev_ = runner.arena_->getDevicePtr(id, device);
            runner.stochastic_target_sample_ready_.resize(1);
            auto *event = backend.createEvent(device.gpu_ordinal());
            if (!event) return false;
            runner.stochastic_target_sample_ready_[0].event = std::shared_ptr<void>(event,
                [&backend, device](void *value) { backend.destroyEvent(value, device.gpu_ordinal()); });
            return true;
        }

        /** @brief Publish a real preceding forward completion on the held stream. */
        static bool publishForward(DeviceGraphOrchestrator &runner, DeviceId device,
                                   void *stream, ForwardExecutionRole role)
        {
            ForwardOutput output;
            output.execution = {.valid = true, .device = device, .stream = stream,
                .execution_role = role, .is_decode = true,
                .all_position_logits = role == ForwardExecutionRole::GroupedMTPVerifier,
                .graph_seq_len = 1, .graph_batch_size = 1};
            return runner.publishForwardGraphOutputReady(output);
        }

        /** @brief Access the same stable slot consumed by the production sidecar. */
        static void *targetSlot(const DeviceGraphOrchestrator &runner)
        {
            return runner.stochastic_target_sample_tokens_dev_;
        }

        /** @brief Inspect the exact event published by the real forced-token method. */
        static void *targetReadyEvent(const DeviceGraphOrchestrator &runner)
        {
            return runner.stochastic_target_sample_ready_[0].event.get();
        }
    };

    namespace test
    {
        /**
         * @brief Forced publication must retire the preceding forward before reuse.
         * @param context Exact current worker, also owning the control-token stream.
         * @param backend Backend for this worker's physical device.
         * @param device Complete backend/ordinal identity.
         *
         * A captured predecessor borrows the current token while a host-owned test
         * gate holds it pending. Calling the real forced-token API must return
         * asynchronously, but its publication cannot complete before that reader.
         * No logits handoff is armed, exercising the durable event after one-shot
         * handoffs have already been consumed. Every output role and twenty slot
         * generations are covered without model weights or test-only runtime waits.
         */
        inline void proveMTPForcedTokenForwardBoundary(
            IWorkerGPUContext &context, IBackend &backend, DeviceId device)
        {
            using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
            const int ordinal = context.deviceOrdinal();
            TransferEngine transfers;
            const DeviceId devices[] = {device};
            auto control = transfers.allocateMappedHostRegion(4096, devices);
            auto observation = transfers.allocateDeviceTransferBuffer(sizeof(std::int32_t), device);
            ASSERT_TRUE(control && observation);
            auto *word = static_cast<std::uint64_t *>(control->mutableHostData());
            std::atomic_ref<std::uint64_t>(*word).store(0u, std::memory_order_release);
            void *stream = context.createStream();
            ASSERT_NE(stream, nullptr);
            std::unique_ptr<IGPUGraphCapture> graph;
            std::unique_ptr<DeviceGraphOrchestrator> runner;
            auto cleanup = [&](void *) {
                std::atomic_ref<std::uint64_t>(*word).store(1000u, std::memory_order_release);
                context.synchronizeStream(stream);
                context.synchronizeStream(context.defaultStream());
                graph.reset();
                runner.reset();
                context.destroyStream(stream);
            };
            std::unique_ptr<void, decltype(cleanup)> lifetime(word, cleanup);
            auto builder = std::make_shared<QwenStandardGraph>(GraphConfig{}, nullptr);
            runner = std::make_unique<DeviceGraphOrchestrator>(builder, nullptr);
            ASSERT_TRUE(Peer::initializeTargetSlot(*runner, backend, device));
            std::int32_t previous = 123;
            // First-use CUDA module loading may synchronize internally. Complete
            // that setup before holding a reader, as production graph preparation
            // does; the measured invariant is publication ordering, not JIT setup.
            ASSERT_TRUE(runner->stageStochasticTargetTokenForDeviceSampling(previous, 0));
            context.synchronizeEvent(Peer::targetReadyEvent(*runner));
            graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph && graph->beginCapture());
            const bool copied = backend.deviceCopyAsync(observation->mutableDeviceData(),
                Peer::targetSlot(*runner), sizeof(previous), ordinal, stream);
            const bool ended = graph->endCapture();
            ASSERT_TRUE(copied && ended && graph->instantiate());
            constexpr std::array roles{ForwardExecutionRole::MainInference,
                ForwardExecutionRole::MTPCondition, ForwardExecutionRole::GroupedMTPVerifier};
            for (std::uint64_t round = 1; round <= 20; ++round)
            {
                SCOPED_TRACE(round);
                ASSERT_TRUE(backend.streamWaitTimelineSignal64(
                    stream, control->deviceAlias(device), round, ordinal));
                ASSERT_TRUE(graph->launch());
                ASSERT_TRUE(Peer::publishForward(*runner, device, stream, roles[(round - 1) % roles.size()]));
                const auto next = static_cast<std::int32_t>(321 + round);
                ASSERT_TRUE(runner->stageStochasticTargetTokenForDeviceSampling(next, 0));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                bool complete = false;
                ASSERT_TRUE(context.queryEventChecked(Peer::targetReadyEvent(*runner), complete));
                EXPECT_FALSE(complete) << "forced token escaped the preceding forward boundary";
                std::atomic_ref<std::uint64_t>(*word).store(round, std::memory_order_release);
                context.synchronizeEvent(Peer::targetReadyEvent(*runner));
                std::int32_t observed = -1;
                ASSERT_TRUE(backend.deviceToHost(&observed, observation->deviceData(),
                    sizeof(observed), ordinal, stream));
                EXPECT_EQ(observed, previous);
                ASSERT_TRUE(backend.deviceToHost(&observed, Peer::targetSlot(*runner),
                    sizeof(observed), ordinal, stream));
                EXPECT_EQ(observed, next);
                previous = next;
            }
        }

        /**
         * @brief Hold a sidecar read while all main-graph roles attempt overwrite.
         * @param context Current worker owning the exact streams and graph handles.
         * @param backend Same-device backend used for test storage and timeline gate.
         * @param device Complete backend/ordinal identity, never inferred from rank.
         *
         * Storage is tested in native bytes, so the lifetime invariant is independent
         * of expert format or activation precision. Twenty retained replays check
         * actual bytes, nonblocking submission, and non-consuming event ownership.
         */
        inline void proveMTPMainForwardReadRetirement(
            IWorkerGPUContext &context, IBackend &backend, DeviceId device)
        {
            constexpr std::size_t bytes = 4096;
            const int ordinal = context.deviceOrdinal();
            ASSERT_TRUE(backend.supportsStreamTimelineSignal64(ordinal));
            std::array<void *, 2> streams{};
            std::array<void *, 2> events{};
            std::array<std::unique_ptr<IGPUGraphCapture>, 3> graphs;
            std::unique_ptr<DeviceGraphOrchestrator> runner;
            auto *storage = static_cast<std::uint8_t *>(backend.allocate(4 * bytes, ordinal));
            ASSERT_NE(storage, nullptr);
            TransferEngine transfers;
            const DeviceId devices[] = {device};
            auto control = transfers.allocateMappedHostRegion(4096u, devices);
            ASSERT_TRUE(control);
            auto *word = static_cast<std::uint64_t *>(control->mutableHostData());
            std::atomic_ref<std::uint64_t>(*word).store(0u, std::memory_order_release);
            auto cleanup = [&](void *) {
                // The gate is owned by this host test. Releasing it directly
                // cannot become queued behind the DMA operation it is holding.
                // This also makes failed assertions safe to unwind on HIP.
                std::atomic_ref<std::uint64_t>(*word).store(1000u, std::memory_order_release);
                for (auto stream : streams)
                    if (stream) context.synchronizeStream(stream);
                runner.reset();
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
            std::array<std::uint8_t, 2 * bytes> sources{};
            for (std::size_t i = 0; i < bytes; ++i)
            {
                sources[i] = static_cast<std::uint8_t>(i % 127);
                sources[bytes + i] = static_cast<std::uint8_t>(128 + i % 127);
            }
            ASSERT_TRUE(backend.hostToDevice(storage, sources.data(), sources.size(), ordinal, streams[0]));
            for (std::size_t phase = 0; phase < graphs.size(); ++phase)
            {
                void *stream = streams[phase == 1 ? 1 : 0];
                auto &graph = graphs[phase];
                graph = context.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                ASSERT_TRUE(graph->beginCapture());
                const std::size_t src = phase == 0 ? 0 : phase == 1 ? 2 : 1;
                const std::size_t dst = phase == 1 ? 3 : 2;
                // Use a compute kernel for overwrite, not another copy-engine
                // command: physical DMA serialization must not mask a missing
                // logical wait on the sidecar's copy stream.
                const bool copied = phase == 2
                    ? backend.enqueuePublishInt32ControlScalarDevice(0x6a432167,
                        reinterpret_cast<std::int32_t *>(storage + 2 * bytes), ordinal, stream)
                    : backend.deviceCopyAsync(storage + dst * bytes,
                        storage + src * bytes, bytes, ordinal, stream);
                const bool ended = graph->endCapture();
                ASSERT_TRUE(copied);
                ASSERT_TRUE(ended);
                ASSERT_TRUE(graph->instantiate());
            }

            GraphConfig config;
            auto builder = std::make_shared<QwenStandardGraph>(config, nullptr);
            runner = std::make_unique<DeviceGraphOrchestrator>(builder, nullptr);
            using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
            ASSERT_TRUE(Peer::initialize(*runner, device));
            constexpr std::array roles{ForwardExecutionRole::MainInference,
                ForwardExecutionRole::MTPCondition, ForwardExecutionRole::GroupedMTPVerifier};
            for (std::uint32_t round = 1; round <= 20; ++round)
            {
                SCOPED_TRACE(round);
                ASSERT_TRUE(graphs[0]->launch());
                ASSERT_TRUE(context.recordEventChecked(events[0], streams[0]));
                ASSERT_TRUE(context.waitEventChecked(events[0], streams[1]));
                ASSERT_TRUE(backend.streamWaitTimelineSignal64(
                    streams[1], control->deviceAlias(device), round, ordinal));
                ASSERT_TRUE(graphs[1]->launch());
                ASSERT_TRUE(Peer::publishRead(*runner, streams[1]));
                if (round == 1)
                    EXPECT_THROW(Peer::prepareMain(*runner, nullptr, device, roles[0]),
                                 std::runtime_error);
                ASSERT_TRUE(Peer::prepareMain(*runner, streams[0], device, roles[(round - 1) % roles.size()]));
                EXPECT_TRUE(Peer::publicationRetained(*runner));
                ASSERT_TRUE(graphs[2]->launch());
                ASSERT_TRUE(context.recordEventChecked(events[1], streams[0]));
                // Give an incorrectly unordered overwrite time to finish. The
                // gate remains closed, so a correct event edge cannot complete.
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                bool overwritten = false;
                ASSERT_TRUE(context.queryEventChecked(events[1], overwritten));
                EXPECT_FALSE(overwritten) << "main overwrote a still-borrowed terminal hidden row";
                std::atomic_ref<std::uint64_t>(*word).store(round, std::memory_order_release);
                context.synchronizeEvent(events[1]);
                std::array<std::uint8_t, bytes> observed{};
                ASSERT_TRUE(backend.deviceToHost(observed.data(), storage + 3 * bytes,
                    bytes, ordinal, streams[1]));
                EXPECT_TRUE(std::equal(observed.begin(), observed.end(), sources.begin()));
            }
        }
    }
}
