/**
 * @file MTPMainForwardReadRetirementProof.h
 * @brief Exercise the real main-forward prelude against a held sidecar reader.
 *
 * A narrow peer supplies a device and the production shifted-KV publication,
 * without loading model weights. Retained copy/scalar-write graphs stand in for
 * hidden producers/readers; prepareLiveStateForForwardGraphExecution owns
 * the ordering edge. A timeline gate makes the race deterministic and proves
 * that submission returns while the reader is still pending on another stream.
 * Metadata observation must retain that publication for a later mailbox writer;
 * waiting on an unrelated metadata stream is not ownership of the write frontier.
 * The same single test peer can seed terminal logits on an exact owned surface;
 * prefix tests then exercise production archive, diagnostic and restore operations.
 */
#pragma once

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/orchestrators/PipelineForwardGraphEdges.h"
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
        /**
         * @brief Seed one exact producer surface; archive/restore remain production operations.
         * @param runner Participant owning the arena and its explicit device stream.
         * @param values Row-zero FP32 values, retained by the caller through archive completion.
         * @param surface Canonical or all-position, full or participant-local storage.
         * @return Whether the real tensor write and scalar publication succeeded.
         */
        static bool publishTerminalRow(DeviceGraphOrchestrator &runner,
            const std::vector<float> &values, ForwardLogitsStorageSurface surface)
        {
            TensorBase *tensor = nullptr;
            if (surface == ForwardLogitsStorageSurface::AllPositionFull ||
                surface == ForwardLogitsStorageSurface::AllPositionLocal)
            {
                TensorBase *full = nullptr, *local = nullptr;
                // Prepare the real two-row storage outside capture. Scalar
                // prefix bridges publish row zero, not the unused final row.
                if (!runner.bindAllPositionLogitsOutputs(ForwardExecutionRole::MainInference,
                    2, full, local)) return false;
                tensor = surface == ForwardLogitsStorageSurface::AllPositionFull ? full : local;
            }
            else if (surface == ForwardLogitsStorageSurface::CanonicalFull)
                tensor = runner.state_.logits.get();
            else if (surface == ForwardLogitsStorageSurface::CanonicalLocal)
                tensor = runner.state_.logits_local.get();
            if (!tensor || tensor->native_type() != TensorType::FP32 ||
                tensor->cols() != values.size()) return false;
            const auto device = runner.state_.device_id;
            if (device.is_gpu())
            {
                void *stream = runner.explicitGPUStreamForOperation("prefix_terminal_test_producer");
                // The model-free fixture has no compute-stage preparation to
                // materialize this output yet. Prepare it once, before any
                // archive/replay, on the same explicit producer stream.
                if (!stream || (!tensor->gpu_data_ptr() && !tensor->allocateOnDevice(device, stream)) ||
                    !getBackendFor(device)->hostToDeviceOnStream(
                    tensor->gpu_data_ptr(), values.data(), values.size() * sizeof(float),
                    device.gpu_ordinal(), stream)) return false;
                TransferEngine::publishDeviceWrite(tensor, device, stream);
            }
            else
            {
                auto *destination = static_cast<float *>(tensor->raw_mutable_data());
                if (!destination) return false;
                std::copy(values.begin(), values.end(), destination);
                TransferEngine::publishHostWrite(tensor);
            }
            return runner.publishCurrentMainLogits(tensor,
                {.execution_role = ForwardExecutionRole::MainInference,
                 .logical_all_position_logits = false,
                 .storage_surface = surface},
                DeviceGraphOrchestrator::MainLogitsPublicationSource::ForwardGraph,
                "model_free_terminal_producer");
        }

        /** @return Exact last-forward producer for terminal-only device assertions. */
        static void *mainForwardProducerStream(DeviceGraphOrchestrator &runner)
        {
            const auto forward = runner.forward_engine_ ? runner.forward_engine_->lastExecutedForwardGraph() : std::nullopt;
            return forward ? forward->stream : nullptr;
        }

        /** @brief Borrow the follower's admitted checkpoint, never a tail rollback bank. */
        static PipelineForwardGraphEdges::FollowerState pipelineFollowerState(DeviceGraphOrchestrator &runner)
        {
            const auto &checkpoint = runner.mtp_publication_main_kv_base_checkpoints_.at(0);
            return {.backend = getBackendFor(runner.state_.device_id),
                .checkpoint = {.cache = runner.state_.kv_cache.get(), .sequence_index = 0,
                    .checkpoint_device = checkpoint.data(), .checkpoint_bytes = checkpoint.bytes}};
        }

        /** @brief Bind the real immutable contributor without constructing model arithmetic. */
        static void bindPipelineEdges(DeviceGraphOrchestrator &runner, PipelineForwardGraphEdges *edges)
        { runner.pipeline_forward_edges_ = edges; }

        /** @brief Enter the real follower materializer without synthesizing a tail outcome. */
        static bool prepareFollowerPublication(DeviceGraphOrchestrator &runner,
            ComputeGraph &verifier, int rows, std::string *error)
        { return runner.materializePipelineFollowerMTPPublicationGraph(verifier, rows, error); }

        /** @return The installed graph, borrowed only while its DGO owns the workspace. */
        static ComputeGraph *publicationGraph(DeviceGraphOrchestrator &runner)
        { return runner.mtp_speculative_state_publication_graph_.graph.get(); }

        /** @return Immutable production publication bindings, not device values. */
        static const MTPSpeculativeStatePublicationStage::Params &publicationParams(DeviceGraphOrchestrator &runner)
        { return runner.mtp_speculative_state_publication_graph_.stage->getParams(); }

        /** @brief Exercise exact-stream and retained-transport checks before publication. */
        static bool executePublication(DeviceGraphOrchestrator &runner, void *stream, std::string *error)
        { return runner.executeMTPStatePublicationGraph(stream, error); }

        /** @brief Install test outcome bindings through the real canonical checkpoint materializer. */
        static bool installPublication(DeviceGraphOrchestrator &runner,
            MTPSpeculativeStatePublicationStage::Params params, ComputeGraph &verifier, std::string *error)
        { return runner.installMTPStatePublicationGraph(std::move(params), verifier, error); }

        /** @brief Capture the complete local publication without launching a collective. */
        static bool capturePublication(DeviceGraphOrchestrator &runner, void *stream, std::string *error)
        { return runner.executeMTPStatePublicationGraph(stream, error,
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch); }

        /** @return The exact compiled production parent, never its uncomposed local child. */
        static const IGPUGraphCapture *publicationCapture(DeviceGraphOrchestrator &runner)
        {
            auto &cache = runner.mtp_speculative_state_publication_graph_;
            const auto &params = cache.stage->getParams();
            const auto view = params.authority == MTPSpeculativeStatePublicationStage::Authority::PipelineFollower
                ? runner.mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(
                    params.request_count, params.verifier_rows_per_request, nullptr)
                : cache.segment_cache.deviceLoopGraphTemplate(*cache.graph);
            return view ? view->capture : nullptr;
        }

        /** @brief Freeze native transport through the participant's ordinary executor. */
        static bool capturePublicationTransport(DeviceGraphOrchestrator &runner, PipelineForwardGraphEdges &edges,
            PipelineForwardGraphEdges::PublicationBanks banks, IDeviceContext &context, IWorkerGPUContext &gpu)
        {
            runner.pipeline_publication_transport_ = edges.materializePublicationTransport(banks, runner.executor_, context, gpu);
            return static_cast<bool>(runner.pipeline_publication_transport_);
        }

        /** @return Opaque retained setup identity, without exposing native graph internals. */
        static const PipelinePublicationTransport *publicationTransportIdentity(DeviceGraphOrchestrator &runner)
        { return runner.pipeline_publication_transport_.get(); }

        /** @brief Test terminal native resource retirement without exposing arena storage. */
        static void retirePublicationTransport(DeviceGraphOrchestrator &runner)
        { runner.pipeline_publication_transport_.reset(); }

        /** @brief Retire borrowed graph/stage identities before a topology-only fixture exits. */
        static void retirePublication(DeviceGraphOrchestrator &runner)
        { runner.mtp_speculative_state_publication_graph_.invalidate(); }

        /** @brief Describe the canonical local input banks used by captured MTP forwards. */
        static ForwardInput followerInput(DeviceGraphOrchestrator &runner, int rows)
        {
            ForwardInput input;
            input.device = runner.state_.device_id;
            input.seq_len = rows;
            input.batch_size = 1;
            input.execution_phase = ForwardExecutionPhase::Decode;
            input.execution_role = rows == 1 ? ForwardExecutionRole::MTPCondition
                                             : ForwardExecutionRole::GroupedMTPVerifier;
            input.kv_cache = runner.state_.kv_cache.get();
            input.external_hidden_state = runner.pp_stage_config_->has_embedding
                ? nullptr : runner.state_.hidden.get();
            if (rows == 1)
            {
                const auto &logical = runner.device_resident_logical_sequence_state_storage_;
                input.token_ids_device = logical.next_condition_tokens_device;
                input.position_ids_device = logical.target_cached_tokens_device;
                input.sequence_lengths_device = logical.target_cached_tokens_device;
            }
            else
            {
                input.token_ids_device = runner.mtp_verifier_input_tokens_dev_;
                input.position_ids_device = static_cast<const int32_t *>(runner.mtp_verifier_position_ids_dev_);
                input.sequence_lengths_device = static_cast<const int32_t *>(runner.mtp_verifier_request_lengths_dev_);
            }
            return input;
        }

        /** @brief Exercise real output binding without borrowing a terminal LM head. */
        static bool bindFollowerOutputs(DeviceGraphOrchestrator &runner, ForwardExecutionRole role, size_t rows)
        {
            auto *full = runner.state_.logits.get();
            auto *local = runner.state_.logits_local.get();
            return runner.bindAllPositionLogitsOutputs(role, rows, full, local) && !full && !local;
        }

        /** @brief A follower's main condition must not require a shifted predictor or archive. */
        static bool bindFollowerCondition(DeviceGraphOrchestrator &runner, ForwardStateTransaction transaction)
        {
            ForwardInput input;
            input.device = runner.state_.device_id;
            input.seq_len = input.batch_size = 1;
            input.execution_role = ForwardExecutionRole::MTPCondition;
            input.execution_phase = ForwardExecutionPhase::Decode;
            input.state_transaction = transaction;
            return runner.bindShiftedMTPPrefillTransaction(input, 1) &&
                runner.bindMTPMainTerminalHiddenTransaction(input, 1) &&
                !input.shifted_mtp_prefill && !input.mtp_main_terminal_hidden;
        }

        /** @brief Bind one real backend and preallocate its production event. */
        static bool initialize(DeviceGraphOrchestrator &runner, DeviceId device)
        {
            runner.state_.device_id = device;
            return runner.initializeShiftedMTPKVReadyEvent() &&
                   runner.initializeMTPPrefillTerminalArchiveReadyEvent() &&
                   runner.initializeForwardGraphOutputReadyEvent();
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

        /** @brief Borrow the production metadata edge without retiring its owner. */
        static bool observeMetadata(const DeviceGraphOrchestrator &runner, void *stream)
        {
            return runner.waitForPendingShiftedMTPKVReadyForObservation(
                stream, "shifted_row_device_outcome_metadata");
        }

        /** @brief Enter the real mailbox-writer boundary on a different stream. */
        static bool prepareMailbox(DeviceGraphOrchestrator &runner, void *stream)
        {
            return runner.beginMTPTerminalHiddenMailboxWrite(stream, "metadata_then_mailbox_test");
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
        /** @brief Two distinct production consumers of a retained sidecar read. */
        enum class MTPReadRetirementBoundary
        {
            MainForward, ///< A main graph borrows without taking the sidecar handoff.
            MetadataThenMailbox, ///< Metadata borrows; the actual mailbox writer consumes.
            PipelineForwardPredecessor, ///< A headless participant has no sampler to join its last forward.
        };
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
         * @param boundary Actual production consumer sequence exercised by this proof.
         *
         * Storage is tested in native bytes, so the lifetime invariant is independent
         * of expert format or activation precision. Twenty retained replays check
         * actual bytes, nonblocking submission, and non-consuming event ownership.
         */
        inline void proveMTPMainForwardReadRetirement(
            IWorkerGPUContext &context, IBackend &backend, DeviceId device,
            MTPReadRetirementBoundary boundary = MTPReadRetirementBoundary::MainForward)
        {
            constexpr std::size_t bytes = 4096;
            const int ordinal = context.deviceOrdinal();
            ASSERT_TRUE(backend.supportsStreamTimelineSignal64(ordinal));
            std::array<void *, 3> streams{};
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
            if (boundary == MTPReadRetirementBoundary::PipelineForwardPredecessor)
                runner->setPPStageConfig({.first_layer = 0, .last_layer = 1,
                    .has_embedding = true, .has_lm_head = false});
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
                if (boundary == MTPReadRetirementBoundary::PipelineForwardPredecessor)
                    ASSERT_TRUE(Peer::publishForward(*runner, device, streams[1], roles[(round - 1) % roles.size()]));
                else
                    ASSERT_TRUE(Peer::publishRead(*runner, streams[1]));
                if (boundary == MTPReadRetirementBoundary::MetadataThenMailbox)
                {
                    // Repeated metadata readers join a third stream. That wait
                    // cannot protect the writer on stream 0 unless the original
                    // publication remains available for its own acquire.
                    ASSERT_TRUE(Peer::observeMetadata(*runner, streams[2]));
                    ASSERT_TRUE(Peer::observeMetadata(*runner, streams[2]));
                    EXPECT_TRUE(Peer::publicationRetained(*runner));
                    ASSERT_TRUE(Peer::prepareMailbox(*runner, streams[0]));
                    EXPECT_FALSE(Peer::publicationRetained(*runner));
                }
                else
                {
                    if (round == 1 && boundary != MTPReadRetirementBoundary::PipelineForwardPredecessor)
                        EXPECT_THROW(Peer::prepareMain(*runner, nullptr, device, roles[0]),
                                     std::runtime_error);
                    ASSERT_TRUE(Peer::prepareMain(*runner, streams[0], device, roles[(round - 1) % roles.size()]));
                    if (boundary != MTPReadRetirementBoundary::PipelineForwardPredecessor)
                        EXPECT_TRUE(Peer::publicationRetained(*runner));
                }
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
