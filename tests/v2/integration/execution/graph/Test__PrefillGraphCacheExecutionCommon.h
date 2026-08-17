/**
 * @file Test__PrefillGraphCacheExecutionCommon.h
 * @brief Shared GPU integration test body for bucketed prefill graph capture.
 *
 * Exercises the production ForwardExecutionEngine prefill path with a small
 * graph containing a real GPU residual-add kernel. Padded-bucket cases append a
 * HiddenStateRowSelectStage so replay-param updates are exercised across real
 * lengths. The graph is intentionally tiny so the test isolates cache lifecycle
 * behavior while still using DeviceGraphExecutor, backend streams, and HIP/CUDA
 * graph capture/replay.
 * Backend-specific wrapper files provide the registration/support/device hooks.
 */

#pragma once

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

#include "backends/GPUDeviceContextPool.h"
#include "backends/BackendManager.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/compute_stages/stages/RoPEStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/engine/ForwardExecutionEngine.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "../../../utils/GraphArenaTestHarness.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr int kExactBucketSeqLen = 64;
    constexpr int kLargeBucketSeqLen = 4096;
    constexpr int kLongContextSeqLen = 256 * 1024 + 1;
    constexpr int kHiddenDim = 64;
    constexpr int kKVProbeHeadDim = 32;
    constexpr int kKVProbeHeads = 1;
    constexpr int kKVProbeDim = kKVProbeHeads * kKVProbeHeadDim;
    constexpr int kPadTokenId = 0;

    double findPrefillGraphLifecycleCounter(
        const std::vector<PerfStatRecord> &records,
        const PerfStatsCollector::Tags &tags)
    {
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == "forward_graph" &&
                record.name == "prefill_graph_lifecycle" &&
                record.phase == "prefill" &&
                record.tags == tags)
            {
                return record.value;
            }
        }
        return 0.0;
    }

    /**
     * @brief Scoped environment override that reloads debugEnv() immediately.
     *
     * DebugEnv caches environment values, so tests that toggle graph-capture
     * gates must reload after setting variables and again after restoring them.
     */
    class ScopedDebugEnv
    {
    public:
        explicit ScopedDebugEnv(std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old_value = std::getenv(name))
                {
                    entry.had_value = true;
                    entry.old_value = old_value;
                }
                entries_.push_back(entry);
                ::setenv(name, value, 1);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedDebugEnv()
        {
            for (const auto &entry : entries_)
            {
                if (entry.had_value)
                    ::setenv(entry.name.c_str(), entry.old_value.c_str(), 1);
                else
                    ::unsetenv(entry.name.c_str());
            }
            mutableDebugEnv().reload();
        }

        ScopedDebugEnv(const ScopedDebugEnv &) = delete;
        ScopedDebugEnv &operator=(const ScopedDebugEnv &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_value = false;
            std::string old_value;
        };

        std::vector<Entry> entries_;
    };

    /**
     * @brief Capturable one-kernel GPU stage used by the prefill cache test.
     *
     * The stage uses the same BufferArena contract and prepared-input API as a
     * production stage. The actual work is the backend residual-add kernel, so
     * HIP/CUDA graph capture records real GPU nodes rather than a mock callback.
     */
    class GPUResidualAddProbeStage final : public IComputeStage
    {
    public:
        GPUResidualAddProbeStage(
            std::string name,
            DeviceId device,
            FP32Tensor *input,
            FP32Tensor *residual,
            FP32Tensor *output,
            int rows,
            int cols)
            : IComputeStage(device), name_(std::move(name)), input_(input), residual_(residual),
              output_(output), rows_(rows), cols_(cols)
        {
        }

        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || !ctx->isGPU() || ctx->deviceId() != device())
            {
                LOG_ERROR("[GPUResidualAddProbeStage] Invalid GPU context");
                return false;
            }
            if (!input_ || !residual_ || !output_)
            {
                LOG_ERROR("[GPUResidualAddProbeStage] Missing tensor pointer");
                return false;
            }

            /*
             * DeviceGraphExecutor owns all movement and storage preparation.
             * The stage may only consume the exact device buffers established
             * from bufferContract() on its producer stream.
             */
            const StageGPUExecution execution = gpuExecution();
            execution.requirePreparedInput(input_);
            execution.requirePreparedInput(residual_);
            execution.requirePreparedOutput(output_);

            ITensorResidualAdd *kernel = nullptr;
            try
            {
                kernel = llaminar::v2::kernels::KernelFactory::getOrCreateResidualAdd(input_, device());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[GPUResidualAddProbeStage] Kernel creation failed: " << e.what());
                return false;
            }
            if (!kernel)
            {
                LOG_ERROR("[GPUResidualAddProbeStage] KernelFactory returned null residual-add kernel");
                return false;
            }

            kernel->setGPUStream(gpuStream());
            const size_t num_elements = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
            const bool ok = kernel->apply_tensor(
                input_, residual_, output_, num_elements, nullptr, device().toKernelDeviceIndex());
            if (!ok)
                return false;

            ++execute_count_;
            TransferEngine::publishDeviceWrite(output_, device(), gpuStream());
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        std::string name() const override { return name_; }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::GPU_CUDA ||
                   backend == ComputeBackendType::GPU_ROCM;
        }

        StageBufferContract bufferContract() const override
        {
            return StageBufferContract::build()
                .addInput(BufferId::HIDDEN_STATE)
                .addInput(BufferId::RESIDUAL)
                .addOutput(BufferId::ATTN_OUTPUT);
        }

        bool isGraphCapturable() const override { return true; }

        size_t estimatedFlops() const override
        {
            return static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
        }

        size_t estimatedMemoryBytes() const override
        {
            return 3 * static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * sizeof(float);
        }

        int executeCount() const { return execute_count_; }

    private:
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addInput("input", input_, rows_, cols_);
            info.addInput("residual", residual_, rows_, cols_);
            info.addOutput("output", output_, rows_, cols_);
            return info;
        }

        std::string name_;
        FP32Tensor *input_ = nullptr;
        FP32Tensor *residual_ = nullptr;
        FP32Tensor *output_ = nullptr;
        int rows_ = 0;
        int cols_ = 0;
        int execute_count_ = 0;
    };

    /**
     * @brief RoPE probe that keeps production dynamic-position behavior.
     *
     * This subclass changes only backend eligibility for the small integration
     * graph. Buffer ownership and stream ordering remain the production
     * RoPEStage contract.
     */
    class GPURoPEProbeStage final : public RoPEStage
    {
    public:
        explicit GPURoPEProbeStage(RoPEStage::Params params)
            : RoPEStage(std::move(params)) {}

        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::GPU_CUDA ||
                   backend == ComputeBackendType::GPU_ROCM;
        }
    };

    /**
     * @brief Minimal ForwardExecutionEngine host that builds one GPU graph.
     */
    class PrefillGraphCacheTestHost final : public IForwardExecutionHost
    {
    public:
        PrefillGraphCacheTestHost(
            DeviceId device,
            IDeviceContext *ctx,
            test::GraphArenaTestHarness *arena_harness)
            : device_(device), ctx_(ctx), arena_harness_(arena_harness)
        {
            initializePersistentArenaTensors();
        }

        ~PrefillGraphCacheTestHost() override
        {
            if (!resident_length_backend_)
                return;

            /*
             * Integration teardown is an explicit host ownership boundary. The
             * test has already materialized its final graph output, but drain
             * once before releasing the admission event and storage so a failed
             * assertion cannot let asynchronous GPU work outlive this fixture.
             */
            if (ctx_)
                (void)ctx_->synchronize();
            const int ordinal = device_.toKernelDeviceIndex();
            if (resident_length_event_)
                resident_length_backend_->destroyEvent(
                    resident_length_event_,
                    ordinal);
            if (resident_length_device_)
                resident_length_backend_->free(
                    resident_length_device_,
                    ordinal);
            if (resident_length_host_)
                resident_length_backend_->freePinned(
                    resident_length_host_,
                    ordinal);
            if (resident_position_ids_device_)
                resident_length_backend_->free(
                    resident_position_ids_device_,
                    ordinal);
            if (resident_position_ids_host_)
                resident_length_backend_->freePinned(
                    resident_position_ids_host_,
                    ordinal);
        }

        /**
         * @brief Admit one real request length into persistent device storage.
         *
         * This is the small integration equivalent of
         * DeviceGraphOrchestrator::admitRequestInputsOnDevice(). The allocation
         * and readiness event are persistent; each invocation changes only the
         * pinned source scalar and enqueues one H2D admission copy. The forward
         * prelude consumes the event with a GPU-side wait, so no host stream or
         * device synchronization enters the execution path.
         */
        bool admitResidentRequestLength(int real_seq_len)
        {
            return admitResidentRequestMetadata(
                real_seq_len,
                /*position_ids=*/nullptr,
                /*position_row_count=*/0);
        }

        /**
         * @brief Admit real-row count and absolute positions as one GPU transaction.
         *
         * The pinned source storage, device destination storage, producer
         * stream, and publication event all persist for the fixture lifetime.
         * Replays therefore mutate only buffer contents while every captured
         * kernel retains the same device address. The event is recorded after
         * both H2D copies, making one stream wait sufficient to order every
         * request-metadata consumer.
         *
         * @param real_seq_len Number of non-padding rows in the request.
         * @param position_ids Flattened absolute position rows, or null when
         *        this invocation only needs resident length metadata.
         * @param position_row_count Physical number of position rows.
         * @return true after the complete metadata transaction is enqueued.
         */
        bool admitResidentRequestMetadata(
            int real_seq_len,
            const int *position_ids,
            int position_row_count)
        {
            if (real_seq_len <= 0 || !device_.is_gpu())
                return false;
            if ((position_ids == nullptr) != (position_row_count == 0) ||
                position_row_count < 0)
            {
                return false;
            }

            if (!resident_length_backend_)
            {
                resident_length_backend_ = getBackendFor(device_);
                if (!resident_length_backend_)
                    return false;

                const int ordinal = device_.toKernelDeviceIndex();
                resident_length_device_ =
                    resident_length_backend_->allocate(
                        sizeof(int32_t),
                        ordinal);
                resident_length_host_ =
                    static_cast<int32_t *>(
                        resident_length_backend_->allocatePinned(
                            sizeof(int32_t),
                            ordinal));
                resident_length_event_ =
                    resident_length_backend_->createEvent(ordinal);
                IWorkerGPUContext *gpu_ctx = getWorkerGPUContext(device_);
                if (gpu_ctx)
                {
                    /*
                     * The worker creates its default stream during asynchronous
                     * context initialization. Reading defaultStream() directly
                     * from the test thread races that publication and can
                     * observe a transient null handle. Resolve the stream on
                     * the owning worker after initialization has completed;
                     * subsequent admission copies may enqueue against the
                     * stable CUDA/HIP stream handle from any host thread.
                     */
                    gpu_ctx->submitAndWait([&]()
                    {
                        resident_length_producer_stream_ =
                            gpu_ctx->defaultStream();
                    });
                }
                if (!resident_length_device_ ||
                    !resident_length_host_ ||
                    !resident_length_event_ ||
                    !resident_length_producer_stream_)
                {
                    return false;
                }
            }

            if (position_row_count > 0 &&
                !ensureResidentPositionCapacity(position_row_count))
            {
                return false;
            }

            *resident_length_host_ = static_cast<int32_t>(real_seq_len);
            const int ordinal = device_.toKernelDeviceIndex();
            if (!resident_length_backend_->hostToDeviceOnStream(
                    resident_length_device_,
                    resident_length_host_,
                    sizeof(int32_t),
                    ordinal,
                    resident_length_producer_stream_))
            {
                return false;
            }

            if (position_row_count > 0)
            {
                static_assert(
                    sizeof(int) == sizeof(int32_t),
                    "Resident position admission requires 32-bit host ints");
                const size_t position_bytes =
                    static_cast<size_t>(position_row_count) * sizeof(int32_t);
                std::memcpy(
                    resident_position_ids_host_,
                    position_ids,
                    position_bytes);
                if (!resident_length_backend_->hostToDeviceOnStream(
                        resident_position_ids_device_,
                        resident_position_ids_host_,
                        position_bytes,
                        ordinal,
                        resident_length_producer_stream_))
                {
                    return false;
                }
            }

            if (!resident_length_backend_->recordEvent(
                    resident_length_event_,
                    ordinal,
                    resident_length_producer_stream_))
            {
                return false;
            }
            resident_length_pending_ = true;
            resident_position_ids_pending_ = position_row_count > 0;
            return true;
        }

        /// @brief Return the stable device owner used by padded graph kernels.
        const int32_t *residentRequestLengthDevice() const
        {
            return static_cast<const int32_t *>(resident_length_device_);
        }

        /// @brief Return the stable device owner for admitted absolute positions.
        const int32_t *residentPositionIdsDevice() const
        {
            return static_cast<const int32_t *>(resident_position_ids_device_);
        }

        GraphBuildResult buildForwardGraph(const ForwardInput &input) override
        {
            ++build_calls;
            last_build_seq_len = input.seq_len;
            last_build_bucket_seq_len = input.bucket_seq_len;
            last_build_real_seq_len = input.real_seq_len;
            output_tensor_ = nullptr;
            stage_ = nullptr;
            row_select_stage_ = nullptr;
            kv_append_stage_ = nullptr;
            rope_stage_ = nullptr;

            if (input.device != device_ || input.seq_len <= 0)
                return GraphBuildResult("invalid input for GPU prefill graph cache test");
            if (!arena_bindings_ready_)
                return GraphBuildResult("failed to initialize persistent arena bindings");
            if (input.seq_len > kLargeBucketSeqLen)
                return GraphBuildResult("prefill graph cache test exceeded persistent bucket capacity");

            if (use_kv_append_probe_ &&
                !initializeKVCacheProbeForTesting())
            {
                return GraphBuildResult(
                    "failed to create GPU KV cache probe");
            }

            FP32Tensor *input_ptr = input_tensor_;
            FP32Tensor *residual_ptr = residual_tensor_;
            FP32Tensor *residual_output_ptr = residual_output_tensor_;

            auto stage = std::make_unique<GPUResidualAddProbeStage>(
                "gpu_residual_add_probe",
                device_,
                input_ptr,
                residual_ptr,
                residual_output_ptr,
                input.seq_len,
                kHiddenDim);
            stage_ = stage.get();

            ComputeGraph graph;
            graph.addNode("gpu_residual_add_probe", std::move(stage), device_);

            if (use_kv_append_probe_)
            {
                FP32Tensor *k_ptr = k_tensor_;
                FP32Tensor *v_ptr = v_tensor_;

                KVCacheAppendStage::Params kv_params;
                kv_params.K = k_ptr;
                kv_params.V = v_ptr;
                kv_params.kv_cache = kv_cache_.get();
                kv_params.layer_idx = 0;
                kv_params.seq_idx = 0;
                kv_params.num_tokens = input.seq_len;
                kv_params.seq_len = input.seq_len;
                kv_params.batch_size = 1;
                kv_params.head_dim = kKVProbeHeadDim;
                kv_params.device_id = device_;
                kv_params.request_sequence_lengths_device =
                    input.sequence_lengths_device;
                kv_params.k_buffer_id = BufferId::K_PROJ;
                kv_params.v_buffer_id = BufferId::V_PROJ;

                auto kv_stage = std::make_unique<KVCacheAppendStage>(kv_params);
                kv_append_stage_ = kv_stage.get();
                graph.addNode("kv_append_probe", std::move(kv_stage), device_);
                graph.addDependency("kv_append_probe", "gpu_residual_add_probe");
            }

            if (use_row_select_probe_)
            {
                output_tensor_ = selected_row_tensor_;

                const int initial_real_seq_len = input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
                HiddenStateRowSelectStage::Params row_params;
                row_params.input = residual_output_ptr;
                row_params.output = output_tensor_;
                row_params.seq_len = input.seq_len;
                row_params.d_model = kHiddenDim;
                row_params.selected_row_idx = initial_real_seq_len - 1;
                row_params.device_id = device_;
                row_params.selection_policy =
                    input.sequence_lengths_device
                        ? HiddenStateRowSelectStage::SelectionPolicy::
                              DeviceResidentRequestLength
                        : HiddenStateRowSelectStage::SelectionPolicy::
                              DynamicDeviceScalar;
                row_params.request_sequence_length_device =
                    input.sequence_lengths_device;
                row_params.input_buffer_id = BufferId::ATTN_OUTPUT;
                row_params.output_buffer_id = BufferId::LM_HEAD_INPUT_ROW;

                auto row_select_stage = std::make_unique<HiddenStateRowSelectStage>(row_params);
                row_select_stage_ = row_select_stage.get();
                graph.addNode("hidden_state_row_select", std::move(row_select_stage), device_);
                graph.addDependency("hidden_state_row_select", "gpu_residual_add_probe");
            }
            else
            {
                output_tensor_ = residual_output_ptr;
            }

            if (use_rope_probe_)
            {
                RoPEStage::Params rope_params;
                rope_params.Q = residual_output_ptr;
                rope_params.K = nullptr;
                rope_params.n_heads = 1;
                rope_params.n_kv_heads = 0;
                rope_params.head_dim = kHiddenDim;
                rope_params.pos_offset = input.position_offset;
                rope_params.theta_base = 10000.0f;
                rope_params.seq_len = input.seq_len;
                rope_params.partial_rotary_factor = 1.0f;
                rope_params.position_ids = input.position_ids;
                rope_params.position_ids_device = input.position_ids_device;
                rope_params.device_id = device_;
                rope_params.q_buffer_id = BufferId::ATTN_OUTPUT;

                auto rope_stage = std::make_unique<GPURoPEProbeStage>(rope_params);
                rope_stage_ = rope_stage.get();
                graph.addNode("rope_position_probe", std::move(rope_stage), device_);
                graph.addDependency("rope_position_probe", "gpu_residual_add_probe");
                output_tensor_ = residual_output_ptr;
            }

            ForwardOutput output;
            output.logits = output_tensor_;
            output.hidden = output_tensor_;
            return GraphBuildResult(std::move(graph), output);
        }

        IDeviceContext *getDeviceContext(DeviceId device) override
        {
            ++get_context_calls;
            return device == device_ ? ctx_ : nullptr;
        }

        IWorkerGPUContext *getWorkerGPUContext(DeviceId device) override
        {
            if (device != device_ || !device.is_gpu())
                return nullptr;
            return &GPUDeviceContextPool::instance().getContext(device);
        }

        bool workerGPUContextUsesProcessPool(DeviceId device) const override
        {
            return device == device_ && device.is_gpu();
        }

        std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() override
        {
            return {{device_, ctx_}};
        }

        bool ensureDeviceWorkspaceAllocated(const ComputeGraph &graph, int workspace_seq_len) override
        {
            ++ensure_workspace_calls;
            last_workspace_seq_len = workspace_seq_len;

            WorkspaceRequirements combined;
            std::vector<IWorkspaceConsumer *> consumers;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const ComputeNode *node = graph.getNode(node_name);
                if (!node || !node->stage)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(node->stage.get());
                if (!consumer)
                    continue;
                consumers.push_back(consumer);
                combined.merge(consumer->getWorkspaceRequirements(workspace_seq_len, 0, 0));
            }

            if (consumers.empty())
                return true;

            if (!workspace_ || workspace_seq_len != workspace_seq_len_)
            {
                workspace_ = std::make_unique<DeviceWorkspaceManager>(device_, 1 << 20);
                workspace_seq_len_ = workspace_seq_len;
                if (!workspace_->allocate(combined))
                    return false;
            }

            for (auto *consumer : consumers)
                consumer->bindWorkspace(workspace_.get());
            return true;
        }

        bool publishForwardResultAtBoundary(
            const ForwardOutput &output,
            IDeviceContext *ctx) override
        {
            ++sync_logits_calls;
            if (!ctx || !ctx->isGPU() || !output.execution.stream ||
                !output.logits || output.logits != output_tensor_)
                return false;

            /*
             * Mirror the production device result boundary exactly. Tensor
             * event records made while capture is active are graph nodes rather
             * than replay-completion fences, so republish the graph-declared
             * tensor after executable launch without forcing host visibility.
             */
            TransferEngine::publishDeviceWrite(
                output.logits,
                ctx->deviceId(),
                output.execution.stream);
            return true;
        }

        void commitSuccessfulForwardOutput(
            const ForwardOutput &output) override
        {
            if (!output.execution.valid || !output.execution.stream ||
                output.execution.device != device_ ||
                output.logits != output_tensor_)
            {
                throw std::logic_error(
                    "Prefill graph fixture received an invalid successful-output publication");
            }
            ++committed_forward_output_calls;
            last_committed_logits = output.logits;
            last_committed_execution = output.execution;
        }

        bool prepareLiveStateForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device) override
        {
            if (!resident_length_pending_)
                return true;
            if (!resident_length_backend_ ||
                !resident_length_event_ ||
                !resident_length_producer_stream_ ||
                !execution_stream ||
                execution_device != device_ ||
                input.sequence_lengths_device != residentRequestLengthDevice() ||
                (resident_position_ids_pending_ &&
                 input.position_ids_device != residentPositionIdsDevice()))
            {
                return false;
            }

            if (execution_stream != resident_length_producer_stream_ &&
                !resident_length_backend_->streamWaitEvent(
                    execution_stream,
                    resident_length_event_,
                    device_.toKernelDeviceIndex()))
            {
                return false;
            }
            resident_length_pending_ = false;
            resident_position_ids_pending_ = false;
            return true;
        }

        DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool, IDeviceContext *) const override
        {
            return {};
        }

        PPCopyInfo resolvePPCopyInfo(const ForwardInput &) const override { return {}; }

        uint64_t moePlacementEpoch() const override { return placement_epoch; }
        std::string prefillGraphDomainId() const override { return domain_id; }
        int prefillGraphParticipantId() const override { return participant_id; }
        uint64_t prefillGraphTopologySignature() const override { return topology_signature; }

        PrefillChunkMaintenanceState prefillChunkMaintenanceState(
            const PrefillChunkPlan &chunk) const override
        {
            PrefillChunkMaintenanceState state;
            state.chunk_index = chunk.chunk_index;
            state.histograms_merged = true;
            state.manual_boundaries_complete = true;
            state.participants_at_same_boundary = true;
            state.rebalance_requested = rebalance_requested_on_boundary_;
            return state;
        }

        bool onPrefillChunkMaintenance(
            const PrefillChunkPlan &chunk,
            const PrefillChunkMaintenanceDecision &decision) override
        {
            ++maintenance_calls;
            last_maintenance_chunk = chunk;
            last_maintenance_decision = decision;
            if (!decision.ok)
                return false;
            if (bump_epoch_on_maintenance_)
                ++placement_epoch;
            if (topology_delta_on_maintenance_ != 0)
                topology_signature += topology_delta_on_maintenance_;
            if (engine_to_clear_on_maintenance_)
                engine_to_clear_on_maintenance_->discardAllCachedGraphs();
            return true;
        }

        /// @brief Enable the row-select replay-param consumer for padded bucket tests.
        void setUseRowSelectProbe(bool enabled) { use_row_select_probe_ = enabled; }

        /// @brief Enable the real GPU KV append replay-param consumer.
        void setUseKVAppendProbe(bool enabled) { use_kv_append_probe_ = enabled; }

        /**
         * @brief Materialize the persistent KV probe before graph construction.
         *
         * Device-owned chunk materialization binds the canonical cached-token
         * address as graph identity. The cache therefore has to exist before
         * ForwardExecutionEngine asks this host to build its first graph.
         */
        bool initializeKVCacheProbeForTesting()
        {
            if (!use_kv_append_probe_)
                return false;
            if (kv_cache_)
                return true;
            try
            {
                llaminar::v2::kernels::KVCacheConfig config;
                config.precision = ActivationPrecision::FP32;
                config.device = device_;
                config.num_layers = 1;
                config.batch_size = 1;
                config.max_seq_len = kv_cache_capacity_;
                config.n_kv_heads = kKVProbeHeads;
                config.head_dim = kKVProbeHeadDim;
                kv_cache_ =
                    llaminar::v2::kernels::KernelFactory::createKVCache(
                        config);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[PrefillGraphCacheTestHost] Failed to initialize KV probe: "
                          << e.what());
                return false;
            }
            return kv_cache_ != nullptr;
        }

        /** @return Canonical device progress address owned by the KV probe. */
        const int32_t *kvSequenceCachedTokensDeviceForTesting() const
        {
            return kv_cache_
                       ? kv_cache_->deviceSequenceCachedTokenCountPtr(0)
                       : nullptr;
        }

        /**
         * @brief Set the logical capacity of the lazily-created KV probe.
         *
         * The capacity is immutable once the cache exists because changing it
         * would replace device addresses embedded in captured executables.
         * Long-context tests call this before their first graph build, while
         * ordinary lifecycle tests retain the compact default allocation.
         *
         * @param max_tokens Positive logical token capacity.
         * @return true when the capacity was accepted before cache creation.
         */
        bool setKVCacheCapacityForTesting(int max_tokens)
        {
            if (max_tokens <= 0 || kv_cache_)
                return false;
            kv_cache_capacity_ = max_tokens;
            return true;
        }

        /// @brief Enable the real GPU RoPE dynamic-position consumer.
        void setUseRoPEProbe(bool enabled) { use_rope_probe_ = enabled; }

        /// @brief Make chunk maintenance emulate a placement-changing rebalance.
        void setPlacementChangingMaintenance(
            ForwardExecutionEngine *engine,
            bool bump_epoch,
            uint64_t topology_delta)
        {
            engine_to_clear_on_maintenance_ = engine;
            bump_epoch_on_maintenance_ = bump_epoch;
            topology_delta_on_maintenance_ = topology_delta;
        }

        void setRebalanceRequestedOnBoundary(bool requested)
        {
            rebalance_requested_on_boundary_ = requested;
        }

        /// @brief Return the tensor exposed as logits/hidden by the synthetic graph.
        FP32Tensor *outputTensor() const { return output_tensor_; }

        /// @brief Return the residual probe stage built for the cached graph.
        GPUResidualAddProbeStage *stage() const { return stage_; }

        /// @brief Return the optional row-select stage built for padded bucket tests.
        HiddenStateRowSelectStage *rowSelectStage() const { return row_select_stage_; }

        /// @brief Return the optional KV append stage built for padded bucket tests.
        KVCacheAppendStage *kvAppendStage() const { return kv_append_stage_; }

        /// @brief Return the optional RoPE stage built for dynamic-position tests.
        RoPEStage *ropeStage() const { return rope_stage_; }

        /// @brief Return the logical cached-token count for the probe KV cache.
        int kvCachedTokensForTesting() const
        {
            return kv_cache_ ? kv_cache_->get_cached_tokens(0, 0) : 0;
        }

        /**
         * @brief Read the device-owned probe count on one exact observation stream.
         *
         * The caller must first enqueue any producer-event dependency required
         * by that stream. Synchronizing only the observation stream then proves
         * the transfer cannot accidentally rely on unrelated device work.
         *
         * @param observation_stream Explicit stream for the terminal D2H copy,
         *        or null to use the fixture's backend stream in legacy tests.
         * @return Cached-token count, or -1 when the observation contract fails.
         */
        int kvDeviceCachedTokensForTesting(
            void *observation_stream = nullptr) const
        {
            if (!kv_cache_ || !ctx_)
                return -1;
            const int *device_count = kv_cache_->deviceCachedTokenCountPtr(0, 0);
            if (!device_count)
                return -1;

            IBackend *backend = getBackendFor(device_);
            if (!backend)
                return -1;

            int value = -1;
            void *stream = observation_stream;
            if (!stream && device_.is_cuda())
            {
                stream = GPUDeviceContextPool::instance()
                             .getNvidiaContext(device_.toKernelDeviceIndex())
                             .defaultStream();
            }
            else if (!stream && device_.is_rocm())
            {
                stream = GPUDeviceContextPool::instance()
                             .getAMDContext(device_.toKernelDeviceIndex())
                             .defaultStream();
            }
            if (!stream)
                return -1;
            if (!backend->deviceToHostFast(
                    &value,
                    device_count,
                    sizeof(value),
                    device_.toKernelDeviceIndex(),
                    stream))
            {
                return -1;
            }
            if (!backend->synchronizeStream(
                    stream,
                    device_.toKernelDeviceIndex()))
            {
                return -1;
            }
            return value;
        }

        int build_calls = 0;
        int get_context_calls = 0;
        int ensure_workspace_calls = 0;
        int sync_logits_calls = 0;
        int committed_forward_output_calls = 0;
        TensorBase *last_committed_logits = nullptr;
        ForwardExecutionProvenance last_committed_execution{};
        int last_workspace_seq_len = -1;
        int last_build_seq_len = 0;
        int last_build_bucket_seq_len = 0;
        int last_build_real_seq_len = 0;
        std::string domain_id = "single";
        int participant_id = 0;
        uint64_t placement_epoch = 0;
        uint64_t topology_signature = 0;
        int maintenance_calls = 0;
        PrefillChunkPlan last_maintenance_chunk{};
        PrefillChunkMaintenanceDecision last_maintenance_decision{};

    private:
        /**
         * @brief Create one stable arena address set shared by every cached bucket.
         *
         * The production graph cache captures arena-owned addresses once and may
         * retain several bucket geometries concurrently. This fixture therefore
         * allocates the largest tested shape once, binds each semantic BufferId
         * once, and varies only the active row count in stage parameters. A
         * per-build rebind would invalidate older cached executables and conceal
         * exactly the lifetime bugs this suite is intended to catch.
         */
        void initializePersistentArenaTensors()
        {
            if (!arena_harness_)
                return;

            const std::vector<size_t> hidden_shape{
                static_cast<size_t>(kLargeBucketSeqLen),
                static_cast<size_t>(kHiddenDim)};
            const std::vector<size_t> kv_shape{
                static_cast<size_t>(kLargeBucketSeqLen),
                static_cast<size_t>(kKVProbeDim)};

            input_tensor_ = arena_harness_->createPersistentTensor<FP32Tensor>(
                BufferId::HIDDEN_STATE,
                hidden_shape,
                DeviceId::cpu());
            residual_tensor_ = arena_harness_->createPersistentTensor<FP32Tensor>(
                BufferId::RESIDUAL,
                hidden_shape,
                DeviceId::cpu());
            residual_output_tensor_ =
                arena_harness_->createPersistentTensor<FP32Tensor>(
                    BufferId::ATTN_OUTPUT,
                    hidden_shape,
                    DeviceId::cpu());
            selected_row_tensor_ =
                arena_harness_->createPersistentTensor<FP32Tensor>(
                    BufferId::LM_HEAD_INPUT_ROW,
                std::vector<size_t>{1, static_cast<size_t>(kHiddenDim)},
                DeviceId::cpu());
            k_tensor_ = arena_harness_->createPersistentTensor<FP32Tensor>(
                BufferId::K_PROJ,
                kv_shape,
                DeviceId::cpu());
            v_tensor_ = arena_harness_->createPersistentTensor<FP32Tensor>(
                BufferId::V_PROJ,
                kv_shape,
                DeviceId::cpu());

            const size_t hidden_count =
                static_cast<size_t>(kLargeBucketSeqLen) *
                static_cast<size_t>(kHiddenDim);
            for (size_t i = 0; i < hidden_count; ++i)
            {
                input_tensor_->mutable_data()[i] =
                    1.0f + static_cast<float>(i % 17) * 0.125f;
                residual_tensor_->mutable_data()[i] =
                    0.25f + static_cast<float>(i % 13) * 0.0625f;
            }

            for (int row = 0; row < kLargeBucketSeqLen; ++row)
            {
                for (int col = 0; col < kKVProbeDim; ++col)
                {
                    const size_t index =
                        static_cast<size_t>(row) *
                            static_cast<size_t>(kKVProbeDim) +
                        static_cast<size_t>(col);
                    k_tensor_->mutable_data()[index] =
                        0.01f * static_cast<float>((row + col) % 19 + 1);
                    v_tensor_->mutable_data()[index] =
                        0.02f * static_cast<float>((row * 3 + col) % 23 + 1);
                }
            }

            arena_bindings_ready_ = true;
        }

        /**
         * @brief Materialize stable pinned/device position storage before launch.
         *
         * Capacity may grow only between fixture invocations, before admission
         * is published. It never changes while an executable is capturing or
         * replaying. Production obtains the same stronger property from
         * BufferArena's request-input allocation.
         */
        bool ensureResidentPositionCapacity(int required_rows)
        {
            if (required_rows <= resident_position_ids_capacity_)
                return resident_position_ids_device_ && resident_position_ids_host_;
            if (!resident_length_backend_ || resident_length_pending_)
                return false;

            const int ordinal = device_.toKernelDeviceIndex();
            if (resident_position_ids_device_)
                resident_length_backend_->free(
                    resident_position_ids_device_,
                    ordinal);
            if (resident_position_ids_host_)
                resident_length_backend_->freePinned(
                    resident_position_ids_host_,
                    ordinal);
            resident_position_ids_device_ = nullptr;
            resident_position_ids_host_ = nullptr;
            resident_position_ids_capacity_ = 0;

            const size_t bytes =
                static_cast<size_t>(required_rows) * sizeof(int32_t);
            resident_position_ids_device_ =
                resident_length_backend_->allocate(bytes, ordinal);
            resident_position_ids_host_ =
                static_cast<int32_t *>(
                    resident_length_backend_->allocatePinned(bytes, ordinal));
            if (!resident_position_ids_device_ || !resident_position_ids_host_)
                return false;

            resident_position_ids_capacity_ = required_rows;
            return true;
        }

        DeviceId device_;
        IDeviceContext *ctx_ = nullptr;
        test::GraphArenaTestHarness *arena_harness_ = nullptr; ///< Borrowed stable arena/tensor owner.
        bool arena_bindings_ready_ = false;
        bool use_row_select_probe_ = false; ///< Whether to append HiddenStateRowSelectStage after residual add.
        bool use_kv_append_probe_ = false;  ///< Whether to append real GPU KVCacheAppendStage after residual add.
        bool use_rope_probe_ = false;       ///< Whether to append real GPU RoPEStage after residual add.
        int kv_cache_capacity_ = 512;       ///< Immutable logical capacity selected before lazy cache creation.
        bool rebalance_requested_on_boundary_ = false;
        bool bump_epoch_on_maintenance_ = false;
        uint64_t topology_delta_on_maintenance_ = 0;
        ForwardExecutionEngine *engine_to_clear_on_maintenance_ = nullptr;
        FP32Tensor *input_tensor_ = nullptr;
        FP32Tensor *residual_tensor_ = nullptr;
        FP32Tensor *residual_output_tensor_ = nullptr;
        FP32Tensor *selected_row_tensor_ = nullptr;
        FP32Tensor *k_tensor_ = nullptr;
        FP32Tensor *v_tensor_ = nullptr;
        std::unique_ptr<IKVCache> kv_cache_;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        int workspace_seq_len_ = 0;
        FP32Tensor *output_tensor_ = nullptr;
        GPUResidualAddProbeStage *stage_ = nullptr;
        HiddenStateRowSelectStage *row_select_stage_ = nullptr;
        KVCacheAppendStage *kv_append_stage_ = nullptr;
        RoPEStage *rope_stage_ = nullptr;
        IBackend *resident_length_backend_ = nullptr; ///< Borrowed backend for request admission.
        int32_t *resident_length_host_ = nullptr; ///< Persistent pinned scalar copied at admission.
        void *resident_length_device_ = nullptr; ///< Persistent device INT32 real-row owner.
        void *resident_length_event_ = nullptr; ///< Exact admission-completion event.
        void *resident_length_producer_stream_ = nullptr; ///< Stream that publishes admission.
        bool resident_length_pending_ = false; ///< True until the graph stream consumes the event.
        int32_t *resident_position_ids_host_ = nullptr; ///< Persistent pinned absolute positions.
        void *resident_position_ids_device_ = nullptr; ///< Stable device absolute-position rows.
        int resident_position_ids_capacity_ = 0; ///< Allocated flattened position-row capacity.
        bool resident_position_ids_pending_ = false; ///< Whether the current event publishes positions.
    };

    ForwardGraphSignature bucketedPrefillSignature(
        DeviceId device,
        int seq_len,
        uint64_t moe_placement_epoch = 0,
        bool uses_device_sequence_lengths = false)
    {
        ForwardGraphSignature signature;
        signature.seq_len = seq_len;
        signature.batch_size = 1;
        signature.device = device;
        signature.decode = false;
        signature.position_policy = ForwardPositionPolicy::ContiguousOffset;
        signature.standard_path = true;
        signature.pp_stage_enabled = false;
        signature.pp_first_layer = -1;
        signature.pp_last_layer = -1;
        signature.pp_has_embedding = false;
        signature.pp_has_lm_head = false;
        signature.is_bucketed_prefill = true;
        signature.bucket_seq_len = seq_len;
        signature.uses_device_sequence_lengths =
            uses_device_sequence_lengths;
        signature.moe_placement_epoch = moe_placement_epoch;
        return signature;
    }

    PrefillGraphCacheKey prefillGraphKey(
        DeviceId device,
        int seq_len,
        const std::string &domain_id = "single",
        int participant_id = 0,
        uint64_t placement_epoch = 0,
        uint64_t topology_signature = 0)
    {
        PrefillGraphCacheKey key;
        key.seq_len = seq_len;
        key.device_id = device;
        key.domain_id = domain_id;
        key.participant_id = participant_id;
        key.placement_epoch = placement_epoch;
        key.topology_signature = topology_signature;
        return key;
    }

    std::vector<int> makeSequentialInts(int count, int base)
    {
        std::vector<int> values(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            values[static_cast<size_t>(i)] = base + i;
        return values;
    }

    /// @brief Build absolute position IDs for a raw server-style execute() input.
    std::vector<int> makeSequentialPositions(int count, int offset)
    {
        return makeSequentialInts(count, offset);
    }

    /// @brief Return the deterministic residual-add result for a flattened bucket index.
    float expectedProbeValueAtIndex(size_t index)
    {
        const float input = 1.0f + static_cast<float>(index % 17) * 0.125f;
        const float residual = 0.25f + static_cast<float>(index % 13) * 0.0625f;
        return input + residual;
    }

    /// @brief Verify the full bucket residual-add output for exact-bucket tests.
    void expectProbeOutputMatches(PrefillGraphCacheTestHost &host, int seq_len)
    {
        auto *output = host.outputTensor();
        ASSERT_NE(output, nullptr);
        const float *data = output->data();
        ASSERT_NE(data, nullptr);

        const size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(kHiddenDim);
        for (size_t i = 0; i < std::min<size_t>(count, 256); ++i)
        {
            EXPECT_NEAR(data[i], expectedProbeValueAtIndex(i), 1e-5f) << "Mismatch at output index " << i;
        }
    }

    /// @brief Verify that row-select copied the last real bucket row into the one-row output.
    void expectSelectedProbeOutputMatches(PrefillGraphCacheTestHost &host, int real_seq_len)
    {
        ASSERT_GT(real_seq_len, 0);
        auto *output = host.outputTensor();
        ASSERT_NE(output, nullptr);
        const float *data = output->data();
        ASSERT_NE(data, nullptr);

        const size_t source_offset = static_cast<size_t>(real_seq_len - 1) * static_cast<size_t>(kHiddenDim);
        for (size_t col = 0; col < static_cast<size_t>(kHiddenDim); ++col)
        {
            EXPECT_NEAR(data[col], expectedProbeValueAtIndex(source_offset + col), 1e-5f)
                << "Mismatch at selected-row column " << col << " for real_seq_len=" << real_seq_len;
        }
    }

    /**
     * @brief Capture a stable prefix of the current probe output tensor.
     *
     * Prefill graph replay leaves tensors device-authoritative; FP32Tensor::data()
     * is the fixture boundary that synchronizes the small output back to host so
     * tests can compare graph launches without adding bespoke backend copies.
     */
    std::vector<float> captureProbeOutputPrefix(
        PrefillGraphCacheTestHost &host,
        size_t max_elements)
    {
        auto *output = host.outputTensor();
        EXPECT_NE(output, nullptr);
        if (!output)
            return {};
        const float *data = output->data();
        EXPECT_NE(data, nullptr);
        if (!data)
            return {};

        const size_t count = std::min(
            max_elements,
            static_cast<size_t>(kExactBucketSeqLen) * static_cast<size_t>(kHiddenDim));
        return std::vector<float>(data, data + count);
    }

    /// @brief Return the maximum absolute elementwise difference between two vectors.
    double maxAbsDiff(const std::vector<float> &lhs, const std::vector<float> &rhs)
    {
        const size_t count = std::min(lhs.size(), rhs.size());
        double max_abs = 0.0;
        for (size_t i = 0; i < count; ++i)
            max_abs = std::max(max_abs, std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
        if (lhs.size() != rhs.size())
            max_abs = std::numeric_limits<double>::infinity();
        return max_abs;
    }

    class PrefillGraphCacheExecutionTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ensurePrefillGraphCacheBackendRegistered();
            if (!hasPrefillGraphCacheBackendSupport())
                GTEST_SKIP() << prefillGraphCacheBackendSkipMessage();

            device_ = prefillGraphCacheBackendDeviceId();
            device_ctx_ = IDeviceContext::create(device_, 1);
            ASSERT_NE(device_ctx_, nullptr);

            GraphExecutorConfig executor_config;
            executor_config.enable_validation = false;
            executor_ = std::make_unique<DeviceGraphExecutor>(executor_config);
            graph_arena_.bindExecutor(*executor_);

            ForwardExecutionEngine::Config engine_config;
            engine_config.cache_config.enabled = true;
            engine_config.has_unified_pp = false;
            engine_ = std::make_unique<ForwardExecutionEngine>(std::move(engine_config), *executor_);
            host_ = std::make_unique<PrefillGraphCacheTestHost>(
                device_,
                device_ctx_.get(),
                &graph_arena_);
        }

        void TearDown() override
        {
            host_.reset();
            engine_.reset();
            executor_.reset();
            device_ctx_.reset();
        }

        /**
         * @brief Execute one planned prefill chunk through resident length metadata.
         */
        bool runResidentPrefillChunk(
            const ForwardInput &input,
            const ForwardExecutionEngine::PrefillChunkRuntimePlan &plan,
            ForwardOutput &output)
        {
            if (!host_->admitResidentRequestLength(plan.chunk.real_count))
                return false;
            ForwardInput resident_input = input;
            resident_input.sequence_lengths_device =
                host_->residentRequestLengthDevice();
            return engine_->runPrefillChunk(
                resident_input,
                plan,
                output,
                *host_);
        }

        /**
         * @brief Execute a prefill chunk with all mutable GPU metadata resident.
         *
         * This is the production-equivalent path used by graph replay
         * regressions whose result depends on absolute positions. The host
         * vectors are admission sources only; RoPE receives the persistent
         * device pointer and the graph stream consumes one publication event.
         */
        bool runFullyResidentPrefillChunk(
            const ForwardInput &input,
            const ForwardExecutionEngine::PrefillChunkRuntimePlan &plan,
            ForwardOutput &output)
        {
            const auto &positions = plan.chunk.position_ids;
            if (!host_->admitResidentRequestMetadata(
                    plan.chunk.real_count,
                    positions.data(),
                    static_cast<int>(positions.size())))
            {
                return false;
            }
            ForwardInput resident_input = input;
            resident_input.position_ids_device =
                host_->residentPositionIdsDevice();
            resident_input.sequence_lengths_device =
                host_->residentRequestLengthDevice();
            return engine_->runPrefillChunk(
                resident_input,
                plan,
                output,
                *host_);
        }

        /**
         * @brief Execute one raw server-style prefill through resident metadata.
         */
        bool executeResidentPrefill(
            const ForwardInput &input,
            ForwardOutput &output)
        {
            const int real_seq_len =
                input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
            if (!host_->admitResidentRequestLength(real_seq_len))
                return false;
            ForwardInput resident_input = input;
            resident_input.sequence_lengths_device =
                host_->residentRequestLengthDevice();
            return engine_->execute(
                resident_input,
                output,
                *host_);
        }

        test::GraphArenaTestHarness graph_arena_;
        DeviceId device_ = DeviceId::cpu();
        std::unique_ptr<IDeviceContext> device_ctx_;
        std::unique_ptr<DeviceGraphExecutor> executor_;
        std::unique_ptr<ForwardExecutionEngine> engine_;
        std::unique_ptr<PrefillGraphCacheTestHost> host_;
    };

    TEST_F(PrefillGraphCacheExecutionTest, ExactBucketWarmupCaptureReplayLifecycle)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens = makeSequentialInts(kExactBucketSeqLen, 1000);
        ForwardInput base_input;
        base_input.token_ids = tokens.data();
        base_input.batch_size = 1;
        base_input.seq_len = kExactBucketSeqLen;
        base_input.device = device_;

        const auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            base_input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan) << plan.error;
        ASSERT_FALSE(plan.padding_required);
        ASSERT_EQ(plan.chunk.bucket_seq_len, kExactBucketSeqLen);

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(
            device_,
            kExactBucketSeqLen,
            host_->placement_epoch);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(base_input, plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->last_build_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_build_real_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_build_bucket_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_workspace_seq_len, kExactBucketSeqLen);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_build = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_build.has_value());
        EXPECT_TRUE(after_build->forward_cache_valid);
        ASSERT_TRUE(after_build->prefill_cache_initialized);
        EXPECT_EQ(after_build->phase, PrefillGraphPhase::Warmup)
            << "The first bucketed request should build the reusable forward graph and arm prefill capture.";
        EXPECT_EQ(after_build->warmup_count, 1u);
        EXPECT_EQ(after_build->capture_count, 0u);
        EXPECT_EQ(after_build->replay_count, 0);

        ASSERT_TRUE(engine_->runPrefillChunk(base_input, plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_capture = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_capture.has_value());
        EXPECT_EQ(after_capture->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_capture->warmup_count, 1u);
        EXPECT_EQ(after_capture->capture_count, 1u);
        EXPECT_EQ(after_capture->replay_count, 1)
            << "Capture path launches the newly instantiated graph once so this request produces output.";
        EXPECT_GT(after_capture->node_count, 0u)
            << "The probe stage must record real GPU graph nodes, not an empty/mock capture.";

        ASSERT_TRUE(engine_->runPrefillChunk(base_input, plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_replay.has_value());
        EXPECT_EQ(after_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_replay->warmup_count, 1u);
        EXPECT_EQ(after_replay->capture_count, 1u);
        EXPECT_EQ(after_replay->replay_count, 2);
        EXPECT_EQ(after_replay->eviction_count, 0u);

        ASSERT_NE(host_->stage(), nullptr);
        EXPECT_GE(host_->stage()->executeCount(), 2)
            << "Normal build/warmup and capture recording execute the stage directly.";
    }

    TEST_F(PrefillGraphCacheExecutionTest, SessionResetDropsCapturedPrefillExecutable)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens = makeSequentialInts(kExactBucketSeqLen, 1100);
        ForwardInput input;
        input.token_ids = tokens.data();
        input.batch_size = 1;
        input.seq_len = kExactBucketSeqLen;
        input.device = device_;

        const auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan) << plan.error;

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(
            device_,
            kExactBucketSeqLen,
            host_->placement_epoch);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));

        auto ready = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(ready.has_value());
        ASSERT_TRUE(ready->prefill_cache_initialized);
        ASSERT_EQ(ready->phase, PrefillGraphPhase::Ready);
        const int replay_count_before_reset = ready->replay_count;
        EXPECT_GE(replay_count_before_reset, 1);

        engine_->resetSessionReplayState();

        auto after_reset = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_reset.has_value());
        ASSERT_TRUE(after_reset->prefill_cache_initialized);
        EXPECT_EQ(after_reset->phase, PrefillGraphPhase::Cold)
            << "Request/session reset clears live KV/GDN/short-conv state, so "
               "the previous request's monolithic prefill executable must be "
               "dropped before the next request begins.";
        EXPECT_EQ(after_reset->replay_count, 0);
        EXPECT_EQ(after_reset->capture_count, ready->capture_count);

        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_warmup = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_warmup.has_value());
        EXPECT_EQ(after_warmup->phase, PrefillGraphPhase::Warmup)
            << "The first request after session reset executes normally and "
               "arms a fresh capture against the newly cleared live state.";
        EXPECT_EQ(after_warmup->capture_count, ready->capture_count);
        EXPECT_EQ(after_warmup->replay_count, 0);
        EXPECT_EQ(after_warmup->warmup_count, ready->warmup_count + 1);
    }

    TEST_F(PrefillGraphCacheExecutionTest, RequestResetPreservesReadyPrefillExecutable)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens = makeSequentialInts(kExactBucketSeqLen, 1150);
        ForwardInput input;
        input.token_ids = tokens.data();
        input.batch_size = 1;
        input.seq_len = kExactBucketSeqLen;
        input.device = device_;

        const auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan) << plan.error;

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(
            device_,
            kExactBucketSeqLen,
            host_->placement_epoch);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));

        auto ready = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(ready.has_value());
        ASSERT_EQ(ready->phase, PrefillGraphPhase::Ready);
        ASSERT_GE(ready->replay_count, 1);
        const int replay_count_before_reset = ready->replay_count;
        const uint64_t warmups_before_reset = ready->warmup_count;
        const uint64_t captures_before_reset = ready->capture_count;
        const size_t nodes_before_reset = ready->node_count;
        ASSERT_GT(nodes_before_reset, 0u);

        engine_->resetSessionReplayState(
            /*preserve_replay_safe_graphs=*/true);

        auto preserved = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(preserved.has_value());
        ASSERT_TRUE(preserved->prefill_cache_initialized);
        EXPECT_EQ(preserved->phase, PrefillGraphPhase::Ready)
            << "Request-boundary reset should preserve a proven Ready exact-bucket prefill executable.";
        EXPECT_EQ(preserved->replay_count, replay_count_before_reset);
        EXPECT_EQ(preserved->warmup_count, warmups_before_reset);
        EXPECT_EQ(preserved->capture_count, captures_before_reset);
        EXPECT_EQ(preserved->node_count, nodes_before_reset);

        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto after_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_replay.has_value());
        EXPECT_EQ(after_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_replay->replay_count, replay_count_before_reset + 1)
            << "The first request after reset should replay the preserved executable, not warm/capture again.";
        EXPECT_EQ(after_replay->warmup_count, warmups_before_reset);
        EXPECT_EQ(after_replay->capture_count, captures_before_reset);
        EXPECT_EQ(host_->build_calls, 1)
            << "Preserved prefill replay must not rebuild the forward graph.";
    }

    TEST_F(PrefillGraphCacheExecutionTest, RequestResetDemotesWarmupAndCapturesFromLazyInitialization)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens = makeSequentialInts(kExactBucketSeqLen, 1150);
        ForwardInput input;
        input.token_ids = tokens.data();
        input.batch_size = 1;
        input.seq_len = kExactBucketSeqLen;
        input.device = device_;

        const auto plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan) << plan.error;

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen, host_->placement_epoch);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto warmed = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(warmed.has_value());
        ASSERT_TRUE(warmed->prefill_cache_initialized);
        ASSERT_EQ(warmed->phase, PrefillGraphPhase::Warmup);
        ASSERT_EQ(warmed->warmup_count, 1u);
        ASSERT_EQ(warmed->capture_count, 0u);

        engine_->resetSessionReplayState(
            /*preserve_replay_safe_graphs=*/true);

        auto initialized = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(initialized.has_value());
        EXPECT_EQ(initialized->phase, PrefillGraphPhase::Initialized)
            << "clear_cache() must not carry a request-armed Warmup entry across the boundary.";
        EXPECT_EQ(initialized->initialized_count, 1u);
        EXPECT_EQ(initialized->warmup_count, 1u);
        EXPECT_EQ(initialized->capture_count, 0u);
        EXPECT_EQ(initialized->replay_count, 0);

        ASSERT_TRUE(engine_->runPrefillChunk(input, plan, output, *host_));
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        auto captured = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(captured.has_value());
        EXPECT_EQ(captured->phase, PrefillGraphPhase::Ready)
            << "Initialized preserves lazy stage/kernel readiness so the next request can capture "
               "after fresh metadata preparation and strict capture-readiness preflight.";
        EXPECT_EQ(captured->warmup_count, 1u)
            << "The second request should not need another normal warmup when lazy init survived reset.";
        EXPECT_EQ(captured->initialized_count, 1u);
        EXPECT_EQ(captured->capture_count, 1u);
        EXPECT_EQ(captured->replay_count, 1)
            << "Capture path launches the newly instantiated graph once for the current request.";
        EXPECT_EQ(host_->build_calls, 1)
            << "Initialized capture should reuse the cached forward graph topology.";
    }

    TEST_F(PrefillGraphCacheExecutionTest, ChunkScheduleFixedPlacementReachesCapturedReplay)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        host_->domain_id = "overlay_fixed_rocm_hot";
        host_->participant_id = 4;
        host_->placement_epoch = 7;
        host_->topology_signature = 0x4400u;

        const int chunk_count = 4;
        auto tokens = makeSequentialInts(chunk_count * kExactBucketSeqLen, 9000);
        ForwardInput input;
        input.token_ids = tokens.data();
        input.batch_size = 1;
        input.seq_len = static_cast<int>(tokens.size());
        input.real_seq_len = static_cast<int>(tokens.size());
        input.device = device_;

        PrefillChunkSchedulerPolicy policy;
        policy.bucket_sizes = debugEnv().execution.prefill_graph_bucket_sizes;
        policy.fixed_chunk_real_tokens = kExactBucketSeqLen;
        policy.real_token_count = static_cast<int>(tokens.size());

        auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
            input,
            policy,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(schedule) << schedule.error;
        ASSERT_EQ(schedule.chunks.size(), static_cast<size_t>(chunk_count));

        ForwardOutput output;
        ASSERT_TRUE(engine_->runPrefillChunkSchedule(input, schedule, output, *host_));

        EXPECT_EQ(host_->build_calls, 1)
            << "Fixed placement should reuse one bucketed forward graph across chunks.";
        EXPECT_EQ(host_->maintenance_calls, 0);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen, host_->placement_epoch);
        const auto key = prefillGraphKey(
            device_,
            kExactBucketSeqLen,
            host_->domain_id,
            host_->participant_id,
            host_->placement_epoch,
            host_->topology_signature);
        auto snapshot = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(snapshot.has_value());
        EXPECT_EQ(snapshot->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(snapshot->warmup_count, 1u);
        EXPECT_EQ(snapshot->capture_count, 1u);
        EXPECT_EQ(snapshot->replay_count, 3)
            << "Chunk 1 launches after capture, then chunks 2 and 3 replay the captured graph.";
        EXPECT_TRUE(snapshot->observation_valid);
        EXPECT_EQ(snapshot->chunk_index, 3);
        EXPECT_EQ(snapshot->real_token_start, 3 * kExactBucketSeqLen);
        EXPECT_EQ(snapshot->real_token_count, kExactBucketSeqLen);
        EXPECT_EQ(snapshot->domain_id, "overlay_fixed_rocm_hot");
        EXPECT_EQ(snapshot->participant_id, 4);
        EXPECT_EQ(snapshot->placement_epoch, 7u);
        EXPECT_EQ(snapshot->topology_signature, 0x4400u);
        EXPECT_EQ(snapshot->capture_phase, "replay");
        EXPECT_EQ(snapshot->recapture_reason, "none");
    }

    /**
     * @brief Prove 256K-context chunk totality through one captured GPU graph.
     *
     * A 256K request contains 64 complete 4096-row prefill buckets. The extra
     * token deliberately creates a sixty-fifth padded bucket, exercising the
     * two boundaries most likely to hide an off-by-one or padded-row advance:
     * exactly 256K and the first row beyond it. Every chunk publishes its real
     * row count through resident device metadata, and the production KV append
     * stage must advance by those real rows while the same graph executable is
     * captured once and replayed for the entire logical request. The fixture
     * deliberately retains the production 256-row raw-prompt graph minimum:
     * the one-row tail is scheduler-admitted into the transaction bucket and
     * must not be mistaken for an independent short prompt.
     */
    TEST_F(PrefillGraphCacheExecutionTest, LongContext256KPlusTailIsMTotal)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "4096"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "256"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
            {"LLAMINAR_PERF_STATS_JSON", "1"},
        });
        PerfStatsCollector::reset();

        host_->setUseKVAppendProbe(true);
        ASSERT_TRUE(host_->setKVCacheCapacityForTesting(kLongContextSeqLen));
        ASSERT_TRUE(host_->initializeKVCacheProbeForTesting());

        IBackend *backend = getBackendFor(device_);
        ASSERT_NE(backend, nullptr);
        void *admission_stream = nullptr;
        if (device_.is_cuda())
        {
            admission_stream = GPUDeviceContextPool::instance()
                                   .getNvidiaContext(
                                       device_.toKernelDeviceIndex())
                                   .defaultStream();
        }
        else if (device_.is_rocm())
        {
            admission_stream = GPUDeviceContextPool::instance()
                                   .getAMDContext(
                                       device_.toKernelDeviceIndex())
                                   .defaultStream();
        }
        ASSERT_NE(admission_stream, nullptr);

        std::vector<int32_t> request_tokens(
            static_cast<size_t>(kLongContextSeqLen));
        std::vector<int32_t> request_positions(
            static_cast<size_t>(kLongContextSeqLen));
        for (int row = 0; row < kLongContextSeqLen; ++row)
        {
            request_tokens[static_cast<size_t>(row)] = 17000 + row;
            request_positions[static_cast<size_t>(row)] = row;
        }
        /*
         * Bind every captured address through the same semantic arena slots as
         * production.  The harness owns these tensors until teardown, while
         * discardAllCachedGraphs() below releases executable references before
         * that ownership ends.
         */
        auto *request_token_bank =
            graph_arena_.createPersistentTensor<INT32Tensor>(
                BufferId::REQUEST_TOKEN_IDS,
                std::vector<size_t>{
                    static_cast<size_t>(kLongContextSeqLen)},
                request_tokens);
        auto *request_position_bank =
            graph_arena_.createPersistentTensor<INT32Tensor>(
                BufferId::REQUEST_POSITION_IDS,
                std::vector<size_t>{
                    static_cast<size_t>(kLongContextSeqLen)},
                request_positions);
        auto *request_total_rows =
            graph_arena_.createPersistentTensor<INT32Tensor>(
                BufferId::REQUEST_BATCH_GEOMETRY,
                std::vector<size_t>{1},
                std::vector<int32_t>{kLongContextSeqLen});
        auto *chunk_token_bank =
            graph_arena_.createPersistentTensor<INT32Tensor>(
                BufferId::PREFILL_CHUNK_TOKEN_IDS,
                std::vector<size_t>{
                    static_cast<size_t>(kLargeBucketSeqLen)});
        auto *chunk_position_bank =
            graph_arena_.createPersistentTensor<INT32Tensor>(
                BufferId::PREFILL_CHUNK_POSITION_IDS,
                std::vector<size_t>{
                    static_cast<size_t>(kLargeBucketSeqLen)});
        auto *chunk_geometry =
            graph_arena_.createPersistentTensor<INT32Tensor>(
                BufferId::PREFILL_CHUNK_GEOMETRY,
                std::vector<size_t>{2});
        ASSERT_TRUE(request_token_bank->ensureOnDevice(
            device_, admission_stream));
        ASSERT_TRUE(request_position_bank->ensureOnDevice(
            device_, admission_stream));
        ASSERT_TRUE(request_total_rows->ensureOnDevice(
            device_, admission_stream));
        ASSERT_TRUE(chunk_token_bank->allocateOnDevice(
            device_, admission_stream));
        ASSERT_TRUE(chunk_position_bank->allocateOnDevice(
            device_, admission_stream));
        ASSERT_TRUE(chunk_geometry->allocateOnDevice(
            device_, admission_stream));
        ASSERT_TRUE(backend->synchronizeStream(
            admission_stream,
            device_.toKernelDeviceIndex()));

        const int32_t *cached_tokens_device =
            host_->kvSequenceCachedTokensDeviceForTesting();
        ASSERT_NE(cached_tokens_device, nullptr);
        constexpr uint64_t kChunkCaptureIdentity = UINT64_C(0x25600001);
        ForwardInput input;
        input.token_ids = nullptr;
        input.token_ids_device = chunk_token_bank->gpu_data_ptr();
        input.position_ids = nullptr;
        input.position_ids_device = chunk_position_bank->gpu_data_ptr();
        input.position_policy = ForwardPositionPolicy::ExplicitRows;
        input.batch_size = 1;
        input.seq_len = kLongContextSeqLen;
        input.real_seq_len = kLongContextSeqLen;
        input.sequence_lengths_device =
            static_cast<const int32_t *>(chunk_geometry->gpu_data_ptr());
        input.device = device_;
        input.device_prefill_chunk = DevicePrefillChunkGraphBinding{
            .backend = backend,
            .request_token_ids_device = static_cast<const int32_t *>(
                request_token_bank->gpu_data_ptr()),
            .request_position_ids_device = static_cast<const int32_t *>(
                request_position_bank->gpu_data_ptr()),
            .request_total_rows_device = static_cast<const int32_t *>(
                request_total_rows->gpu_data_ptr()),
            .cached_tokens_device = cached_tokens_device,
            .chunk_token_ids_device = static_cast<int32_t *>(
                chunk_token_bank->gpu_data_ptr()),
            .chunk_position_ids_device = static_cast<int32_t *>(
                chunk_position_bank->gpu_data_ptr()),
            .chunk_real_rows_device = static_cast<int32_t *>(
                chunk_geometry->gpu_data_ptr()),
            .chunk_row_stride_device = static_cast<int32_t *>(
                                           chunk_geometry->gpu_data_ptr()) +
                                       1,
            .request_row_capacity = kLongContextSeqLen,
            .bucket_seq_len = kLargeBucketSeqLen,
            .pad_token_id = kPadTokenId,
            .capture_identity = kChunkCaptureIdentity,
        };
        ASSERT_TRUE(input.device_prefill_chunk->valid());

        PrefillChunkSchedulerPolicy policy;
        policy.bucket_sizes = {kLargeBucketSeqLen};
        policy.fixed_chunk_real_tokens = kLargeBucketSeqLen;
        policy.real_token_count = kLongContextSeqLen;

        auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
            input,
            policy,
            kPadTokenId,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(schedule) << schedule.error;
        ASSERT_EQ(schedule.chunks.size(), 65u);

        int expected_offset = 0;
        uint64_t real_rows = 0;
        for (size_t index = 0; index < schedule.chunks.size(); ++index)
        {
            const auto &plan = schedule.chunks[index];
            SCOPED_TRACE(index);
            EXPECT_EQ(plan.chunk_index, static_cast<int>(index));
            EXPECT_EQ(plan.chunk.token_offset, expected_offset);
            EXPECT_EQ(plan.chunk.bucket_seq_len, kLargeBucketSeqLen);
            const int expected_real_rows =
                index + 1 == schedule.chunks.size()
                    ? 1
                    : kLargeBucketSeqLen;
            EXPECT_EQ(plan.chunk.real_count, expected_real_rows);
            expected_offset += plan.chunk.real_count;
            real_rows += static_cast<uint64_t>(plan.chunk.real_count);
        }
        EXPECT_EQ(expected_offset, kLongContextSeqLen);
        EXPECT_EQ(real_rows, static_cast<uint64_t>(kLongContextSeqLen));

        ForwardOutput output;
        ASSERT_TRUE(engine_->runPrefillChunkSchedule(
            input,
            schedule,
            output,
            *host_));

        EXPECT_EQ(host_->committed_forward_output_calls, 65)
            << "Every chunk, including the one-row terminal tail, must publish "
               "through the same successful-forward boundary used by sampling.";
        EXPECT_EQ(host_->last_committed_logits, output.logits);
        EXPECT_TRUE(host_->last_committed_execution.valid);
        EXPECT_EQ(host_->last_committed_execution.graph_seq_len,
                  kLargeBucketSeqLen);

        ASSERT_TRUE(output.execution.valid);
        ASSERT_EQ(output.execution.device, device_);
        ASSERT_NE(output.execution.stream, nullptr);

        /*
         * Cached replay is deliberately asynchronous. Terminal diagnostics
         * therefore consume its exact producer edge before reading any KV or
         * chunk state on the fixture's observation stream. A profiler can
         * stretch the final replay enough to expose this ordering requirement;
         * relying on ordinary launch timing made the old test intermittently
         * observe 262144 rows before the one-row tail append completed.
         */
        const int device_ordinal = device_.toKernelDeviceIndex();
        void *const raw_completion_event =
            backend->createEvent(device_ordinal);
        ASSERT_NE(raw_completion_event, nullptr);
        const std::shared_ptr<void> completion_event(
            raw_completion_event,
            [backend, device_ordinal](void *event)
            {
                backend->destroyEvent(event, device_ordinal);
            });
        ASSERT_TRUE(backend->recordEvent(
            completion_event.get(),
            device_ordinal,
            output.execution.stream));
        ASSERT_TRUE(backend->streamWaitEvent(
            admission_stream,
            completion_event.get(),
            device_ordinal));

        EXPECT_EQ(host_->build_calls, 1)
            << "Every 4096-row chunk must share one stable graph topology.";
        EXPECT_EQ(host_->maintenance_calls, 0);
        EXPECT_EQ(
            host_->kvDeviceCachedTokensForTesting(admission_stream),
            kLongContextSeqLen)
            << "The padded one-row tail must advance KV by one, not 4096.";

        const auto signature = bucketedPrefillSignature(
            device_,
            kLargeBucketSeqLen,
            host_->placement_epoch,
            /*uses_device_sequence_lengths=*/true);
        auto device_chunk_signature = signature;
        device_chunk_signature.uses_device_token_ids = true;
        device_chunk_signature.uses_device_position_ids = true;
        device_chunk_signature.position_policy =
            ForwardPositionPolicy::ExplicitRows;
        device_chunk_signature.device_prefill_chunk_capture_identity =
            kChunkCaptureIdentity;
        const auto key = prefillGraphKey(
            device_,
            kLargeBucketSeqLen,
            host_->domain_id,
            host_->participant_id,
            host_->placement_epoch,
            host_->topology_signature);
        const auto snapshot = engine_->prefillGraphCacheSnapshot(
            device_chunk_signature,
            key);
        ASSERT_TRUE(snapshot.has_value());
        EXPECT_EQ(snapshot->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(snapshot->warmup_count, 1u);
        EXPECT_EQ(snapshot->capture_count, 1u);
        EXPECT_EQ(snapshot->replay_count, 64);
        EXPECT_GE(snapshot->node_count, 2u)
            << "The complete residual-plus-KV probe must be one non-empty graph.";
        EXPECT_EQ(snapshot->chunk_index, 64);
        EXPECT_EQ(snapshot->real_token_start, 256 * 1024);
        EXPECT_EQ(snapshot->real_token_count, 1);
        EXPECT_EQ(snapshot->real_token_end, kLongContextSeqLen);
        EXPECT_EQ(snapshot->capture_phase, "replay");
        EXPECT_EQ(snapshot->recapture_reason, "none");

        const auto records = PerfStatsCollector::snapshot({"forward_graph"});
        const PerfStatsCollector::Tags terminal_replay_tags = {
            {"bucket_seq_len", std::to_string(kLargeBucketSeqLen)},
            {"cache_phase", "ready"},
            {"capture_phase", "replay"},
            {"chunk_index", "64"},
            {"domain_id", "single"},
            {"participant_id", "0"},
            {"placement_epoch", "0"},
            {"real_token_count", "1"},
            {"real_token_end", std::to_string(kLongContextSeqLen)},
            {"real_token_start", std::to_string(256 * 1024)},
            {"recapture_reason", "none"},
            {"topology_signature", "0"}};
        EXPECT_DOUBLE_EQ(
            findPrefillGraphLifecycleCounter(records, terminal_replay_tags),
            1.0);

        std::array<int32_t, 4> terminal_tokens{};
        std::array<int32_t, 4> terminal_positions{};
        std::array<int32_t, 2> terminal_geometry{};
        ASSERT_TRUE(backend->deviceToHostOnStream(
            terminal_tokens.data(),
            chunk_token_bank->gpu_data_ptr(),
            terminal_tokens.size() * sizeof(int32_t),
            device_.toKernelDeviceIndex(),
            admission_stream));
        ASSERT_TRUE(backend->deviceToHostOnStream(
            terminal_positions.data(),
            chunk_position_bank->gpu_data_ptr(),
            terminal_positions.size() * sizeof(int32_t),
            device_.toKernelDeviceIndex(),
            admission_stream));
        ASSERT_TRUE(backend->deviceToHostOnStream(
            terminal_geometry.data(),
            chunk_geometry->gpu_data_ptr(),
            terminal_geometry.size() * sizeof(int32_t),
            device_.toKernelDeviceIndex(),
            admission_stream));
        ASSERT_TRUE(backend->synchronizeStream(
            admission_stream,
            device_.toKernelDeviceIndex()));
        EXPECT_EQ(terminal_geometry[0], 1);
        EXPECT_EQ(terminal_geometry[1], kLargeBucketSeqLen);
        EXPECT_EQ(terminal_tokens[0], request_tokens.back());
        EXPECT_EQ(terminal_tokens[1], kPadTokenId);
        EXPECT_EQ(terminal_tokens[2], kPadTokenId);
        EXPECT_EQ(terminal_tokens[3], kPadTokenId);
        EXPECT_EQ(terminal_positions[0], kLongContextSeqLen - 1);
        EXPECT_EQ(terminal_positions[1], kLongContextSeqLen);
        EXPECT_EQ(terminal_positions[2], kLongContextSeqLen + 1);
        EXPECT_EQ(terminal_positions[3], kLongContextSeqLen + 2);

        engine_->discardAllCachedGraphs();
        PerfStatsCollector::reset();
    }

    TEST_F(PrefillGraphCacheExecutionTest, ChunkScheduleForcedRebalanceRecapturesNewPlacementEpoch)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        host_->domain_id = "overlay_forced_rebalance";
        host_->participant_id = 5;
        host_->placement_epoch = 11;
        host_->topology_signature = 0x5500u;
        host_->setPlacementChangingMaintenance(
            engine_.get(),
            /*bump_epoch=*/true,
            /*topology_delta=*/0x10u);

        const int chunk_count = 9;
        auto tokens = makeSequentialInts(chunk_count * kExactBucketSeqLen, 11000);
        ForwardInput input;
        input.token_ids = tokens.data();
        input.batch_size = 1;
        input.seq_len = static_cast<int>(tokens.size());
        input.real_seq_len = static_cast<int>(tokens.size());
        input.device = device_;

        PrefillChunkSchedulerPolicy policy;
        policy.bucket_sizes = debugEnv().execution.prefill_graph_bucket_sizes;
        policy.fixed_chunk_real_tokens = kExactBucketSeqLen;
        policy.max_rebalance_interval_tokens = 5 * kExactBucketSeqLen;
        policy.real_token_count = static_cast<int>(tokens.size());

        auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
            input,
            policy,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(schedule) << schedule.error;
        ASSERT_EQ(schedule.chunks.size(), static_cast<size_t>(chunk_count));
        ASSERT_TRUE(schedule.chunks[4].rebalance_required_after);
        for (size_t i = 0; i < schedule.chunks.size(); ++i)
        {
            if (i != 4)
                EXPECT_FALSE(schedule.chunks[i].rebalance_required_after) << "chunk=" << i;
        }

        ForwardOutput output;
        ASSERT_TRUE(engine_->runPrefillChunkSchedule(input, schedule, output, *host_));

        EXPECT_EQ(host_->build_calls, 2)
            << "The forced placement boundary should clear the old bucket graph and rebuild once.";
        EXPECT_EQ(host_->maintenance_calls, 1);
        EXPECT_EQ(host_->last_maintenance_chunk.chunk_index, 4);
        EXPECT_TRUE(host_->last_maintenance_decision.required);
        EXPECT_EQ(host_->placement_epoch, 12u);
        EXPECT_EQ(host_->topology_signature, 0x5510u);
        expectProbeOutputMatches(*host_, kExactBucketSeqLen);

        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen, host_->placement_epoch);
        const auto new_key = prefillGraphKey(
            device_,
            kExactBucketSeqLen,
            host_->domain_id,
            host_->participant_id,
            host_->placement_epoch,
            host_->topology_signature);
        auto snapshot = engine_->prefillGraphCacheSnapshot(signature, new_key);
        ASSERT_TRUE(snapshot.has_value());
        EXPECT_EQ(snapshot->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(snapshot->warmup_count, 1u);
        EXPECT_EQ(snapshot->capture_count, 1u);
        EXPECT_EQ(snapshot->replay_count, 3)
            << "The post-rebalance graph should capture on chunk 6 and replay on chunks 7 and 8.";
        EXPECT_TRUE(snapshot->observation_valid);
        EXPECT_EQ(snapshot->chunk_index, 8);
        EXPECT_EQ(snapshot->real_token_start, 8 * kExactBucketSeqLen);
        EXPECT_EQ(snapshot->real_token_count, kExactBucketSeqLen);
        EXPECT_EQ(snapshot->domain_id, "overlay_forced_rebalance");
        EXPECT_EQ(snapshot->participant_id, 5);
        EXPECT_EQ(snapshot->placement_epoch, 12u);
        EXPECT_EQ(snapshot->topology_signature, 0x5510u);
        EXPECT_EQ(snapshot->capture_phase, "replay");
        EXPECT_EQ(snapshot->recapture_reason, "none");
    }

    TEST_F(PrefillGraphCacheExecutionTest, PaddedSafeBucketReplaysAcrossDifferentRealLengths)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
            {"LLAMINAR_PERF_STATS_JSON", "1"},
        });
        PerfStatsCollector::reset();

        host_->setUseRowSelectProbe(true);
        host_->setUseKVAppendProbe(true);
        host_->domain_id = "overlay_routed_rocm_hot";
        host_->participant_id = 2;
        host_->placement_epoch = 17;
        host_->topology_signature = 0x321u;

        auto tokens61 = makeSequentialInts(kExactBucketSeqLen - 3, 3000);
        ForwardInput input61;
        input61.token_ids = tokens61.data();
        input61.batch_size = 1;
        input61.seq_len = kExactBucketSeqLen - 3;
        input61.token_offset = 128;
        input61.device = device_;

        auto tokens63 = makeSequentialInts(kExactBucketSeqLen - 1, 4000);
        ForwardInput input63;
        input63.token_ids = tokens63.data();
        input63.batch_size = 1;
        input63.seq_len = kExactBucketSeqLen - 1;
        input63.token_offset = 512;
        input63.device = device_;

        const auto plan61 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input61,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(plan61) << plan61.error;
        ASSERT_TRUE(plan61.padding_required);
        ASSERT_EQ(plan61.chunk.bucket_seq_len, kExactBucketSeqLen);

        const auto plan63 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input63,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(plan63) << plan63.error;
        ASSERT_TRUE(plan63.padding_required);
        ASSERT_EQ(plan63.chunk.bucket_seq_len, kExactBucketSeqLen);

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(
            device_,
            kExactBucketSeqLen,
            host_->placement_epoch,
            /*uses_device_sequence_lengths=*/true);
        const auto key = prefillGraphKey(
            device_,
            kExactBucketSeqLen,
            host_->domain_id,
            host_->participant_id,
            host_->placement_epoch,
            host_->topology_signature);

        ASSERT_TRUE(runResidentPrefillChunk(input61, plan61, output));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->last_build_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_build_real_seq_len, kExactBucketSeqLen - 3);
        EXPECT_EQ(host_->last_build_bucket_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(host_->last_workspace_seq_len, kExactBucketSeqLen);
        ASSERT_NE(host_->rowSelectStage(), nullptr);
        ASSERT_NE(host_->kvAppendStage(), nullptr);
        EXPECT_EQ(
            host_->rowSelectStage()->selectionPolicyForTesting(),
            HiddenStateRowSelectStage::SelectionPolicy::
                DeviceResidentRequestLength);
        EXPECT_EQ(
            host_->rowSelectStage()->requestSequenceLengthDeviceForTesting(),
            host_->residentRequestLengthDevice());
        EXPECT_EQ(host_->kvCachedTokensForTesting(), kExactBucketSeqLen - 3)
            << "First padded cache miss must append only real prompt tokens.";
        EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(), host_->kvCachedTokensForTesting())
            << "Warmup must keep the GPU KV count mirror aligned to the real prompt length.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 3);

        auto after_build = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_build.has_value());
        EXPECT_TRUE(after_build->forward_cache_valid);
        ASSERT_TRUE(after_build->prefill_cache_initialized);
        EXPECT_EQ(after_build->phase, PrefillGraphPhase::Warmup);
        EXPECT_EQ(after_build->warmup_count, 1u);
        EXPECT_EQ(after_build->capture_count, 0u);
        EXPECT_TRUE(after_build->observation_valid);
        EXPECT_EQ(after_build->chunk_index, 0);
        EXPECT_EQ(after_build->bucket_seq_len, kExactBucketSeqLen);
        EXPECT_EQ(after_build->real_token_start, 128);
        EXPECT_EQ(after_build->real_token_count, kExactBucketSeqLen - 3);
        EXPECT_EQ(after_build->real_token_end, 128 + kExactBucketSeqLen - 3);
        EXPECT_EQ(after_build->domain_id, "overlay_routed_rocm_hot");
        EXPECT_EQ(after_build->participant_id, 2);
        EXPECT_EQ(after_build->placement_epoch, 17u);
        EXPECT_EQ(after_build->topology_signature, 0x321u);
        EXPECT_EQ(after_build->capture_phase, "warmup");
        EXPECT_EQ(after_build->recapture_reason, "none");

        ASSERT_TRUE(runResidentPrefillChunk(input63, plan63, output));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->kvCachedTokensForTesting(), (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1))
            << "Capture launch callback must advance KV metadata by real tokens only.";
        EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(), host_->kvCachedTokensForTesting())
            << "Capture launch must advance the GPU KV count mirror by real tokens, not bucket tokens.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 1);

        auto after_capture = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_capture.has_value());
        EXPECT_EQ(after_capture->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_capture->warmup_count, 1u);
        EXPECT_EQ(after_capture->capture_count, 1u);
        EXPECT_EQ(after_capture->replay_count, 1);
        EXPECT_GT(after_capture->node_count, 0u);
        EXPECT_TRUE(after_capture->observation_valid);
        EXPECT_EQ(after_capture->real_token_start, 512);
        EXPECT_EQ(after_capture->real_token_count, kExactBucketSeqLen - 1);
        EXPECT_EQ(after_capture->real_token_end, 512 + kExactBucketSeqLen - 1);
        EXPECT_EQ(after_capture->capture_phase, "capture");
        EXPECT_EQ(after_capture->recapture_reason, "armed_warmup");

        ASSERT_TRUE(runResidentPrefillChunk(input61, plan61, output));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->kvCachedTokensForTesting(),
                  (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1) + (kExactBucketSeqLen - 3))
            << "Ready replay must advance KV metadata by the latest real length, not the bucket length.";
        EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(), host_->kvCachedTokensForTesting())
            << "Ready replay must keep GPU KV count mirror aligned to the latest real length.";

        auto after_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_replay.has_value());
        EXPECT_EQ(after_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_replay->warmup_count, 1u);
        EXPECT_EQ(after_replay->capture_count, 1u)
            << "Changing real_seq_len inside one bucket must update replay params, not recapture.";
        EXPECT_EQ(after_replay->replay_count, 2);
        EXPECT_EQ(after_replay->eviction_count, 0u);
        EXPECT_TRUE(after_replay->observation_valid);
        EXPECT_EQ(after_replay->real_token_start, 128);
        EXPECT_EQ(after_replay->real_token_count, kExactBucketSeqLen - 3);
        EXPECT_EQ(after_replay->real_token_end, 128 + kExactBucketSeqLen - 3);
        EXPECT_EQ(after_replay->domain_id, "overlay_routed_rocm_hot");
        EXPECT_EQ(after_replay->participant_id, 2);
        EXPECT_EQ(after_replay->placement_epoch, 17u);
        EXPECT_EQ(after_replay->topology_signature, 0x321u);
        EXPECT_EQ(after_replay->capture_phase, "replay");
        EXPECT_EQ(after_replay->recapture_reason, "none");

        ASSERT_TRUE(runResidentPrefillChunk(input63, plan63, output));
        EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(), host_->kvCachedTokensForTesting())
            << "Second Ready replay must keep GPU KV count mirror aligned.";
        auto after_second_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_second_replay.has_value());
        EXPECT_EQ(after_second_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_second_replay->replay_count, 3);
        EXPECT_EQ(after_second_replay->real_token_start, 512);
        EXPECT_EQ(after_second_replay->real_token_count, kExactBucketSeqLen - 1);
        EXPECT_EQ(after_second_replay->real_token_end, 512 + kExactBucketSeqLen - 1);

        const auto records = PerfStatsCollector::snapshot({"forward_graph"});
        const PerfStatsCollector::Tags replay_tags = {
            {"bucket_seq_len", std::to_string(kExactBucketSeqLen)},
            {"cache_phase", "ready"},
            {"capture_phase", "replay"},
            {"chunk_index", "0"},
            {"domain_id", "overlay_routed_rocm_hot"},
            {"participant_id", "2"},
            {"placement_epoch", "17"},
            {"real_token_count", std::to_string(kExactBucketSeqLen - 1)},
            {"real_token_end", std::to_string(512 + kExactBucketSeqLen - 1)},
            {"real_token_start", "512"},
            {"recapture_reason", "none"},
            {"topology_signature", std::to_string(0x321u)}};
        EXPECT_DOUBLE_EQ(findPrefillGraphLifecycleCounter(records, replay_tags), 1.0);
        PerfStatsCollector::reset();
    }

    TEST_F(PrefillGraphCacheExecutionTest, RoPEPositionRowsRefreshAcrossPaddedBucketReplay)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        host_->setUseRoPEProbe(true);

        auto tokens61 = makeSequentialInts(kExactBucketSeqLen - 3, 7000);
        ForwardInput input61;
        input61.token_ids = tokens61.data();
        input61.batch_size = 1;
        input61.seq_len = kExactBucketSeqLen - 3;
        input61.token_offset = 128;
        input61.position_offset = 128;
        input61.device = device_;

        auto tokens63 = makeSequentialInts(kExactBucketSeqLen - 1, 8000);
        ForwardInput input63;
        input63.token_ids = tokens63.data();
        input63.batch_size = 1;
        input63.seq_len = kExactBucketSeqLen - 1;
        input63.token_offset = 512;
        input63.position_offset = 512;
        input63.device = device_;

        const auto plan61 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input61,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(plan61) << plan61.error;
        ASSERT_TRUE(plan61.padding_required);

        const auto plan63 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input63,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/true);
        ASSERT_TRUE(plan63) << plan63.error;
        ASSERT_TRUE(plan63.padding_required);

        ForwardOutput output;
        ASSERT_TRUE(runFullyResidentPrefillChunk(input61, plan61, output));
        ASSERT_NE(host_->ropeStage(), nullptr);
        const auto output_pos128_warmup = captureProbeOutputPrefix(*host_, 512);
        ASSERT_FALSE(output_pos128_warmup.empty());

        ASSERT_TRUE(runFullyResidentPrefillChunk(input63, plan63, output));
        const auto output_pos512_capture = captureProbeOutputPrefix(*host_, 512);
        ASSERT_FALSE(output_pos512_capture.empty());
        EXPECT_GT(maxAbsDiff(output_pos128_warmup, output_pos512_capture), 1.0e-3)
            << "RoPE probe positions 128..191 and 512..575 must produce distinct output, "
               "otherwise this regression cannot prove metadata freshness.";

        ASSERT_TRUE(runFullyResidentPrefillChunk(input61, plan61, output));
        const auto output_pos128_replay = captureProbeOutputPrefix(*host_, 512);
        ASSERT_FALSE(output_pos128_replay.empty());
        const double warmup_replay_diff =
            maxAbsDiff(output_pos128_warmup, output_pos128_replay);
        const double capture_replay_diff =
            maxAbsDiff(output_pos512_capture, output_pos128_replay);
        EXPECT_LT(warmup_replay_diff, 1.0e-5)
            << "Ready replay must refresh RoPE's workspace position rows back to the current "
               "chunk instead of reusing the rows captured for the previous real length. "
            << "warmup_replay_diff=" << warmup_replay_diff
            << " capture_replay_diff=" << capture_replay_diff;
    }

    TEST_F(PrefillGraphCacheExecutionTest, ServerStyleShortRawExecuteUsesMinimumBucketAndReusesAcrossRealLengths)
    {
        constexpr int kMinimumPhysicalBucket = 256;
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64,256"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "256"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        host_->setUseRowSelectProbe(true);
        host_->setUseKVAppendProbe(true);

        auto tokens61 = makeSequentialInts(kExactBucketSeqLen - 3, 5000);
        auto positions61 = makeSequentialPositions(kExactBucketSeqLen - 3, 128);
        ForwardInput input61;
        input61.token_ids = tokens61.data();
        input61.position_ids = positions61.data();
        input61.batch_size = 1;
        input61.seq_len = kExactBucketSeqLen - 3;
        input61.position_offset = 128;
        input61.device = device_;

        auto tokens63 = makeSequentialInts(kExactBucketSeqLen - 1, 6000);
        auto positions63 = makeSequentialPositions(kExactBucketSeqLen - 1, 512);
        ForwardInput input63;
        input63.token_ids = tokens63.data();
        input63.position_ids = positions63.data();
        input63.batch_size = 1;
        input63.seq_len = kExactBucketSeqLen - 1;
        input63.position_offset = 512;
        input63.device = device_;

        ForwardOutput output;
        const auto signature = bucketedPrefillSignature(
            device_,
            kMinimumPhysicalBucket,
            /*moe_placement_epoch=*/0,
            /*uses_device_sequence_lengths=*/true);
        const auto key = prefillGraphKey(device_, kMinimumPhysicalBucket);

        ASSERT_TRUE(executeResidentPrefill(input61, output));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->last_build_seq_len, kMinimumPhysicalBucket);
        EXPECT_EQ(host_->last_build_real_seq_len, kExactBucketSeqLen - 3);
        EXPECT_EQ(host_->last_build_bucket_seq_len, kMinimumPhysicalBucket);
        ASSERT_NE(host_->rowSelectStage(), nullptr);
        ASSERT_NE(host_->kvAppendStage(), nullptr);
        EXPECT_EQ(
            host_->rowSelectStage()->selectionPolicyForTesting(),
            HiddenStateRowSelectStage::SelectionPolicy::
                DeviceResidentRequestLength);
        EXPECT_EQ(
            host_->rowSelectStage()->requestSequenceLengthDeviceForTesting(),
            host_->residentRequestLengthDevice());
        EXPECT_EQ(host_->kvCachedTokensForTesting(), kExactBucketSeqLen - 3)
            << "Raw execute cache miss must append only real prompt tokens.";
        EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(), host_->kvCachedTokensForTesting())
            << "Raw warmup must keep GPU KV count mirror aligned.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 3);

        auto after_build = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_build.has_value());
        EXPECT_TRUE(after_build->forward_cache_valid);
        ASSERT_TRUE(after_build->prefill_cache_initialized);
        EXPECT_EQ(after_build->phase, PrefillGraphPhase::Warmup);
        EXPECT_EQ(after_build->warmup_count, 1u);
        EXPECT_EQ(after_build->capture_count, 0u);

        ASSERT_TRUE(executeResidentPrefill(input63, output));
        EXPECT_EQ(host_->build_calls, 1)
            << "Server-style prompts in one bucket must reuse the cached forward graph.";
        EXPECT_EQ(host_->kvCachedTokensForTesting(), (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1))
            << "Capture launch must append by the second request's real length, not the bucket length.";
        EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(), host_->kvCachedTokensForTesting())
            << "Raw capture launch must advance GPU KV count mirror by real length.";
        expectSelectedProbeOutputMatches(*host_, kExactBucketSeqLen - 1);

        auto after_capture = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_capture.has_value());
        EXPECT_EQ(after_capture->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_capture->warmup_count, 1u);
        EXPECT_EQ(after_capture->capture_count, 1u);
        EXPECT_EQ(after_capture->replay_count, 1);
        EXPECT_GT(after_capture->node_count, 0u);

        ASSERT_TRUE(executeResidentPrefill(input61, output));
        EXPECT_EQ(host_->build_calls, 1);
        EXPECT_EQ(host_->kvCachedTokensForTesting(),
                  (kExactBucketSeqLen - 3) + (kExactBucketSeqLen - 1) + (kExactBucketSeqLen - 3))
            << "Ready replay must advance KV metadata by the latest real length, not the bucket length.";
        EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(), host_->kvCachedTokensForTesting())
            << "Raw Ready replay must keep GPU KV count mirror aligned.";

        auto after_replay = engine_->prefillGraphCacheSnapshot(signature, key);
        ASSERT_TRUE(after_replay.has_value());
        EXPECT_EQ(after_replay->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_replay->warmup_count, 1u);
        EXPECT_EQ(after_replay->capture_count, 1u)
            << "Changing raw real_seq_len inside one bucket must update replay params, not recapture.";
        EXPECT_EQ(after_replay->replay_count, 2);
        EXPECT_EQ(after_replay->eviction_count, 0u);
    }

    TEST_F(PrefillGraphCacheExecutionTest, CrossBucketEvictionRecapturesEligibleBucket)
    {
        constexpr int kEvictionBucketSeqLen = 128;
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64,128"},
            {"LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_PREFILL_GRAPH_TRACE", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens64 = makeSequentialInts(kExactBucketSeqLen, 7000);
        ForwardInput input64;
        input64.token_ids = tokens64.data();
        input64.batch_size = 1;
        input64.seq_len = kExactBucketSeqLen;
        input64.device = device_;

        auto tokens128 = makeSequentialInts(kEvictionBucketSeqLen, 8000);
        ForwardInput input128;
        input128.token_ids = tokens128.data();
        input128.batch_size = 1;
        input128.seq_len = kEvictionBucketSeqLen;
        input128.device = device_;

        const auto plan64 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input64,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan64) << plan64.error;
        ASSERT_EQ(plan64.chunk.bucket_seq_len, kExactBucketSeqLen);

        const auto plan128 = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            input128,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_TRUE(plan128) << plan128.error;
        ASSERT_EQ(plan128.chunk.bucket_seq_len, kEvictionBucketSeqLen);

        ForwardOutput output;
        const auto signature64 = bucketedPrefillSignature(device_, kExactBucketSeqLen);
        const auto key64 = prefillGraphKey(device_, kExactBucketSeqLen);
        const auto signature128 = bucketedPrefillSignature(device_, kEvictionBucketSeqLen);
        const auto key128 = prefillGraphKey(device_, kEvictionBucketSeqLen);

        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        EXPECT_EQ(host_->build_calls, 1);

        auto after_64_ready = engine_->prefillGraphCacheSnapshot(signature64, key64);
        ASSERT_TRUE(after_64_ready.has_value());
        EXPECT_EQ(after_64_ready->phase, PrefillGraphPhase::Ready);
        EXPECT_EQ(after_64_ready->warmup_count, 1u);
        EXPECT_EQ(after_64_ready->capture_count, 1u);
        EXPECT_EQ(after_64_ready->eviction_count, 0u);

        ASSERT_TRUE(engine_->runPrefillChunk(input128, plan128, output, *host_));
        EXPECT_EQ(host_->build_calls, 2)
            << "The larger bucket should build once, then evict the older reusable bucket.";
        EXPECT_FALSE(engine_->prefillGraphCacheSnapshot(signature64, key64).has_value())
            << "A max-buckets=1 cap must evict the old top-level bucketed forward graph.";

        auto after_128_build = engine_->prefillGraphCacheSnapshot(signature128, key128);
        ASSERT_TRUE(after_128_build.has_value());
        EXPECT_TRUE(after_128_build->forward_cache_valid);
        ASSERT_TRUE(after_128_build->prefill_cache_initialized);
        EXPECT_EQ(after_128_build->phase, PrefillGraphPhase::Warmup)
            << "The first request for a newly built bucket should arm capture immediately.";
        EXPECT_EQ(after_128_build->warmup_count, 1u);
        EXPECT_EQ(after_128_build->capture_count, 0u);
        EXPECT_EQ(after_128_build->eviction_count, 1u);

        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        EXPECT_EQ(host_->build_calls, 3)
            << "Requesting the evicted bucket must rebuild its forward graph.";

        auto after_64_rebuild = engine_->prefillGraphCacheSnapshot(signature64, key64);
        ASSERT_TRUE(after_64_rebuild.has_value());
        EXPECT_TRUE(after_64_rebuild->forward_cache_valid);
        ASSERT_TRUE(after_64_rebuild->prefill_cache_initialized);
        EXPECT_EQ(after_64_rebuild->phase, PrefillGraphPhase::Warmup)
            << "The first request after eviction rebuilds the forward graph and arms capture.";
        EXPECT_EQ(after_64_rebuild->warmup_count, 1u);
        EXPECT_EQ(after_64_rebuild->capture_count, 0u);
        EXPECT_EQ(after_64_rebuild->eviction_count, 2u)
            << "Rebuilding bucket64 under cap=1 should evict bucket128 at the top-level cache.";

        ASSERT_TRUE(engine_->runPrefillChunk(input64, plan64, output, *host_));
        auto after_64_recapture = engine_->prefillGraphCacheSnapshot(signature64, key64);
        ASSERT_TRUE(after_64_recapture.has_value());
        EXPECT_EQ(after_64_recapture->phase, PrefillGraphPhase::Ready)
            << "The rebuilt bucket must capture again instead of staying on normal prefill.";
        EXPECT_EQ(after_64_recapture->warmup_count, 1u);
        EXPECT_EQ(after_64_recapture->capture_count, 1u);
        EXPECT_EQ(after_64_recapture->replay_count, 1);
        EXPECT_EQ(after_64_recapture->eviction_count, 2u);
        EXPECT_GT(after_64_recapture->node_count, 0u);
    }

    TEST_F(PrefillGraphCacheExecutionTest, NonExactBucketRejectedBeforeGraphBuild)
    {
        ScopedDebugEnv env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
            {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64"},
            {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
            {"LLAMINAR_VALIDATE_BUFFERS", "0"},
            {"LLAMINAR_VALIDATE_INPUTS", "0"},
            {"LLAMINAR_FAIL_ON_ZERO", "0"},
        });

        auto tokens = makeSequentialInts(kExactBucketSeqLen - 1, 2000);
        ForwardInput base_input;
        base_input.token_ids = tokens.data();
        base_input.batch_size = 1;
        base_input.seq_len = kExactBucketSeqLen - 1;
        base_input.device = device_;

        const auto padded_plan = ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
            base_input,
            debugEnv().execution.prefill_graph_bucket_sizes,
            kPadTokenId,
            /*allow_padded_execution=*/false);
        ASSERT_FALSE(padded_plan);
        EXPECT_TRUE(padded_plan.padding_required);
        EXPECT_NE(padded_plan.error.find("requires caller opt-in"), std::string::npos);

        ForwardOutput output;
        EXPECT_FALSE(engine_->runPrefillChunk(base_input, padded_plan, output, *host_));
        EXPECT_EQ(host_->build_calls, 0)
            << "Non-exact bucketed prefill without padded opt-in must reject before graph build.";

        const auto signature = bucketedPrefillSignature(device_, kExactBucketSeqLen);
        const auto key = prefillGraphKey(device_, kExactBucketSeqLen);
        EXPECT_FALSE(engine_->prefillGraphCacheSnapshot(signature, key).has_value());
    }

} // namespace
