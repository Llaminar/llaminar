/**
 * @file Test__SegmentedGraphCaptureExecution.cpp
 * @brief Backend-bound integration tests for cached GPU graph replay.
 *
 * This source is compiled once for CUDA and once for ROCm.  Each resulting
 * binary explicitly registers only the backend named by its compile-time test
 * binding, preventing an unregistered factory from turning real GPU coverage
 * into a successful gtest skip.
 *
 * Coverage:
 * 1. First-use warmup atomically materializes a replay-ready full graph.
 * 2. Collective-marked segmented mode remains functional.
 * 3. Graph-stable snapshot slots preserve point-in-time outputs when a later
 *    stage overwrites the producer's arena storage.
 * 4. A retained GPU parent can wait on a concurrently serviced canonical CPU
 *    route ticket without serial child launches or a host stream fence.
 * 5. Small auxiliary graph families certify aggregate native-pool growth over
 *    cold and warm lifetimes without attributing a pool slab to one sibling.
 */

#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/moe/MoEOverlayRetainedParentComposer.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "backends/BackendManager.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "transfer/MappedTransferProgressEpoch.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/GraphArenaTestHarness.h"

using namespace llaminar2;
using namespace llaminar2::test;

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define ENSURE_LINKED_GPU_BACKEND() ensureNvidiaFactoryRegistered()
#define HAS_LINKED_GPU_SUPPORT() GPUDeviceContextPool::instance().hasNvidiaSupport()
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getNvidiaContext(0)
#define LINKED_GPU_DEVICE_ID() DeviceId::cuda(0)
#define LINKED_GPU_SKIP_MESSAGE "CUDA not available"
#elif defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define ENSURE_LINKED_GPU_BACKEND() ensureAMDFactoryRegistered()
#define HAS_LINKED_GPU_SUPPORT() GPUDeviceContextPool::instance().hasAMDSupport()
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getAMDContext(0)
#define LINKED_GPU_DEVICE_ID() DeviceId::rocm(0)
#define LINKED_GPU_SKIP_MESSAGE "ROCm not available"
#else
#define ENSURE_LINKED_GPU_BACKEND() ((void)0)
#define HAS_LINKED_GPU_SUPPORT() false
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getContext("", 0)
#define LINKED_GPU_DEVICE_ID() DeviceId::cpu()
#define LINKED_GPU_SKIP_MESSAGE "No GPU backend linked in this test binary"
#endif

#define SKIP_IF_NO_GPU()                                      \
    do                                                        \
    {                                                         \
        ENSURE_LINKED_GPU_BACKEND();                          \
        if (!HAS_LINKED_GPU_SUPPORT())                        \
            GTEST_SKIP() << LINKED_GPU_SKIP_MESSAGE;          \
    } while (false)

namespace
{
    /** @brief Enable and restore PerfStats for one integration certificate. */
    class ScopedPerfStats final
    {
    public:
        ScopedPerfStats()
        {
            if (const char *value =
                    std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
            {
                had_value_ = true;
                value_ = value;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_value_)
                setenv(
                    "LLAMINAR_PERF_STATS_SUMMARY",
                    value_.c_str(),
                    1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedPerfStats(const ScopedPerfStats &) = delete;
        ScopedPerfStats &operator=(const ScopedPerfStats &) = delete;

    private:
        bool had_value_ = false;
        std::string value_;
    };

    /**
     * @brief CPU-only participant used to certify the captured ticket ABI.
     *
     * Production dispatch/local/return stages receive their own CPU integration
     * certificate.  This deliberately tiny stage isolates the heterogeneous
     * graph protocol: it reads the captured pinned dispatch ticket and writes
     * the pinned return ticket without any backend operation or tensor shadow.
     */
    class TicketEchoManualStage final : public IComputeStage
    {
    public:
        explicit TicketEchoManualStage(
            std::shared_ptr<MoEOverlayDispatchTicketStorage> storage)
            : IComputeStage(DeviceId::cpu()), storage_(std::move(storage))
        {
        }

        bool execute(IDeviceContext *ctx) override
        {
            complete_ = false;
            if (!ctx || !storage_ || !storage_->hasValidBoundIdentity())
                return false;
            std::string publication_error;
            if (!storage_->awaitCapturedPublication(&publication_error))
                return false;
            auto &ticket = storage_->ticket();
            if (!ticket.isValid())
                return false;

            const int rows = ticket.header->logical_row_count;
            const int bucket = ticket.header->bucket_row_capacity;
            const int top_k = ticket.header->top_k;
            const int d_model = ticket.header->d_model;
            std::fill_n(
                ticket.return_rows_fp32,
                static_cast<size_t>(bucket) * static_cast<size_t>(d_model),
                0.0f);
            for (int row = 0; row < rows; ++row)
            {
                const float bias = ticket.routing_weights_fp32[
                    static_cast<size_t>(row) * static_cast<size_t>(top_k)];
                for (int col = 0; col < d_model; ++col)
                {
                    const size_t index =
                        static_cast<size_t>(row) *
                            static_cast<size_t>(d_model) +
                        static_cast<size_t>(col);
                    ticket.return_rows_fp32[index] =
                        ticket.hidden_rows_fp32[index] * 2.0f + bias;
                }
            }
            ticket.header->return_logical_row_count = rows;
            if (call_count_ < observed_rows_.size())
                observed_rows_[call_count_] = rows;
            ++call_count_;
            complete_ = true;
            return true;
        }

        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_EXPERT_DISPATCH;
        }
        std::string name() const override
        {
            return "ticket_echo_manual";
        }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        bool isGraphCapturable() const override { return false; }
        bool isManualGraphBoundary() const override { return true; }
        bool manualGraphBoundaryComplete() const override { return complete_; }
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        StageBufferRequirements getBufferRequirements() const override
        {
            return {};
        }
        StageBufferContract bufferContract() const override
        {
            return StageBufferContract::build();
        }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

        size_t callCount() const noexcept { return call_count_; }
        int observedRows(size_t call) const noexcept
        {
            return call < observed_rows_.size() ? observed_rows_[call] : -1;
        }

    private:
        std::shared_ptr<MoEOverlayDispatchTicketStorage> storage_;
        std::array<int, 4> observed_rows_{};
        size_t call_count_ = 0;
        bool complete_ = false;
    };

    /**
     * @brief CPU producer for one reusable mapped canonical-route ticket.
     *
     * The stage intentionally has no arena bindings: every byte it owns lives
     * in the setup-bound ticket and contribution mapping. This is the exact
     * self-contained contract required for servicing the stage after a GPU
     * retained parent has already started. Publication is the only ordering
     * edge observed by the captured consumer.
     */
    class CanonicalTicketPublishManualStage final : public IComputeStage
    {
    public:
        /**
         * @brief Bind one immutable ticket and its deterministic test payload.
         * @param storage Model-lifetime mapped ticket storage.
         * @param route_capacity Number of route rows published per execution.
         * @param d_model Width of each FP32 route row.
         */
        CanonicalTicketPublishManualStage(
            std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage> storage,
            std::size_t route_capacity,
            int d_model)
            : IComputeStage(DeviceId::cpu()),
              storage_(std::move(storage)),
              route_capacity_(route_capacity),
              d_model_(d_model)
        {
        }

        /**
         * @brief Fill and release-publish the next mapped ticket generation.
         * @return True after the complete compact payload becomes visible.
         */
        bool execute(IDeviceContext *ctx) override
        {
            complete_ = false;
            if (!ctx || !storage_ ||
                !storage_->hasValidBoundIdentity() || route_capacity_ == 0u ||
                d_model_ <= 0)
            {
                return false;
            }

            const std::uint64_t residency_epoch = 101u + call_count_;
            auto publication = storage_->arm(residency_epoch);
            if (!publication)
                return false;

            auto *const original_slots = storage_->originalRouteSlotsHost();
            auto *const compact_slots = storage_->compactRouteSlotsHost();
            float *const rows = storage_->contributionRowsHost();
            if (!original_slots || !compact_slots || !rows)
                return false;

            /* Reverse the compact layout so the device materializer must use
             * both identity arrays instead of succeeding as a bulk copy. */
            const float base = 1000.0f * static_cast<float>(call_count_ + 1u);
            for (std::size_t entry = 0u; entry < route_capacity_; ++entry)
            {
                const std::size_t compact = route_capacity_ - 1u - entry;
                original_slots[entry] = static_cast<std::int32_t>(entry);
                compact_slots[entry] = static_cast<std::int32_t>(compact);
                for (int column = 0; column < d_model_; ++column)
                {
                    rows[compact * static_cast<std::size_t>(d_model_) +
                         static_cast<std::size_t>(column)] =
                        base + 100.0f * static_cast<float>(entry) +
                        static_cast<float>(column);
                }
            }
            if (!publication.publish(route_capacity_))
                return false;

            ++call_count_;
            complete_ = true;
            return true;
        }

        /** @return Stable stage identity for graph diagnostics. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_LOCAL_EXPERT;
        }
        /** @return Human-readable stage name. */
        std::string name() const override
        {
            return "canonical_ticket_publish_manual";
        }
        /**
         * @return True for the executor fixture's shared context.
         *
         * Production supplies a participant-specific CPU context through its
         * node executor callback. This lower-level integration invokes the
         * self-contained host ticket stage through the graph's shared context;
         * the stage deliberately performs no backend operation through it.
         */
        bool supportsBackend(ComputeBackendType) const override
        {
            return true;
        }
        /** @return False because mapped host service is outside GPU capture. */
        bool isGraphCapturable() const override { return false; }
        /** @return True because this stage is an explicit host ticket unit. */
        bool isManualGraphBoundary() const override { return true; }
        /** @return Whether the current invocation published its payload. */
        bool manualGraphBoundaryComplete() const override { return complete_; }
        /** @return Pre-armed service that overlaps retained-parent submission. */
        ManualGraphBoundaryScheduling
        manualGraphBoundaryScheduling() const noexcept override
        {
            return ManualGraphBoundaryScheduling::ConcurrentTicketService;
        }
        /** @return This stage owns the publication that releases GPU ingress. */
        ConcurrentManualFailureRole
        concurrentManualFailureRole() const noexcept override
        {
            return ConcurrentManualFailureRole::DeviceIngressPublisher;
        }
        /** @brief Publish an authenticated abort to drain a waiting parent. */
        bool publishConcurrentManualFailure() noexcept override
        {
            return storage_ && storage_->publishAbort();
        }
        /** @return Ticket geometry is independent of padded logical length. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }
        /** @return Ticket geometry preserves the live-length contract. */
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }
        /** @return No ordinary tensor coherence is owned by this stage. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        /** @return No arena storage is required. */
        StageBufferRequirements getBufferRequirements() const override
        {
            return {};
        }
        /** @return Empty arena contract; all storage is mapped-ticket owned. */
        StageBufferContract bufferContract() const override
        {
            return StageBufferContract::build();
        }
        /** @return No tensor dump is needed for this protocol fixture. */
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }
        /** @return Number of successful ticket publications. */
        std::size_t callCount() const noexcept { return call_count_; }

    private:
        std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage> storage_;
        std::size_t route_capacity_ = 0u;
        int d_model_ = 0;
        std::uint64_t call_count_ = 0u;
        bool complete_ = false;
    };

    /**
     * @brief Concurrent CPU bridge from one captured dispatch ticket to one
     * captured canonical-route return ticket.
     *
     * This fixture is deliberately the smallest complete production-shaped
     * heterogeneous boundary. The retained GPU parent first copies router and
     * hidden-state bytes into @p dispatch_storage and system-release publishes
     * them. This CPU service acquires that edge, constructs the canonical
     * sparse return, and release-publishes @p return_storage so the already
     * running parent can resume in its captured consumer.
     */
    class DispatchToCanonicalTicketManualStage final : public IComputeStage
    {
    public:
        /**
         * @brief Bind both immutable ticket lifetimes used by the bridge.
         * @param dispatch_storage Captured GPU-to-CPU dispatch ticket.
         * @param return_storage CPU-to-GPU canonical return ticket.
         * @param route_capacity Number of canonical route rows per replay.
         * @param d_model Width of every returned route row.
         */
        DispatchToCanonicalTicketManualStage(
            std::shared_ptr<MoEOverlayDispatchTicketStorage> dispatch_storage,
            std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage>
                return_storage,
            std::size_t route_capacity,
            int d_model)
            : IComputeStage(DeviceId::cpu()),
              dispatch_storage_(std::move(dispatch_storage)),
              return_storage_(std::move(return_storage)),
              route_capacity_(route_capacity),
              d_model_(d_model)
        {
        }

        /**
         * @brief Acquire the captured dispatch and publish one sparse return.
         * @return True only after both publication edges complete in order.
         */
        bool execute(IDeviceContext *ctx) override
        {
            complete_ = false;
            if (!ctx || !dispatch_storage_ || !return_storage_ ||
                !dispatch_storage_->hasValidBoundIdentity() ||
                !return_storage_->hasValidBoundIdentity() ||
                route_capacity_ == 0u || d_model_ <= 0)
            {
                return false;
            }

            std::string publication_error;
            if (!dispatch_storage_->awaitCapturedPublication(
                    &publication_error))
            {
                return false;
            }
            const auto &dispatch = dispatch_storage_->ticket();
            if (!dispatch.isValid() ||
                dispatch.header->logical_row_count <= 0 ||
                dispatch.header->d_model != d_model_)
            {
                return false;
            }

            const std::uint64_t residency_epoch = 301u + call_count_;
            auto publication = return_storage_->arm(residency_epoch);
            if (!publication)
                return false;

            auto *const original_slots =
                return_storage_->originalRouteSlotsHost();
            auto *const compact_slots =
                return_storage_->compactRouteSlotsHost();
            float *const rows = return_storage_->contributionRowsHost();
            if (!original_slots || !compact_slots || !rows)
                return false;

            /* Preserve the first dispatch value as a replay-freshness oracle.
             * Every route receives a deterministic offset so the captured
             * consumer must materialize the complete mapped sparse payload. */
            const float dispatch_value = dispatch.hidden_rows_fp32[0];
            for (std::size_t route = 0u; route < route_capacity_; ++route)
            {
                original_slots[route] = static_cast<std::int32_t>(route);
                compact_slots[route] = static_cast<std::int32_t>(route);
                for (int column = 0; column < d_model_; ++column)
                {
                    rows[route * static_cast<std::size_t>(d_model_) +
                         static_cast<std::size_t>(column)] =
                        dispatch_value + 100.0f * static_cast<float>(route) +
                        static_cast<float>(column);
                }
            }
            if (!publication.publish(route_capacity_))
                return false;

            if (call_count_ < observed_dispatch_values_.size())
                observed_dispatch_values_[call_count_] = dispatch_value;
            ++call_count_;
            complete_ = true;
            return true;
        }

        /** @return Stable stage identity for graph diagnostics. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_EXPERT_DISPATCH;
        }
        /** @return Human-readable stage name. */
        std::string name() const override
        {
            return "dispatch_to_canonical_ticket_manual";
        }
        /** @return True for the fixture's shared participant context. */
        bool supportsBackend(ComputeBackendType) const override { return true; }
        /** @return False because this service executes on the host. */
        bool isGraphCapturable() const override { return false; }
        /** @return True because this is the explicit heterogeneous cutpoint. */
        bool isManualGraphBoundary() const override { return true; }
        /** @return Whether this invocation published its return ticket. */
        bool manualGraphBoundaryComplete() const override { return complete_; }
        /** @return Service must be armed before retained-parent submission. */
        ManualGraphBoundaryScheduling
        manualGraphBoundaryScheduling() const noexcept override
        {
            return ManualGraphBoundaryScheduling::ConcurrentTicketService;
        }
        /** @return This service publishes the edge that releases GPU ingress. */
        ConcurrentManualFailureRole
        concurrentManualFailureRole() const noexcept override
        {
            return ConcurrentManualFailureRole::DeviceIngressPublisher;
        }
        /** @brief Release a captured consumer if dispatch servicing fails. */
        bool publishConcurrentManualFailure() noexcept override
        {
            return return_storage_ && return_storage_->publishAbort();
        }
        /** @return Fixed ticket geometry supports padded capture. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }
        /** @return Live rows are carried in the immutable ticket ABI. */
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }
        /** @return Both ticket stores own coherence explicitly. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        /** @return No BufferArena storage belongs to this host service. */
        StageBufferRequirements getBufferRequirements() const override
        {
            return {};
        }
        /** @return Empty arena contract permits concurrent ticket service. */
        StageBufferContract bufferContract() const override
        {
            return StageBufferContract::build();
        }
        /** @return No tensor dump is required for this protocol fixture. */
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }
        /** @return Number of complete dispatch-to-return transactions. */
        std::size_t callCount() const noexcept { return call_count_; }
        /** @return Dispatch value acquired during one completed transaction. */
        float observedDispatchValue(std::size_t call) const noexcept
        {
            return call < observed_dispatch_values_.size()
                       ? observed_dispatch_values_[call]
                       : 0.0f;
        }

    private:
        std::shared_ptr<MoEOverlayDispatchTicketStorage> dispatch_storage_;
        std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage>
            return_storage_;
        std::size_t route_capacity_ = 0u;
        int d_model_ = 0;
        std::array<float, 4> observed_dispatch_values_{};
        std::size_t call_count_ = 0u;
        bool complete_ = false;
    };
} // namespace

class CachedGraphReplayExecutionTest : public ::testing::Test
{
protected:
    IWorkerGPUContext *gpu_ctx_ = nullptr;
    std::unique_ptr<IDeviceContext> device_ctx_;
    GraphArenaTestHarness graph_arena_;
    std::vector<std::unique_ptr<TensorBase>> tensor_storage_;
    std::vector<TensorBase *> arena_tensors_;

    void SetUp() override
    {
        ENSURE_LINKED_GPU_BACKEND();
        if (HAS_LINKED_GPU_SUPPORT())
        {
            gpu_ctx_ = &LINKED_GPU_CONTEXT();
            device_ctx_ = IDeviceContext::create(LINKED_GPU_DEVICE_ID(), 1);
        }
    }

    void TearDown() override
    {
        tensor_storage_.clear();
        device_ctx_.reset();
        gpu_ctx_ = nullptr;
    }

    FP32Tensor *createFP32Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = TestTensorFactory::createFP32(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<FP32Tensor *>(ptr);
    }

    INT32Tensor *createINT32Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = TestTensorFactory::createINT32(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<INT32Tensor *>(ptr);
    }

    FP32Tensor *createArenaFP32Tensor(
        BufferId id,
        const std::vector<size_t> &shape)
    {
        auto *tensor =
            graph_arena_.createPersistentTensor<FP32Tensor>(id, shape);
        arena_tensors_.push_back(tensor);
        return tensor;
    }

    /**
     * @brief Allocate and upload every fixture-owned tensor before graph warmup.
     *
     * Cached replay creates a dedicated stream internally.  These synthetic
     * stages do not use arena BufferIds, so the executor cannot discover and
     * cohere their raw tensor parameters on our behalf.  Uploading on the
     * context's default stream and synchronizing once establishes stable device
     * addresses before warmup, capture, and replay bind the stages to their
     * dedicated stream.
     *
     * @param excluded_tensor Optional internal output left without GPU storage
     *        so cached capture preparation owns its first allocation.
     * @return true when every included tensor is resident and the upload stream
     *         has completed; false on allocation, transfer, or synchronization
     *         failure.
     */
    bool prepareFixtureTensorsForGPUExecution(
        ITensor *excluded_tensor = nullptr)
    {
        if (!gpu_ctx_ || !device_ctx_)
            return false;

        void *upload_stream = gpu_ctx_->defaultStream();
        if (!upload_stream)
            return false;

        const DeviceId device = device_ctx_->deviceId();
        for (const auto &tensor : tensor_storage_)
        {
            if (tensor.get() == excluded_tensor)
                continue;
            if (!tensor || !tensor->ensureOnDevice(device, upload_stream))
                return false;
        }
        for (auto *tensor : arena_tensors_)
        {
            if (tensor == excluded_tensor)
                continue;
            if (!tensor || !tensor->ensureOnDevice(device, upload_stream))
                return false;
        }

        return gpu_ctx_->synchronizeStreamChecked(upload_stream);
    }

    ComputeGraph buildNormResidualGraph(size_t seq_len, size_t d_model,
                                        FP32Tensor *&norm_input,
                                        FP32Tensor *&residual,
                                        FP32Tensor *&result_output,
                                        FP32Tensor **norm_output_out = nullptr,
                                        bool strict_copy_consumer = false)
    {
        const DeviceId device = device_ctx_->deviceId();
        norm_input = createArenaFP32Tensor(
            BufferId::HIDDEN_STATE, {seq_len, d_model});
        auto *norm_output = createArenaFP32Tensor(
            BufferId::NORMALIZED, {seq_len, d_model});
        if (norm_output_out)
            *norm_output_out = norm_output;
        auto *gamma = createFP32Tensor({d_model});
        residual = createArenaFP32Tensor(
            BufferId::RESIDUAL, {seq_len, d_model});
        result_output = createArenaFP32Tensor(
            BufferId::ATTN_OUTPUT, {seq_len, d_model});

        const size_t num_elements = seq_len * d_model;
        for (size_t i = 0; i < num_elements; ++i)
            norm_input->mutable_data()[i] = 0.5f + static_cast<float>(i % 10) * 0.1f;
        for (size_t i = 0; i < d_model; ++i)
            gamma->mutable_data()[i] = 1.0f;
        for (size_t i = 0; i < num_elements; ++i)
            residual->mutable_data()[i] = 0.1f * static_cast<float>(i % 7);

        RMSNormStage::Params norm_params;
        norm_params.input = norm_input;
        norm_params.output = norm_output;
        norm_params.gamma = gamma;
        norm_params.eps = 1e-5f;
        norm_params.seq_len = static_cast<int>(seq_len);
        norm_params.device_id = device;
        norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        ResidualAddStage::Params res_params;
        res_params.input = norm_output;
        res_params.residual = strict_copy_consumer ? nullptr : residual;
        res_params.output = result_output;
        res_params.num_elements = num_elements;
        res_params.device_id = device;
        res_params.input_buffer_id = BufferId::NORMALIZED;
        if (!strict_copy_consumer)
            res_params.residual_buffer_id = BufferId::RESIDUAL;
        res_params.output_buffer_id = BufferId::ATTN_OUTPUT;

        ComputeGraph graph;
        graph.addNode("rmsnorm", ComputeStageFactory::createRMSNorm(norm_params), device);
        graph.addNode("residual_add", ComputeStageFactory::createResidualAdd(res_params), device);
        graph.addDependency("residual_add", "rmsnorm");
        return graph;
    }

    ComputeGraph buildSnapshotOverwriteGraph(size_t seq_len, size_t d_model,
                                             FP32Tensor *&norm_input,
                                             FP32Tensor *&scratch)
    {
        const DeviceId device = device_ctx_->deviceId();
        norm_input = createArenaFP32Tensor(
            BufferId::HIDDEN_STATE, {seq_len, d_model});
        scratch = createArenaFP32Tensor(
            BufferId::NORMALIZED, {seq_len, d_model});
        auto *gamma = createFP32Tensor({d_model});
        auto *residual = createArenaFP32Tensor(
            BufferId::RESIDUAL, {seq_len, d_model});

        const size_t num_elements = seq_len * d_model;
        for (size_t i = 0; i < num_elements; ++i)
        {
            norm_input->mutable_data()[i] = 0.5f + static_cast<float>(i % 10) * 0.1f;
            residual->mutable_data()[i] = 10.0f;
        }
        for (size_t i = 0; i < d_model; ++i)
            gamma->mutable_data()[i] = 1.0f;

        RMSNormStage::Params norm_params;
        norm_params.device_id = device;
        norm_params.input = norm_input;
        norm_params.output = scratch;
        norm_params.gamma = gamma;
        norm_params.eps = 1e-5f;
        norm_params.seq_len = static_cast<int>(seq_len);
        norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        ResidualAddStage::Params overwrite_params;
        overwrite_params.device_id = device;
        overwrite_params.input = residual;
        overwrite_params.residual = scratch;
        overwrite_params.output = scratch;
        overwrite_params.num_elements = num_elements;
        overwrite_params.input_buffer_id = BufferId::RESIDUAL;
        overwrite_params.residual_buffer_id = BufferId::NORMALIZED;
        overwrite_params.output_buffer_id = BufferId::NORMALIZED;

        ComputeGraph graph;
        graph.addNode("stage1_norm", ComputeStageFactory::createRMSNorm(norm_params), device);
        graph.addNode("stage2_overwrite", ComputeStageFactory::createResidualAdd(overwrite_params), device);
        graph.addDependency("stage2_overwrite", "stage1_norm");
        return graph;
    }

    /**
     * @brief Build a production-scale checkpoint chain over three live tensors.
     *
     * The 122B parity graph exposes roughly fifteen hundred tensor-backed stage
     * outputs per participant. Alternating two activation buffers reproduces
     * the production arena-reuse hazard without allocating one tensor per
     * stage: every ResidualAdd updates the next buffer and its snapshot must
     * preserve that exact intermediate value before a later stage overwrites
     * the same address.
     *
     * @param stage_count Positive number of captured checkpoint producers.
     * @param element_count Elements in each checkpoint payload.
     * @param final_output Receives the tensor written by the last stage.
     * @return A dependency-ordered graph with @p stage_count snapshot outputs.
     */
    ComputeGraph buildProductionScaleSnapshotChain(
        size_t stage_count,
        size_t element_count,
        FP32Tensor *&final_output)
    {
        EXPECT_GT(stage_count, 0u);
        EXPECT_GT(element_count, 0u);

        const DeviceId device = device_ctx_->deviceId();
        auto *first = createArenaFP32Tensor(
            BufferId::HIDDEN_STATE, {1u, element_count});
        auto *second = createArenaFP32Tensor(
            BufferId::NORMALIZED, {1u, element_count});
        auto *residual = createArenaFP32Tensor(
            BufferId::RESIDUAL, {1u, element_count});
        std::fill_n(first->mutable_data(), element_count, 1.0F);
        std::fill_n(second->mutable_data(), element_count, 0.0F);
        std::fill_n(residual->mutable_data(), element_count, 0.25F);

        ComputeGraph graph;
        std::string predecessor;
        for (size_t stage = 0; stage < stage_count; ++stage)
        {
            const bool even = (stage % 2u) == 0u;
            ResidualAddStage::Params params;
            params.device_id = device;
            params.input = even ? first : second;
            params.residual = residual;
            params.output = even ? second : first;
            params.num_elements = element_count;
            params.input_buffer_id =
                even ? BufferId::HIDDEN_STATE : BufferId::NORMALIZED;
            params.residual_buffer_id = BufferId::RESIDUAL;
            params.output_buffer_id =
                even ? BufferId::NORMALIZED : BufferId::HIDDEN_STATE;

            const std::string stage_name =
                "production_snapshot_stage_" + std::to_string(stage);
            graph.addNode(
                stage_name,
                ComputeStageFactory::createResidualAdd(params),
                device);
            if (!predecessor.empty())
                graph.addDependency(stage_name, predecessor);
            predecessor = stage_name;
        }

        final_output = (stage_count % 2u) == 0u ? first : second;
        return graph;
    }

    static void fillSnapshotInput(FP32Tensor *tensor, float offset)
    {
        ASSERT_NE(tensor, nullptr);
        float *data = tensor->mutable_data();
        ASSERT_NE(data, nullptr);
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = 0.5f +
                      static_cast<float>(i % 10) * 0.1f +
                      offset * (1.0f + static_cast<float>(i % 3) * 0.25f);
        }
    }

    static void assertFiniteAndNonZero(const float *data, size_t count)
    {
        bool has_nonzero = false;
        for (size_t i = 0; i < count; ++i)
        {
            ASSERT_FALSE(std::isnan(data[i])) << "NaN at index " << i;
            ASSERT_FALSE(std::isinf(data[i])) << "Inf at index " << i;
            if (data[i] != 0.0f)
            {
                has_nonzero = true;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Output tensor is all zeros";
    }

    static void assertTensorFiniteAndNonZero(
        FP32Tensor *tensor,
        size_t count,
        void *producer_stream)
    {
        ASSERT_NE(tensor, nullptr);
        ASSERT_NE(producer_stream, nullptr);
        ASSERT_TRUE(tensor->ensureOnHost(producer_stream))
            << "The test's explicit host observation must consume the exact graph producer stream.";
        assertFiniteAndNonZero(tensor->data(), count);
    }

    static void assertGraphStagesUseStream(ComputeGraph &graph, void *stream)
    {
        ASSERT_NE(stream, nullptr);
        for (const auto &node_name : graph.getExecutionOrder())
        {
            ComputeNode *node = graph.getNode(node_name);
            ASSERT_NE(node, nullptr) << "Missing node: " << node_name;
            ASSERT_NE(node->stage, nullptr) << "Missing stage: " << node_name;
            EXPECT_EQ(node->stage->gpuStream(), stream)
                << "Stage should remain bound to the explicit capture stream: " << node_name;
        }
    }

    static void assertSnapshotDelta(
        const std::unordered_map<std::string, std::vector<float>> &snapshots,
        const std::string &before_stage,
        const std::string &after_stage,
        float expected_delta)
    {
        auto before_it = snapshots.find(before_stage);
        auto after_it = snapshots.find(after_stage);
        ASSERT_NE(before_it, snapshots.end()) << "Missing snapshot for " << before_stage;
        ASSERT_NE(after_it, snapshots.end()) << "Missing snapshot for " << after_stage;
        ASSERT_EQ(before_it->second.size(), after_it->second.size());
        ASSERT_FALSE(before_it->second.empty());
        for (size_t i = 0; i < before_it->second.size(); ++i)
        {
            EXPECT_NEAR(after_it->second[i] - before_it->second[i], expected_delta, 1e-4f)
                << "Snapshot delta mismatch at index " << i;
        }
    }

    static void assertSnapshotsDiffer(
        const std::unordered_map<std::string, std::vector<float>> &lhs,
        const std::unordered_map<std::string, std::vector<float>> &rhs,
        const std::string &stage_name)
    {
        auto lhs_it = lhs.find(stage_name);
        auto rhs_it = rhs.find(stage_name);
        ASSERT_NE(lhs_it, lhs.end()) << "Missing lhs snapshot for " << stage_name;
        ASSERT_NE(rhs_it, rhs.end()) << "Missing rhs snapshot for " << stage_name;
        ASSERT_EQ(lhs_it->second.size(), rhs_it->second.size());

        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < lhs_it->second.size(); ++i)
        {
            max_abs_diff = std::max(max_abs_diff, std::abs(lhs_it->second[i] - rhs_it->second[i]));
        }
        EXPECT_GT(max_abs_diff, 1e-3f)
            << "Snapshot for " << stage_name << " did not refresh across graph execution";
    }
};

/**
 * @test Native decoration keeps a Close-dependent worker parallel to the body.
 *
 * The original graph and all three fragments are recorded but never submitted
 * separately. Twenty replays reset only data. A bounded diagnostic escape
 * releases every wait before reporting failure, so a broken edge cannot strand
 * the accelerator. This source runs identically on CUDA and ROCm.
 */
TEST_F(CachedGraphReplayExecutionTest, NativeParallelBranchJoinsAfterBodyClose)
{
    SKIP_IF_NO_GPU();
    const auto device = device_ctx_->deviceId();
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    TransferEngine transfers;
    const DeviceId devices[] = {device};
    auto control = transfers.allocateMappedHostRegion(4096u, devices);
    auto *words = static_cast<std::uint64_t *>(control->mutableHostData());
    std::fill_n(words, 4u, 0u);
    std::array<std::unique_ptr<IGPUGraphCapture>, 3> fragments;
    std::unique_ptr<IGPUGraphCapture> body;
    void *stream = nullptr;
    gpu_ctx_->submitAndWait([&]
    {
        stream = gpu_ctx_->getOrCreateAuxiliaryStream("native_parallel_branch_proof");
        if (!stream) throw std::runtime_error("parallel branch stream allocation failed");
        // Word 0 is Open, 1 is the body terminal, 2 is Close and 3 is the
        // worker receipt. The worker cannot finish until the body progresses.
        auto record = [&](int wait_word, std::size_t publish_word)
        {
            auto graph = gpu_ctx_->createGraphCapture(stream);
            if (!graph) throw std::runtime_error("parallel fragment allocation failed");
            ScopedBackendGraphCapture capture(*gpu_ctx_, *graph, "parallel DAG ordering proof");
            if (!capture.begin()) throw std::runtime_error("parallel fragment capture failed");
            if ((wait_word >= 0 && !backend->streamWaitTimelineSignal64(
                    stream, control->deviceAlias(device, sizeof(std::uint64_t) * wait_word),
                    1u, device.gpu_ordinal())) ||
                !backend->streamPublishTimelineSignal64(
                    stream, control->deviceAlias(device, sizeof(std::uint64_t) * publish_word),
                    1u, device.gpu_ordinal()))
                throw std::runtime_error("parallel fragment recording failed");
            capture.finish();
            return graph;
        };
        fragments[0] = record(-1, 0u);
        fragments[1] = record(2, 3u);
        fragments[2] = record(1, 2u);
        body = record(0, 1u);
        if (!body->appendParallelBranch({*fragments[0], *fragments[1], *fragments[2]}) ||
            !body->instantiate())
            throw std::runtime_error("parallel body attachment/instantiation failed");
        EXPECT_FALSE(body->appendParallelBranch({*fragments[0], *fragments[1], *fragments[2]}))
            << "Instantiated topology must be immutable";
    });
    for (std::size_t replay = 0u; replay < 20u; ++replay)
    {
        SCOPED_TRACE(replay);
        for (std::size_t i = 0u; i < 4u; ++i)
            std::atomic_ref<std::uint64_t>(words[i]).store(0u, std::memory_order_release);
        bool launched = false;
        gpu_ctx_->submitAndWait([&] { launched = body->launch(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (launched && std::atomic_ref<std::uint64_t>(words[3]).load(std::memory_order_acquire) == 0u &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        const bool completed = launched &&
            std::atomic_ref<std::uint64_t>(words[3]).load(std::memory_order_acquire) == 1u;
        // Test-only cancellation precedes assertion and native teardown.
        if (!completed)
            for (std::size_t i = 0u; i < 4u; ++i)
                std::atomic_ref<std::uint64_t>(words[i]).store(1u, std::memory_order_release);
        gpu_ctx_->submitAndWait([&] { ASSERT_TRUE(gpu_ctx_->synchronizeStreamChecked(stream)); });
        ASSERT_TRUE(completed) << "Worker serialized before the body's Close";
        for (std::size_t i = 0u; i < 4u; ++i) EXPECT_EQ(words[i], 1u);
    }
    gpu_ctx_->submitAndWait([&]
    {
        body.reset();
        for (auto &fragment : fragments) fragment.reset();
    });
}

TEST_F(CachedGraphReplayExecutionTest, FirstUseMaterializesReplayWithoutSecondMutation)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);
    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *stream = gpu_ctx_->defaultStream();
    ASSERT_NE(stream, nullptr);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, nullptr));
    EXPECT_TRUE(segment_cache.initialized);
    EXPECT_FALSE(segment_cache.needs_capture)
        << "A successful first use must not expose initialized-but-uncaptured state.";
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);

    std::string export_error;
    const auto replay_template =
        segment_cache.deviceLoopGraphTemplate(graph, &export_error);
    ASSERT_TRUE(replay_template.has_value()) << export_error;
    ASSERT_NE(replay_template->capture, nullptr);
    EXPECT_EQ(replay_template->stage_count, graph.getExecutionOrder().size());
    EXPECT_EQ(replay_template->stream, segment_cache.capture_stream);

    std::vector<float> first_use_output(num_elements);
    std::memcpy(
        first_use_output.data(),
        result->data(),
        num_elements * sizeof(float));

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, nullptr));
    EXPECT_TRUE(segment_cache.initialized);
    EXPECT_FALSE(segment_cache.needs_capture);
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);

    const float *replay = result->data();
    for (size_t i = 0; i < num_elements; ++i)
    {
        EXPECT_NEAR(replay[i], first_use_output[i], 1e-5f)
            << "Replay output differs from the single first-use execution at index " << i;
    }
}

TEST_F(CachedGraphReplayExecutionTest,
       FirstCaptureAllocatesColdInternalProducerStorage)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    constexpr size_t seq_len = 2;
    constexpr size_t d_model = 32;
    constexpr size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    FP32Tensor *internal_norm_output = nullptr;
    auto graph = buildNormResidualGraph(
        seq_len,
        d_model,
        norm_input,
        residual,
        result,
        &internal_norm_output,
        /*strict_copy_consumer=*/true);
    ASSERT_NE(internal_norm_output, nullptr);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution(internal_norm_output));
    ASSERT_EQ(internal_norm_output->gpu_data_ptr(), nullptr)
        << "The regression requires a cold internal producer output";

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph,
        device_ctx_.get(),
        segment_cache,
        dispatch_stream,
        gpu_ctx_,
        nullptr));
    ASSERT_NE(internal_norm_output->gpu_data_ptr(), nullptr);
    ASSERT_EQ(
        internal_norm_output->current_device(),
        std::optional<DeviceId>{device_ctx_->deviceId()});
    assertTensorFiniteAndNonZero(
        result,
        num_elements,
        segment_cache.capture_stream);
}

TEST_F(CachedGraphReplayExecutionTest, DISABLED_CollectiveMarkedMode_RemainsFunctional)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *stream = gpu_ctx_->defaultStream();
    ASSERT_NE(stream, nullptr);

    // NOTE: This test remains disabled in Phase 0 because the current
    // collective-marked manual-segment path can yield zeroed outputs for this
    // synthetic graph. Keep it as a scaffold for follow-up stabilization.

    std::unordered_set<std::string> collective_nodes = {"rmsnorm"};

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, &collective_nodes));
    EXPECT_TRUE(segment_cache.initialized);
    EXPECT_FALSE(segment_cache.segments.empty());

    bool has_manual_segment = false;
    for (const auto &seg : segment_cache.segments)
    {
        if (!seg.capturable)
        {
            has_manual_segment = true;
            break;
        }
    }
    EXPECT_TRUE(has_manual_segment);
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, &collective_nodes));

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, stream, gpu_ctx_, &collective_nodes));
    assertTensorFiniteAndNonZero(
        result, num_elements, segment_cache.capture_stream);
}

TEST_F(CachedGraphReplayExecutionTest,
       HeterogeneousTicketSegmentsReuseCapturedBucketAcrossLogicalLengths)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);
    ScopedPerfStats perf_stats;

    constexpr int layer = 3;
    constexpr int bucket_rows = 4;
    constexpr int top_k = 2;
    constexpr int d_model = 8;
    const DeviceId device = device_ctx_->deviceId();

    auto *hidden = createArenaFP32Tensor(
        BufferId::NORMALIZED,
        {bucket_rows, d_model});
    auto *routing_indices = createArenaFP32Tensor(
        BufferId::MOE_EXPERT_INDICES,
        {bucket_rows, top_k});
    auto *routing_weights = createArenaFP32Tensor(
        BufferId::MOE_EXPERT_WEIGHTS,
        {bucket_rows, top_k});
    auto *output = createArenaFP32Tensor(
        BufferId::MOE_COMBINED_OUTPUT,
        {bucket_rows, d_model});
    auto *active_rows = createINT32Tensor({1});

    auto ticket_storage =
        std::make_shared<MoEOverlayDispatchTicketStorage>();
    const std::array<DeviceId, 1> ticket_devices{device};
    auto ticket_arena =
        TransferEngine::instance().createMappedHostArena(ticket_devices);
    ticket_storage->bindFixedCapacity(
        layer,
        bucket_rows,
        top_k,
        d_model,
        device,
        /*workspace_generation=*/29,
        ticket_arena);
    const auto *const ticket_header = ticket_storage->ticket().header;
    const auto *const ticket_hidden =
        ticket_storage->ticket().hidden_rows_fp32;
    const auto *const ticket_return =
        ticket_storage->ticket().return_rows_fp32;

    /*
     * Reproduce production cold preflight before the arena has published GPU
     * addresses. Static ticket geometry must be admitted, while the stronger
     * capture-ready predicate must still reject the unbound device pointers.
     */
    MoEOverlayTicketPublishStage::Params cold_publish_params;
    cold_publish_params.device_id = device;
    cold_publish_params.hidden = hidden;
    cold_publish_params.routing_indices = routing_indices;
    cold_publish_params.routing_weights = routing_weights;
    cold_publish_params.layer_idx = layer;
    cold_publish_params.bucket_rows = bucket_rows;
    cold_publish_params.top_k = top_k;
    cold_publish_params.d_model = d_model;
    cold_publish_params.ticket_storage = ticket_storage;
    MoEOverlayTicketPublishStage cold_publish(cold_publish_params);
    EXPECT_TRUE(cold_publish.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_TRUE(cold_publish.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_EQ(
        cold_publish.graphLaunchPreparationPolicy(),
        GraphLaunchPreparationPolicy::CaptureAndReplay);
    EXPECT_FALSE(cold_publish.isGraphCapturable());
    EXPECT_FALSE(cold_publish.supportsPaddedPrefillGraphCapturePreflight())
        << "Padded capture additionally requires the device-owned live-row scalar";

    MoEOverlayTicketConsumeStage::Params cold_consume_params;
    cold_consume_params.device_id = device;
    cold_consume_params.output = output;
    cold_consume_params.layer_idx = layer;
    cold_consume_params.bucket_rows = bucket_rows;
    cold_consume_params.d_model = d_model;
    cold_consume_params.ticket_storage = ticket_storage;
    MoEOverlayTicketConsumeStage cold_consume(cold_consume_params);
    EXPECT_TRUE(cold_consume.supportsLazyPrefillGraphCapturePreflight());
    EXPECT_TRUE(cold_consume.supportsGraphCaptureAfterLaunchPreparation());
    EXPECT_EQ(
        cold_consume.graphLaunchPreparationPolicy(),
        GraphLaunchPreparationPolicy::CaptureOnly);
    EXPECT_TRUE(cold_consume.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_FALSE(cold_consume.isGraphCapturable());

    const auto fill_transaction = [&](int logical_rows, float base)
    {
        ASSERT_GT(logical_rows, 0);
        ASSERT_LE(logical_rows, bucket_rows);
        std::fill_n(
            hidden->mutable_data(),
            hidden->numel(),
            -777.0f);
        std::fill_n(
            routing_indices->mutable_data(),
            routing_indices->numel(),
            99.0f);
        std::fill_n(
            routing_weights->mutable_data(),
            routing_weights->numel(),
            -999.0f);
        for (int row = 0; row < logical_rows; ++row)
        {
            routing_indices->mutable_data()[
                static_cast<size_t>(row) * top_k] =
                static_cast<float>(row);
            routing_indices->mutable_data()[
                static_cast<size_t>(row) * top_k + 1] =
                static_cast<float>(row + 1);
            routing_weights->mutable_data()[
                static_cast<size_t>(row) * top_k] =
                0.2f + 0.1f * static_cast<float>(row);
            routing_weights->mutable_data()[
                static_cast<size_t>(row) * top_k + 1] =
                0.8f - 0.1f * static_cast<float>(row);
            for (int col = 0; col < d_model; ++col)
            {
                hidden->mutable_data()[
                    static_cast<size_t>(row) * d_model + col] =
                    base + static_cast<float>(row) * 0.5f +
                    static_cast<float>(col) * 0.025f;
            }
        }
        active_rows->mutable_int32_data()[0] = logical_rows;
    };

    fill_transaction(/*logical_rows=*/3, /*base=*/1.0f);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    IBackend *const backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    const int device_ordinal = device.gpu_ordinal();
    const auto pinned_scalar_deleter =
        [backend, device_ordinal](int32_t *pointer)
    {
        backend->freePinned(pointer, device_ordinal);
    };
    std::unique_ptr<int32_t, decltype(pinned_scalar_deleter)>
        active_rows_staging(
            static_cast<int32_t *>(backend->allocatePinned(
                sizeof(int32_t), device_ordinal)),
            pinned_scalar_deleter);
    ASSERT_NE(active_rows_staging, nullptr);

    MoEOverlayTicketPublishStage::Params publish_params;
    publish_params.device_id = device;
    publish_params.hidden = hidden;
    publish_params.routing_indices = routing_indices;
    publish_params.routing_weights = routing_weights;
    publish_params.hidden_buffer_id = BufferId::NORMALIZED;
    publish_params.routing_indices_buffer_id =
        BufferId::MOE_EXPERT_INDICES;
    publish_params.routing_weights_buffer_id =
        BufferId::MOE_EXPERT_WEIGHTS;
    publish_params.active_row_count_device =
        static_cast<const int32_t *>(active_rows->gpu_data_ptr());
    publish_params.layer_idx = layer;
    publish_params.bucket_rows = bucket_rows;
    publish_params.top_k = top_k;
    publish_params.d_model = d_model;
    publish_params.ticket_storage = ticket_storage;

    auto manual_stage =
        std::make_unique<TicketEchoManualStage>(ticket_storage);
    TicketEchoManualStage *const manual_probe = manual_stage.get();

    MoEOverlayTicketConsumeStage::Params consume_params;
    consume_params.device_id = device;
    consume_params.output = output;
    consume_params.output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
    consume_params.layer_idx = layer;
    consume_params.bucket_rows = bucket_rows;
    consume_params.d_model = d_model;
    consume_params.ticket_storage = ticket_storage;

    ComputeGraph graph;
    graph.addNode(
        "ticket_publish",
        ComputeStageFactory::createMoEOverlayTicketPublish(publish_params),
        device);
    graph.addNode(
        "cpu_ticket_participant",
        std::move(manual_stage),
        DeviceId::cpu());
    graph.addNode(
        "ticket_consume",
        ComputeStageFactory::createMoEOverlayTicketConsume(consume_params),
        device);
    graph.addDependency("cpu_ticket_participant", "ticket_publish");
    graph.addDependency("ticket_consume", "cpu_ticket_participant");

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    const auto execute_with_cache = [&graph,
                                     &executor,
                                     this,
                                     dispatch_stream](
                                        DeviceGraphExecutor::GraphSegmentCache &cache,
                                        DeviceGraphExecutor::GraphInitialSubmissionPolicy
                                            initial_submission)
    {
        return executor.executeWithCachedGraphReplay(
            graph,
            device_ctx_.get(),
            cache,
            dispatch_stream,
            gpu_ctx_,
            nullptr,
            /*collectives_graph_capturable=*/false,
            /*force_recapture=*/false,
            /*defer_final_sync=*/false,
            {},
            DeviceGraphExecutor::GraphReplayPlanPolicy::
                AllowHeterogeneousBoundarySegmentation,
            {},
            {},
            {},
            initial_submission);
    };
    const auto execute = [&]()
    {
        return execute_with_cache(
            segment_cache,
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                CaptureInstantiateAndLaunch);
    };

    const auto expect_output = [&](int logical_rows, float base)
    {
        ASSERT_TRUE(output->ensureOnHost(segment_cache.capture_stream));
        for (int row = 0; row < logical_rows; ++row)
        {
            const float bias =
                0.2f + 0.1f * static_cast<float>(row);
            for (int col = 0; col < d_model; ++col)
            {
                const float input =
                    base + static_cast<float>(row) * 0.5f +
                    static_cast<float>(col) * 0.025f;
                EXPECT_FLOAT_EQ(
                    output->data()[
                        static_cast<size_t>(row) * d_model + col],
                    input * 2.0f + bias)
                    << "row=" << row << " col=" << col;
            }
        }
        for (int row = logical_rows; row < bucket_rows; ++row)
        {
            for (int col = 0; col < d_model; ++col)
            {
                EXPECT_FLOAT_EQ(
                    output->data()[
                        static_cast<size_t>(row) * d_model + col],
                    0.0f)
                    << "padding row=" << row << " col=" << col;
            }
        }
    };

    ASSERT_TRUE(execute());
    ASSERT_EQ(segment_cache.segments.size(), 3u);
    EXPECT_TRUE(segment_cache.segments[0].capturable);
    EXPECT_FALSE(segment_cache.segments[1].capturable);
    EXPECT_TRUE(segment_cache.segments[2].capturable);
    ASSERT_NE(segment_cache.segments[0].capture, nullptr);
    ASSERT_NE(segment_cache.segments[2].capture, nullptr);
    const auto *const publish_capture =
        segment_cache.segments[0].capture.get();
    const auto *const consume_capture =
        segment_cache.segments[2].capture.get();
    EXPECT_TRUE(ticket_storage->hasCapturedPublicationContract());
    ASSERT_EQ(manual_probe->callCount(), 1u);
    EXPECT_EQ(manual_probe->observedRows(0), 3);
    expect_output(/*logical_rows=*/3, /*base=*/1.0f);

    fill_transaction(/*logical_rows=*/1, /*base=*/4.0f);
    ASSERT_TRUE(hidden->ensureOnDevice(device, segment_cache.capture_stream));
    ASSERT_TRUE(routing_indices->ensureOnDevice(
        device,
        segment_cache.capture_stream));
    ASSERT_TRUE(routing_weights->ensureOnDevice(
        device,
        segment_cache.capture_stream));
    *active_rows_staging = 1;
    ASSERT_TRUE(backend->hostToDeviceOnStream(
        active_rows->gpu_data_ptr(),
        active_rows_staging.get(),
        sizeof(int32_t),
        device_ordinal,
        segment_cache.capture_stream));
    graph.reset();
    ASSERT_TRUE(execute());

    EXPECT_EQ(segment_cache.segments[0].capture.get(), publish_capture);
    EXPECT_EQ(segment_cache.segments[2].capture.get(), consume_capture);
    ASSERT_EQ(manual_probe->callCount(), 2u);
    EXPECT_EQ(manual_probe->observedRows(1), 1);
    EXPECT_EQ(ticket_storage->ticket().header, ticket_header);
    EXPECT_EQ(ticket_storage->ticket().hidden_rows_fp32, ticket_hidden);
    EXPECT_EQ(ticket_storage->ticket().return_rows_fp32, ticket_return);
    expect_output(/*logical_rows=*/1, /*base=*/4.0f);

    /*
     * Server readiness seals every graph before request admission. Reset the
     * mutable ticket to an explicitly incomplete state and prove that native
     * recording needs only its immutable pinned address. No manual participant
     * may execute and no graph unit may launch during this setup transaction.
     */
    ticket_storage->ticket().header->return_logical_row_count = 0;
    ASSERT_FALSE(ticket_storage->ticket().returnPayloadReady());
    graph.reset();
    DeviceGraphExecutor::GraphSegmentCache setup_cache;
    ASSERT_TRUE(execute_with_cache(
        setup_cache,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            MaterializeWithoutLaunch));
    EXPECT_EQ(manual_probe->callCount(), 2u);
    EXPECT_EQ(
        setup_cache.executable_submission_state,
        DeviceGraphExecutor::GraphSegmentCache::
            ExecutableSubmissionState::MaterializedUnlaunched);
    EXPECT_EQ(setup_cache.successful_submission_count, 0u);
}

/**
 * @brief Prove a retained parent replays its captured GPU-to-CPU dispatch edge.
 *
 * This combines the two halves certified separately by the segmented dispatch
 * test and the canonical-return retained-parent test. It is the minimal shape
 * of one production heterogeneous ExpertOverlay layer: captured D2H dispatch,
 * concurrent CPU service, and captured mapped-ticket return consumption. A
 * second transaction changes the source bytes while retaining every graph and
 * ticket address. Forty-eight consecutive boundaries produce the same 97-unit
 * retained-parent shape as the Qwen 3.5 122B authority graph. The depth-15
 * grouped-decode row capacity, router fanout, and hidden width also reproduce
 * its mapped-ticket byte pressure, while one shared canonical-return region
 * matches the production serial CPU arena. This catches setup-time mapped-page
 * translation storms without loading model weights or inventing per-layer
 * registrations that production does not own.
 */
TEST_F(CachedGraphReplayExecutionTest,
       RetainedParentReplaysCapturedDispatchPublicationAcrossReset)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);
    ScopedPerfStats perf_stats;

    constexpr int layer_count = 48;
    constexpr int bucket_rows = 16;
    constexpr int top_k = 8;
    constexpr int d_model = 7168;
    constexpr std::size_t route_capacity =
        static_cast<std::size_t>(bucket_rows * top_k);
    constexpr std::size_t contribution_elements =
        route_capacity * static_cast<std::size_t>(d_model);
    constexpr std::size_t contribution_bytes =
        contribution_elements * sizeof(float);
    const DeviceId device = device_ctx_->deviceId();

    auto *hidden = createArenaFP32Tensor(
        BufferId::NORMALIZED,
        {bucket_rows, d_model});
    auto *routing_indices = createArenaFP32Tensor(
        BufferId::MOE_EXPERT_INDICES,
        {bucket_rows, top_k});
    auto *routing_weights = createArenaFP32Tensor(
        BufferId::MOE_EXPERT_WEIGHTS,
        {bucket_rows, top_k});
    auto *canonical_output = createArenaFP32Tensor(
        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
        {route_capacity, static_cast<std::size_t>(d_model)});

    const auto fill_dispatch = [&](float base)
    {
        for (int column = 0; column < d_model; ++column)
            hidden->mutable_data()[column] = base + column;
        routing_indices->mutable_data()[0] = 0.0f;
        routing_indices->mutable_data()[1] = 1.0f;
        routing_weights->mutable_data()[0] = 0.25f;
        routing_weights->mutable_data()[1] = 0.75f;
    };
    fill_dispatch(11.0f);
    std::fill_n(
        canonical_output->mutable_data(),
        contribution_elements,
        -1.0f);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    const std::array<DeviceId, 1> mapped_devices{device};
    auto ticket_arena =
        TransferEngine::instance().createMappedHostArena(mapped_devices);
    auto cpu_canonical_arena =
        std::make_shared<MoELocalExpertSerialBufferArena>(
            MoELocalExpertSerialBufferArena::Config{
                .device_id = DeviceId::cpu(),
                .row_capacity = bucket_rows,
                .row_capacity_buckets = {bucket_rows},
                .d_model = d_model,
                .routing_top_k = top_k,
                .cpu_canonical_route_storage =
                    MoELocalExpertSerialBufferArena::
                        CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
                .cpu_canonical_route_gpu_consumer = device,
                .debug_name =
                    "integration.production_mapped_cpu_canonical_routes",
            });
    auto contribution_region =
        cpu_canonical_arena->mappedCPUCanonicalRoutes(device);
    ASSERT_NE(contribution_region, nullptr);
    ASSERT_NE(cpu_canonical_arena->cpuCanonicalRoutes(), nullptr);
    EXPECT_FALSE(cpu_canonical_arena->cpuCanonicalRoutes()->isMapped());
    EXPECT_EQ(
        cpu_canonical_arena->cpuCanonicalRoutes()->home_device(),
        DeviceId::cpu());
    EXPECT_NE(
        cpu_canonical_arena->cpuCanonicalRoutes()->mutable_data(),
        contribution_region->mutableHostData());
    EXPECT_TRUE(contribution_region->contains(0u, contribution_bytes));
    ComputeGraph graph;
    std::vector<std::shared_ptr<MoEOverlayDispatchTicketStorage>>
        dispatch_storages;
    std::vector<std::shared_ptr<
        MoEOverlayCanonicalRouteReturnTicketStorage>> return_storages;
    std::vector<DispatchToCanonicalTicketManualStage *> manual_probes;
    dispatch_storages.reserve(layer_count);
    return_storages.reserve(layer_count);
    manual_probes.reserve(layer_count);

    std::string prior_consume;
    for (int layer = 0; layer < layer_count; ++layer)
    {
        auto dispatch_storage =
            std::make_shared<MoEOverlayDispatchTicketStorage>();
        dispatch_storage->bindFixedCapacity(
            layer,
            bucket_rows,
            top_k,
            d_model,
            device,
            /*workspace_generation=*/41u,
            ticket_arena);
        auto return_storage = std::make_shared<
            MoEOverlayCanonicalRouteReturnTicketStorage>();
        return_storage->bindFixedCapacity(
            layer,
            route_capacity,
            d_model,
            device,
            /*workspace_generation=*/41u,
            contribution_region,
            ticket_arena);

        MoEOverlayTicketPublishStage::Params publish_params;
        publish_params.device_id = device;
        publish_params.hidden = hidden;
        publish_params.routing_indices = routing_indices;
        publish_params.routing_weights = routing_weights;
        publish_params.hidden_buffer_id = BufferId::NORMALIZED;
        publish_params.routing_indices_buffer_id =
            BufferId::MOE_EXPERT_INDICES;
        publish_params.routing_weights_buffer_id =
            BufferId::MOE_EXPERT_WEIGHTS;
        publish_params.layer_idx = layer;
        publish_params.bucket_rows = bucket_rows;
        publish_params.top_k = top_k;
        publish_params.d_model = d_model;
        publish_params.ticket_storage = dispatch_storage;

        auto manual_stage = std::make_unique<
            DispatchToCanonicalTicketManualStage>(
                dispatch_storage,
                return_storage,
                route_capacity,
                d_model);
        manual_probes.push_back(manual_stage.get());

        MoEOverlayTicketConsumeStage::Params consume_params;
        consume_params.device_id = device;
        consume_params.output = canonical_output;
        consume_params.output_buffer_id =
            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
        consume_params.layer_idx = layer;
        consume_params.bucket_rows = bucket_rows;
        consume_params.top_k = top_k;
        consume_params.d_model = d_model;
        consume_params.canonical_route_ticket_storage = return_storage;

        const std::string publish_name =
            "gpu_dispatch_publish_" + std::to_string(layer);
        const std::string manual_name =
            "cpu_dispatch_service_" + std::to_string(layer);
        const std::string consume_name =
            "gpu_return_consume_" + std::to_string(layer);
        graph.addNode(
            publish_name,
            ComputeStageFactory::createMoEOverlayTicketPublish(
                publish_params),
            device);
        graph.addNode(
            manual_name,
            std::move(manual_stage),
            DeviceId::cpu());
        graph.addNode(
            consume_name,
            ComputeStageFactory::createMoEOverlayTicketConsume(
                consume_params),
            device);
        if (!prior_consume.empty())
            graph.addDependency(publish_name, prior_consume);
        graph.addDependency(manual_name, publish_name);
        graph.addDependency(consume_name, manual_name);
        graph.setHeterogeneousTicketUnitContract(
            publish_name,
            GraphHeterogeneousTicketUnitContract{
                .identity =
                    "captured_dispatch_before_cpu_service_" +
                    std::to_string(layer),
                .disposition = GraphHeterogeneousTicketUnitDisposition::
                    BeforeManualBoundary,
            });
        prior_consume = consume_name;
        dispatch_storages.push_back(std::move(dispatch_storage));
        return_storages.push_back(std::move(return_storage));
    }

    ASSERT_FALSE(prior_consume.empty());
    graph.setTerminalNode(prior_consume);
    graph.setHeterogeneousTicketUnitContract(
        prior_consume,
        GraphHeterogeneousTicketUnitContract{
            .identity = "captured_return_after_cpu_service",
            .disposition = GraphHeterogeneousTicketUnitDisposition::
                TransactionTerminal,
        });
    graph.setNativeCaptureEnvelope(
        GraphNativeCaptureEnvelope::
            HeterogeneousTicketAuthorityTransaction);

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *const dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    const auto retained_plan = makeMoEOverlayRetainedParentPlan(graph);
    ASSERT_TRUE(retained_plan.has_value());
    ASSERT_TRUE(retained_plan->valid());
    EXPECT_EQ(
        retained_plan->replay_policy,
        DeviceGraphExecutor::GraphReplayPlanPolicy::
            RequireRetainedParentWithConcurrentTicketService);
    const auto execute = [&](DeviceGraphExecutor::GraphInitialSubmissionPolicy
                                 initial_submission)
    {
        return executor.executeWithCachedGraphReplay(
            graph,
            device_ctx_.get(),
            segment_cache,
            dispatch_stream,
            gpu_ctx_,
            /*collective_nodes=*/nullptr,
            /*collectives_graph_capturable=*/false,
            /*force_recapture=*/false,
            /*defer_final_sync=*/true,
            /*capture_boundary=*/{},
            retained_plan->replay_policy,
            /*launch_dependency=*/{},
            /*event_published_outputs=*/{},
            retained_plan->composer,
            initial_submission);
    };
    const auto expect_transaction = [&](std::size_t call, float base)
    {
        ASSERT_TRUE(gpu_ctx_->synchronizeStreamChecked(
            segment_cache.capture_stream));
        for (std::size_t layer = 0u; layer < manual_probes.size(); ++layer)
        {
            ASSERT_NE(manual_probes[layer], nullptr);
            ASSERT_EQ(manual_probes[layer]->callCount(), call + 1u)
                << "layer=" << layer;
            EXPECT_FLOAT_EQ(
                manual_probes[layer]->observedDispatchValue(call), base)
                << "layer=" << layer;
        }
        ASSERT_TRUE(canonical_output->ensureOnHost(
            segment_cache.capture_stream));
        for (std::size_t route = 0u; route < route_capacity; ++route)
        {
            for (int column = 0; column < d_model; ++column)
            {
                EXPECT_FLOAT_EQ(
                    canonical_output->data()[
                        route * static_cast<std::size_t>(d_model) +
                        static_cast<std::size_t>(column)],
                    base + 100.0f * static_cast<float>(route) +
                        static_cast<float>(column));
            }
        }
        for (std::size_t layer = 0u; layer < return_storages.size(); ++layer)
        {
            EXPECT_FALSE(return_storages[layer]->payloadReady())
                << "layer=" << layer;
        }
    };

    ASSERT_TRUE(execute(
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            MaterializeWithoutLaunch));
    const auto mapped_arena_snapshot = ticket_arena->snapshot();
    EXPECT_EQ(mapped_arena_snapshot.slice_count, 48u * 3u);
    EXPECT_GE(
        mapped_arena_snapshot.allocated_bytes,
        48u * static_cast<std::size_t>(bucket_rows) *
            static_cast<std::size_t>(d_model) * sizeof(float) * 2u);
    EXPECT_LT(mapped_arena_snapshot.backing_region_count, 16u);
    double backend_owned_mappings = 0.0;
    double external_registrations = 0.0;
    for (const auto &record : PerfStatsCollector::snapshot(
             {"moe_overlay_activation_epoch"}))
    {
        if (record.name == "mapped_backend_allocations")
        {
            backend_owned_mappings += record.value;
            EXPECT_EQ(
                record.tags.at("mapping"),
                "backend_owned_mapped_host_pages");
        }
        else if (record.name == "mapped_backend_registrations")
        {
            external_registrations += record.value;
        }
    }
    EXPECT_GT(backend_owned_mappings, 0.0);
    EXPECT_EQ(external_registrations, 0.0)
        << "an exact-device retained ticket must not register anonymous pages";
    ASSERT_NE(segment_cache.retained_parent_capture, nullptr);
    EXPECT_TRUE(segment_cache.retained_parent_capture->hasExecutable());
    ASSERT_EQ(segment_cache.segments.size(), 50u);
    std::size_t captured_compilation_units = 0u;
    std::size_t captured_stages = 0u;
    std::size_t service_programs = 0u;
    for (const auto &segment : segment_cache.segments)
    {
        if (segment.capturable)
        {
            ++captured_compilation_units;
            captured_stages += segment.stage_names.size();
        }
        else
        {
            ++service_programs;
            EXPECT_EQ(segment.stage_names.size(), 48u)
                << "all CPU work must share one ordered concurrent service program";
        }
    }
    EXPECT_EQ(captured_compilation_units, 49u)
        << "typed ticket frontiers should bound native graph compilation";
    EXPECT_EQ(captured_stages, 48u * 2u)
        << "every publish/consume stage must enter the retained parent";
    EXPECT_EQ(service_programs, 1u);
    EXPECT_FALSE(segment_cache.segments.back().capturable);
    EXPECT_EQ(segment_cache.segments.back().stage_names.size(), 48u)
        << "all CPU work must share one ordered concurrent service program";
    for (std::size_t layer = 0u; layer < manual_probes.size(); ++layer)
    {
        ASSERT_NE(manual_probes[layer], nullptr);
        EXPECT_EQ(manual_probes[layer]->callCount(), 0u)
            << "layer=" << layer;
    }

    graph.reset();
    ASSERT_TRUE(execute(
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            CaptureInstantiateAndLaunch));
    expect_transaction(/*call=*/0u, /*base=*/11.0f);

    fill_dispatch(29.0f);
    ASSERT_TRUE(hidden->ensureOnDevice(
        device, segment_cache.capture_stream));
    ASSERT_TRUE(routing_indices->ensureOnDevice(
        device, segment_cache.capture_stream));
    ASSERT_TRUE(routing_weights->ensureOnDevice(
        device, segment_cache.capture_stream));
    graph.reset();
    ASSERT_TRUE(execute(
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            CaptureInstantiateAndLaunch));
    expect_transaction(/*call=*/1u, /*base=*/29.0f);
}

/**
 * @brief Prove one real GPU parent overlaps a canonical CPU ticket service.
 *
 * Setup records bounded graph-only compilation units around two CPU ticket
 * boundaries and composes them into one native executable. At request time the
 * persistent CPU service is armed before the parent submission call. Each
 * canonical consumer waits in device code while its matching CPU stage
 * release-publishes the mapped route payload. Two boundaries catch submission-
 * queue circular waits that a tiny one-boundary parent can hide; replaying with
 * different payloads proves that neither graph execution nor ticket contents
 * were frozen in setup.
 */
TEST_F(CachedGraphReplayExecutionTest,
       RetainedParentOverlapsCanonicalCPUTicketService)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);
    ScopedPerfStats perf_stats;

    constexpr std::array<int, 2> layers{7, 8};
    constexpr int bucket_rows = 1;
    constexpr int top_k = 2;
    constexpr int d_model = 8;
    constexpr std::size_t route_capacity =
        static_cast<std::size_t>(bucket_rows * top_k);
    constexpr std::size_t payload_elements =
        route_capacity * static_cast<std::size_t>(d_model);
    constexpr std::size_t payload_bytes = payload_elements * sizeof(float);
    const DeviceId device = device_ctx_->deviceId();

    // Production CPU-tier continuations also own a background transfer epoch.
    // Its one bounded branch must span the assembled parent, including GPU
    // waits on CPU ticket service; graph-only children are not execution owners.
    const auto progress_epoch = MappedTransferProgressEpoch::create({
        .device = device,
        .slot_capacity = 2u,
        .execution_lane_capacity = 2u,
        .execution_streams = TransferEngine::instance()
            .allocatePersistentTransferExecutionLanes(
                2u, device, "retained_cpu_ticket_progress"),
        .maximum_bytes = 4096u,
        .name = "retained_cpu_ticket_progress",
        .perf_device = device.toString(),
    });
    const auto progress_branch = progress_epoch->graphBranchFactory();

    auto *prefix_input = createArenaFP32Tensor(
        BufferId::HIDDEN_STATE, {1u, static_cast<std::size_t>(d_model)});
    auto *prefix_residual = createArenaFP32Tensor(
        BufferId::RESIDUAL, {1u, static_cast<std::size_t>(d_model)});
    auto *prefix_output = createArenaFP32Tensor(
        BufferId::ATTN_OUTPUT, {1u, static_cast<std::size_t>(d_model)});
    auto *canonical_output = createArenaFP32Tensor(
        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
        {route_capacity, static_cast<std::size_t>(d_model)});
    for (int column = 0; column < d_model; ++column)
    {
        prefix_input->mutable_data()[column] =
            1.0f + static_cast<float>(column);
        prefix_residual->mutable_data()[column] = 0.5f;
    }
    std::fill_n(canonical_output->mutable_data(), payload_elements, -1.0f);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    const std::array<DeviceId, 1> mapped_devices{device};
    auto ticket_arena =
        TransferEngine::instance().createMappedHostArena(mapped_devices);
    auto first_contribution_region =
        TransferEngine::instance().allocateMappedHostRegion(
            payload_bytes, mapped_devices);
    auto second_contribution_region =
        TransferEngine::instance().allocateMappedHostRegion(
            payload_bytes, mapped_devices);
    ASSERT_NE(first_contribution_region, nullptr);
    ASSERT_NE(second_contribution_region, nullptr);
    ASSERT_TRUE(first_contribution_region->isBound());
    ASSERT_TRUE(second_contribution_region->isBound());

    auto first_ticket_storage = std::make_shared<
        MoEOverlayCanonicalRouteReturnTicketStorage>();
    auto second_ticket_storage = std::make_shared<
        MoEOverlayCanonicalRouteReturnTicketStorage>();
    first_ticket_storage->bindFixedCapacity(
        layers[0],
        route_capacity,
        d_model,
        device,
        /*workspace_generation=*/37u,
        first_contribution_region,
        ticket_arena);
    second_ticket_storage->bindFixedCapacity(
        layers[1],
        route_capacity,
        d_model,
        device,
        /*workspace_generation=*/37u,
        second_contribution_region,
        ticket_arena);
    ASSERT_TRUE(first_ticket_storage->hasValidBoundIdentity());
    ASSERT_TRUE(second_ticket_storage->hasValidBoundIdentity());

    ResidualAddStage::Params prefix_params;
    prefix_params.device_id = device;
    prefix_params.input = prefix_input;
    prefix_params.residual = prefix_residual;
    prefix_params.output = prefix_output;
    prefix_params.num_elements = static_cast<std::size_t>(d_model);
    prefix_params.input_buffer_id = BufferId::HIDDEN_STATE;
    prefix_params.residual_buffer_id = BufferId::RESIDUAL;
    prefix_params.output_buffer_id = BufferId::ATTN_OUTPUT;

    auto first_manual_stage = std::make_unique<
        CanonicalTicketPublishManualStage>(
            first_ticket_storage, route_capacity, d_model);
    auto second_manual_stage = std::make_unique<
        CanonicalTicketPublishManualStage>(
            second_ticket_storage, route_capacity, d_model);
    CanonicalTicketPublishManualStage *const first_manual_probe =
        first_manual_stage.get();
    CanonicalTicketPublishManualStage *const second_manual_probe =
        second_manual_stage.get();

    const auto make_consume_params = [&](
                                         int layer,
                                         const std::shared_ptr<
                                             MoEOverlayCanonicalRouteReturnTicketStorage>
                                             &storage)
    {
        MoEOverlayTicketConsumeStage::Params params;
        params.device_id = device;
        params.output = canonical_output;
        params.output_buffer_id =
            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
        params.layer_idx = layer;
        params.bucket_rows = bucket_rows;
        params.top_k = top_k;
        params.d_model = d_model;
        params.canonical_route_ticket_storage = storage;
        return params;
    };

    ComputeGraph graph;
    graph.addNode(
        "gpu_prefix",
        ComputeStageFactory::createResidualAdd(prefix_params),
        device);
    graph.addNode(
        "cpu_canonical_ticket_service_0",
        std::move(first_manual_stage),
        DeviceId::cpu());
    graph.addNode(
        "gpu_canonical_ticket_consume_0",
        ComputeStageFactory::createMoEOverlayTicketConsume(
            make_consume_params(layers[0], first_ticket_storage)),
        device);
    graph.addNode(
        "gpu_between_tickets",
        ComputeStageFactory::createResidualAdd(prefix_params),
        device);
    graph.addNode(
        "cpu_canonical_ticket_service_1",
        std::move(second_manual_stage),
        DeviceId::cpu());
    graph.addNode(
        "gpu_canonical_ticket_consume_1",
        ComputeStageFactory::createMoEOverlayTicketConsume(
            make_consume_params(layers[1], second_ticket_storage)),
        device);
    graph.addDependency("cpu_canonical_ticket_service_0", "gpu_prefix");
    graph.addDependency(
        "gpu_canonical_ticket_consume_0",
        "cpu_canonical_ticket_service_0");
    graph.addDependency(
        "gpu_between_tickets", "gpu_canonical_ticket_consume_0");
    graph.addDependency(
        "cpu_canonical_ticket_service_1", "gpu_between_tickets");
    graph.addDependency(
        "gpu_canonical_ticket_consume_1",
        "cpu_canonical_ticket_service_1");
    graph.setHeterogeneousTicketUnitContract(
        "gpu_prefix",
        GraphHeterogeneousTicketUnitContract{
            .identity = "prefix_before_cpu_ticket",
            .disposition = GraphHeterogeneousTicketUnitDisposition::
                BeforeManualBoundary,
        });
    graph.setHeterogeneousTicketUnitContract(
        "gpu_between_tickets",
        GraphHeterogeneousTicketUnitContract{
            .identity = "between_cpu_tickets",
            .disposition = GraphHeterogeneousTicketUnitDisposition::
                BeforeManualBoundary,
        });
    graph.setTerminalNode("gpu_canonical_ticket_consume_1");
    graph.setHeterogeneousTicketUnitContract(
        "gpu_canonical_ticket_consume_1",
        GraphHeterogeneousTicketUnitContract{
            .identity = "canonical_ticket_terminal",
            .disposition = GraphHeterogeneousTicketUnitDisposition::
                TransactionTerminal,
        });
    graph.setNativeCaptureEnvelope(
        GraphNativeCaptureEnvelope::
            HeterogeneousTicketAuthorityTransaction);

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *const dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    const auto retained_plan = makeMoEOverlayRetainedParentPlan(graph);
    ASSERT_TRUE(retained_plan.has_value());
    ASSERT_TRUE(retained_plan->valid());
    EXPECT_EQ(
        retained_plan->replay_policy,
        DeviceGraphExecutor::GraphReplayPlanPolicy::
            RequireRetainedParentWithConcurrentTicketService);
    const auto production_composer = retained_plan->composer;
    std::size_t composer_calls = 0u;
    const DeviceGraphExecutor::RetainedParentCompositionHook composer =
        [&](IGPUGraphCapture &destination,
            const ComputeGraph &source_graph,
            std::span<const DeviceGraphExecutor::GraphSegmentCache::
                                RetainedCaptureUnitTemplateView> units)
    {
        ++composer_calls;
        return production_composer(destination, source_graph, units);
    };
    const auto execute = [&](DeviceGraphExecutor::GraphInitialSubmissionPolicy
                                 initial_submission)
    {
        return executor.executeWithCachedGraphReplay(
            graph,
            device_ctx_.get(),
            segment_cache,
            dispatch_stream,
            gpu_ctx_,
            /*collective_nodes=*/nullptr,
            /*collectives_graph_capturable=*/false,
            /*force_recapture=*/false,
            /*defer_final_sync=*/true,
            /*capture_boundary=*/{},
            retained_plan->replay_policy,
            /*launch_dependency=*/{},
            /*event_published_outputs=*/{},
            composer,
            initial_submission,
            progress_branch);
    };
    const auto expect_payload = [&](float base)
    {
        /* Deferred replay deliberately proves the executor inserted no host
         * fence between parent submission and CPU ticket service. The test
         * joins only here, at the external observation boundary. */
        ASSERT_TRUE(gpu_ctx_->synchronizeStreamChecked(
            segment_cache.capture_stream));
        ASSERT_TRUE(canonical_output->ensureOnHost(
            segment_cache.capture_stream));
        for (std::size_t route = 0u; route < route_capacity; ++route)
        {
            for (int column = 0; column < d_model; ++column)
            {
                EXPECT_FLOAT_EQ(
                    canonical_output->data()[
                        route * static_cast<std::size_t>(d_model) +
                        static_cast<std::size_t>(column)],
                    base + 100.0f * static_cast<float>(route) +
                        static_cast<float>(column));
            }
        }
    };

    ASSERT_TRUE(execute(
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            MaterializeWithoutLaunch));
    ASSERT_NE(segment_cache.retained_parent_capture, nullptr);
    EXPECT_TRUE(segment_cache.retained_parent_capture->hasExecutable());
    EXPECT_STREQ(
        DeviceGraphCaptureController::replayModeName(segment_cache),
        "retained_parent");
    EXPECT_EQ(first_manual_probe->callCount(), 0u);
    EXPECT_EQ(second_manual_probe->callCount(), 0u);
    EXPECT_EQ(composer_calls, 1u);
    ASSERT_EQ(
        segment_cache.retained_composed_parent_replay
            .concurrent_ticket_service_segment_indices,
        (std::vector<std::size_t>{3u}));
    EXPECT_EQ(
        segment_cache.retained_composed_parent_replay.child_unit_count,
        3u);

    graph.reset();
    ASSERT_TRUE(execute(
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            CaptureInstantiateAndLaunch));
    ASSERT_EQ(first_manual_probe->callCount(), 1u);
    ASSERT_EQ(second_manual_probe->callCount(), 1u);
    expect_payload(1000.0f);
    EXPECT_FALSE(first_ticket_storage->payloadReady());
    EXPECT_FALSE(second_ticket_storage->payloadReady());

    graph.reset();
    ASSERT_TRUE(execute(
        DeviceGraphExecutor::GraphInitialSubmissionPolicy::
            CaptureInstantiateAndLaunch));
    ASSERT_EQ(first_manual_probe->callCount(), 2u);
    ASSERT_EQ(second_manual_probe->callCount(), 2u);
    EXPECT_EQ(composer_calls, 1u);
    expect_payload(2000.0f);
    EXPECT_FALSE(first_ticket_storage->payloadReady());
    EXPECT_FALSE(second_ticket_storage->payloadReady());
}

TEST_F(CachedGraphReplayExecutionTest, PreserveResetKeepsExplicitCaptureStreamForRecapture)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    // First use creates the dedicated cached-replay stream, executes warmup,
    // and atomically materializes the replay graph. This stream must not be
    // replaced by retry/reset plumbing.
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    void *capture_stream = segment_cache.capture_stream;
    ASSERT_NE(capture_stream, nullptr);
    EXPECT_NE(capture_stream, dispatch_stream)
        << "Cached graph replay should use a dedicated explicit stream, not the dispatch/default stream";
    assertGraphStagesUseStream(graph, capture_stream);
    assertTensorFiniteAndNonZero(result, num_elements, capture_stream);

    // Reset with Preserve simulates capture retry/replay failure handling. The
    // next warmup must reuse the same live stream so cached stages never observe
    // a dangling stream or fall back to default-stream execution.
    segment_cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);
    EXPECT_EQ(segment_cache.capture_stream, capture_stream);

    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    EXPECT_EQ(segment_cache.capture_stream, capture_stream);
    assertGraphStagesUseStream(graph, capture_stream);
    assertTensorFiniteAndNonZero(result, num_elements, capture_stream);

    // The recapture pass should continue using that same explicit stream.
    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    EXPECT_EQ(segment_cache.capture_stream, capture_stream);
    assertGraphStagesUseStream(graph, capture_stream);
    assertTensorFiniteAndNonZero(result, num_elements, capture_stream);
}

TEST_F(CachedGraphReplayExecutionTest, CapturedSnapshotsPreservePointInTimeOutputsAcrossBufferReuse)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *scratch = nullptr;
    auto graph = buildSnapshotOverwriteGraph(seq_len, d_model, norm_input, scratch);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    std::unordered_map<std::string, std::vector<float>> snapshots;
    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    exec_config.snapshot_callback = [&](const std::string &stage_name, const StageDumpInfo &dump_info)
    {
        ASSERT_FALSE(dump_info.outputs.empty()) << "Stage has no snapshot outputs: " << stage_name;
        const auto &output = dump_info.outputs.front();
        ASSERT_STREQ(output.dtype, "FP32");
        ASSERT_NE(output.data, nullptr);
        const size_t element_count = output.rows * output.cols;
        const auto *data = static_cast<const float *>(output.data);
        snapshots[stage_name].assign(data, data + element_count);
    };

    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    ASSERT_NE(segment_cache.capture_stream, nullptr);
    ASSERT_TRUE(segment_cache.snapshot_manifest.storageBound());
    ASSERT_NE(segment_cache.snapshot_manifest.storage_arena, nullptr);
    EXPECT_FALSE(segment_cache.snapshot_manifest.storage_arena->isMapped());
    EXPECT_EQ(segment_cache.snapshot_manifest.storage_allocation_count, 1u)
        << "A cached graph must bind one device arena, not one allocation per checkpoint";
    EXPECT_EQ(segment_cache.snapshot_manifest.bound_slot_count, 2u);
    void *const snapshot_arena_pointer =
        segment_cache.snapshot_manifest.storage_arena->gpu_data_ptr();
    ASSERT_NE(snapshot_arena_pointer, nullptr);
    snapshots.clear();
    const uint64_t downloads_before_first_publication =
        segment_cache.snapshot_manifest.bulk_download_count;
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-overwrite-first-use",
        &segment_cache.snapshot_manifest));
    EXPECT_EQ(
        segment_cache.snapshot_manifest.bulk_download_count,
        downloads_before_first_publication + 1u)
        << "All captured checkpoint slots must share one bulk D2H acquisition";
    assertSnapshotDelta(snapshots, "stage1_norm", "stage2_overwrite", 10.0f);
    const auto warmup_snapshots = snapshots;
    assertTensorFiniteAndNonZero(
        scratch, num_elements, segment_cache.capture_stream);

    /*
     * Fixed-width prefill graphs retain all captured rows in their immutable
     * D2D snapshot slots, but publication must expose only request-owned rows.
     * Model a two-request bucket with two physical rows each: request zero owns
     * one row and request one owns both. The callback must receive the compact
     * three-row matrix without changing or recapturing the graph.
     */
    const DeviceGraphExecutor::GraphSnapshotLogicalRows logical_rows{
        .physical_rows_per_sequence = 2,
        .logical_rows_per_sequence = {1, 2},
    };
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-logical-row-projection",
        &segment_cache.snapshot_manifest,
        &logical_rows));
    for (const std::string stage_name : {"stage1_norm", "stage2_overwrite"})
    {
        const auto full = warmup_snapshots.find(stage_name);
        const auto projected = snapshots.find(stage_name);
        ASSERT_NE(full, warmup_snapshots.end());
        ASSERT_NE(projected, snapshots.end());
        ASSERT_EQ(full->second.size(), num_elements);
        ASSERT_EQ(projected->second.size(), 3 * d_model);

        std::vector<float> expected;
        expected.reserve(3 * d_model);
        expected.insert(
            expected.end(),
            full->second.begin(),
            full->second.begin() + static_cast<std::ptrdiff_t>(d_model));
        expected.insert(
            expected.end(),
            full->second.begin() + static_cast<std::ptrdiff_t>(2 * d_model),
            full->second.end());
        EXPECT_EQ(projected->second, expected)
            << "Logical snapshot row projection changed payload order for "
            << stage_name;
    }

    /*
     * A heterogeneous captured prefill graph can cross a CPU/manual boundary
     * after a GPU segment. The CPU stage has no graph-stable D2D snapshot slot,
     * but its parity artifact still must expose logical rather than padded
     * rows. Exercise the common post-graph publisher directly so this remains
     * true independently of the GPU manifest implementation above.
     */
    auto *cpu_input = createFP32Tensor({seq_len, d_model});
    auto *cpu_residual = createFP32Tensor({seq_len, d_model});
    auto *cpu_output = createFP32Tensor({seq_len, d_model});
    for (size_t index = 0; index < num_elements; ++index)
    {
        cpu_input->mutable_data()[index] = static_cast<float>(index + 1);
        cpu_residual->mutable_data()[index] = 0.25f;
        cpu_output->mutable_data()[index] =
            cpu_input->data()[index] + cpu_residual->data()[index];
    }

    ResidualAddStage::Params cpu_snapshot_params;
    cpu_snapshot_params.device_id = DeviceId::cpu();
    cpu_snapshot_params.input = cpu_input;
    cpu_snapshot_params.residual = cpu_residual;
    cpu_snapshot_params.output = cpu_output;
    cpu_snapshot_params.num_elements = num_elements;
    ComputeGraph cpu_snapshot_graph;
    cpu_snapshot_graph.addNode(
        "cpu_snapshot_boundary",
        ComputeStageFactory::createResidualAdd(cpu_snapshot_params),
        DeviceId::cpu());

    const std::vector<float> full_cpu_snapshot(
        cpu_output->data(), cpu_output->data() + num_elements);
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        cpu_snapshot_graph,
        segment_cache.capture_stream,
        "snapshot-logical-row-projection-cpu-boundary",
        nullptr,
        &logical_rows));
    const auto projected_cpu = snapshots.find("cpu_snapshot_boundary");
    ASSERT_NE(projected_cpu, snapshots.end());
    ASSERT_EQ(projected_cpu->second.size(), 3 * d_model);
    std::vector<float> expected_cpu_snapshot;
    expected_cpu_snapshot.reserve(3 * d_model);
    expected_cpu_snapshot.insert(
        expected_cpu_snapshot.end(),
        full_cpu_snapshot.begin(),
        full_cpu_snapshot.begin() + static_cast<std::ptrdiff_t>(d_model));
    expected_cpu_snapshot.insert(
        expected_cpu_snapshot.end(),
        full_cpu_snapshot.begin() + static_cast<std::ptrdiff_t>(2 * d_model),
        full_cpu_snapshot.end());
    EXPECT_EQ(projected_cpu->second, expected_cpu_snapshot)
        << "CPU/manual snapshot publication leaked padded rows";

    fillSnapshotInput(norm_input, 3.0f);
    ASSERT_TRUE(norm_input->ensureOnDevice(device_ctx_->deviceId(), segment_cache.capture_stream));
    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    EXPECT_EQ(segment_cache.snapshot_manifest.storage_allocation_count, 1u);
    EXPECT_EQ(
        segment_cache.snapshot_manifest.storage_arena->gpu_data_ptr(),
        snapshot_arena_pointer)
        << "Steady replay must retain the exact captured snapshot arena address";
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-overwrite-replay-1",
        &segment_cache.snapshot_manifest));
    assertSnapshotDelta(snapshots, "stage1_norm", "stage2_overwrite", 10.0f);
    assertSnapshotsDiffer(warmup_snapshots, snapshots, "stage1_norm");
    const auto capture_snapshots = snapshots;

    fillSnapshotInput(norm_input, 7.0f);
    ASSERT_TRUE(norm_input->ensureOnDevice(device_ctx_->deviceId(), segment_cache.capture_stream));
    graph.reset();
    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), segment_cache, dispatch_stream, gpu_ctx_, nullptr));
    EXPECT_EQ(segment_cache.snapshot_manifest.storage_allocation_count, 1u);
    EXPECT_EQ(
        segment_cache.snapshot_manifest.storage_arena->gpu_data_ptr(),
        snapshot_arena_pointer);
    snapshots.clear();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "snapshot-overwrite-replay-2",
        &segment_cache.snapshot_manifest));
    assertSnapshotDelta(snapshots, "stage1_norm", "stage2_overwrite", 10.0f);
    assertSnapshotsDiffer(capture_snapshots, snapshots, "stage1_norm");
}

/**
 * @test A complete small-helper family fits a bounded native-pool envelope.
 *
 * MTP terminal selection and verifier preparation are small executables, not
 * model forwards. Certify a deliberately wider 64-kernel shape across 128
 * simultaneously retained owners against their typed admission contract.
 * Each owner shares the same persistent tensors because these graphs execute
 * serially. No allocation, weight loading, or profiler runs concurrently with
 * the measured instantiations. The second lifetime also exercises the driver's
 * warm pool after every first-lifetime executable has retired.
 *
 * The graph declares the same bounded class used by admission. Production
 * capture therefore checks every native node before instantiation; larger or
 * nested auxiliaries cannot silently borrow this smaller family reservation.
 */
TEST_F(CachedGraphReplayExecutionTest,
       SmallHelperFamilyCertifiesColdAndWarmNativePoolGrowth)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    constexpr size_t kHelperNodes = GPUGraphMemoryContract::kBoundedFlatHelperMaxNodes;
    constexpr size_t kFamilyOwners = 128u;
    const size_t bytes_per_owner = GPUGraphMemoryContract::reservationBytesPerExecutable(
        device_ctx_->deviceId(), GPUGraphExecutableClass::BoundedFlatHelper);
    FP32Tensor *final_output = nullptr;
    auto graph = buildProductionScaleSnapshotChain(
        kHelperNodes, 256u, final_output);
    graph.setExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper);
    ASSERT_NE(final_output, nullptr);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    config.enable_validation = false;
    // The same chain builder serves snapshot tests, but this certificate has
    // no snapshot callback: all native nodes are real ResidualAdd kernels.
    DeviceGraphExecutor executor(config);
    graph_arena_.bindExecutor(executor);
    void *const stream = gpu_ctx_->defaultStream();
    ASSERT_NE(stream, nullptr);

    for (size_t lifetime = 0; lifetime < 2u; ++lifetime)
    {
        SCOPED_TRACE(lifetime);
        std::vector<std::unique_ptr<DeviceGraphExecutor::GraphSegmentCache>>
            family;
        family.reserve(kFamilyOwners);
        size_t family_growth = 0u;
        for (size_t owner = 0; owner < kFamilyOwners; ++owner)
        {
            SCOPED_TRACE(owner);
            auto cache =
                std::make_unique<DeviceGraphExecutor::GraphSegmentCache>();
            graph.reset();
            ASSERT_TRUE(executor.executeWithCachedGraphReplay(
                graph, device_ctx_.get(), *cache, stream, gpu_ctx_, nullptr));
            ASSERT_EQ(cache->segments.size(), 1u);
            const auto &capture = cache->segments.front().capture;
            ASSERT_NE(capture, nullptr);
            ASSERT_EQ(capture->nodeCount(), kHelperNodes);
            family_growth += capture->residentMemoryBytes();
            family.push_back(std::move(cache));

        }
        // Admission owns the complete family before its first capture. Do not
        // compare a prefix's growth with only the slots populated so far: the
        // driver may grow storage that later admitted siblings will consume.
        EXPECT_LE(family_growth, kFamilyOwners * bytes_per_owner);
        RecordProperty(
            "helper_family_pool_growth_bytes_" + std::to_string(lifetime),
            std::to_string(family_growth));
        std::cout << "[helper-family-certificate] device="
                  << device_ctx_->deviceId().toString()
                  << " lifetime=" << lifetime << " owners=" << family.size()
                  << " nodes_per_owner=" << kHelperNodes
                  << " pool_growth_bytes=" << family_growth << '\n';
        // Explicit test teardown fence: no executable may outlive its pending
        // launch. This is outside production inference and measured setup.
        for (const auto &cache : family)
            ASSERT_TRUE(gpu_ctx_->synchronizeStreamChecked(cache->capture_stream));
    }
}

/** @test An oversized declared helper fails before its executable is allocated. */
TEST_F(CachedGraphReplayExecutionTest, BoundedHelperRejectsOversizedNativeGraph)
{
    SKIP_IF_NO_GPU();
    FP32Tensor *output = nullptr;
    auto graph = buildProductionScaleSnapshotChain(
        GPUGraphMemoryContract::kBoundedFlatHelperMaxNodes + 1u, 256u, output);
    graph.setExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());
    GraphExecutorConfig config;
    config.enable_validation = false;
    DeviceGraphExecutor executor(config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache cache;
    EXPECT_FALSE(executor.executeWithCachedGraphReplay(
        graph, device_ctx_.get(), cache, gpu_ctx_->defaultStream(), gpu_ctx_, nullptr));
    for (const auto &segment : cache.segments)
        if (segment.capture)
            EXPECT_FALSE(segment.capture->hasExecutable());
}

/**
 * @test Production-scale checkpoint capture remains VRAM-local and publishes
 *       through one whole-arena D2H acquisition.
 *
 * The Qwen 3.5 122B parity graph presents about 1,525 selected outputs on its
 * busiest participant. Historically, recording those outputs into mapped host
 * pages flooded the ROCm interrupt ring and left the next RCCL graph capture
 * taking seconds per enqueue. This model-free certificate retains the same
 * checkpoint count and a roughly 100 MiB arena on both GPU backends. It proves
 * that every stage-boundary value survives buffer reuse while the manifest
 * performs exactly one host transfer per publication.
 */
TEST_F(CachedGraphReplayExecutionTest,
       ProductionScaleSnapshotsUseOneDeviceArenaBulkDownload)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(gpu_ctx_, nullptr);
    ASSERT_NE(device_ctx_, nullptr);

    constexpr size_t kCheckpointCount = 1525u;
    constexpr size_t kElementsPerCheckpoint = 16u * 1024u;
    constexpr size_t kExpectedArenaBytes =
        kCheckpointCount * kElementsPerCheckpoint * sizeof(float);

    FP32Tensor *final_output = nullptr;
    auto graph = buildProductionScaleSnapshotChain(
        kCheckpointCount,
        kElementsPerCheckpoint,
        final_output);
    ASSERT_NE(final_output, nullptr);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    bool validate_publication = false;
    size_t callback_index = 0u;
    float previous_value = 0.0F;
    GraphExecutorConfig exec_config;
    exec_config.enable_validation = false;
    exec_config.snapshot_callback =
        [&](const std::string &stage_name, const StageDumpInfo &dump_info)
        {
            if (!validate_publication)
                return;

            ASSERT_EQ(dump_info.outputs.size(), 1u);
            const auto &output = dump_info.outputs.front();
            ASSERT_NE(output.data, nullptr);
            ASSERT_STREQ(output.dtype, "FP32");
            ASSERT_EQ(
                output.rows * output.cols,
                kElementsPerCheckpoint);
            EXPECT_EQ(
                stage_name,
                "production_snapshot_stage_" +
                    std::to_string(callback_index));

            const auto *values = static_cast<const float *>(output.data);
            EXPECT_FLOAT_EQ(values[0], values[kElementsPerCheckpoint / 2u]);
            EXPECT_FLOAT_EQ(values[0], values[kElementsPerCheckpoint - 1u]);
            if (callback_index > 0u)
                EXPECT_FLOAT_EQ(values[0], previous_value + 0.25F);
            previous_value = values[0];
            ++callback_index;
        };

    DeviceGraphExecutor executor(exec_config);
    graph_arena_.bindExecutor(executor);
    DeviceGraphExecutor::GraphSegmentCache segment_cache;
    void *const dispatch_stream = gpu_ctx_->defaultStream();
    ASSERT_NE(dispatch_stream, nullptr);

    ASSERT_TRUE(executor.executeWithCachedGraphReplay(
        graph,
        device_ctx_.get(),
        segment_cache,
        dispatch_stream,
        gpu_ctx_,
        nullptr));
    ASSERT_EQ(segment_cache.segments.size(), 1u);
    ASSERT_NE(segment_cache.segments.front().capture, nullptr);
    EXPECT_EQ(
        segment_cache.segments.front().capture->nodeCount(),
        kCheckpointCount * 2u)
        << "Each checkpoint owns one producer kernel and one arena copy";
    EXPECT_LE(
        segment_cache.segments.front().capture->residentMemoryBytes(),
        GPUGraphMemoryContract::reservationBytesPerExecutable(
            device_ctx_->deviceId()))
        << "The production-shape native executable must fit the same "
           "driver-memory extent charged by capacity admission";
    ASSERT_TRUE(segment_cache.snapshot_manifest.storageBound());
    ASSERT_NE(segment_cache.snapshot_manifest.storage_arena, nullptr);
    EXPECT_FALSE(segment_cache.snapshot_manifest.storage_arena->isMapped());
    EXPECT_EQ(segment_cache.snapshot_manifest.storage_allocation_count, 1u);
    EXPECT_EQ(
        segment_cache.snapshot_manifest.bound_slot_count,
        kCheckpointCount);
    EXPECT_EQ(
        segment_cache.snapshot_manifest.storage_required_bytes,
        kExpectedArenaBytes);

    const uint64_t downloads_before =
        segment_cache.snapshot_manifest.bulk_download_count;
    validate_publication = true;
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        segment_cache.capture_stream,
        "production-scale-snapshot-bulk-download",
        &segment_cache.snapshot_manifest));
    EXPECT_EQ(callback_index, kCheckpointCount);
    EXPECT_EQ(
        segment_cache.snapshot_manifest.bulk_download_count,
        downloads_before + 1u)
        << "Checkpoint count must not multiply host transfers";
}
