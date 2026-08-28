/**
 * @file Test__MappedActivationPacketCUDAAndROCm.cpp
 * @brief Real-device sparse-payload proof for node-local ExpertOverlay epochs.
 *
 * The production node-local transport resolves one mapped packet lane for each
 * planner-selected endpoint. This test runs the complete CUDA/ROCm packet path
 * in both role assignments and across three ordered model stages. The retained
 * endpoint graphs cover continuation route compaction, exact-stream mapped
 * publication, follower validation/CSR expansion, follower device work,
 * compact return publication, and deterministic continuation accumulation.
 * The three stages reuse both alternating packet banks, and all work is
 * submitted before either terminal event is observed.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/moe/MoEOverlayActivationEpochProtocol.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/moe/MoEOverlayRetainedActivationTransaction.h"
#include "execution/moe/MoERuntimeTable.h"
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockMPITopology.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    constexpr std::int32_t kSourceParticipant = 2;
    constexpr std::int32_t kTargetParticipant = 7;
    constexpr std::array<std::int32_t, 3> kLayers{5, 9, 14};
    constexpr std::int32_t kDModel = 8;
    constexpr std::int32_t kTopK = 2;
    constexpr std::uint32_t kExpertCount = 18u;
    constexpr std::uint64_t kPlacementEpoch = 41u;
    constexpr std::uint64_t kSchedulerPlacementEpochFloor =
        kPlacementEpoch - 1u;
    constexpr std::array<std::int32_t, 8> kRouteParticipantPattern{
        kTargetParticipant, kSourceParticipant,
        kSourceParticipant, kSourceParticipant,
        kTargetParticipant, kTargetParticipant,
        kSourceParticipant, kTargetParticipant};
    constexpr std::array<float, 8> kRouteWeightPattern{
        0.75f, 0.25f, 0.6f, 0.4f,
        0.55f, 0.45f, 0.9f, 0.1f};

    /** @return Planner owner used by the deterministic router fixture. */
    constexpr std::int32_t routeParticipant(std::size_t slot) noexcept
    {
        return kRouteParticipantPattern[slot % kRouteParticipantPattern.size()];
    }

    /** @return Whether one logical row carries any route for the follower. */
    constexpr bool rowVisitsTarget(std::int32_t row) noexcept
    {
        for (std::int32_t slot = 0; slot < kTopK; ++slot)
        {
            if (routeParticipant(
                    static_cast<std::size_t>(row * kTopK + slot)) ==
                kTargetParticipant)
            {
                return true;
            }
        }
        return false;
    }

    /** @return Number of compact follower rows for a fixed physical geometry. */
    constexpr std::uint64_t targetLiveRows(std::int32_t rows) noexcept
    {
        std::uint64_t result = 0u;
        for (std::int32_t row = 0; row < rows; ++row)
            result += rowVisitsTarget(row) ? 1u : 0u;
        return result;
    }

    /** @return Number of compact route entries for a fixed physical geometry. */
    constexpr std::uint64_t targetLiveEntries(std::int32_t rows) noexcept
    {
        std::uint64_t result = 0u;
        for (std::int32_t row = 0; row < rows; ++row)
        {
            for (std::int32_t slot = 0; slot < kTopK; ++slot)
            {
                result += routeParticipant(
                              static_cast<std::size_t>(row * kTopK + slot)) ==
                              kTargetParticipant
                              ? 1u
                              : 0u;
            }
        }
        return result;
    }

    /**
     * @brief Enable packet and graph fork/join evidence for this process.
     *
     * CTest requests the same domain, but the binary is also run directly by
     * profiler and repetition workflows. Reloading the parsed environment here
     * keeps those invocations self-certifying without enabling unrelated hot-
     * path instrumentation.
     */
    class ScopedActivationEpochPerfStats final
    {
    public:
        ScopedActivationEpochPerfStats()
        {
            preserve("LLAMINAR_PERF_STATS_SUMMARY", summary_);
            preserve("LLAMINAR_PERF_STATS_FILTER", filter_);
            ::setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            ::setenv(
                "LLAMINAR_PERF_STATS_FILTER",
                "moe_overlay_activation_epoch,forward_graph",
                1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedActivationEpochPerfStats()
        {
            restore("LLAMINAR_PERF_STATS_SUMMARY", summary_);
            restore("LLAMINAR_PERF_STATS_FILTER", filter_);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedActivationEpochPerfStats(
            const ScopedActivationEpochPerfStats &) = delete;
        ScopedActivationEpochPerfStats &operator=(
            const ScopedActivationEpochPerfStats &) = delete;

    private:
        struct SavedValue
        {
            bool present = false;
            std::string value;
        };

        /** @brief Snapshot one variable before the focused test override. */
        static void preserve(const char *name, SavedValue &saved)
        {
            if (const char *const value = ::getenv(name))
            {
                saved.present = true;
                saved.value = value;
            }
        }

        /** @brief Restore one variable exactly as the enclosing process supplied it. */
        static void restore(const char *name, const SavedValue &saved)
        {
            if (saved.present)
                ::setenv(name, saved.value.c_str(), 1);
            else
                ::unsetenv(name);
        }

        SavedValue summary_;
        SavedValue filter_;
    };

    /** @return Main-prefill graph-role bit used by the retained test family. */
    constexpr std::uint32_t prefillRoleBit() noexcept
    {
        return std::uint32_t{1}
               << static_cast<std::uint32_t>(
                      MoEOverlayInferenceGraphRole::MainPrefill);
    }

    /** @brief Poll one terminal event without synchronizing its complete device. */
    bool awaitEvent(
        IBackend *backend,
        void *event,
        std::chrono::steady_clock::duration timeout,
        int device_ordinal = 0)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (backend && event && std::chrono::steady_clock::now() < deadline)
        {
            bool ready = false;
            if (!backend->queryEvent(event, device_ordinal, &ready))
                return false;
            if (ready)
                return true;
            std::this_thread::yield();
        }
        return false;
    }

    /** @return One rank context proving that both synthetic ranks share a node. */
    std::shared_ptr<MockMPIContext> makeContext(int rank)
    {
        auto context = std::make_shared<MockMPIContext>(rank, 2);
        context->set_topology(MockMPITopology::createSimple(
            rank, /*world_size=*/2, /*ranks_per_node=*/2));
        return context;
    }

    /** @return Fixed wire workspace required by transport construction. */
    template <std::int32_t Rows>
    std::shared_ptr<MoEOverlayRankBatchWireWorkspace> makeWireWorkspace()
    {
        constexpr std::size_t entries =
            static_cast<std::size_t>(Rows * kTopK);
        return std::make_shared<MoEOverlayRankBatchWireWorkspace>(
            MoEOverlayRankBatchWireWorkspace::Config{
                .participant_ids = {kTargetParticipant},
                .max_total_rows = static_cast<std::size_t>(Rows),
                .max_total_entries = entries,
                .d_model = kDModel,
                .top_k = kTopK,
            });
    }

    /** @return Exact rank-pair transport config for one process-local endpoint. */
    template <std::int32_t Rows>
    MoEOverlayRankBatchTransportConfig makeTransportConfig(
        int local_rank,
        DeviceId local_device,
        std::string identity)
    {
        constexpr std::size_t entries =
            static_cast<std::size_t>(Rows * kTopK);
        return {
            .mpi_ctx = makeContext(local_rank),
            .source_world_rank = 0,
            .target_world_rank = 1,
            .workspace = makeWireWorkspace<Rows>(),
            .max_rows_per_participant = static_cast<std::size_t>(Rows),
            .max_entries_per_participant = entries,
            .d_model = kDModel,
            .top_k = kTopK,
            .tier_index = 3,
            .domain_ordinal = 9,
            .channel_identity = std::move(identity),
            .transaction_slot_count = 8,
            .transaction_topology = {
                .workspace_generation = 17u,
                .topology_fingerprint_low = 0x1234567812345678ull,
                .topology_fingerprint_high = 0x8765432187654321ull,
                .source_world_rank = 0,
                .target_world_rank = 1,
            },
            .source_endpoint = {
                .world_rank = 0,
                .participant_id = kSourceParticipant,
                .tier_priority = 80,
                .domain_ordinal = 4,
            },
            .target_tier_priority = 30,
            .activation_graph_families = {{
                .graph_role_mask = prefillRoleBit(),
                .model_layer_indices = std::vector<std::int32_t>(
                    kLayers.begin(), kLayers.end()),
            }},
            .local_lanes = {{
                .participant_id = kTargetParticipant,
                .device = local_device,
            }},
        };
    }

    /** @return Backend selected solely from one planner-provided device id. */
    IBackend *backendFor(DeviceId device)
    {
        if (device.is_cuda())
            return getCUDABackend();
        if (device.is_rocm())
            return getROCmBackend();
        return nullptr;
    }

    /** @return Production MoE kernel selected solely from endpoint device type. */
    std::unique_ptr<IMoEKernel> kernelFor(DeviceId device)
    {
        if (device.is_cuda())
            return std::make_unique<CUDAMoEKernel>(device.ordinal);
        if (device.is_rocm())
            return std::make_unique<ROCmMoEKernel>(device.ordinal);
        return nullptr;
    }

    /** @brief One raw allocation tracked with its exact backend owner. */
    struct DeviceAllocation
    {
        IBackend *backend = nullptr;
        void *address = nullptr;
        int ordinal = 0;
    };

    /**
     * @brief Own one role-neutral real-device sparse activation round trip.
     *
     * Teardown first publishes abort sentinels, then drains exact streams and
     * only afterward releases device pointers and mapped transport pages. This
     * keeps failed assertions from leaving a GPU wait referencing unmapped RAM.
     */
    template <std::int32_t Rows>
    class PacketRoundTrip final
    {
    public:
        static_assert(Rows > 0);
        static constexpr std::int32_t kRows = Rows;
        static constexpr std::size_t kEntries =
            static_cast<std::size_t>(Rows * kTopK);
        static constexpr bool kUsesSingleRowDirectPacket = Rows == 1;

        PacketRoundTrip(DeviceId source_device, DeviceId target_device)
            : source_device_(source_device), target_device_(target_device)
        {
        }

        PacketRoundTrip(const PacketRoundTrip &) = delete;
        PacketRoundTrip &operator=(const PacketRoundTrip &) = delete;

        ~PacketRoundTrip()
        {
            publishAbort();
            drain(source_backend_, source_stream_, source_terminal_);
            drain(target_backend_, target_stream_, target_terminal_);
            source_cache_.reset(
                DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::
                    Destroy);
            target_cache_.reset(
                DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::
                    Destroy);
            source_graph_.reset();
            target_graph_.reset();
            source_capture_.reset();
            target_capture_.reset();
            source_dispatch_captures_.clear();
            source_return_captures_.clear();
            target_compute_captures_.clear();
            source_dispatch_graphs_.clear();
            source_return_graphs_.clear();
            target_compute_graphs_.clear();
            source_executor_.reset();
            target_executor_.reset();
            source_arena_.reset();
            target_arena_.reset();
            source_execution_context_.reset();
            target_execution_context_.reset();
            source_hidden_tensor_.reset();
            source_routing_weights_tensor_.reset();
            source_canonical_route_tensor_.reset();
            source_zero_tensor_.reset();
            target_hidden_tensor_.reset();
            target_routing_indices_tensor_.reset();
            target_routing_weights_tensor_.reset();
            target_route_input_tensor_.reset();
            target_canonical_route_tensor_.reset();
            for (const auto &allocation : allocations_)
            {
                if (allocation.backend && allocation.address)
                    allocation.backend->free(
                        allocation.address, allocation.ordinal);
            }
            allocations_.clear();
            if (source_backend_ && source_terminal_)
                source_backend_->destroyEvent(source_terminal_, 0);
            if (target_backend_ && target_terminal_)
                target_backend_->destroyEvent(target_terminal_, 0);
            if (source_backend_ && source_stream_)
                source_backend_->destroyStream(source_stream_, 0);
            if (target_backend_ && target_stream_)
                target_backend_->destroyStream(target_stream_, 0);
            source_transport_.reset();
            target_transport_.reset();
        }

        /** @brief Build planner-derived transports, kernels, and persistent buffers. */
        bool initialize()
        {
            source_backend_ = backendFor(source_device_);
            target_backend_ = backendFor(target_device_);
            if (!source_backend_ || !target_backend_ ||
                source_backend_->deviceCount() < 1 ||
                target_backend_->deviceCount() < 1)
            {
                error_ = "required endpoint backend/device is unavailable";
                return false;
            }

            static std::atomic<std::uint64_t> identity_counter{0u};
            const std::string identity =
                "mapped_activation_packet_pid" +
                std::to_string(static_cast<unsigned long>(::getpid())) + "_" +
                std::to_string(identity_counter.fetch_add(1u));
            try
            {
                /* The production constructors rendezvous before either GPU
                 * driver may pin the full shared mapping. Build these mocked
                 * ranks concurrently to preserve that setup protocol. */
                std::exception_ptr source_error;
                std::exception_ptr target_error;
                std::thread source_builder([&]
                                           {
                                               try
                                               {
                                                   source_transport_ = std::make_unique<
                                                       MoEOverlayNodeLocalRankBatchTransport>(
                                                       makeTransportConfig<Rows>(
                                                           /*local_rank=*/0,
                                                           source_device_,
                                                           identity));
                                               }
                                               catch (...)
                                               {
                                                   source_error =
                                                       std::current_exception();
                                               }
                                           });
                std::thread target_builder([&]
                                           {
                                               try
                                               {
                                                   target_transport_ = std::make_unique<
                                                       MoEOverlayNodeLocalRankBatchTransport>(
                                                       makeTransportConfig<Rows>(
                                                           /*local_rank=*/1,
                                                           target_device_,
                                                           identity));
                                               }
                                               catch (...)
                                               {
                                                   target_error =
                                                       std::current_exception();
                                               }
                                           });
                source_builder.join();
                target_builder.join();
                if (source_error)
                    std::rethrow_exception(source_error);
                if (target_error)
                    std::rethrow_exception(target_error);
                source_lane_ = source_transport_->activationDeviceLane(
                    kTargetParticipant, /*graph_family_ordinal=*/0u,
                    source_device_);
                target_lane_ = target_transport_->activationDeviceLane(
                    kTargetParticipant, /*graph_family_ordinal=*/0u,
                    target_device_);
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }

            source_stream_ = source_backend_->createStream(0);
            target_stream_ = target_backend_->createStream(0);
            source_terminal_ = source_backend_->createEvent(0);
            target_terminal_ = target_backend_->createEvent(0);
            if (!source_stream_ || !target_stream_ || !source_terminal_ ||
                !target_terminal_)
            {
                error_ = "could not create exact endpoint streams or events";
                return false;
            }

            source_placement_banks_ = allocate(
                source_backend_,
                kDeviceMoEOverlayEpochBankCount *
                    sizeof(DeviceMoEPlacementBank));
            source_placement_ticket_ = allocate(
                source_backend_, sizeof(DeviceMoEOverlayEpochTicket));
            source_placement_status_ = allocate(
                source_backend_, sizeof(DeviceMoEOverlayEpochStatus));
            target_placement_banks_ = allocate(
                target_backend_,
                kDeviceMoEOverlayEpochBankCount *
                    sizeof(DeviceMoEPlacementBank));
            target_placement_ticket_ = allocate(
                target_backend_, sizeof(DeviceMoEOverlayEpochTicket));
            target_placement_status_ = allocate(
                target_backend_, sizeof(DeviceMoEOverlayEpochStatus));
            source_active_rows_ = allocate(source_backend_, sizeof(std::int32_t));
            target_active_rows_ = allocate(target_backend_, sizeof(std::int32_t));
            if (!allAllocationsValid())
            {
                error_ = "could not allocate persistent endpoint metadata buffers";
                return false;
            }
            source_hidden_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    static_cast<std::size_t>(kRows),
                    static_cast<std::size_t>(kDModel)});
            source_routing_indices_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    static_cast<std::size_t>(kRows),
                    static_cast<std::size_t>(kTopK)});
            source_routing_weights_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    static_cast<std::size_t>(kRows),
                    static_cast<std::size_t>(kTopK)});
            source_canonical_route_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    kEntries,
                    static_cast<std::size_t>(kDModel)});
            source_zero_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    kEntries,
                    static_cast<std::size_t>(kDModel)});
            target_hidden_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    static_cast<std::size_t>(kRows),
                    static_cast<std::size_t>(kDModel)});
            target_routing_indices_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    static_cast<std::size_t>(kRows),
                    static_cast<std::size_t>(kTopK)});
            target_routing_weights_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    static_cast<std::size_t>(kRows),
                    static_cast<std::size_t>(kTopK)});
            target_route_input_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    kEntries,
                    static_cast<std::size_t>(kDModel)});
            target_canonical_route_tensor_ = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{
                    kEntries,
                    static_cast<std::size_t>(kDModel)});
            return initializeSourcePayload() && initializeProtocol() &&
                   captureDeviceOwnedGraphs();
        }

        /**
         * @brief Materialize the same complete endpoint graphs used in production.
         *
         * Packet stages append admission, dispatch, and return timeline nodes
         * directly to an active full-graph capture. The ordinary executor owns
         * cold capture, instantiation, transaction-zero state, and steady replay;
         * there are no child executables or test-only parent composer in this
         * proof.
         */
        bool captureDeviceOwnedGraphs()
        {
            source_execution_context_ = IDeviceContext::create(source_device_);
            target_execution_context_ = IDeviceContext::create(target_device_);
            source_arena_ = std::make_unique<BufferArena>();
            target_arena_ = std::make_unique<BufferArena>();
            if (!source_execution_context_ || !target_execution_context_ ||
                !source_arena_ || !target_arena_ ||
                !source_arena_->registerExternalBuffer(
                    BufferId::NORMALIZED, source_hidden_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_INDICES,
                    source_routing_indices_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_WEIGHTS,
                    source_routing_weights_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                    source_canonical_route_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::RESIDUAL, source_zero_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::NORMALIZED, target_hidden_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_INDICES,
                    target_routing_indices_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_WEIGHTS,
                    target_routing_weights_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::RESIDUAL,
                    target_route_input_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                    target_canonical_route_tensor_.get()))
            {
                error_ =
                    "could not bind production packet tensors into exact-device arenas";
                return false;
            }

            auto *const source_worker =
                &GPUDeviceContextPool::instance().getContext(source_device_);
            auto *const target_worker =
                &GPUDeviceContextPool::instance().getContext(target_device_);
            GraphExecutorConfig source_executor_config;
            source_executor_config.default_device = source_device_;
            source_executor_config.worker_gpu_context_resolver =
                [source_worker, expected = source_device_](DeviceId requested)
                    -> IWorkerGPUContext *
            {
                return requested == expected ? source_worker : nullptr;
            };
            source_executor_config.worker_gpu_context_uses_process_pool = false;
            GraphExecutorConfig target_executor_config;
            target_executor_config.default_device = target_device_;
            target_executor_config.worker_gpu_context_resolver =
                [target_worker, expected = target_device_](DeviceId requested)
                    -> IWorkerGPUContext *
            {
                return requested == expected ? target_worker : nullptr;
            };
            target_executor_config.worker_gpu_context_uses_process_pool = false;
            source_executor_ = std::make_unique<DeviceGraphExecutor>(
                std::move(source_executor_config));
            target_executor_ = std::make_unique<DeviceGraphExecutor>(
                std::move(target_executor_config));
            source_executor_->setArena(source_arena_.get());
            target_executor_->setArena(target_arena_.get());

            source_graph_ = std::make_unique<ComputeGraph>();
            target_graph_ = std::make_unique<ComputeGraph>();
            source_graph_->setNativeCaptureEnvelope(
                GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
            target_graph_->setNativeCaptureEnvelope(
                GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);

            ResidualAddStage::Params clear_params;
            clear_params.device_id = source_device_;
            clear_params.input = source_zero_tensor_.get();
            clear_params.output = source_canonical_route_tensor_.get();
            clear_params.num_elements =
                kEntries * static_cast<std::size_t>(kDModel);
            clear_params.input_buffer_id = BufferId::RESIDUAL;
            clear_params.output_buffer_id =
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
            const std::string clear_name = "source_clear_canonical_routes";
            source_graph_->addNode(
                clear_name,
                ComputeStageFactory::createResidualAdd(clear_params),
                source_device_);

            std::string source_dependency = clear_name;
            std::string target_dependency;
            for (std::size_t stage_index = 0u;
                 stage_index < kLayers.size();
                 ++stage_index)
            {
                const auto ordinal =
                    static_cast<std::uint32_t>(stage_index);
                const std::int32_t layer = kLayers[stage_index];

                const MoEOverlayRoutePlacementDeviceBinding placement{
                    .banks = {placementBankView(0u), placementBankView(1u)},
                    .ticket = static_cast<
                        const DeviceMoEOverlayEpochTicket *>(
                        source_placement_ticket_),
                    .status = static_cast<
                        const DeviceMoEOverlayEpochStatus *>(
                        source_placement_status_),
                    .expert_count = kExpertCount,
                };
                const std::string dispatch_name =
                    "source_dispatch_pack_" +
                    std::to_string(stage_index);
                std::shared_ptr<MoEOverlayActivationLaneBatchState>
                    lane_transaction;
                if constexpr (Rows > 1)
                {
                    /* Production prefill forks every remote lane onto a
                     * persistent side stream. A one-lane fixture still proves
                     * the exact event fork/import/join lifecycle without
                     * duplicating the planner or transport in test code. */
                    lane_transaction = std::make_shared<
                        MoEOverlayActivationLaneBatchState>(
                        source_device_,
                        std::vector<MoEOverlayMappedActivationDeviceLane>{
                            source_lane_},
                        kRows,
                        ordinal,
                        layer);
                    MoEOverlayActivationDispatchPackBatchStage::Params
                        dispatch_params;
                    dispatch_params.device_id = source_device_;
                    dispatch_params.transaction = lane_transaction;
                    dispatch_params.hidden = source_hidden_tensor_.get();
                    dispatch_params.routing_indices =
                        source_routing_indices_tensor_.get();
                    dispatch_params.routing_weights =
                        source_routing_weights_tensor_.get();
                    dispatch_params.placement = placement;
                    dispatch_params.active_row_count_device =
                        static_cast<const std::int32_t *>(source_active_rows_);
                    dispatch_params.hidden_buffer_id = BufferId::NORMALIZED;
                    dispatch_params.routing_indices_buffer_id =
                        BufferId::MOE_EXPERT_INDICES;
                    dispatch_params.routing_weights_buffer_id =
                        BufferId::MOE_EXPERT_WEIGHTS;
                    source_graph_->addNode(
                        dispatch_name,
                        ComputeStageFactory::
                            createMoEOverlayActivationDispatchPackBatch(
                                dispatch_params),
                        source_device_);
                }
                else
                {
                    MoEOverlayActivationDispatchPackStage::Params
                        dispatch_params;
                    dispatch_params.device_id = source_device_;
                    dispatch_params.lane = source_lane_;
                    dispatch_params.hidden = source_hidden_tensor_.get();
                    dispatch_params.routing_indices =
                        source_routing_indices_tensor_.get();
                    dispatch_params.routing_weights =
                        source_routing_weights_tensor_.get();
                    dispatch_params.placement = placement;
                    dispatch_params.active_row_count_device =
                        static_cast<const std::int32_t *>(source_active_rows_);
                    dispatch_params.hidden_buffer_id = BufferId::NORMALIZED;
                    dispatch_params.routing_indices_buffer_id =
                        BufferId::MOE_EXPERT_INDICES;
                    dispatch_params.routing_weights_buffer_id =
                        BufferId::MOE_EXPERT_WEIGHTS;
                    dispatch_params.physical_rows = kRows;
                    dispatch_params.stage_ordinal = ordinal;
                    dispatch_params.model_layer_index = layer;
                    source_graph_->addNode(
                        dispatch_name,
                        ComputeStageFactory::
                            createMoEOverlayActivationDispatchPack(
                                dispatch_params),
                        source_device_);
                }
                source_graph_->addDependency(
                    dispatch_name, source_dependency);

                const std::string return_consume_name =
                    "source_return_consume_" +
                    std::to_string(stage_index);
                if constexpr (Rows > 1)
                {
                    MoEOverlayActivationReturnConsumeBatchStage::Params
                        return_consume_params;
                    return_consume_params.device_id = source_device_;
                    return_consume_params.transaction = lane_transaction;
                    return_consume_params.canonical_route_contributions =
                        source_canonical_route_tensor_.get();
                    return_consume_params
                        .canonical_route_contributions_buffer_id =
                        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
                    source_graph_->addNode(
                        return_consume_name,
                        ComputeStageFactory::
                            createMoEOverlayActivationReturnConsumeBatch(
                                return_consume_params),
                        source_device_);
                }
                else
                {
                    MoEOverlayActivationReturnConsumeStage::Params
                        return_consume_params;
                    return_consume_params.device_id = source_device_;
                    return_consume_params.lane = source_lane_;
                    return_consume_params.canonical_route_contributions =
                        source_canonical_route_tensor_.get();
                    return_consume_params.physical_rows = kRows;
                    return_consume_params.stage_ordinal = ordinal;
                    return_consume_params.model_layer_index = layer;
                    source_graph_->addNode(
                        return_consume_name,
                        ComputeStageFactory::
                            createMoEOverlayActivationReturnConsume(
                                return_consume_params),
                        source_device_);
                }
                source_graph_->addDependency(
                    return_consume_name, dispatch_name);
                source_dependency = return_consume_name;

                MoEOverlayActivationDispatchConsumeStage::Params
                    consume_params;
                consume_params.device_id = target_device_;
                consume_params.lane = target_lane_;
                consume_params.placement = {
                    .banks = {
                        placementBankView(target_placement_banks_, 0u),
                        placementBankView(target_placement_banks_, 1u),
                    },
                    .ticket = static_cast<
                        const DeviceMoEOverlayEpochTicket *>(
                            target_placement_ticket_),
                    .status = static_cast<
                        const DeviceMoEOverlayEpochStatus *>(
                            target_placement_status_),
                    .expert_count = kExpertCount,
                };
                consume_params.hidden = target_hidden_tensor_.get();
                consume_params.routing_indices =
                    target_routing_indices_tensor_.get();
                consume_params.routing_weights =
                    target_routing_weights_tensor_.get();
                consume_params.active_row_count_device =
                    static_cast<std::int32_t *>(target_active_rows_);
                consume_params.physical_rows = kRows;
                consume_params.stage_ordinal = ordinal;
                consume_params.model_layer_index = layer;
                const std::string consume_name =
                    "target_dispatch_consume_" +
                    std::to_string(stage_index);
                target_graph_->addNode(
                    consume_name,
                    ComputeStageFactory::
                        createMoEOverlayActivationDispatchConsume(
                            consume_params),
                    target_device_);
                if (!target_dependency.empty())
                {
                    target_graph_->addDependency(
                        consume_name, target_dependency);
                }

                ResidualAddStage::Params device_work_params;
                device_work_params.device_id = target_device_;
                device_work_params.input = target_route_input_tensor_.get();
                device_work_params.output =
                    target_canonical_route_tensor_.get();
                device_work_params.num_elements =
                    kEntries * static_cast<std::size_t>(kDModel);
                device_work_params.input_buffer_id = BufferId::RESIDUAL;
                device_work_params.output_buffer_id =
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
                const std::string work_name =
                    "target_device_work_" +
                    std::to_string(stage_index);
                target_graph_->addNode(
                    work_name,
                    ComputeStageFactory::createResidualAdd(
                        device_work_params),
                    target_device_);
                target_graph_->addDependency(work_name, consume_name);

                MoEOverlayActivationReturnPackStage::Params
                    return_pack_params;
                return_pack_params.device_id = target_device_;
                return_pack_params.lane = target_lane_;
                return_pack_params.local_canonical_route_contributions =
                    target_canonical_route_tensor_.get();
                return_pack_params.physical_rows = kRows;
                return_pack_params.stage_ordinal = ordinal;
                return_pack_params.model_layer_index = layer;
                const std::string return_pack_name =
                    "target_return_pack_" +
                    std::to_string(stage_index);
                target_graph_->addNode(
                    return_pack_name,
                    ComputeStageFactory::
                        createMoEOverlayActivationReturnPack(
                            return_pack_params),
                    target_device_);
                target_graph_->addDependency(
                    return_pack_name, work_name);
                target_dependency = return_pack_name;
            }

            source_cache_.perf_context =
                "mapped_activation_packet_continuation";
            target_cache_.perf_context =
                "mapped_activation_packet_follower";
            source_cache_.steady_replay_host_policy =
                DeviceGraphExecutor::GraphSegmentCache::
                    SteadyReplayHostPolicy::RetainedFullGraph;
            target_cache_.steady_replay_host_policy =
                DeviceGraphExecutor::GraphSegmentCache::
                    SteadyReplayHostPolicy::RetainedFullGraph;
            if (!source_cache_.bindBorrowedCaptureStream(
                    source_worker,
                    source_stream_,
                    source_device_,
                    /*context_from_process_pool=*/false) ||
                !target_cache_.bindBorrowedCaptureStream(
                    target_worker,
                    target_stream_,
                    target_device_,
                    /*context_from_process_pool=*/false))
            {
                error_ = "could not bind exact production endpoint streams";
                return false;
            }

            source_graph_->reset();
            target_graph_->reset();
            const auto materialize = [](
                                         DeviceGraphExecutor &executor,
                                         ComputeGraph &graph,
                                         IDeviceContext *context,
                                         DeviceGraphExecutor::GraphSegmentCache &cache,
                                         void *stream,
                                         IWorkerGPUContext *worker)
            {
                return executor.executeWithCachedGraphReplay(
                    graph,
                    context,
                    cache,
                    stream,
                    worker,
                    /*collective_nodes=*/nullptr,
                    /*collectives_graph_capturable=*/false,
                    /*force_recapture=*/false,
                    /*defer_final_sync=*/true,
                    {},
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        RequireFullGraph,
                    {},
                    {},
                    {},
                    DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                        MaterializeWithoutLaunch);
            };
            if (!materialize(
                    *source_executor_,
                    *source_graph_,
                    source_execution_context_.get(),
                    source_cache_,
                    source_stream_,
                    source_worker) ||
                !materialize(
                    *target_executor_,
                    *target_graph_,
                    target_execution_context_.get(),
                    target_cache_,
                    target_stream_,
                    target_worker) ||
                !source_cache_.retained_full_graph_replay.valid() ||
                !target_cache_.retained_full_graph_replay.valid() ||
                source_cache_.executable_submission_state !=
                    DeviceGraphExecutor::GraphSegmentCache::
                        ExecutableSubmissionState::MaterializedUnlaunched ||
                target_cache_.executable_submission_state !=
                    DeviceGraphExecutor::GraphSegmentCache::
                        ExecutableSubmissionState::MaterializedUnlaunched)
            {
                error_ =
                    "production executor did not materialize both endpoint graphs without launching transaction zero";
                return false;
            }

            const IGPUGraphCapture *const source_capture =
                captureFrom(source_cache_);
            const IGPUGraphCapture *const target_capture =
                captureFrom(target_cache_);
            /* Multi-row dispatch publishes one shared physical matrix from the
             * source, while followers and compact returns access the mapping
             * directly. HIP exposes every root node, so these exact totals
             * prove that no per-follower import/export bounce node survived. */
            constexpr std::size_t source_node_count =
                kUsesSingleRowDirectPacket
                    ? 2u + 2u * kLayers.size()
                    : 2u + 6u * kLayers.size();
            constexpr std::size_t target_node_count =
                kUsesSingleRowDirectPacket
                    ? 1u + 3u * kLayers.size()
                    : 1u + 7u * kLayers.size();
            constexpr std::size_t native_wait_nodes =
                kUsesSingleRowDirectPacket ? 1u : kLayers.size() + 1u;
            if (!source_capture || !target_capture ||
                !validateROCmTimelineLowering(
                    *source_capture,
                    source_device_,
                    source_node_count,
                    native_wait_nodes,
                    "continuation") ||
                !validateROCmTimelineLowering(
                    *target_capture,
                    target_device_,
                    target_node_count,
                    native_wait_nodes,
                    "follower"))
            {
                return false;
            }

            const auto &captured_control =
                source_transport_->activationEpochControl(
                    kTargetParticipant, 0u);
            for (std::uint32_t bank = 0u;
                 bank < kMoEOverlayActivationBufferCount;
                 ++bank)
            {
                if (captured_control.buffers[bank].dispatch_signal.value != 0u ||
                    captured_control.buffers[bank].return_signal.value != 0u)
                {
                    error_ =
                        "setup materialization executed a mapped timeline publication";
                    return false;
                }
            }
            return true;
        }

        /** @brief Capture both complete endpoint transactions exactly once. */
        bool captureRetainedGraphs()
        {
            try
            {
                auto &source_context = GPUDeviceContextPool::instance()
                                           .getContext(source_device_);
                auto &target_context = GPUDeviceContextPool::instance()
                                           .getContext(target_device_);
                source_dispatch_captures_.reserve(kLayers.size());
                source_return_captures_.reserve(kLayers.size());
                target_compute_captures_.reserve(kLayers.size());
                for (std::size_t stage = 0; stage < kLayers.size(); ++stage)
                {
                    source_dispatch_captures_.push_back(
                        source_context.createGraphCapture(source_stream_));
                    source_return_captures_.push_back(
                        source_context.createGraphCapture(source_stream_));
                    target_compute_captures_.push_back(
                        target_context.createGraphCapture(target_stream_));
                }
                source_capture_ =
                    source_context.createGraphCapture(source_stream_);
                target_capture_ =
                    target_context.createGraphCapture(target_stream_);
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }
            const auto capturesComplete = [](const auto &captures)
            {
                return std::all_of(
                    captures.begin(), captures.end(),
                    [](const auto &capture) { return capture != nullptr; });
            };
            if (source_dispatch_captures_.size() != kLayers.size() ||
                source_return_captures_.size() != kLayers.size() ||
                target_compute_captures_.size() != kLayers.size() ||
                !capturesComplete(source_dispatch_captures_) ||
                !capturesComplete(source_return_captures_) ||
                !capturesComplete(target_compute_captures_) ||
                !source_capture_ || !target_capture_)
            {
                error_ = "could not create retained endpoint graph fragments";
                return false;
            }

            source_execution_context_ =
                IDeviceContext::create(source_device_);
            target_execution_context_ =
                IDeviceContext::create(target_device_);
            source_arena_ = std::make_unique<BufferArena>();
            target_arena_ = std::make_unique<BufferArena>();
            if (!source_execution_context_ || !target_execution_context_ ||
                !source_arena_ || !target_arena_ ||
                !source_arena_->registerExternalBuffer(
                    BufferId::NORMALIZED, source_hidden_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_INDICES,
                    source_routing_indices_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_WEIGHTS,
                    source_routing_weights_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                    source_canonical_route_tensor_.get()) ||
                !source_arena_->registerExternalBuffer(
                    BufferId::RESIDUAL, source_zero_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::NORMALIZED, target_hidden_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_INDICES,
                    target_routing_indices_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::MOE_EXPERT_WEIGHTS,
                    target_routing_weights_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::RESIDUAL,
                    target_route_input_tensor_.get()) ||
                !target_arena_->registerExternalBuffer(
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                    target_canonical_route_tensor_.get()))
            {
                error_ = "could not bind typed packet tensors into exact-device arenas";
                return false;
            }
            auto *const source_worker =
                &GPUDeviceContextPool::instance().getContext(source_device_);
            auto *const target_worker =
                &GPUDeviceContextPool::instance().getContext(target_device_);
            GraphExecutorConfig source_executor_config;
            source_executor_config.default_device = source_device_;
            source_executor_config.worker_gpu_context_resolver =
                [source_worker,
                 expected = source_device_](DeviceId requested)
                    -> IWorkerGPUContext *
                {
                    return requested == expected ? source_worker : nullptr;
                };
            source_executor_config.worker_gpu_context_uses_process_pool = false;
            GraphExecutorConfig target_executor_config;
            target_executor_config.default_device = target_device_;
            target_executor_config.worker_gpu_context_resolver =
                [target_worker,
                 expected = target_device_](DeviceId requested)
                    -> IWorkerGPUContext *
                {
                    return requested == expected ? target_worker : nullptr;
                };
            target_executor_config.worker_gpu_context_uses_process_pool = false;
            source_executor_ = std::make_unique<DeviceGraphExecutor>(
                std::move(source_executor_config));
            target_executor_ = std::make_unique<DeviceGraphExecutor>(
                std::move(target_executor_config));
            source_executor_->setArena(source_arena_.get());
            target_executor_->setArena(target_arena_.get());

            struct SemanticCaptureUnit
            {
                const IGPUGraphCapture *capture = nullptr;
                std::vector<std::string> stage_names;
            };
            ComputeGraph source_semantic_graph;
            ComputeGraph target_semantic_graph;
            std::vector<SemanticCaptureUnit> source_semantic_units;
            std::vector<SemanticCaptureUnit> target_semantic_units;
            source_semantic_units.reserve(kLayers.size() * 2u);
            target_semantic_units.reserve(kLayers.size());
            for (std::size_t stage_index = 0;
                 stage_index < kLayers.size(); ++stage_index)
            {
                const auto ordinal = static_cast<std::uint32_t>(stage_index);
                const std::int32_t layer = kLayers[stage_index];
                auto &dispatch_capture =
                    source_dispatch_captures_[stage_index];
                auto dispatch_graph = std::make_unique<ComputeGraph>();
                std::string dispatch_dependency;
                std::vector<std::string> semantic_dispatch_stages;
                if (stage_index == 0u)
                {
                    ResidualAddStage::Params clear_params;
                    clear_params.device_id = source_device_;
                    clear_params.input = source_zero_tensor_.get();
                    clear_params.output = source_canonical_route_tensor_.get();
                    clear_params.num_elements =
                        kEntries * static_cast<std::size_t>(kDModel);
                    clear_params.input_buffer_id = BufferId::RESIDUAL;
                    clear_params.output_buffer_id =
                        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
                    dispatch_dependency = "clear_canonical_routes";
                    dispatch_graph->addNode(
                        dispatch_dependency,
                        ComputeStageFactory::createResidualAdd(clear_params),
                        source_device_);
                    const std::string semantic_clear_name =
                        "source_clear_canonical_routes";
                    source_semantic_graph.addNode(
                        semantic_clear_name,
                        ComputeStageFactory::createResidualAdd(clear_params),
                        source_device_);
                    semantic_dispatch_stages.push_back(
                        semantic_clear_name);
                }
                MoEOverlayActivationDispatchPackStage::Params dispatch_params;
                dispatch_params.device_id = source_device_;
                dispatch_params.lane = source_lane_;
                dispatch_params.hidden = source_hidden_tensor_.get();
                dispatch_params.routing_indices =
                    source_routing_indices_tensor_.get();
                dispatch_params.routing_weights =
                    source_routing_weights_tensor_.get();
                dispatch_params.placement = {
                    .banks = {
                        placementBankView(0u),
                        placementBankView(1u),
                    },
                    .ticket =
                        static_cast<const DeviceMoEOverlayEpochTicket *>(
                            source_placement_ticket_),
                    .status =
                        static_cast<const DeviceMoEOverlayEpochStatus *>(
                            source_placement_status_),
                    .expert_count = kExpertCount,
                };
                dispatch_params.active_row_count_device =
                    static_cast<const std::int32_t *>(source_active_rows_);
                dispatch_params.hidden_buffer_id = BufferId::NORMALIZED;
                dispatch_params.routing_indices_buffer_id =
                    BufferId::MOE_EXPERT_INDICES;
                dispatch_params.routing_weights_buffer_id =
                    BufferId::MOE_EXPERT_WEIGHTS;
                dispatch_params.physical_rows = kRows;
                dispatch_params.stage_ordinal = ordinal;
                dispatch_params.model_layer_index = layer;
                const std::string dispatch_name = "dispatch_pack";
                dispatch_graph->addNode(
                    dispatch_name,
                    ComputeStageFactory::
                        createMoEOverlayActivationDispatchPack(
                            dispatch_params),
                    source_device_);
                if (!dispatch_dependency.empty())
                {
                    dispatch_graph->addDependency(
                        dispatch_name, dispatch_dependency);
                }
                if (!source_executor_->captureRetainedGraphFragment(
                        *dispatch_graph,
                        source_execution_context_.get(),
                        dispatch_capture.get(),
                        "typed_continuation_dispatch_pack"))
                {
                    error_ = "typed continuation dispatch fragment capture failed";
                    return false;
                }
                const std::string semantic_dispatch_name =
                    "source_dispatch_pack_" +
                    std::to_string(stage_index);
                source_semantic_graph.addNode(
                    semantic_dispatch_name,
                    ComputeStageFactory::
                        createMoEOverlayActivationDispatchPack(
                            dispatch_params),
                    source_device_);
                if (!semantic_dispatch_stages.empty())
                {
                    source_semantic_graph.addDependency(
                        semantic_dispatch_name,
                        semantic_dispatch_stages.back());
                }
                semantic_dispatch_stages.push_back(
                    semantic_dispatch_name);
                source_semantic_units.push_back({
                    .capture = dispatch_capture.get(),
                    .stage_names = semantic_dispatch_stages,
                });

                auto &target_capture =
                    target_compute_captures_[stage_index];
                auto target_graph = std::make_unique<ComputeGraph>();
                MoEOverlayActivationDispatchConsumeStage::Params
                    consume_params;
                consume_params.device_id = target_device_;
                consume_params.lane = target_lane_;
                consume_params.placement = {
                    .banks = {
                        placementBankView(target_placement_banks_, 0u),
                        placementBankView(target_placement_banks_, 1u),
                    },
                    .ticket = static_cast<
                        const DeviceMoEOverlayEpochTicket *>(
                            target_placement_ticket_),
                    .status = static_cast<
                        const DeviceMoEOverlayEpochStatus *>(
                            target_placement_status_),
                    .expert_count = kExpertCount,
                };
                consume_params.hidden = target_hidden_tensor_.get();
                consume_params.routing_indices =
                    target_routing_indices_tensor_.get();
                consume_params.routing_weights =
                    target_routing_weights_tensor_.get();
                consume_params.active_row_count_device =
                    static_cast<std::int32_t *>(target_active_rows_);
                consume_params.physical_rows = kRows;
                consume_params.stage_ordinal = ordinal;
                consume_params.model_layer_index = layer;
                target_graph->addNode(
                    "dispatch_consume",
                    ComputeStageFactory::
                        createMoEOverlayActivationDispatchConsume(
                            consume_params),
                    target_device_);

                ResidualAddStage::Params device_work_params;
                device_work_params.device_id = target_device_;
                device_work_params.input = target_route_input_tensor_.get();
                device_work_params.output =
                    target_canonical_route_tensor_.get();
                device_work_params.num_elements =
                    kEntries * static_cast<std::size_t>(kDModel);
                device_work_params.input_buffer_id = BufferId::RESIDUAL;
                device_work_params.output_buffer_id =
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
                target_graph->addNode(
                    "device_work",
                    ComputeStageFactory::createResidualAdd(
                        device_work_params),
                    target_device_);
                target_graph->addDependency(
                    "device_work", "dispatch_consume");

                MoEOverlayActivationReturnPackStage::Params return_pack_params;
                return_pack_params.device_id = target_device_;
                return_pack_params.lane = target_lane_;
                return_pack_params.local_canonical_route_contributions =
                    target_canonical_route_tensor_.get();
                return_pack_params.physical_rows = kRows;
                return_pack_params.stage_ordinal = ordinal;
                return_pack_params.model_layer_index = layer;
                target_graph->addNode(
                    "return_pack",
                    ComputeStageFactory::
                        createMoEOverlayActivationReturnPack(
                            return_pack_params),
                    target_device_);
                target_graph->addDependency(
                    "return_pack", "device_work");
                if (!target_executor_->captureRetainedGraphFragment(
                        *target_graph,
                        target_execution_context_.get(),
                        target_capture.get(),
                        "typed_follower_packet_compute"))
                {
                    error_ = "typed follower packet fragment capture failed";
                    return false;
                }
                const std::string semantic_consume_name =
                    "target_dispatch_consume_" +
                    std::to_string(stage_index);
                const std::string semantic_work_name =
                    "target_device_work_" +
                    std::to_string(stage_index);
                const std::string semantic_return_pack_name =
                    "target_return_pack_" +
                    std::to_string(stage_index);
                target_semantic_graph.addNode(
                    semantic_consume_name,
                    ComputeStageFactory::
                        createMoEOverlayActivationDispatchConsume(
                            consume_params),
                    target_device_);
                target_semantic_graph.addNode(
                    semantic_work_name,
                    ComputeStageFactory::createResidualAdd(
                        device_work_params),
                    target_device_);
                target_semantic_graph.addDependency(
                    semantic_work_name,
                    semantic_consume_name);
                target_semantic_graph.addNode(
                    semantic_return_pack_name,
                    ComputeStageFactory::
                        createMoEOverlayActivationReturnPack(
                            return_pack_params),
                    target_device_);
                target_semantic_graph.addDependency(
                    semantic_return_pack_name,
                    semantic_work_name);
                target_semantic_units.push_back({
                    .capture = target_capture.get(),
                    .stage_names = {
                        semantic_consume_name,
                        semantic_work_name,
                        semantic_return_pack_name,
                    },
                });

                auto &return_capture =
                    source_return_captures_[stage_index];
                auto return_graph = std::make_unique<ComputeGraph>();
                MoEOverlayActivationReturnConsumeStage::Params
                    return_consume_params;
                return_consume_params.device_id = source_device_;
                return_consume_params.lane = source_lane_;
                return_consume_params.canonical_route_contributions =
                    source_canonical_route_tensor_.get();
                return_consume_params.physical_rows = kRows;
                return_consume_params.stage_ordinal = ordinal;
                return_consume_params.model_layer_index = layer;
                return_graph->addNode(
                    "return_consume",
                    ComputeStageFactory::
                        createMoEOverlayActivationReturnConsume(
                            return_consume_params),
                    source_device_);
                if (!source_executor_->captureRetainedGraphFragment(
                        *return_graph,
                        source_execution_context_.get(),
                        return_capture.get(),
                        "typed_continuation_return_consume"))
                {
                    error_ = "typed continuation return fragment capture failed";
                    return false;
                }
                const std::string semantic_return_name =
                    "source_return_consume_" +
                    std::to_string(stage_index);
                source_semantic_graph.addNode(
                    semantic_return_name,
                    ComputeStageFactory::
                        createMoEOverlayActivationReturnConsume(
                            return_consume_params),
                    source_device_);
                source_semantic_units.push_back({
                    .capture = return_capture.get(),
                    .stage_names = {semantic_return_name},
                });
                source_dispatch_graphs_.push_back(
                    std::move(dispatch_graph));
                target_compute_graphs_.push_back(std::move(target_graph));
                source_return_graphs_.push_back(std::move(return_graph));
            }
            try
            {
                std::vector<MoEOverlayRetainedCaptureUnit> source_units;
                std::vector<MoEOverlayRetainedCaptureUnit> target_units;
                source_units.reserve(source_semantic_units.size());
                target_units.reserve(target_semantic_units.size());
                for (const auto &unit : source_semantic_units)
                {
                    source_units.push_back({
                        .capture = unit.capture,
                        .stage_names = unit.stage_names,
                    });
                }
                for (const auto &unit : target_semantic_units)
                {
                    target_units.push_back({
                        .capture = unit.capture,
                        .stage_names = unit.stage_names,
                    });
                }
                MoEOverlayRetainedActivationTransaction::
                    buildContinuationFromCapturedUnits(
                        *source_capture_,
                        source_semantic_graph,
                        source_units);
                MoEOverlayRetainedActivationTransaction::
                    buildFollowerFromCapturedUnits(
                        *target_capture_,
                        target_semantic_graph,
                        target_units);
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }
            const auto totalFragmentNodes = [](const auto &captures)
            {
                std::size_t total = 0u;
                for (const auto &capture : captures)
                    total += capture ? capture->nodeCount() : 0u;
                return total;
            };
            const std::size_t native_timeline_nodes =
                1u + 2u * kLayers.size();
            if (!validateROCmTimelineLowering(
                    *source_capture_,
                    source_device_,
                    totalFragmentNodes(source_dispatch_captures_) +
                        totalFragmentNodes(source_return_captures_) +
                        native_timeline_nodes,
                    /*expected_wait_nodes=*/kLayers.size() + 1u,
                    "continuation") ||
                !validateROCmTimelineLowering(
                    *target_capture_,
                    target_device_,
                    totalFragmentNodes(target_compute_captures_) +
                        native_timeline_nodes,
                    /*expected_wait_nodes=*/kLayers.size() + 1u,
                    "follower"))
            {
                return false;
            }
            if (!target_capture_->instantiate() ||
                !source_capture_->instantiate())
            {
                error_ = "retained endpoint parent graph instantiation failed";
                return false;
            }
            const auto &captured_control =
                source_transport_->activationEpochControl(
                    kTargetParticipant, 0u);
            for (std::uint32_t bank = 0u;
                 bank < kMoEOverlayActivationBufferCount;
                 ++bank)
            {
                if (captured_control.buffers[bank].dispatch_signal.value != 0u ||
                    captured_control.buffers[bank].return_signal.value != 0u)
                {
                    error_ =
                        "graph capture executed a mapped timeline publication bank=" +
                        std::to_string(bank) + ": " +
                        std::to_string(
                            captured_control.buffers[bank]
                                .dispatch_signal.value) +
                        "/" +
                        std::to_string(
                            captured_control.buffers[bank]
                                .return_signal.value);
                    return false;
                }
            }
            return true;
        }

        /** @brief Arm and submit one retained replay with no per-layer host work. */
        bool submit(std::uint64_t generation)
        {
            if (!arm(generation) || !source_graph_ || !target_graph_ ||
                !captureFrom(source_cache_) || !captureFrom(target_cache_))
                return false;
            /* Queue the follower first, matching the production transaction
             * service. Prove that its terminal event remains blocked before
             * the continuation is even submitted; an immediate back-to-back
             * launch can accidentally hide a non-functional mapped wait. */
            IWorkerGPUContext *const target_worker =
                &GPUDeviceContextPool::instance().getContext(target_device_);
            IWorkerGPUContext *const source_worker =
                &GPUDeviceContextPool::instance().getContext(source_device_);
            if (!launchEndpoint(
                    *target_executor_,
                    *target_graph_,
                    target_execution_context_.get(),
                    target_cache_,
                    target_stream_,
                    target_worker) ||
                !target_backend_->recordEvent(
                    target_terminal_, 0, target_stream_))
            {
                error_ = "follower production graph submission failed";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            bool follower_completed_before_source = false;
            if (!target_backend_->queryEvent(
                    target_terminal_, 0, &follower_completed_before_source))
            {
                error_ = "could not query the follower-first blocking proof";
                return false;
            }
            if (!launchEndpoint(
                    *source_executor_,
                    *source_graph_,
                    source_execution_context_.get(),
                    source_cache_,
                    source_stream_,
                    source_worker) ||
                !source_backend_->recordEvent(
                    source_terminal_, 0, source_stream_))
            {
                error_ = "continuation production graph submission failed";
                return false;
            }
            terminals_recorded_ = true;
            if (follower_completed_before_source)
            {
                error_ =
                    "follower retained graph bypassed its unpublished dispatch wait";
                return false;
            }
            return true;
        }

        /**
         * @brief Queue generation N+1 while N's reusable admission value is live.
         *
         * Production continuation submission can run ahead of follower-rank
         * retirement. The capture-stable admission threshold is therefore still
         * satisfied by generation N when this replay starts. A correct stage-zero
         * acquisition must remain blocked on its private generation cursor rather
         * than aborting against N's Complete control record.
         */
        bool queueSourceBeforePriorRetirement()
        {
            if (!identity_ || !terminals_recorded_ || source_queued_ahead_ ||
                !source_graph_ || !captureFrom(source_cache_))
            {
                error_ =
                    "stale-admission proof requires one completed live lease";
                return false;
            }
            IWorkerGPUContext *const source_worker =
                &GPUDeviceContextPool::instance().getContext(source_device_);
            if (!launchEndpoint(
                    *source_executor_,
                    *source_graph_,
                    source_execution_context_.get(),
                    source_cache_,
                    source_stream_,
                    source_worker) ||
                !source_backend_->recordEvent(
                    source_terminal_, 0, source_stream_))
            {
                error_ =
                    "ahead-of-retirement continuation replay submission failed";
                return false;
            }

            /* Give the tiny retained graph ample time to expose the former ABA:
             * the old implementation immediately changed Complete to Aborted at
             * layer zero. The fixed graph stays resident on the device wait. */
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            bool source_ready = false;
            if (!source_backend_->queryEvent(
                    source_terminal_, 0, &source_ready))
            {
                error_ =
                    "could not query ahead-of-retirement continuation replay";
                return false;
            }
            const auto &control = source_transport_->activationEpochControl(
                kTargetParticipant, 0u);
            if (source_ready ||
                control.continuation_status.typedState() !=
                    MoEOverlayActivationEndpointState::Complete ||
                control.continuation_status.typedCode() !=
                    MoEOverlayActivationStatusCode::Success)
            {
                error_ =
                    "continuation consumed stale admission instead of waiting for a newer authenticated generation";
                return false;
            }
            source_queued_ahead_ = true;
            return true;
        }

        /**
         * @brief Retire N, arm N+1, and release the already-queued continuation.
         *
         * Only the ordinary protocol reset/arm transition changes mapped state.
         * The continuation executable and its private grant remain untouched;
         * the follower is then submitted exactly as in production.
         */
        bool releaseQueuedSourceWithGeneration(std::uint64_t generation)
        {
            if (!source_queued_ahead_)
            {
                error_ = "no ahead-of-retirement continuation replay is queued";
                return false;
            }
            if (!retire() || !arm(generation))
                return false;

            IWorkerGPUContext *const target_worker =
                &GPUDeviceContextPool::instance().getContext(target_device_);
            if (!launchEndpoint(
                    *target_executor_,
                    *target_graph_,
                    target_execution_context_.get(),
                    target_cache_,
                    target_stream_,
                    target_worker) ||
                !target_backend_->recordEvent(
                    target_terminal_, 0, target_stream_))
            {
                error_ = "follower submission after generation release failed";
                return false;
            }
            source_queued_ahead_ = false;
            terminals_recorded_ = true;
            return true;
        }

        /** @brief Observe terminal events, then copy diagnostic outputs once. */
        bool awaitAndValidate()
        {
            bool source_ready = false;
            bool target_ready = false;
            bool source_query_ok = terminals_recorded_;
            bool target_query_ok = terminals_recorded_;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (terminals_recorded_ &&
                   std::chrono::steady_clock::now() < deadline &&
                   (!source_ready || !target_ready))
            {
                if (!source_ready)
                {
                    source_query_ok = source_backend_->queryEvent(
                        source_terminal_, /*device_id=*/0, &source_ready);
                }
                if (!target_ready)
                {
                    target_query_ok = target_backend_->queryEvent(
                        target_terminal_, /*device_id=*/0, &target_ready);
                }
                if (!source_query_ok || !target_query_ok)
                    break;
                std::this_thread::yield();
            }
            if (!source_ready || !target_ready)
            {
                const auto &control =
                    source_transport_->activationEpochControl(
                        kTargetParticipant, 0u);
                error_ =
                    "packet transaction terminal event did not complete"
                    " source(query/ready)=" +
                    std::to_string(source_query_ok) + "/" +
                    std::to_string(source_ready) +
                    " target(query/ready)=" +
                    std::to_string(target_query_ok) + "/" +
                    std::to_string(target_ready) +
                    " continuation(state/code/pub/consume)=" +
                    std::to_string(control.continuation_status.state) + "/" +
                    std::to_string(control.continuation_status.code) + "/" +
                    std::to_string(
                        control.continuation_status.last_published_stage) +
                    "/" +
                    std::to_string(
                        control.continuation_status.last_consumed_stage) +
                    " follower(state/code/pub/consume)=" +
                    std::to_string(control.follower_status.state) + "/" +
                    std::to_string(control.follower_status.code) + "/" +
                    std::to_string(
                        control.follower_status.last_published_stage) +
                    "/" +
                    std::to_string(
                        control.follower_status.last_consumed_stage) +
                    " signals(dispatch/return)=" +
                    std::to_string(
                        control.buffers[0].dispatch_signal.value) + "/" +
                    std::to_string(control.buffers[0].return_signal.value) +
                    " descriptor_rows=" +
                    std::to_string(
                        control.buffers[0].dispatch_descriptor.live_rows);
                return false;
            }

            std::array<float, kEntries * kDModel> canonical_routes{};
            std::int32_t target_live_rows = -1;
            float target_first_hidden = -1.0f;
            const std::int32_t source_return_live_rows =
                static_cast<std::int32_t>(
                    source_transport_->activationEpochControl(
                        kTargetParticipant, 0u)
                        .buffers[0]
                        .return_descriptor.live_rows);
            float target_return_first = -1.0f;
            if (!source_backend_->deviceToHost(
                    canonical_routes.data(), source_canonical_routes_,
                    sizeof(canonical_routes), 0, source_stream_) ||
                !target_backend_->deviceToHost(
                    &target_live_rows, target_active_rows_,
                    sizeof(target_live_rows), 0, target_stream_) ||
                !target_backend_->deviceToHost(
                    &target_first_hidden, target_hidden_,
                    sizeof(target_first_hidden), 0, target_stream_) ||
                !target_backend_->deviceToHost(
                    &target_return_first, target_canonical_routes_,
                    sizeof(target_return_first), 0, target_stream_))
            {
                error_ = "terminal diagnostic copy failed";
                return false;
            }
            const auto expected_live_rows = static_cast<std::int32_t>(
                targetLiveRows(kRows));
            if (target_live_rows != expected_live_rows)
            {
                const auto &control =
                    source_transport_->activationEpochControl(
                        kTargetParticipant, 0u);
                error_ =
                    "generation=" +
                    std::to_string(
                        identity_ ? identity_->epoch_generation : 0u) +
                    " follower live_rows=" +
                    std::to_string(target_live_rows) +
                    " continuation(state/code/pub/consume)=" +
                    std::to_string(control.continuation_status.state) + "/" +
                    std::to_string(control.continuation_status.code) + "/" +
                    std::to_string(
                        control.continuation_status.last_published_stage) +
                    "/" +
                    std::to_string(
                        control.continuation_status.last_consumed_stage) +
                    " follower(state/code/pub/consume)=" +
                    std::to_string(control.follower_status.state) + "/" +
                    std::to_string(control.follower_status.code) + "/" +
                    std::to_string(
                        control.follower_status.last_published_stage) +
                    "/" +
                    std::to_string(
                        control.follower_status.last_consumed_stage) +
                    " signals(dispatch/return)=" +
                    std::to_string(
                        control.buffers[0].dispatch_signal.value) +
                    "/" +
                    std::to_string(control.buffers[0].return_signal.value) +
                    " descriptor_rows=" +
                    std::to_string(
                        control.buffers[0]
                            .dispatch_descriptor.live_rows);
                return false;
            }
            for (std::int32_t row = 0; row < kRows; ++row)
            {
                for (std::int32_t route = 0; route < kTopK; ++route)
                {
                    const std::size_t original_slot =
                        static_cast<std::size_t>(row * kTopK + route);
                    for (std::int32_t column = 0;
                         column < kDModel;
                         ++column)
                    {
                        const float expected =
                            routeParticipant(original_slot) ==
                                    kTargetParticipant
                                ? kRouteWeightPattern[
                                      original_slot %
                                      kRouteWeightPattern.size()] *
                                      sourceValue(row, column)
                                : 0.0f;
                        const float observed = canonical_routes[
                            original_slot *
                                static_cast<std::size_t>(kDModel) +
                            static_cast<std::size_t>(column)];
                        if (observed != expected)
                        {
                            const auto &control =
                                source_transport_->activationEpochControl(
                                    kTargetParticipant, 0u);
                            error_ =
                                "continuation canonical route bank did not preserve exact original-slot bytes"
                                " generation=" +
                                std::to_string(
                                    identity_
                                        ? identity_->epoch_generation
                                        : 0u) +
                                " row=" + std::to_string(row) +
                                " route=" + std::to_string(route) +
                                " column=" + std::to_string(column) +
                                " expected=" + std::to_string(expected) +
                                " observed=" + std::to_string(observed) +
                                " target_first_hidden=" +
                                std::to_string(target_first_hidden) +
                                " source_return_live_rows=" +
                                std::to_string(source_return_live_rows) +
                                " target_return_first=" +
                                std::to_string(target_return_first) +
                                " mapped_dispatch_first=" +
                                std::to_string(
                                    source_transport_->sharedDispatchRows(
                                        kTargetParticipant)
                                        .hidden_rows_fp32[0]) +
                                " mapped_return_first=" +
                                std::to_string(
                                    source_transport_->sharedReturnRows(
                                        kTargetParticipant)
                                        .output_rows_fp32[0]) +
                                " continuation(state/code)=" +
                                std::to_string(
                                    control.continuation_status.state) + "/" +
                                std::to_string(
                                    control.continuation_status.code) +
                                " follower(state/code)=" +
                                std::to_string(
                                    control.follower_status.state) + "/" +
                                std::to_string(
                                    control.follower_status.code) +
                                " signals(dispatch/return)=" +
                                std::to_string(
                                    control.buffers[0]
                                        .dispatch_signal.value) +
                                "/" +
                                std::to_string(
                                    control.buffers[0]
                                        .return_signal.value) +
                                " full_graph_nodes(source/target)=" +
                                std::to_string(
                                    captureFrom(source_cache_)->nodeCount()) +
                                "," +
                                std::to_string(
                                    captureFrom(target_cache_)->nodeCount());
                            return false;
                        }
                    }
                }
            }
            const auto &control = source_transport_->activationEpochControl(
                kTargetParticipant, 0u);
            if (static_cast<MoEOverlayActivationEndpointState>(
                    control.continuation_status.state) !=
                    MoEOverlayActivationEndpointState::Complete ||
                static_cast<MoEOverlayActivationEndpointState>(
                    control.follower_status.state) !=
                    MoEOverlayActivationEndpointState::Complete ||
                control.continuation_status.code !=
                    static_cast<std::uint32_t>(
                        MoEOverlayActivationStatusCode::Success) ||
                control.follower_status.code !=
                    static_cast<std::uint32_t>(
                        MoEOverlayActivationStatusCode::Success))
            {
                error_ = "mapped endpoint statuses did not complete successfully";
                return false;
            }
            if (!identity_)
            {
                error_ = "mapped endpoint completed without an active epoch identity";
                return false;
            }
            std::string traffic_error;
            const auto traffic = protocol_->completedTraffic(
                *identity_, &traffic_error);
            const std::uint64_t expected_stage_count = kLayers.size();
            const std::uint64_t expected_rows =
                targetLiveRows(kRows) * expected_stage_count;
            const std::uint64_t expected_entries =
                targetLiveEntries(kRows) * expected_stage_count;
            const std::uint64_t expected_dispatch_bytes =
                moeOverlayDispatchPayloadBytes(
                    targetLiveRows(kRows),
                    targetLiveEntries(kRows),
                    static_cast<std::uint32_t>(kDModel)) *
                expected_stage_count;
            const std::uint64_t expected_return_bytes =
                moeOverlayReturnPayloadBytes(
                    targetLiveEntries(kRows),
                    static_cast<std::uint32_t>(kDModel)) *
                expected_stage_count;
            if (!traffic ||
                traffic->dispatch_payload_bytes != expected_dispatch_bytes ||
                traffic->return_payload_bytes != expected_return_bytes ||
                traffic->dispatch_live_rows != expected_rows ||
                traffic->return_live_rows != expected_rows ||
                traffic->dispatch_live_entries != expected_entries ||
                traffic->return_live_entries != expected_entries ||
                traffic->dispatch_stage_count != expected_stage_count ||
                traffic->return_stage_count != expected_stage_count)
            {
                error_ =
                    "device-owned activation traffic totals were incomplete: " +
                    traffic_error;
                return false;
            }
            return true;
        }

        /** @brief Retire both terminal events and reset the leased signal banks. */
        bool retire()
        {
            if (!protocol_ || !identity_ || !terminals_recorded_)
            {
                error_ = "cannot retire an incomplete activation replay";
                return false;
            }
            std::string reset_error;
            last_digest_ = identity_->digest;
            if (!protocol_->reset(*identity_, &reset_error))
            {
                error_ = reset_error;
                return false;
            }
            auto &control = source_transport_->activationEpochControl(
                kTargetParticipant, 0u);
            for (std::uint32_t bank = 0u;
                 bank < kMoEOverlayActivationBufferCount;
                 ++bank)
            {
                if (control.buffers[bank].dispatch_signal.value != 0u ||
                    control.buffers[bank].return_signal.value != 0u)
                {
                    error_ = "retired activation lease retained a timeline signal";
                    return false;
                }
            }
            identity_.reset();
            terminals_recorded_ = false;
            return true;
        }

        /** @return Stable diagnostic from the first failed setup/operation. */
        const std::string &error() const noexcept { return error_; }

    private:
        /** @brief Submit one complete retained endpoint graph asynchronously. */
        static bool launchEndpoint(
            DeviceGraphExecutor &executor,
            ComputeGraph &graph,
            IDeviceContext *context,
            DeviceGraphExecutor::GraphSegmentCache &cache,
            void *stream,
            IWorkerGPUContext *worker)
        {
            return executor.executeWithCachedGraphReplay(
                graph,
                context,
                cache,
                stream,
                worker,
                /*collective_nodes=*/nullptr,
                /*collectives_graph_capturable=*/false,
                /*force_recapture=*/false,
                /*defer_final_sync=*/true,
                {},
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph);
        }

        /** @return Sole complete executable owned by a production full-graph cache. */
        static const IGPUGraphCapture *captureFrom(
            const DeviceGraphExecutor::GraphSegmentCache &cache) noexcept
        {
            if (cache.segments.size() != 1u ||
                !cache.segments.front().capturable ||
                !cache.segments.front().capture ||
                !cache.segments.front().capture->hasExecutable())
            {
                return nullptr;
            }
            return cache.segments.front().capture.get();
        }

        /**
         * @brief Prove HIP imported real fragment nodes behind device waits.
         *
         * CUDA may retain native child graphs. HIP must not: its mapped wait
         * ordering is certified only when every fragment kernel is a root-level
         * node behind an ordinary device wait kernel.
         */
        bool validateROCmTimelineLowering(
            const IGPUGraphCapture &capture,
            DeviceId device,
            std::size_t expected_node_count,
            std::size_t expected_wait_nodes,
            const char *role)
        {
            if (!device.is_rocm())
                return true;
            std::vector<GPUGraphKernelNodeInfo> kernels;
            std::string inspection_error;
            if (capture.nodeCount() != expected_node_count ||
                !capture.inspectKernelNodes(kernels, &inspection_error))
            {
                error_ = std::string(role) +
                         " HIP timeline graph did not expose its complete "
                         "flattened node DAG: expected=" +
                         std::to_string(expected_node_count) +
                         " observed=" +
                         std::to_string(capture.nodeCount()) + " error=" +
                         inspection_error;
                return false;
            }
            std::size_t wait_nodes = 0u;
            std::vector<std::string> wait_node_names;
            for (const auto &kernel : kernels)
            {
                if (kernel.nesting_depth != 0u)
                {
                    error_ = std::string(role) +
                             " HIP timeline retained a nested child kernel: " +
                             kernel.name;
                    return false;
                }
                if (kernel.name.find("waitSystemValue64") !=
                    std::string::npos)
                {
                    ++wait_nodes;
                    wait_node_names.push_back(kernel.name);
                }
            }
            if (wait_nodes != expected_wait_nodes)
            {
                std::ostringstream detail;
                detail << role
                       << " HIP timeline did not contain the exact number of "
                          "device-owned system waits: expected="
                       << expected_wait_nodes << " observed=" << wait_nodes;
                for (const auto &name : wait_node_names)
                    detail << " wait_node='" << name << "'";
                error_ = detail.str();
                return false;
            }
            return true;
        }

        /** @return Deterministic source activation value for one logical row. */
        static float sourceValue(std::int32_t row, std::int32_t column)
        {
            return static_cast<float>(row * 100 + column) + 0.25f;
        }

        /** @brief Allocate and remember one exact-device buffer. */
        void *allocate(IBackend *backend, std::size_t bytes)
        {
            void *address = backend ? backend->allocate(bytes, 0) : nullptr;
            allocations_.push_back({
                .backend = backend, .address = address, .ordinal = 0});
            return address;
        }

        /** @return Whether every allocation attempted by initialize() succeeded. */
        bool allAllocationsValid() const noexcept
        {
            for (const auto &allocation : allocations_)
            {
                if (!allocation.address)
                    return false;
            }
            return !allocations_.empty();
        }

        /**
         * @brief Resolve one exact device alias into the real placement-bank ABI.
         *
         * Production graph construction performs the same setup-only offset
         * calculation from DeviceMoELayerRuntime. Keeping it here makes the
         * integration proof sensitive to host/CUDA/HIP bank layout drift.
         */
        MoEOverlayRoutePlacementBankDeviceView placementBankView(
            void *placement_banks,
            std::uint32_t bank) const
        {
            if (!placement_banks ||
                bank >= kDeviceMoEOverlayEpochBankCount)
            {
                return {};
            }
            auto *const bank_base =
                static_cast<std::byte *>(placement_banks) +
                static_cast<std::size_t>(bank) *
                    sizeof(DeviceMoEPlacementBank);
            return {
                .route_participants =
                    reinterpret_cast<const std::int32_t *>(
                        bank_base + offsetof(
                                        DeviceMoEPlacementBank,
                                        overlay_route_participant)),
                .epoch = reinterpret_cast<const std::uint32_t *>(
                    bank_base + offsetof(DeviceMoEPlacementBank, epoch)),
            };
        }

        /** @return Source-device view used by continuation packet stages. */
        MoEOverlayRoutePlacementBankDeviceView placementBankView(
            std::uint32_t bank) const
        {
            return placementBankView(source_placement_banks_, bank);
        }

        /** @brief Upload deterministic router output before transaction admission. */
        bool initializeSourcePayload()
        {
            if (!source_hidden_tensor_ || !source_routing_indices_tensor_ ||
                !source_routing_weights_tensor_ ||
                !source_canonical_route_tensor_ || !source_zero_tensor_ ||
                !target_hidden_tensor_ || !target_routing_indices_tensor_ ||
                !target_routing_weights_tensor_ ||
                !target_route_input_tensor_ ||
                !target_canonical_route_tensor_)
            {
                error_ = "typed packet tensors were not constructed";
                return false;
            }
            std::array<float, kRows * kDModel> hidden{};
            for (std::int32_t row = 0; row < kRows; ++row)
            {
                for (std::int32_t column = 0; column < kDModel; ++column)
                {
                    hidden[static_cast<std::size_t>(row * kDModel + column)] =
                        sourceValue(row, column);
                }
            }
            std::array<std::int32_t, kEntries> experts{};
            std::array<float, kEntries> weights{};
            std::array<std::int32_t, kEntries> participants{};
            for (std::size_t slot = 0u; slot < kEntries; ++slot)
            {
                experts[slot] = static_cast<std::int32_t>(10u + slot % 8u);
                weights[slot] =
                    kRouteWeightPattern[slot % kRouteWeightPattern.size()];
                participants[slot] = routeParticipant(slot);
            }
            std::array<float, kEntries * kDModel> compact_route_input{};
            std::size_t compact_row = 0u;
            for (std::int32_t row = 0; row < kRows; ++row)
            {
                std::size_t compact_route = 0u;
                for (std::int32_t slot = 0; slot < kTopK; ++slot)
                {
                    const std::size_t original_slot =
                        static_cast<std::size_t>(row * kTopK + slot);
                    if (participants[original_slot] != kTargetParticipant)
                        continue;
                    const std::size_t compact_slot =
                        compact_row * static_cast<std::size_t>(kTopK) +
                        compact_route++;
                    for (std::int32_t column = 0;
                         column < kDModel;
                         ++column)
                    {
                        /* Model expert compute publishes independently rounded,
                         * preweighted rows. This identity-expert fixture keeps
                         * that production arithmetic while isolating packet
                         * slot preservation from GEMM behavior. */
                        compact_route_input[
                            compact_slot *
                                static_cast<std::size_t>(kDModel) +
                            static_cast<std::size_t>(column)] =
                            weights[original_slot] *
                            sourceValue(row, column);
                    }
                }
                compact_row += compact_route != 0u ? 1u : 0u;
            }
            const std::int32_t active_rows = kRows;
            std::array<DeviceMoEPlacementBank,
                       kDeviceMoEOverlayEpochBankCount>
                placement_banks{};
            for (auto &bank : placement_banks)
            {
                bank.epoch = static_cast<std::uint32_t>(kPlacementEpoch);
                bank.expert_count = kExpertCount;
                std::fill_n(
                    bank.overlay_route_participant,
                    kExpertCount,
                    kSourceParticipant);
            }
            /* Bank zero is intentionally adversarial. The request ticket selects
             * bank one, whose global owner map reproduces the router oracle. */
            for (std::size_t slot = 0u; slot < kEntries; ++slot)
            {
                placement_banks[1].overlay_route_participant[
                    static_cast<std::size_t>(experts[slot])] =
                    participants[slot];
            }
            const DeviceMoEOverlayEpochTicket placement_ticket{
                .epoch = kPlacementEpoch,
                .selector = deviceMoEOverlayEpochSelector(
                    /*generation=*/3u,
                    /*bank=*/1u),
            };
            const DeviceMoEOverlayEpochStatus placement_status{
                .epoch = kPlacementEpoch,
                .selector = placement_ticket.selector,
                .operation = static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochOperation::Acquire),
                .code = static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochStatusCode::Success),
                .bank = placement_ticket.bank(),
                .observed_state = static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Published),
            };

            std::copy(
                hidden.begin(), hidden.end(),
                static_cast<float *>(
                    source_hidden_tensor_->raw_mutable_data()));
            std::transform(
                experts.begin(), experts.end(),
                static_cast<float *>(
                    source_routing_indices_tensor_->raw_mutable_data()),
                [](std::int32_t expert)
                {
                    return static_cast<float>(expert);
                });
            std::copy(
                weights.begin(), weights.end(),
                static_cast<float *>(
                    source_routing_weights_tensor_->raw_mutable_data()));
            std::copy(
                compact_route_input.begin(), compact_route_input.end(),
                static_cast<float *>(
                    target_route_input_tensor_->raw_mutable_data()));
            const auto clear_tensor = [](FP32Tensor &tensor)
            {
                std::fill_n(
                    static_cast<float *>(tensor.raw_mutable_data()),
                    tensor.numel(), 0.0f);
            };
            clear_tensor(*source_canonical_route_tensor_);
            clear_tensor(*source_zero_tensor_);
            clear_tensor(*target_hidden_tensor_);
            clear_tensor(*target_routing_indices_tensor_);
            clear_tensor(*target_routing_weights_tensor_);
            clear_tensor(*target_canonical_route_tensor_);

            const bool tensors_ready =
                source_hidden_tensor_->ensureOnDevice(
                    source_device_, source_stream_) &&
                source_routing_indices_tensor_->ensureOnDevice(
                    source_device_, source_stream_) &&
                source_routing_weights_tensor_->ensureOnDevice(
                    source_device_, source_stream_) &&
                source_canonical_route_tensor_->ensureOnDevice(
                    source_device_, source_stream_) &&
                source_zero_tensor_->ensureOnDevice(
                    source_device_, source_stream_) &&
                target_hidden_tensor_->ensureOnDevice(
                    target_device_, target_stream_) &&
                target_routing_indices_tensor_->ensureOnDevice(
                    target_device_, target_stream_) &&
                target_routing_weights_tensor_->ensureOnDevice(
                    target_device_, target_stream_) &&
                target_route_input_tensor_->ensureOnDevice(
                    target_device_, target_stream_) &&
                target_canonical_route_tensor_->ensureOnDevice(
                    target_device_, target_stream_);
            if (!tensors_ready)
            {
                error_ = "could not bind typed packet tensors to their exact devices";
                return false;
            }
            source_hidden_ = source_hidden_tensor_->gpu_data_ptr();
            source_indices_ =
                source_routing_indices_tensor_->gpu_data_ptr();
            source_weights_ =
                source_routing_weights_tensor_->gpu_data_ptr();
            source_canonical_routes_ =
                source_canonical_route_tensor_->gpu_data_ptr();
            target_hidden_ = target_hidden_tensor_->gpu_data_ptr();
            target_indices_ =
                target_routing_indices_tensor_->gpu_data_ptr();
            target_weights_ =
                target_routing_weights_tensor_->gpu_data_ptr();
            target_canonical_routes_ =
                target_canonical_route_tensor_->gpu_data_ptr();
            if (!source_hidden_ || !source_indices_ || !source_weights_ ||
                !source_canonical_routes_ ||
                !target_hidden_ || !target_indices_ || !target_weights_ ||
                !target_route_input_tensor_->gpu_data_ptr() ||
                !target_canonical_routes_)
            {
                error_ = "typed packet tensors did not expose stable device addresses";
                return false;
            }

            const bool uploaded =
                source_backend_->hostToDevice(
                    source_placement_banks_, placement_banks.data(),
                    sizeof(placement_banks), 0, source_stream_) &&
                source_backend_->hostToDevice(
                    source_placement_ticket_, &placement_ticket,
                    sizeof(placement_ticket), 0, source_stream_) &&
                source_backend_->hostToDevice(
                    source_placement_status_, &placement_status,
                    sizeof(placement_status), 0, source_stream_) &&
                target_backend_->hostToDevice(
                    target_placement_banks_, placement_banks.data(),
                    sizeof(placement_banks), 0, target_stream_) &&
                target_backend_->hostToDevice(
                    target_placement_ticket_, &placement_ticket,
                    sizeof(placement_ticket), 0, target_stream_) &&
                target_backend_->hostToDevice(
                    target_placement_status_, &placement_status,
                    sizeof(placement_status), 0, target_stream_) &&
                source_backend_->hostToDevice(
                    source_active_rows_, &active_rows, sizeof(active_rows), 0,
                    source_stream_) &&
                target_backend_->memset(
                    target_active_rows_, 0, sizeof(std::int32_t), 0,
                    target_stream_);
            if (!uploaded)
                error_ = "could not initialize endpoint device buffers";
            return uploaded;
        }

        /** @brief Bind the CPU oracle to the exact planner-produced lane header. */
        bool initializeProtocol()
        {
            auto &control = source_transport_->activationEpochControl(
                kTargetParticipant, 0u);
            const auto &header = control.channel;
            protocol_config_ = MoEOverlayActivationEpochConfig{
                .channel_nonce = header.channel_nonce,
                .topology_fingerprint_low = header.topology_fingerprint_low,
                .topology_fingerprint_high = header.topology_fingerprint_high,
                .workspace_generation = header.workspace_generation,
                .source = {
                    .world_rank = header.source_world_rank,
                    .participant_id = header.source_participant_id,
                    .tier_priority = header.source_tier_priority,
                    .domain_ordinal = header.source_domain_ordinal,
                },
                .target = {
                    .world_rank = header.target_world_rank,
                    .participant_id = header.target_participant_id,
                    .tier_priority = header.target_tier_priority,
                    .domain_ordinal = header.target_domain_ordinal,
                },
                .lane_ordinal = header.lane_ordinal,
                .graph_role_mask = header.graph_role_mask,
                .model_layer_indices = std::vector<std::int32_t>(
                    kLayers.begin(), kLayers.end()),
            };
            protocol_ = std::make_unique<MoEOverlayActivationEpochProtocol>(
                control, protocol_config_);
            return true;
        }

        /** @brief Arm one new authenticated generation for retained replay. */
        bool arm(std::uint64_t generation)
        {
            if (!protocol_ || !protocol_config_.valid() || generation == 0u)
            {
                error_ = "activation protocol was not initialized";
                return false;
            }
            const auto &config = protocol_config_;
            const MoEOverlayInferenceTopologyIdentity topology{
                .workspace_generation = config.workspace_generation,
                .topology_fingerprint_low = config.topology_fingerprint_low,
                .topology_fingerprint_high = config.topology_fingerprint_high,
                .source_world_rank = config.source.world_rank,
                .target_world_rank = config.target.world_rank,
            };
            const MoEOverlayInferenceCommandIdentity command{
                .request_generation = generation,
                .command_id = generation,
                .initial_placement_epoch =
                    kSchedulerPlacementEpochFloor,
            };
            const auto ticket = makeMoEOverlayInferenceExecutionTicket(
                topology,
                command,
                /*transaction_ordinal=*/generation,
                /*logical_step_id=*/generation,
                /*placement_epoch=*/kSchedulerPlacementEpochFloor,
                MoEOverlayInferenceGraphRole::MainPrefill,
                /*request_count=*/1,
                /*logical_rows_per_request=*/kRows,
                /*physical_rows_per_request=*/kRows);
            std::string arm_error;
            identity_ = protocol_->arm(
                ticket,
                /*epoch_generation=*/generation,
                /*deadline_ns=*/10'000'000'000u,
                &arm_error);
            if (!identity_)
            {
                error_ = "generation=" + std::to_string(generation) +
                         " arm failed: " + arm_error;
                return false;
            }
            if (last_digest_.valid() && identity_->digest == last_digest_)
            {
                error_ = "distinct replay generations produced the same digest";
                return false;
            }
            return true;
        }

        /** @brief Release any queued GEQ waits during failure cleanup. */
        void publishAbort() noexcept
        {
            if (!source_transport_)
                return;
            auto &control = source_transport_->activationEpochControl(
                kTargetParticipant, 0u);
            for (std::uint32_t bank = 0u;
                 bank < kMoEOverlayActivationBufferCount;
                 ++bank)
            {
                std::atomic_ref<std::uint64_t>(
                    control.buffers[bank].dispatch_signal.value)
                    .store(kMoEOverlayActivationAbortTimeline,
                           std::memory_order_release);
                std::atomic_ref<std::uint64_t>(
                    control.buffers[bank].return_signal.value)
                    .store(kMoEOverlayActivationAbortTimeline,
                           std::memory_order_release);
            }
            std::atomic_ref<std::uint64_t>(
                control.admission.ready_signal)
                .store(kMoEOverlayActivationAbortTimeline,
                       std::memory_order_release);
        }

        /** @brief Drain one exact stream through an event before releasing pages. */
        static void drain(
            IBackend *backend, void *stream, void *event) noexcept
        {
            if (backend && stream && event &&
                backend->recordEvent(event, 0, stream))
            {
                (void)awaitEvent(backend, event, std::chrono::seconds(5));
            }
        }

        DeviceId source_device_;
        DeviceId target_device_;
        IBackend *source_backend_ = nullptr;
        IBackend *target_backend_ = nullptr;
        std::unique_ptr<MoEOverlayNodeLocalRankBatchTransport> source_transport_;
        std::unique_ptr<MoEOverlayNodeLocalRankBatchTransport> target_transport_;
        MoEOverlayMappedActivationDeviceLane source_lane_;
        MoEOverlayMappedActivationDeviceLane target_lane_;
        std::unique_ptr<MoEOverlayActivationEpochProtocol> protocol_;
        MoEOverlayActivationEpochConfig protocol_config_;
        std::optional<MoEOverlayActivationEpochIdentity> identity_;
        MoEOverlayActivationDigest last_digest_{};
        std::vector<std::unique_ptr<IGPUGraphCapture>>
            source_dispatch_captures_;
        std::vector<std::unique_ptr<IGPUGraphCapture>>
            source_return_captures_;
        std::vector<std::unique_ptr<IGPUGraphCapture>>
            target_compute_captures_;
        std::vector<std::unique_ptr<ComputeGraph>> source_dispatch_graphs_;
        std::vector<std::unique_ptr<ComputeGraph>> source_return_graphs_;
        std::vector<std::unique_ptr<ComputeGraph>> target_compute_graphs_;
        std::unique_ptr<IGPUGraphCapture> source_capture_;
        std::unique_ptr<IGPUGraphCapture> target_capture_;
        std::unique_ptr<IDeviceContext> source_execution_context_;
        std::unique_ptr<IDeviceContext> target_execution_context_;
        std::unique_ptr<BufferArena> source_arena_;
        std::unique_ptr<BufferArena> target_arena_;
        std::unique_ptr<DeviceGraphExecutor> source_executor_;
        std::unique_ptr<DeviceGraphExecutor> target_executor_;
        std::unique_ptr<ComputeGraph> source_graph_;
        std::unique_ptr<ComputeGraph> target_graph_;
        DeviceGraphExecutor::GraphSegmentCache source_cache_;
        DeviceGraphExecutor::GraphSegmentCache target_cache_;
        void *source_stream_ = nullptr;
        void *target_stream_ = nullptr;
        void *source_terminal_ = nullptr;
        void *target_terminal_ = nullptr;
        void *source_hidden_ = nullptr;
        void *source_indices_ = nullptr;
        void *source_weights_ = nullptr;
        void *source_placement_banks_ = nullptr;
        void *source_placement_ticket_ = nullptr;
        void *source_placement_status_ = nullptr;
        void *source_active_rows_ = nullptr;
        void *source_canonical_routes_ = nullptr;
        void *target_hidden_ = nullptr;
        void *target_indices_ = nullptr;
        void *target_weights_ = nullptr;
        void *target_placement_banks_ = nullptr;
        void *target_placement_ticket_ = nullptr;
        void *target_placement_status_ = nullptr;
        void *target_active_rows_ = nullptr;
        void *target_canonical_routes_ = nullptr;
        std::shared_ptr<FP32Tensor> source_hidden_tensor_;
        std::shared_ptr<FP32Tensor> source_routing_indices_tensor_;
        std::shared_ptr<FP32Tensor> source_routing_weights_tensor_;
        std::shared_ptr<FP32Tensor> source_canonical_route_tensor_;
        std::shared_ptr<FP32Tensor> source_zero_tensor_;
        std::shared_ptr<FP32Tensor> target_hidden_tensor_;
        std::shared_ptr<FP32Tensor> target_routing_indices_tensor_;
        std::shared_ptr<FP32Tensor> target_routing_weights_tensor_;
        std::shared_ptr<FP32Tensor> target_route_input_tensor_;
        std::shared_ptr<FP32Tensor> target_canonical_route_tensor_;
        std::vector<DeviceAllocation> allocations_;
        bool terminals_recorded_ = false;
        bool source_queued_ahead_ = false;
        std::string error_;
    };

    /** @brief Replay one retained pair across two authenticated generations. */
    template <std::int32_t Rows>
    void runRoleAssignment(DeviceId source, DeviceId target)
    {
        ScopedActivationEpochPerfStats perf_stats;
        PacketRoundTrip<Rows> transaction(source, target);
        ASSERT_TRUE(transaction.initialize()) << transaction.error();

        double waits_enqueued = 0.0;
        double publications_enqueued = 0.0;
        double kernel_wait_bindings = 0.0;
        double kernel_publication_bindings = 0.0;
        double dispatch_d2h_bytes = 0.0;
        double dispatch_h2d_bytes = 0.0;
        double return_d2h_bytes = 0.0;
        double return_h2d_bytes = 0.0;
        double shared_dispatch_payload_publications = 0.0;
        double asynchronous_lane_forks = 0.0;
        double asynchronous_lane_joins = 0.0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_activation_epoch", "forward_graph"}))
        {
            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_async_lane_forks")
            {
                asynchronous_lane_forks += record.value;
                ASSERT_EQ(record.tags.at("lanes"), "1");
                ASSERT_EQ(record.tags.at("rows"), std::to_string(Rows));
            }
            else if (record.domain == "forward_graph" &&
                     record.name == "moe_overlay_async_lane_joins")
            {
                asynchronous_lane_joins += record.value;
                ASSERT_EQ(record.tags.at("lanes"), "1");
                ASSERT_EQ(record.tags.at("rows"), std::to_string(Rows));
                ASSERT_EQ(
                    record.tags.at("canonical_materialization"),
                    "true");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name == "device_timeline_waits_enqueued")
            {
                waits_enqueued += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("ordering"),
                    "exact_stream_64bit_geq");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                     "device_timeline_publications_enqueued")
            {
                publications_enqueued += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("ordering"),
                    "exact_stream_64bit_fenced");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                     "device_timeline_kernel_wait_bindings")
            {
                kernel_wait_bindings += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("ordering"),
                    "fused_packet_system_acquire");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                     "device_timeline_kernel_publication_bindings")
            {
                kernel_publication_bindings += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("ordering"),
                    "fused_packet_system_release");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_dispatch_d2h_bytes")
            {
                dispatch_d2h_bytes += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("payload_path"),
                    "shared_physical_mapped");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_dispatch_h2d_bytes")
            {
                dispatch_h2d_bytes += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("ordering"),
                    "mapped_timeline_then_bulk_dma");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_return_d2h_bytes")
            {
                return_d2h_bytes += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("ordering"),
                    "device_payload_then_bulk_dma");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_physical_return_h2d_bytes")
            {
                return_h2d_bytes += record.value;
                ASSERT_EQ(record.tags.at("host_blocking"), "false");
                ASSERT_EQ(
                    record.tags.at("ordering"),
                    "mapped_timeline_then_bulk_dma");
            }
            else if (record.domain == "moe_overlay_activation_epoch" &&
                     record.name ==
                         "shared_dispatch_payload_publications")
            {
                shared_dispatch_payload_publications += record.value;
                ASSERT_EQ(record.tags.at("lanes"), "1");
                ASSERT_EQ(record.tags.at("rows"), std::to_string(Rows));
            }
        }
        constexpr bool single_row_direct = Rows == 1;
        const double expected_shared_dispatch_bytes =
            single_row_direct
                ? 0.0
                : static_cast<double>(
                      kLayers.size() *
                      static_cast<std::size_t>(Rows * kDModel) *
                      sizeof(float));
        EXPECT_EQ(
            waits_enqueued,
            single_row_direct ? 2.0 : 2.0 * (kLayers.size() + 1u));
        EXPECT_EQ(
            publications_enqueued,
            single_row_direct ? 0.0 : 2.0 * kLayers.size());
        EXPECT_EQ(
            kernel_wait_bindings,
            single_row_direct ? 2.0 * kLayers.size() : 0.0);
        EXPECT_EQ(
            kernel_publication_bindings,
            single_row_direct ? 2.0 * kLayers.size() : 0.0);
        EXPECT_EQ(dispatch_d2h_bytes, expected_shared_dispatch_bytes);
        EXPECT_EQ(dispatch_h2d_bytes, 0.0);
        EXPECT_EQ(return_d2h_bytes, 0.0);
        EXPECT_EQ(return_h2d_bytes, 0.0);
        EXPECT_EQ(
            shared_dispatch_payload_publications,
            single_row_direct ? 0.0
                              : static_cast<double>(kLayers.size()));
        EXPECT_EQ(
            asynchronous_lane_forks,
            single_row_direct ? 0.0 : static_cast<double>(kLayers.size()));
        EXPECT_EQ(
            asynchronous_lane_joins,
            single_row_direct ? 0.0 : static_cast<double>(kLayers.size()));

        for (std::uint64_t generation = 1u; generation <= 2u; ++generation)
        {
            const auto submit_begin = std::chrono::steady_clock::now();
            ASSERT_TRUE(transaction.submit(generation)) << transaction.error();
            EXPECT_LT(
                std::chrono::steady_clock::now() - submit_begin,
                std::chrono::seconds(2))
                << "submission contains a host-side per-stage wait";
            ASSERT_TRUE(transaction.awaitAndValidate()) << transaction.error();
            ASSERT_TRUE(transaction.retire()) << transaction.error();
        }
    }

    /** @brief Prove an ahead continuation cannot consume a prior lease's `1`. */
    void runStaleAdmissionABAProof(DeviceId source, DeviceId target)
    {
        PacketRoundTrip<1> transaction(source, target);
        ASSERT_TRUE(transaction.initialize()) << transaction.error();
        ASSERT_TRUE(transaction.submit(/*generation=*/1u))
            << transaction.error();
        ASSERT_TRUE(transaction.awaitAndValidate()) << transaction.error();
        ASSERT_TRUE(transaction.queueSourceBeforePriorRetirement())
            << transaction.error();
        ASSERT_TRUE(transaction.releaseQueuedSourceWithGeneration(
            /*generation=*/2u))
            << transaction.error();
        ASSERT_TRUE(transaction.awaitAndValidate()) << transaction.error();
        ASSERT_TRUE(transaction.retire()) << transaction.error();
    }

    /**
     * @brief Own exact streams and buffers for one sparse-route exchange proof.
     *
     * The destructor first publishes the fabric's terminal sentinel and then
     * drains the two exact streams. This makes every assertion path safe even
     * when one endpoint kernel is resident in a device-side epoch wait.
     */
    class SparseRouteExchangeResources final
    {
    public:
        SparseRouteExchangeResources(DeviceId root, DeviceId producer)
            : root_device(root), producer_device(producer),
              root_backend(backendFor(root)),
              producer_backend(backendFor(producer))
        {
        }

        ~SparseRouteExchangeResources()
        {
            if (exchange)
                exchange->abortForShutdown();
            if (root_backend && root_stream)
                (void)root_backend->synchronizeStream(
                    root_stream, root_device.ordinal);
            if (producer_backend && producer_stream)
                (void)producer_backend->synchronizeStream(
                    producer_stream, producer_device.ordinal);
            for (auto iterator = allocations.rbegin();
                 iterator != allocations.rend(); ++iterator)
            {
                if (iterator->backend && iterator->address)
                    iterator->backend->free(
                        iterator->address, iterator->ordinal);
            }
            if (root_backend && root_terminal)
                root_backend->destroyEvent(
                    root_terminal, root_device.ordinal);
            if (producer_backend && producer_terminal)
                producer_backend->destroyEvent(
                    producer_terminal, producer_device.ordinal);
            if (root_backend && root_stream)
                root_backend->destroyStream(
                    root_stream, root_device.ordinal);
            if (producer_backend && producer_stream)
                producer_backend->destroyStream(
                    producer_stream, producer_device.ordinal);
            exchange.reset();
        }

        SparseRouteExchangeResources(
            const SparseRouteExchangeResources &) = delete;
        SparseRouteExchangeResources &operator=(
            const SparseRouteExchangeResources &) = delete;

        /** @brief Allocate one buffer on its exact endpoint and retain ownership. */
        void *allocate(IBackend *backend, DeviceId device, std::size_t bytes)
        {
            void *const address = backend
                                      ? backend->allocate(bytes, device.ordinal)
                                      : nullptr;
            allocations.push_back({
                .backend = backend,
                .address = address,
                .ordinal = device.ordinal,
            });
            return address;
        }

        /** @brief Allocate stable transfer scratch retained through stream drain. */
        std::shared_ptr<DeviceTransferBuffer> allocateTransferBuffer(
            DeviceId device,
            std::size_t bytes)
        {
            auto buffer = TransferEngine::instance()
                              .allocateDeviceTransferBuffer(bytes, device);
            transfer_buffers.push_back(buffer);
            return buffer;
        }

        DeviceId root_device;
        DeviceId producer_device;
        IBackend *root_backend = nullptr;
        IBackend *producer_backend = nullptr;
        void *root_stream = nullptr;
        void *producer_stream = nullptr;
        void *root_terminal = nullptr;
        void *producer_terminal = nullptr;
        std::unique_ptr<MoEOverlayNodeLocalRouteExchange> exchange;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> transfer_buffers;
        std::vector<DeviceAllocation> allocations;
    };

    /**
     * @brief Reuse one CPU-published canonical ticket for twenty GPU replays.
     *
     * Each replay publishes a different route permutation and payload. The CPU
     * must be unable to arm the next replay while the current publication is
     * live; after the exact-stream materialization and acknowledgement event,
     * the same storage must become reusable without a stream/device sync. This
     * proves both mapped bytes and the producer/consumer sequence protocol on
     * the selected continuation backend.
     */
    void runCanonicalRouteTicketReuseProof(DeviceId continuation_device)
    {
        /* Qwen 3.5 122B decode publishes one original slot per top-k route at
         * hidden width 3072. Keeping that exact geometry makes the regression
         * cover the production 96-block mapped materialization launch. */
        constexpr std::size_t route_capacity = 8u;
        constexpr std::int32_t d_model = 3072;
        constexpr std::uint64_t workspace_generation = 17u;
        constexpr std::uint64_t first_residency_epoch = 101u;
        constexpr std::size_t replay_count = 20u;
        constexpr std::size_t element_count =
            route_capacity * static_cast<std::size_t>(d_model);
        constexpr std::size_t payload_bytes = element_count * sizeof(float);

        SparseRouteExchangeResources resources(
            continuation_device, continuation_device);
        ASSERT_NE(resources.root_backend, nullptr);
        ASSERT_GT(
            resources.root_backend->deviceCount(),
            continuation_device.ordinal);
        resources.root_stream = resources.root_backend->createStream(
            continuation_device.ordinal);
        resources.root_terminal = resources.root_backend->createEvent(
            continuation_device.ordinal);
        ASSERT_NE(resources.root_stream, nullptr);
        ASSERT_NE(resources.root_terminal, nullptr);

        void *const output_device = resources.allocate(
            resources.root_backend, continuation_device, payload_bytes);
        ASSERT_NE(output_device, nullptr);
        ASSERT_TRUE(resources.root_backend->memset(
            output_device,
            0,
            payload_bytes,
            continuation_device.ordinal,
            resources.root_stream));

        const std::array<DeviceId, 1> endpoints{continuation_device};
        auto contribution_region =
            TransferEngine::instance().allocateMappedHostRegion(
                payload_bytes, endpoints);
        ASSERT_NE(contribution_region, nullptr);
        ASSERT_TRUE(contribution_region->isBound());

        MoEOverlayCanonicalRouteReturnTicketStorage storage;
        storage.bindFixedCapacity(
            /*layer_idx=*/5,
            route_capacity,
            d_model,
            continuation_device,
            workspace_generation,
            contribution_region);
        ASSERT_TRUE(storage.hasValidBoundIdentity());

        std::unique_ptr<IMoEKernel> kernel;
        if (continuation_device.is_cuda())
        {
            kernel = std::make_unique<CUDAMoEKernel>(
                continuation_device.ordinal);
        }
        else if (continuation_device.is_rocm())
        {
            kernel = std::make_unique<ROCmMoEKernel>(
                continuation_device.ordinal);
        }
        ASSERT_NE(kernel, nullptr);

        MoEOverlayCanonicalRouteTicketConsumeLaunch ticket{
            .control = storage.controlDeviceAlias(),
            .original_route_slots =
                storage.originalRouteSlotsDeviceAlias(),
            .compact_route_slots =
                storage.compactRouteSlotsDeviceAlias(),
            .compact_preweighted_contributions_fp32 =
                storage.contributionRowsDeviceAlias(),
            .canonical_route_contributions_fp32 =
                static_cast<float *>(output_device),
            .route_capacity = route_capacity,
            .d_model = d_model,
        };
        ASSERT_TRUE(ticket.valid());

        {
            auto abandoned = storage.arm(first_residency_epoch - 1u);
            ASSERT_TRUE(abandoned);
        }
        EXPECT_FALSE(storage.payloadReady());
        EXPECT_TRUE(storage.arm(first_residency_epoch - 1u))
            << "dropping an unpublished lease must restore quiescent ownership";

        std::vector<float> actual(element_count, 0.0f);
        std::vector<float> expected(element_count, 0.0f);
        for (std::size_t replay = 0u; replay < replay_count; ++replay)
        {
            const std::uint64_t residency_epoch =
                first_residency_epoch + replay;
            auto publication = storage.arm(residency_epoch);
            ASSERT_TRUE(publication)
                << "replay=" << replay;
            EXPECT_FALSE(storage.arm(residency_epoch + 1u))
                << "a producer cannot arm one ticket twice";

            std::fill(expected.begin(), expected.end(), 0.0f);
            auto *const original_slots = storage.originalRouteSlotsHost();
            auto *const compact_slots = storage.compactRouteSlotsHost();
            float *const contribution_rows = storage.contributionRowsHost();
            ASSERT_NE(original_slots, nullptr);
            ASSERT_NE(compact_slots, nullptr);
            ASSERT_NE(contribution_rows, nullptr);
            for (std::size_t entry = 0u; entry < route_capacity; ++entry)
            {
                const std::size_t original_slot =
                    (entry + replay) % route_capacity;
                const std::size_t compact_slot =
                    (entry + 2u * replay + 1u) % route_capacity;
                original_slots[entry] =
                    static_cast<std::int32_t>(original_slot);
                compact_slots[entry] =
                    static_cast<std::int32_t>(compact_slot);
                for (std::size_t column = 0u;
                     column < static_cast<std::size_t>(d_model);
                     ++column)
                {
                    const float value = static_cast<float>(
                        10000u * (replay + 1u) + 100u * entry + column);
                    contribution_rows[
                        compact_slot * static_cast<std::size_t>(d_model) +
                        column] = value;
                    expected[
                        original_slot * static_cast<std::size_t>(d_model) +
                        column] = value;
                }
            }

            ASSERT_TRUE(publication.publish(route_capacity))
                << "replay=" << replay;
            ASSERT_TRUE(storage.payloadReadyFor(residency_epoch));
            EXPECT_FALSE(publication.publish(route_capacity))
                << "publication requires one fresh arm transition";
            EXPECT_FALSE(storage.arm(residency_epoch + 1u))
                << "mapped payload cannot be overwritten before GPU acknowledgement";

            ASSERT_TRUE(kernel->consumeMoEOverlayCanonicalRouteTicket(
                MoEKernelLaunchContext{.stream = resources.root_stream},
                ticket));
            ASSERT_TRUE(resources.root_backend->recordEvent(
                resources.root_terminal,
                continuation_device.ordinal,
                resources.root_stream));
            ASSERT_TRUE(awaitEvent(
                resources.root_backend,
                resources.root_terminal,
                std::chrono::seconds(5),
                continuation_device.ordinal));
            EXPECT_FALSE(storage.payloadReady())
                << "GPU did not acknowledge replay=" << replay;

            ASSERT_TRUE(resources.root_backend->deviceToHost(
                actual.data(),
                output_device,
                payload_bytes,
                continuation_device.ordinal,
                resources.root_stream));
            for (std::size_t element = 0u; element < element_count; ++element)
            {
                EXPECT_FLOAT_EQ(actual[element], expected[element])
                    << "replay=" << replay << " element=" << element;
            }
        }
    }

    /**
     * @brief Exercise five device-owned epochs, including placement invariance.
     *
     * The final two epochs retain identical per-route bytes while changing the
     * owning participant. Their cancellation-heavy values distinguish the
     * canonical original-slot fold from the obsolete participant-subtotal
     * arithmetic that made Dynamic placement numerically observable.
     */
    void runSparseCanonicalRouteExchange(
        DeviceId root_device,
        DeviceId producer_device)
    {
        constexpr std::uint32_t rows = 64u;
        constexpr std::uint32_t top_k = 8u;
        constexpr std::uint32_t d_model = 512u;
        constexpr std::uint32_t route_slots = rows * top_k;
        constexpr std::int32_t root_participant = 11;
        constexpr std::int32_t producer_participant = 37;
        constexpr std::uint32_t epochs = 5u;
        constexpr std::uint32_t invariant_epoch_begin = 3u;

        SparseRouteExchangeResources resources(
            root_device, producer_device);
        ASSERT_NE(resources.root_backend, nullptr);
        ASSERT_NE(resources.producer_backend, nullptr);
        ASSERT_GT(resources.root_backend->deviceCount(), root_device.ordinal);
        ASSERT_GT(
            resources.producer_backend->deviceCount(),
            producer_device.ordinal);

        resources.root_stream = resources.root_backend->createStream(
            root_device.ordinal);
        resources.producer_stream =
            resources.producer_backend->createStream(
                producer_device.ordinal);
        resources.root_terminal = resources.root_backend->createEvent(
            root_device.ordinal);
        resources.producer_terminal =
            resources.producer_backend->createEvent(
                producer_device.ordinal);
        ASSERT_NE(resources.root_stream, nullptr);
        ASSERT_NE(resources.producer_stream, nullptr);
        ASSERT_NE(resources.root_terminal, nullptr);
        ASSERT_NE(resources.producer_terminal, nullptr);

        resources.exchange =
            std::make_unique<MoEOverlayNodeLocalRouteExchange>(
                MoEOverlayNodeLocalRouteExchange::Config{
                    .devices = {root_device, producer_device},
                    .root_device = root_device,
                    .identity = "real_device_sparse_canonical_route_test",
                });
        resources.exchange->materialize(
            {
                {.participant_id = root_participant, .device = root_device},
                {.participant_id = producer_participant,
                 .device = producer_device},
            },
            root_participant,
            rows,
            top_k,
            d_model);

        const std::size_t route_elements =
            static_cast<std::size_t>(route_slots) * d_model;
        std::vector<float> expected(
            static_cast<std::size_t>(rows) * d_model, 0.0f);
        /*
         * Change both the final assignment ledger and both candidate payloads
         * every epoch. This models a promoted expert becoming executable on a
         * new participant and proves that transport follows the live route
         * publication rather than a setup-time owner map. Distinct bytes on
         * both endpoints also expose a stale ledger or premature lane reuse.
         */
        std::array<std::vector<std::int32_t>, epochs>
            route_participant_epochs;
        std::array<std::vector<float>, epochs> root_route_epochs;
        std::array<std::vector<float>, epochs> producer_route_epochs;
        std::array<std::vector<float>, epochs> expected_epochs;
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            route_participant_epochs[epoch].resize(route_slots);
            root_route_epochs[epoch].resize(route_elements);
            producer_route_epochs[epoch].resize(route_elements);
            expected_epochs[epoch].assign(expected.size(), 0.0f);
            for (std::uint32_t slot = 0u; slot < route_slots; ++slot)
            {
                const bool producer_owns =
                    ((slot + epoch) % 3u) != 0u;
                route_participant_epochs[epoch][slot] =
                    producer_owns
                        ? producer_participant
                        : root_participant;
                for (std::uint32_t column = 0u; column < d_model; ++column)
                {
                    const std::size_t route_index =
                        static_cast<std::size_t>(slot) * d_model + column;
                    const std::uint32_t route = slot % top_k;
                    const float base =
                        epoch < invariant_epoch_begin
                            ? static_cast<float>((slot % 13u) + 1u) +
                                  static_cast<float>(column % 8u) * 0.0625f +
                                  static_cast<float>(epoch) * 0.5f
                            : route == 0u
                                  ? 1.0e20f
                                  : route == 2u ? -1.0e20f : 1.0f;
                    root_route_epochs[epoch][route_index] = base;
                    producer_route_epochs[epoch][route_index] =
                        epoch < invariant_epoch_begin
                            ? base + 32.0f
                            : base;
                    expected_epochs[epoch][
                        static_cast<std::size_t>(slot / top_k) * d_model +
                        column] += producer_owns
                                       ? producer_route_epochs[epoch][route_index]
                                       : root_route_epochs[epoch][route_index];
                }
            }
        }
        ASSERT_EQ(
            std::memcmp(
                expected_epochs[invariant_epoch_begin].data(),
                expected_epochs[invariant_epoch_begin + 1u].data(),
                expected_epochs[invariant_epoch_begin].size() *
                    sizeof(float)),
            0)
            << "placement-invariance oracle changed canonical route order";
        for (std::uint32_t epoch = invariant_epoch_begin;
             epoch < epochs;
             ++epoch)
        {
            bool exposes_participant_grouping = false;
            for (std::uint32_t row = 0u;
                 row < rows && !exposes_participant_grouping;
                 ++row)
            {
                for (std::uint32_t column = 0u;
                     column < d_model;
                     ++column)
                {
                    float root_subtotal = 0.0f;
                    float producer_subtotal = 0.0f;
                    for (std::uint32_t route = 0u;
                         route < top_k;
                         ++route)
                    {
                        const std::uint32_t slot = row * top_k + route;
                        const float value = root_route_epochs[epoch][
                            static_cast<std::size_t>(slot) * d_model +
                            column];
                        if (route_participant_epochs[epoch][slot] ==
                            root_participant)
                        {
                            root_subtotal += value;
                        }
                        else
                        {
                            producer_subtotal += value;
                        }
                    }
                    const float obsolete_grouped =
                        root_subtotal + producer_subtotal;
                    const float canonical = expected_epochs[epoch][
                        static_cast<std::size_t>(row) * d_model + column];
                    exposes_participant_grouping =
                        obsolete_grouped != canonical;
                    if (exposes_participant_grouping)
                        break;
                }
            }
            ASSERT_TRUE(exposes_participant_grouping)
                << "adversarial epoch no longer distinguishes participant grouping from canonical route order";
        }

        const auto &route_participants = route_participant_epochs.front();
        const auto &root_routes = root_route_epochs.front();
        const auto &producer_routes = producer_route_epochs.front();

        const std::size_t route_bytes = root_routes.size() * sizeof(float);
        const std::size_t participant_bytes =
            route_participants.size() * sizeof(std::int32_t);
        const std::size_t output_bytes = expected.size() * sizeof(float);
        void *const root_routes_device = resources.allocate(
            resources.root_backend, root_device, route_bytes);
        void *const producer_routes_device = resources.allocate(
            resources.producer_backend, producer_device, route_bytes);
        void *const root_participants_device = resources.allocate(
            resources.root_backend, root_device, participant_bytes);
        void *const producer_participants_device = resources.allocate(
            resources.producer_backend, producer_device, participant_bytes);
        void *const output_device = resources.allocate(
            resources.root_backend, root_device, output_bytes);
        void *const output_epochs_device = resources.allocate(
            resources.root_backend,
            root_device,
            output_bytes * epochs);
        void *const validation_device = resources.allocate(
            resources.root_backend, root_device, sizeof(std::int32_t));
        const auto root_peer_bindings =
            resources.exchange->rootPeerBindings(root_device);
        ASSERT_EQ(root_peer_bindings.size(), 1u);
        void *const peer_bindings_device = resources.allocate(
            resources.root_backend,
            root_device,
            root_peer_bindings.size() *
                sizeof(MoENodeLocalRoutePeerDeviceBinding));
        ASSERT_TRUE(std::all_of(
            resources.allocations.begin(), resources.allocations.end(),
            [](const DeviceAllocation &allocation)
            {
                return allocation.address != nullptr;
            }));

        ASSERT_TRUE(resources.root_backend->hostToDevice(
            root_routes_device,
            root_routes.data(),
            route_bytes,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->hostToDevice(
            producer_routes_device,
            producer_routes.data(),
            route_bytes,
            producer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(resources.root_backend->hostToDevice(
            root_participants_device,
            route_participants.data(),
            participant_bytes,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->hostToDevice(
            producer_participants_device,
            route_participants.data(),
            participant_bytes,
            producer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(resources.root_backend->hostToDevice(
            peer_bindings_device,
            root_peer_bindings.data(),
            root_peer_bindings.size() *
                sizeof(MoENodeLocalRoutePeerDeviceBinding),
            root_device.ordinal,
            resources.root_stream));

        auto root_kernel = kernelFor(root_device);
        auto producer_kernel = kernelFor(producer_device);
        ASSERT_NE(root_kernel, nullptr);
        ASSERT_NE(producer_kernel, nullptr);
        const MoEKernelLaunchContext root_launch{
            .stream = resources.root_stream,
        };
        const MoEKernelLaunchContext producer_launch{
            .stream = resources.producer_stream,
        };
        const MoENodeLocalRouteConsumeLaunch consume{
            .peers = static_cast<const MoENodeLocalRoutePeerDeviceBinding *>(
                peer_bindings_device),
            .peer_count = 1u,
            .root_canonical_route_contributions =
                static_cast<const float *>(root_routes_device),
            .domain_assignment = {
                .participant_ids =
                    static_cast<const std::int32_t *>(
                        root_participants_device),
                .capacity = route_slots,
            },
            .external_route_source =
                MoEExternalCanonicalRouteSource::DeferredDenseMerge,
            .dense_output = static_cast<float *>(output_device),
            .validation_status =
                static_cast<std::int32_t *>(validation_device),
            .root_participant = root_participant,
            .physical_rows = rows,
            .top_k = top_k,
            .d_model = d_model,
        };
        const MoENodeLocalRoutePublishLaunch publish{
            .lane = resources.exchange->producerBinding(producer_device),
            .canonical_route_contributions =
                static_cast<const float *>(producer_routes_device),
            .domain_assignment = {
                .participant_ids =
                    static_cast<const std::int32_t *>(
                        producer_participants_device),
                .capacity = route_slots,
            },
            .live_route_slots = route_slots,
        };
        ASSERT_TRUE(consume.valid());
        ASSERT_TRUE(publish.valid());

        const auto submission_start = std::chrono::steady_clock::now();
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            ASSERT_TRUE(resources.root_backend->hostToDeviceOnStream(
                root_routes_device,
                root_route_epochs[epoch].data(),
                route_bytes,
                root_device.ordinal,
                resources.root_stream));
            ASSERT_TRUE(resources.root_backend->hostToDeviceOnStream(
                root_participants_device,
                route_participant_epochs[epoch].data(),
                participant_bytes,
                root_device.ordinal,
                resources.root_stream));
            // Submit the waiter first to prove the root graph does not rely on
            // host ordering. Both endpoint streams then advance through their
            // monotonic SPSC epochs without an intervening host observation.
            ASSERT_TRUE(root_kernel->acquireNodeLocalCanonicalRoutes(
                root_launch, consume));
            ASSERT_TRUE(root_kernel->stageNodeLocalCanonicalRoutes(
                root_launch, consume));
            ASSERT_TRUE(root_kernel->foldNodeLocalCanonicalRoutes(
                root_launch, consume));
            ASSERT_TRUE(resources.root_backend->deviceCopyAsync(
                static_cast<std::byte *>(output_epochs_device) +
                    static_cast<std::size_t>(epoch) * output_bytes,
                output_device,
                output_bytes,
                root_device.ordinal,
                resources.root_stream));
            ASSERT_TRUE(resources.producer_backend->hostToDeviceOnStream(
                producer_routes_device,
                producer_route_epochs[epoch].data(),
                route_bytes,
                producer_device.ordinal,
                resources.producer_stream));
            ASSERT_TRUE(resources.producer_backend->hostToDeviceOnStream(
                producer_participants_device,
                route_participant_epochs[epoch].data(),
                participant_bytes,
                producer_device.ordinal,
                resources.producer_stream));
            ASSERT_TRUE(producer_kernel->publishNodeLocalCanonicalRoutes(
                producer_launch, publish));
        }
        EXPECT_LT(
            std::chrono::steady_clock::now() - submission_start,
            std::chrono::seconds(2))
            << "route exchange submission blocked on device-owned progress";
        ASSERT_TRUE(resources.root_backend->recordEvent(
            resources.root_terminal,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->recordEvent(
            resources.producer_terminal,
            producer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(awaitEvent(
            resources.root_backend,
            resources.root_terminal,
            std::chrono::seconds(30),
            root_device.ordinal));
        ASSERT_TRUE(awaitEvent(
            resources.producer_backend,
            resources.producer_terminal,
            std::chrono::seconds(30),
            producer_device.ordinal));

        std::vector<float> actual(
            expected.size() * static_cast<std::size_t>(epochs), 0.0f);
        std::int32_t validation = -1;
        ASSERT_TRUE(resources.root_backend->deviceToHost(
            actual.data(),
            output_epochs_device,
            output_bytes * epochs,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.root_backend->deviceToHost(
            &validation,
            validation_device,
            sizeof(validation),
            root_device.ordinal,
            resources.root_stream));
        EXPECT_EQ(
            validation,
            static_cast<std::int32_t>(
                MoENodeLocalRouteExchangeCode::Success));
        ASSERT_EQ(
            actual.size(),
            expected.size() * static_cast<std::size_t>(epochs));
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            for (std::size_t index = 0u; index < expected.size(); ++index)
            {
                EXPECT_FLOAT_EQ(
                    actual[static_cast<std::size_t>(epoch) *
                               expected.size() +
                           index],
                    expected_epochs[epoch][index])
                    << "epoch=" << epoch << " index=" << index;
            }
        }
        EXPECT_EQ(
            std::memcmp(
                actual.data() +
                    static_cast<std::size_t>(invariant_epoch_begin) *
                        expected.size(),
                actual.data() +
                    static_cast<std::size_t>(invariant_epoch_begin + 1u) *
                        expected.size(),
                output_bytes),
            0)
            << "changing route ownership changed canonical FP32 output bytes";
    }

    /**
     * @brief Exercise three root-to-peer dense epochs through captured-copy APIs.
     *
     * The peer waiter is submitted before the root on every epoch. Immediate
     * submission proves that progress remains entirely device owned; exact
     * equality after the terminal events proves that TransferEngine's D2H/H2D
     * nodes are ordered by the monotonic channel rather than host timing.
     */
    void runDenseContinuationPublication(
        DeviceId root_device,
        DeviceId peer_device)
    {
        constexpr std::uint32_t rows = 64u;
        constexpr std::uint32_t top_k = 8u;
        constexpr std::uint32_t d_model = 512u;
        constexpr std::int32_t root_participant = 11;
        constexpr std::int32_t peer_participant = 37;
        constexpr std::uint32_t epochs = 3u;
        constexpr std::size_t elements =
            static_cast<std::size_t>(rows) * d_model;
        constexpr std::size_t bytes = elements * sizeof(float);

        SparseRouteExchangeResources resources(root_device, peer_device);
        ASSERT_NE(resources.root_backend, nullptr);
        ASSERT_NE(resources.producer_backend, nullptr);
        ASSERT_GT(resources.root_backend->deviceCount(), root_device.ordinal);
        ASSERT_GT(
            resources.producer_backend->deviceCount(), peer_device.ordinal);

        resources.root_stream = resources.root_backend->createStream(
            root_device.ordinal);
        resources.producer_stream =
            resources.producer_backend->createStream(peer_device.ordinal);
        resources.root_terminal = resources.root_backend->createEvent(
            root_device.ordinal);
        resources.producer_terminal =
            resources.producer_backend->createEvent(peer_device.ordinal);
        ASSERT_NE(resources.root_stream, nullptr);
        ASSERT_NE(resources.producer_stream, nullptr);
        ASSERT_NE(resources.root_terminal, nullptr);
        ASSERT_NE(resources.producer_terminal, nullptr);

        resources.exchange =
            std::make_unique<MoEOverlayNodeLocalRouteExchange>(
                MoEOverlayNodeLocalRouteExchange::Config{
                    .devices = {root_device, peer_device},
                    .root_device = root_device,
                    .identity = "real_device_dense_continuation_test",
                });
        resources.exchange->materialize(
            {
                {.participant_id = root_participant, .device = root_device},
                {.participant_id = peer_participant, .device = peer_device},
            },
            root_participant,
            rows,
            top_k,
            d_model);

        auto root_payload =
            resources.allocateTransferBuffer(root_device, bytes);
        auto peer_payload =
            resources.allocateTransferBuffer(peer_device, bytes);
        ASSERT_NE(root_payload, nullptr);
        ASSERT_NE(peer_payload, nullptr);
        ASSERT_TRUE(root_payload->isBound());
        ASSERT_TRUE(peer_payload->isBound());

        std::vector<float> expected(elements);
        for (std::size_t index = 0u; index < elements; ++index)
        {
            expected[index] =
                static_cast<float>((index % 257u) + 1u) * 0.03125f;
        }
        ASSERT_TRUE(resources.root_backend->hostToDevice(
            root_payload->mutableDeviceData(),
            expected.data(),
            bytes,
            root_device.ordinal,
            resources.root_stream));

        auto root_kernel = kernelFor(root_device);
        auto peer_kernel = kernelFor(peer_device);
        ASSERT_NE(root_kernel, nullptr);
        ASSERT_NE(peer_kernel, nullptr);
        const MoEKernelLaunchContext root_launch{
            .stream = resources.root_stream,
        };
        const MoEKernelLaunchContext peer_launch{
            .stream = resources.producer_stream,
        };
        const MoENodeLocalDensePublicationLaunch root_publication{
            .binding =
                resources.exchange->densePublicationBinding(root_device),
            .element_count = elements,
        };
        const MoENodeLocalDensePublicationLaunch peer_publication{
            .binding =
                resources.exchange->densePublicationBinding(peer_device),
            .element_count = elements,
        };
        ASSERT_TRUE(root_publication.valid());
        ASSERT_TRUE(root_publication.binding.isRoot());
        ASSERT_TRUE(peer_publication.valid());
        ASSERT_FALSE(peer_publication.binding.isRoot());

        const auto region = resources.exchange->mappedRegion();
        const std::size_t payload_offset =
            resources.exchange->densePublicationPayloadOffset();
        ASSERT_NE(region, nullptr);
        const auto submission_start = std::chrono::steady_clock::now();
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            ASSERT_TRUE(peer_kernel->beginNodeLocalDensePublicationConsume(
                peer_launch, peer_publication));
            TransferEngine::instance().enqueueMappedHostToDevice(
                *region,
                payload_offset,
                *peer_payload,
                0u,
                bytes,
                peer_device,
                resources.producer_stream);
            ASSERT_TRUE(peer_kernel->finishNodeLocalDensePublicationConsume(
                peer_launch, peer_publication));

            ASSERT_TRUE(root_kernel->beginNodeLocalDensePublication(
                root_launch, root_publication));
            TransferEngine::instance().enqueueDeviceToMappedHost(
                *root_payload,
                0u,
                *region,
                payload_offset,
                bytes,
                root_device,
                resources.root_stream);
            ASSERT_TRUE(root_kernel->finishNodeLocalDensePublication(
                root_launch, root_publication));
        }
        EXPECT_LT(
            std::chrono::steady_clock::now() - submission_start,
            std::chrono::seconds(2))
            << "dense publication submission blocked on peer progress";

        ASSERT_TRUE(resources.root_backend->recordEvent(
            resources.root_terminal,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->recordEvent(
            resources.producer_terminal,
            peer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(awaitEvent(
            resources.root_backend,
            resources.root_terminal,
            std::chrono::seconds(30),
            root_device.ordinal));
        ASSERT_TRUE(awaitEvent(
            resources.producer_backend,
            resources.producer_terminal,
            std::chrono::seconds(30),
            peer_device.ordinal));

        std::vector<float> actual(elements, 0.0f);
        ASSERT_TRUE(resources.producer_backend->deviceToHost(
            actual.data(),
            peer_payload->deviceData(),
            bytes,
            peer_device.ordinal,
            resources.producer_stream));
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0u; index < actual.size(); ++index)
            EXPECT_FLOAT_EQ(actual[index], expected[index]) << "index=" << index;
    }
} // namespace

TEST(Test__MappedActivationPacketCUDAAndROCm,
     PlannerSelectedCUDAContinuationROCmFollowerIsExact)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    runRoleAssignment<4>(DeviceId::cuda(0), DeviceId::rocm(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     PlannerSelectedROCmContinuationCUDAFollowerIsExact)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    runRoleAssignment<4>(DeviceId::rocm(0), DeviceId::cuda(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SingleRowCUDAContinuationROCmFollowerFusesTimelineAndIsExact)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    runRoleAssignment<1>(DeviceId::cuda(0), DeviceId::rocm(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SingleRowROCmContinuationCUDAFollowerFusesTimelineAndIsExact)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    runRoleAssignment<1>(DeviceId::rocm(0), DeviceId::cuda(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     CUDAContinuationWaitsForNewGenerationAfterStaleAdmission)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires CUDA and ROCm devices";
    runStaleAdmissionABAProof(DeviceId::cuda(0), DeviceId::rocm(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     ROCmContinuationWaitsForNewGenerationAfterStaleAdmission)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires CUDA and ROCm devices";
    runStaleAdmissionABAProof(DeviceId::rocm(0), DeviceId::cuda(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     CPUCanonicalRouteTicketReusesSafelyOnCUDA)
{
    IBackend *const cuda = getCUDABackend();
    ASSERT_NE(cuda, nullptr);
    if (cuda->deviceCount() < 1)
        GTEST_SKIP() << "Requires one CUDA device";
    runCanonicalRouteTicketReuseProof(DeviceId::cuda(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     CPUCanonicalRouteTicketReusesSafelyOnROCm)
{
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(rocm, nullptr);
    if (rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires one ROCm device";
    runCanonicalRouteTicketReuseProof(DeviceId::rocm(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossTwoCUDADevicesAreExactAndAsync)
{
    IBackend *const cuda = getCUDABackend();
    ASSERT_NE(cuda, nullptr);
    if (cuda->deviceCount() < 2)
        GTEST_SKIP() << "Requires two CUDA devices";
    runSparseCanonicalRouteExchange(DeviceId::cuda(0), DeviceId::cuda(1));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossTwoROCmDevicesAreExactAndAsync)
{
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(rocm, nullptr);
    if (rocm->deviceCount() < 2)
        GTEST_SKIP() << "Requires two ROCm devices";
    runSparseCanonicalRouteExchange(DeviceId::rocm(0), DeviceId::rocm(1));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossCUDAAndROCmAreExactAndAsync)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires CUDA and ROCm devices";
    runSparseCanonicalRouteExchange(DeviceId::cuda(0), DeviceId::rocm(0));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     DenseContinuationPublicationAcrossTwoCUDADevicesIsExactAndAsync)
{
    IBackend *const cuda = getCUDABackend();
    ASSERT_NE(cuda, nullptr);
    if (cuda->deviceCount() < 2)
        GTEST_SKIP() << "Requires two CUDA devices";
    runDenseContinuationPublication(DeviceId::cuda(0), DeviceId::cuda(1));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     DenseContinuationPublicationAcrossTwoROCmDevicesIsExactAndAsync)
{
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(rocm, nullptr);
    if (rocm->deviceCount() < 2)
        GTEST_SKIP() << "Requires two ROCm devices";
    runDenseContinuationPublication(DeviceId::rocm(0), DeviceId::rocm(1));
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     DenseContinuationPublicationAcrossCUDAAndROCmIsExactAndAsync)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
        GTEST_SKIP() << "Requires CUDA and ROCm devices";
    runDenseContinuationPublication(DeviceId::cuda(0), DeviceId::rocm(0));
}
