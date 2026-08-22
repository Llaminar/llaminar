/**
 * @file MoEOverlayActivationPacketStages.cpp
 * @brief Graph-stage implementation for mapped ExpertOverlay packet kernels.
 *
 * Every execute method validates the exact planner-selected device and obtains
 * its stream from IComputeStage::requireGPUStream(). Backend packet facades own
 * no host wait or synchronization; they enqueue fixed-shape kernels whose
 * mapped system-memory visibility is ordered by the parent timeline graph.
 */

#include "MoEOverlayActivationPacketStages.h"

#include "../../../backends/BackendManager.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace llaminar2
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    namespace
    {
        /** @return Whether the selected execution backend supports native graphs. */
        bool isGPUBackend(ComputeBackendType backend) noexcept
        {
            return backend == ComputeBackendType::GPU_CUDA ||
                   backend == ComputeBackendType::GPU_ROCM;
        }

        /** @return Whether @p tensor can expose at least @p elements of FP32. */
        bool isFP32Capacity(
            const TensorBase *tensor,
            std::size_t elements) noexcept
        {
            return tensor && tensor->native_type() == TensorType::FP32 &&
                   tensor->numel() >= elements;
        }

        /** @return Overflow-free multiplication for small model geometry. */
        bool checkedElements(
            std::int32_t rows,
            std::int32_t width,
            std::size_t *elements) noexcept
        {
            if (!elements || rows <= 0 || width <= 0)
                return false;
            const auto lhs = static_cast<std::size_t>(rows);
            const auto rhs = static_cast<std::size_t>(width);
            if (lhs > static_cast<std::size_t>(-1) / rhs)
                return false;
            *elements = lhs * rhs;
            return true;
        }

        /** @return Overflow-free `[rows * top_k, d_model]` route-bank size. */
        bool checkedCanonicalRouteElements(
            std::int32_t rows,
            std::int32_t top_k,
            std::int32_t d_model,
            std::size_t *elements) noexcept
        {
            std::size_t route_slots = 0u;
            if (!checkedElements(rows, top_k, &route_slots) ||
                !elements || d_model <= 0)
            {
                return false;
            }
            const auto width = static_cast<std::size_t>(d_model);
            if (route_slots > static_cast<std::size_t>(-1) / width)
                return false;
            *elements = route_slots * width;
            return true;
        }

        /** @return Whether one stage exactly matches its immutable lane header. */
        bool validLaneStage(
            const MoEOverlayMappedActivationDeviceLane &lane,
            DeviceId device,
            std::int32_t physical_rows,
            std::uint32_t stage_ordinal,
            std::int32_t model_layer_index) noexcept
        {
            return lane.valid() && lane.device == device && device.is_gpu() &&
                   physical_rows > 0 && model_layer_index >= 0 &&
                   lane.control_host->channel.stage_count > stage_ordinal &&
                   lane.control_host->channel.target_participant_id ==
                       lane.target_participant_id;
        }

        /** @return Exact launch context for one already-bound stage stream. */
        MoEKernelLaunchContext packetLaunchContext(void *stream) noexcept
        {
            return MoEKernelLaunchContext{
                .stream = stream,
                .workspace = nullptr,
            };
        }

        /** @brief Validate exact context ownership shared by all packet stages. */
        bool validateExecutionContext(
            const IDeviceContext *ctx,
            DeviceId device,
            const char *stage_name)
        {
            if (ctx && ctx->deviceId() == device && device.is_gpu())
                return true;
            LOG_ERROR("[" << stage_name
                           << "] Exact planner-selected GPU context is required");
            return false;
        }

        /** @brief Validate exact context and stream during capture preparation. */
        bool validateCaptureContext(
            const IDeviceContext *ctx,
            DeviceId device,
            void *stream,
            const char *stage_name)
        {
            if (ctx && ctx->deviceId() == device && device.is_gpu() && stream)
                return true;
            LOG_ERROR("[" << stage_name
                           << "] Capture preparation requires the exact GPU context and non-null stream");
            return false;
        }

        /** @brief Add one tensor descriptor without duplicating stage code. */
        void addInputRequirement(
            StageBufferRequirements &requirements,
            const char *name,
            const TensorBase *tensor)
        {
            if (tensor)
            {
                requirements.addInput(
                    name,
                    tensor->shape(),
                    toBufferTensorType(tensor->native_type()));
            }
        }

        /** @brief Add one output descriptor without duplicating stage code. */
        void addOutputRequirement(
            StageBufferRequirements &requirements,
            const char *name,
            const TensorBase *tensor)
        {
            if (tensor)
            {
                requirements.addOutput(
                    name,
                    tensor->shape(),
                    toBufferTensorType(tensor->native_type()));
            }
        }

        /** @brief Add one in/out descriptor without duplicating stage code. */
        void addInoutRequirement(
            StageBufferRequirements &requirements,
            const char *name,
            const TensorBase *tensor)
        {
            if (tensor)
            {
                requirements.addInout(
                    name,
                    tensor->shape(),
                    toBufferTensorType(tensor->native_type()));
            }
        }

        /** @return Fixed matrix bytes embedded by one captured packet stage. */
        bool fixedPayloadBytes(
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::int32_t physical_rows,
            std::size_t *bytes) noexcept
        {
            if (!bytes || physical_rows <= 0 || lane.dispatch.d_model <= 0)
                return false;
            const auto rows = static_cast<std::size_t>(physical_rows);
            const auto width = static_cast<std::size_t>(lane.dispatch.d_model);
            constexpr auto maximum = static_cast<std::size_t>(-1);
            if (rows > maximum / width ||
                rows * width > maximum / sizeof(float))
            {
                return false;
            }
            *bytes = rows * width * sizeof(float);
            return true;
        }

        /** @return Whether fixed geometry admits the one-node packet protocol. */
        bool useSingleRowDirectPacket(
            const MoEOverlayMappedActivationDispatchPayloadView &payload)
            noexcept
        {
            return payload.valid() &&
                   payload.selection.physical_rows == 1 &&
                   payload.selection.usesCompactRows();
        }

        /** @brief Resolve one fused system-acquire edge through TransferEngine. */
        bool bindKernelTimelineWait(
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::size_t signal_offset,
            std::uint64_t value,
            MoEOverlayActivationTimelineWaitDeviceBinding *binding,
            const char *role)
        {
            if (!binding)
                return false;
            try
            {
                const auto resolved =
                    TransferEngine::instance().bindMappedTimelineKernelWait64(
                        *lane.mapped_region,
                        signal_offset,
                        value,
                        lane.device);
                *binding = {
                    .signal = resolved.deviceSignal(),
                    .value = resolved.value(),
                };
                return binding->valid();
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationPacketStages] "
                          << role << " fused timeline acquire binding failed: "
                          << error.what());
                return false;
            }
        }

        /** @brief Resolve one fused system-release edge through TransferEngine. */
        bool bindKernelTimelinePublication(
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::size_t signal_offset,
            std::uint64_t value,
            MoEOverlayActivationTimelinePublishDeviceBinding *binding,
            const char *role)
        {
            if (!binding)
                return false;
            try
            {
                const auto resolved = TransferEngine::instance()
                                          .bindMappedTimelineKernelPublish64(
                                              *lane.mapped_region,
                                              signal_offset,
                                              value,
                                              lane.device);
                *binding = {
                    .signal = resolved.deviceSignal(),
                    .value = resolved.value(),
                };
                return binding->valid();
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationPacketStages] "
                          << role
                          << " fused timeline publication binding failed: "
                          << error.what());
                return false;
            }
        }

        /**
         * @brief Publish a fixed tensor range directly into shared pages.
         *
         * This overload is the one-copy shared-physical activation path. It
         * intentionally bypasses per-lane device bounce buffers while retaining
         * TransferEngine as the sole residency and exact-stream authority.
         */
        bool exportBulkPayload(
            const MoEOverlayMappedActivationDeviceLane &lane,
            const AcquiredDeviceTransferInput &source,
            std::size_t source_offset,
            std::size_t destination_offset,
            std::size_t bytes,
            const char *role)
        {
            try
            {
                TransferEngine::instance().enqueueDeviceToMappedHost(
                    source,
                    source_offset,
                    *lane.mapped_region,
                    destination_offset,
                    bytes);
                if (PerfStatsCollector::isDomainEnabled(
                        "moe_overlay_activation_epoch"))
                {
                    PerfStatsCollector::addCounter(
                        "moe_overlay_activation_epoch",
                        "shared_physical_dispatch_d2h_bytes",
                        static_cast<double>(bytes),
                        "graph_setup",
                        lane.device.toString(),
                        {{"host_blocking", "false"},
                         {"payload_layout", "shared_physical_rows"},
                         {"payload_path", "shared_physical_mapped"},
                         {"role", role}});
                }
                return true;
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationPacketStages] "
                          << role << " tensor D2H failed: " << error.what());
                return false;
            }
        }

        /** @brief Publish tensor bytes when producer and DMA share one stream. */
        bool exportTensorBulkPayload(
            const MoEOverlayMappedActivationDeviceLane &lane,
            TensorBase *source,
            std::size_t source_offset,
            std::size_t destination_offset,
            std::size_t bytes,
            void *stream,
            const char *role)
        {
            try
            {
                TransferEngine::instance().enqueueDeviceToMappedHost(
                    source,
                    source_offset,
                    *lane.mapped_region,
                    destination_offset,
                    bytes,
                    lane.device,
                    stream);
                if (PerfStatsCollector::isDomainEnabled(
                        "moe_overlay_activation_epoch"))
                {
                    PerfStatsCollector::addCounter(
                        "moe_overlay_activation_epoch",
                        "shared_physical_dispatch_d2h_bytes",
                        static_cast<double>(bytes),
                        "graph_setup",
                        lane.device.toString(),
                        {{"host_blocking", "false"},
                         {"payload_layout", "shared_physical_rows"},
                         {"payload_path", "shared_physical_mapped"},
                         {"role", role}});
                }
                return true;
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationPacketStages] "
                          << role << " tensor D2H failed: " << error.what());
                return false;
            }
        }

        /**
         * @brief Append one peer-publication acquire to the complete endpoint graph.
         *
         * CUDA lowers this to a batch-memory graph node and HIP lowers it to a
         * one-wave system-acquire kernel. Both backends update the active stream
         * capture frontier, so the following packet kernel or copy engine node
         * cannot issue until the peer-owned value is visible. Outside capture the
         * same call remains an asynchronous exact-stream wait; there is no host
         * polling or synchronization in either case.
         */
        bool waitForMappedTimeline(
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::size_t signal_offset,
            std::uint64_t value,
            void *stream,
            const char *role)
        {
            try
            {
                TransferEngine::instance().enqueueMappedTimelineWait64(
                    *lane.mapped_region,
                    signal_offset,
                    value,
                    lane.device,
                    stream);
                return true;
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationPacketStages] "
                          << role << " timeline acquire failed: "
                          << error.what());
                return false;
            }
        }

        /**
         * @brief Append one system-release publication after packet production.
         *
         * The publication is inserted on the same exact stream as the metadata
         * kernel and optional bulk DMA. A peer can therefore use one 64-bit value
         * as the release/acquire edge for the complete packet without observing a
         * partially copied matrix or requiring an event visible to the host.
         */
        bool publishMappedTimeline(
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::size_t signal_offset,
            std::uint64_t value,
            void *stream,
            const char *role)
        {
            try
            {
                TransferEngine::instance().enqueueMappedTimelinePublish64(
                    *lane.mapped_region,
                    signal_offset,
                    value,
                    lane.device,
                    stream);
                return true;
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationPacketStages] "
                          << role << " timeline publication failed: "
                          << error.what());
                return false;
            }
        }

        /** @brief Join asynchronous grant initialization before capture begins. */
        bool joinGrantInitialization(
            const MoEOverlayMappedActivationDeviceLane &lane,
            void *stream,
            const char *role)
        {
            IBackend *const backend = getBackendFor(lane.device);
            if (backend && stream && lane.grant_initialization_event &&
                backend->streamWaitEvent(
                    stream,
                    lane.grant_initialization_event,
                    lane.device.gpu_ordinal()))
            {
                return true;
            }
            LOG_ERROR("[MoEOverlayActivationPacketStages] "
                      << role
                      << " could not join asynchronous grant initialization");
            return false;
        }

        /**
         * @brief Gate the first stage of one lane on scheduler admission.
         *
         * Grant initialization must already be joined before native capture;
         * an eager diagnostic execution may establish the same edge here.
         * Admission itself remains a future device timeline wait and therefore
         * belongs inside the retained transaction.
         */
        bool waitForAdmissionIfFirstStage(
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::uint32_t stage_ordinal,
            void *stream,
            bool grant_initialization_joined,
            const char *role)
        {
            if (stage_ordinal != 0u)
                return true;
            if (!grant_initialization_joined &&
                !joinGrantInitialization(lane, stream, role))
            {
                return false;
            }
            return waitForMappedTimeline(
                lane,
                lane.admission_signal_offset,
                kMoEOverlayActivationAdmissionTimeline,
                stream,
                role);
        }

        /**
         * @brief Upload one immutable trivially-copyable launch array before capture.
         *
         * INT32Tensor is used only as the canonical RAII allocation owner. The
         * packet ABI remains typed at both the host construction site and kernel
         * boundary; padding is cleared so tools never inspect indeterminate
         * setup bytes.
         */
        template <typename Launch>
        bool uploadPersistentLaunchArray(
            const std::vector<Launch> &launches,
            DeviceId device,
            void *stream,
            std::shared_ptr<MoEOverlayPersistentGraphStorage> *storage,
            const char *role)
        {
            static_assert(std::is_trivially_copyable_v<Launch>);
            static_assert(sizeof(Launch) % alignof(std::int32_t) == 0u);
            if (!storage || launches.empty() || !device.is_gpu() || !stream)
                return false;
            try
            {
                const std::size_t bytes = launches.size() * sizeof(Launch);
                const std::size_t words =
                    (bytes + sizeof(std::int32_t) - 1u) /
                    sizeof(std::int32_t);
                if (!*storage)
                {
                    *storage = std::make_shared<
                        MoEOverlayPersistentGraphStorage>(
                        MoEOverlayPersistentGraphStorage::Config{
                            .device = device,
                            .type = MoEOverlayGraphStorageType::Int32,
                            .shape = {words},
                            .immutable_input = true,
                            .identity = role ? role : "activation_launches",
                        });
                }
                return (*storage)->sizeBytes() == bytes &&
                       (*storage)->publishImmutableBytes(
                           launches.data(), bytes, stream);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationPacketStages] Failed to prepare "
                          << role << " descriptor array: " << error.what());
                return false;
            }
        }

        /** @return Whether every lane is unique and has matching captured geometry. */
        bool validLaneTransaction(
            const std::vector<MoEOverlayMappedActivationDeviceLane> &lanes,
            DeviceId device,
            std::int32_t physical_rows,
            std::uint32_t stage_ordinal,
            std::int32_t model_layer_index,
            std::int32_t *d_model,
            std::int32_t *top_k) noexcept
        {
            if (lanes.empty() || physical_rows <= 0 || !d_model || !top_k)
                return false;
            std::unordered_set<std::int32_t> participants;
            *d_model = lanes.front().dispatch.d_model;
            *top_k = lanes.front().dispatch.top_k;
            for (const auto &lane : lanes)
            {
                if (!validLaneStage(
                        lane,
                        device,
                        physical_rows,
                        stage_ordinal,
                        model_layer_index) ||
                    lane.dispatch.d_model != *d_model ||
                    lane.returned.d_model != *d_model ||
                    lane.dispatch.top_k != *top_k ||
                    !participants.insert(lane.target_participant_id).second)
                {
                    return false;
                }
            }
            return *d_model > 0 && *top_k > 0;
        }

        /** @return Whether fused decode can publish all lanes in one kernel grid. */
        bool validOneRowLaneBatch(
            const std::vector<MoEOverlayMappedActivationDeviceLane> &lanes,
            DeviceId device,
            std::uint32_t stage_ordinal,
            std::int32_t model_layer_index,
            std::int32_t *d_model,
            std::int32_t *top_k) noexcept
        {
            if (lanes.size() < 2u ||
                !validLaneTransaction(
                    lanes,
                    device,
                    /*physical_rows=*/1,
                    stage_ordinal,
                    model_layer_index,
                    d_model,
                    top_k))
            {
                return false;
            }
            return std::all_of(
                lanes.begin(),
                lanes.end(),
                [](const MoEOverlayMappedActivationDeviceLane &lane)
                {
                    const auto payload = lane.dispatchPayload(
                        /*physical_rows=*/1);
                    return payload.valid() &&
                           payload.selection.usesCompactRows();
                });
        }

    } // namespace

    MoEOverlayActivationLaneBatchState::MoEOverlayActivationLaneBatchState(
        DeviceId device,
        std::vector<MoEOverlayMappedActivationDeviceLane> lanes,
        std::int32_t physical_rows,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index)
        : device_(device),
          lanes_(std::move(lanes)),
          physical_rows_(physical_rows),
          stage_ordinal_(stage_ordinal),
          model_layer_index_(model_layer_index)
    {
        std::int32_t d_model = 0;
        std::int32_t top_k = 0;
        if (!validLaneTransaction(
                lanes_,
                device_,
                physical_rows_,
                stage_ordinal_,
                model_layer_index_,
                &d_model,
                &top_k))
        {
            throw std::invalid_argument(
                "MoE overlay activation lane transaction has inconsistent "
                "topology or captured geometry");
        }

        lane_shared_dispatch_payload_groups_.assign(lanes_.size(), -1);
        for (std::size_t lane_index = 0u;
             lane_index < lanes_.size();
             ++lane_index)
        {
            const auto &lane = lanes_[lane_index];
            const auto payload = lane.dispatchPayload(physical_rows_);
            if (!payload.valid())
            {
                throw std::invalid_argument(
                    "MoE overlay activation lane produced an invalid payload view");
            }
            if (!payload.selection.usesSharedPhysicalRows())
            {
                continue;
            }

            const auto found = std::find_if(
                shared_dispatch_payload_groups_.begin(),
                shared_dispatch_payload_groups_.end(),
                [&](const SharedDispatchPayloadGroup &group)
                {
                    return group.mapped_region == lane.mapped_region.get() &&
                           group.destination_offset ==
                               lane.shared_dispatch_hidden_offset;
                });
            std::size_t group_index = 0u;
            if (found == shared_dispatch_payload_groups_.end())
            {
                group_index = shared_dispatch_payload_groups_.size();
                shared_dispatch_payload_groups_.push_back({
                    .mapped_region = lane.mapped_region.get(),
                    .destination_offset =
                        lane.shared_dispatch_hidden_offset,
                    .publisher_lane_index = lane_index,
                    .lane_count = 1u,
                });
            }
            else
            {
                group_index = static_cast<std::size_t>(
                    std::distance(
                        shared_dispatch_payload_groups_.begin(), found));
                ++shared_dispatch_payload_groups_[group_index].lane_count;
            }
            if (group_index > static_cast<std::size_t>(
                                  std::numeric_limits<std::int32_t>::max()))
            {
                throw std::length_error(
                    "MoE overlay shared dispatch group count exceeds the typed lane index");
            }
            lane_shared_dispatch_payload_groups_[lane_index] =
                static_cast<std::int32_t>(group_index);
        }
    }

    MoEOverlayActivationLaneBatchState::~MoEOverlayActivationLaneBatchState()
    {
        release();
    }

    std::int32_t MoEOverlayActivationLaneBatchState::dModel() const noexcept
    {
        return lanes_.empty() ? 0 : lanes_.front().dispatch.d_model;
    }

    std::int32_t MoEOverlayActivationLaneBatchState::topK() const noexcept
    {
        return lanes_.empty() ? 0 : lanes_.front().dispatch.top_k;
    }

    bool MoEOverlayActivationLaneBatchState::usesAsynchronousLanes() const noexcept
    {
        /* The one-row direct packet has a purpose-built topology grid. Every
         * other geometry needs a real fork so DMA and follower work can overlap
         * continuation-local expert execution. */
        if (physical_rows_ != 1)
            return true;
        return std::any_of(
            lanes_.begin(),
            lanes_.end(),
            [](const MoEOverlayMappedActivationDeviceLane &lane)
            {
                const auto payload = lane.dispatchPayload(
                    /*physical_rows=*/1);
                return !payload.valid() ||
                       !payload.selection.usesCompactRows();
            });
    }

    bool MoEOverlayActivationLaneBatchState::materializePersistentResources()
    {
        if (!usesAsynchronousLanes())
            return true;
        if (persistentResourcesReady())
            return true;

        const bool partial = event_backend_ || event_device_ordinal_ >= 0 ||
                             producer_ready_event_ || !lane_streams_.empty() ||
                             !lane_return_ready_events_.empty() ||
                             std::any_of(
                                 shared_dispatch_payload_groups_.begin(),
                                 shared_dispatch_payload_groups_.end(),
                                 [](const SharedDispatchPayloadGroup &group)
                                 { return group.ready_event != nullptr; });
        if (partial)
        {
            LOG_ERROR("[MoEOverlayActivationLaneBatchState] Refusing to repair a partial persistent resource set"
                      << " device=" << device_.toString()
                      << " layer=" << model_layer_index_
                      << " ordinal=" << stage_ordinal_);
            return false;
        }

        event_backend_ = getBackendFor(device_);
        if (!event_backend_)
        {
            LOG_ERROR("[MoEOverlayActivationLaneBatchState] Could not resolve backend for "
                      << device_.toString());
            release();
            return false;
        }
        try
        {
            event_device_ordinal_ = device_.gpu_ordinal();
            auto &gpu_ctx =
                GPUDeviceContextPool::instance().getContext(device_);
            lane_streams_.reserve(lanes_.size());
            lane_return_ready_events_.reserve(lanes_.size());
            for (std::size_t lane_index = 0u;
                 lane_index < lanes_.size();
                 ++lane_index)
            {
                /* Lane ordinals intentionally bound stream count across layers
                 * and graph families. Unique per-layer events retain exact graph
                 * producer/consumer identity even though streams are shared. */
                void *const lane_stream =
                    gpu_ctx.getOrCreateAuxiliaryStream(
                        "moe_overlay_activation_lane:" +
                        std::to_string(lane_index));
                void *const return_ready =
                    event_backend_->createEvent(event_device_ordinal_);
                if (!lane_stream || !return_ready)
                {
                    if (return_ready)
                    {
                        event_backend_->destroyEvent(
                            return_ready, event_device_ordinal_);
                    }
                    throw std::runtime_error(
                        "could not create lane stream/event pair");
                }
                lane_streams_.push_back(lane_stream);
                lane_return_ready_events_.push_back(return_ready);
            }
            producer_ready_event_ =
                event_backend_->createEvent(event_device_ordinal_);
            if (!producer_ready_event_)
            {
                throw std::runtime_error(
                    "could not create main-to-lane fork event");
            }
            for (auto &group : shared_dispatch_payload_groups_)
            {
                group.ready_event =
                    event_backend_->createEvent(event_device_ordinal_);
                if (!group.ready_event)
                {
                    throw std::runtime_error(
                        "could not create shared dispatch publication event");
                }
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[MoEOverlayActivationLaneBatchState] Persistent resource materialization failed: "
                      << error.what());
            release();
            return false;
        }
        return persistentResourcesReady();
    }

    bool MoEOverlayActivationLaneBatchState::persistentResourcesReady() const noexcept
    {
        if (!usesAsynchronousLanes())
            return true;
        return event_backend_ && event_device_ordinal_ >= 0 &&
               producer_ready_event_ &&
               lane_streams_.size() == lanes_.size() &&
               lane_return_ready_events_.size() == lanes_.size() &&
               lane_shared_dispatch_payload_groups_.size() == lanes_.size() &&
               std::all_of(
                   lane_streams_.begin(),
                   lane_streams_.end(),
                   [](const void *stream) { return stream != nullptr; }) &&
               std::all_of(
                   lane_return_ready_events_.begin(),
                   lane_return_ready_events_.end(),
                   [](const void *event) { return event != nullptr; }) &&
               std::all_of(
                   shared_dispatch_payload_groups_.begin(),
                   shared_dispatch_payload_groups_.end(),
                   [](const SharedDispatchPayloadGroup &group)
                   { return group.ready_event != nullptr; });
    }

    void MoEOverlayActivationLaneBatchState::release()
    {
        if (event_backend_ && event_device_ordinal_ >= 0)
        {
            if (producer_ready_event_)
            {
                event_backend_->destroyEvent(
                    producer_ready_event_, event_device_ordinal_);
            }
            for (void *event : lane_return_ready_events_)
            {
                if (event)
                    event_backend_->destroyEvent(event, event_device_ordinal_);
            }
            for (auto &group : shared_dispatch_payload_groups_)
            {
                if (group.ready_event)
                {
                    event_backend_->destroyEvent(
                        group.ready_event, event_device_ordinal_);
                }
            }
        }
        producer_ready_event_ = nullptr;
        lane_return_ready_events_.clear();
        for (auto &group : shared_dispatch_payload_groups_)
            group.ready_event = nullptr;
        /* Auxiliary streams remain owned by IWorkerGPUContext. */
        lane_streams_.clear();
        event_backend_ = nullptr;
        event_device_ordinal_ = -1;
    }

    void *MoEOverlayActivationLaneBatchState::laneStream(
        std::size_t index) const
    {
        if (index >= lane_streams_.size())
            throw std::out_of_range("MoE overlay activation lane stream index");
        return lane_streams_[index];
    }

    void *MoEOverlayActivationLaneBatchState::laneReturnReadyEvent(
        std::size_t index) const
    {
        if (index >= lane_return_ready_events_.size())
        {
            throw std::out_of_range(
                "MoE overlay activation return event index");
        }
        return lane_return_ready_events_[index];
    }

    bool MoEOverlayActivationLaneBatchState::laneUsesSharedDispatchPayload(
        std::size_t index) const
    {
        if (index >= lane_shared_dispatch_payload_groups_.size())
            throw std::out_of_range("MoE overlay shared dispatch lane index");
        return lane_shared_dispatch_payload_groups_[index] >= 0;
    }

    bool MoEOverlayActivationLaneBatchState::
        lanePublishesSharedDispatchPayload(std::size_t index) const
    {
        if (!laneUsesSharedDispatchPayload(index))
            return false;
        const auto group_index = static_cast<std::size_t>(
            lane_shared_dispatch_payload_groups_[index]);
        return shared_dispatch_payload_groups_[group_index]
                   .publisher_lane_index == index;
    }

    std::size_t MoEOverlayActivationLaneBatchState::
        sharedDispatchPayloadLaneCount(std::size_t index) const
    {
        if (!laneUsesSharedDispatchPayload(index))
        {
            throw std::logic_error(
                "MoE overlay compact lane has no shared dispatch group");
        }
        return shared_dispatch_payload_groups_[static_cast<std::size_t>(
                   lane_shared_dispatch_payload_groups_[index])]
            .lane_count;
    }

    void *MoEOverlayActivationLaneBatchState::
        sharedDispatchPayloadReadyEvent(std::size_t index) const
    {
        if (!laneUsesSharedDispatchPayload(index))
        {
            throw std::logic_error(
                "MoE overlay compact lane has no shared dispatch event");
        }
        void *const event =
            shared_dispatch_payload_groups_[static_cast<std::size_t>(
                lane_shared_dispatch_payload_groups_[index])]
                .ready_event;
        if (!event)
        {
            throw std::logic_error(
                "MoE overlay shared dispatch event is not materialized");
        }
        return event;
    }

    MoEOverlayActivationDispatchPackStage::
        MoEOverlayActivationDispatchPackStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(KernelFactory::createMoEKernel(params_.device_id))
    {
    }

    MoEOverlayActivationDispatchPackStage::
        ~MoEOverlayActivationDispatchPackStage() = default;

    bool MoEOverlayActivationDispatchPackStage::hasStaticContract() const noexcept
    {
        std::size_t hidden_elements = 0u;
        std::size_t route_elements = 0u;
        return validLaneStage(
                   params_.lane,
                   params_.device_id,
                   params_.physical_rows,
                   params_.stage_ordinal,
                   params_.model_layer_index) &&
               params_.lane.target_participant_id >= 0 &&
               params_.placement.valid() &&
               checkedElements(
                   params_.physical_rows,
                   params_.lane.dispatch.d_model,
                   &hidden_elements) &&
               checkedElements(
                   params_.physical_rows,
                   params_.lane.dispatch.top_k,
                   &route_elements) &&
               isFP32Capacity(params_.hidden, hidden_elements) &&
               isFP32Capacity(params_.routing_indices, route_elements) &&
               isFP32Capacity(params_.routing_weights, route_elements) &&
               params_.lane.dispatch.row_capacity >=
                   static_cast<std::size_t>(params_.physical_rows) &&
               params_.lane.dispatch.entry_capacity >= route_elements &&
               moe_kernel_;
    }

    bool MoEOverlayActivationDispatchPackStage::hasFixedContract() const noexcept
    {
        return hasStaticContract() && params_.hidden->gpu_data_ptr() &&
               params_.routing_indices->gpu_data_ptr() &&
               params_.routing_weights->gpu_data_ptr();
    }

    bool MoEOverlayActivationDispatchPackStage::execute(IDeviceContext *ctx)
    {
        if (!validateExecutionContext(ctx, params_.device_id, name().c_str()) ||
            !hasFixedContract())
        {
            LOG_ERROR("[MoEOverlayActivationDispatchPackStage] Invalid fixed launch contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }
        void *const stream = requireGPUStream();
        std::size_t payload_bytes = 0u;
        const auto payload =
            params_.lane.dispatchPayload(params_.physical_rows);
        if (!payload.valid())
            return false;
        if (!fixedPayloadBytes(
                params_.lane, params_.physical_rows, &payload_bytes))
        {
            return false;
        }

        /* Transaction zero may not enter packet arithmetic until the scheduler
         * has armed this exact lane generation. This node becomes part of the
         * same full endpoint graph as every following dispatch/return edge. */
        if (!waitForAdmissionIfFirstStage(
                params_.lane,
                params_.stage_ordinal,
                stream,
                grant_initialization_joined_,
                "continuation admission"))
        {
            return false;
        }

        const MoEOverlayActivationDispatchPackLaunch launch{
            .hidden_rows_fp32 =
                static_cast<const float *>(params_.hidden->gpu_data_ptr()),
            .route_expert_ids_fp32 = static_cast<const float *>(
                params_.routing_indices->gpu_data_ptr()),
            .route_weights = static_cast<const float *>(
                params_.routing_weights->gpu_data_ptr()),
            .placement = params_.placement,
            .active_row_count_device = params_.active_row_count_device,
            .packet = payload.packet,
            .hidden_payload_layout = payload.selection.layout,
            .control = params_.lane.control_device,
            .grant = params_.lane.grant_device,
            .target_participant_id = params_.lane.target_participant_id,
            .physical_rows = params_.physical_rows,
            .stage_ordinal = params_.stage_ordinal,
            .model_layer_index = params_.model_layer_index,
        };
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(params_.stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(params_.stage_ordinal));

        /* A one-row direct packet is small enough for one cooperative block.
         * Let that block perform the final system release so packet production
         * and publication become one captured graph node. */
        if (useSingleRowDirectPacket(payload))
        {
            MoEOverlayActivationTimelinePublishDeviceBinding publication;
            if (!bindKernelTimelinePublication(
                    params_.lane,
                    params_.lane.dispatch_signal_offsets[bank],
                    timeline,
                    &publication,
                    "dispatch"))
            {
                return false;
            }
            return moe_kernel_->packSingleRowMoEOverlayActivationDispatch(
                packetLaunchContext(stream),
                MoEOverlayActivationSingleRowDispatchPackLaunch{
                    .packet = launch,
                    .publication = publication,
                });
        }

        if (!moe_kernel_->packMoEOverlayActivationDispatch(
                packetLaunchContext(stream), launch))
        {
            return false;
        }
        if (payload.requiresBulkPublication() &&
            !exportTensorBulkPayload(
                params_.lane,
                params_.hidden,
                0u,
                payload.hidden_payload_offset,
                payload_bytes,
                stream,
                "shared dispatch"))
        {
            return false;
        }

        /* Metadata, the compact row tables, and optional D2H payload DMA all
         * precede this one release edge on the exact capture stream. */
        return publishMappedTimeline(
            params_.lane,
            params_.lane.dispatch_signal_offsets[bank],
            timeline,
            stream,
            "dispatch");
    }

    ComputeStageType MoEOverlayActivationDispatchPackStage::type() const
    {
        return ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK;
    }

    std::string MoEOverlayActivationDispatchPackStage::name() const
    {
        return "moe_overlay_activation_dispatch_pack";
    }

    bool MoEOverlayActivationDispatchPackStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return isGPUBackend(backend);
    }

    bool MoEOverlayActivationDispatchPackStage::isGraphCapturable() const
    {
        return hasFixedContract();
    }

    bool MoEOverlayActivationDispatchPackStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationDispatchPackStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        /* BufferArena publishes the final tensor addresses after cold
         * preflight. The lane, placement banks, and launch geometry are
         * already immutable and are the only facts needed at this boundary. */
        return hasStaticContract();
    }

    bool MoEOverlayActivationDispatchPackStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return hasStaticContract() &&
               supportsPaddedPrefillRealLengthContract();
    }

    bool MoEOverlayActivationDispatchPackStage::
        supportsPaddedPrefillRealLengthContract() const
    {
        return params_.active_row_count_device != nullptr;
    }

    bool MoEOverlayActivationDispatchPackStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!validateCaptureContext(
                ctx, params_.device_id, stream,
                "MoEOverlayActivationDispatchPackStage"))
        {
            return false;
        }
        setGPUStream(stream);
        grant_initialization_joined_ =
            params_.stage_ordinal != 0u ||
            joinGrantInitialization(
                params_.lane,
                stream,
                "continuation capture preparation");
        return grant_initialization_joined_ && hasFixedContract();
    }

    GraphLaunchPreparationPolicy
    MoEOverlayActivationDispatchPackStage::graphLaunchPreparationPolicy() const
    {
        return supportsGraphCaptureAfterLaunchPreparation()
                   ? GraphLaunchPreparationPolicy::CaptureOnly
                   : GraphLaunchPreparationPolicy::None;
    }

    StageBufferContract
    MoEOverlayActivationDispatchPackStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.hidden_buffer_id)
            contract.addInput(*params_.hidden_buffer_id, "FP32");
        if (params_.routing_indices_buffer_id)
            contract.addInput(*params_.routing_indices_buffer_id, "FP32");
        if (params_.routing_weights_buffer_id)
            contract.addInput(*params_.routing_weights_buffer_id, "FP32");
        return contract;
    }

    StageBufferRequirements
    MoEOverlayActivationDispatchPackStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        addInputRequirement(requirements, "hidden", params_.hidden);
        addInputRequirement(
            requirements, "routing_indices", params_.routing_indices);
        addInputRequirement(
            requirements, "routing_weights", params_.routing_weights);
        return requirements;
    }

    std::string MoEOverlayActivationDispatchPackStage::
        graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer=" << params_.model_layer_index
            << " ordinal=" << params_.stage_ordinal
            << " rows=" << params_.physical_rows
            << " lane=" << (params_.lane.valid() ? "valid" : "invalid")
            << " hidden=" << (params_.hidden ? "present" : "missing")
            << " routing_indices="
            << (params_.routing_indices ? "present" : "missing")
            << " routing_weights="
            << (params_.routing_weights ? "present" : "missing")
            << " placement_banks="
            << (params_.placement.banks[0].valid() &&
                        params_.placement.banks[1].valid()
                    ? "present"
                    : "missing")
            << " placement_ticket="
            << (params_.placement.ticket ? "present" : "missing")
            << " placement_status="
            << (params_.placement.status ? "present" : "missing")
            << " experts=" << params_.placement.expert_count
            << " device_storage="
            << (hasFixedContract() ? "ready" : "unbound");
        return out.str();
    }

    StageDumpInfo MoEOverlayActivationDispatchPackStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.hidden)
        {
            info.addInput(
                "hidden",
                params_.hidden,
                static_cast<std::size_t>(params_.physical_rows),
                static_cast<std::size_t>(params_.lane.dispatch.d_model));
        }
        if (params_.routing_weights)
        {
            info.addInput(
                "routing_weights",
                params_.routing_weights,
                static_cast<std::size_t>(params_.physical_rows),
                static_cast<std::size_t>(params_.lane.dispatch.top_k));
        }
        if (params_.routing_indices)
        {
            info.addInput(
                "routing_indices",
                params_.routing_indices,
                static_cast<std::size_t>(params_.physical_rows),
                static_cast<std::size_t>(params_.lane.dispatch.top_k));
        }
        info.addScalarInt("target_participant", params_.lane.target_participant_id)
            .addScalarInt("layer", params_.model_layer_index)
            .addScalarInt("stage_ordinal", static_cast<int>(params_.stage_ordinal));
        return info;
    }

    MoEOverlayActivationDispatchConsumeStage::
        MoEOverlayActivationDispatchConsumeStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(KernelFactory::createMoEKernel(params_.device_id))
    {
    }

    MoEOverlayActivationDispatchConsumeStage::
        ~MoEOverlayActivationDispatchConsumeStage() = default;

    bool MoEOverlayActivationDispatchConsumeStage::hasStaticContract() const noexcept
    {
        std::size_t hidden_elements = 0u;
        std::size_t route_elements = 0u;
        return validLaneStage(
                   params_.lane,
                   params_.device_id,
                   params_.physical_rows,
                   params_.stage_ordinal,
                   params_.model_layer_index) &&
               params_.placement.valid() &&
               params_.active_row_count_device &&
               checkedElements(
                   params_.physical_rows,
                   params_.lane.dispatch.d_model,
                   &hidden_elements) &&
               checkedElements(
                   params_.physical_rows,
                   params_.lane.dispatch.top_k,
                   &route_elements) &&
               isFP32Capacity(params_.hidden, hidden_elements) &&
               isFP32Capacity(params_.routing_indices, route_elements) &&
               isFP32Capacity(params_.routing_weights, route_elements) &&
               params_.lane.dispatch.row_capacity >=
                   static_cast<std::size_t>(params_.physical_rows) &&
               moe_kernel_;
    }

    bool MoEOverlayActivationDispatchConsumeStage::hasFixedContract() const noexcept
    {
        return hasStaticContract() && params_.hidden->gpu_data_ptr() &&
               params_.routing_indices->gpu_data_ptr() &&
               params_.routing_weights->gpu_data_ptr();
    }

    bool MoEOverlayActivationDispatchConsumeStage::execute(IDeviceContext *ctx)
    {
        if (!validateExecutionContext(ctx, params_.device_id, name().c_str()) ||
            !hasFixedContract())
        {
            LOG_ERROR("[MoEOverlayActivationDispatchConsumeStage] Invalid fixed launch contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }
        void *const stream = requireGPUStream();
        std::size_t payload_bytes = 0u;
        const auto payload =
            params_.lane.dispatchPayload(params_.physical_rows);
        if (!payload.valid())
            return false;
        if (!fixedPayloadBytes(
                params_.lane, params_.physical_rows, &payload_bytes))
        {
            return false;
        }

        if (!waitForAdmissionIfFirstStage(
                params_.lane,
                params_.stage_ordinal,
                stream,
                grant_initialization_joined_,
                "follower admission"))
        {
            return false;
        }
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(params_.stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(params_.stage_ordinal));
        const MoEOverlayActivationDispatchConsumeLaunch launch{
            .packet = payload.packet,
            .placement = params_.placement,
            .hidden_payload_layout = payload.selection.layout,
            .control = params_.lane.control_device,
            .grant = params_.lane.grant_device,
            .hidden_rows_fp32 =
                static_cast<float *>(params_.hidden->gpu_data_ptr()),
            .routing_indices_fp32 = static_cast<float *>(
                params_.routing_indices->gpu_data_ptr()),
            .routing_weights_fp32 = static_cast<float *>(
                params_.routing_weights->gpu_data_ptr()),
            .active_row_count_device = params_.active_row_count_device,
            .physical_rows = params_.physical_rows,
            .stage_ordinal = params_.stage_ordinal,
            .model_layer_index = params_.model_layer_index,
        };

        bool launched = false;
        if (useSingleRowDirectPacket(payload))
        {
            MoEOverlayActivationTimelineWaitDeviceBinding acquire;
            if (!bindKernelTimelineWait(
                    params_.lane,
                    params_.lane.dispatch_signal_offsets[bank],
                    timeline,
                    &acquire,
                    "dispatch"))
            {
                return false;
            }
            launched =
                moe_kernel_->consumeSingleRowMoEOverlayActivationDispatch(
                    packetLaunchContext(stream),
                    MoEOverlayActivationSingleRowDispatchConsumeLaunch{
                        .packet = launch,
                        .acquire = acquire,
                    });
        }
        else
        {
            if (!waitForMappedTimeline(
                    params_.lane,
                    params_.lane.dispatch_signal_offsets[bank],
                    timeline,
                    stream,
                    "dispatch"))
            {
                return false;
            }

            /* The release/acquire edge covers the one shared physical
             * activation publication. The materializer gathers selected rows
             * directly from the registered mapping without another copy. */
            launched = moe_kernel_->consumeMoEOverlayActivationDispatch(
                packetLaunchContext(stream), launch);
        }
        if (!launched)
        {
            return false;
        }

        /* The materialization kernel is the final producer for the hidden and
         * route tensors. Publish all three exact-stream edges for the following
         * real expert stage. */
        try
        {
            const auto execution = gpuExecution();
            execution.publish(params_.hidden);
            execution.publish(params_.routing_indices);
            execution.publish(params_.routing_weights);
            return true;
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEOverlayActivationDispatchConsumeStage] Failed to publish mapped-packet follower tensors: "
                << error.what());
            return false;
        }
    }

    ComputeStageType MoEOverlayActivationDispatchConsumeStage::type() const
    {
        return ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_CONSUME;
    }

    std::string MoEOverlayActivationDispatchConsumeStage::name() const
    {
        return "moe_overlay_activation_dispatch_consume";
    }

    bool MoEOverlayActivationDispatchConsumeStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return isGPUBackend(backend);
    }

    bool MoEOverlayActivationDispatchConsumeStage::isGraphCapturable() const
    {
        return hasFixedContract();
    }

    bool MoEOverlayActivationDispatchConsumeStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationDispatchConsumeStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationDispatchConsumeStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationDispatchConsumeStage::
        supportsPaddedPrefillRealLengthContract() const
    {
        /* The validated descriptor writes the compact live-row count before
         * the fixed-grid payload expansion clears all remaining rows. */
        return params_.active_row_count_device != nullptr;
    }

    bool MoEOverlayActivationDispatchConsumeStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!validateCaptureContext(
                ctx, params_.device_id, stream,
                "MoEOverlayActivationDispatchConsumeStage"))
        {
            return false;
        }
        setGPUStream(stream);
        grant_initialization_joined_ =
            params_.stage_ordinal != 0u ||
            joinGrantInitialization(
                params_.lane,
                stream,
                "follower capture preparation");
        return grant_initialization_joined_ && hasFixedContract();
    }

    GraphLaunchPreparationPolicy
    MoEOverlayActivationDispatchConsumeStage::graphLaunchPreparationPolicy() const
    {
        return supportsGraphCaptureAfterLaunchPreparation()
                   ? GraphLaunchPreparationPolicy::CaptureOnly
                   : GraphLaunchPreparationPolicy::None;
    }

    StageBufferContract
    MoEOverlayActivationDispatchConsumeStage::bufferContract() const
    {
        return StageBufferContract::build()
            .addOutput(params_.hidden_buffer_id, "FP32")
            .addOutput(params_.routing_indices_buffer_id, "FP32")
            .addOutput(params_.routing_weights_buffer_id, "FP32");
    }

    StageBufferRequirements
    MoEOverlayActivationDispatchConsumeStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        addOutputRequirement(requirements, "hidden", params_.hidden);
        addOutputRequirement(
            requirements, "routing_indices", params_.routing_indices);
        addOutputRequirement(
            requirements, "routing_weights", params_.routing_weights);
        return requirements;
    }

    std::string MoEOverlayActivationDispatchConsumeStage::
        graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer=" << params_.model_layer_index
            << " ordinal=" << params_.stage_ordinal
            << " rows=" << params_.physical_rows
            << " lane=" << (params_.lane.valid() ? "valid" : "invalid")
            << " placement_banks="
            << (params_.placement.banks[0].valid() &&
                        params_.placement.banks[1].valid()
                    ? "present"
                    : "missing")
            << " placement_ticket="
            << (params_.placement.ticket ? "present" : "missing")
            << " placement_status="
            << (params_.placement.status ? "present" : "missing")
            << " experts=" << params_.placement.expert_count
            << " active_rows="
            << (params_.active_row_count_device ? "present" : "missing")
            << " destinations="
            << (params_.hidden && params_.routing_indices &&
                        params_.routing_weights
                    ? "present"
                    : "missing")
            << " device_storage="
            << (hasFixedContract() ? "ready" : "unbound");
        return out.str();
    }

    StageDumpInfo MoEOverlayActivationDispatchConsumeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.hidden)
        {
            info.addOutput(
                "hidden",
                params_.hidden,
                static_cast<std::size_t>(params_.physical_rows),
                static_cast<std::size_t>(params_.lane.dispatch.d_model));
        }
        if (params_.routing_indices)
        {
            info.addOutput(
                "routing_indices",
                params_.routing_indices,
                static_cast<std::size_t>(params_.physical_rows),
                static_cast<std::size_t>(params_.lane.dispatch.top_k));
        }
        if (params_.routing_weights)
        {
            info.addOutput(
                "routing_weights",
                params_.routing_weights,
                static_cast<std::size_t>(params_.physical_rows),
                static_cast<std::size_t>(params_.lane.dispatch.top_k));
        }
        info.addScalarInt("layer", params_.model_layer_index)
            .addScalarInt("stage_ordinal", static_cast<int>(params_.stage_ordinal));
        return info;
    }

    MoEOverlayActivationReturnPackStage::MoEOverlayActivationReturnPackStage(
        Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(KernelFactory::createMoEKernel(params_.device_id))
    {
    }

    MoEOverlayActivationReturnPackStage::~MoEOverlayActivationReturnPackStage() =
        default;

    bool MoEOverlayActivationReturnPackStage::hasStaticContract() const noexcept
    {
        std::size_t route_elements = 0u;
        return validLaneStage(
                   params_.lane,
                   params_.device_id,
                   params_.physical_rows,
                   params_.stage_ordinal,
                   params_.model_layer_index) &&
               params_.lane.dispatch.d_model == params_.lane.returned.d_model &&
               checkedCanonicalRouteElements(
                   params_.physical_rows,
                   params_.lane.dispatch.top_k,
                   params_.lane.returned.d_model,
                   &route_elements) &&
               isFP32Capacity(
                   params_.local_canonical_route_contributions,
                   route_elements) &&
               moe_kernel_;
    }

    bool MoEOverlayActivationReturnPackStage::hasFixedContract() const noexcept
    {
        return hasStaticContract() &&
               params_.local_canonical_route_contributions->gpu_data_ptr();
    }

    bool MoEOverlayActivationReturnPackStage::execute(IDeviceContext *ctx)
    {
        if (!validateExecutionContext(ctx, params_.device_id, name().c_str()) ||
            !hasFixedContract())
        {
            LOG_ERROR("[MoEOverlayActivationReturnPackStage] Invalid fixed launch contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }
        void *const stream = requireGPUStream();
        std::size_t payload_bytes = 0u;
        const auto payload =
            params_.lane.dispatchPayload(params_.physical_rows);
        if (!payload.valid())
            return false;
        if (!fixedPayloadBytes(
                params_.lane, params_.physical_rows, &payload_bytes))
            return false;

        const MoEOverlayActivationReturnPackLaunch launch{
            .local_canonical_route_contributions_fp32 =
                static_cast<const float *>(
                    params_.local_canonical_route_contributions->gpu_data_ptr()),
            .dispatch = payload.packet,
            .returned = params_.lane.returned,
            .control = params_.lane.control_device,
            .grant = params_.lane.grant_device,
            .physical_rows = params_.physical_rows,
            .stage_ordinal = params_.stage_ordinal,
            .model_layer_index = params_.model_layer_index,
        };
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(params_.stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(params_.stage_ordinal));
        if (useSingleRowDirectPacket(payload))
        {
            MoEOverlayActivationTimelinePublishDeviceBinding publication;
            if (!bindKernelTimelinePublication(
                    params_.lane,
                    params_.lane.return_signal_offsets[bank],
                    timeline,
                    &publication,
                    "return"))
            {
                return false;
            }
            return moe_kernel_->packSingleRowMoEOverlayActivationReturn(
                packetLaunchContext(stream),
                MoEOverlayActivationSingleRowReturnPackLaunch{
                    .packet = launch,
                    .publication = publication,
                });
        }

        if (!moe_kernel_->packMoEOverlayActivationReturn(
                packetLaunchContext(stream), launch))
        {
            return false;
        }

        return publishMappedTimeline(
            params_.lane,
            params_.lane.return_signal_offsets[bank],
            timeline,
            stream,
            "return");
    }

    ComputeStageType MoEOverlayActivationReturnPackStage::type() const
    {
        return ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_PACK;
    }

    std::string MoEOverlayActivationReturnPackStage::name() const
    {
        return "moe_overlay_activation_return_pack";
    }

    bool MoEOverlayActivationReturnPackStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return isGPUBackend(backend);
    }

    bool MoEOverlayActivationReturnPackStage::isGraphCapturable() const
    {
        return hasFixedContract();
    }

    bool MoEOverlayActivationReturnPackStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnPackStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnPackStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnPackStage::
        supportsPaddedPrefillRealLengthContract() const
    {
        /* The preceding dispatch descriptor is the sole row-count authority;
         * return metadata and payload copy exactly that compact prefix. */
        return true;
    }

    bool MoEOverlayActivationReturnPackStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!validateCaptureContext(
                ctx, params_.device_id, stream,
                "MoEOverlayActivationReturnPackStage"))
        {
            return false;
        }
        setGPUStream(stream);
        return hasFixedContract();
    }

    GraphLaunchPreparationPolicy
    MoEOverlayActivationReturnPackStage::graphLaunchPreparationPolicy() const
    {
        return supportsGraphCaptureAfterLaunchPreparation()
                   ? GraphLaunchPreparationPolicy::CaptureOnly
                   : GraphLaunchPreparationPolicy::None;
    }

    StageBufferContract
    MoEOverlayActivationReturnPackStage::bufferContract() const
    {
        return StageBufferContract::build().addInput(
            params_.local_canonical_route_contributions_buffer_id, "FP32");
    }

    StageBufferRequirements
    MoEOverlayActivationReturnPackStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        addInputRequirement(
            requirements,
            "local_canonical_route_contributions",
            params_.local_canonical_route_contributions);
        return requirements;
    }

    std::string MoEOverlayActivationReturnPackStage::
        graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer=" << params_.model_layer_index
            << " ordinal=" << params_.stage_ordinal
            << " rows=" << params_.physical_rows
            << " lane=" << (params_.lane.valid() ? "valid" : "invalid")
            << " canonical_routes="
            << (params_.local_canonical_route_contributions
                    ? "present"
                    : "missing")
            << " device_storage="
            << (hasFixedContract() ? "ready" : "unbound");
        return out.str();
    }

    StageDumpInfo MoEOverlayActivationReturnPackStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.local_canonical_route_contributions)
        {
            info.addInput(
                "local_canonical_route_contributions",
                params_.local_canonical_route_contributions,
                static_cast<std::size_t>(params_.physical_rows) *
                    static_cast<std::size_t>(params_.lane.dispatch.top_k),
                static_cast<std::size_t>(params_.lane.returned.d_model));
        }
        info.addScalarInt("layer", params_.model_layer_index)
            .addScalarInt("stage_ordinal", static_cast<int>(params_.stage_ordinal));
        return info;
    }

    MoEOverlayActivationReturnConsumeStage::
        MoEOverlayActivationReturnConsumeStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(KernelFactory::createMoEKernel(params_.device_id))
    {
    }

    MoEOverlayActivationReturnConsumeStage::
        ~MoEOverlayActivationReturnConsumeStage() = default;

    bool MoEOverlayActivationReturnConsumeStage::hasStaticContract() const noexcept
    {
        std::size_t route_elements = 0u;
        return validLaneStage(
                   params_.lane,
                   params_.device_id,
                   params_.physical_rows,
                   params_.stage_ordinal,
                   params_.model_layer_index) &&
               params_.lane.dispatch.d_model == params_.lane.returned.d_model &&
               checkedCanonicalRouteElements(
                   params_.physical_rows,
                   params_.lane.dispatch.top_k,
                   params_.lane.returned.d_model,
                   &route_elements) &&
               isFP32Capacity(
                   params_.canonical_route_contributions,
                   route_elements) &&
               moe_kernel_;
    }

    bool MoEOverlayActivationReturnConsumeStage::hasFixedContract() const noexcept
    {
        return hasStaticContract() &&
               params_.canonical_route_contributions->gpu_data_ptr();
    }

    bool MoEOverlayActivationReturnConsumeStage::execute(IDeviceContext *ctx)
    {
        if (!validateExecutionContext(ctx, params_.device_id, name().c_str()) ||
            !hasFixedContract())
        {
            LOG_ERROR("[MoEOverlayActivationReturnConsumeStage] Invalid fixed launch contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }
        void *const stream = requireGPUStream();
        std::size_t payload_bytes = 0u;
        const auto payload =
            params_.lane.dispatchPayload(params_.physical_rows);
        if (!payload.valid())
            return false;
        if (!fixedPayloadBytes(
                params_.lane, params_.physical_rows, &payload_bytes))
        {
            return false;
        }

        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(params_.stage_ordinal);
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(params_.stage_ordinal));
        const MoEOverlayActivationReturnConsumeLaunch launch{
            .dispatch = payload.packet,
            .returned = params_.lane.returned,
            .control = params_.lane.control_device,
            .grant = params_.lane.grant_device,
            .canonical_route_contributions_fp32 = static_cast<float *>(
                params_.canonical_route_contributions->gpu_data_ptr()),
            .physical_rows = params_.physical_rows,
            .stage_ordinal = params_.stage_ordinal,
            .model_layer_index = params_.model_layer_index,
        };

        bool launched = false;
        if (useSingleRowDirectPacket(payload))
        {
            MoEOverlayActivationTimelineWaitDeviceBinding acquire;
            if (!bindKernelTimelineWait(
                    params_.lane,
                    params_.lane.return_signal_offsets[bank],
                    timeline,
                    &acquire,
                    "return"))
            {
                return false;
            }
            launched =
                moe_kernel_->consumeSingleRowMoEOverlayActivationReturn(
                    packetLaunchContext(stream),
                    MoEOverlayActivationSingleRowReturnConsumeLaunch{
                        .packet = launch,
                        .acquire = acquire,
                    });
        }
        else
        {
            if (!waitForMappedTimeline(
                    params_.lane,
                    params_.lane.return_signal_offsets[bank],
                    timeline,
                    stream,
                    "return"))
            {
                return false;
            }

            launched = moe_kernel_->consumeMoEOverlayActivationReturn(
                packetLaunchContext(stream), launch);
        }
        if (!launched)
        {
            return false;
        }
        try
        {
            gpuExecution().publish(params_.canonical_route_contributions);
            return true;
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEOverlayActivationReturnConsumeStage] Failed to publish canonical route materialization: "
                << error.what());
            return false;
        }
    }

    ComputeStageType MoEOverlayActivationReturnConsumeStage::type() const
    {
        return ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME;
    }

    std::string MoEOverlayActivationReturnConsumeStage::name() const
    {
        return "moe_overlay_activation_return_consume";
    }

    bool MoEOverlayActivationReturnConsumeStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return isGPUBackend(backend);
    }

    bool MoEOverlayActivationReturnConsumeStage::isGraphCapturable() const
    {
        return hasFixedContract();
    }

    bool MoEOverlayActivationReturnConsumeStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnConsumeStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnConsumeStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnConsumeStage::
        supportsPaddedPrefillRealLengthContract() const
    {
        /* Validated return row identities address only the real root rows; no
         * padded row is eligible for the deterministic accumulation kernel. */
        return true;
    }

    bool MoEOverlayActivationReturnConsumeStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!validateCaptureContext(
                ctx, params_.device_id, stream,
                "MoEOverlayActivationReturnConsumeStage"))
        {
            return false;
        }
        setGPUStream(stream);
        return hasFixedContract();
    }

    GraphLaunchPreparationPolicy
    MoEOverlayActivationReturnConsumeStage::graphLaunchPreparationPolicy() const
    {
        return supportsGraphCaptureAfterLaunchPreparation()
                   ? GraphLaunchPreparationPolicy::CaptureOnly
                   : GraphLaunchPreparationPolicy::None;
    }

    StageBufferContract
    MoEOverlayActivationReturnConsumeStage::bufferContract() const
    {
        return StageBufferContract::build().addInOut(
            params_.canonical_route_contributions_buffer_id, "FP32");
    }

    StageBufferRequirements
    MoEOverlayActivationReturnConsumeStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        addInoutRequirement(
            requirements,
            "canonical_route_contributions",
            params_.canonical_route_contributions);
        return requirements;
    }

    std::string MoEOverlayActivationReturnConsumeStage::
        graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer=" << params_.model_layer_index
            << " ordinal=" << params_.stage_ordinal
            << " rows=" << params_.physical_rows
            << " lane=" << (params_.lane.valid() ? "valid" : "invalid")
            << " canonical_routes="
            << (params_.canonical_route_contributions
                    ? "present"
                    : "missing")
            << " device_storage="
            << (hasFixedContract() ? "ready" : "unbound");
        return out.str();
    }

    StageDumpInfo MoEOverlayActivationReturnConsumeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.canonical_route_contributions)
        {
            info.addInput(
                    "canonical_route_contributions_before",
                    params_.canonical_route_contributions,
                    static_cast<std::size_t>(params_.physical_rows) *
                        static_cast<std::size_t>(params_.lane.dispatch.top_k),
                    static_cast<std::size_t>(params_.lane.returned.d_model))
                .addOutput(
                    "canonical_route_contributions_after",
                    params_.canonical_route_contributions,
                    static_cast<std::size_t>(params_.physical_rows) *
                        static_cast<std::size_t>(params_.lane.dispatch.top_k),
                    static_cast<std::size_t>(params_.lane.returned.d_model));
        }
        info.addScalarInt("layer", params_.model_layer_index)
            .addScalarInt("stage_ordinal", static_cast<int>(params_.stage_ordinal));
        return info;
    }

    MoEOverlayActivationDispatchPackBatchStage::
        MoEOverlayActivationDispatchPackBatchStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(KernelFactory::createMoEKernel(params_.device_id))
    {
    }

    MoEOverlayActivationDispatchPackBatchStage::
        ~MoEOverlayActivationDispatchPackBatchStage() = default;

    bool MoEOverlayActivationDispatchPackBatchStage::
        hasStaticContract() const noexcept
    {
        const auto *const transaction = params_.transaction.get();
        if (!transaction || transaction->device() != params_.device_id)
            return false;

        const auto &lanes = transaction->lanes();
        std::int32_t d_model = 0;
        std::int32_t top_k = 0;
        std::size_t hidden_elements = 0u;
        std::size_t route_elements = 0u;
        const bool lane_contract =
            transaction->usesAsynchronousLanes()
                ? validLaneTransaction(
                      lanes,
                      params_.device_id,
                      transaction->physicalRows(),
                      transaction->stageOrdinal(),
                      transaction->modelLayerIndex(),
                      &d_model,
                      &top_k)
                : validOneRowLaneBatch(
                      lanes,
                      params_.device_id,
                      transaction->stageOrdinal(),
                      transaction->modelLayerIndex(),
                      &d_model,
                      &top_k);
        return lane_contract &&
               params_.placement.valid() &&
               checkedElements(
                   transaction->physicalRows(),
                   d_model,
                   &hidden_elements) &&
               checkedElements(
                   transaction->physicalRows(),
                   top_k,
                   &route_elements) &&
               isFP32Capacity(params_.hidden, hidden_elements) &&
               isFP32Capacity(params_.routing_indices, route_elements) &&
               isFP32Capacity(params_.routing_weights, route_elements) &&
               moe_kernel_;
    }

    bool MoEOverlayActivationDispatchPackBatchStage::
        hasFixedContract() const noexcept
    {
        if (!hasStaticContract() || !params_.hidden->gpu_data_ptr() ||
            !params_.routing_indices->gpu_data_ptr() ||
            !params_.routing_weights->gpu_data_ptr())
        {
            return false;
        }
        if (params_.transaction->usesAsynchronousLanes())
            return params_.transaction->persistentResourcesReady();
        return
               params_.routing_indices->gpu_data_ptr() &&
               params_.routing_weights->gpu_data_ptr() &&
               descriptor_storage_ && descriptor_storage_->deviceData();
    }

    bool MoEOverlayActivationDispatchPackBatchStage::prepareDescriptors(
        void *stream)
    {
        if (!params_.transaction ||
            params_.transaction->usesAsynchronousLanes())
        {
            return false;
        }
        const auto &lanes = params_.transaction->lanes();
        if (descriptor_storage_)
        {
            return descriptor_storage_->requireInput(stream) &&
                   descriptor_storage_->deviceData() != nullptr;
        }

        std::vector<MoEOverlayActivationSingleRowDispatchPackLaunch> launches;
        launches.reserve(lanes.size());
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(
                params_.transaction->stageOrdinal());
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(
                    params_.transaction->stageOrdinal()));
        for (const auto &lane : lanes)
        {
            const auto payload = lane.dispatchPayload(
                /*physical_rows=*/1);
            if (!payload.valid() ||
                !payload.selection.usesCompactRows())
            {
                return false;
            }
            MoEOverlayActivationTimelinePublishDeviceBinding publication;
            if (!bindKernelTimelinePublication(
                    lane,
                    lane.dispatch_signal_offsets[bank],
                    timeline,
                    &publication,
                    "dispatch batch"))
            {
                return false;
            }
            const MoEOverlayActivationDispatchPackLaunch packet{
                .hidden_rows_fp32 = static_cast<const float *>(
                    params_.hidden->gpu_data_ptr()),
                .route_expert_ids_fp32 = static_cast<const float *>(
                    params_.routing_indices->gpu_data_ptr()),
                .route_weights = static_cast<const float *>(
                    params_.routing_weights->gpu_data_ptr()),
                .placement = params_.placement,
                .active_row_count_device =
                    params_.active_row_count_device,
                .packet = payload.packet,
                .hidden_payload_layout = payload.selection.layout,
                .control = lane.control_device,
                .grant = lane.grant_device,
                .target_participant_id = lane.target_participant_id,
                .physical_rows = 1,
                .stage_ordinal = params_.transaction->stageOrdinal(),
                .model_layer_index =
                    params_.transaction->modelLayerIndex(),
            };
            MoEOverlayActivationSingleRowDispatchPackLaunch launch{
                .packet = packet,
                .publication = publication,
            };
            if (!launch.valid())
                return false;
            launches.push_back(launch);
        }
        return uploadPersistentLaunchArray(
            launches,
            params_.device_id,
            stream,
            &descriptor_storage_,
            "one-row dispatch batch");
    }

    bool MoEOverlayActivationDispatchPackBatchStage::execute(
        IDeviceContext *ctx)
    {
        if (!validateExecutionContext(ctx, params_.device_id, name().c_str()) ||
            !hasFixedContract())
        {
            LOG_ERROR("[MoEOverlayActivationDispatchPackBatchStage] Invalid fixed launch contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }
        void *const stream = requireGPUStream();
        auto &transaction = *params_.transaction;
        const auto &lanes = transaction.lanes();

        if (transaction.usesAsynchronousLanes())
        {
            try
            {
                auto &gpu_ctx =
                    GPUDeviceContextPool::instance().getContext(
                        params_.device_id);
                const auto input_fork =
                    TransferEngine::instance().recordDeviceInputFork(
                        params_.hidden,
                        params_.device_id,
                        stream,
                        transaction.producerReadyEvent());

                const std::uint32_t bank =
                    moeOverlayActivationBufferIndex(
                        transaction.stageOrdinal());
                const std::uint64_t timeline =
                    moeOverlayActivationLeasedTimelineValue(
                        moeOverlayActivationBufferVisit(
                            transaction.stageOrdinal()));
                std::size_t logical_lane_payload_bytes = 0u;
                std::size_t physical_dispatch_payload_bytes = 0u;
                std::size_t physical_return_payload_bytes = 0u;
                for (std::size_t lane_index = 0u;
                     lane_index < lanes.size();
                     ++lane_index)
                {
                    const auto &lane = lanes[lane_index];
                    void *const lane_stream =
                        transaction.laneStream(lane_index);
                    const auto acquired_input =
                        TransferEngine::instance().acquireDeviceInputFork(
                            input_fork, lane_stream);
                    if (!waitForAdmissionIfFirstStage(
                            lane,
                            transaction.stageOrdinal(),
                            lane_stream,
                            grant_initialization_joined_,
                            "continuation asynchronous lane admission"))
                    {
                        return false;
                    }

                    std::size_t payload_bytes = 0u;
                    if (!fixedPayloadBytes(
                            lane,
                            transaction.physicalRows(),
                            &payload_bytes))
                    {
                        return false;
                    }
                    logical_lane_payload_bytes += payload_bytes;
                    const auto payload = lane.dispatchPayload(
                        transaction.physicalRows());
                    if (!payload.valid())
                        return false;
                    const MoEOverlayActivationDispatchPackLaunch launch{
                        .hidden_rows_fp32 = static_cast<const float *>(
                            params_.hidden->gpu_data_ptr()),
                        .route_expert_ids_fp32 = static_cast<const float *>(
                            params_.routing_indices->gpu_data_ptr()),
                        .route_weights = static_cast<const float *>(
                            params_.routing_weights->gpu_data_ptr()),
                        .placement = params_.placement,
                        .active_row_count_device =
                            params_.active_row_count_device,
                        .packet = payload.packet,
                        .hidden_payload_layout =
                            payload.selection.layout,
                        .control = lane.control_device,
                        .grant = lane.grant_device,
                        .target_participant_id =
                            lane.target_participant_id,
                        .physical_rows = transaction.physicalRows(),
                        .stage_ordinal = transaction.stageOrdinal(),
                        .model_layer_index =
                            transaction.modelLayerIndex(),
                    };
                    if (!moe_kernel_->packMoEOverlayActivationDispatch(
                            packetLaunchContext(lane_stream), launch))
                    {
                        return false;
                    }
                    if (payload.requiresBulkPublication())
                    {
                        if (transaction.lanePublishesSharedDispatchPayload(
                                lane_index))
                        {
                            if (!exportBulkPayload(
                                    lane,
                                    acquired_input,
                                    0u,
                                    payload.hidden_payload_offset,
                                    payload_bytes,
                                    "shared dispatch lane batch") ||
                                !gpu_ctx.recordEventChecked(
                                    transaction.
                                        sharedDispatchPayloadReadyEvent(
                                            lane_index),
                                    lane_stream))
                            {
                                LOG_ERROR("[MoEOverlayActivationDispatchPackBatchStage] Failed to publish one shared physical activation payload");
                                return false;
                            }
                            physical_dispatch_payload_bytes += payload_bytes;
                            PerfStatsCollector::addCounter(
                                "moe_overlay_activation_epoch",
                                "shared_dispatch_payload_publications",
                                1.0,
                                "prefill",
                                params_.device_id.toString(),
                                {{"lanes", std::to_string(
                                      transaction.
                                          sharedDispatchPayloadLaneCount(
                                              lane_index))},
                                 {"rows", std::to_string(
                                      transaction.physicalRows())},
                                 {"layer", std::to_string(
                                      transaction.modelLayerIndex())}});
                        }
                        else if (!gpu_ctx.waitEventChecked(
                                     transaction.
                                         sharedDispatchPayloadReadyEvent(
                                             lane_index),
                                     lane_stream))
                        {
                            LOG_ERROR("[MoEOverlayActivationDispatchPackBatchStage] Failed to queue shared activation publication dependency");
                            return false;
                        }
                    }
                    if (!publishMappedTimeline(
                            lane,
                            lane.dispatch_signal_offsets[bank],
                            timeline,
                            lane_stream,
                            "dispatch lane batch"))
                    {
                        return false;
                    }

                    /* Queue the complete remote half immediately after its
                     * dispatch. The device timeline wait suspends only this lane;
                     * the main stream is free to execute local experts. */
                    if (!waitForMappedTimeline(
                            lane,
                            lane.return_signal_offsets[bank],
                            timeline,
                            lane_stream,
                            "return lane batch"))
                    {
                        return false;
                    }
                    if (!gpu_ctx.recordEventChecked(
                            transaction.laneReturnReadyEvent(lane_index),
                            lane_stream))
                    {
                        LOG_ERROR("[MoEOverlayActivationDispatchPackBatchStage] Failed to record lane return readiness");
                        return false;
                    }
                }
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_async_lane_forks",
                    1.0,
                    "prefill",
                    params_.device_id.toString(),
                    {{"lanes", std::to_string(lanes.size())},
                     {"rows", std::to_string(
                          transaction.physicalRows())},
                     {"logical_lane_bytes", std::to_string(
                          logical_lane_payload_bytes)},
                     {"physical_dispatch_bytes", std::to_string(
                          physical_dispatch_payload_bytes)},
                     {"physical_return_bytes", std::to_string(
                          physical_return_payload_bytes)},
                     {"shared_dispatch_groups", std::to_string(
                          transaction.sharedDispatchPayloadGroupCount())},
                     {"layer", std::to_string(
                          transaction.modelLayerIndex())}});
                return true;
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationDispatchPackBatchStage] Asynchronous lane fork failed: "
                          << error.what());
                return false;
            }
        }

        if (!descriptor_storage_->requireInput(stream))
        {
            LOG_ERROR("[MoEOverlayActivationDispatchPackBatchStage] Descriptor input join failed");
            return false;
        }
        for (const auto &lane : lanes)
        {
            if (!waitForAdmissionIfFirstStage(
                lane,
                transaction.stageOrdinal(),
                stream,
                grant_initialization_joined_,
                "continuation batch admission"))
            {
                return false;
            }
        }
        const auto launch =
            MoEOverlayActivationSingleRowDispatchBatchLaunch{
                .lanes = static_cast<const
                    MoEOverlayActivationSingleRowDispatchPackLaunch *>(
                        descriptor_storage_->deviceData()),
                .lane_count = static_cast<std::uint32_t>(
                    lanes.size()),
            };
        const bool launched =
            moe_kernel_->packSingleRowMoEOverlayActivationDispatchBatch(
                packetLaunchContext(stream), launch);
        if (launched)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "moe_overlay_one_row_dispatch_batch_stages",
                1.0,
                "decode",
                params_.device_id.toString(),
                {{"lanes", std::to_string(lanes.size())},
                 {"layer", std::to_string(
                      transaction.modelLayerIndex())}});
        }
        return launched;
    }

    ComputeStageType MoEOverlayActivationDispatchPackBatchStage::type() const
    {
        return ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK;
    }

    std::string MoEOverlayActivationDispatchPackBatchStage::name() const
    {
        return "moe_overlay_activation_dispatch_pack_batch";
    }

    bool MoEOverlayActivationDispatchPackBatchStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return isGPUBackend(backend);
    }

    bool MoEOverlayActivationDispatchPackBatchStage::isGraphCapturable() const
    {
        return hasFixedContract();
    }

    bool MoEOverlayActivationDispatchPackBatchStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationDispatchPackBatchStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationDispatchPackBatchStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return params_.transaction &&
               params_.transaction->usesAsynchronousLanes() &&
               hasStaticContract();
    }

    bool MoEOverlayActivationDispatchPackBatchStage::
        supportsPaddedPrefillRealLengthContract() const
    {
        return params_.transaction &&
               params_.transaction->usesAsynchronousLanes() &&
               params_.active_row_count_device != nullptr;
    }

    bool MoEOverlayActivationDispatchPackBatchStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!validateCaptureContext(
                ctx,
                params_.device_id,
                stream,
                "MoEOverlayActivationDispatchPackBatchStage") ||
            !hasStaticContract())
        {
            return false;
        }
        setGPUStream(stream);
        if (params_.transaction->usesAsynchronousLanes())
        {
            if (!params_.transaction->materializePersistentResources())
                return false;
            grant_initialization_joined_ = true;
            if (params_.transaction->stageOrdinal() == 0u)
            {
                const auto &lanes = params_.transaction->lanes();
                for (std::size_t lane_index = 0u;
                     lane_index < lanes.size();
                     ++lane_index)
                {
                    grant_initialization_joined_ =
                        joinGrantInitialization(
                            lanes[lane_index],
                            params_.transaction->laneStream(lane_index),
                            "continuation asynchronous capture preparation") &&
                        grant_initialization_joined_;
                }
            }
            return grant_initialization_joined_ && hasFixedContract();
        }
        grant_initialization_joined_ = true;
        if (params_.transaction->stageOrdinal() == 0u)
        {
            for (const auto &lane : params_.transaction->lanes())
            {
                grant_initialization_joined_ =
                    joinGrantInitialization(
                        lane,
                        stream,
                        "continuation batch capture preparation") &&
                    grant_initialization_joined_;
            }
        }
        return grant_initialization_joined_ && prepareDescriptors(stream) &&
               hasFixedContract();
    }

    GraphLaunchPreparationPolicy
    MoEOverlayActivationDispatchPackBatchStage::
        graphLaunchPreparationPolicy() const
    {
        return supportsGraphCaptureAfterLaunchPreparation()
                   ? GraphLaunchPreparationPolicy::CaptureOnly
                   : GraphLaunchPreparationPolicy::None;
    }

    StageBufferContract
    MoEOverlayActivationDispatchPackBatchStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.hidden_buffer_id)
            contract.addInput(*params_.hidden_buffer_id, "FP32");
        if (params_.routing_indices_buffer_id)
            contract.addInput(*params_.routing_indices_buffer_id, "FP32");
        if (params_.routing_weights_buffer_id)
            contract.addInput(*params_.routing_weights_buffer_id, "FP32");
        return contract;
    }

    StageBufferRequirements
    MoEOverlayActivationDispatchPackBatchStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        addInputRequirement(requirements, "hidden", params_.hidden);
        addInputRequirement(
            requirements, "routing_indices", params_.routing_indices);
        addInputRequirement(
            requirements, "routing_weights", params_.routing_weights);
        return requirements;
    }

    std::string MoEOverlayActivationDispatchPackBatchStage::
        graphCaptureReadinessDebugString() const
    {
        const auto *const transaction = params_.transaction.get();
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer="
            << (transaction ? transaction->modelLayerIndex() : -1)
            << " ordinal="
            << (transaction ? transaction->stageOrdinal() : 0u)
            << " rows="
            << (transaction ? transaction->physicalRows() : 0)
            << " lanes="
            << (transaction ? transaction->lanes().size() : 0u)
            << " static=" << (hasStaticContract() ? "valid" : "invalid")
            << " placement_banks="
            << (params_.placement.banks[0].valid() &&
                        params_.placement.banks[1].valid()
                    ? "present"
                    : "missing")
            << " placement_ticket="
            << (params_.placement.ticket ? "present" : "missing")
            << " placement_status="
            << (params_.placement.status ? "present" : "missing")
            << " experts=" << params_.placement.expert_count
            << " mode="
            << (transaction && transaction->usesAsynchronousLanes()
                    ? "async_fork"
                    : "fused_decode")
            << " resources="
            << (transaction && transaction->usesAsynchronousLanes()
                    ? (transaction->persistentResourcesReady()
                           ? "ready"
                           : "unbound")
                    : (descriptor_storage_ &&
                               descriptor_storage_->deviceData()
                           ? "ready"
                           : "unbound"));
        return out.str();
    }

    StageDumpInfo
    MoEOverlayActivationDispatchPackBatchStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const auto *const transaction = params_.transaction.get();
        if (params_.hidden && transaction && !transaction->lanes().empty())
        {
            info.addInput(
                "hidden",
                params_.hidden,
                static_cast<std::size_t>(transaction->physicalRows()),
                static_cast<std::size_t>(transaction->dModel()));
        }
        info.addScalarInt(
                "lane_count",
                transaction
                    ? static_cast<int>(transaction->lanes().size())
                    : 0)
            .addScalarInt(
                "layer",
                transaction ? transaction->modelLayerIndex() : -1)
            .addScalarInt(
                "stage_ordinal",
                transaction
                    ? static_cast<int>(transaction->stageOrdinal())
                    : 0);
        return info;
    }

    MoEOverlayActivationReturnConsumeBatchStage::
        MoEOverlayActivationReturnConsumeBatchStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(KernelFactory::createMoEKernel(params_.device_id))
    {
    }

    MoEOverlayActivationReturnConsumeBatchStage::
        ~MoEOverlayActivationReturnConsumeBatchStage() = default;

    bool MoEOverlayActivationReturnConsumeBatchStage::
        hasStaticContract() const noexcept
    {
        const auto *const transaction = params_.transaction.get();
        if (!transaction || transaction->device() != params_.device_id)
            return false;
        const auto &lanes = transaction->lanes();
        std::int32_t d_model = 0;
        std::int32_t top_k = 0;
        std::size_t route_elements = 0u;
        const bool lane_contract =
            transaction->usesAsynchronousLanes()
                ? validLaneTransaction(
                      lanes,
                      params_.device_id,
                      transaction->physicalRows(),
                      transaction->stageOrdinal(),
                      transaction->modelLayerIndex(),
                      &d_model,
                      &top_k)
                : validOneRowLaneBatch(
                      lanes,
                      params_.device_id,
                      transaction->stageOrdinal(),
                      transaction->modelLayerIndex(),
                      &d_model,
                      &top_k);
        return lane_contract &&
               checkedCanonicalRouteElements(
                   transaction->physicalRows(),
                   top_k,
                   d_model,
                   &route_elements) &&
               isFP32Capacity(
                   params_.canonical_route_contributions,
                   route_elements) &&
               moe_kernel_;
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::
        hasFixedContract() const noexcept
    {
        if (!hasStaticContract() ||
            !params_.canonical_route_contributions->gpu_data_ptr())
            return false;
        if (params_.transaction->usesAsynchronousLanes())
        {
            return params_.transaction->persistentResourcesReady() &&
                   descriptor_storage_ &&
                   descriptor_storage_->deviceData() &&
                   lane_valid_ && lane_valid_->deviceData();
        }
        return descriptor_storage_ && descriptor_storage_->deviceData() &&
               lane_valid_ && lane_valid_->deviceData();
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::prepareStorage(
        void *stream)
    {
        if (!params_.transaction)
            return false;
        const auto &lanes = params_.transaction->lanes();
        if (descriptor_storage_ && lane_valid_)
        {
            return descriptor_storage_->requireInput(stream) &&
                   lane_valid_->requireOutput(stream) &&
                   hasFixedContract();
        }

        if (params_.transaction->usesAsynchronousLanes())
        {
            std::vector<MoEOverlayActivationReturnConsumeLaunch> launches;
            launches.reserve(lanes.size());
            for (const auto &lane : lanes)
            {
                const auto payload = lane.dispatchPayload(
                    params_.transaction->physicalRows());
                if (!payload.valid())
                    return false;
                MoEOverlayActivationReturnConsumeLaunch launch{
                    .dispatch = payload.packet,
                    .returned = lane.returned,
                    .control = lane.control_device,
                    .grant = lane.grant_device,
                    .canonical_route_contributions_fp32 =
                        static_cast<float *>(
                            params_.canonical_route_contributions->
                                gpu_data_ptr()),
                    .physical_rows = params_.transaction->physicalRows(),
                    .stage_ordinal = params_.transaction->stageOrdinal(),
                    .model_layer_index =
                        params_.transaction->modelLayerIndex(),
                };
                if (!launch.valid() ||
                    launch.returned.d_model !=
                        params_.transaction->dModel())
                {
                    LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] Multi-row descriptor has inconsistent lane geometry");
                    return false;
                }
                launches.push_back(launch);
            }
            if (!uploadPersistentLaunchArray(
                    launches,
                    params_.device_id,
                    stream,
                    &descriptor_storage_,
                    "multi-row return batch"))
            {
                return false;
            }

            lane_valid_ = std::make_shared<
                MoEOverlayPersistentGraphStorage>(
                MoEOverlayPersistentGraphStorage::Config{
                    .device = params_.device_id,
                    .type = MoEOverlayGraphStorageType::Int32,
                    .shape = {lanes.size()},
                    .immutable_input = false,
                    .identity = "multi_row_return_lane_valid",
                });
            try
            {
                return lane_valid_->requireOutput(stream) &&
                       hasFixedContract();
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] Multi-row validation storage preparation failed: "
                          << error.what());
                return false;
            }
        }

        std::vector<MoEOverlayActivationSingleRowReturnConsumeLaunch> launches;
        launches.reserve(lanes.size());
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(
                params_.transaction->stageOrdinal());
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(
                    params_.transaction->stageOrdinal()));
        for (const auto &lane : lanes)
        {
            const auto payload = lane.dispatchPayload(
                /*physical_rows=*/1);
            if (!payload.valid() ||
                !payload.selection.usesCompactRows())
            {
                return false;
            }
            MoEOverlayActivationTimelineWaitDeviceBinding acquire;
            if (!bindKernelTimelineWait(
                    lane,
                    lane.return_signal_offsets[bank],
                    timeline,
                    &acquire,
                    "return batch"))
            {
                return false;
            }
            const MoEOverlayActivationReturnConsumeLaunch packet{
                .dispatch = payload.packet,
                .returned = lane.returned,
                .control = lane.control_device,
                .grant = lane.grant_device,
                .canonical_route_contributions_fp32 = static_cast<float *>(
                    params_.canonical_route_contributions->gpu_data_ptr()),
                .physical_rows = 1,
                .stage_ordinal = params_.transaction->stageOrdinal(),
                .model_layer_index =
                    params_.transaction->modelLayerIndex(),
            };
            MoEOverlayActivationSingleRowReturnConsumeLaunch launch{
                .packet = packet,
                .acquire = acquire,
            };
            if (!launch.valid())
                return false;
            launches.push_back(launch);
        }
        if (!uploadPersistentLaunchArray(
                launches,
                params_.device_id,
                stream,
                &descriptor_storage_,
                "one-row return batch"))
        {
            return false;
        }

        lane_valid_ = std::make_shared<
            MoEOverlayPersistentGraphStorage>(
            MoEOverlayPersistentGraphStorage::Config{
                .device = params_.device_id,
                .type = MoEOverlayGraphStorageType::Int32,
                .shape = {lanes.size()},
                .immutable_input = false,
                .identity = "single_row_return_lane_valid",
            });
        try
        {
            return lane_valid_->requireOutput(stream) &&
                   hasFixedContract();
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] One-row validation storage preparation failed: "
                      << error.what());
            return false;
        }
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::execute(
        IDeviceContext *ctx)
    {
        if (!validateExecutionContext(ctx, params_.device_id, name().c_str()) ||
            !hasFixedContract())
        {
            LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] Invalid fixed launch contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }
        void *const stream = requireGPUStream();
        auto &transaction = *params_.transaction;
        const auto &lanes = transaction.lanes();

        if (transaction.usesAsynchronousLanes())
        {
            try
            {
                auto &gpu_ctx =
                    GPUDeviceContextPool::instance().getContext(
                        params_.device_id);
                for (std::size_t lane_index = 0u;
                     lane_index < lanes.size();
                     ++lane_index)
                {
                    if (!gpu_ctx.waitEventChecked(
                            transaction.laneReturnReadyEvent(lane_index),
                            stream))
                    {
                        LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] Failed to join an imported lane return");
                        return false;
                    }
                }

                /* All arrival timing is outside the payload kernel. Each lane
                 * is authenticated in parallel, then copies only the disjoint
                 * original route slots that it owns. Arithmetic remains the
                 * responsibility of the following canonical reducer. */
                if (!descriptor_storage_->requireInput(stream) ||
                    !lane_valid_->requireOutput(stream))
                {
                    return false;
                }
                const MoEOverlayActivationMultiRowReturnBatchLaunch launch{
                    .lanes = static_cast<const
                        MoEOverlayActivationReturnConsumeLaunch *>(
                            descriptor_storage_->deviceData()),
                    .lane_valid = static_cast<std::int32_t *>(
                        lane_valid_->deviceData()),
                    .canonical_route_contributions_fp32 =
                        static_cast<float *>(
                            params_.canonical_route_contributions->
                                gpu_data_ptr()),
                    .lane_count = static_cast<std::uint32_t>(lanes.size()),
                    .physical_rows = transaction.physicalRows(),
                    .d_model = transaction.dModel(),
                    .top_k = transaction.topK(),
                };
                if (!moe_kernel_->
                        consumeMultiRowMoEOverlayActivationReturnBatch(
                            packetLaunchContext(stream), launch))
                {
                    return false;
                }
                gpuExecution().publish(
                    params_.canonical_route_contributions);
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_async_lane_joins",
                    1.0,
                    "prefill",
                    params_.device_id.toString(),
                    {{"lanes", std::to_string(lanes.size())},
                     {"rows", std::to_string(
                          transaction.physicalRows())},
                     {"canonical_materialization", "true"},
                     {"validation_launches", "1"},
                     {"materialization_launches", "1"},
                     {"legacy_lane_reduction_launches_avoided",
                      std::to_string(lanes.size())},
                     {"layer", std::to_string(
                          transaction.modelLayerIndex())}});
                return true;
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] Asynchronous lane join failed: "
                          << error.what());
                return false;
            }
        }

        if (!descriptor_storage_->requireInput(stream) ||
            !lane_valid_->requireOutput(stream))
        {
            LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] Persistent input/output validation failed");
            return false;
        }

        const auto launch = MoEOverlayActivationSingleRowReturnBatchLaunch{
            .lanes = static_cast<const
                MoEOverlayActivationSingleRowReturnConsumeLaunch *>(
                    descriptor_storage_->deviceData()),
            .lane_valid = static_cast<std::int32_t *>(
                lane_valid_->deviceData()),
            .canonical_route_contributions_fp32 = static_cast<float *>(
                params_.canonical_route_contributions->gpu_data_ptr()),
            .lane_count = static_cast<std::uint32_t>(lanes.size()),
            .d_model = lanes.front().returned.d_model,
            .top_k = lanes.front().dispatch.top_k,
        };
        if (!moe_kernel_->consumeSingleRowMoEOverlayActivationReturnBatch(
                packetLaunchContext(stream), launch))
        {
            return false;
        }
        try
        {
            gpuExecution().publish(params_.canonical_route_contributions);
            PerfStatsCollector::addCounter(
                "forward_graph",
                "moe_overlay_one_row_return_batch_stages",
                1.0,
                "decode",
                params_.device_id.toString(),
                {{"lanes", std::to_string(lanes.size())},
                 {"layer", std::to_string(
                      transaction.modelLayerIndex())}});
            return true;
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[MoEOverlayActivationReturnConsumeBatchStage] Failed to publish canonical route batch materialization: "
                      << error.what());
            return false;
        }
    }

    ComputeStageType MoEOverlayActivationReturnConsumeBatchStage::type() const
    {
        return ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME;
    }

    std::string MoEOverlayActivationReturnConsumeBatchStage::name() const
    {
        return "moe_overlay_activation_return_consume_batch";
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return isGPUBackend(backend);
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::isGraphCapturable() const
    {
        return hasFixedContract();
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::
        supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::
        supportsLazyPrefillGraphCapturePreflight() const
    {
        return hasStaticContract();
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::
        supportsPaddedPrefillGraphCapturePreflight() const
    {
        return params_.transaction &&
               params_.transaction->usesAsynchronousLanes() &&
               hasStaticContract();
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::
        supportsPaddedPrefillRealLengthContract() const
    {
        return params_.transaction &&
               params_.transaction->usesAsynchronousLanes();
    }

    bool MoEOverlayActivationReturnConsumeBatchStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!validateCaptureContext(
                ctx,
                params_.device_id,
                stream,
                "MoEOverlayActivationReturnConsumeBatchStage") ||
            !hasStaticContract())
        {
            return false;
        }
        setGPUStream(stream);
        if (params_.transaction->usesAsynchronousLanes())
        {
            return params_.transaction->materializePersistentResources() &&
                   prepareStorage(stream) && hasFixedContract();
        }
        return prepareStorage(stream) && hasFixedContract();
    }

    GraphLaunchPreparationPolicy
    MoEOverlayActivationReturnConsumeBatchStage::
        graphLaunchPreparationPolicy() const
    {
        return supportsGraphCaptureAfterLaunchPreparation()
                   ? GraphLaunchPreparationPolicy::CaptureOnly
                   : GraphLaunchPreparationPolicy::None;
    }

    StageBufferContract
    MoEOverlayActivationReturnConsumeBatchStage::bufferContract() const
    {
        return StageBufferContract::build().addInOut(
            params_.canonical_route_contributions_buffer_id, "FP32");
    }

    StageBufferRequirements
    MoEOverlayActivationReturnConsumeBatchStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        addInoutRequirement(
            requirements,
            "canonical_route_contributions",
            params_.canonical_route_contributions);
        return requirements;
    }

    std::string MoEOverlayActivationReturnConsumeBatchStage::
        graphCaptureReadinessDebugString() const
    {
        const auto *const transaction = params_.transaction.get();
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer="
            << (transaction ? transaction->modelLayerIndex() : -1)
            << " ordinal="
            << (transaction ? transaction->stageOrdinal() : 0u)
            << " rows="
            << (transaction ? transaction->physicalRows() : 0)
            << " lanes="
            << (transaction ? transaction->lanes().size() : 0u)
            << " static=" << (hasStaticContract() ? "valid" : "invalid")
            << " mode="
            << (transaction && transaction->usesAsynchronousLanes()
                    ? "async_join"
                    : "fused_decode")
            << " resources=";
        if (transaction && transaction->usesAsynchronousLanes())
        {
            out << (transaction->persistentResourcesReady() &&
                             descriptor_storage_ &&
                             descriptor_storage_->deviceData() &&
                             lane_valid_ && lane_valid_->deviceData()
                        ? "ready"
                        : "unbound");
        }
        else
        {
            out << ((descriptor_storage_ &&
                         descriptor_storage_->deviceData() &&
                     lane_valid_ && lane_valid_->deviceData())
                        ? "ready"
                        : "unbound");
        }
        return out.str();
    }

    StageDumpInfo
    MoEOverlayActivationReturnConsumeBatchStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const auto *const transaction = params_.transaction.get();
        if (params_.canonical_route_contributions && transaction &&
            !transaction->lanes().empty())
        {
            info.addInput(
                    "canonical_route_contributions_before",
                    params_.canonical_route_contributions,
                    static_cast<std::size_t>(transaction->physicalRows()) *
                        static_cast<std::size_t>(transaction->topK()),
                    static_cast<std::size_t>(transaction->dModel()))
                .addOutput(
                    "canonical_route_contributions_after",
                    params_.canonical_route_contributions,
                    static_cast<std::size_t>(transaction->physicalRows()) *
                        static_cast<std::size_t>(transaction->topK()),
                    static_cast<std::size_t>(transaction->dModel()));
        }
        info.addScalarInt(
                "lane_count",
                transaction
                    ? static_cast<int>(transaction->lanes().size())
                    : 0)
            .addScalarInt(
                "layer",
                transaction ? transaction->modelLayerIndex() : -1)
            .addScalarInt(
                "stage_ordinal",
                transaction
                    ? static_cast<int>(transaction->stageOrdinal())
                    : 0);
        return info;
    }
} // namespace llaminar2
