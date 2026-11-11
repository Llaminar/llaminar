/**
 * @file ExpertTierGpuPeerTransferLane.cpp
 * @brief Direct peer-copy with original-PMA completion/timestamp event ownership.
 *
 * The immutable native pool supplies the destination's original permit. The
 * source must already belong to that same original preparation/PMA. One exact
 * completion event and the optional timing pair retain their actual flags,
 * streams and conservative process-runtime commitments through checked teardown.
 */

#include "ExpertTierGpuPeerTransferLane.h"
#include "GPUExpertTransferBackend.h"
#include "backends/NativePeerTransferEventOwnership.h"
#include "backends/ReadAheadNativeResourceObligation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"


#include "../../backends/BackendManager.h"
#include "../../backends/GPUDeviceContextPool.h"
#include "../../backends/IBackend.h"
#include "../../backends/IWorkerGPUContext.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    PhysicalMemoryPlan NativePeerTransferEventOwnership::bom(
        PhysicalMemoryResource host,PhysicalMemoryResource gpu,bool timing)
    {
        if(!host.valid() || !gpu.valid() || !host.device.is_cpu() || !gpu.device.is_gpu() || host.world_rank!=gpu.world_rank)
            throw std::invalid_argument("Peer event BOM requires exact original CPU/GPU resources");
        const auto bound=NativeAuxiliaryResourceCatalog::allowance(gpu.device.type,
            NativeAuxiliaryResourceProgram::peerTransferEvents(timing));
        PhysicalMemoryPlanBuilder plan;
        plan.add(host,PhysicalMemoryOwner::NativeTransferEvent,bound.hostBytes);
        if(bound.deviceBytes)plan.add(gpu,PhysicalMemoryOwner::NativeTransferEvent,bound.deviceBytes);
        return plan.build();
    }
    NativePeerTransferEventOwnership::NativePeerTransferEventOwnership(PreparedGPUContextUsePermit original,
        NativeContextRetirementSlot retirement,std::unique_ptr<ReadAheadNativeResourceObligation> pending)
        :original_(std::move(original)),retirement_(std::move(retirement)),pending_(std::move(pending)){}
    std::unique_ptr<NativePeerTransferEventOwnership> NativePeerTransferEventOwnership::materialize(
        const PreparedGPUContextUsePermit& original,bool timing,const std::string& instance)
    {
        if(instance.empty())throw std::invalid_argument("Peer event set requires its admitted physical lane identity");
        if(isGraphCaptureActive())throw std::logic_error("Peer event creation is original cold setup only");
        original.requireWorker(original.worker());
        const auto program=NativeAuxiliaryResourceProgram::peerTransferEvents(timing);
        const auto selected=NativeAuxiliaryResourceCatalog::require(original.device(),program);
        if(selected.contextGeneration!=original.generation() || !selected.hostBytes)
            throw std::logic_error("Peer event program changed original context generation");
        const auto memory=original.memory();
        const auto identity=selected.sourceIdentity+":"+selected.resourceSource+":"+selected.evidence+":"+program.identity();
        auto slot=original.provenance().reserveNativeRetirementSlot(memory,identity);
        // Existing physical edge/projection/lane identity is stable across
        // genuinely retired contexts. The original authority rejects both an
        // overlapping live generation and a foreign source/PMA before creators.
        const auto population="peer-transfer-event-instance-v1:"+std::to_string(instance.size())+":"+instance;
        bool borrowed=false;
        if(selected.lifetime==NativePhysicalLifetime::ProcessRuntime) {
            try {borrowed=slot.tryBorrowProcessRuntime(population);}
            catch(const std::logic_error& error) {
                throw std::logic_error("Peer event population requires its original retired context: "+std::string(error.what()));
            }
        }
        std::unique_ptr<ReadAheadNativeResourceObligation> leaf;
        if(!borrowed) {
            auto host=memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeTransferEvent,selected.hostBytes);
            std::optional<PhysicalMemoryOwnerReservation> gpu;
            if(selected.deviceBytes)gpu.emplace(memory->reserveNewAllocations(original.device(),PhysicalMemoryOwner::NativeTransferEvent,selected.deviceBytes));
            leaf=std::unique_ptr<ReadAheadNativeResourceObligation>(new ReadAheadNativeResourceObligation(
                memory,original.device(),original.generation(),identity,std::move(host),std::move(gpu),selected.lifetime));
        }
        auto owner=std::unique_ptr<NativePeerTransferEventOwnership>(new NativePeerTransferEventOwnership(original,std::move(slot),std::move(leaf)));
        original.worker()->submitAndWait([&] {
            if(owner->pending_)owner->retirement_.adopt(std::move(owner->pending_));
            owner->retirement_.armNativeAttempt();owner->attempted_=true;
            for(std::size_t i=0;i<(timing?3u:1u);++i) {
                owner->events_[i]=original.worker()->createEvent(i==0?GPUEventPurpose::Ordering:GPUEventPurpose::Timing);
                if(!owner->events_[i])throw std::runtime_error("Original peer event creator returned null at index "+std::to_string(i));
            }
        });
        return owner;
    }
    NativePeerTransferEventOwnership::~NativePeerTransferEventOwnership()noexcept
    {
        try {
            if(attempted_) {
                original_.worker()->submitAndWait([&] {
                    for(auto& event:events_)if(event){original_.worker()->destroyEvent(event);event=nullptr;}
                });
                // Both native backends terminate on a failed destroy before
                // returning. Successful return is the original handle proof.
                retirement_.completeNativeHandles();
            }
        }catch(...){std::terminate();}
    }
    PhysicalMemoryPlan ExpertTierGpuPeerTransferLane::nativeEventsBOM(
        PhysicalMemoryResource host,PhysicalMemoryResource gpu,bool timing)
    { return NativePeerTransferEventOwnership::bom(host,gpu,timing); }


    namespace
    {
        /** Store a peer-lane diagnostic only when the caller requested one. */
        void assignPeerError(
            std::string *error,
            const std::string &message) noexcept
        {
            if (error)
                *error = message;
        }

        /** Return whether two endpoints use the same accelerator runtime. */
        bool sameGpuBackend(DeviceId lhs, DeviceId rhs) noexcept
        {
            return (lhs.is_cuda() && rhs.is_cuda()) ||
                   (lhs.is_rocm() && rhs.is_rocm());
        }

        /**
         * @brief Preserve the maintenance thread's ambient runtime device.
         *
         * Context resource/event methods select the resource-owning ordinal.
         * A migration lane can therefore touch a destination device while the
         * caller is about to continue submitting inference work to another
         * participant.  Restoring that caller-owned state at every public lane
         * boundary prevents migration from redirecting subsequent hot-path
         * launches to the wrong GPU.
         */
        class AmbientGpuDeviceGuard final
        {
        public:
            /** Capture the current device for the endpoint's accelerator API. */
            explicit AmbientGpuDeviceGuard(DeviceId endpoint) noexcept
                : endpoint_(endpoint)
            {
#ifdef HAVE_CUDA
                if (endpoint_.is_cuda())
                {
                    saved_ordinal_ = detail::currentCUDADeviceOrdinal();
                    valid_ = saved_ordinal_ >= 0;
                    return;
                }
#endif
#ifdef HAVE_ROCM
                if (endpoint_.is_rocm())
                {
                    saved_ordinal_ = detail::currentROCmDeviceOrdinal();
                    valid_ = saved_ordinal_ >= 0;
                }
#endif
            }

            /** Restore both runtime state and ROCm's device-selection cache. */
            ~AmbientGpuDeviceGuard()
            {
                if (!valid_)
                    return;
#ifdef HAVE_CUDA
                if (endpoint_.is_cuda())
                {
                    if (!detail::restoreCUDADeviceOrdinal(saved_ordinal_))
                    {
                        LOG_WARN("[ExpertTierGpuPeerTransferLane] Could not restore CUDA device "
                                 << saved_ordinal_);
                    }
                    return;
                }
#endif
#ifdef HAVE_ROCM
                if (endpoint_.is_rocm())
                {
                    if (!detail::restoreROCmDeviceOrdinal(saved_ordinal_))
                    {
                        LOG_WARN("[ExpertTierGpuPeerTransferLane] Could not restore ROCm device "
                                 << saved_ordinal_);
                    }
                }
#endif
            }

            AmbientGpuDeviceGuard(const AmbientGpuDeviceGuard &) = delete;
            AmbientGpuDeviceGuard &operator=(
                const AmbientGpuDeviceGuard &) = delete;

        private:
            DeviceId endpoint_;
            int saved_ordinal_ = -1;
            bool valid_ = false;
        };
    } // namespace

    ExpertTierGpuPeerTransferLane::ExpertTierGpuPeerTransferLane(Config config)
        : config_(std::move(config))
    {
        if (!config_.source_device.is_gpu() ||
            !config_.destination_device.is_gpu() ||
            !sameGpuBackend(
                config_.source_device,
                config_.destination_device))
        {
            throw std::invalid_argument(
                "GPU peer tier lane requires two same-backend GPU endpoints");
        }
        if (!config_.execution.valid() ||
            config_.execution.device() != config_.destination_device)
        {
            throw std::invalid_argument(
                "GPU peer tier lane requires one matching destination execution lane");
        }
        // Component stream pools have no original PMA/context authority and
        // cannot enter this admitted native event owner, even before materialize.
        (void)config_.execution.originalContext();
        if (config_.lane_name.empty())
            throw std::invalid_argument(
                "GPU peer tier lane requires a stable non-empty name");
        if (config_.perf_device.empty())
        {
            config_.perf_device = config_.source_device.to_string() + "->" +
                                  config_.destination_device.to_string();
        }
    }

    ExpertTierGpuPeerTransferLane::~ExpertTierGpuPeerTransferLane()
    {
        if (work_may_be_in_flight_)
        {
            LOG_ERROR("[ExpertTierGpuPeerTransferLane] Destroyed with pending work"
                      << " lane=" << config_.lane_name
                      << " src=" << config_.source_device.to_string()
                      << " dst=" << config_.destination_device.to_string());
            std::terminate();
        }
        releaseQuiescentResources();
    }

    bool ExpertTierGpuPeerTransferLane::materialized() const noexcept
    {
        return materialization_phase_==MaterializationPhase::Ready && native_events_ &&
               backend_ && source_context_ && destination_context_ &&
               destination_ordinal_ >= 0 && transfer_stream_ &&
               completion_event_ &&
               (!config_.collect_timing_measurements ||
                (timing_start_event_ && timing_stop_event_));
    }

    bool ExpertTierGpuPeerTransferLane::materialize(
        std::string *error) noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (materialized())
            return true;
        if(materialization_phase_!=MaterializationPhase::Fresh) {
            assignPeerError(error,materialization_failure_.empty()?"GPU peer tier lane setup is not reusable":materialization_failure_);
            return false;
        }
        // Creation may retain process-runtime backing even when the first
        // native API fails. Every attempt is absorbing until fully completed.
        materialization_phase_=MaterializationPhase::Failed;

        if (backend_ || source_context_ || destination_context_ ||
            destination_ordinal_ >= 0 || transfer_stream_ || completion_event_ ||
            timing_start_event_ || timing_stop_event_)
        {
            materialization_failure_="GPU peer tier lane has a partial resource set";
            assignPeerError(error, materialization_failure_);
            return false;
        }

        try
        {
            const auto& destination_original=config_.execution.originalContext();
            if(destination_original.device()!=config_.destination_device)
                throw std::logic_error("Peer lane changed original destination device");
            source_original_=std::make_unique<PreparedGPUContextUsePermit>(
                GPUDeviceContextPool::instance().retainExistingPreparedContext(
                    config_.source_device,destination_original.memory()));
            if (config_.source_device != config_.destination_device)
            {
                bool direct_peer_ready = false;
#ifdef HAVE_CUDA
                if (config_.source_device.is_cuda())
                {
                    direct_peer_ready =
                        detail::prepareDirectPeerAccessCUDABackend(
                            config_.source_device,
                            config_.destination_device);
                }
#endif
#ifdef HAVE_ROCM
                if (config_.source_device.is_rocm())
                {
                    direct_peer_ready =
                        detail::prepareDirectPeerAccessROCmBackend(
                            config_.source_device,
                            config_.destination_device);
                }
#endif
                if (!direct_peer_ready)
                {
                    throw std::runtime_error(
                        "GPU peer tier lane requires a driver-authorized direct edge; use the explicit host-relay lane when peer access is unavailable");
                }
            }

            IBackend *source_backend = getBackendFor(config_.source_device);
            IBackend *destination_backend =
                getBackendFor(config_.destination_device);
            if (!source_backend || source_backend != destination_backend)
            {
                throw std::runtime_error(
                    "GPU peer tier lane endpoints do not share one backend authority");
            }
            backend_ = destination_backend;
            source_context_ = source_original_->worker().get();
            destination_context_ = destination_original.worker().get();
            destination_ordinal_ = config_.destination_device.gpu_ordinal();
            transfer_stream_ = config_.execution.stream();
            const auto& original=config_.execution.originalContext();
            if(original.device()!=config_.destination_device || original.worker().get()!=destination_context_)
                throw std::logic_error("Peer transfer events changed original destination authority");
            native_events_=NativePeerTransferEventOwnership::materialize(original,config_.collect_timing_measurements,config_.lane_name);
            completion_event_=native_events_->event(0);
            timing_start_event_=native_events_->event(1);
            timing_stop_event_=native_events_->event(2);
            if (!transfer_stream_ || !completion_event_ ||
                (config_.collect_timing_measurements &&
                 (!timing_start_event_ || !timing_stop_event_)))
                throw std::runtime_error(
                    "Could not materialize GPU peer stream/event resources");
        }
        catch (const std::exception &exception)
        {
            materialization_failure_=exception.what();
            assignPeerError(error, materialization_failure_);
            releaseQuiescentResources();
            return false;
        }
        catch (...)
        {
            materialization_failure_="GPU peer tier lane materialization threw a non-standard exception";
            assignPeerError(error,materialization_failure_);
            releaseQuiescentResources();
            return false;
        }
        materialization_phase_=MaterializationPhase::Ready;
        return true;
    }

    bool ExpertTierGpuPeerTransferLane::start(
        const GpuExpertPackedDescriptor &source,
        const GpuExpertPackedDescriptor &destination,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        if (!gpuExpertPackedDescriptorsCompatible(source, destination))
        {
            assignPeerError(
                error,
                "GPU peer tier transfer descriptors are not byte-compatible");
            return false;
        }
        return startRegions(
            source.ptrs,
            destination.ptrs,
            source.vnni_bytes,
            source.scales_bytes,
            source.mins_bytes,
            source.emins_bytes,
            source_readiness,
            error);
    }

    bool ExpertTierGpuPeerTransferLane::startContiguous(
        const void *source,
        void *destination,
        std::size_t bytes,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        if (!source || !destination || bytes == 0)
        {
            assignPeerError(
                error,
                "GPU peer contiguous transfer requires two pointers and non-zero bytes");
            return false;
        }
        GPUExpertPointers source_regions;
        source_regions.d_vnni = static_cast<std::uint8_t *>(
            const_cast<void *>(source));
        GPUExpertPointers destination_regions;
        destination_regions.d_vnni =
            static_cast<std::uint8_t *>(destination);
        return startRegions(
            source_regions,
            destination_regions,
            bytes,
            0,
            0,
            0,
            source_readiness,
            error);
    }

    bool ExpertTierGpuPeerTransferLane::startRegions(
        const GPUExpertPointers &source,
        const GPUExpertPointers &destination,
        std::size_t payload_bytes,
        std::size_t scales_bytes,
        std::size_t mins_bytes,
        std::size_t emins_bytes,
        const ExpertTierSourceReadiness &source_readiness,
        std::string *error) noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (!materialized())
        {
            assignPeerError(error, "GPU peer tier lane is not materialized");
            return false;
        }
        if (progress_ == ExpertTierGpuPeerTransferProgress::Pending ||
            progress_ == ExpertTierGpuPeerTransferProgress::Failed)
        {
            assignPeerError(error, "GPU peer tier lane is already occupied");
            return false;
        }

        /*
         * A fresh producer edge is joined by the destination auxiliary stream.
         * A published bank needs no event: its preparation event completed
         * before installation and its epoch lifetime keeps bytes immutable.
         */
        if (source_readiness.requiresProducerWait())
        {
            if (!destination_context_->waitEventChecked(
                    source_readiness.event(), transfer_stream_))
            {
                assignPeerError(
                    error,
                    "Could not enqueue GPU peer source producer dependency");
                return false;
            }
            ++stats_.producer_event_waits;
        }
        else
        {
            ++stats_.published_bank_sources;
        }

        transfer_started_at_ = std::chrono::steady_clock::now();
        bool timing_started = true;
        if (config_.collect_timing_measurements)
        {
            timing_started = backend_->recordEvent(
                timing_start_event_,
                destination_ordinal_,
                transfer_stream_);
        }

        submitted_bytes_ = payload_bytes + scales_bytes + mins_bytes +
                           emins_bytes;
        bool copied = timing_started && GPUExpertTransfer::transferExpert(
            source,
            destination,
            config_.source_device,
            config_.destination_device,
            payload_bytes,
            scales_bytes,
            mins_bytes,
            emins_bytes,
            transfer_stream_);
        bool timing_stopped = true;
        if (config_.collect_timing_measurements)
        {
            timing_stopped = backend_->recordEvent(
                timing_stop_event_,
                destination_ordinal_,
                transfer_stream_);
            copied = copied && timing_stopped;
            if (!timing_started || !timing_stopped)
                ++stats_.timing_measurement_failures;
        }

        /* Fence even a partially accepted copy sequence before reporting it. */
        if (!backend_->recordEvent(
                completion_event_,
                destination_ordinal_,
                transfer_stream_))
        {
            failure_ =
                "GPU peer tier transfer could not fence submitted device work";
            assignPeerError(error, failure_);
            work_may_be_in_flight_ = true;
            progress_ = ExpertTierGpuPeerTransferProgress::Failed;
            ++stats_.failed_transfers;
            return false;
        }

        work_may_be_in_flight_ = true;
        fail_after_event_ = !copied;
        failure_ = copied
                       ? std::string{}
                       : (!timing_started || !timing_stopped)
                             ? "GPU peer tier transfer failed to record exact timing events"
                             : "GPU peer tier transfer failed to submit every packed region";
        progress_ = ExpertTierGpuPeerTransferProgress::Pending;
        ++stats_.transfers_started;
        stats_.bytes_submitted += submitted_bytes_;
        return true;
    }

    ExpertTierGpuPeerTransferProgress ExpertTierGpuPeerTransferLane::poll(
        std::string *error) noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (progress_ != ExpertTierGpuPeerTransferProgress::Pending)
        {
            if (progress_ == ExpertTierGpuPeerTransferProgress::Failed)
                assignPeerError(error, failure_);
            return progress_;
        }

        bool ready = false;
        if (!destination_context_->queryEventChecked(completion_event_, ready))
        {
            return fail(
                "GPU peer tier completion event query failed", error);
        }
        if (!ready)
        {
            ++stats_.pending_event_polls;
            return progress_;
        }

        work_may_be_in_flight_ = false;
        if (fail_after_event_)
        {
            fail_after_event_ = false;
            return fail(failure_, error);
        }

        progress_ = ExpertTierGpuPeerTransferProgress::Ready;
        ++stats_.transfers_completed;
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - transfer_started_at_);
        const std::uint64_t wall_nanoseconds = static_cast<std::uint64_t>(
            std::max<std::int64_t>(1, elapsed.count()));
        std::uint64_t device_nanoseconds = 0;
        if (config_.collect_timing_measurements)
        {
            float elapsed_ms = 0.0F;
            if (!backend_->eventElapsedTimeMs(
                    timing_start_event_,
                    timing_stop_event_,
                    destination_ordinal_,
                    &elapsed_ms) ||
                !std::isfinite(elapsed_ms) || elapsed_ms < 0.0F)
            {
                ++stats_.timing_measurement_failures;
                --stats_.transfers_completed;
                return fail(
                    "GPU peer tier transfer could not read its exact device timing interval",
                    error);
            }
            device_nanoseconds = static_cast<std::uint64_t>(std::max(
                1.0,
                std::ceil(static_cast<double>(elapsed_ms) * 1'000'000.0)));
        }
        stats_.last_measurement = {
            .sequence = stats_.transfers_completed,
            .bytes = static_cast<std::uint64_t>(submitted_bytes_),
            .wall_nanoseconds = wall_nanoseconds,
            .device_nanoseconds = device_nanoseconds,
            .host_nanoseconds = 0,
        };
        const PerfStatsCollector::Tags tags{
            {"lane", config_.lane_name},
            {"src", config_.source_device.to_string()},
            {"dst", config_.destination_device.to_string()}};
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "same_backend_gpu_transfers_completed",
            1.0,
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "same_backend_gpu_bytes_completed",
            static_cast<double>(submitted_bytes_),
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::recordTimingNs(
            "moe_overlay_residency",
            "same_backend_gpu_transfer_wall_time",
            wall_nanoseconds,
            "maintenance",
            config_.perf_device,
            tags);
        if (config_.collect_timing_measurements)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_overlay_residency",
                "same_backend_gpu_transfer_device_work_time",
                device_nanoseconds,
                "maintenance",
                config_.perf_device,
                tags);
        }
        return progress_;
    }

    ExpertTierGpuPeerTransferProgress ExpertTierGpuPeerTransferLane::fail(
        const std::string &message,
        std::string *error) noexcept
    {
        failure_ = message.empty()
                       ? "GPU peer tier transfer failed"
                       : message;
        assignPeerError(error, failure_);
        progress_ = ExpertTierGpuPeerTransferProgress::Failed;
        ++stats_.failed_transfers;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "same_backend_gpu_transfer_failures",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"lane", config_.lane_name},
             {"src", config_.source_device.to_string()},
             {"dst", config_.destination_device.to_string()}});
        return progress_;
    }

    void ExpertTierGpuPeerTransferLane::releaseQuiescentResources() noexcept
    {
        const AmbientGpuDeviceGuard ambient_device(
            config_.destination_device);
        if (work_may_be_in_flight_)
            return;
        native_events_.reset(); // Actual checked destruction precedes original obligation completion.
        completion_event_ = nullptr;
        timing_start_event_ = nullptr;
        timing_stop_event_ = nullptr;
        transfer_stream_ = nullptr; // Context owns the named stream.
        source_context_ = nullptr;
        source_original_.reset();
        destination_context_ = nullptr;
        backend_ = nullptr;
        destination_ordinal_ = -1;
    }
} // namespace llaminar2
