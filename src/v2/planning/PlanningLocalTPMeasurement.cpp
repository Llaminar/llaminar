/**
 * @file PlanningLocalTPMeasurement.cpp
 * @brief Production native LocalTP costs with parallel participant capture and replay.
 *
 * Admission precedes collective construction. All participants finish payload
 * preparation before any capture begins. A phase records participant-local
 * graphs concurrently, then launches them concurrently on their exact streams.
 * Every graph retires before its tensor and before LocalTPContext, whose native
 * communicator may still be borrowed by a retained graph. No graph is recaptured
 * to recover from failure and no failed participant becomes a zero-cost sample.
 * Native allreduce and fixed and counted allgather phases use one contiguous warmup and three timed body imports
 * inside an ordered fixed-clock parent; physical pair timing remains separate.
 */
#include "PlanningLocalTPMeasurement.h"
#include "PlanningExecutionMeasurement.h"
#include "PlanningGPUClockBuffer.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "transfer/PreparedDevicePeerCopy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>
#include <tuple>
#include <exception>
#include <future>
#include <limits>
#include <climits>
#include "collective/DeviceCountedAllGather.h"
#include <set>
#include <stdexcept>

namespace llaminar2
{
    PlanningLocalTPRequest::PlanningLocalTPRequest(std::vector<DeviceId> devices,
        int hidden_width, int prefill_rows, PlanningAllreducePrecision precision)
        : devices_(std::move(devices)), hidden_width_(hidden_width),
          prefill_rows_(prefill_rows), precision_(precision)
    {
        if (devices_.size() < 2 || devices_.size() > size_t(INT_MAX) || hidden_width <= 0 || prefill_rows <= 0 ||
            (precision != PlanningAllreducePrecision::FP32 && precision != PlanningAllreducePrecision::FP16))
            throw std::invalid_argument("Planning native LocalTP requires a group, positive payload geometry and explicit precision");
        std::set<int> ordinals;
        for (const auto device : devices_)
            if ((!device.is_cuda() && !device.is_rocm()) || device.ordinal < 0 ||
                device.type != devices_.front().type || !ordinals.insert(device.ordinal).second)
                throw std::invalid_argument("Planning native LocalTP requires distinct homogeneous rank-local GPU ordinals");
        if (size_t(prefill_rows) > std::numeric_limits<size_t>::max() / sizeof(float) / size_t(hidden_width))
            throw std::overflow_error("Planning native LocalTP payload extent overflow");
        // The canonical contract also checks its physical guard/capacity arithmetic.
        (void)CollectiveMemoryEstimator::localTP(prefill_rows, hidden_width, backend());
    }

    CollectiveBackendType PlanningLocalTPRequest::backend() const noexcept
    {
        return devices_.front().is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
    }

    size_t PlanningLocalTPRequest::activationBytes() const noexcept
    {
        return size_t(prefill_rows_) * size_t(hidden_width_) * sizeof(float);
    }

    size_t PlanningLocalTPRequest::payloadBytes(int rows) const
    {
        if (rows != 1 && rows != prefill_rows_)
            throw std::invalid_argument("Planning native LocalTP payload was not requested");
        return size_t(rows) * size_t(hidden_width_) *
            (precision_ == PlanningAllreducePrecision::FP32 ? sizeof(float) : sizeof(uint16_t));
    }

    std::vector<size_t> PlanningLocalTPRequest::pairPayloadAnchors() const
    {
        const size_t decode = size_t(hidden_width_) * sizeof(float);
        std::vector<size_t> anchors{std::min(size_t{16}, decode), decode, activationBytes()};
        std::sort(anchors.begin(), anchors.end());
        anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
        return anchors;
    }

    std::vector<PlanningLocalTPPayloadRegion> PlanningLocalTPRequest::pairPayloadRegions() const
    {
        std::vector<PlanningLocalTPPayloadRegion> regions;
        size_t cursor = 0;
        constexpr size_t guard = 16;
        for (const size_t bytes : pairPayloadAnchors())
        {
            if (bytes % sizeof(float) || cursor > std::numeric_limits<size_t>::max() - bytes ||
                cursor + bytes > std::numeric_limits<size_t>::max() - guard)
                throw std::overflow_error("Planning guarded pair payload extent/alignment is invalid");
            regions.push_back({cursor, bytes, guard});
            cursor += bytes + guard;
        }
        return regions;
    }

    size_t PlanningLocalTPRequest::storageBytes() const
    {
        const auto regions = pairPayloadRegions();
        const auto &last = regions.back();
        return std::max(activationBytes(), last.byte_offset + last.live_bytes + last.guard_bytes);
    }

    void PlanningLocalTPPairSampling::validate(const PlanningLocalTPRequest &request) const
    {
        if (payload_anchors != request.pairPayloadAnchors() ||
            !std::isfinite(setup_wall_seconds) || setup_wall_seconds < 0 ||
            !std::isfinite(measurement_wall_seconds) || measurement_wall_seconds < 0)
            throw std::invalid_argument("Planning directed pair matrix has foreign payload or setup identity");
        const size_t degree = request.devices().size();
        if (degree > SIZE_MAX / degree || direct_peer_access.size() != degree * degree)
            throw std::invalid_argument("Planning physical pair matrix lacks complete directed access observations");
        for (size_t source = 0; source < degree; ++source)
            for (size_t destination = 0; destination < degree; ++destination)
                if (direct_peer_access[source * degree + destination] > 1 ||
                    (source == destination && direct_peer_access[source * degree + destination]))
                    throw std::invalid_argument("Planning physical pair matrix has invalid direct access observations");
        std::set<std::tuple<size_t, size_t, size_t>> seen;
        for (const auto &pair : directed_pairs)
        {
            if (pair.source_index >= degree || pair.destination_index >= degree ||
                pair.source_index == pair.destination_index || !pair.graph_nodes ||
                !direct_peer_access[pair.source_index * degree + pair.destination_index] ||
                !std::binary_search(payload_anchors.begin(), payload_anchors.end(), pair.payload_bytes) ||
                pair.seconds_per_transfer.size() != kSamples ||
                !seen.emplace(pair.source_index, pair.destination_index, pair.payload_bytes).second)
                throw std::invalid_argument("Planning directed pair matrix has missing/duplicate native edge identity");
            for (const double seconds : pair.seconds_per_transfer)
                if (!std::isfinite(seconds) || seconds <= 0)
                    throw std::invalid_argument("Planning directed pair matrix has incomplete fixed-clock timing");
        }
        // Checking every expected tuple avoids unchecked degree-squared arithmetic
        // and proves both directions even for odd participant counts.
        for (size_t source = 0; source < degree; ++source)
            for (size_t destination = 0; destination < degree; ++destination)
                if (direct_peer_access[source * degree + destination])
                    for (const auto bytes : payload_anchors)
                        if (!seen.contains({source, destination, bytes}))
                            throw std::invalid_argument("Planning directed pair matrix is incomplete");
        if (capture_count != seen.size() / payload_anchors.size() || measured_launches != capture_count)
            throw std::invalid_argument("Planning directed pair matrix has incomplete capture/replay counters");
    }

    double PlanningLocalTPPhaseObservation::secondsPerCollective() const
    {
        if (rows <= 0 || !payload_bytes || endpoints.size() < 2)
            throw std::invalid_argument("Planning native LocalTP phase is incomplete");
        double result = 0;
        std::set<int> ordinals;
        for (const auto &endpoint : endpoints)
        {
            if ((!endpoint.device.is_cuda() && !endpoint.device.is_rocm()) || endpoint.device.ordinal < 0 ||
                endpoint.device.type != endpoints.front().device.type || !ordinals.insert(endpoint.device.ordinal).second ||
                !endpoint.graph_nodes || !std::isfinite(endpoint.seconds_per_collective) ||
                endpoint.seconds_per_collective <= 0)
                throw std::invalid_argument("Planning native LocalTP endpoint did not complete a captured observation");
            result = std::max(result, endpoint.seconds_per_collective);
        }
        return result;
    }

    size_t PlanningLocalTPRequest::gatherStorageBytes() const
    {
        constexpr size_t guard=16;
        if(devices_.size()>(std::numeric_limits<size_t>::max()-guard)/activationBytes())
            throw std::overflow_error("Native gather receive-bank extent overflow");
        return devices_.size()*activationBytes()+guard;
    }

    double PlanningLocalTPGatherPhaseObservation::secondsPerCollective() const
    {
        if((protocol!=PlanningNativeGatherProtocol::FixedAllgather && protocol!=PlanningNativeGatherProtocol::CountedAllgather) ||
           (protocol==PlanningNativeGatherProtocol::FixedAllgather && !payload_bytes) ||
           participant_count<2 || participant_count!=endpoints.size())
            throw std::invalid_argument("Planning native gather phase has incomplete original membership");
        double seconds=0;std::set<int> ordinals;
        for(const auto &endpoint:endpoints) {
            if(!endpoint.device.is_gpu() || endpoint.device.type!=endpoints.front().device.type ||
               endpoint.device.ordinal<0 || !ordinals.insert(endpoint.device.ordinal).second || !endpoint.graph_nodes ||
               !std::isfinite(endpoint.seconds_per_collective) || endpoint.seconds_per_collective<=0)
                throw std::invalid_argument("Planning native gather endpoint did not complete its captured protocol");
            seconds=std::max(seconds,endpoint.seconds_per_collective);
        }
        return seconds;
    }

    namespace
    {
        /** @brief Fail at the boundary that lost its required native operation. */
        void require(bool condition, const std::string &detail)
        {
            if (!condition) throw std::runtime_error(std::string("Planning native LocalTP: ") + detail);
        }

        /**
         * @brief Submit every endpoint before joining; retain the first failure after all jobs retire.
         *
         * An asymmetric preparation/capture failure cancels LocalTP's host
         * rendezvous so peers cannot wait forever for a missing participant.
         * requestAbort deliberately does not destroy native graph borrowers.
         * @param selected Original request indices; empty selects the complete group.
         * Pair setup submits only its two active endpoints. Idle communicator
         * members need no worker closure, graph or native progress operation.
         */
        void onWorkers(const PlanningLocalTPRequest &request, LocalTPContext *collective,
            const std::function<void(size_t, IWorkerGPUContext &)> &operation,
            std::span<const size_t> selected = {})
        {
            for (size_t position = 0; position < selected.size(); ++position)
                if (selected[position] >= request.devices().size() ||
                    std::find(selected.begin(), selected.begin() + position, selected[position]) != selected.begin() + position)
                    throw std::invalid_argument("Planning native worker selection has a duplicate/foreign original index");
            const size_t count = selected.empty() ? request.devices().size() : selected.size();
            std::vector<std::future<void>> jobs;
            // Allocate bookkeeping before submission: a throwing push must not
            // lose the only join handle to an already running GPU operation.
            jobs.reserve(count);
            std::exception_ptr failure;
            try
            {
                for (size_t position = 0; position < count; ++position)
                {
                    const size_t index = selected.empty() ? position : selected[position];
                    auto &worker = GPUDeviceContextPool::instance().getContext(request.devices()[index]);
                    jobs.push_back(worker.submitAsync([&, index, worker_ptr = &worker] {
                        try { operation(index, *worker_ptr); }
                        catch (...) { if (collective) collective->requestAbort(); throw; }
                    }));
                }
            }
            catch (...) { failure = std::current_exception(); if (collective) collective->requestAbort(); }
            for (auto &job : jobs)
                try { job.get(); }
                catch (...) { if (!failure) failure = std::current_exception(); }
            if (failure) std::rethrow_exception(failure);
        }

        /** @brief Payload storage and graph borrows retire in the reverse order of their declarations. */
        struct Participant final
        {
            PhysicalMemoryAllocationLease host_claim, device_claim, receive_host_claim, receive_device_claim;
            PhysicalMemoryOwnerReservation graphs_claim;
            std::shared_ptr<FP32Tensor> tensor,received;
            DeviceCountedAllGatherBinding counted_binding;
            // Native phases retain one four-graph timing transaction at a time.
            // Pair-only capture keeps its separate original graph/clock protocol.
            std::unique_ptr<PlanningGPUClockBuffer> collective_clock;
            std::optional<PlanningPreparedGPUObservation> collective_timing;
            std::unique_ptr<PreparedDevicePeerCopy> pair_copy;
            std::unique_ptr<PlanningNativeGPUClockBuffer> pair_clock;
            std::unique_ptr<PlanningCapturedGPUClockSampler> pair_timer;
            std::unique_ptr<IGPUGraphCapture> graph;
            size_t graph_growth = 0;
            void *stream = nullptr;

            /** @brief Claim the complete sample family before creating any GPU payload. */
            Participant(const PlanningLocalTPRequest &request, DeviceId device,
                const std::shared_ptr<PhysicalMemoryAuthority> &memory, PlanningLocalTPMeasurementMode mode)
                : host_claim(memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                      request.storageBytes())),
                  device_claim(memory->claimNewAllocation(device, PhysicalMemoryOwner::ExecutionWorkspace,
                      request.storageBytes())),
                  graphs_claim(memory->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable,
                      (mode == PlanningLocalTPMeasurementMode::NativeCollective ? PlanningExecutionMeasurement::kPreparedGraphCount : 1) *
                          GPUGraphMemoryContract::reservationBytesPerExecutable(device)))
            {
                if(mode==PlanningLocalTPMeasurementMode::NativeCollective) {
                    receive_host_claim=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,request.gatherStorageBytes());
                    receive_device_claim=memory->claimNewAllocation(device,PhysicalMemoryOwner::ExecutionWorkspace,request.gatherStorageBytes());
                }
            }
        };
    }

    void PlanningLocalTPMeasurement::contributeMemory(const PlanningLocalTPRequest &request,
        PhysicalMemoryResource host, std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryPlanBuilder &builder, PlanningLocalTPMeasurementMode mode)
    {
        if (host.device != DeviceId::cpu() || devices.size() != request.devices().size())
            throw std::invalid_argument("Planning native LocalTP requires same-rank CPU staging and every GPU resource");
        for (size_t index = 0; index < devices.size(); ++index)
            if (devices[index].world_rank != host.world_rank || devices[index].device != request.devices()[index])
                throw std::invalid_argument("Planning native LocalTP cannot use a different rank or device's resource");
        if (mode != PlanningLocalTPMeasurementMode::NativeCollective && mode != PlanningLocalTPMeasurementMode::PhysicalPeersOnly)
            throw std::invalid_argument("Unknown planning measurement mode");
        const auto local = CollectiveMemoryEstimator::localTP(request.prefillRows(), request.hiddenWidth(), request.backend());
        // Native two-slot clocks retire before pair clocks materialize, so one
        // maximum-sized backing allowance covers these mutually exclusive uses.
        const size_t pair_clock_slots = 2 * request.pairPayloadAnchors().size() * PlanningLocalTPPairSampling::kSamples;
        const size_t clock_slots = std::max(pair_clock_slots,
            mode == PlanningLocalTPMeasurementMode::NativeCollective ? size_t{2} : size_t{0});
        for (const auto resource : devices)
        {
            builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, request.storageBytes());
            builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, request.storageBytes());
            builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace,
                PlanningNativeGPUClockBuffer::hostAllocationBytes(clock_slots));
            builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace,
                PlanningNativeGPUClockBuffer::deviceAllocationBytes(clock_slots));
            if (mode == PlanningLocalTPMeasurementMode::NativeCollective) {
                builder.add(resource, PhysicalMemoryOwner::LocalCollective, local.perDeviceBytes());
                builder.add(host,PhysicalMemoryOwner::ExecutionWorkspace,request.gatherStorageBytes());
                builder.add(resource,PhysicalMemoryOwner::ExecutionWorkspace,request.gatherStorageBytes());
                builder.add(resource,PhysicalMemoryOwner::ActivationTransportStaging,
                    NativeCountedAllGatherLayout(devices.size()).bytes());
            }
            builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable,
                (mode == PlanningLocalTPMeasurementMode::NativeCollective ? PlanningExecutionMeasurement::kPreparedGraphCount : 1) *
                    GPUGraphMemoryContract::reservationBytesPerExecutable(resource.device));
        }
    }

    PlanningLocalTPObservations PlanningLocalTPMeasurement::measure(const PlanningLocalTPRequest &request,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, PlanningPairSampling pairs,
        PlanningMeasurementValidation validation, std::span<const std::uint8_t> direct_peer_access,
        PlanningLocalTPMeasurementMode mode,const DeviceCountedAllGatherTopology *counted_topology)
    {
        if(mode==PlanningLocalTPMeasurementMode::NativeCollective && (!counted_topology ||
           counted_topology->devices()!=request.devices() || !memory || counted_topology->worldRank()!=memory->worldRank()))
            throw std::invalid_argument("Native calibration requires its genuine original counted topology");
        if ((mode != PlanningLocalTPMeasurementMode::NativeCollective && mode != PlanningLocalTPMeasurementMode::PhysicalPeersOnly) ||
            (mode == PlanningLocalTPMeasurementMode::PhysicalPeersOnly && pairs != PlanningPairSampling::Enabled))
            throw std::invalid_argument("Physical-only measurement requires explicit directed peer sampling");
        if (pairs != PlanningPairSampling::Enabled && pairs != PlanningPairSampling::Disabled)
            throw std::invalid_argument("Planning directed pair sampling policy is invalid");
        if ((validation != PlanningMeasurementValidation::None && validation != PlanningMeasurementValidation::ObservePayloads) ||
            (pairs == PlanningPairSampling::Disabled && validation != PlanningMeasurementValidation::None &&
                mode!=PlanningLocalTPMeasurementMode::NativeCollective))
            throw std::invalid_argument("Planning payload validation requires native gathers or enabled directed pair samples");
        if (!memory) throw std::invalid_argument("Planning native LocalTP requires admitted physical memory");
        const size_t degree = request.devices().size();
        if ((pairs == PlanningPairSampling::Disabled && !direct_peer_access.empty()) ||
            (pairs == PlanningPairSampling::Enabled &&
                (degree > SIZE_MAX / degree || direct_peer_access.size() != degree * degree)))
            throw std::invalid_argument("Planning physical pair sample requires its exact directed access observations");
        for (size_t source = 0; source < degree && pairs == PlanningPairSampling::Enabled; ++source)
            for (size_t destination = 0; destination < degree; ++destination)
                if (direct_peer_access[source * degree + destination] > 1 ||
                    (source == destination && direct_peer_access[source * degree + destination]))
                    throw std::invalid_argument("Planning physical pair sample has malformed directed access observations");
        for (const auto device : request.devices())
            if (GPUDeviceContextPool::instance().getContext(device).ownsCurrentThread())
                throw std::invalid_argument("Planning native LocalTP must be driven by the rank setup owner, not a participant worker");
        // Claim every endpoint before native initialization. A capacity failure
        // therefore cannot strand another GPU inside collective capture.
        // Declaration order is also the last-resort exception-unwind order:
        // participant graph owners always die before the native communicator.
        std::unique_ptr<LocalTPContext> collective_owner;
        std::vector<std::unique_ptr<Participant>> participants;
        std::shared_ptr<DeviceCountedAllGather> counted;
        std::vector<GlobalDeviceAddress> addresses;
        for (const auto device : request.devices())
        {
            participants.push_back(std::make_unique<Participant>(request, device, memory, mode));
            addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
        }
        // Physical copies use only original workers and the public transfer
        // capability. Creating a native communicator here would calibrate an
        // unused protocol and add seconds to automatic placement startup.
        if (mode == PlanningLocalTPMeasurementMode::NativeCollective)
            collective_owner = std::make_unique<LocalTPContext>(std::move(addresses), std::vector<float>{}, request.backend());
        PlanningLocalTPObservations result{request, {}};
        result.mode = mode;
        std::exception_ptr failure;
        try
        {
            const auto local = CollectiveMemoryEstimator::localTP(request.prefillRows(), request.hiddenWidth(), request.backend());
            if (collective_owner)
                require(collective_owner->reserveCollectiveResources(local.backend_payload_capacity_bytes,
                    local.fp16_scratch_elements, memory), "collective resource admission failed");
            onWorkers(request, collective_owner.get(), [&](size_t index, auto &worker) {
                auto &owner = *participants[index];
                owner.stream = ExplicitGPUStream(worker.defaultStream()).get();
                owner.tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{1, request.storageBytes() / sizeof(float)});
                // Zero is stable under repeated in-place sums. The functional
                // integration oracle separately checks non-zero production sums.
                std::fill_n(owner.tensor->mutable_typed_data(), owner.tensor->numel(), 0.0f);
                TransferEngine::prepareDeviceInput(owner.tensor.get(), request.devices()[index], owner.stream);
                TransferEngine::requireDeviceInput(owner.tensor.get(), request.devices()[index], owner.stream);
                if(collective_owner) {
                    owner.received=std::make_shared<FP32Tensor>(std::vector<size_t>{1,request.gatherStorageBytes()/sizeof(float)});
                    std::memset(owner.received->mutable_typed_data(),0xa5,request.gatherStorageBytes());
                    TransferEngine::prepareDeviceInput(owner.received.get(),request.devices()[index],owner.stream);
                    TransferEngine::prepareDeviceOutput(owner.received.get(),request.devices()[index],owner.stream);
                }
            });
            if (collective_owner)
            {
                std::vector<void *> streams;
                for (const auto &owner : participants) streams.push_back(owner->stream);
                collective_owner->setComputeStreams(streams);
                counted=DeviceCountedAllGather::create(*memory,*collective_owner,DeviceId::cpu(),*counted_topology,streams,request.activationBytes());
                const std::array phase_rows{1, request.prefillRows()};
                for (size_t phase_index = 0; phase_index < 3 * phase_rows.size(); ++phase_index)
                {
                    const bool gather=phase_index>=phase_rows.size();
                    const bool counted_phase=phase_index>=2*phase_rows.size();
                    const int rows=phase_rows[phase_index%phase_rows.size()];
                    const size_t payload=gather ? (phase_index%2 ? request.activationBytes() : counted_phase ? 0 : 1) : request.payloadBytes(rows);
                    PlanningLocalTPPhaseObservation phase{rows,payload,{}};
                    phase.endpoints.resize(request.devices().size());
                    // Every participant finishes allocation/module preparation
                    // before any worker enters native collective or marker capture.
                    onWorkers(request, collective_owner.get(), [&](size_t index, auto &worker) {
                        auto &owner=*participants[index];
                        // A previous diagnostic replay may publish fresh device
                        // source bytes. Join that original producer before the
                        // next capture; binding storage alone cannot authenticate
                        // a newly published event on the consumer stream.
                        TransferEngine::prepareDeviceInput(owner.tensor.get(),request.devices()[index],owner.stream);
                        TransferEngine::requireDeviceInput(owner.tensor.get(),request.devices()[index],owner.stream);
                        if(gather)TransferEngine::prepareDeviceOutput(owner.received.get(),request.devices()[index],owner.stream);
                        owner.collective_clock=std::make_unique<PlanningNativeGPUClockBuffer>(worker,
                            request.devices()[index],ExplicitGPUStream(owner.stream),2,memory);
                        if(counted_phase) {
                            counted->initializeLocalExtent(index,payload,owner.stream);
                            owner.counted_binding=counted->bind(index,{static_cast<uint64_t>(phase_index+1),request.activationBytes()},owner.tensor,owner.received);
                        }
                        owner.graph=worker.createGraphCapture(owner.stream);
                        require(bool(owner.graph),"graph creation failed");
                    });
                    onWorkers(request, collective_owner.get(), [&](size_t index, auto &worker) {
                        auto &owner=*participants[index];auto &graph=owner.graph;
                        ScopedBackendGraphCapture capture(worker,*graph,"planning native LocalTP");
                        require(capture.begin(),"capture begin failed");
                        if(counted_phase)owner.counted_binding.enqueue(owner.stream);
                        else if(gather) {
                            require(collective_owner->allgatherRawOnStream(owner.tensor->gpu_data_ptr(),owner.received->gpu_data_ptr(),
                                payload,CollectiveDataType::INT8,static_cast<int>(index),owner.stream,"planning_fixed_allgather"),
                                "captured native fixed allgather failed");
                            TransferEngine::publishDeviceWrite(owner.received.get(),request.devices()[index],owner.stream);
                        } else require(collective_owner->allreduceOnStream(owner.tensor.get(),"planning_allreduce",
                            size_t(rows)*request.hiddenWidth(),owner.stream,
                            request.precision()==PlanningAllreducePrecision::FP32 ? "fp32" : "fp16"),
                            "captured native reduction failed");
                        capture.finish();
                        const PlanningMeasurementWork work(PlanningWorkUnit::Bytes,
                            gather?(double(payload)+(counted_phase?sizeof(uint64_t):0))*degree:double(payload),
                            std::string(counted_phase?"native counted allgather":gather?"native fixed allgather":"native rank-local allreduce")+
                            "; backend="+std::string(collectiveBackendTypeToString(request.backend()))+
                            "; device="+request.devices()[index].toString()+"; whole invocation; payload per owner="+std::to_string(payload)+
                            (counted_phase?"; required extent metadata per owner=8 bytes":""));
                        owner.collective_timing.emplace(PlanningExecutionMeasurement::recordGPU(work,worker,
                            std::move(owner.collective_clock),std::move(graph)));
                    });
                    // End every body/marker capture before any runtime storage
                    // materialization. Preparation executes no collective body.
                    onWorkers(request, collective_owner.get(), [&](size_t index, auto &) {
                        auto &owner=*participants[index];owner.collective_timing->prepare(memory);
                        const auto growth=owner.collective_timing->residentMemoryBytes();
                        if(growth>std::numeric_limits<size_t>::max()-owner.graph_growth)
                            throw std::overflow_error("Planning native LocalTP graph-family extent overflow");
                        owner.graph_growth+=growth;
                    });
                    if(gather && validation==PlanningMeasurementValidation::ObservePayloads) {
                        // Diagnostic-only late publication defeats capture-time
                        // eager copies. Native production timing never downloads
                        // payloads or performs these oracle resets.
                        onWorkers(request,collective_owner.get(),[&](size_t index,auto &worker) {
                            auto &owner=*participants[index];const auto device=request.devices()[index];
                            auto *backend=getBackendFor(device);
                            require(backend && backend->memset(owner.tensor->gpu_data_ptr(),int((index%251)+1),
                                request.storageBytes(),device.ordinal,owner.stream) &&
                                backend->memset(owner.received->gpu_data_ptr(),0xa5,request.gatherStorageBytes(),device.ordinal,owner.stream),
                                "native gather late oracle publication failed");
                            TransferEngine::publishDeviceWrite(owner.tensor.get(),device,owner.stream);
                            TransferEngine::publishDeviceWrite(owner.received.get(),device,owner.stream);
                            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                            while(worker.queryStreamExecutionState(owner.stream,"planning gather oracle publication")!=GPUStreamExecutionState::Complete) {
                                if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Native gather oracle publication timeout");
                                std::this_thread::yield();
                            }
                        });
                    }
                    // Submit all parents concurrently before joining any worker:
                    // the same four native allreduces must make collective progress.
                    onWorkers(request, collective_owner.get(), [&](size_t index, auto &) {
                        auto &timing=*participants[index]->collective_timing;
                        const auto observed=timing.observe();
                        phase.endpoints[index]={request.devices()[index],timing.graphNodeCount(),
                            observed.elapsedSeconds()/PlanningExecutionMeasurement::kTimedInvocations};
                    });
                    if(gather && validation==PlanningMeasurementValidation::ObservePayloads)
                        onWorkers(request,collective_owner.get(),[&](size_t index,auto &) {
                            auto &owner=*participants[index];
                            require(owner.received->ensureOnHost(owner.stream),"native gather oracle readback failed");
                            const auto *bytes=reinterpret_cast<const uint8_t*>(owner.received->data());
                            const size_t sources=counted_phase?degree-1:degree;
                            const size_t live=sources*payload;
                            for(size_t offset=0;offset<request.gatherStorageBytes();++offset) {
                                uint8_t expected=0xa5;
                                if(offset<live) {
                                    size_t root=offset/payload;
                                    if(counted_phase && root>=index)++root;
                                    expected=static_cast<uint8_t>((root%251)+1);
                                }
                                require(bytes[offset]==expected,"native gather changed source order, empty payload or guard bytes");
                            }
                        });
                    onWorkers(request, collective_owner.get(), [&](size_t index, auto &) {
                        participants[index]->collective_timing.reset();
                        participants[index]->counted_binding={};
                    });
                    if(gather) {
                        PlanningLocalTPGatherPhaseObservation observed{counted_phase?PlanningNativeGatherProtocol::CountedAllgather:
                            PlanningNativeGatherProtocol::FixedAllgather,payload,degree,std::move(phase.endpoints)};
                        (void)observed.secondsPerCollective();result.gather_phases.push_back(std::move(observed));
                    } else {
                        (void)phase.secondsPerCollective();result.phases.push_back(std::move(phase));
                    }
                }
                for (size_t index = 0; index < participants.size(); ++index)
                {
                    const auto device = request.devices()[index];
                    const size_t admitted = 3 * phase_rows.size() * PlanningExecutionMeasurement::kPreparedGraphCount *
                        GPUGraphMemoryContract::reservationBytesPerExecutable(device);
                    const size_t observed = participants[index]->graph_growth;
                    require(GPUGraphMemoryContract::acceptsFamilyObservation(device, observed, admitted),
                        "native graph family exceeded its physical admission: device=" + device.toString() +
                        " observed_pool_growth_bytes=" + std::to_string(observed) +
                        " admitted_family_bytes=" + std::to_string(admitted) +
                        " hidden_width=" + std::to_string(request.hiddenWidth()) +
                        " prefill_rows=" + std::to_string(request.prefillRows()) +
                        " precision=" + (request.precision() == PlanningAllreducePrecision::FP32 ? "fp32" : "fp16"));
                }
            } // Native collective protocol observations are absent in physical-only mode.
            if (pairs == PlanningPairSampling::Enabled)
            {
                auto &matrix = result.pair_sampling;
                matrix.payload_anchors = request.pairPayloadAnchors();
                matrix.direct_peer_access.assign(direct_peer_access.begin(), direct_peer_access.end());
                const auto regions = request.pairPayloadRegions();
                const size_t windows = regions.size() * PlanningLocalTPPairSampling::kSamples;
                using Clock = std::chrono::steady_clock;
                auto started = Clock::now();
                onWorkers(request, collective_owner.get(), [&](size_t index, auto &worker) {
                    auto &owner = *participants[index];
                    owner.graph.reset();
                    bool receives_direct = false;
                    for (size_t source = 0; source < degree; ++source)
                        receives_direct |= direct_peer_access[source * degree + index] != 0;
                    if (!receives_direct) return;
                    owner.pair_clock = std::make_unique<PlanningNativeGPUClockBuffer>(worker,
                        request.devices()[index], ExplicitGPUStream(owner.stream), 2 * windows, memory);
                    owner.pair_timer = std::make_unique<PlanningCapturedGPUClockSampler>(worker,
                        ExplicitGPUStream(owner.stream), *owner.pair_clock);
                });
                matrix.setup_wall_seconds += std::chrono::duration<double>(Clock::now() - started).count();
                for (size_t source = 0; source < degree; ++source)
                    for (size_t destination = 0; destination < degree; ++destination)
                        if (direct_peer_access[source * degree + destination])
                        {
                            const std::array active{source, destination};
                            const std::array receiver{destination};
                            started = Clock::now();
                            onWorkers(request, collective_owner.get(), [&](size_t index, auto &) {
                                auto &owner = *participants[index];
                                if (validation == PlanningMeasurementValidation::ObservePayloads)
                                {
                                    auto *backend = getBackendFor(request.devices()[index]);
                                    require(backend && backend->memset(owner.tensor->gpu_data_ptr(), index == source ? 0x39 : 0xa5,
                                        request.storageBytes(), request.devices()[index].ordinal, owner.stream),
                                        "physical pair seed publication failed");
                                }
                                // Publish each prepared source's completed original stream.
                                // The destination copy capability imports this exact event.
                                TransferEngine::publishDeviceWrite(owner.tensor.get(), request.devices()[index], owner.stream);
                            }, active);
                            size_t graph_nodes = 0;
                            onWorkers(request, collective_owner.get(), [&](size_t index, auto &worker) {
                                auto &owner = *participants[index];
                                auto &graph = owner.graph;
                                graph.reset();
                                owner.pair_copy.reset();
                                owner.pair_copy = std::make_unique<PreparedDevicePeerCopy>(
                                    TransferEngine::instance().prepareDevicePeerCopy(
                                        participants[source]->tensor, owner.tensor, request.devices()[source],
                                        request.devices()[destination], ExplicitGPUStream(owner.stream)));
                                graph = worker.createGraphCapture(owner.stream);
                                require(bool(graph), "physical pair graph creation failed");
                                ScopedBackendGraphCapture capture(worker, *graph, "planning physical peer-copy windows");
                                require(capture.begin(), "physical pair recording failed");
                                for (size_t anchor = 0; anchor < regions.size(); ++anchor)
                                {
                                    const auto &region = regions[anchor];
                                    for (size_t sample = 0; sample < PlanningLocalTPPairSampling::kSamples; ++sample)
                                    {
                                        const size_t window = anchor * PlanningLocalTPPairSampling::kSamples + sample;
                                        owner.pair_timer->recordStart(window);
                                        for (size_t transfer = 0; transfer < PlanningLocalTPPairSampling::kBatchTransfers; ++transfer)
                                            owner.pair_copy->enqueue(region.byte_offset, region.byte_offset, region.live_bytes);
                                        owner.pair_timer->recordStop(window);
                                    }
                                }
                                capture.finish();
                                // Every window owns two markers and the exact copy
                                // count. Some native peer APIs execute during capture
                                // without recording a node; positive timestamps and
                                // a payload copied during setup cannot certify them.
                                require(graph->nodeCount() >= windows * (2 + PlanningLocalTPPairSampling::kBatchTransfers),
                                    "physical pair capture omitted live copy operations");
                                require(graph->prepareRuntimeContextStorage(memory) && graph->instantiate(),
                                    "physical pair executable materialization failed");
                                require(GPUGraphMemoryContract::acceptsFamilyObservation(request.devices()[index],
                                    graph->residentMemoryBytes(),
                                    (mode == PlanningLocalTPMeasurementMode::NativeCollective ? PlanningExecutionMeasurement::kPreparedGraphCount : 1) *
                                        GPUGraphMemoryContract::reservationBytesPerExecutable(request.devices()[index])),
                                    "physical pair executable exceeded admitted graph family");
                                owner.pair_timer->seal(*graph);
                                graph_nodes = graph->nodeCount();
                                if (validation == PlanningMeasurementValidation::ObservePayloads)
                                {
                                    // Poison after recording: an eager capture-time
                                    // copy must not masquerade as executable work.
                                    auto *backend = getBackendFor(request.devices()[index]);
                                    require(backend && backend->memset(owner.tensor->gpu_data_ptr(), 0xa5,
                                        request.storageBytes(), request.devices()[index].ordinal, owner.stream),
                                        "physical pair post-capture guard publication failed");
                                    TransferEngine::publishDeviceWrite(owner.tensor.get(), request.devices()[index], owner.stream);
                                }
                            }, receiver);
                            matrix.setup_wall_seconds += std::chrono::duration<double>(Clock::now() - started).count();
                            ++matrix.capture_count;
                            std::vector<PlanningServiceObservation> observed;
                            started = Clock::now();
                            onWorkers(request, collective_owner.get(), [&](size_t index, auto &) {
                                auto &owner = *participants[index];
                                std::vector<PlanningMeasurementWork> work;
                                work.reserve(windows);
                                for (const auto &region : regions)
                                    for (size_t sample = 0; sample < PlanningLocalTPPairSampling::kSamples; ++sample)
                                        work.emplace_back(PlanningWorkUnit::Bytes,
                                            double(region.live_bytes) * PlanningLocalTPPairSampling::kBatchTransfers,
                                            "prepared physical peer copy; destination fixed-clock windows; not native collective protocol latency");
                                observed = owner.pair_timer->samples(work, *owner.graph);
                            }, receiver);
                            matrix.measurement_wall_seconds += std::chrono::duration<double>(Clock::now() - started).count();
                            ++matrix.measured_launches;
                            for (size_t anchor = 0; anchor < regions.size(); ++anchor)
                            {
                                PlanningLocalTPPairObservation pair{source, destination, regions[anchor].live_bytes, {}, graph_nodes};
                                for (size_t sample = 0; sample < PlanningLocalTPPairSampling::kSamples; ++sample)
                                {
                                    const size_t window = anchor * PlanningLocalTPPairSampling::kSamples + sample;
                                    pair.seconds_per_transfer.push_back(observed.at(window).elapsedSeconds() /
                                        PlanningLocalTPPairSampling::kBatchTransfers);
                                }
                                matrix.directed_pairs.push_back(std::move(pair));
                            }
                            if (validation == PlanningMeasurementValidation::ObservePayloads)
                            {
                                started = Clock::now();
                                onWorkers(request, collective_owner.get(), [&](size_t index, auto &) {
                                    auto &owner = *participants[index];
                                    TransferEngine::publishDeviceWrite(owner.tensor.get(), request.devices()[index], owner.stream);
                                    require(owner.tensor->ensureOnHost(owner.stream), "physical pair byte observation failed");
                                    const auto *data = reinterpret_cast<const unsigned char *>(owner.tensor->data());
                                    for (const auto &region : regions)
                                        for (size_t offset = 0; offset < region.live_bytes + region.guard_bytes; ++offset)
                                            require(data[region.byte_offset + offset] ==
                                                (index == source || offset < region.live_bytes ? 0x39 : 0xa5),
                                                "physical pair changed original source or an inactive anchor guard");
                                }, active);
                                matrix.setup_wall_seconds += std::chrono::duration<double>(Clock::now() - started).count();
                            }
                        }
                matrix.validate(request);
            }
        }
        catch (...) { failure = std::current_exception(); if (collective_owner) collective_owner->requestAbort(); }
        // The communicator stays alive while every worker revokes its captured
        // borrow and destroys its tensor. Only then may LocalTP release scratch.
        // Retire cross-participant copy borrows before any tensor's separate
        // original allocation claim. Destruction order within one participant
        // cannot establish this all-participant lifetime boundary.
        onWorkers(request, nullptr, [&](size_t index, auto &) {
            participants[index]->collective_timing.reset();
            participants[index]->counted_binding={};
            participants[index]->collective_clock.reset();
            participants[index]->graph.reset();
            participants[index]->pair_copy.reset();
        });
        counted.reset();
        onWorkers(request, nullptr, [&](size_t index, auto &) { participants[index].reset(); });
        if (failure) std::rethrow_exception(failure);
        return result;
    }
}
