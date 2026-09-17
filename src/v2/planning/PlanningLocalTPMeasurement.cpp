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
 */
#include "PlanningLocalTPMeasurement.h"
#include "PlanningExecutionMeasurement.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <set>
#include <stdexcept>

namespace llaminar2
{
    PlanningLocalTPRequest::PlanningLocalTPRequest(std::vector<DeviceId> devices,
        int hidden_width, int prefill_rows, PlanningAllreducePrecision precision)
        : devices_(std::move(devices)), hidden_width_(hidden_width),
          prefill_rows_(prefill_rows), precision_(precision)
    {
        if (devices_.size() < 2 || hidden_width <= 0 || prefill_rows <= 0 ||
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

    namespace
    {
        /** @brief Fail at the boundary that lost its required native operation. */
        void require(bool condition, const char *detail)
        {
            if (!condition) throw std::runtime_error(std::string("Planning native LocalTP: ") + detail);
        }

        /**
         * @brief Submit every endpoint before joining; retain the first failure after all jobs retire.
         *
         * An asymmetric preparation/capture failure cancels LocalTP's host
         * rendezvous so peers cannot wait forever for a missing participant.
         * requestAbort deliberately does not destroy native graph borrowers.
         */
        void onWorkers(const PlanningLocalTPRequest &request, LocalTPContext *collective,
            const std::function<void(size_t, IWorkerGPUContext &)> &operation)
        {
            std::vector<std::future<void>> jobs;
            // Allocate bookkeeping before submission: a throwing push must not
            // lose the only join handle to an already running GPU operation.
            jobs.reserve(request.devices().size());
            std::exception_ptr failure;
            try
            {
                for (size_t index = 0; index < request.devices().size(); ++index)
                {
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
            PhysicalMemoryAllocationLease host_claim, device_claim;
            PhysicalMemoryOwnerReservation graphs_claim;
            std::unique_ptr<FP32Tensor> tensor;
            std::unique_ptr<IGPUGraphCapture> graph;
            size_t graph_growth = 0;
            void *stream = nullptr;

            /** @brief Claim the complete sample family before creating any GPU payload. */
            Participant(const PlanningLocalTPRequest &request, DeviceId device,
                const std::shared_ptr<PhysicalMemoryAuthority> &memory)
                : host_claim(memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                      request.activationBytes())),
                  device_claim(memory->claimNewAllocation(device, PhysicalMemoryOwner::ExecutionWorkspace,
                      request.activationBytes())),
                  graphs_claim(memory->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable,
                      2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device))) {}
        };
    }

    void PlanningLocalTPMeasurement::contributeMemory(const PlanningLocalTPRequest &request,
        PhysicalMemoryResource host, std::span<const PhysicalMemoryResource> devices,
        PhysicalMemoryPlanBuilder &builder)
    {
        if (host.device != DeviceId::cpu() || devices.size() != request.devices().size())
            throw std::invalid_argument("Planning native LocalTP requires same-rank CPU staging and every GPU resource");
        for (size_t index = 0; index < devices.size(); ++index)
            if (devices[index].world_rank != host.world_rank || devices[index].device != request.devices()[index])
                throw std::invalid_argument("Planning native LocalTP cannot use a different rank or device's resource");
        const auto local = CollectiveMemoryEstimator::localTP(request.prefillRows(), request.hiddenWidth(), request.backend());
        for (const auto resource : devices)
        {
            builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, request.activationBytes());
            builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, request.activationBytes());
            builder.add(resource, PhysicalMemoryOwner::LocalCollective, local.perDeviceBytes());
            builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable,
                2 * GPUGraphMemoryContract::reservationBytesPerExecutable(resource.device));
        }
    }

    PlanningLocalTPObservations PlanningLocalTPMeasurement::measure(const PlanningLocalTPRequest &request,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!memory) throw std::invalid_argument("Planning native LocalTP requires admitted physical memory");
        for (const auto device : request.devices())
            if (GPUDeviceContextPool::instance().getContext(device).ownsCurrentThread())
                throw std::invalid_argument("Planning native LocalTP must be driven by the rank setup owner, not a participant worker");
        // Claim every endpoint before native initialization. A capacity failure
        // therefore cannot strand another GPU inside collective capture.
        // Declaration order is also the last-resort exception-unwind order:
        // participant graph owners always die before the native communicator.
        std::unique_ptr<LocalTPContext> collective_owner;
        std::vector<std::unique_ptr<Participant>> participants;
        std::vector<GlobalDeviceAddress> addresses;
        for (const auto device : request.devices())
        {
            participants.push_back(std::make_unique<Participant>(request, device, memory));
            addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
        }
        collective_owner = std::make_unique<LocalTPContext>(std::move(addresses), std::vector<float>{}, request.backend());
        auto &collective = *collective_owner;
        PlanningLocalTPObservations result{request, {}};
        std::exception_ptr failure;
        try
        {
            const auto local = CollectiveMemoryEstimator::localTP(request.prefillRows(), request.hiddenWidth(), request.backend());
            require(collective.reserveCollectiveResources(local.backend_payload_capacity_bytes,
                local.fp16_scratch_elements, memory), "collective resource admission failed");
            onWorkers(request, &collective, [&](size_t index, auto &worker) {
                auto &owner = *participants[index];
                owner.stream = ExplicitGPUStream(worker.defaultStream()).get();
                owner.tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(request.prefillRows()), size_t(request.hiddenWidth())});
                // Zero is stable under repeated in-place sums. The functional
                // integration oracle separately checks non-zero production sums.
                std::fill_n(owner.tensor->mutable_typed_data(), owner.tensor->numel(), 0.0f);
                TransferEngine::prepareDeviceInput(owner.tensor.get(), request.devices()[index], owner.stream);
                TransferEngine::requireDeviceInput(owner.tensor.get(), request.devices()[index], owner.stream);
            });
            std::vector<void *> streams;
            for (const auto &owner : participants) streams.push_back(owner->stream);
            collective.setComputeStreams(streams);
            for (const int rows : {1, request.prefillRows()})
            {
                PlanningLocalTPPhaseObservation phase{rows, request.payloadBytes(rows), {}};
                phase.endpoints.resize(request.devices().size());
                onWorkers(request, &collective, [&](size_t index, auto &worker) {
                    auto &owner = *participants[index];
                    owner.graph.reset();
                    owner.graph = worker.createGraphCapture(owner.stream);
                    require(bool(owner.graph), "graph creation failed");
                    ScopedBackendGraphCapture capture(worker, *owner.graph, "planning native LocalTP");
                    require(capture.begin(), "capture begin failed");
                    require(collective.allreduceOnStream(owner.tensor.get(), "planning_allreduce",
                        size_t(rows) * request.hiddenWidth(), owner.stream,
                        request.precision() == PlanningAllreducePrecision::FP32 ? "fp32" : "fp16"),
                        "captured native reduction failed");
                    capture.finish();
                    require(owner.graph->nodeCount() && owner.graph->instantiate(), "nonempty native graph instantiation failed");
                    const size_t growth = owner.graph->residentMemoryBytes();
                    if (growth > std::numeric_limits<size_t>::max() - owner.graph_growth)
                        throw std::overflow_error("Planning native LocalTP graph-family extent overflow");
                    owner.graph_growth += growth;
                });
                // All captures are sealed before any replay. This is a setup
                // dependency, not a per-token host rendezvous in the measured graph.
                onWorkers(request, &collective, [&](size_t index, auto &) {
                    const auto device = request.devices()[index];
                    auto &owner = *participants[index];
                    auto *backend = getBackendFor(device);
                    require(backend != nullptr, "captured backend missing");
                    const PlanningMeasurementWork work(PlanningWorkUnit::Bytes, double(phase.payload_bytes),
                        "native rank-local allreduce; backend=" + std::string(collectiveBackendTypeToString(request.backend())) +
                        "; device=" + device.toString() + "; logical-payload-not-link-bandwidth; rows=" + std::to_string(rows));
                    const auto observed = PlanningExecutionMeasurement::gpu(work, *backend, device, *owner.graph);
                    phase.endpoints[index] = {device, owner.graph->nodeCount(),
                        observed.elapsedSeconds() / PlanningExecutionMeasurement::kTimedInvocations};
                });
                (void)phase.secondsPerCollective();
                result.phases.push_back(std::move(phase));
            }
            for (size_t index = 0; index < participants.size(); ++index)
                require(GPUGraphMemoryContract::acceptsFamilyObservation(request.devices()[index],
                    participants[index]->graph_growth,
                    2 * GPUGraphMemoryContract::reservationBytesPerExecutable(request.devices()[index])),
                    "native graph family exceeded its physical admission");
        }
        catch (...) { failure = std::current_exception(); collective.requestAbort(); }
        // The communicator stays alive while every worker revokes its captured
        // borrow and destroys its tensor. Only then may LocalTP release scratch.
        onWorkers(request, nullptr, [&](size_t index, auto &) { participants[index].reset(); });
        if (failure) std::rethrow_exception(failure);
        return result;
    }
}
