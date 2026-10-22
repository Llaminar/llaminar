/**
 * @file CapturedFixedRankGatherProbe.h
 * @brief Explicit native versus mapped fixed-message gather/fold component probe.
 *
 * The native control uses the original fully-active AllGather and original
 * full-bank fold. The opt-in component uses exactly P*(P-1) original retained
 * TransferEngine channels, complete send/receive batches and a local-aware
 * ascending-rank fold. It never selects a production transport or reads a device
 * cursor/count on the host. Component buffers, channel staging and native graph
 * reservations are PMA-admitted. The original native context separately owns
 * its vendor communicator allocations; this small BOM does not re-account them.
 * Event intervals include device waits; they are not model speedup predictions.
 */
#pragma once
#include "CapturedChannelGraphProof.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/TPRankOrderedReductionKernels.h"
#include "planning/PlanningObservedResource.h"
#include "transfer/CapturedTransferChannel.h"
#include "utils/MPIContext.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <utility>
#include <stdexcept>
#include <vector>

namespace llaminar2::fixed_rank_component_test
{
    using Json = nlohmann::json;

    /** @brief Two explicitly requested components; failure never selects the other. */
    enum class Path { Native, Mapped };

    /** @brief Reject incomplete setup, native enqueue, observation or exact bytes. */
    inline void require(bool value, const char *message)
    { if (!value) throw std::runtime_error(message); }

    /**
     * @brief Emit one bounded machine-readable evidence row, outside captured work.
     * @param row Authored geometry or observed native timing/graph metadata.
     */
    inline void emit(const Json &row)
    { std::printf("FIXED_RANK_COMPONENT %s\n", row.dump().c_str()); std::fflush(stdout); }

    /**
     * @brief One immutable captured family with original endpoint owners.
     *
     * Member order destroys the executable before bindings and its PMA reserve.
     * Each binding retains the actual channel and original device allocation.
     */
    struct Family final
    {
        Path path;
        std::size_t elements;
        int transactions;
        bool in_place;
        PhysicalMemoryOwnerReservation graph_claim;
        std::vector<CapturedTransferBinding> sends, receives;
        std::unique_ptr<IGPUGraphCapture> graph;
    };

    /**
     * @brief One original physical participant and all its retained allocations.
     *
     * Normal retirement joins every exact terminal first, clears all executable
     * families, destroys events, then releases bindings/buffers and their claims.
     * If a native error leaves an accepted launch unobserved, destroying these
     * owners would be unsafe. Fail the disposable diagnostic process immediately
     * instead; the root runner retains its ordinary owned process/driver bracket.
     * This is not a new cancellation or partial-enqueue recovery implementation.
     */
    struct Participant final
    {
        DeviceId device;
        IBackend *backend = nullptr;
        IWorkerGPUContext *worker = nullptr;
        void *stream = nullptr, *begin = nullptr, *end = nullptr;
        PhysicalMemoryAllocationLease frame_claim, host_claim;
        std::shared_ptr<DeviceTransferBuffer> frame;
        std::shared_ptr<MappedHostTransferRegion> initial, readback;
        std::vector<std::unique_ptr<Family>> families;
        bool unobserved = false;

        /** @brief Retire only quiescent original owners; never guess completion. */
        ~Participant()
        {
            if (unobserved)
            {
                std::fputs("FATAL fixed-rank probe has unobserved native work; refusing owner destruction\n", stderr);
                std::fflush(stderr);
                std::_Exit(EXIT_FAILURE);
            }
            families.clear();
            if (end) backend->destroyEvent(end, device.ordinal);
            if (begin) backend->destroyEvent(begin, device.ordinal);
        }
        /** @return Exact scalar address inside the original retained device allocation. */
        float *at(std::size_t element_offset) const
        { return static_cast<float *>(frame->mutableDeviceData(element_offset * sizeof(float))); }
    };

    /**
     * @brief Observe one exact terminal event using the standard 30-second bound.
     * @param p Original event/device owner, after all participant submissions.
     *
     * No stream/device synchronization is used. Successful observation alone
     * permits the enclosing owner to be released; a failed query is fatal.
     */
    inline void await(Participant &p)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        bool ready = false;
        while (!ready)
        {
            require(p.backend->queryEvent(p.end, p.device.ordinal, &ready), "Fixed-rank terminal query failed");
            require(ready || std::chrono::steady_clock::now() < deadline, "Fixed-rank terminal exceeded 30 seconds");
            if (!ready) std::this_thread::yield();
        }
        p.unobserved = false;
    }

    /** @return Explicit actual-backend original full-bank arithmetic, unchanged. */
    inline bool originalFold(Participant &p, const float *banks, float *output,
        std::size_t elements, int degree)
    {
#ifdef HAVE_CUDA
        if (p.device.is_cuda()) return launchCUDATPRankOrderedSumFP32(
            banks, output, elements, degree, p.device.ordinal, p.stream);
#endif
#ifdef HAVE_ROCM
        if (p.device.is_rocm()) return launchROCmTPRankOrderedSumFP32(
            banks, output, elements, degree, p.device.ordinal, p.stream);
#endif
        return false;
    }

    /** @return Explicit actual-backend local-aware arithmetic, never a transport fallback. */
    inline bool localFold(Participant &p, const TPRankOrderedLocalSumFP32 &launch)
    {
#ifdef HAVE_CUDA
        if (p.device.is_cuda()) return launchCUDATPRankOrderedLocalSumFP32(launch, p.device.ordinal, p.stream);
#endif
#ifdef HAVE_ROCM
        if (p.device.is_rocm()) return launchROCmTPRankOrderedLocalSumFP32(launch, p.device.ordinal, p.stream);
#endif
        return false;
    }

    /**
     * @brief Author exact FP32 inputs including cancellation, signed zero and nonfinite values.
     * @param rank Original communicator position, independent of device ordinal.
     * @param column Original scalar position inside the source, not a padding slot.
     * @return Raw source bits; no quantization or floating arithmetic constructs NaNs.
     */
    inline std::uint32_t sourceWord(int rank, std::size_t column)
    {
        switch (column % 31)
        {
        case 0: return 0x80000000u; // All-negative-zero distinguishes an extra initial +0.
        case 1: return rank == 0 ? 0x80000000u : 0u;
        case 2: return rank == 0 ? 0u : 0x80000000u;
        case 3: if (rank == 0) return 0x7f800000u; break;
        case 4: if (rank == 0) return 0xff800000u; break;
        case 5: if (rank == 0) return 0x7fc13579u; break;
        case 6: if (rank < 2) return rank == 0 ? 0x7f800000u : 0xff800000u; break;
        }
        const float value = rank == 0 ? 16777216.0f : rank == 1 ? 1.0f
            : rank == 2 ? -16777216.0f : static_cast<float>(1 + column % 7);
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /**
     * @brief Inspect the actual complete executable, not a flattened sum of unrelated captures.
     * @param p Original device/stream owner.
     * @param family One exact retained graph with immutable path/extent/transaction count.
     * @param degree Original physical membership size.
     */
    inline void inspect(const Participant &p, const Family &family, int degree)
    {
        std::vector<GPUGraphKernelNodeInfo> kernels;
        std::string error;
        require(family.graph->inspectKernelNodes(kernels, &error), "Fixed-rank kernel inspection failed");
        if (family.path == Path::Mapped)
            counted_channel_test::verifyCapturedChannelBatchGraph(*family.graph,
                family.elements * sizeof(float), degree - 1, 2 * family.transactions);
        std::size_t folds = 0, native = 0, transfers = 0;
        std::map<std::string, Json> inventory;
        for (const auto &kernel : kernels)
        {
            require(kernel.valid() && kernel.name_resolved, "Fixed-rank native kernel identity is incomplete");
            if (kernel.name.find("capturedTransfer") != std::string::npos) ++transfers;
            if (kernel.name.find("rankOrdered") != std::string::npos)
            {
                const char *expected = family.path == Path::Mapped
                    ? "rankOrderedLocalSumFP32Kernel" : "rankOrderedSumFP32Kernel";
                require(kernel.name.find(expected) != std::string::npos, "Fixed-rank graph used wrong arithmetic leaf");
                require(kernel.grid_x == std::min<std::size_t>(65535, 1 + (family.elements - 1) / 256) &&
                    kernel.grid_y == 1 && kernel.grid_z == 1 && kernel.block_x == 256 &&
                    kernel.block_y == 1 && kernel.block_z == 1, "Fixed-rank fold geometry changed");
                ++folds;
            }
            else if (kernel.name.find("capturedTransfer") == std::string::npos)
            {
                require(family.path == Path::Native &&
                    (kernel.name.find("nccl") != std::string::npos || kernel.name.find("rccl") != std::string::npos),
                    "Fixed-rank graph contains unrelated kernels");
                ++native;
            }
            const auto key = kernel.name + ":" + std::to_string(kernel.grid_x) + ":" + std::to_string(kernel.grid_y);
            auto &item = inventory[key];
            if (item.is_null()) item = {{"name", kernel.name}, {"grid", {kernel.grid_x, kernel.grid_y, kernel.grid_z}},
                {"block", {kernel.block_x, kernel.block_y, kernel.block_z}}, {"count", 0},
                {"registers_per_thread", kernel.registers_per_thread},
                {"static_shared_memory_bytes", kernel.static_shared_memory_bytes},
                {"dynamic_shared_memory_bytes", kernel.dynamic_shared_memory_bytes},
                {"local_memory_bytes_per_thread", kernel.local_memory_bytes_per_thread},
                {"max_threads_per_block", kernel.max_threads_per_block},
                {"theoretical_max_active_blocks_per_sm", kernel.max_active_blocks_per_sm}};
            item["count"] = item["count"].get<std::size_t>() + 1;
        }
        require(folds == static_cast<std::size_t>(family.transactions) &&
            (family.path == Path::Mapped ? native == 0 : native >= static_cast<std::size_t>(family.transactions)),
            "Fixed-rank graph omitted or duplicated its original complete arithmetic/collective calls");
        emit({{"kind", "capture"}, {"device", p.device.toString()},
            {"path", family.path == Path::Mapped ? "mapped_fixed_local_fold" : "native_fixed_original_fold"},
            {"elements", family.elements}, {"transactions", family.transactions},
            {"in_place", family.in_place}, {"kernels", inventory},
            {"transfer_nodes", transfers},
            {"transfer_nodes_per_transaction", transfers / static_cast<std::size_t>(family.transactions)},
            {"transfer_lowering", family.path == Path::Native ? "native_fixed" :
                capturedTransferBatchKernelPlan(family.elements * sizeof(float)) == CapturedTransferKernelPlan::SingleBlock
                    ? "batched_fused" : "batched_acquire_copy_publish"},
            {"observation", "actual_complete_captured_template_not_replay_counter"}});
    }

    /** @brief Immutable test-authored graph geometry, without a live count or runtime mode. */
    struct ProbeShape final
    {
        std::size_t elements; ///< Positive whole FP32 message extent.
        int transactions; ///< Original captured repetitions, not host graph launches.
        bool in_place; ///< Exact local/output alias; only one transaction is admitted.
    };

    /**
     * @brief Run one explicit homogeneous communicator, preserving original rank order.
     * @param devices Exact ordered physical membership; permutations are intentional.
     * @param timing Whether to run 31 paired samples for exact H2560 counts 1/97.
     * @param selected_shapes Optional bounded functional-only boundary family; timing stays fixed.
     *
     * Both paths use the same original frame pointers and immutable FP32 input
     * bytes. Functional families include retained large->small->large extents,
     * tails and exact in-place output. Timing uses a separate output so repeated
     * transactions consume the same immutable partials in both paths. This
     * isolates gather+fold component cost, not a complete model layer schedule.
     */
    inline void probe(const std::vector<DeviceId> &devices, bool timing,
        std::span<const ProbeShape> selected_shapes = {})
    {
        constexpr std::size_t guard = 16;
        const std::size_t maximum = timing ? 2560 : 8193;
        // Source/output starts and every functional peer slot are 128-byte
        // aligned relative to the backend-aligned base. The H2560 benchmark
        // uses identical packed bank strides for native and mapped paths.
        const std::size_t stride = (maximum + guard + 31u) & ~std::size_t{31u};
        const std::size_t peer_stride = timing ? maximum : stride;
        constexpr std::uint32_t poison = 0xa5a5a5a5u;
        const int degree = static_cast<int>(devices.size());
        require(degree >= 2 && degree <= 4, "Fixed-rank probe supports explicit P2/P4 only");
        const auto first = devices.front();
        for (const auto device : devices)
            require(device.is_gpu() && device.type == first.type &&
                std::count(devices.begin(), devices.end(), device) == 1, "Fixed-rank membership must be homogeneous and unique");
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        require(bool(inventory) && mpi->world_size() == 1, "Fixed-rank probe requires original one-rank inventory");
        const auto &rank = inventory->ranks.at(mpi->rank());
        const std::size_t bank_offset = stride, output_offset = bank_offset + degree * peer_stride;
        const std::size_t frame_elements = output_offset + stride;
        const std::size_t frame_bytes = frame_elements * sizeof(float);
        const auto channel_geometry = CapturedTransferChannel::memoryFor(maximum * sizeof(float));
        const auto mapped_bytes = TransferEngine::mappedHostRegionAllocationBytes(frame_bytes);
        std::vector<ProbeShape> shapes;
        if (timing)
        {
            require(selected_shapes.empty(), "Timing retains only exact original H2560 shapes");
            shapes = {{2560, 1, false}, {2560, 97, false}};
        }
        else if (selected_shapes.empty()) shapes = {{maximum, 1, false}, {257, 1, true}, {2560, 1, false}};
        else shapes.assign(selected_shapes.begin(), selected_shapes.end());
        require(!shapes.empty() && shapes.size() <= 8, "Functional boundary family must be nonempty and bounded");
        for (const auto shape : shapes)
            require(shape.elements > 0 && shape.elements <= maximum && shape.transactions > 0 &&
                (!shape.in_place || shape.transactions == 1), "Invalid original fixed-rank functional geometry");
        const int family_count = static_cast<int>(2 * shapes.size());
        PhysicalMemoryPlanBuilder builder;
        builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            degree * (degree - 1) * channel_geometry.mapped_host_bytes + 2 * degree * mapped_bytes);
        for (const auto device : devices)
        {
            const auto resource = planningObservedResource(rank, device);
            builder.add(resource, PhysicalMemoryOwner::ActivationTransportStaging,
                frame_bytes + 2 * (degree - 1) * channel_geometry.cursor_bytes_per_device);
            builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable,
                family_count * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        }
        emit({{"kind", "layout"}, {"members", degree}, {"timing", timing},
            {"maximum_elements", maximum}, {"source_output_capacity_elements", stride},
            {"peer_stride_elements", peer_stride}, {"bank_offset_bytes", bank_offset * sizeof(float)},
            {"output_offset_bytes", output_offset * sizeof(float)}, {"frame_bytes", frame_bytes},
            {"channel_capacity_bytes", maximum * sizeof(float)},
            {"native_communicator_storage", "separately_owned_by_original_context_not_reaccounted"}});
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        auto &transfer = TransferEngine::instance();
        {
            // Communicator declared first: every captured executable retires before it.
            std::vector<GlobalDeviceAddress> membership;
            for (const auto device : devices) membership.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
            auto native = createLocalTPContext(membership, {}, first.is_cuda()
                ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            require(bool(native) && native->supportsRawAllgatherOnStreamGraphCapture() &&
                native->devices() == membership, "Missing original ordered native gather control");
            std::vector<std::shared_ptr<CapturedTransferChannel>> channels;
            std::vector<std::unique_ptr<Participant>> participants;
            for (int index = 0; index < degree; ++index)
            {
                auto p = std::make_unique<Participant>();
                p->device = devices[index]; p->backend = getBackendFor(p->device);
                require(p->backend && p->backend->deviceCount() > p->device.ordinal, "Missing requested physical GPU");
                p->worker = &GPUDeviceContextPool::instance().getContext(p->device);
                p->worker->submitAndWait([&] {
                    p->stream = p->worker->getOrCreateAuxiliaryStream("fixed_rank_component");
                    require(tpRankOrderedLocalLaunchValid(p->device.ordinal, p->stream), "Missing exact fixed-rank stream");
                    p->frame_claim = authority->claimNewAllocation(p->device, PhysicalMemoryOwner::ActivationTransportStaging, frame_bytes);
                    p->host_claim = authority->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, 2 * mapped_bytes);
                    p->frame = transfer.allocateDeviceTransferBuffer(frame_bytes, p->device);
                    const std::array visible{p->device};
                    p->initial = transfer.allocateMappedHostRegion(frame_bytes, visible);
                    p->readback = transfer.allocateMappedHostRegion(frame_bytes, visible);
                    p->begin = p->backend->createTimingEvent(p->device.ordinal);
                    p->end = p->backend->createTimingEvent(p->device.ordinal);
                    require(p->begin && p->end, "Missing native timing events");
                    auto *words = static_cast<std::uint32_t *>(p->initial->mutableHostData());
                    std::fill(words, words + frame_elements, poison);
                    for (std::size_t c = 0; c < maximum; ++c) words[c] = sourceWord(index, c);
                });
                participants.push_back(std::move(p));
            }
            for (int source = 0; source < degree; ++source)
                for (int target = 0; target < degree; ++target)
                    if (source != target)
                        channels.push_back(transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
                            devices[source], participants[source]->stream, devices[target], participants[target]->stream,
                            maximum * sizeof(float)));
            for (std::size_t shape = 0; shape < shapes.size(); ++shape)
                for (const auto path : {Path::Native, Path::Mapped})
                    for (int index = 0; index < degree; ++index)
                    {
                        auto &p = *participants[index];
                        auto f = std::make_unique<Family>();
                        f->path = path; f->elements = shapes[shape].elements;
                        f->transactions = shapes[shape].transactions; f->in_place = shapes[shape].in_place;
                        f->graph_claim = authority->reserveNewAllocations(p.device, PhysicalMemoryOwner::NativeGraphExecutable,
                            GPUGraphMemoryContract::reservationBytesPerExecutable(p.device));
                        if (path == Path::Mapped)
                        {
                            std::size_t edge = 0;
                            for (int source = 0; source < degree; ++source)
                                for (int target = 0; target < degree; ++target)
                                    if (source != target)
                                    {
                                        const CapturedTransferMessage message{0x46524700u + 8 * shape + source,
                                            f->elements * sizeof(float)};
                                        if (source == index) f->sends.push_back(transfer.bindCapturedTransfer(channels[edge],
                                            CapturedTransferEndpoint::Producer, message, p.frame));
                                        if (target == index) f->receives.push_back(transfer.bindCapturedTransfer(channels[edge],
                                            CapturedTransferEndpoint::Consumer, message, p.frame,
                                            (bank_offset + source * peer_stride) * sizeof(float)));
                                        ++edge;
                                    }
                        }
                        p.families.push_back(std::move(f));
                    }
            for (int slot = 0; slot < family_count; ++slot)
            {
                std::barrier begun(degree), submitted(degree);
                std::vector<std::exception_ptr> begin_errors(degree), body_errors(degree), final_errors(degree);
                std::vector<std::thread> threads;
                threads.reserve(degree);
                try
                {
                    for (int index = 0; index < degree; ++index)
                        threads.emplace_back([&, index] {
                            bool entered_worker = false;
                            try
                            {
                                auto &p = *participants[index]; auto &f = *p.families[slot];
                                p.worker->submitAndWait([&] {
                                    entered_worker = true;
                                    std::unique_ptr<ScopedBackendGraphCapture> capture;
                                    try
                                    {
                                        p.families[slot]->graph = p.worker->createGraphCapture(p.stream);
                                        require(bool(f.graph), "Missing fixed-rank capture factory");
                                        capture = std::make_unique<ScopedBackendGraphCapture>(*p.worker, *f.graph, "fixed rank component");
                                        require(capture->begin(), "Fixed-rank capture begin failed");
                                    }
                                    catch (...) { begin_errors[index] = std::current_exception(); }
                                    begun.arrive_and_wait();
                                    if (std::none_of(begin_errors.begin(), begin_errors.end(), [](const auto &e) { return bool(e); }))
                                        try
                                        {
                                            const auto out = p.at(f.in_place ? 0 : output_offset);
                                            const TPRankOrderedLocalSumFP32 local({p.at(0), stride, index}, p.at(bank_offset),
                                                degree * peer_stride, peer_stride, out, stride, f.elements, degree);
                                            const NativeCollectiveRows rows(DeviceRowRange::fullyActive(1), f.elements);
                                            for (int transaction = 0; transaction < f.transactions; ++transaction)
                                            {
                                                if (f.path == Path::Native)
                                                {
                                                    require(native->nativeRowsOnStream(NativeRowCollective::AllGather,
                                                        p.at(0), p.at(bank_offset), rows, CollectiveDataType::FLOAT32,
                                                        CollectiveOp::ALLGATHER, index, p.stream, "fixed_rank_component_original"),
                                                        "Original fixed native gather failed");
                                                    require(originalFold(p, p.at(bank_offset), out, f.elements, degree), "Original rank fold failed");
                                                }
                                                else
                                                {
                                                    transfer.enqueueCapturedExchange(f.sends, f.receives,
                                                        CapturedTransferEndpointScheduling::Batched, p.stream);
                                                    require(localFold(p, local), "Local-aware rank fold failed");
                                                }
                                            }
                                        }
                                        catch (...) { body_errors[index] = std::current_exception(); }
                                    submitted.arrive_and_wait();
                                    const bool success = std::none_of(begin_errors.begin(), begin_errors.end(), [](const auto &e) { return bool(e); }) &&
                                        std::none_of(body_errors.begin(), body_errors.end(), [](const auto &e) { return bool(e); });
                                    if (capture && !begin_errors[index]) capture->finish();
                                    if (success)
                                    {
                                        inspect(p, f, degree);
                                        require(f.graph->instantiate(), "Fixed-rank graph instantiation failed");
                                    }
                                });
                            }
                            catch (...)
                            {
                                final_errors[index] = std::current_exception();
                                // A rejected host submission must not strand peers
                                // at capture-only barriers before any graph launch.
                                if (!entered_worker)
                                {
                                    // Publish failed admission before releasing the
                                    // barrier: no peer may record a collective for
                                    // a participant whose callback never entered.
                                    begin_errors[index] = final_errors[index];
                                    begun.arrive_and_drop();
                                    submitted.arrive_and_drop();
                                }
                            }
                        });
                }
                catch (...)
                {
                    // A host thread-creation failure is also failed admission.
                    // Fill every absent slot before the last dropped arrival
                    // can release other workers to inspect the error vector.
                    const auto error = std::current_exception();
                    for (std::size_t missing = threads.size(); missing < static_cast<std::size_t>(degree); ++missing)
                    {
                        begin_errors[missing] = error;
                        final_errors[missing] = error;
                    }
                    for (std::size_t missing = threads.size(); missing < static_cast<std::size_t>(degree); ++missing)
                    {
                        begun.arrive_and_drop();
                        submitted.arrive_and_drop();
                    }
                }
                for (auto &thread : threads) thread.join();
                for (const auto *errors : {&begin_errors, &body_errors, &final_errors})
                    for (const auto &error : *errors) if (error) std::rethrow_exception(error);
            }
            bool complete_family_admitted = true;
            for (const auto &p : participants)
            {
                std::size_t resident = 0;
                Json growth = Json::array();
                for (const auto &f : p->families)
                {
                    resident += f->graph->residentMemoryBytes();
                    growth.push_back({{"path", f->path == Path::Native ? "native_fixed_original_fold" : "mapped_fixed_local_fold"},
                        {"transactions", f->transactions}, {"pool_growth_bytes", f->graph->residentMemoryBytes()}});
                }
                emit({{"kind", "graph_memory"}, {"device", p->device.toString()}, {"families", growth},
                    {"observed_pool_growth_bytes", resident}, {"admitted_family_bytes",
                        family_count * GPUGraphMemoryContract::reservationBytesPerExecutable(p->device)}});
                complete_family_admitted &= GPUGraphMemoryContract::acceptsFamilyObservation(p->device, resident,
                    family_count * GPUGraphMemoryContract::reservationBytesPerExecutable(p->device));
            }
            require(complete_family_admitted, "Fixed-rank graph family exceeded BOM");
            /** The entire host frame is reset/uploaded outside every timing interval. */
            const auto initialize = [&] {
                for (auto &owner : participants)
                {
                    auto &p = *owner;
                    p.worker->submitAndWait([&] {
                        p.unobserved = true;
                        transfer.enqueueMappedHostToDevice(*p.initial, 0, *p.frame, 0, frame_bytes, p.device, p.stream);
                        require(p.backend->recordEvent(p.end, p.device.ordinal, p.stream), "Fixed-rank setup terminal failed");
                    });
                }
                for (auto &p : participants) await(*p);
            };
            /** All ranks submit before terminal observation, then full original frames are checked. */
            const auto run = [&](int slot, int first_rank, int sample,
                const std::vector<std::vector<std::uint32_t>> *reference) {
                initialize();
                const auto &f = *participants[0]->families[slot];
                std::vector<double> host_enqueue_begin(degree), host_enqueue_end(degree);
                const auto host_origin = std::chrono::steady_clock::now();
                for (int offset = 0; offset < degree; ++offset)
                {
                    const auto index = (first_rank + offset) % degree;
                    auto &p = *participants[index];
                    host_enqueue_begin[index] = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - host_origin).count();
                    p.worker->submitAndWait([&] {
                        p.unobserved = true;
                        require(p.backend->recordEvent(p.begin, p.device.ordinal, p.stream) && p.families[slot]->graph->launch() &&
                            p.backend->recordEvent(p.end, p.device.ordinal, p.stream), "Fixed-rank retained timed launch failed");
                    });
                    host_enqueue_end[index] = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - host_origin).count();
                }
                const auto host_launch_span = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - host_origin).count();
                for (auto &p : participants) await(*p);
                std::vector<std::vector<std::uint32_t>> outputs(degree);
                std::vector<double> elapsed(degree);
                for (int index = 0; index < degree; ++index)
                {
                    auto &p = *participants[index];
                    float milliseconds = 0;
                    require(p.backend->eventElapsedTimeMs(p.begin, p.end, p.device.ordinal, &milliseconds), "Fixed-rank native event timing failed");
                    require(std::isfinite(milliseconds) && milliseconds > 0, "Fixed-rank native event interval must be finite and positive");
                    elapsed[index] = 1000.0 * milliseconds;
                    p.worker->submitAndWait([&] {
                        p.unobserved = true;
                        transfer.enqueueDeviceToMappedHost(*p.frame, 0, *p.readback, 0, frame_bytes, p.device, p.stream);
                        require(p.backend->recordEvent(p.end, p.device.ordinal, p.stream), "Fixed-rank readback terminal failed");
                    });
                }
                for (auto &p : participants) await(*p);
                for (int index = 0; index < degree; ++index)
                {
                    const auto &p = *participants[index];
                    const auto *actual = static_cast<const std::uint32_t *>(p.readback->mutableHostData());
                    const auto *initial = static_cast<const std::uint32_t *>(p.initial->mutableHostData());
                    const auto result_offset = f.in_place ? 0 : output_offset;
                    outputs[index].assign(actual + result_offset, actual + result_offset + f.elements);
                    if (reference) require(outputs[index] == reference->at(index), "Local-aware fold differs from original full-bank arithmetic bits");
                    if (index) require(outputs[index] == outputs[0], "Original communicator fold differs between physical participants");
                    // Everything outside the exact output and exact published peer
                    // extents must retain its original bytes, including own-slot poison.
                    for (std::size_t element = 0; element < frame_elements; ++element)
                    {
                        if (element >= result_offset && element - result_offset < f.elements) continue;
                        std::uint32_t expected = initial[element];
                        if (element >= bank_offset && element < output_offset)
                        {
                            const auto relative = element - bank_offset;
                            const auto bank_stride = f.path == Path::Native ? f.elements : peer_stride;
                            const auto source = relative / bank_stride;
                            const auto column = relative % bank_stride;
                            if (source < static_cast<std::size_t>(degree) && column < f.elements &&
                                (f.path == Path::Native || source != static_cast<std::size_t>(index)))
                                expected = sourceWord(static_cast<int>(source), column);
                        }
                        require(actual[element] == expected, "Fixed-rank input/peer bank/own-slot/suffix guard changed");
                    }
                    emit({{"kind", "sample"}, {"sample", sample}, {"slot", slot}, {"participant", index},
                        {"device", p.device.toString()}, {"first_host_rank", first_rank},
                        {"path", f.path == Path::Mapped ? "mapped_fixed_local_fold" : "native_fixed_original_fold"},
                        {"elements", f.elements}, {"transactions", f.transactions}, {"in_place", f.in_place},
                        {"native_event_us", elapsed[index]}, {"amortized_us", elapsed[index] / f.transactions},
                        {"host_enqueue_begin_us", host_enqueue_begin[index]}, {"host_enqueue_end_us", host_enqueue_end[index]},
                        {"host_launch_span_us", host_launch_span},
                        {"logical_edge_bytes", f.elements * sizeof(float)}, {"directed_peer_edges", degree * (degree - 1)},
                        {"bank_stride_elements", f.path == Path::Native ? f.elements : peer_stride},
                        {"bank_offset_bytes", bank_offset * sizeof(float)}, {"output_offset_bytes", output_offset * sizeof(float)},
                        {"physical_link_bytes", nullptr}, {"full_frame_exact", true}, {"reference_exact", bool(reference)}});
                }
                return std::pair(std::move(outputs), *std::max_element(elapsed.begin(), elapsed.end()) / f.transactions);
            };
            for (int shape = 0; shape < static_cast<int>(shapes.size()); ++shape)
            {
                const int native_slot = 2 * shape, mapped_slot = native_slot + 1;
                const auto oracle = run(native_slot, 0, -100, nullptr).first;
                for (const auto &row : oracle)
                {
                    require(row.front() == 0x80000000u, "Original rank-zero initialization must preserve all-negative zero");
                    if (degree == 4) require(row[7] == 0x3f800000u, "Original ascending cancellation fold changed");
                }
                // The independently executed original full-bank graph supplies the
                // arithmetic oracle, including signed-zero/NaN bits, at every rank.
                run(mapped_slot, degree - 1, -100, &oracle);
                if (timing)
                {
                    std::vector<double> native_samples, mapped_samples;
                    for (int sample = -5; sample < 31; ++sample)
                    {
                        const int first_rank = (sample + 5) % degree;
                        for (const int slot : sample % 2 == 0 ? std::array{native_slot, mapped_slot}
                                                             : std::array{mapped_slot, native_slot})
                        {
                            const auto measured = run(slot, first_rank, sample, &oracle).second;
                            if (sample >= 0) (slot == native_slot ? native_samples : mapped_samples).push_back(measured);
                        }
                    }
                    std::sort(native_samples.begin(), native_samples.end());
                    std::sort(mapped_samples.begin(), mapped_samples.end());
                    emit({{"kind", "summary"}, {"elements", shapes[shape].elements}, {"transactions", shapes[shape].transactions},
                        {"samples_per_path", 31}, {"warmups_per_path", 5}, {"native_median_amortized_us", native_samples[15]},
                        {"mapped_median_amortized_us", mapped_samples[15]}, {"native_range_us", {native_samples.front(), native_samples.back()}},
                        {"mapped_range_us", {mapped_samples.front(), mapped_samples.back()}},
                        {"scope", "captured_component_max_rank_event_includes_wait_and_host_arrival_not_model"}});
                }
            }
            if (!timing)
            {
                const auto oracle = run(0, degree - 1, -101, nullptr).first;
                run(1, 1, -101, &oracle); // Same retained large graph after small/tail families.
            }
            // Explicitly retire every executable while all channels, original
            // buffer owners, events, native context and PMA are still retained.
            for (auto &p : participants) p->families.clear();
        }
        for (const auto device : devices)
        {
            require(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation) == 0, "Fixed-rank device claim leaked");
            require(authority->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable) == 0,
                "Fixed-rank graph reservation leaked");
        }
        require(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation) == 0, "Fixed-rank mapped host claim leaked");
    }
}
