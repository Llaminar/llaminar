/**
 * @file Test__PlanningKernelServiceCatalogMPI.cpp
 * @brief Real all-rank collection through admitted source publication and captured service.
 *
 * The production collector chooses reporters from actual discovery, including
 * shared GPU visibility. Only root opens the tiny source fixture; the loaded
 * reader retires before preparation. Functional assertions prove complete CPU,
 * CUDA and ROCm evidence and collective rejection, not a performance threshold.
 * The heterogeneous idle-rank case requires a duplicate socket-bound reporter;
 * absent hardware or a malformed rank placement is a failure, never a skip.
 * Source-free services also run separately across all discovered endpoints so
 * driver diagnostics can distinguish bandwidth setup from arithmetic setup.
 * Source-backed startup experts cover exactly decode and prefill; live grouped
 * verifier service is certified separately through prepared runtime endpoints.
 */
#include "planning/PlanningKernelServiceCatalog.h"
#include "planning/PlanningWeightServiceModel.h"
#include "planning/PlanningForwardStateWork.h"
#include "planning/PlanningProjectionMeasurement.h"
#include "planning/PlanningCommunicationService.h"
#include "planning/PlanningHostDeviceMeasurement.h"
#include "planning/PlanningObservedResource.h"
#include "planning/PlanningCommunicationCost.h"
#include "planning/RankHardwareOwnership.h"
#include "planning/AutomaticPlanningStartup.h"
#include "backends/GPUDeviceContextPool.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <set>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Replay all-rank topology preparation with only one GPU/host role active.
     * @param backend Required native family, selected only from original owned GPU observations.
     *
     * No fixture eagerly calls topology(): the production collector must make
     * its lazy WORLD construction safe for the idle original peer. A rejected
     * asymmetric PMA is exercised first, before that construction is legal.
     */
    void proveHostDeviceSameRankRolesWithIdlePeer(DeviceType backend)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        ASSERT_GE(mpi->world_size(), 2);
        std::optional<PlanningHostDeviceGPUExecutor> selected;
        for (const auto& rank : inventory->ranks)
        {
            for (const auto& gpu : rank.gpus)
            {
                if (gpu.type == backend && rankOwnsGPUDevice(rank, gpu))
                {
                    selected = PlanningHostDeviceGPUExecutor{rank.rank, {gpu.type, gpu.local_device_id}};
                    break;
                }
            }
            if (selected) break;
        }
        ASSERT_TRUE(selected.has_value()) << "Required original NUMA-local native executor is absent";
        const auto request = PlanningHostDeviceRequest::fromInventory(*inventory,
            *selected, {selected->rank}, 1025);
        ASSERT_EQ(request.gpuRank(), request.hostRank());
        const auto& original = inventory->ranks.at(mpi->rank());
        const auto host = planningObservedResource(original, DeviceId::cpu());
        PhysicalMemoryPlanBuilder builder;
        // The idle rank has a genuine empty local sample envelope, rather than
        // acquiring a GPU or pretending its own host is the selected allocator.
        builder.add(host, PhysicalMemoryOwner::ActivationTransportStaging, 0);
        if (mpi->rank() == selected->rank)
        {
            PlanningHostDeviceMeasurement::contributeHostMemory(request, host, builder);
            PlanningHostDeviceMeasurement::contributeExecutionMemory(request, host,
                planningObservedResource(original, selected->device), builder);
        }
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), mpi->rank());
        const int idle_rank = selected->rank == 0 ? 1 : 0;
        const auto rejected_memory = mpi->rank() == idle_rank
            ? std::shared_ptr<PhysicalMemoryAuthority>{} : memory;
        EXPECT_THROW(PlanningHostDeviceMeasurement::collect(request, mpi, rejected_memory), std::runtime_error);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        for (int replay = 0; replay < 2; ++replay)
        {
            const auto result = PlanningHostDeviceMeasurement::collect(request, mpi, memory);
            EXPECT_EQ(result.has_value(), mpi->rank() == selected->rank);
            if (result)
            {
                EXPECT_EQ(result->request, request);
                EXPECT_EQ(result->phases.size(), 8u);
                EXPECT_NO_THROW(PlanningHostDeviceMeasurement::validate(*result));
            }
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            // This accessor is deliberately AFTER successful production
            // collection. The original bug never reaches it on the idle rank.
            EXPECT_NE(mpi->topology(), nullptr);
        }
    }

    /** @brief Real bounded native/MPI evidence and collective pre-traffic rejection on discovery membership. */
    void proveCommunicationService(DeviceType backend)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        std::vector<DeviceType> backends{DeviceType::CPU};
        if (backend != DeviceType::CPU) backends.push_back(backend);
        const AutomaticOrchestrationRequest request({.only_backends = backends});
        const std::array precisions{PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16};
        // One rank's changed payload must fail during plan agreement, before
        // any peer enters native collective initialization or posts a packet.
        EXPECT_THROW(PlanningCommunicationService::collect(mpi, *inventory, request, 257,
            mpi->rank() == 1 ? 32 : 31, precisions), std::runtime_error);
        const auto expected = PlanningCommunicationSamplePlan::resolve(*inventory, request, 257, 31, precisions);
        const auto service = PlanningCommunicationService::collect(mpi, *inventory, request, 257, 31, precisions);
        ASSERT_EQ(service.has_value(), mpi->is_root());
        if (!service) return;
        EXPECT_EQ(service->native().size(), expected.native().size());
        EXPECT_EQ(service->mpi().size(), expected.mpi().size());
        EXPECT_EQ(service->hostDevice().size(), expected.hostDevice().size());
        // The expected matrix is derived independently from the original
        // discovery roles, not from a native reporter's local host surrogate.
        size_t expected_host_scopes = 0;
        for (const auto &executor : inventory->ranks)
            for (const auto &gpu : executor.gpus)
                if (request.allows(gpu.type) && rankOwnsGPUDevice(executor, gpu))
                    for (const auto &host : inventory->ranks)
                        if (inventory->connectionBetweenRanks(executor.rank, host.rank).locality() != RankConnectionLocality::CrossNode)
                            ++expected_host_scopes;
        EXPECT_EQ(service->hostDevice().size(), expected_host_scopes);
        const PlanningCommunicationCost costs(*inventory, service->native(), service->mpi(), service->hostDevice());
        for (size_t index = 0; index < service->hostDevice().size(); ++index)
        {
            const auto &observed = service->hostDevice()[index];
            ASSERT_EQ(observed.request, expected.hostDevice()[index]);
            EXPECT_NO_THROW(requireRankLocalGPUOwnership(inventory->getRank(observed.request.gpuRank()),
                observed.request.device(), "host-device native executor"));
            EXPECT_EQ(observed.request.hostNumaNode(), inventory->getRank(observed.request.hostRank()).cpu.numa_node);
            EXPECT_NE(inventory->connectionBetweenRanks(observed.request.gpuRank(), observed.request.hostRank()).locality(),
                RankConnectionLocality::CrossNode);
            EXPECT_NO_THROW(PlanningHostDeviceMeasurement::validate(observed));
            for (const auto &phase : observed.phases)
            {
                const auto estimate = costs.hostTransfer(observed.request.gpuRank(), observed.request.device(),
                    observed.request.hostRank(), phase.mechanism, phase.direction, phase.payload_bytes);
                EXPECT_GT(estimate.seconds, 0);
                EXPECT_EQ(estimate.basis, PlanningCommunicationBasis::SameProtocolPayloadCurve);
                std::cout << "HOST_DEVICE_SAMPLE device=" << observed.request.device().toString()
                    << " gpu_rank=" << observed.request.gpuRank() << " host_rank=" << observed.request.hostRank() << " host_numa=" << observed.request.hostNumaNode()
                    << " mechanism=" << int(phase.mechanism) << " direction=" << int(phase.direction)
                    << " bytes=" << phase.payload_bytes << " seconds=" << phase.service.elapsedSeconds() / 3 << '\n';
            }
        }
        // This shared backend gate also runs on single-GPU hosts. An absent
        // native group has zero samples, not a fabricated one-device allreduce;
        // the dedicated NativeLocalTP gate separately requires multiple GPUs.
        for (size_t index = 0; index < service->native().size(); ++index)
        {
            const auto &native = service->native()[index];
            EXPECT_EQ(native.sample.uuids, expected.native()[index].uuids);
            EXPECT_EQ(native.sample.discovery_rank, expected.native()[index].discovery_rank);
            for (const auto &phase : native.observation.phases)
            {
                EXPECT_GT(phase.secondsPerCollective(), 0);
                EXPECT_EQ(phase.endpoints.size(), native.sample.uuids.size());
                std::cout << "NATIVE_COMM_SAMPLE backend=" << int(backend) << " rows=" << phase.rows
                          << " degree=" << phase.endpoints.size() << " seconds=" << phase.secondsPerCollective() << '\n';
                const auto estimate = costs.nativeAllreduce(native.sample.discovery_rank,
                    native.sample.request.devices(), native.sample.request.hiddenWidth(), phase.rows,
                    native.sample.request.precision());
                EXPECT_EQ(estimate.basis, PlanningCommunicationBasis::SameProtocolPayloadCurve);
                EXPECT_DOUBLE_EQ(estimate.seconds, std::max(native.observation.phases.front().secondsPerCollective(),
                    phase.secondsPerCollective()));
            }
            const std::array single{native.sample.request.devices().front()};
            EXPECT_EQ(costs.nativeAllreduce(native.sample.discovery_rank, single, 257, 31,
                native.sample.request.precision()).seconds, 0);
            if (native.sample.request.devices().size() > 2)
            {
                const std::span subset(native.sample.request.devices().data(), 2);
                EXPECT_EQ(costs.nativeAllreduce(native.sample.discovery_rank, subset, 257, 31,
                    native.sample.request.precision()).basis, PlanningCommunicationBasis::ContainingNativeGroup);
            }
        }
        for (size_t index = 0; index < service->mpi().size(); ++index)
        {
            const auto &sample = service->mpi()[index];
            EXPECT_EQ(sample.request(), expected.mpi()[index]);
            EXPECT_EQ(sample.topology(), inventory->connectionBetweenRanks(sample.request().initiator, sample.request().responder));
            EXPECT_GT(sample.secondsPerExchange(), 0);
        }
        // The three independent completed samples identify one whole control
        // RTT and each direction's payload increment. No GPU staging or remote
        // compute is folded into this host-MPI protocol estimate.
        ASSERT_EQ(service->mpi().size() % 3, 0u);
        for (size_t index = 0; index < service->mpi().size(); index += 3)
        {
            const auto &control = service->mpi()[index];
            const auto &outbound = service->mpi()[index + 1];
            const auto &inbound = service->mpi()[index + 2];
            EXPECT_EQ(control.request().request_bytes, 1u);
            EXPECT_EQ(control.request().reply_bytes, 1u);
            EXPECT_EQ(outbound.request().reply_bytes, 1u);
            EXPECT_EQ(inbound.request().request_bytes, 1u);
            EXPECT_GT(outbound.request().request_bytes, 1u);
            EXPECT_GT(inbound.request().reply_bytes, 1u);
            const double baseline = control.secondsPerExchange();
            EXPECT_DOUBLE_EQ(costs.mpiExchange(control.request()).seconds, baseline);
            EXPECT_DOUBLE_EQ(costs.mpiExchange(outbound.request()).seconds,
                std::max(baseline, outbound.secondsPerExchange()));
            EXPECT_DOUBLE_EQ(costs.mpiExchange(inbound.request()).seconds,
                std::max(baseline, inbound.secondsPerExchange()));
            const auto mixed = PlanningMPITransferRequest{control.request().initiator, control.request().responder,
                outbound.request().request_bytes, inbound.request().reply_bytes};
            const auto estimate = costs.mpiExchange(mixed);
            EXPECT_DOUBLE_EQ(estimate.seconds, baseline +
                (std::max(baseline, outbound.secondsPerExchange()) - baseline) +
                (std::max(baseline, inbound.secondsPerExchange()) - baseline));
            EXPECT_EQ(estimate.basis, PlanningCommunicationBasis::SameProtocolPayloadCurve);
            EXPECT_NE(estimate.evidence.find("excludes GPU DMA"), std::string::npos);
            const auto larger = PlanningMPITransferRequest{mixed.initiator, mixed.responder,
                mixed.request_bytes * 2, mixed.reply_bytes * 3};
            EXPECT_GT(costs.mpiExchange(larger).seconds, estimate.seconds);
        }
        auto missing = service->mpi();
        ASSERT_FALSE(missing.empty());
        missing.pop_back();
        EXPECT_THROW(PlanningCommunicationCost(*inventory, service->native(), missing), std::invalid_argument);
        auto duplicate = service->mpi();
        duplicate.push_back(duplicate.front());
        EXPECT_THROW(PlanningCommunicationCost(*inventory, service->native(), duplicate), std::invalid_argument);
        auto changed = *inventory;
        // Physical membership is authenticated independently of hostnames and
        // observed latency. Reclassifying a rank invalidates the receipt.
        changed.ranks[1].node_id += inventory->world_size + 1;
        EXPECT_THROW(PlanningCommunicationCost(changed, {}, service->mpi()), std::invalid_argument);
        EXPECT_THROW(costs.mpiExchange({0, 0, 1, 1}), std::invalid_argument);
        EXPECT_THROW(costs.mpiExchange({0, 1, 0, 1}), std::invalid_argument);
    }

    /**
     * @brief Prove bounded compute/memory composition inside the common discovery transaction.
     *
     * This is the unmodified default plan/serve preparation policy, including
     * non-weight work and link measurements. The sparse fixture authenticates
     * lifecycle and bounded sampling, not real-model mathematical correctness.
     * Every observer must complete before root begins candidate pricing.
     */
    void proveRequestCostPreparation(DeviceType backend)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        std::unique_ptr<PlanningGGUFFixture> fixture;
        OrchestrationConfig config;
        acceptPlanningCostPreparation(mpi, [&] {
            if (mpi->is_root())
            {
                fixture = std::make_unique<PlanningGGUFFixture>(true, false, GGUFTensorType::F32, 512);
                config.model_path = fixture->path();
            }
            else config.model_path = "/__followers_must_not_read__/missing.gguf";
        });
        config.max_seq_len = 128;
        config.prefill_max_bucket_size = 64;
        // Static is explicit fixture policy, not an automatic runtime change.
        // This metadata fixture certifies initial request costing, not movement.
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        config.automatic_planning.only_backends = backend == DeviceType::CPU ?
            std::vector{DeviceType::CPU} : std::vector{DeviceType::CPU, backend};
        config.automatic_planning.only_strategies = backend == DeviceType::CPU ?
            std::vector{OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel} :
            std::vector{OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel,
                OrchestrationStrategy::ExpertOverlay};
        size_t prices = 0;
        size_t multi_device_prices = 0, overlay_prices = 0;
        const auto result = AutomaticPlanningStartup::run(config, *inventory, mpi,
            [&](const AutomaticPlanningPreparation &context) -> std::optional<AutomaticOrchestrationPlanner::Evaluate> {
                auto service = AutomaticPlanningStartup::preparation()(context);
                if (!context.isRoot())
                {
                    EXPECT_FALSE(service);
                    return std::nullopt;
                }
                if (!service) throw std::logic_error("Root has no completed service model");
                // One lifecycle boundary distinguishes real cold measurements
                // from subsequent host admission/ranking without per-candidate I/O.
                LOG_INFO("[PlanningRequestCostPreparation] measured evidence complete backend="
                    << deviceTypeToString(backend));
                return [service = std::move(*service), &prices, &multi_device_prices, &overlay_prices]
                    (const auto &candidate, const auto &workload) {
                    ++prices;
                    const auto cost = service(candidate, workload);
                    EXPECT_GT(cost.requestSeconds(), 0);
                    EXPECT_NE(cost.evidence().find("bounded measured ranking"), std::string::npos);
                    if (candidate.strategy() == OrchestrationStrategy::SingleDevice)
                        EXPECT_NE(cost.evidence().find("mean_decode_interconnect_s=0"), std::string::npos);
                    else
                    {
                        ++multi_device_prices;
                        EXPECT_EQ(cost.evidence().find("prefill_interconnect_s=0;"), std::string::npos);
                        const bool replicated_decode = std::any_of(candidate.devicePlans().begin(),
                            candidate.devicePlans().end(), [](const auto &device) {
                                return std::find(device.additional_weight_sets.begin(),
                                    device.additional_weight_sets.end(),
                                    AdditionalPersistentWeightSet::ReplicatedDenseDecode) !=
                                    device.additional_weight_sets.end();
                            });
                        // Replicated dense decode removes the TP join, while an
                        // ExpertOverlay still communicates routed activations.
                        // Cost evidence must follow that admitted policy rather
                        // than infer behavior from a broad strategy label.
                        const auto &placement = candidate.config().moe_routed_expert_plan;
                        const bool distributed_down_columns = placement &&
                            std::any_of(placement->domains.begin(), placement->domains.end(),
                                [](const auto &domain) {
                                    return domain.routed_compute_policy ==
                                        RoutedExpertComputePolicy::GateUpOwnedDownColumns;
                                });
                        // A replicated dense view does not replicate the
                        // projection mode's distributed expert down columns.
                        // Their publication remains real decode traffic.
                        if (candidate.strategy() == OrchestrationStrategy::TensorParallel &&
                            replicated_decode && !distributed_down_columns)
                            EXPECT_NE(cost.evidence().find("mean_decode_interconnect_s=0;"), std::string::npos);
                        else
                            EXPECT_EQ(cost.evidence().find("mean_decode_interconnect_s=0;"), std::string::npos);
                    }
                    if (candidate.strategy() == OrchestrationStrategy::ExpertOverlay) ++overlay_prices;
                    return cost;
                };
            });
        if (result.root_selection)
        {
            const auto &counts = result.root_selection->counts();
            LOG_INFO("[PlanningRequestCostPreparation] selection complete backend="
                << deviceTypeToString(backend)
                << " proposed=" << counts.proposed << " evaluated=" << counts.evaluated
                << " capacity_rejected=" << counts.capacity_rejected);
        }
        EXPECT_EQ(result.root_selection.has_value(), mpi->is_root());
        if (mpi->is_root())
        {
            EXPECT_GT(prices, 0u);
            // CPU endpoints are physical NUMA nodes, not MPI ranks. Ranks that
            // share one node (a single-NUMA GB10) cannot form a CPU TP group;
            // the planner must then price no multi-device candidate at all.
            std::set<std::pair<int, int>> cpu_endpoints;
            for (const auto &rank : inventory->ranks)
                cpu_endpoints.emplace(rank.node_id, rank.cpu.numa_node);
            if (backend != DeviceType::CPU || cpu_endpoints.size() >= 2)
                EXPECT_GT(multi_device_prices, 0u);
            else
                EXPECT_EQ(multi_device_prices, 0u);
            if (backend != DeviceType::CPU) EXPECT_GT(overlay_prices, 0u);
        }
        else EXPECT_EQ(prices, 0u);
        EXPECT_TRUE(result.applied.execution_rank_selection);
    }

    /**
     * @brief Exercise one source-free service with every discovered CPU/GPU reporter.
     * @param backend GPU family to include alongside the owned CPU workshares.
     * @param plan The exact streaming or arithmetic service under investigation.
     *
     * Single-device tests do not cover several native GPU contexts in the same
     * process. Keep these phases independently selectable for driver-log guards
     * without changing the production collector's placement or sampling order.
     */
    template<class Plan>
    void proveSourceFreeCollection(DeviceType backend, Plan plan)
    {
        static_assert(std::is_same_v<Plan, PlanningStreamingServicePlan> ||
                      std::is_same_v<Plan, PlanningFP32ArithmeticPlan>);
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_GE(mpi->world_size(), 2);
        const AutomaticOrchestrationRequest selection({.only_backends =
            std::vector{DeviceType::CPU, backend}});
        const auto expected = PlanningKernelServiceCatalog::observers(*inventory, selection);
        constexpr int replays=std::is_same_v<Plan,PlanningFP32ArithmeticPlan>?2:1;
        for(int replay=0;replay<replays;++replay) {
        const auto catalog = PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, plan);
        acceptPlanningCostPreparation(mpi, [&] {
            ASSERT_EQ(catalog.has_value(), mpi->is_root());
            if (!catalog) return;
            ASSERT_EQ(catalog->records().size(), expected.size());
            for (size_t index = 0; index < expected.size(); ++index)
            {
                const auto &record = catalog->records()[index];
                EXPECT_EQ(record.observer, expected[index]);
                if constexpr (std::is_same_v<Plan, PlanningStreamingServicePlan>)
                {
                    const auto &value = std::get<PlanningMemoryBandwidthObservation>(record.observation);
                    EXPECT_EQ(value.request.device(), record.observer.device());
                    EXPECT_EQ(value.graph_nodes != 0u, record.observer.device().is_gpu());
                    EXPECT_GT(value.service.unitsPerSecond(), 0);
                }
                else
                {
                    const auto &value = std::get<PlanningFP32ArithmeticObservations>(record.observation);
                    EXPECT_EQ(value.device, record.observer.device());
                    if(value.device.is_gpu())EXPECT_GT(value.prepared_bytes,0u);
                    else EXPECT_EQ(value.prepared_bytes,0u) << "CPU FP32 borrows its original source";
                    ASSERT_EQ(value.phases.size(), 2u);
                    EXPECT_EQ(value.phases[0].rows,1);
                    EXPECT_EQ(value.phases[1].rows,PlanningFP32ArithmeticPlan::kPrefillRows);
                    for (const auto &phase : value.phases)
                    {
                        EXPECT_EQ(phase.graph_nodes != 0u, value.device.is_gpu());
                        EXPECT_GT(phase.service.unitsPerSecond(), 0);
                        EXPECT_DOUBLE_EQ(phase.service.completedWork(),3.0*2*phase.rows*
                            PlanningFP32ArithmeticPlan::kN*PlanningFP32ArithmeticPlan::kK);
                    }
                }
            }
        });
        }
    }

    /** @brief Replay the real prepared arithmetic owner under one reusable canonical admission.
     * @param backend Native family; only an original NUMA/rank-owned endpoint may execute.
     * Empty local reporters remain idle rather than adopting another rank's GPU.
     */
    void provePreparedArithmeticLifecycle(DeviceType backend)
    {
        const auto mpi=MPIContextFactory::global();
        const auto inventory=mpi->clusterInventory();
        ASSERT_NE(inventory,nullptr);
        const bool present=std::any_of(inventory->ranks.begin(),inventory->ranks.end(),[&](const auto &rank) {
            return std::any_of(rank.gpus.begin(),rank.gpus.end(),[&](const auto &gpu) {
                return gpu.type==backend && rankOwnsGPUDevice(rank,gpu);
            });
        });
        ASSERT_TRUE(present)<<"Required original NUMA-local backend executor is absent";
        const auto &original=inventory->getRank(mpi->rank());
        std::optional<DeviceId> selected;
        for(const auto &gpu:original.gpus)
            if(gpu.type==backend && rankOwnsGPUDevice(original,gpu)) {
                selected=DeviceId(backend,gpu.local_device_id);break;
            }
        if(!selected)return; // This original rank has no role for this backend.
        const auto device=*selected;
        const auto host=planningObservedResource(original,DeviceId::cpu());
        const auto execution=planningObservedResource(original,device);
        auto &worker=GPUDeviceContextPool::instance().getContext(device);
        PhysicalMemoryPlanBuilder builder;
        worker.submitAndWait([&] {
            PlanningProjectionMeasurement::contributeMemory(PlanningFP32ArithmeticPlan{},host,execution,builder);
        });
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()),mpi->rank());
        EXPECT_THROW(PlanningPreparedGPUProjection(PlanningFP32ArithmeticPlan{},device,memory),std::invalid_argument);
        const auto requireRetired=[&] {
            // Native process/context declarations intentionally survive setup;
            // every private sample allocation and graph must retire instead.
            for(const auto allocator:{DeviceId::cpu(),device})
                for(const auto owner:{PhysicalMemoryOwner::ModelSourcePayload,
                    PhysicalMemoryOwner::PrimaryModelWeights,PhysicalMemoryOwner::ActivationArena,
                    PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryOwner::WeightLoadStaging,
                    PhysicalMemoryOwner::NativeGraphExecutable,
                    PhysicalMemoryOwner::NativePreparationGraphExecutable}) {
                    EXPECT_EQ(memory->claimedBytes(allocator,owner,
                        PhysicalMemoryMaterializationKind::NewAllocation),0u);
                    EXPECT_EQ(memory->reservedBytes(allocator,owner),0u);
                }
        };
        // A materialized phase does not oblige the caller to execute. Destroying
        // that prepared phase must revoke its graph/tensor/workspace owners on
        // the original worker before the next producer uses the same admission.
        worker.submitAndWait([&] {
            PlanningPreparedGPUProjection abandoned(PlanningFP32ArithmeticPlan{},device,memory);
            abandoned.prepareNextPhase();
        });
        requireRetired();
        for(int replay=0;replay<2;++replay) {
            worker.submitAndWait([&] {
                PlanningPreparedGPUProjection producer(PlanningFP32ArithmeticPlan{},device,memory);
                EXPECT_THROW(producer.result(),std::logic_error);
                EXPECT_THROW(producer.observePreparedPhase(),std::logic_error);
                EXPECT_THROW(producer.capturePreparedPhase(),std::logic_error);
                EXPECT_THROW(producer.completePreparedPhase(),std::logic_error);
                for(int phase=0;phase<2;++phase) {
                    if(replay==0) {
                        producer.prepareNextPhaseStorage();
                        EXPECT_THROW(producer.prepareNextPhaseStorage(),std::logic_error);
                        EXPECT_THROW(producer.prepareNextPhase(),std::logic_error);
                        EXPECT_THROW(producer.completePreparedPhase(),std::logic_error);
                        EXPECT_THROW(producer.observePreparedPhase(),std::logic_error);
                        EXPECT_THROW(producer.result(),std::logic_error);
                        producer.capturePreparedPhase();
                        EXPECT_THROW(producer.capturePreparedPhase(),std::logic_error);
                        EXPECT_THROW(producer.prepareNextPhaseStorage(),std::logic_error);
                        EXPECT_THROW(producer.observePreparedPhase(),std::logic_error);
                        EXPECT_THROW(producer.result(),std::logic_error);
                        producer.completePreparedPhase();
                    } else producer.prepareNextPhase();
                    EXPECT_THROW(producer.prepareNextPhase(),std::logic_error);
                    EXPECT_THROW(producer.prepareNextPhaseStorage(),std::logic_error);
                    EXPECT_THROW(producer.capturePreparedPhase(),std::logic_error);
                    EXPECT_THROW(producer.completePreparedPhase(),std::logic_error);
                    EXPECT_THROW(producer.result(),std::logic_error);
                    producer.observePreparedPhase();
                    EXPECT_THROW(producer.observePreparedPhase(),std::logic_error);
                    EXPECT_THROW(producer.capturePreparedPhase(),std::logic_error);
                    EXPECT_THROW(producer.completePreparedPhase(),std::logic_error);
                }
                const auto result=std::get<PlanningFP32ArithmeticObservations>(producer.result());
                ASSERT_EQ(result.phases.size(),2u);
                EXPECT_EQ(result.device,device);
                EXPECT_GT(result.prepared_bytes,0u);
                for(size_t phase=0;phase<2;++phase) {
                    const auto &value=result.phases[phase];
                    EXPECT_EQ(value.rows,phase==0?1:PlanningFP32ArithmeticPlan::kPrefillRows);
                    EXPECT_GT(value.graph_nodes,0u);
                    EXPECT_DOUBLE_EQ(value.service.completedWork(),3.0*2*value.rows*
                        PlanningFP32ArithmeticPlan::kN*PlanningFP32ArithmeticPlan::kK);
                }
                EXPECT_EQ(std::get<PlanningFP32ArithmeticObservations>(producer.result()).prepared_bytes,result.prepared_bytes);
                EXPECT_THROW(producer.prepareNextPhase(),std::logic_error);
                EXPECT_THROW(producer.prepareNextPhaseStorage(),std::logic_error);
            });
            requireRetired();
        }
    }

    /** @brief Join real FP32/streaming observations without another sample or model warmup. */
    void proveStreamingCollection(DeviceType backend)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_GE(mpi->world_size(), 2);
        std::vector<DeviceType> backends{DeviceType::CPU};
        if (backend != DeviceType::CPU) backends.push_back(backend);
        const AutomaticOrchestrationRequest selection({.only_backends = backends});
        const auto expected = PlanningKernelServiceCatalog::observers(*inventory, selection);
        const auto catalog = PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, PlanningStreamingServicePlan{});
        const auto arithmetic = PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, PlanningFP32ArithmeticPlan{});
        acceptPlanningCostPreparation(mpi, [&] {
            ASSERT_EQ(catalog.has_value(), mpi->is_root());
            ASSERT_EQ(arithmetic.has_value(), mpi->is_root());
            if (!catalog) return;
            ASSERT_EQ(catalog->records().size(), expected.size());
            ASSERT_EQ(arithmetic->records().size(), expected.size());
            for (size_t index = 0; index < expected.size(); ++index)
            {
                const auto &record = catalog->records()[index];
                EXPECT_EQ(record.observer, expected[index]);
                const auto &value = std::get<PlanningMemoryBandwidthObservation>(record.observation);
                EXPECT_EQ(value.request.rank(), record.observer.discoveryRank());
                EXPECT_EQ(value.request.device(), record.observer.device());
                EXPECT_GT(value.service.unitsPerSecond(), 0);
                EXPECT_EQ(value.service.unit(), PlanningWorkUnit::Bytes);
                const auto &compute = arithmetic->serviceFor(record.observer.discoveryRank(), record.observer.device());
                EXPECT_EQ(compute.observer, record.observer);
                const auto &fp32 = std::get<PlanningFP32ArithmeticObservations>(compute.observation);
                if (fp32.cpu)
                    EXPECT_EQ(fp32.cpu->workers, inventory->ranks.at(record.observer.discoveryRank()).cpuWorkerThreads());
                ASSERT_EQ(fp32.phases.size(), 2u);
                for (const auto &phase : fp32.phases) EXPECT_EQ(phase.graph_nodes != 0, fp32.device.is_gpu());

                // Only the service observations are measured. Metadata below
                // exercises pure cost composition; this is not a model benchmark.
                ModelMemoryProfile profile;
                profile.architecture = "qwen35";
                profile.n_layers = 1;
                profile.n_heads = 4;
                profile.n_kv_heads = 2;
                profile.head_dim = 64;
                profile.d_model = profile.vocab_size = 256;
                profile.d_ff = 512;
                profile.max_seq_len = 32768;
                const PlanningModelMetadata model(profile, 1);
                DevicePlanConfig device;
                device.device = record.observer.device();
                device.activation_seq_len = 64;
                device.max_seq_len = profile.max_seq_len;
                const auto short_work = compilePlanningForwardStateWork(model, device, PlanningMainForwardPhase::Decode, {1, 0});
                const auto long_work = compilePlanningForwardStateWork(model, device, PlanningMainForwardPhase::Decode, {1, 8192});
                ASSERT_EQ(short_work.size(), 1u);
                ASSERT_EQ(long_work.size(), 1u);
                const double short_seconds = planningStateServiceSeconds(short_work[0], fp32, value);
                const double long_seconds = planningStateServiceSeconds(long_work[0], fp32, value);
                EXPECT_GT(short_seconds, 0);
                EXPECT_GT(long_seconds, short_seconds);
                const auto prefill = compilePlanningForwardStateWork(model, device, PlanningMainForwardPhase::Prefill, {64, 512});
                EXPECT_GT(planningStateServiceSeconds(prefill[0], fp32, value), 0);
                std::cout << "STREAMING_SERVICE reporter=" << record.observer.discoveryRank()
                    << " node=" << record.observer.physicalNode() << " device=" << value.request.device().toString()
                    << " workers=" << value.request.workers() << " useful_GB_per_s=" << value.service.unitsPerSecond() / 1e9 << '\n';
            }
        });
    }

    /**
     * @brief Prove source-free GPU measurement does not invent PMA work on idle ranks.
     * @param backends Both GPU backends required by this heterogeneous MPI registration.
     *
     * A physical GPU is intentionally measured by one affinity-selected MPI
     * reporter.  Other discovery ranks still join publication and receipt
     * collectives, but they own neither a source payload nor a workspace for
     * this probe.  This reproduces heterogeneous GPU-only automatic planning:
     * constructing an empty PMA certificate on the non-reporting rank is an
     * accounting error, not a capacity rejection.
     */
    void proveGPUOnlySourceFreeCollectionWithIdleReporterRank(
        std::vector<DeviceType> backends)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_GE(mpi->world_size(), 2);
        const AutomaticOrchestrationRequest selection({.only_backends = std::move(backends)});
        const auto expected = PlanningKernelServiceCatalog::observers(*inventory, selection);
        ASSERT_FALSE(expected.empty()) << "No selected GPU is available";
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [&](const auto& observer) {
                return observer.device().type == backend;
            })) << "The heterogeneous source-free probe requires both selected backends";
        const bool has_idle_rank = std::any_of(inventory->ranks.begin(), inventory->ranks.end(),
            [&](const RankInventory &rank) {
                return std::none_of(expected.begin(), expected.end(), [&](const auto &observer) {
                    return observer.discoveryRank() == rank.rank;
                });
            });
        ASSERT_TRUE(has_idle_rank)
            << "The local MPI registration must include a duplicate socket-bound reporter";

        const auto streaming = PlanningKernelServiceCatalog::collect(
            mpi, *inventory, selection, PlanningStreamingServicePlan{});
        const auto arithmetic = PlanningKernelServiceCatalog::collect(
            mpi, *inventory, selection, PlanningFP32ArithmeticPlan{});
        EXPECT_EQ(streaming.has_value(), mpi->is_root());
        EXPECT_EQ(arithmetic.has_value(), mpi->is_root());
        if (!mpi->is_root()) return;
        ASSERT_TRUE(streaming);
        ASSERT_TRUE(arithmetic);
        EXPECT_EQ(streaming->records().size(), expected.size());
        EXPECT_EQ(arithmetic->records().size(), expected.size());
    }

    /** @return Exact full source triplet; the last expert detects incorrect source slicing. */
    PlanningExpertSampleRequest request()
    {
        return {{"blk.0.ffn_gate_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_up_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_down_exps.weight", PlanningExpertMatrix{7}}};
    }

    /** @brief Collect all actual cards of one backend plus rank-local CPU workshares. */
    void proveCollection(DeviceType backend)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_GE(mpi->world_size(), 2);
        std::vector<DeviceType> backends{DeviceType::CPU};
        if (backend != DeviceType::CPU) backends.push_back(backend);
        const AutomaticOrchestrationRequest selection({.only_backends = backends});
        const auto expected = PlanningKernelServiceCatalog::observers(*inventory, selection);
        if (backend != DeviceType::CPU)
            ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [&](const auto &observer) {
                return observer.device().type == backend;
            }));
        for (auto format : {GGUFTensorType::Q8_0, GGUFTensorType::F16, GGUFTensorType::BF16, GGUFTensorType::F32})
        for(int replay=0;replay<(format==GGUFTensorType::Q8_0?2:1);++replay)
        {
            SCOPED_TRACE(replay);
            std::unique_ptr<PlanningGGUFFixture> fixture;
            std::unique_ptr<PlanningModelSource> source;
            const auto plan = PlanningExpertSamplePublication::describe(mpi, [&] {
                fixture = std::make_unique<PlanningGGUFFixture>(true, false, format, 512);
                source = std::make_unique<PlanningModelSource>(fixture->path());
                return PlanningExpertSamplePlan::resolve(*source, request());
            });
            int loads = 0;
            std::weak_ptr<PhysicalMemoryAuthority> root_memory;
            const auto catalog = PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, plan,
                [&](const auto &memory) {
                    ++loads;
                    root_memory = memory;
                    auto loaded = plan.load(*source, memory, DeviceId::cpu());
                    source.reset();
                    fixture.reset();
                    return loaded;
                });
            EXPECT_EQ(loads, mpi->is_root() ? 1 : 0);
            EXPECT_TRUE(root_memory.expired());
            EXPECT_EQ(catalog.has_value(), mpi->is_root());
            if (!catalog) continue;
            ASSERT_EQ(catalog->records().size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i)
            {
                const auto &record = catalog->records()[i];
                EXPECT_EQ(record.observer, expected[i]);
                EXPECT_EQ(record.observer.physicalNode(),
                    inventory->connectionBetweenRanks(0, record.observer.discoveryRank()).destinationNode());
                std::visit([&](const auto &observation) {
                    if constexpr (!std::is_same_v<std::decay_t<decltype(observation)>, PlanningCPUExpertObservations> &&
                        !std::is_same_v<std::decay_t<decltype(observation)>, PlanningGPUExpertObservations>)
                        FAIL() << "Expert receipt was replaced by another service kind";
                    else
                    {
                    ASSERT_EQ(observation.phases.size(), 2u);
                    EXPECT_EQ(observation.phases[0].phase,ExpertHistogramSource::DecodeToken);
                    EXPECT_EQ(observation.phases[1].phase,ExpertHistogramSource::PrefillChunk);
                    EXPECT_EQ(observation.formats, plan.description().formats);
                    for (const auto &phase : observation.phases)
                    {
                        EXPECT_GT(phase.service.elapsedSeconds(), 0);
                        EXPECT_DOUBLE_EQ(phase.service.completedWork(),3.0*6*phase.rows*
                            observation.input_width*observation.intermediate_width);
                        if constexpr (requires { phase.graph_nodes; }) EXPECT_GT(phase.graph_nodes, 0u);
                    }
                    std::cout << "EXPERT_SERVICE reporter=" << record.observer.discoveryRank()
                        << " node=" << record.observer.physicalNode() << " device=" << observation.device.toString()
                        << " format=" << observation.formats[0]
                        << " decode_sample_seconds=" << observation.phases[0].service.elapsedSeconds() << '\n';
                    }
                }, record.observation);
            }
        }
    }

    /** @brief Collect a source-sharded ordinary projection through the same reporter/PMA lifecycle. */
    /** @brief Replay native and promoted source collection after original readers and PMA owners retire.
     * @param backend Native family; production discovery chooses every actual owned reporter.
     * Small source cases cover native Q8_0/F16/BF16/F32 and F16 GDN promotion;
     * no fixture substitutes synthetic arithmetic for the published source.
     */
    void provePreparedSourceCollectionReplay(DeviceType backend)
    {
        struct RestoreArithmetic {
            GDNProjectionArithmetic previous=debugEnv().gemm.gdn_projection_arithmetic;
            ~RestoreArithmetic(){mutableDebugEnv().gemm.gdn_projection_arithmetic=previous;}
        } restore;
        mutableDebugEnv().gemm.gdn_projection_arithmetic=GDNProjectionArithmetic::FP32;
        const auto mpi=MPIContextFactory::global();
        const auto inventory=mpi->clusterInventory();
        const AutomaticOrchestrationRequest selection({.only_backends=std::vector{backend}});
        const auto expected=PlanningKernelServiceCatalog::observers(*inventory,selection);
        ASSERT_FALSE(expected.empty());
        const std::array cases{
            std::pair{GGUFTensorType::Q8_0,false},std::pair{GGUFTensorType::F16,false},
            std::pair{GGUFTensorType::BF16,false},std::pair{GGUFTensorType::F32,false},
            std::pair{GGUFTensorType::F16,true}};
        for(const auto &[format,converted]:cases)
        for(int replay=0;replay<2;++replay) {
            SCOPED_TRACE(static_cast<int>(format));
            SCOPED_TRACE(converted);
            SCOPED_TRACE(replay);
            std::unique_ptr<PlanningGGUFFixture> fixture;
            std::unique_ptr<PlanningModelSource> source;
            const auto matrix=PlanningMatrixSamplePublication::describe(mpi,[&] {
                fixture=std::make_unique<PlanningGGUFFixture>(false,false,format,256,format);
                source=std::make_unique<PlanningModelSource>(fixture->path());
                return PlanningMatrixSamplePlan::resolve(*source,
                    {converted?"blk.0.ssm_alpha.weight":"blk.0.ffn_gate.weight",PlanningMatrixRows{64,129}});
            });
            EXPECT_EQ(matrix.executionType()!=matrix.tensorType(),converted);
            const PlanningProjectionServicePlan plan(matrix,7);
            std::weak_ptr<PhysicalMemoryAuthority> original_memory;
            int loads=0;
            const auto catalog=PlanningKernelServiceCatalog::collect(mpi,*inventory,selection,plan,[&](const auto &memory) {
                ++loads;original_memory=memory;
                auto sample=matrix.load(*source,memory,DeviceId::cpu());
                source.reset();fixture.reset();return sample;
            });
            EXPECT_EQ(loads,mpi->is_root()?1:0);
            EXPECT_TRUE(original_memory.expired())<<"Source/preparation PMA survived completed collection";
            acceptPlanningCostPreparation(mpi,[&] {
                ASSERT_EQ(catalog.has_value(),mpi->is_root());
                if(!catalog)return;
                ASSERT_EQ(catalog->records().size(),expected.size());
                for(size_t index=0;index<expected.size();++index) {
                    const auto &record=catalog->records()[index];
                    EXPECT_EQ(record.observer,expected[index]);
                    const auto &value=std::get<PlanningProjectionObservations>(record.observation);
                    EXPECT_EQ(value.source.serialize(),matrix.serialize());
                    EXPECT_EQ(value.device,record.observer.device());
                    EXPECT_GT(value.prepared_bytes,0u);
                    ASSERT_EQ(value.phases.size(),2u);
                    for(size_t phase=0;phase<2;++phase) {
                        const auto &work=value.phases[phase];
                        EXPECT_EQ(work.rows,phase==0?1:7);
                        EXPECT_GT(work.graph_nodes,0u);
                        EXPECT_GT(work.service.elapsedSeconds(),0);
                        EXPECT_DOUBLE_EQ(work.service.completedWork(),3.0*2*work.rows*
                            matrix.geometry().n*matrix.geometry().k);
                    }
                }
            });
        }
    }

    void proveProjectionCollection(DeviceType backend)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_GE(mpi->world_size(), 2);
        std::vector<DeviceType> backends{DeviceType::CPU};
        if (backend != DeviceType::CPU) backends.push_back(backend);
        const AutomaticOrchestrationRequest selection({.only_backends = backends});
        const auto expected = PlanningKernelServiceCatalog::observers(*inventory, selection);
        for (auto format : {GGUFTensorType::Q8_0, GGUFTensorType::F16, GGUFTensorType::BF16, GGUFTensorType::F32})
        for (const auto name : {"blk.0.ffn_gate.weight", "blk.0.ssm_alpha.weight"})
        {
            SCOPED_TRACE(name);
            SCOPED_TRACE(static_cast<int>(format));
            std::unique_ptr<PlanningGGUFFixture> fixture;
            std::unique_ptr<PlanningModelSource> source;
            const auto matrix = PlanningMatrixSamplePublication::describe(mpi, [&] {
                fixture = std::make_unique<PlanningGGUFFixture>(false, false, format, 256, format);
                source = std::make_unique<PlanningModelSource>(fixture->path());
                return PlanningMatrixSamplePlan::resolve(*source,
                    {name, PlanningMatrixRows{64, 129}});
            });
            const PlanningProjectionServicePlan plan(matrix, 7);
            int loads = 0;
            std::weak_ptr<PhysicalMemoryAuthority> root_memory;
            const auto catalog = PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, plan,
                [&](const auto &memory) {
                    ++loads;
                    root_memory = memory;
                    auto loaded = matrix.load(*source, memory, DeviceId::cpu());
                    source.reset();
                    fixture.reset();
                    return loaded;
                });
            EXPECT_EQ(loads, mpi->is_root() ? 1 : 0);
            EXPECT_TRUE(root_memory.expired());
            EXPECT_EQ(catalog.has_value(), mpi->is_root());
            if (!catalog) continue;
            ASSERT_EQ(catalog->records().size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i)
            {
                const auto &record = catalog->records()[i];
                EXPECT_EQ(record.observer, expected[i]);
                const auto *observation = std::get_if<PlanningProjectionObservations>(&record.observation);
                ASSERT_NE(observation, nullptr);
                EXPECT_EQ(observation->source.serialize(), matrix.serialize());
                EXPECT_EQ(observation->device, expected[i].device());
                ASSERT_EQ(observation->phases.size(), 2u);
                if (matrix.executionType() != matrix.tensorType())
                {
                    EXPECT_EQ(matrix.executionType(), TensorType::FP32);
                    if (observation->cpu) EXPECT_EQ(observation->prepared_bytes, 0u);
                    else EXPECT_GE(observation->prepared_bytes, matrix.executionPayloadBytes());
                }
                if (observation->cpu)
                    EXPECT_EQ(observation->cpu->execution,
                        inventory->ranks.at(record.observer.discoveryRank()).cpu_execution);
                for (const auto &phase : observation->phases)
                {
                    EXPECT_EQ(phase.graph_nodes != 0, observation->device.is_gpu());
                    EXPECT_EQ(phase.service.completedWork(), 3.0 * 2 * phase.rows * 65 * 256);
                    EXPECT_GT(phase.service.elapsedSeconds(), 0);
                    EXPECT_NE(phase.service.provenance().find("execution-format=" + matrix.executionFormat()),
                        std::string::npos);
                }
            }
        }
    }
}

TEST(PlanningKernelServiceCatalogMPI, CPU_StreamingCollection) { proveStreamingCollection(DeviceType::CPU); }
TEST(PlanningKernelServiceCatalogMPI, CPU_CommunicationService) { proveCommunicationService(DeviceType::CPU); }

/**
 * @test Authenticate projection policy even when it changes no CPU-only sample geometry.
 *
 * A GPU TP request can require counted mapped-host publication only for the
 * projection-owned policy. Participants must agree on that policy before any
 * measurement starts, rather than accidentally agreeing because their current
 * inventory produces the same empty GPU sample lists. The matched collection
 * afterward proves rejection leaves the MPI transaction usable.
 */
TEST(PlanningKernelServiceCatalogMPI, CPU_ProjectionPolicyDisagreementIsCollective)
{
    const auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    ASSERT_GE(mpi->world_size(), 2);
    const AutomaticOrchestrationRequest request({
        .only_backends = std::vector{DeviceType::CPU},
        .only_strategies = std::vector{OrchestrationStrategy::TensorParallel}});
    const std::array precisions{PlanningAllreducePrecision::FP32};
    const auto policy = mpi->rank() == 1
        ? RoutedExpertComputePolicy::GateUpOwnedDownColumns
        : RoutedExpertComputePolicy::Apportioned;
    const auto expected = PlanningCommunicationSamplePlan::resolve(
        *inventory, request, 257, 31, precisions,
        RoutedExpertComputePolicy::GateUpOwnedDownColumns);
    ASSERT_TRUE(expected.native().empty());
    ASSERT_TRUE(expected.hostDevice().empty());
    EXPECT_THROW(PlanningCommunicationService::collect(
        mpi, *inventory, request, 257, 31, precisions, policy), std::runtime_error);

    const auto service = PlanningCommunicationService::collect(
        mpi, *inventory, request, 257, 31, precisions,
        RoutedExpertComputePolicy::GateUpOwnedDownColumns);
    ASSERT_EQ(service.has_value(), mpi->is_root());
    if (service)
    {
        EXPECT_TRUE(service->native().empty());
        EXPECT_TRUE(service->hostDevice().empty());
        EXPECT_EQ(service->mpi().size(), expected.mpi().size());
    }
}
#ifdef HAVE_CUDA
TEST(PlanningKernelServiceCatalogMPI, CUDA_HostDeviceSameRankRolesWithIdlePeer)
{
    ensureNvidiaFactoryRegistered();
    proveHostDeviceSameRankRolesWithIdlePeer(DeviceType::CUDA);
}
TEST(PlanningKernelServiceCatalogMPI, CUDA_CommunicationService)
{
    ensureNvidiaFactoryRegistered();
    proveCommunicationService(DeviceType::CUDA);
}
#endif
#ifdef HAVE_ROCM
TEST(PlanningKernelServiceCatalogMPI, ROCm_HostDeviceSameRankRolesWithIdlePeer)
{
    ensureAMDFactoryRegistered();
    proveHostDeviceSameRankRolesWithIdlePeer(DeviceType::ROCm);
}
TEST(PlanningKernelServiceCatalogMPI, ROCm_CommunicationService)
{
    ensureAMDFactoryRegistered();
    proveCommunicationService(DeviceType::ROCm);
}
#endif

/** @test One malformed worker scope fails collectively, without stranding peers or poisoning reuse. */
TEST(PlanningKernelServiceCatalogMPI, CPU_StreamingWorkerFailureIsCollective)
{
    const auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    ASSERT_GE(mpi->world_size(), 2);
    const AutomaticOrchestrationRequest selection({.only_backends = std::vector{DeviceType::CPU}});
    const int workers = omp_get_max_threads();
    if (mpi->rank() == 1) omp_set_num_threads(workers == 1 ? 2 : 1);
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, selection,
        PlanningStreamingServicePlan{}), std::runtime_error);
    omp_set_num_threads(workers);
    EXPECT_NO_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, PlanningStreamingServicePlan{}));
    int loads = 0;
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, PlanningStreamingServicePlan{},
        [&](const auto &) -> PlanningLoadedKernelSample { ++loads; return std::monostate{}; }), std::runtime_error);
    EXPECT_EQ(loads, 0) << "Streaming observation must not open model source payloads";
}
#ifdef HAVE_CUDA
TEST(PlanningKernelServiceCatalogMPI, CUDA_StreamingCollection) { proveStreamingCollection(DeviceType::CUDA); }
TEST(PlanningKernelServiceCatalogMPI, CUDA_SourceFreeStreaming) { proveSourceFreeCollection(DeviceType::CUDA, PlanningStreamingServicePlan{}); }
TEST(PlanningKernelServiceCatalogMPI, CUDA_PreparedSourceCollectionReplay) {
    ensureNvidiaFactoryRegistered();provePreparedSourceCollectionReplay(DeviceType::CUDA);
}
TEST(PlanningKernelServiceCatalogMPI, CUDA_PreparedArithmeticPhaseLifecycle) {
    ensureNvidiaFactoryRegistered();provePreparedArithmeticLifecycle(DeviceType::CUDA);
}
TEST(PlanningKernelServiceCatalogMPI, CUDA_SourceFreeArithmetic) { proveSourceFreeCollection(DeviceType::CUDA, PlanningFP32ArithmeticPlan{}); }
#endif
#ifdef HAVE_ROCM
TEST(PlanningKernelServiceCatalogMPI, ROCm_StreamingCollection) { proveStreamingCollection(DeviceType::ROCm); }
TEST(PlanningKernelServiceCatalogMPI, ROCm_SourceFreeStreaming) { proveSourceFreeCollection(DeviceType::ROCm, PlanningStreamingServicePlan{}); }
TEST(PlanningKernelServiceCatalogMPI, ROCm_PreparedSourceCollectionReplay) {
    ensureAMDFactoryRegistered();provePreparedSourceCollectionReplay(DeviceType::ROCm);
}
TEST(PlanningKernelServiceCatalogMPI, ROCm_PreparedArithmeticPhaseLifecycle) {
    ensureAMDFactoryRegistered();provePreparedArithmeticLifecycle(DeviceType::ROCm);
}
TEST(PlanningKernelServiceCatalogMPI, ROCm_SourceFreeArithmetic) { proveSourceFreeCollection(DeviceType::ROCm, PlanningFP32ArithmeticPlan{}); }
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
TEST(PlanningKernelServiceCatalogMPI, Heterogeneous_GPUOnlySourceFreeCollectionWithIdleReporterRank)
{
    ensureNvidiaFactoryRegistered();
    ensureAMDFactoryRegistered();
    proveGPUOnlySourceFreeCollectionWithIdleReporterRank({DeviceType::CUDA, DeviceType::ROCm});
}
#endif

TEST(PlanningKernelServiceCatalogMPI, CPU_CompleteSourceAndExecutionEvidence)
{
    proveCollection(DeviceType::CPU);
    proveProjectionCollection(DeviceType::CPU);
}
#ifdef HAVE_CUDA
TEST(PlanningKernelServiceCatalogMPI, CUDA_CompleteSourceAndExecutionEvidence)
{
    proveCollection(DeviceType::CUDA);
    proveProjectionCollection(DeviceType::CUDA);
}
TEST(PlanningKernelServiceCatalogMPI, CUDA_BoundedRequestCostPreparation)
{
    proveRequestCostPreparation(DeviceType::CUDA);
}
#endif
#ifdef HAVE_ROCM
TEST(PlanningKernelServiceCatalogMPI, ROCm_CompleteSourceAndExecutionEvidence)
{
    proveCollection(DeviceType::ROCm);
    proveProjectionCollection(DeviceType::ROCm);
}
TEST(PlanningKernelServiceCatalogMPI, ROCm_BoundedRequestCostPreparation)
{
    proveRequestCostPreparation(DeviceType::ROCm);
}
#endif

TEST(PlanningKernelServiceCatalogMPI, CPU_BoundedRequestCostPreparation)
{
    proveRequestCostPreparation(DeviceType::CPU);
}

TEST(PlanningKernelServiceCatalogMPI, CPU_AsymmetricAssignmentAndReaderFailureAreCollective)
{
    const auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    std::unique_ptr<PlanningGGUFFixture> fixture;
    std::unique_ptr<PlanningModelSource> source;
    const auto plan = PlanningExpertSamplePublication::describe(mpi, [&] {
        fixture = std::make_unique<PlanningGGUFFixture>(true, false, GGUFTensorType::F32, 512);
        source = std::make_unique<PlanningModelSource>(fixture->path());
        return PlanningExpertSamplePlan::resolve(*source, request());
    });
    const AutomaticOrchestrationRequest cpu({.only_backends = std::vector{DeviceType::CPU}});
    const AutomaticOrchestrationRequest gpu({.only_backends = std::vector{DeviceType::CUDA}});
    int loads = 0;
    const auto load = [&](const auto &memory) { ++loads; return plan.load(*source, memory, DeviceId::cpu()); };
    const int workers = omp_get_max_threads();
    if (mpi->rank() == 1) omp_set_num_threads(workers == 1 ? 2 : 1);
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, cpu, plan, load), std::runtime_error);
    omp_set_num_threads(workers);
    EXPECT_EQ(loads, 0) << "A stale CPU expert team must fail before payload materialization";
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, mpi->rank() == 1 ? gpu : cpu,
        plan, load), std::runtime_error);
    EXPECT_EQ(loads, 0); // Reject disagreement before source I/O or device preparation.
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, cpu, plan,
        [](const auto &) -> PlanningLoadedExpertSample { throw std::runtime_error("Injected reader failure"); }), std::runtime_error);
    EXPECT_NO_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, cpu, plan, load));
    EXPECT_EQ(loads, mpi->is_root() ? 1 : 0);
}

/** @test A rank's stale prefill shape is a collective error before any native source is read. */
TEST(PlanningKernelServiceCatalogMPI, CPU_ProjectionShapeDisagreementIsCollective)
{
    const auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    std::unique_ptr<PlanningGGUFFixture> fixture;
    std::unique_ptr<PlanningModelSource> source;
    const auto matrix = PlanningMatrixSamplePublication::describe(mpi, [&] {
        fixture = std::make_unique<PlanningGGUFFixture>();
        source = std::make_unique<PlanningModelSource>(fixture->path());
        return PlanningMatrixSamplePlan::resolve(*source, {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}});
    });
    const AutomaticOrchestrationRequest selection({.only_backends = std::vector{DeviceType::CPU}});
    const PlanningProjectionServicePlan divergent(matrix, mpi->rank() == 1 ? 8 : 7);
    int loads = 0;
    const auto load = [&](const auto &memory) { ++loads; return matrix.load(*source, memory, DeviceId::cpu()); };
    const PlanningProjectionServicePlan agreed(matrix, 7);
    const int workers = omp_get_max_threads();
    if (mpi->rank() == 1) omp_set_num_threads(workers == 1 ? 2 : 1);
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, agreed, load), std::runtime_error);
    omp_set_num_threads(workers);
    EXPECT_EQ(loads, 0) << "A stale CPU projection team must fail before payload materialization";
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, divergent, load), std::runtime_error);
    EXPECT_EQ(loads, 0);
    EXPECT_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, agreed,
        [](const auto &) -> PlanningLoadedMatrixSample { throw std::runtime_error("Injected projection read failure"); }),
        std::runtime_error);
    EXPECT_NO_THROW(PlanningKernelServiceCatalog::collect(mpi, *inventory, selection, agreed, load));
    EXPECT_EQ(loads, mpi->is_root() ? 1 : 0);
}
