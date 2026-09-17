/**
 * @file Test__PlanningKernelServiceCatalogMPI.cpp
 * @brief Real all-rank collection through admitted source publication and captured service.
 *
 * The production collector chooses reporters from actual discovery, including
 * shared GPU visibility. Only root opens the tiny source fixture; the loaded
 * reader retires before preparation. Functional assertions prove complete CPU,
 * CUDA and ROCm evidence and collective rejection, not a performance threshold.
 */
#include "planning/PlanningKernelServiceCatalog.h"
#include "planning/PlanningWeightServiceModel.h"
#include "planning/PlanningForwardStateWork.h"
#include "planning/PlanningCommunicationService.h"
#include "planning/PlanningCommunicationCost.h"
#include "planning/AutomaticPlanningStartup.h"
#include "backends/GPUDeviceContextPool.h"
#include "utils/MPIContext.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
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
        const PlanningCommunicationCost costs(*inventory, service->native(), service->mpi(), service->hostDevice());
        for (size_t index = 0; index < service->hostDevice().size(); ++index)
        {
            const auto &observed = service->hostDevice()[index];
            ASSERT_EQ(observed.request, expected.hostDevice()[index]);
            EXPECT_NO_THROW(PlanningHostDeviceMeasurement::validate(observed));
            for (const auto &phase : observed.phases)
            {
                const auto estimate = costs.hostTransfer(observed.request.rank(), observed.request.device(),
                    observed.request.rank(), phase.mechanism, phase.direction, phase.payload_bytes);
                EXPECT_GT(estimate.seconds, 0);
                EXPECT_EQ(estimate.basis, PlanningCommunicationBasis::SameProtocolPayloadCurve);
                std::cout << "HOST_DEVICE_SAMPLE device=" << observed.request.device().toString()
                    << " rank=" << observed.request.rank() << " host_numa=" << observed.request.hostNumaNode()
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
                        EXPECT_EQ(cost.evidence().find("mean_decode_interconnect_s=0;"), std::string::npos);
                    }
                    if (candidate.strategy() == OrchestrationStrategy::ExpertOverlay) ++overlay_prices;
                    return cost;
                };
            });
        EXPECT_EQ(result.root_selection.has_value(), mpi->is_root());
        if (mpi->is_root())
        {
            EXPECT_GT(prices, 0u);
            EXPECT_GT(multi_device_prices, 0u);
            if (backend != DeviceType::CPU) EXPECT_GT(overlay_prices, 0u);
        }
        else EXPECT_EQ(prices, 0u);
        EXPECT_TRUE(result.applied.execution_rank_selection);
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
        {
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
                    ASSERT_EQ(observation.phases.size(), 3u);
                    EXPECT_EQ(observation.formats, plan.description().formats);
                    for (const auto &phase : observation.phases)
                    {
                        EXPECT_GT(phase.service.elapsedSeconds(), 0);
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
#ifdef HAVE_CUDA
TEST(PlanningKernelServiceCatalogMPI, CUDA_CommunicationService)
{
    ensureNvidiaFactoryRegistered();
    proveCommunicationService(DeviceType::CUDA);
}
#endif
#ifdef HAVE_ROCM
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
#endif
#ifdef HAVE_ROCM
TEST(PlanningKernelServiceCatalogMPI, ROCm_StreamingCollection) { proveStreamingCollection(DeviceType::ROCm); }
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
