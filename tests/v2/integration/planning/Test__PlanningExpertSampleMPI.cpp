/**
 * @file Test__PlanningExpertSampleMPI.cpp
 * @brief Exact distributed native samples, failure consensus and reader-free CPU preparation.
 *
 * Root supplies the final source expert, removes its file, and retires the
 * loader before followers prepare anything. Byte checks and PMA retirement
 * assertions cover every source format. Asymmetric failures must fail all
 * ranks before traffic, then allow a clean next transaction on the same MPI
 * context. Physical-node membership is checked independently of service time.
 */
#include "planning/PlanningExpertSample.h"
#include "planning/PlanningMatrixSample.h"
#include "planning/PlanningCPUExpertMeasurement.h"
#include "planning/PlanningGPUExpertMeasurement.h"
#include "planning/PlanningPublication.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "utils/MPIContext.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <cstring>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @return Exact source selection; nonzero expert index detects parent/offset confusion. */
    PlanningExpertSampleRequest request(size_t expert = 7)
    {
        return {{"blk.0.ffn_gate_exps.weight", PlanningExpertMatrix{expert}},
            {"blk.0.ffn_up_exps.weight", PlanningExpertMatrix{expert}},
            {"blk.0.ffn_down_exps.weight", PlanningExpertMatrix{expert}}};
    }

    /** @return All GGUF source formats from the shared functional kernel catalog, including FP. */
    std::vector<QuantizedVerifierFormatCase> formats()
    {
        auto result = quantizedVerifierFormats();
        std::erase_if(result, [](const auto &format) { return format.tensor_type == TensorType::Q8_1; });
        result.push_back({"FP32", TensorType::FP32, 0, false, 0,
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP32Random(shape, seed); }});
        result.push_back({"FP16", TensorType::FP16, 0, false, 0,
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP16Random(shape, seed); }});
        result.push_back({"BF16", TensorType::BF16, 0, false, 0,
            [](const auto &shape, uint32_t seed) { return TestTensorFactory::createBF16Random(shape, seed); }});
        return result;
    }

    /** @return Actual rank resource; the sampler contributes the only allocation calculation. */
    std::shared_ptr<PhysicalMemoryAuthority> memoryFor(const std::shared_ptr<IMPIContext> &mpi,
        const PlanningExpertSamplePlan &plan, DeviceId device = DeviceId::cpu())
    {
        const auto inventory = mpi->clusterInventory();
        const auto &rank = inventory->ranks.at(mpi->rank());
        PhysicalMemoryPlanBuilder builder;
        const PhysicalMemoryResource host{.world_rank = mpi->rank(), .device = DeviceId::cpu(),
            .total_bytes = rank.cpu.memory_bytes, .admission_available_bytes = rank.cpu.free_memory_bytes};
        const auto origin = mpi->is_root() ? PlanningSampleOrigin::LocalGGUF : PlanningSampleOrigin::PublishedPayload;
        if (device.is_gpu())
        {
            const auto found = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
                return gpu.local_device_id == device.gpu_ordinal() &&
                    gpu.type == (device.is_cuda() ? DeviceType::CUDA : DeviceType::ROCm);
            });
            if (found == rank.gpus.end()) throw std::runtime_error("GPU sample device is absent from discovery inventory");
            GPUDeviceContextPool::instance().getContext(device).submitAndWait([&] {
                PlanningGPUExpertMeasurement::contributeMemory(plan, origin, host,
                    {.world_rank = mpi->rank(), .device = device, .total_bytes = found->memory_bytes,
                        .admission_available_bytes = found->free_memory_bytes}, builder);
            });
        }
        else PlanningCPUExpertMeasurement::contributeMemory(plan, origin, host, builder,
            rank.cpu_execution, rank.cpuWorkerThreads());
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), mpi->rank());
    }

    /** @brief All allocations must retire after both successful and failed publications. */
    void expectRetired(const PhysicalMemoryAuthority &memory, DeviceId device = DeviceId::cpu())
    {
        for (auto allocator : {DeviceId::cpu(), device})
            for (auto owner : {PhysicalMemoryOwner::ModelSourcePayload, PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryOwner::RoutedExpertWeights, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryOwner::NativeGraphExecutable})
            {
                EXPECT_EQ(memory.claimedBytes(allocator, owner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(memory.reservedBytes(allocator, owner), 0u);
            }
    }

    /** @return Source-only admission for a single native matrix, without preparing any backend. */
    std::shared_ptr<PhysicalMemoryAuthority> matrixMemory(const std::shared_ptr<IMPIContext> &mpi,
        const PlanningMatrixSamplePlan &plan)
    {
        const auto &cpu = mpi->clusterInventory()->ranks.at(mpi->rank()).cpu;
        const PhysicalMemoryResource host{.world_rank = mpi->rank(), .device = DeviceId::cpu(),
            .total_bytes = cpu.memory_bytes, .admission_available_bytes = cpu.free_memory_bytes};
        PhysicalMemoryPlanBuilder builder;
        builder.add(host, PhysicalMemoryOwner::ModelSourcePayload, plan.geometry().source_bytes);
        if (mpi->is_root()) builder.add(host, PhysicalMemoryOwner::WeightLoadStaging, plan.geometry().source_bytes);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), mpi->rank());
    }
}

/** @test Every format publishes exact whole/TP-sliced bytes after the only source reader retires. */
TEST(PlanningMatrixSampleMPI, EveryNativeFormatAndSourceAxisWithoutReader)
{
    const auto mpi = MPIContextFactory::global();
    const auto topology = mpi->clusterInventory()->connectionBetweenRanks(0, mpi->rank());
    for (const auto &format : formats())
    {
        SCOPED_TRACE(format.label);
        const auto expected = format.create({256, 512}, 741);
        const auto *native = static_cast<const uint8_t *>(expected->raw_data());
        const size_t row_bytes = expected->size_bytes() / 256;
        const std::vector<PlanningModelSampleRequest> selections{
            {"blk.0.ffn_down.weight", PlanningWholeMatrix{}},
            {"blk.0.ffn_down.weight", PlanningMatrixRows{3, 8}},
            {"blk.0.ffn_down.weight", PlanningMatrixColumns{256, 512}}};
        for (size_t index = 0; index != selections.size(); ++index)
        {
            std::unique_ptr<PlanningGGUFFixture> fixture;
            std::unique_ptr<PlanningModelSource> source;
            const auto plan = PlanningMatrixSamplePublication::describe(mpi, [&] {
                fixture = std::make_unique<PlanningGGUFFixture>(false, false,
                    PlanningGGUFFixture::sourceType(format.tensor_type));
                fixture->writePayload("blk.0.ffn_down.weight", 0, {native, expected->size_bytes()});
                source = std::make_unique<PlanningModelSource>(fixture->path());
                return PlanningMatrixSamplePlan::resolve(*source, selections[index]);
            });
            const auto memory = matrixMemory(mpi, plan);
            {
                const auto sample = PlanningMatrixSamplePublication::publish(mpi, plan, memory, DeviceId::cpu(), [&] {
                    auto loaded = plan.load(*source, memory, DeviceId::cpu());
                    std::filesystem::remove(fixture->path());
                    source.reset();
                    fixture.reset();
                    return loaded;
                });
                EXPECT_FALSE(std::filesystem::exists(plan.modelPath()));
                EXPECT_EQ(sample.tensor().native_type(), format.tensor_type);
                EXPECT_EQ(sample.tensor().size_bytes(), plan.geometry().source_bytes);
                const auto *actual = static_cast<const uint8_t *>(sample.tensor().raw_data());
                if (index < 2)
                    EXPECT_EQ(std::memcmp(actual, native + (index ? 3 * row_bytes : 0), plan.geometry().source_bytes), 0);
                else for (size_t row = 0; row < 256; ++row)
                    EXPECT_EQ(std::memcmp(actual + row * row_bytes / 2,
                        native + row * row_bytes + row_bytes / 2, row_bytes / 2), 0);
                EXPECT_EQ(mpi->clusterInventory()->connectionBetweenRanks(0, mpi->rank()), topology);
            }
            expectRetired(*memory);
        }
    }
}

/** @test Asymmetric admission/read/source errors complete collectively before a clean retry. */
TEST(PlanningMatrixSampleMPI, FailedPublicationRetiresAndTwentyCleanTransactionsComplete)
{
    const auto mpi = MPIContextFactory::global();
    std::unique_ptr<PlanningGGUFFixture> fixture;
    std::unique_ptr<PlanningModelSource> source;
    const auto plan = PlanningMatrixSamplePublication::describe(mpi, [&] {
        fixture = std::make_unique<PlanningGGUFFixture>(false, false, GGUFTensorType::Q8_0);
        source = std::make_unique<PlanningModelSource>(fixture->path());
        return PlanningMatrixSamplePlan::resolve(*source, {"blk.0.ffn_down.weight", PlanningMatrixRows{3, 8}});
    });
    const auto memory = matrixMemory(mpi, plan);
    const auto load = [&] { return plan.load(*source, memory, DeviceId::cpu()); };
    for (int failed_rank : {0, 1})
        for (int defect = 0; defect < 3; ++defect)
        {
            const bool fail = mpi->rank() == failed_rank;
            PhysicalMemoryAllocationLease occupied;
            if (fail && defect == 2) occupied = memory->claimNewAllocation(DeviceId::cpu(),
                PhysicalMemoryOwner::ModelSourcePayload, plan.geometry().source_bytes);
            EXPECT_THROW(PlanningMatrixSamplePublication::publish(mpi, plan,
                fail && defect == 0 ? nullptr : memory,
                fail && defect == 1 ? DeviceId::cuda(0) : DeviceId::cpu(), load), std::runtime_error);
            occupied = {};
            expectRetired(*memory);
        }
    EXPECT_THROW(PlanningMatrixSamplePublication::publish(mpi, plan, memory, DeviceId::cpu(),
        [&]() -> PlanningLoadedMatrixSample { throw std::runtime_error("Injected matrix reader failure"); }), std::runtime_error);
    const auto other = PlanningMatrixSamplePublication::describe(mpi, [&] {
        return PlanningMatrixSamplePlan::resolve(*source, {"blk.0.ffn_down.weight", PlanningMatrixRows{4, 9}});
    });
    EXPECT_THROW(PlanningMatrixSamplePublication::publish(mpi, mpi->rank() == 1 ? other : plan,
        memory, DeviceId::cpu(), load), std::runtime_error);
    EXPECT_THROW(PlanningMatrixSamplePublication::publish(mpi, other, memory, DeviceId::cpu(), load), std::runtime_error);
    expectRetired(*memory);
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        { const auto sample = PlanningMatrixSamplePublication::publish(mpi, plan, memory, DeviceId::cpu(), load); }
        expectRetired(*memory);
    }
}

/** @brief Prove byte identity and production preparation on a reader-free CPU or GPU follower. */
void proveFileIndependentService(DeviceId follower_device)
{
    const auto mpi = MPIContextFactory::global();
    ASSERT_GE(mpi->world_size(), 2);
    const auto topology = mpi->clusterInventory()->connectionBetweenRanks(0, mpi->rank());
    const DeviceId device = mpi->rank() == 1 ? follower_device : DeviceId::cpu();
    for (const auto &format : formats())
    {
        SCOPED_TRACE(format.label);
        std::unique_ptr<PlanningGGUFFixture> fixture;
        std::unique_ptr<PlanningModelSource> source;
        // Every rank independently creates the expected native values, but only
        // root gets a file/loader. The production operation transfers full bytes.
        std::array<std::shared_ptr<TensorBase>, 3> expected;
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = format.create(i == 2 ? std::vector<size_t>{256, 512} : std::vector<size_t>{512, 256}, 751 + i);
        int descriptions = 0, loads = 0;
        const auto plan = PlanningExpertSamplePublication::describe(mpi, [&] {
            ++descriptions;
            fixture = std::make_unique<PlanningGGUFFixture>(true, false, PlanningGGUFFixture::sourceType(format.tensor_type), 512);
            for (size_t i = 0; i < expected.size(); ++i)
                fixture->writePayload(request().projections()[i]->tensor_name, 7 * expected[i]->size_bytes(),
                    {static_cast<const uint8_t *>(expected[i]->raw_data()), expected[i]->size_bytes()});
            source = std::make_unique<PlanningModelSource>(fixture->path());
            return PlanningExpertSamplePlan::resolve(*source, request());
        });
        const auto memory = memoryFor(mpi, plan, device);
        {
            const auto sample = PlanningExpertSamplePublication::publish(mpi, plan, memory, DeviceId::cpu(), [&] {
                ++loads;
                auto loaded = plan.load(*source, memory, DeviceId::cpu());
                std::filesystem::remove(fixture->path());
                source.reset();
                fixture.reset();
                return loaded;
            });
            EXPECT_EQ(descriptions, mpi->is_root() ? 1 : 0);
            EXPECT_EQ(loads, mpi->is_root() ? 1 : 0);
            EXPECT_FALSE(std::filesystem::exists(plan.modelPath()));
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ModelSourcePayload,
                PhysicalMemoryMaterializationKind::NewAllocation), plan.description().source_bytes);
            for (size_t i = 0; i < expected.size(); ++i)
            {
                EXPECT_EQ(sample.tensor(i).native_type(), format.tensor_type);
                ASSERT_EQ(sample.tensor(i).size_bytes(), expected[i]->size_bytes());
                EXPECT_EQ(std::memcmp(sample.tensor(i).raw_data(), expected[i]->raw_data(), expected[i]->size_bytes()), 0);
            }
            // Real prepared service consumes only the received owner. This is
            // not a numerical certificate, nor a performance threshold.
            exchangePlanningSamples(mpi, [&](int count) {
                return std::vector<std::vector<uint8_t>>(count, plan.serialize());
            }, [&](auto envelope) {
                EXPECT_EQ(PlanningExpertSamplePlan::deserialize(envelope).serialize(), plan.serialize());
                if (device.is_gpu())
                    GPUDeviceContextPool::instance().getContext(device).submitAndWait([&] {
                        const auto observation = PlanningGPUExpertMeasurement::measure(sample, device, memory);
                        EXPECT_EQ(observation.phases.size(), 3u);
                        for (const auto &phase : observation.phases)
                        {
                            EXPECT_GT(phase.graph_nodes, 0u);
                            EXPECT_GT(phase.service.elapsedSeconds(), 0);
                        }
                    });
                else
                {
                    const auto observation = PlanningCPUExpertMeasurement::measure(sample, device, memory);
                    EXPECT_EQ(observation.phases.size(), 3u);
                    for (const auto &phase : observation.phases) EXPECT_GT(phase.service.elapsedSeconds(), 0);
                }
                // A functional completion receipt, not a fabricated cost or
                // model benchmark. Local sampler failure reaches consensus.
                return std::vector<uint8_t>{3};
            }, [&](auto completed) {
                EXPECT_EQ(completed.size(), static_cast<size_t>(mpi->world_size()));
                for (const auto &rank : completed)
                {
                    EXPECT_EQ(rank.bytes.size(), 1u);
                    EXPECT_EQ(rank.bytes.front(), 3u);
                    EXPECT_EQ(mpi->clusterInventory()->connectionBetweenRanks(0, rank.discovery_rank).destinationRank(),
                        rank.discovery_rank);
                }
            });
            EXPECT_EQ(mpi->clusterInventory()->connectionBetweenRanks(0, mpi->rank()), topology);
        }
        expectRetired(*memory, device);
    }
}

TEST(PlanningExpertSampleMPI, CPU_EveryNativeFormatWithoutReader)
{
    proveFileIndependentService(DeviceId::cpu());
}
#ifdef HAVE_CUDA
TEST(PlanningExpertSampleMPI, CUDA_EveryNativeFormatWithoutReader)
{
    proveFileIndependentService(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
TEST(PlanningExpertSampleMPI, ROCm_EveryNativeFormatWithoutReader)
{
    proveFileIndependentService(DeviceId::rocm(0));
}
#endif

TEST(PlanningExpertSampleMPI, AsymmetricFailuresAreCollectiveAndTwentyCleanPublicationsRetire)
{
    const auto mpi = MPIContextFactory::global();
    std::unique_ptr<PlanningGGUFFixture> fixture;
    std::unique_ptr<PlanningModelSource> source;
    const auto plan = PlanningExpertSamplePublication::describe(mpi, [&] {
        fixture = std::make_unique<PlanningGGUFFixture>(true, false, GGUFTensorType::Q8_0, 512);
        source = std::make_unique<PlanningModelSource>(fixture->path());
        return PlanningExpertSamplePlan::resolve(*source, request());
    });
    const auto memory = memoryFor(mpi, plan);
    const auto root_load = [&] { return plan.load(*source, memory, DeviceId::cpu()); };
    for (int failed_rank : {0, 1})
        for (int defect = 0; defect < 3; ++defect)
        {
            const bool fail = mpi->rank() == failed_rank;
            PhysicalMemoryAllocationLease occupied;
            if (fail && defect == 2)
                occupied = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ModelSourcePayload,
                    plan.description().source_bytes);
            EXPECT_THROW(PlanningExpertSamplePublication::publish(mpi, plan,
                fail && defect == 0 ? nullptr : memory,
                fail && defect == 1 ? DeviceId::cuda(0) : DeviceId::cpu(), root_load), std::runtime_error);
            occupied = {};
            expectRetired(*memory);
        }
    EXPECT_THROW(PlanningExpertSamplePublication::publish(mpi, plan, memory, DeviceId::cpu(), [&]() -> PlanningLoadedExpertSample {
        throw std::runtime_error("Injected root reader failure");
    }), std::runtime_error);
    expectRetired(*memory);
    const auto other = PlanningExpertSamplePublication::describe(mpi, [&] {
        return PlanningExpertSamplePlan::resolve(*source, request(6));
    });
    EXPECT_THROW(PlanningExpertSamplePublication::publish(mpi, mpi->rank() == 1 ? other : plan,
        memory, DeviceId::cpu(), root_load), std::runtime_error);
    EXPECT_THROW(PlanningExpertSamplePublication::publish(mpi, other, memory, DeviceId::cpu(), root_load), std::runtime_error);
    expectRetired(*memory);
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        { const auto sample = PlanningExpertSamplePublication::publish(mpi, plan, memory, DeviceId::cpu(), root_load); }
        expectRetired(*memory);
    }
}
