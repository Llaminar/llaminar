/**
 * @file Test__PlanningGPUExpertMeasurement.cpp
 * @brief All-source-format proof of admitted, captured GPU expert observations.
 *
 * The final expert in a small GGUF supplies unequal gate/down dimensions. Both
 * GPU backends must execute every phase through the production sampler, and all
 * source, prepared, staging, workspace and graph owners must retire afterwards.
 * These are functional lifecycle tests, not throughput or model-math certificates.
 */
#include "planning/PlanningGPUExpertMeasurement.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include <gtest/gtest.h>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @return All source-backed formats; Q8_1 has no GGUF encoding. */
    std::vector<QuantizedVerifierFormatCase> sourceFormats()
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

    /** @return Last of eight source experts, not an accidentally correct zero offset. */
    PlanningExpertSampleRequest request()
    {
        return {{"blk.0.ffn_gate_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_up_exps.weight", PlanningExpertMatrix{7}},
            {"blk.0.ffn_down_exps.weight", PlanningExpertMatrix{7}}};
    }

    /** @return Exact finite test resources, admitted only by the canonical sample contributor. */
    std::shared_ptr<PhysicalMemoryAuthority> memoryFor(const PlanningModelSource &source, DeviceId device)
    {
        PhysicalMemoryPlanBuilder builder;
        PlanningGPUExpertMeasurement::contributeMemory(source, request(),
            {.world_rank = 0, .device = DeviceId::cpu(), .total_bytes = 256u << 20,
             .admission_available_bytes = 256u << 20},
            {.world_rank = 0, .device = device, .total_bytes = 256u << 20,
             .admission_available_bytes = 256u << 20}, builder);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }

    /** @brief No claim or graph-pool reservation may escape the measurement scope. */
    void expectRetired(const PhysicalMemoryAuthority &memory, DeviceId device)
    {
        for (auto allocator : {DeviceId::cpu(), device})
            for (auto owner : {PhysicalMemoryOwner::ModelSourcePayload, PhysicalMemoryOwner::RoutedExpertWeights,
                PhysicalMemoryOwner::WeightLoadStaging, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryOwner::NativeGraphExecutable})
            {
                EXPECT_EQ(memory.claimedBytes(allocator, owner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(memory.reservedBytes(allocator, owner), 0u);
            }
    }

    /** @brief Exercise actual source loading, native packing and all three captured phases. */
    void proveSamples(DeviceId device)
    {
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        worker.submitAndWait([&] {
            for (const auto &format : sourceFormats())
            {
                SCOPED_TRACE(format.label);
                PlanningGGUFFixture fixture(true, false, PlanningGGUFFixture::sourceType(format.tensor_type), 512);
                const auto sample = request();
                const auto inputs = sample.projections();
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    auto tensor = format.create(i == 2 ? std::vector<size_t>{256, 512} :
                        std::vector<size_t>{512, 256}, static_cast<uint32_t>(731 + i));
                    fixture.writePayload(inputs[i]->tensor_name, 7 * tensor->size_bytes(),
                        {static_cast<const uint8_t *>(tensor->raw_data()), tensor->size_bytes()});
                }
                PlanningModelSource source(fixture.path());
                auto memory = memoryFor(source, device);
                // Consecutive calls exercise a warm driver pool and retired
                // workspace generations without accumulating prepared engines.
                for (int repeat = 0; repeat < 2; ++repeat)
                {
                    const auto result = PlanningGPUExpertMeasurement::measure(source, sample, device, memory);
                    EXPECT_EQ(result.device, device);
                    EXPECT_EQ(result.input_width, 256u);
                    EXPECT_EQ(result.intermediate_width, 512u);
                    EXPECT_GT(result.prepared_bytes, 0u);
                    ASSERT_EQ(result.phases.size(), 3u);
                    for (const auto &phase : result.phases)
                    {
                        EXPECT_GT(phase.graph_nodes, 0u);
                        EXPECT_GT(phase.service.elapsedSeconds(), 0);
                        EXPECT_EQ(phase.service.completedWork(), 3.0 * 6 * phase.rows * 256 * 512);
                        EXPECT_NE(phase.service.provenance().find("expert=7"), std::string::npos);
                        EXPECT_NE(phase.service.provenance().find("repeated-same-prepared-expert"), std::string::npos);
                        EXPECT_NE(phase.service.provenance().find("retained graph native events"), std::string::npos);
                    }
                    expectRetired(*memory, device);
                }
            }
        });
    }

    /** @brief Reject ownership errors after source loading and after pool materialization. */
    void proveFailures(DeviceId device)
    {
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        worker.submitAndWait([&] {
            PlanningGGUFFixture fixture(true);
            PlanningModelSource source(fixture.path());
            auto memory = memoryFor(source, device);
            const auto sample = request();
            EXPECT_THROW(PlanningGPUExpertMeasurement::measure(source, sample, device, nullptr), std::invalid_argument);
            EXPECT_THROW(PlanningGPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), memory), std::invalid_argument);
            auto invalid = sample;
            invalid.down.selection = PlanningExpertMatrix{0};
            EXPECT_THROW(PlanningGPUExpertMeasurement::measure(source, invalid, device, memory), std::invalid_argument);
            for (auto owner : {PhysicalMemoryOwner::RoutedExpertWeights, PhysicalMemoryOwner::ExecutionWorkspace,
                               PhysicalMemoryOwner::NativeGraphExecutable})
            {
                SCOPED_TRACE(static_cast<int>(owner));
                {
                    auto occupied = memory->reserveNewAllocations(device, owner, memory->plannedBytes(device, owner));
                    EXPECT_THROW(PlanningGPUExpertMeasurement::measure(source, sample, device, memory), std::exception);
                }
                expectRetired(*memory, device);
            }
        });
    }
}

#ifdef HAVE_CUDA
/** @test All GGUF expert formats reach captured CUDA execution and retire safely. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_PreparedExpertSamples)
{
    ensureNvidiaFactoryRegistered();
    proveSamples(DeviceId::cuda(0));
}
/** @test Failed CUDA sample admission releases all preceding owners. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_PreparedExpertFailureRetirement)
{
    ensureNvidiaFactoryRegistered();
    proveFailures(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
/** @test All GGUF expert formats reach captured HIP execution and retire safely. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_PreparedExpertSamples)
{
    ensureAMDFactoryRegistered();
    proveSamples(DeviceId::rocm(0));
}
/** @test Failed HIP sample admission releases all preceding owners. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_PreparedExpertFailureRetirement)
{
    ensureAMDFactoryRegistered();
    proveFailures(DeviceId::rocm(0));
}
#endif
