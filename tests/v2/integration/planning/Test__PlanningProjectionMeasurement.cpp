/**
 * @file Test__PlanningProjectionMeasurement.cpp
 * @brief All-source-format ordinary projection, TP shard and retirement witnesses.
 *
 * Tiny valid GGUF payloads exercise the production loader, preparation, scratch
 * admission and captured GPU execution. Timings are checked for identity and
 * completed work, never against a performance threshold. Source, temporary,
 * prepared and graph allocations must retire on both success and failure.
 */
#include "planning/PlanningProjectionMeasurement.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "utils/CPUFeatures.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include <gtest/gtest.h>
#include <omp.h>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Scope the test's physical workshare without leaking OpenMP policy. */
    struct FixedWorkers final
    {
        int workers = omp_get_max_threads(), dynamic = omp_get_dynamic();
        /** @brief Install the selected positive worker count before admission. */
        explicit FixedWorkers(int count) { omp_set_dynamic(0); omp_set_num_threads(count); }
        /** @brief All sampled CPU work completes before restoring the outer policy. */
        ~FixedWorkers() { omp_set_num_threads(workers); omp_set_dynamic(dynamic); }
    };

    /** @return Canonical complete source-format inventory; Q8_1 is not a GGUF format. */
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

    /** @return Test resource ceiling; all demand comes from the production contributor. */
    PhysicalMemoryResource resource(DeviceId device)
    {
        return {.world_rank = 0, .device = device, .total_bytes = 256u << 20,
            .admission_available_bytes = 256u << 20};
    }

    /** @return CPU policy observed after workshare selection; never attached to a GPU receipt. */
    std::optional<PlanningProjectionCPUObservation> cpuObserver(DeviceId device)
    {
        if (!device.is_cpu()) return {};
#if LLAMINAR_COMPILED_WITH_AVX512
        constexpr auto compiled = ISALevel::AVX512;
#else
        constexpr auto compiled = ISALevel::AVX2;
#endif
        return PlanningProjectionCPUObservation{omp_get_max_threads(), activeISALevel(), compiled,
            CPUExecutionGeometry::local()};
    }

    /** @return One PMA-backed sample with source loading and exact execution scratch. */
    std::shared_ptr<PhysicalMemoryAuthority> memoryFor(const PlanningMatrixSamplePlan &plan, DeviceId device)
    {
        PhysicalMemoryPlanBuilder builder;
        PlanningProjectionMeasurement::contributeMemory(plan, PlanningSampleOrigin::LocalGGUF,
            resource(DeviceId::cpu()), resource(device), builder, 7, cpuObserver(device));
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }

    /** @brief Neither physical allocations nor reserved graph capacity may escape a sample. */
    void expectRetired(const PhysicalMemoryAuthority &memory, DeviceId device)
    {
        for (auto allocator : {DeviceId::cpu(), device})
            for (auto owner : {PhysicalMemoryOwner::ModelSourcePayload, PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryOwner::WeightLoadStaging, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryOwner::NativeGraphExecutable})
            {
                EXPECT_EQ(memory.claimedBytes(allocator, owner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(memory.reservedBytes(allocator, owner), 0u);
            }
    }

    /** @brief Prepare and observe every native format as whole, output-sharded and input-sharded matrices. */
    void proveSamples(DeviceId device)
    {
        if (!hasCPUBackend()) initCPUBackend(-1);
        for (const auto &format : formats())
        {
            SCOPED_TRACE(format.label);
            const auto source_type = PlanningGGUFFixture::sourceType(format.tensor_type);
            PlanningGGUFFixture fixture(false, false, source_type, 256, source_type);
            for (const auto &name : {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight",
                "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"})
            {
                const bool down = std::string_view(name).find("down") != std::string_view::npos;
                const bool gdn = std::string_view(name).find("ssm_") != std::string_view::npos;
                auto tensor = format.create(gdn ? std::vector<size_t>{256, 256} :
                    down ? std::vector<size_t>{256, 512} : std::vector<size_t>{512, 256}, 713);
                fixture.writePayload(name, 0, {static_cast<const uint8_t *>(tensor->raw_data()), tensor->size_bytes()});
            }
            PlanningModelSource source(fixture.path());
            const std::array<PlanningModelSampleRequest, 5> requests{{
                {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}},
                {"blk.0.ffn_gate.weight", PlanningMatrixRows{64, 129}},
                {"blk.0.ffn_down.weight", PlanningMatrixColumns{256, 512}},
                {"blk.0.ssm_alpha.weight", PlanningWholeMatrix{}},
                {"blk.0.ssm_beta.weight", PlanningMatrixRows{3, 8}}}};
            for (const auto &request : requests)
            {
                const auto plan = PlanningMatrixSamplePlan::resolve(source, request);
                SCOPED_TRACE(plan.geometry().n);
                auto memory = memoryFor(plan, device);
                {
                    const auto sample = plan.load(source, memory, DeviceId::cpu());
                    // Reuse one immutable source; private packs and capture state
                    // must not accumulate across successive observations.
                    for (int repeat = 0; repeat < 2; ++repeat)
                    {
                        const auto observed = PlanningProjectionMeasurement::measure(sample, device, memory, 7);
                        EXPECT_EQ(observed.source.serialize(), plan.serialize());
                        EXPECT_EQ(observed.device, device);
                        EXPECT_EQ(bool(observed.cpu), device.is_cpu());
                        if (plan.executionType() != plan.tensorType())
                        {
                            EXPECT_EQ(plan.executionType(), TensorType::FP32);
                            if (device.is_cpu()) EXPECT_EQ(observed.prepared_bytes, 0u);
                            else EXPECT_GE(observed.prepared_bytes, plan.executionPayloadBytes());
                        }
                        if (observed.cpu) EXPECT_EQ(observed.cpu->workers, omp_get_max_threads());
                        ASSERT_EQ(observed.phases.size(), 2u);
                        EXPECT_EQ(observed.phases[0].rows, 1);
                        EXPECT_EQ(observed.phases[1].rows, 7);
                        for (const auto &phase : observed.phases)
                        {
                            EXPECT_EQ(phase.service.completedWork(),
                                3.0 * 2 * phase.rows * plan.geometry().n * plan.geometry().k);
                            EXPECT_GT(phase.service.elapsedSeconds(), 0);
                            EXPECT_EQ(phase.graph_nodes != 0, device.is_gpu());
                            EXPECT_NE(phase.service.provenance().find("repeated-same-prepared-matrix"), std::string::npos);
                            EXPECT_NE(phase.service.provenance().find("activations=FP32"), std::string::npos);
                            if (device.is_gpu()) EXPECT_NE(phase.service.provenance().find("retained graph native events"), std::string::npos);
                        }
                        EXPECT_EQ(memory->claimedBytes(device, PhysicalMemoryOwner::PrimaryModelWeights,
                            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                        EXPECT_EQ(memory->claimedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace,
                            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
                            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                    }
                }
                expectRetired(*memory, device);
            }
        }
    }

    /** @brief Promotion retires its larger host allocation even when downstream preparation fails. */
    void provePromotionFailures(DeviceId device)
    {
        PlanningGGUFFixture fixture(false, false, GGUFTensorType::Q8_0, 256, GGUFTensorType::Q8_0);
        PlanningModelSource source(fixture.path());
        const auto plan = PlanningMatrixSamplePlan::resolve(source,
            {"blk.0.ssm_alpha.weight", PlanningMatrixRows{3, 8}});
        auto memory = memoryFor(plan, device);
        ASSERT_EQ(memory->plannedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights),
            plan.executionPayloadBytes());
        {
            auto sample = plan.load(source, memory, DeviceId::cpu());
            for (const auto allocator : {DeviceId::cpu(), device})
                for (const auto owner : {PhysicalMemoryOwner::PrimaryModelWeights,
                    PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryOwner::NativeGraphExecutable})
                {
                    const auto bytes = memory->plannedBytes(allocator, owner);
                    if (!bytes) continue; // Host allocations do not include native graphs.
                    {
                        auto occupied = memory->reserveNewAllocations(allocator, owner, bytes);
                        EXPECT_THROW(PlanningProjectionMeasurement::measure(sample, device, memory, 7), std::exception);
                        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
                            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                    }
                    // One failed observation must not poison source reuse or
                    // consume capacity needed by the next prepared kernel.
                    EXPECT_NO_THROW(PlanningProjectionMeasurement::measure(sample, device, memory, 7));
                }
        }
        expectRetired(*memory, device);
    }

    /** @brief Source-free FP32 arithmetic shares preparation/capture/retirement, not a fictitious GGUF. */
    void proveArithmetic(DeviceId device)
    {
        if (!hasCPUBackend()) initCPUBackend(-1);
        const PlanningFP32ArithmeticPlan plan;
        PhysicalMemoryPlanBuilder builder;
        PlanningProjectionMeasurement::contributeMemory(plan,
            resource(DeviceId::cpu()), resource(device), builder, cpuObserver(device));
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        EXPECT_EQ(memory->plannedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ModelSourcePayload), 0u);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            const auto observed = PlanningProjectionMeasurement::measure(plan, device, memory);
            EXPECT_EQ(observed.device, device);
            EXPECT_EQ(bool(observed.cpu), device.is_cpu());
            if (observed.cpu) EXPECT_EQ(observed.cpu->workers, omp_get_max_threads());
            ASSERT_EQ(observed.phases.size(), 2u);
            for (size_t i = 0; i < observed.phases.size(); ++i)
            {
                const auto &phase = observed.phases[i];
                EXPECT_EQ(phase.rows, i == 0 ? 1 : plan.kPrefillRows);
                EXPECT_EQ(phase.graph_nodes != 0, device.is_gpu());
                EXPECT_EQ(phase.service.completedWork(), 3.0 * 2 * phase.rows * plan.kN * plan.kK);
                EXPECT_GT(phase.service.elapsedSeconds(), 0);
                EXPECT_NE(phase.service.provenance().find("not-model-weights"), std::string::npos);
                EXPECT_NE(phase.service.provenance().find("not-attention-or-GDN-timing"), std::string::npos);
            }
            expectRetired(*memory, device);
        }
        for (auto allocator : {DeviceId::cpu(), device})
            for (auto owner : {PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryOwner::NativeGraphExecutable})
            {
                const auto bytes = memory->plannedBytes(allocator, owner);
                if (!bytes) continue;
                {
                    auto occupied = memory->reserveNewAllocations(allocator, owner, bytes);
                    EXPECT_THROW(PlanningProjectionMeasurement::measure(plan, device, memory), std::exception);
                }
                expectRetired(*memory, device);
            }
        EXPECT_NO_THROW(PlanningProjectionMeasurement::measure(plan, device, memory));
        expectRetired(*memory, device);
        EXPECT_THROW(PlanningProjectionMeasurement::measure(plan, device, {}), std::invalid_argument);
        EXPECT_THROW(PlanningProjectionMeasurement::measure(plan, DeviceId::invalid(), memory), std::invalid_argument);
        expectRetired(*memory, device);
    }

    /** @brief Exhaust each allocation phase and reject malformed requests before execution. */
    void proveFailures(DeviceId device)
    {
        if (!hasCPUBackend()) initCPUBackend(-1);
        FixedWorkers workers(3);
        PlanningGGUFFixture fixture(false, false, GGUFTensorType::Q4_0);
        PlanningModelSource source(fixture.path());
        const auto plan = PlanningMatrixSamplePlan::resolve(source,
            {"blk.0.ffn_gate.weight", PlanningWholeMatrix{}});
        auto memory = memoryFor(plan, device);
        {
            auto sample = plan.load(source, memory, DeviceId::cpu());
            EXPECT_THROW(PlanningProjectionMeasurement::measure(sample, device, {}, 7), std::invalid_argument);
            EXPECT_THROW(PlanningProjectionMeasurement::measure(sample, device, memory, 0), std::invalid_argument);
            EXPECT_THROW(PlanningProjectionMeasurement::measure(sample, DeviceId::invalid(), memory, 7), std::invalid_argument);
            for (auto owner : {PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryOwner::NativeGraphExecutable})
            {
                const auto bytes = memory->plannedBytes(device, owner);
                if (!bytes) continue; // CPU has no native executable allocation.
                auto occupied = memory->reserveNewAllocations(device, owner, bytes);
                EXPECT_THROW(PlanningProjectionMeasurement::measure(sample, device, memory, 7), std::exception);
                EXPECT_EQ(memory->claimedBytes(device, PhysicalMemoryOwner::PrimaryModelWeights,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            }
            if (device.is_cpu())
            {
                omp_set_dynamic(1);
                EXPECT_THROW(PlanningProjectionMeasurement::measure(sample, device, memory, 7), std::invalid_argument);
                omp_set_dynamic(0);
            }
        }
        expectRetired(*memory, device);
        PhysicalMemoryPlanBuilder builder;
        EXPECT_THROW(PlanningProjectionMeasurement::contributeMemory(plan, PlanningSampleOrigin::LocalGGUF,
            resource(DeviceId::cpu()), resource(device), builder, -1, cpuObserver(device)), std::invalid_argument);
        provePromotionFailures(device);
    }
}

TEST(PlanningExecutionMeasurementIntegration, CPU_ProjectionSamples)
{
    for (int count : {1, 3}) { FixedWorkers workers(count); proveSamples(DeviceId::cpu()); }
}
TEST(PlanningExecutionMeasurementIntegration, CPU_ProjectionFailures) { proveFailures(DeviceId::cpu()); }
TEST(PlanningExecutionMeasurementIntegration, CPU_ProjectionArithmetic)
{
    for (int count : {1, 3}) { FixedWorkers workers(count); proveArithmetic(DeviceId::cpu()); }
}
#ifdef HAVE_CUDA
TEST(PlanningExecutionMeasurementIntegration, CUDA_ProjectionArithmetic)
{
    ensureNvidiaFactoryRegistered();
    GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0)).submitAndWait([] { proveArithmetic(DeviceId::cuda(0)); });
}
TEST(PlanningExecutionMeasurementIntegration, CUDA_ProjectionSamples)
{
    ensureNvidiaFactoryRegistered();
    auto &worker = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    worker.submitAndWait([] { proveSamples(DeviceId::cuda(0)); proveFailures(DeviceId::cuda(0)); });
}
#endif
#ifdef HAVE_ROCM
TEST(PlanningExecutionMeasurementIntegration, ROCm_ProjectionArithmetic)
{
    ensureAMDFactoryRegistered();
    GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0)).submitAndWait([] { proveArithmetic(DeviceId::rocm(0)); });
}
TEST(PlanningExecutionMeasurementIntegration, ROCm_ProjectionSamples)
{
    ensureAMDFactoryRegistered();
    auto &worker = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
    worker.submitAndWait([] { proveSamples(DeviceId::rocm(0)); proveFailures(DeviceId::rocm(0)); });
}
#endif
