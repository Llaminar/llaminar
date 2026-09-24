/**
 * @file Test__PlanningCPUExpertMeasurement.cpp
 * @brief Source-backed production CPU FFN and allocation-retirement regressions.
 *
 * Valid native tensors are written into the final expert of a tiny GGUF. The
 * production sampler must load that triplet, use normal expert preparation and
 * complete the existing grouped FFN at all three phases. These are lifecycle
 * witnesses, not real-model throughput certificates or performance thresholds.
 */
#include "planning/PlanningCPUExpertMeasurement.h"
#include "backends/BackendManager.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <optional>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Restore fixed worker policy even when a negative observation throws. */
    struct FixedWorkers final
    {
        int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
        /** @brief Install only this test's fixed, explicitly requested worker count. */
        explicit FixedWorkers(int count) { omp_set_dynamic(0); omp_set_num_threads(count); }
        /** @brief Restore the surrounding test process after the workshare retires. */
        ~FixedWorkers() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
    };

    /** @return One final expert, exposing coalesced-parent offset mistakes. */
    PlanningExpertSampleRequest request()
    {
        return {{"blk.0.ffn_gate_exps.weight", PlanningExpertMatrix{7}},
                {"blk.0.ffn_up_exps.weight", PlanningExpertMatrix{7}},
                {"blk.0.ffn_down_exps.weight", PlanningExpertMatrix{7}}};
    }

    /** @brief A finite test allocator; only the canonical contributor supplies its BOM. */
    std::shared_ptr<PhysicalMemoryAuthority> memoryFor(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &sample)
    {
        // Reproduce frontend CPU allocator setup without probing accelerators.
        if (!hasCPUBackend()) initCPUBackend(-1);
        PhysicalMemoryPlanBuilder builder;
        PlanningCPUExpertMeasurement::contributeMemory(source, sample,
            {.world_rank = 0, .device = DeviceId::cpu(), .total_bytes = 64u << 20,
             .admission_available_bytes = 64u << 20}, builder, CPUExecutionGeometry::local(), omp_get_max_threads());
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }

    /** @brief Every payload owner must retire on success and on preparation failure. */
    void expectRetired(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        for (const auto owner : {PhysicalMemoryOwner::ModelSourcePayload, PhysicalMemoryOwner::RoutedExpertWeights,
                                PhysicalMemoryOwner::WeightLoadStaging, PhysicalMemoryOwner::ExecutionWorkspace})
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), owner,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}

TEST(PlanningExecutionMeasurementIntegration, CPU_PreparedExpertSamples)
{
    auto formats = quantizedVerifierFormats();
    formats.push_back({"FP32", TensorType::FP32, 0, false, 0,
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP32Random(shape, seed); }});
    formats.push_back({"FP16", TensorType::FP16, 0, false, 0,
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP16Random(shape, seed); }});
    formats.push_back({"BF16", TensorType::BF16, 0, false, 0,
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createBF16Random(shape, seed); }});
    for (const auto &format : formats)
    {
        if (format.tensor_type == TensorType::Q8_1)
        {
            // Q8_1 is runtime-only: prove its exact preparation demand without
            // inventing a GGUF encoding or claiming source-loading coverage.
            FixedWorkers budget(3);
            using llaminar::v2::kernels::KernelFactory;
            const auto tensor = format.create({65, 256}, 731);
            const auto engine = KernelFactory::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
            ASSERT_NE(engine, nullptr);
            EXPECT_EQ(engine->packedWeightBytes(),
                CPUWeightPreparationMemory::sourceNative("Q8_1", 65, 256).persistent_bytes);
            continue;
        }
        SCOPED_TRACE(format.label);
        PlanningGGUFFixture fixture(true, false, PlanningGGUFFixture::sourceType(format.tensor_type), 512);
        const auto sample = request();
        const std::array inputs{sample.gate, sample.up, sample.down};
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto tensor = format.create(i == 2 ? std::vector<size_t>{256, 512} : std::vector<size_t>{512, 256},
                static_cast<uint32_t>(731 + i));
            fixture.writePayload(inputs[i].tensor_name, 7 * tensor->size_bytes(),
                {static_cast<const uint8_t *>(tensor->raw_data()), tensor->size_bytes()});
        }
        PlanningModelSource source(fixture.path());
        auto memory = memoryFor(source, sample);
        for (int workers : {1, 3})
        {
            FixedWorkers budget(workers);
            const auto observed = PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), memory);
            EXPECT_EQ(observed.worker_threads, workers);
            EXPECT_EQ(observed.input_width, 256u);
            EXPECT_EQ(observed.intermediate_width, 512u);
            EXPECT_EQ(observed.source.gate.tensor_name, sample.gate.tensor_name);
            EXPECT_EQ(std::get<PlanningExpertMatrix>(observed.source.gate.selection).index, 7u);
            EXPECT_GT(observed.prepared_bytes, 0u);
            ASSERT_EQ(observed.phases.size(), 3u);
            for (const auto &phase : observed.phases)
            {
                EXPECT_EQ(phase.rows, MoEOverlayCPUServiceMeasurement::rowsForPhase(phase.phase));
                EXPECT_GT(phase.service.elapsedSeconds(), 0);
                EXPECT_EQ(phase.service.completedWork(), 3.0 * 6 * phase.rows * 256 * 512);
                EXPECT_NE(phase.service.provenance().find("repeated-same-prepared-expert"), std::string::npos);
            }
            expectRetired(memory);
        }
    }
}

TEST(PlanningExecutionMeasurementIntegration, CPU_PreparedExpertFailureRetirement)
{
    FixedWorkers budget(3);
    PlanningGGUFFixture fixture(true, false, GGUFTensorType::Q4_0, 512);
    PlanningModelSource source(fixture.path());
    const auto sample = request();
    auto memory = memoryFor(source, sample);
    // Occupy the entire prepared line; source loading succeeds, preparation
    // admission fails, and all sampler-owned bytes must still retire.
    const auto capacity = memory->remainingAdmittedNewAllocationBytes(DeviceId::cpu(), PhysicalMemoryOwner::RoutedExpertWeights);
    {
        auto occupied = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::RoutedExpertWeights, capacity);
        EXPECT_THROW(PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), memory), std::exception);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ModelSourcePayload,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
    expectRetired(memory);
    EXPECT_NO_THROW(PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), memory));
    expectRetired(memory);
    {
        // Fail after all three matrices were successfully prepared. The probe's
        // workspace admission must not leak those engines or their source slabs.
        auto occupied = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
            memory->remainingAdmittedNewAllocationBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace));
        EXPECT_THROW(PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), memory), std::exception);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::RoutedExpertWeights,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
    expectRetired(memory);
    auto invalid = sample;
    invalid.up.selection = PlanningExpertMatrix{0};
    EXPECT_THROW(PlanningCPUExpertMeasurement::measure(source, invalid, DeviceId::cpu(), memory), std::invalid_argument);
    EXPECT_THROW(PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cuda(0), memory), std::invalid_argument);
    EXPECT_THROW(PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), {}), std::invalid_argument);
    omp_set_dynamic(1);
    EXPECT_THROW(PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), memory), std::invalid_argument);
    omp_set_dynamic(0);
    expectRetired(memory);
}

TEST(PlanningExecutionMeasurementIntegration, CPU_PreparedExpertPartialReadRetirement)
{
    FixedWorkers budget(3);
    PlanningGGUFFixture fixture(true, false, GGUFTensorType::Q4_0, 512);
    PlanningModelSource source(fixture.path());
    const auto sample = request();
    auto memory = memoryFor(source, sample);
    const auto &directory = source.loader().getModel();
    const auto *down = directory.findTensor(sample.down.tensor_name);
    ASSERT_NE(down, nullptr);
    // Preserve the first two source parents, then interrupt the third expert
    // read. The fixture owns this exact temporary file and removes it on exit.
    std::filesystem::resize_file(fixture.path(), directory.data_offset + down->offset + down->size_bytes - 1);
    EXPECT_THROW(PlanningCPUExpertMeasurement::measure(source, sample, DeviceId::cpu(), memory), std::exception);
    expectRetired(memory);
}

TEST(PlanningExecutionMeasurementIntegration, CPU_FloatingPreparedAccounting)
{
    using llaminar::v2::kernels::KernelFactory;
    for (auto tensor : {std::shared_ptr<TensorBase>(TestTensorFactory::createFP32Random({65, 256}, 3)),
                        std::shared_ptr<TensorBase>(TestTensorFactory::createFP16Random({65, 256}, 3)),
                        std::shared_ptr<TensorBase>(TestTensorFactory::createBF16Random({65, 256}, 3))})
    {
        const auto borrowed = KernelFactory::prepareGemmHandleLocal(tensor.get(), DeviceId::cpu());
        EXPECT_EQ(KernelFactory::getOrCreateGemmEngine(borrowed.get())->packedWeightBytes(), 0u);
        auto prepared = KernelFactory::prepareExpertGemmLocal(std::shared_ptr<const TensorBase>(tensor), DeviceId::cpu());
        ASSERT_NE(prepared, nullptr);
        EXPECT_EQ(prepared->packedWeightBytes(), tensor->size_bytes());
        auto view = KernelFactory::createExpertServiceExecutionView(prepared, DeviceId::cpu());
        ASSERT_NE(view, nullptr);
        EXPECT_EQ(view->packedWeightBytes(), 0u); // A view does not allocate a second tensor.
        EXPECT_TRUE(view->canReleaseSourceWeightTensor());
        prepared.reset();
        ContiguousFloatingPointWeightDescriptor retained;
        EXPECT_TRUE(view->exportContiguousFloatingPointWeights(retained));
        EXPECT_EQ(retained.bytes, tensor->size_bytes());
    }
}
