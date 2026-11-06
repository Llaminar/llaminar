/**
 * @file Test__W4A4PlanningSamples.cpp
 * @brief Native online-A4 sample ownership, original-K identity and observation contracts.
 *
 * Real source reads, sealed publication, factory preparation and PMA accounting
 * execute without a model or accelerator. Unselected source rows remain part
 * of policy admission; received samples cannot invent a local activation unit.
 * Timings prove completed production work, not an economy or model certificate.
 */
#include "planning/PlanningMatrixSample.h"
#include "planning/PlanningProjectionMeasurement.h"
#include "planning/PlanningKernelServiceCatalog.h"
#include "kernels/KernelFactory.h"
#include "tensors/W4A4Tensor.h"
#include "backends/BackendManager.h"
#include "utils/CPUFeatures.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/W4A4SourceFixture.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <omp.h>

namespace llaminar2
{
    using llaminar::v2::kernels::KernelFactory;

    namespace
    {
        constexpr auto kSourceOwner = PhysicalMemoryOwner::ModelSourcePayload;
        constexpr const char *kName = "blk.0.ffn_down.weight";
        constexpr size_t kN = 256, kK = 512;

        /** @brief Restore the observing thread's numerical worker policy after the fixture. */
        struct Workers
        {
            int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
            Workers() { omp_set_dynamic(0); omp_set_num_threads(1); }
            ~Workers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
        };

        /** @return Source-only admission; no second pack or reader buffer can fit. */
        std::shared_ptr<PhysicalMemoryAuthority> admit(size_t bytes)
        {
            if (!hasCPUBackend()) initCPUBackend(-1);
            PhysicalMemoryPlanBuilder builder;
            builder.add({0, DeviceId::cpu(), bytes, bytes}, kSourceOwner, bytes);
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        }

        /** @brief Populate one complete original matrix, leaving the fixture directory unchanged. */
        void populate(test::PlanningGGUFFixture &fixture, std::span<const w4a4::SourceBlock> blocks,
            const std::string &name = kName)
        {
            fixture.writePayload(name, 0,
                {reinterpret_cast<const uint8_t *>(blocks.data()), blocks.size_bytes()});
        }

        /** @return Current exact source allocation claim, not a separate accounting estimate. */
        size_t claimed(const PhysicalMemoryAuthority &memory, PhysicalMemoryOwner owner = kSourceOwner)
        {
            return memory.claimedBytes(DeviceId::cpu(), owner, PhysicalMemoryMaterializationKind::NewAllocation);
        }
    }

    TEST(W4A4PlanningSamples, SealedRowsRetainOneSourceOwnerAfterReaderRetirement)
    {
        Workers workers;
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
            for (const auto &selection : std::vector<PlanningModelSampleRequest>{
                    {kName, PlanningWholeMatrix{}}, {kName, PlanningMatrixRows{3, 8}},
                    {kName, PlanningMatrixColumns{0, kK}}})
            {
                test::PlanningGGUFFixture fixture(false, false, test::PlanningGGUFFixture::sourceType(type));
                const auto blocks = test::w4a4SourceBlocks(type, kN, kK);
                populate(fixture, blocks);
                auto source = std::make_unique<PlanningModelSource>(fixture.path());
                const auto plan = PlanningMatrixSamplePlan::resolve(*source, selection);
                EXPECT_EQ(plan.originalColumns(), kK);
                EXPECT_EQ(plan.executionType(), type);
                const auto received = PlanningMatrixSamplePlan::deserialize(plan.serialize());
                EXPECT_EQ(received.serialize(), plan.serialize());
                auto memory = admit(received.geometry().physicalBytes());
                std::optional sample(PlanningMatrixSamplePublication::publish(nullptr, received, memory,
                    DeviceId::cpu(), [&] {
                        auto loaded = received.load(*source, memory, DeviceId::cpu());
                        source.reset();
                        std::filesystem::remove(fixture.path());
                        return loaded;
                    }));
                const auto &tensor = sample->tensor();
                const auto *native = dynamic_cast<const W4A4Tensor *>(&tensor);
                ASSERT_NE(native, nullptr);
                EXPECT_FALSE(tensor.is_mmap_data());
                EXPECT_THROW(const_cast<TensorBase &>(tensor).raw_mutable_data(), std::logic_error);
                const auto first = std::holds_alternative<PlanningMatrixRows>(selection.selection) ? 3u : 0u;
                EXPECT_EQ(std::memcmp(tensor.raw_data(), blocks.data() + first * kK / w4a4::kBlockElements,
                    plan.geometry().source_bytes), 0);
                EXPECT_EQ(claimed(*memory), received.geometry().physicalBytes());
                EXPECT_EQ(claimed(*memory, PhysicalMemoryOwner::WeightLoadStaging), 0u);
                EXPECT_EQ(claimed(*memory, PhysicalMemoryOwner::PrimaryModelWeights), 0u);
                auto engine = KernelFactory::createGemm(native, DeviceType::CPU, -1);
                sample.reset();
                EXPECT_EQ(engine->packedWeightBytes(), received.geometry().source_bytes);
                EXPECT_EQ(claimed(*memory), received.geometry().physicalBytes());
                engine.reset();
                EXPECT_EQ(claimed(*memory), 0u);
            }
    }

    TEST(W4A4PlanningSamples, OriginalKAndUnselectedSourcePolicyCannotBeNarrowedAway)
    {
        Workers workers;
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            test::PlanningGGUFFixture fixture(false, false, test::PlanningGGUFFixture::sourceType(type));
            auto blocks = test::w4a4SourceBlocks(type, kN, kK);
            populate(fixture, blocks);
            PlanningModelSource source(fixture.path());
            const auto plan = PlanningMatrixSamplePlan::resolve(source, {kName, PlanningMatrixRows{3, 6}});
            auto memory = admit(plan.geometry().physicalBytes());
            EXPECT_THROW(PlanningMatrixSamplePlan::resolve(source, {kName, PlanningMatrixColumns{0, 256}}), std::invalid_argument);
            EXPECT_THROW(PlanningMatrixSamplePlan::resolve(source, {kName, PlanningMatrixColumns{256, 512}}), std::invalid_argument);
            for (int defect = 0; defect < 8; ++defect)
            {
                auto wire = nlohmann::json::parse(plan.serialize());
                if (defect == 0) wire.erase("original_k");
                if (defect == 1) wire["schema"] = "llaminar.planning-matrix-sample.v3";
                if (defect == 2) wire["original_k"] = 0;
                if (defect == 3) wire["original_k"] = 256;
                if (defect == 4) wire["original_k"] = 1024;
                if (defect == 5) wire["original_k"] = -1;
                if (defect == 6) wire["selection"] = {{"kind", "expert"}, {"index", 0}};
                if (defect == 7)
                {
                    wire["selection"] = {{"kind", "columns"}, {"first", 256}, {"last", 512}};
                    wire["matrix"]["k"] = 256;
                    wire["matrix"]["bytes"] = plan.geometry().source_bytes / 2;
                }
                const auto text = wire.dump();
                EXPECT_THROW(PlanningMatrixSamplePlan::deserialize({
                    reinterpret_cast<const uint8_t *>(text.data()), text.size()}), std::exception) << defect;
            }
            for (bool rotation : {false, true})
            {
                blocks = test::w4a4SourceBlocks(type, kN, kK);
                for (size_t block = blocks.size() - 2; block < blocks.size(); ++block)
                    if (rotation) blocks[block].rotation_length = 64;
                    else blocks[block].clip_bits = std::bit_cast<uint32_t>(0.5f);
                populate(fixture, blocks);
                try
                {
                    static_cast<void>(plan.load(source, memory, DeviceId::cpu()));
                    FAIL() << "Unselected row changed the source policy";
                }
                catch (const std::invalid_argument &error)
                {
                    EXPECT_NE(std::string(error.what()).find("original row 255"), std::string::npos);
                }
                EXPECT_EQ(claimed(*memory), 0u);
                EXPECT_EQ(claimed(*memory, PhysicalMemoryOwner::WeightLoadStaging), 0u);
            }
            const std::string last = "blk.1.ffn_down.weight";
            populate(fixture, test::w4a4SourceBlocks(type, kN, kK), last);
            const auto truncated = PlanningMatrixSamplePlan::resolve(source, {last, PlanningMatrixRows{3, 6}});
            std::filesystem::resize_file(fixture.path(), std::filesystem::file_size(fixture.path()) - 1);
            EXPECT_THROW(truncated.load(source, memory, DeviceId::cpu()), std::exception);
            EXPECT_EQ(claimed(*memory), 0u);
        }
    }

    TEST(W4A4PlanningSamples, ProductionProjectionMeasurementBorrowsTheOnlyNativePayload)
    {
        Workers workers;
        if (!hasCPUBackend()) initCPUBackend(-1);
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            test::PlanningGGUFFixture fixture(false, false, test::PlanningGGUFFixture::sourceType(type));
            populate(fixture, test::w4a4SourceBlocks(type, kN, kK));
            PlanningModelSource source(fixture.path());
            const auto plan = PlanningMatrixSamplePlan::resolve(source, {kName, PlanningMatrixRows{3, 8}});
            constexpr size_t ceiling = 4u << 20;
            const PhysicalMemoryResource host{0, DeviceId::cpu(), ceiling, ceiling};
#if LLAMINAR_COMPILED_WITH_AVX512
            constexpr auto compiled = ISALevel::AVX512;
#else
            constexpr auto compiled = ISALevel::AVX2;
#endif
            const PlanningProjectionCPUObservation cpu{1, activeISALevel(), compiled, CPUExecutionGeometry::local()};
            PhysicalMemoryPlanBuilder builder;
            PlanningProjectionMeasurement::contributeMemory(plan, PlanningSampleOrigin::LocalGGUF,
                host, host, builder, 7, cpu);
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
            {
                const auto sample = plan.load(source, memory, DeviceId::cpu());
                for (int repeat = 0; repeat < 2; ++repeat)
                {
                    const auto result = PlanningProjectionMeasurement::measure(sample, DeviceId::cpu(), memory, 7);
                    EXPECT_EQ(result.source.serialize(), plan.serialize());
                    EXPECT_EQ(result.prepared_bytes, 0u);
                    ASSERT_TRUE(result.cpu);
                    EXPECT_EQ(result.cpu->workers, 1);
                    ASSERT_EQ(result.phases.size(), 2u);
                    for (const auto &phase : result.phases)
                    {
                        EXPECT_EQ(phase.service.completedWork(), 3.0 * 2 * phase.rows * plan.geometry().n * plan.geometry().k);
                        EXPECT_GT(phase.service.elapsedSeconds(), 0.0);
                    }
                    const PlanningProjectionServicePlan service_plan(plan, 7);
                    const std::array<PlanningKernelObservation, 1> observations{result};
                    EXPECT_NO_THROW(PlanningKernelServiceCatalog::encode(service_plan, observations));
                    auto duplicated_pack = result;
                    duplicated_pack.prepared_bytes = plan.geometry().physicalBytes();
                    const std::array<PlanningKernelObservation, 1> invalid{duplicated_pack};
                    EXPECT_THROW(PlanningKernelServiceCatalog::encode(service_plan, invalid), std::invalid_argument);
                    EXPECT_EQ(claimed(*memory), plan.geometry().physicalBytes());
                    for (auto owner : {PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryOwner::ExecutionWorkspace,
                                       PhysicalMemoryOwner::WeightLoadStaging})
                        EXPECT_EQ(claimed(*memory, owner), 0u);
                }
            }
            EXPECT_EQ(claimed(*memory), 0u);
        }
    }
}
