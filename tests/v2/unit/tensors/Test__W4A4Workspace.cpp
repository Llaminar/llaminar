/**
 * @file Test__W4A4Workspace.cpp
 * @brief Native W4A4 scratch declarations and production arena binding contracts.
 *
 * The real CPU backend, DeviceWorkspaceManager and PhysicalMemoryAuthority own
 * every workspace byte. Tests vary live rows beneath one retained envelope,
 * execute the production arithmetic and verify that invocation borrowing never
 * allocates or changes storage identity. No model or accelerator is involved.
 */
#include "kernels/cpu/gemm/CPUW4A4Workspace.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "backends/BackendManager.h"
#include "utils/Assertions.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstring>
#include <vector>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr std::size_t kCapacity = 64 * 1024;
        constexpr auto kOwner = PhysicalMemoryOwner::ExecutionWorkspace;

        /** @return Bounded CPU workspace admission, without hardware capacity probes. */
        std::shared_ptr<PhysicalMemoryAuthority> workspaceAuthority()
        {
            PhysicalMemoryPlanBuilder builder;
            builder.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = kCapacity, .admission_available_bytes = kCapacity}, kOwner, kCapacity);
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        }

        /** @brief Small native policy whose lifetime remains inside this fixture. */
        class Fixture
        {
        public:
            std::array<SourceBlock, 2> blocks{}; ///< Two complete blocks in one output row.
            /** @brief Initialize a native matrix without weight conversion. */
            Fixture()
            {
                for (auto &block : blocks)
                {
                    block.unit = 0.125f;
                    block.descriptor = 0x60001111;
                    block.rotation_length = 512;
                    std::fill_n(block.codes, 128, std::uint8_t{0x11});
                }
            }
            /** @return Exact admitted source view, or a test-construction failure. */
            SourceView weights(Transform transform) const
            {
                const auto result = SourceView::admit(transform, 1, 512, blocks);
                if (!result) throw std::logic_error(result.error().message());
                return *result;
            }
        };
    }

    TEST(W4A4WorkspaceContract, TypedBOMDeclaresOnlyTheSelectedTransformBanks)
    {
        const Fixture fixture;
        for (const auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            const auto weights = fixture.weights(transform);
            EXPECT_TRUE(cpuWorkspaceRequirements(weights.contract(), 0).empty());
            const auto empty = borrowCPUWorkspace(weights.contract(), 0, nullptr);
            EXPECT_TRUE(empty.blocks.empty());
            const auto requirements = cpuWorkspaceRequirements(weights.contract(), 5);
            EXPECT_EQ(requirements.buffers.size(), transform == Transform::Identity ? 4u : 6u);
            const auto counts = cpuWorkspaceCounts(weights.contract(), 5);
            EXPECT_EQ(requirements.buffers[0].size_bytes, counts.blocks * sizeof(ActivationBlock));
            EXPECT_EQ(requirements.buffers[1].size_bytes, counts.ranges * sizeof(ActivationRange));
            EXPECT_EQ(requirements.buffers[2].size_bytes, counts.row_ranges * sizeof(float));
            for (const auto &bank : requirements.buffers)
            {
                EXPECT_TRUE(bank.required);
                EXPECT_EQ(bank.alignment, 64u);
                EXPECT_EQ(bank.content_lifetime, WorkspaceContentLifetime::ParticipantExecution);
            }
            EXPECT_THROW(static_cast<void>(borrowCPUWorkspace(weights.contract(), 1, nullptr)), std::runtime_error);
        }
    }

    TEST(W4A4WorkspaceContract, SourcePricingMatchesPreparedBanksAndMergesSequentialLayers)
    {
        const Fixture fixture;
        for (auto type : {TensorType::W4A4, TensorType::W4A4H})
        {
            const auto transform = type == TensorType::W4A4 ? Transform::Identity : Transform::PaleySylvester;
            const auto weights = fixture.weights(transform);
            for (int rows : {1, 2, 15, 31, 65})
            {
                const auto actual = cpuWorkspaceRequirements(weights.contract(), rows);
                const auto quoted = CPUProjectionWorkspaceContract::sourceNative(type,
                    {.rows = rows, .n = 1, .k = 512});
                ASSERT_EQ(actual.buffers.size(), quoted.buffers.size());
                for (std::size_t i = 0; i < actual.buffers.size(); ++i)
                {
                    EXPECT_EQ(actual.buffers[i].name, quoted.buffers[i].name);
                    EXPECT_EQ(actual.buffers[i].size_bytes, quoted.buffers[i].size_bytes);
                    EXPECT_EQ(actual.buffers[i].alignment, quoted.buffers[i].alignment);
                }
            }
            ModelMemoryProfile profile;
            profile.architecture = "qwen35";
            profile.n_layers = 2;
            profile.d_model = 512;
            profile.d_ff = 1024;
            profile.vocab_size = 1024;
            WorkspaceMemoryGeometry geometry{.device = DeviceId::cpu(), .device_compute_units = 1,
                .cpu_execution = test::kSyntheticCPUExecutionGeometry, .batch_size = 2,
                .resident_graph_rows = 15, .local_d_ff = 1024, .first_layer = 0, .last_layer = 1};
            const auto baseline = WorkspaceMemoryEstimator::estimate(profile, geometry);
            profile.tensors = {
                {.name = "blk.0.ffn_gate.weight", .quant_type = tensorTypeName(type),
                    .elements = 512u * 1024u, .K = 512, .layer_index = 0},
                {.name = "blk.0.ffn_up.weight", .quant_type = tensorTypeName(type),
                    .elements = 512u * 1024u, .K = 512, .layer_index = 0},
                {.name = "blk.0.ffn_down.weight", .quant_type = tensorTypeName(type),
                    .elements = 512u * 1024u, .K = 1024, .layer_index = 0}};
            auto expected = cpuWorkspaceRequirements(transform, 1024, 30);
            expected.merge(cpuSwiGLUWorkspaceRequirements(30, 1024));
            expected.merge(cpuFusedProjectionWorkspaceRequirements(2));
            const auto one = WorkspaceMemoryEstimator::estimate(profile, geometry);
            EXPECT_EQ(one - baseline, expected.total_bytes_with_alignment());
            const auto layer = profile.tensors;
            for (auto tensor : layer)
            {
                tensor.name.replace(4, 1, "1");
                tensor.layer_index = 1;
                profile.tensors.push_back(std::move(tensor));
            }
            EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry), one);
            geometry.total_shards = 2;
            geometry.local_d_ff = 512;
            EXPECT_THROW(static_cast<void>(WorkspaceMemoryEstimator::estimate(profile, geometry)), std::invalid_argument);
        }
        EXPECT_TRUE(cpuFusedProjectionWorkspaceRequirements(1).empty());
        const auto bundle = cpuFusedProjectionWorkspaceRequirements(2);
        ASSERT_EQ(bundle.buffers.size(), 1u);
        EXPECT_EQ(bundle.buffers.front().size_bytes, 2 * sizeof(ProjectionInvocation));
        EXPECT_THROW(static_cast<void>(cpuFusedProjectionWorkspaceRequirements(0)), std::invalid_argument);
        EXPECT_THROW(static_cast<void>(cpuWorkspaceCounts(static_cast<Transform>(99), 512, 1)), AssertionError);
        EXPECT_THROW(static_cast<void>(cpuWorkspaceCounts(Transform::Identity, 511, 1)), AssertionError);
    }

    TEST(W4A4WorkspaceContract, RepeatedPartialInvocationsRetainExactAdmittedStorage)
    {
        if (!hasCPUBackend()) initCPUBackend(-1);
        const Fixture fixture;
        const auto weights = fixture.weights(Transform::PaleySylvester);
        auto memory = workspaceAuthority();
        const auto claimed = [&] {
            return memory->claimedBytes(DeviceId::cpu(), kOwner, PhysicalMemoryMaterializationKind::NewAllocation);
        };
        {
            DeviceWorkspaceManager manager(DeviceId::cpu(), kCapacity, memory);
            const auto requirements = cpuWorkspaceRequirements(weights.contract(), 5);
            ASSERT_TRUE(manager.allocate(requirements));
            const auto retained = claimed();
            ASSERT_GT(retained, 0u);
            std::array<void *, 6> pointers;
            for (std::size_t i = 0; i < pointers.size(); ++i)
                pointers[i] = manager.getBuffer(requirements.buffers[i].name);
            std::vector<float> input(5 * 512);
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = static_cast<float>(static_cast<int>(i % 53) - 26) * 0.0625f;
            std::array<float, 5> expected{}, output{};
            const auto all = prepareCPUActivations(weights.contract(), input, 5, 512,
                                                   borrowCPUWorkspace(weights.contract(), 5, &manager));
            projectCPU(weights, all, expected, 1);
            for (int rows : {5, 1, 0, 3, 5})
            {
                output.fill(std::bit_cast<float>(std::uint32_t{0x7fc12345}));
                const auto prepared = prepareCPUActivations(weights.contract(), input, rows, 512,
                    borrowCPUWorkspace(weights.contract(), rows, &manager));
                projectCPU(weights, prepared, output, 1);
                for (int row = 0; row < 5; ++row)
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(output[row]), row < rows
                        ? std::bit_cast<std::uint32_t>(expected[row]) : std::uint32_t{0x7fc12345});
                EXPECT_EQ(claimed(), retained);
                for (std::size_t i = 0; i < pointers.size(); ++i)
                    EXPECT_EQ(manager.getBuffer(requirements.buffers[i].name), pointers[i]);
            }
            EXPECT_THROW(static_cast<void>(borrowCPUWorkspace(weights.contract(), 6, &manager)), std::runtime_error);
            EXPECT_EQ(claimed(), retained);
        }
        EXPECT_EQ(claimed(), 0u);
    }
} // namespace llaminar2::w4a4
