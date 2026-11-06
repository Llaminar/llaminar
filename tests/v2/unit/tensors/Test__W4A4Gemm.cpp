/**
 * @file Test__W4A4Gemm.cpp
 * @brief Production ITensorGemm ownership, fused policy and counted-row contracts.
 *
 * Small shared tensor fixtures exercise the real native engine, PMA and stage
 * workspace manager. These are interface/lifecycle checks, not substitutes for
 * the immutable upstream arithmetic replay or full-model accuracy gates. All
 * input transforms and projections execute actual production implementations.
 * Mixed GDN stages retain separate native online-A4 and floating subgroups;
 * fused FFN gate/up stages preserve equal and distinct online-A4 policies.
 * Passive counters prove the grouped routes survive partial work and reset.
 * Actual stage declarations must also preserve native weight identity and exact
 * block extents, including after source retirement and for sliced outputs.
 */
#include "kernels/cpu/gemm/CPUW4A4GemmKernel.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "memory/CPUStorageRoot.h"
#include "backends/BackendManager.h"
#include "kernels/KernelFactory.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/W4A4Tensor.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/FusedGateUpGEMMStage.h"
#include "utils/PerfStatsCollector.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "../../utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <omp.h>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr int kWidth = 512, kOutputs = 7, kRows = 9;
        constexpr std::size_t kWeightBudget = 64 * 1024, kWorkspaceBudget = 1024 * 1024;
        constexpr std::uint32_t kGuard = 0x7fc12345;
        using Factory = test::TestTensorFactory;

        /** @brief Restore the caller's OpenMP policy after each small fixture. */
        struct Threads
        {
            int previous = omp_get_max_threads();
            int previous_dynamic = omp_get_dynamic();
            /** @brief Keep requested team widths observable instead of runtime-shrunk. */
            Threads() { omp_set_dynamic(0); omp_set_num_threads(1); }
            /** @brief Restore both the thread budget and dynamic-team policy. */
            ~Threads() { omp_set_num_threads(previous); omp_set_dynamic(previous_dynamic); }
        };

        /** @brief Observe actual native/floating subgroup submissions without enabling stage timers. */
        class ProjectionEvidence final
        {
            std::optional<std::string> enabled_, filter_;
            /** @return Original optional environment value, preserving absence. */
            static std::optional<std::string> prior(const char *name)
            {
                const char *value = std::getenv(name);
                return value ? std::optional<std::string>{value} : std::nullopt;
            }
            /** @brief Restore one setting after all synchronous producers have returned. */
            static void restore(const char *name, const std::optional<std::string> &value)
            {
                if (value) setenv(name, value->c_str(), 1);
                else unsetenv(name);
            }
        public:
            /** @brief Select passive projection evidence and start a clean test epoch. */
            ProjectionEvidence() : enabled_(prior("LLAMINAR_PERF_STATS_JSON")),
                filter_(prior("LLAMINAR_PERF_STATS_FILTER"))
            {
                setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
                setenv("LLAMINAR_PERF_STATS_FILTER",
                    "kernel.cpu_w4a4_fused_projection_calls,kernel.gdn_projection_route", 1);
                PerfStatsCollector::reloadConfigurationFromEnvironment();
                PerfStatsCollector::reset();
            }
            /** @brief Retire this epoch and restore the caller's collection policy. */
            ~ProjectionEvidence()
            {
                PerfStatsCollector::reset();
                restore("LLAMINAR_PERF_STATS_JSON", enabled_);
                restore("LLAMINAR_PERF_STATS_FILTER", filter_);
                PerfStatsCollector::reloadConfigurationFromEnvironment();
            }
            ProjectionEvidence(const ProjectionEvidence &) = delete;
            ProjectionEvidence &operator=(const ProjectionEvidence &) = delete;
        };

        /** @brief Explicit physical admission for native weights and invocation workspaces. */
        class Fixture
        {
        public:
            std::shared_ptr<PhysicalMemoryAuthority> memory;
            /** @brief Initialize only the CPU backend and this bounded test admission. */
            Fixture()
            {
                if (!hasCPUBackend()) initCPUBackend(-1);
                PhysicalMemoryPlanBuilder builder;
                const PhysicalMemoryResource resource{.world_rank = 0, .device = DeviceId::cpu(),
                    .total_bytes = kWeightBudget + kWorkspaceBudget,
                    .admission_available_bytes = kWeightBudget + kWorkspaceBudget};
                builder.add(resource, PhysicalMemoryOwner::PrimaryModelWeights, kWeightBudget);
                builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, kWorkspaceBudget);
                memory = std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
            }

            /** @brief Publish tiny native rows without a weight-format conversion.
             * @param transform Selected native activation transform.
             * @param clip Exact source clipping value.
             * @return Lifetime-owning final native allocation.
             */
            WeightStorage weights(Transform transform, float clip = 0.75f)
            {
                constexpr auto owner = PhysicalMemoryOwner::PrimaryModelWeights;
                std::array<SourceBlock, kOutputs * 2> source{};
                for (std::size_t index = 0; index < source.size(); ++index)
                {
                    auto &block = source[index];
                    block.unit = (index / 2 + 1) * 0.0625f;
                    block.descriptor = 0x60004321;
                    block.rotation_length = transform == Transform::Identity ? kWidth : 128;
                    block.clip_bits = std::bit_cast<std::uint32_t>(clip);
                    for (int pair = 0; pair < 128; ++pair)
                        block.codes[pair] = static_cast<std::uint8_t>(index * 23 + pair * 17);
                }
                auto bytes = CPUStorageRoot<std::uint8_t>(sizeof(source), owner, DeviceId::cpu(),
                    CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized).allocate(memory);
                std::memcpy(bytes.data(), source.data(), sizeof(source));
                return WeightStorage::adopt(transform, kOutputs, kWidth, std::move(bytes), owner);
            }

            /** @brief Materialize the exact supplied production declaration.
             * @param requirements Complete selected operation requirements.
             * @return Stage-owned retained manager; no kernel receives a persistent binding.
             */
            std::unique_ptr<DeviceWorkspaceManager> workspace(const WorkspaceRequirements &requirements)
            {
                auto manager = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), kWorkspaceBudget, memory);
                if (!manager->allocate(requirements)) throw std::runtime_error("Native workspace fixture admission failed");
                return manager;
            }
        };

        /** @brief Fill small native FP32 tensors with deterministic finite values. */
        void fill(FP32Tensor &tensor, int phase = 0)
        {
            for (std::size_t i = 0; i < tensor.numel(); ++i)
                tensor.mutable_typed_data()[i] = (static_cast<int>((i * 19 + phase) % 97) - 48) * 0.03125f;
        }

        /**
         * @brief Check actual stage metadata without reading or materializing weight data.
         * @param requirements Production stage declaration.
         * @param name Declared immutable operand name.
         * @param type Exact native tensor identity.
         * @param bytes Required complete-source byte extent.
         */
        void expectWeightDescriptor(const StageBufferRequirements &requirements,
            const char *name, TensorType type, size_t bytes)
        {
            const auto found = std::find_if(requirements.buffers.begin(), requirements.buffers.end(),
                [name](const BufferDescriptor &buffer) { return buffer.name == name; });
            ASSERT_NE(found, requirements.buffers.end()) << name;
            EXPECT_EQ(found->role, BufferRole::WEIGHT);
            EXPECT_STREQ(bufferTensorTypeName(found->tensor_type), tensorTypeName(type));
            EXPECT_EQ(found->sizeBytes(), bytes);
            EXPECT_TRUE(found->isReadOnly());
            EXPECT_FALSE(found->isAliasable());
        }

        /** @brief Mark every output byte with an unmistakable dirty value. */
        void poison(FP32Tensor &tensor)
        {
            std::fill_n(tensor.mutable_typed_data(), tensor.numel(), std::bit_cast<float>(kGuard));
        }
    }

    TEST(W4A4GemmContract, TensorEntryRetainsNativeStorageAndOrderedOffsetEpilogue)
    {
        Threads threads;
        Fixture fixture;
        for (const auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            std::optional<WeightStorage> source(fixture.weights(transform));
            CPUW4A4GemmKernel kernel(*source);
            auto requirements = kernel.getWorkspaceRequirements(kRows);
            kernel.appendOutputAccumulationWorkspaceRequirements(requirements, kRows, kOutputs);
            auto workspace = fixture.workspace(requirements);
            auto input = Factory::createFP32({kRows + 2, kWidth});
            auto output = Factory::createFP32({kRows + 2, kOutputs});
            auto bias = Factory::createFP32({kOutputs});
            fill(*input);
            fill(*bias, 5);
            std::vector<float> expected(kRows * kOutputs);
            const auto prepared = prepareCPUActivations(source->view().contract(),
                {input->typed_data() + 2 * kWidth, kRows * kWidth}, kRows, kWidth,
                borrowCPUWorkspace(source->view().contract(), kRows, workspace.get()));
            projectCPU(source->view(), prepared, expected, kOutputs);
            source.reset();
            EXPECT_TRUE(kernel.canReleaseSourceWeightTensor());
            EXPECT_GT(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(kernel.get_n(), kOutputs);
            EXPECT_EQ(kernel.get_k(), kWidth);
            EXPECT_FALSE(kernel.hasWorkspace());
            for (int workers : {1, 3, 8})
            for (bool transpose : {true, false})
            {
                omp_set_num_threads(workers);
                poison(*output);
                std::fill_n(output->mutable_typed_data(), kRows * kOutputs, 0.5f);
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), kRows, kOutputs, kWidth,
                    transpose, 0.75f, 0.25f, bias.get(), nullptr, -1, workspace.get(), 2));
                for (int row = 0; row < kRows; ++row)
                    for (int column = 0; column < kOutputs; ++column)
                    {
                        const int index = row * kOutputs + column;
                        float value = 0.75f * expected[index];
                        value = value + 0.25f * 0.5f;
                        value = value + bias->typed_data()[column];
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(output->typed_data()[index]),
                                  std::bit_cast<std::uint32_t>(value));
                    }
                for (std::size_t i = kRows * kOutputs; i < output->numel(); ++i)
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(output->typed_data()[i]), kGuard);
            }
            EXPECT_THROW(kernel.multiply_tensor(input.get(), output.get(), kRows, kOutputs, kWidth,
                true, 1, 0, nullptr, nullptr, -1, nullptr), std::runtime_error);
            EXPECT_THROW(kernel.multiply_tensor(input.get(), output.get(), kRows, kOutputs, kWidth - 256,
                true, 1, 0, nullptr, nullptr, -1, workspace.get()), std::invalid_argument);
            EXPECT_THROW(kernel.multiply_tensor(input.get(), input.get(), kRows, kOutputs, kWidth,
                true, 1, 0, nullptr, nullptr, -1, workspace.get()), std::invalid_argument);
        }
    }

    TEST(W4A4GemmContract, PreparedStoreAndStagesRetainNativeRowsAndArithmetic)
    {
        Threads threads;
        Fixture fixture;
        using KernelFactory = llaminar::v2::kernels::KernelFactory;
        static_assert(isW4A4Format(TensorType::W4A4) && isW4A4Format(TensorType::W4A4H));
        for (const auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            auto source = std::make_shared<W4A4Tensor>(fixture.weights(transform));
            const auto claimed = fixture.memory->claimedBytes(DeviceId::cpu(),
                PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryMaterializationKind::NewAllocation);
            const auto placement = CPUWeightStoragePlacement::accounted(fixture.memory, DeviceId::cpu(),
                PhysicalMemoryOwner::PrimaryModelWeights, CPUWeightStoragePlacement::local());
            const auto prepare = [&](KernelFactory::GemmPreparationKind kind,
                                     FloatingActivationOperandPrecision precision) {
                return KernelFactory::prepareGemmHandleLocal(source.get(), DeviceId::cpu(), kind, precision, placement);
            };
            auto handle = prepare(KernelFactory::GemmPreparationKind::AUTO, FloatingActivationOperandPrecision::Native);
            ASSERT_TRUE(handle);
            EXPECT_EQ(handle->kind, KernelFactory::GemmPreparationKind::CPU_NATIVE_ONLINE_A4);
            EXPECT_THROW(prepare(KernelFactory::GemmPreparationKind::FLOATING_POINT,
                FloatingActivationOperandPrecision::Native), std::invalid_argument);
            EXPECT_THROW(prepare(KernelFactory::GemmPreparationKind::CPU_PACKED,
                FloatingActivationOperandPrecision::Native), std::invalid_argument);
            EXPECT_THROW(prepare(KernelFactory::GemmPreparationKind::AUTO,
                FloatingActivationOperandPrecision::FP16RNE), std::invalid_argument);
            handle.reset();

            const ModelContextId model{179};
            PreparedWeightStore store(model);
            store.installPhysicalMemoryAuthority(fixture.memory);
            WeightBinding binding;
            binding.binding_id = 1;
            binding.identity = makeSourceWeightIdentity("blk.0.ffn_down.weight", model, 1);
            binding.residency.home_device = DeviceId::cpu();
            binding.residency.resident_device = DeviceId::cpu();
            binding.tensor_owner = source;
            binding.tensor = source.get();
            binding.immutable = true;
            const auto ref = store.prepareGemm(binding);
            auto input = Factory::createFP32({kRows, kWidth});
            auto output = Factory::createFP32({kRows, kOutputs});
            auto sliced_output = Factory::createFP32({kRows, 3});
            fill(*input);
            GEMMStage::Params params{.device_id = DeviceId::cpu(), .A = input.get(), .B = source.get(),
                .C = output.get(), .m = kRows, .n = kOutputs, .k = kWidth,
                .prepared_ref = ref, .prepared_store = &store};
            GEMMStage ordinary(params);
            std::int32_t live = kRows;
            params.force_decode_equivalent_verifier_prefill = true;
            params.verifier_row_range = DeviceRowRange::deviceCounted(kRows, &live);
            GEMMStage verifier(params);
            params.output_range = {2, 5};
            params.C = sliced_output.get();
            GEMMStage sliced(params);
            const auto check_descriptors = [&] {
                for (const auto *stage : {&ordinary, &verifier, &sliced})
                    expectWeightDescriptor(stage->getBufferRequirements(), "B", source->native_type(),
                        kOutputs * (kWidth / kBlockElements) * kSourceBlockBytes);
                EXPECT_EQ(ordinary.getBufferRequirements().totalOutputBytes(), output->size_bytes());
                EXPECT_EQ(sliced.getBufferRequirements().totalOutputBytes(), sliced_output->size_bytes());
            };
            check_descriptors();
            auto requirements = ordinary.getWorkspaceRequirements(kRows, kOutputs, kWidth);
            requirements.merge(verifier.getWorkspaceRequirements(kRows, kOutputs, kWidth));
            requirements.merge(sliced.getWorkspaceRequirements(kRows, 3, kWidth));
            EXPECT_TRUE(std::none_of(requirements.buffers.begin(), requirements.buffers.end(),
                [](const auto &buffer) { return buffer.name == kCPUProjectionBundle; }));
            auto workspace = fixture.workspace(requirements);
            const auto workspace_claim = fixture.memory->claimedBytes(DeviceId::cpu(),
                PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryMaterializationKind::NewAllocation);
            ordinary.bindWorkspace(workspace.get());
            verifier.bindWorkspace(workspace.get());
            sliced.bindWorkspace(workspace.get());
            std::vector<float> expected(kRows * kOutputs);
            {
                const auto storage = source->retainWeightStorage();
                const auto prepared = prepareCPUActivations(storage.view().contract(),
                    {input->typed_data(), input->numel()}, kRows, kWidth,
                    borrowCPUWorkspace(storage.view().contract(), kRows, workspace.get()));
                projectCPU(storage.view(), prepared, expected, kOutputs);
            }
            source->release_raw_data();
            EXPECT_EQ(source->raw_data(), nullptr);
            check_descriptors();
            EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryMaterializationKind::NewAllocation), claimed);
            for (int workers : {1, 3, 8})
            {
                omp_set_num_threads(workers);
                CPUDeviceContext context(DeviceId::cpu(), workers);
                poison(*output);
                ASSERT_TRUE(ordinary.execute(&context));
                EXPECT_EQ(std::memcmp(output->typed_data(), expected.data(), expected.size() * sizeof(float)), 0);
                for (int count : {kRows, 1, 0, 3, kRows})
                {
                    live = count;
                    poison(*output);
                    poison(*sliced_output);
                    ASSERT_TRUE(verifier.execute(&context));
                    ASSERT_TRUE(sliced.execute(&context));
                    EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                        PhysicalMemoryMaterializationKind::NewAllocation), workspace_claim);
                    for (int row = 0; row < kRows; ++row)
                    {
                        for (int column = 0; column < kOutputs; ++column)
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(output->typed_data()[row * kOutputs + column]),
                                row < live ? std::bit_cast<std::uint32_t>(expected[row * kOutputs + column]) : kGuard);
                        for (int column = 0; column < 3; ++column)
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(sliced_output->typed_data()[row * 3 + column]),
                                row < live ? std::bit_cast<std::uint32_t>(expected[row * kOutputs + column + 2]) : kGuard);
                    }
                }
            }
            sliced.unbindWorkspace();
            verifier.unbindWorkspace();
            ordinary.unbindWorkspace();
        }
        EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    /** @test The real gate/up adapter retains native policy groups, original rows and arena ownership through reset. */
    TEST(W4A4GemmContract, FusedGateUpStagesRetainNativePoliciesThroughReset)
    {
        Threads threads;
        ProjectionEvidence evidence;
        Fixture fixture;
        struct Policy { Transform gate, up; float up_clip; };
        constexpr std::array<Policy, 5> policies{{
            {Transform::Identity, Transform::Identity, 0.75f},
            {Transform::Identity, Transform::Identity, 0.875f},
            {Transform::Identity, Transform::PaleySylvester, 0.875f},
            {Transform::PaleySylvester, Transform::Identity, 0.75f},
            {Transform::PaleySylvester, Transform::PaleySylvester, 0.75f}}};
        constexpr std::array<int, 2> widths{kOutputs, 5};
        for (const auto &policy : policies)
        {
            const ModelContextId model{185};
            PreparedWeightStore store(model);
            store.installPhysicalMemoryAuthority(fixture.memory);
            std::array<std::shared_ptr<W4A4Tensor>, 2> weights{
                std::make_shared<W4A4Tensor>(fixture.weights(policy.gate)),
                std::make_shared<W4A4Tensor>(fixture.weights(policy.up, policy.up_clip).rowSlice(1, widths[1]))};
            std::array<PreparedWeightRef, 2> refs;
            std::array<std::unique_ptr<FP32Tensor>, 2> outputs, biases;
            for (size_t leaf = 0; leaf < 2; ++leaf)
            {
                WeightBinding binding;
                binding.binding_id = leaf + 1;
                binding.identity = makeSourceWeightIdentity(leaf == 0 ? "blk.0.ffn_gate.weight" : "blk.0.ffn_up.weight",
                    model, leaf + 1);
                binding.residency.home_device = DeviceId::cpu();
                binding.residency.resident_device = DeviceId::cpu();
                binding.tensor_owner = weights[leaf]; binding.tensor = weights[leaf].get(); binding.immutable = true;
                refs[leaf] = store.prepareGemm(binding);
                outputs[leaf] = Factory::createFP32({kRows, size_t(widths[leaf])});
                biases[leaf] = Factory::createFP32({size_t(widths[leaf])});
                fill(*biases[leaf], int(leaf) + 3);
            }
            auto input = Factory::createFP32({kRows, kWidth});
            fill(*input);
            FusedGateUpGEMMStage::Params params;
            params.device_id = DeviceId::cpu(); params.input = input.get(); params.m = kRows; params.k = kWidth;
            params.w_gate = weights[0].get(); params.output_gate = outputs[0].get(); params.n_gate = widths[0];
            params.w_up = weights[1].get(); params.output_up = outputs[1].get(); params.n_up = widths[1];
            params.bias_gate = biases[0].get(); params.bias_up = biases[1].get();
            params.prepared_ref_gate = refs[0]; params.prepared_ref_up = refs[1]; params.prepared_store = &store;
            FusedGateUpGEMMStage ordinary(params);
            int32_t live = kRows;
            params.force_decode_equivalent_verifier_prefill = true;
            params.verifier_row_range = DeviceRowRange::deviceCounted(kRows, &live);
            FusedGateUpGEMMStage verifier(params);
            auto requirements = ordinary.getWorkspaceRequirements(kRows);
            requirements.merge(verifier.getWorkspaceRequirements(kRows));
            EXPECT_EQ(std::count_if(requirements.buffers.begin(), requirements.buffers.end(),
                [](const auto &buffer) { return buffer.name == kCPUProjectionBundle; }), 1);
            auto workspace = fixture.workspace(requirements);
            ordinary.bindWorkspace(workspace.get()); verifier.bindWorkspace(workspace.get());
            struct Revoke
            {
                FusedGateUpGEMMStage &ordinary, &verifier;
                /** @brief Withdraw both stage borrows before the nonconcurrent shared arena retires. */
                ~Revoke() { verifier.unbindWorkspace(); ordinary.unbindWorkspace(); }
            } revoke{ordinary, verifier};
            const auto workspace_claim = fixture.memory->claimedBytes(DeviceId::cpu(),
                PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryMaterializationKind::NewAllocation);
            const auto weight_claim = fixture.memory->claimedBytes(DeviceId::cpu(),
                PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryMaterializationKind::NewAllocation);
            std::array<std::vector<float>, 2> expected;
            auto serial_input = Factory::createFP32({1, kWidth});
            // Serial rows are only this test's byte-equivalence oracle. The
            // production stage must submit one native grouped bundle, not replay.
            for (size_t leaf = 0; leaf < 2; ++leaf)
            {
                auto serial_output = Factory::createFP32({1, size_t(widths[leaf])});
                expected[leaf].resize(kRows * widths[leaf]);
                auto *engine = store.gemmKernel(refs[leaf]);
                ASSERT_NE(engine, nullptr);
                for (int row = 0; row < kRows; ++row)
                {
                    std::copy_n(input->typed_data() + row * kWidth, kWidth, serial_input->mutable_typed_data());
                    ASSERT_TRUE(engine->multiply_tensor(serial_input.get(), serial_output.get(), 1, widths[leaf], kWidth,
                        false, 1, 0, biases[leaf].get(), nullptr, -1, workspace.get()));
                    std::copy_n(serial_output->typed_data(), widths[leaf], expected[leaf].data() + row * widths[leaf]);
                }
                weights[leaf]->release_raw_data();
            }
            for (const auto *stage : {&ordinary, &verifier})
            {
                expectWeightDescriptor(stage->getBufferRequirements(), "w_gate", weights[0]->native_type(),
                    widths[0] * (kWidth / kBlockElements) * kSourceBlockBytes);
                expectWeightDescriptor(stage->getBufferRequirements(), "w_up", weights[1]->native_type(),
                    widths[1] * (kWidth / kBlockElements) * kSourceBlockBytes);
            }
            std::string error;
            ASSERT_TRUE(ordinary.validatePreparedWeights(&error)) << error;
            ASSERT_TRUE(verifier.validatePreparedWeights(&error)) << error;
            const auto identity = verifier.graphCaptureVariantSignature();
            for (int workers : {1, 3, 8})
            {
                omp_set_num_threads(workers);
                CPUDeviceContext context(DeviceId::cpu(), workers);
                constexpr std::array<int, 6> counts{kRows, kRows, 3, 0, 1, kRows};
                for (size_t phase = 0; phase < counts.size(); ++phase)
                {
                    SCOPED_TRACE(::testing::Message() << "workers=" << workers << " phase=" << phase
                        << " gate=" << int(policy.gate) << " up=" << int(policy.up) << " clip=" << policy.up_clip);
                    live = counts[phase];
                    if (phase == 4) verifier.resetSessionStatePreservingCapturedReplay();
                    for (auto &output : outputs) poison(*output);
                    PerfStatsCollector::reset();
                    ASSERT_TRUE(phase == 0 ? ordinary.execute(&context) : verifier.execute(&context));
                    EXPECT_EQ(verifier.graphCaptureVariantSignature(), identity);
                    for (size_t leaf = 0; leaf < 2; ++leaf)
                        for (int row = 0; row < kRows; ++row)
                            for (int column = 0; column < widths[leaf]; ++column)
                            {
                                const auto index = row * widths[leaf] + column;
                                EXPECT_EQ(std::bit_cast<uint32_t>(outputs[leaf]->typed_data()[index]), row < live ?
                                    std::bit_cast<uint32_t>(expected[leaf][index]) : kGuard);
                            }
                    const auto calls = PerfStatsCollector::snapshot({"kernel.cpu_w4a4_fused_projection_calls"});
                    ASSERT_EQ(calls.size(), 1u);
                    EXPECT_EQ(calls.front().count, 1u);
                    EXPECT_DOUBLE_EQ(calls.front().value, 1.0);
                    EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
                        PhysicalMemoryMaterializationKind::NewAllocation), weight_claim);
                    EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                        PhysicalMemoryMaterializationKind::NewAllocation), workspace_claim);
                }
            }
        }
        EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    TEST(W4A4GemmContract, MixedGDNRetainsNativeAndFloatingSubgroupsThroughReset)
    {
        Threads threads;
        ProjectionEvidence evidence;
        Fixture fixture;
        constexpr std::array<int, 4> widths{kOutputs, 5, 2, 3};
        constexpr std::array<const char *, 4> names{
            "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"};
        for (const auto second_transform : {Transform::Identity, Transform::PaleySylvester})
        {
            const ModelContextId model{180};
            PreparedWeightStore store(model);
            store.installPhysicalMemoryAuthority(fixture.memory);
            std::array<std::shared_ptr<TensorBase>, 4> weights;
            weights[0] = std::make_shared<W4A4Tensor>(fixture.weights(Transform::Identity));
            weights[1] = std::make_shared<W4A4Tensor>(fixture.weights(second_transform,
                second_transform == Transform::Identity ? 0.75f : 0.875f).rowSlice(1, widths[1]));
            for (size_t leaf = 2; leaf < weights.size(); ++leaf)
            {
                auto bytes = CPUStorageRoot<float>(widths[leaf] * kWidth, PhysicalMemoryOwner::PrimaryModelWeights,
                    DeviceId::cpu(), CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized).allocate(fixture.memory);
                auto value = std::make_shared<FP32Tensor>(std::vector<size_t>{size_t(widths[leaf]), kWidth},
                    std::move(bytes), DeviceId::cpu());
                fill(*value, int(leaf) * 7);
                weights[leaf] = std::move(value);
            }
            std::array<PreparedWeightRef, 4> refs;
            for (size_t leaf = 0; leaf < weights.size(); ++leaf)
            {
                WeightBinding binding;
                binding.binding_id = leaf + 1;
                binding.identity = makeSourceWeightIdentity(names[leaf], model, leaf + 1);
                binding.residency.home_device = DeviceId::cpu();
                binding.residency.resident_device = DeviceId::cpu();
                binding.tensor_owner = weights[leaf];
                binding.tensor = weights[leaf].get();
                binding.immutable = true;
                refs[leaf] = store.prepareGemm(binding);
            }
            auto input = Factory::createFP32({kRows, kWidth});
            fill(*input);
            std::array<std::unique_ptr<FP32Tensor>, 4> outputs;
            for (size_t leaf = 0; leaf < outputs.size(); ++leaf)
                outputs[leaf] = Factory::createFP32({kRows, size_t(widths[leaf])});
            GDNProjectionStage::Params params;
            params.device_id = DeviceId::cpu();
            params.input = input.get(); params.m = kRows; params.k = kWidth;
            params.w_qkv = weights[0].get(); params.output_qkv = outputs[0].get(); params.n_qkv = widths[0];
            params.w_z = weights[1].get(); params.output_z = outputs[1].get(); params.n_z = widths[1];
            params.w_a = weights[2].get(); params.output_a = outputs[2].get(); params.n_a = widths[2];
            params.w_b = weights[3].get(); params.output_b = outputs[3].get(); params.n_b = widths[3];
            params.prepared_ref_qkv = refs[0]; params.prepared_ref_z = refs[1];
            params.prepared_ref_a = refs[2]; params.prepared_ref_b = refs[3];
            params.prepared_store = &store;
            GDNProjectionStage ordinary(params);
            int32_t live = kRows;
            params.force_decode_equivalent_verifier_prefill = true;
            params.verifier_row_range = DeviceRowRange::deviceCounted(kRows, &live);
            GDNProjectionStage verifier(params);
            const auto check_descriptors = [&] {
                const std::array<const char *, 4> descriptor_names{"w_qkv", "w_z", "w_a", "w_b"};
                for (const auto *stage : {&ordinary, &verifier})
                    for (size_t leaf = 0; leaf < weights.size(); ++leaf)
                        expectWeightDescriptor(stage->getBufferRequirements(), descriptor_names[leaf],
                            weights[leaf]->native_type(), leaf < 2
                                ? widths[leaf] * (kWidth / kBlockElements) * kSourceBlockBytes
                                : widths[leaf] * kWidth * sizeof(float));
            };
            check_descriptors();
            std::string binding_error;
            ASSERT_TRUE(ordinary.validatePreparedWeights(&binding_error)) << binding_error;
            ASSERT_TRUE(verifier.validatePreparedWeights(&binding_error)) << binding_error;
            auto requirements = ordinary.getWorkspaceRequirements(kRows);
            requirements.merge(verifier.getWorkspaceRequirements(kRows));
            auto workspace = fixture.workspace(requirements);
            ordinary.bindWorkspace(workspace.get());
            verifier.bindWorkspace(workspace.get());
            struct Revoke
            {
                GDNProjectionStage &ordinary, &verifier;
                /** @brief Retire both stages' borrows before destroying the admitted arena. */
                ~Revoke() { verifier.unbindWorkspace(); ordinary.unbindWorkspace(); }
            } revoke{ordinary, verifier};
            const auto workspace_claim = fixture.memory->claimedBytes(DeviceId::cpu(),
                PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryMaterializationKind::NewAllocation);
            const auto weight_claim = fixture.memory->claimedBytes(DeviceId::cpu(),
                PhysicalMemoryOwner::PrimaryModelWeights, PhysicalMemoryMaterializationKind::NewAllocation);
            std::array<std::vector<float>, 4> expected;
            auto serial_input = Factory::createFP32({1, kWidth});
            // Row execution is an explicit test oracle, never a stage implementation.
            // The FP32 grid also keeps these tiny products/sums exactly representable.
            for (size_t leaf = 0; leaf < outputs.size(); ++leaf)
            {
                expected[leaf].resize(kRows * widths[leaf]);
                auto serial_output = Factory::createFP32({1, size_t(widths[leaf])});
                auto *engine = store.gemmKernel(refs[leaf]);
                ASSERT_NE(engine, nullptr);
                for (int row = 0; row < kRows; ++row)
                {
                    std::copy_n(input->typed_data() + row * kWidth, kWidth, serial_input->mutable_typed_data());
                    ASSERT_TRUE(engine->multiply_tensor(serial_input.get(), serial_output.get(), 1,
                        widths[leaf], kWidth, leaf >= 2, 1, 0, nullptr, nullptr, -1, workspace.get()));
                    std::copy_n(serial_output->typed_data(), widths[leaf], expected[leaf].data() + row * widths[leaf]);
                }
            }
            weights[0]->release_raw_data();
            weights[1]->release_raw_data();
            EXPECT_EQ(weights[0]->raw_data(), nullptr);
            EXPECT_EQ(weights[1]->raw_data(), nullptr);
            check_descriptors();
            ASSERT_TRUE(verifier.validatePreparedWeights(&binding_error)) << binding_error;
            const auto identity = verifier.graphCaptureVariantSignature();
            for (int workers : {1, 3, 8})
            {
                omp_set_num_threads(workers);
                CPUDeviceContext context(DeviceId::cpu(), workers);
                for (int phase = 0; phase < 6; ++phase)
                {
                    SCOPED_TRACE(::testing::Message() << "workers=" << workers << " phase=" << phase
                        << " transform=" << int(second_transform));
                    const std::array<int, 6> counts{kRows, kRows, 3, 0, 1, kRows};
                    live = counts[phase];
                    if (phase == 4) verifier.resetSessionStatePreservingCapturedReplay();
                    for (auto &output : outputs) poison(*output);
                    PerfStatsCollector::reset();
                    ASSERT_TRUE(phase == 0 ? ordinary.execute(&context) : verifier.execute(&context));
                    EXPECT_EQ(verifier.graphCaptureVariantSignature(), identity);
                    EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
                        PhysicalMemoryMaterializationKind::NewAllocation), weight_claim);
                    EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                        PhysicalMemoryMaterializationKind::NewAllocation), workspace_claim);
                    for (size_t leaf = 0; leaf < outputs.size(); ++leaf)
                        for (int row = 0; row < kRows; ++row)
                            for (int column = 0; column < widths[leaf]; ++column)
                            {
                                const auto index = row * widths[leaf] + column;
                                EXPECT_EQ(std::bit_cast<uint32_t>(outputs[leaf]->typed_data()[index]), row < live ?
                                    std::bit_cast<uint32_t>(expected[leaf][index]) : kGuard);
                            }
                    const auto calls = PerfStatsCollector::snapshot({"kernel.cpu_w4a4_fused_projection_calls"});
                    ASSERT_EQ(calls.size(), 1u);
                    EXPECT_EQ(calls.front().count, 1u);
                    EXPECT_DOUBLE_EQ(calls.front().value, 1.0);
                    bool native_group = false, floating_group = false;
                    const auto routes = PerfStatsCollector::snapshot({"kernel.gdn_projection_route"});
                    ASSERT_EQ(routes.size(), phase == 0 ? 2u : 3u)
                        << PerfStatsCollector::jsonString({"kernel.gdn_projection_route"});
                    bool complete_bundle = false;
                    for (const auto &record : routes)
                    {
                        EXPECT_EQ(record.count, 1u);
                        EXPECT_DOUBLE_EQ(record.value, 1.0);
                        if (phase != 0 && record.tags.at("route") == "grouped_decode_equivalent_verifier_mixed")
                        {
                            EXPECT_EQ(record.tags.at("names"), "qkv+z+alpha+beta");
                            EXPECT_EQ(record.tags.at("codebooks"), "none+none+none+none");
                            EXPECT_EQ(record.tags.at("projections"), "4");
                            complete_bundle = true;
                            continue;
                        }
                        EXPECT_EQ(record.tags.at("route"), phase == 0 ? "same_kernel_mixed_codebook_subgroup" :
                            "grouped_decode_equivalent_verifier_subgroup");
                        EXPECT_EQ(record.tags.at("projections"), "2");
                        EXPECT_EQ(record.tags.at("codebooks"), "none+none");
                        native_group |= record.tags.at("names") == "qkv+z";
                        floating_group |= record.tags.at("names") == "alpha+beta";
                    }
                    EXPECT_TRUE(native_group);
                    EXPECT_TRUE(floating_group);
                    EXPECT_EQ(complete_bundle, phase != 0);
                }
            }
        }
        EXPECT_EQ(fixture.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrimaryModelWeights,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    TEST(W4A4GemmContract, FusedPoliciesRetainCountedCoordinatesAcrossPartialAndEmptyWork)
    {
        Threads threads;
        Fixture fixture;
        CPUW4A4GemmKernel first(fixture.weights(Transform::Identity));
        CPUW4A4GemmKernel second(fixture.weights(Transform::PaleySylvester, 0.875f));
        CPUW4A4GemmKernel third(fixture.weights(Transform::Identity));
        const std::array<CPUW4A4GemmKernel *, 3> kernels{&first, &second, &third};
        auto input = Factory::createFP32({kRows, kWidth});
        fill(*input);
        std::array<std::unique_ptr<FP32Tensor>, 3> outputs, expected;
        WorkspaceRequirements requirements;
        for (std::size_t i = 0; i < kernels.size(); ++i)
        {
            outputs[i] = Factory::createFP32({kRows, kOutputs});
            expected[i] = Factory::createFP32({kRows, kOutputs});
            requirements.merge(kernels[i]->getWorkspaceRequirements(kRows));
        }
        const std::array<int, 3> widths{kOutputs, kOutputs, kOutputs};
        first.appendFusedProjectionWorkspaceRequirements(requirements, kRows, widths, kWidth);
        auto workspace = fixture.workspace(requirements);
        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        for (std::size_t i = 0; i < kernels.size(); ++i)
        {
            ASSERT_TRUE(kernels[i]->multiply_tensor(input.get(), expected[i].get(), kRows, kOutputs, kWidth,
                true, 1, 0, nullptr, nullptr, -1, workspace.get()));
            projections.emplace_back(kernels[i], outputs[i].get(), kOutputs);
        }
        for (int workers : {1, 3, 8})
        {
            omp_set_num_threads(workers);
            for (std::int32_t live : {kRows, 3, 0, 1, kRows})
            {
                for (auto &output : outputs) poison(*output);
                auto scope = first.beginVerifierDecodeEquivalentScope(DeviceRowRange::deviceCounted(kRows, &live));
                ASSERT_TRUE(first.multiply_fused_verifier_rows_decode_equivalent(
                    input.get(), projections, kRows, kWidth, nullptr, workspace.get()));
                for (std::size_t leaf = 0; leaf < kernels.size(); ++leaf)
                    for (int row = 0; row < kRows; ++row)
                        for (int column = 0; column < kOutputs; ++column)
                        {
                            const int index = row * kOutputs + column;
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(outputs[leaf]->typed_data()[index]), row < live
                                ? std::bit_cast<std::uint32_t>(expected[leaf]->typed_data()[index]) : kGuard);
                        }
            }
        }
        std::array<std::int32_t, 6> state{0, 3, 1, 1, 0, 2};
        const auto work = ProjectionRowWork::causalStrided(3, 3, state.data(), state.data() + 1, 2, 10, 1);
        for (auto &output : outputs) poison(*output);
        {
            ITensorGemm::ProjectionWorkScope scope(work);
            ASSERT_TRUE(first.multiply_fused_tensor(input.get(), projections, kRows, kWidth, nullptr, workspace.get()));
        }
        for (int row = 0; row < kRows; ++row)
            for (int column = 0; column < kOutputs; ++column)
                for (std::size_t leaf = 0; leaf < kernels.size(); ++leaf)
                {
                    const int index = row * kOutputs + column;
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(outputs[leaf]->typed_data()[index]),
                        work.resolveHost(work.requestForRow(row)).contains(row)
                            ? std::bit_cast<std::uint32_t>(expected[leaf]->typed_data()[index]) : kGuard);
                }
        auto missing_bundle = fixture.workspace(requirements = first.getWorkspaceRequirements(kRows));
        EXPECT_THROW(first.multiply_fused_tensor(input.get(), projections, kRows, kWidth,
            nullptr, missing_bundle.get()), std::runtime_error);
        projections.back().kernel = nullptr;
        EXPECT_THROW(first.multiply_fused_tensor(input.get(), projections, kRows, kWidth,
            nullptr, workspace.get()), std::invalid_argument);
    }

    TEST(W4A4GemmContract, SwiGLUVerifierUsesTheProductionPointwiseBoundary)
    {
        Threads threads;
        Fixture fixture;
        CPUW4A4GemmKernel kernel(fixture.weights(Transform::PaleySylvester));
        auto requirements = kernel.getWorkspaceRequirements(kRows);
        kernel.appendSwiGLUWorkspaceRequirements(requirements, kRows, kWidth);
        auto workspace = fixture.workspace(requirements);
        auto gate = Factory::createFP32({kRows, kWidth});
        auto up = Factory::createFP32({kRows, kWidth});
        auto activated = Factory::createFP32({kRows, kWidth});
        auto output = Factory::createFP32({kRows, kOutputs});
        auto expected = Factory::createFP32({kRows, kOutputs});
        fill(*gate, 13);
        fill(*up, 5);
        primitives::compute_swiglu(gate->typed_data(), up->typed_data(), activated->mutable_typed_data(), kRows * kWidth);
        ASSERT_TRUE(kernel.multiply_tensor(activated.get(), expected.get(), kRows, kOutputs, kWidth,
            true, 1, 0, nullptr, nullptr, -1, workspace.get()));
        for (int workers : {1, 3, 8})
        {
            omp_set_num_threads(workers);
            for (std::int32_t live : {kRows, 1, 0, 5, kRows})
            {
                poison(*output);
                auto scope = kernel.beginVerifierDecodeEquivalentScope(DeviceRowRange::deviceCounted(kRows, &live));
                ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    gate.get(), up.get(), output.get(), kRows, kOutputs, kWidth, 1, 0, workspace.get()));
                for (int row = 0; row < kRows; ++row)
                    for (int column = 0; column < kOutputs; ++column)
                    {
                        const int index = row * kOutputs + column;
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(output->typed_data()[index]), row < live
                            ? std::bit_cast<std::uint32_t>(expected->typed_data()[index]) : kGuard);
                    }
            }
        }
    }
} // namespace llaminar2::w4a4
