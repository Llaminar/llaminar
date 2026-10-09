/**
 * @file Test__PreparedWeightSourceLifetime.cpp
 * @brief Shared TP source bytes outlive every native preparation consumer.
 *
 * Sequentially completing one participant while another has only frozen its
 * source makes a concurrent upload/free race deterministic. The real loader,
 * GPU packing, prepared store and physical memory authority run on every
 * supported source format, in both completion orders. The final graph gate
 * must reclaim transient host bytes without retiring either GPU handle.
 * A retired source may reuse its existing prepared owner, but cannot be
 * silently reloaded for an unprepared participant after a lifecycle violation.
 */
#include "../../mocks/MockModelLoader.h"
#include "../../utils/EmbeddingVerifierFormats.h"
#include "backends/BackendManager.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "loaders/GPUVramPreflight.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>

namespace llaminar2::test
{
namespace
{
    /** @brief Exact pair of native consumers of the same immutable source. */
    struct Consumers
    {
        std::array<DeviceId, 2> devices;
        const char *name;
    };

    /** @return Compiled native pairings; missing required hardware is a failure. */
    std::vector<Consumers> consumerCases()
    {
        std::vector<Consumers> result;
#ifdef HAVE_CUDA
        result.push_back({{DeviceId::cuda(0), DeviceId::cuda(1)}, "CUDA"});
#endif
#ifdef HAVE_ROCM
        result.push_back({{DeviceId::rocm(0), DeviceId::rocm(1)}, "ROCm"});
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        result.push_back({{DeviceId::rocm(0), DeviceId::cuda(0)}, "Hybrid"});
#endif
        return result;
    }

    /** @return Canonical source identity for exact prepared memory geometry. */
    ExpertWeightFormat weightFormat(const TensorBase &source)
    {
        const auto &formats = quantizedVerifierFormats();
        const auto found = std::find_if(formats.begin(), formats.end(),
            [&](const auto &format) { return format.tensor_type == source.native_type(); });
        if (found == formats.end()) return ExpertWeightFormat::floating(source.native_type());
        return ExpertWeightFormat::nativeVnni(
            {found->source_codebook_id, found->source_is_superblock, true});
    }

    /**
     * @brief Admit only the fixture's actual prepared matrix and load staging.
     * @param source Native source whose format and geometry own every byte extent.
     * @param consumers Exact native endpoints; no capacity reserve is added.
     * @return One physical authority shared by preparation and prepared handles.
     */
    std::shared_ptr<PhysicalMemoryAuthority> sourceMemory(
        const TensorBase &source, const Consumers &consumers)
    {
        const auto persistent = resolveExpertPreparedProjectionMemoryGeometry(
            source.rows(), source.cols(), weightFormat(source));
        const auto staging = resolveGPUWeightLoadMemoryGeometry(
            source.size_bytes(), configuredGPUWeightLoadMemoryPolicy());
        const PhysicalMemoryResource host{.world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
        PhysicalMemoryPlanBuilder memory;
        for (const auto device : consumers.devices)
        {
            const PhysicalMemoryResource gpu{.world_rank = 0, .device = device,
                .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
            memory.add(gpu, PhysicalMemoryOwner::PrimaryModelWeights, persistent.gpu_live_bytes)
                .add(gpu, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes)
                .add(host, PhysicalMemoryOwner::WeightLoadStaging, staging.host_staging_bytes);
        }
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(memory.build()), 0);
    }

    /**
     * @brief Declare the same replicated NextN operand for a native participant.
     * @param strategy Immutable model and topology identity shared by both consumers.
     * @param index Device index in that topology, also retained as TP identity.
     * @param host_policy This consumer's declared requirement for the shared bytes.
     * @return Exact source requirement retaining the caller's host lifetime.
     */
    WeightPlan sourcePlan(const InferenceStrategy &strategy, int index,
        WeightHostPolicy host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized)
    {
        WeightPlan plan(strategy);
        WeightRequirement requirement;
        requirement.canonical_name = "blk.2.nextn.eh_proj.weight";
        requirement.target_device = strategy.devices.at(index);
        requirement.expected_prepared_kind = requirement.target_device.is_cuda()
            ? PreparedWeightKind::CudaInt8PackedGemm : PreparedWeightKind::RocmInt8PackedGemm;
        requirement.lookup_device = DeviceId::cpu();
        requirement.tp_rank_or_device_index = index;
        requirement.host_policy = host_policy;
        plan.add(requirement);
        return plan;
    }

    /** @brief Native setup fixture; no GPU work belongs to the Unit suite. */
    class PreparedWeightSourceLifetime : public ::testing::TestWithParam<Consumers> {};

    TEST_P(PreparedWeightSourceLifetime, SharedBytesSurviveUntilEveryGraphIsMaterialized)
    {
        const auto consumers = GetParam();
        for (const auto device : consumers.devices)
        {
            const auto *backend = getBackendFor(device);
            ASSERT_NE(backend, nullptr);
            ASSERT_GT(backend->deviceCount(), device.ordinal);
        }
        const std::array policies{
            std::array{WeightHostPolicy::RequiredUntilGraphMaterialized,
                       WeightHostPolicy::RequiredUntilGraphMaterialized},
            std::array{WeightHostPolicy::ReleasableAfterPreparation,
                       WeightHostPolicy::RequiredUntilGraphMaterialized},
            std::array{WeightHostPolicy::RequiredUntilGraphMaterialized,
                       WeightHostPolicy::RequiredForCPUExecution}};
        for (const auto &format : embeddingVerifierFormats())
        for (const auto &host_policies : policies)
        for (const bool reverse : {false, true})
        {
            SCOPED_TRACE(format.label);
            SCOPED_TRACE(reverse);
            SCOPED_TRACE(static_cast<int>(host_policies[0]));
            SCOPED_TRACE(static_cast<int>(host_policies[1]));
            const bool cpu_retained = host_policies[1] == WeightHostPolicy::RequiredForCPUExecution;
            constexpr const char *name = "blk.2.nextn.eh_proj.weight";
            const ModelContextId model_id{32001};
            std::shared_ptr<TensorBase> source = format.create({256, 512}, 32001);
            const auto *bytes = source->raw_data();
            ASSERT_NE(bytes, nullptr);
            MockModelLoader loader;
            loader.addTensor(name, source);
            WeightManager manager(loader);
            manager.setWeightShardingConfig(Qwen35MoESchemaFactory{}.getWeightShardingConfig());
            manager.setPreparedWeightStore(std::make_shared<PreparedWeightStore>(model_id));
            manager.installPhysicalMemoryAuthority(sourceMemory(*source, consumers));
            InferenceStrategy strategy;
            strategy.model_id = model_id;
            strategy.mode = WeightInferenceMode::LocalTP;
            strategy.tp_degree = 2;
            strategy.devices.assign(consumers.devices.begin(), consumers.devices.end());
            std::vector<FrozenModelWeightSet> frozen;
            for (int index = 0; index < 2; ++index)
            {
                frozen.push_back(manager.materialize(sourcePlan(strategy, index, host_policies[index])));
                ASSERT_EQ(frozen.back().bindings().size(), 1u);
                ASSERT_EQ(frozen.back().bindings().front().tensor, source.get());
            }
            manager.markMaterializationComplete();
            std::array<PreparedWeightRef, 2> refs;
            for (const int index : reverse ? std::array{1, 0} : std::array{0, 1})
            {
                const auto device = consumers.devices[index];
                ASSERT_TRUE(manager.prepareWeightsForDevice(frozen[index], device, false));
                // The delayed consumer owns a binding but has not uploaded.
                // A shared_ptr alone cannot protect mutable tensor storage.
                ASSERT_FALSE(source->is_raw_data_released());
                ASSERT_EQ(source->raw_data(), bytes);
                const auto ref = manager.preparedWeightStore()->preparedRefForBinding(
                    frozen[index].bindings().front().binding_id, device);
                ASSERT_TRUE(ref);
                refs[index] = *ref;
                const auto binding = manager.preparedWeightStore()->binding(*ref);
                ASSERT_TRUE(binding);
                EXPECT_EQ(binding->residency.host_policy,
                    host_policies[index]);
                EXPECT_EQ(manager.releaseAllHostWeightData(), 0u);
            }
            manager.markDevicePreparationComplete();
            EXPECT_EQ(manager.releaseAllHostWeightData(), 0u);
            ASSERT_EQ(source->raw_data(), bytes);
            manager.markGraphMaterializationComplete();
            EXPECT_EQ(manager.releaseAllHostWeightData(), cpu_retained ? 0u : 1u);
            EXPECT_EQ(source->is_raw_data_released(), !cpu_retained);
            EXPECT_EQ(manager.releaseAllHostWeightData(), 0u);
            const auto reads = loader.loadTensorCallCount();
            for (int index = 0; index < 2; ++index)
            {
                auto *kernel = manager.preparedWeightStore()->gemmKernel(refs[index]);
                ASSERT_NE(kernel, nullptr);
                const auto device = consumers.devices[index];
                const auto live = manager.retainedPreparedDeviceBytes(device);
                auto rebound = manager.materialize(sourcePlan(strategy, index, host_policies[index]));
                ASSERT_TRUE(manager.prepareWeightsForDevice(rebound, device, false));
                const auto ref = manager.preparedWeightStore()->preparedRefForBinding(
                    rebound.bindings().front().binding_id, device);
                ASSERT_TRUE(ref);
                EXPECT_EQ(manager.preparedWeightStore()->gemmKernel(*ref), kernel);
                EXPECT_EQ(manager.retainedPreparedDeviceBytes(device), live);
                EXPECT_EQ(loader.loadTensorCallCount(), reads);
                EXPECT_EQ(source->is_raw_data_released(), !cpu_retained);
            }
        }
    }

    TEST_P(PreparedWeightSourceLifetime, RetiredUnpreparedSourceFailsWithoutReload)
    {
        const auto consumers = GetParam();
        for (const auto &format : embeddingVerifierFormats())
        {
            SCOPED_TRACE(format.label);
            const ModelContextId model_id{34001};
            std::shared_ptr<TensorBase> source = format.create({256, 512}, 34001);
            MockModelLoader loader;
            loader.addTensor("blk.2.nextn.eh_proj.weight", source);
            WeightManager manager(loader);
            manager.setWeightShardingConfig(Qwen35MoESchemaFactory{}.getWeightShardingConfig());
            manager.setPreparedWeightStore(std::make_shared<PreparedWeightStore>(model_id));
            manager.installPhysicalMemoryAuthority(sourceMemory(*source, consumers));
            InferenceStrategy strategy;
            strategy.model_id = model_id;
            strategy.mode = WeightInferenceMode::LocalTP;
            strategy.tp_degree = 2;
            strategy.devices.assign(consumers.devices.begin(), consumers.devices.end());
            std::vector<FrozenModelWeightSet> frozen;
            for (int index = 0; index < 2; ++index)
                frozen.push_back(manager.materialize(sourcePlan(strategy, index)));
            const auto reads = loader.loadTensorCallCount();
            // Deliberately violate the source lifetime before any participant
            // owns a prepared handle. Setup must reject, never repair, this state.
            source->release_host_weight_data();
            for (int index = 0; index < 2; ++index)
            {
                const auto device = consumers.devices[index];
                SCOPED_TRACE(device.toString());
                try
                {
                    (void)manager.prepareWeightsForDevice(frozen[index], device, false);
                    FAIL() << "Missing immutable source was accepted";
                }
                catch (const std::runtime_error &error)
                {
                    const std::string diagnostic = error.what();
                    EXPECT_NE(diagnostic.find("live immutable source"), std::string::npos);
                    EXPECT_NE(diagnostic.find("blk.2.nextn.eh_proj.weight"), std::string::npos);
                    EXPECT_NE(diagnostic.find(device.toString()), std::string::npos);
                }
                EXPECT_EQ(loader.loadTensorCallCount(), reads);
                EXPECT_EQ(manager.preparedRecordCountForDevice(device), 0u);
                EXPECT_EQ(manager.retainedPreparedDeviceBytes(device), 0u);
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PreparedWeightSourceLifetime,
        ::testing::ValuesIn(consumerCases()), [](const auto &info) { return info.param.name; });
}
}
