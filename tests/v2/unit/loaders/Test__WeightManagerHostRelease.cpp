/**
 * @file Test__WeightManagerHostRelease.cpp
 * @brief Unit tests for WeightManager host memory release decision logic
 *
 * Tests verify:
 * 1. releaseAllHostWeightData() retains tensors with no device copy
 * 2. releaseAllHostWeightData() releases host-resident tensors with prepared device state
 * 3. releaseAllHostWeightData() retains host-resident tensors without prepared device state
 * 4. releaseAllHostWeightData() skips already-released tensors
 * 5. releaseHostResidentWeightData() releases only host-resident tensors
 * 6. releaseHostResidentWeightData() retains non-host-resident tensors
 * 7. releaseHostResidentWeightData() deduplicates shared tensor pointers
 * 8. Scheduled post-prefill reclaim owns residual host release
 * 9. Return counts are correct
 *
 * Uses lifecycle-gate control plus TensorBase prepared-device-state metadata to
 * exercise the release decision tree without depending on KernelFactory static state.
 */

#include <gtest/gtest.h>
#include <memory>

#include "config/TensorParallelConfig.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "mocks/MockModelLoader.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    void registerPreparedState(
        PreparedWeightStore &store,
        const std::shared_ptr<TensorBase> &tensor,
        uint64_t binding_id)
    {
        WeightBinding binding;
        binding.binding_id = binding_id;
        binding.identity = makeSourceWeightIdentity(
            "test.prepared." + std::to_string(binding_id),
            ModelContextId{99},
            binding_id);
        binding.tensor = tensor.get();
        binding.residency.home_device = DeviceId::cpu();
        binding.residency.resident_device = DeviceId::cpu();
        binding.immutable = true;

        auto handle =
            std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
        handle->tensor = tensor.get();
        handle->device_id = DeviceId::cpu();
        handle->kind =
            llaminar::v2::kernels::KernelFactory::GemmPreparationKind::CPU_PACKED;
        handle->prepared_weights =
            std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();

        store.registerPreparedGemmHandle(
            binding,
            PreparedWeightKind::CpuPackedGemm,
            DeviceId::cpu(),
            std::move(handle));
    }
}

// ============================================================================
// TestableWeightManager — opens lifecycle gates for release decision tests
// ============================================================================

class TestableWeightManager : public WeightManager
{
public:
    TestableWeightManager(IModelLoader &loader)
        : WeightManager(loader, nullptr, nullptr,
                        WeightDistributionStrategy::REPLICATED,
                        WeightPrecision::NATIVE)
    {
        // Mark all lifecycle gates open so release decision logic is testable
        markMaterializationComplete();
        markDevicePreparationComplete();
        markGraphMaterializationComplete();
    }
};

// ============================================================================
// Test Fixture
// ============================================================================

class Test__WeightManagerHostRelease : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_loader_ = MockModelLoader::createMinimal();

        // Add a few test tensors — Q8_0 for embedding-like, FP32 for norms
        mock_loader_->addQ8_0RandomTensor("token_embd.weight", {256, 64});
        mock_loader_->addFP32RandomTensor("output_norm.weight", {64});
        mock_loader_->addQ8_0RandomTensor("blk.0.attn_q.weight", {64, 64});
    }

    /// Load a tensor into the WeightManager cache via getWeightForDevice and return it
    std::shared_ptr<TensorBase> loadTensor(TestableWeightManager &wm, const std::string &name)
    {
        return wm.getWeightForDevice(name, DeviceId::cpu(), 0);
    }

    std::shared_ptr<MockModelLoader> mock_loader_;
};

// ============================================================================
// releaseAllHostWeightData() Tests
// ============================================================================

TEST_F(Test__WeightManagerHostRelease, RetainsWhenNoDeviceData)
{
    // Tensor loaded on CPU only, no GPU copy, not host-resident, no prepared device state
    // → should be retained (no safe device copy exists)
    TestableWeightManager wm(*mock_loader_);

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    ASSERT_FALSE(tensor->isHostResident());

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, ReleasesHostResidentWhenPreparedExists)
{
    // Tensor is host-resident AND has prepared device state → should release
    TestableWeightManager wm(*mock_loader_);

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();
    PreparedWeightStore prepared_store(ModelContextId{99});
    registerPreparedState(prepared_store, tensor, 1);
    ASSERT_TRUE(tensor->isHostResident());

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_GE(released, 1);
    EXPECT_TRUE(tensor->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, RetainsHostResidentWhenNoPrepared)
{
    // Tensor is host-resident but has NO prepared device state → should retain
    TestableWeightManager wm(*mock_loader_);

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, SkipsAlreadyReleased)
{
    // Tensor already released → releaseAll should skip it
    TestableWeightManager wm(*mock_loader_);

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();

    // Pre-release the tensor manually
    tensor->release_host_weight_data();
    ASSERT_TRUE(tensor->is_raw_data_released());

    size_t released = wm.releaseAllHostWeightData();

    // Should not count already-released tensors
    EXPECT_EQ(released, 0);
}

TEST_F(Test__WeightManagerHostRelease, ReturnsCorrectReleaseCount)
{
    // Load multiple tensors as host-resident with prepared device state
    TestableWeightManager wm(*mock_loader_);

    auto t1 = loadTensor(wm, "token_embd.weight");
    auto t2 = loadTensor(wm, "output_norm.weight");
    auto t3 = loadTensor(wm, "blk.0.attn_q.weight");
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);
    ASSERT_NE(t3, nullptr);

    // Mark only two as host-resident
    PreparedWeightStore prepared_store(ModelContextId{99});
    t1->setHostResident();
    registerPreparedState(prepared_store, t1, 1);
    t3->setHostResident();
    registerPreparedState(prepared_store, t3, 3);
    // t2 stays non host-resident with no device data → retained

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_EQ(released, 2);
    EXPECT_TRUE(t1->is_raw_data_released());
    EXPECT_FALSE(t2->is_raw_data_released()); // Retained: no device, not host-resident
    EXPECT_TRUE(t3->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, MixedStatesProcessedCorrectly)
{
    // Mix of: already-released, host-resident with prepared state, non-host-resident
    TestableWeightManager wm(*mock_loader_);

    auto already_released = loadTensor(wm, "token_embd.weight");
    auto host_resident = loadTensor(wm, "output_norm.weight");
    auto cpu_only = loadTensor(wm, "blk.0.attn_q.weight");

    // Pre-release one
    already_released->release_host_weight_data();
    // Mark one host-resident
    host_resident->setHostResident();
    PreparedWeightStore prepared_store(ModelContextId{99});
    registerPreparedState(prepared_store, host_resident, 2);
    // cpu_only stays default (no device, not host-resident)

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_EQ(released, 1);                                // Only host_resident should be released
    EXPECT_TRUE(already_released->is_raw_data_released()); // Was already
    EXPECT_TRUE(host_resident->is_raw_data_released());    // Newly released
    EXPECT_FALSE(cpu_only->is_raw_data_released());        // Retained
}

// ============================================================================
// releaseHostResidentWeightData() Tests
// ============================================================================

TEST_F(Test__WeightManagerHostRelease, ReleaseHostResident_ReleasesOnlyHostResident)
{
    TestableWeightManager wm(*mock_loader_);

    auto embd = loadTensor(wm, "token_embd.weight");
    auto norm = loadTensor(wm, "output_norm.weight");
    ASSERT_NE(embd, nullptr);
    ASSERT_NE(norm, nullptr);

    // Only mark embedding as host-resident
    embd->setHostResident();

    size_t released = wm.releaseHostResidentWeightData();

    EXPECT_EQ(released, 1);
    EXPECT_TRUE(embd->is_raw_data_released());
    EXPECT_FALSE(norm->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, ReleaseHostResident_RetainsNonHostResident)
{
    TestableWeightManager wm(*mock_loader_);

    auto tensor = loadTensor(wm, "blk.0.attn_q.weight");
    ASSERT_NE(tensor, nullptr);
    ASSERT_FALSE(tensor->isHostResident());

    size_t released = wm.releaseHostResidentWeightData();

    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, ReleaseHostResident_SkipsAlreadyReleased)
{
    TestableWeightManager wm(*mock_loader_);

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();

    // Pre-release
    tensor->release_host_weight_data();
    ASSERT_TRUE(tensor->is_raw_data_released());

    size_t released = wm.releaseHostResidentWeightData();

    EXPECT_EQ(released, 0);
}

TEST_F(Test__WeightManagerHostRelease, ReleaseHostResident_ReturnsZeroWhenNoneHostResident)
{
    TestableWeightManager wm(*mock_loader_);

    // Load tensors but don't mark any as host-resident
    loadTensor(wm, "token_embd.weight");
    loadTensor(wm, "output_norm.weight");

    size_t released = wm.releaseHostResidentWeightData();

    EXPECT_EQ(released, 0);
}

TEST_F(Test__WeightManagerHostRelease, ReleaseHostResident_SkipsBorrowedTensorSliceViews)
{
    constexpr const char *kWeightName = "blk.0.ffn_gate.weight";
    mock_loader_->addFP32RandomTensor(kWeightName, {64, 16});

    TestableWeightManager wm(*mock_loader_);

    WeightShardingConfig sharding;
    sharding.exact_matches[kWeightName] = WeightShardingMode::ColumnParallel;
    sharding.exact_dimension_matches[kWeightName] = WeightDimensionType::FFNHidden;
    wm.setWeightShardingConfig(sharding);

    const std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(
            2,
            4,
            2,
            64,
            128,
            devices));
    wm.setTensorParallelConfig(tp_config);

    const auto &assignment = tp_config->forDevice(DeviceId::cuda(1));
    auto slice = wm.getShardedWeightForAssignment(kWeightName, DeviceId::cuda(1), assignment, 0);
    ASSERT_NE(slice, nullptr);
    ASSERT_TRUE(slice->is_view());

    slice->setHostResident();
    ASSERT_TRUE(slice->isHostResident());
    ASSERT_FALSE(slice->is_raw_data_released());

    size_t released = wm.releaseHostResidentWeightData();

    EXPECT_EQ(released, 0);
    EXPECT_FALSE(slice->is_raw_data_released())
        << "Borrowed TensorSlice views must not release their wrapped storage during the broad host-resident sweep";
}

TEST_F(
    Test__WeightManagerHostRelease,
    ScheduledPostPrefillReclaimReleasesResidualHostWeightsBeforeAllocationBarrier)
{
    TestableWeightManager wm(*mock_loader_);

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();
    ASSERT_FALSE(tensor->is_raw_data_released());

    EXPECT_EQ(
        wm.scheduleMmapReclaim(),
        MmapReclaimLifecycle::Submission::Scheduled);
    const auto completion =
        wm.awaitMmapReclaimBeforeHostAllocation();

    EXPECT_EQ(completion.state, MmapReclaimLifecycle::State::Complete);
    EXPECT_TRUE(tensor->is_raw_data_released())
        << "The worker must complete residual host release before model/JIT allocation proceeds";
    EXPECT_EQ(
        wm.scheduleMmapReclaim(),
        MmapReclaimLifecycle::Submission::AlreadyComplete);
}
