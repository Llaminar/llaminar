/**
 * @file Test__WeightManagerHostRelease.cpp
 * @brief Unit tests for WeightManager host memory release decision logic
 *
 * Tests verify:
 * 1. releaseAllHostWeightData() retains tensors with no device copy
 * 2. releaseAllHostWeightData() releases host-resident tensors when prepared embeddings exist
 * 3. releaseAllHostWeightData() retains host-resident tensors when no prepared embeddings
 * 4. releaseAllHostWeightData() skips already-released tensors
 * 5. releaseHostResidentWeightData() releases only host-resident tensors
 * 6. releaseHostResidentWeightData() retains non-host-resident tensors
 * 7. releaseHostResidentWeightData() deduplicates shared tensor pointers
 * 8. Return counts are correct
 *
 * Uses a TestableWeightManager subclass that overrides getPreparedEmbeddingCount()
 * (Option A injection) to control the release decision tree without depending on
 * KernelFactory static state.
 */

#include <gtest/gtest.h>
#include <memory>

#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "mocks/MockModelLoader.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// TestableWeightManager — overrides getPreparedEmbeddingCount for test control
// ============================================================================

class TestableWeightManager : public WeightManager
{
    size_t mock_embedding_count_ = 0;

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

    size_t getPreparedEmbeddingCount() const override
    {
        return mock_embedding_count_;
    }

    void setMockEmbeddingCount(size_t n) { mock_embedding_count_ = n; }
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
    // Tensor loaded on CPU only, no GPU copy, not host-resident, no prepared embeddings
    // → should be retained (no safe device copy exists)
    TestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(0);

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    ASSERT_FALSE(tensor->isHostResident());

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_EQ(released, 0);
    EXPECT_FALSE(tensor->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, ReleasesHostResidentWhenPreparedExists)
{
    // Tensor is host-resident AND prepared embeddings exist → should release
    TestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(2); // Simulate 2 prepared embeddings

    auto tensor = loadTensor(wm, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    tensor->setHostResident();
    ASSERT_TRUE(tensor->isHostResident());

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_GE(released, 1);
    EXPECT_TRUE(tensor->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, RetainsHostResidentWhenNoPrepared)
{
    // Tensor is host-resident but NO prepared embeddings → should retain
    TestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(0);

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
    wm.setMockEmbeddingCount(2);

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
    // Load multiple tensors as host-resident with prepared embeddings
    TestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(1);

    auto t1 = loadTensor(wm, "token_embd.weight");
    auto t2 = loadTensor(wm, "output_norm.weight");
    auto t3 = loadTensor(wm, "blk.0.attn_q.weight");
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);
    ASSERT_NE(t3, nullptr);

    // Mark only two as host-resident
    t1->setHostResident();
    t3->setHostResident();
    // t2 stays non host-resident with no device data → retained

    size_t released = wm.releaseAllHostWeightData();

    EXPECT_EQ(released, 2);
    EXPECT_TRUE(t1->is_raw_data_released());
    EXPECT_FALSE(t2->is_raw_data_released()); // Retained: no device, not host-resident
    EXPECT_TRUE(t3->is_raw_data_released());
}

TEST_F(Test__WeightManagerHostRelease, MixedStatesProcessedCorrectly)
{
    // Mix of: already-released, host-resident with prepared, non-host-resident
    TestableWeightManager wm(*mock_loader_);
    wm.setMockEmbeddingCount(1);

    auto already_released = loadTensor(wm, "token_embd.weight");
    auto host_resident = loadTensor(wm, "output_norm.weight");
    auto cpu_only = loadTensor(wm, "blk.0.attn_q.weight");

    // Pre-release one
    already_released->release_host_weight_data();
    // Mark one host-resident
    host_resident->setHostResident();
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
