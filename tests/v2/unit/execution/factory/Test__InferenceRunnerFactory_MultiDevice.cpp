/**
 * @file Test__InferenceRunnerFactory_MultiDevice.cpp
 * @brief Unit tests for MultiDevice factory functions in InferenceRunnerFactory
 *
 * Tests the createMultiDeviceOrchestrator and createTestableMultiDeviceOrchestrator
 * factory functions with various configurations.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/orchestrators/IMultiDeviceOrchestrator.h"
#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "mocks/MockModelContext.h"

using namespace llaminar2;
using namespace llaminar2::test;
using namespace testing;

namespace
{

    // =============================================================================
    // Mock LocalTPContext for factory tests
    // =============================================================================

    /**
     * @brief Simple mock for ILocalTPContext used in factory tests
     */
    class MockLocalTPContext : public ILocalTPContext
    {
    public:
        MockLocalTPContext(
            std::vector<GlobalDeviceAddress> devices,
            std::vector<float> weights)
            : devices_(std::move(devices)), weights_(std::move(weights))
        {
        }

        const std::vector<GlobalDeviceAddress> &devices() const override { return devices_; }
        const std::vector<float> &weights() const override { return weights_; }
        CollectiveBackendType backend() const override { return CollectiveBackendType::HOST; }
        int degree() const override { return static_cast<int>(devices_.size()); }
        int myIndex() const override { return 0; }

        bool allreduce(TensorBase * /*tensor*/) override { return true; }
        bool allreduce(TensorBase *tensor, const std::string & /*stage_name*/, size_t /*count*/ = 0) override { return allreduce(tensor); }
        bool allreduce(const TensorBase * /*input*/, TensorBase * /*output*/) override { return true; }
        bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override { return true; }
        bool reduceScatter(const TensorBase * /*input*/, TensorBase * /*output_shard*/) override { return true; }

        void synchronize() override {}

        int indexForDevice(const GlobalDeviceAddress &device) const override
        {
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device)
                {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        const GlobalDeviceAddress &deviceAt(int index) const override
        {
            return devices_.at(static_cast<size_t>(index));
        }

        float weightForDevice(const GlobalDeviceAddress &device) const override
        {
            int idx = indexForDevice(device);
            return (idx >= 0) ? weights_[static_cast<size_t>(idx)] : 0.0f;
        }

        int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
        {
            float w = weightForDevice(device);
            return static_cast<int>(w * static_cast<float>(total_heads) + 0.5f);
        }

        std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const override
        {
            int idx = indexForDevice(device);
            if (idx < 0)
                return {0, 0};

            float cumulative = 0.0f;
            for (int i = 0; i < idx; ++i)
            {
                cumulative += weights_[static_cast<size_t>(i)];
            }
            int start = static_cast<int>(cumulative * static_cast<float>(total_rows));
            int end = static_cast<int>((cumulative + weights_[static_cast<size_t>(idx)]) * static_cast<float>(total_rows));
            return {start, end};
        }

        std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const override
        {
            return rowRangeForDevice(device, total_cols);
        }

        bool gatherFromDevices(
            const std::vector<const TensorBase *> & /*shards*/,
            TensorBase * /*output*/) override
        {
            return true;
        }

        // BAR Registry (no-ops for tests)
        void registerBARBackedOutput(
            const std::string & /*stage_name*/,
            const GlobalDeviceAddress & /*device*/,
            TensorBase * /*tensor*/) override
        {
        }
        bool hasBARBackedOutputs(const std::string & /*stage_name*/) const override { return false; }
        void clearBARBackedOutputs() override {}
        bool reserveTempBufferBytes(size_t /*bytes*/) override { return true; }

        // Broadcast (no-op)
        bool broadcast(TensorBase * /*tensor*/, int /*source_device_index*/ = 0) override { return true; }

        void requestAbort() override {}
        bool isAbortRequested() const override { return false; }

    private:
        std::vector<GlobalDeviceAddress> devices_;
        std::vector<float> weights_;
    };

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__InferenceRunnerFactory_MultiDevice : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            model_ctx_ = MockModelContextBuilder()
                             .usePreset(ModelPreset::MINIMAL)
                             .build();
        }

        std::shared_ptr<MockModelContext> model_ctx_;
    };

    // =============================================================================
    // createMultiDeviceOrchestrator Tests
    // =============================================================================

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, NullModelContextReturnsNull)
    {
        // Create valid TP context
        auto tp_ctx = std::make_unique<MockLocalTPContext>(
            std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cpu()},
            std::vector<float>{1.0f});

        MultiDeviceOrchestrator::Config config;
        config.devices = tp_ctx->devices();

        auto result = createMultiDeviceOrchestrator(nullptr, std::move(tp_ctx), config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, NullTPContextReturnsNull)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};

        auto result = createMultiDeviceOrchestrator(model_ctx_, nullptr, config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, EmptyDevicesReturnsNull)
    {
        auto tp_ctx = std::make_unique<MockLocalTPContext>(
            std::vector<GlobalDeviceAddress>{},
            std::vector<float>{});

        MultiDeviceOrchestrator::Config config;
        // config.devices is empty - should fail validation

        auto result = createMultiDeviceOrchestrator(model_ctx_, std::move(tp_ctx), config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, InvalidWeightsReturnsNull)
    {
        auto tp_ctx = std::make_unique<MockLocalTPContext>(
            std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()},
            std::vector<float>{0.3f, 0.3f}); // Sum != 1.0

        MultiDeviceOrchestrator::Config config;
        config.devices = tp_ctx->devices();
        config.weights = {0.3f, 0.3f}; // Invalid - doesn't sum to 1.0

        auto result = createMultiDeviceOrchestrator(model_ctx_, std::move(tp_ctx), config);
        EXPECT_EQ(result, nullptr);
    }

    // =============================================================================
    // createTestableMultiDeviceOrchestrator Tests
    // =============================================================================

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, TestableWithNullModelCtxReturnsNull)
    {
        std::vector<std::unique_ptr<IInferenceRunner>> runners;
        // Can't create real runners without model - this tests null check

        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};

        auto result = createTestableMultiDeviceOrchestrator(
            nullptr, std::move(runners), nullptr, config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, TestableWithEmptyRunnersReturnsNull)
    {
        std::vector<std::unique_ptr<IInferenceRunner>> empty_runners;

        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};

        auto result = createTestableMultiDeviceOrchestrator(
            model_ctx_, std::move(empty_runners), nullptr, config);
        EXPECT_EQ(result, nullptr);
    }

    // =============================================================================
    // Config Validation Tests
    // =============================================================================

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateEmptyDevicesFails)
    {
        MultiDeviceOrchestrator::Config config;
        EXPECT_FALSE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateSingleDeviceSucceeds)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateTwoDevicesSucceeds)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateWithEqualWeightsSucceeds)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.5f, 0.5f};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateWithUnequalWeightsSucceeds)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.73f, 0.27f};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateWeightCountMismatchFails)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {1.0f}; // Only one weight for two devices
        EXPECT_FALSE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateWeightsSumWrongFails)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.4f, 0.4f}; // Sum to 0.8, not 1.0
        EXPECT_FALSE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, GetNormalizedWeightsDefault)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        // No weights set

        auto weights = config.getNormalizedWeights();
        ASSERT_EQ(weights.size(), 2u);
        EXPECT_FLOAT_EQ(weights[0], 0.5f);
        EXPECT_FLOAT_EQ(weights[1], 0.5f);
    }

    TEST(Test__MultiDeviceOrchestratorConfig, GetNormalizedWeightsExplicit)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.7f, 0.3f};

        auto weights = config.getNormalizedWeights();
        ASSERT_EQ(weights.size(), 2u);
        EXPECT_FLOAT_EQ(weights[0], 0.7f);
        EXPECT_FLOAT_EQ(weights[1], 0.3f);
    }

    TEST(Test__MultiDeviceOrchestratorConfig, GetNormalizedWeightsThreeDevices)
    {
        MultiDeviceOrchestrator::Config config;
        config.devices = {
            GlobalDeviceAddress::cpu(),
            GlobalDeviceAddress::cpu(),
            GlobalDeviceAddress::cpu()};
        // No weights set - should get equal distribution

        auto weights = config.getNormalizedWeights();
        ASSERT_EQ(weights.size(), 3u);
        EXPECT_NEAR(weights[0], 1.0f / 3.0f, 0.0001f);
        EXPECT_NEAR(weights[1], 1.0f / 3.0f, 0.0001f);
        EXPECT_NEAR(weights[2], 1.0f / 3.0f, 0.0001f);
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateWeightsWithSmallTolerance)
    {
        // Test that weights summing to ~1.0 with floating point error still validate
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.6666666f, 0.3333334f}; // Sum = 1.0000000

        EXPECT_TRUE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, ValidateMixedDeviceTypes)
    {
        // Test heterogeneous device configuration
        MultiDeviceOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
        config.weights = {0.73f, 0.27f};

        EXPECT_TRUE(config.validate());
    }

    TEST(Test__MultiDeviceOrchestratorConfig, DefaultBackendIsAuto)
    {
        MultiDeviceOrchestrator::Config config;
        EXPECT_EQ(config.backend, CollectiveBackendType::AUTO);
    }

    TEST(Test__MultiDeviceOrchestratorConfig, DefaultMaxSeqLen)
    {
        MultiDeviceOrchestrator::Config config;
        EXPECT_EQ(config.max_seq_len, 4096u);
    }

    TEST(Test__MultiDeviceOrchestratorConfig, DefaultBatchSize)
    {
        MultiDeviceOrchestrator::Config config;
        EXPECT_EQ(config.batch_size, 1);
    }

} // namespace
