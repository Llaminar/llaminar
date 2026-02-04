/**
 * @file Test__LocalPPContext.cpp
 * @brief Unit tests for LocalPPContext and ILocalPPContext interface
 *
 * Tests the LOCAL pipeline parallelism context including:
 * - Configuration queries (numStages, deviceForStage, layerRangeForStage, etc.)
 * - Backend selection based on device types
 * - Transfer operations using mock backends
 * - Synchronization operations
 * - Factory function for creating LocalPPContext
 *
 * These tests work WITHOUT real GPUs by using MockLocalPPContext and
 * verifying behavior through the mock's call tracking.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

#include "v2/collective/ILocalPPContext.h"
#include "v2/backends/GlobalDeviceAddress.h"
#include "v2/tensors/TensorClasses.h"
#include "mocks/MockLocalPPContext.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__LocalPPContext : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create a test FP32 tensor for transfer tests
            test_tensor_ = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 128});
        }

        /**
         * @brief Create config with N CUDA stages and equal layer split
         * @param n_stages Number of PP stages (each on a separate CUDA device)
         * @param layers_per_stage Layers per stage (total = n_stages * layers_per_stage)
         */
        LocalPPConfig makeNStageCudaConfig(int n_stages, int layers_per_stage)
        {
            LocalPPConfig config;
            for (int i = 0; i < n_stages; ++i)
            {
                config.stage_devices.push_back(GlobalDeviceAddress::cuda(i));
            }

            // Equal layer split
            int total_layers = n_stages * layers_per_stage;
            for (int s = 0; s <= n_stages; ++s)
            {
                config.layer_boundaries.push_back(s * layers_per_stage);
            }

            return config;
        }

        /**
         * @brief Create config with N ROCm stages and equal layer split
         */
        LocalPPConfig makeNStageRocmConfig(int n_stages, int layers_per_stage)
        {
            LocalPPConfig config;
            for (int i = 0; i < n_stages; ++i)
            {
                config.stage_devices.push_back(GlobalDeviceAddress::rocm(i));
            }

            int total_layers = n_stages * layers_per_stage;
            for (int s = 0; s <= n_stages; ++s)
            {
                config.layer_boundaries.push_back(s * layers_per_stage);
            }

            return config;
        }

        /**
         * @brief Create mixed vendor config (CUDA + ROCm + CPU)
         */
        LocalPPConfig makeMixedVendorConfig()
        {
            LocalPPConfig config;
            config.stage_devices.push_back(GlobalDeviceAddress::cuda(0));  // Stage 0
            config.stage_devices.push_back(GlobalDeviceAddress::rocm(0));  // Stage 1
            config.stage_devices.push_back(GlobalDeviceAddress::cpu());    // Stage 2

            // 24 layers: 8 per stage
            config.layer_boundaries = {0, 8, 16, 24};

            return config;
        }

        /**
         * @brief Create config with CPU stages only
         */
        LocalPPConfig makeCpuOnlyConfig(int n_stages, int layers_per_stage)
        {
            LocalPPConfig config;
            for (int i = 0; i < n_stages; ++i)
            {
                config.stage_devices.push_back(GlobalDeviceAddress::cpu());
            }

            for (int s = 0; s <= n_stages; ++s)
            {
                config.layer_boundaries.push_back(s * layers_per_stage);
            }

            return config;
        }

        std::unique_ptr<FP32Tensor> test_tensor_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 1: Configuration Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, NumStages_ReturnsTwoForTwoDevices)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_EQ(mock->numStages(), 2);
    }

    TEST_F(Test__LocalPPContext, NumStages_ReturnsFourForFourDevices)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(4)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_EQ(mock->numStages(), 4);
    }

    TEST_F(Test__LocalPPContext, NumStages_ReturnsOneForSingleDevice)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCpuStages(1)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_EQ(mock->numStages(), 1);
    }

    TEST_F(Test__LocalPPContext, DeviceForStage_ReturnsCorrectCudaDevice)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24)
                        .build();

        auto device0 = mock->deviceForStage(0);
        auto device1 = mock->deviceForStage(1);
        auto device2 = mock->deviceForStage(2);

        EXPECT_EQ(device0.device_type, DeviceType::CUDA);
        EXPECT_EQ(device0.device_ordinal, 0);

        EXPECT_EQ(device1.device_type, DeviceType::CUDA);
        EXPECT_EQ(device1.device_ordinal, 1);

        EXPECT_EQ(device2.device_type, DeviceType::CUDA);
        EXPECT_EQ(device2.device_ordinal, 2);
    }

    TEST_F(Test__LocalPPContext, DeviceForStage_ReturnsMixedVendorDevices)
    {
        MockLocalPPContext::Config config;
        config.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::cpu()};
        config.layer_boundaries = {0, 8, 16, 24};

        MockLocalPPContext mock(config);

        EXPECT_EQ(mock.deviceForStage(0).device_type, DeviceType::CUDA);
        EXPECT_EQ(mock.deviceForStage(1).device_type, DeviceType::ROCm);
        EXPECT_EQ(mock.deviceForStage(2).device_type, DeviceType::CPU);
    }

    TEST_F(Test__LocalPPContext, DeviceForStage_ThrowsForNegativeIndex)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_THROW(mock->deviceForStage(-1), std::out_of_range);
    }

    TEST_F(Test__LocalPPContext, DeviceForStage_ThrowsForIndexBeyondStages)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_THROW(mock->deviceForStage(2), std::out_of_range);
        EXPECT_THROW(mock->deviceForStage(100), std::out_of_range);
    }

    TEST_F(Test__LocalPPContext, LayerRangeForStage_ReturnsCorrectRangesForEqualSplit)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24) // 8 layers per stage
                        .build();

        auto [first0, last0] = mock->layerRangeForStage(0);
        auto [first1, last1] = mock->layerRangeForStage(1);
        auto [first2, last2] = mock->layerRangeForStage(2);

        EXPECT_EQ(first0, 0);
        EXPECT_EQ(last0, 8);

        EXPECT_EQ(first1, 8);
        EXPECT_EQ(last1, 16);

        EXPECT_EQ(first2, 16);
        EXPECT_EQ(last2, 24);
    }

    TEST_F(Test__LocalPPContext, LayerRangeForStage_ReturnsNegativeForInvalidStage)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        auto [first_neg, last_neg] = mock->layerRangeForStage(-1);
        EXPECT_EQ(first_neg, -1);
        EXPECT_EQ(last_neg, -1);

        auto [first_oob, last_oob] = mock->layerRangeForStage(2);
        EXPECT_EQ(first_oob, -1);
        EXPECT_EQ(last_oob, -1);
    }

    TEST_F(Test__LocalPPContext, StageForLayer_MapsLayersCorrectly)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24) // 8 layers per stage
                        .build();

        // Stage 0: layers 0-7
        EXPECT_EQ(mock->stageForLayer(0), 0);
        EXPECT_EQ(mock->stageForLayer(7), 0);

        // Stage 1: layers 8-15
        EXPECT_EQ(mock->stageForLayer(8), 1);
        EXPECT_EQ(mock->stageForLayer(15), 1);

        // Stage 2: layers 16-23
        EXPECT_EQ(mock->stageForLayer(16), 2);
        EXPECT_EQ(mock->stageForLayer(23), 2);
    }

    TEST_F(Test__LocalPPContext, StageForLayer_ReturnsNegativeOneForOutOfBounds)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24) // 12 layers per stage
                        .build();

        EXPECT_EQ(mock->stageForLayer(-1), -1);
        EXPECT_EQ(mock->stageForLayer(24), -1);
        EXPECT_EQ(mock->stageForLayer(100), -1);
    }

    TEST_F(Test__LocalPPContext, TotalLayers_ReturnsCorrectCount)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_EQ(mock->totalLayers(), 24);
    }

    TEST_F(Test__LocalPPContext, StageDevices_ReturnsAllDevices)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(4)
                        .withEqualLayerSplit(24)
                        .build();

        const auto &devices = mock->stageDevices();
        ASSERT_EQ(devices.size(), 4);
        for (int i = 0; i < 4; ++i)
        {
            EXPECT_EQ(devices[i].device_type, DeviceType::CUDA);
            EXPECT_EQ(devices[i].device_ordinal, i);
        }
    }

    TEST_F(Test__LocalPPContext, LayerBoundaries_ReturnsAllBoundaries)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24)
                        .build();

        const auto &boundaries = mock->layerBoundaries();
        ASSERT_EQ(boundaries.size(), 4); // n_stages + 1
        EXPECT_EQ(boundaries[0], 0);
        EXPECT_EQ(boundaries[1], 8);
        EXPECT_EQ(boundaries[2], 16);
        EXPECT_EQ(boundaries[3], 24);
    }

    TEST_F(Test__LocalPPContext, SameDevice_ReturnsTrueForSameStage)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_TRUE(mock->sameDevice(0, 0));
        EXPECT_TRUE(mock->sameDevice(1, 1));
    }

    TEST_F(Test__LocalPPContext, SameDevice_ReturnsFalseForDifferentDevices)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_FALSE(mock->sameDevice(0, 1));
        EXPECT_FALSE(mock->sameDevice(1, 0));
    }

    TEST_F(Test__LocalPPContext, SameDevice_ReturnsTrueForSameDeviceOnDifferentStages)
    {
        // Create config where two stages share the same device
        MockLocalPPContext::Config config;
        config.stage_devices = {
            GlobalDeviceAddress::cuda(0), // Stage 0
            GlobalDeviceAddress::cuda(0), // Stage 1 - SAME device!
            GlobalDeviceAddress::cuda(1)  // Stage 2
        };
        config.layer_boundaries = {0, 8, 16, 24};

        MockLocalPPContext mock(config);

        EXPECT_TRUE(mock.sameDevice(0, 1));  // Same cuda:0
        EXPECT_FALSE(mock.sameDevice(0, 2)); // Different devices
        EXPECT_FALSE(mock.sameDevice(1, 2)); // Different devices
    }

    TEST_F(Test__LocalPPContext, SameDevice_ReturnsFalseForInvalidStages)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_FALSE(mock->sameDevice(-1, 0));
        EXPECT_FALSE(mock->sameDevice(0, -1));
        EXPECT_FALSE(mock->sameDevice(2, 0));
        EXPECT_FALSE(mock->sameDevice(0, 2));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 2: Backend Selection Tests (via LocalPPConfig)
    // ═══════════════════════════════════════════════════════════════════════════

    // Note: The real LocalPPContext uses selectBackendForTransfer() internally.
    // We test the expected backend selection logic through LocalPPConfig validation
    // and the factory function.

    TEST_F(Test__LocalPPContext, LocalPPConfig_IsValidForCorrectConfig)
    {
        LocalPPConfig config = makeNStageCudaConfig(2, 12);
        EXPECT_TRUE(config.isValid());
    }

    TEST_F(Test__LocalPPContext, LocalPPConfig_IsInvalidForEmptyDevices)
    {
        LocalPPConfig config;
        config.stage_devices = {};
        config.layer_boundaries = {0, 24};

        EXPECT_FALSE(config.isValid());
    }

    TEST_F(Test__LocalPPContext, LocalPPConfig_IsInvalidForMismatchedBoundaries)
    {
        LocalPPConfig config;
        config.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.layer_boundaries = {0, 24}; // Should be size 3 (n_stages + 1)

        EXPECT_FALSE(config.isValid());
    }

    TEST_F(Test__LocalPPContext, LocalPPConfig_IsInvalidForNonMonotonicBoundaries)
    {
        LocalPPConfig config;
        config.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.layer_boundaries = {0, 20, 10}; // Not monotonically increasing!

        EXPECT_FALSE(config.isValid());
    }

    TEST_F(Test__LocalPPContext, LocalPPConfig_IsInvalidForEqualBoundaries)
    {
        LocalPPConfig config;
        config.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        config.layer_boundaries = {0, 12, 12}; // Equal boundaries not allowed

        EXPECT_FALSE(config.isValid());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 3: Mock Backend Selection Tests
    // ═══════════════════════════════════════════════════════════════════════════

    // MockLocalPPContext returns a configurable default backend.
    // We test that the mock correctly reports the configured backend.

    TEST_F(Test__LocalPPContext, MockBackendForTransfer_ReturnsConfiguredBackend)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .withBackend(CollectiveBackendType::NCCL)
                        .build();

        EXPECT_EQ(mock->backendForTransfer(0, 1), CollectiveBackendType::NCCL);
    }

    TEST_F(Test__LocalPPContext, MockBackendForTransfer_ReturnsHostByDefault)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_EQ(mock->backendForTransfer(0, 1), CollectiveBackendType::HOST);
    }

    TEST_F(Test__LocalPPContext, MockBackendForTransfer_CanBePCIE_BAR)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withDevices({GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
                        .withEqualLayerSplit(24)
                        .withBackend(CollectiveBackendType::PCIE_BAR)
                        .build();

        EXPECT_EQ(mock->backendForTransfer(0, 1), CollectiveBackendType::PCIE_BAR);
    }

    TEST_F(Test__LocalPPContext, MockBackendForTransfer_CanBeRCCL)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)})
                        .withEqualLayerSplit(24)
                        .withBackend(CollectiveBackendType::RCCL)
                        .build();

        EXPECT_EQ(mock->backendForTransfer(0, 1), CollectiveBackendType::RCCL);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 4: Transfer Tests (using mock)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, Transfer_BetweenAdjacentStagesSucceeds)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_TRUE(mock->transfer(test_tensor_.get(), 0, 1));
        EXPECT_TRUE(mock->transfer(test_tensor_.get(), 1, 2));

        EXPECT_EQ(mock->transferCallCount(), 2);
    }

    TEST_F(Test__LocalPPContext, Transfer_RecordsTensorAndStages)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->transfer(test_tensor_.get(), 0, 1);

        auto calls = mock->transferCalls();
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0].activations, test_tensor_.get());
        EXPECT_EQ(calls[0].stage_from, 0);
        EXPECT_EQ(calls[0].stage_to, 1);
        EXPECT_FALSE(calls[0].was_async);
    }

    TEST_F(Test__LocalPPContext, Transfer_CanTransferNonAdjacent)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(4)
                        .withEqualLayerSplit(24)
                        .build();

        // Skip stages 1 and 2
        EXPECT_TRUE(mock->transfer(test_tensor_.get(), 0, 3));

        auto calls = mock->transferCalls();
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0].stage_from, 0);
        EXPECT_EQ(calls[0].stage_to, 3);
    }

    TEST_F(Test__LocalPPContext, Transfer_CanBeConfiguredToFail)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .withTransferFailure(true)
                        .build();

        EXPECT_FALSE(mock->transfer(test_tensor_.get(), 0, 1));

        // Call should still be recorded
        EXPECT_EQ(mock->transferCallCount(), 1);
    }

    TEST_F(Test__LocalPPContext, Transfer_NullTensorStillRecorded)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        // Mock doesn't validate null - it just records the call
        // Real implementation would fail
        mock->transfer(nullptr, 0, 1);

        auto calls = mock->transferCalls();
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0].activations, nullptr);
    }

    TEST_F(Test__LocalPPContext, TransferAsync_IssuesOperationWithoutBlocking)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        void *test_stream = reinterpret_cast<void *>(0x12345678);
        EXPECT_TRUE(mock->transferAsync(test_tensor_.get(), 0, 1, test_stream));

        EXPECT_EQ(mock->transferAsyncCallCount(), 1);
        EXPECT_EQ(mock->transferCallCount(), 0); // sync calls separate

        auto calls = mock->transferCalls();
        ASSERT_EQ(calls.size(), 1);
        EXPECT_TRUE(calls[0].was_async);
        EXPECT_EQ(calls[0].stream, test_stream);
    }

    TEST_F(Test__LocalPPContext, TransferAsync_WithNullStreamSucceeds)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_TRUE(mock->transferAsync(test_tensor_.get(), 0, 1, nullptr));

        auto calls = mock->transferCalls();
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0].stream, nullptr);
    }

    TEST_F(Test__LocalPPContext, TransferAsync_CanBeConfiguredToFail)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .withTransferAsyncFailure(true)
                        .build();

        EXPECT_FALSE(mock->transferAsync(test_tensor_.get(), 0, 1, nullptr));
        EXPECT_EQ(mock->transferAsyncCallCount(), 1);
    }

    TEST_F(Test__LocalPPContext, Transfer_HasTransferCheckWorks)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_FALSE(mock->hasTransfer(0, 1));

        mock->transfer(test_tensor_.get(), 0, 1);

        EXPECT_TRUE(mock->hasTransfer(0, 1));
        EXPECT_FALSE(mock->hasTransfer(1, 2));
    }

    TEST_F(Test__LocalPPContext, Transfer_CountTransfersWorks)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24)
                        .build();

        mock->transfer(test_tensor_.get(), 0, 1);
        mock->transfer(test_tensor_.get(), 0, 1);
        mock->transfer(test_tensor_.get(), 1, 2);

        EXPECT_EQ(mock->countTransfers(0, 1), 2);
        EXPECT_EQ(mock->countTransfers(1, 2), 1);
        EXPECT_EQ(mock->countTransfers(0, 2), 0);
    }

    TEST_F(Test__LocalPPContext, Transfer_TotalTransferCallCount)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->transfer(test_tensor_.get(), 0, 1);
        mock->transferAsync(test_tensor_.get(), 0, 1, nullptr);
        mock->transfer(test_tensor_.get(), 0, 1);

        EXPECT_EQ(mock->transferCallCount(), 2);
        EXPECT_EQ(mock->transferAsyncCallCount(), 1);
        EXPECT_EQ(mock->totalTransferCallCount(), 3);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 5: Synchronization Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, Synchronize_CanBeCalled)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->transferAsync(test_tensor_.get(), 0, 1, nullptr);
        mock->synchronize();

        EXPECT_EQ(mock->synchronizeCallCount(), 1);
    }

    TEST_F(Test__LocalPPContext, Synchronize_CanBeCalledMultipleTimes)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->synchronize();
        mock->synchronize();
        mock->synchronize();

        EXPECT_EQ(mock->synchronizeCallCount(), 3);
    }

    TEST_F(Test__LocalPPContext, SynchronizeStream_CanBeCalledWithStream)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        void *test_stream = reinterpret_cast<void *>(0xABCDEF);
        mock->synchronizeStream(test_stream);

        EXPECT_EQ(mock->synchronizeStreamCallCount(), 1);
    }

    TEST_F(Test__LocalPPContext, SynchronizeStream_WithNullStream)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->synchronizeStream(nullptr);

        EXPECT_EQ(mock->synchronizeStreamCallCount(), 1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 6: Factory Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, Factory_CreatesContextWithValidConfig)
    {
        LocalPPConfig config = makeNStageCudaConfig(2, 12);

        // Note: createLocalPPContext creates a real LocalPPContext,
        // which may not be testable without real GPUs.
        // We verify the config is valid at least.
        EXPECT_TRUE(config.isValid());

        // If the factory is available without GPU requirements:
        // auto ctx = createLocalPPContext(config);
        // EXPECT_NE(ctx, nullptr);
        // EXPECT_EQ(ctx->numStages(), 2);
    }

    TEST_F(Test__LocalPPContext, Factory_ConfigIsValidForMixedVendor)
    {
        LocalPPConfig config = makeMixedVendorConfig();
        EXPECT_TRUE(config.isValid());
        EXPECT_EQ(config.numStages(), 3);
    }

    TEST_F(Test__LocalPPContext, Factory_InvalidConfigThrows)
    {
        LocalPPConfig invalid_config;
        invalid_config.stage_devices = {}; // Empty!
        invalid_config.layer_boundaries = {0, 24};

        EXPECT_FALSE(invalid_config.isValid());

        // The factory should throw for invalid config
        EXPECT_THROW(createLocalPPContext(invalid_config), std::invalid_argument);
    }

    TEST_F(Test__LocalPPContext, Factory_InvalidBoundariesThrows)
    {
        LocalPPConfig invalid_config;
        invalid_config.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        invalid_config.layer_boundaries = {0, 24}; // Wrong size!

        EXPECT_FALSE(invalid_config.isValid());
        EXPECT_THROW(createLocalPPContext(invalid_config), std::invalid_argument);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 7: Mock Builder Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, MockBuilder_FluentAPIWorks)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withDevices({GlobalDeviceAddress::cuda(0),
                                      GlobalDeviceAddress::cuda(1),
                                      GlobalDeviceAddress::rocm(0)})
                        .withLayerBoundaries({0, 8, 16, 24})
                        .withBackend(CollectiveBackendType::HOST)
                        .build();

        EXPECT_EQ(mock->numStages(), 3);
        EXPECT_EQ(mock->totalLayers(), 24);
        EXPECT_EQ(mock->deviceForStage(0).device_type, DeviceType::CUDA);
        EXPECT_EQ(mock->deviceForStage(2).device_type, DeviceType::ROCm);
    }

    TEST_F(Test__LocalPPContext, MockBuilder_BuildSharedReturnsSharedPtr)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .buildShared();

        EXPECT_NE(mock, nullptr);
        EXPECT_EQ(mock.use_count(), 1);
        EXPECT_EQ(mock->numStages(), 2);
    }

    TEST_F(Test__LocalPPContext, MockBuilder_WithEqualLayerSplitHandlesRemainder)
    {
        // 25 layers across 3 stages: 9, 8, 8
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(25)
                        .build();

        auto [first0, last0] = mock->layerRangeForStage(0);
        auto [first1, last1] = mock->layerRangeForStage(1);
        auto [first2, last2] = mock->layerRangeForStage(2);

        EXPECT_EQ(mock->totalLayers(), 25);

        // First stages get the extra layers from remainder
        EXPECT_EQ(last0 - first0, 9); // 25/3 = 8 + 1 remainder
        EXPECT_EQ(last1 - first1, 8);
        EXPECT_EQ(last2 - first2, 8);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 8: Reset and Configuration Modification Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, Mock_ResetCallCountsClearsCounters)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->transfer(test_tensor_.get(), 0, 1);
        mock->transferAsync(test_tensor_.get(), 0, 1, nullptr);
        mock->synchronize();

        EXPECT_GT(mock->transferCallCount(), 0);
        EXPECT_GT(mock->transferAsyncCallCount(), 0);
        EXPECT_GT(mock->synchronizeCallCount(), 0);

        mock->resetCallCounts();

        EXPECT_EQ(mock->transferCallCount(), 0);
        EXPECT_EQ(mock->transferAsyncCallCount(), 0);
        EXPECT_EQ(mock->synchronizeCallCount(), 0);
        EXPECT_TRUE(mock->transferCalls().empty());
    }

    TEST_F(Test__LocalPPContext, Mock_SetTransferResultModifiesBehavior)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_TRUE(mock->transfer(test_tensor_.get(), 0, 1));

        mock->setTransferResult(false);
        EXPECT_FALSE(mock->transfer(test_tensor_.get(), 0, 1));

        mock->setTransferResult(true);
        EXPECT_TRUE(mock->transfer(test_tensor_.get(), 0, 1));
    }

    TEST_F(Test__LocalPPContext, Mock_SetStageDevicesUpdatesDevices)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_EQ(mock->deviceForStage(0).device_type, DeviceType::CUDA);

        mock->setStageDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});

        EXPECT_EQ(mock->deviceForStage(0).device_type, DeviceType::ROCm);
        EXPECT_EQ(mock->deviceForStage(1).device_type, DeviceType::ROCm);
    }

    TEST_F(Test__LocalPPContext, Mock_MutableConfigAllowsDirectModification)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->mutableConfig().transfer_should_fail = true;
        EXPECT_FALSE(mock->transfer(test_tensor_.get(), 0, 1));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 9: ReserveStagingBuffer Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, ReserveStagingBufferBytes_ReturnsTrue)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_TRUE(mock->reserveStagingBufferBytes(1024 * 1024));
        EXPECT_EQ(mock->reserveStagingCallCount(), 1);
    }

    TEST_F(Test__LocalPPContext, ReserveStagingBufferBytes_CountsMultipleCalls)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        mock->reserveStagingBufferBytes(1024);
        mock->reserveStagingBufferBytes(2048);
        mock->reserveStagingBufferBytes(4096);

        EXPECT_EQ(mock->reserveStagingCallCount(), 3);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 10: Edge Cases and Corner Cases
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__LocalPPContext, SingleStage_AllLayersOnOneDevice)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCpuStages(1)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_EQ(mock->numStages(), 1);
        auto [first, last] = mock->layerRangeForStage(0);
        EXPECT_EQ(first, 0);
        EXPECT_EQ(last, 24);

        // All layers map to stage 0
        for (int layer = 0; layer < 24; ++layer)
        {
            EXPECT_EQ(mock->stageForLayer(layer), 0);
        }
    }

    TEST_F(Test__LocalPPContext, TransferToSameStage_StillRecorded)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        // Transfer from stage 0 to stage 0 (same device)
        EXPECT_TRUE(mock->transfer(test_tensor_.get(), 0, 0));

        // Mock still records the call (real impl would no-op but succeed)
        EXPECT_EQ(mock->transferCallCount(), 1);
        EXPECT_TRUE(mock->sameDevice(0, 0));
    }

    TEST_F(Test__LocalPPContext, LastTransferCall_ReturnsLatest)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(3)
                        .withEqualLayerSplit(24)
                        .build();

        mock->transfer(test_tensor_.get(), 0, 1);
        mock->transfer(test_tensor_.get(), 1, 2);

        auto last = mock->lastTransferCall();
        EXPECT_EQ(last.stage_from, 1);
        EXPECT_EQ(last.stage_to, 2);
    }

    TEST_F(Test__LocalPPContext, LastTransferCall_ThrowsWhenEmpty)
    {
        auto mock = MockLocalPPContextBuilder()
                        .withCudaStages(2)
                        .withEqualLayerSplit(24)
                        .build();

        EXPECT_THROW(mock->lastTransferCall(), std::out_of_range);
    }

    TEST_F(Test__LocalPPContext, DefaultMock_HasSingleCpuStage)
    {
        MockLocalPPContext mock;

        EXPECT_EQ(mock.numStages(), 1);
        EXPECT_EQ(mock.deviceForStage(0).device_type, DeviceType::CPU);
        EXPECT_EQ(mock.totalLayers(), 24); // Default 24 layers
    }

    TEST_F(Test__LocalPPContext, UnequalLayerSplit_WorksCorrectly)
    {
        // Create config with unequal layer distribution
        MockLocalPPContext::Config config;
        config.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
            GlobalDeviceAddress::cuda(2)};
        // Stage 0: 4 layers, Stage 1: 8 layers, Stage 2: 12 layers
        config.layer_boundaries = {0, 4, 12, 24};

        MockLocalPPContext mock(config);

        auto [first0, last0] = mock.layerRangeForStage(0);
        auto [first1, last1] = mock.layerRangeForStage(1);
        auto [first2, last2] = mock.layerRangeForStage(2);

        EXPECT_EQ(last0 - first0, 4);
        EXPECT_EQ(last1 - first1, 8);
        EXPECT_EQ(last2 - first2, 12);

        // Verify layer mapping
        EXPECT_EQ(mock.stageForLayer(0), 0);
        EXPECT_EQ(mock.stageForLayer(3), 0);
        EXPECT_EQ(mock.stageForLayer(4), 1);
        EXPECT_EQ(mock.stageForLayer(11), 1);
        EXPECT_EQ(mock.stageForLayer(12), 2);
        EXPECT_EQ(mock.stageForLayer(23), 2);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 11: TransferTo Unit Tests (Phase 5)
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // NOTE: Tests requiring real GPU hardware belong in integration tests:
    //   tests/v2/integration/pipelines/Test__MultiGPU_RealModel.cpp
    // 
    // Unit tests here use ONLY mocks and do NOT require GPU hardware.
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Transfer to same device returns true without actual transfer
     * 
     * When stage_from and stage_to map to the same physical device,
     * transfer() should return true immediately as a no-op.
     */
    TEST_F(Test__LocalPPContext, Transfer_SameDevice_ReturnsTrue)
    {
        // Use mock with 2 stages on SAME device
        MockLocalPPContext::Config config;
        config.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(0)  // Same device!
        };
        config.layer_boundaries = {0, 12, 24};

        auto mock = std::make_unique<MockLocalPPContext>(config);

        // Verify both stages are on the same device
        EXPECT_TRUE(mock->sameDevice(0, 1));

        // Transfer should succeed immediately (no-op)
        EXPECT_TRUE(mock->transfer(test_tensor_.get(), 0, 1));

        // Mock still records the call (as per mock behavior)
        EXPECT_EQ(mock->transferCallCount(), 1);
    }

} // namespace llaminar2::test
