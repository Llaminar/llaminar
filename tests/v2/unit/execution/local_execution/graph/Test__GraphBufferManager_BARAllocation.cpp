/**
 * @file Test__GraphBufferManager_BARAllocation.cpp
 * @brief Unit tests for GraphBufferManager BAR-backed allocation
 *
 * Tests the integration between GraphBufferManager and TensorFactory for
 * BAR-backed tensor allocation when using PCIeBAR backend with heterogeneous
 * CUDA/ROCm GPUs.
 *
 * Test Coverage:
 * 1. Default config has no BAR allocation enabled
 * 2. Config with PCIeBAR + TP + devices enables BAR allocation
 * 3. Row-parallel outputs get BAR-backed allocation when configured
 * 4. Non-row-parallel outputs get standard allocation
 * 5. Falls back to standard allocation when factory cannot create BAR-backed
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/GraphBufferManager.h"
#include "tensors/TensorFactory.h"
#include "models/qwen/Qwen2BufferSpec.h"
#include "config/OrchestrationConfig.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"

namespace llaminar2
{
    namespace test
    {

        // =========================================================================
        // Test Fixture
        // =========================================================================

        class Test__GraphBufferManager_BARAllocation : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Create MPIContext for TensorFactory (rank=0, world_size=1)
                mpi_ctx_ = std::make_unique<MPIContext>(0, 1);

                // Create TensorFactory for CPU-only tests
                factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
            }

            void TearDown() override
            {
                factory_.reset();
                mpi_ctx_.reset();
            }

            std::unique_ptr<MPIContext> mpi_ctx_;
            std::unique_ptr<TensorFactory> factory_;

            // Helper to create config for BAR allocation testing
            GraphBufferManagerConfig createPCIeBarConfig()
            {
                GraphBufferManagerConfig config;
                config.tp_degree = 2;
                config.collective_backend = CollectiveBackendType::PCIE_BAR;
                config.rocm_device = DeviceId::rocm(0);
                config.cuda_device = DeviceId::cuda(0);
                return config;
            }

            // Helper to create a buffer descriptor
            BufferDescriptor createBufferDesc(const std::string &name,
                                              BufferRole role = BufferRole::OUTPUT)
            {
                BufferDescriptor desc;
                desc.name = name;
                desc.role = role;
                desc.shape = {32, 64};
                desc.tensor_type = BufferTensorType::FP32;
                desc.device = DeviceId::cpu();
                return desc;
            }
        };

        // =========================================================================
        // Test: Default config has standard allocation
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, DefaultConfig_StandardAllocation)
        {
            // Default config should not trigger BAR allocation
            GraphBufferManagerConfig config; // All defaults

            // Verify default config doesn't enable BAR allocation
            EXPECT_EQ(1, config.tp_degree);
            EXPECT_EQ(CollectiveBackendType::AUTO, config.collective_backend);
            EXPECT_FALSE(config.rocm_device.is_rocm());
            EXPECT_FALSE(config.cuda_device.is_cuda());

            // With default config, even row-parallel buffers shouldn't need BAR
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", config.collective_backend, config.tp_degree));
        }

        // =========================================================================
        // Test: PCIeBAR config enables BAR for row-parallel outputs
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, PCIeBarConfig_EnablesBAR_ForRowParallel)
        {
            auto config = createPCIeBarConfig();

            // Verify config enables BAR allocation checks
            EXPECT_EQ(2, config.tp_degree);
            EXPECT_EQ(CollectiveBackendType::PCIE_BAR, config.collective_backend);
            EXPECT_TRUE(config.rocm_device.is_rocm());
            EXPECT_TRUE(config.cuda_device.is_cuda());

            // Row-parallel outputs should need BAR with this config
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", config.collective_backend, config.tp_degree));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "attn_wo_output", config.collective_backend, config.tp_degree));
        }

        // =========================================================================
        // Test: Non-row-parallel outputs don't get BAR allocation
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, NonRowParallel_StandardAllocation)
        {
            auto config = createPCIeBarConfig();

            // Non-row-parallel outputs should NOT need BAR even with PCIeBAR config
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "residual", config.collective_backend, config.tp_degree));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "Q", config.collective_backend, config.tp_degree));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "K", config.collective_backend, config.tp_degree));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "logits", config.collective_backend, config.tp_degree));
        }

        // =========================================================================
        // Test: GraphBufferManager allocates standard tensors without BAR
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, AllocateStandardTensor_NoBAR)
        {
            // Create manager with default config (no BAR)
            GraphBufferManagerConfig config;
            GraphBufferManager manager(factory_.get(), nullptr, config);

            // Allocate a buffer
            auto desc = createBufferDesc("residual");
            EXPECT_TRUE(manager.allocateBuffer("layer0", desc));

            // Buffer should exist
            EXPECT_TRUE(manager.hasBuffer("layer0", "residual"));

            auto *tensor = manager.getBuffer("layer0", "residual");
            ASSERT_NE(nullptr, tensor);

            // Should be regular allocation (not BAR-backed)
            // Note: We can't directly check isBARBacked() without a BAR-backed tensor,
            // but we verify the tensor exists and has correct shape
            EXPECT_EQ(32 * 64, tensor->numel());
        }

        // =========================================================================
        // Test: GraphBufferManager falls back gracefully when BAR unavailable
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, FallbackWhenBARUnavailable)
        {
            // Create manager with PCIeBAR config but factory has no P2P set
            // This simulates the case where BAR is requested but not available
            auto config = createPCIeBarConfig();
            GraphBufferManager manager(factory_.get(), nullptr, config);

            // Factory should NOT be able to create BAR-backed tensors without P2P
            EXPECT_FALSE(factory_->canCreateBARBacked());

            // Allocate a row-parallel buffer (would normally want BAR)
            auto desc = createBufferDesc("ffn_down_output");
            EXPECT_TRUE(manager.allocateBuffer("layer0", desc));

            // Buffer should still be allocated (fallback to standard)
            EXPECT_TRUE(manager.hasBuffer("layer0", "ffn_down_output"));

            auto *tensor = manager.getBuffer("layer0", "ffn_down_output");
            ASSERT_NE(nullptr, tensor);
            EXPECT_EQ(32 * 64, tensor->numel());
        }

        // =========================================================================
        // Test: AllocationStrategy enum values
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, AllocationStrategyEnum_Values)
        {
            // Verify enum values exist and are distinct
            AllocationStrategy standard = AllocationStrategy::STANDARD;
            AllocationStrategy bar_backed = AllocationStrategy::BAR_BACKED;
            AllocationStrategy pinned = AllocationStrategy::PINNED_HOST;

            EXPECT_NE(standard, bar_backed);
            EXPECT_NE(standard, pinned);
            EXPECT_NE(bar_backed, pinned);
        }

        // =========================================================================
        // Test: getAllocationStrategy returns correct values
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, GetAllocationStrategy_Correct)
        {
            auto config = createPCIeBarConfig();

            // Row-parallel with PCIeBAR -> BAR_BACKED
            EXPECT_EQ(AllocationStrategy::BAR_BACKED,
                      Qwen2BufferSpec::getAllocationStrategy(
                          "ffn_down_output", config.collective_backend, config.tp_degree));

            // Non-row-parallel -> STANDARD
            EXPECT_EQ(AllocationStrategy::STANDARD,
                      Qwen2BufferSpec::getAllocationStrategy(
                          "residual", config.collective_backend, config.tp_degree));

            // Row-parallel with NCCL -> STANDARD (not PCIeBAR)
            EXPECT_EQ(AllocationStrategy::STANDARD,
                      Qwen2BufferSpec::getAllocationStrategy(
                          "ffn_down_output", CollectiveBackendType::NCCL, config.tp_degree));
        }

        // =========================================================================
        // Test: Layer-prefixed buffer names work correctly
        // =========================================================================

        TEST_F(Test__GraphBufferManager_BARAllocation, LayerPrefixed_BufferNames)
        {
            auto config = createPCIeBarConfig();

            // Layer-prefixed row-parallel outputs should need BAR
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer0_ffn_down_output", config.collective_backend, config.tp_degree));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer15_attn_wo_allreduce", config.collective_backend, config.tp_degree));

            // Layer-prefixed non-row-parallel should NOT need BAR
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "layer0_Q", config.collective_backend, config.tp_degree));
        }

    } // namespace test
} // namespace llaminar2
