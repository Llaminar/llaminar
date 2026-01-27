/**
 * @file Test__Qwen2BufferSpec.cpp
 * @brief Unit tests for Qwen2BufferSpec buffer allocation specification
 *
 * Tests the Qwen2BufferSpec utility class that determines which buffers
 * need BAR-backed allocation based on tensor parallelism and collective
 * backend configuration.
 *
 * Test Coverage:
 * 1. requiresBARBacked() returns true for FFN down output with PCIeBAR + TP
 * 2. requiresBARBacked() returns true for attention Wo output with PCIeBAR + TP
 * 3. requiresBARBacked() returns false when tp_degree=1 (no TP)
 * 4. requiresBARBacked() returns false with NCCL backend
 * 5. requiresBARBacked() returns false for non-row-parallel buffers
 * 6. requiresBARBacked() works with layer-prefixed names
 * 7. getAllocationStrategy() returns correct enum values
 * 8. getRowParallelOutputSuffixes() returns expected set
 */

#include <gtest/gtest.h>
#include "models/qwen/Qwen2BufferSpec.h"
#include "config/OrchestrationConfig.h"

namespace llaminar2
{
    namespace test
    {

        // =========================================================================
        // Test Fixture
        // =========================================================================

        class Test__Qwen2BufferSpec : public ::testing::Test
        {
        protected:
            // Common test parameters
            static constexpr int TP_ENABLED = 2;  // TP degree > 1
            static constexpr int TP_DISABLED = 1; // No TP
        };

        // =========================================================================
        // Test: requiresBARBacked - FFN Down Output with PCIeBAR + TP
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, RequiresBARBacked_FFNDown_PCIeBAR)
        {
            // FFN down projection output should require BAR-backed allocation
            // when using PCIeBAR backend with LOCAL TP enabled

            // Test various FFN down buffer names
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_allreduce", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));
        }

        // =========================================================================
        // Test: requiresBARBacked - Attention Wo Output with PCIeBAR + TP
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, RequiresBARBacked_AttnWO_PCIeBAR)
        {
            // Attention Wo projection output should require BAR-backed allocation
            // when using PCIeBAR backend with LOCAL TP enabled

            // Test various attention Wo buffer names
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "attention_wo_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "attn_wo_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "attn_wo_allreduce", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "wo_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "attn_proj", CollectiveBackendType::PCIE_BAR, TP_ENABLED));
        }

        // =========================================================================
        // Test: requiresBARBacked - Returns false when TP disabled
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, RequiresBARBacked_False_NoTP)
        {
            // With tp_degree=1 (no TP), no buffers need BAR-backed allocation
            // even with PCIeBAR backend

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::PCIE_BAR, TP_DISABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "attn_wo_output", CollectiveBackendType::PCIE_BAR, TP_DISABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_output", CollectiveBackendType::PCIE_BAR, TP_DISABLED));

            // Also test with tp_degree=0 (invalid but should return false)
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::PCIE_BAR, 0));
        }

        // =========================================================================
        // Test: requiresBARBacked - Returns false with NCCL backend
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, RequiresBARBacked_False_NCCL)
        {
            // With NCCL backend (same-vendor GPUs), no BAR-backed allocation needed
            // NCCL handles allreduce efficiently without PCIe BAR

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::NCCL, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "attn_wo_output", CollectiveBackendType::NCCL, TP_ENABLED));

            // Also test other non-PCIeBAR backends
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::RCCL, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::MPI, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::AUTO, TP_ENABLED));
        }

        // =========================================================================
        // Test: requiresBARBacked - Returns false for non-row-parallel buffers
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, RequiresBARBacked_False_OtherBuffer)
        {
            // Buffers that are NOT row-parallel outputs should not need BAR-backed allocation
            // even with PCIeBAR + TP

            // Input/activation buffers
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "residual", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "normalized", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "current_hidden", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // Column-parallel projection outputs (don't need allreduce)
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "Q", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "K", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "V", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "gate", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "up", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // Model-level buffers
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "logits", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "embedding_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // Empty buffer name
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "", CollectiveBackendType::PCIE_BAR, TP_ENABLED));
        }

        // =========================================================================
        // Test: requiresBARBacked - Works with layer-prefixed names
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, RequiresBARBacked_LayerPrefixed)
        {
            // Layer-prefixed buffer names should also match via suffix matching

            // FFN down with layer prefix
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer0_ffn_down_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer15_ffn_down_allreduce", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer99_ffn_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // Attention Wo with layer prefix
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer0_attn_wo_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer7_wo_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "layer12_attn_proj", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // Non-row-parallel with layer prefix should NOT match
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "layer0_Q", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "layer5_gate", CollectiveBackendType::PCIE_BAR, TP_ENABLED));
        }

        // =========================================================================
        // Test: getAllocationStrategy - Returns correct enum values
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, GetAllocationStrategy_ReturnsCorrectValues)
        {
            // BAR-backed case
            EXPECT_EQ(
                AllocationStrategy::BAR_BACKED,
                Qwen2BufferSpec::getAllocationStrategy(
                    "ffn_down_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            EXPECT_EQ(
                AllocationStrategy::BAR_BACKED,
                Qwen2BufferSpec::getAllocationStrategy(
                    "attn_wo_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // STANDARD cases
            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "ffn_down_output", CollectiveBackendType::NCCL, TP_ENABLED));

            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "ffn_down_output", CollectiveBackendType::PCIE_BAR, TP_DISABLED));

            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "residual", CollectiveBackendType::PCIE_BAR, TP_ENABLED));
        }

        // =========================================================================
        // Test: getRowParallelOutputSuffixes - Returns expected set
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, GetRowParallelOutputSuffixes_ContainsExpected)
        {
            const auto &suffixes = Qwen2BufferSpec::getRowParallelOutputSuffixes();

            // Should contain FFN down variants
            EXPECT_TRUE(suffixes.count("ffn_down_output") > 0);
            EXPECT_TRUE(suffixes.count("ffn_down_allreduce") > 0);
            EXPECT_TRUE(suffixes.count("ffn_output") > 0);

            // Should contain attention Wo variants
            EXPECT_TRUE(suffixes.count("attention_wo_output") > 0);
            EXPECT_TRUE(suffixes.count("attn_wo_output") > 0);
            EXPECT_TRUE(suffixes.count("attn_wo_allreduce") > 0);
            EXPECT_TRUE(suffixes.count("wo_output") > 0);
            EXPECT_TRUE(suffixes.count("attn_proj") > 0);

            // Should NOT contain non-row-parallel buffers
            EXPECT_TRUE(suffixes.count("residual") == 0);
            EXPECT_TRUE(suffixes.count("Q") == 0);
            EXPECT_TRUE(suffixes.count("gate") == 0);
        }

        // =========================================================================
        // Test: Edge cases and boundary conditions
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, EdgeCases)
        {
            // Very high TP degree should still work
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::PCIE_BAR, 8));

            // Partial suffix match should NOT work (security)
            // "my_ffn_output" should not match "ffn_output" due to separator check
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "myffn_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // But proper separator should work
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "my_ffn_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));

            // Dot separator should also work
            EXPECT_TRUE(Qwen2BufferSpec::requiresBARBacked(
                "stage.ffn_output", CollectiveBackendType::PCIE_BAR, TP_ENABLED));
        }

    } // namespace test
} // namespace llaminar2
