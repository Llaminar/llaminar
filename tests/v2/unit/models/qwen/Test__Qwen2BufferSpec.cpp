/**
 * @file Test__Qwen2BufferSpec.cpp
 * @brief Unit tests for Qwen2BufferSpec buffer allocation specification
 *
 * Tests the Qwen2BufferSpec utility class that determines buffer allocation
 * strategies based on tensor parallelism and collective backend configuration.
 *
 * Test Coverage:
 * 1. requiresBARBacked() always returns false (BAR-backed removed)
 * 2. getAllocationStrategy() always returns STANDARD
 * 3. getRowParallelOutputSuffixes() returns expected set
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
        // Test: requiresBARBacked - Always returns false (BAR removed)
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, RequiresBARBacked_AlwaysFalse)
        {
            // BAR-backed allocation has been removed. requiresBARBacked() always returns false.

            // Row-parallel outputs that previously required BAR-backed allocation
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_allreduce", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_output", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "attention_wo_output", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "attn_wo_output", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "wo_output", CollectiveBackendType::HOST, TP_ENABLED));

            // Non-row-parallel buffers
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "residual", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "Q", CollectiveBackendType::HOST, TP_ENABLED));

            // With TP disabled
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::HOST, TP_DISABLED));

            // With various backends
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::NCCL, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::RCCL, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::AUTO, TP_ENABLED));

            // Empty name, zero TP
            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_FALSE(Qwen2BufferSpec::requiresBARBacked(
                "ffn_down_output", CollectiveBackendType::HOST, 0));
        }

        // =========================================================================
        // Test: getAllocationStrategy - Always returns STANDARD
        // =========================================================================

        TEST_F(Test__Qwen2BufferSpec, GetAllocationStrategy_AlwaysStandard)
        {
            // BAR_BACKED strategy has been removed; all buffers use STANDARD

            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "ffn_down_output", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "attn_wo_output", CollectiveBackendType::HOST, TP_ENABLED));

            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "ffn_down_output", CollectiveBackendType::NCCL, TP_ENABLED));

            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "ffn_down_output", CollectiveBackendType::HOST, TP_DISABLED));

            EXPECT_EQ(
                AllocationStrategy::STANDARD,
                Qwen2BufferSpec::getAllocationStrategy(
                    "residual", CollectiveBackendType::HOST, TP_ENABLED));
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

    } // namespace test
} // namespace llaminar2
