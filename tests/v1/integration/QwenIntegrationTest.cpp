/**
 * @file QwenIntegrationTest.cpp
 * @brief Comprehensive integration tests for Qwen models
 * @author David Sanftenberg
 *
 * This test suite verifies layer-by-layer correctness for Qwen models across:
 * - Different precisions (FP32, Q6_K, Q4_0)
 * - Prefill and decode stages
 * - Various prompt lengths
 * - Edge cases (single token, long context, etc.)
 */

#include "ModelIntegrationTestBase.h"
#include "Logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <iostream>

namespace llaminar
{
    namespace testing
    {

        // ============================================================================
        // Qwen 2.5 0.5B Instruction Model Tests
        // ============================================================================

        /**
         * @brief Standard test tokens: "1+1=" (common math prompt)
         */
        const std::vector<int> QWEN_STANDARD_TOKENS = {1639, 266, 285, 17, 10, 17, 30};

        /**
         * @brief Short prompt for basic testing
         */
        const std::vector<int> QWEN_SHORT_TOKENS = {1639, 266, 285};

        /**
         * @brief Single token for minimal decode testing
         */
        const std::vector<int> QWEN_SINGLE_TOKEN = {1639};

        // ----------------------------------------------------------------------------
        // Prefill Stage Tests - Q4_0 Precision
        // ----------------------------------------------------------------------------

        TEST_F(ModelIntegrationTestBase, QwenPrefillQ4_0_StandardPrompt)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q4_0";
            config.stage = "prefill";
            config.tokens = QWEN_STANDARD_TOKENS;
            config.tolerances = ToleranceConfig::for_precision("q4_0");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen Q4_0 prefill failed layer-by-layer comparison";
        }

        TEST_F(ModelIntegrationTestBase, QwenPrefillQ4_0_ShortPrompt)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q4_0";
            config.stage = "prefill";
            config.tokens = QWEN_SHORT_TOKENS;
            config.tolerances = ToleranceConfig::for_precision("q4_0");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen Q4_0 short prefill failed";
        }

        TEST_F(ModelIntegrationTestBase, QwenPrefillQ4_0_SingleToken)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q4_0";
            config.stage = "prefill";
            config.tokens = QWEN_SINGLE_TOKEN;
            config.tolerances = ToleranceConfig::for_precision("q4_0");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen Q4_0 single token prefill failed";
        }

        // ----------------------------------------------------------------------------
        // Decode Stage Tests - Q4_0 Precision
        // ----------------------------------------------------------------------------

        TEST_F(ModelIntegrationTestBase, QwenDecodeQ4_0_AfterStandardPrefill)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q4_0";
            config.stage = "decode";
            config.tokens = QWEN_STANDARD_TOKENS; // Last token will be decoded
            config.tolerances = ToleranceConfig::for_precision("q4_0");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen Q4_0 decode failed layer-by-layer comparison";
        }

        TEST_F(ModelIntegrationTestBase, QwenDecodeQ4_0_AfterShortPrefill)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q4_0";
            config.stage = "decode";
            config.tokens = QWEN_SHORT_TOKENS;
            config.tolerances = ToleranceConfig::for_precision("q4_0");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen Q4_0 decode after short prefill failed";
        }

        // ----------------------------------------------------------------------------
        // Prefill Stage Tests - Q6_K Precision (Higher Quality)
        // ----------------------------------------------------------------------------

        TEST_F(ModelIntegrationTestBase, QwenPrefillQ6_K_StandardPrompt)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q6_k";
            config.stage = "prefill";
            config.tokens = QWEN_STANDARD_TOKENS;
            config.tolerances = ToleranceConfig::for_precision("q6_k");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen Q6_K prefill failed layer-by-layer comparison";
        }

        // ----------------------------------------------------------------------------
        // Decode Stage Tests - Q6_K Precision
        // ----------------------------------------------------------------------------

        TEST_F(ModelIntegrationTestBase, QwenDecodeQ6_K_AfterStandardPrefill)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q6_k";
            config.stage = "decode";
            config.tokens = QWEN_STANDARD_TOKENS;
            config.tolerances = ToleranceConfig::for_precision("q6_k");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen Q6_K decode failed layer-by-layer comparison";
        }

        // ----------------------------------------------------------------------------
        // FP32 Precision Tests (Gold Standard)
        // ----------------------------------------------------------------------------

        // Note: FP32 tests require FP32 model weights, which may not be available
        // These tests will auto-skip if golden references are missing

        TEST_F(ModelIntegrationTestBase, QwenPrefillFP32_StandardPrompt)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "fp32";
            config.stage = "prefill";
            config.tokens = QWEN_STANDARD_TOKENS;
            config.tolerances = ToleranceConfig::for_precision("fp32");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen FP32 prefill failed layer-by-layer comparison";
        }

        TEST_F(ModelIntegrationTestBase, QwenDecodeFP32_AfterStandardPrefill)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "fp32";
            config.stage = "decode";
            config.tokens = QWEN_STANDARD_TOKENS;
            config.tolerances = ToleranceConfig::for_precision("fp32");

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen FP32 decode failed layer-by-layer comparison";
        }

        // ============================================================================
        // Parameterized Tests for Comprehensive Coverage
        // ============================================================================

        /**
         * @brief Generate all test configurations for Qwen models
         */
        std::vector<TestConfig> generate_qwen_test_configs()
        {
            std::vector<TestConfig> configs;

            // Precisions to test
            std::vector<std::string> precisions = {"q4_0", "q6_k"};
            // Stages to test
            std::vector<std::string> stages = {"prefill", "decode"};
            // Token sets to test
            std::vector<std::vector<int>> token_sets = {
                QWEN_STANDARD_TOKENS,
                QWEN_SHORT_TOKENS,
                QWEN_SINGLE_TOKEN};

            for (const auto &precision : precisions)
            {
                for (const auto &stage : stages)
                {
                    for (const auto &tokens : token_sets)
                    {
                        TestConfig config;
                        config.model_name = "qwen2.5-0.5b-instruct";
                        config.precision = precision;
                        config.stage = stage;
                        config.tokens = tokens;
                        config.tolerances = ToleranceConfig::for_precision(precision);
                        configs.push_back(config);
                    }
                }
            }

            return configs;
        }

        /**
         * @brief Parameterized test for all Qwen configurations
         */
        TEST_P(ParameterizedModelTest, QwenComprehensiveIntegration)
        {
            TestConfig config = GetParam();

            std::cout << "=== Parameterized Test ===" << std::endl;
            std::cout << "Precision: " << config.precision << std::endl;
            std::cout << "Stage: " << config.stage << std::endl;
            std::cout << "Tokens: " << config.tokens.size() << std::endl;

            bool success = run_integration_test(config);
            ASSERT_TRUE(success) << "Qwen " << config.precision << " " << config.stage
                                 << " with " << config.tokens.size() << " tokens failed";
        }

        // Instantiate parameterized tests
        INSTANTIATE_TEST_SUITE_P(
            QwenAllConfigurations,
            ParameterizedModelTest,
            ::testing::ValuesIn(generate_qwen_test_configs()),
            [](const ::testing::TestParamInfo<TestConfig> &info)
            {
                // Generate test name from config
                std::string name = "Qwen_" + info.param.precision + "_" +
                                   info.param.stage + "_" +
                                   std::to_string(info.param.tokens.size()) + "tokens";
                return name;
            });

        // ============================================================================
        // Edge Case Tests
        // ============================================================================

        /**
         * @brief Test with empty token sequence (should gracefully handle)
         */
        TEST_F(ModelIntegrationTestBase, QwenEdgeCase_EmptyTokens)
        {
            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q4_0";
            config.stage = "prefill";
            config.tokens = {}; // Empty
            config.tolerances = ToleranceConfig::for_precision("q4_0");

            // This should either skip or handle gracefully
            // Don't assert success, just ensure no crash
            run_integration_test(config);
        }

        /**
         * @brief Test with very long context (stress test)
         */
        TEST_F(ModelIntegrationTestBase, QwenEdgeCase_LongContext)
        {
            // Generate 128 tokens (might need golden reference generation)
            std::vector<int> long_tokens;
            for (int i = 0; i < 128; ++i)
            {
                long_tokens.push_back(1639 + (i % 100)); // Varied tokens
            }

            TestConfig config;
            config.model_name = "qwen2.5-0.5b-instruct";
            config.precision = "q4_0";
            config.stage = "prefill";
            config.tokens = long_tokens;
            config.tolerances = ToleranceConfig::for_precision("q4_0");

            // May skip if golden reference not available
            bool success = run_integration_test(config);
            if (success)
            {
                std::cout << "Long context test passed!" << std::endl;
            }
        }

    } // namespace testing
} // namespace llaminar

/**
 * @brief Main entry point for integration tests
 */
int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
