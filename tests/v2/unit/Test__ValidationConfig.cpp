/**
 * @file Test__ValidationConfig.cpp
 * @brief Unit tests for ValidationConfig and buffer validation in GraphExecutor
 * @author GitHub Copilot
 * @date December 2025
 *
 * Tests the Buffer Contract Validation System - Phase 5:
 * - ValidationConfig environment variable parsing
 * - Buffer validation integration with GraphExecutor
 *
 * @see DebugEnv.h for ValidationConfig struct
 * @see GraphExecutor.cpp for validateStageOutputs implementation
 */

#include <gtest/gtest.h>
#include "utils/DebugEnv.h"
#include "utils/Assertions.h" // For LLAMINAR_ASSERTIONS_ACTIVE
#include "execution/GraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/TensorValidation.h"
#include "tensors/TensorVerification.h" // For VerificationFailure exception
#include <cmath>

namespace llaminar2::test
{

    // =============================================================================
    // ValidationConfig Tests
    // =============================================================================

    class ValidationConfigTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Save original env vars
            saved_validate_ = getEnvSafe("LLAMINAR_VALIDATE_BUFFERS");
            saved_fail_zero_ = getEnvSafe("LLAMINAR_FAIL_ON_ZERO");
            saved_fail_nan_ = getEnvSafe("LLAMINAR_FAIL_ON_NAN");

            // Clear env vars for clean test state
            clearEnvVars();
        }

        void TearDown() override
        {
            // Restore original env vars
            restoreEnv("LLAMINAR_VALIDATE_BUFFERS", saved_validate_);
            restoreEnv("LLAMINAR_FAIL_ON_ZERO", saved_fail_zero_);
            restoreEnv("LLAMINAR_FAIL_ON_NAN", saved_fail_nan_);

            // Reload debug env to restore original state
            mutableDebugEnv().reload();
        }

        static std::string getEnvSafe(const char *name)
        {
            const char *val = std::getenv(name);
            return val ? val : "";
        }

        static void clearEnvVars()
        {
            unsetenv("LLAMINAR_VALIDATE_BUFFERS");
            unsetenv("LLAMINAR_FAIL_ON_ZERO");
            unsetenv("LLAMINAR_FAIL_ON_NAN");
        }

        static void restoreEnv(const char *name, const std::string &value)
        {
            if (value.empty())
            {
                unsetenv(name);
            }
            else
            {
                setenv(name, value.c_str(), 1);
            }
        }

    private:
        std::string saved_validate_;
        std::string saved_fail_zero_;
        std::string saved_fail_nan_;
    };

    TEST_F(ValidationConfigTest, DefaultConfig_MatchesBuildType)
    {
        // Reload with cleared env vars
        ValidationConfig config;

        // In Debug/Integration builds (LLAMINAR_ASSERTIONS_ACTIVE=1):
        //   - validate_buffers defaults to true
        //   - fail_on_nan defaults to true
        // In Release builds:
        //   - All defaults are false
#if LLAMINAR_ASSERTIONS_ACTIVE
        EXPECT_TRUE(config.validate_buffers) << "Should be auto-enabled in Debug builds";
        EXPECT_FALSE(config.fail_on_zero); // Still false by default
        EXPECT_TRUE(config.fail_on_nan) << "Should be auto-enabled in Debug builds";
#else
        EXPECT_FALSE(config.validate_buffers);
        EXPECT_FALSE(config.fail_on_zero);
        EXPECT_FALSE(config.fail_on_nan);
#endif
    }

    TEST_F(ValidationConfigTest, ValidateBuffers_EnabledViaEnv)
    {
        // Explicitly disable Debug defaults, then enable only validate_buffers
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "0", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "0", 1);

        ValidationConfig config;

        EXPECT_TRUE(config.validate_buffers);
        EXPECT_FALSE(config.fail_on_zero);
        EXPECT_FALSE(config.fail_on_nan);
    }

    TEST_F(ValidationConfigTest, FailOnZero_EnabledViaEnv)
    {
        // Explicitly disable Debug defaults, then enable only fail_on_zero
        setenv("LLAMINAR_VALIDATE_BUFFERS", "0", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "1", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "0", 1);

        ValidationConfig config;

        EXPECT_FALSE(config.validate_buffers);
        EXPECT_TRUE(config.fail_on_zero);
        EXPECT_FALSE(config.fail_on_nan);
    }

    TEST_F(ValidationConfigTest, FailOnNaN_EnabledViaEnv)
    {
        // Explicitly disable Debug defaults, then enable only fail_on_nan
        setenv("LLAMINAR_VALIDATE_BUFFERS", "0", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "0", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "1", 1);

        ValidationConfig config;

        EXPECT_FALSE(config.validate_buffers);
        EXPECT_FALSE(config.fail_on_zero);
        EXPECT_TRUE(config.fail_on_nan);
    }

    TEST_F(ValidationConfigTest, AllFlags_EnabledTogether)
    {
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "1", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "1", 1);

        ValidationConfig config;

        EXPECT_TRUE(config.validate_buffers);
        EXPECT_TRUE(config.fail_on_zero);
        EXPECT_TRUE(config.fail_on_nan);
    }

    TEST_F(ValidationConfigTest, Reload_UpdatesFromEnv)
    {
        // Start with all disabled via env vars
        setenv("LLAMINAR_VALIDATE_BUFFERS", "0", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "0", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "0", 1);

        ValidationConfig config;
        EXPECT_FALSE(config.validate_buffers);

        // Set env var after construction
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        config.reload();

        EXPECT_TRUE(config.validate_buffers);
    }

    TEST_F(ValidationConfigTest, DebugEnv_IncludesValidation)
    {
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        mutableDebugEnv().reload();

        EXPECT_TRUE(debugEnv().validation.validate_buffers);
    }

    // =============================================================================
    // Mock Stage for Validation Testing
    // =============================================================================

    class MockValidationStage : public IComputeStage
    {
    public:
        explicit MockValidationStage(float *data, size_t rows, size_t cols)
            : data_(data), rows_(rows), cols_(cols) {}

        bool execute(IDeviceContext * /*ctx*/) override
        {
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        bool supportsBackend(ComputeBackendType /*backend*/) const override { return true; }

        StageDumpInfo getDumpInfo() const override
        {
            StageDumpInfo info;
            info.addOutput("output", data_, rows_, cols_);
            return info;
        }

    private:
        float *data_;
        size_t rows_;
        size_t cols_;
    };

    // =============================================================================
    // GraphExecutor Validation Integration Tests
    // =============================================================================

#ifndef NDEBUG
    class GraphExecutorValidationTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Save and enable validation
            saved_validate_ = std::getenv("LLAMINAR_VALIDATE_BUFFERS");
            saved_fail_zero_ = std::getenv("LLAMINAR_FAIL_ON_ZERO");
            saved_fail_nan_ = std::getenv("LLAMINAR_FAIL_ON_NAN");
        }

        void TearDown() override
        {
            // Restore env vars
            if (saved_validate_)
                setenv("LLAMINAR_VALIDATE_BUFFERS", saved_validate_, 1);
            else
                unsetenv("LLAMINAR_VALIDATE_BUFFERS");

            if (saved_fail_zero_)
                setenv("LLAMINAR_FAIL_ON_ZERO", saved_fail_zero_, 1);
            else
                unsetenv("LLAMINAR_FAIL_ON_ZERO");

            if (saved_fail_nan_)
                setenv("LLAMINAR_FAIL_ON_NAN", saved_fail_nan_, 1);
            else
                unsetenv("LLAMINAR_FAIL_ON_NAN");

            mutableDebugEnv().reload();
        }

        const char *saved_validate_ = nullptr;
        const char *saved_fail_zero_ = nullptr;
        const char *saved_fail_nan_ = nullptr;
    };

    TEST_F(GraphExecutorValidationTest, ValidOutput_PassesValidation)
    {
        // Enable validation without fail flags (just warnings)
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "0", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "0", 1);
        mutableDebugEnv().reload();

        // Create a valid (non-zero) tensor
        std::vector<float> data(16);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = static_cast<float>(i + 1); // Non-zero values
        }

        // Create graph with mock stage
        ComputeGraph graph;
        graph.addNode("test_stage",
                      std::make_unique<MockValidationStage>(data.data(), 4, 4),
                      0);

        // Execute
        GraphExecutorConfig config;
        GraphExecutor executor(config);
        CPUDeviceContext ctx(0);

        bool success = executor.execute(graph, &ctx);

        EXPECT_TRUE(success);
    }

    TEST_F(GraphExecutorValidationTest, ZeroOutput_WarnsButPasses_WhenFailDisabled)
    {
        // Enable validation but disable fail_on_zero
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "0", 1);
        mutableDebugEnv().reload();

        // Create a zero tensor
        std::vector<float> data(16, 0.0f);

        // Create graph with mock stage
        ComputeGraph graph;
        graph.addNode("test_stage",
                      std::make_unique<MockValidationStage>(data.data(), 4, 4),
                      0);

        // Execute - should succeed (warning only)
        GraphExecutorConfig config;
        GraphExecutor executor(config);
        CPUDeviceContext ctx(0);

        bool success = executor.execute(graph, &ctx);

        EXPECT_TRUE(success); // Warning issued but no failure
    }

    TEST_F(GraphExecutorValidationTest, ZeroOutput_Fails_WhenFailEnabled)
    {
        // Enable validation with fail_on_zero
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "1", 1);
        mutableDebugEnv().reload();

        // Create a zero tensor
        std::vector<float> data(16, 0.0f);

        // Create graph with mock stage
        ComputeGraph graph;
        graph.addNode("test_stage",
                      std::make_unique<MockValidationStage>(data.data(), 4, 4),
                      0);

        // Execute - should throw VerificationFailure exception (fail-fast behavior)
        GraphExecutorConfig config;
        GraphExecutor executor(config);
        CPUDeviceContext ctx(0);

        EXPECT_THROW({
            executor.execute(graph, &ctx);
        }, verification::VerificationFailure);
    }

    TEST_F(GraphExecutorValidationTest, NaNOutput_WarnsButPasses_WhenFailDisabled)
    {
        // Enable validation but disable fail_on_nan
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "0", 1);
        mutableDebugEnv().reload();

        // Create a tensor with NaN
        std::vector<float> data(16);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = (i == 0) ? std::nanf("") : static_cast<float>(i + 1);
        }

        // Create graph with mock stage
        ComputeGraph graph;
        graph.addNode("test_stage",
                      std::make_unique<MockValidationStage>(data.data(), 4, 4),
                      0);

        // Execute - should succeed (warning only)
        GraphExecutorConfig config;
        GraphExecutor executor(config);
        CPUDeviceContext ctx(0);

        bool success = executor.execute(graph, &ctx);

        EXPECT_TRUE(success); // Warning issued but no failure
    }

    TEST_F(GraphExecutorValidationTest, NaNOutput_Fails_WhenFailEnabled)
    {
        // Enable validation with fail_on_nan
        setenv("LLAMINAR_VALIDATE_BUFFERS", "1", 1);
        setenv("LLAMINAR_FAIL_ON_NAN", "1", 1);
        mutableDebugEnv().reload();

        // Create a tensor with NaN
        std::vector<float> data(16);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = (i == 0) ? std::nanf("") : static_cast<float>(i + 1);
        }

        // Create graph with mock stage
        ComputeGraph graph;
        graph.addNode("test_stage",
                      std::make_unique<MockValidationStage>(data.data(), 4, 4),
                      0);

        // Execute - should throw VerificationFailure exception (fail-fast behavior)
        GraphExecutorConfig config;
        GraphExecutor executor(config);
        CPUDeviceContext ctx(0);

        EXPECT_THROW({
            executor.execute(graph, &ctx);
        }, verification::VerificationFailure);
    }

    TEST_F(GraphExecutorValidationTest, ValidationDisabled_SkipsChecks)
    {
        // Disable validation
        setenv("LLAMINAR_VALIDATE_BUFFERS", "0", 1);
        setenv("LLAMINAR_FAIL_ON_ZERO", "1", 1); // Would fail if checked
        mutableDebugEnv().reload();

        // Create a zero tensor
        std::vector<float> data(16, 0.0f);

        // Create graph with mock stage
        ComputeGraph graph;
        graph.addNode("test_stage",
                      std::make_unique<MockValidationStage>(data.data(), 4, 4),
                      0);

        // Execute - should succeed because validation is disabled
        GraphExecutorConfig config;
        GraphExecutor executor(config);
        CPUDeviceContext ctx(0);

        bool success = executor.execute(graph, &ctx);

        EXPECT_TRUE(success);
    }
#endif // NDEBUG

} // namespace llaminar2::test
