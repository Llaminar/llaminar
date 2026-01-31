/**
 * @file Test__ExecutionConfigAutoEnable.cpp
 * @brief Unit tests for ExecutionConfig auto-enable behavior
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests the fix for the "empty graph" bug where graph buffer management
 * was enabled but no exec_* flags were set, resulting in 0 nodes being
 * added to the graph.
 *
 * Root Cause (fixed):
 *   When use_graph_buffer_management=true but all exec_* flags defaulted
 *   to false, the graph would be empty because no ComputeStage nodes
 *   would be created.
 *
 * Fix:
 *   ExecutionConfig::reload() now auto-enables all exec_* flags when
 *   use_graph_buffer_management=true.
 *
 * Scenarios Tested:
 * 1. Default state: all flags false
 * 2. use_graph_buffer_management=true → all exec_* flags become true
 * 3. exec_gemm=true → all layer-level flags become true (data flow dependency)
 * 4. Individual flag override still works
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

#include "v2/utils/DebugEnv.h"

namespace
{

    /**
     * @brief RAII helper to set/restore environment variables during tests
     */
    class EnvVarGuard
    {
    public:
        EnvVarGuard(const std::string &name, const std::string &value)
            : name_(name), had_value_(false)
        {
            const char *existing = std::getenv(name.c_str());
            if (existing)
            {
                had_value_ = true;
                old_value_ = existing;
            }
            setenv(name.c_str(), value.c_str(), 1);
        }

        ~EnvVarGuard()
        {
            if (had_value_)
            {
                setenv(name_.c_str(), old_value_.c_str(), 1);
            }
            else
            {
                unsetenv(name_.c_str());
            }
        }

    private:
        std::string name_;
        std::string old_value_;
        bool had_value_;
    };

    /**
     * @brief RAII helper to unset environment variable during tests
     */
    class EnvVarUnsetGuard
    {
    public:
        explicit EnvVarUnsetGuard(const std::string &name)
            : name_(name), had_value_(false)
        {
            const char *existing = std::getenv(name.c_str());
            if (existing)
            {
                had_value_ = true;
                old_value_ = existing;
            }
            unsetenv(name.c_str());
        }

        ~EnvVarUnsetGuard()
        {
            if (had_value_)
            {
                setenv(name_.c_str(), old_value_.c_str(), 1);
            }
        }

    private:
        std::string name_;
        std::string old_value_;
        bool had_value_;
    };

    class Test__ExecutionConfigAutoEnable : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Clear all relevant env vars to ensure clean state
            unsetenv("LLAMINAR_USE_LAYER_EXECUTOR");
            unsetenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT");
            unsetenv("LLAMINAR_EXEC_RMSNORM");
            unsetenv("LLAMINAR_EXEC_ROPE");
            unsetenv("LLAMINAR_EXEC_ATTENTION");
            unsetenv("LLAMINAR_EXEC_GEMM");
            unsetenv("LLAMINAR_EXEC_SWIGLU");
            unsetenv("LLAMINAR_EXEC_RESIDUAL");
            unsetenv("LLAMINAR_EXEC_EMBEDDING");
            unsetenv("LLAMINAR_EXEC_LM_HEAD");
        }

        void TearDown() override
        {
            // Clean up after test
            unsetenv("LLAMINAR_USE_LAYER_EXECUTOR");
            unsetenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT");
            unsetenv("LLAMINAR_EXEC_RMSNORM");
            unsetenv("LLAMINAR_EXEC_ROPE");
            unsetenv("LLAMINAR_EXEC_ATTENTION");
            unsetenv("LLAMINAR_EXEC_GEMM");
            unsetenv("LLAMINAR_EXEC_SWIGLU");
            unsetenv("LLAMINAR_EXEC_RESIDUAL");
            unsetenv("LLAMINAR_EXEC_EMBEDDING");
            unsetenv("LLAMINAR_EXEC_LM_HEAD");
        }
    };

    // =============================================================================
    // Default State Tests
    // =============================================================================

    TEST_F(Test__ExecutionConfigAutoEnable, DefaultState_AllFlagsEnabled)
    {
        // As of Dec 2025, all flags default to TRUE (graph execution is the default path)
        llaminar2::ExecutionConfig config;

        EXPECT_TRUE(config.use_layer_executor);
        EXPECT_TRUE(config.use_graph_buffer_management);
        EXPECT_TRUE(config.exec_rmsnorm);
        EXPECT_TRUE(config.exec_rope);
        EXPECT_TRUE(config.exec_attention);
        EXPECT_TRUE(config.exec_gemm);
        EXPECT_TRUE(config.exec_swiglu);
        EXPECT_TRUE(config.exec_residual);
        EXPECT_TRUE(config.exec_embedding);
        EXPECT_TRUE(config.exec_lm_head);
    }

    // =============================================================================
    // Graph Buffer Management Auto-Enable Tests
    // =============================================================================

    /**
     * @brief This is the key regression test for the "empty graph" bug.
     *
     * Bug scenario (before fix):
     *   - User sets LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=1
     *   - All exec_* flags remain false (default)
     *   - Graph has 0 nodes
     *   - No computation happens
     *   - Output is garbage (repeating tokens)
     *
     * After fix:
     *   - When use_graph_buffer_management=true, all exec_* flags are auto-enabled
     *   - Graph has correct number of nodes (7 per layer)
     *   - Computation proceeds correctly
     */
    TEST_F(Test__ExecutionConfigAutoEnable, GraphBufferManagement_AutoEnablesAllExecFlags)
    {
        EnvVarGuard guard("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT", "1");

        llaminar2::ExecutionConfig config;

        // Graph buffer management should be enabled
        EXPECT_TRUE(config.use_graph_buffer_management);

        // ALL layer-level exec_* flags should be auto-enabled
        EXPECT_TRUE(config.exec_rmsnorm) << "exec_rmsnorm should be auto-enabled for graph buffer management";
        EXPECT_TRUE(config.exec_rope) << "exec_rope should be auto-enabled for graph buffer management";
        EXPECT_TRUE(config.exec_attention) << "exec_attention should be auto-enabled for graph buffer management";
        EXPECT_TRUE(config.exec_gemm) << "exec_gemm should be auto-enabled for graph buffer management";
        EXPECT_TRUE(config.exec_swiglu) << "exec_swiglu should be auto-enabled for graph buffer management";
        EXPECT_TRUE(config.exec_residual) << "exec_residual should be auto-enabled for graph buffer management";
    }

    TEST_F(Test__ExecutionConfigAutoEnable, GraphBufferManagement_WithValue0_DisablesOnlyThatFlag)
    {
        // Explicitly setting to 0 disables only that flag, others remain at default (true)
        EnvVarGuard guard("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT", "0");

        llaminar2::ExecutionConfig config;

        EXPECT_FALSE(config.use_graph_buffer_management);
        // Other flags remain at their defaults (true)
        EXPECT_TRUE(config.exec_rmsnorm);
        EXPECT_TRUE(config.exec_rope);
        EXPECT_TRUE(config.exec_attention);
        EXPECT_TRUE(config.exec_gemm);
        EXPECT_TRUE(config.exec_swiglu);
        EXPECT_TRUE(config.exec_residual);
    }

    // =============================================================================
    // GEMM Flag Cascade Tests
    // =============================================================================

    /**
     * @brief GEMM requires correct data flow from all preceding stages
     *
     * When exec_gemm=true, all other layer-level flags must be enabled
     * to maintain correct data flow through the pipeline.
     */
    TEST_F(Test__ExecutionConfigAutoEnable, ExecGemm_CascadesToOtherFlags)
    {
        EnvVarGuard guard("LLAMINAR_EXEC_GEMM", "1");

        llaminar2::ExecutionConfig config;

        // GEMM should be enabled
        EXPECT_TRUE(config.exec_gemm);

        // All layer-level flags should cascade
        EXPECT_TRUE(config.exec_rmsnorm) << "RMSNorm must be enabled for GEMM data flow";
        EXPECT_TRUE(config.exec_rope) << "RoPE must be enabled for GEMM data flow";
        EXPECT_TRUE(config.exec_attention) << "Attention must be enabled for GEMM data flow";
        EXPECT_TRUE(config.exec_swiglu) << "SwiGLU must be enabled for GEMM data flow";
        EXPECT_TRUE(config.exec_residual) << "Residual must be enabled for GEMM data flow";
    }

    // =============================================================================
    // Individual Flag Override Tests
    // =============================================================================

    TEST_F(Test__ExecutionConfigAutoEnable, IndividualFlags_CanBeDisabledIndependently)
    {
        // Test disabling individual flags (all default to true)
        EnvVarGuard guard1("LLAMINAR_EXEC_RMSNORM", "0");

        llaminar2::ExecutionConfig config;

        // Only RMSNorm should be disabled, others remain true
        EXPECT_FALSE(config.exec_rmsnorm);
        EXPECT_TRUE(config.exec_rope);
        EXPECT_TRUE(config.exec_attention);
        EXPECT_TRUE(config.exec_gemm);
        EXPECT_TRUE(config.exec_swiglu);
        EXPECT_TRUE(config.exec_residual);
    }

    TEST_F(Test__ExecutionConfigAutoEnable, MultipleIndividualFlags_CanBeDisabled)
    {
        EnvVarGuard guard1("LLAMINAR_EXEC_GEMM", "0");
        EnvVarGuard guard2("LLAMINAR_EXEC_SWIGLU", "0");
        EnvVarGuard guard3("LLAMINAR_EXEC_RESIDUAL", "0");

        llaminar2::ExecutionConfig config;

        EXPECT_TRUE(config.exec_rmsnorm);
        EXPECT_TRUE(config.exec_rope);
        EXPECT_TRUE(config.exec_attention);
        EXPECT_FALSE(config.exec_gemm);
        EXPECT_FALSE(config.exec_swiglu);
        EXPECT_FALSE(config.exec_residual);
    }

    // =============================================================================
    // Model-Level Flags Tests (Embedding, LM Head)
    // =============================================================================

    TEST_F(Test__ExecutionConfigAutoEnable, ModelLevelFlags_EnabledByDefault)
    {
        // As of Dec 2025, model-level flags are also true by default
        llaminar2::ExecutionConfig config;

        // Model-level flags are enabled by default
        EXPECT_TRUE(config.exec_embedding) << "Embedding should be enabled by default";
        EXPECT_TRUE(config.exec_lm_head) << "LM head should be enabled by default";
    }

    TEST_F(Test__ExecutionConfigAutoEnable, ModelLevelFlags_CanBeSetExplicitly)
    {
        EnvVarGuard guard1("LLAMINAR_EXEC_EMBEDDING", "1");
        EnvVarGuard guard2("LLAMINAR_EXEC_LM_HEAD", "1");

        llaminar2::ExecutionConfig config;

        EXPECT_TRUE(config.exec_embedding);
        EXPECT_TRUE(config.exec_lm_head);
    }

    // =============================================================================
    // Reload Tests
    // =============================================================================

    TEST_F(Test__ExecutionConfigAutoEnable, Reload_PicksUpNewEnvVars)
    {
        llaminar2::ExecutionConfig config;

        // Initially all true (defaults)
        EXPECT_TRUE(config.use_graph_buffer_management);
        EXPECT_TRUE(config.exec_rmsnorm);

        // Set env vars to disable
        setenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT", "0", 1);
        setenv("LLAMINAR_EXEC_RMSNORM", "0", 1);

        // Reload should pick up new values
        config.reload();

        EXPECT_FALSE(config.use_graph_buffer_management);
        EXPECT_FALSE(config.exec_rmsnorm);
        // Other flags remain at their defaults (true) unless explicitly set
        EXPECT_TRUE(config.exec_rope);
        EXPECT_TRUE(config.exec_attention);
        EXPECT_TRUE(config.exec_gemm);
        EXPECT_TRUE(config.exec_swiglu);
        EXPECT_TRUE(config.exec_residual);

        // Clean up
        unsetenv("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT");
        unsetenv("LLAMINAR_EXEC_RMSNORM");
    }

    // =============================================================================
    // Edge Cases
    // =============================================================================

    TEST_F(Test__ExecutionConfigAutoEnable, InvalidEnvValue_TreatedAsFalse)
    {
        EnvVarGuard guard("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT", "invalid");

        llaminar2::ExecutionConfig config;

        // atoi("invalid") returns 0, so should be treated as false
        EXPECT_FALSE(config.use_graph_buffer_management);
    }

    TEST_F(Test__ExecutionConfigAutoEnable, EmptyEnvValue_TreatedAsFalse)
    {
        EnvVarGuard guard("LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT", "");

        llaminar2::ExecutionConfig config;

        // atoi("") returns 0, so should be treated as false
        EXPECT_FALSE(config.use_graph_buffer_management);
    }

} // namespace
