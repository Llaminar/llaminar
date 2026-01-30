/**
 * @file Test__ExecutionPolicy.cpp
 * @brief Unit tests for ExecutionPolicy struct
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the ExecutionPolicy declarative configuration system including:
 * - Factory methods (allEnabled, noop, ffnOnly, attentionOnly)
 * - Environment variable parsing (fromEnvironment via DebugEnv)
 * - Utility methods (isFullyEnabled, isNoop, toString)
 * - Comparison operators
 */

#include <gtest/gtest.h>
#include "execution/config/ExecutionPolicy.h"
#include "utils/DebugEnv.h"
#include <cstdlib>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class ExecutionPolicyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear all LLAMINAR_EXEC_* environment variables before each test
        clearAllExecEnvVars();
        // Reset ExecutionConfig to fresh default state and reload
        resetExecutionConfig();
    }

    void TearDown() override
    {
        // Clean up environment variables after each test
        clearAllExecEnvVars();
        // Reset ExecutionConfig to fresh default state
        resetExecutionConfig();
    }

    /// Helper: Clear all LLAMINAR_EXEC_* environment variables
    void clearAllExecEnvVars()
    {
        unsetenv("LLAMINAR_EXEC_RMSNORM");
        unsetenv("LLAMINAR_EXEC_ROPE");
        unsetenv("LLAMINAR_EXEC_ATTENTION");
        unsetenv("LLAMINAR_EXEC_GEMM");
        unsetenv("LLAMINAR_EXEC_SWIGLU");
        unsetenv("LLAMINAR_EXEC_RESIDUAL");
    }

    /// Helper: Reset ExecutionConfig to default state
    /// Since reload() doesn't reset to defaults, we need to manually reset fields
    void resetExecutionConfig()
    {
        auto &exec = mutableDebugEnv().execution;
        exec.exec_rmsnorm = false;
        exec.exec_rope = false;
        exec.exec_attention = false;
        exec.exec_gemm = false;
        exec.exec_swiglu = false;
        exec.exec_residual = false;
        // Now reload from (cleared) environment
        exec.reload();
    }

    /// Helper: Set environment variable, clear all others, and reload DebugEnv
    void setOnlyEnvAndReload(const char *name, const char *value)
    {
        clearAllExecEnvVars();
        setenv(name, value, 1);
        resetExecutionConfig();
    }

    /// Helper: Set multiple environment variables and reload DebugEnv
    void setEnvsAndReload(std::initializer_list<std::pair<const char *, const char *>> vars)
    {
        clearAllExecEnvVars();
        for (const auto &[name, value] : vars)
        {
            setenv(name, value, 1);
        }
        resetExecutionConfig();
    }
};

// =============================================================================
// Default Constructor Tests
// =============================================================================

TEST_F(ExecutionPolicyTest, DefaultConstructor_AllEnabled)
{
    ExecutionPolicy policy;

    // Default-constructed policy should have all flags enabled
    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_TRUE(policy.rope);
    EXPECT_TRUE(policy.attention);
    EXPECT_TRUE(policy.gemm);
    EXPECT_TRUE(policy.swiglu);
    EXPECT_TRUE(policy.residual);
}

// =============================================================================
// Factory Method Tests: allEnabled()
// =============================================================================

TEST_F(ExecutionPolicyTest, AllEnabled_ReturnsFullyEnabledPolicy)
{
    auto policy = ExecutionPolicy::allEnabled();

    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_TRUE(policy.rope);
    EXPECT_TRUE(policy.attention);
    EXPECT_TRUE(policy.gemm);
    EXPECT_TRUE(policy.swiglu);
    EXPECT_TRUE(policy.residual);
}

TEST_F(ExecutionPolicyTest, AllEnabled_IsFullyEnabled)
{
    auto policy = ExecutionPolicy::allEnabled();
    EXPECT_TRUE(policy.isFullyEnabled());
    EXPECT_FALSE(policy.isNoop());
}

// =============================================================================
// Factory Method Tests: noop()
// =============================================================================

TEST_F(ExecutionPolicyTest, Noop_ReturnsFullyDisabledPolicy)
{
    auto policy = ExecutionPolicy::noop();

    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, Noop_IsNoop)
{
    auto policy = ExecutionPolicy::noop();
    EXPECT_TRUE(policy.isNoop());
    EXPECT_FALSE(policy.isFullyEnabled());
}

// =============================================================================
// Factory Method Tests: ffnOnly()
// =============================================================================

TEST_F(ExecutionPolicyTest, FfnOnly_DisablesAttentionAndRope)
{
    auto policy = ExecutionPolicy::ffnOnly();

    // FFN operations enabled
    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_TRUE(policy.gemm);
    EXPECT_TRUE(policy.swiglu);
    EXPECT_TRUE(policy.residual);

    // Attention operations disabled
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
}

TEST_F(ExecutionPolicyTest, FfnOnly_NotFullyEnabled)
{
    auto policy = ExecutionPolicy::ffnOnly();
    EXPECT_FALSE(policy.isFullyEnabled());
    EXPECT_FALSE(policy.isNoop());
}

// =============================================================================
// Factory Method Tests: attentionOnly()
// =============================================================================

TEST_F(ExecutionPolicyTest, AttentionOnly_DisablesSwiGLU)
{
    auto policy = ExecutionPolicy::attentionOnly();

    // Attention operations enabled
    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_TRUE(policy.rope);
    EXPECT_TRUE(policy.attention);
    EXPECT_TRUE(policy.gemm);
    EXPECT_TRUE(policy.residual);

    // FFN-specific operations disabled
    EXPECT_FALSE(policy.swiglu);
}

TEST_F(ExecutionPolicyTest, AttentionOnly_NotFullyEnabled)
{
    auto policy = ExecutionPolicy::attentionOnly();
    EXPECT_FALSE(policy.isFullyEnabled());
    EXPECT_FALSE(policy.isNoop());
}

// =============================================================================
// Factory Method Tests: fromEnvironment()
// =============================================================================

TEST_F(ExecutionPolicyTest, FromEnvironment_DefaultsToAllDisabled)
{
    // No environment variables set - DebugEnv defaults are all false
    // (ComputeStage migration is opt-in)
    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_EnableRmsNorm)
{
    setOnlyEnvAndReload("LLAMINAR_EXEC_RMSNORM", "1");

    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_EnableRope)
{
    setOnlyEnvAndReload("LLAMINAR_EXEC_ROPE", "1");

    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_TRUE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_EnableAttention)
{
    setOnlyEnvAndReload("LLAMINAR_EXEC_ATTENTION", "1");

    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_TRUE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_EnableSwiGLU)
{
    setOnlyEnvAndReload("LLAMINAR_EXEC_SWIGLU", "1");

    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_TRUE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_EnableResidual)
{
    setOnlyEnvAndReload("LLAMINAR_EXEC_RESIDUAL", "1");

    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_TRUE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_EnableMultiple)
{
    setEnvsAndReload({{"LLAMINAR_EXEC_RMSNORM", "1"},
                      {"LLAMINAR_EXEC_ROPE", "1"},
                      {"LLAMINAR_EXEC_ATTENTION", "1"}});

    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_TRUE(policy.rope);
    EXPECT_TRUE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_GemmEnablesOnlyGemm)
{
    // Enable GEMM only - cascade logic was removed (Dec 2025)
    // Each flag is independent; setting GEMM=1 only enables GEMM
    setOnlyEnvAndReload("LLAMINAR_EXEC_GEMM", "1");

    auto policy = ExecutionPolicy::fromEnvironment();

    // Only GEMM is enabled; others remain at baseline (false from resetExecutionConfig)
    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_TRUE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_DisableGemm)
{
    // Explicitly set GEMM=0 (should remain disabled)
    setOnlyEnvAndReload("LLAMINAR_EXEC_GEMM", "0");

    auto policy = ExecutionPolicy::fromEnvironment();

    // All should be disabled (defaults)
    EXPECT_FALSE(policy.rmsnorm);
    EXPECT_FALSE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_FALSE(policy.gemm);
    EXPECT_FALSE(policy.swiglu);
    EXPECT_FALSE(policy.residual);
}

TEST_F(ExecutionPolicyTest, FromEnvironment_NonZeroMeansEnabled)
{
    // Any non-zero value should mean enabled
    setEnvsAndReload({{"LLAMINAR_EXEC_RMSNORM", "1"},
                      {"LLAMINAR_EXEC_ROPE", "42"},
                      {"LLAMINAR_EXEC_ATTENTION", "-1"}});

    auto policy = ExecutionPolicy::fromEnvironment();

    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_TRUE(policy.rope);
    EXPECT_TRUE(policy.attention);
}

// =============================================================================
// Utility Method Tests: toString()
// =============================================================================

TEST_F(ExecutionPolicyTest, ToString_AllEnabled)
{
    auto policy = ExecutionPolicy::allEnabled();
    std::string str = policy.toString();

    EXPECT_NE(str.find("ExecutionPolicy{"), std::string::npos);
    EXPECT_NE(str.find("rmsnorm=1"), std::string::npos);
    EXPECT_NE(str.find("rope=1"), std::string::npos);
    EXPECT_NE(str.find("attention=1"), std::string::npos);
    EXPECT_NE(str.find("gemm=1"), std::string::npos);
    EXPECT_NE(str.find("swiglu=1"), std::string::npos);
    EXPECT_NE(str.find("residual=1"), std::string::npos);
}

TEST_F(ExecutionPolicyTest, ToString_Noop)
{
    auto policy = ExecutionPolicy::noop();
    std::string str = policy.toString();

    EXPECT_NE(str.find("ExecutionPolicy{"), std::string::npos);
    EXPECT_NE(str.find("rmsnorm=0"), std::string::npos);
    EXPECT_NE(str.find("rope=0"), std::string::npos);
    EXPECT_NE(str.find("attention=0"), std::string::npos);
    EXPECT_NE(str.find("gemm=0"), std::string::npos);
    EXPECT_NE(str.find("swiglu=0"), std::string::npos);
    EXPECT_NE(str.find("residual=0"), std::string::npos);
}

TEST_F(ExecutionPolicyTest, ToString_Mixed)
{
    auto policy = ExecutionPolicy::ffnOnly();
    std::string str = policy.toString();

    EXPECT_NE(str.find("rope=0"), std::string::npos);
    EXPECT_NE(str.find("attention=0"), std::string::npos);
    EXPECT_NE(str.find("gemm=1"), std::string::npos);
    EXPECT_NE(str.find("swiglu=1"), std::string::npos);
}

// =============================================================================
// Comparison Operator Tests
// =============================================================================

TEST_F(ExecutionPolicyTest, Equality_SamePolicies)
{
    auto policy1 = ExecutionPolicy::allEnabled();
    auto policy2 = ExecutionPolicy::allEnabled();

    EXPECT_EQ(policy1, policy2);
    EXPECT_FALSE(policy1 != policy2);
}

TEST_F(ExecutionPolicyTest, Equality_DifferentPolicies)
{
    auto policy1 = ExecutionPolicy::allEnabled();
    auto policy2 = ExecutionPolicy::noop();

    EXPECT_NE(policy1, policy2);
    EXPECT_FALSE(policy1 == policy2);
}

TEST_F(ExecutionPolicyTest, Equality_SingleFlagDifference)
{
    auto policy1 = ExecutionPolicy::allEnabled();
    auto policy2 = ExecutionPolicy::allEnabled();
    policy2.rmsnorm = false;

    EXPECT_NE(policy1, policy2);
}

TEST_F(ExecutionPolicyTest, Equality_FactoryMethodsProduceSameResults)
{
    // Multiple calls to same factory should produce equal policies
    EXPECT_EQ(ExecutionPolicy::allEnabled(), ExecutionPolicy::allEnabled());
    EXPECT_EQ(ExecutionPolicy::noop(), ExecutionPolicy::noop());
    EXPECT_EQ(ExecutionPolicy::ffnOnly(), ExecutionPolicy::ffnOnly());
    EXPECT_EQ(ExecutionPolicy::attentionOnly(), ExecutionPolicy::attentionOnly());
}

TEST_F(ExecutionPolicyTest, Inequality_AllFactoryMethodsDiffer)
{
    auto allEnabled = ExecutionPolicy::allEnabled();
    auto noop = ExecutionPolicy::noop();
    auto ffnOnly = ExecutionPolicy::ffnOnly();
    auto attentionOnly = ExecutionPolicy::attentionOnly();

    // All factory methods should produce different policies
    EXPECT_NE(allEnabled, noop);
    EXPECT_NE(allEnabled, ffnOnly);
    EXPECT_NE(allEnabled, attentionOnly);
    EXPECT_NE(noop, ffnOnly);
    EXPECT_NE(noop, attentionOnly);
    EXPECT_NE(ffnOnly, attentionOnly);
}

// =============================================================================
// Copy/Move Semantics Tests
// =============================================================================

TEST_F(ExecutionPolicyTest, CopyConstructor)
{
    auto original = ExecutionPolicy::ffnOnly();
    ExecutionPolicy copy = original;

    EXPECT_EQ(original, copy);
    EXPECT_EQ(original.rmsnorm, copy.rmsnorm);
    EXPECT_EQ(original.rope, copy.rope);
    EXPECT_EQ(original.attention, copy.attention);
    EXPECT_EQ(original.gemm, copy.gemm);
    EXPECT_EQ(original.swiglu, copy.swiglu);
    EXPECT_EQ(original.residual, copy.residual);
}

TEST_F(ExecutionPolicyTest, CopyAssignment)
{
    auto original = ExecutionPolicy::attentionOnly();
    ExecutionPolicy copy = ExecutionPolicy::noop();

    EXPECT_NE(original, copy);

    copy = original;

    EXPECT_EQ(original, copy);
}

TEST_F(ExecutionPolicyTest, MoveConstructor)
{
    auto original = ExecutionPolicy::ffnOnly();
    auto originalCopy = original; // Keep a copy for comparison

    ExecutionPolicy moved = std::move(original);

    EXPECT_EQ(moved, originalCopy);
}

TEST_F(ExecutionPolicyTest, MoveAssignment)
{
    auto original = ExecutionPolicy::attentionOnly();
    auto originalCopy = original;

    ExecutionPolicy moved = ExecutionPolicy::noop();
    moved = std::move(original);

    EXPECT_EQ(moved, originalCopy);
}

// =============================================================================
// Utility Method Edge Cases
// =============================================================================

TEST_F(ExecutionPolicyTest, IsFullyEnabled_SingleFlagDisabled)
{
    auto policy = ExecutionPolicy::allEnabled();

    // Disable each flag one at a time and verify isFullyEnabled returns false
    policy.rmsnorm = false;
    EXPECT_FALSE(policy.isFullyEnabled());
    policy.rmsnorm = true;

    policy.rope = false;
    EXPECT_FALSE(policy.isFullyEnabled());
    policy.rope = true;

    policy.attention = false;
    EXPECT_FALSE(policy.isFullyEnabled());
    policy.attention = true;

    policy.gemm = false;
    EXPECT_FALSE(policy.isFullyEnabled());
    policy.gemm = true;

    policy.swiglu = false;
    EXPECT_FALSE(policy.isFullyEnabled());
    policy.swiglu = true;

    policy.residual = false;
    EXPECT_FALSE(policy.isFullyEnabled());
    policy.residual = true;

    // Back to all enabled
    EXPECT_TRUE(policy.isFullyEnabled());
}

TEST_F(ExecutionPolicyTest, IsNoop_SingleFlagEnabled)
{
    auto policy = ExecutionPolicy::noop();

    // Enable each flag one at a time and verify isNoop returns false
    policy.rmsnorm = true;
    EXPECT_FALSE(policy.isNoop());
    policy.rmsnorm = false;

    policy.rope = true;
    EXPECT_FALSE(policy.isNoop());
    policy.rope = false;

    policy.attention = true;
    EXPECT_FALSE(policy.isNoop());
    policy.attention = false;

    policy.gemm = true;
    EXPECT_FALSE(policy.isNoop());
    policy.gemm = false;

    policy.swiglu = true;
    EXPECT_FALSE(policy.isNoop());
    policy.swiglu = false;

    policy.residual = true;
    EXPECT_FALSE(policy.isNoop());
    policy.residual = false;

    // Back to all disabled
    EXPECT_TRUE(policy.isNoop());
}

// =============================================================================
// Direct Field Modification Tests
// =============================================================================

TEST_F(ExecutionPolicyTest, DirectFieldModification)
{
    ExecutionPolicy policy = ExecutionPolicy::allEnabled();

    // Modify individual fields
    policy.attention = false;

    EXPECT_TRUE(policy.rmsnorm);
    EXPECT_TRUE(policy.rope);
    EXPECT_FALSE(policy.attention);
    EXPECT_TRUE(policy.gemm);
    EXPECT_TRUE(policy.swiglu);
    EXPECT_TRUE(policy.residual);
}

TEST_F(ExecutionPolicyTest, CreateCustomPolicy)
{
    // Test creating a custom policy via designated initializers
    ExecutionPolicy custom{
        .rmsnorm = true,
        .rope = false,
        .attention = true,
        .gemm = false,
        .swiglu = true,
        .residual = false};

    EXPECT_TRUE(custom.rmsnorm);
    EXPECT_FALSE(custom.rope);
    EXPECT_TRUE(custom.attention);
    EXPECT_FALSE(custom.gemm);
    EXPECT_TRUE(custom.swiglu);
    EXPECT_FALSE(custom.residual);

    EXPECT_FALSE(custom.isFullyEnabled());
    EXPECT_FALSE(custom.isNoop());
}
