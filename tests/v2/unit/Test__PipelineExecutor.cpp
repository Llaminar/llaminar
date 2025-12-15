/**
 * @file Test__PipelineExecutor.cpp
 * @brief Unit tests for PipelineExecutor adapter
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the PipelineExecutor which bridges LayerExecutor with existing pipeline
 * infrastructure. Tests focus on:
 * 1. Configuration and setup
 * 2. Device context management
 * 3. Individual operation execution
 * 4. Integration with ComputeStage
 */

#include <gtest/gtest.h>
#include "pipelines/PipelineExecutor.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include <memory>
#include <cmath>

namespace llaminar2
{
    namespace
    {

        // =============================================================================
        // Test Fixture
        // =============================================================================

        class Test__PipelineExecutor : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Create default config with layer executor disabled
                config_.use_layer_executor = false;
                config_.use_compute_graph = false;
                config_.execution_mode = ExecutionMode::SEQUENTIAL;
                config_.enable_profiling = false;

                // Create executor without MPI context (single rank test)
                executor_ = std::make_unique<PipelineExecutor>(config_, nullptr);
            }

            void TearDown() override
            {
                executor_.reset();
            }

            // Create a simple FP32 tensor for testing
            std::unique_ptr<FP32Tensor> createTestTensor(int rows, int cols, float init_val = 0.0f)
            {
                auto tensor = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
                    0 // device_idx
                );

                // Initialize with value
                float *data = tensor->mutable_data();
                size_t numel = tensor->numel();
                for (size_t i = 0; i < numel; ++i)
                {
                    data[i] = init_val;
                }

                return tensor;
            }

            PipelineExecutorConfig config_;
            std::unique_ptr<PipelineExecutor> executor_;
        };

        // =============================================================================
        // Construction Tests
        // =============================================================================

        TEST_F(Test__PipelineExecutor, Construction_Default)
        {
            EXPECT_FALSE(executor_->config().use_layer_executor);
            EXPECT_FALSE(executor_->config().use_compute_graph);
            EXPECT_EQ(executor_->config().execution_mode, ExecutionMode::SEQUENTIAL);
        }

        TEST_F(Test__PipelineExecutor, Construction_WithConfig)
        {
            PipelineExecutorConfig config;
            config.use_layer_executor = true;
            config.enable_profiling = true;
            config.execution_mode = ExecutionMode::PARALLEL;

            PipelineExecutor exec(config, nullptr);

            EXPECT_TRUE(exec.config().use_layer_executor);
            EXPECT_TRUE(exec.config().enable_profiling);
            EXPECT_EQ(exec.config().execution_mode, ExecutionMode::PARALLEL);
        }

        TEST_F(Test__PipelineExecutor, Construction_WithMPIContext)
        {
            auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

            PipelineExecutorConfig config;
            PipelineExecutor exec(config, mpi_ctx);

            // Should not crash
            EXPECT_FALSE(exec.config().use_layer_executor);
        }

        // =============================================================================
        // Configuration Tests
        // =============================================================================

        TEST_F(Test__PipelineExecutor, SetConfig)
        {
            PipelineExecutorConfig new_config;
            new_config.use_layer_executor = true;
            new_config.executor_ffn_norm = true;
            new_config.executor_ffn_swiglu = true;

            executor_->setConfig(new_config);

            EXPECT_TRUE(executor_->config().use_layer_executor);
            EXPECT_TRUE(executor_->config().executor_ffn_norm);
            EXPECT_TRUE(executor_->config().executor_ffn_swiglu);
        }

        // =============================================================================
        // Device Context Tests
        // =============================================================================

        TEST_F(Test__PipelineExecutor, SetDeviceContext_CPU)
        {
            executor_->setDeviceContext(0, 4);

            auto *ctx = executor_->getDeviceContext(0);
            ASSERT_NE(ctx, nullptr);
            EXPECT_EQ(ctx->deviceIndex(), 0);
            EXPECT_EQ(ctx->backendType(), ComputeBackendType::CPU_OPENBLAS);
        }

        TEST_F(Test__PipelineExecutor, SetDeviceContext_MultipleDevices)
        {
            executor_->setDeviceContext(0, 4);
            executor_->setDeviceContext(1, 8);

            auto *ctx0 = executor_->getDeviceContext(0);
            auto *ctx1 = executor_->getDeviceContext(1);

            ASSERT_NE(ctx0, nullptr);
            ASSERT_NE(ctx1, nullptr);
            EXPECT_EQ(ctx0->deviceIndex(), 0);
            EXPECT_EQ(ctx1->deviceIndex(), 1);
        }

        TEST_F(Test__PipelineExecutor, GetDeviceContext_NonExistent)
        {
            auto *ctx = executor_->getDeviceContext(99);
            EXPECT_EQ(ctx, nullptr);
        }

        TEST_F(Test__PipelineExecutor, SetDeviceContext_Idempotent)
        {
            executor_->setDeviceContext(0, 4);
            auto *ctx1 = executor_->getDeviceContext(0);

            // Setting again should not change anything
            executor_->setDeviceContext(0, 8);
            auto *ctx2 = executor_->getDeviceContext(0);

            EXPECT_EQ(ctx1, ctx2); // Same pointer
        }

        // =============================================================================
        // Operation Execution Tests - Disabled Mode
        // =============================================================================

        TEST_F(Test__PipelineExecutor, ExecuteRMSNorm_Disabled)
        {
            // Layer executor is disabled by default
            auto input = createTestTensor(32, 128, 1.0f);
            auto gamma = createTestTensor(1, 128, 1.0f);
            auto output = createTestTensor(32, 128);

            bool result = executor_->executeRMSNorm(
                input.get(), gamma.get(), output.get(),
                32, 128, 1e-5f, 0);

            EXPECT_FALSE(result); // Should fail when disabled
        }

        TEST_F(Test__PipelineExecutor, ExecuteSwiGLU_Disabled)
        {
            auto gate = createTestTensor(32, 256);
            auto up = createTestTensor(32, 256);
            auto output = createTestTensor(32, 256);

            bool result = executor_->executeSwiGLU(
                gate.get(), up.get(), output.get(),
                32, 256, 0);

            EXPECT_FALSE(result); // Should fail when disabled
        }

        TEST_F(Test__PipelineExecutor, ExecuteResidualAdd_Disabled)
        {
            auto input = createTestTensor(32, 128);
            auto residual = createTestTensor(32, 128);
            auto output = createTestTensor(32, 128);

            bool result = executor_->executeResidualAdd(
                input.get(), residual.get(), output.get(),
                32 * 128, 0);

            EXPECT_FALSE(result); // Should fail when disabled
        }

        TEST_F(Test__PipelineExecutor, ExecuteRoPE_Disabled)
        {
            auto Q = createTestTensor(32, 896); // seq_len * (n_heads * head_dim)
            auto K = createTestTensor(32, 128); // seq_len * (n_kv_heads * head_dim)

            bool result = executor_->executeRoPE(
                Q.get(), K.get(), nullptr,
                32, 14, 2, 64, 10000.0f, 0);

            EXPECT_FALSE(result); // Should fail when disabled
        }

        // =============================================================================
        // Operation Execution Tests - Enabled Mode
        // =============================================================================

        TEST_F(Test__PipelineExecutor, ExecuteRMSNorm_Enabled)
        {
            // Enable layer executor
            PipelineExecutorConfig config;
            config.use_layer_executor = true;
            executor_->setConfig(config);
            executor_->setDeviceContext(0, 4);

            // Create tensors
            auto input = createTestTensor(32, 128, 1.0f);
            auto gamma = createTestTensor(1, 128, 1.0f);
            auto output = createTestTensor(32, 128);

            bool result = executor_->executeRMSNorm(
                input.get(), gamma.get(), output.get(),
                32, 128, 1e-5f, 0);

            // Execute should succeed
            EXPECT_TRUE(result);
        }

        TEST_F(Test__PipelineExecutor, ExecuteSwiGLU_Enabled)
        {
            PipelineExecutorConfig config;
            config.use_layer_executor = true;
            executor_->setConfig(config);
            executor_->setDeviceContext(0, 4);

            auto gate = createTestTensor(32, 256, 0.5f);
            auto up = createTestTensor(32, 256, 0.5f);
            auto output = createTestTensor(32, 256);

            bool result = executor_->executeSwiGLU(
                gate.get(), up.get(), output.get(),
                32, 256, 0);

            EXPECT_TRUE(result);
        }

        TEST_F(Test__PipelineExecutor, ExecuteResidualAdd_Enabled)
        {
            PipelineExecutorConfig config;
            config.use_layer_executor = true;
            executor_->setConfig(config);
            executor_->setDeviceContext(0, 4);

            auto input = createTestTensor(32, 128, 1.0f);
            auto residual = createTestTensor(32, 128, 2.0f);
            auto output = createTestTensor(32, 128);

            bool result = executor_->executeResidualAdd(
                input.get(), residual.get(), output.get(),
                32 * 128, 0);

            EXPECT_TRUE(result);
        }

        TEST_F(Test__PipelineExecutor, ExecuteRoPE_Enabled)
        {
            PipelineExecutorConfig config;
            config.use_layer_executor = true;
            executor_->setConfig(config);
            executor_->setDeviceContext(0, 4);

            // Q tensor: [seq_len, n_heads * head_dim]
            auto Q = createTestTensor(32, 14 * 64, 1.0f);
            // K tensor: [seq_len, n_kv_heads * head_dim]
            auto K = createTestTensor(32, 2 * 64, 1.0f);

            bool result = executor_->executeRoPE(
                Q.get(), K.get(), nullptr,
                32, 14, 2, 64, 10000.0f, 0);

            EXPECT_TRUE(result);
        }

        // =============================================================================
        // Statistics Tests
        // =============================================================================

        TEST_F(Test__PipelineExecutor, Stats_Initial)
        {
            const auto &stats = executor_->stats();
            EXPECT_EQ(stats.total_stages_executed, 0u);
            EXPECT_EQ(stats.total_flops, 0u);
            EXPECT_DOUBLE_EQ(stats.total_time_ms, 0.0);
        }

        TEST_F(Test__PipelineExecutor, Stats_Reset)
        {
            executor_->resetStats();

            const auto &stats = executor_->stats();
            EXPECT_EQ(stats.total_stages_executed, 0u);
            EXPECT_TRUE(stats.stage_times_ms.empty());
        }

        // =============================================================================
        // Feature Flag Tests
        // =============================================================================

        TEST_F(Test__PipelineExecutor, FeatureFlags_FFN)
        {
            PipelineExecutorConfig config;
            config.executor_ffn_norm = true;
            config.executor_ffn_swiglu = true;
            config.executor_ffn_residual = true;

            executor_->setConfig(config);

            EXPECT_TRUE(executor_->config().executor_ffn_norm);
            EXPECT_TRUE(executor_->config().executor_ffn_swiglu);
            EXPECT_TRUE(executor_->config().executor_ffn_residual);
        }

        TEST_F(Test__PipelineExecutor, FeatureFlags_Attention)
        {
            PipelineExecutorConfig config;
            config.executor_attn_norm = true;
            config.executor_attn_residual = true;

            executor_->setConfig(config);

            EXPECT_TRUE(executor_->config().executor_attn_norm);
            EXPECT_TRUE(executor_->config().executor_attn_residual);
        }

        // =============================================================================
        // Auto-Context Creation Tests
        // =============================================================================

        TEST_F(Test__PipelineExecutor, AutoContextCreation)
        {
            PipelineExecutorConfig config;
            config.use_layer_executor = true;
            executor_->setConfig(config);

            // Don't explicitly create context - let operation auto-create it
            auto input = createTestTensor(8, 64, 1.0f);
            auto gamma = createTestTensor(1, 64, 1.0f);
            auto output = createTestTensor(8, 64);

            bool result = executor_->executeRMSNorm(
                input.get(), gamma.get(), output.get(),
                8, 64, 1e-5f, 0);

            EXPECT_TRUE(result);

            // Context should have been auto-created
            auto *ctx = executor_->getDeviceContext(0);
            EXPECT_NE(ctx, nullptr);
        }

        // =============================================================================
        // Multi-Device Tests
        // =============================================================================

        TEST_F(Test__PipelineExecutor, MultiDevice_DifferentContexts)
        {
            PipelineExecutorConfig config;
            config.use_layer_executor = true;
            executor_->setConfig(config);

            executor_->setDeviceContext(0, 4);
            executor_->setDeviceContext(1, 8);

            auto input0 = createTestTensor(32, 128, 1.0f);
            auto gamma0 = createTestTensor(1, 128, 1.0f);
            auto output0 = createTestTensor(32, 128);

            auto input1 = createTestTensor(32, 128, 2.0f);
            auto gamma1 = createTestTensor(1, 128, 1.0f);
            auto output1 = createTestTensor(32, 128);

            // Execute on device 0
            bool result0 = executor_->executeRMSNorm(
                input0.get(), gamma0.get(), output0.get(),
                32, 128, 1e-5f, 0);

            // Execute on device 1
            bool result1 = executor_->executeRMSNorm(
                input1.get(), gamma1.get(), output1.get(),
                32, 128, 1e-5f, 1);

            EXPECT_TRUE(result0);
            EXPECT_TRUE(result1);
        }

    } // namespace
} // namespace llaminar2
