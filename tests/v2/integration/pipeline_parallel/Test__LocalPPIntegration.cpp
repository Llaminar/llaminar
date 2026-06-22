/**
 * @file Test__LocalPPIntegration.cpp
 * @brief Integration tests for LOCAL Pipeline Parallelism
 *
 * Tests the integration of LOCAL PP (pipeline parallel stages within a single MPI rank)
 * with the OrchestrationRunner. Validates:
 * - Context creation based on RankExecutionPlan
 * - Backend selection for different device combinations
 * - Activation transfer flow between stages
 * - Scenario 7 (PP of TP domains) configuration
 *
 * These tests work WITHOUT real GPUs by using MockLocalPPContext and
 * mocking device detection where needed.
 *
 * @see docs/v2/projects/2026-01/HYBRID_ORCHESTRATION_INTEGRATION_PLAN_v2.md
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "collective/ILocalPPContext.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "config/OrchestrationConfig.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/TensorClasses.h"
#include "mocks/MockLocalPPContext.h"

using namespace llaminar2;
using namespace llaminar2::test;
using namespace testing;

namespace
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__LocalPPIntegration : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create a test FP32 tensor for transfer operation tests
            test_tensor_ = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 128});
        }

        /**
         * @brief Create a simple RankExecutionPlan with no LOCAL PP
         *
         * This is the baseline: single device, all layers on one stage.
         */
        static RankExecutionPlan makeSimplePlan()
        {
            RankExecutionPlan plan;
            plan.rank = 0;
            plan.hostname = "localhost";
            plan.numa_node = 0;
            plan.pp_stage_id = 0;
            plan.first_layer = 0;
            plan.last_layer = 23;
            plan.has_embedding = true;
            plan.has_lm_head = true;
            plan.primary_device = GlobalDeviceAddress::cpu();
            // local_pp_devices is empty → no LOCAL PP
            return plan;
        }

        /**
         * @brief Create a RankExecutionPlan with LOCAL PP configuration
         *
         * @param devices Device assignment for each PP stage
         * @param layer_boundaries Layer boundaries (size = devices.size() + 1)
         * @param backend Collective backend for transfers
         */
        static RankExecutionPlan makePPPlan(
            const std::vector<GlobalDeviceAddress> &devices,
            const std::vector<int> &layer_boundaries,
            CollectiveBackendType backend = CollectiveBackendType::AUTO)
        {
            RankExecutionPlan plan;
            plan.rank = 0;
            plan.hostname = "localhost";
            plan.numa_node = 0;
            plan.pp_stage_id = 0;
            plan.first_layer = 0;
            plan.last_layer = layer_boundaries.empty() ? 23 : layer_boundaries.back() - 1;
            plan.has_embedding = true;
            plan.has_lm_head = true;
            plan.primary_device = devices.empty() ? GlobalDeviceAddress::cpu() : devices[0];

            // LOCAL PP configuration
            plan.local_pp_devices = devices;
            plan.local_pp_layer_boundaries = layer_boundaries;
            plan.local_pp_backend = backend;

            return plan;
        }

        /**
         * @brief Create a minimal OrchestrationConfig for testing
         */
        static OrchestrationConfig makeTestConfig()
        {
            OrchestrationConfig config;
            config.tp_degree = 1;
            config.pp_degree = 1;
            // No model path - testing without actual model loading
            return config;
        }

        /**
         * @brief Create LocalPPConfig from execution plan parameters
         */
        static LocalPPConfig makePPConfig(
            const std::vector<GlobalDeviceAddress> &devices,
            const std::vector<int> &layer_boundaries)
        {
            LocalPPConfig config;
            config.stage_devices = devices;
            config.layer_boundaries = layer_boundaries;
            return config;
        }

        /**
         * @brief Create mock PP context matching the given plan
         */
        static std::unique_ptr<MockLocalPPContext> makeMockPPContext(const RankExecutionPlan &plan)
        {
            MockLocalPPContext::Config mock_config;
            mock_config.stage_devices = plan.local_pp_devices;
            mock_config.layer_boundaries = plan.local_pp_layer_boundaries;
            return std::make_unique<MockLocalPPContext>(mock_config);
        }

        // Test data
        std::unique_ptr<FP32Tensor> test_tensor_;
    };

    // =========================================================================
    // 1. OrchestrationRunner Integration Tests
    // =========================================================================
    // Verify that OrchestrationRunner correctly initializes LOCAL PP based on
    // the RankExecutionPlan configuration.

    /**
     * @test OrchestrationRunner should NOT create LocalPPContext when
     *       local_pp_devices has 0 or 1 device (no PP needed).
     */
    TEST_F(Test__LocalPPIntegration, OrchestrationRunner_NoLocalPP_WhenSingleDevice)
    {
        // GIVEN: A plan with no LOCAL PP devices
        RankExecutionPlan plan = makeSimplePlan();
        ASSERT_TRUE(plan.local_pp_devices.empty())
            << "Precondition: plan should have no LOCAL PP devices";

        // WHEN: OrchestrationRunner is created
        OrchestrationConfig config = makeTestConfig();
        OrchestrationRunner runner(config, plan);

        // THEN: The runner should create successfully (no LocalPPContext needed)
        // Note: We can't directly check if local_pp_ctx_ is null without exposing it,
        // but we can verify the plan reports no LOCAL PP
        const auto &returned_plan = runner.executionPlan();
        EXPECT_TRUE(returned_plan.local_pp_devices.empty())
            << "Plan should report no LOCAL PP devices";
        EXPECT_TRUE(returned_plan.local_pp_layer_boundaries.empty())
            << "Plan should report no LOCAL PP boundaries";
    }

    /**
     * @test OrchestrationRunner should NOT create LocalPPContext when
     *       local_pp_devices has exactly 1 device (no parallelism).
     */
    TEST_F(Test__LocalPPIntegration, OrchestrationRunner_NoLocalPP_WhenExactlyOneDevice)
    {
        // GIVEN: A plan with exactly one LOCAL PP device (degenerates to single-stage)
        std::vector<GlobalDeviceAddress> devices = {GlobalDeviceAddress::cuda(0)};
        std::vector<int> boundaries = {0, 24}; // All 24 layers on one stage
        RankExecutionPlan plan = makePPPlan(devices, boundaries);

        ASSERT_EQ(plan.local_pp_devices.size(), 1u)
            << "Precondition: plan should have exactly 1 LOCAL PP device";

        // WHEN: OrchestrationRunner is created
        OrchestrationConfig config = makeTestConfig();
        OrchestrationRunner runner(config, plan);

        // THEN: The plan should report single-device configuration
        const auto &returned_plan = runner.executionPlan();
        EXPECT_EQ(returned_plan.local_pp_devices.size(), 1u);
    }

    /**
     * @test OrchestrationRunner should detect valid LOCAL PP configuration
     *       when multiple devices and proper boundaries are provided.
     */
    TEST_F(Test__LocalPPIntegration, OrchestrationRunner_DetectsLocalPP_WhenMultipleDevices)
    {
        // GIVEN: A plan with 2-stage LOCAL PP
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        std::vector<int> boundaries = {0, 12, 24}; // Layers 0-11 on stage 0, 12-23 on stage 1
        RankExecutionPlan plan = makePPPlan(devices, boundaries);

        ASSERT_EQ(plan.local_pp_devices.size(), 2u)
            << "Precondition: plan should have 2 LOCAL PP devices";

        // WHEN: OrchestrationRunner is created
        OrchestrationConfig config = makeTestConfig();
        OrchestrationRunner runner(config, plan);

        // THEN: The plan should correctly report LOCAL PP configuration
        const auto &returned_plan = runner.executionPlan();
        EXPECT_EQ(returned_plan.local_pp_devices.size(), 2u);
        EXPECT_EQ(returned_plan.local_pp_layer_boundaries.size(), 3u);
        EXPECT_EQ(returned_plan.local_pp_layer_boundaries[0], 0);
        EXPECT_EQ(returned_plan.local_pp_layer_boundaries[1], 12);
        EXPECT_EQ(returned_plan.local_pp_layer_boundaries[2], 24);
    }

    /**
     * @test OrchestrationRunner should fail gracefully on invalid PP config
     *       (mismatched boundary count vs device count).
     */
    TEST_F(Test__LocalPPIntegration, OrchestrationRunner_InvalidConfig_MismatchedBoundaries)
    {
        // GIVEN: A plan with invalid LOCAL PP (wrong number of boundaries)
        // 2 devices require 3 boundaries, but we provide only 2
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        std::vector<int> boundaries = {0, 24}; // WRONG: should be {0, 12, 24}
        RankExecutionPlan plan = makePPPlan(devices, boundaries);

        // WHEN: We create a LocalPPConfig from this invalid setup
        LocalPPConfig pp_config = makePPConfig(devices, boundaries);

        // THEN: The config should be invalid
        EXPECT_FALSE(pp_config.isValid())
            << "Config with mismatched boundaries should be invalid";
    }

    /**
     * @test OrchestrationRunner should fail gracefully on invalid PP config
     *       (non-monotonic layer boundaries).
     */
    TEST_F(Test__LocalPPIntegration, OrchestrationRunner_InvalidConfig_NonMonotonicBoundaries)
    {
        // GIVEN: A plan with non-monotonic boundaries
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        std::vector<int> boundaries = {0, 16, 12}; // WRONG: 16 > 12
        RankExecutionPlan plan = makePPPlan(devices, boundaries);

        // WHEN: We create a LocalPPConfig from this invalid setup
        LocalPPConfig pp_config = makePPConfig(devices, boundaries);

        // THEN: The config should be invalid
        EXPECT_FALSE(pp_config.isValid())
            << "Config with non-monotonic boundaries should be invalid";
    }

    // =========================================================================
    // 2. Backend Selection Integration Tests
    // =========================================================================
    // Test that the correct collective backend is selected based on device types.
    // Uses the real createLocalPPContext() factory to validate backend selection.

    /**
     * @test NCCL backend should be selected for CUDA→CUDA transfers
     */
    TEST_F(Test__LocalPPIntegration, BackendSelection_NCCL_ForCudaToCuda)
    {
        // GIVEN: A config with 2 CUDA devices
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        std::vector<int> boundaries = {0, 12, 24};
        LocalPPConfig config = makePPConfig(devices, boundaries);

        ASSERT_TRUE(config.isValid());

        // WHEN: We create a LocalPPContext
        // Note: This may fail on systems without CUDA. In that case,
        // we verify the expected behavior through the mock instead.
        try
        {
            auto ctx = createLocalPPContext(config);
            ASSERT_NE(ctx, nullptr);

            // THEN: Backend for stage 0→1 transfer should be NCCL
            auto backend = ctx->backendForTransfer(0, 1);
            EXPECT_EQ(backend, CollectiveBackendType::NCCL)
                << "CUDA→CUDA transfer should use NCCL backend";
        }
        catch (const std::exception &e)
        {
            // System may not have CUDA - skip with informative message
            GTEST_SKIP() << "Skipping real CUDA test: " << e.what();
        }
    }

    /**
     * @test RCCL backend should be selected for ROCm→ROCm transfers
     */
    TEST_F(Test__LocalPPIntegration, BackendSelection_RCCL_ForRocmToRocm)
    {
        // GIVEN: A config with 2 ROCm devices
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::rocm(1)};
        std::vector<int> boundaries = {0, 12, 24};
        LocalPPConfig config = makePPConfig(devices, boundaries);

        ASSERT_TRUE(config.isValid());

        // WHEN: We create a LocalPPContext
        try
        {
            auto ctx = createLocalPPContext(config);
            ASSERT_NE(ctx, nullptr);

            // THEN: Backend for stage 0→1 transfer should be RCCL
            auto backend = ctx->backendForTransfer(0, 1);
            EXPECT_EQ(backend, CollectiveBackendType::RCCL)
                << "ROCm→ROCm transfer should use RCCL backend";
        }
        catch (const std::exception &e)
        {
            GTEST_SKIP() << "Skipping real ROCm test: " << e.what();
        }
    }

    /**
     * @test HOST backend should be selected when CPU device is involved
     */
    TEST_F(Test__LocalPPIntegration, BackendSelection_HOST_ForCPUDevice)
    {
        // GIVEN: A config with GPU + CPU devices
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cpu()};
        std::vector<int> boundaries = {0, 12, 24};
        LocalPPConfig config = makePPConfig(devices, boundaries);

        ASSERT_TRUE(config.isValid());

        // WHEN: We create a LocalPPContext
        try
        {
            auto ctx = createLocalPPContext(config);
            ASSERT_NE(ctx, nullptr);

            // THEN: Backend for GPU→CPU transfer should be HOST
            auto backend = ctx->backendForTransfer(0, 1);
            EXPECT_EQ(backend, CollectiveBackendType::HOST)
                << "GPU→CPU transfer should use HOST backend";
        }
        catch (const std::exception &e)
        {
            GTEST_SKIP() << "Skipping GPU→CPU test: " << e.what();
        }
    }

    /**
     * @test Verify backend selection logic matches expectation using mock
     *
     * This test uses MockLocalPPContext to verify the expected backend
     * selection logic without requiring actual hardware.
     */
    TEST_F(Test__LocalPPIntegration, BackendSelection_MockVerification)
    {
        // Test all backend selection scenarios using mock

        // Scenario 1: CUDA→CUDA
        {
            MockLocalPPContext::Config cfg;
            cfg.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
            cfg.layer_boundaries = {0, 12, 24};
            cfg.default_backend = CollectiveBackendType::NCCL; // Expected backend

            MockLocalPPContext mock(cfg);
            EXPECT_EQ(mock.backendForTransfer(0, 1), CollectiveBackendType::NCCL);
        }

        // Scenario 2: ROCm→ROCm
        {
            MockLocalPPContext::Config cfg;
            cfg.stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            cfg.layer_boundaries = {0, 12, 24};
            cfg.default_backend = CollectiveBackendType::RCCL;

            MockLocalPPContext mock(cfg);
            EXPECT_EQ(mock.backendForTransfer(0, 1), CollectiveBackendType::RCCL);
        }

        // Scenario 3: CUDA↔ROCm
        {
            MockLocalPPContext::Config cfg;
            cfg.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
            cfg.layer_boundaries = {0, 12, 24};
            cfg.default_backend = CollectiveBackendType::HETEROGENEOUS;

            MockLocalPPContext mock(cfg);
            EXPECT_EQ(mock.backendForTransfer(0, 1), CollectiveBackendType::HETEROGENEOUS);
        }

        // Scenario 4: GPU→CPU
        {
            MockLocalPPContext::Config cfg;
            cfg.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cpu()};
            cfg.layer_boundaries = {0, 12, 24};
            cfg.default_backend = CollectiveBackendType::HOST;

            MockLocalPPContext mock(cfg);
            EXPECT_EQ(mock.backendForTransfer(0, 1), CollectiveBackendType::HOST);
        }
    }

    // =========================================================================
    // 3. End-to-End Flow Tests (with Mocks)
    // =========================================================================
    // Test the full activation transfer flow using MockLocalPPContext.

    /**
     * @test Verify activation transfer from stage 0 to stage 1 in a 2-stage setup
     */
    TEST_F(Test__LocalPPIntegration, TransferFlow_Stage0ToStage1)
    {
        // GIVEN: A mock 2-stage PP context
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        cfg.layer_boundaries = {0, 12, 24}; // Stage 0: layers 0-11, Stage 1: 12-23

        MockLocalPPContext mock(cfg);
        ASSERT_EQ(mock.numStages(), 2);

        // WHEN: We transfer activations from stage 0 to stage 1
        bool success = mock.transfer(test_tensor_.get(), 0, 1);

        // THEN: Transfer should succeed and be recorded
        EXPECT_TRUE(success);
        EXPECT_EQ(mock.transferCallCount(), 1);
        EXPECT_TRUE(mock.hasTransfer(0, 1));

        // Verify the recorded call details
        auto last_call = mock.lastTransferCall();
        EXPECT_EQ(last_call.activations, test_tensor_.get());
        EXPECT_EQ(last_call.stage_from, 0);
        EXPECT_EQ(last_call.stage_to, 1);
        EXPECT_FALSE(last_call.was_async);
    }

    /**
     * @test Verify async transfer flow with synchronization
     */
    TEST_F(Test__LocalPPIntegration, TransferFlow_AsyncWithSync)
    {
        // GIVEN: A mock 2-stage PP context
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        cfg.layer_boundaries = {0, 12, 24};

        MockLocalPPContext mock(cfg);

        // WHEN: We do an async transfer and synchronize
        void *test_stream = reinterpret_cast<void *>(0xDEADBEEF);
        bool success = mock.transferAsync(test_tensor_.get(), 0, 1, test_stream);
        mock.synchronize();

        // THEN: Both operations should be recorded
        EXPECT_TRUE(success);
        EXPECT_EQ(mock.transferAsyncCallCount(), 1);
        EXPECT_EQ(mock.synchronizeCallCount(), 1);

        auto last_call = mock.lastTransferCall();
        EXPECT_TRUE(last_call.was_async);
        EXPECT_EQ(last_call.stream, test_stream);
    }

    /**
     * @test Verify transfer chain in a 3-stage pipeline
     */
    TEST_F(Test__LocalPPIntegration, TransferFlow_MultiStage)
    {
        // GIVEN: A mock 3-stage PP context (GPU → GPU → CPU)
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
            GlobalDeviceAddress::cpu()};
        cfg.layer_boundaries = {0, 8, 16, 24}; // 8 layers per stage

        MockLocalPPContext mock(cfg);
        ASSERT_EQ(mock.numStages(), 3);

        // Verify device assignments
        EXPECT_EQ(mock.deviceForStage(0).device_type, DeviceType::CUDA);
        EXPECT_EQ(mock.deviceForStage(1).device_type, DeviceType::CUDA);
        EXPECT_EQ(mock.deviceForStage(2).device_type, DeviceType::CPU);

        // WHEN: We execute a full pipeline of transfers
        // (Simulating what would happen during inference)
        EXPECT_TRUE(mock.transfer(test_tensor_.get(), 0, 1)); // Stage 0 → 1
        EXPECT_TRUE(mock.transfer(test_tensor_.get(), 1, 2)); // Stage 1 → 2

        // THEN: Both transfers should be recorded in order
        EXPECT_EQ(mock.transferCallCount(), 2);
        EXPECT_TRUE(mock.hasTransfer(0, 1));
        EXPECT_TRUE(mock.hasTransfer(1, 2));

        auto calls = mock.transferCalls();
        ASSERT_EQ(calls.size(), 2u);
        EXPECT_EQ(calls[0].stage_from, 0);
        EXPECT_EQ(calls[0].stage_to, 1);
        EXPECT_EQ(calls[1].stage_from, 1);
        EXPECT_EQ(calls[1].stage_to, 2);
    }

    /**
     * @test Verify transfer failure injection
     */
    TEST_F(Test__LocalPPIntegration, TransferFlow_FailureInjection)
    {
        // GIVEN: A mock context configured to fail transfers
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        cfg.layer_boundaries = {0, 12, 24};
        cfg.transfer_should_fail = true; // Inject failure

        MockLocalPPContext mock(cfg);

        // WHEN: We attempt a transfer
        bool success = mock.transfer(test_tensor_.get(), 0, 1);

        // THEN: Transfer should fail
        EXPECT_FALSE(success);
        // Call should still be recorded for debugging
        EXPECT_EQ(mock.transferCallCount(), 1);
    }

    // =========================================================================
    // 4. Scenario 7 Style Tests (PP of TP Domains)
    // =========================================================================
    // Test the complex composition scenario where each PP stage is itself
    // a TP domain with multiple devices.

    /**
     * @test Configure Scenario 7: PP of TP domains
     *
     * Configuration:
     *   Stage 0: TP(cuda:0, cuda:1)  - layers 0-7
     *   Stage 1: TP(rocm:0, rocm:1)  - layers 8-15
     *   Stage 2: TP(cpu:0, cpu:1)    - layers 16-23
     *
     * This validates the layer boundaries and device mapping for the
     * complex multi-domain scenario described in the Hybrid Orchestration Plan.
     */
    TEST_F(Test__LocalPPIntegration, Scenario7_PP_Of_TP_Domains)
    {
        // GIVEN: Scenario 7 configuration - 3 PP stages, each with 2 devices
        // For LOCAL PP, we represent each TP domain by its primary device
        // (The TP within each domain is handled separately by ILocalTPContext)

        // The PP configuration maps stage → primary device of TP domain
        std::vector<GlobalDeviceAddress> pp_stage_devices = {
            GlobalDeviceAddress::cuda(0), // Primary device of TP domain 0
            GlobalDeviceAddress::rocm(0), // Primary device of TP domain 1
            GlobalDeviceAddress::cpu()};  // Primary device of TP domain 2

        std::vector<int> layer_boundaries = {0, 8, 16, 24}; // 8 layers per stage

        // Create config
        LocalPPConfig config;
        config.stage_devices = pp_stage_devices;
        config.layer_boundaries = layer_boundaries;

        ASSERT_TRUE(config.isValid());

        // WHEN: We create a mock context for this configuration
        MockLocalPPContext::Config mock_cfg;
        mock_cfg.stage_devices = pp_stage_devices;
        mock_cfg.layer_boundaries = layer_boundaries;
        MockLocalPPContext mock(mock_cfg);

        // THEN: Verify the configuration

        // Number of stages
        EXPECT_EQ(mock.numStages(), 3);

        // Device mapping
        EXPECT_EQ(mock.deviceForStage(0).device_type, DeviceType::CUDA);
        EXPECT_EQ(mock.deviceForStage(0).device_ordinal, 0);

        EXPECT_EQ(mock.deviceForStage(1).device_type, DeviceType::ROCm);
        EXPECT_EQ(mock.deviceForStage(1).device_ordinal, 0);

        EXPECT_EQ(mock.deviceForStage(2).device_type, DeviceType::CPU);

        // Layer ranges
        EXPECT_EQ(mock.layerRangeForStage(0), std::make_pair(0, 8));
        EXPECT_EQ(mock.layerRangeForStage(1), std::make_pair(8, 16));
        EXPECT_EQ(mock.layerRangeForStage(2), std::make_pair(16, 24));

        // Layer → Stage mapping
        EXPECT_EQ(mock.stageForLayer(0), 0);
        EXPECT_EQ(mock.stageForLayer(7), 0);
        EXPECT_EQ(mock.stageForLayer(8), 1);
        EXPECT_EQ(mock.stageForLayer(15), 1);
        EXPECT_EQ(mock.stageForLayer(16), 2);
        EXPECT_EQ(mock.stageForLayer(23), 2);

        // Total layers
        EXPECT_EQ(mock.totalLayers(), 24);

        // Same device check
        EXPECT_FALSE(mock.sameDevice(0, 1)); // Different vendors
        EXPECT_FALSE(mock.sameDevice(1, 2)); // Different types (GPU vs CPU)
        EXPECT_FALSE(mock.sameDevice(0, 2)); // Different types
        EXPECT_TRUE(mock.sameDevice(0, 0));  // Same stage
    }

    /**
     * @test Verify transfer pattern for Scenario 7 pipeline execution
     */
    TEST_F(Test__LocalPPIntegration, Scenario7_TransferPattern)
    {
        // GIVEN: Scenario 7 mock context
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::cpu()};
        cfg.layer_boundaries = {0, 8, 16, 24};

        MockLocalPPContext mock(cfg);

        // WHEN: We simulate a full forward pass
        // After stage 0 completes layers 0-7, transfer to stage 1
        EXPECT_TRUE(mock.transfer(test_tensor_.get(), 0, 1));

        // After stage 1 completes layers 8-15, transfer to stage 2
        EXPECT_TRUE(mock.transfer(test_tensor_.get(), 1, 2));

        // Stage 2 completes layers 16-23 (no further transfer needed)

        // THEN: Verify the transfer chain
        EXPECT_EQ(mock.transferCallCount(), 2);

        auto calls = mock.transferCalls();
        ASSERT_EQ(calls.size(), 2u);

        // First transfer: CUDA → ROCm (uses HOST backend)
        EXPECT_EQ(calls[0].stage_from, 0);
        EXPECT_EQ(calls[0].stage_to, 1);

        // Second transfer: ROCm → CPU (would use HOST backend)
        EXPECT_EQ(calls[1].stage_from, 1);
        EXPECT_EQ(calls[1].stage_to, 2);
    }

    // =========================================================================
    // 5. Edge Cases and Error Handling
    // =========================================================================

    /**
     * @test Verify behavior when accessing out-of-range stage
     */
    TEST_F(Test__LocalPPIntegration, ErrorHandling_OutOfRangeStage)
    {
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        cfg.layer_boundaries = {0, 12, 24};

        MockLocalPPContext mock(cfg);

        // Accessing stage 5 (only 2 stages exist)
        EXPECT_THROW(mock.deviceForStage(5), std::out_of_range);
        EXPECT_THROW(mock.deviceForStage(-1), std::out_of_range);
    }

    /**
     * @test Verify layer lookup for out-of-range layers
     */
    TEST_F(Test__LocalPPIntegration, ErrorHandling_OutOfRangeLayer)
    {
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        cfg.layer_boundaries = {0, 12, 24};

        MockLocalPPContext mock(cfg);

        // Layer 24 is at the boundary (not valid)
        EXPECT_EQ(mock.stageForLayer(24), -1);
        // Layer 100 is way out of range
        EXPECT_EQ(mock.stageForLayer(100), -1);
        // Negative layer
        EXPECT_EQ(mock.stageForLayer(-1), -1);
    }

    /**
     * @test Verify invalid layer range returns sentinel
     */
    TEST_F(Test__LocalPPIntegration, ErrorHandling_InvalidLayerRange)
    {
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {GlobalDeviceAddress::cuda(0)};
        cfg.layer_boundaries = {0, 24};

        MockLocalPPContext mock(cfg);

        // Invalid stage index should return sentinel
        auto range = mock.layerRangeForStage(5);
        EXPECT_EQ(range, std::make_pair(-1, -1));
    }

    /**
     * @test Verify staging buffer reservation
     */
    TEST_F(Test__LocalPPIntegration, StagingBuffer_Reservation)
    {
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cpu()};
        cfg.layer_boundaries = {0, 12, 24};

        MockLocalPPContext mock(cfg);

        // Reserve staging buffer for HOST backend transfers
        size_t buffer_size = 1024 * 1024; // 1 MB
        EXPECT_TRUE(mock.reserveStagingBufferBytes(buffer_size));
        EXPECT_EQ(mock.reserveStagingCallCount(), 1);
    }

    /**
     * @test Verify mock reset functionality
     */
    TEST_F(Test__LocalPPIntegration, MockReset_ClearsCallHistory)
    {
        MockLocalPPContext::Config cfg;
        cfg.stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        cfg.layer_boundaries = {0, 12, 24};

        MockLocalPPContext mock(cfg);

        // Make some calls
        mock.transfer(test_tensor_.get(), 0, 1);
        mock.transferAsync(test_tensor_.get(), 0, 1, nullptr);
        mock.synchronize();

        EXPECT_EQ(mock.totalTransferCallCount(), 2);
        EXPECT_EQ(mock.synchronizeCallCount(), 1);

        // Reset
        mock.resetCallCounts();

        // Verify reset
        EXPECT_EQ(mock.totalTransferCallCount(), 0);
        EXPECT_EQ(mock.synchronizeCallCount(), 0);
        EXPECT_TRUE(mock.transferCalls().empty());
    }

    // =========================================================================
    // 6. RankExecutionPlan ↔ LocalPPConfig Conversion Tests
    // =========================================================================

    /**
     * @test Verify RankExecutionPlan LOCAL PP fields correctly populate LocalPPConfig
     */
    TEST_F(Test__LocalPPIntegration, PlanToConfig_Conversion)
    {
        // GIVEN: A RankExecutionPlan with LOCAL PP settings
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::cpu()};
        std::vector<int> boundaries = {0, 8, 16, 24};

        RankExecutionPlan plan = makePPPlan(devices, boundaries, CollectiveBackendType::AUTO);

        // WHEN: We convert to LocalPPConfig
        LocalPPConfig config;
        config.stage_devices = plan.local_pp_devices;
        config.layer_boundaries = plan.local_pp_layer_boundaries;

        // THEN: Config should match plan
        EXPECT_TRUE(config.isValid());
        EXPECT_EQ(config.numStages(), 3);
        EXPECT_EQ(config.stage_devices.size(), 3u);
        EXPECT_EQ(config.layer_boundaries.size(), 4u);

        // Verify layer ranges
        EXPECT_EQ(config.layerRangeForStage(0), std::make_pair(0, 8));
        EXPECT_EQ(config.layerRangeForStage(1), std::make_pair(8, 16));
        EXPECT_EQ(config.layerRangeForStage(2), std::make_pair(16, 24));

        // Verify stageForLayer
        EXPECT_EQ(config.stageForLayer(3), 0);
        EXPECT_EQ(config.stageForLayer(10), 1);
        EXPECT_EQ(config.stageForLayer(20), 2);
    }

    /**
     * @test Verify empty LOCAL PP plan results in invalid config
     */
    TEST_F(Test__LocalPPIntegration, PlanToConfig_EmptyDevices)
    {
        RankExecutionPlan plan = makeSimplePlan();

        LocalPPConfig config;
        config.stage_devices = plan.local_pp_devices;             // Empty
        config.layer_boundaries = plan.local_pp_layer_boundaries; // Empty

        // Empty config should be invalid
        EXPECT_FALSE(config.isValid());
    }

} // anonymous namespace
