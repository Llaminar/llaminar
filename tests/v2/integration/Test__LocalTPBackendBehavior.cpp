/**
 * @file Test__LocalTPBackendBehavior.cpp
 * @brief Integration tests for LocalTPContext backend behavior
 * @author David Sanftenberg
 * @date January 2026
 *
 * These tests were migrated from unit tests because they require actual GPU hardware
 * to properly test backend auto-detection, barrier synchronization, and BAR management.
 *
 * Test groups:
 * 1. Backend auto-detection - Verify AUTO backend selects correct backend type
 * 2. PCIeBAR barrier synchronization - Verify multi-threaded barrier rendezvous
 * 3. BAR-backed tensor management - Verify zero-copy tensor registry
 *
 * Hardware requirements vary per test group:
 * - ROCm backend tests require 2+ ROCm GPUs
 * - PCIeBAR tests require 1 CUDA + 1 ROCm GPU
 * - HETEROGENEOUS tests require multiple CUDA and/or ROCm GPUs
 */

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <condition_variable>
#include <mutex>

#include "collective/LocalTPContext.h"
#include "collective/ICollectiveBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalTPBackendBehavior : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get device counts from DeviceManager
#ifdef HAVE_CUDA
        auto *cuda_backend = getCUDABackend();
        cuda_count_ = (cuda_backend != nullptr) ? cuda_backend->deviceCount() : 0;
#else
        cuda_count_ = 0;
#endif

#ifdef HAVE_ROCM
        auto *rocm_backend = getROCmBackend();
        rocm_count_ = (rocm_backend != nullptr) ? rocm_backend->deviceCount() : 0;
#else
        rocm_count_ = 0;
#endif

        std::cout << "Test__LocalTPBackendBehavior: Found " << cuda_count_
                  << " CUDA GPU(s), " << rocm_count_ << " ROCm GPU(s)" << std::endl;
    }

    void TearDown() override
    {
        // Synchronize all GPUs
#ifdef HAVE_CUDA
        auto *cuda_backend = getCUDABackend();
        if (cuda_backend != nullptr)
        {
            for (int i = 0; i < cuda_count_; ++i)
            {
                cuda_backend->synchronize(i);
            }
        }
#endif

#ifdef HAVE_ROCM
        auto *rocm_backend = getROCmBackend();
        if (rocm_backend != nullptr)
        {
            for (int i = 0; i < rocm_count_; ++i)
            {
                rocm_backend->synchronize(i);
            }
        }
#endif
    }

    // Skip macros - GTEST_SKIP must be used directly in test to properly return
#define SKIP_IF_LESS_THAN_2_ROCM()                                          \
    do                                                                      \
    {                                                                       \
        if (rocm_count_ < 2)                                                \
        {                                                                   \
            GTEST_SKIP() << "Requires 2+ ROCm GPUs, found " << rocm_count_; \
        }                                                                   \
    } while (0)

#define SKIP_IF_NO_HETEROGENEOUS()                               \
    do                                                           \
    {                                                            \
        if (cuda_count_ == 0 || rocm_count_ == 0)                \
        {                                                        \
            GTEST_SKIP() << "Requires 1+ CUDA and 1+ ROCm GPUs"; \
        }                                                        \
    } while (0)

#define SKIP_IF_NO_HETEROGENEOUS_4WAY()                          \
    do                                                           \
    {                                                            \
        if (cuda_count_ < 2 || rocm_count_ < 2)                  \
        {                                                        \
            GTEST_SKIP() << "Requires 2+ CUDA and 2+ ROCm GPUs"; \
        }                                                        \
    } while (0)

    int cuda_count_ = 0;
    int rocm_count_ = 0;
};

// =============================================================================
// Backend Auto-Detection Tests
// =============================================================================
// These tests verify that AUTO backend selection correctly chooses the
// appropriate backend based on device configuration.

/**
 * @test AUTO backend with all ROCm devices selects RCCL
 */
TEST_F(Test__LocalTPBackendBehavior, AutoBackend_AllRocm_SelectsRCCL)
{
    SKIP_IF_LESS_THAN_2_ROCM();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::RCCL)
        << "AUTO backend should select RCCL for all-ROCm configuration";
}

/**
 * @test AUTO backend with mixed GPU types (1 CUDA + 1 ROCm) selects PCIeBAR
 */
TEST_F(Test__LocalTPBackendBehavior, AutoBackend_MixedGPUs_SelectsPCIeBAR)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::PCIE_BAR)
        << "AUTO backend should select PCIeBAR for 1+1 mixed GPU configuration";
}

/**
 * @test AUTO backend with 1 CUDA + 2 ROCm selects HETEROGENEOUS
 */
TEST_F(Test__LocalTPBackendBehavior, AutoBackend_1Cuda2Rocm_SelectsHeterogeneous)
{
    if (cuda_count_ < 1 || rocm_count_ < 2)
    {
        GTEST_SKIP() << "Requires 1+ CUDA and 2+ ROCm GPUs";
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HETEROGENEOUS)
        << "AUTO backend should select HETEROGENEOUS for >2 mixed GPU configuration";
}

/**
 * @test AUTO backend with 2 CUDA + 1 ROCm selects HETEROGENEOUS
 */
TEST_F(Test__LocalTPBackendBehavior, AutoBackend_2Cuda1Rocm_SelectsHeterogeneous)
{
    if (cuda_count_ < 2 || rocm_count_ < 1)
    {
        GTEST_SKIP() << "Requires 2+ CUDA and 1+ ROCm GPUs";
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HETEROGENEOUS)
        << "AUTO backend should select HETEROGENEOUS for >2 mixed GPU configuration";
}

/**
 * @test AUTO backend with 2 CUDA + 2 ROCm selects HETEROGENEOUS
 */
TEST_F(Test__LocalTPBackendBehavior, AutoBackend_2Cuda2Rocm_SelectsHeterogeneous)
{
    SKIP_IF_NO_HETEROGENEOUS_4WAY();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1),
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HETEROGENEOUS)
        << "AUTO backend should select HETEROGENEOUS for >2 mixed GPU configuration";
}

// =============================================================================
// PCIeBAR Barrier Synchronization Tests
// =============================================================================
// These tests verify the multi-threaded barrier mechanism used by PCIeBAR
// backend for heterogeneous CUDA+ROCm allreduce operations.

/**
 * @test PCIeBAR barrier requires both devices to participate before proceeding
 *
 * This test verifies that when multiple device threads call allreduce,
 * they properly synchronize via the barrier before data transfer.
 */
TEST_F(Test__LocalTPBackendBehavior, PCIeBar_BarrierRendezvous)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::PCIE_BAR);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->backend(), CollectiveBackendType::PCIE_BAR)
        << "This test requires PCIeBAR backend";

    std::atomic<int> arrived_count{0};
    std::atomic<int> completed_count{0};

    auto tensor1 = TestTensorFactory::createFP32({1024});
    auto tensor2 = TestTensorFactory::createFP32({1024});

    for (size_t i = 0; i < tensor1->numel(); ++i)
    {
        tensor1->mutable_data()[i] = 1.0f;
        tensor2->mutable_data()[i] = 2.0f;
    }

    // Thread 1: CUDA device
    std::thread cuda_thread([&]()
                            {
        arrived_count++;
        bool result = ctx->allreduce(tensor1.get());
        completed_count++;
        EXPECT_TRUE(result); });

    // Thread 2: ROCm device (delayed start to test waiting)
    std::thread rocm_thread([&]()
                            {
        // Delay to ensure CUDA thread arrives at barrier first
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        arrived_count++;
        bool result = ctx->allreduce(tensor2.get());
        completed_count++;
        EXPECT_TRUE(result); });

    cuda_thread.join();
    rocm_thread.join();

    EXPECT_EQ(arrived_count.load(), 2);
    EXPECT_EQ(completed_count.load(), 2);
}

/**
 * @test PCIeBAR barrier can be reused across multiple allreduce cycles
 */
TEST_F(Test__LocalTPBackendBehavior, PCIeBar_MultipleBarrierCycles)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::PCIE_BAR);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->backend(), CollectiveBackendType::PCIE_BAR);

    constexpr int NUM_CYCLES = 5;
    std::atomic<int> completed_cycles{0};

    auto tensor1 = TestTensorFactory::createFP32({256});
    auto tensor2 = TestTensorFactory::createFP32({256});

    std::thread cuda_thread([&]()
                            {
        for (int cycle = 0; cycle < NUM_CYCLES; ++cycle)
        {
            for (size_t i = 0; i < tensor1->numel(); ++i)
            {
                tensor1->mutable_data()[i] = static_cast<float>(cycle + 1);
            }
            bool result = ctx->allreduce(tensor1.get());
            EXPECT_TRUE(result) << "Cycle " << cycle << " failed on CUDA thread";
        }
        completed_cycles += NUM_CYCLES; });

    std::thread rocm_thread([&]()
                            {
        for (int cycle = 0; cycle < NUM_CYCLES; ++cycle)
        {
            for (size_t i = 0; i < tensor2->numel(); ++i)
            {
                tensor2->mutable_data()[i] = static_cast<float>(cycle + 10);
            }
            bool result = ctx->allreduce(tensor2.get());
            EXPECT_TRUE(result) << "Cycle " << cycle << " failed on ROCm thread";
        }
        completed_cycles += NUM_CYCLES; });

    cuda_thread.join();
    rocm_thread.join();

    EXPECT_EQ(completed_cycles.load(), NUM_CYCLES * 2);
}

/**
 * @test PCIeBAR barrier stress test with many rapid cycles
 */
TEST_F(Test__LocalTPBackendBehavior, PCIeBar_BarrierStressTest)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::PCIE_BAR);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->backend(), CollectiveBackendType::PCIE_BAR);

    constexpr int NUM_ITERATIONS = 100;
    std::atomic<int> success_count{0};
    std::atomic<bool> any_failure{false};

    auto tensor1 = TestTensorFactory::createFP32({64});
    auto tensor2 = TestTensorFactory::createFP32({64});

    std::thread cuda_thread([&]()
                            {
        for (int i = 0; i < NUM_ITERATIONS; ++i)
        {
            tensor1->mutable_data()[0] = static_cast<float>(i);
            if (ctx->allreduce(tensor1.get()))
            {
                success_count++;
            }
            else
            {
                any_failure = true;
            }
        } });

    std::thread rocm_thread([&]()
                            {
        for (int i = 0; i < NUM_ITERATIONS; ++i)
        {
            tensor2->mutable_data()[0] = static_cast<float>(i * 2);
            if (ctx->allreduce(tensor2.get()))
            {
                success_count++;
            }
            else
            {
                any_failure = true;
            }
        } });

    cuda_thread.join();
    rocm_thread.join();

    EXPECT_FALSE(any_failure.load()) << "Some allreduce operations failed";
    EXPECT_EQ(success_count.load(), NUM_ITERATIONS * 2)
        << "Expected " << (NUM_ITERATIONS * 2) << " successful allreduce calls";
}

// =============================================================================
// BAR-Backed Tensor Management Tests
// =============================================================================
// These tests verify the zero-copy tensor registry for PCIeBAR backend.

/**
 * @brief Mock BAR-backed FP32 tensor for testing
 *
 * Creates a tensor that reports isBARBacked() = true by calling
 * initBARBackedDirect() with mock pointers.
 */
class MockBARBackedTensor : public FP32Tensor
{
public:
    explicit MockBARBackedTensor(const std::vector<size_t> &shape)
        : FP32Tensor(shape)
    {
        // Set up mock BAR state with non-null pointers
        initBARBackedDirect(
            reinterpret_cast<void *>(0x1000), // Mock ROCm pointer
            reinterpret_cast<void *>(0x2000), // Mock CUDA pointer
            DeviceId::rocm(0),
            DeviceId::cuda(0),
            numel() * sizeof(float));
    }
};

/**
 * @test Regression: CUDA tensor must not be skipped during BAR registration
 *
 * BUG: registerBARBackedOutput() had an early return that skipped non-BAR-backed
 * tensors. Since CUDA tensors are NOT BAR-backed (only ROCm tensors in BAR memory),
 * this caused CUDA device's tensor to never be registered.
 */
TEST_F(Test__LocalTPBackendBehavior, BAR_CUDATensorMustNotBeSkipped)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::PCIE_BAR);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->backend(), CollectiveBackendType::PCIE_BAR)
        << "This test requires PCIeBAR backend";

    auto *concrete_ctx = dynamic_cast<LocalTPContext *>(ctx.get());
    ASSERT_NE(concrete_ctx, nullptr);

    // Create a regular (non-BAR-backed) FP32 tensor simulating CUDA device output
    auto cuda_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 64});
    ASSERT_FALSE(cuda_tensor->isBARBacked());

    // Registration should succeed without throwing
    EXPECT_NO_THROW(
        concrete_ctx->registerBARBackedOutput("layer0_wo_allreduce",
                                              devices[0], cuda_tensor.get()));

    // Verify registration succeeded
    EXPECT_TRUE(concrete_ctx->hasBARBackedOutputs("layer0_wo_allreduce"));

    auto outputs = concrete_ctx->getBARBackedOutputs("layer0_wo_allreduce");
    ASSERT_EQ(outputs.size(), static_cast<size_t>(concrete_ctx->degree()));
    EXPECT_EQ(outputs[0], cuda_tensor.get());
}

/**
 * @test Regression: Both CUDA and ROCm devices must have registered tensors
 *
 * BUG: Only ROCm tensor was registered, causing allreduce to either:
 * 1. Use same tensor twice (A + A = 2A instead of A + B)
 * 2. Fail completely due to missing tensor
 */
TEST_F(Test__LocalTPBackendBehavior, BAR_BothDevicesMustHaveRegisteredTensors)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::PCIE_BAR);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->backend(), CollectiveBackendType::PCIE_BAR);

    auto *concrete_ctx = dynamic_cast<LocalTPContext *>(ctx.get());
    ASSERT_NE(concrete_ctx, nullptr);

    // CUDA device: regular FP32 tensor (NOT BAR-backed)
    auto cuda_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 64});
    // ROCm device: BAR-backed FP32 tensor
    auto rocm_tensor = std::make_unique<MockBARBackedTensor>(std::vector<size_t>{32, 64});

    ASSERT_FALSE(cuda_tensor->isBARBacked());
    ASSERT_TRUE(rocm_tensor->isBARBacked());

    // Register both - NEITHER should throw
    EXPECT_NO_THROW(
        concrete_ctx->registerBARBackedOutput("layer0_wo_allreduce",
                                              devices[0], cuda_tensor.get()));
    EXPECT_NO_THROW(
        concrete_ctx->registerBARBackedOutput("layer0_wo_allreduce",
                                              devices[1], rocm_tensor.get()));

    auto outputs = concrete_ctx->getBARBackedOutputs("layer0_wo_allreduce");
    ASSERT_EQ(outputs.size(), static_cast<size_t>(concrete_ctx->degree()));

    // CRITICAL: Both tensors must be non-null and DISTINCT
    EXPECT_NE(outputs[0], nullptr);
    EXPECT_NE(outputs[1], nullptr);
    EXPECT_NE(outputs[0], outputs[1]) << "Must be different tensors!";

    EXPECT_EQ(outputs[0], cuda_tensor.get());
    EXPECT_EQ(outputs[1], rocm_tensor.get());
}

/**
 * @test Regression: Zero-copy allreduce requires DISTINCT tensor pointers
 *
 * BUG: Both CUDA and ROCm device threads were passing the same tensor
 * pointer to allreduce, resulting in A + A = 2A instead of A + B.
 */
TEST_F(Test__LocalTPBackendBehavior, BAR_DistinctTensorPointersRequired)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::PCIE_BAR);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->backend(), CollectiveBackendType::PCIE_BAR);

    auto *concrete_ctx = dynamic_cast<LocalTPContext *>(ctx.get());
    ASSERT_NE(concrete_ctx, nullptr);

    // Create two DIFFERENT tensors with DIFFERENT data
    auto tensor_a = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8});
    auto tensor_b = std::make_unique<MockBARBackedTensor>(std::vector<size_t>{4, 8});

    // Fill with distinct values
    for (size_t i = 0; i < tensor_a->numel(); ++i)
    {
        tensor_a->mutable_data()[i] = 1.0f;
        tensor_b->mutable_data()[i] = 3.0f;
    }

    concrete_ctx->registerBARBackedOutput("layer0_wo_allreduce", devices[0], tensor_a.get());
    concrete_ctx->registerBARBackedOutput("layer0_wo_allreduce", devices[1], tensor_b.get());

    auto outputs = concrete_ctx->getBARBackedOutputs("layer0_wo_allreduce");

    // Critical: pointers must be different
    ASSERT_NE(outputs[0], outputs[1])
        << "BUG: Same tensor pointer for both devices would cause A+A=2A!";

    // Verify data is different (1.0 vs 3.0)
    EXPECT_NE(outputs[0]->data()[0], outputs[1]->data()[0])
        << "Tensors must contain different data for correct allreduce";
}
