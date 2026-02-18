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
#include "utils/DebugEnv.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /**
     * @brief RAII helper for temporarily overriding an environment variable in a test.
     *
     * Why this helper exists:
     * - Tests should not leak process-wide env changes into later tests.
     * - `DebugEnv` caches values, so we also reload config on entry/exit.
     */
    class ScopedEnvVar
    {
    public:
        /**
         * @brief Set @p name to @p value for this scope, preserving prior state.
         */
        ScopedEnvVar(const char *name, const char *value)
            : name_(name)
        {
            const char *prev = std::getenv(name_);
            if (prev)
            {
                had_previous_ = true;
                previous_value_ = prev;
            }

            setenv(name_, value, 1);
            mutableDebugEnv().reload();
        }

        /**
         * @brief Restore original env state and reload DebugEnv cache.
         */
        ~ScopedEnvVar()
        {
            if (had_previous_)
            {
                setenv(name_, previous_value_.c_str(), 1);
            }
            else
            {
                unsetenv(name_);
            }
            mutableDebugEnv().reload();
        }

        ScopedEnvVar(const ScopedEnvVar &) = delete;
        ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

    private:
        const char *name_;
        bool had_previous_ = false;
        std::string previous_value_;
    };
} // namespace

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
 * @test LocalTP multi-GPU allreduce fails fast when requested count exceeds tensor size
 *
 * This is a Phase 2 correctness guardrail test. A bad element count can otherwise
 * produce backend-side undefined behavior. We assert that LocalTP validation catches
 * this before entering NCCL collective launch.
 */
TEST_F(Test__LocalTPBackendBehavior, NCCLAllreduce_CountExceedsTensorNumel_FailsFast)
{
    if (cuda_count_ < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found " << cuda_count_;
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    auto tensor0 = TestTensorFactory::createFP32({4, 4});
    auto tensor1 = TestTensorFactory::createFP32({4, 4});

    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::cuda(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::cuda(1)));

    const size_t invalid_count = tensor0->numel() + 8;

    std::atomic<bool> result0{true};
    std::atomic<bool> result1{true};

    std::thread t0([&]()
                   { result0.store(ctx->allreduce(tensor0.get(), "invalid_count_test", invalid_count)); });
    std::thread t1([&]()
                   { result1.store(ctx->allreduce(tensor1.get(), "invalid_count_test", invalid_count)); });

    t0.join();
    t1.join();

    EXPECT_FALSE(result0.load());
    EXPECT_FALSE(result1.load());
}

/**
 * @test LocalTP multi-GPU allreduce fails fast when participant dtypes differ
 *
 * This verifies the Phase 2 dtype-consistency invariant. NCCL allreduce assumes
 * all participants use the same element type; mixed dtypes should be rejected
 * by LocalTP validation before backend launch.
 */
TEST_F(Test__LocalTPBackendBehavior, NCCLAllreduce_DTypeMismatchAcrossParticipants_FailsFast)
{
    if (cuda_count_ < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found " << cuda_count_;
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    // Intentionally use different tensor dtypes across participants.
    auto tensor_fp32 = TestTensorFactory::createFP32({4, 4});
    auto tensor_int32 = std::make_unique<INT32Tensor>(std::vector<size_t>{4, 4});

    ASSERT_TRUE(tensor_fp32->ensureOnDevice(DeviceId::cuda(0)));
    ASSERT_TRUE(tensor_int32->ensureOnDevice(DeviceId::cuda(1)));

    const size_t count = tensor_fp32->numel();

    std::atomic<bool> result0{true};
    std::atomic<bool> result1{true};

    std::thread t0([&]()
                   { result0.store(ctx->allreduce(tensor_fp32.get(), "dtype_mismatch_test", count)); });
    std::thread t1([&]()
                   { result1.store(ctx->allreduce(tensor_int32.get(), "dtype_mismatch_test", count)); });

    t0.join();
    t1.join();

    EXPECT_FALSE(result0.load());
    EXPECT_FALSE(result1.load());
}

/**
 * @test LocalTP NCCL fails fast when GPU graphs are enabled without segmented collectives
 *
 * Phase 3 support policy requires segmented collective mode when running LocalTP
 * NCCL collectives under GPU graph mode.
 */
TEST_F(Test__LocalTPBackendBehavior, NCCLGraphPolicy_GraphsWithoutSegmentedCollectives_FailsFast)
{
    if (cuda_count_ < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found " << cuda_count_;
    }

    ScopedEnvVar graphs_guard("LLAMINAR_GPU_GRAPHS", "1");
    ScopedEnvVar segmented_guard("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0");

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    auto tensor0 = TestTensorFactory::createFP32({8, 8});
    auto tensor1 = TestTensorFactory::createFP32({8, 8});

    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::cuda(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::cuda(1)));

    std::atomic<bool> result0{true};
    std::atomic<bool> result1{true};

    std::thread t0([&]()
                   { result0.store(ctx->allreduce(tensor0.get(), "graph_policy_reject", tensor0->numel())); });
    std::thread t1([&]()
                   { result1.store(ctx->allreduce(tensor1.get(), "graph_policy_reject", tensor1->numel())); });

    t0.join();
    t1.join();

    EXPECT_FALSE(result0.load());
    EXPECT_FALSE(result1.load());
}

/**
 * @test LocalTP NCCL accepts segmented collective mode when GPU graphs are enabled
 *
 * This validates the supported Phase 3 graph policy. In this mode, LocalTP should
 * proceed through normal NCCL execution.
 */
TEST_F(Test__LocalTPBackendBehavior, NCCLGraphPolicy_GraphsWithSegmentedCollectives_AllowsExecution)
{
    if (cuda_count_ < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found " << cuda_count_;
    }

    ScopedEnvVar graphs_guard("LLAMINAR_GPU_GRAPHS", "1");
    ScopedEnvVar segmented_guard("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "1");

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    auto tensor0 = TestTensorFactory::createFP32({8, 8});
    auto tensor1 = TestTensorFactory::createFP32({8, 8});

    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::cuda(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::cuda(1)));

    std::atomic<bool> result0{false};
    std::atomic<bool> result1{false};

    std::thread t0([&]()
                   { result0.store(ctx->allreduce(tensor0.get(), "graph_policy_allow", tensor0->numel())); });
    std::thread t1([&]()
                   { result1.store(ctx->allreduce(tensor1.get(), "graph_policy_allow", tensor1->numel())); });

    t0.join();
    t1.join();

    // NCCL allreduce should succeed under supported graph policy.
    EXPECT_TRUE(result0.load());
    EXPECT_TRUE(result1.load());
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
 * @brief Mock BAR-backed FP32 tensor for testing
 *
 * Creates a tensor that reports isBARBacked() = true by calling
 * initBARBackedDirect() with mock pointers. Used to simulate ROCm device
 * tensors in heterogeneous allreduce tests.
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
 * @test PCIeBAR barrier requires both devices to participate before proceeding
 *
 * This test verifies that when multiple device threads call allreduce,
 * they properly synchronize via the barrier before data transfer.
 *
 * NOTE: This test uses mock BAR tensors which don't have valid GPU pointers,
 * so the actual data transfer may fail. The test validates barrier synchronization
 * behavior (both threads arrive and complete) rather than data correctness.
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

    // Cast to LocalTPContext for tensor registration
    auto *concrete_ctx = dynamic_cast<LocalTPContext *>(ctx.get());
    ASSERT_NE(concrete_ctx, nullptr);

    std::atomic<int> arrived_count{0};
    std::atomic<int> completed_count{0};

    // Create properly typed tensors:
    // - CUDA device: regular FP32Tensor (not BAR-backed)
    // - ROCm device: MockBARBackedTensor (simulates BAR-backed tensor)
    auto tensor1 = std::make_unique<FP32Tensor>(std::vector<size_t>{1024});
    auto tensor2 = std::make_unique<FP32Tensor>(std::vector<size_t>{1024});

    for (size_t i = 0; i < tensor1->numel(); ++i)
    {
        tensor1->mutable_data()[i] = 1.0f;
        tensor2->mutable_data()[i] = 2.0f;
    }

    // Register tensors with the context so PCIeBAR allreduce can find them
    // Using regular FP32Tensors (not MockBARBackedTensor) so the code takes
    // the fallback path that doesn't try to do actual BAR transfers
    const std::string stage_name = "barrier_test_stage";
    concrete_ctx->registerBARBackedOutput(stage_name, devices[0], tensor1.get());
    concrete_ctx->registerBARBackedOutput(stage_name, devices[1], tensor2.get());

    // Thread 1: CUDA device
    std::thread cuda_thread([&]()
                            {
        arrived_count++;
        // The allreduce may fail due to mock tensors not having valid GPU pointers,
        // but we're testing that the barrier mechanism works (threads synchronize)
        ctx->allreduce(tensor1.get(), stage_name, tensor1->numel());
        completed_count++; });

    // Thread 2: ROCm device (delayed start to test waiting)
    std::thread rocm_thread([&]()
                            {
        // Delay to ensure CUDA thread arrives at barrier first
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        arrived_count++;
        ctx->allreduce(tensor2.get(), stage_name, tensor2->numel());
        completed_count++; });

    cuda_thread.join();
    rocm_thread.join();

    // The key assertion: both threads arrived at the barrier and both completed
    // This validates the barrier synchronization mechanism
    EXPECT_EQ(arrived_count.load(), 2);
    EXPECT_EQ(completed_count.load(), 2);
}

/**
 * @test PCIeBAR barrier can be reused across multiple allreduce cycles
 *
 * NOTE: This test uses regular FP32 tensors without actual BAR backing,
 * so the data transfer may fail. The test validates that the barrier
 * mechanism can be reused across multiple cycles.
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

    // Cast to LocalTPContext for tensor registration
    auto *concrete_ctx = dynamic_cast<LocalTPContext *>(ctx.get());
    ASSERT_NE(concrete_ctx, nullptr);

    constexpr int NUM_CYCLES = 5;
    std::atomic<int> completed_cycles{0};

    // Create regular FP32 tensors (no mock BAR - test barrier mechanism only)
    auto tensor1 = std::make_unique<FP32Tensor>(std::vector<size_t>{256});
    auto tensor2 = std::make_unique<FP32Tensor>(std::vector<size_t>{256});

    // Register tensors with the context
    const std::string stage_name = "multi_cycle_test_stage";
    concrete_ctx->registerBARBackedOutput(stage_name, devices[0], tensor1.get());
    concrete_ctx->registerBARBackedOutput(stage_name, devices[1], tensor2.get());

    std::thread cuda_thread([&]()
                            {
        for (int cycle = 0; cycle < NUM_CYCLES; ++cycle)
        {
            for (size_t i = 0; i < tensor1->numel(); ++i)
            {
                tensor1->mutable_data()[i] = static_cast<float>(cycle + 1);
            }
            // allreduce may fail but we're testing barrier reuse
            ctx->allreduce(tensor1.get(), stage_name, tensor1->numel());
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
            // allreduce may fail but we're testing barrier reuse
            ctx->allreduce(tensor2.get(), stage_name, tensor2->numel());
        }
        completed_cycles += NUM_CYCLES; });

    cuda_thread.join();
    rocm_thread.join();

    EXPECT_EQ(completed_cycles.load(), NUM_CYCLES * 2);
}

/**
 * @test PCIeBAR barrier stress test with many rapid cycles
 *
 * NOTE: This test uses regular FP32 tensors without actual BAR backing,
 * so the data transfer may fail. The test validates that the barrier
 * mechanism can handle rapid concurrent access.
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

    // Cast to LocalTPContext for tensor registration
    auto *concrete_ctx = dynamic_cast<LocalTPContext *>(ctx.get());
    ASSERT_NE(concrete_ctx, nullptr);

    constexpr int NUM_ITERATIONS = 100;
    std::atomic<int> completed_count{0};

    // Create regular FP32 tensors (no mock BAR - test barrier mechanism only)
    auto tensor1 = std::make_unique<FP32Tensor>(std::vector<size_t>{64});
    auto tensor2 = std::make_unique<FP32Tensor>(std::vector<size_t>{64});

    // Register tensors with the context
    const std::string stage_name = "stress_test_stage";
    concrete_ctx->registerBARBackedOutput(stage_name, devices[0], tensor1.get());
    concrete_ctx->registerBARBackedOutput(stage_name, devices[1], tensor2.get());

    std::thread cuda_thread([&]()
                            {
        for (int i = 0; i < NUM_ITERATIONS; ++i)
        {
            tensor1->mutable_data()[0] = static_cast<float>(i);
            // allreduce may fail but we're testing barrier stress
            ctx->allreduce(tensor1.get(), stage_name, tensor1->numel());
            completed_count++;
        } });

    std::thread rocm_thread([&]()
                            {
        for (int i = 0; i < NUM_ITERATIONS; ++i)
        {
            tensor2->mutable_data()[0] = static_cast<float>(i * 2);
            // allreduce may fail but we're testing barrier stress
            ctx->allreduce(tensor2.get(), stage_name, tensor2->numel());
            completed_count++;
        } });

    cuda_thread.join();
    rocm_thread.join();

    // Key assertion: all iterations completed (barrier didn't deadlock)
    EXPECT_EQ(completed_count.load(), NUM_ITERATIONS * 2)
        << "Expected " << (NUM_ITERATIONS * 2) << " completed barrier cycles";
}

// =============================================================================
// BAR-Backed Tensor Management Tests
// =============================================================================
// These tests verify the zero-copy tensor registry for PCIeBAR backend.
// Note: MockBARBackedTensor is defined above near the PCIeBAR Barrier tests.

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
