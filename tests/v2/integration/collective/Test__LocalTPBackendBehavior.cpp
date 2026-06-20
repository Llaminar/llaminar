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
 * 2. Cross-vendor HETEROGENEOUS backend selection
 *
 * Hardware requirements vary per test group:
 * - ROCm backend tests require 2+ ROCm GPUs
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

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

/**
 * @test AUTO backend with mixed GPU types (1 CUDA + 1 ROCm) selects HOST
 *       (host-staged cross-vendor transfer for the 1+1 case)
 */
TEST_F(Test__LocalTPBackendBehavior, AutoBackend_MixedGPUs_SelectsHeterogeneous)
{
    SKIP_IF_NO_HETEROGENEOUS();

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST)
        << "AUTO backend should select HOST for 1+1 mixed GPU configuration (host-staged cross-vendor)";
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

#endif // HAVE_CUDA || HAVE_ROCM
