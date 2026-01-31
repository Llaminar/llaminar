/**
 * @file Test__HeterogeneousBackend_ReduceScatter.cpp
 * @brief Integration tests for HeterogeneousBackend reduce-scatter pattern
 *
 * These tests validate the reduce-scatter + allgather optimization for
 * large tensors in heterogeneous GPU configurations. The pattern reduces
 * cross-vendor PCIe BAR traffic from 100% to 1/max(N,M).
 *
 * IMPORTANT: These tests require both CUDA and ROCm GPUs present.
 * They are skipped if hardware is not available.
 *
 * NOTE: This test uses IBackend interface via BackendManager to avoid
 * directly including both cuda_runtime.h and hip_runtime.h (which conflict).
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HeterogeneousBackend.h"
#include "v2/collective/backends/NCCLBackend.h"
#include "v2/collective/backends/RCCLBackend.h"
#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"
#include "v2/utils/Logger.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace llaminar2::test
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend_ReduceScatter : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Get backends via global accessors (avoids CUDA/HIP header conflicts)
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();

            // Count available devices via IBackend interface
            cuda_count_ = cuda_backend_ ? cuda_backend_->deviceCount() : 0;
            rocm_count_ = rocm_backend_ ? rocm_backend_->deviceCount() : 0;

            // Check minimum requirements
            has_hardware_ = (cuda_count_ >= 1 && rocm_count_ >= 1);

            if (has_hardware_)
            {
                backend_ = std::make_unique<HeterogeneousBackend>();
            }
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
            buffers_.clear();
        }

        // ─────────────────────────────────────────────────────────────────────
        // ─────────────────────────────────────────────────────────────────────
        // Skip helpers - note GTEST_SKIP contains 'return', so these must only
        // be called directly from test bodies, not from helper functions.
        // ─────────────────────────────────────────────────────────────────────

#define REQUIRE_HARDWARE()                                      \
    do                                                          \
    {                                                           \
        if (!has_hardware_)                                     \
        {                                                       \
            GTEST_SKIP() << "Test requires CUDA and ROCm GPUs"; \
        }                                                       \
    } while (0)

#define REQUIRE_MULTIPLE_ROCM()                                                                \
    do                                                                                         \
    {                                                                                          \
        if (rocm_count_ < 2)                                                                   \
        {                                                                                      \
            GTEST_SKIP() << "Test requires at least 2 ROCm GPUs (have " << rocm_count_ << ")"; \
        }                                                                                      \
    } while (0)

#define REQUIRE_MULTIPLE_CUDA()                                                                \
    do                                                                                         \
    {                                                                                          \
        if (cuda_count_ < 2)                                                                   \
        {                                                                                      \
            GTEST_SKIP() << "Test requires at least 2 CUDA GPUs (have " << cuda_count_ << ")"; \
        }                                                                                      \
    } while (0)

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Create device groups
        // ─────────────────────────────────────────────────────────────────────

        DeviceGroup create1Cuda1RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("1_cuda_1_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }

        DeviceGroup create1Cuda2RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("1_cuda_2_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        DeviceGroup create2Cuda2RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("2_cuda_2_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Allocate and initialize GPU buffers (via IBackend interface)
        // ─────────────────────────────────────────────────────────────────────

        void *allocateCUDABuffer(int device_ordinal, size_t count)
        {
            if (!cuda_backend_)
                return nullptr;
            void *ptr = cuda_backend_->allocate(count * sizeof(float), device_ordinal);
            if (ptr)
            {
                buffers_.push_back({ptr, DeviceType::CUDA, device_ordinal});
            }
            return ptr;
        }

        void *allocateROCmBuffer(int device_ordinal, size_t count)
        {
            if (!rocm_backend_)
                return nullptr;
            void *ptr = rocm_backend_->allocate(count * sizeof(float), device_ordinal);
            if (ptr)
            {
                buffers_.push_back({ptr, DeviceType::ROCm, device_ordinal});
            }
            return ptr;
        }

        void initializeCUDABuffer(void *ptr, int device_ordinal, const std::vector<float> &data)
        {
            if (cuda_backend_)
            {
                cuda_backend_->hostToDevice(ptr, data.data(), data.size() * sizeof(float), device_ordinal);
                cuda_backend_->synchronize(device_ordinal);
            }
        }

        void initializeROCmBuffer(void *ptr, int device_ordinal, const std::vector<float> &data)
        {
            if (rocm_backend_)
            {
                rocm_backend_->hostToDevice(ptr, data.data(), data.size() * sizeof(float), device_ordinal);
                rocm_backend_->synchronize(device_ordinal);
            }
        }

        std::vector<float> readCUDABuffer(void *ptr, int device_ordinal, size_t count)
        {
            std::vector<float> result(count);
            if (cuda_backend_)
            {
                cuda_backend_->deviceToHost(result.data(), ptr, count * sizeof(float), device_ordinal);
                cuda_backend_->synchronize(device_ordinal);
            }
            return result;
        }

        std::vector<float> readROCmBuffer(void *ptr, int device_ordinal, size_t count)
        {
            std::vector<float> result(count);
            if (rocm_backend_)
            {
                rocm_backend_->deviceToHost(result.data(), ptr, count * sizeof(float), device_ordinal);
                rocm_backend_->synchronize(device_ordinal);
            }
            return result;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Verify allreduce result
        // ─────────────────────────────────────────────────────────────────────

        bool verifyAllReduceResult(const std::vector<float> &result,
                                   const std::vector<float> &expected,
                                   float tolerance = 1e-5f)
        {
            if (result.size() != expected.size())
            {
                LOG_ERROR("Size mismatch: " << result.size() << " vs " << expected.size());
                return false;
            }

            for (size_t i = 0; i < result.size(); ++i)
            {
                if (std::abs(result[i] - expected[i]) > tolerance)
                {
                    LOG_ERROR("Mismatch at index " << i << ": got " << result[i]
                                                   << ", expected " << expected[i]);
                    return false;
                }
            }
            return true;
        }

        std::unique_ptr<HeterogeneousBackend> backend_;
        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;
        bool has_hardware_ = false;
        int cuda_count_ = 0;
        int rocm_count_ = 0;

        struct BufferInfo
        {
            void *ptr;
            DeviceType type;
            int ordinal;
        };
        std::vector<BufferInfo> buffers_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Pattern Selection Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify reduce-scatter pattern is NOT used for asymmetric configs (1 CUDA + 2 ROCm)
     *
     * Reduce-scatter requires symmetric device counts because each domain produces
     * a different number of chunks after reduce-scatter, which don't align for
     * bridge exchange. Asymmetric configs correctly fall back to standard 3-phase.
     */
    TEST_F(Test__HeterogeneousBackend_ReduceScatter, PatternSelection_1Cuda2Rocm_LargeTensor)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        // Initialize - will fail if hardware setup is wrong, but that's OK
        // We're testing the pattern selection logic here
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed (PCIeBAR not available?)";
        }

        // Even for large tensors, asymmetric configs should NOT use reduce-scatter
        // because the chunk alignment doesn't work (1 CUDA chunk vs 2 ROCm chunks)
        size_t tensor_bytes = 5 * 1024 * 1024;
        EXPECT_FALSE(backend_->shouldUseReduceScatterPattern(tensor_bytes))
            << "Asymmetric config (1 CUDA + 2 ROCm) should use standard 3-phase pattern, not reduce-scatter";

        // Verify plan reflects the fallback to standard pattern
        size_t count = tensor_bytes / sizeof(float);
        auto plan = backend_->planReduceScatter(count, sizeof(float));

        // With asymmetric counts, reduce-scatter is disabled
        EXPECT_FALSE(plan.use_reduce_scatter_pattern)
            << "Plan should indicate standard pattern for asymmetric config";
    }

    /**
     * @test Verify standard pattern is used for small tensors
     */
    TEST_F(Test__HeterogeneousBackend_ReduceScatter, PatternSelection_SmallTensorUsesStandard)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed (PCIeBAR not available?)";
        }

        // 1MB tensor should NOT trigger reduce-scatter pattern
        size_t tensor_bytes = 1 * 1024 * 1024;
        EXPECT_FALSE(backend_->shouldUseReduceScatterPattern(tensor_bytes))
            << "1MB tensor should use standard pattern";
    }

    /**
     * @test Verify minimal config (1+1) doesn't use reduce-scatter even for large tensors
     */
    TEST_F(Test__HeterogeneousBackend_ReduceScatter, PatternSelection_MinimalConfigUsesStandard)
    {
        REQUIRE_HARDWARE();

        auto group = create1Cuda1RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed (PCIeBAR not available?)";
        }

        // Large tensor but minimal config - no bandwidth savings possible
        size_t tensor_bytes = 8 * 1024 * 1024; // 8MB
        EXPECT_FALSE(backend_->shouldUseReduceScatterPattern(tensor_bytes))
            << "Large tensor with 1+1 config should use standard pattern (no savings)";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Chunk Calculation Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify chunk size calculations for 2 CUDA + 2 ROCm configuration
     */
    TEST_F(Test__HeterogeneousBackend_ReduceScatter, ChunkCalculation_2Cuda2Rocm)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_CUDA();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create2Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // 8MB = 2M float elements
        size_t count = 2 * 1024 * 1024;
        auto plan = backend_->planReduceScatter(count, sizeof(float));

        EXPECT_TRUE(plan.use_reduce_scatter_pattern);
        EXPECT_EQ(plan.cuda_device_count, 2u);
        EXPECT_EQ(plan.rocm_device_count, 2u);

        // With 2 CUDA and 2 ROCm devices:
        // - CUDA chunk = count/2 = 1M elements
        // - ROCm chunk = count/2 = 1M elements
        // - Bridge exchange = min(1M, 1M) = 1M elements
        EXPECT_EQ(plan.cuda_chunk_count, count / 2);
        EXPECT_EQ(plan.rocm_chunk_count, count / 2);
        EXPECT_EQ(plan.bridge_exchange_count, count / 2);

        // This is 50% bandwidth reduction compared to standard pattern
        LOG_INFO("Chunk calculation test: "
                 << "Full tensor=" << count << " elements, "
                 << "Bridge exchange=" << plan.bridge_exchange_count << " elements "
                 << "(" << (100.0 * plan.bridge_exchange_count / count) << "% of full)");
    }

    /**
     * @test Verify chunk calculations with asymmetric counts fall back to standard pattern
     *
     * With asymmetric device counts (1 CUDA + 2 ROCm), reduce-scatter is disabled
     * and the plan reflects the standard 3-phase pattern instead.
     */
    TEST_F(Test__HeterogeneousBackend_ReduceScatter, ChunkCalculation_NonDivisible)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Non-divisible count: 1,000,001 elements (not evenly divisible by 2)
        size_t count = 1000001;
        auto plan = backend_->planReduceScatter(count, sizeof(float));

        // With 1 CUDA + 2 ROCm (asymmetric), reduce-scatter is disabled
        // The plan reflects standard 3-phase pattern where all elements are processed
        EXPECT_FALSE(plan.use_reduce_scatter_pattern)
            << "Asymmetric config should disable reduce-scatter";

        // Chunk counts default to full tensor size when reduce-scatter is disabled
        EXPECT_EQ(plan.cuda_chunk_count, count)
            << "CUDA chunk should be full tensor size in 3-phase mode";
        EXPECT_EQ(plan.rocm_chunk_count, count)
            << "ROCm chunk should be full tensor size in 3-phase mode";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Correctness Tests (Full AllReduce with Pattern)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Full allreduce correctness with 1 CUDA + 2 ROCm using large tensor
     *
     * This test validates that the reduce-scatter pattern produces correct
     * allreduce results.
     */
    TEST_F(Test__HeterogeneousBackend_ReduceScatter, Correctness_1Cuda2Rocm_LargeTensor)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Use 5MB = 1.25M float elements
        // This is above the 4MB threshold, so reduce-scatter pattern will be used
        size_t count = 5 * 1024 * 1024 / sizeof(float); // 1.25M elements

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);
        ASSERT_NE(rocm1_buf, nullptr);

        // Initialize with different patterns:
        // cuda:0 = [1, 1, 1, ...]
        // rocm:0 = [2, 2, 2, ...]
        // rocm:1 = [3, 3, 3, ...]
        std::vector<float> cuda0_data(count, 1.0f);
        std::vector<float> rocm0_data(count, 2.0f);
        std::vector<float> rocm1_data(count, 3.0f);

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);
        initializeROCmBuffer(rocm1_buf, 1, rocm1_data);

        // Execute allreduce (should use reduce-scatter pattern for this size)
        std::vector<void *> buffers = {cuda0_buf, rocm0_buf, rocm1_buf};
        bool success = backend_->allreduceMulti(buffers, count,
                                                CollectiveDataType::FLOAT32,
                                                CollectiveOp::ALLREDUCE_SUM);
        ASSERT_TRUE(success) << "allreduceMulti failed: " << backend_->lastError();

        backend_->synchronize();

        // Expected result: 1 + 2 + 3 = 6 on all devices
        std::vector<float> expected(count, 6.0f);

        // Read back and verify all buffers
        auto cuda0_result = readCUDABuffer(cuda0_buf, 0, count);
        auto rocm0_result = readROCmBuffer(rocm0_buf, 0, count);
        auto rocm1_result = readROCmBuffer(rocm1_buf, 1, count);

        EXPECT_TRUE(verifyAllReduceResult(cuda0_result, expected))
            << "CUDA:0 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(rocm0_result, expected))
            << "ROCm:0 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(rocm1_result, expected))
            << "ROCm:1 result incorrect";
    }

    /**
     * @test Verify small tensor uses standard pattern and produces correct result
     */
    TEST_F(Test__HeterogeneousBackend_ReduceScatter, Correctness_SmallTensorUsesStandardPattern)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // 1KB = 256 float elements (well below 4MB threshold)
        size_t count = 256;

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);
        ASSERT_NE(rocm1_buf, nullptr);

        // Initialize with different values
        std::vector<float> cuda0_data(count, 10.0f);
        std::vector<float> rocm0_data(count, 20.0f);
        std::vector<float> rocm1_data(count, 30.0f);

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);
        initializeROCmBuffer(rocm1_buf, 1, rocm1_data);

        // Verify standard pattern will be used
        size_t tensor_bytes = count * sizeof(float);
        EXPECT_FALSE(backend_->shouldUseReduceScatterPattern(tensor_bytes))
            << "Small tensor should use standard pattern";

        // Execute allreduce
        std::vector<void *> buffers = {cuda0_buf, rocm0_buf, rocm1_buf};
        bool success = backend_->allreduceMulti(buffers, count,
                                                CollectiveDataType::FLOAT32,
                                                CollectiveOp::ALLREDUCE_SUM);
        ASSERT_TRUE(success) << "allreduceMulti failed: " << backend_->lastError();

        backend_->synchronize();

        // Expected: 10 + 20 + 30 = 60
        std::vector<float> expected(count, 60.0f);

        auto cuda0_result = readCUDABuffer(cuda0_buf, 0, count);
        auto rocm0_result = readROCmBuffer(rocm0_buf, 0, count);
        auto rocm1_result = readROCmBuffer(rocm1_buf, 1, count);

        EXPECT_TRUE(verifyAllReduceResult(cuda0_result, expected));
        EXPECT_TRUE(verifyAllReduceResult(rocm0_result, expected));
        EXPECT_TRUE(verifyAllReduceResult(rocm1_result, expected));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Topology Analysis Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend_ReduceScatter, TopologyAnalysis_1Cuda2Rocm)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        // Initialize with 1 CUDA + 2 ROCm
        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024); // 8 MB

        EXPECT_EQ(analysis.cuda_count, 1);
        EXPECT_EQ(analysis.rocm_count, 2);
        EXPECT_FALSE(analysis.is_symmetric);
        EXPECT_TRUE(analysis.is_cuda_singleton);
        EXPECT_FALSE(analysis.is_rocm_singleton);
        EXPECT_EQ(analysis.gcd, 1); // GCD(1, 2) = 1

        // For singleton configs (1+N or N+1) with large tensors, partial RS is preferred
        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::PARTIAL_REDUCE_SCATTER);
        EXPECT_EQ(analysis.bridge_parallelism, 1.0); // Single bridge bottleneck

        LOG_INFO("TopologyAnalysis 1+2: " << analysis.reason);
    }

    TEST_F(Test__HeterogeneousBackend_ReduceScatter, TopologyAnalysis_SmallTensor)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Small tensor (1 KB) should always use standard pattern
        auto analysis = backend_->analyzeTopology(1024);

        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::STANDARD_3PHASE);
        EXPECT_NE(analysis.reason.find("Small tensor"), std::string::npos)
            << "Reason should mention small tensor, got: " << analysis.reason;
    }

    TEST_F(Test__HeterogeneousBackend_ReduceScatter, TopologyAnalysis_IntraDomainParallelism)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        // 1 CUDA + 2 ROCm: intra-domain parallelism = max(1, 2) = 2
        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024);
        EXPECT_EQ(analysis.intra_domain_parallelism, 2.0); // max(1, 2) = 2
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
