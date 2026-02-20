/**
 * @file Test__HeterogeneousBackend_PartialRS.cpp
 * @brief Integration tests for HeterogeneousBackend Partial Reduce-Scatter pattern
 *
 * These tests validate the partial reduce-scatter pattern for singleton GPU
 * configurations. Singleton configs are:
 *   - 1 CUDA + N ROCm (N > 1): CUDA is singleton, ROCm does reduce-scatter
 *   - N CUDA + 1 ROCm (N > 1): ROCm is singleton, CUDA does reduce-scatter
 *
 * Our hardware (1 CUDA + 2 ROCm) is a perfect test case for this pattern!
 *
 * Pattern for 1 CUDA + 2 ROCm:
 *   Phase 1: RCCL reduce-scatter → ROCm[0] has chunk[0], ROCm[1] has chunk[1]
 *   Phase 2: Chunked bridge exchange:
 *            - ROCm[0] ↔ CUDA[0] for chunk[0] (direct bridge)
 *            - Stage chunk[1]: ROCm[1] → ROCm[0] → CUDA[0] → ROCm[0] → ROCm[1]
 *   Phase 3: RCCL allgather to reconstruct full tensor on ROCm devices
 *
 * IMPORTANT: These tests require both CUDA and ROCm GPUs present.
 * They are skipped if hardware is not available.
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

    class Test__HeterogeneousBackend_PartialRS : public ::testing::Test
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
            // Free all allocated buffers
            for (auto &info : buffers_)
            {
                if (info.type == DeviceType::CUDA && cuda_backend_)
                {
                    cuda_backend_->free(info.ptr, info.ordinal);
                }
                else if (info.type == DeviceType::ROCm && rocm_backend_)
                {
                    rocm_backend_->free(info.ptr, info.ordinal);
                }
            }
            buffers_.clear();

            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

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

#define REQUIRE_1CUDA_2ROCM()                                                      \
    do                                                                             \
    {                                                                              \
        REQUIRE_HARDWARE();                                                        \
        REQUIRE_MULTIPLE_ROCM();                                                   \
        if (cuda_count_ < 1)                                                       \
        {                                                                          \
            GTEST_SKIP() << "Test requires 1 CUDA + 2 ROCm (our hardware config)"; \
        }                                                                          \
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

        DeviceGroup create2Cuda1RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("2_cuda_1_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::rocm(0))
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
    // Topology Analysis Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify topology correctly identifies 1 CUDA + 2 ROCm as singleton configuration
     *
     * This is our actual hardware configuration. The analyzeTopology() should:
     * - Set is_cuda_singleton = true (1 CUDA device)
     * - Set is_rocm_singleton = false (2 ROCm devices > 1)
     * - Set is_minimal = false (not 1+1)
     * - Set num_chunks = 2 (larger domain device count)
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_TopologyAnalysis_1Cuda2Rocm)
    {
        REQUIRE_1CUDA_2ROCM();

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed (PCIeBAR not available?)";
        }

        // Analyze topology for a large tensor (above threshold)
        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024); // 8 MB

        // Verify device counts
        EXPECT_EQ(analysis.cuda_count, 1u)
            << "Should detect 1 CUDA device";
        EXPECT_EQ(analysis.rocm_count, 2u)
            << "Should detect 2 ROCm devices";

        // Verify singleton classification
        EXPECT_TRUE(analysis.is_cuda_singleton)
            << "1 CUDA + 2 ROCm should have is_cuda_singleton = true";
        EXPECT_FALSE(analysis.is_rocm_singleton)
            << "1 CUDA + 2 ROCm should have is_rocm_singleton = false";

        // Verify not minimal (1+1) and not symmetric (1 != 2)
        EXPECT_FALSE(analysis.is_minimal)
            << "1 CUDA + 2 ROCm is not minimal (1+1)";
        EXPECT_FALSE(analysis.is_symmetric)
            << "1 CUDA + 2 ROCm is not symmetric (1 != 2)";

        // GCD(1, 2) = 1
        EXPECT_EQ(analysis.gcd, 1u)
            << "GCD(1, 2) should be 1";

        // For singleton configs, num_chunks = larger domain device count
        EXPECT_EQ(analysis.num_chunks, 2u)
            << "num_chunks should match larger domain (2 ROCm devices)";

        LOG_INFO("Topology analysis for 1 CUDA + 2 ROCm: " << analysis.reason);
    }

    /**
     * @test Verify PARTIAL_REDUCE_SCATTER pattern is selected for 1+2 config with large tensor
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_CorrectPattern_Selected)
    {
        REQUIRE_1CUDA_2ROCM();

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Large tensor (8 MB) should trigger PARTIAL_REDUCE_SCATTER
        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024);

        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::PARTIAL_REDUCE_SCATTER)
            << "1 CUDA + 2 ROCm with large tensor should select PARTIAL_REDUCE_SCATTER, "
            << "got pattern enum value " << static_cast<int>(analysis.pattern)
            << ". Reason: " << analysis.reason;

        // Verify pattern parameters
        // For 1+2, reduce-scatter in ROCm domain produces 2 chunks
        EXPECT_EQ(analysis.num_chunks, 2u)
            << "num_chunks should be 2 for 1+2 singleton config";

        // Log the analysis for debugging
        LOG_INFO("Pattern analysis for 1+2: "
                 << "pattern=" << static_cast<int>(analysis.pattern)
                 << ", num_chunks=" << analysis.num_chunks
                 << ", bridge_traffic_fraction=" << analysis.bridge_traffic_fraction
                 << ", reason: " << analysis.reason);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Allreduce Correctness Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify small tensor correctness (may use standard pattern, not partial RS)
     *
     * Small tensors (< 4MB) should use the standard 3-phase pattern, but
     * correctness must still be maintained.
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_Allreduce_Correctness_SmallTensor)
    {
        REQUIRE_1CUDA_2ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Small tensor: 1 KB = 256 float elements (well below 4MB threshold)
        size_t count = 256;

        // Verify standard pattern will be used
        size_t tensor_bytes = count * sizeof(float);
        auto analysis = backend_->analyzeTopology(tensor_bytes);
        EXPECT_NE(analysis.pattern, HeterogeneousBackend::AllreducePattern::PARTIAL_REDUCE_SCATTER)
            << "Small tensor should NOT use partial RS pattern";

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr) << "Failed to allocate CUDA buffer";
        ASSERT_NE(rocm0_buf, nullptr) << "Failed to allocate ROCm:0 buffer";
        ASSERT_NE(rocm1_buf, nullptr) << "Failed to allocate ROCm:1 buffer";

        // Initialize with distinct patterns:
        // cuda:0 = [1, 1, 1, ...]
        // rocm:0 = [2, 2, 2, ...]
        // rocm:1 = [3, 3, 3, ...]
        std::vector<float> cuda0_data(count, 1.0f);
        std::vector<float> rocm0_data(count, 2.0f);
        std::vector<float> rocm1_data(count, 3.0f);

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);
        initializeROCmBuffer(rocm1_buf, 1, rocm1_data);

        // Execute allreduce
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
     * @test Verify large tensor correctness using Partial RS pattern
     *
     * Large tensors (>= 4MB) should use the PARTIAL_REDUCE_SCATTER pattern.
     * This test validates that the pattern produces correct allreduce results.
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_Allreduce_Correctness_LargeTensor)
    {
        REQUIRE_1CUDA_2ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti/sendrecvMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Large tensor: 8 MB = 2M float elements (above 4MB threshold)
        size_t count = 2 * 1024 * 1024;

        // Verify partial RS pattern will be used
        size_t tensor_bytes = count * sizeof(float);
        auto analysis = backend_->analyzeTopology(tensor_bytes);
        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::PARTIAL_REDUCE_SCATTER)
            << "Large tensor with 1+2 config should use PARTIAL_REDUCE_SCATTER. "
            << "Reason: " << analysis.reason;

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr) << "Failed to allocate CUDA buffer";
        ASSERT_NE(rocm0_buf, nullptr) << "Failed to allocate ROCm:0 buffer";
        ASSERT_NE(rocm1_buf, nullptr) << "Failed to allocate ROCm:1 buffer";

        // Initialize with distinct patterns:
        // cuda:0 = [1, 1, 1, ...]
        // rocm:0 = [2, 2, 2, ...]
        // rocm:1 = [3, 3, 3, ...]
        std::vector<float> cuda0_data(count, 1.0f);
        std::vector<float> rocm0_data(count, 2.0f);
        std::vector<float> rocm1_data(count, 3.0f);

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);
        initializeROCmBuffer(rocm1_buf, 1, rocm1_data);

        LOG_INFO("Starting Partial RS allreduce with " << count << " elements ("
                                                       << (count * sizeof(float) / (1024 * 1024)) << " MB)");

        // Execute allreduce (should use partial RS pattern internally)
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
            << "CUDA:0 result incorrect after Partial RS allreduce";
        EXPECT_TRUE(verifyAllReduceResult(rocm0_result, expected))
            << "ROCm:0 result incorrect after Partial RS allreduce";
        EXPECT_TRUE(verifyAllReduceResult(rocm1_result, expected))
            << "ROCm:1 result incorrect after Partial RS allreduce";

        LOG_INFO("Partial RS allreduce completed successfully with correct results");
    }

    /**
     * @test Verify 1+1 minimal config does NOT use partial RS (no benefit)
     *
     * Minimal configuration (1 CUDA + 1 ROCm) has no parallelism in either domain,
     * so there's no benefit from reduce-scatter. Should use STANDARD_3PHASE.
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_MinimalConfig_UsesStandard)
    {
        REQUIRE_HARDWARE();

        auto group = create1Cuda1RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Even large tensors should use standard pattern for 1+1
        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024);

        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::STANDARD_3PHASE)
            << "1+1 minimal config should use STANDARD_3PHASE, not partial RS. "
            << "Reason: " << analysis.reason;

        EXPECT_TRUE(analysis.is_minimal)
            << "1+1 config should be classified as minimal";
        EXPECT_FALSE(analysis.is_cuda_singleton)
            << "1+1 config should NOT be classified as CUDA singleton";
        EXPECT_FALSE(analysis.is_rocm_singleton)
            << "1+1 config should NOT be classified as ROCm singleton";
    }

    /**
     * @test Verify varying tensor patterns produce correct results
     *
     * Test with non-uniform data to catch any indexing or staging bugs.
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_Allreduce_VaryingPattern)
    {
        REQUIRE_1CUDA_2ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti/sendrecvMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Large tensor with varying values
        size_t count = 2 * 1024 * 1024; // 8 MB

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);
        ASSERT_NE(rocm1_buf, nullptr);

        // Initialize with varying patterns:
        // cuda:0[i] = i % 100 (0, 1, 2, ..., 99, 0, 1, ...)
        // rocm:0[i] = (i % 100) * 2 (0, 2, 4, ...)
        // rocm:1[i] = (i % 100) * 3 (0, 3, 6, ...)
        std::vector<float> cuda0_data(count);
        std::vector<float> rocm0_data(count);
        std::vector<float> rocm1_data(count);
        std::vector<float> expected(count);

        for (size_t i = 0; i < count; ++i)
        {
            float base = static_cast<float>(i % 100);
            cuda0_data[i] = base;
            rocm0_data[i] = base * 2.0f;
            rocm1_data[i] = base * 3.0f;
            expected[i] = base + base * 2.0f + base * 3.0f; // = base * 6
        }

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);
        initializeROCmBuffer(rocm1_buf, 1, rocm1_data);

        // Execute allreduce
        std::vector<void *> buffers = {cuda0_buf, rocm0_buf, rocm1_buf};
        bool success = backend_->allreduceMulti(buffers, count,
                                                CollectiveDataType::FLOAT32,
                                                CollectiveOp::ALLREDUCE_SUM);
        ASSERT_TRUE(success) << "allreduceMulti failed: " << backend_->lastError();

        backend_->synchronize();

        // Verify results
        auto cuda0_result = readCUDABuffer(cuda0_buf, 0, count);
        auto rocm0_result = readROCmBuffer(rocm0_buf, 0, count);
        auto rocm1_result = readROCmBuffer(rocm1_buf, 1, count);

        EXPECT_TRUE(verifyAllReduceResult(cuda0_result, expected))
            << "CUDA:0 result incorrect with varying pattern";
        EXPECT_TRUE(verifyAllReduceResult(rocm0_result, expected))
            << "ROCm:0 result incorrect with varying pattern";
        EXPECT_TRUE(verifyAllReduceResult(rocm1_result, expected))
            << "ROCm:1 result incorrect with varying pattern";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ROCm Singleton Tests (N CUDA + 1 ROCm)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify 2 CUDA + 1 ROCm (ROCm singleton) topology analysis
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_TopologyAnalysis_2Cuda1Rocm)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_CUDA();

        auto group = create2Cuda1RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024);

        // Verify device counts
        EXPECT_EQ(analysis.cuda_count, 2u);
        EXPECT_EQ(analysis.rocm_count, 1u);

        // Verify ROCm singleton classification
        EXPECT_FALSE(analysis.is_cuda_singleton)
            << "2 CUDA + 1 ROCm should have is_cuda_singleton = false";
        EXPECT_TRUE(analysis.is_rocm_singleton)
            << "2 CUDA + 1 ROCm should have is_rocm_singleton = true";

        // Pattern should be PARTIAL_REDUCE_SCATTER for large tensors
        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::PARTIAL_REDUCE_SCATTER)
            << "2 CUDA + 1 ROCm with large tensor should select PARTIAL_REDUCE_SCATTER. "
            << "Reason: " << analysis.reason;

        // num_chunks = larger domain (2 CUDA)
        EXPECT_EQ(analysis.num_chunks, 2u);
    }

    /**
     * @test Verify 2 CUDA + 1 ROCm allreduce correctness
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_Allreduce_2Cuda1Rocm)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_CUDA();

        // TODO: Remove skip once NCCLCoordinator::sendrecvMulti() is implemented
        GTEST_SKIP() << "NCCL sendrecvMulti not yet supported with NCCLCoordinator";

        auto group = create2Cuda1RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Large tensor
        size_t count = 2 * 1024 * 1024;

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *cuda1_buf = allocateCUDABuffer(1, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(cuda1_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);

        // Initialize with distinct values
        std::vector<float> cuda0_data(count, 1.0f);
        std::vector<float> cuda1_data(count, 2.0f);
        std::vector<float> rocm0_data(count, 4.0f);

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeCUDABuffer(cuda1_buf, 1, cuda1_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);

        // Execute allreduce
        std::vector<void *> buffers = {cuda0_buf, cuda1_buf, rocm0_buf};
        bool success = backend_->allreduceMulti(buffers, count,
                                                CollectiveDataType::FLOAT32,
                                                CollectiveOp::ALLREDUCE_SUM);
        ASSERT_TRUE(success) << "allreduceMulti failed: " << backend_->lastError();

        backend_->synchronize();

        // Expected: 1 + 2 + 4 = 7
        std::vector<float> expected(count, 7.0f);

        auto cuda0_result = readCUDABuffer(cuda0_buf, 0, count);
        auto cuda1_result = readCUDABuffer(cuda1_buf, 1, count);
        auto rocm0_result = readROCmBuffer(rocm0_buf, 0, count);

        EXPECT_TRUE(verifyAllReduceResult(cuda0_result, expected))
            << "CUDA:0 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(cuda1_result, expected))
            << "CUDA:1 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(rocm0_result, expected))
            << "ROCm:0 result incorrect";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Edge Cases and Stress Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify partial RS handles non-divisible tensor sizes correctly
     *
     * When tensor size is not evenly divisible by num_chunks, the last chunk
     * may have a different size. Verify correctness is maintained.
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_NonDivisibleTensorSize)
    {
        REQUIRE_1CUDA_2ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti/sendrecvMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Non-divisible count: 2,097,153 elements (2^21 + 1, not divisible by 2)
        // This creates chunk imbalance: chunk[0] = 1048577, chunk[1] = 1048576
        size_t count = 2 * 1024 * 1024 + 1;

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);
        ASSERT_NE(rocm1_buf, nullptr);

        // Initialize
        std::vector<float> cuda0_data(count, 1.0f);
        std::vector<float> rocm0_data(count, 2.0f);
        std::vector<float> rocm1_data(count, 3.0f);

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);
        initializeROCmBuffer(rocm1_buf, 1, rocm1_data);

        // Execute allreduce
        std::vector<void *> buffers = {cuda0_buf, rocm0_buf, rocm1_buf};
        bool success = backend_->allreduceMulti(buffers, count,
                                                CollectiveDataType::FLOAT32,
                                                CollectiveOp::ALLREDUCE_SUM);
        ASSERT_TRUE(success) << "allreduceMulti failed: " << backend_->lastError();

        backend_->synchronize();

        // Expected: 1 + 2 + 3 = 6
        std::vector<float> expected(count, 6.0f);

        auto cuda0_result = readCUDABuffer(cuda0_buf, 0, count);
        auto rocm0_result = readROCmBuffer(rocm0_buf, 0, count);
        auto rocm1_result = readROCmBuffer(rocm1_buf, 1, count);

        EXPECT_TRUE(verifyAllReduceResult(cuda0_result, expected))
            << "CUDA:0 result incorrect with non-divisible size";
        EXPECT_TRUE(verifyAllReduceResult(rocm0_result, expected))
            << "ROCm:0 result incorrect with non-divisible size";
        EXPECT_TRUE(verifyAllReduceResult(rocm1_result, expected))
            << "ROCm:1 result incorrect with non-divisible size";
    }

    /**
     * @test Verify partial RS at exact threshold boundary (4 MB)
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_ExactThresholdBoundary)
    {
        REQUIRE_1CUDA_2ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti/sendrecvMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        auto group = create1Cuda2RocmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Exactly 4 MB = 1M float elements (at threshold)
        size_t count = 1 * 1024 * 1024;
        size_t tensor_bytes = count * sizeof(float);

        // Verify pattern selection at threshold
        auto analysis = backend_->analyzeTopology(tensor_bytes);
        LOG_INFO("At threshold (" << (tensor_bytes / (1024 * 1024)) << " MB): "
                                  << "pattern = " << static_cast<int>(analysis.pattern)
                                  << ", reason: " << analysis.reason);

        // Allocate buffers
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);
        ASSERT_NE(rocm1_buf, nullptr);

        // Initialize
        std::vector<float> cuda0_data(count, 10.0f);
        std::vector<float> rocm0_data(count, 20.0f);
        std::vector<float> rocm1_data(count, 30.0f);

        initializeCUDABuffer(cuda0_buf, 0, cuda0_data);
        initializeROCmBuffer(rocm0_buf, 0, rocm0_data);
        initializeROCmBuffer(rocm1_buf, 1, rocm1_data);

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

        EXPECT_TRUE(verifyAllReduceResult(cuda0_result, expected))
            << "CUDA:0 result incorrect at threshold";
        EXPECT_TRUE(verifyAllReduceResult(rocm0_result, expected))
            << "ROCm:0 result incorrect at threshold";
        EXPECT_TRUE(verifyAllReduceResult(rocm1_result, expected))
            << "ROCm:1 result incorrect at threshold";
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
