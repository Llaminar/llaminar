/**
 * @file Test__HeterogeneousBackend_Pipelining.cpp
 * @brief Integration tests for HeterogeneousBackend Phase 2→3 pipelining
 *
 * These tests validate the chunk-based pipelining optimization that overlaps
 * Phase 2 (PCIe BAR bridge transfer) with Phase 3 (intra-domain allgather).
 *
 * NOTE: This test uses global backend accessors for device allocation to avoid
 * directly including both cuda_runtime.h and hip_runtime.h (which conflict).
 *
 * Test naming: V2_Integration_HeterogeneousBackend_Pipelining_*
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

#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

namespace llaminar2::test
{

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend_Pipelining : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Get backends via global accessors (avoids header conflicts)
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();

            // Count available devices
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
            FreeBuffers();
        }

        // ─────────────────────────────────────────────────────────────────────
        // ─────────────────────────────────────────────────────────────────────
        // Skip helpers - note GTEST_SKIP contains 'return', so these must only
        // be called directly from test bodies via macros.
        // ─────────────────────────────────────────────────────────────────────

#define REQUIRE_HARDWARE()                                      \
    do                                                          \
    {                                                           \
        if (!has_hardware_)                                     \
        {                                                       \
            GTEST_SKIP() << "Test requires CUDA and ROCm GPUs"; \
        }                                                       \
    } while (0)

#define REQUIRE_MULTI_ROCM()                                                                   \
    do                                                                                         \
    {                                                                                          \
        if (rocm_count_ < 2)                                                                   \
        {                                                                                      \
            GTEST_SKIP() << "Test requires at least 2 ROCm GPUs (have " << rocm_count_ << ")"; \
        }                                                                                      \
    } while (0)

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Create device group
        // ─────────────────────────────────────────────────────────────────────

        DeviceGroup createDeviceGroup()
        {
            DeviceGroupBuilder builder;
            builder.setName("pipelining_test")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0));

            for (int i = 0; i < rocm_count_; ++i)
            {
                builder.addDevice(DeviceId::rocm(i));
            }

            builder.setLocalRank(0);
            return builder.build();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Allocate and initialize test buffers via IBackend
        // ─────────────────────────────────────────────────────────────────────

        bool AllocateBuffers(size_t count)
        {
            count_ = count;
            size_t bytes = count * sizeof(float);

            // Allocate CUDA buffer via IBackend
            if (cuda_backend_)
            {
                cuda_buffer_ = cuda_backend_->allocate(bytes, 0);
                if (!cuda_buffer_)
                {
                    LOG_ERROR("Failed to allocate CUDA buffer");
                    return false;
                }
                cuda_buffers_.push_back(cuda_buffer_);
            }

            // Allocate ROCm buffers via IBackend
            if (rocm_backend_)
            {
                for (int i = 0; i < rocm_count_; ++i)
                {
                    void *buf = rocm_backend_->allocate(bytes, i);
                    if (!buf)
                    {
                        LOG_ERROR("Failed to allocate ROCm buffer " << i);
                        return false;
                    }
                    rocm_buffers_.push_back(buf);
                }
            }

            return true;
        }

        void FreeBuffers()
        {
            if (cuda_backend_ && cuda_buffer_)
            {
                cuda_backend_->free(cuda_buffer_, 0);
                cuda_buffer_ = nullptr;
            }
            cuda_buffers_.clear();

            if (rocm_backend_)
            {
                for (size_t i = 0; i < rocm_buffers_.size(); ++i)
                {
                    rocm_backend_->free(rocm_buffers_[i], static_cast<int>(i));
                }
            }
            rocm_buffers_.clear();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Initialize buffers with test data
        // ─────────────────────────────────────────────────────────────────────

        void InitializeBuffers(float cuda_value, const std::vector<float> &rocm_values)
        {
            std::vector<float> cuda_data(count_, cuda_value);
            if (cuda_backend_ && cuda_buffer_)
            {
                cuda_backend_->hostToDevice(cuda_buffer_, cuda_data.data(), count_ * sizeof(float), 0);
            }

            if (rocm_backend_)
            {
                for (size_t i = 0; i < rocm_buffers_.size(); ++i)
                {
                    float val = (i < rocm_values.size()) ? rocm_values[i] : 0.0f;
                    std::vector<float> rocm_data(count_, val);
                    rocm_backend_->hostToDevice(
                        rocm_buffers_[i], rocm_data.data(), count_ * sizeof(float), static_cast<int>(i));
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Copy results back to host
        // ─────────────────────────────────────────────────────────────────────

        std::vector<float> GetCUDAResult()
        {
            std::vector<float> result(count_);
            if (cuda_backend_ && cuda_buffer_)
            {
                cuda_backend_->deviceToHost(result.data(), cuda_buffer_, count_ * sizeof(float), 0);
            }
            return result;
        }

        std::vector<float> GetROCmResult(int device_idx)
        {
            std::vector<float> result(count_);
            if (rocm_backend_ && device_idx < static_cast<int>(rocm_buffers_.size()))
            {
                rocm_backend_->deviceToHost(
                    result.data(), rocm_buffers_[device_idx], count_ * sizeof(float), device_idx);
            }
            return result;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Build buffer list in device order
        // ─────────────────────────────────────────────────────────────────────

        std::vector<void *> BuildBufferList()
        {
            std::vector<void *> buffers;
            buffers.push_back(cuda_buffer_);
            for (void *buf : rocm_buffers_)
            {
                buffers.push_back(buf);
            }
            return buffers;
        }

        // Member variables
        std::unique_ptr<HeterogeneousBackend> backend_;
        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;
        int cuda_count_ = 0;
        int rocm_count_ = 0;
        bool has_hardware_ = false;

        void *cuda_buffer_ = nullptr;
        std::vector<void *> cuda_buffers_;
        std::vector<void *> rocm_buffers_;
        size_t count_ = 0;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Correctness Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend_Pipelining, LargeTensor_Correctness)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTI_ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        // 8MB tensor with 1 CUDA + 2 ROCm
        size_t count = 2 * 1024 * 1024; // 2M float32 elements = 8MB

        ASSERT_TRUE(AllocateBuffers(count));

        auto group = createDeviceGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Initialize: CUDA has 1.0, ROCm0 has 2.0, ROCm1 has 3.0
        InitializeBuffers(1.0f, {2.0f, 3.0f});

        // Execute pipelined allreduce
        auto buffers = BuildBufferList();
        ASSERT_TRUE(backend_->executeReduceScatterPatternPipelined(
            cuda_buffers_, rocm_buffers_, count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify results: all buffers should have sum = 1.0 + 2.0 + 3.0 = 6.0
        auto cuda_result = GetCUDAResult();
        auto rocm0_result = GetROCmResult(0);
        auto rocm1_result = GetROCmResult(1);

        // Check first few elements
        for (size_t i = 0; i < 10 && i < count; ++i)
        {
            EXPECT_NEAR(cuda_result[i], 6.0f, 0.01f)
                << "CUDA result mismatch at index " << i;
            EXPECT_NEAR(rocm0_result[i], 6.0f, 0.01f)
                << "ROCm0 result mismatch at index " << i;
            EXPECT_NEAR(rocm1_result[i], 6.0f, 0.01f)
                << "ROCm1 result mismatch at index " << i;
        }

        // Check a random sample
        std::vector<size_t> samples = {100, 1000, 10000, count / 2, count - 1};
        for (size_t idx : samples)
        {
            if (idx < count)
            {
                EXPECT_NEAR(cuda_result[idx], 6.0f, 0.01f);
                EXPECT_NEAR(rocm0_result[idx], 6.0f, 0.01f);
                EXPECT_NEAR(rocm1_result[idx], 6.0f, 0.01f);
            }
        }
    }

    TEST_F(Test__HeterogeneousBackend_Pipelining, ChunkBoundary_ExactDivisible)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTI_ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        // Tensor size exactly divisible by 1MB chunk
        // 4MB = 1M float32 elements = exactly 4 chunks
        size_t count = 1 * 1024 * 1024;

        ASSERT_TRUE(AllocateBuffers(count));

        auto group = createDeviceGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Initialize buffers
        InitializeBuffers(1.0f, {2.0f, 3.0f});

        // Execute
        ASSERT_TRUE(backend_->executeReduceScatterPatternPipelined(
            cuda_buffers_, rocm_buffers_, count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify
        auto cuda_result = GetCUDAResult();
        EXPECT_NEAR(cuda_result[0], 6.0f, 0.01f);
        EXPECT_NEAR(cuda_result[count - 1], 6.0f, 0.01f);
    }

    TEST_F(Test__HeterogeneousBackend_Pipelining, ChunkBoundary_WithRemainder)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTI_ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        // Tensor size NOT divisible by 1MB chunk
        // 4.5MB = 1152K float32 elements (4 full chunks + 128K remainder)
        size_t count = 1152 * 1024;

        ASSERT_TRUE(AllocateBuffers(count));

        auto group = createDeviceGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Initialize buffers
        InitializeBuffers(1.0f, {2.0f, 3.0f});

        // Verify the plan shows correct remainder handling
        auto plan = backend_->planPipelining(count, sizeof(float));
        EXPECT_GT(plan.num_chunks, 1u);
        EXPECT_NE(plan.chunk_elements, plan.last_chunk_elements);

        // Execute
        ASSERT_TRUE(backend_->executeReduceScatterPatternPipelined(
            cuda_buffers_, rocm_buffers_, count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify - check at chunk boundaries and remainder
        auto cuda_result = GetCUDAResult();
        size_t chunk_elements = plan.chunk_elements;

        // First chunk boundary
        EXPECT_NEAR(cuda_result[chunk_elements - 1], 6.0f, 0.01f);
        EXPECT_NEAR(cuda_result[chunk_elements], 6.0f, 0.01f);

        // Last element (in remainder chunk)
        EXPECT_NEAR(cuda_result[count - 1], 6.0f, 0.01f);
    }

    TEST_F(Test__HeterogeneousBackend_Pipelining, SmallTensor_FallsBackToStandard)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTI_ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        // 2MB tensor (below 4MB pipelining threshold)
        size_t count = 512 * 1024; // 512K elements = 2MB

        ASSERT_TRUE(AllocateBuffers(count));

        auto group = createDeviceGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Verify plan shows no pipelining
        auto plan = backend_->planPipelining(count, sizeof(float));
        EXPECT_FALSE(plan.will_use_pipelining);
        EXPECT_EQ(plan.num_chunks, 1u);

        // Initialize and execute (should fall back to standard pattern)
        InitializeBuffers(1.0f, {2.0f, 3.0f});

        ASSERT_TRUE(backend_->executeReduceScatterPatternPipelined(
            cuda_buffers_, rocm_buffers_, count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify correctness (same as standard)
        auto cuda_result = GetCUDAResult();
        EXPECT_NEAR(cuda_result[0], 6.0f, 0.01f);
        EXPECT_NEAR(cuda_result[count - 1], 6.0f, 0.01f);
    }

    TEST_F(Test__HeterogeneousBackend_Pipelining, MatchesNonPipelinedResult)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTI_ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        // Use a large tensor where pipelining kicks in
        size_t count = 2 * 1024 * 1024; // 8MB

        // Run non-pipelined first
        ASSERT_TRUE(AllocateBuffers(count));
        auto group = createDeviceGroup();
        ASSERT_TRUE(backend_->initialize(group));

        InitializeBuffers(1.0f, {2.0f, 3.0f});
        ASSERT_TRUE(backend_->executeReduceScatterPattern(
            cuda_buffers_, rocm_buffers_, count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        auto non_pipelined_result = GetCUDAResult();

        // Shutdown and reinitialize for clean state
        backend_->shutdown();
        FreeBuffers();

        ASSERT_TRUE(AllocateBuffers(count));
        ASSERT_TRUE(backend_->initialize(group));

        // Run pipelined
        InitializeBuffers(1.0f, {2.0f, 3.0f});
        ASSERT_TRUE(backend_->executeReduceScatterPatternPipelined(
            cuda_buffers_, rocm_buffers_, count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        auto pipelined_result = GetCUDAResult();

        // Results should match exactly
        for (size_t i = 0; i < count; i += 10000) // Sample every 10K elements
        {
            EXPECT_NEAR(pipelined_result[i], non_pipelined_result[i], 0.001f)
                << "Mismatch at index " << i
                << ": pipelined=" << pipelined_result[i]
                << ", non-pipelined=" << non_pipelined_result[i];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Performance Tests (Optional - may be moved to perf suite)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend_Pipelining, Performance_MeasureTiming)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTI_ROCM();

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        // 16MB tensor for meaningful timing
        size_t count = 4 * 1024 * 1024; // 4M elements = 16MB

        ASSERT_TRUE(AllocateBuffers(count));
        auto group = createDeviceGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Warmup run
        InitializeBuffers(1.0f, {2.0f, 3.0f});
        ASSERT_TRUE(backend_->executeReduceScatterPatternPipelined(
            cuda_buffers_, rocm_buffers_, count,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Measure non-pipelined
        double non_pipelined_ms = 0.0;
        const int iterations = 3;

        for (int i = 0; i < iterations; ++i)
        {
            InitializeBuffers(1.0f, {2.0f, 3.0f});
            auto start = std::chrono::high_resolution_clock::now();
            backend_->executeReduceScatterPattern(
                cuda_buffers_, rocm_buffers_, count,
                CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
            auto end = std::chrono::high_resolution_clock::now();
            non_pipelined_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        non_pipelined_ms /= iterations;

        // Measure pipelined
        double pipelined_ms = 0.0;

        for (int i = 0; i < iterations; ++i)
        {
            InitializeBuffers(1.0f, {2.0f, 3.0f});
            auto start = std::chrono::high_resolution_clock::now();
            backend_->executeReduceScatterPatternPipelined(
                cuda_buffers_, rocm_buffers_, count,
                CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
            auto end = std::chrono::high_resolution_clock::now();
            pipelined_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        pipelined_ms /= iterations;

        // Log results (not asserting specific speedup since V1 is serial)
        LOG_INFO("Performance: 16MB allreduce"
                 << "\n  Non-pipelined: " << non_pipelined_ms << " ms"
                 << "\n  Pipelined:     " << pipelined_ms << " ms"
                 << "\n  Ratio:         " << (non_pipelined_ms / pipelined_ms) << "x");

        // V1 Note: Currently pipelined may be similar or slightly slower due to
        // serial chunk processing. True async overlap is future work.
        // Just verify correctness here.
        auto result = GetCUDAResult();
        EXPECT_NEAR(result[0], 6.0f, 0.01f);
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
