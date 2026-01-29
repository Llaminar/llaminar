/**
 * @file Test__HeterogeneousBackend_GcdMultiBridge.cpp
 * @brief Integration tests for HeterogeneousBackend GCD multi-bridge pattern
 *
 * These tests validate the GCD-based multi-bridge pattern for asymmetric
 * GPU configurations where GCD(cuda_count, rocm_count) > 1. The pattern
 * uses GCD parallel PCIeBAR bridges for increased bandwidth.
 *
 * IMPORTANT: These tests require both CUDA and ROCm GPUs present.
 * They are skipped if hardware is not available.
 *
 * Hardware requirements for full testing:
 * - 2 CUDA + 4 ROCm → GCD=2 (ideal for multi-bridge testing)
 * - Current test system: 1 CUDA + 2 ROCm → GCD=1 (fallback to standard)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HeterogeneousBackend.h"
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

    class Test__HeterogeneousBackend_GcdMultiBridge : public ::testing::Test
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

            // Calculate GCD for current hardware
            if (has_hardware_)
            {
                gcd_ = computeGCD(cuda_count_, rocm_count_);
            }

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
            freeBuffers();
        }

        // ─────────────────────────────────────────────────────────────────────
        // GCD computation helper
        // ─────────────────────────────────────────────────────────────────────

        static int computeGCD(int a, int b)
        {
            while (b != 0)
            {
                int temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Skip macros
        // ─────────────────────────────────────────────────────────────────────

#define REQUIRE_HARDWARE()                                      \
    do                                                          \
    {                                                           \
        if (!has_hardware_)                                     \
        {                                                       \
            GTEST_SKIP() << "Test requires CUDA and ROCm GPUs"; \
        }                                                       \
    } while (0)

#define REQUIRE_GCD_GT_1()                                                             \
    do                                                                                 \
    {                                                                                  \
        if (gcd_ <= 1)                                                                 \
        {                                                                              \
            GTEST_SKIP() << "Test requires GCD > 1 (have " << cuda_count_              \
                         << " CUDA + " << rocm_count_ << " ROCm, GCD=" << gcd_ << ")"; \
        }                                                                              \
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

        DeviceGroup createCurrentHardwareGroup()
        {
            DeviceGroupBuilder builder;
            builder.setName("current_hardware")
                .setScope(CollectiveScope::LOCAL)
                .setLocalRank(0);

            // Add all available CUDA devices
            for (int i = 0; i < cuda_count_; ++i)
            {
                builder.addDevice(DeviceId::cuda(i));
            }

            // Add all available ROCm devices
            for (int i = 0; i < rocm_count_; ++i)
            {
                builder.addDevice(DeviceId::rocm(i));
            }

            return builder.build();
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

        DeviceGroup create2Cuda4RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("2_cuda_4_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .addDevice(DeviceId::rocm(2))
                .addDevice(DeviceId::rocm(3))
                .setLocalRank(0)
                .build();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Allocate and manage GPU buffers (via IBackend interface)
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

        void freeBuffers()
        {
            for (const auto &info : buffers_)
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
        int gcd_ = 1;

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
     * @test Verify topology analysis for current hardware (1 CUDA + 2 ROCm)
     *
     * With GCD(1, 2) = 1, the backend should select STANDARD_3PHASE or
     * PARTIAL_REDUCE_SCATTER pattern, not GCD_MULTI_BRIDGE.
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, TopologyAnalysis_CurrentHardware)
    {
        REQUIRE_HARDWARE();

        auto group = createCurrentHardwareGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed (PCIeBAR not available?)";
        }

        // Log detected configuration
        LOG_INFO("Detected hardware: " << cuda_count_ << " CUDA + " << rocm_count_ << " ROCm (GCD=" << gcd_ << ")");

        // Analyze topology for a large tensor
        size_t tensor_bytes = 8 * 1024 * 1024; // 8 MB
        auto analysis = backend_->analyzeTopology(tensor_bytes);

        // Verify device counts match actual hardware
        EXPECT_EQ(analysis.cuda_count, static_cast<size_t>(cuda_count_));
        EXPECT_EQ(analysis.rocm_count, static_cast<size_t>(rocm_count_));
        EXPECT_EQ(analysis.gcd, static_cast<size_t>(gcd_));

        // Log analysis results
        LOG_INFO("Topology analysis:");
        LOG_INFO("  Pattern: " << static_cast<int>(analysis.pattern));
        LOG_INFO("  GCD: " << analysis.gcd);
        LOG_INFO("  is_symmetric: " << analysis.is_symmetric);
        LOG_INFO("  is_minimal: " << analysis.is_minimal);
        LOG_INFO("  is_cuda_singleton: " << analysis.is_cuda_singleton);
        LOG_INFO("  is_rocm_singleton: " << analysis.is_rocm_singleton);
        LOG_INFO("  Reason: " << analysis.reason);

        // With GCD=1, should NOT select GCD_MULTI_BRIDGE
        if (gcd_ == 1)
        {
            EXPECT_NE(analysis.pattern, HeterogeneousBackend::AllreducePattern::GCD_MULTI_BRIDGE)
                << "GCD=1 should not select GCD_MULTI_BRIDGE pattern";
        }
    }

    /**
     * @test Verify topology analysis for 1 CUDA + 2 ROCm configuration
     *
     * This is a GCD=1 case, so it should use STANDARD_3PHASE or PARTIAL_REDUCE_SCATTER.
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, TopologyAnalysis_1Cuda2Rocm)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        size_t tensor_bytes = 8 * 1024 * 1024; // 8 MB
        auto analysis = backend_->analyzeTopology(tensor_bytes);

        // Verify device counts
        EXPECT_EQ(analysis.cuda_count, 1u);
        EXPECT_EQ(analysis.rocm_count, 2u);
        EXPECT_EQ(analysis.gcd, 1u); // GCD(1,2) = 1

        // Should be classified as CUDA singleton
        EXPECT_TRUE(analysis.is_cuda_singleton)
            << "1 CUDA + N ROCm should be classified as cuda_singleton";
        EXPECT_FALSE(analysis.is_rocm_singleton);
        EXPECT_FALSE(analysis.is_symmetric);
        EXPECT_FALSE(analysis.is_minimal);

        // GCD=1 means no multi-bridge benefit
        EXPECT_NE(analysis.pattern, HeterogeneousBackend::AllreducePattern::GCD_MULTI_BRIDGE)
            << "GCD=1 config should not use GCD_MULTI_BRIDGE";
    }

    /**
     * @test Verify topology analysis selects GCD_MULTI_BRIDGE when GCD > 1
     *
     * This test verifies that with a GCD > 1 configuration (e.g., 2 CUDA + 4 ROCm),
     * the topology analysis would select GCD_MULTI_BRIDGE pattern.
     *
     * NOTE: This test may be skipped if hardware doesn't support the configuration.
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, TopologyAnalysis_GCD2_SelectsMultiBridge)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_CUDA(); // Need at least 2 CUDA

        // Check if we have enough ROCm GPUs for GCD > 1
        if (rocm_count_ < 4)
        {
            GTEST_SKIP() << "Test requires 4 ROCm GPUs for GCD=2 config (have " << rocm_count_ << ")";
        }

        auto group = create2Cuda4RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        size_t tensor_bytes = 8 * 1024 * 1024; // 8 MB
        auto analysis = backend_->analyzeTopology(tensor_bytes);

        // Verify device counts and GCD
        EXPECT_EQ(analysis.cuda_count, 2u);
        EXPECT_EQ(analysis.rocm_count, 4u);
        EXPECT_EQ(analysis.gcd, 2u); // GCD(2,4) = 2

        // With GCD=2, should select GCD_MULTI_BRIDGE for large tensors
        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::GCD_MULTI_BRIDGE)
            << "GCD=2 config with large tensor should select GCD_MULTI_BRIDGE";
    }

    /**
     * @test Verify topology analysis keeps STANDARD for coprime counts
     *
     * With coprime device counts (e.g., 2 CUDA + 3 ROCm), GCD=1,
     * so standard pattern should be used.
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, TopologyAnalysis_Coprime_SelectsStandard)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_CUDA();

        if (rocm_count_ < 3)
        {
            GTEST_SKIP() << "Test requires 3 ROCm GPUs for 2+3 coprime config";
        }

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("2_cuda_3_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .setLocalRank(0)
                         .build();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        size_t tensor_bytes = 8 * 1024 * 1024; // 8 MB
        auto analysis = backend_->analyzeTopology(tensor_bytes);

        // Verify GCD = 1 (coprime)
        EXPECT_EQ(analysis.cuda_count, 2u);
        EXPECT_EQ(analysis.rocm_count, 3u);
        EXPECT_EQ(analysis.gcd, 1u); // GCD(2,3) = 1

        // Should NOT select GCD_MULTI_BRIDGE
        EXPECT_NE(analysis.pattern, HeterogeneousBackend::AllreducePattern::GCD_MULTI_BRIDGE)
            << "Coprime config (GCD=1) should not use GCD_MULTI_BRIDGE";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Bridge Pair Computation Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify bridge pairs computation for current hardware
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, ComputeBridgePairs_CurrentHardware)
    {
        REQUIRE_HARDWARE();

        auto group = createCurrentHardwareGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto pairs = backend_->computeBridgePairs();

        // Number of pairs should equal GCD
        EXPECT_EQ(pairs.size(), static_cast<size_t>(gcd_))
            << "Number of bridge pairs should equal GCD(" << cuda_count_
            << "," << rocm_count_ << ") = " << gcd_;

        LOG_INFO("Bridge pairs for " << cuda_count_ << " CUDA + " << rocm_count_ << " ROCm:");
        for (size_t i = 0; i < pairs.size(); ++i)
        {
            LOG_INFO("  Pair " << i << ": CUDA[" << pairs[i].cuda_device.ordinal
                               << "] <-> ROCm[" << pairs[i].rocm_device.ordinal << "]");

            // Verify pair index
            EXPECT_EQ(pairs[i].pair_index, static_cast<int>(i));

            // Verify device types
            EXPECT_EQ(pairs[i].cuda_device.type, DeviceType::CUDA);
            EXPECT_EQ(pairs[i].rocm_device.type, DeviceType::ROCm);
        }
    }

    /**
     * @test Verify bridge pairs for 1 CUDA + 2 ROCm (GCD=1, single pair)
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, ComputeBridgePairs_1Cuda2Rocm)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto pairs = backend_->computeBridgePairs();

        // GCD(1,2) = 1, so only 1 pair
        ASSERT_EQ(pairs.size(), 1u) << "GCD=1 should produce 1 bridge pair";

        // The pair should be CUDA[0] <-> ROCm[0]
        EXPECT_EQ(pairs[0].cuda_device.ordinal, 0);
        EXPECT_EQ(pairs[0].rocm_device.ordinal, 0);
        EXPECT_EQ(pairs[0].pair_index, 0);
    }

    /**
     * @test Verify bridge pairs for 2 CUDA + 4 ROCm (GCD=2, two pairs)
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, ComputeBridgePairs_2Cuda4Rocm)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_CUDA();

        if (rocm_count_ < 4)
        {
            GTEST_SKIP() << "Test requires 4 ROCm GPUs";
        }

        auto group = create2Cuda4RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto pairs = backend_->computeBridgePairs();

        // GCD(2,4) = 2, so 2 pairs
        ASSERT_EQ(pairs.size(), 2u) << "GCD=2 should produce 2 bridge pairs";

        // Expected pairs with ROCm spacing = 4/2 = 2:
        // Pair 0: CUDA[0] <-> ROCm[0]
        // Pair 1: CUDA[1] <-> ROCm[2]
        EXPECT_EQ(pairs[0].cuda_device.ordinal, 0);
        EXPECT_EQ(pairs[0].rocm_device.ordinal, 0);

        EXPECT_EQ(pairs[1].cuda_device.ordinal, 1);
        EXPECT_EQ(pairs[1].rocm_device.ordinal, 2); // Spacing = M/G = 4/2 = 2

        LOG_INFO("Bridge pairs verification for 2 CUDA + 4 ROCm:");
        LOG_INFO("  Pair 0: CUDA[" << pairs[0].cuda_device.ordinal
                                   << "] <-> ROCm[" << pairs[0].rocm_device.ordinal << "]");
        LOG_INFO("  Pair 1: CUDA[" << pairs[1].cuda_device.ordinal
                                   << "] <-> ROCm[" << pairs[1].rocm_device.ordinal << "]");
    }

    /**
     * @test Verify bridge pairs for 3 CUDA + 6 ROCm (GCD=3, three pairs)
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, ComputeBridgePairs_3Cuda6Rocm)
    {
        REQUIRE_HARDWARE();

        if (cuda_count_ < 3 || rocm_count_ < 6)
        {
            GTEST_SKIP() << "Test requires 3 CUDA + 6 ROCm GPUs";
        }

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("3_cuda_6_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::cuda(2))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .addDevice(DeviceId::rocm(3))
                         .addDevice(DeviceId::rocm(4))
                         .addDevice(DeviceId::rocm(5))
                         .setLocalRank(0)
                         .build();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto pairs = backend_->computeBridgePairs();

        // GCD(3,6) = 3, so 3 pairs
        ASSERT_EQ(pairs.size(), 3u) << "GCD=3 should produce 3 bridge pairs";

        // Expected pairs with ROCm spacing = 6/3 = 2:
        // Pair 0: CUDA[0] <-> ROCm[0]
        // Pair 1: CUDA[1] <-> ROCm[2]
        // Pair 2: CUDA[2] <-> ROCm[4]
        EXPECT_EQ(pairs[0].cuda_device.ordinal, 0);
        EXPECT_EQ(pairs[0].rocm_device.ordinal, 0);

        EXPECT_EQ(pairs[1].cuda_device.ordinal, 1);
        EXPECT_EQ(pairs[1].rocm_device.ordinal, 2);

        EXPECT_EQ(pairs[2].cuda_device.ordinal, 2);
        EXPECT_EQ(pairs[2].rocm_device.ordinal, 4);
    }

    /**
     * @test Verify bridge pairs for 4 CUDA + 4 ROCm (symmetric, GCD=4)
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, ComputeBridgePairs_4Cuda4Rocm_Symmetric)
    {
        REQUIRE_HARDWARE();

        if (cuda_count_ < 4 || rocm_count_ < 4)
        {
            GTEST_SKIP() << "Test requires 4 CUDA + 4 ROCm GPUs";
        }

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("4_cuda_4_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::cuda(2))
                         .addDevice(DeviceId::cuda(3))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .addDevice(DeviceId::rocm(3))
                         .setLocalRank(0)
                         .build();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        auto pairs = backend_->computeBridgePairs();

        // GCD(4,4) = 4, so 4 pairs (symmetric 1:1 mapping)
        ASSERT_EQ(pairs.size(), 4u) << "GCD=4 should produce 4 bridge pairs";

        // Symmetric config: each CUDA maps to corresponding ROCm
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(pairs[i].cuda_device.ordinal, static_cast<int>(i));
            EXPECT_EQ(pairs[i].rocm_device.ordinal, static_cast<int>(i));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Execution Path Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Verify execution path falls back to standard when GCD=1
     *
     * With current hardware (1 CUDA + 2 ROCm, GCD=1), the GCD multi-bridge
     * execution should fall back to the standard 3-phase pattern.
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, ExecutionPath_FallsBackWhenGcd1)
    {
        REQUIRE_HARDWARE();
        REQUIRE_MULTIPLE_ROCM();

        auto group = create1Cuda2RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Verify GCD is 1
        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024);
        EXPECT_EQ(analysis.gcd, 1u);

        // With GCD=1, pattern should not be GCD_MULTI_BRIDGE
        EXPECT_NE(analysis.pattern, HeterogeneousBackend::AllreducePattern::GCD_MULTI_BRIDGE);

        // Execute allreduce - should use standard pattern, not GCD multi-bridge
        size_t count = 1024; // Small tensor for quick test
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);
        ASSERT_NE(rocm1_buf, nullptr);

        // Initialize with different values
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

        // Verify correctness: 1 + 2 + 3 = 6
        std::vector<float> expected(count, 6.0f);

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
     * @test Verify GCD multi-bridge execution produces correct results
     *
     * This test requires GCD > 1 hardware (e.g., 2 CUDA + 4 ROCm).
     */
    TEST_F(Test__HeterogeneousBackend_GcdMultiBridge, ExecutionPath_GcdMultiBridgeCorrectness)
    {
        REQUIRE_HARDWARE();
        REQUIRE_GCD_GT_1();
        REQUIRE_MULTIPLE_CUDA();

        if (rocm_count_ < 4)
        {
            GTEST_SKIP() << "Test requires 4 ROCm GPUs for GCD=2 test";
        }

        auto group = create2Cuda4RocmGroup();

        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        // Verify GCD > 1
        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024);
        EXPECT_EQ(analysis.gcd, 2u);
        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::GCD_MULTI_BRIDGE);

        // Allocate buffers for 2 CUDA + 4 ROCm
        size_t count = 1024;
        void *cuda0_buf = allocateCUDABuffer(0, count);
        void *cuda1_buf = allocateCUDABuffer(1, count);
        void *rocm0_buf = allocateROCmBuffer(0, count);
        void *rocm1_buf = allocateROCmBuffer(1, count);
        void *rocm2_buf = allocateROCmBuffer(2, count);
        void *rocm3_buf = allocateROCmBuffer(3, count);

        ASSERT_NE(cuda0_buf, nullptr);
        ASSERT_NE(cuda1_buf, nullptr);
        ASSERT_NE(rocm0_buf, nullptr);
        ASSERT_NE(rocm1_buf, nullptr);
        ASSERT_NE(rocm2_buf, nullptr);
        ASSERT_NE(rocm3_buf, nullptr);

        // Initialize: each device has its (index + 1) as value
        initializeCUDABuffer(cuda0_buf, 0, std::vector<float>(count, 1.0f));
        initializeCUDABuffer(cuda1_buf, 1, std::vector<float>(count, 2.0f));
        initializeROCmBuffer(rocm0_buf, 0, std::vector<float>(count, 3.0f));
        initializeROCmBuffer(rocm1_buf, 1, std::vector<float>(count, 4.0f));
        initializeROCmBuffer(rocm2_buf, 2, std::vector<float>(count, 5.0f));
        initializeROCmBuffer(rocm3_buf, 3, std::vector<float>(count, 6.0f));

        // Execute allreduce
        std::vector<void *> buffers = {cuda0_buf, cuda1_buf, rocm0_buf, rocm1_buf, rocm2_buf, rocm3_buf};
        bool success = backend_->allreduceMulti(buffers, count,
                                                CollectiveDataType::FLOAT32,
                                                CollectiveOp::ALLREDUCE_SUM);
        ASSERT_TRUE(success) << "allreduceMulti failed: " << backend_->lastError();

        backend_->synchronize();

        // Expected: 1+2+3+4+5+6 = 21
        std::vector<float> expected(count, 21.0f);

        EXPECT_TRUE(verifyAllReduceResult(readCUDABuffer(cuda0_buf, 0, count), expected))
            << "CUDA:0 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(readCUDABuffer(cuda1_buf, 1, count), expected))
            << "CUDA:1 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(readROCmBuffer(rocm0_buf, 0, count), expected))
            << "ROCm:0 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(readROCmBuffer(rocm1_buf, 1, count), expected))
            << "ROCm:1 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(readROCmBuffer(rocm2_buf, 2, count), expected))
            << "ROCm:2 result incorrect";
        EXPECT_TRUE(verifyAllReduceResult(readROCmBuffer(rocm3_buf, 3, count), expected))
            << "ROCm:3 result incorrect";
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
