/**
 * @file Test__PCIeBarBackend_MultiPair.cpp
 * @brief Integration tests for PCIeBARBackend multi-pair operations
 *
 * Tests the multi-pair CUDA↔ROCm collective backend functionality where
 * multiple device pairs can perform allreduce operations in parallel.
 *
 * Prerequisites:
 * - At least 1 CUDA + 2 ROCm devices (or 2 CUDA + 2 ROCm for full multi-pair tests)
 * - CAP_SYS_ADMIN capability for PCIe BAR mapping
 * - AMD GPUs with large BAR support
 *
 * Test coverage:
 * - Single-pair initialization via multi-pair API (backward compatibility)
 * - Multi-pair initialization with 2+ pairs
 * - Multi-pair allreduce correctness
 * - Backward compatibility: old initialize() API still works
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"
#include "v2/backends/p2p/DirectP2P.h"
#include "v2/utils/Logger.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__PCIeBarBackend_MultiPair : public ::testing::Test
    {
    protected:
        std::unique_ptr<PCIeBARBackend> backend_;
        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;

        int cuda_count_ = 0;
        int rocm_count_ = 0;
        bool has_pcie_bar_p2p_ = false;

        // Minimum test buffer size
        static constexpr size_t TEST_COUNT = 1024;
        static constexpr size_t TEST_BYTES = TEST_COUNT * sizeof(float);

        void SetUp() override
        {
            // Get backend instances
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();

            if (cuda_backend_)
            {
                cuda_count_ = cuda_backend_->deviceCount();
            }
            if (rocm_backend_)
            {
                rocm_count_ = rocm_backend_->deviceCount();
            }

            // Check if PCIe BAR P2P is available
            auto caps = DirectP2PEngine::probeCapabilities();
            has_pcie_bar_p2p_ = caps.canDoPCIeBarP2P();

            if (has_pcie_bar_p2p_)
            {
                LOG_INFO("PCIe BAR P2P available:");
                LOG_INFO("  CUDA devices: " << cuda_count_);
                LOG_INFO("  ROCm devices: " << rocm_count_);
                LOG_INFO("  AMD BARs discovered: " << caps.discovered_bars.size());
            }
            else
            {
                LOG_WARN("PCIe BAR P2P not available:");
                LOG_WARN("  BAR access: " << (caps.pcie_bar_accessible ? "YES" : "NO"));
                LOG_WARN("  IOMEMORY: " << (caps.pcie_bar_iomemory_supported ? "YES" : "NO"));
            }
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

        // ─────────────────────────────────────────────────────────────────────────────
        // Skip Macros - must be called directly from test bodies
        // ─────────────────────────────────────────────────────────────────────────────

#define REQUIRE_PCIE_BAR_P2P()                                     \
    do                                                             \
    {                                                              \
        if (!has_pcie_bar_p2p_)                                    \
        {                                                          \
            GTEST_SKIP() << "PCIe BAR P2P hardware not available"; \
        }                                                          \
    } while (0)

#define REQUIRE_MIN_CUDA(n)                                               \
    do                                                                    \
    {                                                                     \
        if (cuda_count_ < (n))                                            \
        {                                                                 \
            GTEST_SKIP() << "Test requires at least " << (n)              \
                         << " CUDA devices (have " << cuda_count_ << ")"; \
        }                                                                 \
    } while (0)

#define REQUIRE_MIN_ROCM(n)                                               \
    do                                                                    \
    {                                                                     \
        if (rocm_count_ < (n))                                            \
        {                                                                 \
            GTEST_SKIP() << "Test requires at least " << (n)              \
                         << " ROCm devices (have " << rocm_count_ << ")"; \
        }                                                                 \
    } while (0)

        // ─────────────────────────────────────────────────────────────────────────────
        // Helper: Allocate and initialize buffers
        // ─────────────────────────────────────────────────────────────────────────────

        void *allocateCUDA(int device, size_t bytes, float init_value = 0.0f)
        {
            void *ptr = cuda_backend_->allocate(bytes, device);
            if (ptr && init_value != 0.0f)
            {
                std::vector<float> host(bytes / sizeof(float), init_value);
                cuda_backend_->hostToDevice(ptr, host.data(), bytes, device);
                cuda_backend_->synchronize(device);
            }
            return ptr;
        }

        void *allocateROCm(int device, size_t bytes, float init_value = 0.0f)
        {
            void *ptr = rocm_backend_->allocate(bytes, device);
            if (ptr && init_value != 0.0f)
            {
                std::vector<float> host(bytes / sizeof(float), init_value);
                rocm_backend_->hostToDevice(ptr, host.data(), bytes, device);
                rocm_backend_->synchronize(device);
            }
            return ptr;
        }

        void freeCUDA(int device, void *ptr)
        {
            if (ptr)
                cuda_backend_->free(ptr, device);
        }

        void freeROCm(int device, void *ptr)
        {
            if (ptr)
                rocm_backend_->free(ptr, device);
        }

        std::vector<float> readFromCUDA(int device, void *ptr, size_t count)
        {
            std::vector<float> host(count);
            cuda_backend_->deviceToHost(host.data(), ptr, count * sizeof(float), device);
            cuda_backend_->synchronize(device);
            return host;
        }

        std::vector<float> readFromROCm(int device, void *ptr, size_t count)
        {
            std::vector<float> host(count);
            rocm_backend_->deviceToHost(host.data(), ptr, count * sizeof(float), device);
            rocm_backend_->synchronize(device);
            return host;
        }

        // Read from BAR-mapped memory (host-accessible)
        std::vector<float> readFromBAR(void *ptr, size_t count)
        {
            std::vector<float> host(count);
            std::memcpy(host.data(), ptr, count * sizeof(float));
            return host;
        }

        // ─────────────────────────────────────────────────────────────────────────────
        // Helper: MSE and accuracy metrics
        // ─────────────────────────────────────────────────────────────────────────────

        static double computeMSE(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size())
                return std::numeric_limits<double>::max();
            double sum = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum / a.size();
        }

        static float computeMaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size())
                return std::numeric_limits<float>::max();
            float max_diff = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
            }
            return max_diff;
        }
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-Pair Initialization Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Test single-pair initialization via multi-pair API
     *
     * Verifies that the multi-pair API works correctly with just one pair,
     * providing backward-compatible functionality with additional flexibility.
     */
    TEST_F(Test__PCIeBarBackend_MultiPair, MultiPair_InitializeWithOnePair)
    {
        REQUIRE_PCIE_BAR_P2P();
        REQUIRE_MIN_CUDA(1);
        REQUIRE_MIN_ROCM(1);

        backend_ = std::make_unique<PCIeBARBackend>();

        // Initialize with single pair via multi-pair API
        std::vector<DevicePair> pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0}};

        bool result = backend_->initializeMultiPair(pairs);
        EXPECT_TRUE(result) << "Single-pair initialization via multi-pair API should succeed";

        EXPECT_TRUE(backend_->isInitialized()) << "Backend should be initialized";
        EXPECT_TRUE(backend_->isMultiPairMode()) << "Should be in multi-pair mode";

        const auto &stored_pairs = backend_->getDevicePairs();
        ASSERT_EQ(stored_pairs.size(), 1) << "Should have exactly 1 pair";
        EXPECT_EQ(stored_pairs[0].cuda_device, DeviceId::cuda(0));
        EXPECT_EQ(stored_pairs[0].rocm_device, DeviceId::rocm(0));
        EXPECT_EQ(stored_pairs[0].pair_index, 0);
    }

    /**
     * @brief Test two-pair initialization
     *
     * Tests the full multi-pair mode with 2 CUDA↔ROCm pairs.
     * Requires 2 CUDA and 2 ROCm devices.
     *
     * NOTE: allreduceMultiPair for 2+ pairs is a feature under development.
     * This test verifies initialization but skips allreduce verification
     * until the multi-pair allreduce implementation is complete.
     */
    TEST_F(Test__PCIeBarBackend_MultiPair, MultiPair_InitializeWithTwoPairs)
    {
        REQUIRE_PCIE_BAR_P2P();
        REQUIRE_MIN_CUDA(2);
        REQUIRE_MIN_ROCM(2);

        backend_ = std::make_unique<PCIeBARBackend>();

        // Initialize with two pairs: CUDA[0]↔ROCm[0], CUDA[1]↔ROCm[1]
        std::vector<DevicePair> pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0},
            {DeviceId::cuda(1), DeviceId::rocm(1), 1}};

        bool result = backend_->initializeMultiPair(pairs);
        EXPECT_TRUE(result) << "Two-pair initialization should succeed";

        EXPECT_TRUE(backend_->isInitialized());
        EXPECT_TRUE(backend_->isMultiPairMode());

        const auto &stored_pairs = backend_->getDevicePairs();
        ASSERT_EQ(stored_pairs.size(), 2) << "Should have exactly 2 pairs";

        // Verify pair 0
        EXPECT_EQ(stored_pairs[0].cuda_device, DeviceId::cuda(0));
        EXPECT_EQ(stored_pairs[0].rocm_device, DeviceId::rocm(0));

        // Verify pair 1
        EXPECT_EQ(stored_pairs[1].cuda_device, DeviceId::cuda(1));
        EXPECT_EQ(stored_pairs[1].rocm_device, DeviceId::rocm(1));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-Pair AllReduce Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Test allreduceMultiPair correctness with single pair
     *
     * Verifies that allreduceMultiPair produces correct results when
     * using a single pair, matching the behavior of registered allreduce.
     */
    TEST_F(Test__PCIeBarBackend_MultiPair, MultiPair_AllreduceCorrectness_SinglePair)
    {
        REQUIRE_PCIE_BAR_P2P();
        REQUIRE_MIN_CUDA(1);
        REQUIRE_MIN_ROCM(1);

        backend_ = std::make_unique<PCIeBARBackend>();

        // Initialize with single pair
        std::vector<DevicePair> pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0}};
        ASSERT_TRUE(backend_->initializeMultiPair(pairs));

        // Allocate CUDA buffer on device 0
        void *cuda_buf = allocateCUDA(0, TEST_BYTES, 0.0f);
        ASSERT_NE(cuda_buf, nullptr) << "Failed to allocate CUDA buffer";

        // Allocate ROCm buffer from BAR region
        auto rocm_alloc = backend_->allocateInBarRegion(TEST_BYTES);
        ASSERT_TRUE(rocm_alloc.has_value()) << "Failed to allocate BAR region";
        auto [rocm_buf, bar_offset] = *rocm_alloc;

        // Initialize buffers with known values
        // CUDA: all 3.0f, ROCm (via BAR): all 7.0f
        {
            std::vector<float> cuda_init(TEST_COUNT, 3.0f);
            cuda_backend_->hostToDevice(cuda_buf, cuda_init.data(), TEST_BYTES, 0);
            cuda_backend_->synchronize(0);

            std::vector<float> rocm_init(TEST_COUNT, 7.0f);
            std::memcpy(rocm_buf, rocm_init.data(), TEST_BYTES);
        }

        // Prepare buffer vectors for allreduceMultiPair
        std::vector<void *> cuda_buffers = {cuda_buf};
        std::vector<void *> rocm_buffers = {rocm_buf};

        // Perform multi-pair allreduce
        ASSERT_TRUE(backend_->allreduceMultiPair(
            cuda_buffers, rocm_buffers, TEST_COUNT, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair should succeed";

        // Verify result: 3.0 + 7.0 = 10.0 on both devices
        auto cuda_result = readFromCUDA(0, cuda_buf, TEST_COUNT);
        auto rocm_result = readFromBAR(rocm_buf, TEST_COUNT);

        std::vector<float> expected(TEST_COUNT, 10.0f);

        // Check CUDA results
        EXPECT_NEAR(cuda_result[0], 10.0f, 0.001f) << "CUDA result[0] mismatch";
        EXPECT_NEAR(cuda_result[TEST_COUNT / 2], 10.0f, 0.001f) << "CUDA result[mid] mismatch";
        EXPECT_NEAR(cuda_result[TEST_COUNT - 1], 10.0f, 0.001f) << "CUDA result[last] mismatch";

        // Check ROCm results
        EXPECT_NEAR(rocm_result[0], 10.0f, 0.001f) << "ROCm result[0] mismatch";
        EXPECT_NEAR(rocm_result[TEST_COUNT / 2], 10.0f, 0.001f) << "ROCm result[mid] mismatch";
        EXPECT_NEAR(rocm_result[TEST_COUNT - 1], 10.0f, 0.001f) << "ROCm result[last] mismatch";

        // Compute overall accuracy
        double cuda_mse = computeMSE(cuda_result, expected);
        double rocm_mse = computeMSE(rocm_result, expected);
        LOG_INFO("Single-pair allreduce MSE: CUDA=" << cuda_mse << ", ROCm=" << rocm_mse);

        EXPECT_LT(cuda_mse, 1e-6) << "CUDA MSE should be near zero";
        EXPECT_LT(rocm_mse, 1e-6) << "ROCm MSE should be near zero";

        // Cleanup
        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    /**
     * @brief Test allreduceMultiPair with two pairs
     *
     * Verifies that multiple pairs can perform allreduce simultaneously
     * with correct results on all devices.
     *
     * NOTE: This test is marked as DISABLED until the multi-pair allreduce
     * implementation is complete. Enable via --gtest_also_run_disabled_tests
     * or by removing the DISABLED_ prefix.
     */
    TEST_F(Test__PCIeBarBackend_MultiPair, DISABLED_MultiPair_AllreduceCorrectness_TwoPairs)
    {
        REQUIRE_PCIE_BAR_P2P();
        REQUIRE_MIN_CUDA(2);
        REQUIRE_MIN_ROCM(2);

        backend_ = std::make_unique<PCIeBARBackend>();

        // Initialize with two pairs
        std::vector<DevicePair> pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0},
            {DeviceId::cuda(1), DeviceId::rocm(1), 1}};
        ASSERT_TRUE(backend_->initializeMultiPair(pairs));

        // Allocate CUDA buffers on both devices
        void *cuda_buf0 = allocateCUDA(0, TEST_BYTES, 0.0f);
        void *cuda_buf1 = allocateCUDA(1, TEST_BYTES, 0.0f);
        ASSERT_NE(cuda_buf0, nullptr) << "Failed to allocate CUDA buffer 0";
        ASSERT_NE(cuda_buf1, nullptr) << "Failed to allocate CUDA buffer 1";

        // Allocate ROCm buffers from BAR region
        auto rocm_alloc0 = backend_->allocateInBarRegion(TEST_BYTES);
        auto rocm_alloc1 = backend_->allocateInBarRegion(TEST_BYTES);
        ASSERT_TRUE(rocm_alloc0.has_value()) << "Failed to allocate BAR region 0";
        ASSERT_TRUE(rocm_alloc1.has_value()) << "Failed to allocate BAR region 1";

        void *rocm_buf0 = rocm_alloc0->first;
        void *rocm_buf1 = rocm_alloc1->first;

        // Initialize buffers with different values per pair:
        // Pair 0: CUDA=1.0, ROCm=2.0 → result=3.0
        // Pair 1: CUDA=4.0, ROCm=5.0 → result=9.0
        {
            std::vector<float> cuda_init0(TEST_COUNT, 1.0f);
            std::vector<float> cuda_init1(TEST_COUNT, 4.0f);
            cuda_backend_->hostToDevice(cuda_buf0, cuda_init0.data(), TEST_BYTES, 0);
            cuda_backend_->hostToDevice(cuda_buf1, cuda_init1.data(), TEST_BYTES, 1);
            cuda_backend_->synchronize(0);
            cuda_backend_->synchronize(1);

            std::vector<float> rocm_init0(TEST_COUNT, 2.0f);
            std::vector<float> rocm_init1(TEST_COUNT, 5.0f);
            std::memcpy(rocm_buf0, rocm_init0.data(), TEST_BYTES);
            std::memcpy(rocm_buf1, rocm_init1.data(), TEST_BYTES);
        }

        // Prepare buffer vectors
        std::vector<void *> cuda_buffers = {cuda_buf0, cuda_buf1};
        std::vector<void *> rocm_buffers = {rocm_buf0, rocm_buf1};

        // Perform multi-pair allreduce
        ASSERT_TRUE(backend_->allreduceMultiPair(
            cuda_buffers, rocm_buffers, TEST_COUNT, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair should succeed";

        // Verify results
        auto cuda_result0 = readFromCUDA(0, cuda_buf0, TEST_COUNT);
        auto cuda_result1 = readFromCUDA(1, cuda_buf1, TEST_COUNT);
        auto rocm_result0 = readFromBAR(rocm_buf0, TEST_COUNT);
        auto rocm_result1 = readFromBAR(rocm_buf1, TEST_COUNT);

        std::vector<float> expected0(TEST_COUNT, 3.0f); // 1.0 + 2.0
        std::vector<float> expected1(TEST_COUNT, 9.0f); // 4.0 + 5.0

        // Check pair 0 results
        EXPECT_NEAR(cuda_result0[0], 3.0f, 0.001f) << "Pair 0 CUDA result mismatch";
        EXPECT_NEAR(rocm_result0[0], 3.0f, 0.001f) << "Pair 0 ROCm result mismatch";

        // Check pair 1 results
        EXPECT_NEAR(cuda_result1[0], 9.0f, 0.001f) << "Pair 1 CUDA result mismatch";
        EXPECT_NEAR(rocm_result1[0], 9.0f, 0.001f) << "Pair 1 ROCm result mismatch";

        // Compute accuracy metrics
        double cuda_mse0 = computeMSE(cuda_result0, expected0);
        double cuda_mse1 = computeMSE(cuda_result1, expected1);
        double rocm_mse0 = computeMSE(rocm_result0, expected0);
        double rocm_mse1 = computeMSE(rocm_result1, expected1);

        LOG_INFO("Two-pair allreduce MSE:");
        LOG_INFO("  Pair 0: CUDA=" << cuda_mse0 << ", ROCm=" << rocm_mse0);
        LOG_INFO("  Pair 1: CUDA=" << cuda_mse1 << ", ROCm=" << rocm_mse1);

        EXPECT_LT(cuda_mse0, 1e-6) << "Pair 0 CUDA MSE should be near zero";
        EXPECT_LT(cuda_mse1, 1e-6) << "Pair 1 CUDA MSE should be near zero";
        EXPECT_LT(rocm_mse0, 1e-6) << "Pair 0 ROCm MSE should be near zero";
        EXPECT_LT(rocm_mse1, 1e-6) << "Pair 1 ROCm MSE should be near zero";

        // Cleanup
        freeCUDA(0, cuda_buf0);
        freeCUDA(1, cuda_buf1);
        backend_->freeBarBuffer(rocm_buf0);
        backend_->freeBarBuffer(rocm_buf1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Backward Compatibility Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Test that old initialize() API still works
     *
     * Verifies backward compatibility: the original DeviceGroup-based
     * initialize() should still work after adding multi-pair support.
     */
    TEST_F(Test__PCIeBarBackend_MultiPair, MultiPair_BackwardCompat_OldInitializeWorks)
    {
        REQUIRE_PCIE_BAR_P2P();
        REQUIRE_MIN_CUDA(1);
        REQUIRE_MIN_ROCM(1);

        backend_ = std::make_unique<PCIeBARBackend>();

        // Use the old DeviceGroup-based initialize() API
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("backward_compat_group")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .setLocalRank(0)
                         .build();

        bool result = backend_->initialize(group);
        EXPECT_TRUE(result) << "Old initialize() API should still work";

        EXPECT_TRUE(backend_->isInitialized());
        // Old API does NOT activate multi-pair mode
        EXPECT_FALSE(backend_->isMultiPairMode())
            << "Old initialize() should NOT set multi-pair mode";

        // Old allreduce should still work
        const size_t count = 256;
        const size_t bytes = count * sizeof(float);

        void *cuda_buf = allocateCUDA(0, bytes, 0.0f);
        ASSERT_NE(cuda_buf, nullptr);

        auto rocm_alloc = backend_->allocateInBarRegion(bytes);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;

        // Initialize
        std::vector<float> cuda_init(count, 5.0f);
        std::vector<float> rocm_init(count, 3.0f);
        cuda_backend_->hostToDevice(cuda_buf, cuda_init.data(), bytes, 0);
        cuda_backend_->synchronize(0);
        std::memcpy(rocm_buf, rocm_init.data(), bytes);

        // Register and allreduce using old API
        const std::string coll_id = "backward_compat_test";
        ASSERT_TRUE(backend_->registerBuffer(coll_id, DeviceId::cuda(0), cuda_buf, bytes));
        ASSERT_TRUE(backend_->registerBuffer(coll_id, DeviceId::rocm(0), rocm_buf, bytes));

        ASSERT_TRUE(backend_->allreduceRegistered(
            coll_id, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify result: 5.0 + 3.0 = 8.0
        auto cuda_result = readFromCUDA(0, cuda_buf, count);
        auto rocm_result = readFromBAR(rocm_buf, count);

        EXPECT_NEAR(cuda_result[0], 8.0f, 0.001f) << "Old API allreduce CUDA result mismatch";
        EXPECT_NEAR(rocm_result[0], 8.0f, 0.001f) << "Old API allreduce ROCm result mismatch";

        // Cleanup
        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    /**
     * @brief Test that multi-pair mode doesn't break single-pair allreduce
     *
     * When initialized via multi-pair API with a single pair, the
     * registered allreduce should also work (for backward compatibility
     * in code that uses both APIs).
     */
    TEST_F(Test__PCIeBarBackend_MultiPair, MultiPair_BackwardCompat_RegisteredAllreduceStillWorks)
    {
        REQUIRE_PCIE_BAR_P2P();
        REQUIRE_MIN_CUDA(1);
        REQUIRE_MIN_ROCM(1);

        backend_ = std::make_unique<PCIeBARBackend>();

        // Initialize via multi-pair API
        std::vector<DevicePair> pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0}};
        ASSERT_TRUE(backend_->initializeMultiPair(pairs));
        EXPECT_TRUE(backend_->isMultiPairMode());

        // Test that allreduceRegistered still works in multi-pair mode
        const size_t count = 256;
        const size_t bytes = count * sizeof(float);

        void *cuda_buf = allocateCUDA(0, bytes, 0.0f);
        ASSERT_NE(cuda_buf, nullptr);

        auto rocm_alloc = backend_->allocateInBarRegion(bytes);
        ASSERT_TRUE(rocm_alloc.has_value());
        auto [rocm_buf, offset] = *rocm_alloc;

        // Initialize with values
        std::vector<float> cuda_init(count, 11.0f);
        std::vector<float> rocm_init(count, 4.0f);
        cuda_backend_->hostToDevice(cuda_buf, cuda_init.data(), bytes, 0);
        cuda_backend_->synchronize(0);
        std::memcpy(rocm_buf, rocm_init.data(), bytes);

        // Register buffers
        const std::string coll_id = "multi_pair_compat_test";
        ASSERT_TRUE(backend_->registerBuffer(coll_id, DeviceId::cuda(0), cuda_buf, bytes));
        ASSERT_TRUE(backend_->registerBuffer(coll_id, DeviceId::rocm(0), rocm_buf, bytes));

        // Use allreduceRegistered (should still work in multi-pair mode)
        ASSERT_TRUE(backend_->allreduceRegistered(
            coll_id, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify result: 11.0 + 4.0 = 15.0
        auto cuda_result = readFromCUDA(0, cuda_buf, count);
        auto rocm_result = readFromBAR(rocm_buf, count);

        EXPECT_NEAR(cuda_result[0], 15.0f, 0.001f) << "Registered allreduce CUDA mismatch";
        EXPECT_NEAR(rocm_result[0], 15.0f, 0.001f) << "Registered allreduce ROCm mismatch";

        // Cleanup
        freeCUDA(0, cuda_buf);
        backend_->freeBarBuffer(rocm_buf);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Error Handling Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Test allreduceMultiPair with mismatched buffer counts
     *
     * Verifies that allreduceMultiPair properly validates that the
     * number of CUDA buffers matches the number of ROCm buffers and
     * the number of configured pairs.
     */
    TEST_F(Test__PCIeBarBackend_MultiPair, MultiPair_AllreduceMismatchedBufferCounts)
    {
        REQUIRE_PCIE_BAR_P2P();
        REQUIRE_MIN_CUDA(1);
        REQUIRE_MIN_ROCM(1);

        backend_ = std::make_unique<PCIeBARBackend>();

        // Initialize with one pair
        std::vector<DevicePair> pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0}};
        ASSERT_TRUE(backend_->initializeMultiPair(pairs));

        // Create mismatched buffer counts
        void *cuda_buf = allocateCUDA(0, TEST_BYTES, 1.0f);
        ASSERT_NE(cuda_buf, nullptr);

        // Two CUDA buffers, but only one pair configured
        std::vector<void *> cuda_buffers = {cuda_buf, cuda_buf}; // 2 buffers
        std::vector<void *> rocm_buffers = {nullptr};            // 1 buffer

        EXPECT_FALSE(backend_->allreduceMultiPair(
            cuda_buffers, rocm_buffers, TEST_COUNT, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair should fail with mismatched buffer counts";

        freeCUDA(0, cuda_buf);
    }

} // namespace llaminar2::test

#else // !HAVE_CUDA || !HAVE_ROCM

TEST(Test__PCIeBarBackend_MultiPair, RequiresCUDAAndROCm)
{
    GTEST_SKIP() << "PCIeBarBackend_MultiPair tests require both HAVE_CUDA and HAVE_ROCM";
}

#endif // HAVE_CUDA && HAVE_ROCM
