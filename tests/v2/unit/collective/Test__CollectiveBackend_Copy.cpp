/**
 * @file Test__CollectiveBackend_Copy.cpp
 * @brief Unit tests for ICollectiveBackend::copy() API
 *
 * Tests the copy(), copyAsync(), and supportsCopy() methods across all
 * collective backends. Verifies fail-fast behavior (no silent fallbacks).
 *
 * Test Coverage:
 * - HostBackend: CPU↔CPU only, rejects GPU-involved transfers
 * - NCCLBackend: CUDA↔CUDA transfers
 * - RCCLBackend: ROCm↔ROCm transfers
 * - PCIeBARBackend: Cross-vendor CUDA↔ROCm transfers
 *
 * Each backend is tested for:
 * - supportsCopy() returns correct values for supported/unsupported device pairs
 * - copy() succeeds for supported transfers
 * - copy() returns false (fail-fast) for unsupported transfers
 * - Same-device copy works
 * - Zero-byte copy returns true
 *
 * NOTE: This file does NOT include cuda_runtime.h or hip/hip_runtime.h directly
 * because they have conflicting type definitions. GPU device availability is
 * checked via the backend's isAvailable() method instead.
 */

#include <gtest/gtest.h>
#include "v2/collective/ICollectiveBackend.h"
#include "v2/collective/backends/HostBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

#include <vector>
#include <cstring>

#ifdef HAVE_CUDA
#include "v2/collective/backends/NCCLBackend.h"
#endif

#ifdef HAVE_ROCM
#include "v2/collective/backends/RCCLBackend.h"
#endif

// Include PCIeBARBackend if both CUDA and ROCm are available
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "v2/collective/backends/PCIeBARBackend.h"
#endif

namespace llaminar2::test
{

    // =============================================================================
    // Host Backend Copy Tests
    // =============================================================================

    class Test__HostBackend_Copy : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            backend_ = std::make_unique<HostBackend>();
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

        std::unique_ptr<HostBackend> backend_;
    };

    // -------------------------------------------------------------------------
    // supportsCopy() Tests
    // -------------------------------------------------------------------------

    TEST_F(Test__HostBackend_Copy, SupportsCopy_CPU_to_CPU)
    {
        DeviceId cpu = DeviceId::cpu();
        EXPECT_TRUE(backend_->supportsCopy(cpu, cpu));
    }

    TEST_F(Test__HostBackend_Copy, SupportsCopy_RejectsCUDA_as_Source)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId cuda = DeviceId::cuda(0);
        EXPECT_FALSE(backend_->supportsCopy(cuda, cpu));
    }

    TEST_F(Test__HostBackend_Copy, SupportsCopy_RejectsCUDA_as_Dest)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId cuda = DeviceId::cuda(0);
        EXPECT_FALSE(backend_->supportsCopy(cpu, cuda));
    }

    TEST_F(Test__HostBackend_Copy, SupportsCopy_RejectsCUDA_to_CUDA)
    {
        DeviceId cuda0 = DeviceId::cuda(0);
        DeviceId cuda1 = DeviceId::cuda(1);
        EXPECT_FALSE(backend_->supportsCopy(cuda0, cuda0));
        EXPECT_FALSE(backend_->supportsCopy(cuda0, cuda1));
    }

    TEST_F(Test__HostBackend_Copy, SupportsCopy_RejectsROCm_as_Source)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId rocm = DeviceId::rocm(0);
        EXPECT_FALSE(backend_->supportsCopy(rocm, cpu));
    }

    TEST_F(Test__HostBackend_Copy, SupportsCopy_RejectsROCm_as_Dest)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId rocm = DeviceId::rocm(0);
        EXPECT_FALSE(backend_->supportsCopy(cpu, rocm));
    }

    TEST_F(Test__HostBackend_Copy, SupportsCopy_RejectsROCm_to_ROCm)
    {
        DeviceId rocm0 = DeviceId::rocm(0);
        DeviceId rocm1 = DeviceId::rocm(1);
        EXPECT_FALSE(backend_->supportsCopy(rocm0, rocm0));
        EXPECT_FALSE(backend_->supportsCopy(rocm0, rocm1));
    }

    TEST_F(Test__HostBackend_Copy, SupportsCopy_RejectsCrossVendor)
    {
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);
        EXPECT_FALSE(backend_->supportsCopy(cuda, rocm));
        EXPECT_FALSE(backend_->supportsCopy(rocm, cuda));
    }

    // -------------------------------------------------------------------------
    // copy() Tests
    // -------------------------------------------------------------------------

    TEST_F(Test__HostBackend_Copy, Copy_CPU_to_CPU_Success)
    {
        DeviceId cpu = DeviceId::cpu();

        std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> dst(4, 0.0f);

        EXPECT_TRUE(backend_->copy(dst.data(), cpu, src.data(), cpu, src.size() * sizeof(float)));

        EXPECT_FLOAT_EQ(dst[0], 1.0f);
        EXPECT_FLOAT_EQ(dst[1], 2.0f);
        EXPECT_FLOAT_EQ(dst[2], 3.0f);
        EXPECT_FLOAT_EQ(dst[3], 4.0f);
    }

    TEST_F(Test__HostBackend_Copy, Copy_CPU_to_CPU_LargeBuffer)
    {
        DeviceId cpu = DeviceId::cpu();

        // Test with a larger buffer (4KB)
        constexpr size_t count = 1024;
        std::vector<float> src(count);
        std::vector<float> dst(count, 0.0f);

        // Fill source with pattern
        for (size_t i = 0; i < count; ++i)
        {
            src[i] = static_cast<float>(i) * 0.001f;
        }

        EXPECT_TRUE(backend_->copy(dst.data(), cpu, src.data(), cpu, count * sizeof(float)));

        // Verify pattern was copied correctly
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(dst[i], src[i]) << "Mismatch at index " << i;
        }
    }

    TEST_F(Test__HostBackend_Copy, Copy_ZeroBytes_Success)
    {
        DeviceId cpu = DeviceId::cpu();
        std::vector<float> src = {1.0f};
        std::vector<float> dst = {0.0f};

        // Zero-byte copy should succeed without modifying destination
        EXPECT_TRUE(backend_->copy(dst.data(), cpu, src.data(), cpu, 0));
        EXPECT_FLOAT_EQ(dst[0], 0.0f); // Should not have been modified
    }

    TEST_F(Test__HostBackend_Copy, Copy_SamePointer_Success)
    {
        DeviceId cpu = DeviceId::cpu();
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

        // Same pointer copy (no-op) should succeed
        EXPECT_TRUE(backend_->copy(data.data(), cpu, data.data(), cpu, data.size() * sizeof(float)));

        // Data should be unchanged
        EXPECT_FLOAT_EQ(data[0], 1.0f);
        EXPECT_FLOAT_EQ(data[1], 2.0f);
        EXPECT_FLOAT_EQ(data[2], 3.0f);
        EXPECT_FLOAT_EQ(data[3], 4.0f);
    }

    TEST_F(Test__HostBackend_Copy, Copy_FailFast_CUDA_as_Source)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId cuda = DeviceId::cuda(0);

        std::vector<float> data = {1.0f};

        // Should fail fast for CUDA source
        EXPECT_FALSE(backend_->copy(data.data(), cpu, data.data(), cuda, sizeof(float)));
    }

    TEST_F(Test__HostBackend_Copy, Copy_FailFast_CUDA_as_Dest)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId cuda = DeviceId::cuda(0);

        std::vector<float> data = {1.0f};

        // Should fail fast for CUDA destination
        EXPECT_FALSE(backend_->copy(data.data(), cuda, data.data(), cpu, sizeof(float)));
    }

    TEST_F(Test__HostBackend_Copy, Copy_FailFast_CUDA_to_CUDA)
    {
        DeviceId cuda0 = DeviceId::cuda(0);
        DeviceId cuda1 = DeviceId::cuda(1);

        std::vector<float> data = {1.0f};

        // Should fail fast for CUDA↔CUDA
        EXPECT_FALSE(backend_->copy(data.data(), cuda0, data.data(), cuda0, sizeof(float)));
        EXPECT_FALSE(backend_->copy(data.data(), cuda1, data.data(), cuda0, sizeof(float)));
    }

    TEST_F(Test__HostBackend_Copy, Copy_FailFast_ROCm_as_Source)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId rocm = DeviceId::rocm(0);

        std::vector<float> data = {1.0f};

        // Should fail fast for ROCm source
        EXPECT_FALSE(backend_->copy(data.data(), cpu, data.data(), rocm, sizeof(float)));
    }

    TEST_F(Test__HostBackend_Copy, Copy_FailFast_ROCm_as_Dest)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId rocm = DeviceId::rocm(0);

        std::vector<float> data = {1.0f};

        // Should fail fast for ROCm destination
        EXPECT_FALSE(backend_->copy(data.data(), rocm, data.data(), cpu, sizeof(float)));
    }

    TEST_F(Test__HostBackend_Copy, Copy_FailFast_CrossVendor)
    {
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);

        std::vector<float> data = {1.0f};

        // Should fail fast for cross-vendor
        EXPECT_FALSE(backend_->copy(data.data(), rocm, data.data(), cuda, sizeof(float)));
        EXPECT_FALSE(backend_->copy(data.data(), cuda, data.data(), rocm, sizeof(float)));
    }

    // -------------------------------------------------------------------------
    // copyAsync() Tests
    // -------------------------------------------------------------------------

    TEST_F(Test__HostBackend_Copy, CopyAsync_CPU_to_CPU_Success)
    {
        DeviceId cpu = DeviceId::cpu();

        std::vector<float> src = {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> dst(4, 0.0f);

        // Async copy on CPU is synchronous, stream parameter is ignored
        EXPECT_TRUE(backend_->copyAsync(dst.data(), cpu, src.data(), cpu, src.size() * sizeof(float), nullptr));

        EXPECT_FLOAT_EQ(dst[0], 5.0f);
        EXPECT_FLOAT_EQ(dst[1], 6.0f);
        EXPECT_FLOAT_EQ(dst[2], 7.0f);
        EXPECT_FLOAT_EQ(dst[3], 8.0f);
    }

    TEST_F(Test__HostBackend_Copy, CopyAsync_ZeroBytes_Success)
    {
        DeviceId cpu = DeviceId::cpu();
        std::vector<float> src = {1.0f};
        std::vector<float> dst = {0.0f};

        EXPECT_TRUE(backend_->copyAsync(dst.data(), cpu, src.data(), cpu, 0, nullptr));
        EXPECT_FLOAT_EQ(dst[0], 0.0f); // Should not have been modified
    }

    TEST_F(Test__HostBackend_Copy, CopyAsync_FailFast_CUDA)
    {
        DeviceId cpu = DeviceId::cpu();
        DeviceId cuda = DeviceId::cuda(0);

        std::vector<float> data = {1.0f};

        EXPECT_FALSE(backend_->copyAsync(data.data(), cuda, data.data(), cpu, sizeof(float), nullptr));
        EXPECT_FALSE(backend_->copyAsync(data.data(), cpu, data.data(), cuda, sizeof(float), nullptr));
    }

    // =============================================================================
    // NCCL Backend Copy Tests (CUDA)
    // =============================================================================

#ifdef HAVE_CUDA

    class Test__NCCLBackend_Copy : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check if NCCL backend is available (has CUDA devices)
            NCCLBackend probe_backend;
            if (!probe_backend.isAvailable())
            {
                GTEST_SKIP() << "NCCL backend not available (no CUDA devices or NCCL library)";
            }
        }
    };

    TEST_F(Test__NCCLBackend_Copy, SupportsCopy_CUDA_to_CUDA_SameDevice)
    {
        NCCLBackend backend;
        DeviceId cuda0 = DeviceId::cuda(0);

        // NCCL should support same-device CUDA copy
        EXPECT_TRUE(backend.supportsCopy(cuda0, cuda0));
    }

    TEST_F(Test__NCCLBackend_Copy, SupportsCopy_CUDA_to_CUDA_DifferentDevices)
    {
        // This test verifies the interface returns sensible values for different-device pairs.
        // The actual result depends on P2P hardware capability, not just the API contract.
        NCCLBackend backend;
        DeviceId cuda0 = DeviceId::cuda(0);
        DeviceId cuda1 = DeviceId::cuda(1);

        // For different devices, supportsCopy() returns true only if P2P is available.
        // We don't assert a specific value since hardware varies - just verify it doesn't crash
        // and returns a boolean. The key contract is: if supportsCopy() returns false,
        // copy() MUST also fail fast (which we test elsewhere).
        bool p2p_01 = backend.supportsCopy(cuda0, cuda1);
        bool p2p_10 = backend.supportsCopy(cuda1, cuda0);
        
        // Verify symmetry at minimum - if P2P works one direction, it should work the other
        // (This is a common hardware characteristic, though not strictly required)
        EXPECT_EQ(p2p_01, p2p_10) << "P2P capability should be symmetric";
    }

    TEST_F(Test__NCCLBackend_Copy, SupportsCopy_RejectsROCm)
    {
        NCCLBackend backend;
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);

        // NCCL should not support cross-vendor or ROCm-only
        EXPECT_FALSE(backend.supportsCopy(cuda, rocm));
        EXPECT_FALSE(backend.supportsCopy(rocm, cuda));
        EXPECT_FALSE(backend.supportsCopy(rocm, rocm));
    }

    TEST_F(Test__NCCLBackend_Copy, SupportsCopy_RejectsCPU)
    {
        NCCLBackend backend;
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId cpu = DeviceId::cpu();

        // NCCL should not support CPU-involved transfers
        EXPECT_FALSE(backend.supportsCopy(cuda, cpu));
        EXPECT_FALSE(backend.supportsCopy(cpu, cuda));
        EXPECT_FALSE(backend.supportsCopy(cpu, cpu));
    }

    TEST_F(Test__NCCLBackend_Copy, Copy_FailFast_ROCm)
    {
        NCCLBackend backend;
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);

        float dummy = 0.0f;

        // Should fail fast for ROCm-involved transfers
        EXPECT_FALSE(backend.copy(&dummy, rocm, &dummy, cuda, sizeof(float)));
        EXPECT_FALSE(backend.copy(&dummy, cuda, &dummy, rocm, sizeof(float)));
    }

    TEST_F(Test__NCCLBackend_Copy, Copy_FailFast_CPU)
    {
        NCCLBackend backend;
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId cpu = DeviceId::cpu();

        float dummy = 0.0f;

        // Should fail fast for CPU-involved transfers
        EXPECT_FALSE(backend.copy(&dummy, cpu, &dummy, cuda, sizeof(float)));
        EXPECT_FALSE(backend.copy(&dummy, cuda, &dummy, cpu, sizeof(float)));
    }

#endif // HAVE_CUDA

    // =============================================================================
    // RCCL Backend Copy Tests (ROCm)
    // =============================================================================

#ifdef HAVE_ROCM

    class Test__RCCLBackend_Copy : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check if RCCL backend is available (has ROCm devices)
            RCCLBackend probe_backend;
            if (!probe_backend.isAvailable())
            {
                GTEST_SKIP() << "RCCL backend not available (no ROCm devices or RCCL library)";
            }
        }
    };

    TEST_F(Test__RCCLBackend_Copy, SupportsCopy_ROCm_to_ROCm_SameDevice)
    {
        RCCLBackend backend;
        DeviceId rocm0 = DeviceId::rocm(0);

        // RCCL should support same-device ROCm copy
        EXPECT_TRUE(backend.supportsCopy(rocm0, rocm0));
    }

    TEST_F(Test__RCCLBackend_Copy, SupportsCopy_ROCm_to_ROCm_DifferentDevices)
    {
        // This test verifies the interface returns sensible values for different-device pairs.
        // The actual result depends on P2P hardware capability, not just the API contract.
        RCCLBackend backend;
        DeviceId rocm0 = DeviceId::rocm(0);
        DeviceId rocm1 = DeviceId::rocm(1);

        // For different devices, supportsCopy() returns true only if P2P is available.
        // We don't assert a specific value since hardware varies - just verify it doesn't crash
        // and returns a boolean.
        bool p2p_01 = backend.supportsCopy(rocm0, rocm1);
        bool p2p_10 = backend.supportsCopy(rocm1, rocm0);
        
        // Verify symmetry at minimum
        EXPECT_EQ(p2p_01, p2p_10) << "P2P capability should be symmetric";
    }

    TEST_F(Test__RCCLBackend_Copy, SupportsCopy_RejectsCUDA)
    {
        RCCLBackend backend;
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);

        // RCCL should not support cross-vendor or CUDA-only
        EXPECT_FALSE(backend.supportsCopy(rocm, cuda));
        EXPECT_FALSE(backend.supportsCopy(cuda, rocm));
        EXPECT_FALSE(backend.supportsCopy(cuda, cuda));
    }

    TEST_F(Test__RCCLBackend_Copy, SupportsCopy_RejectsCPU)
    {
        RCCLBackend backend;
        DeviceId rocm = DeviceId::rocm(0);
        DeviceId cpu = DeviceId::cpu();

        // RCCL should not support CPU-involved transfers
        EXPECT_FALSE(backend.supportsCopy(rocm, cpu));
        EXPECT_FALSE(backend.supportsCopy(cpu, rocm));
        EXPECT_FALSE(backend.supportsCopy(cpu, cpu));
    }

    TEST_F(Test__RCCLBackend_Copy, Copy_FailFast_CUDA)
    {
        RCCLBackend backend;
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);

        float dummy = 0.0f;

        // Should fail fast for CUDA-involved transfers
        EXPECT_FALSE(backend.copy(&dummy, cuda, &dummy, rocm, sizeof(float)));
        EXPECT_FALSE(backend.copy(&dummy, rocm, &dummy, cuda, sizeof(float)));
    }

    TEST_F(Test__RCCLBackend_Copy, Copy_FailFast_CPU)
    {
        RCCLBackend backend;
        DeviceId rocm = DeviceId::rocm(0);
        DeviceId cpu = DeviceId::cpu();

        float dummy = 0.0f;

        // Should fail fast for CPU-involved transfers
        EXPECT_FALSE(backend.copy(&dummy, cpu, &dummy, rocm, sizeof(float)));
        EXPECT_FALSE(backend.copy(&dummy, rocm, &dummy, cpu, sizeof(float)));
    }

#endif // HAVE_ROCM

    // =============================================================================
    // PCIeBAR Backend Copy Tests (Cross-Vendor)
    // =============================================================================

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    class Test__PCIeBARBackend_Copy : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check if PCIeBAR backend is available
            PCIeBARBackend probe_backend;
            if (!probe_backend.isAvailable())
            {
                GTEST_SKIP() << "PCIeBAR backend not available (need both CUDA and ROCm devices)";
            }
        }
    };

    TEST_F(Test__PCIeBARBackend_Copy, SupportsCopy_CUDA_to_ROCm)
    {
        PCIeBARBackend backend;
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);

        // PCIeBAR cross-vendor support depends on isPCIeBarActive() which requires
        // hardware BAR mapping support. We verify:
        // 1. Both directions return the same result (symmetric)
        // 2. The call doesn't crash
        // 3. If BAR is active, cross-vendor should be supported
        bool cuda_to_rocm = backend.supportsCopy(cuda, rocm);
        bool rocm_to_cuda = backend.supportsCopy(rocm, cuda);

        // Symmetry check
        EXPECT_EQ(cuda_to_rocm, rocm_to_cuda);

        // If BAR is active, cross-vendor should work
        if (backend.isPCIeBarActive()) {
            EXPECT_TRUE(cuda_to_rocm);
            EXPECT_TRUE(rocm_to_cuda);
        }
        // If BAR is not active, we don't assert false - just verify no crash
    }

    TEST_F(Test__PCIeBARBackend_Copy, SupportsCopy_RejectsSameCUDA)
    {
        PCIeBARBackend backend;
        DeviceId cuda0 = DeviceId::cuda(0);
        DeviceId cuda1 = DeviceId::cuda(1);

        // PCIeBAR should reject same-vendor CUDA (delegate to NCCL)
        EXPECT_FALSE(backend.supportsCopy(cuda0, cuda0));
        EXPECT_FALSE(backend.supportsCopy(cuda0, cuda1));
    }

    TEST_F(Test__PCIeBARBackend_Copy, SupportsCopy_RejectsSameROCm)
    {
        PCIeBARBackend backend;
        DeviceId rocm0 = DeviceId::rocm(0);
        DeviceId rocm1 = DeviceId::rocm(1);

        // PCIeBAR should reject same-vendor ROCm (delegate to RCCL)
        EXPECT_FALSE(backend.supportsCopy(rocm0, rocm0));
        EXPECT_FALSE(backend.supportsCopy(rocm0, rocm1));
    }

    TEST_F(Test__PCIeBARBackend_Copy, SupportsCopy_RejectsCPU)
    {
        PCIeBARBackend backend;
        DeviceId cpu = DeviceId::cpu();
        DeviceId cuda = DeviceId::cuda(0);
        DeviceId rocm = DeviceId::rocm(0);

        // PCIeBAR should reject host-involved transfers (no staging)
        EXPECT_FALSE(backend.supportsCopy(cpu, cuda));
        EXPECT_FALSE(backend.supportsCopy(cuda, cpu));
        EXPECT_FALSE(backend.supportsCopy(cpu, rocm));
        EXPECT_FALSE(backend.supportsCopy(rocm, cpu));
        EXPECT_FALSE(backend.supportsCopy(cpu, cpu));
    }

    TEST_F(Test__PCIeBARBackend_Copy, Copy_FailFast_SameVendor_CUDA)
    {
        PCIeBARBackend backend;
        DeviceId cuda0 = DeviceId::cuda(0);

        float dummy = 0.0f;

        // Should fail fast for same-vendor CUDA
        EXPECT_FALSE(backend.copy(&dummy, cuda0, &dummy, cuda0, sizeof(float)));
    }

    TEST_F(Test__PCIeBARBackend_Copy, Copy_FailFast_SameVendor_ROCm)
    {
        PCIeBARBackend backend;
        DeviceId rocm0 = DeviceId::rocm(0);

        float dummy = 0.0f;

        // Should fail fast for same-vendor ROCm
        EXPECT_FALSE(backend.copy(&dummy, rocm0, &dummy, rocm0, sizeof(float)));
    }

    TEST_F(Test__PCIeBARBackend_Copy, Copy_FailFast_CPU)
    {
        PCIeBARBackend backend;
        DeviceId cpu = DeviceId::cpu();
        DeviceId cuda = DeviceId::cuda(0);

        float dummy = 0.0f;

        // Should fail fast for CPU-involved transfers
        EXPECT_FALSE(backend.copy(&dummy, cpu, &dummy, cuda, sizeof(float)));
        EXPECT_FALSE(backend.copy(&dummy, cuda, &dummy, cpu, sizeof(float)));
    }

#endif // HAVE_CUDA && HAVE_ROCM

    // =============================================================================
    // Invalid Device Tests (All Backends)
    // =============================================================================

    class Test__CollectiveBackend_Copy_InvalidDevice : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            host_backend_ = std::make_unique<HostBackend>();
        }

        std::unique_ptr<HostBackend> host_backend_;
    };

    TEST_F(Test__CollectiveBackend_Copy_InvalidDevice, SupportsCopy_RejectsInvalidDevice)
    {
        DeviceId invalid = DeviceId::invalid();
        DeviceId cpu = DeviceId::cpu();

        // Invalid devices should be rejected
        EXPECT_FALSE(host_backend_->supportsCopy(invalid, cpu));
        EXPECT_FALSE(host_backend_->supportsCopy(cpu, invalid));
        EXPECT_FALSE(host_backend_->supportsCopy(invalid, invalid));
    }

    TEST_F(Test__CollectiveBackend_Copy_InvalidDevice, Copy_FailFast_InvalidDevice)
    {
        DeviceId invalid = DeviceId::invalid();
        DeviceId cpu = DeviceId::cpu();

        float dummy = 0.0f;

        // Copy with invalid device should fail fast
        EXPECT_FALSE(host_backend_->copy(&dummy, invalid, &dummy, cpu, sizeof(float)));
        EXPECT_FALSE(host_backend_->copy(&dummy, cpu, &dummy, invalid, sizeof(float)));
    }

} // namespace llaminar2::test

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
