/**
 * @file Test__BackendSelector.cpp
 * @brief Unit tests for BackendSelector
 *
 * Tests automatic backend selection for PP transfers and TP domains.
 * Covers:
 * - PP transfer backend selection (CUDA↔CUDA, ROCm↔ROCm, cross-vendor, CPU)
 * - TP domain backend selection (homogeneous, heterogeneous, mixed)
 * - Backend availability checking (compiled, usable)
 * - Utility methods (isHomogeneous, hasGPU, isCrossVendor, resolve)
 */

#include <gtest/gtest.h>
#include "config/BackendSelector.h"
#include "backends/DeviceId.h"

namespace llaminar2
{
    namespace
    {

        class Test__BackendSelector : public ::testing::Test
        {
        protected:
            // Convenience device IDs for test cases
            DeviceId cuda0 = DeviceId::cuda(0);
            DeviceId cuda1 = DeviceId::cuda(1);
            DeviceId rocm0 = DeviceId::rocm(0);
            DeviceId rocm1 = DeviceId::rocm(1);
            DeviceId cpu0 = DeviceId::cpu();
        };

        // ============================================================================
        // PP Transfer Backend Selection Tests
        // ============================================================================

        TEST_F(Test__BackendSelector, TransferCudaToCudaSelectsNCCL)
        {
            auto backend = BackendSelector::selectForTransfer(cuda0, cuda1);
            EXPECT_EQ(backend, CollectiveBackendType::NCCL);
        }

        TEST_F(Test__BackendSelector, TransferRocmToRocmSelectsRCCL)
        {
            auto backend = BackendSelector::selectForTransfer(rocm0, rocm1);
            EXPECT_EQ(backend, CollectiveBackendType::RCCL);
        }

        TEST_F(Test__BackendSelector, TransferCudaToRocmSelectsHETEROGENEOUS)
        {
            auto backend = BackendSelector::selectForTransfer(cuda0, rocm0);
            EXPECT_EQ(backend, CollectiveBackendType::HETEROGENEOUS);
        }

        TEST_F(Test__BackendSelector, TransferRocmToCudaSelectsHETEROGENEOUS)
        {
            auto backend = BackendSelector::selectForTransfer(rocm0, cuda0);
            EXPECT_EQ(backend, CollectiveBackendType::HETEROGENEOUS);
        }

        TEST_F(Test__BackendSelector, TransferCpuToCpuSelectsHOST)
        {
            auto backend = BackendSelector::selectForTransfer(cpu0, cpu0);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);
        }

        TEST_F(Test__BackendSelector, TransferGpuToCpuSelectsHOST)
        {
            // CUDA → CPU should use HOST (staging through host memory)
            auto backend = BackendSelector::selectForTransfer(cuda0, cpu0);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);

            // ROCm → CPU should also use HOST
            backend = BackendSelector::selectForTransfer(rocm0, cpu0);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);
        }

        TEST_F(Test__BackendSelector, TransferCpuToGpuSelectsHOST)
        {
            // CPU → CUDA should use HOST (staging through host memory)
            auto backend = BackendSelector::selectForTransfer(cpu0, cuda0);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);

            // CPU → ROCm should also use HOST
            backend = BackendSelector::selectForTransfer(cpu0, rocm0);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);
        }

        // Test DeviceType overload for convenience
        TEST_F(Test__BackendSelector, TransferByDeviceTypeWorks)
        {
            auto backend = BackendSelector::selectForTransfer(DeviceType::CUDA, DeviceType::CUDA);
            EXPECT_EQ(backend, CollectiveBackendType::NCCL);

            backend = BackendSelector::selectForTransfer(DeviceType::ROCm, DeviceType::ROCm);
            EXPECT_EQ(backend, CollectiveBackendType::RCCL);

            backend = BackendSelector::selectForTransfer(DeviceType::CUDA, DeviceType::ROCm);
            EXPECT_EQ(backend, CollectiveBackendType::HETEROGENEOUS);

            backend = BackendSelector::selectForTransfer(DeviceType::CPU, DeviceType::CUDA);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);
        }

        // ============================================================================
        // TP Domain Backend Selection Tests
        // ============================================================================

        TEST_F(Test__BackendSelector, TPDomainAllCudaSelectsNCCL)
        {
            std::vector<DeviceId> devices = {cuda0, cuda1};
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::NCCL);
        }

        TEST_F(Test__BackendSelector, TPDomainAllRocmSelectsRCCL)
        {
            std::vector<DeviceId> devices = {rocm0, rocm1};
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::RCCL);
        }

        TEST_F(Test__BackendSelector, TPDomainMixedGpuTwoDevicesSelectsHETEROGENEOUS)
        {
            std::vector<DeviceId> devices = {cuda0, rocm0};
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HETEROGENEOUS);
        }

        TEST_F(Test__BackendSelector, TPDomainMixedGpuThreeDevicesSelectsHETEROGENEOUS)
        {
            std::vector<DeviceId> devices = {cuda0, cuda1, rocm0};
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HETEROGENEOUS);
        }

        TEST_F(Test__BackendSelector, TPDomainMixedGpuFourDevicesSelectsHETEROGENEOUS)
        {
            // 4 devices: 2 CUDA + 2 ROCm
            std::vector<DeviceId> devices = {cuda0, cuda1, rocm0, rocm1};
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HETEROGENEOUS);
        }

        TEST_F(Test__BackendSelector, TPDomainAllCpuSelectsMPI)
        {
            std::vector<DeviceId> devices = {cpu0, cpu0}; // Two CPU "devices"
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::MPI);
        }

        TEST_F(Test__BackendSelector, TPDomainSingleDeviceSelectsHOST)
        {
            // Single CUDA device
            std::vector<DeviceId> devices = {cuda0};
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);

            // Single ROCm device
            devices = {rocm0};
            backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);

            // Single CPU device
            devices = {cpu0};
            backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);
        }

        TEST_F(Test__BackendSelector, TPDomainEmptySelectsHOST)
        {
            std::vector<DeviceId> devices;
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);
        }

        TEST_F(Test__BackendSelector, TPDomainMixedGpuCpuSelectsHOST)
        {
            // GPU + CPU combination should use HOST (staging through host memory)
            std::vector<DeviceId> devices = {cuda0, cpu0};
            auto backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);

            devices = {rocm0, cpu0};
            backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);

            // Multiple GPUs + CPU
            devices = {cuda0, cuda1, cpu0};
            backend = BackendSelector::selectForTPDomain(devices);
            EXPECT_EQ(backend, CollectiveBackendType::HOST);
        }

        // ============================================================================
        // Backend Availability Tests
        // ============================================================================

        TEST_F(Test__BackendSelector, IsBackendCompiledForHOSTReturnsTrue)
        {
            // HOST backend should always be available
            EXPECT_TRUE(BackendSelector::isBackendCompiled(CollectiveBackendType::HOST));
        }

        TEST_F(Test__BackendSelector, IsBackendCompiledForMPIReturnsTrue)
        {
            // MPI backend should always be available (we build with MPI)
            EXPECT_TRUE(BackendSelector::isBackendCompiled(CollectiveBackendType::MPI));
        }

        TEST_F(Test__BackendSelector, IsBackendCompiledForAUTOReturnsTrue)
        {
            // AUTO is a meta-backend, should return true (it's always "available")
            EXPECT_TRUE(BackendSelector::isBackendCompiled(CollectiveBackendType::AUTO));
        }

        TEST_F(Test__BackendSelector, IsBackendUsableForNCCLRequiresAllCuda)
        {
            // NCCL with all CUDA should be usable (if compiled)
            std::vector<DeviceId> cuda_devices = {cuda0, cuda1};
            bool usable = BackendSelector::isBackendUsable(CollectiveBackendType::NCCL, cuda_devices);
            bool compiled = BackendSelector::isBackendCompiled(CollectiveBackendType::NCCL);
            if (compiled)
            {
                EXPECT_TRUE(usable);
            }
            else
            {
                EXPECT_FALSE(usable);
            }

            // NCCL with mixed devices should not be usable (wrong device types)
            std::vector<DeviceId> mixed_devices = {cuda0, rocm0};
            EXPECT_FALSE(BackendSelector::isBackendUsable(CollectiveBackendType::NCCL, mixed_devices));

            // NCCL with ROCm-only should not be usable
            std::vector<DeviceId> rocm_devices = {rocm0, rocm1};
            EXPECT_FALSE(BackendSelector::isBackendUsable(CollectiveBackendType::NCCL, rocm_devices));
        }

        TEST_F(Test__BackendSelector, IsBackendUsableForRCCLRequiresAllRocm)
        {
            // RCCL with all ROCm should be usable (if compiled)
            std::vector<DeviceId> rocm_devices = {rocm0, rocm1};
            bool usable = BackendSelector::isBackendUsable(CollectiveBackendType::RCCL, rocm_devices);
            bool compiled = BackendSelector::isBackendCompiled(CollectiveBackendType::RCCL);
            if (compiled)
            {
                EXPECT_TRUE(usable);
            }
            else
            {
                EXPECT_FALSE(usable);
            }

            // RCCL with CUDA devices should not be usable
            std::vector<DeviceId> cuda_devices = {cuda0, cuda1};
            EXPECT_FALSE(BackendSelector::isBackendUsable(CollectiveBackendType::RCCL, cuda_devices));

            // RCCL with mixed devices should not be usable
            std::vector<DeviceId> mixed_devices = {cuda0, rocm0};
            EXPECT_FALSE(BackendSelector::isBackendUsable(CollectiveBackendType::RCCL, mixed_devices));
        }

        TEST_F(Test__BackendSelector, IsBackendUsableForHOSTAlwaysTrue)
        {
            // HOST backend should always be usable regardless of devices
            std::vector<DeviceId> any_devices = {cuda0, rocm0, cpu0};
            EXPECT_TRUE(BackendSelector::isBackendUsable(CollectiveBackendType::HOST, any_devices));

            std::vector<DeviceId> empty;
            EXPECT_TRUE(BackendSelector::isBackendUsable(CollectiveBackendType::HOST, empty));

            std::vector<DeviceId> single = {cuda0};
            EXPECT_TRUE(BackendSelector::isBackendUsable(CollectiveBackendType::HOST, single));
        }

        TEST_F(Test__BackendSelector, IsBackendUsableForAUTOReturnsFalse)
        {
            // AUTO is not a concrete backend - it needs to be resolved first
            std::vector<DeviceId> devices = {cuda0, cuda1};
            EXPECT_FALSE(BackendSelector::isBackendUsable(CollectiveBackendType::AUTO, devices));
        }

        // ============================================================================
        // Resolve Tests
        // ============================================================================

        TEST_F(Test__BackendSelector, ResolveAUTOSelectsAppropriateBackend)
        {
            // AUTO with all CUDA should resolve to NCCL
            std::vector<DeviceId> cuda_devices = {cuda0, cuda1};
            auto resolved = BackendSelector::resolve(CollectiveBackendType::AUTO, cuda_devices);
            EXPECT_EQ(resolved, CollectiveBackendType::NCCL);

            // AUTO with all ROCm should resolve to RCCL
            std::vector<DeviceId> rocm_devices = {rocm0, rocm1};
            resolved = BackendSelector::resolve(CollectiveBackendType::AUTO, rocm_devices);
            EXPECT_EQ(resolved, CollectiveBackendType::RCCL);

            // AUTO with mixed should resolve to HETEROGENEOUS (2 devices)
            std::vector<DeviceId> mixed_devices = {cuda0, rocm0};
            resolved = BackendSelector::resolve(CollectiveBackendType::AUTO, mixed_devices);
            EXPECT_EQ(resolved, CollectiveBackendType::HETEROGENEOUS);

            // AUTO with single device should resolve to HOST
            std::vector<DeviceId> single = {cuda0};
            resolved = BackendSelector::resolve(CollectiveBackendType::AUTO, single);
            EXPECT_EQ(resolved, CollectiveBackendType::HOST);
        }

        TEST_F(Test__BackendSelector, ResolveExplicitBackendReturnsUnchanged)
        {
            std::vector<DeviceId> devices = {cuda0, cuda1};

            // Explicit NCCL should return NCCL
            auto resolved = BackendSelector::resolve(CollectiveBackendType::NCCL, devices);
            EXPECT_EQ(resolved, CollectiveBackendType::NCCL);

            // Explicit HOST should return HOST
            resolved = BackendSelector::resolve(CollectiveBackendType::HOST, devices);
            EXPECT_EQ(resolved, CollectiveBackendType::HOST);

            // Even if backend doesn't match devices, resolve returns it unchanged
            // (validation happens elsewhere)
            resolved = BackendSelector::resolve(CollectiveBackendType::RCCL, devices);
            EXPECT_EQ(resolved, CollectiveBackendType::RCCL);

            resolved = BackendSelector::resolve(CollectiveBackendType::MPI, devices);
            EXPECT_EQ(resolved, CollectiveBackendType::MPI);
        }

        // ============================================================================
        // Utility Method Tests: isHomogeneous
        // ============================================================================

        TEST_F(Test__BackendSelector, IsHomogeneousWithSameTypes)
        {
            // All CUDA should be homogeneous
            std::vector<DeviceId> all_cuda = {cuda0, cuda1};
            EXPECT_TRUE(BackendSelector::isHomogeneous(all_cuda));

            // All ROCm should be homogeneous
            std::vector<DeviceId> all_rocm = {rocm0, rocm1};
            EXPECT_TRUE(BackendSelector::isHomogeneous(all_rocm));

            // All CPU should be homogeneous
            std::vector<DeviceId> all_cpu = {cpu0, cpu0};
            EXPECT_TRUE(BackendSelector::isHomogeneous(all_cpu));
        }

        TEST_F(Test__BackendSelector, IsHomogeneousWithMixedTypes)
        {
            // CUDA + ROCm should not be homogeneous
            std::vector<DeviceId> mixed = {cuda0, rocm0};
            EXPECT_FALSE(BackendSelector::isHomogeneous(mixed));

            // CUDA + CPU should not be homogeneous
            mixed = {cuda0, cpu0};
            EXPECT_FALSE(BackendSelector::isHomogeneous(mixed));

            // ROCm + CPU should not be homogeneous
            mixed = {rocm0, cpu0};
            EXPECT_FALSE(BackendSelector::isHomogeneous(mixed));

            // Three different types should not be homogeneous
            mixed = {cuda0, rocm0, cpu0};
            EXPECT_FALSE(BackendSelector::isHomogeneous(mixed));
        }

        TEST_F(Test__BackendSelector, IsHomogeneousWithSingleOrEmpty)
        {
            // Single device should be homogeneous (trivially)
            std::vector<DeviceId> single = {cuda0};
            EXPECT_TRUE(BackendSelector::isHomogeneous(single));

            single = {rocm0};
            EXPECT_TRUE(BackendSelector::isHomogeneous(single));

            single = {cpu0};
            EXPECT_TRUE(BackendSelector::isHomogeneous(single));

            // Empty list should be homogeneous (vacuously true)
            std::vector<DeviceId> empty;
            EXPECT_TRUE(BackendSelector::isHomogeneous(empty));
        }

        // ============================================================================
        // Utility Method Tests: hasGPU
        // ============================================================================

        TEST_F(Test__BackendSelector, HasGPUWithGpuDevices)
        {
            // CUDA should be detected
            std::vector<DeviceId> with_cuda = {cuda0};
            EXPECT_TRUE(BackendSelector::hasGPU(with_cuda));

            // ROCm should be detected
            std::vector<DeviceId> with_rocm = {rocm0};
            EXPECT_TRUE(BackendSelector::hasGPU(with_rocm));

            // GPU + CPU should detect GPU
            std::vector<DeviceId> mixed = {cuda0, cpu0};
            EXPECT_TRUE(BackendSelector::hasGPU(mixed));

            // Multiple GPUs should be detected
            std::vector<DeviceId> multi_gpu = {cuda0, cuda1, rocm0};
            EXPECT_TRUE(BackendSelector::hasGPU(multi_gpu));
        }

        TEST_F(Test__BackendSelector, HasGPUWithCpuOnly)
        {
            // CPU only should not have GPU
            std::vector<DeviceId> cpu_only = {cpu0};
            EXPECT_FALSE(BackendSelector::hasGPU(cpu_only));

            // Multiple CPUs should not have GPU
            std::vector<DeviceId> multi_cpu = {cpu0, cpu0};
            EXPECT_FALSE(BackendSelector::hasGPU(multi_cpu));

            // Empty should not have GPU
            std::vector<DeviceId> empty;
            EXPECT_FALSE(BackendSelector::hasGPU(empty));
        }

        // ============================================================================
        // Utility Method Tests: isCrossVendor
        // ============================================================================

        TEST_F(Test__BackendSelector, IsCrossVendorWithBothVendors)
        {
            // CUDA + ROCm is cross-vendor
            std::vector<DeviceId> cross = {cuda0, rocm0};
            EXPECT_TRUE(BackendSelector::isCrossVendor(cross));

            // Multiple of each is still cross-vendor
            cross = {cuda0, cuda1, rocm0, rocm1};
            EXPECT_TRUE(BackendSelector::isCrossVendor(cross));

            // With CPU mixed in, still cross-vendor if both GPU types present
            cross = {cuda0, rocm0, cpu0};
            EXPECT_TRUE(BackendSelector::isCrossVendor(cross));
        }

        TEST_F(Test__BackendSelector, IsCrossVendorWithSingleVendor)
        {
            // CUDA only is not cross-vendor
            std::vector<DeviceId> cuda_only = {cuda0, cuda1};
            EXPECT_FALSE(BackendSelector::isCrossVendor(cuda_only));

            // ROCm only is not cross-vendor
            std::vector<DeviceId> rocm_only = {rocm0, rocm1};
            EXPECT_FALSE(BackendSelector::isCrossVendor(rocm_only));

            // CPU only is not cross-vendor
            std::vector<DeviceId> cpu_only = {cpu0};
            EXPECT_FALSE(BackendSelector::isCrossVendor(cpu_only));

            // CUDA + CPU (no ROCm) is not cross-vendor
            std::vector<DeviceId> cuda_cpu = {cuda0, cpu0};
            EXPECT_FALSE(BackendSelector::isCrossVendor(cuda_cpu));

            // ROCm + CPU (no CUDA) is not cross-vendor
            std::vector<DeviceId> rocm_cpu = {rocm0, cpu0};
            EXPECT_FALSE(BackendSelector::isCrossVendor(rocm_cpu));
        }

        TEST_F(Test__BackendSelector, IsCrossVendorWithEmptyOrSingle)
        {
            // Empty should not be cross-vendor
            std::vector<DeviceId> empty;
            EXPECT_FALSE(BackendSelector::isCrossVendor(empty));

            // Single GPU should not be cross-vendor
            std::vector<DeviceId> single_cuda = {cuda0};
            EXPECT_FALSE(BackendSelector::isCrossVendor(single_cuda));

            std::vector<DeviceId> single_rocm = {rocm0};
            EXPECT_FALSE(BackendSelector::isCrossVendor(single_rocm));
        }

    } // namespace
} // namespace llaminar2
