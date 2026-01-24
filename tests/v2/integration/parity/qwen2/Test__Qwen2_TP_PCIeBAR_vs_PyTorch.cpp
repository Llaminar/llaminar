/**
 * @file Test__Qwen2_TP_PCIeBAR_vs_PyTorch.cpp
 * @brief Integration: Qwen2 LOCAL Tensor-Parallel with PCIeBAR Backend vs PyTorch
 *
 * Validates LOCAL-scope tensor parallelism using PCIeBAR backend for intra-process
 * collectives between a CUDA device and a ROCm device. This test runs as a SINGLE
 * MPI rank with heterogeneous GPUs performing TP inference via direct PCIe BAR P2P.
 *
 * Architecture:
 *   - Scope: LOCAL (single process, CollectiveScope::LOCAL)
 *   - Backend: PCIeBAR (direct CUDA↔ROCm P2P via mapped BAR, ~2.65 GB/s)
 *   - Devices: 1 CUDA GPU + 1 ROCm GPU (preferably same NUMA for best bandwidth)
 *   - MPI: Single rank (mpirun -np 1)
 *
 * PCIe BAR P2P mechanism:
 *   1. AMD GPU's BAR (Base Address Register) is memory-mapped via mmap()
 *   2. CUDA registers this mapping as IOMEMORY via cuMemHostRegister()
 *   3. CUDA kernels can directly write to AMD GPU memory (no host staging)
 *   4. Achieves ~2.65 GB/s on PCIe 3.0 x16 (vs ~12 GB/s with CUDA P2P)
 *
 * This differs from Test__Qwen2_TP_MPI_vs_PyTorch which uses GLOBAL scope
 * across multiple MPI ranks with Host+MPI staging.
 *
 * Test requirements:
 *   - At least 1 CUDA device and 1 ROCm device available
 *   - AMD GPU with Large BAR support (32GB BAR for MI50/MI60)
 *   - CAP_SYS_ADMIN capability or appropriate udev rule for BAR access
 *   - Requires models/qwen2.5-0.5b-instruct-q4_0.gguf
 *
 * @author David Sanftenberg
 * @date 2026-01-24
 */

#include "Qwen2ParityTestBase.h"
#include "models/qwen/Qwen2Schema.h"
#include "loaders/WeightManager.h"
#include "utils/NUMATopology.h"
#include "backends/p2p/DirectP2P.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "collective/backends/PCIeBARBackend.h"
#include "collective/DeviceGroup.h"

// NOTE: Cannot include both cuda_runtime.h and hip/hip_runtime.h in same TU
// due to type redefinitions (dim3, vector types, etc.). Use DeviceManager for counts.

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

namespace
{

    /**
     * @brief Find CUDA and ROCm device IDs, preferring same NUMA node
     *
     * Uses DeviceManager for device enumeration (avoids CUDA/HIP header conflicts)
     *
     * @param out_cuda_id Output: CUDA device ID
     * @param out_rocm_id Output: ROCm device ID
     * @return true if a pair was found (same NUMA preferred but not required)
     */
    bool findCUDAROCmPair(int &out_cuda_id, int &out_rocm_id)
    {
        out_cuda_id = -1;
        out_rocm_id = -1;

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        auto &dm = DeviceManager::instance();

        // Ensure DeviceManager is initialized (enumerate all devices, no NUMA filter)
        dm.initialize(-1);

        int cuda_count = dm.cuda_device_count();
        int rocm_count = dm.rocm_device_count();

        LOG_DEBUG("[PCIeBAR] DeviceManager: " << cuda_count << " CUDA, " << rocm_count << " ROCm devices");

        if (cuda_count < 1 || rocm_count < 1)
        {
            LOG_WARN("[PCIeBAR] Need at least 1 CUDA and 1 ROCm device");
            return false;
        }

        // Try to find same-NUMA pair first
        for (int cuda_idx = 0; cuda_idx < cuda_count; ++cuda_idx)
        {
            auto cuda_numa = NUMATopology::getCUDAGPUNUMANode(cuda_idx);

            for (int rocm_idx = 0; rocm_idx < rocm_count; ++rocm_idx)
            {
                auto rocm_numa = NUMATopology::getROCmGPUNUMANode(rocm_idx);

                if (cuda_numa.numa_node == rocm_numa.numa_node && cuda_numa.numa_node >= 0)
                {
                    out_cuda_id = cuda_idx;
                    out_rocm_id = rocm_idx;
                    LOG_INFO("[PCIeBAR] Found same-NUMA pair: CUDA " << cuda_idx
                                                                     << " + ROCm " << rocm_idx << " (NUMA " << cuda_numa.numa_node << ")");
                    return true;
                }
            }
        }

        // Fallback: any CUDA + any ROCm
        out_cuda_id = 0;
        out_rocm_id = 0;
        LOG_INFO("[PCIeBAR] Using cross-NUMA pair: CUDA " << out_cuda_id
                                                          << " + ROCm " << out_rocm_id << " (may have reduced bandwidth)");
        return true;
#else
        return false;
#endif
    }

} // namespace

/**
 * @brief Test fixture for LOCAL TP with PCIeBAR Backend
 *
 * Runs as single MPI rank with 1 CUDA + 1 ROCm device using PCIeBAR for collectives.
 */
class Test__Qwen2_TP_PCIeBAR_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    bool pciebar_available_ = false;
    bool hetero_gpus_available_ = false;
    int cuda_device_id_ = -1;
    int rocm_device_id_ = -1;

    // ==========================================================================
    // Qwen2ParityTestBase overrides
    // ==========================================================================

    BackendThresholds getBackendThresholds() override
    {
        // PCIeBAR LOCAL TP has cross-vendor numerical differences:
        //   - Different quantization rounding (CUDA INT8 vs ROCm INT8)
        //   - Different FP32 accumulation order in kernels
        //   - No host staging overhead (direct P2P), but same vendor drift
        return BackendThresholds{
            .cosine_threshold = 0.95f, // Cross-vendor drift
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 6,
            .min_early_layers_passed = 6,
            .kl_threshold = 0.15f};
    }

    std::string getBackendName() override
    {
        return "PCIeBAR_LOCAL_TP(CUDA+ROCm)";
    }

    DeviceId getDeviceForRank() override
    {
        // For LOCAL TP, return CUDA device as "primary"
        // The actual TP happens via CollectiveContext with both devices
        return DeviceId::cuda(cuda_device_id_);
    }

    // ==========================================================================
    // SetUp / TearDown
    // ==========================================================================

    void SetUp() override
    {
        // Check for CUDA + ROCm
        hetero_gpus_available_ = findCUDAROCmPair(cuda_device_id_, rocm_device_id_);

        if (!hetero_gpus_available_)
        {
            GTEST_SKIP() << "Test requires 1 CUDA + 1 ROCm device";
        }

        // Check PCIe BAR P2P availability
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        auto caps = DirectP2PEngine::probeCapabilities();
        pciebar_available_ = caps.canDoPCIeBarP2P();

        LOG_INFO("[PCIeBAR TP Parity] PCIe BAR P2P: " << (pciebar_available_ ? "available" : "NOT available"));
        LOG_INFO("[PCIeBAR TP Parity]   BAR accessible: " << (caps.pcie_bar_accessible ? "YES" : "NO"));
        LOG_INFO("[PCIeBAR TP Parity]   IOMEMORY support: " << (caps.pcie_bar_iomemory_supported ? "YES" : "NO"));
        LOG_INFO("[PCIeBAR TP Parity]   Discovered BARs: " << caps.discovered_bars.size());
#endif

        if (!pciebar_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P not available (need root or udev rule)";
        }

        // For LOCAL scope, we run single-rank but with 2 devices
        int rank = 0, world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_size != 1)
        {
            GTEST_SKIP() << "LOCAL TP test must run with -np 1 (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        LOG_INFO("[PCIeBAR TP Parity] Using CUDA " << cuda_device_id_
                                                   << " + ROCm " << rocm_device_id_ << " with PCIeBAR backend");

        Qwen2ParityTestBase::SetUp();
    }

    void TearDown() override
    {
        Qwen2ParityTestBase::TearDown();
    }
};

/**
 * @brief Test: Verify PCIeBAR backend initializes and measures bandwidth
 */
TEST_F(Test__Qwen2_TP_PCIeBAR_vs_PyTorch, BackendSelection_IsPCIeBAR)
{
    if (!hetero_gpus_available_ || !pciebar_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    // Build a LOCAL device group with CUDA + ROCm
    DeviceGroupBuilder builder;
    auto group = builder
                     .setName("pciebar_local_test_group")
                     .setScope(CollectiveScope::LOCAL)
                     .addDevice(DeviceId::cuda(cuda_device_id_))
                     .addDevice(DeviceId::rocm(rocm_device_id_))
                     .setLocalRank(0)
                     .build();

    // Verify group properties
    EXPECT_TRUE(group.isHeterogeneous()) << "Group should be heterogeneous";
    EXPECT_TRUE(group.isLocal()) << "Group should be LOCAL scope";
    EXPECT_EQ(group.cuda_count, 1) << "Should have 1 CUDA device";
    EXPECT_EQ(group.rocm_count, 1) << "Should have 1 ROCm device";

    // Initialize PCIeBAR backend
    PCIeBARBackend backend;
    bool init_ok = backend.initialize(group);

    LOG_INFO("[PCIeBAR TP Test] Backend initialization: " << (init_ok ? "SUCCESS" : "FAILED"));

    ASSERT_TRUE(init_ok) << "PCIeBARBackend initialization failed";
    EXPECT_TRUE(backend.isPCIeBarActive()) << "PCIe BAR should be active";
    EXPECT_EQ(backend.type(), CollectiveBackendType::PCIE_BAR);

    // Verify reasonable bandwidth
    double bandwidth = backend.getMeasuredBandwidthGBps();
    LOG_INFO("[PCIeBAR TP Test] Measured bandwidth: " << bandwidth << " GB/s");
    EXPECT_GT(bandwidth, 1.0) << "PCIe BAR bandwidth should be > 1 GB/s";

    backend.shutdown();
#endif
}

/**
 * @brief Test: Verify NUMA topology detection
 */
TEST_F(Test__Qwen2_TP_PCIeBAR_vs_PyTorch, DeviceTopology_NUMAInfo)
{
    if (!hetero_gpus_available_)
    {
        GTEST_SKIP() << "Heterogeneous GPUs not available";
    }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    auto cuda_numa = NUMATopology::getCUDAGPUNUMANode(cuda_device_id_);
    auto rocm_numa = NUMATopology::getROCmGPUNUMANode(rocm_device_id_);

    LOG_INFO("[PCIeBAR Test] CUDA " << cuda_device_id_ << " NUMA: " << cuda_numa.numa_node);
    LOG_INFO("[PCIeBAR Test] ROCm " << rocm_device_id_ << " NUMA: " << rocm_numa.numa_node);

    // Both should have valid NUMA info
    EXPECT_GE(cuda_numa.numa_node, 0) << "CUDA device should have valid NUMA node";
    EXPECT_GE(rocm_numa.numa_node, 0) << "ROCm device should have valid NUMA node";

    if (cuda_numa.numa_node == rocm_numa.numa_node)
    {
        LOG_INFO("[PCIeBAR Test] ✓ Same NUMA node - optimal for PCIe BAR P2P");
    }
    else
    {
        LOG_WARN("[PCIeBAR Test] ✗ Different NUMA nodes - cross-socket P2P may have higher latency");
    }
#endif
}

// Instantiate standard parity tests
INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_TP_PCIeBAR_vs_PyTorch);

// =============================================================================
// Main (MPI wrapper)
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "WARNING: MPI does not provide MPI_THREAD_MULTIPLE support" << std::endl;
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
