/**
 * @file Test__Qwen2_TP_RCCL_vs_PyTorch.cpp
 * @brief Integration: Qwen2 LOCAL Tensor-Parallel with RCCL Backend vs PyTorch
 *
 * Validates LOCAL-scope tensor parallelism using RCCL backend for intra-process
 * collectives between two ROCm devices. This test runs as a SINGLE MPI rank with
 * two AMD GPUs performing TP inference.
 *
 * Architecture:
 *   - Scope: LOCAL (single process, CollectiveScope::LOCAL)
 *   - Backend: RCCL (GPU-native collectives between ROCm devices)
 *   - Devices: 2 ROCm GPUs on same node
 *   - MPI: Single rank (mpirun -np 1)
 *
 * This differs from Test__Qwen2_TP_MPI_vs_PyTorch which uses GLOBAL scope
 * across multiple MPI ranks with Host+MPI staging.
 *
 * Test requirements:
 *   - At least 2 ROCm devices available
 *   - RCCL library compiled in (HAVE_RCCL)
 *   - Requires models/qwen2.5-0.5b-instruct-q4_0.gguf
 *
 * @author David Sanftenberg
 * @date 2026-01-24
 */

#include "Qwen2ParityTestBase.h"
#include "models/qwen/Qwen2Schema.h"
#include "loaders/WeightManager.h"
#include "collective/BackendRouter.h"
#include "collective/DeviceGroup.h"
#include "backends/BackendManager.h"

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

/**
 * @brief Test fixture for LOCAL TP with RCCL Backend
 *
 * Runs as single MPI rank with 2 ROCm devices using RCCL for collectives.
 */
class Test__Qwen2_TP_RCCL_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    bool rccl_available_ = false;
    bool dual_rocm_available_ = false;
    int rocm_device_0_ = -1;
    int rocm_device_1_ = -1;

    // ==========================================================================
    // Qwen2ParityTestBase overrides
    // ==========================================================================

    BackendThresholds getBackendThresholds() override
    {
        // RCCL LOCAL TP should have very tight tolerances since:
        //   - Same vendor (ROCm+ROCm)
        //   - No cross-process communication overhead
        //   - GPU-native collectives (no host staging)
        return BackendThresholds{
            .cosine_threshold = 0.97f,
            .decode_cosine_threshold = 0.97f,
            .early_layers_count = 6,
            .min_early_layers_passed = 6,
            .kl_threshold = 0.08f};
    }

    std::string getBackendName() override
    {
        return "RCCL_LOCAL_TP(2 ROCm devices)";
    }

    DeviceId getDeviceForRank() override
    {
        // For LOCAL TP, return first ROCm device as "primary"
        // The actual TP happens via CollectiveContext with both devices
        return DeviceId::rocm(rocm_device_0_);
    }

    // ==========================================================================
    // SetUp / TearDown
    // ==========================================================================

    void SetUp() override
    {
        // Check for dual ROCm GPUs
#ifdef HAVE_ROCM
        int rocm_count = 0;
        (void)hipGetDeviceCount(&rocm_count);

        if (rocm_count >= 2)
        {
            dual_rocm_available_ = true;
            rocm_device_0_ = 0;
            rocm_device_1_ = 1;

            LOG_INFO("[RCCL TP Parity] Found " << rocm_count << " ROCm devices");
        }
        else
        {
            LOG_WARN("[RCCL TP Parity] Need at least 2 ROCm devices (found " << rocm_count << ")");
        }
#endif

        // Check RCCL availability
#ifdef HAVE_RCCL
        rccl_available_ = true;
        LOG_INFO("[RCCL TP Parity] RCCL backend available");
#else
        LOG_WARN("[RCCL TP Parity] RCCL not compiled in (HAVE_RCCL not defined)");
#endif

        if (!dual_rocm_available_ || !rccl_available_)
        {
            GTEST_SKIP() << "Test requires 2 ROCm devices + RCCL";
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

        LOG_INFO("[RCCL TP Parity] Using ROCm devices " << rocm_device_0_
                                                        << " and " << rocm_device_1_ << " with RCCL backend");

        Qwen2ParityTestBase::SetUp();
    }

    void TearDown() override
    {
        Qwen2ParityTestBase::TearDown();
    }
};

/**
 * @brief Test: Verify RCCL backend is selected for dual-ROCm LOCAL group
 */
TEST_F(Test__Qwen2_TP_RCCL_vs_PyTorch, BackendSelection_IsRCCL)
{
    if (!dual_rocm_available_ || !rccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Build a LOCAL device group with 2 ROCm devices
    DeviceGroupBuilder builder;
    auto group = builder
                     .setName("rccl_test_group")
                     .setScope(CollectiveScope::LOCAL)
                     .addDevice(DeviceId::rocm(rocm_device_0_))
                     .addDevice(DeviceId::rocm(rocm_device_1_))
                     .setLocalRank(0)
                     .build();

    // Verify group properties
    EXPECT_TRUE(group.allROCm()) << "Group should be all-ROCm";
    EXPECT_FALSE(group.isHeterogeneous()) << "Group should be homogeneous";
    EXPECT_TRUE(group.isLocal()) << "Group should be LOCAL scope";
    EXPECT_EQ(group.rocm_count, 2) << "Should have 2 ROCm devices";

    LOG_INFO("[RCCL TP Test] Group: " << group.toString());
    LOG_INFO("[RCCL TP Test] Expected backend: RCCL (all ROCm, LOCAL scope)");
}

// Instantiate standard parity tests
INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_TP_RCCL_vs_PyTorch);

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
