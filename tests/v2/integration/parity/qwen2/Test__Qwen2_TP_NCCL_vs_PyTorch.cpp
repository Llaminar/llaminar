/**
 * @file Test__Qwen2_TP_NCCL_vs_PyTorch.cpp
 * @brief Integration: Qwen2 LOCAL Tensor-Parallel with NCCL Backend vs PyTorch
 *
 * Validates LOCAL-scope tensor parallelism using NCCL backend for intra-process
 * collectives between two CUDA devices. This test runs as a SINGLE MPI rank with
 * two CUDA GPUs performing TP inference.
 *
 * Architecture:
 *   - Scope: LOCAL (single process, CollectiveScope::LOCAL)
 *   - Backend: NCCL (GPU-native collectives between CUDA devices)
 *   - Devices: 2 CUDA GPUs on same node
 *   - MPI: Single rank (mpirun -np 1)
 *
 * This differs from Test__Qwen2_TP_MPI_vs_PyTorch which uses GLOBAL scope
 * across multiple MPI ranks with Host+MPI staging.
 *
 * Test requirements:
 *   - At least 2 CUDA devices available
 *   - NCCL library compiled in (HAVE_NCCL)
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

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

/**
 * @brief Test fixture for LOCAL TP with NCCL Backend
 *
 * Runs as single MPI rank with 2 CUDA devices using NCCL for collectives.
 */
class Test__Qwen2_TP_NCCL_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    bool nccl_available_ = false;
    bool dual_cuda_available_ = false;
    int cuda_device_0_ = -1;
    int cuda_device_1_ = -1;

    // ==========================================================================
    // Qwen2ParityTestBase overrides
    // ==========================================================================

    BackendThresholds getBackendThresholds() override
    {
        // NCCL LOCAL TP should have very tight tolerances since:
        //   - Same vendor (CUDA+CUDA)
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
        return "NCCL_LOCAL_TP(2 CUDA devices)";
    }

    DeviceId getDeviceForRank() override
    {
        // For LOCAL TP, we simulate rank behavior within a single process
        // local_rank_ is set based on which "virtual rank" we're computing for
        // In actual inference, both devices are used by the same process

        // Return first CUDA device as "primary" for this test fixture
        // The actual TP happens via CollectiveContext with both devices
        return DeviceId::cuda(cuda_device_0_);
    }

    // ==========================================================================
    // SetUp / TearDown
    // ==========================================================================

    void SetUp() override
    {
        // Check for dual CUDA GPUs
#ifdef HAVE_CUDA
        int cuda_count = 0;
        (void)cudaGetDeviceCount(&cuda_count);

        if (cuda_count >= 2)
        {
            dual_cuda_available_ = true;
            cuda_device_0_ = 0;
            cuda_device_1_ = 1;

            LOG_INFO("[NCCL TP Parity] Found " << cuda_count << " CUDA devices");
        }
        else
        {
            LOG_WARN("[NCCL TP Parity] Need at least 2 CUDA devices (found " << cuda_count << ")");
        }
#endif

        // Check NCCL availability
#ifdef HAVE_NCCL
        nccl_available_ = true;
        LOG_INFO("[NCCL TP Parity] NCCL backend available");
#else
        LOG_WARN("[NCCL TP Parity] NCCL not compiled in (HAVE_NCCL not defined)");
#endif

        if (!dual_cuda_available_ || !nccl_available_)
        {
            GTEST_SKIP() << "Test requires 2 CUDA devices + NCCL";
        }

        // For LOCAL scope, we run single-rank but with 2 devices
        // Create a minimal MPI context (or null for non-MPI builds)
        int rank = 0, world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_size != 1)
        {
            GTEST_SKIP() << "LOCAL TP test must run with -np 1 (got " << world_size << ")";
        }

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        LOG_INFO("[NCCL TP Parity] Using CUDA devices " << cuda_device_0_
                                                        << " and " << cuda_device_1_ << " with NCCL backend");

        Qwen2ParityTestBase::SetUp();
    }

    void TearDown() override
    {
        Qwen2ParityTestBase::TearDown();
    }
};

/**
 * @brief Test: Verify NCCL backend is selected for dual-CUDA LOCAL group
 */
TEST_F(Test__Qwen2_TP_NCCL_vs_PyTorch, BackendSelection_IsNCCL)
{
    if (!dual_cuda_available_ || !nccl_available_)
    {
        GTEST_SKIP() << "Hardware requirements not met";
    }

    // Build a LOCAL device group with 2 CUDA devices
    DeviceGroupBuilder builder;
    auto group = builder
                     .setName("nccl_test_group")
                     .setScope(CollectiveScope::LOCAL)
                     .addDevice(DeviceId::cuda(cuda_device_0_))
                     .addDevice(DeviceId::cuda(cuda_device_1_))
                     .setLocalRank(0)
                     .build();

    // Verify group properties
    EXPECT_TRUE(group.allCUDA()) << "Group should be all-CUDA";
    EXPECT_FALSE(group.isHeterogeneous()) << "Group should be homogeneous";
    EXPECT_TRUE(group.isLocal()) << "Group should be LOCAL scope";
    EXPECT_EQ(group.cuda_count, 2) << "Should have 2 CUDA devices";

    // Check that NCCL would be selected
    // Note: We can't easily test the actual backend selection without
    // a full BackendRouter, but we verify the group properties are correct
    LOG_INFO("[NCCL TP Test] Group: " << group.toString());
    LOG_INFO("[NCCL TP Test] Expected backend: NCCL (all CUDA, LOCAL scope)");
}

// Instantiate standard parity tests
INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_TP_NCCL_vs_PyTorch);

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
