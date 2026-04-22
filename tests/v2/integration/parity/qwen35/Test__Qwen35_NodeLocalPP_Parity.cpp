/**
 * @file Test__Qwen35_NodeLocalPP_Parity.cpp
 * @brief Node-Local PP parity tests for Qwen3.5 using MPI (multi-rank, same node)
 *
 * These tests validate Node-Local Pipeline Parallelism (NodeLocalPP) via
 * GlobalOrchestrator for Qwen3.5 GDN+FA hybrid architecture. Each MPI rank
 * handles a disjoint subset of transformer layers, with GlobalOrchestrator
 * coordinating activation transfer between ranks via MPI send/recv.
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - 2 MPI ranks for 2-way PP
 *
 * Test configurations:
 *   - NodeLocalPP_2xMPI_CPU_08B: 2 MPI ranks, CPU, Qwen3.5-0.8B Q4_0
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "Qwen35ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

static const std::vector<TestConfig> kNodeLocalPPTestConfigs = {
    // =========================================================================
    // Qwen3.5-0.8B (Q4_0) — 2-way Node-Local PP with CPU
    // =========================================================================
    {
        .name = "NodeLocalPP_2xMPI_CPU_08B",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeLocalPP,
        .collective = Collective::None, // PP uses MPI send/recv, not collectives
        .thresholds = {
            .cosine_threshold = 0.94f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.012f,
        },
        .mpi_ranks = 2,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35NodeLocalPPParityTest : public Qwen35ConfigDrivenParityTest<Qwen35NodeLocalPPParityTest>,
                                    public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Verify GlobalOrchestrator pipeline setup for PP
 */
TEST_P(Qwen35NodeLocalPPParityTest, GlobalOrchestratorSetup)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    ASSERT_NE(mpi_ctx_, nullptr) << "MPI context should be initialized";
    EXPECT_GE(mpi_ctx_->world_size(), cfg().mpi_ranks)
        << "World size should be at least " << cfg().mpi_ranks;

    // Verify GlobalOrchestrator was created
    ASSERT_NE(global_orchestrator_ptr_, nullptr)
        << "GlobalOrchestrator should be created for NodeLocalPP";

    // Verify pipeline topology
    bool is_head = global_orchestrator_ptr_->isPipelineHead();
    bool is_tail = global_orchestrator_ptr_->isPipelineTail();

    if (mpi_ctx_->rank() == 0)
    {
        EXPECT_TRUE(is_head) << "Rank 0 should be pipeline head";
        EXPECT_FALSE(is_tail) << "Rank 0 should NOT be pipeline tail (2-way PP)";
    }
    else if (mpi_ctx_->rank() == mpi_ctx_->world_size() - 1)
    {
        EXPECT_FALSE(is_head) << "Last rank should NOT be pipeline head";
        EXPECT_TRUE(is_tail) << "Last rank should be pipeline tail";
    }

    LOG_INFO("[NodeLocalPP Qwen3.5] Rank " << mpi_ctx_->rank()
             << " verified GlobalOrchestrator (head=" << is_head
             << ", tail=" << is_tail << ")");
}

/**
 * @brief Prefill parity test with NodeLocalPP via GlobalOrchestrator
 */
TEST_P(Qwen35NodeLocalPPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

/**
 * @brief Decode parity test with NodeLocalPP via GlobalOrchestrator
 */
TEST_P(Qwen35NodeLocalPPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen35NodeLocalPP,
    Qwen35NodeLocalPPParityTest,
    ::testing::ValuesIn(kNodeLocalPPTestConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0)
    {
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║         QWEN3.5 NODE-LOCAL PP PARITY TEST SUITE                  ║\n";
        std::cout << "║                                                                  ║\n";
        std::cout << "║  MPI ranks: " << world_size << "                                                   ║\n";
        std::cout << "║  Testing: Cross-rank pipeline parallelism via GlobalOrchestrator ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // Shutdown GPU context pools gracefully before MPI_Finalize
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    return result;
}
