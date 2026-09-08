/**
 * @file Test__Qwen3_NodePP_Parity.cpp
 * @brief Node PP parity tests for Qwen3 using MPI (multi-rank, same node)
 *
 * These tests validate Node-scoped Pipeline Parallelism (NodePP) via
 * GlobalOrchestrator for Qwen3 architecture. Unlike LocalPP which uses
 * intra-process device-to-device transfers, NodePP uses MPI send/recv
 * for cross-rank activation transfer — each MPI rank handles a disjoint
 * subset of transformer layers.
 *
 * The GlobalOrchestrator coordinates:
 *   - Head rank: embedding + first half of layers → MPI_Send hidden state
 *   - Tail rank: MPI_Recv hidden state → second half of layers + LM head
 *   - Token broadcast: tail rank samples, broadcasts token to all ranks
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - 2 MPI ranks for 2-way PP
 *
 * Test configurations:
 *   - NodePP_2xMPI_CPU: 2 MPI ranks, CPU, Qwen3-0.6B Q8_0
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen3ModelParityDefinitions.h"
#include "Qwen3ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen3;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

/** @return Canonically generated two-rank CPU pipeline case. */
static const std::vector<ModelParityCase> &qwen3NodePipelineCases()
{
    static const auto cases = expandModelParityDefinition(
        qwen3Q80ParityDefinition(
            ModelParityTopologyDefinition{
                .test_id = "NodePP_2xMPI_CPU",
                .kind = ModelParityTopologyKind::NodePipelineParallel,
                .participants = {
                    {GlobalDeviceAddress::cpu(0), 0},
                    {GlobalDeviceAddress::cpu(1), 1},
                },
                .collective = Collective::None,
                .mpi_ranks = 2,
                .pipeline_stage_sizes = {1, 1},
            },
            qwen3ParityThresholds(0.012f)));
    return cases;
}

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

/**
 * @brief Generated node-scoped pipeline parity fixture.
 *
 * The typed case owns rank placement and precision; this suite names only the
 * execution family shared by its production and focused setup checks.
 */
class Qwen3NodePPParityTest : public Qwen3ConfigDrivenParityTest<Qwen3NodePPParityTest>,
                                   public ModelParityCaseParameter
{
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Verify GlobalOrchestrator pipeline setup for PP
 */
TEST_P(Qwen3NodePPParityTest, GlobalOrchestratorSetup)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    ASSERT_NE(mpi_ctx_, nullptr) << "MPI context should be initialized";
    EXPECT_GE(mpi_ctx_->world_size(), cfg().mpi_ranks)
        << "World size should be at least " << cfg().mpi_ranks;

    // Verify GlobalOrchestrator was created
    ASSERT_NE(global_orchestrator_ptr_, nullptr)
        << "GlobalOrchestrator should be created for NodePP";

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

    LOG_INFO("[NodePP Qwen3] Rank " << mpi_ctx_->rank()
                                         << " verified GlobalOrchestrator (head=" << is_head
                                         << ", tail=" << is_tail << ")");
}

/**
 * @brief Full prefill/decode production parity via GlobalOrchestrator.
 */
TEST_P(Qwen3NodePPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen3NodePP,
    Qwen3NodePPParityTest,
    ::testing::ValuesIn(qwen3NodePipelineCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    {
        return info.param.testName();
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
        std::cout << "║         QWEN3 NODE-LOCAL PP PARITY TEST SUITE                    ║\n";
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

    // Skip static destructors — see Test__Qwen2_SingleDevice_Parity.cpp for rationale.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
