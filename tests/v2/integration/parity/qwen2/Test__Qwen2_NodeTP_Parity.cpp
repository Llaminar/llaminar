/**
 * @file Test__Qwen2_NodeTP_Parity.cpp
 * @brief NodeTP parity tests using MPI across ranks on one node.
 *
 * These tests validate Node-Local Tensor Parallelism (NodeTP) infrastructure
 * where tensor parallelism spans multiple MPI ranks on the same physical node.
 * Unlike LocalTP which uses NCCL/RCCL/HOST for intra-process multi-device
 * communication, NodeTP uses MPI collectives for cross-rank communication
 * within the same machine (e.g., CPU sockets connected via UPI).
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - Each rank participates in NodeTP collective operations (via GlobalTPContext)
 *
 * Test configurations:
 *   - NodeTP_2xMPI_CPU: 2 MPI ranks, each using CPU (UPI backend)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen2ModelParityDefinitions.h"
#include "Qwen2ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

// Common excluded stages for NodeTP (sharded outputs can't be compared directly)
// For NodeTP, intermediate activations are SHARDED across ranks. Only the final
// LM_HEAD output (after allgather) can be compared against PyTorch reference.
// ROW_PARALLEL outputs (ATTENTION_OUTPUT, FFN_DOWN) contain partial sums before allreduce.
// COLUMN_PARALLEL outputs are local slices that need to be gathered.
static const std::vector<std::string> kNodeTPExcludedStages = {
    // Column-parallel: outputs are sharded slices
    "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "Q_ROPE", "K_ROPE",
    "ATTENTION_CONTEXT",
    "FFN_GATE", "FFN_UP", "FFN_SWIGLU",
    // Row-parallel: outputs are partial sums before allreduce
    "ATTENTION_OUTPUT", "FFN_DOWN",
    // These also have sharded intermediate states
    "ATTN_RESIDUAL", "FFN_RESIDUAL"};

/** @return Canonically expanded two-rank CPU NodeTP case. */
static const std::vector<ModelParityCase> &qwen2NodeTPCases()
{
    static const auto cases = expandModelParityDefinition(
        qwen2Q40ParityDefinition(
            ModelParityTopologyDefinition{
                .test_id = "NodeTP_2xMPI_CPU",
                .kind = ModelParityTopologyKind::NodeTensorParallel,
                .participants = {
                    {GlobalDeviceAddress::cpu(0), 0},
                    {GlobalDeviceAddress::cpu(1), 1},
                },
                .collective = Collective::MPI,
                .mpi_ranks = 2,
            },
            BackendThresholds{
                .cosine_threshold = 0.99f,
                .decode_cosine_threshold = 0.98f,
                .early_layers_count = 4,
                .min_early_layers_passed = 3,
                .kl_threshold = 0.008f,
                .excluded_stages = kNodeTPExcludedStages,
            }));
    return cases;
}

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

/**
 * @brief Parameterized test fixture for NodeTP parity tests
 *
 * Inherits from ConfigDrivenParityTest which handles all setup/teardown
 * including MPI context creation and NodeTP context initialization.
 */
class Qwen2NodeTPParityTest : public ConfigDrivenParityTest<Qwen2NodeTPParityTest>,
                               public ModelParityCaseParameter
{};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Verify NodeTP infrastructure initialization
 *
 * Tests that:
 * - NodeTP context (GlobalTPContext) is created successfully on each rank
 * - MPI communicator is valid
 * - Rank indices are correct
 */
TEST_P(Qwen2NodeTPParityTest, NodeTPContextInitialization)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    // Verify MPI context
    ASSERT_NE(mpi_ctx_, nullptr) << "MPI context should be initialized";
    EXPECT_GE(mpi_ctx_->world_size(), cfg().mpi_ranks)
        << "World size should be at least " << cfg().mpi_ranks;

    // Verify NodeTP context (GlobalTPContext implements cross-rank TP communication)
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeTP context should be created";
    EXPECT_EQ(global_tp_ctx_->degree(), mpi_ctx_->world_size())
        << "TP degree should match world size";
    EXPECT_EQ(global_tp_ctx_->myIndex(), mpi_ctx_->rank())
        << "TP index should match MPI rank";
    EXPECT_FALSE(global_tp_ctx_->isLocal())
        << "NodeTP context should not be local (uses MPI, not intra-process)";
    EXPECT_EQ(global_tp_ctx_->backend(), CollectiveBackendType::UPI)
        << "NodeTP should use UPI backend";

    LOG_INFO("[NodeTP] Rank " << mpi_ctx_->rank() << " verified TPContext");
}

/**
 * @brief Test NodeTP allreduce operation
 *
 * Each rank creates a tensor with its rank value, performs allreduce,
 * and verifies the result is the sum of all ranks.
 */
TEST_P(Qwen2NodeTPParityTest, NodeTPAllreduce)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeTP context required";

    // Create a simple tensor with rank value
    auto tensor_factory = std::make_unique<TensorFactory>(*mpi_ctx_);
    auto tensor = tensor_factory->createFP32({1, 128});

    // Initialize with rank value
    float rank_value = static_cast<float>(mpi_ctx_->rank() + 1);
    std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), rank_value);

    LOG_INFO("[NodeTP] Rank " << mpi_ctx_->rank() << " input value: " << rank_value);

    // Perform allreduce
    ASSERT_TRUE(global_tp_ctx_->allreduce(tensor.get())) << "Allreduce failed";

    // Verify: sum of 1 + 2 + ... + world_size
    int world_size = mpi_ctx_->world_size();
    float expected_sum = static_cast<float>(world_size * (world_size + 1)) / 2.0f;

    float actual_value = tensor->data()[0];
    EXPECT_NEAR(actual_value, expected_sum, 1e-5f)
        << "Allreduce result should be sum of ranks";

    LOG_INFO("[NodeTP] Rank " << mpi_ctx_->rank() << " allreduce result: "
                                   << actual_value << " (expected: " << expected_sum << ")");
}

/**
 * @brief Test NodeTP broadcast operation
 *
 * Rank 0 broadcasts a tensor with specific values, all ranks verify receipt.
 */
TEST_P(Qwen2NodeTPParityTest, NodeTPBroadcast)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeTP context required";

    auto tensor_factory = std::make_unique<TensorFactory>(*mpi_ctx_);
    auto tensor = tensor_factory->createFP32({1, 64});

    const float broadcast_value = 42.0f;
    int source_rank = 0;

    if (mpi_ctx_->rank() == source_rank)
    {
        // Source rank initializes tensor
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), broadcast_value);
        LOG_INFO("[NodeTP] Rank " << mpi_ctx_->rank() << " broadcasting value: " << broadcast_value);
    }
    else
    {
        // Other ranks initialize with different value
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), 0.0f);
    }

    // Perform broadcast
    ASSERT_TRUE(global_tp_ctx_->broadcast(tensor.get(), source_rank)) << "Broadcast failed";

    // Verify all ranks have broadcast value
    float received_value = tensor->data()[0];
    EXPECT_NEAR(received_value, broadcast_value, 1e-5f)
        << "Broadcast should deliver source rank's value";

    LOG_INFO("[NodeTP] Rank " << mpi_ctx_->rank() << " received: " << received_value);
}

/**
 * @brief Test NodeTP barrier synchronization
 */
TEST_P(Qwen2NodeTPParityTest, NodeTPBarrier)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeTP context required";

    LOG_INFO("[NodeTP] Rank " << mpi_ctx_->rank() << " entering barrier");

    // This should not deadlock or hang
    global_tp_ctx_->barrier();

    LOG_INFO("[NodeTP] Rank " << mpi_ctx_->rank() << " exited barrier");
}

/**
 * @brief Full prefill/decode production parity with cross-rank TP sharding.
 */
TEST_P(Qwen2NodeTPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen2NodeTP,
    Qwen2NodeTPParityTest,
    ::testing::ValuesIn(qwen2NodeTPCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    {
        return info.param.testName();
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

/**
 * @brief Custom main() with MPI initialization for NodeTP tests
 *
 * NodeTP tests REQUIRE MPI to be initialized with multiple ranks.
 * Run with: mpirun -np 2 ./v2_integration_node_tp_parity
 */
int main(int argc, char **argv)
{
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0)
    {
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║         NODE-LOCAL TP PARITY TEST SUITE                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  MPI world size: " << world_size << " ranks" << std::string(42 - std::to_string(world_size).length(), ' ') << "║\n";
        std::cout << "║  Thread support: " << (provided >= MPI_THREAD_MULTIPLE ? "MPI_THREAD_MULTIPLE" : "limited") << std::string(26, ' ') << "║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    // Barrier to ensure clean output
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Reduce result across all ranks (any failure = overall failure)
    int global_result;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // CRITICAL: Shutdown GlobalBackendRouter before MPI_Finalize to ensure
    // NCCLCoordinator cleanup happens while CUDA runtime is still active.
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    // Finalize MPI
    MPI_Finalize();

    // Skip static destructors — see Test__Qwen2_SingleDevice_Parity.cpp for rationale.
    std::cout.flush();
    std::cerr.flush();
    _exit(global_result);
}
