/**
 * @file Test__Qwen2_NodeLocalTP_Parity.cpp
 * @brief Node-Local TP parity tests using MPI (multi-rank, same node)
 *
 * These tests validate Node-Local Tensor Parallelism (NodeLocalTP) infrastructure
 * where tensor parallelism spans multiple MPI ranks on the same physical node.
 * Unlike LocalTP which uses NCCL/RCCL/PCIeBAR for intra-process multi-device
 * communication, NodeLocalTP uses MPI collectives for cross-rank communication
 * within the same machine (e.g., CPU sockets connected via UPI).
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - Each rank participates in GlobalTPContext collective operations
 *
 * Test configurations:
 *   - NodeLocalTP_2xMPI_CPU: 2 MPI ranks, each using CPU (UPI backend)
 *   - NodeLocalTP_4xMPI_CPU: 4 MPI ranks (requires 4 NUMA nodes)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "Qwen2ParityTestBase.h"
#include "collective/BackendRouter.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================
// Test Configuration Definitions
// =============================================================================

// Common excluded stages for NodeLocalTP (sharded outputs can't be compared directly)
// For NodeLocalTP, intermediate activations are SHARDED across ranks. Only the final
// LM_HEAD output (after allgather) can be compared against PyTorch reference.
// ROW_PARALLEL outputs (ATTENTION_OUTPUT, FFN_DOWN) contain partial sums before allreduce.
// COLUMN_PARALLEL outputs are local slices that need to be gathered.
static const std::vector<std::string> kNodeLocalTPExcludedStages = {
    // Column-parallel: outputs are sharded slices
    "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "Q_ROPE", "K_ROPE",
    "ATTENTION_CONTEXT",
    "FFN_GATE", "FFN_UP", "FFN_SWIGLU",
    // Row-parallel: outputs are partial sums before allreduce
    "ATTENTION_OUTPUT", "FFN_DOWN",
    // These also have sharded intermediate states
    "ATTN_RESIDUAL", "FFN_RESIDUAL"};

/**
 * @brief NodeLocalTP test configurations
 *
 * Each config specifies:
 * - devices: What device type each rank uses (typically CPU for NodeLocalTP)
 * - parallelism: Parallelism::NodeLocalTP
 * - collective: Collective::MPI
 * - mpi_ranks: Required number of MPI ranks
 */
static const std::vector<TestConfig> kNodeLocalTPTestConfigs = {
    // =========================================================================
    // NodeLocalTP with CPU (UPI interconnect)
    // =========================================================================
    // 2-way Node-Local TP with CPU devices using MPI collectives
    // This is the primary use case: CPU-only tensor parallelism
    // across NUMA nodes connected via UPI (~50 GB/s)
    {
        .name = "NodeLocalTP_2xMPI_CPU",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeLocalTP,
        .collective = Collective::MPI,
        .thresholds = {
            .cosine_threshold = 0.99f,
            .decode_cosine_threshold = 0.98f,
            .early_layers_count = 4,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.20f,
            .excluded_stages = kNodeLocalTPExcludedStages,
        },
        .mpi_ranks = 2,
    },

    // 4-way Node-Local TP with CPU (for larger models, if 4 NUMA nodes available)
    {
        .name = "NodeLocalTP_4xMPI_CPU",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU,
                    ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeLocalTP,
        .collective = Collective::MPI,
        .thresholds = {
            .cosine_threshold = 0.98f,
            .decode_cosine_threshold = 0.95f,
            .early_layers_count = 4,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.30f,  // More variance with 4-way sharding
            .excluded_stages = kNodeLocalTPExcludedStages,
        },
        .mpi_ranks = 4,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

/**
 * @brief Parameterized test fixture for NodeLocalTP parity tests
 *
 * Inherits from ConfigDrivenParityTest which handles all setup/teardown
 * including MPI context creation and GlobalTPContext initialization.
 */
class Qwen2NodeLocalTPParityTest : public ConfigDrivenParityTest<Qwen2NodeLocalTPParityTest>,
                                    public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Verify NodeLocalTP infrastructure initialization
 *
 * Tests that:
 * - GlobalTPContext is created successfully on each rank
 * - MPI communicator is valid
 * - Rank indices are correct
 */
TEST_P(Qwen2NodeLocalTPParityTest, NodeLocalTPContextInitialization)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    // Verify MPI context
    ASSERT_NE(mpi_ctx_, nullptr) << "MPI context should be initialized";
    EXPECT_GE(mpi_ctx_->world_size(), cfg().mpi_ranks)
        << "World size should be at least " << cfg().mpi_ranks;

    // Verify GlobalTPContext (implements cross-rank TP communication)
    ASSERT_NE(global_tp_ctx_, nullptr) << "GlobalTPContext should be created";
    EXPECT_EQ(global_tp_ctx_->degree(), mpi_ctx_->world_size())
        << "TP degree should match world size";
    EXPECT_EQ(global_tp_ctx_->myIndex(), mpi_ctx_->rank())
        << "TP index should match MPI rank";
    EXPECT_FALSE(global_tp_ctx_->isLocal())
        << "GlobalTPContext should not be local";
    EXPECT_EQ(global_tp_ctx_->backend(), CollectiveBackendType::UPI)
        << "NodeLocalTP should use UPI backend";

    LOG_INFO("[NodeLocalTP] Rank " << mpi_ctx_->rank() << " verified TPContext");
}

/**
 * @brief Test GlobalTP allreduce operation
 *
 * Each rank creates a tensor with its rank value, performs allreduce,
 * and verifies the result is the sum of all ranks.
 */
TEST_P(Qwen2NodeLocalTPParityTest, NodeLocalTPAllreduce)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "GlobalTPContext required";

    // Create a simple tensor with rank value
    auto tensor_factory = std::make_unique<TensorFactory>(*mpi_ctx_);
    auto tensor = tensor_factory->createFP32({1, 128});

    // Initialize with rank value
    float rank_value = static_cast<float>(mpi_ctx_->rank() + 1);
    std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), rank_value);

    LOG_INFO("[NodeLocalTP] Rank " << mpi_ctx_->rank() << " input value: " << rank_value);

    // Perform allreduce
    ASSERT_TRUE(global_tp_ctx_->allreduce(tensor.get())) << "Allreduce failed";

    // Verify: sum of 1 + 2 + ... + world_size
    int world_size = mpi_ctx_->world_size();
    float expected_sum = static_cast<float>(world_size * (world_size + 1)) / 2.0f;

    float actual_value = tensor->data()[0];
    EXPECT_NEAR(actual_value, expected_sum, 1e-5f)
        << "Allreduce result should be sum of ranks";

    LOG_INFO("[NodeLocalTP] Rank " << mpi_ctx_->rank() << " allreduce result: "
             << actual_value << " (expected: " << expected_sum << ")");
}

/**
 * @brief Test GlobalTP broadcast operation
 *
 * Rank 0 broadcasts a tensor with specific values, all ranks verify receipt.
 */
TEST_P(Qwen2NodeLocalTPParityTest, NodeLocalTPBroadcast)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "GlobalTPContext required";

    auto tensor_factory = std::make_unique<TensorFactory>(*mpi_ctx_);
    auto tensor = tensor_factory->createFP32({1, 64});

    const float broadcast_value = 42.0f;
    int source_rank = 0;

    if (mpi_ctx_->rank() == source_rank)
    {
        // Source rank initializes tensor
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), broadcast_value);
        LOG_INFO("[NodeLocalTP] Rank " << mpi_ctx_->rank() << " broadcasting value: " << broadcast_value);
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

    LOG_INFO("[NodeLocalTP] Rank " << mpi_ctx_->rank() << " received: " << received_value);
}

/**
 * @brief Test GlobalTP barrier synchronization
 */
TEST_P(Qwen2NodeLocalTPParityTest, NodeLocalTPBarrier)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "GlobalTPContext required";

    LOG_INFO("[NodeLocalTP] Rank " << mpi_ctx_->rank() << " entering barrier");

    // This should not deadlock or hang
    global_tp_ctx_->barrier();

    LOG_INFO("[NodeLocalTP] Rank " << mpi_ctx_->rank() << " exited barrier");
}

/**
 * @brief Prefill parity test with NodeLocalTP
 *
 * Runs prefill inference with cross-rank TP sharding and compares against
 * PyTorch reference. Each rank runs its own sharded pipeline; post-allreduce
 * activations (norms, residuals) and post-allgather logits match PyTorch.
 * Sharded intermediate stages are excluded from comparison.
 */
TEST_P(Qwen2NodeLocalTPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

/**
 * @brief Decode parity test with NodeLocalTP
 *
 * Runs prefill + incremental decode and compares logit distributions
 * against PyTorch reference at each decode step.
 */
TEST_P(Qwen2NodeLocalTPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen2NodeLocalTP,
    Qwen2NodeLocalTPParityTest,
    ::testing::ValuesIn(kNodeLocalTPTestConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

/**
 * @brief Custom main() with MPI initialization for NodeLocalTP tests
 *
 * NodeLocalTP tests REQUIRE MPI to be initialized with multiple ranks.
 * Run with: mpirun -np 2 ./v2_integration_node_local_tp_parity
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

    // Finalize MPI
    MPI_Finalize();

    return global_result;
}
