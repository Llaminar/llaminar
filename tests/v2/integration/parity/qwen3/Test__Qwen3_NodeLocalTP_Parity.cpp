/**
 * @file Test__Qwen3_NodeLocalTP_Parity.cpp
 * @brief Node-Local TP parity tests for Qwen3 using MPI (multi-rank, same node)
 *
 * These tests validate Node-Local Tensor Parallelism (NodeLocalTP) infrastructure
 * for Qwen3 architecture (per-head QK RMSNorm, no QKV biases), where tensor
 * parallelism spans multiple MPI ranks on the same physical node. Unlike LocalTP
 * which uses NCCL/RCCL/HOST for intra-process multi-device communication,
 * NodeLocalTP uses MPI collectives for cross-rank communication — the correct
 * approach for multi-socket CPU TP.
 *
 * NOTE: CPU TP MUST use NodeLocalTP (MPI), not LocalTP, because
 * DeviceId::cpu() is a singleton — LocalTP cannot distinguish multiple
 * CPU sockets. Each MPI rank gets its own process with distinct
 * WeightManager, so weight sharding works correctly.
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - Each rank participates in NodeLocalTP collective operations (via GlobalTPContext)
 *
 * Test configurations:
 *   - NodeLocalTP_2xMPI_CPU: 2 MPI ranks, CPU, Qwen3-0.6B Q8_0
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen3ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen3;

// =============================================================================
// Common Excluded Stages for NodeLocalTP
// =============================================================================

// For NodeLocalTP, intermediate activations are SHARDED across ranks. Only the final
// LM_HEAD output (after allgather) can be compared against PyTorch reference.
// ROW_PARALLEL outputs (ATTENTION_OUTPUT, FFN_DOWN) contain partial sums before allreduce.
// COLUMN_PARALLEL outputs are local slices that need to be gathered.
static const std::vector<std::string> kNodeLocalTPExcludedStages = {
    // Column-parallel: outputs are sharded slices
    "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "Q_NORM", "K_NORM", // Qwen3 per-head norms are also head-sharded
    "Q_ROPE", "K_ROPE",
    "ATTENTION_CONTEXT",
    "FFN_GATE", "FFN_UP", "FFN_SWIGLU",
    // Row-parallel: outputs are partial sums before allreduce
    "ATTENTION_OUTPUT", "FFN_DOWN",
    // These also have sharded intermediate states
    "ATTN_RESIDUAL", "FFN_RESIDUAL"};

// =============================================================================
// Test Configuration Definitions
// =============================================================================

static const std::vector<TestConfig> kNodeLocalTPTestConfigs = {
    // =========================================================================
    // Qwen3-0.6B (Q8_0) — 2-way Node-Local TP with CPU (UPI interconnect)
    // =========================================================================
    {
        .name = "NodeLocalTP_2xMPI_CPU",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeLocalTP,
        .collective = Collective::MPI,
        .thresholds = {
            .cosine_threshold = 0.94f,
            .decode_cosine_threshold = 0.90f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.012f, // Observed: 0.003 prefill KL (was 0.20 = 80x over-relaxed)
            .excluded_stages = kNodeLocalTPExcludedStages,
        },
        .mpi_ranks = 2,
        .model_path = "models/Qwen3-0.6B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen3_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen3NodeLocalTPParityTest : public Qwen3ConfigDrivenParityTest<Qwen3NodeLocalTPParityTest>,
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
 */
TEST_P(Qwen3NodeLocalTPParityTest, NodeLocalTPContextInitialization)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    ASSERT_NE(mpi_ctx_, nullptr) << "MPI context should be initialized";
    EXPECT_GE(mpi_ctx_->world_size(), cfg().mpi_ranks)
        << "World size should be at least " << cfg().mpi_ranks;

    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeLocalTP context should be created";
    EXPECT_EQ(global_tp_ctx_->degree(), mpi_ctx_->world_size())
        << "TP degree should match world size";
    EXPECT_EQ(global_tp_ctx_->myIndex(), mpi_ctx_->rank())
        << "TP index should match MPI rank";
    EXPECT_FALSE(global_tp_ctx_->isLocal())
        << "NodeLocalTP context should not be local (uses MPI, not intra-process)";
    EXPECT_EQ(global_tp_ctx_->backend(), CollectiveBackendType::UPI)
        << "NodeLocalTP should use UPI backend";

    LOG_INFO("[NodeLocalTP Qwen3] Rank " << mpi_ctx_->rank() << " verified TPContext");
}

/**
 * @brief Test NodeLocalTP allreduce operation
 */
TEST_P(Qwen3NodeLocalTPParityTest, NodeLocalTPAllreduce)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeLocalTP context required";

    auto tensor_factory = std::make_unique<TensorFactory>(*mpi_ctx_);
    auto tensor = tensor_factory->createFP32({1, 128});

    float rank_value = static_cast<float>(mpi_ctx_->rank() + 1);
    std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), rank_value);

    ASSERT_TRUE(global_tp_ctx_->allreduce(tensor.get())) << "Allreduce failed";

    int world_size = mpi_ctx_->world_size();
    float expected_sum = static_cast<float>(world_size * (world_size + 1)) / 2.0f;
    float actual_value = tensor->data()[0];
    EXPECT_NEAR(actual_value, expected_sum, 1e-5f)
        << "Allreduce result should be sum of ranks";

    LOG_INFO("[NodeLocalTP Qwen3] Rank " << mpi_ctx_->rank() << " allreduce result: "
                                         << actual_value << " (expected: " << expected_sum << ")");
}

/**
 * @brief Test NodeLocalTP broadcast operation
 */
TEST_P(Qwen3NodeLocalTPParityTest, NodeLocalTPBroadcast)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeLocalTP context required";

    auto tensor_factory = std::make_unique<TensorFactory>(*mpi_ctx_);
    auto tensor = tensor_factory->createFP32({1, 64});

    const float broadcast_value = 42.0f;
    int source_rank = 0;

    if (mpi_ctx_->rank() == source_rank)
    {
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), broadcast_value);
    }
    else
    {
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), 0.0f);
    }

    ASSERT_TRUE(global_tp_ctx_->broadcast(tensor.get(), source_rank)) << "Broadcast failed";

    float received_value = tensor->data()[0];
    EXPECT_NEAR(received_value, broadcast_value, 1e-5f)
        << "Broadcast should deliver source rank's value";

    LOG_INFO("[NodeLocalTP Qwen3] Rank " << mpi_ctx_->rank() << " received: " << received_value);
}

/**
 * @brief Test NodeLocalTP barrier synchronization
 */
TEST_P(Qwen3NodeLocalTPParityTest, NodeLocalTPBarrier)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeLocalTP context required";

    LOG_INFO("[NodeLocalTP Qwen3] Rank " << mpi_ctx_->rank() << " entering barrier");
    global_tp_ctx_->barrier();
    LOG_INFO("[NodeLocalTP Qwen3] Rank " << mpi_ctx_->rank() << " exited barrier");
}

/**
 * @brief Prefill parity test with NodeLocalTP
 */
TEST_P(Qwen3NodeLocalTPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

/**
 * @brief Decode parity test with NodeLocalTP
 */
TEST_P(Qwen3NodeLocalTPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen3NodeLocalTP,
    Qwen3NodeLocalTPParityTest,
    ::testing::ValuesIn(kNodeLocalTPTestConfigs),
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
        std::cout << "║         QWEN3 NODE-LOCAL TP PARITY TEST SUITE                    ║\n";
        std::cout << "║                                                                  ║\n";
        std::cout << "║  MPI ranks: " << world_size << "                                                   ║\n";
        std::cout << "║  Testing: Cross-rank CPU tensor parallelism via UPI              ║\n";
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
