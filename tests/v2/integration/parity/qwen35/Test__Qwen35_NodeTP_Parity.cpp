/**
 * @file Test__Qwen35_NodeTP_Parity.cpp
 * @brief Qwen3.5 NodeTP parity tests across MPI ranks on one node.
 *
 * These tests validate Node-Local Tensor Parallelism (NodeTP) infrastructure
 * for Qwen3.5 GDN+FA hybrid architecture, where tensor parallelism spans
 * multiple MPI ranks on the same physical node. Unlike LocalTP which uses
 * NCCL/RCCL/HOST for intra-process multi-device communication, NodeTP
 * uses MPI collectives for cross-rank communication — the correct approach
 * for multi-socket CPU TP.
 *
 * NOTE: CPU TP MUST use NodeTP (MPI), not LocalTP, because
 * DeviceId::cpu() is a singleton — LocalTP cannot distinguish multiple
 * CPU sockets. Each MPI rank gets its own process with distinct
 * WeightManager, so weight sharding works correctly.
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - Each rank participates in NodeTP collective operations (via GlobalTPContext)
 *
 * Test configurations:
 *   - NodeTP_2xMPI_CPU_08B: 2 MPI ranks, CPU, Qwen3.5-0.8B Q4_0
 *   - NodeTP_2xMPI_CPU_4B:  2 MPI ranks, CPU, Qwen3.5-4B Q8_0
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35ParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35;

// =============================================================================
// Common Excluded Stages for NodeTP
// =============================================================================

// For NodeTP, intermediate activations are SHARDED across ranks. Only the final
// LM_HEAD output (after allgather) can be compared against PyTorch reference.
// ROW_PARALLEL outputs (ATTENTION_OUTPUT, FFN_DOWN) contain partial sums before allreduce.
// COLUMN_PARALLEL outputs are local slices that need to be gathered.
// Includes GDN-specific stages that are also sharded under TP.
static const std::vector<std::string> kNodeTPExcludedStages = {
    // Standard attention projections (FA layers) — column-parallel slices
    "Q_PROJECTION",
    "K_PROJECTION",
    "V_PROJECTION",
    "Q_NORM",
    "K_NORM", // Qwen3.5 per-head norms are head-sharded in TP
    "Q_ROPE",
    "K_ROPE",
    "ATTENTION_CONTEXT",
    "FA_GATE",
    "ATTENTION_CONTEXT_GATED",
    // FFN sharded intermediates
    "FFN_GATE",
    "FFN_UP",
    "FFN_SWIGLU",
    // NOTE: ATTN_RESIDUAL and FFN_RESIDUAL are post-allreduce and should be
    // replicated — we include them in the comparison to help isolate TP bugs.
    // "ATTN_RESIDUAL",
    // "FFN_RESIDUAL",
    // GDN-specific: QKV projection covers gdn_proj (same snapshot key)
    "QKV_PROJECTION",
    // GDN-specific: short-conv preserves the sharded QKV packed layout
    "GDN_CONV1D_OUTPUT",
    // GDN-specific: Z gate output is column-parallel (sharded by v_heads)
    "GDN_Z_PROJECTION",
    // GDN-specific: recurrence output is per-local-heads under TP
    "GDN_DELTA_RULE_OUTPUT",
    // GDN-specific: gated norm output is per-local-heads under TP
    "GDN_NORM_GATE_OUTPUT",
};

// Row-parallel projection snapshots are captured before the production
// collective. The parity harness reconstructs their semantic tensors with the
// same fixed rank order through MPI, while ATTN_RESIDUAL and FFN_RESIDUAL above
// independently prove that the live post-collective production values were
// consumed by the following graph stages.
static const std::vector<std::string> kNodeTPAllreduceStages = {
    "ATTENTION_OUTPUT",
    "FFN_DOWN",
};

// =============================================================================
// Test Configuration Definitions
// =============================================================================

static const std::vector<TestConfig> kNodeTPTestConfigs = {
    // =========================================================================
    // Qwen3.5-0.8B (Q4_0) — 2-way Node-Local TP with CPU (UPI interconnect)
    // =========================================================================
    {
        .name = "NodeTP_2xMPI_CPU_08B",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeTP,
        .collective = Collective::MPI,
        .thresholds = {
            .cosine_threshold = 0.96f,        // Observed: 0.999 prefill cosine (was 0.90)
            .decode_cosine_threshold = 0.96f, // Observed: 0.994 avg decode cosine (was 0.90)
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.012f, // Observed: 0.002 prefill KL (was 0.35 = 148x over-relaxed)
            .excluded_stages = kNodeTPExcludedStages,
            .allreduce_stages = kNodeTPAllreduceStages,
        },
        .mpi_ranks = 2,
        .model_path = "models/Qwen3.5-0.8B-Q4_0.gguf",
        .snapshot_dir = "pytorch_qwen35_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },

    // =========================================================================
    // Qwen3.5-4B (Q8_0) — 2-way Node-Local TP with CPU (UPI interconnect)
    // =========================================================================
    {
        .name = "NodeTP_2xMPI_CPU_4B",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeTP,
        .collective = Collective::MPI,
        .thresholds = {
            .cosine_threshold = 0.96f,        // Observed: 0.998 prefill cosine (was 0.90)
            .decode_cosine_threshold = 0.96f, // Observed: 0.997 avg decode cosine (was 0.90)
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.015f, // Observed: 0.004 prefill KL (was 0.35 = 94x over-relaxed)
            .excluded_stages = kNodeTPExcludedStages,
            .allreduce_stages = kNodeTPAllreduceStages,
        },
        .mpi_ranks = 2,
        .model_path = "models/Qwen3.5-4B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_4b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },

    // =========================================================================
    // Qwen3.5-27B (Q8_0) — 2-way Node-Local TP with CPU (UPI interconnect)
    //
    // 27B is the critical 27B-sized model where GDN under TP=2 exhibits
    // non-integer V/K head ratios (n_v_heads_local=24, n_k_heads=16) that
    // previously triggered expansion-branch bugs. This test is the primary
    // parity guard for the GDN deinterleave modular-mapping path.
    // =========================================================================
    {
        .name = "NodeTP_2xMPI_CPU_27B",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeTP,
        .collective = Collective::MPI,
        .thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.96f,
            .early_layers_count = 6,
            .min_early_layers_passed = 4,
            .kl_threshold = 0.020f,
            .excluded_stages = kNodeTPExcludedStages,
            .allreduce_stages = kNodeTPAllreduceStages,
        },
        .mpi_ranks = 2,
        .model_path = "models/Qwen3.5-27B-Q8_0.gguf",
        .snapshot_dir = "pytorch_qwen35_27b_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35NodeTPParityTest : public Qwen35ConfigDrivenParityTest<Qwen35NodeTPParityTest>,
                                    public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

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
TEST_P(Qwen35NodeTPParityTest, NodeTPContextInitialization)
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

    LOG_INFO("[NodeTP Qwen3.5] Rank " << mpi_ctx_->rank() << " verified TPContext");
}

/**
 * @brief Test NodeTP allreduce operation
 *
 * Each rank creates a tensor with its rank value, performs allreduce,
 * and verifies the result is the sum of all ranks.
 */
TEST_P(Qwen35NodeTPParityTest, NodeTPAllreduce)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeTP context required";

    // Create a simple tensor with rank value
    auto tensor_factory = std::make_unique<TensorFactory>(*mpi_ctx_);
    auto tensor = tensor_factory->createFP32({1, 128});

    // Initialize with rank value
    float rank_value = static_cast<float>(mpi_ctx_->rank() + 1);
    std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), rank_value);

    LOG_INFO("[NodeTP Qwen3.5] Rank " << mpi_ctx_->rank() << " input value: " << rank_value);

    // Perform allreduce
    ASSERT_TRUE(global_tp_ctx_->allreduce(tensor.get())) << "Allreduce failed";

    // Verify: sum of 1 + 2 + ... + world_size
    int world_size = mpi_ctx_->world_size();
    float expected_sum = static_cast<float>(world_size * (world_size + 1)) / 2.0f;

    float actual_value = tensor->data()[0];
    EXPECT_NEAR(actual_value, expected_sum, 1e-5f)
        << "Allreduce result should be sum of ranks";

    LOG_INFO("[NodeTP Qwen3.5] Rank " << mpi_ctx_->rank() << " allreduce result: "
                                           << actual_value << " (expected: " << expected_sum << ")");
}

/**
 * @brief Test NodeTP broadcast operation
 *
 * Rank 0 broadcasts a tensor with specific values, all ranks verify receipt.
 */
TEST_P(Qwen35NodeTPParityTest, NodeTPBroadcast)
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
        LOG_INFO("[NodeTP Qwen3.5] Rank " << mpi_ctx_->rank() << " broadcasting value: " << broadcast_value);
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

    LOG_INFO("[NodeTP Qwen3.5] Rank " << mpi_ctx_->rank() << " received: " << received_value);
}

/**
 * @brief Test NodeTP barrier synchronization
 */
TEST_P(Qwen35NodeTPParityTest, NodeTPBarrier)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    ASSERT_NE(global_tp_ctx_, nullptr) << "NodeTP context required";

    LOG_INFO("[NodeTP Qwen3.5] Rank " << mpi_ctx_->rank() << " entering barrier");

    // This should not deadlock or hang
    global_tp_ctx_->barrier();

    LOG_INFO("[NodeTP Qwen3.5] Rank " << mpi_ctx_->rank() << " exited barrier");
}

/**
 * @brief Full prefill/decode production parity with cross-rank TP sharding.
 */
TEST_P(Qwen35NodeTPParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen35NodeTP,
    Qwen35NodeTPParityTest,
    ::testing::ValuesIn(kNodeTPTestConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &info)
    {
        return info.param.name;
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

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
        std::cout << "║       QWEN3.5 NODE-LOCAL TP PARITY TEST SUITE                    ║\n";
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
