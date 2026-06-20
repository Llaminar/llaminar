/**
 * @file Test__Qwen35MoE_NodeLocalTP_Parity.cpp
 * @brief Node-Local TP parity tests for Qwen3.5 MoE using MPI (multi-rank, same node)
 *
 * These tests validate Node-Local Tensor Parallelism (NodeLocalTP) infrastructure
 * for Qwen3.5 MoE, where TP spans multiple MPI ranks on the same physical node.
 *
 * TP strategy for MoE:
 *   - Expert Parallelism (EP): Each rank has ALL expert weights (replicated) but
 *     only COMPUTES its assigned subset. Output is a partial sum.
 *   - Shared Expert: Standard Megatron TP (gate/up = ColumnParallel, down = InputParallel).
 *     Output is a partial sum.
 *   - Shared Expert Sigmoid Gate: Distributes over partial sums (s⊙(y₀+y₁) = s⊙y₀ + s⊙y₁),
 *     so gating happens before allreduce.
 *   - Single allreduce per layer on attn_proj (combines EP partial + shared expert TP partial).
 *
 * NOTE: CPU TP MUST use NodeLocalTP (MPI), not LocalTP, because
 * DeviceId::cpu() is a singleton. Each MPI rank gets its own process
 * with distinct WeightManager.
 *
 * REQUIREMENTS:
 *   - Must run via ctest for proper MPI rank settings and initialization
 *   - Model at /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35MoEParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

// =============================================================================
// Common Excluded Stages for NodeLocalTP MoE
// =============================================================================

// For NodeLocalTP, intermediate activations are SHARDED across ranks. Only the final
// LM_HEAD output (after allgather) can be compared against PyTorch reference.
// Includes GDN-specific stages (same as dense Qwen3.5) plus MoE-specific partials.
static const std::vector<std::string> kNodeLocalTPMoEExcludedStages = {
    // --- Standard attention projections (FA layers) — column-parallel slices ---
    "Q_PROJECTION",
    "K_PROJECTION",
    "V_PROJECTION",
    "Q_NORM",
    "K_NORM",
    "Q_ROPE",
    "K_ROPE",
    "ATTENTION_CONTEXT",
    "FA_GATE",
    "ATTENTION_CONTEXT_GATED",
    // --- FFN sharded intermediates (shared expert TP) ---
    "FFN_GATE",
    "FFN_UP",
    "FFN_SWIGLU",
    // --- GDN-specific: column-parallel slices ---
    "QKV_PROJECTION",
    "GDN_CONV1D_OUTPUT",
    "GDN_Z_PROJECTION",
    "GDN_DELTA_RULE_OUTPUT",
    "GDN_NORM_GATE_OUTPUT",
};

// MoE EP/TP partial-sum stages: each rank holds a partial contribution.
// Allreduce (SUM) across ranks before comparing to PyTorch reference.
static const std::vector<std::string> kNodeLocalTPMoEAllreduceStages = {
    "MOE_EXPERT_OUTPUT",        // EP partial sum (only local experts computed)
    "MOE_SHARED_EXPERT_OUTPUT", // Shared expert TP partial sum (down_proj row-parallel)
    "MOE_SHARED_GATE_OUTPUT",   // Sigmoid-gated partial sum (s⊙y distributes over partials)
    "MOE_COMBINED_OUTPUT",      // Total partial (expert EP + shared TP + gate)
};

// =============================================================================
// Test Configuration Definitions
// =============================================================================

static const std::vector<TestConfig> kNodeLocalTPMoETestConfigs = {
    // =========================================================================
    // Qwen3.5-35B MoE (Q4_K_XL) — 2-way Node-Local TP with CPU (UPI interconnect)
    //
    // This is the primary TP parity configuration for MoE. Each MPI rank
    // computes half the experts (EP) and half the shared expert intermediate
    // (Megatron TP), then allreduces attn_proj once per layer.
    //
    // Thresholds calibrated from observed results (2026-04-26):
    //   Prefill: worst min cosine 0.9335 (L15, MOE_EXPERT_OUTPUT)
    //   Decode:  worst min cosine 0.8298 (L15, MOE_EXPERT_OUTPUT)
    //   LM_HEAD: cosine 0.9978, KL 0.0057, Top-1 100%, Top-5 100%
    //   Decode:  5/5 steps, AvgCosine 0.9970, Top-1 100%, RefInTop3 5/5
    // Note: MOE_ROUTING_INDICES uses set-overlap (Jaccard) and is excluded
    //       from layer-level aggregation by ParityTestBase.
    // =========================================================================
    {
        .name = "NodeLocalTP_2xMPI_CPU_35B_MoE",
        .devices = {ParityDeviceType::CPU, ParityDeviceType::CPU},
        .parallelism = Parallelism::NodeLocalTP,
        .collective = Collective::MPI,
        .thresholds = {
            .cosine_threshold = 0.90f,        // Worst observed: 0.9335 (3.5% margin)
            .decode_cosine_threshold = 0.80f, // Worst observed: 0.8298 (3.5% margin)
            .early_layers_count = 6,
            .min_early_layers_passed = 5, // 6/6 observed; require 5
            .kl_threshold = 0.03f,        // LM_HEAD observed: 0.005-0.032 (run-to-run variance with dynamic rebalance)
            .excluded_stages = kNodeLocalTPMoEExcludedStages,
            .allreduce_stages = kNodeLocalTPMoEAllreduceStages,
            .min_top1_accuracy = 0.80f, // Observed: 100%; allow 1 miss in 5
            .min_top5_accuracy = 0.80f, // Observed: 100%; allow 1 miss in 5
            .pytorch_top1_in_topk = 4,  // Observed: 5/5 RefInTop3; require 3
        },
        .mpi_ranks = 2,
        .model_path = "/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        .snapshot_dir = "pytorch_qwen35_moe_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35MoENodeLocalTPParityTest : public Qwen35MoEConfigDrivenParityTest<Qwen35MoENodeLocalTPParityTest>,
                                       public ::testing::WithParamInterface<TestConfig>
{
public:
    const TestConfig &getTestConfig() const { return GetParam(); }
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Verify NodeLocalTP infrastructure initialization for MoE
 */
TEST_P(Qwen35MoENodeLocalTPParityTest, NodeLocalTPContextInitialization)
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

    LOG_INFO("[NodeLocalTP Qwen3.5 MoE] Rank " << mpi_ctx_->rank() << " verified TPContext");
}

/**
 * @brief Prefill parity test with NodeLocalTP for MoE
 *
 * Runs prefill inference with cross-rank EP + shared expert TP sharding and
 * compares against PyTorch reference. Post-allreduce activations (norms,
 * residuals) and post-allgather logits should match PyTorch.
 */
TEST_P(Qwen35MoENodeLocalTPParityTest, PrefillParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runPrefillParity();
    assertParity(summary);
}

/**
 * @brief Decode parity test with NodeLocalTP for MoE
 *
 * Runs prefill + incremental decode and compares logit distributions
 * against PyTorch reference at each decode step.
 */
TEST_P(Qwen35MoENodeLocalTPParityTest, DecodeParity)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";
    auto summary = runDecodeParity();
    assertDecodeParity(summary);
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoENodeLocalTP,
    Qwen35MoENodeLocalTPParityTest,
    ::testing::ValuesIn(kNodeLocalTPMoETestConfigs),
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
        std::cout << "║     QWEN3.5 MoE NODE-LOCAL TP PARITY TEST SUITE                  ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  MPI world size: " << world_size << " ranks" << std::string(42 - std::to_string(world_size).length(), ' ') << "║\n";
        std::cout << "║  Thread support: " << (provided >= MPI_THREAD_MULTIPLE ? "MPI_THREAD_MULTIPLE" : "limited") << std::string(26, ' ') << "║\n";
        std::cout << "║  TP strategy:   EP (experts) + Megatron (shared expert)" << std::string(7, ' ') << "║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    int global_result;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(global_result);
}
