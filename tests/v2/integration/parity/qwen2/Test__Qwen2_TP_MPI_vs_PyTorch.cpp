/**
 * @file Test__Qwen2_TP_MPI_vs_PyTorch.cpp
 * @brief Integration: Cross-Vendor CUDA+ROCm Tensor-Parallel with MPI Backend
 *
 * =============================================================================
 * EXPLICIT TEST INTENT
 * =============================================================================
 *
 * This test validates CROSS-VENDOR tensor parallelism:
 *   - Rank 0: NVIDIA GPU (CUDA)
 *   - Rank 1: AMD GPU (ROCm)
 *   - Backend: MPI with HOST STAGING (GPU→Host→MPI→Host→GPU)
 *
 * This is the ONLY test that exercises heterogeneous cross-vendor TP.
 * It requires BOTH CUDA and ROCm hardware to be present.
 *
 * =============================================================================
 * HARDWARE REQUIREMENTS
 * =============================================================================
 *
 *   - At least 1 NVIDIA GPU (for rank 0)
 *   - At least 1 AMD GPU (for rank 1)
 *   - MPI runtime with 2 ranks
 *
 * If these requirements are not met, the test SKIPS (not fails).
 *
 * =============================================================================
 * ARCHITECTURE
 * =============================================================================
 *
 *   Scope: GLOBAL (multiple MPI ranks, CollectiveScope::GLOBAL)
 *   Backend: MPI with host staging (required for cross-vendor)
 *
 *   Data flow for AllReduce:
 *     CUDA buffer → cudaMemcpyD2H → Host staging → MPI_Allreduce
 *       → Host staging → hipMemcpyH2D → ROCm buffer
 *
 * =============================================================================
 * WHAT THIS TESTS
 * =============================================================================
 *
 *   - Weight sharding (column/row parallel) across heterogeneous GPUs
 *   - AllReduce after row-parallel GEMM (Wo, FFN_down) via MPI host staging
 *   - AllGather after column-parallel LM head
 *   - Numerical parity with PyTorch ground truth
 *   - Cross-vendor kernel consistency (CUDA vs ROCm produce same results)
 *
 * =============================================================================
 * RELATED TESTS
 * =============================================================================
 *
 *   - Test__Qwen2_TP_NCCL_vs_PyTorch: LOCAL scope, 2 CUDA GPUs, NCCL backend
 *   - Test__Qwen2_TP_RCCL_vs_PyTorch: LOCAL scope, 2 ROCm GPUs, RCCL backend
 *   - Test__Qwen2_TP_PCIeBAR_vs_PyTorch: LOCAL scope, CUDA+ROCm, PCIeBAR backend
 *
 * @author David Sanftenberg
 * @date 2026-01-24
 */

#include "Qwen2ParityTestBase.h"
#include "models/qwen/Qwen2Schema.h"
#include "loaders/WeightManager.h"
#include "utils/MPITopology.h"
#include "backends/ComputeBackend.h"

// NOTE: We cannot include both cuda_runtime.h and hip/hip_runtime_api.h
// in the same translation unit due to conflicting type definitions (dim3, etc.)
// Use DeviceManager for device enumeration instead.

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen2;

namespace
{

    /**
     * @brief Compute max absolute difference (for rank consistency check)
     */
    float max_abs_diff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    /**
     * @brief Check hardware availability for cross-vendor TP
     * @return pair<cuda_device_id, rocm_device_id> or {-1,-1} if unavailable
     *
     * Uses DeviceManager to enumerate devices (avoids CUDA/HIP header conflict)
     */
    std::pair<int, int> findCrossVendorDevices()
    {
        auto &dm = DeviceManager::instance();
        dm.initialize(-1); // Initialize all devices

        int cuda_count = dm.cuda_device_count();
        int rocm_count = dm.rocm_device_count();

        if (cuda_count < 1)
        {
            LOG_WARN("[CrossVendor] No CUDA devices found");
            return {-1, -1};
        }

        if (rocm_count < 1)
        {
            LOG_WARN("[CrossVendor] No ROCm devices found");
            return {-1, -1};
        }

        LOG_INFO("[CrossVendor] Found " << cuda_count << " CUDA device(s) and "
                                        << rocm_count << " ROCm device(s)");

        // Find first CUDA and first ROCm device
        int cuda_id = -1;
        int rocm_id = -1;

        for (const auto &dev : dm.devices())
        {
            if (dev.type == ComputeBackendType::GPU_CUDA && cuda_id < 0)
            {
                cuda_id = dev.device_id;
            }
            else if (dev.type == ComputeBackendType::GPU_ROCM && rocm_id < 0)
            {
                rocm_id = dev.device_id;
            }
        }

        return {cuda_id, rocm_id};
    }

} // namespace

/**
 * @brief Test fixture for Cross-Vendor CUDA+ROCm Tensor-Parallel Parity
 *
 * EXPLICIT REQUIREMENTS:
 *   - Rank 0: CUDA device
 *   - Rank 1: ROCm device
 *   - MPI host-staged allreduce between them
 *
 * This test SKIPS if cross-vendor hardware is not available.
 */
class Test__Qwen2_TP_MPI_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    // MPI topology for device selection
    std::unique_ptr<MPITopology> topology_;

    // Device assignments (populated in SetUp)
    int cuda_device_id_ = -1;
    int rocm_device_id_ = -1;

    // ==========================================================================
    // Qwen2ParityTestBase overrides
    // ==========================================================================

    BackendThresholds getBackendThresholds() override
    {
        // Tensor parallelism introduces numerical differences from:
        //   1. Weight sharding (different accumulation order in GEMM)
        //   2. AllReduce rounding (FP32 reduction order)
        //   3. Cross-vendor heterogeneity (CUDA vs ROCm kernel differences)
        //
        // Observed results (Qwen2.5-0.5B, 2 ranks):
        //   - CUDA-only TP:      cosine=0.977, KL=0.031, top5_overlap=4/5
        //   - Cross-vendor TP:   cosine=0.950, KL=0.080, top5_overlap=4/5
        //   - Single-rank CUDA:  cosine=0.978, KL=0.032 (reference)
        //
        // IMPORTANT: For GLOBAL scope MPI (multi-rank), many stages produce PARTIAL
        // outputs on each rank due to column-parallel sharding:
        //   - Q/K/V projections: each rank has local_n_heads, not total n_heads
        //   - Q_ROPE, K_ROPE: derived from sharded Q/K
        //   - ATTENTION_CONTEXT: head dimension sharded
        //   - FFN_GATE, FFN_UP, FFN_SWIGLU: intermediate_size sharded
        //
        // These cannot be directly compared against FULL PyTorch outputs.
        // Parity is verified through:
        //   - ATTENTION_OUTPUT (after Wo allreduce)
        //   - FFN_DOWN (after down_proj allreduce)
        //   - ATTENTION_RESIDUAL, FFN_RESIDUAL (full tensors)
        //   - LM_HEAD (after allgather)
        return BackendThresholds{
            .cosine_threshold = 0.94f,        // Relaxed for cross-vendor
            .decode_cosine_threshold = 0.94f, // Relaxed for decode
            .early_layers_count = 6,
            .min_early_layers_passed = 6,
            .kl_threshold = 0.20f, // Relaxed for TP drift
            .excluded_stages = {
                // Column-parallel projections produce partial outputs per-rank
                "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
                // RoPE outputs derived from sharded Q/K (half the heads)
                "Q_ROPE", "K_ROPE",
                // Attention context has sharded head dimension
                "ATTENTION_CONTEXT",
                // FFN gate/up are column-parallel in TP
                "FFN_GATE", "FFN_UP",
                // SwiGLU output is also column-parallel (half intermediate_size)
                "FFN_SWIGLU"}};
    }

    std::string getBackendName() override
    {
        return "MPI_CrossVendor_TP(CUDA:rank0 + ROCm:rank1)";
    }

    /**
     * @brief Get device for this rank - EXPLICIT cross-vendor assignment
     *
     * This is the core of the test's intent:
     *   - Rank 0 → CUDA device
     *   - Rank 1 → ROCm device
     *
     * No fallbacks. No conditional compilation. No ambiguity.
     * If the expected device is not available, the test should have
     * already been skipped in SetUp().
     */
    DeviceId getDeviceForRank() override
    {
        if (mpiRank() == 0)
        {
            // Rank 0: CUDA (NVIDIA)
            LOG_INFO("[Rank 0] Using CUDA device " << cuda_device_id_);
            return DeviceId::cuda(cuda_device_id_);
        }
        else
        {
            // Rank 1+: ROCm (AMD)
            LOG_INFO("[Rank " << mpiRank() << "] Using ROCm device " << rocm_device_id_);
            return DeviceId::rocm(rocm_device_id_);
        }
    }

    // ==========================================================================
    // ParityTestBase overrides for tensor parallelism
    // ==========================================================================

    WeightDistributionStrategy getWeightStrategy() override
    {
        return WeightDistributionStrategy::SHARDED;
    }

    void configureModel(std::shared_ptr<ModelContext> model_ctx) override
    {
        // Configure weight sharding from Qwen2 schema
        Qwen2SchemaFactory schema_factory;
        model_ctx->weightManager()->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
    }

    // ==========================================================================
    // SetUp / TearDown
    // ==========================================================================

    void SetUp() override
    {
        // Initialize MPI context BEFORE calling parent SetUp
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // =====================================================================
        // REQUIREMENT 1: Exactly 2 MPI ranks
        // =====================================================================
        if (world_size != 2)
        {
            GTEST_SKIP() << "Cross-vendor TP test requires exactly 2 MPI ranks (got "
                         << world_size << ")";
        }

        // Create MPI context (must be done before parent SetUp)
        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);

        // =====================================================================
        // REQUIREMENT 2: Cross-vendor hardware (CUDA + ROCm)
        // =====================================================================
        auto [cuda_dev, rocm_dev] = findCrossVendorDevices();
        if (cuda_dev < 0 || rocm_dev < 0)
        {
            GTEST_SKIP() << "Cross-vendor TP test requires both CUDA and ROCm devices";
        }

        cuda_device_id_ = cuda_dev;
        rocm_device_id_ = rocm_dev;

        // =====================================================================
        // Log explicit test configuration
        // =====================================================================
        if (isRank0())
        {
            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║           CROSS-VENDOR TENSOR PARALLELISM TEST                   ║");
            LOG_INFO("╠══════════════════════════════════════════════════════════════════╣");
            LOG_INFO("║  Rank 0: CUDA device " << cuda_device_id_ << " (NVIDIA)                              ║");
            LOG_INFO("║  Rank 1: ROCm device " << rocm_device_id_ << " (AMD)                                 ║");
            LOG_INFO("║  Backend: MPI with host staging                                  ║");
            LOG_INFO("║  Collective: GPU→Host→MPI_Allreduce→Host→GPU                     ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");
        }

        // Call parent SetUp (regenerates snapshots on rank 0, etc.)
        Qwen2ParityTestBase::SetUp();
    }

    void TearDown() override
    {
        topology_.reset();
        Qwen2ParityTestBase::TearDown();
    }

    // ==========================================================================
    // TP-Specific Verification Helpers
    // ==========================================================================

    /**
     * @brief Verify all ranks agree on logits after AllGather
     */
    void verifyRankConsistency(const float *local_logits, size_t vocab_size)
    {
        // Gather logits from all ranks
        std::vector<float> all_logits(mpiWorldSize() * vocab_size);
        MPI_Allgather(
            local_logits, static_cast<int>(vocab_size), MPI_FLOAT,
            all_logits.data(), static_cast<int>(vocab_size), MPI_FLOAT,
            MPI_COMM_WORLD);

        // Compare all ranks against rank 0
        const float *ref_logits = all_logits.data();
        for (int r = 1; r < mpiWorldSize(); ++r)
        {
            const float *cmp_logits = all_logits.data() + r * vocab_size;
            float max_diff = max_abs_diff(ref_logits, cmp_logits, vocab_size);
            double cosine = computeCosineSimilarity(ref_logits, cmp_logits, vocab_size);

            LOG_DEBUG("[Rank " << mpiRank() << "] Comparing rank 0 vs rank " << r
                               << ": max_diff=" << max_diff << ", cosine=" << cosine);

            // After AllGather, ranks should have nearly identical logits
            EXPECT_LT(max_diff, 1e-3f)
                << "Rank 0 vs Rank " << r << " max_diff too large (expected identical after AllGather)";
            EXPECT_GT(cosine, 0.999)
                << "Rank 0 vs Rank " << r << " cosine similarity too low";
        }
    }
};

// =============================================================================
// Standard Parity Tests (from Qwen2ParityTestBase)
// =============================================================================

// Generates: PrefillParity_LayerByLayer, DecodeParity_Incremental, SnapshotInfrastructure
INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_TP_MPI_vs_PyTorch);

// =============================================================================
// TP-Specific Tests
// =============================================================================

/**
 * @brief Verify weight sharding is correct for tensor parallelism
 */
TEST_F(Test__Qwen2_TP_MPI_vs_PyTorch, WeightShardingIsCorrect)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    // Verify weight sharding only if we have the model context
    auto weight_mgr = model_ctx_->weightManager();
    ASSERT_NE(weight_mgr, nullptr);

    // Q weight should be column-sharded
    auto wq = weight_mgr->getWeight("blk.0.attn_q.weight");
    ASSERT_NE(wq, nullptr);

    // Expected shape: [local_n_heads * head_dim, d_model]
    // Qwen2.5-0.5B: n_heads=14, head_dim=64, d_model=896
    int n_heads = 14;
    int head_dim = 64;
    int local_n_heads = n_heads / mpiWorldSize();
    size_t expected_rows = local_n_heads * head_dim;

    EXPECT_EQ(wq->shape()[0], expected_rows)
        << "[Rank " << mpiRank() << "] Q weight row count mismatch";

    mpiBarrier();
    if (isRank0())
    {
        LOG_INFO("[TP Parity] Weight sharding verified");
    }
}

/**
 * @brief Verify all ranks produce identical logits after AllGather
 */
TEST_F(Test__Qwen2_TP_MPI_vs_PyTorch, RankConsistencyVerification)
{
    ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";

    // Run prefill
    int seq_len = static_cast<int>(config_.token_ids.size());
    ASSERT_TRUE(runner_->forward(config_.token_ids.data(), seq_len));

    const float *logits_ptr = runner_->logits();
    ASSERT_NE(logits_ptr, nullptr);

    // Get vocab size from model
    size_t vocab_size = 151936; // Qwen2.5 0.5B vocab size
    const float *last_logits = logits_ptr + (seq_len - 1) * vocab_size;

    // Strict rank consistency check
    std::vector<float> all_logits(mpiWorldSize() * vocab_size);
    MPI_Allgather(
        last_logits, static_cast<int>(vocab_size), MPI_FLOAT,
        all_logits.data(), static_cast<int>(vocab_size), MPI_FLOAT,
        MPI_COMM_WORLD);

    // All ranks must have identical logits (within floating-point tolerance)
    const float *ref = all_logits.data();
    for (int r = 1; r < mpiWorldSize(); ++r)
    {
        const float *cmp = all_logits.data() + r * vocab_size;
        float max_diff = max_abs_diff(ref, cmp, vocab_size);
        double cosine = computeCosineSimilarity(ref, cmp, vocab_size);

        // Very strict: after AllGather, should be essentially identical
        EXPECT_LT(max_diff, 1e-4f)
            << "[Rank " << mpiRank() << "] Rank 0 vs Rank " << r << " have divergent logits";
        EXPECT_GT(cosine, 0.9999)
            << "[Rank " << mpiRank() << "] Rank 0 vs Rank " << r << " cosine too low";
    }

    if (isRank0())
    {
        LOG_INFO("[TP Parity] Rank consistency verified");
    }
    mpiBarrier();
}

// =============================================================================
// Main (MPI wrapper)
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "WARNING: MPI does not provide MPI_THREAD_MULTIPLE support" << std::endl;
    }

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
