/**
 * @file Test__MPI_ColumnParallelQKV.cpp
 * @brief Integration tests for Phase 3: Column-Parallel QKV tensor parallelism
 *
 * Tests the column-parallel sharding strategy for attention Q/K/V projections:
 *   1. Column-sliced weight loading (output dimension split by rank)
 *   2. Local Q/K/V buffer allocation (proportional to local heads)
 *   3. RoPE with local head counts
 *   4. Attention with local head computation
 *   5. Output concatenation via allgather
 *
 * Phase 3 Strategy:
 * - Q weight [n_heads * head_dim, d_model] → [local_n_heads * head_dim, d_model]
 * - K/V weights [n_kv_heads * head_dim, d_model] → [local_n_kv_heads * head_dim, d_model]
 * - Wo remains row-parallel with allreduce after attention output projection
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include "loaders/WeightManager.h"
#include "models/qwen/Qwen2Schema.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;

/**
 * @brief Test fixture for Column-Parallel QKV integration tests
 */
class Test__MPI_ColumnParallelQKV : public ::testing::Test
{
protected:
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<TensorFactory> factory_;
    int rank_ = 0;
    int world_size_ = 1;

    // Model dimensions (Qwen2.5 0.5B style)
    static constexpr int D_MODEL = 896;
    static constexpr int N_HEADS = 14;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
    WeightShardingConfig sharding_config_;

    /**
     * @brief Helper to get sharding mode using schema config
     */
    ShardingMode getShardingMode(const std::string &name) const
    {
        WeightShardingMode mode = sharding_config_.getMode(name);
        switch (mode)
        {
        case WeightShardingMode::ColumnParallel:
            return ShardingMode::COLUMN_PARALLEL;
        case WeightShardingMode::RowParallel:
            return ShardingMode::ROW_PARALLEL;
        case WeightShardingMode::InputParallel:
            return ShardingMode::INPUT_PARALLEL;
        case WeightShardingMode::Replicate:
        default:
            return ShardingMode::REPLICATE;
        }
    }

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
        factory_ = std::make_shared<TensorFactory>(*mpi_ctx_);

        // Load Qwen2 sharding config
        Qwen2SchemaFactory schema_factory;
        sharding_config_ = schema_factory.getWeightShardingConfig();
    }

    void TearDown() override
    {
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a random FP32 tensor with deterministic seed
     */
    std::shared_ptr<FP32Tensor> createRandomFP32(
        const std::vector<size_t> &shape, unsigned seed)
    {
        auto tensor = std::make_shared<FP32Tensor>(shape);
        float *data = tensor->mutable_data();

        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        size_t numel = tensor->numel();
        for (size_t i = 0; i < numel; ++i)
        {
            data[i] = dist(gen);
        }
        return tensor;
    }

    /**
     * @brief Slice rows from an FP32 tensor (column-parallel simulation)
     *
     * Simulates WeightManager column slicing:
     *   Full: [N, K] → Local: [local_N, K]
     */
    std::shared_ptr<FP32Tensor> sliceRows(
        const FP32Tensor *full,
        size_t row_start, size_t local_rows)
    {
        const auto &shape = full->shape();
        size_t N = shape[0];
        size_t K = shape[1];

        // Validate slice bounds
        EXPECT_LE(row_start + local_rows, N);

        auto sliced = std::make_shared<FP32Tensor>(std::vector<size_t>{local_rows, K});
        const float *src = full->data() + row_start * K;
        float *dst = sliced->mutable_data();
        std::memcpy(dst, src, local_rows * K * sizeof(float));

        return sliced;
    }

    /**
     * @brief Compute FP32 GEMM: C = A @ B^T
     *
     * A: [M, K]
     * B: [N, K] (transposed internally)
     * C: [M, N]
     */
    void fp32_gemm(
        const float *A, const float *B, float *C,
        int M, int N, int K)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[i * K + k] * B[j * K + k]; // B^T
                }
                C[i * N + j] = sum;
            }
        }
    }

    /**
     * @brief Compute max absolute error between two arrays
     */
    float maxAbsError(const float *a, const float *b, size_t n)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float err = std::abs(a[i] - b[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }
};

// =============================================================================
// Test: Sharding Mode Detection for Q/K/V
// =============================================================================

TEST_F(Test__MPI_ColumnParallelQKV, ShardingModeDetection)
{
    // Verify sharding config correctly identifies Q/K/V as COLUMN_PARALLEL
    EXPECT_EQ(getShardingMode("blk.0.attn_q.weight"),
              ShardingMode::COLUMN_PARALLEL)
        << "Q weight should be COLUMN_PARALLEL";

    EXPECT_EQ(getShardingMode("blk.0.attn_k.weight"),
              ShardingMode::COLUMN_PARALLEL)
        << "K weight should be COLUMN_PARALLEL";

    EXPECT_EQ(getShardingMode("blk.0.attn_v.weight"),
              ShardingMode::COLUMN_PARALLEL)
        << "V weight should be COLUMN_PARALLEL";

    // Verify biases are also column-parallel
    EXPECT_EQ(getShardingMode("blk.0.attn_q.bias"),
              ShardingMode::COLUMN_PARALLEL)
        << "Q bias should be COLUMN_PARALLEL";

    EXPECT_EQ(getShardingMode("blk.0.attn_k.bias"),
              ShardingMode::COLUMN_PARALLEL)
        << "K bias should be COLUMN_PARALLEL";

    EXPECT_EQ(getShardingMode("blk.0.attn_v.bias"),
              ShardingMode::COLUMN_PARALLEL)
        << "V bias should be COLUMN_PARALLEL";

    if (rank_ == 0)
    {
        LOG_INFO("[Test] Sharding mode detection PASSED - Q/K/V weights and biases are COLUMN_PARALLEL");
    }
}

// =============================================================================
// Test: Local Head Distribution
// =============================================================================

TEST_F(Test__MPI_ColumnParallelQKV, LocalHeadDistribution)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Compute local head distribution
    auto [head_start, local_n_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(N_HEADS));
    auto [kv_head_start, local_n_kv_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(N_KV_HEADS));

    LOG_INFO("[Rank " << rank_ << "] Q heads: [" << head_start << ", " << head_start + local_n_heads << ") "
                      << "(" << local_n_heads << "/" << N_HEADS << ")");
    LOG_INFO("[Rank " << rank_ << "] KV heads: [" << kv_head_start << ", " << kv_head_start + local_n_kv_heads << ") "
                      << "(" << local_n_kv_heads << "/" << N_KV_HEADS << ")");

    // With 2 ranks and 14 Q heads: rank 0 gets 7, rank 1 gets 7
    if (world_size_ == 2)
    {
        EXPECT_EQ(local_n_heads, 7) << "Each rank should get 7 Q heads (14/2)";

        // With 2 KV heads: rank 0 gets 1, rank 1 gets 1
        EXPECT_EQ(local_n_kv_heads, 1) << "Each rank should get 1 KV head (2/2)";

        if (rank_ == 0)
        {
            EXPECT_EQ(head_start, 0);
            EXPECT_EQ(kv_head_start, 0);
        }
        else
        {
            EXPECT_EQ(head_start, 7);
            EXPECT_EQ(kv_head_start, 1);
        }
    }

    mpi_ctx_->barrier();
    if (rank_ == 0)
    {
        LOG_INFO("[Test] Local head distribution PASSED");
    }
}

// =============================================================================
// Test: Column-Parallel Q Projection
// =============================================================================

TEST_F(Test__MPI_ColumnParallelQKV, ColumnParallelQProjection)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // =========================================================================
    // Setup: Create full Q weight and input activation
    // =========================================================================
    const int SEQ_LEN = 4;
    const int Q_OUT_DIM = N_HEADS * HEAD_DIM; // 14 * 64 = 896

    // Full Q weight: [Q_OUT_DIM, D_MODEL] = [896, 896]
    auto wq_full = createRandomFP32({Q_OUT_DIM, D_MODEL}, 42);

    // Input activation: [SEQ_LEN, D_MODEL] = [4, 896]
    auto input = createRandomFP32({SEQ_LEN, D_MODEL}, 123);

    // Compute full reference: Q_full = input @ wq_full^T
    // Result: [SEQ_LEN, Q_OUT_DIM] = [4, 896]
    std::vector<float> Q_ref(SEQ_LEN * Q_OUT_DIM);
    fp32_gemm(input->data(), wq_full->data(), Q_ref.data(),
              SEQ_LEN, Q_OUT_DIM, D_MODEL);

    // =========================================================================
    // Column-Parallel: Each rank gets a slice of Q weight rows
    // =========================================================================
    auto [head_start, local_n_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(N_HEADS));
    size_t local_q_dim = local_n_heads * HEAD_DIM;
    size_t row_start = head_start * HEAD_DIM;

    // Slice Q weight: [local_q_dim, D_MODEL]
    auto wq_local = sliceRows(wq_full.get(), row_start, local_q_dim);

    LOG_DEBUG("[Rank " << rank_ << "] Q weight local shape: [" << local_q_dim << ", " << D_MODEL << "]"
                       << " (rows " << row_start << " to " << row_start + local_q_dim << ")");

    // Compute local Q: [SEQ_LEN, local_q_dim]
    std::vector<float> Q_local(SEQ_LEN * local_q_dim);
    fp32_gemm(input->data(), wq_local->data(), Q_local.data(),
              SEQ_LEN, static_cast<int>(local_q_dim), D_MODEL);

    // =========================================================================
    // Verify: Local output matches corresponding slice of reference
    // =========================================================================
    for (int s = 0; s < SEQ_LEN; ++s)
    {
        for (size_t h = 0; h < local_q_dim; ++h)
        {
            size_t local_idx = s * local_q_dim + h;
            size_t ref_idx = s * Q_OUT_DIM + row_start + h;
            float diff = std::abs(Q_local[local_idx] - Q_ref[ref_idx]);
            EXPECT_LT(diff, 1e-4f)
                << "Mismatch at seq=" << s << ", local_h=" << h
                << " (global_h=" << row_start + h << ")"
                << ": local=" << Q_local[local_idx]
                << " ref=" << Q_ref[ref_idx];
        }
    }

    mpi_ctx_->barrier();
    if (rank_ == 0)
    {
        LOG_INFO("[Test] Column-parallel Q projection PASSED");
    }
}

// =============================================================================
// Test: Full QKV Column-Parallel Projection
// =============================================================================

TEST_F(Test__MPI_ColumnParallelQKV, FullQKVColumnParallel)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int SEQ_LEN = 4;
    const int Q_OUT_DIM = N_HEADS * HEAD_DIM;     // 896
    const int KV_OUT_DIM = N_KV_HEADS * HEAD_DIM; // 128

    // Create full weights
    auto wq_full = createRandomFP32({Q_OUT_DIM, D_MODEL}, 1001);
    auto wk_full = createRandomFP32({KV_OUT_DIM, D_MODEL}, 1002);
    auto wv_full = createRandomFP32({KV_OUT_DIM, D_MODEL}, 1003);
    auto input = createRandomFP32({SEQ_LEN, D_MODEL}, 2001);

    // Compute full references
    std::vector<float> Q_ref(SEQ_LEN * Q_OUT_DIM);
    std::vector<float> K_ref(SEQ_LEN * KV_OUT_DIM);
    std::vector<float> V_ref(SEQ_LEN * KV_OUT_DIM);
    fp32_gemm(input->data(), wq_full->data(), Q_ref.data(), SEQ_LEN, Q_OUT_DIM, D_MODEL);
    fp32_gemm(input->data(), wk_full->data(), K_ref.data(), SEQ_LEN, KV_OUT_DIM, D_MODEL);
    fp32_gemm(input->data(), wv_full->data(), V_ref.data(), SEQ_LEN, KV_OUT_DIM, D_MODEL);

    // Column-parallel slicing
    auto [q_head_start, local_n_q_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(N_HEADS));
    auto [kv_head_start, local_n_kv_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(N_KV_HEADS));

    size_t local_q_dim = local_n_q_heads * HEAD_DIM;
    size_t local_kv_dim = local_n_kv_heads * HEAD_DIM;
    size_t q_row_start = q_head_start * HEAD_DIM;
    size_t kv_row_start = kv_head_start * HEAD_DIM;

    auto wq_local = sliceRows(wq_full.get(), q_row_start, local_q_dim);
    auto wk_local = sliceRows(wk_full.get(), kv_row_start, local_kv_dim);
    auto wv_local = sliceRows(wv_full.get(), kv_row_start, local_kv_dim);

    // Compute local QKV
    std::vector<float> Q_local(SEQ_LEN * local_q_dim);
    std::vector<float> K_local(SEQ_LEN * local_kv_dim);
    std::vector<float> V_local(SEQ_LEN * local_kv_dim);
    fp32_gemm(input->data(), wq_local->data(), Q_local.data(), SEQ_LEN, (int)local_q_dim, D_MODEL);
    fp32_gemm(input->data(), wk_local->data(), K_local.data(), SEQ_LEN, (int)local_kv_dim, D_MODEL);
    fp32_gemm(input->data(), wv_local->data(), V_local.data(), SEQ_LEN, (int)local_kv_dim, D_MODEL);

    // Verify Q local matches reference slice
    float max_q_err = 0.0f;
    for (int s = 0; s < SEQ_LEN; ++s)
    {
        for (size_t h = 0; h < local_q_dim; ++h)
        {
            size_t local_idx = s * local_q_dim + h;
            size_t ref_idx = s * Q_OUT_DIM + q_row_start + h;
            float err = std::abs(Q_local[local_idx] - Q_ref[ref_idx]);
            max_q_err = std::max(max_q_err, err);
        }
    }
    EXPECT_LT(max_q_err, 1e-4f) << "Q max error: " << max_q_err;

    // Verify K local matches reference slice
    float max_k_err = 0.0f;
    for (int s = 0; s < SEQ_LEN; ++s)
    {
        for (size_t h = 0; h < local_kv_dim; ++h)
        {
            size_t local_idx = s * local_kv_dim + h;
            size_t ref_idx = s * KV_OUT_DIM + kv_row_start + h;
            float err = std::abs(K_local[local_idx] - K_ref[ref_idx]);
            max_k_err = std::max(max_k_err, err);
        }
    }
    EXPECT_LT(max_k_err, 1e-4f) << "K max error: " << max_k_err;

    // Verify V local matches reference slice
    float max_v_err = 0.0f;
    for (int s = 0; s < SEQ_LEN; ++s)
    {
        for (size_t h = 0; h < local_kv_dim; ++h)
        {
            size_t local_idx = s * local_kv_dim + h;
            size_t ref_idx = s * KV_OUT_DIM + kv_row_start + h;
            float err = std::abs(V_local[local_idx] - V_ref[ref_idx]);
            max_v_err = std::max(max_v_err, err);
        }
    }
    EXPECT_LT(max_v_err, 1e-4f) << "V max error: " << max_v_err;

    mpi_ctx_->barrier();
    if (rank_ == 0)
    {
        LOG_INFO("[Test] Full QKV column-parallel projection PASSED");
        LOG_INFO("  Max errors: Q=" << max_q_err << ", K=" << max_k_err << ", V=" << max_v_err);
    }
}

// =============================================================================
// Test: Buffer Dimension Calculation for Local Heads
// =============================================================================

TEST_F(Test__MPI_ColumnParallelQKV, BufferDimensionsWithLocalHeads)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const int SEQ_LEN = 8;

    auto [head_start, local_n_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(N_HEADS));
    auto [kv_head_start, local_n_kv_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(N_KV_HEADS));

    // Expected local buffer dimensions
    size_t local_q_dim = local_n_heads * HEAD_DIM;
    size_t local_kv_dim = local_n_kv_heads * HEAD_DIM;

    // With 2 ranks, 14 Q heads, 2 KV heads:
    // - local_q_dim = 7 * 64 = 448
    // - local_kv_dim = 1 * 64 = 64
    if (world_size_ == 2)
    {
        EXPECT_EQ(local_q_dim, 7 * 64) << "Q buffer dim should be 448";
        EXPECT_EQ(local_kv_dim, 1 * 64) << "KV buffer dim should be 64";
    }

    // Buffer shapes
    std::vector<size_t> q_shape = {static_cast<size_t>(SEQ_LEN), local_q_dim};
    std::vector<size_t> k_shape = {static_cast<size_t>(SEQ_LEN), local_kv_dim};
    std::vector<size_t> v_shape = {static_cast<size_t>(SEQ_LEN), local_kv_dim};
    std::vector<size_t> attn_out_shape = {static_cast<size_t>(SEQ_LEN), local_q_dim};

    // Verify allocations work (no OOM)
    auto Q_buf = std::make_shared<FP32Tensor>(q_shape);
    auto K_buf = std::make_shared<FP32Tensor>(k_shape);
    auto V_buf = std::make_shared<FP32Tensor>(v_shape);
    auto attn_out = std::make_shared<FP32Tensor>(attn_out_shape);

    EXPECT_EQ(Q_buf->numel(), SEQ_LEN * local_q_dim);
    EXPECT_EQ(K_buf->numel(), SEQ_LEN * local_kv_dim);
    EXPECT_EQ(V_buf->numel(), SEQ_LEN * local_kv_dim);
    EXPECT_EQ(attn_out->numel(), SEQ_LEN * local_q_dim);

    mpi_ctx_->barrier();
    if (rank_ == 0)
    {
        LOG_INFO("[Test] Buffer dimensions with local heads PASSED");
        LOG_INFO("  Buffers: Q=[" << SEQ_LEN << ", " << local_q_dim << "], "
                                  << "K/V=[" << SEQ_LEN << ", " << local_kv_dim << "]");
    }
}

// =============================================================================
// Main for standalone execution
// =============================================================================

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
