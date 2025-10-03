/**
 * @file test_mpi_softmax_parity.cpp
 * @brief Validates parity between distributed softmax (column partition) and replicated row-major reference.
 */
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include "kernels/common/softmax_core.h"
#include "test_mpi_utils.h" // for LLAMINAR_DEFINE_GTEST_MPI_MAIN and MPIEnvironment

using namespace llaminar::kernels;

namespace
{
    void reference_softmax(std::vector<float> &data, int rows, int cols, bool causal, float scale)
    {
        for (int r = 0; r < rows; ++r)
        {
            float *row = data.data() + size_t(r) * cols;
            float m = -INFINITY;
            for (int c = 0; c < cols; ++c)
            {
                bool masked = causal && c > r;
                if (masked)
                    continue;
                float v = row[c];
                if (scale != 1.f)
                    v *= scale;
                m = std::max(m, v);
            }
            if (!std::isfinite(m))
                m = 0.f;
            double s = 0.0;
            for (int c = 0; c < cols; ++c)
            {
                bool masked = causal && c > r;
                if (masked)
                {
                    row[c] = 0.f;
                    continue;
                }
                float v = row[c];
                if (scale != 1.f)
                    v *= scale;
                float e = std::exp(v - m);
                row[c] = e;
                s += e;
            }
            if (s <= 0.0)
                s = 1.0;
            float inv = float(1.0 / s);
            for (int c = 0; c < cols; ++c)
                row[c] *= inv;
        }
    }

    double rel_l2(const std::vector<float> &a, const std::vector<float> &b)
    {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double d = double(a[i]) - double(b[i]);
            num += d * d;
            den += double(a[i]) * double(a[i]);
        }
        if (den == 0.0)
            den = 1.0;
        return std::sqrt(num / den);
    }
}

class MPISoftmaxParityTest : public ::testing::TestWithParam<std::tuple<int, int, bool, float>>
{
protected:
    void SetUp() override
    {
        // Ensure MPI is initialized (tests may be launched without the custom MPI main)
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int provided = 0;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
    }
    int rank{0};
    int world{1};
};

TEST_P(MPISoftmaxParityTest, ColumnPartitionParity)
{
    auto [rows, cols, causal, scale] = GetParam();
    if (world != 2)
    {
        GTEST_SKIP() << "Requires exactly 2 ranks";
    }
    std::mt19937 rng(42 + rank);
    std::uniform_real_distribution<float> dist(-4.f, 4.f);

    // Full replicated matrix built on rank 0 for reference
    std::vector<float> full(size_t(rows) * cols);
    if (rank == 0)
    {
        for (auto &v : full)
            v = dist(rng);
    }
    // Broadcast initial data to all ranks so each can carve out its column slice
    MPI_Bcast(full.data(), (int)full.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Split columns: first half rank 0, second half rank 1 (handle odd by giving extra to rank 0)
    int cols0 = cols / 2 + (cols % 2);
    int cols1 = cols - cols0;
    int local_cols = (rank == 0) ? cols0 : cols1;
    int local_col_offset = (rank == 0) ? 0 : cols0;

    std::vector<float> local(size_t(rows) * local_cols);
    for (int r = 0; r < rows; ++r)
    {
        const float *src_row = full.data() + size_t(r) * cols + local_col_offset;
        std::copy(src_row, src_row + local_cols, local.data() + size_t(r) * local_cols);
    }

    // Run distributed softmax on local slice
    SoftmaxRowArgs local_args;
    local_args.scores = local.data();
    local_args.rows = rows;
    local_args.cols = local_cols;
    local_args.causal = causal;
    local_args.scale = scale;
    DistributedSoftmaxCtx ctx;
    ctx.comm = MPI_COMM_WORLD;
    ctx.world_size = world;
    ctx.rank = rank;
    ctx.use_barrier = false;
    softmax_distributed(local_args, /*global_rows=*/rows, /*row_offset=*/0, /*global_cols=*/cols, local_col_offset, ctx);

    // Ensure all ranks have completed normalization before gather
    MPI_Barrier(MPI_COMM_WORLD);

    // Generalized reconstruction for arbitrary world size with contiguous column blocks per rank.
    // Compute column partition plan (root) then bcast to all.
    std::vector<int> part_offsets(world, 0), part_widths(world, 0);
    if (rank == 0)
    {
        int remaining = cols;
        int offset = 0;
        for (int rnk = 0; rnk < world; ++rnk)
        {
            int share = cols / world;
            if (rnk < cols % world)
                ++share; // distribute remainder
            part_offsets[rnk] = offset;
            part_widths[rnk] = share;
            offset += share;
        }
    }
    MPI_Bcast(part_offsets.data(), world, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(part_widths.data(), world, MPI_INT, 0, MPI_COMM_WORLD);

    // Validate local partition matches expectation (debug assert style)
    assert(part_offsets[rank] == local_col_offset);
    assert(part_widths[rank] == local_cols);

    std::vector<float> gathered;
    if (rank == 0)
        gathered.resize(size_t(rows) * cols);
    // Root copies its slice directly
    if (rank == 0)
    {
        for (int r = 0; r < rows; ++r)
        {
            float *dst_row = gathered.data() + size_t(r) * cols + part_offsets[0];
            const float *src_row = local.data() + size_t(r) * local_cols;
            std::memcpy(dst_row, src_row, sizeof(float) * local_cols);
        }
    }
    // Non-root ranks send their row slices; root receives in rank order.
    for (int rnk = 1; rnk < world; ++rnk)
    {
        if (rank == rnk)
        {
            for (int r = 0; r < rows; ++r)
            {
                const float *src_row = local.data() + size_t(r) * local_cols;
                MPI_Send(src_row, local_cols, MPI_FLOAT, 0, r, MPI_COMM_WORLD);
            }
        }
        else if (rank == 0)
        {
            for (int r = 0; r < rows; ++r)
            {
                MPI_Status st{};
                float *dst_row = gathered.data() + size_t(r) * cols + part_offsets[rnk];
                MPI_Recv(dst_row, part_widths[rnk], MPI_FLOAT, rnk, r, MPI_COMM_WORLD, &st);
            }
        }
    }

    // Reference on rank 0
    if (rank == 0)
    {
        auto reference = full; // original logits
        reference_softmax(reference, rows, cols, causal, scale);
        double rl2 = rel_l2(reference, gathered);
        // Debug: print first row probabilities for inspection
        if (rows > 0)
        {
            fprintf(stderr, "[MPISoftmaxParityTest] gathered row0: ");
            for (int c = 0; c < cols; ++c)
                fprintf(stderr, "%0.6f ", gathered[c]);
            fprintf(stderr, "\n");
        }
        // Validate per-row sums ~1 and print diagnostics if off
        for (int r = 0; r < rows; ++r)
        {
            double s = 0.0;
            for (int c = 0; c < cols; ++c)
                s += gathered[size_t(r) * cols + c];
            if (std::abs(s - 1.0) > 1e-6)
            {
                fprintf(stderr, "[MPISoftmaxParityTest] row %d sum=%f rl2=%e causal=%d scale=%f\n", r, s, rl2, (int)causal, scale);
            }
            ASSERT_NEAR(s, 1.0, 1e-6) << "Row sum deviates r=" << r;
        }
        ASSERT_LT(rl2, 1e-7) << "Distributed softmax mismatch rl2=" << rl2;
    }
}

INSTANTIATE_TEST_SUITE_P(MPIShapes, MPISoftmaxParityTest,
                         ::testing::Values(
                             std::make_tuple(4, 8, false, 1.0f),
                             std::make_tuple(8, 16, true, 1.0f),
                             std::make_tuple(6, 13, true, 0.8f)));

// Provide a custom main that initializes and finalizes MPI consistently with other MPI tests.
LLAMINAR_DEFINE_GTEST_MPI_MAIN();
