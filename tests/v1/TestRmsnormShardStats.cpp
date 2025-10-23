#include <mpi.h>
#include <gtest/gtest.h>
#include "operators/MPIRMSNormOperator.h"
#include "tensors/TensorFactory.h"
#include "tensors/ShardedSimpleTensor.h"
#include "tensors/ShardSpec.h"
#include "utils/DebugEnv.h"
#include <random>

using namespace llaminar;

// Helper to create a sharded tensor along Hidden axis with balanced split
static std::shared_ptr<ShardedSimpleTensor> make_hidden_shard(int global_rows, int global_hidden, int world, int rank)
{
    int base = global_hidden / world;
    int rem = global_hidden % world;
    int local_dim = base + (rank < rem ? 1 : 0);
    int offset = rank * base + std::min(rank, rem);
    ShardSpec spec;
    spec.type = ShardSpec::Type::Sharded;
    spec.axis = ShardSpec::Axis::Hidden;
    spec.world = world;
    spec.rank = rank;
    spec.global_dim = global_hidden;
    spec.local_offset = offset;
    spec.local_dim = local_dim;
    return std::make_shared<ShardedSimpleTensor>(std::vector<int>{global_rows, local_dim}, spec);
}

TEST(RMSNormShardStatsTest, GlobalInverseScaleParity)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    bool need_finalize = false;
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
        need_finalize = true;
    }
    int world = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
        GTEST_SKIP() << "Requires >=2 MPI ranks";

    const int seq_len = 8; // small rows
    const int hidden = 33; // intentionally not divisible by typical world sizes
    const float eps = 1e-6f;

    // Create global (conceptual) full input on rank0 for reference
    std::vector<float> full_input(seq_len * hidden, 0.0f);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    if (rank == 0)
    {
        for (float &v : full_input)
            v = dist(rng);
    }
    // Broadcast so every rank has the reference for recomposition checks
    MPI_Bcast(full_input.data(), (int)full_input.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Create gamma (weight) all ones for simplicity (unit scaling effect only from RMS)
    std::vector<float> gamma(hidden, 1.0f);

    // Build sharded input tensor (feature/hidden sharded) local slice only
    auto shard_tensor = make_hidden_shard(seq_len, hidden, world, rank);
    // Populate local slice from full_input
    const ShardSpec &spec = shard_tensor->shard_spec();
    for (int r = 0; r < seq_len; ++r)
    {
        const float *src_row_full = full_input.data() + r * hidden + spec.local_offset;
        float *dst_row = shard_tensor->data() + r * spec.local_dim;
        std::memcpy(dst_row, src_row_full, sizeof(float) * spec.local_dim);
    }

    // Replicated weight tensor (simplifies parity expectation)
    auto weight_repl = TensorFactory::create_simple({hidden});
    std::memcpy(weight_repl->data(), gamma.data(), sizeof(float) * hidden);

    // Output tensor (same shard layout)
    auto out_shard = make_hidden_shard(seq_len, hidden, world, rank);

    // Use sequence-wise strategy; each rank will gather feature slices for parity reconstruction
    MPIRMSNormOperator kernel(MPIRMSNormOperator::DistributionStrategy::SEQUENCE_WISE);
    kernel.setEpsilon(eps);

    std::vector<std::shared_ptr<TensorBase>> inputs = {shard_tensor, weight_repl};
    std::vector<std::shared_ptr<TensorBase>> outputs = {out_shard};

    ASSERT_TRUE(kernel.execute(inputs, outputs));

    // Reconstruct full output on rank0 by Allgather the feature slices for each row
    // Gather each row separately for simplicity
    std::vector<int> counts(world), displs(world);
    for (int r = 0; r < world; ++r)
    {
        int base = hidden / world;
        int rem = hidden % world;
        int ldim = base + (r < rem ? 1 : 0);
        counts[r] = ldim;
    }
    displs[0] = 0;
    for (int r = 1; r < world; ++r)
        displs[r] = displs[r - 1] + counts[r - 1];

    std::vector<float> recon_row(hidden);
    std::vector<float> full_output(seq_len * hidden, 0.0f);
    for (int r = 0; r < seq_len; ++r)
    {
        const float *row_slice = out_shard->data() + r * spec.local_dim;
        MPI_Gatherv(row_slice, spec.local_dim, MPI_FLOAT,
                    recon_row.data(), counts.data(), displs.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::memcpy(full_output.data() + r * hidden, recon_row.data(), sizeof(float) * hidden);
        }
    }

    if (rank == 0)
    {
        // Compute reference RMSNorm (same gamma=1) centrally
        std::vector<float> ref(full_input.size());
        for (int r = 0; r < seq_len; ++r)
        {
            const float *src = full_input.data() + r * hidden;
            float *dst = ref.data() + r * hidden;
            double sumsq = 0.0;
            for (int c = 0; c < hidden; ++c)
            {
                double v = src[c];
                sumsq += v * v;
            }
            double inv = 1.0 / std::sqrt(sumsq / hidden + eps);
            for (int c = 0; c < hidden; ++c)
                dst[c] = (float)(src[c] * inv);
        }
        // Compare
        double max_abs = 0.0, rel_l2_num = 0.0, rel_l2_den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            double diff = (double)ref[i] - (double)full_output[i];
            max_abs = std::max(max_abs, std::abs(diff));
            rel_l2_num += diff * diff;
            rel_l2_den += (double)ref[i] * ref[i];
        }
        double rel_l2 = rel_l2_den > 0 ? std::sqrt(rel_l2_num / rel_l2_den) : 0.0;
        EXPECT_LT(max_abs, 1e-5) << "max_abs=" << max_abs;
        EXPECT_LT(rel_l2, 1e-6) << "rel_l2=" << rel_l2;
    }

    // Ensure all ranks reach end before finalize to prevent abnormal termination
    MPI_Barrier(MPI_COMM_WORLD);
    if (need_finalize)
    {
        MPI_Finalize();
    }
}
