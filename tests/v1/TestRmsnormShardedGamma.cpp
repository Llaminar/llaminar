#include <mpi.h>
#include <gtest/gtest.h>
#include <random>
#include <cstring>
#include "operators/MPIRMSNormOperator.h"
#include "tensors/TensorFactory.h"
#include "tensors/ShardedSimpleTensor.h"
#include "tensors/ShardSpec.h"
#include "utils/DebugEnv.h"

using namespace llaminar;

static ShardSpec make_hidden_spec(int global_hidden, int world, int rank)
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
    return spec;
}

// Build sharded tensor for gamma (weight) along hidden dimension
static std::shared_ptr<ShardedSimpleTensor> make_gamma_shard(int /*global_hidden*/, const ShardSpec &spec)
{
    auto t = std::make_shared<ShardedSimpleTensor>(std::vector<int>{static_cast<int>(spec.local_dim)}, spec);
    return t;
}

TEST(RMSNormShardStatsTest, ShardedGammaParity)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    int world = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
        GTEST_SKIP() << "Requires >=2 MPI ranks";

    const int seq_len = 10;
    const int hidden = 37; // irregular to stress offsets
    const float eps = 1e-6f;

    // Full input (identical across ranks for reconstruction reference)
    std::vector<float> full_input(seq_len * hidden);
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    if (rank == 0)
    {
        for (float &v : full_input)
            v = dist(rng);
    }
    MPI_Bcast(full_input.data(), (int)full_input.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Build sharded input activation
    ShardSpec act_spec = make_hidden_spec(hidden, world, rank);
    auto act_shard = std::make_shared<ShardedSimpleTensor>(std::vector<int>{seq_len, static_cast<int>(act_spec.local_dim)}, act_spec);
    for (int r = 0; r < seq_len; ++r)
    {
        const float *src = full_input.data() + r * hidden + act_spec.local_offset;
        std::memcpy(act_shard->data() + r * act_spec.local_dim, src, sizeof(float) * act_spec.local_dim);
    }

    // Build sharded gamma (random values) + gather full gamma on all ranks for reference
    std::vector<float> full_gamma(hidden);
    if (rank == 0)
    {
        for (float &g : full_gamma)
            g = 0.5f + 0.5f * dist(rng);
    }
    MPI_Bcast(full_gamma.data(), hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);

    ShardSpec gamma_spec = make_hidden_spec(hidden, world, rank);
    auto gamma_shard = make_gamma_shard(hidden, gamma_spec);
    std::memcpy(gamma_shard->data(), full_gamma.data() + gamma_spec.local_offset, sizeof(float) * gamma_spec.local_dim);

    // Output shard
    auto out_shard = std::make_shared<ShardedSimpleTensor>(std::vector<int>{seq_len, static_cast<int>(act_spec.local_dim)}, act_spec);

    // For test harness we create a temporary *replicated* gamma tensor so the kernel receives the full weight vector.
    // When native sharded-gamma support is added, replace with gamma_shard directly.

    auto gamma_repl = TensorFactory::create_simple({hidden});
    std::memcpy(gamma_repl->data(), full_gamma.data(), sizeof(float) * hidden);

    MPIRMSNormOperator kernel(MPIRMSNormOperator::DistributionStrategy::SEQUENCE_WISE);
    kernel.setEpsilon(eps);

    std::vector<std::shared_ptr<TensorBase>> inputs = {act_shard, gamma_repl};
    std::vector<std::shared_ptr<TensorBase>> outputs = {out_shard};
    ASSERT_TRUE(kernel.execute(inputs, outputs));

    // NOTE: Kernel already applied replicated gamma slice to this feature shard.
    // Do NOT rescale by local gamma shard (would double-apply gamma). Keeping gamma_shard
    // construction for forward compatibility when kernel accepts sharded gamma directly.

    // Reconstruct full output via Gatherv per row
    std::vector<int> counts(world), displs(world);
    for (int r = 0; r < world; ++r)
    {
        int base = hidden / world;
        int rem = hidden % world;
        counts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < world; ++r)
        displs[r] = displs[r - 1] + counts[r - 1];

    std::vector<float> recon_row(hidden), full_out(seq_len * hidden, 0.f);
    for (int r = 0; r < seq_len; ++r)
    {
        const float *local_row = out_shard->data() + r * act_spec.local_dim;
        MPI_Gatherv(local_row, act_spec.local_dim, MPI_FLOAT, recon_row.data(), counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (rank == 0)
            std::memcpy(full_out.data() + r * hidden, recon_row.data(), sizeof(float) * hidden);
    }

    if (rank == 0)
    {
        // Reference RMSNorm + gamma scaling
        std::vector<float> ref(seq_len * hidden);
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
            {
                dst[c] = (float)(src[c] * inv * full_gamma[c]);
            }
        }
        double max_abs = 0.0, rel_num = 0.0, rel_den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            double d = (double)ref[i] - (double)full_out[i];
            max_abs = std::max(max_abs, std::abs(d));
            rel_num += d * d;
            rel_den += (double)ref[i] * ref[i];
        }
        double rel_l2 = rel_den > 0 ? std::sqrt(rel_num / rel_den) : 0.0;
        EXPECT_LT(max_abs, 1e-5) << "max_abs=" << max_abs;
        EXPECT_LT(rel_l2, 1e-6) << "rel_l2=" << rel_l2;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
    {
        MPI_Finalize();
    }
}
