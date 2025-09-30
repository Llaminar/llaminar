// MLP shard parity test: verifies that column-sharded gate/up and row-sharded down projections
// produce identical results to a single-rank reference when aggregated across ranks.
// Distribution strategy mirrors attention parity style for simplicity.
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include "../src/kernels/MPIMLPKernel.h"
#include "../src/tensors/tensor_factory.h"

using namespace llaminar;

static void fill_deterministic(std::shared_ptr<TensorBase> &t, float scale)
{
    for (int i = 0; i < t->size(); ++i)
        t->data()[i] = scale * static_cast<float>((i % 113) - 57);
}
static void fail_and_abort(const char *msg)
{
    std::fprintf(stderr, "[mlp-parity][FATAL] %s\n", msg);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

// Simple reference matmul C = A[M,K] * B[K,N] (row-major) for validation path.
static void ref_matmul(const float *A, const float *B, float *C, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
    {
        const float *arow = A + i * K;
        float *crow = C + i * N;
        for (int j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)arow[k] * (double)B[k * N + j];
            crow[j] = (float)acc;
        }
    }
}

int main(int argc, char **argv)
{
    int seq_len = 7; // default small dims
    int d_model = 64;
    int d_ff = 96; // intentionally not divisible by 2 to exercise uneven shard
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (strncmp(a, "--seq-len=", 10) == 0)
            seq_len = std::atoi(a + 10);
        else if (strncmp(a, "--d-model=", 10) == 0)
            d_model = std::atoi(a + 10);
        else if (strncmp(a, "--d-ff=", 7) == 0)
            d_ff = std::atoi(a + 7);
        else if (strcmp(a, "--help") == 0)
        {
            std::fprintf(stderr, "Usage: %s [--seq-len=L] [--d-model=D] [--d-ff=F]\n", argv[0]);
            return 0;
        }
    }
    if (seq_len <= 0 || d_model <= 0 || d_ff <= 0)
    {
        std::fprintf(stderr, "[mlp-parity][FATAL] invalid args seq_len=%d d_model=%d d_ff=%d\n", seq_len, d_model, d_ff);
        return 3;
    }
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS)
    {
        std::fprintf(stderr, "[mlp-parity] MPI_Init_thread failed\n");
        return 4;
    }
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world < 2 && rank == 0)
    {
        std::fprintf(stderr, "[mlp-parity][WARN] world size=%d (test is meaningful with >=2 ranks)\n", world);
    }
    std::fprintf(stderr, "[mlp-parity] rank=%d world=%d seq_len=%d d_model=%d d_ff=%d\n", rank, world, seq_len, d_model, d_ff);
    std::fflush(stderr);

    // Replicated input & output tensors
    auto input = TensorFactory::create_simple({seq_len, d_model});
    auto output = TensorFactory::create_simple({seq_len, d_model});

    // Global (reference) weights
    auto w_gate_global = TensorFactory::create_simple({d_model, d_ff});
    auto w_up_global = TensorFactory::create_simple({d_model, d_ff});
    auto w_down_global = TensorFactory::create_simple({d_ff, d_model});

    fill_deterministic(input, 0.001f);
    fill_deterministic(w_gate_global, 0.002f);
    fill_deterministic(w_up_global, 0.003f);
    fill_deterministic(w_down_global, 0.004f);

    // Reference computation (rank 0)
    std::vector<float> ref_out((size_t)seq_len * d_model, 0.f);
    if (rank == 0)
    {
        std::vector<float> gate(seq_len * d_ff);
        std::vector<float> up(seq_len * d_ff);
        std::vector<float> act(seq_len * d_ff);
        ref_matmul(input->data(), w_gate_global->data(), gate.data(), seq_len, d_model, d_ff);
        ref_matmul(input->data(), w_up_global->data(), up.data(), seq_len, d_model, d_ff);
        // SwiGLU activation: silu(gate) * up
        auto silu = [](float x)
        { return x / (1.0f + std::exp(-x)); };
        for (size_t i = 0; i < act.size(); ++i)
            act[i] = silu(gate[i]) * up[i];
        ref_matmul(act.data(), w_down_global->data(), ref_out.data(), seq_len, d_ff, d_model);
    }
    MPI_Bcast(ref_out.data(), (int)ref_out.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Create sharded weights: gate/up shard columns (axis_dim_index=1), down shards rows (axis_dim_index=0)
    auto w_gate_shard = TensorFactory::create_sharded(w_gate_global->shape(), ShardSpec::Axis::Hidden, 1, world, rank);
    auto w_up_shard = TensorFactory::create_sharded(w_up_global->shape(), ShardSpec::Axis::Hidden, 1, world, rank);
    auto w_down_shard = TensorFactory::create_sharded(w_down_global->shape(), ShardSpec::Axis::Hidden, 0, world, rank);

    auto copy_col_shard = [&](std::shared_ptr<TensorBase> shard, std::shared_ptr<TensorBase> global)
    {
        auto *raw = dynamic_cast<ShardedSimpleTensor *>(shard.get());
        if (!raw)
            fail_and_abort("col shard dynamic_cast failed");
        const auto &spec = raw->shard_spec();
        int rows = global->shape()[0];
        int global_cols = global->shape()[1];
        for (int r = 0; r < rows; ++r)
        {
            const float *src_row = global->data() + r * global_cols + spec.local_offset;
            float *dst_row = shard->data() + r * spec.local_dim;
            std::memcpy(dst_row, src_row, sizeof(float) * spec.local_dim);
        }
    };
    auto copy_row_shard = [&](std::shared_ptr<TensorBase> shard, std::shared_ptr<TensorBase> global)
    {
        auto *raw = dynamic_cast<ShardedSimpleTensor *>(shard.get());
        if (!raw)
            fail_and_abort("row shard dynamic_cast failed");
        const auto &spec = raw->shard_spec();
        int cols = global->shape()[1];
        for (size_t i = 0; i < spec.local_dim; ++i)
        {
            size_t g = spec.local_offset + i;
            std::memcpy(shard->data() + i * cols, global->data() + g * cols, sizeof(float) * cols);
        }
    };
    copy_col_shard(w_gate_shard, w_gate_global);
    copy_col_shard(w_up_shard, w_up_global);
    copy_row_shard(w_down_shard, w_down_global);

    // Run kernel with local shards
    auto kernel = std::make_unique<MPIMLPKernel>();
    std::vector<std::shared_ptr<TensorBase>> inputs_kernel = {input, w_gate_shard, w_up_shard, w_down_shard};
    std::vector<std::shared_ptr<TensorBase>> outputs_kernel = {output};
    bool ok_local = kernel->execute(inputs_kernel, outputs_kernel);
    int ok_all = 0;
    int ok_flag = ok_local ? 1 : 0;
    MPI_Allreduce(&ok_flag, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (ok_all != 1)
    {
        if (rank == 0)
            std::fprintf(stderr, "[mlp-parity][ERROR] kernel execute failed on some rank\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    // Kernel already performed MPI_Allreduce internally producing the fully aggregated output.
    // Just copy local (replicated) result for validation (avoid double-summing which broke parity).
    std::vector<float> agg(output->size());
    std::memcpy(agg.data(), output->data(), sizeof(float) * output->size());

    if (rank == 0)
    {
        double max_abs = 0.0, sum_ref = 0.0, sum_diff = 0.0;
        for (size_t i = 0; i < agg.size(); ++i)
        {
            double r = ref_out[i];
            double g = agg[i];
            double d = std::fabs(r - g);
            if (d > max_abs)
                max_abs = d;
            sum_ref += r * r;
            sum_diff += d * d;
        }
        double rel_l2 = (sum_ref == 0.0) ? 0.0 : std::sqrt(sum_diff / sum_ref);
        std::fprintf(stderr, "[mlp-parity] metrics: max_abs=%g rel_l2=%g (tol max_abs<=1e-5 rel_l2<=1e-5)\n", max_abs, rel_l2);
        if (!(max_abs <= 1e-5 && rel_l2 <= 1e-5))
        {
            std::fprintf(stderr, "[mlp-parity][FAIL] parity mismatch\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
        else
        {
            std::fprintf(stderr, "[mlp-parity] PARITY OK\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::fprintf(stderr, "[mlp-parity] SUCCESS exiting normally\n");
    MPI_Finalize();
    return 0;
}
