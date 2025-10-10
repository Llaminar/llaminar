// Standalone MPI attention kernel smoke test (no GTest) to avoid pre-main hang.
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>
#include <chrono>
#include "../src/kernels/MPIAttentionKernel.h"
#include "../src/tensors/tensor_factory.h"

using namespace llaminar;

static void fill_seq(float *ptr, int n, float scale)
{
    for (int i = 0; i < n; ++i)
        ptr[i] = scale * (float)((i % 101) - 50);
}
static void fatal(const char *msg, int code = 2)
{
    std::fprintf(stderr, "[mpi-attn-test][FATAL] %s\n", msg);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, code);
}

namespace
{
    struct StaticProbe
    {
        StaticProbe()
        {
            std::fprintf(stderr, "[mpi-attn-test][probe] static init reached before main()\n");
            std::fflush(stderr);
        }
    };
    static StaticProbe _static_probe;
}

int main(int argc, char **argv)
{
    std::fprintf(stderr, "[mpi-attn-test] <enter main> argc=%d\n", argc);
    std::fflush(stderr);
    // Minimal sleep to allow output ordering in case of buffered IO
    fflush(nullptr);
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS)
    {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 3;
    }
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 2 && rank == 0)
    {
        std::fprintf(stderr, "[mpi-attn-test] expected world=2 got %d (continuing)\n", world);
    }
    int n_head = 8, head_dim = 64, seq_len = 4;
    int n_head_kv = n_head; // fixed dims for now
    const int d_model = n_head * head_dim;
    if (rank == 0)
        std::fprintf(stderr, "[mpi-attn-test] start world=%d heads=%d head_dim=%d seq=%d\n", world, n_head, head_dim, seq_len);

    // Allocate global weights/tensors (replicated input path)
    auto input = TensorFactory::create_simple({seq_len, d_model});
    auto wq_g = TensorFactory::create_simple({d_model, n_head * head_dim});
    auto wk_g = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    auto wv_g = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    auto wo_g = TensorFactory::create_simple({n_head * head_dim, d_model});
    auto bq_g = TensorFactory::create_simple({n_head * head_dim});
    auto bk_g = TensorFactory::create_simple({n_head_kv * head_dim});
    auto bv_g = TensorFactory::create_simple({n_head_kv * head_dim});
    auto k_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});
    auto v_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});
    auto out_partial = TensorFactory::create_simple({seq_len, d_model});

    fill_seq(input->data(), input->size(), 0.001f);
    fill_seq(wq_g->data(), wq_g->size(), 0.002f);
    fill_seq(wk_g->data(), wk_g->size(), 0.002f);
    fill_seq(wv_g->data(), wv_g->size(), 0.002f);
    fill_seq(wo_g->data(), wo_g->size(), 0.002f);
    fill_seq(bq_g->data(), bq_g->size(), 0.0001f);
    fill_seq(bk_g->data(), bk_g->size(), 0.0001f);
    fill_seq(bv_g->data(), bv_g->size(), 0.0001f);
    std::memset(k_cache->data(), 0, sizeof(float) * k_cache->size());
    std::memset(v_cache->data(), 0, sizeof(float) * v_cache->size());

    // Execute kernel (replicated weight path triggers internal slicing)
    MPIAttentionKernel kernel(n_head, n_head_kv, head_dim);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq_g, wk_g, wv_g, wo_g, bq_g, bk_g, bv_g, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {out_partial};
    bool ok = kernel.execute(inputs, outputs);
    int all_ok = 0;
    int local_ok = ok ? 1 : 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (all_ok != 1)
        fatal("kernel execute failed on some rank", 4);

    // Reconstruct full output by summing partials (row-sharded Wo -> additive over heads)
    std::vector<float> aggregated(out_partial->size(), 0.f);
    MPI_Allreduce(out_partial->data(), aggregated.data(), (int)aggregated.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Rank 0 basic sanity: non-zero and bounded
    if (rank == 0)
    {
        bool nonzero = false;
        bool sane = true;
        for (size_t i = 0; i < aggregated.size(); ++i)
        {
            float v = aggregated[i];
            if (fabs(v) > 1e-6f)
                nonzero = true;
            if (!std::isfinite(v) || fabs(v) > 1e3f)
            {
                sane = false;
                break;
            }
        }
        if (!nonzero)
            fatal("aggregated output all zeros", 5);
        if (!sane)
            fatal("aggregated output contains NaN/Inf or large values", 6);
        std::fprintf(stderr, "[mpi-attn-test] SUCCESS partials aggregated; sample=%g %g %g %g\n", aggregated[0], aggregated[1], aggregated[2], aggregated[3]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}