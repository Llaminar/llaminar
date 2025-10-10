// Minimal, single implementation (cleanup after duplication removal)
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "../src/kernels/MPIAttentionKernel.h"
#include "../src/tensors/tensor_factory.h"
using namespace llaminar;
static void fill_deterministic(std::shared_ptr<TensorBase> &t, float scale)
{
    for (int i = 0; i < t->size(); ++i)
        t->data()[i] = scale * static_cast<float>((i % 97) - 48);
}
static void fail_and_abort(const char *msg)
{
    std::fprintf(stderr, "[parity-test][FATAL] %s\n", msg);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 2);
}
int main(int argc, char **argv)
{
    int n_head = 4, head_dim = 16, seq_len = 5, n_head_kv = -1; // defaults
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (strncmp(a, "--heads=", 8) == 0)
            n_head = atoi(a + 8);
        else if (strncmp(a, "--head-dim=", 11) == 0)
            head_dim = atoi(a + 11);
        else if (strncmp(a, "--seq-len=", 10) == 0)
            seq_len = atoi(a + 10);
        else if (strncmp(a, "--kv-heads=", 11) == 0)
            n_head_kv = atoi(a + 11);
        else if (strcmp(a, "--help") == 0)
        {
            fprintf(stderr, "Usage: %s [--heads=N] [--head-dim=D] [--seq-len=L] [--kv-heads=K]\n", argv[0]);
            return 0;
        }
    }
    if (n_head_kv < 0)
        n_head_kv = n_head;
    if (n_head <= 0 || head_dim <= 0 || seq_len <= 0 || n_head_kv <= 0)
    {
        fprintf(stderr, "[parity-test][FATAL] invalid args heads=%d head_dim=%d seq_len=%d kv=%d\n", n_head, head_dim, seq_len, n_head_kv);
        return 2;
    }
    fprintf(stderr, "[parity-test] <entry-before-mpi-init> argc=%d (heads=%d head_dim=%d seq_len=%d kv=%d)\n", argc, n_head, head_dim, seq_len, n_head_kv);
    fflush(stderr);
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS)
    {
        fprintf(stderr, "[parity-test] MPI_Init_thread failed\n");
        return 3;
    }
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    fprintf(stderr, "[parity-test] MPI init complete rank=%d world=%d thread=%d params(h=%d hd=%d L=%d kv=%d)\n", rank, world, provided, n_head, head_dim, seq_len, n_head_kv);
    fflush(stderr);
    const int d_model = n_head * head_dim;

    auto make = [](std::vector<int> shape)
    { return TensorFactory::create_simple(shape); };
    auto input = make({seq_len, d_model});
    auto wq_global = make({n_head * head_dim, d_model}); // GGUF format: [out_features, in_features]
    auto wk_global = make({n_head_kv * head_dim, d_model});
    auto wv_global = make({n_head_kv * head_dim, d_model});
    auto wo_global = make({d_model, n_head * head_dim}); // wo is [in_features, out_features]
    auto bq_global = make({n_head * head_dim});
    auto bk_global = make({n_head_kv * head_dim});
    auto bv_global = make({n_head_kv * head_dim});
    auto k_cache = make({seq_len, n_head_kv * head_dim});
    auto v_cache = make({seq_len, n_head_kv * head_dim});
    auto out_multi = make({seq_len, d_model});

    fill_deterministic(input, 0.001f);
    fill_deterministic(wq_global, 0.002f);
    fill_deterministic(wk_global, 0.002f);
    fill_deterministic(wv_global, 0.002f);
    fill_deterministic(wo_global, 0.002f);
    fill_deterministic(bq_global, 0.0001f);
    fill_deterministic(bk_global, 0.0001f);
    fill_deterministic(bv_global, 0.0001f);
    std::fill(k_cache->data(), k_cache->data() + k_cache->size(), 0.f);
    std::fill(v_cache->data(), v_cache->data() + v_cache->size(), 0.f);

    // --- Reference single-rank naive attention (rank 0 only) ---
    // Mirrors kernel semantics: Q,K,V projections -> RoPE -> causal masked scaled softmax attention -> output projection.
    std::vector<float> ref_out(seq_len * d_model, 0.f);
    if (rank == 0)
    {
        auto matmul = [](const float *A, const float *B, float *C, int M, int K, int N)
        {
            for (int i = 0; i < M; ++i)
            {
                const float *arow = A + i * K;
                float *crow = C + i * N;
                for (int j = 0; j < N; ++j)
                {
                    double acc = 0.0;
                    for (int k = 0; k < K; ++k)
                    {
                        acc += (double)arow[k] * (double)B[k * N + j];
                    }
                    crow[j] = (float)acc;
                }
            }
        };
        // Allocate Q,K,V and O_cat
        std::vector<float> Q(seq_len * n_head * head_dim);
        std::vector<float> K(seq_len * n_head_kv * head_dim);
        std::vector<float> V(seq_len * n_head_kv * head_dim);
        matmul(input->data(), wq_global->data(), Q.data(), seq_len, d_model, n_head * head_dim);
        matmul(input->data(), wk_global->data(), K.data(), seq_len, d_model, n_head_kv * head_dim);
        matmul(input->data(), wv_global->data(), V.data(), seq_len, d_model, n_head_kv * head_dim);

        // Add biases (reference must match kernel behavior)
        for (int t = 0; t < seq_len; ++t)
        {
            for (int i = 0; i < n_head * head_dim; ++i)
            {
                Q[t * n_head * head_dim + i] += bq_global->data()[i];
            }
            for (int i = 0; i < n_head_kv * head_dim; ++i)
            {
                K[t * n_head_kv * head_dim + i] += bk_global->data()[i];
                V[t * n_head_kv * head_dim + i] += bv_global->data()[i];
            }
        }

        std::vector<float> Ocat(seq_len * n_head * head_dim, 0.f);
        const float scale = 1.0f / std::sqrt((float)head_dim);
        // Apply RoPE (same as kernel). rope_freq_base_ default assumed 10000.f.
        const float rope_base = 10000.f;
        for (int h = 0; h < n_head; ++h)
        {
            for (int t = 0; t < seq_len; ++t)
            {
                float *q_head = Q.data() + (t * (n_head * head_dim) + h * head_dim);
                float *k_head = K.data() + (t * (n_head_kv * head_dim) + h * head_dim); // n_head_kv==n_head here
                for (int dim_pair = 0; dim_pair < head_dim / 2; ++dim_pair)
                {
                    float theta = 1.0f / std::pow(rope_base, (2.0f * dim_pair) / head_dim);
                    float angle = (float)t * theta; // n_past_=0
                    float cos_t = std::cos(angle);
                    float sin_t = std::sin(angle);
                    // Q rotate
                    float q0 = q_head[2 * dim_pair];
                    float q1 = q_head[2 * dim_pair + 1];
                    q_head[2 * dim_pair] = q0 * cos_t - q1 * sin_t;
                    q_head[2 * dim_pair + 1] = q0 * sin_t + q1 * cos_t;
                    // K rotate
                    float k0 = k_head[2 * dim_pair];
                    float k1 = k_head[2 * dim_pair + 1];
                    k_head[2 * dim_pair] = k0 * cos_t - k1 * sin_t;
                    k_head[2 * dim_pair + 1] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
        for (int h = 0; h < n_head; ++h)
        {
            int kv_h = (h % n_head_kv);
            const float *qh = Q.data() + h * head_dim; // layout: interleaved heads contiguous blocks per row
            const float *kh = K.data() + kv_h * head_dim;
            const float *vh = V.data() + kv_h * head_dim;
            for (int t = 0; t < seq_len; ++t)
            {
                const float *qrow = Q.data() + (t * (n_head * head_dim) + h * head_dim);
                // Scores over sequence positions (seq_len)
                std::vector<float> scores(seq_len);
                float max_s = -1e30f;
                for (int s = 0; s < seq_len; ++s)
                {
                    const float *krow = K.data() + (s * (n_head_kv * head_dim) + kv_h * head_dim);
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += (double)qrow[d] * (double)krow[d];
                    float val = (float)(dot * scale);
                    scores[s] = val;
                    if (val > max_s)
                        max_s = val;
                }
                // Causal mask: zero out future positions (s>t) by setting to large negative prior to softmax normalization
                for (int s = t + 1; s < seq_len; ++s)
                {
                    scores[s] = -1e30f;
                }
                // Recompute max with masked values
                max_s = -1e30f;
                for (int s = 0; s < seq_len; ++s)
                {
                    if (scores[s] > max_s)
                        max_s = scores[s];
                }
                double sum_exp = 0.0;
                for (int s = 0; s < seq_len; ++s)
                {
                    scores[s] = std::exp(scores[s] - max_s);
                    sum_exp += scores[s];
                }
                float inv_sum = (float)(1.0 / sum_exp);
                float *out_block = Ocat.data() + (t * (n_head * head_dim) + h * head_dim);
                std::memset(out_block, 0, sizeof(float) * head_dim);
                for (int s = 0; s < seq_len; ++s)
                {
                    const float *vrow = V.data() + (s * (n_head_kv * head_dim) + kv_h * head_dim);
                    float w = scores[s] * inv_sum;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        out_block[d] += w * vrow[d];
                    }
                }
            }
        }
        // Final projection Ocat * Wo
        // wo_global is [d_model, n_head*head_dim], already in correct orientation for matmul
        matmul(Ocat.data(), wo_global->data(), ref_out.data(), seq_len, n_head * head_dim, d_model);
    }
    // Broadcast reference to all (optional; parity check only rank 0 needed but allows future per-rank checks)
    MPI_Bcast(ref_out.data(), (int)ref_out.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Kernel always emits partial local contribution now; no gather env needed.
    setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);

    int base = n_head / world;
    int rem = n_head % world;
    int local_heads = base + (rank < rem ? 1 : 0);
    int head_offset = base * rank + (rank < rem ? rank : rem);
    int local_head_dim = local_heads * head_dim;
    (void)local_head_dim; // presently unused beyond diagnostics
    std::fprintf(stderr, "[parity-test] rank %d local_heads=%d head_offset=%d\n", rank, local_heads, head_offset);
    std::fflush(stderr);

    auto wq = TensorFactory::create_heads_sharded(wq_global->shape(), 0, n_head, head_dim, world, rank);
    auto wk = TensorFactory::create_heads_sharded(wk_global->shape(), 0, n_head_kv, head_dim, world, rank);
    auto wv = TensorFactory::create_heads_sharded(wv_global->shape(), 0, n_head_kv, head_dim, world, rank);
    auto wo = TensorFactory::create_heads_sharded(wo_global->shape(), 1, n_head, head_dim, world, rank);

    // Create sharded bias tensors (1D, sharded along head dimension)
    auto bq = TensorFactory::create_heads_sharded({n_head * head_dim}, 0, n_head, head_dim, world, rank);
    auto bk = TensorFactory::create_heads_sharded({n_head_kv * head_dim}, 0, n_head_kv, head_dim, world, rank);
    auto bv = TensorFactory::create_heads_sharded({n_head_kv * head_dim}, 0, n_head_kv, head_dim, world, rank);

    auto copy_col_shard = [&](std::shared_ptr<TensorBase> shard, std::shared_ptr<TensorBase> global)
    {
        auto *raw = dynamic_cast<ShardedSimpleTensor *>(shard.get());
        if (!raw)
            fail_and_abort("col shard dynamic_cast failed");
        const auto &spec = raw->shard_spec();
        int rows = global->shape()[0];
        int total_cols = global->shape()[1];
        if (spec.local_dim % head_dim != 0)
            fail_and_abort("local_dim not multiple of head_dim");
        for (int r = 0; r < rows; ++r)
        {
            const float *src_row = global->data() + r * total_cols + spec.local_offset;
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
    copy_row_shard(wq, wq_global);
    copy_row_shard(wk, wk_global);
    copy_row_shard(wv, wv_global);
    copy_col_shard(wo, wo_global);

    // Copy bias shards (1D tensors, treated as row shards)
    auto copy_bias_shard = [&](std::shared_ptr<TensorBase> shard, std::shared_ptr<TensorBase> global)
    {
        auto *raw = dynamic_cast<ShardedSimpleTensor *>(shard.get());
        if (!raw)
            fail_and_abort("bias shard dynamic_cast failed");
        const auto &spec = raw->shard_spec();
        std::memcpy(shard->data(), global->data() + spec.local_offset, sizeof(float) * spec.local_dim);
    };
    copy_bias_shard(bq, bq_global);
    copy_bias_shard(bk, bk_global);
    copy_bias_shard(bv, bv_global);

    std::fprintf(stderr, "[parity-test] rank %d shards populated\n", rank);
    std::fprintf(stderr, "[parity-test] rank %d wo shape: [%d, %d]\n",
                 rank, wo->shape()[0], wo->shape()[1]);
    std::fprintf(stderr, "[parity-test] rank %d wo[0,0:4]: %.6f, %.6f, %.6f, %.6f\n",
                 rank, wo->data()[0], wo->data()[1], wo->data()[2], wo->data()[3]);
    std::fflush(stderr);

    auto kernel_multi = std::make_unique<MPIAttentionKernel>(n_head, n_head_kv, head_dim);
    std::vector<std::shared_ptr<TensorBase>> inputs_multi = {input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs_multi = {out_multi};
    bool exec_ok = kernel_multi->execute(inputs_multi, outputs_multi);
    int local_ok = exec_ok ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (global_ok != 1)
    {
        if (rank == 0)
            std::fprintf(stderr, "[parity-test][ERROR] execute failed on some rank\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    // Aggregate partial outputs: since Wo rows are head-partitioned, summation reconstructs full output.
    std::vector<float> agg(out_multi->size(), 0.f);
    MPI_Allreduce(out_multi->data(), agg.data(), (int)agg.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
    {
        // Compute errors
        double max_abs = 0.0, sum_ref = 0.0, sum_diff = 0.0;
        for (size_t i = 0; i < agg.size(); ++i)
        {
            double r = ref_out[i];
            double g = agg[i];
            double d = fabs(r - g);
            if (d > max_abs)
                max_abs = d;
            sum_ref += r * r;
            sum_diff += d * d;
        }
        double rel_l2 = (sum_ref == 0.0) ? 0.0 : std::sqrt(sum_diff / sum_ref);
        std::fprintf(stderr, "[parity-test] parity metrics: max_abs=%g rel_l2=%g (tol max_abs<=1e-5 rel_l2<=1e-5)\n", max_abs, rel_l2);
        std::fflush(stderr);
        if (!(max_abs <= 1e-5 && rel_l2 <= 1e-5))
        {
            std::fprintf(stderr, "[parity-test][FAIL] parity mismatch\n");
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
        else
        {
            std::fprintf(stderr, "[parity-test] PARITY OK\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::fprintf(stderr, "[parity-test] SUCCESS exiting normally\n");
    MPI_Finalize();
    return 0;
}
