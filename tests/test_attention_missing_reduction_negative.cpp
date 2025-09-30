// Negative test: verifies that using the MPIAttentionKernel partial (per-rank) output
// without performing the required post-attention reduction does NOT accidentally
// match the full single-rank reference output. The test PASSES only if a large
// mismatch is observed (i.e. misuse is detectable). It also enables the
// LLAMINAR_ASSERT_REPLICATED_MISUSE guard so a microscopic per-rank canary is
// embedded, though the primary detection is the large numerical discrepancy from
// missing head contributions.
//
// Labels (via CTest): attention;negative;sharding
// Expected execution: mpirun -np 2 test_attention_missing_reduction_negative

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

static void fatal(const char *msg)
{
    std::fprintf(stderr, "[missing-reduction-neg][FATAL] %s\n", msg);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

int main(int argc, char **argv)
{
    int n_head = 4, head_dim = 16, seq_len = 5, n_head_kv = -1; // default small config
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
            std::fprintf(stderr, "Usage: %s [--heads=N] [--head-dim=D] [--seq-len=L] [--kv-heads=K]\n", argv[0]);
            return 0;
        }
    }
    if (n_head_kv < 0)
        n_head_kv = n_head;
    if (n_head <= 0 || head_dim <= 0 || seq_len <= 0 || n_head_kv <= 0)
    {
        std::fprintf(stderr, "[missing-reduction-neg][FATAL] invalid args heads=%d head_dim=%d seq_len=%d kv=%d\n", n_head, head_dim, seq_len, n_head_kv);
        return 2;
    }

    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS)
    {
        std::fprintf(stderr, "[missing-reduction-neg] MPI_Init_thread failed\n");
        return 3;
    }
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world < 2 && rank == 0)
    {
        std::fprintf(stderr, "[missing-reduction-neg][WARN] world size <2; test requires >=2 ranks to be meaningful\n");
    }
    const int d_model = n_head * head_dim;

    auto make = [](std::vector<int> shape)
    { return TensorFactory::create_simple(shape); };
    auto input = make({seq_len, d_model});
    auto wq_global = make({d_model, n_head * head_dim});
    auto wk_global = make({d_model, n_head_kv * head_dim});
    auto wv_global = make({d_model, n_head_kv * head_dim});
    auto wo_global = make({n_head * head_dim, d_model});
    auto k_cache = make({seq_len, n_head_kv * head_dim});
    auto v_cache = make({seq_len, n_head_kv * head_dim});
    auto out_partial = make({seq_len, d_model});

    fill_deterministic(input, 0.001f);
    fill_deterministic(wq_global, 0.002f);
    fill_deterministic(wk_global, 0.002f);
    fill_deterministic(wv_global, 0.002f);
    fill_deterministic(wo_global, 0.002f);
    std::fill(k_cache->data(), k_cache->data() + k_cache->size(), 0.f);
    std::fill(v_cache->data(), v_cache->data() + v_cache->size(), 0.f);

    // Reference full (single-rank global) attention computed by rank 0.
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
                        acc += (double)arow[k] * (double)B[k * N + j];
                    crow[j] = (float)acc;
                }
            }
        };
        std::vector<float> Q(seq_len * n_head * head_dim);
        std::vector<float> K(seq_len * n_head_kv * head_dim);
        std::vector<float> V(seq_len * n_head_kv * head_dim);
        matmul(input->data(), wq_global->data(), Q.data(), seq_len, d_model, n_head * head_dim);
        matmul(input->data(), wk_global->data(), K.data(), seq_len, d_model, n_head_kv * head_dim);
        matmul(input->data(), wv_global->data(), V.data(), seq_len, d_model, n_head_kv * head_dim);
        const float rope_base = 10000.f;
        for (int h = 0; h < n_head; ++h)
        {
            for (int t = 0; t < seq_len; ++t)
            {
                float *q_head = Q.data() + (t * (n_head * head_dim) + h * head_dim);
                float *k_head = K.data() + (t * (n_head_kv * head_dim) + h * head_dim);
                for (int dp = 0; dp < head_dim / 2; ++dp)
                {
                    float theta = 1.0f / std::pow(rope_base, (2.0f * dp) / head_dim);
                    float angle = (float)t * theta;
                    float cos_t = std::cos(angle);
                    float sin_t = std::sin(angle);
                    float q0 = q_head[2 * dp];
                    float q1 = q_head[2 * dp + 1];
                    q_head[2 * dp] = q0 * cos_t - q1 * sin_t;
                    q_head[2 * dp + 1] = q0 * sin_t + q1 * cos_t;
                    float k0 = k_head[2 * dp];
                    float k1 = k_head[2 * dp + 1];
                    k_head[2 * dp] = k0 * cos_t - k1 * sin_t;
                    k_head[2 * dp + 1] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
        std::vector<float> Ocat(seq_len * n_head * head_dim, 0.f);
        const float scale = 1.0f / std::sqrt((float)head_dim);
        for (int h = 0; h < n_head; ++h)
        {
            int kv_h = (h % n_head_kv);
            for (int t = 0; t < seq_len; ++t)
            {
                const float *qrow = Q.data() + (t * (n_head * head_dim) + h * head_dim);
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
                for (int s = t + 1; s < seq_len; ++s)
                    scores[s] = -1e30f; // causal mask
                max_s = -1e30f;
                for (int s = 0; s < seq_len; ++s)
                    if (scores[s] > max_s)
                        max_s = scores[s];
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
                        out_block[d] += w * vrow[d];
                }
            }
        }
        matmul(Ocat.data(), wo_global->data(), ref_out.data(), seq_len, n_head * head_dim, d_model);
    }
    MPI_Bcast(ref_out.data(), (int)ref_out.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Disable COSMA for determinism & inject canary markers.
    setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);
    setenv("LLAMINAR_ASSERT_REPLICATED_MISUSE", "1", 1);

    // Prepare sharded weights for local heads only.
    int base = n_head / world;
    int rem = n_head % world;
    int local_heads = base + (rank < rem ? 1 : 0);
    int head_offset = base * rank + (rank < rem ? rank : rem);
    (void)head_offset; // diagnostic only

    auto wq = TensorFactory::create_heads_sharded(wq_global->shape(), 1, n_head, head_dim, world, rank);
    auto wk = TensorFactory::create_heads_sharded(wk_global->shape(), 1, n_head_kv, head_dim, world, rank);
    auto wv = TensorFactory::create_heads_sharded(wv_global->shape(), 1, n_head_kv, head_dim, world, rank);
    auto wo = TensorFactory::create_heads_sharded(wo_global->shape(), 0, n_head, head_dim, world, rank);

    auto copy_col_shard = [&](std::shared_ptr<TensorBase> shard, std::shared_ptr<TensorBase> global)
    {
        auto *raw = dynamic_cast<ShardedSimpleTensor *>(shard.get());
        if (!raw)
            fatal("col shard dynamic_cast failed");
        const auto &spec = raw->shard_spec();
        int rows = global->shape()[0];
        int total_cols = global->shape()[1];
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
            fatal("row shard dynamic_cast failed");
        const auto &spec = raw->shard_spec();
        int cols = global->shape()[1];
        for (size_t i = 0; i < (size_t)spec.local_dim; ++i)
        {
            size_t g = spec.local_offset + i;
            std::memcpy(shard->data() + i * cols, global->data() + g * cols, sizeof(float) * cols);
        }
    };
    copy_col_shard(wq, wq_global);
    copy_col_shard(wk, wk_global);
    copy_col_shard(wv, wv_global);
    copy_row_shard(wo, wo_global);

    auto kernel = std::make_unique<MPIAttentionKernel>(n_head, n_head_kv, head_dim);
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {out_partial};
    bool ok = kernel->execute(inputs, outputs);
    int local_ok = ok ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (global_ok != 1)
    {
        if (rank == 0)
            std::fprintf(stderr, "[missing-reduction-neg][ERROR] kernel execute failed on some rank\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    // (Intentional MISUSE): do NOT aggregate partials. Rank 0 compares its own partial to full reference.
    if (rank == 0)
    {
        double max_abs = 0.0, sum_ref = 0.0, sum_diff = 0.0;
        for (size_t i = 0; i < ref_out.size(); ++i)
        {
            double r = ref_out[i];
            double g = out_partial->data()[i];
            double d = fabs(r - g);
            if (d > max_abs)
                max_abs = d;
            sum_ref += r * r;
            sum_diff += d * d;
        }
        double rel_l2 = (sum_ref == 0.0) ? 0.0 : std::sqrt(sum_diff / sum_ref);
        std::fprintf(stderr, "[missing-reduction-neg] misuse metrics (expected MISMATCH): max_abs=%g rel_l2=%g\n", max_abs, rel_l2);
        // We expect a large mismatch because roughly (world-1)/world heads are missing; enforce a lower bound.
        // If max_abs is suspiciously tiny, misuse went undetected -> FAIL.
        if (max_abs < 1e-4)
        {
            std::fprintf(stderr, "[missing-reduction-neg][FAIL] partial output too close to full reference (reduction missing but not detected)\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
        else
        {
            std::fprintf(stderr, "[missing-reduction-neg] Expected mismatch confirmed (test passes)\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::fprintf(stderr, "[missing-reduction-neg] SUCCESS exiting\n");
    MPI_Finalize();
    return 0;
}
