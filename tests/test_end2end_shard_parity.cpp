// End-to-end (single block) shard parity test.
// Pipeline: X -> RMSNorm(attn) -> Attention(Q,K,V,RoPE,masked softmax, Wo) + residual -> RMSNorm(ffn) -> MLP (SwiGLU) + residual.
// Verifies distributed sharded execution (2 ranks) matches single-rank reference to tight tolerance.
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>

#include "../src/kernels/MPIAttentionKernel.h"
#include "../src/kernels/MPIRMSNormKernel.h"
#include "../src/kernels/MPIMLPKernel.h"
#include "../src/utils/shard_reduce.h"
#include "../src/tensors/tensor_factory.h"
#include "../src/tensors/sharded_simple_tensor.h"

using namespace llaminar;

static void fill_deterministic(std::shared_ptr<TensorBase> &t, float scale)
{
    for (int i = 0; i < t->size(); ++i)
        t->data()[i] = scale * (float)((i % 97) - 48);
}

static void ref_matmul(const float *A, const float *B, float *C, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
    {
        const float *ar = A + i * K;
        float *cr = C + i * N;
        for (int j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)ar[k] * (double)B[k * N + j];
            cr[j] = (float)acc;
        }
    }
}

static void reference_block(int seq_len, int n_head, int head_dim, int n_head_kv, int d_ff,
                            const std::vector<float> &input, const std::vector<float> &attn_gamma,
                            const std::vector<float> &ffn_gamma,
                            const std::vector<float> &wq, const std::vector<float> &wk, const std::vector<float> &wv, const std::vector<float> &wo,
                            const std::vector<float> &w_gate, const std::vector<float> &w_up, const std::vector<float> &w_down,
                            std::vector<float> &out,
                            std::vector<float> *out_post_attn)
{
    const int d_model = n_head * head_dim;
    const float eps = 1e-6f;
    // 1) attn RMSNorm
    std::vector<float> x_norm(seq_len * d_model);
    for (int r = 0; r < seq_len; ++r)
    {
        const float *row = input.data() + r * d_model;
        double sumsq = 0.0;
        for (int c = 0; c < d_model; ++c)
        {
            double v = row[c];
            sumsq += v * v;
        }
        double inv = 1.0 / std::sqrt(sumsq / d_model + eps);
        float *dst = x_norm.data() + r * d_model;
        for (int c = 0; c < d_model; ++c)
        {
            dst[c] = (float)(row[c] * inv) * attn_gamma[c];
        }
    }
    // 2) Attention projections
    int q_cols = n_head * head_dim;
    int kv_cols = n_head_kv * head_dim;
    std::vector<float> Q(seq_len * q_cols), K(seq_len * kv_cols), V(seq_len * kv_cols);
    ref_matmul(x_norm.data(), wq.data(), Q.data(), seq_len, d_model, q_cols);
    ref_matmul(x_norm.data(), wk.data(), K.data(), seq_len, d_model, kv_cols);
    ref_matmul(x_norm.data(), wv.data(), V.data(), seq_len, d_model, kv_cols);
    // 3) RoPE
    const float rope_base = 10000.f;
    (void)rope_base;
    for (int h = 0; h < n_head; ++h)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            float *q_head = Q.data() + (t * q_cols + h * head_dim);
            float *k_head = K.data() + (t * kv_cols + (h % n_head_kv) * head_dim);
            for (int p = 0; p < head_dim / 2; ++p)
            {
                float theta = 1.0f / std::pow(rope_base, (2.0f * p) / head_dim);
                float angle = (float)t * theta;
                float c = std::cos(angle), s = std::sin(angle);
                float q0 = q_head[2 * p], q1 = q_head[2 * p + 1];
                q_head[2 * p] = q0 * c - q1 * s;
                q_head[2 * p + 1] = q0 * s + q1 * c;
                float k0 = k_head[2 * p], k1 = k_head[2 * p + 1];
                k_head[2 * p] = k0 * c - k1 * s;
                k_head[2 * p + 1] = k0 * s + k1 * c;
            }
        }
    }
    // 4) Scaled masked softmax attention per head
    std::vector<float> Ocat(seq_len * q_cols, 0.f);
    float scale = 1.0f / std::sqrt((float)head_dim);
    for (int h = 0; h < n_head; ++h)
    {
        int kvh = h % n_head_kv;
        for (int t = 0; t < seq_len; ++t)
        {
            const float *qrow = Q.data() + (t * q_cols + h * head_dim);
            std::vector<float> scores(seq_len);
            float max_s = -1e30f;
            for (int s = 0; s < seq_len; ++s)
            {
                const float *krow = K.data() + (s * kv_cols + kvh * head_dim);
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d)
                    dot += (double)qrow[d] * (double)krow[d];
                float val = (float)(dot * scale);
                if (s > t)
                    val = -1e30f;
                scores[s] = val;
                if (val > max_s)
                    max_s = val;
            }
            double sum_exp = 0.0;
            for (int s = 0; s < seq_len; ++s)
            {
                scores[s] = std::exp(scores[s] - max_s);
                sum_exp += scores[s];
            }
            float inv_sum = (float)(1.0 / sum_exp);
            float *out_block = Ocat.data() + (t * q_cols + h * head_dim);
            for (int s = 0; s < seq_len; ++s)
            {
                const float *vrow = V.data() + (s * kv_cols + kvh * head_dim);
                float w = scores[s] * inv_sum;
                for (int d = 0; d < head_dim; ++d)
                    out_block[d] += w * vrow[d];
            }
        }
    }
    // 5) Output projection
    std::vector<float> attn_out(seq_len * d_model);
    ref_matmul(Ocat.data(), wo.data(), attn_out.data(), seq_len, q_cols, d_model);
    // 6) Add residual
    std::vector<float> post_attn(seq_len * d_model);
    for (size_t i = 0; i < post_attn.size(); ++i)
        post_attn[i] = input[i] + attn_out[i];
    if (out_post_attn)
        *out_post_attn = post_attn; // capture intermediate if requested
    // 7) FFN RMSNorm
    std::vector<float> ffn_norm(seq_len * d_model);
    for (int r = 0; r < seq_len; ++r)
    {
        const float *row = post_attn.data() + r * d_model;
        double sumsq = 0.0;
        for (int c = 0; c < d_model; ++c)
        {
            double v = row[c];
            sumsq += v * v;
        }
        double inv = 1.0 / std::sqrt(sumsq / d_model + eps);
        float *dst = ffn_norm.data() + r * d_model;
        for (int c = 0; c < d_model; ++c)
            dst[c] = (float)(row[c] * inv) * ffn_gamma[c];
    }
    // 8) MLP (SwiGLU)
    std::vector<float> gate(seq_len * d_ff), up(seq_len * d_ff), act(seq_len * d_ff);
    ref_matmul(ffn_norm.data(), w_gate.data(), gate.data(), seq_len, d_model, d_ff);
    ref_matmul(ffn_norm.data(), w_up.data(), up.data(), seq_len, d_model, d_ff);
    for (size_t i = 0; i < act.size(); ++i)
    {
        float x = gate[i];
        float silu = x / (1.0f + std::exp(-x));
        act[i] = silu * up[i];
    }
    std::vector<float> ffn_down(seq_len * d_model);
    ref_matmul(act.data(), w_down.data(), ffn_down.data(), seq_len, d_ff, d_model);
    // 9) Final residual
    out.resize(seq_len * d_model);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = post_attn[i] + ffn_down[i];
}

int main(int argc, char **argv)
{
    int n_head = 4, head_dim = 8, seq_len = 6, d_ff = 48, n_head_kv = -1;
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (strncmp(a, "--heads=", 8) == 0)
            n_head = atoi(a + 8);
        else if (strncmp(a, "--head-dim=", 11) == 0)
            head_dim = atoi(a + 11);
        else if (strncmp(a, "--seq-len=", 10) == 0)
            seq_len = atoi(a + 10);
        else if (strncmp(a, "--d-ff=", 7) == 0)
            d_ff = atoi(a + 7);
        else if (strncmp(a, "--kv-heads=", 11) == 0)
            n_head_kv = atoi(a + 11);
        else if (strcmp(a, "--help") == 0)
        {
            std::fprintf(stderr, "Usage: %s [--heads=N] [--kv-heads=K] [--head-dim=D] [--seq-len=L] [--d-ff=F]\n", argv[0]);
            return 0;
        }
    }
    if (n_head_kv < 0)
        n_head_kv = n_head;
    if (n_head <= 0 || head_dim <= 0 || seq_len <= 0 || d_ff <= 0 || n_head_kv <= 0)
    {
        std::fprintf(stderr, "[e2e-parity][FATAL] invalid args\n");
        return 2;
    }
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS)
    {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 3;
    }
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    const int d_model = n_head * head_dim;
    std::fprintf(stderr, "[e2e-parity] rank=%d world=%d heads=%d kv=%d head_dim=%d seq_len=%d d_ff=%d\n", rank, world, n_head, n_head_kv, head_dim, seq_len, d_ff);
    fflush(stderr);

    setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);

    // Global full tensors for reference & slicing
    auto input = TensorFactory::create_simple({seq_len, d_model});
    fill_deterministic(input, 0.001f);
    auto attn_gamma = TensorFactory::create_simple({d_model});
    fill_deterministic(attn_gamma, 0.0007f);
    auto ffn_gamma = TensorFactory::create_simple({d_model});
    fill_deterministic(ffn_gamma, 0.0009f);
    auto wq_g = TensorFactory::create_simple({d_model, n_head * head_dim});
    fill_deterministic(wq_g, 0.002f);
    auto wk_g = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    fill_deterministic(wk_g, 0.0021f);
    auto wv_g = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    fill_deterministic(wv_g, 0.0022f);
    auto wo_g = TensorFactory::create_simple({n_head * head_dim, d_model});
    fill_deterministic(wo_g, 0.0023f);
    auto w_gate_g = TensorFactory::create_simple({d_model, d_ff});
    fill_deterministic(w_gate_g, 0.003f);
    auto w_up_g = TensorFactory::create_simple({d_model, d_ff});
    fill_deterministic(w_up_g, 0.0031f);
    auto w_down_g = TensorFactory::create_simple({d_ff, d_model});
    fill_deterministic(w_down_g, 0.0032f);

    // Reference (rank 0)
    std::vector<float> ref_output;
    std::vector<float> ref_post_attn;
    bool dump_stages = std::getenv("LLAMINAR_E2E_PARITY_DUMP_STAGE") != nullptr;
    if (rank == 0)
    {
        reference_block(seq_len, n_head, head_dim, n_head_kv, d_ff,
                        std::vector<float>(input->data(), input->data() + input->size()),
                        std::vector<float>(attn_gamma->data(), attn_gamma->data() + d_model),
                        std::vector<float>(ffn_gamma->data(), ffn_gamma->data() + d_model),
                        std::vector<float>(wq_g->data(), wq_g->data() + wq_g->size()),
                        std::vector<float>(wk_g->data(), wk_g->data() + wk_g->size()),
                        std::vector<float>(wv_g->data(), wv_g->data() + wv_g->size()),
                        std::vector<float>(wo_g->data(), wo_g->data() + wo_g->size()),
                        std::vector<float>(w_gate_g->data(), w_gate_g->data() + w_gate_g->size()),
                        std::vector<float>(w_up_g->data(), w_up_g->data() + w_up_g->size()),
                        std::vector<float>(w_down_g->data(), w_down_g->data() + w_down_g->size()),
                        ref_output,
                        dump_stages ? &ref_post_attn : nullptr);
    }
    if (rank != 0)
        ref_output.resize((size_t)seq_len * d_model);
    MPI_Bcast(ref_output.data(), (int)ref_output.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (dump_stages)
    {
        if (rank != 0)
            ref_post_attn.resize((size_t)seq_len * d_model);
        MPI_Bcast(ref_post_attn.data(), (int)ref_post_attn.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Shards
    auto wq = TensorFactory::create_heads_sharded(wq_g->shape(), 1, n_head, head_dim, world, rank);
    auto wk = TensorFactory::create_heads_sharded(wk_g->shape(), 1, n_head_kv, head_dim, world, rank);
    auto wv = TensorFactory::create_heads_sharded(wv_g->shape(), 1, n_head_kv, head_dim, world, rank);
    auto wo = TensorFactory::create_heads_sharded(wo_g->shape(), 0, n_head, head_dim, world, rank);
    auto w_gate = TensorFactory::create_sharded(w_gate_g->shape(), ShardSpec::Axis::Hidden, 1, world, rank);
    auto w_up = TensorFactory::create_sharded(w_up_g->shape(), ShardSpec::Axis::Hidden, 1, world, rank);
    auto w_down = TensorFactory::create_sharded(w_down_g->shape(), ShardSpec::Axis::Hidden, 0, world, rank);

    auto copy_col_shard = [](std::shared_ptr<TensorBase> shard, std::shared_ptr<TensorBase> global)
    {
        auto *raw = dynamic_cast<ShardedSimpleTensor*>(shard.get()); if(!raw){ fprintf(stderr,"[e2e-parity][FATAL] col shard cast fail\n"); MPI_Abort(MPI_COMM_WORLD,5);} const auto &spec = raw->shard_spec(); int rows=global->shape()[0]; int gcols=global->shape()[1]; for(int r=0;r<rows;++r){ const float* src = global->data()+ r*gcols + spec.local_offset; float* dst = shard->data() + r*spec.local_dim; std::memcpy(dst, src, sizeof(float)*spec.local_dim);} };
    auto copy_row_shard = [](std::shared_ptr<TensorBase> shard, std::shared_ptr<TensorBase> global)
    {
        auto *raw = dynamic_cast<ShardedSimpleTensor*>(shard.get()); if(!raw){ fprintf(stderr,"[e2e-parity][FATAL] row shard cast fail\n"); MPI_Abort(MPI_COMM_WORLD,6);} const auto &spec = raw->shard_spec(); int cols=global->shape()[1]; for(size_t i=0;i<spec.local_dim;++i){ size_t g = spec.local_offset + i; std::memcpy(shard->data()+ i*cols, global->data()+ g*cols, sizeof(float)*cols);} };
    copy_col_shard(wq, wq_g);
    copy_col_shard(wk, wk_g);
    copy_col_shard(wv, wv_g);
    copy_row_shard(wo, wo_g);
    copy_col_shard(w_gate, w_gate_g);
    copy_col_shard(w_up, w_up_g);
    copy_row_shard(w_down, w_down_g);

    // Activation & temporary tensors
    auto k_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});
    std::fill(k_cache->data(), k_cache->data() + k_cache->size(), 0.f);
    auto v_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});
    std::fill(v_cache->data(), v_cache->data() + v_cache->size(), 0.f);
    auto attn_out = TensorFactory::create_simple({seq_len, d_model});

    // 1) Attention Norm (RMSNorm) replicated gamma
    auto attn_norm_out = TensorFactory::create_simple({seq_len, d_model});
    {
        MPIRMSNormKernel norm_kernel; // default strategy
        std::vector<std::shared_ptr<TensorBase>> in = {input, attn_gamma};
        std::vector<std::shared_ptr<TensorBase>> outv = {attn_norm_out};
        if (!norm_kernel.execute(in, outv))
        {
            fprintf(stderr, "[e2e-parity][ERROR] attn norm failed\n");
            MPI_Abort(MPI_COMM_WORLD, 7);
        }
    }
    // 2) Attention
    auto attn_kernel = std::make_unique<MPIAttentionKernel>(n_head, n_head_kv, head_dim);
    std::vector<std::shared_ptr<TensorBase>> attn_inputs = {attn_norm_out, wq, wk, wv, wo, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out};
    if (!attn_kernel->execute(attn_inputs, attn_outputs))
    {
        fprintf(stderr, "[e2e-parity][ERROR] attention failed\n");
        MPI_Abort(MPI_COMM_WORLD, 8);
    }
    // Attention kernel currently produces fully aggregated output (due to internal Allreduce) when gather disabled via reduction path.
    // 3) Reduce attention partial shards (row-sharded additive) using helper
    reduce_row_sharded_inplace(attn_out, /*validate_axis=*/false, /*treat_nonsharded_as_additive=*/true); // attn_out logically additive across ranks
    auto post_attn = TensorFactory::create_simple({seq_len, d_model});
    for (int r = 0; r < seq_len; ++r)
    {
        float *dst = post_attn->data() + r * d_model;
        const float *a = input->data() + r * d_model;
        const float *b = attn_out->data() + r * d_model;
        for (int c = 0; c < d_model; ++c)
            dst[c] = a[c] + b[c];
    }
    // 4) FFN Norm
    auto ffn_norm_out = TensorFactory::create_simple({seq_len, d_model});
    {
        MPIRMSNormKernel ffn_norm;
        std::vector<std::shared_ptr<TensorBase>> in = {post_attn, ffn_gamma};
        std::vector<std::shared_ptr<TensorBase>> outv = {ffn_norm_out};
        if (!ffn_norm.execute(in, outv))
        {
            fprintf(stderr, "[e2e-parity][ERROR] ffn norm failed\n");
            MPI_Abort(MPI_COMM_WORLD, 9);
        }
    }
    // 5) MLP
    auto mlp_out = TensorFactory::create_simple({seq_len, d_model});
    auto mlp_kernel = std::make_unique<MPIMLPKernel>();
    std::vector<std::shared_ptr<TensorBase>> mlp_inputs = {ffn_norm_out, w_gate, w_up, w_down};
    std::vector<std::shared_ptr<TensorBase>> mlp_outputs = {mlp_out};
    if (!mlp_kernel->execute(mlp_inputs, mlp_outputs))
    {
        fprintf(stderr, "[e2e-parity][ERROR] mlp failed\n");
        MPI_Abort(MPI_COMM_WORLD, 10);
    }
    // 6) Final residual
    std::vector<float> final_out((size_t)seq_len * d_model);
    for (size_t i = 0; i < final_out.size(); ++i)
        final_out[i] = post_attn->data()[i] + mlp_out->data()[i];

    // Stage diff (post-attn) if requested
    if (dump_stages)
    {
        // compute distributed post_attn diff vs reference
        double max_abs_pa = 0.0, sum_ref_pa = 0.0, sum_diff_pa = 0.0;
        for (size_t i = 0; i < ref_post_attn.size(); ++i)
        {
            double r = ref_post_attn[i];
            double g = post_attn->data()[i];
            double d = fabs(r - g);
            if (d > max_abs_pa)
                max_abs_pa = d;
            sum_ref_pa += r * r;
            sum_diff_pa += d * d;
        }
        double rel_l2_pa = (sum_ref_pa == 0.0) ? 0.0 : std::sqrt(sum_diff_pa / sum_ref_pa);
        if (rank == 0)
            std::fprintf(stderr, "[e2e-parity][stage post_attn] max_abs=%g rel_l2=%g\n", max_abs_pa, rel_l2_pa);
    }

    // Compare final to reference (rank 0)
    if (rank == 0)
    {
        double max_abs = 0.0, sum_ref = 0.0, sum_diff = 0.0;
        for (size_t i = 0; i < final_out.size(); ++i)
        {
            double r = ref_output[i];
            double g = final_out[i];
            double d = fabs(r - g);
            if (d > max_abs)
                max_abs = d;
            sum_ref += r * r;
            sum_diff += d * d;
        }
        double rel_l2 = (sum_ref == 0.0) ? 0.0 : std::sqrt(sum_diff / sum_ref);
        std::fprintf(stderr, "[e2e-parity] metrics: max_abs=%g rel_l2=%g (tol 1e-5)\n", max_abs, rel_l2);
        if (!(max_abs <= 1e-5 && rel_l2 <= 1e-5))
        {
            std::fprintf(stderr, "[e2e-parity][FAIL] parity mismatch\n");
            MPI_Abort(MPI_COMM_WORLD, 11);
        }
        else
        {
            std::fprintf(stderr, "[e2e-parity] PARITY OK\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::fprintf(stderr, "[e2e-parity] SUCCESS exiting normally\n");
    MPI_Finalize();
    return 0;
}
