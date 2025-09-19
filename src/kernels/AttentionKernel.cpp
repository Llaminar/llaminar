#include "AttentionKernel.h"
#include "graph_compute.h" // For Tensor definition
#include "logger.h"
#include <cblas.h>
#include <cmath>
#include <chrono>

namespace llaminar
{

    AttentionKernel::AttentionKernel()
        : n_head_(32), n_head_kv_(32), head_dim_(128), n_past_(0)
    {
        LOG_DEBUG("AttentionKernel initialized");
    }

    bool AttentionKernel::execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                  std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Input tensors
        auto input = inputs[0];         // [seq_len, hidden_size]
        auto q_weight = inputs[1];      // [hidden_size, hidden_size]
        auto k_weight = inputs[2];      // [hidden_size, hidden_size]
        auto v_weight = inputs[3];      // [hidden_size, hidden_size]
        auto output_weight = inputs[4]; // [hidden_size, hidden_size]
        auto k_cache = inputs[5];       // [max_seq_len, n_head_kv, head_dim]
        auto v_cache = inputs[6];       // [max_seq_len, n_head_kv, head_dim]

        // Output tensors
        auto output = outputs[0]; // [seq_len, hidden_size]

        int seq_len = input->shape[0];
        int hidden_size = input->shape[1];

        // Temporary tensors for intermediate computations
        std::vector<float> q_proj(seq_len * hidden_size);
        std::vector<float> k_proj(seq_len * hidden_size);
        std::vector<float> v_proj(seq_len * hidden_size);
        std::vector<float> attn_output(seq_len * hidden_size);

        // Compute Q, K, V projections
        computeQKVProjections(input->data.data(),
                              q_weight->data.data(), k_weight->data.data(), v_weight->data.data(),
                              nullptr, nullptr, nullptr, // No bias for now
                              q_proj.data(), k_proj.data(), v_proj.data(),
                              seq_len, hidden_size);

        // Update KV cache
        updateKVCache(k_proj.data(), v_proj.data(),
                      k_cache->data.data(), v_cache->data.data(),
                      seq_len, n_head_kv_, head_dim_, n_past_);

        // Compute scaled dot-product attention
        std::vector<float> attn_weights(seq_len * (n_past_ + seq_len) * n_head_);
        computeScaledDotProductAttention(q_proj.data(), k_cache->data.data(), v_cache->data.data(),
                                         attn_weights.data(), attn_output.data(),
                                         seq_len, n_head_, head_dim_);

        // Apply output projection
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, hidden_size, hidden_size,
                    1.0f, attn_output.data(), hidden_size,
                    output_weight->data.data(), hidden_size,
                    0.0f, output->data.data(), hidden_size);

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("Attention executed in " + std::to_string(execution_time) + " ms");
        return true;
    }

    bool AttentionKernel::validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                   const std::vector<std::shared_ptr<Tensor>> &outputs) const
    {
        if (inputs.size() != 7)
        {
            LOG_ERROR("Attention requires exactly 7 inputs");
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("Attention requires exactly 1 output");
            return false;
        }

        // Basic shape validation - could be more comprehensive
        auto input = inputs[0];
        auto output = outputs[0];

        if (input->shape.size() != 2 || output->shape.size() != 2)
        {
            LOG_ERROR("Attention input and output must be 2D tensors");
            return false;
        }

        if (input->shape[0] != output->shape[0] || input->shape[1] != output->shape[1])
        {
            LOG_ERROR("Attention output dimensions must match input");
            return false;
        }

        return true;
    }

    void AttentionKernel::setHeadDimensions(int n_head, int n_head_kv, int head_dim)
    {
        n_head_ = n_head;
        n_head_kv_ = n_head_kv;
        head_dim_ = head_dim;
    }

    void AttentionKernel::computeQKVProjections(const float *input,
                                                const float *q_weight, const float *k_weight, const float *v_weight,
                                                const float *q_bias, const float *k_bias, const float *v_bias,
                                                float *q_proj, float *k_proj, float *v_proj,
                                                int seq_len, int hidden_size)
    {
        // Q projection: input @ q_weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, hidden_size, hidden_size,
                    1.0f, input, hidden_size,
                    q_weight, hidden_size,
                    0.0f, q_proj, hidden_size);

        // K projection: input @ k_weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, hidden_size, hidden_size,
                    1.0f, input, hidden_size,
                    k_weight, hidden_size,
                    0.0f, k_proj, hidden_size);

        // V projection: input @ v_weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, hidden_size, hidden_size,
                    1.0f, input, hidden_size,
                    v_weight, hidden_size,
                    0.0f, v_proj, hidden_size);
    }

    void AttentionKernel::computeScaledDotProductAttention(const float *q, const float *k, const float *v,
                                                           float *attn_weights, float *output,
                                                           int seq_len, int n_head, int head_dim)
    {
        // Simplified attention computation - implement proper multi-head attention here
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // For now, just copy input to output (placeholder)
        int hidden_size = n_head * head_dim;
        for (int i = 0; i < seq_len * hidden_size; ++i)
        {
            output[i] = q[i] * 0.5f + v[i] * 0.5f; // Simple combination
        }
    }

    void AttentionKernel::updateKVCache(const float *k_new, const float *v_new,
                                        float *k_cache, float *v_cache,
                                        int seq_len, int n_head_kv, int head_dim, int n_past)
    {
        // Copy new K and V to cache at position n_past
        int hidden_size = n_head_kv * head_dim;
        for (int seq = 0; seq < seq_len; ++seq)
        {
            for (int dim = 0; dim < hidden_size; ++dim)
            {
                k_cache[(n_past + seq) * hidden_size + dim] = k_new[seq * hidden_size + dim];
                v_cache[(n_past + seq) * hidden_size + dim] = v_new[seq * hidden_size + dim];
            }
        }
    }

} // namespace llaminar