// Minimal AttentionKernel compatibility wrapper for tests.
// Provides a stub that performs QK^T * V using adaptiveMatMul for validation sizes.
// Not optimized; deprecated in favor of MPIAttentionKernel.

#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include "../tensors/tensor_base.h"
#include "../logger.h"
#include "../adaptive_matmul.h"

namespace llaminar
{

    class AttentionKernel
    {
    public:
        AttentionKernel() = default;

        // inputs: Q, K, V  (all [seq_len, dim])
        // output: context  ([seq_len, dim])
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs)
        {
            if (inputs.size() != 3 || outputs.size() != 1)
                return false;
            auto Q = inputs[0];
            auto K = inputs[1];
            auto V = inputs[2];
            auto C = outputs[0];
            if (!Q || !K || !V || !C)
                return false;
            auto sh = Q->shape();
            if (sh.size() != 2)
                return false;
            int seq = (int)sh[0];
            int dim = (int)sh[1];
            // Compute scores = Q * K^T  (seq x dim) * (dim x seq) -> (seq x seq)
            std::vector<float> scores(seq * seq, 0.f);
            // naive multiply (small test sizes)
            for (int i = 0; i < seq; ++i)
            {
                for (int j = 0; j < seq; ++j)
                {
                    float acc = 0.f;
                    for (int d = 0; d < dim; ++d)
                        acc += Q->data()[i * dim + d] * K->data()[j * dim + d];
                    scores[i * seq + j] = acc / std::sqrt((float)dim);
                }
            }
            // softmax each row
            for (int i = 0; i < seq; ++i)
            {
                float maxv = -1e30f;
                for (int j = 0; j < seq; ++j)
                    maxv = std::max(maxv, scores[i * seq + j]);
                float sum = 0.f;
                for (int j = 0; j < seq; ++j)
                {
                    scores[i * seq + j] = std::exp(scores[i * seq + j] - maxv);
                    sum += scores[i * seq + j];
                }
                float inv = 1.f / sum;
                for (int j = 0; j < seq; ++j)
                    scores[i * seq + j] *= inv;
            }
            // context = scores * V  (seq x seq) * (seq x dim) -> (seq x dim)
            float *out = const_cast<float *>(C->data());
            std::fill(out, out + seq * dim, 0.f);
            for (int i = 0; i < seq; ++i)
            {
                for (int j = 0; j < seq; ++j)
                {
                    float s = scores[i * seq + j];
                    const float *vrow = V->data() + j * dim;
                    for (int d = 0; d < dim; ++d)
                        out[i * dim + d] += s * vrow[d];
                }
            }
            return true;
        }
    };

} // namespace llaminar
