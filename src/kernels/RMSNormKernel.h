// Minimal RMSNormKernel compatibility wrapper.
// y = x * scale / sqrt(mean(x^2) + eps)

#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include "../tensors/tensor_base.h"
#include "../logger.h"

namespace llaminar
{

    class RMSNormKernel
    {
    public:
        RMSNormKernel(float eps = 1e-6f) : eps_(eps) {}

        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs)
        {
            if (inputs.size() < 1 || outputs.size() != 1)
                return false;
            auto X = inputs[0];
            auto Y = outputs[0];
            if (!X || !Y)
                return false;
            const float *scale = nullptr;
            if (inputs.size() >= 2 && inputs[1])
                scale = inputs[1]->data();
            auto sh = X->shape();
            if (sh.size() != 2)
                return false;
            int rows = (int)sh[0];
            int dim = (int)sh[1];
            float *out = const_cast<float *>(Y->data());
            for (int r = 0; r < rows; ++r)
            {
                const float *xrow = X->data() + r * dim;
                float mean_sq = 0.f;
                for (int d = 0; d < dim; ++d)
                    mean_sq += xrow[d] * xrow[d];
                mean_sq /= dim;
                float inv = 1.f / std::sqrt(mean_sq + eps_);
                for (int d = 0; d < dim; ++d)
                {
                    float val = xrow[d] * inv;
                    if (scale)
                        val *= scale[d];
                    out[r * dim + d] = val;
                }
            }
            return true;
        }

    private:
        float eps_;
    };

} // namespace llaminar
