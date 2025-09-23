// Legacy LinearKernel compatibility wrapper.
// Provides a simple (non-MPI) linear layer: Y = X * W (+ bias)
// Delegates matmul to adaptiveMatMul unified backend. Deprecated.

#pragma once

#include <vector>
#include <memory>
#include "../tensors/tensor_base.h"
#include "../logger.h"
#include "../adaptive_matmul.h"

namespace llaminar
{

    class LinearKernel
    {
    public:
        LinearKernel() = default;

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const
        {
            if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
                return false;
            auto A = inputs[0];
            auto W = inputs[1];
            auto Y = outputs[0];
            if (!A || !W || !Y)
                return false;
            const auto &Ash = A->shape();
            const auto &Wsh = W->shape();
            const auto &Ysh = Y->shape();
            if (Ash.size() != 2 || Wsh.size() != 2 || Ysh.size() != 2)
                return false;
            if (Ash[1] != Wsh[0])
                return false;
            if (Ysh[0] != Ash[0] || Ysh[1] != Wsh[1])
                return false;
            if (inputs.size() == 3 && inputs[2])
            {
                const auto &Bsh = inputs[2]->shape();
                if (Bsh.size() != 1 || Bsh[0] != Wsh[1])
                    return false;
            }
            return true;
        }

        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs)
        {
            if (!validate(inputs, outputs))
            {
                LOG_ERROR("LinearKernel (compat): validation failed");
                return false;
            }
            auto A = inputs[0];
            auto W = inputs[1];
            auto Y = outputs[0];
            int m = static_cast<int>(A->shape()[0]);
            int k = static_cast<int>(A->shape()[1]);
            int n = static_cast<int>(W->shape()[1]);

            bool is_prefill = (m >= 64);
            bool ok = adaptiveMatMul(A->data(), W->data(), const_cast<float *>(Y->data()),
                                     m, n, k, is_prefill, /*distributed_partition*/ false);
            if (!ok)
                return false;

            if (inputs.size() == 3 && inputs[2])
            {
                auto B = inputs[2];
                const float *bias = B->data();
                float *y = const_cast<float *>(Y->data());
                for (int row = 0; row < m; ++row)
                {
                    for (int col = 0; col < n; ++col)
                    {
                        y[row * n + col] += bias[col];
                    }
                }
            }
            return true;
        }
    };

} // namespace llaminar
