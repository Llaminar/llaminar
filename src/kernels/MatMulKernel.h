// Legacy MatMulKernel compatibility wrapper
// This file restores the previous test dependency while routing all
// matrix multiplications through the unified AdaptiveMatMulManager.
//
// STATUS: DEPRECATED. New code should call adaptiveMatMul()/AdaptiveMatMulManager
// directly. This wrapper exists only so older tests (test_cosma, etc.) build
// without re-writing them immediately. It intentionally does NOT expose any
// COSMA specific knobs beyond a minimal strategy string accepted previously.
//
// Unification notes:
//  - All decisions (OpenBLAS vs COSMA) now happen in AdaptiveMatMulManager.
//  - The test_cosma benchmark still exercises COSMA when thresholds permit.
//  - Remove this file once tests are migrated to adaptive API.

#pragma once

#include <vector>
#include <memory>
#include "../tensors/tensor_base.h"
#include "../logger.h"
#include "../adaptive_matmul.h"

namespace llaminar
{

    class MatMulKernel
    {
    public:
        MatMulKernel() = default;

        // Retained for backward compatibility; currently ignored except for logging.
        void setStrategy(const std::string &strategy) { strategy_ = strategy; }
        void setBlockSizes(int /*mb*/, int /*nb*/, int /*kb*/) { /* ignored in unified path */ }

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const
        {
            if (inputs.size() != 2 || outputs.size() != 1)
            {
                LOG_ERROR("MatMulKernel (compat): expected 2 inputs, 1 output");
                return false;
            }
            if (!inputs[0] || !inputs[1] || !outputs[0])
            {
                LOG_ERROR("MatMulKernel (compat): null tensor provided");
                return false;
            }
            const auto &A_shape = inputs[0]->shape();
            const auto &B_shape = inputs[1]->shape();
            const auto &C_shape = outputs[0]->shape();
            if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2)
            {
                LOG_ERROR("MatMulKernel (compat): all tensors must be 2D");
                return false;
            }
            if (A_shape[1] != B_shape[0])
            {
                LOG_ERROR("MatMulKernel (compat): inner dimension mismatch");
                return false;
            }
            if (C_shape[0] != A_shape[0] || C_shape[1] != B_shape[1])
            {
                LOG_ERROR("MatMulKernel (compat): output shape mismatch");
                return false;
            }
            return true;
        }

        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs)
        {
            if (!validate(inputs, outputs))
                return false;

            auto A = inputs[0];
            auto B = inputs[1];
            auto C = outputs[0];
            int m = static_cast<int>(A->shape()[0]);
            int k = static_cast<int>(A->shape()[1]);
            int n = static_cast<int>(B->shape()[1]);

            // Determine if this looks like a prefill (sequence length heuristic)
            bool is_prefill = (m >= 64); // mirrors COSMA_MIN_SEQ_LEN threshold

            LOG_DEBUG("MatMulKernel (compat): delegating " << m << "x" << k << " * " << k << "x" << n
                                                           << (is_prefill ? " (prefill heuristic)" : ""));

            bool ok = adaptiveMatMul(A->data(), B->data(), const_cast<float *>(C->data()),
                                     m, n, k, is_prefill, /*distributed_partition*/ false);
            if (!ok)
            {
                LOG_ERROR("MatMulKernel (compat): adaptiveMatMul failed");
            }
            return ok;
        }

    private:
        std::string strategy_{"auto"};
    };

} // namespace llaminar
