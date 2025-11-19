/**
 * @file CPUSwiGLUKernel.cpp
 * @brief CPU SwiGLU kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPUSwiGLUKernel.h"
#include <cmath>
#include <omp.h>

namespace llaminar2
{

    bool CPUSwiGLUKernel::apply(
        const float *gate, const float *up, float *output,
        int seq_len, int d_ff,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        const int total_elements = seq_len * d_ff;

#pragma omp parallel for
        for (int i = 0; i < total_elements; ++i)
        {
            float g = gate[i];
            float u = up[i];

            // SwiGLU: gate * silu(up)
            // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float sigmoid_u = 1.0f / (1.0f + std::exp(-u));
            float silu_u = u * sigmoid_u;

            output[i] = g * silu_u;
        }

        return true;
    }

} // namespace llaminar2
