#include "RMSNormKernel.h"
#include "graph_compute.h" // For Tensor definition
#include "logger.h"
#include <cmath>
#include <chrono>

namespace llaminar
{

    RMSNormKernel::RMSNormKernel() : epsilon_(1e-6f)
    {
        LOG_DEBUG("RMSNormKernel initialized with epsilon=" + std::to_string(epsilon_));
    }

    bool RMSNormKernel::execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto input = inputs[0];   // [seq_len, hidden_size]
        auto weight = inputs[1];  // [hidden_size]
        auto output = outputs[0]; // [seq_len, hidden_size]

        int seq_len = input->shape[0];
        int hidden_size = input->shape[1];

        computeRMSNorm(input->data.data(), weight->data.data(),
                       output->data.data(), seq_len, hidden_size);

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("RMSNorm executed in " + std::to_string(execution_time) + " ms");
        return true;
    }

    bool RMSNormKernel::validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                 const std::vector<std::shared_ptr<Tensor>> &outputs) const
    {
        if (inputs.size() != 2)
        {
            LOG_ERROR("RMSNorm requires exactly 2 inputs (input, weight)");
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("RMSNorm requires exactly 1 output");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (input->shape.size() != 2 || weight->shape.size() != 1 || output->shape.size() != 2)
        {
            LOG_ERROR("RMSNorm tensor shape mismatch");
            return false;
        }

        if (input->shape[1] != weight->shape[0] ||
            input->shape[0] != output->shape[0] ||
            input->shape[1] != output->shape[1])
        {
            LOG_ERROR("RMSNorm dimension mismatch");
            return false;
        }

        return true;
    }

    void RMSNormKernel::computeRMSNorm(const float *input, const float *weight,
                                       float *output, int seq_len, int hidden_size)
    {
        for (int seq = 0; seq < seq_len; ++seq)
        {
            const float *x = input + seq * hidden_size;
            float *y = output + seq * hidden_size;

            // Compute RMS (Root Mean Square)
            float sum_squares = 0.0f;
            for (int i = 0; i < hidden_size; ++i)
            {
                sum_squares += x[i] * x[i];
            }
            float rms = sqrtf(sum_squares / hidden_size + epsilon_);

            // Apply normalization and weight
            for (int i = 0; i < hidden_size; ++i)
            {
                y[i] = (x[i] / rms) * weight[i];
            }
        }
    }

} // namespace llaminar