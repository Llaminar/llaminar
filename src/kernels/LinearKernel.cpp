#include "LinearKernel.h"
#include "graph_compute.h" // For Tensor definition
#include "logger.h"
#include <cblas.h>
#include <chrono>

namespace llaminar
{

    LinearKernel::LinearKernel()
    {
        LOG_DEBUG("LinearKernel initialized");
    }

    bool LinearKernel::execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                               std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto input = inputs[0];                              // [seq_len, input_size]
        auto weight = inputs[1];                             // [input_size, output_size]
        auto bias = inputs.size() > 2 ? inputs[2] : nullptr; // [output_size] (optional)
        auto output = outputs[0];                            // [seq_len, output_size]

        int seq_len = input->shape[0];
        int input_size = input->shape[1];
        int output_size = weight->shape[1];

        computeLinearProjection(input->data.data(), weight->data.data(),
                                bias ? bias->data.data() : nullptr,
                                output->data.data(), seq_len, input_size, output_size);

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("Linear executed in " + std::to_string(execution_time) + " ms");
        return true;
    }

    bool LinearKernel::validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                const std::vector<std::shared_ptr<Tensor>> &outputs) const
    {
        if (inputs.size() < 2 || inputs.size() > 3)
        {
            LOG_ERROR("Linear requires 2 or 3 inputs (input, weight, optional bias)");
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("Linear requires exactly 1 output");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (input->shape.size() != 2 || weight->shape.size() != 2 || output->shape.size() != 2)
        {
            LOG_ERROR("Linear requires 2D tensors");
            return false;
        }

        if (input->shape[1] != weight->shape[0])
        {
            LOG_ERROR("Linear dimension mismatch: input cols != weight rows");
            return false;
        }

        if (input->shape[0] != output->shape[0] || weight->shape[1] != output->shape[1])
        {
            LOG_ERROR("Linear output dimension mismatch");
            return false;
        }

        // Validate bias if present
        if (inputs.size() == 3)
        {
            auto bias = inputs[2];
            if (bias->shape.size() != 1 || bias->shape[0] != weight->shape[1])
            {
                LOG_ERROR("Linear bias dimension mismatch");
                return false;
            }
        }

        return true;
    }

    void LinearKernel::computeLinearProjection(const float *input, const float *weight, const float *bias,
                                               float *output, int seq_len, int input_size, int output_size)
    {
        // Matrix multiplication: input @ weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, output_size, input_size,
                    1.0f, input, input_size,
                    weight, output_size,
                    0.0f, output, output_size);

        // Add bias if provided
        if (bias)
        {
            for (int seq = 0; seq < seq_len; ++seq)
            {
                for (int out = 0; out < output_size; ++out)
                {
                    output[seq * output_size + out] += bias[out];
                }
            }
        }
    }

} // namespace llaminar