#pragma once

#include <vector>
#include <memory>
#include "../tensor.h"

namespace llaminar
{

    /**
     * @brief Linear projection kernel for transformer layers
     *
     * Implements linear transformation: y = x * W + b
     * where W is the weight matrix and b is the optional bias vector
     *
     * Expected inputs:
     * - input: [seq_len, input_size] - input tensor
     * - weight: [input_size, output_size] - weight matrix
     * - bias: [output_size] - bias vector (optional)
     *
     * Expected outputs:
     * - output: [seq_len, output_size] - transformed output tensor
     */
    class LinearKernel
    {
    public:
        LinearKernel();

        /**
         * @brief Execute linear projection
         * @param inputs Vector containing input tensor, weight matrix, and optional bias
         * @param outputs Vector containing output tensor
         * @return true if execution succeeded, false otherwise
         */
        bool execute(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                     std::vector<std::shared_ptr<llaminar::Tensor>> &outputs);

        /**
         * @brief Validate input and output tensor shapes and types
         * @param inputs Input tensors to validate
         * @param outputs Output tensors to validate
         * @return true if tensors are valid, false otherwise
         */
        bool validate(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) const;

    private:
        /**
         * @brief Core linear projection computation
         * @param input Input data pointer
         * @param weight Weight matrix data pointer
         * @param bias Bias vector data pointer (can be nullptr)
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param input_size Input dimension size
         * @param output_size Output dimension size
         */
        void computeLinearProjection(const float *input, const float *weight, const float *bias,
                                     float *output, int seq_len, int input_size, int output_size);
    };

} // namespace llaminar