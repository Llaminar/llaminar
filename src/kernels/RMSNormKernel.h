#pragma once

#include <vector>
#include <memory>
#include "../tensor.h"

namespace llaminar
{

    /**
     * @brief RMS Normalization kernel for transformer layers
     *
     * Implements Root Mean Square Layer Normalization:
     * y = (x / sqrt(mean(x^2) + eps)) * weight
     *
     * Expected inputs:
     * - input: [seq_len, hidden_size] - input tensor
     * - weight: [hidden_size] - learnable scale parameters
     *
     * Expected outputs:
     * - output: [seq_len, hidden_size] - normalized output tensor
     */
    class RMSNormKernel
    {
    public:
        RMSNormKernel();

        /**
         * @brief Execute RMS normalization
         * @param inputs Vector containing input tensor and weight tensor
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

        // Configuration
        void setEpsilon(float eps) { epsilon_ = eps; }
        float getEpsilon() const { return epsilon_; }

    private:
        /**
         * @brief Core RMS normalization computation
         * @param input Input data pointer
         * @param weight Weight data pointer
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param hidden_size Hidden dimension size
         */
        void computeRMSNorm(const float *input, const float *weight,
                            float *output, int seq_len, int hidden_size);

        float epsilon_; ///< Small value to prevent division by zero
    };

} // namespace llaminar