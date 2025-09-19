#pragma once

#include <vector>
#include <memory>
#include <cmath> // For expf
#include "../tensor.h"

namespace llaminar
{

    /**
     * @brief MLP (Multi-Layer Perceptron) kernel for transformer feed-forward networks
     *
     * Implements the SwiGLU activation pattern commonly used in modern transformers:
     * gate_proj = input @ gate_weight
     * up_proj = input @ up_weight
     * activated = gate_proj * silu(up_proj)
     * output = activated @ down_weight
     *
     * where silu(x) = x / (1 + exp(-x))
     *
     * Expected inputs:
     * - input: [seq_len, hidden_size] - input tensor
     * - gate_weight: [hidden_size, ff_size] - gate projection weight
     * - up_weight: [hidden_size, ff_size] - up projection weight
     * - down_weight: [ff_size, hidden_size] - down projection weight
     *
     * Expected outputs:
     * - output: [seq_len, hidden_size] - transformed output tensor
     */
    class MLPKernel
    {
    public:
        MLPKernel();

        /**
         * @brief Execute MLP forward pass
         * @param inputs Vector containing input, gate_weight, up_weight, down_weight
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
         * @brief Compute gate and up projections
         * @param input Input data pointer
         * @param gate_weight Gate weight matrix pointer
         * @param up_weight Up weight matrix pointer
         * @param gate_proj Output gate projection pointer
         * @param up_proj Output up projection pointer
         * @param seq_len Sequence length
         * @param hidden_size Hidden dimension size
         * @param ff_size Feed-forward dimension size
         */
        void computeGateUpProjections(const float *input,
                                      const float *gate_weight, const float *up_weight,
                                      float *gate_proj, float *up_proj,
                                      int seq_len, int hidden_size, int ff_size);

        /**
         * @brief Apply SiLU activation: gate_proj * silu(up_proj)
         * @param gate_proj Gate projection data
         * @param up_proj Up projection data
         * @param activated Output activated data
         * @param seq_len Sequence length
         * @param ff_size Feed-forward dimension size
         */
        void applySiLUActivation(const float *gate_proj, const float *up_proj,
                                 float *activated, int seq_len, int ff_size);

        /**
         * @brief Compute down projection
         * @param activated Activated data pointer
         * @param down_weight Down weight matrix pointer
         * @param output Output data pointer
         * @param seq_len Sequence length
         * @param ff_size Feed-forward dimension size
         * @param hidden_size Hidden dimension size
         */
        void computeDownProjection(const float *activated, const float *down_weight,
                                   float *output, int seq_len, int ff_size, int hidden_size);

        /**
         * @brief SiLU (Swish) activation function
         * @param x Input value
         * @return SiLU activated value
         */
        static float silu(float x) { return x / (1.0f + expf(-x)); }
    };

} // namespace llaminar