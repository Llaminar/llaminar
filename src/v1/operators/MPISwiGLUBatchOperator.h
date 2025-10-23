#pragma once

#include "../MpiKernelBase.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar
{

    /**
     * @brief MPI-enabled SwiGLU activation operator with native batch support
     *
     * This operator extends MPISwiGLUOperator to support native 3D batch tensors.
     * SwiGLU applies the activation: swish(gate) ⊗ up (element-wise product).
     *
     * Key Differences from MPISwiGLUOperator:
     * - Native 3D input support: [batch, seq_len, hidden_ff] instead of [seq_len, hidden_ff]
     * - Native 3D output: [batch, seq_len, hidden_ff]
     * - Optimized for batch processing without reshape overhead
     * - Maintains backward compatibility: batch=1 produces identical results
     *
     * Formula:
     *  - swish(x) = x * sigmoid(x)
     *  - output = swish(gate) ⊗ up  (element-wise multiplication)
     *
     * Contract:
     * - Inputs[0]: Gate tensor [batch, seq_len, hidden_ff] (for swish activation)
     * - Inputs[1]: Up tensor [batch, seq_len, hidden_ff] (multiplication partner)
     * - Outputs[0]: Activated tensor [batch, seq_len, hidden_ff]
     *
     * @note This is a local operation (no MPI communication needed).
     *       Each rank processes its local data independently.
     *
     * @author David Sanftenberg
     */
    class MPISwiGLUBatchOperator : public MPIOperatorBase
    {
    public:
        MPISwiGLUBatchOperator(MPI_Comm comm = MPI_COMM_WORLD);
        ~MPISwiGLUBatchOperator() = default;

        // OperatorBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

    private:
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getOperatorType() const override { return "MPISwiGLUBatch"; }
        size_t getExpectedInputCount() const override { return 2; } // gate + up
        size_t getExpectedOutputCount() const override { return 1; }

        /**
         * @brief Configure OpenMP threading based on problem size
         * @param total_elements Total number of elements to process
         */
        void configureOpenMPThreading(size_t total_elements);

        /**
         * @brief Fast sigmoid approximation for swish activation
         * @param x Input value
         * @return Sigmoid(x) approximation
         */
        inline float fastSigmoid(float x) const;

        /**
         * @brief Compute swish activation: x * sigmoid(x)
         * @param x Input value
         * @return swish(x)
         */
        inline float swish(float x) const;

        int num_threads_;
    };

} // namespace llaminar
