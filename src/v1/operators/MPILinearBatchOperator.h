#pragma once

#include "../MpiKernelBase.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace llaminar
{

    /**
     * @brief MPI-enabled Linear layer operator for distributed matrix multiplication with batch support
     *
     * This operator extends MPILinearOperator to support native 3D batch tensors without flattening.
     * It processes batches of sequences efficiently while maintaining the same distribution strategy
     * as the original operator.
     *
     * Key Differences from MPILinearOperator:
     * - Native 3D input support: [batch, seq_len, in_dim] instead of [seq_len, in_dim]
     * - Native 3D output: [batch, seq_len, out_dim] instead of [seq_len, out_dim]
     * - Optimized for batch processing without reshape overhead
     * - Maintains backward compatibility: batch=1 produces identical results to MPILinearOperator
     *
     * Distribution strategy (same as MPILinearOperator):
     * - Weight matrix: Row-wise distribution across processes (each process gets subset of output dimensions)
     * - Input tensor: Replicated across all processes (broadcasted if needed)
     * - Bias vector: Row-wise distribution matching weight matrix distribution
     * - Output tensor: Row-wise distribution, gathered to all processes at the end
     *
     * Contract:
     * - Inputs[0]: Activations [batch, seq_len, in_dim] (row-major, replicated on all ranks)
     * - Inputs[1]: Global weight [out_dim, in_dim] (row-major; column-partition distributed)
     * - Inputs[2] (optional): Global bias [out_dim]
     * - Outputs[0]: Global output [batch, seq_len, out_dim] (row-major; assembled via Allgatherv)
     *
     * @note Weight convention: [out_features, in_features] matching PyTorch nn.Linear and GGUF format.
     *       Applied as output = input @ weight^T (transpose during matmul).
     *       See docs/WEIGHT_MATRIX_CONVENTIONS.md for detailed rationale.
     *
     * @author David Sanftenberg
     */
    class MPILinearBatchOperator : public MPIOperatorBase
    {
    public:
        MPILinearBatchOperator(MPI_Comm comm = MPI_COMM_WORLD);
        ~MPILinearBatchOperator() = default;

        // OperatorBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

    private:
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getOperatorType() const override { return "MPILinearBatch"; }
        size_t getExpectedInputCount() const override { return 2; } // input + weight (bias optional)
        size_t getExpectedOutputCount() const override { return 1; }

        // Weight distribution cache: maps global weight pointer to distributed local weight
        // This eliminates redundant memcpy operations across multiple forward passes
        std::unordered_map<const float *, std::shared_ptr<TensorBase>> weight_cache_;
        std::unordered_map<const float *, std::shared_ptr<TensorBase>> bias_cache_;

        /**
         * @brief Distribute weight matrix across MPI processes
         * @param global_weight Global weight matrix [out_dim, in_dim]
         * @param local_weight Local weight matrix [local_out_dim, in_dim]
         * @param output_size Global output dimension size
         */
        void distributeWeight(const std::shared_ptr<TensorBase> &global_weight,
                              std::shared_ptr<TensorBase> &local_weight,
                              size_t output_size);

        /**
         * @brief Distribute bias vector across MPI processes
         * @param global_bias Global bias vector [out_dim]
         * @param local_bias Local bias vector [local_out_dim]
         * @param output_size Global output dimension size
         */
        void distributeBias(const std::shared_ptr<TensorBase> &global_bias,
                            std::shared_ptr<TensorBase> &local_bias,
                            size_t output_size);

        /**
         * @brief Gather local output results from all processes
         * @param local_output Local output tensor [batch, seq_len, local_out_dim]
         * @param global_output Global output tensor [batch, seq_len, out_dim]
         * @param batch_size Batch size
         * @param seq_len Sequence length
         * @param output_size Global output dimension size
         */
        void gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                          std::shared_ptr<TensorBase> &global_output,
                          size_t batch_size, size_t seq_len, size_t output_size);

        /**
         * @brief Add bias to local output tensor (broadcast across batch and sequence)
         * @param output Local output data pointer [batch * seq_len, local_out_dim]
         * @param bias Local bias vector data pointer [local_out_dim]
         * @param batch_size Batch size
         * @param seq_len Sequence length
         * @param local_output_size Local output dimension size
         */
        void addBiasLocal(float *output, const float *bias,
                          size_t batch_size, size_t seq_len, size_t local_output_size);

        /**
         * @brief Create local tensor with specified dimensions
         * @param shape Shape vector for the tensor
         * @return Shared pointer to created tensor
         */
        std::shared_ptr<TensorBase> createLocalTensor(const std::vector<size_t> &shape);
    };

} // namespace llaminar
