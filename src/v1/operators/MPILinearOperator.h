#pragma once

#include "../MpiKernelBase.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace llaminar
{

    /**
     * @brief MPI-enabled Linear layer operator for distributed matrix multiplication with optional bias
     *
     * This operator implements distributed linear projection using row-wise distribution of the weight matrix.
     * Each MPI process handles a subset of the output dimensions, computing partial results locally
     * and using MPI communication to gather the complete output.
     *
     * Distribution strategy:
     * - Weight matrix: Row-wise distribution across processes (each process gets subset of output dimensions)
     * - Input tensor: Replicated across all processes (broadcasted if needed)
     * - Bias vector: Row-wise distribution matching weight matrix distribution
     * - Output tensor: Row-wise distribution, gathered to all processes at the end
     */
    class MPILinearOperator : public MPIOperatorBase
    {
    public:
        MPILinearOperator(MPI_Comm comm = MPI_COMM_WORLD);
        ~MPILinearOperator() = default;

        // OperatorBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

    private:
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getOperatorType() const override { return "MPILinear"; }
        size_t getExpectedInputCount() const override { return 2; } // input + weight (bias optional)
        size_t getExpectedOutputCount() const override { return 1; }

        // Weight distribution cache: maps global weight pointer to distributed local weight
        // This eliminates redundant memcpy operations across multiple forward passes
        std::unordered_map<const float *, std::shared_ptr<TensorBase>> weight_cache_;
        std::unordered_map<const float *, std::shared_ptr<TensorBase>> bias_cache_;

        /**
         * @brief Distribute weight matrix across MPI processes
         * @param global_weight Global weight matrix [input_size, output_size]
         * @param local_weight Local weight matrix [input_size, local_output_size]
         * @param output_size Global output dimension size
         */
        void distributeWeight(const std::shared_ptr<TensorBase> &global_weight,
                              std::shared_ptr<TensorBase> &local_weight,
                              size_t output_size);

        /**
         * @brief Distribute bias vector across MPI processes
         * @param global_bias Global bias vector [output_size]
         * @param local_bias Local bias vector [local_output_size]
         * @param output_size Global output dimension size
         */
        void distributeBias(const std::shared_ptr<TensorBase> &global_bias,
                            std::shared_ptr<TensorBase> &local_bias,
                            size_t output_size);

        /**
         * @brief Gather local output results from all processes
         * @param local_output Local output tensor [seq_len, local_output_size]
         * @param global_output Global output tensor [seq_len, output_size]
         * @param seq_len Sequence length
         * @param output_size Global output dimension size
         */
        void gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                          std::shared_ptr<TensorBase> &global_output,
                          size_t seq_len, size_t output_size);

        /**
         * @brief Add bias to local output tensor
         * @param output Local output data pointer [seq_len, local_output_size]
         * @param bias Local bias vector data pointer [local_output_size]
         * @param seq_len Sequence length
         * @param local_output_size Local output dimension size
         */
        void addBiasLocal(float *output, const float *bias,
                          size_t seq_len, size_t local_output_size);

        /**
         * @brief Create local tensor with specified dimensions
         * @param shape Shape vector for the tensor
         * @return Shared pointer to created tensor
         */
        std::shared_ptr<TensorBase> createLocalTensor(const std::vector<size_t> &shape);
    };

} // namespace llaminar