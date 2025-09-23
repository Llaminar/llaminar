#pragma once

#include "../mpi_kernel_base.h"
#include <string>
#include <vector>
#include <memory>

namespace llaminar
{

    /**
     * @brief MPI-distributed Multi-layer perceptron (MLP) kernel with SwiGLU activation
     *
     * Distribution strategy: Pipeline pattern with COSMA-powered matrix operations
     * - Gate/Up projections: column-wise distribution across ranks
     * - SwiGLU activation: local element-wise operation (no communication)
     * - Down projection: row-wise distribution with MPI_Allreduce
     * - All matrix multiplications use COSMA for optimal performance
     */
    class MPIMLPKernel : public MPIKernelBase
    {
    public:
        MPIMLPKernel();
        ~MPIMLPKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        std::string getKernelType() const override { return "MPIMLP"; }
        size_t getExpectedInputCount() const override { return 4; } // input, w_gate, w_up, w_down
        size_t getExpectedOutputCount() const override { return 1; }

    private:
        /**
         * @brief Compute gate projection using COSMA distributed matrix multiplication
         * @param input Input tensor [seq_len, d_model]
         * @param w_gate Gate weight matrix [d_model, d_ff] (distributed across ranks)
         * @param gate_output Gate projection output [seq_len, local_d_ff]
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param local_d_ff Local feed-forward dimension on this rank
         */
        void computeGateProjection(const std::shared_ptr<TensorBase> &input,
                                   const std::shared_ptr<TensorBase> &w_gate,
                                   std::shared_ptr<TensorBase> &gate_output,
                                   size_t seq_len, size_t d_model, size_t local_d_ff);

        /**
         * @brief Compute up projection using COSMA distributed matrix multiplication
         * @param input Input tensor [seq_len, d_model]
         * @param w_up Up weight matrix [d_model, d_ff] (distributed across ranks)
         * @param up_output Up projection output [seq_len, local_d_ff]
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param local_d_ff Local feed-forward dimension on this rank
         */
        void computeUpProjection(const std::shared_ptr<TensorBase> &input,
                                 const std::shared_ptr<TensorBase> &w_up,
                                 std::shared_ptr<TensorBase> &up_output,
                                 size_t seq_len, size_t d_model, size_t local_d_ff);

        /**
         * @brief Apply SwiGLU activation function locally (no MPI communication)
         * @param gate_output Gate projection values [seq_len, local_d_ff]
         * @param up_output Up projection values [seq_len, local_d_ff]
         * @param activated_output Combined output after SwiGLU [seq_len, local_d_ff]
         * @param seq_len Sequence length
         * @param local_d_ff Local feed-forward dimension on this rank
         */
        void applySwiGLU(const std::shared_ptr<TensorBase> &gate_output,
                         const std::shared_ptr<TensorBase> &up_output,
                         std::shared_ptr<TensorBase> &activated_output,
                         size_t seq_len, size_t local_d_ff);

        /**
         * @brief Compute down projection using COSMA and gather results
         * @param activated_input Activated input from SwiGLU [seq_len, local_d_ff]
         * @param w_down Down weight matrix [d_ff, d_model] (distributed across ranks)
         * @param local_output Local output before reduction [seq_len, d_model]
         * @param seq_len Sequence length
         * @param local_d_ff Local feed-forward dimension on this rank
         * @param d_model Model dimension
         */
        void computeDownProjection(const std::shared_ptr<TensorBase> &activated_input,
                                   const std::shared_ptr<TensorBase> &w_down,
                                   std::shared_ptr<TensorBase> &local_output,
                                   size_t seq_len, size_t local_d_ff, size_t d_model);

        /**
         * @brief Gather final output using MPI_Allreduce
         * @param local_output Local output from this rank [seq_len, d_model]
         * @param global_output Final combined output [seq_len, d_model]
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void gatherFinalOutput(const std::shared_ptr<TensorBase> &local_output,
                               std::shared_ptr<TensorBase> &global_output,
                               size_t seq_len, size_t d_model);

        /**
         * @brief SiLU (Swish) activation function: x * sigmoid(x)
         * @param x Input value
         * @return SiLU(x)
         */
        float silu(float x) const;

        /**
         * @brief Calculate local feed-forward dimension for this rank
         * @param global_d_ff Global feed-forward dimension
         * @return Local portion of d_ff for this rank
         */
        size_t calculateLocalDff(size_t global_d_ff) const;

        /**
         * @brief Validate tensor shapes and dimensions for MPI MLP
         */
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;
    };

} // namespace llaminar