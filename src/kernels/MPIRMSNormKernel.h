#pragma once

#include "../mpi_kernel_base.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar
{

    /**
     * @brief MPI-enabled RMS Normalization kernel for distributed transformer layers
     *
     * Implements distributed Root Mean Square Layer Normalization with sequence-wise distribution:
     * - Each process handles a subset of sequence positions
     * - Global RMS statistics are computed via MPI communication
     * - Supports both sequence-wise and feature-wise distribution patterns
     *
     * Distribution strategies:
     * 1. SEQUENCE_WISE: Distribute sequence positions across processes
     * 2. FEATURE_WISE: Distribute hidden dimensions across processes (future)
     *
     * Expected inputs:
     * - input: [seq_len, hidden_size] - distributed input tensor
     * - weight: [hidden_size] - scale parameters (replicated on all processes)
     *
     * Expected outputs:
     * - output: [seq_len, hidden_size] - distributed normalized output tensor
     */
    class MPIRMSNormKernel : public MPIKernelBase
    {
    public:
        enum class DistributionStrategy
        {
            SEQUENCE_WISE, ///< Distribute sequence positions across processes
            FEATURE_WISE   ///< Distribute hidden dimensions across processes (future)
        };

        MPIRMSNormKernel(DistributionStrategy strategy = DistributionStrategy::SEQUENCE_WISE);

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getKernelType() const override { return "MPIRMSNorm"; }
        size_t getExpectedInputCount() const override { return 2; }
        size_t getExpectedOutputCount() const override { return 1; }

        // Configuration
        void setEpsilon(float eps) { epsilon_ = eps; }
        float getEpsilon() const { return epsilon_; }

        void setDistributionStrategy(DistributionStrategy strategy) { strategy_ = strategy; }
        DistributionStrategy getDistributionStrategy() const { return strategy_; }

    private:
        /**
         * @brief Distribute input tensor across processes according to strategy
         * @param global_input Global input tensor [seq_len, hidden_size]
         * @param local_input Local input tensor to populate
         * @param global_seq_len Global sequence length
         * @param hidden_size Hidden dimension size
         */
        void distributeInput(const std::shared_ptr<TensorBase> &global_input,
                             std::shared_ptr<TensorBase> &local_input,
                             size_t global_seq_len, size_t hidden_size);

        /**
         * @brief Gather local outputs to form complete global output
         * @param local_output Local output tensor
         * @param global_output Global output tensor to populate
         * @param global_seq_len Global sequence length
         * @param hidden_size Hidden dimension size
         */
        void gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                          std::shared_ptr<TensorBase> &global_output,
                          size_t global_seq_len, size_t hidden_size);

        /**
         * @brief Compute RMS normalization with distributed sequence-wise processing
         * @param local_input Local input data [local_seq_len, hidden_size]
         * @param weight Weight data [hidden_size] (replicated)
         * @param local_output Local output data [local_seq_len, hidden_size]
         * @param local_seq_len Local sequence length
         * @param hidden_size Hidden dimension size
         * @param global_seq_len Global sequence length (for proper RMS scaling)
         */
        void computeDistributedRMSNorm(const float *local_input, const float *weight,
                                       float *local_output, size_t local_seq_len,
                                       size_t hidden_size, size_t global_seq_len);

        /**
         * @brief Compute global RMS statistics across all processes
         * @param local_input Local input data
         * @param local_seq_len Local sequence length
         * @param hidden_size Hidden dimension size
         * @param global_seq_len Global sequence length
         * @return Global RMS value
         */
        float computeGlobalRMS(const float *local_input, size_t local_seq_len,
                               size_t hidden_size, size_t global_seq_len);

        /**
         * @brief Create a local tensor with specified dimensions
         * @param shape Tensor shape
         * @return New local tensor
         */
        std::shared_ptr<TensorBase> createLocalTensor(const std::vector<size_t> &shape);

        float epsilon_;                 ///< Small value to prevent division by zero
        DistributionStrategy strategy_; ///< Distribution strategy for parallelization
    };

} // namespace llaminar