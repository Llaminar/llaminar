#pragma once

#include "../MpiKernelBase.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar
{

    /**
     * @brief MPI-enabled Residual Connection kernel for distributed transformer layers
     *
     * Implements distributed residual connections: output = input + residual
     * Common in transformer architectures for gradient flow and training stability.
     *
     * Distribution strategies:
     * - SEQUENCE_WISE: Distribute sequence positions across MPI processes
     * - ELEMENT_WISE: Distribute all tensor elements across MPI processes
     *
     * Optimization features:
     * - OpenMP SIMD parallelization for element-wise addition
     * - Broadcasting support for different tensor shapes
     * - NUMA-aware memory access patterns
     * - Vectorized operations with cache optimization
     *
     * Expected inputs:
     * - input: [seq_len, hidden_size] - primary input tensor
     * - residual: [seq_len, hidden_size] - residual connection tensor
     *
     * Expected outputs:
     * - output: [seq_len, hidden_size] - sum of input and residual
     */
    class MPIResidualOperator : public MPIOperatorBase
    {
    public:
        enum class DistributionStrategy
        {
            SEQUENCE_WISE, ///< Distribute sequence positions across processes
            ELEMENT_WISE   ///< Distribute all elements across processes
        };

        /**
         * @brief Construct MPIResidualOperator with specified distribution strategy
         * @param strategy Distribution strategy for MPI parallelization
         */
        MPIResidualOperator(DistributionStrategy strategy = DistributionStrategy::SEQUENCE_WISE);

        /**
         * @brief Destructor
         */
        ~MPIResidualOperator() override = default;

        /**
         * @brief Execute residual connection with MPI distribution and OpenMP parallelization
         *
         * Performs: output = input + residual
         *
         * @param inputs Vector containing [input, residual]
         * @param outputs Vector containing [output]
         * @return true if execution successful, false otherwise
         */
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        /**
         * @brief Validate input and output tensors
         * @param inputs Input tensors to validate
         * @param outputs Output tensors to validate
         * @return true if validation passes, false otherwise
         */
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        /**
         * @brief Get kernel name for debugging and profiling
         * @return Kernel name string
         */
        std::string getName() const { return "MPIResidualOperator"; }

        /**
         * @brief Get the kernel type name for debugging/logging
         * @return String identifying the kernel type
         */
        std::string getOperatorType() const override { return "MPIResidual"; }

        /**
         * @brief Get expected number of input tensors
         * @return Number of input tensors this kernel expects
         */
        size_t getExpectedInputCount() const override { return 2; } // input, residual

        /**
         * @brief Get expected number of output tensors
         * @return Number of output tensors this kernel produces
         */
        size_t getExpectedOutputCount() const override { return 1; } // result

        /**
         * @brief Set distribution strategy
         * @param strategy New distribution strategy
         */
        void setDistributionStrategy(DistributionStrategy strategy);

        /**
         * @brief Get current distribution strategy
         * @return Current distribution strategy
         */
        DistributionStrategy getDistributionStrategy() const { return strategy_; }

        /**
         * @brief Enable or disable broadcasting for shape-mismatched tensors
         * @param enable Whether to enable broadcasting
         */
        void setBroadcasting(bool enable) { broadcasting_enabled_ = enable; }

        /**
         * @brief Check if broadcasting is enabled
         * @return true if broadcasting is enabled
         */
        bool isBroadcastingEnabled() const { return broadcasting_enabled_; }

    private:
        DistributionStrategy strategy_;
        int num_threads_;
        bool broadcasting_enabled_;

        /**
         * @brief Configure OpenMP threading based on tensor size and MPI context
         * @param tensor_size Total number of elements to process
         */
        void configureOpenMPThreading(size_t tensor_size);

        /**
         * @brief Distribute work across MPI ranks
         * @param total_elements Total elements in tensor
         * @param start_idx Output: starting index for this rank
         * @param end_idx Output: ending index for this rank
         */
        void distributeMPIWork(size_t total_elements, size_t &start_idx, size_t &end_idx, bool replicated) const;

        /**
         * @brief Execute sequence-wise distributed residual addition
         * @param input_data Input tensor data
         * @param residual_data Residual tensor data
         * @param output_data Output tensor data
         * @param seq_len Sequence length
         * @param hidden_size Hidden dimension size
         */
        void executeSequenceWise(const float *input_data, const float *residual_data, float *output_data,
                                 int seq_len, int hidden_size, bool replicated);

        /**
         * @brief Execute element-wise distributed residual addition
         * @param input_data Input tensor data
         * @param residual_data Residual tensor data
         * @param output_data Output tensor data
         * @param total_elements Total number of elements
         */
        void executeElementWise(const float *input_data, const float *residual_data, float *output_data,
                                size_t total_elements, bool replicated);

        /**
         * @brief Check if tensor shapes are compatible for residual connection
         * @param input_shape Input tensor shape
         * @param residual_shape Residual tensor shape
         * @return true if compatible, false otherwise
         */
        bool areShapesCompatible(const std::vector<int> &input_shape,
                                 const std::vector<int> &residual_shape) const;

        /**
         * @brief Execute residual connection with broadcasting support
         * @param input_data Input tensor data
         * @param residual_data Residual tensor data
         * @param output_data Output tensor data
         * @param input_shape Input tensor shape
         * @param residual_shape Residual tensor shape
         */
        void executeBroadcast(const float *input_data, const float *residual_data, float *output_data,
                              const std::vector<int> &input_shape,
                              const std::vector<int> &residual_shape);
    };

} // namespace llaminar