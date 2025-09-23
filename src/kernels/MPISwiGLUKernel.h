#pragma once

#include "../mpi_kernel_base.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar
{

    /**
     * @brief MPI-enabled SwiGLU activation kernel for distributed transformer FFN layers
     *
     * Implements distributed Swish-Gated Linear Unit activation:
     * SwiGLU(gate, up) = gate * silu(up) where silu(x) = x / (1 + exp(-x))
     *
     * Distribution strategies:
     * - SEQUENCE_WISE: Distribute sequence positions across MPI processes
     * - FEATURE_WISE: Distribute hidden dimensions across MPI processes
     *
     * Optimization features:
     * - OpenMP SIMD parallelization for element-wise operations
     * - NUMA-aware memory access patterns
     * - Vectorized activation function computation
     * - Work distribution balancing across MPI ranks
     *
     * Expected inputs:
     * - gate_projection: [seq_len, d_ff] - gate projection output
     * - up_projection: [seq_len, d_ff] - up projection output
     *
     * Expected outputs:
     * - swiglu_result: [seq_len, d_ff] - SwiGLU activation result
     */
    class MPISwiGLUKernel : public MPIKernelBase
    {
    public:
        enum class DistributionStrategy
        {
            SEQUENCE_WISE, ///< Distribute sequence positions across processes
            FEATURE_WISE   ///< Distribute hidden dimensions across processes
        };

        /**
         * @brief Construct MPISwiGLUKernel with specified distribution strategy
         * @param strategy Distribution strategy for MPI parallelization
         */
        MPISwiGLUKernel(DistributionStrategy strategy = DistributionStrategy::SEQUENCE_WISE);

        /**
         * @brief Destructor
         */
        ~MPISwiGLUKernel() override = default;

        /**
         * @brief Execute SwiGLU activation with MPI distribution and OpenMP parallelization
         *
         * Performs: output = gate * silu(up) where silu(x) = x / (1 + exp(-x))
         *
         * @param inputs Vector containing [gate_projection, up_projection]
         * @param outputs Vector containing [swiglu_result]
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
        std::string getName() const { return "MPISwiGLUKernel"; }

        /**
         * @brief Get the kernel type name for debugging/logging
         * @return String identifying the kernel type
         */
        std::string getKernelType() const override { return "MPISwiGLU"; }

        /**
         * @brief Get expected number of input tensors
         * @return Number of input tensors this kernel expects
         */
        size_t getExpectedInputCount() const override { return 3; } // input, gate_proj, up_proj

        /**
         * @brief Get expected number of output tensors
         * @return Number of output tensors this kernel produces
         */
        size_t getExpectedOutputCount() const override { return 1; } // swiglu_result

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

    private:
        DistributionStrategy strategy_;
        int num_threads_;

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
        void distributeMPIWork(size_t total_elements, size_t &start_idx, size_t &end_idx) const;

        /**
         * @brief Execute sequence-wise distributed SwiGLU
         * @param gate_data Gate projection data
         * @param up_data Up projection data
         * @param output_data Output data
         * @param seq_len Sequence length
         * @param d_ff Feed-forward dimension
         */
        void executeSequenceWise(const float *gate_data, const float *up_data, float *output_data,
                                 int seq_len, int d_ff);

        /**
         * @brief Execute feature-wise distributed SwiGLU
         * @param gate_data Gate projection data
         * @param up_data Up projection data
         * @param output_data Output data
         * @param seq_len Sequence length
         * @param d_ff Feed-forward dimension
         */
        void executeFeatureWise(const float *gate_data, const float *up_data, float *output_data,
                                int seq_len, int d_ff);

        /**
         * @brief Compute SiLU activation: x / (1 + exp(-x))
         * @param x Input value
         * @return SiLU activation result
         */
        inline float computeSiLU(float x) const;
    };

} // namespace llaminar