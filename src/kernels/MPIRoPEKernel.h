#pragma once

#include "../mpi_kernel_base.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar
{

    /**
     * @brief MPI-enabled Rotary Position Embedding (RoPE) kernel for distributed attention layers
     *
     * Implements distributed Rotary Position Embedding as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
     * https://arxiv.org/abs/2104.09864
     *
     * RoPE applies rotary transformations to query and key embeddings based on position:
     * - Precomputes sine/cosine tables for efficiency
     * - Applies rotation in the complex plane representation
     * - Preserves relative position information across sequence lengths
     *
     * Distribution strategies:
     * - SEQUENCE_WISE: Distribute sequence positions across MPI processes
     * - HEAD_WISE: Distribute attention heads across MPI processes
     *
     * Optimization features:
     * - OpenMP SIMD parallelization for rotation computations
     * - Precomputed frequency tables to avoid repeated calculations
     * - Cache-friendly memory access patterns
     * - NUMA-aware work distribution
     *
     * Expected inputs:
     * - input: [seq_len, n_heads, head_dim] - query or key tensor to rotate
     * - position_ids: [seq_len] - position indices for each token
     *
     * Expected outputs:
     * - rotated_output: [seq_len, n_heads, head_dim] - position-encoded tensor
     */
    class MPIRoPEKernel : public MPIKernelBase
    {
    public:
        enum class DistributionStrategy
        {
            SEQUENCE_WISE, ///< Distribute sequence positions across processes
            HEAD_WISE      ///< Distribute attention heads across processes
        };

        /**
         * @brief Construct MPIRoPEKernel with specified parameters
         * @param max_seq_len Maximum sequence length for precomputation
         * @param head_dim Dimension of each attention head
         * @param theta Base frequency for rotary embeddings (default: 10000.0)
         * @param strategy Distribution strategy for MPI parallelization
         */
        MPIRoPEKernel(int max_seq_len, int head_dim, float theta = 10000.0f,
                      DistributionStrategy strategy = DistributionStrategy::SEQUENCE_WISE);

        /**
         * @brief Destructor
         */
        ~MPIRoPEKernel() override = default;

        /**
         * @brief Execute RoPE with MPI distribution and OpenMP parallelization
         *
         * Applies rotary position embedding: output = rotate(input, position_ids)
         *
         * @param inputs Vector containing [input_tensor, position_ids]
         * @param outputs Vector containing [rotated_output]
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
        std::string getName() const { return "MPIRoPEKernel"; }

        /**
         * @brief Get the kernel type name for debugging/logging
         * @return String identifying the kernel type
         */
        std::string getKernelType() const override { return "MPIRoPE"; }

        /**
         * @brief Get expected number of input tensors
         * @return Number of input tensors this kernel expects
         */
        size_t getExpectedInputCount() const override { return 2; } // q_tensor, k_tensor

        /**
         * @brief Get expected number of output tensors
         * @return Number of output tensors this kernel produces
         */
        size_t getExpectedOutputCount() const override { return 2; } // modified q_tensor, k_tensor

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
         * @brief Update maximum sequence length and recompute frequency tables
         * @param max_seq_len New maximum sequence length
         */
        void updateMaxSeqLen(int max_seq_len);

        /**
         * @brief Get current maximum sequence length
         * @return Maximum sequence length
         */
        int getMaxSeqLen() const { return max_seq_len_; }

    private:
        int max_seq_len_;
        int head_dim_;
        float theta_;
        DistributionStrategy strategy_;
        int num_threads_;

        // Precomputed frequency tables
        std::vector<float> cos_table_;
        std::vector<float> sin_table_;

        /**
         * @brief Precompute sine and cosine frequency tables
         */
        void precomputeFrequencyTables();

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
         * @brief Execute sequence-wise distributed RoPE
         * @param input_data Input tensor data
         * @param position_ids Position indices
         * @param output_data Output tensor data
         * @param seq_len Sequence length
         * @param n_heads Number of attention heads
         * @param head_dim Head dimension
         */
        void executeSequenceWise(const float *input_data, const int *position_ids, float *output_data,
                                 int seq_len, int n_heads, int head_dim);

        /**
         * @brief Execute head-wise distributed RoPE
         * @param input_data Input tensor data
         * @param position_ids Position indices
         * @param output_data Output tensor data
         * @param seq_len Sequence length
         * @param n_heads Number of attention heads
         * @param head_dim Head dimension
         */
        void executeHeadWise(const float *input_data, const int *position_ids, float *output_data,
                             int seq_len, int n_heads, int head_dim);

        /**
         * @brief Apply rotary embedding to a single position and head
         * @param input_ptr Pointer to input data for this position/head
         * @param output_ptr Pointer to output data for this position/head
         * @param position Position index
         * @param head_dim Head dimension
         */
        void applyRotaryEmbedding(const float *input_ptr, float *output_ptr, int position, int head_dim);

        /**
         * @brief Get precomputed cosine value for position and dimension
         * @param position Position index
         * @param dim Dimension index (must be even)
         * @return Cosine value
         */
        inline float getCos(int position, int dim) const;

        /**
         * @brief Get precomputed sine value for position and dimension
         * @param position Position index
         * @param dim Dimension index (must be even)
         * @return Sine value
         */
        inline float getSin(int position, int dim) const;
    };

} // namespace llaminar