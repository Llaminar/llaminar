#pragma once

#include "../tensors/tensor_base.h"
#include "../tensor.h"
#include <vector>
#include <memory>
#include <string>

// Forward declarations
namespace cosma
{
    template <typename T>
    class CosmaMatrix;
    class Strategy;
}

namespace llaminar
{

    /**
     * @brief High-performance matrix multiplication kernel using COSMA
     *
     * Implements distributed matrix multiplication using the COSMA library
     * for optimal communication patterns and performance on MPI systems.
     *
     * Now supports hybrid tensor types:
     * - COSMATensor: Zero-copy operations with optimal COSMA performance
     * - SimpleTensor: Fallback with data conversion for compatibility
     *
     * Expected inputs:
     * - A: [m, k] - left matrix
     * - B: [k, n] - right matrix
     *
     * Expected outputs:
     * - C: [m, n] - result matrix (C = A × B)
     */
    class MatMulKernel
    {
    public:
        MatMulKernel();

        /**
         * @brief Execute matrix multiplication using COSMA with hybrid tensor support
         * @param inputs Vector containing A and B matrices (TensorBase-derived)
         * @param outputs Vector containing result matrix C (TensorBase-derived)
         * @return true if execution succeeded, false otherwise
         */
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs);

        // Legacy interface for compatibility
        bool execute(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                     std::vector<std::shared_ptr<llaminar::Tensor>> &outputs);

        /**
         * @brief Validate input and output tensor shapes for matrix multiplication
         * @param inputs Input tensors to validate
         * @param outputs Output tensors to validate
         * @return true if tensors are valid, false otherwise
         */
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const;

        // Legacy validation interface
        bool validate(const std::vector<std::shared_ptr<llaminar::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<llaminar::Tensor>> &outputs) const;

        // COSMA-specific configuration
        void setStrategy(const std::string &strategy) { strategy_ = strategy; }
        void setBlockSizes(int block_m, int block_n, int block_k)
        {
            block_m_ = block_m;
            block_n_ = block_n;
            block_k_ = block_k;
        }

        const std::string &getStrategy() const { return strategy_; }
        void getBlockSizes(int &block_m, int &block_n, int &block_k) const
        {
            block_m = block_m_;
            block_n = block_n_;
            block_k = block_k_;
        }

    private:
        std::string strategy_;            ///< COSMA strategy ("auto", "custom", etc.)
        int block_m_, block_n_, block_k_; ///< Block sizes for COSMA algorithm

        /**
         * @brief Zero-copy COSMA matrix multiplication for COSMATensor inputs
         * @param A Left COSMA matrix
         * @param B Right COSMA matrix
         * @param C Result COSMA matrix
         * @return true if COSMA execution succeeded
         */
        bool executeCOSMANative(cosma::CosmaMatrix<float> &A,
                                cosma::CosmaMatrix<float> &B,
                                cosma::CosmaMatrix<float> &C,
                                const cosma::Strategy &strategy);

        /**
         * @brief Legacy COSMA execution with data copying (for SimpleTensor)
         * @param A Left matrix
         * @param B Right matrix
         * @param C Result matrix
         * @return true if COSMA execution succeeded
         */
        bool executeCOSMA(const TensorBase &A, const TensorBase &B, TensorBase &C);

        /**
         * @brief Detect optimal execution path based on tensor types
         * @param inputs Input tensors to analyze
         * @param outputs Output tensors to analyze
         * @return true if zero-copy COSMA execution is possible
         */
        bool canUseZeroCopyCOSMA(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                 const std::vector<std::shared_ptr<TensorBase>> &outputs) const;

        /**
         * @brief Initialize COSMA context and configuration
         */
        void initializeCOSMAContext();
    };

} // namespace llaminar