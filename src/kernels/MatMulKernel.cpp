#include "MatMulKernel.h"
#include "../tensors/tensor_base.h"
#include "../tensors/simple_tensor.h"
#include "../tensors/cosma_tensor.h"
#include "../tensors/tensor_factory.h" // Include factory for conversion utilities
#include "../logger.h"
#include <iostream>
#include <mpi.h>
#include <chrono>

// COSMA headers
#include <cosma/multiply.hpp>
#include <cosma/strategy.hpp>
#include <cosma/matrix.hpp>
#include <cosma/context.hpp>

namespace llaminar
{

    MatMulKernel::MatMulKernel()
        : strategy_("auto"), block_m_(256), block_n_(256), block_k_(256)
    {
        LOG_DEBUG("MatMulKernel initialized with strategy=" + strategy_);
    }

    bool MatMulKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                               std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto A = inputs[0];
        auto B = inputs[1];
        auto C = outputs[0];

        bool success = false;

        // Check if we can use zero-copy COSMA execution
        if (canUseZeroCopyCOSMA(inputs, outputs))
        {
            LOG_DEBUG("Using zero-copy COSMA execution path");

            // Cast to COSMA tensors for direct access
            auto cosma_A = std::dynamic_pointer_cast<COSMATensor>(A);
            auto cosma_B = std::dynamic_pointer_cast<COSMATensor>(B);
            auto cosma_C = std::dynamic_pointer_cast<COSMATensor>(C);

            if (cosma_A && cosma_B && cosma_C)
            {
                success = executeCOSMANative(cosma_A->cosma_matrix(),
                                             cosma_B->cosma_matrix(),
                                             cosma_C->cosma_matrix(),
                                             cosma_A->strategy());
            }
        }
        else
        {
            LOG_DEBUG("Using legacy COSMA execution path with data conversion");
            success = executeCOSMA(*A, *B, *C);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        if (success)
        {
            LOG_DEBUG("MatMul kernel executed successfully in " + std::to_string(execution_time) + " ms");
        }
        else
        {
            LOG_ERROR("MatMul kernel execution failed");
        }

        return success;
    }

    // Legacy interface for compatibility with old Tensor struct
    bool MatMulKernel::execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                               std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        // Convert legacy Tensor structs to SimpleTensor wrappers
        std::vector<std::shared_ptr<TensorBase>> tensor_inputs;
        std::vector<std::shared_ptr<TensorBase>> tensor_outputs;

        for (auto &input : inputs)
        {
            if (input)
            {
                // Create SimpleTensor wrapper that shares data with legacy Tensor
                auto simple_tensor = std::make_shared<SimpleTensor>(input->shape, input->data);
                tensor_inputs.push_back(simple_tensor);
            }
        }

        for (auto &output : outputs)
        {
            if (output)
            {
                // Create SimpleTensor wrapper for output
                auto simple_tensor = std::make_shared<SimpleTensor>(output->shape);
                tensor_outputs.push_back(simple_tensor);
            }
        }

        bool result = execute(tensor_inputs, tensor_outputs);

        // Copy results back to legacy tensors
        for (size_t i = 0; i < outputs.size() && i < tensor_outputs.size(); ++i)
        {
            if (outputs[i] && tensor_outputs[i])
            {
                auto simple_output = std::dynamic_pointer_cast<SimpleTensor>(tensor_outputs[i]);
                if (simple_output)
                {
                    outputs[i]->data = simple_output->get_data();
                }
            }
        }

        return result;
    }

    bool MatMulKernel::validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                const std::vector<std::shared_ptr<Tensor>> &outputs) const
    {
        // Check input count
        if (inputs.size() != 2)
        {
            LOG_ERROR("MatMul kernel requires exactly 2 inputs (A, B), got " + std::to_string(inputs.size()));
            return false;
        }

        // Check output count
        if (outputs.size() != 1)
        {
            LOG_ERROR("MatMul kernel requires exactly 1 output (C), got " + std::to_string(outputs.size()));
            return false;
        }

        auto A = inputs[0];
        auto B = inputs[1];
        auto C = outputs[0];

        // Check tensor rank (must be 2D matrices)
        if (A->shape.size() != 2 || B->shape.size() != 2 || C->shape.size() != 2)
        {
            LOG_ERROR("MatMul kernel requires 2D tensors (matrices)");
            return false;
        }

        // Check matrix dimensions for compatibility (A: m×k, B: k×n, C: m×n)
        int m = A->shape[0];
        int k_a = A->shape[1];
        int k_b = B->shape[0];
        int n = B->shape[1];
        int m_c = C->shape[0];
        int n_c = C->shape[1];

        if (k_a != k_b)
        {
            LOG_ERROR("MatMul dimension mismatch: A cols (" + std::to_string(k_a) +
                      ") != B rows (" + std::to_string(k_b) + ")");
            return false;
        }

        if (m != m_c || n != n_c)
        {
            LOG_ERROR("MatMul dimension mismatch: C(" + std::to_string(m_c) + "×" + std::to_string(n_c) +
                      ") should be (" + std::to_string(m) + "×" + std::to_string(n) + ")");
            return false;
        }

        return true;
    }

    bool MatMulKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Check input count
        if (inputs.size() != 2)
        {
            LOG_ERROR("MatMul kernel requires exactly 2 inputs (A, B), got " + std::to_string(inputs.size()));
            return false;
        }

        // Check output count
        if (outputs.size() != 1)
        {
            LOG_ERROR("MatMul kernel requires exactly 1 output (C), got " + std::to_string(outputs.size()));
            return false;
        }

        auto A = inputs[0];
        auto B = inputs[1];
        auto C = outputs[0];

        if (!A || !B || !C)
        {
            LOG_ERROR("MatMul kernel received null tensor pointers");
            return false;
        }

        // Check that all tensors are 2D matrices
        if (!A->is_matrix() || !B->is_matrix() || !C->is_matrix())
        {
            LOG_ERROR("MatMul kernel requires 2D matrices");
            return false;
        }

        // Check matrix multiplication compatibility: A[m,k] × B[k,n] = C[m,n]
        const auto &shape_A = A->shape();
        const auto &shape_B = B->shape();
        const auto &shape_C = C->shape();

        int m = shape_A[0];
        int k_a = shape_A[1];
        int k_b = shape_B[0];
        int n = shape_B[1];
        int m_c = shape_C[0];
        int n_c = shape_C[1];

        if (k_a != k_b)
        {
            LOG_ERROR("MatMul dimension mismatch: A cols (" + std::to_string(k_a) +
                      ") != B rows (" + std::to_string(k_b) + ")");
            return false;
        }

        if (m != m_c || n != n_c)
        {
            LOG_ERROR("MatMul dimension mismatch: C(" + std::to_string(m_c) + "×" + std::to_string(n_c) +
                      ") should be (" + std::to_string(m) + "×" + std::to_string(n) + ")");
            return false;
        }

        return true;
    }

    bool MatMulKernel::canUseZeroCopyCOSMA(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                           const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            return false;
        }

        // Check if all tensors are COSMA tensors
        auto cosma_A = std::dynamic_pointer_cast<COSMATensor>(inputs[0]);
        auto cosma_B = std::dynamic_pointer_cast<COSMATensor>(inputs[1]);
        auto cosma_C = std::dynamic_pointer_cast<COSMATensor>(outputs[0]);

        if (!cosma_A || !cosma_B || !cosma_C)
        {
            return false;
        }

        // Verify all tensors are COSMA-compatible
        return cosma_A->is_cosma_compatible() &&
               cosma_B->is_cosma_compatible() &&
               cosma_C->is_cosma_compatible();
    }

    bool MatMulKernel::executeCOSMANative(cosma::CosmaMatrix<float> &A,
                                          cosma::CosmaMatrix<float> &B,
                                          cosma::CosmaMatrix<float> &C,
                                          const cosma::Strategy &strategy)
    {
        try
        {
            // Get MPI context
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            if (rank == 0)
            {
                LOG_INFO("COSMA Native MatMul: " + std::to_string(A.m()) + "×" + std::to_string(A.n()) +
                         " × " + std::to_string(B.m()) + "×" + std::to_string(B.n()) +
                         " → " + std::to_string(C.m()) + "×" + std::to_string(C.n()) +
                         " with " + std::to_string(size) + " processes (zero-copy)");
            }

            // Perform COSMA multiplication: C = A × B (alpha=1.0, beta=0.0)
            // This is the optimal zero-copy path!
            cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, 1.0f, 0.0f);

            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("COSMA native execution failed: " + std::string(e.what()));
            return false;
        }
    }

    // Legacy executeCOSMA method for SimpleTensor compatibility
    bool MatMulKernel::executeCOSMA(const TensorBase &A, const TensorBase &B, TensorBase &C)
    {
        try
        {
            // Get MPI context
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            // Matrix dimensions using TensorBase interface
            const auto &shape_A = A.shape();
            const auto &shape_B = B.shape();
            int m = shape_A[0];
            int n = shape_B[1];
            int k = shape_A[1];

            if (rank == 0)
            {
                LOG_INFO("COSMA MatMul (legacy): " + std::to_string(m) + "×" + std::to_string(k) +
                         " × " + std::to_string(k) + "×" + std::to_string(n) +
                         " → " + std::to_string(m) + "×" + std::to_string(n) +
                         " with " + std::to_string(size) + " processes (with copy)");
            }

            // Create COSMA strategy
            cosma::Strategy strategy(m, n, k, size);

            // Create COSMA matrices (using float instead of double)
            cosma::CosmaMatrix<float> cosma_A('A', strategy, rank);
            cosma::CosmaMatrix<float> cosma_B('B', strategy, rank);
            cosma::CosmaMatrix<float> cosma_C('C', strategy, rank);

            // Get local matrix pointers
            auto local_A = cosma_A.matrix_pointer();
            auto local_B = cosma_B.matrix_pointer();
            auto local_C = cosma_C.matrix_pointer();

            // Copy input data to COSMA matrices (local portions)
            // Note: In a real distributed implementation, data would be properly distributed
            // For now, we handle the case where data might already be distributed
            size_t local_size_A = cosma_A.matrix_size();
            size_t local_size_B = cosma_B.matrix_size();

            if (local_size_A <= A.size())
            {
                std::copy(A.data(), A.data() + local_size_A, local_A);
            }
            else
            {
                LOG_WARN("Local A matrix size (" + std::to_string(local_size_A) +
                         ") exceeds input data size (" + std::to_string(A.size()) + ")");
                // Fill with available data and pad with zeros
                std::copy(A.data(), A.data() + A.size(), local_A);
                std::fill(local_A + A.size(), local_A + local_size_A, 0.0f);
            }

            if (local_size_B <= B.size())
            {
                std::copy(B.data(), B.data() + local_size_B, local_B);
            }
            else
            {
                LOG_WARN("Local B matrix size (" + std::to_string(local_size_B) +
                         ") exceeds input data size (" + std::to_string(B.size()) + ")");
                // Fill with available data and pad with zeros
                std::copy(B.data(), B.data() + B.size(), local_B);
                std::fill(local_B + B.size(), local_B + local_size_B, 0.0f);
            }

            // Perform COSMA multiplication: C = A × B (alpha=1.0, beta=0.0)
            cosma::multiply(cosma_A, cosma_B, cosma_C, strategy, MPI_COMM_WORLD, 1.0f, 0.0f);

            // Copy result back to output tensor
            size_t local_size_C = cosma_C.matrix_size();
            if (C.size() >= local_size_C)
            {
                std::copy(local_C, local_C + local_size_C, C.data());
            }
            else
            {
                LOG_WARN("Output tensor size (" + std::to_string(C.size()) +
                         ") is smaller than local result size (" + std::to_string(local_size_C) + ")");
                // Copy what we can
                std::copy(local_C, local_C + C.size(), C.data());
            }

            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("COSMA execution failed: " + std::string(e.what()));
            return false;
        }
    }

    void MatMulKernel::initializeCOSMAContext()
    {
        // Initialize COSMA context if needed
        // This could include setting up memory pools, communication patterns, etc.
        LOG_DEBUG("Initializing COSMA context for MatMul kernel with strategy: " + strategy_);
    }

} // namespace llaminar
