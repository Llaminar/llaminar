/**
 * @file CUDAFloatingPointGemmWorkspaceContract.h
 * @brief Exact named CUDA BLAS/wrapper workspace shared by runtime and admission.
 *
 * The BLAS adapter and floating projection wrapper own distinct device-pointer
 * triplets. Both coexist in a captured graph; counting only the outer triplet
 * understates physical storage. This device-free contract composes their names
 * before alignment, along with the shared library scratch and mapped redirect.
 * Explicit FP16RNE operands additionally own exactly rows*k half elements;
 * a sequential fused group casts once and shares that bank on its exact stream.
 * Their fixed-K prefill policy may partition narrow groups; one bounded bank
 * of partition chains covers every group at that row envelope.
 */
#pragma once
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/FloatingActivationOperandPrecision.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include "kernels/common/FixedKProjectionContract.h"
#include <limits>
#include <cstdint>
#include <stdexcept>

namespace llaminar2::cuda::floating_gemm_workspace
{
    inline constexpr const char *kBatchedSameAAArray = "cublas_batched_same_a_a_ptrs";
    inline constexpr const char *kBatchedSameABArray = "cublas_batched_same_a_b_ptrs";
    inline constexpr const char *kBatchedSameACArray = "cublas_batched_same_a_c_ptrs";

    /** @return Library scratch and the adapter's own three device-pointer arrays. */
    inline WorkspaceRequirements blasRequirements()
    {
        WorkspaceRequirements result;
        for (auto name : {kBatchedSameAAArray, kBatchedSameABArray, kBatchedSameACArray})
            result.buffers.push_back({name, floating_gemm_abi::kMaxBatchedProjections * sizeof(void *), 256, true});
        result.buffers.push_back({floating_gemm_abi::kCudaBlasMatmulWorkspace,
            floating_gemm_abi::kBlasMatmulWorkspaceBytes, 256, true});
        return result;
    }

    /**
     * @return Complete wrapper plus BLAS scratch for FP32, FP16 or BF16 weights.
     * @param rows Active/family row envelope; zero omits only mapped output scratch.
     * @param columns Output width of one projection; zero omits mapped scratch.
     * @throws std::overflow_error if the mapped-output extent cannot be represented.
     */
    inline WorkspaceRequirements projectionRequirements(size_t rows, size_t columns)
    {
        auto result = blasRequirements();
        for (auto name : {GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS,
                         GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS, GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS})
            result.buffers.push_back({name, floating_gemm_abi::kMaxBatchedProjections * sizeof(void *), 256, true});
        if (rows && columns)
        {
            constexpr size_t bytes_per_column = floating_gemm_abi::kMaxBatchedProjections * sizeof(float);
            if (rows > std::numeric_limits<size_t>::max() / bytes_per_column / columns)
                throw std::overflow_error("CUDA floating projection redirect extent overflow");
            result.buffers.push_back({GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT,
                rows * columns * bytes_per_column, 256, true});
        }
        return result;
    }

    /**
     * @brief Add explicit-operand scratch: the fixed-K chain bank and, for half, the half-A bank.
     * @param rows Complete retained row envelope, not the allocation capacity.
     * @param columns Output width used by the existing redirect declaration.
     * @param k Prepared activation reduction width.
     * @param operand Immutable arithmetic; Native adds nothing, FP32 the chain bank,
     *        FP16RNE the chain bank and the half-A bank.
     * @return Existing requirements plus the bounded chain bank and, for half,
     *         exactly rows*k*sizeof(uint16_t).
     * @throws std::invalid_argument Half operand geometry is not positive.
     * @throws std::overflow_error Named half bank bytes cannot be represented.
     */
    inline WorkspaceRequirements projectionRequirements(size_t rows, size_t columns,
        size_t k, FloatingActivationOperandPrecision operand)
    {
        auto result = projectionRequirements(rows, columns);
        if (operand == FloatingActivationOperandPrecision::Native)
            return result;
        if (operand == FloatingActivationOperandPrecision::FP16RNE && (!rows || !k))
            throw std::invalid_argument("CUDA FP16 operand workspace requires rows and k");
        if (!rows)
            return result;  // a rowless strict-FP32 envelope launches nothing
        if (rows > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("CUDA fixed-K partial row envelope overflow");
        // Ordinary explicit-operand prefill may partition narrow groups; one
        // bounded chain bank covers every group at this row envelope.
        result.buffers.push_back({GemmWorkspaceBuffers::FIXED_K_PARTIALS,
            static_cast<size_t>(fixed_k_projection::partialBound(static_cast<int>(rows))), 256, true});
        if (operand == FloatingActivationOperandPrecision::FP16RNE)
        {
            if (rows > std::numeric_limits<size_t>::max() / sizeof(std::uint16_t) / k)
                throw std::overflow_error("CUDA FP16 activation operand extent overflow");
            result.buffers.push_back({GemmWorkspaceBuffers::FULL_A_FP16,
                rows * k * sizeof(std::uint16_t), 256, true});
        }
        return result;
    }
}
