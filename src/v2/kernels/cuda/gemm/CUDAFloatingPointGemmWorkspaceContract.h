/**
 * @file CUDAFloatingPointGemmWorkspaceContract.h
 * @brief Exact named CUDA BLAS/wrapper workspace shared by runtime and admission.
 *
 * The BLAS adapter and floating projection wrapper own distinct device-pointer
 * triplets. Both coexist in a captured graph; counting only the outer triplet
 * understates physical storage. This device-free contract composes their names
 * before alignment, along with the shared library scratch and mapped redirect.
 */
#pragma once
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include <limits>
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
}
