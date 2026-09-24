/**
 * @file ROCmFloatingPointGemmWorkspaceContract.h
 * @brief One named hipBLAS/wrapper scratch declaration for runtime and admission.
 *
 * Floating projections retain library scratch and three captured pointer arrays.
 * Metadata-only planning must declare those same names before preparing weights,
 * without constructing a library handle or independently reconstructing a byte sum.
 */
#pragma once
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"

namespace llaminar2::rocm::floating_gemm_workspace
{
    /** @return The one mutually exclusive hipBLAS/hipBLASLt scratch region. */
    inline WorkspaceRequirements blasRequirements()
    {
        WorkspaceRequirements result;
        result.buffers.push_back({floating_gemm_abi::kROCmBlasMatmulWorkspace,
            floating_gemm_abi::kBlasMatmulWorkspaceBytes, 256, true});
        return result;
    }

    /** @return Complete, geometry-independent FP32/FP16/BF16 projection scratch. */
    inline WorkspaceRequirements projectionRequirements()
    {
        auto result = blasRequirements();
        for (auto name : {GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS,
                         GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS,
                         GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS})
            result.buffers.push_back({name,
                floating_gemm_abi::kMaxBatchedProjections * sizeof(void *), 256, true});
        return result;
    }
}
