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
#include "kernels/FloatingActivationOperandPrecision.h"
#include <cstdint>
#include <limits>
#include <stdexcept>

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
    /**
     * @brief Declare the optional physical half-A bank through the existing BOM.
     * @param m Maximum ordinary rows for the retained graph.
     * @param k Positive input width; weight/output banks retain their own owners.
     * @param operand_precision Immutable prepared-engine activation policy.
     * @return Existing scratch plus exactly m*k half elements when requested.
     * @throws std::invalid_argument For invalid or overflowing half geometry.
     *
     * A fused GDN group publishes this bank once and submits all consumers on
     * the same stream. Mutually exclusive graph stages may share its region;
     * concurrent stage plans must retain their existing exclusive arena owners.
     */
    inline WorkspaceRequirements projectionRequirements(int m, int k,
        FloatingActivationOperandPrecision operand_precision)
    {
        auto result = projectionRequirements();
        if (operand_precision != FloatingActivationOperandPrecision::FP16RNE)
            return result;
        if (m <= 0 || k <= 0 || static_cast<std::size_t>(m) >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) /
                static_cast<std::size_t>(k))
            throw std::invalid_argument("Half-operand projection scratch has invalid geometry");
        result.buffers.push_back({GemmWorkspaceBuffers::FULL_A_FP16,
            static_cast<std::size_t>(m) * static_cast<std::size_t>(k) * sizeof(std::uint16_t),
            256, true});
        return result;
    }

}
