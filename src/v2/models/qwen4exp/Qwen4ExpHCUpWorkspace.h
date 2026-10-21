/**
 * @file Qwen4ExpHCUpWorkspace.h
 * @brief Exact arithmetic-specific HC-up workspace contribution for all backends.
 *
 * Source semantics select the candidate; raw shapes never do. The canonical
 * requirement factories own names, capacities and alignment. Repeated serial
 * HC layers merge those names instead of adding per-layer scratch reserves.
 */
#pragma once
#include "backends/DeviceId.h"
#include "loaders/WeightIdentity.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "Qwen4ExpHCUpArithmetic.h"
#include "kernels/cpu/gemm/CPUQ8ResidualCorrectionWorkspace.h"
#include <string_view>
namespace llaminar2 {
/** @brief Model-specific selection of a generic native projection contract. */
struct Qwen4ExpHCUpWorkspace final
{
    /**
     * @brief Select only authenticated block, terminal or predictor HC-up roles.
     * @param policy Frozen setup arithmetic; invalid enum values are rejected.
     * @param flash_next Whether the profile contains authenticated QFN geometry.
     * @param role Canonical source role produced by inferWeightRole.
     * @param name Exact original GGUF tensor name.
     * @return True only for the explicit correction's HC-up consumers.
     * @throws std::invalid_argument For an unknown arithmetic policy.
     */
    static bool selected(QuantizedActivationOperandPrecision policy,bool flash_next,
        WeightRole role,std::string_view name)
    {
        if(policy==QuantizedActivationOperandPrecision::SingleQ8)return false;
        if(policy!=QuantizedActivationOperandPrecision::TwoTermQ8)
            throw std::invalid_argument("HC workspace has an unknown activation arithmetic policy");
        return flash_next && isHyperConnectionUpProjection(role,name);
    }
    /**
     * @brief Contribute the actual two-term stage contract without preparing weights.
     * @param device Exact participant backend; no device discovery is performed.
     * @param policy Frozen arithmetic selection.
     * @param flash_next Authenticated model-family presence.
     * @param role Canonical source role.
     * @param name Original native tensor name, including its source block.
     * @param native_format Original source format; only quantized sources own Q8 banks.
     * @param m Admitted physical rows for this precise graph role.
     * @param n Participant-local output columns.
     * @param k Participant-local reduction width.
     * @return Empty ordinary contribution or the exact native named bank set.
     * @throws std::exception For incompatible source, geometry or device policy.
     */
    static WorkspaceRequirements requirements(DeviceId device,
        QuantizedActivationOperandPrecision policy,bool flash_next,WeightRole role,
        std::string_view name,std::string_view native_format,int m,int n,int k)
    {
        if(!selected(policy,flash_next,role,name))return {};
        if(resolveQwen4ExpHCUpArithmetic(policy,native_format)==Qwen4ExpHCUpArithmetic::FloatingNative)
            return {};
        (void)requireQ8ResidualSourceFormat(native_format,m,n,k);
        if(device.is_cpu())return cpuQ8ResidualCorrectionWorkspaceRequirements(m,n,k);
        if(device.is_cuda()||device.is_rocm())return gpuQ8ResidualCorrectionWorkspaceRequirements(m,n,k);
        throw std::invalid_argument("Two-term HC-up workspace requires a supported physical participant");
    }
};
} // namespace llaminar2
