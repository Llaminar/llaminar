/**
 * @file ROCmRetainedM1HCFFNBinding.h
 * @brief Resolve the retained ROCm single-row arithmetic envelope from original prepared owners.
 *
 * This cold binding consumes the original phase records after their existing
 * storage and alias validation. It borrows original device addresses only.
 * The caller retains the original graph, workspace and source leases.
 */
#pragma once

#include <stdexcept>
#include "execution/moe/MoERuntimeTable.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"

namespace llaminar2::rocm {

/**
 * @brief Bind the one original count and the already prepared shared weights.
 *
 * The device publisher must independently authenticate the reachable producer
 * and its original local-ready membership before launch admission. This
 * binding verifies immutable arithmetic inputs and never reads the bank or
 * the contents of the controller's count.
 */
class ROCmRetainedM1HCFFNBinding final {
public:
    /**
     * @brief Resolve the exact arithmetic specialized in the retained body.
     * @param full Original complete prepared phase arguments.
     * @param shared Original by-value prepared shared descriptors from the stage owner.
     * @param masks Original authenticated reachable source and execution masks.
     * @return Checked borrowed binding to the original arguments and source.
     * @throws std::invalid_argument If the declared shape, row or source policy does not match.
     */
    static ROCmRetainedM1HCFFNBinding resolve(const FullNativeHCFFNArgs& full,
        const DeviceMoEExpertDescriptor& shared, ResidentMoEFFNPolicyMasks masks)
    {
        const auto& suffix=full.suffix;
        const auto& ffn=suffix.ffn;
        const auto& hc=full.prefix.hc;
        if(ffn.hidden_columns!=2560 || ffn.routed_columns!=256 || ffn.shared_columns!=192 ||
           ffn.experts!=512 || ffn.top_k!=10 || hc.norm.hidden!=ffn.hidden_columns ||
           hc.mixed!=ffn.hidden || hc.publication.columns!=320)
            throw std::invalid_argument("Retained ROCm HC/FFN arithmetic requires H2560/R320/F256/S192/E512/topK10");

        const auto rows=ffn.rows.captureIdentity();
        const auto counted_one=DeviceRowRange::CaptureIdentity{rows.count,1,0,1,1};
        if(!rows.count || rows!=counted_one || hc.norm.extent.rows!=1 ||
           hc.norm.extent.liveRows!=rows.count || hc.up.rows.captureIdentity()!=rows ||
           hc.publication.rows.captureIdentity()!=rows)
            throw std::invalid_argument("Retained ROCm HC/FFN phases require the same original counted M1 row");

        if(hc.injection.weights || hc.injection.output)
            throw std::invalid_argument("Retained ROCm HC/FFN admission has no independent FP32 injection");
        if(hc.up.quantizer!=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal)
            throw std::invalid_argument("Retained ROCm HC up requires its canonical finite-half primary quantizer");

        if(shared.weight_format!=DeviceMoEWeightFormat::NativeQuantised ||
           !nativeQ8(hc.down_weights,320,10240) || !nativeQ8(hc.up.weights,10240,320) ||
           !nativeQ8(shared.gate,192,2560) || !nativeQ8(shared.up,192,2560) ||
           !nativeQ8(shared.down,2560,192))
            throw std::invalid_argument("Retained ROCm HC/FFN requires original native Q8 HC and shared sources");

        constexpr std::uint32_t q4_k=1u<<5, q5_1=1u<<7, q8_0=1u<<19;
        if(masks.routed_gate!=q4_k || masks.routed_down!=q5_1 ||
           masks.shared_gate!=q8_0 || masks.shared_down!=q8_0 ||
           masks.execution.routed_gate!=q4_k || masks.execution.routed_down!=q5_1 ||
           masks.execution.shared_gate!=q8_0 || masks.execution.shared_down!=q8_0)
            throw std::invalid_argument("Retained ROCm HC/FFN requires its exact original and reachable native role envelopes");

        return ROCmRetainedM1HCFFNBinding(rows.count);
    }

    /** @return Original M1 execution owner, borrowed for the prepared native graph. */
    const std::int32_t* countOwner() const noexcept {return count_;}

private:
    /** @brief Preserve the original count after complete arithmetic validation. */
    explicit ROCmRetainedM1HCFFNBinding(const std::int32_t* count) : count_(count) {}

    /** @brief Validate exact native and source identities of canonical Q8 operands. */
    static bool nativeQ8(const DeviceNativeQuantisedMatrixDesc& source, int n, int k) noexcept
    {
        return source.valid() && source.n==n && source.k==k && source.blocks_per_row==unsigned(k/32) &&
            source.codebook_id==19 && source.source_identity_present==1 && source.source_codebook_id==19 &&
            !source.source_is_superblock;
    }

    const std::int32_t* count_; ///< Original phase owner, retained by the caller.
};

}
