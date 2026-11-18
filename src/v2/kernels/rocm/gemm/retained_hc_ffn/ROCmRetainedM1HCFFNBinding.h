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
#include "ROCmRetainedM1HCFFNDevice.h"

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
        const int routed=ffn.routed_columns;
        const int shared_width=ffn.shared_columns;
        // Routed cuts rotate independently of the dense/shared assignment, so
        // all four TP8 F/S pairs can occur across the original layer sequence.
        const bool original_shard=(routed==640 && shared_width==640) ||
            ((routed==256 || routed==384) && shared_width==320) ||
            ((routed==128 || routed==256) && shared_width==160) ||
            ((routed==64 || routed==96) && (shared_width==64 || shared_width==96));
        const bool matched_reference_shard=routed==256 && shared_width==192;
        if(ffn.hidden_columns!=2560 || (!original_shard && !matched_reference_shard) ||
           ffn.experts!=512 || ffn.top_k!=retainedM1SelectedExperts || hc.norm.hidden!=ffn.hidden_columns ||
           hc.mixed!=ffn.hidden || hc.publication.columns!=320)
            throw std::invalid_argument("Retained ROCm HC/FFN requires H2560/R320/E512/topK10 and a declared native TP1/2/4/8 or F256/S192 reference shard");

        const auto* count=validateRowAuthority(ffn.rows,hc);

        if(hc.injection.weights || hc.injection.output)
            throw std::invalid_argument("Retained ROCm HC/FFN admission has no independent FP32 injection");
        if(hc.up.quantizer!=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal)
            throw std::invalid_argument("Retained ROCm HC up requires its canonical finite-half primary quantizer");

        if(shared.weight_format!=DeviceMoEWeightFormat::NativeQuantised ||
           !nativeQ8(hc.down_weights,320,10240) || !nativeQ8(hc.up.weights,10240,320) ||
           !nativeQ8(shared.gate,ffn.shared_columns,ffn.hidden_columns) ||
           !nativeQ8(shared.up,ffn.shared_columns,ffn.hidden_columns) ||
           !nativeQ8(shared.down,ffn.hidden_columns,ffn.shared_columns))
            throw std::invalid_argument("Retained ROCm HC/FFN requires original native Q8 HC and shared sources");

        constexpr std::uint32_t q8_0=1u<<19;
        if(!retainedM1RoutedFormat(masks.execution) ||
           masks.routed_gate!=masks.execution.routed_gate || masks.routed_down!=masks.execution.routed_down ||
           masks.shared_gate!=q8_0 || masks.shared_down!=q8_0 ||
           masks.execution.shared_gate!=q8_0 || masks.execution.shared_down!=q8_0)
            throw std::invalid_argument("Retained ROCm HC/FFN requires its exact original and reachable native role envelopes");

        return ROCmRetainedM1HCFFNBinding(count);
    }

    /**
     * @brief Authenticate one identical ordinary row contract across the original phases.
     * @param suffix Original FFN row range.
     * @param hc Exact source-owned HC argument views.
     * @return The original counted row owner, or null for the original fully-active row.
     * @throws std::invalid_argument For a foreign owner, expansion or logical/physical extent.
     */
    static const std::int32_t* validateRowAuthority(DeviceRowRange suffix,const PersistentHCReadArgs& hc)
    {
        const auto rows=suffix.captureIdentity();
        const auto expected=DeviceRowRange::CaptureIdentity{rows.count,1,0,1,1};
        if(rows!=expected || hc.norm.extent.rows!=1 || hc.norm.extent.liveRows!=rows.count ||
           hc.up.rows.captureIdentity()!=rows || hc.publication.rows.captureIdentity()!=rows)
            throw std::invalid_argument("Retained ROCm HC/FFN phases require the same original M1 row authority");
        return rows.count;
    }

    /** @return Original counted row owner, or null for the fully-active contract. */
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
