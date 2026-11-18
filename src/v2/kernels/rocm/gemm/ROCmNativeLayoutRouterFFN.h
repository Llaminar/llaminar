/**
 * @file ROCmNativeLayoutRouterFFN.h
 * @brief Prepared-layout translation of the pinned T1 POST router/expert dataflow.
 *
 * Existing canonical NativeQuantised descriptors, source policy, named scratch
 * and router publication remain the sole owners. The fixed1024thread program
 * maps contiguous N columns to lanes, rather than changing the weight layout.
 */
#pragma once
#include "ROCmResidentMoEFFN.h"
#include "kernels/common/MoEDecodeDescriptorPublication.h"
#include "kernels/common/NativeLayoutBarrierLayout.h"
namespace llaminar2::rocm {
/**
 * @brief Immutable coefficient-window geometry of the complete HC/FFN program.
 * @details Cold preparation intersects live source geometry with cached device
 * LDS capacity. The actual full-function admission verifies that storage. These
 * scalar dimensions travel in kernargs; execution never queries device metadata
 * or changes its captured arena. Standalone suffix entries do not consume them.
 */
struct FullNativeHCFFNSuffixPanel {
    unsigned rows=0; ///< Consecutive live column capacity of one source window.
    unsigned blocks=0; ///< Bounded K32 capacity, preserving each original K/P fold.
    bool operator==(const FullNativeHCFFNSuffixPanel&) const = default;
};
/** @brief Borrow one original participant-local router and complete local FFN. */
struct NativeLayoutRouterFFNArgs {
    std::uint32_t* barrier_state=nullptr; ///< NativeLayoutBarrierLayout::words in the retained per-instance serial-family slot.
    ResidentMoEFFNArgs ffn; ///< Existing prepared weights, source policies, scratch and public planes.
    const void* router_weights=nullptr; ///< Original prepared FP16 router matrix.
    float* router_logits=nullptr; ///< Original public logits/probabilities storage.
    float* router_indices=nullptr; ///< Original complete selected-ID output, when declared.
    float* router_weights_output=nullptr; ///< Original complete selected-weight output, when declared.
    MoEDecodeDescriptorPublication router_publication; ///< Existing selected descriptor bank.
    bool normalize_router_weights=false;
    bool write_router_outputs=false;
    FullNativeHCFFNSuffixPanel full_entry_panel; ///< Cold-sealed full-entry layout; prior field offsets remain unchanged.
};
/** @return Exact native function used by admission and captured-node inspection. */
const void* nativeLayoutRouterFFNKernelFunction(ResidentMoEFFNDecoderProgram program);
/**
 * @brief Resolve original source policies and admit one fixed1024thread native program.
 * @param device Original participant. @param args Original prepared owner bindings.
 * @param generation Existing prepared-owner lifecycle generation.
 * @return Actual function/resource/resident-grid admission; no fallback is selected.
 */
PreparedMoEFFNProgram prepareNativeLayoutRouterFFN(DeviceId device,
    NativeLayoutRouterFFNArgs& args,ResidentMoEFFNPolicyMasks masks,std::uint64_t generation);
/** @brief Submit one occupancy-admitted ordinary entry on the sealed owner stream. */
void launchNativeLayoutRouterFFN(const PreparedMoEFFNProgram&,
    const NativeLayoutRouterFFNArgs&,void* stream,std::uint64_t generation);
}
