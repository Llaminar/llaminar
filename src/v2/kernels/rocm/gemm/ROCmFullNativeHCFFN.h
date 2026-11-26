/**
 * @file ROCmFullNativeHCFFN.h
 * @brief One original HC-write/read and router/FFN transaction on an admitted grid.
 *
 * The existing HC and FFN owners retain all tensor, source, workspace and row
 * leases. These records borrow those original bytes; they allocate nothing and
 * introduce no host copy of live device state. Preparation authenticates the
 * complete composed function, not either of its independently callable phases.
 */
#pragma once
#include "ROCmPersistentHCRead.h"
#include "ROCmNativeLayoutRouterFFN.h"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
#include "kernels/common/FullNativeHCFFNOutputDemand.h"

namespace llaminar2::rocm {
/** @brief Explicit complete-local native program selected by the original graph owner. */
enum class FullNativeHCFFNImplementation : std::uint8_t { Canonical };
/** @return Exact barrier slot byte extent for the canonical full HC/FFN implementation. */
inline constexpr std::size_t fullNativeHCFFNSlotBytes(FullNativeHCFFNImplementation) noexcept
{
    return FullNativeHCFFNBarrierLayout::bytes;
}
/**
 * @brief Original HC prefix and its distinct instance-owned barrier subregion.
 * @details The backend owner supplies barrier_state at cold preparation. The
 * graph owner retains the original write/read/injection declarations and their
 * leases. The independent injection output remains a public graph export.
 */
struct FullNativeHCFFNPrefixArgs {
    PersistentHCReadArgs hc; ///< Original norm/down/up/injection arithmetic and bindings.
    PersistentHCWriteInput write; ///< Original previous write; output is exactly hc.norm.input.
    std::uint32_t* barrier_state=nullptr; ///< Start of the complete 56-byte retained bank.
    FullNativeHCFFNOutputDemand output_demand=FullNativeHCFFNOutputDemand::all(); ///< Cold graph/observer demand; never mutable during replay.
    FullNativeHCFFNImplementation implementation=FullNativeHCFFNImplementation::Canonical; ///< Admission and launch identity.
    MoESelectedDescriptorDemand selected_descriptor_demand=MoESelectedDescriptorDemand::ExternalTables; ///< Sealed outside role-table reads.
};

// The selector crosses the CXX/HIP host call boundary with a 32-bit enum.
// Keep it in the existing by-value tail so earlier prefix fields retain offsets.
static_assert(std::is_same_v<std::underlying_type_t<MoESelectedDescriptorDemand>,std::uint32_t>);
static_assert(std::is_standard_layout_v<FullNativeHCFFNPrefixArgs>);
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(FullNativeHCFFNPrefixArgs)==464 && alignof(FullNativeHCFFNPrefixArgs)==8);
static_assert(offsetof(FullNativeHCFFNPrefixArgs,output_demand)==456);
static_assert(offsetof(FullNativeHCFFNPrefixArgs,selected_descriptor_demand)==460);
#endif
/**
 * @brief Cold host binding for the complete participant-local operation.
 * @details The native entry takes suffix FIRST and prefix SECOND as separate
 * by-value arguments. Existing FFN kernarg-offset loads therefore keep their
 * original ABI; this host envelope is never passed as one native argument.
 */
struct FullNativeHCFFNArgs {
    NativeLayoutRouterFFNArgs suffix; ///< Original FFN descriptors/publication; hidden is prefix.hc.mixed.
    FullNativeHCFFNPrefixArgs prefix; ///< Original HC producer; its scratch retires before the suffix.
};
static_assert(std::is_trivially_copyable_v<FullNativeHCFFNPrefixArgs>);
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(FullNativeHCFFNArgs)==992 && offsetof(FullNativeHCFFNArgs,prefix)==528);
#endif
static_assert(std::is_trivially_copyable_v<FullNativeHCFFNArgs>);

/**
 * @param program Complete immutable original physical decoder envelope.
 * @return Actual full-entry function admitted and inspected before capture.
 */
const void* fullNativeHCFFNKernelFunction(ResidentMoEFFNDecoderProgram program);
/**
 * @brief Resolve canonical phase policies and admit the exact full native entry.
 * @param device Original ROCm participant.
 * @param args Original retained phase bindings; receives the shared admitted grid.
 * @param masks Complete source-policy and physical decoder reachability.
 * @param generation Existing nonzero workspace/owner generation.
 * @return One ordinary resident program with the full function's resource evidence.
 * @throws std::invalid_argument For inconsistent geometry, source, row or storage identities.
 */
PreparedMoEFFNProgram prepareFullNativeHCFFN(DeviceId device,FullNativeHCFFNArgs& args,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation);
/**
 * @brief Submit the exact retained full entry on its sealed nondefault stream.
 * @param program Original full-function admission, never standalone-phase evidence.
 * @param args Original frozen complete binding and admitted live-task grid.
 * @param stream Exact prepared participant stream.
 * @param generation Original retained workspace/owner generation.
 */
void launchFullNativeHCFFN(const PreparedMoEFFNProgram& program,const FullNativeHCFFNArgs& args,
    void* stream,std::uint64_t generation);
}
