/**
 * @file ROCmMoEGroupedPopulationProjection.h
 * @brief Shared row ownership for population-selected Qwen projection consumers.
 *
 * The directory and consumers borrow one immutable device publication.
 * Global native selection uses sixteen rows even for concentrated counts below
 * sixteen. Ordered Qwen gate/up uses twelve rows: the only legacy sixteen-row
 * exception (all positive counts equal sixteen) always selects native arithmetic.
 * Compact geometry has a different ordered width and is qualified separately.
 */
#pragma once
#include "ROCmMoEGroupedWmma.h"
namespace llaminar2::rocm
{
/** @brief Frozen consumer control; family suppression is accepted only by test mutation of a retained node. */
struct MoEGroupedPopulationControl
{
    MoEGroupedPopulationBinding binding;
    MoEGroupedPopulationFamily omitted = MoEGroupedPopulationFamily::Invalid; ///< Invalid means no suppressed family.
};
static_assert(std::is_trivially_copyable_v<MoEGroupedPopulationControl>);
/** @brief Borrowed writable publication storage; the retained consumer borrows the same addresses. */
struct MoEGroupedPopulationWorkspace
{
    uint32_t *selection;
    uint32_t *directory;
    std::size_t directory_words; ///< Available words supplied by the canonical workspace contract.
};
/** @brief Authenticate immutable hardware resources shared by both grouped native owners.
 * @param device Admitted visible HIP ordinal. @return True for native gfx1100 resources. */
bool nativeGroupedDevice(int device);
/** @brief Publish this shard's private IQ2_S lookup table during backend initialization.
 * @param device Target device. @param iq2s Host table with 1024 uint64 entries.
 * @return True after publication; never called from capture or submission. */
bool initializeMoEGroupedPopulationGrids(int device,const void *iq2s);
/** @brief Submit the exact admitted Qwen role, with an exact role-specific consumer roster.
 * @param codebooks Actual physical mask. @param args Exact original arithmetic and workspace bindings.
 * @param role Explicit projection role. @param workspace Writable canonical producer storage.
 * @param control Retained device publication; production requires no suppressed family.
 * @param stream Exact non-default producer stream. @param submitted Optional successful-submission receipt.
 * @return Whether the complete role was accepted; no alternate launch is attempted. */
bool submitMoEGroupedPopulation(MoEPrefillProjectionRole role, MoEGroupedPopulationWorkspace workspace,
    uint32_t codebooks,const MoEGroupedWmmaArguments &args,
    MoEGroupedPopulationControl control,void *stream,MoEGroupedPrefillRoleSubmission *submitted = nullptr);
}
