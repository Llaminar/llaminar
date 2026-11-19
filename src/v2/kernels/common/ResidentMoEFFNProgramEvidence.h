/**
 * @file ResidentMoEFFNProgramEvidence.h
 * @brief Lightweight exact native-entry observations for the prepared local FFN owner.
 *
 * This header carries no runtime/table/tensor dependency. These passive
 * observations describe the sealed backend program; they do not authorize
 * allocation, residency, submission, or a change of execution mechanism.
 */
#pragma once
#include "NativeCooperativeKernel.h"
#include <array>
#include <span>

namespace llaminar2
{
/** @brief Exact submission mechanism selected explicitly before graph recording. */
enum class ResidentMoEFFNLaunchKind { Cooperative, OrdinaryPhases, OrdinaryResident };

/** @brief Observed native entry and its exact captured one-dimensional launch. */
struct ResidentMoEFFNEntryEvidence
{
    const void* function=nullptr; ///< Loaded native entry, never a symbol-name approximation.
    unsigned grid=0,threads=0; ///< Actual launch extents, not allocation or residency capacity.
    std::size_t dynamic_shared_bytes=0; ///< Exact per-launch dynamic shared memory.
    NativeCooperativeKernelResources resources; ///< Compiler/runtime resource observation; implies no cooperative admission.
};

/**
 * @brief Read-only native program observations, not allocation or residency authority.
 * @details The backend prepared owner seals these observations together with its
 * native submission object. Ordinary phases do not fabricate a cooperative
 * admission; both entries execute in order on the same original non-null stream.
 */
struct ResidentMoEFFNProgramEvidence
{
    ResidentMoEFFNLaunchKind kind=ResidentMoEFFNLaunchKind::Cooperative;
    DeviceId device;
    std::uint64_t generation=0;
    std::array<ResidentMoEFFNEntryEvidence,2> entries{};
    /** @return The exact ordered native entries retained by the prepared owner. */
    std::span<const ResidentMoEFFNEntryEvidence> nativeEntries() const noexcept
    {return {entries.data(),kind!=ResidentMoEFFNLaunchKind::OrdinaryPhases?std::size_t(1):std::size_t(2)};}
};

}
