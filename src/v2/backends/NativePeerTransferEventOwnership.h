/**
 * @file NativePeerTransferEventOwnership.h
 * @brief One original admitted peer-lane completion/timestamp event set.
 *
 * This closed owner owns the actual handles, not a second event registry. The
 * existing original native retirement slot retains conservative opaque PMA
 * commitments after checked handle destruction through their physical lifetime.
 */
#pragma once
#include "NativeAuxiliaryResourceCatalog.h"
#include "NativeContextPreparation.h"
#include <array>
#include <cstddef>
#include <memory>
namespace llaminar2 {
class ExpertTierGpuPeerTransferLane;
class ReadAheadNativeResourceObligation;
namespace test {struct NativePeerTransferEventOwnershipReplayAccess;struct OriginalPeerTransferEventFixtureAccess;}
/** @brief Actual one-ordering/optional-two-timing creator and destruction boundary. */
class NativePeerTransferEventOwnership final {
public:
    /**
     * @brief Declare the original peer-event family's conservative physical obligations.
     * @param host Original CPU allocator belonging to the same world rank as the GPU.
     * @param gpu Original GPU allocator which creates and owns every event in this set.
     * @param timing False owns one Ordering receipt; true additionally owns exactly two Timing events.
     * @return Opaque family ceilings for the canonical PMA, without native creation or a live permit.
     * @throws std::invalid_argument For invalid/nonmatching original CPU/GPU resource identities.
     */
    static PhysicalMemoryPlan bom(PhysicalMemoryResource host,PhysicalMemoryResource gpu,bool timing);
    /**
     * @brief Destroy accepted handles on the exact original worker, then close their native slot.
     * @pre The owning lane or fixture has proved all uses of these handles quiescent.
     * @note Partial creation follows this same path. A failed native destruction is fatal
     * before slot completion; process-runtime backing remains charged to the original PMA.
     */
    ~NativePeerTransferEventOwnership()noexcept;
    NativePeerTransferEventOwnership(const NativePeerTransferEventOwnership&)=delete;
    NativePeerTransferEventOwnership& operator=(const NativePeerTransferEventOwnership&)=delete;
private:
    friend class ExpertTierGpuPeerTransferLane;
    friend struct test::NativePeerTransferEventOwnershipReplayAccess;
    friend struct test::OriginalPeerTransferEventFixtureAccess;
    /**
     * @brief Materialize one immutable original peer-event set during cold setup.
     * @param original Genuine source-prepared permit for the event-owning GPU and its sole PMA.
     * @param timing Exact fixed purpose sequence: Ordering, optionally Timing then Timing.
     * @param instance Existing stable physical edge/projection/lane name, never a new per-transfer ID.
     * @return Complete handle owner; repeated transfers retain this same owner.
     * @throws std::exception Before creators for absent authority/budget or incompatible live population,
     * or after retiring every accepted handle when a native creator fails.
     * Retired-context reuse borrows only the same original process population/certificate.
     * Same-live-context recreation is rejected; failed lane setup remains absorbing.
     */
    static std::unique_ptr<NativePeerTransferEventOwnership> materialize(
        const PreparedGPUContextUsePermit& original,bool timing,const std::string& instance);
    /** @brief Borrow a handle without changing native ownership.
     * @param index Fixed receipt/start/stop coordinate0/1/2; unused timing handles are null.
     * @return Original handle retained until the owning lane's terminal retirement.
     * @throws std::out_of_range For a coordinate outside the closed three-handle layout.
     */
    void* event(std::size_t index)const{return events_.at(index);}
    /** @brief Bind original claims without entering a native creator.
     * @param original Strong original worker/PMA borrow through accepted-handle teardown.
     * @param retirement Same original generation's preallocated canonical native slot.
     * @param pending Unadopted original tokens, or null when borrowing retired process backing.
     */
    NativePeerTransferEventOwnership(PreparedGPUContextUsePermit original,NativeContextRetirementSlot retirement,
        std::unique_ptr<ReadAheadNativeResourceObligation> pending);
    PreparedGPUContextUsePermit original_;
    NativeContextRetirementSlot retirement_;
    std::unique_ptr<ReadAheadNativeResourceObligation> pending_;
    std::array<void*,3> events_{};
    bool attempted_=false;
};
}
