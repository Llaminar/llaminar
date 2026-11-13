/**
 * @file NativeContextRetirement.h
 * @brief Original reservation ownership retained through one physical context reset.
 *
 * This is a lifecycle owner, never a byte ledger. Closed resource owners move
 * their existing PMA tokens into a preallocated slot before native creation.
 * The canonical pool carries those exact tokens across worker destruction;
 * TransferEngine's validated original reset discharges context-owned tokens.
 * Process-owned roots close generation use there, retaining the original PMA
 * tokens with the existing process authority until actual OS reclamation.
 */
#pragma once
#include "DeviceId.h"
#include "NativePhysicalLifetime.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace llaminar2 {
namespace test {struct ReadAheadNativeResourceReplayAccess; struct FollowerPredecessorDiagnosticGraphResources;}
class PhysicalMemoryAuthority;
class GPUDeviceContextPool;
class GPUContextProvenance;
class ExclusiveGPUDeviceGenerationRetirement;
class NativeContextRetirementOwner;
class GPUDeviceProcessLifetime;
class NativeStoragePreparationPmaBinding;
class PhysicalMemoryOpaqueRuntimeCommitment;
class NativeContextOpaqueContract;
class ReadAheadNativeResourceFamily;
class ReadAheadNativeParentTerminal;
class RetainedParticipantGraphExecutable;
struct NativeContextRetirementSlotState;
/** @brief Move-only token payload implemented by the original closed resource owner. */
class IContextNativeRetirementObligation {
public:
    /** @brief Destroy only when original reset or pre-native cancellation authorizes release. */
    virtual ~IContextNativeRetirementObligation() noexcept=default;
    /** @return Exact original canonical authority, not a recreated equal plan. */
    virtual const std::shared_ptr<PhysicalMemoryAuthority>& originalAuthority()const noexcept=0;
    /** @return Exact selected physical endpoint. */
    virtual DeviceId device()const noexcept=0;
    /** @return Exact original canonical pool generation. */
    virtual std::uint64_t poolGeneration()const noexcept=0;
    /** @return Original source/runtime/population certificate identity. */
    virtual const std::string& certificateIdentity()const noexcept=0;
    /** @return Whether actual reset reclaims this opaque native root. */
    virtual NativePhysicalLifetime physicalLifetime()const noexcept=0;
};
/** @brief Preallocated original generation slot; never retains a native worker. */
class NativeContextRetirementSlot final {
public:
    NativeContextRetirementSlot(const NativeContextRetirementSlot&)=delete;
    NativeContextRetirementSlot& operator=(const NativeContextRetirementSlot&)=delete;
    /** @brief Move the sole resource lifecycle without duplicating its obligation. */
    NativeContextRetirementSlot(NativeContextRetirementSlot&&)noexcept;
    /** @brief Close the prior exact slot before taking another lifecycle. */
    NativeContextRetirementSlot& operator=(NativeContextRetirementSlot&&)noexcept;
    /** @brief Reject live handles; retain adopted tokens until original reset. */
    ~NativeContextRetirementSlot()noexcept;
    /** @brief Move original tokens into this preallocated owner before native creators.
     * @param obligation Closed resource leaf with exact authority/device/generation/certificate.
     * @note A mismatch is fatal before cleanup can release a live obligation.
     * @throws std::bad_alloc If pre-native process ownership bookkeeping cannot be retained.
     */
    void adopt(std::unique_ptr<IContextNativeRetirementObligation> obligation);
    /** @brief Mark the first native creator attempt before invoking its API. */
    void armNativeAttempt()noexcept;
    /** @brief Publish exact terminal-aware native handle teardown, without releasing tokens. */
    void completeNativeHandles()noexcept;
private:
    friend class NativeContextRetirementOwner;
    friend struct test::ReadAheadNativeResourceReplayAccess;
    friend struct test::FollowerPredecessorDiagnosticGraphResources; ///< Actual original-PMA diagnostic General obligations only.
    friend class ReadAheadNativeResourceFamily;
    friend class ReadAheadNativeParentTerminal;
    friend class RetainedParticipantGraphExecutable;
    friend class NativeOrderingEventGroup;
    friend class NativeAuxiliaryStreamOwnership;
    friend class NativePeerTransferEventOwnership;
    /** @brief Delegate construction-only events or a live stream only after its actual context record takes ownership. */
    void delegateHandlesToContext()noexcept;
    /** @brief Borrow only the same original bounded process population before reserving again. */
    bool tryBorrowProcessRuntime(const std::string& population);
    /** @brief Freeze the unique original process population before adopting its first tokens. */
    void prepareProcessRuntime(const std::string& population);
    NativeContextRetirementSlot(std::shared_ptr<NativeContextRetirementOwner>,std::shared_ptr<NativeContextRetirementSlotState>)noexcept;
    /** @brief Cancel an unadopted slot or close a never-started adopted lifecycle. */
    void release()noexcept;
    std::shared_ptr<NativeContextRetirementOwner> owner_;
    std::shared_ptr<NativeContextRetirementSlotState> state_;
};
/** @brief Actual canonical generation owner, containing tokens but no byte counters or worker. */
class NativeContextRetirementOwner final:public std::enable_shared_from_this<NativeContextRetirementOwner> {
public:
    /** @brief Destruction with an undischargeable original obligation is fatal. */
    ~NativeContextRetirementOwner()noexcept;
private:
    friend class GPUDeviceContextPool;
    friend class NativeContextRetirementSlot;
    friend class GPUDeviceProcessLifetime;
    friend class ExclusiveGPUDeviceGenerationRetirement;
    NativeContextRetirementOwner(DeviceId,std::uint64_t);
    /** @brief Allocate bookkeeping only after original resource PMA commitments exist. */
    NativeContextRetirementSlot reserve(const std::shared_ptr<PhysicalMemoryAuthority>&,const std::string&);
    /** @brief Reject live or still-borrowed resource slots before worker removal. */
    void requireHandlesRetired()const;
    /** @return Whether no original slot remains; no physical observation occurs. */
    bool empty()const noexcept;
    /** @brief Release exact original tokens only from the validated reset transaction. */
    void physicallyRetired()noexcept;
    /** @return Whether original reset has discharged this owner. */
    bool discharged()const noexcept;
    /** @brief Authenticate a published process root against exact original source authority. */
    static bool matchesProcessRoot(const std::shared_ptr<NativeContextRetirementSlotState>&,
        const std::shared_ptr<PhysicalMemoryAuthority>&,DeviceId,const std::string&,const std::string&);
    /** @brief Close only the current generation use, never physical process ownership. */
    /** @brief Reject a duplicate publication atomically before native creators. */
    static void requireUniqueProcessRoot(const std::shared_ptr<NativeContextRetirementSlotState>&,
        const std::shared_ptr<NativeContextRetirementSlotState>&);
    /** @brief Preserve the actual Preparing source/opaque shared owners without fake generation. */
    static std::shared_ptr<NativeContextRetirementSlotState> preparingProcessRoot(
        const std::shared_ptr<NativeStoragePreparationPmaBinding>&,
        const std::shared_ptr<PhysicalMemoryOpaqueRuntimeCommitment>&,const NativeContextOpaqueContract&);
    /** @brief Reject independent source adoption while actual process backing survives. */
    /** @brief Authenticate the original PMA independently of a native GPU endpoint. */
    static void requireOriginalAuthority(const std::shared_ptr<NativeContextRetirementSlotState>&,const std::shared_ptr<PhysicalMemoryAuthority>&);
    static void requireOriginalProcessAuthority(const std::shared_ptr<NativeContextRetirementSlotState>&,
        DeviceId,const std::shared_ptr<PhysicalMemoryAuthority>&);
    static void startProcessUse(const std::shared_ptr<NativeContextRetirementSlotState>&)noexcept;
    static void finishProcessUse(const std::shared_ptr<NativeContextRetirementSlotState>&)noexcept;
    mutable std::mutex mutex_;
    DeviceId device_;
    std::uint64_t generation_;
    std::shared_ptr<PhysicalMemoryAuthority> authority_;
    std::vector<std::shared_ptr<NativeContextRetirementSlotState>> slots_;
    bool retired_=false;
};
}
