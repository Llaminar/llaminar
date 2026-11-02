/**
 * @file ReadAheadNativeParentTerminal.h
 * @brief Independent certified baseline owner for exactly one test-parent terminal event.
 *
 * This event is not part of the one-stream/two-event read-ahead family. A
 * strong alias retains its separate same-PMA commitment until all actual
 * readers retire. Production certificate selection is closed and initially
 * empty; diagnostic calibration is the only route to justified native bounds.
 * A distinct closed diagnostic variant shares the original creator/destructor
 * while granting no numeric bound or PMA opaque reservation.
 */
#pragma once
#include "ReadAheadNativeResourceFamily.h"
namespace llaminar2 {
namespace test {struct ReadAheadNativeParentTerminalReplayAccess;struct PooledNativeProgramDiagnosticAccess;}
/** @brief Original one-event parent baseline, with independent source-bound limits and lifetime. */
class ReadAheadNativeParentTerminal final:public std::enable_shared_from_this<ReadAheadNativeParentTerminal> {
public:
    /** @brief Project independently certified CPU/GPU terminal-event bounds into the existing BOM. */
    static PhysicalMemoryPlan bom(DeviceId device,PhysicalMemoryResource host,PhysicalMemoryResource gpu);
    /** @brief Commit both original allocator bounds before creating precisely one native event. */
    static std::shared_ptr<ReadAheadNativeParentTerminal> materialize(std::shared_ptr<PhysicalMemoryAuthority> memory,
        std::shared_ptr<IWorkerGPUContext> worker,DeviceId device);
    /** @return Strong alias of this exact original native event and its independent PMA commitments. */
    std::shared_ptr<void> eventOwner();
    /** @brief Require actual nonblocking event completion before native cleanup and commitment release. */
    ~ReadAheadNativeParentTerminal()noexcept;
    ReadAheadNativeParentTerminal(const ReadAheadNativeParentTerminal&)=delete;
    ReadAheadNativeParentTerminal& operator=(const ReadAheadNativeParentTerminal&)=delete;
    /** @return This independent event's actual provenance, separate from the side family. */
    NativeAuxiliaryOwnershipKind ownershipKind()const noexcept;
private:
    friend struct test::ReadAheadNativeParentTerminalReplayAccess;
    friend struct test::PooledNativeProgramDiagnosticAccess;
    /** @brief Closed raw state; no numeric certificate or commitment can be fabricated. */
    struct UnqualifiedDiagnostic final {
        UnqualifiedDiagnostic(const UnqualifiedDiagnostic&)=default;
    private:
        friend struct test::PooledNativeProgramDiagnosticAccess;
        UnqualifiedDiagnostic()=default;
    };
    /** @brief Defined only in the dedicated actual-program diagnostic fixture. */
    ReadAheadNativeParentTerminal(std::shared_ptr<PhysicalMemoryAuthority>,std::shared_ptr<IWorkerGPUContext>,UnqualifiedDiagnostic);
    /** @brief Run the exact original one-event creator for both explicitly typed ownership states. */
    void createNativeHandle();
    /** @brief Immutable replay/source contract; no public caller-supplied budget or signature exists. */
    struct Certificate {ReadAheadNativeResourceClosure closure;std::size_t hostBytes,deviceBytes;std::string proof;
        ReadAheadNativeRetirementPolicy retirement=ReadAheadNativeRetirementPolicy::HandleDestruction;};
    /** @brief Resolve only the installed original native closure; absent certificates reject before native work. */
    static Certificate requireInstalled(DeviceId device);
    /** @brief Actual positive construction shared with private device-free failure replay only. */
    static std::shared_ptr<ReadAheadNativeParentTerminal> materializeCertified(std::shared_ptr<PhysicalMemoryAuthority>,
        std::shared_ptr<IWorkerGPUContext>,Certificate,const GPUContextProvenance&);
    /** @brief Retain already-committed original owners before native event creation. */
    ReadAheadNativeParentTerminal(std::shared_ptr<PhysicalMemoryAuthority>,std::shared_ptr<IWorkerGPUContext>,Certificate,
        PhysicalMemoryOwnerReservation,std::optional<PhysicalMemoryOwnerReservation>,std::optional<NativeContextRetirementSlot>);
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::shared_ptr<IWorkerGPUContext> worker_;
    std::variant<Certificate,UnqualifiedDiagnostic> certificate_;
    PhysicalMemoryOwnerReservation host_;
    std::optional<PhysicalMemoryOwnerReservation> device_;
    std::optional<NativeContextRetirementSlot> retirementSlot_;
    void* event_=nullptr;
};
} // namespace llaminar2
