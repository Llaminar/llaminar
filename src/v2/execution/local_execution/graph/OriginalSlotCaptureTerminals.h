/**
 * @file OriginalSlotCaptureTerminals.h
 * @brief Native terminal events owned by one cache over its original publication-slot roster.
 *
 * Slot admission remains entirely in the original protocol. This object owns
 * only the corresponding exact ordering-event receipts and their same-source
 * PMA obligations. It cannot allocate a publication slot or authorize execution.
 */
#pragma once
#include "OriginalSlotCaptureTerminalState.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
namespace llaminar2 {
class IWorkerGPUContext;
class DeviceGraphExecutor;
/** @brief Cache-owned native event population with original admitted slot identity. */
class OriginalSlotCaptureTerminals final {
public:
    using Receipt=OriginalSlotCaptureTerminalReceipt;
    /** @brief Price precisely K−1 additional standalone parent-terminal events on original resources. */
    static PhysicalMemoryPlan extraEventBOM(const MoEOverlayInferenceOriginalSlotGeometry&,
        PhysicalMemoryResource host,PhysicalMemoryResource gpu);
private:
    friend class DeviceGraphExecutor;
    /**
     * @brief Cold-create only additional events; slot0 borrows the existing singleton event.
     * @throws std::exception Before publication for a changed original source/worker/stream or failed creator.
     */
    static std::unique_ptr<OriginalSlotCaptureTerminals> create(
        std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain>,
        std::shared_ptr<IWorkerGPUContext>,std::shared_ptr<PhysicalMemoryAuthority>,
        DeviceId,void* stream,void* existing_singleton_event);
public:
    /** @brief Destroy additional event owners only after every exact receipt has been observed. */
    ~OriginalSlotCaptureTerminals()noexcept;
    OriginalSlotCaptureTerminals(const OriginalSlotCaptureTerminals&)=delete;
    OriginalSlotCaptureTerminals&operator=(const OriginalSlotCaptureTerminals&)=delete;
private:
    /** @brief Authenticate original publication and exact event owner before native record. */
    void* eventForPublication(const MoEOverlayInferenceOriginalSlot&,IWorkerGPUContext*,void* stream)const;
    /** @brief Retain exact receipt only after its native event record succeeds. */
    Receipt published(const MoEOverlayInferenceOriginalSlot&);
    /** @brief Authenticate exact oldest receipt and original source before native observation. */
    void* eventForObservation(const Receipt&,IWorkerGPUContext*,void* stream)const;
    /** @brief Authenticate an exact current native dependency without requiring oldest host observation. */
    void* eventForDeviceDependency(const Receipt&,IWorkerGPUContext*,void* stream)const;
    /** @brief Record actual native observation without releasing its original protocol slot. */
    void observed(const Receipt&);
    /** @return Oldest outstanding receipt for original cache teardown. */
    std::optional<Receipt> nextOutstanding()const;
    /** @return Exact original immutable roster, never independently selected terminal capacity. */
    const std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain>&domain()const noexcept;
private:
    struct State;
    /** @brief Retain cold native/source ownership before any event use. */
    explicit OriginalSlotCaptureTerminals(std::unique_ptr<State>);
    /** @brief Original worker and exact non-null stream cannot change during event lifetime. */
    void requireOwner(IWorkerGPUContext*,void*)const;
    std::unique_ptr<State> state_;
};
}
