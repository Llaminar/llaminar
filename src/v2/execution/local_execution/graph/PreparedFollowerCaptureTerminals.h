/**
 * @file PreparedFollowerCaptureTerminals.h
 * @brief Original prepared-model ownership of serially borrowed follower terminal handles.
 *
 * Physical handles and their original PMA obligations survive runner replacement.
 * Protocol slots, publication receipts and graph execution remain runner-owned.
 */
#pragma once
#include "execution/moe/MoEOverlayInferenceOriginalSlot.h"
#include "backends/DeviceId.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
namespace llaminar2 {
class PreparedGPUContextUsePermit;
class IWorkerGPUContext;
class OriginalSlotCaptureTerminals;
class DeviceGraphExecutor;
/** @brief Exact original follower cache coordinate, without arena or runner pointers. */
struct PreparedFollowerCaptureTerminalKey final {
    int participant_id; ///< Original declared target participant, not a local cache index.
    std::size_t graph_family; ///< Actual retained family ordinal, stable across runner recreation.
    int physical_rows; ///< Exact captured physical-row extent of this family.
    DeviceId device; ///< Original prepared GPU owning the physical handles.
    bool operator==(const PreparedFollowerCaptureTerminalKey&)const=default;
};
/** @brief One original singleton plus K-1 event population; grants no execution permission. */
class PreparedFollowerCaptureTerminalPopulation final {
public:
    PreparedFollowerCaptureTerminalPopulation(const PreparedFollowerCaptureTerminalPopulation&)=delete;
    PreparedFollowerCaptureTerminalPopulation& operator=(const PreparedFollowerCaptureTerminalPopulation&)=delete;
    PreparedFollowerCaptureTerminalPopulation(PreparedFollowerCaptureTerminalPopulation&&)=delete;
    PreparedFollowerCaptureTerminalPopulation& operator=(PreparedFollowerCaptureTerminalPopulation&&)=delete;
    /** @brief Destroy handles only after the final exclusive borrower has retired. */
    ~PreparedFollowerCaptureTerminalPopulation()noexcept;
    /**
     * @param geometry Same canonical retained-slot geometry used by the event BOM.
     * @return Exact known CPU root bytes, distinct from unchanged opaque allowances.
     * @throws std::overflow_error If the fixed trailing handle coordinates overflow.
     */
    static std::size_t physicalBytes(const MoEOverlayInferenceOriginalSlotGeometry& geometry);
private:
    friend class PreparedFollowerCaptureTerminalOwner;
    friend class OriginalSlotCaptureTerminals;
    friend class DeviceGraphExecutor;
    struct State;
    explicit PreparedFollowerCaptureTerminalPopulation(State* state):state_(state){}
    /**
     * @brief Authenticate the physical loan without authorizing graph execution.
     * @param original Same PMA/control block, worker and native context generation.
     * @param exact_stream Non-null stream originally owning these handles.
     * @param geometry Original admitted retained-slot geometry.
     * @throws std::exception If any original physical identity changed.
     */
    void requireOriginal(const PreparedGPUContextUsePermit& original,void* exact_stream,
        const MoEOverlayInferenceOriginalSlotGeometry& geometry)const;
    /**
     * @param index Original slot coordinate authenticated by the caller's receipt state.
     * @return Actual retained native event, never a new receipt or slot permission.
     * @throws std::out_of_range If the coordinate exceeds original geometry.
     */
    void* event(std::size_t index)const;
    State* state_;
};
/**
 * @brief Fixed actual cache roster retained by one prepared model, not a capacity ledger.
 *
 * Create under the original admitted model, acquire every actual cold family and
 * bind each loan to its new cache/Protocol before capture. Seal only after the
 * complete roster is borrowed; subsequent runners must reproduce that roster.
 * Each cache then observes its native receipts, destroys embedded parents and
 * children, and returns its loan. Only after all prior loans retire may another
 * runner acquire the same populations. Physical events survive those returns;
 * request reset never recreates them or resets the original reservation ledger.
 */
class PreparedFollowerCaptureTerminalOwner final {
public:
    PreparedFollowerCaptureTerminalOwner(const PreparedFollowerCaptureTerminalOwner&)=delete;
    PreparedFollowerCaptureTerminalOwner& operator=(const PreparedFollowerCaptureTerminalOwner&)=delete;
    PreparedFollowerCaptureTerminalOwner(PreparedFollowerCaptureTerminalOwner&&)=delete;
    PreparedFollowerCaptureTerminalOwner& operator=(PreparedFollowerCaptureTerminalOwner&&)=delete;
    /** @return Exact CPU holder bytes admitted once for the original model. */
    static std::size_t physicalBytes();
    /**
     * @brief Cold-create the model-owned declaration under its original admission.
     * @param memory Same original rank PMA/control block retaining all event units.
     * @param geometry Canonical retained-capacity geometry already included in the BOM.
     * @return Original holder; it retains no graph, runner or Protocol backreference.
     * @throws std::exception If admission, geometry or known root allocation fails.
     */
    static std::shared_ptr<PreparedFollowerCaptureTerminalOwner> create(
        std::shared_ptr<PhysicalMemoryAuthority> memory,
        const MoEOverlayInferenceOriginalSlotGeometry& geometry);
    /**
     * @brief Exclusively borrow one actual cache population before capture.
     * @param key Original participant/family/physical-row/device declaration.
     * @param original Same prepared source/PMA, worker and native context generation.
     * @param exact_stream Original non-null participant stream; first native creators
     *        execute on its owning worker.
     * @return Physical loan to retain inside fresh cache receipt state through drain.
     * @throws std::exception For a live borrower, changed identity/roster, or native
     *         creator failure. Actual creation failure is absorbing; no retry occurs.
     */
    std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> acquire(
        PreparedFollowerCaptureTerminalKey key,const PreparedGPUContextUsePermit& original,
        void* exact_stream);
    /**
     * @brief Seal or verify the complete borrowed roster before runner execution.
     * @throws std::exception If construction failed, no cache was declared, or any
     *         original population is missing/duplicated rather than exclusively borrowed.
     * @note Call after all family loans are bound, never after returning them.
     */
    void seal();
    /** @brief Release only the original holder; live borrowers retain physical handles. */
    ~PreparedFollowerCaptureTerminalOwner();
private:
    struct State;
    explicit PreparedFollowerCaptureTerminalOwner(State* state):state_(state){}
    State* state_;
};
}
