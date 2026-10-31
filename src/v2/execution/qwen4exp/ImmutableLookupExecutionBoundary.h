/**
 * @file ImmutableLookupExecutionBoundary.h
 * @brief Typed retained-generation ownership for one immutable lookup service lane.
 *
 * The graph retains one exclusive channel per concurrently executable device
 * lane. All lane variants share its monotonic GPU protocol state and original
 * RAM/mmap source. Admission publishes only an I/O generation identity; the
 * complete GPU generation remains the sole model/sampling decision authority.
 * Standalone prefill owns the same I/O interval until an event-ordered request
 * reset retires it; the active query exposes only this existing CPU I/O duty.
 */
#pragma once
#include "../local_execution/graph/IRetainedGraphStorageBoundary.h"
#include "ImmutableLookupService.h"
#include "Qwen4ExpPreparedStorage.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "kernels/qwen4exp/PooledImmutableSourceKernels.h"
#include "tensors/Tensors.h"
#include <atomic>
#include <memory>
#include <optional>

namespace llaminar2::qwen4exp
{
class ImmutableLookupSubmissionRetirement;
/** @brief Immutable capture bindings plus explicit admission/terminal lifecycle. */
class ImmutableLookupExecutionBoundary final : public IRetainedGraphStorageBoundary
{
public:
    /** @brief Materialize one complete admitted GPU storage lane during setup.
     * @param table Model/rank-owned immutable native source in its explicit RAM/mmap mode.
     * @param device Exact participant owning the channel aliases and protocol state.
     * @param protocol Original retained canonical INT32/64B tensor owner, already allocated on this exact device.
     * @param maximumTokenRows Retained transaction capacity; live communication uses only device counts.
     * @param memory Canonical PMA certificate charging mapped bytes to table->hostDevice().
     * @return Stable participant-local service/kernel/state binding, with no model graph backend switch.
     */
    static std::shared_ptr<ImmutableLookupExecutionBoundary> create(std::shared_ptr<HostImmutableRowProvider> table,
        DeviceId device,std::shared_ptr<TensorBase> protocol,std::size_t maximumTokenRows,
        std::shared_ptr<PhysicalMemoryAuthority> memory);
    /**
     * @brief Bind one real provider/channel to its existing pooled owner/coordinator roots and retained graph boundary.
     * @param service Actual already-admitted subordinate RAM/SSD source worker.
     * @param geometry Exact original source/accepted-owner/coordinator staging declaration.
     * @param owner Same original actual slot/grant owner.
     * @param coordinator Same independent policy/coalescing owner.
     * @param authority Original PMA that admitted all source, channel and native roots.
     * @param setupStream Exact ordinary setup stream, retained even through partial initialization failure.
     * @return Mutually exclusive pooled source arm, without a serving/topology capability advertisement.
     * @note Missing directed owner edges fail before materialization; the first executable slice is one local edge.
     */
    static std::shared_ptr<ImmutableLookupExecutionBoundary> createPooled(std::shared_ptr<ImmutableLookupService> service,
        PooledImmutableSourceGeometry geometry,std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> authority,
        void* setupStream);
    /** @return Whether this original storage boundary owns the real pooled source arm. */
    bool isPooled()const noexcept{return pooled_.has_value();}
    /** @brief Bind only the original canonical request controller before any pooled source phase is recorded. */
    void bindPooledGenerationControl(DeviceGenerationControlBinding control);
    /** @return Same strong original source packet storage; ordinary/adaptive arms reject this operation. */
    const std::shared_ptr<PooledImmutableSourcePreparedStorage>& pooledStorage()const;
    /** @return Actual retained provider-native code bundle; no test source producer can substitute. */
    const IPooledImmutableSourceKernels& pooledKernels()const;
    /** @return Actual CPU provider plus mapped-channel leases after worker retirement. */
    HostImmutableRowOwnerObservation retainedSourceOwnerObservation() const;
    /** @brief Bind admitted state/generation storage and setup-created backend kernels.
     * @param service Exact lane owner retaining the immutable table and mapped backing.
     * @param device Exact alias address space and owning execution participant.
     * @param kernels Prepared native decoder/publication bundle; no backend switch at execution.
     * @param protocol Retained original INT32/64B protocol root; state and admission generation are derived from its sealed native allocation.
     */
    ImmutableLookupExecutionBoundary(std::shared_ptr<ImmutableLookupService> service,DeviceId device,
        std::unique_ptr<IImmutableLookupKernels> kernels,std::shared_ptr<TensorBase> protocol);
    /** @copydoc IRetainedGraphStorageBoundary::device */
    DeviceId device() const noexcept override
    {
        return device_;
    }
    /** @copydoc IRetainedGraphStorageBoundary::captureIdentity */
    std::uint64_t captureIdentity() const noexcept override{return identity_;}
    /** @copydoc IRetainedGraphStorageBoundary::bindGenerationControl */
    void bindGenerationControl(RetainedStorageGenerationControl control) override;
    /** @copydoc IRetainedGraphStorageBoundary::generationControl */
    RetainedStorageGenerationControl generationControl() const override;
    /** @copydoc IRetainedGraphStorageBoundary::beginAdmission */
    void beginAdmission(std::uint64_t generation,void *stream) override;
    /** @copydoc IRetainedGraphStorageBoundary::hasActiveInterval */
    bool hasActiveInterval() const noexcept override{return armed_.load(std::memory_order_acquire);}
    /** @copydoc IRetainedGraphStorageBoundary::requireArmedForLaunch */
    void requireArmedForLaunch() const override;
    /** @copydoc IRetainedGraphStorageBoundary::recordTerminalClose */
    void recordTerminalClose(void *stream,const int *healthy=nullptr,const int *complete=nullptr) const override;
    /** @copydoc IRetainedGraphStorageBoundary::awaitTerminal */
    void awaitTerminal() override;
    /** @copydoc IRetainedGraphStorageBoundary::terminalObservation */
    RetainedStorageTerminalObservation terminalObservation() const override;
    /** @brief Exact channel view for the participant-local native lookup compute stage. */
    const ImmutableLookupDeviceView &channel() const;
    /** @brief Borrow the immutable prepared bundle without changing backend or codebook policy. */
    const IImmutableLookupKernels &kernels() const;
    /** @brief Persistent GPU protocol position, consumed directly by captured kernels. */
    ImmutableLookupDeviceState *state() const {requireProtocolOwner();return state_;}
    /** @brief Persistent GPU I/O admission identity, with exact admission-stream publication. */
    const std::uint64_t *generation() const {requireProtocolOwner();return generation_;}
private:
    /** @brief Reject changed original tensor/native device/capacity/backend generation before any operation. */
    void requireProtocolOwner() const;
    std::shared_ptr<ImmutableLookupService> service_;
    /** @brief Source packet roots only; physical cache state remains with the original pooled owners. */
    struct Pooled final {
        std::shared_ptr<PooledImmutableSourcePreparedStorage> storage;
        std::unique_ptr<IPooledImmutableSourceKernels> kernels;
        std::optional<DeviceGenerationControlBinding> control;
        std::shared_ptr<ImmutableLookupSubmissionRetirement> setupRetirement; ///< Retires before the original storage/code/source owners.
    };
    /** @brief Only the closed real-provider factory constructs the pooled arm. */
    ImmutableLookupExecutionBoundary(std::shared_ptr<ImmutableLookupService> service,
        std::shared_ptr<TensorBase> protocol,Pooled pooled);
    DeviceId device_;
    std::optional<ImmutableLookupService::Binding> binding_;
    std::optional<Pooled> pooled_;
    std::unique_ptr<IImmutableLookupKernels> kernels_;
    std::shared_ptr<TensorBase> protocol_; ///< Sole retained physical protocol owner, never a second allocation.
    std::unique_ptr<Qwen4ExpPreparedStorage> protocolSeal_; ///< Existing canonical seal authenticates the tensor allocator backend as well as pointer metadata.
    std::shared_ptr<ImmutableLookupSubmissionRetirement>
        admissionRetirement_;   ///< Setup-owned ordinary admission event; destroys before original
                                ///< protocol/source owners.
    IBackend *backend_=nullptr; ///< Exact public native backend at setup.
    std::uint64_t runtimeGeneration_=0,identity_=0; ///< Immutable storage/capture metadata, not protocol values.
    ImmutableLookupDeviceState *state_;
    std::uint64_t *generation_;
    ImmutableLookupService::Ticket ticket_;
    std::uint64_t requestEpoch_=0,intervalGeneration_=0; ///< Host-owned immutable I/O admission identity only.
    ImmutableLookupAdmissionArgs::ProtocolInitialization initialization_=ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce;
    std::atomic<bool> armed_{false};
    RetainedStorageGenerationControl control_;
    ImmutableLookupServiceStats intervalStart_; ///< Snapshot of I/O counters before the worker starts.
    std::optional<RetainedStorageTerminalObservation> terminal_; ///< Immutable only after worker join.
};
} // namespace llaminar2::qwen4exp
