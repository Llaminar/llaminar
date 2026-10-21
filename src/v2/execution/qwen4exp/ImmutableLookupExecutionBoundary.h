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
#include "kernels/qwen4exp/AdaptiveLookupPreparedStorage.h"
#include "loaders/AdaptiveImmutableLookupRootStorage.h"
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
     * @brief Retain a prepared protocol2 component in the existing storage
     * interval authority.
     * @param service Exact single-GPU subordinate service, already PMA
     * materialized.
     * @param plan Checked original source/policy/frame declaration.
     * @param roots Actual model-owned root carrier retaining all original GPU PMA leases.
     * @param setupStream Exact setup producer stream, outside graph capture.
     * @return One boundary retained by the model/source owner across graph
     * families and resets.
     * @throws std::exception For foreign/unprepared roots, duplicate metadata
     * initialization, unsupported native backend, or partial setup/publication
     * failure.
     * @note This factory neither allocates roots nor admits memory. The model
     * owner must create it once and retain it; later graph builders borrow the
     * same boundary. Once-only initialization is event-published without a
     * success wait. The boundary retains that setup event through final teardown,
     * so abandoned runner setup cannot free roots ahead of native metadata writes.
     */
    static std::shared_ptr<ImmutableLookupExecutionBoundary>
    createAdaptive(std::shared_ptr<ImmutableLookupService> service, AdaptiveImmutableRowCacheNativePlan plan,
                   std::shared_ptr<const AdaptiveImmutableLookupRootStorage> roots, void *setupStream);
    /** @return Whether the declared transport is the checked protocol2 native
     * cache component. */
    bool isAdaptive() const noexcept
    {
        return adaptive_.has_value();
    }
    /**
     * @brief Bind the stage's actual canonical controller before captured
     * terminal construction.
     * @param control Original controller owner, not caller-provided
     * health/completion addresses.
     * @throws std::exception For another controller/request/device or
     * conflicting prior binding.
     */
    void bindAdaptiveGenerationControl(DeviceGenerationControlBinding control);
    /** @return The same original root aliases, each retaining their actual physical lease owner. */
    AdaptiveImmutableLookupRoots adaptiveRoots() const;
    /** @brief Register/validate original model roots as external canonical arena roles before capture.
     * @param arena Actual participant arena, retained by the enclosing graph family.
     * @throws std::exception For conflicts or capture-time registration before mutating roles.
     */
    void bindAdaptiveArena(BufferArena &arena) const;
    /** @return Actual GPU root/lease owner, rejecting ordinary protocol storage. */
    const std::shared_ptr<const AdaptiveImmutableLookupRootStorage> &rootStorage() const;
    /** @return Actual CPU provider plus mapped-channel leases after worker retirement. */
    HostImmutableRowOwnerObservation retainedSourceOwnerObservation() const;
    /** @return Actual source-owned prepared storage, rejecting the ordinary
     * protocol arm. */
    const std::shared_ptr<AdaptiveLookupPreparedStorage> &adaptiveStorage() const;
    /** @return Actual stateless native bundle, rejecting the ordinary protocol
     * arm. */
    const IImmutableAdaptiveLookupKernels &adaptiveKernels() const;
    /**
     * @brief Authenticate all original root aliases before graph binding or
     * frontier acquisition.
     * @param cache Original optional cache; must be absent at zero slots.
     * @param frame Original complete work-frame root.
     * @param protocol Original complete protocol root.
     * @throws std::invalid_argument For copied, replaced, foreign or missing
     * roots.
     */
    void requireAdaptiveRoots(const TensorBase *cache, const TensorBase *frame,
                              const TensorBase *protocol) const;
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
    /** @brief One native component; no second interval, sequence, cache-state
     * or memory ledger. */
    struct Adaptive final
    {
        std::shared_ptr<const AdaptiveImmutableLookupRootStorage> roots;
        std::shared_ptr<const PhysicalMemoryAuthority> authority;
        std::shared_ptr<AdaptiveLookupPreparedStorage> storage;
        std::unique_ptr<IImmutableAdaptiveLookupKernels> kernels;
        std::optional<DeviceGenerationControlBinding> control;
        // Declared last so setup retirement completes before roots or kernels
        // release ownership, including setup abandoned before the first graph.
        std::shared_ptr<ImmutableLookupSubmissionRetirement> setupRetirement;
    };
    /** @brief Only createAdaptive constructs this already-prepared transport
     * alternative. */
    ImmutableLookupExecutionBoundary(std::shared_ptr<ImmutableLookupService> service,
                                     std::shared_ptr<TensorBase> protocol, Adaptive adaptive);
    DeviceId device_;
    std::optional<ImmutableLookupService::Binding> binding_;
    std::optional<Adaptive> adaptive_;
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
