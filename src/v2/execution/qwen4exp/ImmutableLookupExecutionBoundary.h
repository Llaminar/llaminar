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
#include "PooledImmutableRetirementAuthority.h"
#include "../local_execution/graph/IRetainedGraphStorageBoundary.h"
#include "ImmutableLookupService.h"
#include "Qwen4ExpPreparedStorage.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "kernels/qwen4exp/PooledImmutableSourceKernels.h"
#include "tensors/Tensors.h"
#include <atomic>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::test { struct ImmutableLookupSourceAdmissionReplayAccess; }

namespace llaminar2::qwen4exp
{
class ImmutableLookupSubmissionRetirement;
class PooledImmutableCompleteCohort;
/** @brief Immutable capture bindings plus explicit admission/terminal lifecycle. */
class ImmutableLookupExecutionBoundary final : public IRetainedGraphStorageBoundary
{
public:
    /** @brief Original participant-local host admission metadata gathered by its actual TP parent. */
    struct SourceAdmissionParticipant
    {
        std::shared_ptr<ImmutableLookupExecutionBoundary> boundary;
        std::uint64_t request_epoch=0;
    };
    /** @brief One original CPU source invocation retaining all of its actual consumer boundaries. */
    class SourceGroupAdmission final
    {
    public:
        SourceGroupAdmission(SourceGroupAdmission&&) noexcept=default;
        SourceGroupAdmission& operator=(SourceGroupAdmission&&)=delete;
        SourceGroupAdmission(const SourceGroupAdmission&)=delete;
        SourceGroupAdmission& operator=(const SourceGroupAdmission&)=delete;
        /** @brief Cancel/join an abandoned CPU group without consuming adopted GPU retirement duties. */
        ~SourceGroupAdmission() noexcept;
        /** @brief Authenticate reuse of this same still-active interval from the same actual parent participants. */
        bool reuses(std::span<const SourceAdmissionParticipant>)const;
        /** @brief Observe all original closes and retire unadmitted channel tickets only. */
        bool finish();
    private:
        friend class ImmutableLookupExecutionBoundary;
        /** @brief Retain only the original service/group and original native boundary owners. */
        SourceGroupAdmission(std::shared_ptr<ImmutableLookupService> service,
            ImmutableLookupService::GroupAdmission admission,std::vector<SourceAdmissionParticipant> participants);
        std::shared_ptr<ImmutableLookupService> service_;
        std::vector<SourceAdmissionParticipant> participants_;
        ImmutableLookupService::GroupAdmission admission_;
        bool finished_=false,result_=false;
    };
    /** @return Whether this exact predeclared CPU service requires complete-group admission. */
    bool requiresSourceGroupAdmission()const noexcept{return service_->channelDevices().size()>1;}
    /**
     * @brief Arm the complete original CPU source group before any participant submits GPU work.
     * @param participants Actual parent members and their unchanged host request epochs.
     * @pre Initial owners come from the actual Rank/graph-builder ModelContext path.
     * This method retains those ownership blocks; reuse authenticates against them.
     * @return Move-only original service invocation; per-device GPU admission remains with DGO.
     * @throws std::exception For incomplete/duplicate/foreign/stale source membership or active channels.
     */
    static std::unique_ptr<SourceGroupAdmission> armSourceGroup(std::span<const SourceAdmissionParticipant> participants);
    /** @brief Materialize one complete admitted GPU storage lane during setup.
     * @param table Model/rank-owned immutable native source in its explicit RAM/mmap mode.
     * @param device Exact participant owning the channel aliases and protocol state.
     * @param protocol Original retained canonical INT32/64B tensor owner, already allocated on this exact device.
     * @param maximumTokenRows Retained transaction capacity; live communication uses only device counts.
     * @param memory Canonical PMA certificate charging mapped bytes to table->hostDevice().
     * @param worker Original admitted pre-quota I/O instance token; never a late-created worker.
     * @return Stable participant-local service/kernel/state binding, with no model graph backend switch.
     */
    static std::shared_ptr<ImmutableLookupExecutionBoundary> create(std::shared_ptr<HostImmutableRowProvider> table,
        DeviceId device,std::shared_ptr<TensorBase> protocol,std::size_t maximumTokenRows,
        std::shared_ptr<PhysicalMemoryAuthority> memory,RetainedHostServiceUseToken worker);
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
        PooledImmutableSourceGeometry geometry,std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> authority,
        void* setupStream);
    /** @brief One-owner spelling delegates to complete original source/cohort validation. */
    static std::shared_ptr<ImmutableLookupExecutionBoundary> createPooled(std::shared_ptr<ImmutableLookupService> service,
        PooledImmutableSourceGeometry geometry,std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> authority,
        void* setupStream)
    {return createPooled(std::move(service),std::move(geometry),std::vector{std::move(owner)},std::move(coordinator),std::move(authority),setupStream);}
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
     * @param originalMemory Original mutable authority admitting the one-event retirement program.
     * @param protocol Retained original INT32/64B protocol root; state and admission generation are derived from its sealed native allocation.
     */
    ImmutableLookupExecutionBoundary(std::shared_ptr<ImmutableLookupService> service,DeviceId device,
        std::unique_ptr<IImmutableLookupKernels> kernels,std::shared_ptr<TensorBase> protocol,
        std::shared_ptr<PhysicalMemoryAuthority> originalMemory);
    /** @copydoc IRetainedGraphStorageBoundary::device */
    DeviceId device() const noexcept override
    {
        return device_;
    }
    /** @copydoc IRetainedGraphStorageBoundary::captureIdentity */
    std::uint64_t captureIdentity() const noexcept override{return identity_;}
    /** @brief Retain the exact original DGO parent stream authority, without creating another stream. */
    /** @return Original consumed coordinator permit, without acquiring or creating any worker. */
    std::optional<PreparedGPUContextUsePermit> retainedParticipantSource()const override;
    void bindSourceSubmissionStream(const RetainedGraphParticipantParentStream&)override;
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
    friend struct llaminar2::test::ImmutableLookupSourceAdmissionReplayAccess;
    /** @brief Immutable membership metadata only; never grants native admission or resource authority. */
    struct SourceAdmissionIdentity {
        DeviceId device;
        std::shared_ptr<const void> owner;
        std::uint64_t epoch=0;
    };
    /** @brief Match one original pointer, ownership block, device and host epoch without observing GPU state. */
    static bool sameSourceAdmissionIdentity(const SourceAdmissionIdentity& a,const SourceAdmissionIdentity& b) noexcept {
        return a.device==b.device&&a.epoch&&a.epoch==b.epoch&&a.owner&&a.owner==b.owner&&
            !a.owner.owner_before(b.owner)&&!b.owner.owner_before(a.owner);
    }
    /** @brief Reuse only the same closed membership; projection reads original immutable owner metadata only. */
    template<class Member,class Project>
    static bool sourceAdmissionReuses(std::span<const Member> original,std::span<const Member> current,Project project) {
        if(original.empty()||original.size()!=current.size())return false;
        for(const auto& member:original){
            const auto expected=project(member);bool found=false;
            for(const auto& candidate:current){
                const auto actual=project(candidate);
                if(actual.device!=expected.device)continue;
                if(found||!sameSourceAdmissionIdentity(expected,actual))return false;
                found=true;
            }
            if(!found)return false;
        }
        return true;
    }
    /** @brief Resolve every canonical source channel exactly once, independently of authored TP order.
     * @param channels Original service geometry membership, in physical channel order.
     * @param members Actual parent-projected original owners, in unchanged TP order.
     * @return Indices into members in channel order; compute participants are never reordered.
     * @throws std::invalid_argument For missing, duplicate, foreign or incomplete membership.
     */
    static std::vector<std::size_t> sourceAdmissionOrder(std::span<const DeviceId> channels,
        std::span<const SourceAdmissionIdentity> members) {
        if(channels.empty()||channels.size()!=members.size())
            throw std::invalid_argument("Original source group has incomplete membership");
        std::vector<std::size_t> order;order.reserve(channels.size());
        for(const auto channel:channels){
            std::size_t found=members.size();
            for(std::size_t i=0;i<members.size();++i)if(members[i].device==channel){
                if(found!=members.size()||!members[i].owner||!members[i].epoch)
                    throw std::invalid_argument("Original source group has duplicate or incomplete membership");
                found=i;
            }
            if(found==members.size())throw std::invalid_argument("Original source group lost a declared channel");
            for(const auto prior:order)if(prior==found)
                throw std::invalid_argument("Original source group repeats a declared channel");
            order.push_back(found);
        }
        return order;
    }

    friend struct ::llaminar2::test::PooledImmutableFactoryReplayAccess;
    /**
     * @brief Original factory with exact preclaimed source and boundary ordering groups.
     * @param service Original retained provider/channel. @param geometry Checked original source geometry.
     * @param owners Original ordered cache roots. @param coordinator Original policy roots.
     * @param authority Same final PMA. @param stream Exact setup producer for all original edges.
     * @param sourceGroup Closed source-root claim, empty for installed public selection.
     * @param boundaryGroup Closed submission-frontier claim, empty for installed public selection.
     * @return Original retained pooled boundary with unchanged initialization and retirement.
     * @throws std::exception For invalid ownership or any failed setup prefix.
     */
    static std::shared_ptr<ImmutableLookupExecutionBoundary> createPooledImpl(std::shared_ptr<ImmutableLookupService> service,
        PooledImmutableSourceGeometry geometry,std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> authority,
        void* stream,std::unique_ptr<NativeOrderingEventGroup> sourceGroup,std::unique_ptr<NativeOrderingEventGroup> boundaryGroup);

    friend class PooledImmutableCompleteCohort;
    friend class PooledImmutableRetainedParticipants;
    /** @brief Publish one checked complete original cohort before any service arm. */
    void bindCompleteCohort(const std::shared_ptr<PooledImmutableCompleteCohort>& cohort);
    /** @brief Publish only the checked model-owned recurring participant authority. */
    void bindRetainedParticipants(const std::shared_ptr<PooledImmutableRetainedParticipants>&);
    /** @brief Lock the complete owner through each distributed lifecycle operation. */
    std::shared_ptr<PooledImmutableRetirementAuthority> requireCompleteCohort()const;
    /** @brief Reject changed original tensor/native device/capacity/backend generation before any operation. */
    void requireProtocolOwner() const;
    std::atomic_flag cohortOwnerClaim_=ATOMIC_FLAG_INIT; ///< One transferable CPU ownership claim, held through final native retirement.
    std::atomic<std::weak_ptr<PooledImmutableRetirementAuthority>> completeCohort_; ///< Session owns execution; weak back-edge avoids a graph/boundary cycle.
    std::shared_ptr<PhysicalMemoryAuthority> originalMemory_; ///< Original mutable authority, retained without a copied ledger.
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
        std::shared_ptr<TensorBase> protocol,Pooled pooled,std::shared_ptr<PhysicalMemoryAuthority> originalMemory);
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
    /** @brief Parent-issued original channel ticket, staged before this device's native admission writer runs. */
    struct PendingGroupAdmission
    {
        ImmutableLookupService::Ticket ticket;
        std::uint64_t request_epoch=0;
        ImmutableLookupServiceStats start;
    };
    std::optional<PendingGroupAdmission> pending_group_admission_;
    std::uint64_t requestEpoch_=0,intervalGeneration_=0; ///< Host-owned immutable I/O admission identity only.
    ImmutableLookupAdmissionArgs::ProtocolInitialization initialization_=ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce;
    std::atomic<bool> armed_{false};
    RetainedStorageGenerationControl control_;
    ImmutableLookupServiceStats intervalStart_; ///< Snapshot of I/O counters before the worker starts.
    std::optional<RetainedStorageTerminalObservation> terminal_; ///< Immutable only after worker join.
};
} // namespace llaminar2::qwen4exp
