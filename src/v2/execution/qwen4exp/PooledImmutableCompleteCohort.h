/**
 * @file PooledImmutableCompleteCohort.h
 * @brief Original complete reader/retirement ownership at source admission.
 *
 * This cold binding records actual participant graphs through their sole original executable owners,
 * workers and original channel/root holders. It has no request decisions or
 * live physical accounting: its pure event-program BOM feeds the original PMA,
 * which remains canonical together with the original boundary/service.
 * A session must retain it through terminal join. Destroying the last binding
 * while that original interval is active terminates before any member releases.
 */
#pragma once
#include "PooledImmutableRetirementAuthority.h"
#include "PooledImmutableParticipantRetirementGraph.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "backends/NativeOrderingEventGroup.h"
#include <optional>
#include "execution/local_execution/device/DeviceContext.h"
namespace llaminar2::test {struct PooledImmutableCompleteCohortReplayAccess;}
namespace llaminar2::qwen4exp {
class ImmutableLookupSubmissionRetirement;
/** @brief Original participant declarations retain the closed native owner, never a raw capture or caller stamp. */
struct PooledImmutableCohortParticipant {
    std::shared_ptr<IWorkerGPUContext> worker;
    std::shared_ptr<IDeviceContext> context;
    std::shared_ptr<BufferArena> arena;
    std::shared_ptr<ComputeGraph> graph;
    void* stream=nullptr;
    std::shared_ptr<RetainedParticipantGraphExecutable> executable; ///< Sole original prepared-PMA/native definition authority; no raw capture injection.

};
/** @brief Exact remote owner terminal tail, in original physical coordinate order. */
struct PooledImmutableCohortRemote {
    std::uint32_t coordinate;
    PooledImmutableCohortParticipant participant;
    std::string retiredSent;
};
/** @brief Checked complete original source/owner execution lifetime, not a native factory certificate. */
class PooledImmutableCompleteCohort final:public PooledImmutableRetirementAuthority {
public:
    /**
     * @brief Price this legacy capture owner's additional retirement event per remote.
     * @param host Original CPU allocator observation.
     * @param remotes Exact ordered remote GPU observations; source is excluded.
     * @return Fixed one-event programs, separate from root initialization and source boundary.
     */
    static PhysicalMemoryPlan submissionRetirementBOM(PhysicalMemoryResource host,
        std::span<const PhysicalMemoryResource> remotes);
    /**
     * @brief Validate and record only the actual original source and remote terminal programs.
     * @param source Original source graph/arena/context and exact owning worker stream.
     * @param completion Actual authenticated Read→last-reader→physical-retirement descriptor.
     * @param remotes Every nonlocal original owner once, in physical coordinate order.
     * @return Strong session owner; the original boundary retains only a weak reference.
     * @note The CPU coordinator must be outside every participating original
     * worker; final destruction follows the same rule before peer retirement.
     * This original owner performs cold stage preparation and actual
     * DeviceGraphExecutor recording before first admission. No public executable
     * injection or caller-owned provenance stamp can authorize an interval.
     * Opaque-runtime and complete-model admission remain separate contracts.
     */
    static std::shared_ptr<PooledImmutableCompleteCohort> bind(PooledImmutableCohortParticipant source,
        PooledImmutableRetirementCompletion completion,std::vector<PooledImmutableCohortRemote> remotes);
    /** @brief Submit only this fixed original captured cohort; partial native launch failure is fatal. */
    void launch()const;
    /** @return Actual source kernel attributes, queried on its original worker outside admission/capture. */
    std::vector<GPUGraphKernelNodeInfo> sourceKernelInventory()const;
    /** @return Actual addressed remote-owner kernel attributes, never mutable capture access. */
    std::vector<GPUGraphKernelNodeInfo> ownerKernelInventory(std::uint32_t coordinate)const;
    /** @brief Reject original graph, prepared stream, worker, capture or root identity drift. */
    void requireCurrent()const override;
    /** @brief Authenticate the exact original source storage and retained worker stream for typed reset. */
    void requireSource(const PooledImmutableSourcePreparedStorage& storage,void* stream)const override;
    /** @brief Authenticate the existing fixed source stream; legacy component graphs cannot adopt a new parent. */
    void bindSourceSubmissionStream(const RetainedGraphParticipantParentStream&)override;
    /** @brief Observe original submitted handles only at the existing whole-request terminal boundary. */
    void awaitNativeTerminal(std::chrono::steady_clock::time_point)const override;
    /** @brief Fail fatally before releasing active operands; retire cold participant frontiers otherwise. */
    ~PooledImmutableCompleteCohort() noexcept override;
    PooledImmutableCompleteCohort(const PooledImmutableCompleteCohort&)=delete;
    PooledImmutableCompleteCohort& operator=(const PooledImmutableCompleteCohort&)=delete;
private:
    friend class ImmutableLookupExecutionBoundary;
    friend struct llaminar2::test::PooledImmutableCompleteCohortReplayAccess;
    /** @brief Replay the same original binding with exact preclaimed original remote event programs; no catalog changes. */
    static std::shared_ptr<PooledImmutableCompleteCohort> bindWithOrderingGroups(PooledImmutableCohortParticipant,
        PooledImmutableRetirementCompletion,std::vector<PooledImmutableCohortRemote>,
        std::vector<std::unique_ptr<NativeOrderingEventGroup>>);
    /** @brief Share installed and sealed recorded inputs through the one original capture/ownership implementation. */
    static std::shared_ptr<PooledImmutableCompleteCohort> bindImpl(PooledImmutableCohortParticipant,
        PooledImmutableRetirementCompletion,std::vector<PooledImmutableCohortRemote>,
        std::optional<std::vector<std::unique_ptr<NativeOrderingEventGroup>>>);
    struct State;
    struct OwnershipClaim;
    /** @brief Run the claimed cold transaction from a caller outside all original participant workers. */
    static std::shared_ptr<PooledImmutableCompleteCohort> bindClaimed(PooledImmutableCohortParticipant source,
        PooledImmutableRetirementCompletion completion,std::vector<PooledImmutableCohortRemote> remotes,
        std::unique_ptr<OwnershipClaim> ownership,
        std::optional<std::vector<std::unique_ptr<NativeOrderingEventGroup>>> ordering);
    /** @brief Construct only after the checked original graph inventory is complete. */
    explicit PooledImmutableCompleteCohort(std::unique_ptr<OwnershipClaim> ownership,std::unique_ptr<State> state);
    /** @brief Acquire immutable original remote setup/input producers once during off-worker cold binding. */
    void acquireRemoteSetup()const;
    /** @brief Retire all potentially submitted remote frontiers after partial failure. */
    void retireFailedSetup()const noexcept;
    /** @brief Query one original private capture without exposing its mutable executable. */
    std::vector<GPUGraphKernelNodeInfo> kernelInventory(std::optional<std::uint32_t> coordinate)const;
    /** @brief Require the original source worker stream, without creating a context. */
    void requireSourceStream(void* stream)const override;
    std::unique_ptr<OwnershipClaim> ownership_; ///< Releases only after State and all native aliases/frontiers are destroyed.
    std::unique_ptr<State> state_;
};
}
