/**
 * @file NativeContextPreparation.h
 * @brief Original-PMA source preparation and retained selected worker birth authority.
 *
 * A source session owns the original Preparing ledger, exact stack roots and
 * source-qualified opaque obligations. Pool publication retains that same
 * authority until the existing exclusive physical runtime-reset transaction.
 */
#pragma once
#include "NativeContextOpaqueContract.h"
#include "GPUDeviceContextPool.h"
#include "PreparedGPUWorkerThread.h"
#include "collective/NativeStoragePreparationPmaBinding.h"
#include <span>
#include <vector>
#include <optional>
namespace llaminar2 {
namespace test {struct NativeContextBirthDiagnosticAccess;}
class GPUDeviceContextPool;
class IWorkerGPUContext;
class TransferEngine;
/** @brief Original physical obligations for one selected source worker, not an ownership mirror. */
class PreparedGPUContextBirth final {
public:
    /** @return Exact source-qualified selected physical endpoint. */
    DeviceId device() const noexcept {return contract_.key().device;}
    /** @return Original source authority; no second ledger is constructed for this birth. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const noexcept{return source_->memory();}
private:
    friend class NativeContextPreparation;
    friend class GPUDeviceContextPool;
    friend struct test::NativeContextPreparationReplayAccess;
    /** @brief Claim all physical obligations before any original native creator executes. */
    PreparedGPUContextBirth(std::shared_ptr<NativeStoragePreparationPmaBinding>,const NativeContextOpaqueContract&,GPUWorkerStackProgram);
    /** @brief Mark the original canonical obligation active before native initialization. */
    void begin();
    /** @brief Publish only after the original constructor has completed its initialization contract. */
    void created();
    /** @brief Release only after original worker destruction and validated physical runtime reset. */
    void physicallyRetired();
    std::shared_ptr<NativeStoragePreparationPmaBinding> source_;
    const NativeContextOpaqueContract contract_;
    std::shared_ptr<PhysicalMemoryOpaqueRuntimeCommitment> opaque_;
    std::shared_ptr<PreparedGPUWorkerStack> stack_;
};
/** @brief Exact original worker borrow retaining its certified original source authority. */
class PreparedGPUContextUsePermit final {
public:
    /** @return Actual original pool worker; callers cannot substitute a replacement context. */
    const std::shared_ptr<IWorkerGPUContext>& worker()const noexcept{return provenance_.worker();}
    /** @brief Reject a substituted pointer or independent shared owner before native family creation. */
    void requireWorker(const std::shared_ptr<IWorkerGPUContext>& candidate)const {provenance_.requireWorker(candidate);}
    /** @return Same canonical generic pool capability, additionally bound to this source's original PMA. */
    const GPUContextProvenance& provenance()const noexcept{return provenance_;}
    /** @return Same original source-admission PMA. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const noexcept{return birth_->memory();}
    /** @return Exact selected physical device. */
    DeviceId device()const noexcept{return birth_->device();}
    /** @return Original prebirth-assigned pool generation. */
    std::uint64_t generation()const noexcept{return provenance_.generation();}
private:
    friend class GPUDeviceContextPool;
    PreparedGPUContextUsePermit(std::shared_ptr<PreparedGPUContextBirth> birth,GPUContextProvenance provenance)
        :birth_(std::move(birth)),provenance_(std::move(provenance)){}
    std::shared_ptr<PreparedGPUContextBirth> birth_;
    GPUContextProvenance provenance_;
};
/** @brief Cold source owner consumed by original pool birth and final same-state admission. */
class NativeContextPreparation final {
public:
    /** @brief Prepare exactly the declared selected worker population using installed certificates. */
    static std::shared_ptr<NativeContextPreparation> create(std::span<const PhysicalMemoryResource>,int,std::span<const DeviceId>);
    /** @brief Retain the selected original worker; missing/foreign source identities reject before creation. */
    PreparedGPUContextUsePermit acquire(DeviceId);
    /** @return Original physical authority for planning and real source-loader handoff. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const noexcept{return source_->memory();}
    /** @brief Close exact creator roots; no later source growth may be introduced. */
    NativePreparationPlanningSnapshot closeForPlanning();
    /** @brief Close the original snapshot internally; Runner never opens a raw source binding/epoch. */
    void freezeForPlanning();
    /** @brief Admit and seal the original worker family against the complete canonical certificate. */
    void admitFinal(std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>);
    /** @brief Reset only this source's installed unadmitted births through original native-only retirement. */
    void retireUnadmitted();
    /** @brief Retire remaining original births only after all model/graph owners are released.
     * @details Existing native-only TransferEngine tickets require zero canonical allocations,
     * exclude acquisitions, destroy every original worker, then authenticate physical reset.
     * Context/process reservations keep their original certified retirement lifetime.
     * @throws std::logic_error For live worker aliases or an inactive original source.
     * @throws std::runtime_error For live allocations or incomplete backend reset evidence.
     */
    void retireReleasedContexts();
    /** @return Original frozen declaration merged before optional model/table/expert capacity; never a live byte ledger. */
    const PhysicalMemoryPlan& mandatoryBOM()const;
    /** @brief Require membership in the originally selected worker population without creating or extending it. */
    void requireSelectedDevice(DeviceId)const;
    /** @return Original immutable allocator observation, never a new driver/free-memory sample. */
    const PhysicalMemoryResource& originalResource(DeviceId)const;
    /** @brief Publish the original Preparing state with the complete model/native admission plan. */
    std::shared_ptr<NativeStoragePmaBinding> admit(NativePreparationPlanningSnapshot&&,std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>);
private:
    friend struct test::NativeContextPreparationReplayAccess;
    NativeContextPreparation(std::span<const PhysicalMemoryResource>,int);
    /** @brief Same original retirement body with an explicit infrastructure transfer authority for device-free replay. */
    void retireReleasedContexts(const TransferEngine&);
    static std::shared_ptr<NativeContextPreparation> createQualified(std::span<const PhysicalMemoryResource>,int,std::span<const NativeContextBirthKey>);
    std::shared_ptr<NativeStoragePreparationPmaBinding> source_;
    std::vector<std::shared_ptr<PreparedGPUContextBirth>> births_;
    std::optional<NativePreparationPlanningSnapshot> planning_snapshot_;
    std::shared_ptr<NativeStoragePmaBinding> admitted_binding_;
    std::shared_ptr<const PhysicalMemoryPlan> mandatory_bom_; ///< Immutable source declaration only; all live claims remain in memory().
};
/** @brief Actual compiled native context object declaration; no native runtime call is needed. */
class PreparedGPUContextObjectGeometry final {
public:
    std::size_t bytes()const noexcept{return bytes_;}
    std::size_t alignment()const noexcept{return alignment_;}
private:
    friend class PreparedGPUContextFactory;
    PreparedGPUContextObjectGeometry(std::size_t bytes,std::size_t alignment):bytes_(bytes),alignment_(alignment){}
    std::size_t bytes_,alignment_;
};
/** @brief Backend bridge alone can invoke the original private prepared constructors. */
class PreparedGPUContextFactory final {
    friend struct test::NativeContextPreparationReplayAccess;
public:
    /** @brief Read the selected compiled sizeof/alignof declaration; never discover or initialize a device. */
    static PreparedGPUContextObjectGeometry objectGeometry(DeviceId);
    /** @brief Project actual selected runtime/software/hardware identity after immutable metadata startup. */
    static NativeContextBirthKey birthKey(DeviceId);
private:
    friend struct test::NativeContextBirthDiagnosticAccess;
    static NativeContextBirthKey cudaBirthKey(int);
    static NativeContextBirthKey rocmBirthKey(int);
    static NativeContextBirthKey commonBirthKey(DeviceId);
    static std::string libraryIdentity(const void*);
    /** @brief Hash the exact already-loaded DT_NEEDED closure without loading a new library. */
    static std::string libraryClosureIdentity(const void*);
    static std::string digestIdentity(const std::string&);
    static PreparedGPUContextObjectGeometry cudaObjectGeometry();
    static PreparedGPUContextObjectGeometry rocmObjectGeometry();
    friend void ensureNvidiaPreparedFactoryRegistered();
    friend void ensureAMDPreparedFactoryRegistered();
    /** @brief Construct the original CUDA worker on its preclaimed exact source stack. */
    static std::shared_ptr<IWorkerGPUContext> cuda(int,std::shared_ptr<PreparedGPUWorkerStack>);
    /** @brief Construct the original HIP worker on its preclaimed exact source stack. */
    static std::shared_ptr<IWorkerGPUContext> rocm(int,std::shared_ptr<PreparedGPUWorkerStack>);
};
}
