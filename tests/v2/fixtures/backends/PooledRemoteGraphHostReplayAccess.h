/**
 * @file PooledRemoteGraphHostReplayAccess.h
 * @brief Closed recorded source/host selection for the actual retained remote owner.
 *
 * Toy replay bounds are selector inputs only, never installed certification.
 * Raw native diagnostics are unavailable without their explicit target macro.
 */
#pragma once
#include "backends/RetainedParticipantGraphExecutable.h"
namespace llaminar2::test {
/** @brief Access the same selector/BOM/creator/retirement implementation with recorded evidence. */
struct PooledRemoteGraphHostReplayAccess final {
    /** @return Actual selector result for the recorded original source/program, with no catalog override. */
    static NativeAuxiliaryResourceSelection selection(const PreparedGPUContextUsePermit& original,const PooledRemoteGraphProgram& program,std::size_t hostUnitBytes=4096){
        const auto key=original.provenance().originalBirthKey();
        const auto resource=NativeAuxiliaryResourceCatalog::resourceSourceIdentity();
        const std::string evidence(64,'b');
        const NativeAuxiliaryRemoteFamilyScope scope{PooledRemoteGraphProgramClass::familyVersion(),program.limits(),
            PooledRemoteGraphProgramClass::argumentLayout()};
        const detail::NativeAuxiliaryQualifiedClass row{key->device.type,
            NativeAuxiliaryResourceProgram::pooledRemoteOwner(program),hostUnitBytes,0,evidence,NativePhysicalLifetime::ProcessRuntime,scope};
        return NativeAuxiliaryResourceCatalog::select(*key,original.generation(),row.program,resource,{&row,1});
    }
    /** @return Same actual selector with a separately recorded representative and immutable tested family envelope. */
    static NativeAuxiliaryResourceSelection familySelection(const PreparedGPUContextUsePermit& original,
        const PooledRemoteGraphProgram& requested,const PooledRemoteGraphProgram& representative,NativeAuxiliaryRemoteFamilyScope scope){
        const auto key=original.provenance().originalBirthKey();
        const auto resource=NativeAuxiliaryResourceCatalog::resourceSourceIdentity();const std::string evidence(64,'b');
        const detail::NativeAuxiliaryQualifiedClass row{key->device.type,
            NativeAuxiliaryResourceProgram::pooledRemoteOwner(representative),4096,0,evidence,NativePhysicalLifetime::ProcessRuntime,scope};
        return NativeAuxiliaryResourceCatalog::select(*key,original.generation(),NativeAuxiliaryResourceProgram::pooledRemoteOwner(requested),resource,{&row,1});
    }
    /** @return Same known/recorded opaque BOM before final admission, not an invented production class. */
    static PhysicalMemoryPlan bom(const PooledRemoteGraphProgram& program,PhysicalMemoryResource host,PhysicalMemoryResource gpu,
        const std::optional<NativeAuxiliaryResourceSelection>& selected,std::size_t recordedObjectBytes){
        return RetainedParticipantGraphExecutable::hostBOMWithSelection(program,host,gpu,selected,recordedObjectBytes);
    }
    /** @return Actual original creator using a recorded selector result and exact fake-runtime object extent. */
    static auto create(const PreparedGPUContextUsePermit& original,const PooledRemoteGraphProgram& program,
        NativeAuxiliaryResourceSelection selected,std::size_t recordedObjectBytes){
        return RetainedParticipantGraphExecutable::createRecordedRemote(original,original.worker()->defaultStream(),program,std::move(selected),recordedObjectBytes);
    }
    /** @return Original recording facade, retained solely to verify its one-shot lifecycle. */
    static IGPUGraphCapture& recording(RetainedParticipantGraphExecutable& owner){return owner.capture();}
    /** @return Actual owner-controlled enqueue, never a direct capture launch. */
    static bool submit(RetainedParticipantGraphExecutable& owner,std::uint64_t ordinal){return owner.submit(ordinal);}
    /** @return Exact original terminal observation. */
    static bool observe(RetainedParticipantGraphExecutable& owner){return owner.observeTerminal();}
    /** @brief Close actual handles while metadata aliases may remain alive. */
    static void retire(RetainedParticipantGraphExecutable& owner){owner.retireDefinition();}
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    /** @return Explicit unqualified measurement through the same source descriptor/native inspection body. */
    static auto raw(const PreparedGPUContextUsePermit& original,const PooledRemoteGraphProgram& program){
        return RetainedParticipantGraphExecutable::createDiagnosticRemote(original,original.worker()->defaultStream(),program);
    }
#endif
    /** @brief Exercise only the original owner's closed graph-only storage seal. */
    static void sealStorage(RetainedParticipantGraphExecutable& owner){owner.sealRecordedStorageBody();}
    /** @brief Claim the original fresh definition before native recording, as production does. */
    static void claimStorage(RetainedParticipantGraphExecutable& owner){owner.bindCohortDefinition();}
    /** @return The original owner primitive; no public raw native definition escapes. */
    static auto storageBody(RetainedParticipantGraphExecutable& owner,const PreparedGPUContextUsePermit& parent,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation){
        return owner.retainRecordedStorageDefinition(parent,std::move(preparation));
    }

};
}
