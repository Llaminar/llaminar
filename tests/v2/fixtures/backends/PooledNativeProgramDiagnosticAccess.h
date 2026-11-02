/**
 * @file PooledNativeProgramDiagnosticAccess.h
 * @brief Explicit uncertified native resource ownership for the actual-program measurement target only.
 *
 * These constructors mint no certificate, numeric bound, reservation or physical
 * release claim. They use the SAME production creators/submission/destructors.
 * The original prepared source permit supplies every PMA/worker/generation.
 */
#pragma once
#ifndef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
#error "Unqualified native program construction belongs only to its explicit diagnostic target"
#endif
#include "backends/ReadAheadNativeResourceFamily.h"
#include "backends/ReadAheadNativeParentTerminal.h"
#include "backends/NativeAuxiliaryResourceCatalog.h"
#include "backends/NativeContextPreparation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>
namespace llaminar2 {
/** @brief Test-target-only constructor definition: a typed diagnostic state, never an empty certificate. */
inline ReadAheadNativeResourceFamily::ReadAheadNativeResourceFamily(
    std::shared_ptr<PhysicalMemoryAuthority> memory,std::shared_ptr<IWorkerGPUContext> worker,
    ReadAheadNativeResourceRequest request,UnqualifiedDiagnostic diagnostic)
    :memory_(std::move(memory)),worker_(std::move(worker)),request_(std::move(request)),certificate_(diagnostic){}
/** @brief Test-target-only independent parent-event constructor, with no invented commitment. */
inline ReadAheadNativeParentTerminal::ReadAheadNativeParentTerminal(
    std::shared_ptr<PhysicalMemoryAuthority> memory,std::shared_ptr<IWorkerGPUContext> worker,UnqualifiedDiagnostic diagnostic)
    :memory_(std::move(memory)),worker_(std::move(worker)),certificate_(diagnostic){}
namespace test {
struct PooledNativeProgramDiagnosticAccess {
    /** @return Actual original closure; no observation or certificate is fabricated. */
    static ReadAheadNativeResourceClosure closure(const PreparedGPUContextUsePermit& original){
        if(isGraphCaptureActive() || !original.memory() || original.memory()->phase()!=PhysicalMemoryAuthorityPhase::Admitted)
            throw std::invalid_argument("Raw native measurement requires its original admitted source, outside capture");
        const auto actual=GPUDeviceContextPool::instance().retainExistingContextProvenance(original.device());
        actual.requireWorker(original.worker());
        if(actual.generation()!=original.generation())throw std::invalid_argument("Raw native measurement has stale original generation");
        const auto key=actual.originalBirthKey(),retained=original.provenance().originalBirthKey();
        if(key.get()!=retained.get() || key.owner_before(retained) || retained.owner_before(key))
            throw std::invalid_argument("Raw native measurement substituted original birth identity");
        return {original.device(),original.generation(),key->physical_uuid,key->architecture,key->driver_version,
            key->runtime_version,key->backend_sha256,NativeAuxiliaryResourceCatalog::resourceSourceIdentity(),
            NativeAuxiliaryResourceCatalog::liveSourceIdentity(*key)};
    }
    /** @return Exactly one actual side stream and fork/join pair, explicitly unqualified. */
    static std::shared_ptr<ReadAheadNativeResourceFamily> side(const PreparedGPUContextUsePermit& original,void* primary){
        auto actual=closure(original);
        if(reinterpret_cast<std::uintptr_t>(primary)<=2 || primary!=original.worker()->defaultStream())
            throw std::invalid_argument("Raw side family requires the original existing primary stream");
        auto family=std::shared_ptr<ReadAheadNativeResourceFamily>(new ReadAheadNativeResourceFamily(
            original.memory(),original.worker(),{std::move(actual),primary},ReadAheadNativeResourceFamily::UnqualifiedDiagnostic{}));
        family->self_=family;
        try{family->createNativeHandles();}catch(...){family->destroyHandles();throw;}
        return family;
    }
    /** @return Independently labeled exactly-one actual parent terminal event. */
    static std::shared_ptr<ReadAheadNativeParentTerminal> parent(const PreparedGPUContextUsePermit& original){
        (void)closure(original);
        auto parent=std::shared_ptr<ReadAheadNativeParentTerminal>(new ReadAheadNativeParentTerminal(
            original.memory(),original.worker(),ReadAheadNativeParentTerminal::UnqualifiedDiagnostic{}));
        parent->createNativeHandle();return parent;
    }
};
}
}
