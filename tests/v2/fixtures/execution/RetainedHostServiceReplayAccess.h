/** @file RetainedHostServiceReplayAccess.h
 * @brief Sealed recorded libc/TLS program inputs; actual admitted allocation/thread/state bodies execute.
 */
#pragma once
#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
namespace llaminar2::test {
/** @brief The sole device-free recorded certificate bridge, never a production selector. */
struct RetainedHostServiceReplayAccess final {
    /** @return Sealed recorded original-rank budget metadata for candidate replay only. */
    static RetainedHostServiceBudgetProjection projection(std::vector<RetainedHostServiceProgram> programs){
        for(const auto& p:programs)if(p.evidenceKind()!=RetainedHostServiceEvidence::Recorded)throw std::invalid_argument("Recorded projection requires recorded role evidence");
        return RetainedHostServiceBudgetProjection(std::move(programs));
    }
    /** @return Explicit diagnostic-only serial-reuse policy with recorded loader evidence. */
    static RetainedHostServiceProgram diagnosticProgram(RetainedHostServiceRole role){return RetainedHostServiceProgram::recordedDiagnostic(role);}
    /** @return Actual small loader provider evidence without hashing passive TLS modules. */
    static std::shared_ptr<const RetainedHostServiceProgram::TLSProviderEvidence> tlsProvider(){return RetainedHostServiceProgram::originalTLSProviderEvidence();}
    /** @brief Replay the actual provider validator with a recorded selected-address observation. */
    static void requireTLSProvider(const decltype(tlsProvider())& evidence,const void* selected){RetainedHostServiceProgram::requireTLSProviderEvidence(evidence,selected);}
    /** @return Recorded closed role with independently authored opaque input, not installed data. */
    static RetainedHostServiceUseToken claim(std::shared_ptr<PhysicalMemoryAuthority> memory,RetainedHostServiceProgram program){const RetainedHostServiceInstance key{memory->worldRank(),DeviceId::cpu(),program.role()};return RetainedHostServiceUseToken::claim(std::move(memory),std::move(program),key);}
    /** @return Recorded numeric input with actual original loader/default evidence. */
    static RetainedHostServiceProgram withEvidence(RetainedHostServiceRole role,std::shared_ptr<const RetainedHostServiceProgram::Evidence> evidence,std::size_t opaque=4096){return RetainedHostServiceProgram::recordedWithEvidence(role,std::move(evidence),opaque);}
    static RetainedHostServiceProgram program(RetainedHostServiceRole role,std::size_t opaque=4096){return RetainedHostServiceProgram::recorded(role,opaque);}
};
}
