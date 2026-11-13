/** @file Measure__RetainedHostService.cpp
 * @brief Uninstrumented original CPU service key and actual constructor/task/join resource diagnostic.
 *
 * Exact admitted mappings are separated from unqualified outside-root libc/TLS
 * backing. Passive parent observation owns process-exit evidence; this child
 * never installs numerical limits or claims whole-process byte completeness.
 */
#include "fixtures/execution/RetainedHostServiceCalibrationAccess.h"
#include "execution/local_execution/graph/RetainedHostServiceClassPolicy.h"
#include "execution/local_execution/graph/RetainedParentTicketServiceWorker.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <sys/sysinfo.h>
#include <malloc.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string_view>
#include <optional>
#include <charconv>
#include "NativeHostServicePopulationDiagnostic.h"
namespace llaminar2::test {
/** @brief Emit only observations, never convert a delta to capacity or a certificate. */
void checkpoint(std::string_view phase,const RetainedHostServiceProgram& program,const std::shared_ptr<PhysicalMemoryAuthority>& memory,unsigned lifetime,const RetainedHostServiceProgram::EvidenceObservation* evidence=nullptr){
    const auto heap=::mallinfo2();nlohmann::json row{{"schema","retained-host-service-diagnostic-v1"},{"phase",phase},{"role",program.role()==RetainedHostServiceRole::ImmutableLookupIO?"immutable_lookup_io":"executor_parent_ticket"},{"lifetime",lifetime},{"identity",program.identity()},{"resource_class",program.resourceIdentity()},{"evidence_role","unqualified_actual_program"},{"whole_process_bytes_complete",false},{"opaque_limit_installed",false},{"exact_stack_bytes",program.stackBytes()},{"virtual_guard_bytes",program.guardBytes()},{"exact_allocation_bytes",program.allocationBytes()},{"malloc_uordblks",heap.uordblks},{"malloc_hblkhd",heap.hblkhd}};
    // Historical field names the closed resource ABI, never application source.
    row["creator_source"]=std::string("mapped-pthread-v")+std::to_string(static_cast<unsigned>(::llaminar2::detail::retainedHostServiceCreatorVersion));
    row["diagnostic_source"]=HOST_SERVICE_DIAGNOSTIC_SOURCE_SHA256;
    if(evidence)row["cold_evidence"]={{"files_hashed",evidence->filesHashed},{"bytes_hashed",evidence->bytesHashed},{"elapsed_ns",evidence->elapsedNanoseconds}};
    if(memory)row["exact_claimed_bytes"]=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation);
    std::cout<<row.dump()<<std::endl;
}
}
int main(int argc,char** argv){
    using namespace llaminar2;using namespace llaminar2::test;
    try{
        enum class Mode{Key,Lifecycle,Population,Claims};
        std::optional<Mode> selectedMode;std::optional<RetainedHostServiceRole> selectedRole;
        std::optional<unsigned> io,ticket;
        const auto count=[](std::string_view value){unsigned result=0;const auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);if(parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size()||result>16)throw std::invalid_argument("CPU diagnostic worker count must be0..16");return result;};
        for(int i=1;i<argc;++i){
            const std::string_view a=argv[i];
            if(a.starts_with("--mode=")){
                if(selectedMode)throw std::invalid_argument("CPU service mode must be specified exactly once");
                if(a=="--mode=key")selectedMode=Mode::Key;
                else if(a=="--mode=diagnostic-lifecycle")selectedMode=Mode::Lifecycle;
                else if(a=="--mode=diagnostic-population")selectedMode=Mode::Population;
                else if(a=="--mode=diagnostic-claims")selectedMode=Mode::Claims;
                else throw std::invalid_argument("Unknown CPU diagnostic mode");
            }else if(a=="--role=io"||a=="--role=executor-ticket"){
                if(selectedRole)throw std::invalid_argument("CPU service role must be specified exactly once");
                selectedRole=a=="--role=io"?RetainedHostServiceRole::ImmutableLookupIO:RetainedHostServiceRole::ExecutorParentTicket;
            }else if(a.starts_with("--io-workers=")){
                if(io)throw std::invalid_argument("CPU IO population repeated");io=count(a.substr(13));
            }else if(a.starts_with("--executor-workers=")){
                if(ticket)throw std::invalid_argument("CPU executor population repeated");ticket=count(a.substr(19));
            }else throw std::invalid_argument("Unknown CPU service diagnostic argument");
        }
        if(!selectedMode)throw std::invalid_argument("Explicit --mode is required");
        if(*selectedMode==Mode::Population||*selectedMode==Mode::Claims){
            if(selectedRole||!io||!ticket||*io+*ticket==0||*io+*ticket>16)throw std::invalid_argument("Population modes require explicit IO/executor counts totaling1..16 and no --role");
            return runHostServicePopulation(*io,*ticket,*selectedMode==Mode::Claims?HostServicePopulationWork::ClaimsOnly:HostServicePopulationWork::AllModuleTLS);
        }
        if(!selectedRole||io||ticket)throw std::invalid_argument("Key/single lifecycle require one --role and no population flags");
        GPUDeviceProcessLifetime process;
        const bool keyOnly=*selectedMode==Mode::Key;const auto role=*selectedRole;
        const auto evidence=RetainedHostServiceProgram::originalEvidence();
        const auto observation=RetainedHostServiceProgram::evidenceObservation(evidence);
        const auto program=RetainedHostServiceCalibrationAccess::program(role,evidence);checkpoint("original_metadata_complete",program,{},0,&observation);
        if(keyOnly){checkpoint("key_only_complete",program,{},0);return 0;}
        struct sysinfo observed{};if(::sysinfo(&observed))throw std::runtime_error("Original host observation unavailable");
        const auto total=static_cast<std::size_t>(observed.totalram)*observed.mem_unit;
        const auto free=static_cast<std::size_t>(observed.freeram)*observed.mem_unit;
        const PhysicalMemoryResource cpu{0,DeviceId::cpu(),total,free};
        const RetainedHostServiceInstance key{0,DeviceId::cpu(),role};RetainedHostServiceDeclarations::Builder declare(0);declare.add(key,program);const auto declarations=std::move(declare).freeze();
        auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(declarations->bom(key,cpu)),0);
        for(unsigned i=1;i<=21;++i){
            checkpoint("before_exact_claim",program,memory,i);auto binding=declarations->bind(memory,std::array{key});
            auto token=binding->take(key);checkpoint("original_claim_complete",program,memory,i);
            auto worker=PreparedRetainedHostService::materialize(std::move(token));checkpoint("original_worker_ready",program,memory,i);
            unsigned count=0;const auto increment=[](void* p){++*static_cast<unsigned*>(p);return true;};
            for(unsigned request=0;request!=20;++request){const auto ticket=worker->worker().dispatch(increment,&count);if(!worker->worker().await(ticket))throw std::runtime_error("Original CPU service task failed");}
            if(count!=20)throw std::runtime_error("Original CPU service exact task count changed");checkpoint("original_worker_used",program,memory,i);
            worker.reset();binding.reset();checkpoint("exact_join_unmap_complete",program,memory,i);
        }
        process.retire();checkpoint("original_process_use_closed",program,memory,21);return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
