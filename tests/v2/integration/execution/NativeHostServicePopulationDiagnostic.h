/** @file NativeHostServicePopulationDiagnostic.h
 * @brief Concurrent original CPU worker/TLS population and zero-native bookkeeping controls.
 *
 * Exact mapped roots use the ordinary admitted PMA. Opaque limits remain absent;
 * no observation is converted to a reservation or a production certificate.
 */
#pragma once
#include "NativeHostServiceTLSProgram.h"
#include "NativeHostServicePopulationRelease.h"
#include "fixtures/execution/RetainedHostServiceCalibrationAccess.h"
#include "execution/local_execution/graph/RetainedHostServiceClassPolicy.h"
#include "execution/local_execution/graph/RetainedParentTicketServiceWorker.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <sys/sysinfo.h>
#include <malloc.h>
#include <nlohmann/json.hpp>
#include <array>
#include <barrier>
#include <latch>
#include <memory>
#include <iostream>
#include <vector>
namespace llaminar2::test {
/** @brief Different closed diagnostic operations, not production role or admission modes. */
enum class HostServicePopulationWork {ClaimsOnly,AllModuleTLS};
/** @brief Emit cold immutable observations, never accounting decisions. */
inline void populationCheckpoint(const char* phase,unsigned lifetime,unsigned io,unsigned ticket,
    HostServicePopulationWork work,const std::vector<RetainedHostServiceProgram>& programs,
    const std::shared_ptr<PhysicalMemoryAuthority>& memory,const NativeHostServiceTLSProgram& tls,
    const RetainedHostServiceProgram::EvidenceObservation* evidence=nullptr){
    const auto heap=::mallinfo2();
    nlohmann::json classes=nlohmann::json::array();
    for(const auto& program:programs)classes.push_back({{"role",program.role()==RetainedHostServiceRole::ImmutableLookupIO?"immutable_lookup_io":"executor_parent_ticket"},{"identity",program.identity()},{"resource_class",program.resourceIdentity()},{"exact_allocation_bytes",program.allocationBytes()},{"exact_stack_bytes",program.stackBytes()},{"virtual_guard_bytes",program.guardBytes()}});
    nlohmann::json row{{"schema","retained-host-service-population-v1"},{"phase",phase},{"lifetime",lifetime},
        {"io_workers",io},{"executor_workers",ticket},{"creator_source",std::string("mapped-pthread-v")+std::to_string(static_cast<unsigned>(::llaminar2::detail::retainedHostServiceCreatorVersion))},
        {"diagnostic_source",HOST_SERVICE_DIAGNOSTIC_SOURCE_SHA256},{"classes",std::move(classes)},
        {"work",work==HostServicePopulationWork::ClaimsOnly?"zero_native_claim_control":"all_original_tls_blocks"},
        {"actual_consumers_qualified",false},{"opaque_limit_installed",false},{"whole_process_bytes_complete",false},
        {"tls_modules",tls.modules()},{"tls_memory_bytes_per_worker",tls.bytes()},
        {"loader_adds",tls.loaderAdds()},{"loader_subs",tls.loaderSubs()},
        {"malloc_uordblks",heap.uordblks},{"malloc_hblkhd",heap.hblkhd}};
    if(evidence)row["cold_evidence"]={{"files_hashed",evidence->filesHashed},{"bytes_hashed",evidence->bytesHashed},{"elapsed_ns",evidence->elapsedNanoseconds}};
    if(memory)row["exact_claimed_bytes"]=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation);
    std::cout<<row.dump()<<std::endl;
}
/** @brief Run one explicit population on actual ordinary owners; no synthetic TLS storage. */
inline int runHostServicePopulation(unsigned io,unsigned ticket,HostServicePopulationWork work){
    if(io>16||ticket>16||io+ticket==0||io+ticket>16)throw std::invalid_argument("Explicit CPU diagnostic population must contain1..16 workers");
    GPUDeviceProcessLifetime process;
    const auto evidence=RetainedHostServiceProgram::originalEvidence();
    const auto observation=RetainedHostServiceProgram::evidenceObservation(evidence);
    std::vector<RetainedHostServiceProgram> programs;
    if(io)programs.push_back(RetainedHostServiceCalibrationAccess::program(RetainedHostServiceRole::ImmutableLookupIO,evidence));
    if(ticket)programs.push_back(RetainedHostServiceCalibrationAccess::program(RetainedHostServiceRole::ExecutorParentTicket,evidence));
    for(const auto& program:programs)RetainedHostServiceCalibrationAccess::requireCurrent(program);
    const auto tls=NativeHostServiceTLSProgram::original();
    for(const auto& program:programs)RetainedHostServiceCalibrationAccess::requireCurrent(program);
    populationCheckpoint("original_metadata_complete",0,io,ticket,work,programs,{},tls,&observation);
    struct sysinfo original{};if(::sysinfo(&original))throw std::runtime_error("Original CPU observation unavailable");
    const auto total=static_cast<std::size_t>(original.totalram)*original.mem_unit;
    const auto available=static_cast<std::size_t>(original.freeram)*original.mem_unit;
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),total,available};
    RetainedHostServiceDeclarations::Builder declare(0);PhysicalMemoryPlanBuilder bom;
    std::vector<RetainedHostServiceInstance> keys;
    for(const auto& program:programs){
        const unsigned count=program.role()==RetainedHostServiceRole::ImmutableLookupIO?io:ticket;
        for(unsigned i=0;i<count;++i){
            const RetainedHostServiceInstance key{0,DeviceId::cpu(),program.role(),i};
            declare.add(key,program);keys.push_back(key);const auto contribution=program.bom(cpu);for(const auto& row:contribution.resources())bom.add(row);
        }
    }
    const auto declarations=std::move(declare).freeze();
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(bom.build()),0);
    const auto workers=keys.size();
    for(unsigned lifetime=1;lifetime<=21;++lifetime){
        std::barrier start(static_cast<std::ptrdiff_t>(workers+1));std::latch touched(static_cast<std::ptrdiff_t>(workers)),release(1);
        struct Task {const NativeHostServiceTLSProgram* tls;std::barrier<>* start;std::latch* touched;const std::latch* release;};
        Task task{&tls,&start,&touched,&release};
        auto binding=declarations->bind(memory,keys);
        std::vector<RetainedHostServiceUseToken> tokens;tokens.reserve(workers);
        std::vector<std::unique_ptr<PreparedRetainedHostService>> owners;owners.reserve(workers);
        std::vector<RetainedParentTicketServiceWorker::Ticket> accepted;accepted.reserve(workers);
        populationCheckpoint("before_exact_claim",lifetime,io,ticket,work,programs,memory,tls);
        for(const auto key:keys)tokens.push_back(binding->take(key));
        populationCheckpoint("original_claims_complete",lifetime,io,ticket,work,programs,memory,tls);
        if(work==HostServicePopulationWork::AllModuleTLS){
            for(auto& token:tokens)owners.push_back(PreparedRetainedHostService::materialize(std::move(token)));
            populationCheckpoint("all_original_workers_ready",lifetime,io,ticket,work,programs,memory,tls);
            try{for(const auto& owner:owners)accepted.push_back(owner->worker().dispatch([](void* opaque){
                auto& task=*static_cast<Task*>(opaque);task.start->arrive_and_wait();const bool ok=task.tls->touchAll();task.touched->count_down();task.release->wait();return ok;
            },&task));}catch(...){std::terminate();} // Never unwind owners with tasks blocked at an incomplete start barrier.
            observeLiveHostServicePopulation(start,touched,release,[&]{
                populationCheckpoint("all_original_tls_live",lifetime,io,ticket,work,programs,memory,tls);
            });
            for(std::size_t i=0;i<owners.size();++i)if(!owners[i]->worker().await(accepted[i]))throw std::runtime_error("Original TLS lazy-work task failed");
            populationCheckpoint("all_original_tasks_complete",lifetime,io,ticket,work,programs,memory,tls);
            owners.clear();
        }
        tokens.clear();binding.reset();
        for(const auto& program:programs)RetainedHostServiceCalibrationAccess::requireCurrent(program);
        populationCheckpoint("exact_join_unmap_complete",lifetime,io,ticket,work,programs,memory,tls);
    }
    process.retire();populationCheckpoint("original_process_use_closed",21,io,ticket,work,programs,memory,tls);return 0;
}
}
