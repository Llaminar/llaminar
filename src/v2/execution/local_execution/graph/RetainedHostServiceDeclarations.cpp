/** @file RetainedHostServiceDeclarations.cpp
 * @brief Freeze topology declarations before quotas and consume admitted original instance claims once.
 */
#include "RetainedHostServiceDeclarations.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
RetainedHostServiceBudgetProjection RetainedHostServiceBudgetProjection::original(){
    auto evidence=RetainedHostServiceProgram::originalEvidence();
    std::vector<RetainedHostServiceProgram> programs;
    for(auto role:{RetainedHostServiceRole::ImmutableLookupIO,RetainedHostServiceRole::ExecutorParentTicket})programs.push_back(RetainedHostServiceProgram::original(role,evidence));
    return RetainedHostServiceBudgetProjection(std::move(programs));
}
std::vector<RetainedHostServiceClassProjection> RetainedHostServiceBudgetProjection::classes()const{
    if(!receipts_.empty())return receipts_;
    std::vector<RetainedHostServiceClassProjection> result;for(const auto& program:programs_)result.push_back({program.role(),program.resourceIdentity(),program.identity(),program.resourceDeclaration()});return result;
}
RetainedHostServiceBudgetProjection RetainedHostServiceBudgetProjection::installed(std::span<const RetainedHostServiceClassProjection> receipts){
    std::vector<RetainedHostServiceClassProjection> original;
    const auto sha=[](const std::string& value){return value.size()==64&&std::all_of(value.begin(),value.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});};
    for(const auto& receipt:receipts){
        receipt.declaration.requireValid();
        if((receipt.role!=RetainedHostServiceRole::ImmutableLookupIO&&receipt.role!=RetainedHostServiceRole::ExecutorParentTicket)||!sha(receipt.resourceIdentity)||!sha(receipt.liveIdentity))throw std::invalid_argument("CPU service original-rank receipt identity is incomplete");
        if(std::any_of(original.begin(),original.end(),[&](const auto& p){return p.role==receipt.role;}))throw std::logic_error("CPU service original-rank receipt role duplicated");
        original.push_back(receipt);
    }
    return RetainedHostServiceBudgetProjection(std::move(original));
}
RetainedHostServiceProgram RetainedHostServiceBudgetProjection::program(RetainedHostServiceRole role)const{
    const auto found=std::find_if(programs_.begin(),programs_.end(),[&](const auto& p){return p.role()==role;});
    if(found!=programs_.end())return *found;
    const auto receipt=std::find_if(receipts_.begin(),receipts_.end(),[&](const auto& p){return p.role==role;});
    if(receipt==receipts_.end())throw std::logic_error("CPU service candidate lacks the original rank's role evidence");
    return RetainedHostServiceProgram::fromInstalledWire(receipt->role,receipt->resourceIdentity,receipt->liveIdentity,receipt->declaration);
}
RetainedHostServiceDeclarations::Builder& RetainedHostServiceDeclarations::Builder::addProjected(RetainedHostServiceInstance key,const RetainedHostServiceBudgetProjection& projection){return add(key,projection.program(key.role));}
RetainedHostServiceDeclarations::Builder::Builder(int rank):rank_(rank){if(rank<0)throw std::invalid_argument("CPU service declarations require original physical rank");}
RetainedHostServiceDeclarations::Builder& RetainedHostServiceDeclarations::Builder::add(RetainedHostServiceInstance key,RetainedHostServiceProgram program){
    if(key.worldRank!=rank_||!key.consumer.is_valid()||key.role!=program.role())throw std::invalid_argument("CPU service declaration has foreign rank/consumer/role");
    if(std::any_of(entries_.begin(),entries_.end(),[&](const auto& entry){return entry.first==key;}))throw std::logic_error("CPU service role/consumer declared twice");
    entries_.emplace_back(key,std::move(program));return *this;
}
RetainedHostServiceDeclarations::Builder& RetainedHostServiceDeclarations::Builder::addOriginal(RetainedHostServiceInstance key){
    if(!original_evidence_)original_evidence_=RetainedHostServiceProgram::originalEvidence();
    return add(key,RetainedHostServiceProgram::original(key.role,original_evidence_));
}
std::shared_ptr<const RetainedHostServiceDeclarations> RetainedHostServiceDeclarations::Builder::freeze()&&{
    if(rank_<0)throw std::logic_error("CPU service declaration builder was already frozen");
    const auto rank=rank_;rank_=-1;
    return std::shared_ptr<const RetainedHostServiceDeclarations>(new RetainedHostServiceDeclarations(rank,std::move(entries_)));
}
RetainedHostServiceDeclarations::RetainedHostServiceDeclarations(int rank,std::vector<std::pair<RetainedHostServiceInstance,RetainedHostServiceProgram>> entries):rank_(rank),entries_(std::move(entries)){}
std::vector<RetainedHostServiceInstance> RetainedHostServiceDeclarations::instances()const{
    std::vector<RetainedHostServiceInstance> result;result.reserve(entries_.size());for(const auto& entry:entries_)result.push_back(entry.first);return result;
}
const RetainedHostServiceProgram& RetainedHostServiceDeclarations::program(RetainedHostServiceInstance key)const{
    if(key.worldRank!=rank_)throw std::logic_error("CPU service declaration crossed physical rank");
    const auto found=std::find_if(entries_.begin(),entries_.end(),[&](const auto& entry){return entry.first==key;});
    if(found==entries_.end())throw std::logic_error("CPU service role is absent from original pre-quota declaration");return found->second;
}
bool RetainedHostServiceDeclarations::contains(RetainedHostServiceInstance key)const{
    if(key.worldRank!=rank_)throw std::logic_error("CPU service membership crossed original rank");
    return std::any_of(entries_.begin(),entries_.end(),[&](const auto& entry){return entry.first==key;});
}
std::vector<RetainedHostServiceInstance> RetainedHostServiceDeclarations::selectedForLookup(const ResolvedImmutableRowTablePlacement* placement)const{
    std::vector<RetainedHostServiceInstance> selected;
    for(const auto& [key,program]:entries_){
        if(key.role==RetainedHostServiceRole::ExecutorParentTicket){selected.push_back(key);continue;}
        if(!placement)throw std::logic_error("CPU I/O role requires final source-selected lookup placement");
        if (key.consumer == DeviceId::cpu() && placement->isGPUAdaptive() &&
            placement->gpuAdaptiveRequest().placement.mode == AdaptiveGPUCachePlacementMode::ConsumerLocal &&
            placement->gpuAdaptiveLayout().adaptiveGPUDomain() == ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal) {
            if (placement->gpuAdaptiveLayout().sourceServices().front().allocator !=
                PhysicalMemoryAllocatorIdentity{key.worldRank, DeviceId::cpu()})
                throw std::logic_error("Grouped lookup worker belongs to another original rank source");
            selected.push_back(key);
            continue;
        }
        const auto execution=placement->lookupForConsumer({key.worldRank,key.consumer});
        if(execution==ImmutableRowLookupExecution::HostProvider || execution==ImmutableRowLookupExecution::AdaptiveGPUCache)selected.push_back(key);
    }
    return selected;
}
PhysicalMemoryPlan RetainedHostServiceDeclarations::bom(RetainedHostServiceInstance key,PhysicalMemoryResource cpu)const{
    if(cpu.world_rank!=rank_)throw std::invalid_argument("CPU service BOM changed original rank observation");return program(key).bom(cpu);
}
std::unique_ptr<RetainedHostServiceBinding> RetainedHostServiceDeclarations::bind(std::shared_ptr<PhysicalMemoryAuthority> memory,std::span<const RetainedHostServiceInstance> selected)const{
    if(!memory||memory->worldRank()!=rank_||memory->phase()!=PhysicalMemoryAuthorityPhase::Admitted)throw std::logic_error("CPU service binding requires same original admitted rank authority");
    memory->requireOriginalHostServiceOwner(memory);
    for(std::size_t i=0;i<selected.size();++i){(void)program(selected[i]);if(std::find(selected.begin(),selected.begin()+i,selected[i])!=selected.begin()+i)throw std::logic_error("CPU service selected role duplicated");}
    std::vector<RetainedHostServiceBinding::Entry> entries;entries.reserve(selected.size());
    std::shared_ptr<const RetainedHostServiceProgram::Evidence> localEvidence;
    for(const auto& key:selected){
        auto selectedProgram=program(key);
        if(selectedProgram.evidenceKind()==RetainedHostServiceEvidence::InstalledWireProjection){
            if(!localEvidence)localEvidence=RetainedHostServiceProgram::originalEvidence();
            auto actual=RetainedHostServiceProgram::original(key.role,localEvidence);
            if(actual.identity()!=selectedProgram.identity()||actual.resourceIdentity()!=selectedProgram.resourceIdentity()||actual.resourceDeclaration()!=selectedProgram.resourceDeclaration())throw std::logic_error("CPU service selected original rank evidence changed before local binding");
            selectedProgram=std::move(actual);
        }
        entries.push_back({key,std::move(selectedProgram),false});
    }
    return std::unique_ptr<RetainedHostServiceBinding>(new RetainedHostServiceBinding(std::move(memory),std::move(entries)));
}
RetainedHostServiceBinding::RetainedHostServiceBinding(std::shared_ptr<PhysicalMemoryAuthority> memory,std::vector<Entry> entries):memory_(std::move(memory)),entries_(std::move(entries)){}
RetainedHostServiceUseToken RetainedHostServiceBinding::take(RetainedHostServiceInstance key){
    const auto program=[&]{
        std::lock_guard lock(mutex_);auto found=std::find_if(entries_.begin(),entries_.end(),[&](const auto& entry){return entry.key==key;});
        if(found==entries_.end()||found->consumed)throw std::logic_error("CPU service attempted undeclared/foreign/already consumed instance");
        found->consumed=true;return found->program;
    }();
    // No application lock spans loader/source validation, mapping or creators.
    // A failed one-shot construction remains failed; no alternate role or retry
    // can consume another instance's original admission.
    return RetainedHostServiceUseToken::claim(memory_,program,key);
}
}
