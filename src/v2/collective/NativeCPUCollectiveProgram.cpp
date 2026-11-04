/**
 * @file NativeCPUCollectiveProgram.cpp
 * @brief Validate original CPU domain geometry and hold PMA commitments through native retirement.
 *
 * Native begin is private to the existing GlobalTPContext creator. An armed
 * token cannot disappear on an exception: partial communicators and the arena
 * must first retire through that same original context cleanup boundary.
 */
#include "collective/NativeCPUCollectiveProgram.h"
#include "app/MPIProcessSession.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace llaminar2 {
/** @brief Per-domain reservation and original process-use borrower, never allocation counters. */
struct NativeCPUCollectiveProgram::DomainUse::State {
    NativeCPUCollectiveProgram program;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    PhysicalMemoryOwnerReservation domain;
    std::shared_ptr<detail::MPICollectiveUse> process;
    int rank;
    bool native = false;
    State(NativeCPUCollectiveProgram p, std::shared_ptr<PhysicalMemoryAuthority> m,
          PhysicalMemoryOwnerReservation d, std::shared_ptr<detail::MPICollectiveUse> s, int r)
        :program(std::move(p)),memory(std::move(m)),domain(std::move(d)),process(std::move(s)),rank(r){}
};
NativeCPUCollectiveProgram::DomainUse::DomainUse(std::unique_ptr<State> state):state_(std::move(state)){}
NativeCPUCollectiveProgram::DomainUse::DomainUse(DomainUse&&) noexcept = default;
NativeCPUCollectiveProgram::DomainUse::~DomainUse() noexcept { if(state_ && state_->native) std::terminate(); }
const std::shared_ptr<PhysicalMemoryAuthority>& NativeCPUCollectiveProgram::DomainUse::originalMemory() const & {
    if(!state_) throw std::logic_error("CPU domain use has no original ownership"); return state_->memory;
}
const NativeCPUCollectiveProgram& NativeCPUCollectiveProgram::DomainUse::program() const & {
    if(!state_) throw std::logic_error("CPU domain use has no original program"); return state_->program;
}
int NativeCPUCollectiveProgram::DomainUse::originalWorldRank() const {
    if(!state_) throw std::logic_error("CPU domain use has no original rank"); return state_->rank;
}
void NativeCPUCollectiveProgram::DomainUse::beginNative() {
    if(!state_ || state_->native) throw std::logic_error("CPU domain native entry requires its fresh original admission");
    MPIProcessSession::requireCPUCollectiveNativeEntry(state_->process);
    state_->native=true;
}
void NativeCPUCollectiveProgram::DomainUse::retireAfterNativeDestruction() {
    if(!state_ || !state_->native) throw std::logic_error("CPU domain retirement requires its original armed native entry");
    state_->native=false;
    state_.reset();
}
/** @brief Excluded parent owns only an original session borrow, never a fake domain allocation. */
struct NativeCPUCollectiveProgram::ProcessUse::State {
    NativeCPUCollectiveProgram program;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<detail::MPICollectiveUse> process;
    int rank;
    bool native = false;
    State(NativeCPUCollectiveProgram p,std::shared_ptr<PhysicalMemoryAuthority> m,std::shared_ptr<detail::MPICollectiveUse> s,int r)
        :program(std::move(p)),memory(std::move(m)),process(std::move(s)),rank(r){}
};
NativeCPUCollectiveProgram::ProcessUse::ProcessUse(std::unique_ptr<State> state):state_(std::move(state)){}
NativeCPUCollectiveProgram::ProcessUse::ProcessUse(ProcessUse&&) noexcept=default;
NativeCPUCollectiveProgram::ProcessUse::~ProcessUse() noexcept {if(state_ && state_->native)std::terminate();}
const std::shared_ptr<PhysicalMemoryAuthority>& NativeCPUCollectiveProgram::ProcessUse::originalMemory() const & {
    if(!state_)throw std::logic_error("CPU excluded split use has no original ownership");return state_->memory;
}
const NativeCPUCollectiveProgram& NativeCPUCollectiveProgram::ProcessUse::program() const & {
    if(!state_)throw std::logic_error("CPU excluded split use has no original program");return state_->program;
}
int NativeCPUCollectiveProgram::ProcessUse::originalWorldRank() const {
    if(!state_)throw std::logic_error("CPU excluded split use has no original rank");return state_->rank;
}
void NativeCPUCollectiveProgram::ProcessUse::beginNative() {
    if(!state_ || state_->native)throw std::logic_error("CPU excluded split requires its fresh original process scope");
    MPIProcessSession::requireCPUCollectiveNativeEntry(state_->process);
    state_->native=true;
}
void NativeCPUCollectiveProgram::ProcessUse::retireAfterNativeSplit() {
    if(!state_ || !state_->native)throw std::logic_error("CPU excluded split retirement requires its original native entry");
    state_->native=false;state_.reset();
}
NativeCPUCollectiveProgram NativeCPUCollectiveProgram::expertShardRoutePartials(
    int domain, CollectiveBackendType backend, std::vector<Participant> members) {
    if(domain<0 || (backend!=CollectiveBackendType::UPI && backend!=CollectiveBackendType::MPI) ||
       members.empty() || members.size()>8)
        throw std::invalid_argument("CPU expert collective requires a one-through-eight-rank explicit MPI/UPI domain");
    for(std::size_t i=0;i<members.size();++i) {
        const auto& p=members[i];
        if(p.world_rank<0 || p.address.device_type!=DeviceType::CPU || p.address.device_ordinal!=0 ||
           p.address.numa_node<0 || p.address.hostname.empty())
            throw std::invalid_argument("CPU expert collective requires original rank-local CPU0 and exact NUMA endpoint");
        for(std::size_t j=0;j<i;++j) if(members[j].world_rank==p.world_rank)
            throw std::invalid_argument("CPU expert collective repeats an original process rank");
    }
    return NativeCPUCollectiveProgram(domain,backend,std::move(members));
}
void NativeCPUCollectiveProgram::contributeDomainBOM(PhysicalMemoryPlanBuilder& plan,const PhysicalMemoryResource& cpu,int rank) const {
    if(!cpu.valid() || cpu.world_rank!=rank || cpu.device!=DeviceId::cpu() ||
       std::none_of(participants_.begin(),participants_.end(),[&](const auto& p){return p.world_rank==rank;}))
        throw std::invalid_argument("CPU expert collective BOM requires its participating original CPU allocator");
    plan.add(cpu,PhysicalMemoryOwner::NativeCPUCollectiveDomain,domainBytes());
}
void NativeCPUCollectiveProgram::contributeProcessPoolBOM(PhysicalMemoryPlanBuilder& plan,const PhysicalMemoryResource& cpu) {
    if (!cpu.valid() || cpu.world_rank < 0 || cpu.device != DeviceId::cpu())
        throw std::invalid_argument("CPU process pool BOM requires its original rank-local CPU allocator");
    plan.add(cpu,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool,processPoolBytes());
}
NativeCPUCollectiveProgram::DomainUse NativeCPUCollectiveProgram::admit(std::shared_ptr<PhysicalMemoryAuthority> memory,int rank) const {
    const auto original = memory ? memory->weak_from_this().lock() : nullptr;
    if(!original || original.get()!=memory.get() || original.owner_before(memory) || memory.owner_before(original) || memory->worldRank()!=rank || std::none_of(participants_.begin(),participants_.end(),[&](const auto& p){return p.world_rank==rank;}))
        throw std::logic_error("CPU expert collective admission requires its original participating rank PMA");
    // Domain first: a failed per-domain admission must not create a process pool.
    auto domain=memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveDomain,domainBytes());
    auto process=MPIProcessSession::retainCPUCollectivePool(memory);
    return DomainUse(std::make_unique<DomainUse::State>(*this,std::move(memory),std::move(domain),std::move(process),rank));
}
NativeCPUCollectiveProgram::ProcessUse NativeCPUCollectiveProgram::admitExcludedParent(std::shared_ptr<PhysicalMemoryAuthority> memory,int rank) const {
    const auto original=memory ? memory->weak_from_this().lock() : nullptr;
    if(!original || original.get()!=memory.get() || original.owner_before(memory) || memory.owner_before(original) ||
       memory->worldRank()!=rank || rank<0 || std::any_of(participants_.begin(),participants_.end(),[&](const auto& p){return p.world_rank==rank;}))
        throw std::logic_error("CPU excluded split requires its original nonmember process PMA, never fabricated domain membership");
    auto process=MPIProcessSession::retainCPUCollectivePool(memory);
    return ProcessUse(std::make_unique<ProcessUse::State>(*this,std::move(memory),std::move(process),rank));
}

}
