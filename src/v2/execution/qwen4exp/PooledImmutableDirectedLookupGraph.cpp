/**
 * @file PooledImmutableDirectedLookupGraph.cpp
 * @brief Compose existing native participant phases into actual source and remote owner programs.
 *
 * Local operands remain direct and each remote borrows only its addressed two
 * channels. The source program consumes current original Hash IDs and preserves
 * its output identity. Every bank-reuse edge comes from the existing typed
 * mixed participant builders; this file allocates no native storage or event.
 */
#include "PooledImmutableDirectedLookupGraph.h"
#include "PooledImmutableReplicatedReaderGraph.h"
#include "PooledImmutableParticipantLocalGraph.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/NativeContextPreparation.h"
#include <stdexcept>
namespace llaminar2 {
namespace {
namespace q=qwen4exp;
namespace n=q::pooled_native;
/** @brief Merge only a fully constructed participant-local fragment, rejecting name collisions first. */
void publishFragment(ComputeGraph& graph,ComputeGraph fragment){
    for(const auto& name:fragment.getExecutionOrder())
        if(graph.getNode(name))throw std::invalid_argument("Directed lookup fragment collides with an original stage: "+name);
    graph.merge(std::move(fragment));
}
}
PooledImmutableDirectedLookupEndpoints appendPooledImmutableDirectedLookupGraph(
    ComputeGraph& graph,BufferArena& arena,std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary,
    std::shared_ptr<const q::PooledImmutableDirectedTransport> transport,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code,n::OriginalRows original,
    q::PooledNativeOperandSpans begin,DeviceGenerationControlBinding control,q::PooledNativeTensorSpan output,
    const std::string& identity,const std::string& output_identity){
    if(isGraphCaptureActive() || !boundary || !boundary->isPooled() || !transport || !code || identity.empty() || output_identity.empty())
        throw std::invalid_argument("Directed lookup requires original source, transport, code and output before capture");
    transport->requireCurrent();
    if(transport->sourceBoundary()!=boundary)
        throw std::invalid_argument("Directed lookup changed its original source boundary");
    const auto storage=boundary->pooledStorage();const auto& prepared=storage->retainedPreparedSourceWorkers();
    if(!prepared || storage->originalOwners().empty() || code->backend()!=boundary->device().type)
        throw std::invalid_argument("Directed lookup requires its consumed original multiowner source and source-native code");
    const auto coordinator=storage->coordinator();const auto memory=prepared->source()->memory();
    const auto source_allocator=coordinator->plan().allocator();
    // P1 has no distributed ingress frontier. Its actual existing local graph
    // owns Read/release; only compute-result/ACK channels are added for readers.
    if(storage->originalOwners().size()==1 &&
       storage->originalOwners().front()->plan().allocator()==source_allocator){
        if(!transport->remotes().empty())
            throw std::invalid_argument("Sole local owner cannot retain a remote storage program");
        bindPooledNativeRoots(arena,*coordinator);
        bindPooledNativeRoots(arena,*storage->originalOwners().front());
        bindPooledSourceRoots(arena,*boundary);
        ComputeGraph fragment;
        std::shared_ptr<PooledImmutableLocalRetirementCompletion> terminal;
        const auto count=begin[static_cast<unsigned>(q::PooledNativeOperand::OriginalCount)];
        const auto endpoints=appendPooledImmutableLocalLookupGraph(fragment,arena,boundary,code,
            original,std::move(begin),control,std::move(output),identity,output_identity,&terminal);
        if(storage->geometry().parameters().consumers>1)
            terminal->readers=appendPooledReplicatedSourceReaders(fragment,arena,transport,control,
                original.rows,terminal->output,terminal->read,terminal->readersRetired,identity+"/readers",count);
        PooledImmutableSourceRetirementCompletion completion{
            std::shared_ptr<const PooledImmutableLocalRetirementCompletion>(std::move(terminal))};
        requirePooledSourceRetirementCompletion(fragment,completion);
        publishFragment(graph,std::move(fragment));graph.setRetainedStorageBoundary(std::move(boundary));
        return {endpoints,std::move(completion)};
    }
    std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> local;
    std::vector<q::PooledImmutablePinParticipant> pins;
    std::vector<q::PooledImmutableFreeParticipant> free_members;
    std::vector<q::PooledImmutableVictimParticipant> victims;
    std::vector<q::PooledImmutableSourceParticipant> sources;
    std::vector<q::PooledImmutablePublicationParticipant> publications;
    std::vector<q::PooledImmutableReadParticipant> readers;
    std::vector<q::PooledImmutableRetirementParticipant> retirement_members;
    for(std::uint32_t i=0;i<storage->originalOwners().size();++i){
        const auto& owner=storage->originalOwners()[i];
        if(owner->plan().allocator()==source_allocator){
            if(local)throw std::invalid_argument("Directed lookup duplicated its actual local physical owner");
            local=q::PooledImmutableLocalOwnerBinding::bind(coordinator,owner,memory);
            pins.emplace_back(local);free_members.emplace_back(local);victims.emplace_back(local);sources.emplace_back(local);
            publications.emplace_back(local);readers.emplace_back(local);retirement_members.emplace_back(local);
        }else{
            const auto& remote=transport->remote(i);
            pins.emplace_back(remote.returned);free_members.emplace_back(remote.free);victims.emplace_back(remote.victim);
            sources.emplace_back(remote.source);publications.emplace_back(remote.publication);readers.emplace_back(remote.read);
            retirement_members.emplace_back(remote.retirement);
        }
    }
    if(transport->remotes().size()+(local?1u:0u)!=storage->originalOwners().size())
        throw std::invalid_argument("Directed lookup requires every original physical owner exactly once");
    const auto originalCount = begin[static_cast<unsigned>(q::PooledNativeOperand::OriginalCount)];
    bindPooledNativeRoots(arena,*coordinator);
    if(local)bindPooledNativeRoots(arena,*local->ownerStorage());
    bindPooledSourceRoots(arena,*boundary);
    ComputeGraph fragment;
    const auto first=identity+"/begin",commands=identity+"/commands";
    fragment.addNode(first,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,
        PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::Begin,coordinator,
            q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::Begin,coordinator->plan(),original,std::move(begin))}),boundary->device());
    fragment.addNode(commands,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,
        PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::PinCommands,coordinator,
            q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::PinCommands,coordinator->plan(),{},{})}),boundary->device());
    fragment.addDependency(commands,first);
    std::vector<std::string> sent(storage->originalOwners().size());
    for(const auto& remote:transport->remotes())
        sent[remote.coordinate]=appendPooledPinSender(fragment,arena,remote.pin,first,commands,
            identity+"/pin_send"+std::to_string(remote.coordinate)).last;
    std::optional<PooledImmutableMixedVictimCompletion> mixedVictim;
    std::string victimsComplete;
    if(local){
        const auto pin=appendPooledPinCoordinatorMembers(fragment,arena,pins,code,sent,first,commands,identity+"/pin");
        const auto free=appendPooledFreeCoordinatorMembers(fragment,arena,free_members,code,pin.coordinator.last,identity+"/free",&pin);
        mixedVictim=appendPooledVictimCoordinatorMembers(fragment,arena,victims,code,free.coordinator.last,identity+"/victim",&free);
        victimsComplete=mixedVictim->coordinator.last;
    }else{
        std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> returns;
        std::vector<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> frees;
        std::vector<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> remoteVictims;
        for(const auto& remote:transport->remotes()){
            returns.push_back(remote.returned);frees.push_back(remote.free);remoteVictims.push_back(remote.victim);
        }
        const auto pin=appendPooledPinCoordinatorReturn(fragment,arena,returns,code,sent,commands,identity+"/pin");
        const auto free=appendPooledFreeCoordinatorWave(fragment,arena,frees,code,pin.last,identity+"/free");
        const auto victim=appendPooledVictimCoordinatorWave(fragment,arena,remoteVictims,code,free.last,identity+"/victim");
        victimsComplete=victim.last;
    }
    const auto source=appendPooledSourceCoordinatorMembers(fragment,arena,sources,boundary,code,control,
        victimsComplete,identity+"/source",mixedVictim?&*mixedVictim:nullptr);
    const auto publication=appendPooledPublicationCoordinator(fragment,arena,source,publications,code,identity+"/publication");
    const auto read=appendPooledReadCoordinator(fragment,arena,publication,readers,std::move(output),identity+"/read",output_identity);
    auto retired=appendPooledRetirementCoordinator(fragment,arena,read,retirement_members,code,identity+"/retirement");
    if (storage->geometry().parameters().consumers > 1)
        retired.readers = appendPooledReplicatedSourceReaders(fragment, arena, transport, control,
            original.rows, retired.read.outputSpan, read.output, retired.readersRetired, identity + "/readers", originalCount);
    requirePooledRetirementCompletion(fragment,retired);
    const auto last=retired.coordinator.last;
    publishFragment(graph,std::move(fragment));graph.setRetainedStorageBoundary(std::move(boundary));
    return {{first,last,PooledImmutableDemandFrontier{PooledImmutableDemandRole::Source,source.published,source.acquired}},std::move(retired)};
}
void requirePooledSourceRetirementCompletion(const ComputeGraph& graph,
    const PooledImmutableSourceRetirementCompletion& original){
    std::visit([&](const auto& completion){
        using T=std::decay_t<decltype(completion)>;
        if constexpr(std::is_same_v<T,PooledImmutableRetirementCompletion>)
            requirePooledRetirementCompletion(graph,completion);
        else{
            if(!completion || !completion->sourceBoundary() || (completion->sourceBoundary()->pooledStorage()->geometry().parameters().consumers>1 && !completion->sharedReaders()))
                throw std::invalid_argument("Local recurring source omitted original shared-reader completion");
            requirePooledLocalRetirementCompletion(graph,*completion);
        }
    },original);
}
PooledPinParticipantNodes appendPooledImmutableDirectedOwnerGraph(ComputeGraph& graph,BufferArena& arena,
    const q::PooledImmutableRemoteTransport& remote,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    const PooledRemoteGraphProgram& program,const std::string& identity){
    if(isGraphCaptureActive() || !remote.pin || !remote.retirement || !code || identity.empty())
        throw std::invalid_argument("Directed owner program requires original addressed transport and native code before capture");
    remote.retirement->requireCurrent();
    if(remote.coordinate!=remote.pin->declaration().coordinate() || code->backend()!=remote.pin->declaration().receiver().device.type)
        throw std::invalid_argument("Directed owner program changed original coordinate or native backend");
    // Registration borrows only the original admitted roots; no arena allocation
    // or replacement is permitted when the first addressed endpoint is declared.
    bindPooledNativeRoots(arena,*remote.pin->ownerStorage());
    ComputeGraph fragment;
    const auto pin=appendPooledPinReceiver(fragment,arena,remote.pin,code,identity+"/pin");
    const auto returned=appendPooledPinOwnerReturn(fragment,arena,remote.returned,code,pin.last,identity+"/return");
    const auto free=appendPooledFreeOwnerWave(fragment,arena,remote.free,code,returned.last,identity+"/free");
    const auto victim=appendPooledVictimOwnerWave(fragment,arena,remote.victim,code,free.last,identity+"/victim");
    const auto source=appendPooledSourceOwnerFill(fragment,arena,remote.source,code,victim.last,identity+"/source");
    const auto publication=appendPooledPublicationOwner(fragment,arena,remote.publication,code,source.last,identity+"/publication");
    const auto read=appendPooledReadOwner(fragment,arena,remote.read,code,publication.last,identity+"/read");
    const auto retired=appendPooledRetirementOwner(fragment,arena,remote.retirement,code,read.last,identity+"/retirement");
    requirePooledRetirementOwnerCompletion(fragment,remote.retirement,retired.last);
    q::PooledImmutableDirectedOwnerProgram::requireActual(fragment,remote.pin->ownerStorage()->plan(),remote.pin->coordinatorStorage()->plan(),program);
    publishFragment(graph,std::move(fragment));return {pin.first,retired.last};
}
PooledPinParticipantNodes appendPooledImmutableInlineOwnerPrefix(
    ComputeGraph& graph, BufferArena& arena,
    const q::PooledImmutableRemoteTransport& remote,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    const std::string& identity, PooledImmutableDemandFrontier* demand)
{
    if (isGraphCaptureActive() || !remote.pin || !remote.retirement || !code || identity.empty())
        throw std::invalid_argument("Inline storage requires its original cold addressed owner declaration");
    remote.retirement->requireCurrent();
    if (remote.coordinate != remote.pin->declaration().coordinate() ||
        code->backend() != remote.pin->declaration().receiver().device.type)
        throw std::invalid_argument("Inline storage changed its original owner coordinate or native backend");
    bindPooledNativeRoots(arena, *remote.pin->ownerStorage());
    const auto pin = appendPooledPinReceiver(graph, arena, remote.pin, code, identity + "/pin");
    const auto returned = appendPooledPinOwnerReturn(graph, arena, remote.returned, code,
        pin.last, identity + "/return");
    const auto free = appendPooledFreeOwnerWave(graph, arena, remote.free, code,
        returned.last, identity + "/free");
    const auto victim = appendPooledVictimOwnerWave(graph, arena, remote.victim, code,
        free.last, identity + "/victim");
    const auto source = appendPooledSourceOwnerFill(graph, arena, remote.source, code,
        victim.last, identity + "/source", demand);
    const auto publication = appendPooledPublicationOwner(graph, arena, remote.publication, code,
        source.last, identity + "/publication");
    const auto read = appendPooledReadOwner(graph, arena, remote.read, code,
        publication.last, identity + "/read");
    requirePooledReadOwnerCompletion(graph, remote.read, read.last);
    return {pin.first, read.last};
}

PooledPinParticipantNodes appendPooledImmutableInlineOwnerRetirement(
    ComputeGraph& graph, BufferArena& arena,
    const q::PooledImmutableRemoteTransport& remote,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    const std::string& rowsSent, const PooledImmutableReplicatedConsumerCompletion& consumer,
    const std::string& identity)
{
    if (isGraphCaptureActive() || !remote.retirement || !code || identity.empty())
        throw std::invalid_argument("Inline storage retirement requires its original cold read declaration");
    requirePooledReplicatedConsumerCompletion(graph, consumer);
    const auto retired = appendPooledRetirementOwner(graph, arena, remote.retirement, code,
        rowsSent, identity, &consumer);
    requirePooledRetirementOwnerCompletion(graph, remote.retirement, retired.last, &consumer);
    return retired;
}

}
