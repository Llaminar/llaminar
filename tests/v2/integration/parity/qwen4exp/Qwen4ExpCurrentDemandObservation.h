/**
 * @file Qwen4ExpCurrentDemandObservation.h
 * @brief Immutable inspection of genuine Rank-owned captured current-demand graphs.
 *
 * Existing closed test friends inspect only after real runner initialization.
 * No constructor, source permit, stream or mutable graph is returned. Detached
 * observations retain weak source-token witnesses so inspection cannot prevent
 * original parent retirement. Numerical execution remains a separate fixture.
 */
#pragma once
#include "Qwen4ExpPooledCurrentDemandCases.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLERetainedStage.h"
#include "execution/qwen4exp/PooledImmutableParticipantSourceGraph.h"
#include "execution/qwen4exp/PooledImmutableDirectedOperation.h"
#include "execution/qwen4exp/PooledImmutableReplicatedReaderGraph.h"
#include "loaders/ModelContext.h"
#include <fstream>
#include <map>

namespace llaminar2 {
/** @brief Existing production predicate, declared here only for immutable actual-node negative inspection. */
bool isPooledPureReaderDemandHeader(const ComputeNode&,const RetainedGraphParticipantDefinitionDomain&);
}
namespace llaminar2::test::parity::qwen4exp {
namespace current_demand_observation {
namespace q=::llaminar2::qwen4exp;
/** @brief Detached actual cache/frontier facts; no executable or execution authority escapes. */
struct Graph {
    DeviceId device;
    PooledImmutableDemandRole role;
    int rows;
    bool decode;
    std::uint64_t topology,variant,capture_generation=0,replay_steps=0;
    std::weak_ptr<const RetainedGraphParticipantPreparationStream> source;
    bool verifier=false; ///< Actual dedicated prior and retained-publication stages, never inferred from M.
};
/** @brief Require a genuine cold observation, preserving the actual original failure. */
inline void require(bool value,const char* why){if(!value)throw std::logic_error(why);}
/** @return Whether this actual stage is the original query hash, independent of its name. */
inline bool isHash(const ComputeNode& node){
    const auto* primitive=dynamic_cast<const Qwen4ExpPrimitiveStage*>(node.stage.get());
    return primitive && std::holds_alternative<Qwen4ExpPrimitiveStage::NgramHash>(primitive->getParams().operation);
}
/**
 * @brief Authenticate the actual source, inline or pure-reader cut on its retained graph.
 * @param graph Actual cache-owned immutable graph; caller keeps its original DGO alive.
 * @param original Same model-owned retained participants/transport that materialized it.
 * @param device Original child device, never an inferred role from the test ID.
 * @return Actual role; response and independent trunk order are validated first.
 */
inline PooledImmutableDemandRole inspectCut(const ComputeGraph& graph,
    const std::shared_ptr<q::PooledImmutableRetainedParticipants>& original,DeviceId device,bool* verifier=nullptr)
{
    require(bool(original),"Current-demand graph lost original retained participants");original->requireCurrent();
    const auto transport=original->transport();require(bool(transport) && transport->readerGeometry().has_value(),"Current-demand graph lost original reader transport");
    const auto& order=graph.getExecutionOrder();std::string hash;
    for(const auto& name:order)if(isHash(*graph.getNode(name))){require(hash.empty(),"Current-demand graph has ambiguous query hashes");hash=name;}
    require(!hash.empty(),"Current-demand graph omitted its original query hash");
    PooledImmutableDemandFrontier frontier{PooledImmutableDemandRole::PureReader,{},{}};
    std::vector<std::string> trunk;
    if(device==transport->readerGeometry()->coordinator().device){
        frontier.role=PooledImmutableDemandRole::Source;
        for(const auto& name:order){
            const auto* stage=dynamic_cast<const PooledImmutableSourcePhaseStage*>(graph.getNode(name)->stage.get());
            if(!stage)continue;
            if(stage->matchesDemand(original->boundary().get(),q::PooledSourceAction::Publish)){require(frontier.published.empty(),"Ambiguous original demand Publish");frontier.published=name;}
            if(stage->matchesDemand(original->boundary().get(),q::PooledSourceAction::AcquireResponse)){require(frontier.consume.empty(),"Ambiguous original response acquisition");frontier.consume=name;trunk=stage->currentDemandTrunk();}
        }
    }else{
        const auto& owners=original->boundary()->pooledStorage()->geometry().originalOwners();
        const bool inline_owner=std::any_of(owners.begin(),owners.end(),[&](const auto& owner){return owner.allocator().device==device;});
        if(inline_owner){
            frontier.role=PooledImmutableDemandRole::InlineOwner;std::string terminal;
            for(const auto& name:order){
                const auto transfer=describePooledSourceTransfer(*graph.getNode(name)->stage);if(!transfer)continue;
                if(transfer->role==static_cast<std::uint32_t>(q::PooledSourceWireRole::SealReceipt) && transfer->endpoint==CapturedTransferEndpoint::Producer)frontier.published=name;
                if(transfer->endpoint==CapturedTransferEndpoint::Consumer){
                    if(transfer->role==static_cast<std::uint32_t>(q::PooledSourceWireRole::Fills))frontier.consume=name;
                    if(transfer->role==static_cast<std::uint32_t>(q::PooledSourceWireRole::TerminalFrontier))terminal=name;
                }
            }
            if(frontier.consume.empty())frontier.consume=terminal;
        }else{
            const auto completion=findPooledReplicatedConsumerCompletion(graph,transport,device);
            frontier.consume=completion.header;
            const auto& domain=graph.retainedParticipantDefinitionDomain();require(domain.has_value(),"Actual pure reader lost its source domain");
            require(isPooledPureReaderDemandHeader(*graph.getNode(frontier.consume),*domain),"Actual captured Header is not its original pure-reader endpoint");
            auto foreign=*domain;foreign.authority=&graph;
            require(!isPooledPureReaderDemandHeader(*graph.getNode(frontier.consume),foreign),"Genuine ReaderTransfer accepted a foreign source domain");
        }
        require(!frontier.consume.empty(),"Original reader has no actual response endpoint");
        for(const auto& name:graph.getNode(frontier.consume)->dependencies)
            if(name!=frontier.published && !isHash(*graph.getNode(name)))trunk.push_back(name);
    }
    validatePooledDemandFrontier(graph,frontier);
    require(!trunk.empty(),"Captured current demand has no consumed independent trunk join");
    const auto position=[&](const std::string& name){const auto found=std::find(order.begin(),order.end(),name);require(found!=order.end(),"Current-demand frontier lost a retained node");return found-order.begin();};
    const auto topology=graph.topologyGeneration();
    for(const auto& name:trunk){
        const auto* node=graph.getNode(name);require(node && node->device==device,"Current-demand trunk changed participant");
        require(position(hash)<position(name) && position(name)<position(frontier.consume),"Current-demand hash/trunk/response order changed");
        if(!frontier.published.empty())require(position(frontier.published)<position(name),"Current demand was not published before ordinary trunk work");
    }
    const Qwen4ExpPLERetainedStage* snapshot=nullptr;const Qwen4ExpPLERetainedStage* retained=nullptr;
    std::string snapshotName,retainedName;
    for(const auto& name:order){
        const auto* stage=dynamic_cast<const Qwen4ExpPLERetainedStage*>(graph.getNode(name)->stage.get());
        if(!stage)continue;
        require(stage->device()==device&&graph.getNode(name)->device==device,"Verifier prior changed original participant");
        if(stage->getParams().operation==Qwen4ExpPLERetainedStage::Operation::SnapshotPrior){require(!snapshot,"Ambiguous original PLE prior");snapshot=stage;snapshotName=name;}
        else {require(!retained,"Ambiguous original PLE candidate retention");retained=stage;retainedName=name;}
    }
    if(snapshot||retained){
        require(snapshot&&retained,"Captured grouped lookup lost its original snapshot/publication pair");
        const auto& before=snapshot->getParams();const auto& after=retained->getParams();
        const auto& hashParams=dynamic_cast<const Qwen4ExpPrimitiveStage&>(*graph.getNode(hash)->stage).getParams();
        const auto& hashInput=std::get<Qwen4ExpPrimitiveStage::NgramHash>(hashParams.operation);
        const auto sameBank=[](const auto& a,const auto& b){return a.tensor==b.tensor&&a.buffer==b.buffer&&a.byteOffset==b.byteOffset;};
        const auto* count=hashParams.rows.liveRows?std::get_if<Qwen4ExpNativeIntInput>(&hashParams.rows.liveRows->value):nullptr;
        const auto* priorCount=before.liveCounts?std::get_if<Qwen4ExpNativeIntInput>(&before.liveCounts->value):nullptr;
        const auto* retainCount=after.liveCounts?std::get_if<Qwen4ExpNativeIntInput>(&after.liveCounts->value):nullptr;
        require(before.requests==1&&after.requests==1&&before.requestIndex==0&&after.requestIndex==0&&
            before.rowCapacity==hashParams.rows.capacity&&after.rowCapacity==before.rowCapacity&&
            count&&priorCount&&retainCount&&count->owner==device&&priorCount->owner==device&&retainCount->owner==device&&
            count->capacity==1&&priorCount->capacity==1&&retainCount->capacity==1&&
            count->data==priorCount->data&&count->data==retainCount->data&&
            sameBank(hashInput.priorTokens,before.priorTokens)&&sameBank(before.priorTokens,after.priorTokens)&&
            sameBank(before.priorConvolution,after.priorConvolution),
            "Grouped lookup changed its original immutable prior/count/history authority");
        require(position(snapshotName)<position(hash)&&position(frontier.consume)<position(retainedName),
            "Grouped lookup moved hash before snapshot or retention before response");
    }
    if(verifier)*verifier=snapshot!=nullptr;
    require(graph.topologyGeneration()==topology,"Immutable current-demand inspection changed graph topology");
    return frontier.role;
}
}
}

namespace llaminar2 {
/** @brief Read-only use of the pre-existing DGO test friendship; no materialization capability is issued. */
struct DeviceGraphOrchestratorLiveStateTestAccess {
    static std::vector<test::parity::qwen4exp::current_demand_observation::Graph> currentDemand(
        const DeviceGraphOrchestrator& runner,const ModelContext& model,bool executed=false)
    {
        namespace o=test::parity::qwen4exp::current_demand_observation;
        o::require(bool(runner.forward_engine_),"Current-demand inspection precedes original forward materialization");
        std::vector<o::Graph> result;
        const auto original=model.adaptiveImmutableLookupParticipants(runner.state_.device_id);
        o::require(bool(original),"Current-demand inspection lost the actual model participants");
        original->requireCurrent();
        // Reader graphs borrow a definition domain but deliberately own no
        // source preparation token. The source-family export rejects that
        // substitution; inspect the reader's actual retained decode cache
        // instead. Source-prefill observations remain coordinator-owned.
        const bool reader=original->boundary()->device()!=runner.state_.device_id;
        const auto observed=runner.forward_engine_->replayCacheObservations(0);
        if(executed || reader){
            for(const auto& replay:observed){
                if(!replay.signature.decode || !replay.valid || !replay.segment_initialized ||
                   replay.segment_needs_capture || (executed && replay.segment_decode_step==0))continue;
                std::string error;const auto native=runner.forward_engine_->retainedDecodeGraph(replay.signature,&error);
                o::require(native.has_value() && bool(*native) && native->graph_owner && native->capture_generation>0,
                    "Current-demand cache lost its actual original retained decode parent");
                const auto device=replay.signature.device;
                bool verifier=false;
                const auto role=o::inspectCut(*native->graph_owner,model.adaptiveImmutableLookupParticipants(device),device,&verifier);
                result.push_back({device,role,replay.signature.batch_size*replay.signature.seq_len,true,
                    native->topology_generation,native->capture_variant_signature,native->capture_generation,
                    replay.segment_decode_step,{},verifier});
            }
            o::require(!result.empty(),executed ?
                "Current-demand native exact proof observed no executed retained decode parent" :
                "Current-demand reader inspection observed no original retained decode parent");
            return result;
        }
        const auto definitions=runner.retainedExpertSourceFamilyDefinitions();
        for(const auto& definition:definitions){
            if(!definition.signature || definition.role==MoEOverlayInferenceGraphRole::MTPDraft)continue;
            definition.requireCurrent();
            const auto* graph=static_cast<const ComputeGraph*>(definition.graph_identity);
            o::require(graph && graph->topologyGeneration()==definition.topology_generation,"Actual retained graph changed during immutable inspection");
            const auto device=definition.signature->device;
            const auto original=model.adaptiveImmutableLookupParticipants(device);
            bool verifier=false;const auto role=o::inspectCut(*graph,original,device,&verifier);
            o::Graph row{device,role,definition.physical_rows,definition.signature->decode,
                definition.topology_generation,definition.capture_variant_signature,0,0,definition.source,verifier};
            if(row.decode){
                std::string error;const auto native=runner.forward_engine_->retainedDecodeGraph(*definition.signature,&error);
                o::require(native.has_value() && bool(*native) && native->graph_owner==graph &&
                    native->topology_generation==row.topology && native->capture_variant_signature==row.variant &&
                    native->capture_generation>0,"Current demand lost its actual retained decode parent/cache identity");
                row.capture_generation=native->capture_generation;
            }
            for(const auto& replay:observed)if(replay.signature==*definition.signature){
                o::require(replay.valid && replay.segment_initialized && !replay.segment_needs_capture,"Current-demand cache became stale");row.replay_steps=replay.segment_decode_step;
            }
            result.push_back(std::move(row));
        }
        o::require(!result.empty(),"Current-demand child has no genuine retained source/reader graphs");return result;
    }
};
namespace test {
/** @brief Read-only use of the existing Rank test friendship on genuine factory children only. */
struct RankRetainedStorageRetirementReplayAccess {
    static auto currentDemand(const RankOrchestrator& rank,const ModelContext& model,bool executed=false){
        namespace o=parity::qwen4exp::current_demand_observation;
        std::vector<o::Graph> result;
        for(const auto& child:rank.device_runners_){
            const auto* device=dynamic_cast<const DeviceGraphOrchestrator*>(child.get());
            o::require(device!=nullptr,"Current-demand candidate changed its original flat TP2 factory child");
            auto rows=DeviceGraphOrchestratorLiveStateTestAccess::currentDemand(*device,model,executed);
            result.insert(result.end(),std::make_move_iterator(rows.begin()),std::make_move_iterator(rows.end()));
        }
        return result;
    }
};
}
/** @brief Read-only use of existing runner friendship; never installs source, model or lifecycle state. */
class OrchestrationRunnerResolvedAdmissionTestAccess {
public:
    static auto currentDemand(const IOrchestrationRunner& interface,bool executed=false){
        const auto* concrete=dynamic_cast<const OrchestrationRunner*>(&interface);
        if(!concrete)throw std::logic_error("Current-demand observation requires original concrete runner factory");
        const auto& runner=*concrete;
        namespace o=test::parity::qwen4exp::current_demand_observation;
        const auto* rank=dynamic_cast<const RankOrchestrator*>(runner.runner_.get());
        const auto* model=dynamic_cast<const ModelContext*>(runner.model_ctx_.get());
        o::require(rank && model,"Current-demand inspection requires its actual Rank/ModelContext factory owners");
        return test::RankRetainedStorageRetirementReplayAccess::currentDemand(*rank,*model,executed);
    }
};
}
