/**
 * @file ImmutableVerifierPrior.cpp
 * @brief Setup validation and concrete DAG edges for dedicated prior capabilities.
 *
 * All validation is metadata-only and happens before capture. No mutable host
 * execution state is inferred from the device. Storage exclusion comes from
 * dedicated declaration/PMA ownership plus the canonical MTP transaction, not
 * a second graph-local registry or informal reuse of ordinary next banks.
 */
#include "ImmutableVerifierPrior.h"
#include "../local_execution/graph/ComputeGraph.h"
#include <stdexcept>
namespace llaminar2 {
ImmutableVerifierPrior::ImmutableVerifierPrior(ComputeGraph&graph,DeviceId device,std::string producer,
 std::vector<ImmutableVerifierPriorRegion>regions):graph_(&graph),device_(device),producer_(std::move(producer)),regions_(std::move(regions)){
 const auto*node=graph.getNode(producer_);if(!device.is_valid()||!node||!node->stage||node->stage->device()!=device||regions_.empty())throw std::invalid_argument("Immutable verifier prior needs its exact snapshot producer/participant");
 const auto*identity=dynamic_cast<const ImmutableVerifierPriorProducer*>(node->stage.get());
 if(!identity)throw std::invalid_argument("Immutable verifier prior needs a stage-owned snapshot identity");
 producerStage_=node->stage.get();producerIdentity_=identity->setupIdentity_;
 for(const auto&r:regions_)std::visit([&](const auto&b){if(!b.tensor||b.buffer>=BufferId::_COUNT||!r.bytes||r.bytes%4||b.byteOffset%4||b.byteOffset>b.tensor->size_bytes()||r.bytes>b.tensor->size_bytes()-b.byteOffset)throw std::invalid_argument("Immutable verifier prior exceeds its admitted native region");},r.binding);
}
ImmutableVerifierPrior::ImmutableVerifierPrior(ImmutableVerifierPrior&&other)noexcept:graph_(other.graph_),device_(other.device_),producer_(std::move(other.producer_)),producerStage_(other.producerStage_),producerIdentity_(std::move(other.producerIdentity_)),regions_(std::move(other.regions_)){other.graph_=nullptr;}
const std::string&ImmutableVerifierPrior::snapshotProducer()const{if(!graph_||producerIdentity_.expired())throw std::logic_error("Immutable verifier prior capability was consumed or its snapshot was retired");requireFor(*graph_,device_);return producer_;}
std::span<const ImmutableVerifierPriorRegion>ImmutableVerifierPrior::regions()const{(void)snapshotProducer();return regions_;}
void ImmutableVerifierPrior::requireFor(const ComputeGraph&graph,DeviceId device)const{if(!graph_||graph_!=&graph||device_!=device)throw std::invalid_argument("Immutable verifier prior cannot cross graph/participant or be reused");
 const auto proof=producerIdentity_.lock();const auto*node=graph.getNode(producer_);
 const auto*identity=node&&node->stage?dynamic_cast<const ImmutableVerifierPriorProducer*>(node->stage.get()):nullptr;
 if(!proof||!node||node->stage.get()!=producerStage_||!identity||identity->setupIdentity_!=proof)
  throw std::invalid_argument("Immutable verifier prior snapshot producer was replaced or retired");}
void ImmutableVerifierPrior::orderBefore(ComputeGraph&graph,const std::string&consumer)const{requireFor(graph,device_);const auto*node=graph.getNode(consumer);if(!node||!node->stage||node->stage->device()!=device_)throw std::invalid_argument("Immutable verifier prior consumer is missing or foreign");graph.addDependency(consumer,producer_);}
std::string ImmutableVerifierPrior::retainForPublication(ComputeGraph&graph,const std::string&consumer)&&{requireFor(graph,device_);const auto*node=graph.getNode(consumer);if(!node||!node->stage||!node->stage->hasVerifierStateCapture()||!node->stage->requiresVerifierStateCaptureForPublication())throw std::invalid_argument("Immutable verifier prior requires the existing accepted-state publication hook");orderBefore(graph,consumer);graph_=nullptr;return consumer;}
}
