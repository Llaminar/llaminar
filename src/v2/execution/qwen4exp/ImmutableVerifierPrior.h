/**
 * @file ImmutableVerifierPrior.h
 * @brief Move-only setup capability for dedicated immutable verifier prior storage.
 *
 * This capability names borrowed, already-admitted storage and an actual snapshot
 * producer in one graph. It does not track execution progress, allocate state,
 * register graph-local ownership or introduce a scheduler. Dedicated prior roles
 * have no ordinary/reset/archive writer. The canonical complete MTP transaction
 * orders prior readers/publication before another verifier snapshot; existing
 * terminal/reset event retirement owns reuse and destruction.
 */
#pragma once
#include "../compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include <span>
#include <string>
namespace llaminar2 {
class ComputeGraph;class Qwen4ExpAuxiliaryState;
/** @brief Exact immutable borrowed native interval; packed replay records remain INT32 backing. */
struct ImmutableVerifierPriorRegion {
 std::variant<Qwen4ExpIntBinding,Qwen4ExpFloatBinding> binding;
 std::size_t bytes;
};
/**
 * @brief Immutable setup identity owned by the actual snapshot-stage lifetime.
 *
 * A capability holds only a weak reference. Replacing/clearing the stage expires
 * its proof even if an allocator later reuses the same stage address or name.
 * This is graph-construction identity, not mutable execution or retirement state.
 */
class ImmutableVerifierPriorProducer {
public:
 /** @brief Destroy the stage-owned identity; outstanding capabilities then fail. */
 virtual ~ImmutableVerifierPriorProducer()=default;
 ImmutableVerifierPriorProducer(const ImmutableVerifierPriorProducer&)=delete;
 ImmutableVerifierPriorProducer&operator=(const ImmutableVerifierPriorProducer&)=delete;
protected:
 /** @brief Allocate one immutable setup token before capture, never during execution. */
 ImmutableVerifierPriorProducer():setupIdentity_(std::make_shared<const std::uint8_t>(0)){}
private:
 friend class ImmutableVerifierPrior;
 std::shared_ptr<const std::uint8_t>setupIdentity_;
};
/** @brief One graph-construction capability consumed by its retained-publication stage. */
class ImmutableVerifierPrior final {
public:
 ImmutableVerifierPrior(const ImmutableVerifierPrior&)=delete;
 ImmutableVerifierPrior&operator=(const ImmutableVerifierPrior&)=delete;
 /** @brief Transfer graph-composition authority and invalidate the original capability. */
 ImmutableVerifierPrior(ImmutableVerifierPrior&&other)noexcept;
 ImmutableVerifierPrior&operator=(ImmutableVerifierPrior&&)=delete;
 /** @brief Actual snapshot node, not an inferred cache frontier or overwritten tail. */
 const std::string&snapshotProducer()const;
 /** @brief Exact immutable borrowed prior regions, independent of communication capacity. */
 std::span<const ImmutableVerifierPriorRegion>regions()const;
 /** @brief Reject reuse, another graph or another participant before graph mutation. */
 void requireFor(const ComputeGraph&graph,DeviceId device)const;
 /** @brief Add the real snapshot producer edge before one same-graph consumer. */
 void orderBefore(ComputeGraph&graph,const std::string&consumer)const;
 /** @brief Bind the existing publication-hook consumer, then consume this capability. */
 std::string retainForPublication(ComputeGraph&graph,const std::string&consumer)&&;
private:
 friend class Qwen4ExpAuxiliaryState;
 /** @brief Mint only after the auxiliary owner installs the sole snapshot writer. */
 ImmutableVerifierPrior(ComputeGraph&graph,DeviceId device,std::string producer,
     std::vector<ImmutableVerifierPriorRegion>regions);
 ComputeGraph*graph_;DeviceId device_;std::string producer_;
 const IComputeStage*producerStage_=nullptr;
 std::weak_ptr<const std::uint8_t>producerIdentity_;
 std::vector<ImmutableVerifierPriorRegion>regions_;
};
}
