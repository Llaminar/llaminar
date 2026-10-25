/**
 * @file ImmutableRowTableExecutionProvision.h
 * @brief Source-authored mandatory lookup execution bound, separate from bulk table storage.
 *
 * This immutable setup BOM is not an allocation authority or live ledger. The
 * ordinary/overlay fixed plan pays it before graph-row and expert decisions.
 * Final admission authenticates the same source/consumer geometry and proves
 * its complete fixed footprint is covered before extending only table storage.
 */
#pragma once
#include "planning/ModelMemoryProfile.h"
#include "planning/ActivationMemoryEstimator.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include <utility>
#include <tuple>
#include <vector>
namespace llaminar2 {
class MemoryPlanner;
class ImmutableRowTablePlacementInput;
/**
 * @brief Resolve exact original ngram IDs for one admitted temporal consumer.
 * @param ordinary_rows Ordinary physical token-row envelope.
 * @param batch Actual supported request batch.
 * @param heads Source-owned ngram heads per token.
 * @param verifier_rows Admitted grouped verifier token-row capacity.
 * @param retained_rows Retained transaction token-row capacity.
 * @return Checked batch * heads * maximum admitted temporal rows, without reducing M.
 * @throws std::exception For invalid geometry or unrepresentable multiplication.
 */
std::size_t immutableRowTableMaximumLiveIds(int ordinary_rows,int batch,std::size_t heads,
    int verifier_rows,int retained_rows);
/** @brief Exact immutable graph geometry for one temporal retained lookup instance. */
struct ImmutableRowTableExecutionConsumer final {
    PhysicalMemoryResource resource;
    int rows, batch, context, local_heads, first_layer, last_layer, shards;
    int mtp_rows, retained_mtp_rows, request_capacity;
    ActivationGraphMemoryGeometry activation; ///< Exact common arena geometry, independently priced at wire decode.
    int tp_coordinate = -1; ///< Original communicator coordinate; absent for replicated requests.
    ImmutableRowTableTPTopology tp_topology = ImmutableRowTableTPTopology::None;
    /**
     * @brief Compare immutable source geometry without consulting live execution state.
     * @param other Another source-authored temporal consumer observation.
     * @return Full geometry/resource equality, including retained row capacity.
     */
    bool operator==(const ImmutableRowTableExecutionConsumer &other) const noexcept {
        return resource.sameAllocator(other.resource) && rows==other.rows && batch==other.batch &&
            context==other.context && local_heads==other.local_heads && first_layer==other.first_layer &&
            last_layer==other.last_layer && shards==other.shards && mtp_rows==other.mtp_rows &&
            retained_mtp_rows==other.retained_mtp_rows && request_capacity==other.request_capacity &&
            sameActivation(activation,other.activation) && tp_coordinate==other.tp_coordinate && tp_topology==other.tp_topology;
    }
private:
    /** @return All immutable activation geometry; never a count value or allocation address. */
    static bool sameActivation(const ActivationGraphMemoryGeometry &a,const ActivationGraphMemoryGeometry &b) noexcept
    {
        const auto key = [](const ActivationGraphMemoryGeometry &g)
        {
            return std::tuple{g.ngram_lookup,
                              g.batch_size,
                              g.resident_graph_rows,
                              g.context_tokens,
                              g.local_d_ff,
                              g.local_n_heads,
                              g.local_n_kv_heads,
                              g.local_vocab,
                              g.replicated_dense_decode,
                              g.first_layer,
                              g.last_layer,
                              g.total_shards,
                              g.routed_compute_policy,
                              g.mtp_target_query_rows,
                              g.retained_mtp_candidate_rows,
                              g.mtp_terminal_logits_layout,
                              g.generation_request_capacity,
                              g.ngram_fixed_head_participants,
                              g.attention_arithmetic,
                              g.attention_phase_schedule};
        };
        return key(a)==key(b);
    }
};
/** @brief Typed immutable provision minted only by the canonical source/geometry planner. */
class ImmutableRowTableExecutionProvision final {
public:
    /** @return Exact requested policy, never the later selected storage tier. */
    const ImmutableRowTablePolicy &requested() const noexcept { return requested_; }
    /** @return Full source/layout metadata that determined the declared arena roles. */
    const Qwen4ExpMemoryGeometry &source() const noexcept { return source_; }
    /** @return Exact per-instance retained graph geometry; temporal instances are additive. */
    const std::vector<ImmutableRowTableExecutionConsumer> &consumers() const noexcept { return consumers_; }
    /**
     * @brief Derive the sole source-range authority from the checked declaration coordinates.
     * @param input Original extent and observed actual consumer resources.
     * @return Fixed placement; no capacity decision or materialization is performed.
     * @throws std::invalid_argument For unsupported/foreign/incomplete/reordered topology or source.
     */
    std::shared_ptr<const ResolvedImmutableRowTablePlacement> fixedHeadPlacement(
        const ImmutableRowTablePlacementInput &input) const;
    /** @return Checked maximum original ngram ID rows across actual serialized temporal consumers. */
    std::size_t maximumLiveIds() const;
    /** @return Compiled host locality distinct from physical allocator identity. */
    const ImmutableRowTableHostPlacement &hostPlacement() const noexcept { return host_placement_; }
    /** @return Only mandatory execution bytes, without original native table payload. */
    const PhysicalMemoryPlan &executionPlan() const noexcept { return execution_; }
    /** @return Complete source-owned fixed footprint against which final base coverage is checked. */
    const PhysicalMemoryPlan &requiredBase() const noexcept { return required_base_; }
    /**
     * @brief Reject foreign request identity or an incomplete pre-quota base.
     * @param requested Exact requested policy retained through quota freezing.
     * @param input Final original source/physical consumer/locality observation.
     * @param base Frozen full plan including experts but excluding this table payload.
     * @throws std::invalid_argument For stale/missing provision or underpriced fixed owners.
     */
    void require(const ImmutableRowTablePolicy &requested,const ImmutableRowTablePlacementInput &input,
        const PhysicalMemoryPlan &base) const;
private:
    friend class MemoryPlanner;
    /**
     * @brief Retain only canonical setup metadata; this creates no allocation authority.
     * @param requested Lossless pre-quota policy.
     * @param source Full authenticated QFN layout/source extent.
     * @param host_placement Exact compiled RAM locality.
     * @param consumers Checked temporal graph-instance geometry.
     * @param execution Canonical mutually-exclusive execution bound.
     * @param required_base Complete fixed consumer footprint including that bound.
     */
    ImmutableRowTableExecutionProvision(ImmutableRowTablePolicy requested,Qwen4ExpMemoryGeometry source,
        ImmutableRowTableHostPlacement host_placement,std::vector<ImmutableRowTableExecutionConsumer> consumers,
        PhysicalMemoryPlan execution,PhysicalMemoryPlan required_base)
        : requested_(requested),source_(std::move(source)),host_placement_(host_placement),
          consumers_(std::move(consumers)),execution_(std::move(execution)),required_base_(std::move(required_base)) {}
    ImmutableRowTablePolicy requested_;
    Qwen4ExpMemoryGeometry source_;
    ImmutableRowTableHostPlacement host_placement_;
    std::vector<ImmutableRowTableExecutionConsumer> consumers_;
    PhysicalMemoryPlan execution_,required_base_;
};
} // namespace llaminar2
