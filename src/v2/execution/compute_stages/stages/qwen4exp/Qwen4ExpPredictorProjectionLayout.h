/**
 * @file Qwen4ExpPredictorProjectionLayout.h
 * @brief Checked existing MTP dense-placement policy over exact frozen local matrices.
 *
 * Predictor helpers receive this immutable layout explicitly. It neither
 * changes main configuration nor borrows an ambient replicated-weight scope.
 * Routed expert participant ownership remains the existing independent policy.
 */
#pragma once
#include "../../../../execution/config/RuntimeConfig.h"
#include <span>
#include <array>
#include <limits>
namespace llaminar2 {
struct GraphConfig;
struct MTPDepthWeightBindings;
struct WeightBinding;
class TensorBase;
/** @brief Existing resolved participant geometry, independent of mutable request state. */
struct Qwen4ExpPredictorParticipantGeometry {
    DeviceId device;
    MTPShiftedKVHeadLayout canonical_layout;
    int participants,participant;
    bool dense_tp,qkv_column_parallel;
    int hidden,head_width,global_query_heads,global_kv_heads;
    int local_query_heads,local_kv_heads,query_head_start;
    int shared_intermediate;
    int kv_head_start;
};
/** @brief Actual native local matrix dimensions and the canonical row-parallel slice tags. */
struct Qwen4ExpPredictorLocalMatrices {
    std::span<const size_t> q,k,v,wo,shared_gate,shared_up,shared_down;
    bool wo_row_parallel,shared_down_row_parallel;
};
/** @brief Pure validation result; not an authenticated frozen construction capability. */
struct Qwen4ExpPredictorProjectionDecision {
    int query_heads,kv_heads,query_start,kv_start;
    bool attention_reduction,shared_reduction;
};
/** @brief Sealed predictor-only heads and reduction decisions authenticated against frozen shapes. */
class Qwen4ExpPredictorProjectionLayout final {
public:
    /**
     * @brief Admit exact setup metadata without inspecting payloads or preparing native memory.
     * @param selected Explicit existing canonical predictor cache/dense placement policy.
     * @param p Existing participant assignment and global checkpoint widths.
     * @param w Actual frozen local weight matrix shapes and slice tags.
     * @return Complete immutable head/reduction contract.
     * @throws std::invalid_argument For foreign policy/participant, missing or incompatible shards.
     * @throws std::overflow_error When native projected widths exceed INT32 geometry.
     */
    static Qwen4ExpPredictorProjectionDecision validate(MTPShiftedKVHeadLayout selected,
        const Qwen4ExpPredictorParticipantGeometry &p,const Qwen4ExpPredictorLocalMatrices &w)
    {
        if (selected!=p.canonical_layout ||
            (selected!=MTPShiftedKVHeadLayout::PrimaryTensorParallelShard && selected!=MTPShiftedKVHeadLayout::FullModelPerParticipant) ||
            !p.device.is_valid() || p.participants<=0 || p.participant<0 || p.participant>=p.participants ||
            p.hidden<=0 || p.head_width<=0 || p.global_query_heads<=0 || p.global_kv_heads<=0 ||
            p.global_query_heads%p.global_kv_heads || p.shared_intermediate<=0)
            throw std::invalid_argument("Dense0 projection layout requires its canonical participant policy and positive global geometry");
        const bool sharded=selected==MTPShiftedKVHeadLayout::PrimaryTensorParallelShard && p.dense_tp && p.participants>1;
        const int q=sharded && p.qkv_column_parallel ? p.local_query_heads : p.global_query_heads;
        const int kv=sharded && p.qkv_column_parallel ? p.local_kv_heads : p.global_kv_heads;
        const int start=sharded && p.qkv_column_parallel ? p.query_head_start : 0;
        const int kv_start=sharded && p.qkv_column_parallel ? p.kv_head_start : 0;
        if (q<=0 || kv<=0 || q>p.global_query_heads || kv>p.global_kv_heads || start<0 ||
            start>p.global_query_heads-q || (q<p.global_query_heads && !sharded) || kv_start<0 || kv_start>p.global_kv_heads-kv ||
            start/(p.global_query_heads/p.global_kv_heads)<kv_start ||
            (start+q-1)/(p.global_query_heads/p.global_kv_heads)>=kv_start+kv)
            throw std::invalid_argument("Dense0 projection layout requires its exact admitted native head assignment");
        if (p.global_query_heads>std::numeric_limits<int>::max()/p.head_width/2 ||
            p.global_kv_heads>std::numeric_limits<int>::max()/p.head_width)
            throw std::overflow_error("Dense0 QKV projected widths exceed native INT32 geometry");
        const auto matrix=[](std::span<const size_t> shape,size_t rows,size_t columns) {
            return shape.size()==2 && shape[0]==rows && shape[1]==columns;
        };
        const bool attention_partial=q<p.global_query_heads;
        if (!matrix(w.q,static_cast<size_t>(q)*p.head_width*2,p.hidden) ||
            !matrix(w.k,static_cast<size_t>(kv)*p.head_width,p.hidden) ||
            !matrix(w.v,static_cast<size_t>(kv)*p.head_width,p.hidden) ||
            !matrix(w.wo,p.hidden,static_cast<size_t>(q)*p.head_width) ||
            w.wo_row_parallel!=attention_partial)
            throw std::invalid_argument("Dense0 frozen Q/K/V/Wo do not match its explicit full or participant-local layout");
        if (w.shared_gate.size()!=2 || w.shared_gate[0]==0 || w.shared_gate[0]>static_cast<size_t>(p.shared_intermediate) ||
            !matrix(w.shared_gate,w.shared_gate[0],p.hidden) || !matrix(w.shared_up,w.shared_gate[0],p.hidden) ||
            !matrix(w.shared_down,p.hidden,w.shared_gate[0]))
            throw std::invalid_argument("Dense0 frozen shared gate/up/down dimensions disagree");
        const bool shared_partial=w.shared_gate[0]<static_cast<size_t>(p.shared_intermediate);
        if ((shared_partial && !sharded) || w.shared_down_row_parallel!=shared_partial)
            throw std::invalid_argument("Dense0 shared-expert partial requires its real admitted row-parallel shard");
        return {q,kv,start,kv_start,attention_partial,shared_partial};
    }
    /**
     * @brief Project existing configuration/frozen matrices into the same pure checked contract.
     * @param selected Canonical existing MTP dense/cache policy, never an inferred fallback.
     * @param config Unmodified main graph configuration and TP assignment.
     * @param weights Exact immutable shared predictor depth/source bindings.
     * @param device Executing participant, authenticated against the resolved assignment.
     * @return Sealed geometry; performs no allocation, payload read or source replacement.
     * @throws std::exception For invalid source, prepared participant, shapes or assignment.
     */
    static Qwen4ExpPredictorProjectionLayout fromFrozen(MTPShiftedKVHeadLayout selected,
        const GraphConfig &config,const MTPDepthWeightBindings &weights,DeviceId device);
    /** @return Exact local query-head count used by native split, norm, rotary and attention. */
    int queryHeads() const noexcept{return decision_.query_heads;}
    /** @return Exact local KV-head count used by native projections/publication/attention. */
    int kvHeads() const noexcept{return decision_.kv_heads;}
    /** @return Global origin of this participant's query heads; full replicas start0. */
    int queryHeadStart() const noexcept{return decision_.query_start;}
    /** @return Whether this real row-parallel Wo partial needs canonical TP reconstruction. */
    bool attentionNeedsReduction() const noexcept{return decision_.attention_reduction;}
    /** @return Whether this real shared-down partial needs canonical TP reconstruction. */
    bool sharedNeedsReduction() const noexcept{return decision_.shared_reduction;}
    /** @return Global KV-head origin, authenticated against the same participant assignment. */
    int kvHeadStart() const noexcept{return decision_.kv_start;}
    /** @brief Reject using a layout admitted over a different frozen source/weight projection.
     * @param depth Exact frozen input supplied to fromFrozen.
     * @throws std::invalid_argument For a different source object or a changed native matrix pointer.
     */
    void requireFrozenDepth(const MTPDepthWeightBindings &depth) const;
    /** @return The exact existing placement policy admitted at construction. */
    MTPShiftedKVHeadLayout policy() const noexcept{return policy_;}
    /** @return Exact immutable participant owning attention execution. */
    DeviceId device() const noexcept{return device_;}
private:
    /** @brief Only checked metadata admission can issue this policy value. */
    Qwen4ExpPredictorProjectionLayout(MTPShiftedKVHeadLayout policy,DeviceId device,
        Qwen4ExpPredictorProjectionDecision decision,const MTPDepthWeightBindings &depth,
        std::array<const WeightBinding *,7> matrices,std::array<const TensorBase *,7> tensors,int source,int depth_index)
        :policy_(policy),device_(device),decision_(decision),depth_(&depth),matrices_(matrices),tensors_(tensors),source_(source),depth_index_(depth_index){}
    MTPShiftedKVHeadLayout policy_;
    DeviceId device_;
    Qwen4ExpPredictorProjectionDecision decision_;
    const MTPDepthWeightBindings *depth_;
    std::array<const WeightBinding *,7> matrices_;
    std::array<const TensorBase *,7> tensors_;
    int source_,depth_index_;

};
} // namespace llaminar2
