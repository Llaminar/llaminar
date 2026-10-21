/**
 * @file Qwen4ExpPredictorProjectionLayout.cpp
 * @brief Authenticate dense predictor layout from existing TP assignment and frozen native matrices.
 *
 * Shape/slice metadata comes from the canonical frozen source. Prepared refs
 * retain the real participant and model identity; the native matrix bytes are
 * neither decoded nor repacked by this graph-construction adapter.
 */
#include "Qwen4ExpPredictorProjectionLayout.h"
#include "../../../../models/GraphTypes.h"
#include "../../../../models/qwen4exp/Qwen4ExpConfig.h"
#include "../../../../execution/local_execution/graph/GraphBuildUtils.h"
namespace llaminar2 {
Qwen4ExpPredictorProjectionLayout Qwen4ExpPredictorProjectionLayout::fromFrozen(
    MTPShiftedKVHeadLayout selected,const GraphConfig &config,const MTPDepthWeightBindings &weights,DeviceId device)
{
    if (!config.flash_next || weights.source_layer_index!=static_cast<int>(config.flash_next->mainLayers()) || weights.depth_index!=0)
        throw std::invalid_argument("Dense0 projection requires the separate sourceN/depth0 policy of its exact main model");
    const int participants=config.tp_config ? config.tp_config->worldSize() : 1;
    const int participant=config.tp_config ? config.local_rank : 0;
    int kv_head_start=0;
    const auto canonical=resolveMTPShiftedKVHeadLayout(config.mtp,config.dense_tp_enabled,participants);
    if (config.tp_config) {
        const auto &assignment=config.tp_config->forRank(participant);
        kv_head_start=assignment.kv_head_start;
        if (assignment.device!=device || (selected==MTPShiftedKVHeadLayout::PrimaryTensorParallelShard &&
            config.dense_tp_enabled && participants>1 && config.qkv_column_parallel &&
            (assignment.head_start!=config.head_start || assignment.head_count!=config.local_n_heads ||
             assignment.kv_head_count!=config.local_n_kv_heads)))
            throw std::invalid_argument("Dense0 projection layout disagrees with its canonical TP participant/head assignment");
    }
    const auto &f=weights.fa_block;
    const auto tensor=[&](const WeightBinding *binding,bool attention)->const TensorBase & {
        if (!binding || !binding->immutable || !binding->tensor || binding->identity.layer!=weights.source_layer_index ||
            !binding->prepared || (attention && binding->prepared->device!=device) ||
            binding->prepared->model_id!=binding->identity.model_id || binding->prepared->binding_id!=binding->binding_id)
            throw std::invalid_argument("Dense0 projection requires its exact immutable source/prepared native matrix identity");
        return *binding->tensor;
    };
    const auto &q=tensor(f.wq,true),&k=tensor(f.wk,true),&v=tensor(f.wv,true),&wo=tensor(f.wo,true);
    const auto &gate=tensor(f.shared_expert_gate,false),&up=tensor(f.shared_expert_up,false),&down=tensor(f.shared_expert_down,false);
    auto decision=validate(selected,{device,canonical,participants,participant,config.dense_tp_enabled,config.qkv_column_parallel,
        config.d_model,config.head_dim,config.n_heads,config.n_kv_heads,config.local_n_heads,config.local_n_kv_heads,
        config.head_start,config.moe.shared_intermediate_size,kv_head_start},
        {q.shape(),k.shape(),v.shape(),wo.shape(),gate.shape(),up.shape(),down.shape(),
         graph_utils::isRowParallelSharded(&wo),graph_utils::isRowParallelSharded(&down)});
    if ((decision.attention_reduction || decision.shared_reduction) &&
        (!config.tp_ctx || config.tp_ctx->degree()!=participants))
        throw std::invalid_argument("Dense0 row-parallel projection requires its admitted typed TP transport context");
    return {selected,device,decision,weights,{f.wq,f.wk,f.wv,f.wo,f.shared_expert_gate,f.shared_expert_up,f.shared_expert_down},
        {&q,&k,&v,&wo,&gate,&up,&down},weights.source_layer_index,weights.depth_index};
}
void Qwen4ExpPredictorProjectionLayout::requireFrozenDepth(const MTPDepthWeightBindings &depth) const
{
    const auto &f=depth.fa_block;
    const std::array<const WeightBinding *,7> current{f.wq,f.wk,f.wv,f.wo,f.shared_expert_gate,f.shared_expert_up,f.shared_expert_down};
    if (&depth!=depth_ || current!=matrices_ || depth.source_layer_index!=source_ || depth.depth_index!=depth_index_)
        throw std::invalid_argument("Dense0 projection layout cannot be rebound to a different frozen source");
    for (size_t i=0;i<current.size();++i)
        if (!current[i] || !current[i]->immutable || current[i]->tensor!=tensors_[i] || current[i]->identity.layer!=source_)
        throw std::invalid_argument("Dense0 projection layout cannot be rebound to a different frozen source");
}
} // namespace llaminar2
