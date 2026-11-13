/**
 * @file Qwen4ExpPredictorResidualBinding.cpp
 * @brief Authenticate Dense0 source roles and borrow existing admitted4H arena roots.
 *
 * Factories inspect immutable setup metadata only. They never load weights,
 * prepare GPU memory, change main source/cache identity or copy a residual.
 */
#include "Qwen4ExpPredictorResidualBinding.h"
#include "../../../../memory/BufferArena.h"
#include "../../../../models/GraphTypes.h"
#include "../../../../tensors/TensorClasses.h"
namespace llaminar2
{
Qwen4ExpPredictorTrunkTarget Qwen4ExpPredictorTrunkTarget::fromArena(BufferArena &arena,
    Qwen4ExpPredictorTrunk role, int rows, int width, DeviceId device)
{
    auto root = arena.getSharedTensor(bufferFor(role));
    if (!root || root->transferStorageOwner() != root.get() || root->is_view() || root->isMapped() ||
        root->is_raw_data_released() || root->memoryResidency() != MemoryResidency::STANDARD ||
        arena.getTensor(bufferFor(role)) != root.get())
        throw std::invalid_argument("Dense0 HC target requires one owned canonical ordinary arena root");
    requireDeclared(role, {bufferFor(role), root->native_type(), root->shape(), root->home_device()}, rows, width, device);
    return {arena, role, *root, width};
}
void Qwen4ExpPredictorResidualBinding::requireFrozenSource(const LayerGraphSourceIdentity &source,
    const MTPDepthWeightBindings &weights)
{
    if (source.sourcePolicy() != GraphLayerSourcePolicy::Qwen4ExpSharedPredictor ||
        source.predictorDepth() != weights.depth_index || weights.depth_index != 0)
        throw std::invalid_argument("Dense0 HC construction requires its paired shared depth0 source");
    source.requireFrozenSource(weights.source_layer_index);
    const auto &extension = weights.fa_block.flash_next;
    if (extension.indexer.query || extension.indexer.key || extension.indexer.query_norm || extension.indexer.key_norm ||
        extension.ple.key || extension.ple.value || extension.ple.key_norm || extension.ple.query_norm ||
        extension.ple.convolution_norm || extension.ple.convolution)
        throw std::invalid_argument("Dense0 predictor cannot bind QSA indexers or PLE state/weights");
    for (const auto *mixer : {&extension.attention, &extension.ffn})
        for (const auto *weight : {mixer->norm, mixer->down, mixer->up, mixer->inject})
            if (!weight || !weight->immutable || !weight->tensor || weight->identity.layer != source.sourceLayer())
                throw std::invalid_argument("Dense0 HC requires immutable exact-source norm/down/up/inject bindings");
}
void Qwen4ExpPredictorResidualBinding::requireLayerSource(const LayerWeights &layer) const
{
    layout_.requireFrozenDepth(*weights_);
    const auto expected=toLegacyLayerWeights(weights_->fa_block);
    const std::array actual{layer.wq,layer.wk,layer.wv,layer.wo,layer.q_norm,layer.k_norm,
        layer.moe_gate,layer.moe_gate_exps,layer.moe_up_exps,layer.moe_down_exps,
        layer.shared_expert_gate,layer.shared_expert_up,layer.shared_expert_down,layer.shared_expert_gate_inp};
    const std::array source{expected.wq,expected.wk,expected.wv,expected.wo,expected.q_norm,expected.k_norm,
        expected.moe_gate,expected.moe_gate_exps,expected.moe_up_exps,expected.moe_down_exps,
        expected.shared_expert_gate,expected.shared_expert_up,expected.shared_expert_down,expected.shared_expert_gate_inp};
    if (actual!=source)
        throw std::invalid_argument("Dense0 common FA/FFN operands must preserve the sealed native source48 bundle");
}
Qwen4ExpPredictorResidualBinding Qwen4ExpPredictorResidualBinding::fromArena(
    Qwen4ExpPredictorInvocation invocation, const LayerGraphSourceIdentity &source,
    const MTPDepthWeightBindings &weights, Qwen4ExpPredictorRows rows,
    Qwen4ExpPredictorProjectionLayout layout, BufferArena &arena, int width)
{
    switch (invocation)
    {
    case Qwen4ExpPredictorInvocation::ShiftedPrefill:
        // No Draft destination is inspected: the bucket owns only input-mix/KV writes.
        return forShiftedPrefill(source, weights, std::move(rows), std::move(layout));
    case Qwen4ExpPredictorInvocation::Draft:
    {
        const auto attention = Qwen4ExpPredictorTrunkTarget::fromArena(arena,
            Qwen4ExpPredictorTrunk::Intermediate, rows.tokens().physicalRows(), width, rows.device());
        const auto ffn = Qwen4ExpPredictorTrunkTarget::fromArena(arena,
            Qwen4ExpPredictorTrunk::Terminal, rows.tokens().physicalRows(), width, rows.device());
        return admit(source, weights, std::move(rows), std::move(layout), attention, ffn);
    }
    }
    throw std::invalid_argument("Unknown Dense0 predictor residual invocation");
}

Qwen4ExpPredictorResidualBinding Qwen4ExpPredictorResidualBinding::forShiftedPrefill(
    const LayerGraphSourceIdentity &source, const MTPDepthWeightBindings &weights,
    Qwen4ExpPredictorRows rows, Qwen4ExpPredictorProjectionLayout layout)
{
    requireFrozenSource(source,weights);
    layout.requireFrozenDepth(weights);
    rows.requireCompactCPUQuery();
    if(layout.device()!=rows.device())
        throw std::invalid_argument("Dense0 shifted-prefill projection and original rows require the same participant");
    return {weights,std::move(rows),std::move(layout)};
}
Qwen4ExpPredictorResidualBinding Qwen4ExpPredictorResidualBinding::admit(
    const LayerGraphSourceIdentity &source, const MTPDepthWeightBindings &weights,
    Qwen4ExpPredictorRows rows, Qwen4ExpPredictorProjectionLayout layout,
    Qwen4ExpPredictorTrunkTarget attention_output,
    Qwen4ExpPredictorTrunkTarget ffn_output)
{
    requireFrozenSource(source, weights);
    layout.requireFrozenDepth(weights);
    if (layout.device()!=rows.device())
        throw std::invalid_argument("Dense0 projection layout and original rows must name the same participant");
    if (attention_output.role() != Qwen4ExpPredictorTrunk::Intermediate ||
        ffn_output.role() != Qwen4ExpPredictorTrunk::Terminal ||
        &attention_output.arena() != &ffn_output.arena() ||
        attention_output.binding().tensor == ffn_output.binding().tensor ||
        attention_output.width() != ffn_output.width())
        throw std::invalid_argument("Dense0 attention/FFN HC writes require distinct same-arena intermediate/terminal roots");
    const auto check = [&](const auto &target) {
        auto &tensor = *target.binding().tensor;
        Qwen4ExpPredictorTrunkTarget::requireDeclared(target.role(),
            {target.binding().buffer, tensor.native_type(), tensor.shape(), tensor.home_device()},
            rows.tokens().physicalRows(), target.width(), rows.device());
    };
    check(attention_output); check(ffn_output);
    return {weights, std::move(rows), std::move(layout), std::move(attention_output), std::move(ffn_output)};
}
} // namespace llaminar2
