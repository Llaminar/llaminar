/**
 * @file Qwen4ExpPredictorGlobalBindings.cpp
 * @brief Authenticate selected target globals against existing frozen/native prepared authorities.
 *
 * CPU original source slices and GPU PreparedEmbeddingHandle own their
 * respective vocabulary ranges. No CPU prepared copy is constructed. A sidecar never
 * creates a renamed embedding/output or changes the target source identity.
 * Missing selected replica/transport is a hard admission failure.
 */
#include "Qwen4ExpPredictorGlobalBindings.h"
#include <limits>
#include "../../../../models/GraphTypes.h"
#include "../../../../loaders/PreparedWeightStore.h"
#include "../../../../tensors/TensorClasses.h"
namespace llaminar2 {
namespace {
/** @brief Project metadata from a real immutable target tensor, without preparing or reading payloads. */
Qwen4ExpPredictorGlobalDescriptor descriptor(const WeightBinding *binding)
{
    if(!binding || !binding->immutable || !binding->tensor)
        throw std::invalid_argument("Dense0 selected target global is missing its existing immutable tensor");
    return {binding->identity,binding->binding_id,binding->prepared,binding->tensor->native_type(),binding->tensor->shape()};
}
}
Qwen4ExpPredictorGlobalBindings Qwen4ExpPredictorGlobalBindings::fromFrozen(const GraphConfig &config,
    const ModelWeightBindings &primary,const ModelWeightBindings &replicated,PreparedWeightStore &store,
    DeviceId device,Qwen4ExpPredictorInvocation invocation)
{
    const auto selection=select(invocation,config.dense_tp_enabled,config.dense_tp_decode_replicated,
        config.dense_tp_decode_mirrored_embedding,config.mtpTerminalLogitsLayout(),config.lm_head_column_parallel);
    const auto *embedding=selection.embedding==Qwen4ExpPredictorGlobalSource::PreparedReplica
        ? replicated.embedding_table : primary.embedding_table;
    const auto selected=descriptor(embedding),target=descriptor(primary.embedding_table);
    int rows = 0;
    int offset = 0;
    if (device.is_cpu())
    {
        // CPU embedding kernels read the original frozen native tensor. The
        // weight plan owns its vocabulary slice and RequiredForCPUExecution
        // policy; manufacturing a prepared handle would duplicate that owner.
        const auto &slice = embedding->slice;
        const auto &residency = embedding->residency;
        if (selected.prepared || config.vocab_size <= 0 || config.d_model <= 0 ||
            residency.home_device != device || residency.resident_device != device ||
            residency.host_policy != WeightHostPolicy::RequiredForCPUExecution ||
            !residency.raw_host_data_available || embedding->tensor->is_raw_data_released() ||
            slice.source_rows != static_cast<size_t>(config.vocab_size) ||
            slice.source_cols != static_cast<size_t>(config.d_model) ||
            slice.row_count == 0 || slice.row_count > slice.source_rows ||
            slice.row_start > slice.source_rows - slice.row_count ||
            slice.col_start != 0 || slice.col_count != slice.source_cols ||
            slice.expert_start != 0 || slice.expert_count != 0 || !slice.expert_ids.empty())
            throw std::invalid_argument("Dense0 CPU embedding requires its exact retained original native source slice and CPU execution owner");
        // Both values are bounded above by the positive int vocabulary size.
        rows = static_cast<int>(slice.row_count);
        offset = static_cast<int>(slice.row_start);
    }
    else
    {
        if (!selected.prepared)
            throw std::invalid_argument("Dense0 target embedding has no admitted native prepared reference");
        const auto *native = store.embeddingHandle(*selected.prepared);
        if (!native || native->tensor != embedding->tensor || native->device_id != device || !native->weights ||
            native->weights->total_vocab != static_cast<size_t>(config.vocab_size) || native->weights->d_model != config.d_model ||
            native->weights->vocab_size > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            native->weights->vocab_offset > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            native->weights->vocab_offset > native->weights->total_vocab ||
            native->weights->vocab_size > native->weights->total_vocab - native->weights->vocab_offset)
            throw std::invalid_argument("Dense0 embedding must reuse its exact existing prepared native vocabulary range");
        rows = static_cast<int>(native->weights->vocab_size);
        offset = static_cast<int>(native->weights->vocab_offset);
    }
    requireDescriptor(WeightRole::Embedding,selected,target,device,rows,config.d_model);
    if(selection.embedding==Qwen4ExpPredictorGlobalSource::PreparedReplica && (rows!=config.vocab_size || offset!=0))
        throw std::invalid_argument("Dense0 selected embedding replica must own the complete target vocabulary");
    if(rows!=config.vocab_size) {
        const auto *assignment=config.getAssignment();
        if(!assignment || assignment->device!=device || assignment->vocab_start!=offset || assignment->vocab_count!=rows ||
            !config.tp_ctx || config.tp_ctx->degree()<=1)
            throw std::invalid_argument("Dense0 sharded embedding requires its canonical vocabulary assignment and TP transport");
    }
    const WeightBinding *head=nullptr;int head_rows=0;
    if(selection.head!=Qwen4ExpPredictorGlobalSource::NotConsumed) {
        head=selection.head==Qwen4ExpPredictorGlobalSource::PreparedReplica?replicated.lm_head:primary.lm_head;
        head_rows=selection.terminal_layout==MTPTerminalLogitsLayout::VocabularyShardPerParticipant?config.vocab_local:config.vocab_size;
        requireDescriptor(WeightRole::LMHead,descriptor(head),descriptor(primary.lm_head),device,head_rows,config.d_model);
        if(selection.terminal_layout==MTPTerminalLogitsLayout::VocabularyShardPerParticipant) {
            const auto *assignment=config.getAssignment();
            if(!assignment || assignment->device!=device || assignment->vocab_count!=head_rows || !config.tp_ctx || config.tp_ctx->degree()<=1)
                throw std::invalid_argument("Dense0 vocabulary-sharded head requires its canonical assignment and typed TP transport");
        }
        if(!head->prepared || !store.gemmKernel(*head->prepared))
            throw std::invalid_argument("Dense0 output must reuse its existing target prepared GEMM owner");
    }
    return {*embedding,head,selection,offset,rows,head_rows,config.vocab_size};
}
} // namespace llaminar2
