/**
 * @file Qwen4ExpPredictorGlobalBindings.h
 * @brief Explicit frozen target embedding/output selection for Dense0 consumers.
 *
 * Invocation kind selects consumed roles independently of matrix row count.
 * Target globals retain their main logical/native identity; a selected existing
 * replica remains that target source, never an invented sidecar weight owner.
 */
#pragma once
#include "../../../../execution/config/RuntimeConfig.h"
#include "../../../../loaders/WeightPlan.h"
#include "../../../../tensors/TensorType.h"
#include <span>
namespace llaminar2 {
struct GraphConfig;
struct ModelWeightBindings;
class PreparedWeightStore;
/** @brief Explicit transaction role, never inferred from a shape threshold. */
enum class Qwen4ExpPredictorInvocation { Draft, ShiftedPrefill };
/** @brief Existing target source projection selected before graph insertion. */
enum class Qwen4ExpPredictorGlobalSource { Primary, PreparedReplica, NotConsumed };
/** @brief Pure role selection; only fromFrozen authenticates native operands. */
struct Qwen4ExpPredictorGlobalSelection {
    Qwen4ExpPredictorGlobalSource embedding,head;
    MTPTerminalLogitsLayout terminal_layout;
    bool mirrored_head;
};
/** @brief Allocation-free frozen metadata projection used by the native factory and tests. */
struct Qwen4ExpPredictorGlobalDescriptor {
    const WeightIdentity &identity;
    uint64_t binding_id;
    std::optional<PreparedWeightRef> prepared;
    TensorType native_type;
    std::span<const size_t> shape;
};
/** @brief Authenticated borrowed target globals with canonical vocab and terminal ownership. */
class Qwen4ExpPredictorGlobalBindings final {
public:
    /**
     * @brief Select consumed sources from invocation and existing frozen policy, never M.
     * @param invocation Draft versus shifted KV-only prefill transaction.
     * @param dense_tp Existing resolved dense TP policy.
     * @param decode_replicated Existing decoded embedding replication policy.
     * @param decode_mirrored_embedding Existing decoded embedding mirroring policy.
     * @param layout Canonical GraphConfig::mtpTerminalLogitsLayout result.
     * @param primary_head_sharded Existing target head shard policy.
     * @return Pure source selection; no source is substituted when the selected binding is absent.
     * @throws std::invalid_argument For unknown invocation/layout tags.
     */
    static Qwen4ExpPredictorGlobalSelection select(Qwen4ExpPredictorInvocation invocation,
        bool dense_tp,bool decode_replicated,bool decode_mirrored_embedding,
        MTPTerminalLogitsLayout layout,bool primary_head_sharded)
    {
        if ((invocation!=Qwen4ExpPredictorInvocation::Draft && invocation!=Qwen4ExpPredictorInvocation::ShiftedPrefill) ||
            (layout!=MTPTerminalLogitsLayout::FullVocabularyPerParticipant && layout!=MTPTerminalLogitsLayout::VocabularyShardPerParticipant))
            throw std::invalid_argument("Dense0 globals require an explicit invocation and compiled terminal layout");
        if (invocation==Qwen4ExpPredictorInvocation::ShiftedPrefill)
            return {Qwen4ExpPredictorGlobalSource::Primary,Qwen4ExpPredictorGlobalSource::NotConsumed,layout,false};
        const bool mirrored=layout==MTPTerminalLogitsLayout::FullVocabularyPerParticipant && primary_head_sharded;
        return {dense_tp && (decode_replicated || decode_mirrored_embedding)
                    ? Qwen4ExpPredictorGlobalSource::PreparedReplica : Qwen4ExpPredictorGlobalSource::Primary,
                mirrored ? Qwen4ExpPredictorGlobalSource::PreparedReplica : Qwen4ExpPredictorGlobalSource::Primary,
                layout,mirrored};
    }
    /**
     * @brief Authenticate exact main source and its participant-specific native representation.
     * CPU embedding consumes its frozen original native tensor directly; GPU
     * embedding and every LM head retain their existing prepared references.
     * @param role Required Embedding or LMHead semantic role.
     * @param selected Actual selected frozen descriptor, projected from its real tensor.
     * @param target Original primary target descriptor, never a sidecar alias.
     * @param device Exact prepared execution participant.
     * @param rows Required local/full vocabulary rows.
     * @param hidden Exact H-wide matrix columns.
     * @throws std::invalid_argument For source/role/format/shape/prepared identity contradictions.
     */
    static void requireDescriptor(WeightRole role,const Qwen4ExpPredictorGlobalDescriptor &selected,
        const Qwen4ExpPredictorGlobalDescriptor &target,DeviceId device,int rows,int hidden)
    {
        const auto &id=selected.identity;const auto &main=target.identity;
        const bool embedding=role==WeightRole::Embedding;
        // A CPU embedding owns its original native source, not a prepared
        // allocation. Every other consumed role retains its prepared identity.
        const bool native_cpu_embedding = embedding && device.is_cpu() && !selected.prepared;
        const bool prepared_identity_valid = selected.prepared &&
            selected.prepared->model_id == id.model_id &&
            selected.prepared->binding_id == selected.binding_id && selected.prepared->device == device &&
            (embedding ? selected.prepared->kind == PreparedWeightKind::PreparedEmbedding :
                (selected.prepared->kind == PreparedWeightKind::CpuPackedGemm ||
                 selected.prepared->kind == PreparedWeightKind::CudaInt8PackedGemm ||
                 selected.prepared->kind == PreparedWeightKind::RocmInt8PackedGemm));
        if ((!embedding && role!=WeightRole::LMHead) || rows<=0 || hidden<=0 || !device.is_valid() ||
            main.layer!=-1 || main.role!=role || id.layer!=-1 || id.role!=role ||
            main.canonical_name!=(embedding?"token_embd.weight":"output.weight") || id.canonical_name!=main.canonical_name ||
            !main.model_id.value || !main.logical_id || id.model_id!=main.model_id || id.logical_id!=main.logical_id ||
            !selected.binding_id || (!native_cpu_embedding && !prepared_identity_valid) ||
            selected.native_type!=target.native_type || selected.shape.size()!=2 ||
            selected.shape[0]!=static_cast<size_t>(rows) || selected.shape[1]!=static_cast<size_t>(hidden))
            throw std::invalid_argument("Dense0 globals must preserve exact target source/native/prepared vocabulary identity");
    }
    /**
     * @brief Borrow selected immutable globals from the existing canonical model and prepared store.
     * @param config Unmodified main configuration and compiled terminal/TP policies.
     * @param primary Existing frozen target globals.
     * @param replicated Existing admitted target replicas; required only when explicitly selected.
     * @param store Existing prepared store for GPU embedding and all terminal GEMM owners.
     * @param device Executing participant.
     * @param invocation Exact draft versus shifted-prefill consumer role.
     * @return Sealed borrowed bindings; no allocation, preparation, retargeting or payload conversion.
     * @throws std::exception For missing selected operands, invalid identity or vocab/TP geometry.
     */
    static Qwen4ExpPredictorGlobalBindings fromFrozen(const GraphConfig &config,
        const ModelWeightBindings &primary,const ModelWeightBindings &replicated,PreparedWeightStore &store,
        DeviceId device,Qwen4ExpPredictorInvocation invocation);
    /** @return Exact immutable target embedding, including an existing admitted replica when selected. */
    const WeightBinding &embedding() const noexcept{return *embedding_;}
    /** @return Exact target output matrix; shifted-prefill rejects this unconsumed role. */
    const WeightBinding &head() const {
        if(!head_) throw std::logic_error("Shifted Dense0 prefill consumes no terminal head");return *head_;
    }
    /** @return Authenticated target vocabulary size, independent of selected shard geometry. */
    int vocabulary() const noexcept{return vocabulary_;}
    /** @return Frozen native embedding vocabulary origin, authenticated by source slice or GPU prepared handle. */
    int embeddingOffset() const noexcept{return embedding_offset_;}
    /** @return Authenticated native embedding local vocabulary rows. */
    int embeddingRows() const noexcept{return embedding_rows_;}
    /** @return True only when the declared native embedding source needs canonical reconstruction. */
    bool reconstructEmbedding() const noexcept{return embedding_rows_!=vocabulary_;}
    /** @return Exact terminal projection vocabulary width; absent for shifted prefill. */
    int headRows() const noexcept{return head_rows_;}
    /** @return Canonical head source/layout role selected before any stage is inserted. */
    const Qwen4ExpPredictorGlobalSelection &selection() const noexcept{return selection_;}
private:
    /** @brief Only fromFrozen can retain native graph construction capabilities. */
    Qwen4ExpPredictorGlobalBindings(const WeightBinding &embedding,const WeightBinding *head,
        Qwen4ExpPredictorGlobalSelection selection,int offset,int rows,int head_rows,int vocabulary)
        :embedding_(&embedding),head_(head),selection_(selection),embedding_offset_(offset),embedding_rows_(rows),head_rows_(head_rows),vocabulary_(vocabulary){}
    const WeightBinding *embedding_,*head_;
    Qwen4ExpPredictorGlobalSelection selection_;
    int embedding_offset_,embedding_rows_,head_rows_,vocabulary_;
};
} // namespace llaminar2
