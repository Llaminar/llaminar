/**
 * @file Qwen4ExpPredictorBuilder.h
 * @brief Dense0/PerBranch input mixing and head collapse over canonical native stages.
 *
 * This construction-only builder borrows frozen source48 weights and existing
 * activation roles. Its FC stage retains whole-storage reshape handles; every
 * counted operation derives from the same original token-row descriptor. There
 * is no count upload, separate weight owner, activation allocation or new ledger.
 */
#pragma once
#include "Qwen4ExpPredictorResidualBinding.h"
#include "Qwen4ExpPredictorGlobalBindings.h"
#include "../../../../models/qwen4exp/Qwen4ExpPredictorProjectionGeometry.h"
#include <array>
#include <span>
namespace llaminar2 {
class ComputeGraph;
class PreparedWeightStore;
class IComputeStage;
struct ActivationBuffers;
struct MTPForwardInput;
struct MTPForwardOutput;
/** @brief Checked whole-storage FC reshape geometry, independent of physical preparation. */
struct Qwen4ExpPredictorFCGeometry final {
    std::array<size_t,2> input,output;
    /**
     * @brief Validate both declared roots and derive four contiguous branch rows per token.
     * @param concat Actual MTP_CONCAT root [R,8H].
     * @param projected Actual MTP_PROJECTED root [R,4H].
     * @param token_rows Current positive admitted query rows M, at most R.
     * @param hidden Authenticated per-branch compute width H.
     * @return Exact whole-storage [4R,2H] and [4R,H] shapes; no pointer is observed.
     * @throws std::invalid_argument For rank, capacity or width mismatch.
     * @throws std::overflow_error For native geometry or physical byte overflow.
     */
    static Qwen4ExpPredictorFCGeometry admit(std::span<const size_t> concat,
        std::span<const size_t> projected, int token_rows, int hidden)
    {
        if (hidden<=0 || token_rows<=0 || concat.size()!=2 || projected.size()!=2 ||
            concat[0]<static_cast<size_t>(token_rows) || concat[0]!=projected[0])
            throw std::invalid_argument("Dense0 FC requires same-capacity complete input/output rank-two roots");
        if (hidden>std::numeric_limits<int>::max()/8 || concat[0]>static_cast<size_t>(std::numeric_limits<int>::max()/4))
            throw std::overflow_error("Dense0 FC branch expansion exceeds native INT32 matrix geometry");
        if (concat[1]!=static_cast<size_t>(hidden)*8 || projected[1]!=static_cast<size_t>(hidden)*4)
            throw std::invalid_argument("Dense0 FC requires native [R,8H] and [R,4H] root shapes");
        if (concat[0]>std::numeric_limits<size_t>::max()/concat[1]/sizeof(float))
            throw std::overflow_error("Dense0 FC whole-storage bytes overflow");
        return {{static_cast<size_t>(Qwen4ExpPredictorProjectionGeometry::perBranchRows(static_cast<int>(concat[0]),qwen4exp::kResidualBranches)),static_cast<size_t>(hidden)*2},{static_cast<size_t>(Qwen4ExpPredictorProjectionGeometry::perBranchRows(static_cast<int>(projected[0]),qwen4exp::kResidualBranches)),static_cast<size_t>(hidden)}};
    }
};
/** @brief Frozen construction policy; selected main/sidecar identities never depend on ambient scopes. */
struct Qwen4ExpPredictorContext final {
    FullAttentionLayerIdentity layer;
    Qwen4ExpPredictorRows rows;
    int hidden,low_rank;
    float epsilon;
    DeviceId device;
    const IMPIContext *mpi;
    std::string prefix;
};
/** @brief Exact setup-only Dense0 input mix over one existing activation arena. */
class Qwen4ExpPredictorBuilder final {
public:
    /**
     * @brief Append existing target embedding and optional explicit canonical reconstruction.
     * @param graph Participant-local graph receiving this native producer chain.
     * @param arena Existing activation owner of MTP_EMBEDDING.
     * @param context Same explicit source/count/geometry policy used by the complete predictor.
     * @param globals Sealed target source and native vocabulary range.
     * @param input Canonical token owner and exact query invocation.
     * @param store Existing model prepared authority.
     * @param reconstruction Existing typed TP stage, mandatory exactly for a declared vocab shard.
     * @return Reconstructed H-wide embedding producer.
     * @throws std::exception For missing selected source/token owner or contradictory collective.
     */
    static std::string appendEmbedding(ComputeGraph &graph,BufferArena &arena,
        const Qwen4ExpPredictorContext &context,const Qwen4ExpPredictorGlobalBindings &globals,
        const MTPForwardInput &input,PreparedWeightStore &store,std::unique_ptr<IComputeStage> reconstruction);
    /**
     * @brief Bind the existing canonical MTP roles for the shared FA/FFN implementation.
     * @param arena Existing root storage owner, with no allocations made by this call.
     * @param output Exact caller output descriptors to authenticate against arena roles.
     * @param device Participant owning every native activation root.
     * @param kv_only Excludes Q/gate/routed/head consumer roles for shifted prefill.
     * @return Borrowed activation projection with H-wide compute scratch and 4H trunk.
     * @throws std::exception For missing/foreign/rebound roles or illegal view storage.
     */
    static ActivationBuffers activationBuffers(BufferArena &arena,const MTPForwardOutput &output,
        DeviceId device,bool kv_only);
    /**
     * @brief Append the target-owned output matrix over predictor-specific HC-head output.
     * @param graph Existing graph after its dedicated HC head collapse.
     * @param arena Same existing activation owner.
     * @param context Exact original token rows and participant.
     * @param globals Sealed canonical target output ownership/layout.
     * @param store Existing prepared model store.
     * @param serial_partition_width Existing canonical mirrored-head arithmetic partition, zero otherwise.
     * @param head_ready Exact completed H-wide head-read producer.
     * @return Participant-local logits producer; caller uses the existing canonical terminal collective.
     * @throws std::exception For unconsumed/foreign head, missing producer or wrong prepared identity.
     */
    static std::string appendVocabulary(ComputeGraph &graph,BufferArena &arena,
        const Qwen4ExpPredictorContext &context,const Qwen4ExpPredictorGlobalBindings &globals,
        PreparedWeightStore &store,int serial_partition_width,const std::string &head_ready);
    /**
     * @brief Append per-branch hidden norm, embedding norm, branch concat and native original FC.
     * @param graph Participant-local graph already containing the canonical embedding producer.
     * @param arena Sole activation owner; all operands are existing admitted roles.
     * @param context Sealed source/cache/dense and original token count policy.
     * @param weights Frozen native sidecar depth; never a synthesized weight view.
     * @param store Canonical prepared weight authority for FC.
     * @param hidden Original pre-collapse4H hidden input and its exact arena identity.
     * @param invocation Existing authenticated draft versus shifted-prefill role; only draft publishes readonly input.
     * @param embedding_ready Existing target-embedding reconstruction producer.
     * @param hidden_ready Existing hidden producer, empty only for an admitted graph input.
     * @return FC output producer, publishing MTP_PROJECTED4H without copying.
     * @throws std::exception For invalid roles, source, storage, geometry or prepared identity.
     */
    static std::string appendInputMix(ComputeGraph &graph, BufferArena &arena,
        const Qwen4ExpPredictorContext &context, const MTPDepthWeightBindings &weights,
        PreparedWeightStore &store, Qwen4ExpFloatBinding hidden, Qwen4ExpPredictorInvocation invocation,
        const std::string &embedding_ready, const std::string &hidden_ready);
    /**
     * @brief Collapse the direct recursive4H MTP_HIDDEN output with the predictor's own HC head.
     * @param graph Existing participant graph after the shared native FA/FFN bodies.
     * @param arena Same activation owner used by input mix and direct residual writes.
     * @param context Same immutable predictor source/count/arithmetic contract.
     * @param weights Frozen source48 predictor including its dedicated HC head.
     * @param store Existing native prepared projection authority.
     * @param trunk_ready Exact final FFN write producer of MTP_HIDDEN.
     * @return H-wide MTP_NORM_EMBEDDING producer consumed by the target-owned LM head.
     * @throws std::exception For missing head/source/arena or geometry contradictions.
     */
    static std::string appendHeadRead(ComputeGraph &graph, BufferArena &arena,
        const Qwen4ExpPredictorContext &context, const MTPDepthWeightBindings &weights,
        PreparedWeightStore &store, const std::string &trunk_ready);
};
} // namespace llaminar2
