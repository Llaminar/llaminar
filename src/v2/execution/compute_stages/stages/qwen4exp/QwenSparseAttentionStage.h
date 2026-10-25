/**
 * @file QwenSparseAttentionStage.h
 * @brief Declarative native sparse-indexer, selected-attention and acceptance stages.
 *
 * All scratch and persistent state are admitted arena views. The execution
 * owner publishes replay and canonical row parameters on the participant
 * device; this stage neither infers horizons from cache counts nor owns a
 * shadow frontier. Native cache storage remains under its cache authority.
 */
#pragma once
#include "Qwen4ExpPrimitiveStage.h"
#include "execution/qwen4exp/ImmutableVerifierPrior.h"
#include "execution/qwen4exp/QwenSparseCounterBinding.h"
#include <array>
#include "../../../../kernels/IKVCache.h"
#include "../../../../kernels/qwen4exp/QwenSparseAttention.h"
#include "../../../../kernels/qwen4exp/QwenSparseReplay.h"

namespace llaminar2
{
/** @brief Canonical construction-only phase; definition belongs to IGraphBuilder. */
enum class ForwardExecutionPhase : std::uint8_t;
/** @brief Packed prior codec binding; it cannot be reinterpreted as an FP32 arena tensor. */
struct QwenSparseTransactionPriorBinding {
    Qwen4ExpIntBinding storage; ///< Exact admitted packed INT32 record, aligned to16 bytes.
};
/** @brief Backend-neutral Qwen sparse operation with sealed arena/cache identity. */
class QwenSparseAttentionStage final : public IComputeStage, public ImmutableVerifierPriorProducer
{
public:
    /** @brief Publish one layer's original canonical cache geometry before any append. */
    struct PrepareReplay
    {
        IKVCache &cache;
        int layer, sequence;
        std::optional<Qwen4ExpIntInput> liveRows; ///< Absent explicitly selects fully active rowCapacity.
        struct ContiguousPositions { int first; }; ///< Immutable CPU position policy, part of capture identity.
        std::variant<Qwen4ExpIntInput,ContiguousPositions> positionIds;
        std::optional<Qwen4ExpIntBinding> rowParameters; ///< Required only on GPU; absent CPU metadata consumes no arena storage.
    };
    /** @brief Sole writer of an immutable original replay and live raw tail. */
    struct SnapshotPrior {
        Qwen4ExpFloatBinding acceptedPartial;
        QwenSparseTransactionPriorBinding output;
        Qwen4ExpIntBinding traffic;
        std::optional<QwenSparseCounterBinding> terminalCounters; ///< Original controller-row borrow, outside per-writer coherence.
    };
    /**
     * @brief Pool/norm/rotate indexer keys and publish exact causal token indices.
     * @details ProvenNoRanking is a decode-admission program: query/queryNorm,
     * preparedQuery and scores are not registered, waited on, published or
     * exposed as mathematical snapshots. Their containing arena may remain
     * admitted for mutually exclusive full-program graphs; no bytes are mirrored.
     */
    struct Indexer
    {
        Qwen4ExpFloatBinding query, rawKey, priorPartial, pooledKeys;
        Qwen4ExpImmutableWeightBinding queryNorm, keyNorm;
        std::optional<Qwen4ExpImmutableWeightBinding> cosine, sine;
        Qwen4ExpFloatBinding preparedQuery, scores;
        Qwen4ExpIntBinding selected, selectedCount;
        int blockCapacity, positionCapacity;
        int rotaryDim = 64;
        float epsilon = 1.0e-6f;
        qwen4exp::QwenSparseScalar representation = qwen4exp::QwenSparseScalar::FP32;
        qwen4exp::QwenSparseRotarySource rotarySource = qwen4exp::QwenSparseRotarySource::TablePlanes;
        float ropeTheta = 10000.0f;
        std::optional<QwenSparseTransactionPriorBinding> transactionPrior; ///< Original typed replay/tail source in verifier mode.
        qwen4exp::QwenSparsePoolSelection poolSelection = qwen4exp::QwenSparsePoolSelection::Installed; ///< Immutable native selection implementation, sealed into capture identity.
        qwen4exp::QwenSparseQueryDemand queryDemand = qwen4exp::QwenSparseQueryDemand::AllLive; ///< Original-query authorship, independent of selection arithmetic.
    };
    /** @brief Immutable selected-read cache identity and explicit publication encoding. */
    struct NativeCache
    {
        IKVCache &cache; ///< Retained native banks/tables; no stage-owned KV mirror.
        int layer, sequence;
        attention::AttentionKeyCachePolicy keyPolicy;
    };
    /** @brief Exact selected-index core attention over native physical cache bytes. */
    struct SelectedAttention
    {
        NativeCache native;
        Qwen4ExpFloatBinding query;
        Qwen4ExpIntBinding selected, selectedCount;
        Qwen4ExpFloatBinding output;
        std::optional<Qwen4ExpFloatBinding> gate;
        std::optional<Qwen4ExpIntBinding> rowParameters; ///< Required GPU canonical AttentionDeviceParams[M].
        std::optional<Qwen4ExpIntBinding> gpuPartials; ///< Required GPU exclusive native partial bytes.
        std::optional<Qwen4ExpIntBinding> observations; ///< Optional exact decoded-coordinate counters.
        std::optional<Qwen4ExpImmutableWeightBinding> keyCosine, keySine;
        int keyPositionCapacity = 0, keyRotaryDim = 64;
        int queryHeads = 24, kvHeads = 2, queryHeadStart = 0, kvHeadStart = 0;
        qwen4exp::QwenSparseRotarySource keyRotarySource = qwen4exp::QwenSparseRotarySource::TablePlanes;
        qwen4exp::QwenSparseAttentionArithmetic arithmetic = qwen4exp::kDefaultQwenSparseAttentionArithmetic; ///< Resolved immutable program; copied to native args and capture identity.
        std::optional<ForwardExecutionPhase> executionPhase; ///< Original graph phase for cold admission/counters only; never forwarded as a native arithmetic switch.
        int retainedMtpCandidateRows = 0; ///< Existing canonical retained R; proves33 Decode capacity without a new admission ledger.
    };
    /** @brief Publish only the accepted raw incomplete pool; frontier is published separately. */
    struct CommitPartial
    {
        Qwen4ExpFloatBinding priorPartial, rawKey;
        Qwen4ExpIntInput acceptedRows;
        Qwen4ExpFloatBinding outputPartial;
    };
    /** @brief Retain intentional verifier candidates before later layers reuse shared raw projection storage. */
    struct RetainCandidates
    {
        Qwen4ExpFloatBinding rawKey,retainedRawKey;
        Qwen4ExpIntBinding copiedBytes; ///< One aligned UINT64 backed by two native INT32 arena words.
        Qwen4ExpFloatBinding priorPartial; ///< Accepted raw tail destination; original source lives in transactionPrior.
        int requestIndex, requestCount; ///< Exact request slot in the canonical publication vector.
        RequestRowOrigins verifierOrigins; ///< Padded GPU or native compact CPU first-row authority.
        QwenSparseTransactionPriorBinding transactionPrior; ///< Mandatory immutable original state for every acceptance/republication.
        std::optional<QwenSparseCounterBinding> terminalCounters; ///< Exact device-owned original request row.
    };
    /** @brief Closed semantic operation set, without optional required operands. */
    using Operation = std::variant<PrepareReplay, SnapshotPrior, Indexer, SelectedAttention, CommitPartial, RetainCandidates>;
    /** @brief Complete capture identity; all live cardinalities remain in replay bytes. */
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        int rowCapacity, tokenCapacity;
        Qwen4ExpIntBinding replay; ///< Native QwenSparseReplay, three INT32 words.
        Operation operation;
    };
    static_assert(StageParamsRequired<Params>);
    /** @brief Validate byte extents, ownership, aliases and backend policies before capture.
     * @param params Immutable participant-local operation and prepared storage bindings.
     */
    explicit QwenSparseAttentionStage(Params params);
    /** @brief Submit one operation using prepared authority and exact executor stream.
     * @param context Context matching the sealed participant device.
     * @return True once CPU work or GPU submission succeeds.
     */
    bool execute(IDeviceContext *context) override;
    /** @brief Semantic graph-stage operation tag. */
    ComputeStageType type() const override;
    /** @brief Authenticate every indexer selector/geometry/binding and retained publication identity. */
    uint64_t graphCaptureVariantSignature() const override;
    /** @brief Arithmetic bound using admitted rows and actual selected-cap limit. */
    std::size_t estimatedFlops() const override;
    /** @brief Every operation consumes its execution-owner live extent directly. */
    bool supportsPaddedPrefillRealLengthContract() const override { return true; }
    /** @brief Native declared operand extents, excluding unused tensor capacity. */
    std::size_t estimatedMemoryBytes() const override;
    /** @brief Whether placement matches the constructed backend bundle.
     * @param backend Backend selected by graph placement.
     */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @brief No stage-owned allocations; all scratch is explicit admitted arena storage. */
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    /** @brief Mandatory coherent input/output contracts and immutable model weights. */
    StageBufferContract bufferContract() const override;
    /** @brief Exact native views and immutable geometry for stage diagnostics. */
    StageDumpInfo buildDumpInfoImpl() const override;
    /**
     * @brief Accepted-tail state may remain zero when complete four-token pools leave no tail.
     *
     * This permits zero only for the typed state-publication operation. It
     * retains finite/storage checks and never inspects device counts on host.
     */
    bool allowsZeroOutput() const override;
    /** @brief Retained raw projections are the exact state evidence required by verifier publication. */
    bool hasVerifierStateCapture() const override;
    /** @brief Omitting a retained QSA tail publication would corrupt the next incomplete pool. */
    bool requiresVerifierStateCaptureForPublication() const override { return hasVerifierStateCapture(); }
    /** @brief Validate CPU row and exact tail write interval without changing accepted state. */
    CPUVerifierStateRestorePlan planCPUVerifierStateRestoreRow(int row) override;
    /** @brief Plan this retained request's exact accepted tail from the complete CPU decision vector. */
    bool planCPUVerifierStateRestoreRows(const int *rows,int requests,
        std::vector<CPUVerifierStateRestorePlan> &plans) override;
    /** @brief Commit one host-owned accepted flat row after complete publication validation. */
    bool restoreVerifierStateCaptureRow(int row,void *stream) override;
    /** @brief Consume the complete CPU publication vector with an explicit request-local slot. */
    bool restoreVerifierStateCaptureRows(const int *rows,int requestCount,void *stream) override;
    /** @brief Publish one GPU-owned accepted flat row without readback or conversion storage. */
    bool restoreVerifierStateCaptureRowFromDeviceIndex(const int *row,void *stream) override;
    /** @brief Accepted QSA publication writes only the persistent raw partial-key bank. */
    std::span<const BufferBinding> verifierStatePublicationOutputs() const noexcept override;

    /** @brief Publish this request's retained state from the canonical independently-strided vector. */
    bool restoreVerifierStateCaptureRowsFromDeviceIndices(const int *rows,int requestCount,int rowIndexStride,void *stream) override;
    /** @brief Inspect sealed operation identity without reading live state. */
    const Params &getParams() const noexcept { return params_; }
private:
    enum class Access { Input, Output }; ///< Declared executor preparation/publication policy.
    /** @brief Setup-created native byte view; shared by validation, contracts and diagnostics. */
    struct Operand
    {
        TensorBase *tensor;
        BufferId buffer;
        TensorType dtype;
        std::size_t offset, bytes;
        Access access;
        const char *name;
        bool exactAlias;
    };
    /** @brief Register and validate a native arena byte view before execution. */
    template<TensorType Type> void add(Qwen4ExpArenaBinding<Type> binding, std::size_t bytes,
        Access access, const char *name, bool exactAlias = false);
    /** @brief Retain one exact arena or generation-session INT32 input. */
    void addIntInput(const Qwen4ExpIntInput &binding,std::size_t count,const char *name);
    /** @brief Resolve an authoritative INT32 input without transfers or host mirrors. */
    const std::int32_t *intInput(const Qwen4ExpIntInput &binding) const;
    /** @brief Validate and retain a native immutable FP32 weight extent. */
    void addWeight(Qwen4ExpImmutableWeightBinding binding, std::size_t elements, const char *name);
    /** @brief Reject overlapping output and immutable inputs, including weight storage. */
    void validateAliases() const;
    /** @brief Resolve prepared authoritative bytes; never initiate transfer or allocation. */
    template<typename T, TensorType Type> T *pointer(Qwen4ExpArenaBinding<Type> binding, Access access) const;
    /** @brief Resolve an immutable model-owned FP32 pointer already prepared by executor. */
    const float *weightPointer(Qwen4ExpImmutableWeightBinding binding) const;
    /** @brief Frozen model input independently published through raw-weight contracts. */
    struct Weight { const TensorBase *tensor; std::size_t elements; const char *name; };
    /** @brief Declare one exact packed prior with mandatory role, alignment and original replay identity. */
    void addTransactionPrior(QwenSparseTransactionPriorBinding binding,Access access);
    /** @brief Resolve the typed packed codec on the stage's sole execution authority. */
    const qwen4exp::QwenSparseTransactionPrior* transactionPrior(QwenSparseTransactionPriorBinding binding) const;
    /** @brief Resolve the sealed request's first-row policy without reading GPU words on host. */
    RequestRowOrigins retainedOrigins() const;
    /** @brief Enqueue one validated native tail splice and publish its exact output event. */
    bool publishRetained(const int *flatRow,void *stream);
    std::array<BufferBinding, 1> acceptedPublicationOutputs_{}; ///< Immutable cold accepted-state declaration.
    std::array<CPUVerifierStateRestoreDestination,1> cpuPublicationWrites_{}; ///< CPU-only immutable planning span.
    Params params_;
    qwen4exp::QwenSparsePowerFrequencies indexerFrequencies_{}; ///< Setup-baked exact HF power invariants; theta/dimension define capture identity.
    float keyNegativeLogTheta_=0.0f; ///< Ordinary core RoPE's exact setup scalar, separate from indexer phases.
    std::vector<Operand> operands_;
    std::vector<Weight> weights_;
    std::vector<Qwen4ExpNativeIntInput> nativeInputs_;
    std::unique_ptr<qwen4exp::IQwenSparseAttentionKernels> kernels_;
};
} // namespace llaminar2
