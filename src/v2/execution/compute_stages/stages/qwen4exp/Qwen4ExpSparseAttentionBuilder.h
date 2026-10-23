/**
 * @file Qwen4ExpSparseAttentionBuilder.h
 * @brief Declarative split-indexer projections and native selected attention wiring.
 *
 * The production GGUF exports independent replicated [512,H] query and [128,H]
 * raw-key matrices. This helper uses their exact prepared identities and the
 * ordinary layer's already-prepared core query/native KV publication. It never
 * constructs a dense attention mask, gathered KV bank or hidden collective.
 */
#pragma once
#include "QwenSparseAttentionStage.h"
#include "Qwen4ExpHyperconnectionBuilder.h"
#include "../AttentionConsumerBinding.h"
#include <cstdint>

namespace llaminar2
{
class ComputeGraph;
/** @brief Canonical invocation phase, defined by IGraphBuilder. */
enum class ForwardExecutionPhase : std::uint8_t;
/** @brief Explicit acceptance lifetime; ordinary rows commit now, verifier rows await the controller. */
enum class Qwen4ExpSparsePublication { AcceptAllLiveRows, DeferredVerifierAcceptance };

/** @brief Immutable sparse arithmetic/geometry, separate from live replay state. */
struct Qwen4ExpSparseAttentionPolicy
{
    int hidden, tokenCapacity, blockCapacity, positionCapacity;
    int rotaryDimension = 64;
    float epsilon = 1.0e-6f;
    qwen4exp::QwenSparseScalar representation = qwen4exp::QwenSparseScalar::FP32;
    Qwen4ExpProjectionArithmetic projectionArithmetic = Qwen4ExpProjectionArithmetic::Ordinary;
    std::optional<DeviceRowRange> projectionRows;
    const IMPIContext *mpi = nullptr;
    qwen4exp::QwenSparseRotarySource rotarySource = qwen4exp::QwenSparseRotarySource::TablePlanes;
    float ropeTheta = 10000.0f;
    Qwen4ExpSparsePublication publication = Qwen4ExpSparsePublication::AcceptAllLiveRows;
    qwen4exp::QwenSparsePoolSelection poolSelection = qwen4exp::QwenSparsePoolSelection::Installed; ///< Sealed native selector, independent of live prefix and row counts.
    qwen4exp::QwenSparseAttentionArithmetic attentionArithmetic = qwen4exp::kDefaultQwenSparseAttentionArithmetic; ///< Explicit selected-core arithmetic; no runtime automatic selection.
};
/** @brief Replicated production GGUF projection identities and immutable indexer gains/planes. */
struct Qwen4ExpSparseAttentionWeights
{
    Qwen4ExpPreparedProjection query, rawKey;
    Qwen4ExpImmutableWeightBinding queryNorm, keyNorm;
    std::optional<Qwen4ExpImmutableWeightBinding> cosine, sine;
};
/** @brief Intentional retained transaction bank and canonical verifier coordinates. */
struct Qwen4ExpSparseAttentionRetention
{
    Qwen4ExpFloatBinding rawKey;
    Qwen4ExpIntBinding copiedBytes;
    RequestRowOrigins verifierOrigins;
    QwenSparseTransactionPriorBinding transactionPrior; ///< Mandatory immutable original state; never an accepted-bank alias.
    std::optional<QwenSparseCounterBinding> terminalCounters; ///< Borrow of the original device-generation request row.
};
/** @brief Admitted indexer/native reader views; live state has one execution owner. */
struct Qwen4ExpSparseAttentionBuffers
{
    Qwen4ExpIntBinding replay;
    Qwen4ExpFloatBinding query, rawKey, priorPartial, pooledKeys, preparedQuery, scores;
    Qwen4ExpIntBinding selected, selectedCount;
    std::optional<Qwen4ExpIntBinding> rowParameters, gpuPartials, observations;
    std::string replayReady; ///< Exact execution-owner replay producer, also orders projection live counts.
    std::string rowParametersReady; ///< Required GPU producer of canonical per-query ring/horizon records.
    std::optional<Qwen4ExpSparseAttentionRetention> retained; ///< Required only for deferred verifier publication.
};
/** @brief One canonical cache request and its explicitly offset arena/state views. */
struct Qwen4ExpSparseAttentionRequest
{
    int sequence;
    Qwen4ExpSparseAttentionBuffers buffers;
};
/** @brief Flat reusable sparse graph composition; all ownership is caller supplied. */
class Qwen4ExpSparseAttentionBuilder final
{
public:
    /**
     * @brief Declare the native pool selector for one typed production invocation.
     * @param phase Original forward phase; a one-row prefill remains Prefill.
     * @param device Exact participant/backend owning this graph, without a live probe.
     * @param rowCapacity Positive immutable captured query capacity, not a live-row count.
     * @return Bounded1024_1024 for ROCm Decode with capacity 1 through 16;
     * Installed for every other valid phase/backend/positive capacity combination.
     * @throws std::invalid_argument For an unknown phase/backend, invalid participant
     * ordinal, or nonpositive capacity, before any graph nodes are constructed.
     * @details The declaration depends only on typed graph inputs. Native selection
     * still reads device-owned causal context and directly publishes every visible
     * ID when at most 512 pools exist; no context readback, host mirror, new kernel
     * or context-switch graph is needed. The returned immutable enum already flows
     * through Indexer admission and complete capture identity.
     */
    static qwen4exp::QwenSparsePoolSelection poolSelectionForInvocation(
        ForwardExecutionPhase phase,DeviceId device,int rowCapacity);
    /** @brief Seal the canonical single-request verifier projection prefix, including M1.
     * @param device Exact participant owning the borrowed count word.
     * @param rows Admitted row capacity and optional native execution-authority count.
     * @return Native GPU prefix, or absent for explicit full work / actual-shape CPU graphs.
     * @throws std::invalid_argument On a foreign/arena count or padded CPU invocation.
     * @details This helper never reads GPU words, changes arithmetic policy, allocates,
     * or drops an explicit M1 count. Ordinary prefill must select its own row policy.
     */
    static std::optional<DeviceRowRange> verifierProjectionRows(DeviceId device,Qwen4ExpStageRows rows);
    /** @brief Append two prepared indexer GEMMs, causal selection and native core attention.
     * @param graph Existing participant-local graph; all dependencies must already exist.
     * @param consumer Ordinary prepared-Q/native-cache handoff before any dense source selection.
     * @param policy Immutable projection/indexer arithmetic and capture geometry.
     * @param weights Frozen prepared split GGUF matrices, actual gains and full rotary planes.
     * @param buffers Admitted scratch/persistent cache and execution-owner metadata views.
     * @return Terminal native consumer/publication node, preserving the existing explicit sigmoid-gate/Wo chain.
     * @throws std::invalid_argument On invalid ownership, geometry, aliases, names or dependencies.
     */
    static std::string append(ComputeGraph &graph,const AttentionConsumerBinding &consumer,
        const Qwen4ExpSparseAttentionPolicy &policy,const Qwen4ExpSparseAttentionWeights &weights,
        const Qwen4ExpSparseAttentionBuffers &buffers);
    /** @brief Append two shared flat projections and independent native request-local consumers.
     * @param rows Sole canonical physical request geometry; inactive holes remain request-local.
     * @param requests Exact cache slots and per-request arena views, in canonical matrix order.
     * @return Actual independent terminal dependencies for the next ordinary consumer.
     * @details Projection outputs are shared [R*M,512/128] tensors; all native QSA operations
     * consume only their replay-authored live rows. Deferred publication retains only M-by128
     * raw projections, never native KV or completed pool banks.
     */
    static std::vector<std::string> appendRequests(ComputeGraph &graph,const AttentionConsumerBinding &consumer,
        const Qwen4ExpSparseAttentionPolicy &policy,const Qwen4ExpSparseAttentionWeights &weights,
        DeviceRequestRowRanges rows,const std::vector<Qwen4ExpSparseAttentionRequest> &requests);
};
} // namespace llaminar2
