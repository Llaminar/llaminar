/**
 * @file Qwen4ExpAuxiliaryState.h
 * @brief Cache-owned borrowed QSA/PLE state and accepted-prefix graph publication.
 *
 * Registry-created views retain one physical arena authority. QSA pools are
 * append-only, raw tails are committed alias-safely, and PLE uses fixed accepted
 * and next banks. Neither replay metadata nor a second frontier is persisted:
 * every transaction regenerates them from the restored canonical cache state.
 */
#pragma once
#include "../../kernels/kvcache/CacheAuxiliaryState.h"
#include "../../kernels/qwen4exp/Qwen4ExpArchiveGeometry.h"
#include "../compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "ImmutableVerifierPrior.h"
#include "QwenSparseCounterBinding.h"
#include "../compute_stages/stages/qwen4exp/Qwen4ExpPLERetainedStage.h"
#include <optional>
#include <vector>

namespace llaminar2
{
class ComputeGraph;
/** @brief Borrowed persistent state attached once to the canonical QFN cache. */
class Qwen4ExpAuxiliaryState final : public ICacheAuxiliaryState
{
public:
    /** @brief Admitted ephemeral verifier banks, never reset or exported as accepted cache state. */
    struct SparseVerifierStorage {
        int rowCapacity; ///< Canonical maximum retained target rows for this R1 transaction.
        Qwen4ExpFloatBinding rawKeys;
        Qwen4ExpIntBinding candidateCopyBytes;
        QwenSparseTransactionPriorBinding prior;
        Qwen4ExpIntBinding priorTraffic;
    };
    /** @brief One exact model-layer/request pool/tail bank and optional MTP candidates. */
    struct SparseLayer
    {
        int layer, cacheLayer, sequence, contextTokens;
        Qwen4ExpFloatBinding pooledKeys, partialKeys;
        std::optional<SparseVerifierStorage> verifier; ///< Dedicated original state/candidates through terminal retirement.
    };
    /** @brief Dedicated intentional verifier evidence, excluded from ordinary reset/archive state. */
    struct PLEVerifierStorage
    {
        int rowCapacity;
        Qwen4ExpIntBinding priorTokens, transactionLive, retainedTokens, copiedBytes;
        Qwen4ExpFloatBinding priorConvolution, retainedNormalized;
    };
    /** @brief Fixed accepted/next PLE banks for one request and the actual owning model layer. */
    struct PLEHistory
    {
        int layer, cacheLayer, sequence, channels;
        Qwen4ExpIntBinding acceptedTokens, nextTokens;
        Qwen4ExpFloatBinding acceptedConvolution, nextConvolution;
        std::optional<PLEVerifierStorage> verifier; ///< Dedicated backing, never inferred from spare next-bank capacity.
    };
    /** @brief Existing declared arena owners; no allocation is performed by this state extension. */
    struct Params
    {
        IKVCache &cache; ///< Canonical sequence frontier owner; this extension is attached to that same instance.
        DeviceId device;
        int requestCapacity, eosToken;
        std::vector<SparseLayer> sparseLayers;
        std::vector<PLEHistory> pleHistories;
    };
    /** @brief Validate all native subviews and seal cache/request ownership before graph capture. */
    explicit Qwen4ExpAuxiliaryState(Params params);
    /** @brief Exact sparse state for a declared global layer and admitted request. */
    const SparseLayer &sparse(int globalLayer,int sequence) const;
    /** @brief Exact fixed PLE accepted/next banks for one admitted request. */
    const PLEHistory &ple(int sequence) const;
    /** @brief Return the cache's retained and fully immutable binding manifest. */
    const Params &bindings() const noexcept { return params_; }
    /** @return Immutable per-request archive geometry, independent of live state and rollback capacity. */
    qwen4exp::Qwen4ExpArchiveGeometry archiveGeometry(int sequence) const;
    /** @brief An invocation-owned retained verifier candidate, separate from persistent request state. */
    struct SparseCandidates
    {
        int layer, sequence, rowCapacity;
        Qwen4ExpFloatBinding rawKeys;
    };
    /** @brief Install the sole original-state snapshot and mint a same-graph immutable read/publication capability.
     * @param graph Exact participant-local verifier graph.
     * @param prefix Unique layer prefix; the actual snapshot producer is returned by the lease.
     * @param globalLayer Exact main sparse layer identity.
     * @param sequence Canonical cache request slot; current model composition supports slot0/R1.
     * @param rowCapacity This invocation's physical verifier rows, bounded by admitted retained storage.
     * @param replay Canonical preappend replay producer output.
     * @param dependencies Existing exact replay/input producer edges.
     * @param terminalCounters Optional original device controller borrow; no additional coherence owner.
     */
    ImmutableVerifierPrior appendSparseVerifierPrior(ComputeGraph& graph,const std::string& prefix,
        int globalLayer,int sequence,int rowCapacity,Qwen4ExpIntBinding replay,
        const std::vector<std::string>& dependencies,
        std::optional<QwenSparseCounterBinding> terminalCounters=std::nullopt) const;
    /** @brief Commit retained verifier candidates using the device-owned accepted prefix.
     * @param graph Existing graph containing the acceptance producer.
     * @param prefix Unique publication node prefix.
     * @param sequence Exact request slot being accepted.
     * @param replay Original immutable transaction snapshot, never inferred from postappend counts.
     * @param acceptedRows Exact generation-controller decision on this state device.
     * @param dependencies Explicit acceptance and candidate production edges.
     * @return Terminal publication node; every affected state bank is ordered before it.
     */
    std::string appendAcceptedSparseCommit(ComputeGraph &graph,const std::string &prefix,int sequence,
        const std::vector<SparseCandidates> &candidates,Qwen4ExpIntBinding replay,
        Qwen4ExpIntInput acceptedRows,const std::vector<std::string> &dependencies) const;
    /** @brief Read-only prior handles plus the single-use snapshot composition capability. */
    class PLEVerifierPrior final
    {
    public:
        PLEVerifierPrior(const PLEVerifierPrior&)=delete;
        PLEVerifierPrior&operator=(const PLEVerifierPrior&)=delete;
        PLEVerifierPrior(PLEVerifierPrior&&)=default;
        PLEVerifierPrior&operator=(PLEVerifierPrior&&)=delete;
        /** @brief Native original token suffix supplied to the verifier hash builder. */
        Qwen4ExpIntBinding tokens()const{(void)capability_.snapshotProducer();return tokens_;}
        /** @brief Native original normalized suffix supplied to verifier convolution. */
        Qwen4ExpFloatBinding convolution()const{(void)capability_.snapshotProducer();return convolution_;}
        /** @brief Exact snapshot producer consumed by both verifier readers. */
        const std::string&snapshotProducer()const{return capability_.snapshotProducer();}
    private:
        friend class Qwen4ExpAuxiliaryState;
        /** @brief Construct only after the auxiliary owner appends the dedicated snapshot. */
        PLEVerifierPrior(int sequence,Qwen4ExpStageRows rows,Qwen4ExpIntBinding tokens,
            Qwen4ExpFloatBinding convolution,ImmutableVerifierPrior capability)
            :sequence_(sequence),rows_(std::move(rows)),tokens_(tokens),convolution_(convolution),capability_(std::move(capability)){}
        int sequence_;Qwen4ExpStageRows rows_;Qwen4ExpIntBinding tokens_;Qwen4ExpFloatBinding convolution_;
        ImmutableVerifierPrior capability_;
    };
    /**
     * @brief Snapshot original tails before the first verifier hash/convolution consumer.
     * @param graph Participant-local model fragment receiving the actual snapshot producer.
     * @param prefix Unique semantic prefix for the transaction's producer.
     * @param sequence Admitted single-request history index.
     * @param rows Original verifier capacity and native live-row owner.
     * @param candidateTokens Original candidate token rows for this transaction.
     * @param candidateNormalized Admitted normalized candidate bank retained by the transaction.
     * @param dependencies Exact local producers; empty only when this is the fragment's first node.
     * @return Move-only prior capability consumed by retained candidate publication.
     * @throws std::invalid_argument For missing, foreign or omitted non-entry producers.
     * @note PipelineForwardGraphEdges attaches native metadata and activation
     * dependencies to all model roots after construction. A pipeline entry
     * therefore remains a real snapshot root, without a dummy readiness stage.
     */
    PLEVerifierPrior appendVerifierPLEPrior(ComputeGraph&graph,const std::string&prefix,int sequence,
        Qwen4ExpStageRows rows,Qwen4ExpIntInput candidateTokens,Qwen4ExpFloatBinding candidateNormalized,
        const std::vector<std::string>&dependencies)const;
    /** @brief Retain exact candidates, consume the prior capability and join canonical publication hooks. */
    std::string appendVerifierPLERetain(ComputeGraph&graph,const std::string&prefix,PLEVerifierPrior&&prior,
        Qwen4ExpIntInput candidateTokens,Qwen4ExpFloatBinding candidateNormalized,
        const std::vector<std::string>&dependencies)const;
    /** @brief Commit accepted lexical histories, then promote fixed next banks to their accepted banks. */
    std::string appendAcceptedPLECommit(ComputeGraph &graph,const std::string &prefix,int sequence,
        Qwen4ExpStageRows rows,Qwen4ExpStageAcceptance acceptance,Qwen4ExpIntInput candidateTokens,
        Qwen4ExpFloatBinding candidateNormalized,const std::vector<std::string> &dependencies) const;
    const IKVCache &cache() const noexcept override { return params_.cache; }
    void initialize(void *exactStream) override;
    void requireReady(void *exactStream) const override;
    CacheAuxiliaryStateKind kind() const noexcept override { return CacheAuxiliaryStateKind::QwenFlashNext; }
    DeviceId device() const noexcept override { return params_.device; }
    std::size_t checkpointCapacity(int sequence) const override;
    std::size_t blockArchiveBytes(CacheAuxiliaryBlockRange range) const override;
    void reset(CacheAuxiliaryResetRange range,void *exactStream) override;
    void captureCheckpoint(int sequence,CacheAuxiliaryMemory destination,void *exactStream) const override;
    void restoreCheckpoint(int sequence,CacheAuxiliaryMemory source,void *exactStream) override;
    std::size_t endpointArchiveBytes(int sequence,int logicalTokens) const override;
    void exportEndpoint(int sequence,int logicalTokens,CacheAuxiliaryMemory destination,void *exactStream) const override;
    void importEndpoint(int sequence,int logicalTokens,CacheAuxiliaryMemory source,void *exactStream) override;
    void exportBlock(CacheAuxiliaryBlockRange range,CacheAuxiliaryMemory destination,void *exactStream) const override;
    void importBlock(CacheAuxiliaryBlockRange range,CacheAuxiliaryMemory source,void *exactStream) override;
private:
    /** @brief Resolve static native pointers without changing placement or allocating storage. */
    template<typename T,TensorType Type> T *pointer(Qwen4ExpArenaBinding<Type> binding) const;
    /** @brief Copy exact immutable archive extents across the intentional prefix-cache I/O boundary. */
    void archiveCopy(void *destination,const void *source,std::size_t bytes,CacheAuxiliaryMemoryDomain domain,
        bool exportData,void *stream) const;
    /** @brief Publish exact arena owner events after an infrastructure-owned state mutation. */
    void publish(void *stream) const;
    Params params_;
    std::vector<TensorBase *> owners_; ///< Setup-sealed unique tensor event authorities, never rebuilt during execution.
    bool initialized_ = false; ///< Setup lifecycle only; no mutable GPU execution state is mirrored.

};
} // namespace llaminar2
