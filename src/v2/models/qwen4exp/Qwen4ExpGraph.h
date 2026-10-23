/**
 * @file Qwen4ExpGraph.h
 * @brief Declarative Flash Next composition over shared GDN/MoE and native sparse stages.
 *
 * The model declares exact weights, placement and admitted arena views. Reusable
 * builders own GR/PLE/QSA node wiring; the canonical cache owns auxiliary state
 * and the immutable-storage boundary owns CPU table I/O. No backend-specific
 * arithmetic, communication or model-state mirror lives in this adapter.
 */
#pragma once
#include "../qwen35moe/Qwen35MoEGraph.h"
#include "../../execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "../../execution/qwen4exp/Qwen4ExpAuxiliaryState.h"
#include "../../kernels/qwen4exp/QwenSparseAttentionArithmetic.h"

namespace llaminar2
{
struct Qwen4ExpPLEContext;
/** @brief Participant-local graph for the exact qwen4exp GGUF architecture. */
class Qwen4ExpGraph final : public Qwen35MoEGraph
{
public:
    /** @brief Bind authenticated metadata; model/arena/authority injection follows the shared factory lifecycle. */
    explicit Qwen4ExpGraph(const GraphConfig &config,std::shared_ptr<IMPIContext> mpi=nullptr);
    /**
     * @brief Compose the authenticated single-request Dense0/PerBranch predictor over canonical native helpers.
     * @param depth Learned depth0, reused recursively by the existing controller.
     * @param weights Exact frozen sidecar native roles, borrowing unchanged target globals.
     * @param input Explicit draft or shifted-prefill invocation with its original row authority.
     * @param output Existing admitted roles; Draft writes recursive4H MTP_HIDDEN, while shifted prefill owns only input-mix/KV outputs.
     * @return One participant-local transaction; no QSA/PLE predictor state or second ownership framework.
     * @throws std::exception For missing manifest, non-R1 input, inconsistent source/count/layout or arena roles.
     */
    ComputeGraph buildMTPGraph(int depth,const MTPDepthWeightBindings &weights,
        const MTPForwardInput &input,MTPForwardOutput &output) override;
    /** @brief Reject legacy unsealed predictor bindings; the shared artifact requires its frozen source manifest. */
    ComputeGraph buildMTPGraph(int depth,const MTPDepthWeights &weights,
        const MTPForwardInput &input,MTPForwardOutput &output);
    std::string architectureName() const override { return "qwen4exp"; }
    /**
     * @brief Retain shared MoE provenance and authenticate this graph's frozen attention base.
     * @param material Prefix identity being assembled before any cached payload is restored.
     * @throws std::invalid_argument For an invalid base snapshot or duplicate identity authority.
     * @note Uses the same construction-time snapshot as sparse stage wiring; setup reload cannot relabel it.
     */
    void appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const override;
    /** @brief Extend the actual common schema using the single typed QFN buffer inventory. */
    GraphSchema getSchema() const override;
    /** @brief Resolve exact branch, lexical and sparse geometry over the admitted common envelope. */
    GraphResolverConfig getResolverConfig(int seq_len) const override;
    /**
     * @brief Bind declared QFN arena roles before source materialization or cache setup.
     * @param arena Declaration arena, or null when detaching its borrowed views.
     * @note No table boundary is acquired here. Actual PLE graph construction
     * requires the exact prepared model placement and binds its external roots
     * through retainedStorageBoundary before constructing native lookup stages.
     */
    void setArena(BufferArena *arena) override;
    /** @copydoc IGraphBuilder::bindStorageMemoryAuthority */
    void bindStorageMemoryAuthority(std::shared_ptr<PhysicalMemoryAuthority> authority) override;
    /** @brief Exact model-owned auxiliary content in this participant's declared PP interval. */
    enum class AuxiliaryContent { Absent, PLEOnly, SparseOnly, PLEAndSparse };
    /**
     * @brief Project auxiliary ownership from authenticated mixer/PLE metadata.
     * @return Truthful empty or actual local QSA/PLE ownership; GDN remains cache-owned.
     * @throws std::invalid_argument For an invalid/out-of-model PP interval.
     */
    AuxiliaryContent auxiliaryContent() const;
    /** @copydoc IGraphBuilder::initializeCacheAuxiliaryState */
    void initializeCacheAuxiliaryState(IKVCache &cache,void *publicationStream) override;
    /** @brief Preserve resident prefill errors in the canonical request controller until ordered reset. */
    sampling_math::DeviceGenerationForwardFailurePolicy deviceGenerationForwardFailurePolicy() const override;
    /** @copydoc IGraphBuilder::retainedStorageBoundary */
    std::shared_ptr<IRetainedGraphStorageBoundary> retainedStorageBoundary(DeviceId device) override;
protected:
    /** @copydoc QwenGraphBase::addModelLayerInput */
    std::string addModelLayerInput(ComputeGraph &graph,const ForwardInput &input,
        int layer,DeviceId device,const std::string &dependency) override;
    /** @copydoc QwenGraphBase::addModelAttentionPreparation */
    std::string addModelAttentionPreparation(ComputeGraph &graph,const AttentionConsumerBinding &binding) override;
    /** @copydoc QwenGraphBase::addModelAttentionConsumer */
    std::optional<std::string> addModelAttentionConsumer(ComputeGraph &graph,const AttentionConsumerBinding &binding) override;
private:
  const qwen4exp::QwenSparseAttentionArithmetic sparse_attention_arithmetic_; ///< Existing explicit base arithmetic snapshot.
  const qwen4exp::QwenSparseAttentionPhaseSchedule sparse_attention_phase_schedule_; ///< Canonical immutable setup scalar shared with arena planning; no DebugEnv reread.
  /**
   * @brief Project only the existing helper-authenticated original query count's arena role.
   * @param input Complete actual main/verifier invocation, including typed prefill/request producers.
   * @param rows Exact singleRequestProjectionQueryRows result, never a sliced/repeated substitute.
   * @return Original request/chunk/verifier subview, or absence for fully-active ordinary M1.
   * @throws std::exception For a foreign role/root, unsupported producer or changed original pointer.
   * @note Workspace declarations use their permanent request/verifier roots but remain non-executable.
   */
  std::optional<Qwen4ExpIntBinding> adaptiveCountBinding(const ForwardInput &input,
                                                         const DeviceRowRange &rows) const;
  friend struct Qwen4ExpPLEProjectionTestPeer;
  /**
   * @brief Bind PLE arithmetic and query rows from the actual complete invocation.
   * @param input Exact input retained by the current forward construction scope.
   * @param device Participant owning tokens, history, count and prepared projections.
   * @param prefix Existing semantic PLE node prefix.
   * @return Original source/hash geometry with explicit grouped-verifier policy and query authority.
   * @throws std::invalid_argument Before graph mutation for foreign or noncanonical grouped queries.
   * @details Ordinary arithmetic keeps its existing uncounted projection policy. The grouped
   * count is borrowed from singleRequestProjectionQueryRows, including counted physical M1.
   */
  Qwen4ExpPLEContext pleContext(const ForwardInput &input, DeviceId device, const std::string &prefix) const;
  /** @brief Require one registered arena role without allocating or rebinding storage. */
  TensorBase &tensor(BufferId role) const;
  /** @brief Borrow an exact native FP32 subview from that role. */
  Qwen4ExpFloatBinding fp32(BufferId role, std::size_t byteOffset = 0) const;
  /** @brief Borrow an exact native integer subview from that role. */
  Qwen4ExpIntBinding integer(BufferId role, std::size_t byteOffset = 0) const;
  /** @brief Attach initialized model state once to its canonical cache, retaining admitted arena ownership.
   */
  Qwen4ExpAuxiliaryState &state(IKVCache &cache, void *publicationStream);
  /** @brief Require the real immutable prepared matrix identity for an extension projection. */
  Qwen4ExpPreparedProjection projection(const WeightBinding *weight, DeviceId device) const;
  /** @brief Require the exact imported FP32 gain/convolution source. */
  Qwen4ExpImmutableWeightBinding gain(const WeightBinding *weight) const;
  /** @brief Resolve the optional native per-query metadata bank after all request replay records. */
  std::optional<Qwen4ExpIntBinding> rowParameters(DeviceId device) const;
  /** @brief Recover the declared block capacity from the arena's flattened physical matrix. */
  int poolCapacity() const;
  /** @brief Whether this PP interval actually contains the sole lexical layer. */
  bool ownsPLE() const;
  std::shared_ptr<PhysicalMemoryAuthority>
      memory_; ///< Same admitted rank authority, never an independent ledger.
  std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary>
      lookup_; ///< One exclusive mapped lane for this participant.
};
} // namespace llaminar2
