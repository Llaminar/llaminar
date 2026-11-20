/**
 * @file Qwen4ExpHyperconnectionBuilder.h
 * @brief Reusable participant-local declarative hyperconnection graph composition.
 *
 * The builder wires native branch normalization, ordinary prepared projection
 * stages and the GR-specific primitives. It owns no activation, weight or GPU
 * state. A move-only read result preserves the original and normalized branches
 * and the paired injection identity until one explicit block-write composition.
 * Terminal mixers expose a separate read-only entrypoint and cannot produce a
 * write handle. All graph nodes retain one explicit participant device.
 */
#pragma once
#include "execution/moe/GPUExpertIntermediateTiling.h"
#include "Qwen4ExpPrimitiveStage.h"
#include "../../../../loaders/WeightPlan.h"
#include "../../../../kernels/common/DeviceRowRange.h"
#include "../../../../kernels/common/HCDownOutputPublicationArgs.h"
#include "../../../../kernels/common/FullNativeHCFFNOutputDemand.h"
#include <optional>
#include <array>
#include <string>
#include <vector>

namespace llaminar2
{
class ComputeGraph;
class PreparedWeightStore;
class DeviceWorkspaceManager;
class IDeviceContext;
struct WorkspaceRequirements;
namespace rocm { struct FullNativeHCFFNPrefixArgs; }

/** @brief Exact frozen matrix and prepared store identity validated before graph mutation. */
class Qwen4ExpPreparedProjection final
{
public:
    /** @brief Authenticate a model-owned prepared matrix for one participant.
     * @param binding Frozen immutable source with an exact prepared reference.
     * @param store Sole owner of the referenced prepared representation.
     * @param device Exact consuming participant; cross-device refs are rejected.
     * @throws std::invalid_argument On mutable, missing, stale or mismatched identities.
     */
    Qwen4ExpPreparedProjection(const WeightBinding &binding, PreparedWeightStore &store, DeviceId device);
    /** @brief Verify checkpoint-native [output,input] geometry before graph insertion.
     * @param output Required matrix output width.
     * @param input Required matrix reduction width.
     * @throws std::invalid_argument For nonpositive or mismatched source dimensions.
     */
    void requireGeometry(int output, int input) const;
    /** @brief Return the exact frozen source matrix consumed by the canonical GEMM stage. */
    const TensorBase *tensor() const noexcept { return tensor_; }
    /** @brief Return the authenticated prepared-weight authority. */
    PreparedWeightStore &store() const noexcept { return *store_; }
    /** @brief Return the immutable complete prepared identity. */
    PreparedWeightRef ref() const noexcept { return ref_; }
private:
    const TensorBase *tensor_;
    PreparedWeightStore *store_;
    PreparedWeightRef ref_;
};

/** @brief Projection arithmetic is an explicit graph policy, never an automatic switch. */
enum class Qwen4ExpProjectionArithmetic
{
    Ordinary, ///< Existing ordinary prefill/decode projection policy.
    DecodeEquivalentVerifier, ///< Existing grouped verifier serial-byte arithmetic policy.
};
/** @brief Explicit original branch-norm publication policy. */
enum class Qwen4ExpHCBranchNormExecution { Staged, PublishOriginalSingleQ8 };
/** @brief Explicit complete HC read launch policy; defaults preserve original graph. */
enum class Qwen4ExpHCReadExecution {
    Staged, ProducerGridShortK,
    CTAOwnedOriginalTail, ///< Original norm/injection nodes plus ordinary original producer and CTA-owned completion.
    PersistentOriginalPhases ///< Explicit ROCm ordinary-M1 HC read+original injection, one admitted cooperative program.
};
/** @brief Original reducer may author the existing HC publisher without another launch. */
enum class Qwen4ExpHCDownPublication { Staged, ReducerEpilogue };
/** @brief Exact cold resident-grid economy policy; neither value changes arithmetic. */
enum class Qwen4ExpHCPersistentGrid { OnePerProcessor, MaximumResident };
/** @brief Immutable participant, geometry and arithmetic identity for one mixer. */
struct Qwen4ExpHyperconnectionContext
{
    DeviceId device; ///< Exact participant; no nested multi-device graph is created.
    const IMPIContext *mpi = nullptr; ///< Existing projection MPI metadata, without hidden collectives.
    std::string prefix; ///< Unique caller-owned node-name prefix.
    Qwen4ExpStageRows rows; ///< Admitted rows and primitive live-count subview.
    int hidden; ///< Per-branch compute width H.
    int lowRank; ///< Read-gate bottleneck width R.
    float epsilon; ///< Positive checkpoint branch RMS epsilon.
    Qwen4ExpProjectionArithmetic arithmetic; ///< Explicit projection numeric policy.
    std::optional<DeviceRowRange> projectionRows; ///< Existing verifier row contract, when selected.
    std::string readNodeName; ///< Optional exact public terminal name; empty selects prefix+read.
    QuantizedActivationOperandPrecision upOperandPrecision=QuantizedActivationOperandPrecision::SingleQ8; ///< Frozen quantized HC-up policy; floating up sources retain native floating GEMM.
    std::optional<DeviceRowRange> upProjectionRows; ///< Explicit TwoTerm up-only original query authority; down/inject retain their existing dispatch.
    Qwen4ExpHCDownPublication downPublication=Qwen4ExpHCDownPublication::Staged; ///< Explicit original reducer/publisher scope.
    HCDownProducerProgram downProducerProgram=HCDownProducerProgram::Original; ///< Explicit cold-admitted program, only with the declared HC reducer publication.
    Qwen4ExpHCBranchNormExecution branchNormExecution=Qwen4ExpHCBranchNormExecution::Staged; ///< Independent original norm-to-down SingleQ8 publication candidate.
    Qwen4ExpHCReadExecution readExecution=Qwen4ExpHCReadExecution::Staged; ///< GPU ordinary M1 candidate, explicit and source-checked.
    LocalGPUExpertKernelPolicy verifierKernelPolicy=LocalGPUExpertKernelPolicy::Generic; ///< Qualified original TP4 verifier schedule only.
    Qwen4ExpHCPersistentGrid persistentGrid=Qwen4ExpHCPersistentGrid::OnePerProcessor; ///< Immutable occupancy-bounded persistent program grid.
};
/** @brief Read weights, including already-converted distinct norm gains. */
struct Qwen4ExpHyperconnectionReadWeights
{
    Qwen4ExpImmutableWeightBinding norm;
    Qwen4ExpPreparedProjection down, up;
};
/** @brief Block read/write weights keep the paired injection identity together. */
struct Qwen4ExpHyperconnectionBlockWeights
{
    Qwen4ExpHyperconnectionReadWeights read;
    Qwen4ExpPreparedProjection inject;
};
/** @brief Mandatory arena buffers with independent lifetimes and explicit widths. */
struct Qwen4ExpHyperconnectionReadBuffers
{
    Qwen4ExpFloatBinding trunk; ///< [M,4H], retained unchanged until block write/MTP publication.
    Qwen4ExpFloatBinding normalized; ///< [M,4H], retained for the paired injection projection.
    Qwen4ExpFloatBinding down; ///< [M,R] prepared down-projection output.
    Qwen4ExpFloatBinding activatedDown; ///< [M,R], may exactly alias down.
    Qwen4ExpFloatBinding readLogits; ///< [M,4H] prepared up-projection output.
    Qwen4ExpFloatBinding mixed; ///< [M,H] block input or terminal head input.
};
/** @brief Explicit block result and write temporaries. */
struct Qwen4ExpHyperconnectionWriteBuffers
{
    Qwen4ExpFloatBinding block; ///< [M,H] completed attention/MoE output.
    Qwen4ExpFloatBinding injectionLogits; ///< [M,4] raw prepared injection projection.
    Qwen4ExpFloatBinding outputTrunk; ///< [M,4H], may exactly alias the retained original trunk.
};

/** @brief Which original producer owns the trunk consumed by HC normalization. */
enum class Qwen4ExpHCTrunkSource { PreviousWrite, PublishedTrunk };

/** @brief Whether a composed local kernel absorbs the original injection projection. */
enum class Qwen4ExpHCInjectionOwnership { DiscoverInRegion, KeepOriginalStage };

/** @brief Exact original declarations forming one native HC read region. */
struct Qwen4ExpFullNativeHCRegion {
    Qwen4ExpHyperconnectionContext context; ///< Original immutable arithmetic/row declaration.
    std::string previous_write,branch_norm,down,up,read,injection; ///< Exact existing nodes, no graph search.
};
/**
 * @brief Retain the original HC binding authority without admitting a standalone kernel.
 * @details One existing HC owner retains source/count/tensor/workspace leases.
 * Full execution is submitted only by the native resident FFN owner.
 */
class Qwen4ExpHCReadPhaseOwner {
public:
    virtual ~Qwen4ExpHCReadPhaseOwner()=default;
    /** @return Original declaration's public buffer/weight and diagnostic contracts. */
    virtual const IComputeStage& phaseDeclaration() const=0;
    /** @brief Bind the existing admitted workspace. @param workspace Original executor owner. */
    virtual void bindPhaseWorkspace(DeviceWorkspaceManager* workspace)=0;
    /** @brief Prepare original leases/arguments only. @param ctx Exact worker. @param stream Exact cold stream. */
    virtual void preparePhaseBindings(IDeviceContext* ctx,void* stream)=0;
    /** @brief Validate retained storage. @param stream Exact prepared stream. */
    virtual void validatePhaseBindings(void* stream) const=0;
    /** @return Exact original HC named-bank quote in its original merge order. */
    virtual WorkspaceRequirements phaseWorkspaceRequirements() const=0;
    /** @return Complete unbound prefix; the backend supplies its original barrier lease. */
    virtual rocm::FullNativeHCFFNPrefixArgs fullPrefixArguments() const=0;
    /** @return Original external tensor inputs, excluding internally produced trunk/mixed values. */
    virtual std::vector<Qwen4ExpFloatBinding> phaseInputs() const=0;
    /** @return Every original public output, including independent injection. */
    virtual std::vector<Qwen4ExpFloatBinding> phaseOutputs() const=0;
    /** @brief Freeze actual full-region materializations before allocating or retaining storage.
     * @param demand Original graph readers plus requested observations. Standalone HC owns all outputs.
     */
    virtual void sealFullOutputDemand(FullNativeHCFFNOutputDemand demand)=0;
    /** @return Original optional activated/up bindings, independent of the sealed store policy. */
    virtual std::array<Qwen4ExpFloatBinding,2> optionalPhaseOutputs() const=0;
    /** @brief Retire phase borrows after the enclosing native program retires. */
    virtual void invalidatePhaseBindings()=0;
    /** @return Exact original source kind, never inferred from nullable kernel operands. */
    virtual Qwen4ExpHCTrunkSource trunkSource() const noexcept { return Qwen4ExpHCTrunkSource::PreviousWrite; }
};

/** @brief Reusable graph composition with a single-use paired-write capability. */
class Qwen4ExpHyperconnectionBuilder final
{
public:
    /** @brief Read completion carrying immutable pairing and retained activation identities. */
    class BlockRead final
    {
    public:
        BlockRead(const BlockRead &) = delete;
        BlockRead &operator=(const BlockRead &) = delete;
        /** @brief Move the single write capability and invalidate its previous owner. */
        BlockRead(BlockRead &&other) noexcept;
        BlockRead &operator=(BlockRead &&) = delete;
        /** @brief Name of the mixed-input producer consumed by the attention/MoE block. */
        const std::string &terminal() const noexcept { return terminal_; }
    private:
        friend class Qwen4ExpHyperconnectionBuilder;
        /** @brief Seal the pairing after a complete validated read subgraph is installed. */
        BlockRead(ComputeGraph &graph,Qwen4ExpHyperconnectionContext context,
            Qwen4ExpHyperconnectionReadBuffers buffers,Qwen4ExpPreparedProjection inject,std::string terminal);
        ComputeGraph *graph_;
        Qwen4ExpHyperconnectionContext context_;
        Qwen4ExpHyperconnectionReadBuffers buffers_;
        Qwen4ExpPreparedProjection inject_;
        std::string terminal_;
    };
    /** @brief Append the block read chain and return its single-use paired-write capability.
     * @param graph Existing participant-local graph receiving ordinary flat nodes.
     * @param context Complete geometry, count and arithmetic policy.
     * @param weights Frozen read/injection identities for this exact mixer.
     * @param buffers Caller-admitted arena views; no buffers are allocated here.
     * @param dependencies Existing graph producers of trunk and count state.
     * @return Read completion and the exact normalized/original state for one write.
     */
    static BlockRead appendRead(ComputeGraph &graph,const Qwen4ExpHyperconnectionContext &context,
        const Qwen4ExpHyperconnectionBlockWeights &weights,const Qwen4ExpHyperconnectionReadBuffers &buffers,
        const std::vector<std::string> &dependencies);
    /** @brief Append a paired injection projection and branch write, consuming the read capability.
     * @param graph The same graph that owns the read chain.
     * @param read Single-use capability from appendRead; rejected after moving or consumption.
     * @param buffers Completed block output and caller-admitted write storage.
     * @param blockDependencies Existing producer nodes of the completed block output.
     * @return Name of the updated-trunk producer.
     */
    static std::string appendWrite(ComputeGraph &graph,BlockRead &&read,
        const Qwen4ExpHyperconnectionWriteBuffers &buffers,const std::vector<std::string> &blockDependencies);
    /**
     * @brief Lower one explicit already-composed write/next-normalization boundary.
     * @param graph Cold participant-local graph after the two block fragments are merged.
     * @param writeProducer Exact original first block terminal supplied by the layer composer.
     * @return True after one original write and its direct normalization successor become one operation.
     * @throws std::invalid_argument For missing/foreign stages, row ownership, hazards or capture metadata.
     * @details Only the named direct edge is considered. All original write
     * consumers remain connected. No inter-layer/PLE/PP search or execution is
     * performed, and no inferred normalized capability escapes the builder.
     */
    static bool lowerAdjacentWriteNorm(ComputeGraph &graph,const std::string &writeProducer);
    /**
     * @brief Retain exact existing HC declarations for full-region composition only.
     * @param graph Original cold participant-local graph.
     * @param region Explicit write/norm/down/up/read/injection nodes and original context.
     * @return One preparation-only owner; no standalone function is admitted.
     */
    /**
     * @brief Resolve only the named original read's direct typed HC producers.
     * @param graph Cold participant-local graph. @param previous_write Exact previous write terminal.
     * @param read Exact original residual-read terminal.
     * @return Validated original region; injection is absent for an original final read.
     */
    static Qwen4ExpFullNativeHCRegion describeFullNativeReadRegion(
        ComputeGraph& graph,const std::string& previous_write,const std::string& read);
    /**
     * @brief Resolve original direct HC edges with explicit injection ownership.
     * @param graph Original participant graph.
     * @param previous_write Exact prior write name, or empty for a separately published trunk.
     * @param read Exact original residual read.
     * @param injection Policy for the original independent four-logit projection.
     * @return Cold region declarations with all opted-in stage identities.
     */
    static Qwen4ExpFullNativeHCRegion describeFullNativeReadRegion(
        ComputeGraph& graph,const std::string& previous_write,const std::string& read,
        Qwen4ExpHCInjectionOwnership injection);
    /** @return Original preparation-only root preserving the exact source/row/workspace leases. */
    static std::shared_ptr<Qwen4ExpHCReadPhaseOwner> retainFullNativeReadPhase(
        ComputeGraph& graph,const Qwen4ExpFullNativeHCRegion& region);
    /** @brief Retain the original read and its explicit distinct trunk publication authority. */
    static std::shared_ptr<Qwen4ExpHCReadPhaseOwner> retainFullNativeReadPhase(
        ComputeGraph& graph,const Qwen4ExpFullNativeHCRegion& region,Qwen4ExpHCTrunkSource source);

    /** @brief Append a final read-only mixer; no write capability is created.
     * @param graph Existing participant-local graph.
     * @param context Complete final-mixer geometry and arithmetic policy.
     * @param weights Read-only frozen norm/down/up identities.
     * @param buffers Caller-admitted final-mixer activations.
     * @param dependencies Existing producers of the pre-final-mixer trunk/count state.
     * @return Name of the H-wide terminal-head input producer.
     */
    static std::string appendFinalRead(ComputeGraph &graph,const Qwen4ExpHyperconnectionContext &context,
        const Qwen4ExpHyperconnectionReadWeights &weights,const Qwen4ExpHyperconnectionReadBuffers &buffers,
        const std::vector<std::string> &dependencies);
};
} // namespace llaminar2
