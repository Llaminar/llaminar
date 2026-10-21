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
#include "Qwen4ExpPrimitiveStage.h"
#include "../../../../loaders/WeightPlan.h"
#include "../../../../kernels/common/DeviceRowRange.h"
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
class ComputeGraph;
class PreparedWeightStore;

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
    QuantizedActivationOperandPrecision upOperandPrecision=QuantizedActivationOperandPrecision::SingleQ8; ///< Explicit HC-up arithmetic; other projections retain SingleQ8.
    std::optional<DeviceRowRange> upProjectionRows; ///< Explicit TwoTerm up-only original query authority; down/inject retain their existing dispatch.
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
