/**
 * @file Qwen4ExpPLEBuilder.h
 * @brief Declarative native n-gram lookup, similarity gating and lexical convolution.
 *
 * This participant-local builder composes the canonical prepared GEMMs with
 * backend-neutral model primitives. It retains immutable prior histories and
 * exposes candidate normalized rows to the separate accepted-state authority.
 * Native VRAM, RAM or SSD mmap storage is resolved at admission.
 */
#pragma once
#include "ImmutableLookupStage.h"
#include "Qwen4ExpFixedHeadLookupStage.h"
#include "Qwen4ExpHyperconnectionBuilder.h"
#include <variant>

namespace llaminar2
{
/** @brief Complete immutable geometry and arithmetic of one lexical operation. */
struct Qwen4ExpPLEContext
{
    DeviceId device;
    const IMPIContext *mpi=nullptr;
    std::string prefix;
    Qwen4ExpStageRows rows;
    int hidden;
    float epsilon;
    qwen4exp::NgramHashPolicy hash;
    TensorType nativeFormat;
    int nativeRowElements;
    std::size_t nativeRowBytes;
    Qwen4ExpProjectionArithmetic arithmetic=Qwen4ExpProjectionArithmetic::Ordinary;
    /** @brief Original complete token-query descriptor under explicit verifier arithmetic.
     * @details Includes the exact primitive count owner/capacity/origin/repeat. Omission is
     * valid only when primitives are fully active; counted physical M1 keeps its descriptor.
     * Ordinary arithmetic does not accept a counted verifier descriptor.
     */
    std::optional<DeviceRowRange> projectionRows;
};
/** @brief Frozen exact projection/gain/convolution bindings; table storage is separate. */
struct Qwen4ExpPLEWeights
{
    Qwen4ExpPreparedProjection key,value;
    Qwen4ExpImmutableWeightBinding keyNorm,queryNorm,convolutionNorm,convolution;
};
/** @brief All scratch and immutable request history comes from the admitted arena. */
struct Qwen4ExpPLEBuffers
{
    Qwen4ExpIntInput tokens;
    Qwen4ExpIntBinding priorTokens,ids;
    Qwen4ExpFloatBinding embedding,key,value,query,gated,normalized;
    Qwen4ExpFloatBinding priorConvolution,trunk;
};
/** @brief Reusable flat graph wiring with independent accepted-state publication. */
class Qwen4ExpPLEBuilder final
{
public:
    /** @brief Explicit whole-storage versus fixed-range composition; no automatic substitution. */
    using LookupStorage=std::variant<ImmutableLookupStage::Storage,Qwen4ExpFixedHeadLookupStorage>;
    /**
     * @brief Append lexical computation using its explicitly admitted whole-table or fixed-head storage.
     * @param graph Existing participant-local declarative graph.
     * @param context Original query geometry/arithmetic and complete source hash policy.
     * @param weights Original prepared model weights.
     * @param buffers Exact existing PLE views and accepted/prior history.
     * @param storage Checked CPU/hosted/resident whole-table or fixed-head composition; no automatic substitution.
     * @param dependencies Existing original token/trunk/history producers.
     * @return Terminal updated-trunk node, leaving candidate histories to their existing authority.
     * @throws std::exception Before graph mutation for absent producer/cohort, bad buffers or source mismatch.
     */
    static std::string append(ComputeGraph &graph,const Qwen4ExpPLEContext &context,
        const Qwen4ExpPLEWeights &weights,const Qwen4ExpPLEBuffers &buffers,
        LookupStorage storage,const std::vector<std::string> &dependencies);
};
} // namespace llaminar2
