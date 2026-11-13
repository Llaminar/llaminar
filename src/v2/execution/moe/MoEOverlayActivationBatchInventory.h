/**
 * @file MoEOverlayActivationBatchInventory.h
 * @brief Immutable source/row declarations shared by mapped Batch BOM and graph construction.
 *
 * Entries name retained model graph owners, rather than channel capacity or
 * layer-count approximations. This metadata owns no allocation or live state.
 */
#pragma once
#include "execution/config/RuntimeConfig.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "kernels/common/SparseQueryExecutionProgram.h"
#include "execution/mtp/MTPSidecarCaptureLayout.h"
#include "execution/mtp/OrdinaryGenerationGraphPlan.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>
namespace llaminar2 {
struct ForwardInput;
class LayerGraphSourceIdentity;
/** @brief Complete admitted continuation/predictor Batch source roster. */
class MoEOverlayActivationBatchInventory final {
public:
    /** @brief Distinct graph owners, including named variable prefill cache reserves. */
    enum class SourceKind { PrefillBucket, RequestPrefill, PrefixRehydration, Decode,
        RestoredPrefixBridge, SpeculativeCondition, CommittedCondition,
        VerifierGreedy, VerifierStochastic, ForwardOnly, Draft };
    /** @brief Original immutable token input source of a learned sidecar cache. */
    enum class DraftTokenSource { Host, Device };
    /** @brief One actual graph source, shared across its original ordered routed layers. */
    struct Source {
        SourceKind kind;
        std::size_t family_ordinal;
        SparseQueryExecutionProgram query_program;
        int physical_rows; ///< Exact retained source width; zero denotes a named variable cache slot.
        int maximum_rows; ///< Positive memory envelope, never a communication extent.
        std::vector<int> model_layers;
        std::optional<MTPSidecarCaptureRole> sidecar_role;
        std::optional<DraftTokenSource> draft_token_source;
    };
    /**
     * @brief Freeze the original retained prefill and MTP graph source roster.
     * @param family Original complete main and learned-source layer identity.
     * @param prefill_rows Exact admitted bucket ladder in per-request rows.
     * @param segment_rows Maximum source-owned variable prefill segment rows.
     * @param batch Original retained request batch capacity.
     * @param mtp Original retained MTP policy, independent of active request depth.
     * @param verifier_family Same typed verifier family consumed by materialization.
     * @param query_programs Actual model-owned main query programs.
     * @param forward_only_owner Original PP membership declaring a separate zero-output source.
     * @return Immutable declaration shared by planning and actual channel registry.
     * @throws std::invalid_argument For malformed original rows, source identities or policies.
     * @throws std::overflow_error For an unrepresentable flattened row count.
     */
    static std::shared_ptr<const MoEOverlayActivationBatchInventory> declare(
        const MoEOverlayInferenceGraphFamilyIdentity& family, std::span<const int> prefill_rows,
        int segment_rows, int batch, const MTPRuntimeConfig& mtp,
        MTPVerifierGraphFamily verifier_family, std::size_t query_programs,
        PipelineForwardOnlyGraphOwner forward_only_owner = PipelineForwardOnlyGraphOwner::None);
    /**
     * @brief Declare exactly the original sources constructed by a component/model owner.
     * @param sources Complete typed source roster, without invented cache reserves or serving variants.
     * @return Immutable metadata consumed by the same require() and physical BOM primitives.
     * @throws std::invalid_argument For empty, duplicate or malformed source/role/row/layer declarations.
     * @details This issues no native, allocation, MPI or runtime authority. The caller must
     * admit each named source through its original PMA before actual graph construction.
     */
    static std::shared_ptr<const MoEOverlayActivationBatchInventory> declareSources(std::span<const Source> sources);
    /** @return Original distinct source owners; each entry contributes one Batch per listed layer. */
    const std::vector<Source>& sources() const noexcept { return sources_; }
    /** @return Common serving main graph count, excluding separate PP-only and predictor owners. */
    std::size_t mainGraphIdentityCount() const noexcept;
    /**
     * @brief Authenticate the real graph invocation before Batch native/storage construction.
     * @param input Exact current ForwardInput authored by its original execution owner.
     * @param layer Original typed main/predictor source and depth identity.
     * @param physical_rows Exact rows embedded by this graph, excluding inactive backing capacity.
     * @param stage_ordinal Layer's original channel manifest ordinal.
     * @throws std::invalid_argument For an undeclared source, row, query program or layer ordinal.
     */
    void require(const ForwardInput& input, const LayerGraphSourceIdentity& layer,
        int physical_rows, std::uint32_t stage_ordinal) const;
private:
    std::vector<Source> sources_;
};
} // namespace llaminar2
