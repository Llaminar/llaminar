/**
 * @file Qwen4ExpBufferLayout.h
 * @brief One declaration of Flash Next arena identities, geometry and lifetimes.
 *
 * Schema allocation and graph lookup consume the same typed buffer inventory.
 * Scratch is shared by sequential layers; request histories and indexer pools
 * retain independent model-layer/request slices across captured replays.
 */
#pragma once
#include "../../memory/BufferId.h"
#include "../../kernels/qwen4exp/QwenSparseAttentionArithmetic.h"
#include "../../execution/local_execution/graph/GraphSchema.h"
#include "../../planning/ModelMemoryProfile.h"
#include "../../loaders/ImmutableRowTablePolicy.h"
#include <optional>
#include <span>
#include <string_view>

namespace llaminar2
{
struct GraphConfig;
struct GraphResolverConfig;

/** @brief Explicit participant envelope shared by graph allocation and metadata-only admission. */
struct Qwen4ExpBufferDimensions
{
    int main_layers = 0; ///< Total ordinary decoder blocks, excluding predictor blocks.
    int first_layer = 0; ///< Global first ordinary block assigned to this participant.
    int layer_count = 0; ///< Number of ordinary blocks assigned to this participant.
    int compute_width = 0; ///< GEMM width, independent of the widened residual trunk.
    int context_tokens = 0; ///< Admitted per-request context capacity.
    int request_capacity = 0; ///< Independent retained request slots; never inferred from row capacity.
    int flattened_rows = 0; ///< Exact maximum resident graph rows across requests.
    int local_query_heads = 0; ///< Physical participant-local attention heads.
    int retained_mtp_candidate_rows = 0; ///< Canonical flattened verifier envelope; zero omits the retained bank.
    DeviceId device; ///< Backend-specific declaration placement.
    qwen4exp::QwenSparseAttentionPhaseSchedule attention_phase_schedule = qwen4exp::QwenSparseAttentionPhaseSchedule::Uniform; ///< Immutable canonical setup scalar; formulas never consult the environment.
    /** Concrete table tier; absent only in the planner's higher-priority base BOM.
     * The unresolved base excludes tier-specific native-row/protocol scratch.
     * Production graph allocation requires an admitted concrete placement. */
    std::optional<ImmutableRowLookupExecution> ngram_lookup;
    int ngram_fixed_head_participants = 0; ///< Exact existing fixed cohort P2/4/8; zero for other execution modes.
};

/** @brief The graph placement that actually consumes a declaration. */
enum class Qwen4ExpBufferPlacement { EveryParticipant, PLE, PLEHostStorage, PLEGPUHostStorage, PLEFixedHeadStorage, Sparse, SparseGPU, SparseMTP, PLEMTP };

/** @brief One registered buffer and its immutable schema allocation contract. */
struct Qwen4ExpBufferDeclaration
{
    BufferId id;
    BufferSpec spec;
    Qwen4ExpBufferPlacement placement = Qwen4ExpBufferPlacement::EveryParticipant;
};

/** @brief Return the single inventory used by schema, resolver and arena binding. */
std::span<const Qwen4ExpBufferDeclaration> qwen4ExpBufferDeclarations();

/**
 * @brief Extend an existing MoE schema with explicit branch and PLE/QSA storage.
 * @param schema Shared GDN/MoE activation inventory, extended in place.
 * @param model Exact participant device and main-layer interval.
 * @details Residual and predictor trunk roles widen to 4H. Ordinary block and
 * vocabulary input/output projections retain H, so changing a residual width
 * cannot silently expand GEMM or communication operands.
 */
void applyQwen4ExpBufferLayout(GraphSchema &schema, const GraphConfig &model);

/** @brief Apply the same declaration placement and branch-width edits using metadata-only geometry. */
void applyQwen4ExpBufferLayout(GraphSchema &schema, const Qwen4ExpMemoryGeometry &geometry,
                             const Qwen4ExpBufferDimensions &dimensions);

/**
 * @brief Publish exact model-derived dimensions and every buffer name/identity together.
 * @param resolver Existing shared Qwen GDN/MoE resolver configuration.
 * @param model Authenticated QFN geometry and admitted context/request envelope.
 * @throws std::invalid_argument On absent metadata or impossible model/PP geometry.
 */
void applyQwen4ExpBufferGeometry(GraphResolverConfig &resolver, const GraphConfig &model);

/** @brief Resolve the same checked formulas without constructing a GGUF-backed graph config. */
void applyQwen4ExpBufferGeometry(GraphResolverConfig &resolver, const Qwen4ExpMemoryGeometry &geometry,
                               const Qwen4ExpBufferDimensions &dimensions);

/**
 * @brief Price only declarative QFN additions and widened roles over the shared base arena.
 * @param common_schema Actual shared GDN/MoE buffer inventory before QFN extension.
 * @param resolver Common row/model/MTP formulas already used by admission.
 * @param geometry Immutable model metadata projection.
 * @param dimensions Exact participant envelope.
 * @return Additional ActivationArena bytes, including retained PLE/indexer state.
 */
[[nodiscard]] std::size_t qwen4ExpArenaExtensionBytes(const GraphSchema &common_schema,
    GraphResolverConfig resolver, const Qwen4ExpMemoryGeometry &geometry,
    const Qwen4ExpBufferDimensions &dimensions);

/**
 * @brief Price tier-specific lookup scratch using the same registered arena declarations.
 * @param resolver Frozen common geometry after higher-priority row planning.
 * @param geometry Exact source and model metadata.
 * @param dimensions Participant envelope with an explicit candidate storage tier.
 * @return Only the native-row/protocol bytes absent from the unresolved base.
 * @details This contributes a typed ActivationArena BOM item to the final PMA
 * candidate; it neither changes row capacity nor makes an allocation decision.
 */
[[nodiscard]] std::size_t qwen4ExpLookupArenaBytes(GraphResolverConfig resolver,
    const Qwen4ExpMemoryGeometry &geometry, const Qwen4ExpBufferDimensions &dimensions);
} // namespace llaminar2
