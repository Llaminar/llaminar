/**
 * @file WorkspaceMemoryEstimator.h
 * @brief Declares pre-allocation workspace sizing from model and graph geometry.
 *
 * These estimates run before device graph construction. They must therefore
 * contain every geometry term that can affect the later exact workspace-family
 * plan; under-estimation can consume VRAM with weights and persistent state
 * before the graph allocator has a chance to publish its stable buffers.
 */

#pragma once
#include "backends/DeviceId.h"
#include "execution/config/RuntimeConfig.h"
#include "planning/ModelMemoryProfile.h"
#include <cstddef>

namespace llaminar2
{

/**
 * @brief Immutable graph and device geometry used by workspace admission.
 *
 * Every field is known before weight preparation. Keeping the record typed is
 * important: resident graph rows and full KV-cache capacity affect different
 * workspace families, while physical SM/CU count participates in the exact
 * capture-time attention policy.
 */
struct WorkspaceMemoryGeometry
{
    DeviceId device = DeviceId::invalid(); ///< Participant being admitted.
    /**
     * CUDA SMs, ROCm CUs, or configured physical-core CPU workers.
     * This is execution parallelism, not the host's SMT thread count.
     */
    int device_compute_units = 0;
    int batch_size = 1; ///< Maximum simultaneously admitted requests.
    int resident_graph_rows = 1; ///< Largest token-row capture bucket.
    int max_context_rows = 1; ///< Stable KV-cache capacity in token rows.
    bool owns_embedding = false; ///< Participant publishes token-ID state.
    int shard_index = 0; ///< Zero-based participant index in the TP group.
    bool has_exact_tensor_parallel_assignment = false; ///< Local ranges below are authoritative.
    int local_d_ff = 0; ///< Dense FFN output width owned locally.
    int local_d_ff_start = 0; ///< First global dense-FFN column owned locally.
    int local_query_head_start = 0; ///< First global query head owned locally.
    int local_query_heads = 0; ///< Exact participant-local query heads.
    int local_kv_head_start = 0; ///< First global KV head owned locally.
    int local_kv_heads = 0; ///< Exact participant-local KV heads.
    int local_vocab_start = 0; ///< First global vocabulary row owned locally.
    int local_vocab = 0; ///< Exact participant-local vocabulary rows.
    int first_layer = 0; ///< First model layer owned by this participant.
    int last_layer = -1; ///< Last model layer owned by this participant.
    int total_shards = 1; ///< Tensor-parallel degree for local dimensions.
    bool apportioned_routed_experts = false; ///< Experts are whole-owner slices, not TP slices.
    /**
     * Largest flattened grouped-verifier shape retained beside the main graph.
     * Zero means no MTP graph family is materialized on this participant.
     */
    int mtp_target_query_rows = 0;
    /** Terminal projection ownership retained by the MTP graph family. */
    MTPTerminalLogitsLayout mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
};

class WorkspaceMemoryEstimator
{
public:
    /**
     * @brief Estimate dense kernel workspace from explicit dimensions.
     *
     * GPU devices reserve graph-stable GEMM, terminal projection, and
     * quantization scratch. This legacy dimensional overload lacks CPU
     * attention/head/worker geometry and therefore returns zero for CPU; use
     * the model-aware overload for production CPU admission.
     *
     * @param batch_size Maximum simultaneously admitted request count.
     * @param max_seq_len Maximum rows resident in one captured graph.
     * @param d_model Hidden width.
     * @param d_ff Local dense FFN intermediate width.
     * @param vocab_size Terminal projection width.
     * @param device Planned execution device.
     * @return Required pre-allocation reserve in bytes.
     */
    static size_t estimate(
        int batch_size,
        int max_seq_len,
        int d_model,
        int d_ff,
        int vocab_size,
        DeviceId device
    );

    /**
     * @brief Estimate the production graph-family workspace from GGUF geometry.
     *
     * In addition to dense requirements, this overload accounts for hybrid
     * recurrent scratch and exact backend-specific MoE workspace declarations.
     * A positive `mtp_target_query_rows` also reserves a second compact graph
     * envelope. Main prefill and grouped-verifier/MTP graphs are event-ordered,
     * but their stable workspace names and initialize-once publications are not
     * fully aliasable. Hybrid models use the same exact GDN rollback-state
     * contract as their graph stages, add every per-layer slot bank, and share
     * only the larger serial transient envelope. Non-hybrid models retain the
     * conservative sum of independently complete envelopes.
     * Declared MoE models with incomplete routing geometry are rejected because
     * silently returning a dense-only estimate would permit a late VRAM failure.
     *
     * @param profile Parsed model geometry and tensor inventory.
     * @param geometry Exact graph, placement, and physical-device geometry.
     * @return Required pre-allocation reserve in bytes.
     * @throws std::runtime_error when declared MoE geometry is incomplete or
     *         the requested row envelope cannot be represented safely.
     */
    static size_t estimate(
        const ModelMemoryProfile& profile,
        const WorkspaceMemoryGeometry& geometry);

    /**
     * @brief Estimate one sparse ExpertOverlay endpoint's GPU workspace.
     *
     * This covers both legal endpoint graph forms without charging terminal
     * projection, attention, or other continuation-only graph names. A mapped
     * node-local follower executes the router-shaped `(rows, top_k)` graph
     * directly, while a host-boundary follower compacts routes to
     * `(rows * top_k, 1)`. Both variants can remain retained in one admitted
     * family, so preflight prices their exact merged workspace names.
     *
     * @param profile Parsed MoE geometry.
     * @param geometry Exact graph and participant geometry.
     * @return Required graph-stable workspace bytes, or zero for CPU.
     * @throws std::runtime_error for incomplete MoE geometry or non-GPU misuse.
     */
    static size_t estimateRoutedExpertParticipant(
        const ModelMemoryProfile& profile,
        const WorkspaceMemoryGeometry& geometry);
};

} // namespace llaminar2
