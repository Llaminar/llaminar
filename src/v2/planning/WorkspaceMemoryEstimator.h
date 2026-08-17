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
    int device_compute_units = 0; ///< CUDA SMs or ROCm CUs visible to it.
    int batch_size = 1; ///< Maximum simultaneously admitted requests.
    int resident_graph_rows = 1; ///< Largest token-row capture bucket.
    int max_context_rows = 1; ///< Stable KV-cache capacity in token rows.
    int local_d_ff = 0; ///< Dense FFN output width owned locally.
    int first_layer = 0; ///< First model layer owned by this participant.
    int last_layer = -1; ///< Last model layer owned by this participant.
    int total_shards = 1; ///< Tensor-parallel degree for local dimensions.
    bool apportioned_routed_experts = false; ///< Experts are whole-owner slices, not TP slices.
};

class WorkspaceMemoryEstimator
{
public:
    /**
     * @brief Estimate dense kernel workspace from explicit dimensions.
     *
     * GPU devices reserve graph-stable GEMM, terminal projection, and
     * quantization scratch. CPU kernels do not use DeviceWorkspaceManager and
     * therefore return zero.
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
