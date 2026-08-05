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
     * @param batch_size Maximum simultaneously admitted request count.
     * @param resident_graph_rows Maximum token rows resident in one graph.
     * @param local_d_ff Dense FFN width owned by this participant.
     * @param first_layer First model layer assigned to this participant.
     * @param last_layer Last model layer assigned to this participant.
     * @param total_shards Number of tensor-parallel participants.
     * @param device Planned execution device.
     * @return Required pre-allocation reserve in bytes.
     * @throws std::runtime_error when declared MoE geometry is incomplete or
     *         the requested row envelope cannot be represented safely.
     */
    static size_t estimate(
        const ModelMemoryProfile& profile,
        int batch_size,
        int resident_graph_rows,
        int local_d_ff,
        int first_layer,
        int last_layer,
        int total_shards,
        DeviceId device
    );
};

} // namespace llaminar2
