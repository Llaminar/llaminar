/**
 * @file ActivationMemoryEstimator.h
 * @brief Declares exact graph-arena memory geometry and sizing helpers.
 *
 * Production graph capture gives every registered BufferId a stable physical
 * allocation.  These declarations price that complete owner set before model
 * weights consume device capacity, including model-specific Qwen hybrid, MTP,
 * and MoE publication buffers.
 */

#pragma once
#include "backends/DeviceId.h"
#include "execution/config/RuntimeConfig.h"
#include "planning/ModelMemoryProfile.h"
#include <cstddef>

namespace llaminar2
{

/**
 * @brief Complete runtime geometry that changes a captured activation arena.
 *
 * The fields mirror graph-resolver formulas rather than backend allocator
 * observations.  This keeps admission deterministic and lets tests compare
 * the planner directly with the declarative BufferArena schema.
 */
struct ActivationGraphMemoryGeometry
{
    int batch_size = 1;             ///< Requests represented by the main graph.
    int resident_graph_rows = 1;    ///< Largest retained prefill bucket.
    int local_d_ff = 0;             ///< Participant-local dense FFN width.
    int local_n_heads = 0;          ///< Participant-local query head count.
    int local_n_kv_heads = 0;       ///< Participant-local KV head count.
    int first_layer = 0;            ///< First main-model layer assigned here.
    int last_layer = -1;            ///< Last main-model layer assigned here.
    int total_shards = 1;           ///< Dense tensor-parallel participant count.
    int mtp_target_query_rows = 2;  ///< Flattened maximum verifier-row capacity.
    MTPTerminalLogitsLayout mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
};

/** @brief Sizes stable activation owners used by captured inference graphs. */
class ActivationMemoryEstimator
{
public:
    /**
     * @brief Estimate the common transformer BufferArena owner set.
     *
     * This scalar overload intentionally covers only architecture-neutral
     * hidden, QKV, FFN, mask, and terminal-logit buffers.  Production model
     * admission uses the typed model-profile overload below.
     */
    static size_t estimate(
        int batch_size,
        int max_seq_len,
        int d_model,
        int d_ff,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int vocab_size,
        DeviceId device
    );

    /**
     * @brief Estimate the logical BufferArena ownership for a model profile.
     *
     * This overload includes architecture-specific projection buffers inferred
     * from the GGUF tensor inventory, plus the declarative MTP and MoE owners
     * whose dimensions come from @p geometry. It is the production planning
     * entrypoint; the scalar overload remains useful for generic-model tests.
     *
     * @param profile GGUF-derived model and tensor geometry.
     * @param geometry Complete participant-local captured graph geometry.
     * @param device Device being admitted; activation ownership is symmetric
     *        across backends, but retaining the device makes the API explicit.
     * @return Required bytes for the graph-owned BufferArena tensors.
     */
    static size_t estimate(
        const ModelMemoryProfile& profile,
        const ActivationGraphMemoryGeometry& geometry,
        DeviceId device
    );
};

} // namespace llaminar2
