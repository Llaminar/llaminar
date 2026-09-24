/**
 * @file MTPWeightManifest.h
 * @brief Declares the typed inventory and ownership views of MTP weights.
 *
 * The discovered manifest is the single source of truth for both graph
 * bindings and auxiliary participant-local replicas.  In particular, it
 * distinguishes the predictor's dense/shared tensors from routed-expert
 * parents whose physical ownership may belong to ExpertOverlay.
 */

#pragma once

#include <string>
#include <vector>

namespace llaminar2
{
    class IModelLoader;

    /**
     * @enum MTPRoutedExpertWeightAuthority
     * @brief Physical owner of routed-expert parents in an MTP sidecar.
     */
    enum class MTPRoutedExpertWeightAuthority
    {
        /** The participant-local sidecar replica owns complete expert parents. */
        SidecarParticipant,

        /** ExpertOverlay owns per-expert prepared residency and publication. */
        ExpertOverlay,
    };

    struct MTPDepthWeightNames
    {
        int depth_index = 0;
        int source_layer_index = -1;
        bool nextn_block_layout = false;
        bool moe_ffn_layout = false;

        std::string fc;
        std::string pre_fc_norm_hidden;
        std::string pre_fc_norm_embedding;
        std::string final_norm;

        std::string attn_norm;
        std::string wq;
        std::string wk;
        std::string wv;
        std::string wo;
        std::string q_norm;
        std::string k_norm;
        std::string ffn_norm;
        std::string gate_proj;
        std::string up_proj;
        std::string down_proj;

        std::string moe_gate;
        std::string moe_gate_exps;
        std::string moe_up_exps;
        std::string moe_down_exps;
        std::string shared_expert_gate;
        std::string shared_expert_up;
        std::string shared_expert_down;
        std::string shared_expert_gate_inp;

        std::vector<std::string> requiredNames() const;
    };

    struct MTPWeightManifest
    {
        bool available = false;
        int depth = 0;
        bool use_dedicated_embeddings = false;
        std::vector<MTPDepthWeightNames> depths;
        std::vector<std::string> missing_required;
        std::string diagnostic;

        std::vector<std::string> requiredNames() const;

        /**
         * @brief Return the exact weights for one complete local predictor.
         *
         * Attention, predictor FC/norm, router, and shared-expert tensors are
         * always included. Routed-expert parents are included only when the
         * sidecar participant is their declared physical authority; an
         * ExpertOverlay topology obtains those parents from its prepared
         * residency registry instead of creating a second frozen owner.
         *
         * @param routed_authority Typed physical owner of routed experts.
         * @return Sorted, duplicate-free canonical GGUF tensor names.
         */
        [[nodiscard]] std::vector<std::string> participantReplicaNames(
            MTPRoutedExpertWeightAuthority routed_authority) const;
    };

    MTPWeightManifest discoverMTPWeightManifest(
        const IModelLoader &loader,
        const std::string &architecture,
        int base_layer_count,
        bool explicit_mtp);

    /**
     * @brief Return the number of decoder layers that belong to the main graph.
     *
     * Some Qwen3.6 GGUFs are encoded as qwen35 and report block_count including
     * trailing nextn/MTP sidecar block(s). The sidecar weights must remain in
     * the tensor inventory, but orchestration planners and main graph builders
     * should not assign those blocks as ordinary decoder layers.
     */
    int mainLayerCountExcludingMTP(
        const IModelLoader &loader,
        const std::string &architecture,
        int raw_layer_count);

} // namespace llaminar2
