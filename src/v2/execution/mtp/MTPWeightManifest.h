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

    /** @brief Authenticated tensor roles for one learned predictor block. */
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

        /** @return Nonempty mandatory roles, including routed/shared MoE parents. */
        std::vector<std::string> requiredNames() const;
    };

    /** @brief Complete discovery result; unavailable weights never imply a synthetic predictor. */
    struct MTPWeightManifest
    {
        bool available = false;
        int depth = 0;
        bool use_dedicated_embeddings = false;
        std::vector<MTPDepthWeightNames> depths;
        std::vector<std::string> missing_required;
        std::string diagnostic;

        /** @return Sorted, duplicate-free source names for all learned blocks. */
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

    /**
     * @brief Inspect metadata and the tensor directory without loading payloads.
     * @param loader Exact model source whose tensor inventory will be materialized.
     * @param architecture GGUF architecture owning the metadata namespace.
     * @param base_layer_count Raw block count or known main-layer count.
     * @param explicit_mtp Select an actionable diagnostic when MTP was requested.
     * @return Available complete learned weights, or an unavailable diagnostic.
     */
    MTPWeightManifest discoverMTPWeightManifest(
        const IModelLoader &loader,
        const std::string &architecture,
        int base_layer_count,
        bool explicit_mtp);

    /**
     * @brief Require real learned MTP weights before admission or graph allocation.
     * @param loader Exact source directory, not a runtime cache or host mirror.
     * @param architecture GGUF metadata namespace.
     * @param base_layer_count Raw block count or known main-layer count.
     * @return Complete manifest produced by the canonical discovery algorithm.
     * @throws std::invalid_argument For missing, incomplete or unsupported predictors.
     *
     * A rolled predictor still needs learned weights. Increasing the memory BOM
     * cannot make a plain GGUF support MTP, and startup must not silently disable
     * an explicitly requested policy. Draft depth is independent of the number
     * of learned predictor blocks discovered here.
     */
    [[nodiscard]] MTPWeightManifest requireMTPWeightManifest(
        const IModelLoader &loader,
        const std::string &architecture,
        int base_layer_count);

    /**
     * @brief Return the number of decoder layers that belong to the main graph.
     *
     * Some Qwen3.6 GGUFs are encoded as qwen35 and report block_count including
     * trailing nextn/MTP sidecar block(s). The sidecar weights must remain in
     * the tensor inventory, but orchestration planners and main graph builders
     * should not assign those blocks as ordinary decoder layers.
     * @param loader Source metadata and tensor directory, without materialization.
     * @param architecture GGUF metadata namespace.
     * @param raw_layer_count Unadjusted GGUF block count.
     * @return Main-forward count, excluding an authenticated trailing NextN block.
     */
    int mainLayerCountExcludingMTP(
        const IModelLoader &loader,
        const std::string &architecture,
        int raw_layer_count);

} // namespace llaminar2
