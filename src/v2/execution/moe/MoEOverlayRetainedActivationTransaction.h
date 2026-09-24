/**
 * @file MoEOverlayRetainedActivationTransaction.h
 * @brief Topology-neutral lowering for retained node-local activation epochs.
 *
 * A heterogeneous ExpertOverlay transaction is one device graph per physical
 * endpoint. Ordinary captured fragments perform packet preparation, expert
 * compute, and deterministic return folding; mapped 64-bit timeline nodes
 * connect those fragments without a host dispatch or wait between model
 * layers. This file owns the typed ordering contract used to assemble those
 * endpoint graphs. It deliberately contains no CUDA/ROCm role assumptions,
 * tier labels, or participant-count policy: all endpoint and lane identities
 * arrive from the topology-selected node-local transport.
 */

#pragma once

#include "MoEOverlayNodeLocalRankBatchTransport.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    class IGPUGraphCapture;
    class ComputeGraph;

    /**
     * @brief One already-coordinated native child and its declarative stages.
     *
     * Capture-wave ownership remains with the ordinary graph controller. This
     * narrow borrowed view gives the ExpertOverlay lowerer only the information
     * needed to place typed timeline edges around packet stages. Both the capture
     * and stage-name storage must outlive parent construction and replay.
     */
    struct MoEOverlayRetainedCaptureUnit
    {
        const IGPUGraphCapture *capture = nullptr; ///< Non-empty native child graph.
        std::span<const std::string> stage_names; ///< Exact contiguous graph stages in this child.
    };

    /**
     * @brief Captured continuation fragments for one ordered model stage.
     *
     * The dispatch fragment writes the shared packet descriptor and bytes. The
     * return fragment consumes the matching shared return and folds it into the
     * continuation tensor. Both fragments execute on the exact device carried
     * by the surrounding lane binding.
     */
    struct MoEOverlayContinuationActivationStage
    {
        std::int32_t model_layer_index = -1; ///< Exact manifest layer at this ordinal.
        const IGPUGraphCapture *dispatch_fragment = nullptr; ///< Packet pack fragment.
        const IGPUGraphCapture *return_fragment = nullptr; ///< Ordered return-consume fragment.
    };

    /**
     * @brief One planner-ordered continuation lane at one model stage.
     *
     * A continuation device may fan one routed layer out to any number of
     * node-local GPU participants.  Each lane carries its own mapped-region
     * lifetime because endpoints may belong to different ranks, domains, or
     * integer-priority tiers.  The two fragments still execute on the single
     * continuation device that owns the surrounding parent graph.
     */
    struct MoEOverlayContinuationActivationLaneStage
    {
        MoEOverlayMappedActivationDeviceLane lane; ///< Exact source-device alias selected by planning.
        const IGPUGraphCapture *dispatch_fragment = nullptr; ///< Pack only this participant's routed rows.
        const IGPUGraphCapture *return_fragment = nullptr; ///< Fold this participant in canonical planner order.
    };

    /**
     * @brief Full continuation work surrounding one multi-lane sparse boundary.
     *
     * `prefix_fragments` contain captured work through routing. Every dispatch
     * is then published before `overlap_fragments` run, allowing local expert or
     * shared-expert work to overlap follower execution.  Returns are awaited and
     * folded in the exact supplied lane order, after which `suffix_fragments`
     * may continue the dense layer.  Optional fragments are omitted rather than
     * represented by empty native graphs.
     */
    struct MoEOverlayContinuationActivationFanoutStage
    {
        std::int32_t model_layer_index = -1; ///< Exact manifest layer at this ordinal.
        std::vector<const IGPUGraphCapture *> prefix_fragments; ///< Ordered captured work ending at the routed frontier.
        std::vector<MoEOverlayContinuationActivationLaneStage> lanes; ///< Canonical planner lane order.
        std::vector<const IGPUGraphCapture *> overlap_fragments; ///< Local work queued after every dispatch publication.
        std::vector<const IGPUGraphCapture *> suffix_fragments; ///< Dense continuation after all ordered folds.
    };

    /**
     * @brief Captured follower fragment for one ordered model stage.
     *
     * The fragment consumes the shared dispatch, performs the real local expert
     * work, and packs the compact return. Its final packet writes precede the
     * native return-publication node inserted by the transaction lowerer.
     */
    struct MoEOverlayFollowerActivationStage
    {
        std::int32_t model_layer_index = -1; ///< Exact manifest layer at this ordinal.
        std::vector<const IGPUGraphCapture *> prefix_fragments; ///< Work ordered after admission but before dispatch wait.
        std::vector<const IGPUGraphCapture *> compute_fragments; ///< Consume, expert compute, return pack.
        std::vector<const IGPUGraphCapture *> suffix_fragments; ///< Work ordered after return publication.
    };

    /**
     * @brief Build complete device-owned endpoint graphs for one mapped lane.
     *
     * Stage ordinals are their positions in the supplied span. The lowerer
     * derives alternating buffer indexes and monotonic visit values from those
     * ordinals, so callers cannot accidentally publish a layer through the
     * wrong shared bank. Every captured fragment remains owned by the caller
     * and must outlive the destination graph executable.
     */
    class MoEOverlayRetainedActivationTransaction final
    {
    public:
        MoEOverlayRetainedActivationTransaction() = delete;

        /**
         * @brief Assemble the continuation's full retained sparse transaction.
         *
         * The resulting order is admission, then for every manifest stage:
         * dispatch pack, dispatch publication, return wait, and ordered return
         * consume. No host-visible operation exists between those nodes.
         *
         * @param destination Empty graph owner for @p lane.device.
         * @param lane Planner-resolved source-device alias of one participant lane.
         * @param manifest_layers Exact retained graph-family layer manifest.
         * @param stages Captured fragments in the same manifest order.
         * @throws std::invalid_argument for an incomplete or divergent identity.
         * @throws std::runtime_error when native graph lowering fails.
         */
        static void buildContinuation(
            IGPUGraphCapture &destination,
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::span<const std::int32_t> manifest_layers,
            std::span<const MoEOverlayContinuationActivationStage> stages);

        /**
         * @brief Assemble one arbitrary-cardinality continuation transaction.
         *
         * The lane sequence is supplied by the resolved topology and must be
         * identical at every manifest stage.  The lowerer never assigns meaning
         * to a backend family, rank number, tier index, or lane count.  It first
         * admits every endpoint, publishes every dispatch for a layer, runs the
         * optional local overlap fragment, and then consumes returns in the
         * caller's canonical order.
         *
         * @param destination Empty native graph on the continuation device.
         * @param manifest_layers Complete ordered graph-family manifest.
         * @param stages Captured dense and sparse fragments in manifest order.
         * @throws std::invalid_argument for changing, duplicate, or incomplete
         *         planner lanes and for divergent fragment/manifest identity.
         * @throws std::runtime_error when native graph lowering fails.
         */
        static void buildContinuationFanout(
            IGPUGraphCapture &destination,
            std::span<const std::int32_t> manifest_layers,
            std::span<const MoEOverlayContinuationActivationFanoutStage> stages);

        /**
         * @brief Assemble a follower's full retained sparse transaction.
         *
         * The resulting order is admission, then for every manifest stage:
         * dispatch wait, packet consume plus real expert compute plus return
         * pack, and return publication. The endpoint may be any planner-chosen
         * GPU backend and may appear at any integer-priority tier.
         *
         * @param destination Empty graph owner for @p lane.device.
         * @param lane Planner-resolved follower-device alias of one participant lane.
         * @param manifest_layers Exact retained graph-family layer manifest.
         * @param stages Captured fragments in the same manifest order.
         * @throws std::invalid_argument for an incomplete or divergent identity.
         * @throws std::runtime_error when native graph lowering fails.
         */
        static void buildFollower(
            IGPUGraphCapture &destination,
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::span<const std::int32_t> manifest_layers,
            std::span<const MoEOverlayFollowerActivationStage> stages);

        /**
         * @brief Derive and assemble a continuation parent from captured graph units.
         *
         * Packet stage types are the declarative boundary contract. The lowerer
         * derives stage ordinals, model-layer manifest, planner lane order, and
         * dense prefix/overlap/suffix fragments from @p graph and @p units. It
         * rejects missing zero-row lanes, reordered return folds, packet stages
         * fused across a timeline edge, or any follower-only packet operation.
         *
         * @param destination Empty native parent on the continuation device.
         * @param graph Exact declarative graph represented by @p units.
         * @param units Active native units in graph execution order.
         */
        static void buildContinuationFromCapturedUnits(
            IGPUGraphCapture &destination,
            const ComputeGraph &graph,
            std::span<const MoEOverlayRetainedCaptureUnit> units);

        /**
         * @brief Derive and assemble one follower parent from captured graph units.
         *
         * Every stage ordinal must contain exactly one dispatch-consume frontier
         * and one return-pack frontier for the same planner lane. Ordinary child
         * graphs between those markers remain in the device-owned compute body;
         * setup/teardown children outside them are retained before the wait or
         * after the publication without introducing host execution.
         *
         * @param destination Empty native parent on the follower device.
         * @param graph Exact single-endpoint declarative graph.
         * @param units Active native units in graph execution order.
         */
        static void buildFollowerFromCapturedUnits(
            IGPUGraphCapture &destination,
            const ComputeGraph &graph,
            std::span<const MoEOverlayRetainedCaptureUnit> units);

        /**
         * @brief Retain ordered children whose timeline edges are stage-owned.
         *
         * Scalar and multi-row packet stages capture their complete mapped
         * timeline or multi-stream fork/join DAG inside ordinary child units.
         * Likewise, a same-domain sibling may contain only authority-aligned
         * collective cutpoints and no packet stage of its own. In every case
         * the parent adds exact child-to-child dependencies but must not
         * manufacture a second mapped wait or publication.
         *
         * Every non-manual, non-passive graph node must occur exactly once in
         * @p units and all such nodes must belong to one GPU. Packet stages are
         * deliberately accepted here because their exact timeline nodes are
         * already part of their captured child.
         *
         * @param destination Empty native parent on the participant GPU.
         * @param graph Exact declarative heterogeneous endpoint graph.
         * @param units Active native units in graph execution order.
         * @throws std::invalid_argument for missing, reordered, or mixed-device
         *         units.
         * @throws std::runtime_error when native graph lowering fails.
         */
        static void buildStageOwnedTransactionFromCapturedUnits(
            IGPUGraphCapture &destination,
            const ComputeGraph &graph,
            std::span<const MoEOverlayRetainedCaptureUnit> units);
    };
} // namespace llaminar2
