/**
 * @file MoEGroupedVerifierHistogramBoundarySet.h
 * @brief Typed discovery and resolution of per-layer MTP histogram boundaries.
 *
 * A grouped main-model verifier contains one router and exactly one routing-
 * history boundary per MoE layer.  Static placement deliberately resolves that
 * boundary without a publisher, while Observe/Dynamic placement resolves it to
 * one accepted-row publisher and one shared model-lifetime stream.  This class
 * owns that small setup-time state machine so graph orchestrators cannot rebuild
 * the lifecycle from nullable pointers and loosely related booleans.
 */

#pragma once

#include "IMoEGroupedVerifierHistogramPublisher.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Select the explicit role for one graph-lowered verifier boundary.
     *
     * @param owns_boundary Whether this concrete router/expert is the sole
     *        per-layer history boundary.
     * @param collects_history Whether the production maintenance policy owns
     *        runtime demand (Observe/Dynamic rather than Static).
     * @return A total typed role; no pointer state is consulted.
     */
    [[nodiscard]] constexpr MoEGroupedVerifierHistogramRole
    selectMoEGroupedVerifierHistogramRole(
        bool owns_boundary,
        bool collects_history) noexcept
    {
        if (!owns_boundary)
            return MoEGroupedVerifierHistogramRole::NotOwner;
        return collects_history
                   ? MoEGroupedVerifierHistogramRole::DeferredAcceptedRows
                   : MoEGroupedVerifierHistogramRole::StaticNoPublication;
    }

    /**
     * @brief Resolved accepted-state bindings in routed-layer order.
     *
     * Static graphs produce an empty publisher vector and a null stream.
     * Observe/Dynamic graphs produce one publisher per routed layer and one
     * exact stream shared by all of them.
     */
    struct MoEGroupedVerifierHistogramBoundaryResolution
    {
        std::vector<IMoEGroupedVerifierHistogramPublisher *> publishers;
        void *publication_stream = nullptr;
    };

    /**
     * @brief Setup-time state machine for grouped-verifier history ownership.
     *
     * The set starts in @ref State::Collecting. Callers register each router and
     * each candidate router/expert stage while walking one verifier graph, then
     * call @ref resolve exactly once. Any malformed transition moves the object
     * permanently to @ref State::Faulted; successful resolution moves it to
     * @ref State::Resolved. This fail-closed lifecycle prevents a partially
     * discovered graph from being reused after an error.
     */
    class MoEGroupedVerifierHistogramBoundarySet final
    {
    public:
        /** @brief Complete lifecycle states for one discovery pass. */
        enum class State : std::uint8_t
        {
            /** Router and stage identities may still be registered. */
            Collecting,
            /** One complete, immutable resolution has been published. */
            Resolved,
            /** A rejected transition permanently invalidated this pass. */
            Faulted,
        };

        /**
         * @brief Register the sole router for a routed MoE layer.
         *
         * @param layer Non-negative model layer index.
         * @param node_name Stable diagnostic graph-node name.
         * @param error Receives an actionable failure reason.
         * @return true when the router was admitted.
         */
        bool registerRoutedLayer(
            int layer,
            std::string node_name,
            std::string *error)
        {
            if (!requireCollecting("register a routed layer", error))
                return false;
            if (layer < 0 || node_name.empty())
                return fault("routed MoE layer has an invalid identity", error);
            const auto [existing, inserted] =
                routed_layers_.emplace(layer, std::move(node_name));
            if (!inserted)
            {
                return fault(
                    "duplicate routers for routed MoE layer " +
                        std::to_string(layer) + ": " + existing->second,
                    error);
            }
            return true;
        }

        /**
         * @brief Register one router/expert stage's explicit boundary role.
         *
         * @ref MoEGroupedVerifierHistogramRole::NotOwner is a valid no-op.
         * Static boundaries must expose no stream. Deferred boundaries must
         * expose the same non-null table-owned stream across every layer.
         *
         * @param stage Typed stage contract discovered in the verifier graph.
         * @param node_name Stable diagnostic graph-node name.
         * @param error Receives an actionable failure reason.
         * @return true when the role was admitted or was an explicit non-owner.
         */
        bool registerStage(
            IMoEGroupedVerifierHistogramPublisher *stage,
            std::string node_name,
            std::string *error)
        {
            if (!requireCollecting("register a history boundary", error))
                return false;
            if (!stage || node_name.empty())
                return fault("verifier history boundary has an invalid stage identity", error);

            const MoEGroupedVerifierHistogramRole role =
                stage->groupedVerifierHistogramRole();
            if (role == MoEGroupedVerifierHistogramRole::NotOwner)
                return true;

            const int layer = stage->groupedVerifierHistogramLayerIndex();
            if (layer < 0)
            {
                return fault(
                    "verifier history boundary has an invalid layer: " +
                        node_name,
                    error);
            }

            void *const stream =
                stage->groupedVerifierHistogramPublicationStream();
            switch (role)
            {
            case MoEGroupedVerifierHistogramRole::StaticNoPublication:
                if (stream)
                {
                    return fault(
                        "Static verifier history boundary unexpectedly owns a publication stream: " +
                            node_name,
                        error);
                }
                break;
            case MoEGroupedVerifierHistogramRole::DeferredAcceptedRows:
                if (!stream)
                {
                    return fault(
                        "deferred verifier history boundary has no model-lifetime publication stream: " +
                            node_name,
                        error);
                }
                if (publication_stream_ && publication_stream_ != stream)
                {
                    return fault(
                        "deferred verifier history boundaries do not share one table-owned publication stream: " +
                            node_name,
                        error);
                }
                publication_stream_ = stream;
                break;
            case MoEGroupedVerifierHistogramRole::NotOwner:
                break;
            default:
                return fault("verifier history boundary has an unknown typed role", error);
            }

            const auto [existing, inserted] = boundaries_.emplace(
                layer,
                Boundary{
                    .role = role,
                    .stage = stage,
                    .node_name = std::move(node_name),
                });
            if (!inserted)
            {
                return fault(
                    "multiple verifier history boundaries for routed MoE layer " +
                        std::to_string(layer) + ": " +
                        existing->second.node_name,
                    error);
            }
            return true;
        }

        /**
         * @brief Resolve the complete immutable publication binding.
         *
         * Every routed layer must own exactly one boundary and no boundary may
         * exist without a matching router. Mixed Static and deferred roles are
         * rejected because one verifier graph has one maintenance policy.
         *
         * @param output Receives ordered active publishers and their stream.
         * @param error Receives an actionable failure reason.
         * @return true after the state transitions to @ref State::Resolved.
         */
        bool resolve(
            MoEGroupedVerifierHistogramBoundaryResolution &output,
            std::string *error)
        {
            output = {};
            if (!requireCollecting("resolve history boundaries", error))
                return false;
            if (boundaries_.size() != routed_layers_.size())
            {
                return fault(
                    "verifier history boundary cardinality does not match routed layers",
                    error);
            }

            MoEGroupedVerifierHistogramRole graph_role =
                MoEGroupedVerifierHistogramRole::NotOwner;
            output.publishers.reserve(routed_layers_.size());
            for (const auto &[layer, router_node] : routed_layers_)
            {
                const auto boundary = boundaries_.find(layer);
                if (boundary == boundaries_.end())
                {
                    return fault(
                        "routed MoE layer " + std::to_string(layer) +
                            " has no explicit history boundary: " + router_node,
                        error);
                }
                if (graph_role == MoEGroupedVerifierHistogramRole::NotOwner)
                    graph_role = boundary->second.role;
                else if (graph_role != boundary->second.role)
                {
                    return fault(
                        "one verifier graph mixes Static and deferred history roles",
                        error);
                }
                if (boundary->second.role ==
                    MoEGroupedVerifierHistogramRole::DeferredAcceptedRows)
                {
                    output.publishers.push_back(boundary->second.stage);
                }
            }
            output.publication_stream = publication_stream_;
            if (output.publishers.empty() !=
                (output.publication_stream == nullptr))
            {
                return fault(
                    "resolved verifier publishers and publication stream are incomplete",
                    error);
            }
            state_ = State::Resolved;
            return true;
        }

        /** @return Current fail-closed lifecycle state. */
        [[nodiscard]] State state() const noexcept { return state_; }

    private:
        /** @brief One selected router/expert boundary recorded during discovery. */
        struct Boundary
        {
            MoEGroupedVerifierHistogramRole role =
                MoEGroupedVerifierHistogramRole::NotOwner;
            IMoEGroupedVerifierHistogramPublisher *stage = nullptr;
            std::string node_name;
        };

        /**
         * @brief Reject mutation outside the collecting state.
         * @param action Human-readable attempted transition.
         * @param error Receives the failure reason.
         * @return true only while discovery remains open.
         */
        bool requireCollecting(const char *action, std::string *error)
        {
            if (state_ == State::Collecting)
                return true;
            if (error)
            {
                *error = std::string("cannot ") + action +
                         " after verifier boundary discovery has terminated";
            }
            return false;
        }

        /**
         * @brief Enter the terminal fault state and publish one diagnostic.
         * @param reason Complete failure reason.
         * @param error Optional diagnostic destination.
         * @return Always false for fluent validation paths.
         */
        bool fault(std::string reason, std::string *error)
        {
            state_ = State::Faulted;
            if (error)
                *error = std::move(reason);
            return false;
        }

        State state_ = State::Collecting;
        std::map<int, std::string> routed_layers_;
        std::map<int, Boundary> boundaries_;
        void *publication_stream_ = nullptr;
    };
}
