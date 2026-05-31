/**
 * @file Qwen35MoEGraph.h
 * @brief Qwen 3.5 MoE compute graph builder (hybrid GDN + FA + MoE FFN)
 *
 * Extends Qwen35Graph with Mixture-of-Experts FFN blocks.
 * Attention architecture (GDN + FA hybrid) is identical to dense Qwen3.5.
 */

#pragma once

#include "../qwen35/Qwen35Graph.h"
#include "../../execution/moe/MoERuntimeTable.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace llaminar2
{
    struct PrefixFingerprintMaterial;

    /**
     * @brief Qwen 3.5 MoE graph builder
     *
     * Inherits hybrid GDN+FA attention from Qwen35Graph.
     * Overrides FFN graph building to use SparseMoeBlock:
     *   Router → top-K experts + shared expert + sigmoid gate
     */
    class Qwen35MoEGraph : public Qwen35Graph
    {
    public:
        /// Construct with full model context
        Qwen35MoEGraph(std::shared_ptr<ModelContext> model_ctx,
                       std::shared_ptr<IMPIContext> mpi_ctx,
                       const GraphConfig &config);

        /// Construct for layer-level operations only
        Qwen35MoEGraph(const GraphConfig &config,
                       std::shared_ptr<IMPIContext> mpi_ctx = nullptr);

        ~Qwen35MoEGraph() = default;

        // =====================================================================
        // IGraphBuilder overrides
        // =====================================================================

        std::string architectureName() const override { return "qwen35moe"; }

        GraphSchema getSchema() const override;

        /// Override FFN graph building for MoE layers
        ComputeGraph buildFFNGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            DeviceId device) override;

        ComputeGraph buildMTPGraph(
            int depth_idx,
            const MTPDepthWeights &weights,
            const MTPForwardInput &input,
            MTPForwardOutput &output);

        ComputeGraph buildMTPGraph(
            int depth_idx,
            const MTPDepthWeightBindings &bindings,
            const MTPForwardInput &input,
            MTPForwardOutput &output) override;

        /// Override resolver config to register MoE buffer IDs and formulas
        GraphResolverConfig getResolverConfig(int seq_len) const override;

        /// Reset MoE runtime state between independent inference sessions.
        void resetState() override;

        /// Append active MoE runtime placement state to prefix-cache fingerprints.
        void appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const override;

    private:
        struct ScopedMTPGraphContext
        {
            ScopedMTPGraphContext(Qwen35MoEGraph &graph, int depth_idx);
            ~ScopedMTPGraphContext();

            Qwen35MoEGraph &graph;
            bool previous_active = false;
            int previous_depth_idx = -1;
        };

        IMoERuntimeTable *moeRuntimeTableForDevice(DeviceId device,
                                                   int prefill_token_capacity = 0,
                                                   const std::string &key_suffix = {});

        std::unordered_map<std::string, std::unique_ptr<MoERuntimeTable>> moe_runtime_tables_;
        bool mtp_graph_context_active_ = false;
        int mtp_graph_depth_idx_ = -1;
    };

} // namespace llaminar2
