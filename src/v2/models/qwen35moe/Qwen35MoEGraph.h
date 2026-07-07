/**
 * @file Qwen35MoEGraph.h
 * @brief Qwen 3.5 MoE compute graph builder (hybrid GDN + FA + MoE FFN)
 *
 * Extends Qwen35Graph with Mixture-of-Experts FFN blocks.
 * Attention architecture (GDN + FA hybrid) is identical to dense Qwen3.5.
 */

#pragma once

#include "../qwen35/Qwen35Graph.h"
#include "../../execution/moe/DeviceMoERebalanceController.h"
#include "../../execution/moe/MoERuntimeTable.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llaminar2
{
    class DeviceMoETransferSlotDirectory;
    class DeviceMoERebalanceTransferState;
    class ILocalTPContext;
    class IMoERuntimeTable;
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

        ComputeGraph buildDeviceMoERebalanceMaintenanceGraph(
            DeviceId device,
            DeviceMoERebalanceMaintenanceGraphKind kind = DeviceMoERebalanceMaintenanceGraphKind::Probe,
            uint64_t payload_edge_mask = 0) override;

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

        /**
         * @brief Restore the pre-decode MoE runtime boundary for a prefix hit without a payload.
         *
         * Prefix cache entries may omit the MoE runtime payload when the
         * harvesting runner's temporary execution domain cannot be replayed by
         * the restoring domain.  In that case the cached KV/GDN/MTP tensors are
         * still valid, but any decode-era dynamic placement, transfer slots, or
         * graph-side rebalance scratch owned by this graph builder must not
         * survive into suffix prefill.  The implementation resets decode
         * runtime tables to their empty pre-decode state while preserving their
         * allocated prefill scratch bindings, then clears transient movement
         * helpers so suffix prefill observes the same model-runtime baseline as
         * an uncached split prefill from the same token boundary.
         */
        void resetPrefixCacheRuntimeStateWithoutSnapshot() override;

        /// Append active MoE runtime placement state to prefix-cache fingerprints.
        void appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const override;

        bool capturePrefixCacheRuntimeState(std::vector<uint8_t> &state, void *stream) override;
        bool restorePrefixCacheRuntimeState(const std::vector<uint8_t> &state, void *stream) override;

    private:
        struct ScopedMTPGraphContext
        {
            ScopedMTPGraphContext(Qwen35MoEGraph &graph, int depth_idx);
            ~ScopedMTPGraphContext();

            Qwen35MoEGraph &graph;
            bool previous_active = false;
            int previous_depth_idx = -1;
        };

        /**
         * @brief Role of a graph-side rebalance binding.
         *
         * Decode maintenance bindings are long-lived, domain-wide control lanes
         * used by the async device-resident rebalance maintenance graph.  Prefill
         * LLEP transfer bindings are short-lived, layer-local lanes used while a
         * phase-split prompt is publishing transient expert payloads.  Both can
         * exist for the same device during prefix-cache+MTP runs, so selection
         * must be role-based rather than map-order-based.
         */
        enum class GraphSideRebalanceBindingRole
        {
            DecodeMaintenance,
            PrefillLLEPTransfer
        };

        struct GraphSideRebalanceBinding
        {
            std::string transfer_key;
            std::string workspace_name;
            GraphSideRebalanceBindingRole role =
                GraphSideRebalanceBindingRole::DecodeMaintenance;
            DeviceId device_id = DeviceId::invalid();
            ILocalTPContext *decode_tp_ctx = nullptr;
            ILocalTPContext *maintenance_tp_ctx = nullptr;
            IMoERuntimeTable *moe_runtime_table = nullptr;
            int tp_device_idx = -1;
            DeviceMoERebalanceConfig config;
            DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr;
            uint32_t local_transfer_slot_count = 0;
            DeviceMoERebalanceTransferMode transfer_mode =
                DeviceMoERebalanceTransferMode::ResidentOnly;
            uint64_t collective_payload_slot_bytes = 0;
            uint32_t collective_payload_slot_capacity = 0;
            std::shared_ptr<DeviceMoERebalanceTransferState> transfer_state;
            int producer_layer_idx = -1;
            bool state_sideband_enabled = false;
        };

        IMoERuntimeTable *moeRuntimeTableForDevice(DeviceId device,
                                                   int prefill_token_capacity = 0,
                                                   const std::string &key_suffix = {},
                                                   int num_layers_override = -1,
                                                   bool register_decode_histogram = true);
        /**
         * @brief Build a prefix-restore payload resolver for a MoE runtime table.
         *
         * The portable prefix blob stores only logical placement and stable
         * transfer-slot ids.  Qwen35MoEGraph owns the transfer-slot directories
         * that keep those VRAM payloads alive across a prefix restore with a
         * model-runtime snapshot, so it supplies the resolver that rehydrates a
         * local-compute expert descriptor from the graph-owned directory.
         */
        MoERuntimeTable::LocalPayloadDescriptorResolver
        localPayloadDescriptorResolverForRuntimeTable(const MoERuntimeTable *table) const;
        ILocalTPContext *maintenanceTPContextForDomain(
            const std::string &domain_key,
            ILocalTPContext &decode_tp_ctx);
        /**
         * @brief Return the device-local decode maintenance binding for async
         * graph-side rebalance.
         *
         * The maintenance graph runs after prefill and during decode, so it must
         * consume the domain-wide decode workspace.  It must never attach to a
         * layer-local prefill LLEP workspace, because that workspace may contain
         * an unrelated in-flight prefill transfer wave and would make decode
         * maintenance report a permanent busy window.
         */
        const GraphSideRebalanceBinding *
        findDeviceMoERebalanceMaintenanceBinding(DeviceId device) const;

    protected:
        void registerRuntimeTableHistogramSyncIfNeeded(
            const std::string &key,
            IMoERuntimeTable *table,
            bool register_decode_histogram);

    private:
        std::unordered_map<std::string, std::unique_ptr<MoERuntimeTable>> moe_runtime_tables_;
        std::unordered_map<std::string, std::shared_ptr<DeviceMoETransferSlotDirectory>> moe_transfer_slot_directories_;
        std::unordered_map<std::string, std::shared_ptr<DeviceMoERebalanceTransferState>> moe_rebalance_transfer_states_;
        std::unordered_map<std::string, GraphSideRebalanceBinding> moe_graph_rebalance_bindings_;
        std::unordered_map<std::string, std::shared_ptr<ILocalTPContext>> moe_maintenance_tp_contexts_;
        std::unordered_set<std::string> moe_runtime_histogram_sync_keys_;
        bool mtp_graph_context_active_ = false;
        int mtp_graph_depth_idx_ = -1;
    };

} // namespace llaminar2
