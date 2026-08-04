/**
 * @file Qwen35MoEGraph.h
 * @brief Qwen 3.5 MoE compute graph builder (hybrid GDN + FA + MoE FFN)
 *
 * Extends Qwen35Graph with Mixture-of-Experts FFN blocks.
 * Attention architecture (GDN + FA hybrid) is identical to dense Qwen3.5.
 */

#pragma once

#include "../qwen35/Qwen35Graph.h"
#include "../../execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "../../execution/moe/DeviceMoERebalanceController.h"
#include "../../execution/moe/MoERuntimeTable.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
            DeviceId device,
            void *device_state_publication_stream,
            const int32_t *sequence_lengths_device = nullptr,
            const int32_t *absolute_position_ids_device = nullptr) override;

        ComputeGraph buildDeviceMoERebalanceMaintenanceGraph(
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
        void resetState(void *execution_stream = nullptr) override;

        /**
         * @brief Restore the pre-decode MoE runtime boundary for a prefix hit without a payload.
         *
         * Prefix cache entries may omit the MoE runtime payload when the
         * harvesting runner's temporary execution domain cannot be replayed by
         * the restoring domain.  In that case the cached KV/GDN/MTP tensors are
         * still valid, but any decode-era dynamic placement, transfer slots, or
         * graph-side rebalance scratch owned by this graph builder must not
         * survive into suffix prefill. The implementation restores the
         * immutable model-lifetime placement template at the same stable device
         * addresses used by captured graphs, then clears transient movement
         * publications so suffix prefill and grouped decode observe the same
         * canonical ownership as an uncached request.
         */
        void resetPrefixCacheRuntimeStateWithoutSnapshot(
            void *execution_stream = nullptr) override;

        /// Append active MoE runtime placement state to prefix-cache fingerprints.
        void appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const override;

        bool capturePrefixCacheRuntimeState(std::vector<uint8_t> &state, void *stream) override;
        PrefixCacheRuntimeRestoreResult restorePrefixCacheRuntimeState(
            const std::vector<uint8_t> &state,
            void *stream) override;
        bool prefixCacheRuntimeStateRequiresDeviceRehydration() const override
        {
            return prefix_runtime_device_rehydration_pending_;
        }
        void completePrefixCacheRuntimeStateDeviceRehydration() override;

        /**
         * @brief Describe device-retained main-graph layer checkpoints.
         *
         * Checkpoints exist only when
         * `LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS=1` was set before graph
         * construction. Each tensor contains one terminal activation row
         * copied by a graph-captured row-select stage. Normal execution never
         * copies checkpoint data to the host; the mirrored-MTP failure path
         * reads these tensors only after participant tokens disagree.
         */
        struct MirroredLayerCheckpoint
        {
            std::string name;       ///< Stable backend-neutral graph boundary name.
            const TensorBase *tensor = nullptr; ///< One-row device checkpoint.
        };

        /**
         * @brief Return all graph variants whose checkpoint tensors were built.
         *
         * The returned descriptors borrow graph-owned tensors and remain valid
         * for this graph builder's lifetime. Results are sorted by name so
         * participant diagnostics compare identical boundaries by index.
         */
        std::vector<MirroredLayerCheckpoint>
        mirroredLayerCheckpoints() const;

    protected:
        /**
         * @brief Retain every grouped-verifier embedding row on device.
         *
         * This opt-in diagnostic establishes the first data boundary after
         * verifier token publication. It is built only for exact grouped
         * verifier graphs, and every row-copy node is chained before layer zero
         * so a terminal mismatch cannot report stale checkpoint bytes.
         */
        std::string maybeAddEmbeddingDiagnosticCheckpoints(
            ComputeGraph &graph,
            TensorBase *source,
            const std::string &dependency,
            int total_tokens,
            DeviceId device) override;

        /**
         * @brief Retain terminal rows immediately before and after final norm.
         *
         * This override is active only for the opt-in mirrored-layer
         * diagnostic and only for condition-producing main graphs. The
         * returned node is chained into the LM-head dependency, making
         * checkpoint completion an explicit part of graph completion.
         */
        std::string maybeAddFinalNormDiagnosticCheckpoint(
            ComputeGraph &graph,
            const std::string &boundary,
            TensorBase *source,
            const std::string &dependency,
            int total_tokens,
            DeviceId device,
            const int32_t *sequence_lengths_device) override;

        /**
         * @brief Resolve checkpoint row ownership from immutable graph policy.
         *
         * Ordinary multi-row GPU prefill and grouped verification may both have
         * fewer logical rows than their captured physical M and therefore use
         * resident request-length metadata. Scalar and request-condition graphs
         * retain fixed-row ownership. The method throws when a resident-length
         * graph lacks the device owner needed to select its terminal row.
         */
        HiddenStateRowSelectStage::SelectionPolicy
        mirroredCheckpointSelectionPolicy(
            int total_tokens,
            DeviceId device,
            const int32_t *sequence_lengths_device) const;

        /**
         * @brief Bind row ownership to one checkpoint stage parameter block.
         *
         * Keeping policy and pointer assignment in one helper prevents a graph
         * caller from selecting resident-length mode while forgetting to bind
         * the corresponding device owner, or from attaching mutable request
         * metadata to an exact fixed-row graph.
         */
        void configureMirroredCheckpointRowOwnership(
            HiddenStateRowSelectStage::Params &params,
            int total_tokens,
            DeviceId device,
            const int32_t *sequence_lengths_device) const;

        /**
         * @brief Retain one GDN intermediate row in the mirrored checkpoint bank.
         *
         * This hook is called by Qwen35Graph only when constructing the GDN
         * subgraph. It uses the same graph-regime identity and resident row
         * ownership contract as the outer MoE layer checkpoints.
         */
        std::string maybeAddGDNDiagnosticCheckpoint(
            ComputeGraph &graph,
            const std::string &boundary,
            const ITensor *source,
            const std::string &dependency,
            int layer_idx,
            int total_tokens,
            int feature_dim,
            DeviceId device,
            const int32_t *sequence_lengths_device) override;

    private:
        struct ScopedMTPGraphContext
        {
            ScopedMTPGraphContext(
                Qwen35MoEGraph &graph,
                int depth_idx);
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
            /**
             * @brief Domain collective context shared with the enclosing graph.
             *
             * Rebalance work is ordered against forward/MTP work through
             * explicit producer and consumer events.  It must therefore reuse
             * the already initialized NCCL/RCCL domain instead of creating a
             * second communicator whose resources would be allocated lazily
             * after model placement.
             */
            ILocalTPContext *collective_tp_ctx = nullptr;
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
        /**
         * @brief Per-device transient route scratch shared by serial graph roles.
         *
         * Main prefill, grouped verifier, and every MTP sidecar retain separate
         * runtime metadata tables, but the orchestrator orders their execution.
         * Their transient route planners therefore bind one largest-participant
         * arena per device instead of multiplying scratch by layer and depth.
         */
        std::unordered_map<
            std::string,
            std::shared_ptr<DeviceMoESerialRouteScratchArena>>
            moe_serial_route_scratch_arenas_;
        std::unordered_map<std::string, std::shared_ptr<DeviceMoETransferSlotDirectory>> moe_transfer_slot_directories_;
        std::unordered_map<std::string, std::shared_ptr<DeviceMoERebalanceTransferState>> moe_rebalance_transfer_states_;
        std::unordered_map<std::string, GraphSideRebalanceBinding> moe_graph_rebalance_bindings_;
        std::unordered_set<std::string> moe_runtime_histogram_sync_keys_;
        /**
         * @brief One-shot graph regime requested by portable prefix restoration.
         *
         * This is graph-construction policy, not a host mirror of live placement.
         * The actual desired transfers live in each runtime table's persistent
         * device scratch. The flag remains set until a successful dedicated
         * graph executes all layer-local rehydration transactions.
         */
        bool prefix_runtime_device_rehydration_pending_ = false;

        /**
         * @brief Persistent one-row buffers populated by optional graph diagnostics.
         *
         * The key includes graph regime, runtime M, layer, and boundary. Reusing
         * a key across recapture preserves the destination address required by
         * CUDA/HIP graph executables while later launches overwrite the bytes
         * with the newest point-in-time row.
         */
        std::unordered_map<std::string, std::unique_ptr<FP32Tensor>>
            mirrored_layer_checkpoints_;

        bool mtp_graph_context_active_ = false;
        int mtp_graph_depth_idx_ = -1;
    };

} // namespace llaminar2
