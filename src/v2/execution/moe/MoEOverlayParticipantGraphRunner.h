/**
 * @file MoEOverlayParticipantGraphRunner.h
 * @brief Expert-only graph runner for one heterogeneous MoE overlay rank.
 *
 * A continuation rank owns token embedding, attention, routing, shared-expert
 * work, and logits.  Every other rank owns only the routed experts assigned to
 * its rank-local participants. This runner materializes that smaller authority as a
 * declarative graph: for every layer and target participant it enters the same
 * sparse MPI dispatch boundary as the continuation graph, optionally executes
 * the locally owned expert stage, and enters the matching return boundary.
 *
 * No token or activation tensor is carried by the public runner API.  Hidden
 * rows and route records arrive exclusively through the sparse collective, so
 * remote ranks never construct a dense model graph or a host shadow of root
 * execution state.
 */

#pragma once

#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "loaders/WeightPlan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class ComputeGraph;
    class ModelContext;
    class MoELocalExpertSerialBufferArena;
    class PreparedWeightStore;
    class DecodeExpertHistogram;
    class MoEOverlayResidencyAuthority;
    class MoEOverlayParticipantResidencyRegistry;
    class MoEOverlayNodeLocalDeviceControllerFabric;
    class IMoEOverlayRankBatchTransport;
    class MoEOverlayRankBatchTransportRegistry;
    class TensorBase;
    class WorkspaceAllocator;
    struct MoERoutedExpertPlacementPlan;

    /**
     * @brief Durable placement lifecycle represented by participant graphs.
     *
     * Current-batch LLEP is intentionally orthogonal: it may alter one request's
     * assignment without publishing or retiring a durable placement bank.
     */
    enum class MoEOverlayDurableMaintenancePolicy : std::uint8_t
    {
        Immutable = 0, ///< Static/observe residency; no durable bank retirement.
        DynamicPlacement = 1, ///< Dynamic promotion/rebalance may retire a bank.
    };

    /**
     * @brief Construction contract for one rank-local expert graph.
     *
     * Device and participant membership are derived from the resolved
     * placement plan and MPI rank. Callers cannot provide a second, divergent
     * device list.
     */
    struct MoEOverlayParticipantGraphRunnerConfig
    {
        /** @brief Immutable model/weight authority for this rank-local graph. */
        std::shared_ptr<ModelContext> model_context;
        /** @brief World used exclusively by the matched sparse-collective protocol. */
        std::shared_ptr<IMPIContext> mpi_context;
        /** @brief Authenticated whole-expert ownership and tier placement plan. */
        std::shared_ptr<MoERoutedExpertPlacementPlan> placement_plan;
        /** @brief Same RCU residency authority used by the continuation graph. */
        std::shared_ptr<MoEOverlayResidencyAuthority> residency_authority;
        /** @brief Exact prepared-engine banks for endpoints hosted by this rank. */
        std::shared_ptr<MoEOverlayParticipantResidencyRegistry>
            participant_residency;
        /** @brief Shared route-histogram lifetime for dynamic residency. */
        std::shared_ptr<DecodeExpertHistogram> decode_histogram;
        /**
         * @brief Process-local aliases into the topology-wide GPU controller.
         *
         * Every mapped GPU follower binds its runtime epoch admission directly
         * to this fabric. A Dynamic runner must never invent an independent
         * host-owned placement generation for remote participants.
         */
        std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric>
            device_controller_fabric;
        /**
         * @brief Exact durable placement lifecycle lowered into this graph.
         *
         * Dynamic placement enables both service telemetry and the captured
         * post-inference retirement-readiness receipt. Immutable residency owns
         * neither. Keeping one typed policy prevents those two pieces of the
         * same lifecycle from diverging behind independent booleans.
         */
        MoEOverlayDurableMaintenancePolicy durable_maintenance_policy =
            MoEOverlayDurableMaintenancePolicy::Immutable;
        /**
         * @brief Setup-owned node-local channels after bilateral NUMA first-touch.
         *
         * A same-node follower requires its exact channel from this registry;
         * constructing one while preparing weights would serialize the source
         * and target ranks and make page placement timing-dependent.
         */
        std::shared_ptr<MoEOverlayRankBatchTransportRegistry>
            rank_batch_transport_registry;
        /** @brief Full request/KV context limit; this is not a graph scratch allocation size. */
        int max_seq_len = 0;
        /**
         * @brief Planner-admitted maximum live rows for one remote sparse graph.
         *
         * The participant runner shares one immutable compact-route arena per
         * local participant across its serial decode and prefill graph cache.
         * This is the resolved segment/MTP envelope, not the full KV context.
         * The arena contains separately coherent power-of-two capacity
         * families through this value. A larger prefill request must arrive
         * through the continuation graph's segmented schedule and must never
         * trigger a hot-path resize.
         */
        int max_graph_activation_rows = 0;
        /**
         * @brief Exact physical prefill rows retained by the continuation rank.
         *
         * This is the immutable bucket manifest published by the distributed
         * prefill schedule contract. A remote GPU follower must materialize
         * the same shapes because packet descriptors authenticate physical
         * rows, and a native graph embeds the matching tensor geometry. The
         * participant runner may add decode/MTP shapes below this ladder, but
         * it must never invent or omit a prefill shape independently.
         */
        std::vector<int> prefill_graph_row_shapes;
        /**
         * @brief Largest decode-family row count, including grouped MTP verification.
         *
         * The value is planned from the request-batch and maximum dynamic MTP
         * depth. It creates a small independently coherent compact-buffer
         * family beside the maximum prefill family; it does not relax the
         * overall graph admission above.
         */
        int max_decode_activation_rows = 1;
        /** @brief Whether setup must retain the model's routed MTP sidecar graph. */
        bool mtp_enabled = false;
        /** @brief Largest speculative draft depth admitted by the root controller. */
        int max_mtp_draft_depth = 0;
        /** @brief Largest request batch represented by one retained transaction. */
        int max_request_count = 1;
    };

    /**
     * @brief Build an expert-only immutable weight plan for one participant.
     *
     * Each requirement is an explicit expert-axis selection.  Materialization
     * therefore faults only the GGUF pages assigned to @p participant and
     * preserves the logical-to-physical expert-slot map in WeightSliceSpec.
     *
     * @param model_context Metadata and loader authority for the real model.
     * @param execution_plan Rank/domain ownership resolved from the topology.
     * @param owner_map Whole-expert ownership for every model layer.
     * @param participant Local participant whose weights are requested.
     * @return A typed plan containing only gate/up/down routed-expert parents.
     */
    WeightPlan buildMoEOverlayParticipantWeightPlan(
        const ModelContext &model_context,
        const MoEExpertOverlayExecutionPlan &execution_plan,
        const MoEExpertOwnerMap &owner_map,
        const MoEExpertOwnerParticipant &participant);

    /**
     * @brief Execute all rank-local sparse expert endpoints without a dense model.
     *
     * One fixed-capacity graph serves decode, full prefill buckets, and short
     * tails. This is intentionally independent of the continuation GPU's
     * physical capture bucket: sparse packets carry their own live-row count,
     * so a 16-row participant graph can service 9 rows without executing or
     * transporting padding. Graph topology and packet addresses are therefore
     * materialized once before inference rather than cached on first use for
     * every possible tail length.
     */
    class MoEOverlayParticipantGraphRunner final
        : public IInferenceRunner,
          public IMoEOverlayInferenceTransactionExecutor
    {
    public:
        /**
     * @brief Construct and fully prepare every real-weight local participant.
         * @throws std::invalid_argument for an invalid rank/device topology.
         * @throws std::runtime_error when exact expert preparation fails.
         */
        explicit MoEOverlayParticipantGraphRunner(
            MoEOverlayParticipantGraphRunnerConfig config);

        ~MoEOverlayParticipantGraphRunner() override;

        /** @brief Execute one matched sparse-collective step. */
        bool forward(const int *tokens, int seq_len) override;
        /**
         * @brief Execute a prompt transaction with explicit prefill semantics.
         *
         * Row count is not a phase discriminator: a one-token prompt and the
         * final one-row chunk of a longer prompt must carry `PrefillChunk` in
         * the sparse wire key just like the continuation graph. This override
         * therefore bypasses the generic forward() shape heuristic.
         */
        bool forwardPrefill(const int *tokens, int seq_len) override;
        /** @copydoc IInferenceRunner::servingGraphPreparationKind */
        ServingGraphPreparationKind
        servingGraphPreparationKind() const noexcept override;
        /**
         * @brief Seal every mapped follower endpoint after transfer epochs bind.
         *
         * Graph construction publishes prepared residency banks, but native
         * capture is deliberately deferred until the physical migration fabric
         * has installed each GPU's topology-accounted progress epoch. This
         * setup-only phase embeds that epoch as a cache-owned root fork/terminal
         * join in every admitted row and MTP family before any ticket can run.
         *
         * @param plan Frozen distributed prefill/decode graph inventory.
         * @return True only when every mapped GPU executable is resident and no
         *         standalone progress submission remains necessary.
         */
        bool materializeServingGraphFamilyWithoutLaunch(
            const ServingGraphFamilyMaterializationPlan &plan) override;
        /** @copydoc IInferenceRunner::installMoEOverlayTransferProgressEpoch */
        bool installMoEOverlayTransferProgressEpoch(
            std::shared_ptr<MappedTransferProgressEpoch> epoch) override;
        /**
         * @brief Report support for the root-published bounded prefill schedule.
         *
         * The remote graph executes only the live sparse rows from each
         * captured continuation bucket.  Its immutable arena capacity is
         * nevertheless checked before the root may select this runner for a
         * segmented request.
         */
        bool supportsPrefillChunkSchedule(int seq_len) const override;
        /**
         * @brief Execute every live-row chunk from the shared prefill contract.
         *
         * The continuation graph may capture a padded physical bucket, while
         * this expert-only graph intentionally receives only its real prefix
         * through the sparse collective.  Executing each @c real_count in the
         * same order preserves collective keys, KV position, and the serial
         * compact-buffer ownership contract without materializing padding.
         */
        bool forwardPrefillChunkSchedule(
            const int *tokens,
            int seq_len,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution) override;
        /**
         * @brief Bind rank-local calibration timing to retained graph groups.
         *
         * A remote expert rank does not run the continuation RankOrchestrator,
         * so its authenticated follower transaction is the only truthful
         * rank-wide inference boundary. Direct test forwards use the same
         * owner through @ref executeAtLogicalStep.
         */
        bool setMoEOverlayInferenceInterferenceProbe(
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe)
            override;
        /**
         * @brief Install the root-published generation for sparse wire keys.
         *
         * Every cached graph variant reads this value immediately before its
         * manual dispatch boundary executes. It is intentionally retained by
         * the runner rather than by a graph object so building a new tail-row
         * graph cannot reset distributed identity.
         */
        bool setMoEOverlayCollectiveRequestGeneration(
            uint64_t generation_id) override;
        /**
         * @brief Execute one ticket-selected retained participant graph.
         *
         * This entry point is the production heterogeneous-MTP authority. It
         * consumes only immutable scheduling geometry and never advances the
         * legacy token cursor, samples, or mutates continuation KV state.
         */
        bool executeMoEOverlayInferenceTransaction(
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error = nullptr) override;
        /** @return nullptr because only the continuation graph owns logits. */
        const float *logits() const override;
        /** @return zero because this participant owns no vocabulary projection. */
        int vocab_size() const override;
        /** @brief Reset request position while retaining prepared graphs/weights. */
        void clear_cache() override;
        /** @return Logical position consumed by matched forward calls. */
        int get_position() const override;
        /** @return GRAPH; all participant operations are graph stages. */
        ExecutionPath executionPath() const override;
        /** @return Stable model architecture string. */
        const char *architecture() const override;

        /** @return Stable global participant ids owned by this MPI rank. */
        std::vector<int> participantIds() const;
        /** @return Exact local compute devices, sorted and deduplicated. */
        std::vector<DeviceId> localDevices() const;
        /** @return One after successful fixed-capacity setup, otherwise zero. */
        size_t cachedGraphCount() const noexcept;
        /** @return Fingerprint and generation for this retained graph family. */
        const MoEOverlayInferenceTopologyIdentity &
        inferenceTransactionTopologyIdentity() const noexcept
        {
            return transaction_topology_identity_;
        }
        /** @return Fixed follower protocol bounds derived during setup. */
        MoEOverlayInferenceTransactionProtocol::Config
        inferenceTransactionProtocolConfig() const;
        /** @return One live runtime/epoch source for every local GPU participant. */
        std::vector<MoEOverlayDeviceControllerRuntimeBinding>
        moeOverlayDeviceControllerRuntimeBindings() const override;

    private:
        /**
         * @brief Semantic request phase for one sparse participant transaction.
         *
         * The public expert-runner interface predates typed forward inputs, so
         * this small private enum transports the one fact needed for accurate
         * PerfStats attribution without making the remote sparse graph depend
         * on dense-model ForwardInput ownership.
         */
        enum class SparseTransactionPhase : uint8_t
        {
            Prefill, ///< Prompt-ingestion chunk, including a one-row final tail.
            Decode,  ///< Autoregressive continuation row or verifier transaction.
            MTPDraft, ///< One execution of the retained NextN sidecar graph.
            GroupedVerifier, ///< Main graph over draft-depth-plus-one decode rows.
        };

        /** @brief Immutable graph topology selected during model setup. */
        enum class ParticipantGraphFamilyRole : uint8_t
        {
            Main, ///< Ordinary decoder layers; serves prefill/decode/verifier.
            MTPDraft, ///< One learned NextN sidecar weight graph.
        };

        /**
         * @brief Complete setup-time recipe for one retained participant graph.
         *
         * `source_layers` are GGUF/runtime layer identities and therefore also
         * the sparse wire layer identities. They are deliberately distinct
         * from the speculative sidecar ordinal in a ticket: Qwen reuses its
         * one learned NextN layer for every drafted token.
         */
        struct ParticipantGraphBuildSpec
        {
            ParticipantGraphFamilyRole role =
                ParticipantGraphFamilyRole::Main;
            std::vector<int> source_layers;
            int mtp_graph_depth = -1;
            int row_capacity = 0;
        };

        /**
         * @brief Own one graph and the allocations whose addresses its stages use.
         */
        struct CachedParticipantGraph;
        /**
         * @brief Model-lifetime runtime, epoch, and event authority for one GPU.
         *
         * The definition remains private to the implementation so callers can
         * neither mutate placement banks nor record a boundary on the wrong
         * stream.
         */
        struct ParticipantGpuRuntime;

        /** @brief Validate metadata and resolve every rank-local participant. */
        void resolveTopology();
        /** @brief Resolve disjoint main and routed-MTP source-layer families. */
        void resolveGraphFamilies();
        /**
         * @brief Allocate one immutable compact-route arena for each owned participant.
         *
         * All graph-cache entries are serial through this runner's single
         * inference stream, so they can retain the same participant-owned
         * tensors.  Different participant ids always receive distinct arenas,
         * including CPU endpoints that share one backend identity.
         */
        void createSerialCompactBufferArenas();
        /** @brief Materialize, pack, and register only rank-local expert weights. */
        void prepareParticipantWeights();
        /** @brief Create exact CPU/GPU contexts for every local endpoint. */
        void createDeviceContexts();
        /**
         * @brief Allocate every participant-local GPU placement runtime.
         *
         * GPU execution state remains device-owned under both authority
         * regimes. A host-resident heterogeneous policy publishes inactive
         * banks through the retained host-authority publisher, while an
         * all-GPU policy additionally binds the same runtime to the mapped
         * device-controller fabric. Graph capture always consumes the one
         * device table created here.
         */
        void createParticipantGpuRuntimes();
        /** @return The exact participant GPU runtime or throw for stale topology. */
        ParticipantGpuRuntime &participantGpuRuntimeForParticipant(
            int participant_id) const;
        /**
         * @brief Return the prebuilt fixed-capacity graph after validating live rows.
         *
         * This method never allocates or constructs graph state. The sparse
         * packet received at execution carries @p logical_rows, while every
         * stage retains the planner-admitted physical capacity.
         */
        CachedParticipantGraph &graphForRows(int logical_rows);
        /** @brief Materialize one setup-declared retained protocol graph. */
        std::unique_ptr<CachedParticipantGraph> buildGraph(
            const ParticipantGraphBuildSpec &spec);
        /**
         * @brief Stamp every sparse boundary in one graph before execution.
         *
         * @param graph Cached participant graph for one exact live-row shape.
         * @param logical_step Absolute request position of this operation.
         * @return False when a graph-native sparse stage lacks a generation.
         */
        bool stampMoEOverlayCollectiveRuntime(
            CachedParticipantGraph &graph,
            uint64_t generation_id,
            uint64_t logical_step,
            SparseTransactionPhase phase,
            int mtp_graph_depth = -1);
        /**
         * @brief Arm and execute one node-local mapped follower epoch.
         *
         * Every physical GPU parent is submitted before the first terminal
         * event is observed. CPU endpoints, when declared, consume only their
         * own shared-page lane at the explicit heterogeneous boundary while
         * those parents execute. The host then fences the GPU endpoint events,
         * waits for both semantic endpoint states, and retires the lease.
         */
        bool executeMappedFollowerTransaction(
            CachedParticipantGraph &graph,
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error);
        /**
         * @brief Execute one sparse graph at its root-published request cursor.
         *
         * @p logical_step is protocol data, not an inference convenience. It
         * must equal this runner's current cursor, and is stamped directly
         * into every sparse boundary before graph execution. This lets a
         * continuation graph and an expert-only graph have unrelated capture
         * histories without inventing different collective keys.  @p phase is
         * supplied by the caller that owns the request transaction; a final
         * one-row prefill chunk must remain prefill rather than being inferred
         * as decode from its row count.
         */
        bool executeAtLogicalStep(
            const int *tokens,
            int seq_len,
            int logical_step,
            SparseTransactionPhase phase,
            int physical_rows = -1);
        /** @brief Release raw selected GGUF tensors after prepared engines own them. */
        void releasePreparedSourceBytes();
        /**
         * @brief Return the immutable compact arena owned by one local participant.
         *
         * Verifies both the global logical id and the exact physical device so
         * a topology/configuration error cannot silently alias two endpoints.
         */
        std::shared_ptr<MoELocalExpertSerialBufferArena>
        serialCompactBufferArenaForParticipant(
            const MoEExpertOwnerParticipant &participant) const;

        MoEOverlayParticipantGraphRunnerConfig config_;
        std::shared_ptr<MoEExpertOverlayRuntimePlan> runtime_plan_;
        std::shared_ptr<MoEExpertOverlayExecutionPlan> execution_plan_;
        std::unique_ptr<MoEExpertOwnerMap> owner_map_;
        std::shared_ptr<MoEOverlayResidencyAuthority> residency_authority_;
        std::shared_ptr<DecodeExpertHistogram> decode_histogram_;
        /** Lock-free timing authority claimed around complete retained graphs. */
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe>
            interference_probe_;
        std::vector<const MoEExpertOwnerParticipant *> local_participants_;
        std::shared_ptr<PreparedWeightStore> prepared_store_;
        std::unique_ptr<FrozenModelWeightSet> frozen_weights_;
        std::unordered_map<DeviceId, std::unique_ptr<IDeviceContext>>
            owned_device_contexts_;
        std::unordered_map<DeviceId, IDeviceContext *> execution_contexts_;
        /**
         * @brief One stable compact tensor family per globally unique local participant.
         *
         * Cached remote graphs are serial by the IInferenceRunner contract.
         * The map deliberately uses participant id rather than DeviceId: two
         * NodeTP CPU sockets can share a backend name while still needing
         * independent route packet ownership.
         */
        std::unordered_map<int, std::shared_ptr<MoELocalExpertSerialBufferArena>>
            serial_compact_buffer_arenas_;
        /**
         * @brief One maximum-shape canonical route bank per mapped GPU follower.
         *
         * Every retained decode, verifier, prefill, and MTP graph for a
         * participant executes serially on the participant worker stream. The
         * graphs may therefore share this immutable-address
         * `[max_rows * top_k, d_model]` bank. Keeping it outside the per-shape
         * compact families avoids multiplying a large route-contribution
         * allocation by the complete prefill bucket ladder. Portable MPI and
         * CPU followers do not allocate this mapped-return resource.
         */
        std::unordered_map<int, std::shared_ptr<TensorBase>>
            participant_canonical_route_buffers_;
        /**
         * @brief One durable device-owned execution runtime per GPU participant.
         *
         * Main, MTP, decode, and prefill graph specializations share these
         * addresses regardless of whether placement policy is authored on the
         * host or device. The map precedes graph members so graphs are
         * destroyed before their embedded runtime pointers and epoch tickets.
         */
        std::unordered_map<int, std::unique_ptr<ParticipantGpuRuntime>>
            participant_gpu_runtimes_;
        /** One exact graph-branch authority for each mapped relay GPU. */
        std::unordered_map<
            DeviceId,
            std::shared_ptr<MappedTransferProgressEpoch>>
            transfer_progress_epochs_;
        /** @brief Main decoder graph retained at the admitted prefill capacity. */
        std::unique_ptr<CachedParticipantGraph> main_graph_;
        /** @brief Learned MTP sidecar graphs indexed by manifest graph depth. */
        std::vector<std::unique_ptr<CachedParticipantGraph>>
            mtp_sidecar_graphs_;
        /** Typed setup state shared by eager CPU and native GPU followers. */
        enum class ServingGraphFamilyLifecycle : std::uint8_t
        {
            Built,  ///< Rank-local graph objects exist but cannot accept tickets.
            Sealed, ///< Every declared endpoint is certified for transaction zero.
        };
        /** Sole authority for request-ticket admission into this graph family. */
        ServingGraphFamilyLifecycle serving_graph_family_lifecycle_ =
            ServingGraphFamilyLifecycle::Built;
        /** @brief Main decoder layer count after excluding trailing NextN blocks. */
        int main_layer_count_ = 0;
        /** @brief Routed NextN GGUF source layer for each retained graph depth. */
        std::vector<int> mtp_source_layers_;
        /** @brief Pointer-independent identity shared with the continuation rank. */
        MoEOverlayInferenceTopologyIdentity transaction_topology_identity_{};
        /** @brief Exact main/MTP retained graph geometry used by every channel. */
        MoEOverlayInferenceGraphFamilyIdentity transaction_graph_family_{};
        /**
         * @brief One rank-batch transport per rank/tier/domain group.
         *
         * Main, verifier, and MTP graph families are orchestrator-serial and must
         * reuse one POSIX mapping and one driver registration rather than mapping
         * the same pages once per graph specialization.
         */
        std::unordered_map<
            std::string,
            std::shared_ptr<IMoEOverlayRankBatchTransport>>
            rank_batch_transports_;
        DeviceGraphExecutor executor_;
        std::string architecture_;
        int position_ = 0;
        /** Root-published request generation; zero means forwarding is forbidden. */
        uint64_t overlay_collective_request_generation_ = 0;
    };

    /**
     * @brief Create a participant runner and convert construction failures to logs.
     * @return A ready runner, or nullptr after a precise diagnostic.
     */
    std::unique_ptr<IInferenceRunner> createMoEOverlayParticipantGraphRunner(
        MoEOverlayParticipantGraphRunnerConfig config);

} // namespace llaminar2
