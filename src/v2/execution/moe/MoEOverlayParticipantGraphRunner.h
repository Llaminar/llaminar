/**
 * @file MoEOverlayParticipantGraphRunner.h
 * @brief Retained expert graph family for original rank or subordinate endpoint ownership.
 *
 * A continuation rank owns token embedding, attention, routing, shared-expert
 * work, and logits.  Every other rank owns only the routed experts assigned to
 * its rank-local participants. A continuation Rank may also own this same
 * smaller expert family on genuine off-continuation GPUs. Those endpoints do
 * not become continuation TP members or an independent inference runner.
 * This runner materializes that authority as a declarative graph: for every layer and target participant it enters the same
 * original sparse dispatch boundary as the continuation graph, optionally executes
 * the locally owned expert stage, and enters the matching return boundary.
 *
 * No token or activation tensor is carried by the public runner API.  Hidden
 * rows and route records arrive exclusively through the sparse collective, so
 * remote ranks never construct a dense model graph or a host shadow of root
 * execution state.
 * CPU serial storage declarations survive admission and runtime projection.
 */

#pragma once
#include <mutex>
#include "execution/local_execution/engine/ForwardGraphTypes.h"

#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "loaders/PreparedWeightAdmission.h"
#include "loaders/WeightPlan.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <exception>
#include <algorithm>
#include <type_traits>

namespace llaminar2
{
    namespace qwen4exp { class PooledImmutableRetainedParticipants; }

    namespace test { struct MoEOverlayParticipantNativeRetirementReplayAccess; struct PooledExpertSourceFailureContractAccess; }
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
    class PhysicalMemoryOwnerReservation;
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

    /** @brief Original owner which submits and retires the expert graph family. */
    enum class MoEOverlayParticipantGraphOwnership : std::uint8_t
    {
        ExpertOnlyRank, ///< Existing remote command/ticket follower lifecycle.
        RankLocalAuxiliary, ///< Subordinate endpoints of the real continuation Rank.
    };

    /**
     * @brief Validate the exact fabric and follower GPU projections of original owners.
     *
     * The local fabric includes continuation DGO endpoints as well as follower
     * endpoints. A rank-local auxiliary runner owns only the non-continuation
     * GPU endpoints; an expert-only rank must contain no continuation GPUs.
     * This cold validation creates no runtime authority or membership ledger.
     * @param plan Original plan naming the continuation domain.
     * @param original_owners Original admitted participant catalog.
     * @param world_rank Exact rank whose native runtimes are being constructed.
     * @param ownership Existing original runner role.
     * @param fabric_local_ids Actual local controller-fabric participant IDs.
     * @param follower_gpu_ids Actual GPU IDs selected for this participant runner.
     * @throws std::invalid_argument For missing, foreign, duplicate or wrong-role IDs.
     */
    void requireMoEOverlayParticipantGPUOwnership(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMap &original_owners,
        int world_rank,
        MoEOverlayParticipantGraphOwnership ownership,
        std::span<const int> fabric_local_ids,
        std::span<const int> follower_gpu_ids);

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
        /** Original rank lifecycle; endpoint membership still comes from placement. */
        MoEOverlayParticipantGraphOwnership ownership =
            MoEOverlayParticipantGraphOwnership::ExpertOnlyRank;
        /// Frozen CPU roots for every admitted endpoint/controller on this rank.
        std::vector<std::shared_ptr<const CPUSerialExpertStorageAdmission>> cpu_serial_expert_storage;
        /**
         * @brief Whether this runner creates or adopts its expert engines.
         *
         * Certified reuse consumes only the terminally sealed participant and
         * domain registry aliases. It must never rematerialize GGUF selections
         * because migrated logical experts may occupy different physical slots.
         */
        PreparedWeightAdmission prepared_weight_admission =
            PreparedWeightAdmission::AllocateCompleteSet;
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
        /// Common capacity named by the exact published prefill bucket policy.
        int prefill_graph_row_capacity = 0;
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
        /** @brief Exact setup-time MTP graph-family retention policy. */
        MoEOverlayMTPGraphFamilyPolicy mtp_graph_family_policy =
            MoEOverlayMTPGraphFamilyPolicy::MainOnly;
        /** @brief Largest speculative draft depth admitted by the root controller. */
        int max_mtp_draft_depth = 0;
        /** @brief Largest request batch represented by one retained transaction. */
        int max_request_count = 1;
        /** Original once-per-declared-domain CPU shard context; never constructed by a stage. */
        std::shared_ptr<IGlobalTPContext> cpu_joint_context;
    };

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
        /** @copydoc IInferenceRunner::admitSparseQueryExecutionProgram */
        bool admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &request) override;
        /** @copydoc IInferenceRunner::endSparseQueryExecutionProgram */
        void endSparseQueryExecutionProgram() noexcept override;

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
        /** @brief No-op: sparse participant graphs own no reusable prefix archive. */
        bool purgePrefixCache() override;
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
        friend struct test::MoEOverlayParticipantNativeRetirementReplayAccess;
        friend struct test::PooledExpertSourceFailureContractAccess;
        /**
         * @brief Publish original source cancellation without replacing an enqueue failure or skipping joins.
         * @param cancellation Same Rank source/ticket callback bound before launch.
         * @param failure_mutex Protects only concurrent exception publication, never held while cancelling.
         * @param failure First cancellation exception retained by the original submission through drain.
         */
        static void cancelOriginalPooledSource(const std::function<void()>& cancellation,
            std::mutex& failure_mutex,std::exception_ptr& failure) noexcept;
        /**
         * @brief Preserve source-cancellation failure after exact native terminal observation.
         * @param native_terminals_joined Existing original submission's native retirement proof.
         * @param failure_mutex Protects its existing sticky cancellation exception.
         * @param failure Original source-cancellation failure, never cleared by native proof.
         * @param drain Existing native drain, invoked only before terminal observation.
         * @param error Optional stable original cancellation diagnostic.
         * @return True only after original native drain and successful source cancellation.
         * @throws Any original native-drain exception; the caller still attempts sibling drains.
         */
        template <class Drain>
        static bool drainOriginalSubmissionCancellation(bool native_terminals_joined,
            std::mutex &failure_mutex, const std::exception_ptr &failure, Drain drain, std::string *error)
        {
            if (!native_terminals_joined)
                return drain(error);
            std::exception_ptr original;
            {
                std::lock_guard lock(failure_mutex);
                original = failure;
            }
            if (!original)
                return true;
            try
            {
                std::rethrow_exception(original);
            }
            catch (const std::exception &source_error)
            {
                if (error)
                    *error = std::string("Original pooled source cancellation failed: ") + source_error.what();
            }
            catch (...)
            {
                if (error)
                    *error = "Original pooled source cancellation failed with a non-standard exception";
            }
            return false;
        }
        friend class RankOrchestrator;
        /** @brief Bind actual source cache owners after original continuation setup and before follower capture. */
        void bindOriginalPooledSourceFamily(
            std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> original,
            std::vector<RetainedForwardSourceDefinition> definitions);
        /** @brief Close follower native/cache borrows before the original source arenas retire. */
        void retireOriginalPooledStorageAttachments();
        /** @brief Retain the SAME Rank source/ticket cancellation until all original jobs drain at teardown. */
        void bindOriginalPooledSourceCancellation(std::function<void()> cancellation);
        /** @brief Rank clears the request callback only after complete original SDK/native/channel retirement. */
        void clearOriginalPooledSourceCancellation() noexcept { original_pooled_source_cancellation_={}; }
        std::shared_ptr<const std::function<void()>> original_pooled_source_cancellation_; ///< Captures service/ticket/native abort only, never Rank or this runner.
        std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> original_pooled_participants_;
        std::vector<RetainedForwardSourceDefinition> original_pooled_source_family_;
        friend class OrchestrationRunner; // Original cold coordinator construction owns the sole binding.
        /**
         * @brief Bind the existing coordinator publisher to original same-rank endpoints.
         * @param original Fully prepared auxiliary runner retained through all native joins.
         * @return Existing publisher interface; no MPI self-channel or progress thread.
         * @throws std::invalid_argument For an ordinary follower or foreign endpoint topology.
         */
        static std::shared_ptr<IMoEOverlayInferenceTransactionPublisher>
        createRankLocalInferencePublisher(std::shared_ptr<MoEOverlayParticipantGraphRunner> original);

        /**
         * @brief Authenticate the complete original rank registry before selecting a role subset.
         * @param owners Complete immutable owner map used by this model instance.
         * @param registry Original shared prepared-bank registry, never a cloned subset.
         * @param world_rank Actual process rank owning these endpoint banks.
         * @param epoch Exact original residency snapshot epoch.
         * @throws std::invalid_argument For stale epochs or missing/foreign rank participants.
         */
        template <typename Owners, typename Registry>
        static void requireOriginalRankRegistry(
            const Owners &owners, const Registry &registry, int world_rank, std::uint64_t epoch)
        {
            std::vector<int> original;
            for (const auto &participant : owners.participants())
                if (participant.world_rank_known && participant.world_rank == world_rank)
                    original.push_back(participant.participant_id);
            std::sort(original.begin(), original.end());
            if (world_rank < 0 || epoch == 0u || registry.initialEpoch() != epoch ||
                registry.localParticipantIds() != original)
                throw std::invalid_argument(
                    "MoE overlay participant registry differs from the original full rank topology or epoch");
        }
        /**
         * @brief Join every original endpoint even after an earlier native proof fails.
         * @param terminals Original submitted endpoint/ticket records, in stable order.
         * @param wait Exact existing terminal-event observer for one record.
         * @param abort Nonthrowing original-lane cancellation, preserving first errors.
         * @param error First native proof failure, never a semantic-success receipt.
         * @return True only if every original native observation succeeded.
         *
         * Callers may publish physical completion receipts only after true. No
         * semantic success, source retirement, or reuse is inferred here.
         */
        template <typename Terminals, typename Wait, typename Abort>
        static bool joinAllOriginalNativeTerminals(
            Terminals &terminals, Wait wait, Abort abort, std::string *error)
        {
            static_assert(std::is_nothrow_invocable_v<Abort>);
            std::exception_ptr first;
            for (auto &terminal : terminals)
            {
                try { wait(terminal); }
                catch (...)
                {
                    if (!first)
                        first = std::current_exception();
                    abort();
                }
            }
            if (!first)
                return true;
            try { std::rethrow_exception(first); }
            catch (const std::exception &failure)
            {
                if (error)
                    *error = failure.what();
            }
            catch (...)
            {
                if (error)
                    *error = "Unknown native expert endpoint retirement failure";
            }
            return false;
        }
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
         * @param generation_id Root-authoritative request generation.
         * @param logical_step Absolute request position of this operation.
         * @param phase Mathematical sparse-transaction phase.
         * @param mtp_graph_depth MTP graph namespace depth, or -1 for main execution.
         * @param placement_epoch Exact transaction residency epoch, or zero
         *        for a process-local operation without sequence authority.
         * @return False when a graph-native sparse stage lacks a generation.
         */
        bool stampMoEOverlayCollectiveRuntime(
            CachedParticipantGraph &graph,
            uint64_t generation_id,
            uint64_t logical_step,
            SparseTransactionPhase phase,
            int mtp_graph_depth = -1,
            uint64_t placement_epoch = 0);
        /**
         * @brief Fixed metadata lifecycle of the original same-rank publisher.
         *
         * Native submission and retirement remain on the original Runner. This
         * helper composes its existing protocol, owns no transport, and cannot
         * certify native work. Typed callbacks run at the actual submit/join
         * boundary; only their successful return permits protocol advancement.
         */
        class RankLocalPublicationState
        {
        public:
            /**
             * @brief Allocate publication handles for the original fixed protocol ring during cold setup.
             * @param config Original topology, ring capacity and admitted geometry.
             */
            explicit RankLocalPublicationState(MoEOverlayInferenceTransactionProtocol::Config config)
                : protocol_(std::move(config)), active_(protocol_.slotCount())
            {
            }
            /** @brief Open only after the prior original command retired successfully. */
            bool begin(const MoEOverlayInferenceCommandIdentity &command, std::string *error)
            {
                if (active() || failed_)
                    return reject(error, "Original local expert command is active or failed");
                if (!protocol_.beginCommand(command, error))
                    return false;
                command_ = command;
                return true;
            }
            /**
             * @brief Authenticate a ticket, retain its slot, then submit its original endpoints.
             * @param descriptor Original continuation graph descriptor.
             * @param submit Original runner's exact family validation and all-submit operation.
             * @return Handle retained even after partial failure until abort drains the original work.
             */
            template <class Submit>
            MoEOverlayPublishedInferenceTransaction publish(
                const MoEOverlayInferenceExecutionDescriptor &descriptor, Submit submit)
            {
                MoEOverlayPublishedInferenceTransaction out;
                if (failed_ || !command_.valid())
                {
                    out.error = "Local publisher has no reusable original command slot";
                    return out;
                }
                try
                {
                    out.ticket = makeMoEOverlayInferenceExecutionTicket(topology(), command_,
                        protocol_.nextTransactionOrdinal(), descriptor.logical_step_id,
                        descriptor.placement_epoch, descriptor.graph_role, descriptor.request_count,
                        descriptor.logical_rows_per_request, descriptor.physical_rows_per_request,
                        descriptor.draft_depth, descriptor.sidecar_depth,
                        descriptor.prefill_schedule_workload, descriptor.retired_decode_progress_tokens);
                    const auto admission = protocol_.accept(out.ticket);
                    if (!admission.accepted())
                    {
                        out.error = admission.error;
                        failed_ = true;
                        return out;
                    }
                    out.slot_index = admission.slot_index;
                    active_.at(out.slot_index) = out;
                    if (!protocol_.markSubmitted(out.slot_index, out.ticket, &out.error) ||
                        !submit(out.ticket, &out.error))
                    {
                        failed_ = true;
                        return out;
                    }
                    out.ok = true;
                    active_.at(out.slot_index) = out;
                }
                catch (const std::exception &exception)
                {
                    failed_ = true;
                    out.error = exception.what();
                }
                return out;
            }
            /**
             * @brief Admit the actual follower publication and collect every host SDK enqueue before continuation.
             * @param descriptor Original continuation graph descriptor.
             * @param submit Actual ticket/family selection and complete original all-submit operation.
             * @param collect Actual original host enqueue-future collector; no GPU terminal wait belongs here.
             * @return Continuation permission only after every SDK enqueue succeeds; partial failure keeps its original slot for drain.
             */
            template<class Submit,class Collect>
            MoEOverlayPublishedInferenceTransaction publishOriginalFollowers(
                const MoEOverlayInferenceExecutionDescriptor& descriptor,Submit submit,Collect collect)
            {
                return publish(descriptor,[&](const auto& ticket,std::string* error) {
                    return submit(ticket,error) && collect(error);
                });
            }
            /** @brief Join the exact original handle before advancing its existing protocol slot. */
            template <class Join>
            bool retire(const MoEOverlayPublishedInferenceTransaction &handle, Join join, std::string *error)
            {
                const auto *original = activeHandle(handle.slot_index);
                if (!original || !handle.ok || !original->ok || handle.ticket != original->ticket ||
                    handle.slot_index != original->slot_index)
                    return reject(error, "Local retirement lost its exact original publication");
                if (failed_)
                    return reject(error, "Original local expert command has failed");
                try
                {
                    if (!join(error))
                    {
                        failed_ = true;
                        return false;
                    }
                }
                catch (const std::exception &exception)
                {
                    failed_ = true;
                    return reject(error, exception.what());
                }
                if (!protocol_.markReturnReady(handle.slot_index, handle.ticket, error) ||
                    !protocol_.retire(handle.slot_index, handle.ticket, error))
                {
                    failed_ = true;
                    return false;
                }
                active_[handle.slot_index].reset();
                return true;
            }
            /** @brief Successful completion is impossible while any slot or failure remains. */
            bool complete(std::uint64_t epoch, std::uint64_t progress, std::string *error)
            {
                if (active() || failed_)
                    return reject(error, "Original local expert work is not successfully retired");
                return terminal(MoEOverlayInferenceTransactionAction::Complete, epoch, progress, 0, error);
            }
            /**
             * @brief Drain every partial original submission before accepting absorbing abort.
             * @param drain Original cancellation plus all-native-proof operation, not semantic success.
             */
            template <class Drain>
            bool abort(std::uint64_t epoch, int code, Drain drain, std::string *error)
            {
                if (!command_.valid() || epoch < protocol_.currentPlacementEpoch() || code <= 0)
                    return reject(error, "Local abort lost its original command epoch or failure code");
                failed_ = true;
                if (active())
                {
                    try
                    {
                        if (!drain(error))
                            return false;
                    }
                    catch (const std::exception &exception)
                    {
                        return reject(error, exception.what());
                    }
                    for (auto &handle : active_)
                        handle.reset();
                }
                if (protocol_.state() == MoEOverlayInferenceProtocolState::Failed)
                    return true;
                return terminal(MoEOverlayInferenceTransactionAction::Abort, epoch, 0u, code, error);
            }
            /** @return Whether original work still needs its exact native retirement. */
            bool active() const noexcept
            {
                return std::any_of(active_.begin(), active_.end(), [](const auto &handle)
                {
                    return handle.has_value();
                });
            }
            /** @return Latest original active ticket for absorbing abort, never mutable device state. */
            const MoEOverlayInferenceTransactionTicket *activeTicket() const noexcept
            {
                const MoEOverlayInferenceTransactionTicket *latest = nullptr;
                for (const auto &handle : active_)
                    if (handle && (!latest || handle->ticket.transaction_ordinal > latest->transaction_ordinal))
                        latest = &handle->ticket;
                return latest;
            }
            /** @return Original admitted protocol capacity, never a new concurrency budget. */
            std::size_t slotCount() const noexcept
            {
                return protocol_.slotCount();
            }
            /**
             * @param slot Index in the original protocol ring.
             * @return Exact retained publication, or null when that slot is reusable.
             */
            const MoEOverlayPublishedInferenceTransaction *activeHandle(std::size_t slot) const noexcept
            {
                return slot < active_.size() && active_[slot] ? &*active_[slot] : nullptr;
            }
            /**
             * @param ticket Immutable original execution identity.
             * @return Existing slot owning this ticket; no second ring mapping is derived.
             */
            std::optional<std::size_t> activeSlot(const MoEOverlayInferenceTransactionTicket &ticket) const noexcept
            {
                for (std::size_t index = 0; index < active_.size(); ++index)
                    if (active_[index] && active_[index]->ticket == ticket)
                        return index;
                return std::nullopt;
            }
            /** @return The same original topology retained by the protocol ring. */
            const MoEOverlayInferenceTopologyIdentity &topology() const noexcept
            {
                return protocol_.topologyIdentity();
            }
        private:
            /** @brief Submit a terminal through the original ticket validator, without transport. */
            bool terminal(MoEOverlayInferenceTransactionAction action, std::uint64_t epoch,
                          std::uint64_t progress, int code, std::string *error)
            {
                try
                {
                    const auto ticket = makeMoEOverlayInferenceTerminalTicket(topology(), command_,
                        protocol_.nextTransactionOrdinal(), epoch, action, code, progress);
                    const auto admission = protocol_.accept(ticket);
                    const bool expected = action == MoEOverlayInferenceTransactionAction::Complete ?
                        admission.status == MoEOverlayInferenceAdmissionStatus::Complete :
                        admission.status == MoEOverlayInferenceAdmissionStatus::Aborted;
                    if (!expected)
                        return reject(error, admission.error);
                    return true;
                }
                catch (const std::exception &exception)
                {
                    return reject(error, exception.what());
                }
            }
            /** @brief Publish a stable diagnostic without altering original protocol state. */
            static bool reject(std::string *error, const std::string &message)
            {
                if (error)
                    *error = message;
                return false;
            }
            MoEOverlayInferenceTransactionProtocol protocol_;
            MoEOverlayInferenceCommandIdentity command_{};
            /** Native publication handles indexed by the sole original protocol ring. */
            std::vector<std::optional<MoEOverlayPublishedInferenceTransaction>> active_;
            bool failed_ = false;
        };
        class RankLocalInferencePublisher;
        struct RetainedGraphSelection;
        /**
         * @brief Select the original main/sidecar family from exact ticket geometry.
         * @param ticket Existing validated scheduler transaction.
         * @param error Precise unsupported role/geometry diagnostic.
         * @return Original retained family, or empty without launching any work.
         */
        std::optional<RetainedGraphSelection> selectRetainedGraph(
            const MoEOverlayInferenceTransactionTicket &ticket, std::string *error);
        /**
         * @brief Publish the same immutable successful transaction evidence for either original owner role.
         * @param ticket Exact original ticket whose native work and returns have retired.
         * @param selection Original role/family/geometry selected before submission.
         */
        void recordSuccessfulRetainedTransaction(const MoEOverlayInferenceTransactionTicket &ticket,
                                                const RetainedGraphSelection &selection);
        struct MappedFollowerTransaction;
        struct MappedFollowerEnqueueJoin;
        /**
         * @brief Validate the original submission before clearing any queued-worker metadata.
         * @param enqueues Original queued/joined worker handles; worker-written fields are read only after join.
         * @param boundaries Original physical admission/launch receipts.
         * @param native_proven Whether every exact submitted native terminal was successfully observed.
         * @throws std::logic_error For a pending worker, unproven launch, failed/unretired admission or missing owner.
         */
        template <class Enqueues, class Boundaries>
        static void requireOriginalSubmissionRetired(const Enqueues &enqueues,
            const Boundaries &boundaries, bool native_proven)
        {
            for (const auto &pending : enqueues)
                if (pending.queued && !pending.joined)
                    throw std::logic_error("Cannot replace an original queued worker before its enqueue joined");
            for (const auto &pending : enqueues)
                if (pending.joined && pending.launch_attempted && !native_proven)
                    throw std::logic_error("Cannot replace an attempted original launch without native proof");
            for (const auto &boundary : boundaries)
            {
                if (!boundary.runtime || boundary.generation == 0u)
                    throw std::logic_error("Original expert admission lost its runtime owner or generation");
                const auto &receipt = boundary.runtime->inference_boundary_receipt;
                if (boundary.graph_submitted)
                {
                    // This same original runtime publishes a completed prefix.
                    // Later retired generations cannot revoke this record's
                    // exact native proof or its already completed generation.
                    if (!native_proven || receipt.completedGeneration() < boundary.generation)
                        throw std::logic_error("Cannot replace an unretired original expert submission");
                }
                else if (receipt.submittedGeneration() != boundary.generation - 1u ||
                         receipt.completedGeneration() != boundary.generation - 1u)
                    throw std::logic_error("Cannot replace an original unlaunched admission before rollback");
            }
        }
        /** @brief Cancel exact original lane identities without changing their first error. */
        void cancelMappedFollowerTransaction(MappedFollowerTransaction &submission) noexcept;
        /** @brief Join every bounded original worker enqueue before consuming its native terminal. */
        MappedFollowerEnqueueJoin joinMappedFollowerEnqueues(MappedFollowerTransaction &submission);
        /**
         * @brief Publish original cancellation immediately when an actual worker enqueue fails.
         * @param work Actual original worker-native submission body.
         * @param cancel Nonthrowing original source/follower lane cancellation.
         * @return Actual enqueue success; exceptions retain their original identity after cancellation.
         */
        template <class Work, class Cancel>
        static bool runOriginalEndpointEnqueue(Work work, Cancel cancel)
        {
            static_assert(std::is_nothrow_invocable_v<Cancel>);
            try
            {
                const bool success = work();
                if (!success)
                    cancel();
                return success;
            }
            catch (...)
            {
                cancel();
                throw;
            }
        }
        /** @brief Arm and submit every original native endpoint without awaiting peers. */
        bool submitMappedFollowerTransaction(MappedFollowerTransaction &submission, std::string *error);
        /** @brief Service only the explicit CPU endpoints after native all-submit. */
        bool serviceMappedCPUFollowers(MappedFollowerTransaction &submission, std::string *error);
        /** @brief Join every original terminal and retire the exact activation lease. */
        bool retireMappedFollowerTransaction(MappedFollowerTransaction &submission, std::string *error);
        /**
         * @brief Join a cancelled partial submission without publishing successful service evidence.
         * @param submission Original retained native tickets and admission receipts.
         * @param error First missing/failed native proof, preserving all-peer join attempts.
         * @return True only when every submitted parent and unlaunched rollback is proven.
         */
        bool drainCancelledMappedFollowerTransaction(MappedFollowerTransaction &submission, std::string *error);
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
        std::shared_ptr<const MoEExpertOwnerMap> owner_map_;
        std::shared_ptr<MoEOverlayResidencyAuthority> residency_authority_;
        std::shared_ptr<DecodeExpertHistogram> decode_histogram_;
        /** Lock-free timing authority claimed around complete retained graphs. */
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe>
            interference_probe_;
        std::vector<const MoEExpertOwnerParticipant *> local_participants_;
        /** Frozen expert packet participation; physical controller/storage owners remain separate. */
        std::vector<int> admitted_rank_local_endpoint_ids_;
        std::shared_ptr<PreparedWeightStore> prepared_store_;
        std::unique_ptr<FrozenModelWeightSet> frozen_weights_;
        /**
         * @brief Canonical opaque graph-pool commitments for rank-local GPUs.
         *
         * One reservation per physical device covers every main/MTP row-shape
         * graph declared later in this runner. Individual graph objects cannot
         * own allocator leases because CUDA/HIP expose only shared driver-pool
         * growth, not independently releasable allocation handles.
         */
        std::unordered_map<
            DeviceId,
            std::shared_ptr<PhysicalMemoryOwnerReservation>>
            native_graph_memory_reservations_;
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
        /**
         * @brief One physical serial workspace family for every local device.
         *
         * Main, prefill-shape, grouped-verifier, and learned MTP graphs are
         * retained concurrently but the request protocol permits only one of
         * them to execute on a device at a time.  Their workspace addresses
         * must therefore come from this runner-owned allocator, not from a
         * temporary allocator created by each graph builder invocation.  The
         * latter would materialize the same admitted owner line repeatedly and
         * exhaust the canonical physical-memory ledger during setup.
         *
         * This member precedes the cached graphs so reverse-order destruction
         * retires every graph that embeds its pointers before releasing the
         * backing allocation.
         */
        std::shared_ptr<WorkspaceAllocator>
            serial_graph_family_workspace_allocator_;
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
            Sealed, ///< Every declared endpoint is certified for transaction zero; cold diagnostic aliases may still extend.
            ExecutionAdmitted, ///< Original coordinator installed; source aliases/topology are immutable.
        };
        /** @brief Reject changed source aliases after the existing original execution-admission frontier. */
        static void requireColdSourceAliasRefresh(ServingGraphFamilyLifecycle lifecycle) {
            if (lifecycle == ServingGraphFamilyLifecycle::ExecutionAdmitted)
                throw std::invalid_argument("Expert storage source aliases are immutable after original execution admission");
        }
        /** Sole authority for request-ticket admission into this graph family. */
        ServingGraphFamilyLifecycle serving_graph_family_lifecycle_ =
            ServingGraphFamilyLifecycle::Built;
        /** @brief Main decoder layer count after excluding trailing NextN blocks. */
        int main_layer_count_ = 0;
        /** @brief Model-global placement/runtime slots retained by this family. */
        int routed_layer_capacity_ = 0;
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
