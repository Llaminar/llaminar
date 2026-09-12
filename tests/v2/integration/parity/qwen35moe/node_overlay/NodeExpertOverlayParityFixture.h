/**
 * @file NodeExpertOverlayParityFixture.h
 * @brief Declared production fixture shared by both model-size registrations.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#pragma once

#include "NodeExpertOverlayParitySupport.h"
#include "../../ParityPrefillSnapshotEvidence.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{
/** Typed cases own configuration; this fixture owns their proof lifecycle. */
class Qwen35MoENodeExpertOverlayParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoENodeExpertOverlayParityTest>
      , public ::testing::WithParamInterface<ModelParityCase>
{
public:
    /**
     * @brief Resolve forced-branch HF evidence after production residency ends.
     *
     * GoogleTest calls this only after every fixture TearDown has destroyed its
     * runner. Releasing the external prepared-weight cache on every MPI process
     * then creates an explicit memory phase boundary: the sole artifact owner
     * loads Hugging Face once for the union of observed branches, while every
     * other rank waits without retaining a model authority. The final broadcast
     * makes a reference or numerical failure visible to the complete test world.
     */
    static void TearDownTestSuite();

    /** @brief Return one immutable config per exact generated matrix cell. */
    static const TestConfig &generatedConfig();

    /**
     * @brief Return the configuration whose declared backends match this exact cell.
     *
     * The production campaign discovery derives its resource signature from
     * the test name; this return value keeps the test fixture's result labels,
     * reference evidence, and runner configuration aligned with that same
     * declarative topology.
     */
    const TestConfig &getTestConfig() const;

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoENodeExpertOverlayParityTest>;

    /** @return Capacity-resolved immutable placement installed in production. */
    const MoERoutedExpertPlacementPlan &resolvedOverlayPlan() const;

    /**
     * @brief Retain the exact device-owned route epoch consumed by diagnostics.
     *
     * The Hugging Face pack authenticates model tensors, but it cannot name
     * Llaminar's placement-bank selector or final domain schedule. These
     * additional keys bind each compared routed contribution to the acquired
     * overlay epoch. Declaring them through the shared typed snapshot policy
     * makes them part of graph identity before capture; diagnostics never
     * mutate or recapture the serving graph after setup.
     */
    ParityGraphSnapshotPolicy parityGraphSnapshotPolicy(
        ParityForwardPhase phase) const override;

    /** @brief The 122B matrix mathematically compares recursive MTP sidecars. */
    bool requiresMTPSidecarReferenceSnapshots() const override;

    /** @return Maximum recurrent sidecar depth admitted by the 122B matrix. */
    int requiredMTPSidecarReferenceDraftDepth() const override;

    /** @return Whether this process intentionally runs consecutive 122B cells. */
    bool mayReuseQwen122OverlayModelContext() const;

    /**
     * @return Complete topology and policy identity for prepared-weight reuse.
     *
     * The two bits encode residency policy and whole-expert owner order. Every
     * cell in one topology reserves the same maximum retained graph geometry,
     * matching production reuse identity without coupling physical placement
     * to the fixed depth selected for one request. Topology identity prevents
     * a differently sized participant catalogue from reusing those pointers.
     */
    Qwen122OverlayPhysicalIdentity
    qwen122ModelContextPhysicalIdentity() const;

    /** @return Whether this cell may preserve immutable runner topology. */
    bool mayRetainQwen122OverlayRunner() const;

    /** @return Exact immutable identity required by the active runner cell. */
    Qwen122OverlayRunnerIdentity qwen122RunnerIdentity() const;

    /**
     * @brief Probe the bounded cache without transferring runner ownership.
     * @param error Receives an internally inconsistent cache diagnostic.
     * @return True only for one complete exact-identity retained runner.
     */
    bool hasCompatibleQwen122OverlayRunner(std::string *error) const;

    /** @return Retained setup proof only for the exact still-owned runner. */
    ParityRunnerEvidenceLifetime productionParityRunnerEvidenceLifetime() const override;

    /**
     * @brief Transfer an exact yielded runner into this fixture.
     * @return Sole runner ownership, or null on an exact cache miss.
     */
    std::unique_ptr<IOrchestrationRunner>
    takeCompatibleQwen122OverlayRunner();

    /**
     * @brief Destroy an incompatible yielded runner before common setup.
     *
     * Every rank performs this transition before Base::SetUp can clear the
     * process kernel registry. The two barriers ensure no rank constructs the
     * next topology while a peer still owns streams or graph executables from
     * the prior one.
     */
    void retireIncompatibleQwen122OverlayRunner();

    /**
     * @brief Find the prior runner's rank-local prepared-weight certificate.
     *
     * A miss does not synthesize ModelContext configuration from test metadata;
     * only a previously initialized production runner may populate the slot.
     */
    std::optional<ModelContextReuseContract>
    findQwen122OverlayModelContext(
        bool *cache_hit,
        std::string *error) const;

    /**
     * @brief Publish one initialized runner's immutable rank-local authority.
     *
     * The cache never replaces a live slot: doing so could conceal a teardown
     * or identity defect. The contract itself remains the production source of
     * truth for participant topology and prepared-weight compatibility.
     */
    bool publishQwen122OverlayModelContext(
        const ModelContextReuseContract &contract,
        std::string *error) const;

    /** Stable identity and cumulative value for one PerfStats timer. */
    struct ConvergenceTimerRecord
    {
        std::string domain;
        std::string name;
        std::string phase;
        std::string device;
        PerfStatsCollector::Tags tags;
        std::uint64_t count = 0u;
        std::uint64_t total_ns = 0u;

        /** @return Strict canonical ordering excluding cumulative values. */
        bool operator<(const ConvergenceTimerRecord &other) const
        {
            return std::tie(domain, name, phase, device, tags) <
                   std::tie(
                       other.domain,
                       other.name,
                       other.phase,
                       other.device,
                       other.tags);
        }
    };

    /** Canonically keyed cumulative or interval-local timer values. */
    using ConvergenceTimerSnapshot =
        std::map<ConvergenceTimerRecord, ConvergenceTimerRecord>;

    /**
     * @brief Allocation-owning samples for the observed convergence gate.
     *
     * Baseline values are ordinary production intervals claimed before any
     * residency epoch can publish. Converged values use the same authenticated
     * prompt, restored terminal state, sampled decode input, and one-forward
     * decode budget after the required profitable epochs. A sample whose
     * surrounding committed-wave count changes is rejected instead of being
     * attributed to either layout.
     */
    struct ResidencyConvergenceTimings
    {
        /**
         * @brief One complete request measured in one immutable residency epoch.
         *
         * Candidate identity is retained to match the exact same request on
         * both sides of the comparison. Decode vectors preserve their
         * per-request transaction boundary. Timer deltas cover the same request
         * identity, including its ordinary prefix restores and boundary
         * transactions, so diagnostics never compare different routing
         * workloads.
         */
        struct RequestSample
        {
            int prompt_identity = -1;
            std::uint64_t prefill_ns = 0u;
            std::uint64_t prefill_epoch = 0u;
            std::vector<std::uint64_t> decode_ns;
            std::vector<std::uint64_t> decode_epochs;
            std::vector<int32_t> decode_input_tokens;
            ConvergenceTimerSnapshot timer_deltas;
        };

        /** Exact measured initial-epoch requests matched after convergence. */
        std::vector<RequestSample> baseline_candidates;
        /** Exact selected samples measured after the final publication. */
        std::vector<RequestSample> converged_samples;
        std::vector<std::uint64_t> baseline_prefill_ns;
        std::vector<std::uint64_t> baseline_decode_ns;
        std::vector<std::uint64_t> baseline_prefill_epochs;
        std::vector<std::uint64_t> baseline_decode_epochs;
        std::vector<std::uint64_t> converged_prefill_ns;
        std::vector<std::uint64_t> converged_decode_ns;
        std::vector<std::uint64_t> converged_prefill_epochs;
        std::vector<std::uint64_t> converged_decode_epochs;
        /** Exact sampled inputs consumed by every timed decode forward. */
        std::vector<int32_t> baseline_decode_input_tokens;
    };

    /** Typed side of the immutable-epoch inference comparison. */
    enum class ResidencyTimingCohort : std::uint8_t
    {
        InitialEpoch,
        ConvergedEpoch,
    };

    /**
     * @brief Disjoint first-block identities for production economy traffic.
     *
     * Production parity deliberately uses one-token prefix-cache blocks so it
     * can prove an exact partial restore cheaply.  The stationary before/after
     * workload must therefore exclude every leading token in the bounded
     * service-certification corpus.  A probabilistic collision silently turns
     * a full-compute sample into a partial restore.
     */
    enum class EconomyPromptNamespace : std::uint8_t
    {
        ServiceCertification,
        StationaryConvergence,
    };

    /** @return Monotonic elapsed nanoseconds, clamped away from zero. */
    static std::uint64_t elapsedNanoseconds(
        std::chrono::steady_clock::time_point start) noexcept;

    /**
     * @brief Snapshot cumulative timers needed to attribute an A/B cohort.
     *
     * Snapshot construction occurs outside every measured interval. Exact tags
     * retain layer, sparse endpoint, replay segment, and route identity so the
     * resulting CSV can locate a regression without adding clocks to the live
     * graph.
     */
    static ConvergenceTimerSnapshot convergenceTimerSnapshot();

    /**
     * @brief Subtract two cumulative timer views without losing exact tags.
     *
     * @param before Cumulative snapshot immediately before ordinary requests.
     * @param after Cumulative snapshot immediately after ordinary requests.
     * @return Interval-local records; zero-occurrence keys are omitted.
     */
    static ConvergenceTimerSnapshot convergenceTimerDelta(
        const ConvergenceTimerSnapshot &before,
        const ConvergenceTimerSnapshot &after);

    /**
     * @brief Write request-matched initial/converged timer evidence.
     *
     * Every CSV row belongs to one of the exact three prompt identities retained
     * on both sides of the economy assertion. Movement training owns a disjoint
     * cache-identity interval, so no overlap filtering is necessary.
     *
     * @param baseline Selected initial-epoch samples in comparison order.
     * @param converged Selected converged-epoch samples in comparison order.
     */
    void writeConvergenceTimerSamples(
        const std::vector<const ResidencyConvergenceTimings::RequestSample *>
            &baseline,
        const std::vector<ResidencyConvergenceTimings::RequestSample>
            &converged);

    /**
     * @brief Build one cache-distinct valid-token service workload.
     *
     * Certification needs broad natural routing so every real sparse
     * participant and runtime phase contributes a measured service profile.
     * The deterministic SplitMix corpus changes only valid embedding rows and
     * enters the graph through the ordinary serving API. Its routed rows are
     * calibration evidence, not optimization demand: the production admission
     * state quarantines the complete corpus and discards its histogram bank
     * before the next public request boundary.
     *
     * @param request_index Stable request ordinal within the service corpus.
     * @return Prompt-sized deterministic token vector unique to this ordinal.
     */
    std::vector<int32_t> makeEconomyWorkloadPrompt(int request_index) const;

    /**
     * @brief Map one workload identity to a collision-free prefix-cache block.
     *
     * The usable embedding interval is split between service certification and
     * stationary convergence. Within either typed half the mapping is
     * injective and skips the authenticated Hugging Face prompt's first token,
     * so neither calibration nor timing can seed the later parity prefix.
     *
     * @param prompt_namespace Typed economy-traffic namespace.
     * @param request_index Non-negative identity within that namespace.
     * @return Valid vocabulary row reserved for this exact request identity.
     */
    int32_t economyPromptLeadingToken(
        EconomyPromptNamespace prompt_namespace,
        int request_index) const;

    /**
     * @brief Preserve the authenticated workload while changing its cache key.
     *
     * Timing requests use one cache-distinct leading token followed by repeated
     * authenticated rows, preserving the exact matched A/B geometry. A
     * movement-only request instead executes the leading causal rows of the
     * Hugging Face prompt itself. Any expert promoted from that admitted demand
     * is therefore exercised again by the later fixed parity prefill. If a
     * model's initial histogram window exceeds the reference prompt, the full
     * prompt is used rather than inventing unauthenticated suffix rows.
     *
     * @param request_index Stable request ordinal within the measured corpus.
     * @param role Typed economy phase that owns this request geometry.
     * @return Typed timing corpus or an exact causal prefix of the HF corpus.
     */
    std::vector<int32_t> makeReferenceShapedEconomyPrompt(
        int request_index,
        ReferenceEconomyPromptRole role) const;

    /**
     * @brief Build the exact stationary prefill that closes a demand bank.
     *
     * The authority supplies the remaining logical routed-row count. This
     * helper repeats authenticated model tokens without injecting a new leading
     * token. The caller purges reusable prefix state through the production API
     * so these same causal rows execute rather than being served from cache.
     *
     * @param routed_rows Exact positive active-bank headroom to consume.
     * @return One valid model prompt with exactly @p routed_rows rows.
     */
    std::vector<int32_t> makeDemandWindowClosurePrompt(
        std::uint64_t routed_rows) const;

    /** @return Median of a non-empty timing corpus without changing it. */
    static std::uint64_t medianNanoseconds(
        const std::vector<std::uint64_t> &samples);

    /** @return Whether the central matrix assigned this cell the speed witness. */
    bool requiresObservedConvergenceSpeedup() const noexcept;

    /**
     * @brief Select the one retained graph-family lifecycle this cell needs.
     *
     * A numerical/movement cell consumes diagnostic checkpoints throughout
     * its request, so preparing an additional lean family cannot contribute
     * evidence.  Only a centrally designated throughput witness must measure
     * ordinary inference without diagnostic D2D publication before switching
     * to the already-prepared diagnostic family for mathematical parity.
     * Keeping this decision beside the typed evidence role prevents every
     * Dynamic matrix cell from paying a second native graph capture merely
     * because it permits expert movement.
     *
     * @return Immediate diagnostics for ordinary cells, or a prepared lean to
     *         diagnostic transition for an observed-throughput witness.
     */
    ParitySnapshotSetupMode paritySnapshotSetupMode() const noexcept;

    /**
     * @return Model-workload and topology-specific convergence objective.
     *
     * An ordinary 122B cell needs one wide publication whose durable ledger
     * proves both available axes. The centrally designated speed witness and
     * the smaller-model campaign require four successive publications to reach
     * the measured taper before timing. Every caller consumes this one value
     * rather than reproducing model/evidence-role integer tests at later
     * lifecycle boundaries.
     */
    DynamicResidencyConvergenceTarget dynamicResidencyConvergenceTarget() const;

    /**
     * @brief Classify the complete topology plus mathematical-workload target.
     *
     * A broad service-certificate request may validly move experts that the
     * fixed parity prompt never selects. Such a wave remains in the movement
     * ledger and economy CSV, but it cannot close the numerical proof. The
     * exact reference prefill must author at least one later promotion so the
     * captured parity graph can execute that destination and compare its
     * canonical per-route contribution.
     */
    DynamicResidencyConvergenceState
    classifyDynamicResidencyProofConvergence(
        const DynamicResidencyConvergenceTarget &target,
        const DynamicResidencyConvergenceOrigin &origin,
        const MoEOptimizationStatus &status,
        const MoEOptimizationMovementLedger &ledger) const noexcept;

    void SetUp() override;

    /** @brief Retire the typed case only after production runner teardown. */
    void TearDown() override;

    /**
     * @brief Preserve process caches only while a static MTP runner may live.
     * @return True for exact process-campaign cells eligible for typed reuse.
     */
    bool preserveParityPipelineCachesBetweenTests() const override;

    /**
     * @brief Move one successfully yielded runner into the bounded cache.
     *
     * A red cell never publishes reusable execution state. Its runner is
     * destroyed and the kernel registry is cleared here because the common
     * teardown deliberately skipped that operation for an otherwise eligible
     * retained cell.
     */
    void retireOwnedParityRunners() override;

    void applyModelOverrides() override;

    bool broadcastRootFlag(bool root_value) const;

    /**
     * @brief Prove each expert-only endpoint owns one immutable compact arena.
     *
     * Every process retains its own PerfStats records, while the graph-native
     * overlay spreads its accelerator tier and the two CPU-NUMA endpoints
     * across MPI instances. The dense continuation uses its model graph's
     * activation arena and must not duplicate the follower runner's storage.
     * Aggregate setup evidence only after the worker loop closes; this proves
     * the real follower graph did not allocate one packet per layer or alias
     * independent participants.
     */
    void assertParticipantCompactBufferArenaEvidence() const;

    /**
     * @brief Prove a one-rank GPU+CPU graph used canonical ticket boundaries.
     *
     * A rank-local topology has no auxiliary MPI runner and must not advertise
     * one merely to satisfy the mapped-follower evidence used by node-wide
     * cases. Instead, every MoE layer must materialize and capture its exact
     * canonical-ticket consumer, while the full graph must lower to typed
     * captured/manual/captured transactions with one successor per boundary.
     */
    void assertRankLocalCanonicalTicketGraphEvidence() const;

    /**
     * @brief Prove tickets selected the topology's production specialization.
     *
     * Device-owned epochs replaced the old host-scheduled fixed-capacity
     * participant runner. Setup materializes a bounded row-shape family once;
     * each authenticated ticket selects one member without mutating token or
     * position state on the follower. Native parent capture/replay is asserted
     * independently by the shared production-path evidence gate.
     */
    void assertMappedParticipantGraphEvidence() const;

    /**
     * @brief Exact device snapshots for one routed layer under one request.
     *
     * `overlay_participants` is the overlay-wide `(expert -> global
     * participant)` bank selected by the request's acquired epoch status.
     * `domain_participants` is the invocation-local `(route slot -> domain
     * participant)` schedule; `-1` means the selected expert belongs to another
     * overlay domain and is completed by the heterogeneous return transaction.
     * Both pointers remain owned by SnapshotCapture.
     */
    struct PinnedDeviceRouteEvidence
    {
        const float *overlay_participants = nullptr;
        const float *domain_participants = nullptr;
        const float *runtime_weights = nullptr;
        size_t expert_count = 0u;
        size_t route_count = 0u;
        uint64_t epoch = 0u;
        int selected_bank = -1;
    };

    /**
     * @brief Exact placement epoch consumed by one completed graph transaction.
     *
     * Snapshot capture copies these scalar values on the producer stream before
     * the request releases its RCU reader. Combining the selected-bank status
     * with that bank's own published epoch therefore observes the same device
     * authority used by packet dispatch, on local and remote topologies alike.
     */
    struct PinnedDevicePlacementEpochEvidence
    {
        uint64_t epoch = 0u;
        int selected_bank = -1;
    };

    /**
     * @brief Resolve and cross-check the request-pinned epoch on every MoE layer.
     *
     * @return One model-wide execution identity, or no value after recording a
     *         focused test failure for missing, malformed, or split evidence.
     */
    std::optional<PinnedDevicePlacementEpochEvidence>
    pinnedDevicePlacementEpochEvidence() const;

    /**
     * @brief Resolve both typed route projections from live device checkpoints.
     *
     * @param production_stage_prefix Exact live snapshot prefix ending in `_`.
     * @param expected_route_count Router slots in the current checkpoint.
     * @param expected_expert_count Logical routed experts in the model layer.
     * @return Complete evidence, or no value after recording a test failure.
     */
    std::optional<PinnedDeviceRouteEvidence>
    pinnedDeviceRouteEvidence(
        std::string_view production_stage_prefix,
        size_t expected_route_count,
        size_t expected_expert_count) const;

    /**
     * @brief Prove one live MTP sidecar consumed its request-pinned route bank.
     *
     * The ordinary parity forward cannot prove this value because speculative
     * sidecars do not execute in teacher-forced prefill/decode. This check runs
     * immediately after a real grouped transaction while its context-qualified
     * snapshots are live. It binds numerical router output to the exact RCU
     * bank/epoch and final domain assignment used by sparse dispatch.
     *
     * @param production_stage_prefix Context-qualified `MTP0_` prefix.
     * @param expected_epoch Main-model placement epoch for the transaction.
     * @param top_k Number of live router slots in the one-row sidecar.
     * @param num_experts Logical routed-expert count.
     * @return True only when every route-authority value is complete and valid.
     */
    bool validatePinnedMTPSidecarRouteEvidence(
        std::string_view production_stage_prefix,
        uint64_t expected_epoch,
        int top_k,
        int num_experts) const;

    /**
     * @brief Attribute live parity router checkpoints to the published epoch.
     *
     * Snapshot values are exact integer expert ids produced by the real router.
     * Pair them with the request-selected global placement bank and the final
     * domain-local schedule while all checkpoints remain live, before the
     * parity harness clears diagnostics. The setup-time residency snapshot
     * supplies only stable endpoint topology; it is deliberately not used to
     * reconstruct live placement after Dynamic movement.
     */
    void cacheDeviceRouteAssignmentEvidence();

    /**
     * @brief Persist immutable setup ownership and participant topology.
     *
     * The host residency snapshot is the cold-start topology authority. In an
     * all-GPU Dynamic cell it intentionally does not shadow later device-owned
     * placement epochs. Keep the established `expert_owner_map.csv` artifact
     * as the setup baseline and endpoint dictionary; exact live assignments
     * are recorded per route in `prefill_routed_expert_routes.csv` from the
     * reducer's device ledger.
     */
    void writeExpertOwnerTopologyBaselineCsv() const;

    /**
     * @brief Persist value-level routed-expert evidence for baseline and moves.
     *
     * Aggregate cosine metrics cannot distinguish a missing participant from
     * a correct route computed with the wrong weight slice.  The baseline
     * layer and every layer containing a committed promotion are therefore
     * recorded element by element against Hugging Face.  Per-route norms also
     * expose a zero, duplicated, or explosive migrated contribution without
     * requiring another instrumented inference run.  This diagnostic executes
     * only after the captured production forward and before snapshot teardown.
     */
    void writePrefillRoutedExpertDiagnosticCsv();

    /**
     * @brief Prove resident participants executed and capacity-idle ones did not.
     *
     * The device-owned placement bank determines whether each declared
     * endpoint owns any expert after automatic capacity resolution. Routed
     * checkpoints prove every selected endpoint was resident under that exact
     * epoch. Cross-rank residents additionally require endpoint-owned traffic
     * somewhere in the real production workload after both retained graphs
     * publish Complete. Residency alone cannot require one bounded prompt to
     * select an expert; an idle endpoint, however, can never appear in the
     * pinned route schedule. Local arithmetic remains covered by the same
     * layer/LM-head parity.
     */
    void assertActiveTierRouteEvidence() const;

    /**
     * @brief Prove the real sparse transport moved compact packets between tiers.
     *
     * Local-route completion proves that every participant ran an expert, but
     * it does not independently prove that the production sparse collective
     * carried compact request and result packets. The graph-native transport
     * publishes those byte counts through PerfStats. Folding them across both
     * MPI instances makes a missing dispatch, missing return, or silently
     * bypassed CPU cold tier a fatal parity failure without paying for a second
     * model setup in a profiler-only smoke test.
     */
    void assertSparseTransportPerfStatsEvidence() const;

    /**
     * @brief Prove the full request lifecycle honored the shared segmented graph.
     *
     * PrefixRuntimeStateSnapshot proves the public OrchestrationRunner admitted
     * a chunked request instead of treating the test as three unrelated
     * forwards. Per-rank PerfStats then prove the distributed schedule contract
     * was published, an expert-only participant consumed the same ordered
     * transactions, and the CUDA continuation retained complete prompt-wide
     * checkpoints for the existing Hugging Face/CSV comparator. Dynamic cells
     * deliberately execute calibration and migration-training requests before
     * parity, so PerfStats and the prefix probe are cumulative by design. The
     * assertion consequently proves the whole lifecycle rather than inventing
     * a test-only reset edge: all schedules succeed, every captured row is
     * accounted for, both roles retain identical ordered digests, and the one
     * request-scoped parity evidence independently proves `[4,4,1]`. Training
     * and prefix seeding may also publish valid snapshot aggregations. Mandatory
     * prefix restore then contributes exactly one serial suffix-decode record.
     */
    void assertSegmentedPrefillEvidence() const;

    /**
     * @brief Verify every chunk-scoped Hugging Face checkpoint became full-prompt data.
     *
     * SnapshotCapture retains context-qualified copies such as
     * `PREFILL_CHUNK_0_layer0_...` for diagnosis and rewrites the bare semantic
     * key to the ordered aggregate. This check prevents a short final chunk
     * from passing merely because a comparison used the shorter tensor length.
     * @param before Immutable counters immediately before the parity request;
     *               its delta excludes earlier training and later prefix seeds.
     */
    void assertSegmentedPrefillCheckpointCoverage(
        const ParityPrefillSnapshotEvidence &before);

    bool collectivelyCheckHardwareAndModel() const;

    /**
     * @brief Lexicographic objective for one adversarial CPU-tier subsequence.
     *
     * The first component maximizes authenticated routes assigned to the CPU
     * participant remote from the continuation rank. Only after that total is
     * fixed does the second component minimize traffic owned by the colocated
     * CPU participant. This is an initial-layout construction objective, never
     * a runtime histogram or placement decision.
     */
    struct AdversarialCpuPlacementScore
    {
        std::uint64_t remote_routes = 0;
        std::uint64_t local_routes = 0;
        bool reachable = false;
    };

    /** @return Whether @p candidate strictly improves the layout objective. */
    static bool improvesAdversarialCpuPlacement(
        const AdversarialCpuPlacementScore &candidate,
        const AdversarialCpuPlacementScore &incumbent) noexcept;

    /**
     * @brief Select exact lower-tier membership under random owner partitioning.
     *
     * Whole-expert ownership first sorts the selected expert IDs, applies the
     * production ordinal/random permutation to those positions, and gives one
     * balanced span to each participant. Selecting a different set therefore
     * changes the sorted-rank occupied by every later expert. A small dynamic
     * program solves that subsequence problem exactly instead of assuming an
     * expert ID maps directly to a participant.
     *
     * @param routes Authenticated prefill route count for every expert.
     * @param layer Model layer used by the production random permutation.
     * @param tier_index Lower-priority CPU tier index used by that permutation.
     * @param selected_count Exact tier quota for this layer.
     * @param participant_count Number of balanced NodeTP CPU owners.
     * @return Boolean expert mask with exactly @p selected_count entries.
     */
    static std::vector<bool> selectAdversarialCpuExperts(
        const std::vector<std::uint64_t> &routes,
        int layer,
        int tier_index,
        int selected_count,
        int participant_count,
        RoutedExpertOwnerOrder owner_order);

    /**
     * @brief Load exact Hugging Face prefill demand by model layer and expert.
     *
     * This histogram is immutable mathematical evidence. Runtime placement is
     * still authored exclusively by the production controller from its live
     * device histograms; the fixture uses these counts only to decide whether
     * a completed promotion can be re-exercised by the later parity prefill.
     *
     * @param layer_count Number of main-model or complete model layers to load.
     * @param expert_count Authenticated routed-expert cardinality.
     * @return Dense `[layer][expert]` route counts.
     * @throws std::exception for missing or malformed reference evidence.
     */
    std::vector<std::vector<std::uint64_t>>
    loadAuthenticatedPrefillRouteCounts(
        int layer_count,
        int expert_count);

    /**
     * @brief Install an authenticated workload-adversarial initial tier layout.
     *
     * The Hugging Face pack is already the mathematical oracle for this parity
     * cell. Its routing IDs define only the starting condition: high-demand
     * experts are deliberately left on lower-priority participants, with the
     * strongest CPU candidates owned by the rank remote from continuation.
     * Runtime movement remains driven exclusively by histograms emitted from
     * the real Llaminar sparse-collective graphs.
     *
     * @param requested Inventory-bound dynamic production request.
     * @return Same request with complete explicit per-layer initial placement.
     * @throws std::exception For missing/malformed reference evidence or a
     *         layout that cannot make remote CPU demand strictly dominant.
     */
    MoERoutedExpertPlacementPlan installReferenceAdversarialPlacements(
        MoERoutedExpertPlacementPlan requested);

    /**
     * @brief Certify the exact capacity-resolved adversarial epoch-one layout.
     *
     * The setup request carries only a complete expert permutation. This check
     * runs after the production runner has resolved physical quotas and frozen
     * concrete placements. It proves that lower-priority CPU membership owns
     * the hottest suffix and that the first, remote CPU participant received a
     * genuinely hotter expert than its colocated peer on at least one layer.
     *
     * @return True when the frozen production owner map satisfies the complete
     *         adversarial contract, or when this cell declared no such order.
     */
    bool certifyInstalledReferenceAdversarialPlacement() const;

    bool setupPipeline();

    /**
     * @return Passive status projected by the sole production authority.
     * @throws std::logic_error when the runner is absent or its lifecycle failed.
     */
    MoEOptimizationStatus optimizationStatus() const;

    /** @return Exact durable movement-wave count from the production owner. */
    std::uint64_t localCommittedWaveCount() const;

    /**
     * @brief Measure one exact production workload inside one residency epoch.
     *
     * Both sides use identical prompt IDs, boundary calls, decode budgets, and
     * greedy token trajectories. This convergence cell selects the production
     * invalidate-on-rebalance prefix policy, so publication advances the
     * fingerprint and the post-movement replay cannot restore an initial-epoch
     * entry. Both sides therefore execute full model compute. Ordinary
     * maintenance notifications remain enabled. Any publication during the
     * cohort is a hard protocol failure instead of a sample that can be hidden
     * by median selection.
     *
     * @param cohort Initial adversarial epoch or post-movement epoch.
     * @return True only after the complete workload ran in one exact epoch.
     */
    bool collectInferenceTimings(ResidencyTimingCohort cohort);

    /** @return Stable diagnostic spelling for the active priority topology. */
    std::string convergenceTopologyName() const;

    /**
     * @brief Assert and serialize the real before/after convergence evidence.
     *
     * A two-percent floor is deliberately larger than timer quantization and
     * ordinary run-to-run jitter on this host. The median makes the gate robust
     * to background OS activity while retaining a directional performance
     * requirement for both time-to-first-token prefill and steady decode.
     */
    void assertAndWriteObservedConvergenceSpeedup();

    /**
     * @brief Retire Dynamic proof prefixes before the mathematical prefill.
     *
     * The movement proof is ordinary production traffic and intentionally
     * executes a causal prefix of the authenticated Hugging Face prompt. A
     * movement wave may publish before that request is harvested, making its
     * cache entry valid under the final placement epoch. Clearing request KV
     * alone cannot distinguish that entry from the fresh parity seed. Cross
     * the public coordinated purge boundary while worker ranks are still live,
     * then publish the typed lifecycle transition that admits parity traffic.
     *
     * Prefix caching remains enabled. The fresh parity prefill immediately
     * seeds the new entry and the standard decode phase must restore and
     * numerically certify it.
     *
     * @return True only after every coordinated participant has purged the
     *         reusable archive and the lifecycle is ready for numerical parity.
     */
    bool prepareDynamicNumericalParityBoundary();

    /**
     * @brief Execute one ordinary request prefill for economy evidence.
     *
     * @param tokens Real model tokens supplied through the serving API.
     * @param purpose Stable diagnostic name for the traffic lifecycle phase.
     * @return Whether production prefill completed successfully.
     */
    bool runDynamicEconomyPrefill(
        const std::vector<int32_t> &tokens,
        const char *purpose);

    /**
     * @brief Execute one ordinary bounded decode and notify maintenance.
     *
     * The first generated token after prefill consumes existing logits; the
     * next budgeted call executes a real DecodeToken graph. Callers that need
     * decode-route evidence therefore issue the typed boundary/forward pair
     * without invoking any test-only routing surface.
     *
     * @param response_token_budget Positive serving response budget.
     * @param purpose Stable diagnostic name for the traffic lifecycle phase.
     * @return Completion state, or no value when inference/maintenance failed.
     */
    std::optional<bool> runDynamicEconomyDecode(
        int response_token_budget,
        const char *purpose);

    /**
     * @brief Replay one untimed request with the exact measured route shape.
     *
     * Every pair restores the same terminal prefill state, consumes its logits
     * with one boundary sample, and executes one DecodeToken forward. This is
     * deliberately the same production sequence as collectInferenceTimings(),
     * minus clocks and assertions, so the planner cannot optimize a longer
     * autoregressive trajectory than the one judged by the convergence gate.
     *
     * @param corpus_request Stable request ordinal in the timing corpus.
     * @return Whether all production requests and maintenance notifications ran.
     */
    bool replayStationaryConvergenceRequest(int corpus_request);

    /**
     * @brief Execute one movement request whose hot routes remain replayable.
     *
     * The prefill is the exact authenticated Hugging Face request. Its pending
     * progress is retired by the next ordinary production request before that
     * request submits another captured prefill. This is sufficient to close
     * every histogram bank and ensures later numerical parity replays precisely
     * the route family that selected each promoted expert.
     *
     * @param corpus_request Stable request ordinal used by the traffic driver.
     * @return True after the boundary and one canonical routed forward finish.
     */
    bool replayStationaryMovementProofRequest(int corpus_request);

    /**
     * @brief Learn and publish the measured service profile from real traffic.
     *
     * Transport preparation already completed through prepareForInference().
     * This phase supplies a bounded broad token corpus so every live sparse
     * participant receives natural prefill/decode work. Production publishes
     * the certificate while the corpus remains quarantined from optimization
     * demand. The following public request boundary discards that calibration
     * bank and admits the real convergence/parity workload.
     *
     * @return True after the sole production authority reports certification.
     */
    bool certifyDynamicResidencyEconomy();

    /**
     * @brief Feed bounded real requests until distributed residency actually moves.
     *
     * Physical topology preparation, measured service certification, and the
     * initial stationary timing cohort are complete before this method begins.
     * This driver replays those exact prompt identities while optimizing, so
     * phase-weighted placement sees the same prefill and decode trajectories
     * that the A/B gate will judge. Only a typed demand-window-rotation phase
     * uses cache-distinct identities after the movement objective is already
     * satisfied. The background worker owns interval
     * selection, staging, transfer, overlap validation, and publication; no
     * histogram, placement, or completion value is injected here.
     *
     * The parity-artifact rank is the sole traffic-control authority. Remote
     * ranks are already inside `MPIWorkerLoop` and execute authenticated
     * transaction-follower commands; they are not peer test drivers. Calling a
     * test-owned MPI collective here would create a second command protocol
     * and collide with the follower's next typed command receive.
     *
     * @return True on the coordinated root after the typed production owner
     *         satisfies the required publication and movement-axis target.
     *         The post-shutdown PerfStats gate mirrors payload and economy
     *         diagnostics, but never controls this driver.
     */
    bool driveDynamicResidencyToDistributedMigration();

    /**
     * @brief Prove all-GPU placement with the sole device-resident authority.
     *
     * The heterogeneous host authority publishes `committed_waves` and its
     * migration ledger under `moe_overlay_residency`.  An all-GPU topology has
     * no such second authority: the authenticated device-controller command is
     * the placement decision and its completed physical transaction is the
     * evidence. This fold checks numeric-priority promotion/demotion in every
     * Dynamic topology and additionally requires same-priority skew movement
     * whenever one declared priority owns two or more physical participants.
     */
    void assertDeviceResidentMovementEvidence() const;

    /**
     * @brief Prove Dynamic movement used the bounded physical GPU stream pool.
     *
     * The typed handles and real-device integration tests are the stream-
     * ownership authority. This fold is deliberately observability-only: it
     * proves that the real model constructed that path, without treating a
     * counter as evidence that a transfer or movement objective completed.
     */
    void assertPersistentTransferExecutionPoolEvidence() const;

    /**
     * @brief Fold and validate static immobility or dynamic movement evidence.
     *
     * Dynamic movement must be a capacity-preserving promotion/demotion cycle
     * across the two distinct domains, MPI ranks, and GPU backends. Static cells
     * must publish their typed immobility check and no positive movement edge.
     */
    void assertResidencyMovementEvidence() const;

    /**
     * @brief Prove request-local LLEP movement or static immobility.
     *
     * Durable tier migration is certified by moe_overlay_residency above.
     * LLEP additionally owns a request-scoped assignment/copy/apply graph, so
     * its proof must come from the independent moe_rebalance counters. Static
     * cells check the same counters at zero; setup-time initial placement is
     * intentionally outside this request-movement surface.
     */
    void assertRequestMovementPolicyEvidence() const;

    /** @brief Stable identity of one histogram-driven promotion edge. */
    struct PromotedExpert
    {
        int layer = -1;
        int expert = -1;
        int destination_participant = -1;
        uint64_t candidate_epoch = 0u;

        /** @return Whether two records name the same routed expert. */
        bool operator==(const PromotedExpert &) const = default;
    };

    /** @brief Exact post-publication route and numerical checkpoint witness. */
    struct PromotedExpertExecutionWitness
    {
        ParityForwardPhase phase = ParityForwardPhase::Prefill;
        int step = -1;
        PromotedExpert promotion;
        int selected_placement_bank = -1;
        size_t routed_rows = 0u;
        size_t comparable_route_rows = 0u;
        size_t production_executed_route_rows = 0u;
        size_t reference_executed_route_rows = 0u;
        size_t exact_zero_route_rows = 0u;
        size_t one_sided_zero_route_rows = 0u;
        size_t compared_elements = 0u;
        float expert_contribution_cosine = 0.0f;
        float production_l2_norm = 0.0f;
        float reference_l2_norm = 0.0f;
        float absolute_l2_error = 0.0f;
        float root_mean_square_error = 0.0f;
        RoutedExpertContributionState contribution_state =
            RoutedExpertContributionState::InvalidGeometry;
        RoutedExpertContributionProof numerical_proof =
            RoutedExpertContributionProof::Invalid;
        RoutedExpertReferenceLineage reference_lineage =
            RoutedExpertReferenceLineage::Canonical;
        RoutedExpertContributionDisposition disposition =
            RoutedExpertContributionDisposition::Failed;
        bool valid_geometry = false;
        bool finite = false;
        bool numerically_comparable = false;
        bool numerically_passed = false;
    };

    /** @brief Per-expert contribution evidence for every route at a moved layer. */
    struct RoutedExpertContributionWitness
    {
        ParityForwardPhase phase = ParityForwardPhase::Prefill;
        int step = -1;
        int layer = -1;
        int expert = -1;
        int domain_participant = -1;
        int selected_placement_bank = -1;
        RoutedExpertContributionComparison comparison;
        RoutedExpertContributionPublication publication =
            RoutedExpertContributionPublication::ContinuationCanonical;
        RoutedExpertContributionProof numerical_proof =
            RoutedExpertContributionProof::Invalid;
        RoutedExpertReferenceLineage reference_lineage =
            RoutedExpertReferenceLineage::Canonical;
        RoutedExpertContributionDisposition disposition =
            RoutedExpertContributionDisposition::Failed;
        float post_return_expert_output_cosine = 0.0f;
        bool numerically_comparable = false;
        bool evidence_passed = false;
    };

    /**
     * @brief Resolve route and input lineage for every compared routed layer.
     *
     * A current route-set difference preserves comparability for matched
     * per-route expert values because the layer input is still canonical. It
     * invalidates a dense aggregate as proof of an individual remote addend,
     * and it changes the residual consumed by every later layer. The shared
     * typed transition records all three cases explicitly. Ordered layer keys
     * make the rule independent of callback vector order.
     *
     * @param layers Complete per-layer comparison records for one checkpoint.
     * @return Input lineage keyed by model layer.
     */
    static std::map<int, RoutedExpertReferenceLineage>
    routedExpertReferenceLineageByLayer(
        const std::vector<LayerStats> &layers);

    /**
     * @brief Retain promotion identities before a parity collector reset.
     *
     * The parity harness resets live PerfStats between campaign phases so the
     * CSV for each numerical comparison has an unambiguous interval. Movement
     * is model-lifetime state and deliberately survives that reset. Preserve
     * only the immutable layer/expert identities from the production movement
     * ledger; routing values and numerical outputs are still read from the
     * later live graph checkpoints.
     */
    void cacheCommittedPromotionEvidence();

    /**
     * @brief Persist the authenticated physical movement ledger used by parity.
     *
     * Numerical CSVs name the layer and routed expert that diverged, but that is
     * not enough to diagnose a moved-weight defect: the production transaction
     * may have crossed a rank, backend, or numeric-priority boundary.  Export the
     * authority's typed edge ledger before the parity harness resets optional
     * telemetry. The CSV serializes authoritative state and never reconstructs
     * placement from PerfStats tags.
     */
    void writeCommittedMovementEvidenceCsv() const;

    /**
     * @brief Preserve the complete process-local residency decision trail.
     *
     * The compact `expert_movement.csv` contains committed edges only. This
     * companion artifact retains proposal, capacity, economy, and physical
     * publication records, including failures. PerfStats is process-local, so
     * followers use rank-qualified names while the artifact authority retains
     * the canonical filename. Export occurs after the production worker loop
     * has closed and cannot affect placement or inference ordering.
     */
    void writeResidencyDiagnosticsCsv() const noexcept;

    /** @brief Typed source of one routed-expert comparison callback. */
    struct RoutedExpertCheckpointContext
    {
        ParityForwardPhase phase = ParityForwardPhase::Prefill;
        int step = -1;
        /** Engaged only for an exact production sidecar snapshot bank. */
        std::optional<ComparedMTPParityCheckpoint> mtp;

        /** @return Whether phase, step, and optional sidecar agree. */
        [[nodiscard]] bool valid() const noexcept
        {
            if (mtp.has_value())
            {
                return mtp->valid() &&
                       phase == ParityForwardPhase::Decode &&
                       step == mtp->reference_step;
            }
            return phase == ParityForwardPhase::Prefill
                       ? step == -1
                       : step >= 0;
        }
    };

    /** @brief Exact production/reference prefixes for one routed layer. */
    struct RoutedExpertSnapshotNamespace
    {
        std::string production_stage_prefix;
        std::string reference_stage_prefix;

        /** @return Exact production key for a semantic stage suffix. */
        [[nodiscard]] std::string productionKey(
            std::string_view semantic_stage) const
        {
            return production_stage_prefix + std::string(semantic_stage);
        }

        /** @return Exact reference key for a semantic stage suffix. */
        [[nodiscard]] std::string referenceKey(
            std::string_view semantic_stage) const
        {
            return reference_stage_prefix + std::string(semantic_stage);
        }
    };

    /** @brief One already-compared sidecar bank retained for CSV projection. */
    struct MTPCheckpointDiagnostic
    {
        ComparedMTPParityCheckpoint checkpoint;
        std::vector<StageComparisonResult> stages;
    };

    /** @brief One row in the optional Qwen MTP diagnostic artifact. */
    struct MTPNumericalDiagnostic
    {
        int call = 0;
        int reference_step = 0;
        int reference_depth = 0;
        std::string stage;
        std::string production_key;
        std::string reference_key;
        StageComparisonResult comparison;
        bool exact_indices = true;
        bool finite = true;
        bool passed = false;
    };

    /** @brief Full values retained only when a diagnostic comparison fails. */
    struct MTPFailureDiagnostic
    {
        int call = 0;
        int reference_step = 0;
        int reference_depth = 0;
        std::string stage;
        std::vector<float> production;
        std::vector<float> reference;
    };

    /**
     * @brief Bounded serial row needed for grouped-verifier batch invariance.
     *
     * Only the transaction plan's authenticated checkpoint row is retained.
     * Other serial tokens establish the exact trajectory but are not needed
     * to compare grouped verifier row zero, so copying every large per-layer
     * snapshot would waste host memory and time.
     */
    struct MTPSerialVerifierDiagnostic
    {
        ProductionParityMTPSerialOracleBoundary boundary;
        std::map<std::string, std::vector<float>> snapshots;
        uint64_t execution_epoch = 0;
    };

    /**
     * @brief Resolve one available routed layer without guessing its namespace.
     *
     * Main-model snapshots are named by their real transformer layer. MTP
     * snapshots instead inherit an exact context-qualified graph namespace
     * from the comparison that just consumed them. A synthetic CSV layer is
     * deliberately never converted into a production key here.
     */
    std::optional<RoutedExpertSnapshotNamespace>
    routedExpertSnapshotNamespace(
        const RoutedExpertCheckpointContext &context,
        int layer) const;

    /** @return Whether @p key is a grouped main-verifier checkpoint. */
    static bool isMTPMainVerifierDiagnosticKey(std::string_view key);

    /**
     * @brief Copy the bounded main-verifier diagnostic surface now live.
     * @return Keyed production snapshots, excluding unrelated graph outputs.
     */
    std::map<std::string, std::vector<float>>
    captureMTPMainVerifierDiagnostics() const;

    /** @return Semantic stage suffix from a layer-qualified snapshot key. */
    static std::string mtpMainVerifierStage(std::string_view key);

    /** @return Whether all tensor statistics prove finite input and output. */
    static bool mtpComparisonIsFinite(const StageComparisonResult &comparison);

    /**
     * @brief Preserve full values only for a failed numerical diagnostic.
     *
     * Successful cells retain compact scalar rows.  Failure-only copying keeps
     * the artifact as actionable as the historical long-horizon proof without
     * making every matrix cell duplicate hundreds of megabytes of tensors.
     */
    void retainMTPFailureValues(
        const MTPNumericalDiagnostic &diagnostic,
        std::span<const float> production,
        std::span<const float> reference);

    /**
     * @brief Observe an ordinary main-model routed checkpoint.
     */
    void observeComparedParityCheckpoint(
        ParityForwardPhase phase,
        int step,
        const std::vector<LayerStats> &layers) override;

    /**
     * @brief Retain exactly one same-prefix serial verifier oracle for this cell.
     * @param boundary The checkpoint published by the generic captured request.
     *
     * Duplicate publication is fatal. Another request in the same placement
     * epoch is not interchangeable with this request's hidden/KV input state.
     */
    void retainMTPSerialVerifierDiagnostic(
        const ProductionParityMTPSerialOracleBoundary &boundary);


    /**
     * @brief Observe the primary sidecar bank under its exact live namespace.
     *
     * A deeper transaction overwrites one reusable chained bank repeatedly.
     * Its terminal numerical row remains certified by the generic MTP gate,
     * while moved-expert provenance is attributed once from the primary bank
     * whose input lineage is canonical and independently named.
     */
    void observeComparedMTPParityCheckpoint(
        const ComparedMTPParityCheckpoint &checkpoint,
        const std::vector<StageComparisonResult> &stages) override;

    /**
     * @brief Retain the selected serial row from the generic production oracle.
     *
     * The grouped verifier comparison consumes its first physical row, whose
     * logical decode-step identity is selected before serial execution.
     * Capturing any other row would reproduce the historical multi-request
     * diagnostic cost without increasing the batch-invariance proof surface.
     */
    void observeProductionParityMTPSerialOracleBoundary(
        const ProductionParityMTPSerialOracleBoundary &boundary) override;

    /**
     * @brief Compare grouped verifier row zero with HF and serial M=1 rows.
     *
     * This is the Qwen-specific batch-invariance proof previously implemented
     * by launching a second serial/grouped campaign.  The generic production
     * proof has already produced both authorities, so this method only reads
     * their live diagnostic banks and emits scalar evidence.
     */
    void compareReusedMTPMainVerifierRows(
        const ProductionParityMTPTransactionBoundary &boundary);

    /**
     * @brief Consume the generic transaction as the Qwen MTP diagnostic source.
     */
    void observeComparedProductionParityMTPTransaction(
        const ProductionParityMTPTransactionBoundary &boundary,
        std::span<const int32_t> serial_oracle) override;

    /**
     * @brief Retain exact promoted-expert use from one compared checkpoint.
     *
     * Histogram movement is trained by the complete authenticated request, so
     * a profitable promotion may be hot only during decode or the recursive
     * predictor. The typed context admits only layers actually compared at
     * this boundary and supplies the exact graph/reference namespace.
     */
    void observeComparedRoutedExpertCheckpoint(
        const RoutedExpertCheckpointContext &context,
        const std::vector<LayerStats> &layers);

    /**
     * @brief Assert and export a post-publication promoted-expert witness.
     *
     * Movement, route selection, and numerical comparison remain three
     * independently produced authorities.  This epilogue only joins their
     * immutable evidence; it neither chooses a route nor causes maintenance.
     */
    void assertParityExecutionExercisesPromotedExpert();

    /**
     * @brief Return the inventory-resolved dense continuation authority.
     *
     * Before runner setup, rank zero retains reference-pack preparation. Once
     * production has bound the topology, comparisons and CSV output move to
     * the same rank that owns logits and stage snapshots.
     */
    int parityArtifactAuthorityRank() const override;

    /**
     * @brief Keep additive HF reference work off the production worker ranks.
     *
     * During parity, non-continuation ranks execute `runMPIWorkerLoop()` and
     * consume only typed serving commands. An unmatched test-only broadcast or
     * barrier would corrupt that protocol, so the continuation/artifact
     * authority alone owns the filesystem reference lease.
     */
    ParityReferenceGenerationCoordination
    parityReferenceGenerationCoordination() const override;

    /** @return Whether this process owns the inventory-resolved continuation. */
    bool isRootParityRank() const;

    bool synchronizedDecodeWorkAvailable();

    bool producedPrefillSummary(const ParityTestSummary &summary) const;

    bool producedDecodeSummary(const DecodeParitySummary &summary) const;

    /**
     * @brief Test whether one exact noncanonical HF branch is already complete.
     *
     * Missing branches are deliberately not generated here. This method runs
     * while the production graph and prepared 122B weights are resident; loading
     * the Python model at this boundary previously overlapped roughly 500 GB of
     * live state and was killed by the host OOM policy. The immutable production
     * checkpoints are queued below and the suite resolves them after all model
     * authorities retire.
     *
     * @param reference_step Main decode step that owns the sidecar transaction.
     * @param condition_tokens Recursive condition tokens consumed by MTP1..N.
     * @return True only when the deepest branch checkpoint is complete on disk.
     */
    bool hasHuggingFaceMTPBranchReference(
        int reference_step,
        const std::vector<int32_t> &condition_tokens) const;

    /**
     * @brief Copy one missing recursive context into the post-residency queue.
     * @param call Public grouped-decode call index used by the CSV.
     * @param reference_step Canonical main-model decode position.
     * @param reference_depth Number of recursive condition tokens consumed.
     * @param condition_tokens Exact device-owned condition-token trajectory.
     * @param production_prefix Snapshot namespace for the live recursive row.
     * @param required_stages Complete sidecar checkpoint contract.
     * @param snapshot_csv_path Existing per-cell diagnostic CSV to append later.
     * @return True only when every live checkpoint was copied successfully.
     */
    bool deferHuggingFaceMTPBranchReference(
        int call,
        int reference_step,
        int reference_depth,
        const std::vector<int32_t> &condition_tokens,
        const std::string &production_prefix,
        std::span<const std::string_view> required_stages,
        const std::filesystem::path &snapshot_csv_path);

    /** @return Semicolon-delimited token identity for a diagnostic CSV field. */
    static std::string joinMTPDiagnosticTokens(
        std::span<const int32_t> tokens);

    /**
     * @brief Write Qwen-specific diagnostics from the canonical MTP proof.
     *
     * No inference is legal here.  Sidecar rows were compared by the generic
     * production campaign, while grouped-main and serial-row diagnostics were
     * compared by the typed transaction observer before the reusable graph
     * banks could be overwritten.  This method only projects those immutable
     * results into the historical CSV schemas.
     */
    void writeReusedMTPHuggingFaceCheckpointEvidence();

    /**
     * @brief Compare the live grouped-MTP graph with recursive HF checkpoints.
     *
     * The classic decode parity loop is deliberately teacher forced so every
     * main-model row follows the exact Hugging Face trajectory. That loop does
     * not execute speculative sidecars. This check first records a serial
     * production trajectory by constraining the public decodeStep boundary to
     * one token. When both commands hold the same authenticated residency
     * epoch, grouped MTP
     * must reproduce that trajectory exactly. Dynamic and LLEP requests may
     * legitimately publish a new placement between those independent requests;
     * such rows are instead compared directly with their Hugging Face main-model
     * checkpoints and the epoch mismatch is retained in the diagnostic CSV.
     *
     * Sidecar tensors remain compared directly with Hugging Face whenever the
     * serial production prefix still names the same main-model row. Recursive
     * predictors select branch-qualified reference tensors using the proposal
     * tokens observed from the device authority. This keeps every checkpoint
     * mathematically comparable even when a narrow quantized-logit tie sends
     * production down a different draft branch from canonical HF argmax. The
     * two diagnostic CSVs complement—never replace—the six canonical
     * prefill/decode artifacts.
     */
    void runMTPHuggingFaceCheckpointParity();

    [[noreturn]] void abortGraphNativeWorld(const std::string &reason) const;

    [[noreturn]] void abortAfterRootThrow(const char *phase, const std::string &what) const;

    /**
     * @brief Persist participant-local sparse endpoint evidence beside parity CSVs.
     *
     * PerfStats is process-local by design, so a root-only export cannot show
     * which CUDA/ROCm follower accepted each routed-expert packet.  Every MPI
     * instance writes a rank-qualified file after the production worker loop
     * has closed. Records aggregate complete service timing by layer,
     * participant, tier, and retained graph-family geometry. Exact route
     * counts and the union of executed experts remain in the bounded overlay
     * profiler evidence; movement epochs remain in the residency artifacts.
     * Keeping transaction-varying values out of this timing key prevents the
     * diagnostic collector from perturbing long-horizon inference.
     */
    void writeSparseEndpointEvidenceCsv() const;

    /**
     * @brief Prove the live MTP sidecar consumed a stable read-only mailbox.
     *
     * Unit contracts establish that every supported sidecar graph declares
     * `PREFIX_TERMINAL_HIDDEN` read-only. This production assertion closes the
     * other half of the invariant: the mixed-vendor runner must actually build
     * that graph and retain the exact typed publication lease across a live
     * resident-logical-state append. Follower ranks own sparse participants,
     * not the public MTP response, so the continuation authority alone judges
     * these process-local counters after the serving command loop has closed.
     */
    void assertMTPTerminalHiddenMailboxEvidence() const;

    /**
     * @brief Prove TP>2 MTP executed the captured canonical reduction route.
     *
     * The typed graph policy and byte-exact grouped-vs-serial CSV comparisons
     * remain the arithmetic authorities. PerfStats is used only as route
     * evidence: it proves the live retained graph actually launched the native
     * allgather plus ascending-rank device fold selected by that policy.
     */
    void assertCanonicalTPAllreduceRouteEvidence() const;

    /**
     * @brief Fold the identical post-loop evidence sequence on every MPI rank.
     *
     * Worker ranks enter this sequence immediately after receiving either the
     * typed terminal SHUTDOWN or nonterminal retained-runner YIELD command.
     * The continuation authority calls it in the same order after closing the
     * loop so no test-only collective can race the production command channel.
     */
    void assertEvidenceAfterWorkerLoopExit();

    /**
     * @brief Run the complete three-tier graph-native parity contract once.
     */
    void runGraphNativeProductionParityBody();

    std::shared_ptr<MoERoutedExpertPlacementPlan> overlay_plan_;
    ClusterInventory cluster_inventory_;
    /** Authenticated route demand used only to certify epoch-one ordering. */
    std::vector<std::vector<std::uint64_t>> reference_adversarial_routes_;
    /** Reference prefill demand used to require a witnessable promotion. */
    std::vector<std::vector<std::uint64_t>>
        authenticated_movement_routes_;
    /** Exact CPU tier index retained across deferred capacity resolution. */
    int reference_adversarial_cpu_tier_index_ = -1;
    /** Dense continuation rank used to classify remote CPU ownership. */
    int reference_adversarial_continuation_rank_ = -1;
    DynamicResidencyProofLifecycle dynamic_residency_proof_lifecycle_;
    ResidencyConvergenceTimings convergence_timings_;
    /**
     * Physical wave width selected from authenticated model geometry.
     *
     * CPU-tier convergence cells replace the minimum with one tier-migration
     * slot per transformer layer plus one independent within-tier skew slot.
     * GPU-only cells retain the bounded minimum until their own performance
     * convergence proof derives an equivalent model-owned width.
    */
    std::uint32_t convergence_migration_transfer_slots_ = 0u;
    /** Physical GPU queue pool retained independently from logical slots. */
    std::uint32_t convergence_migration_execution_streams_ = 0u;
    /** Active policy cap that may deliberately use only part of the fabric. */
    std::uint32_t convergence_migration_cycles_per_wave_ = 0u;
    /** True only after the command protocol published an idle YIELD edge. */
    bool campaign_runner_yielded_for_reuse_ = false;
    std::vector<uint64_t>
        parity_route_counts_by_participant_; ///< Live checkpoint routes under the published epoch.
    std::vector<bool>
        parity_route_requires_remote_completion_; ///< Participants requiring a follower Complete proof.
    std::vector<PublishedParticipantResidency>
        parity_residency_by_participant_; ///< Device-bank authority for resident versus capacity-idle endpoints.
    std::vector<PromotedExpert>
        promoted_experts_; ///< Promotion identities retained across parity collector resets.
    std::vector<PromotedExpertExecutionWitness>
        promoted_expert_execution_witnesses_; ///< Exact parity routes through promoted destinations.
    std::vector<RoutedExpertContributionWitness>
        routed_expert_contribution_witnesses_; ///< Every comparable route at a physically moved layer.
    /** Generic sidecar comparisons projected into the optional Qwen CSV. */
    std::vector<MTPCheckpointDiagnostic> mtp_checkpoint_diagnostics_;
    /** Grouped-main HF and serial-row comparisons from the canonical request. */
    std::vector<MTPNumericalDiagnostic> mtp_numerical_diagnostics_;
    /** Full tensors retained only for failed MTP comparisons. */
    std::vector<MTPFailureDiagnostic> mtp_failure_diagnostics_;
    /** Bounded serial M=1 row paired with grouped verifier row zero. */
    std::optional<MTPSerialVerifierDiagnostic>
        mtp_serial_verifier_diagnostic_;
    /** Exact generic production transaction reused by Qwen diagnostics. */
    std::optional<ProductionParityMTPTransactionBoundary>
        mtp_transaction_diagnostic_;
    /** Device-authenticated placement epoch that executed grouped verification. */
    uint64_t mtp_grouped_execution_epoch_ = 0u;
    /** Top-1 identities retained while the primary sidecar bank is live. */
    std::optional<int> mtp_primary_production_top1_;
    std::optional<int> mtp_primary_reference_top1_;
};

}
