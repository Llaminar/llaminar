/**
 * @file Test__PipelineDomainGeneration.cpp
 * @brief Real captured inter-vendor generation with native TP inside each domain.
 *
 * The tiny numerical graph uses production embedding, KV and sampling kernels.
 * Its independent four-token distribution detects bad metadata/activation
 * handoffs without loading a model. Public request APIs exercise first admission,
 * continuation and reset. Main forwards must be complete before serving;
 * request-specific helper captures must survive reset. Shallow Dynamic first
 * transactions must cold-materialize the complete retained depth-15 helper
 * family without executing padded initial work. Initial1 repeats this proof
 * with both stochastic and greedy laws. Penalized target requests additionally
 * compare fresh retained-capacity Off execution with MTP after the identical
 * complete serving setup. Stochastic one-hot proposals remain unpenalized;
 * greedy proposals retain request penalties. Direct-output cases additionally
 * prove exactly-once forced and sampled history, terminal-control-only forced
 * stops, and carried continuation without duplicate publication. Nonterminal
 * force results independently prove actual main KV rows from returned tokens
 * and their explicit Pending/Committed disposition, including greedy cases
 * whose coincident output could hide a dropped model row. This is model-free
 * preflight. Ordinary cross-vendor cases use the original MPI rank service;
 * each process constructs only the GPU cohort on its observed NUMA node.
 * Every worker is born from that rank's original native source, whose same PMA
 * admits the complete graph/transport BOM before any collective or graph owner
 * borrows it. Ordinary pool workers cannot certify native TP fanout ownership.
 * Three-stage cases retain those same two original ranks: one owns distinct
 * entry/tail CUDA GPUs and the other owns the middle ROCm GPU. Both directed
 * cross-rank boundaries retain exact backing; the source rank must execute its
 * local entry follower and terminal graph through continuation and reset.
 * Adjacent same-backend domains require native composition and are not selected
 * by this captured cross-vendor fixture.
 */
#include "PipelineGenerationTestSupport.h"
#include "../../../utils/ObservedISAPipelinePlacement.h"
#include "backends/NativeContextPreparation.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "config/OrchestrationStartupPolicy.h"
#include <map>
#include <set>
#include <tuple>
#include <cstdio>
#include <exception>
#include "execution/global/GlobalOrchestrator.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "planning/RankHardwareOwnership.h"
#include "planning/PlanningObservedResource.h"
#include "transfer/CapturedTransferChannel.h"

namespace llaminar2::test
{
/** @brief Each serving policy is selected explicitly and retains its own evidence. */
enum class PipelineGenerationLane { Ordinary, FixedMTP, DynamicMTP };

/** @brief Explicit physical topology and generation lane, never inferred from a name. */
struct PipelineDomainCase
{
    bool reverse;
    int width;
    PipelineGenerationLane lane;
    /** @return Whether this case executes speculative verification. */
    bool usesMTP() const noexcept { return lane != PipelineGenerationLane::Ordinary; }
    int admitted_rows = 256; ///< Exact PMA-selected arena capacity, independent of startup ceiling.
    int initial_depth = 15; ///< Dynamic first transaction; retained capacity remains fifteen.
    bool greedy = false; ///< Explicit sampling axis; existing cases remain stochastic.
    int stage_count = 2; ///< Three selects distinct CUDA entry/tail GPUs and a remote ROCm middle.
};

/** @brief Shared numerical fixture with explicit one-, two- and four-member TP domains. */
class PipelineDomainGeneration : public ::testing::TestWithParam<PipelineDomainCase>
{
protected:
    /** @brief Separate model forwards from the request-specific helper family. */
    enum class CaptureScope { MainForward, CompleteGeneration };

    /**
     * @brief Count nodes at the native forward-capture boundary for each device.
     * @param scope Main-model serving family or all generation helpers as well.
     * @return Monotonic capture totals, including retained child templates.
     *
     * Setup's semantic family counter alone cannot expose a request-time cache
     * miss. These counters are emitted by the native capture controller after
     * recording, so any later recording changes the observed inventory.
     */
    static std::map<std::string, double> capturedForwardNodes(
        CaptureScope scope = CaptureScope::MainForward)
    {
        std::map<std::string, double> result;
        for (const auto &record : PerfStatsCollector::snapshot())
        {
            const auto context = record.tags.find("context");
            if (scope == CaptureScope::MainForward &&
                (context == record.tags.end() ||
                 (!context->second.starts_with("main_") && !context->second.starts_with("prefill"))))
                continue;
            if (record.domain == "forward_graph" &&
                (record.name == "full_graph_capture_executable_nodes" ||
                 record.name == "segmented_graph_capture_executable_nodes" ||
                 record.name == "retained_parent_child_graph_nodes"))
                result[record.device] += record.value;
        }
        return result;
    }

    /**
     * @brief Require both verifier outcomes at every admitted request width.
     * @param participants Actual GPU owners whose setup has completed.
     * @param enabled Whether this case retains the depth-15 MTP family.
     */
    static void expectRetainedVerifierSetup(
        const std::vector<DeviceGraphOrchestrator *> &participants, bool enabled)
    {
        using Key = std::tuple<std::string, std::string, std::string>;
        std::map<Key, double> expected, actual;
        if (enabled)
            for (const auto *participant : participants)
            {
                const auto execution = participant->declaredDeviceGenerationExecutionPolicy();
                EXPECT_EQ(execution, DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions);
                const auto family = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(execution));
                const auto count = mtpVerifierGraphFamilyWidthCount(16, family);
                for (std::size_t index = 0; index < count; ++index)
                {
                    const int rows = mtpVerifierGraphFamilyWidthAt(index, 16, family);
                    for (const auto outcome : {MTPVerifierOutcomeGraphMode::Greedy,
                                               MTPVerifierOutcomeGraphMode::Disabled})
                        expected[{participant->primaryDeviceId().toString(), std::to_string(rows),
                            std::to_string(static_cast<int>(outcome))}] = 1;
                }
            }
        for (const auto &record : PerfStatsCollector::snapshot())
        {
            if (record.domain != "forward_graph" || record.name != "serving_graph_family_materializations")
                continue;
            const auto role = record.tags.find("role");
            if (role == record.tags.end() || role->second != "mtp_grouped_verifier") continue;
            EXPECT_EQ(record.phase, "setup");
            EXPECT_EQ(record.tags.at("sparse_query_program"), std::to_string(
                sparseQueryExecutionProgramIndex(SparseQueryExecutionProgram::RankingPossible)));
            EXPECT_EQ(record.tags.at("submission"), "materialized_unlaunched");
            actual[{record.device, record.tags.at("physical_rows"), record.tags.at("outcome_mode")}] += record.value;
        }
        EXPECT_EQ(actual, expected) << "All request widths must exist before the first prompt is admitted.";
    }

    /** @brief Request sampling axis; every original case keeps its original neutral oracle. */
    enum class RequestLaw { Neutral, PresencePenalty };
    /** @brief Separate full-response and direct-publication lifecycle assertions. */
    enum class RequestScenario { FullResponse, DirectHistoryBoundaries };

    /**
     * @brief Execute exact public requests after complete retained serving setup.
     * @param request_law Explicit target policy; penalized cases use a real Off
     *        request as the serial oracle rather than duplicating penalty math.
     * @param scenario Full requests or focused force/stop/continuation boundaries.
     */
    void run(RequestLaw request_law = RequestLaw::Neutral,
             RequestScenario scenario = RequestScenario::FullResponse)
    {
        const auto test = GetParam();
        const bool penalized = request_law == RequestLaw::PresencePenalty;
        ASSERT_TRUE(test.stage_count == 2 || test.stage_count == 3);
        if (test.stage_count == 3)
        {
            ASSERT_FALSE(test.reverse);
            ASSERT_EQ(test.width, 1);
        }
        const int terminal_domain = test.stage_count - 1;
        const int fixed_depth = penalized ? 1 : 7;
        initCPUBackend(-1);
        const auto mpi = MPIContextFactory::global();
        ASSERT_TRUE(mpi);
        ASSERT_EQ(mpi->world_size(), 2);
        const auto inventory = mpi->clusterInventory();
        ASSERT_TRUE(inventory);
        struct Startup
        {
            std::optional<std::string> previous;
            /** @brief Publish the same bucket policy as the frontend. */
            Startup()
            {
                if (const char *value = std::getenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES")) previous = value;
                OrchestrationConfig config;
                config.prefill_max_bucket_size = 256;
                publishOrchestrationStartupPolicy(config);
            }
            /** @brief Restore policy after every graph and worker has retired. */
            ~Startup()
            {
                if (previous) setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", previous->c_str(), 1);
                else unsetenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES");
                mutableDebugEnv().reload();
            }
        } startup;
        MTPRuntimeConfig mtp;
        mtp.enabled = test.usesMTP();
        mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
        if (test.usesMTP())
        {
            mtp.graph_capacity_draft_tokens = 15;
            // Fixed execution deliberately retains the same depth-15 graph
            // capacity as dynamic execution; seven is its active depth only.
            mtp.draft_tokens = test.lane == PipelineGenerationLane::DynamicMTP && test.initial_depth < 15
                ? test.initial_depth : fixed_depth; // Original cases retain their authored depth seven.
            mtp.depth_policy.mode = test.lane == PipelineGenerationLane::DynamicMTP
                ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed;
            mtp.depth_policy.initial_depth = test.lane == PipelineGenerationLane::DynamicMTP ? test.initial_depth : fixed_depth;
            mtp.depth_policy.max_depth = 15;
            mtp.depth_policy.window_size = 1;
            mtp.depth_policy.min_samples = 1;
            mtp.depth_policy.cooldown_steps = 0;
        }
        OrdinaryGraphEvidence evidence("generation,forward_graph,mtp,transfer");
        ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("transfer"))
            << "Zero-copy evidence requires the transfer collection domain";
        // This owner predates every graph, context, arena and table borrower.
        // Its final reset follows destruction of all those actual native uses.
        struct OriginalSourceLifetime final
        {
            std::shared_ptr<NativeContextPreparation> source;
            /** @brief Retire only after all later-declared graph/context/storage owners are gone. */
            ~OriginalSourceLifetime() noexcept
            {
                if (!source) return;
                try { source->retireReleasedContexts(); }
                catch (...) { std::terminate(); }
            }
        } original_source;
        std::vector<std::vector<GlobalDeviceAddress>> members(test.stage_count);
        std::vector<std::unique_ptr<LocalTPContext>> contexts(test.stage_count);
        std::vector<DevicePlanConfig> inputs;
        // Resolve the full original-world stage demand jointly. A second
        // individually eligible ROCm owner is legitimate inventory, while this
        // distributed fixture requires distinct native backend owners and
        // disjoint devices for repeated stages on the same original owner.
        std::vector<ObservedNativePipelineStage> request;
        for(int domain=0;domain<test.stage_count;++domain) {
            const auto backend=test.stage_count==3
                ? (domain==1?DeviceType::ROCm:DeviceType::CUDA)
                : (bool(domain)==test.reverse?DeviceType::CUDA:DeviceType::ROCm);
            request.push_back({backend,static_cast<std::size_t>(test.width)});
        }
        const auto physical=ObservedCrossVendorPipelinePlacement::require(*inventory,request);
        const auto& owners=physical.owners;
        ASSERT_EQ(static_cast<std::size_t>(mpi->world_size()),physical.selection().discoveryRanks().size());
        for(int domain=0;domain<test.stage_count;++domain) {
            const auto& owner=inventory->getRank(owners[domain]);
            for(const auto device:physical.stages[domain])
                members[domain].push_back(GlobalDeviceAddress::fromLocalDeviceId(
                    device,owner.hostname,owner.cpu.numa_node));
        }
        ASSERT_NE(owners[0], owners[1]);
        if (test.stage_count == 3)
        {
            ASSERT_EQ(owners[0], owners[2]);
            ASSERT_NE(members[0].front().toLocalDeviceId(), members[2].front().toLocalDeviceId());
        }
        std::vector<int> local_domains;
        for (int domain = 0; domain < test.stage_count; ++domain)
            if (owners[domain] == mpi->rank()) local_domains.push_back(domain);
        ASSERT_FALSE(local_domains.empty());
        const bool source = owners.back() == mpi->rank();
        if (test.stage_count == 3) ASSERT_EQ(local_domains.size(), source ? 2u : 1u);
        for (int domain = 0; domain < test.stage_count; ++domain)
        {
            if (owners[domain] != mpi->rank()) continue;
            const bool is_cuda = members[domain].front().toLocalDeviceId().is_cuda();
            for (int member = 0; member < test.width; ++member)
            {
                const auto device = members[domain][member].toLocalDeviceId();
                const int ordinal = device.ordinal;
                DevicePlanConfig input;
                input.world_rank = mpi->rank();
                input.device = device;
                input.first_layer = input.last_layer = domain;
                const auto &observed_rank = inventory->getRank(mpi->rank());
                const auto resource = planningObservedResource(observed_rank, device);
                const auto observed_gpu = std::find_if(observed_rank.gpus.begin(), observed_rank.gpus.end(),
                    [&](const auto &gpu) { return gpu.type == device.type && gpu.local_device_id == ordinal; });
                ASSERT_NE(observed_gpu, observed_rank.gpus.end());
                input.device_total_bytes = resource.total_bytes;
                input.device_free_bytes = resource.admission_available_bytes;
                input.associated_host_memory = planningObservedResource(observed_rank, DeviceId::cpu());
                input.device_compute_units = observed_gpu->compute_units;
                input.pipeline_forward_only_graph_owner = PipelineForwardOnlyGraphOwner::Participant;
                input.pipeline_native_activation_fanout = test.width > 1 && domain > 0;
                input.max_seq_len = 512;
                input.activation_seq_len = test.admitted_rows;
                input.mtp_enabled = test.usesMTP();
                input.mtp_target_query_rows = test.usesMTP() ? 16 : 1;
                const auto execution = declaredDeviceGenerationExecutionPolicy(
                    device.type, DeviceGenerationExecutionDomain::HostedBoundary);
                const auto family = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(execution));
                input.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory(
                    {test.admitted_rows},
                    mtp,
                    family,
                    domain != terminal_domain ? MTPGenerationParentRole::PipelineFollower : MTPGenerationParentRole::Controller,
                    DeviceGenerationExecutionDomain::HostedBoundary);
                if (test.width > 1) input.local_tp_backend = is_cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                if (!member)
                {
                    if (domain > 0) input.captured_pipeline_boundaries.push_back(PipelineBoundarySide::LaterDomain);
                    if (domain < terminal_domain) input.captured_pipeline_boundaries.push_back(PipelineBoundarySide::EarlierDomain);
                }
                inputs.push_back(input);
            }
        }
        ModelMemoryProfile profile;
        profile.architecture = "qwen35";
        profile.n_layers = test.stage_count + (test.usesMTP() ? 1 : 0);
        profile.mtp_layer_count = test.usesMTP() ? 1 : 0;
        profile.d_model = profile.head_dim = profile.vocab_size = 32;
        profile.d_ff = 64;
        profile.n_heads = profile.n_kv_heads = 1;
        profile.max_seq_len = 512;
        const auto &original_rank = inventory->getRank(mpi->rank());
        std::vector<PhysicalMemoryResource> observations{
            planningObservedResource(original_rank, DeviceId::cpu())};
        std::vector<DeviceId> selected;
        for (const auto &input : inputs)
        {
            observations.push_back(planningObservedResource(original_rank, input.device));
            selected.push_back(input.device);
        }
        original_source.source = NativeContextPreparation::create(observations, mpi->rank(), selected);
        const auto memory = original_source.source->memory();
        for (const auto device : selected)
        {
            const auto original = original_source.source->acquire(device);
            const auto retained = GPUDeviceContextPool::instance().retainExistingPreparedContext(device, memory);
            retained.requireWorker(original.worker());
            ASSERT_EQ(original.memory(), memory);
            ASSERT_EQ(retained.generation(), original.generation());
            ASSERT_GT(original.generation(), 0u);
        }
        original_source.source->freezeForPlanning();
        for (auto &input : inputs) input.native_context_preparation = original_source.source;
        const auto certificate = std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
            MemoryPlanner::plan(profile, inputs).physicalPlan());
        original_source.source->admitFinal(certificate);
        ASSERT_EQ(original_source.source->memory(), memory);
        ASSERT_EQ(memory->admission(), certificate);
        // Native collectives may now borrow only the already-created original
        // local workers and the same final model/transport memory authority.
        if (test.width > 1)
            for (const int domain : local_domains)
                contexts[domain] = std::make_unique<LocalTPContext>(members[domain],
                    std::vector<float>{}, CollectiveBackendType::AUTO);
        std::vector<std::shared_ptr<OrdinaryDGOGraph>> graphs;
        std::vector<DeviceGraphOrchestrator *> participants;
        std::vector<int> participant_domains; ///< Exact stage identity for each original local leaf.
        std::vector<std::unique_ptr<IInferenceRunner>> domains;
        for (int domain = 0; domain < test.stage_count; ++domain)
        {
            if (owners[domain] != mpi->rank()) continue;
            if (contexts[domain]) ASSERT_TRUE(contexts[domain]->reserveGraphCaptureBoundaryResources(memory));
            std::vector<std::unique_ptr<IInferenceRunner>> children;
            for (int member = 0; member < test.width; ++member)
            {
                const auto device = members[domain][member].toLocalDeviceId();
                GraphConfig config;
                config.d_model = config.head_dim = config.vocab_size = config.vocab_local = 32;
                config.d_ff = config.d_ff_local = 64;
                config.n_heads = config.local_n_heads = config.n_kv_heads = config.local_n_kv_heads = 1;
                config.n_layers = 1;
                config.total_n_layers = profile.n_layers;
                config.pp_layer_offset = domain;
                config.layer_types.assign(profile.n_layers, "full_attention");
                // The shared hybrid schema declares both attention families,
                // even when this numerical probe only executes full attention.
                config.gdn.conv_kernel_size = 4;
                config.gdn.state_size = config.head_dim;
                config.gdn.inner_size = config.d_model;
                config.gdn.group_count = config.n_kv_heads;
                config.gdn.time_step_rank = config.n_heads;
                config.default_device = device;
                config.max_seq_len = 512;
                config.kv_cache_precision = KVCachePrecision::FP16;
                config.mtp = mtp;
                config.mtp_request_terminal_hidden_publication = MTPRequestTerminalHiddenPublicationPolicy::GraphCapturedDeviceGeometry;
                config.mtp_shifted_prefill_hidden_publication = MTPShiftedPrefillHiddenPublicationPolicy::GraphIntegratedKVTransaction;
                config.mtp_verifier_outcome_ownership = MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal;
                config.tp_ctx = contexts[domain].get();
                config.tp_device_idx = config.local_rank = member;
                config.lm_head_column_parallel = test.width > 1 && domain == terminal_domain;
                config.prefix_cache.enabled = false;
                config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
                auto graph = std::make_shared<OrdinaryDGOGraph>(config, domain == 0, domain == terminal_domain);
                auto owner = DeviceGraphOrchestrator::createForTest({.model_ctx = MockModelContext::createMinimal(),
                    .graph_builder = graph, .physical_memory_authority = memory});
                FactoryPPStageConfig stage{.first_layer = domain, .last_layer = domain + 1,
                    .has_embedding = domain == 0, .has_lm_head = domain == terminal_domain};
                stage.generation_domain = DeviceGenerationExecutionDomain::HostedBoundary;
                owner->setPPStageConfig(stage);
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                context.submitAndWait([&] {
                    ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
                    TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
                    ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 512, device,
                        {.activation_seq_len = test.admitted_rows}));
                });
                ASSERT_FALSE(HasFatalFailure());
                if (test.usesMTP() && domain == terminal_domain) owner->setFrozenWeightSet(makeOrdinaryMTPFrozenWeightSet(*graph, device));
                participants.push_back(owner.get());
                participant_domains.push_back(domain);
                graphs.push_back(graph);
                children.push_back(std::move(owner));
            }
            if (test.width == 1) domains.push_back(std::move(children.front()));
            else
            {
                RankOrchestrator::Config config;
                config.mode = RankOrchestrator::ParallelismMode::TP;
                config.devices = members[domain];
                config.max_seq_len = 512;
                config.mtp = mtp;
                config.nested_pp_stage_config = FactoryPPStageConfig{.first_layer = domain, .last_layer = domain + 1,
                    .has_embedding = domain == 0, .has_lm_head = domain == terminal_domain};
                config.nested_pp_stage_config->generation_domain = DeviceGenerationExecutionDomain::HostedBoundary;
                auto domain_model = MockModelContext::createMinimal();
                domain_model->setVocabSize(32);
                domains.push_back(RankOrchestrator::createForTest(std::move(domain_model),
                    std::move(children), std::move(contexts[domain]), config));
            }
        }

        std::vector<GlobalPPStageSpec> specifications;
        for (int domain = 0; domain < test.stage_count; ++domain)
        {
            GlobalPPStageSpec stage;
            stage.stage_id = stage.first_layer = stage.last_layer = domain;
            stage.domain_name = "ordinary_domain_" + std::to_string(domain);
            stage.has_embedding = domain == 0;
            stage.has_lm_head = domain == terminal_domain;
            stage.owning_rank = owners[domain];
            stage.inner_mode = test.width > 1 ? InnerParallelism::LOCAL_TP : InnerParallelism::SINGLE_DEVICE;
            stage.devices = members[domain];
            stage.backend = members[domain].front().toLocalDeviceId().is_cuda()
                ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            specifications.push_back(std::move(stage));
        }
        GlobalOrchestrator::Config global;
        global.topology = GlobalPPTopology::build(std::move(specifications), test.stage_count, *inventory);
        global.rank = mpi->rank();
        global.world_size = mpi->world_size();
        global.mpi_ctx = mpi.get();
        global.original_mpi_ctx = mpi;
        global.vocab_size = global.d_model = 32;
        global.architecture_name = "qwen35";
        const auto local_plan = GlobalPPRankPlanBuilder::build(global.topology, mpi->rank());
        const auto local_actions = local_plan.executeStages();
        ASSERT_EQ(local_actions.size(), local_domains.size());
        ASSERT_EQ(domains.size(), local_domains.size());
        for (size_t local = 0; local < local_actions.size(); ++local)
        {
            ASSERT_EQ(local_actions[local]->stage_id, local_domains[local]);
            StageRunnerEntry entry;
            entry.action = *local_actions[local];
            entry.stage_id = entry.action.stage_id;
            entry.domain_name = entry.action.domain_name;
            entry.original_mpi_ctx = mpi;
            entry.runner = std::move(domains[local]);
            global.stage_runners.push_back(std::move(entry));
        }
        auto pipeline_owner = std::make_unique<GlobalOrchestrator>(std::move(global));
        auto *pipeline = pipeline_owner.get();
        pipeline->validateOriginalRankGPUTransportContext(mpi);
        struct Lifetime final
        {
            std::shared_ptr<CapturedTransferRankSession> session;
            std::unique_ptr<IInferenceRunner> prepared;
            std::unique_ptr<OrchestrationRunner> runner;
            std::shared_ptr<IMPIContext> original_context; ///< Exact original owner for fatal asymmetric failure.
            /** @brief Abort divergent failure before any original-rank cleanup collective. */
            [[noreturn]] void abortOriginal(const char *diagnostic) const noexcept
            {
                std::fprintf(stderr, "Original pipeline fixture rank %d fatal: %s\n",
                    original_context->rank(), diagnostic);
                std::fflush(stderr);
                MPI_Abort(original_context->communicator(), 1);
                std::terminate();
            }
            /** @brief Destroy every native graph/alias before original all-rank backing retirement. */
            ~Lifetime()
            {
                if (::testing::Test::HasFatalFailure())
                {
                    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
                    if (info)
                        for (int index = 0; index < info->result()->total_part_count(); ++index)
                        {
                            const auto &part = info->result()->GetTestPartResult(index);
                            if (part.failed()) std::fprintf(stderr, "%s\n", part.message());
                        }
                    abortOriginal("GTest fatal failure precedes collective graph retirement");
                }
                if (std::uncaught_exceptions())
                    abortOriginal("exception escaped reporting boundary before collective graph retirement");
                runner.reset();
                prepared.reset();
                session->retireAfterOriginalGraphRelease();
            }
        } lifetime{CapturedTransferRankSession::create(mpi), std::move(pipeline_owner), nullptr, mpi};
        // Catch while Lifetime is still alive. Reporting after its destructor
        // would hide the original exception inside an unmatched retirement
        // collective while the peer enters its next public request boundary.
        try
        {
        pipeline->bindOriginalRankGPUTransportSession(mpi, lifetime.session);
        const ServingGraphFamilyMaterializationPlan serving{
            .prefill_bucket_rows = {test.admitted_rows}, .prefill_pad_token_id = 0,
            .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial};
        // The injected fixture explicitly executes the same public cold binding
        // and complete family materialization used by production initialization.
        ASSERT_TRUE(pipeline->materializeServingGraphFamilyWithoutLaunch(serving));
        ASSERT_TRUE(pipeline->originalRankPipelineReady());
        expectRetainedVerifierSetup(participants, test.usesMTP());
        const auto captured_before_requests = capturedForwardNodes();
        ASSERT_EQ(captured_before_requests.size(), participants.size());
        for (const auto *participant : participants)
            ASSERT_GT(captured_before_requests.at(participant->primaryDeviceId().toString()), 0);
        const auto geometry = PipelineTransferMemory::forRows(32, test.admitted_rows, test.usesMTP() ? 16 : 1);
        size_t expected_host_bytes = 0;
        for (size_t index = 0; index < participants.size(); ++index)
        {
            const int domain = participant_domains[index];
            const bool leader = participants[index]->primaryDeviceId() == members[domain].front().toLocalDeviceId();
            const size_t adjacent_boundaries = size_t(domain > 0) + size_t(domain < terminal_domain);
            const bool receives_native_fanout = domain > 0 && test.width > 1;
            EXPECT_EQ(memory->claimedBytes(participants[index]->primaryDeviceId(), PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
                (leader ? adjacent_boundaries * geometry.device_bytes_per_boundary : 0) +
                    (receives_native_fanout ? PipelineTransferMemory::fanoutDeviceBytes() : 0));
            if (leader && domain < terminal_domain) expected_host_bytes += geometry.host_bytes_per_boundary;
            if (receives_native_fanout) expected_host_bytes += PipelineTransferMemory::fanoutHostBytes();
        }
        // Every rank owns its original source's CPU obligations. Channel
        // pages still belong only to each earlier stage: CUDA entry and ROCm
        // middle. Native fanout owns only its exact per-member operands.
        // CPU allocator presence cannot substitute for a zero transport claim.
        ASSERT_TRUE(memory->contains(DeviceId::cpu()));
        EXPECT_EQ(memory->plannedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging), expected_host_bytes);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), expected_host_bytes);
        if (test.stage_count == 3)
        {
            std::map<int, double> actual_boundaries, expected_boundaries;
            for (int boundary = 0; boundary < terminal_domain; ++boundary)
                if (owners[boundary] == mpi->rank() || owners[boundary + 1] == mpi->rank())
                    expected_boundaries[boundary] = geometry.activation_capacity_bytes;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "forward_graph" && record.name == "original_rank_pipeline_activation_capacity")
                {
                    const int earlier = std::stoi(record.tags.at("source_stage"));
                    const int later = std::stoi(record.tags.at("consumer_stage"));
                    ASSERT_GE(earlier, 0);
                    ASSERT_LT(earlier, terminal_domain);
                    ASSERT_EQ(later, earlier + 1);
                    EXPECT_EQ(std::stoi(record.tags.at("source_rank")), owners[earlier]);
                    EXPECT_EQ(std::stoi(record.tags.at("consumer_rank")), owners[later]);
                    EXPECT_EQ(std::stoi(record.tags.at("local_rank")), mpi->rank());
                    EXPECT_EQ(record.tags.at("extent_kind"), "storage_capacity_only");
                    actual_boundaries[earlier] += record.value;
                }
            EXPECT_EQ(actual_boundaries, expected_boundaries)
                << "Both original cross-rank edges require distinct actual admitted channels";
        }
        OrchestrationConfig config;
        config.max_seq_len = 512;
        config.prefill_max_bucket_size = test.admitted_rows;
        config.mtp = mtp;
        config.prefix_cache.enabled = false;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        RankExecutionPlan plan;
        plan.rank = mpi->rank();
        plan.hostname = inventory->getRank(mpi->rank()).hostname;
        plan.primary_device = members[source ? terminal_domain : local_domains.front()].front();
        plan.has_lm_head = source;
        plan.has_embedding = owners.front() == mpi->rank();
        plan.runtime.max_seq_len = 512;
        plan.runtime.resident_graph_rows = test.admitted_rows;
        plan.runtime.mtp = mtp;
        plan.runtime.prefix_cache = config.prefix_cache;
        lifetime.runner = std::make_unique<OrchestrationRunner>(config, plan, std::move(lifetime.prepared), mpi);
        auto &runner = *lifetime.runner;
        ASSERT_EQ(runner.coordinatedRootRank(), owners.back());
        int consumed_main_rows = 0;
        /** @brief Validate completed public response cardinality without broadcasting token/model state. */
        const auto terminal_count = [&](const GenerationResult &result, int expected) {
            const int32_t count = source ? static_cast<int32_t>(result.tokens.size()) : 0;
            std::array<int32_t, 2> counts{};
            mpi->allgather_bytes(&count, counts.data(), sizeof(count));
            for (int rank = 0; rank < mpi->world_size(); ++rank)
                EXPECT_EQ(counts[rank], rank == owners.back() ? expected : 0);
            EXPECT_EQ(result.tokens.size(), source ? size_t(expected) : 0u);
            return counts[owners.back()];
        };

        const bool shallow_initial = test.lane == PipelineGenerationLane::DynamicMTP && test.initial_depth < 15;
        // These terminal tags authenticate a successfully committed first
        // transaction, rather than merely repeating the requested policy.
        const auto firstTransactionEvidence = [&] {
            // External drafts, all completions, matching authored starts,
            // and completions entered through an authenticated admission ticket.
            std::array<double, 5> counts{};
            for (const auto &record : PerfStatsCollector::snapshot())
            {
                if (record.domain != "mtp" || record.phase != "decode") continue;
                if (record.name == "draft_steps") counts[0] += record.value;
                if (record.name == "device_generation_continuation_admission_tickets")
                {
                    EXPECT_EQ(record.tags.at("entry_source"), "native_continue_response_admission");
                    EXPECT_EQ(record.tags.at("family"), "exact_rows");
                    const auto depth = std::stoi(record.tags.at("depth"));
                    EXPECT_GE(depth, 1);
                    EXPECT_LE(depth, 15);
                    EXPECT_EQ(std::stoi(record.tags.at("physical_rows")), depth + 1);
                    EXPECT_EQ(record.sequence_minimum_words.size(), 3u);
                    EXPECT_EQ(record.sequence_maximum_words.size(), 3u);
                    if (record.sequence_minimum_words.size() != 3u || record.sequence_maximum_words.size() != 3u)
                        continue;
                    EXPECT_EQ(record.sequence_word_count, record.count * 3u);
                    EXPECT_EQ(record.sequence_minimum_words[0], 0u);
                    EXPECT_EQ(record.sequence_maximum_words[0], 0u);
                    for (std::size_t word = 1; word < 3; ++word)
                    {
                        EXPECT_GT(record.sequence_minimum_words[word], 0u);
                        EXPECT_GE(record.sequence_maximum_words[word], record.sequence_minimum_words[word]);
                    }
                    EXPECT_EQ(record.tags.count("session_epoch"), 0u);
                    EXPECT_EQ(record.tags.count("workspace_generation"), 0u);
                    if (test.greedy && test.initial_depth == 1)
                        EXPECT_GT(depth, 1) << "This fixture must actually retain an adapted depth at continuation.";
                    counts[4] += record.value;
                }
                if (record.name == "device_generation_retained_continuation_requests")
                {
                    EXPECT_EQ(record.tags.at("first_transaction_source"), "admission_ticket");
                    EXPECT_EQ(record.tags.at("sampling"), deviceGenerationSamplingModeName(test.greedy
                        ? DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic));
                    EXPECT_EQ(record.tags.count("initial_transaction_depth"), 0u);
                    counts[3] += record.value;
                }
                if (record.name != "device_resident_generation_requests") continue;
                counts[1] += record.value;
                const auto initial = record.tags.find("initial_transaction_depth");
                if (initial == record.tags.end())
                {
                    EXPECT_EQ(record.tags.at("first_transaction_source"), "admission_ticket");
                    continue;
                }
                EXPECT_EQ(record.tags.at("first_transaction_source"), "external");
                if (initial->second == std::to_string(test.initial_depth) &&
                    record.tags.at("capture_depth") == "15" &&
                    record.tags.at("execution_policy") == deviceGenerationExecutionPolicyName(
                        DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions) &&
                    record.tags.at("sampling") == deviceGenerationSamplingModeName(test.greedy
                        ? DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic))
                    counts[2] += record.value;
            }
            return counts;
        };
        std::optional<std::map<std::string, double>> retained_generation_captures;
        const auto active_request_policy = runner.mtpRequestPolicy();
        std::array<std::vector<int32_t>, 2> serial_controls;
        /**
         * Observe after a normal decode terminal or on the actual asynchronous
         * force/publication producer stream, whose copy completion event is joined.
         * Actual returned tokens (or explicitly staged diagnostic publications)
         * define the independent generated-count oracle;
         * prompt membership and sampling decisions are not reconstructed here.
         * Checking every chunk catches loss of ordinary output on first MTP
         * admission, and catches reset/carry duplication on later continuations.
         */
        const auto expect_generated_history = [&](const char *execution, int repetition,
                                                   int budget, const std::vector<int32_t> &tokens)
        {
            std::array<int32_t, 32> expected{};
            for (const auto token : tokens)
            {
                ASSERT_GE(token, 0);
                ASSERT_LT(token, int(expected.size()));
                ++expected[size_t(token)];
            }
            for (size_t participant = (source ? size_t{0} : participants.size());
                 participant < participants.size(); ++participant)
            {
                auto &owner = *participants[participant];
                const auto device = owner.primaryDeviceId();
                const auto &buffers = owner.inferenceState().extension_buffers;
                const auto found = buffers.find(BufferId::MTP_GENERATED_TOKEN_COUNTS);
                ASSERT_NE(found, buffers.end());
                ASSERT_NE(found->second, nullptr);
                ASSERT_NE(found->second->gpu_data_ptr(), nullptr);
                ASSERT_GE(found->second->size_bytes(), expected.size() * sizeof(int32_t));
                std::array<int32_t, 32> encoded{};
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                context.submitAndWait([&]
                {
                    auto *backend = getBackendFor(device);
                    ASSERT_NE(backend, nullptr);
                    // forceDecodeToken returns an asynchronous publication,
                    // whereas decodeStep has already joined its terminal. Read
                    // on the actual target/history producer, or the ordinary
                    // forced mailbox producer, so this diagnostic cannot race
                    // a valid write on a different stream.
                    const auto sampled = owner.deviceStochasticTargetSampleProducerSlot(0);
                    const auto logical = owner.deviceResidentLogicalSequenceState();
                    void *const producer_stream = sampled.valid() ? sampled.stream :
                        logical.coversRequest(0) ? logical.stream : context.defaultStream();
                    ASSERT_NE(producer_stream, nullptr);
                    ASSERT_TRUE(backend->deviceToHost(encoded.data(), found->second->gpu_data_ptr(),
                        encoded.size() * sizeof(int32_t), device.gpu_ordinal(), producer_stream));
                });
                ASSERT_FALSE(HasFatalFailure());
                for (size_t token = 0; token < expected.size(); ++token)
                {
                    ASSERT_GE(encoded[token], 0);
                    EXPECT_EQ(sampling_math::TokenPenaltyHistory::generatedCount(encoded[token]), expected[token])
                        << "execution=" << execution << " repetition=" << repetition << " budget=" << budget
                        << " device=" << device.toString() << " token=" << token
                        << " actual_returned_tokens=" << tokens.size();
                }
            }
        };
        if (scenario == RequestScenario::DirectHistoryBoundaries)
        {
            ASSERT_TRUE(penalized);
            ASSERT_TRUE(test.usesMTP());
            int direct_publications = 0;
            double completed_forward_only_transactions = 0;
            // Admission owns one flat helper per sampling participant. Counts
            // survive every request reset; Off and forced stops must not replay it.
            const auto expect_direct_history_evidence = [&]
            {
                std::map<std::string, double> materializations, publications, native_nodes, forward_only_materializations;
                completed_forward_only_transactions = 0;
                for (const auto &record : PerfStatsCollector::snapshot())
                {
                    if (record.domain == "mtp" && record.name == "direct_token_history_graph_materializations")
                    {
                        EXPECT_EQ(record.phase, "graph_setup");
                        EXPECT_EQ(record.tags.at("target_slot"), "0");
                        EXPECT_EQ(record.tags.at("initial_submission"), "materialize_without_launch");
                        materializations[record.device] += record.value;
                    }
                    else if (record.domain == "mtp" && record.name == "direct_token_history_publications")
                    {
                        EXPECT_EQ(record.phase, "decode");
                        EXPECT_EQ(record.tags.at("target_slot"), "0");
                        EXPECT_EQ(record.tags.at("source"), "resident_sample");
                        publications[record.device] += record.value;
                    }
                    else if (record.domain == "generation" && record.name == "pipeline_forward_only_graph_materializations")
                    {
                        EXPECT_EQ(record.phase, "graph_setup");
                        EXPECT_EQ(record.tags.at("transport"), "captured_domains");
                        EXPECT_EQ(record.tags.at("executable_owners"), "1");
                        EXPECT_EQ(record.tags.at("sampler"), "absent");
                        forward_only_materializations[record.device] += record.value;
                    }
                    else if (record.domain == "generation" && record.name == "pipeline_forward_only_completed_transactions")
                    {
                        EXPECT_EQ(record.phase, "decode");
                        EXPECT_EQ(record.tags.at("transport"), "captured_domains");
                        EXPECT_EQ(record.tags.at("sampler"), "absent");
                        EXPECT_EQ(record.tags.at("response_tokens"), "0");
                        EXPECT_EQ(record.tags.at("model_commits"), "1");
                        completed_forward_only_transactions += record.value;
                    }
                    else if (record.domain == "forward_graph" && record.name == "full_graph_capture_executable_nodes")
                    {
                        const auto context = record.tags.find("context");
                        if (context != record.tags.end() && context->second == "mtp_direct_token_history_publication")
                            native_nodes[record.device] += record.value;
                    }
                }
                for (size_t participant = 0; participant < participants.size(); ++participant)
                {
                    const auto device = participants[participant]->primaryDeviceId().toString();
                    const bool controller = source;
                    EXPECT_EQ(materializations[device], controller ? 1.0 : 0.0) << device;
                    EXPECT_EQ(forward_only_materializations[device], 1.0)
                        << "Every physical participant retains exactly one sampler-free parent: " << device;
                    EXPECT_EQ(native_nodes[device], controller ? 1.0 : 0.0)
                        << "One retained native kernel, without query/depth/request replicas: " << device;
                    EXPECT_EQ(publications[device], controller ? double(direct_publications) : 0.0) << device;
                }
            };
            expect_direct_history_evidence();
            // Exercise the real typed publication right without a model row.
            // Same token value is not a new event publication. All diagnostic
            // state is reset before the public Off/MTP request comparisons.
            runner.clearCache();
            ASSERT_TRUE(runner.configureMTPRequestPolicy(active_request_policy)) << runner.lastError();
            SamplingParams probe_law;
            probe_law.temperature = test.greedy ? 0.0F : 1.0F;
            probe_law.top_k = 4;
            probe_law.seed = 0xFE000001u;
            probe_law.presence_penalty = 1.5F;
            runner.setSamplingParams(probe_law);
            runner.setStopTokens({});
            ASSERT_TRUE(runner.prefill(std::vector<int>(256, 0))) << runner.lastError();
            using Disposition = MTPDirectTokenHistoryDisposition;
            // Rank treats any failed participant transaction as fatal and
            // aborts its native TP cohort. Exercise expected guard rejection on
            // each original DGO/context instead; positive publications continue
            // through the real public pipeline/Rank fanout below.
            const auto expect_local_rejection = [&](int slot, Disposition disposition, const char *reason)
            {
                for (size_t participant = (source ? size_t{0} : participants.size());
                     participant < participants.size(); ++participant)
                {
                    auto &owner = *participants[participant];
                    const auto device = owner.primaryDeviceId();
                    auto &context = GPUDeviceContextPool::instance().getContext(device);
                    context.submitAndWait([&]
                    {
                        EXPECT_FALSE(owner.publishMTPDirectTokenHistory(slot, disposition))
                            << "device=" << device.toString() << " slot=" << slot << " reason=" << reason;
                    });
                }
            };
            expect_local_rejection(0, Disposition::NewlyEmitted, "fresh prefill has no target event");
            expect_local_rejection(-1, Disposition::NewlyEmitted, "negative slot");
            expect_local_rejection(1, Disposition::NewlyEmitted, "noncanonical slot");
            expect_local_rejection(0, static_cast<Disposition>(255), "unknown disposition");
            expect_generated_history("explicit_publication", 0, 0, {});
            expect_direct_history_evidence();
            ASSERT_FALSE(HasFatalFailure());
            if (source) ASSERT_TRUE(pipeline->stageStochasticTargetTokenForDeviceSampling(4, 0));
            ASSERT_TRUE(pipeline->publishMTPDirectTokenHistory(0, Disposition::NewlyEmitted));
            ++direct_publications;
            expect_generated_history("explicit_publication", 0, 1, {4});
            expect_direct_history_evidence();
            ASSERT_FALSE(HasFatalFailure());
            expect_local_rejection(0, Disposition::NewlyEmitted, "same target event already consumed");
            ASSERT_TRUE(pipeline->publishMTPDirectTokenHistory(0, Disposition::AlreadyPublished));
            expect_generated_history("explicit_publication", 0, 1, {4});
            expect_direct_history_evidence();
            ASSERT_FALSE(HasFatalFailure());
            if (source) ASSERT_TRUE(pipeline->stageStochasticTargetTokenForDeviceSampling(4, 0));
            ASSERT_TRUE(pipeline->publishMTPDirectTokenHistory(0, Disposition::NewlyEmitted));
            ++direct_publications;
            expect_generated_history("explicit_publication", 0, 2, {4, 4});
            expect_direct_history_evidence();
            ASSERT_FALSE(HasFatalFailure());
            runner.clearCache();
            ASSERT_TRUE(runner.prefill(std::vector<int>(256, 0))) << runner.lastError();
            expect_local_rejection(0, Disposition::NewlyEmitted, "reset retired the old event/right");
            expect_generated_history("explicit_publication_reset", 0, 0, {});
            expect_direct_history_evidence();
            ASSERT_FALSE(HasFatalFailure());
            for (int repetition = 0; repetition < 2; ++repetition)
            {
                std::vector<int32_t> ordinary_response;
                for (bool enabled : {false, true})
                {
                    runner.clearCache();
                    auto policy = active_request_policy;
                    policy.enabled = enabled;
                    ASSERT_TRUE(runner.configureMTPRequestPolicy(policy)) << runner.lastError();
                    const char *execution = enabled ? "mtp_direct" : "off_direct";
                    constexpr int prompt_token_count = 256;
                    const auto fresh_request = [&](std::vector<int> stops)
                    {
                        runner.clearCache();
                        SamplingParams law;
                        law.temperature = test.greedy ? 0.0F : 1.0F;
                        law.top_k = 4;
                        law.seed = 0xFE000001u + repetition;
                        law.presence_penalty = 1.5F;
                        law.frequency_penalty = 0.0F;
                        law.repetition_penalty = 1.0F;
                        runner.setSamplingParams(law);
                        runner.setStopTokens(stops);
                        ASSERT_TRUE(runner.prefill(std::vector<int>(prompt_token_count, repetition * 7))) << runner.lastError();
                    };
                    fresh_request({});
                    ASSERT_FALSE(HasFatalFailure());
                    std::vector<int32_t> response;
                    /**
                     * @brief Prove the public forced disposition against actual model rows.
                     * @param forced Successful nonterminal force result for this response.
                     *
                     * Every earlier returned token must now be in main KV. Only
                     * the newly forced token can remain pending, as named by the
                     * actual public force result. This independent count catches
                     * a dropped emitted carry even when greedy output coincides.
                     * Generic decodeStep commit fields are never interpreted here.
                     * Each tail TP participant is observed after its exact existing
                     * model-output completion event; no observation feeds inference.
                     */
                    const auto expect_forced_frontier = [&](const GenerationResult &forced)
                    {
                        ASSERT_TRUE(forced.success()) << forced.error;
                        ASSERT_FALSE(forced.is_complete);
                        ASSERT_TRUE(forced.returned_token_commit == ReturnedTokenCommitState::Pending ||
                            forced.returned_token_commit == ReturnedTokenCommitState::Committed);
                        ASSERT_LE(response.size(), size_t(config.max_seq_len - prompt_token_count));
                        const int expected_cached_tokens = prompt_token_count + int(response.size()) -
                            (forced.returned_token_commit == ReturnedTokenCommitState::Pending ? 1 : 0);
                        const int32_t expected_frontier = source ? expected_cached_tokens : -1;
                        std::array<int32_t, 2> expected_frontiers{};
                        mpi->allgather_bytes(&expected_frontier, expected_frontiers.data(), sizeof(expected_frontier));
                        ASSERT_GE(expected_frontiers[owners.back()], prompt_token_count);
                        for (int rank = 0; rank < mpi->world_size(); ++rank)
                            if (rank != owners.back()) EXPECT_EQ(expected_frontiers[rank], -1);
                        for (size_t participant = 0; participant < participants.size(); ++participant)
                        {
                            auto &owner = *participants[participant];
                            const auto device = owner.primaryDeviceId();
                            auto &context = GPUDeviceContextPool::instance().getContext(device);
                            int32_t actual_cached_tokens = -1;
                            context.submitAndWait([&]
                            {
                                // forceDecodeToken may return an asynchronous
                                // main commit. Join the existing exact output
                                // event before observing the canonical KV word.
                                ASSERT_TRUE(owner.waitForLastInferenceCompletionForBenchmark());
                                auto *backend = getBackendFor(device);
                                ASSERT_NE(backend, nullptr);
                                ASSERT_NE(owner.inferenceState().kv_cache, nullptr);
                                const auto *cached_tokens = owner.inferenceState().kv_cache
                                    ->deviceSequenceCachedTokenCountPtr(0);
                                ASSERT_NE(cached_tokens, nullptr);
                                void *const stream = context.defaultStream();
                                ASSERT_NE(stream, nullptr);
                                ASSERT_TRUE(backend->deviceToHost(&actual_cached_tokens, cached_tokens,
                                    sizeof(actual_cached_tokens), device.gpu_ordinal(), stream));
                            });
                            ASSERT_FALSE(HasFatalFailure());
                            EXPECT_EQ(actual_cached_tokens, expected_frontiers[owners.back()])
                                << "execution=" << execution << " repetition=" << repetition
                                << " device=" << device.toString() << " returned=" << response.size()
                                << " forced_commit=" << int(forced.returned_token_commit);
                        }
                    };
                    // Repeating a forced value proves exactly-once history,
                    // including the existing ordinary forced publisher.
                    for (int index = 0; index < 2; ++index)
                    {
                        const auto forced = runner.forceDecodeToken(4);
                        ASSERT_TRUE(forced.success()) << forced.error;
                        ASSERT_EQ(forced.tokens, source ? std::vector<int32_t>{4} : std::vector<int32_t>{});
                        (void)terminal_count(forced, 1);
                        ASSERT_FALSE(forced.is_complete);
                        response.insert(response.end(), forced.tokens.begin(), forced.tokens.end());
                        expect_forced_frontier(forced);
                        ASSERT_FALSE(HasFatalFailure());
                        direct_publications += enabled ? 1 : 0;
                        expect_direct_history_evidence();
                        if (source && !enabled && index > 0)
                            EXPECT_GT(completed_forward_only_transactions, 0)
                                << "Repeated ordinary force must execute the sampler-free captured-domain transaction";
                        expect_generated_history(execution, repetition, 1, response);
                        ASSERT_FALSE(HasFatalFailure());
                    }
                    // One direct sampled output followed by real speculative
                    // continuation must retain both forced outputs without
                    // recommitting an already-published carried condition.
                    for (const int budget : {1, 17})
                    {
                        runner.setDecodeStepTokenBudget(budget);
                        const auto result = runner.decodeStep();
                        ASSERT_TRUE(result.success()) << result.error;
                        ASSERT_EQ(result.tokens.size(), source ? size_t(budget) : 0u);
                        (void)terminal_count(result, budget);
                        response.insert(response.end(), result.tokens.begin(), result.tokens.end());
                        // The full17-token transaction commits its own history.
                        // Only the single direct output replays this scalar helper.
                        direct_publications += enabled && budget == 1 ? 1 : 0;
                        expect_direct_history_evidence();
                        expect_generated_history(execution, repetition, budget, response);
                        ASSERT_FALSE(HasFatalFailure());
                    }
                    // A grouped terminal can carry an already-counted ready
                    // condition. The next public forced output is genuinely
                    // new and must not inherit that condition's history bit.
                    const auto forward_only_before_force = completed_forward_only_transactions;
                    const auto forced_after_grouped = runner.forceDecodeToken(4);
                    ASSERT_TRUE(forced_after_grouped.success()) << forced_after_grouped.error;
                    ASSERT_EQ(forced_after_grouped.tokens, source ? std::vector<int32_t>{4} : std::vector<int32_t>{});
                    (void)terminal_count(forced_after_grouped, 1);
                    response.insert(response.end(), forced_after_grouped.tokens.begin(), forced_after_grouped.tokens.end());
                    expect_forced_frontier(forced_after_grouped);
                    ASSERT_FALSE(HasFatalFailure());
                    direct_publications += enabled ? 1 : 0;
                    expect_generated_history(execution, repetition, 1, response);
                    expect_direct_history_evidence();
                    if (source && !enabled) EXPECT_GT(completed_forward_only_transactions, forward_only_before_force);
                    ASSERT_FALSE(HasFatalFailure());
                    runner.setDecodeStepTokenBudget(4);
                    const auto after_force = runner.decodeStep();
                    ASSERT_TRUE(after_force.success()) << after_force.error;
                    ASSERT_EQ(after_force.tokens.size(), source ? 4u : 0u);
                    (void)terminal_count(after_force, 4);
                    response.insert(response.end(), after_force.tokens.begin(), after_force.tokens.end());
                    expect_generated_history(execution, repetition, 4, response);
                    expect_direct_history_evidence();
                    ASSERT_FALSE(HasFatalFailure());
                    if (!enabled) ordinary_response = response;
                    else EXPECT_EQ(response, ordinary_response);
                    EXPECT_EQ(capturedForwardNodes(), captured_before_requests);

                    // A forced stop is terminal control only. Preserve its
                    // explicit no-model-row and no-device-history contract.
                    fresh_request({4});
                    ASSERT_FALSE(HasFatalFailure());
                    expect_generated_history(execution, repetition, 0, {});
                    ASSERT_FALSE(HasFatalFailure());
                    const auto forced_stop = runner.forceDecodeToken(4);
                    ASSERT_TRUE(forced_stop.success()) << forced_stop.error;
                    ASSERT_EQ(forced_stop.tokens, source ? std::vector<int32_t>{4} : std::vector<int32_t>{});
                    (void)terminal_count(forced_stop, 1);
                    EXPECT_TRUE(forced_stop.is_complete);
                    EXPECT_EQ(forced_stop.returned_token_commit, ReturnedTokenCommitState::NoModelRow);
                    expect_direct_history_evidence();
                    expect_generated_history(execution, repetition, 0, {});
                    ASSERT_FALSE(HasFatalFailure());
                    EXPECT_EQ(capturedForwardNodes(), captured_before_requests);

                    // Both laws' actual first sample is in this stop set for
                    // the repetition-zero prompt. A sampled stop is output and
                    // therefore belongs to generated history exactly once.
                    // Reuse that prompt/seed here to hold this boundary fixed.
                    runner.clearCache();
                    SamplingParams stop_law;
                    stop_law.temperature = test.greedy ? 0.0F : 1.0F;
                    stop_law.top_k = 4;
                    stop_law.seed = 0xFE000001u;
                    stop_law.presence_penalty = 1.5F;
                    runner.setSamplingParams(stop_law);
                    runner.setStopTokens({1, 4});
                    ASSERT_TRUE(runner.prefill(std::vector<int>(256, 0))) << runner.lastError();
                    runner.setDecodeStepTokenBudget(1);
                    const auto sampled_stop = runner.decodeStep();
                    ASSERT_TRUE(sampled_stop.success()) << sampled_stop.error;
                    ASSERT_EQ(sampled_stop.tokens.size(), source ? 1u : 0u);
                    (void)terminal_count(sampled_stop, 1);
                    if (source) EXPECT_EQ(sampled_stop.tokens.front(), test.greedy ? 1 : 4);
                    EXPECT_TRUE(sampled_stop.is_complete);
                    direct_publications += enabled ? 1 : 0;
                    expect_direct_history_evidence();
                    expect_generated_history(execution, repetition, 1, sampled_stop.tokens);
                    ASSERT_FALSE(HasFatalFailure());
                    EXPECT_EQ(capturedForwardNodes(), captured_before_requests);
                }
            }
            return;
        }
        if (penalized)
        {
            // Build both actual serial controls before speculative requests;
            // subsequent MTP resets never change the admitted execution mode.
            auto off = active_request_policy;
            off.enabled = false;
            ASSERT_TRUE(runner.configureMTPRequestPolicy(off)) << runner.lastError();
            for (int repetition = 0; repetition < 2; ++repetition)
            {
                runner.clearCache();
                SamplingParams law;
                law.temperature = test.greedy ? 0.0F : 1.0F;
                law.top_k = 4;
                law.seed = 0xFE000001u + repetition;
                law.presence_penalty = 1.5F;
                law.frequency_penalty = 0.0F;
                law.repetition_penalty = 1.0F;
                runner.setSamplingParams(law);
                runner.setStopTokens({});
                ASSERT_TRUE(runner.prefill(std::vector<int>(256, repetition * 7))) << runner.lastError();
                for (const int budget : {1, 65, 1, 17})
                {
                    runner.setDecodeStepTokenBudget(budget);
                    const auto result = runner.decodeStep();
                    ASSERT_TRUE(result.success()) << result.error;
                    ASSERT_EQ(result.tokens.size(), source ? size_t(budget) : 0u);
                    (void)terminal_count(result, budget);
                    serial_controls[repetition].insert(serial_controls[repetition].end(),
                        result.tokens.begin(), result.tokens.end());
                    expect_generated_history("off", repetition, budget, serial_controls[repetition]);
                    ASSERT_FALSE(HasFatalFailure());
                }
                ASSERT_EQ(serial_controls[repetition].size(), source ? 84u : 0u);
                // A successful MTP/Off comparison must exercise a policy that
                // changes an actual token, not merely publish enabled tags.
                // These neutral token IDs belong to this authored numerical
                // fixture; the oracle itself remains the real Off response.
                if (source && test.greedy)
                {
                    const size_t sensitivity_row = repetition == 0 ? 54u : 47u;
                    EXPECT_NE(serial_controls[repetition][sensitivity_row], 26)
                        << "Greedy Off control must observe generated-token presence history";
                }
                else if (source && repetition == 0)
                {
                    EXPECT_NE(serial_controls[repetition][13], 7)
                        << "Stochastic Off control must observe generated-token presence history";
                }
            }
            runner.clearCache();
            ASSERT_TRUE(runner.configureMTPRequestPolicy(active_request_policy)) << runner.lastError();
        }
        for (int repetition = 0; repetition < 2; ++repetition)
        {
            runner.clearCache();
            SamplingParams law;
            law.temperature = test.greedy ? 0.0F : 1.0F;
            law.top_k = 4;
            law.seed = 0xFE000001u + repetition;
            if (penalized)
            {
                law.presence_penalty = 1.5F;
                law.frequency_penalty = 0.0F;
                law.repetition_penalty = 1.0F;
            }
            runner.setSamplingParams(law);
            runner.setStopTokens({});
            int previous = repetition * 7;
            std::vector<int> prompt(256, previous);
            const auto &serial_control = serial_controls[repetition];
            ASSERT_TRUE(runner.prefill(prompt)) << runner.lastError();
            EXPECT_EQ(capturedForwardNodes(), captured_before_requests)
                << "Prefill must reuse the admitted native forward family.";
            if (retained_generation_captures)
                EXPECT_EQ(capturedForwardNodes(CaptureScope::CompleteGeneration), *retained_generation_captures)
                    << "Request reset must preserve every prior generation helper.";
            int emitted = 0;
            std::vector<int32_t> actual_response;
            // Exercise the one-token boundary before a verifier has ever run,
            // as well as between speculative chunks. Otherwise a prior graph
            // output can accidentally mask missing scalar-state publication.
            for (const int budget : {1, 65, 1, 17})
            {
                runner.setDecodeStepTokenBudget(budget);
                const auto initial_before = firstTransactionEvidence();
                const auto result = runner.decodeStep();
                ASSERT_TRUE(result.success()) << result.error;
                ASSERT_EQ(result.tokens.size(), source ? size_t(budget) : 0u);
                const auto completed_count = terminal_count(result, budget);
                if (penalized)
                {
                    actual_response.insert(actual_response.end(), result.tokens.begin(), result.tokens.end());
                    expect_generated_history("mtp", repetition, budget, actual_response);
                    ASSERT_FALSE(HasFatalFailure());
                }
                if (source && shallow_initial && budget == 65)
                {
                    const auto initial_after = firstTransactionEvidence();
                    // Host sampling owns only the actual initial drafts; all
                    // remaining transactions execute retained native branches.
                    EXPECT_EQ(initial_after[0] - initial_before[0], test.initial_depth);
                    EXPECT_GT(initial_after[1] - initial_before[1], 0);
                    EXPECT_EQ(initial_after[2] - initial_before[2], initial_after[1] - initial_before[1]);
                    if (!retained_generation_captures)
                        retained_generation_captures = capturedForwardNodes(CaptureScope::CompleteGeneration);
                }
                if (source && test.lane == PipelineGenerationLane::DynamicMTP && budget == 17)
                {
                    const auto continued = firstTransactionEvidence();
                    EXPECT_EQ(continued[0] - initial_before[0], 0)
                        << "The retained learner must choose the whole continuation branch without external drafts.";
                    EXPECT_EQ(continued[1] - initial_before[1], 1);
                    EXPECT_EQ(continued[3] - initial_before[3], 1)
                        << "Successful resumed output must come from authenticated admission-ticket entry.";
                    EXPECT_EQ(continued[4] - initial_before[4], test.width)
                        << "Every terminal TP participant must authenticate native ticket zero once.";
                }
                EXPECT_EQ(capturedForwardNodes(), captured_before_requests)
                    << "Decode must reuse the admitted native forward family: repetition="
                    << repetition << " budget=" << budget;
                if (retained_generation_captures)
                    EXPECT_EQ(capturedForwardNodes(CaptureScope::CompleteGeneration), *retained_generation_captures)
                        << "Repeated generation must reuse every retained helper.";
                for (const auto token : result.tokens)
                {
                    if (penalized)
                    {
                        ASSERT_LT(size_t(emitted), serial_control.size());
                        previous = serial_control[size_t(emitted)];
                    }
                    else
                    {
                        std::array<int, 4> support;
                        for (int i = 0; i < 4; ++i) support[i] = (previous + i + 1) % 32;
                        std::sort(support.begin(), support.end());
                        const auto draw = sampling_math::mtp_spec_threshold_from_seed(law.seed, 256 + emitted, 0);
                        previous = support[test.greedy ? 0 : std::min(3, int(draw * 4))];
                    }
                    ASSERT_EQ(token, previous) << "repetition=" << repetition << " output_row=" << emitted;
                    ++emitted;
                }
                if (!source) emitted += completed_count;
            }
            // A token-only oracle cannot prove the follower wrote its own KV.
            // Inspect original cache payloads only after terminal completion,
            // so a diagnostic wait cannot hide an ingress ordering defect.
            int32_t cached = 0;
            auto &first = *participants.front();
            auto &context = GPUDeviceContextPool::instance().getContext(first.primaryDeviceId());
            context.submitAndWait([&] {
                ASSERT_TRUE(getBackendFor(first.primaryDeviceId())->deviceToHost(&cached,
                    first.inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0), sizeof(cached),
                    first.primaryDeviceId().gpu_ordinal(), context.defaultStream()));
            });
            ASSERT_FALSE(HasFatalFailure());
            EXPECT_GE(cached, int(prompt.size()) + emitted - 1);
            expectOrdinaryPipelinePromptKV(*pipeline, prompt, cached, participants.size());
            consumed_main_rows += cached - int(prompt.size());
            retained_generation_captures = capturedForwardNodes(CaptureScope::CompleteGeneration);
        }
        if (penalized && source)
        {
            // The target request owns presence1.5, while the actual original
            // stochastic proposal is an unpenalized one-hot argmax. Greedy
            // proposals legitimately consume the same request penalty history.
            for (const auto &member : members.back())
            {
                const auto device = member.toLocalDeviceId().toString();
                double target_policy = 0, draft_publications = 0;
                for (const auto &record : PerfStatsCollector::snapshot())
                {
                    if (record.domain != "mtp" || record.device != device) continue;
                    if (record.name == "request_penalty_policy_publications" &&
                        record.tags.at("enabled") == "true" && record.tags.at("presence_penalty") == "1.500000" &&
                        record.tags.at("frequency_penalty") == "0.000000" && record.tags.at("repetition_penalty") == "1.000000")
                        target_policy += record.value;
                    if (record.name == "device_resident_generation_requests" &&
                        record.tags.at("first_transaction_source") == "external")
                        EXPECT_EQ(record.tags.at("initial_transaction_depth"), "1");
                    if (record.name == "captured_mtp_draft_token_publications")
                    {
                        EXPECT_EQ(record.tags.at("penalties"), test.greedy ? "true" : "false");
                        EXPECT_EQ(record.tags.at("host_observation"), "false");
                        draft_publications += record.value;
                    }
                }
                EXPECT_GT(target_policy, 0) << device;
                EXPECT_GT(draft_publications, 0) << device;
            }
        }
        const auto sum = [&](const char *domain, const char *name, const char *phase = nullptr) {
            double value = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == domain && record.name == name && (!phase || record.phase == phase))
                    value += record.value;
            return value;
        };
        // Canonical native receive banks are the model's hidden owner. Actual
        // captured prefill/condition/verifier/decode must never invoke a second
        // external tensor copy at either domain, including request resets.
        for (const char *name : {"activation_copy_host_bytes", "activation_copy_d2d_bytes",
                 "activation_copy_peer_bytes", "activation_copy_d2h_bytes", "activation_copy_h2d_bytes"})
            EXPECT_EQ(sum("transfer", name), 0.0) << name;
        EXPECT_EQ(sum("forward_graph", "segmented_plan_segments"), 0);
        EXPECT_EQ(sum("forward_graph", "segmented_graph_capture_segments"), 0);
        EXPECT_EQ(sum("forward_graph", "segmented_replay_segments", "prefill"), 0);
        EXPECT_EQ(sum("forward_graph", "segmented_replay_segments", "decode"), 0);
        if (test.usesMTP() && !source)
            EXPECT_GT(sum("generation", "pipeline_mtp_follower_transaction_submissions"), 0);
        if (test.stage_count == 3 && test.usesMTP())
            for (size_t index = 0; index < participants.size(); ++index)
                if (participant_domains[index] != terminal_domain)
                {
                    double submitted = 0;
                    for (const auto &record : PerfStatsCollector::snapshot())
                        if (record.domain == "generation" && record.name == "pipeline_mtp_follower_transaction_submissions" &&
                            record.device == participants[index]->primaryDeviceId().toString())
                            submitted += record.value;
                    EXPECT_GT(submitted, 0) << "Every nonterminal stage must execute, including the entry on the terminal owner's rank";
                }
        if (test.usesMTP() && source)
        {
            EXPECT_GT(sum("mtp", "budget_limited_direct_emits"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_transactions"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_attempted_draft_tokens"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_verifier_tokens"), 0);
            if (test.lane == PipelineGenerationLane::DynamicMTP)
                EXPECT_GT(sum("mtp", "device_generation_terminal_depth_evaluated_windows"), 0);
            else
                EXPECT_EQ(sum("mtp", "device_generation_terminal_depth_evaluated_windows"), 0);
        }
        else if (!test.usesMTP())
        {
            // Ordinary sampling uses the same controller/terminal ledger.
            // Its transactions are real, but none may perform speculative work.
            if (source) EXPECT_GT(sum("mtp", "device_generation_terminal_transactions"), 0);
            // Each independently observed cache row requires one traversal of
            // every local participant, rather than one per returned token.
            EXPECT_EQ(sum("forward_graph", "decode_graph_phase", "decode"), participants.size() * consumed_main_rows);
            for (const char *counter : {"device_generation_terminal_attempted_draft_tokens",
                     "device_generation_terminal_verifier_tokens",
                     "device_generation_terminal_compact_outcome_reductions",
                     "device_generation_terminal_depth_evaluated_windows"})
                EXPECT_EQ(sum("mtp", counter), 0) << counter;
        }
        }
        catch (const std::exception &error)
        {
            lifetime.abortOriginal(error.what());
        }
        catch (...)
        {
            lifetime.abortOriginal("nonstandard exception before original collective retirement");
        }
    }
};

TEST_P(PipelineDomainGeneration, ExactCapturedRequestsAndContinuation) { run(); }

/** @brief Same production native/TP setup with a non-neutral request target law. */
class PipelineDomainPenalizedGeneration : public PipelineDomainGeneration {};

/** @test Fresh and reset penalized requests match actual Off tokens without changing draft law. */
TEST_P(PipelineDomainPenalizedGeneration, PenalizedTargetKeepsOriginalDraftLaw)
{
    run(RequestLaw::PresencePenalty);
}

INSTANTIATE_TEST_SUITE_P(PenalizedTarget, PipelineDomainPenalizedGeneration,
    ::testing::ValuesIn([] {
        std::vector<PipelineDomainCase> cases;
        for (const bool reverse : {false, true})
            for (const int width : {1, 2})
                for (const bool greedy : {false, true})
                    cases.push_back({reverse, width, PipelineGenerationLane::FixedMTP, 128, 15, greedy});
        return cases;
    }()),
    [](const auto &info) {
        const auto &p = info.param;
        return std::string(p.reverse ? "ROCm" : "CUDA") + std::to_string(p.width) + "_" +
            (p.reverse ? "CUDA" : "ROCm") + std::to_string(p.width) + "_Fixed1_PresencePenalty_" +
            (p.greedy ? "Greedy" : "Stochastic") + "_Admitted128";
    });

/** @brief Native direct-output boundaries reuse the original serving fixture. */
class PipelineDomainDirectOutputHistory : public PipelineDomainGeneration {};

/** @test Direct, forced and stop outputs preserve their distinct history contracts. */
TEST_P(PipelineDomainDirectOutputHistory, DirectForceStopAndContinuationPublishHistoryOnce)
{
    run(RequestLaw::PresencePenalty, RequestScenario::DirectHistoryBoundaries);
}

INSTANTIATE_TEST_SUITE_P(DirectHistory, PipelineDomainDirectOutputHistory,
    ::testing::ValuesIn([] {
        std::vector<PipelineDomainCase> cases;
        for (const bool reverse : {false, true})
            for (const int width : {1, 2})
                for (const bool greedy : {false, true})
                    cases.push_back({reverse, width, PipelineGenerationLane::FixedMTP, 128, 15, greedy});
        return cases;
    }()),
    [](const auto &info) {
        const auto &p = info.param;
        return std::string(p.reverse ? "ROCm" : "CUDA") + std::to_string(p.width) + "_" +
            (p.reverse ? "CUDA" : "ROCm") + std::to_string(p.width) +
            (p.greedy ? "_Greedy" : "_Stochastic");
    });

INSTANTIATE_TEST_SUITE_P(CrossBackend, PipelineDomainGeneration,
    ::testing::ValuesIn([] {
        std::vector<PipelineDomainCase> cases;
        for (const bool reverse : {false, true})
            for (const int width : {1, 2, 4})
                for (const auto lane : {PipelineGenerationLane::Ordinary, PipelineGenerationLane::FixedMTP,
                                       PipelineGenerationLane::DynamicMTP})
                    for (const int rows : {128, 256})
                        cases.push_back({reverse, width, lane, rows});
        for (const auto lane : {PipelineGenerationLane::Ordinary, PipelineGenerationLane::FixedMTP,
                               PipelineGenerationLane::DynamicMTP})
            cases.push_back({false, 1, lane, 128, 15, false, 3});
        // Keep every original initial-15 case; these twelve cold declarations
        // expose helpers that shallow initial execution never warmed itself.
        for (const bool reverse : {false, true})
            for (const int width : {1, 2})
                for (const int initial : {1, 3, 7})
                    cases.push_back({reverse, width, PipelineGenerationLane::DynamicMTP, 128, initial});
        // Initial1 has no executed chained proposal to warm its retained family;
        // exercise that cold boundary under the independent greedy law as well.
        for (const bool reverse : {false, true})
            for (const int width : {1, 2})
                cases.push_back({reverse, width, PipelineGenerationLane::DynamicMTP, 128, 1, true});
        return cases;
    }()),
    [](const auto &info) {
        const auto &p = info.param;
        const auto topology = p.stage_count == 3 ? std::string("CUDA1_ROCm1_CUDA1") :
            std::string(p.reverse ? "ROCm" : "CUDA") + std::to_string(p.width) + "_" +
                (p.reverse ? "CUDA" : "ROCm") + std::to_string(p.width);
        return topology + (p.lane == PipelineGenerationLane::Ordinary ? "_Ordinary" :
                p.lane == PipelineGenerationLane::FixedMTP ? "_FixedMTP" : "_DynamicMTP") +
            (p.lane == PipelineGenerationLane::DynamicMTP && p.initial_depth < 15
                ? "_Initial" + std::to_string(p.initial_depth) : "") +
            (p.greedy ? "_Greedy" : "") +
            (p.admitted_rows == 256 ? "" : "_Admitted" + std::to_string(p.admitted_rows));
    });
}
