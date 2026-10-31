/**
 * @file ModelKLDCanary.cpp
 * @brief Observe full-vocabulary production logits on an identical forced trajectory.
 *
 * RuntimeInitPhase owns model admission, MPI and captured execution. Explicit
 * diagnostic host-logit publication is enabled at request boundaries; its
 * timings are never used as canonical throughput. The forced-token result owns
 * whether a row was committed, preventing a guessed or double-forwarded token.
 * Legacy diagnostics accept one CUDA or ROCm device. The versioned QFN request
 * additionally authenticates supported CPU, CPU-tail and LocalTP runner owners.
 * Both use the production host-logit consumer without arithmetic interposition.
 * Frozen workload and artifact paths are explicit tool arguments, independent
 * of a workspace mount. Both backends use the same observer and phase schedule.
 * A lexical entry owner retires native workers before vendor exit callbacks;
 * MPI shutdown closes that same owner before finalizing communicator state.
 */
#include "utils/MPISessionOwner.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "app/RuntimeInitPhase.h"
#include "app/MPIBootstrapPhase.h"
#include "ModelKLDQFNContract.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "planning/ModelMemoryProfile.h"
#include "execution/moe/MoEOverlayCapacityResolver.h"
#include "execution/moe/MoEPipelinePreparedPlan.h"
#include "execution/local_execution/graph/IGraphExecutor.h"
#include "utils/Sha256.h"
#include <array>
#include <map>
#include <set>
#include "utils/MPIBootstrap.h"
#include "utils/PerfStatsCollector.h"
#include "utils/DebugEnv.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/runner/IOrchestrationRunner.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace llaminar2;
using json = nlohmann::json;

/** @brief Publish one complete report only after its output streams are closed. */
void publish(const std::filesystem::path &path, const json &value)
{
    std::ofstream output(path);
    output << value.dump(2) << '\n';
    if (!output) throw std::runtime_error("Cannot write accuracy report");
}

/** @brief Save one fresh finite full-vocabulary row at a joined result boundary. */
json saveLogits(IOrchestrationRunner &runner, std::ofstream &output, size_t position)
{
    /*
     * lastLogits() authenticates the current device publication, queues the
     * exact producer-event wait on the explicit host bridge, and returns only
     * after the full vocabulary D2H completes. Accuracy has no timing interval
     * to close. An extra benchmark rendezvous incorrectly applies a collective
     * deadline to all already-submitted prefill buckets on a single GPU.
     * Missing publications and transfer failures remain fatal in the native
     * consuming API. The passive lifetime observer records health throughout.
     */
    const float *logits = runner.lastLogits();
    const int vocabulary = runner.vocabSize();
    if (!logits || vocabulary <= 0)
        throw std::runtime_error("No complete host logits at the diagnostic result boundary");
    if (!std::all_of(logits, logits+vocabulary, [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("Nonfinite next-token logits");
    output.write(reinterpret_cast<const char *>(logits), vocabulary*sizeof(float));
    if (!output) throw std::runtime_error("Could not persist full-vocabulary row");
    return {{"consumed_tokens",position}, {"argmax",std::max_element(logits,logits+vocabulary)-logits},
            {"vocabulary",vocabulary}};
}

/**
 * @brief Observe only real prepared/PMA/source owners after original RuntimeInit.
 * @param request Checked frozen diagnostic request, never an allocation authority.
 * @param runner Actual ready production runner; no test factory or owner mirror.
 * @return Passive exact source/config/plan descriptors for the result artifact.
 * @throws std::invalid_argument For changed source, unsupported topology or missing owner proof.
 */
json authenticateQFN(const accuracy::QFNObserverRequest &request, IOrchestrationRunner &runner)
{
    using accuracy::requireQFN;
    const auto owner = runner.modelContextReuseContract();
    requireQFN(owner && owner->context && owner->reuse_authority && owner->physical_memory_authority,
               "QFN observer needs the actual admitted model/prepared/PMA owner");
    auto &model = *owner->context;
    const auto profile = ModelMemoryProfile::fromLoader(model.concreteLoader());
    requireQFN(runner.architecture() == "qwen4exp" && model.architecture() == "qwen4exp" &&
               profile.n_layers == 48 && profile.vocab_size == 248320 && profile.expert_count == 512 &&
               runner.vocabSize() == 248320, "QFN observer actual original source geometry differs");
    const auto surface = accuracy::qfnObservationSurface(runner.executionPlan());
    const auto &expected = request.corpus().at("source");
    requireQFN(model.path() == runner.config().model_path, "QFN original model path differs from actual parsed configuration");
    const auto paths = model.artifactPaths();
    json descriptors=expected.at("shards");
    if (runner.config().shared_predictor)
    {
        requireQFN(model.concreteLoader().ggufPredictorDirectory() &&
                   model.concreteLoader().predictorManifest() &&
                   model.concreteLoader().predictorMaterialization()==SharedPredictorMaterialization::Inactive,
                   "QFN MTP-off paired source must remain original authenticated inactive metadata");
        descriptors.push_back(expected.at("shared_predictor"));
    }
    requireQFN(paths.size() == descriptors.size(), "QFN source shard inventory changed");
    json source = json::array();
    for (std::size_t i = 0; i < paths.size(); ++i)
    {
        const auto &descriptor = descriptors.at(i);
        source.push_back(accuracy::qfnRequireOriginalSourceStat(descriptor, paths[i]));
    }
    requireQFN(request.topology().at("arithmetic").at("gdn_projection") == gdnProjectionArithmeticName(debugEnv().gemm.gdn_projection_arithmetic) &&
               request.topology().at("arithmetic").at("moe_router") == moeRouterArithmeticName(resolveMoERouterArithmetic(
                   debugEnv().gemm.moe_router_arithmetic, runner.architecture())),
               "QFN observer actual source/default arithmetic policy differs");
    const auto &gemm = debugEnv().gemm;
    const json actual_residual = {{"hc_up", quantizedActivationOperandPrecisionName(gemm.qfn_hc_up_activation)},
        {"gdn_output", quantizedActivationOperandPrecisionName(gemm.qfn_gdn_output_activation)},
        {"fa_output", quantizedActivationOperandPrecisionName(gemm.qfn_fa_output_activation)},
        {"gdn_qkv", quantizedActivationOperandPrecisionName(gemm.qfn_gdn_qkv_activation)}};
    for (const auto &[key, value] : actual_residual.items())
        requireQFN(request.topology().at("arithmetic").at(key) == value, "QFN inherited correction selector differs");
    const auto placement = model.resolvedImmutableRowTablePlacement();
    requireQFN(bool(placement), "QFN observer original PLE placement is missing");
    placement->requireRequestedPolicy(runner.config().ngram_table_policy);
    if (surface == accuracy::QFNObservationSurface::CPU)
    {
        // The ordinary CPU main cell has no ExpertOverlay authority. Its real
        // reuse contract already authenticates complete graph preparation and
        // the same PMA; require its actual CPU store instead of inventing one.
        const auto weights = model.concreteWeightManager();
        const auto store = weights ? weights->preparedWeightStoreIfInitialized() : nullptr;
        requireQFN(weights && store && store->physicalMemoryAuthority().get() == owner->physical_memory_authority.get() &&
                   weights->preparedRecordCountForDevice(runner.executionPlan().primary_device.toLocalDeviceId()) > 0 &&
                   !owner->pipeline_prepared_plan,
                   "QFN observer lacks its actual ordinary CPU prepared/store/PMA owner");
    }
    else if (surface == accuracy::QFNObservationSurface::CPUTailPipeline)
    {
        requireQFN(owner->pipeline_prepared_plan && owner->pipeline_prepared_plan->admission() &&
                   owner->pipeline_prepared_plan->admission()->physicalAdmission() &&
                   owner->pipeline_prepared_plan->stages().size() == runner.executionPlan().local_pp_devices.size(),
                   "QFN observer lacks original per-stage prepared/source admission");
        for (const auto &stage : owner->pipeline_prepared_plan->stages())
            requireQFN(stage.placement && stage.placement->enabled && !stage.placement->placements.empty() &&
                       stage.metadata.num_layers > 0 && stage.metadata.num_experts == 512 &&
                       validateMoERoutedExpertPlacementPlan(*stage.placement,
                           {.layer_count=stage.metadata.num_layers, .routed_expert_count=512}).ok(),
                       "QFN observer pipeline has incomplete original expert coverage");
    }
    else
    {
        requireQFN(owner->prepared_routed_weight_plan && owner->prepared_routed_weight_plan->enabled &&
                   !owner->prepared_routed_weight_plan->placements.empty() &&
                   owner->prepared_routed_weight_plan->first_model_layer == 0 && owner->expert_overlay_memory_admission &&
                   owner->expert_overlay_memory_admission->physical_memory_admission &&
                   !owner->routed_weight_authority_identity.empty() &&
                   owner->expert_overlay_memory_admission->layer_footprints.size() >= 48 &&
                   validateMoERoutedExpertPlacementPlan(*owner->prepared_routed_weight_plan,
                       {.layer_count=static_cast<int>(owner->expert_overlay_memory_admission->layer_footprints.size()),
                        .routed_expert_count=512}).ok(),
                   "QFN observer lacks complete original expert/prepared/admission authority");
        if (surface == accuracy::QFNObservationSurface::GPUCPUOverlay)
        {
            bool cpu = false;
            for (const auto &domain : owner->prepared_routed_weight_plan->domains)
                for (const auto &address : domain.participants) cpu |= address.device_type == DeviceType::CPU;
            requireQFN(cpu, "QFN observer single-GPU arm requires its original CPU expert capacity domain");
        }
        if (surface == accuracy::QFNObservationSurface::ROCmLocalTP4)
            for (const auto &domain : owner->prepared_routed_weight_plan->domains)
                for (const auto &address : domain.participants)
                    requireQFN(address.device_type == DeviceType::ROCm,
                               "QFN observer full-ROCm TP4 cannot borrow a CPU or foreign expert tier");
    }
    std::vector<std::filesystem::path> artifacts(paths.begin(),paths.end());
    const auto source_identity=sha256FileSetIdentityHex(artifacts);
    requireQFN(bool(source_identity), "QFN source metadata identity is incomplete");
    return {{"source_file_set_identity",*source_identity}, {"surface", int(surface)},
            {"source_artifacts", source}, {"model_architecture", profile.architecture},
            {"main_layers", profile.n_layers}, {"vocabulary", profile.vocab_size}, {"experts", profile.expert_count},
            {"actual_config", json::parse(serializeOrchestrationConfig(runner.config()))},
            {"actual_rank_plan", runner.executionPlan().toString()},
            {"prepared_rank_plan", owner->prepared_weight_plan.toString()},
            {"routed_weight_authority_identity", owner->routed_weight_authority_identity},
            {"original_pma_owner", true},
            {"commit_contract", "actual Pending results; final successor is publication only"}};
}


/**
 * @brief Produce shared corpus-v2 coordinates after a successful public observation.
 * @param history Original immutable prompt and Pending teacher sequence.
 * @param offset Negative for prefill, otherwise the consumed teacher's zero-based offset.
 * @param vocabulary Actual admitted model vocabulary.
 * @return Metadata only; no row or numerical success is fabricated by this projection.
 */
json qfnRow(const json &history, int offset, int vocabulary)
{
    const auto id = history.at("id").get<std::string>();
    return {{"history_id", id}, {"context_group", history.at("context_group")},
        {"row_id", id + (offset < 0 ? ":prefill" : ":decode:"+std::to_string(offset))},
        {"phase", offset < 0 ? "prefill" : "decode"},
        {"decode_offset", offset < 0 ? json(nullptr) : json(offset)},
        {"consumed_tokens", history.at("prompt_ids").size() + (offset < 0 ? 0u : std::size_t(offset+1))},
        {"input_token", offset < 0 ? json(nullptr) : history.at("continuation_ids").at(std::size_t(offset))},
        {"vocabulary", vocabulary}};
}

/**
 * @brief Observe one frozen history with the same admitted production runner.
 * @param runner Original ready runner, never re-created between histories.
 * @param history Immutable prompt/teacher/reference control from the shared corpus.
 * @param output Exclusive fresh output directory owned by this invocation.
 * @return Closed full-vocabulary file and exact source-owned row metadata.
 * @throws std::runtime_error For failed publication, changed commit semantics or incomplete output.
 */
json observeQFNHistory(IOrchestrationRunner &runner, const json &history, const std::filesystem::path &output)
{
    const auto prompt = history.at("prompt_ids").get<std::vector<int32_t>>();
    const auto continuation = history.at("continuation_ids").get<std::vector<int32_t>>();
    const auto name = history.at("id").get<std::string>();
    runner.clearCache();
    {
        accuracy::ScopedQFNObservationPhase phase(ExecutionPhase::PREFILL);
        if (!runner.prefill(prompt)) throw std::runtime_error("QFN prefill failed: "+runner.lastError());
    }
    const auto path = std::filesystem::absolute(output/(name+".f32"));
    std::ofstream logits(path,std::ios::binary);
    json rows = json::array();
    (void)saveLogits(runner,logits,prompt.size());
    rows.push_back(qfnRow(history,-1,runner.vocabSize()));
    for (std::size_t i=0;i<continuation.size();++i)
    {
        const auto forced = [&] {
            accuracy::ScopedQFNObservationPhase phase(ExecutionPhase::DECODE);
            return runner.forceDecodeToken(continuation[i]);
        }();
        if (!forced.success() || forced.tokens.size()!=1 || forced.tokens[0]!=continuation[i] ||
            forced.returned_token_commit!=ReturnedTokenCommitState::Pending)
            throw std::runtime_error("QFN original Pending teacher transaction failed: "+forced.error);
        if (i>0)
        {
            (void)saveLogits(runner,logits,prompt.size()+i);
            rows.push_back(qfnRow(history,int(i-1),runner.vocabSize()));
        }
    }
    logits.close();
    if (!logits || rows.size()!=history.at("decode_rows").get<std::size_t>()+1 ||
        std::filesystem::file_size(path)!=rows.size()*std::size_t(runner.vocabSize())*sizeof(float))
        throw std::runtime_error("QFN observation inventory/output is incomplete");
    return {{"history_id",name},{"logits_file",path.string()},{"rows",rows}};
}

/**
 * @brief Compare distinct closed reset banks with bounded CPU buffers.
 * @param a Original native observation file, after its stream is closed.
 * @param b Final independent replay file from the same admitted runner.
 * @return True only for separate files with identical original FP32 bytes.
 * @throws std::filesystem::filesystem_error For missing or inaccessible files.
 */
bool qfnResetBytesEqual(const std::filesystem::path &a, const std::filesystem::path &b)
{
    if (std::filesystem::equivalent(a,b) || std::filesystem::file_size(a)!=std::filesystem::file_size(b)) return false;
    std::ifstream first(a,std::ios::binary), second(b,std::ios::binary);
    std::array<char,65536> left{},right{};
    while (first)
    {
        first.read(left.data(),left.size()); second.read(right.data(),right.size());
        if (first.gcount()!=second.gcount() || !std::equal(left.begin(),left.begin()+first.gcount(),right.begin())) return false;
    }
    return first.eof() && second.eof();
}

/**
 * @brief Require actual phase/path records from every admitted GPU; CPU remains host graph execution.
 * @param runner Actual runner whose histories have completed, before shutdown.
 * @param observation Existing owner-authored physical shape and source evidence.
 * @return Original positive path records retained for external source/build/driver joins.
 * @throws std::runtime_error For absent, partial, eager or unsupported native phase evidence.
 */
json requireQFNPath(IOrchestrationRunner &runner, const json &observation)
{
    const auto surface = accuracy::qfnObservationSurface(runner.executionPlan());
    std::set<std::string> gpu;
    const auto add = [&](const GlobalDeviceAddress &address) {
        if (address.device_type==DeviceType::CUDA || address.device_type==DeviceType::ROCm)
            gpu.insert(address.toLocalDeviceId().toString());
    };
    for (const auto &address : runner.executionPlan().local_tp_devices) add(address);
    for (const auto &address : runner.executionPlan().local_pp_devices) add(address);
    add(runner.executionPlan().primary_device);
    if (!PerfStatsCollector::isDomainEnabled("forward_graph"))
        throw std::runtime_error("QFN observer requires actual forward_graph path evidence");
    const auto records = PerfStatsCollector::snapshot({"forward_graph"});
    json raw=json::array();
    const std::set<std::string_view> families = {
        "host_graph_completed_calls", "prefill_graph_phase", "decode_graph_phase", "decode_capture_policy",
        "full_graph_capture_executable_nodes", "full_graph_replay_calls", "segmented_plan_segments",
        "segmented_graph_capture_executable_nodes", "segmented_graph_capture_segments", "segmented_replay_segments",
        "retained_parent_transaction_zero_launches", "retained_parent_transaction_captures",
        "retained_parent_replays", "retained_parent_transaction_replays"};
    for (const auto &record : records)
    {
        if (!(record.count > 0 || record.value > 0)) continue;
        if (!families.contains(record.name)) continue;
        raw.push_back({{"domain", record.domain}, {"name", record.name}, {"device", record.device},
            {"phase", record.phase}, {"tags", record.tags}, {"count", record.count}, {"value", record.value}});
    }
    // Only the prior actual prepared-owner check can authorize a heterogeneous
    // boundary. Reuse the same pure certifier as the canonical all-stage gate.
    accuracy::qfnRequireGraphPathRecords(records, surface, gpu);
    (void)observation;
    return raw;
}

/**
 * @brief Read only the ordinary LocalTP host-publication owner's passive records.
 * @return Validated cumulative actual one-row publications; no native probe.
 * @throws std::runtime_error If the required sampling family was not collected.
 */
std::uint64_t qfnTPHostPublications()
{
    if (!PerfStatsCollector::isDomainEnabled("sampling"))
        throw std::runtime_error("QFN TP4 observer requires sampling publication evidence");
    const auto records = PerfStatsCollector::snapshot({"sampling"});
    return accuracy::qfnTPHostPublicationCount(records);
}

/**
 * @brief Execute one shared corpus-v2 descriptor through exactly one admitted model.
 * @param workload Shared approved corpus and selected canonical topology.
 * @param output Fresh root-owned artifact directory; existing output is rejected.
 * @param engine_argc Existing runtime argument count.
 * @param engine_argv Existing runtime argument vector retained by the original main scope.
 * @param original_argc Complete observer argument count for canonical self-launch.
 * @param original_argv Original wrapper and engine argv, preserving its parser boundary.
 * @return Zero only for complete authentic observations/retirement/reset; no numeric verdict.
 */
int runQFNObserver(const json &workload, const std::filesystem::path &output,
                   int &engine_argc, char **&engine_argv, int original_argc, char **original_argv)
{
    json result={{"schema","llaminar.model-kld-observation.v2"},{"complete",false},
                 {"cases",json::array()},{"proofs",json::object()}};
    bool owns_output=false;
    try
    {
        using accuracy::requireQFN;
        auto config=OrchestrationConfigParser{}.parseArgs(engine_argc,engine_argv);
        const auto request=accuracy::QFNObserverRequest::parse(workload,config);
        requireQFN(!std::filesystem::exists(output),"QFN observer output directory must be fresh");
        if (MPIBootstrapPhase::classifyDeviceIntent(config) == BootstrapDeviceIntent::CpuOnly)
        {
            // Reuse production's NUMA/physical-core launch before creating an
            // output owner or admitting a model. The child parses the exact
            // original wrapper/engine arguments again; no Python affinity map
            // or rewritten config can override the canonical verifier.
            requireQFN(config.mpi_procs == 1 && config.hostfile.empty() &&
                !config.execution_rank_selection, "QFN CPU observer requires explicit one-rank local bootstrap");
            const auto launch = MPIBootstrap::detectMPIEnvironment();
            requireQFN(!launch.is_mpi_process || launch.detected_world_size == 1,
                       "QFN CPU observer rejects multi-rank bootstrap membership");
            const auto bootstrap = MPIBootstrapPhase{}.execute(config, original_argc, original_argv,
                MPIBootstrapArgumentPolicy::PreserveOriginalArguments);
            if (bootstrap.action == BootstrapResult::Action::EXIT)
                return bootstrap.exit_code;
        }
        owns_output=std::filesystem::create_directories(output);
        requireQFN(owns_output,"QFN observer cannot create its output directory");
        result["corpus_manifest_digest"]=accuracy::qfnMetadataDigest(request.corpus());
        result["source_digest"]=accuracy::qfnMetadataDigest(request.corpus().at("source"));
        result["reference_report_digest"]=request.corpus().at("reference").at("report_digest");
        for(const char *key:{"backend","runtime_config_digest","arithmetic"}) result[key]=request.topology().at(key);
        result["topology_id"]=request.topology().at("id");
        const auto launch=MPIBootstrap::detectMPIEnvironment();
        requireQFN(!launch.is_mpi_process || launch.detected_world_size==1,
                   "QFN observer rejects multi-rank or unresolved MPI launch membership");
        auto startup=RuntimeInitPhase().execute(config,engine_argc,engine_argv);
        requireQFN(std::holds_alternative<AppContext>(startup),"QFN runtime did not return its original model owner");
        auto context=std::move(std::get<AppContext>(startup));
        requireQFN(context.mpi_ctx && context.mpi_ctx->world_size()==1,"QFN observer requires exactly one MPI rank");
        auto &runner=*context.runner;
        runner.setSkipLogitsGatherDecode(false); runner.setSkipLogitsGatherPrefill(false);
        requireQFN(runner.prepareForInference(),"QFN production preparation failed");
        result["actual_owner"]=authenticateQFN(request,runner);
        const auto executable=sha256FileHex("/proc/self/exe");
        requireQFN(bool(executable),"QFN observer cannot authenticate its actual executable");
        result["native_identity"]={{"executable_sha256",*executable},
            {"actual_config_sha256",accuracy::qfnMetadataDigest(result["actual_owner"].at("actual_config"))},
            {"actual_source_descriptor_sha256",result["actual_owner"].at("source_file_set_identity")}};
        const bool tp4=accuracy::qfnObservationSurface(runner.executionPlan())==accuracy::QFNObservationSurface::ROCmLocalTP4;
        const auto publication_before=tp4?qfnTPHostPublications():0;
        std::uint64_t expected_publications=0;
        for(const auto &history:request.corpus().at("histories"))
        {
            result["cases"].push_back(observeQFNHistory(runner,history,output));
            expected_publications+=history.at("decode_rows").get<std::uint64_t>();
        }
        const auto &reset=request.corpus().at("reset_control");
        json original, replay;
        for(const auto &history:request.corpus().at("histories"))
            if(history.at("id")==reset.at("history_id")) { replay=history; break; }
        for(const auto &value:result["cases"])
            if(value.at("history_id")==reset.at("history_id")) { original=value; break; }
        replay["id"]=reset.at("replay_id");
        result["reset_case"]=observeQFNHistory(runner,replay,output);
        expected_publications+=replay.at("decode_rows").get<std::uint64_t>();
        if(tp4)
        {
            const auto after=qfnTPHostPublications();
            requireQFN(after>=publication_before && after-publication_before==expected_publications,
                       "QFN TP4 did not publish every original committed row through its actual host gather");
            result["tp_host_publications"]={{"before",publication_before},{"after",after},
                {"observed_decodes_including_reset",expected_publications}};
        }
        requireQFN(qfnResetBytesEqual(original.at("logits_file").get<std::string>(),
                    result["reset_case"].at("logits_file").get<std::string>()),"QFN real request reset changed original logits bytes");
        result["path_records"]=requireQFNPath(runner,result["actual_owner"]);
        // Recheck original admitted source/policies before terminal teardown.
        requireQFN(authenticateQFN(request,runner)==result["actual_owner"],"QFN original owner/config changed across histories");
        runner.shutdown(); context.runner.reset();
        requireQFN(PerfStatsCollector::flushFromEnv(),"QFN observer cannot publish original path counters");
        result["proofs"]={{"source_authenticated",true},{"production_path_authenticated",true},
            {"publication_authenticated",true},{"native_retired",true},{"reset_byte_equal",true}};
        result["complete"]=true;
    }
    catch(const std::exception &error)
    {
        result["error"]=error.what(); std::cerr<<error.what()<<'\n';
    }
    if(owns_output) publish(output/"logits.json",result);
    return result.at("complete").get<bool>()?0:1;
}


/** @brief Execute the exact production runner with fixed token histories. */
int main(int argc, char **argv)
{
    // Ownership starts before this frontend creates MPI or runtime state.
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_lifetime;
    json result={{"complete",false},{"passed",false},{"diagnostic_only",true},{"cases",json::array()}};
    std::filesystem::path output;
    try
    {
        if (argc >= 4 && std::string_view(argv[1]) == "--describe-config" && std::string_view(argv[2]) == "--")
        {
            std::vector<char *> described_argv{argv[0]};
            for(int i=3;i<argc;++i) described_argv.push_back(argv[i]);
            const auto exact_argv=std::vector<std::string>(described_argv.begin()+1,described_argv.end());
            described_argv.push_back(nullptr);
            const auto parsed=OrchestrationConfigParser{}.parseArgs(int(described_argv.size())-1,described_argv.data());
            const auto document=json::parse(serializeOrchestrationConfig(parsed));
            std::cout<<json({{"schema","llaminar.model-kld-described-config.v1"},
                {"config",document},{"runtime_config_digest",accuracy::qfnMetadataDigest(document)},
                {"engine_argv",exact_argv},
                {"qfn_canonical_intent",accuracy::qfnCanonicalConfigurationIntent(parsed)},
                {"qfn_arithmetic_intent",accuracy::qfnActualArithmeticIntent()}}).dump(2)<<'\n';
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--help")
        {
            std::cout << "llaminar_model_kld_canary --workload FILE --output DIR -- ENGINE_ARGS\n";
            return 0;
        }
        if (argc < 8 || std::string_view(argv[1]) != "--workload" ||
            std::string_view(argv[3]) != "--output" || std::string_view(argv[5]) != "--")
            throw std::invalid_argument("Require --workload FILE --output DIR -- ENGINE_ARGS");
        const std::filesystem::path input = argv[2];
        output = argv[4];
        std::vector<char *> engine_argv{argv[0]};
        for (int index=6; index<argc; ++index) engine_argv.push_back(argv[index]);
        int engine_argc = static_cast<int>(engine_argv.size());
        engine_argv.push_back(nullptr);
        char **engine_argv_pointer = engine_argv.data();
        json workload;
        try { std::ifstream(input) >> workload; }
        catch (...) { std::filesystem::create_directories(output); throw; }
        if (workload.value("schema", std::string{}) == "llaminar.model-kld-workload.v2")
            return runQFNObserver(workload, output, engine_argc, engine_argv_pointer, argc, argv);
        std::filesystem::create_directories(output);
        OrchestrationConfigParser parser;
        auto config = parser.parseArgs(engine_argc,engine_argv.data());
        if (config.mtp_activation_policy != MTPActivationPolicy::Disabled || config.mtp.enabled ||
            config.prefix_cache.enabled || config.activation_precision != "fp32" ||
            config.kv_cache_precision != "fp16" || config.prefill_max_bucket_size != 512 ||
            (!config.tp_devices.empty() || !config.device_for_this_rank.has_value() ||
             (config.device_for_this_rank->device_type != DeviceType::CUDA &&
              config.device_for_this_rank->device_type != DeviceType::ROCm) ||
             config.device_for_this_rank->device_ordinal != 0 ||
             config.device_for_this_rank->hostname != "localhost"))
            throw std::runtime_error("KL observer requires explicit cuda:0 or rocm:0, no TP, M512, FP32 activations, FP16 KV, MTP/prefix off");
        result["contract"] = {{"topology","single_device"},{"device",config.device_for_this_rank->device_type == DeviceType::CUDA ? "cuda:0" : "rocm:0"},
            {"physical_devices",1},{"local_tp_devices",0},{"prefill_max_bucket",512},
            {"activation_precision","fp32"},{"kv_precision","fp16"},{"mtp",false},
            {"prefix_cache",false},{"event_ordered_host_logits",true},
            {"gdn_arithmetic",gdnProjectionArithmeticName(debugEnv().gemm.gdn_projection_arithmetic)}};
        auto startup = RuntimeInitPhase().execute(config,engine_argc,engine_argv_pointer);
        if (!std::holds_alternative<AppContext>(startup))
            throw std::runtime_error("Production runtime initialization did not return an owner");
        auto context = std::move(std::get<AppContext>(startup));
        auto &runner = *context.runner;
        // Record the source-owned concrete policy, never an unresolved setup selection.
        result["contract"]["router_arithmetic"] = moeRouterArithmeticName(
            resolveMoERouterArithmetic(debugEnv().gemm.moe_router_arithmetic, runner.architecture()));
        if (context.mpi_ctx->world_size()!=1)
            throw std::runtime_error("This observer requires exactly one MPI rank");
        runner.setSkipLogitsGatherDecode(false);
        runner.setSkipLogitsGatherPrefill(false);
        if (!runner.prepareForInference())
            throw std::runtime_error("Production preparation failed: "+runner.lastError());
        for (const auto &item : workload.at("cases"))
        {
            const auto prompt=item.at("prompt_ids").get<std::vector<int32_t>>();
            const auto continuation=item.at("continuation_ids").get<std::vector<int32_t>>();
            const auto name=item.at("name").get<std::string>();
            if (prompt.empty() || continuation.size()<2 || name.find('/')!=std::string::npos)
                throw std::runtime_error("Malformed teacher-forcing case");
            runner.clearCache();
            if (!runner.prefill(prompt)) throw std::runtime_error("Prefill failed: "+runner.lastError());
            std::ofstream logits(output/(name+".f32"),std::ios::binary);
            json rows=json::array();
            rows.push_back(saveLogits(runner,logits,prompt.size()));
            for (size_t step=0; step<continuation.size(); ++step)
            {
                const auto forced=runner.forceDecodeToken(continuation[step]);
                if (!forced.success() || forced.tokens.size()!=1 || forced.tokens[0]!=continuation[step])
                    throw std::runtime_error("Forced production token failed: "+forced.error);
                // This experiment deliberately measures serial target arithmetic.
                // Its public deferred contract means the first call publishes a
                // pending token, and each successor forwards exactly that token.
                if (forced.returned_token_commit!=ReturnedTokenCommitState::Pending)
                    throw std::runtime_error("Unexpected forced-token commit contract");
                if (step>0) rows.push_back(saveLogits(runner,logits,prompt.size()+step));
            }
            logits.close();
            result["cases"].push_back({{"name",name},{"prompt_tokens",prompt.size()},
                {"rows",rows},{"continuation_ids",continuation},
                {"logits_file",(output/(name+".f32")).string()}});
            publish(output/"logits.json",result);
        }
        runner.shutdown();
        context.runner.reset();
        if (!PerfStatsCollector::flushFromEnv())
            throw std::runtime_error("Accuracy observer could not publish production-path counters");
        result["passed"]=true;
    }
    catch (const std::exception &error)
    {
        result["error"]=error.what();
        std::cerr << error.what() << '\n';
    }
    result["complete"]=true;
    if (!output.empty()) publish(output/"logits.json",result);
    return result["passed"].get<bool>() ? 0 : 1;
}
