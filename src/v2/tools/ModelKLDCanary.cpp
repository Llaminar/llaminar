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
 * Dynamic CPU joint request reset compares a distinct late baseline and replay under one
 * original checked placement scope; Static retains its early/late byte oracle.
 * The ordinary numerical histories and actual movement remain unchanged.
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
#include "ModelKLDQFNOwner.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/ExpertGemmRegistry.h"
#include "execution/moe/CPUExpertShardSourceSelection.h"
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
#include <chrono>
#include <optional>
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

/**
 * @brief Save one fresh finite full-vocabulary row at its existing joined result boundary.
 * @param runner Original production runner owning the published logits.
 * @param output Caller-owned original history file stream.
 * @param position Exact consumed-token position of the original observation.
 * @param phase_begin Optional diagnostic host phase start, never a device timing event.
 * @return Existing row metadata and, when requested, elapsed host nanoseconds through
 * the original logits join. Validation, JSON construction and file I/O are excluded.
 * @throws std::runtime_error If the original result or output is incomplete.
 */
json saveLogits(IOrchestrationRunner &runner, std::ofstream &output, size_t position,
               std::optional<std::chrono::steady_clock::time_point> phase_begin = std::nullopt)
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
    const auto joined = std::chrono::steady_clock::now();
    const int vocabulary = runner.vocabSize();
    if (!logits || vocabulary <= 0)
        throw std::runtime_error("No complete host logits at the diagnostic result boundary");
    if (!std::all_of(logits, logits+vocabulary, [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("Nonfinite next-token logits");
    output.write(reinterpret_cast<const char *>(logits), vocabulary*sizeof(float));
    if (!output) throw std::runtime_error("Could not persist full-vocabulary row");
    json row={{"consumed_tokens",position}, {"argmax",std::max_element(logits,logits+vocabulary)-logits},
              {"vocabulary",vocabulary}};
    if (phase_begin)
        row["diagnostic_host_phase_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(joined-*phase_begin).count();
    return row;
}

/**
 * @brief Observe only real prepared/PMA/source owners after original RuntimeInit.
 * @param request Checked frozen diagnostic request, never an allocation authority.
 * @param runner Actual ready production runner; no test factory or owner mirror.
 * @param original_owner Sole exported contract retained by the complete observer lifetime.
 * @return Passive exact source/config/plan descriptors for the result artifact.
 * @throws std::invalid_argument For changed source, unsupported topology or missing owner proof.
 */
json authenticateQFN(const accuracy::QFNObserverRequest &request, IOrchestrationRunner &runner,
                     const ModelContextReuseContract &original_owner,bool joint_follower=false)
{
    using accuracy::requireQFN;
    const auto *owner = &original_owner;
    requireQFN(owner->context && owner->reuse_authority && owner->physical_memory_authority,
               "QFN observer needs the actual admitted model/prepared/PMA owner");
    auto &model = *owner->context;
    const auto profile = ModelMemoryProfile::fromLoader(model.concreteLoader());
    accuracy::requireQFNSourceGeometry({.runner_architecture=runner.architecture(),
        .model_architecture=model.architecture(),.main_layers=profile.n_layers,
        .source_vocabulary=profile.vocab_size,.experts=profile.expert_count,
        .runner_vocabulary=runner.vocabSize()},runner.config(),runner.executionPlan(),joint_follower);
    requireQFN(!joint_follower || bool(owner->prepared_routed_weight_plan),
               "QFN follower lacks original prepared source placement");
    auto surface = joint_follower
        ? accuracy::qfnCPUJointFollowerSurface(runner.config(),runner.executionPlan(),
            owner->prepared_weight_plan,*owner->prepared_routed_weight_plan)
        : accuracy::qfnObservationSurface(runner.executionPlan(),&runner.config());
    const auto &expected = request.corpus().at("source");
    requireQFN(model.path() == runner.config().model_path, "QFN original model path differs from actual parsed configuration");
    const auto paths = model.artifactPaths();
    json descriptors=expected.at("shards");
    if (runner.config().shared_predictor)
    {
        requireQFN(model.concreteLoader().ggufPredictorDirectory() &&
                   model.concreteLoader().predictorManifest() &&
                   model.concreteLoader().predictorMaterialization()==accuracy::qfnPredictorMaterialization(runner.config().mtp),
                   "QFN paired predictor materialization differs from original retained graph capacity");
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
    if(!joint_follower) {
        const auto placement = model.resolvedImmutableRowTablePlacement();
        requireQFN(bool(placement), "QFN observer original PLE placement is missing");
        placement->requireRequestedPolicy(runner.config().ngram_table_policy);
    } else requireQFN(accuracy::qfnDistributedCPUJointIntent(runner.config()) && surface==accuracy::QFNObservationSurface::CPUExpertFollower,
                      "QFN follower source authentication requires original distributed CPU joint intent");
    if (joint_follower)
    {
        // The fragment address comes from original prepared placement, never
        // the rank scheduler primary (which can carry a different NUMA label).
        const auto weights=model.concreteWeightManager();
        const auto store=weights?weights->preparedWeightStoreIfInitialized():nullptr;
        requireQFN(weights && store && store->physicalMemoryAuthority().get()==owner->physical_memory_authority.get() &&
            owner->prepared_routed_weight_plan && !owner->pipeline_prepared_plan,
            "QFN follower lacks original source/store/PMA/placement authority");
        const auto& prepared=*owner->prepared_routed_weight_plan;
        requireQFN(MoEExpertOwnerMap::usesCPUJointFragments(prepared) && prepared.first_model_layer==0 &&
            prepared.placements.size()==size_t(profile.n_layers),
            "QFN follower original fragment layer coverage differs");
        const auto participants=MoEExpertOwnerMap::declaredParticipants(prepared,{.reject_tensor_sharded_domains=false});
        size_t local_members=0;
        for(const auto& participant:participants) {
            if(participant.world_rank!=runner.executionPlan().rank || !participant.address.isCPU())continue;
            const auto domain=std::find_if(prepared.domains.begin(),prepared.domains.end(),[&](const auto& value){return value.name==participant.domain_name;});
            requireQFN(domain!=prepared.domains.end() && domain->routed_compute_policy==RoutedExpertComputePolicy::TensorSharded &&
                domain->participants.size()>1,"QFN follower original CPU member is not joint");
            const std::string identity="rank="+std::to_string(participant.world_rank)+" domain="+participant.domain_name+
                " member="+std::to_string(participant.domain_participant_index)+" fragment_numa="+std::to_string(participant.address.numa_node)+
                " primary_numa="+std::to_string(runner.executionPlan().primary_device.numa_node);
            const auto fail=[&](bool valid,const char* message){if(!valid)throw std::invalid_argument(std::string(message)+"; "+identity);};
            fail(participant.world_rank_known && participant.address.device_ordinal==0 && participant.address.numa_node>=0 &&
                accuracy::qfnAddressOwnedByRank(runner.executionPlan(),participant.address),"QFN follower original fragment endpoint differs");
            for(const auto& layer:prepared.placements) {
                const auto lease=weights->expertGemmRegistry().acquireCPUJointLayer(store->modelId(),participant,layer.layer);
                fail(lease && lease->cpu_joint_source,"QFN follower original prepared fragment lease missing");
                const auto& source=*lease->cpu_joint_source;
                fail(source.participant()==participant && source.layer()==layer.layer && source.cohort() &&
                    source.ownership().geometry().experts==profile.expert_count,"QFN follower prepared source/cohort differs");
                std::vector<int> expected;
                for(size_t expert=0;expert<layer.routed_expert_tier.size();++expert)
                    if(layer.routed_expert_tier[expert]==participant.tier_idx)expected.push_back(static_cast<int>(expert));
                fail(std::vector<int>(source.expertIds().begin(),source.expertIds().end())==expected &&
                    lease->experts.size()==layer.routed_expert_tier.size() && lease->sources.size()==expected.size()*3,
                    "QFN follower original selected expert coverage differs");
            }
            ++local_members;
        }
        requireQFN(local_members>0,"QFN follower has no original local prepared CPU joint member");
    }
    else if (surface == accuracy::QFNObservationSurface::CPU)
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
    else if (surface == accuracy::QFNObservationSurface::CPUTailPipeline ||
             surface == accuracy::QFNObservationSurface::FixedNativePipeline)
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
        if (surface == accuracy::QFNObservationSurface::HomogeneousLocalTP)
            surface=accuracy::qfnPreparedLocalTPSurface(runner.executionPlan(),*owner->prepared_routed_weight_plan);
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
    const auto path = std::filesystem::absolute(output/(name+".f32"));
    std::ofstream logits(path,std::ios::binary);
    json rows = json::array();
    const auto prefill_begin = std::chrono::steady_clock::now();
    {
        accuracy::ScopedQFNObservationPhase phase(ExecutionPhase::PREFILL);
        if (!runner.prefill(prompt)) throw std::runtime_error("QFN prefill failed: "+runner.lastError());
    }
    const auto prefill_observation=saveLogits(runner,logits,prompt.size(),prefill_begin);
    rows.push_back(qfnRow(history,-1,runner.vocabSize()));
    json phase_timings=json::array({{{"row_id",rows.back().at("row_id")},{"phase","prefill"},
        {"original_input_tokens",prompt.size()},
        {"host_wall_ns",prefill_observation.at("diagnostic_host_phase_ns")}}});
    for (std::size_t i=0;i<continuation.size();++i)
    {
        const auto decode_begin = std::chrono::steady_clock::now();
        const auto forced = [&] {
            accuracy::ScopedQFNObservationPhase phase(ExecutionPhase::DECODE);
            return runner.forceDecodeToken(continuation[i]);
        }();
        if (!forced.success() || forced.tokens.size()!=1 || forced.tokens[0]!=continuation[i] ||
            forced.returned_token_commit!=ReturnedTokenCommitState::Pending)
            throw std::runtime_error("QFN original Pending teacher transaction failed: "+forced.error);
        if (i>0)
        {
            const auto observation=saveLogits(runner,logits,prompt.size()+i,decode_begin);
            rows.push_back(qfnRow(history,int(i-1),runner.vocabSize()));
            phase_timings.push_back({{"row_id",rows.back().at("row_id")},{"phase","decode"},
                {"original_input_tokens",1},{"host_wall_ns",observation.at("diagnostic_host_phase_ns")}});
        }
    }
    logits.close();
    if (!logits || rows.size()!=history.at("decode_rows").get<std::size_t>()+1 ||
        std::filesystem::file_size(path)!=rows.size()*std::size_t(runner.vocabSize())*sizeof(float))
        throw std::runtime_error("QFN observation inventory/output is incomplete");
    // The first Pending teacher only stages a token; it is deliberately not
    // labeled as a completed decode. Every timed row uses the existing host
    // logits join, including its diagnostic D2H, without another native probe.
    return {{"history_id",name},{"logits_file",path.string()},{"rows",rows},
        {"diagnostic_phase_timings",{{"clock","steady_clock"},{"unit","nanoseconds"},
            {"scope","original_runner_call_through_existing_host_logits_join"},
            {"includes_diagnostic_full_vocabulary_d2h",true},
            {"includes_reset_validation_or_file_io",false},
            {"lookup_stall_timing",false},{"release_throughput_qualified",false},
            {"phases",std::move(phase_timings)}}}};
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
    const auto physical=accuracy::qfnObservationSurface(runner.executionPlan(),&runner.config());
    const auto surface=physical==accuracy::QFNObservationSurface::HomogeneousLocalTP
        ? static_cast<accuracy::QFNObservationSurface>(observation.at("surface").get<int>()):physical;
    accuracy::requireQFN(physical!=accuracy::QFNObservationSurface::HomogeneousLocalTP ||
        surface==accuracy::QFNObservationSurface::HomogeneousLocalTP || surface==accuracy::QFNObservationSurface::GPUCPULocalTP,
        "QFN observed TP path lost original prepared ownership classification");
    std::set<std::string> gpu;
    const auto add = [&](const GlobalDeviceAddress &address) {
        if (address.device_type==DeviceType::CUDA || address.device_type==DeviceType::ROCm)
            gpu.insert(address.toLocalDeviceId().toString());
    };
    for (const auto &address : runner.executionPlan().local_tp_devices) add(address);
    for (const auto &address : runner.executionPlan().local_pp_devices) add(address);
    if (surface==accuracy::QFNObservationSurface::FixedNativePipeline)
        for (const auto& stage:runner.executionPlan().local_pp_stage_tp_info)
            for (const auto& member:stage.devices) add(member);
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
 * @param participants Exact original admitted TP compute extent, independent of expert tier membership.
 * @return Validated cumulative actual one-row publications; no native probe.
 * @throws std::runtime_error If the required sampling family was not collected.
 */
std::uint64_t qfnTPHostPublications(std::size_t participants)
{
    if (!PerfStatsCollector::isDomainEnabled("sampling"))
        throw std::runtime_error("QFN LocalTP observer requires sampling publication evidence");
    const auto records = PerfStatsCollector::snapshot({"sampling"});
    return accuracy::qfnTPHostPublicationCount(records,participants);
}

/**
 * @brief Export passive completed per-rank path counters after original runner/model retirement.
 * @return Existing immutable terminal records only; no live probes or scheduling observations.
 * @note The caller owns final-retirement proof before entering this observation helper.
 */
json completedQFNRankPathRecords()
{
    json records=json::array();
    for(const auto& record:PerfStatsCollector::snapshot({"forward_graph","moe_overlay","moe_canonical_publication"}))
        if(record.count>0 || record.value>0)records.push_back({{"domain",record.domain},{"name",record.name},
            {"device",record.device},{"phase",record.phase},{"tags",record.tags},{"count",record.count},{"value",record.value}});
    return records;
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
    bool distributed_joint=false;int observation_rank=-1;
    try
    {
        using accuracy::requireQFN;
        auto config=OrchestrationConfigParser{}.parseArgs(engine_argc,engine_argv);
        const auto request=accuracy::QFNObserverRequest::parse(workload,config);
        const bool observed_pipeline=isObservedFixedNativePipeline(config);
        distributed_joint=accuracy::qfnDistributedCPUJointIntent(config);
        const bool original_multi_rank=distributed_joint || observed_pipeline;
        const int original_size=original_multi_rank?config.mpi_procs:1;
        if(!original_multi_rank)requireQFN(!std::filesystem::exists(output),"QFN observer output directory must be fresh");
        if (original_multi_rank || MPIBootstrapPhase::classifyDeviceIntent(config) == BootstrapDeviceIntent::CpuOnly)
        {
            // Reuse production's NUMA/physical-core launch before creating an
            // output owner or admitting a model. The child parses the exact
            // original wrapper/engine arguments again; no Python affinity map
            // or rewritten config can override the canonical verifier.
            requireQFN(original_multi_rank || (config.mpi_procs == 1 && config.hostfile.empty() &&
                !config.execution_rank_selection), "QFN CPU observer requires explicit one-rank local bootstrap");
            const auto launch = MPIBootstrap::detectMPIEnvironment();
            requireQFN(!launch.is_mpi_process || launch.detected_world_size == original_size,
                       "QFN observer rejects unexpected bootstrap membership");
            const auto bootstrap = MPIBootstrapPhase{}.execute(config, original_argc, original_argv,
                MPIBootstrapArgumentPolicy::PreserveOriginalArguments);
            if (bootstrap.action == BootstrapResult::Action::EXIT)
                return bootstrap.exit_code;
        }
        if(!original_multi_rank) {
            owns_output=std::filesystem::create_directories(output);
            requireQFN(owns_output,"QFN observer cannot create its output directory");
        }
        result["corpus_manifest_digest"]=accuracy::qfnMetadataDigest(request.corpus());
        result["source_digest"]=accuracy::qfnMetadataDigest(request.corpus().at("source"));
        result["reference_report_digest"]=request.corpus().at("reference").at("report_digest");
        for(const char *key:{"backend","runtime_config_digest","arithmetic"}) result[key]=request.topology().at(key);
        result["topology_id"]=request.topology().at("id");
        const auto launch=MPIBootstrap::detectMPIEnvironment();
        requireQFN(!launch.is_mpi_process || launch.detected_world_size==original_size,
                   "QFN observer rejects unresolved original MPI launch membership");
        const auto original_selection=config.execution_rank_selection;
        auto startup=RuntimeInitPhase().execute(config,engine_argc,engine_argv);
        if(observed_pipeline && !std::holds_alternative<AppContext>(startup)) {
            // The production initializer already retired this excluded rank's
            // original MPI session. It cannot publish numerical or model work.
            requireQFN(std::get<RuntimeInitExit>(startup)==RuntimeInitExit::Completed &&
                original_selection && launch.detected_rank>=0 &&
                launch.detected_rank<original_size && !original_selection->executionRank(launch.detected_rank),
                "QFN selected pipeline lost its active original runtime");
            requireQFN(std::filesystem::is_directory(output),
                "QFN inactive discovery rank has no terminal artifact owner");
            publish(output/("rank-"+std::to_string(launch.detected_rank)+".json"),
                {{"schema","llaminar.qfn-observed-pipeline-rank.v1"},{"discovery_rank",launch.detected_rank},
                 {"discovery_size",original_size},{"execution_rank",nullptr},{"active",false},
                 {"runtime_config_digest",request.topology().at("runtime_config_digest")},
                 {"corpus_manifest_digest",result.at("corpus_manifest_digest")},{"native_retired",true}});
            return 0;
        }
        requireQFN(std::holds_alternative<AppContext>(startup),"QFN runtime did not return its original model owner");
        auto context=std::move(std::get<AppContext>(startup));
        requireQFN(context.mpi_ctx && context.mpi_ctx->world_size()==(distributed_joint?config.mpi_procs:1),
                   "QFN observer lost original MPI membership");
        observation_rank=context.mpi_ctx->rank();
        requireQFN(bool(context.runner), "QFN observer lost its original runner");
        accuracy::requireQFNOriginalRank(context.runner->executionPlan(), observation_rank);
        const bool authority=context.coordinatedRequestRole()==CoordinatedRequestRole::Authority;
        if(observed_pipeline) {
            requireQFN(authority && original_selection &&
                original_selection->executionRank(launch.detected_rank)==observation_rank,
                "QFN fixed pipeline substituted its original selected request owner");
            owns_output=std::filesystem::create_directories(output);
            requireQFN(owns_output,"QFN observer cannot create its selected original output owner");
            result["observed_pipeline_binding"]=observeFixedNativePipelineBinding(*context.runner,*context.mpi_ctx);
        }
        if(distributed_joint) {
            requireQFN(context.runner->coordinatedRootRank()==0,"QFN CPU joint observer lost original rank-zero authority");
            const auto ready=RankInitializationLifecycle::execute({0,"qfn_observer_output"},[&] {
                if(authority) {owns_output=std::filesystem::create_directories(output);return owns_output;}
                return true;
            },[&](auto identity,auto outcome){return MPIRankInitializationConsensus::reach(MPI_COMM_WORLD,identity,outcome);});
            requireQFN(ready.succeeded(),"QFN observer cannot create its original output owner");
        }
        auto &runner=*context.runner;
        runner.setSkipLogitsGatherDecode(false); runner.setSkipLogitsGatherPrefill(false);
        requireQFN(runner.prepareForInference(),"QFN production preparation failed");
        json rank_record={{"schema","llaminar.qfn-observer-rank.v1"},{"rank",observation_rank},
            {"world_size",context.mpi_ctx->world_size()},{"authority_rank",runner.coordinatedRootRank()},
            {"role",authority?"authority":"follower"},{"runtime_config_digest",request.topology().at("runtime_config_digest")},
            {"corpus_manifest_digest",accuracy::qfnMetadataDigest(request.corpus())},{"complete",false}};
        const auto observe=[&](const ModelContextReuseContract &original_owner) {
        result["actual_owner"]=authenticateQFN(request,runner,original_owner);
        rank_record["actual_owner"]=result["actual_owner"];
        const auto executable=sha256FileHex("/proc/self/exe");
        requireQFN(bool(executable),"QFN observer cannot authenticate its actual executable");
        result["native_identity"]={{"executable_sha256",*executable},
            {"actual_config_sha256",accuracy::qfnMetadataDigest(result["actual_owner"].at("actual_config"))},
            {"actual_source_descriptor_sha256",result["actual_owner"].at("source_file_set_identity")}};
        const bool local_tp=accuracy::qfnObservationSurface(runner.executionPlan(),&runner.config())==accuracy::QFNObservationSurface::HomogeneousLocalTP;
        const auto publication_before=local_tp?qfnTPHostPublications(runner.executionPlan().local_tp_devices.size()):0;
        std::uint64_t expected_publications=0;
        std::uint64_t expected_prefill_rows=0;
        for(const auto &history:request.corpus().at("histories"))
        {
            result["cases"].push_back(observeQFNHistory(runner,history,output));
            expected_publications+=history.at("decode_rows").get<std::uint64_t>();
            expected_prefill_rows+=history.at("prompt_ids").size();
        }
        const auto &reset=request.corpus().at("reset_control");
        json baseline, replay;
        for(const auto &history:request.corpus().at("histories"))
            if(history.at("id")==reset.at("history_id")) { replay=history; break; }
        const json original_history = replay;
        replay["id"]=reset.at("replay_id");
        if (accuracy::qfnResetComparisonPolicy(runner.config()) ==
            accuracy::QFNResetComparisonPolicy::SamePlacementLatePair)
        {
            // The ordinary scored histories stay untouched. A moving placement
            // may use different floating arithmetic than its early occurrence;
            // prove reset with two new complete requests under the same original
            // checked parent. Each observe call retains its public clearCache.
            const auto baseline_output = output / "reset-baseline";
            requireQFN(std::filesystem::create_directory(baseline_output),
                       "QFN reset baseline requires a fresh independent artifact owner");
            const auto scope = runner.withMoEOverlayInferencePlacementScope([&] {
                baseline = observeQFNHistory(runner, original_history, baseline_output);
                result["reset_case"] = observeQFNHistory(runner, replay, output);
            });
            requireQFN(scope.placement_epoch > 0 && scope.next_sequence_id > scope.first_sequence_id &&
                       scope.next_sequence_id - scope.first_sequence_id >= 2,
                       "QFN reset pair did not complete both requests under its original placement scope");
            result["reset_pair"] = {
                {"schema", "llaminar.qfn-reset-pair.v1"},
                {"history_id", reset.at("history_id")}, {"baseline", baseline},
                {"placement_scope", {{"placement_epoch", scope.placement_epoch},
                    {"first_sequence_id", scope.first_sequence_id},
                    {"next_sequence_id", scope.next_sequence_id}}}};
            expected_publications += original_history.at("decode_rows").get<std::uint64_t>();
            expected_prefill_rows += original_history.at("prompt_ids").size();
        }
        else
        {
            // Static retains the original early/late exact-byte contract.
            for(const auto &value:result["cases"])
                if(value.at("history_id")==reset.at("history_id")) { baseline=value; break; }
            result["reset_case"]=observeQFNHistory(runner,replay,output);
        }
        expected_publications+=replay.at("decode_rows").get<std::uint64_t>();
        expected_prefill_rows+=replay.at("prompt_ids").size();
        if(local_tp)
        {
            const auto after=qfnTPHostPublications(runner.executionPlan().local_tp_devices.size());
            requireQFN(after>=publication_before && after-publication_before==expected_publications,
                       "QFN LocalTP did not publish every original committed row through its actual host gather");
            result["tp_host_publications"]={{"before",publication_before},{"after",after},
                {"observed_decodes_including_reset",expected_publications}};
        }
        requireQFN(qfnResetBytesEqual(baseline.at("logits_file").get<std::string>(),
                    result["reset_case"].at("logits_file").get<std::string>()),"QFN real request reset changed original logits bytes");
        result["path_records"]=requireQFNPath(runner,result["actual_owner"]);
        // Recheck original admitted source/policies before terminal teardown.
        requireQFN(authenticateQFN(request,runner,original_owner)==result["actual_owner"],"QFN original owner/config changed across histories");
        if(observed_pipeline) {
            requireQFN(observeFixedNativePipelineBinding(runner,*context.mpi_ctx)==result["observed_pipeline_binding"],
                "QFN original selected fixed owners changed across histories");
            // The first forced token stages its pending input and contributes
            // no model row. Every observed decode row is one later true input.
            result["fixed_native_pipeline_terminal"]=observeFixedNativePipelineTerminal(
                runner, expected_prefill_rows+expected_publications);
        }
        };
        const auto record_follower=[&](const ModelContextReuseContract& original_owner) {
            requireQFN(original_owner.prepared_routed_weight_plan && original_owner.physical_memory_authority &&
                original_owner.context,"QFN follower lost original admitted source/prepared/PMA authority");
            rank_record["actual_owner"]=authenticateQFN(request,runner,original_owner,true);
        };
        if(distributed_joint)
            accuracy::observeQFNCoordinatedWithFinalOwner(context.runner,authority,observe,record_follower);
        else accuracy::observeQFNWithFinalOwner(context.runner,observe);
        // Every rank reaches this publication only after its actual runner and
        // original model/native source retire. No follower writes final logits.
        if(distributed_joint) {
            rank_record["path_records"]=completedQFNRankPathRecords();rank_record["native_retired"]=true;rank_record["complete"]=true;
            publish(output/("rank-"+std::to_string(observation_rank)+".json"),rank_record);
            if(!authority)return 0;
            json ranks=json::array();
            for(int rank=0;rank<config.mpi_procs;++rank)ranks.push_back("rank-"+std::to_string(rank)+".json");
            result["distributed_observer"]={{"world_size",config.mpi_procs},{"authority_rank",0},{"rank_reports",ranks}};
        }
        if(observed_pipeline) {
            // The selected rank has retired the runner before publishing its
            // proof. Excluded ranks report only their original MPI retirement.
            const int original_owner=result.at("observed_pipeline_binding").at("original_owner").get<int>();
            publish(output/("rank-"+std::to_string(original_owner)+".json"),
                {{"schema","llaminar.qfn-observed-pipeline-rank.v1"},{"discovery_rank",original_owner},
                 {"discovery_size",original_size},{"execution_rank",observation_rank},{"active",true},
                 {"runtime_config_digest",request.topology().at("runtime_config_digest")},
                 {"corpus_manifest_digest",result.at("corpus_manifest_digest")},{"native_retired",true}});
        }
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
