/**
 * @file Test__ModelKLDQFNContract.cpp
 * @brief Fast device-free guards for the real QFN observer request/shape API.
 *
 * Configs and rank plans below are explicit synthetic control metadata. They
 * exercise only validation; no model, prepared owner, PMA grant, native graph
 * or runtime proof is minted. The same API is consumed by ModelKLDCanary.
 */
#include "tools/ModelKLDQFNContract.h"
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <limits>
#include <vector>
#include <unistd.h>

namespace llaminar2::accuracy
{
namespace
{
using json = nlohmann::json;

/** @brief Construct explicit CPU request intent without initialization or source reads. */
OrchestrationConfig cpuConfig()
{
    auto config = OrchestrationConfig::defaults();
    config.model_path = "/synthetic/qfn-00001-of-00004.gguf";
    config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
    config.mtp_activation_policy = MTPActivationPolicy::Disabled;
    config.mtp.enabled = false;
    config.prefix_cache.enabled = false;
    config.activation_precision = "fp32";
    config.kv_cache_precision = "fp16";
    config.prefill_max_bucket_size = 512;
    config.max_seq_len = 4096;
    return config;
}

/** @brief Reuse the actual complete config codec in one tiny synthetic corpus control. */
json descriptor(const OrchestrationConfig &config)
{
    const auto document = json::parse(serializeOrchestrationConfig(config));
    json shards = json::array();
    for (int i=0;i<4;++i)
        shards.push_back({{"name","part"+std::to_string(i)}, {"path","/synthetic/part"+std::to_string(i)},
            {"bytes",1}, {"object_identity","synthetic-not-original-source"}});
    return {{"schema","llaminar.model-kld-workload.v2"},{"topology_id","cpu"},
        {"corpus",{{"schema","llaminar.model-kld-corpus.v2"},
            {"metric","full_vocabulary_fp32_kl_reference_candidate_temperature1_v1"},
            {"temperature",1.0},{"vocabulary",248320},
            {"source",{{"architecture","qwen4exp"},{"main_layers",48},{"expert_count",512},
                {"vocabulary",248320},{"reference_arithmetic","cpu_fp32"},{"reference_execution","incremental_cached"},
                {"tokenizer_revision","synthetic-tokenizer-metadata"},{"shards",shards}}},
            {"histories",json::array({{{"id","first"},{"context_group","short"},{"prompt","synthetic"},
                {"prompt_ids",{1,2}},{"continuation_ids",{3,4,5}},{"decode_rows",2},
                {"reference_directory","/synthetic/reference"},{"reference_scope","full_stages"}}})},
            {"topologies",json::array({{{"id","cpu"},{"backend","CPU"},{"canonical_cell","synthetic"},
                {"expected_config",document},{"runtime_config_digest",qfnMetadataDigest(document)},
                {"arithmetic",{{"gdn_projection","fp16_moe_alpha_beta"},{"moe_router","fp16"},
                    {"hc_up","single_q8"},{"gdn_output","single_q8"},{"fa_output","single_q8"},{"gdn_qkv","single_q8"}}}}})},
            {"reference",{{"report","reference.json"},{"report_digest",std::string(64,'a')},{"format","float32_le_row_major"}}},
            {"reset_control",{{"history_id","first"},{"replay_id","reset"}}}}}};
}
} // namespace

/** @test The full actual config codec is the request boundary, not CLI fragments. */
TEST(ModelKLDQFNContract, ExactCompleteConfigRequired)
{
    const auto config=cpuConfig();
    const auto value=descriptor(config);
    EXPECT_NO_THROW((void)QFNObserverRequest::parse(value,config));
    auto changed=config;
    changed.max_seq_len=2048;
    EXPECT_THROW((void)QFNObserverRequest::parse(value,changed),std::invalid_argument);
    auto bad=value;
    bad["corpus"]["topologies"][0]["runtime_config_digest"]="foreign";
    EXPECT_THROW((void)QFNObserverRequest::parse(bad,config),std::invalid_argument);
}

/** @test Pending D+1 history is mandatory, and malformed IDs/names never become rows. */
TEST(ModelKLDQFNContract, CompleteOriginalPendingHistoryAndDistinctReset)
{
    const auto config=cpuConfig();
    const auto value=descriptor(config);
    for (int control=0;control<8;++control)
    {
        auto bad=value;
        auto &history=bad["corpus"]["histories"][0];
        switch(control)
        {
        case 0: history["continuation_ids"]={3,4}; break;
        case 1: history["decode_rows"]=true; break;
        case 2: history["prompt_ids"]={true}; break;
        case 3: history["continuation_ids"][1]=248320; break;
        case 4: history["id"]="../escape"; break;
        case 5: bad["corpus"]["reset_control"]["replay_id"]="first"; break;
        case 6: bad["corpus"]["histories"].push_back(history); break;
        case 7:
            history["id"]="different";
            bad["corpus"]["histories"].push_back(value["corpus"]["histories"][0]);
            break;
        }
        EXPECT_THROW((void)QFNObserverRequest::parse(bad,config),std::invalid_argument);
    }
}

/** @test Native policy/placement preconditions reject before any RuntimeInit call. */
TEST(ModelKLDQFNContract, UnsupportedPoliciesAndCrossRankIntentReject)
{
    for (int control=0;control<6;++control)
    {
        auto config=cpuConfig();
        switch(control)
        {
        case 0: config.mtp.enabled=true; break;
        case 1: config.prefix_cache.enabled=true; break;
        case 2: config.activation_precision="fp16"; break;
        case 3: config.kv_cache_precision="fp32"; break;
        case 4: config.tp_scope=TPScope::GLOBAL; break;
        case 5: config.prefill_max_bucket_size=256; break;
        }
        EXPECT_THROW((void)QFNObserverRequest::parse(descriptor(config),config),std::invalid_argument);
    }
}

/** @test Actual shape classification never grants an unsupported all-GPU pipeline protocol. */
TEST(ModelKLDQFNContract, OriginalPhysicalShapesAndGPUOnlyPPRejection)
{
    RankExecutionPlan plan;
    plan.primary_device=GlobalDeviceAddress::cpu(0);
    EXPECT_EQ(qfnObservationSurface(plan),QFNObservationSurface::CPU);
    plan.primary_device=GlobalDeviceAddress::rocm(0);
    EXPECT_EQ(qfnObservationSurface(plan),QFNObservationSurface::GPUCPUOverlay);
    for(int i=0;i<4;++i) plan.local_tp_devices.push_back(GlobalDeviceAddress::rocm(i));
    EXPECT_EQ(qfnObservationSurface(plan),QFNObservationSurface::ROCmLocalTP4);
    plan.local_tp_devices[3]=plan.local_tp_devices[0];
    EXPECT_THROW((void)qfnObservationSurface(plan),std::invalid_argument);
    plan.local_tp_devices.clear();
    plan.local_pp_devices={GlobalDeviceAddress::cuda(0),GlobalDeviceAddress::rocm(0),GlobalDeviceAddress::cpu(0)};
    EXPECT_EQ(qfnObservationSurface(plan),QFNObservationSurface::CPUTailPipeline);
    plan.local_pp_devices.clear();
    for(int i=0;i<4;++i) plan.local_pp_devices.push_back(GlobalDeviceAddress::rocm(i));
    EXPECT_THROW((void)qfnObservationSurface(plan),std::invalid_argument);
    plan.local_pp_devices.clear();
    plan.prev_rank=1;
    EXPECT_THROW((void)qfnObservationSurface(plan),std::invalid_argument);
}

/** @test Python sort/minify/ensure_ascii digest serialization survives Unicode and exponents. */
TEST(ModelKLDQFNContract, CanonicalMetadataTextMatchesPythonConvention)
{
    const auto value=json::parse(R"({"prompt":"\u00e9\ud83d\ude00","tiny":1e-6,"config":{"zero":0.0}})");
    const std::string canonical=R"({"config":{"zero":0.0},"prompt":"\u00e9\ud83d\ude00","tiny":1e-06})";
    EXPECT_EQ(value.dump(-1,' ',true),canonical);
    EXPECT_EQ(qfnMetadataDigest(value),*sha256BytesHex(canonical));
}
/** @test Source metadata, inactive paired declarations and frozen arithmetic stay exact. */
TEST(ModelKLDQFNContract, OriginalSourceAndPairedMetadataRemainExplicit)
{
    auto config = cpuConfig();
    config.shared_predictor = SharedPredictorGGUFSource("/synthetic/predictor.gguf");
    auto value = descriptor(config);
    value["corpus"]["source"]["shared_predictor"] = {{"name", "predictor.gguf"},
        {"path", "/synthetic/predictor.gguf"}, {"bytes", 1}, {"object_identity", "synthetic-only"}};
    EXPECT_NO_THROW((void)QFNObserverRequest::parse(value, config));
    for (int control = 0; control < 6; ++control)
    {
        auto bad = value;
        auto &source = bad["corpus"]["source"];
        switch (control)
        {
        case 0: source["shared_predictor"]["path"] = "/foreign/predictor.gguf"; break;
        case 1: source["shards"][1] = source["shards"][0]; break;
        case 2: source["shards"][0]["bytes"] = 0; break;
        case 3: source["tokenizer_revision"] = ""; break;
        case 4: bad["corpus"]["topologies"][0]["arithmetic"]["gdn_qkv"] = "unknown"; break;
        case 5: bad["corpus"]["reference"]["report_digest"] = "foreign"; break;
        }
        EXPECT_THROW((void)QFNObserverRequest::parse(bad, config), std::invalid_argument);
    }
}

/** @test Diagnostic variants retain explicit production policies without changing defaults. */
TEST(ModelKLDQFNContract, ExplicitDiagnosticArithmeticRemainsCanonical)
{
    const auto config = cpuConfig();
    auto value = descriptor(config);
    auto &arithmetic = value["corpus"]["topologies"][0]["arithmetic"];
    const auto original = arithmetic;
    for (const char *name : {"hc_up", "gdn_output", "fa_output", "gdn_qkv"})
    {
        arithmetic = original;
        arithmetic[name] = "two_term_q8";
        const auto request = QFNObserverRequest::parse(value, config);
        EXPECT_EQ(request.topology().at("arithmetic"), arithmetic);
    }
    for (const auto &[name, choices] :
         std::map<std::string, std::vector<std::string>>{
             {"gdn_projection", {"native", "fp32", "fp16", "fp16_alpha_beta", "fp16_moe_alpha_beta"}},
             {"moe_router", {"native", "fp16", "native_hadamard32"}}})
    {
        for (const auto &choice : choices)
        {
            arithmetic = original;
            arithmetic[name] = choice;
            EXPECT_NO_THROW((void)QFNObserverRequest::parse(value, config));
        }
    }
    for (const auto &[name, ignored] : original.items())
    {
        (void)ignored;
        for (const auto &invalid : {json(""), json("automatic"), json(true), json(nullptr)})
        {
            arithmetic = original;
            arithmetic[name] = invalid;
            EXPECT_THROW((void)QFNObserverRequest::parse(value, config), std::invalid_argument);
        }
        arithmetic = original;
        arithmetic.erase(name);
        EXPECT_THROW((void)QFNObserverRequest::parse(value, config), std::invalid_argument);
    }
    arithmetic = original;
    arithmetic["unrecorded_policy"] = "native";
    EXPECT_THROW((void)QFNObserverRequest::parse(value, config), std::invalid_argument);
}

/** @test Only actual ordinary TP4 one-row owner receipts can prove row freshness. */
TEST(ModelKLDQFNContract, OriginalTPPublicationSchemaAndBoundedWindow)
{
    PerfStatRecord original;
    original.domain = "sampling";
    original.name = "rank_resident_condition_host_logits_publications";
    original.device = "rank";
    original.phase = "decode";
    original.tags = {{"operation", "advanceOrdinaryMainConditionFromDeviceResidentLogicalState"},
                     {"participants", "4"}, {"rows", "1"}};
    original.count = 34;
    original.value = 34;
    std::vector<PerfStatRecord> records{original};
    EXPECT_EQ(qfnTPHostPublicationCount(records), 34);
    for (int control = 0; control < 7; ++control)
    {
        records = {original};
        auto &record = records.front();
        switch (control)
        {
        case 0: record.tags["operation"] = "advanceMTPMainConditionFromDeviceResidentLogicalState"; break;
        case 1: record.tags["participants"] = "2"; break;
        case 2: record.tags["rows"] = "2"; break;
        case 3: record.phase = "prefill"; break;
        case 4: record.value = 35; break;
        case 5: record.tags.erase("operation"); break;
        case 6:
            record.count = 8192;
            record.value = 8192;
            records.push_back(original);
            break;
        }
        EXPECT_THROW((void)qfnTPHostPublicationCount(records), std::invalid_argument);
    }
}
/** @test Original real file stat binding rejects every field drift and same-size replacement. */
TEST(ModelKLDQFNContract, OriginalStatIdentityRejectsSameSizeReplacement)
{
    char pattern[] = "/tmp/qfn-original-stat-unit-XXXXXX";
    const char *created = ::mkdtemp(pattern);
    ASSERT_NE(created, nullptr);
    struct TemporaryOwner
    {
        std::filesystem::path path;
        ~TemporaryOwner() { std::error_code error; std::filesystem::remove_all(path, error); }
    } owner{created};
    const auto path = owner.path / "original.gguf";
    const auto other = owner.path / "replacement.gguf";
    { std::ofstream file(path); file << 'a'; }
    { std::ofstream file(other); file << 'b'; }
    const auto frozen = qfnOriginalSourceStat(path);
    const json descriptor = {{"name", "original.gguf"}, {"path", path.string()}, {"bytes", 1},
                             {"object_identity", frozen.dump()}};
    EXPECT_EQ(qfnRequireOriginalSourceStat(descriptor, path), frozen);
    for (const char *field : {"device", "inode", "bytes", "mtime_ns", "ctime_ns"})
    {
        auto stale = descriptor;
        auto identity = frozen;
        identity[field] = identity[field].get<std::int64_t>() + 1;
        stale["object_identity"] = identity.dump();
        EXPECT_THROW((void)qfnRequireOriginalSourceStat(stale, path), std::invalid_argument);
        // JSON's numeric comparison may otherwise narrow nanosecond integers
        // to double and accept a timestamp that no longer identifies this file.
        identity = frozen;
        identity[field] = identity[field].get<double>();
        stale["object_identity"] = identity.dump();
        EXPECT_THROW((void)qfnRequireOriginalSourceStat(stale, path), std::invalid_argument);
    }
    std::filesystem::rename(other, path);
    ASSERT_EQ(std::filesystem::file_size(path), 1);
    EXPECT_THROW((void)qfnRequireOriginalSourceStat(descriptor, path), std::invalid_argument);
}
/** @test Real retained-parent vocabulary admits only a declared heterogeneous owner boundary. */
TEST(ModelKLDQFNContract, CanonicalHeterogeneousPathAndStrictTP4RemainDistinct)
{
    std::vector<PerfStatRecord> records;
    const auto add = [&](const char *name, const char *device, const char *phase,
                         PerfStatsCollector::Tags tags = {}) {
        PerfStatRecord record;
        record.domain = "forward_graph";
        record.name = name;
        record.device = device;
        record.phase = phase;
        record.tags = std::move(tags);
        record.count = 1;
        record.value = 1;
        records.push_back(std::move(record));
    };
    add("segmented_plan_segments", "", "decode");
    add("retained_parent_transaction_captures", "ROCm:0", "decode");
    add("retained_parent_replays", "ROCm:0", "decode");
    add("prefill_graph_phase", "ROCm:0", "prefill", {{"capture_phase", "capture"}});
    EXPECT_NO_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::GPUCPUOverlay, {"ROCm:0"}));
    EXPECT_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::ROCmLocalTP4, {"ROCm:0"}), std::invalid_argument);
    auto incomplete = records;
    incomplete.erase(incomplete.begin() + 2);
    EXPECT_THROW(qfnRequireGraphPathRecords(incomplete, QFNObservationSurface::GPUCPUOverlay, {"ROCm:0"}), std::invalid_argument);
    add("eager_decode_calls", "ROCm:0", "decode");
    EXPECT_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::GPUCPUOverlay, {"ROCm:0"}), std::invalid_argument);

    records.clear();
    add("full_graph_capture_executable_nodes", "ROCm:0", "decode");
    add("full_graph_replay_calls", "ROCm:0", "decode");
    add("decode_graph_phase", "ROCm:0", "decode", {{"phase", "replay"}});
    add("prefill_graph_phase", "ROCm:0", "prefill", {{"capture_phase", "capture"}});
    EXPECT_NO_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::ROCmLocalTP4, {"ROCm:0"}));
    EXPECT_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::GPUCPUOverlay, {"ROCm:0"}), std::invalid_argument);
}

namespace
{
/** @brief Author passive successful-completion metadata; this does not execute or certify a model. */
PerfStatRecord hostCompletion(const char *phase)
{
    PerfStatRecord record;
    record.domain = "forward_graph";
    record.name = "host_graph_completed_calls";
    record.device = DeviceId::cpu().toString();
    record.phase = phase;
    record.count = 1;
    record.value = 1;
    record.tags = {{"source","device_graph_executor"},{"authority","runtime_execution"},{"stages","7"}};
    return record;
}
} // namespace

/** @test Pure observer acceptance uses completed CPU prefill AND decode, not profiling/cache calls. */
TEST(ModelKLDQFNContract, CPUCompletionRequiresBothSuccessfulRuntimePhases)
{
    const std::vector<PerfStatRecord> valid{hostCompletion("prefill"),hostCompletion("decode")};
    EXPECT_NO_THROW(qfnRequireGraphPathRecords(valid,QFNObservationSurface::CPU,{}));
    EXPECT_THROW(qfnRequireGraphPathRecords({},QFNObservationSurface::CPU,{}),std::invalid_argument);
    for (int missing : {0,1})
    {
        auto records = valid;
        records.erase(records.begin()+missing);
        EXPECT_THROW(qfnRequireGraphPathRecords(records,QFNObservationSurface::CPU,{}),std::invalid_argument);
    }
    for (int mutation=0; mutation<16; ++mutation)
    {
        SCOPED_TRACE(mutation);
        auto records = valid;
        auto &record = records.back();
        switch (mutation)
        {
        case 0: record.device="ROCm:0"; break;
        case 1: record.device="CPU:0"; break; // Noncanonical device text is not actual CPU telemetry.
        case 2: record.phase="setup"; break;
        case 3: record.tags["authority"]="setup_addresses_only"; break;
        case 4: record.tags["source"]="facade"; break;
        case 5: record.tags["stages"]="0"; break;
        case 6: record.tags["stages"]="7x"; break;
        case 7: record.tags["stages"]="07"; break;
        case 8: record.tags.erase("stages"); break;
        case 9: record.count=0; break;
        case 10: record.value=0; break;
        case 11: record.value=std::numeric_limits<double>::infinity(); break;
        case 12: record.value=std::numeric_limits<double>::quiet_NaN(); break;
        case 13: record.value=.5; break;
        case 14: record.name="forward_cache_hit_total"; break;
        case 15: record.kind=PerfStatRecord::Kind::Timer; break;
        }
        EXPECT_THROW(qfnRequireGraphPathRecords(records,QFNObservationSurface::CPU,{}),std::invalid_argument);
    }
}

/** @test CPU-tail completion supplements, and never replaces, original per-GPU captured path proof. */
TEST(ModelKLDQFNContract, CPUTailRequiresHostPhasesAndUnchangedGPUContracts)
{
    std::vector<PerfStatRecord> records{hostCompletion("prefill"),hostCompletion("decode")};
    const auto add = [&](const char *name,const char *device,const char *phase,PerfStatsCollector::Tags tags={})
    {
        PerfStatRecord record;
        record.domain="forward_graph";record.name=name;record.device=device;record.phase=phase;
        record.tags=std::move(tags);record.count=1;record.value=1;
        records.push_back(std::move(record));
    };
    add("segmented_plan_segments","","decode");
    for (const auto *device : {"CUDA:0","ROCm:0"})
    {
        add("retained_parent_transaction_captures",device,"decode");
        add("retained_parent_replays",device,"decode");
        add("prefill_graph_phase",device,"prefill",{{"capture_phase","capture"}});
    }
    const std::set<std::string> gpu{"CUDA:0","ROCm:0"};
    EXPECT_NO_THROW(qfnRequireGraphPathRecords(records,QFNObservationSurface::CPUTailPipeline,gpu));
    auto missingHost = records;
    missingHost.erase(missingHost.begin()+1);
    EXPECT_THROW(qfnRequireGraphPathRecords(missingHost,QFNObservationSurface::CPUTailPipeline,gpu),std::invalid_argument);
    EXPECT_NO_THROW(qfnRequireGraphPathRecords(missingHost,QFNObservationSurface::GPUCPUOverlay,gpu));
    auto missingGPU = records;
    missingGPU.erase(missingGPU.begin()+4); // CUDA retained replay, leaving only captures.
    EXPECT_THROW(qfnRequireGraphPathRecords(missingGPU,QFNObservationSurface::CPUTailPipeline,gpu),std::invalid_argument);
}

/** @test Nested/throwing observer scopes restore exact prior metadata and reject ambiguous phase. */
TEST(ModelKLDQFNContract, ObservationPhaseScopesRestoreAfterNestedCallsAndExceptions)
{
    const auto original = GraphExecutorStats::currentPhase();
    {
        ScopedQFNObservationPhase prefill(ExecutionPhase::PREFILL);
        EXPECT_EQ(GraphExecutorStats::currentPhase(),ExecutionPhase::PREFILL);
        {
            ScopedQFNObservationPhase decode(ExecutionPhase::DECODE);
            EXPECT_EQ(GraphExecutorStats::currentPhase(),ExecutionPhase::DECODE);
        }
        EXPECT_EQ(GraphExecutorStats::currentPhase(),ExecutionPhase::PREFILL);
        EXPECT_THROW(([&] {
            ScopedQFNObservationPhase decode(ExecutionPhase::DECODE);
            EXPECT_EQ(GraphExecutorStats::currentPhase(),ExecutionPhase::DECODE);
            throw std::runtime_error("authored observer failure");
        })(),std::runtime_error);
        EXPECT_EQ(GraphExecutorStats::currentPhase(),ExecutionPhase::PREFILL);
        EXPECT_THROW((ScopedQFNObservationPhase(ExecutionPhase::COMBINED)),std::invalid_argument);
        EXPECT_EQ(GraphExecutorStats::currentPhase(),ExecutionPhase::PREFILL);
    }
    EXPECT_EQ(GraphExecutorStats::currentPhase(),original);
    EXPECT_THROW((ScopedQFNObservationPhase(static_cast<ExecutionPhase>(255))),std::invalid_argument);
    EXPECT_EQ(GraphExecutorStats::currentPhase(),original);
}

/** @test Shared parser intent retains placement/math while allowing explicit diagnostic controls. */
TEST(ModelKLDQFNContract, CanonicalIntentProjectionIsSharedAndRejectsChangedMathOrPlacement)
{
    const auto canonical = OrchestrationConfig::defaults();
    auto observed = canonical;
    observed.n_threads = 3;
    observed.prefix_cache.enabled = !canonical.prefix_cache.enabled;
    observed.prefill_max_bucket_size = kDefaultPrefillGraphMaxBucketSize;
    const auto arithmetic = qfnActualArithmeticIntent();
    EXPECT_EQ(qfnCanonicalConfigurationIntent(canonical), qfnCanonicalConfigurationIntent(observed));
    EXPECT_NO_THROW(qfnRequireCanonicalIntent(canonical, observed, arithmetic));
    observed.tp_degree = canonical.tp_degree + 1;
    EXPECT_THROW(qfnRequireCanonicalIntent(canonical, observed, arithmetic), std::invalid_argument);
    observed = canonical;
    observed.kv_cache_precision = canonical.kv_cache_precision == "fp16" ? "fp32" : "fp16";
    EXPECT_THROW(qfnRequireCanonicalIntent(canonical, observed, arithmetic), std::invalid_argument);
    observed = canonical;
    observed.prefill_max_bucket_size = 1024;
    EXPECT_THROW(qfnRequireCanonicalIntent(canonical, observed, arithmetic), std::invalid_argument);
    auto foreign = arithmetic;
    foreign["hc_up"] = "foreign";
    EXPECT_THROW(qfnRequireCanonicalIntent(canonical, canonical, foreign), std::invalid_argument);
}

} // namespace llaminar2::accuracy
