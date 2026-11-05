/**
 * @file Test__ModelKLDQFNContract.cpp
 * @brief Fast device-free guards for the real QFN observer request/shape API.
 *
 * Configs and rank plans below are explicit synthetic control metadata. They
 * exercise only validation; no model, prepared owner, PMA grant, native graph
 * or runtime proof is minted. The same API is consumed by ModelKLDCanary.
 */
#include "tools/ModelKLDQFNContract.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
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
    EXPECT_EQ(qfnObservationSurface(plan),QFNObservationSurface::HomogeneousLocalTP);
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
    EXPECT_EQ(qfnTPHostPublicationCount(records,4), 34);
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
        EXPECT_THROW((void)qfnTPHostPublicationCount(records,4), std::invalid_argument);
    }
}
/** @test Persistent file identity rejects file drift/replacement and malformed device metadata. */
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
        if (std::string_view(field) == "device")
            EXPECT_EQ(qfnRequireOriginalSourceStat(stale, path), frozen);
        else
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
/** @test A reboot may renumber a source device; current complete runtime metadata is returned. */
TEST(ModelKLDQFNContract, OriginalStatDeviceRenumberingKeepsRuntimeIdentity)
{
    char pattern[] = "/tmp/qfn-device-renumber-unit-XXXXXX";
    const int fd = ::mkstemp(pattern);
    ASSERT_GE(fd, 0);
    struct FileOwner {
        int fd;
        const char *path;
        ~FileOwner() { ::close(fd); ::unlink(path); }
    } owner{fd, pattern};
    ASSERT_EQ(::write(fd, "x", 1), 1);
    const auto current = qfnOriginalSourceStat(pattern);
    auto reference = current;
    reference["device"] = current.at("device").get<std::uint64_t>() + 1;
    json descriptor{{"path", pattern}, {"bytes", 1}, {"object_identity", reference.dump()}};
    EXPECT_EQ(qfnRequireOriginalSourceStat(descriptor, pattern), current);
    EXPECT_NE(current, reference); // The owner can still detect a live mount change.
    for (const json bad : {json(-1), json(true), json(1.0), json("other")})
    {
        reference["device"] = bad;
        descriptor["object_identity"] = reference.dump();
        EXPECT_THROW((void)qfnRequireOriginalSourceStat(descriptor, pattern), std::invalid_argument);
    }
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
    EXPECT_NO_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::GPUCPULocalTP, {"ROCm:0"}));
    EXPECT_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::GPUCPULocalTP, {"ROCm:0","ROCm:1"}), std::invalid_argument);
    EXPECT_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::HomogeneousLocalTP, {"ROCm:0"}), std::invalid_argument);
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
    EXPECT_NO_THROW(qfnRequireGraphPathRecords(records, QFNObservationSurface::HomogeneousLocalTP, {"ROCm:0"}));
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

namespace llaminar2::accuracy {
/** @test Shape classification and prepared expert ownership remain separate original authorities. */
TEST(ModelKLDQFNContract, HomogeneousLocalTPUsesActualDegreeAndSelectedExpertOwnership)
{
    for(auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(int degree:{2,4,8}){
        RankExecutionPlan plan;
        for(int i=0;i<degree;++i)plan.local_tp_devices.push_back(backend==DeviceType::CUDA?GlobalDeviceAddress::cuda(i):GlobalDeviceAddress::rocm(i));
        plan.primary_device=plan.local_tp_devices.front();
        EXPECT_EQ(qfnObservationSurface(plan),QFNObservationSurface::HomogeneousLocalTP);
        MoERoutedExpertPlacementPlan prepared;
        RoutedExpertDomain gpu;gpu.name="gpu";gpu.participants=plan.local_tp_devices;
        RoutedExpertDomain cpu;cpu.name="cpu";cpu.participants={GlobalDeviceAddress::cpu(0)};
        prepared.domains={gpu,cpu};
        RoutedExpertTier g;g.domain="gpu";RoutedExpertTier c;c.domain="cpu";
        prepared.routed_tiers={g,c};prepared.placements.push_back({.layer=0,.routed_expert_tier={0,0}});
        EXPECT_EQ(qfnPreparedLocalTPSurface(plan,prepared),QFNObservationSurface::HomogeneousLocalTP);
        prepared.placements[0].routed_expert_tier[1]=1;
        EXPECT_EQ(qfnPreparedLocalTPSurface(plan,prepared),QFNObservationSurface::GPUCPULocalTP);
        auto bad=prepared;bad.domains[0].participants.push_back(gpu.participants[0]);
        EXPECT_THROW(qfnPreparedLocalTPSurface(plan,bad),std::invalid_argument);
        bad=prepared;bad.domains[0].participants[0]=backend==DeviceType::CUDA?GlobalDeviceAddress::rocm(0):GlobalDeviceAddress::cuda(0);
        EXPECT_THROW(qfnPreparedLocalTPSurface(plan,bad),std::invalid_argument);
        auto wrong=plan;wrong.local_tp_devices[1]=wrong.local_tp_devices[0];
        EXPECT_THROW(qfnObservationSurface(wrong),std::invalid_argument);
        wrong=plan;wrong.local_tp_devices[1].hostname="foreign.invalid";
        EXPECT_THROW(qfnObservationSurface(wrong),std::invalid_argument);
        wrong=plan;wrong.primary_device=GlobalDeviceAddress::cpu(0);
        EXPECT_THROW(qfnObservationSurface(wrong),std::invalid_argument);
        wrong=plan;
        if(degree==2)wrong.local_tp_devices.push_back(backend==DeviceType::CUDA?GlobalDeviceAddress::cuda(2):GlobalDeviceAddress::rocm(2));
        else wrong.local_tp_devices.resize(3);
        EXPECT_THROW(qfnObservationSurface(wrong),std::invalid_argument);
        PerfStatRecord receipt;receipt.domain="sampling";receipt.name="rank_resident_condition_host_logits_publications";
        receipt.kind=PerfStatRecord::Kind::Counter;receipt.device="rank";receipt.phase="decode";receipt.count=3;receipt.value=3;
        receipt.tags={{"operation","advanceOrdinaryMainConditionFromDeviceResidentLogicalState"},{"participants",std::to_string(degree)},{"rows","1"}};
        const std::array records{receipt};EXPECT_EQ(qfnTPHostPublicationCount(records,degree),3);
        EXPECT_THROW(qfnTPHostPublicationCount(records,degree==2?4:2),std::invalid_argument);
    }
}

/** @brief Build only parser/codec metadata; no source or prepared authority exists in this control. */
OrchestrationConfig distributedCPUJointConfig(int members) {
    auto config=cpuConfig();config.mpi_procs=members;config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
    config.device_for_this_rank.reset();
    auto plan=std::make_shared<MoERoutedExpertPlacementPlan>();plan->enabled=true;
    plan->continuation_domain="gpu";plan->base_model_domain="gpu";plan->shared_expert_domain="gpu";
    plan->residency_policy=RoutedExpertResidencyPolicy::StaticById;
    RoutedExpertDomain gpu;gpu.name="gpu";gpu.scope=ExecutionDomainScope::SINGLE;gpu.backend=CollectiveBackendType::HOST;
    gpu.owner_rank=0;gpu.participants={GlobalDeviceAddress::rocm(0)};
    RoutedExpertDomain cpu;cpu.name="cpu";cpu.scope=ExecutionDomainScope::NODE_LOCAL;cpu.backend=CollectiveBackendType::UPI;
    cpu.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    for(int member=0;member<members;++member){cpu.participants.push_back(GlobalDeviceAddress::cpu(member));cpu.world_ranks.push_back(member);}
    plan->domains={gpu,cpu};
    for(const auto &domain:plan->domains)
        config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain.toExecutionDomainDefinition()));
    config.moe_routed_expert_plan=std::move(plan);return config;
}
/**
 * @test Dynamic observation retains the same complete target/rank request contract.
 * @note Metadata proves observer admission only; actual movement/publication,
 * incoming-fragment use and retirement remain mandatory production evidence.
 */
TEST(ModelKLDQFNContract, DistributedCPUJointDynamicUsesOriginalHostAuthority)
{
    for (int members = 2; members <= 8; ++members)
    {
        for (const auto device : {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
        {
            auto config = distributedCPUJointConfig(members);
            config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
            auto &plan = *config.moe_routed_expert_plan;
            plan.residency_policy = RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.domains.front().participants = {device};
            plan.routed_tiers = {{.name="gpu", .domain="gpu", .priority=0, .max_experts_per_layer=64},
                                 {.name="cpu", .domain="cpu", .priority=10, .max_experts_per_layer=448}};
            config.domain_definitions.clear();
            for (const auto &domain : plan.domains)
                config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain.toExecutionDomainDefinition()));
            ASSERT_EQ(resolveMoEOverlayAuthorityExecutionKind(plan), MoEOverlayAuthorityExecutionKind::HostResident);
            EXPECT_TRUE(qfnDistributedCPUJointIntent(config));
            EXPECT_NO_THROW((void)QFNObserverRequest::parse(descriptor(config), config));
            const auto original = serializeOrchestrationConfig(config);
            EXPECT_NO_THROW((void)QFNObserverRequest::parse(descriptor(config), config));
            EXPECT_EQ(serializeOrchestrationConfig(config), original);
            plan.authority_execution = MoEOverlayAuthorityExecutionKind::HostResident;
            EXPECT_TRUE(qfnDistributedCPUJointIntent(config));
            plan.authority_execution = MoEOverlayAuthorityExecutionKind::DeviceResident;
            EXPECT_THROW(qfnDistributedCPUJointIntent(config), std::invalid_argument);
            plan.authority_execution = MoEOverlayAuthorityExecutionKind::Unresolved;
            plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
            EXPECT_THROW(qfnDistributedCPUJointIntent(config), std::invalid_argument);
            plan.residency_policy = RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
            EXPECT_THROW(qfnDistributedCPUJointIntent(config), std::invalid_argument);
            config.moe_rebalance.mode = MoERebalanceRuntimeMode::Observe;
            EXPECT_THROW(qfnDistributedCPUJointIntent(config), std::invalid_argument);
        }
    }
}
/** @test Exact distributed intent keeps the same full-history request contract for two through eight members. */
TEST(ModelKLDQFNContract, DistributedCPUJointUsesOriginalRanksAndFrozenHistory) {
    for(int members=2;members<=8;++members){auto config=distributedCPUJointConfig(members);
        EXPECT_TRUE(qfnDistributedCPUJointIntent(config));
        EXPECT_NO_THROW((void)QFNObserverRequest::parse(descriptor(config),config));
    }
    EXPECT_FALSE(qfnDistributedCPUJointIntent(cpuConfig()));
}
/** @test Foreign geometry and unmatched maintenance/residency intent remain rejected. */
TEST(ModelKLDQFNContract, DistributedCPUJointRejectsForeignPartialAndMigratingIntent) {
    for(int mutation=0;mutation<10;++mutation){auto config=distributedCPUJointConfig(2);auto& plan=*config.moe_routed_expert_plan;
        switch(mutation){
        case 0:plan.domains[1].routed_compute_policy=RoutedExpertComputePolicy::Apportioned;break;
        case 1:plan.domains[1].world_ranks[0]=-1;break;
        case 2:plan.domains[1].world_ranks[1]=0;break;
        case 3:plan.domains[1].participants[1]=plan.domains[1].participants[0];break;
        case 4:plan.domains[0].owner_rank=1;break;
        case 5:config.moe_rebalance.mode=MoERebalanceRuntimeMode::Dynamic;break;
        case 6:config.moe_rebalance.mode=MoERebalanceRuntimeMode::Observe;break;
        case 7:plan.domains[1].scope=ExecutionDomainScope::GLOBAL;break;
        case 8:plan.domains[1].world_ranks.pop_back();break;
        case 9:config.mpi_procs=4;break;
        }
        EXPECT_THROW(qfnDistributedCPUJointIntent(config),std::invalid_argument);
        EXPECT_THROW((void)QFNObserverRequest::parse(descriptor(config),config),std::invalid_argument);
    }
}

} // namespace llaminar2::accuracy

namespace llaminar2::accuracy {
/** @test Parse the literal world2 diagnostic CLI before traversing every request guard. */
TEST(ModelKLDQFNContract, DistributedCPUJointRealCLITraversesCompleteRequestContract) {
    auto arguments = nlohmann::json::parse(R"qfn([
    "qfn-observer-contract",
    "--mpi-procs",
    "2",
    "--activation-precision",
    "fp32",
    "--kv-cache-precision",
    "fp16",
    "--ngram-storage",
    "ram",
    "--ngram-mmap-window-bytes",
    "65536",
    "--ngram-mmap-window-count",
    "1024",
    "--mtp-graph-capacity-draft-tokens",
    "0",
    "--backend",
    "host",
    "--moe-routed-expert-placement",
    "tiered-overlay",
    "--moe-routed-expert-continuation-domain",
    "rocm_continuation",
    "--moe-routed-expert-base-model-domain",
    "rocm_continuation",
    "--moe-routed-expert-shared-domain",
    "rocm_continuation",
    "--moe-continuation-dense-policy",
    "replicated",
    "--moe-routed-expert-owner-order",
    "ordinal",
    "--moe-routed-expert-residency",
    "static-by-id",
    "--moe-routed-expert-domain",
    "rocm_continuation=localhost:-1:rocm:0;scope=single;backend=rccl;owner=0;expert_compute=apportioned;expert_phase=uniform;expert_decode_assignment=static-owner;expert_prefill_assignment=static-owner",
    "--moe-routed-expert-domain",
    "cpu_capacity=localhost:0:cpu:0,localhost:1:cpu:0;scope=node_local;backend=upi;ranks=0,1;expert_compute=tensor-sharded;expert_phase=uniform;expert_decode_assignment=static-owner;expert_prefill_assignment=static-owner",
    "--moe-routed-expert-tier",
    "tier_0@rocm_continuation;priority=0;max-experts-per-layer=64;memory-mb=auto",
    "--moe-routed-expert-tier",
    "tier_1@cpu_capacity;priority=10;max-experts-per-layer=448;memory-mb=auto",
    "--no-mtp",
    "--mtp-depth-policy",
    "fixed",
    "--moe-residency-maintenance",
    "off",
    "-m",
    "/opt/llaminar-models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "--mtp-model",
    "/opt/llaminar-models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
    "--context-length",
    "4096",
    "--threads",
    "28",
    "--no-prefix-cache",
    "--prefill-max-bucket-size",
    "512"
])qfn").get<std::vector<std::string>>();
    std::vector<char *> argv;for (auto &value : arguments) argv.push_back(value.data());
    OrchestrationConfigParser parser;
    const auto original=parser.parseArgs(static_cast<int>(argv.size()),argv.data());
    ASSERT_EQ(original.mpi_procs,2);ASSERT_EQ(original.domain_definitions.size(),2u);
    ASSERT_TRUE(qfnDistributedCPUJointIntent(original));
    const auto request=[](const OrchestrationConfig &config) {
        auto result=descriptor(config);
        if(config.shared_predictor) {
            const auto path=config.shared_predictor->path();
            result["corpus"]["source"]["shared_predictor"]={
                {"name",std::filesystem::path(path).filename().string()}, {"path",path},
                {"bytes",1}, {"object_identity","synthetic-not-original-source"}};
        }
        return result;
    };
    EXPECT_NO_THROW((void)QFNObserverRequest::parse(request(original),original));
    // Same literal source/transport/quotas; only the authored durable policy changes.
    auto dynamic_arguments = arguments;
    for (auto &argument : dynamic_arguments)
    {
        if (argument == "static-by-id") argument = "rebalanced";
        else if (argument == "off") argument = "dynamic";
    }
    std::vector<char *> dynamic_argv;
    for (auto &argument : dynamic_arguments) dynamic_argv.push_back(argument.data());
    const auto dynamic = parser.parseArgs(static_cast<int>(dynamic_argv.size()), dynamic_argv.data());
    EXPECT_EQ(dynamic.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
    ASSERT_EQ(dynamic.moe_routed_expert_plan->residency_policy, RoutedExpertResidencyPolicy::RoutedTierRebalanced);
    EXPECT_TRUE(qfnDistributedCPUJointIntent(dynamic));
    EXPECT_NO_THROW((void)QFNObserverRequest::parse(request(dynamic), dynamic));
    for(int mutation=0;mutation<8;++mutation) {
        auto candidate=original;
        switch(mutation) {
        case 0:candidate.domain_definitions.back().explicit_ranks={1,0};break;
        case 1:candidate.domain_definitions.back().scope=TPScope::GLOBAL;break;
        case 2:candidate.domain_definitions.back().routed_compute_policy=RoutedExpertComputePolicy::Apportioned;break;
        case 3:candidate.domain_definitions.back().devices[1]=candidate.domain_definitions.back().devices[0];break;
        case 4:candidate.domain_definitions.push_back(candidate.domain_definitions.back());break;
        case 5:candidate.domain_definitions.back().name="foreign";break;
        case 6:candidate.domain_definitions.pop_back();break;
        case 7:candidate.domain_definitions.back().backend=CollectiveBackendType::HOST;break;
        }
        // Bind the descriptor to the mutated complete config so rejection must
        // come from the actual domain guard, not config-document inequality.
        EXPECT_THROW((void)QFNObserverRequest::parse(request(candidate),candidate),std::invalid_argument);
    }
}
/** @test Original full-source geometry remains strict for expert-only worker observations. */
TEST(ModelKLDQFNContract, SourceGeometrySeparatesFollowerLogitCapability) {
    auto config=distributedCPUJointConfig(2);
    RankExecutionPlan plan;plan.rank=1;plan.primary_device=config.moe_routed_expert_plan->domains[1].participants[1];
    QFNSourceGeometry observed{.runner_architecture="qwen4exp",.model_architecture="qwen4exp",
        .main_layers=48,.source_vocabulary=248320,.experts=512,.runner_vocabulary=0};
    EXPECT_NO_THROW(requireQFNSourceGeometry(observed,config,plan,true));
    for(const auto field:{&QFNSourceGeometry::main_layers,&QFNSourceGeometry::source_vocabulary,
                         &QFNSourceGeometry::experts,&QFNSourceGeometry::runner_vocabulary}) {
        auto bad=observed;bad.*field=1;
        EXPECT_THROW(requireQFNSourceGeometry(bad,config,plan,true),std::invalid_argument);
    }
    for(const auto field:{&QFNSourceGeometry::runner_architecture,&QFNSourceGeometry::model_architecture}) {
        auto bad=observed;bad.*field="foreign";
        EXPECT_THROW(requireQFNSourceGeometry(bad,config,plan,true),std::invalid_argument);
    }
    // General rank scheduling metadata is not its routed source-fragment address.
    auto foreign=plan;foreign.primary_device.numa_node=0;
    EXPECT_NO_THROW(requireQFNSourceGeometry(observed,config,foreign,true));
    auto prepared=*config.moe_routed_expert_plan;
    RoutedExpertTier cpuTier;cpuTier.domain="cpu";cpuTier.name="cpu-fragments";
    prepared.routed_tiers={cpuTier};
    const auto participants=MoEExpertOwnerMap::declaredParticipants(prepared,
        {.reject_tensor_sharded_domains=false});
    const auto local=std::find_if(participants.begin(),participants.end(),[](const auto& value){return value.world_rank==1;});
    ASSERT_NE(local,participants.end());
    EXPECT_EQ(local->address.numa_node,1);
    EXPECT_NE(local->address.numa_node,foreign.primary_device.numa_node);
    foreign.primary_device.hostname="foreign-host";
    EXPECT_THROW(requireQFNSourceGeometry(observed,config,foreign,true),std::invalid_argument);
    foreign=plan;foreign.rank=0;
    EXPECT_THROW(requireQFNSourceGeometry(observed,config,foreign,true),std::invalid_argument);
    auto root=plan;root.rank=0;root.primary_device=GlobalDeviceAddress::rocm(0);
    EXPECT_THROW(requireQFNSourceGeometry(observed,config,root,false),std::invalid_argument);
    auto missing=cpuConfig();
    EXPECT_THROW(requireQFNSourceGeometry(observed,missing,plan,true),std::invalid_argument);
    observed.runner_vocabulary=248320;
    EXPECT_NO_THROW(requireQFNSourceGeometry(observed,config,plan,false));
    EXPECT_THROW(requireQFNSourceGeometry(observed,config,plan,true),std::invalid_argument);
}

} // namespace llaminar2::accuracy

namespace llaminar2::accuracy {
/** @test Observation retains the actual MPI rank and accepts only local aliases or its original canonical host. */
TEST(ModelKLDQFNContract, OriginalRankAndCanonicalHostOwnership) {
 for(const int rank:{0,1}) {
  RankExecutionPlan plan;plan.rank=rank;plan.hostname="original-inventory-node";
  EXPECT_NO_THROW(requireQFNOriginalRank(plan,rank));
  EXPECT_THROW(requireQFNOriginalRank(plan,-1),std::invalid_argument);
  EXPECT_THROW(requireQFNOriginalRank(plan,1-rank),std::invalid_argument);
  for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm}) {
   plan.primary_device=backend==DeviceType::CPU?GlobalDeviceAddress::cpu(0):
    backend==DeviceType::CUDA?GlobalDeviceAddress::cuda(0):GlobalDeviceAddress::rocm(0);
   for(const std::string host:{std::string{},std::string{"localhost"},plan.hostname}) {
    plan.primary_device.hostname=host;
    EXPECT_TRUE(qfnAddressOwnedByRank(plan,plan.primary_device));
    EXPECT_EQ(qfnObservationSurface(plan),backend==DeviceType::CPU?QFNObservationSurface::CPU:QFNObservationSurface::GPUCPUOverlay);
   }
   plan.primary_device.hostname="foreign-inventory-node";
   EXPECT_FALSE(qfnAddressOwnedByRank(plan,plan.primary_device));
   EXPECT_THROW(qfnObservationSurface(plan),std::invalid_argument);
   plan.primary_device.hostname="original-inventory-node.suffix";
   EXPECT_FALSE(qfnAddressOwnedByRank(plan,plan.primary_device));
   plan.primary_device.hostname=plan.hostname;auto missing=plan;missing.hostname.clear();
   EXPECT_THROW(qfnObservationSurface(missing),std::invalid_argument);
  }
 }
 RankExecutionPlan invalid;invalid.rank=-1;
 EXPECT_THROW(requireQFNOriginalRank(invalid,-1),std::invalid_argument);
 EXPECT_THROW(requireQFNOriginalRank(invalid,0),std::invalid_argument);
}
/** @test Canonical host matching is shared by TP, CPU-tail PP and original prepared expert-domain validation. */
TEST(ModelKLDQFNContract, CanonicalHostTPPPAndPreparedDomains) {
 for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8}) {
  RankExecutionPlan plan;plan.hostname="original-inventory-node";
  for(int ordinal=0;ordinal<degree;++ordinal) {
   auto address=backend==DeviceType::CUDA?GlobalDeviceAddress::cuda(ordinal):GlobalDeviceAddress::rocm(ordinal);
   address.hostname=plan.hostname;plan.local_tp_devices.push_back(address);
  }
  plan.primary_device=plan.local_tp_devices.front();
  EXPECT_EQ(qfnObservationSurface(plan),QFNObservationSurface::HomogeneousLocalTP);
  MoERoutedExpertPlacementPlan prepared;RoutedExpertDomain gpu;gpu.name="gpu";gpu.participants=plan.local_tp_devices;
  RoutedExpertDomain cpu;cpu.name="cpu";auto host=GlobalDeviceAddress::cpu(0);host.hostname=plan.hostname;cpu.participants={host};
  prepared.domains={gpu,cpu};RoutedExpertTier g;g.domain="gpu";RoutedExpertTier c;c.domain="cpu";
  prepared.routed_tiers={g,c};prepared.placements.push_back({.layer=0,.routed_expert_tier={0,1}});
  EXPECT_EQ(qfnPreparedLocalTPSurface(plan,prepared),QFNObservationSurface::GPUCPULocalTP);
  auto foreign=prepared;foreign.domains[1].participants[0].hostname="foreign-inventory-node";
  EXPECT_THROW(qfnPreparedLocalTPSurface(plan,foreign),std::invalid_argument);
  foreign=prepared;foreign.domains[0].participants[0].hostname="foreign-inventory-node";
  EXPECT_THROW(qfnPreparedLocalTPSurface(plan,foreign),std::invalid_argument);
  auto changed=plan;changed.local_tp_devices.back().hostname="foreign-inventory-node";
  EXPECT_THROW(qfnObservationSurface(changed),std::invalid_argument);
  changed=plan;changed.primary_device.hostname="foreign-inventory-node";
  EXPECT_THROW(qfnObservationSurface(changed),std::invalid_argument);
 }
 RankExecutionPlan pipeline;pipeline.hostname="original-inventory-node";
 pipeline.local_pp_devices={GlobalDeviceAddress::cuda(0),GlobalDeviceAddress::rocm(0),GlobalDeviceAddress::cpu(0)};
 for(auto& address:pipeline.local_pp_devices)address.hostname=pipeline.hostname;
 pipeline.primary_device=pipeline.local_pp_devices.front();
 EXPECT_EQ(qfnObservationSurface(pipeline),QFNObservationSurface::CPUTailPipeline);
 for(size_t participant=0;participant<pipeline.local_pp_devices.size();++participant) {
  auto foreign=pipeline;foreign.local_pp_devices[participant].hostname="foreign-inventory-node";
  EXPECT_THROW(qfnObservationSurface(foreign),std::invalid_argument);
 }
}
}

namespace llaminar2::accuracy {
/** @test Actual builder GlobalTP metadata is an expert-only follower, never a logit authority. */
TEST(ModelKLDQFNContract, CPUJointFollowerRetainsOriginalGlobalDomain) {
    for(int population=2;population<=8;++population) {
        auto config=distributedCPUJointConfig(population);
        ClusterInventory inventory;inventory.world_size=population;inventory.node_count=1;
        for(int rank=0;rank<population;++rank) {
            RankInventory local;local.rank=rank;local.local_rank=rank;local.node_id=0;
            local.hostname="original-node";local.cpu.numa_node=rank;
            local.cpu_sockets=population;local.numa_nodes=population;
            if(rank==0) {
                DeviceInfo gpu;gpu.type=DeviceType::ROCm;gpu.local_device_id=0;
                gpu.uuid="metadata-original-mi50";gpu.numa_node=0;
                local.gpus.push_back(gpu);
            }
            inventory.ranks.push_back(std::move(local));
        }
        ModelConfig model;model.name="qwen4exp";model.n_layers=48;model.n_heads=20;
        model.n_kv_heads=4;model.hidden_size=2560;model.intermediate_size=640;
        model.vocab_size=248320;model.head_dim=128;
        const auto plans=ExecutionPlanBuilder().buildAllPlans(config,model,inventory);
        ASSERT_EQ(plans.size(),size_t(population));
        auto prepared=*config.moe_routed_expert_plan;
        RoutedExpertTier gpu;gpu.name="hot";gpu.domain="gpu";
        RoutedExpertTier cpu;cpu.name="cold";cpu.domain="cpu";
        prepared.routed_tiers={gpu,cpu};
        for(int layer=0;layer<48;++layer) {
            RoutedExpertLayerPlacement placement;placement.layer=layer;
            placement.routed_expert_tier.assign(512,1);
            std::fill_n(placement.routed_expert_tier.begin(),64,0);
            prepared.placements.push_back(std::move(placement));
        }
        for(int rank=1;rank<population;++rank) {
            SCOPED_TRACE(population);
            SCOPED_TRACE(rank);
            const auto &plan=plans[size_t(rank)];
            ASSERT_TRUE(plan.usesGlobalTP());
            EXPECT_THROW(qfnObservationSurface(plan),std::invalid_argument);
            EXPECT_EQ(qfnCPUJointFollowerSurface(config,plan,plan,prepared),QFNObservationSurface::CPUExpertFollower);
            for(int mutation=0;mutation<10;++mutation) {
                auto wrong=plan;
                if(mutation==0)wrong.rank=0;
                if(mutation==1)wrong.global_tp_domain_id.reset();
                if(mutation==2)wrong.global_tp_domain_size+=1;
                if(mutation==3)wrong.my_domains.clear();
                if(mutation==4)wrong.my_domains.front().devices.back().numa_node=population;
                if(mutation==5)wrong.primary_device=GlobalDeviceAddress::rocm(0);
                if(mutation==6)wrong.local_tp_devices={GlobalDeviceAddress::cpu(0),GlobalDeviceAddress::cpu(1)};
                if(mutation==7)wrong.my_domains.front().domain_name="foreign";
                if(mutation==8)wrong.prev_rank=0;
                if(mutation==9)wrong.my_domains.front().devices.back().hostname="foreign-node";
                EXPECT_THROW(qfnCPUJointFollowerSurface(config,wrong,plan,prepared),std::invalid_argument);
            }
            auto wrong_source=prepared;wrong_source.domains[1].participants.back().numa_node=population;
            EXPECT_THROW(qfnCPUJointFollowerSurface(config,plan,plan,wrong_source),std::invalid_argument);
            wrong_source=prepared;wrong_source.placements.clear();
            EXPECT_THROW(qfnCPUJointFollowerSurface(config,plan,plan,wrong_source),std::invalid_argument);
            auto missing=config;missing.moe_routed_expert_plan.reset();
            EXPECT_THROW(qfnCPUJointFollowerSurface(missing,plan,plan,prepared),std::invalid_argument);
            auto foreign=plan;foreign.rank=population;
            EXPECT_THROW(qfnCPUJointFollowerSurface(config,foreign,foreign,prepared),std::invalid_argument);
        }
        EXPECT_THROW(qfnCPUJointFollowerSurface(config,plans[0],plans[0],prepared),std::invalid_argument);
    }
}
/** @test Reset selection uses the same typed CPU-joint host authority and preserves GPU Dynamic. */
TEST(ModelKLDQFNContract, ResetComparisonPolicyScopesOnlyOriginalDynamicCPUJoint)
{
    EXPECT_EQ(qfnResetComparisonPolicy(cpuConfig()),QFNResetComparisonPolicy::OriginalEarlyReplay);
    auto gpu=cpuConfig();gpu.device_for_this_rank.reset();gpu.mpi_procs=1;
    gpu.moe_rebalance.mode=MoERebalanceRuntimeMode::Dynamic;
    gpu.moe_routed_expert_plan=std::make_shared<MoERoutedExpertPlacementPlan>();
    auto &gpu_plan=*gpu.moe_routed_expert_plan;
    gpu_plan.enabled=true;gpu_plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
    gpu_plan.continuation_domain=gpu_plan.base_model_domain=gpu_plan.shared_expert_domain="gpu";
    gpu_plan.authority_execution=MoEOverlayAuthorityExecutionKind::DeviceResident;
    gpu_plan.residency_policy=RoutedExpertResidencyPolicy::RoutedTierRebalanced;
    RoutedExpertDomain domain;domain.name="gpu";domain.scope=ExecutionDomainScope::RANK_LOCAL;
    domain.backend=CollectiveBackendType::RCCL;domain.owner_rank=0;
    domain.routed_compute_policy=RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    for(int member=0;member<4;++member)domain.participants.push_back(GlobalDeviceAddress::rocm(member));
    gpu_plan.domains={domain};gpu_plan.routed_tiers={{.name="gpu",.domain="gpu",.priority=0,.max_experts_per_layer=512}};
    const auto before=serializeOrchestrationConfig(gpu);
    EXPECT_EQ(qfnResetComparisonPolicy(gpu),QFNResetComparisonPolicy::OriginalEarlyReplay);
    EXPECT_EQ(serializeOrchestrationConfig(gpu),before);
    for (int members=2;members<=8;++members)
    {
        auto joint=distributedCPUJointConfig(members);
        auto &plan=*joint.moe_routed_expert_plan;
        plan.routed_tiers={{.name="gpu",.domain="gpu",.priority=0,.max_experts_per_layer=64},
                           {.name="cpu",.domain="cpu",.priority=10,.max_experts_per_layer=448}};
        EXPECT_EQ(qfnResetComparisonPolicy(joint),QFNResetComparisonPolicy::OriginalEarlyReplay);
        joint.moe_rebalance.mode=MoERebalanceRuntimeMode::Dynamic;
        plan.residency_policy=RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        EXPECT_EQ(qfnResetComparisonPolicy(joint),QFNResetComparisonPolicy::SamePlacementLatePair);
        plan.authority_execution=MoEOverlayAuthorityExecutionKind::DeviceResident;
        EXPECT_THROW(qfnResetComparisonPolicy(joint),std::invalid_argument);
        plan.authority_execution=MoEOverlayAuthorityExecutionKind::HostResident;
        plan.domains.back().world_ranks.front()=members;
        EXPECT_THROW(qfnResetComparisonPolicy(joint),std::invalid_argument);
    }
}
} // namespace llaminar2::accuracy
