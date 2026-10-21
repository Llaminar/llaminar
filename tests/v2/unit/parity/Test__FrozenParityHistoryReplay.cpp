/**
 * @file Test__FrozenParityHistoryReplay.cpp
 * @brief Device-free original history, reference and handoff rejection controls.
 *
 * Tiny owned files below are metadata test inputs, never GGUF/NPY execution or
 * model proof. They exercise the same checked request used by the real fixture;
 * no runner, native buffer, PMA certificate or numerical evidence is created.
 * Original binding publication also exercises the actual artifact reset helper,
 * rank-owner isolation and fatal metadata/filesystem failure status.
 */
#include "integration/parity/FrozenParityHistoryReplay.h"
#include "integration/parity/ParityTestBase.h"
#include "integration/parity/ParityArtifactDirectory.h"
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveParityCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "config/OrchestrationConfigParser.h"
#include <gtest/gtest.h>
#include <unistd.h>

namespace llaminar2::test::parity
{
namespace
{
using json = nlohmann::json;
constexpr const char *Cell = "Qwen4ExpMain/Qwen4ExpMainParityTest.ProductionParity/FrozenSynthetic";

/** @brief Retire only this test's bounded metadata directory. */
struct MetadataOwner
{
    std::filesystem::path root;
    /** @brief Create private metadata backing, without any device initialization. */
    MetadataOwner()
    {
        char pattern[] = "/tmp/qfn-frozen-replay-unit-XXXXXX";
        const char *path = ::mkdtemp(pattern);
        if (!path) throw std::runtime_error("Cannot create test metadata directory");
        root = path;
    }
    /** @brief Release the test's files after every success/failure exit. */
    ~MetadataOwner()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

/** @brief Build explicit tiny original-source stat metadata for validation only. */
json descriptor(const MetadataOwner &owner)
{
    json shards = json::array();
    for (int i = 0; i < 4; ++i)
    {
        const auto path = owner.root / ("part" + std::to_string(i) + ".gguf");
        { std::ofstream file(path); file << char('a' + i); }
        shards.push_back({{"name", path.filename().string()}, {"path", path.string()}, {"bytes", 1},
                          {"object_identity", accuracy::qfnOriginalSourceStat(path).dump()}});
    }
    const auto directory = owner.root / "reference";
    std::filesystem::create_directory(directory);
    const auto metadata = directory / "metadata.txt";
    { std::ofstream file(metadata); file << "token_ids: 1,2\ndecode_tokens: 3,4,5\n"; }
    const json source = {{"architecture", "qwen4exp"}, {"main_layers", 48}, {"expert_count", 512},
                         {"vocabulary", 248320}, {"reference_arithmetic", "cpu_fp32"},
                         {"reference_execution", "incremental_cached"}, {"shards", shards}};
    const auto config = OrchestrationConfig::defaults();
    const auto &gemm = debugEnv().gemm;
    const json arithmetic = {
        {"gdn_projection", std::string(gdnProjectionArithmeticName(gemm.gdn_projection_arithmetic))},
        {"moe_router", std::string(moeRouterArithmeticName(resolveMoERouterArithmetic(gemm.moe_router_arithmetic, "qwen4exp")))},
        {"hc_up", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_hc_up_activation))},
        {"gdn_output", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_gdn_output_activation))},
        {"fa_output", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_fa_output_activation))},
        {"gdn_qkv", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_gdn_qkv_activation))}};
    return {{"expected_config", json::parse(serializeOrchestrationConfig(config))}, {"arithmetic", arithmetic},
            {"schema", "llaminar.model-kld-history-replay.v1"}, {"canonical_cell", Cell},
            {"corpus_manifest_digest", std::string(64, 'a')}, {"source_digest", accuracy::qfnMetadataDigest(source)},
            {"reference_report_digest", std::string(64, 'b')}, {"full_stage_metadata_digest", *sha256FileHex(metadata)},
            {"source", source}, {"history_id", "original"}, {"context_group", "short"},
            {"prompt", "synthetic metadata only"}, {"prompt_ids", {1, 2}}, {"continuation_ids", {3, 4, 5}},
            {"decode_rows", 2}, {"reference_directory", directory.string()}, {"reference_scope", "full_stages"}};
}

/**
 * @brief Exercise the real pre-runner setup state without calling fixture SetUp.
 *
 * The tiny history and existing declared snapshot keys are metadata only. No
 * model, backend, source payload, runner or PMA certificate is constructed.
 */
class SnapshotAdmissionGeometryProbe final : public ParityTestBase
{
public:
    /** @brief Restore every original environment value after assertion unwind. */
    ~SnapshotAdmissionGeometryProbe() override
    {
        restoreParityEnvOverrides();
    }
    /** @brief This probe is manually scoped and never registered as a model fixture. */
    void TestBody() override {}
    /** @return A diagnostic label only, without a device initialization. */
    std::string getBackendName() override { return "MetadataOnlyAdmissionGeometry"; }
    /**
     * @brief Install original source-owned keys and an explicit inactive serving ladder.
     * @param rows Authored real prompt row count.
     * @param replay Optional already-validated frozen history.
     */
    void configure(int rows, std::optional<FrozenParityHistoryReplay> replay = std::nullopt)
    {
        config_.token_ids.assign(static_cast<std::size_t>(rows), 1);
        config_.graph_snapshot_policy = qwen4exp::qwen4ExpProductionDefinition(
            qwen4exp::Qwen4ExpParityTopology::CPU).graph_snapshot_policy;
        frozen_history_replay_ = std::move(replay);
        setScopedParityEnvOverride("LLAMINAR_PRODUCTION_PARITY", "0");
        setScopedParityEnvOverride("LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "64");
        setScopedParityEnvOverride("LLAMINAR_PREFILL_GRAPH_BUCKETS", "1");
        setScopedParityEnvOverride("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", "64,128,256,512");
    }
    /** @brief Expose the production pre-admission transaction, not a mirrored resolver. */
    using ParityTestBase::beginProductionParitySnapshotAdmission;
    using ParityTestBase::configureExactProductionParityPrefillGraphBucket;
    /** @return Effective diagnostic capacity from the same current startup inventory. */
    int retainedRows() const
    {
        return paritySnapshotServingGraphRows(static_cast<int>(config_.token_ids.size()),
            4096, debugEnv().execution.prefill_graph_bucket_sizes, true, MTPRuntimeConfig{});
    }
    /** @return Existing source-owned complete checkpoint policy, unchanged by geometry setup. */
    const ParityGraphSnapshotPolicy &snapshotPolicy() const { return config_.graph_snapshot_policy; }
};

/** @brief Parse a metadata request; no source or reference owner is granted. */
FrozenParityHistoryReplay checked(const json &value)
{
    return FrozenParityHistoryReplay::parse(value, accuracy::qfnMetadataDigest(value), Cell);
}
} // namespace

/** @test A partial environment locator cannot silently alter a canonical history. */
TEST(FrozenParityHistoryReplay, ExplicitCompleteCellAndDigestHandoff)
{
    EXPECT_FALSE(FrozenParityHistoryReplay::load("", "", "", Cell));
    EXPECT_THROW((void)FrozenParityHistoryReplay::load("file", "", "", Cell), std::invalid_argument);
    EXPECT_THROW((void)FrozenParityHistoryReplay::load("file", std::string(64, 'a'), "other", Cell), std::invalid_argument);
    MetadataOwner owner;
    const auto value = descriptor(owner);
    const auto file = owner.root / "replay.json";
    { std::ofstream output(file); output << value.dump(); }
    const auto digest = *sha256FileHex(file);
    EXPECT_TRUE(FrozenParityHistoryReplay::load(file.string(), digest, Cell, Cell));
    { std::ofstream output(file, std::ios::app); output << ' '; }
    EXPECT_THROW((void)FrozenParityHistoryReplay::load(file.string(), digest, Cell, Cell), std::invalid_argument);
}

/** @test LM-only, foreign-source and incomplete teachers reject before model admission. */
TEST(FrozenParityHistoryReplay, OriginalScopeSourceAndTrajectoryRequired)
{
    MetadataOwner owner;
    const auto original = descriptor(owner);
    EXPECT_NO_THROW((void)checked(original));
    for (int control = 0; control < 8; ++control)
    {
        auto bad = original;
        switch (control)
        {
        case 0: bad["reference_scope"] = "final_logits_only"; break;
        case 1: bad["canonical_cell"] = "foreign"; break;
        case 2: bad["continuation_ids"] = {3, 4}; break;
        case 3: bad["decode_rows"] = true; break;
        case 4: bad["prompt_ids"] = {true}; break;
        case 5: bad["source_digest"] = std::string(64, 'c'); break;
        case 6: bad["full_stage_metadata_digest"] = "missing"; break;
        case 7: bad["history_id"] = "../escape"; break;
        }
        EXPECT_THROW((void)checked(bad), std::invalid_argument);
    }
}

/** @test The exact frozen source and positive original observation contract cannot be substituted. */
TEST(FrozenParityHistoryReplay, ActualOriginalStatAndCanonicalWorkRemainBound)
{
    MetadataOwner owner;
    const auto original = descriptor(owner);
    const auto replay = checked(original);
    const auto main = original["source"]["shards"][0]["path"].get<std::string>();
    EXPECT_NO_THROW(replay.requireCanonicalSource(main, std::nullopt, 2));
    EXPECT_THROW(replay.requireCanonicalSource(main, std::nullopt, 0), std::invalid_argument);
    EXPECT_THROW(replay.requireCanonicalSource(original["source"]["shards"][1]["path"].get<std::string>(), std::nullopt, 2), std::invalid_argument);
    const auto replacement = owner.root / "replacement";
    { std::ofstream file(replacement); file << 'z'; }
    std::filesystem::rename(replacement, main);
    EXPECT_THROW(replay.requireCanonicalSource(main, std::nullopt, 2), std::invalid_argument);
}

/** @test Full-stage metadata owns the original prompt and every Pending teacher. */
TEST(FrozenParityHistoryReplay, ExactReferencePromptTeachersAndMetadata)
{
    MetadataOwner owner;
    const auto original = descriptor(owner);
    const auto replay = checked(original);
    const auto metadata = std::filesystem::path(replay.referenceDirectory()) / "metadata.txt";
    EXPECT_NO_THROW(replay.requireReference(metadata, {1, 2}, {3, 4, 5}));
    EXPECT_THROW(replay.requireReference(metadata, {1, 9}, {3, 4, 5}), std::invalid_argument);
    EXPECT_THROW(replay.requireReference(metadata, {1, 2}, {3, 9, 5}), std::invalid_argument);
    EXPECT_THROW(replay.requireReference(metadata, {1, 2}, {3, 4}), std::invalid_argument);
    const auto binding = owner.root / "history_replay_binding.json";
    EXPECT_THROW(replay.publishBinding(binding, replay.prompt(), {1, 9}, replay.referenceDirectory(), 2), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(binding));
    EXPECT_NO_THROW(replay.publishBinding(binding, replay.prompt(), replay.promptIds(), replay.referenceDirectory(), 2));
    json actual;
    std::ifstream(binding) >> actual;
    EXPECT_EQ(actual.at("descriptor_sha256"), replay.identity());
    EXPECT_EQ(actual.at("continuation_ids"), original.at("continuation_ids"));
    { std::ofstream file(metadata, std::ios::app); file << "changed\n"; }
    EXPECT_THROW(replay.requireReference(metadata, {1, 2}, {3, 4, 5}), std::invalid_argument);
}

/** @test Observer-only controls never replace canonical prefix/path work; placement changes reject. */
TEST(FrozenParityHistoryReplay, CanonicalIntentSurvivesObserverOnlyControls)
{
    MetadataOwner owner;
    auto value = descriptor(owner);
    auto observer = OrchestrationConfig::defaults();
    observer.prefix_cache.enabled = false;
    observer.max_seq_len = 2048;
    observer.n_predict = 3;
    observer.n_threads = 2;
    observer.prompt = "different observer control";
    observer.shared_predictor = SharedPredictorGGUFSource("/synthetic/inactive-metadata.gguf");
    value["expected_config"] = json::parse(serializeOrchestrationConfig(observer));
    auto canonical = OrchestrationConfig::defaults();
    canonical.prefix_cache.enabled = true;
    EXPECT_NO_THROW(checked(value).requireCanonicalConfiguration(canonical));
    canonical.tp_degree = 4;
    EXPECT_THROW(checked(value).requireCanonicalConfiguration(canonical), std::invalid_argument);
    canonical = OrchestrationConfig::defaults();
    canonical.kv_cache_precision = "fp32";
    EXPECT_THROW(checked(value).requireCanonicalConfiguration(canonical), std::invalid_argument);
    value["arithmetic"]["moe_router"] = "foreign";
    EXPECT_THROW(checked(value).requireCanonicalConfiguration(OrchestrationConfig::defaults()), std::invalid_argument);
}

/** @test Required stage geometry is taken from the actual original canonical definition. */
TEST(FrozenParityHistoryReplay, CanonicalAll48InventoryRemainsMandatory)
{
    MetadataOwner owner;
    const auto replay = checked(descriptor(owner));
    const auto definition = qwen4exp::qwen4ExpProductionDefinition(qwen4exp::Qwen4ExpParityTopology::CPU);
    const auto &keys = definition.graph_snapshot_policy.required_prefill_snapshot_keys;
    ASSERT_FALSE(keys.empty());
    EXPECT_THROW(replay.requireSnapshotInventory(keys, -1), std::invalid_argument);
    for (const auto &key : keys)
    {
        // Deliberately not valid NPY data: this tests only the metadata presence
        // guard, never numerical or full-reference qualification.
        std::ofstream file(std::filesystem::path(replay.referenceDirectory()) / (key + ".npy"));
        file << 'x';
    }
    EXPECT_NO_THROW(replay.requireSnapshotInventory(keys, -1));
    EXPECT_THROW(replay.requireSnapshotInventory(keys, 0), std::invalid_argument);
    std::filesystem::remove(std::filesystem::path(replay.referenceDirectory()) / (keys.back() + ".npy"));
    EXPECT_THROW(replay.requireSnapshotInventory(keys, -1), std::invalid_argument);
}
/** @test Explicit six-step work changes cardinality only and requires the later reference keys. */
TEST(FrozenParityHistoryReplay, SixStepObservationKeepsLimitsAndRequiresOffsetFive)
{
    MetadataOwner owner;
    auto value = descriptor(owner);
    value["decode_rows"] = 6;
    value["continuation_ids"] = {3, 4, 5, 6, 7, 8, 9};
    const auto replay = checked(value);
    int rows = 2;
    std::optional<FinalLogitMeanKLContract> contract = FinalLogitMeanKLContract{};
    const float prefill = contract->prefill_maximum, decode = contract->decode_maximum;
    EXPECT_NO_THROW(replay.applyObservationDepth(rows, contract));
    EXPECT_EQ(rows, 6);
    EXPECT_EQ(contract->decode_steps, 6);
    EXPECT_EQ(contract->prefill_maximum, prefill);
    EXPECT_EQ(contract->decode_maximum, decode);
    EXPECT_FALSE(contract->accepts(decode, false));
    EXPECT_THROW(replay.requireReference(std::filesystem::path(replay.referenceDirectory()) / "metadata.txt",
                                        {1, 2}, {3, 4, 5}), std::invalid_argument);
    const auto definition = qwen4exp::qwen4ExpProductionDefinition(qwen4exp::Qwen4ExpParityTopology::CPU);
    const auto &all_keys = definition.graph_snapshot_policy.required_decode_snapshot_keys;
    ASSERT_FALSE(all_keys.empty());
    const std::vector<std::string> bounded_key{all_keys.back()};
    for (int step = 0; step < 5; ++step)
    {
        std::ofstream file(std::filesystem::path(replay.referenceDirectory()) /
                           ("decode_step" + std::to_string(step) + "_" + bounded_key.front() + ".npy"));
        file << 'x'; // Metadata presence only; deliberately not a numerical pack.
        file.close();
        EXPECT_NO_THROW(replay.requireSnapshotInventory(bounded_key, step));
    }
    EXPECT_THROW(replay.requireSnapshotInventory(bounded_key, 5), std::invalid_argument);
    std::ofstream file(std::filesystem::path(replay.referenceDirectory()) / ("decode_step5_" + bounded_key.front() + ".npy"));
    file << 'x'; file.close();
    EXPECT_NO_THROW(replay.requireSnapshotInventory(bounded_key, 5));
    EXPECT_THROW(replay.requireSnapshotInventory(bounded_key, 6), std::invalid_argument);
    int divergent = 2;
    contract->decode_steps = 3;
    EXPECT_THROW(replay.applyObservationDepth(divergent, contract), std::invalid_argument);
    EXPECT_EQ(divergent, 2);
}

/** @test Actual replay metadata must prove the serving bucket, never only materialization. */
TEST(FrozenParityHistoryReplay, ProductionServingBucketRejectsExactLiveAndMissingReplay)
{
    MetadataOwner owner;
    auto value = descriptor(owner);
    auto config = OrchestrationConfig::defaults();
    config.prefill_max_bucket_size = 512;
    value["expected_config"] = json::parse(serializeOrchestrationConfig(config));
    const auto replay = checked(value);
    const auto capture = replay.servingPrefillCapture();
    EXPECT_EQ(capture.bucket_rows, prefillGraphBucketSizes(512));
    EXPECT_EQ(capture.minimum_rows, kDefaultPrefillGraphMinBucketSize);
    EXPECT_EQ(capture.selected_rows, capture.bucket_rows.front());
    PerfStatRecord record;
    record.kind = PerfStatRecord::Kind::Counter;
    record.domain = "forward_graph"; record.name = "prefill_graph_phase";
    record.phase = "prefill"; record.device = "ROCm:0"; record.count = 1; record.value = 1;
    record.tags = {{"capture_phase", "replay"}, {"real_token_count", "2"},
                   {"bucket_seq_len", std::to_string(capture.selected_rows)}};
    std::vector<PerfStatRecord> records{record};
    const auto proof = replay.servingPrefillEvidence(records, true);
    EXPECT_EQ(proof.at("physical_rows"), capture.selected_rows);
    EXPECT_EQ(proof.at("logical_rows"), 2);
    EXPECT_EQ(proof.at("native_replay_observed"), true);
    auto cpu = record;
    cpu.device = "CPU:0";
    cpu.tags["bucket_seq_len"] = "2";
    records.push_back(cpu); // A heterogeneous follower performs only its live CPU rows.
    EXPECT_NO_THROW((void)replay.servingPrefillEvidence(records, true));
    records.pop_back();
    records.front().tags["bucket_seq_len"] = "2";
    EXPECT_THROW((void)replay.servingPrefillEvidence(records, true), std::invalid_argument);
    records.front() = record;
    records.front().tags["capture_phase"] = "materialized_without_launch";
    EXPECT_THROW((void)replay.servingPrefillEvidence(records, true), std::invalid_argument);
    records.clear();
    EXPECT_THROW((void)replay.servingPrefillEvidence(records, true), std::invalid_argument);
    EXPECT_EQ(replay.servingPrefillEvidence(records, false).at("physical_rows"), 2);
    config.prefill_max_bucket_size = 1;
    value["expected_config"] = json::parse(serializeOrchestrationConfig(config));
    EXPECT_THROW((void)checked(value).servingPrefillCapture(), std::invalid_argument);
}
/** @test Explicit workers change only the public runner field and reject socket budgets. */
TEST(FrozenParityHistoryReplay, ExplicitWorkersRejectNarrowedOrBoundExecution)
{
    MetadataOwner owner;
    auto value = descriptor(owner);
    value["expected_config"]["configuration"]["n_threads"] = 4;
    value["expected_config"]["configuration"]["mpi_procs"] = 1;
    const auto replay = checked(value);
    auto config = OrchestrationConfig::defaults();
    const auto original = json::parse(serializeOrchestrationConfig(config));
    replay.applyCPUWorkers(config);
    auto projected = json::parse(serializeOrchestrationConfig(config));
    EXPECT_EQ(config.n_threads, 4);
    projected["configuration"]["n_threads"] = original["configuration"]["n_threads"];
    EXPECT_EQ(projected, original);
    FrozenParityHistoryReplay::CPUExecutionObservation actual{{0, 2, 4, 6}, 4, 4, false, true, "cores"};
    EXPECT_NO_THROW(replay.requireCPUExecution(actual));
    actual.physical_cores = 2;
    EXPECT_THROW(replay.requireCPUExecution(actual), std::invalid_argument);
    actual.physical_cores = 4;
    actual.max_workers = 2;
    EXPECT_THROW(replay.requireCPUExecution(actual), std::invalid_argument);
    actual.max_workers = 4;
    actual.unbound = false;
    EXPECT_THROW(replay.requireCPUExecution(actual), std::invalid_argument);
    actual.unbound = true;
    actual.dynamic = true;
    EXPECT_THROW(replay.requireCPUExecution(actual), std::invalid_argument);
    value["expected_config"]["configuration"]["n_threads"] = -1;
    EXPECT_THROW(checked(value).requestedCPUWorkers(), std::invalid_argument);
    value["expected_config"]["configuration"]["n_threads"] = 4;
    value["expected_config"]["configuration"]["mpi_procs"] = 2;
    EXPECT_THROW(checked(value).requestedCPUWorkers(), std::invalid_argument);
}

/** @test Real CPU worker records are required only where the declared topology computes on CPU. */
TEST(FrozenParityHistoryReplay, ActualWorkersRequiredForCPUButNotGPUOnly)
{
    MetadataOwner owner;
    auto value = descriptor(owner);
    auto config = OrchestrationConfig::defaults();
    config.n_threads = 4;
    config.mpi_procs = 1;
    config.device_mode = DeviceAssignmentMode::EXPLICIT;
    config.device_for_this_rank = GlobalDeviceAddress::cpu();
    value["expected_config"] = json::parse(serializeOrchestrationConfig(config));
    FrozenParityHistoryReplay::CPUExecutionObservation actual{{0, 2, 4, 6}, 4, 4, false, true, "cores"};
    auto replay = checked(value);
    std::vector<PerfStatRecord> records;
    EXPECT_THROW(replay.cpuExecutionEvidence(records, actual), std::invalid_argument);
    PerfStatRecord record;
    record.kind = PerfStatRecord::Kind::Counter;
    record.domain = "kernel";
    record.name = "cpu_moe_router_q8_worker_rows";
    record.phase = "moe";
    record.device = "cpu";
    record.count = 1;
    record.value = 1;
    record.tags = {{"workers", "4"}};
    records.push_back(record);
    EXPECT_TRUE(replay.cpuExecutionEvidence(records, actual).at("parallel_work_observed").get<bool>());
    records.front().tags["workers"] = "2";
    EXPECT_THROW(replay.cpuExecutionEvidence(records, actual), std::invalid_argument);
    records.front().tags["workers"] = "1";
    EXPECT_THROW(replay.cpuExecutionEvidence(records, actual), std::invalid_argument);
    config.device_for_this_rank = GlobalDeviceAddress::cuda(0);
    value["expected_config"] = json::parse(serializeOrchestrationConfig(config));
    const auto gpu = checked(value);
    const auto proof = gpu.cpuExecutionEvidence({}, actual);
    EXPECT_FALSE(proof.at("cpu_compute_required").get<bool>());
    EXPECT_FALSE(proof.at("parallel_work_observed").get<bool>());
    // A CPU expert domain still owns compute even with a native continuation.
    config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    RoutedExpertDomain domain;
    domain.name = "cpu-experts";
    domain.participants = {GlobalDeviceAddress::cpu()};
    config.moe_routed_expert_plan->domains.push_back(domain);
    value["expected_config"] = json::parse(serializeOrchestrationConfig(config));
    EXPECT_THROW(checked(value).cpuExecutionEvidence({}, actual), std::invalid_argument);
}

/**
 * @test Real CUDA/ROCm adaptive declarations retain the default serving cap identity.
 * @note Only tiny source/reference metadata is synthetic. Public canonical cases,
 *       exporter/parser, explicit observer cap and replay comparison are real.
 */
TEST(FrozenParityHistoryReplay, ActualAdaptiveCaseResolvesDefaultBucketAndRejectsDifferentCap)
{
    for (const auto &cell : qwen4exp::qwen4ExpGPUAdaptiveProductionCases())
    {
        SCOPED_TRACE(cell.testName());
        MetadataOwner owner;
        auto value = descriptor(owner);
        auto arguments = modelParityDiagnosticBenchmarkArguments(cell);
        arguments.insert(arguments.begin(), "frozen-history-unit");
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        const auto canonical = OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
        ASSERT_FALSE(canonical.prefill_max_bucket_size.has_value());
        auto observer = canonical;
        observer.prefill_max_bucket_size = kDefaultPrefillGraphMaxBucketSize;
        value["expected_config"] = json::parse(serializeOrchestrationConfig(observer));
        value["prompt"] = cell.model.prompt;
        value["prompt_ids"] = cell.model.token_ids;
        value["continuation_ids"] = {13, 271, 248045};
        const auto replay = checked(value);
        EXPECT_NO_THROW(replay.requireCanonicalConfiguration(canonical));
        const auto capture = replay.servingPrefillCapture();
        EXPECT_EQ(capture.bucket_rows, defaultPrefillGraphBucketSizes());
        EXPECT_EQ(capture.minimum_rows, kDefaultPrefillGraphMinBucketSize);
        EXPECT_EQ(capture.selected_rows, 64);
        EXPECT_EQ(cell.model.token_ids.size(), 9u);
        auto explicit_canonical = canonical;
        explicit_canonical.prefill_max_bucket_size = kDefaultPrefillGraphMaxBucketSize;
        value["expected_config"] = json::parse(serializeOrchestrationConfig(canonical));
        EXPECT_NO_THROW(checked(value).requireCanonicalConfiguration(explicit_canonical));
        // Even though this short prompt would still select64, a larger cap
        // changes retained workspace/graph geometry and cannot compare equal.
        observer.prefill_max_bucket_size = 1024;
        value["expected_config"] = json::parse(serializeOrchestrationConfig(observer));
        try
        {
            checked(value).requireCanonicalConfiguration(canonical);
            FAIL() << "Different explicit serving cap was accepted";
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("field=prefill_max_bucket_size"), std::string::npos);
            EXPECT_NE(message.find("canonical=null"), std::string::npos);
            EXPECT_NE(message.find("observer=1024"), std::string::npos);
        }
        observer = canonical;
        observer.kv_cache_precision = "fp32";
        value["expected_config"] = json::parse(serializeOrchestrationConfig(observer));
        try
        {
            checked(value).requireCanonicalConfiguration(canonical);
            FAIL() << "Different canonical arithmetic was accepted";
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("field=kv_cache_precision"), std::string::npos);
            EXPECT_NE(message.find("canonical=\"fp16\""), std::string::npos);
            EXPECT_NE(message.find("observer=\"fp32\""), std::string::npos);
        }
    }
}

} // namespace llaminar2::test::parity

namespace llaminar2::test::parity
{
/** @test Original binding survives real directory reset; the former pre-reset ordering loses it. */
TEST(FrozenParityHistoryReplay, ArtifactReadyPublicationSurvivesFreshDirectoryReset)
{
    MetadataOwner owner;
    const auto original = descriptor(owner);
    const auto replay = checked(original);
    const auto directory = owner.root / "artifacts";
    const auto binding = directory / "history_replay_binding.json";
    std::filesystem::create_directories(directory);
    replay.publishBinding(binding, replay.prompt(), replay.promptIds(), replay.referenceDirectory(), 2);
    // Negative control reproduces the former lifecycle using the actual publisher.
    const auto former = prepareParityArtifactDirectory(directory, true, [] {});
    ASSERT_EQ(former.ready, 1) << former.error;
    EXPECT_FALSE(std::filesystem::exists(binding));
    { std::ofstream stale(directory / "old.csv"); stale << "stale"; }
    int publications = 0;
    const auto actual = prepareParityArtifactDirectory(directory, true, [&] {
        ++publications;
        EXPECT_FALSE(std::filesystem::exists(directory / "old.csv"));
        replay.publishBinding(binding, replay.prompt(), replay.promptIds(), replay.referenceDirectory(), 2);
    });
    ASSERT_EQ(actual.ready, 1) << actual.error;
    EXPECT_EQ(publications, 1);
    std::ifstream input(binding);
    ASSERT_TRUE(input);
    json observed; input >> observed;
    EXPECT_EQ(observed.at("descriptor_sha256"), replay.identity());
    EXPECT_EQ(observed.at("canonical_cell"), original.at("canonical_cell"));
    EXPECT_EQ(observed.at("prompt_ids"), original.at("prompt_ids"));
    EXPECT_EQ(observed.at("continuation_ids"), original.at("continuation_ids"));
    EXPECT_EQ(observed.at("reference_directory"), original.at("reference_directory"));
    EXPECT_EQ(observed.at("decode_rows"), 2);
}

/** @test Peers never publish or touch the directory, and owner failures become a failed consensus input. */
TEST(FrozenParityHistoryReplay, ArtifactReadyPublicationRejectsValidationCloseAndResetFailures)
{
    MetadataOwner owner;
    const auto replay = checked(descriptor(owner));
    const auto directory = owner.root / "artifacts";
    const auto binding = directory / "history_replay_binding.json";
    int publications = 0;
    const auto peer = prepareParityArtifactDirectory(directory, false, [&] { ++publications; });
    EXPECT_EQ(peer.ready, 1);
    EXPECT_EQ(publications, 0);
    EXPECT_FALSE(std::filesystem::exists(directory));
    const auto invalid = prepareParityArtifactDirectory(directory, true, [&] {
        replay.publishBinding(binding, replay.prompt(), {9}, replay.referenceDirectory(), 2);
    });
    EXPECT_EQ(invalid.ready, 0);
    EXPECT_FALSE(invalid.error.empty());
    EXPECT_FALSE(std::filesystem::exists(binding));
    const auto close_failure = prepareParityArtifactDirectory(directory, true, [&] {
        std::filesystem::create_directory(binding);
        replay.publishBinding(binding, replay.prompt(), replay.promptIds(), replay.referenceDirectory(), 2);
    });
    EXPECT_EQ(close_failure.ready, 0);
    EXPECT_NE(close_failure.error.find("Cannot close"), std::string::npos);
    const auto blocked = owner.root / "blocked";
    { std::ofstream file(blocked); file << "not a directory"; }
    const auto reset_failure = prepareParityArtifactDirectory(blocked / "artifacts", true,
        [&] { ++publications; });
    EXPECT_EQ(reset_failure.ready, 0);
    EXPECT_FALSE(reset_failure.error.empty());
    EXPECT_EQ(publications, 0);
}


/** @test Standalone admission activates its production lifetime before exact-history setup. */
TEST(FrozenParityHistoryReplay, StandaloneSnapshotAdmissionUsesExactHistoryBeforeRunnerSetup)
{
    SnapshotAdmissionGeometryProbe probe;
    probe.configure(9);
    const auto keys = probe.snapshotPolicy().required_prefill_snapshot_keys;
    ASSERT_FALSE(keys.empty());
    // This is the original failing call order: inactive SetUp leaves the
    // serving maximum in place even though the oracle has only nine rows.
    ASSERT_NO_FATAL_FAILURE(probe.configureExactProductionParityPrefillGraphBucket());
    EXPECT_EQ(probe.retainedRows(), 512);
    ASSERT_NO_FATAL_FAILURE(probe.beginProductionParitySnapshotAdmission());
    EXPECT_EQ(probe.retainedRows(), 9);
    EXPECT_EQ(debugEnv().execution.prefill_graph_bucket_sizes, std::vector<int>{9});
    EXPECT_EQ(probe.snapshotPolicy().required_prefill_snapshot_keys, keys);
}

/** @test An explicit production-serving diagnosis retains M512 instead of exact-live buckets. */
TEST(FrozenParityHistoryReplay, StandaloneSnapshotAdmissionPreservesFrozenServingGeometry)
{
    MetadataOwner owner;
    auto value = descriptor(owner);
    auto config = OrchestrationConfig::defaults();
    config.prefill_max_bucket_size = 512;
    value["expected_config"] = json::parse(serializeOrchestrationConfig(config));
    const auto replay = checked(value);
    const auto serving = replay.servingPrefillCapture();
    SnapshotAdmissionGeometryProbe probe;
    probe.configure(2, replay);
    const auto keys = probe.snapshotPolicy().required_prefill_snapshot_keys;
    ASSERT_NO_FATAL_FAILURE(probe.beginProductionParitySnapshotAdmission());
    EXPECT_EQ(debugEnv().execution.prefill_graph_bucket_sizes, serving.bucket_rows);
    EXPECT_EQ(debugEnv().execution.prefill_graph_min_seq, serving.minimum_rows);
    EXPECT_EQ(probe.retainedRows(), 512);
    EXPECT_EQ(probe.snapshotPolicy().required_prefill_snapshot_keys, keys);
}

} // namespace llaminar2::test::parity
