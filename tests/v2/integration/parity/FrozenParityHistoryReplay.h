/**
 * @file FrozenParityHistoryReplay.h
 * @brief Checked history-only projection into the original all-stage QFN fixture.
 *
 * The explicit exact-cell driver supplies a closed metadata descriptor and its
 * byte digest. This object owns only diagnostic request metadata: it cannot
 * select a topology, precision, numerical policy, allocation or runner. Explicit
 * replay also binds the public CPU worker request and verifies actual resource
 * observations; ordinary canonical launch resources remain unchanged. Source
 * stat and full reference metadata are checked before model admission. Existing
 * required checkpoint keys remain the canonical inventory authority. An explicit
 * noncertifying observation shard may select diagnostic storage/comparisons only
 * after the complete original gold inventory is authenticated.
 */
#pragma once
#include "tools/ModelKLDQFNContract.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include <map>
#include <algorithm>
#include "app/MPIBootstrapPhase.h"
#include <omp.h>
#include <sched.h>
#include <set>
#include "utils/DebugEnv.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "utils/PerfStatsCollector.h"
#include "FinalLogitMeanKLContract.h"
#include <charconv>
#include <cctype>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <optional>

namespace llaminar2::test::parity
{
/** @brief One immutable history override; construction requires the complete handoff. */
class FrozenParityHistoryReplay final
{
public:
    /** Maximum metadata size; this interface never reads model or snapshot payloads. */
    static constexpr std::uintmax_t MaximumDescriptorBytes = 16 * 1024 * 1024;

    /**
     * @brief Load an explicit driver handoff, rejecting partial ambient overrides.
     * @param file Original --history-replay file, empty for ordinary execution.
     * @param digest Exact original metadata-byte SHA256 from the explicit driver.
     * @param driver_cell Exact selected --exact-cell identity.
     * @param actual_cell Current original GTest suite/case/parameter identity.
     * @return No projection when all handoff fields are absent; otherwise checked metadata.
     * @throws std::invalid_argument For incomplete, foreign or changed handoff.
     */
    static std::optional<FrozenParityHistoryReplay> load(
        const std::string &file, const std::string &digest,
        const std::string &driver_cell, const std::string &actual_cell)
    {
        using accuracy::requireQFN;
        if (file.empty() && digest.empty() && driver_cell.empty()) return std::nullopt;
        requireQFN(!file.empty() && isDigest(digest) && !driver_cell.empty() && driver_cell == actual_cell,
                   "Frozen history requires the explicit exact-cell file/digest/cell handoff");
        const auto bytes = std::filesystem::file_size(file);
        requireQFN(bytes > 0 && bytes <= MaximumDescriptorBytes, "Frozen history descriptor extent is invalid");
        std::ifstream input(file, std::ios::binary);
        std::string text(static_cast<std::size_t>(bytes), '\0');
        input.read(text.data(), static_cast<std::streamsize>(text.size()));
        requireQFN(input && input.peek() == std::char_traits<char>::eof() &&
                   sha256BytesHex(text) == std::optional<std::string>(digest),
                   "Frozen history descriptor changed after explicit driver selection");
        const auto value = nlohmann::json::parse(text);
        return parse(value, digest, actual_cell);
    }

    /**
     * @brief Read only the test driver's explicit three-field handoff.
     * @param actual_cell Current original exact GTest identity.
     * @return Checked request or no override for the ordinary canonical matrix.
     * @throws std::invalid_argument For an environment locator without its typed descriptor.
     */
    static std::optional<FrozenParityHistoryReplay> fromDriver(const std::string &actual_cell)
    {
        const auto read = [](const char *key) {
            const char *value = std::getenv(key);
            return value ? std::string(value) : std::string{};
        };
        return load(read("LLAMINAR_PARITY_HISTORY_REPLAY_FILE"),
                    read("LLAMINAR_PARITY_HISTORY_REPLAY_SHA256"),
                    read("LLAMINAR_PARITY_HISTORY_REPLAY_CELL"), actual_cell);
    }

    /**
     * @brief Parse metadata only, without granting a model or reference owner.
     * @param value Explicit history descriptor derived from the frozen corpus.
     * @param digest Original handoff byte digest, used in reference/cache identity.
     * @param actual_cell Original canonical exact cell.
     * @return Checked immutable history selection.
     * @throws std::invalid_argument For schema, trajectory, source or scope errors.
     */
    static FrozenParityHistoryReplay parse(const nlohmann::json &value,
                                           const std::string &digest,
                                           const std::string &actual_cell)
    {
        using accuracy::requireQFN;
        requireQFN(value.at("schema") == "llaminar.model-kld-history-replay.v1" &&
                   isDigest(digest) && value.at("canonical_cell") == actual_cell &&
                   actual_cell.find(".ProductionParity/") != std::string::npos,
                   "Frozen history must name the selected original ProductionParity cell");
        for (const char *key : {"corpus_manifest_digest", "source_digest", "reference_report_digest", "full_stage_metadata_digest"})
            requireQFN(value.at(key).is_string() && isDigest(value.at(key).get<std::string>()),
                       "Frozen history is missing original corpus/source/reference identity");
        (void)deserializeOrchestrationConfig(value.at("expected_config").dump());
        accuracy::requireQFN(value.at("arithmetic").is_object(), "Frozen history requires its original arithmetic selectors");
        const auto &source = value.at("source");
        requireQFN(value.at("source_digest") == accuracy::qfnMetadataDigest(source) &&
                   source.at("architecture") == "qwen4exp" && source.at("main_layers") == 48 &&
                   source.at("expert_count") == 512 && source.at("vocabulary") == 248320 &&
                   source.at("reference_arithmetic") == "cpu_fp32" && source.at("reference_execution") == "incremental_cached" &&
                   source.at("shards").is_array() && source.at("shards").size() == 4,
                   "Frozen history needs its original complete QFN source");
        requireQFN(value.at("reference_scope") == "full_stages" &&
                   value.at("reference_directory").is_string() && !value.at("reference_directory").get<std::string>().empty(),
                   "Frozen history requires regenerated full-stage reference; LM-only pack is insufficient");
        requireQFN(accuracy::QFNObserverRequest::safeName(value.at("history_id").get<std::string>()) &&
                   accuracy::QFNObserverRequest::safeName(value.at("context_group").get<std::string>()) &&
                   value.at("prompt").is_string() && !value.at("prompt").get<std::string>().empty(),
                   "Frozen history has invalid original prompt identity");
        const auto &depth = value.at("decode_rows");
        requireQFN(depth.is_number_integer() && !depth.is_boolean() && depth.get<int>() > 0 && depth.get<int>() <= 256,
                   "Frozen history needs its original bounded decode depth");
        const auto &prompt = value.at("prompt_ids");
        const auto &teachers = value.at("continuation_ids");
        requireQFN(prompt.is_array() && !prompt.empty() && prompt.size() <= 4096 && teachers.is_array() &&
                   teachers.size() == std::size_t(depth.get<int>() + 1),
                   "Frozen history needs exact prompt and D+1 Pending teacher IDs");
        for (const auto *ids : {&prompt, &teachers})
            for (const auto &id : *ids)
                requireQFN(id.is_number_integer() && !id.is_boolean() && id.get<std::int64_t>() >= 0 &&
                           id.get<std::int64_t>() < 248320, "Frozen history has an invalid original token ID");
        if(value.contains("observation_shard")) {
            const auto& shard=value.at("observation_shard");
            requireQFN(shard.is_object() && shard.size()==4 && shard.at("schema")==1 &&
                shard.at("scope")=="noncertifying", "Frozen observation shard has incompatible scope/schema");
            for(const char* marker:{"LLAMINAR_PRODUCTION_PARITY","LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN"})
                requireQFN(!DebugEnv::isTruthyEnv(marker),
                    "Observation shards cannot enter complete production campaigns");
            const auto full=shard.at("full_keys").get<std::vector<std::string>>();
            const auto selected=shard.at("selected_keys").get<std::vector<std::string>>();
            const auto valid=[](const auto& keys) {
                if(keys.empty())return false;
                std::set<std::string> unique;
                for(const auto& key:keys) {
                    if(key.empty() || !unique.insert(key).second ||
                       !std::all_of(key.begin(),key.end(),[](unsigned char c){return std::isalnum(c)||c=='_';}) ||
                       key.starts_with("decode_step"))return false;
                }
                return true;
            };
            requireQFN(valid(full) && valid(selected) && selected.size()<full.size(),
                "Observation shard requires duplicate-free exact keys and incomplete coverage");
            const std::set<std::string> inventory(full.begin(),full.end()),selection(selected.begin(),selected.end());
            requireQFN(std::includes(inventory.begin(),inventory.end(),selection.begin(),selection.end()) &&
                selection.contains("EMBEDDING") && selection.contains("LM_HEAD"),
                "Observation shard selected foreign keys or omitted terminal/global evidence");
        }
        return FrozenParityHistoryReplay(value, digest);
    }

    /**
     * @brief Compare only original canonical placement/math intent, never observer workload controls.
     * @param canonical Public parser result of the actual case's existing diagnostic export.
     * @throws std::invalid_argument For changed source placement, capacity, topology or arithmetic.
     *
     * Both operands use the same lossless document codec. The retained fields
     * are actual physical/topology, expert-capacity, storage and math intent.
     * Prompt/prefix/context/generation controls and an inactive paired
     * metadata declaration deliberately do not replace the fixture's policies.
     * Explicit replay applies the checked public worker count separately.
     */
    void requireCanonicalConfiguration(const OrchestrationConfig &canonical) const
    {
        const auto expected = deserializeOrchestrationConfig(value_.at("expected_config").dump());
        accuracy::qfnRequireCanonicalIntent(canonical, expected, value_.at("arithmetic"));
    }

    /**
     * @brief Authenticate the unchanged canonical source before applying history fields.
     * @param model Original canonical main source path, before tmpfs staging resolution.
     * @param predictor Canonically declared paired path, if any; never added by replay.
     * @param decode_rows Positive original cell observation depth, before explicit extension.
     * @throws std::invalid_argument For source replacement or changed canonical work.
     */
    void requireCanonicalSource(const std::string &model,
                                const std::optional<SharedPredictorGGUFSource> &predictor,
                                int decode_rows) const
    {
        using accuracy::requireQFN;
        requireQFN(decode_rows > 0, "Frozen history requires a positive original canonical observation depth");
        const auto &source = value_.at("source");
        requireQFN(std::filesystem::canonical(model).string() == source.at("shards").at(0).at("path").get<std::string>(),
                   "Frozen history belongs to a different original canonical model");
        std::set<std::string> paths;
        for (const auto &descriptor : source.at("shards"))
        {
            const auto stat = accuracy::qfnRequireOriginalSourceStat(descriptor, descriptor.at("path").get<std::string>());
            requireQFN(paths.insert(stat.at("path").get<std::string>()).second,
                       "Frozen original main-source descriptors are duplicated");
        }
        if (source.contains("shared_predictor"))
        {
            const auto &descriptor = source.at("shared_predictor");
            (void)accuracy::qfnRequireOriginalSourceStat(descriptor, descriptor.at("path").get<std::string>());
            if (predictor)
                requireQFN(std::filesystem::canonical(predictor->path()).string() == descriptor.at("path").get<std::string>(),
                           "Frozen history names a different original inactive predictor");
        }
        else requireQFN(!predictor, "Frozen history omitted the canonical paired source");
    }

    /** @brief Passive current-process CPU resource facts, never a worker/affinity authority. */
    struct CPUExecutionObservation
    {
        std::vector<int> allowed_cpus; ///< Actual Linux process affinity, in ascending order.
        int physical_cores = 0; ///< Unique canonical physical representatives in that affinity.
        int max_workers = 0; ///< Actual current OpenMP maximum team size.
        bool dynamic = true; ///< Actual dynamic-team setting; diagnostic replay requires false.
        bool unbound = false; ///< Actual OpenMP proc-bind policy, not an environment inference.
        std::string places; ///< Before-exec OpenMP place declaration.
    };

    /**
     * @brief Read the original public worker request for one rank, CPU joint or observed fixed pipeline.
     * @return Explicit positive worker count from the original lossless configuration.
     * @throws std::invalid_argument Without positive workers and a supported original typed rank population.
     */
    int requestedCPUWorkers() const
    {
        const auto config = deserializeOrchestrationConfig(value_.at("expected_config").dump());
        accuracy::requireQFN(config.n_threads > 0 && (config.mpi_procs == 1 || accuracy::qfnDistributedCPUJointIntent(config) ||
                             isObservedFixedNativePipeline(config)),
                             "Frozen replay requires explicit positive workers on its original supported rank population");
        return config.n_threads;
    }

    /**
     * @brief Project only public n_threads into the actual canonical runner request.
     * @param config Original canonical runner configuration, otherwise left unchanged.
     */
    void applyCPUWorkers(OrchestrationConfig &config) const
    {
        config.n_threads = requestedCPUWorkers();
    }

    /**
     * @brief Read current CPU resources without starting a team, model or device.
     * @return Actual affinity and OpenMP ICVs; physical identity uses canonical bootstrap code.
     * @throws std::invalid_argument If Linux cannot provide a complete physical CPU mask.
     */
    static CPUExecutionObservation observeCPUExecution()
    {
        for (const char *name : {"GOMP_CPU_AFFINITY", "KMP_HW_SUBSET", "KMP_PLACE_THREADS", "OMP_THREAD_LIMIT"})
        {
            const char *value = std::getenv(name);
            accuracy::requireQFN(!value || !*value,
                                 "Frozen replay inherited an additional CPU resource control");
        }
        CPUExecutionObservation result;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        accuracy::requireQFN(::sched_getaffinity(0, sizeof(mask), &mask) == 0,
                             "Frozen replay cannot observe original process CPU affinity");
        std::set<int> physical;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        {
            if (!CPU_ISSET(cpu, &mask)) continue;
            const int representative = MPIBootstrapPhase::physicalRepresentativeForCpu(cpu);
            accuracy::requireQFN(representative >= 0, "Frozen replay cannot resolve physical CPU resources");
            result.allowed_cpus.push_back(cpu);
            physical.insert(representative);
        }
        result.physical_cores = static_cast<int>(physical.size());
        result.max_workers = omp_get_max_threads();
        result.dynamic = omp_get_dynamic() != 0;
        result.unbound = omp_get_proc_bind() == omp_proc_bind_false;
        const char *places = std::getenv("OMP_PLACES");
        result.places = places ? places : "";
        return result;
    }

    /**
     * @brief Reject narrowed/oversubscribed/bound CPU resources before model admission.
     * @param observed Actual process facts; Unit tests inject only this passive observation.
     * @throws std::invalid_argument For a resource policy differing from the frozen observer.
     */
    void requireCPUExecution(const CPUExecutionObservation &observed) const
    {
        const int workers = requestedCPUWorkers();
        accuracy::requireQFN(!observed.allowed_cpus.empty() &&
                             std::is_sorted(observed.allowed_cpus.begin(), observed.allowed_cpus.end()) &&
                             std::adjacent_find(observed.allowed_cpus.begin(), observed.allowed_cpus.end()) == observed.allowed_cpus.end() &&
                             observed.physical_cores > 0 && observed.physical_cores <= static_cast<int>(observed.allowed_cpus.size()) &&
                             workers <= observed.physical_cores && observed.max_workers == workers &&
                             !observed.dynamic && observed.unbound && observed.places == "cores",
                             "Frozen replay CPU worker count or unbound physical affinity differs from declared workload");
    }

    /**
     * @brief Install the existing public n_threads ICV before canonical CPU setup.
     * @throws std::invalid_argument If the driver did not establish representable resources.
     * @note MPI/places are established before exec, never repaired after worker pools exist.
     */
    void configureCPUExecution() const
    {
        // Do not silently fix an incorrectly launched process. The inherited
        // OpenMP policy must already match before applying the public request.
        requireCPUExecution(observeCPUExecution());
        auto config = OrchestrationConfig::defaults();
        applyCPUWorkers(config);
        MPIBootstrapPhase::configureRequestedCPUThreads(config);
        requireCPUExecution(observeCPUExecution());
    }

    /**
     * @brief Join declared CPU compute with real production worker-row submissions.
     * @param records Original CPU quantizer counters observed at the compared prefill.
     * @param observed Actual current resource observation, also checked before admission.
     * @return Passive workload-bound CPU execution proof for the existing geometry artifact.
     * @throws std::invalid_argument For missing CPU compute proof or a different parallel team.
     * @note GPU-only cohorts have no CPU router obligation. Serial workers=1 remains valid.
     */
    nlohmann::json cpuExecutionEvidence(std::span<const PerfStatRecord> records,
                                      const CPUExecutionObservation &observed) const
    {
        requireCPUExecution(observed);
        const int workers = requestedCPUWorkers();
        const auto config = deserializeOrchestrationConfig(value_.at("expected_config").dump());
        bool cpu_compute = MPIBootstrapPhase::classifyDeviceIntent(config) == BootstrapDeviceIntent::CpuOnly || config.cpu_layers > 0;
        if (config.topology_tree)
            for (const auto *leaf : config.topology_tree->root.leafDevices())
                cpu_compute = cpu_compute || leaf->device.isCPU();
        if (config.moe_routed_expert_plan)
        {
            for (const auto &domain : config.moe_routed_expert_plan->domains)
                for (const auto &participant : domain.participants)
                    cpu_compute = cpu_compute || participant.isCPU();
            for (const auto &domain : config.moe_routed_expert_plan->dense_domains)
                for (const auto &participant : domain.participants)
                    cpu_compute = cpu_compute || participant.isCPU();
        }
        bool parallel = false;
        for (const auto &record : records)
        {
            if (record.domain != "kernel" || record.name != "cpu_moe_router_q8_worker_rows") continue;
            const auto entry = record.tags.find("workers");
            int actual = 0;
            if (entry != record.tags.end())
            {
                const auto &text = entry->second;
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), actual);
                accuracy::requireQFN(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
                                     "Frozen replay has malformed CPU worker evidence");
            }
            accuracy::requireQFN(record.kind == PerfStatRecord::Kind::Counter && record.phase == "moe" && record.device == "cpu" &&
                                 record.count > 0 && std::isfinite(record.value) && record.value > 0 &&
                                 (actual == 1 || actual == workers),
                                 "Frozen replay executed a different actual CPU worker team");
            parallel = parallel || actual == workers;
        }
        accuracy::requireQFN(!cpu_compute || parallel,
                             "Frozen replay lacks actual declared CPU compute worker evidence");
        return {{"requested_workers", workers}, {"max_workers", observed.max_workers},
                {"allowed_cpus", observed.allowed_cpus}, {"physical_cores", observed.physical_cores},
                {"dynamic", observed.dynamic}, {"unbound", observed.unbound}, {"places", observed.places},
                {"cpu_compute_required", cpu_compute}, {"parallel_work_observed", parallel}};
    }

    /** @brief Immutable serving capture geometry derived from the public configuration. */
    struct ServingPrefillCapture
    {
        std::vector<int> bucket_rows; ///< Original canonical serving ladder, never a copied table.
        int minimum_rows = 0; ///< Canonical default floor clipped to the explicit public cap.
        int selected_rows = 0; ///< Single padded transaction width, or zero for exact chunks.
    };

    /**
     * @brief Resolve the original serving capture family for this geometry diagnostic.
     * @return Public ladder and floor before admission. Exact pipelines use the
     *         admitted ladder to validate their actual transactions after execution.
     * @throws std::invalid_argument If the public cap cannot represent the capture policy.
     */
    ServingPrefillCapture servingPrefillCapture() const
    {
        const auto config = deserializeOrchestrationConfig(value_.at("expected_config").dump());
        const int maximum = config.prefill_max_bucket_size.value_or(kDefaultPrefillGraphMaxBucketSize);
        auto buckets = prefillGraphBucketSizes(maximum);
        const int floor = std::min(maximum, kDefaultPrefillGraphMinBucketSize);
        if (isObservedFixedNativePipeline(config))
            return {std::move(buckets), floor, 0};
        const auto selected = std::lower_bound(buckets.begin(), buckets.end(), static_cast<int>(promptIds().size()));
        accuracy::requireQFN(selected != buckets.end(), "Frozen geometry diagnostic exceeds its original serving bucket cap");
        const int physical = *selected;
        return {std::move(buckets), floor, physical};
    }

    /**
     * @brief Extend only the explicit observation cardinality, preserving numerical budgets.
     * @param original_rows Original fixture observation count, replaced by the frozen history count.
     * @param contract Existing exact final-logit contract; both phase limits are untouched.
     * @throws std::invalid_argument For absent, invalid or already-divergent canonical contracts.
     */
    void applyObservationDepth(int &original_rows, std::optional<FinalLogitMeanKLContract> &contract) const
    {
        accuracy::requireQFN(original_rows > 0 && contract && contract->valid() &&
                             contract->decode_steps == original_rows,
                             "Frozen history needs the unchanged canonical final-logit limits and observation contract");
        original_rows = decodeRows();
        contract->decode_steps = original_rows;
    }

    /**
     * @brief Validate actual replay geometry after the original compared prefill has retired.
     * @param records Passive original forward_graph collector records, including setup.
     * @param native_required Actual fixture topology classification, false only for CPU-only.
     * @param pipeline Actual selected owner's admitted plan for an exact fixed pipeline.
     * @return Small owner-authored geometry handoff.
     * @throws std::invalid_argument For missing native replay, wrong bucket or malformed records.
     * @note Setup materialization cannot satisfy this proof. CPU retains its live-row graph.
     */
    nlohmann::json servingPrefillEvidence(std::span<const PerfStatRecord> records, bool native_required,
                                         const RankExecutionPlan *pipeline = nullptr) const
    {
        const auto capture = servingPrefillCapture();
        const auto logical = static_cast<int>(promptIds().size());
        if (capture.selected_rows == 0)
        {
            accuracy::requireQFN(native_required && pipeline,
                                 "Frozen exact pipeline requires its actual admitted native prefill plan");
            return exactPipelinePrefillEvidence(records, *pipeline, logical);
        }
        bool observed = false;
        if (native_required)
        {
            for (const auto &record : records)
            {
                if (record.domain != "forward_graph" || record.name != "prefill_graph_phase" ||
                    !(record.device.starts_with("CUDA:") || record.device.starts_with("ROCm:"))) continue;
                const auto phase = record.tags.find("capture_phase");
                if (phase == record.tags.end() || phase->second != "replay") continue;
                const auto real = record.tags.find("real_token_count");
                accuracy::requireQFN(real != record.tags.end(), "Frozen prefill replay omitted live row authority");
                if (positivePrefillRows(real->second) != logical) continue;
                const auto bucket = record.tags.find("bucket_seq_len");
                accuracy::requireQFN(record.kind == PerfStatRecord::Kind::Counter && record.phase == "prefill" &&
                                     record.count > 0 && std::isfinite(record.value) && record.value > 0 &&
                                     bucket != record.tags.end() && positivePrefillRows(bucket->second) == capture.selected_rows,
                                     "Frozen geometry diagnostic did not replay the original production-serving physical bucket");
                observed = true;
            }
            accuracy::requireQFN(observed, "Frozen geometry diagnostic has no actual native prefill replay evidence");
        }
        return {{"schema", "llaminar.model-kld-history-prefill-geometry.v1"},
                {"descriptor_sha256", digest_}, {"canonical_cell", value_.at("canonical_cell")},
                {"history_id", value_.at("history_id")}, {"capture_mode", "production_serving_ladder"},
                {"bucket_rows", capture.bucket_rows}, {"minimum_rows", capture.minimum_rows},
                {"logical_rows", logical}, {"physical_rows", native_required ? capture.selected_rows : logical},
                {"native_replay_required", native_required}, {"native_replay_observed", observed}};
    }

    /**
     * @brief Require exactly the frozen full reference metadata and original tokens.
     * @param metadata Existing canonical metadata.txt path.
     * @param prompt_ids Original parsed token_ids metadata.
     * @param teacher_ids Original parsed decode_tokens metadata.
     * @throws std::invalid_argument For stale pack, mismatched prompt or Pending conditions.
     */
    void requireReference(const std::filesystem::path &metadata,
                          const std::vector<int> &prompt_ids,
                          const std::vector<int> &teacher_ids) const
    {
        using accuracy::requireQFN;
        requireQFN(metadata.parent_path() == std::filesystem::path(referenceDirectory()) &&
                   std::filesystem::file_size(metadata) <= 1024 * 1024 &&
                   sha256FileHex(metadata) == std::optional<std::string>(value_.at("full_stage_metadata_digest").get<std::string>()),
                   "Frozen full-stage reference metadata/path identity changed");
        const auto expected = value_.at("continuation_ids").get<std::vector<int>>();
        requireQFN(prompt_ids == promptIds() && teacher_ids.size() >= expected.size() &&
                   std::equal(expected.begin(), expected.end(), teacher_ids.begin()),
                   "Frozen reference prompt or D+1 teacher IDs differ from the original history");
    }

    /** @return Whether this explicitly incomplete observation is forbidden as a campaign certificate. */
    bool hasObservationShard() const noexcept { return value_.contains("observation_shard"); }

    /** @return Immutable exact semantic selection; empty preserves the ordinary full-inventory mode. */
    std::vector<std::string> observationKeys() const
    { return hasObservationShard()?value_.at("observation_shard").at("selected_keys").get<std::vector<std::string>>():std::vector<std::string>{}; }

    /**
     * @brief Check a reference key against the same capture/BOM selection, retaining original decode coordinates.
     * @param key Original semantic key, optionally prefixed by its decode step.
     * @return Whether this diagnostic owns the value; ordinary histories select all values.
     * @throws std::invalid_argument For a malformed explicit decode key.
     */
    bool observesReferenceKey(std::string key) const
    {
        if(!hasObservationShard())return true;
        if(key.starts_with("decode_step")) {
            const auto separator=key.find('_',11);
            accuracy::requireQFN(separator!=std::string::npos,"Invalid observation decode key");
            key=key.substr(separator+1);
        }
        const auto selected=observationKeys();
        return std::find(selected.begin(),selected.end(),key)!=selected.end();
    }

    /**
     * @brief Authenticate complete semantic gold inventory and every selected original phase before setup.
     * @throws std::invalid_argument For stale/partial inventory or missing selected gold, never synthesizing payloads.
     */
    void requireObservationInventory() const
    {
        if(!hasObservationShard())return;
        std::set<std::string> actual;
        for(const auto& entry:std::filesystem::directory_iterator(referenceDirectory())) {
            if(entry.path().extension()!=".npy")continue;
            std::string key=entry.path().stem().string();
            if(key.starts_with("decode_step")) {
                const auto separator=key.find('_',11);
                accuracy::requireQFN(separator!=std::string::npos,"Invalid original gold phase key");
                key=key.substr(separator+1);
            }
            actual.insert(std::move(key));
        }
        const auto full=value_.at("observation_shard").at("full_keys").get<std::vector<std::string>>();
        accuracy::requireQFN(actual==std::set<std::string>(full.begin(),full.end()),"Observation shard full gold inventory is stale or incomplete");
        for(int step=-1;step<decodeRows();++step)requireSnapshotInventory(full,step);
    }

    /**
     * @brief Require the canonical case's complete stage inventory before admission.
     * @param keys Unmodified required snapshot keys from the actual typed case.
     * @param decode_step Negative for prefill, otherwise an original decode offset.
     * @throws std::invalid_argument For an empty, missing or non-regular checkpoint,
     *         naming its exact phase, original step, key, path and failure state.
     * @throws std::filesystem::filesystem_error For metadata I/O failures, retaining
     *         the original error code with the same checkpoint context.
     *
     * This checks metadata/extents only. Actual typed NPY loading, every raw
     * numerical comparison and six-CSV publication remain in the original gate.
     */
    void requireSnapshotInventory(const std::vector<std::string> &keys, int decode_step) const
    {
        using accuracy::requireQFN;
        requireQFN(!keys.empty() && decode_step >= -1 && decode_step < decodeRows(),
                   "Frozen full-stage replay requires the complete bounded canonical checkpoint inventory");
        const auto prefix = decode_step < 0 ? std::string{} : "decode_step" + std::to_string(decode_step) + "_";
        for (const auto &key : keys)
        {
            const auto path = std::filesystem::path(referenceDirectory()) / (prefix + key + ".npy");
            // Format context only on failure; ordinary admission still performs
            // the same metadata checks and never opens/hashes a tensor payload.
            const auto failure = [&](const char *state) {
                return std::string("Frozen full-stage reference inventory failed: phase=") +
                    (decode_step < 0 ? "prefill" : "decode") +
                    " step=" + std::to_string(decode_step) + " key=" + key +
                    " path=" + path.string() + " state=" + state;
            };
            try
            {
                if (!std::filesystem::is_regular_file(path))
                    throw std::invalid_argument(failure(std::filesystem::exists(path) ? "non_regular" : "missing"));
                if (std::filesystem::file_size(path) == 0)
                    throw std::invalid_argument(failure("empty"));
            }
            catch (const std::filesystem::filesystem_error &error)
            {
                throw std::filesystem::filesystem_error(failure("metadata_error"), path, error.code());
            }
        }
    }

    /**
     * @brief Publish the actual fixture's history binding, never numerical success.
     * @param path Original per-cell artifact directory's closed metadata output.
     * @param actual_prompt Prompt currently installed in the original fixture.
     * @param actual_ids Token IDs currently installed in the original fixture.
     * @param actual_reference Original fixture's selected full-stage directory.
     * @param actual_depth Explicit frozen decode observation count; phase budgets are unchanged.
     * @throws std::invalid_argument For a missing or incorrectly applied projection.
     * @throws std::runtime_error If the passive metadata output cannot close.
     */
    void publishBinding(const std::filesystem::path &path, const std::string &actual_prompt,
                        const std::vector<int> &actual_ids, const std::string &actual_reference,
                        int actual_depth) const
    {
        accuracy::requireQFN(actual_prompt == prompt() && actual_ids == promptIds() &&
                             actual_reference == referenceDirectory() && actual_depth == value_.at("decode_rows").get<int>(),
                             "Frozen history was not installed in the actual canonical fixture");
        const nlohmann::json binding = {{"schema", "llaminar.model-kld-history-binding.v1"},
            {"descriptor_sha256", digest_}, {"canonical_cell", value_.at("canonical_cell")},
            {"history_id", value_.at("history_id")}, {"source_digest", value_.at("source_digest")},
            {"full_stage_metadata_digest", value_.at("full_stage_metadata_digest")},
            {"prompt", actual_prompt}, {"prompt_ids", actual_ids},
            {"continuation_ids", value_.at("continuation_ids")}, {"decode_rows", actual_depth},
            {"reference_directory", actual_reference}};
        if(hasObservationShard()) {
            const auto& shard=value_.at("observation_shard");
            auto full=shard.at("full_keys").get<std::vector<std::string>>();
            auto selected=observationKeys();std::sort(full.begin(),full.end());std::sort(selected.begin(),selected.end());
            std::vector<std::string> missing;
            std::set_difference(full.begin(),full.end(),selected.begin(),selected.end(),std::back_inserter(missing));
            const nlohmann::json scope={{"schema",1},{"scope","noncertifying_observation_shard"},
                {"certification_eligible",false},{"descriptor_sha256",digest_},{"canonical_cell",value_.at("canonical_cell")},
                {"source_digest",value_.at("source_digest")},{"full_stage_metadata_digest",value_.at("full_stage_metadata_digest")},
                {"full_keys",full},{"selected_keys",selected},{"missing_coverage",missing},
                {"terminal_rows_required",decodeRows()+1},{"terminal_rows_authority","llaminar_snapshots/manifest.csv"}};
            std::ofstream file(path.parent_path()/"history_replay_observation_scope.json");file<<scope.dump(2)<<'\n';file.close();
            if(!file)throw std::runtime_error("Cannot publish observation shard coverage");
        }
        std::ofstream output(path, std::ios::trunc);
        output << binding.dump(2) << '\n';
        output.close();
        if (!output) throw std::runtime_error("Cannot close actual frozen-history fixture binding");
    }

    /** @return Explicit frozen diagnostic cardinality, independent of the original cell default. */
    int decodeRows() const { return value_.at("decode_rows").get<int>(); }

    /** @return Original full-stage directory; no inference or alternate cache path. */
    std::string referenceDirectory() const { return value_.at("reference_directory").get<std::string>(); }
    /** @return Exact original UTF8 prompt used by the reference generator. */
    std::string prompt() const { return value_.at("prompt").get<std::string>(); }
    /** @return Original tokenizer IDs; thresholds and topology are not projected. */
    std::vector<int> promptIds() const { return value_.at("prompt_ids").get<std::vector<int>>(); }
    /** @return Explicit handoff identity, including all source and teacher metadata. */
    const std::string &identity() const noexcept { return digest_; }

private:

    /** @brief Parse one required positive physical/live replay dimension. */
    static int positivePrefillRows(const std::string &text)
    {
        int rows = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), rows);
        accuracy::requireQFN(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && rows > 0,
                             "Frozen prefill replay has malformed physical or live row metadata");
        return rows;
    }

    /**
     * @brief Join the actual admitted exact schedule to every original native replay.
     * @param records Original request's prefill graph lifecycle records.
     * @param pipeline Same selected owner's admitted stage and retained-width contract.
     * @param logical Complete frozen prompt length, before any continuation.
     * @return Exact chunk offsets/widths and the verified native participant roster.
     * @throws std::invalid_argument If any required participant/chunk is missing, duplicated or padded.
     */
    nlohmann::json exactPipelinePrefillEvidence(std::span<const PerfStatRecord> records,
                                               const RankExecutionPlan &pipeline, int logical) const
    {
        using accuracy::requireQFN;
        const auto &admitted = pipeline.runtime.overlay_prefill_schedule;
        requireQFN(pipeline.usesLocalPP() && pipeline.local_pp_stage_tp_info.size() == pipeline.local_pp_devices.size() &&
                   admitted.enabled() && admitted.bucket_rows == exactPrefillGraphWidths(admitted.graph_row_capacity),
                   "Frozen exact pipeline lost its admitted retained prefill family");
        const auto schedule = planPrefillChunkSchedule(PrefillChunkSchedulerPolicy::forRetainedBucketFamily(
            admitted.bucket_rows, 0, logical, PrefillChunkWidthPolicy::ExactRetained));
        requireQFN(schedule && !schedule.chunks.empty(), "Frozen exact pipeline could not schedule the complete prompt");

        std::map<std::string, std::set<int>> replays;
        for (std::size_t stage = 0; stage < pipeline.local_pp_devices.size(); ++stage)
        {
            const auto &members = pipeline.local_pp_stage_tp_info[stage].devices;
            requireQFN(!members.empty() && members.front() == pipeline.local_pp_devices[stage],
                       "Frozen exact pipeline changed an admitted stage's original participants");
            for (const auto &address : members)
            {
                const auto device = address.toLocalDeviceId();
                requireQFN(device.is_gpu() && replays.try_emplace(device.toString()).second,
                           "Frozen exact pipeline has a nonnative or duplicate admitted participant");
            }
        }

        // PerfStats groups repeated shapes under stable dimensions. Reuse the
        // ordered witness for each width. A new placement key may delimit a
        // later span of that width, but a chunk belongs to exactly one record.
        constexpr std::size_t words_per_chunk = 3;
        std::map<int, std::vector<std::uint64_t>> by_width;
        for (const auto &chunk : schedule.chunks)
        {
            auto &words = by_width[chunk.bucket_seq_len];
            words.insert(words.end(), {static_cast<std::uint64_t>(chunk.chunk_index),
                static_cast<std::uint64_t>(chunk.token_offset),
                static_cast<std::uint64_t>(chunk.token_offset + chunk.real_count)});
        }
        for (const auto &record : records)
        {
            if (record.domain != "forward_graph" || record.name != "prefill_graph_phase") continue;
            const auto capture = record.tags.find("capture_phase");
            if (capture == record.tags.end() || capture->second != "replay") continue;
            const auto device = replays.find(record.device);
            const auto live = record.tags.find("real_token_count");
            const auto width = record.tags.find("bucket_seq_len");
            requireQFN(device != replays.end() && record.phase == "prefill" && record.count > 0 &&
                       record.value == static_cast<double>(record.count) && live != record.tags.end() &&
                       width != record.tags.end() && record.sequence_minimum_words.size() == words_per_chunk,
                       "Frozen exact pipeline replay did not identify its original native chunk span");
            const int rows = positivePrefillRows(width->second);
            const auto expected = by_width.find(rows);
            requireQFN(positivePrefillRows(live->second) == rows && expected != by_width.end(),
                       "Frozen exact pipeline replay changed its original live chunk width");

            const auto &words = expected->second;
            const auto first_index = record.sequence_minimum_words.front();
            std::size_t offset = 0;
            while (offset < words.size() && words[offset] < first_index)
                offset += words_per_chunk;
            requireQFN(offset < words.size() && words[offset] == first_index &&
                       record.count <= (words.size() - offset) / words_per_chunk,
                       "Frozen exact pipeline replay changed its original native chunk range");
            const auto span = std::span<const std::uint64_t>(words).subspan(
                offset, static_cast<std::size_t>(record.count) * words_per_chunk);
            requireQFN(PerfStatsCollector::matchesCounterSequence(record, span, words_per_chunk),
                       "Frozen exact pipeline replay changed its original live chunk order or offsets");
            for (std::size_t step = 0; step < span.size(); step += words_per_chunk)
                requireQFN(device->second.insert(static_cast<int>(span[step])).second,
                           "Frozen exact pipeline replay repeated an original native chunk");
        }

        auto devices = nlohmann::json::array();
        for (const auto &[device, chunks] : replays)
        {
            requireQFN(chunks.size() == schedule.chunks.size(),
                       "Frozen exact pipeline is missing an original participant's chunk replay");
            devices.push_back(device);
        }
        auto chunks = nlohmann::json::array();
        for (const auto &chunk : schedule.chunks)
            chunks.push_back({{"token_offset", chunk.token_offset}, {"logical_rows", chunk.real_count},
                              {"physical_rows", chunk.bucket_seq_len}});
        return {{"schema", "llaminar.model-kld-history-prefill-geometry.v1"},
                {"descriptor_sha256", digest_}, {"canonical_cell", value_.at("canonical_cell")},
                {"history_id", value_.at("history_id")}, {"capture_mode", "production_exact_retained"},
                {"bucket_rows", admitted.bucket_rows}, {"minimum_rows", 1},
                {"logical_rows", logical}, {"physical_rows", logical},
                {"chunks", std::move(chunks)}, {"native_devices", std::move(devices)},
                {"native_replay_required", true}, {"native_replay_observed", true}};
    }
    /** @brief Admit only a complete parsed metadata request, never runtime ownership. */
    FrozenParityHistoryReplay(nlohmann::json value, std::string digest)
        : value_(std::move(value)), digest_(std::move(digest)) {}
    /** @brief Validate the existing lowercase SHA256 representation. */
    static bool isDigest(const std::string &value)
    {
        return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
    }
    nlohmann::json value_;
    std::string digest_;
};
} // namespace llaminar2::test::parity
