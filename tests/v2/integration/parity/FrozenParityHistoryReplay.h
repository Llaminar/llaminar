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
 * required checkpoint keys remain the canonical inventory authority.
 */
#pragma once
#include "tools/ModelKLDQFNContract.h"
#include "app/MPIBootstrapPhase.h"
#include <omp.h>
#include <sched.h>
#include <set>
#include "utils/DebugEnv.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "utils/PerfStatsCollector.h"
#include "FinalLogitMeanKLContract.h"
#include <charconv>
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
     * @brief Read the existing public worker request, rejecting implicit or multi-rank work.
     * @return Explicit positive worker count from the original lossless configuration.
     * @throws std::invalid_argument Without one representable single-rank request.
     */
    int requestedCPUWorkers() const
    {
        const auto config = deserializeOrchestrationConfig(value_.at("expected_config").dump());
        accuracy::requireQFN(config.n_threads > 0 && config.mpi_procs == 1,
                             "Frozen replay requires explicit positive single-rank CPU workers");
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
        int selected_rows = 0; ///< First retained bucket containing this history's real rows.
    };

    /**
     * @brief Resolve the original serving capture family for this geometry diagnostic.
     * @return Canonical typed bucket ladder and its selected physical transaction.
     * @throws std::invalid_argument If this history cannot fit one retained bucket.
     * @note This changes capture shape only; prompt, teachers, math and placement stay bound.
     */
    ServingPrefillCapture servingPrefillCapture() const
    {
        const auto config = deserializeOrchestrationConfig(value_.at("expected_config").dump());
        const int maximum = config.prefill_max_bucket_size.value_or(kDefaultPrefillGraphMaxBucketSize);
        auto buckets = prefillGraphBucketSizes(maximum);
        const auto selected = std::lower_bound(buckets.begin(), buckets.end(), static_cast<int>(promptIds().size()));
        accuracy::requireQFN(selected != buckets.end(), "Frozen geometry diagnostic exceeds its original serving bucket cap");
        const int physical = *selected;
        return {std::move(buckets), std::min(maximum, kDefaultPrefillGraphMinBucketSize), physical};
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
     * @return Small owner-authored geometry handoff; no numerical verdict or runtime certificate.
     * @throws std::invalid_argument For missing native replay, wrong bucket or malformed records.
     * @note Setup-only materialization cannot satisfy this replay proof. Counts are submissions,
     *       not a timing or per-token performance metric. CPU retains its original live-row graph.
     */
    nlohmann::json servingPrefillEvidence(std::span<const PerfStatRecord> records, bool native_required) const
    {
        const auto capture = servingPrefillCapture();
        const auto logical = promptIds().size();
        bool observed = false;
        if (native_required)
        {
            const auto integer = [](const std::string &text) {
                int value = 0;
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
                accuracy::requireQFN(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && value > 0,
                                     "Frozen prefill replay has malformed physical or live row metadata");
                return value;
            };
            for (const auto &record : records)
            {
                if (record.domain != "forward_graph" || record.name != "prefill_graph_phase" ||
                    !(record.device.starts_with("CUDA:") || record.device.starts_with("ROCm:"))) continue;
                const auto phase = record.tags.find("capture_phase");
                if (phase == record.tags.end() || phase->second != "replay") continue;
                const auto real = record.tags.find("real_token_count");
                accuracy::requireQFN(real != record.tags.end(), "Frozen prefill replay omitted live row authority");
                if (integer(real->second) != static_cast<int>(logical)) continue;
                const auto bucket = record.tags.find("bucket_seq_len");
                accuracy::requireQFN(record.kind == PerfStatRecord::Kind::Counter && record.phase == "prefill" &&
                                     record.count > 0 && std::isfinite(record.value) && record.value > 0 &&
                                     bucket != record.tags.end() && integer(bucket->second) == capture.selected_rows,
                                     "Frozen geometry diagnostic did not replay the original production-serving physical bucket");
                observed = true;
            }
            accuracy::requireQFN(observed, "Frozen geometry diagnostic has no actual native prefill replay evidence");
        }
        return {{"schema", "llaminar.model-kld-history-prefill-geometry.v1"},
                {"descriptor_sha256", digest_}, {"canonical_cell", value_.at("canonical_cell")},
                {"history_id", value_.at("history_id")}, {"capture_mode", "production_serving_ladder"},
                {"bucket_rows", capture.bucket_rows}, {"minimum_rows", capture.minimum_rows},
                {"logical_rows", logical}, {"physical_rows", native_required ? capture.selected_rows : static_cast<int>(logical)},
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

    /**
     * @brief Require the canonical case's complete stage inventory before admission.
     * @param keys Unmodified required snapshot keys from the actual typed case.
     * @param decode_step Negative for prefill, otherwise an original decode offset.
     * @throws std::invalid_argument For an empty or missing full-stage checkpoint set.
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
            requireQFN(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) > 0,
                       "Frozen full-stage reference is incomplete; regenerate exact history before admission");
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
