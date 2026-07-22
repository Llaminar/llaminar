/**
 * @file NativeVNNIPairedRequestManifest.h
 * @brief Strict JSON contract for backend-qualified NativeVNNI paired timings.
 *
 * The Python development-CV planner emits nominal policy decisions only after
 * resolving shape-dependent KPAR formulas to concrete exact-KB candidates.
 * This header validates that transaction before the CUDA trainer allocates a
 * tensor or launches a kernel. A malformed, partial, duplicated, or stale
 * request therefore fails at startup instead of silently measuring a different
 * candidate pair. Version two adds the architecture class required to keep CPU
 * AVX2-build, forced-AVX2, and AVX512 runtime evidence in separate domains.
 *
 * This is performance-tooling infrastructure. Production inference never reads
 * a request manifest and never depends on nlohmann/json at dispatch time.
 */

#pragma once

#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

namespace llaminar2::test::native_vnni_dispatch
{
    /** Retained CUDA-only schema used by already-published request manifests. */
    inline constexpr const char *kNativeVNNIPairedRequestSchemaV1 =
        "native-vnni-cuda-paired-request-v1";

    /** Current backend-qualified request schema. */
    inline constexpr const char *kNativeVNNIPairedRequestSchema =
        "native-vnni-paired-request-v2";

    /** Timing-free reason emitted only by a frozen M=1 certification plan. */
    inline constexpr const char *kNativeVNNIM1SealedChallengerReason =
        "sealed_frozen_leaf_exhaustive_challenger";

    /** Timing-free reason emitted only by a frozen grouped certification plan. */
    inline constexpr const char *kNativeVNNIGroupedSealedChallengerReason =
        "grouped_frozen_leaf_exhaustive_challenger";

    /**
     * Timing-free route/byte witness for a grouped domain with one schedule.
     *
     * Both interleaved roles name the same forceable grouped candidate. The
     * final certifier treats the cell as zero dispatch regret while retaining
     * the fresh-geometry production-route and serial-M1 byte proof.
     */
    inline constexpr const char *kNativeVNNIGroupedSingleCandidateReason =
        "grouped_frozen_leaf_single_candidate_witness";

    /**
     * @brief One concrete selected/reference edge to measure interleaved.
     *
     * Candidate IDs in this record are forceable production launcher IDs. In
     * particular, a `kpar_formula` policy ID is forbidden here: the Python
     * planner must resolve it to the exact `kpar.*.kbN` schedule backed by the
     * broad observation before publishing the request.
     */
    struct NativeVNNIPairedTimingRequest
    {
        std::string request_id;
        std::string source_format;
        int source_codebook = -1;
        int execution_codebook = -1;
        std::string architecture_class;
        std::string shape;
        std::string shape_group_id;
        std::string execution_mode;
        int m = 0;
        int n = 0;
        int k = 0;
        std::string selected_candidate_id;
        std::string exact_candidate_id;
        double observed_cv_regret = 0.0;
        std::string reason;
    };

    /**
     * @brief Validated root metadata and ordered request inventory.
     */
    struct NativeVNNIPairedRequestManifest
    {
        std::string schema_version;
        std::string backend;
        std::string development_corpus_digest;
        std::string paired_evidence_digest;
        double max_regret = 0.0;
        std::vector<NativeVNNIPairedTimingRequest> requests;
    };

    namespace detail
    {
        /** Return the exact key set of one JSON object. */
        inline std::set<std::string> jsonKeys(const nlohmann::json &value)
        {
            std::set<std::string> result;
            for (auto iterator = value.begin(); iterator != value.end(); ++iterator)
                result.insert(iterator.key());
            return result;
        }

        /**
         * @brief Require an exact object schema without ignored extra fields.
         *
         * Ignoring an unknown field is unsafe for a measurement authority: a
         * newer planner could attach a discriminator that this trainer does not
         * understand and the resulting CSV would be mislabeled. Exact key
         * equality turns every schema evolution into an explicit version bump.
         */
        inline void requireKeys(
            const nlohmann::json &value,
            const std::set<std::string> &expected,
            const std::string &label)
        {
            if (!value.is_object() || jsonKeys(value) != expected)
            {
                throw std::runtime_error(
                    label + " has fields incompatible with paired request schema");
            }
        }

        /** Return whether a string carries a nonempty SHA-256 identity. */
        inline bool isSha256Digest(const std::string &value)
        {
            return value.size() == 71 && value.rfind("sha256:", 0) == 0;
        }
    } // namespace detail

    /**
     * @brief Load and fully validate one paired request JSON transaction.
     *
     * @param path Filesystem path emitted by `paired_requests.py`.
     * @return Root metadata and manifest-order concrete requests.
     * @throws std::runtime_error for I/O, JSON, schema, identity, dimension,
     *         duplicate, or unsupported execution-mode failures.
     */
    inline NativeVNNIPairedRequestManifest loadNativeVnniPairedRequestManifest(
        const std::string &path)
    {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("Unable to open paired request manifest: " + path);

        nlohmann::json root;
        try
        {
            input >> root;
        }
        catch (const nlohmann::json::exception &error)
        {
            throw std::runtime_error(
                "Unable to parse paired request manifest " + path + ": " +
                error.what());
        }

        detail::requireKeys(
            root,
            {
                "schema_version",
                "backend",
                "development_corpus_digest",
                "paired_evidence_digest",
                "max_regret",
                "request_count",
                "requests",
            },
            "paired request root");
        const std::string schema_version =
            root.at("schema_version").get<std::string>();
        const bool legacy_cuda =
            schema_version == kNativeVNNIPairedRequestSchemaV1;
        if (!legacy_cuda && schema_version != kNativeVNNIPairedRequestSchema)
        {
            throw std::runtime_error("Unsupported paired request schema in " + path);
        }
        const std::string backend = root.at("backend").get<std::string>();
        if ((legacy_cuda && backend != "cuda") ||
            (!legacy_cuda && backend != "cpu" && backend != "cuda" &&
             backend != "rocm"))
        {
            throw std::runtime_error("Paired request backend is unsupported");
        }

        NativeVNNIPairedRequestManifest manifest;
        manifest.schema_version = schema_version;
        manifest.backend = backend;
        manifest.development_corpus_digest =
            root.at("development_corpus_digest").get<std::string>();
        manifest.paired_evidence_digest =
            root.at("paired_evidence_digest").get<std::string>();
        manifest.max_regret = root.at("max_regret").get<double>();
        if (!detail::isSha256Digest(manifest.development_corpus_digest) ||
            !detail::isSha256Digest(manifest.paired_evidence_digest))
        {
            throw std::runtime_error("Paired request provenance digest is invalid");
        }
        if (!std::isfinite(manifest.max_regret) || manifest.max_regret <= 0.0 ||
            manifest.max_regret >= 1.0)
        {
            throw std::runtime_error("Paired request max_regret is invalid");
        }

        const nlohmann::json &requests = root.at("requests");
        if (!requests.is_array())
            throw std::runtime_error("Paired request inventory must be an array");
        const int declared_count = root.at("request_count").get<int>();
        if (declared_count < 0 ||
            static_cast<size_t>(declared_count) != requests.size())
        {
            throw std::runtime_error("Paired request_count does not match inventory");
        }

        std::set<std::string> request_ids;
        std::set<std::tuple<
            std::string,
            std::string,
            std::string,
            std::string,
            int,
            int,
            int,
            std::string,
            std::string>> edge_identities;
        manifest.requests.reserve(requests.size());
        for (size_t index = 0; index < requests.size(); ++index)
        {
            const nlohmann::json &record = requests.at(index);
            std::set<std::string> request_fields = {
                    "request_id",
                    "source_format",
                    "source_codebook",
                    "execution_codebook",
                    "shape",
                    "shape_group_id",
                    "execution_mode",
                    "m",
                    "n",
                    "k",
                    "selected_candidate_id",
                    "exact_candidate_id",
                    "observed_cv_regret",
                    "reason",
                };
            if (!legacy_cuda)
                request_fields.insert("architecture_class");
            detail::requireKeys(
                record,
                request_fields,
                "paired request record " + std::to_string(index));

            NativeVNNIPairedTimingRequest request;
            try
            {
                request.request_id = record.at("request_id").get<std::string>();
                request.source_format =
                    record.at("source_format").get<std::string>();
                request.source_codebook = record.at("source_codebook").get<int>();
                request.execution_codebook =
                    record.at("execution_codebook").get<int>();
                request.architecture_class = legacy_cuda
                    ? std::string{}
                    : record.at("architecture_class").get<std::string>();
                request.shape = record.at("shape").get<std::string>();
                request.shape_group_id =
                    record.at("shape_group_id").get<std::string>();
                request.execution_mode =
                    record.at("execution_mode").get<std::string>();
                request.m = record.at("m").get<int>();
                request.n = record.at("n").get<int>();
                request.k = record.at("k").get<int>();
                request.selected_candidate_id =
                    record.at("selected_candidate_id").get<std::string>();
                request.exact_candidate_id =
                    record.at("exact_candidate_id").get<std::string>();
                request.observed_cv_regret =
                    record.at("observed_cv_regret").get<double>();
                request.reason = record.at("reason").get<std::string>();
            }
            catch (const nlohmann::json::exception &error)
            {
                throw std::runtime_error(
                    "Invalid paired request record " + std::to_string(index) +
                    ": " + error.what());
            }

            const bool text_ok = !request.request_id.empty() &&
                                 !request.source_format.empty() &&
                                 (legacy_cuda ||
                                  !request.architecture_class.empty()) &&
                                 !request.shape.empty() &&
                                 !request.shape_group_id.empty() &&
                                 !request.selected_candidate_id.empty() &&
                                 !request.exact_candidate_id.empty() &&
                                 !request.reason.empty();
            const bool dimensions_ok = request.m > 0 && request.n > 0 &&
                                       request.k > 0 && request.k % 32 == 0;
            const bool mode_ok = request.execution_mode == "eager" ||
                                 request.execution_mode == "graph_captured";
            const bool codebooks_ok = request.source_codebook >= 0 &&
                                      request.execution_codebook >= 0;
            const bool m1_sealed =
                request.reason == kNativeVNNIM1SealedChallengerReason;
            const bool grouped_sealed =
                request.reason == kNativeVNNIGroupedSealedChallengerReason;
            const bool grouped_single_candidate =
                request.reason == kNativeVNNIGroupedSingleCandidateReason;
            const bool sealed_request =
                m1_sealed || grouped_sealed || grouped_single_candidate;
            const bool sealed_m_ok = !sealed_request ||
                                     (m1_sealed && request.m == 1) ||
                                     ((grouped_sealed ||
                                       grouped_single_candidate) &&
                                      request.m > 1);

            /**
             * Development refinement requests are actionable only when their
             * held-out regret reaches the promotion budget. Frozen plans have
             * a different contract: every challenger edge is measured without
             * consulting timing, so zero is the sole permitted placeholder.
             * The final Python certifier cryptographically binds those rows to
             * the frozen plan before they can influence installation.
             */
            const bool regret_ok = std::isfinite(request.observed_cv_regret) &&
                (sealed_request
                     ? request.observed_cv_regret == 0.0
                     : request.observed_cv_regret >= manifest.max_regret);
            const bool candidates_ok =
                ((grouped_single_candidate &&
                  request.selected_candidate_id == request.exact_candidate_id) ||
                 (!grouped_single_candidate &&
                  request.selected_candidate_id != request.exact_candidate_id)) &&
                request.selected_candidate_id.find("kpar_formula") ==
                    std::string::npos &&
                request.exact_candidate_id.find("kpar_formula") ==
                    std::string::npos;
            if (!text_ok || !dimensions_ok || !mode_ok || !codebooks_ok ||
                !sealed_m_ok || !regret_ok || !candidates_ok)
            {
                throw std::runtime_error(
                    "Paired request record is not forceable: " +
                    request.request_id);
            }
            if (!request_ids.insert(request.request_id).second)
                throw std::runtime_error("Duplicate paired request ID");
            const auto edge = std::make_tuple(
                request.architecture_class,
                request.source_format,
                request.shape,
                request.execution_mode,
                request.m,
                request.n,
                request.k,
                request.selected_candidate_id,
                request.exact_candidate_id);
            if (!edge_identities.insert(edge).second)
                throw std::runtime_error("Duplicate paired request edge");
            manifest.requests.push_back(std::move(request));
        }
        return manifest;
    }
} // namespace llaminar2::test::native_vnni_dispatch
