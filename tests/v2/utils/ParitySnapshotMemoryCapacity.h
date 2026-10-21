/**
 * @file ParitySnapshotMemoryCapacity.h
 * @brief Metadata-only memory envelope for production parity graph snapshots.
 *
 * The authenticated reference directory is also the authoritative checkpoint
 * inventory selected into production graphs. This helper reads only each NPY
 * header, never its numerical payload, and converts that inventory into a
 * conservative per-accelerator capacity. Prefill payloads are widened to the
 * retained graph-row capacity and completed-collective aliases are charged as
 * distinct live values. Explicit model-owned state selections replace live NPY
 * prefixes with native state capacity in the existing planner. Results are cached because a process campaign reuses
 * the same immutable reference pack across many generated cells.
 */

#pragma once

#include "planning/GraphSnapshotMemoryCapacity.h"
#include "TypedNpySnapshot.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"

#include <cnpy.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test::parity
{
    /** @brief Explicit reference payload projection; observation count is not graph-owner count. */
    enum class ParitySnapshotPayloadProjection { ObservationEnvelope, RetainedMainServingPhases };

    /**
     * @brief Resolve the original serving-family row envelope before snapshot admission.
     * @param reference_rows Live rows represented by the immutable reference pack.
     * @param context_rows Authored maximum logical context; required for captured prefill.
     * @param configured_buckets Original startup-policy prefill inventory.
     * @param captures_prefill Whether this participant retains GPU prefill bucket graphs.
     * @param mtp Original retained MTP capacity, independent of active request depth.
     * @return Maximum reference, permitted serving-prefill, and retained verifier rows.
     * @throws std::invalid_argument For nonpositive reference rows or an unknown
     *         context bound when captured prefill must be priced.
     *
     * Admission may fit a smaller candidate, but cannot materialize a bucket outside
     * this original permitted envelope. Reuse the production candidate resolver:
     * prompt length is a communication extent, not the wide arena capacity. This
     * authors only the existing snapshot BOM; PMA alone owns physical admission.
     * CPU/no-bucket callers retain their existing reference/MTP envelope.
     */
    inline int paritySnapshotServingGraphRows(
        int reference_rows, int context_rows,
        const std::vector<int> &configured_buckets, bool captures_prefill,
        const MTPRuntimeConfig &mtp)
    {
        if (reference_rows <= 0 || (captures_prefill && context_rows <= 0))
            throw std::invalid_argument(
                "Parity snapshot serving geometry requires positive reference rows "
                "and a positive context bound for captured prefill");
        int rows = reference_rows;
        if (captures_prefill)
        {
            const auto candidates = residentPrefillGraphRowCandidates(
                configured_buckets, context_rows);
            rows = std::max(rows, candidates.front());
        }
        return resolveRetainedGraphRowCapacity(rows, mtp);
    }

    /** @brief Auditable evidence behind one graph-snapshot capacity declaration. */
    struct ParitySnapshotMemoryCapacityEvidence
    {
        GraphSnapshotMemoryCapacity capacity;
        std::size_t reference_file_count = 0u;
        std::size_t unscaled_payload_bytes = 0u;
        std::size_t collective_alias_bytes = 0u;
        /** Live state-file bytes excluded before native context-capacity replacement. */
        std::size_t replaced_state_payload_bytes = 0u;
        int reference_rows = 0;
        int retained_graph_rows = 0;

        /** @return Whether the evidence can safely enter memory admission. */
        [[nodiscard]] bool valid() const noexcept
        {
            return capacity.valid() && reference_file_count != 0u &&
                   reference_rows > 0 && retained_graph_rows > 0;
        }
    };

    namespace detail
    {
        /** @brief Checked addition used while building one physical byte bound. */
        inline std::size_t paritySnapshotCheckedAdd(
            std::size_t left,
            std::size_t right,
            std::string_view what)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    "Parity snapshot " + std::string(what) +
                    " overflows size_t");
            }
            return left + right;
        }

        /** @brief Checked multiplication for NPY element and shape geometry. */
        inline std::size_t paritySnapshotCheckedMultiply(
            std::size_t left,
            std::size_t right,
            std::string_view what)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    "Parity snapshot " + std::string(what) +
                    " overflows size_t");
            }
            return left * right;
        }

        /**
         * @brief Widen a prefill artifact without reading or interpreting data.
         *
         * Some checkpoints encode rows in dimension zero, some behind a batch
         * dimension, and grouped recurrent values may flatten rows with another
         * axis. Scaling the complete payload is deliberately conservative and
         * independent of those model-specific layouts. Never scale downward:
         * fixed state embedded beside row data must remain fully covered.
         */
        inline std::size_t widenParityPrefillPayload(
            std::size_t payload_bytes,
            int reference_rows,
            int retained_graph_rows)
        {
            if (retained_graph_rows <= reference_rows)
                return payload_bytes;
            const std::size_t rows =
                static_cast<std::size_t>(retained_graph_rows);
            const std::size_t divisor =
                static_cast<std::size_t>(reference_rows);
            const std::size_t quotient = payload_bytes / divisor;
            const std::size_t remainder = payload_bytes % divisor;
            std::size_t widened = paritySnapshotCheckedMultiply(
                quotient, rows, "row-widened payload");
            if (remainder != 0u)
            {
                const std::size_t remainder_product =
                    paritySnapshotCheckedMultiply(
                        remainder, rows, "row-widened remainder");
                const std::size_t rounded_remainder_product =
                    paritySnapshotCheckedAdd(
                        remainder_product,
                        divisor - 1u,
                        "row-widened remainder rounding");
                widened = paritySnapshotCheckedAdd(
                    widened,
                    rounded_remainder_product / divisor,
                    "row-widened payload total");
            }
            return std::max(payload_bytes, widened);
        }

        /** @brief RAII closer that avoids function-attribute loss in a deleter type. */
        struct ParitySnapshotFileCloser final
        {
            /** @brief Close one successfully opened metadata stream. */
            void operator()(std::FILE *file) const noexcept
            {
                if (file != nullptr)
                    (void)std::fclose(file);
            }
        };

        /** @return Whether one reference stem receives a completed-collective alias. */
        inline bool paritySnapshotHasCollectiveAlias(
            std::string_view stem,
            const std::vector<std::string> &collective_stage_names)
        {
            for (const auto &stage : collective_stage_names)
            {
                if (stage.empty())
                    continue;
                if (stem == stage)
                    return true;
                const std::string suffix = "_" + stage;
                if (stem.size() >= suffix.size() &&
                    stem.substr(stem.size() - suffix.size()) == suffix)
                {
                    return true;
                }
            }
            return false;
        }
    } // namespace detail

    /**
     * @brief Derive the complete parity snapshot allocation envelope.
     *
     * @param snapshot_dir Authenticated directory containing NPY checkpoints.
     * @param reference_rows Logical prompt rows represented by prefill files.
     * @param retained_graph_rows Maximum physical rows in any retained graph.
     * @param collective_stage_names Values whose post-collective publication is
     *        captured beside its rank-local producer.
     * @param source_outputs Model-owned selection from the same frozen capture filter.
     *        Live state payload is excluded and replaced by metadata-native capacity in planning.
     * @return Cached metadata evidence and per-accelerator ordinary-payload capacity.
     * @throws std::invalid_argument for missing geometry/inventory.
     * @throws std::runtime_error for malformed or unreadable NPY metadata.
     */
    inline ParitySnapshotMemoryCapacityEvidence
    deriveParitySnapshotMemoryCapacity(
        const std::filesystem::path &snapshot_dir,
        int reference_rows,
        int retained_graph_rows,
        std::vector<std::string> collective_stage_names,
        std::vector<GraphSnapshotSourceOutput> source_outputs = {},
        ParitySnapshotPayloadProjection projection = ParitySnapshotPayloadProjection::ObservationEnvelope)
    {
        if (reference_rows <= 0 || retained_graph_rows <= 0 ||
            !std::filesystem::is_directory(snapshot_dir))
        {
            throw std::invalid_argument(
                "Parity snapshot capacity requires a reference directory and positive row geometry");
        }

        std::sort(
            collective_stage_names.begin(),
            collective_stage_names.end());
        collective_stage_names.erase(
            std::unique(
                collective_stage_names.begin(),
                collective_stage_names.end()),
            collective_stage_names.end());
        std::sort(source_outputs.begin(), source_outputs.end(), [](const auto &left, const auto &right) {
            return left.semanticKey() < right.semanticKey();
        });
        if (std::adjacent_find(source_outputs.begin(), source_outputs.end()) != source_outputs.end())
            throw std::invalid_argument("Parity snapshot source inventory contains duplicate producers");
        std::set<std::string> observed_sources;
        std::string cache_key = snapshot_dir.lexically_normal().string() +
                                "|" + std::to_string(reference_rows) +
                                "|" + std::to_string(retained_graph_rows) +
                                "|projection=" + std::to_string(static_cast<int>(projection));
        for (const auto &stage : collective_stage_names)
            cache_key += "|" + stage;
        for (const auto &source : source_outputs)
            cache_key += "|native-source=" + source.semanticKey();

        static std::mutex cache_mutex;
        static std::map<
            std::string,
            ParitySnapshotMemoryCapacityEvidence>
            cache;
        {
            std::lock_guard lock(cache_mutex);
            const auto found = cache.find(cache_key);
            if (found != cache.end())
                return found->second;
        }

        ParitySnapshotMemoryCapacityEvidence evidence;
        evidence.reference_rows = reference_rows;
        evidence.retained_graph_rows = retained_graph_rows;
        std::size_t capacity_bytes = 0u;
        std::map<std::string, std::size_t> main_prefill_sources, main_scalar_sources;
        std::size_t other_reference_bytes = 0u;

        for (const auto &entry :
             std::filesystem::directory_iterator(snapshot_dir))
        {
            if (!entry.is_regular_file() ||
                entry.path().extension() != ".npy")
            {
                continue;
            }

            const std::string stem = entry.path().stem().string();
            const bool decode_artifact =
                stem.rfind("decode_step", 0u) == 0u;
            std::string semantic_key = stem;
            std::size_t reference_prefix_tokens = static_cast<std::size_t>(reference_rows);
            if (decode_artifact)
            {
                const auto separator = stem.find('_', std::string_view("decode_step").size());
                if (separator != std::string::npos)
                {
                    const auto digits = std::string_view(stem).substr(11, separator - 11);
                    if (!digits.empty() && std::all_of(digits.begin(), digits.end(),
                        [](char c) { return c >= '0' && c <= '9'; }))
                    {
                        std::size_t step = 0;
                        const auto parsed = std::from_chars(digits.data(), digits.data()+digits.size(), step);
                        if (parsed.ec != std::errc{} || parsed.ptr != digits.data()+digits.size())
                            throw std::invalid_argument("Selected snapshot decode index is not representable");
                        reference_prefix_tokens = detail::paritySnapshotCheckedAdd(reference_prefix_tokens,
                            detail::paritySnapshotCheckedAdd(step, 1u, "source decode row"), "source prefix rows");
                        semantic_key = stem.substr(separator + 1);
                    }
                }
            }
            const auto source = std::find_if(source_outputs.begin(), source_outputs.end(),
                [&](const auto &entry) { return entry.semanticKey() == semantic_key; });
            const bool native_state = source != source_outputs.end() && source->replacesReferencePayload();
            if (source != source_outputs.end()) observed_sources.insert(source->semanticKey());
            if (source != source_outputs.end())
            {
                // Authenticate dtype, complete file extent and exact logical
                // rows before cnpy can assert on an unchecked descriptor.
                const auto descriptor = test::inspectNpySnapshot(entry.path().string());
                const auto role = source->role();
                const bool selected_indices = role == GraphSnapshotSourceRole::QwenSparseSelectedIndices;
                if (descriptor.scalar != test::NpySnapshotScalar::FP32 &&
                    !(selected_indices && descriptor.scalar == test::NpySnapshotScalar::INT64))
                    throw std::invalid_argument("Selected QSA reference scalar disagrees with its source role");
                const std::size_t logical_rows = native_state ? reference_prefix_tokens / qwen4exp::QwenSparsePool :
                    decode_artifact ? 1u : static_cast<std::size_t>(reference_rows);
                const std::size_t columns = native_state ? qwen4exp::QwenSparseIndexDim :
                    selected_indices ? qwen4exp::QwenSparseSelected :
                    qwen4exp::QwenSparseIndexHeads * qwen4exp::QwenSparseIndexDim;
                const auto &dimensions = descriptor.shape;
                const bool native_layout = dimensions == std::vector<std::size_t>{logical_rows, columns} ||
                    dimensions == std::vector<std::size_t>{1u, logical_rows, columns};
                const bool query_head_layout = role == GraphSnapshotSourceRole::QwenSparseQuery &&
                    dimensions == std::vector<std::size_t>{1u, logical_rows,
                        qwen4exp::QwenSparseIndexHeads, qwen4exp::QwenSparseIndexDim};
                if (!native_layout && !query_head_layout)
                    throw std::invalid_argument("Selected QSA reference shape disagrees with its authored rows and source role");
            }

            std::unique_ptr<std::FILE, detail::ParitySnapshotFileCloser> file(
                std::fopen(entry.path().c_str(), "rb"));
            if (!file)
            {
                throw std::runtime_error(
                    "Cannot open parity NPY header: " +
                    entry.path().string());
            }

            std::size_t word_size = 0u;
            std::vector<std::size_t> shape;
            bool fortran_order = false;
            try
            {
                cnpy::parse_npy_header(
                    file.get(), word_size, shape, fortran_order);
            }
            catch (const std::exception &error)
            {
                throw std::runtime_error(
                    "Cannot parse parity NPY header " +
                    entry.path().string() + ": " + error.what());
            }
            if (word_size == 0u || shape.empty())
            {
                throw std::runtime_error(
                    "Parity NPY header has empty dtype or shape: " +
                    entry.path().string());
            }

            std::size_t elements = 1u;
            for (const std::size_t dimension : shape)
            {
                elements = detail::paritySnapshotCheckedMultiply(
                    elements, dimension, "NPY element count");
            }
            const std::size_t payload_bytes =
                detail::paritySnapshotCheckedMultiply(
                    elements, word_size, "NPY payload bytes");
            evidence.unscaled_payload_bytes =
                detail::paritySnapshotCheckedAdd(
                    evidence.unscaled_payload_bytes,
                    payload_bytes,
                    "reference payload total");

            const std::size_t admitted_payload = decode_artifact
                ? payload_bytes
                : detail::widenParityPrefillPayload(
                      payload_bytes,
                      reference_rows,
                      retained_graph_rows);
            std::size_t projected_payload = native_state ? 0u : admitted_payload;
            if (native_state)
            {
                // Authentication is against the exact selected source key, not
                // an arbitrary filename suffix. State rows may be zero, while
                // the native captured slot still retains its full context bank.
                if (word_size != sizeof(float) || fortran_order ||
                    (shape.size() != 2 && shape.size() != 3) ||
                    shape.back() != qwen4exp::QwenSparseIndexDim ||
                    (shape.size() == 3 && shape.front() != 1))
                    throw std::invalid_argument("Parity QSA pool reference disagrees with native FP32 state geometry");
                if (detail::paritySnapshotHasCollectiveAlias(stem, collective_stage_names))
                    throw std::invalid_argument("Native QSA pool snapshot cannot acquire an unowned collective alias");
                evidence.replaced_state_payload_bytes = detail::paritySnapshotCheckedAdd(
                    evidence.replaced_state_payload_bytes, admitted_payload, "replaced live state inventory");
            }
            else
            {
                capacity_bytes = detail::paritySnapshotCheckedAdd(
                    capacity_bytes, admitted_payload, "capacity inventory");
            }

            if (detail::paritySnapshotHasCollectiveAlias(
                    stem, collective_stage_names))
            {
                evidence.collective_alias_bytes =
                    detail::paritySnapshotCheckedAdd(
                        evidence.collective_alias_bytes,
                        admitted_payload,
                        "collective alias inventory");
                capacity_bytes = detail::paritySnapshotCheckedAdd(
                    capacity_bytes,
                    admitted_payload,
                    "capacity plus collective alias");
                projected_payload = detail::paritySnapshotCheckedAdd(projected_payload,
                    admitted_payload, "phase collective alias");
            }
            if (semantic_key.starts_with("MTP"))
                other_reference_bytes = detail::paritySnapshotCheckedAdd(other_reference_bytes,
                    projected_payload, "separate predictor reference payload");
            else
            {
                auto &sources = decode_artifact ? main_scalar_sources : main_prefill_sources;
                sources[semantic_key] = std::max(sources[semantic_key], projected_payload);
            }
            ++evidence.reference_file_count;
        }

        if (observed_sources.size() != source_outputs.size())
            throw std::invalid_argument("Selected source-owned snapshot has no authenticated reference header");
        evidence.capacity.authored_source_outputs = std::move(source_outputs);
        if (evidence.reference_file_count == 0u ||
            (capacity_bytes == 0u && evidence.capacity.authored_source_outputs.empty()))
        {
            throw std::invalid_argument(
                "Parity snapshot capacity found no NPY checkpoint payloads in " +
                snapshot_dir.string());
        }

        // Device allocations and individual slots are at least float-aligned.
        constexpr std::size_t kArenaAlignment = alignof(float);
        const std::size_t alignment_remainder =
            capacity_bytes % kArenaAlignment;
        if (alignment_remainder != 0u)
        {
            capacity_bytes = detail::paritySnapshotCheckedAdd(
                capacity_bytes,
                kArenaAlignment - alignment_remainder,
                "arena alignment");
        }
        evidence.capacity.per_accelerator_bytes = capacity_bytes;
        if (projection == ParitySnapshotPayloadProjection::RetainedMainServingPhases)
        {
            if (main_prefill_sources.empty() || main_scalar_sources.empty())
                throw std::invalid_argument("Retained serving snapshots require authored prefill and scalar reference phases");
            // A prefill-only observation also belongs to the frozen capture
            // filter. Project its token payload to scalar rows before taking
            // the per-source union, rather than silently omitting that output.
            for (const auto &[key, bytes] : main_prefill_sources)
            {
                const auto rows = static_cast<std::size_t>(retained_graph_rows);
                const auto scalar = bytes / rows + (bytes % rows != 0u);
                main_scalar_sources[key] = std::max(main_scalar_sources[key], scalar);
            }
            std::size_t scalar = 0u;
            for (const auto &[key, bytes] : main_scalar_sources)
            {
                scalar = detail::paritySnapshotCheckedAdd(scalar, bytes, "scalar source union");
                // The captured filter is a union across observations. A source
                // absent from a prefill observation still needs its admitted
                // wide-row geometry; never substitute max(phase totals).
                const auto wide = detail::paritySnapshotCheckedMultiply(bytes,
                    static_cast<std::size_t>(retained_graph_rows), "wide scalar-source projection");
                main_prefill_sources[key] = std::max(main_prefill_sources[key], wide);
            }
            std::size_t wide = 0u;
            for (const auto &[key, bytes] : main_prefill_sources)
                wide = detail::paritySnapshotCheckedAdd(wide, bytes, "wide source union");
            evidence.capacity.main_phase_payload = MainSnapshotPhasePayload{
                .wide_main_bytes = wide, .scalar_main_bytes = scalar,
                .separate_reference_bytes = other_reference_bytes};
        }

        {
            std::lock_guard lock(cache_mutex);
            cache.emplace(cache_key, evidence);
        }
        return evidence;
    }
} // namespace llaminar2::test::parity
