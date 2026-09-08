/**
 * @file ParitySnapshotMemoryCapacity.h
 * @brief Metadata-only memory envelope for production parity graph snapshots.
 *
 * The authenticated reference directory is also the authoritative checkpoint
 * inventory selected into production graphs. This helper reads only each NPY
 * header, never its numerical payload, and converts that inventory into a
 * conservative per-accelerator capacity. Prefill payloads are widened to the
 * retained graph-row capacity and completed-collective aliases are charged as
 * distinct live values. Results are cached because a process campaign reuses
 * the same immutable reference pack across many generated cells.
 */

#pragma once

#include "planning/GraphSnapshotMemoryCapacity.h"

#include <cnpy.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test::parity
{
    /** @brief Auditable evidence behind one graph-snapshot capacity declaration. */
    struct ParitySnapshotMemoryCapacityEvidence
    {
        GraphSnapshotMemoryCapacity capacity;
        std::size_t reference_file_count = 0u;
        std::size_t unscaled_payload_bytes = 0u;
        std::size_t collective_alias_bytes = 0u;
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
     * @return Cached metadata evidence and per-accelerator byte capacity.
     * @throws std::invalid_argument for missing geometry/inventory.
     * @throws std::runtime_error for malformed or unreadable NPY metadata.
     */
    inline ParitySnapshotMemoryCapacityEvidence
    deriveParitySnapshotMemoryCapacity(
        const std::filesystem::path &snapshot_dir,
        int reference_rows,
        int retained_graph_rows,
        std::vector<std::string> collective_stage_names)
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
        std::string cache_key = snapshot_dir.lexically_normal().string() +
                                "|" + std::to_string(reference_rows) +
                                "|" + std::to_string(retained_graph_rows);
        for (const auto &stage : collective_stage_names)
            cache_key += "|" + stage;

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

        for (const auto &entry :
             std::filesystem::directory_iterator(snapshot_dir))
        {
            if (!entry.is_regular_file() ||
                entry.path().extension() != ".npy")
            {
                continue;
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

            const std::string stem = entry.path().stem().string();
            const bool decode_artifact =
                stem.rfind("decode_step", 0u) == 0u;
            const std::size_t admitted_payload = decode_artifact
                ? payload_bytes
                : detail::widenParityPrefillPayload(
                      payload_bytes,
                      reference_rows,
                      retained_graph_rows);
            capacity_bytes = detail::paritySnapshotCheckedAdd(
                capacity_bytes,
                admitted_payload,
                "capacity inventory");

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
            }
            ++evidence.reference_file_count;
        }

        if (evidence.reference_file_count == 0u || capacity_bytes == 0u)
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

        {
            std::lock_guard lock(cache_mutex);
            cache.emplace(cache_key, evidence);
        }
        return evidence;
    }
} // namespace llaminar2::test::parity
