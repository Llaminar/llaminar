/**
 * @file MoEProjectionArenaGeometry.h
 * @brief One metadata-derived activation BOM for the distributed projection graph.
 *
 * Admission and graph materialization consume the same five row-bank geometries.
 * They replace, rather than accompany, whole-expert route publication storage.
 * The packet envelope includes a four-byte original-route ID per compact record.
 * The packet retains the down kernel's existing activation encoding: it is not
 * a new activation-precision choice or an independent live memory ledger.
 */
#pragma once

#include "memory/BufferId.h"
#include <array>
#include <cstddef>
#include <string_view>

namespace llaminar2
{
    struct ModelMemoryProfile;
    struct GraphSchema;
    struct GraphResolverConfig;

    /** @brief A persistent graph buffer's identity and uint32/FP32 row capacity. */
    struct MoEProjectionArenaBank
    {
        BufferId id;
        std::string_view name;
        std::size_t words_per_row;
    };

    /**
     * @brief Validated maximum row geometry across main and MTP expert layers.
     *
     * Layers execute sequentially and reuse these banks. Packet width is the
     * maximum actual encoded width, not FP32 width guessed for quantized models.
     * Distinct banks remain independently owned because producers and consumers
     * coexist across the native collective edges.
     */
    class MoEProjectionArenaGeometry final
    {
    public:
        /**
         * @brief Check the native equal-column partition before proposing a topology.
         * @param columns Source model's positive output width.
         * @param participants Proposed native multi-device degree, at least two.
         * @return Whether every output column has one equal-width participant owner.
         * @throws std::invalid_argument For missing source width or invalid degree.
         *
         * Automatic search and physical admission share this shape invariant.
         * An optional nonintegral topology is not an executable proposal; an
         * explicitly applied incompatible plan still fails admission. This
         * neither changes its compute policy nor invents padded columns.
         */
        [[nodiscard]] static bool hasIntegralOutputPartition(int columns, int participants);

        /**
         * @brief Resolve the exact shared arena envelope from source metadata.
         * @param profile Complete source tensor directory, including MTP layers.
         * @param participants Homogeneous native collective degree.
         * @return The five persistent buffer capacities, in 32-bit words per row.
         * @throws std::invalid_argument For missing layers, formats or incompatible slices.
         * @throws std::overflow_error For unrepresentable buffer geometry.
         */
        [[nodiscard]] static MoEProjectionArenaGeometry resolve(
            const ModelMemoryProfile &profile, int participants);

        /** @return Buffer identities shared by schema, resolver and memory admission. */
        [[nodiscard]] const std::array<MoEProjectionArenaBank, 5> &banks() const noexcept
        { return banks_; }

        /** @brief Maximum per-producer packet storage shared by all sequential model layers.
         * @param rows Admitted simultaneous token rows, including retained MTP width.
         * @return Exact transfer capacity; replay's wire length remains device-authored.
         * @throws std::invalid_argument For zero rows.
         * @throws std::overflow_error For unrepresentable storage. */
        [[nodiscard]] std::size_t packetCapacityBytes(std::size_t rows) const;

        /**
         * @brief Sum the logical owned bytes at an admitted row capacity.
         * @param rows Maximum concurrent prefill or grouped-verifier rows.
         * @return Required BufferArena bytes, contributed to PhysicalMemoryAuthority.
         * @throws std::invalid_argument For an empty row capacity.
         * @throws std::overflow_error If the total cannot be represented.
         */
        [[nodiscard]] std::size_t bytes(std::size_t rows) const;

        /**
         * @brief Replace whole-expert publication storage with these five owners.
         * @param schema Fresh model schema containing exactly one canonical route bank.
         * @throws std::invalid_argument For missing or already-replaced publication storage.
         */
        void replaceWholeExpertSchema(GraphSchema &schema) const;

        /** @brief Bind the same buffer IDs and row widths to the model's resolver. */
        void bindResolver(GraphResolverConfig &resolver) const;

    private:
        /** @brief Seal checked row widths; callers cannot construct a partial BOM. */
        MoEProjectionArenaGeometry(std::size_t packet_words, std::size_t top_k,
            std::size_t model_columns, std::size_t participants);

        std::array<MoEProjectionArenaBank, 5> banks_;
    };
} // namespace llaminar2
