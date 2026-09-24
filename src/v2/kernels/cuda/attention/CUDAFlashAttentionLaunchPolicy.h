/**
 * @file CUDAFlashAttentionLaunchPolicy.h
 * @brief Capture-time CUDA prefill-attention launch geometry policy.
 *
 * CUDA Flash Attention owns several byte-equivalent query-row partitions for
 * each supported head dimension.  A smaller partition duplicates the K/V scan
 * across more blocks, but can be substantially faster when the ordinary grid
 * leaves most streaming multiprocessors idle.  This header centralizes the
 * immutable geometry calculation so production capture, device-free unit
 * tests, and kernel performance tournaments all reason about the same policy.
 *
 * The policy is deliberately model agnostic.  Qwen dense and MoE releases are
 * represented by their physical launch geometry: batch size, query rows, local
 * query-head count, head dimension, and device SM count.  Consequently new
 * model sizes and tensor-parallel head shards receive a total decision without
 * adding a model-name branch or a runtime host callback.
 */

#pragma once

#include "../../attention/AttentionExecutionPolicy.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2::cuda::fa2_policy
{
    /** Number of query rows owned by one WMMA score group. */
    inline constexpr int kFA2QueryRowsPerWarpGroup = 16;

    /** Number of K/V shared-memory buffers in the producer pipeline. */
    inline constexpr int kFA2KVSharedMemoryStages = 2;

    /** Half elements appended to each Q/K/V shared-memory row. */
    inline constexpr int kFA2QKVSharedMemoryPad = 8;

    /** FP32 elements appended to each score shared-memory row. */
    inline constexpr int kFA2ScoreSharedMemoryPad = 8;

    /**
     * @brief Keys in one canonical batch-invariant online-softmax partition.
     *
     * Direct FA2 finalizes this exact span inside its owning CTA. Context-
     * parallel FA2 publishes the same summaries to a deterministic reducer.
     * Keeping one global width makes physical scheduling byte-invariant across
     * M, TP geometry, and captured direct/context execution.
     */
    inline constexpr int kFA2CanonicalContextPartitionKeys = 256;

    /** First row count not owned by grouped serial-decode-equivalent attention. */
    inline constexpr int kFA2ContextPrefillMinimumQueryRows = 17;

    /**
     * @brief Productive grid-wave budget for the persistent K/V-context phase.
     *
     * A GA102 tournament over the low-slot Qwen crossover geometries found that
     * two waves severely under-filled phase one.  Thirty-two waves mapped HD128
     * and HD256 grids to 11--24 persistent slots and stayed within about one
     * percent of their measured throughput winner at both 8K and 128K context.
     * The budget scales with the physical SM count rather than a model name.
     */
    inline constexpr int kFA2ContextTargetBlockWaves = 32;

    /**
     * @brief Immutable inputs used while constructing a captured FA2 launch.
     */
    struct FA2QueryPartitionGeometry
    {
        int batch_size = 0;        ///< Captured request batch size.
        int query_rows = 0;        ///< Physical query rows in the graph bucket.
        int local_query_heads = 0; ///< Query heads executed by this participant.
        int head_dim = 0;          ///< Elements in one query/key head.
        int sm_count = 0;          ///< SMs visible on the selected CUDA device.
    };

    /**
     * @brief Physical K/V tile inputs fixed when the FA2 graph is captured.
     *
     * Query partitioning already incorporates batch, M, local head count, and
     * the device SM count.  The resulting warp-group count is therefore the
     * compact work-geometry signal needed by the physical K/V tile policy.  The
     * shared-memory ceiling remains explicit so an otherwise valid policy can
     * never launch an unsupported specialization on a smaller CUDA device.
     */
    struct FA2KVTileGeometry
    {
        int head_dim = 0;                   ///< Elements in one attention head.
        int query_warp_groups = 0;          ///< Selected 16-row query groups.
        std::size_t max_dynamic_smem = 0;   ///< Device opt-in shared-memory cap.
    };

    /**
     * @brief Query-to-KV head mapping visible to one participant launch.
     *
     * A locally sharded KV tensor uses `replicated_gqa_n_rep == 0`; local Q
     * heads must then divide evenly over the local KV heads.  A replicated KV
     * tensor supplies the full-model GQA ratio and global Q-head offset.  That
     * second form deliberately permits a local Q shard that is not divisible
     * by the global KV-head count, as occurs at TP=4/8 for compact GQA models.
     */
    struct FA2HeadMappingGeometry
    {
        int local_query_heads = 0;   ///< Q heads executed by this participant.
        int visible_kv_heads = 0;    ///< KV heads addressable from the pointer.
        int head_start = 0;          ///< Global first Q head for replicated KV.
        int replicated_gqa_n_rep = 0; ///< Global Q heads mapped to each KV head.
    };

    /**
     * @brief Validate local-sharded or global-replicated GQA head mapping.
     *
     * @param geometry Participant-visible head mapping.
     * @return True exactly when every launched Q head maps to an addressable KV
     *         head without relying on a local divisibility assumption.
     */
    [[nodiscard]] inline constexpr bool isValidFA2HeadMapping(
        const FA2HeadMappingGeometry &geometry)
    {
        if (geometry.local_query_heads <= 0 ||
            geometry.visible_kv_heads <= 0 ||
            geometry.head_start < 0 ||
            geometry.replicated_gqa_n_rep < 0)
        {
            return false;
        }

        if (geometry.replicated_gqa_n_rep == 0)
        {
            return geometry.local_query_heads % geometry.visible_kv_heads == 0;
        }

        const std::int64_t final_query_head =
            static_cast<std::int64_t>(geometry.head_start) +
            geometry.local_query_heads;
        const std::int64_t represented_query_heads =
            static_cast<std::int64_t>(geometry.visible_kv_heads) *
            geometry.replicated_gqa_n_rep;
        return final_query_head <= represented_query_heads;
    }

    /**
     * @brief Return the widest compiled query grouping for a head dimension.
     *
     * Wider groups amortize each block's K/V scan across more query rows.  The
     * values mirror the established HD64, HD128, and HD256 FA2 geometries.
     * Returning zero marks an unsupported head dimension and is a fatal launch
     * configuration error at the production boundary.
     *
     * @param head_dim Elements in one attention head.
     * @return Maximum compiled group count, or zero when unsupported.
     */
    [[nodiscard]] inline constexpr int maximumFA2QueryWarpGroups(int head_dim)
    {
        return head_dim <= 0
                   ? 0
               : head_dim <= 64
                   ? 6
               : head_dim <= 128
                   ? 4
               : head_dim <= 256
                   ? 2
                   : 0;
    }

    /**
     * @brief Return the compiled P@V stripe envelope for one query group.
     *
     * HD256 assigns two lanes per query row. Powers-of-two stripe counts through
     * four divide its 256 output dimensions exactly. Narrower heads retain the
     * established one-warp ownership.
     *
     * @param head_dim Elements in one attention output head.
     * @return Maximum compiled stripe warps, or zero when unsupported.
     */
    [[nodiscard]] inline constexpr int maximumFA2PVWarpsPerQueryGroup(
        int head_dim)
    {
        return head_dim <= 0
                   ? 0
               : head_dim <= 128
                   ? 1
               : head_dim <= 256
                   ? 4
                   : 0;
    }

    /**
     * @brief Validate one compiled, byte-equivalent P@V stripe geometry.
     *
     * @param head_dim Elements in one attention output head.
     * @param pv_warps Requested P@V warps per 16-row query group.
     * @return True exactly for a supported power-of-two specialization.
     */
    [[nodiscard]] inline constexpr bool isCompiledFA2PVWarpGeometry(
        int head_dim,
        int pv_warps)
    {
        const int maximum_warps =
            maximumFA2PVWarpsPerQueryGroup(head_dim);
        return maximum_warps > 0 && pv_warps > 0 &&
               pv_warps <= maximum_warps &&
               (pv_warps & (pv_warps - 1)) == 0;
    }

    /**
     * @brief Calculate exact dynamic shared memory for one compiled FA2 tile.
     *
     * Production launch selection, captured-graph inspection, and performance
     * tournaments must use one byte calculation. Duplicating this formula made
     * it possible for a benchmark to label one physical TILE_KV while the graph
     * actually contained another. The layout is one Q tile, two pipelined K/V
     * buffer pairs, and one FP32 score tile with the bank-conflict padding used
     * by the kernel.
     *
     * @param head_dim Elements in one query/key/value head.
     * @param query_warp_groups Independent 16-row query groups in one block.
     * @param tile_kv Physical K/V rows staged per pipeline iteration.
     * @return Required bytes, or zero for an unsupported specialization.
     */
    [[nodiscard]] inline constexpr std::size_t fa2DynamicSharedMemoryBytes(
        int head_dim,
        int query_warp_groups,
        int tile_kv)
    {
        const int maximum_groups = maximumFA2QueryWarpGroups(head_dim);
        if (maximum_groups == 0 || query_warp_groups <= 0 ||
            query_warp_groups > maximum_groups ||
            (tile_kv != 16 && tile_kv != 32 && tile_kv != 64))
        {
            return 0;
        }

        const std::size_t tile_q =
            static_cast<std::size_t>(query_warp_groups) *
            kFA2QueryRowsPerWarpGroup;
        const std::size_t qkv_stride =
            static_cast<std::size_t>(head_dim) +
            kFA2QKVSharedMemoryPad;
        const std::size_t score_stride =
            static_cast<std::size_t>(tile_kv) +
            kFA2ScoreSharedMemoryPad;
        const std::size_t q_bytes = tile_q * qkv_stride * sizeof(std::uint16_t);
        const std::size_t kv_bytes =
            kFA2KVSharedMemoryStages * 2u *
            static_cast<std::size_t>(tile_kv) * qkv_stride *
            sizeof(std::uint16_t);
        const std::size_t score_bytes =
            tile_q * score_stride * sizeof(float);
        return q_bytes + kv_bytes + score_bytes;
    }

    /**
     * @brief Select the physical K/V tile for a captured CUDA FA2 launch.
     *
     * A 288-cell tournament over every released Qwen attention geometry,
     * TP=1/2/4/8, M=64/128/512, and KV lengths through 131072 established three
     * stable facts on GA102: the 64-row tile never won; HD128 and HD256 stayed
     * within 3.31 percent of the winner with TILE_KV=16; and HD64's three- and
     * six-group query blocks required TILE_KV=32 (the old selector missed by as
     * much as 8.25 percent).  The rule below expresses those physical facts in
     * head/query geometry rather than model names or measured-point overlays.
     *
     * The explicit tile is reserved for isolated parity/profiling tournaments.
     * It must name a compiled specialization and fit the device exactly; bad
     * requests return zero so capture fails instead of silently substituting a
     * different kernel.  Production selection may choose TILE_KV=16 when the
     * preferred HD64 tile cannot fit, because shared-memory capacity is an
     * input to the policy rather than a launch-failure recovery path.
     *
     * @param geometry Immutable physical launch geometry.
     * @param explicit_tile_kv Zero for production policy, otherwise 16, 32, or
     *        64 for an authenticated tuning launch.
     * @return Selected tile width, or zero for invalid/unsupported geometry.
     */
    [[nodiscard]] inline constexpr int selectFA2KVTile(
        const FA2KVTileGeometry &geometry,
        int explicit_tile_kv = 0)
    {
        const int maximum_groups =
            maximumFA2QueryWarpGroups(geometry.head_dim);
        if (maximum_groups == 0 || geometry.query_warp_groups <= 0 ||
            geometry.query_warp_groups > maximum_groups ||
            geometry.max_dynamic_smem == 0 || explicit_tile_kv < 0 ||
            (explicit_tile_kv != 0 && explicit_tile_kv != 16 &&
             explicit_tile_kv != 32 && explicit_tile_kv != 64))
        {
            return 0;
        }

        const auto fits = [&](int tile_kv) constexpr
        {
            const std::size_t bytes = fa2DynamicSharedMemoryBytes(
                geometry.head_dim,
                geometry.query_warp_groups,
                tile_kv);
            return bytes > 0 && bytes <= geometry.max_dynamic_smem;
        };

        if (explicit_tile_kv != 0)
            return fits(explicit_tile_kv) ? explicit_tile_kv : 0;

        const int preferred_tile =
            geometry.head_dim <= 64 && geometry.query_warp_groups >= 3
                ? 32
                : 16;
        if (fits(preferred_tile))
            return preferred_tile;

        return preferred_tile != 16 && fits(16) ? 16 : 0;
    }

    /**
     * @brief Return the useful grid-residency budget for a head dimension.
     *
     * HD128 query groups use compact blocks whose measured occupancy sustains
     * two productive grid waves on GA102.  Restricting them to one wave chooses
     * an unnecessarily wide block for geometries such as 28 local heads at
     * M=128.  HD256 uses four waves because its 256-wide P@V stripe geometry
     * loses more to the wider two-group block than it gains by collapsing
     * small-M grids: the complete Qwen 0.8B-through-397B, TP1/2/4/8 tournament
     * selected one group at M=64 and M=128.  HD64 retains one wave because its
     * wider group inventory amortizes K/V scans without HD256's register and
     * consumer-warp cost.  The return value is dimension-derived and therefore
     * remains total for unseen model names, TP degrees, and SM counts.
     *
     * @param head_dim Elements in one query/key head.
     * @return Preferred complete-grid wave count, or zero when unsupported.
     */
    [[nodiscard]] inline constexpr int preferredFA2QueryGridWaves(int head_dim)
    {
        return head_dim > 128 && head_dim <= 256
                   ? 4
               : head_dim > 64 && head_dim <= 128
                   ? 2
                   : maximumFA2QueryWarpGroups(head_dim) > 0 ? 1 : 0;
    }

    /**
     * @brief Calculate the complete block count for one query partition.
     *
     * @param geometry Immutable capture geometry.
     * @param query_warp_groups Independent 16-row score groups in each block.
     * @return Grid blocks, or zero for invalid input/overflow.
     */
    [[nodiscard]] inline constexpr std::int64_t fa2QueryGridBlocks(
        const FA2QueryPartitionGeometry &geometry,
        int query_warp_groups)
    {
        if (geometry.batch_size <= 0 || geometry.query_rows <= 0 ||
            geometry.local_query_heads <= 0 || query_warp_groups <= 0)
        {
            return 0;
        }

        const std::int64_t tile_rows =
            static_cast<std::int64_t>(query_warp_groups) *
            kFA2QueryRowsPerWarpGroup;
        const std::int64_t query_tiles =
            (static_cast<std::int64_t>(geometry.query_rows) + tile_rows - 1) /
            tile_rows;
        return static_cast<std::int64_t>(geometry.batch_size) *
               geometry.local_query_heads * query_tiles;
    }

    /**
     * @brief Choose the narrowest partition within its useful residency budget.
     *
     * Query-group counts are examined from one through the widest compiled
     * specialization.  The first count whose complete grid does not exceed the
     * head-dimension-specific SM-wave budget maximizes useful occupancy without
     * repeating more block-owned K/V work than that block geometry can hide.
     * When even the widest geometry exceeds the budget, the widest
     * specialization is the deliberate throughput choice because it minimizes
     * duplicated K/V work.
     *
     * An explicit override is a profiling/tournament control.  It is accepted
     * only when that exact specialization exists for the requested head
     * dimension; invalid input returns zero so the launcher can fail hard.
     *
     * @param geometry Immutable capture geometry.
     * @param explicit_query_warp_groups Zero for policy, otherwise an exact
     *        compiled group count requested by a tuning run.
     * @return Selected group count, or zero for invalid geometry/override.
     */
    [[nodiscard]] inline constexpr int selectFA2QueryWarpGroups(
        const FA2QueryPartitionGeometry &geometry,
        int explicit_query_warp_groups = 0)
    {
        const int maximum_groups =
            maximumFA2QueryWarpGroups(geometry.head_dim);
        const int preferred_waves =
            preferredFA2QueryGridWaves(geometry.head_dim);
        if (maximum_groups == 0 || geometry.sm_count <= 0 ||
            preferred_waves == 0 ||
            fa2QueryGridBlocks(geometry, 1) == 0 ||
            explicit_query_warp_groups < 0 ||
            explicit_query_warp_groups > maximum_groups)
        {
            return 0;
        }

        if (explicit_query_warp_groups > 0)
            return explicit_query_warp_groups;

        const std::int64_t grid_budget =
            static_cast<std::int64_t>(geometry.sm_count) * preferred_waves;
        for (int groups = 1; groups <= maximum_groups; ++groups)
        {
            if (fa2QueryGridBlocks(geometry, groups) <= grid_budget)
                return groups;
        }
        return maximum_groups;
    }

    /**
     * @brief Concrete prefill axis embedded in one captured CUDA graph.
     */
    enum class FA2PrefillPhysicalMode : std::uint8_t
    {
        QuerySequence,
        KeyValueContext,
        DeviceAdaptive,
    };

    /**
     * @brief Return the stable diagnostic spelling of a physical FA2 mode.
     *
     * @param mode Concrete capture-time mode.
     * @return Process-lifetime string literal for PerfStats and diagnostics.
     */
    [[nodiscard]] inline constexpr const char *fa2PrefillPhysicalModeName(
        FA2PrefillPhysicalMode mode)
    {
        switch (mode)
        {
        case FA2PrefillPhysicalMode::QuerySequence:
            return "query_sequence";
        case FA2PrefillPhysicalMode::KeyValueContext:
            return "key_value_context";
        case FA2PrefillPhysicalMode::DeviceAdaptive:
            return "device_adaptive";
        }
        return "invalid";
    }

    /**
     * @brief Complete immutable geometry used to select a prefill graph shape.
     */
    struct FA2PrefillParallelGeometry
    {
        int batch_size = 0;        ///< Captured request batch size.
        int query_rows = 0;        ///< Physical rows in the prefill bucket.
        int local_query_heads = 0; ///< Heads computed by this participant.
        int head_dim = 0;          ///< Elements in one attention head.
        int kv_capacity = 0;       ///< Graph-stable cache stride/envelope.
        int sm_count = 0;          ///< SM count on the selected device.
        attention::AttentionPrefillParallelAxis requested_axis =
            attention::AttentionPrefillParallelAxis::QuerySequence;
    };

    /**
     * @brief One fully resolved, capture-stable CUDA prefill plan.
     *
     * Context-summary byte counts are zero for query-sequence execution. For a
     * context plan they describe the exact persistent arena capacity required
     * by `[batch, query, head, partition, dimension]` summaries and their
     * scalar `(m, l)` companions. A false `valid` bit is a fatal planning
     * result; callers must never substitute another physical mode.
     */
    struct FA2PrefillParallelPlan
    {
        FA2PrefillPhysicalMode mode =
            FA2PrefillPhysicalMode::QuerySequence;
        int query_warp_groups = 0;
        int reducer_dimension_warps = 0;
        int context_partition_slots = 0;
        int device_direct_partition_limit = 0;
        int max_context_partitions = 0;
        std::int64_t query_grid_blocks = 0;
        std::size_t partial_output_bytes = 0;
        std::size_t partial_m_bytes = 0;
        std::size_t partial_l_bytes = 0;
        bool valid = false;

        /** @return True when this plan captures the K/V-context transaction. */
        [[nodiscard]] constexpr bool usesContextParallelism() const
        {
            return valid &&
                   (mode == FA2PrefillPhysicalMode::KeyValueContext ||
                    mode == FA2PrefillPhysicalMode::DeviceAdaptive);
        }

        /**
         * @return True when live device K/V length selects direct versus context
         *         arithmetic inside one immutable device-adaptive graph. The
         *         producer feeds a guarded query root and a concurrent IF-only
         *         context body, independent of the selected direct window.
         */
        [[nodiscard]] constexpr bool usesDeviceAdaptiveParallelism() const
        {
            return valid && mode == FA2PrefillPhysicalMode::DeviceAdaptive;
        }
    };

    /**
     * @brief Multiply positive workspace cardinalities without wrapping.
     *
     * @return Product, or zero when either input is zero or the product would
     *         exceed `size_t`.
     */
    [[nodiscard]] inline constexpr std::size_t checkedFA2WorkspaceProduct(
        std::size_t lhs,
        std::size_t rhs)
    {
        return lhs == 0 || rhs == 0 ||
                       lhs > std::numeric_limits<std::size_t>::max() / rhs
                   ? 0
                   : lhs * rhs;
    }

    /** Forward declaration for the complete prefill-plan resolver below. */
    [[nodiscard]] inline constexpr int selectFA2ReducerDimensionWarps(
        const FA2QueryPartitionGeometry &geometry,
        int explicit_dimension_warps = 0);

    /**
     * @brief Select persistent phase-one slots for K/V-context execution.
     *
     * The captured cache may reserve hundreds of canonical partitions even
     * when only a short prefix is live. Launching one CTA for every reserved
     * partition makes empty graph capacity visible as milliseconds of work.
     * Instead, this policy launches a bounded number of persistent CTAs sized
     * to the measured productive grid-wave budget.
     * Each CTA strides over live canonical partitions obtained from the
     * device-owned attention parameters. This preserves one immutable graph,
     * bounds launch overhead, and leaves the arithmetic partition tree intact.
     *
     * @param geometry Immutable query/head/device launch geometry.
     * @param max_context_partitions Capacity-sized canonical partition count.
     * @param explicit_slots Zero for generic policy, otherwise an exact slot
     *        count used by isolated parity and performance tournaments.
     * @return Slots per query-tile/head pair, or zero for invalid geometry.
     */
    [[nodiscard]] inline constexpr int selectFA2ContextPartitionSlots(
        const FA2QueryPartitionGeometry &geometry,
        int max_context_partitions,
        int explicit_slots = 0)
    {
        if (max_context_partitions <= 0 || explicit_slots < 0)
            return 0;
        if (explicit_slots > 0)
            return explicit_slots <= max_context_partitions
                       ? explicit_slots
                       : 0;

        const int query_warp_groups =
            selectFA2QueryWarpGroups(geometry);
        const std::int64_t base_blocks =
            fa2QueryGridBlocks(geometry, query_warp_groups);
        if (base_blocks <= 0 || geometry.sm_count <= 0)
            return 0;

        const std::int64_t target_blocks =
            static_cast<std::int64_t>(geometry.sm_count) *
            kFA2ContextTargetBlockWaves;
        const std::int64_t slots =
            (target_blocks + base_blocks - 1) / base_blocks;
        return static_cast<int>(
            slots < 1
                ? 1
                : slots > max_context_partitions
                      ? max_context_partitions
                      : slots);
    }

    /**
     * @brief Select the live-prefix window executed directly inside an adaptive graph.
     *
     * The returned value is a count of canonical 256-key partitions.  At or
     * below that count, the guarded direct root owns output and the IF-only
     * context body is absent. Above the limit, the root uniformly retires while
     * persistent slots publish canonical summaries and the reducer merges them.
     * Both siblings consume the same device-owned `kv_len`; graph topology and
     * launch arguments remain immutable across request reset, prefix restore,
     * and replay.
     *
     * A 720-domain GA102 tournament spanning every released Qwen head geometry,
     * TP=1/2/4/8, M=17/32/64/128, and live prefixes from 256 through 131072 found
     * one partition to be the general crossover. HD128 and HD256 grids at or
     * above one SM wave retain direct execution through two partitions. Fully
     * saturated HD128 grids need a wider window. M below 128 remains direct. At
     * M=128, a one-group grid already supplying at least seven quarter-waves
     * remains direct, while the 1.5-wave-to-1.75-wave band crosses after one
     * partition because K/V partitioning fills the missing residency. A
     * two-group saturated geometry crosses after 32 partitions. These are
     * physical occupancy rules; model names and host-observed live state never
     * participate.
     *
     * @param geometry Immutable query/head/device launch geometry.
     * @param query_warp_groups Compiled query grouping selected for this graph.
     * @param max_context_partitions Capacity-sized canonical partition count.
     * @return Maximum directly executed live partition count, or `-1` for an
     *         invalid geometry.
     */
    [[nodiscard]] inline constexpr int selectFA2DeviceDirectPartitionLimit(
        const FA2QueryPartitionGeometry &geometry,
        int query_warp_groups,
        int max_context_partitions)
    {
        const std::int64_t query_grid_blocks =
            fa2QueryGridBlocks(geometry, query_warp_groups);
        if (query_grid_blocks <= 0 || geometry.sm_count <= 0 ||
            query_warp_groups <= 0 || max_context_partitions <= 0)
        {
            return -1;
        }

        int direct_limit = 1;
        const std::int64_t sm_count = geometry.sm_count;
        if (geometry.head_dim > 64 && geometry.head_dim <= 128 &&
            query_grid_blocks >= sm_count)
        {
            direct_limit = 2;
        }
        if (geometry.head_dim > 128 && geometry.head_dim <= 256 &&
            query_grid_blocks >= sm_count &&
            query_grid_blocks < 2 * sm_count)
        {
            direct_limit = 2;
        }

        const bool saturated_hd128_grid =
            geometry.head_dim > 64 && geometry.head_dim <= 128 &&
            2 * query_grid_blocks >= 3 * sm_count;
        if (saturated_hd128_grid)
        {
            if (geometry.query_rows < 128)
                return max_context_partitions;

            if (query_warp_groups == 1)
            {
                const bool nearly_two_complete_grid_waves =
                    4 * query_grid_blocks >= 7 * sm_count;
                if (nearly_two_complete_grid_waves)
                    return max_context_partitions;
                direct_limit = 1;
            }
            else
            {
                direct_limit = 32;
            }
        }

        return direct_limit > max_context_partitions
                   ? max_context_partitions
                   : direct_limit;
    }

    /**
     * @brief Resolve query-sequence versus K/V-context execution for capture.
     *
     * Geometry-selected execution captures one device-adaptive transaction when
     * the 32-wave persistent scheduler exposes more than one useful context
     * slot. Device-owned live K/V length then selects direct publication for the
     * short-prefix window or canonical context summaries plus reduction beyond
     * it. If context has no economical window within the captured capacity, the
     * geometry policy selects the ordinary query-sequence transaction and
     * allocates no summary arena. This is the resolved physical policy, not a
     * runtime failure recovery, and no host-observed live scalar participates.
     *
     * Explicit query or context policies are honored exactly. Invalid geometry,
     * unsupported head dimensions, unknown enum values, and workspace-size
     * overflow return an invalid plan instead of changing modes.
     *
     * @param geometry Immutable graph and device geometry.
     * @return Concrete capture plan and exact persistent workspace sizes.
     */
    [[nodiscard]] inline constexpr FA2PrefillParallelPlan
    selectFA2PrefillParallelPlan(
        const FA2PrefillParallelGeometry &geometry)
    {
        const FA2QueryPartitionGeometry query_geometry{
            .batch_size = geometry.batch_size,
            .query_rows = geometry.query_rows,
            .local_query_heads = geometry.local_query_heads,
            .head_dim = geometry.head_dim,
            .sm_count = geometry.sm_count,
        };
        const int query_warp_groups =
            selectFA2QueryWarpGroups(query_geometry);
        const std::int64_t query_grid_blocks =
            fa2QueryGridBlocks(query_geometry, query_warp_groups);
        if (geometry.kv_capacity <= 0 || query_warp_groups <= 0 ||
            query_grid_blocks <= 0)
        {
            return {};
        }

        const std::int64_t partition_count_64 =
            (static_cast<std::int64_t>(geometry.kv_capacity) +
             kFA2CanonicalContextPartitionKeys - 1) /
            kFA2CanonicalContextPartitionKeys;
        if (partition_count_64 <= 0 ||
            partition_count_64 > std::numeric_limits<int>::max())
        {
            return {};
        }

        const int max_context_partitions =
            static_cast<int>(partition_count_64);
        const int context_partition_slots =
            selectFA2ContextPartitionSlots(
                query_geometry,
                max_context_partitions);
        if (context_partition_slots <= 0)
            return {};

        FA2PrefillPhysicalMode mode = FA2PrefillPhysicalMode::QuerySequence;
        int device_direct_partition_limit = 0;
        switch (geometry.requested_axis)
        {
        case attention::AttentionPrefillParallelAxis::QuerySequence:
            break;
        case attention::AttentionPrefillParallelAxis::KeyValueContext:
            mode = FA2PrefillPhysicalMode::KeyValueContext;
            break;
        case attention::AttentionPrefillParallelAxis::GeometrySelected:
        {
            if (geometry.query_rows < kFA2ContextPrefillMinimumQueryRows ||
                max_context_partitions <= 1 ||
                context_partition_slots <= 1)
            {
                break;
            }

            device_direct_partition_limit =
                selectFA2DeviceDirectPartitionLimit(
                    query_geometry,
                    query_warp_groups,
                    max_context_partitions);
            if (device_direct_partition_limit < 0)
                return {};
            if (device_direct_partition_limit < max_context_partitions)
                mode = FA2PrefillPhysicalMode::DeviceAdaptive;
            break;
        }
        default:
            return {};
        }
        if (mode == FA2PrefillPhysicalMode::QuerySequence)
            device_direct_partition_limit = 0;

        FA2PrefillParallelPlan plan{
            .mode = mode,
            .query_warp_groups = query_warp_groups,
            .reducer_dimension_warps = 0,
            .context_partition_slots = 0,
            .device_direct_partition_limit = device_direct_partition_limit,
            .max_context_partitions = 0,
            .query_grid_blocks = query_grid_blocks,
            .partial_output_bytes = 0,
            .partial_m_bytes = 0,
            .partial_l_bytes = 0,
            .valid = true,
        };
        if (!plan.usesContextParallelism())
            return plan;

        plan.reducer_dimension_warps =
            selectFA2ReducerDimensionWarps(query_geometry);
        plan.max_context_partitions = max_context_partitions;
        plan.context_partition_slots = context_partition_slots;
        if (plan.reducer_dimension_warps <= 0 ||
            plan.context_partition_slots <= 0)
            return {};

        std::size_t scalar_count = checkedFA2WorkspaceProduct(
            static_cast<std::size_t>(geometry.batch_size),
            static_cast<std::size_t>(geometry.query_rows));
        scalar_count = checkedFA2WorkspaceProduct(
            scalar_count,
            static_cast<std::size_t>(geometry.local_query_heads));
        scalar_count = checkedFA2WorkspaceProduct(
            scalar_count,
            static_cast<std::size_t>(plan.max_context_partitions));
        const std::size_t output_elements = checkedFA2WorkspaceProduct(
            scalar_count,
            static_cast<std::size_t>(geometry.head_dim));
        plan.partial_output_bytes = checkedFA2WorkspaceProduct(
            output_elements,
            sizeof(float));
        plan.partial_m_bytes = checkedFA2WorkspaceProduct(
            scalar_count,
            sizeof(float));
        plan.partial_l_bytes = plan.partial_m_bytes;
        if (plan.partial_output_bytes == 0 || plan.partial_m_bytes == 0)
            return {};
        return plan;
    }

    /**
     * @brief Return the widest useful output-dimension striping for the reducer.
     *
     * One warp owns 32 output dimensions.  Additional warps assigned to the
     * same query-row/head pair therefore expose independent output work without
     * changing the ascending context-partition reduction performed by any
     * output element.  Values are powers of two so a 256-thread block always
     * contains an integral number of row owners.
     *
     * @param head_dim Elements in one attention output head.
     * @return Maximum useful warps per row, or zero when unsupported.
     */
    [[nodiscard]] inline constexpr int maximumFA2ReducerDimensionWarps(
        int head_dim)
    {
        return head_dim <= 0
                   ? 0
               : head_dim <= 32
                   ? 1
               : head_dim <= 64
                   ? 2
               : head_dim <= 128
                   ? 4
               : head_dim <= 256
                   ? 8
                   : 0;
    }

    /**
     * @brief Select deterministic reducer dimension parallelism at capture time.
     *
     * The context reducer has eight warps per block.  With one warp per row it
     * can launch only a handful of blocks for small-M tensor-parallel shards,
     * leaving most SMs idle while every warp walks a long partition list.
     * Assigning more warps to independent dimension stripes increases the block
     * grid without partitioning a single output element's reduction.  The
     * resulting physical schedules are therefore byte-equivalent.
     *
     * Production chooses the narrowest stripe count that supplies two complete
     * block waves, matching the reducer's `__launch_bounds__(256, 2)` contract.
     * If the geometry cannot fill that envelope, the widest useful striping is
     * selected.  An explicit value is accepted only for the compiled power-of-
     * two candidates and exists for isolated parity/performance tournaments.
     *
     * @param geometry Immutable batch/query/head/device capture geometry.
     * @param explicit_dimension_warps Zero for production policy, otherwise
     *        one of 1, 2, 4, or 8 within the head's useful range.
     * @return Selected warps per row, or zero for invalid geometry/override.
     */
    [[nodiscard]] inline constexpr int selectFA2ReducerDimensionWarps(
        const FA2QueryPartitionGeometry &geometry,
        int explicit_dimension_warps)
    {
        constexpr int kReducerWarpsPerBlock = 8;
        constexpr int kTargetBlockWaves = 2;

        const int maximum_warps =
            maximumFA2ReducerDimensionWarps(geometry.head_dim);
        if (maximum_warps == 0 || geometry.batch_size <= 0 ||
            geometry.query_rows <= 0 || geometry.local_query_heads <= 0 ||
            geometry.sm_count <= 0 || explicit_dimension_warps < 0)
        {
            return 0;
        }

        const auto is_compiled_candidate = [&](int warps) constexpr
        {
            return warps > 0 && warps <= maximum_warps &&
                   (warps & (warps - 1)) == 0 &&
                   kReducerWarpsPerBlock % warps == 0;
        };
        if (explicit_dimension_warps > 0)
        {
            return is_compiled_candidate(explicit_dimension_warps)
                       ? explicit_dimension_warps
                       : 0;
        }

        const std::int64_t target_blocks =
            static_cast<std::int64_t>(geometry.sm_count) *
            kTargetBlockWaves;
        for (int warps = 1; warps <= maximum_warps; warps *= 2)
        {
            const int rows_per_block = kReducerWarpsPerBlock / warps;
            const std::int64_t row_blocks =
                (static_cast<std::int64_t>(geometry.query_rows) +
                 rows_per_block - 1) /
                rows_per_block;
            const std::int64_t total_blocks =
                static_cast<std::int64_t>(geometry.batch_size) *
                geometry.local_query_heads * row_blocks;
            if (total_blocks >= target_blocks)
                return warps;
        }
        return maximum_warps;
    }

} // namespace llaminar2::cuda::fa2_policy
