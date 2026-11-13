/**
 * @file ROCmFlashAttentionLaunchPolicy.h
 * @brief Capture-time ROCm FA2 query/context parallel launch policy.
 *
 * ROCm Flash Attention has two byte-equivalent physical schedules. Query-
 * sequence execution assigns a complete K/V scan to each query tile. Context-
 * parallel execution assigns fixed 256-key summaries to a bounded persistent
 * grid and merges those summaries in ascending key order. This header resolves
 * that choice solely from immutable graph geometry and reports the exact arena
 * capacity required by the selected graph. Query score ownership is likewise
 * immutable: the measured native-half M512 XOR tile keeps scores in registers
 * while preserving the same canonical FP32 arithmetic and KV state bytes.
 *
 * Cached physical CU/LDS limits bind planning and the actual launch selector
 * before capture; neither may replace the physical limit with a chip default. The installed
 * ordered wave64 program remains distinct from hardware native-wave preference.
 * Live K/V length is deliberately absent from every policy input. HIP graph
 * replay reads that value from device-owned attention parameters. A context
 * graph therefore remains valid as a request grows, restores a prefix, or
 * reuses a captured bucket, without host dispatch or graph recapture.
 */

#pragma once

#include "../../attention/AttentionExecutionPolicy.h"
#include "kernels/common/ROCmCompiledWaveContract.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "../../../tensors/TensorType.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2::rocm::fa2_policy
{
    /** Threads in the MI50 prefill CTA. */
    inline constexpr int kROCmFA2ThreadsPerBlock = 256;

    /** Compiled ordered-reduction program width; independent of native hardware preference. */
    inline constexpr int kROCmFA2WavefrontSize = static_cast<int>(rocmOrderedWaveWidth);

    /**
     * @brief Select the physical K/V buffers for an exact compiled tile.
     *
     * The 256-wide 16x16 tile has one pipeline stage. Its query scan keeps
     * distinct K/V planes, while its context producer reuses one plane after QK. Narrower heads and generic tile geometries retain their
     * two-buffer schedule. The arithmetic tile and 256-key reduction boundary
     * remain unchanged; only storage lifetime and cooperative copy width vary.
     *
     * @param head_dim Width of one attention head, or zero for a generic kernel.
     * @param query_rows Physical query tile height, or zero for a generic kernel.
     * @param kv_rows Physical K/V tile height, or zero for a generic kernel.
     * @return Exactly one or two K/V buffer pairs.
     */
    [[nodiscard]]
#if defined(__HIPCC__)
    __host__ __device__
#endif
    inline constexpr int rocmFA2KVStages(
        int head_dim, int query_rows, int kv_rows) noexcept
    {
        return head_dim == 256 && query_rows == 16 && kv_rows == 16 ? 1 : 2;
    }

    /** Keep every compiled Q/K/V LDS row aligned for 128-bit reads and writes. */
    inline constexpr int kROCmFA2QKVLDSPad = 8;

    /** FP32 elements appended to each score LDS row. */
    inline constexpr int kROCmFA2ScoreLDSPad = 1;

    /** MI50's per-workgroup LDS capacity. */
    inline constexpr std::size_t kROCmFA2LDSCapacityBytes = 64U * 1024U;

    /**
     * @brief Keys in one immutable online-softmax arithmetic partition.
     *
     * Both query and context schedules close and merge this exact interval.
     * Physical K/V tiles are divisors of this width, so changing tile geometry
     * or persistent-slot ownership cannot change the reduction tree.
     */
    inline constexpr int kROCmFA2CanonicalContextPartitionKeys = 256;

    /**
     * @brief Initial productive phase-grid envelope expressed in CU waves.
     *
     * This is the query-grid occupancy threshold for selecting context mode.
     * Physical context residency is planned independently below, so capacity
     * partitions never become a captured grid of empty workgroups.
     */
    inline constexpr int kROCmFA2ContextTargetBlockWaves = 8;

    /**
     * @brief Resident context-phase workgroups assigned to each MI50 CU.
     *
     * Query scans and other distinct-plane kernels retain the measured two-
     * group register envelope. The wide context producer proves three groups;
     * exact LDS capacity independently bounds both physical schedules.
     */
    inline constexpr int kROCmFA2ContextPhaseMaximumBlocksPerCU = 2;

    /** Near-full single-wave occupancy required for device-direct merging. */
    inline constexpr int kROCmFA2DirectWaveOccupancyNumerator = 9;
    inline constexpr int kROCmFA2DirectWaveOccupancyDenominator = 10;

    /** Canonical partitions in the measured medium-context direct regime. */
    inline constexpr int kROCmFA2DirectMediumContextPartitions = 32;

    /**
     * @brief Immutable physical properties required by ROCm FA2 planning.
     *
     * HIP owns discovery of these values, but launch-policy consumers should
     * not need to import HIP runtime types. Keeping this small typed record at
     * the policy boundary also prevents a heterogeneous host translation unit
     * from including both CUDA and HIP vector-type headers.
     */
    struct ROCmFA2PhysicalDeviceProperties
    {
        int compute_unit_count = 0; ///< CUs visible on the selected HIP device.
        std::size_t lds_capacity_bytes = 0; ///< LDS bytes available per workgroup.
    };

    /**
     * @brief Bind admitted physical limits to the installed fixed-wave FA2 program.
     * @param hardware Immutable visible-device record, independent of context generation.
     * @return Exact CU/LDS inputs used to choose the pre-capture launch variant.
     * @throws std::invalid_argument If this hardware cannot launch the installed CTA.
     */
    [[nodiscard]] inline ROCmFA2PhysicalDeviceProperties bindROCmFA2Hardware(
        const ROCmDeviceHardware &hardware)
    {
        if (hardware.maximum_threads_per_block < kROCmFA2ThreadsPerBlock ||
            hardware.compute_units <= 0 || hardware.lds_per_block == 0)
            throw std::invalid_argument("ROCm FA2 hardware cannot launch the compiled ordered CTA");
        return {.compute_unit_count = hardware.compute_units,
                .lds_capacity_bytes = hardware.lds_per_block};
    }


    /** Immutable visibility facts that select an ordinary query implementation. */
    using ROCmFA2VisibilityPolicy = attention::AttentionVisibilityPolicy;

    /** Physical authority for the QK score consumed by its P/V lane. */
    enum class ROCmFA2ScoreStorage : std::uint8_t
    {
        SharedMatrix, ///< Canonical padded LDS score matrix.
        RegisterOwnedXorTile, ///< Native-half Q/K tiles; score remains in its lane.
    };

    /** Setup-owned query facts; live KV length never selects a physical layout. */
    struct ROCmFA2QueryScoreGeometry
    {
        int batch_size = 0; ///< Captured requests.
        int query_rows = 0; ///< Captured physical query rows, including bucket padding.
        int head_dim = 0; ///< Immutable head width.
        TensorType kv_format = TensorType::FP32; ///< Actual native execution operands.
        ROCmFA2VisibilityPolicy visibility = ROCmFA2VisibilityPolicy::General;
    };

    /**
     * @brief Select the measured exact score owner solely from immutable inputs.
     * @param geometry Native operand, visibility and captured query geometry.
     * @return One first-class score layout; all other shapes retain their canonical layout.
     *
     * The native-half body preserves the four FP32 QK chains, ascending
     * FP32 P/V operations and 256-key merge order. Its XOR Q/K layout removes
     * score LDS traffic and admits four spill-free workgroups per gfx906 CU.
     * Larger buckets add independent sixteen-query tiles, not per-block
     * storage. M512 is the minimum qualified bucket, never an upper bound.
     */
    [[nodiscard]]
#if defined(__HIPCC__)
    __host__ __device__
#endif
    inline constexpr ROCmFA2ScoreStorage selectROCmFA2QueryScoreStorage(
        const ROCmFA2QueryScoreGeometry &geometry) noexcept
    {
        return geometry.batch_size == 1 && geometry.query_rows >= 512 &&
                       geometry.head_dim == 256 && geometry.kv_format == TensorType::FP16 &&
                       geometry.visibility == ROCmFA2VisibilityPolicy::CausalFullContext
                   ? ROCmFA2ScoreStorage::RegisterOwnedXorTile
                   : ROCmFA2ScoreStorage::SharedMatrix;
    }

    /** @return A stable capture-inventory spelling for the physical score owner. */
    [[nodiscard]] inline constexpr const char *rocmFA2ScoreStorageName(
        ROCmFA2ScoreStorage storage) noexcept
    {
        switch (storage)
        {
        case ROCmFA2ScoreStorage::SharedMatrix: return "shared_matrix";
        case ROCmFA2ScoreStorage::RegisterOwnedXorTile: return "register_owned_xor_tile";
        }
        return "invalid";
    }

    /** Immutable launch geometry for one participant-local prefill graph. */
    struct ROCmFA2PrefillParallelGeometry
    {
        int batch_size = 0;        ///< Captured request count.
        int query_rows = 0;        ///< Physical query rows in the graph bucket.
        int local_query_heads = 0; ///< Query heads owned by this participant.
        int head_dim = 0;          ///< Elements in one query/key/value head.
        int kv_capacity = 0;       ///< Stable physical K/V row capacity.
        int compute_unit_count = 0; ///< CUs visible on the selected HIP device.
        std::size_t lds_capacity_bytes = kROCmFA2LDSCapacityBytes;
        attention::AttentionPrefillParallelAxis requested_axis =
            attention::AttentionPrefillParallelAxis::QuerySequence;
        TensorType kv_format = TensorType::FP32; ///< Actual native execution operands.
        ROCmFA2VisibilityPolicy visibility = ROCmFA2VisibilityPolicy::General;
    };

    /** Concrete prefill axis embedded in one HIP graph. */
    enum class ROCmFA2PrefillPhysicalMode : std::uint8_t
    {
        QuerySequence,
        KeyValueContext,
    };

    /** Fully resolved direct-kernel tile geometry. */
    struct ROCmFA2TilePlan
    {
        int query_rows = 0;       ///< Query rows owned by one workgroup.
        int kv_rows = 0;          ///< Physical K/V rows staged per iteration.
        int threads_per_row = 0;  ///< Output-dimension owners per query row.
        std::size_t lds_bytes = 0; ///< Exact dynamic LDS allocation.
        ROCmFA2PrefillPhysicalMode schedule =
            ROCmFA2PrefillPhysicalMode::QuerySequence; ///< Physical reader lifetime.
        ROCmFA2ScoreStorage score_storage = ROCmFA2ScoreStorage::SharedMatrix;
        bool valid = false;       ///< True only for a compiled specialization.
    };

    /**
     * @brief One complete capture-stable ROCm prefill transaction plan.
     *
     * Context plans reserve `[batch, query, head, partition, dimension]`
     * unnormalized output summaries and matching `(m, l)` scalar summaries.
     * Query plans reserve no context workspace. An invalid plan is a fatal
     * capture result; callers must not substitute another physical mode.
     */
    struct ROCmFA2PrefillParallelPlan
    {
        ROCmFA2PrefillPhysicalMode mode =
            ROCmFA2PrefillPhysicalMode::QuerySequence;
        ROCmFA2TilePlan tile{};
        int context_partition_slots = 0;
        int context_phase_block_slots = 0;
        int device_direct_partition_limit = 0;
        int reducer_dimension_wavefronts = 0;
        int reducer_block_slots = 0;
        int max_context_partitions = 0;
        std::int64_t query_grid_blocks = 0;
        std::size_t partial_output_bytes = 0;
        std::size_t partial_m_bytes = 0;
        std::size_t partial_l_bytes = 0;
        bool valid = false;

        /** @return True when phase summaries and a reducer are captured. */
        [[nodiscard]] constexpr bool usesContextParallelism() const noexcept
        {
            return valid && mode == ROCmFA2PrefillPhysicalMode::KeyValueContext;
        }
    };

    /** Return a stable diagnostic spelling for PerfStats and logs. */
    [[nodiscard]] inline constexpr const char *rocmFA2PrefillPhysicalModeName(
        ROCmFA2PrefillPhysicalMode mode) noexcept
    {
        switch (mode)
        {
        case ROCmFA2PrefillPhysicalMode::QuerySequence:
            return "query_sequence";
        case ROCmFA2PrefillPhysicalMode::KeyValueContext:
            return "key_value_context";
        }
        return "invalid";
    }

    /** Multiply positive workspace cardinalities without wrapping `size_t`. */
    [[nodiscard]] inline constexpr std::size_t checkedROCmFA2Product(
        std::size_t lhs,
        std::size_t rhs) noexcept
    {
        return lhs == 0 || rhs == 0 ||
                       lhs > std::numeric_limits<std::size_t>::max() / rhs
                   ? 0
                   : lhs * rhs;
    }

    /**
     * @brief Count simultaneously live K/V planes for the physical schedule.
     * @param head_dim Compiled head width, or zero for generic geometry.
     * @param query_rows Compiled query tile height, or zero for generic geometry.
     * @param kv_rows Compiled K/V tile height, or zero for generic geometry.
     * @param schedule Physical query scan or context-summary producer.
     * @param score_storage Immutable shared-matrix or register-owned layout.
     * @return One shared plane for the wide context producer, otherwise two.
     *
     * The wide context producer retires every K reader before V overwrites the
     * same plane. Query scans retain concurrent K/V publication because smaller
     * grids cannot amortize the additional overwrite/publication barrier.
     */
    [[nodiscard]]
#if defined(__HIPCC__)
    __host__ __device__
#endif
    inline constexpr int rocmFA2LiveKVPlanes(
        int head_dim, int query_rows, int kv_rows, ROCmFA2PrefillPhysicalMode schedule,
        ROCmFA2ScoreStorage score_storage = ROCmFA2ScoreStorage::SharedMatrix) noexcept
    {
        if (score_storage == ROCmFA2ScoreStorage::RegisterOwnedXorTile)
            return schedule == ROCmFA2PrefillPhysicalMode::QuerySequence &&
                           head_dim == 256 && query_rows == 16 && kv_rows == 16 ? 1 : 0;
        if (score_storage != ROCmFA2ScoreStorage::SharedMatrix)
            return 0;
        if (schedule != ROCmFA2PrefillPhysicalMode::QuerySequence &&
            schedule != ROCmFA2PrefillPhysicalMode::KeyValueContext)
            return 0;
        return schedule == ROCmFA2PrefillPhysicalMode::KeyValueContext && head_dim == 256 &&
               query_rows == 16 && kv_rows == 16 ? 1 : 2;
    }

    /**
     * @brief Calculate the exact LDS bytes for one MI50 prefill tile.
     *
     * The layout is one FP16 Q tile, the schedule-owned FP16 K/V planes, and one
     * FP32 score tile. The formula is shared by policy tests and the HIP launch
     * bridge so a benchmark cannot label a geometry different from the kernel
     * it actually launches.
     * @param head_dim Width of each Q/K/V head.
     * @param query_rows Compiled physical query tile height.
     * @param kv_rows Compiled physical K/V tile height.
     * @param schedule Immutable physical query/context reader lifetime.
     * @param score_storage Actual shared-matrix or register-owned score layout.
     * @return Exact consumed LDS bytes, or zero for an invalid geometry.
     */
    [[nodiscard]] inline constexpr std::size_t rocmFA2LDSBytes(
        int head_dim,
        int query_rows,
        int kv_rows,
        ROCmFA2PrefillPhysicalMode schedule = ROCmFA2PrefillPhysicalMode::QuerySequence,
        ROCmFA2ScoreStorage score_storage = ROCmFA2ScoreStorage::SharedMatrix) noexcept
    {
        if (score_storage == ROCmFA2ScoreStorage::RegisterOwnedXorTile)
        {
            return schedule == ROCmFA2PrefillPhysicalMode::QuerySequence &&
                           head_dim == 256 && query_rows == 16 && kv_rows == 16
                       ? 2U * 16U * 256U * sizeof(std::uint16_t) : 0;
        }
        if (score_storage != ROCmFA2ScoreStorage::SharedMatrix)
            return 0;
        if ((schedule != ROCmFA2PrefillPhysicalMode::QuerySequence &&
             schedule != ROCmFA2PrefillPhysicalMode::KeyValueContext) ||
            head_dim <= 0 || head_dim > 256 || query_rows <= 0 ||
            kv_rows <= 0 ||
            kROCmFA2CanonicalContextPartitionKeys % kv_rows != 0)
        {
            return 0;
        }

        const std::size_t qkv_stride =
            static_cast<std::size_t>(head_dim + kROCmFA2QKVLDSPad);
        const std::size_t score_stride =
            static_cast<std::size_t>(kv_rows + kROCmFA2ScoreLDSPad);
        const std::size_t q_bytes =
            static_cast<std::size_t>(query_rows) * qkv_stride *
            sizeof(std::uint16_t);
        const std::size_t kv_bytes =
            static_cast<std::size_t>(rocmFA2KVStages(head_dim, query_rows, kv_rows)) *
            static_cast<std::size_t>(rocmFA2LiveKVPlanes(head_dim, query_rows, kv_rows, schedule)) *
            static_cast<std::size_t>(kv_rows) * qkv_stride *
            sizeof(std::uint16_t);
        const std::size_t score_bytes =
            static_cast<std::size_t>(query_rows) * score_stride * sizeof(float);
        return q_bytes + kv_bytes + score_bytes;
    }

    /**
     * @brief Select the widest compiled MI50 tile that fits the LDS envelope.
     *
     * Candidate order is deterministic: larger QxKV area wins and equal-area
     * candidates prefer more query reuse. An explicit K/V width is accepted
     * only by the isolated tournament and never causes substitution when it
     * cannot fit.
     * @param head_dim Width of one attention head.
     * @param lds_capacity_bytes Physical per-workgroup LDS limit.
     * @param explicit_kv_rows Exact diagnostic tile width, or zero for selection.
     * @param schedule Physical query/context schedule recorded in the tile.
     * @param score_storage Exact compiled score layout, never silently substituted.
     * @return The admitted tile and exact storage extent, or an invalid plan.
     */
    [[nodiscard]] inline constexpr ROCmFA2TilePlan selectROCmFA2Tile(
        int head_dim,
        std::size_t lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
        int explicit_kv_rows = 0,
        ROCmFA2PrefillPhysicalMode schedule = ROCmFA2PrefillPhysicalMode::QuerySequence,
        ROCmFA2ScoreStorage score_storage = ROCmFA2ScoreStorage::SharedMatrix) noexcept
    {
        if (head_dim <= 0 || head_dim > 256 || lds_capacity_bytes == 0 ||
            explicit_kv_rows < 0)
        {
            return {};
        }

        if (score_storage == ROCmFA2ScoreStorage::RegisterOwnedXorTile)
        {
            const std::size_t bytes = rocmFA2LDSBytes(head_dim, 16, 16, schedule, score_storage);
            if (bytes == 0 || bytes > lds_capacity_bytes ||
                (explicit_kv_rows != 0 && explicit_kv_rows != 16))
                return {};
            return {.query_rows = 16, .kv_rows = 16, .threads_per_row = 16,
                    .lds_bytes = bytes, .schedule = schedule,
                    .score_storage = score_storage, .valid = true};
        }
        if (score_storage != ROCmFA2ScoreStorage::SharedMatrix)
            return {};

        struct Candidate
        {
            int query_rows;
            int kv_rows;
        };
        constexpr Candidate candidates[] = {
            {64, 64}, {64, 32}, {32, 64}, {32, 32}, {32, 16},
            {16, 32}, {16, 16}, {8, 16}, {8, 8},
        };

        const int threads_per_row = head_dim <= 128 ? 4 : 16;
        const int maximum_query_rows =
            kROCmFA2ThreadsPerBlock / threads_per_row;
        ROCmFA2TilePlan winner{};
        int winner_area = -1;
        for (const Candidate candidate : candidates)
        {
            if (candidate.query_rows > maximum_query_rows ||
                (explicit_kv_rows > 0 &&
                 candidate.kv_rows != explicit_kv_rows))
            {
                continue;
            }

            const std::size_t bytes = rocmFA2LDSBytes(
                head_dim,
                candidate.query_rows,
                candidate.kv_rows, schedule);
            if (bytes == 0 || bytes > lds_capacity_bytes)
                continue;

            const int area = candidate.query_rows * candidate.kv_rows;
            if (area > winner_area ||
                (area == winner_area &&
                 candidate.query_rows > winner.query_rows))
            {
                winner = {
                    .query_rows = candidate.query_rows,
                    .kv_rows = candidate.kv_rows,
                    .threads_per_row = threads_per_row,
                    .lds_bytes = bytes,
                    .schedule = schedule,
                    .valid = true,
                };
                winner_area = area;
            }
        }
        return winner;
    }

    /**
     * @brief Select an existing compiled tile from admitted physical LDS limits.
     * @param hardware Exact immutable participant record admitted before capture.
     * @param head_dim Width of one attention head.
     * @param explicit_kv_rows Exact diagnostic tile width, or zero for selection.
     * @param schedule Physical query/context schedule, including its K/V lifetime.
     * @param score_storage Existing compiled shared/register score layout.
     * @return The same canonical tile selection used by planning, or an invalid tile.
     * @throws std::invalid_argument If the admitted hardware cannot launch this CTA.
     *
     * This host-only launch binding owns no hardware query or resource cache.
     * Context transactions and their guarded direct launch share the physical
     * limit while selecting their own existing schedule-specific storage layout.
     */
    [[nodiscard]] inline ROCmFA2TilePlan selectROCmFA2TileForHardware(
        const ROCmDeviceHardware &hardware,
        int head_dim,
        int explicit_kv_rows = 0,
        ROCmFA2PrefillPhysicalMode schedule = ROCmFA2PrefillPhysicalMode::QuerySequence,
        ROCmFA2ScoreStorage score_storage = ROCmFA2ScoreStorage::SharedMatrix)
    {
        const auto physical = bindROCmFA2Hardware(hardware);
        return selectROCmFA2Tile(head_dim, physical.lds_capacity_bytes,
                                explicit_kv_rows, schedule, score_storage);
    }

    /** Return the participant-local direct query grid cardinality. */
    [[nodiscard]] inline constexpr std::int64_t rocmFA2QueryGridBlocks(
        const ROCmFA2PrefillParallelGeometry &geometry,
        const ROCmFA2TilePlan &tile) noexcept
    {
        if (!tile.valid || geometry.batch_size <= 0 ||
            geometry.query_rows <= 0 || geometry.local_query_heads <= 0)
        {
            return 0;
        }
        const std::int64_t query_tiles =
            (static_cast<std::int64_t>(geometry.query_rows) +
             tile.query_rows - 1) /
            tile.query_rows;
        return static_cast<std::int64_t>(geometry.batch_size) *
               geometry.local_query_heads * query_tiles;
    }

    /**
     * @brief Select logical K/V-partition owners per query/head tile.
     *
     * Every captured capacity partition receives an independent arithmetic
     * owner. Replay activates only the device-owned live prefix, and the
     * separate CU-sized physical grid strides those logical tasks. This
     * provides short/medium-context direct-merge economy and full long-context
     * partition parallelism in the same graph. Explicit values remain available
     * only for isolated tournament and byte-equivalence coverage.
     */
    [[nodiscard]] inline constexpr int selectROCmFA2ContextPartitionSlots(
        const ROCmFA2PrefillParallelGeometry &geometry,
        const ROCmFA2TilePlan &tile,
        int max_context_partitions,
        int explicit_slots = 0) noexcept
    {
        if (geometry.compute_unit_count <= 0 || max_context_partitions <= 0 ||
            rocmFA2QueryGridBlocks(geometry, tile) <= 0 ||
            explicit_slots < 0)
        {
            return 0;
        }
        if (explicit_slots > 0)
            return explicit_slots <= max_context_partitions
                       ? explicit_slots
                       : 0;

        return max_context_partitions;
    }

    /**
     * @brief Count logical phase workgroups represented by a context graph.
     *
     * A partition slot is an arithmetic owner that strides canonical K/V
     * summaries. It need not own a distinct physical workgroup: persistent
     * phase blocks may execute several logical owners in a deterministic
     * strided order while retaining the exact same summary indices.
     */
    [[nodiscard]] inline constexpr std::int64_t
    rocmFA2ContextPhaseLogicalBlocks(
        const ROCmFA2PrefillParallelGeometry &geometry,
        const ROCmFA2TilePlan &tile,
        int context_partition_slots) noexcept
    {
        const std::int64_t query_blocks =
            rocmFA2QueryGridBlocks(geometry, tile);
        if (query_blocks <= 0 || context_partition_slots <= 0 ||
            query_blocks >
                std::numeric_limits<std::int64_t>::max() /
                    context_partition_slots)
        {
            return 0;
        }
        return query_blocks * context_partition_slots;
    }

    /**
     * @brief Select the bounded physical grid for persistent phase execution.
     *
     * The default exposes the LDS-resident workgroups per CU and lets each block
     * stride the immutable logical partition owners. Explicit values exist for
     * the isolated tournament and byte-equivalence sweep only; they must cover
     * at least one task and cannot exceed the logical task space.
     */
    [[nodiscard]] inline constexpr int selectROCmFA2ContextPhaseBlockSlots(
        const ROCmFA2PrefillParallelGeometry &geometry,
        const ROCmFA2TilePlan &tile,
        int context_partition_slots,
        int explicit_block_slots = 0) noexcept
    {
        if (geometry.compute_unit_count <= 0 || explicit_block_slots < 0)
            return 0;
        const std::int64_t logical_blocks =
            rocmFA2ContextPhaseLogicalBlocks(
                geometry,
                tile,
                context_partition_slots);
        if (logical_blocks <= 0 ||
            logical_blocks > std::numeric_limits<int>::max())
        {
            return 0;
        }
        if (explicit_block_slots > 0)
        {
            return explicit_block_slots <= logical_blocks
                       ? explicit_block_slots
                       : 0;
        }

        if (!tile.valid || tile.lds_bytes == 0 ||
            tile.lds_bytes > geometry.lds_capacity_bytes)
        {
            return 0;
        }
        const auto lds_residency = geometry.lds_capacity_bytes / tile.lds_bytes;
        // Only the one-plane 256-wide kernel has a measured three-group
        // register envelope. Other compiled geometries retain their own cap.
        const auto register_residency = rocmFA2LiveKVPlanes(
            geometry.head_dim, tile.query_rows, tile.kv_rows, tile.schedule) == 1
                ? 3 : kROCmFA2ContextPhaseMaximumBlocksPerCU;
        const auto resident_blocks_per_cu =
            lds_residency < register_residency ? lds_residency : register_residency;
        const std::int64_t resident_blocks =
            static_cast<std::int64_t>(geometry.compute_unit_count) *
            resident_blocks_per_cu;
        return static_cast<int>(
            logical_blocks < resident_blocks
                ? logical_blocks
                : resident_blocks);
    }

    /** Maximum useful 64-dimension reducer stripes for one output row. */
    [[nodiscard]] inline constexpr int maximumROCmFA2ReducerWavefronts(
        int head_dim) noexcept
    {
        return head_dim <= 0
                   ? 0
               : head_dim <= 64
                   ? 1
               : head_dim <= 128
                   ? 2
               : head_dim <= 256
                   ? 4
                   : 0;
    }

    /**
     * @brief Select deterministic output-dimension striping for the reducer.
     *
     * Four wavefronts occupy each 256-thread reducer workgroup. The narrowest
     * compiled stripe count that provides two complete CU waves is selected;
     * if the geometry remains smaller, the widest useful stripe count exposes
     * all independent output work without partitioning any element's reduction.
     */
    [[nodiscard]] inline constexpr int selectROCmFA2ReducerWavefronts(
        const ROCmFA2PrefillParallelGeometry &geometry,
        int explicit_wavefronts = 0) noexcept
    {
        constexpr int wavefronts_per_block =
            kROCmFA2ThreadsPerBlock / kROCmFA2WavefrontSize;
        constexpr int target_block_waves = 2;
        const int maximum_wavefronts =
            maximumROCmFA2ReducerWavefronts(geometry.head_dim);
        if (maximum_wavefronts == 0 || geometry.batch_size <= 0 ||
            geometry.query_rows <= 0 || geometry.local_query_heads <= 0 ||
            geometry.compute_unit_count <= 0 || explicit_wavefronts < 0)
        {
            return 0;
        }

        const auto compiled = [&](int wavefronts) constexpr
        {
            return wavefronts > 0 && wavefronts <= maximum_wavefronts &&
                   (wavefronts & (wavefronts - 1)) == 0 &&
                   wavefronts_per_block % wavefronts == 0;
        };
        if (explicit_wavefronts > 0)
            return compiled(explicit_wavefronts) ? explicit_wavefronts : 0;

        const std::int64_t target_blocks =
            static_cast<std::int64_t>(geometry.compute_unit_count) *
            target_block_waves;
        for (int wavefronts = 1;
             wavefronts <= maximum_wavefronts;
             wavefronts *= 2)
        {
            const int rows_per_block =
                wavefronts_per_block / wavefronts;
            const std::int64_t row_blocks =
                (static_cast<std::int64_t>(geometry.query_rows) +
                 rows_per_block - 1) /
                rows_per_block;
            const std::int64_t blocks =
                static_cast<std::int64_t>(geometry.batch_size) *
                geometry.local_query_heads * row_blocks;
            if (blocks >= target_blocks)
                return wavefronts;
        }
        return maximum_wavefronts;
    }

    /**
     * @brief Count independent row/head reductions in one captured graph.
     *
     * The reducer assigns four physical wavefronts to each workgroup. Increasing
     * dimension striping therefore decreases the number of query rows owned by
     * a logical workgroup. This count describes arithmetic ownership only; a
     * bounded physical grid may stride over these tasks without changing the
     * ascending partition order within any output element.
     */
    [[nodiscard]] inline constexpr std::int64_t
    rocmFA2ReducerLogicalBlocks(
        const ROCmFA2PrefillParallelGeometry &geometry,
        int reducer_dimension_wavefronts) noexcept
    {
        constexpr int wavefronts_per_block =
            kROCmFA2ThreadsPerBlock / kROCmFA2WavefrontSize;
        if (geometry.batch_size <= 0 || geometry.query_rows <= 0 ||
            geometry.local_query_heads <= 0 ||
            reducer_dimension_wavefronts <= 0 ||
            wavefronts_per_block % reducer_dimension_wavefronts != 0)
        {
            return 0;
        }

        const int rows_per_block =
            wavefronts_per_block / reducer_dimension_wavefronts;
        const std::int64_t row_blocks =
            (static_cast<std::int64_t>(geometry.query_rows) +
             rows_per_block - 1) /
            rows_per_block;
        const std::int64_t participant_blocks =
            static_cast<std::int64_t>(geometry.batch_size) *
            geometry.local_query_heads;
        if (participant_blocks <= 0 ||
            row_blocks >
                std::numeric_limits<std::int64_t>::max() /
                    participant_blocks)
        {
            return 0;
        }
        return participant_blocks * row_blocks;
    }

    /**
     * @brief Select the persistent physical grid for the context reducer.
     *
     * A physical block repeatedly claims logical row/head tasks separated by
     * the fixed grid width. The default currently preserves one physical block
     * per logical task while the all-format MI50 tournament establishes the
     * economical resident envelope. Explicit values are accepted only by that
     * tournament and correctness sweep; they may reduce physical residency but
     * can never exceed or omit the logical task space.
     */
    [[nodiscard]] inline constexpr int selectROCmFA2ReducerBlockSlots(
        const ROCmFA2PrefillParallelGeometry &geometry,
        int reducer_dimension_wavefronts,
        int explicit_block_slots = 0) noexcept
    {
        if (explicit_block_slots < 0)
            return 0;
        const std::int64_t logical_blocks =
            rocmFA2ReducerLogicalBlocks(
                geometry,
                reducer_dimension_wavefronts);
        if (logical_blocks <= 0 ||
            logical_blocks > std::numeric_limits<int>::max())
        {
            return 0;
        }
        if (explicit_block_slots > 0)
        {
            return explicit_block_slots <= logical_blocks
                       ? explicit_block_slots
                       : 0;
        }
        return static_cast<int>(logical_blocks);
    }

    /** @return Whether the direct query grid fills at least 90% of one CU wave. */
    [[nodiscard]] inline constexpr bool rocmFA2HasPackedSingleCUWave(
        const ROCmFA2PrefillParallelGeometry &geometry,
        const ROCmFA2TilePlan &tile) noexcept
    {
        const std::int64_t query_blocks =
            rocmFA2QueryGridBlocks(geometry, tile);
        if (query_blocks <= 0 || geometry.compute_unit_count <= 0)
        {
            return false;
        }

        const std::int64_t compute_units = geometry.compute_unit_count;
        return
            query_blocks <= compute_units &&
            query_blocks * kROCmFA2DirectWaveOccupancyDenominator >=
                compute_units * kROCmFA2DirectWaveOccupancyNumerator;
    }

    /**
     * @brief Select the device-owned direct span inside a context transaction.
     *
     * Underfilled grids execute the exact direct kernel for one canonical
     * partition, avoiding partition-publication overhead at short prefixes.
     * An explicitly requested context transaction with a nearly full CU wave
     * retains the measured 32-partition boundary; geometry-selected production
     * chooses the simpler query transaction for that packed geometry instead.
     * The live comparison is always performed by captured kernels on device.
     */
    [[nodiscard]] inline constexpr int
    selectROCmFA2DeviceDirectPartitionLimit(
        const ROCmFA2PrefillParallelGeometry &geometry,
        const ROCmFA2TilePlan &tile,
        int max_context_partitions) noexcept
    {
        if (max_context_partitions <= 0 ||
            rocmFA2QueryGridBlocks(geometry, tile) <= 0)
        {
            return 0;
        }
        if (!rocmFA2HasPackedSingleCUWave(geometry, tile))
            return 1;
        return max_context_partitions <
                       kROCmFA2DirectMediumContextPartitions
                   ? max_context_partitions
                   : kROCmFA2DirectMediumContextPartitions;
    }

    /**
     * @brief Resolve one complete query/context transaction at capture time.
     *
     * Geometry-selected graphs use context parallelism only when the captured
     * direct-query grid cannot fill the measured persistent phase envelope and
     * the cache can contain more than one canonical partition. Speculative depth
     * is deliberately absent from this decision: M=1 through M=16 are ordinary
     * physical attention geometries and may benefit most from K/V parallelism.
     * Explicit policies are honored exactly. A context graph directly publishes
     * output while the device-live partition count fits its capture-time span,
     * and otherwise publishes summaries for its deterministic reducer. HIP
     * conditional graph nodes and host-side live-length dispatch are unnecessary.
     */
    [[nodiscard]] inline constexpr ROCmFA2PrefillParallelPlan
    selectROCmFA2PrefillParallelPlan(
        const ROCmFA2PrefillParallelGeometry &geometry,
        int explicit_context_slots = 0,
        int explicit_reducer_wavefronts = 0,
        int explicit_reducer_block_slots = 0,
        int explicit_context_phase_block_slots = 0) noexcept
    {
        ROCmFA2TilePlan tile = selectROCmFA2Tile(
            geometry.head_dim,
            geometry.lds_capacity_bytes, 0, ROCmFA2PrefillPhysicalMode::KeyValueContext);
        const std::int64_t query_blocks =
            rocmFA2QueryGridBlocks(geometry, tile);
        if (!tile.valid || query_blocks <= 0 || geometry.kv_capacity <= 0 ||
            geometry.compute_unit_count <= 0)
        {
            return {};
        }

        const std::int64_t partition_count_64 =
            (static_cast<std::int64_t>(geometry.kv_capacity) +
             kROCmFA2CanonicalContextPartitionKeys - 1) /
            kROCmFA2CanonicalContextPartitionKeys;
        if (partition_count_64 <= 0 ||
            partition_count_64 > std::numeric_limits<int>::max())
        {
            return {};
        }
        const int max_partitions = static_cast<int>(partition_count_64);
        const int slots = selectROCmFA2ContextPartitionSlots(
            geometry,
            tile,
            max_partitions,
            explicit_context_slots);
        if (slots <= 0)
            return {};

        ROCmFA2PrefillPhysicalMode mode =
            ROCmFA2PrefillPhysicalMode::QuerySequence;
        switch (geometry.requested_axis)
        {
        case attention::AttentionPrefillParallelAxis::QuerySequence:
            break;
        case attention::AttentionPrefillParallelAxis::KeyValueContext:
            mode = ROCmFA2PrefillPhysicalMode::KeyValueContext;
            break;
        case attention::AttentionPrefillParallelAxis::GeometrySelected:
        {
            std::int64_t context_phase_envelope =
                static_cast<std::int64_t>(geometry.compute_unit_count) *
                kROCmFA2ContextTargetBlockWaves;
            const auto query_storage = selectROCmFA2QueryScoreStorage({
                .batch_size = geometry.batch_size, .query_rows = geometry.query_rows,
                .head_dim = geometry.head_dim, .kv_format = geometry.kv_format,
                .visibility = geometry.visibility});
            if (query_storage == ROCmFA2ScoreStorage::RegisterOwnedXorTile)
            {
                // The register-score body is economical once its independent
                // query grid fills the actual retained context producer grid.
                // The old eight-wave target described the shared-score body;
                // applying it here needlessly selects a slower transaction.
                context_phase_envelope = selectROCmFA2ContextPhaseBlockSlots(
                    geometry, tile, slots);
            }
            if (max_partitions > 1 && slots > 1 &&
                !(query_storage == ROCmFA2ScoreStorage::SharedMatrix &&
                  rocmFA2HasPackedSingleCUWave(geometry, tile)) &&
                query_blocks < context_phase_envelope)
            {
                mode = ROCmFA2PrefillPhysicalMode::KeyValueContext;
            }
            break;
        }
        default:
            return {};
        }

        if (mode == ROCmFA2PrefillPhysicalMode::QuerySequence)
        {
            const auto score_storage = selectROCmFA2QueryScoreStorage({
                .batch_size = geometry.batch_size, .query_rows = geometry.query_rows,
                .head_dim = geometry.head_dim, .kv_format = geometry.kv_format,
                .visibility = geometry.visibility});
            tile = selectROCmFA2Tile(geometry.head_dim, geometry.lds_capacity_bytes,
                                     0, ROCmFA2PrefillPhysicalMode::QuerySequence, score_storage);
            if (!tile.valid)
                return {};
        }

        ROCmFA2PrefillParallelPlan plan{
            .mode = mode,
            .tile = tile,
            .context_partition_slots = 0,
            .context_phase_block_slots = 0,
            .device_direct_partition_limit = 0,
            .reducer_dimension_wavefronts = 0,
            .reducer_block_slots = 0,
            .max_context_partitions = 0,
            .query_grid_blocks = query_blocks,
            .partial_output_bytes = 0,
            .partial_m_bytes = 0,
            .partial_l_bytes = 0,
            .valid = true,
        };
        if (!plan.usesContextParallelism())
            return plan;

        plan.context_partition_slots = slots;
        plan.context_phase_block_slots =
            selectROCmFA2ContextPhaseBlockSlots(
                geometry,
                tile,
                slots,
                explicit_context_phase_block_slots);
        plan.device_direct_partition_limit =
            selectROCmFA2DeviceDirectPartitionLimit(
                geometry,
                tile,
                max_partitions);
        plan.reducer_dimension_wavefronts =
            selectROCmFA2ReducerWavefronts(
                geometry,
                explicit_reducer_wavefronts);
        plan.reducer_block_slots =
            selectROCmFA2ReducerBlockSlots(
                geometry,
                plan.reducer_dimension_wavefronts,
                explicit_reducer_block_slots);
        plan.max_context_partitions = max_partitions;
        if (plan.context_phase_block_slots <= 0 ||
            plan.device_direct_partition_limit <= 0 ||
            plan.reducer_dimension_wavefronts <= 0 ||
            plan.reducer_block_slots <= 0)
            return {};

        std::size_t scalar_count = checkedROCmFA2Product(
            static_cast<std::size_t>(geometry.batch_size),
            static_cast<std::size_t>(geometry.query_rows));
        scalar_count = checkedROCmFA2Product(
            scalar_count,
            static_cast<std::size_t>(geometry.local_query_heads));
        scalar_count = checkedROCmFA2Product(
            scalar_count,
            static_cast<std::size_t>(max_partitions));
        const std::size_t output_elements = checkedROCmFA2Product(
            scalar_count,
            static_cast<std::size_t>(geometry.head_dim));
        plan.partial_output_bytes = checkedROCmFA2Product(
            output_elements,
            sizeof(float));
        plan.partial_m_bytes = checkedROCmFA2Product(
            scalar_count,
            sizeof(float));
        plan.partial_l_bytes = plan.partial_m_bytes;
        if (plan.partial_output_bytes == 0 || plan.partial_m_bytes == 0)
            return {};
        return plan;
    }

    /**
     * @brief Find the largest context-parallel workspace in a graph family.
     *
     * Geometry-selected execution is deliberately non-monotonic in query M:
     * underfilled and medium query grids use K/V-context parallelism, while a
     * sufficiently large query grid returns to the workspace-free direct
     * schedule. Consequently, asking only the largest prefill participant for
     * workspace can miss the true maximum at an intermediate M.
     *
     * This helper resolves the largest context-eligible query-tile count
     * analytically from the same occupancy boundary used by
     * `selectROCmFA2PrefillParallelPlan()`. It then walks downward only across
     * the narrow packed-single-CU exclusion, if necessary. The returned plan
     * is a capacity envelope, not an executable chosen for the caller's live
     * request. Its arithmetic partitioning and byte counts are therefore
     * identical to a real graph captured at that exact M.
     *
     * @param geometry Immutable graph-family geometry. `query_rows` is the
     *        largest M the family may admit, normally the KV-cache capacity.
     * @return Largest valid context plan, or an invalid plan when no M in the
     *         family selects context parallelism.
     */
    [[nodiscard]] inline constexpr ROCmFA2PrefillParallelPlan
    selectROCmFA2GeometrySelectedWorkspaceEnvelope(
        const ROCmFA2PrefillParallelGeometry &geometry) noexcept
    {
        if (geometry.requested_axis !=
                attention::AttentionPrefillParallelAxis::GeometrySelected ||
            geometry.batch_size <= 0 || geometry.query_rows <= 0 ||
            geometry.local_query_heads <= 0 ||
            geometry.compute_unit_count <= 0)
        {
            return {};
        }

        ROCmFA2TilePlan tile = selectROCmFA2Tile(
            geometry.head_dim,
            geometry.lds_capacity_bytes, 0, ROCmFA2PrefillPhysicalMode::KeyValueContext);
        if (!tile.valid)
            return {};

        const std::int64_t blocks_per_query_tile =
            static_cast<std::int64_t>(geometry.batch_size) *
            geometry.local_query_heads;
        const std::int64_t context_selection_ceiling =
            static_cast<std::int64_t>(geometry.compute_unit_count) *
            kROCmFA2ContextTargetBlockWaves;
        if (blocks_per_query_tile <= 0 ||
            context_selection_ceiling <= blocks_per_query_tile)
        {
            return {};
        }

        const std::int64_t maximum_context_query_tiles =
            (context_selection_ceiling - 1) / blocks_per_query_tile;
        if (maximum_context_query_tiles <= 0 ||
            maximum_context_query_tiles >
                std::numeric_limits<int>::max() / tile.query_rows)
        {
            return {};
        }

        const int occupancy_limited_rows =
            static_cast<int>(maximum_context_query_tiles) *
            tile.query_rows;
        int candidate_rows =
            geometry.query_rows < occupancy_limited_rows
                ? geometry.query_rows
                : occupancy_limited_rows;

        const auto query_storage = selectROCmFA2QueryScoreStorage({
            .batch_size = geometry.batch_size, .query_rows = candidate_rows,
            .head_dim = geometry.head_dim, .kv_format = geometry.kv_format,
            .visibility = geometry.visibility});
        if (query_storage == ROCmFA2ScoreStorage::RegisterOwnedXorTile)
        {
            const auto context_slots = selectROCmFA2ContextPartitionSlots(
                geometry, tile,
                static_cast<int>((static_cast<std::int64_t>(geometry.kv_capacity) +
                    kROCmFA2CanonicalContextPartitionKeys - 1) /
                    kROCmFA2CanonicalContextPartitionKeys));
            const std::int64_t phase_blocks = selectROCmFA2ContextPhaseBlockSlots(
                geometry, tile, context_slots);
            if (phase_blocks <= 0)
                return {};
            const std::int64_t fast_context_rows =
                ((phase_blocks - 1) / blocks_per_query_tile) * tile.query_rows;
            // M511 remains a legal shared-score graph. Its workspace may be
            // larger than every fast context graph, even when M512 is direct.
            const int largest_context_rows = static_cast<int>(
                fast_context_rows > 511 ? fast_context_rows : 511);
            if (candidate_rows > largest_context_rows)
                candidate_rows = largest_context_rows;
        }

        while (candidate_rows > 0)
        {
            ROCmFA2PrefillParallelGeometry candidate = geometry;
            candidate.query_rows = candidate_rows;
            const ROCmFA2PrefillParallelPlan plan =
                selectROCmFA2PrefillParallelPlan(candidate);
            if (plan.valid && plan.usesContextParallelism())
                return plan;

            const int candidate_tiles =
                (candidate_rows + tile.query_rows - 1) /
                tile.query_rows;
            if (candidate_tiles <= 1)
                break;
            candidate_rows = (candidate_tiles - 1) * tile.query_rows;
        }
        return {};
    }

} // namespace llaminar2::rocm::fa2_policy
