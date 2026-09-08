/**
 * @file CPUFlashAttentionLaunchPolicy.h
 * @brief Typed, cache-derived physical launch policy for CPU FlashAttention2.
 *
 * CPU attention has two independent physical choices:
 *
 * 1. A query-sequence plan assigns one complete `(request, query, head)` row to
 *    one worker. This is the normal prefill plan once the output grid contains
 *    enough independent rows to occupy the physical cores.
 * 2. A key/value-context plan divides an underfilled output row into canonical
 *    contiguous K/V summaries and combines those summaries in ascending span
 *    order. This exposes useful work for M=1 and small-M long-context calls
 *    without changing the logical attention result or relying on nested replay.
 *
 * Tile selection models the bytes live in one *phase* of the fused kernel. The
 * QK phase streams one K row while retaining scores; the P@V phase streams one
 * V row while retaining those scores and the output accumulator. K and V are
 * not simultaneously resident, so the former `K + V + scores` L1-percentage
 * estimate systematically selected tiles that were too narrow. A one-socket
 * tournament on a Cascade Lake Xeon with 1 MiB private L2 established a useful
 * phase-resident band of roughly one sixth of private L2. The rule below
 * normalizes that measured band by the detected private-L2 size and exact
 * storage row width, so it remains total on CPUs and tensor formats that were
 * not present in the tournament.
 *
 * Physical K/V tiling is independent of arithmetic partitioning. Every
 * invocation forms the same canonical 256-row online-softmax summaries and
 * merges them in ascending K/V order; the selected tile only bounds the K or V
 * rows streamed through private cache during each phase. Consequently cache
 * tuning cannot change output bytes. The current installed policy deliberately
 * remains geometry-only until a regime-specific policy is supported by measured
 * evidence, but byte equivalence no longer depends on that restriction.
 *
 * This file contains policy only. It performs no allocation, launches no
 * OpenMP region, and does not inspect model names. Tests and performance
 * tournaments can therefore authenticate every decision without executing a
 * production kernel.
 */

#pragma once

#include "../../attention/AttentionExecutionPolicy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2::cpu::fa2_policy
{
    /** Smallest compiled K/V tile used by the FP32 and quantized row kernels. */
    inline constexpr int kMinimumKVTile = 4;

    /**
     * @brief K/V rows in one deterministic context summary.
     *
     * The span is deliberately independent of worker count. A query-sequence
     * worker and a context-parallel worker therefore create the same summary
     * boundaries, and the ordered reducer can preserve one arithmetic contract
     * as physical core counts change.
     */
    inline constexpr int kCanonicalContextPartitionRows = 256;

    /**
     * Largest K/V tile compatible with one canonical arithmetic summary.
     *
     * A wider tile would cross a 256-row summary boundary. That would make the
     * tile itself an alternative reduction tree instead of a cache-only launch
     * choice, so such a candidate is structurally excluded rather than merely
     * left out of automatic dispatch.
     */
    inline constexpr int kMaximumKVTile = kCanonicalContextPartitionRows;

    /**
     * Largest tile currently admitted by automatic production dispatch.
     *
     * This separate name makes installation intent explicit even though the
     * currently certified maximum is also the arithmetic maximum.
     */
    inline constexpr int kInstalledMaximumKVTile = 256;

    /** Minimum K/V span that can profitably pay for a second context summary. */
    inline constexpr int kMinimumContextParallelKVRows =
        2 * kCanonicalContextPartitionRows;

    /** Candidate tiles emitted by the compiled attention inner loops. */
    inline constexpr std::array<int, 7> kCompiledKVTiles{
        4, 8, 16, 32, 64, 128, 256};

    static_assert(
        std::all_of(
            kCompiledKVTiles.begin(),
            kCompiledKVTiles.end(),
            [](int tile) constexpr
            { return tile >= 4 && tile % 4 == 0; }),
        "CPU FA2 physical tiles must preserve canonical four-row vector groups");

    /**
     * Native K/V codec pair executed by one CPU attention invocation.
     *
     * A pair is more precise than byte width: FP16, BF16, and Q16 can occupy
     * similar storage while executing different conversion or VNNI loops, and
     * TurboQuant permits independent TQ4/TQ8 choices for K and V. `Invalid` is
     * an API poison value so callers cannot accidentally authenticate an
     * unclassified production path.
     */
    enum class CPUFA2KVStoragePair : std::uint8_t
    {
        Invalid,
        FP32,
        FP16,
        BF16,
        Q16_1,
        Q8_1,
        TQ4_TQ4,
        TQ4_TQ8,
        TQ8_TQ4,
        TQ8_TQ8,
    };

    /** Vector implementation selected by the CPU runtime dispatcher. */
    enum class CPUFA2VectorISA : std::uint8_t
    {
        Invalid,
        Scalar,
        AVX2,
        AVX512,
    };

    /** Highest vector ISA enabled while compiling the attention translation unit. */
    enum class CPUFA2CodegenISA : std::uint8_t
    {
        Invalid,
        Scalar,
        AVX2,
        AVX512,
    };

    /** @return True only for a concrete production K/V codec pair. */
    [[nodiscard]] inline constexpr bool isCPUFA2KVStoragePair(
        CPUFA2KVStoragePair pair) noexcept
    {
        return pair >= CPUFA2KVStoragePair::FP32 &&
               pair <= CPUFA2KVStoragePair::TQ8_TQ8;
    }

    /** @return True only for a concrete runtime vector implementation. */
    [[nodiscard]] inline constexpr bool isCPUFA2VectorISA(
        CPUFA2VectorISA isa) noexcept
    {
        return isa >= CPUFA2VectorISA::Scalar &&
               isa <= CPUFA2VectorISA::AVX512;
    }

    /** @return True only for a concrete CPU attention code-generation profile. */
    [[nodiscard]] inline constexpr bool isCPUFA2CodegenISA(
        CPUFA2CodegenISA isa) noexcept
    {
        return isa >= CPUFA2CodegenISA::Scalar &&
               isa <= CPUFA2CodegenISA::AVX512;
    }

    /** @return True when one compiled profile can execute the selected runtime path. */
    [[nodiscard]] inline constexpr bool cpuFA2CodegenSupportsRuntimeISA(
        CPUFA2CodegenISA codegen_isa,
        CPUFA2VectorISA runtime_isa) noexcept
    {
        if (!isCPUFA2CodegenISA(codegen_isa) ||
            !isCPUFA2VectorISA(runtime_isa))
        {
            return false;
        }
        switch (codegen_isa)
        {
        case CPUFA2CodegenISA::Scalar:
            return runtime_isa == CPUFA2VectorISA::Scalar;
        case CPUFA2CodegenISA::AVX2:
            return runtime_isa == CPUFA2VectorISA::Scalar ||
                   runtime_isa == CPUFA2VectorISA::AVX2;
        case CPUFA2CodegenISA::AVX512:
            return true;
        case CPUFA2CodegenISA::Invalid:
            return false;
        }
        return false;
    }

    /**
     * @brief Return the highest ISA enabled for this translation unit.
     *
     * Runtime ISA selection and compiler code generation are independent axes.
     * In particular, an AVX-512 build may deliberately select its AVX2
     * algorithm while surrounding inlined code still reflects AVX-512 code
     * generation. That regime has separately measured cache-tile winners and
     * must not masquerade as an AVX2-only build.
     */
    [[nodiscard]] inline constexpr CPUFA2CodegenISA
    compiledCPUFA2CodegenISA() noexcept
    {
#if defined(LLAMINAR_COMPILED_WITH_AVX512) && LLAMINAR_COMPILED_WITH_AVX512
        return CPUFA2CodegenISA::AVX512;
#elif defined(LLAMINAR_COMPILED_WITH_AVX2) && LLAMINAR_COMPILED_WITH_AVX2
        return CPUFA2CodegenISA::AVX2;
#else
        return CPUFA2CodegenISA::Scalar;
#endif
    }

    /** @return Stable diagnostic name for one native K/V codec pair. */
    [[nodiscard]] inline constexpr const char *cpuFA2KVStoragePairName(
        CPUFA2KVStoragePair pair) noexcept
    {
        switch (pair)
        {
        case CPUFA2KVStoragePair::FP32:
            return "fp32";
        case CPUFA2KVStoragePair::FP16:
            return "fp16";
        case CPUFA2KVStoragePair::BF16:
            return "bf16";
        case CPUFA2KVStoragePair::Q16_1:
            return "q16_1";
        case CPUFA2KVStoragePair::Q8_1:
            return "q8_1";
        case CPUFA2KVStoragePair::TQ4_TQ4:
            return "tq4_tq4";
        case CPUFA2KVStoragePair::TQ4_TQ8:
            return "tq4_tq8";
        case CPUFA2KVStoragePair::TQ8_TQ4:
            return "tq8_tq4";
        case CPUFA2KVStoragePair::TQ8_TQ8:
            return "tq8_tq8";
        case CPUFA2KVStoragePair::Invalid:
            break;
        }
        return "invalid";
    }

    /** @return Stable diagnostic name for one CPU vector implementation. */
    [[nodiscard]] inline constexpr const char *cpuFA2VectorISAName(
        CPUFA2VectorISA isa) noexcept
    {
        switch (isa)
        {
        case CPUFA2VectorISA::Scalar:
            return "scalar";
        case CPUFA2VectorISA::AVX2:
            return "avx2";
        case CPUFA2VectorISA::AVX512:
            return "avx512";
        case CPUFA2VectorISA::Invalid:
            break;
        }
        return "invalid";
    }

    /** @return Stable diagnostic name for one compiler code-generation profile. */
    [[nodiscard]] inline constexpr const char *cpuFA2CodegenISAName(
        CPUFA2CodegenISA isa) noexcept
    {
        switch (isa)
        {
        case CPUFA2CodegenISA::Scalar:
            return "scalar";
        case CPUFA2CodegenISA::AVX2:
            return "avx2";
        case CPUFA2CodegenISA::AVX512:
            return "avx512";
        case CPUFA2CodegenISA::Invalid:
            break;
        }
        return "invalid";
    }

    /** Physical CPU attention implementation selected for one invocation. */
    enum class CPUFA2PhysicalMode : std::uint8_t
    {
        QuerySequence,
        KeyValueContext,
    };

    /**
     * @brief Cache capacities relevant to one CPU worker.
     *
     * L1 and L2 must describe one physical core, not the sum printed by tools
     * such as `lscpu`. Shared L3 is retained for diagnostics and future
     * prefetch policy work; physical tile capacity is intentionally bounded by
     * private cache so a neighboring worker cannot invalidate the assumption.
     */
    struct CPUFA2CacheGeometry
    {
        std::size_t private_l1d_bytes = 0;
        std::size_t private_l2_bytes = 0;
        std::size_t shared_l3_bytes = 0;
        std::size_t cache_line_bytes = 64;

        /** @return True when every cache field needed by dispatch is usable. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return private_l1d_bytes > 0 &&
                   private_l2_bytes >= private_l1d_bytes &&
                   cache_line_bytes > 0;
        }
    };

    /** Exact storage and cache geometry used to choose one K/V tile. */
    struct CPUFA2KVTileGeometry
    {
        CPUFA2KVStoragePair storage_pair = CPUFA2KVStoragePair::Invalid;
        CPUFA2VectorISA vector_isa = CPUFA2VectorISA::Invalid;
        CPUFA2CodegenISA codegen_isa = CPUFA2CodegenISA::Invalid;
        int head_dim = 0;
        int kv_rows = 0; ///< Positive invocation span; currently not a policy feature.
        std::size_t key_head_row_bytes = 0;
        std::size_t value_head_row_bytes = 0;
        CPUFA2CacheGeometry cache{};

        /** @return True when the tile footprint can be calculated exactly. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return isCPUFA2KVStoragePair(storage_pair) &&
                   cpuFA2CodegenSupportsRuntimeISA(codegen_isa, vector_isa) &&
                   head_dim > 0 && kv_rows > 0 &&
                   key_head_row_bytes > 0 && value_head_row_bytes > 0 &&
                   cache.valid();
        }
    };

    /** Stable per-kernel controls resolved before CPU attention executes. */
    struct CPUFA2KernelLaunchPolicy
    {
        /** Zero selects the measured policy; positive values force one candidate. */
        int explicit_kv_tile = 0;

        /** @return True when the override is disabled or names a compiled tile. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return explicit_kv_tile == 0 ||
                   std::find(
                       kCompiledKVTiles.begin(),
                       kCompiledKVTiles.end(),
                       explicit_kv_tile) != kCompiledKVTiles.end();
        }
    };

    /** Three measured Qwen head-dimension bands for one codec/ISA family. */
    struct CPUFA2EmpiricalTileBand
    {
        int through_64 = 0;
        int through_128 = 0;
        int above_128 = 0;

        /** @return Calibrated tile for every positive head dimension. */
        [[nodiscard]] constexpr int select(int head_dim) const noexcept
        {
            if (head_dim <= 0)
                return 0;
            if (head_dim <= 64)
                return through_64;
            if (head_dim <= 128)
                return through_128;
            return above_128;
        }
    };

    /**
     * @brief Return a measured native-codec tile before cache-capacity clamping.
     *
     * These bands come from repeated, balanced production-path tournaments on
     * a 28-core Cascade Lake socket with 1 MiB private L2 per core. Every entry
     * minimizes normalized regret across serial decode, grouped verification,
     * and prefill rather than selecting the winner from one thermally sensitive
     * run. Compiler code generation is an explicit axis because AVX2 algorithm
     * selection inside an AVX-512 translation unit produced materially
     * different winners from an AVX2-only build. The installed table contains
     * no M, K/V-length, model, TP-degree, or worker feature because the present
     * corpus has not certified those additional dimensions. Scalar execution
     * deliberately returns zero so the total cache-derived rule remains its
     * first-class policy.
     */
    [[nodiscard]] inline constexpr int cpuFA2EmpiricalTile(
        CPUFA2KVStoragePair pair,
        CPUFA2VectorISA isa,
        CPUFA2CodegenISA codegen_isa,
        int head_dim) noexcept
    {
        CPUFA2EmpiricalTileBand band{};
        switch (isa)
        {
        case CPUFA2VectorISA::AVX2:
            if (codegen_isa == CPUFA2CodegenISA::AVX512)
            {
                switch (pair)
                {
                case CPUFA2KVStoragePair::Q16_1:
                    return head_dim > 0 && head_dim <= 64 ? 32 : 0;
                case CPUFA2KVStoragePair::TQ4_TQ4:
                    band = {128, 64, 128};
                    break;
                case CPUFA2KVStoragePair::TQ4_TQ8:
                    band = {256, 256, 32};
                    break;
                case CPUFA2KVStoragePair::TQ8_TQ4:
                    band = {128, 16, 32};
                    break;
                case CPUFA2KVStoragePair::TQ8_TQ8:
                    band = {4, 256, 64};
                    break;
                default:
                    return 0;
                }
                break;
            }
            if (codegen_isa != CPUFA2CodegenISA::AVX2)
                return 0;
            switch (pair)
            {
            case CPUFA2KVStoragePair::TQ4_TQ4:
                band = {128, 64, 128};
                break;
            case CPUFA2KVStoragePair::TQ4_TQ8:
                band = {256, 256, 32};
                break;
            case CPUFA2KVStoragePair::TQ8_TQ4:
                band = {128, 128, 128};
                break;
            case CPUFA2KVStoragePair::TQ8_TQ8:
                band = {256, 256, 64};
                break;
            default:
                return 0;
            }
            break;
        case CPUFA2VectorISA::AVX512:
            if (codegen_isa != CPUFA2CodegenISA::AVX512)
                return 0;
            switch (pair)
            {
            case CPUFA2KVStoragePair::FP32:
                band = {256, 128, 256};
                break;
            case CPUFA2KVStoragePair::Q8_1:
                band = {256, 256, 256};
                break;
            case CPUFA2KVStoragePair::TQ4_TQ4:
                band = {256, 64, 128};
                break;
            case CPUFA2KVStoragePair::TQ4_TQ8:
                band = {256, 256, 64};
                break;
            case CPUFA2KVStoragePair::TQ8_TQ4:
                band = {256, 128, 64};
                break;
            case CPUFA2KVStoragePair::TQ8_TQ8:
                band = {128, 128, 128};
                break;
            default:
                return 0;
            }
            break;
        case CPUFA2VectorISA::Scalar:
        case CPUFA2VectorISA::Invalid:
            return 0;
        }
        return band.select(head_dim);
    }

    /** Immutable logical geometry used to choose query or context ownership. */
    struct CPUFA2ParallelGeometry
    {
        int batch_size = 0;
        int query_rows = 0;
        int local_query_heads = 0;
        int kv_rows = 0;
        int physical_workers = 0;
        attention::AttentionPrefillParallelAxis requested_axis =
            attention::AttentionPrefillParallelAxis::QuerySequence;

        /** @return True when all products can be represented and scheduled. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            if (batch_size <= 0 || query_rows <= 0 ||
                local_query_heads <= 0 || kv_rows <= 0 ||
                physical_workers <= 0)
            {
                return false;
            }
            const std::int64_t rows =
                static_cast<std::int64_t>(batch_size) * query_rows *
                local_query_heads;
            return rows > 0 && rows <= std::numeric_limits<int>::max();
        }
    };

    /** Fully resolved CPU attention plan consumed by the kernel scheduler. */
    struct CPUFA2ParallelPlan
    {
        CPUFA2PhysicalMode mode = CPUFA2PhysicalMode::QuerySequence;
        int output_rows = 0;
        int arithmetic_partitions = 1; ///< Canonical 256-row summaries.
        int context_partitions = 1; ///< Physical summary producers per wave.
        int context_partition_rows = kCanonicalContextPartitionRows;
        int partial_slots = 0; ///< Context workspace slots including merged rows.
        bool valid = false;

        /** @return True when phase one emits more than one K/V summary per row. */
        [[nodiscard]] constexpr bool usesContextParallelism() const noexcept
        {
            return valid && mode == CPUFA2PhysicalMode::KeyValueContext &&
                   context_partitions > 1;
        }
    };

    /** @return Stable diagnostic name for a resolved CPU physical mode. */
    [[nodiscard]] inline constexpr const char *cpuFA2PhysicalModeName(
        CPUFA2PhysicalMode mode) noexcept
    {
        switch (mode)
        {
        case CPUFA2PhysicalMode::QuerySequence:
            return "query_sequence";
        case CPUFA2PhysicalMode::KeyValueContext:
            return "key_value_context";
        }
        return "invalid";
    }

    /**
     * @brief Calculate the peak per-worker bytes live during one FA2 tile.
     *
     * The larger of the K and V storage rows represents phase streaming. The
     * FP32 score tile, Q row, output accumulator, and two cache lines of loop
     * metadata remain resident across that stream.
     */
    [[nodiscard]] inline constexpr std::size_t cpuFA2TileWorkingSetBytes(
        const CPUFA2KVTileGeometry &geometry,
        int tile_rows) noexcept
    {
        if (!geometry.valid() || tile_rows <= 0)
            return 0;

        const std::size_t streamed_row_bytes =
            std::max(geometry.key_head_row_bytes,
                     geometry.value_head_row_bytes);
        const std::size_t persistent_vector_bytes =
            static_cast<std::size_t>(geometry.head_dim) * sizeof(float) * 2;
        const std::size_t score_bytes =
            static_cast<std::size_t>(tile_rows) * sizeof(float);
        const std::size_t stream_bytes =
            static_cast<std::size_t>(tile_rows) * streamed_row_bytes;
        return persistent_vector_bytes + score_bytes + stream_bytes +
               2 * geometry.cache.cache_line_bytes;
    }

    /**
     * @brief Select the largest measured-safe tile for detected private L2.
     *
     * `explicit_tile` is an authenticated tournament control. It accepts only
     * a compiled power-of-two candidate and never silently rounds an invalid
     * request. Production selection chooses from the same candidate set using
     * the exact phase footprint and the detected private-L2 capacity.
     */
    [[nodiscard]] inline constexpr int selectCPUFA2KVTile(
        const CPUFA2KVTileGeometry &geometry,
        int explicit_tile = 0) noexcept
    {
        if (!geometry.valid() || explicit_tile < 0)
            return 0;

        const auto compiled = [](int tile) constexpr
        {
            return std::find(kCompiledKVTiles.begin(),
                             kCompiledKVTiles.end(),
                             tile) != kCompiledKVTiles.end();
        };
        if (explicit_tile > 0)
            return compiled(explicit_tile) ? explicit_tile : 0;

        /*
         * The measured capacity band is expressed as a cache ratio only so
         * dispatch scales with private-L2 size. `kv_rows` authenticates
         * invocation totality but does not participate in the current choice; a
         * short final tile is simply bounded by the kernel loop. The scheduler's
         * fixed canonical summaries make physical tile selection byte-invisible.
         */
        const std::size_t budget = geometry.cache.private_l2_bytes / 6;

        int winner = kMinimumKVTile;
        for (const int candidate : kCompiledKVTiles)
        {
            if (candidate > kInstalledMaximumKVTile)
                break;
            const std::size_t footprint =
                cpuFA2TileWorkingSetBytes(geometry, candidate);
            if (footprint == 0 || footprint > budget)
                break;
            winner = candidate;
        }
        const int empirical_tile = cpuFA2EmpiricalTile(
            geometry.storage_pair,
            geometry.vector_isa,
            geometry.codegen_isa,
            geometry.head_dim);
        if (empirical_tile <= 0)
            return winner;

        /*
         * An exact measured overlay supersedes the conservative one-sixth-L2
         * generic budget, but never the physical private-L2 capacity itself.
         * This distinction lets a proven wide tile win without pretending an
         * over-capacity working set is cache resident on a smaller CPU.
         */
        const std::size_t empirical_footprint =
            cpuFA2TileWorkingSetBytes(geometry, empirical_tile);
        return empirical_footprint > 0 &&
                       empirical_footprint <= geometry.cache.private_l2_bytes
                   ? empirical_tile
                   : winner;
    }

    /**
     * @brief Resolve physical row ownership for every positive CPU geometry.
     *
     * Geometry-selected execution enters context mode only when complete output
     * rows cannot occupy the worker team and at least two canonical K/V spans
     * exist. Explicit query/context requests are honored directly. The number
     * of K/V partitions depends on local heads and the physical worker team,
     * never on M or request batch. Serial M=1 and grouped M>1 rows therefore use
     * identical partition boundaries at a fixed thread count, which is required
     * for grouped-verifier batch invariance.
     */
    [[nodiscard]] inline constexpr CPUFA2ParallelPlan
    selectCPUFA2ParallelPlan(const CPUFA2ParallelGeometry &geometry) noexcept
    {
        CPUFA2ParallelPlan plan{};
        if (!geometry.valid())
            return plan;

        plan.output_rows = geometry.batch_size * geometry.query_rows *
                           geometry.local_query_heads;
        const int available_context_spans =
            (geometry.kv_rows + kCanonicalContextPartitionRows - 1) /
            kCanonicalContextPartitionRows;
        const int saturation_partitions =
            (geometry.physical_workers + geometry.local_query_heads - 1) /
            geometry.local_query_heads;
        plan.context_partitions = std::max(
            1,
            std::min(available_context_spans, saturation_partitions));
        plan.arithmetic_partitions = std::max(1, available_context_spans);

        switch (geometry.requested_axis)
        {
        case attention::AttentionPrefillParallelAxis::QuerySequence:
            plan.mode = CPUFA2PhysicalMode::QuerySequence;
            plan.context_partitions = 1;
            break;
        case attention::AttentionPrefillParallelAxis::KeyValueContext:
            plan.mode = CPUFA2PhysicalMode::KeyValueContext;
            break;
        case attention::AttentionPrefillParallelAxis::GeometrySelected:
            plan.mode =
                plan.output_rows < geometry.physical_workers &&
                        geometry.kv_rows >= kMinimumContextParallelKVRows &&
                        plan.context_partitions > 1
                    ? CPUFA2PhysicalMode::KeyValueContext
                    : CPUFA2PhysicalMode::QuerySequence;
            if (plan.mode == CPUFA2PhysicalMode::QuerySequence)
                plan.context_partitions = 1;
            break;
        default:
            return {};
        }

        const int slots_per_row =
            plan.mode == CPUFA2PhysicalMode::KeyValueContext
                ? plan.context_partitions + 1
                : 1;
        const std::int64_t partial_slots =
            static_cast<std::int64_t>(plan.output_rows) * slots_per_row;
        if (partial_slots <= 0 ||
            partial_slots > std::numeric_limits<int>::max())
        {
            return {};
        }
        plan.partial_slots = static_cast<int>(partial_slots);
        plan.valid = true;
        return plan;
    }

} // namespace llaminar2::cpu::fa2_policy
