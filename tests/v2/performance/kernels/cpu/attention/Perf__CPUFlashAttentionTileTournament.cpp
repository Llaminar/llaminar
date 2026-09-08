/**
 * @file Perf__CPUFlashAttentionTileTournament.cpp
 * @brief Production-path CPU FA2 K/V-tile tournament for Qwen geometries.
 *
 * This harness measures every executable CPU K/V tile against native cache
 * tensors rather than an FP32 proxy. The matrix spans:
 *
 * - released Qwen 2.5 and Qwen 3.5/3.6 attention geometries;
 * - participant geometries induced by TP=1, 2, 4, and 8, including replicated
 *   compact-GQA K/V heads when the K/V head count cannot be evenly sharded;
 * - serial decode, grouped MTP verification, and persistent-context prefill;
 * - FP32, FP16, BF16, Q16_1, Q8_1, and every TQ4/TQ8 K/V pairing;
 * - all explicit tile candidates admitted by `CPUFlashAttentionLaunchPolicy`.
 *
 * Candidate selection changes only the typed per-kernel launch policy consumed
 * by the real kernel. Tensor creation, quantization, workspace allocation,
 * first touch, byte-equivalence authentication, and warmup all happen before
 * canonical timing. Every candidate must produce exactly the same output bytes
 * before it is eligible for measurement. Candidate samples are then collected
 * in balanced forward/reverse blocks and scored against the fastest candidate
 * block in the same cycle, so frequency, thermal, cache, and background-load
 * drift cannot be mistaken for a tile win. Raw throughput remains attached to
 * every record.
 * Immediately before each sample, an empty OpenMP team primes the production
 * worker pool; this prevents libgomp wake latency from being attributed to a
 * K/V tile. No conversion shadow, row replay, scalar attention implementation,
 * global environment mutation, or debug-configuration reload participates in a
 * measured launch.
 *
 * The registered CTest runs a compact, all-format representative matrix. Run
 * the exhaustive catalog manually with `LLAMINAR_CPU_FA2_TOURNAMENT_FULL=1`.
 * A profiler can isolate exactly one candidate by combining the filter
 * variables documented by `TournamentFilter` with `perf stat`; unrelated
 * candidates never execute in that process.
 */

#include <gtest/gtest.h>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFlashAttentionLaunchPolicy.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/TQ4Tensor.h"
#include "tensors/TQ8Tensor.h"
#include "tensors/Tensors.h"
#include "utils/CPUFeatures.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <omp.h>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::cpu::fa2_policy;

    /** Native K/V tensor pairing exercised by one production invocation. */
    enum class NativeKVFormat : std::uint8_t
    {
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

    /** @return Stable CSV spelling for one native K/V pairing. */
    [[nodiscard]] constexpr std::string_view formatName(
        NativeKVFormat format) noexcept
    {
        switch (format)
        {
        case NativeKVFormat::FP32:
            return "fp32";
        case NativeKVFormat::FP16:
            return "fp16";
        case NativeKVFormat::BF16:
            return "bf16";
        case NativeKVFormat::Q16_1:
            return "q16_1";
        case NativeKVFormat::Q8_1:
            return "q8_1";
        case NativeKVFormat::TQ4_TQ4:
            return "tq4_tq4";
        case NativeKVFormat::TQ4_TQ8:
            return "tq4_tq8";
        case NativeKVFormat::TQ8_TQ4:
            return "tq8_tq4";
        case NativeKVFormat::TQ8_TQ8:
            return "tq8_tq8";
        }
        return "invalid";
    }

    /** @return Typed production codec pair for one tournament format. */
    [[nodiscard]] constexpr CPUFA2KVStoragePair storagePair(
        NativeKVFormat format) noexcept
    {
        switch (format)
        {
        case NativeKVFormat::FP32:
            return CPUFA2KVStoragePair::FP32;
        case NativeKVFormat::FP16:
            return CPUFA2KVStoragePair::FP16;
        case NativeKVFormat::BF16:
            return CPUFA2KVStoragePair::BF16;
        case NativeKVFormat::Q16_1:
            return CPUFA2KVStoragePair::Q16_1;
        case NativeKVFormat::Q8_1:
            return CPUFA2KVStoragePair::Q8_1;
        case NativeKVFormat::TQ4_TQ4:
            return CPUFA2KVStoragePair::TQ4_TQ4;
        case NativeKVFormat::TQ4_TQ8:
            return CPUFA2KVStoragePair::TQ4_TQ8;
        case NativeKVFormat::TQ8_TQ4:
            return CPUFA2KVStoragePair::TQ8_TQ4;
        case NativeKVFormat::TQ8_TQ8:
            return CPUFA2KVStoragePair::TQ8_TQ8;
        }
        return CPUFA2KVStoragePair::Invalid;
    }

    /** @return Typed policy ISA matching the runtime-dispatched kernel path. */
    [[nodiscard]] CPUFA2VectorISA activePolicyISA() noexcept
    {
        switch (activeISALevel())
        {
        case ISALevel::Scalar:
            return CPUFA2VectorISA::Scalar;
        case ISALevel::AVX2:
            return CPUFA2VectorISA::AVX2;
        case ISALevel::AVX512:
            return CPUFA2VectorISA::AVX512;
        }
        return CPUFA2VectorISA::Invalid;
    }

    inline constexpr std::array<NativeKVFormat, 9> kNativeKVFormats{
        NativeKVFormat::FP32,
        NativeKVFormat::FP16,
        NativeKVFormat::BF16,
        NativeKVFormat::Q16_1,
        NativeKVFormat::Q8_1,
        NativeKVFormat::TQ4_TQ4,
        NativeKVFormat::TQ4_TQ8,
        NativeKVFormat::TQ8_TQ4,
        NativeKVFormat::TQ8_TQ8,
    };

    /** Full-model geometry used only to derive participant-local dimensions. */
    struct ModelAttentionGeometry
    {
        const char *release_label = nullptr;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
    };

    /**
     * Released Qwen geometries relevant to the current dense and MoE graphs.
     * Releases with identical geometry share one row because model identity is
     * intentionally absent from production attention dispatch.
     */
    inline constexpr std::array<ModelAttentionGeometry, 10>
        kQwenAttentionGeometries{{
            {"Qwen2.5-0.5B", 14, 2, 64},
            {"Qwen2.5-1.5B", 12, 2, 128},
            {"Qwen2.5-3B", 16, 2, 128},
            {"Qwen2.5-7B", 28, 4, 128},
            {"Qwen2.5-14B/32B", 40, 8, 128},
            {"Qwen3.5-0.8B/2B", 8, 2, 256},
            {"Qwen3.5-4B/9B", 16, 4, 256},
            {"Qwen3.5/3.6-27B", 24, 4, 256},
            {"Qwen3.5/3.6-35B-A3B", 16, 2, 256},
            {"Qwen3.5-122B-A10B/397B-A17B", 32, 2, 256},
        }};

    /** Exact geometry passed to one participant-local CPU attention kernel. */
    struct ParticipantAttentionGeometry
    {
        std::string sources;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int head_start = 0;
        int replicated_gqa_n_rep = 0;

        /** @return Geometry-only key used to collapse model aliases. */
        [[nodiscard]] auto key() const noexcept
        {
            return std::tuple{
                n_heads,
                n_kv_heads,
                head_dim,
                head_start,
                replicated_gqa_n_rep};
        }
    };

    /**
     * @brief Resolve rank-zero dimensions for one equal tensor split.
     *
     * Query heads are sharded when divisible. K/V heads are sharded when their
     * count is also divisible; otherwise the compact K/V tensor is replicated
     * and the global GQA ratio maps local query heads into that tensor.
     */
    [[nodiscard]] ParticipantAttentionGeometry resolveParticipantGeometry(
        const ModelAttentionGeometry &model,
        int tp_degree)
    {
        if (tp_degree <= 0 || model.n_heads % tp_degree != 0)
            return {};

        ParticipantAttentionGeometry participant{
            .sources = std::string(model.release_label) + "@TP" +
                       std::to_string(tp_degree),
            .n_heads = model.n_heads / tp_degree,
            .n_kv_heads = model.n_kv_heads,
            .head_dim = model.head_dim,
        };
        if (model.n_kv_heads % tp_degree == 0)
        {
            participant.n_kv_heads = model.n_kv_heads / tp_degree;
        }
        else
        {
            participant.replicated_gqa_n_rep =
                model.n_heads / model.n_kv_heads;
        }
        return participant;
    }

    /** @return Every distinct participant geometry induced by TP=1/2/4/8. */
    [[nodiscard]] std::vector<ParticipantAttentionGeometry>
    buildDistinctParticipantGeometries()
    {
        constexpr std::array<int, 4> tp_degrees{1, 2, 4, 8};
        std::vector<ParticipantAttentionGeometry> participants;
        for (const ModelAttentionGeometry &model : kQwenAttentionGeometries)
        {
            for (const int tp_degree : tp_degrees)
            {
                ParticipantAttentionGeometry candidate =
                    resolveParticipantGeometry(model, tp_degree);
                if (candidate.n_heads <= 0)
                    continue;

                const auto existing = std::find_if(
                    participants.begin(),
                    participants.end(),
                    [&](const ParticipantAttentionGeometry &participant)
                    { return participant.key() == candidate.key(); });
                if (existing == participants.end())
                {
                    participants.push_back(std::move(candidate));
                }
                else
                {
                    existing->sources += "|" + candidate.sources;
                }
            }
        }
        return participants;
    }

    /** Execution regime whose K/V tile economics are measured. */
    enum class AttentionRegime : std::uint8_t
    {
        Decode,
        GroupedVerifier,
        Prefill,
    };

    /** @return Stable CSV spelling for one attention regime. */
    [[nodiscard]] constexpr std::string_view regimeName(
        AttentionRegime regime) noexcept
    {
        switch (regime)
        {
        case AttentionRegime::Decode:
            return "decode";
        case AttentionRegime::GroupedVerifier:
            return "grouped_verifier";
        case AttentionRegime::Prefill:
            return "prefill";
        }
        return "invalid";
    }

    /** One production work point. K/V length includes grouped/prefill rows. */
    struct WorkPoint
    {
        AttentionRegime regime = AttentionRegime::Decode;
        int query_rows = 1;
        int kv_len = 1;
    };

    /**
     * Environment filters used both for quick iteration and isolated perf.
     * Zero/empty values mean total coverage for that dimension.
     */
    struct TournamentFilter
    {
        std::string format;
        attention::AttentionPrefillParallelAxis requested_axis =
            attention::AttentionPrefillParallelAxis::GeometrySelected;
        int head_dim = 0;
        int local_heads = 0;
        int local_kv_heads = 0;
        int query_rows = 0;
        int kv_len = 0;
        int tile = 0;
        int samples = 28;
        int warmup = 2;
        bool cold_cache = false;

        /** @return Positive integer environment value or `fallback`. */
        [[nodiscard]] static int envInt(
            const char *name,
            int fallback)
        {
            const char *text = std::getenv(name);
            if (!text || !*text)
                return fallback;
            char *end = nullptr;
            const long value = std::strtol(text, &end, 10);
            if (end == text || value <= 0 ||
                value > std::numeric_limits<int>::max())
            {
                return fallback;
            }
            return static_cast<int>(value);
        }

        /** @return Current process filters. */
        [[nodiscard]] static TournamentFilter fromEnvironment()
        {
            const char *format_text =
                std::getenv("LLAMINAR_CPU_FA2_TOURNAMENT_FORMAT");
            const char *axis_text =
                std::getenv("LLAMINAR_CPU_FA2_TOURNAMENT_AXIS");
            const std::string_view axis = axis_text ? axis_text : "";
            const auto requested_axis = [&]()
            {
                if (axis.empty() || axis == "geometry_selected")
                {
                    return attention::AttentionPrefillParallelAxis::
                        GeometrySelected;
                }
                if (axis == "query_sequence")
                {
                    return attention::AttentionPrefillParallelAxis::
                        QuerySequence;
                }
                if (axis == "key_value_context")
                {
                    return attention::AttentionPrefillParallelAxis::
                        KeyValueContext;
                }
                throw std::invalid_argument(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_AXIS must be "
                    "geometry_selected, query_sequence, or "
                    "key_value_context");
            }();
            return {
                .format = format_text ? format_text : "",
                .requested_axis = requested_axis,
                .head_dim = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_HEAD_DIM", 0),
                .local_heads = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_LOCAL_HEADS", 0),
                .local_kv_heads = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_LOCAL_KV_HEADS", 0),
                .query_rows = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_M", 0),
                .kv_len = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_KV", 0),
                .tile = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_TILE", 0),
                .samples = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_SAMPLES", 28),
                .warmup = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_WARMUP", 2),
                .cold_cache = envInt(
                    "LLAMINAR_CPU_FA2_TOURNAMENT_COLD", 0) != 0,
            };
        }
    };

    /** K/V tensors plus exact bytes streamed for one logical head row. */
    struct NativeKVPair
    {
        std::shared_ptr<ITensor> key;
        std::shared_ptr<ITensor> value;
        std::unique_ptr<TurboQuantContext> turboquant_context;
        std::size_t key_head_row_bytes = 0;
        std::size_t value_head_row_bytes = 0;
    };

    /** Fill setup tensors deterministically without touching timed execution. */
    void fillRandom(std::vector<float> &values, std::uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> distribution(-0.35f, 0.35f);
        for (float &value : values)
            value = distribution(rng);
    }

    /** Rearrange position-major FP32 K/V into production Q16 head-major rows. */
    [[nodiscard]] std::vector<float> toQ16HeadMajor(
        const std::vector<float> &position_major,
        int kv_len,
        int n_kv_heads,
        int head_dim)
    {
        std::vector<float> head_major(position_major.size());
        for (int head = 0; head < n_kv_heads; ++head)
        {
            for (int row = 0; row < kv_len; ++row)
            {
                const float *source =
                    position_major.data() +
                    (static_cast<std::size_t>(row) * n_kv_heads + head) *
                        head_dim;
                float *destination =
                    head_major.data() +
                    (static_cast<std::size_t>(head) * kv_len + row) *
                        head_dim;
                std::copy_n(source, head_dim, destination);
            }
        }
        return head_major;
    }

    /** @return Q16 block size used by the production cache for one head. */
    [[nodiscard]] constexpr Q16BlockSize q16BlockSizeForHead(
        int head_dim) noexcept
    {
        if (head_dim <= 32)
            return Q16BlockSize::BLOCK_32;
        if (head_dim <= 64)
            return Q16BlockSize::BLOCK_64;
        return Q16BlockSize::BLOCK_128;
    }

    /**
     * @brief Build one native-format K/V pair outside the measured region.
     *
     * Q16 uses the real `[kv_head][position][dimension]` cache layout. All
     * other formats retain the production position-major cache tensor. The TQ
     * context remains owned beside its tensors for the entire benchmark case.
     */
    [[nodiscard]] NativeKVPair makeNativeKVPair(
        NativeKVFormat format,
        const std::vector<float> &key_fp32,
        const std::vector<float> &value_fp32,
        int kv_len,
        int n_kv_heads,
        int head_dim)
    {
        const std::size_t kv_columns =
            static_cast<std::size_t>(n_kv_heads) * head_dim;
        const std::vector<std::size_t> position_major_shape{
            static_cast<std::size_t>(kv_len), kv_columns};
        NativeKVPair pair;

        if (format == NativeKVFormat::FP32)
        {
            auto key = std::make_shared<FP32Tensor>(position_major_shape);
            auto value = std::make_shared<FP32Tensor>(position_major_shape);
            std::copy(key_fp32.begin(), key_fp32.end(), key->mutable_data());
            std::copy(value_fp32.begin(), value_fp32.end(), value->mutable_data());
            pair.key = std::move(key);
            pair.value = std::move(value);
            pair.key_head_row_bytes =
                static_cast<std::size_t>(head_dim) * sizeof(float);
            pair.value_head_row_bytes = pair.key_head_row_bytes;
            return pair;
        }

        if (format == NativeKVFormat::FP16)
        {
            std::vector<std::uint16_t> key_data(key_fp32.size());
            std::vector<std::uint16_t> value_data(value_fp32.size());
            std::transform(
                key_fp32.begin(), key_fp32.end(), key_data.begin(),
                [](float value) { return fp32_to_fp16(value); });
            std::transform(
                value_fp32.begin(), value_fp32.end(), value_data.begin(),
                [](float value) { return fp32_to_fp16(value); });
            pair.key = std::make_shared<FP16Tensor>(
                position_major_shape, key_data);
            pair.value = std::make_shared<FP16Tensor>(
                position_major_shape, value_data);
            pair.key_head_row_bytes =
                static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
            pair.value_head_row_bytes = pair.key_head_row_bytes;
            return pair;
        }

        if (format == NativeKVFormat::BF16)
        {
            std::vector<std::uint16_t> key_data(key_fp32.size());
            std::vector<std::uint16_t> value_data(value_fp32.size());
            std::transform(
                key_fp32.begin(), key_fp32.end(), key_data.begin(),
                [](float value) { return simd::fp32_to_bf16(value); });
            std::transform(
                value_fp32.begin(), value_fp32.end(), value_data.begin(),
                [](float value) { return simd::fp32_to_bf16(value); });
            pair.key = std::make_shared<BF16Tensor>(
                position_major_shape, key_data);
            pair.value = std::make_shared<BF16Tensor>(
                position_major_shape, value_data);
            pair.key_head_row_bytes =
                static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
            pair.value_head_row_bytes = pair.key_head_row_bytes;
            return pair;
        }

        if (format == NativeKVFormat::Q16_1)
        {
            constexpr float kCacheScale = 8.0f;
            const Q16BlockSize block_size = q16BlockSizeForHead(head_dim);
            const std::vector<std::size_t> head_major_shape{
                static_cast<std::size_t>(n_kv_heads) * kv_len,
                static_cast<std::size_t>(head_dim)};
            auto key = std::make_shared<Q16_1Tensor>(
                head_major_shape, block_size);
            auto value = std::make_shared<Q16_1Tensor>(
                head_major_shape, block_size);
            const std::vector<float> key_head_major = toQ16HeadMajor(
                key_fp32, kv_len, n_kv_heads, head_dim);
            const std::vector<float> value_head_major = toQ16HeadMajor(
                value_fp32, kv_len, n_kv_heads, head_dim);
            if (!key->copyFrom_fp32_fixed_scale(
                    key_head_major.data(), kCacheScale, head_dim) ||
                !value->copyFrom_fp32_fixed_scale(
                    value_head_major.data(), kCacheScale, head_dim))
            {
                return {};
            }
            pair.key_head_row_bytes =
                key->blocks_per_row() * q16_block_size_bytes(block_size);
            pair.value_head_row_bytes =
                value->blocks_per_row() * q16_block_size_bytes(block_size);
            pair.key = std::move(key);
            pair.value = std::move(value);
            return pair;
        }

        if (format == NativeKVFormat::Q8_1)
        {
            auto key = Q8_1Tensor::quantize_from_fp32(
                key_fp32.data(), position_major_shape);
            auto value = Q8_1Tensor::quantize_from_fp32(
                value_fp32.data(), position_major_shape);
            if (!key || !value)
                return {};
            const std::size_t blocks_per_head =
                (static_cast<std::size_t>(head_dim) +
                 Q8_1Block::BLOCK_SIZE - 1) /
                Q8_1Block::BLOCK_SIZE;
            pair.key_head_row_bytes =
                blocks_per_head * sizeof(Q8_1Block);
            pair.value_head_row_bytes = pair.key_head_row_bytes;
            pair.key = std::move(key);
            pair.value = std::move(value);
            return pair;
        }

        pair.turboquant_context = std::make_unique<TurboQuantContext>(
            head_dim,
            /*rotation_seed=*/42,
            /*projection_seed=*/42);
        const bool key_is_tq4 =
            format == NativeKVFormat::TQ4_TQ4 ||
            format == NativeKVFormat::TQ4_TQ8;
        const bool value_is_tq4 =
            format == NativeKVFormat::TQ4_TQ4 ||
            format == NativeKVFormat::TQ8_TQ4;

        if (key_is_tq4)
        {
            auto key = TQ4Tensor::quantize_from_fp32(
                key_fp32.data(), position_major_shape, head_dim,
                *pair.turboquant_context);
            if (!key)
                return {};
            pair.key_head_row_bytes = key->block_bytes();
            pair.key = std::move(key);
        }
        else
        {
            auto key = TQ8Tensor::quantize_from_fp32(
                key_fp32.data(), position_major_shape, head_dim,
                *pair.turboquant_context);
            if (!key)
                return {};
            pair.key_head_row_bytes = key->block_bytes();
            pair.key = std::move(key);
        }

        if (value_is_tq4)
        {
            auto value = TQ4Tensor::quantize_from_fp32(
                value_fp32.data(), position_major_shape, head_dim,
                *pair.turboquant_context);
            if (!value)
                return {};
            pair.value_head_row_bytes = value->block_bytes();
            pair.value = std::move(value);
        }
        else
        {
            auto value = TQ8Tensor::quantize_from_fp32(
                value_fp32.data(), position_major_shape, head_dim,
                *pair.turboquant_context);
            if (!value)
                return {};
            pair.value_head_row_bytes = value->block_bytes();
            pair.value = std::move(value);
        }
        return pair;
    }

    /** Evict one immutable tensor payload before a cold-cache timing sample. */
    void flushTensor(const ITensor &tensor)
    {
#if defined(__x86_64__) || defined(_M_X64)
        const auto *data = static_cast<const std::byte *>(tensor.raw_data());
        const std::size_t bytes = tensor.size_bytes();
        for (std::size_t offset = 0; offset < bytes; offset += 64)
            _mm_clflush(data + offset);
        _mm_mfence();
#else
        (void)tensor;
#endif
    }

    /** One fully prepared production invocation with persistent workspace. */
    class PreparedAttentionInvocation
    {
    public:
        /**
         * Build and first-touch all input, output, and context-summary storage.
         * No allocation performed here is reachable from `runOnce()`.
         */
        PreparedAttentionInvocation(
            ParticipantAttentionGeometry geometry,
            NativeKVFormat format,
            WorkPoint point,
            attention::AttentionPrefillParallelAxis requested_axis =
                attention::AttentionPrefillParallelAxis::GeometrySelected)
            : geometry_(std::move(geometry)),
              format_(format),
              point_(point),
              requested_axis_(requested_axis),
              query_({static_cast<std::size_t>(point.query_rows),
                      static_cast<std::size_t>(geometry_.n_heads) *
                          geometry_.head_dim}),
              output_({static_cast<std::size_t>(point.query_rows),
                       static_cast<std::size_t>(geometry_.n_heads) *
                           geometry_.head_dim}),
              workspace_(
                  DeviceId::cpu(),
                  workspaceCapacity(
                      point.query_rows,
                      geometry_.n_heads,
                      geometry_.head_dim))
        {
            std::vector<float> query_data(
                static_cast<std::size_t>(point.query_rows) *
                geometry_.n_heads * geometry_.head_dim);
            std::vector<float> key_data(
                static_cast<std::size_t>(point.kv_len) *
                geometry_.n_kv_heads * geometry_.head_dim);
            std::vector<float> value_data(key_data.size());
            fillRandom(query_data, 11001U + point.query_rows);
            fillRandom(key_data, 22001U + point.kv_len);
            fillRandom(value_data, 33001U + point.kv_len);
            std::copy(
                query_data.begin(), query_data.end(), query_.mutable_data());

            kv_ = makeNativeKVPair(
                format,
                key_data,
                value_data,
                point.kv_len,
                geometry_.n_kv_heads,
                geometry_.head_dim);
            if (!kv_.key || !kv_.value)
            {
                error_ = "native K/V construction failed";
                return;
            }

            const WorkspaceRequirements requirements =
                kernel_.getWorkspaceRequirements(
                    point.query_rows,
                    geometry_.n_heads,
                    geometry_.head_dim);
            if (!workspace_.allocate(requirements))
            {
                error_ = "persistent CPU FA2 workspace allocation failed";
                return;
            }
            kernel_.bindWorkspace(&workspace_);
            ready_ = true;
        }

        PreparedAttentionInvocation(const PreparedAttentionInvocation &) = delete;
        PreparedAttentionInvocation &operator=(
            const PreparedAttentionInvocation &) = delete;

        /** @return True when every setup-owned resource is bound. */
        [[nodiscard]] bool ready() const noexcept { return ready_; }

        /** @return Setup failure text when `ready()` is false. */
        [[nodiscard]] const std::string &error() const noexcept
        {
            return error_;
        }

        /** @return Exact key bytes read for one logical head row. */
        [[nodiscard]] std::size_t keyHeadRowBytes() const noexcept
        {
            return kv_.key_head_row_bytes;
        }

        /** @return Exact value bytes read for one logical head row. */
        [[nodiscard]] std::size_t valueHeadRowBytes() const noexcept
        {
            return kv_.value_head_row_bytes;
        }

        /** Install one typed candidate before a synchronous CPU launch. */
        void configureKVTileCandidate(int tile)
        {
            kernel_.configureLaunchPolicy({.explicit_kv_tile = tile});
        }

        /** Run exactly one production kernel invocation. */
        [[nodiscard]] bool runOnce()
        {
            if (!ready_)
                return false;

            if (point_.regime == AttentionRegime::GroupedVerifier)
            {
                return kernel_.compute_verifier_rows_decode_equivalent(
                    &query_,
                    kv_.key.get(),
                    kv_.value.get(),
                    &output_,
                    point_.query_rows,
                    point_.kv_len,
                    geometry_.n_heads,
                    geometry_.n_kv_heads,
                    geometry_.head_dim,
                    /*causal=*/true,
                    /*window_size=*/-1,
                    /*mpi_ctx=*/nullptr,
                    /*device_idx=*/-1,
                    geometry_.head_start,
                    geometry_.replicated_gqa_n_rep,
                    /*kv_logical_view=*/{},
                    {
                        .prefill_parallel_axis =
                            requested_axis_,
                    });
            }

            return kernel_.compute_tensor(
                &query_,
                kv_.key.get(),
                kv_.value.get(),
                &output_,
                /*batch_size=*/1,
                point_.query_rows,
                point_.kv_len,
                geometry_.n_heads,
                geometry_.n_kv_heads,
                geometry_.head_dim,
                /*causal=*/true,
                /*window_size=*/-1,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                /*mpi_ctx=*/nullptr,
                /*device_idx=*/-1,
                geometry_.head_start,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                geometry_.replicated_gqa_n_rep,
                {
                    .prefill_parallel_axis =
                        requested_axis_,
                });
        }

        /** Evict native K/V payloads without including eviction in timing. */
        void flushKV() const
        {
            flushTensor(*kv_.key);
            flushTensor(*kv_.value);
        }

        /** @return True when the final output remains finite. */
        [[nodiscard]] bool outputFinite() const
        {
            const std::size_t elements =
                static_cast<std::size_t>(point_.query_rows) *
                geometry_.n_heads * geometry_.head_dim;
            return std::all_of(
                output_.data(), output_.data() + elements,
                [](float value) { return std::isfinite(value); });
        }

        /**
         * @brief Copy the exact FP32 result representation for authentication.
         *
         * This setup-only method runs outside canonical timing. Comparing bytes,
         * rather than applying a numerical tolerance, proves that cache tiling
         * cannot alter online-softmax boundaries or vector reduction order.
         *
         * @return Complete logical output as an owning byte vector.
         */
        [[nodiscard]] std::vector<std::byte> copyOutputBytes() const
        {
            const std::size_t bytes =
                static_cast<std::size_t>(point_.query_rows) *
                geometry_.n_heads * geometry_.head_dim * sizeof(float);
            const auto *begin = reinterpret_cast<const std::byte *>(
                output_.data());
            return {begin, begin + bytes};
        }

    private:
        /** Calculate a bounded manager capacity before member construction. */
        [[nodiscard]] static std::size_t workspaceCapacity(
            int rows,
            int heads,
            int head_dim)
        {
            CPUFlashAttentionKernelT<ActivationPrecision::FP32> probe;
            const WorkspaceRequirements requirements =
                probe.getWorkspaceRequirements(rows, heads, head_dim);
            return requirements.total_bytes_with_alignment() + 4096;
        }

        ParticipantAttentionGeometry geometry_;
        NativeKVFormat format_;
        WorkPoint point_;
        attention::AttentionPrefillParallelAxis requested_axis_ =
            attention::AttentionPrefillParallelAxis::GeometrySelected;
        FP32Tensor query_;
        FP32Tensor output_;
        NativeKVPair kv_;
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;
        DeviceWorkspaceManager workspace_;
        bool ready_ = false;
        std::string error_;
    };

    /** Robust latency summary for one exact candidate. */
    struct CandidateTiming
    {
        int tile = 0;
        double minimum_us = 0.0;
        double median_us = 0.0;
        double paired_median_ratio = 0.0;
    };

    /**
     * @brief Wake the production OpenMP worker team outside canonical timing.
     *
     * Candidate blocks perform a short serial policy assignment between OpenMP
     * regions. That setup can outlast libgomp's team-end spin interval,
     * especially for sub-millisecond decode kernels. Without this prime, a
     * candidate sample may include several milliseconds of futex wake latency
     * while its neighbor sees an already-running team. Real CPU inference
     * executes parallel stages back to back; an empty region restores that
     * precondition without touching Q, K, V, output, or attention workspace.
     *
     * @return Number of physical workers that entered the region.
     */
    [[nodiscard]] int primeOpenMPWorkerTeam()
    {
        int participants = 0;
#pragma omp parallel reduction(+ : participants)
        {
            participants += 1;
        }
        return participants;
    }

    /**
     * @brief Prove that every physical tile preserves exact result bytes.
     *
     * The first compiled candidate is the byte oracle only for this invariant;
     * it is not a serial implementation or a performance fallback. Every launch
     * executes the same optimized production kernel with a different typed
     * physical cache tile. Any mismatch is fatal and identifies both candidates
     * and the first differing output byte.
     *
     * @param invocation Fully prepared production attention invocation.
     * @param candidates Exact compiled physical tile candidates.
     */
    void verifyCandidateByteEquivalence(
        PreparedAttentionInvocation &invocation,
        const std::vector<int> &candidates)
    {
        if (candidates.empty())
            throw std::invalid_argument("CPU FA2 tournament has no candidates");

        std::vector<std::byte> reference;
        const int reference_tile = candidates.front();
        for (const int tile : candidates)
        {
            invocation.configureKVTileCandidate(tile);
            if (!invocation.runOnce())
            {
                throw std::runtime_error(
                    "CPU FA2 byte-authentication launch failed for tile=" +
                    std::to_string(tile));
            }
            if (!invocation.outputFinite())
            {
                throw std::runtime_error(
                    "CPU FA2 byte authentication produced non-finite output for tile=" +
                    std::to_string(tile));
            }

            std::vector<std::byte> candidate = invocation.copyOutputBytes();
            if (reference.empty())
            {
                reference = std::move(candidate);
                continue;
            }

            const auto mismatch = std::mismatch(
                reference.begin(),
                reference.end(),
                candidate.begin(),
                candidate.end());
            if (mismatch.first != reference.end())
            {
                const std::size_t byte_offset = static_cast<std::size_t>(
                    std::distance(reference.begin(), mismatch.first));
                throw std::runtime_error(
                    "CPU FA2 physical tile changed output bytes: reference_tile=" +
                    std::to_string(reference_tile) + " candidate_tile=" +
                    std::to_string(tile) + " first_byte=" +
                    std::to_string(byte_offset));
            }
        }
    }

    /**
     * @brief Measure candidates in balanced rounds without setup or flush time.
     *
     * A contiguous candidate-at-a-time sweep aliases CPU frequency changes,
     * package thermals, and neighboring load with tile order. Round `r` starts
     * at candidate `r mod C`; complete cycles therefore place every candidate
     * in every temporal position exactly once. The second cycle reverses travel
     * direction as a guard against predecessor/cache bias. Each position is a
     * short block of consecutive launches for one candidate, approximating
     * steady production replay instead of inheriting another candidate's cache
     * state. Block means are normalized against the fastest candidate block in
     * the same cycle before the median tournament score is formed; absolute
     * minima and medians remain available for throughput diagnosis.
     *
     * @param invocation Fully prepared production attention invocation.
     * @param candidates Exact compiled tile candidates to compare.
     * @param filter Warmup/sample count and optional cold-cache policy.
     * @return One robust timing summary per candidate, in candidate order.
     */
    [[nodiscard]] std::vector<CandidateTiming> measureCandidates(
        PreparedAttentionInvocation &invocation,
        const std::vector<int> &candidates,
        const TournamentFilter &filter)
    {
        if (candidates.empty())
            throw std::invalid_argument("CPU FA2 tournament has no candidates");
        const int balanced_cycle_count = candidates.size() > 1
                                             ? 2 * static_cast<int>(
                                                       candidates.size())
                                             : 1;
        if (filter.samples % balanced_cycle_count != 0)
        {
            throw std::invalid_argument(
                "CPU FA2 certifying samples must contain complete "
                "forward/reverse candidate cycles");
        }

        verifyCandidateByteEquivalence(invocation, candidates);

        for (const int tile : candidates)
        {
            invocation.configureKVTileCandidate(tile);
            for (int warmup = 0; warmup < filter.warmup; ++warmup)
            {
                if (!invocation.runOnce())
                    throw std::runtime_error("CPU FA2 warmup launch failed");
            }
        }

        std::vector<std::vector<double>> samples(candidates.size());
        for (auto &candidate_samples : samples)
            candidate_samples.reserve(static_cast<std::size_t>(filter.samples));

        const std::size_t candidate_count = candidates.size();
        const int samples_per_block =
            filter.samples / balanced_cycle_count;
        std::vector<std::vector<double>> cycle_block_means(
            candidate_count,
            std::vector<double>(
                static_cast<std::size_t>(balanced_cycle_count), 0.0));

        for (int cycle = 0; cycle < balanced_cycle_count; ++cycle)
        {
            const std::size_t start =
                static_cast<std::size_t>(cycle) % candidate_count;
            const bool reverse =
                static_cast<std::size_t>(cycle) >= candidate_count;
            for (std::size_t position = 0;
                 position < candidate_count;
                 ++position)
            {
                const std::size_t candidate = reverse
                                                  ? (start + candidate_count -
                                                     position) %
                                                        candidate_count
                                                  : (start + position) %
                                                        candidate_count;
                invocation.configureKVTileCandidate(candidates[candidate]);
                const int participants = primeOpenMPWorkerTeam();
                if (participants != omp_get_max_threads())
                {
                    throw std::runtime_error(
                        "CPU FA2 tournament worker team changed size");
                }

                double block_total_us = 0.0;
                for (int sample = 0; sample < samples_per_block; ++sample)
                {
                    if (filter.cold_cache)
                        invocation.flushKV();
                    const auto begin = std::chrono::steady_clock::now();
                    if (!invocation.runOnce())
                    {
                        throw std::runtime_error(
                            "CPU FA2 measured launch failed");
                    }
                    const auto end = std::chrono::steady_clock::now();
                    const double elapsed_us =
                        std::chrono::duration<double, std::micro>(end - begin)
                            .count();
                    samples[candidate].push_back(elapsed_us);
                    block_total_us += elapsed_us;
                }
                cycle_block_means[candidate][
                    static_cast<std::size_t>(cycle)] =
                    block_total_us / static_cast<double>(samples_per_block);
            }
        }

        std::vector<CandidateTiming> timings;
        timings.reserve(candidate_count);

        std::vector<double> cycle_fastest(
            static_cast<std::size_t>(balanced_cycle_count),
            std::numeric_limits<double>::max());
        for (const auto &candidate_blocks : cycle_block_means)
        {
            if (candidate_blocks.size() != cycle_fastest.size())
            {
                throw std::runtime_error(
                    "CPU FA2 tournament produced an incomplete candidate cycle");
            }
            for (std::size_t cycle = 0; cycle < cycle_fastest.size(); ++cycle)
            {
                cycle_fastest[cycle] = std::min(
                    cycle_fastest[cycle], candidate_blocks[cycle]);
            }
        }

        for (std::size_t candidate = 0;
             candidate < candidate_count;
             ++candidate)
        {
            std::vector<double> candidate_samples = samples[candidate];
            std::vector<double> paired_ratios;
            paired_ratios.reserve(cycle_fastest.size());
            for (std::size_t cycle = 0;
                 cycle < cycle_fastest.size();
                 ++cycle)
            {
                if (!(cycle_fastest[cycle] > 0.0))
                {
                    throw std::runtime_error(
                        "CPU FA2 tournament observed a non-positive latency");
                }
                paired_ratios.push_back(
                    cycle_block_means[candidate][cycle] /
                    cycle_fastest[cycle]);
            }
            std::sort(candidate_samples.begin(), candidate_samples.end());
            std::sort(paired_ratios.begin(), paired_ratios.end());
            timings.push_back({
                .tile = candidates[candidate],
                .minimum_us = candidate_samples.front(),
                .median_us = candidate_samples[
                    candidate_samples.size() / 2],
                .paired_median_ratio = paired_ratios[
                    paired_ratios.size() / 2],
            });
        }
        return timings;
    }

    /** One policy-regret record retained for aggregate certification. */
    struct DomainResult
    {
        double regret_percent = 0.0;
    };

    /** @return Nearest-rank percentile from a non-empty value set. */
    [[nodiscard]] double percentile(
        std::vector<double> values,
        double probability)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t index = std::min(
            values.size() - 1,
            static_cast<std::size_t>(
                std::ceil(probability * values.size()) - 1.0));
        return values[index];
    }

    /** Return the compact or exhaustive work-point inventory. */
    [[nodiscard]] std::vector<WorkPoint> workPoints(bool full)
    {
        if (!full)
        {
            return {
                {AttentionRegime::Decode, 1, 8192},
                {AttentionRegime::GroupedVerifier, 15, 8192 + 15},
                {AttentionRegime::Prefill, 128, 8192 + 128},
            };
        }
        return {
            {AttentionRegime::Decode, 1, 8192},
            {AttentionRegime::GroupedVerifier, 2, 8192 + 2},
            {AttentionRegime::GroupedVerifier, 4, 8192 + 4},
            {AttentionRegime::GroupedVerifier, 8, 8192 + 8},
            {AttentionRegime::GroupedVerifier, 15, 8192 + 15},
            {AttentionRegime::Prefill, 32, 8192 + 32},
            {AttentionRegime::Prefill, 128, 8192 + 128},
        };
    }

    /**
     * @return Bounded work points for the canonical all-geometry byte gate.
     *
     * A 512-row prefix opens at least three canonical 256-row summaries once
     * the live query rows are included. That is sufficient to exercise the
     * context-parallel reducer and every physical tile without paying the
     * 8192-row steady-state timing cost. Grouped depths cover every production
     * MTP boundary and prefill points cross both launch and summary boundaries.
     */
    [[nodiscard]] std::vector<WorkPoint> byteTotalityWorkPoints()
    {
        constexpr int prefix_rows = 512;
        return {
            {AttentionRegime::Decode, 1, prefix_rows + 1},
            {AttentionRegime::GroupedVerifier, 2, prefix_rows + 2},
            {AttentionRegime::GroupedVerifier, 4, prefix_rows + 4},
            {AttentionRegime::GroupedVerifier, 8, prefix_rows + 8},
            {AttentionRegime::GroupedVerifier, 15, prefix_rows + 15},
            {AttentionRegime::Prefill, 32, prefix_rows + 32},
            {AttentionRegime::Prefill, 128, prefix_rows + 128},
            {AttentionRegime::Prefill, 257, prefix_rows + 257},
        };
    }

    /** Restore the caller's OpenMP preference after a bounded integration gate. */
    class ScopedOpenMPThreadLimit
    {
    public:
        /** @brief Limit the all-geometry byte sweep to physical-work test scale. */
        explicit ScopedOpenMPThreadLimit(int requested)
            : previous_(omp_get_max_threads())
        {
            omp_set_num_threads(std::max(1, std::min(requested, previous_)));
        }

        /** @brief Restore the process preference observed at construction. */
        ~ScopedOpenMPThreadLimit()
        {
            omp_set_num_threads(previous_);
        }

        ScopedOpenMPThreadLimit(const ScopedOpenMPThreadLimit &) = delete;
        ScopedOpenMPThreadLimit &operator=(
            const ScopedOpenMPThreadLimit &) = delete;

    private:
        int previous_ = 1; ///< OpenMP preference restored at scope exit.
    };

    /**
     * @brief Authenticate every Qwen/TP/native-format/physical-tile combination.
     *
     * This is intentionally a correctness gate, not a timing tournament. Each
     * candidate launches the optimized production kernel exactly once and must
     * match the first candidate byte for byte. The compact process-isolated
     * tournament remains the economy certificate; separating the two concerns
     * keeps this exhaustive matrix deterministic and suitable for Integration
     * CTest while retaining complete combinatorial coverage.
     */
    void runFullQwenTPByteTotality()
    {
        ScopedOpenMPThreadLimit thread_limit(/*requested=*/8);
        const std::vector<ParticipantAttentionGeometry> participants =
            buildDistinctParticipantGeometries();
        ASSERT_FALSE(participants.empty());
        const std::vector<WorkPoint> points = byteTotalityWorkPoints();
        const std::vector<int> candidates(
            kCompiledKVTiles.begin(), kCompiledKVTiles.end());

        std::size_t domains = 0;
        for (const ParticipantAttentionGeometry &geometry : participants)
        {
            for (const NativeKVFormat format : kNativeKVFormats)
            {
                for (const WorkPoint point : points)
                {
                    PreparedAttentionInvocation invocation(
                        geometry, format, point);
                    ASSERT_TRUE(invocation.ready())
                        << geometry.sources << ' ' << formatName(format)
                        << " M=" << point.query_rows
                        << " KV=" << point.kv_len << ": "
                        << invocation.error();
                    try
                    {
                        verifyCandidateByteEquivalence(
                            invocation, candidates);
                    }
                    catch (const std::exception &error)
                    {
                        FAIL() << geometry.sources << ' '
                               << formatName(format)
                               << " regime=" << regimeName(point.regime)
                               << " M=" << point.query_rows
                               << " KV=" << point.kv_len << ": "
                               << error.what();
                    }
                    ++domains;
                }
            }
        }

        EXPECT_EQ(
            domains,
            participants.size() * kNativeKVFormats.size() * points.size());
        std::cout << "CPU_FA2_BYTE_TOTALITY,domains=" << domains
                  << ",candidates_per_domain=" << candidates.size()
                  << ",workers=" << omp_get_max_threads() << '\n';
    }

    /**
     * Reduce the catalog to one underfilled geometry per head dimension for the
     * routine gate. The exhaustive run still covers every deduplicated TP
     * participant; both matrices retain every native K/V format.
     */
    [[nodiscard]] std::vector<ParticipantAttentionGeometry>
    representativeGeometries(
        const std::vector<ParticipantAttentionGeometry> &catalog)
    {
        std::vector<ParticipantAttentionGeometry> selected;
        for (const int head_dim : {64, 128, 256})
        {
            const auto participant = std::min_element(
                catalog.begin(), catalog.end(),
                [=](const ParticipantAttentionGeometry &lhs,
                    const ParticipantAttentionGeometry &rhs)
                {
                    const int lhs_heads = lhs.head_dim == head_dim
                                              ? lhs.n_heads
                                              : std::numeric_limits<int>::max();
                    const int rhs_heads = rhs.head_dim == head_dim
                                              ? rhs.n_heads
                                              : std::numeric_limits<int>::max();
                    return lhs_heads < rhs_heads;
                });
            if (participant != catalog.end() &&
                participant->head_dim == head_dim)
            {
                selected.push_back(*participant);
            }
        }
        return selected;
    }

    /** Execute and certify either the compact or exhaustive tournament. */
    void runTournament(bool full)
    {
        const TournamentFilter filter = TournamentFilter::fromEnvironment();
        const CacheInfo &cache = cache_info();
        ASSERT_GT(cache.l2_size, 0U);
        ASSERT_GT(cache.cache_line, 0U);

        const std::vector<ParticipantAttentionGeometry> catalog =
            buildDistinctParticipantGeometries();
        const std::vector<ParticipantAttentionGeometry> participants =
            full ? catalog : representativeGeometries(catalog);
        ASSERT_FALSE(participants.empty());

        std::cout
            << "CPU_FA2_TOURNAMENT_AXIS,"
            << attention::attentionPrefillParallelAxisName(
                   filter.requested_axis)
            << '\n'
            << "sources,format,codegen_isa,runtime_isa,regime,M,KV,heads,"
               "kv_heads,head_dim,gqa_rep,"
               "workers,key_row_bytes,value_row_bytes,policy_tile,winner_tile,"
               "policy_us,winner_us,regret_pct,tile,tile_min_us,tile_median_us,"
               "tile_paired_median_ratio\n";

        std::vector<DomainResult> domains;
        for (const ParticipantAttentionGeometry &geometry : participants)
        {
            if (filter.head_dim > 0 && geometry.head_dim != filter.head_dim)
                continue;
            if (filter.local_heads > 0 &&
                geometry.n_heads != filter.local_heads)
            {
                continue;
            }
            if (filter.local_kv_heads > 0 &&
                geometry.n_kv_heads != filter.local_kv_heads)
            {
                continue;
            }

            for (const NativeKVFormat format : kNativeKVFormats)
            {
                if (!filter.format.empty() &&
                    filter.format != formatName(format))
                {
                    continue;
                }

                for (const WorkPoint point : workPoints(full))
                {
                    if (filter.query_rows > 0 &&
                        point.query_rows != filter.query_rows)
                    {
                        continue;
                    }
                    if (filter.kv_len > 0 && point.kv_len != filter.kv_len)
                        continue;

                    PreparedAttentionInvocation invocation(
                        geometry, format, point, filter.requested_axis);
                    ASSERT_TRUE(invocation.ready())
                        << geometry.sources << " " << formatName(format)
                        << " M=" << point.query_rows << " KV=" << point.kv_len
                        << ": " << invocation.error();

                    const CPUFA2KVTileGeometry tile_geometry{
                        .storage_pair = storagePair(format),
                        .vector_isa = activePolicyISA(),
                        .codegen_isa = compiledCPUFA2CodegenISA(),
                        .head_dim = geometry.head_dim,
                        .kv_rows = point.kv_len,
                        .key_head_row_bytes = invocation.keyHeadRowBytes(),
                        .value_head_row_bytes = invocation.valueHeadRowBytes(),
                        .cache = {
                            .private_l1d_bytes = cache.l1_size,
                            .private_l2_bytes = cache.l2_size,
                            .shared_l3_bytes = cache.l3_size,
                            .cache_line_bytes = cache.cache_line,
                        },
                    };
                    const int policy_tile = selectCPUFA2KVTile(tile_geometry);
                    ASSERT_GT(policy_tile, 0);

                    std::vector<int> candidate_tiles;
                    for (const int tile : kCompiledKVTiles)
                    {
                        if (filter.tile > 0 && tile != filter.tile)
                            continue;
                        candidate_tiles.push_back(tile);
                    }
                    const std::vector<CandidateTiming> timings =
                        measureCandidates(
                            invocation,
                            candidate_tiles,
                            filter);
                    ASSERT_FALSE(timings.empty());
                    ASSERT_TRUE(invocation.outputFinite())
                        << geometry.sources << " " << formatName(format)
                        << " M=" << point.query_rows;

                    const auto winner = std::min_element(
                        timings.begin(), timings.end(),
                        [](const CandidateTiming &lhs,
                           const CandidateTiming &rhs)
                        {
                            return lhs.paired_median_ratio <
                                   rhs.paired_median_ratio;
                        });
                    const auto policy = std::find_if(
                        timings.begin(), timings.end(),
                        [=](const CandidateTiming &timing)
                        { return timing.tile == policy_tile; });

                    /*
                     * A single-candidate profiler run intentionally lacks the
                     * winner/policy pair. It emits isolated evidence without
                     * pretending to certify regret.
                     */
                    const bool certifiable =
                        timings.size() > 1 && policy != timings.end();
                    const double policy_us = certifiable
                                                 ? policy->median_us
                                                 : 0.0;
                    const double regret = certifiable
                                              ? 100.0 *
                                                    (policy->paired_median_ratio /
                                                         winner->paired_median_ratio -
                                                     1.0)
                                              : 0.0;
                    if (certifiable)
                        domains.push_back({.regret_percent = regret});

                    for (const CandidateTiming &timing : timings)
                    {
                        std::cout
                            << geometry.sources << ','
                            << formatName(format) << ','
                            << cpuFA2CodegenISAName(
                                   tile_geometry.codegen_isa)
                            << ','
                            << cpuFA2VectorISAName(tile_geometry.vector_isa) << ','
                            << regimeName(point.regime) << ','
                            << point.query_rows << ','
                            << point.kv_len << ','
                            << geometry.n_heads << ','
                            << geometry.n_kv_heads << ','
                            << geometry.head_dim << ','
                            << geometry.replicated_gqa_n_rep << ','
                            << omp_get_max_threads() << ','
                            << invocation.keyHeadRowBytes() << ','
                            << invocation.valueHeadRowBytes() << ','
                            << policy_tile << ','
                            << winner->tile << ','
                            << std::fixed << std::setprecision(3)
                            << policy_us << ','
                            << winner->median_us << ','
                            << regret << ','
                            << timing.tile << ','
                            << timing.minimum_us << ','
                            << timing.median_us << ','
                            << timing.paired_median_ratio << '\n';
                    }
                }
            }
        }

        if (filter.tile > 0)
            return;

        ASSERT_FALSE(domains.empty());
        std::vector<double> regrets;
        regrets.reserve(domains.size());
        for (const DomainResult &domain : domains)
            regrets.push_back(domain.regret_percent);
        const double p50 = percentile(regrets, 0.50);
        const double p95 = percentile(regrets, 0.95);
        const double maximum = *std::max_element(regrets.begin(), regrets.end());
        std::cout << "CPU_FA2_POLICY_SUMMARY,domains=" << domains.size()
                  << ",p50_regret_pct=" << p50
                  << ",p95_regret_pct=" << p95
                  << ",max_regret_pct=" << maximum << '\n';

        const int allowed_p95 = TournamentFilter::envInt(
            "LLAMINAR_CPU_FA2_TOURNAMENT_MAX_P95_REGRET", 5);
        EXPECT_LE(p95, static_cast<double>(allowed_p95))
            << "The installed common CPU FA2 tile policy is not economical";
    }
} // namespace

TEST(Perf__CPUFlashAttentionTileTournament,
     RepresentativeAllFormatPolicyGate)
{
    runTournament(/*full=*/false);
}

TEST(Perf__CPUFlashAttentionTileTournament,
     FullQwenTPAllFormatTournament)
{
    if (TournamentFilter::envInt(
            "LLAMINAR_CPU_FA2_TOURNAMENT_FULL", 0) == 0)
    {
        GTEST_SKIP()
            << "Set LLAMINAR_CPU_FA2_TOURNAMENT_FULL=1 for the exhaustive run";
    }
    runTournament(/*full=*/true);
}

TEST(Perf__CPUFlashAttentionTileTournament,
     FullQwenTPAllFormatByteTotality)
{
    runFullQwenTPByteTotality();
}
