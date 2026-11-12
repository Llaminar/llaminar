/**
 * @file Perf__CPUCompactIQ3SDecode.cpp
 * @brief Diagnostic compact-decode tournament for the ARM prepared layout
 *        (idea 08 sub-change B bound; non-installed; no production path).
 *
 * The prepared IQ3_S matrix today materializes the decoded 512-entry grid
 * into 2048 bytes per 64-column/32-K unit plus metadata (2432 bytes,
 * 9.5 bpw) against a 3.75 bpw catalog-logical source. This harness asks the
 * idea-08 Phase A tournament question at the unit level: can an ARM-native
 * compact layout — the raw IQ3_S payload rearranged per decode group, plus
 * the exact same scale planes — feed the same production NEON dot/scale
 * program, byte-identically, and at what decode-ALU cost against the
 * 55 percent smaller weight stream?
 *
 * Compact layout per unit (live 1088 bytes = 4.25 bpw; disclosed, exact):
 *   [8 groups x (64 index bytes | 8 B ninth-bit plane | 32 B sign plane)]
 *   [64 fp16 scale halves][64 uint16 raw scale nibbles]
 * The per-group payload arrangement is the pack-time transpose that makes
 * each four-column quad's indices contiguous; the scale/nibble planes
 * mirror the production admitted-planes layout exactly so both arms run
 * the identical scale program.
 *
 * Compact decode per (group, column quad), vectorized except the grid
 * gather: four scalar `vld1q_lane_u32` gathers from the L1-resident
 * 512-entry grid build the magnitude vector directly in the production
 * byte order; a per-byte sign mask built from the quad's sign nibbles
 * fixes signs with the production test/negate/select program. The gather
 * is the irreducibly scalar part on this ISA (NEON has no gather load);
 * a two-level `vqtbl4` scheme is the Phase-B refinement candidate and can
 * only reduce the measured decode cost.
 *
 * Byte oracles, before and after timing: (1) the compact decode of every
 * unit reproduces the prepared data plane byte for byte (memcmp against
 * the production-expanded engine's own data plane); (2) the full-matrix
 * FP32 outputs of the compact arm and the expanded arm are bit-identical
 * for every timed row shape and compiled family. Both arms share one
 * microkernel template and differ only in their weight-vector source, so
 * the comparison isolates the decode cost. The production dot leaves
 * (`NEONMultiScaleOps::dot`, `pairDot`) and the production
 * `accumulateSingleScaleBlockNEON` program are used unchanged, with the
 * same IQ3_S scale equation (`half(d) * (2*metadata + 1)`).
 *
 * Families: ASIMD always; DotProd/I8MM compiled when the translation unit
 * exposes the feature macros, runtime-admitted against native CPU
 * capabilities, mirroring the ISA-cell build convention (the window runs
 * the bench under the NEON / NEON_DOTPROD / NEON_I8MM build cells; the
 * I8MM cell also exercises the SMMLA row-pair walk). Row shapes M=1, M=4
 * and M=15; I8MM walks row pairs (an odd trailing row uses the family's
 * SDOT leaf, exactly as production row pairs do). Timing is unprofiled,
 * alternating paired samples per arm.
 *
 * This bench grants no production change: it is a diagnostic bound for the
 * idea-08 Phase B gate, in the non-installed diagnostic namespace.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gemm/CPUNativeNEONContributionContract.h"
#include "kernels/cpu/gemm/CPUNativeNEONMultiScaleOps.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedDecode.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "tensors/AlignedVector.h"
#include "tensors/TensorClasses.h"
#include "tensors/VnniPackContext.h"
#include "utils/CPUFeatures.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace llaminar2;
using llaminar2::cpu::native_quantised::CPUNativeQuantisedGemmKernel;
using llaminar2::cpu::native_quantised::CPUNativeQuantisedPackedWeights;

namespace
{
    constexpr int kRows = 512;      ///< Matrix outputs; the fixture packs
                                    ///  chunk zero (64 columns) per unit.
    constexpr int kK = 2048;        ///< Matrix inputs (64 blocks of 32).
    constexpr int kBlocksPerRow = kK / 32;
    constexpr int kMaxRows = 15;    ///< Largest timed row shape.

    /// Compact unit stride: live 1088 bytes, exact (no padding).
    constexpr size_t kCompactUnitBytes =
        8u * (64u + 8u + 32u) + 128u + 128u;

    int envPositiveInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max())
        {
            throw std::runtime_error(
                std::string(name) + " must be a positive integer");
        }
        return static_cast<int>(parsed);
    }

    /** @brief The production expanded weight source (the current layout). */
    struct ExpandedSource
    {
        const CPUNativeQuantisedPackedWeights *packed = nullptr;

        /// Sixteen decoded bytes for (block, group, column quad), in the
        /// production interleaved order: four columns x four K values.
        inline int8x16_t weights16(int block, int group, int quad) const
        {
            const auto *unit = packed->interleavedBase() +
                static_cast<size_t>(block) *
                    static_cast<size_t>(packed->interleaved_block_stride);
            return vld1q_s8(reinterpret_cast<const int8_t *>(
                unit + static_cast<size_t>(group) * 256u +
                static_cast<size_t>(quad) * 4u));
        }

        /// One 64-column chunk: the scale plane after data + compensation.
        inline const uint8_t *scalesHalf(int block) const
        {
            const auto *unit = packed->interleavedBase() +
                static_cast<size_t>(block) *
                    static_cast<size_t>(packed->interleaved_block_stride);
            return unit + static_cast<size_t>(packed->data_stride) + 128u;
        }

        /// The raw-coefficient plane after data + comp + scales.
        inline const uint8_t *nibbleWords(int block) const
        {
            const auto *unit = packed->interleavedBase() +
                static_cast<size_t>(block) *
                    static_cast<size_t>(packed->interleaved_block_stride);
            return unit + static_cast<size_t>(packed->data_stride) + 256u;
        }
    };

    /** @brief The diagnostic compact weight source (the candidate layout). */
    struct CompactSource
    {
        const uint8_t *base = nullptr; ///< [units][1088] compact stream.
        size_t unit_stride = kCompactUnitBytes;

        /**
         * @brief Decode one (block, group, quad) weight vector from the
         *        compact layout, in the production byte order.
         *
         * Four scalar grid gathers build the magnitude vector directly in
         * production order; the quad's four sign nibbles become a per-byte
         * mask applied with the production test/negate/select program.
         */
        inline int8x16_t weights16(int block, int group, int quad) const
        {
            const uint8_t *unit = base +
                static_cast<size_t>(block) * unit_stride;
            const uint8_t *group_section =
                unit + static_cast<size_t>(group) * 104u;
            const uint8_t *indices = group_section;
            const uint8_t *bits = group_section + 64u;
            const uint8_t *signs = group_section + 72u;

            const auto gather = [&](int lane) {
                const int column = quad * 4 + lane;
                const unsigned index =
                    static_cast<unsigned>(indices[column]) |
                    (((bits[column / 8] >> (column % 8)) & 1u) << 8);
                return llaminar2::iq3s_grid[index];
            };
            const uint32_t entry0 = gather(0);
            const uint32_t entry1 = gather(1);
            const uint32_t entry2 = gather(2);
            const uint32_t entry3 = gather(3);
            uint32x4_t magnitudes;
            magnitudes = vld1q_lane_u32(&entry0, magnitudes, 0);
            magnitudes = vld1q_lane_u32(&entry1, magnitudes, 1);
            magnitudes = vld1q_lane_u32(&entry2, magnitudes, 2);
            magnitudes = vld1q_lane_u32(&entry3, magnitudes, 3);

            // Four sign nibbles, two per byte: quad*4 is a multiple of four,
            // so the quad's nibbles never straddle a byte boundary.
            const unsigned sign_word =
                static_cast<unsigned>(signs[quad / 2]) |
                (static_cast<unsigned>(signs[quad / 2 + 1]) << 8);
            const unsigned shift = static_cast<unsigned>(quad % 2) * 8u;
            const auto sign_nibble = [&](int lane) -> unsigned {
                return (sign_word >> (shift + 4u * static_cast<unsigned>(lane))) &
                    0xFu;
            };
            const auto expand = [](unsigned nibble) -> unsigned {
                // Expand bit j of the nibble to byte j: {1,2,4,8} positions.
                return ((nibble & 1u) ? 0xFFu : 0u) |
                    ((nibble & 2u) ? 0xFF00u : 0u) |
                    ((nibble & 4u) ? 0xFF0000u : 0u) |
                    ((nibble & 8u) ? 0xFF000000u : 0u);
            };
            const unsigned expanded0 = expand(sign_nibble(0));
            const unsigned expanded1 = expand(sign_nibble(1));
            const unsigned expanded2 = expand(sign_nibble(2));
            const unsigned expanded3 = expand(sign_nibble(3));
            uint32x4_t sign_mask;
            sign_mask = vld1q_lane_u32(&expanded0, sign_mask, 0);
            sign_mask = vld1q_lane_u32(&expanded1, sign_mask, 1);
            sign_mask = vld1q_lane_u32(&expanded2, sign_mask, 2);
            sign_mask = vld1q_lane_u32(&expanded3, sign_mask, 3);

            // Production sign program: test each byte against its {1,2,4,8}
            // bit position, negate where set, select without branching.
            const auto bit_positions = vdupq_n_u32(0x08040201u);
            const auto negative = vtstq_u8(
                vreinterpretq_u8_u32(sign_mask),
                vreinterpretq_u8_u32(bit_positions));
            const auto bytes = std::bit_cast<int8x16_t>(magnitudes);
            return std::bit_cast<int8x16_t>(
                vbslq_s8(negative, vnegq_s8(bytes), bytes));
        }

        inline const uint8_t *scalesHalf(int block) const
        {
            return base + static_cast<size_t>(block) * unit_stride + 832u;
        }

        inline const uint8_t *nibbleWords(int block) const
        {
            return base + static_cast<size_t>(block) * unit_stride + 960u;
        }
    };

    /**
     * @brief One row's unit walk shared by both arms: the production
     *        program with the weight-vector source as the only difference.
     *
     * Mirrors the production per-unit loop for IQ3SExpandedSourceScale: per
     * block, eight groups of four K values dot four columns; after the
     * groups, one exact FP32 scale/accumulate step per block with the IQ3_S
     * source-scale equation. The accumulated FP32 output is bit-identical
     * between arms because the decoded weight bytes are identical.
     */
    template <typename Ops, typename WeightSource>
    void walkUnitRows(const WeightSource &source,
        const Q8_1Block *activation_row, float out[64])
    {
        using I = typename Ops::I;
        using F = typename Ops::F;
        for (int quad = 0; quad < 16; ++quad)
        {
            float32x4_t accumulator = vdupq_n_f32(0.0f);
            for (int block = 0; block < kBlocksPerRow; ++block)
            {
                I dots = I{};
                for (int group = 0; group < 8; ++group)
                {
                    const auto weights = source.weights16(block, group, quad);
                    int word = 0;
                    std::memcpy(&word, activation_row[block].qs + group * 4,
                        sizeof(word));
                    dots = Ops::dot(dots, std::bit_cast<I>(weights),
                        I{} + word);
                }
                const F scale0 = Ops::half(reinterpret_cast<const uint16_t *>(
                    source.scalesHalf(block) + static_cast<size_t>(quad) * 4u));
                const I metadata = Ops::words(reinterpret_cast<const uint16_t *>(
                    source.nibbleWords(block) + static_cast<size_t>(quad) * 4u));
                const I coefficient = metadata * 2 + 1;
                const F scale = scale0 * __builtin_convertvector(coefficient, F);
                const float activation_scale =
                    cpu::native_quantised::nativeQuantisedFP16ScaleToFP32(
                        activation_row[block].d);
                accumulator = cpu::native_quantised::
                    accumulateSingleScaleBlockNEON(accumulator, dots, scale,
                        activation_scale,
                        CPUProjectionNumericalPolicy::GPUAlignedExpert);
            }
            vst1q_f32(out + static_cast<size_t>(quad) * 4u, accumulator);
        }
    }

    /**
     * @brief I8MM row-pair unit walk: four adjacent group pairs feed four
     *        SMMLA pairs per quad, exactly as the production pair path
     *        consumes them; the same weight source and scale program.
     */
    template <typename Ops, typename WeightSource>
    void walkUnitPairs(const WeightSource &source,
        const Q8_1Block *activation_row0, const Q8_1Block *activation_row1,
        float out0[64], float out1[64])
    {
        using I = typename Ops::I;
        using F = typename Ops::F;
        for (int quad = 0; quad < 16; ++quad)
        {
            float32x4_t accumulator0 = vdupq_n_f32(0.0f);
            float32x4_t accumulator1 = vdupq_n_f32(0.0f);
            for (int block = 0; block < kBlocksPerRow; ++block)
            {
                I columns01 = I{};
                I columns23 = I{};
                for (int pair_group = 0; pair_group < 4; ++pair_group)
                {
                    const int group = pair_group * 2;
                    const auto first = source.weights16(block, group, quad);
                    const auto second =
                        source.weights16(block, group + 1, quad);
                    const auto act = Ops::activationPair(
                        activation_row0[block].qs + group * 4,
                        activation_row1[block].qs + group * 4);
                    Ops::pairDot(columns01, columns23, act,
                        std::bit_cast<I>(first), std::bit_cast<I>(second));
                }
                I dots[2] = {columns01, columns23};
                Ops::publishPairs(columns01, columns23, dots);
                const F scale0 = Ops::half(reinterpret_cast<const uint16_t *>(
                    source.scalesHalf(block) + static_cast<size_t>(quad) * 4u));
                const I metadata = Ops::words(reinterpret_cast<const uint16_t *>(
                    source.nibbleWords(block) + static_cast<size_t>(quad) * 4u));
                const I coefficient = metadata * 2 + 1;
                const F scale = scale0 * __builtin_convertvector(coefficient, F);
                const float activation_scale0 =
                    cpu::native_quantised::nativeQuantisedFP16ScaleToFP32(
                        activation_row0[block].d);
                const float activation_scale1 =
                    cpu::native_quantised::nativeQuantisedFP16ScaleToFP32(
                        activation_row1[block].d);
                accumulator0 = cpu::native_quantised::
                    accumulateSingleScaleBlockNEON(accumulator0, dots[0],
                        scale, activation_scale0,
                        CPUProjectionNumericalPolicy::GPUAlignedExpert);
                accumulator1 = cpu::native_quantised::
                    accumulateSingleScaleBlockNEON(accumulator1, dots[1],
                        scale, activation_scale1,
                        CPUProjectionNumericalPolicy::GPUAlignedExpert);
            }
            vst1q_f32(out0 + static_cast<size_t>(quad) * 4u, accumulator0);
            vst1q_f32(out1 + static_cast<size_t>(quad) * 4u, accumulator1);
        }
    }

    /** @brief Deterministic hidden rows with quantization-relevant tails. */
    AlignedVector<float> makeHidden(int rows, int k)
    {
        AlignedVector<float> hidden(static_cast<size_t>(rows) *
                                    static_cast<size_t>(k));
        for (size_t index = 0; index < hidden.size(); ++index)
        {
            const int centered =
                static_cast<int>((index * 37u + index / 13u) % 257u) - 128;
            hidden[index] =
                static_cast<float>(centered) * 0.004125f +
                static_cast<float>(static_cast<int>(index % 11u) - 5) *
                    0.000071f;
        }
        return hidden;
    }

    /** @brief Median of a sample list (upper median). */
    double medianOf(std::vector<double> samples)
    {
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2u];
    }
} // namespace

TEST(Perf__CPUCompactIQ3SDecode, CompactDecodeIsByteExactAndTournamentTimed)
{
    // ------------------------------------------------------------------
    // Setup: one production-prepared IQ3_S engine plus its raw source
    // blocks, rearranged into the diagnostic compact layout.
    // ------------------------------------------------------------------
    const auto &format = []() -> const test::QuantizedVerifierFormatCase &
    {
        for (const auto &entry : test::quantizedMoEVerifierFormats())
            if (entry.label == std::string("IQ3_S"))
                return entry;
        throw std::runtime_error("IQ3_S missing from the canonical registry");
    }();
    auto source_weights = format.create(
        {static_cast<size_t>(kRows), static_cast<size_t>(kK)}, 41000u);
    ASSERT_TRUE(source_weights) << "IQ3_S source creation failed";

    CPUNativeQuantisedGemmKernel engine(
        source_weights.get(), 0, -1,
        CPUProjectionNumericalPolicy::GPUAlignedExpert);
    ASSERT_TRUE(engine.isValid()) << "IQ3_S preparation failed";
    const auto &packed = engine.packedWeights();
    ASSERT_EQ(packed.encoding,
        cpu::native_quantised::CPUNativeQuantisedEncoding::
            IQ3SExpandedSourceScale)
        << "the fixture requires the production expanded IQ3_S encoding";

    // Extract the raw source blocks exactly as the packer does, then build
    // the compact layout: per (group, column) index byte, ninth bit and
    // sign nibble; per (block) scale and nibble planes.
    const auto *unpackable =
        IINT8Unpackable::fromTensor(source_weights.get());
    ASSERT_TRUE(unpackable) << "IQ3_S source lacks native unpacking";
    const auto *fmt = unpackable->vnniFormatInfo();
    ASSERT_TRUE(fmt) << "IQ3_S source lacks format metadata";
    ASSERT_EQ(fmt->payload_bytes, 13) << "IQ3_S payload contract changed";

    AlignedVector<uint8_t> compact_stream(static_cast<size_t>(kBlocksPerRow) *
                                          kCompactUnitBytes);
    std::vector<uint8_t> payload(13);
    std::vector<uint16_t> scale(1), min(1);
    for (int block = 0; block < kBlocksPerRow; ++block)
    {
        uint8_t *unit = compact_stream.data() +
            static_cast<size_t>(block) * kCompactUnitBytes;
        std::memset(unit, 0, kCompactUnitBytes);
        for (int column = 0; column < 64; ++column)
        {
            unpackable->packVnniBlock(
                VnniPackContext{
                    .raw_bytes = nullptr,
                    .N = 1,
                    .K = kK,
                    .blocks_per_row = kBlocksPerRow,
                    .payload_bytes = fmt->payload_bytes,
                    .payload_array = payload.data(),
                    .scales_array = scale.data(),
                    .mins_array = min.data(),
                    .emins_array = nullptr,
                    .destination_block_origin = block,
                    .destination_block_count = 1,
                },
                column, 0, block);
            for (int group = 0; group < 8; ++group)
            {
                uint8_t *section =
                    unit + static_cast<size_t>(group) * 104u;
                section[column] = payload[static_cast<size_t>(group)];
                section[64u + static_cast<size_t>(column) / 8u] |=
                    static_cast<uint8_t>(((payload[8] >> group) & 1)
                        << (column % 8));
                const unsigned signs = static_cast<unsigned>(
                    payload[9u + static_cast<size_t>(group) / 2u]);
                const unsigned nibble_position =
                    4u * static_cast<unsigned>(column % 2);
                section[72u + static_cast<size_t>(column) / 2u] |=
                    static_cast<uint8_t>(((signs >> nibble_position) & 0xFu)
                        << (4u * (column % 2)));
            }
            std::memcpy(unit + 832u + static_cast<size_t>(column) * 2u,
                scale.data(), sizeof(uint16_t));
            std::memcpy(unit + 960u + static_cast<size_t>(column) * 2u,
                min.data(), sizeof(uint16_t));
        }
    }

    // Activations: M=15 deterministic rows, quantized to Q8_1.
    const auto hidden = makeHidden(kMaxRows, kK);
    std::vector<Q8_1Block> activations(
        static_cast<size_t>(kMaxRows) * kBlocksPerRow);
    cpu::native_quantised::quantize_activations_to_q8_1(hidden.data(),
        activations.data(), kMaxRows, kK, kBlocksPerRow,
        CPUProjectionNumericalPolicy::GPUAlignedExpert);

    const CompactSource compact{compact_stream.data(), kCompactUnitBytes};
    const ExpandedSource expanded{&packed};

    // ------------------------------------------------------------------
    // Oracle 1: the compact decode reproduces the prepared data plane
    // byte for byte, for every block and every (group, quad) vector.
    // ------------------------------------------------------------------
    for (int block = 0; block < kBlocksPerRow; ++block)
        for (int group = 0; group < 8; ++group)
            for (int quad = 0; quad < 16; ++quad)
            {
                const int8x16_t decoded =
                    compact.weights16(block, group, quad);
                const int8x16_t reference =
                    expanded.weights16(block, group, quad);
                if (std::memcmp(&decoded, &reference, 16) != 0)
                {
                    ADD_FAILURE()
                        << "compact decode differs from the prepared plane at "
                        << "block=" << block << " group=" << group
                        << " quad=" << quad;
                    return;
                }
            }

    // ------------------------------------------------------------------
    // Oracle 2: full-matrix FP32 outputs are bit-identical between arms
    // on the compiled families, for the ASIMD baseline and the row-pair
    // I8MM walk when this translation unit exposes I8MM.
    // ------------------------------------------------------------------
    auto compareOutputs = [](const float *expanded_out,
                              const float *compact_out, size_t elements,
                              const std::string &label) {
        if (std::memcmp(expanded_out, compact_out,
                elements * sizeof(float)) == 0)
            return;
        const auto *a = reinterpret_cast<const uint8_t *>(expanded_out);
        const auto *b = reinterpret_cast<const uint8_t *>(compact_out);
        size_t first = 0;
        while (first + 4 <= elements * sizeof(float) &&
            std::memcmp(a + first, b + first, 4) == 0)
            first += 4;
        ADD_FAILURE() << label << " differs at float " << (first / 4);
    };

    float row_expanded[64];
    float row_compact[64];
    for (int row = 0; row < kMaxRows; ++row)
    {
        walkUnitRows<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::ASIMD>>(
            expanded, &activations[static_cast<size_t>(row) *
                static_cast<size_t>(kBlocksPerRow)],
            row_expanded);
        walkUnitRows<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::ASIMD>>(
            compact,
            &activations[static_cast<size_t>(row) *
                static_cast<size_t>(kBlocksPerRow)],
            row_compact);
        compareOutputs(row_expanded, row_compact, 64,
            "expanded vs compact FP32 (ASIMD, row " + std::to_string(row) +
                ")");
        if (::testing::Test::HasFailure())
            return;
    }

#if defined(__ARM_FEATURE_MATMUL_INT8)
    {
        float pair_expanded0[64];
        float pair_compact0[64];
        float pair_expanded1[64];
        float pair_compact1[64];
        walkUnitPairs<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::I8MM>>(
            expanded, &activations[0], &activations[kBlocksPerRow],
            pair_expanded0, pair_expanded1);
        walkUnitPairs<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::I8MM>>(
            compact, &activations[0], &activations[kBlocksPerRow],
            pair_compact0, pair_compact1);
        compareOutputs(pair_expanded0, pair_compact0, 64,
            "expanded vs compact FP32 (I8MM pair, row 0)");
        compareOutputs(pair_expanded1, pair_compact1, 64,
            "expanded vs compact FP32 (I8MM pair, row 1)");
        if (::testing::Test::HasFailure())
            return;
    }
#endif

    // ------------------------------------------------------------------
    // Timing: alternating paired samples per arm, per row shape, on the
    // compiled family. The ASIMD/DotProd cells walk rows one by one; the
    // I8MM cell walks row pairs plus one SDOT row for an odd count.
    // ------------------------------------------------------------------
#if defined(__ARM_FEATURE_DOTPROD)
    cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::DotProd);
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    cpu::ARMCPUCapabilities::native().require(
        cpu::ARMCPUFeature::IntegerMatMul);
#endif

    const int warmup = envPositiveInt("LLAMINAR_CPU_COMPACT_WARMUP", 3);
    const int samples = envPositiveInt("LLAMINAR_CPU_COMPACT_SAMPLES", 40);
    const char *family_label =
#if defined(__ARM_FEATURE_MATMUL_INT8)
        "I8MM";
#elif defined(__ARM_FEATURE_DOTPROD)
        "DotProd";
#else
        "ASIMD";
#endif


    float pair_expanded0[64];
    float pair_compact0[64];
    float pair_expanded1[64];
    float pair_compact1[64];

    struct RowShape
    {
        int rows;
        const char *label;
    };
    const RowShape shapes[] = {{1, "1"}, {4, "4"}, {15, "15"}};

    for (const RowShape &shape : shapes)
    {
        std::vector<double> expanded_samples;
        std::vector<double> compact_samples;
        auto one_pass = [&](bool expanded_arm) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
            if (shape.rows % 2 == 0)
            {
                for (int pair = 0; pair < shape.rows / 2; ++pair)
                {
                    const auto *act0 =
                        &activations[static_cast<size_t>(pair * 2) *
                                     static_cast<size_t>(kBlocksPerRow)];
                    const auto *act1 =
                        &activations[static_cast<size_t>(pair * 2 + 1) *
                                     static_cast<size_t>(kBlocksPerRow)];
                    if (expanded_arm)
                        walkUnitPairs<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::I8MM>>(expanded, act0, act1,
                            pair_expanded0, pair_expanded1);
                    else
                        walkUnitPairs<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::I8MM>>(compact, act0, act1,
                            pair_compact0, pair_compact1);
                }
                return;
            }
#endif
            for (int row = 0; row < shape.rows; ++row)
            {
                const auto *act =
                    &activations[static_cast<size_t>(row) *
                                 static_cast<size_t>(kBlocksPerRow)];
#if defined(__ARM_FEATURE_DOTPROD)
                if (expanded_arm)
                    walkUnitRows<cpu::native_quantised::multi_scale::
                            NEONMultiScaleOps<cpu::native_quantised::
                                multi_scale::NEONByteDot::DotProd>>(
                        expanded, act, row_expanded);
                else
                    walkUnitRows<cpu::native_quantised::multi_scale::
                            NEONMultiScaleOps<cpu::native_quantised::
                                multi_scale::NEONByteDot::DotProd>>(
                        compact, act, row_compact);
#else
                if (expanded_arm)
                    walkUnitRows<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::ASIMD>>(expanded, act, row_expanded);
                else
                    walkUnitRows<cpu::native_quantised::multi_scale::NEONMultiScaleOps<cpu::native_quantised::multi_scale::NEONByteDot::ASIMD>>(compact, act, row_compact);
#endif
            }
        };
        for (int pair = 0; pair < warmup; ++pair)
        {
            one_pass(true);
            one_pass(false);
        }
        for (int pair = 0; pair < samples; ++pair)
        {
            const bool expanded_first = (pair % 2) == 0;
            for (int pass = 0; pass < 2; ++pass)
            {
                const bool this_expanded =
                    expanded_first ? (pass == 0) : (pass == 1);
                const auto begin = std::chrono::steady_clock::now();
                one_pass(this_expanded);
                const auto end = std::chrono::steady_clock::now();
                const double microseconds =
                    std::chrono::duration<double, std::micro>(end - begin)
                        .count();
                (this_expanded ? expanded_samples : compact_samples)
                    .push_back(microseconds);
            }
        }
        const double expanded_median = medianOf(expanded_samples);
        const double compact_median = medianOf(compact_samples);
        std::cout << "CPU_COMPACT_IQ3S,family=" << family_label
                  << ",rows=" << shape.label << ",samples=" << samples
                  << ",expanded_median_us=" << std::fixed
                  << std::setprecision(3) << expanded_median
                  << ",compact_median_us=" << compact_median
                  << ",ratio=" << (compact_median / expanded_median)
                  << ",prepared_bpw=9.5,compact_bpw=4.25,unprofiled=1\n";
    }

    // Post-timing oracle: the compact decode still matches byte for byte.
    for (int block = 0; block < kBlocksPerRow; ++block)
        for (int group = 0; group < 8; ++group)
            for (int quad = 0; quad < 16; ++quad)
            {
                const int8x16_t decoded =
                    compact.weights16(block, group, quad);
                const int8x16_t reference =
                    expanded.weights16(block, group, quad);
                if (std::memcmp(&decoded, &reference, 16) != 0)
                {
                    ADD_FAILURE()
                        << "post-timing compact decode drifted at block="
                        << block << " group=" << group << " quad=" << quad;
                    return;
                }
            }
}
