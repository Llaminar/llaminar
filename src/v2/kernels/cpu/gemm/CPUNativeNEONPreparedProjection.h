/**
 * @file CPUNativeNEONPreparedProjection.h
 * @brief Native AArch64 prepared quantized projection tiles over all six encodings.
 *
 * A checked synchronous tile borrows canonical resident weights, persistent Q8
 * activations and exclusive output spans. Physical rows share integer payload
 * decode; no serial-row replay, repacking, decoded matrix, workspace allocation
 * or source-weight mirror is used. Serial/grouped execution owns the same K
 * program. Expert policy retains the common ordered sixteen-partition tree;
 * ordinary ARM projections retain one increasing-K accumulator per column.
 */
#pragma once

#include "CPUNativeQuantisedMultiScaleKernels.h"
#include "CPUNativeQuantisedWeightView.h"
#include "utils/ARMCPUFeatures.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised
{
#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_PROJECTION_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_PROJECTION_EXACT
#endif

    /** @brief Closed synchronous binding for up to four independently accumulated rows.
     * @tparam Rows Physical row tile in [1,4], independent of logical request depth.
     */
    template <std::size_t Rows> class CPUNativeNEONProjectionTile final
    {
        static_assert(Rows >= 1 && Rows <= 4);
        using Encoding = CPUNativeQuantisedEncoding;
        using Dot = multi_scale::NEONByteDot;
        /** @brief Exact source nibble interpretation, selected outside K traversal. */
        enum class NibbleMap { Centered, Unsigned, NonLinear };

    public:
        /**
         * @brief Validate one native tile before reading/writing any projection data.
         * @param weights Published immutable matrix view; its external owner retains storage.
         * @param activation Complete persistent Q8 K row for each physical row.
         * @param output Exact live output chunk per row, with exclusive writable bytes.
         * @param chunk Physical 64-column chunk within the matrix.
         * @return Checked synchronous tile; the owners retain all borrowed allocations.
         * @throws std::invalid_argument For missing storage, shape, policy or alias conflicts.
         */
        [[nodiscard]] static CPUNativeNEONProjectionTile bind(CPUNativeQuantisedWeightView weights,
            const std::array<std::span<const Q8_1Block>, Rows> &activation,
            const std::array<std::span<float>, Rows> &output, int chunk)
        {
            // The compiler envelope is admitted even when the caller deliberately
            // selects ASIMD in a DotProd build: auto-generated code may use SDOT.
            ARMCPUCapabilities::native().require(ARMCPUFeature::ASIMD);
#if defined(__ARM_FEATURE_DOTPROD)
            ARMCPUCapabilities::native().require(ARMCPUFeature::DotProd);
#endif
            if (chunk < 0 || chunk >= weights.N_padded / 64)
                throw std::invalid_argument("Native NEON projection chunk is outside its matrix");
            const int columns = std::min(64, weights.N - chunk * 64);
            const int partitions = weights.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert
                ? MoEProjectionNumericalContract::orderedKPartitionsForWidth(weights.K) : 1;
            if (partitions <= 0)
                throw std::invalid_argument("Native NEON expert projection requires the canonical aligned K tree");
            for (std::size_t row = 0; row < Rows; ++row)
            {
                if (!activation[row].data() || activation[row].size() != std::size_t(weights.blocks_per_row) ||
                    !output[row].data() || output[row].size() != std::size_t(columns))
                    throw std::invalid_argument("Native NEON projection requires exact live activation/output spans");
            }
            for (std::size_t row = 0; row < Rows; ++row)
            {
                if (overlaps(output[row].data(), output[row].size_bytes(), weights.bytes().data(), weights.bytes().size()))
                    throw std::invalid_argument("Native NEON projection output aliases immutable prepared weights");
                for (std::size_t other = 0; other < Rows; ++other)
                {
                    if (overlaps(output[row].data(), output[row].size_bytes(),
                                 activation[other].data(), activation[other].size_bytes()))
                        throw std::invalid_argument("Native NEON projection output aliases live quantized input");
                    if (other < row && overlaps(output[row].data(), output[row].size_bytes(),
                                               output[other].data(), output[other].size_bytes()))
                        throw std::invalid_argument("Native NEON projection rows require exclusive output bytes");
                }
            }
            return CPUNativeNEONProjectionTile(weights, activation, output, chunk, columns, partitions);
        }

        /**
         * @brief Execute one explicitly selected native integer instruction family.
         * @tparam Instruction ASIMD widening or separately compiled/admitted DotProd.
         * @throws std::invalid_argument For an unsupported source/encoding/policy tuple.
         * @details Selection occurs once per tile, never inside a byte-dot loop.
         * Corrupt source data is fatal, not transactional rollback: earlier
         * vectors may have been authored. Only a successful complete operation
         * may be published by the outer CPU execution/coherence owner.
         */
        template <Dot Instruction = Dot::ASIMD> void execute() const
        {
            if constexpr (Instruction == Dot::DotProd || Instruction == Dot::I8MM)
                ARMCPUCapabilities::native().require(ARMCPUFeature::DotProd);
            if constexpr (Instruction == Dot::I8MM)
                ARMCPUCapabilities::native().require(ARMCPUFeature::IntegerMatMul);
            executeWith<multi_scale::NEONMultiScaleOps<leafInstruction<Instruction>()>>();
        }

        /**
         * @brief Execute with one explicit native operation set.
         * @tparam Ops NEONMultiScaleOps of an admitted family, or a test type
         * derived from one that observes its integer leaf calls.
         * @details execute() is the production entry and admits the hardware;
         * callers of this overload must have admitted Ops' instructions.
         * @throws std::invalid_argument For an unsupported source/encoding/policy tuple.
         */
        template <typename Ops> void executeWith() const
        {
            if (weights_.usesCompactMultiScale())
            {
                std::array<const Q8_1Block *, Rows> input;
                std::array<float *, Rows> output;
                for (std::size_t row = 0; row < Rows; ++row)
                {
                    input[row] = activation_[row].data();
                    output[row] = output_[row].data();
                }
                if (weights_.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                    return multi_scale::dispatchOrdered<4, Rows, Ops>(weights_, input, output,
                        chunk_, weights_.blocks_per_row, partitions_, blocks_per_partition_, columns_);
                return multi_scale::dispatch<4, Rows, Ops>(weights_, input, output,
                    chunk_, 0, weights_.blocks_per_row, false, columns_);
            }
            switch (weights_.numerical_policy)
            {
            case CPUProjectionNumericalPolicy::GPUAlignedExpert:
                return dispatch<Ops, CPUProjectionNumericalPolicy::GPUAlignedExpert>();
            case CPUProjectionNumericalPolicy::BackendNative:
                return dispatch<Ops, CPUProjectionNumericalPolicy::BackendNative>();
            }
            throw std::invalid_argument("Native NEON projection requires a known arithmetic policy");
        }

        /**
         * @brief Resolve the exact leaf instruction family one tile executes.
         * @tparam Instruction Admitted family requested for the whole projection.
         * @return The requested family, except that a lone I8MM row is the SDOT leaf.
         * @details SMMLA needs two rows. A single-row tile (M=1 decode or the
         * odd last row) therefore compiles the very DotProd instantiation, so
         * it emits no SMMLA and is byte- and code-identical to serial decode.
         */
        template <Dot Instruction> static constexpr Dot leafInstruction() noexcept
        { return Instruction == Dot::I8MM && Rows == 1 ? Dot::DotProd : Instruction; }

    private:
        /// Two-row SMMLA accumulator sets per tile; at least one keeps the array type valid.
        static constexpr std::size_t Pairs = Rows / 2 == 0 ? 1 : Rows / 2;

        /** @brief Seal borrowed spans and a validated arithmetic tree.
         * @param weights Checked immutable matrix publication.
         * @param activation Complete Q8 rows.
         * @param output Exclusive live output spans.
         * @param chunk Physical output chunk.
         * @param columns Live columns in this chunk.
         * @param partitions Canonical arithmetic partition count.
         */
        CPUNativeNEONProjectionTile(CPUNativeQuantisedWeightView weights,
            const std::array<std::span<const Q8_1Block>, Rows> &activation,
            const std::array<std::span<float>, Rows> &output, int chunk, int columns, int partitions)
            : weights_(weights), activation_(activation), output_(output), chunk_(chunk), columns_(columns),
              partitions_(partitions), blocks_per_partition_((weights.blocks_per_row + partitions - 1) / partitions) {}

        /** @brief Test byte-range overlap without forming overflowing end addresses.
         * @param a First byte root.
         * @param a_bytes First live byte extent.
         * @param b Second byte root.
         * @param b_bytes Second live byte extent.
         * @return Whether the two nonempty bounded storage ranges intersect.
         */
        static bool overlaps(const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes) noexcept
        {
            if (a_bytes == 0 || b_bytes == 0) return false;
            const auto first = reinterpret_cast<std::uintptr_t>(a);
            const auto second = reinterpret_cast<std::uintptr_t>(b);
            return first <= second ? second - first < a_bytes : first - second < b_bytes;
        }

        /** @brief Select source map and secondary-word semantics before traversal.
         * @tparam Ops Admitted native integer instruction type.
         * @tparam Policy Immutable complete projection arithmetic identity.
         * @throws std::invalid_argument For a source/encoding mismatch.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy> void dispatch() const
        {
            switch (weights_.encoding)
            {
            case Encoding::ExpandedInt8:
                if (weights_.is_asymmetric)
                    return run<Ops, Policy, Encoding::ExpandedInt8, NibbleMap::Centered, true>();
                return run<Ops, Policy, Encoding::ExpandedInt8>();
            case Encoding::IQ3SExpandedSourceScale:
                return run<Ops, Policy, Encoding::IQ3SExpandedSourceScale>();
            case Encoding::Q6KNativeDualScale:
                return run<Ops, Policy, Encoding::Q6KNativeDualScale>();
            case Encoding::NibbleLUTSourceScale:
                return run<Ops, Policy, Encoding::NibbleLUTSourceScale, NibbleMap::NonLinear>();
            case Encoding::NibbleLUT:
                switch (weights_.codebook_id)
                {
                case 0: return run<Ops, Policy, Encoding::NibbleLUT, NibbleMap::Centered>();
                case 4: return run<Ops, Policy, Encoding::NibbleLUT, NibbleMap::NonLinear>();
                case 5: return run<Ops, Policy, Encoding::NibbleLUT, NibbleMap::Unsigned, true>();
                }
                break;
            case Encoding::CompactMultiScale:
                break; // The shared compact implementation was selected before this switch.
            }
            throw std::invalid_argument("Native NEON projection has no exact source/encoding implementation");
        }

        /** @brief Decode four columns' source nibbles entirely in native registers.
         * @tparam Map Centered linear, unsigned linear or canonical non-linear grid.
         * @param index Four consecutive nibble indices per column, each in [0,15].
         * @return Sixteen original signed source integers.
         */
        template <NibbleMap Map> static int8x16_t nibbles(uint8x16_t index) noexcept
        {
            if constexpr (Map == NibbleMap::NonLinear)
                return vqtbl1q_s8(vld1q_s8(kvalues_iq4nl_i8), index);
            else if constexpr (Map == NibbleMap::Centered)
                return vsubq_s8(vreinterpretq_s8_u8(index), vdupq_n_s8(8));
            else
                return vreinterpretq_s8_u8(index);
        }

        /** @brief Add one shared signed-byte weight vector to every independent row dot.
         * @tparam Ops Selected native instruction implementation.
         * @param weights Four signed source bytes for each of four output columns.
         * @param block Logical K block.
         * @param group Four-value activation group in [0,7].
         * @param dots Exact independent INT32 accumulators, updated without saturation.
         */
        template <typename Ops> void consume(int8x16_t weights, int block, int group,
            typename Ops::I (&dots)[Rows]) const noexcept
        {
#pragma GCC unroll 4
            for (std::size_t row = 0; row < Rows; ++row)
            {
                int word = 0;
                std::memcpy(&word, activation_[row][block].qs + group * 4, sizeof(word));
                dots[row] = Ops::dot(dots[row], std::bit_cast<typename Ops::I>(weights), typename Ops::I{} + word);
            }
        }

        /**
         * @brief Add two adjacent decoded groups to every row with row-pair matrix dots.
         * @tparam Ops Admitted I8MM operation set.
         * @param first Four signed source bytes per column for @p group.
         * @param second The same columns' bytes for @p group + 1.
         * @param block Logical K block.
         * @param group Even first group in [0,6]; both groups lie in one scale half.
         * @param columns01 Exact SMMLA column-0/1 sums, one per row pair.
         * @param columns23 Exact SMMLA column-2/3 sums, one per row pair.
         * @param dots Single-row sums; only the odd last row is updated here, with SDOT.
         */
        template <typename Ops> void consumePair(int8x16_t first, int8x16_t second, int block, int group,
            typename Ops::I (&columns01)[Pairs], typename Ops::I (&columns23)[Pairs],
            typename Ops::I (&dots)[Rows]) const noexcept
        {
            using I = typename Ops::I;
#pragma GCC unroll 2
            for (std::size_t pair = 0; pair < Rows / 2; ++pair)
                Ops::pairDot(columns01[pair], columns23[pair],
                    Ops::activationPair(activation_[2 * pair][block].qs + group * 4,
                                        activation_[2 * pair + 1][block].qs + group * 4),
                    std::bit_cast<I>(first), std::bit_cast<I>(second));
            if constexpr (Rows % 2 != 0)
            {
                int word0 = 0, word1 = 0;
                std::memcpy(&word0, activation_[Rows - 1][block].qs + group * 4, sizeof(word0));
                std::memcpy(&word1, activation_[Rows - 1][block].qs + group * 4 + 4, sizeof(word1));
                dots[Rows - 1] = Ops::dot(dots[Rows - 1], std::bit_cast<I>(first), I{} + word0);
                dots[Rows - 1] = Ops::dot(dots[Rows - 1], std::bit_cast<I>(second), I{} + word1);
            }
        }

        /** @brief Add completed row-pair sums into their rows' exact INT32 block dots.
         * @tparam Ops Admitted I8MM operation set.
         * @param columns01 Completed column-0/1 sums per pair.
         * @param columns23 Completed column-2/3 sums per pair.
         * @param dots Per-row INT32 sums for the same block half.
         */
        template <typename Ops> static void publishPairs(const typename Ops::I (&columns01)[Pairs],
            const typename Ops::I (&columns23)[Pairs], typename Ops::I (&dots)[Rows]) noexcept
        {
#pragma GCC unroll 2
            for (std::size_t pair = 0; pair < Rows / 2; ++pair)
            {
                typename Ops::I row0, row1;
                Ops::splitPair(columns01[pair], columns23[pair], row0, row1);
                dots[2 * pair] += row0;
                dots[2 * pair + 1] += row1;
            }
        }

        /** @brief Expand the selected sixteen Q6_K high-plane bits into source byte positions.
         * @param mask Four high bits per column, packed in increasing source order.
         * @param bit Original six-bit bit-position value, either 16 or 32.
         * @return Sixteen independent high-bit source bytes.
         */
        static uint8x16_t q6Plane(std::uint16_t mask, std::uint8_t bit) noexcept
        {
            const uint8x16_t positions{1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
            const auto raw = vcombine_u8(vdup_n_u8(static_cast<std::uint8_t>(mask)),
                vdup_n_u8(static_cast<std::uint8_t>(mask >> 8)));
            return vandq_u8(vtstq_u8(raw, positions), vdupq_n_u8(bit));
        }

        /** @brief Decode one Q6_K group's six-bit source integers for four columns.
         * @param unit Prepared interleaved unit of the current block.
         * @param column First of four output columns.
         * @param group Four-value group in [0,7].
         * @return Sixteen centered signed source bytes.
         */
        static int8x16_t q6Group(const std::uint8_t *unit, int column, int group) noexcept
        {
            const auto raw = vld1q_u8(unit + (group % 4) * 256 + column * 4);
            auto values = group < 4 ? vandq_u8(raw, vdupq_n_u8(15)) : vshrq_n_u8(raw, 4);
            const auto *planes = unit + 1024 + group * 64 + (column / 16) * 16;
            std::uint64_t bits0 = 0, bits1 = 0;
            std::memcpy(&bits0, planes, sizeof(bits0));
            std::memcpy(&bits1, planes + 8, sizeof(bits1));
            const int shift = (column % 16) * 4;
            values = vorrq_u8(values, q6Plane(static_cast<std::uint16_t>(bits0 >> shift), 16));
            values = vorrq_u8(values, q6Plane(static_cast<std::uint16_t>(bits1 >> shift), 32));
            return vsubq_s8(vreinterpretq_s8_u8(values), vdupq_n_s8(32));
        }

        /**
         * @brief Compute one block's exact INT32 dots for every row with row-pair SMMLA.
         * @tparam Ops Admitted I8MM operation set.
         * @tparam Physical Non-compact execution encoding.
         * @tparam Map Nibble decoder for nibble encodings.
         * @param unit Prepared interleaved unit of the current block.
         * @param column First of four output columns.
         * @param block Logical K block.
         * @param low Receives K0..31 dots, or the Q6_K low-scale half K0..15.
         * @param high Receives the Q6_K high-scale half K16..31; otherwise untouched.
         * @details Groups are decoded with the same expressions as the single-row
         * walk and consumed as adjacent pairs that never straddle a scale half.
         * The FP32 block program that follows is shared and unchanged.
         */
        template <typename Ops, Encoding Physical, NibbleMap Map>
        void pairBlock(const std::uint8_t *unit, int column, int block,
            typename Ops::I (&low)[Rows], typename Ops::I (&high)[Rows]) const noexcept
        {
            using I = typename Ops::I;
            I low01[Pairs]{}, low23[Pairs]{};
            if constexpr (Physical == Encoding::Q6KNativeDualScale)
            {
                I high01[Pairs]{}, high23[Pairs]{};
                // Two decoded groups live at a time, the pairwise analogue of
                // the single-row walk's one-group bound against panel spills.
#pragma GCC unroll 1
                for (int group = 0; group < 8; group += 2)
                {
                    const auto first = q6Group(unit, column, group);
                    const auto second = q6Group(unit, column, group + 1);
                    if (group < 4) consumePair<Ops>(first, second, block, group, low01, low23, low);
                    else consumePair<Ops>(first, second, block, group, high01, high23, high);
                }
                publishPairs<Ops>(high01, high23, high);
            }
            else if constexpr (Physical == Encoding::NibbleLUT || Physical == Encoding::NibbleLUTSourceScale)
            {
                // Low nibbles of byte groups q and q+1 are K groups q and q+1;
                // their high nibbles are K groups q+4 and q+5.
#pragma GCC unroll 2
                for (int group = 0; group < 4; group += 2)
                {
                    const auto raw0 = vld1q_u8(unit + group * 256 + column * 4);
                    const auto raw1 = vld1q_u8(unit + (group + 1) * 256 + column * 4);
                    consumePair<Ops>(nibbles<Map>(vandq_u8(raw0, vdupq_n_u8(15))),
                        nibbles<Map>(vandq_u8(raw1, vdupq_n_u8(15))), block, group, low01, low23, low);
                    consumePair<Ops>(nibbles<Map>(vshrq_n_u8(raw0, 4)),
                        nibbles<Map>(vshrq_n_u8(raw1, 4)), block, group + 4, low01, low23, low);
                }
            }
            else
            {
#pragma GCC unroll 4
                for (int group = 0; group < 8; group += 2)
                    consumePair<Ops>(vld1q_s8(reinterpret_cast<const std::int8_t *>(unit + group * 256 + column * 4)),
                        vld1q_s8(reinterpret_cast<const std::int8_t *>(unit + (group + 1) * 256 + column * 4)),
                        block, group, low01, low23, low);
            }
            publishPairs<Ops>(low01, low23, low);
        }

        /**
         * @brief Execute the source's exact block program with a fixed serial/grouped K tree.
         * @tparam Ops Admitted integer instruction implementation.
         * @tparam Policy Backend-native FMA or separately rounded expert program.
         * @tparam Physical One of the five non-compact execution encodings.
         * @tparam Map Nibble decoder, used only by nibble encodings.
         * @tparam WithMinimum Whether the source owns an additive minimum vector.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical,
                  NibbleMap Map = NibbleMap::Centered, bool WithMinimum = false>
        LLAMINAR_NEON_PROJECTION_EXACT void run() const
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            using I = typename Ops::I;
            using F = typename Ops::F;
            constexpr bool nibble = Physical == Encoding::NibbleLUT || Physical == Encoding::NibbleLUTSourceScale;
            for (int column = 0; column < columns_; column += 4)
            {
                float32x4_t reduced[Rows]{};
                for (int partition = 0; partition < partitions_; ++partition)
                {
                    float32x4_t accumulator[Rows]{};
                    const int first = partition * blocks_per_partition_;
                    const int last = std::min(first + blocks_per_partition_, weights_.blocks_per_row);
                    for (int block = first; block < last; ++block)
                    {
                        const auto *unit = weights_.interleavedBase() +
                            (std::size_t(chunk_) * weights_.blocks_per_row + block) * weights_.interleaved_block_stride;
                        I low[Rows]{}, high[Rows]{};
                        if constexpr (multi_scale::usesRowPairs<Ops>)
                            pairBlock<Ops, Physical, Map>(unit, column, block, low, high);
                        else if constexpr (Physical == Encoding::Q6KNativeDualScale)
                        {
                            // Keep one decoded group live at a time. Fully
                            // unrolling all eight groups lets the compiler
                            // hoist a complete four-row activation panel into
                            // registers and spill it inside the K traversal.
#pragma GCC unroll 1
                            for (int group = 0; group < 8; ++group)
                            {
                                const auto centered = q6Group(unit, column, group);
                                if (group < 4) consume<Ops>(centered, block, group, low);
                                else consume<Ops>(centered, block, group, high);
                            }
                        }
                        else if constexpr (nibble)
                        {
#pragma GCC unroll 4
                            for (int group = 0; group < 4; ++group)
                            {
                                const auto raw = vld1q_u8(unit + group * 256 + column * 4);
                                consume<Ops>(nibbles<Map>(vandq_u8(raw, vdupq_n_u8(15))), block, group, low);
                                consume<Ops>(nibbles<Map>(vshrq_n_u8(raw, 4)), block, group + 4, low);
                            }
                        }
                        else
                        {
#pragma GCC unroll 8
                            for (int group = 0; group < 8; ++group)
                                consume<Ops>(vld1q_s8(reinterpret_cast<const std::int8_t *>(unit + group * 256 + column * 4)),
                                    block, group, low);
                        }
                        F scale0 = Ops::half(weights_.chunkScales(chunk_, block) + column);
                        F scale1{};
                        if constexpr (Physical == Encoding::Q6KNativeDualScale)
                        {
                            const I metadata = Ops::words(weights_.chunkMins(chunk_, block) + column);
                            const I low_coefficient = (metadata & 127) - (metadata & 128);
                            const I high_coefficient = ((metadata >> 8) & 127) - ((metadata >> 8) & 128);
                            scale1 = scale0 * __builtin_convertvector(high_coefficient, F);
                            scale0 = scale0 * __builtin_convertvector(low_coefficient, F);
                        }
                        else if constexpr (Physical == Encoding::IQ3SExpandedSourceScale || Physical == Encoding::NibbleLUTSourceScale)
                        {
                            const I metadata = Ops::words(weights_.chunkMins(chunk_, block) + column);
                            constexpr std::uint32_t maximum = Physical == Encoding::IQ3SExpandedSourceScale ? 15 : 63;
                            if (!Ops::metadataFits(metadata, maximum))
                                throw std::invalid_argument("Native NEON source-scale metadata contains reserved bits");
                            const I coefficient = Physical == Encoding::IQ3SExpandedSourceScale ? metadata * 2 + 1 : metadata - 32;
                            scale0 = scale0 * __builtin_convertvector(coefficient, F);
                        }
                        else if constexpr (WithMinimum)
                            scale1 = Ops::half(weights_.chunkMins(chunk_, block) + column);
#pragma GCC unroll 4
                        for (std::size_t row = 0; row < Rows; ++row)
                        {
                            const auto &a = activation_[row][block];
                            const float activation_scale = nativeQuantisedFP16ScaleToFP32(a.d);
                            if constexpr (Physical == Encoding::Q6KNativeDualScale)
                            {
                                const F term0 = scale0 * __builtin_convertvector(low[row], F);
                                const F term1 = scale1 * __builtin_convertvector(high[row], F);
                                const F contribution = std::bit_cast<F>(vdupq_n_f32(activation_scale)) * (term0 + term1);
                                accumulator[row] = vaddq_f32(accumulator[row], std::bit_cast<float32x4_t>(contribution));
                            }
                            else if constexpr (WithMinimum)
                                accumulator[row] = accumulateCorrectedSingleScaleBlockNEON(accumulator[row],
                                    std::bit_cast<int32x4_t>(low[row]), std::bit_cast<float32x4_t>(scale0), a.sum_qs,
                                    std::bit_cast<float32x4_t>(scale1), activation_scale, Policy);
                            else
                                accumulator[row] = accumulateSingleScaleBlockNEON(accumulator[row],
                                    std::bit_cast<int32x4_t>(low[row]), std::bit_cast<float32x4_t>(scale0), activation_scale, Policy);
                        }
                    }
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
                        reduced[row] = partition == 0 ? accumulator[row] : vaddq_f32(reduced[row], accumulator[row]);
                }
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
                {
                    if (column + 4 <= columns_)
                        vst1q_f32(output_[row].data() + column, reduced[row]);
                    else
                    {
                        // Scalar lane stores are the bounded final-column tail,
                        // not a row replay. Never expose the accumulator array's
                        // address to memcpy: that forces all full-column partials
                        // into stack slots even when the tail branch is unused.
                        auto *destination = output_[row].data() + column;
                        vst1q_lane_f32(destination, reduced[row], 0);
                        if (columns_ - column >= 2) vst1q_lane_f32(destination + 1, reduced[row], 1);
                        if (columns_ - column == 3) vst1q_lane_f32(destination + 2, reduced[row], 2);
                    }
                }
            }
        }

        CPUNativeQuantisedWeightView weights_; ///< Borrowed publication; no physical storage is duplicated.
        std::array<std::span<const Q8_1Block>, Rows> activation_; ///< Retained complete quantized input rows.
        std::array<std::span<float>, Rows> output_; ///< Exclusive live output spans.
        int chunk_; ///< Checked physical output chunk.
        int columns_; ///< Exact logical output width of the chunk.
        int partitions_; ///< M/thread-independent arithmetic tree.
        int blocks_per_partition_; ///< Increasing K interval owned by each partial.
    };
#undef LLAMINAR_NEON_PROJECTION_EXACT
} // namespace llaminar2::cpu::native_quantised
