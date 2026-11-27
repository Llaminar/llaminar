/**
 * @file CPUNativeNEONPreparedProjection.h
 * @brief Native AArch64 prepared quantized projection tiles over all admitted encodings.
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
#include "CPUNativeNEONRowDecomposition.h"
#include "CPUNativeQuantisedWeightView.h"
#include "tensors/MXFP4Format.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "utils/ARMCPUFeatures.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <utility>
#include <span>
#include <stdexcept>
#include <string_view>

namespace llaminar2::cpu::native_quantised
{
#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_PROJECTION_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#define LLAMINAR_NEON_PROJECTION_CODED __attribute__((optimize("fp-contract=off", "no-associative-math", "no-thread-jumps")))
#define LLAMINAR_NEON_PROJECTION_EXACT_UNSCHEDULED \
    __attribute__((optimize("fp-contract=off", "no-associative-math", "no-schedule-insns")))
#define LLAMINAR_NEON_PROJECTION_CODED_UNSCHEDULED \
    __attribute__((optimize("fp-contract=off", "no-associative-math", "no-thread-jumps", "no-schedule-insns")))
#else
#define LLAMINAR_NEON_PROJECTION_EXACT
#define LLAMINAR_NEON_PROJECTION_CODED
#define LLAMINAR_NEON_PROJECTION_EXACT_UNSCHEDULED
#define LLAMINAR_NEON_PROJECTION_CODED_UNSCHEDULED
#endif
// The exact programs listed in kUnthreadedPrograms compile like the coded programs (no-thread-jumps).
#define LLAMINAR_NEON_PROJECTION_EXACT_UNTHREADED LLAMINAR_NEON_PROJECTION_CODED
#define LLAMINAR_NEON_PROJECTION_EXACT_UNSCHEDULED_UNTHREADED LLAMINAR_NEON_PROJECTION_CODED_UNSCHEDULED

    class CPUNativeNEONProjectionBatch;

    /**
     * @brief Running-sum addition of one numerical policy.
     * @tparam Policy GPU-aligned experts add with addOrderedNEON() (earlier NaN wins);
     *         backend-native programs keep their unchanged plain addition.
     * @param earlier Running sum of the earlier blocks or partitions.
     * @param later This block's contribution or this partition's partial.
     * @return earlier + later with one rounding.
     */
    template <CPUProjectionNumericalPolicy Policy>
    [[gnu::always_inline]] inline float32x4_t foldNEON(float32x4_t earlier, float32x4_t later) noexcept
    {
        if constexpr (Policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            return addOrderedNEON(earlier, later);
        else
            return vaddq_f32(earlier, later);
    }

    /**
     * @brief Seed of an ordered (GPU-aligned) partition reduction.
     * @return Negative zero in every lane, folded with partition 0 like every later partition.
     * @details Under the admitted FPCR profile (RNE, FZ = 0, DN = 0, no AFP;
     * cpu::ARMFloatingPointProfile) `-0 + a` is bitwise `a` for every operand
     * that is not a signaling NaN, and a partition partial never is one: it is
     * +0 or the output of an FP arithmetic instruction. Folding from this seed
     * therefore yields exactly the bits of starting from partition 0's partial,
     * with the same ordered-add instruction per fold, but without a
     * data-dependent first-partition branch that GCC threads into a second
     * cycle of the K walk.
     */
    [[gnu::always_inline]] inline float32x4_t orderedPartitionSeedNEON() noexcept { return vdupq_n_f32(-0.0f); }

    /** @brief Closed synchronous binding for up to four independently accumulated rows.
     * @tparam Rows Physical row tile in [1,4], independent of logical request depth.
     */
    template <std::size_t Rows> class CPUNativeNEONProjectionTile final
    {
        static_assert(Rows >= 1 && Rows <= 4);
        using Encoding = CPUNativeQuantisedEncoding;
        using Dot = multi_scale::NEONByteDot;
        /** @brief Exact source nibble interpretation, selected outside K traversal.
         *
         * MXFP4 is a distinct map (doubled OCP E2M1 magnitudes with a sign bit),
         * never IQ4_NL's grid, although both are sixteen-entry signed tables.
         * Prepared MXFP4 stores the native E8M0 scale bytes.
         */
        enum class NibbleMap { Centered, Unsigned, NonLinear, MXFP4 };
        using ScaleWords = CPUNativeQuantisedScaleEncoding;

        /** @brief Whether this nibble map uses E8M0 scales. */
        template <NibbleMap Map> static constexpr bool kE8M0Scales = Map == NibbleMap::MXFP4;

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
            // The whole compiler envelope is admitted even when the caller
            // deliberately selects a narrower leaf: auto-generated code may use
            // any instruction of the envelope (SDOT, or SMMLA in an I8MM build).
            ARMCPUCapabilities::native().requireCompiledEnvelope();
            requireExecutable(weights);
            if (chunk < 0 || chunk >= weights.N_padded / 64)
                throw std::invalid_argument("Native NEON projection chunk is outside its matrix");
            const int columns = std::min(64, weights.N - chunk * 64);
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
            return admitted(weights, activation, output, chunk);
        }

        /**
         * @brief Reject a publication this tile family cannot execute, before any team exists.
         * @param weights Sealed publication; its metadata was admitted by the seal.
         * @throws std::invalid_argument For an unknown policy, an encoding/codebook
         *         without a native implementation, or a non-canonical expert K tree.
         * @details After this check every dispatch branch taken by execution is
         * implemented, so the execution path itself cannot throw.
         */
        static void requireExecutable(const CPUNativeQuantisedWeightView &weights)
        {
            if (weights.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert &&
                weights.numerical_policy != CPUProjectionNumericalPolicy::BackendNative)
                throw std::invalid_argument("Native NEON projection requires a known arithmetic policy");
            if (partitionsFor(weights) <= 0)
                throw std::invalid_argument("Native NEON expert projection requires the canonical aligned K tree");
            bool implemented = false;
            switch (weights.encoding)
            {
            case Encoding::Q5NativeSingleScale:
            case Encoding::SignedInt8SingleScale:
            case Encoding::Q6KNativeDualScale:
            case Encoding::Q4KSuperblockCoded:
            case Encoding::NibbleLUTSourceScale:
                implemented = true;
                break;
            case Encoding::NibbleLUT:
                implemented = weights.codebook_id == 0 || weights.codebook_id == 4 || weights.codebook_id == 5 ||
                    (weights.codebook_id == kNativeQuantisedMXFP4Codebook &&
                     weights.scaleEncoding() == ScaleWords::E8M0);
                break;
            case Encoding::CompactMultiScale:
                switch (weights.codebook_id)
                {
                case 7: case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16: case 17: implemented = true; break;
                }
                break;
            }
            if (!implemented)
                throw std::invalid_argument("Native NEON projection has no exact source/encoding implementation");
        }

        /**
         * @brief Execute one explicitly selected native integer instruction family.
         * @tparam Instruction ASIMD widening or separately compiled/admitted DotProd.
         * @throws std::runtime_error If the machine lacks the requested family.
         * @details Selection occurs once per tile, never inside a byte-dot loop.
         * bind() admitted the publication, so the arithmetic itself cannot fail.
         * The calling thread's FPCR must be the admitted profile
         * (cpu::ARMFloatingPointProfile); any other profile is fatal, because
         * the staged factors and the FP32 contribution program round under it.
         */
        template <Dot Instruction = Dot::ASIMD> void execute() const
        {
            ARMFloatingPointProfile::require("Native NEON quantized projection");
            if constexpr (Instruction == Dot::DotProd || Instruction == Dot::I8MM)
                ARMCPUCapabilities::native().require(ARMCPUFeature::DotProd);
            if constexpr (Instruction == Dot::I8MM)
                ARMCPUCapabilities::native().require(ARMCPUFeature::IntegerMatMul);
            executeAdmitted<Instruction>();
        }

        /**
         * @brief Execute with one explicit native operation set.
         * @tparam Ops NEONMultiScaleOps of an admitted family, or a test type
         * derived from one that observes its integer leaf calls.
         * @tparam Strips Panel width in strips for tournaments/tests; 0 selects the installed width.
         * @details Admits the complete compiler envelope itself, so an Ops type
         * cannot run instructions the machine lacks whichever entry is used.
         * @throws std::runtime_error If the compiled envelope is not supported.
         */
        template <typename Ops, int Strips = 0> void executeWith() const
        {
            // Every entry, including an observer seam, admits the full envelope
            // and the calling thread's floating-point profile.
            ARMCPUCapabilities::native().requireCompiledEnvelope();
            ARMFloatingPointProfile::require("Native NEON quantized projection");
            executeWithAdmitted<Ops, Strips>();
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

        /**
         * @brief Installed panel width, in four-column strips, for one row tile, family and encoding.
         * @tparam Ops Admitted operation set.
         * @tparam Physical Non-compact execution encoding of the program.
         * @tparam WithMinimum Whether the program applies an additive minimum.
         * @return For the single-row SDOT leaf (DotProd, and the I8MM family's
         * lone row): four for signed Q8, whose native byte payload needs no
         * inline decode; one (the strip walk) for native Q5 and IQ4_XS; two for
         * the remaining formats. Wider panels are profitable only when their
         * live vectors fit the register budget: the native Q5 fifth-bit decode
         * and the IQ4_XS non-linear table plus its source-scale reconstruction
         * keep too many vectors live for two strips, so GCC 14 spilled the
         * panel's partition totals (a decode program must not spill). The strip
         * walk evaluates the same per-column decode, INT32 group order, FP32
         * block program and partition tree, so the bytes are unchanged. The
         * remaining entries were measured in the GB10 panel tournament.
         */
        template <typename Ops, Encoding Physical, bool WithMinimum = false> static constexpr int panelStrips() noexcept
        {
            if constexpr (multi_scale::usesRowPairs<Ops>) return 1;
            else if constexpr (Rows != 1 || Ops::instruction == multi_scale::NEONByteDot::ASIMD) return 1;
            else if constexpr (Physical == Encoding::SignedInt8SingleScale)
                return 4;
            else if constexpr (Physical == Encoding::Q5NativeSingleScale || Physical == Encoding::NibbleLUTSourceScale ||
                               Physical == Encoding::Q4KSuperblockCoded)
                return 1;
            else return 2;
        }

    private:
        // Only the batch binding, which admits CPU features, publication,
        // policy, K tree and every task's spans before forming a team, may build
        // and run tiles without per-task checks.
        template <std::size_t> friend class CPUNativeNEONProjectionTile;
        friend struct CPUNativeNEONProjectionWideOracle;
        friend class CPUNativeNEONProjectionBatch;
        friend class CPUNativeNEONPackedPrefill;

        /**
         * @brief Execute an admitted family inside a worker team.
         * @tparam Instruction Family whose CPU features the caller has already admitted.
         * @details No check can fail here: publication, policy, encoding and
         * geometry were admitted before the team, and metadata by the seal.
         */
        template <Dot Instruction, int Strips = 0> void executeAdmitted() const noexcept
        {
            executeWithAdmitted<multi_scale::NEONMultiScaleOps<leafInstruction<Instruction>()>, Strips>();
        }

        /**
         * @brief Construct a tile whose publication and geometry an enclosing binding admitted.
         * @param weights Executable sealed publication (requireExecutable passed).
         * @param activation Complete Q8 rows of the bound invocation.
         * @param output Exclusive live output spans of the bound invocation.
         * @param chunk Physical chunk inside the matrix.
         * @return Tile ready for executeAdmitted(); no check is repeated per task.
         */
        [[nodiscard, gnu::always_inline]] static inline CPUNativeNEONProjectionTile admitted(CPUNativeQuantisedWeightView weights,
            const std::array<std::span<const Q8_1Block>, Rows> &activation,
            const std::array<std::span<float>, Rows> &output, int chunk) noexcept
        {
            return CPUNativeNEONProjectionTile(weights, activation, output, chunk,
                std::min(64, weights.N - chunk * 64), partitionsFor(weights));
        }

        /** @return Canonical arithmetic partition count; zero for an invalid expert K tree. */
        static constexpr int partitionsFor(const CPUNativeQuantisedWeightView &weights) noexcept
        {
            return weights.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert
                ? MoEProjectionNumericalContract::orderedKPartitionsForWidth(weights.K) : 1;
        }

        /** @brief Run one admitted tile with an explicit operation set; cannot fail. */
        template <typename Ops, int Strips = 0> void executeWithAdmitted() const noexcept
        {
            if (weights_.usesCompactMultiScale() && weights_.codebook_id != 7 && weights_.codebook_id != 11)
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
            if (weights_.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                return dispatch<Ops, CPUProjectionNumericalPolicy::GPUAlignedExpert, Strips>();
            return dispatch<Ops, CPUProjectionNumericalPolicy::BackendNative, Strips>();
        }

        /// Two-row SMMLA accumulator sets per tile; at least one keeps the array type valid.
        static constexpr std::size_t Pairs = Rows / 2 == 0 ? 1 : Rows / 2;

        /** @brief Seal borrowed spans and a validated arithmetic tree.
         * @param weights Checked immutable matrix publication.
         * @param activation Complete Q8 rows.
         * @param output Exclusive live output spans.
         * @param chunk Physical output chunk.
         * @param columns Live columns in this chunk.
         * @param partitions Canonical partition count; requireOrderedPartitionWidth() admits the plan (cold, fatal).
         */
        [[gnu::always_inline]] CPUNativeNEONProjectionTile(CPUNativeQuantisedWeightView weights,
            const std::array<std::span<const Q8_1Block>, Rows> &activation,
            const std::array<std::span<float>, Rows> &output, int chunk, int columns, int partitions)
            : weights_(weights), activation_(activation), output_(output), chunk_(chunk), columns_(columns),
              partitions_(partitions), blocks_per_partition_(requireOrderedPartitionWidth(0, weights.blocks_per_row, std::size_t(weights.blocks_per_row), partitions)) {}

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
         * @pre requireExecutable() admitted the publication.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, int Strips = 0> void dispatch() const noexcept
        {
            switch (weights_.encoding)
            {
            case Encoding::SignedInt8SingleScale:
                return dispatchRows<Ops, Policy, Encoding::SignedInt8SingleScale, NibbleMap::Centered, false, Strips>();
            case Encoding::Q5NativeSingleScale:
                if (weights_.is_asymmetric)
                    return dispatchRows<Ops, Policy, Encoding::Q5NativeSingleScale, NibbleMap::Unsigned, true, Strips>();
                return dispatchRows<Ops, Policy, Encoding::Q5NativeSingleScale, NibbleMap::Centered, false, Strips>();
            case Encoding::Q6KNativeDualScale:
                return dispatchRows<Ops, Policy, Encoding::Q6KNativeDualScale, NibbleMap::Centered, false, Strips>();
            case Encoding::Q4KSuperblockCoded:
                return dispatchRows<Ops, Policy, Encoding::Q4KSuperblockCoded, NibbleMap::Unsigned, true, Strips>();
            case Encoding::NibbleLUTSourceScale:
                return dispatchRows<Ops, Policy, Encoding::NibbleLUTSourceScale, NibbleMap::NonLinear, false, Strips>();
            case Encoding::NibbleLUT:
                switch (weights_.codebook_id)
                {
                case 0: return dispatchRows<Ops, Policy, Encoding::NibbleLUT, NibbleMap::Centered, false, Strips>();
                case 4: return dispatchRows<Ops, Policy, Encoding::NibbleLUT, NibbleMap::NonLinear, false, Strips>();
                case 5: return dispatchRows<Ops, Policy, Encoding::NibbleLUT, NibbleMap::Unsigned, true, Strips>();
                case kNativeQuantisedMXFP4Codebook:
                    if (weights_.scaleEncoding() != ScaleWords::E8M0)
                        break;
                    return dispatchRows<Ops, Policy, Encoding::NibbleLUT, NibbleMap::MXFP4, false, Strips>();
                }
                break;
            case Encoding::CompactMultiScale:
                if (weights_.codebook_id == 7)
                    return dispatchCodedRows<Ops, Policy, 7, Strips>();
                if (weights_.codebook_id == 11)
                    return dispatchCodedRows<Ops, Policy, 11, Strips>();
                break; // The shared compact implementation was selected before this switch.
            }
            // requireExecutable() rejected every other tuple before the team.
            __builtin_unreachable();
        }

        /**
         * @brief Borrow exactly one contiguous live row slice for a physical subprogram.
         * @tparam Count Number of rows consumed, never padded.
         * @tparam Offset First logical row of the slice.
         * @return A smaller tile with the identical column and K-partition identity.
         */
        template <std::size_t Count, std::size_t Offset>
        [[gnu::always_inline]] inline auto rowSlice() const noexcept
        {
            static_assert(Offset + Count <= Rows);
            std::array<std::span<const Q8_1Block>, Count> activation;
            std::array<std::span<float>, Count> output;
            for (std::size_t row = 0; row < Count; ++row)
            {
                activation[row] = activation_[Offset + row];
                output[row] = output_[Offset + row];
            }
            return CPUNativeNEONProjectionTile<Count>::admitted(weights_, activation, output, chunk_);
        }

        /** @brief Select a first-class row decomposition before instantiating a non-coded K program. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical, NibbleMap Map,
                  bool Minimum, int Strips>
        [[gnu::always_inline]] inline void dispatchRows() const noexcept
        {
            if constexpr (decomposeNEONRows<Rows, Ops::instruction, Policy, Physical, int(Map), Minimum>())
            {
                constexpr auto family = decomposedNEONPairFamily<Rows, Ops::instruction, Policy,
                    Physical, int(Map), Minimum>();
                using PairOps = std::conditional_t<family == Ops::instruction, Ops,
                    multi_scale::NEONMultiScaleOps<family>>;
                constexpr auto tail = decomposedNEONPairFamily<Rows, Ops::instruction, Policy,
                    Physical, int(Map), Minimum, true>();
                using LastOps = std::conditional_t<tail == Ops::instruction, Ops,
                    multi_scale::NEONMultiScaleOps<tail>>;
                if constexpr (Rows <= 2)
                    return run<PairOps, Policy, Physical, Map, Minimum, Strips>();
                else
                {
                    using FirstMap = typename CPUNativeNEONProjectionTile<(Rows > 2 ? 2 : Rows)>::NibbleMap;
                    using LastMap = typename CPUNativeNEONProjectionTile<Rows - 2>::NibbleMap;
                    rowSlice<2, 0>().template run<PairOps, Policy, Physical, FirstMap(Map), Minimum, Strips>();
                    rowSlice<Rows - 2, 2>().template run<LastOps, Policy, Physical, LastMap(Map), Minimum, Strips>();
                }
            }
            else
                run<Ops, Policy, Physical, Map, Minimum, Strips>();
        }

        /** @brief Select existing coded row pairs, with SDOT pairs for ordered four-row I8MM IQ3. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, int Codebook, int Strips>
        [[gnu::always_inline]] inline void dispatchCodedRows() const noexcept
        {
            if constexpr (decomposeNEONRows<Rows, Ops::instruction, Policy,
                          Encoding::CompactMultiScale, Codebook, Codebook == 7>())
            {
                // I8MM's two-row IQ3 program is not admitted by the stack gate;
                // SDOT computes the identical per-row integer dot and FP program.
                constexpr auto family = decomposedNEONPairFamily<Rows, Ops::instruction, Policy,
                    Encoding::CompactMultiScale, Codebook, Codebook == 7>();
                using PairOps = std::conditional_t<family == Ops::instruction, Ops,
                    multi_scale::NEONMultiScaleOps<family>>;
                constexpr auto tail = decomposedNEONPairFamily<Rows, Ops::instruction, Policy,
                    Encoding::CompactMultiScale, Codebook, Codebook == 7, true>();
                using LastOps = std::conditional_t<tail == Ops::instruction, Ops,
                    multi_scale::NEONMultiScaleOps<tail>>;
                if constexpr (Rows <= 2)
                    return runCoded<PairOps, Policy, Codebook, Strips>();
                else
                {
                    rowSlice<2, 0>().template runCoded<PairOps, Policy, Codebook, Strips>();
                    rowSlice<Rows - 2, 2>().template runCoded<LastOps, Policy, Codebook, Strips>();
                }
            }
            else
                runCoded<Ops, Policy, Codebook, Strips>();
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
            else if constexpr (Map == NibbleMap::MXFP4)
                return vqtbl1q_s8(vld1q_s8(mxfp4::kValues), index);
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

        /** @brief Decode one Q6_K group's six-bit source integers for four columns (I8MM pair block only).
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

        /** @brief Merge four columns' contiguous Q5_K high-plane slice into the dot bytes.
         * @param low Sixteen unpacked low nibbles in dot-lane order.
         * @param plane Two payload bytes holding four high bits per column. The
         *        single-row reader may also read six adjacent bytes in the unit.
         * @tparam LoadLowHalf Use a scalar SIMD load for the single-row SDOT schedule.
         * @return Sixteen complete unsigned five-bit source values.
         * @details The lookup reads only SIMD bytes 0 and 1, so the decode leaf
         * can load a low SIMD half and avoid broadcasting a halfword before TBL.
         * Retain the existing load schedule for the multi-row and widening leaves.
         */
        template <bool LoadLowHalf = false>
        [[gnu::always_inline]] static uint8x16_t q5KMerge(uint8x16_t low, const std::uint8_t *plane) noexcept
        {
            const uint8x16_t select{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
            uint8x16_t source;
            if constexpr (LoadLowHalf)
            {
                // Every plane slice belongs to a sealed unit with the code
                // bytes immediately after the plane, including the last slice
                // of the last column strip. TBL ignores bytes 2 through 15.
                source = vcombine_u8(vld1_u8(plane), vdup_n_u8(0));
            }
            else
            {
                std::uint16_t packed;
                std::memcpy(&packed, plane, sizeof(packed));
                source = vreinterpretq_u8_u16(vdupq_n_u16(packed));
            }
            const auto bits = vqtbl1q_u8(source, select);
            const auto mask = vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201ull));
            return vbslq_u8(vdupq_n_u8(16), vtstq_u8(bits, mask), low);
        }

        /** @brief Decode one Q5_K strip's K block once, sharing each vector across all activation rows.
         * @param unit Current borrowed prepared unit.
         * @param column First of four output columns.
         * @param block Logical K block.
         * @param dots Receives the whole-block integer dot for each row.
         */
        template <typename Ops>
        [[gnu::always_inline]] void q5KBlock(const std::uint8_t *unit, int column, int block,
            typename Ops::I (&dots)[Rows]) const noexcept
        {
            using I = typename Ops::I;
            const std::uint8_t *nibbles = unit + column * 4;
            const std::uint8_t *planes = unit + 1024 + column / 2;
            // The three-row tile uses one shared SDOT group rather than keeping
            // row-pair and odd-row accumulators live through both group decodes.
            if constexpr (multi_scale::usesRowPairs<Ops> && Rows != 3)
            {
                I columns01[Pairs]{}, columns23[Pairs]{};
                // Visit one eight-K pair at a time. A uniform slice avoids
                // duplicating the row-pair state for the low and high K halves.
#pragma GCC unroll 1
                for (int group = 0; group < 8; group += 2)
                {
                    const auto sh = vdupq_n_s8(static_cast<std::int8_t>(-(group & 4)));
                    const auto a = vshlq_u8(vld1q_u8(nibbles + (group & 2) * 256), sh);
                    const auto b = vshlq_u8(vld1q_u8(nibbles + ((group & 2) + 1) * 256), sh);
                    consumePair<Ops>(vreinterpretq_s8_u8(q5KMerge(vandq_u8(a, vdupq_n_u8(15)), planes + group * 32)),
                        vreinterpretq_s8_u8(q5KMerge(vandq_u8(b, vdupq_n_u8(15)), planes + (group + 1) * 32)),
                        block, group, columns01, columns23, dots);
                }
                publishPairs<Ops>(columns01, columns23, dots);
            }
            else if constexpr (Ops::instruction == Dot::DotProd && Rows == 1)
            {
                static_assert(preparedInterleavedBlockStride(Encoding::CompactMultiScale, true, 7) >= 1024 + 256 + 6);
                // Read the activation once and keep the two independent dot
                // chains until the exact integer sum is consumed by FP32.
                const auto a = Ops::activationBlock(activation_[0][block].qs);
                I lo{}, hi{};
                forEachGroup<4>([&]<int Group>() __attribute__((always_inline)) {
                    const auto raw = vld1q_u8(nibbles + Group * 256);
                    const auto low = q5KMerge<true>(vandq_u8(raw, vdupq_n_u8(15)), planes + Group * 32);
                    const auto high = q5KMerge<true>(vshrq_n_u8(raw, 4), planes + (Group + 4) * 32);
                    lo = Ops::template dotLane<Group>(lo, std::bit_cast<I>(low), a);
                    hi = Ops::template dotLane<Group + 4>(hi, std::bit_cast<I>(high), a);
                    // Retire this group's decode before the next group's loads: GCC
                    // otherwise decodes all eight groups at once and spills.
                    asm("" : "+w"(lo), "+w"(hi), "+r"(nibbles), "+r"(planes));
                });
                dots[0] = lo + hi;
            }
            else if constexpr (Rows >= 3)
            {
                // The wider non-paired tiles retire one decoded group before
                // the next load; the two-half walk spills an accumulator.
#pragma GCC unroll 1
                for (int group = 0; group < 8; ++group)
                {
                    const auto raw = vld1q_u8(nibbles + (group & 3) * 256);
                    const auto low = group < 4 ? vandq_u8(raw, vdupq_n_u8(15)) : vshrq_n_u8(raw, 4);
                    consume<Ops>(vreinterpretq_s8_u8(q5KMerge(low, planes + group * 32)), block, group, dots);
                }
            }
            else
            {
#pragma GCC unroll 1
                for (int group = 0; group < 4; ++group)
                {
                    const auto raw = vld1q_u8(nibbles + group * 256);
                    consume<Ops>(vreinterpretq_s8_u8(q5KMerge(vandq_u8(raw, vdupq_n_u8(15)), planes + group * 32)),
                        block, group, dots);
                    consume<Ops>(vreinterpretq_s8_u8(q5KMerge(vshrq_n_u8(raw, 4), planes + (group + 4) * 32)),
                        block, group + 4, dots);
                }
            }
        }

        /**
         * @brief Build two adjacent IQ3_S groups' nine-bit indices together.
         * @tparam Pair Eight-value K pair in [0,3].
         * @tparam Ops Admitted SIMD operations for four columns.
         * @param qs Four bytes per column, holding two pairs' low index bits.
         * @param qh Column high bytes duplicated into the high byte of each
         *        halfword, with the second group's bits pre-shifted by one.
         * @return Four column indices for the first group, then four for the second.
         * @details Use one byte shuffle for both groups, reusing its constant
         * across pairs. One bit select supplies bit 8 without widening each
         * group's indices or keeping two independent index vectors live.
         */
        template <int Pair, typename Ops>
        [[gnu::always_inline]] static uint16x8_t iq3PairIndices(typename Ops::I qs, uint16x8_t qh) noexcept
        {
            const uint8x16_t pick{0, 255, 4, 255, 8, 255, 12, 255, 1, 255, 5, 255, 9, 255, 13, 255};
            if constexpr (Pair % 2 != 0)
                qs = std::bit_cast<typename Ops::I>(vshrq_n_u32(std::bit_cast<uint32x4_t>(qs), 16));
            const auto indices = vreinterpretq_u16_u8(vqtbl1q_u8(std::bit_cast<uint8x16_t>(qs), pick));
            if constexpr (Pair != 0)
                qh = vshrq_n_u16(qh, 2 * Pair);
            return vbslq_u16(vdupq_n_u16(256), qh, indices);
        }

        /**
         * @brief Decode one four-value group of a paired IQ3_S lookup.
         * @tparam Half First or second group, 0 or 1.
         * @tparam Ops Admitted SIMD operations for four columns.
         * @param indices Register-resident halfword indices from iq3PairIndices().
         * @param signs The source sign byte repeated across each column's four lanes.
         * @return Signed source bytes in production dot-lane order.
         * @details The immutable 512-word grid is larger than a NEON TBL
         * table. Look up each column's word directly and apply its four sign
         * bits with the existing SIMD mask, without a scratch gather buffer.
         */
        template <int Half, typename Ops>
        [[gnu::always_inline]] static int8x16_t iq3PairGroup(uint16x8_t indices, typename Ops::I signs) noexcept
        {
            constexpr int lane = 4 * Half;
            const auto packed = uint32x4_t{
                iq3s_grid[vgetq_lane_u16(indices, lane)], iq3s_grid[vgetq_lane_u16(indices, lane + 1)],
                iq3s_grid[vgetq_lane_u16(indices, lane + 2)], iq3s_grid[vgetq_lane_u16(indices, lane + 3)]};
            return std::bit_cast<int8x16_t>(Ops::template signGrid<Half>(std::bit_cast<typename Ops::I>(packed), signs));
        }

        /** @brief Decode one IQ3_S K block, sharing each decoded vector across all live rows.
         * @tparam Ops Admitted SIMD operations for four columns.
         * @param unit Current native payload.
         * @param column First of four output columns.
         * @param block Logical K block.
         * @param dots Receives each row's whole-block exact integer dot.
         */
        template <typename Ops>
        [[gnu::always_inline]] void iq3sBlock(const std::uint8_t *unit, int column, int block,
            typename Ops::I (&dots)[Rows]) const noexcept
        {
            using I = typename Ops::I;
            // Recompute IQ3_S plane offsets for this K block rather than retaining
            // loop-invariant derived addresses across the decode-heavy walk.
            if constexpr (Rows != 4 || Ops::instruction == Dot::ASIMD)
                asm("" : "+r"(column) : "r"(block));
            const auto metadata = multi_scale::payloadWord<11, Ops>(unit, 2, column);
            // Align both groups' first high bits to bit 8; each pair then
            // shifts by 2 * Pair with an immediate rather than a live mask.
            const uint8x16_t qh_gather{255, 0, 255, 4, 255, 8, 255, 12, 255, 0, 255, 4, 255, 8, 255, 12};
            const int16x8_t align_groups{0, 0, 0, 0, -1, -1, -1, -1};
            [[maybe_unused]] const auto packed_qh = vshlq_u16(
                vreinterpretq_u16_u8(vqtbl1q_u8(std::bit_cast<uint8x16_t>(metadata), qh_gather)), align_groups);
            const auto signsFor = [&]<int Pair>() __attribute__((always_inline)) {
                if constexpr (Pair == 3)
                {
                    // The last sign bytes are the one-byte tail plane, not
                    // part of the interleaved metadata words. Duplicate its
                    // contiguous four-byte slice directly into the vector file.
                    const auto tail = vreinterpretq_u8_u32(vld1q_dup_u32(reinterpret_cast<const std::uint32_t *>(
                        unit + preparedCompactPayloadWords(11) * 256 + column)));
                    const uint8x16_t select{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
                    return std::bit_cast<I>(vqtbl1q_u8(tail, select));
                }
                else
                    return Ops::template repeatByte<1 + Pair>(metadata);
            };
            if constexpr (multi_scale::usesRowPairs<Ops>)
            {
                I columns01[Pairs]{}, columns23[Pairs]{};
                if constexpr (Rows >= 3)
                {
                    // Three- and four-row tiles bound decode live ranges with
                    // a two-pair unroll, retaining independent row dot chains
                    // and reducing spill pressure on outer FP accumulators.
#pragma GCC unroll 2
                    for (int group = 0; group < 8; group += 2)
                        consumePair<Ops>(std::bit_cast<int8x16_t>(multi_scale::decodeIQ3SGroup<4, Ops>(unit, column, group)),
                            std::bit_cast<int8x16_t>(multi_scale::decodeIQ3SGroup<4, Ops>(unit, column, group + 1)),
                            block, group, columns01, columns23, dots);
                }
                else
                {
                    forEachGroup<4>([&]<int Pair>() __attribute__((always_inline)) {
                        constexpr int group = Pair * 2;
                        const auto qs = multi_scale::payloadWord<11, Ops>(unit, Pair / 2, column);
                        const auto signs = signsFor.template operator()<Pair>();
                        const auto indices = iq3PairIndices<Pair, Ops>(qs, packed_qh);
                        consumePair<Ops>(iq3PairGroup<0, Ops>(indices, signs),
                            iq3PairGroup<1, Ops>(indices, signs), block, group, columns01, columns23, dots);
                    });
                }
                publishPairs<Ops>(columns01, columns23, dots);
            }
            else if constexpr (Ops::instruction == Dot::DotProd && Rows == 1)
            {
                const auto a = Ops::activationBlock(activation_[0][block].qs);
                I even{}, odd{};
                forEachGroup<4>([&]<int Pair>() __attribute__((always_inline)) {
                    constexpr int group = Pair * 2;
                    const auto qs = multi_scale::payloadWord<11, Ops>(unit, Pair / 2, column);
                    const auto signs = signsFor.template operator()<Pair>();
                    const auto indices = iq3PairIndices<Pair, Ops>(qs, packed_qh);
                    even = Ops::template dotLane<group>(even, std::bit_cast<I>(iq3PairGroup<0, Ops>(indices, signs)), a);
                    odd = Ops::template dotLane<group + 1>(odd, std::bit_cast<I>(iq3PairGroup<1, Ops>(indices, signs)), a);
                    // Retire this pair before the next pair's loads (see q5KBlock()).
                    asm("" : "+w"(even), "+w"(odd), "+r"(unit));
                });
                dots[0] = even + odd;
            }
            else
            {
                // Baseline ASIMD widens each group. Retire it immediately
                // before moving to the next to bound live product vectors.
#pragma GCC unroll 1
                for (int group = 0; group < 8; ++group)
                    consume<Ops>(std::bit_cast<int8x16_t>(multi_scale::decodeIQ3SGroup<4, Ops>(unit, column, group)),
                        block, group, dots);
            }
        }

        /** @brief Select the binary16 Q5_K factor program on tiles with room for its constants.
         * @tparam Ops Admitted NEON instruction family.
         * @return True for SDOT tiles of up to three rows and for SMMLA row pairs
         *         under a binary16-arithmetic compiler envelope. Four-row SDOT
         *         and ASIMD use the widened program to avoid vector spills.
         */
        template <typename Ops> static constexpr bool q5KUsesHalfStaging() noexcept
        {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            return multi_scale::usesRowPairs<Ops> || (Ops::instruction == Dot::DotProd && Rows < 4);
#else
            return false;
#endif
        }

        /** @brief Load a strip's original source factors once per partition/superblock intersection.
         * @tparam Codebook Q5_K (7) or IQ3_S (11).
         * @tparam Ops Admitted native lane operations.
         * @param header Sealed source superblock header owning the current K block.
         * @param column First of four consecutive column headers.
         * @return Q5_K's Q4_K-compatible staging base, or widened source d for IQ3_S.
         */
        template <int Codebook, typename Ops>
        [[gnu::always_inline]] static auto codedBase(const std::uint8_t *header, int column) noexcept
        {
            if constexpr (Codebook == 7)
            {
                if constexpr (q5KUsesHalfStaging<Ops>())
                    return q4KStripBase<Ops>(header, column);
                else
                {
                    using F = typename Ops::F;
                    const auto *words = reinterpret_cast<const std::uint16_t *>(header) + column * 2;
                    if constexpr (Rows == 1)
                    {
                        // Deinterleave the four (d, dmin) words in registers: the 1-row
                        // tile materialises a vld2 structure through the stack.
                        const uint16x8_t packed = vld1q_u16(words);
                        return coded_metadata::Q4KBaseFactors<F>{
                            std::bit_cast<F>(neon_fp16::widen(vget_low_u16(vuzp1q_u16(packed, packed)))),
                            std::bit_cast<F>(neon_fp16::widen(vget_low_u16(vuzp2q_u16(packed, packed))))};
                    }
                    else
                    {
                        const auto sources = vld2_u16(words);
                        return coded_metadata::Q4KBaseFactors<F>{std::bit_cast<F>(neon_fp16::widen(sources.val[0])),
                            std::bit_cast<F>(neon_fp16::widen(sources.val[1]))};
                    }
                }
            }
            else
            {
                static_assert(Codebook == 11);
                return Ops::half(reinterpret_cast<const std::uint16_t *>(header) + column);
            }
        }

        /** @brief Apply the block's coded factors to its already widened source superblock header.
         * @tparam Codebook Q5_K (7) or IQ3_S (11).
         * @tparam Ops Admitted native lane operations.
         * @tparam Base Widened source factor type returned by codedBase().
         * @param unit Current prepared unit including its block code plane.
         * @param base Source factors shared by this strip's current K interval.
         * @param column First of four consecutive coded columns.
         * @return Exact previous FP32 factors: staged scale/minimum for Q5_K,
         *         or source d times the odd IQ3_S coefficient without staging.
         */
        template <int Codebook, typename Ops, typename Base>
        [[gnu::always_inline]] static coded_metadata::Factors<typename Ops::F> codedFactors(
            const std::uint8_t *unit, const Base &base, int column) noexcept
        {
            constexpr auto data_stride = preparedDataStride(Encoding::CompactMultiScale, Codebook);
            if constexpr (Codebook == 7)
            {
                // Q4_K and Q5_K store the same 6+6 code and use the same
                // signed minimum and binary16 staging. Share the exact
                // binary16 multiply without changing either payload reader.
                if constexpr (q5KUsesHalfStaging<Ops>())
                    return q4KStripFactors<Ops>(unit + data_stride, column, base);
                else
                {
                    using L = coded_metadata::Lanes<Ops>;
                    return coded_metadata::reconstruct<5, Ops>(L::template code<7>(unit + data_stride, column), base);
                }
            }
            else
            {
                static_assert(Codebook == 11);
                // Four consecutive nibbles fit a single halfword load. A
                // register shift/extract builds the four original odd factors.
                std::uint16_t packed;
                std::memcpy(&packed, unit + data_stride + column / 2, sizeof(packed));
                const auto spread = vshlq_u32(vdupq_n_u32(packed), int32x4_t{0, -4, -8, -12});
                const auto nibble = vandq_u32(spread, vdupq_n_u32(15));
                const auto coefficient = vorrq_u32(vshlq_n_u32(nibble, 1), vdupq_n_u32(1));
                return {.primary = base * std::bit_cast<typename Ops::F>(vcvtq_f32_u32(coefficient))};
            }
        }

        /**
         * @brief Whether a per-block superblock-coded walk streams units and hoists its source header.
         * @tparam Ops Admitted operation set.
         * @tparam Physical Prepared encoding.
         * @return True for the single-row SDOT IQ4_XS tile. The multi-row tiles
         * address each block's unit and header from the constant geometry in one
         * flat block loop: a superblock loop around their blocks keeps the
         * bounds, unit and header addresses live through every block and spills
         * the column and partition state there. The ASIMD widening walk keeps
         * per-block header addressing too: its widened products already occupy
         * the vector file. Q6_K keeps its per-block addressing for the same reason.
         */
        template <typename Ops, Encoding Physical> static constexpr bool hoistsSourceHeader() noexcept
        { return Physical == Encoding::NibbleLUTSourceScale && Ops::instruction == Dot::DotProd && Rows == 1; }

        /**
         * @brief Whether a coded tile walks its K blocks flat, widening the header factors per block.
         * @tparam Coded Coded source codebook (Q5_K 7 or IQ3_S 11).
         * @tparam Ops Admitted integer instruction family.
         * @tparam Ordered Whether the tile runs the GPU-aligned partition tree.
         * @return True for Q5_K except the backend-native four-row SMMLA tile; IQ3_S and that tile widen once
         *         per superblock intersection because their decode leaves no register to widen
         *         the header again in every block.
         */
        template <int Coded, typename Ops, bool Ordered> static constexpr bool flatCodedWalk() noexcept
        { return Coded == 7 && !(multi_scale::usesRowPairs<Ops> && Rows == 4 && !Ordered); }

        /** @brief Visit increasing-K blocks without recomputing addresses or header factors in the K loop.
         * @tparam Physical Prepared encoding.
         * @tparam Coded Codebook of a coded source, or zero for the generic per-block tile.
         * @tparam Ops Admitted SIMD family.
         * @tparam Ordered Whether the tile runs the GPU-aligned partition tree (flatCodedWalk()).
         * @param first First block of the current ordered partition.
         * @param last Exclusive end of the current ordered partition.
         * @param column First lane of the four-column strip.
         * @param visit Always-inlined consumer of (block, unit, base): the block's source
         *        superblock header address for compact coded sources and when
         *        hoistsSourceHeader(), or nullptr (Q6_K and the other IQ4_XS walks derive
         *        their header from the unit after the decode).
         * @details The coded walk is one flat, branch-free increasing-K loop. Nested
         * superblock/block loops and per-superblock reloads let GCC thread the
         * empty-partition exit into a second cycle of the walk, which the stack
         * witness's loop roles read as a nested partition loop.
         */
        template <Encoding Physical, int Coded, typename Ops, bool Ordered = false, typename Visit>
        [[gnu::always_inline]] void forPartitionBlocks(int first, int last, [[maybe_unused]] int column, Visit &&visit) const noexcept
        {
            if constexpr (Coded == 0 && hoistsSourceHeader<Ops, Physical>())
            {
                // IQ4_XS: per-block payload and code planes, one source header
                // per superblock. Address the header and the first unit once per
                // partition/superblock intersection and stream the units; the
                // visitor receives the header so its source half is loaded and
                // widened outside the per-block work.
                // The admitted publication's stride and header width are the
                // encoding's constants; building the geometry from them keeps the
                // address arithmetic scalar.
                constexpr std::size_t stride = preparedInterleavedBlockStride(Physical, false, 0);
                const CPUNativeQuantisedBlockGeometry geometry{.block_stride = stride,
                    .header_bytes = preparedSuperblockHeaderBytes(Physical, 0),
                    .blocks_per_row = std::size_t(weights_.blocks_per_row)};
                const auto *root = weights_.interleavedBase() + std::size_t(chunk_) * geometry.rowBytes();
                for (int block = first; block < last; )
                {
                    const int end = std::min(last, static_cast<int>(geometry.superblockEnd(block)));
                    const auto *unit = root + geometry.blockOffset(0, block);
                    const auto *header = root + geometry.headerOffset(0, block);
                    for (; block < end; ++block, unit += stride)
                        visit(block, unit, header);
                }
            }
            else if constexpr (Coded == 0)
            {
                for (int block = first; block < last; ++block)
                {
                    const auto *unit = preparedSuperblockCoded(Physical)
                        ? weights_.interleavedBase() + codedGeometry<Physical>(weights_.blocks_per_row).blockOffset(std::size_t(chunk_), std::size_t(block))
                        : weights_.interleavedBase() +
                            (std::size_t(chunk_) * weights_.blocks_per_row + block) * weights_.interleaved_block_stride;
                    visit(block, unit, nullptr);
                }
            }
            else
            {
                constexpr auto stride = preparedInterleavedBlockStride(Encoding::CompactMultiScale, Coded == 7, Coded);
                const CPUNativeQuantisedBlockGeometry geometry{
                    .block_stride = stride,
                    .header_bytes = preparedSuperblockHeaderBytes(Encoding::CompactMultiScale, Coded),
                    .blocks_per_row = std::size_t(weights_.blocks_per_row)};
                // Tile binding already admitted the resident geometry. A wire
                // intersection is never passed directly to this execution ABI.
                const auto *root = weights_.interleavedBase() + std::size_t(chunk_) * geometry.rowBytes();
                if constexpr (flatCodedWalk<Coded, Ops, Ordered>())
                {
                    // Q5_K: one flat, branch-free increasing-K walk. Each block
                    // addresses its unit and source superblock header from the
                    // constant geometry (resident phase 0); the visitor widens the
                    // header factors after the block's integer decode.
                    for (int block = first; block < last; ++block)
                        visit(block, root + geometry.blockOffset(0, block), root + geometry.headerOffset(0, block));
                }
                else
                {
                    // IQ3_S: one header per superblock intersection, widened once outside
                    // its blocks; its decode leaves no register to widen it again in every
                    // block.
                    for (int block = first; block < last; )
                    {
                        const int end = std::min(last, static_cast<int>(geometry.superblockEnd(block)));
                        const auto *unit = root + geometry.blockOffset(0, block);
                        const auto base = codedBase<Coded, Ops>(root + geometry.headerOffset(0, block), column);
                        for (; block < end; ++block, unit += stride)
                            visit(block, unit, base);
                    }
                }
            }
        }

        /** @brief Decode both K halves of four columns from one pair of Q5 bitplanes.
         * @tparam Centered Subtract sixteen for Q5_0; Q5_1 uses unsigned source values.
         * @param unit Prepared 64-column unit.
         * @param column First of four output columns.
         * @param group Nibble group in [0,3] covering K groups @p group and @p group + 4.
         * @return The two signed-byte vectors consumed by the exact integer dot.
         * @details One table lookup broadcasts each column's fifth-bit byte to its
         * four value lanes; one bit test per half selects the lane's bit. Q5_0
         * tests the complemented byte, so a clear fifth bit yields an all-ones
         * lane whose upper nibble 0xF is inserted above the nibble value:
         * `n | 0xF0` is exactly `n - 16` as a signed byte, and a set fifth bit
         * leaves `n = (n + 16) - 16`. Q5_1 inserts bit 4 with one bitwise
         * select. Both produce the original signed source bytes.
         * @tparam GroupFifth Load the fifth bits from this group's byte, so one lane
         *         table serves every group (q5GroupFifth()); otherwise load group 0's
         *         bytes and add the group to the lane table. Same selected bits.
         */
        template <bool Centered, bool GroupFifth = false> [[gnu::always_inline]] static inline std::array<int8x16_t, 2>
        ordinaryQ5Groups(const std::uint8_t *unit, int column, int group) noexcept
        {
            const auto raw = vld1q_u8(unit + Q5NativePayload::nibbleOffset(group, column));
            // The 16-byte read from a group's fifth-bit byte stays inside the sealed
            // unit: the per-block scales follow the payload.
            auto fifth = vld1q_u8(unit + Q5NativePayload::fifthOffset(GroupFifth ? group : 0, column));
            if constexpr (Centered)
                fifth = vmvnq_u8(fifth);
            const uint8x16_t lane{0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12};
            const uint8x16_t positions{1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8};
            const auto bits = GroupFifth ? vqtbl1q_u8(fifth, lane)
                : vqtbl1q_u8(fifth, vaddq_u8(lane, vdupq_n_u8(static_cast<std::uint8_t>(group))));
            const auto low_selected = vtstq_u8(bits, positions);
            const auto high_selected = vtstq_u8(bits, vshlq_n_u8(positions, 4));
            if constexpr (Centered)
            {
                // Insert the all-ones/zero lane as the upper nibble: under the
                // high nibble shifted down (SRI, raw stays intact), then over the
                // low nibble kept in place (SLI, raw's last use), without
                // separately extracting either nibble.
                const auto high_value = vsriq_n_u8(high_selected, raw, 4);
                const auto low_value = vsliq_n_u8(raw, low_selected, 4);
                return {vreinterpretq_s8_u8(low_value), vreinterpretq_s8_u8(high_value)};
            }
            else
                return {vreinterpretq_s8_u8(vbslq_u8(vdupq_n_u8(16), low_selected, vandq_u8(raw, vdupq_n_u8(15)))),
                        vreinterpretq_s8_u8(vbslq_u8(vdupq_n_u8(16), high_selected, vshrq_n_u8(raw, 4)))};
        }

        /**
         * @brief Whether a Q5 program reads each group's fifth bits from that group's byte.
         * @tparam Ops Admitted integer instruction family.
         * @tparam Ordered Whether the tile runs the GPU-aligned partition tree.
         * @tparam WithMinimum Whether the source is Q5_1 (additive minimum).
         * @return The addressing measured to need fewer live constants for this program
         *         with the GCC 14 tile witness; both select the same bits.
         */
        template <typename Ops, bool Ordered, bool WithMinimum> static constexpr bool q5GroupFifth() noexcept
        {
            constexpr bool pairs = multi_scale::usesRowPairs<Ops>;
            if constexpr (Rows == 4)
                return !(pairs && !Ordered && !WithMinimum);
            else if constexpr (Rows == 3)
                return Ordered && WithMinimum && !pairs;
            else
                return Rows == 2 && pairs && !Ordered && WithMinimum;
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
        template <typename Ops, Encoding Physical, NibbleMap Map, bool Ordered = false>
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
            else if constexpr (Physical == Encoding::Q5NativeSingleScale)
            {
#pragma GCC unroll 1
                for (int group = 0; group < 4; group += 2)
                {
                    constexpr bool fifth = q5GroupFifth<Ops, Ordered, Map == NibbleMap::Unsigned>();
                    const auto first = ordinaryQ5Groups<Map == NibbleMap::Centered, fifth>(unit, column, group);
                    const auto second = ordinaryQ5Groups<Map == NibbleMap::Centered, fifth>(unit, column, group + 1);
                    consumePair<Ops>(first[0], second[0], block, group, low01, low23, low);
                    consumePair<Ops>(first[1], second[1], block, group + 4, low01, low23, low);
                }
            }
            else if constexpr (Physical == Encoding::NibbleLUT || Physical == Encoding::NibbleLUTSourceScale ||
                               Physical == Encoding::Q4KSuperblockCoded)
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

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        /// Q4_K superblock factors held across one superblock's blocks (binary16 d and -dmin).
        template <typename Ops> using Q4KBase = coded_metadata::neon_half::Q4KHalfBase;
#else
        /// Q4_K superblock factors held across one superblock's blocks (widened d and dmin).
        template <typename Ops> using Q4KBase = coded_metadata::Q4KBaseFactors<typename Ops::F>;
#endif

        /**
         * @brief Load the reusable Q4_K superblock factors of one four-column strip.
         * @param header First byte of the 64-column superblock header.
         * @param column First of four columns.
         * @return Factors reused by every block of the superblock intersection.
         */
        template <typename Ops>
        [[gnu::always_inline]] static Q4KBase<Ops> q4KStripBase(const std::uint8_t *header, int column) noexcept
        {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            return coded_metadata::neon_half::q4KHalfBase(header, column);
#else
            using L = coded_metadata::Lanes<Ops>;
            return coded_metadata::q4KBase<Ops>(L::template superblockWords<5>(header, column));
#endif
        }

        /**
         * @brief Reconstruct one block's staged Q4_K scale and minimum for one strip.
         * @param codes First byte of the block's 12-bit code planes.
         * @param column First of four columns.
         * @param base Superblock factors from q4KStripBase().
         * @return Exactly the widened per-block words of reconstruct<5>().
         * @details FP16-arithmetic envelopes (NEON_I8MM) compute both factors
         * with one binary16 multiply, which is the codec's single RNE rounding
         * of the exact binary32 product (proven exhaustively by
         * V2_Unit_CPUCodedCompactUnits); the baseline envelopes keep the
         * binary32 program of reconstruct<5>().
         */
        template <typename Ops>
        [[gnu::always_inline]] static coded_metadata::Factors<typename Ops::F> q4KStripFactors(
            const std::uint8_t *codes, int column, const Q4KBase<Ops> &base) noexcept
        {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            float32x4_t scale, minimum;
            coded_metadata::neon_half::q4KFactors(codes, column, base, scale, minimum);
            return {.primary = std::bit_cast<typename Ops::F>(scale), .secondary = std::bit_cast<typename Ops::F>(minimum)};
#else
            using L = coded_metadata::Lanes<Ops>;
            return coded_metadata::reconstruct<5, Ops>(L::template code<5>(codes, column), base);
#endif
        }

        /** @brief Q4_K panel walk with source d/dmin hoisted above its block dots.
         * @tparam Ops Selected signed SDOT/SMMLA/ASIMD family.
         * @tparam Policy Immutable FP32 contribution and K-fold policy.
         * @tparam S Four-column strips per panel (installed: 2 for the single-row SDOT leaf).
         *         Row-pair SMMLA tiles always walk one strip.
         * @details Partitions may split a source superblock. Each strip loads
         * exactly one header for each partition intersection, then streams this
         * intersection's contiguous payloads and 12-bit codes. The final
         * ragged superblock ends at the same geometry-owned header as packing.
         * A panel loads each row's activation block once and feeds every strip
         * with lane-indexed dots; per column, the integer dot, the factor
         * expressions, the per-block FP32 program, block order and partition
         * tree are those of the one-strip walk. Strips in the chunk's sealed
         * padding columns are computed but never stored.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, int S>
        [[gnu::always_inline]] LLAMINAR_NEON_PROJECTION_EXACT inline void runQ4K() const noexcept
        {
            static_assert(S == 1 || ((S == 2 || S == 4) && !multi_scale::usesRowPairs<Ops>));
            using I = typename Ops::I;
            constexpr int superblock_blocks = native_quantised_superblock::kBlocksPerSuperblock;
            constexpr int data_bytes = preparedDataStride(Encoding::Q4KSuperblockCoded, 5);
            constexpr int block_bytes = preparedInterleavedBlockStride(Encoding::Q4KSuperblockCoded, true, 5);
            const CPUNativeQuantisedBlockGeometry geometry{
                .block_stride = block_bytes, .header_bytes = preparedSuperblockHeaderBytes(Encoding::Q4KSuperblockCoded, 5),
                .blocks_per_row = std::size_t(weights_.blocks_per_row)};
            const auto *root = weights_.interleavedBase() + std::size_t(chunk_) * geometry.rowBytes();
            for (int column0 = 0; column0 < columns_; column0 += 4 * S)
            {
                constexpr bool ordered = Policy == CPUProjectionNumericalPolicy::GPUAlignedExpert;
                // The three- and four-row SMMLA tiles keep the partition-0 select: the
                // seed's live register spills there (orderedPartitionSeedNEON()).
                constexpr bool seeded = ordered && !(multi_scale::usesRowPairs<Ops> && Rows >= 3) && !(Rows == 4 && Ops::instruction == Dot::ASIMD);
                float32x4_t reduced[Rows][S]{};
                if constexpr (seeded)
                {
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                        for (int strip = 0; strip < S; ++strip)
                            reduced[row][strip] = orderedPartitionSeedNEON();
                }
                for (int partition = 0; partition < (ordered ? partitions_ : 1); ++partition)
                {
                    float32x4_t partial[Rows][S]{};
                    const int first = ordered ? partition * blocks_per_partition_ : 0;
                    const int last = ordered ? std::min(first + blocks_per_partition_, weights_.blocks_per_row)
                                             : weights_.blocks_per_row;
                    for (int block = first; block < last;)
                    {
                        const int end = std::min(last, (block / superblock_blocks + 1) * superblock_blocks);
                        const auto *header = root + geometry.headerOffset(0, block);
                        Q4KBase<Ops> base[S];
#pragma GCC unroll 4
                        for (int strip = 0; strip < S; ++strip)
                            base[strip] = q4KStripBase<Ops>(header, column0 + 4 * strip);
                        const auto *unit = root + geometry.blockOffset(0, block);
                        for (; block < end; ++block, unit += block_bytes)
                        {
                            I dots[Rows][S]{};
                            if constexpr (S > 1)
                            {
                                // One activation block load per row; every strip
                                // reads its groups by lane (exact integer sums).
                                const auto blocks = activationBlocks<Ops>(block);
                                forEachGroup<4>([&]<int Group>() __attribute__((always_inline)) {
#pragma GCC unroll 4
                                    for (int strip = 0; strip < S; ++strip)
                                    {
                                        const auto raw = vld1q_u8(unit + Group * 256 + (column0 + 4 * strip) * 4);
                                        const auto low = std::bit_cast<I>(vandq_u8(raw, vdupq_n_u8(15)));
                                        const auto high = std::bit_cast<I>(vshrq_n_u8(raw, 4));
#pragma GCC unroll 4
                                        for (std::size_t row = 0; row < Rows; ++row)
                                        {
                                            dots[row][strip] = Ops::template dotLane<Group>(dots[row][strip], low, blocks[row]);
                                            dots[row][strip] = Ops::template dotLane<Group + 4>(dots[row][strip], high, blocks[row]);
                                        }
                                    }
                                });
                            }
                            else if constexpr (multi_scale::usesRowPairs<Ops>)
                            {
                                I strip_dots[Rows]{}, unused[Rows]{};
                                pairBlock<Ops, Encoding::Q4KSuperblockCoded, NibbleMap::Unsigned>(
                                    unit, column0, block, strip_dots, unused);
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row) dots[row][0] = strip_dots[row];
                            }
                            else
                            {
                                I strip_dots[Rows]{};
                                if constexpr (Rows >= 3 && Ops::instruction == Dot::ASIMD)
                                {
                                    // ASIMD widens each byte-product group. Keeping
                                    // only one decoded group live bounds the three-
                                    // and four-row block's temporary register usage.
#pragma GCC unroll 1
                                    for (int group = 0; group < 8; ++group)
                                    {
                                        const auto raw = vld1q_u8(unit + (group % 4) * 256 + column0 * 4);
                                        const auto values = group < 4 ? vandq_u8(raw, vdupq_n_u8(15)) : vshrq_n_u8(raw, 4);
                                        consume<Ops>(vreinterpretq_s8_u8(values), block, group, strip_dots);
                                    }
                                }
                                else
                                {
#pragma GCC unroll 1
                                    for (int group = 0; group < 4; ++group)
                                    {
                                        const auto raw = vld1q_u8(unit + group * 256 + column0 * 4);
                                        consume<Ops>(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(15))), block, group, strip_dots);
                                        consume<Ops>(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), block, group + 4, strip_dots);
                                    }
                                }
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row) dots[row][0] = strip_dots[row];
                            }
                            float activation_scale[Rows];
#pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                                activation_scale[row] = nativeQuantisedFP16ScaleToFP32(activation_[row][block].d);
#pragma GCC unroll 4
                            for (int strip = 0; strip < S; ++strip)
                            {
                                const auto factors = q4KStripFactors<Ops>(unit + data_bytes, column0 + 4 * strip, base[strip]);
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row)
                                    partial[row][strip] = accumulateCorrectedSingleScaleBlockNEON(partial[row][strip],
                                        std::bit_cast<int32x4_t>(dots[row][strip]), std::bit_cast<float32x4_t>(factors.primary),
                                        activation_[row][block].sum_qs, std::bit_cast<float32x4_t>(factors.secondary),
                                        activation_scale[row], Policy);
                            }
                        }
                    }
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                        for (int strip = 0; strip < S; ++strip)
                            reduced[row][strip] = !ordered || (!seeded && partition == 0) ? partial[row][strip]
                                                           : addOrderedNEON(reduced[row][strip], partial[row][strip]);
                }
#pragma GCC unroll 4
                for (int strip = 0; strip < S; ++strip)
                {
                    const int column = column0 + 4 * strip;
                    if (column >= columns_)
                        break;
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
                    {
                        auto *destination = output_[row].data() + column;
                        if (column + 4 <= columns_) vst1q_f32(destination, reduced[row][strip]);
                        else
                        {
                            vst1q_lane_f32(destination, reduced[row][strip], 0);
                            if (columns_ - column >= 2) vst1q_lane_f32(destination + 1, reduced[row][strip], 1);
                            if (columns_ - column == 3) vst1q_lane_f32(destination + 2, reduced[row][strip], 2);
                        }
                    }
                }
            }
        }

        /**
         * @brief Decode four IQ4_XS coefficients from their packed nibble and crumb planes.
         * @param planes First byte of the block's six-bit code plane.
         * @param column First column of the four-column strip.
         * @return Six-bit coefficients as four zero-extended lanes.
         */
        template <typename Ops>
        static typename Ops::I iq4xsCodes(const std::uint8_t *planes, int column) noexcept
        {
            std::uint16_t nibbles;
            std::memcpy(&nibbles, planes + column / 2, sizeof(nibbles));
            const std::uint32_t crumbs = planes[32 + column / 4];
            const std::uint32_t bytes = (nibbles & 0x000fu) | ((nibbles & 0x00f0u) << 4) |
                ((nibbles & 0x0f00u) << 8) | ((nibbles & 0xf000u) << 12) |
                ((crumbs & 0x03u) << 4) | ((crumbs & 0x0cu) << 10) |
                ((crumbs & 0x30u) << 16) | ((crumbs & 0xc0u) << 22);
            return std::bit_cast<typename Ops::I>(vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(bytes)))));
        }

        /**
         * @brief Resident (phase 0) block geometry of a per-block superblock-coded encoding.
         * @tparam Physical Q6KNativeDualScale or NibbleLUTSourceScale (the only callers).
         * @param blocks_per_row K blocks of the admitted matrix row.
         * @return The encoding's constant stride and header width over this matrix's K blocks.
         * @details The admitted publication's stride and header width are the
         * encoding's constants, so the hot addressing needs no descriptor fields.
         */
        template <Encoding Physical> static constexpr CPUNativeQuantisedBlockGeometry codedGeometry(int blocks_per_row) noexcept
        {
            constexpr int codebook = Physical == Encoding::Q6KNativeDualScale ? 8 : 0;
            return {.block_stride = preparedInterleavedBlockStride(Physical, false, codebook),
                    .header_bytes = preparedSuperblockHeaderBytes(Physical, codebook),
                    .blocks_per_row = std::size_t(blocks_per_row)};
        }

        /**
         * @brief Source superblock header of one per-block superblock-coded K block.
         * @tparam Ops Admitted integer instruction implementation.
         * @tparam Physical Q6_K or IQ4_XS encoding.
         * @param unit Admitted block address of the current K block.
         * @param header Header from forPartitionBlocks() when hoistsSourceHeader(); else ignored.
         * @param block Logical K block.
         * @return The header serving @p block. Without a hoisted header it follows the
         *         superblock's last block (resident phase 0) and is derived from @p unit
         *         where the factors are read, after the block's decode retired.
         */
        template <typename Ops, Encoding Physical>
        [[gnu::always_inline]] const std::uint8_t *sourceHeader(const std::uint8_t *unit,
            [[maybe_unused]] const std::uint8_t *header, int block) const noexcept
        {
            if constexpr (hoistsSourceHeader<Ops, Physical>())
                return header;
            else
            {
                const int end = std::min(weights_.blocks_per_row, (block | 7) + 1);
                return unit + std::size_t(end - block) * codedGeometry<Physical>(end).block_stride;
            }
        }

        /**
         * @brief Read original source-scale factors for one four-column strip.
         * @tparam Ops Admitted integer instruction implementation.
         * @tparam Physical Q6_K or IQ4_XS encoding.
         * @param unit Admitted block address of the current K block.
         * @param source_header Source superblock header serving the block (sourceHeader()).
         * @param column First of four output columns.
         * @return Exact FP32 factors for the two Q6_K halves or the whole IQ4_XS block.
         */
        template <typename Ops, Encoding Physical>
        [[gnu::always_inline]] auto codedScales(const std::uint8_t *unit, const std::uint8_t *source_header,
            int column) const noexcept
        {
            constexpr auto codebook = superblockMetadataCodebook(Physical, 8);
            const auto *plane = unit + preparedCodedPlaneOffset(Physical, preparedDataStride(Physical, 8));
            const auto *bases = reinterpret_cast<const std::uint16_t *>(source_header);
            typename Ops::I code;
            if constexpr (Physical == Encoding::Q6KNativeDualScale)
                code = Ops::words(reinterpret_cast<const std::uint16_t *>(plane) + column);
            else
                code = iq4xsCodes<Ops>(plane, column);
            return coded_metadata::sourceScaleFactors<codebook, Ops>(code, Ops::half(bases + column));
        }

        /**
         * @brief Tile programs compiled without GCC's pre-allocation instruction scheduler.
         * @details Keys are the tile witness's program keys,
         * rows:family:policy:encoding:map:minimum (family 0 ASIMD, 1 SDOT, 2 SMMLA;
         * policy 1 the ordered expert tree; a coded program's map is its codebook).
         * The first scheduling pass hoists every load of a K block ahead of its
         * decode, which in these programs keeps more vectors live than the register
         * file holds and spills accumulators or partition totals around the K walk.
         * Without it the instructions keep source order, with the same operations,
         * operands and rounding: scheduling never changes arithmetic. Each entry was
         * measured per program with the GCC 14 tile register and stack-traffic witness.
         */
        static constexpr std::string_view kUnscheduledPrograms[] = {
            "1:0:1:3:11:0", "1:1:0:1:0:0", "1:1:0:2:0:0", "1:1:1:2:0:0", "2:0:1:3:11:0",
            "2:1:0:2:0:0", "2:1:1:3:11:0", "2:2:0:3:11:0", "2:2:0:4:2:0", "2:2:1:4:2:0",
            "2:2:1:7:0:0", "2:2:1:7:1:1", "3:0:0:2:0:0", "3:0:0:3:11:0", "3:0:1:3:7:1",
            "3:1:0:2:0:0", "3:1:0:3:11:0", "3:2:0:2:0:0", "3:2:0:3:11:0",
            "3:2:0:4:2:0", "3:2:1:2:0:0", "3:2:1:3:11:0", "3:2:1:4:2:0", "3:2:1:6:1:1", "4:0:0:2:0:0",
            "4:0:0:3:11:0", "4:0:0:6:1:1", "4:1:0:3:11:0", "4:1:0:4:2:0",
            "4:1:0:7:0:0", "4:1:0:7:1:1", "4:1:1:7:1:1",
            "4:2:0:3:11:0", "4:2:0:4:2:0", "4:2:1:4:2:0", "4:2:1:6:1:1",
        };

        /**
         * @brief Exact (non-coded) tile programs compiled without GCC's jump threading, like every coded program.
         * @details Same key format as kUnscheduledPrograms. Jump threading duplicates the partition and block
         * walks' exit tests into extra copies of the K loop, each holding its own copy of the column and partition
         * state in registers; in these programs that state then spills around the block walk. Threading only
         * duplicates control flow: operations, operands and rounding are unchanged. Each entry was measured per
         * program with the GCC 14 tile register and stack-traffic witness.
         */
        static constexpr std::string_view kUnthreadedPrograms[] = {
            "1:1:0:0:1:1", "1:1:0:1:0:0", "1:1:0:2:0:0", "2:0:0:4:2:0", "2:1:0:2:0:0", "2:1:0:4:2:0",
            "2:2:1:4:2:0", "3:0:0:4:2:0", "3:0:0:7:0:0", "3:1:0:4:2:0", "3:1:0:7:0:0",
            "3:2:1:4:2:0", "4:0:0:4:2:0", "4:0:0:6:1:1", "4:0:0:7:0:0", "4:0:1:4:2:0", "4:1:0:4:2:0",
            "4:1:0:7:0:0", "4:1:1:4:2:0", "4:2:1:6:1:1",
        };

        /**
         * @brief Strip-walk programs that re-derive the stored column after the partition walk.
         * @details Same key format as kUnscheduledPrograms. GCC otherwise keeps the column's
         * output offset and tail bound as separate induction variables, live through every
         * partition and block, where they spill. An empty register barrier on the column
         * before the stores makes GCC compute both from the column there. The stores write
         * the same bytes. Each entry was measured per program with the GCC 14 tile register
         * and stack-traffic witness.
         */
        static constexpr std::string_view kOpaqueStorePrograms[] = {
            "2:0:1:3:11:0", "2:1:1:3:11:0", "3:0:0:4:2:0", "3:0:1:3:7:1",
            "3:1:0:4:2:0", "3:2:1:3:11:0", "4:0:0:3:11:0",
            "4:1:0:3:11:0", "4:1:1:4:2:0",
        };

        /** @brief Program-key fields rows:family:policy:encoding:map:minimum of one listed key; -1 if malformed. */
        static consteval std::array<int, 6> programKeyFields(std::string_view key) noexcept
        {
            std::array<int, 6> fields{};
            std::size_t at = 0;
            for (int &field : fields)
            {
                if (at >= key.size() || key[at] < '0' || key[at] > '9')
                    return {-1, -1, -1, -1, -1, -1};
                while (at < key.size() && key[at] >= '0' && key[at] <= '9') field = field * 10 + (key[at++] - '0');
                if (&field != &fields.back() && (at >= key.size() || key[at++] != ':'))
                    return {-1, -1, -1, -1, -1, -1};
            }
            if (at != key.size())
                return {-1, -1, -1, -1, -1, -1};
            return fields;
        }

        /**
         * @brief Whether every key of @p Programs is a well-formed, unique program key of an instantiable tile program.
         * @details Instantiable: rows 1-4; family ASIMD, SDOT or SMMLA (SMMLA only for two or more rows: a lone
         * row is the SDOT leaf); either policy; and an (encoding, map, minimum) that dispatch() or the coded
         * dispatch selects. The tile witness additionally checks the keys against the programs it audits.
         */
        template <const auto &Programs> static consteval bool wellFormedProgramList() noexcept
        {
            constexpr int dispatched[][3] = {{1, 0, 0}, {2, 0, 0}, {4, 2, 0}, {0, 0, 0}, {0, 2, 0}, {0, 1, 1},
                                             {0, 3, 0}, {6, 1, 1}, {7, 0, 0}, {7, 1, 1}, {3, 7, 1}, {3, 11, 0}};
            for (std::size_t index = 0; index < std::size(Programs); ++index)
            {
                const auto key = programKeyFields(Programs[index]);
                if (key[0] < 1 || key[0] > 4 || key[1] < 0 || key[1] > 2 || (key[1] == 2 && key[0] < 2) ||
                    key[2] < 0 || key[2] > 1)
                    return false;
                bool known = false;
                for (const auto &tuple : dispatched)
                    known = known || (tuple[0] == key[3] && tuple[1] == key[4] && tuple[2] == key[5]);
                if (!known)
                    return false;
                for (std::size_t other = 0; other < index; ++other)
                    if (Programs[other] == Programs[index])
                        return false;
            }
            return true;
        }
        static_assert(wellFormedProgramList<kUnscheduledPrograms>(), "kUnscheduledPrograms holds a malformed key");
        static_assert(wellFormedProgramList<kUnthreadedPrograms>(), "kUnthreadedPrograms holds a malformed key");
        static_assert(wellFormedProgramList<kOpaqueStorePrograms>(), "kOpaqueStorePrograms holds a malformed key");
        static_assert([] {
            for (const std::string_view key : kUnthreadedPrograms)
                if (programKeyFields(key)[3] == static_cast<int>(Encoding::CompactMultiScale))
                    return false;
            return true;
        }(), "coded programs always compile without jump threading (runCoded)");

        /** @brief Whether the program of these tile arguments is listed in @p Programs. */
        template <const auto &Programs, typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical, int Map,
                  bool WithMinimum>
        static consteval bool listedProgram() noexcept
        {
            const std::array<int, 6> fields{static_cast<int>(Rows), static_cast<int>(Ops::instruction),
                                            static_cast<int>(Policy), static_cast<int>(Physical), Map, WithMinimum ? 1 : 0};
            for (const std::string_view key : Programs)
                if (programKeyFields(key) == fields)
                    return true;
            return false;
        }

        /** @brief Codegen profile of one exact program: bit 0 unscheduled, bit 1 unthreaded. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical, int Map, bool WithMinimum>
        static consteval int exactProfile() noexcept
        {
            return (listedProgram<kUnscheduledPrograms, Ops, Policy, Physical, Map, WithMinimum>() ? 1 : 0) |
                   (listedProgram<kUnthreadedPrograms, Ops, Policy, Physical, Map, WithMinimum>() ? 2 : 0);
        }

        /** @brief Whether the program of these tile arguments is listed in kUnscheduledPrograms. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical, int Map, bool WithMinimum>
        static consteval bool unscheduledProgram() noexcept
        { return listedProgram<kUnscheduledPrograms, Ops, Policy, Physical, Map, WithMinimum>(); }

        /**
         * @brief Execute the source's exact block program with a fixed serial/grouped K tree.
         * @tparam Ops Admitted integer instruction implementation.
         * @tparam Policy Backend-native FMA or separately rounded expert program.
         * @tparam Physical Native nibble or expanded execution encoding.
         * @tparam Map Nibble decoder, used only by nibble encodings.
         * @tparam WithMinimum Whether the source owns an additive minimum vector.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical,
                  NibbleMap Map = NibbleMap::Centered, bool WithMinimum = false, int Strips = 0>
            requires(exactProfile<Ops, Policy, Physical, static_cast<int>(Map), WithMinimum>() == 0)
        LLAMINAR_NEON_PROJECTION_EXACT void run() const noexcept
        {
            executeBlockProgram<Ops, Policy, Physical, Map, WithMinimum, Strips, 0>();
        }

        /** @brief run() of a program listed in kUnscheduledPrograms: the same body without pre-allocation scheduling. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical,
                  NibbleMap Map = NibbleMap::Centered, bool WithMinimum = false, int Strips = 0>
            requires(exactProfile<Ops, Policy, Physical, static_cast<int>(Map), WithMinimum>() == 1)
        LLAMINAR_NEON_PROJECTION_EXACT_UNSCHEDULED void run() const noexcept
        {
            executeBlockProgram<Ops, Policy, Physical, Map, WithMinimum, Strips, 0>();
        }

        /** @brief run() of a program listed in kUnthreadedPrograms: the same body without jump threading. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical,
                  NibbleMap Map = NibbleMap::Centered, bool WithMinimum = false, int Strips = 0>
            requires(exactProfile<Ops, Policy, Physical, static_cast<int>(Map), WithMinimum>() == 2)
        LLAMINAR_NEON_PROJECTION_EXACT_UNTHREADED void run() const noexcept
        {
            executeBlockProgram<Ops, Policy, Physical, Map, WithMinimum, Strips, 0>();
        }

        /** @brief run() of a program in both lists: without pre-allocation scheduling and jump threading. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical,
                  NibbleMap Map = NibbleMap::Centered, bool WithMinimum = false, int Strips = 0>
            requires(exactProfile<Ops, Policy, Physical, static_cast<int>(Map), WithMinimum>() == 3)
        LLAMINAR_NEON_PROJECTION_EXACT_UNSCHEDULED_UNTHREADED void run() const noexcept
        {
            executeBlockProgram<Ops, Policy, Physical, Map, WithMinimum, Strips, 0>();
        }

        /** @brief Execute a coded tile with the same FP32 tree and a source-specific decoder. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, int Codebook, int Strips>
            requires(!unscheduledProgram<Ops, Policy, Encoding::CompactMultiScale, Codebook, Codebook == 7>())
        LLAMINAR_NEON_PROJECTION_CODED void runCoded() const noexcept
        {
            constexpr bool minimum = Codebook == 7;
            executeBlockProgram<Ops, Policy, Encoding::CompactMultiScale, NibbleMap::Centered, minimum, Strips, Codebook>();
        }

        /** @brief runCoded() of a program listed in kUnscheduledPrograms, without pre-allocation scheduling. */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, int Codebook, int Strips>
            requires(unscheduledProgram<Ops, Policy, Encoding::CompactMultiScale, Codebook, Codebook == 7>())
        LLAMINAR_NEON_PROJECTION_CODED_UNSCHEDULED void runCoded() const noexcept
        {
            constexpr bool minimum = Codebook == 7;
            executeBlockProgram<Ops, Policy, Encoding::CompactMultiScale, NibbleMap::Centered, minimum, Strips, Codebook>();
        }

        /**
         * @brief Whether a strip-walk program re-derives its column offsets in every K block.
         * @tparam Ops Admitted integer instruction family.
         * @tparam Policy Arithmetic policy (ordered expert or backend-native).
         * @tparam Physical Prepared encoding.
         * @tparam Map Nibble decoder of nibble encodings.
         * @tparam WithMinimum Whether the program applies an additive minimum.
         * @tparam Coded Coded source codebook, or zero.
         * @return True when the strip's column is made opaque at the top of every block.
         * @details An empty register barrier on the column stops GCC from hoisting
         * every column-derived offset (payload, bit planes, codes, header words)
         * out of the K walk, where each would occupy a register through every
         * nested loop and evict the column and partition state to the stack. Where
         * the decode keeps few offsets, hoisting is cheaper. An empty barrier whose
         * inputs do not change in the K walk can itself be hoisted out of it; the
         * IQ4_XS programs therefore tie it to the block. The choice was measured
         * per program with the GCC 14 tile register and stack-traffic witness, and
         * changes no arithmetic: the same offsets address the same bytes.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical, NibbleMap Map,
                  bool WithMinimum, int Coded>
        static constexpr bool opaqueStripColumn() noexcept
        {
            constexpr bool ordered = Policy == CPUProjectionNumericalPolicy::GPUAlignedExpert;
            constexpr bool asimd = Ops::instruction == Dot::ASIMD;
            constexpr bool sdot = Ops::instruction == Dot::DotProd;
            constexpr bool pairs = multi_scale::usesRowPairs<Ops>;
            if constexpr (Coded == 11)
                return !((Rows == 1 && sdot && ordered) || (pairs && !ordered && (Rows == 2 || Rows == 3)));
            else if constexpr (Coded == 7)
                return !(ordered && ((Rows == 1 && sdot) || (Rows == 2 && pairs) || (Rows == 3 && sdot)));
            else if constexpr (Physical == Encoding::NibbleLUT)
                return (Map == NibbleMap::Unsigned && (Rows == 2 || (Rows == 3 && pairs))) || (pairs && Rows == 4 && ordered);
            else if constexpr (Physical == Encoding::SignedInt8SingleScale)
                return !(sdot && Rows == 4 && !ordered);
            else if constexpr (Physical == Encoding::Q6KNativeDualScale)
                return !(pairs && Rows == 3 && ordered);
            else if constexpr (Physical == Encoding::NibbleLUTSourceScale)
                return true;
            else if constexpr (Physical == Encoding::Q5NativeSingleScale)
                return Rows == 4 && WithMinimum && !ordered;
            else
                return false;
        }

        /**
         * @brief Execute the source's exact block program with a fixed serial/grouped K tree.
         * @tparam Ops Admitted integer instruction implementation.
         * @tparam Policy Backend-native FMA or separately rounded expert program.
         * @tparam Physical Prepared encoding.
         * @tparam Map Nibble decoder, used only by nibble encodings.
         * @tparam WithMinimum Whether the source owns an additive minimum vector.
         * @tparam Strips Override for a measured panel width, or zero for the installed width.
         * @tparam Coded Coded source codebook, or zero for the legacy per-block program.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical,
                  NibbleMap Map, bool WithMinimum, int Strips, int Coded>
        [[gnu::always_inline]] LLAMINAR_NEON_PROJECTION_EXACT inline void executeBlockProgram() const noexcept
        {
            // Native coded groups use the strip walk to bound register liveness.
            constexpr int S = Coded != 0 ? 1 : Strips > 0 ? Strips : panelStrips<Ops, Physical, WithMinimum>();
            // A panel never crosses its 64-column chunk: S strips must tile the chunk's 16.
            static_assert(S == 1 || S == 2 || S == 4);
            if constexpr (Physical == Encoding::Q4KSuperblockCoded)
                return runQ4K<Ops, Policy, multi_scale::usesRowPairs<Ops> ? 1 : S>();
            else if constexpr (S > 1 && !multi_scale::usesRowPairs<Ops>)
            {
                // Panels cover the whole chunk: a final panel may extend into the
                // chunk's sealed padding columns, which it computes but never stores.
                for (int column = 0; column < columns_; column += 4 * S)
                    panel<Ops, Policy, Physical, Map, WithMinimum, S>(column);
            }
            else
            {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
                using I = typename Ops::I;
                using F = typename Ops::F;
                constexpr bool ordered = Policy == CPUProjectionNumericalPolicy::GPUAlignedExpert;
                // Q5_K SMMLA retires each row's scalar half conversion at its
                // contribution. Widening four scales together keeps the portable
                // conversion's masks live across the decode and spills two vectors.
                constexpr bool widen_activation_rows =
                    Coded != 0 && Rows == 4 && ordered && Ops::instruction != Dot::ASIMD &&
                    !(Coded == 7 && multi_scale::usesRowPairs<Ops>);
                constexpr bool nibble = Physical == Encoding::NibbleLUT || Physical == Encoding::NibbleLUTSourceScale;
                for (int column = 0; column < columns_; column += 4)
                {
                    float32x4_t reduced[Rows]{};
                    constexpr bool seeded = ordered && !(multi_scale::usesRowPairs<Ops> && Rows == 3 && Coded == 0) &&
                                           !(Coded == 11 && Rows == 1 && Ops::instruction == Dot::DotProd);
                    if constexpr (seeded)
                    {
#pragma GCC unroll 4
                        for (std::size_t row = 0; row < Rows; ++row)
                            reduced[row] = orderedPartitionSeedNEON();
                    }
                    // A backend-native tile has exactly one partition (partitionsFor()), so its
                    // reduction is its single interval's accumulator. The decode-heavy row-pair
                    // walks (Q5, Q6_K, IQ4_XS) carry that chain instead of keeping a second
                    // live copy for an ordered fold that cannot occur.
                    constexpr bool carry = !ordered && Coded == 0 && multi_scale::usesRowPairs<Ops> &&
                        ((Physical == Encoding::Q5NativeSingleScale && (Rows != 4 || WithMinimum)) ||
                         Physical == Encoding::Q6KNativeDualScale ||
                         Physical == Encoding::NibbleLUTSourceScale);
                    [[maybe_unused]] float32x4_t carried[Rows]{};
                    for (int partition = 0; partition < (ordered || Coded == 0 ? partitions_ : 1); ++partition)
                    {
                        float32x4_t accumulator[Rows]{};
                        if constexpr (carry)
                        {
#pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                                accumulator[row] = carried[row];
                        }
                        const int first = ordered || Coded == 0 ? partition * blocks_per_partition_ : 0;
                        const int last = ordered || Coded == 0 ? std::min(first + blocks_per_partition_, weights_.blocks_per_row)
                            : weights_.blocks_per_row;
                        forPartitionBlocks<Physical, Coded, Ops, ordered>(first, last, column, [&](int block, const std::uint8_t *unit,
                            [[maybe_unused]] const auto &base) __attribute__((always_inline)) {
                            // Re-derive this strip's column offsets in every block where that
                            // frees registers (opaqueStripColumn()); otherwise GCC hoists them.
                            int strip = column;
                            if constexpr (opaqueStripColumn<Ops, Policy, Physical, Map, WithMinimum, Coded>())
                            {
                                // IQ4_XS ties the barrier to the block, so GCC cannot hoist it
                                // (and every offset derived from it) out of the K walk.
                                if constexpr (Physical == Encoding::NibbleLUTSourceScale)
                                    asm("" : "+r"(strip) : "r"(block));
                                else
                                    asm("" : "+r"(strip));
                            }
                            I low[Rows]{}, high[Rows]{};
                            if constexpr (Coded == 7)
                            {
                                q5KBlock<Ops>(unit, strip, block, low);
                            }
                            else if constexpr (Coded == 11)
                                iq3sBlock<Ops>(unit, strip, block, low);
                            else if constexpr (multi_scale::usesRowPairs<Ops>)
                                pairBlock<Ops, Physical, Map, ordered>(unit, strip, block, low, high);
                            else if constexpr (Physical == Encoding::Q6KNativeDualScale)
                            {
                                // Keep one decoded group live at a time. Fully
                                // unrolling all eight groups lets the compiler
                                // hoist a complete four-row activation panel into
                                // registers and spill it inside the K traversal.
#if defined(__clang__)
#pragma GCC unroll 1
                                for (int group = 0; group < 8; ++group)
                                {
                                    const auto raw = vld1q_u8(unit + (group % 4) * 256 + strip * 4);
                                    auto values = group < 4 ? vandq_u8(raw, vdupq_n_u8(15)) : vshrq_n_u8(raw, 4);
                                    const auto *planes = unit + 1024 + group * 64 + (strip / 16) * 16;
                                    std::uint64_t bits0 = 0, bits1 = 0;
                                    std::memcpy(&bits0, planes, sizeof(bits0));
                                    std::memcpy(&bits1, planes + 8, sizeof(bits1));
                                    const int shift = (strip % 16) * 4;
                                    values = vorrq_u8(values, q6Plane(static_cast<std::uint16_t>(bits0 >> shift), 16));
                                    values = vorrq_u8(values, q6Plane(static_cast<std::uint16_t>(bits1 >> shift), 32));
                                    const auto centered = vsubq_s8(vreinterpretq_s8_u8(values), vdupq_n_s8(32));
                                    if (group < 4) consume<Ops>(centered, block, group, low);
                                    else consume<Ops>(centered, block, group, high);
                                }
#else
#pragma GCC unroll 1
                                for (int group = 0; group < 4; ++group)
                                    consume<Ops>(q6Group(unit, strip, group), block, group, low);
#pragma GCC unroll 1
                                for (int group = 4; group < 8; ++group)
                                    consume<Ops>(q6Group(unit, strip, group), block, group, high);
#endif
                            }
                            else if constexpr (Physical == Encoding::Q5NativeSingleScale)
                            {
                                // Bound the live decode and activation vectors independently of Rows.
                                // SDOT decode panels use lane dots; this strip shares each decoded pair
                                // across rows before advancing to the next four-byte group.
#pragma GCC unroll 1
                                for (int group = 0; group < Q5NativePayload::groups; ++group)
                                {
                                    const auto weights = ordinaryQ5Groups<Map == NibbleMap::Centered,
                                        q5GroupFifth<Ops, ordered, WithMinimum>()>(unit, strip, group);
                                    consume<Ops>(weights[0], block, group, low);
                                    consume<Ops>(weights[1], block, group + 4, low);
                                }
                            }
#if defined(__clang__)
                            else if constexpr (Rows >= 3 && Ops::instruction == Dot::ASIMD)
                            {
                                // Widened products cost more registers than SDOT.
                                // Keep one decoded group live for the wider ASIMD tiles.
#pragma clang loop unroll(disable)
                                for (int group = 0; group < 8; ++group)
                                {
                                    if constexpr (nibble)
                                    {
                                        const auto raw = vld1q_u8(unit + (group % 4) * 256 + strip * 4);
                                        const auto index = group < 4 ? vandq_u8(raw, vdupq_n_u8(15)) : vshrq_n_u8(raw, 4);
                                        consume<Ops>(nibbles<Map>(index), block, group, low);
                                    }
                                    else
                                        consume<Ops>(vld1q_s8(reinterpret_cast<const std::int8_t *>(unit + group * 256 + strip * 4)),
                                            block, group, low);
                                }
                            }
#endif
                            else if constexpr (nibble && kE8M0Scales<Map>)
                            {
                                // Lane-indexed dots on each row's once-loaded block: broadcasting
                                // every row/group activation word lets GCC hoist 24 broadcasts and
                                // spill the row accumulators. Integer dots are exact in any order.
                                const auto blocks = activationBlocks<Ops>(block);
                                forEachGroup<4>([&]<int Group>() __attribute__((always_inline)) {
                                    const auto raw = vld1q_u8(unit + Group * 256 + strip * 4);
                                    const auto low_nibbles = std::bit_cast<I>(nibbles<Map>(vandq_u8(raw, vdupq_n_u8(15))));
                                    const auto high_nibbles = std::bit_cast<I>(nibbles<Map>(vshrq_n_u8(raw, 4)));
#pragma GCC unroll 4
                                    for (std::size_t row = 0; row < Rows; ++row)
                                    {
                                        low[row] = Ops::template dotLane<Group>(low[row], low_nibbles, blocks[row]);
                                        low[row] = Ops::template dotLane<Group + 4>(low[row], high_nibbles, blocks[row]);
                                    }
                                });
                            }
                            else if constexpr (Physical == Encoding::NibbleLUTSourceScale && Rows > 1)
                            {
#pragma GCC unroll 1
                                for (int group = 0; group < 4; ++group)
                                {
                                    const auto raw = vld1q_u8(unit + group * 256 + strip * 4);
                                    consume<Ops>(nibbles<Map>(vandq_u8(raw, vdupq_n_u8(15))), block, group, low);
                                    consume<Ops>(nibbles<Map>(vshrq_n_u8(raw, 4)), block, group + 4, low);
                                }
                            }
                            else if constexpr (nibble)
                            {
#pragma GCC unroll 4
                                for (int group = 0; group < 4; ++group)
                                {
                                    const auto raw = vld1q_u8(unit + group * 256 + strip * 4);
                                    consume<Ops>(nibbles<Map>(vandq_u8(raw, vdupq_n_u8(15))), block, group, low);
                                    consume<Ops>(nibbles<Map>(vshrq_n_u8(raw, 4)), block, group + 4, low);
                                }
                            }
                            else
                            {
#pragma GCC unroll 8
                                for (int group = 0; group < 8; ++group)
                                    consume<Ops>(vld1q_s8(reinterpret_cast<const std::int8_t *>(unit + group * 256 + strip * 4)),
                                        block, group, low);
                            }
                            F scale0, scale1{};
                            if constexpr (Coded != 0)
                            {
                                // The single-row SDOT decode leaves no register for the factor
                                // program: make its addresses depend on the finished dot so GCC
                                // cannot hoist the factor loads and widening into the decode.
                                const std::uint8_t *factor_unit = unit;
                                [[maybe_unused]] auto factor_base = base;
                                if constexpr (Rows == 1 && Ops::instruction == Dot::DotProd)
                                {
                                    if constexpr (flatCodedWalk<Coded, Ops, ordered>())
                                        asm("" : "+r"(factor_unit), "+r"(factor_base) : "w"(low[0]));
                                    else
                                        asm("" : "+r"(factor_unit) : "w"(low[0]));
                                }
                                const auto factors = [&]() __attribute__((always_inline)) {
                                    // A flat walk passes the header; the grouped IQ3_S walk the widened base.
                                    if constexpr (flatCodedWalk<Coded, Ops, ordered>())
                                        return codedFactors<Coded, Ops>(factor_unit, codedBase<Coded, Ops>(factor_base, strip), strip);
                                    else
                                        return codedFactors<Coded, Ops>(factor_unit, base, strip);
                                }();
                                scale0 = factors.primary;
                                scale1 = factors.secondary;
                            }
                            else if constexpr (preparedSuperblockCoded(Physical))
                            {
                                const auto factors = codedScales<Ops, Physical>(unit, sourceHeader<Ops, Physical>(unit, base, block), strip);
                                scale0 = factors.primary;
                                scale1 = factors.secondary;
                            }
                            else if constexpr (Physical == Encoding::Q5NativeSingleScale)
                            {
                                const auto *scales = reinterpret_cast<const std::uint16_t *>(unit + Q5NativePayload::bytes);
                                scale0 = Ops::half(scales + strip);
                                if constexpr (WithMinimum)
                                    scale1 = Ops::half(scales + 64 + strip);
                            }
                            else
                            {
                                scale0 = kE8M0Scales<Map>
                                    ? Ops::e8m0(weights_.admittedE8M0Scales(chunk_, block) + strip)
                                    : Ops::half(weights_.admittedPerBlockPrimary<Physical, WithMinimum>(chunk_, block) + strip);
                                if constexpr (WithMinimum)
                                    scale1 = Ops::half(weights_.admittedPerBlockSecondary<Physical>(chunk_, block) + strip);
                            }
                            [[maybe_unused]] float32x4_t activation_scales;
                            if constexpr (widen_activation_rows)
                            {
                                // Widen the row scales as one vector with the same half-word
                                // payload bits, rather than branching after each row's dot.
                                uint16x4_t halves{};
                                forEachGroup<Rows>([&]<int Row>() __attribute__((always_inline)) {
                                    halves = vset_lane_u16(activation_[Row][block].d, halves, Row);
                                });
                                activation_scales = neon_fp16::widen(halves);
                            }
#pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                            {
                                const auto &a = activation_[row][block];
                                const float activation_scale = [&]() __attribute__((always_inline)) {
                                    if constexpr (widen_activation_rows)
                                        return activation_scales[row];
                                    else
                                        return nativeQuantisedFP16ScaleToFP32(a.d);
                                }();
                                if constexpr (Physical == Encoding::Q6KNativeDualScale)
                                {
                                    const F term0 = scale0 * __builtin_convertvector(low[row], F);
                                    const F term1 = scale1 * __builtin_convertvector(high[row], F);
                                    const F contribution = std::bit_cast<F>(vdupq_n_f32(activation_scale)) * (term0 + term1);
                                    accumulator[row] = foldNEON<Policy>(accumulator[row], std::bit_cast<float32x4_t>(contribution));
                                }
                                else if constexpr (WithMinimum)
                                    accumulator[row] = accumulateCorrectedSingleScaleBlockNEON(accumulator[row],
                                        std::bit_cast<int32x4_t>(low[row]), std::bit_cast<float32x4_t>(scale0), a.sum_qs,
                                        std::bit_cast<float32x4_t>(scale1), activation_scale, Policy);
                                else
                                    accumulator[row] = accumulateSingleScaleBlockNEON(accumulator[row],
                                        std::bit_cast<int32x4_t>(low[row]), std::bit_cast<float32x4_t>(scale0), activation_scale, Policy);
                            }
                        });
                        if constexpr (carry)
                        {
#pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                                carried[row] = accumulator[row];
                        }
                        else
                        {
#pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row)
                                reduced[row] = !seeded && ((!ordered && Coded != 0) || partition == 0)
                                    ? accumulator[row] : foldNEON<Policy>(reduced[row], accumulator[row]);
                        }
                    }
                    if constexpr (carry)
                    {
#pragma GCC unroll 4
                        for (std::size_t row = 0; row < Rows; ++row)
                            reduced[row] = carried[row];
                    }
                    // Re-derive the stored column after the partition walk where that frees
                    // registers (kOpaqueStorePrograms); otherwise GCC keeps its own
                    // induction copies of the column's store offsets live across the walk.
                    int stored = column;
                    if constexpr (listedProgram<kOpaqueStorePrograms, Ops, Policy, Physical,
                                      Coded != 0 ? Coded : static_cast<int>(Map), WithMinimum>())
                        asm("" : "+r"(stored));
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
                    {
                        if (stored + 4 <= columns_)
                            vst1q_f32(output_[row].data() + stored, reduced[row]);
                        else
                        {
                            // Scalar lane stores are the bounded final-column tail,
                            // not a row replay. Never expose the accumulator array's
                            // address to memcpy: that forces all full-column partials
                            // into stack slots even when the tail branch is unused.
                            auto *destination = output_[row].data() + stored;
                            vst1q_lane_f32(destination, reduced[row], 0);
                            if (columns_ - stored >= 2) vst1q_lane_f32(destination + 1, reduced[row], 1);
                            if (columns_ - stored == 3) vst1q_lane_f32(destination + 2, reduced[row], 2);
                        }
                    }
                }
            }
        }

        /**
         * @brief Broadcast one row's four-byte activation group to every column lane.
         * @param block Logical K block. @param group Four-value group in [0,7].
         * @return The group word repeated in all four lanes, shared by every strip of a panel.
         */
        template <typename Ops> typename Ops::I activationGroup(std::size_t row, int block, int group) const noexcept
        {
            int word = 0;
            std::memcpy(&word, activation_[row][block].qs + group * 4, sizeof(word));
            return typename Ops::I{} + word;
        }

        /**
         * @brief Load every row's activation block once for a panel (C3 seam).
         * @param block Logical K block.
         * @return One ActivationBlock per row, read by lane by every strip.
         */
        template <typename Ops> std::array<typename Ops::ActivationBlock, Rows> activationBlocks(int block) const noexcept
        {
            std::array<typename Ops::ActivationBlock, Rows> blocks;
#pragma GCC unroll 4
            for (std::size_t row = 0; row < Rows; ++row)
                blocks[row] = Ops::activationBlock(activation_[row][block].qs);
            return blocks;
        }

        /** @brief Invoke body.template operator()<G>() for G = 0..Count-1, in order; callers pass always_inline bodies so nothing is outlined. */
        template <int Count, typename Body> [[gnu::always_inline]] static void forEachGroup(Body &&body) noexcept
        {
            [&]<int... Group>(std::integer_sequence<int, Group...>) __attribute__((always_inline)) {
                (body.template operator()<Group>(), ...);
            }(std::make_integer_sequence<int, Count>{});
        }

        /**
         * @brief Execute S adjacent four-column strips together with one broadcast per row and group.
         * @tparam S Strips per panel, 2..4.
         * @param column0 First column; later strips may be partial or lie wholly in the
         * chunk's padding columns (inside the 64-column chunk, sealed at publication),
         * whose results are discarded.
         * @details Decode expressions, INT32 group order, scale/coefficient
         * expressions, the per-block FP32 program, block order and partition tree
         * are those of the strip walk in run() for every column; only S strips' independent
         * chains are interleaved and each activation group/scale is read once.
         */
        template <typename Ops, CPUProjectionNumericalPolicy Policy, Encoding Physical,
                  NibbleMap Map, bool WithMinimum, int S>
        [[gnu::always_inline]] LLAMINAR_NEON_PROJECTION_EXACT inline void panel(int column0) const noexcept
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            using I = typename Ops::I;
            using F = typename Ops::F;
            constexpr bool nibble = Physical == Encoding::NibbleLUT || Physical == Encoding::NibbleLUTSourceScale;
            float32x4_t reduced[Rows][S]{};
            constexpr bool ordered = Policy == CPUProjectionNumericalPolicy::GPUAlignedExpert;
            if constexpr (ordered)
            {
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                    for (int strip = 0; strip < S; ++strip)
                        reduced[row][strip] = orderedPartitionSeedNEON();
            }
            for (int partition = 0; partition < partitions_; ++partition)
            {
                float32x4_t accumulator[Rows][S]{};
                const int first = partition * blocks_per_partition_;
                const int last = std::min(first + blocks_per_partition_, weights_.blocks_per_row);
                forPartitionBlocks<Physical, 0, Ops>(first, last, column0, [&](int block, const std::uint8_t *unit,
                    [[maybe_unused]] const auto &base) __attribute__((always_inline)) {
                    I low[Rows][S]{}, high[Rows][S]{};
                    if constexpr (Physical == Encoding::Q6KNativeDualScale)
                    {
                        // One group's S decoded strips live at a time, the
                        // panel analogue of the strip walk's spill bound.
#pragma GCC unroll 1
                        for (int group = 0; group < 8; ++group)
                        {
                            I words[Rows];
#pragma GCC unroll 4
                            for (std::size_t row = 0; row < Rows; ++row) words[row] = activationGroup<Ops>(row, block, group);
#pragma GCC unroll 4
                            for (int strip = 0; strip < S; ++strip)
                            {
                                const auto centered = std::bit_cast<I>(q6Group(unit, column0 + 4 * strip, group));
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row)
                                {
                                    auto &dots = group < 4 ? low[row][strip] : high[row][strip];
                                    dots = Ops::dot(dots, centered, words[row]);
                                }
                            }
                        }
                    }
                    else if constexpr (Physical == Encoding::Q5NativeSingleScale)
                    {
                        // Strip-outer: one strip's fifth-bit bytes are live at a time
                        // while every group of the shared activation block is consumed.
                        const auto blocks = activationBlocks<Ops>(block);
#pragma GCC unroll 4
                        for (int strip = 0; strip < S; ++strip)
                            forEachGroup<4>([&]<int Group>() __attribute__((always_inline)) {
                                const auto weights = ordinaryQ5Groups<Map == NibbleMap::Centered>(unit, column0 + 4 * strip, Group);
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row)
                                {
                                    low[row][strip] = Ops::template dotLane<Group>(low[row][strip], std::bit_cast<I>(weights[0]), blocks[row]);
                                    low[row][strip] = Ops::template dotLane<Group + 4>(low[row][strip], std::bit_cast<I>(weights[1]), blocks[row]);
                                }
                            });
                    }
                    else if constexpr (nibble)
                    {
                        // Each row's block is loaded once; every strip reads its groups by lane.
                        const auto blocks = activationBlocks<Ops>(block);
                        forEachGroup<4>([&]<int Group>() __attribute__((always_inline)) {
#pragma GCC unroll 4
                            for (int strip = 0; strip < S; ++strip)
                            {
                                const auto raw = vld1q_u8(unit + Group * 256 + (column0 + 4 * strip) * 4);
                                const auto low_nibbles = std::bit_cast<I>(nibbles<Map>(vandq_u8(raw, vdupq_n_u8(15))));
                                const auto high_nibbles = std::bit_cast<I>(nibbles<Map>(vshrq_n_u8(raw, 4)));
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row)
                                {
                                    low[row][strip] = Ops::template dotLane<Group>(low[row][strip], low_nibbles, blocks[row]);
                                    low[row][strip] = Ops::template dotLane<Group + 4>(low[row][strip], high_nibbles, blocks[row]);
                                }
                            }
                        });
                    }
                    else
                    {
                        const auto blocks = activationBlocks<Ops>(block);
                        forEachGroup<8>([&]<int Group>() __attribute__((always_inline)) {
#pragma GCC unroll 4
                            for (int strip = 0; strip < S; ++strip)
                            {
                                const auto weights = std::bit_cast<I>(vld1q_s8(
                                    reinterpret_cast<const std::int8_t *>(unit + Group * 256 + (column0 + 4 * strip) * 4)));
#pragma GCC unroll 4
                                for (std::size_t row = 0; row < Rows; ++row)
                                    low[row][strip] = Ops::template dotLane<Group>(low[row][strip], weights, blocks[row]);
                            }
                        });
                    }
                    float activation_scale[Rows];
#pragma GCC unroll 4
                    for (std::size_t row = 0; row < Rows; ++row)
                        activation_scale[row] = nativeQuantisedFP16ScaleToFP32(activation_[row][block].d);
                    // One source header address per block, shared by every strip of the panel.
                    [[maybe_unused]] const std::uint8_t *source_header = nullptr;
                    if constexpr (preparedSuperblockCoded(Physical))
                        source_header = sourceHeader<Ops, Physical>(unit, base, block);
#pragma GCC unroll 4
                    for (int strip = 0; strip < S; ++strip)
                    {
                        const int column = column0 + 4 * strip;
                        F scale0, scale1{};
                        if constexpr (preparedSuperblockCoded(Physical))
                        {
                            // Retire one strip's factor program before the next strip's
                            // loads: interleaving both strips' widening constants spills.
                            if (strip != 0)
                                asm volatile("" ::: "memory");
                            const auto factors = codedScales<Ops, Physical>(unit, source_header, column);
                            scale0 = factors.primary;
                            scale1 = factors.secondary;
                        }
                        else if constexpr (Physical == Encoding::Q5NativeSingleScale)
                        {
                            const auto *scales = reinterpret_cast<const std::uint16_t *>(unit + Q5NativePayload::bytes);
                            scale0 = Ops::half(scales + column);
                            if constexpr (WithMinimum)
                                scale1 = Ops::half(scales + 64 + column);
                        }
                        else
                        {
                            scale0 = kE8M0Scales<Map>
                                ? Ops::e8m0(weights_.admittedE8M0Scales(chunk_, block) + column)
                                : Ops::half(weights_.admittedPerBlockPrimary<Physical, WithMinimum>(chunk_, block) + column);
                            if constexpr (WithMinimum)
                                scale1 = Ops::half(weights_.admittedPerBlockSecondary<Physical>(chunk_, block) + column);
                        }
#pragma GCC unroll 4
                        for (std::size_t row = 0; row < Rows; ++row)
                        {
                            auto &partial = accumulator[row][strip];
                            if constexpr (Physical == Encoding::Q6KNativeDualScale)
                            {
                                const F term0 = scale0 * __builtin_convertvector(low[row][strip], F);
                                const F term1 = scale1 * __builtin_convertvector(high[row][strip], F);
                                const F contribution = std::bit_cast<F>(vdupq_n_f32(activation_scale[row])) * (term0 + term1);
                                partial = foldNEON<Policy>(partial, std::bit_cast<float32x4_t>(contribution));
                            }
                            else if constexpr (WithMinimum)
                                partial = accumulateCorrectedSingleScaleBlockNEON(partial,
                                    std::bit_cast<int32x4_t>(low[row][strip]), std::bit_cast<float32x4_t>(scale0),
                                    activation_[row][block].sum_qs, std::bit_cast<float32x4_t>(scale1),
                                    activation_scale[row], Policy);
                            else
                                partial = accumulateSingleScaleBlockNEON(partial,
                                    std::bit_cast<int32x4_t>(low[row][strip]), std::bit_cast<float32x4_t>(scale0),
                                    activation_scale[row], Policy);
                        }
                    }
                });
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
#pragma GCC unroll 4
                    for (int strip = 0; strip < S; ++strip)
                        reduced[row][strip] = !ordered && partition == 0
                            ? accumulator[row][strip] : foldNEON<Policy>(reduced[row][strip], accumulator[row][strip]);
            }
#pragma GCC unroll 4
            for (int strip = 0; strip < S; ++strip)
            {
                const int column = column0 + 4 * strip;
                if (column >= columns_)
                    break;
#pragma GCC unroll 4
                for (std::size_t row = 0; row < Rows; ++row)
                {
                    if (column + 4 <= columns_)
                        vst1q_f32(output_[row].data() + column, reduced[row][strip]);
                    else
                    {
                        auto *destination = output_[row].data() + column;
                        vst1q_lane_f32(destination, reduced[row][strip], 0);
                        if (columns_ - column >= 2) vst1q_lane_f32(destination + 1, reduced[row][strip], 1);
                        if (columns_ - column == 3) vst1q_lane_f32(destination + 2, reduced[row][strip], 2);
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
#undef LLAMINAR_NEON_PROJECTION_CODED
} // namespace llaminar2::cpu::native_quantised
