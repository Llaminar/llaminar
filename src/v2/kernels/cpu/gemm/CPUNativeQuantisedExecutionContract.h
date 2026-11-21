/**
 * @file CPUNativeQuantisedExecutionContract.h
 * @brief Shared CPU prepared-projection invocation, scheduling and coordinate ABI.
 *
 * Architecture-native implementations consume one set of descriptor and scope
 * types. Original source/output geometry is independent of physical placement
 * and runtime ISA. This contract borrows pointers only; owners retain storage
 * and publication lifetime. It makes no allocation or capacity decision.
 */
#pragma once
#include "CPUNativeQuantisedPrefillSchedule.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"
#include <omp.h>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised
{
    struct CPUNativeQuantisedPackedWeights;

    /** @brief Explicit ISA request; a foreign/uncompiled route is a hard error. */
    enum class ISAPath
    {
        AUTO,   // Runtime detection: AVX512-VNNI if available, else AVX2
        AVX512, // Force AVX512-VNNI path (only valid on AVX512 hardware)
        AVX2,   // Force AVX2 emulated-VNNI path
        SCALAR, // Independent diagnostic oracle on the x86 implementation only.
        NEON,   // Native AArch64 ASIMD signed-byte implementation.
        NEON_DOTPROD, // Native AArch64 SDOT implementation; requires compiled/hardware support.
        NEON_I8MM // Native AArch64 row-pair SMMLA family; requires the NEON_I8MM envelope/hardware.
    };

    /** @brief Reject a foreign/uncompiled ISA before planning or execution.
     * @param path Explicit route or automatic native dispatch.
     * @throws std::invalid_argument For an unsupported architecture/build route.
     * @details Hardware admission remains the execution implementation's job;
     * this shape-only check is also safe in non-executing bundle planning.
     */
    inline void requireNativeQuantisedISAPath(ISAPath path)
    {
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (path == ISAPath::AUTO || path == ISAPath::NEON) return;
#if defined(__ARM_FEATURE_DOTPROD)
        if (path == ISAPath::NEON_DOTPROD) return;
#endif
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
        if (path == ISAPath::NEON_I8MM) return;
#endif
#else
        if (path == ISAPath::AUTO || path == ISAPath::AVX2 || path == ISAPath::AVX512 || path == ISAPath::SCALAR) return;
#endif
        throw std::invalid_argument("CPU NativeQuantised ISA request is foreign or outside this compiled envelope");
    }

    /**
     * @brief Return the maximum CPU ISA compiled into this binary.
     *
     * An AVX512 build and an AVX2-only build may make different verifier-row
     * policy choices even when both execute their AVX2 runtime path. In
     * particular, an AVX512 translation unit can have different code
     * generation, register allocation, and inlining decisions from the
     * AVX2-only artifact. Keep the build envelope visible to telemetry and
     * generated dispatch instead of treating the active runtime ISA as a
     * complete binary identity.
     *
     * NEON_I8MM is a local build/telemetry name only. Its arithmetic level is
     * NEONDotProd (SMMLA and SDOT publish identical exact INT32 sums), so
     * sealed CPU identities, MPI wire records and archive/cache keys never
     * carry this name.
     */
    inline constexpr const char *compiledNativeQuantisedBuildISAName()
    {
#if defined(__aarch64__) && defined(__ARM_NEON)
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
        return "NEON_I8MM";
#elif defined(__ARM_FEATURE_DOTPROD)
        return "NEON_DOTPROD";
#else
        return "NEON";
#endif
#elif LLAMINAR_COMPILED_WITH_AVX512
        return "AVX512";
#else
        return "AVX2";
#endif
    }

    /**
     * @brief Runtime policy for grouped verifier-row CPU NativeQuantised kernels.
     *
     * Pairwise tiles an arbitrary verifier batch into two-row chunk kernels,
     * with one ordinary decode-equivalent tail row when M is odd. WideRows is
     * the verifier-specialized three-row or four-row AVX512 policy that shares
     * each decoded B chunk across more rows. The generated table is trained from strict
     * decode-equivalent microbench CSVs independently for the compiled ISA,
     * effective runtime ISA, and OpenMP thread count. OrderedThreeRows is a
     * separately named explicit Prefill experiment using the existing ordered
     * three-row body for regular tiles. It is not emitted by existing tables
     * and never changes the meaning of WideRows or historical evidence.
     * An untrained runtime key
     * is an unsupported production configuration and fails explicitly; it is
     * never converted into an unmeasured Pairwise policy.
     */
    enum class VerifierRowsPolicy
    {
        /**
         * Use the generated production selector.  Perf harnesses may pass an
         * explicit policy to compare candidates on the same k-tiled path, but
         * normal inference should leave this as Auto.
         */
        Auto,
        Pairwise,
        WideRows,
        FullKRowChunkGrid,
        FullKTwoRowNbc1,
        FullKTwoRowNbc2,
        FullKTwoRowPairGridNbc1,
        FullKTwoRowPairGridNbc2,
        FullKTwoRowPairGridNbc4,
        FullKTwoRowPairGridNbc8,
        /** Explicit AVX512 GPU-aligned nibble Prefill candidate; never an Auto alias. */
        OrderedThreeRows,
    };

    /**
     * @brief Return the stable compiler and telemetry name for one grouped policy.
     *
     * These names are shared by the forceable trainer registry, route
     * telemetry, generated C++ policy, and sealed evidence. Keep the spelling
     * centralized so a new physical schedule cannot be timed under one name
     * and executed under another.
     */
    inline const char *verifierRowsPolicyName(VerifierRowsPolicy policy)
    {
        switch (policy)
        {
        case VerifierRowsPolicy::Auto:
            return "Auto";
        case VerifierRowsPolicy::Pairwise:
            return "Pairwise";
        case VerifierRowsPolicy::WideRows:
            return "WideRows";
        case VerifierRowsPolicy::OrderedThreeRows:
            return "OrderedThreeRows";
        case VerifierRowsPolicy::FullKRowChunkGrid:
            return "FullKRowChunkGrid";
        case VerifierRowsPolicy::FullKTwoRowNbc1:
            return "FullKTwoRowNbc1";
        case VerifierRowsPolicy::FullKTwoRowNbc2:
            return "FullKTwoRowNbc2";
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc1:
            return "FullKTwoRowPairGridNbc1";
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc2:
            return "FullKTwoRowPairGridNbc2";
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc4:
            return "FullKTwoRowPairGridNbc4";
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc8:
            return "FullKTwoRowPairGridNbc8";
        }
        return "Invalid";
    }


    /**
     * @brief Forceable CPU M=1 N-chunk ownership policy.
     *
     * Every value preserves the frozen serial-M1 K partition and ascending
     * FP32 reduction tree.  The policy changes only how many adjacent
     * 64-column chunks one OpenMP task owns.  `FrozenSerialOracle` exists for
     * correctness diagnostics and trainer references; production callers use
     * `Auto`. x86 uses its sealed generated table; ARM uses a total native
     * policy augmented only by separately authenticated native exact overlays.
     */
    enum class DecodeSchedulePolicy
    {
        Auto,
        FrozenSerialOracle,
        Nbc1,
        Nbc2,
        Nbc4,
        Nbc8,
        Nbc16,
    };

    /**
     * @brief Return the caller-thread's serial output-partition width.
     *
     * A mirrored TP terminal head owns and writes the complete vocabulary on
     * every rank. Its generated dispatch decision must nevertheless use the
     * same logical N geometry as the serial column-parallel oracle so exact
     * overlays, K partitioning, and task granularity cannot diverge merely
     * because ownership became replicated. The value is thread-local because
     * independent rank workers may lower different graph transactions in the
     * same process.
     *
     * @return Zero outside an equivalence scope, otherwise the positive serial
     *         output-partition width.
     */
    namespace detail
    {
        /** @brief Internal storage for the caller-thread output-partition scope. */
        inline int &cpuNativeQuantisedSerialOutputPartitionNStorage()
        {
            static thread_local int serial_partition_n = 0;
            return serial_partition_n;
        }
    }

    namespace detail
    {
        /** @brief Authenticated leaf extent permitting policy N above physical N. */
        struct OriginalOutputSlicePolicy { int physical_n = 0; int policy_n = 0; };
        /** @return Caller-thread metadata used only inside an authenticated leaf scope. */
        inline OriginalOutputSlicePolicy &originalOutputSlicePolicy()
        {
            static thread_local OriginalOutputSlicePolicy policy;
            return policy;
        }
    }

    /** @brief Read the immutable public view of the caller-thread scope. */
    inline int cpuNativeQuantisedSerialOutputPartitionN()
    {
        return detail::cpuNativeQuantisedSerialOutputPartitionNStorage();
    }

    /**
     * @brief Publish a serial output-partition width for the caller thread.
     *
     * This setter is the narrow implementation hook used by the RAII scope on
     * `CPUNativeQuantisedGemmKernel`. Production stages must enter the typed
     * `ITensorGemm::beginOutputPartitionEquivalenceScope()` API instead of
     * mutating this state directly.
     *
     * @param serial_partition_n Zero to clear the scope, otherwise a positive
     *        logical output width.
     */
    inline void setCPUNativeQuantisedSerialOutputPartitionN(int serial_partition_n)
    {
        if (serial_partition_n < 0)
        {
            throw std::invalid_argument(
                "CPU NativeQuantised serial output partition cannot be negative");
        }
        detail::cpuNativeQuantisedSerialOutputPartitionNStorage() = serial_partition_n;
    }

    /**
     * @brief Resolve the logical N used by generated policy and tile selection.
     *
     * Physical pointers, output strides, and launch extents continue to use
     * `actual_n`. Only decisions that could differ from the serial TP shard use
     * the returned width. A shorter remainder shard is legal, so the physical
     * width need not be an exact multiple of the canonical partition width.
     *
     * @param actual_n Full physical output width for this projection.
     * @return `actual_n` outside a scope, otherwise the serial partition width.
     */
    inline int cpuNativeQuantisedSerialEquivalentPolicyN(int actual_n)
    {
        if (actual_n <= 0)
            throw std::invalid_argument(
                "CPU NativeQuantised output width must be positive");
        const int serial_partition_n =
            cpuNativeQuantisedSerialOutputPartitionN();
        if (serial_partition_n == 0)
            return actual_n;
        const auto original = detail::originalOutputSlicePolicy();
        if (serial_partition_n > actual_n &&
            (original.physical_n != actual_n || original.policy_n != serial_partition_n))
        {
            throw std::logic_error(
                "CPU NativeQuantised serial output partition exceeds physical output width");
        }
        return serial_partition_n;
    }

    /**
     * @brief Worker geometry shared by declaration, dispatch and diagnostics.
     * @return Active workshare team size, or the budget for a new standalone team.
     *
     * OpenMP's max-thread query describes a future nested team, not necessarily
     * the workers entering this invocation. Using it inside a retained team can
     * change the ordered K tree and demand a different partial bank. Follow the
     * same active-team decision as OMP_WORKSHARE_REGION, without a new barrier.
     */
    inline int nativeQuantisedInvocationWorkerCount() noexcept
    {
        return omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
    }


    /** @brief Resolve an explicit physical N width without selecting arithmetic.
     * @param policy Concrete Nbc candidate.
     * @return Number of adjacent canonical 64-column chunks.
     * @throws std::invalid_argument For an automatic/oracle/unknown policy.
     */
    inline int decodeScheduleNBlockChunks(DecodeSchedulePolicy policy)
    {
        switch (policy)
        {
        case DecodeSchedulePolicy::Nbc1:
            return 1;
        case DecodeSchedulePolicy::Nbc2:
            return 2;
        case DecodeSchedulePolicy::Nbc4:
            return 4;
        case DecodeSchedulePolicy::Nbc8:
            return 8;
        case DecodeSchedulePolicy::Nbc16:
            return 16;
        case DecodeSchedulePolicy::Auto:
        case DecodeSchedulePolicy::FrozenSerialOracle:
            break;
        }
        throw std::invalid_argument(
            "CPU NativeQuantised decode policy has no explicit N-block width");
    }

    /** Return the stable route name published by production telemetry. */
    inline const char *decodeSchedulePolicyName(DecodeSchedulePolicy policy)
    {
        switch (policy)
        {
        case DecodeSchedulePolicy::Auto:
            return "Auto";
        case DecodeSchedulePolicy::FrozenSerialOracle:
            return "FrozenSerialOracle";
        case DecodeSchedulePolicy::Nbc1:
            return "Nbc1";
        case DecodeSchedulePolicy::Nbc2:
            return "Nbc2";
        case DecodeSchedulePolicy::Nbc4:
            return "Nbc4";
        case DecodeSchedulePolicy::Nbc8:
            return "Nbc8";
        case DecodeSchedulePolicy::Nbc16:
            return "Nbc16";
        }
        return "Invalid";
    }


    /**
     * @brief Ownership of unchanged fused output jobs, independent of row arithmetic.
     * Current preserves the original static mapping; Contiguous is its retained
     * large-chunk control. PanelBlocks is the measured explicit small-block path.
     * Auto selects bounded panel blocks only for qualified AVX512 GPU-aligned
     * ordinary Prefill in the original local-output route. Other supported
     * execution domains retain Current; scratch and arithmetic are never changed.
     */
    enum class FusedOutputJobOwnership { Current=0, Contiguous=1, PanelBlocks=4, Auto=5 };

    /**
     * @brief Name an invocation-local diagnostic without selecting arithmetic.
     * @param ownership Complete typed choice; unknown enum values are rejected.
     * @return Stable observation token.
     * @throws std::invalid_argument For an unknown ownership value.
     */
    [[nodiscard]] constexpr const char *fusedOutputJobOwnershipName(FusedOutputJobOwnership ownership)
    {
        switch (ownership) {
        case FusedOutputJobOwnership::Current: return "Current";
        case FusedOutputJobOwnership::Contiguous: return "Contiguous";
        case FusedOutputJobOwnership::PanelBlocks: return "PanelBlocks";
        case FusedOutputJobOwnership::Auto: return "Auto";
        }
        throw std::invalid_argument("CPU fused output ownership has an unknown value");
    }

    /**
     * @brief Descriptor for one projection in a grouped verifier-row bundle.
     *
     * All descriptors share the same pre-quantized activation rows.  Each
     * projection keeps its own packed weight matrix, output stride, and optional
     * bias. The helper below schedules arbitrary runtime-M verifier rows for
     * several projections in one OpenMP region while preserving the exact M=1
     * GEMV chunk kernels used by serial decode.
     */
    struct FusedVerifierRowsDesc
    {
        const CPUNativeQuantisedPackedWeights *packed = nullptr;
        float *output = nullptr;
        const float *bias = nullptr;
        int N = 0;
        int ldc = 0;
        /**
         * Projection-local row count. Zero inherits the function-level M.
         *
         * MoE layers route a different number of rows to each expert. Keeping
         * that geometry in the descriptor lets one persistent OpenMP team own
         * every expert projection in the layer instead of repeatedly opening
         * a whole-socket team for each small matrix.
         */
        int rows = 0;
        /**
         * Optional projection-local activation base. A null value means that
         * every projection shares the function-level activation base. This
         * keeps gate/up bundles and multi-expert down bundles on one audited
         * scheduler without copying or repacking Q8_1 activations.
         */
        const Q8_1Block *input = nullptr;
        /**
         * Forceable M=1 schedule used by the decode trainer. Grouped M>1
         * verifier launches must leave this as Auto because their row policy
         * is selected independently for the admitted architecture/ISA policy.
         */
        DecodeSchedulePolicy decode_schedule = DecodeSchedulePolicy::Auto;
        /**
         * Forceable grouped schedule for M>1 trainer and regression launches.
         * Production descriptors leave this as Auto and resolve the sealed
         * grouped policy independently for each projection geometry/codebook.
         */
        VerifierRowsPolicy verifier_schedule = VerifierRowsPolicy::Auto;
        /** Scheduling intent; changing it never changes the numerical K tree. */
        ProjectionRowsPurpose purpose = ProjectionRowsPurpose::Decode;
        int policy_n = 0; ///< Explicit authenticated leaf width; zero retains existing ambient policy.
    };

    /**
     * @brief Descriptor for one projection in a fused GEMV call.
     *
     * Every projection owns the eager interleaved representation used by the
     * ordinary GEMV entry point. Keeping one prepared representation prevents
     * fused inference from silently selecting a different Q8_0 arithmetic and
     * scheduling path than the path certified by the all-format trainer.
     */
    struct FusedGemvDesc
    {
        const CPUNativeQuantisedPackedWeights *packed; // eager interleaved weights
        float *output;                            // output vector [N]
        const float *bias;                        // optional bias [N], nullptr if none
        int N;                                    // number of output rows
        int policy_n = 0; ///< Projection-local original width; never an output stride.
    };

    /**
     * @brief Descriptor for one projection with its own pre-quantized input.
     *
     * Unlike FusedGemvDesc (which shares a single Q8_1 input across all
     * projections), each descriptor here carries its own A_q8 pointer.
     * Used for MoE expert down projections where each expert has a different
     * SwiGLU activation as input.
     */
    struct FusedGemvMultiInputDesc
    {
        const Q8_1Block *A_q8;                    // per-projection Q8_1 input
        const CPUNativeQuantisedPackedWeights *packed;  // packed weights
        float *output;                            // output vector [N]
        int N;                                    // number of output rows
    };
} // namespace llaminar2::cpu::native_quantised
