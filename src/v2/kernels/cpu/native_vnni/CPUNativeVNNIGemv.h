/**
 * @file CPUNativeVNNIGemv.h
 * @brief Optimized M=1 GEMV kernels for native-interleaved VNNI weights + AVX-512.
 *
 * ## Packed VNNI Path (all deferred formats: Q4_0, IQ4_NL, Q4_1, Q8_0, Q5_0, Q5_1)
 *
 * Activations are quantized FP32→Q8_1, weights are pre-packed into VNNI
 * interleaved order. Inner loop uses vpdpbusd (INT8 dot product).
 *
 * Entry points:
 *   - gemv_native_vnni()           — quantize FP32→Q8_1 + dispatch to packed VNNI GEMV
 *
 * ## Packed VNNI Architecture
 *
 * The weight packer stores native payload bytes (Q4_0 nibbles, IQ4_NL nibbles)
 * in VNNI-interleaved order at pack time:
 *   native_interleaved: [N_chunks][bpr][4 groups][4 ZMMs][64 bytes]
 *
 * Each 64-byte ZMM holds 16 columns × 4 consecutive native bytes.
 * Total weight memory = native payload size (zero expansion).
 *
 * At runtime, AVX-512 vpshufb decodes 4-bit nibbles to signed INT8 directly
 * in VNNI lane order. This gives native-size bandwidth (0.5 byte/element
 * for Q4_0) with vectorized VNNI accumulation — matching the GPU approach.
 *
 * ## Inner Loop (64 columns per N-chunk, 32 K-elements per block)
 *
 * Per K-block:
 *   4 groups × (4 ZMM loads + 8 vpshufb + 8 vpdpbusd) = 16 loads + 32 decode + 32 VNNI
 *   Memory traffic: 1024 bytes (native size — half of pre-decoded INT8)
 *
 * Post-K-block:
 *   4 comp loads + 4 mullo + 4 sub (bias correction)
 *   4 scale loads + 4 mul + 4 fmadd (FP32 accumulation)
 *
 * ## Cache-Aware Tiling
 *
 * Uses NativeVNNITileConfig (from CPUNativeVNNITileConfig.h) to select
 * N-block size based on detected L1/L2/L3 and shape category.
 */

#pragma once

#include <immintrin.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>

#include "CPUNativeVNNIDecode.h"
#include "CPUNativeVNNIWeightPacker.h"
#include "CPUNativeVNNITileConfig.h"
#include "tensors/AlignedVector.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"

// AVX2 GEMV/GEMM kernels (same packed weight format, uses emulated VNNI)
#include "CPUNativeAVX2Gemv.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIDecodePolicyGenerated.inc"
#include "kernels/cpu/native_vnni/CPUNativeVNNIPrefillPolicyGenerated.inc"
#include "kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc"

// Development-only artifacts intentionally omit this marker. Absence must
// fail closed so a stale local include cannot gain production authority merely
// by having the generated selector's expected filename.
#ifndef LLAMINAR_CPU_NVNNI_PREFILL_POLICY_CERTIFIED
#define LLAMINAR_CPU_NVNNI_PREFILL_POLICY_CERTIFIED 0
#endif

namespace llaminar2::cpu::native_vnni
{

    // =========================================================================
    // ISA dispatch enum for runtime-selectable GEMV/GEMM paths
    // =========================================================================
    //
    // Both AVX512 and AVX2 paths are always compiled (no #ifdef gates around
    // the dispatch logic). Tests can force either path on AVX512 hardware for
    // parity comparison. AUTO selects the best available at runtime.
    // =========================================================================

    enum class ISAPath
    {
        AUTO,   // Runtime detection: AVX512-VNNI if available, else AVX2
        AVX512, // Force AVX512-VNNI path (only valid on AVX512 hardware)
        AVX2,   // Force AVX2 emulated-VNNI path
        SCALAR  // Force scalar decode path; used as the correctness floor
    };

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
     */
    inline constexpr const char *compiledNativeVNNIBuildISAName()
    {
#if LLAMINAR_COMPILED_WITH_AVX512
        return "AVX512";
#else
        return "AVX2";
#endif
    }

    /**
     * @brief Runtime policy for grouped verifier-row CPU NativeVNNI kernels.
     *
     * Pairwise tiles an arbitrary verifier batch into two-row chunk kernels,
     * with one ordinary decode-equivalent tail row when M is odd. WideRows is
     * the verifier-specialized three-row or four-row AVX512 policy that shares
     * each decoded B chunk across more rows. The generated table is trained from strict
     * decode-equivalent microbench CSVs independently for the compiled ISA,
     * effective runtime ISA, and OpenMP thread count. An untrained runtime key
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
     * @brief Test whether a grouped policy requires serial M=1 full-K arithmetic.
     *
     * The prefill-style candidates reuse packed weights across rows while each
     * row still traverses K in serial order. They are valid only when serial
     * M=1 itself owns one K tile. Long-K domains must retain the independently
     * reduced Pairwise/WideRows K-partition kernels.
     */
    inline bool verifierRowsPolicyRequiresFullK(VerifierRowsPolicy policy)
    {
        switch (policy)
        {
        case VerifierRowsPolicy::FullKRowChunkGrid:
        case VerifierRowsPolicy::FullKTwoRowNbc1:
        case VerifierRowsPolicy::FullKTwoRowNbc2:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc1:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc2:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc4:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc8:
            return true;
        case VerifierRowsPolicy::Auto:
        case VerifierRowsPolicy::Pairwise:
        case VerifierRowsPolicy::WideRows:
            return false;
        }
        return false;
    }

    /** Return true for the N-major two-row full-K candidate family. */
    inline bool verifierRowsPolicyUsesFullKNMajor(VerifierRowsPolicy policy)
    {
        return policy == VerifierRowsPolicy::FullKTwoRowNbc1 ||
               policy == VerifierRowsPolicy::FullKTwoRowNbc2;
    }

    /** Return true for the Cartesian row-pair/N-block full-K family. */
    inline bool verifierRowsPolicyUsesFullKPairGrid(VerifierRowsPolicy policy)
    {
        switch (policy)
        {
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc1:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc2:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc4:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc8:
            return true;
        case VerifierRowsPolicy::Auto:
        case VerifierRowsPolicy::Pairwise:
        case VerifierRowsPolicy::WideRows:
        case VerifierRowsPolicy::FullKRowChunkGrid:
        case VerifierRowsPolicy::FullKTwoRowNbc1:
        case VerifierRowsPolicy::FullKTwoRowNbc2:
            return false;
        }
        return false;
    }

    /**
     * @brief Return an explicit full-K N-block width or a caller fallback.
     */
    inline int verifierRowsPolicyNBlockChunks(
        VerifierRowsPolicy policy,
        int fallback)
    {
        switch (policy)
        {
        case VerifierRowsPolicy::FullKTwoRowNbc1:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc1:
            return 1;
        case VerifierRowsPolicy::FullKTwoRowNbc2:
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc2:
            return 2;
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc4:
            return 4;
        case VerifierRowsPolicy::FullKTwoRowPairGridNbc8:
            return 8;
        case VerifierRowsPolicy::Auto:
        case VerifierRowsPolicy::Pairwise:
        case VerifierRowsPolicy::WideRows:
        case VerifierRowsPolicy::FullKRowChunkGrid:
            return fallback;
        }
        return fallback;
    }

    /**
     * @brief Normalize a requested policy to its unique physical grouped route.
     *
     * A nominal N-block width wider than the complete N inventory aliases a
     * narrower launch. WideRows likewise aliases Pairwise when the active ISA
     * or runtime M cannot execute a three/four-row microkernel. Publishing the
     * normalized identity prevents duplicate labels from entering the learner.
     */
    inline VerifierRowsPolicy normalizeVerifierRowsPolicy(
        VerifierRowsPolicy policy,
        int n_chunks,
        bool use_avx512,
        int M)
    {
        if (policy == VerifierRowsPolicy::WideRows &&
            (!use_avx512 || M < 3))
        {
            return VerifierRowsPolicy::Pairwise;
        }
        if (policy == VerifierRowsPolicy::FullKTwoRowNbc2 && n_chunks <= 1)
            return VerifierRowsPolicy::FullKTwoRowNbc1;
        if (verifierRowsPolicyUsesFullKPairGrid(policy))
        {
            const int width = verifierRowsPolicyNBlockChunks(policy, 1);
            if (n_chunks <= 1)
                return VerifierRowsPolicy::FullKTwoRowPairGridNbc1;
            if (width >= n_chunks && n_chunks <= 2)
                return VerifierRowsPolicy::FullKTwoRowPairGridNbc2;
            if (width >= n_chunks && n_chunks <= 4)
                return VerifierRowsPolicy::FullKTwoRowPairGridNbc4;
            if (width >= n_chunks && n_chunks <= 8)
                return VerifierRowsPolicy::FullKTwoRowPairGridNbc8;
        }
        return policy;
    }

    /**
     * @brief Convert the generated grouped-policy ABI to the runtime enum.
     *
     * The checked-in generated include may temporarily predate a newly
     * admitted candidate family while its replacement corpus is being
     * measured and certified. Decode the stable generated ordinal instead of
     * naming every generated enumerator here, allowing that transaction to
     * compile without weakening validation of unknown policy bytes.
     *
     * @param policy Policy byte returned by the generated selector.
     * @return The corresponding forceable runtime grouped schedule.
     * @throws std::runtime_error if generated data is outside the registered ABI.
     */
    inline VerifierRowsPolicy verifierRowsPolicyFromGenerated(
        generated::CPUNativeVNNIVerifierRowsPolicy policy)
    {
        switch (static_cast<uint8_t>(policy))
        {
        case 0:
            return VerifierRowsPolicy::Pairwise;
        case 1:
            return VerifierRowsPolicy::WideRows;
        case 2:
            return VerifierRowsPolicy::FullKRowChunkGrid;
        case 3:
            return VerifierRowsPolicy::FullKTwoRowNbc1;
        case 4:
            return VerifierRowsPolicy::FullKTwoRowNbc2;
        case 5:
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc1;
        case 6:
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc2;
        case 7:
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc4;
        case 8:
            return VerifierRowsPolicy::FullKTwoRowPairGridNbc8;
        default:
            throw std::runtime_error(
                "Generated CPU NativeVNNI verifier policy is outside the runtime ABI");
        }
    }

    /**
     * @brief Forceable CPU M=1 N-chunk ownership policy.
     *
     * Every value preserves the frozen serial-M1 K partition and ascending
     * FP32 reduction tree.  The policy changes only how many adjacent
     * 64-column chunks one OpenMP task owns.  `FrozenSerialOracle` exists for
     * correctness diagnostics and trainer references; production callers use
     * `Auto`, which requires a sealed generated table.
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

    /** Return the requested N-chunk count for one explicit decode policy. */
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
            "CPU NativeVNNI decode policy has no explicit N-block width");
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
     * @brief Collapse nominal widths that own the same physical N task.
     *
     * A request wider than the complete N-chunk inventory cannot create a new
     * launch.  Normalize it to the smallest forceable width that covers every
     * chunk so timing and profiler evidence never relabel duplicate work as a
     * distinct candidate.
     */
    inline DecodeSchedulePolicy normalizeDecodeSchedulePolicy(
        DecodeSchedulePolicy policy,
        int n_chunks)
    {
        if (policy == DecodeSchedulePolicy::Auto ||
            policy == DecodeSchedulePolicy::FrozenSerialOracle)
        {
            return policy;
        }
        if (n_chunks <= 1)
            return DecodeSchedulePolicy::Nbc1;
        if (decodeScheduleNBlockChunks(policy) < n_chunks)
            return policy;
        if (n_chunks <= 2)
            return DecodeSchedulePolicy::Nbc2;
        if (n_chunks <= 4)
            return DecodeSchedulePolicy::Nbc4;
        if (n_chunks <= 8)
            return DecodeSchedulePolicy::Nbc8;
        return DecodeSchedulePolicy::Nbc16;
    }

    /** Map one generated policy value to the forceable runtime enum. */
    inline DecodeSchedulePolicy decodeSchedulePolicyFromGenerated(
        generated::CPUNativeVNNIDecodePolicy policy)
    {
        switch (policy)
        {
        case generated::CPUNativeVNNIDecodePolicy::Nbc1:
            return DecodeSchedulePolicy::Nbc1;
        case generated::CPUNativeVNNIDecodePolicy::Nbc2:
            return DecodeSchedulePolicy::Nbc2;
        case generated::CPUNativeVNNIDecodePolicy::Nbc4:
            return DecodeSchedulePolicy::Nbc4;
        case generated::CPUNativeVNNIDecodePolicy::Nbc8:
            return DecodeSchedulePolicy::Nbc8;
        case generated::CPUNativeVNNIDecodePolicy::Nbc16:
            return DecodeSchedulePolicy::Nbc16;
        }
        throw std::runtime_error(
            "Generated CPU NativeVNNI decode policy is outside the runtime ABI");
    }

    /**
     * @brief Resolve one learned nominal width to its physical launch schedule.
     *
     * Generic trees learn candidate families from measured geometries.  A
     * family wider than a new geometry's complete N-chunk inventory aliases
     * the smallest power-of-two schedule that owns that inventory.  Resolve
     * that geometry-dependent identity at the generated-policy boundary so
     * every production `Auto` result is directly forceable by the kernel.
     *
     * @param policy Nominal candidate emitted by the generated generic tree.
     * @param N Runtime output width in scalar columns.
     * @return The unique physical schedule for this candidate and geometry.
     */
    inline DecodeSchedulePolicy resolveGeneratedDecodeSchedulePolicy(
        generated::CPUNativeVNNIDecodePolicy policy,
        int N)
    {
        if (N <= 0)
        {
            throw std::invalid_argument(
                "CPU NativeVNNI decode policy requires positive N");
        }
        return normalizeDecodeSchedulePolicy(
            decodeSchedulePolicyFromGenerated(policy),
            (N + 63) / 64);
    }

    /**
     * @brief Resolve the sealed CPU M=1 schedule for a physical launch.
     *
     * @param packed Prepared weights whose execution codebook owns the domain.
     * @param N Output width.
     * @param K Input width.
     * @param use_avx512 Whether this invocation selected AVX512 kernels.
     * @param use_avx2 Whether this invocation selected AVX2 kernels.
     * @param serial_kpart Whether frozen serial arithmetic has multiple K tiles.
     * @param serial_k_tiles Exact frozen K-tile count for producer-grid policy.
     * @return A forceable N-chunk schedule backed by certified evidence.
     * @throws std::runtime_error when no certified total policy is installed.
     */
    inline DecodeSchedulePolicy selectDecodeSchedulePolicy(
        const CPUNativeVNNIPackedWeights &packed,
        int N,
        int K,
        bool use_avx512,
        bool use_avx2,
        bool serial_kpart,
        int serial_k_tiles)
    {
        if constexpr (!LLAMINAR_CPU_NVNNI_DECODE_POLICY_CERTIFIED)
        {
            throw std::runtime_error(
                "CPU NativeVNNI M=1 Auto dispatch requires a sealed-certified "
                "generated decode policy");
        }

#if LLAMINAR_COMPILED_WITH_AVX512
        constexpr auto build_isa =
            generated::CPUNativeVNNIDecodeBuildISA::AVX512;
#else
        constexpr auto build_isa =
            generated::CPUNativeVNNIDecodeBuildISA::AVX2;
#endif
        generated::CPUNativeVNNIDecodeRuntimeISA runtime_isa{};
        if (use_avx512)
            runtime_isa = generated::CPUNativeVNNIDecodeRuntimeISA::AVX512;
        else if (use_avx2)
            runtime_isa = generated::CPUNativeVNNIDecodeRuntimeISA::AVX2;
        else
            throw std::runtime_error(
                "CPU NativeVNNI M=1 decode requires AVX2 or AVX512");

        generated::CPUNativeVNNIDecodePolicy generated_policy{};
        if (generated::selectCPUNativeVNNIDecodeGeneratedPolicy(
                build_isa,
                runtime_isa,
                omp_get_max_threads(),
                packed.codebook_id,
                N,
                K,
                serial_kpart,
                serial_k_tiles,
                generated_policy))
        {
            return resolveGeneratedDecodeSchedulePolicy(generated_policy, N);
        }
        throw std::runtime_error(
            std::string("No certified CPU NativeVNNI M=1 decode policy for build=") +
            compiledNativeVNNIBuildISAName() +
            " runtime=" + (use_avx512 ? "AVX512" : "AVX2") +
            " threads=" + std::to_string(omp_get_max_threads()) +
            " codebook=" + std::to_string(packed.codebook_id) +
            " N=" + std::to_string(N) +
            " K=" + std::to_string(K) +
            " serial_kpart=" + (serial_kpart ? "true" : "false"));
    }

    /**
     * @brief Forceable ordinary-prefill work-sharing schedule.
     *
     * Production callers use Auto and receive the sealed generated policy.
     * The explicit values exist for integration tests and the strong perf
     * trainer, where every physical candidate must be requested independently
     * and authenticated through launch telemetry. They select only task
     * ownership: every schedule invokes the same serial-M1-equivalent chunk
     * microkernels and leaves each output row's increasing-K arithmetic tree
     * unchanged.
     */
    enum class PrefillSchedulePolicy
    {
        /** Resolve the sealed generated production policy. */
        Auto,

        /** Schedule one independent `(row, 64-column chunk)` task. */
        RowChunkGrid,

        /** Schedule N blocks and visit every two-row tile inside each task. */
        TwoRowNMajor,

        /** Schedule the Cartesian product of two-row tiles and N blocks. */
        TwoRowPairGrid,
    };

    /**
     * @brief One immutable grouped-policy lookup cached by the caller thread.
     *
     * Generated policy publication is immutable for the lifetime of a binary,
     * but OpenMP thread count, effective runtime ISA, and serial K partition
     * remain part of the certified dispatch identity. Keeping those dimensions
     * beside the packed geometry key prevents tests or applications that
     * intentionally change a runtime regime from reusing a decision certified
     * for another regime.
     */
    struct VerifierRowsPolicyCacheEntry
    {
        uint64_t geometry_key = 0;
        int threads = 0;
        int k_tiles = 0;
        generated::CPUNativeVNNIRuntimeISA runtime_isa =
            generated::CPUNativeVNNIRuntimeISA::AVX2;
        VerifierRowsPolicy policy = VerifierRowsPolicy::Pairwise;
        bool valid = false;

        /** Return true only for the complete certified runtime identity. */
        bool matches(
            uint64_t expected_geometry_key,
            int expected_threads,
            int expected_k_tiles,
            generated::CPUNativeVNNIRuntimeISA expected_runtime_isa) const
        {
            return valid && geometry_key == expected_geometry_key &&
                   threads == expected_threads &&
                   k_tiles == expected_k_tiles &&
                   runtime_isa == expected_runtime_isa;
        }
    };

    /**
     * @brief Hash one grouped-policy identity into the allocation-free cache.
     */
    inline size_t verifierRowsPolicyCacheIndex(
        uint64_t geometry_key,
        int threads,
        int k_tiles,
        generated::CPUNativeVNNIRuntimeISA runtime_isa)
    {
        /*
         * The generated ABI already packs N, K, M, and codebook into distinct
         * byte ranges. Folding those ranges is both cheaper and better suited
         * to this tiny direct-mapped table than a general-purpose avalanche
         * hash with multiple 64-bit multiplications. Thread count and ISA are
         * folded into the same low byte because each is part of cache identity.
         */
        const uint64_t folded =
            geometry_key ^ (geometry_key >> 24) ^ (geometry_key >> 48) ^
            (geometry_key >> 56) ^
            (static_cast<uint64_t>(static_cast<uint32_t>(threads)) << 1) ^
            (static_cast<uint64_t>(static_cast<uint32_t>(k_tiles)) << 9) ^
            (static_cast<uint64_t>(runtime_isa) << 7);
        return static_cast<size_t>(folded & 255ULL);
    }

    /**
     * @brief Resolve grouped policy for an already-known runtime context.
     *
     * The grouped launcher has already resolved its effective ISA and OpenMP
     * team size before selecting a physical row schedule. Accepting that
     * context avoids repeating runtime probes in the hot path. A one-entry MRU
     * serves the normal graph-replay case in which the same tensor geometry and
     * M recur on every token. A 256-entry direct-mapped table retains nearby M,
     * codebook, and geometry variants without allocation, locking, or mutable
     * state in the shared packed-weight object.
     *
     * The cache is thread-local because multiple inference requests may read
     * one packed tensor concurrently. Generated policy data is immutable, so a
     * cache hit is exactly equivalent to rerunning the generated selector.
     */
    inline VerifierRowsPolicy selectVerifierRowsPolicy(
        const CPUNativeVNNIPackedWeights &packed,
        int M,
        int N,
        int K,
        ISALevel effective_isa,
        int threads,
        int serial_k_tiles)
    {
        if (M < 2)
            throw std::invalid_argument(
                "CPU NativeVNNI grouped verifier policy requires M >= 2");
        if (threads <= 0)
            throw std::invalid_argument(
                "CPU NativeVNNI grouped verifier policy requires a positive thread count");
        if (serial_k_tiles < 0)
            throw std::invalid_argument(
                "CPU NativeVNNI grouped verifier policy requires non-negative K tiles");

        /*
         * The kernel inventory is physical-row-tile based, not speculative
         * depth specialized. Generated evidence chooses among K-partition and
         * full-K grouped schedules because task count, weight reuse, and tail
         * composition affect economy. Every result must still resolve to one
         * registered physical policy before the cache may publish it.
         */
        generated::CPUNativeVNNIVerifierRowsPolicy generated_policy{};
#if LLAMINAR_COMPILED_WITH_AVX512
        constexpr auto build_isa =
            generated::CPUNativeVNNIBuildISA::AVX512;
#else
        constexpr auto build_isa =
            generated::CPUNativeVNNIBuildISA::AVX2;
#endif
        generated::CPUNativeVNNIRuntimeISA runtime_isa{};
        switch (effective_isa)
        {
        case ISALevel::AVX512:
            runtime_isa = generated::CPUNativeVNNIRuntimeISA::AVX512;
            break;
        case ISALevel::AVX2:
            runtime_isa = generated::CPUNativeVNNIRuntimeISA::AVX2;
            break;
        case ISALevel::Scalar:
        default:
            throw std::runtime_error(
                "CPU NativeVNNI verifier rows require AVX2 or AVX512");
        }

        const uint64_t geometry_key =
            generated::packCPUNativeVNNIVerifierRowsPolicyKey(
                packed.codebook_id, M, N, K);
        constexpr size_t cache_capacity = 256;
        static thread_local VerifierRowsPolicyCacheEntry most_recent;
        static thread_local std::array<
            VerifierRowsPolicyCacheEntry,
            cache_capacity>
            cache{};

        if (most_recent.matches(
                geometry_key, threads, serial_k_tiles, runtime_isa))
            return most_recent.policy;

        VerifierRowsPolicyCacheEntry &cache_entry = cache[
            verifierRowsPolicyCacheIndex(
                geometry_key, threads, serial_k_tiles, runtime_isa)];
        if (cache_entry.matches(
                geometry_key, threads, serial_k_tiles, runtime_isa))
        {
            most_recent = cache_entry;
            return cache_entry.policy;
        }

#if LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI >= 3
        const bool selected_generated_policy =
            generated::selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
                build_isa,
                runtime_isa,
                threads,
                packed.codebook_id,
                M,
                N,
                K,
                serial_k_tiles,
                generated_policy);
#else
        const bool selected_generated_policy =
            generated::selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
                build_isa,
                runtime_isa,
                threads,
                packed.codebook_id,
                M,
                N,
                K,
                generated_policy);
#endif
        if (selected_generated_policy)
        {
            const VerifierRowsPolicy policy =
                verifierRowsPolicyFromGenerated(generated_policy);
            cache_entry = {
                .geometry_key = geometry_key,
                .threads = threads,
                .k_tiles = serial_k_tiles,
                .runtime_isa = runtime_isa,
                .policy = policy,
                .valid = true,
            };
            most_recent = cache_entry;
            return policy;
        }

        throw std::runtime_error(
            std::string("No certified CPU NativeVNNI verifier-row policy for build=") +
            compiledNativeVNNIBuildISAName() +
            " runtime=" +
            (effective_isa == ISALevel::AVX512 ? "AVX512" : "AVX2") +
            " threads=" + std::to_string(threads) +
            " codebook=" + std::to_string(packed.codebook_id) +
            " M=" + std::to_string(M) +
            " N=" + std::to_string(N) +
            " K=" + std::to_string(K));
    }

    /**
     * @brief Resolve grouped policy from the ambient CPU runtime context.
     *
     * Diagnostic callers that have not already selected an ISA use this
     * convenience overload. Production grouped GEMM passes its pre-resolved
     * context to the overload above and therefore does not pay these probes a
     * second time.
     */
    inline VerifierRowsPolicy selectVerifierRowsPolicy(
        const CPUNativeVNNIPackedWeights &packed,
        int M,
        int N,
        int K)
    {
        const int threads = omp_get_max_threads();
        const NativeVNNITileConfig serial_geometry = computeTileConfig(
            N, K, 1, packed.payload_bytes, threads);
        return selectVerifierRowsPolicy(
            packed,
            M,
            N,
            K,
            activeISALevel(),
            threads,
            serial_geometry.k_tiles);
    }

    /**
     * @brief Resolve the certified ordinary-prefill physical schedule.
     *
     * Ordinary M>1 projection and grouped verifier rows have different launch
     * economics even though both must preserve the serial-M1 accumulation
     * tree. The generated prefill selector therefore owns row-chunk versus
     * two-row N sharing as well as Pairwise versus WideRows publication for a
     * serial K-part geometry. The caller supplies the *effective* ISA selected
     * for this invocation so an AVX512 build forced to AVX2 uses its separately
     * trained policy surface.
     *
     * @param packed Prepared production weights and execution codebook.
     * @param M Runtime row count before model-tier bucketing.
     * @param N Output width.
     * @param K Input width.
     * @param use_avx512 True when this invocation executes AVX512 kernels.
     * @param use_avx2 True when this invocation executes AVX2 kernels.
     * @param serial_kpart True when production serial M1 owns multiple ordered
     * K partitions for this geometry.
     * @return A measured, byte-exact ordinary-prefill schedule.
     * @throws std::runtime_error when the active build/ISA/thread/codebook/shape
     * surface has no generated policy. Production never substitutes the old
     * tile heuristic for an untrained policy cell.
     */
    inline generated::CPUNativeVNNIPrefillPolicy selectPrefillPolicy(
        const CPUNativeVNNIPackedWeights &packed,
        int M,
        int N,
        int K,
        bool use_avx512,
        bool use_avx2,
        bool serial_kpart)
    {
        if constexpr (!LLAMINAR_CPU_NVNNI_PREFILL_POLICY_CERTIFIED)
        {
            throw std::runtime_error(
                "CPU NativeVNNI prefill Auto dispatch requires a sealed-certified "
                "generated policy; the compiled table is development-only");
        }

#if LLAMINAR_COMPILED_WITH_AVX512
        constexpr auto build_isa =
            generated::CPUNativeVNNIPrefillBuildISA::AVX512;
#else
        constexpr auto build_isa =
            generated::CPUNativeVNNIPrefillBuildISA::AVX2;
#endif
        generated::CPUNativeVNNIPrefillRuntimeISA runtime_isa{};
        if (use_avx512)
        {
            runtime_isa = generated::CPUNativeVNNIPrefillRuntimeISA::AVX512;
        }
        else if (use_avx2)
        {
            runtime_isa = generated::CPUNativeVNNIPrefillRuntimeISA::AVX2;
        }
        else
        {
            throw std::runtime_error(
                "CPU NativeVNNI prefill requires AVX2 or AVX512");
        }

        generated::CPUNativeVNNIPrefillPolicy policy{};
        if (generated::selectCPUNativeVNNIPrefillGeneratedPolicy(
                build_isa,
                runtime_isa,
                omp_get_max_threads(),
                packed.codebook_id,
                M,
                N,
                K,
                serial_kpart,
                policy))
        {
            return policy;
        }

        throw std::runtime_error(
            std::string("No certified CPU NativeVNNI prefill policy for build=") +
            compiledNativeVNNIBuildISAName() +
            " runtime=" + (use_avx512 ? "AVX512" : "AVX2") +
            " threads=" + std::to_string(omp_get_max_threads()) +
            " codebook=" + std::to_string(packed.codebook_id) +
            " M=" + std::to_string(M) +
            " N=" + std::to_string(N) +
            " K=" + std::to_string(K) +
            " serial_kpart=" + (serial_kpart ? "true" : "false"));
    }

    // =========================================================================
    // Scalar reference GEMV (any format, for correctness verification)
    // =========================================================================

    /**
     * @brief Scalar NativeVNNI GEMV for correctness reference.
     *
     * For nibble-LUT formats (Q4_0, IQ4_NL, Q4_1, IQ4_XS): decodes from payload.
     * For INT8 pre-decoded formats: reads from int8_flat buffer.
     */
    inline void gemv_native_vnni_scalar(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int N,
        int K_blocks)
    {
        const int N_chunks = (N + 63) / 64;

        for (int chunk = 0; chunk < N_chunks; ++chunk)
        {
            const int n_start = chunk * 64;
            const int n_end = std::min(n_start + 64, N);

            for (int n_local = 0; n_local < (n_end - n_start); ++n_local)
            {
                const int n = n_start + n_local;
                float acc = 0.0f;

                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    const Q8_1Block &a_blk = A_q8[kb];
                    float a_scale = simd::fp16_to_fp32(a_blk.d);

                    int8_t b_vals[32];
                    if (packed.is_nibble_lut)
                    {
                        const uint8_t *payload = packed.blockPayload(chunk, kb, n_local);
                        decode_native_block(packed.codebook_id, payload, b_vals);
                    }
                    else
                    {
                        if (!packed.int8_flat.empty())
                        {
                            std::memcpy(b_vals, packed.blockInt8(chunk, kb, n_local), 32);
                        }
                        else
                        {
                            /*
                             * Modern packed weights release int8_flat after
                             * interleaving to avoid carrying two full decoded
                             * copies.  Reconstruct the scalar reference block
                             * from the same native_interleaved layout used by
                             * AVX2/AVX512: group covers four K values, and z
                             * selects the 16-column lane group inside a chunk.
                             */
                            const int z = n_local / 16;
                            const int lane = n_local % 16;
                            for (int group = 0; group < 8; ++group)
                            {
                                const auto *group_data = reinterpret_cast<const int8_t *>(
                                    packed.interleavedB(chunk, kb, group, z));
                                const int src = lane * 4;
                                const int dst = group * 4;
                                b_vals[dst + 0] = group_data[src + 0];
                                b_vals[dst + 1] = group_data[src + 1];
                                b_vals[dst + 2] = group_data[src + 2];
                                b_vals[dst + 3] = group_data[src + 3];
                            }
                        }
                    }

                    float b_scale = packed.blockScale(chunk, kb, n_local);
                    float b_min = packed.blockMin(chunk, kb, n_local);

                    int32_t dot = 0;
                    int32_t b_comp = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        uint8_t a_u8 = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[i]) + 128);
                        dot += static_cast<int32_t>(a_u8) * static_cast<int32_t>(b_vals[i]);
                        b_comp += b_vals[i];
                    }

                    int32_t corrected = dot - 128 * b_comp;
                    acc += static_cast<float>(corrected) * a_scale * b_scale;

                    if (packed.is_asymmetric && b_min != 0.0f)
                    {
                        acc += static_cast<float>(a_blk.sum_qs) * a_scale * b_min;
                    }
                }

                C[n] = acc;
            }
        }
    }

    // =========================================================================
    // AVX-512 VNNI GEMV with native-interleaved weights (optimized hot path)
    // =========================================================================
    //
    // The weight packer stores native payload bytes in VNNI-interleaved order:
    //   native_interleaved: [N_chunks][bpr][4 groups][4 ZMMs][64 bytes]
    //
    // Each 64-byte ZMM holds 16 columns × 4 consecutive native bytes.
    // At runtime, vpshufb decodes nibbles→INT8 directly in VNNI lane order:
    //   - Low nibbles  → K-elements [group*4 .. group*4+3]
    //   - High nibbles → K-elements [group*4+16 .. group*4+19]
    //
    // Memory traffic per K-block per 64-col chunk: 1024 bytes (= native size!)
    // vs 2048 bytes for the old pre-decoded INT8 path.
    // =========================================================================

    // Decode LUT tables for nibble→INT8 conversion via vpshufb.
    // Each 16-entry table maps a 4-bit nibble to a signed INT8 value.
    // Q4_0: nibble → (nibble - 8) = [-8, -7, ..., +7]
    alignas(16) static constexpr int8_t Q4_0_DECODE_LUT[16] = {
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
    // IQ4_NL / IQ4_XS: nibble → kvalues_iq4nl[nibble]
    alignas(16) static constexpr int8_t IQ4_NL_DECODE_LUT[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    // Q4_1: nibble → nibble (unsigned identity [0..15])
    alignas(16) static constexpr int8_t Q4_1_DECODE_LUT[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

    /**
     * @brief Build the 512-bit decode LUT for a given codebook_id.
     *
     * Broadcasts a 16-byte LUT to all four 128-bit lanes of a ZMM register.
     * Used with vpshufb to decode 4-bit nibbles to signed INT8 in one instruction.
     * Only valid for nibble-LUT formats (Q4_0, IQ4_NL, Q4_1, IQ4_XS).
     */
    inline __m512i build_decode_lut(uint8_t codebook_id)
    {
        const int8_t *lut_data;
        switch (codebook_id)
        {
        case 4: // IQ4_NL / IQ4_XS (both use kvalues_iq4nl LUT)
            lut_data = IQ4_NL_DECODE_LUT;
            break;
        case 5: // Q4_1
            lut_data = Q4_1_DECODE_LUT;
            break;
        default: // Q4_0 (codebook 0)
            lut_data = Q4_0_DECODE_LUT;
            break;
        }
        __m128i lut_128 = _mm_load_si128(reinterpret_cast<const __m128i *>(lut_data));
        return _mm512_broadcast_i32x4(lut_128);
    }

    enum class NibbleDecodeKind : uint8_t
    {
        LUT,
        Q4_0_LINEAR,
        Q4_1_IDENTITY,
    };

    /**
     * @brief Classify nibble decode math for AVX512 NativeVNNI kernels.
     *
     * Q4_0 and Q4_1-family payloads are linear nibble mappings, so they can be
     * decoded with byte arithmetic instead of `vpshufb`.  IQ4 formats keep the
     * lookup-table path because their codebook is non-linear.  This is a
     * format-semantic choice, not a dispatch fallback: every branch is still
     * mathematically identical to the scalar NativeVNNI decode.
     */
    inline NibbleDecodeKind nibbleDecodeKind(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:
            return NibbleDecodeKind::Q4_0_LINEAR;
        case 5:
            return NibbleDecodeKind::Q4_1_IDENTITY;
        default:
            return NibbleDecodeKind::LUT;
        }
    }

    /**
     * @brief Pack four signed Q8_1 activation bytes exactly like M=1 GEMV.
     *
     * The VNNI dot instruction consumes unsigned activation bytes.  The decode
     * GEMV path converts each signed Q8_1 byte with `int16(q) + 128` before
     * broadcasting the four-byte word.  Multi-row verifier kernels must use the
     * same helper instead of clever bit tricks: even mathematically equivalent
     * shortcuts make it harder to reason about strict decode equivalence when
     * a verifier row later publishes KV/GDN state.
     */
    inline int32_t pack_q8_1_unsigned_word(const Q8_1Block &block, int base_idx)
    {
        uint8_t vals[4];
        vals[0] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 0]) + 128);
        vals[1] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 1]) + 128);
        vals[2] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 2]) + 128);
        vals[3] = static_cast<uint8_t>(static_cast<int16_t>(block.qs[base_idx + 3]) + 128);
        int32_t packed_word;
        std::memcpy(&packed_word, vals, sizeof(packed_word));
        return packed_word;
    }

    /**
     * @brief Decode low nibbles to signed INT8 VNNI lanes.
     */
    inline __m512i decode_low_nibbles_avx512(
        __m512i raw,
        __m512i decode_lut,
        NibbleDecodeKind kind,
        __m512i mask_0F,
        __m512i q4_zero_offset)
    {
        const __m512i lo = _mm512_and_si512(raw, mask_0F);
        switch (kind)
        {
        case NibbleDecodeKind::Q4_0_LINEAR:
            return _mm512_sub_epi8(lo, q4_zero_offset);
        case NibbleDecodeKind::Q4_1_IDENTITY:
            return lo;
        case NibbleDecodeKind::LUT:
        default:
            return _mm512_shuffle_epi8(decode_lut, lo);
        }
    }

    /**
     * @brief Decode high nibbles to signed INT8 VNNI lanes.
     */
    inline __m512i decode_high_nibbles_avx512(
        __m512i raw,
        __m512i decode_lut,
        NibbleDecodeKind kind,
        __m512i mask_0F,
        __m512i q4_zero_offset)
    {
        const __m512i hi =
            _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask_0F);
        switch (kind)
        {
        case NibbleDecodeKind::Q4_0_LINEAR:
            return _mm512_sub_epi8(hi, q4_zero_offset);
        case NibbleDecodeKind::Q4_1_IDENTITY:
            return hi;
        case NibbleDecodeKind::LUT:
        default:
            return _mm512_shuffle_epi8(decode_lut, hi);
        }
    }

    /**
     * @brief AVX-512 VNNI GEMV for one N-chunk of 64 columns.
     *
     * Reads native-interleaved bytes (1024 B/K-block = native payload size),
     * decodes nibbles to INT8 via vpshufb, and feeds directly into vpdpbusd.
     *
     * Per K-block inner loop (4 groups × 2 subs each = 8 sub-iterations):
     *   16 ZMM loads (native data) + 32 vpshufb (decode) + 32 vpdpbusd
     *   Memory traffic: 1024 bytes (native size — zero expansion)
     */
    inline void gemv_native_vnni_avx512_chunk_native(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut)
    {
        __m512 fp_acc0 = _mm512_setzero_ps();
        __m512 fp_acc1 = _mm512_setzero_ps();
        __m512 fp_acc2 = _mm512_setzero_ps();
        __m512 fp_acc3 = _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            __m512i int_acc0 = _mm512_setzero_si512();
            __m512i int_acc1 = _mm512_setzero_si512();
            __m512i int_acc2 = _mm512_setzero_si512();
            __m512i int_acc3 = _mm512_setzero_si512();

            // 4 groups: each loads 4 native bytes per column and extracts lo+hi nibbles.
            // Group g covers native bytes [g*4..g*4+3]:
            //   low  nibbles → K-elements [g*4 .. g*4+3]     (1st sub)
            //   high nibbles → K-elements [g*4+16 .. g*4+19]  (2nd sub)
            for (int group = 0; group < 4; ++group)
            {
                // Load 4 ZMMs of interleaved native bytes (64 cols × 4 bytes)
                __m512i raw0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i raw1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i raw2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i raw3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // Decode low nibbles via vpshufb LUT → signed INT8 in VNNI lane order
                __m512i lo0 = decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo1 = decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo2 = decode_low_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo3 = decode_low_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                // A broadcast for low-nibble sub (K-elements group*4..group*4+3)
                uint8_t a_lo[4];
                a_lo[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_lo[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_lo[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_lo[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_lo_i32;
                std::memcpy(&a_lo_i32, a_lo, 4);
                __m512i a_lo_bcast = _mm512_set1_epi32(a_lo_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_lo_bcast, lo0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_lo_bcast, lo1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_lo_bcast, lo2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_lo_bcast, lo3);

                // Decode high nibbles: shift right 4, mask, then vpshufb LUT
                __m512i hi0 = decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi1 = decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi2 = decode_high_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi3 = decode_high_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                // A broadcast for high-nibble sub (K-elements group*4+16..group*4+19)
                uint8_t a_hi[4];
                a_hi[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 16]) + 128);
                a_hi[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 17]) + 128);
                a_hi[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 18]) + 128);
                a_hi[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 19]) + 128);
                int32_t a_hi_i32;
                std::memcpy(&a_hi_i32, a_hi, 4);
                __m512i a_hi_bcast = _mm512_set1_epi32(a_hi_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_hi_bcast, hi0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_hi_bcast, hi1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_hi_bcast, hi2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_hi_bcast, hi3);
            }

            // Bias correction: corrected = int_acc - 128 * comp
            // comp is INT16 (halved metadata); load via sign-extend to INT32
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i comp0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i comp1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i comp2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i comp3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            int_acc0 = _mm512_sub_epi32(int_acc0, _mm512_mullo_epi32(bias_128_i32, comp0));
            int_acc1 = _mm512_sub_epi32(int_acc1, _mm512_mullo_epi32(bias_128_i32, comp1));
            int_acc2 = _mm512_sub_epi32(int_acc2, _mm512_mullo_epi32(bias_128_i32, comp2));
            int_acc3 = _mm512_sub_epi32(int_acc3, _mm512_mullo_epi32(bias_128_i32, comp3));

            // Convert to FP32 and scale: fp_val = int32_val * a_scale * b_scale[n]
            // scales are FP16 (halved metadata); load via F16C convert to FP32
            __m512 a_scale_v = _mm512_set1_ps(a_scale);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 cs0 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales))));
            __m512 cs1 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16))));
            __m512 cs2 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32))));
            __m512 cs3 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48))));

            fp_acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc0), cs0, fp_acc0);
            fp_acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc1), cs1, fp_acc1);
            fp_acc2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc2), cs2, fp_acc2);
            fp_acc3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc3), cs3, fp_acc3);

            // Asymmetric correction: acc += a_scale * sum_qs * b_min[n]
            // Formula: weight = scale * int_val + min, so the offset term
            // contributes min * Σ A[k] ≈ min * a_scale * sum_qs per block.
            // mins are FP16; load via F16C convert to FP32
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 a_corr_v = _mm512_set1_ps(static_cast<float>(a_sum) * a_scale);
                fp_acc0 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins))), fp_acc0);
                fp_acc1 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16))), fp_acc1);
                fp_acc2 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32))), fp_acc2);
                fp_acc3 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48))), fp_acc3);
            }
        }

        _mm512_storeu_ps(C, fp_acc0);
        _mm512_storeu_ps(C + 16, fp_acc1);
        _mm512_storeu_ps(C + 32, fp_acc2);
        _mm512_storeu_ps(C + 48, fp_acc3);
    }

    // =========================================================================
    // AVX-512 VNNI GEMV with pre-decoded INT8 weights (non-4-bit formats)
    // =========================================================================
    //
    // For formats that cannot use vpshufb LUT decode (Q5_0, Q5_1, Q6_K, Q3_K,
    // Q2_K, IQ2/3/1 formats), the weight packer pre-decodes to INT8 at pack
    // time and stores them in VNNI-interleaved order:
    //   native_interleaved: [N_chunks][bpr][8 groups][4 ZMMs][64 bytes]
    //
    // Each group covers 4 consecutive K-elements (vs the nibble path which
    // covers 8 K-elements per group via lo/hi nibble split).
    //
    // Memory traffic per K-block: 2048 bytes (1.0 byte/element INT8)
    // This is 2× the native nibble path, but universally applicable.
    // =========================================================================

    /**
     * @brief AVX-512 VNNI GEMV for one N-chunk of 64 columns (INT8 pre-decoded path).
     *
     * Loads pre-decoded INT8 values directly from the interleaved buffer.
     * 8 groups × 4 K-elements per group = 32 K-elements per block.
     */
    inline void gemv_native_vnni_avx512_chunk_int8(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end)
    {
        __m512 fp_acc0 = _mm512_setzero_ps();
        __m512 fp_acc1 = _mm512_setzero_ps();
        __m512 fp_acc2 = _mm512_setzero_ps();
        __m512 fp_acc3 = _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            __m512i int_acc0 = _mm512_setzero_si512();
            __m512i int_acc1 = _mm512_setzero_si512();
            __m512i int_acc2 = _mm512_setzero_si512();
            __m512i int_acc3 = _mm512_setzero_si512();

            // 8 groups × 4 K-elements each = 32 K-elements per block.
            // Each group loads pre-decoded signed INT8 in VNNI-interleaved order.
            for (int group = 0; group < 8; ++group)
            {
                // Load 4 ZMMs of pre-decoded INT8 (16 cols × 4 INT8 values each)
                __m512i b0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i b1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i b2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i b3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // A broadcast for this group's 4 consecutive K-elements [group*4 .. group*4+3]
                uint8_t a_u8[4];
                a_u8[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_u8[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_u8[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_u8[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_i32;
                std::memcpy(&a_i32, a_u8, 4);
                __m512i a_bcast = _mm512_set1_epi32(a_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_bcast, b0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_bcast, b1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_bcast, b2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_bcast, b3);
            }

            // Bias correction: corrected = int_acc - 128 * comp
            // comp is INT16 (halved metadata); load via sign-extend to INT32
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i comp0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i comp1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i comp2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i comp3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            int_acc0 = _mm512_sub_epi32(int_acc0, _mm512_mullo_epi32(bias_128_i32, comp0));
            int_acc1 = _mm512_sub_epi32(int_acc1, _mm512_mullo_epi32(bias_128_i32, comp1));
            int_acc2 = _mm512_sub_epi32(int_acc2, _mm512_mullo_epi32(bias_128_i32, comp2));
            int_acc3 = _mm512_sub_epi32(int_acc3, _mm512_mullo_epi32(bias_128_i32, comp3));

            // Convert to FP32 and scale: fp_val = int32_val * a_scale * b_scale[n]
            // scales are FP16 (halved metadata); load via F16C convert to FP32
            __m512 a_scale_v = _mm512_set1_ps(a_scale);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 cs0 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales))));
            __m512 cs1 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16))));
            __m512 cs2 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32))));
            __m512 cs3 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48))));

            fp_acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc0), cs0, fp_acc0);
            fp_acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc1), cs1, fp_acc1);
            fp_acc2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc2), cs2, fp_acc2);
            fp_acc3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc3), cs3, fp_acc3);

            // Asymmetric correction: acc += a_scale * sum_qs * b_min[n]
            // mins are FP16; load via F16C convert to FP32
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 a_corr_v = _mm512_set1_ps(static_cast<float>(a_sum) * a_scale);
                fp_acc0 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins))), fp_acc0);
                fp_acc1 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16))), fp_acc1);
                fp_acc2 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32))), fp_acc2);
                fp_acc3 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48))), fp_acc3);
            }
        }

        _mm512_storeu_ps(C, fp_acc0);
        _mm512_storeu_ps(C + 16, fp_acc1);
        _mm512_storeu_ps(C + 32, fp_acc2);
        _mm512_storeu_ps(C + 48, fp_acc3);
    }

    /**
     * @brief Multi-chunk GEMV processing a block of consecutive N-chunks.
     *
     * Processes n_block_chunks consecutive 64-column chunks with L2 prefetch.
     */
    inline void gemv_native_vnni_avx512_block(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk_start,
        int chunk_count,
        int K_blocks,
        int N,
        const __m512i decode_lut)
    {
        const bool use_nibble_lut = packed.is_nibble_lut;

        for (int ci = 0; ci < chunk_count; ++ci)
        {
            int chunk = chunk_start + ci;
            int n_start = chunk * 64;
            int n_cols = std::min(64, N - n_start);

            // Prefetch next chunk's first group data into L2
            if (ci + 1 < chunk_count)
            {
                _mm_prefetch(reinterpret_cast<const char *>(packed.interleavedB(chunk + 1, 0, 0, 0)),
                             _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char *>(packed.chunkScales(chunk + 1, 0)),
                             _MM_HINT_T1);
            }

            if (n_cols < 64)
            {
                // Tail chunk: kernel writes 64 floats, but only n_cols are valid.
                // Use a temp buffer to avoid overflowing the caller's C buffer.
                alignas(64) float tmp[64];
                if (use_nibble_lut)
                    gemv_native_vnni_avx512_chunk_native(packed, A_q8, tmp, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_native_vnni_avx512_chunk_int8(packed, A_q8, tmp, chunk, 0, K_blocks);
                std::memcpy(C + n_start, tmp, n_cols * sizeof(float));
            }
            else
            {
                if (use_nibble_lut)
                    gemv_native_vnni_avx512_chunk_native(packed, A_q8, C + n_start, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_native_vnni_avx512_chunk_int8(packed, A_q8, C + n_start, chunk, 0, K_blocks);
            }
        }
    }

#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

    /**
     * @brief Reduce one 64-column K-tile partial stream in decode order.
     *
     * @param base          Address of tile zero. Consecutive tiles are 64
     *                      floats apart.
     * @param destination   Final output address for this N chunk.
     * @param valid_columns Number of valid columns in the chunk, in [1, 64].
     * @param k_tiles       Number of K partials to combine, at least two.
     * @param use_avx512    True only when runtime dispatch selected AVX512.
     *
     * Serial decode and grouped verifier kernels both partition long K in the
     * same way. Batch invariance therefore depends on adding tile 0, tile 1,
     * and so on in exactly this order for every FP32 lane. Keeping the reduction
     * here gives all callers one implementation of that contract.
     *
     * The runtime ISA branch is intentional. An AVX512 build can be launched
     * with AVX2 runtime dispatch for training or deployment, and that regime
     * must execute AVX2 instructions throughout the kernel rather than quietly
     * using an AVX512 reduction compiled into the binary.
     */
    inline void reduceNativeVNNIKTilePartialsExact(
        const float *base,
        float *destination,
        int valid_columns,
        int k_tiles,
        bool use_avx512)
    {
#if defined(__AVX512F__)
        if (use_avx512)
        {
            __m512 sum0 = _mm512_loadu_ps(base);
            __m512 sum1 = _mm512_loadu_ps(base + 16);
            __m512 sum2 = _mm512_loadu_ps(base + 32);
            __m512 sum3 = _mm512_loadu_ps(base + 48);
            for (int kt = 1; kt < k_tiles; ++kt)
            {
                const float *src = base + static_cast<size_t>(kt) * 64;
                sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(src));
                sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(src + 16));
                sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(src + 32));
                sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(src + 48));
            }

            if (valid_columns < 64)
            {
                alignas(64) float temporary[64];
                _mm512_store_ps(temporary, sum0);
                _mm512_store_ps(temporary + 16, sum1);
                _mm512_store_ps(temporary + 32, sum2);
                _mm512_store_ps(temporary + 48, sum3);
                std::memcpy(
                    destination,
                    temporary,
                    static_cast<size_t>(valid_columns) * sizeof(float));
            }
            else
            {
                _mm512_storeu_ps(destination, sum0);
                _mm512_storeu_ps(destination + 16, sum1);
                _mm512_storeu_ps(destination + 32, sum2);
                _mm512_storeu_ps(destination + 48, sum3);
            }
            return;
        }
#else
        (void)use_avx512;
#endif

        __m256 sum0 = _mm256_loadu_ps(base);
        __m256 sum1 = _mm256_loadu_ps(base + 8);
        __m256 sum2 = _mm256_loadu_ps(base + 16);
        __m256 sum3 = _mm256_loadu_ps(base + 24);
        __m256 sum4 = _mm256_loadu_ps(base + 32);
        __m256 sum5 = _mm256_loadu_ps(base + 40);
        __m256 sum6 = _mm256_loadu_ps(base + 48);
        __m256 sum7 = _mm256_loadu_ps(base + 56);
        for (int kt = 1; kt < k_tiles; ++kt)
        {
            const float *src = base + static_cast<size_t>(kt) * 64;
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(src));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(src + 8));
            sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(src + 16));
            sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(src + 24));
            sum4 = _mm256_add_ps(sum4, _mm256_loadu_ps(src + 32));
            sum5 = _mm256_add_ps(sum5, _mm256_loadu_ps(src + 40));
            sum6 = _mm256_add_ps(sum6, _mm256_loadu_ps(src + 48));
            sum7 = _mm256_add_ps(sum7, _mm256_loadu_ps(src + 56));
        }

        if (valid_columns < 64)
        {
            alignas(64) float temporary[64];
            _mm256_store_ps(temporary, sum0);
            _mm256_store_ps(temporary + 8, sum1);
            _mm256_store_ps(temporary + 16, sum2);
            _mm256_store_ps(temporary + 24, sum3);
            _mm256_store_ps(temporary + 32, sum4);
            _mm256_store_ps(temporary + 40, sum5);
            _mm256_store_ps(temporary + 48, sum6);
            _mm256_store_ps(temporary + 56, sum7);
            std::memcpy(
                destination,
                temporary,
                static_cast<size_t>(valid_columns) * sizeof(float));
        }
        else
        {
            _mm256_storeu_ps(destination, sum0);
            _mm256_storeu_ps(destination + 8, sum1);
            _mm256_storeu_ps(destination + 16, sum2);
            _mm256_storeu_ps(destination + 24, sum3);
            _mm256_storeu_ps(destination + 32, sum4);
            _mm256_storeu_ps(destination + 40, sum5);
            _mm256_storeu_ps(destination + 48, sum6);
            _mm256_storeu_ps(destination + 56, sum7);
        }
    }

    /**
     * @brief Add one projection bias row using the selected runtime ISA.
     *
     * Fused verifier projection bundles apply bias after every grouped GEMV
     * row is complete. The operation is elementwise and therefore has no
     * cross-lane reduction order, but it still belongs to the runtime dispatch
     * contract: forcing AVX2 in an AVX512 build must not execute AVX512
     * instructions in this epilogue.
     */
    inline void addNativeVNNIBiasRow(
        float *row,
        const float *bias,
        int columns,
        bool use_avx512)
    {
        int column = 0;
#if defined(__AVX512F__)
        if (use_avx512)
        {
            for (; column + 15 < columns; column += 16)
            {
                const __m512 value = _mm512_add_ps(
                    _mm512_loadu_ps(row + column),
                    _mm512_loadu_ps(bias + column));
                _mm512_storeu_ps(row + column, value);
            }
        }
#else
        (void)use_avx512;
#endif
        for (; column + 7 < columns; column += 8)
        {
            const __m256 value = _mm256_add_ps(
                _mm256_loadu_ps(row + column),
                _mm256_loadu_ps(bias + column));
            _mm256_storeu_ps(row + column, value);
        }
        for (; column < columns; ++column)
            row[column] += bias[column];
    }

    // =========================================================================
    // Pre-quantized GEMV (M=1) — compute only, skips quantization
    // =========================================================================
    //
    // Caller is responsible for providing pre-quantized Q8_1 blocks.
    // Used by multiply_fused() to avoid redundant quantization when
    // the same input is projected through multiple weight matrices.
    // =========================================================================

    inline void gemv_native_vnni_preq(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        ISAPath isa_path = ISAPath::AUTO,
        DecodeSchedulePolicy schedule_override = DecodeSchedulePolicy::Auto)
    {
        const int N = packed.N;
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;

        // Runtime ISA selection
        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);

        // Compute tile configuration
        int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(N, K, 1, packed.payload_bytes, num_threads);

        // Initialize decode LUTs for selected ISA
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i decode_lut_512 = _mm512_setzero_si512();
        if (use_avx512 && packed.is_nibble_lut)
            decode_lut_512 = build_decode_lut(packed.codebook_id);
#endif
        __m256i decode_lut_256 = _mm256_setzero_si256();
        if (use_avx2 && packed.is_nibble_lut)
            decode_lut_256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);

        if (!use_avx512 && !use_avx2)
        {
            if (isa_path != ISAPath::SCALAR ||
                schedule_override != DecodeSchedulePolicy::FrozenSerialOracle)
            {
                throw std::runtime_error(
                    "CPU NativeVNNI production decode requires an AVX2 or "
                    "AVX512 grouped implementation; scalar execution is an "
                    "explicit frozen diagnostic oracle only");
            }
            gemv_native_vnni_scalar(packed, A_q8, C, N, K_blocks);
            return;
        }

        /*
         * K partition boundaries remain the frozen serial arithmetic contract.
         * The learned policy owns task granularity only, so changing a table
         * can improve OpenMP economy without changing any FP32 parenthesization
         * inherited by grouped verifier execution.
         */
        const bool serial_kpart = cfg.k_tiles > 1;
        const DecodeSchedulePolicy requested_schedule = schedule_override;
        DecodeSchedulePolicy selected_schedule = schedule_override;
        int n_block_chunks = cfg.n_block_chunks;
        if (selected_schedule == DecodeSchedulePolicy::Auto)
        {
            selected_schedule = selectDecodeSchedulePolicy(
                packed,
                N,
                K,
                use_avx512,
                use_avx2,
                serial_kpart,
                cfg.k_tiles);
        }
        if (selected_schedule != DecodeSchedulePolicy::FrozenSerialOracle)
        {
            selected_schedule =
                normalizeDecodeSchedulePolicy(selected_schedule, N_chunks);
            n_block_chunks = decodeScheduleNBlockChunks(selected_schedule);
        }

        if (PerfStatsCollector::isEnabled())
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_native_vnni_decode_launch",
                1.0,
                "gemv",
                {},
                PerfStatsCollector::Tags{
                    {"n", std::to_string(N)},
                    {"k", std::to_string(K)},
                    {"codebook", std::to_string(packed.codebook_id)},
                    {"build_isa", compiledNativeVNNIBuildISAName()},
                    {"isa", use_avx512 ? "AVX512" : "AVX2"},
                    {"requested_policy",
                     decodeSchedulePolicyName(requested_schedule)},
                    {"effective_policy",
                     decodeSchedulePolicyName(selected_schedule)},
                    {"n_block_chunks", std::to_string(n_block_chunks)},
                    {"k_tiles", std::to_string(cfg.k_tiles)},
                    {"serial_kpart", serial_kpart ? "1" : "0"},
                    {"threads", std::to_string(num_threads)}});
        }

        // K-parallel GEMV when N-parallelism is insufficient
        if (cfg.k_tiles > 1)
        {
            int k_tiles = cfg.k_tiles;
            int k_blocks_per_tile = (K_blocks + k_tiles - 1) / k_tiles;
            /*
             * Each [N chunk, K tile] partial is fully overwritten before the
             * reduction pass.  Use default-initialized storage so long-K decode
             * verifier rows do not pay a redundant memset on every GEMV call.
             */
            const size_t partial_sum_floats =
                static_cast<size_t>(N_chunks) * k_tiles * 64;
            thread_local AlignedVector<float> partial_sums_tls;
            if (partial_sums_tls.size() < partial_sum_floats)
                partial_sums_tls.resize_uninitialized(partial_sum_floats);
            float *partial_sums = partial_sums_tls.data();

            auto do_gemv_kpar = [&]()
            {
                const int n_blocks =
                    (N_chunks + n_block_chunks - 1) / n_block_chunks;
                const int total_2d = n_blocks * k_tiles;
#pragma omp for schedule(static)
                for (int task = 0; task < total_2d; ++task)
                {
                    const int block_idx = task / k_tiles;
                    const int kt = task % k_tiles;
                    const int chunk_start = block_idx * n_block_chunks;
                    const int chunk_count =
                        std::min(n_block_chunks, N_chunks - chunk_start);
                    const int kb_start = kt * k_blocks_per_tile;
                    const int kb_end =
                        std::min(kb_start + k_blocks_per_tile, K_blocks);
                    for (int offset = 0; offset < chunk_count; ++offset)
                    {
                        const int chunk_idx = chunk_start + offset;
                        float *dest = partial_sums +
                                      (static_cast<size_t>(chunk_idx) * k_tiles + kt) * 64;
                        if (use_avx512)
                        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (packed.is_nibble_lut)
                                gemv_native_vnni_avx512_chunk_native(
                                    packed, A_q8, dest, chunk_idx, kb_start,
                                    kb_end, decode_lut_512);
                            else
                                gemv_native_vnni_avx512_chunk_int8(
                                    packed, A_q8, dest, chunk_idx, kb_start,
                                    kb_end);
#endif
                        }
                        else if (packed.is_nibble_lut)
                        {
                            gemv_avx2_chunk_native(
                                packed, A_q8, dest, chunk_idx, kb_start, kb_end,
                                decode_lut_256);
                        }
                        else
                        {
                            gemv_avx2_chunk_int8(
                                packed, A_q8, dest, chunk_idx, kb_start, kb_end);
                        }
                    }
                }

                // Reduce partial sums across K-tiles
#pragma omp for schedule(static)
                for (int chunk_idx = 0; chunk_idx < N_chunks; ++chunk_idx)
                {
                    int n_start = chunk_idx * 64;
                    int n_cols = std::min(64, N - n_start);
                    const float *base = partial_sums +
                                        static_cast<size_t>(chunk_idx) * k_tiles * 64;
                    reduceNativeVNNIKTilePartialsExact(
                        base, C + n_start, n_cols, k_tiles, use_avx512);
                }
            };

            OMP_WORKSHARE_REGION(do_gemv_kpar);
            return;
        }

        // Standard N-parallel GEMV
        int total_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

        // Serial fast path: when there are fewer tasks than threads and we're
        // not already inside a parallel region, skip OMP entirely.  For MoE
        // expert decode (N=512 → 8 tasks, N=2048 → 32 tasks with 28 threads)
        // the fork/join overhead (~5-10µs) dominates the per-call compute.
        bool serialize = !omp_in_parallel() && total_blocks < num_threads;
        if (serialize)
        {
            for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
            {
                int chunk_start = block_idx * n_block_chunks;
                int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                if (use_avx512)
                {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    gemv_native_vnni_avx512_block(packed, A_q8, C,
                                                  chunk_start, chunk_count, K_blocks, N,
                                                  decode_lut_512);
#endif
                }
                else
                {
                    gemv_avx2_block(packed, A_q8, C,
                                    chunk_start, chunk_count, K_blocks, N,
                                    decode_lut_256);
                }
            }
            return;
        }

        auto do_gemv = [&]()
        {
#pragma omp for schedule(static)
            for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
            {
                int chunk_start = block_idx * n_block_chunks;
                int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                if (use_avx512)
                {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    gemv_native_vnni_avx512_block(packed, A_q8, C,
                                                  chunk_start, chunk_count, K_blocks, N,
                                                  decode_lut_512);
#endif
                }
                else
                {
                    gemv_avx2_block(packed, A_q8, C,
                                    chunk_start, chunk_count, K_blocks, N,
                                    decode_lut_256);
                }
            }
        };

        OMP_WORKSHARE_REGION(do_gemv);
    }

    // =========================================================================
    // Full GEMV dispatcher (M=1) — unified eager-interleaved entrypoint
    // =========================================================================

    inline void gemv_native_vnni(
        const CPUNativeVNNIPackedWeights &packed,
        const float *A_fp32,
        float *C)
    {
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;

        // Step 1: Quantize activations to Q8_1 (thread-local buffer avoids heap alloc per call)
        thread_local std::vector<Q8_1Block> A_q8_tls;
        if (static_cast<int>(A_q8_tls.size()) < K_blocks)
            A_q8_tls.resize(K_blocks);
        Q8_1Block *A_q8 = A_q8_tls.data();

        // Process blocks in pairs for 2-way ILP (AVX-512)
        const bool k_aligned = (K % 32 == 0);
        int kb = 0;
#if defined(__AVX512F__)
        if (k_aligned)
        {
            for (; kb + 1 < K_blocks; kb += 2)
            {
                simd::quantize_two_blocks_avx512(A_fp32 + kb * 32, A_q8[kb], A_q8[kb + 1]);
            }
        }
#endif
        for (; kb < K_blocks; ++kb)
        {
            int block_start = kb * 32;
            int block_len = std::min(32, K - block_start);
            simd::quantize_single_block(A_fp32 + block_start, A_q8[kb], block_len);
        }

        // Step 2+: Delegate to pre-quantized compute path
        gemv_native_vnni_preq(packed, A_q8, C);
    }

    // =========================================================================
    // 2-Row GEMM Microkernels (share B loads across 2 M rows)
    // =========================================================================
    //
    // The key optimization for M>1 GEMM: load each B weight block once and
    // compute dot products for 2 rows simultaneously. This doubles the
    // compute-to-load ratio vs row-by-row GEMV.
    //
    // Register layout per 64-col chunk (2 rows):
    //   Row 0: fp_acc0..fp_acc3 (4 × __m512, 64 FP32 accumulators)
    //   Row 1: fp_acc4..fp_acc7 (4 × __m512, 64 FP32 accumulators)
    //   B data: loaded once, used for both rows
    //   A broadcasts: separate per row (different activation values)
    // =========================================================================

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

    /**
     * @brief 2-row nibble-LUT GEMM microkernel for one 64-col chunk.
     *
     * Processes rows m0 and m1 simultaneously, sharing B loads.
     * Accumulates partial results for a K-block range [kb_start, kb_end).
     * Caller must add results to existing C values when K-tiling.
     */
    inline void gemm_2row_native_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        float *C_row0,
        float *C_row1,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut,
        bool accumulate)
    {
        __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0) : _mm512_setzero_ps();
        __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + 16) : _mm512_setzero_ps();
        __m512 fp0_2 = accumulate ? _mm512_loadu_ps(C_row0 + 32) : _mm512_setzero_ps();
        __m512 fp0_3 = accumulate ? _mm512_loadu_ps(C_row0 + 48) : _mm512_setzero_ps();
        __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1) : _mm512_setzero_ps();
        __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + 16) : _mm512_setzero_ps();
        __m512 fp1_2 = accumulate ? _mm512_loadu_ps(C_row1 + 32) : _mm512_setzero_ps();
        __m512 fp1_3 = accumulate ? _mm512_loadu_ps(C_row1 + 48) : _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a0 = A_q8_row0[kb];
            const Q8_1Block &a1 = A_q8_row1[kb];
            float a0_scale = simd::fp16_to_fp32(a0.d);
            float a1_scale = simd::fp16_to_fp32(a1.d);

            __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
            __m512i ia0_2 = _mm512_setzero_si512(), ia0_3 = _mm512_setzero_si512();
            __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
            __m512i ia1_2 = _mm512_setzero_si512(), ia1_3 = _mm512_setzero_si512();

            for (int group = 0; group < 4; ++group)
            {
                // Load B once — shared across both rows
                __m512i raw0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i raw1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i raw2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i raw3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // Decode nibbles — shared across rows
                __m512i lo0 = decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo1 = decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo2 = decode_low_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i lo3 = decode_low_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                __m512i hi0 = decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi1 = decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi2 = decode_high_nibbles_avx512(raw2, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                __m512i hi3 = decode_high_nibbles_avx512(raw3, decode_lut, decode_kind, mask_0F, q4_zero_offset);

                // Row 0 A broadcasts + VPDPBUSD
                {
                    uint8_t vals[4];
                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 0]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 1]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 2]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 3]) + 128);
                    int32_t v;
                    std::memcpy(&v, vals, 4);
                    __m512i ab = _mm512_set1_epi32(v);
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, ab, lo0);
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, ab, lo1);
                    ia0_2 = _mm512_dpbusd_epi32(ia0_2, ab, lo2);
                    ia0_3 = _mm512_dpbusd_epi32(ia0_3, ab, lo3);

                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 16]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 17]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 18]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a0.qs[group * 4 + 19]) + 128);
                    std::memcpy(&v, vals, 4);
                    ab = _mm512_set1_epi32(v);
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, ab, hi0);
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, ab, hi1);
                    ia0_2 = _mm512_dpbusd_epi32(ia0_2, ab, hi2);
                    ia0_3 = _mm512_dpbusd_epi32(ia0_3, ab, hi3);
                }

                // Row 1 A broadcasts + VPDPBUSD (same B data, different A)
                {
                    uint8_t vals[4];
                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 0]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 1]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 2]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 3]) + 128);
                    int32_t v;
                    std::memcpy(&v, vals, 4);
                    __m512i ab = _mm512_set1_epi32(v);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, ab, lo0);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, ab, lo1);
                    ia1_2 = _mm512_dpbusd_epi32(ia1_2, ab, lo2);
                    ia1_3 = _mm512_dpbusd_epi32(ia1_3, ab, lo3);

                    vals[0] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 16]) + 128);
                    vals[1] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 17]) + 128);
                    vals[2] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 18]) + 128);
                    vals[3] = static_cast<uint8_t>(static_cast<int16_t>(a1.qs[group * 4 + 19]) + 128);
                    std::memcpy(&v, vals, 4);
                    ab = _mm512_set1_epi32(v);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, ab, hi0);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, ab, hi1);
                    ia1_2 = _mm512_dpbusd_epi32(ia1_2, ab, hi2);
                    ia1_3 = _mm512_dpbusd_epi32(ia1_3, ab, hi3);
                }
            }

            // Bias correction + scale (shared comp/scales loads, per-row a_scale)
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i c0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i c1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i c2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i c3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            __m512i bias_c0 = _mm512_mullo_epi32(bias_128_i32, c0);
            __m512i bias_c1 = _mm512_mullo_epi32(bias_128_i32, c1);
            __m512i bias_c2 = _mm512_mullo_epi32(bias_128_i32, c2);
            __m512i bias_c3 = _mm512_mullo_epi32(bias_128_i32, c3);

            ia0_0 = _mm512_sub_epi32(ia0_0, bias_c0);
            ia0_1 = _mm512_sub_epi32(ia0_1, bias_c1);
            ia0_2 = _mm512_sub_epi32(ia0_2, bias_c2);
            ia0_3 = _mm512_sub_epi32(ia0_3, bias_c3);
            ia1_0 = _mm512_sub_epi32(ia1_0, bias_c0);
            ia1_1 = _mm512_sub_epi32(ia1_1, bias_c1);
            ia1_2 = _mm512_sub_epi32(ia1_2, bias_c2);
            ia1_3 = _mm512_sub_epi32(ia1_3, bias_c3);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 bs0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales)));
            __m512 bs1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16)));
            __m512 bs2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32)));
            __m512 bs3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48)));

            __m512 as0 = _mm512_set1_ps(a0_scale);
            __m512 cs0_0 = _mm512_mul_ps(as0, bs0), cs0_1 = _mm512_mul_ps(as0, bs1);
            __m512 cs0_2 = _mm512_mul_ps(as0, bs2), cs0_3 = _mm512_mul_ps(as0, bs3);
            fp0_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_0), cs0_0, fp0_0);
            fp0_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_1), cs0_1, fp0_1);
            fp0_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_2), cs0_2, fp0_2);
            fp0_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_3), cs0_3, fp0_3);

            __m512 as1 = _mm512_set1_ps(a1_scale);
            __m512 cs1_0 = _mm512_mul_ps(as1, bs0), cs1_1 = _mm512_mul_ps(as1, bs1);
            __m512 cs1_2 = _mm512_mul_ps(as1, bs2), cs1_3 = _mm512_mul_ps(as1, bs3);
            fp1_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_0), cs1_0, fp1_0);
            fp1_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_1), cs1_1, fp1_1);
            fp1_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_2), cs1_2, fp1_2);
            fp1_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_3), cs1_3, fp1_3);

            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 bm0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins)));
                __m512 bm1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16)));
                __m512 bm2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32)));
                __m512 bm3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48)));

                __m512 corr0 = _mm512_set1_ps(static_cast<float>(a0.sum_qs) * a0_scale);
                fp0_0 = _mm512_fmadd_ps(corr0, bm0, fp0_0);
                fp0_1 = _mm512_fmadd_ps(corr0, bm1, fp0_1);
                fp0_2 = _mm512_fmadd_ps(corr0, bm2, fp0_2);
                fp0_3 = _mm512_fmadd_ps(corr0, bm3, fp0_3);

                __m512 corr1 = _mm512_set1_ps(static_cast<float>(a1.sum_qs) * a1_scale);
                fp1_0 = _mm512_fmadd_ps(corr1, bm0, fp1_0);
                fp1_1 = _mm512_fmadd_ps(corr1, bm1, fp1_1);
                fp1_2 = _mm512_fmadd_ps(corr1, bm2, fp1_2);
                fp1_3 = _mm512_fmadd_ps(corr1, bm3, fp1_3);
            }
        }

        _mm512_storeu_ps(C_row0, fp0_0);
        _mm512_storeu_ps(C_row0 + 16, fp0_1);
        _mm512_storeu_ps(C_row0 + 32, fp0_2);
        _mm512_storeu_ps(C_row0 + 48, fp0_3);
        _mm512_storeu_ps(C_row1, fp1_0);
        _mm512_storeu_ps(C_row1 + 16, fp1_1);
        _mm512_storeu_ps(C_row1 + 32, fp1_2);
        _mm512_storeu_ps(C_row1 + 48, fp1_3);
    }

    /**
     * @brief 2-row INT8 pre-decoded GEMM microkernel for one 64-col chunk.
     */
    inline void gemm_2row_int8_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        float *C_row0,
        float *C_row1,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0) : _mm512_setzero_ps();
        __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + 16) : _mm512_setzero_ps();
        __m512 fp0_2 = accumulate ? _mm512_loadu_ps(C_row0 + 32) : _mm512_setzero_ps();
        __m512 fp0_3 = accumulate ? _mm512_loadu_ps(C_row0 + 48) : _mm512_setzero_ps();
        __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1) : _mm512_setzero_ps();
        __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + 16) : _mm512_setzero_ps();
        __m512 fp1_2 = accumulate ? _mm512_loadu_ps(C_row1 + 32) : _mm512_setzero_ps();
        __m512 fp1_3 = accumulate ? _mm512_loadu_ps(C_row1 + 48) : _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a0 = A_q8_row0[kb];
            const Q8_1Block &a1 = A_q8_row1[kb];
            float a0_scale = simd::fp16_to_fp32(a0.d);
            float a1_scale = simd::fp16_to_fp32(a1.d);

            __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
            __m512i ia0_2 = _mm512_setzero_si512(), ia0_3 = _mm512_setzero_si512();
            __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
            __m512i ia1_2 = _mm512_setzero_si512(), ia1_3 = _mm512_setzero_si512();

            for (int group = 0; group < 8; ++group)
            {
                // GPR-only A-prep: XOR with 0x80 converts signed→unsigned bytes,
                // avoids xmm intermediaries that alias zmm accumulators.
                // vpbroadcastd has a GPR source form on AVX-512 (no xmm needed).
                __m512i a0_bc =
                    _mm512_set1_epi32(pack_q8_1_unsigned_word(a0, group * 4));
                __m512i a1_bc =
                    _mm512_set1_epi32(pack_q8_1_unsigned_word(a1, group * 4));

// Load-use-discard: 1 B register live at a time (not 4).
// Peak: 8 INT32 + 8 FP32 + 2 A + 1 B + 1 const = 20 ZMMs.
#define INT8_SUBCHUNK(Z, IA0, IA1)                                               \
    {                                                                            \
        __m512i b = _mm512_load_si512(packed.interleavedB(chunk, kb, group, Z)); \
        IA0 = _mm512_dpbusd_epi32(IA0, a0_bc, b);                                \
        IA1 = _mm512_dpbusd_epi32(IA1, a1_bc, b);                                \
    }
                INT8_SUBCHUNK(0, ia0_0, ia1_0)
                INT8_SUBCHUNK(1, ia0_1, ia1_1)
                INT8_SUBCHUNK(2, ia0_2, ia1_2)
                INT8_SUBCHUNK(3, ia0_3, ia1_3)
#undef INT8_SUBCHUNK
            }

            // Bias correction (shared comp loads)
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i cc0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i cc1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i cc2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i cc3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            __m512i bc0 = _mm512_mullo_epi32(bias_128_i32, cc0);
            __m512i bc1 = _mm512_mullo_epi32(bias_128_i32, cc1);
            __m512i bc2 = _mm512_mullo_epi32(bias_128_i32, cc2);
            __m512i bc3 = _mm512_mullo_epi32(bias_128_i32, cc3);

            ia0_0 = _mm512_sub_epi32(ia0_0, bc0);
            ia0_1 = _mm512_sub_epi32(ia0_1, bc1);
            ia0_2 = _mm512_sub_epi32(ia0_2, bc2);
            ia0_3 = _mm512_sub_epi32(ia0_3, bc3);
            ia1_0 = _mm512_sub_epi32(ia1_0, bc0);
            ia1_1 = _mm512_sub_epi32(ia1_1, bc1);
            ia1_2 = _mm512_sub_epi32(ia1_2, bc2);
            ia1_3 = _mm512_sub_epi32(ia1_3, bc3);

            // Scale (shared b_scales loads)
            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 bs0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales)));
            __m512 bs1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16)));
            __m512 bs2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32)));
            __m512 bs3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48)));

            __m512 as0 = _mm512_set1_ps(a0_scale);
            fp0_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_0), _mm512_mul_ps(as0, bs0), fp0_0);
            fp0_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_1), _mm512_mul_ps(as0, bs1), fp0_1);
            fp0_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_2), _mm512_mul_ps(as0, bs2), fp0_2);
            fp0_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia0_3), _mm512_mul_ps(as0, bs3), fp0_3);

            __m512 as1 = _mm512_set1_ps(a1_scale);
            fp1_0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_0), _mm512_mul_ps(as1, bs0), fp1_0);
            fp1_1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_1), _mm512_mul_ps(as1, bs1), fp1_1);
            fp1_2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_2), _mm512_mul_ps(as1, bs2), fp1_2);
            fp1_3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia1_3), _mm512_mul_ps(as1, bs3), fp1_3);

            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 bm0 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins)));
                __m512 bm1 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16)));
                __m512 bm2 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32)));
                __m512 bm3 = _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48)));

                __m512 corr0 = _mm512_set1_ps(static_cast<float>(a0.sum_qs) * a0_scale);
                fp0_0 = _mm512_fmadd_ps(corr0, bm0, fp0_0);
                fp0_1 = _mm512_fmadd_ps(corr0, bm1, fp0_1);
                fp0_2 = _mm512_fmadd_ps(corr0, bm2, fp0_2);
                fp0_3 = _mm512_fmadd_ps(corr0, bm3, fp0_3);

                __m512 corr1 = _mm512_set1_ps(static_cast<float>(a1.sum_qs) * a1_scale);
                fp1_0 = _mm512_fmadd_ps(corr1, bm0, fp1_0);
                fp1_1 = _mm512_fmadd_ps(corr1, bm1, fp1_1);
                fp1_2 = _mm512_fmadd_ps(corr1, bm2, fp1_2);
                fp1_3 = _mm512_fmadd_ps(corr1, bm3, fp1_3);
            }
        }

        _mm512_storeu_ps(C_row0, fp0_0);
        _mm512_storeu_ps(C_row0 + 16, fp0_1);
        _mm512_storeu_ps(C_row0 + 32, fp0_2);
        _mm512_storeu_ps(C_row0 + 48, fp0_3);
        _mm512_storeu_ps(C_row1, fp1_0);
        _mm512_storeu_ps(C_row1 + 16, fp1_1);
        _mm512_storeu_ps(C_row1 + 32, fp1_2);
        _mm512_storeu_ps(C_row1 + 48, fp1_3);
    }

    /**
     * @brief Three-row AVX512 verifier microkernel for nibble-LUT formats.
     *
     * MTP commonly verifies `draft_count + 1` rows.  For draft depth two, that
     * means M=3 target rows.  The older safe path ran one 2-row verifier chunk
     * plus one serial GEMV tail, which preserved decode equivalence but loaded
     * and decoded the same packed B chunk twice.  This kernel shares each
     * decoded B vector across three independent row accumulators while keeping
     * the per-row K-block, compensation, scale, and min-correction order
     * identical to serial decode GEMV.
     */
    inline void gemm_3row_native_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();

                for (int group = 0; group < 4; ++group)
                {
                    const __m512i raw0 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    const __m512i raw1 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    const __m512i lo0 =
                        decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i lo1 =
                        decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi0 =
                        decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi1 =
                        decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);

#define NIBBLE_3ROW_ACCUM(ROW, ABLK)                                             \
                    {                                                            \
                        const __m512i a_lo = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4));           \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_lo, lo0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_lo, lo1); \
                        const __m512i a_hi = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4 + 16));      \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_hi, hi0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_hi, hi1); \
                    }
                    NIBBLE_3ROW_ACCUM(0, a0)
                    NIBBLE_3ROW_ACCUM(1, a1)
                    NIBBLE_3ROW_ACCUM(2, a2)
#undef NIBBLE_3ROW_ACCUM
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);

#define FMA_3ROW_NATIVE(ROW, ASCALE)                                            \
                {                                                               \
                    const __m512 as = _mm512_set1_ps(ASCALE);                   \
                    fp##ROW##_0 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_0),                        \
                        _mm512_mul_ps(as, bscale0),                             \
                        fp##ROW##_0);                                           \
                    fp##ROW##_1 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_1),                        \
                        _mm512_mul_ps(as, bscale1),                             \
                        fp##ROW##_1);                                           \
                }
                FMA_3ROW_NATIVE(0, a0_scale)
                FMA_3ROW_NATIVE(1, a1_scale)
                FMA_3ROW_NATIVE(2, a2_scale)
#undef FMA_3ROW_NATIVE

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_3ROW_NATIVE(ROW, ABLK, ASCALE)                                      \
                    {                                                           \
                        const __m512 corr = _mm512_set1_ps(                     \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);          \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_3ROW_NATIVE(0, a0, a0_scale)
                    MIN_3ROW_NATIVE(1, a1, a1_scale)
                    MIN_3ROW_NATIVE(2, a2, a2_scale)
#undef MIN_3ROW_NATIVE
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
        }
    }

    /**
     * @brief Three-row AVX512 verifier microkernel for INT8-predecoded formats.
     *
     * This is the INT8-layout sibling of gemm_3row_native_2z_chunk().  It is
     * decode-equivalent to three serial M=1 NativeVNNI GEMVs and shares each
     * interleaved B vector across all three verifier rows.
     */
    inline void gemm_3row_int8_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();

                for (int group = 0; group < 8; ++group)
                {
                    const __m512i a0_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a0, group * 4));
                    const __m512i a1_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a1, group * 4));
                    const __m512i a2_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a2, group * 4));

                    const __m512i b0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, a0_bc, b0);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, a1_bc, b0);
                    ia2_0 = _mm512_dpbusd_epi32(ia2_0, a2_bc, b0);

                    const __m512i b1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, a0_bc, b1);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, a1_bc, b1);
                    ia2_1 = _mm512_dpbusd_epi32(ia2_1, a2_bc, b1);
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);

#define FMA_3ROW_INT8(ROW, ASCALE)                                              \
                {                                                               \
                    const __m512 as = _mm512_set1_ps(ASCALE);                   \
                    fp##ROW##_0 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_0),                        \
                        _mm512_mul_ps(as, bscale0),                             \
                        fp##ROW##_0);                                           \
                    fp##ROW##_1 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_1),                        \
                        _mm512_mul_ps(as, bscale1),                             \
                        fp##ROW##_1);                                           \
                }
                FMA_3ROW_INT8(0, a0_scale)
                FMA_3ROW_INT8(1, a1_scale)
                FMA_3ROW_INT8(2, a2_scale)
#undef FMA_3ROW_INT8

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_3ROW_INT8(ROW, ABLK, ASCALE)                                        \
                    {                                                           \
                        const __m512 corr = _mm512_set1_ps(                     \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);          \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_3ROW_INT8(0, a0, a0_scale)
                    MIN_3ROW_INT8(1, a1, a1_scale)
                    MIN_3ROW_INT8(2, a2, a2_scale)
#undef MIN_3ROW_INT8
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
        }
    }

    /**
     * @brief Four-row AVX512 verifier microkernel for nibble-LUT formats.
     *
     * Q4_0, Q4_1/Q4_K, IQ4_NL/IQ4_XS and their aliases decode packed nibbles
     * through the same semantic helper as the M=1 GEMV path: linear Q4
     * codebooks use byte arithmetic, while non-linear IQ4 codebooks use the
     * per-codebook LUT.  The verifier-specialized kernel keeps the exact M=1
     * accumulation order: each K block first builds the INT32 dot-product
     * accumulators, then applies compensation, scale and optional min
     * correction once.  It simply shares the decoded B vectors across four
     * independent verifier rows.
     *
     * The implementation processes two 16-column z-lanes at a time.  That is a
     * deliberate register-pressure compromise: it gives B decode reuse for M=4
     * without carrying the full 4-row x 64-column accumulator set live at once.
     */
    inline void gemm_4row_native_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        const Q8_1Block *A_q8_row3,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        float *C_row3,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const NibbleDecodeKind decode_kind = nibbleDecodeKind(packed.codebook_id);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);
        const __m512i q4_zero_offset = _mm512_set1_epi8(8);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp3_0 = accumulate ? _mm512_loadu_ps(C_row3 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp3_1 = accumulate ? _mm512_loadu_ps(C_row3 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];
                const Q8_1Block &a3 = A_q8_row3[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();
                __m512i ia3_0 = _mm512_setzero_si512(), ia3_1 = _mm512_setzero_si512();

                for (int group = 0; group < 4; ++group)
                {
                    const __m512i raw0 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    const __m512i raw1 =
                        _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    const __m512i lo0 =
                        decode_low_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i lo1 =
                        decode_low_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi0 =
                        decode_high_nibbles_avx512(raw0, decode_lut, decode_kind, mask_0F, q4_zero_offset);
                    const __m512i hi1 =
                        decode_high_nibbles_avx512(raw1, decode_lut, decode_kind, mask_0F, q4_zero_offset);

#define NIBBLE_4ROW_ACCUM(ROW, ABLK)                                             \
                    {                                                            \
                        const __m512i a_lo = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4));           \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_lo, lo0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_lo, lo1); \
                        const __m512i a_hi = _mm512_set1_epi32(                  \
                            pack_q8_1_unsigned_word(ABLK, group * 4 + 16));      \
                        ia##ROW##_0 = _mm512_dpbusd_epi32(ia##ROW##_0, a_hi, hi0); \
                        ia##ROW##_1 = _mm512_dpbusd_epi32(ia##ROW##_1, a_hi, hi1); \
                    }
                    NIBBLE_4ROW_ACCUM(0, a0)
                    NIBBLE_4ROW_ACCUM(1, a1)
                    NIBBLE_4ROW_ACCUM(2, a2)
                    NIBBLE_4ROW_ACCUM(3, a3)
#undef NIBBLE_4ROW_ACCUM
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia3_0 = _mm512_sub_epi32(ia3_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);
                ia3_1 = _mm512_sub_epi32(ia3_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));

#define FMA_4ROW_NATIVE(ROW, ABLK, ASCALE)                                      \
                {                                                               \
                    const __m512 as = _mm512_set1_ps(ASCALE);                   \
                    fp##ROW##_0 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_0),                        \
                        _mm512_mul_ps(as, bscale0),                             \
                        fp##ROW##_0);                                           \
                    fp##ROW##_1 = _mm512_fmadd_ps(                              \
                        _mm512_cvtepi32_ps(ia##ROW##_1),                        \
                        _mm512_mul_ps(as, bscale1),                             \
                        fp##ROW##_1);                                           \
                }
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);
                const float a3_scale = simd::fp16_to_fp32(a3.d);
                FMA_4ROW_NATIVE(0, a0, a0_scale)
                FMA_4ROW_NATIVE(1, a1, a1_scale)
                FMA_4ROW_NATIVE(2, a2, a2_scale)
                FMA_4ROW_NATIVE(3, a3, a3_scale)
#undef FMA_4ROW_NATIVE

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_4ROW_NATIVE(ROW, ABLK, ASCALE)                                      \
                    {                                                           \
                        const __m512 corr = _mm512_set1_ps(                     \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);          \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_4ROW_NATIVE(0, a0, a0_scale)
                    MIN_4ROW_NATIVE(1, a1, a1_scale)
                    MIN_4ROW_NATIVE(2, a2, a2_scale)
                    MIN_4ROW_NATIVE(3, a3, a3_scale)
#undef MIN_4ROW_NATIVE
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
            _mm512_storeu_ps(C_row3 + zbase * 16, fp3_0);
            _mm512_storeu_ps(C_row3 + (zbase + 1) * 16, fp3_1);
        }
    }

    /**
     * @brief Four-row AVX512 verifier microkernel for INT8-predecoded formats.
     *
     * This is the first real M=4 CPU verifier candidate.  It keeps the same
     * decode-equivalent order as four independent one-token GEMVs:
     * for each K block, compute the INT32 dot for a column group, apply the
     * same compensation/scales/mins, then accumulate into FP32.  The economy
     * comes from loading each interleaved B vector once and applying it to four
     * independent activation rows.
     *
     * The kernel works on two 16-column subchunks at a time.  That gives useful
     * B reuse without the register pressure of a full 4-row x 64-column kernel.
     */
    inline void gemm_4row_int8_2z_chunk(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        const Q8_1Block *A_q8_row2,
        const Q8_1Block *A_q8_row3,
        float *C_row0,
        float *C_row1,
        float *C_row2,
        float *C_row3,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int zbase = 0; zbase < 4; zbase += 2)
        {
            __m512 fp0_0 = accumulate ? _mm512_loadu_ps(C_row0 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp0_1 = accumulate ? _mm512_loadu_ps(C_row0 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp1_0 = accumulate ? _mm512_loadu_ps(C_row1 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp1_1 = accumulate ? _mm512_loadu_ps(C_row1 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp2_0 = accumulate ? _mm512_loadu_ps(C_row2 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp2_1 = accumulate ? _mm512_loadu_ps(C_row2 + (zbase + 1) * 16) : _mm512_setzero_ps();
            __m512 fp3_0 = accumulate ? _mm512_loadu_ps(C_row3 + zbase * 16) : _mm512_setzero_ps();
            __m512 fp3_1 = accumulate ? _mm512_loadu_ps(C_row3 + (zbase + 1) * 16) : _mm512_setzero_ps();

            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                const Q8_1Block &a2 = A_q8_row2[kb];
                const Q8_1Block &a3 = A_q8_row3[kb];

                __m512i ia0_0 = _mm512_setzero_si512(), ia0_1 = _mm512_setzero_si512();
                __m512i ia1_0 = _mm512_setzero_si512(), ia1_1 = _mm512_setzero_si512();
                __m512i ia2_0 = _mm512_setzero_si512(), ia2_1 = _mm512_setzero_si512();
                __m512i ia3_0 = _mm512_setzero_si512(), ia3_1 = _mm512_setzero_si512();

                for (int group = 0; group < 8; ++group)
                {
                    const __m512i a0_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a0, group * 4));
                    const __m512i a1_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a1, group * 4));
                    const __m512i a2_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a2, group * 4));
                    const __m512i a3_bc =
                        _mm512_set1_epi32(pack_q8_1_unsigned_word(a3, group * 4));

                    const __m512i b0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase));
                    ia0_0 = _mm512_dpbusd_epi32(ia0_0, a0_bc, b0);
                    ia1_0 = _mm512_dpbusd_epi32(ia1_0, a1_bc, b0);
                    ia2_0 = _mm512_dpbusd_epi32(ia2_0, a2_bc, b0);
                    ia3_0 = _mm512_dpbusd_epi32(ia3_0, a3_bc, b0);

                    const __m512i b1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, zbase + 1));
                    ia0_1 = _mm512_dpbusd_epi32(ia0_1, a0_bc, b1);
                    ia1_1 = _mm512_dpbusd_epi32(ia1_1, a1_bc, b1);
                    ia2_1 = _mm512_dpbusd_epi32(ia2_1, a2_bc, b1);
                    ia3_1 = _mm512_dpbusd_epi32(ia3_1, a3_bc, b1);
                }

                const int16_t *comp_ptr = packed.chunkComp(chunk, kb) + zbase * 16;
                const __m512i comp0 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
                const __m512i comp1 =
                    _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
                const __m512i bias0 = _mm512_mullo_epi32(bias_128_i32, comp0);
                const __m512i bias1 = _mm512_mullo_epi32(bias_128_i32, comp1);

                ia0_0 = _mm512_sub_epi32(ia0_0, bias0);
                ia1_0 = _mm512_sub_epi32(ia1_0, bias0);
                ia2_0 = _mm512_sub_epi32(ia2_0, bias0);
                ia3_0 = _mm512_sub_epi32(ia3_0, bias0);
                ia0_1 = _mm512_sub_epi32(ia0_1, bias1);
                ia1_1 = _mm512_sub_epi32(ia1_1, bias1);
                ia2_1 = _mm512_sub_epi32(ia2_1, bias1);
                ia3_1 = _mm512_sub_epi32(ia3_1, bias1);

                const uint16_t *scale_ptr = packed.chunkScales(chunk, kb) + zbase * 16;
                const __m512 bscale0 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr)));
                const __m512 bscale1 =
                    _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(scale_ptr + 16)));

#define FMA_4ROW_INT8(ROW, ABLK, ASCALE)                                           \
                {                                                                  \
                    const __m512 as = _mm512_set1_ps(ASCALE);                      \
                    fp##ROW##_0 = _mm512_fmadd_ps(                                 \
                        _mm512_cvtepi32_ps(ia##ROW##_0), _mm512_mul_ps(as, bscale0), fp##ROW##_0); \
                    fp##ROW##_1 = _mm512_fmadd_ps(                                 \
                        _mm512_cvtepi32_ps(ia##ROW##_1), _mm512_mul_ps(as, bscale1), fp##ROW##_1); \
                }
                const float a0_scale = simd::fp16_to_fp32(a0.d);
                const float a1_scale = simd::fp16_to_fp32(a1.d);
                const float a2_scale = simd::fp16_to_fp32(a2.d);
                const float a3_scale = simd::fp16_to_fp32(a3.d);
                FMA_4ROW_INT8(0, a0, a0_scale)
                FMA_4ROW_INT8(1, a1, a1_scale)
                FMA_4ROW_INT8(2, a2, a2_scale)
                FMA_4ROW_INT8(3, a3, a3_scale)
#undef FMA_4ROW_INT8

                if (packed.is_asymmetric)
                {
                    const uint16_t *min_ptr = packed.chunkMins(chunk, kb) + zbase * 16;
                    const __m512 bmin0 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr)));
                    const __m512 bmin1 =
                        _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(min_ptr + 16)));

#define MIN_4ROW_INT8(ROW, ABLK, ASCALE)                                  \
                    {                                                      \
                        const __m512 corr = _mm512_set1_ps(                \
                            static_cast<float>(ABLK.sum_qs) * ASCALE);     \
                        fp##ROW##_0 = _mm512_fmadd_ps(corr, bmin0, fp##ROW##_0); \
                        fp##ROW##_1 = _mm512_fmadd_ps(corr, bmin1, fp##ROW##_1); \
                    }
                    MIN_4ROW_INT8(0, a0, a0_scale)
                    MIN_4ROW_INT8(1, a1, a1_scale)
                    MIN_4ROW_INT8(2, a2, a2_scale)
                    MIN_4ROW_INT8(3, a3, a3_scale)
#undef MIN_4ROW_INT8
                }
            }

            _mm512_storeu_ps(C_row0 + zbase * 16, fp0_0);
            _mm512_storeu_ps(C_row0 + (zbase + 1) * 16, fp0_1);
            _mm512_storeu_ps(C_row1 + zbase * 16, fp1_0);
            _mm512_storeu_ps(C_row1 + (zbase + 1) * 16, fp1_1);
            _mm512_storeu_ps(C_row2 + zbase * 16, fp2_0);
            _mm512_storeu_ps(C_row2 + (zbase + 1) * 16, fp2_1);
            _mm512_storeu_ps(C_row3 + zbase * 16, fp3_0);
            _mm512_storeu_ps(C_row3 + (zbase + 1) * 16, fp3_1);
        }
    }

#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

    // =========================================================================
    // Full GEMM dispatcher (M>1) — dual-strategy
    // =========================================================================
    //
    // Strategy 1 — Small-N (N-tasks <= threads/4):
    //   M×N 2D task grid. Tasks ordered (chunk, m-row) so consecutive
    //   M rows for the same chunk land on the same thread (B stays in L2).
    //   Each task calls the GEMV chunk kernel directly — no new SIMD code.
    //   Activates when N-only parallelism fills <25% of threads.
    //
    // Strategy 2 — Tiled GEMM (N-tasks > threads/4):
    //   Outer loop: parallel over N-block chunks
    //   Middle loop: K-tiles (each fits in L2, reused across all M rows)
    //   Inner loop: M rows (2 at a time via 2-row microkernel)
    //   B-tile reuse: each N-chunk × K-tile loaded once, scanned M times.
    // =========================================================================

    // =========================================================================
    // Shared activation quantization (for multiply_fused quantize-once)
    // =========================================================================

    inline void quantize_activations_to_q8_1(
        const float *A_fp32,
        Q8_1Block *A_q8,
        int M,
        int K,
        int K_blocks)
    {
        const bool k_aligned = (K % 32 == 0);

        auto do_quantize = [&]()
        {
#pragma omp for schedule(static)
            for (int m = 0; m < M; ++m)
            {
                const float *row_a = A_fp32 + m * K;
                Q8_1Block *row_q8 = A_q8 + static_cast<size_t>(m) * K_blocks;
                int kb = 0;
#if defined(__AVX512F__)
                if (k_aligned)
                {
                    for (; kb + 1 < K_blocks; kb += 2)
                    {
                        simd::quantize_two_blocks_avx512(row_a + kb * 32, row_q8[kb], row_q8[kb + 1]);
                    }
                }
#endif
                for (; kb < K_blocks; ++kb)
                {
                    int block_start = kb * 32;
                    int block_len = std::min(32, K - block_start);
                    simd::quantize_single_block(row_a + block_start, row_q8[kb], block_len);
                }
            }
        };

        OMP_WORKSHARE_REGION(do_quantize);
    }

    // =========================================================================
    // Pre-quantized GEMM (M>1) — compute only, skips quantization
    // =========================================================================
    //
    // A_q8_all: Pre-quantized Q8_1 blocks, [M * K_blocks] contiguous layout.
    //           Row m starts at A_q8_all + m * K_blocks.
    // =========================================================================

    inline void gemm_native_vnni_preq_decode_equivalent_rows(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_all,
        float *C,
        int M,
        int ldc,
        ISAPath isa_path,
        VerifierRowsPolicy verifier_policy_override,
        bool publish_verifier_route,
        int full_k_n_block_chunks_override = 0);

    inline void gemm_native_vnni_preq(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_all,
        float *C,
        int M,
        int ldc,
        ISAPath isa_path = ISAPath::AUTO,
        VerifierRowsPolicy verifier_policy_override = VerifierRowsPolicy::Auto,
        PrefillSchedulePolicy schedule_override = PrefillSchedulePolicy::Auto,
        int n_block_chunks_override = 0)
    {
        const int N = packed.N;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;

        // Runtime ISA selection
        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);

        int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(N, packed.K, M, packed.payload_bytes, num_threads);
        int k_tile_blocks = cfg.k_tile_blocks > 0 ? cfg.k_tile_blocks : K_blocks;
        int num_k_tiles = (K_blocks + k_tile_blocks - 1) / k_tile_blocks;
        const NativeVNNITileConfig serial_cfg = computeTileConfig(
            N, packed.K, 1, packed.payload_bytes, num_threads);
        const bool use_decode_equivalent_kpart = serial_cfg.k_tiles > 1;

        generated::CPUNativeVNNIPrefillPolicy generated_policy{};
        const bool use_generated_policy =
            verifier_policy_override == VerifierRowsPolicy::Auto &&
            schedule_override == PrefillSchedulePolicy::Auto;
        if (use_generated_policy)
        {
            if (ldc < 64)
            {
                throw std::runtime_error(
                    "Generated CPU NativeVNNI prefill policy does not admit "
                    "an output stride narrower than one 64-value chunk");
            }
            generated_policy = selectPrefillPolicy(
                packed,
                M,
                N,
                packed.K,
                use_avx512,
                use_avx2,
                use_decode_equivalent_kpart);
        }

        int n_block_chunks = cfg.n_block_chunks;
        if (n_block_chunks_override < 0)
        {
            throw std::invalid_argument(
                "CPU NativeVNNI prefill N-block override must be non-negative");
        }
        if (n_block_chunks_override > 0)
        {
            if (use_generated_policy)
            {
                throw std::invalid_argument(
                    "CPU NativeVNNI generated prefill policy cannot be combined "
                    "with an explicit N-block override");
            }
            n_block_chunks = n_block_chunks_override;
        }
        bool use_row_chunk_grid =
            (N_chunks + n_block_chunks - 1) / n_block_chunks <=
                    num_threads / 4 &&
                M >= 2;
        bool use_two_row_pair_grid = false;
        if (!use_generated_policy)
        {
            switch (schedule_override)
            {
            case PrefillSchedulePolicy::Auto:
                use_row_chunk_grid = use_row_chunk_grid || ldc < 64;
                break;
            case PrefillSchedulePolicy::RowChunkGrid:
                use_row_chunk_grid = true;
                break;
            case PrefillSchedulePolicy::TwoRowNMajor:
                use_row_chunk_grid = false;
                break;
            case PrefillSchedulePolicy::TwoRowPairGrid:
                use_row_chunk_grid = false;
                use_two_row_pair_grid = true;
                break;
            }
        }
        else
        {
            /*
             * Decode stable ordinals rather than naming every generated enum
             * member. A newly admitted candidate family must compile while the
             * previous development-only table is still installed; only the
             * regenerated certified include can return its new ordinals.
             */
            switch (static_cast<uint8_t>(generated_policy))
            {
            case 0: // RowChunkGrid
                use_row_chunk_grid = true;
                break;
            case 1: // TwoRowNbc1
                n_block_chunks = 1;
                use_row_chunk_grid = false;
                break;
            case 2: // TwoRowNbc2
                n_block_chunks = 2;
                use_row_chunk_grid = false;
                break;
            case 3: // TwoRowNbc4
                n_block_chunks = 4;
                use_row_chunk_grid = false;
                break;
            case 4: // TwoRowNbc8
                n_block_chunks = 8;
                use_row_chunk_grid = false;
                break;
            case 5: // TwoRowNbc16
                n_block_chunks = 16;
                use_row_chunk_grid = false;
                break;
            case 8: // TwoRowPairGridNbc1
                n_block_chunks = 1;
                use_row_chunk_grid = false;
                use_two_row_pair_grid = true;
                break;
            case 9: // TwoRowPairGridNbc2
                n_block_chunks = 2;
                use_row_chunk_grid = false;
                use_two_row_pair_grid = true;
                break;
            case 10: // TwoRowPairGridNbc4
                n_block_chunks = 4;
                use_row_chunk_grid = false;
                use_two_row_pair_grid = true;
                break;
            case 11: // TwoRowPairGridNbc8
                n_block_chunks = 8;
                use_row_chunk_grid = false;
                use_two_row_pair_grid = true;
                break;
            case 12: // TwoRowPairGridNbc16
                n_block_chunks = 16;
                use_row_chunk_grid = false;
                use_two_row_pair_grid = true;
                break;
            case 6: // KPartPairwise
            case 7: // KPartWideRows
                if (!use_decode_equivalent_kpart)
                {
                    throw std::runtime_error(
                        "Generated CPU prefill K-part policy disagrees with "
                        "the production serial-M1 route");
                }
                use_row_chunk_grid = false;
                break;
            default:
                throw std::runtime_error(
                    "Generated CPU NativeVNNI prefill policy is outside the runtime ABI");
            }
        }
        const int total_n_blocks =
            (N_chunks + n_block_chunks - 1) / n_block_chunks;
        const VerifierRowsPolicy requested_kpart_policy =
            use_decode_equivalent_kpart
                ? (use_generated_policy
                       ? (static_cast<uint8_t>(generated_policy) == 7
                              ? VerifierRowsPolicy::WideRows
                              : VerifierRowsPolicy::Pairwise)
                       : verifier_policy_override)
                : VerifierRowsPolicy::Pairwise;
        const bool effective_wide_kpart =
            use_decode_equivalent_kpart && use_avx512 && M >= 3 &&
            requested_kpart_policy == VerifierRowsPolicy::WideRows;
        const int effective_k_tiles = use_decode_equivalent_kpart
                                          ? serial_cfg.k_tiles
                                          : num_k_tiles;
        const int effective_k_tile_blocks = use_decode_equivalent_kpart
                                                ? (K_blocks + serial_cfg.k_tiles - 1) /
                                                      serial_cfg.k_tiles
                                                : k_tile_blocks;
        const int parallel_tasks = use_decode_equivalent_kpart
                                       ? M * N_chunks * effective_k_tiles
                                   : use_row_chunk_grid
                                       ? M * N_chunks
                                   : use_two_row_pair_grid
                                       ? ((M + 1) / 2) * total_n_blocks
                                       : total_n_blocks;

        /*
         * Ordinary prefill and the dedicated verifier-row kernel are distinct
         * policy surfaces. Publish the exact production M>1 route here so an
         * all-format trainer can prove that a requested N-task granularity and
         * full-K arithmetic schedule really reached the launcher. Without this
         * record, an override normalized by the small-N branch could be
         * mislabeled as an independently measured candidate.
         */
        if (PerfStatsCollector::isEnabled())
        {
            const char *effective_isa =
                use_avx512 ? "AVX512" : (use_avx2 ? "AVX2" : "Scalar");
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_native_vnni_prefill_gemm_launch",
                1.0,
                "gemm",
                {},
                PerfStatsCollector::Tags{
                    {"m", std::to_string(M)},
                    {"n", std::to_string(N)},
                    {"k", std::to_string(packed.K)},
                    {"codebook", std::to_string(packed.codebook_id)},
                    {"build_isa", compiledNativeVNNIBuildISAName()},
                    {"isa", effective_isa},
                    {"route", use_decode_equivalent_kpart
                                  ? "decode_equivalent_kpart_rows"
                                  : (use_row_chunk_grid
                                         ? "row_chunk_grid"
                                     : use_two_row_pair_grid
                                         ? "two_row_pair_grid"
                                         : "two_row_n_major")},
                    {"requested_policy",
                     requested_kpart_policy == VerifierRowsPolicy::WideRows
                         ? "WideRows"
                         : "Pairwise"},
                    {"effective_policy",
                     effective_wide_kpart ? "WideRows" : "Pairwise"},
                    {"n_block_chunks",
                     std::to_string(use_decode_equivalent_kpart
                                        ? serial_cfg.n_block_chunks
                                        : n_block_chunks)},
                    {"k_tile_blocks", std::to_string(effective_k_tile_blocks)},
                    {"k_tiles", std::to_string(effective_k_tiles)},
                    {"parallel_tasks", std::to_string(parallel_tasks)},
                    {"threads", std::to_string(num_threads)}});
        }

        if (use_decode_equivalent_kpart)
        {
            gemm_native_vnni_preq_decode_equivalent_rows(
                packed,
                A_q8_all,
                C,
                M,
                ldc,
                isa_path,
                requested_kpart_policy,
                /*publish_verifier_route=*/false);
            return;
        }

        /*
         * The grouped verifier helper owns the economical full-K Cartesian
         * row-pair grid. Calling it with an explicit Pairwise arithmetic
         * policy bypasses verifier-policy lookup while retaining the selected
         * physical row width. Every task writes a disjoint row tile and N
         * block, so scheduling cannot alter an output row's K accumulation.
         */
        if (use_two_row_pair_grid)
        {
            gemm_native_vnni_preq_decode_equivalent_rows(
                packed,
                A_q8_all,
                C,
                M,
                ldc,
                isa_path,
                VerifierRowsPolicy::Pairwise,
                /*publish_verifier_route=*/false,
                n_block_chunks);
            return;
        }

        // Initialize decode LUTs for selected ISA
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i decode_lut_512 = _mm512_setzero_si512();
        if (use_avx512 && packed.is_nibble_lut)
            decode_lut_512 = build_decode_lut(packed.codebook_id);
#endif
        __m256i decode_lut_256 = _mm256_setzero_si256();
        if (!use_avx512 && packed.is_nibble_lut)
            decode_lut_256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);

        // Small-N dispatch: M×N 2D parallel grid
        // Also forced when ldc < 64: the 2-row tiled microkernel always stores
        // 64 floats per row via _mm512_storeu_ps, which overflows into adjacent
        // rows when the output stride is narrower than one VNNI chunk.
        if (use_row_chunk_grid)
        {
            auto do_compute = [&]()
            {
                int total_tasks = N_chunks * M;
#pragma omp for schedule(static)
                for (int task = 0; task < total_tasks; ++task)
                {
                    int chunk = task / M;
                    int m = task % M;
                    const Q8_1Block *aq = A_q8_all + static_cast<size_t>(m) * K_blocks;
                    float *c_out = C + m * ldc + chunk * 64;

                    int n_cols_actual = std::min(64, N - chunk * 64);
                    if (n_cols_actual < 64)
                    {
                        alignas(64) float tmp[64];
                        if (use_avx512)
                        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (packed.is_nibble_lut)
                                gemv_native_vnni_avx512_chunk_native(packed, aq, tmp, chunk, 0, K_blocks, decode_lut_512);
                            else
                                gemv_native_vnni_avx512_chunk_int8(packed, aq, tmp, chunk, 0, K_blocks);
#endif
                        }
                        else
                        {
                            if (packed.is_nibble_lut)
                                gemv_avx2_chunk_native(packed, aq, tmp, chunk, 0, K_blocks, decode_lut_256);
                            else
                                gemv_avx2_chunk_int8(packed, aq, tmp, chunk, 0, K_blocks);
                        }
                        std::memcpy(c_out, tmp, n_cols_actual * sizeof(float));
                    }
                    else
                    {
                        if (use_avx512)
                        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (packed.is_nibble_lut)
                                gemv_native_vnni_avx512_chunk_native(packed, aq, c_out, chunk, 0, K_blocks, decode_lut_512);
                            else
                                gemv_native_vnni_avx512_chunk_int8(packed, aq, c_out, chunk, 0, K_blocks);
#endif
                        }
                        else
                        {
                            if (packed.is_nibble_lut)
                                gemv_avx2_chunk_native(packed, aq, c_out, chunk, 0, K_blocks, decode_lut_256);
                            else
                                gemv_avx2_chunk_int8(packed, aq, c_out, chunk, 0, K_blocks);
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_compute);
            return;
        }

        // N-major GEMM path: each N block owns all two-row tiles.
        auto do_compute = [&]()
        {
#pragma omp for schedule(static)
            for (int block_idx = 0; block_idx < total_n_blocks; ++block_idx)
            {
                int chunk_start = block_idx * n_block_chunks;
                int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                for (int kt = 0; kt < num_k_tiles; ++kt)
                {
                    int kb_start = kt * k_tile_blocks;
                    int kb_end = std::min(kb_start + k_tile_blocks, K_blocks);
                    bool accum = (kt > 0);

                    int m = 0;
                    for (; m + 1 < M; m += 2)
                    {
                        const Q8_1Block *aq0 = A_q8_all + static_cast<size_t>(m) * K_blocks;
                        const Q8_1Block *aq1 = A_q8_all + static_cast<size_t>(m + 1) * K_blocks;

                        for (int ci = 0; ci < chunk_count; ++ci)
                        {
                            int chunk = chunk_start + ci;
                            int n_start = chunk * 64;
                            float *c0 = C + m * ldc + n_start;
                            float *c1 = C + (m + 1) * ldc + n_start;

                            /*
                             * The physical microkernel always publishes one
                             * complete 64-column chunk. A logical tensor row,
                             * however, is allowed to end partway through that
                             * chunk. Publishing the last chunk directly would
                             * overwrite the beginning of the following row and
                             * would run past the allocation for the final row.
                             *
                             * Keep the weight-sharing two-row microkernel for
                             * the tail, but point it at stack-owned full chunks
                             * and copy only the logical columns back. For a
                             * K-tiled call, seed the valid prefix with the
                             * previously accumulated output; zero padding is
                             * private scratch and never becomes tensor state.
                             */
                            const int n_cols_actual =
                                std::min(64, N - n_start);
                            const auto compute_two_rows =
                                [&](float *output0, float *output1)
                            {
                                if (use_avx512)
                                {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                    if (packed.is_nibble_lut)
                                        gemm_2row_native_chunk(
                                            packed, aq0, aq1, output0, output1,
                                            chunk, kb_start, kb_end,
                                            decode_lut_512, accum);
                                    else
                                        gemm_2row_int8_chunk(
                                            packed, aq0, aq1, output0, output1,
                                            chunk, kb_start, kb_end, accum);
#endif
                                }
                                else
                                {
                                    if (packed.is_nibble_lut)
                                        gemm_2row_native_chunk_avx2(
                                            packed, aq0, aq1, output0, output1,
                                            chunk, kb_start, kb_end,
                                            decode_lut_256, accum);
                                    else
                                        gemm_2row_int8_chunk_avx2(
                                            packed, aq0, aq1, output0, output1,
                                            chunk, kb_start, kb_end,
                                            accum);
                                }
                            };

                            if (n_cols_actual == 64)
                            {
                                compute_two_rows(c0, c1);
                            }
                            else
                            {
                                /*
                                 * Keep scratch construction out of the common
                                 * full-chunk path. This route is selected for
                                 * throughput, so complete chunks must not pay
                                 * for 128 unnecessary stores per K tile.
                                 */
                                alignas(64) float tail0[64] = {};
                                alignas(64) float tail1[64] = {};
                                if (accum)
                                {
                                    std::copy_n(c0, n_cols_actual, tail0);
                                    std::copy_n(c1, n_cols_actual, tail1);
                                }
                                compute_two_rows(tail0, tail1);
                                std::copy_n(tail0, n_cols_actual, c0);
                                std::copy_n(tail1, n_cols_actual, c1);
                            }
                        }
                    }

                    // Odd M tail: single row
                    if (m < M)
                    {
                        const Q8_1Block *aq = A_q8_all + static_cast<size_t>(m) * K_blocks;
                        for (int ci = 0; ci < chunk_count; ++ci)
                        {
                            int chunk = chunk_start + ci;
                            int n_start = chunk * 64;
                            float *c_row = C + m * ldc + n_start;

                            /*
                             * An odd final M row uses the serial-shaped chunk
                             * kernel, which also writes 64 columns. Give a
                             * partial N tail the same bounded publication
                             * contract as the paired-row path above.
                             */
                            const int n_cols_actual =
                                std::min(64, N - n_start);
                            const auto compute_one_row = [&](float *output)
                            {
                                if (use_avx512)
                                {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                    if (accum)
                                    {
                                        alignas(64) float tmp[64] = {};
                                        if (packed.is_nibble_lut)
                                            gemv_native_vnni_avx512_chunk_native(packed, aq, tmp, chunk, kb_start, kb_end, decode_lut_512);
                                        else
                                            gemv_native_vnni_avx512_chunk_int8(packed, aq, tmp, chunk, kb_start, kb_end);
                                        __m512 s0 = _mm512_add_ps(_mm512_loadu_ps(output), _mm512_load_ps(tmp));
                                        __m512 s1 = _mm512_add_ps(_mm512_loadu_ps(output + 16), _mm512_load_ps(tmp + 16));
                                        __m512 s2 = _mm512_add_ps(_mm512_loadu_ps(output + 32), _mm512_load_ps(tmp + 32));
                                        __m512 s3 = _mm512_add_ps(_mm512_loadu_ps(output + 48), _mm512_load_ps(tmp + 48));
                                        _mm512_storeu_ps(output, s0);
                                        _mm512_storeu_ps(output + 16, s1);
                                        _mm512_storeu_ps(output + 32, s2);
                                        _mm512_storeu_ps(output + 48, s3);
                                    }
                                    else
                                    {
                                        if (packed.is_nibble_lut)
                                            gemv_native_vnni_avx512_chunk_native(packed, aq, output, chunk, kb_start, kb_end, decode_lut_512);
                                        else
                                            gemv_native_vnni_avx512_chunk_int8(packed, aq, output, chunk, kb_start, kb_end);
                                    }
#endif
                                }
                                else
                                {
                                    if (accum)
                                    {
                                        alignas(64) float tmp[64] = {};
                                        if (packed.is_nibble_lut)
                                            gemv_avx2_chunk_native(packed, aq, tmp, chunk, kb_start, kb_end, decode_lut_256);
                                        else
                                            gemv_avx2_chunk_int8(packed, aq, tmp, chunk, kb_start, kb_end);
                                        for (int i = 0; i < 64; i += 8)
                                        {
                                            __m256 s = _mm256_add_ps(
                                                _mm256_loadu_ps(output + i),
                                                _mm256_load_ps(tmp + i));
                                            _mm256_storeu_ps(output + i, s);
                                        }
                                    }
                                    else
                                    {
                                        if (packed.is_nibble_lut)
                                            gemv_avx2_chunk_native(packed, aq, output, chunk, kb_start, kb_end, decode_lut_256);
                                        else
                                            gemv_avx2_chunk_int8(packed, aq, output, chunk, kb_start, kb_end);
                                    }
                                }
                            };

                            if (n_cols_actual == 64)
                            {
                                compute_one_row(c_row);
                            }
                            else
                            {
                                alignas(64) float tail[64] = {};
                                if (accum)
                                    std::copy_n(c_row, n_cols_actual, tail);
                                compute_one_row(tail);
                                std::copy_n(tail, n_cols_actual, c_row);
                            }
                        }
                    }
                }
            }
        };

        OMP_WORKSHARE_REGION(do_compute);
    }

    /**
     * @brief Grouped runtime-M verifier GEMM with M=1 decode-equivalent row math.
     *
     * Grouped rows must preserve the production M1 reduction tree whenever
     * their results can feed speculative or recurrent state. Ordinary full-K
     * prefill uses its economical multi-row kernels directly; when serial M1
     * selects K partitioning, the ordinary dispatcher enters this same helper.
     * Work remains grouped across `(row, N-block[, K-tile])` tasks while every
     * row preserves the exact GEMV chunk and reduction order used by
     * gemv_native_vnni_preq().
     *
     * @param full_k_n_block_chunks_override Exact N-chunk granularity for an
     *        explicitly selected full-K pair-grid prefill candidate. Zero
     *        retains the serial-decode-derived verifier geometry. A nonzero
     *        value is invalid for K-partitioned decode-equivalent execution.
     */
    inline void gemm_native_vnni_preq_decode_equivalent_rows(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8_all,
        float *C,
        int M,
        int ldc,
        ISAPath isa_path = ISAPath::AUTO,
        VerifierRowsPolicy verifier_policy_override = VerifierRowsPolicy::Auto,
        bool publish_verifier_route = true,
        int full_k_n_block_chunks_override)
    {
        const int N = packed.N;
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;

        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);
        const ISALevel effective_isa =
            use_avx512
                ? ISALevel::AVX512
                : (use_avx2 ? ISALevel::AVX2 : ISALevel::Scalar);
        const int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(
            N, K, 1, packed.payload_bytes, num_threads);
        const VerifierRowsPolicy selected_policy =
            verifier_policy_override == VerifierRowsPolicy::Auto
                ? selectVerifierRowsPolicy(
                      packed,
                      M,
                      N,
                      K,
                      effective_isa,
                      num_threads,
                      cfg.k_tiles)
                : verifier_policy_override;
        VerifierRowsPolicy verifier_policy = normalizeVerifierRowsPolicy(
            selected_policy, N_chunks, use_avx512, M);
        if (verifierRowsPolicyRequiresFullK(verifier_policy) &&
            cfg.k_tiles > 1)
        {
            if (verifier_policy_override == VerifierRowsPolicy::Auto)
            {
                throw std::runtime_error(
                    "Generated CPU grouped verifier policy selected a full-K "
                    "schedule for a serial M=1 K-partition domain");
            }
            /*
             * Explicit trainer requests must remain observable as unsupported
             * evidence. Execute the established Pairwise K-partition route and
             * publish its physical identity; route mismatch then prevents the
             * requested full-K label from entering timing or policy fitting.
             */
            verifier_policy = VerifierRowsPolicy::Pairwise;
        }
        const bool use_wide_rows =
            verifier_policy == VerifierRowsPolicy::WideRows;
        if (full_k_n_block_chunks_override < 0)
        {
            throw std::invalid_argument(
                "CPU NativeVNNI full-K N-block override must be non-negative");
        }
        if (full_k_n_block_chunks_override > 0 && cfg.k_tiles > 1)
        {
            throw std::invalid_argument(
                "CPU NativeVNNI pair-grid N-block override cannot replace "
                "the serial decode K-partition geometry");
        }
        const int policy_n_block_chunks = verifierRowsPolicyNBlockChunks(
            verifier_policy, cfg.n_block_chunks);
        const int effective_n_block_chunks =
            full_k_n_block_chunks_override > 0
                ? full_k_n_block_chunks_override
                : policy_n_block_chunks;

        /*
         * Route identity is part of the grouped verifier contract.  WideRows
         * has a distinct implementation only for AVX512 M=3/4; M=2 and all
         * AVX2/scalar requests intentionally normalize to Pairwise.  Publishing
         * both requested and effective policy prevents trainers and production
         * diagnostics from mistaking a normalized launch for a measured wide
         * candidate.
         */
        if (publish_verifier_route && PerfStatsCollector::isEnabled())
        {
            const char *requested_policy = verifierRowsPolicyName(
                verifier_policy_override);
            const char *effective_policy = verifierRowsPolicyName(
                verifier_policy);
            const char *effective_isa =
                use_avx512 ? "AVX512" : (use_avx2 ? "AVX2" : "Scalar");
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_native_vnni_verifier_rows_launch",
                1.0,
                "gemm",
                {},
                PerfStatsCollector::Tags{
                    {"m", std::to_string(M)},
                    {"n", std::to_string(N)},
                    {"k", std::to_string(K)},
                    {"codebook", std::to_string(packed.codebook_id)},
                    {"build_isa", compiledNativeVNNIBuildISAName()},
                    {"requested_policy", requested_policy},
                    {"effective_policy", effective_policy},
                    {"isa", effective_isa},
                    {"k_tiles", std::to_string(cfg.k_tiles)},
                    {"n_block_chunks", std::to_string(effective_n_block_chunks)},
                    {"threads", std::to_string(num_threads)}});
        }

        if (full_k_n_block_chunks_override > 0 &&
            PerfStatsCollector::isEnabled())
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_native_vnni_pair_grid_execution",
                1.0,
                "gemm",
                {},
                PerfStatsCollector::Tags{
                    {"m", std::to_string(M)},
                    {"n", std::to_string(N)},
                    {"k", std::to_string(K)},
                    {"codebook", std::to_string(packed.codebook_id)},
                    {"n_block_chunks",
                     std::to_string(effective_n_block_chunks)},
                    {"parallel_tasks",
                     std::to_string(
                         ((M + 1) / 2) *
                         ((N_chunks + effective_n_block_chunks - 1) /
                          effective_n_block_chunks))},
                    {"threads", std::to_string(num_threads)}});
        }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i decode_lut_512 = _mm512_setzero_si512();
        if (use_avx512 && packed.is_nibble_lut)
            decode_lut_512 = build_decode_lut(packed.codebook_id);
#endif
        __m256i decode_lut_256 = _mm256_setzero_si256();
        if (use_avx2 && packed.is_nibble_lut)
            decode_lut_256 = build_decode_lut_avx2_for_codebook(packed.codebook_id);

        if (!use_avx512 && !use_avx2)
        {
            auto do_scalar_rows = [&]()
            {
#pragma omp for schedule(static)
                for (int row = 0; row < M; ++row)
                {
                    gemv_native_vnni_scalar(
                        packed,
                        A_q8_all + static_cast<size_t>(row) * K_blocks,
                        C + static_cast<size_t>(row) * ldc,
                        N,
                        K_blocks);
                }
            };
            OMP_WORKSHARE_REGION(do_scalar_rows);
            return;
        }

        if (verifier_policy == VerifierRowsPolicy::FullKRowChunkGrid ||
            verifierRowsPolicyUsesFullKNMajor(verifier_policy))
        {
            /*
             * These two schedules already have one audited implementation in
             * the ordinary prefill engine. Enter it with explicit policy and
             * width controls after the grouped route has been published. This
             * is direct grouped execution over the complete M-by-N tensor; it
             * never replays serial rows and never consults the prefill table.
             */
            gemm_native_vnni_preq(
                packed,
                A_q8_all,
                C,
                M,
                ldc,
                isa_path,
                VerifierRowsPolicy::Pairwise,
                verifier_policy == VerifierRowsPolicy::FullKRowChunkGrid
                    ? PrefillSchedulePolicy::RowChunkGrid
                    : PrefillSchedulePolicy::TwoRowNMajor,
                effective_n_block_chunks);
            return;
        }

        if (cfg.k_tiles > 1)
        {
            const int k_tiles = cfg.k_tiles;
            const int k_blocks_per_tile = (K_blocks + k_tiles - 1) / k_tiles;
            const int n_block_chunks = cfg.n_block_chunks;
            const int total_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

            const size_t partial_sum_floats =
                static_cast<size_t>(M) * static_cast<size_t>(N_chunks) *
                static_cast<size_t>(k_tiles) * 64;
            /*
             * Tile kernels write every partial slot before reduction. Reuse
             * thread-local aligned storage so tiny verifier rows do not pay
             * malloc/free or zero-fill cost on every decode step.
             */
            thread_local AlignedVector<float> partial_sums_tls;
            if (partial_sums_tls.size() < partial_sum_floats)
                partial_sums_tls.resize_uninitialized(partial_sum_floats);
            float *partial_sums = partial_sums_tls.data();

            auto do_kpar_partial_rows = [&]()
            {
                /*
                 * Preserve the exact serial decode K-parallel contract:
                 *
                 *   1. compute one independent partial for every
                 *      [row, N-chunk, K-tile]
                 *   2. reduce tile0 + tile1 + ... in increasing tile order
                 *
                 * A task shares the decoded B tile across two AVX2 rows or up
                 * to four AVX512 rows. The shared microkernels still write one
                 * independent partial per row, so increasing runtime M changes
                 * scheduling and weight reuse but never FP32 parenthesization.
                 */
                const int row_tile_width =
                    use_avx512 && use_wide_rows ? 4 : 2;
                const int row_tile_count =
                    (M + row_tile_width - 1) / row_tile_width;
                const int total_shared_tile_tasks =
                    row_tile_count * N_chunks * k_tiles;
#pragma omp for schedule(static)
                for (int task = 0; task < total_shared_tile_tasks; ++task)
                {
                    const int kt = task % k_tiles;
                    const int chunk_and_row_tile = task / k_tiles;
                    const int chunk = chunk_and_row_tile % N_chunks;
                    const int row_tile = chunk_and_row_tile / N_chunks;
                    const int row0 = row_tile * row_tile_width;
                    const int tile_rows = std::min(row_tile_width, M - row0);
                    const int kb_start = kt * k_blocks_per_tile;
                    const int kb_end =
                        std::min(kb_start + k_blocks_per_tile, K_blocks);
                    auto partial_for_row = [&](int row)
                    {
                        return partial_sums +
                               (((static_cast<size_t>(row) * N_chunks + chunk) *
                                 k_tiles) +
                                kt) *
                                   64;
                    };

                    const Q8_1Block *row0_q8 =
                        A_q8_all + static_cast<size_t>(row0) * K_blocks;
                    float *dst0 = partial_for_row(row0);

                    if (tile_rows >= 2)
                    {
                        const Q8_1Block *row1_q8 = row0_q8 + K_blocks;
                        float *dst1 = partial_for_row(row0 + 1);

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        if (use_avx512 && tile_rows == 4)
                        {
                            const Q8_1Block *row2_q8 = row1_q8 + K_blocks;
                            const Q8_1Block *row3_q8 = row2_q8 + K_blocks;
                            float *dst2 = partial_for_row(row0 + 2);
                            float *dst3 = partial_for_row(row0 + 3);
                            if (packed.is_nibble_lut)
                                gemm_4row_native_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8,
                                    row3_q8, dst0, dst1, dst2, dst3, chunk,
                                    kb_start, kb_end, decode_lut_512,
                                    /*accumulate=*/false);
                            else
                                gemm_4row_int8_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8,
                                    row3_q8, dst0, dst1, dst2, dst3, chunk,
                                    kb_start, kb_end, /*accumulate=*/false);
                            continue;
                        }
                        if (use_avx512 && tile_rows == 3)
                        {
                            const Q8_1Block *row2_q8 = row1_q8 + K_blocks;
                            float *dst2 = partial_for_row(row0 + 2);
                            if (packed.is_nibble_lut)
                                gemm_3row_native_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8, dst0,
                                    dst1, dst2, chunk, kb_start, kb_end,
                                    decode_lut_512, /*accumulate=*/false);
                            else
                                gemm_3row_int8_2z_chunk(
                                    packed, row0_q8, row1_q8, row2_q8, dst0,
                                    dst1, dst2, chunk, kb_start, kb_end,
                                    /*accumulate=*/false);
                            continue;
                        }
                        if (use_avx512)
                        {
                            if (packed.is_nibble_lut)
                                gemm_2row_native_chunk(
                                    packed, row0_q8, row1_q8, dst0, dst1,
                                    chunk, kb_start, kb_end, decode_lut_512,
                                    /*accumulate=*/false);
                            else
                                gemm_2row_int8_chunk(
                                    packed, row0_q8, row1_q8, dst0, dst1,
                                    chunk, kb_start, kb_end,
                                    /*accumulate=*/false);
                            continue;
                        }
#endif
                        if (packed.is_nibble_lut)
                            gemm_2row_native_chunk_avx2(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, kb_start, kb_end, decode_lut_256,
                                /*accumulate=*/false);
                        else
                            gemm_2row_int8_chunk_avx2(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, kb_start, kb_end,
                                /*accumulate=*/false);
                        continue;
                    }

                    if (use_avx512)
                    {
#if defined(__AVX512F__)
                        if (packed.is_nibble_lut)
                            gemv_native_vnni_avx512_chunk_native(
                                packed, row0_q8, dst0, chunk, kb_start, kb_end,
                                decode_lut_512);
                        else
                            gemv_native_vnni_avx512_chunk_int8(
                                packed, row0_q8, dst0, chunk, kb_start, kb_end);
#endif
                    }
                    else if (packed.is_nibble_lut)
                    {
                        gemv_avx2_chunk_native(
                            packed, row0_q8, dst0, chunk, kb_start, kb_end,
                            decode_lut_256);
                    }
                    else
                    {
                        gemv_avx2_chunk_int8(
                            packed, row0_q8, dst0, chunk, kb_start, kb_end);
                    }
                }

#pragma omp for schedule(static)
                for (int task = 0; task < M * N_chunks; ++task)
                {
                    const int chunk = task % N_chunks;
                    const int row = task / N_chunks;
                    const int n_start = chunk * 64;
                    const int n_cols = std::min(64, N - n_start);
                    const float *base =
                        partial_sums +
                        (static_cast<size_t>(row) * N_chunks + chunk) *
                            k_tiles * 64;
                    float *dst = C + static_cast<size_t>(row) * ldc + n_start;
                    reduceNativeVNNIKTilePartialsExact(
                        base, dst, n_cols, k_tiles, use_avx512);
                }
            };
            OMP_WORKSHARE_REGION(do_kpar_partial_rows);
            return;
        }

        const int n_block_chunks = effective_n_block_chunks;
        const int total_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

        if (use_avx512 && M >= 3 && use_wide_rows)
        {
            /*
             * Compose arbitrary runtime M from bounded four-row physical tiles.
             * AVX512 has native three/four-row kernels. Every tail still owns
             * disjoint output rows and increasing-K arithmetic.
             */
            const int row_tile_count = (M + 3) / 4;
            const int total_four_row_tasks = row_tile_count * total_blocks;
            auto do_four_rows = [&]()
            {
#pragma omp for schedule(static)
                for (int task = 0; task < total_four_row_tasks; ++task)
                {
                    const int block_idx = task / row_tile_count;
                    const int row_tile = task % row_tile_count;
                    const int first_row = row_tile * 4;
                    const int tile_rows = std::min(4, M - first_row);
                    const int chunk_start = block_idx * n_block_chunks;
                    const int chunk_count =
                        std::min(n_block_chunks, N_chunks - chunk_start);

                    const Q8_1Block *row_q8[4] = {};
                    float *row_output[4] = {};
                    for (int row = 0; row < tile_rows; ++row)
                    {
                        row_q8[row] =
                            A_q8_all +
                            static_cast<size_t>(first_row + row) * K_blocks;
                        row_output[row] =
                            C + static_cast<size_t>(first_row + row) * ldc;
                    }

                    for (int ci = 0; ci < chunk_count; ++ci)
                    {
                        const int chunk = chunk_start + ci;
                        const int n_start = chunk * 64;
                        const int n_cols = std::min(64, N - n_start);
                        float *compact_output[4] = {};
                        float *kernel_output[4] = {};
                        alignas(64) float tail_output[4][64];
                        for (int row = 0; row < tile_rows; ++row)
                        {
                            compact_output[row] = row_output[row] + n_start;
                            kernel_output[row] =
                                n_cols == 64 ? compact_output[row]
                                             : tail_output[row];
                        }

                        if (use_avx512)
                        {
                            /*
                             * Keep all AVX512-only symbols inside the compile
                             * guard so the AVX2-only trainer builds the same
                             * runtime launcher without dead declarations.
                             */
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (tile_rows == 4)
                            {
                                if (packed.is_nibble_lut)
                                    gemm_4row_native_2z_chunk(
                                        packed, row_q8[0], row_q8[1], row_q8[2],
                                        row_q8[3], kernel_output[0], kernel_output[1],
                                        kernel_output[2], kernel_output[3], chunk, 0,
                                        K_blocks, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_4row_int8_2z_chunk(
                                        packed, row_q8[0], row_q8[1], row_q8[2],
                                        row_q8[3], kernel_output[0], kernel_output[1],
                                        kernel_output[2], kernel_output[3], chunk, 0,
                                        K_blocks, /*accumulate=*/false);
                            }
                            else if (tile_rows == 3)
                            {
                                if (packed.is_nibble_lut)
                                    gemm_3row_native_2z_chunk(
                                        packed, row_q8[0], row_q8[1], row_q8[2],
                                        kernel_output[0], kernel_output[1],
                                        kernel_output[2], chunk, 0, K_blocks,
                                        decode_lut_512, /*accumulate=*/false);
                                else
                                    gemm_3row_int8_2z_chunk(
                                        packed, row_q8[0], row_q8[1], row_q8[2],
                                        kernel_output[0], kernel_output[1],
                                        kernel_output[2], chunk, 0, K_blocks,
                                        /*accumulate=*/false);
                            }
                            else if (tile_rows == 2)
                            {
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk(
                                        packed, row_q8[0], row_q8[1],
                                        kernel_output[0], kernel_output[1], chunk,
                                        0, K_blocks, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_2row_int8_chunk(
                                        packed, row_q8[0], row_q8[1],
                                        kernel_output[0], kernel_output[1], chunk,
                                        0, K_blocks, /*accumulate=*/false);
                            }
                            else if (packed.is_nibble_lut)
                            {
                                gemv_native_vnni_avx512_chunk_native(
                                    packed, row_q8[0], kernel_output[0], chunk, 0,
                                    K_blocks, decode_lut_512);
                            }
                            else
                            {
                                gemv_native_vnni_avx512_chunk_int8(
                                    packed, row_q8[0], kernel_output[0], chunk, 0,
                                    K_blocks);
                            }
#endif
                        }
                        if (n_cols < 64)
                        {
                            for (int row = 0; row < tile_rows; ++row)
                            {
                                std::memcpy(
                                    compact_output[row],
                                    tail_output[row],
                                    static_cast<size_t>(n_cols) * sizeof(float));
                            }
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_four_rows);
            return;
        }

        /*
         * Pair verifier rows for AVX2 and AVX512.
         *
         * Each pair task still visits K blocks in the exact serial decode
         * order for each row.  The only difference from row-by-row GEMV is
         * that the 2-row chunk microkernel loads and decodes the packed B
         * chunk once, then applies it to two independent row accumulators.
         * Tail chunks use temporary 64-float rows so the microkernel never
         * writes past a compact verifier output buffer.
         */
        const int row_pairs = (M + 1) / 2;
        const int total_tasks = row_pairs * total_blocks;

        auto process_pair_task = [&](int task)
        {
            /*
             * Keep block as the outer scheduling dimension so odd-M cases
             * such as M=3 interleave pair work and tail-row work across the
             * OpenMP team.  A pair task is heavier than a one-row tail task;
             * grouping all pairs first leaves half the team waiting at the
             * barrier on realistic verifier shapes.
             */
            const int block_idx = task / row_pairs;
            const int pair = task % row_pairs;
            const int row0 = pair * 2;
            const int row1 = row0 + 1;
            const int chunk_start = block_idx * n_block_chunks;
            const int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

            if (row1 < M)
            {
                const Q8_1Block *row0_q8 =
                    A_q8_all + static_cast<size_t>(row0) * K_blocks;
                const Q8_1Block *row1_q8 =
                    A_q8_all + static_cast<size_t>(row1) * K_blocks;
                float *row0_out = C + static_cast<size_t>(row0) * ldc;
                float *row1_out = C + static_cast<size_t>(row1) * ldc;

                for (int ci = 0; ci < chunk_count; ++ci)
                {
                    const int chunk = chunk_start + ci;
                    const int n_start = chunk * 64;
                    const int n_cols = std::min(64, N - n_start);
                    float *c0 = row0_out + n_start;
                    float *c1 = row1_out + n_start;
                    float *dst0 = c0;
                    float *dst1 = c1;
                    alignas(64) float tmp0[64];
                    alignas(64) float tmp1[64];
                    if (n_cols < 64)
                    {
                        dst0 = tmp0;
                        dst1 = tmp1;
                    }

                    if (use_avx512)
                    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        if (packed.is_nibble_lut)
                            gemm_2row_native_chunk(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks, decode_lut_512,
                                /*accumulate=*/false);
                        else
                            gemm_2row_int8_chunk(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks,
                                /*accumulate=*/false);
#endif
                    }
                    else
                    {
                        if (packed.is_nibble_lut)
                            gemm_2row_native_chunk_avx2(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks, decode_lut_256,
                                /*accumulate=*/false);
                        else
                            gemm_2row_int8_chunk_avx2(
                                packed, row0_q8, row1_q8, dst0, dst1,
                                chunk, 0, K_blocks,
                                /*accumulate=*/false);
                    }

                    if (n_cols < 64)
                    {
                        std::memcpy(c0, tmp0, static_cast<size_t>(n_cols) * sizeof(float));
                        std::memcpy(c1, tmp1, static_cast<size_t>(n_cols) * sizeof(float));
                    }
                }
                return;
            }

            const Q8_1Block *row_q8 =
                A_q8_all + static_cast<size_t>(row0) * K_blocks;
            float *row_out = C + static_cast<size_t>(row0) * ldc;
            if (use_avx512)
            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                gemv_native_vnni_avx512_block(
                    packed, row_q8, row_out,
                    chunk_start, chunk_count, K_blocks, N,
                    decode_lut_512);
#endif
            }
            else
            {
                gemv_avx2_block(
                    packed, row_q8, row_out,
                    chunk_start, chunk_count, K_blocks, N,
                    decode_lut_256);
            }
        };

        /*
         * MoE expert verifier projections are small enough that OpenMP
         * fork/join and barrier cost can dominate the M=2 compute.  Serial
         * decode already has a direct path for these shapes; keep grouped M=2
         * on the same footing by running the exact same 2-row chunk kernel
         * directly when the scheduler would create only a small number of
         * blocks.  This is still the grouped/economical kernel, not a hidden
         * fallback to serial row-by-row GEMV.
         */
        const bool direct_small_m2 =
            M == 2 && !omp_in_parallel() &&
            total_tasks <= std::max(1, num_threads * 2) &&
            K_blocks <= 64;
        if (direct_small_m2)
        {
            for (int task = 0; task < total_tasks; ++task)
                process_pair_task(task);
            return;
        }

        auto do_rows = [&]()
        {
#pragma omp for schedule(static)
            for (int task = 0; task < total_tasks; ++task)
                process_pair_task(task);
        };
        OMP_WORKSHARE_REGION(do_rows);
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
        const CPUNativeVNNIPackedWeights *packed = nullptr;
        float *output = nullptr;
        const float *bias = nullptr;
        int N = 0;
        int ldc = 0;
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
         * is selected independently by the verifier policy table.
         */
        DecodeSchedulePolicy decode_schedule = DecodeSchedulePolicy::Auto;
        /**
         * Forceable grouped schedule for M>1 trainer and regression launches.
         * Production descriptors leave this as Auto and resolve the sealed
         * grouped policy independently for each projection geometry/codebook.
         */
        VerifierRowsPolicy verifier_schedule = VerifierRowsPolicy::Auto;
    };

    /**
     * @brief Fused multi-projection runtime-M verifier rows.
     *
     * This is the projection-bundle analogue of
     * gemm_native_vnni_preq_decode_equivalent_rows(). Full-K projections use
     * two-row physical tiles, while long-K projections use two-row AVX2 or
     * four-row AVX512 tiles. Each tile shares packed-weight decode work across
     * independent row accumulators without changing any row's K traversal.
     * K-parallel partials are reduced through
     * reduceNativeVNNIKTilePartialsExact(), so grouped and serial decode add
     * tile partials in identical order.
     *
     * All descriptors execute under one OpenMP team. This avoids per-projection
     * fork/join overhead while preserving each projection's own codebook,
     * output stride, bias, and tail-column handling.
     */
    inline bool gemm_native_vnni_fused_verifier_rows_preq(
        const Q8_1Block *A_q8_all,
        const FusedVerifierRowsDesc *descs,
        int num_descs,
        int M,
        int K_blocks,
        ISAPath isa_path = ISAPath::AUTO)
    {
        if (!descs || num_descs <= 0 || M <= 0 || K_blocks <= 0)
            return false;

        const int num_threads = omp_get_max_threads();
        for (int p = 0; p < num_descs; ++p)
        {
            const auto &d = descs[p];
            if (!d.packed || !d.output || (!A_q8_all && !d.input) ||
                d.N <= 0 || d.ldc < d.N ||
                d.packed->blocks_per_row != K_blocks)
            {
                return false;
            }
            if (M > 1 &&
                d.decode_schedule != DecodeSchedulePolicy::Auto)
            {
                throw std::invalid_argument(
                    "CPU NativeVNNI grouped verifier bundles cannot override "
                    "the M=1 decode schedule");
            }
            if (M == 1 &&
                d.verifier_schedule != VerifierRowsPolicy::Auto)
            {
                throw std::invalid_argument(
                    "CPU NativeVNNI fused M=1 bundles cannot override the "
                    "grouped verifier schedule");
            }
        }

        const ISALevel active_isa = activeISALevel();
        bool use_avx512 = false;
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        use_avx512 = (isa_path == ISAPath::AUTO) ? (active_isa >= ISALevel::AVX512)
                                                 : (isa_path == ISAPath::AVX512);
#endif
        const bool use_avx2 =
            !use_avx512 &&
            ((isa_path == ISAPath::AUTO && active_isa >= ISALevel::AVX2) ||
             isa_path == ISAPath::AVX2);
        if (!use_avx512 && !use_avx2)
        {
            throw std::runtime_error(
                "CPU NativeVNNI fused decode/verifier execution requires an "
                "AVX2 or AVX512 grouped implementation");
        }

        struct FusedVerifierRowsPlan
        {
            int n_chunks = 0;
            int n_block_chunks = 1;
            int total_blocks = 0;
            int k_tiles = 0;
            int k_blocks_per_tile = 0;
            size_t partial_sums_offset = 0;
            size_t partial_sums_size = 0;
            DecodeSchedulePolicy requested_decode_schedule =
                DecodeSchedulePolicy::Auto;
            DecodeSchedulePolicy effective_decode_schedule =
                DecodeSchedulePolicy::Auto;
            VerifierRowsPolicy requested_verifier_schedule =
                VerifierRowsPolicy::Auto;
            VerifierRowsPolicy effective_verifier_schedule =
                VerifierRowsPolicy::Pairwise;
        };

        std::vector<FusedVerifierRowsPlan> plans(static_cast<size_t>(num_descs));
        size_t fused_partial_sum_floats = 0;
        for (int p = 0; p < num_descs; ++p)
        {
            const auto &d = descs[p];
            const NativeVNNITileConfig cfg =
                computeTileConfig(d.packed->N, d.packed->K, 1,
                                  d.packed->payload_bytes, num_threads);
            auto &plan = plans[static_cast<size_t>(p)];
            plan.n_chunks = (d.packed->N + 63) / 64;
            plan.n_block_chunks = std::max(1, cfg.n_block_chunks);
            plan.k_tiles = cfg.k_tiles;
            if (M == 1)
            {
                plan.requested_decode_schedule = d.decode_schedule;
                plan.effective_decode_schedule = d.decode_schedule;
                if (plan.effective_decode_schedule ==
                    DecodeSchedulePolicy::Auto)
                {
                    plan.effective_decode_schedule = selectDecodeSchedulePolicy(
                        *d.packed,
                        d.N,
                        d.packed->K,
                        use_avx512,
                        use_avx2,
                        plan.k_tiles > 1,
                        plan.k_tiles);
                }
                if (plan.effective_decode_schedule !=
                    DecodeSchedulePolicy::FrozenSerialOracle)
                {
                    plan.effective_decode_schedule =
                        normalizeDecodeSchedulePolicy(
                            plan.effective_decode_schedule,
                            plan.n_chunks);
                    plan.n_block_chunks = decodeScheduleNBlockChunks(
                        plan.effective_decode_schedule);
                }
            }
            else
            {
                plan.requested_verifier_schedule = d.verifier_schedule;
                const VerifierRowsPolicy selected =
                    d.verifier_schedule == VerifierRowsPolicy::Auto
                        ? selectVerifierRowsPolicy(
                              *d.packed,
                              M,
                              d.N,
                              d.packed->K,
                              use_avx512 ? ISALevel::AVX512 : ISALevel::AVX2,
                              num_threads,
                              plan.k_tiles)
                        : d.verifier_schedule;
                plan.effective_verifier_schedule =
                    normalizeVerifierRowsPolicy(
                        selected, plan.n_chunks, use_avx512, M);
                if (verifierRowsPolicyRequiresFullK(
                        plan.effective_verifier_schedule) &&
                    plan.k_tiles > 1)
                {
                    if (d.verifier_schedule == VerifierRowsPolicy::Auto)
                    {
                        throw std::runtime_error(
                            "Generated fused CPU grouped verifier policy selected "
                            "a full-K schedule for a K-partition domain");
                    }
                    plan.effective_verifier_schedule =
                        VerifierRowsPolicy::Pairwise;
                }
                plan.n_block_chunks = verifierRowsPolicyNBlockChunks(
                    plan.effective_verifier_schedule,
                    plan.n_block_chunks);
            }
            plan.total_blocks =
                (plan.n_chunks + plan.n_block_chunks - 1) /
                plan.n_block_chunks;
            plan.k_blocks_per_tile =
                (plan.k_tiles > 1)
                    ? (K_blocks + plan.k_tiles - 1) / plan.k_tiles
                    : K_blocks;
            if (plan.k_tiles > 1)
            {
                /*
                 * CPU-only verifier workspace.  GPU stages must use the graph
                 * workspace binding contract instead; this helper never runs on
                 * device backends.  The layout mirrors the single-projection
                 * exact verifier path: [row][N chunk][K tile][64 columns].
                 */
                plan.partial_sums_offset = fused_partial_sum_floats;
                plan.partial_sums_size =
                    static_cast<size_t>(M) *
                    static_cast<size_t>(plan.n_chunks) *
                    static_cast<size_t>(plan.k_tiles) * 64;
                fused_partial_sum_floats += plan.partial_sums_size;
            }
        }

        if (PerfStatsCollector::isEnabled())
        {
            const char *effective_isa =
                use_avx512 ? "AVX512" : (use_avx2 ? "AVX2" : "Scalar");
            for (int p = 0; p < num_descs; ++p)
            {
                const auto &plan = plans[static_cast<size_t>(p)];
                const bool grouped_k_parallel =
                    plan.k_tiles > 1;
                const bool full_k_row_chunks =
                    plan.effective_verifier_schedule ==
                    VerifierRowsPolicy::FullKRowChunkGrid;
                const bool full_k_wide_rows =
                    !grouped_k_parallel && use_avx512 && M >= 3 &&
                    plan.effective_verifier_schedule ==
                        VerifierRowsPolicy::WideRows;
                const bool kpart_wide_rows =
                    grouped_k_parallel && use_avx512 && M >= 3 &&
                    plan.effective_verifier_schedule ==
                        VerifierRowsPolicy::WideRows;
                const int physical_row_tile = M == 1
                                                  ? 1
                                              : full_k_row_chunks
                                                  ? 1
                                              : (full_k_wide_rows ||
                                                 kpart_wide_rows)
                                                  ? 4
                                                  : 2;
                const char *grouped_route = grouped_k_parallel
                    ? "grouped_k_parallel_row_tiles"
                    : full_k_row_chunks
                        ? "grouped_full_k_row_chunk_grid"
                    : verifierRowsPolicyUsesFullKNMajor(
                          plan.effective_verifier_schedule)
                        ? "grouped_full_k_two_row_n_major"
                    : full_k_wide_rows
                        ? "grouped_full_k_wide_rows"
                        : "grouped_full_k_pair_grid";
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_vnni_fused_verifier_rows_projection_launch",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(M)},
                        {"n", std::to_string(descs[p].N)},
                        {"k", std::to_string(descs[p].packed->K)},
                        {"codebook", std::to_string(descs[p].packed->codebook_id)},
                        {"projection", std::to_string(p)},
                        {"isa", effective_isa},
                        {"k_tiles", std::to_string(plan.k_tiles)},
                        {"n_block_chunks",
                         std::to_string(plan.n_block_chunks)},
                        {"requested_decode_policy",
                         decodeSchedulePolicyName(
                             plan.requested_decode_schedule)},
                        {"effective_decode_policy",
                         decodeSchedulePolicyName(
                             plan.effective_decode_schedule)},
                        {"requested_verifier_policy",
                         verifierRowsPolicyName(
                             plan.requested_verifier_schedule)},
                        {"effective_verifier_policy",
                         verifierRowsPolicyName(
                             plan.effective_verifier_schedule)},
                        {"physical_row_tile", std::to_string(physical_row_tile)},
                        {"route",
                         M == 1
                             ? (grouped_k_parallel
                                    ? "decode_k_parallel_row"
                                    : "decode_full_k_row")
                             : grouped_route}});
            }
        }
        /*
         * Every K-tile partial is overwritten before reduction.  A single
         * thread-local arena avoids repeated per-projection allocations while
         * keeping the layout explicit and easy to audit.
         */
        thread_local AlignedVector<float> fused_partial_sums_tls;
        if (fused_partial_sums_tls.size() < fused_partial_sum_floats)
            fused_partial_sums_tls.resize_uninitialized(fused_partial_sum_floats);
        /*
         * Capture the caller thread's arena before the OpenMP region.  The arena
         * itself is thread_local only to amortize allocations between calls; the
         * grouped verifier work-sharing region needs one shared scratch buffer so
         * all partial tasks and the later reduction tasks address the same layout.
         */
        float *fused_partial_sums_base = fused_partial_sums_tls.data();

        auto do_fused_rows = [&]()
        {
            for (int p = 0; p < num_descs; ++p)
            {
                const auto &d = descs[p];
                const auto &packed = *d.packed;
                const Q8_1Block *projection_input =
                    d.input ? d.input : A_q8_all;
                const int N = packed.N;
                auto &plan = plans[static_cast<size_t>(p)];
                const int N_chunks = plan.n_chunks;
                const int n_block_chunks = plan.n_block_chunks;
                const int total_blocks = plan.total_blocks;

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                const __m512i decode_lut_512 =
                    (use_avx512 && packed.is_nibble_lut)
                        ? build_decode_lut(packed.codebook_id)
                        : _mm512_setzero_si512();
#endif
                const __m256i decode_lut_256 =
                    (use_avx2 && packed.is_nibble_lut)
                        ? build_decode_lut_avx2_for_codebook(packed.codebook_id)
                        : _mm256_setzero_si256();

                if (M == 1 && plan.k_tiles > 1)
                {
                    /*
                     * The fused M=1 path must expose the same forceable
                     * N-block candidates as ordinary decode even when the
                     * frozen arithmetic contract uses K-partials. Each task
                     * owns one `(N block, K tile)` rectangle and writes the
                     * ordinary `[N chunk][K tile][64]` layout. The following
                     * reduction therefore has byte-for-byte the same inputs
                     * and increasing tile order as gemv_native_vnni_preq().
                     */
                    const int k_tiles = plan.k_tiles;
                    const int k_blocks_per_tile = plan.k_blocks_per_tile;
                    float *partial_sums =
                        fused_partial_sums_base + plan.partial_sums_offset;
                    const int total_tasks = total_blocks * k_tiles;
#pragma omp for schedule(static)
                    for (int task = 0; task < total_tasks; ++task)
                    {
                        const int block_idx = task / k_tiles;
                        const int kt = task % k_tiles;
                        const int chunk_start = block_idx * n_block_chunks;
                        const int chunk_count = std::min(
                            n_block_chunks, N_chunks - chunk_start);
                        const int kb_start = kt * k_blocks_per_tile;
                        const int kb_end = std::min(
                            kb_start + k_blocks_per_tile, K_blocks);
                        for (int offset = 0; offset < chunk_count; ++offset)
                        {
                            const int chunk = chunk_start + offset;
                            float *destination =
                                partial_sums +
                                (static_cast<size_t>(chunk) * k_tiles + kt) *
                                    64;
                            if (use_avx512)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (packed.is_nibble_lut)
                                    gemv_native_vnni_avx512_chunk_native(
                                        packed,
                                        projection_input,
                                        destination,
                                        chunk,
                                        kb_start,
                                        kb_end,
                                        decode_lut_512);
                                else
                                    gemv_native_vnni_avx512_chunk_int8(
                                        packed,
                                        projection_input,
                                        destination,
                                        chunk,
                                        kb_start,
                                        kb_end);
#endif
                            }
                            else if (packed.is_nibble_lut)
                            {
                                gemv_avx2_chunk_native(
                                    packed,
                                    projection_input,
                                    destination,
                                    chunk,
                                    kb_start,
                                    kb_end,
                                    decode_lut_256);
                            }
                            else
                            {
                                gemv_avx2_chunk_int8(
                                    packed,
                                    projection_input,
                                    destination,
                                    chunk,
                                    kb_start,
                                    kb_end);
                            }
                        }
                    }

#pragma omp for schedule(static)
                    for (int chunk = 0; chunk < N_chunks; ++chunk)
                    {
                        const int n_start = chunk * 64;
                        const int n_columns = std::min(64, N - n_start);
                        const float *base =
                            partial_sums +
                            static_cast<size_t>(chunk) * k_tiles * 64;
                        reduceNativeVNNIKTilePartialsExact(
                            base,
                            d.output + n_start,
                            n_columns,
                            k_tiles,
                            use_avx512);
                    }

                    if (d.bias)
                    {
#pragma omp for schedule(static) nowait
                        for (int column = 0; column < d.N; ++column)
                            d.output[column] += d.bias[column];
                    }
                    continue;
                }

                if (plan.k_tiles > 1 && (use_avx512 || use_avx2))
                {
                    /*
                     * Long-K fused verifier projections need K-parallel work
                     * for economy, but they must reduce partials in the exact
                     * same order as the M=1 decode GEMV.  This mirrors
                     * gemm_native_vnni_preq_decode_equivalent_rows(), only
                     * keeping all projections under this one OpenMP team.
                     */
                    const int k_tiles = plan.k_tiles;
                    const int k_blocks_per_tile = plan.k_blocks_per_tile;
                    float *partial_sums =
                        fused_partial_sums_base + plan.partial_sums_offset;

                    /*
                     * Compose a fixed physical row tile across the complete
                     * runtime M. AVX2 shares each decoded B chunk across two
                     * rows; AVX512 shares it across up to four. The final tile
                     * uses the matching 1/2/3-row kernel, so M=5..31 and larger
                     * configured capacities never degrade into independent
                     * one-row K-part tasks.
                     */
                    const int row_tile_width =
                        use_avx512 &&
                                plan.effective_verifier_schedule ==
                                    VerifierRowsPolicy::WideRows
                            ? 4
                            : 2;
                    const int row_tile_count =
                        (M + row_tile_width - 1) / row_tile_width;
                    const int total_shared_tile_tasks =
                        row_tile_count * N_chunks * k_tiles;
#pragma omp for schedule(static)
                    for (int task = 0; task < total_shared_tile_tasks; ++task)
                    {
                        const int kt = task % k_tiles;
                        const int chunk_and_row_tile = task / k_tiles;
                        const int chunk = chunk_and_row_tile % N_chunks;
                        const int row_tile = chunk_and_row_tile / N_chunks;
                        const int row0 = row_tile * row_tile_width;
                        const int tile_rows =
                            std::min(row_tile_width, M - row0);
                        const int kb_start = kt * k_blocks_per_tile;
                        const int kb_end =
                            std::min(kb_start + k_blocks_per_tile, K_blocks);
                        auto partial_for_row = [&](int row)
                        {
                            return partial_sums +
                                   (((static_cast<size_t>(row) * N_chunks + chunk) *
                                     k_tiles) +
                                    kt) *
                                       64;
                        };

                        const Q8_1Block *row0_q8 =
                            projection_input +
                            static_cast<size_t>(row0) * K_blocks;
                        float *dst0 = partial_for_row(row0);
                        if (tile_rows >= 2)
                        {
                            const Q8_1Block *row1_q8 = row0_q8 + K_blocks;
                            float *dst1 = partial_for_row(row0 + 1);

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (use_avx512 && tile_rows == 4)
                            {
                                const Q8_1Block *row2_q8 = row1_q8 + K_blocks;
                                const Q8_1Block *row3_q8 = row2_q8 + K_blocks;
                                float *dst2 = partial_for_row(row0 + 2);
                                float *dst3 = partial_for_row(row0 + 3);
                                if (packed.is_nibble_lut)
                                    gemm_4row_native_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        row3_q8, dst0, dst1, dst2, dst3,
                                        chunk, kb_start, kb_end,
                                        decode_lut_512, /*accumulate=*/false);
                                else
                                    gemm_4row_int8_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        row3_q8, dst0, dst1, dst2, dst3,
                                        chunk, kb_start, kb_end,
                                        /*accumulate=*/false);
                                continue;
                            }
                            if (use_avx512 && tile_rows == 3)
                            {
                                const Q8_1Block *row2_q8 = row1_q8 + K_blocks;
                                float *dst2 = partial_for_row(row0 + 2);
                                if (packed.is_nibble_lut)
                                    gemm_3row_native_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        dst0, dst1, dst2, chunk, kb_start,
                                        kb_end, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_3row_int8_2z_chunk(
                                        packed, row0_q8, row1_q8, row2_q8,
                                        dst0, dst1, dst2, chunk, kb_start,
                                        kb_end, /*accumulate=*/false);
                                continue;
                            }
                            if (use_avx512)
                            {
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, kb_start, kb_end,
                                        decode_lut_512, /*accumulate=*/false);
                                else
                                    gemm_2row_int8_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, kb_start, kb_end,
                                        /*accumulate=*/false);
                                continue;
                            }
#endif
                            if (packed.is_nibble_lut)
                                gemm_2row_native_chunk_avx2(
                                    packed, row0_q8, row1_q8, dst0, dst1,
                                    chunk, kb_start, kb_end, decode_lut_256,
                                    /*accumulate=*/false);
                            else
                                gemm_2row_int8_chunk_avx2(
                                    packed, row0_q8, row1_q8, dst0, dst1,
                                    chunk, kb_start, kb_end,
                                    /*accumulate=*/false);
                            continue;
                        }

                        if (use_avx512)
                        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (packed.is_nibble_lut)
                                gemv_native_vnni_avx512_chunk_native(
                                    packed, row0_q8, dst0, chunk, kb_start,
                                    kb_end, decode_lut_512);
                            else
                                gemv_native_vnni_avx512_chunk_int8(
                                    packed, row0_q8, dst0, chunk, kb_start,
                                    kb_end);
#endif
                        }
                        else if (packed.is_nibble_lut)
                        {
                            gemv_avx2_chunk_native(
                                packed, row0_q8, dst0, chunk, kb_start,
                                kb_end, decode_lut_256);
                        }
                        else
                        {
                            gemv_avx2_chunk_int8(
                                packed, row0_q8, dst0, chunk, kb_start,
                                kb_end);
                        }
                    }

#pragma omp for schedule(static)
                    for (int task = 0; task < M * N_chunks; ++task)
                    {
                        const int chunk = task % N_chunks;
                        const int row = task / N_chunks;
                        const int n_start = chunk * 64;
                        const int n_cols = std::min(64, N - n_start);
                        const float *base =
                            partial_sums +
                            (static_cast<size_t>(row) * N_chunks + chunk) *
                                k_tiles * 64;
                        float *dst =
                            d.output + static_cast<size_t>(row) * d.ldc + n_start;
                        reduceNativeVNNIKTilePartialsExact(
                            base, dst, n_cols, k_tiles, use_avx512);
                    }

                    if (d.bias)
                    {
#pragma omp for schedule(static) nowait
                        for (int row = 0; row < M; ++row)
                        {
                            float *row_out =
                                d.output + static_cast<size_t>(row) * d.ldc;
                            addNativeVNNIBiasRow(
                                row_out, d.bias, d.N, use_avx512);
                        }
                    }
                    continue;
                }

                const auto compute_full_k_one_row =
                    [&](const Q8_1Block *row_q8, float *destination, int chunk)
                {
                    if (use_avx512)
                    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        if (packed.is_nibble_lut)
                            gemv_native_vnni_avx512_chunk_native(
                                packed, row_q8, destination, chunk, 0,
                                K_blocks, decode_lut_512);
                        else
                            gemv_native_vnni_avx512_chunk_int8(
                                packed, row_q8, destination, chunk, 0,
                                K_blocks);
#endif
                    }
                    else if (packed.is_nibble_lut)
                    {
                        gemv_avx2_chunk_native(
                            packed, row_q8, destination, chunk, 0, K_blocks,
                            decode_lut_256);
                    }
                    else
                    {
                        gemv_avx2_chunk_int8(
                            packed, row_q8, destination, chunk, 0, K_blocks);
                    }
                };
                const auto compute_full_k_two_rows =
                    [&](const Q8_1Block *row0_q8,
                        const Q8_1Block *row1_q8,
                        float *destination0,
                        float *destination1,
                        int chunk)
                {
                    if (use_avx512)
                    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        if (packed.is_nibble_lut)
                            gemm_2row_native_chunk(
                                packed, row0_q8, row1_q8, destination0,
                                destination1, chunk, 0, K_blocks,
                                decode_lut_512, /*accumulate=*/false);
                        else
                            gemm_2row_int8_chunk(
                                packed, row0_q8, row1_q8, destination0,
                                destination1, chunk, 0, K_blocks,
                                /*accumulate=*/false);
#endif
                    }
                    else if (packed.is_nibble_lut)
                    {
                        gemm_2row_native_chunk_avx2(
                            packed, row0_q8, row1_q8, destination0,
                            destination1, chunk, 0, K_blocks,
                            decode_lut_256, /*accumulate=*/false);
                    }
                    else
                    {
                        gemm_2row_int8_chunk_avx2(
                            packed, row0_q8, row1_q8, destination0,
                            destination1, chunk, 0, K_blocks,
                        /*accumulate=*/false);
                    }
                };
                if (plan.effective_verifier_schedule ==
                    VerifierRowsPolicy::FullKRowChunkGrid)
                {
                    /*
                     * One task owns one `(row, 64-column chunk)`. This exposes
                     * both dimensions to the shared OpenMP team and is usually
                     * strongest on native AVX512, where a two-row task can
                     * otherwise leave too little outer parallelism.
                     */
                    const int total_tasks = M * N_chunks;
#pragma omp for schedule(static)
                    for (int task = 0; task < total_tasks; ++task)
                    {
                        const int chunk = task / M;
                        const int row = task % M;
                        const int n_start = chunk * 64;
                        const int n_columns = std::min(64, N - n_start);
                        const Q8_1Block *row_q8 =
                            projection_input +
                            static_cast<size_t>(row) * K_blocks;
                        float *compact =
                            d.output + static_cast<size_t>(row) * d.ldc + n_start;
                        alignas(64) float tail[64];
                        float *destination = n_columns == 64 ? compact : tail;
                        compute_full_k_one_row(row_q8, destination, chunk);
                        if (n_columns < 64)
                        {
                            std::memcpy(
                                compact,
                                tail,
                                static_cast<size_t>(n_columns) * sizeof(float));
                        }
                    }
                    if (d.bias)
                    {
#pragma omp for schedule(static) nowait
                        for (int row = 0; row < M; ++row)
                        {
                            addNativeVNNIBiasRow(
                                d.output + static_cast<size_t>(row) * d.ldc,
                                d.bias,
                                d.N,
                                use_avx512);
                        }
                    }
                    continue;
                }

                if (verifierRowsPolicyUsesFullKNMajor(
                        plan.effective_verifier_schedule))
                {
                    /*
                     * One task owns an N block and visits every verifier row.
                     * Adjacent rows share packed-weight decode through the
                     * two-row microkernel; an odd tail uses the exact serial
                     * chunk kernel. No task shares an output address.
                     */
#pragma omp for schedule(static)
                    for (int block_idx = 0;
                         block_idx < total_blocks;
                         ++block_idx)
                    {
                        const int chunk_start = block_idx * n_block_chunks;
                        const int chunk_count = std::min(
                            n_block_chunks, N_chunks - chunk_start);
                        int row = 0;
                        for (; row + 1 < M; row += 2)
                        {
                            const Q8_1Block *row0_q8 =
                                projection_input +
                                static_cast<size_t>(row) * K_blocks;
                            const Q8_1Block *row1_q8 = row0_q8 + K_blocks;
                            for (int offset = 0;
                                 offset < chunk_count;
                                 ++offset)
                            {
                                const int chunk = chunk_start + offset;
                                const int n_start = chunk * 64;
                                const int n_columns =
                                    std::min(64, N - n_start);
                                float *compact0 =
                                    d.output + static_cast<size_t>(row) * d.ldc +
                                    n_start;
                                float *compact1 =
                                    d.output +
                                    static_cast<size_t>(row + 1) * d.ldc +
                                    n_start;
                                alignas(64) float tail0[64];
                                alignas(64) float tail1[64];
                                float *destination0 =
                                    n_columns == 64 ? compact0 : tail0;
                                float *destination1 =
                                    n_columns == 64 ? compact1 : tail1;
                                compute_full_k_two_rows(
                                    row0_q8,
                                    row1_q8,
                                    destination0,
                                    destination1,
                                    chunk);
                                if (n_columns < 64)
                                {
                                    const size_t bytes =
                                        static_cast<size_t>(n_columns) *
                                        sizeof(float);
                                    std::memcpy(compact0, tail0, bytes);
                                    std::memcpy(compact1, tail1, bytes);
                                }
                            }
                        }
                        if (row < M)
                        {
                            const Q8_1Block *row_q8 =
                                projection_input +
                                static_cast<size_t>(row) * K_blocks;
                            for (int offset = 0;
                                 offset < chunk_count;
                                 ++offset)
                            {
                                const int chunk = chunk_start + offset;
                                const int n_start = chunk * 64;
                                const int n_columns =
                                    std::min(64, N - n_start);
                                float *compact =
                                    d.output + static_cast<size_t>(row) * d.ldc +
                                    n_start;
                                alignas(64) float tail[64];
                                float *destination =
                                    n_columns == 64 ? compact : tail;
                                compute_full_k_one_row(
                                    row_q8, destination, chunk);
                                if (n_columns < 64)
                                {
                                    std::memcpy(
                                        compact,
                                        tail,
                                        static_cast<size_t>(n_columns) *
                                            sizeof(float));
                                }
                            }
                        }
                    }
                    if (d.bias)
                    {
#pragma omp for schedule(static) nowait
                        for (int row = 0; row < M; ++row)
                        {
                            addNativeVNNIBiasRow(
                                d.output + static_cast<size_t>(row) * d.ldc,
                                d.bias,
                                d.N,
                                use_avx512);
                        }
                    }
                    continue;
                }

                if (use_avx512 && M >= 3 &&
                    plan.effective_verifier_schedule ==
                        VerifierRowsPolicy::WideRows)
                {
                    /*
                     * WideRows owns bounded AVX512 four-row physical tiles. A
                     * partial tile uses the matching smaller AVX512 kernel.
                     */
                    const int row_tile_count = (M + 3) / 4;
                    const int total_tasks = row_tile_count * total_blocks;
#pragma omp for schedule(static)
                    for (int task = 0; task < total_tasks; ++task)
                    {
                        const int block_idx = task / row_tile_count;
                        const int row_tile = task % row_tile_count;
                        const int first_row = row_tile * 4;
                        const int tile_rows = std::min(4, M - first_row);
                        const int chunk_start = block_idx * n_block_chunks;
                        const int chunk_count = std::min(
                            n_block_chunks, N_chunks - chunk_start);
                        const Q8_1Block *row_q8[4] = {};
                        float *row_output[4] = {};
                        for (int row = 0; row < tile_rows; ++row)
                        {
                            row_q8[row] = projection_input +
                                static_cast<size_t>(first_row + row) * K_blocks;
                            row_output[row] = d.output +
                                static_cast<size_t>(first_row + row) * d.ldc;
                        }
                        for (int offset = 0;
                             offset < chunk_count;
                             ++offset)
                        {
                            const int chunk = chunk_start + offset;
                            const int n_start = chunk * 64;
                            const int n_columns = std::min(64, N - n_start);
                            float *compact[4] = {};
                            float *destination[4] = {};
                            alignas(64) float tail[4][64];
                            for (int row = 0; row < tile_rows; ++row)
                            {
                                compact[row] = row_output[row] + n_start;
                                destination[row] =
                                    n_columns == 64 ? compact[row] : tail[row];
                            }
                            if (tile_rows == 4)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (packed.is_nibble_lut)
                                    gemm_4row_native_2z_chunk(
                                        packed, row_q8[0], row_q8[1], row_q8[2],
                                        row_q8[3], destination[0], destination[1],
                                        destination[2], destination[3], chunk, 0,
                                        K_blocks, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_4row_int8_2z_chunk(
                                        packed, row_q8[0], row_q8[1], row_q8[2],
                                        row_q8[3], destination[0], destination[1],
                                        destination[2], destination[3], chunk, 0,
                                        K_blocks, /*accumulate=*/false);
#endif
                            }
                            else if (tile_rows == 3)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (packed.is_nibble_lut)
                                    gemm_3row_native_2z_chunk(
                                        packed, row_q8[0], row_q8[1],
                                        row_q8[2], destination[0],
                                        destination[1], destination[2],
                                        chunk, 0, K_blocks, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_3row_int8_2z_chunk(
                                        packed, row_q8[0], row_q8[1],
                                        row_q8[2], destination[0],
                                        destination[1], destination[2],
                                        chunk, 0, K_blocks,
                                        /*accumulate=*/false);
#endif
                            }
                            else if (tile_rows == 2)
                            {
                                compute_full_k_two_rows(
                                    row_q8[0], row_q8[1], destination[0],
                                    destination[1], chunk);
                            }
                            else
                            {
                                compute_full_k_one_row(
                                    row_q8[0], destination[0], chunk);
                            }
                            if (n_columns < 64)
                            {
                                const size_t bytes =
                                    static_cast<size_t>(n_columns) *
                                    sizeof(float);
                                for (int row = 0; row < tile_rows; ++row)
                                    std::memcpy(compact[row], tail[row], bytes);
                            }
                        }
                    }
                    if (d.bias)
                    {
#pragma omp for schedule(static) nowait
                        for (int row = 0; row < M; ++row)
                        {
                            addNativeVNNIBiasRow(
                                d.output + static_cast<size_t>(row) * d.ldc,
                                d.bias,
                                d.N,
                                use_avx512);
                        }
                    }
                    continue;
                }

                {
                    const int row_pairs = (M + 1) / 2;
                    const int total_tasks = row_pairs * total_blocks;
#pragma omp for schedule(static) nowait
                    for (int task = 0; task < total_tasks; ++task)
                    {
                    /*
                     * Interleave pair/tail-row tasks per N-block.  This keeps
                     * fused projection bundles economical for M=3, where the
                     * final verifier row is necessarily a single-row task unless
                     * a wider 3-row microkernel is selected.
                     */
                    const int block_idx = task / row_pairs;
                    const int pair = task % row_pairs;
                    const int row0 = pair * 2;
                    const int row1 = row0 + 1;
                    const int chunk_start = block_idx * n_block_chunks;
                    const int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

                    if (row1 < M)
                    {
                        const Q8_1Block *row0_q8 =
                            projection_input +
                            static_cast<size_t>(row0) * K_blocks;
                        const Q8_1Block *row1_q8 =
                            projection_input +
                            static_cast<size_t>(row1) * K_blocks;
                        float *row0_out = d.output + static_cast<size_t>(row0) * d.ldc;
                        float *row1_out = d.output + static_cast<size_t>(row1) * d.ldc;

                        for (int ci = 0; ci < chunk_count; ++ci)
                        {
                            const int chunk = chunk_start + ci;
                            const int n_start = chunk * 64;
                            const int n_cols = std::min(64, N - n_start);
                            float *c0 = row0_out + n_start;
                            float *c1 = row1_out + n_start;
                            float *dst0 = c0;
                            float *dst1 = c1;
                            alignas(64) float tmp0[64];
                            alignas(64) float tmp1[64];
                            if (n_cols < 64)
                            {
                                dst0 = tmp0;
                                dst1 = tmp1;
                            }

                            if (use_avx512)
                            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks, decode_lut_512,
                                        /*accumulate=*/false);
                                else
                                    gemm_2row_int8_chunk(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks,
                                        /*accumulate=*/false);
#endif
                            }
                            else
                            {
                                if (packed.is_nibble_lut)
                                    gemm_2row_native_chunk_avx2(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks, decode_lut_256,
                                        /*accumulate=*/false);
                                else
                                    gemm_2row_int8_chunk_avx2(
                                        packed, row0_q8, row1_q8, dst0, dst1,
                                        chunk, 0, K_blocks,
                                        /*accumulate=*/false);
                            }

                            if (n_cols < 64)
                            {
                                std::memcpy(c0, tmp0, static_cast<size_t>(n_cols) * sizeof(float));
                                std::memcpy(c1, tmp1, static_cast<size_t>(n_cols) * sizeof(float));
                            }
                        }
                        continue;
                    }

                    const Q8_1Block *row_q8 =
                        projection_input +
                        static_cast<size_t>(row0) * K_blocks;
                    float *row_out = d.output + static_cast<size_t>(row0) * d.ldc;
                    if (use_avx512)
                    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        gemv_native_vnni_avx512_block(
                            packed, row_q8, row_out,
                            chunk_start, chunk_count, K_blocks, N,
                            decode_lut_512);
#endif
                    }
                    else
                    {
                        gemv_avx2_block(
                            packed, row_q8, row_out,
                        chunk_start, chunk_count, K_blocks, N,
                        decode_lut_256);
                    }
                    }
                }

                if (d.bias)
                {
#pragma omp barrier
#pragma omp for schedule(static) nowait
                    for (int row = 0; row < M; ++row)
                    {
                        float *row_out = d.output + static_cast<size_t>(row) * d.ldc;
                        addNativeVNNIBiasRow(
                            row_out, d.bias, d.N, use_avx512);
                    }
                }
            }
#pragma omp barrier
        };

        OMP_WORKSHARE_REGION(do_fused_rows);
        return true;
    }

    // =========================================================================
    // Full GEMM dispatcher (M>1) — quantizes then delegates to preq path
    // =========================================================================

    inline void gemm_native_vnni(
        const CPUNativeVNNIPackedWeights &packed,
        const float *A_fp32,
        float *C,
        int M,
        int ldc)
    {
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;

        // Pre-quantize all M rows of A
        std::vector<Q8_1Block> all_A_q8(static_cast<size_t>(M) * K_blocks);
        quantize_activations_to_q8_1(A_fp32, all_A_q8.data(), M, K, K_blocks);

        // Delegate to pre-quantized compute path
        gemm_native_vnni_preq(packed, all_A_q8.data(), C, M, ldc);
    }

    // =========================================================================
    // Bias epilogue: C[m, j] += bias[j] for all M rows
    // =========================================================================

    inline void apply_bias_epilogue(float *C, const float *bias, int M, int N, int ldc)
    {
        for (int m = 0; m < M; ++m)
        {
            float *row = C + m * ldc;
            int j = 0;
#if defined(__AVX512F__)
            for (; j + 15 < N; j += 16)
            {
                __m512 c = _mm512_loadu_ps(row + j);
                __m512 b = _mm512_loadu_ps(bias + j);
                _mm512_storeu_ps(row + j, _mm512_add_ps(c, b));
            }
#endif
            for (; j < N; ++j)
                row[j] += bias[j];
        }
    }

    // =========================================================================
    // Fused multi-projection GEMV (single OMP region, all formats)
    // =========================================================================

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
        const CPUNativeVNNIPackedWeights *packed; // eager interleaved weights
        float *output;                            // output vector [N]
        const float *bias;                        // optional bias [N], nullptr if none
        int N;                                    // number of output rows
    };

    /**
     * @brief Fused multi-projection GEMV in a single OMP region.
     *
     * Processes all projections without re-entering the OMP parallel region.
     * Uses `nowait` between projections so threads finishing one projection
     * can immediately start the next — critical for work balancing when
     * projection sizes differ (e.g., Q=3584, K=512, V=512 with 56 threads).
     *
     * @param A_q8              Pre-quantized Q8_1 activations [K_blocks]
     * @param descs             Array of projection descriptors
     * @param num_descs         Number of projections
     * @param isa_path          Runtime ISA selection used by every projection
     * @param schedule_override Forceable M=1 ownership policy for training
     *
     * @throws std::invalid_argument if descriptors do not share one K width.
     * @throws std::runtime_error if the shared grouped engine rejects the
     *         production bundle or no vector implementation is available.
     */
    inline void gemv_native_vnni_fused_preq(
        const Q8_1Block *__restrict A_q8,
        const FusedGemvDesc *descs,
        int num_descs,
        ISAPath isa_path = ISAPath::AUTO,
        DecodeSchedulePolicy schedule_override = DecodeSchedulePolicy::Auto)
    {
        if (!A_q8 || !descs || num_descs <= 0)
            throw std::invalid_argument(
                "CPU NativeVNNI fused decode requires input and descriptors");

        const int K_blocks = descs[0].packed
                                 ? descs[0].packed->blocks_per_row
                                 : 0;
        if (K_blocks <= 0)
            throw std::invalid_argument(
                "CPU NativeVNNI fused decode has an invalid first projection");

        std::vector<FusedVerifierRowsDesc> fused_rows;
        fused_rows.reserve(static_cast<size_t>(num_descs));
        for (int projection = 0; projection < num_descs; ++projection)
        {
            const FusedGemvDesc &d = descs[projection];
            if (!d.packed || !d.output || d.N <= 0 ||
                d.packed->blocks_per_row != K_blocks)
            {
                throw std::invalid_argument(
                    "CPU NativeVNNI fused decode descriptors must have valid "
                    "outputs and one shared K width");
            }
            fused_rows.push_back(FusedVerifierRowsDesc{
                .packed = d.packed,
                .output = d.output,
                .bias = d.bias,
                .N = d.N,
                .ldc = d.N,
                .input = nullptr,
                .decode_schedule = schedule_override,
            });
        }

        if (!gemm_native_vnni_fused_verifier_rows_preq(
                A_q8,
                fused_rows.data(),
                static_cast<int>(fused_rows.size()),
                1,
                K_blocks,
                isa_path))
        {
            throw std::runtime_error(
                "CPU NativeVNNI fused decode bundle was rejected");
        }
    }

    // =========================================================================
    // Fused multi-input GEMV (single OMP region, different Q8_1 input per desc)
    // =========================================================================

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
        const CPUNativeVNNIPackedWeights *packed;  // packed weights
        float *output;                            // output vector [N]
        int N;                                    // number of output rows
    };

    /**
     * @brief Fused multi-input GEMV: multiple projections with different inputs.
     *
     * Saves OMP fork/join overhead (3×~8µs per MoE layer for 4 experts)
     * and improves load balance via nowait between projections
     * (128 total chunks vs 4×32 = better utilization with 28 threads).
     */
    inline void gemv_fused_multi_input_preq(
        const FusedGemvMultiInputDesc *descs,
        int num_descs,
        ISAPath isa_path = ISAPath::AUTO,
        DecodeSchedulePolicy schedule_override = DecodeSchedulePolicy::Auto)
    {
        if (!descs || num_descs <= 0)
            throw std::invalid_argument(
                "CPU NativeVNNI fused multi-input decode requires descriptors");

        const int K_blocks = descs[0].packed
                                 ? descs[0].packed->blocks_per_row
                                 : 0;
        if (K_blocks <= 0)
            throw std::invalid_argument(
                "CPU NativeVNNI fused multi-input decode has an invalid first "
                "projection");

        std::vector<FusedVerifierRowsDesc> fused_rows;
        fused_rows.reserve(static_cast<size_t>(num_descs));
        for (int projection = 0; projection < num_descs; ++projection)
        {
            const FusedGemvMultiInputDesc &d = descs[projection];
            if (!d.A_q8 || !d.packed || !d.output || d.N <= 0 ||
                d.packed->blocks_per_row != K_blocks)
            {
                throw std::invalid_argument(
                    "CPU NativeVNNI fused multi-input descriptors must have "
                    "valid buffers and one shared K width");
            }
            fused_rows.push_back(FusedVerifierRowsDesc{
                .packed = d.packed,
                .output = d.output,
                .bias = nullptr,
                .N = d.N,
                .ldc = d.N,
                .input = d.A_q8,
                .decode_schedule = schedule_override,
            });
        }

        if (!gemm_native_vnni_fused_verifier_rows_preq(
                nullptr,
                fused_rows.data(),
                static_cast<int>(fused_rows.size()),
                1,
                K_blocks,
                isa_path))
        {
            throw std::runtime_error(
                "CPU NativeVNNI fused multi-input decode bundle was rejected");
        }
    }

} // namespace llaminar2::cpu::native_vnni
