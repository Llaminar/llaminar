/**
 * @file CPUBF16Gemm.h
 * @brief Native AMX-BF16 and AVX512-BF16 GEMM for homogeneous BF16 weights.
 *
 * Ownership and why: the CPU BF16 homogeneous route (BF16 × BF16 → FP32) was a
 * library (oneDNN) matmul whose internal reduction order is implementation
 * defined. Sapphire Rapids-class hosts additionally carry two deterministic
 * hardware dot-product engines this file exposes as first-class routes:
 *
 *  - AMX-BF16: 16×16 tile matrix-multiply (`tdpbf16ps`), one instruction per
 *    16×16×16 tile product with FP32 accumulation.
 *  - AVX512-BF16: 32 BF16 elements per register, two products per FP32 lane
 *    (`vdpbf16ps`), FP32 accumulation.
 *
 * Both routes publish one fixed reduction contract per route: K is visited in
 * ascending chunks of the instruction's native width, the hardware lane
 * accumulators hold disjoint K partitions, and the inter-lane reduction is a
 * fixed pairwise tree. Outputs are therefore byte-identical across runs and
 * thread counts for the same route; they are deliberately NOT serial-fma
 * byte-identical (the hardware fuses pair/tile sums), so serial decode rows
 * and placement-invariant expert paths keep their exact programs:
 *
 *  - M == 1 automatic policy resolves the existing serial-exact decode route
 *    (`run_bf16_skinny_matmul` in FloatingPointGemmKernel.h).
 *  - GPUAlignedExpert BF16/FP16 experts keep the cross-tier exact reduction
 *    tree; this file never participates in those lanes.
 *
 * Route selection is explicit and fails fast: an engine that is not compiled
 * into the envelope, not admitted by the executing hardware, or not admitted
 * by the geometry raises a typed exception instead of silently substituting
 * another engine. `Automatic` resolution is the single policy authority and
 * intentionally keeps `Library` (the current production route) until the
 * native routes are certified and measured.
 *
 * @author Llaminar V2 CPU kernels
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib> // std::getenv
#include <cstring> // std::memcpy
#include <atomic>
#include <stdexcept>
#include <string>
#include <algorithm>

#include "utils/CPUFeatures.h"
#include "utils/OpenMPUtils.h"
#include "utils/Logger.h"

#if defined(__AVX512BF16__)
#include <immintrin.h>
#endif
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace gemm
    {
        namespace cpu_bf16
        {

            // =================================================================
            // Typed engine identity
            // =================================================================

            /**
             * @brief One native or library BF16 GEMM engine.
             *
             * `Automatic` selects through the process policy; it never reaches
             * a kernel. `Library` means the caller's existing oneDNN/serial
             * route owns the call. `AVX512BF16` and `AMXBF16` are the native
             * hardware dot-product engines compiled into the matching
             * `LLAMINAR_CPU_ISA` envelope.
             */
            enum class Engine : std::uint8_t
            {
                Automatic = 0,
                Library = 1,
                AVX512BF16 = 2,
                AMXBF16 = 3,
            };

            /** @brief Stable spelling for diagnostics and route reports. */
            inline constexpr const char *engineName(Engine engine) noexcept
            {
                switch (engine)
                {
                case Engine::Automatic: return "automatic";
                case Engine::Library: return "library";
                case Engine::AVX512BF16: return "avx512bf16";
                case Engine::AMXBF16: return "amxbf16";
                }
                return "unknown";
            }

            // =================================================================
            // Compiled-envelope and hardware admission
            // =================================================================

            /** @return Whether this binary carries AVX512-BF16 instructions. */
            inline constexpr bool compiledWithAVX512BF16() noexcept
            {
                return LLAMINAR_COMPILED_WITH_AVX512_BF16 != 0;
            }

            /** @return Whether this binary carries AMX tile instructions. */
            inline constexpr bool compiledWithAMXBF16() noexcept
            {
                return LLAMINAR_COMPILED_WITH_AMX_BF16 != 0;
            }

            /**
             * @brief Authenticate one requested engine against the executing host.
             * @param engine Native or library engine identity.
             * @return Whether the host can execute the engine's instructions.
             * @details Library is always executable. Native engines require the
             * compiled envelope AND the CPUID/XCR0 capabilities. This query is
             * read-only; startup admission lives in resolveCPUISALevel().
             */
            inline bool engineHardwareAdmitted(Engine engine) noexcept
            {
                switch (engine)
                {
                case Engine::Automatic:
                    return true;
                case Engine::Library:
                    return true;
                case Engine::AVX512BF16:
                    return compiledWithAVX512BF16() && cpu_supports_avx512_bf16();
                case Engine::AMXBF16:
                    return compiledWithAMXBF16() && cpu_supports_amx_bf16()
                        && cpu_amx_data_instructions_usable();
                }
                return false;
            }

            /**
             * @brief Whether one native engine admits the requested geometry.
             * @param engine Native engine identity (Automatic/Library always yes).
             * @param M Number of activation rows.
             * @param N Number of output columns.
             * @param K Reduction width.
             * @param transpose_B True for row-major transposed weights [N,K].
             * @return Whether the engine's hard geometry contract is satisfied.
             * @details AVX512-BF16 processes K in 32-element chunks and reads B
             * weight rows contiguously, so it admits `transpose_B=true` with
             * `K % 32 == 0`. AMX-BF16 admits both weight layouts through
             * strided tile loads but requires full 16×16×16 tiles. M must be
             * positive for both engines.
             */
            inline bool engineGeometryAdmits(Engine engine, int M, int N, int K,
                                             bool transpose_B) noexcept
            {
                if (M <= 0 || N <= 0 || K <= 0)
                    return false;
                switch (engine)
                {
                case Engine::Automatic:
                case Engine::Library:
                    return true;
                case Engine::AVX512BF16:
                    return transpose_B && (K % 32 == 0);
                case Engine::AMXBF16:
                    // 32-element K windows: the universal palette gives every
                    // tile 64-byte rows (32 BF16 per A/B tile row).
                    return (M % 16 == 0) && (N % 16 == 0) && (K % 32 == 0);
                }
                return false;
            }

            /**
             * @brief Resolve one engine name spelling (case-insensitive).
             * @param name Null/empty means Automatic; otherwise the stable
             *        spelling of engineName().
             * @return Parsed engine identity.
             * @throws std::invalid_argument For unknown or empty-known spellings.
             */
            inline Engine parseEngineName(const char *name)
            {
                if (!name || !*name)
                    return Engine::Automatic;
                const auto eq = [](const char *left, const char *right)
                {
                    for (; *left && *right; ++left, ++right)
                        if ((*left | 0x20) != (*right | 0x20))
                            return false;
                    return *left == *right;
                };
                if (eq(name, "library")) return Engine::Library;
                if (eq(name, "avx512bf16")) return Engine::AVX512BF16;
                if (eq(name, "amxbf16")) return Engine::AMXBF16;
                if (eq(name, "auto") || eq(name, "automatic")) return Engine::Automatic;
                throw std::invalid_argument(
                    std::string("LLAMINAR_CPU_BF16_GEMM_ENGINE names an unknown engine: ") + name);
            }

            /**
             * @brief Resolve the process BF16 GEMM engine request once.
             * @param env Null/empty for the automatic policy.
             * @return Requested engine (Automatic, Library or one native name).
             * @throws std::invalid_argument For unknown or unavailable explicit names.
             * @details This is the one getenv boundary for the BF16 route; it is
             * deliberately kept beside the kernels (the CPUFeatures ISA boundary
             * owns the same pattern for LLAMINAR_ISA_LEVEL) so tests can resolve
             * an independent request without mutating process state.
             */
            inline Engine resolveRequestedEngine(const char *env = nullptr)
            {
                // Explicit native requests must be real on this binary/host;
                // a missing capability is a hard error, never a silent route.
                const Engine requested = parseEngineName(env ? env : std::getenv("LLAMINAR_CPU_BF16_GEMM_ENGINE"));
                if (requested != Engine::Automatic && requested != Engine::Library &&
                    !engineHardwareAdmitted(requested))
                {
                    throw std::runtime_error(
                        std::string("The requested BF16 GEMM engine '") + engineName(requested) +
                        "' is not compiled into this CPU ISA envelope or the executing host does not advertise it");
                }
                return requested;
            }

            /**
             * @brief Cached process engine request (the startup-frozen policy).
             * @return Resolved request; environment is read exactly once.
             */
            inline Engine activeEngineRequest()
            {
                static const Engine request = []() { return resolveRequestedEngine(); }();
                return request;
            }

            // =================================================================
            // Route evidence (opt-in counters, the FMLAL-leaf pattern)
            // =================================================================

            /**
             * @brief Per-native-engine live dispatch counter (evidence-only).
             * @param engine Native engine; Automatic/Library return a shared no-op slot.
             * @return Atomic counter admitted by one probe-enabled contract.
             */
            inline std::atomic<std::uint64_t> &bf16NativeRouteCounter(Engine engine)
            {
                static std::atomic<std::uint64_t> avx512_counter{0};
                static std::atomic<std::uint64_t> amx_counter{0};
                switch (engine)
                {
                case Engine::AVX512BF16: return avx512_counter;
                case Engine::AMXBF16: return amx_counter;
                default: return avx512_counter; // never counted
                }
            }

            /** @brief Evidence-only: whether the route probe records dispatches. */
            inline std::atomic<bool> &bf16RouteProbeEnabled() noexcept
            {
                static std::atomic<bool> enabled{false};
                return enabled;
            }

            /**
             * @return Admitted live native BF16 GEMM dispatches for one engine.
             * @brief Read the probe evidence without enabling it.
             */
            inline std::uint64_t bf16NativeRouteInvocations(Engine engine) noexcept
            {
                return bf16NativeRouteCounter(engine).load(std::memory_order_relaxed);
            }

            // =================================================================
            // Shared rounding epilogue (the skinny-matmul contract, no bias)
            // =================================================================

            /**
             * @brief Publish one completed dot product with row-independent rounding.
             * @param dot Completed route dot product.
             * @param alpha Dot-product multiplier.
             * @param beta Previous-output multiplier; zero suppresses its read.
             * @param previous Address of the existing output element.
             * @return Exactly rounded output word.
             * @details The native BF16 engines expose no fused bias, so this is
             * the same arithmetic as gemm::completeSkinnyMatmulValue() under the
             * no-bias profile: the alpha product stays in a register across an
             * explicit compiler boundary, then the beta contribution is one
             * explicit FMA. Unit tests pin this against the skinny-matmul
             * epilogue so the two implementations cannot drift.
             */
            inline float completeEpilogueValue(float dot, float alpha, float beta,
                                               const float *previous) noexcept
            {
                float value = alpha * dot;
#if defined(__aarch64__)
                asm volatile("" : "+w"(value));
#elif defined(__GNUC__) || defined(__clang__)
                asm volatile("" : "+x"(value));
#else
                volatile float rounded = value;
                value = rounded;
#endif
                if (beta != 0.0f)
                    value = std::fma(beta, *previous, value);
                return value;
            }

            // =================================================================
            // AVX512-BF16 engine
            // =================================================================

#if defined(__AVX512BF16__)

            /**
             * @brief Load 32 BF16 elements into one BF16 vector register.
             * @param ptr Contiguous 64-byte source (alignment not required).
             * @return The loaded chunk.
             * @details GCC lowers this word copy to a single unaligned
             * vmovdqu16; a bare __m512bh dereference would emit an ALIGNED
             * vmovdqa64 (compilers assume vector-size alignment for vector
             * types) and fault on 16-byte-aligned tensor storage, so the copy
             * form with an unaligned move is the contract here.
             */
            inline __m512bh loadBf16Chunk(const void *ptr) noexcept
            {
                __m512bh value;
                std::memcpy(&value, ptr, sizeof(value));
                return value;
            }

            /**
             * @brief Fixed 16-lane pairwise reduction to one FP32 word.
             * @param v Hardware lane accumulators (lane i holds K partition i).
             * @return The fixed root of the tree
             *         ((p0+p4)+(p8+p12)) + ((p1+p5)+(p9+p13)) + ... in the
             *         exact grouping below.
             * @details Deterministic across compilers, thread counts and runs.
             * The tree is: q0=(p0+p4)+(p8+p12); q1=(p1+p5)+(p9+p13);
             * q2=(p2+p6)+(p10+p14); q3=(p3+p7)+(p11+p15); then
             * r0=q0+q2; r1=q1+q3; result=r0+r1.
             */
            inline float fixedReduceLanes16(__m512 v) noexcept
            {
                const __m128 lo = _mm512_castps512_ps128(v);
                const __m128 h1 = _mm512_extractf32x4_ps(v, 1);
                const __m128 h2 = _mm512_extractf32x4_ps(v, 2);
                const __m128 h3 = _mm512_extractf32x4_ps(v, 3);
                // Stage 1: (p0+p4)..(p3+p7) and (p8+p12)..(p11+p15).
                const __m128 q01 = _mm_add_ps(lo, h1);
                const __m128 q23 = _mm_add_ps(h2, h3);
                __m128 s = _mm_add_ps(q01, q23);
                // Stage 2: r0 = q0+q2, r1 = q1+q3.
                const __m128 swap = _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2));
                s = _mm_add_ps(s, swap);
                // Stage 3: r0+r1.
                const __m128 hi = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 1, 0, 1));
                s = _mm_add_ps(s, hi);
                return _mm_cvtss_f32(s);
            }

            /**
             * @brief One pipelined 4×4 BF16 microkernel tile over the full K.
             *
             * Hand-scheduled because compilers serialize load and FMA issue:
             * the GCC-intrinsic form reached ~0.45 vdpbf16/cycle; this
             * pipeline reaches ~1.0-1.2 with the identical arithmetic
             * contract (K ascending in 32-element chunks, hardware lane i
             * accumulating pair partition i, per-lane FP32 accumulation).
             *
             * Register contract (all 32 ZMM, ABI-saved around the body):
             *   zmm0-15  -> sixteen FP32 lane accumulators, acc(4*r+c)
             *   zmm16-23 -> double-buffer set 1 (a0-3, b0-3)
             *   zmm24-31 -> double-buffer set 2
             * The pipeline keeps one 32-element chunk in flight: the loads
             * for chunk kt+1 are issued before the 16 FMAs of chunk kt, so
             * L2 latency overlaps the FMA work. A-phase consumes set 1 and
             * loads set 2; B-phase consumes set 2 and loads set 1, so no
             * register moves are needed. K = 32 runs the single-chunk tail.
             *
             * @param pa[4] Activation row bases (row-major [M,K] rows of the tile).
             * @param pb[4] Weight row bases (row-major [N,K] columns of the tile).
             * @param K Reduction width (positive multiple of 32).
             * @param acc[16] Receives the lane partials with acc[4*r+c].
             */
            inline void avx512Bf16TileKernel(const uint16_t *pa[4], const uint16_t *pb[4],
                                             int K, float acc_lanes[16][16]) noexcept
            {
                alignas(64) unsigned char saved[1024];
                // zmm16-31 are callee-saved in the System V x86-64 ABI.
                asm volatile(
                    "vmovups %%zmm16, 0(%0)\n\t"
                    "vmovups %%zmm17, 64(%0)\n\t"
                    "vmovups %%zmm18, 128(%0)\n\t"
                    "vmovups %%zmm19, 192(%0)\n\t"
                    "vmovups %%zmm20, 256(%0)\n\t"
                    "vmovups %%zmm21, 320(%0)\n\t"
                    "vmovups %%zmm22, 384(%0)\n\t"
                    "vmovups %%zmm23, 448(%0)\n\t"
                    "vmovups %%zmm24, 512(%0)\n\t"
                    "vmovups %%zmm25, 576(%0)\n\t"
                    "vmovups %%zmm26, 640(%0)\n\t"
                    "vmovups %%zmm27, 704(%0)\n\t"
                    "vmovups %%zmm28, 768(%0)\n\t"
                    "vmovups %%zmm29, 832(%0)\n\t"
                    "vmovups %%zmm30, 896(%0)\n\t"
                    "vmovups %%zmm31, 960(%0)\n\t"
                    :: "r"(saved) : "memory");
                asm volatile(
                    "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"
                    "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
                    "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"
                    "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
                    "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"
                    "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
                    "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"
                    "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
                    "vpxord %%zmm8, %%zmm8, %%zmm8\n\t"
                    "vpxord %%zmm9, %%zmm9, %%zmm9\n\t"
                    "vpxord %%zmm10, %%zmm10, %%zmm10\n\t"
                    "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
                    "vpxord %%zmm12, %%zmm12, %%zmm12\n\t"
                    "vpxord %%zmm13, %%zmm13, %%zmm13\n\t"
                    "vpxord %%zmm14, %%zmm14, %%zmm14\n\t"
                    "vpxord %%zmm15, %%zmm15, %%zmm15\n\t"
                    // Prologue: set A (zmm16-23) = chunk 0. Invariant at label 1:
                    // set A holds the current chunk; %8 = K in BYTES (2*K elements).
                    "vmovdqu16 (%0), %%zmm16\n\t"
                    "vmovdqu16 (%1), %%zmm17\n\t"
                    "vmovdqu16 (%2), %%zmm18\n\t"
                    "vmovdqu16 (%3), %%zmm19\n\t"
                    "vmovdqu16 (%4), %%zmm20\n\t"
                    "vmovdqu16 (%5), %%zmm21\n\t"
                    "vmovdqu16 (%6), %%zmm22\n\t"
                    "vmovdqu16 (%7), %%zmm23\n\t"
                    "xor %%r12d, %%r12d\n\t"
                    "1:\n\t"
                    // If the next chunk does not exist, consume set A only.
                    "mov %%r12d, %%r13d\n\t"
                    "add $64, %%r13d\n\t"
                    "cmp %8, %%r13d\n\t"
                    "jge 6f\n\t"
                    // Load set B = chunk i+1, FMA set A = chunk i.
                    "vmovdqu16 (%0,%%r13,1), %%zmm24\n\t"
                    "vmovdqu16 (%1,%%r13,1), %%zmm25\n\t"
                    "vmovdqu16 (%2,%%r13,1), %%zmm26\n\t"
                    "vmovdqu16 (%3,%%r13,1), %%zmm27\n\t"
                    "vmovdqu16 (%4,%%r13,1), %%zmm28\n\t"
                    "vmovdqu16 (%5,%%r13,1), %%zmm29\n\t"
                    "vmovdqu16 (%6,%%r13,1), %%zmm30\n\t"
                    "vmovdqu16 (%7,%%r13,1), %%zmm31\n\t"
                    "vdpbf16ps %%zmm20, %%zmm16, %%zmm0\n\t"
                    "vdpbf16ps %%zmm21, %%zmm16, %%zmm1\n\t"
                    "vdpbf16ps %%zmm22, %%zmm16, %%zmm2\n\t"
                    "vdpbf16ps %%zmm23, %%zmm16, %%zmm3\n\t"
                    "vdpbf16ps %%zmm20, %%zmm17, %%zmm4\n\t"
                    "vdpbf16ps %%zmm21, %%zmm17, %%zmm5\n\t"
                    "vdpbf16ps %%zmm22, %%zmm17, %%zmm6\n\t"
                    "vdpbf16ps %%zmm23, %%zmm17, %%zmm7\n\t"
                    "vdpbf16ps %%zmm20, %%zmm18, %%zmm8\n\t"
                    "vdpbf16ps %%zmm21, %%zmm18, %%zmm9\n\t"
                    "vdpbf16ps %%zmm22, %%zmm18, %%zmm10\n\t"
                    "vdpbf16ps %%zmm23, %%zmm18, %%zmm11\n\t"
                    "vdpbf16ps %%zmm20, %%zmm19, %%zmm12\n\t"
                    "vdpbf16ps %%zmm21, %%zmm19, %%zmm13\n\t"
                    "vdpbf16ps %%zmm22, %%zmm19, %%zmm14\n\t"
                    "vdpbf16ps %%zmm23, %%zmm19, %%zmm15\n\t"
                    "add $64, %%r12d\n\t"
                    // If chunk i+2 does not exist, consume set B only.
                    "mov %%r12d, %%r14d\n\t"
                    "add $64, %%r14d\n\t"
                    "cmp %8, %%r14d\n\t"
                    "jge 7f\n\t"
                    // Load set A = chunk i+2, FMA set B = chunk i+1.
                    "vmovdqu16 (%0,%%r14,1), %%zmm16\n\t"
                    "vmovdqu16 (%1,%%r14,1), %%zmm17\n\t"
                    "vmovdqu16 (%2,%%r14,1), %%zmm18\n\t"
                    "vmovdqu16 (%3,%%r14,1), %%zmm19\n\t"
                    "vmovdqu16 (%4,%%r14,1), %%zmm20\n\t"
                    "vmovdqu16 (%5,%%r14,1), %%zmm21\n\t"
                    "vmovdqu16 (%6,%%r14,1), %%zmm22\n\t"
                    "vmovdqu16 (%7,%%r14,1), %%zmm23\n\t"
                    "add $64, %%r12d\n\t"
                    "vdpbf16ps %%zmm28, %%zmm24, %%zmm0\n\t"
                    "vdpbf16ps %%zmm29, %%zmm24, %%zmm1\n\t"
                    "vdpbf16ps %%zmm30, %%zmm24, %%zmm2\n\t"
                    "vdpbf16ps %%zmm31, %%zmm24, %%zmm3\n\t"
                    "vdpbf16ps %%zmm28, %%zmm25, %%zmm4\n\t"
                    "vdpbf16ps %%zmm29, %%zmm25, %%zmm5\n\t"
                    "vdpbf16ps %%zmm30, %%zmm25, %%zmm6\n\t"
                    "vdpbf16ps %%zmm31, %%zmm25, %%zmm7\n\t"
                    "vdpbf16ps %%zmm28, %%zmm26, %%zmm8\n\t"
                    "vdpbf16ps %%zmm29, %%zmm26, %%zmm9\n\t"
                    "vdpbf16ps %%zmm30, %%zmm26, %%zmm10\n\t"
                    "vdpbf16ps %%zmm31, %%zmm26, %%zmm11\n\t"
                    "vdpbf16ps %%zmm28, %%zmm27, %%zmm12\n\t"
                    "vdpbf16ps %%zmm29, %%zmm27, %%zmm13\n\t"
                    "vdpbf16ps %%zmm30, %%zmm27, %%zmm14\n\t"
                    "vdpbf16ps %%zmm31, %%zmm27, %%zmm15\n\t"
                    "jmp 1b\n\t"
                    "6:\n\t"
                    // Final chunk lives in set A.
                    "vdpbf16ps %%zmm20, %%zmm16, %%zmm0\n\t"
                    "vdpbf16ps %%zmm21, %%zmm16, %%zmm1\n\t"
                    "vdpbf16ps %%zmm22, %%zmm16, %%zmm2\n\t"
                    "vdpbf16ps %%zmm23, %%zmm16, %%zmm3\n\t"
                    "vdpbf16ps %%zmm20, %%zmm17, %%zmm4\n\t"
                    "vdpbf16ps %%zmm21, %%zmm17, %%zmm5\n\t"
                    "vdpbf16ps %%zmm22, %%zmm17, %%zmm6\n\t"
                    "vdpbf16ps %%zmm23, %%zmm17, %%zmm7\n\t"
                    "vdpbf16ps %%zmm20, %%zmm18, %%zmm8\n\t"
                    "vdpbf16ps %%zmm21, %%zmm18, %%zmm9\n\t"
                    "vdpbf16ps %%zmm22, %%zmm18, %%zmm10\n\t"
                    "vdpbf16ps %%zmm23, %%zmm18, %%zmm11\n\t"
                    "vdpbf16ps %%zmm20, %%zmm19, %%zmm12\n\t"
                    "vdpbf16ps %%zmm21, %%zmm19, %%zmm13\n\t"
                    "vdpbf16ps %%zmm22, %%zmm19, %%zmm14\n\t"
                    "vdpbf16ps %%zmm23, %%zmm19, %%zmm15\n\t"
                    "jmp 3f\n\t"
                    "7:\n\t"
                    // Final chunk lives in set B.
                    "vdpbf16ps %%zmm28, %%zmm24, %%zmm0\n\t"
                    "vdpbf16ps %%zmm29, %%zmm24, %%zmm1\n\t"
                    "vdpbf16ps %%zmm30, %%zmm24, %%zmm2\n\t"
                    "vdpbf16ps %%zmm31, %%zmm24, %%zmm3\n\t"
                    "vdpbf16ps %%zmm28, %%zmm25, %%zmm4\n\t"
                    "vdpbf16ps %%zmm29, %%zmm25, %%zmm5\n\t"
                    "vdpbf16ps %%zmm30, %%zmm25, %%zmm6\n\t"
                    "vdpbf16ps %%zmm31, %%zmm25, %%zmm7\n\t"
                    "vdpbf16ps %%zmm28, %%zmm26, %%zmm8\n\t"
                    "vdpbf16ps %%zmm29, %%zmm26, %%zmm9\n\t"
                    "vdpbf16ps %%zmm30, %%zmm26, %%zmm10\n\t"
                    "vdpbf16ps %%zmm31, %%zmm26, %%zmm11\n\t"
                    "vdpbf16ps %%zmm28, %%zmm27, %%zmm12\n\t"
                    "vdpbf16ps %%zmm29, %%zmm27, %%zmm13\n\t"
                    "vdpbf16ps %%zmm30, %%zmm27, %%zmm14\n\t"
                    "vdpbf16ps %%zmm31, %%zmm27, %%zmm15\n\t"
                    "3:\n\t"
                    "4:\n\t"
                    // Publish the 16 accumulators and restore callee-saved ZMMs
                    // inside the SAME asm: the compiler never sees the accs as
                    // outputs, so a separate store block could be scheduled over
                    // the clobbered registers.
                    "vmovups %%zmm0, 0(%9)\n\t"
                    "vmovups %%zmm1, 64(%9)\n\t"
                    "vmovups %%zmm2, 128(%9)\n\t"
                    "vmovups %%zmm3, 192(%9)\n\t"
                    "vmovups %%zmm4, 256(%9)\n\t"
                    "vmovups %%zmm5, 320(%9)\n\t"
                    "vmovups %%zmm6, 384(%9)\n\t"
                    "vmovups %%zmm7, 448(%9)\n\t"
                    "vmovups %%zmm8, 512(%9)\n\t"
                    "vmovups %%zmm9, 576(%9)\n\t"
                    "vmovups %%zmm10, 640(%9)\n\t"
                    "vmovups %%zmm11, 704(%9)\n\t"
                    "vmovups %%zmm12, 768(%9)\n\t"
                    "vmovups %%zmm13, 832(%9)\n\t"
                    "vmovups %%zmm14, 896(%9)\n\t"
                    "vmovups %%zmm15, 960(%9)\n\t"
                    "vmovups 0(%10), %%zmm16\n\t"
                    "vmovups 64(%10), %%zmm17\n\t"
                    "vmovups 128(%10), %%zmm18\n\t"
                    "vmovups 192(%10), %%zmm19\n\t"
                    "vmovups 256(%10), %%zmm20\n\t"
                    "vmovups 320(%10), %%zmm21\n\t"
                    "vmovups 384(%10), %%zmm22\n\t"
                    "vmovups 448(%10), %%zmm23\n\t"
                    "vmovups 512(%10), %%zmm24\n\t"
                    "vmovups 576(%10), %%zmm25\n\t"
                    "vmovups 640(%10), %%zmm26\n\t"
                    "vmovups 704(%10), %%zmm27\n\t"
                    "vmovups 768(%10), %%zmm28\n\t"
                    "vmovups 832(%10), %%zmm29\n\t"
                    "vmovups 896(%10), %%zmm30\n\t"
                    "vmovups 960(%10), %%zmm31\n\t"
                    : "+r"(pa[0]), "+r"(pa[1]), "+r"(pa[2]), "+r"(pa[3]),
                      "+r"(pb[0]), "+r"(pb[1]), "+r"(pb[2]), "+r"(pb[3])
                    : "r"(static_cast<std::uint32_t>(K) * 2u), "r"(acc_lanes), "r"(saved)
                    : "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7",
                      "zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",
                      "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23",
                      "zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31",
                      "r12", "r13", "r14", "cc", "memory");
            }

            /**
             * @brief Execute the AVX512-BF16 GEMM route (pipelined 4×4 tiles).
             * @param A Row-major BF16 activations [M,K].
             * @param B Row-major BF16 transposed weights [N,K].
             * @param C FP32 output [M,N].
             * @param M Row count.
             * @param N Column count.
             * @param K Reduction width; must be a positive multiple of 32.
             * @param transpose_B Must be true (row-major weights, the CPU
             *        floating GEMM contract).
             * @param alpha Output scale.
             * @param beta Existing-output scale.
             * @throws std::invalid_argument For non-admitted geometry or null pointers.
             * @throws std::runtime_error When the engine is not compiled/admitted.
             * @details Contract: K is visited in ascending 32-element chunks.
             * Within one chunk, hardware lane i accumulates the pair sum
             * a[2i]·b[2i] + a[2i+1]·b[2i+1] — each lane owns a disjoint K
             * partition (pairs with index i mod 16). After the K loop the 16
             * lanes reduce through fixedReduceLanes16(), then the shared
             * epilogue rounds the output.
             *
             * Economy: each 4×4 microtile runs through the hand-scheduled
             * avx512Bf16TileKernel pipeline (one 32-element chunk in flight),
             * then the fixed lane tree and rounded epilogue publish the word.
             * N panels are the parallel dimension with the M sweep inside so
             * the K×128-column weight panel is shared across the row sweep.
             * Blocking and scheduling change only data movement; every output
             * word still visits K in the same ascending chunk order with the
             * same per-lane partitions, so the route contract and its bytes
             * are unchanged. Output tiles are independent; OpenMP schedules
             * the N panels statically so every worker builds identical words.
             */
            inline bool run_avx512bf16_matmul(const uint16_t *A, const uint16_t *B,
                                              float *C, int M, int N, int K,
                                              bool transpose_B, float alpha, float beta)
            {
                if (!compiledWithAVX512BF16() || !cpu_supports_avx512_bf16())
                    throw std::runtime_error(
                        "The AVX512-BF16 GEMM engine is not compiled into this CPU ISA envelope or the host does not advertise AVX512-BF16");
                if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0 || !transpose_B)
                    throw std::invalid_argument(
                        "AVX512-BF16 GEMM requires non-null A/B/C, M,N,K > 0, K % 32 == 0 and transpose_B=true");
                if (bf16RouteProbeEnabled().load(std::memory_order_relaxed))
                    bf16NativeRouteCounter(Engine::AVX512BF16).fetch_add(1, std::memory_order_relaxed);

                // 4×4 microkernel: 16 accumulators + 4 A + 4 B chunks keep the
                // working set register-resident. Loop order is the GEMM bold
                // blocking rule: N panels are the parallel dimension (one per
                // thread) and the M sweep runs inside, so the Kx128-column
                // weight panel is L2-resident across the ENTIRE row sweep and
                // the weight matrix is streamed from DRAM once per N panel
                // (the previous collapse(2) schedule let row panels migrate
                // across threads and re-stream weights). Loads are unaligned
                // vector dereferences (one vmovdqu per chunk) and the K loop
                // prefetches the next chunk of every stream.
                constexpr int kRowBlock = 4;
                constexpr int kColumnBlock = 4;
                const int row_group_count = (M + kRowBlock - 1) / kRowBlock;
                const int col_tile_count = (N + kColumnBlock - 1) / kColumnBlock;
                auto work = [&]()
                {
#pragma omp for schedule(static)
                    for (int nbg = 0; nbg < col_tile_count; nbg += 32)
                    {
                        const int ct_end = std::min(nbg + 32, col_tile_count);
                        for (int mbg = 0; mbg < row_group_count; mbg += 8)
                        {
                            const int rg_end = std::min(mbg + 8, row_group_count);
                            for (int rg = mbg; rg < rg_end; ++rg)
                            {
                                const int row0 = rg * kRowBlock;
                                const int rows = std::min(kRowBlock, M - row0);
                                for (int ct = nbg; ct < ct_end; ++ct)
                                {
                                    const int first_col = ct * kColumnBlock;
                                    const int cols = std::min(kColumnBlock, N - first_col);
                                    const uint16_t *pa[4]
                                        = {A + static_cast<std::size_t>(row0) * K,
                                           A + static_cast<std::size_t>(row0 + 1) * K,
                                           A + static_cast<std::size_t>(row0 + 2) * K,
                                           A + static_cast<std::size_t>(row0 + 3) * K};
                                    const uint16_t *pb[4]
                                        = {B + static_cast<std::size_t>(first_col) * K,
                                           B + static_cast<std::size_t>(first_col + 1) * K,
                                           B + static_cast<std::size_t>(first_col + 2) * K,
                                           B + static_cast<std::size_t>(first_col + 3) * K};
                                    alignas(64) float acc_lanes[kRowBlock * kColumnBlock][16];
                                    avx512Bf16TileKernel(pa, pb, K, acc_lanes);
                                    for (int row = 0; row < rows; ++row)
                                    {
                                        float *c = C + static_cast<std::size_t>(row0 + row) * N;
                                        for (int col = 0; col < cols; ++col)
                                        {
                                            const float dot = fixedReduceLanes16(
                                                _mm512_loadu_ps(acc_lanes[row * kColumnBlock + col]));
                                            float *out = c + first_col + col;
                                            *out = completeEpilogueValue(dot, alpha, beta, out);
                                        }
                                    }
                                }
                            }
                        }
                    }
                };
                OMP_WORKSHARE_REGION(work);
                return true;
            }

#endif // __AVX512BF16__

            // =================================================================
            // AMX-BF16 engine
            // =================================================================

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)

            /**
             * @brief AMX palette-1 tile configuration (BF16 tiles, FP32 tiles).
             *
             * The layout is the 64-byte TILECFG memory operand consumed by
             * tileloadconfig, packed exactly as oneDNN's `palette_config_t`:
             * one palette byte, one start-row byte, 14 reserved bytes, then
             * sixteen uint16 column-size words (offset 16) and sixteen uint8
             * row counts (offset 48). Entry i (i = 0..7) is one tile's shape:
             * `cols[i]` = column bytes, `rows[i]` = row count. The hardware
             * parses this image directly, so the layout must be exact — a
             * paired {rows, colsb} descriptor table faults ldtilecfg with #GP.
             * Palette 1 is the 16-bit floating-point palette: A/B tiles 16
             * rows × 32 bytes (16 BF16), D tiles 16 rows × 64 bytes (16 FP32).
             */
#pragma pack(push, 1)
            struct alignas(64) AMXPalette1Config
            {
                std::uint8_t palette_id = 1;
                std::uint8_t start_row = 0;
                std::uint8_t reserved[14] = {};
                std::uint16_t cols[16] = {};
                std::uint8_t rows[16] = {};
            };
#pragma pack(pop)
            static_assert(sizeof(AMXPalette1Config) == 64,
                "AMX TILECFG is exactly 64 bytes: 16-byte header + 16 uint16 cols + 16 uint8 rows");

            /** @brief Tile indices of the 1A+3B+4D AMX-BF16 GEMM program. */
            enum AMXTile : int
            {
                AMX_TILE_A = 0,
                AMX_TILE_B0 = 1, AMX_TILE_B1 = 2, AMX_TILE_B2 = 3,
                AMX_TILE_D0 = 4, // four D tiles: 64 output rows per block
            };
            // Each A/B tile holds 16 rows x 32 BF16 (64-byte rows, one
            // 16x16x32 tdpbf16ps per tile pair); each D tile holds 16 rows
            // x 16 FP32 (64-byte rows).

            /** @brief Register-blocking geometry: 64 rows × 48 columns per tile block. */
            inline constexpr int kAMXBf16MBlock = 64;
            inline constexpr int kAMXBf16NBlock = 48;

            /**
             * @brief One AMX tile-state scope: loadconfig on entry, release on exit.
             *
             * AMX tile registers and the tile config are per-logical-processor
             * state. Every worker thread (including the submitting thread when
             * it executes a worksharing chunk) owns exactly one scope per
             * GEMM; tile state is always released before the thread leaves the
             * kernel so no stale tile data leaks into later kernels.
             *
             * The load/release/tile instructions are reached through the
             * canonical `<immintrin.h>` AMX macros (`_tile_loadconfig`,
             * `_tile_release`, `_tile_loadd`, `_tile_stored`, `_tile_zero`,
             * `_tile_dpbf16ps`), the API the GNU x86 toolchain (the canonical
             * x86 deployment compiler) exposes. The configure-time envelope
             * probe compiles the same calls, so a toolchain with a different
             * AMX intrinsic ABI fails configuration instead of emitting a
             * binary that cannot be built here.
             */
            class AMXTileScope final
            {
            public:
                AMXTileScope() noexcept
                {
                    // Universal palette: every tile 64 bytes wide, matching
                    // oneDNN's palette image. A mixed 32-byte/64-byte palette
                    // is refused by some virtualized AMX implementations, so
                    // the uniform shape is the deployment-compatible one.
                    AMXPalette1Config config{};
                    for (int t = 0; t < 8; ++t)
                    {
                        config.cols[t] = 64;
                        config.rows[t] = 16; // 16 rows × 32 BF16 or 16 FP32
                    }
                    _tile_loadconfig(&config);
                }
                ~AMXTileScope() noexcept
                {
                    _tile_release();
                }
                AMXTileScope(const AMXTileScope &) = delete;
                AMXTileScope &operator=(const AMXTileScope &) = delete;
            };

            /** @brief Zero one FP32 destination tile before accumulation. */
            inline void amxZeroTile(int tile) noexcept
            {
                switch (tile) // GNU toolchain requires a compile-time constant tile operand
                {
                case 0: _tile_zero(0); return;
                case 1: _tile_zero(1); return;
                case 2: _tile_zero(2); return;
                case 3: _tile_zero(3); return;
                case 4: _tile_zero(4); return;
                case 5: _tile_zero(5); return;
                case 6: _tile_zero(6); return;
                case 7: _tile_zero(7); return;
                default: std::abort(); // unreachable: engine call sites use constant enum tiles
                }
            }

            /** @brief Load one 16×32 BF16 tile directly from strided matrix rows.
             *
             * The tile base is any alignment the executing implementation
             * accepts (this appliance does not enforce the documented 64-byte
             * alignment; hardware may), so callers pre-check alignment and
             * fall back to `amxLoadBf16Tile` over the scratch-copy path when
             * the base is not 64-byte aligned.
             * @param base First row of the tile inside the source matrix.
             * @param stride Row stride in bytes (the producer's row length).
             */
            inline void amxLoadBf16TileStrided(int tile, const void *base,
                                               std::size_t stride) noexcept
            {
                switch (tile)
                {
                case 0: asm volatile("tileloadd (%0,%1,1), %%tmm0" :: "r"(base), "r"(stride) : "memory"); return;
                case 1: asm volatile("tileloadd (%0,%1,1), %%tmm1" :: "r"(base), "r"(stride) : "memory"); return;
                case 2: asm volatile("tileloadd (%0,%1,1), %%tmm2" :: "r"(base), "r"(stride) : "memory"); return;
                case 3: asm volatile("tileloadd (%0,%1,1), %%tmm3" :: "r"(base), "r"(stride) : "memory"); return;
                case 4: asm volatile("tileloadd (%0,%1,1), %%tmm4" :: "r"(base), "r"(stride) : "memory"); return;
                case 5: asm volatile("tileloadd (%0,%1,1), %%tmm5" :: "r"(base), "r"(stride) : "memory"); return;
                case 6: asm volatile("tileloadd (%0,%1,1), %%tmm6" :: "r"(base), "r"(stride) : "memory"); return;
                case 7: asm volatile("tileloadd (%0,%1,1), %%tmm7" :: "r"(base), "r"(stride) : "memory"); return;
                default: std::abort(); // unreachable: constant enum tiles
                }
            }

            /** @brief Load one 16×32 BF16 tile from a 64-byte-aligned buffer. */
            inline void amxLoadBf16Tile(int tile, const void *base) noexcept
            {
                switch (tile)
                {
                case 0: _tile_loadd(0, base, 64); return;
                case 1: _tile_loadd(1, base, 64); return;
                case 2: _tile_loadd(2, base, 64); return;
                case 3: _tile_loadd(3, base, 64); return;
                case 4: _tile_loadd(4, base, 64); return;
                case 5: _tile_loadd(5, base, 64); return;
                case 6: _tile_loadd(6, base, 64); return;
                case 7: _tile_loadd(7, base, 64); return;
                default: std::abort(); // unreachable: engine call sites use constant enum tiles
                }
            }

            /** @brief Store one 16×16 FP32 tile to a 64-byte-aligned buffer. */
            inline void amxStoreF32Tile(int tile, void *base) noexcept
            {
                switch (tile)
                {
                case 0: _tile_stored(0, base, 64); return;
                case 1: _tile_stored(1, base, 64); return;
                case 2: _tile_stored(2, base, 64); return;
                case 3: _tile_stored(3, base, 64); return;
                case 4: _tile_stored(4, base, 64); return;
                case 5: _tile_stored(5, base, 64); return;
                case 6: _tile_stored(6, base, 64); return;
                case 7: _tile_stored(7, base, 64); return;
                default: std::abort(); // unreachable: engine call sites use constant enum tiles
                }
            }

            /** @brief Accumulate A×B into the destination tile (16×16×16). */
            inline void amxDpbf16ConstAll(int dst, int a, int b) noexcept;
            inline void amxDpbf16ConstAB(int dst, int a, int b) noexcept;
            inline void amxDpbf16Const(int dst, int a, int b) noexcept;
            inline void amxDpbf16(int dst, int a, int b) noexcept
            {
                switch (dst) // dst is the discriminant; a/b are runtime tiles below
                {
                case 0: amxDpbf16Const(0, a, b); return;
                case 1: amxDpbf16Const(1, a, b); return;
                case 2: amxDpbf16Const(2, a, b); return;
                case 3: amxDpbf16Const(3, a, b); return;
                case 4: amxDpbf16Const(4, a, b); return;
                case 5: amxDpbf16Const(5, a, b); return;
                case 6: amxDpbf16Const(6, a, b); return;
                case 7: amxDpbf16Const(7, a, b); return;
                default: std::abort(); // unreachable: constant enum tiles
                }
            }

            /** @brief Constant-dispatch helper for amxDpbf16 (a/b runtime). */
            inline void amxDpbf16Const(int dst, int a, int b) noexcept
            {
                switch (a)
                {
                case 0: amxDpbf16ConstAB(dst, 0, b); return;
                case 1: amxDpbf16ConstAB(dst, 1, b); return;
                case 2: amxDpbf16ConstAB(dst, 2, b); return;
                case 3: amxDpbf16ConstAB(dst, 3, b); return;
                case 4: amxDpbf16ConstAB(dst, 4, b); return;
                case 5: amxDpbf16ConstAB(dst, 5, b); return;
                case 6: amxDpbf16ConstAB(dst, 6, b); return;
                case 7: amxDpbf16ConstAB(dst, 7, b); return;
                default: std::abort(); // unreachable: constant enum tiles
                }
            }
            inline void amxDpbf16ConstAB(int dst, int a, int b) noexcept
            {
                switch (b)
                {
                case 0: amxDpbf16ConstAll(dst, a, 0); return;
                case 1: amxDpbf16ConstAll(dst, a, 1); return;
                case 2: amxDpbf16ConstAll(dst, a, 2); return;
                case 3: amxDpbf16ConstAll(dst, a, 3); return;
                case 4: amxDpbf16ConstAll(dst, a, 4); return;
                case 5: amxDpbf16ConstAll(dst, a, 5); return;
                case 6: amxDpbf16ConstAll(dst, a, 6); return;
                case 7: amxDpbf16ConstAll(dst, a, 7); return;
                default: std::abort(); // unreachable: constant enum tiles
                }
            }
            inline void amxDpbf16ConstAll(int dst, int a, int b) noexcept
            {
                switch (dst)
                {
                case 0: _tile_dpbf16ps(0, a, b); return;
                case 1: _tile_dpbf16ps(1, a, b); return;
                case 2: _tile_dpbf16ps(2, a, b); return;
                case 3: _tile_dpbf16ps(3, a, b); return;
                case 4: _tile_dpbf16ps(4, a, b); return;
                case 5: _tile_dpbf16ps(5, a, b); return;
                case 6: _tile_dpbf16ps(6, a, b); return;
                case 7: _tile_dpbf16ps(7, a, b); return;
                default: std::abort(); // unreachable: constant enum tiles
                }
            }
            /**
             * @brief Copy one 16×32 BF16 matrix tile with a row stride.
             * @param dst 16×32 BF16 scratch (64-byte aligned by the caller).
             * @param base First element of the tile inside the source matrix.
             * @param row_stride_bytes Distance in bytes between source rows.
             * @details `tileloadd64` requires the tile base 64-byte aligned,
             * which production matrix rows cannot guarantee (BF16 rows are
             * 2·K bytes apart); the scratch copy makes the alignment contract
             * explicit and local. The PreparedWeightStore follow-up can own
             * aligned packed BF16 tiles and remove this copy.
             */
            inline void copyAMXTile(std::uint16_t *dst, const std::uint16_t *base,
                                    std::size_t row_stride_bytes) noexcept
            {
                const auto *bytes = reinterpret_cast<const std::byte *>(base);
                for (int row = 0; row < 16; ++row)
                    std::memcpy(dst + static_cast<std::size_t>(row) * 32,
                                bytes + static_cast<std::size_t>(row) * row_stride_bytes, 64);
            }

            /**
             * @brief Execute the AMX-BF16 tile GEMM route.
             * @param A Row-major BF16 activations [M,K].
             * @param B BF16 weights; transpose_B=true means row-major [N,K],
             *        otherwise row-major [K,N] (both layouts are admitted).
             * @param C FP32 output [M,N].
             * @param M Row count (multiple of 16).
             * @param N Column count (multiple of 16).
             * @param K Reduction width (multiple of 32).
             * @param transpose_B Weight layout selector.
             * @param alpha Output scale.
             * @param beta Existing-output scale.
             * @throws std::invalid_argument For non-tile geometry or null pointers.
             * @throws std::runtime_error When the engine is not compiled/admitted.
             * @details Contract: each 16×16 output sub-block zeroes its FP32
             * accumulator, then walks K in ascending 16-element tiles with
             * `tdpbf16ps` (the hardware's deterministic 16-wide FP32 dot
             * product per output element and its fixed internal accumulation
             * order). Register blocking (one A tile + three B tiles + four D
             * tiles) makes each weight tile load serve four activation
             * sub-tiles and each activation tile load serve three weight
             * tiles; blocking changes only tile load traffic, never the
             * per-element K-ascending accumulation sequence, so the route
             * contract and bytes are unchanged. Output blocks are
             * independent, so OpenMP schedules the 64-row m-blocks statically.
             */
            inline bool run_amxbf16_matmul(const uint16_t *A, const uint16_t *B,
                                           float *C, int M, int N, int K,
                                           bool transpose_B, float alpha, float beta)
            {
                if (!compiledWithAMXBF16() || !cpu_supports_amx_bf16()
                        || !cpu_amx_data_instructions_usable())
                    throw std::runtime_error(
                        "The AMX-BF16 GEMM engine is not compiled into this CPU ISA envelope, the host does not advertise AMX-BF16, or the executing host refuses XTILEDATA permission for this process (tile data instructions are not usable)");
                if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 ||
                    (M % 16) != 0 || (N % 16) != 0 || (K % 32) != 0)
                    throw std::invalid_argument(
                        "AMX-BF16 GEMM requires non-null A/B/C, M,N,K > 0 and M % 16 == 0, N % 16 == 0, K % 32 == 0");
                if (bf16RouteProbeEnabled().load(std::memory_order_relaxed))
                    bf16NativeRouteCounter(Engine::AMXBF16).fetch_add(1, std::memory_order_relaxed);

                const std::size_t a_stride = static_cast<std::size_t>(K) * 2;
                const std::size_t b_stride =
                    static_cast<std::size_t>(transpose_B ? K : N) * 2;
                const int row_blocks = M / 16;
                const int m_blocks = (row_blocks + 3) / 4;
                const int n_blocks = (N + kAMXBf16NBlock - 1) / kAMXBf16NBlock;
                auto work = [&]()
                {
#pragma omp for schedule(static)
                    for (int chunk = 0; chunk < m_blocks * n_blocks; ++chunk)
                    {
                        // Flattened (mb, nb) chunking: a 64-row x 48-column
                        // block is the unit of worker parallelism. The small
                        // M shapes that dominate serving expose only one or
                        // two row blocks, so partitioning by M alone would
                        // leave most workers idle; the column dimension
                        // supplies the remaining parallelism.
                        const int mb = chunk / n_blocks;
                        const int nb = (chunk % n_blocks) * kAMXBf16NBlock;
                        // Per-thread tile scratch: AMX tile state and the tile
                        // buffers are per-logical-processor state, so every
                        // worker owns its own storage and scope.
                        alignas(64) std::uint16_t a_tile[16 * 32];
                        alignas(64) std::uint16_t b_tile[16 * 32];
                        alignas(64) float c_tile[16 * 16];
                        AMXTileScope scope;
                        const int msub_live = std::min(4, row_blocks - mb * 4);
                        // Register blocking: one A tile + three B tiles + four
                        // D tiles = seven live tiles, so each 16×16 output
                        // sub-block shares its weight/activation tile loads
                        // with the whole 64×48 block (tile loadd traffic is
                        // divided by the block factors versus one 16×16
                        // sub-block per pass). K stays ascending in 32-element
                        // windows and every D element accumulates through exactly
                        // the same sequence of tdpbf16ps instructions, so the
                        // route's arithmetic contract is unchanged. The three B
                        // column groups are processed in separate passes: D
                        // tiles would otherwise sum A x B0 + A x B1 + A x B2
                        // into one 16-column output instead of keeping the
                        // column groups side by side.
                        {
                            // Every 16-column group gets its own D pass; the
                            // last 48-block may hold fewer than three groups.
                            const int bsub_live = std::min(3, (N - nb) / 16);
                            for (int c = 0; c < bsub_live; ++c)
                            {
                                for (int d = 0; d < msub_live; ++d)
                                    amxZeroTile(AMX_TILE_D0 + d);
                                for (int kt = 0; kt < K; kt += 32)
                                {
                                    // The weight tile stays resident across all
                                    // four activation sub-tiles of this K step.
                                    const uint16_t *b_base =
                                        transpose_B
                                            ? B + static_cast<std::size_t>(nb + c * 16) * K + kt
                                            : B + static_cast<std::size_t>(kt) * N + nb + c * 16;
                                    if ((reinterpret_cast<std::uintptr_t>(b_base) & 63u) == 0)
                                        amxLoadBf16TileStrided(AMX_TILE_B0, b_base, b_stride);
                                    else
                                    {
                                        copyAMXTile(b_tile, b_base, b_stride);
                                        amxLoadBf16Tile(AMX_TILE_B0, b_tile);
                                    }
                                    for (int d = 0; d < msub_live; ++d)
                                    {
                                        const uint16_t *a_base =
                                            A + static_cast<std::size_t>(mb * 4 + d) * 16 * K + kt;
                                        if ((reinterpret_cast<std::uintptr_t>(a_base) & 63u) == 0)
                                            amxLoadBf16TileStrided(AMX_TILE_A, a_base, a_stride);
                                        else
                                        {
                                            copyAMXTile(a_tile, a_base, a_stride);
                                            amxLoadBf16Tile(AMX_TILE_A, a_tile);
                                        }
                                        amxDpbf16(AMX_TILE_D0 + d, AMX_TILE_A, AMX_TILE_B0);
                                    }
                                }
                                for (int d = 0; d < msub_live; ++d)
                                {
                                    amxStoreF32Tile(AMX_TILE_D0 + d, c_tile);
                                    for (int row = 0; row < 16; ++row)
                                    {
                                        float *out = C + static_cast<std::size_t>(mb * 4 + d) * 16 * N
                                            + static_cast<std::size_t>(row) * N + nb + c * 16;
                                        for (int col = 0; col < 16; ++col)
                                        {
                                            const float dot = c_tile[row * 16 + col];
                                            out[col] = completeEpilogueValue(dot, alpha, beta, out + col);
                                        }
                                    }
                                }
                            }
                        }
                    }
                };
                OMP_WORKSHARE_REGION(work);
                return true;
            }

#endif // __AMX_TILE__ && __AMX_BF16__

            // =================================================================
            // Automatic policy and public entry
            // =================================================================

            /**
             * @brief Select the engine for one call without executing it.
             * @param M Number of activation rows.
             * @param N Number of output columns.
             * @param K Reduction width.
             * @param transpose_B Weight layout selector.
             * @param requested Requested engine (Automatic for the policy).
             * @return Resolved engine identity.
             * @throws std::invalid_argument For an explicit engine whose
             *         geometry is not admitted.
             * @throws std::runtime_error For an explicit engine that is not
             *         compiled/admitted on this binary+host.
             * @details Automatic policy (single authority):
             *   1. M == 1 resolves Library (decode-exact serial rows own the
             *      contract).
             *   2. AMX-BF16 when compiled/admitted and full 16×16×16 tiles.
             *   3. AVX512-BF16 when compiled/admitted and K % 32 == 0 with
             *      transpose_B=true.
             *   4. Otherwise Library (the current production route).
             * An explicit request either resolves to itself (after admission
             * checks) or throws; it never degrades to another engine.
             */
            inline Engine selectEngine(int M, int N, int K, bool transpose_B,
                                       Engine requested = Engine::Automatic)
            {
                if (requested == Engine::Automatic)
                {
                    if (M <= 1)
                        return Engine::Library;
                    if (engineHardwareAdmitted(Engine::AMXBF16) &&
                        engineGeometryAdmits(Engine::AMXBF16, M, N, K, transpose_B))
                        return Engine::AMXBF16;
                    if (engineHardwareAdmitted(Engine::AVX512BF16) &&
                        engineGeometryAdmits(Engine::AVX512BF16, M, N, K, transpose_B))
                        return Engine::AVX512BF16;
                    return Engine::Library;
                }
                if (requested == Engine::Library)
                    return Engine::Library;
                if (!engineHardwareAdmitted(requested))
                    throw std::runtime_error(
                        std::string("The requested BF16 GEMM engine '") + engineName(requested) +
                        "' is not compiled into this CPU ISA envelope or the executing host does not advertise it");
                if (!engineGeometryAdmits(requested, M, N, K, transpose_B))
                    throw std::invalid_argument(
                        std::string("The requested BF16 GEMM engine '") + engineName(requested) +
                        "' does not admit the geometry M=" + std::to_string(M) +
                        " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                        " transpose_B=" + (transpose_B ? "true" : "false"));
                return requested;
            }

            /**
             * @brief Run one homogeneous BF16 GEMM through the selected engine.
             * @param A Row-major BF16 activations [M,K].
             * @param B BF16 weights (layout per transpose_B).
             * @param C FP32 output [M,N].
             * @param M Row count.
             * @param N Column count.
             * @param K Reduction width.
             * @param transpose_B Weight layout selector.
             * @param alpha Output scale.
             * @param beta Existing-output scale.
             * @param requested Engine request (Automatic default).
             * @param selected Receives the resolved engine before execution.
             * @return True when a native engine executed; false only when the
             *         automatic policy resolved Library (the caller retains
             *         its existing library/serial route).
             * @throws std::invalid_argument / std::runtime_error As documented
             *         by selectEngine() and the native entries.
             */
            inline bool run_bf16_gemm(const uint16_t *A, const uint16_t *B,
                                      float *C, int M, int N, int K,
                                      bool transpose_B, float alpha, float beta,
                                      Engine requested = Engine::Automatic,
                                      Engine *selected = nullptr)
            {
                const Engine engine = selectEngine(M, N, K, transpose_B, requested);
                if (selected)
                    *selected = engine;
                switch (engine)
                {
                case Engine::AVX512BF16:
#if defined(__AVX512BF16__)
                    return run_avx512bf16_matmul(A, B, C, M, N, K, transpose_B, alpha, beta);
#else
                    throw std::runtime_error("AVX512-BF16 engine unreachable without the compiled envelope");
#endif
                case Engine::AMXBF16:
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
                    return run_amxbf16_matmul(A, B, C, M, N, K, transpose_B, alpha, beta);
#else
                    throw std::runtime_error("AMX-BF16 engine unreachable without the compiled envelope");
#endif
                case Engine::Library:
                case Engine::Automatic:
                    return false;
                }
                return false;
            }

        } // namespace cpu_bf16
    } // namespace gemm
} // namespace llaminar2
