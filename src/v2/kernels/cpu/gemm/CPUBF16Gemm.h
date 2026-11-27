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
                    return (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
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
             * @param ptr Contiguous 64-byte source.
             * @return The loaded chunk.
             * @details Both GCC and Clang define __m512bh as a vector type, so a
             * word copy is ABI-portable without depending on either compiler's
             * cast intrinsic spelling. Release code lowers this to one load.
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
             * @brief Execute the AVX512-BF16 GEMM route.
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
             * epilogue rounds the output. Rows are independent; OpenMP
             * schedules them statically so every worker builds identical words.
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

                constexpr int kColumnBlock = 8; // 8 accumulator ZMMs stay register-resident
                const int col_tiles = (N + kColumnBlock - 1) / kColumnBlock;
                auto work = [&]()
                {
#pragma omp for schedule(static)
                    for (int row = 0; row < M; ++row)
                    {
                        const uint16_t *a = A + static_cast<std::size_t>(row) * K;
                        float *c = C + static_cast<std::size_t>(row) * N;
                        for (int ct = 0; ct < col_tiles; ++ct)
                        {
                            const int first_col = ct * kColumnBlock;
                            const int cols = std::min(kColumnBlock, N - first_col);
                            __m512 acc[kColumnBlock];
#pragma unroll
                            for (int col = 0; col < kColumnBlock; ++col)
                                acc[col] = _mm512_setzero_ps();
                            for (int kt = 0; kt < K; kt += 32)
                            {
                                const __m512bh a_chunk = loadBf16Chunk(a + kt);
                                for (int col = 0; col < cols; ++col)
                                {
                                    const uint16_t *b_row =
                                        B + static_cast<std::size_t>(first_col + col) * K + kt;
                                    acc[col] = _mm512_dpbf16_ps(
                                        acc[col], a_chunk, loadBf16Chunk(b_row));
                                }
                            }
                            for (int col = 0; col < cols; ++col)
                            {
                                const float dot = fixedReduceLanes16(acc[col]);
                                float *out = c + first_col + col;
                                *out = completeEpilogueValue(dot, alpha, beta, out);
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

            /** @brief Tile indices used by the AMX-BF16 GEMM program. */
            enum AMXTile : int { AMX_TILE_A = 0, AMX_TILE_B = 1, AMX_TILE_D = 2 };

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
                    AMXPalette1Config config{};
                    config.cols[AMX_TILE_A] = 32; config.rows[AMX_TILE_A] = 16; // 16 rows × 16 BF16
                    config.cols[AMX_TILE_B] = 32; config.rows[AMX_TILE_B] = 16;
                    config.cols[AMX_TILE_D] = 64; config.rows[AMX_TILE_D] = 16; // 16 rows × 16 FP32
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
                _tile_zero(tile);
            }

            /** @brief Load one 16×16 BF16 tile from a 64-byte-aligned buffer. */
            inline void amxLoadBf16Tile(int tile, const void *base) noexcept
            {
                _tile_loadd(tile, base, 32);
            }

            /** @brief Store one 16×16 FP32 tile to a 64-byte-aligned buffer. */
            inline void amxStoreF32Tile(int tile, void *base) noexcept
            {
                _tile_stored(tile, base, 64);
            }

            /** @brief Accumulate A×B into the destination tile (16×16×16). */
            inline void amxDpbf16(int dst, int a, int b) noexcept
            {
                _tile_dpbf16ps(dst, a, b);
            }

            /**
             * @brief Copy one 16×16 BF16 matrix tile with a row stride.
             * @param dst 16×16 BF16 scratch (64-byte aligned by the caller).
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
                    std::memcpy(dst + static_cast<std::size_t>(row) * 16,
                                bytes + static_cast<std::size_t>(row) * row_stride_bytes, 32);
            }

            /**
             * @brief Execute the AMX-BF16 tile GEMM route.
             * @param A Row-major BF16 activations [M,K].
             * @param B BF16 weights; transpose_B=true means row-major [N,K],
             *        otherwise row-major [K,N] (both layouts are admitted).
             * @param C FP32 output [M,N].
             * @param M Row count (multiple of 16).
             * @param N Column count (multiple of 16).
             * @param K Reduction width (multiple of 16).
             * @param transpose_B Weight layout selector.
             * @param alpha Output scale.
             * @param beta Existing-output scale.
             * @throws std::invalid_argument For non-tile geometry or null pointers.
             * @throws std::runtime_error When the engine is not compiled/admitted.
             * @details Contract: each 16×16 output tile zeroes its FP32
             * accumulator, then walks K in ascending 16-element tiles with
             * `tdpbf16ps` (the hardware's deterministic 16-wide FP32 dot
             * product per output element and its fixed internal accumulation
             * order). The finished tile is stored to a local scratch and each
             * word publishes through the shared epilogue. Output tiles are
             * independent, so OpenMP schedules the 16-row row-blocks statically.
             */
            inline bool run_amxbf16_matmul(const uint16_t *A, const uint16_t *B,
                                           float *C, int M, int N, int K,
                                           bool transpose_B, float alpha, float beta)
            {
                if (!compiledWithAMXBF16() || !cpu_supports_amx_bf16())
                    throw std::runtime_error(
                        "The AMX-BF16 GEMM engine is not compiled into this CPU ISA envelope or the host does not advertise AMX-BF16");
                if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 ||
                    (M % 16) != 0 || (N % 16) != 0 || (K % 16) != 0)
                    throw std::invalid_argument(
                        "AMX-BF16 GEMM requires non-null A/B/C, M,N,K > 0 and M % 16 == 0, N % 16 == 0, K % 16 == 0");
                if (bf16RouteProbeEnabled().load(std::memory_order_relaxed))
                    bf16NativeRouteCounter(Engine::AMXBF16).fetch_add(1, std::memory_order_relaxed);

                alignas(64) std::uint16_t a_tile[16 * 16];
                alignas(64) std::uint16_t b_tile[16 * 16];
                alignas(64) float c_tile[16 * 16];
                const std::size_t a_stride = static_cast<std::size_t>(K) * 2;
                const std::size_t b_stride =
                    static_cast<std::size_t>(transpose_B ? K : N) * 2;
                const int row_blocks = M / 16;
                auto work = [&]()
                {
#pragma omp for schedule(static)
                    for (int rb = 0; rb < row_blocks; ++rb)
                    {
                        AMXTileScope scope;
                        const int row0 = rb * 16;
                        for (int nt = 0; nt < N; nt += 16)
                        {
                            amxZeroTile(AMX_TILE_D);
                            for (int kt = 0; kt < K; kt += 16)
                            {
                                // A tile: 16 rows of the activation block.
                                copyAMXTile(a_tile, A + static_cast<std::size_t>(row0) * K + kt,
                                            a_stride);
                                amxLoadBf16Tile(AMX_TILE_A, a_tile);
                                // B tile: 16 K rows of the weight block; the two
                                // layouts share one strided row copy.
                                copyAMXTile(b_tile,
                                            transpose_B
                                                ? B + static_cast<std::size_t>(nt) * K + kt
                                                : B + static_cast<std::size_t>(kt) * N + nt,
                                            b_stride);
                                amxLoadBf16Tile(AMX_TILE_B, b_tile);
                                amxDpbf16(AMX_TILE_D, AMX_TILE_A, AMX_TILE_B);
                            }
                            amxStoreF32Tile(AMX_TILE_D, c_tile);
                            for (int row = 0; row < 16; ++row)
                            {
                                float *out = C + static_cast<std::size_t>(row0 + row) * N + nt;
                                for (int col = 0; col < 16; ++col)
                                {
                                    const float dot = c_tile[row * 16 + col];
                                    out[col] = completeEpilogueValue(dot, alpha, beta, out + col);
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
