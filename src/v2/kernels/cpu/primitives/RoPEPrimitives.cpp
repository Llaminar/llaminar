/**
 * @file RoPEPrimitives.cpp
 * @brief Native CPU rotary embeddings with absolute-position angle ownership.
 *
 * Prefill, prefix-restored suffixes and grouped/ordinary decode share the same
 * absolute-position coefficient authority, never a history-dependent recurrence.
 * Native floating pairs retain explicit rounding; quantized block cycles consume
 * both original halves before replacing their scale/payload. One head workshare
 * owns one-row decode even inside retained teams, with no repeated bank rotation.
 *
 * Angle rows come from one function (fill_rope_angle_row) and live only in the
 * caller's borrowed planes (RoPEAngleRows, the stage's admitted cpu_rope_angles
 * workspace): no per-call vectors, no thread-local tables and no decode memo.
 * Inside an existing team every member shares one omp-for fill whose implicit
 * barrier publishes the planes before any head reads them. The Q16/hybrid
 * family keeps its own Q15 angle authority and storage (tracked follow-up A).
 * @author David Sanftenberg
 */

#include "RoPEPrimitives.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/Logger.h"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <array>
#include <atomic>
#include <type_traits>
#include <limits>

#include <omp.h>

#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/CPUFeatures.h"
#if LLAMINAR_COMPILED_WITH_NEON
#include "NEONRoPEPrimitives.h"
#include "NEONQuantizedRoPE.h"
#include "NEONIntegerRoPE.h"
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives
{

    // ============================================================================
    // Inverse Frequency Cache
    // ============================================================================

    namespace
    {
        /// Process-lifetime table; entries are never erased, and unordered_map
        /// nodes keep their addresses across rehashing.
        static std::unordered_map<uint64_t, std::vector<float>> g_inv_freq_cache;
        static std::mutex g_inv_freq_mutex;

        uint64_t make_cache_key(int head_dim, float freq_base)
        {
            uint32_t freq_bits;
            std::memcpy(&freq_bits, &freq_base, sizeof(float));
            return ((uint64_t)head_dim << 32) | freq_bits;
        }

        /**
         * @brief Find or publish one geometry's inverse frequencies under the table lock.
         * @param key Exact (head_dim, freq_base bits) identity.
         * @param head_dim Rotary width.
         * @param freq_base RoPE base.
         * @return Immutable values with process lifetime.
         */
        const std::vector<float> &publish_inv_freq(uint64_t key, int head_dim, float freq_base)
        {
            {
                std::lock_guard<std::mutex> lock(g_inv_freq_mutex);
                auto it = g_inv_freq_cache.find(key);
                if (it != g_inv_freq_cache.end())
                {
                    return it->second;
                }
            }

            // Compute inverse frequencies
            const int half_dim = head_dim / 2;
            std::vector<float> inv_freq(half_dim);
            const float log_base = std::log(freq_base);

            for (int i = 0; i < half_dim; ++i)
            {
                float exponent = (2.0f * i) / head_dim;
                inv_freq[i] = std::exp(-log_base * exponent);
            }

            std::lock_guard<std::mutex> lock(g_inv_freq_mutex);
            auto res = g_inv_freq_cache.emplace(key, std::move(inv_freq));
            return res.first->second;
        }
    } // anonymous namespace

    const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base)
    {
        const uint64_t key = make_cache_key(head_dim, freq_base);

        // Every RoPE call, and every worker of a retained team, asks for the same
        // few model-constant geometries. A per-thread memo of published node
        // addresses answers those hits without contending on the table lock.
        struct Memo
        {
            uint64_t key = 0;
            const std::vector<float> *values = nullptr;
        };
        thread_local std::array<Memo, 8> memo{};
        thread_local std::size_t next_slot = 0;
        for (const Memo &entry : memo)
        {
            if (entry.values && entry.key == key)
                return *entry.values;
        }

        const std::vector<float> &values = publish_inv_freq(key, head_dim, freq_base);
        memo[next_slot] = Memo{key, &values};
        next_slot = (next_slot + 1) % memo.size();
        return values;
    }

    // ============================================================================
    // Vectorized RoPE Application - Separated Implementations
    // ============================================================================

#if LLAMINAR_COMPILED_WITH_NEON
    /** @brief Rotate an original retained head through the checked native pair leaf.
     * @tparam Format Actual original floating storage, never inferred from pointer width.
     * @param head Original writable head.
     * @param cosine Borrowed absolute-position cosine words.
     * @param sine Matching sine words.
     * @param head_dim Complete nonnegative even rotary width.
     */
    template <neon_rope::Storage Format>
    static void apply_rope_to_head_cached_neon(
        neon_rope::Word<Format> *head, const float *cosine, const float *sine, int head_dim)
    { neon_rope::rotate_head<Format>(head, cosine, sine, head_dim); }

    /** @brief Generate bounded absolute-position coefficients and rotate native lanes.
     * @tparam Format Original retained floating storage.
     * @param head Original in-place head, retained until return.
     * @param position Absolute position, not a recurrence seed.
     * @param inv_freq Immutable original coefficient frequencies.
     * @param head_dim Nonnegative even original rotary width.
     * @return Complete pair count, including the identically computed native tail.
     * @throws std::invalid_argument Before mutation on malformed head/frequency geometry.
     */
    template <neon_rope::Storage Format>
    static int apply_rope_to_head_neon(
        neon_rope::Word<Format> *head, int position,
        const std::vector<float> &inv_freq, int head_dim)
    {
        if (head_dim < 0 || head_dim % 2 != 0 ||
            inv_freq.size() < static_cast<std::size_t>(head_dim / 2) || (head_dim && !head))
            throw std::invalid_argument("Native direct RoPE requires complete head and frequency geometry");
        const int half = head_dim / 2;
        for (int first = 0; first < half;)
        {
            const int count = std::min(4, half - first);
            float cosine[4] = {}, sine[4] = {};
            // Trigonometry follows the existing CPU coefficient authority. Only
            // four pairs are private; there is no expanded head/tensor mirror.
            for (int lane = 0; lane < count; ++lane)
            {
                const float angle = static_cast<float>(position) * inv_freq[static_cast<std::size_t>(first + lane)];
                cosine[lane] = std::cos(angle);
                sine[lane] = std::sin(angle);
            }
            neon_rope::rotate_pairs<Format>(head + first, head + half + first, cosine, sine, count);
            first += count;
        }
        return half;
    }
#endif

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation using cached sin/cos (AVX2)
     */
    void apply_rope_to_head_cached_avx2(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 8 <= half_dim; i += 8)
        {
            __m256 x_first = _mm256_loadu_ps(head_ptr + i);
            __m256 x_second = _mm256_loadu_ps(head_ptr + i + half_dim);
            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            _mm256_storeu_ps(head_ptr + i, new_first);
            _mm256_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];
            float cos_val = cos_cache[i];
            float sin_val = sin_cache[i];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation using cached sin/cos (AVX512)
     */
    void apply_rope_to_head_cached_avx512(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 16 <= half_dim; i += 16)
        {
            __m512 x_first = _mm512_loadu_ps(head_ptr + i);
            __m512 x_second = _mm512_loadu_ps(head_ptr + i + half_dim);
            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            _mm512_storeu_ps(head_ptr + i, new_first);
            _mm512_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];
            float cos_val = cos_cache[i];
            float sin_val = sin_cache[i];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }
#endif

    /**
     * @brief Apply RoPE rotation to a single head (scalar implementation)
     *
     * This is the reference implementation used for:
     * - Scalar-only builds
     * - Tail processing after vectorized loops
     * - Testing/validation
     */
    void apply_rope_to_head_scalar(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation to a single head (AVX2 implementation)
     *
     * Processes 8 float pairs at a time using AVX2 intrinsics
     *
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_avx2(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load and rotate
            __m256 x_first = _mm256_loadu_ps(head_ptr + i);
            __m256 x_second = _mm256_loadu_ps(head_ptr + i + half_dim);
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            _mm256_storeu_ps(head_ptr + i, new_first);
            _mm256_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation to a single head (AVX512 implementation)
     *
     * Processes 16 float pairs at a time using AVX512 intrinsics
     *
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_avx512(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load values
            __m512 x_first = _mm512_loadu_ps(head_ptr + i);
            __m512 x_second = _mm512_loadu_ps(head_ptr + i + half_dim);
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate: new_first = x_first * cos - x_second * sin
            //         new_second = x_first * sin + x_second * cos
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            _mm512_storeu_ps(head_ptr + i, new_first);
            _mm512_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        return i;
    }
#endif

    void fill_rope_angle_row(const std::vector<float> &inv_freq, int position, float *cosine, float *sine)
    {
        // The multiplication rounds to FP32 before libm evaluates either
        // function, and cosine/sine stay separate calls: a merged sincos is a
        // different libm program. Every RoPE path computes a row only here, so
        // serial, decode, verifier-row and prefill angles agree by construction.
        for (size_t index = 0; index < inv_freq.size(); ++index)
        {
            const float angle = static_cast<float>(position) * inv_freq[index];
            cosine[index] = std::cos(angle);
            sine[index] = std::sin(angle);
        }
    }

    namespace
    {
        /**
         * @brief Which rows a consumer rotates, and therefore which rows the fill publishes.
         *
         * The policy belongs to the consumer's loop, not to the position value:
         * a consumer that rotates every row must receive angles for every row,
         * including a negative absolute position, exactly as the original
         * per-call tables computed them.
         */
        enum class RoPEPaddingRows
        {
            VerifierPadding, ///< Grouped verifier rows: a negative id is padding; its row stays
                             ///< inactive and the consumer skips it by its active flag.
            EveryPosition,   ///< Contiguous ordinary rows and the Q8_1 integer program rotate
                             ///< every row by its stored position, negative or not.
        };

        /**
         * @brief Fill the borrowed angle planes for every live row once.
         * @tparam WithQ15 Also publish the Q8_1 format's truncated Q15 rows.
         * @tparam PositionOf Callable mapping a row to its absolute position.
         * @param inv_freq Immutable inverse frequencies; their count is the row width.
         * @param angles Borrowed planes covering rows x inv_freq.size().
         * @param rows Positive live rows.
         * @param padding Rows the calling consumer rotates (see RoPEPaddingRows).
         * @param position_of Row-to-position map, evaluated once per row.
         *
         * Rows are independent functions of their absolute position, so any
         * partition of the rows stores identical bytes. Inside an existing team
         * every member reaches this single omp-for; its implicit barrier (no
         * nowait) publishes the complete planes before the head workshare reads
         * them, and no member skips the fill. Standalone calls with one row fill
         * it on the caller with the same row function; more rows get a team.
         */
        template <bool WithQ15, typename PositionOf>
        void fill_shared_angle_rows(
            const std::vector<float> &inv_freq, const RoPEAngleRows &angles,
            int rows, RoPEPaddingRows padding, PositionOf position_of)
        {
            const int values = static_cast<int>(inv_freq.size());
            require_rope_angle_rows(angles, rows, values);
            const auto fill_row = [&](int row)
            {
                const int position = position_of(row);
                const bool active = position >= 0 || padding == RoPEPaddingRows::EveryPosition;
                angles.active[static_cast<size_t>(row)] = active ? 1 : 0;
                if (!active)
                    return;
                float *cosine = angles.cos_row(row);
                float *sine = angles.sin_row(row);
                fill_rope_angle_row(inv_freq, position, cosine, sine);
                if constexpr (WithQ15)
                {
                    // The Q8 integer authority truncates the same FP32 values.
                    int16_t *cos_q15 = angles.cos_q15_row(row);
                    int16_t *sin_q15 = angles.sin_q15_row(row);
                    for (int index = 0; index < values; ++index)
                    {
                        cos_q15[index] = static_cast<int16_t>(cosine[index] * 32767.0f);
                        sin_q15[index] = static_cast<int16_t>(sine[index] * 32767.0f);
                    }
                }
            };
            auto fill = [&]()
            {
#pragma omp for schedule(static)
                for (int row = 0; row < rows; ++row)
                    fill_row(row);
            };
            if (::llaminar::v2::inLexicalWorkshare() || rows > 1)
            {
                OMP_WORKSHARE_REGION(fill);
            }
            else
            {
                fill_row(0);
            }
        }
    } // namespace

    /**
     * @brief Share angle preparation and one head workshare across Q and K.
     * @param q Native query rows, modified in place.
     * @param k Optional native key rows, modified in place.
     * @param rows Number of live contiguous positions.
     * @param q_heads Query heads in each row.
     * @param k_heads Key heads in each row; ignored when k is null.
     * @param head_dim Complete physical head width.
     * @param first_position Absolute start of the input rows.
     * @param inv_freq Immutable inverse frequencies for these heads.
     * @param angles Borrowed planes for rows x head_dim/2 pairs.
     * @param rotate Native single-head SIMD operation, preserving storage bytes.
     *
     * Q and K have identical angle geometry even under GQA, so one angle fill
     * serves both. Each worker owns disjoint output heads; no head reduction
     * order changes.
     */
    template <typename Native, typename Rotate>
    static void apply_rope_pair_positions(
        Native *q, Native *k, int rows, int q_heads, int k_heads, int head_dim,
        int first_position, const std::vector<float> &inv_freq, const RoPEAngleRows &angles, Rotate rotate)
    {
        const int heads = q_heads + (k ? k_heads : 0);
        fill_shared_angle_rows<false>(inv_freq, angles, rows, RoPEPaddingRows::EveryPosition,
            [first_position](int row) { return first_position + row; });
        auto work = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < rows; ++row)
                for (int head = 0; head < heads; ++head)
                {
                    Native *data = head < q_heads
                        ? q + (static_cast<size_t>(row) * q_heads + head) * head_dim
                        : k + (static_cast<size_t>(row) * k_heads + head - q_heads) * head_dim;
                    rotate(data, angles.cos_row(row), angles.sin_row(row));
                }
        };
        OMP_WORKSHARE_REGION(work);
    }

    /** @brief Rotate FP32 heads using shared absolute-position angle rows. */
    static void apply_rope_to_tensor_positions(
        float *q, float *k,
        int seq_len, int q_heads, int k_heads, int head_dim,
        int n_past, const std::vector<float> &inv_freq, const RoPEAngleRows &angles)
    {
        const auto rotate = [head_dim](float *head_ptr, const float *cos_ptr, const float *sin_ptr)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::FP32>(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_avx512(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_avx2(head_ptr, cos_ptr, sin_ptr, head_dim);
#else
            const int half_dim = head_dim / 2;
            // Explicit scalar diagnostics preserve their fixed rotation arithmetic.
            for (int i = 0; i < half_dim; ++i)
            {
                float x_first = head_ptr[i];
                float x_second = head_ptr[i + half_dim];
                float cos_val = cos_ptr[i];
                float sin_val = sin_ptr[i];

                head_ptr[i] = x_first * cos_val - x_second * sin_val;
                head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
            }
#endif
        };
        apply_rope_pair_positions(q, k, seq_len, q_heads, k_heads, head_dim, n_past, inv_freq, angles, rotate);
    }

    /**
     * @brief Own each one-row query/key head exactly once in standalone or retained execution.
     * @tparam Native Original floating storage word.
     * @tparam Rotate Architecture-native single-head operation, not a whole-bank replay.
     * @param q Original writable query bank.
     * @param k Optional original writable key bank, disjoint from q.
     * @param q_heads Query head count.
     * @param k_heads Key head count when k is present.
     * @param head_dim Nonnegative even physical head width.
     * @param cosine Completed shared row of head_dim/2 cosines.
     * @param sine Matching shared row of sines.
     * @param rotate One original-head rotation with borrowed coefficient pointers.
     * @throws std::invalid_argument Before mutation on incomplete geometry or writable aliases.
     *
     * Retained-team callers share the output banks. Repeating a serial head
     * loop per member would transform those banks multiple times. One head
     * workshare and its final edge own publication, including no heads.
     */
    template <typename Native, typename Rotate>
    static void apply_rope_row_heads(
        Native *q, Native *k, int q_heads, int k_heads, int head_dim,
        const float *cosine, const float *sine, Rotate rotate)
    {
        if (q_heads < 0 || k_heads < 0 || head_dim < 0 || head_dim % 2 != 0 ||
            (q_heads && head_dim && !q) || !cosine || !sine)
            throw std::invalid_argument("One-row CPU RoPE requires complete head and angle geometry");
        const auto q_bytes = std::size_t(q_heads) * head_dim * sizeof(Native);
        const auto k_bytes = k ? std::size_t(k_heads) * head_dim * sizeof(Native) : 0;
        const auto coefficient_bytes = std::size_t(head_dim / 2) * sizeof(float);
        const auto reject_overlap = [](const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes) {
            const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
            if (a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes))
                throw std::invalid_argument("One-row CPU RoPE writable banks alias another retained authority");
        };
        reject_overlap(q, q_bytes, k, k_bytes);
        for (const float *coefficients : {cosine, sine})
        {
            reject_overlap(q, q_bytes, coefficients, coefficient_bytes);
            reject_overlap(k, k_bytes, coefficients, coefficient_bytes);
        }
        const std::int64_t heads = head_dim ? std::int64_t(q_heads) + (k ? k_heads : 0) : 0;
        auto work = [&]()
        {
#pragma omp for schedule(static)
            for (std::int64_t head = 0; head < heads; ++head)
            {
                Native *data = head < q_heads
                    ? q + static_cast<std::size_t>(head) * head_dim
                    : k + static_cast<std::size_t>(head - q_heads) * head_dim;
                rotate(data, cosine, sine);
            }
        };
        // The conditional macro creates a private one-thread region when its
        // condition is false. Preserve the active outer team explicitly; the
        // size hint controls only standalone team creation, never membership.
        OMP_WORKSHARE_REGION_IF(work, omp_in_parallel() || heads >= 4);
    }

    /** @brief Rotate one FP32 row's heads through the shared participant workshare.
     * @param q Original query bank.
     * @param k Optional original key bank.
     * @param q_heads Query head count.
     * @param k_heads Key head count when k is present.
     * @param head_dim Original even head width.
     * @param cosine Completed shared cosine row.
     * @param sine Completed shared sine row.
     */
    static void apply_rope_row_fp32(
        float *q, float *k,
        int q_heads, int k_heads, int head_dim,
        const float *cosine, const float *sine)
    {
        const auto rotate = [head_dim](float *head, const float *c, const float *s)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::FP32>(head, c, s, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_avx512(head, c, s, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_avx2(head, c, s, head_dim);
#else
            const int half = head_dim / 2;
            for (int i = 0; i < half; ++i)
            {
                const float x = head[i], y = head[i + half];
                head[i] = x * c[i] - y * s[i];
                head[i + half] = x * s[i] + y * c[i];
            }
#endif
        };
        apply_rope_row_heads(q, k, q_heads, k_heads, head_dim, cosine, sine, rotate);
    }

    // ============================================================================
    // Public API
    // ============================================================================

    void apply_rope_vectorized(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        const RoPEAngleRows &angles)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        if (seq_len == 1)
        {
            // One shared row; every head reads it through one head workshare.
            fill_shared_angle_rows<false>(inv_freq, angles, 1, RoPEPaddingRows::EveryPosition,
                [n_past](int) { return n_past; });
            apply_rope_row_fp32(q, k, q_heads, k_heads, head_dim, angles.cos_row(0), angles.sin_row(0));
        }
        else
        {
            // Prefill uses the same position-owned angles as ordinary decode.
            apply_rope_to_tensor_positions(q, k, seq_len, q_heads, k_heads, head_dim, n_past, inv_freq, angles);
        }
    }

    // ============================================================================
    // Partial RoPE Implementation (partial_rotary_factor < 1.0)
    // ============================================================================

    namespace detail
    {
        void rope_rotate_head_scalar(float *head_ptr, const float *cos_ptr,
                                     const float *sin_ptr, int half_rotary)
        {
            for (int i = 0; i < half_rotary; ++i)
            {
                float x0 = head_ptr[i], x1 = head_ptr[i + half_rotary];
                head_ptr[i] = x0 * cos_ptr[i] - x1 * sin_ptr[i];
                head_ptr[i + half_rotary] = x0 * sin_ptr[i] + x1 * cos_ptr[i];
            }
        }

#if defined(__AVX2__)
        void rope_rotate_head_avx2(float *head_ptr, const float *cos_ptr,
                                   const float *sin_ptr, int half_rotary)
        {
            int i = 0;
            for (; i + 8 <= half_rotary; i += 8)
            {
                __m256 x0 = _mm256_loadu_ps(head_ptr + i);
                __m256 x1 = _mm256_loadu_ps(head_ptr + i + half_rotary);
                __m256 cv = _mm256_loadu_ps(cos_ptr + i);
                __m256 sv = _mm256_loadu_ps(sin_ptr + i);
                _mm256_storeu_ps(head_ptr + i,
                                 _mm256_sub_ps(_mm256_mul_ps(x0, cv), _mm256_mul_ps(x1, sv)));
                _mm256_storeu_ps(head_ptr + i + half_rotary,
                                 _mm256_add_ps(_mm256_mul_ps(x0, sv), _mm256_mul_ps(x1, cv)));
            }
            for (; i < half_rotary; ++i)
            {
                float x0 = head_ptr[i], x1 = head_ptr[i + half_rotary];
                head_ptr[i] = x0 * cos_ptr[i] - x1 * sin_ptr[i];
                head_ptr[i + half_rotary] = x0 * sin_ptr[i] + x1 * cos_ptr[i];
            }
        }
#endif

#if defined(__AVX512F__)
        void rope_rotate_head_avx512(float *head_ptr, const float *cos_ptr,
                                     const float *sin_ptr, int half_rotary)
        {
            int i = 0;
            for (; i + 16 <= half_rotary; i += 16)
            {
                __m512 x0 = _mm512_loadu_ps(head_ptr + i);
                __m512 x1 = _mm512_loadu_ps(head_ptr + i + half_rotary);
                __m512 cv = _mm512_loadu_ps(cos_ptr + i);
                __m512 sv = _mm512_loadu_ps(sin_ptr + i);
                _mm512_storeu_ps(head_ptr + i,
                                 _mm512_sub_ps(_mm512_mul_ps(x0, cv), _mm512_mul_ps(x1, sv)));
                _mm512_storeu_ps(head_ptr + i + half_rotary,
                                 _mm512_add_ps(_mm512_mul_ps(x0, sv), _mm512_mul_ps(x1, cv)));
            }
            for (; i < half_rotary; ++i)
            {
                float x0 = head_ptr[i], x1 = head_ptr[i + half_rotary];
                head_ptr[i] = x0 * cos_ptr[i] - x1 * sin_ptr[i];
                head_ptr[i + half_rotary] = x0 * sin_ptr[i] + x1 * cos_ptr[i];
            }
        }
#endif

#if LLAMINAR_COMPILED_WITH_NEON
        /** @brief Rotate the original partial FP32 interval with the native four-lane program.
         * @param head_ptr Original first rotary word.
         * @param cos_ptr Original cosine coefficients.
         * @param sin_ptr Matching sine coefficients.
         * @param half_rotary Half of the complete even rotary interval.
         */
        void rope_rotate_head_neon(float *head_ptr, const float *cos_ptr, const float *sin_ptr, int half_rotary)
        {
            if (half_rotary < 0 || half_rotary > std::numeric_limits<int>::max() / 2)
                throw std::invalid_argument("Native partial RoPE pair count is not representable");
            neon_rope::rotate_head<neon_rope::Storage::FP32>(head_ptr, cos_ptr, sin_ptr, half_rotary * 2);
        }
#endif

// --- ISA stubs for non-ARM diagnostic builds ---
#if !defined(__AVX2__) && !LLAMINAR_COMPILED_WITH_NEON
        void rope_rotate_head_avx2(float *head_ptr, const float *cos_ptr,
                                   const float *sin_ptr, int half_rotary)
        {
            rope_rotate_head_scalar(head_ptr, cos_ptr, sin_ptr, half_rotary);
        }
#endif
#if !defined(__AVX512F__) && !LLAMINAR_COMPILED_WITH_NEON
        void rope_rotate_head_avx512(float *head_ptr, const float *cos_ptr,
                                     const float *sin_ptr, int half_rotary)
        {
            rope_rotate_head_avx2(head_ptr, cos_ptr, sin_ptr, half_rotary);
        }
#endif

        inline void rope_rotate_head_dispatch(float *head_ptr, const float *cos_ptr,
                                              const float *sin_ptr, int half_rotary)
        {
            ISA_DISPATCH_VOID(rope_rotate_head, head_ptr, cos_ptr, sin_ptr, half_rotary);
        }
    } // namespace detail
    using namespace detail;

    void prepare_rope_single_position_angles(
        int rotary_dim,
        int position,
        float freq_base,
        const RoPEAngleRows &angles)
    {
        if (rotary_dim <= 0 || (rotary_dim % 2) != 0)
            return;
        const auto &inv_freq = get_inv_freq_cached(rotary_dim, freq_base);
        fill_rope_angle_row(inv_freq, position, angles.cos_row(0), angles.sin_row(0));
    }

    void rotate_rope_head_in_place(
        float *head_ptr,
        const float *cos_ptr,
        const float *sin_ptr,
        int half_rotary)
    {
        if (!head_ptr || !cos_ptr || !sin_ptr || half_rotary <= 0)
            return;
        rope_rotate_head_dispatch(head_ptr, cos_ptr, sin_ptr, half_rotary);
    }

    void apply_rope_partial(
        float *q, float *k,
        int seq_len, int head_dim, int rotary_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        const RoPEAngleRows &angles)
    {
        if (rotary_dim <= 0 || rotary_dim % 2 != 0)
            return;

        // Frequencies computed using rotary_dim (NOT head_dim)
        const auto &inv_freq = get_inv_freq_cached(rotary_dim, freq_base);
        const int half_rotary = rotary_dim / 2;

        // Partial rotation shares the same absolute-position angle authority.
        fill_shared_angle_rows<false>(inv_freq, angles, seq_len, RoPEPaddingRows::EveryPosition,
            [n_past](int row) { return n_past + row; });

        // Apply rotation (head_dim for stride, rotary_dim for loop)
        auto do_partial_rope = [&]()
        {
        // Q heads
#pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < q_heads; ++h)
                {
                    float *head_ptr = q + (static_cast<size_t>(t) * q_heads + h) * head_dim;
                    const float *cos_ptr = angles.cos_row(t);
                    const float *sin_ptr = angles.sin_row(t);
                    rope_rotate_head_dispatch(head_ptr, cos_ptr, sin_ptr, half_rotary);
                }
            }

            // K heads (same logic, different head count)
            if (k)
            {
#pragma omp for collapse(2) schedule(static)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < k_heads; ++h)
                    {
                        float *head_ptr = k + (static_cast<size_t>(t) * k_heads + h) * head_dim;
                        const float *cos_ptr = angles.cos_row(t);
                        const float *sin_ptr = angles.sin_row(t);
                        rope_rotate_head_dispatch(head_ptr, cos_ptr, sin_ptr, half_rotary);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_partial_rope);
    }

    namespace
    {
        /**
         * @brief Materialize exactly the angle rows produced by serial decode.
         * @tparam WithQ15 Also publish the Q8_1 format's truncated Q15 rows.
         * @param position_ids Optional absolute position per row; null means pos_offset + row.
         * @param verifier_rows Positive grouped rows.
         * @param head_dim Physical head width.
         * @param rotary_dim Rotated width; 0 or head_dim means full RoPE.
         * @param pos_offset Contiguous first position when position_ids is null.
         * @param freq_base RoPE frequency base.
         * @param angles Borrowed planes covering verifier_rows x rotary pairs.
         *
         * Full and partial RoPE compute each absolute-position angle directly,
         * including non-contiguous jumps and padding rows, through the same row
         * function as serial decode. One grouped row/head workshare then
         * consumes the planes without changing arithmetic.
         */
        template <bool WithQ15>
        void prepare_verifier_angle_rows(
            const int *position_ids,
            int verifier_rows,
            int head_dim,
            int rotary_dim,
            int pos_offset,
            float freq_base,
            const RoPEAngleRows &angles)
        {
            const int effective_rotary =
                rotary_dim > 0 && rotary_dim < head_dim ? rotary_dim : head_dim;
            const auto &inv_freq = get_inv_freq_cached(effective_rotary, freq_base);
            fill_shared_angle_rows<WithQ15>(inv_freq, angles, verifier_rows, RoPEPaddingRows::VerifierPadding,
                [position_ids, pos_offset](int row)
                { return position_ids ? position_ids[row] : pos_offset + row; });
        }
    } // namespace

    void apply_rope_decode_equivalent_rows(
        float *q, float *k,
        const int *position_ids,
        int verifier_rows,
        int head_dim, int rotary_dim,
        int q_heads, int k_heads,
        int pos_offset, float freq_base,
        const RoPEAngleRows &angles)
    {
        if (!q || verifier_rows <= 0 || head_dim <= 0 || (head_dim % 2) != 0)
            return;

        const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
        if (eff_rotary <= 0 || (eff_rotary % 2) != 0)
            return;

        prepare_verifier_angle_rows<false>(
            position_ids, verifier_rows, head_dim, rotary_dim,
            pos_offset, freq_base, angles);
        const bool full_rope = eff_rotary == head_dim;
        const int q_row_stride = q_heads * head_dim;
        const int k_row_stride = k_heads * head_dim;

        auto apply_full_head = [&](float *head,
                                   const float *cos_values,
                                   const float *sin_values)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::FP32>(head, cos_values, sin_values, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_avx512(
                head, cos_values, sin_values, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_avx2(
                head, cos_values, sin_values, head_dim);
#else
            const int half_dim = head_dim / 2;
            for (int index = 0; index < half_dim; ++index)
            {
                const float first = head[index];
                const float second = head[index + half_dim];
                head[index] = first * cos_values[index] - second * sin_values[index];
                head[index + half_dim] =
                    first * sin_values[index] + second * cos_values[index];
            }
#endif
        };

        auto apply_grouped_rows = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < verifier_rows; ++row)
            {
                for (int head_index = 0; head_index < q_heads; ++head_index)
                {
                    if (!angles.active[static_cast<size_t>(row)])
                        continue;
                    float *head = q + static_cast<size_t>(row) * q_row_stride +
                                  static_cast<size_t>(head_index) * head_dim;
                    const float *row_cos = angles.cos_row(row);
                    const float *row_sin = angles.sin_row(row);
                    if (full_rope)
                        apply_full_head(head, row_cos, row_sin);
                    else
                        rope_rotate_head_dispatch(
                            head, row_cos, row_sin, angles.values_per_row);
                }
            }

            if (k)
            {
#pragma omp for collapse(2) schedule(static)
                for (int row = 0; row < verifier_rows; ++row)
                {
                    for (int head_index = 0; head_index < k_heads; ++head_index)
                    {
                        if (!angles.active[static_cast<size_t>(row)])
                            continue;
                        float *head = k + static_cast<size_t>(row) * k_row_stride +
                                      static_cast<size_t>(head_index) * head_dim;
                        const float *row_cos = angles.cos_row(row);
                        const float *row_sin = angles.sin_row(row);
                        if (full_rope)
                            apply_full_head(head, row_cos, row_sin);
                        else
                            rope_rotate_head_dispatch(
                                head, row_cos, row_sin, angles.values_per_row);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(apply_grouped_rows);
    }

    // ============================================================================
    // BF16 Native Precision Implementations
    // ============================================================================

#if defined(__AVX512F__)
    void apply_rope_to_head_cached_bf16_avx512(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 16 <= half_dim; i += 16)
        {
            // Load BF16
            __m256i bf16_first = _mm256_loadu_si256((__m256i *)(head_ptr + i));
            __m256i bf16_second = _mm256_loadu_si256((__m256i *)(head_ptr + i + half_dim));

            // Convert to FP32 (shift left 16)
            __m512i u32_first = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_first), 16);
            __m512i u32_second = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_second), 16);

            __m512 x_first = _mm512_castsi512_ps(u32_first);
            __m512 x_second = _mm512_castsi512_ps(u32_second);

            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            // Convert back to BF16 (shift right 16)
            __m512i res_first = _mm512_srli_epi32(_mm512_castps_si512(new_first), 16);
            __m512i res_second = _mm512_srli_epi32(_mm512_castps_si512(new_second), 16);

            // Pack 32-bit integers to 16-bit integers
            __m256i out_first = _mm512_cvtepi32_epi16(res_first);
            __m256i out_second = _mm512_cvtepi32_epi16(res_second);

            _mm256_storeu_si256((__m256i *)(head_ptr + i), out_first);
            _mm256_storeu_si256((__m256i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::bf16_to_fp32(head_ptr[i]);
            float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_bf16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_bf16(n2);
        }
    }
#endif

#if defined(__AVX2__)
    void apply_rope_to_head_cached_bf16_avx2(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 8 <= half_dim; i += 8)
        {
            // Load 8 BF16 values (128 bits)
            __m128i bf16_first = _mm_loadu_si128((__m128i *)(head_ptr + i));
            __m128i bf16_second = _mm_loadu_si128((__m128i *)(head_ptr + i + half_dim));

            // Convert to FP32 (expand to 32-bit, shift left 16)
            __m256i u32_first = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_first), 16);
            __m256i u32_second = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_second), 16);

            __m256 x_first = _mm256_castsi256_ps(u32_first);
            __m256 x_second = _mm256_castsi256_ps(u32_second);

            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            // Convert back (shift right 16)
            __m256i res_first = _mm256_srli_epi32(_mm256_castps_si256(new_first), 16);
            __m256i res_second = _mm256_srli_epi32(_mm256_castps_si256(new_second), 16);

            // Pack using packus and permute
            __m256i packed_first = _mm256_packus_epi32(res_first, _mm256_setzero_si256());
            __m256i permuted_first = _mm256_permute4x64_epi64(packed_first, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_first = _mm256_castsi256_si128(permuted_first);

            __m256i packed_second = _mm256_packus_epi32(res_second, _mm256_setzero_si256());
            __m256i permuted_second = _mm256_permute4x64_epi64(packed_second, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_second = _mm256_castsi256_si128(permuted_second);

            _mm_storeu_si128((__m128i *)(head_ptr + i), out_first);
            _mm_storeu_si128((__m128i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::bf16_to_fp32(head_ptr[i]);
            float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_bf16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_bf16(n2);
        }
    }
#endif

    void apply_rope_to_head_bf16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            // Convert BF16 → FP32
            float x_first = simd::bf16_to_fp32(head_ptr[i]);
            float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);

            // Rotate
            float new_first = x_first * cos_val - x_second * sin_val;
            float new_second = x_first * sin_val + x_second * cos_val;

            // Convert FP32 → BF16
            head_ptr[i] = simd::fp32_to_bf16(new_first);
            head_ptr[i + half_dim] = simd::fp32_to_bf16(new_second);
        }
    }

#if defined(__AVX2__)
    int apply_rope_to_head_bf16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load 8 BF16 values (128 bits)
            __m128i bf16_first = _mm_loadu_si128((__m128i *)(head_ptr + i));
            __m128i bf16_second = _mm_loadu_si128((__m128i *)(head_ptr + i + half_dim));

            // Convert to FP32 (expand to 32-bit, shift left 16)
            __m256i u32_first = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_first), 16);
            __m256i u32_second = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_second), 16);

            __m256 x_first_vec = _mm256_castsi256_ps(u32_first);
            __m256 x_second_vec = _mm256_castsi256_ps(u32_second);

            // Load sin/cos
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            // Rotate
            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first_vec, cos_vec),
                _mm256_mul_ps(x_second_vec, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first_vec, sin_vec),
                _mm256_mul_ps(x_second_vec, cos_vec));

            // Convert back (shift right 16)
            __m256i res_first = _mm256_srli_epi32(_mm256_castps_si256(new_first), 16);
            __m256i res_second = _mm256_srli_epi32(_mm256_castps_si256(new_second), 16);

            // Pack using packus and permute
            __m256i packed_first = _mm256_packus_epi32(res_first, _mm256_setzero_si256());
            __m256i permuted_first = _mm256_permute4x64_epi64(packed_first, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_first = _mm256_castsi256_si128(permuted_first);

            __m256i packed_second = _mm256_packus_epi32(res_second, _mm256_setzero_si256());
            __m256i permuted_second = _mm256_permute4x64_epi64(packed_second, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i out_second = _mm256_castsi256_si128(permuted_second);

            // Store back
            _mm_storeu_si128((__m128i *)(head_ptr + i), out_first);
            _mm_storeu_si128((__m128i *)(head_ptr + i + half_dim), out_second);
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    int apply_rope_to_head_bf16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load BF16 values (16 values = 256 bits)
            __m256i bf16_first = _mm256_loadu_si256((__m256i *)(head_ptr + i));
            __m256i bf16_second = _mm256_loadu_si256((__m256i *)(head_ptr + i + half_dim));

            // Convert BF16 → FP32 (vectorized)
            // Shift left by 16 to zero-extend mantissa
            __m512i u32_first = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_first), 16);
            __m512i u32_second = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_second), 16);

            __m512 x_first = _mm512_castsi512_ps(u32_first);
            __m512 x_second = _mm512_castsi512_ps(u32_second);

            // Load sin/cos
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            // Convert FP32 → BF16 (vectorized truncation)
            // Shift right by 16
            __m512i res_first = _mm512_srli_epi32(_mm512_castps_si512(new_first), 16);
            __m512i res_second = _mm512_srli_epi32(_mm512_castps_si512(new_second), 16);

            // Pack 32-bit integers to 16-bit integers
            __m256i out_first = _mm512_cvtepi32_epi16(res_first);
            __m256i out_second = _mm512_cvtepi32_epi16(res_second);

            // Store back
            _mm256_storeu_si256((__m256i *)(head_ptr + i), out_first);
            _mm256_storeu_si256((__m256i *)(head_ptr + i + half_dim), out_second);
        }

        return i;
    }
#endif

    static void apply_rope_to_head_bf16_vectorized(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if LLAMINAR_COMPILED_WITH_NEON
        processed = apply_rope_to_head_neon<neon_rope::Storage::BFloat16>(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX512F__)
        processed = apply_rope_to_head_bf16_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_bf16_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_bf16_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    /** @brief Rotate BF16 heads using shared absolute-position angle rows. */
    static void apply_rope_to_tensor_bf16_positions(
        uint16_t *q, uint16_t *k,
        int seq_len, int q_heads, int k_heads, int head_dim,
        int n_past, const std::vector<float> &inv_freq, const RoPEAngleRows &angles)
    {
        const auto rotate = [head_dim](uint16_t *head_ptr, const float *cos_ptr, const float *sin_ptr)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::BFloat16>(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_bf16_avx512(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_bf16_avx2(head_ptr, cos_ptr, sin_ptr, head_dim);
#else
            const int half_dim = head_dim / 2;
            // Explicit scalar diagnostics preserve their fixed rotation arithmetic.
            for (int i = 0; i < half_dim; ++i)
            {
                float x_first = simd::bf16_to_fp32(head_ptr[i]);
                float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);
                float cos_val = cos_ptr[i];
                float sin_val = sin_ptr[i];

                float n1 = x_first * cos_val - x_second * sin_val;
                float n2 = x_first * sin_val + x_second * cos_val;

                head_ptr[i] = simd::fp32_to_bf16(n1);
                head_ptr[i + half_dim] = simd::fp32_to_bf16(n2);
            }
#endif
        };
        apply_rope_pair_positions(q, k, seq_len, q_heads, k_heads, head_dim, n_past, inv_freq, angles, rotate);
    }

    /** @brief Rotate one row of BFloat16 heads once through the shared participant workshare.
     * @param q Original query bank.
     * @param k Optional original key bank.
     * @param q_heads Query head count.
     * @param k_heads Key head count when k is present.
     * @param head_dim Original even head width.
     * @param cosine Completed shared cosine row.
     * @param sine Completed shared sine row.
     */
    static void apply_rope_row_bf16(
        uint16_t *q, uint16_t *k,
        int q_heads, int k_heads, int head_dim,
        const float *cosine, const float *sine)
    {
        const auto rotate = [head_dim](uint16_t *head, const float *cosine, const float *sine)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::BFloat16>(head, cosine, sine, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_bf16_avx512(head, cosine, sine, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_bf16_avx2(head, cosine, sine, head_dim);
#else
            const int half = head_dim / 2;
            for (int i = 0; i < half; ++i)
            {
                const float x = simd::bf16_to_fp32(head[i]), y = simd::bf16_to_fp32(head[i + half]);
                head[i] = simd::fp32_to_bf16(x * cosine[i] - y * sine[i]);
                head[i + half] = simd::fp32_to_bf16(x * sine[i] + y * cosine[i]);
            }
#endif
        };
        apply_rope_row_heads(q, k, q_heads, k_heads, head_dim, cosine, sine, rotate);
    }

    static void apply_rope_to_tensor_bf16_direct(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        // Avoid OpenMP overhead for single-token decode
        if (seq_len == 1)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + h * head_dim;
                apply_rope_to_head_bf16_vectorized(head_ptr, n_past, inv_freq, head_dim);
            }
        }
        else
        {
            auto do_rope_direct_work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < num_heads; ++h)
                    {
                        uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                        apply_rope_to_head_bf16_vectorized(head_ptr, n_past + t, inv_freq, head_dim);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_rope_direct_work);
        }
    }

    void apply_rope_bf16(
        uint16_t *q_bf16, uint16_t *k_bf16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        const RoPEAngleRows &angles)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        if (seq_len == 1)
        {
            // One shared row; every head reads it through one head workshare.
            fill_shared_angle_rows<false>(inv_freq, angles, 1, RoPEPaddingRows::EveryPosition,
                [n_past](int) { return n_past; });
            apply_rope_row_bf16(q_bf16, k_bf16, q_heads, k_heads, head_dim, angles.cos_row(0), angles.sin_row(0));
            return;
        }

        // Short multi-row sequences keep their direct per-head program.
        if (seq_len < 32)
        {
            apply_rope_to_tensor_bf16_direct(q_bf16, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_bf16_direct(k_bf16, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
        else
        {
            // Share each absolute-position angle across the head workshare.
            apply_rope_to_tensor_bf16_positions(q_bf16, k_bf16, seq_len, q_heads, k_heads, head_dim, n_past, inv_freq, angles);
        }
    }

    void apply_rope_bf16_decode_equivalent_rows(
        uint16_t *q_bf16, uint16_t *k_bf16,
        const int *position_ids,
        int verifier_rows,
        int head_dim,
        int q_heads, int k_heads,
        int pos_offset, float freq_base,
        const RoPEAngleRows &angles)
    {
        if (!q_bf16 || verifier_rows <= 0 ||
            head_dim <= 0 || (head_dim % 2) != 0)
        {
            return;
        }

        prepare_verifier_angle_rows<false>(
            position_ids, verifier_rows, head_dim, 0,
            pos_offset, freq_base, angles);
        const int q_row_stride = q_heads * head_dim;
        const int k_row_stride = k_heads * head_dim;

        auto apply_head = [&](uint16_t *head,
                              const float *cos_values,
                              const float *sin_values)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::BFloat16>(head, cos_values, sin_values, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_bf16_avx512(
                head, cos_values, sin_values, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_bf16_avx2(
                head, cos_values, sin_values, head_dim);
#else
            const int half_dim = head_dim / 2;
            for (int index = 0; index < half_dim; ++index)
            {
                const float first = simd::bf16_to_fp32(head[index]);
                const float second = simd::bf16_to_fp32(head[index + half_dim]);
                const float rotated_first =
                    first * cos_values[index] - second * sin_values[index];
                const float rotated_second =
                    first * sin_values[index] + second * cos_values[index];
                head[index] = simd::fp32_to_bf16(rotated_first);
                head[index + half_dim] = simd::fp32_to_bf16(rotated_second);
            }
#endif
        };

        auto apply_grouped_rows = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < verifier_rows; ++row)
            {
                for (int head_index = 0; head_index < q_heads; ++head_index)
                {
                    if (!angles.active[static_cast<size_t>(row)])
                        continue;
                    uint16_t *head =
                        q_bf16 + static_cast<size_t>(row) * q_row_stride +
                        static_cast<size_t>(head_index) * head_dim;
                    const float *row_cos = angles.cos_row(row);
                    const float *row_sin = angles.sin_row(row);
                    apply_head(head, row_cos, row_sin);
                }
            }

            if (k_bf16)
            {
#pragma omp for collapse(2) schedule(static)
                for (int row = 0; row < verifier_rows; ++row)
                {
                    for (int head_index = 0; head_index < k_heads; ++head_index)
                    {
                        if (!angles.active[static_cast<size_t>(row)])
                            continue;
                        uint16_t *head =
                            k_bf16 + static_cast<size_t>(row) * k_row_stride +
                            static_cast<size_t>(head_index) * head_dim;
                        const float *row_cos = angles.cos_row(row);
                        const float *row_sin = angles.sin_row(row);
                        apply_head(head, row_cos, row_sin);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(apply_grouped_rows);
    }

    // ============================================================================
    // FP16 Native Precision Implementations
    // ============================================================================

#if defined(__AVX512F__)
    void apply_rope_to_head_cached_fp16_avx512(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 16 <= half_dim; i += 16)
        {
            // Load FP16 (16 values = 256 bits)
            __m256i fp16_first = _mm256_loadu_si256((__m256i *)(head_ptr + i));
            __m256i fp16_second = _mm256_loadu_si256((__m256i *)(head_ptr + i + half_dim));

            // Convert to FP32
            __m512 x_first = _mm512_cvtph_ps(fp16_first);
            __m512 x_second = _mm512_cvtph_ps(fp16_second);

            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            // Convert back to FP16
            __m256i out_first = _mm512_cvtps_ph(new_first, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i out_second = _mm512_cvtps_ph(new_second, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            _mm256_storeu_si256((__m256i *)(head_ptr + i), out_first);
            _mm256_storeu_si256((__m256i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::fp16_to_fp32(head_ptr[i]);
            float x_second = simd::fp16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_fp16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_fp16(n2);
        }
    }
#endif

#if defined(__AVX2__)
    void apply_rope_to_head_cached_fp16_avx2(
        uint16_t *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 8 <= half_dim; i += 8)
        {
            // Load FP16 (8 values = 128 bits)
            __m128i fp16_first = _mm_loadu_si128((__m128i *)(head_ptr + i));
            __m128i fp16_second = _mm_loadu_si128((__m128i *)(head_ptr + i + half_dim));

            // Convert to FP32
            __m256 x_first = _mm256_cvtph_ps(fp16_first);
            __m256 x_second = _mm256_cvtph_ps(fp16_second);

            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            // Convert back to FP16
            __m128i out_first = _mm256_cvtps_ph(new_first, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i out_second = _mm256_cvtps_ph(new_second, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            _mm_storeu_si128((__m128i *)(head_ptr + i), out_first);
            _mm_storeu_si128((__m128i *)(head_ptr + i + half_dim), out_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = simd::fp16_to_fp32(head_ptr[i]);
            float x_second = simd::fp16_to_fp32(head_ptr[i + half_dim]);
            float c = cos_cache[i];
            float s = sin_cache[i];
            float n1 = x_first * c - x_second * s;
            float n2 = x_first * s + x_second * c;
            head_ptr[i] = simd::fp32_to_fp16(n1);
            head_ptr[i + half_dim] = simd::fp32_to_fp16(n2);
        }
    }
#endif

    void apply_rope_to_head_fp16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            // Convert FP16 → FP32
            float x_first = fp16_to_fp32(head_ptr[i]);
            float x_second = fp16_to_fp32(head_ptr[i + half_dim]);

            // Rotate
            float new_first = x_first * cos_val - x_second * sin_val;
            float new_second = x_first * sin_val + x_second * cos_val;

            // Convert FP32 → FP16
            head_ptr[i] = fp32_to_fp16(new_first);
            head_ptr[i + half_dim] = fp32_to_fp16(new_second);
        }
    }

#if defined(__AVX2__)
    int apply_rope_to_head_fp16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load FP16 values
            alignas(32) uint16_t first_fp16[8];
            alignas(32) uint16_t second_fp16[8];
            std::memcpy(first_fp16, head_ptr + i, 8 * sizeof(uint16_t));
            std::memcpy(second_fp16, head_ptr + i + half_dim, 8 * sizeof(uint16_t));

            // Convert FP16 → FP32
            alignas(32) float x_first[8];
            alignas(32) float x_second[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                x_first[lane] = fp16_to_fp32(first_fp16[lane]);
                x_second[lane] = fp16_to_fp32(second_fp16[lane]);
            }

            // Load and rotate
            __m256 x_first_vec = _mm256_loadu_ps(x_first);
            __m256 x_second_vec = _mm256_loadu_ps(x_second);
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first_vec, cos_vec),
                _mm256_mul_ps(x_second_vec, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first_vec, sin_vec),
                _mm256_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm256_storeu_ps(x_first, new_first);
            _mm256_storeu_ps(x_second, new_second);

            // Convert FP32 → FP16
            for (int lane = 0; lane < 8; ++lane)
            {
                first_fp16[lane] = fp32_to_fp16(x_first[lane]);
                second_fp16[lane] = fp32_to_fp16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_fp16, 8 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_fp16, 8 * sizeof(uint16_t));
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    int apply_rope_to_head_fp16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
            }

            // Load FP16 values
            alignas(64) uint16_t first_fp16[16];
            alignas(64) uint16_t second_fp16[16];
            std::memcpy(first_fp16, head_ptr + i, 16 * sizeof(uint16_t));
            std::memcpy(second_fp16, head_ptr + i + half_dim, 16 * sizeof(uint16_t));

            // Convert FP16 → FP32
            alignas(64) float x_first[16];
            alignas(64) float x_second[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                x_first[lane] = fp16_to_fp32(first_fp16[lane]);
                x_second[lane] = fp16_to_fp32(second_fp16[lane]);
            }

            // Load values
            __m512 x_first_vec = _mm512_loadu_ps(x_first);
            __m512 x_second_vec = _mm512_loadu_ps(x_second);
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first_vec, cos_vec),
                _mm512_mul_ps(x_second_vec, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first_vec, sin_vec),
                _mm512_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm512_storeu_ps(x_first, new_first);
            _mm512_storeu_ps(x_second, new_second);

            // Convert FP32 → FP16
            for (int lane = 0; lane < 16; ++lane)
            {
                first_fp16[lane] = fp32_to_fp16(x_first[lane]);
                second_fp16[lane] = fp32_to_fp16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_fp16, 16 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_fp16, 16 * sizeof(uint16_t));
        }

        return i;
    }
#endif

    static void apply_rope_to_head_fp16_vectorized(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if LLAMINAR_COMPILED_WITH_NEON
        processed = apply_rope_to_head_neon<neon_rope::Storage::Half>(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX512F__)
        processed = apply_rope_to_head_fp16_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_fp16_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_fp16_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    /** @brief Rotate FP16 heads using shared absolute-position angle rows. */
    static void apply_rope_to_tensor_fp16_positions(
        uint16_t *q, uint16_t *k,
        int seq_len, int q_heads, int k_heads, int head_dim,
        int n_past, const std::vector<float> &inv_freq, const RoPEAngleRows &angles)
    {
        const auto rotate = [head_dim](uint16_t *head_ptr, const float *cos_ptr, const float *sin_ptr)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::Half>(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_fp16_avx512(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_fp16_avx2(head_ptr, cos_ptr, sin_ptr, head_dim);
#else
            const int half_dim = head_dim / 2;
            // Explicit scalar diagnostics preserve their fixed rotation arithmetic.
            for (int i = 0; i < half_dim; ++i)
            {
                float x_first = simd::fp16_to_fp32(head_ptr[i]);
                float x_second = simd::fp16_to_fp32(head_ptr[i + half_dim]);
                float cos_val = cos_ptr[i];
                float sin_val = sin_ptr[i];

                float n1 = x_first * cos_val - x_second * sin_val;
                float n2 = x_first * sin_val + x_second * cos_val;

                head_ptr[i] = simd::fp32_to_fp16(n1);
                head_ptr[i + half_dim] = simd::fp32_to_fp16(n2);
            }
#endif
        };
        apply_rope_pair_positions(q, k, seq_len, q_heads, k_heads, head_dim, n_past, inv_freq, angles, rotate);
    }

    /** @brief Rotate one row of Half heads once through the shared participant workshare.
     * @param q Original query bank.
     * @param k Optional original key bank.
     * @param q_heads Query head count.
     * @param k_heads Key head count when k is present.
     * @param head_dim Original even head width.
     * @param cosine Completed shared cosine row.
     * @param sine Completed shared sine row.
     */
    static void apply_rope_row_fp16(
        uint16_t *q, uint16_t *k,
        int q_heads, int k_heads, int head_dim,
        const float *cosine, const float *sine)
    {
        const auto rotate = [head_dim](uint16_t *head, const float *cosine, const float *sine)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::Half>(head, cosine, sine, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_fp16_avx512(head, cosine, sine, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_fp16_avx2(head, cosine, sine, head_dim);
#else
            const int half = head_dim / 2;
            for (int i = 0; i < half; ++i)
            {
                const float x = simd::fp16_to_fp32(head[i]), y = simd::fp16_to_fp32(head[i + half]);
                head[i] = simd::fp32_to_fp16(x * cosine[i] - y * sine[i]);
                head[i + half] = simd::fp32_to_fp16(x * sine[i] + y * cosine[i]);
            }
#endif
        };
        apply_rope_row_heads(q, k, q_heads, k_heads, head_dim, cosine, sine, rotate);
    }

    static void apply_rope_to_tensor_fp16_direct(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        // Avoid OpenMP overhead for single-token decode
        if (seq_len == 1)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + h * head_dim;
                apply_rope_to_head_fp16_vectorized(head_ptr, n_past, inv_freq, head_dim);
            }
        }
        else
        {
            auto do_rope_fp16_direct_work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < num_heads; ++h)
                    {
                        uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                        apply_rope_to_head_fp16_vectorized(head_ptr, n_past + t, inv_freq, head_dim);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_rope_fp16_direct_work);
        }
    }

    void apply_rope_fp16(
        uint16_t *q_fp16, uint16_t *k_fp16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        const RoPEAngleRows &angles)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        if (seq_len == 1)
        {
            // One shared row; every head reads it through one head workshare.
            fill_shared_angle_rows<false>(inv_freq, angles, 1, RoPEPaddingRows::EveryPosition,
                [n_past](int) { return n_past; });
            apply_rope_row_fp16(q_fp16, k_fp16, q_heads, k_heads, head_dim, angles.cos_row(0), angles.sin_row(0));
            return;
        }

        // Short multi-row sequences keep their direct per-head program.
        if (seq_len < 32)
        {
            apply_rope_to_tensor_fp16_direct(q_fp16, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_fp16_direct(k_fp16, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
        else
        {
            // Share each absolute-position angle across the head workshare.
            apply_rope_to_tensor_fp16_positions(q_fp16, k_fp16, seq_len, q_heads, k_heads, head_dim, n_past, inv_freq, angles);
        }
    }

    void apply_rope_fp16_decode_equivalent_rows(
        uint16_t *q_fp16, uint16_t *k_fp16,
        const int *position_ids,
        int verifier_rows,
        int head_dim,
        int q_heads, int k_heads,
        int pos_offset, float freq_base,
        const RoPEAngleRows &angles)
    {
        if (!q_fp16 || verifier_rows <= 0 ||
            head_dim <= 0 || (head_dim % 2) != 0)
        {
            return;
        }

        prepare_verifier_angle_rows<false>(
            position_ids, verifier_rows, head_dim, 0,
            pos_offset, freq_base, angles);
        const int q_row_stride = q_heads * head_dim;
        const int k_row_stride = k_heads * head_dim;

        auto apply_head = [&](uint16_t *head,
                              const float *cos_values,
                              const float *sin_values)
        {
#if LLAMINAR_COMPILED_WITH_NEON
            apply_rope_to_head_cached_neon<neon_rope::Storage::Half>(head, cos_values, sin_values, head_dim);
#elif defined(__AVX512F__)
            apply_rope_to_head_cached_fp16_avx512(
                head, cos_values, sin_values, head_dim);
#elif defined(__AVX2__)
            apply_rope_to_head_cached_fp16_avx2(
                head, cos_values, sin_values, head_dim);
#else
            const int half_dim = head_dim / 2;
            for (int index = 0; index < half_dim; ++index)
            {
                const float first = simd::fp16_to_fp32(head[index]);
                const float second = simd::fp16_to_fp32(head[index + half_dim]);
                const float rotated_first =
                    first * cos_values[index] - second * sin_values[index];
                const float rotated_second =
                    first * sin_values[index] + second * cos_values[index];
                head[index] = simd::fp32_to_fp16(rotated_first);
                head[index + half_dim] = simd::fp32_to_fp16(rotated_second);
            }
#endif
        };

        auto apply_grouped_rows = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < verifier_rows; ++row)
            {
                for (int head_index = 0; head_index < q_heads; ++head_index)
                {
                    if (!angles.active[static_cast<size_t>(row)])
                        continue;
                    uint16_t *head =
                        q_fp16 + static_cast<size_t>(row) * q_row_stride +
                        static_cast<size_t>(head_index) * head_dim;
                    const float *row_cos = angles.cos_row(row);
                    const float *row_sin = angles.sin_row(row);
                    apply_head(head, row_cos, row_sin);
                }
            }

            if (k_fp16)
            {
#pragma omp for collapse(2) schedule(static)
                for (int row = 0; row < verifier_rows; ++row)
                {
                    for (int head_index = 0; head_index < k_heads; ++head_index)
                    {
                        if (!angles.active[static_cast<size_t>(row)])
                            continue;
                        uint16_t *head =
                            k_fp16 + static_cast<size_t>(row) * k_row_stride +
                            static_cast<size_t>(head_index) * head_dim;
                        const float *row_cos = angles.cos_row(row);
                        const float *row_sin = angles.sin_row(row);
                        apply_head(head, row_cos, row_sin);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(apply_grouped_rows);
    }

    // ============================================================================
    // INT32 Stub (Not Supported)
    // ============================================================================

    bool apply_rope_int32(
        int32_t *, int32_t *, int, int, int, int, int, float)
    {
        // RoPE is an activation operation and cannot be applied to quantized INT32 accumulators
        return false;
    }

    // ============================================================================
    // Q8_1 Pure-Integer RoPE Implementation
    // ============================================================================

    void compute_rope_sincos_q15(
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        RoPESinCosQ15 &out)
    {
        const int half_dim = head_dim / 2;
        out.resize(half_dim);

        for (int i = 0; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
            float cos_val, sin_val;
#if defined(__APPLE__)
            __sincosf(angle, &sin_val, &cos_val);
#elif defined(__GNUC__)
            sincosf(angle, &sin_val, &cos_val);
#else
            cos_val = std::cos(angle);
            sin_val = std::sin(angle);
#endif
            // Convert to Q15: multiply by 32767 and clamp
            // Q15 range: [-32768, 32767] represents [-1.0, ~1.0)
            out.cos_q15[i] = static_cast<int16_t>(std::max(-32767.0f, std::min(32767.0f, cos_val * 32767.0f)));
            out.sin_q15[i] = static_cast<int16_t>(std::max(-32767.0f, std::min(32767.0f, sin_val * 32767.0f)));
        }
    }

    /**
     * @brief Helper: FP16 to FP32 conversion for Q8_1 scale
     */
    static inline float fp16_to_fp32_rope(uint16_t h)
    {
        // IEEE 754 half-precision to single-precision conversion
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0)
            {
                // Zero
                uint32_t result = sign << 31;
                float f;
                std::memcpy(&f, &result, sizeof(float));
                return f;
            }
            else
            {
                // Subnormal
                while ((mant & 0x400) == 0)
                {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= 0x3FF;
            }
        }
        else if (exp == 31)
        {
            // Inf or NaN
            uint32_t result = (sign << 31) | 0x7F800000 | (mant << 13);
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }

        exp = exp + (127 - 15);
        uint32_t result = (sign << 31) | (exp << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }

    /**
     * @brief Helper: FP32 to FP16 conversion for Q8_1 scale
     */
    // static inline uint16_t fp32_to_fp16_rope(float f) - REMOVED, use llaminar2::fp32_to_fp16

    /**
     * @brief Reference/scalar implementation of Q8_1 RoPE for one head
     *
     * This is the baseline implementation. SIMD versions are used when available.
     */
    static void apply_rope_q8_1_integer_head_ref(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;

        // NOTE: We use FP32 for rotation to correctly handle scale differences between blocks.
        // The previous "pure integer" implementation assumed scaleA == scaleB, which is incorrect
        // and caused massive divergence (20x) in dot products.
        //
        // While this involves int8->fp32->int8 conversion, it is necessary for correctness.
        // Modern CPUs have fast int<->float conversion instructions.

        for (int b = 0; b < half_blocks; ++b)
        {
            Q8_1Block &blockA = head_blocks[b];
            Q8_1Block &blockB = head_blocks[b + half_blocks];

            float scaleA = fp16_to_fp32_rope(blockA.d);
            float scaleB = fp16_to_fp32_rope(blockB.d);

            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            float rotA[32];
            float rotB[32];

            // Compute rotated values in FP32
            // We unroll the loop slightly for better pipelining
            for (int i = 0; i < 32; ++i)
            {
                // Dequantize
                float x = static_cast<float>(blockA.qs[i]) * scaleA;
                float y = static_cast<float>(blockB.qs[i]) * scaleB;

                // Convert Q15 sin/cos to float
                float c = static_cast<float>(cos_ptr[i]) * (1.0f / 32767.0f);
                float s = static_cast<float>(sin_ptr[i]) * (1.0f / 32767.0f);

                // Rotate
                rotA[i] = x * c - y * s;
                rotB[i] = x * s + y * c;
            }

            // Requantize blockA
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_val = std::max(max_val, std::abs(rotA[i]));
                }

                float new_scale = max_val / 127.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f; // Avoid div by zero
                float inv_scale = 1.0f / new_scale;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotA[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    blockA.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                blockA.d = llaminar2::fp32_to_fp16(new_scale);
                blockA.sum_qs = static_cast<int16_t>(sum_qs);
            }

            // Requantize blockB
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_val = std::max(max_val, std::abs(rotB[i]));
                }

                float new_scale = max_val / 127.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f;
                float inv_scale = 1.0f / new_scale;

                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotB[i] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    blockB.qs[i] = static_cast<int8_t>(q);
                    sum_qs += q;
                }
                blockB.d = llaminar2::fp32_to_fp16(new_scale);
                blockB.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }
    }

    // ============================================================================
    // Q8_1 RoPE SIMD Implementations
    // ============================================================================

    // Scalar reference implementation (also used as fallback)
    void apply_rope_q8_1_integer_head_scalar(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        // Delegate to the reference implementation
        apply_rope_q8_1_integer_head_ref(head_blocks, blocks_per_head, cos_q15, sin_q15);
    }

#if defined(__AVX2__)
    void apply_rope_q8_1_integer_head_avx2(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m256 q15_scale = _mm256_set1_ps(1.0f / 32767.0f);
        const __m256i clamp_min = _mm256_set1_epi32(-127);
        const __m256i clamp_max = _mm256_set1_epi32(127);

        for (int b = 0; b < half_blocks; ++b)
        {
            Q8_1Block &blockA = head_blocks[b];
            Q8_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m256 scaleA = _mm256_set1_ps(fp16_to_fp32_rope(blockA.d));
            const __m256 scaleB = _mm256_set1_ps(fp16_to_fp32_rope(blockB.d));
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Fully unrolled, interleaved loads
            // Process all 32 elements in 4 chunks of 8
            // Q8_1 uses int8 storage, so we load 8 bytes and sign-extend
            // ================================================================

            // Load all int8 data and sign-extend to int32
            // Use _mm_cvtepi8_epi32 which takes the low 4 bytes of xmm

            // Chunk 0: elements [0:8)
            __m128i qa0_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs));
            __m128i qb0_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs));
            __m128i cos0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr));
            __m128i sin0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr));

            // Chunk 1: elements [8:16)
            __m128i qa1_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs + 8));
            __m128i qb1_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs + 8));
            __m128i cos1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 8));
            __m128i sin1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 8));

            // Chunk 2: elements [16:24)
            __m128i qa2_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs + 16));
            __m128i qb2_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs + 16));
            __m128i cos2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 16));
            __m128i sin2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 16));

            // Chunk 3: elements [24:32)
            __m128i qa3_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockA.qs + 24));
            __m128i qb3_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(blockB.qs + 24));
            __m128i cos3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 24));
            __m128i sin3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 24));

            // Convert chunk 0: int8 → int32 → float, then rotate
            // For int8: first extend to int32 using _mm256_cvtepi8_epi32
            __m256 x0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa0_i8)), scaleA);
            __m256 y0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb0_i8)), scaleB);
            __m256 c0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m256 s0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin0_i16)), q15_scale);
            __m256 rotA0 = _mm256_fmsub_ps(x0, c0, _mm256_mul_ps(y0, s0));
            __m256 rotB0 = _mm256_fmadd_ps(x0, s0, _mm256_mul_ps(y0, c0));

            // Convert chunk 1
            __m256 x1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa1_i8)), scaleA);
            __m256 y1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb1_i8)), scaleB);
            __m256 c1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m256 s1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m256 rotA1 = _mm256_fmsub_ps(x1, c1, _mm256_mul_ps(y1, s1));
            __m256 rotB1 = _mm256_fmadd_ps(x1, s1, _mm256_mul_ps(y1, c1));

            // Convert chunk 2
            __m256 x2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa2_i8)), scaleA);
            __m256 y2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb2_i8)), scaleB);
            __m256 c2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos2_i16)), q15_scale);
            __m256 s2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin2_i16)), q15_scale);
            __m256 rotA2 = _mm256_fmsub_ps(x2, c2, _mm256_mul_ps(y2, s2));
            __m256 rotB2 = _mm256_fmadd_ps(x2, s2, _mm256_mul_ps(y2, c2));

            // Convert chunk 3
            __m256 x3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qa3_i8)), scaleA);
            __m256 y3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qb3_i8)), scaleB);
            __m256 c3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos3_i16)), q15_scale);
            __m256 s3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin3_i16)), q15_scale);
            __m256 rotA3 = _mm256_fmsub_ps(x3, c3, _mm256_mul_ps(y3, s3));
            __m256 rotB3 = _mm256_fmadd_ps(x3, s3, _mm256_mul_ps(y3, c3));

            // ================================================================
            // PHASE 2 & 3: Requantize blockA and blockB INTERLEAVED
            // ================================================================

            // Compute abs and max (interleaved)
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 absA0 = _mm256_andnot_ps(sign_mask, rotA0);
            __m256 absB0 = _mm256_andnot_ps(sign_mask, rotB0);
            __m256 absA1 = _mm256_andnot_ps(sign_mask, rotA1);
            __m256 absB1 = _mm256_andnot_ps(sign_mask, rotB1);
            __m256 absA2 = _mm256_andnot_ps(sign_mask, rotA2);
            __m256 absB2 = _mm256_andnot_ps(sign_mask, rotB2);
            __m256 absA3 = _mm256_andnot_ps(sign_mask, rotA3);
            __m256 absB3 = _mm256_andnot_ps(sign_mask, rotB3);

            // Horizontal max reduction
            __m256 maxA_01 = _mm256_max_ps(absA0, absA1);
            __m256 maxB_01 = _mm256_max_ps(absB0, absB1);
            __m256 maxA_23 = _mm256_max_ps(absA2, absA3);
            __m256 maxB_23 = _mm256_max_ps(absB2, absB3);
            __m256 maxA_all = _mm256_max_ps(maxA_01, maxA_23);
            __m256 maxB_all = _mm256_max_ps(maxB_01, maxB_23);

            // Reduce to scalar
            __m128 maxA_lo = _mm256_castps256_ps128(maxA_all);
            __m128 maxA_hi = _mm256_extractf128_ps(maxA_all, 1);
            __m128 maxA_128 = _mm_max_ps(maxA_lo, maxA_hi);
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxA_val = _mm_cvtss_f32(maxA_128);

            __m128 maxB_lo = _mm256_castps256_ps128(maxB_all);
            __m128 maxB_hi = _mm256_extractf128_ps(maxB_all, 1);
            __m128 maxB_128 = _mm_max_ps(maxB_lo, maxB_hi);
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxB_val = _mm_cvtss_f32(maxB_128);

            // Compute scales
            float scaleA_new = maxA_val * (1.0f / 127.0f);
            float scaleB_new = maxB_val * (1.0f / 127.0f);
            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;

            __m256 invScaleA = _mm256_set1_ps(1.0f / scaleA_new);
            __m256 invScaleB = _mm256_set1_ps(1.0f / scaleB_new);

            // Scale, round, clamp (interleaved)
            __m256 scaledA0 = _mm256_mul_ps(rotA0, invScaleA);
            __m256 scaledB0 = _mm256_mul_ps(rotB0, invScaleB);
            __m256 scaledA1 = _mm256_mul_ps(rotA1, invScaleA);
            __m256 scaledB1 = _mm256_mul_ps(rotB1, invScaleB);
            __m256 scaledA2 = _mm256_mul_ps(rotA2, invScaleA);
            __m256 scaledB2 = _mm256_mul_ps(rotB2, invScaleB);
            __m256 scaledA3 = _mm256_mul_ps(rotA3, invScaleA);
            __m256 scaledB3 = _mm256_mul_ps(rotB3, invScaleB);

            __m256i qA0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT));
            __m256i qA1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT));
            __m256i qA2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA2, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB2, _MM_FROUND_TO_NEAREST_INT));
            __m256i qA3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA3, _MM_FROUND_TO_NEAREST_INT));
            __m256i qB3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB3, _MM_FROUND_TO_NEAREST_INT));

            // Clamp
            qA0 = _mm256_max_epi32(_mm256_min_epi32(qA0, clamp_max), clamp_min);
            qB0 = _mm256_max_epi32(_mm256_min_epi32(qB0, clamp_max), clamp_min);
            qA1 = _mm256_max_epi32(_mm256_min_epi32(qA1, clamp_max), clamp_min);
            qB1 = _mm256_max_epi32(_mm256_min_epi32(qB1, clamp_max), clamp_min);
            qA2 = _mm256_max_epi32(_mm256_min_epi32(qA2, clamp_max), clamp_min);
            qB2 = _mm256_max_epi32(_mm256_min_epi32(qB2, clamp_max), clamp_min);
            qA3 = _mm256_max_epi32(_mm256_min_epi32(qA3, clamp_max), clamp_min);
            qB3 = _mm256_max_epi32(_mm256_min_epi32(qB3, clamp_max), clamp_min);

            // Pack int32 → int16 → int8 and store
            // _mm256_packs_epi32 packs 8 int32 to 8 int16 (saturating)
            // _mm256_packs_epi16 packs 16 int16 to 16 int8 (saturating)
            // But the lane ordering is tricky with AVX2, so we use a different approach

            // Pack chunks 0+1 together, then 2+3
            __m256i qA01_16 = _mm256_packs_epi32(qA0, qA1); // [A0_lo, A1_lo, A0_hi, A1_hi] as int16
            __m256i qA23_16 = _mm256_packs_epi32(qA2, qA3);
            __m256i qB01_16 = _mm256_packs_epi32(qB0, qB1);
            __m256i qB23_16 = _mm256_packs_epi32(qB2, qB3);

            // Pack int16 → int8
            __m256i qA_8 = _mm256_packs_epi16(qA01_16, qA23_16); // Mixed order due to lane crossing
            __m256i qB_8 = _mm256_packs_epi16(qB01_16, qB23_16);

            // Fix the lane ordering: AVX2 packs work within 128-bit lanes
            // After packs: [0-3, 8-11, 16-19, 24-27 | 4-7, 12-15, 20-23, 28-31]
            // We need: [0-7, 8-15, 16-23, 24-31]
            // Use permute to fix: vpermd with indices [0,4,1,5,2,6,3,7]
            const __m256i fix_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            qA_8 = _mm256_permutevar8x32_epi32(qA_8, fix_order);
            qB_8 = _mm256_permutevar8x32_epi32(qB_8, fix_order);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockA.qs), qA_8);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockB.qs), qB_8);

            // Compute sum_qs
            __m256i sumA_01 = _mm256_add_epi32(qA0, qA1);
            __m256i sumA_23 = _mm256_add_epi32(qA2, qA3);
            __m256i sumA_all = _mm256_add_epi32(sumA_01, sumA_23);
            __m128i sumA_lo = _mm256_castsi256_si128(sumA_all);
            __m128i sumA_hi = _mm256_extracti128_si256(sumA_all, 1);
            __m128i sumA_128 = _mm_add_epi32(sumA_lo, sumA_hi);
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(0, 0, 0, 1)));

            __m256i sumB_01 = _mm256_add_epi32(qB0, qB1);
            __m256i sumB_23 = _mm256_add_epi32(qB2, qB3);
            __m256i sumB_all = _mm256_add_epi32(sumB_01, sumB_23);
            __m128i sumB_lo = _mm256_castsi256_si128(sumB_all);
            __m128i sumB_hi = _mm256_extracti128_si256(sumB_all, 1);
            __m128i sumB_128 = _mm_add_epi32(sumB_lo, sumB_hi);
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(0, 0, 0, 1)));

            blockA.d = llaminar2::fp32_to_fp16(scaleA_new);
            blockB.d = llaminar2::fp32_to_fp16(scaleB_new);
            blockA.sum_qs = static_cast<int16_t>(_mm_cvtsi128_si32(sumA_128));
            blockB.sum_qs = static_cast<int16_t>(_mm_cvtsi128_si32(sumB_128));
        }
    }
#endif // __AVX2__

#if defined(__AVX512F__)
    void apply_rope_q8_1_integer_head_avx512(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m512 q15_scale = _mm512_set1_ps(1.0f / 32767.0f);
        const __m512i clamp_min = _mm512_set1_epi32(-127);
        const __m512i clamp_max = _mm512_set1_epi32(127);

        for (int b = 0; b < half_blocks; ++b)
        {
            Q8_1Block &blockA = head_blocks[b];
            Q8_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m512 scaleA = _mm512_set1_ps(fp16_to_fp32_rope(blockA.d));
            const __m512 scaleB = _mm512_set1_ps(fp16_to_fp32_rope(blockB.d));
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Process 32 elements in 2 chunks of 16
            // ================================================================

            // Load 16 int8 values each
            __m128i qa0_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs));
            __m128i qb0_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs));
            __m128i qa1_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 16));
            __m128i qb1_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 16));

            __m256i cos0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr));
            __m256i sin0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr));
            __m256i cos1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr + 16));
            __m256i sin1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr + 16));

            // Convert int8 → int32 → float (AVX512 has _mm512_cvtepi8_epi32 for 16 elements)
            __m512 x0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qa0_i8)), scaleA);
            __m512 y0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qb0_i8)), scaleB);
            __m512 c0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m512 s0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin0_i16)), q15_scale);
            __m512 rotA0 = _mm512_fmsub_ps(x0, c0, _mm512_mul_ps(y0, s0));
            __m512 rotB0 = _mm512_fmadd_ps(x0, s0, _mm512_mul_ps(y0, c0));

            __m512 x1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qa1_i8)), scaleA);
            __m512 y1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qb1_i8)), scaleB);
            __m512 c1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m512 s1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m512 rotA1 = _mm512_fmsub_ps(x1, c1, _mm512_mul_ps(y1, s1));
            __m512 rotB1 = _mm512_fmadd_ps(x1, s1, _mm512_mul_ps(y1, c1));

            // ================================================================
            // PHASE 2 & 3: Requantize blockA and blockB INTERLEAVED
            // ================================================================

            // Compute abs and max (interleaved)
            __m512 absA0 = _mm512_abs_ps(rotA0);
            __m512 absB0 = _mm512_abs_ps(rotB0);
            __m512 absA1 = _mm512_abs_ps(rotA1);
            __m512 absB1 = _mm512_abs_ps(rotB1);

            __m512 maxA_all = _mm512_max_ps(absA0, absA1);
            __m512 maxB_all = _mm512_max_ps(absB0, absB1);

            float maxA_val = _mm512_reduce_max_ps(maxA_all);
            float maxB_val = _mm512_reduce_max_ps(maxB_all);

            // Compute scales
            float scaleA_new = maxA_val * (1.0f / 127.0f);
            float scaleB_new = maxB_val * (1.0f / 127.0f);
            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;

            __m512 invScaleA = _mm512_set1_ps(1.0f / scaleA_new);
            __m512 invScaleB = _mm512_set1_ps(1.0f / scaleB_new);

            // Scale, round, clamp (interleaved)
            __m512 scaledA0 = _mm512_mul_ps(rotA0, invScaleA);
            __m512 scaledB0 = _mm512_mul_ps(rotB0, invScaleB);
            __m512 scaledA1 = _mm512_mul_ps(rotA1, invScaleA);
            __m512 scaledB1 = _mm512_mul_ps(rotB1, invScaleB);

            __m512i qA0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qA1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT));

            qA0 = _mm512_max_epi32(_mm512_min_epi32(qA0, clamp_max), clamp_min);
            qB0 = _mm512_max_epi32(_mm512_min_epi32(qB0, clamp_max), clamp_min);
            qA1 = _mm512_max_epi32(_mm512_min_epi32(qA1, clamp_max), clamp_min);
            qB1 = _mm512_max_epi32(_mm512_min_epi32(qB1, clamp_max), clamp_min);

            // Pack int32 → int8 using AVX512 _mm512_cvtepi32_epi8
            __m128i qA0_8 = _mm512_cvtepi32_epi8(qA0);
            __m128i qB0_8 = _mm512_cvtepi32_epi8(qB0);
            __m128i qA1_8 = _mm512_cvtepi32_epi8(qA1);
            __m128i qB1_8 = _mm512_cvtepi32_epi8(qB1);

            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs), qA0_8);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs), qB0_8);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 16), qA1_8);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 16), qB1_8);

            // Compute sum_qs
            int32_t sumA = _mm512_reduce_add_epi32(_mm512_add_epi32(qA0, qA1));
            int32_t sumB = _mm512_reduce_add_epi32(_mm512_add_epi32(qB0, qB1));

            blockA.d = llaminar2::fp32_to_fp16(scaleA_new);
            blockB.d = llaminar2::fp32_to_fp16(scaleB_new);
            blockA.sum_qs = static_cast<int16_t>(sumA);
            blockB.sum_qs = static_cast<int16_t>(sumB);
        }
    }
#endif // __AVX512F__

    /**
     * @brief Apply Q8_1 RoPE to one head with automatic SIMD dispatch
     *
     * Dispatches to the best available SIMD implementation:
     * - AVX512 if available (fastest, processes 16 elements at a time)
     * - AVX2 if available (8 elements at a time)
     * - Scalar fallback (baseline)
     */
    void apply_rope_q8_1_integer_head(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        // Dispatch to best available SIMD implementation
#if LLAMINAR_COMPILED_WITH_NEON
        neon_rope::rotate_quantized(head_blocks, blocks_per_head, cos_q15, sin_q15);
#elif defined(__AVX512F__)
        apply_rope_q8_1_integer_head_avx512(head_blocks, blocks_per_head, cos_q15, sin_q15);
#elif defined(__AVX2__)
        apply_rope_q8_1_integer_head_avx2(head_blocks, blocks_per_head, cos_q15, sin_q15);
#else
        apply_rope_q8_1_integer_head_scalar(head_blocks, blocks_per_head, cos_q15, sin_q15);
#endif
    }

    void apply_rope_q8_1_integer(
        Q8_1Block *Q,
        Q8_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        const RoPEAngleRows &angles,
        int pos_offset)
    {
        if (head_dim % 32 != 0)
        {
            return; // head_dim must be divisible by Q8_1 block size
        }
        if (seq_len <= 0)
            return;

        const int blocks_per_head = head_dim / 32;
        const int q_stride_blocks = n_heads * blocks_per_head;
        const int k_stride_blocks = n_kv_heads * blocks_per_head;
        const int half_dim = head_dim / 2;

        // Get inverse frequencies
        const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);

        // Q15 rows truncate the shared FP32 row function's values. This integer
        // program rotates every row by its stored position, padding included,
        // so a one-row call no longer reads a previous call's table.
        fill_shared_angle_rows<true>(inv_freq, angles, seq_len, RoPEPaddingRows::EveryPosition,
            [position_ids, pos_offset](int row)
            { return position_ids ? position_ids[row] : pos_offset + row; });
        const int16_t *cos_table = angles.cos_q15.data();
        const int16_t *sin_table = angles.sin_q15.data();

        // Apply RoPE using precomputed tables
        // Collapse heads and tokens for better load balancing
        if (seq_len == 1)
        {
            // One row: standalone callers rotate every head on the caller.
            // Inside an existing team the banks are shared, so one head
            // workshare owns each head exactly once instead of every member
            // rotating all of them.
            const int16_t *c_ptr = cos_table;
            const int16_t *s_ptr = sin_table;
            const int heads = n_heads + (K ? n_kv_heads : 0);
            const auto rotate = [&](int head)
            {
                Q8_1Block *head_ptr = head < n_heads
                    ? Q + static_cast<size_t>(head) * blocks_per_head
                    : K + static_cast<size_t>(head - n_heads) * blocks_per_head;
                apply_rope_q8_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
            };
            if (::llaminar::v2::inLexicalWorkshare())
            {
#pragma omp for schedule(static)
                for (int head = 0; head < heads; ++head)
                    rotate(head);
            }
            else
            {
                for (int head = 0; head < heads; ++head)
                    rotate(head);
            }
        }
        else
        {
            auto do_apply_q_work = [&]()
            {
#pragma omp for collapse(2)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < n_heads; ++h)
                    {
                        Q8_1Block *head_ptr = Q + t * q_stride_blocks + h * blocks_per_head;
                        const int16_t *c_ptr = cos_table + t * half_dim;
                        const int16_t *s_ptr = sin_table + t * half_dim;
                        apply_rope_q8_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_apply_q_work);

            if (K)
            {
                auto do_apply_k_work = [&]()
                {
#pragma omp for collapse(2)
                    for (int t = 0; t < seq_len; ++t)
                    {
                        for (int h = 0; h < n_kv_heads; ++h)
                        {
                            Q8_1Block *head_ptr = K + t * k_stride_blocks + h * blocks_per_head;
                            const int16_t *c_ptr = cos_table + t * half_dim;
                            const int16_t *s_ptr = sin_table + t * half_dim;
                            apply_rope_q8_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                        }
                    }
                };
                OMP_WORKSHARE_REGION(do_apply_k_work);
            }
        }
    }

    void apply_rope_q8_1_decode_equivalent_rows(
        Q8_1Block *Q,
        Q8_1Block *K,
        const int *position_ids,
        int verifier_rows,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int pos_offset,
        float rope_theta,
        const RoPEAngleRows &angles)
    {
        if (!Q || verifier_rows <= 0 ||
            head_dim <= 0 || head_dim % Q8_1Block::BLOCK_SIZE != 0)
        {
            return;
        }

        // Q15 rows truncate the same FP32 rows as one-token Q8 decode.
        prepare_verifier_angle_rows<true>(
            position_ids, verifier_rows, head_dim, 0,
            pos_offset, rope_theta, angles);

        const int blocks_per_head = head_dim / Q8_1Block::BLOCK_SIZE;
        const int q_row_stride = n_heads * blocks_per_head;
        const int k_row_stride = n_kv_heads * blocks_per_head;
        auto apply_grouped_rows = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < verifier_rows; ++row)
            {
                for (int head_index = 0; head_index < n_heads; ++head_index)
                {
                    if (!angles.active[static_cast<size_t>(row)])
                        continue;
                    Q8_1Block *head =
                        Q + static_cast<size_t>(row) * q_row_stride +
                        static_cast<size_t>(head_index) * blocks_per_head;
                    apply_rope_q8_1_integer_head(
                        head,
                        blocks_per_head,
                        angles.cos_q15_row(row),
                        angles.sin_q15_row(row));
                }
            }

            if (K)
            {
#pragma omp for collapse(2) schedule(static)
                for (int row = 0; row < verifier_rows; ++row)
                {
                    for (int head_index = 0; head_index < n_kv_heads; ++head_index)
                    {
                        if (!angles.active[static_cast<size_t>(row)])
                            continue;
                        Q8_1Block *head =
                            K + static_cast<size_t>(row) * k_row_stride +
                            static_cast<size_t>(head_index) * blocks_per_head;
                        apply_rope_q8_1_integer_head(
                            head,
                            blocks_per_head,
                            angles.cos_q15_row(row),
                            angles.sin_q15_row(row));
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(apply_grouped_rows);
    }

    // =========================================================================
    // Q8_1 → FP32 RoPE (Hybrid Mode - No Requantization)
    // =========================================================================

    /** @brief Reject overlapping retained conversion authorities without pointer-end overflow.
     * @param a First byte root; @param a_bytes Complete first extent.
     * @param b Second byte root; @param b_bytes Complete second extent.
     */
    static void require_rope_disjoint(const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes)
    {
        const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        if (a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes))
            throw std::invalid_argument("CPU RoPE conversion output aliases a retained input/publication authority");
    }

    /** @brief Price a physical byte extent only to validate integer geometry, not memory admission.
     * @param rows Nonnegative original physical M.
     * @param heads Nonnegative physical heads per row.
     * @param words Positive physical storage words per head.
     * @param bytes Positive bytes per physical storage word.
     * @return Complete nonoverflowing bank extent.
     */
    static std::size_t rope_bank_bytes(int rows, int heads, int words, std::size_t bytes)
    {
        if (rows < 0 || heads < 0 || words <= 0 || !bytes)
            throw std::invalid_argument("CPU RoPE conversion has invalid physical geometry");
        if (!rows || !heads) return 0;
        if (std::size_t(rows) > std::numeric_limits<std::size_t>::max() / heads / words / bytes)
            throw std::invalid_argument("CPU RoPE conversion physical extent overflows");
        return std::size_t(rows) * heads * words * bytes;
    }

    /** @brief Stream one original Q8 head into its FP32 output without a head-sized mirror.
     * @param input Complete original readable Q8 head.
     * @param output Disjoint original FP32 head.
     * @param dim Complete Q8-block-aligned logical width.
     * @param cosine Borrowed original FP32 cosine plane.
     * @param sine Matching original sine plane.
     */
    static void apply_rope_q8_head_to_fp32(const Q8_1Block *input, float *output, int dim,
        const float *cosine, const float *sine)
    {
#if LLAMINAR_COMPILED_WITH_NEON
        neon_rope::q8_to_fp32_head(input, output, dim, cosine, sine);
#else
        // This existing x86 scalar program is streamed instead of using its
        // formerly fixed 256-value stack mirror. No ISA mode is substituted.
        const int half = dim / 2;
        for (int first = 0; first < half;)
        {
            const int second = first + half;
            const auto &a = input[first / 32], &b = input[second / 32];
            const float a_scale = fp16_to_fp32_rope(a.d), b_scale = fp16_to_fp32_rope(b.d);
            const int count = std::min({32 - first % 32, 32 - second % 32, half - first});
            for (int i = 0; i < count; ++i)
            {
                const float x = float(a.qs[first % 32 + i]) * a_scale;
                const float y = float(b.qs[second % 32 + i]) * b_scale;
                output[first + i] = x * cosine[first + i] - y * sine[first + i];
                output[second + i] = x * sine[first + i] + y * cosine[first + i];
            }
            first += count;
        }
#endif
    }

    /** @brief Convert original Q/K rows through the existing reciprocal-power coefficient program.
     * @param Q_in Original query Q8 bank; @param K_in Optional original key Q8 bank.
     * @param Q_out Original disjoint query FP32 destination; @param K_out Matching optional key destination.
     * @param position_ids Original position words; negative padding retains the established position-zero program.
     * @param seq_len Nonnegative original physical row count.
     * @param n_heads Query heads per row; @param n_kv_heads Key heads per row.
     * @param head_dim Positive Q8-block-aligned logical head width.
     * @param rope_theta Positive finite coefficient base.
     */
    void apply_rope_q8_1_to_fp32(
        const Q8_1Block *Q_in, const Q8_1Block *K_in, float *Q_out, float *K_out,
        const int *position_ids, int seq_len, int n_heads, int n_kv_heads, int head_dim, float rope_theta)
    {
        if (seq_len < 0 || n_heads < 0 || n_kv_heads < 0 || head_dim <= 0 || head_dim % 32 ||
            !std::isfinite(rope_theta) || rope_theta <= 0 || (!K_in != !K_out))
            throw std::invalid_argument("CPU Q8/FP32 RoPE requires complete row/head/key geometry");
        const int blocks = head_dim / 32, half = head_dim / 2;
        const auto q_in_bytes = rope_bank_bytes(seq_len, n_heads, blocks, sizeof(Q8_1Block));
        const auto q_out_bytes = rope_bank_bytes(seq_len, n_heads, head_dim, sizeof(float));
        const auto k_in_bytes = K_in ? rope_bank_bytes(seq_len, n_kv_heads, blocks, sizeof(Q8_1Block)) : 0;
        const auto k_out_bytes = K_out ? rope_bank_bytes(seq_len, n_kv_heads, head_dim, sizeof(float)) : 0;
        const auto coefficient_bytes = rope_bank_bytes(seq_len, 1, half, sizeof(float));
        if ((q_in_bytes && (!Q_in || !Q_out)) || (k_in_bytes && (!K_in || !K_out)))
            throw std::invalid_argument("CPU Q8/FP32 RoPE requires every live original bank");
        for (const auto write : {std::pair<const void *, std::size_t>{Q_out, q_out_bytes}, {K_out, k_out_bytes}})
        {
            require_rope_disjoint(write.first, write.second, Q_in, q_in_bytes);
            require_rope_disjoint(write.first, write.second, K_in, k_in_bytes);
            require_rope_disjoint(write.first, write.second, position_ids, position_ids ? std::size_t(seq_len) * sizeof(int) : 0);
        }
        require_rope_disjoint(Q_out, q_out_bytes, K_out, k_out_bytes);
        if (!seq_len || (!n_heads && (!K_in || !n_kv_heads)))
        {
            (void)llaminar::v2::beginWorkshare(false);
            return;
        }
        const auto q_block_stride = std::size_t(n_heads) * blocks, k_block_stride = std::size_t(n_kv_heads) * blocks;
        const auto q_float_stride = std::size_t(n_heads) * head_dim, k_float_stride = std::size_t(n_kv_heads) * head_dim;
        std::vector<float> cosine(coefficient_bytes / sizeof(float)), sine(cosine.size()), inv_freq(half);
        for (int i = 0; i < half; ++i)
            inv_freq[i] = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
        for (int row = 0; row < seq_len; ++row)
        {
            const int position = position_ids ? std::max(position_ids[row], 0) : row;
            for (int i = 0; i < half; ++i)
            {
                const float angle = static_cast<float>(position) * inv_freq[i];
                cosine[std::size_t(row) * half + i] = std::cos(angle);
                sine[std::size_t(row) * half + i] = std::sin(angle);
            }
        }
        auto work = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < seq_len; ++row)
                for (int head = 0; head < n_heads; ++head)
                    apply_rope_q8_head_to_fp32(Q_in + std::size_t(row) * q_block_stride + std::size_t(head) * blocks,
                        Q_out + std::size_t(row) * q_float_stride + std::size_t(head) * head_dim, head_dim,
                        cosine.data() + std::size_t(row) * half, sine.data() + std::size_t(row) * half);
            if (K_in)
            {
#pragma omp for collapse(2) schedule(static)
                for (int row = 0; row < seq_len; ++row)
                    for (int head = 0; head < n_kv_heads; ++head)
                        apply_rope_q8_head_to_fp32(K_in + std::size_t(row) * k_block_stride + std::size_t(head) * blocks,
                            K_out + std::size_t(row) * k_float_stride + std::size_t(head) * head_dim, head_dim,
                            cosine.data() + std::size_t(row) * half, sine.data() + std::size_t(row) * half);
            }
        };
        OMP_WORKSHARE_REGION(work);
    }

    // ============================================================================
    // Q16_1 In-Place RoPE Implementation - Vectorized
    // ============================================================================
    //
    // Algorithm: Dequant Q16_1 → Rotate in FP32 → Requant to Q16_1
    // Processes block pairs (first half vs second half of head) for RoPE rotation.
    //
    // Optimization techniques:
    // - Dual load port exploitation: Interleaved loads from A/B blocks
    // - Loop unrolling: Process 2 block pairs per outer iteration
    // - Register blocking: Keep rotated values in registers, minimize spills
    // - Prefetching: Hint at upcoming block pairs
    // - FMA fusion: Use _mm256_fmadd_ps / _mm512_fmadd_ps where beneficial
    //
    // Expected speedups: AVX512 ~2x AVX2, AVX2 ~8x scalar
    // ============================================================================

    // Q15 to FP32 conversion constant
    namespace
    {
        constexpr float Q15_TO_FP32 = 1.0f / 32767.0f;
    }

    void apply_rope_q16_1_integer_head_scalar(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;

        for (int b = 0; b < half_blocks; ++b)
        {
            Q16_1Block &blockA = head_blocks[b];
            Q16_1Block &blockB = head_blocks[b + half_blocks];

            float scaleA = blockA.d;
            float scaleB = blockB.d;

            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            float rotA[32];
            float rotB[32];

            // Dequant + Rotate
            for (int i = 0; i < 32; ++i)
            {
                float x = static_cast<float>(blockA.qs[i]) * scaleA;
                float y = static_cast<float>(blockB.qs[i]) * scaleB;
                float c = static_cast<float>(cos_ptr[i]) * Q15_TO_FP32;
                float s = static_cast<float>(sin_ptr[i]) * Q15_TO_FP32;
                rotA[i] = x * c - y * s;
                rotB[i] = x * s + y * c;
            }

            // Requantize blockA
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                    max_val = std::max(max_val, std::abs(rotA[i]));

                float new_scale = max_val / 32767.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f;
                float inv_scale = 1.0f / new_scale;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotA[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    blockA.qs[i] = static_cast<int16_t>(q);
                    sum_qs += q;
                }
                blockA.d = new_scale;
                blockA.sum_qs = static_cast<int32_t>(sum_qs);
            }

            // Requantize blockB
            {
                float max_val = 0.0f;
                for (int i = 0; i < 32; ++i)
                    max_val = std::max(max_val, std::abs(rotB[i]));

                float new_scale = max_val / 32767.0f;
                if (new_scale < 1e-20f)
                    new_scale = 1e-20f;
                float inv_scale = 1.0f / new_scale;

                int64_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int32_t q = static_cast<int32_t>(std::round(rotB[i] * inv_scale));
                    q = std::max(-32767, std::min(32767, q));
                    blockB.qs[i] = static_cast<int16_t>(q);
                    sum_qs += q;
                }
                blockB.d = new_scale;
                blockB.sum_qs = static_cast<int32_t>(sum_qs);
            }
        }
    }

#if defined(__AVX2__)
    void apply_rope_q16_1_integer_head_avx2(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m256 q15_scale = _mm256_set1_ps(Q15_TO_FP32);
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        const __m256i clamp_min = _mm256_set1_epi32(-32767);
        const __m256i clamp_max = _mm256_set1_epi32(32767);

        // Process block pairs - fully unrolled inner loop for 32-element blocks
        for (int b = 0; b < half_blocks; ++b)
        {
            Q16_1Block &blockA = head_blocks[b];
            Q16_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair (if exists)
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m256 scaleA = _mm256_set1_ps(blockA.d);
            const __m256 scaleB = _mm256_set1_ps(blockB.d);
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Fully unrolled, interleaved loads
            // Process all 32 elements in 4 chunks of 8
            // ================================================================

            // Chunk 0: elements [0:8)
            // Interleaved loads exploit dual load ports
            __m128i qa0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs));
            __m128i qb0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs));
            __m128i cos0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr));
            __m128i sin0_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr));

            // Chunk 1: elements [8:16) - load while chunk 0 converts
            __m128i qa1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 8));
            __m128i qb1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 8));
            __m128i cos1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 8));
            __m128i sin1_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 8));

            // Chunk 2: elements [16:24)
            __m128i qa2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 16));
            __m128i qb2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 16));
            __m128i cos2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 16));
            __m128i sin2_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 16));

            // Chunk 3: elements [24:32)
            __m128i qa3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockA.qs + 24));
            __m128i qb3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blockB.qs + 24));
            __m128i cos3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cos_ptr + 24));
            __m128i sin3_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sin_ptr + 24));

            // Convert chunk 0: int16 → int32 → float, then rotate
            __m256 x0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa0_i16)), scaleA);
            __m256 y0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb0_i16)), scaleB);
            __m256 c0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m256 s0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin0_i16)), q15_scale);
            // rotA = x*c - y*s, rotB = x*s + y*c (use FMA)
            __m256 rotA0 = _mm256_fmsub_ps(x0, c0, _mm256_mul_ps(y0, s0));
            __m256 rotB0 = _mm256_fmadd_ps(x0, s0, _mm256_mul_ps(y0, c0));

            // Convert chunk 1
            __m256 x1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa1_i16)), scaleA);
            __m256 y1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb1_i16)), scaleB);
            __m256 c1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m256 s1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m256 rotA1 = _mm256_fmsub_ps(x1, c1, _mm256_mul_ps(y1, s1));
            __m256 rotB1 = _mm256_fmadd_ps(x1, s1, _mm256_mul_ps(y1, c1));

            // Convert chunk 2
            __m256 x2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa2_i16)), scaleA);
            __m256 y2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb2_i16)), scaleB);
            __m256 c2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos2_i16)), q15_scale);
            __m256 s2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin2_i16)), q15_scale);
            __m256 rotA2 = _mm256_fmsub_ps(x2, c2, _mm256_mul_ps(y2, s2));
            __m256 rotB2 = _mm256_fmadd_ps(x2, s2, _mm256_mul_ps(y2, c2));

            // Convert chunk 3
            __m256 x3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qa3_i16)), scaleA);
            __m256 y3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qb3_i16)), scaleB);
            __m256 c3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(cos3_i16)), q15_scale);
            __m256 s3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sin3_i16)), q15_scale);
            __m256 rotA3 = _mm256_fmsub_ps(x3, c3, _mm256_mul_ps(y3, s3));
            __m256 rotB3 = _mm256_fmadd_ps(x3, s3, _mm256_mul_ps(y3, c3));

            // ================================================================
            // PHASE 2: Requantize blockA - find max, scale, pack
            // ================================================================

            // Find max_abs across all 4 chunks (parallel reduction)
            __m256 absA0 = _mm256_andnot_ps(sign_mask, rotA0);
            __m256 absA1 = _mm256_andnot_ps(sign_mask, rotA1);
            __m256 absA2 = _mm256_andnot_ps(sign_mask, rotA2);
            __m256 absA3 = _mm256_andnot_ps(sign_mask, rotA3);

            __m256 maxA_01 = _mm256_max_ps(absA0, absA1);
            __m256 maxA_23 = _mm256_max_ps(absA2, absA3);
            __m256 maxA_all = _mm256_max_ps(maxA_01, maxA_23);

            // Horizontal max reduction
            __m128 maxA_lo = _mm256_castps256_ps128(maxA_all);
            __m128 maxA_hi = _mm256_extractf128_ps(maxA_all, 1);
            __m128 maxA_128 = _mm_max_ps(maxA_lo, maxA_hi);
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxA_128 = _mm_max_ps(maxA_128, _mm_shuffle_ps(maxA_128, maxA_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxA_val = _mm_cvtss_f32(maxA_128);

            float scaleA_new = maxA_val / 32767.0f;
            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            __m256 invScaleA = _mm256_set1_ps(1.0f / scaleA_new);

            // Scale, round, clamp, pack all 4 chunks
            __m256 scaledA0 = _mm256_mul_ps(rotA0, invScaleA);
            __m256 scaledA1 = _mm256_mul_ps(rotA1, invScaleA);
            __m256 scaledA2 = _mm256_mul_ps(rotA2, invScaleA);
            __m256 scaledA3 = _mm256_mul_ps(rotA3, invScaleA);

            __m256i qA0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qA1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qA2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qA3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledA3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            // Clamp
            qA0 = _mm256_max_epi32(_mm256_min_epi32(qA0, clamp_max), clamp_min);
            qA1 = _mm256_max_epi32(_mm256_min_epi32(qA1, clamp_max), clamp_min);
            qA2 = _mm256_max_epi32(_mm256_min_epi32(qA2, clamp_max), clamp_min);
            qA3 = _mm256_max_epi32(_mm256_min_epi32(qA3, clamp_max), clamp_min);

            // Pack int32 → int16 and store
            __m128i qA0_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA0), _mm256_extracti128_si256(qA0, 1));
            __m128i qA1_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA1), _mm256_extracti128_si256(qA1, 1));
            __m128i qA2_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA2), _mm256_extracti128_si256(qA2, 1));
            __m128i qA3_16 = _mm_packs_epi32(_mm256_castsi256_si128(qA3), _mm256_extracti128_si256(qA3, 1));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs), qA0_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 8), qA1_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 16), qA2_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockA.qs + 24), qA3_16);

            // Compute sum_qs (parallel reduction)
            __m256i sumA_01 = _mm256_add_epi32(qA0, qA1);
            __m256i sumA_23 = _mm256_add_epi32(qA2, qA3);
            __m256i sumA_all = _mm256_add_epi32(sumA_01, sumA_23);
            __m128i sumA_lo = _mm256_castsi256_si128(sumA_all);
            __m128i sumA_hi = _mm256_extracti128_si256(sumA_all, 1);
            __m128i sumA_128 = _mm_add_epi32(sumA_lo, sumA_hi);
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumA_128 = _mm_add_epi32(sumA_128, _mm_shuffle_epi32(sumA_128, _MM_SHUFFLE(0, 0, 0, 1)));

            blockA.d = scaleA_new;
            blockA.sum_qs = _mm_cvtsi128_si32(sumA_128);

            // ================================================================
            // PHASE 3: Requantize blockB - same pattern
            // ================================================================

            __m256 absB0 = _mm256_andnot_ps(sign_mask, rotB0);
            __m256 absB1 = _mm256_andnot_ps(sign_mask, rotB1);
            __m256 absB2 = _mm256_andnot_ps(sign_mask, rotB2);
            __m256 absB3 = _mm256_andnot_ps(sign_mask, rotB3);

            __m256 maxB_01 = _mm256_max_ps(absB0, absB1);
            __m256 maxB_23 = _mm256_max_ps(absB2, absB3);
            __m256 maxB_all = _mm256_max_ps(maxB_01, maxB_23);

            __m128 maxB_lo = _mm256_castps256_ps128(maxB_all);
            __m128 maxB_hi = _mm256_extractf128_ps(maxB_all, 1);
            __m128 maxB_128 = _mm_max_ps(maxB_lo, maxB_hi);
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            maxB_128 = _mm_max_ps(maxB_128, _mm_shuffle_ps(maxB_128, maxB_128, _MM_SHUFFLE(0, 0, 0, 1)));
            float maxB_val = _mm_cvtss_f32(maxB_128);

            float scaleB_new = maxB_val / 32767.0f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;
            __m256 invScaleB = _mm256_set1_ps(1.0f / scaleB_new);

            __m256 scaledB0 = _mm256_mul_ps(rotB0, invScaleB);
            __m256 scaledB1 = _mm256_mul_ps(rotB1, invScaleB);
            __m256 scaledB2 = _mm256_mul_ps(rotB2, invScaleB);
            __m256 scaledB3 = _mm256_mul_ps(rotB3, invScaleB);

            __m256i qB0 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qB1 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qB2 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i qB3 = _mm256_cvtps_epi32(_mm256_round_ps(scaledB3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            qB0 = _mm256_max_epi32(_mm256_min_epi32(qB0, clamp_max), clamp_min);
            qB1 = _mm256_max_epi32(_mm256_min_epi32(qB1, clamp_max), clamp_min);
            qB2 = _mm256_max_epi32(_mm256_min_epi32(qB2, clamp_max), clamp_min);
            qB3 = _mm256_max_epi32(_mm256_min_epi32(qB3, clamp_max), clamp_min);

            __m128i qB0_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB0), _mm256_extracti128_si256(qB0, 1));
            __m128i qB1_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB1), _mm256_extracti128_si256(qB1, 1));
            __m128i qB2_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB2), _mm256_extracti128_si256(qB2, 1));
            __m128i qB3_16 = _mm_packs_epi32(_mm256_castsi256_si128(qB3), _mm256_extracti128_si256(qB3, 1));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs), qB0_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 8), qB1_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 16), qB2_16);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(blockB.qs + 24), qB3_16);

            __m256i sumB_01 = _mm256_add_epi32(qB0, qB1);
            __m256i sumB_23 = _mm256_add_epi32(qB2, qB3);
            __m256i sumB_all = _mm256_add_epi32(sumB_01, sumB_23);
            __m128i sumB_lo = _mm256_castsi256_si128(sumB_all);
            __m128i sumB_hi = _mm256_extracti128_si256(sumB_all, 1);
            __m128i sumB_128 = _mm_add_epi32(sumB_lo, sumB_hi);
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(1, 0, 3, 2)));
            sumB_128 = _mm_add_epi32(sumB_128, _mm_shuffle_epi32(sumB_128, _MM_SHUFFLE(0, 0, 0, 1)));

            blockB.d = scaleB_new;
            blockB.sum_qs = _mm_cvtsi128_si32(sumB_128);
        }
    }
#endif // __AVX2__

#if defined(__AVX512F__)
    void apply_rope_q16_1_integer_head_avx512(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        const int half_blocks = blocks_per_head / 2;
        const __m512 q15_scale = _mm512_set1_ps(Q15_TO_FP32);

        const __m512i clamp_min = _mm512_set1_epi32(-32767);
        const __m512i clamp_max = _mm512_set1_epi32(32767);

        for (int b = 0; b < half_blocks; ++b)
        {
            Q16_1Block &blockA = head_blocks[b];
            Q16_1Block &blockB = head_blocks[b + half_blocks];

            // Prefetch next block pair
            if (b + 1 < half_blocks)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&head_blocks[b + 1 + half_blocks]), _MM_HINT_T0);
            }

            const __m512 scaleA = _mm512_set1_ps(blockA.d);
            const __m512 scaleB = _mm512_set1_ps(blockB.d);
            const int16_t *cos_ptr = cos_q15 + b * 32;
            const int16_t *sin_ptr = sin_q15 + b * 32;

            // ================================================================
            // PHASE 1: Dequant + Rotate - Fully unrolled, interleaved loads
            // Process all 32 elements in 2 chunks of 16
            // ================================================================

            // Load all data upfront (exploit dual load ports)
            __m256i qa0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockA.qs));
            __m256i qb0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockB.qs));
            __m256i qa1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockA.qs + 16));
            __m256i qb1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blockB.qs + 16));

            __m256i cos0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr));
            __m256i sin0_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr));
            __m256i cos1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cos_ptr + 16));
            __m256i sin1_i16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sin_ptr + 16));

            // Convert and rotate chunk 0 (elements 0-15)
            __m512 x0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa0_i16)), scaleA);
            __m512 y0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb0_i16)), scaleB);
            __m512 c0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos0_i16)), q15_scale);
            __m512 s0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin0_i16)), q15_scale);
            // rotA = x*c - y*s, rotB = x*s + y*c (use FMA)
            __m512 rotA0 = _mm512_fmsub_ps(x0, c0, _mm512_mul_ps(y0, s0));
            __m512 rotB0 = _mm512_fmadd_ps(x0, s0, _mm512_mul_ps(y0, c0));

            // Convert and rotate chunk 1 (elements 16-31)
            __m512 x1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qa1_i16)), scaleA);
            __m512 y1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(qb1_i16)), scaleB);
            __m512 c1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(cos1_i16)), q15_scale);
            __m512 s1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(sin1_i16)), q15_scale);
            __m512 rotA1 = _mm512_fmsub_ps(x1, c1, _mm512_mul_ps(y1, s1));
            __m512 rotB1 = _mm512_fmadd_ps(x1, s1, _mm512_mul_ps(y1, c1));

            // ================================================================
            // PHASE 2 & 3: Requantize blockA and blockB INTERLEAVED
            // Process both blocks together to hide latency and maximize ILP
            // ================================================================

            // 1. Compute Abs and Max (Interleaved)
            __m512 absA0 = _mm512_abs_ps(rotA0);
            __m512 absB0 = _mm512_abs_ps(rotB0);
            __m512 absA1 = _mm512_abs_ps(rotA1);
            __m512 absB1 = _mm512_abs_ps(rotB1);

            __m512 maxA_all = _mm512_max_ps(absA0, absA1);
            __m512 maxB_all = _mm512_max_ps(absB0, absB1);

            // Horizontal reductions (expensive, but independent)
            float maxA_val = _mm512_reduce_max_ps(maxA_all);
            float maxB_val = _mm512_reduce_max_ps(maxB_all);

            // 2. Compute Scales (Scalar ops, high latency)
            // Use multiplication by reciprocal for speed
            float scaleA_new = maxA_val * (1.0f / 32767.0f);
            float scaleB_new = maxB_val * (1.0f / 32767.0f);

            if (scaleA_new < 1e-20f)
                scaleA_new = 1e-20f;
            if (scaleB_new < 1e-20f)
                scaleB_new = 1e-20f;

            // 3. Compute Inverse Scales (Scalar division)
            // Compiler can interleave these independent divisions
            __m512 invScaleA = _mm512_set1_ps(1.0f / scaleA_new);
            __m512 invScaleB = _mm512_set1_ps(1.0f / scaleB_new);

            // 4. Scale, Round, Clamp (Interleaved)
            __m512 scaledA0 = _mm512_mul_ps(rotA0, invScaleA);
            __m512 scaledB0 = _mm512_mul_ps(rotB0, invScaleB);
            __m512 scaledA1 = _mm512_mul_ps(rotA1, invScaleA);
            __m512 scaledB1 = _mm512_mul_ps(rotB1, invScaleB);

            __m512i qA0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB0, _MM_FROUND_TO_NEAREST_INT));
            __m512i qA1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledA1, _MM_FROUND_TO_NEAREST_INT));
            __m512i qB1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaledB1, _MM_FROUND_TO_NEAREST_INT));

            qA0 = _mm512_max_epi32(_mm512_min_epi32(qA0, clamp_max), clamp_min);
            qB0 = _mm512_max_epi32(_mm512_min_epi32(qB0, clamp_max), clamp_min);
            qA1 = _mm512_max_epi32(_mm512_min_epi32(qA1, clamp_max), clamp_min);
            qB1 = _mm512_max_epi32(_mm512_min_epi32(qB1, clamp_max), clamp_min);

            // 5. Pack and Store (Interleaved)
            __m256i qA0_16 = _mm512_cvtepi32_epi16(qA0);
            __m256i qB0_16 = _mm512_cvtepi32_epi16(qB0);
            __m256i qA1_16 = _mm512_cvtepi32_epi16(qA1);
            __m256i qB1_16 = _mm512_cvtepi32_epi16(qB1);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockA.qs), qA0_16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockB.qs), qB0_16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockA.qs + 16), qA1_16);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(blockB.qs + 16), qB1_16);

            // 6. Compute Sums (Optimized Reduction)
            // Add vectors first, then reduce (saves 2 reductions)
            int32_t sumA = _mm512_reduce_add_epi32(_mm512_add_epi32(qA0, qA1));
            int32_t sumB = _mm512_reduce_add_epi32(_mm512_add_epi32(qB0, qB1));

            blockA.d = scaleA_new;
            blockB.d = scaleB_new;
            blockA.sum_qs = sumA;
            blockB.sum_qs = sumB;
        }
    }
#endif // __AVX512F__

    void apply_rope_q16_1_integer_head(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15)
    {
        // Dispatch to best available SIMD implementation
#if LLAMINAR_COMPILED_WITH_NEON
        neon_rope::rotate_quantized(head_blocks, blocks_per_head, cos_q15, sin_q15);
#elif defined(__AVX512F__)
        apply_rope_q16_1_integer_head_avx512(head_blocks, blocks_per_head, cos_q15, sin_q15);
#elif defined(__AVX2__)
        apply_rope_q16_1_integer_head_avx2(head_blocks, blocks_per_head, cos_q15, sin_q15);
#else
        apply_rope_q16_1_integer_head_scalar(head_blocks, blocks_per_head, cos_q15, sin_q15);
#endif
    }

    void apply_rope_q16_1_integer(
        Q16_1Block *Q,
        Q16_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta)
    {
        if (head_dim % 32 != 0)
        {
            return; // head_dim must be divisible by Q16_1 block size
        }

        const int blocks_per_head = head_dim / 32;
        const int q_stride_blocks = n_heads * blocks_per_head;
        const int k_stride_blocks = n_kv_heads * blocks_per_head;
        const int half_dim = head_dim / 2;

        // Get inverse frequencies (shared with Q8_1 RoPE)
        const auto &inv_freq = get_inv_freq_cached(head_dim, rope_theta);

        // Precompute sin/cos tables for all positions (Q15 format)
        // We reuse the same Q15 format as Q8_1 since sin/cos are bounded to [-1, 1]
        std::vector<int16_t> cos_table(seq_len * half_dim);
        std::vector<int16_t> sin_table(seq_len * half_dim);

        for (int t = 0; t < seq_len; ++t)
        {
            const int pos = position_ids[t];
            if (pos < 0)
                continue; // Skip padding

            for (int i = 0; i < half_dim; ++i)
            {
                float angle = static_cast<float>(pos) * inv_freq[i];
                float c = std::cos(angle);
                float s = std::sin(angle);
                // Quantize to Q15
                cos_table[t * half_dim + i] = static_cast<int16_t>(std::round(c * 32767.0f));
                sin_table[t * half_dim + i] = static_cast<int16_t>(std::round(s * 32767.0f));
            }
        }

        /*
         * Q and K use independent row/head loops but share one OpenMP team.
         * Keeping the implicit barrier between loops preserves the old memory
         * ordering while removing a second parallel-region launch for every
         * grouped verifier invocation.
         */
        auto do_grouped_work = [&]()
        {
#pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const int pos = position_ids[t];
                    if (pos < 0)
                        continue;

                    Q16_1Block *head_ptr = Q + t * q_stride_blocks + h * blocks_per_head;
                    const int16_t *c_ptr = cos_table.data() + t * half_dim;
                    const int16_t *s_ptr = sin_table.data() + t * half_dim;
                    apply_rope_q16_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                }
            }

            if (K)
            {
#pragma omp for collapse(2) schedule(static)
                for (int t = 0; t < seq_len; ++t)
                {
                    for (int h = 0; h < n_kv_heads; ++h)
                    {
                        const int pos = position_ids[t];
                        if (pos < 0)
                            continue;

                        Q16_1Block *head_ptr = K + t * k_stride_blocks + h * blocks_per_head;
                        const int16_t *c_ptr = cos_table.data() + t * half_dim;
                        const int16_t *s_ptr = sin_table.data() + t * half_dim;
                        apply_rope_q16_1_integer_head(head_ptr, blocks_per_head, c_ptr, s_ptr);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_grouped_work);

    }


} // namespace llaminar2::primitives
