/**
 * @file Test__Q16DecodeHeadMajor.cpp
 * @brief Regression tests for Q16_1 decode attention with HEAD_MAJOR KV layout
 *
 * These tests exercise the unified native-Q16 VNNI attention path with Q16_1
 * tensors arranged in HEAD_MAJOR layout [head][position][head_dim], which is
 * the layout used by CPURingKVCache for Q16_1 KV caches.
 *
 * Regression context: A bug in HEAD_MAJOR addressing caused the per-head
 * block offset to use POSITION_MAJOR formula, producing garbage output
 * for models with n_kv_heads > 1 (e.g., head_dim=64, Qwen2.5-0.5B).
 *
 * Test strategy:
 *   1. Generate random FP32 Q/K/V data in POSITION_MAJOR format
 *   2. Rearrange K/V into HEAD_MAJOR and quantize to Q16_1
 *   3. Also quantize K/V in POSITION_MAJOR Q16_1 for comparison
 *   4. Call compute_tensor() through the production all-M Q16 scheduler
 *   5. Compare both against scalar FP32 reference
 *   6. Compare HEAD_MAJOR vs POSITION_MAJOR to ensure layout equivalence
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "v2/kernels/attention/AttentionDeviceParams.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/CPUFeatures.h"
#include "v2/utils/DebugEnv.h"
#include "v2/utils/PerfStatsCollector.h"
#include "../../../../utils/VerifierRowTestInventory.h"

using namespace llaminar2;

// ---------------------------------------------------------------------------
// Scalar reference (same as Test__CPUFlashAttentionKernelT.cpp)
// ---------------------------------------------------------------------------
namespace ref
{
    static void attention(
        const float *Q, const float *K, const float *V, float *O,
        int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int position_offset,
        int window_size = -1)
    {
        const int heads_per_kv = n_heads / n_kv_heads;
        const int q_stride = n_heads * head_dim;
        const int kv_stride = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < n_heads; ++h)
        {
            const int kv_h = h / heads_per_kv;
            for (int q_pos = 0; q_pos < seq_len; ++q_pos)
            {
                const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                float *out = O + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                const int q_abs = position_offset + q_pos;

                std::vector<float> scores(kv_len);
                for (int k = 0; k < kv_len; ++k)
                {
                    const float *k_ptr = K + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                    const bool masked =
                        (causal && k > q_abs) ||
                        (window_size > 0 &&
                         k < q_abs - window_size + 1);
                    if (masked)
                    {
                        scores[k] = -std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot += q_ptr[d] * k_ptr[d];
                        scores[k] = dot * scale;
                    }
                }

                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_len; ++k)
                {
                    if (std::isfinite(scores[k]))
                    {
                        scores[k] = std::exp(scores[k] - max_s);
                        sum_exp += scores[k];
                    }
                    else
                    {
                        scores[k] = 0.0f;
                    }
                }
                if (sum_exp > 0.0f)
                    for (int k = 0; k < kv_len; ++k)
                        scores[k] /= sum_exp;

                std::fill(out, out + head_dim, 0.0f);
                for (int k = 0; k < kv_len; ++k)
                {
                    if (scores[k] == 0.0f)
                        continue;
                    const float *v_ptr = V + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        out[d] += scores[k] * v_ptr[d];
                }
            }
        }
    }
} // namespace ref

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
namespace
{
    std::mt19937 &rng()
    {
        static std::mt19937 gen(12345);
        return gen;
    }

    void fill_random(float *buf, size_t n, float lo = -1.0f, float hi = 1.0f)
    {
        std::uniform_real_distribution<float> dist(lo, hi);
        auto &g = rng();
        for (size_t i = 0; i < n; ++i)
            buf[i] = dist(g);
    }

    float cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom < 1e-12)
            return 1.0f;
        return static_cast<float>(dot / denom);
    }

    float max_abs_error(const float *a, const float *b, size_t n)
    {
        float mx = 0.0f;
        for (size_t i = 0; i < n; ++i)
            mx = std::max(mx, std::abs(a[i] - b[i]));
        return mx;
    }

    bool is_finite(const float *buf, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(buf[i]))
                return false;
        return true;
    }

    /** @brief Enable route telemetry for grouped Q16 verifier attention tests. */
    class ScopedPerfStats
    {
    public:
        ScopedPerfStats()
        {
            const char *old_value = std::getenv("LLAMINAR_PERF_STATS_SUMMARY");
            if (old_value)
            {
                had_old_value_ = true;
                old_value_ = old_value;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_old_value_)
                setenv("LLAMINAR_PERF_STATS_SUMMARY", old_value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

    private:
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /**
     * @brief Select and later restore one deterministic OpenMP worker count.
     *
     * Physical attention policy depends on the available CPU worker team.
     * Focused tests use this scope so they can prove a particular production
     * route without leaking process-global OpenMP state into later tests.
     */
    class ScopedOpenMPThreadCount
    {
    public:
        /** @param threads Positive worker count exposed to attention policy. */
        explicit ScopedOpenMPThreadCount(int threads)
            : previous_(omp_get_max_threads())
        {
            omp_set_num_threads(threads);
        }

        /** @brief Restore the OpenMP preference observed at construction. */
        ~ScopedOpenMPThreadCount()
        {
            omp_set_num_threads(previous_);
        }

        ScopedOpenMPThreadCount(const ScopedOpenMPThreadCount &) = delete;
        ScopedOpenMPThreadCount &operator=(
            const ScopedOpenMPThreadCount &) = delete;

    private:
        int previous_ = 1; ///< Process preference restored at scope exit.
    };

    /** @brief Report the first bitwise mismatch in a grouped Q16 attention row. */
    void expectByteExactFP32(const float *actual,
                             const float *expected,
                             size_t count,
                             const std::string &context)
    {
        if (std::memcmp(actual, expected, count * sizeof(float)) == 0)
            return;
        for (size_t index = 0; index < count; ++index)
        {
            uint32_t actual_bits = 0;
            uint32_t expected_bits = 0;
            std::memcpy(&actual_bits, actual + index, sizeof(actual_bits));
            std::memcpy(&expected_bits, expected + index, sizeof(expected_bits));
            if (actual_bits != expected_bits)
            {
                ADD_FAILURE() << context << " first byte mismatch at element " << index
                              << " actual=" << actual[index]
                              << " expected=" << expected[index]
                              << " actual_bits=" << actual_bits
                              << " expected_bits=" << expected_bits;
                return;
            }
        }
    }

    /** @brief Assert that grouped Q16 attention used the decode-equivalent tile route. */
    void expectGroupedQ16AttentionCounter(int verifier_rows, int kv_len)
    {
        bool found = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_attention_grouped_verifier_rows_calls"}))
        {
            const auto format = record.tags.find("cache_format");
            const auto rows = record.tags.find("verifier_rows");
            const auto length = record.tags.find("kv_len");
            const auto policy = record.tags.find("tile_policy");
            found = found ||
                    (format != record.tags.end() && format->second == "q16_1" &&
                     rows != record.tags.end() && rows->second == std::to_string(verifier_rows) &&
                     length != record.tags.end() && length->second == std::to_string(kv_len) &&
                     policy != record.tags.end() && policy->second == "serial_decode_equivalent");
        }
        EXPECT_TRUE(found)
            << "Grouped Q16 verifier attention did not publish its production route counter\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cpu_attention_grouped_verifier_rows_calls"}, 20);
    }

    /**
     * @brief Assert the exact physical branch and K/V tile published by FA2.
     * @param query_rows Logical query rows supplied to the grouped scheduler.
     * @param selected_mode Expected physical ownership mode.
     * @param physical_kv_tile Expected compiled cache tile.
     */
    void expectCPUFA2PlanCounter(
        int query_rows,
        const char *selected_mode,
        int physical_kv_tile)
    {
        bool found = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_fa2_parallel_plan_executions"}))
        {
            const auto requested = record.tags.find("requested_axis");
            const auto selected = record.tags.find("selected_mode");
            const auto rows = record.tags.find("query_rows");
            const auto partitions = record.tags.find("context_partitions");
            const auto tile = record.tags.find("physical_kv_tile");
            found = found ||
                    (record.phase == "execute" && record.device == "cpu" &&
                     requested != record.tags.end() &&
                     requested->second == "geometry_selected" &&
                     selected != record.tags.end() &&
                     selected->second == selected_mode &&
                     rows != record.tags.end() &&
                     rows->second == std::to_string(query_rows) &&
                     partitions != record.tags.end() &&
                     std::stoi(partitions->second) > 1 &&
                     tile != record.tags.end() &&
                     tile->second == std::to_string(physical_kv_tile));
        }
        EXPECT_TRUE(found)
            << "CPU FA2 did not publish the expected production physical plan\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cpu_fa2_parallel_plan_executions"}, 20);
    }

    /// Q16_1 quantisation introduces bounded error. Tolerances are looser than FP32
    /// but tight enough to catch addressing bugs (which produce cosine < 0.7).
    constexpr float Q16_COSINE_THRESHOLD = 0.995f;
    constexpr float Q16_MAX_ABS_TOLERANCE = 0.05f;

    /// Scale for Q16_1 quantisation — typical KV cache scale
    constexpr float KV_CACHE_SCALE = 8.0f;

    /**
     * @brief Rearrange FP32 K/V from POSITION_MAJOR to HEAD_MAJOR layout
     *
     * POSITION_MAJOR: [kv_len, n_kv_heads * head_dim]
     *   - Row k contains all heads' data: [head0_d0..head0_dN, head1_d0..head1_dN, ...]
     *
     * HEAD_MAJOR: [n_kv_heads * kv_len, head_dim]
     *   - First kv_len rows are head 0's positions
     *   - Next kv_len rows are head 1's positions, etc.
     *   - Row (h * kv_len + k) = head h, position k, [d0..dN]
     */
    std::vector<float> rearrange_to_head_major(
        const float *pos_major, int kv_len, int n_kv_heads, int head_dim)
    {
        const size_t total = static_cast<size_t>(n_kv_heads) * kv_len * head_dim;
        std::vector<float> head_major(total);

        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int k = 0; k < kv_len; ++k)
            {
                // Source: row k, head h at offset h * head_dim
                const float *src = pos_major + static_cast<size_t>(k) * n_kv_heads * head_dim +
                                   static_cast<size_t>(h) * head_dim;
                // Dest: row (h * kv_len + k), full row is head_dim wide
                float *dst = head_major.data() + (static_cast<size_t>(h) * kv_len + k) * head_dim;
                std::copy(src, src + head_dim, dst);
            }
        }
        return head_major;
    }

    /**
     * @brief Create a Q16_1Tensor from FP32 data in HEAD_MAJOR layout
     *
     * Shape: [n_kv_heads * kv_len, head_dim]
     * Block size chosen to match head_dim (64 → BLOCK_64, 128 → BLOCK_128)
     */
    std::unique_ptr<Q16_1Tensor> create_q16_head_major(
        const float *fp32_pos_major, int kv_len, int n_kv_heads, int head_dim)
    {
        // Rearrange to head-major
        auto hm_data = rearrange_to_head_major(fp32_pos_major, kv_len, n_kv_heads, head_dim);

        // Choose block size matching head_dim
        Q16BlockSize blk_size;
        if (head_dim <= 32)
            blk_size = Q16BlockSize::BLOCK_32;
        else if (head_dim <= 64)
            blk_size = Q16BlockSize::BLOCK_64;
        else
            blk_size = Q16BlockSize::BLOCK_128;

        const size_t rows = static_cast<size_t>(n_kv_heads) * kv_len;
        const size_t cols = static_cast<size_t>(head_dim);
        auto tensor = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{rows, cols}, blk_size);

        bool ok = tensor->copyFrom_fp32_fixed_scale(hm_data.data(), KV_CACHE_SCALE, head_dim);
        if (!ok)
            return nullptr;
        return tensor;
    }

    /**
     * @brief Create a HEAD_MAJOR Q16_1Tensor with extra physical rows per head.
     *
     * Runtime Q16 KV caches are allocated as [kv_head][max_seq_len][head_dim],
     * while decode attention receives a smaller logical kv_len. This helper
     * mirrors that physical ring-cache shape so tests can prove future rows do
     * not influence the currently active prefix.
     */
    std::unique_ptr<Q16_1Tensor> create_q16_head_major_physical(
        const float *fp32_pos_major, int physical_rows_per_head, int n_kv_heads, int head_dim)
    {
        auto hm_data = rearrange_to_head_major(fp32_pos_major, physical_rows_per_head, n_kv_heads, head_dim);

        Q16BlockSize blk_size;
        if (head_dim <= 32)
            blk_size = Q16BlockSize::BLOCK_32;
        else if (head_dim <= 64)
            blk_size = Q16BlockSize::BLOCK_64;
        else
            blk_size = Q16BlockSize::BLOCK_128;

        const size_t rows = static_cast<size_t>(n_kv_heads) * physical_rows_per_head;
        const size_t cols = static_cast<size_t>(head_dim);
        auto tensor = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{rows, cols}, blk_size);

        bool ok = tensor->copyFrom_fp32_fixed_scale(hm_data.data(), KV_CACHE_SCALE, head_dim);
        if (!ok)
            return nullptr;
        return tensor;
    }

    /**
     * @brief Create a Q16_1Tensor from FP32 data in POSITION_MAJOR layout
     *
     * Shape: [kv_len, n_kv_heads * head_dim]
     * Block size chosen to match head_dim
     */
    std::unique_ptr<Q16_1Tensor> create_q16_position_major(
        const float *fp32_pos_major, int kv_len, int n_kv_heads, int head_dim)
    {
        Q16BlockSize blk_size;
        if (head_dim <= 32)
            blk_size = Q16BlockSize::BLOCK_32;
        else if (head_dim <= 64)
            blk_size = Q16BlockSize::BLOCK_64;
        else
            blk_size = Q16BlockSize::BLOCK_128;

        const size_t rows = static_cast<size_t>(kv_len);
        const size_t cols = static_cast<size_t>(n_kv_heads) * head_dim;
        auto tensor = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{rows, cols}, blk_size);

        bool ok = tensor->copyFrom_fp32_fixed_scale(fp32_pos_major, KV_CACHE_SCALE, head_dim);
        if (!ok)
            return nullptr;
        return tensor;
    }

} // anonymous namespace

// ===========================================================================
// Test fixture
// ===========================================================================
class Test__Q16DecodeHeadMajor : public ::testing::Test
{
protected:
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;

    void SetUp() override
    {
        rng().seed(12345);
    }

    /**
     * @brief Call compute_tensor() with Q16_1 K/V tensors for decode
     *
     * Wraps Q/output in FP32Tensor, passes Q16_1 K/V, and dispatches through
     * the public compute_tensor() API which routes to native Q16 VNNI attention.
     * position_offset is computed internally as kv_len - seq_len (= kv_len - 1).
     */
    bool callQ16Decode(
        const float *Q_data, float *out_data,
        const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal = true,
        const attention::AttentionExecutionPolicy &execution_policy = {},
        const attention::AttentionKVLogicalView &kv_logical_view = {},
        int window_size = -1)
    {
        const size_t q_size = static_cast<size_t>(n_heads) * head_dim;

        // Wrap Q in FP32Tensor — shape [1, n_heads * head_dim] (seq_len=1 for decode)
        FP32Tensor Q_tensor({1, q_size});
        std::memcpy(Q_tensor.mutable_data(), Q_data, q_size * sizeof(float));

        // Wrap output in FP32Tensor
        FP32Tensor O_tensor({1, q_size});
        std::memset(O_tensor.mutable_data(), 0, q_size * sizeof(float));

        // seq_len=1 (decode), kv_len > seq_len triggers Q16 decode path
        bool ok = kernel_.compute_tensor(
            &Q_tensor,
            K_q16,
            V_q16,
            &O_tensor,
            /*batch_size=*/1,
            /*seq_len=*/1,
            /*kv_len=*/kv_len,
            n_heads, n_kv_heads, head_dim,
            causal,
            window_size,
            /*workspace_scores=*/nullptr,
            /*workspace_mask=*/nullptr,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*local_n_heads=*/-1,
            /*local_n_kv_heads=*/-1,
            /*gqa_n_rep=*/0,
            execution_policy,
            kv_logical_view);

        if (ok)
        {
            std::memcpy(out_data, O_tensor.data(), q_size * sizeof(float));
        }
        return ok;
    }

    /**
     * @brief Execute native Q16 causal prefill through the public tensor API.
     *
     * The query remains row-major FP32 while K/V use the production head-major
     * Q16 cache layout. Selecting query-sequence ownership keeps this focused
     * unit test independent of the persistent context-parallel workspace; the
     * arithmetic path and cache-derived K/V tile are otherwise the real
     * production implementation.
     *
     * @param query Row-major query matrix with `rows * n_heads * head_dim`
     *        elements.
     * @param output Destination matrix with the same logical shape as query.
     * @param key Native head-major Q16 key cache.
     * @param value Native head-major Q16 value cache.
     * @param rows Positive causal prefill row count and visible K/V length.
     * @param n_heads Local query-head count.
     * @param n_kv_heads Local K/V-head count.
     * @param head_dim Elements in one head.
     * @return True when the production kernel accepts and completes the call.
     */
    bool callQ16CausalPrefill(
        const float *query,
        float *output,
        const Q16_1Tensor *key,
        const Q16_1Tensor *value,
        int rows,
        int n_heads,
        int n_kv_heads,
        int head_dim)
    {
        const std::size_t row_width =
            static_cast<std::size_t>(n_heads) * head_dim;
        const std::size_t element_count =
            static_cast<std::size_t>(rows) * row_width;
        FP32Tensor query_tensor(
            {static_cast<std::size_t>(rows), row_width});
        FP32Tensor output_tensor(
            {static_cast<std::size_t>(rows), row_width});
        std::copy_n(query, element_count, query_tensor.mutable_data());

        const attention::AttentionExecutionPolicy execution_policy{
            .prefill_parallel_axis =
                attention::AttentionPrefillParallelAxis::QuerySequence,
        };
        const bool ok = kernel_.compute_tensor(
            &query_tensor,
            key,
            value,
            &output_tensor,
            /*batch_size=*/1,
            /*seq_len=*/rows,
            /*kv_len=*/rows,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            /*window_size=*/-1,
            /*workspace_scores=*/nullptr,
            /*workspace_mask=*/nullptr,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*local_n_heads=*/-1,
            /*local_n_kv_heads=*/-1,
            /*gqa_n_rep=*/0,
            execution_policy);
        if (ok)
            std::copy_n(output_tensor.data(), element_count, output);
        return ok;
    }

    /**
     * @brief Call the explicit grouped verifier attention hook.
     *
     * MTP verifier graphs use this API directly.  Comparing it to repeated
     * one-row decode calls at this boundary catches grouped-kernel drift before
     * it can hide inside a larger graph parity failure.
     */
    bool callQ16GroupedVerifier(
        const float *Q_data, float *out_data,
        const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
        int verifier_rows, int kv_len, int n_heads, int n_kv_heads, int head_dim,
        const attention::AttentionKVLogicalView &kv_logical_view = {},
        const attention::AttentionExecutionPolicy &execution_policy = {})
    {
        const size_t q_size = static_cast<size_t>(verifier_rows) *
                              static_cast<size_t>(n_heads) *
                              static_cast<size_t>(head_dim);

        FP32Tensor Q_tensor({static_cast<size_t>(verifier_rows),
                             static_cast<size_t>(n_heads) * head_dim});
        std::memcpy(Q_tensor.mutable_data(), Q_data, q_size * sizeof(float));

        FP32Tensor O_tensor({static_cast<size_t>(verifier_rows),
                             static_cast<size_t>(n_heads) * head_dim});
        std::memset(O_tensor.mutable_data(), 0, q_size * sizeof(float));

        const bool ok = kernel_.compute_verifier_rows_decode_equivalent(
            &Q_tensor,
            K_q16,
            V_q16,
            &O_tensor,
            verifier_rows,
            kv_len,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            /*window_size=*/-1,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*gqa_n_rep=*/0,
            kv_logical_view,
            execution_policy);

        if (ok)
        {
            std::memcpy(out_data, O_tensor.data(), q_size * sizeof(float));
        }
        return ok;
    }

    /**
     * @brief Core test: run Q16 decode with HEAD_MAJOR layout, compare to FP32 ref
     */
    void runHeadMajorDecodeTest(
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        const char *label)
    {
        const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
        const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

        std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
        fill_random(Q.data(), q_size);
        fill_random(K_fp32.data(), kv_size);
        fill_random(V_fp32.data(), kv_size);

        auto K_q16_hm = create_q16_head_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto V_q16_hm = create_q16_head_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
        ASSERT_NE(K_q16_hm, nullptr) << label << ": failed to create HEAD_MAJOR K tensor";
        ASSERT_NE(V_q16_hm, nullptr) << label << ": failed to create HEAD_MAJOR V tensor";

        ASSERT_EQ(K_q16_hm->rows(), static_cast<size_t>(n_kv_heads) * kv_len)
            << label << ": K rows mismatch";
        ASSERT_EQ(K_q16_hm->cols(), static_cast<size_t>(head_dim))
            << label << ": K cols mismatch";

        std::vector<float> out_q16(q_size, 0.0f);

        bool ok = callQ16Decode(
            Q.data(), out_q16.data(),
            K_q16_hm.get(), V_q16_hm.get(),
            kv_len, n_heads, n_kv_heads, head_dim);
        ASSERT_TRUE(ok) << label << ": compute_tensor() failed";
        ASSERT_TRUE(is_finite(out_q16.data(), q_size))
            << label << ": Q16 HEAD_MAJOR output has NaN/Inf";

        // Reference uses position_offset = kv_len - 1 (same as compute_tensor internally)
        std::vector<float> out_ref(q_size, 0.0f);
        ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                       1, kv_len, n_heads, n_kv_heads, head_dim,
                       true, kv_len - 1);
        ASSERT_TRUE(is_finite(out_ref.data(), q_size))
            << label << ": reference output has NaN/Inf";

        float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
        float mae = max_abs_error(out_q16.data(), out_ref.data(), q_size);

        EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
            << label << ": HEAD_MAJOR cosine " << cos << " < " << Q16_COSINE_THRESHOLD
            << " (addressing bug likely)";
        EXPECT_LE(mae, Q16_MAX_ABS_TOLERANCE)
            << label << ": HEAD_MAJOR max abs error " << mae << " > " << Q16_MAX_ABS_TOLERANCE;
    }

    /**
     * @brief Compare HEAD_MAJOR vs POSITION_MAJOR Q16 decode outputs
     *
     * Both layouts should produce identical results if addressing is correct.
     */
    void runLayoutEquivalenceTest(
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        const char *label)
    {
        const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
        const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

        std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
        fill_random(Q.data(), q_size);
        fill_random(K_fp32.data(), kv_size);
        fill_random(V_fp32.data(), kv_size);

        auto K_q16_hm = create_q16_head_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto V_q16_hm = create_q16_head_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto K_q16_pm = create_q16_position_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto V_q16_pm = create_q16_position_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
        ASSERT_NE(K_q16_hm, nullptr);
        ASSERT_NE(V_q16_hm, nullptr);
        ASSERT_NE(K_q16_pm, nullptr);
        ASSERT_NE(V_q16_pm, nullptr);

        const int position_offset = kv_len - 1;

        std::vector<float> out_hm(q_size, 0.0f);
        bool ok_hm = callQ16Decode(
            Q.data(), out_hm.data(),
            K_q16_hm.get(), V_q16_hm.get(),
            kv_len, n_heads, n_kv_heads, head_dim);
        ASSERT_TRUE(ok_hm) << label << ": HEAD_MAJOR compute failed";

        std::vector<float> out_pm(q_size, 0.0f);
        bool ok_pm = callQ16Decode(
            Q.data(), out_pm.data(),
            K_q16_pm.get(), V_q16_pm.get(),
            kv_len, n_heads, n_kv_heads, head_dim);
        ASSERT_TRUE(ok_pm) << label << ": POSITION_MAJOR compute failed";

        ASSERT_TRUE(is_finite(out_hm.data(), q_size)) << label << ": HEAD_MAJOR has NaN/Inf";
        ASSERT_TRUE(is_finite(out_pm.data(), q_size)) << label << ": POSITION_MAJOR has NaN/Inf";

        float cos = cosine_similarity(out_hm.data(), out_pm.data(), q_size);
        float mae = max_abs_error(out_hm.data(), out_pm.data(), q_size);

        EXPECT_GE(cos, 0.9999f)
            << label << ": HEAD_MAJOR vs POSITION_MAJOR cosine " << cos
            << " — layouts not equivalent (addressing bug)";
        EXPECT_LE(mae, 0.001f)
            << label << ": HEAD_MAJOR vs POSITION_MAJOR max abs error " << mae;
    }
};

// ===========================================================================
// HEAD_MAJOR decode accuracy vs FP32 reference
// ===========================================================================

// --- head_dim=64 (BLOCK_64) — the configuration that triggered the regression ---

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_2KVHeads_Short)
{
    // Qwen2.5-0.5B config: n_heads=14, n_kv_heads=2, head_dim=64
    runHeadMajorDecodeTest(16, 14, 2, 64, "HD64_2KV_Short");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_2KVHeads_Medium)
{
    runHeadMajorDecodeTest(128, 14, 2, 64, "HD64_2KV_Medium");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_2KVHeads_Long)
{
    runHeadMajorDecodeTest(512, 14, 2, 64, "HD64_2KV_Long");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_4KVHeads)
{
    // 4 KV heads, 8 query heads — exercises GQA with head_dim=64
    runHeadMajorDecodeTest(64, 8, 4, 64, "HD64_4KV");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_8KVHeads)
{
    // 8 KV heads, 16 query heads — no GQA
    runHeadMajorDecodeTest(64, 16, 8, 64, "HD64_8KV");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_OddKVLen)
{
    // Non-power-of-2 kv_len to test tile boundary handling
    runHeadMajorDecodeTest(37, 14, 2, 64, "HD64_OddKV");
}

// --- head_dim=128 (BLOCK_128) — standard Llama/Qwen3 config ---

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_4KVHeads_Short)
{
    // Qwen2.5-7B config: n_heads=28, n_kv_heads=4, head_dim=128
    runHeadMajorDecodeTest(16, 28, 4, 128, "HD128_4KV_Short");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_4KVHeads_Medium)
{
    runHeadMajorDecodeTest(128, 28, 4, 128, "HD128_4KV_Medium");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_4KVHeads_Long)
{
    runHeadMajorDecodeTest(512, 28, 4, 128, "HD128_4KV_Long");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_8KVHeads)
{
    // Qwen3-0.6B: n_heads=16, n_kv_heads=8, head_dim=128
    runHeadMajorDecodeTest(64, 16, 8, 128, "HD128_8KV");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_OddKVLen)
{
    runHeadMajorDecodeTest(37, 28, 4, 128, "HD128_OddKV");
}

// ===========================================================================
// HEAD_MAJOR vs POSITION_MAJOR equivalence
// ===========================================================================

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim64_2KV)
{
    runLayoutEquivalenceTest(64, 14, 2, 64, "Equiv_HD64_2KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim64_4KV)
{
    runLayoutEquivalenceTest(64, 8, 4, 64, "Equiv_HD64_4KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim128_4KV)
{
    runLayoutEquivalenceTest(64, 28, 4, 128, "Equiv_HD128_4KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim128_8KV)
{
    runLayoutEquivalenceTest(64, 16, 8, 128, "Equiv_HD128_8KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_LongContext)
{
    // Longer context to exercise multiple KV tiles
    runLayoutEquivalenceTest(256, 14, 2, 64, "Equiv_Long_HD64");
}

TEST_F(Test__Q16DecodeHeadMajor,
       QueryAndContextPhysicalModesAreByteEquivalent)
{
    constexpr int KV_LEN = 1024;
    constexpr int N_HEADS = 4;
    constexpr int N_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr std::size_t Q_SIZE =
        static_cast<std::size_t>(N_HEADS) * HEAD_DIM;
    constexpr std::size_t KV_SIZE =
        static_cast<std::size_t>(KV_LEN) * N_KV_HEADS * HEAD_DIM;

    rng().seed(918271);
    std::vector<float> q(Q_SIZE);
    std::vector<float> k_fp32(KV_SIZE);
    std::vector<float> v_fp32(KV_SIZE);
    fill_random(q.data(), q.size(), -0.25f, 0.25f);
    fill_random(k_fp32.data(), k_fp32.size(), -0.25f, 0.25f);
    fill_random(v_fp32.data(), v_fp32.size(), -0.25f, 0.25f);

    auto k_q16 = create_q16_head_major(
        k_fp32.data(), KV_LEN, N_KV_HEADS, HEAD_DIM);
    auto v_q16 = create_q16_head_major(
        v_fp32.data(), KV_LEN, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(k_q16, nullptr);
    ASSERT_NE(v_q16, nullptr);

    std::vector<float> query_output(Q_SIZE, 0.0f);
    const attention::AttentionExecutionPolicy query_policy{
        .prefill_parallel_axis =
            attention::AttentionPrefillParallelAxis::QuerySequence,
    };
    ASSERT_TRUE(callQ16Decode(
        q.data(),
        query_output.data(),
        k_q16.get(),
        v_q16.get(),
        KV_LEN,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        /*causal=*/true,
        query_policy));

    const WorkspaceRequirements requirements =
        kernel_.getWorkspaceRequirements(
            attention::kMaxGroupedVerifierAttentionRows,
            N_HEADS,
            HEAD_DIM);
    DeviceWorkspaceManager workspace(
        DeviceId::cpu(),
        requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(requirements));
    kernel_.bindWorkspace(&workspace);

    std::vector<float> context_output(Q_SIZE, 0.0f);
    const attention::AttentionExecutionPolicy context_policy{
        .prefill_parallel_axis =
            attention::AttentionPrefillParallelAxis::KeyValueContext,
    };
    ASSERT_TRUE(callQ16Decode(
        q.data(),
        context_output.data(),
        k_q16.get(),
        v_q16.get(),
        KV_LEN,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        /*causal=*/true,
        context_policy));

    expectByteExactFP32(
        context_output.data(),
        query_output.data(),
        Q_SIZE,
        "Q16 CPU FA2 query/context physical modes");
}

/**
 * @brief Prove the optimized M=1 VNNI path enforces its sliding window.
 *
 * Two caches differ in every row older than the visible window and are byte
 * identical inside it. Equal outputs prove the specialized decoder never reads
 * old K or V. Running the same cache through both physical parallel policies
 * simultaneously locks the canonical partition/merge arithmetic.
 */
TEST_F(Test__Q16DecodeHeadMajor,
       SlidingWindowDecodeIgnoresOlderRowsAndModesRemainByteExact)
{
    constexpr int KV_LEN = 257;
    constexpr int WINDOW_SIZE = 17;
    constexpr int N_HEADS = 4;
    constexpr int N_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr std::size_t Q_SIZE =
        static_cast<std::size_t>(N_HEADS) * HEAD_DIM;
    constexpr std::size_t KV_SIZE =
        static_cast<std::size_t>(KV_LEN) * KV_DIM;

    rng().seed(773104);
    std::vector<float> query(Q_SIZE);
    std::vector<float> key(KV_SIZE);
    std::vector<float> value(KV_SIZE);
    fill_random(query.data(), query.size(), -0.25f, 0.25f);
    fill_random(key.data(), key.size(), -0.25f, 0.25f);
    fill_random(value.data(), value.size(), -0.25f, 0.25f);

    std::vector<float> poisoned_key = key;
    std::vector<float> poisoned_value = value;
    const int first_visible_row = KV_LEN - WINDOW_SIZE;
    for (int row = 0; row < first_visible_row; ++row)
    {
        float *key_row = poisoned_key.data() +
                         static_cast<std::size_t>(row) * KV_DIM;
        float *value_row = poisoned_value.data() +
                           static_cast<std::size_t>(row) * KV_DIM;
        fill_random(key_row, KV_DIM, 0.5f, 0.9f);
        fill_random(value_row, KV_DIM, -0.9f, -0.5f);
    }

    auto native_key = create_q16_head_major(
        key.data(), KV_LEN, N_KV_HEADS, HEAD_DIM);
    auto native_value = create_q16_head_major(
        value.data(), KV_LEN, N_KV_HEADS, HEAD_DIM);
    auto poisoned_native_key = create_q16_head_major(
        poisoned_key.data(), KV_LEN, N_KV_HEADS, HEAD_DIM);
    auto poisoned_native_value = create_q16_head_major(
        poisoned_value.data(), KV_LEN, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(native_key, nullptr);
    ASSERT_NE(native_value, nullptr);
    ASSERT_NE(poisoned_native_key, nullptr);
    ASSERT_NE(poisoned_native_value, nullptr);

    const WorkspaceRequirements requirements =
        kernel_.getWorkspaceRequirements(
            attention::kMaxGroupedVerifierAttentionRows,
            N_HEADS,
            HEAD_DIM);
    DeviceWorkspaceManager workspace(
        DeviceId::cpu(),
        requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(requirements));
    kernel_.bindWorkspace(&workspace);

    const attention::AttentionExecutionPolicy query_policy{
        .prefill_parallel_axis =
            attention::AttentionPrefillParallelAxis::QuerySequence,
    };
    const attention::AttentionExecutionPolicy context_policy{
        .prefill_parallel_axis =
            attention::AttentionPrefillParallelAxis::KeyValueContext,
    };
    std::vector<float> query_output(Q_SIZE, 0.0f);
    std::vector<float> poisoned_output(Q_SIZE, 0.0f);
    std::vector<float> context_output(Q_SIZE, 0.0f);

    ASSERT_TRUE(callQ16Decode(
        query.data(), query_output.data(), native_key.get(), native_value.get(),
        KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM, /*causal=*/true,
        query_policy, {}, WINDOW_SIZE));
    ASSERT_TRUE(callQ16Decode(
        query.data(), poisoned_output.data(), poisoned_native_key.get(),
        poisoned_native_value.get(), KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        /*causal=*/true, query_policy, {}, WINDOW_SIZE));
    ASSERT_TRUE(callQ16Decode(
        query.data(), context_output.data(), native_key.get(), native_value.get(),
        KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM, /*causal=*/true,
        context_policy, {}, WINDOW_SIZE));

    expectByteExactFP32(
        poisoned_output.data(), query_output.data(), Q_SIZE,
        "Q16 sliding-window decode excludes every older native row");
    expectByteExactFP32(
        context_output.data(), query_output.data(), Q_SIZE,
        "Q16 sliding-window query/context physical modes");

    std::vector<float> reference_output(Q_SIZE, 0.0f);
    ref::attention(
        query.data(), key.data(), value.data(), reference_output.data(),
        /*seq_len=*/1, KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        /*causal=*/true, /*position_offset=*/KV_LEN - 1, WINDOW_SIZE);
    EXPECT_GE(
        cosine_similarity(
            query_output.data(), reference_output.data(), Q_SIZE),
        Q16_COSINE_THRESHOLD);
    EXPECT_LE(
        max_abs_error(
            query_output.data(), reference_output.data(), Q_SIZE),
        Q16_MAX_ABS_TOLERANCE);
}

TEST_F(Test__Q16DecodeHeadMajor, HeadMajorPhysicalRowsIgnoreFuturePrefix)
{
    const int logical_kv_len = 37;
    const int physical_rows_per_head = 64;
    const int n_heads = 16;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t logical_kv_size = static_cast<size_t>(logical_kv_len) * n_kv_heads * head_dim;
    const size_t physical_kv_size = static_cast<size_t>(physical_rows_per_head) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_logical(logical_kv_size), V_logical(logical_kv_size);
    std::vector<float> K_physical(physical_kv_size), V_physical(physical_kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_logical.data(), logical_kv_size);
    fill_random(V_logical.data(), logical_kv_size);

    const size_t row_width = static_cast<size_t>(n_kv_heads) * head_dim;
    for (int row = 0; row < physical_rows_per_head; ++row)
    {
        float *k_row = K_physical.data() + static_cast<size_t>(row) * row_width;
        float *v_row = V_physical.data() + static_cast<size_t>(row) * row_width;
        if (row < logical_kv_len)
        {
            std::copy(K_logical.data() + static_cast<size_t>(row) * row_width,
                      K_logical.data() + static_cast<size_t>(row + 1) * row_width,
                      k_row);
            std::copy(V_logical.data() + static_cast<size_t>(row) * row_width,
                      V_logical.data() + static_cast<size_t>(row + 1) * row_width,
                      v_row);
        }
        else
        {
            // Future physical rows should be completely invisible while the
            // logical kv_len is still inside the prefix. Large values make any
            // accidental read show up as a large numerical mismatch.
            fill_random(k_row, row_width, 80.0f, 120.0f);
            fill_random(v_row, row_width, -120.0f, -80.0f);
        }
    }

    auto K_compact = create_q16_head_major(K_logical.data(), logical_kv_len, n_kv_heads, head_dim);
    auto V_compact = create_q16_head_major(V_logical.data(), logical_kv_len, n_kv_heads, head_dim);
    auto K_physical_q16 = create_q16_head_major_physical(
        K_physical.data(), physical_rows_per_head, n_kv_heads, head_dim);
    auto V_physical_q16 = create_q16_head_major_physical(
        V_physical.data(), physical_rows_per_head, n_kv_heads, head_dim);
    ASSERT_NE(K_compact, nullptr);
    ASSERT_NE(V_compact, nullptr);
    ASSERT_NE(K_physical_q16, nullptr);
    ASSERT_NE(V_physical_q16, nullptr);

    std::vector<float> out_compact(q_size, 0.0f);
    std::vector<float> out_physical(q_size, 0.0f);
    ASSERT_TRUE(callQ16Decode(
        Q.data(), out_compact.data(),
        K_compact.get(), V_compact.get(),
        logical_kv_len, n_heads, n_kv_heads, head_dim));
    ASSERT_TRUE(callQ16Decode(
        Q.data(), out_physical.data(),
        K_physical_q16.get(), V_physical_q16.get(),
        logical_kv_len, n_heads, n_kv_heads, head_dim));

    expectByteExactFP32(
        out_physical.data(),
        out_compact.data(),
        q_size,
        "Q16 physical-capacity prefix addressing");
}

TEST_F(Test__Q16DecodeHeadMajor,
       HeadMajorWrappedRingMatchesCompactForDecodeAndGroupedRuntimeM)
{
    constexpr int LOGICAL_KV = 37;
    constexpr int PHYSICAL_CAPACITY = 64;
    constexpr int LOGICAL_ORIGIN = 51;
    constexpr int N_HEADS = 8;
    constexpr int N_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int MAX_M = 15;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr attention::AttentionKVLogicalView WRAPPED_VIEW{
        .logical_row_origin = LOGICAL_ORIGIN,
        .physical_row_capacity = PHYSICAL_CAPACITY,
    };
    static_assert(WRAPPED_VIEW.validFor(LOGICAL_KV));
    static_assert(
        WRAPPED_VIEW.physicalRow(LOGICAL_KV - 1) < LOGICAL_ORIGIN,
        "The test history must cross the physical ring boundary");

    rng().seed(918272);
    std::vector<float> query(static_cast<size_t>(MAX_M) * Q_DIM);
    std::vector<float> compact_key(static_cast<size_t>(LOGICAL_KV) * KV_DIM);
    std::vector<float> compact_value(static_cast<size_t>(LOGICAL_KV) * KV_DIM);
    std::vector<float> physical_key(
        static_cast<size_t>(PHYSICAL_CAPACITY) * KV_DIM);
    std::vector<float> physical_value(
        static_cast<size_t>(PHYSICAL_CAPACITY) * KV_DIM);
    fill_random(query.data(), query.size(), -0.25f, 0.25f);
    fill_random(compact_key.data(), compact_key.size(), -0.25f, 0.25f);
    fill_random(compact_value.data(), compact_value.size(), -0.25f, 0.25f);

    for (size_t element = 0; element < physical_key.size(); ++element)
    {
        physical_key[element] =
            2.0f + static_cast<float>(element % 7U) * 0.03125f;
        physical_value[element] =
            -2.25f - static_cast<float>(element % 5U) * 0.03125f;
    }
    for (int logical_row = 0; logical_row < LOGICAL_KV; ++logical_row)
    {
        const int physical_row = WRAPPED_VIEW.physicalRow(logical_row);
        std::copy_n(
            compact_key.data() + static_cast<size_t>(logical_row) * KV_DIM,
            KV_DIM,
            physical_key.data() + static_cast<size_t>(physical_row) * KV_DIM);
        std::copy_n(
            compact_value.data() + static_cast<size_t>(logical_row) * KV_DIM,
            KV_DIM,
            physical_value.data() + static_cast<size_t>(physical_row) * KV_DIM);
    }

    auto compact_key_q16 = create_q16_head_major(
        compact_key.data(), LOGICAL_KV, N_KV_HEADS, HEAD_DIM);
    auto compact_value_q16 = create_q16_head_major(
        compact_value.data(), LOGICAL_KV, N_KV_HEADS, HEAD_DIM);
    auto physical_key_q16 = create_q16_head_major_physical(
        physical_key.data(), PHYSICAL_CAPACITY, N_KV_HEADS, HEAD_DIM);
    auto physical_value_q16 = create_q16_head_major_physical(
        physical_value.data(), PHYSICAL_CAPACITY, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(compact_key_q16, nullptr);
    ASSERT_NE(compact_value_q16, nullptr);
    ASSERT_NE(physical_key_q16, nullptr);
    ASSERT_NE(physical_value_q16, nullptr);

    const WorkspaceRequirements requirements =
        kernel_.getWorkspaceRequirements(MAX_M, N_HEADS, HEAD_DIM);
    DeviceWorkspaceManager workspace(
        DeviceId::cpu(),
        requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(requirements));
    kernel_.bindWorkspace(&workspace);

    for (const int verifier_rows : {1, 2, 4, 8, 15})
    {
        const size_t output_elements =
            static_cast<size_t>(verifier_rows) * Q_DIM;
        std::vector<float> compact_output(output_elements, 0.0f);
        std::vector<float> physical_output(output_elements, 0.0f);

        if (verifier_rows == 1)
        {
            for (const auto axis : {
                     attention::AttentionPrefillParallelAxis::QuerySequence,
                     attention::AttentionPrefillParallelAxis::KeyValueContext})
            {
                const attention::AttentionExecutionPolicy policy{
                    .prefill_parallel_axis = axis,
                };
                ASSERT_TRUE(callQ16Decode(
                    query.data(),
                    compact_output.data(),
                    compact_key_q16.get(),
                    compact_value_q16.get(),
                    LOGICAL_KV,
                    N_HEADS,
                    N_KV_HEADS,
                    HEAD_DIM,
                    /*causal=*/true,
                    policy));
                ASSERT_TRUE(callQ16Decode(
                    query.data(),
                    physical_output.data(),
                    physical_key_q16.get(),
                    physical_value_q16.get(),
                    LOGICAL_KV,
                    N_HEADS,
                    N_KV_HEADS,
                    HEAD_DIM,
                    /*causal=*/true,
                    policy,
                    WRAPPED_VIEW));
                expectByteExactFP32(
                    physical_output.data(),
                    compact_output.data(),
                    output_elements,
                    std::string("Q16 wrapped decode axis=") +
                        attention::attentionPrefillParallelAxisName(axis));
            }
            continue;
        }

        ASSERT_TRUE(callQ16GroupedVerifier(
            query.data(),
            compact_output.data(),
            compact_key_q16.get(),
            compact_value_q16.get(),
            verifier_rows,
            LOGICAL_KV,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM));
        ASSERT_TRUE(callQ16GroupedVerifier(
            query.data(),
            physical_output.data(),
            physical_key_q16.get(),
            physical_value_q16.get(),
            verifier_rows,
            LOGICAL_KV,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            WRAPPED_VIEW));
        expectByteExactFP32(
            physical_output.data(),
            compact_output.data(),
            output_elements,
            "Q16 wrapped grouped verifier M=" +
                std::to_string(verifier_rows));
    }
}

TEST_F(Test__Q16DecodeHeadMajor,
       NativeRequestBatchWithUnequalWrappedHeadMajorHistoriesIsByteExact)
{
    constexpr int REQUESTS = 3;
    constexpr int PHYSICAL_CAPACITY = 64;
    constexpr int N_HEADS = 8;
    constexpr int N_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    const std::vector<int> kv_lens = {37, 43, 51};
    const std::vector<int> origins = {51, 57, 49};
    std::vector<attention::AttentionKVLogicalView> views(REQUESTS);
    std::vector<std::unique_ptr<Q16_1Tensor>> keys;
    std::vector<std::unique_ptr<Q16_1Tensor>> values;
    keys.reserve(REQUESTS);
    values.reserve(REQUESTS);

    rng().seed(918273);
    for (int request = 0; request < REQUESTS; ++request)
    {
        const int kv_len = kv_lens[static_cast<size_t>(request)];
        const int origin = origins[static_cast<size_t>(request)];
        views[static_cast<size_t>(request)] = {
            .logical_row_origin = origin,
            .physical_row_capacity = PHYSICAL_CAPACITY,
        };
        ASSERT_TRUE(views[static_cast<size_t>(request)].validFor(kv_len));
        ASSERT_LT(
            views[static_cast<size_t>(request)].physicalRow(kv_len - 1),
            origin);

        std::vector<float> logical_key(
            static_cast<size_t>(kv_len) * KV_DIM);
        std::vector<float> logical_value(
            static_cast<size_t>(kv_len) * KV_DIM);
        std::vector<float> physical_key(
            static_cast<size_t>(PHYSICAL_CAPACITY) * KV_DIM);
        std::vector<float> physical_value(
            static_cast<size_t>(PHYSICAL_CAPACITY) * KV_DIM);
        fill_random(logical_key.data(), logical_key.size(), -0.25f, 0.25f);
        fill_random(logical_value.data(), logical_value.size(), -0.25f, 0.25f);
        for (size_t element = 0; element < physical_key.size(); ++element)
        {
            physical_key[element] =
                2.0f + static_cast<float>(element % 7U) * 0.03125f;
            physical_value[element] =
                -2.25f - static_cast<float>(element % 5U) * 0.03125f;
        }
        for (int logical_row = 0; logical_row < kv_len; ++logical_row)
        {
            const int physical_row =
                views[static_cast<size_t>(request)].physicalRow(logical_row);
            std::copy_n(
                logical_key.data() + static_cast<size_t>(logical_row) * KV_DIM,
                KV_DIM,
                physical_key.data() + static_cast<size_t>(physical_row) * KV_DIM);
            std::copy_n(
                logical_value.data() + static_cast<size_t>(logical_row) * KV_DIM,
                KV_DIM,
                physical_value.data() + static_cast<size_t>(physical_row) * KV_DIM);
        }
        keys.push_back(create_q16_head_major_physical(
            physical_key.data(),
            PHYSICAL_CAPACITY,
            N_KV_HEADS,
            HEAD_DIM));
        values.push_back(create_q16_head_major_physical(
            physical_value.data(),
            PHYSICAL_CAPACITY,
            N_KV_HEADS,
            HEAD_DIM));
        ASSERT_NE(keys.back(), nullptr);
        ASSERT_NE(values.back(), nullptr);
    }

    std::vector<const ITensor *> key_views(REQUESTS);
    std::vector<const ITensor *> value_views(REQUESTS);
    for (int request = 0; request < REQUESTS; ++request)
    {
        key_views[static_cast<size_t>(request)] =
            keys[static_cast<size_t>(request)].get();
        value_views[static_cast<size_t>(request)] =
            values[static_cast<size_t>(request)].get();
    }

    const WorkspaceRequirements requirements =
        kernel_.getWorkspaceRequirements(
            REQUESTS * 15,
            N_HEADS,
            HEAD_DIM);
    DeviceWorkspaceManager workspace(
        DeviceId::cpu(),
        requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(requirements));
    kernel_.bindWorkspace(&workspace);
    const attention::AttentionExecutionPolicy execution_policy{
        .prefill_parallel_axis =
            attention::AttentionPrefillParallelAxis::KeyValueContext,
    };

    for (const int query_rows : {1, 4, 15})
    {
        const size_t request_elements =
            static_cast<size_t>(query_rows) * Q_DIM;
        const size_t total_elements =
            static_cast<size_t>(REQUESTS) * request_elements;
        std::vector<float> query_data(total_elements);
        fill_random(query_data.data(), query_data.size(), -0.25f, 0.25f);
        FP32Tensor query(
            {static_cast<size_t>(REQUESTS * query_rows),
             static_cast<size_t>(Q_DIM)});
        std::copy(
            query_data.begin(), query_data.end(), query.mutable_data());
        FP32Tensor grouped_output(
            {static_cast<size_t>(REQUESTS * query_rows),
             static_cast<size_t>(Q_DIM)});
        std::vector<float> serial_output(total_elements, 0.0f);

        ASSERT_TRUE(kernel_.compute_request_batch_decode_equivalent(
            &query,
            key_views.data(),
            value_views.data(),
            kv_lens.data(),
            &grouped_output,
            REQUESTS,
            query_rows,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            /*causal=*/true,
            /*window_size=*/-1,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*gqa_n_rep=*/0,
            execution_policy,
            views.data()))
            << "M=" << query_rows;

        for (int request = 0; request < REQUESTS; ++request)
        {
            const float *request_query =
                query_data.data() + static_cast<size_t>(request) * request_elements;
            float *request_output =
                serial_output.data() + static_cast<size_t>(request) * request_elements;
            if (query_rows == 1)
            {
                ASSERT_TRUE(callQ16Decode(
                    request_query,
                    request_output,
                    keys[static_cast<size_t>(request)].get(),
                    values[static_cast<size_t>(request)].get(),
                    kv_lens[static_cast<size_t>(request)],
                    N_HEADS,
                    N_KV_HEADS,
                    HEAD_DIM,
                    /*causal=*/true,
                    execution_policy,
                    views[static_cast<size_t>(request)]));
            }
            else
            {
                ASSERT_TRUE(callQ16GroupedVerifier(
                    request_query,
                    request_output,
                    keys[static_cast<size_t>(request)].get(),
                    values[static_cast<size_t>(request)].get(),
                    query_rows,
                    kv_lens[static_cast<size_t>(request)],
                    N_HEADS,
                    N_KV_HEADS,
                    HEAD_DIM,
                    views[static_cast<size_t>(request)],
                    execution_policy));
            }
        }

        expectByteExactFP32(
            grouped_output.data(),
            serial_output.data(),
            total_elements,
            "Q16 native unequal request batch M=" +
                std::to_string(query_rows));
    }
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST_F(Test__Q16DecodeHeadMajor, SingleKVPosition)
{
    // kv_len=1: degenerate case, attention should output V directly (scaled)
    runHeadMajorDecodeTest(1, 4, 2, 64, "SingleKV_HD64");
}

TEST_F(Test__Q16DecodeHeadMajor, SingleKVPosition_HD128)
{
    runHeadMajorDecodeTest(1, 4, 2, 128, "SingleKV_HD128");
}

TEST_F(Test__Q16DecodeHeadMajor, GroupedVerifierRowsMatchSerialDecode_RuntimeM)
{
    ScopedPerfStats perfstats;

    /*
     * Production Q16 KV caches are head-major, while the MTP verifier presents
     * a compact group of candidate rows.  Row r in the group must match serial
     * decode at BASE_KV + r + 1 visible KV positions.
     */
    constexpr int BASE_KV = 19;
    constexpr int MAX_M = test::kGroupedVerifierRuntimeRows.back();
    constexpr int FULL_KV = BASE_KV + MAX_M;
    constexpr int N_HEADS = 8;
    constexpr int N_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    rng().seed(88001);
    std::vector<float> Q_all(static_cast<size_t>(MAX_M) * Q_DIM);
    std::vector<float> K_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    std::vector<float> V_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    fill_random(Q_all.data(), Q_all.size(), -0.75f, 0.75f);
    fill_random(K_fp32.data(), K_fp32.size(), -0.5f, 0.5f);
    fill_random(V_fp32.data(), V_fp32.size(), -0.5f, 0.5f);

    auto K_q16 = create_q16_head_major(K_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    auto V_q16 = create_q16_head_major(V_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    for (int m : test::kGroupedVerifierRuntimeRows)
    {
        const int kv_len = BASE_KV + m;
        std::vector<float> grouped(static_cast<size_t>(m) * Q_DIM, 0.0f);
        PerfStatsCollector::reset();
        ASSERT_TRUE(callQ16GroupedVerifier(
            Q_all.data(),
            grouped.data(),
            K_q16.get(),
            V_q16.get(),
            m,
            kv_len,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM))
            << "M=" << m;
        expectGroupedQ16AttentionCounter(m, kv_len);

        for (int row = 0; row < m; ++row)
        {
            std::vector<float> serial(Q_DIM, 0.0f);
            ASSERT_TRUE(callQ16Decode(
                Q_all.data() + static_cast<size_t>(row) * Q_DIM,
                serial.data(),
                K_q16.get(),
                V_q16.get(),
                BASE_KV + row + 1,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM))
                << "M=" << m << " row=" << row;

            const float *grouped_row = grouped.data() + static_cast<size_t>(row) * Q_DIM;
            expectByteExactFP32(
                grouped_row,
                serial.data(),
                Q_DIM,
                "Q16 grouped verifier attention M=" + std::to_string(m) +
                    " row=" + std::to_string(row));
        }
    }
}

/**
 * @brief Prove geometry-selected grouped Q16 attention across a summary edge.
 *
 * This is the production physical-policy route, not an explicitly pinned
 * query-sequence oracle. The first verifier row ends exactly on a canonical
 * 256-row summary boundary and the second row opens the next summary. That
 * geometry catches grouped implementations that derive a different reduction
 * tree from M than repeated one-token decode. Every compiled physical cache
 * tile is exercised; each must match both the smallest-tile byte oracle and
 * serial decode. All calls retain the optimized native-Q16 VNNI implementation.
 */
TEST_F(Test__Q16DecodeHeadMajor,
       GeometrySelectedLongContextGroupedVerifierMatchesSerialDecode)
{
    ScopedPerfStats perfstats;
    ScopedOpenMPThreadCount threads(/*threads=*/8);

    constexpr int BASE_KV = 1023;
    constexpr int VERIFIER_ROWS = 2;
    constexpr int FULL_KV = BASE_KV + VERIFIER_ROWS;
    constexpr int N_HEADS = 2;
    constexpr int N_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    const attention::AttentionExecutionPolicy execution_policy{
        .prefill_parallel_axis =
            attention::AttentionPrefillParallelAxis::GeometrySelected,
    };

    const auto grouped_plan = cpu::fa2_policy::selectCPUFA2ParallelPlan({
        .batch_size = 1,
        .query_rows = VERIFIER_ROWS,
        .local_query_heads = N_HEADS,
        .kv_rows = FULL_KV,
        .physical_workers = omp_get_max_threads(),
        .requested_axis = execution_policy.prefill_parallel_axis,
    });
    ASSERT_TRUE(grouped_plan.usesContextParallelism());
    for (int row = 0; row < VERIFIER_ROWS; ++row)
    {
        const auto serial_plan = cpu::fa2_policy::selectCPUFA2ParallelPlan({
            .batch_size = 1,
            .query_rows = 1,
            .local_query_heads = N_HEADS,
            .kv_rows = BASE_KV + row + 1,
            .physical_workers = omp_get_max_threads(),
            .requested_axis = execution_policy.prefill_parallel_axis,
        });
        ASSERT_TRUE(serial_plan.usesContextParallelism()) << "row=" << row;
    }

    rng().seed(88002);
    std::vector<float> query(
        static_cast<std::size_t>(VERIFIER_ROWS) * Q_DIM);
    std::vector<float> key_rows(
        static_cast<std::size_t>(FULL_KV) * KV_DIM);
    std::vector<float> value_rows(
        static_cast<std::size_t>(FULL_KV) * KV_DIM);
    fill_random(query.data(), query.size(), -0.5f, 0.5f);
    fill_random(key_rows.data(), key_rows.size(), -0.5f, 0.5f);
    fill_random(value_rows.data(), value_rows.size(), -0.5f, 0.5f);

    auto key = create_q16_head_major(
        key_rows.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    auto value = create_q16_head_major(
        value_rows.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);

    const WorkspaceRequirements requirements =
        kernel_.getWorkspaceRequirements(
            VERIFIER_ROWS,
            N_HEADS,
            HEAD_DIM);
    DeviceWorkspaceManager workspace(
        DeviceId::cpu(),
        requirements.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(requirements));
    kernel_.bindWorkspace(&workspace);

    std::vector<float> grouped_output(
        static_cast<std::size_t>(VERIFIER_ROWS) * Q_DIM,
        0.0f);
    std::vector<float> reference_grouped;
    for (const int tile : cpu::fa2_policy::kCompiledKVTiles)
    {
        kernel_.configureLaunchPolicy({.explicit_kv_tile = tile});
        ASSERT_TRUE(callQ16GroupedVerifier(
            query.data(),
            grouped_output.data(),
            key.get(),
            value.get(),
            VERIFIER_ROWS,
            FULL_KV,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            /*kv_logical_view=*/{},
            execution_policy))
            << "tile=" << tile;
        expectCPUFA2PlanCounter(
            VERIFIER_ROWS,
            "key_value_context",
            tile);

        if (reference_grouped.empty())
        {
            reference_grouped = grouped_output;
        }
        else
        {
            expectByteExactFP32(
                grouped_output.data(),
                reference_grouped.data(),
                grouped_output.size(),
                "Q16 physical tile=" + std::to_string(tile));
        }

        for (int row = 0; row < VERIFIER_ROWS; ++row)
        {
            std::vector<float> serial_output(Q_DIM, 0.0f);
            ASSERT_TRUE(callQ16Decode(
                query.data() + static_cast<std::size_t>(row) * Q_DIM,
                serial_output.data(),
                key.get(),
                value.get(),
                BASE_KV + row + 1,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM,
                /*causal=*/true,
                execution_policy))
                << "tile=" << tile << " row=" << row;
            expectByteExactFP32(
                grouped_output.data() + static_cast<std::size_t>(row) * Q_DIM,
                serial_output.data(),
                Q_DIM,
                "Q16 geometry-selected grouped verifier tile=" +
                    std::to_string(tile) + " row=" + std::to_string(row));
        }
    }
}

/**
 * @brief Lock ordinary Q16 causal prefill to the serial-decode byte contract.
 *
 * Values immediately around compiled K/V tile sizes and the canonical
 * 256-row context-summary span ensure cache-policy tuning cannot silently
 * install a different floating-point order for prefill. Both sides invoke the
 * optimized native-Q16 production implementation; repeated M=1 calls are a
 * test oracle only and are never reachable from production dispatch.
 */
TEST_F(Test__Q16DecodeHeadMajor,
       CausalPrefillMatchesSerialDecodeAcrossTileBoundaries)
{
    constexpr int maximum_rows = 257;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int query_width = n_heads * head_dim;
    constexpr int kv_width = n_kv_heads * head_dim;
    constexpr std::array<int, 12> row_counts{
        1, 2, 15, 31, 32, 33, 127, 128, 129, 255, 256, 257};

    rng().seed(88004);
    std::vector<float> query(
        static_cast<std::size_t>(maximum_rows) * query_width);
    std::vector<float> key_rows(
        static_cast<std::size_t>(maximum_rows) * kv_width);
    std::vector<float> value_rows(
        static_cast<std::size_t>(maximum_rows) * kv_width);
    fill_random(query.data(), query.size(), -0.4f, 0.4f);
    fill_random(key_rows.data(), key_rows.size(), -0.4f, 0.4f);
    fill_random(value_rows.data(), value_rows.size(), -0.4f, 0.4f);

    auto key = create_q16_head_major(
        key_rows.data(), maximum_rows, n_kv_heads, head_dim);
    auto value = create_q16_head_major(
        value_rows.data(), maximum_rows, n_kv_heads, head_dim);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);

    /*
     * Serial row r depends only on query r and K/V prefix [0, r]. Compute that
     * immutable oracle once, then reuse its prefixes for every prefill launch.
     * This removes redundant kernel invocations without reducing M coverage.
     */
    std::vector<float> serial(
        static_cast<std::size_t>(maximum_rows) * query_width);
    for (int row = 0; row < maximum_rows; ++row)
    {
        ASSERT_TRUE(callQ16Decode(
            query.data() + static_cast<std::size_t>(row) * query_width,
            serial.data() + static_cast<std::size_t>(row) * query_width,
            key.get(),
            value.get(),
            row + 1,
            n_heads,
            n_kv_heads,
            head_dim))
            << "serial row=" << row;
    }

    for (const int rows : row_counts)
    {
        const std::size_t element_count =
            static_cast<std::size_t>(rows) * query_width;
        std::vector<float> prefill(element_count);
        ASSERT_TRUE(callQ16CausalPrefill(
            query.data(),
            prefill.data(),
            key.get(),
            value.get(),
            rows,
            n_heads,
            n_kv_heads,
            head_dim))
            << "M=" << rows;

        expectByteExactFP32(
            prefill.data(),
            serial.data(),
            element_count,
            "Q16 causal prefill M=" + std::to_string(rows));
    }
}

/**
 * @brief Prove the grouped kernel consumes native head-major Q without a transpose.
 *
 * Hybrid Q16 RoPE publishes compact verifier queries as
 * `[head][verifier_row][dim]`. The ordinary row-major test above cannot catch
 * an indexing error or the reintroduction of a transient layout conversion.
 * This case therefore supplies a real Q16 tensor in the production physical
 * layout and reconstructs each serial decode query only in the test oracle.
 */
TEST_F(Test__Q16DecodeHeadMajor, GroupedVerifierConsumesHeadMajorQuery_RuntimeM)
{
    constexpr int BASE_KV = 23;
    constexpr int MAX_M = test::kGroupedVerifierRuntimeRows.back();
    constexpr int FULL_KV = BASE_KV + MAX_M;
    constexpr int N_HEADS = 8;
    constexpr int N_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    rng().seed(88003);
    std::vector<float> query_rows(static_cast<size_t>(MAX_M) * Q_DIM);
    std::vector<float> key_rows(static_cast<size_t>(FULL_KV) * KV_DIM);
    std::vector<float> value_rows(static_cast<size_t>(FULL_KV) * KV_DIM);
    fill_random(query_rows.data(), query_rows.size(), -0.5f, 0.5f);
    fill_random(key_rows.data(), key_rows.size(), -0.5f, 0.5f);
    fill_random(value_rows.data(), value_rows.size(), -0.5f, 0.5f);

    auto key = create_q16_head_major(
        key_rows.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    auto value = create_q16_head_major(
        value_rows.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);

    for (int m : test::kGroupedVerifierRuntimeRows)
    {
        auto query = create_q16_head_major(
            query_rows.data(), m, N_HEADS, HEAD_DIM);
        ASSERT_NE(query, nullptr) << "M=" << m;

        /*
         * Materialize the immutable test tensor before entering the kernel.
         * Production Q16 activations already own this FP32 query plane because
         * RoPE writes it in place; the explicit access here gives the oracle
         * the exact quantized values consumed by the grouped call.
         */
        const float *head_major_query = query->data();
        ASSERT_NE(head_major_query, nullptr) << "M=" << m;

        FP32Tensor grouped_output(
            {static_cast<size_t>(m), static_cast<size_t>(Q_DIM)});
        ASSERT_TRUE(kernel_.compute_verifier_rows_decode_equivalent(
            query.get(),
            key.get(),
            value.get(),
            &grouped_output,
            m,
            BASE_KV + m,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            /*causal=*/true))
            << "M=" << m;

        for (int row = 0; row < m; ++row)
        {
            std::vector<float> serial_query(Q_DIM);
            for (int head = 0; head < N_HEADS; ++head)
            {
                const float *source =
                    head_major_query +
                    (static_cast<size_t>(head) * m + row) * HEAD_DIM;
                std::copy_n(
                    source,
                    HEAD_DIM,
                    serial_query.data() +
                        static_cast<size_t>(head) * HEAD_DIM);
            }

            std::vector<float> serial_output(Q_DIM, 0.0f);
            ASSERT_TRUE(callQ16Decode(
                serial_query.data(),
                serial_output.data(),
                key.get(),
                value.get(),
                BASE_KV + row + 1,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM))
                << "M=" << m << " row=" << row;

            expectByteExactFP32(
                grouped_output.data() + static_cast<size_t>(row) * Q_DIM,
                serial_output.data(),
                Q_DIM,
                "Q16 head-major Q grouped verifier M=" +
                    std::to_string(m) + " row=" + std::to_string(row));
        }
    }
}

TEST_F(Test__Q16DecodeHeadMajor, GroupedVerifierRowsMatchSerialDecode_Qwen36Shape_RuntimeBoundaries)
{
    ScopedPerfStats perfstats;

    /*
     * Qwen3.6 dense attention uses a wider head layout than the small unit
     * case above.  This regression keeps the grouped verifier path honest for
     * the model shape that exposed the first real parity drift.
     */
    constexpr int BASE_KV = 37;
    constexpr int MAX_M = test::kGroupedVerifierBoundaryRows.back();
    constexpr int FULL_KV = BASE_KV + MAX_M;
    constexpr int N_HEADS = 28;
    constexpr int N_KV_HEADS = 4;
    constexpr int HEAD_DIM = 128;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    rng().seed(88002);
    std::vector<float> Q_all(static_cast<size_t>(MAX_M) * Q_DIM);
    std::vector<float> K_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    std::vector<float> V_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    fill_random(Q_all.data(), Q_all.size(), -0.25f, 0.25f);
    fill_random(K_fp32.data(), K_fp32.size(), -0.25f, 0.25f);
    fill_random(V_fp32.data(), V_fp32.size(), -0.25f, 0.25f);

    auto K_q16 = create_q16_head_major(K_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    auto V_q16 = create_q16_head_major(V_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    for (int m : test::kGroupedVerifierBoundaryRows)
    {
        const int kv_len = BASE_KV + m;
        std::vector<float> grouped(static_cast<size_t>(m) * Q_DIM, 0.0f);
        PerfStatsCollector::reset();
        ASSERT_TRUE(callQ16GroupedVerifier(
            Q_all.data(),
            grouped.data(),
            K_q16.get(),
            V_q16.get(),
            m,
            kv_len,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM))
            << "M=" << m;
        expectGroupedQ16AttentionCounter(m, kv_len);

        for (int row = 0; row < m; ++row)
        {
            std::vector<float> serial(Q_DIM, 0.0f);
            ASSERT_TRUE(callQ16Decode(
                Q_all.data() + static_cast<size_t>(row) * Q_DIM,
                serial.data(),
                K_q16.get(),
                V_q16.get(),
                BASE_KV + row + 1,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM))
                << "M=" << m << " row=" << row;

            const float *grouped_row = grouped.data() + static_cast<size_t>(row) * Q_DIM;
            expectByteExactFP32(
                grouped_row,
                serial.data(),
                Q_DIM,
                "Q16 Qwen3.6 grouped verifier attention M=" + std::to_string(m) +
                    " row=" + std::to_string(row));
        }
    }
}

TEST_F(Test__Q16DecodeHeadMajor, NoCausal)
{
    // Non-causal attention with HEAD_MAJOR layout
    const int kv_len = 64;
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_fp32.data(), kv_size);
    fill_random(V_fp32.data(), kv_size);

    auto K_q16 = create_q16_head_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
    auto V_q16 = create_q16_head_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    std::vector<float> out_q16(q_size, 0.0f);
    bool ok = callQ16Decode(
        Q.data(), out_q16.data(),
        K_q16.get(), V_q16.get(),
        kv_len, n_heads, n_kv_heads, head_dim,
        /*causal=*/false);
    ASSERT_TRUE(ok) << "NoCausal: compute_tensor() failed";
    ASSERT_TRUE(is_finite(out_q16.data(), q_size)) << "NoCausal: output has NaN/Inf";

    // For non-causal, position_offset doesn't affect results
    std::vector<float> out_ref(q_size, 0.0f);
    ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                   1, kv_len, n_heads, n_kv_heads, head_dim,
                   false, kv_len - 1);

    float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
    EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
        << "NoCausal: cosine " << cos << " below threshold";
}

// ===========================================================================
// MQA (Multi-Query Attention) — single KV head
// Note: n_kv_heads=1 does not trigger HEAD_MAJOR detection (is_head_major
// requires n_kv_heads > 1), but it's important to verify it still works.
// ===========================================================================

TEST_F(Test__Q16DecodeHeadMajor, MQA_SingleKVHead_HD64)
{
    // MQA: 8 query heads, 1 KV head
    // With n_kv_heads=1, blocks_per_kv_row == blocks_per_head regardless of
    // layout, so the is_head_major heuristic returns false. The addressing
    // is identical for both layouts when n_kv_heads=1.
    const int kv_len = 64;
    const int n_heads = 8;
    const int n_kv_heads = 1;
    const int head_dim = 64;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_fp32.data(), kv_size);
    fill_random(V_fp32.data(), kv_size);

    // For n_kv_heads=1, HEAD_MAJOR and POSITION_MAJOR are the same shape
    auto K_q16 = create_q16_position_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
    auto V_q16 = create_q16_position_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    std::vector<float> out_q16(q_size, 0.0f);
    bool ok = callQ16Decode(
        Q.data(), out_q16.data(),
        K_q16.get(), V_q16.get(),
        kv_len, n_heads, n_kv_heads, head_dim);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(is_finite(out_q16.data(), q_size));

    std::vector<float> out_ref(q_size, 0.0f);
    ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                   1, kv_len, n_heads, n_kv_heads, head_dim,
                   true, kv_len - 1);

    float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
    EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
        << "MQA_HD64: cosine " << cos << " below threshold";
}

TEST_F(Test__Q16DecodeHeadMajor, MQA_SingleKVHead_HD128)
{
    const int kv_len = 64;
    const int n_heads = 8;
    const int n_kv_heads = 1;
    const int head_dim = 128;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_fp32.data(), kv_size);
    fill_random(V_fp32.data(), kv_size);

    auto K_q16 = create_q16_position_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
    auto V_q16 = create_q16_position_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    std::vector<float> out_q16(q_size, 0.0f);
    bool ok = callQ16Decode(
        Q.data(), out_q16.data(),
        K_q16.get(), V_q16.get(),
        kv_len, n_heads, n_kv_heads, head_dim);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(is_finite(out_q16.data(), q_size));

    std::vector<float> out_ref(q_size, 0.0f);
    ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                   1, kv_len, n_heads, n_kv_heads, head_dim,
                   true, kv_len - 1);

    float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
    EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
        << "MQA_HD128: cosine " << cos << " below threshold";
}
