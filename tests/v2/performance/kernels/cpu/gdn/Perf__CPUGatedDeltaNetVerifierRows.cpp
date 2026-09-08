/**
 * @file Perf__CPUGatedDeltaNetVerifierRows.cpp
 * @brief Production-shaped CPU GDN grouped-verifier economy benchmark.
 *
 * This harness compares one grouped merged-QKV verifier transaction with the
 * exact production M=1 merged-QKV recurrence repeated in serial row order.  It
 * includes direct post-row snapshot materialization because it is required by
 * MTP state commitment. The optimized verifier writes recurrence results into
 * those slots directly, so no speculative clone or post-row copy is permitted.
 * Vector construction, input generation, capacity growth, and state reset stay
 * outside canonical timing samples.
 *
 * Correctness is a byte contract.  Every grouped output row and every published
 * recurrence-state snapshot must match the corresponding production M=1 row.
 * The default inventory contains M=1 as a control, every supported MTP verifier
 * row count, and the canonical extended runtime-M probes.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Timing and geometry evidence emitted for one runtime-M case. */
    struct BenchmarkResult
    {
        int rows = 0;
        int n_k_heads = 0;
        int n_heads = 0;
        int d_k = 0;
        int d_v = 0;
        double grouped_ms = 0.0;
        double serial_ms = 0.0;
        double speedup = 0.0;
    };

    /** @brief Parse one strictly positive integer environment override. */
    int envInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;

        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max())
        {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    /** @brief Parse one strictly positive floating-point environment override. */
    double envDouble(const char *name, double fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;

        char *end = nullptr;
        const double parsed = std::strtod(raw, &end);
        if (end == raw || *end != '\0' || parsed <= 0.0)
            return fallback;
        return parsed;
    }

    /**
     * @brief Select runtime-M cases without imposing an artificial upper bound.
     *
     * An explicit comma-separated override is useful for profiler runs.  The
     * ordinary benchmark uses the shared complete verifier inventory, preceded
     * by M=1 so the serial boundary remains visible.
     */
    std::vector<int> envRows()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_GDN_VERIFIER_M");
        if (!raw || !*raw)
        {
            std::vector<int> rows = {1};
            rows.insert(
                rows.end(),
                test::kGroupedVerifierRuntimeRows.begin(),
                test::kGroupedVerifierRuntimeRows.end());
            return rows;
        }

        std::vector<int> rows;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            char *end = nullptr;
            const long parsed = std::strtol(token.c_str(), &end, 10);
            if (end != token.c_str() && *end == '\0' && parsed > 0 &&
                parsed <= std::numeric_limits<int>::max())
            {
                rows.push_back(static_cast<int>(parsed));
            }
        }

        if (rows.empty())
            rows = {1, 2, 3, 4, 8, 15, 16};
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        return rows;
    }

    /**
     * @brief Report the first FP32 mismatch after an exact byte comparison.
     */
    void expectByteExact(
        const float *actual,
        const float *expected,
        size_t count,
        const std::string &context)
    {
        if (std::memcmp(actual, expected, count * sizeof(float)) == 0)
            return;

        for (size_t index = 0; index < count; ++index)
        {
            if (std::memcmp(actual + index, expected + index, sizeof(float)) != 0)
            {
                uint32_t actual_bits = 0;
                uint32_t expected_bits = 0;
                std::memcpy(&actual_bits, actual + index, sizeof(actual_bits));
                std::memcpy(&expected_bits, expected + index, sizeof(expected_bits));
                ADD_FAILURE() << context << " first mismatch at element " << index
                              << " actual=" << actual[index]
                              << " expected=" << expected[index]
                              << " actual_bits=" << actual_bits
                              << " expected_bits=" << expected_bits;
                return;
            }
        }
    }

    /**
     * @brief Own all persistent buffers and kernels for one measured geometry.
     *
     * The object is constructed before warmup and never changes capacity during
     * a timing sample.  Q/K/V use the same merged row layout consumed by the real
     * Qwen GDN recurrence stage.  Separate kernel instances prevent scratch or
     * speculative workspace from leaking between grouped and serial samples.
     */
    class GDNVerifierCase
    {
    public:
        GDNVerifierCase(
            int rows,
            int n_k_heads,
            int n_heads,
            int d_k,
            int d_v)
            : rows_(rows),
              n_k_heads_(n_k_heads),
              n_heads_(n_heads),
              d_k_(d_k),
              d_v_(d_v),
              q_src_dim_(n_k_heads * d_k),
              k_src_dim_(n_k_heads * d_k),
              value_dim_(n_heads * d_v),
              qkv_stride_(q_src_dim_ + k_src_dim_ + value_dim_),
              state_floats_(n_heads * d_k * d_v),
              merged_qkv_(static_cast<size_t>(rows) * qkv_stride_),
              alpha_(static_cast<size_t>(rows) * n_heads),
              beta_(static_cast<size_t>(rows) * n_heads),
              a_log_(n_heads),
              dt_bias_(n_heads),
              initial_state_(state_floats_),
              grouped_live_state_(state_floats_),
              serial_state_(state_floats_),
              grouped_output_(static_cast<size_t>(rows) * value_dim_),
              serial_output_(static_cast<size_t>(rows) * value_dim_),
              grouped_snapshots_(static_cast<size_t>(rows) * state_floats_),
              serial_snapshots_(static_cast<size_t>(rows) * state_floats_)
        {
            fillInputs();
            resetGrouped();
            resetSerial();
        }

        /** @brief Restore the grouped live-state source outside measured time. */
        void resetGrouped()
        {
            std::memcpy(
                grouped_live_state_.data(),
                initial_state_.data(),
                static_cast<size_t>(state_floats_) * sizeof(float));
        }

        /** @brief Restore serial production state outside measured time. */
        void resetSerial()
        {
            std::memcpy(
                serial_state_.data(),
                initial_state_.data(),
                static_cast<size_t>(state_floats_) * sizeof(float));
        }

        /**
         * @brief Run one grouped production-ABI verifier transaction.
         *
         * Each immutable post-row snapshot is the direct recurrence destination;
         * no live-to-work clone or post-row copy belongs in this path.
         */
        bool runGrouped()
        {
            return grouped_kernel_.chunkForwardMergedQKVWithStateSnapshots(
                merged_qkv_.data(),
                qkv_stride_,
                alpha_.data(),
                beta_.data(),
                a_log_.data(),
                dt_bias_.data(),
                grouped_output_.data(),
                grouped_live_state_.data(),
                rows_,
                n_k_heads_,
                n_heads_,
                d_k_,
                d_v_,
                /*global_v_head_offset=*/0,
                /*chunk_size=*/64,
                /*use_qk_l2norm=*/true,
                grouped_snapshots_.data(),
                state_floats_,
                rows_);
        }

        /**
         * @brief Run the actual production M=1 merged-QKV path for every row.
         *
         * @param publish_snapshots Whether to copy each post-row state for the
         *        correctness oracle. Canonical timing disables these diagnostic
         *        copies because ordinary serial decode does not publish them.
         */
        bool runSerial(bool publish_snapshots)
        {
            for (int row = 0; row < rows_; ++row)
            {
                if (!serial_kernel_.chunkForwardMergedQKV(
                        merged_qkv_.data() + static_cast<size_t>(row) * qkv_stride_,
                        qkv_stride_,
                        alpha_.data() + static_cast<size_t>(row) * n_heads_,
                        beta_.data() + static_cast<size_t>(row) * n_heads_,
                        a_log_.data(),
                        dt_bias_.data(),
                        serial_output_.data() + static_cast<size_t>(row) * value_dim_,
                        serial_state_.data(),
                        /*seq_len=*/1,
                        n_k_heads_,
                        n_heads_,
                        d_k_,
                        d_v_,
                        /*global_v_head_offset=*/0,
                        /*chunk_size=*/64,
                        /*use_qk_l2norm=*/true))
                {
                    return false;
                }

                if (publish_snapshots)
                {
                    std::memcpy(
                        serial_snapshots_.data() +
                            static_cast<size_t>(row) * state_floats_,
                        serial_state_.data(),
                        static_cast<size_t>(state_floats_) * sizeof(float));
                }
            }
            return true;
        }

        /** @brief Prove grouped output, snapshots, and live-state isolation. */
        void verifyByteEquality()
        {
            resetGrouped();
            resetSerial();
            ASSERT_TRUE(runGrouped());
            ASSERT_TRUE(runSerial(/*publish_snapshots=*/true));

            const std::string prefix =
                "CPU GDN merged verifier M=" + std::to_string(rows_);
            expectByteExact(
                grouped_output_.data(),
                serial_output_.data(),
                grouped_output_.size(),
                prefix + " output");
            expectByteExact(
                grouped_snapshots_.data(),
                serial_snapshots_.data(),
                grouped_snapshots_.size(),
                prefix + " state snapshots");
            expectByteExact(
                grouped_live_state_.data(),
                initial_state_.data(),
                initial_state_.size(),
                prefix + " live-state isolation");
        }

        int rows() const { return rows_; }

    private:
        /** @brief Fill deterministic finite values spanning every source region. */
        void fillInputs()
        {
            for (int row = 0; row < rows_; ++row)
            {
                float *source =
                    merged_qkv_.data() + static_cast<size_t>(row) * qkv_stride_;
                for (int index = 0; index < q_src_dim_; ++index)
                {
                    source[index] =
                        0.002f * static_cast<float>((row * 17 + index) % 29 - 14);
                }
                for (int index = 0; index < k_src_dim_; ++index)
                {
                    source[q_src_dim_ + index] =
                        0.0015f * static_cast<float>((row * 19 + index) % 31 - 15);
                }
                for (int index = 0; index < value_dim_; ++index)
                {
                    source[q_src_dim_ + k_src_dim_ + index] =
                        0.001f * static_cast<float>((row * 23 + index) % 23 - 11);
                }
            }

            for (size_t index = 0; index < alpha_.size(); ++index)
            {
                alpha_[index] =
                    -0.2f + 0.004f * static_cast<float>(index % 37);
                beta_[index] =
                    0.1f - 0.003f * static_cast<float>(index % 41);
            }
            for (int head = 0; head < n_heads_; ++head)
            {
                a_log_[static_cast<size_t>(head)] =
                    -0.35f - 0.002f * static_cast<float>(head);
                dt_bias_[static_cast<size_t>(head)] =
                    0.01f * static_cast<float>(head % 7 - 3);
            }
            for (int index = 0; index < state_floats_; ++index)
            {
                initial_state_[static_cast<size_t>(index)] =
                    0.0001f * static_cast<float>(index % 43 - 21);
            }
        }

        int rows_ = 0;
        int n_k_heads_ = 0;
        int n_heads_ = 0;
        int d_k_ = 0;
        int d_v_ = 0;
        int q_src_dim_ = 0;
        int k_src_dim_ = 0;
        int value_dim_ = 0;
        int qkv_stride_ = 0;
        int state_floats_ = 0;

        std::vector<float> merged_qkv_;
        std::vector<float> alpha_;
        std::vector<float> beta_;
        std::vector<float> a_log_;
        std::vector<float> dt_bias_;
        std::vector<float> initial_state_;
        std::vector<float> grouped_live_state_;
        std::vector<float> serial_state_;
        std::vector<float> grouped_output_;
        std::vector<float> serial_output_;
        std::vector<float> grouped_snapshots_;
        std::vector<float> serial_snapshots_;
        CPUGatedDeltaNet grouped_kernel_;
        CPUGatedDeltaNet serial_kernel_;
    };

    /** @brief Measure one invocation after untimed state preparation. */
    double timeOneMs(
        const std::function<void()> &prepare,
        const std::function<bool()> &invoke)
    {
        prepare();
        const auto start = std::chrono::steady_clock::now();
        const bool ok = invoke();
        const auto end = std::chrono::steady_clock::now();
        EXPECT_TRUE(ok);
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    /**
     * @brief Alternate grouped and serial order and retain each best stable run.
     *
     * Alternation prevents one path from systematically inheriting hotter code
     * or data.  Best-of-N estimates the kernel ceiling and is intentionally not
     * an end-to-end latency percentile.
     */
    void timePairedBestMs(
        GDNVerifierCase *benchmark_case,
        int iterations,
        double *grouped_ms,
        double *serial_ms)
    {
        double best_grouped = std::numeric_limits<double>::infinity();
        double best_serial = std::numeric_limits<double>::infinity();

        const auto measure_grouped = [&]()
        {
            best_grouped = std::min(
                best_grouped,
                timeOneMs(
                    [&]() { benchmark_case->resetGrouped(); },
                    [&]() { return benchmark_case->runGrouped(); }));
        };
        const auto measure_serial = [&]()
        {
            best_serial = std::min(
                best_serial,
                timeOneMs(
                    [&]() { benchmark_case->resetSerial(); },
                    [&]() { return benchmark_case->runSerial(false); }));
        };

        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            if ((iteration & 1) == 0)
            {
                measure_grouped();
                measure_serial();
            }
            else
            {
                measure_serial();
                measure_grouped();
            }
        }

        *grouped_ms = best_grouped;
        *serial_ms = best_serial;
    }

    /** @brief Run correctness, warmup, and measurement for one runtime M. */
    BenchmarkResult runCase(
        int rows,
        int n_k_heads,
        int n_heads,
        int d_k,
        int d_v)
    {
        const int warmups = envInt("LLAMINAR_CPU_GDN_VERIFIER_WARMUP", 3);
        const int iterations = envInt("LLAMINAR_CPU_GDN_VERIFIER_ITERS", 12);
        GDNVerifierCase benchmark_case(rows, n_k_heads, n_heads, d_k, d_v);

        benchmark_case.verifyByteEquality();
        for (int warmup = 0; warmup < warmups; ++warmup)
        {
            benchmark_case.resetGrouped();
            EXPECT_TRUE(benchmark_case.runGrouped());
            benchmark_case.resetSerial();
            EXPECT_TRUE(benchmark_case.runSerial(false));
        }

        BenchmarkResult result;
        result.rows = rows;
        result.n_k_heads = n_k_heads;
        result.n_heads = n_heads;
        result.d_k = d_k;
        result.d_v = d_v;
        timePairedBestMs(
            &benchmark_case,
            iterations,
            &result.grouped_ms,
            &result.serial_ms);
        result.speedup = result.serial_ms / std::max(result.grouped_ms, 1e-9);

        benchmark_case.verifyByteEquality();
        return result;
    }

    /** @brief Persist concise additive evidence when a CSV path is requested. */
    void writeCsv(const std::vector<BenchmarkResult> &results)
    {
        const char *path = std::getenv("LLAMINAR_CPU_GDN_VERIFIER_CSV");
        if (!path || !*path)
            return;

        FILE *file = std::fopen(path, "w");
        ASSERT_NE(file, nullptr) << "failed to open " << path;
        std::fprintf(
            file,
            "backend,phase,rows,n_k_heads,n_heads,d_k,d_v,grouped_ms,"
            "serial_ms,speedup,byte_exact\n");
        for (const BenchmarkResult &result : results)
        {
            std::fprintf(
                file,
                "cpu,gdn_verifier_rows,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,1\n",
                result.rows,
                result.n_k_heads,
                result.n_heads,
                result.d_k,
                result.d_v,
                result.grouped_ms,
                result.serial_ms,
                result.speedup);
        }
        std::fclose(file);
    }
} // namespace

TEST(Perf__CPUGatedDeltaNetVerifierRows, RuntimeM_GroupedVsSerial_ProductionMergedQKV)
{
    const int n_heads = envInt("LLAMINAR_CPU_GDN_VERIFIER_HEADS", 16);
    const int n_k_heads =
        envInt("LLAMINAR_CPU_GDN_VERIFIER_K_HEADS", n_heads);
    const int d_k = envInt("LLAMINAR_CPU_GDN_VERIFIER_DK", 128);
    const int d_v = envInt("LLAMINAR_CPU_GDN_VERIFIER_DV", 128);
    const double min_speedup =
        envDouble("LLAMINAR_CPU_GDN_VERIFIER_MIN_SPEEDUP", 1.0);

    std::vector<BenchmarkResult> results;
    for (const int rows : envRows())
    {
        BenchmarkResult result =
            runCase(rows, n_k_heads, n_heads, d_k, d_v);
        results.push_back(result);

        std::cerr << "[CPU GDN verifier rows] M=" << result.rows
                  << " k_heads=" << result.n_k_heads
                  << " v_heads=" << result.n_heads
                  << " d_k=" << result.d_k
                  << " d_v=" << result.d_v
                  << " grouped_ms=" << result.grouped_ms
                  << " serial_ms=" << result.serial_ms
                  << " speedup=" << result.speedup
                  << " byte_exact=true\n";

        if (rows > 1)
        {
            EXPECT_GT(result.speedup, min_speedup)
                << "M=" << rows
                << " grouped_ms=" << result.grouped_ms
                << " serial_ms=" << result.serial_ms
                << " min_speedup=" << min_speedup;
        }
    }
    writeCsv(results);
}
