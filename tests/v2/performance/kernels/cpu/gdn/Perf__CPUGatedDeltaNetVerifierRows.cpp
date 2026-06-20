/**
 * @file Perf__CPUGatedDeltaNetVerifierRows.cpp
 * @brief Synthetic CPU GDN verifier-row microbenchmark for Phase 9.8.
 *
 * The dense Qwen3.6 verifier replay is the production proof, but it is too
 * heavy for kernel iteration.  This benchmark isolates the Gated Delta Net
 * recurrence contract that MTP all-position publication depends on:
 *
 * - grouped verifier rows must publish the same post-row states as explicit
 *   serial decode;
 * - M=2,3,4 must all be covered because those are the draft depths used by the
 *   dynamic depth controller;
 * - numerical checks use cosine, relative L2, symmetric KL, and max absolute
 *   error so an optimization cannot hide behind a single loose metric.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    struct VectorMetrics
    {
        double cosine = 0.0;
        double relative_l2 = 0.0;
        double symmetric_kl = 0.0;
        double max_abs = 0.0;
    };

    struct BenchmarkResult
    {
        int rows = 0;
        int n_heads = 0;
        int d_k = 0;
        int d_v = 0;
        double grouped_ms = 0.0;
        double serial_ms = 0.0;
        double speedup = 0.0;
        VectorMetrics output_metrics;
        VectorMetrics state_metrics;
    };

    int envInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || parsed <= 0)
            return fallback;
        return static_cast<int>(parsed);
    }

    double envDouble(const char *name, double fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const double parsed = std::strtod(raw, &end);
        if (end == raw || parsed <= 0.0)
            return fallback;
        return parsed;
    }

    std::vector<int> envRows()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_GDN_VERIFIER_M");
        if (!raw || !*raw)
            return {2, 3, 4};

        std::vector<int> rows;
        std::stringstream ss(raw);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            const int value = std::atoi(token.c_str());
            if (value >= 1 && value <= 4)
                rows.push_back(value);
        }
        if (rows.empty())
            rows = {2, 3, 4};
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        return rows;
    }

    /**
     * @brief Symmetric KL after softmax-normalizing arbitrary vectors.
     *
     * GDN recurrence states are signed, so KL is undefined on the raw values.
     * We softmax both vectors and compare the resulting distributions.  This is
     * intentionally diagnostic rather than a replacement for cosine/L2.
     */
    double symmetricSoftmaxKL(const float *a, const float *b, size_t count)
    {
        if (count == 0)
            return 0.0;
        const float max_a = *std::max_element(a, a + count);
        const float max_b = *std::max_element(b, b + count);

        long double sum_a = 0.0;
        long double sum_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            sum_a += std::exp(static_cast<long double>(a[i] - max_a));
            sum_b += std::exp(static_cast<long double>(b[i] - max_b));
        }

        constexpr long double eps = 1e-30L;
        long double kl_ab = 0.0;
        long double kl_ba = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const long double pa =
                std::exp(static_cast<long double>(a[i] - max_a)) / sum_a;
            const long double pb =
                std::exp(static_cast<long double>(b[i] - max_b)) / sum_b;
            kl_ab += pa * std::log((pa + eps) / (pb + eps));
            kl_ba += pb * std::log((pb + eps) / (pa + eps));
        }
        return static_cast<double>(0.5L * (kl_ab + kl_ba));
    }

    VectorMetrics computeMetrics(const float *a, const float *b, size_t count)
    {
        VectorMetrics metrics;
        long double dot = 0.0;
        long double norm_a = 0.0;
        long double norm_b = 0.0;
        long double diff2 = 0.0;

        for (size_t i = 0; i < count; ++i)
        {
            const long double av = a[i];
            const long double bv = b[i];
            const long double d = av - bv;
            dot += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
            diff2 += d * d;
            metrics.max_abs =
                std::max(metrics.max_abs, static_cast<double>(std::abs(d)));
        }

        constexpr long double eps = 1e-30L;
        metrics.cosine =
            static_cast<double>(dot / (std::sqrt(norm_a * norm_b) + eps));
        metrics.relative_l2 =
            static_cast<double>(std::sqrt(diff2) / (std::sqrt(norm_b) + eps));
        metrics.symmetric_kl = symmetricSoftmaxKL(a, b, count);
        return metrics;
    }

    void fillSyntheticInputs(
        std::vector<float> &q,
        std::vector<float> &k,
        std::vector<float> &v,
        std::vector<float> &alpha,
        std::vector<float> &beta,
        std::vector<float> &a_log,
        std::vector<float> &dt_bias,
        std::vector<float> &initial_state)
    {
        for (size_t i = 0; i < q.size(); ++i)
        {
            q[i] = 0.002f * static_cast<float>(static_cast<int>(i % 29) - 14);
            k[i] = 0.0015f * static_cast<float>(static_cast<int>(i % 31) - 15);
        }
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = 0.001f * static_cast<float>(static_cast<int>(i % 23) - 11);
        for (size_t i = 0; i < alpha.size(); ++i)
        {
            alpha[i] = -0.2f + 0.004f * static_cast<float>(i % 37);
            beta[i] = 0.1f - 0.003f * static_cast<float>(i % 41);
        }
        for (size_t h = 0; h < a_log.size(); ++h)
        {
            a_log[h] = -0.35f - 0.002f * static_cast<float>(h);
            dt_bias[h] = 0.01f * static_cast<float>(static_cast<int>(h % 7) - 3);
        }
        for (size_t i = 0; i < initial_state.size(); ++i)
            initial_state[i] =
                0.0001f * static_cast<float>(static_cast<int>(i % 43) - 21);
    }

    bool runGrouped(
        int rows,
        int n_heads,
        int d_k,
        int d_v,
        const std::vector<float> &q,
        const std::vector<float> &k,
        const std::vector<float> &v,
        const std::vector<float> &alpha,
        const std::vector<float> &beta,
        const std::vector<float> &a_log,
        const std::vector<float> &dt_bias,
        const std::vector<float> &initial_state,
        std::vector<float> *output,
        std::vector<float> *capture)
    {
        const int state_floats = n_heads * d_k * d_v;
        const int v_stride = n_heads * d_v;

        CPUGatedDeltaNet kernel;
        std::vector<float> state = initial_state;
        output->assign(static_cast<size_t>(rows) * v_stride, 0.0f);
        capture->assign(static_cast<size_t>(rows) * state_floats, 0.0f);

        /**
         * Measure the grouped recurrence kernel ceiling, not caller-side
         * speculative-state preparation.  The production transaction still has
         * to make live state safe before verification, but the serial oracle
         * below also starts from a private state copy.  Calling the explicit
         * snapshot API keeps the comparison fair: one prepared state in, exact
         * M=2..4 verifier rows and post-row snapshots out.
         */
        return kernel.chunkForwardWithStateSnapshots(
            q.data(),
            k.data(),
            v.data(),
            alpha.data(),
            beta.data(),
            a_log.data(),
            dt_bias.data(),
            output->data(),
            state.data(),
            rows,
            n_heads,
            d_k,
            d_v,
            /*chunk_size=*/64,
            /*use_qk_l2norm=*/true,
            capture->data(),
            state_floats,
            rows);
    }

    bool runSerial(
        int rows,
        int n_heads,
        int d_k,
        int d_v,
        const std::vector<float> &q,
        const std::vector<float> &k,
        const std::vector<float> &v,
        const std::vector<float> &alpha,
        const std::vector<float> &beta,
        const std::vector<float> &a_log,
        const std::vector<float> &dt_bias,
        const std::vector<float> &initial_state,
        std::vector<float> *output,
        std::vector<float> *capture)
    {
        const int state_floats = n_heads * d_k * d_v;
        const int qk_stride = n_heads * d_k;
        const int v_stride = n_heads * d_v;

        CPUGatedDeltaNet kernel;
        std::vector<float> state = initial_state;
        output->assign(static_cast<size_t>(rows) * v_stride, 0.0f);
        capture->assign(static_cast<size_t>(rows) * state_floats, 0.0f);

        for (int t = 0; t < rows; ++t)
        {
            if (!kernel.recurrent_step(
                    q.data() + static_cast<size_t>(t) * qk_stride,
                    k.data() + static_cast<size_t>(t) * qk_stride,
                    v.data() + static_cast<size_t>(t) * v_stride,
                    alpha.data() + static_cast<size_t>(t) * n_heads,
                    beta.data() + static_cast<size_t>(t) * n_heads,
                    a_log.data(),
                    dt_bias.data(),
                    output->data() + static_cast<size_t>(t) * v_stride,
                    state.data(),
                    n_heads,
                    d_k,
                    d_v,
                    /*use_qk_l2norm=*/true))
            {
                return false;
            }
            std::memcpy(capture->data() + static_cast<size_t>(t) * state_floats,
                        state.data(),
                        static_cast<size_t>(state_floats) * sizeof(float));
        }
        return true;
    }

    double timeOneMs(const std::function<void()> &fn)
    {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    /**
     * @brief Measure grouped and serial paths with balanced ordering.
     *
     * The M=2 verifier-row GDN recurrence is intentionally tiny.  Measuring all
     * grouped iterations first and all serial iterations second lets the second
     * path inherit warmer instruction/data caches and occasionally flips a real
     * small win into a false loss.  This helper alternates the order on each
     * sample and records the best stable sample for each path, which is the
     * usual microbenchmark signal for kernel ceiling work.
     */
    void timePairedBestMs(
        const std::function<void()> &grouped,
        const std::function<void()> &serial,
        int iters,
        double *grouped_ms,
        double *serial_ms)
    {
        double best_grouped = std::numeric_limits<double>::infinity();
        double best_serial = std::numeric_limits<double>::infinity();
        for (int i = 0; i < iters; ++i)
        {
            if ((i & 1) == 0)
            {
                best_grouped = std::min(best_grouped, timeOneMs(grouped));
                best_serial = std::min(best_serial, timeOneMs(serial));
            }
            else
            {
                best_serial = std::min(best_serial, timeOneMs(serial));
                best_grouped = std::min(best_grouped, timeOneMs(grouped));
            }
        }

        *grouped_ms = best_grouped;
        *serial_ms = best_serial;
    }

    BenchmarkResult runCase(int rows, int n_heads, int d_k, int d_v)
    {
        const int max_rows = 4;
        const int state_floats = n_heads * d_k * d_v;
        const int qk_stride = n_heads * d_k;
        const int v_stride = n_heads * d_v;
        const int warmups = envInt("LLAMINAR_CPU_GDN_VERIFIER_WARMUP", 2);
        const int iters = envInt("LLAMINAR_CPU_GDN_VERIFIER_ITERS", 10);

        std::vector<float> q(static_cast<size_t>(max_rows) * qk_stride);
        std::vector<float> k(q.size());
        std::vector<float> v(static_cast<size_t>(max_rows) * v_stride);
        std::vector<float> alpha(static_cast<size_t>(max_rows) * n_heads);
        std::vector<float> beta(static_cast<size_t>(max_rows) * n_heads);
        std::vector<float> a_log(n_heads);
        std::vector<float> dt_bias(n_heads);
        std::vector<float> initial_state(state_floats);
        fillSyntheticInputs(q, k, v, alpha, beta, a_log, dt_bias, initial_state);

        std::vector<float> grouped_output;
        std::vector<float> grouped_capture;
        std::vector<float> serial_output;
        std::vector<float> serial_capture;

        auto grouped = [&]()
        {
            ASSERT_TRUE(runGrouped(
                rows, n_heads, d_k, d_v, q, k, v, alpha, beta,
                a_log, dt_bias, initial_state,
                &grouped_output, &grouped_capture));
        };
        auto serial = [&]()
        {
            ASSERT_TRUE(runSerial(
                rows, n_heads, d_k, d_v, q, k, v, alpha, beta,
                a_log, dt_bias, initial_state,
                &serial_output, &serial_capture));
        };

        for (int i = 0; i < warmups; ++i)
        {
            grouped();
            serial();
        }

        BenchmarkResult result;
        result.rows = rows;
        result.n_heads = n_heads;
        result.d_k = d_k;
        result.d_v = d_v;
        timePairedBestMs(grouped, serial, iters, &result.grouped_ms, &result.serial_ms);
        result.speedup = result.serial_ms / std::max(result.grouped_ms, 1e-9);
        result.output_metrics = computeMetrics(
            grouped_output.data(),
            serial_output.data(),
            static_cast<size_t>(rows) * v_stride);
        result.state_metrics = computeMetrics(
            grouped_capture.data(),
            serial_capture.data(),
            static_cast<size_t>(rows) * state_floats);
        return result;
    }

    void appendCsv(const std::vector<BenchmarkResult> &results)
    {
        const char *path = std::getenv("LLAMINAR_CPU_GDN_VERIFIER_CSV");
        if (!path || !*path)
            return;

        FILE *file = std::fopen(path, "w");
        ASSERT_NE(file, nullptr) << "failed to open " << path;
        std::fprintf(
            file,
            "backend,phase,rows,n_heads,d_k,d_v,grouped_ms,serial_ms,speedup,"
            "output_cosine,output_relative_l2,output_symmetric_kl,output_max_abs,"
            "state_cosine,state_relative_l2,state_symmetric_kl,state_max_abs\n");
        for (const BenchmarkResult &r : results)
        {
            std::fprintf(
                file,
                "cpu,gdn_verifier_rows,%d,%d,%d,%d,%.9f,%.9f,%.9f,"
                "%.12f,%.12g,%.12g,%.12g,%.12f,%.12g,%.12g,%.12g\n",
                r.rows,
                r.n_heads,
                r.d_k,
                r.d_v,
                r.grouped_ms,
                r.serial_ms,
                r.speedup,
                r.output_metrics.cosine,
                r.output_metrics.relative_l2,
                r.output_metrics.symmetric_kl,
                r.output_metrics.max_abs,
                r.state_metrics.cosine,
                r.state_metrics.relative_l2,
                r.state_metrics.symmetric_kl,
                r.state_metrics.max_abs);
        }
        std::fclose(file);
    }

    void expectTightMetrics(const BenchmarkResult &r)
    {
        const double min_speedup =
            envDouble("LLAMINAR_CPU_GDN_VERIFIER_MIN_SPEEDUP", 1.0);

        EXPECT_GE(r.output_metrics.cosine, 0.999999999)
            << "rows=" << r.rows;
        EXPECT_LE(r.output_metrics.relative_l2, 1e-7)
            << "rows=" << r.rows;
        EXPECT_LE(r.output_metrics.symmetric_kl, 1e-12)
            << "rows=" << r.rows;
        EXPECT_LE(r.output_metrics.max_abs, 1e-7)
            << "rows=" << r.rows;

        EXPECT_GE(r.state_metrics.cosine, 0.999999999)
            << "rows=" << r.rows;
        EXPECT_LE(r.state_metrics.relative_l2, 1e-7)
            << "rows=" << r.rows;
        EXPECT_LE(r.state_metrics.symmetric_kl, 1e-12)
            << "rows=" << r.rows;
        EXPECT_LE(r.state_metrics.max_abs, 1e-7)
            << "rows=" << r.rows;

        EXPECT_GT(r.speedup, min_speedup)
            << "rows=" << r.rows
            << " grouped_ms=" << r.grouped_ms
            << " serial_ms=" << r.serial_ms
            << " min_speedup=" << min_speedup;
    }
}

TEST(Perf__CPUGatedDeltaNetVerifierRows, M234_GroupedVsSerial_Synthetic)
{
    const int n_heads = envInt("LLAMINAR_CPU_GDN_VERIFIER_HEADS", 32);
    const int d_k = envInt("LLAMINAR_CPU_GDN_VERIFIER_DK", 128);
    const int d_v = envInt("LLAMINAR_CPU_GDN_VERIFIER_DV", 128);

    std::vector<BenchmarkResult> results;
    for (int rows : envRows())
    {
        BenchmarkResult result = runCase(rows, n_heads, d_k, d_v);
        results.push_back(result);

        std::cerr << "[CPU GDN verifier rows] M=" << result.rows
                  << " heads=" << result.n_heads
                  << " d_k=" << result.d_k
                  << " d_v=" << result.d_v
                  << " grouped_ms=" << result.grouped_ms
                  << " serial_ms=" << result.serial_ms
                  << " speedup=" << result.speedup
                  << " output_cos=" << result.output_metrics.cosine
                  << " output_l2=" << result.output_metrics.relative_l2
                  << " output_skl=" << result.output_metrics.symmetric_kl
                  << " state_cos=" << result.state_metrics.cosine
                  << " state_l2=" << result.state_metrics.relative_l2
                  << " state_skl=" << result.state_metrics.symmetric_kl
                  << "\n";

        expectTightMetrics(result);
    }
    appendCsv(results);
}
