/**
 * @file RopeBenchmark.cpp
 * @brief Micro-benchmark harness for comparing legacy vs experimental RoPE implementations.
 *
 * Usage:
 *   ./rope_benchmark --seq 2048 --heads 16 --head-dim 64 --n-past 0 --repeat 50 [--experimental]
 *
 * Outputs summary with median / mean ns per token-head-dim element and speedup.
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "utils/DebugEnv.h"
#include "operators/common/AttentionPrimitives.h"

using clock_type = std::chrono::high_resolution_clock;

struct Args
{
    int seq = 1024;
    int heads = 16;
    int head_dim = 64;
    int n_past = 0;
    int repeat = 50;
    float freq_base = 10000.f; // default typical HF style
    bool experimental = false;
    bool validate = false;
};

static bool parse_int(const char *s, int &out)
{
    if (!s)
        return false;
    out = std::atoi(s);
    return out > 0;
}

Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--seq") && i + 1 < argc)
            parse_int(argv[++i], a.seq);
        else if (!std::strcmp(argv[i], "--heads") && i + 1 < argc)
            parse_int(argv[++i], a.heads);
        else if (!std::strcmp(argv[i], "--head-dim") && i + 1 < argc)
            parse_int(argv[++i], a.head_dim);
        else if (!std::strcmp(argv[i], "--n-past") && i + 1 < argc)
            parse_int(argv[++i], a.n_past);
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc)
            parse_int(argv[++i], a.repeat);
        else if (!std::strcmp(argv[i], "--freq-base") && i + 1 < argc)
            a.freq_base = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--experimental"))
            a.experimental = true;
        else if (!std::strcmp(argv[i], "--validate"))
            a.validate = true;
        else if (!std::strcmp(argv[i], "--help"))
        {
            std::cout << "RopeBenchmark options:\n"
                      << "  --seq N            sequence length (default 1024)\n"
                      << "  --heads H          attention heads (default 16)\n"
                      << "  --head-dim D       head dimension (even, default 64)\n"
                      << "  --n-past P         past tokens offset (default 0)\n"
                      << "  --repeat R         repetitions (default 50)\n"
                      << "  --freq-base F      RoPE frequency base (default 10000)\n"
                      << "  --experimental     benchmark experimental implementation (timing still does both paths)\n"
                      << "  --validate         enable internal validation logging (env flag)\n"
                      << std::endl;
            std::exit(0);
        }
    }
    return a;
}

template <typename Fn>
std::vector<double> time_loop(int repeat, Fn fn)
{
    std::vector<double> ns;
    ns.reserve(repeat);
    for (int i = 0; i < repeat; ++i)
    {
        auto t0 = clock_type::now();
        fn();
        auto t1 = clock_type::now();
        ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    return ns;
}

static double median(std::vector<double> v)
{
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);
    if (args.head_dim % 2 != 0)
    {
        std::cerr << "head_dim must be even" << std::endl;
        return 1;
    }
    size_t elems = (size_t)args.seq * args.heads * args.head_dim;

    if (args.validate)
    {
        setenv("LLAMINAR_ATTN_PRIM_ROPE_DEBUG_VALIDATE", "1", 1);
    }
    std::vector<float> q(elems), k(elems);
    // Fill deterministic data
    // NOTE: Previous version performed arithmetic directly on size_t:
    //   q[i] = (float)((i%97)-48)/50.f;
    // When (i % 97) < 48 this underflowed in unsigned domain, producing values near 2^64 / 50 ≈ 3.69e17.
    // Cast to signed int before subtracting to ensure proper negative values in expected small range.
    for (size_t i = 0; i < elems; ++i)
    {
        int vq = static_cast<int>(i % 97) - 48;          // range now [-48, 48]
        int vk = static_cast<int>((i * 131) % 113) - 56; // range [-56, 56]
        q[i] = static_cast<float>(vq) / 50.f;            // approx [-0.96, 0.96]
        k[i] = static_cast<float>(vk) / 60.f;            // approx [-0.933.., 0.933..]
    }

    std::vector<float> q_orig = q, k_orig = k;
    if (args.validate)
    {
        auto validate_init = [&](const char *tag, const std::vector<float> &buf)
        {
            double max_abs = 0.0, mean_abs = 0.0;
            double mn = 0.0, mx = 0.0;
            bool first = true;
            bool finite = true;
            for (float v : buf)
            {
                if (!std::isfinite(v))
                    finite = false;
                double av = std::abs((double)v);
                if (av > max_abs)
                    max_abs = av;
                mean_abs += av;
                if (first)
                {
                    mn = mx = v;
                    first = false;
                }
                else
                {
                    if (v < mn)
                        mn = v;
                    if (v > mx)
                        mx = v;
                }
            }
            mean_abs /= buf.size();
            std::cout << "[RopeBenchmark-INIT] " << tag << " max_abs=" << max_abs << " mean_abs=" << mean_abs << " finite=" << finite
                      << " range=[" << mn << "," << mx << "] sample[0:4]=" << buf[0] << "," << buf[1] << "," << buf[2] << "," << buf[3] << "\n";
        };
        validate_init("Q", q_orig);
        validate_init("K", k_orig);
    }
    std::vector<double> base_times, exp_times;
    base_times.reserve(args.repeat);
    exp_times.reserve(args.repeat);
    double global_max_q = 0.0, global_max_k = 0.0;
    double max_mag_base = 0.0, max_mag_exp = 0.0;
    int inf_nan_base = 0, inf_nan_exp = 0;

    for (int r = 0; r < args.repeat; ++r)
    {
        std::vector<float> q_base = q_orig, k_base = k_orig;
        auto t0 = clock_type::now();
        llaminar::attn::apply_rope(q_base.data(), k_base.data(), args.seq, args.head_dim, args.heads, args.heads, args.n_past, args.freq_base);
        auto t1 = clock_type::now();
        base_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        std::vector<float> q_exp = q_orig, k_exp = k_orig;
        auto t2 = clock_type::now();
        llaminar::attn::apply_rope_experimental(q_exp.data(), k_exp.data(), args.seq, args.head_dim, args.heads, args.heads, args.n_past, args.freq_base);
        auto t3 = clock_type::now();
        exp_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());

        double mdq = 0.0, mdk = 0.0;
        for (size_t i = 0; i < elems; ++i)
        {
            mdq = std::max(mdq, std::abs((double)q_base[i] - q_exp[i]));
            mdk = std::max(mdk, std::abs((double)k_base[i] - k_exp[i]));
            max_mag_base = std::max(max_mag_base, std::abs((double)q_base[i]));
            max_mag_base = std::max(max_mag_base, std::abs((double)k_base[i]));
            max_mag_exp = std::max(max_mag_exp, std::abs((double)q_exp[i]));
            max_mag_exp = std::max(max_mag_exp, std::abs((double)k_exp[i]));
            if (!std::isfinite(q_base[i]) || !std::isfinite(k_base[i]))
                inf_nan_base = 1;
            if (!std::isfinite(q_exp[i]) || !std::isfinite(k_exp[i]))
                inf_nan_exp = 1;
        }
        global_max_q = std::max(global_max_q, mdq);
        global_max_k = std::max(global_max_k, mdk);
    }

    double med_base = median(base_times);
    double med_exp = median(exp_times);
    double mean_base = std::accumulate(base_times.begin(), base_times.end(), 0.0) / base_times.size();
    double mean_exp = std::accumulate(exp_times.begin(), exp_times.end(), 0.0) / exp_times.size();

    std::cout << "RoPE Benchmark (seq=" << args.seq << " heads=" << args.heads << " head_dim=" << args.head_dim << " n_past=" << args.n_past << ")\n";
    std::cout << "  repeats=" << args.repeat << " freq_base=" << args.freq_base << "\n";
    std::cout << "  legacy   median=" << med_base / 1e6 << " ms  mean=" << mean_base / 1e6 << " ms\n";
    std::cout << "  experimental median=" << med_exp / 1e6 << " ms  mean=" << mean_exp / 1e6 << " ms\n";
    std::cout << "  speedup (median)=" << (med_base / med_exp) << "x\n";
    std::cout << "  diff   max_abs_q=" << global_max_q << " max_abs_k=" << global_max_k << "\n";
    std::cout << "  magnitude base_max=" << max_mag_base << " exp_max=" << max_mag_exp
              << " inf_nan_base=" << inf_nan_base << " inf_nan_exp=" << inf_nan_exp << "\n";

    return 0;
}
