#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include "src/utils/debug_env.h"
#include "src/kernels/common/attention_primitives.h"

using namespace llaminar::attn;

struct RopeResult
{
    int seq_len;
    double total_ms;
    double per_iter_ms;
};

static RopeResult bench_rope_once(int heads, int head_dim, int seq_len, int n_past, int iters)
{
    std::vector<float> q((size_t)seq_len * heads * head_dim);
    std::vector<float> k((size_t)seq_len * heads * head_dim);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &v : q)
        v = dist(rng);
    for (auto &v : k)
        v = dist(rng);

    // Warmup
    apply_rope(q.data(), k.data(), seq_len, head_dim, heads, n_past, 10000.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        // Reinitialize to avoid cache benefits from already rotated values
        for (auto &v : q)
            v = dist(rng);
        for (auto &v : k)
            v = dist(rng);
        apply_rope(q.data(), k.data(), seq_len, head_dim, heads, n_past, 10000.0f);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    return {seq_len, ms, ms / iters};
}

int main(int argc, char **argv)
{
    int heads = 8, head_dim = 64, seq_len = 256, n_past = 0, iters = 50;
    int seq_len_min = -1, seq_len_max = -1, seq_len_step = 0, seq_len_mul = 0;
    bool csv = false;
    std::string seq_list_spec;
    bool compare_recurrence = false; // run both recurrence path and legacy trig path
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&](int &var)
        { if(i+1<argc) var=std::atoi(argv[++i]); };
        if (a == "--heads")
            next(heads);
        else if (a == "--head-dim")
            next(head_dim);
        else if (a == "--seq-len")
            next(seq_len);
        else if (a == "--n-past")
            next(n_past);
        else if (a == "--iters")
            next(iters);
        else if (a == "--seq-lens" && i + 1 < argc)
        {
            seq_list_spec = argv[++i];
        }
        else if (a == "--seq-len-min")
            next(seq_len_min);
        else if (a == "--seq-len-max")
            next(seq_len_max);
        else if (a == "--seq-len-step")
            next(seq_len_step);
        else if (a == "--seq-len-mul")
            next(seq_len_mul);
        else if (a == "--csv")
            csv = true;
        else if (a == "--compare-recurrence")
            compare_recurrence = true;
    }
    std::vector<int> seqs;
    if (!seq_list_spec.empty())
    {
        std::stringstream ss(seq_list_spec);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            if (!tok.empty())
                seqs.push_back(std::atoi(tok.c_str()));
        }
    }
    else if (seq_len_min > 0 && seq_len_max > 0)
    {
        if (seq_len_mul > 1)
        {
            for (int v = seq_len_min; v <= seq_len_max; v = (int)((long long)v * seq_len_mul))
                seqs.push_back(v);
        }
        else
        {
            int step = seq_len_step > 0 ? seq_len_step : std::max(1, seq_len_min);
            for (int v = seq_len_min; v <= seq_len_max; v += step)
                seqs.push_back(v);
        }
    }
    else
    {
        seqs.push_back(seq_len);
    }
    if (seqs.empty())
    {
        std::cerr << "No sequence lengths resolved\n";
        return 1;
    }

    struct ExtResult
    {
        RopeResult base;
        std::string mode;
        double elems;
        double GBs;
    };
    std::vector<ExtResult> results;
    results.reserve(seqs.size() * (compare_recurrence ? 2 : 1));

    auto compute_metrics = [&](const RopeResult &r)
    {
        double elems = (double)r.seq_len * heads * head_dim;
        double bytes = 2.0 * elems * sizeof(float);
        double GBs = (bytes / 1e9) / (r.per_iter_ms / 1000.0);
        return std::pair<double, double>{elems, GBs};
    };

    for (int sl : seqs)
    {
        if (compare_recurrence)
        {
            // First ensure recurrence enabled (disable flag unset)
            ::setenv("LLAMINAR_ATTN_PRIM_ROPE_DISABLE_RECURRENCE", "0", 1);
            llaminar::debugEnvRefresh();
            RopeResult rec = bench_rope_once(heads, head_dim, sl, n_past, iters);
            auto [elems_r, gbs_r] = compute_metrics(rec);
            results.push_back({rec, "recurrence", elems_r, gbs_r});
            // Now disable recurrence
            ::setenv("LLAMINAR_ATTN_PRIM_ROPE_DISABLE_RECURRENCE", "1", 1);
            llaminar::debugEnvRefresh();
            RopeResult legacy = bench_rope_once(heads, head_dim, sl, n_past, iters);
            auto [elems_l, gbs_l] = compute_metrics(legacy);
            results.push_back({legacy, "legacy", elems_l, gbs_l});
            // Restore env for next iteration (leave disabled state unspecified)
        }
        else
        {
            RopeResult single = bench_rope_once(heads, head_dim, sl, n_past, iters);
            auto [elems_s, gbs_s] = compute_metrics(single);
            results.push_back({single, "auto", elems_s, gbs_s});
        }
    }

    // Output
    if (csv)
    {
        std::cout << "seq_len,heads,head_dim,iters,mode,total_ms,per_iter_ms,elems_per_iter,GBs_est" << '\n';
        for (auto &r : results)
        {
            std::cout << r.base.seq_len << ',' << heads << ',' << head_dim << ',' << iters << ',' << r.mode << ','
                      << std::fixed << std::setprecision(4) << r.base.total_ms << ',' << r.base.per_iter_ms << ','
                      << std::setprecision(0) << r.elems << ',' << std::setprecision(3) << r.GBs << '\n';
        }
    }
    else
    {
        std::cout << "RoPE Benchmark (heads=" << heads << " head_dim=" << head_dim << " iters=" << iters << ")\n";
        std::cout << std::left << std::setw(10) << "seq_len" << std::setw(10) << "mode" << std::setw(14) << "per_iter_ms" << std::setw(14) << "total_ms" << std::setw(14) << "GB/s(est)" << "elems" << '\n';
        for (auto &r : results)
        {
            std::cout << std::left << std::setw(10) << r.base.seq_len
                      << std::setw(10) << r.mode
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.base.per_iter_ms
                      << std::setw(14) << r.base.total_ms
                      << std::setw(14) << std::setprecision(3) << r.GBs
                      << (long long)r.elems << '\n';
        }
    }
    return 0;
}
