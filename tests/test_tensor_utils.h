#pragma once
/**
 * @file test_tensor_utils.h
 * @brief Shared deterministic utilities for kernel unit tests (allocation, PRNG, metrics).
 *
 * Provides lightweight helpers so each micro-test avoids duplicating boilerplate.
 * Intentionally header-only to keep CTest target wiring minimal.
 */
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <string>

namespace testutils
{

    struct PRNG
    {
        explicit PRNG(uint64_t seed = 0xC0FFEEULL) : eng(seed) {}
        std::mt19937_64 eng;
        // Uniform float in [-scale, scale]
        std::vector<float> uniform(int n, float scale = 1.f)
        {
            std::uniform_real_distribution<float> dist(-scale, scale);
            std::vector<float> v(n);
            for (auto &x : v)
                x = dist(eng);
            return v;
        }
        // Normal(0,1)
        std::vector<float> normal(int n)
        {
            std::normal_distribution<float> dist(0.f, 1.f);
            std::vector<float> v(n);
            for (auto &x : v)
                x = dist(eng);
            return v;
        }
    };

    struct DiffStats
    {
        double max_abs = 0.0;
        double rel_l2 = 0.0; // ||a-b||2 / (||b||2 + eps)
        double mean_abs = 0.0;
        int mismatches = 0; // elements where abs diff > per_elem_tol (if requested)
    };

    inline DiffStats diff(const std::vector<float> &a,
                          const std::vector<float> &b,
                          double per_elem_tol = -1.0)
    {
        DiffStats s;
        const size_t n = std::min(a.size(), b.size());
        double num_l2 = 0.0;
        double den_l2 = 0.0;
        double sum_abs = 0.0;
        const double eps = 1e-12;
        for (size_t i = 0; i < n; ++i)
        {
            double da = static_cast<double>(a[i]);
            double db = static_cast<double>(b[i]);
            double d = da - db;
            double ad = std::abs(d);
            s.max_abs = std::max(s.max_abs, ad);
            num_l2 += d * d;
            den_l2 += db * db;
            sum_abs += ad;
            if (per_elem_tol >= 0.0 && ad > per_elem_tol)
                s.mismatches++;
        }
        s.rel_l2 = std::sqrt(num_l2) / (std::sqrt(den_l2) + eps);
        s.mean_abs = sum_abs / (n + eps);
        return s;
    }

    inline std::string summarize(const DiffStats &s)
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "max_abs=%.3e rel_l2=%.3e mean_abs=%.3e mismatches=%d",
                      s.max_abs, s.rel_l2, s.mean_abs, s.mismatches);
        return std::string(buf);
    }

    inline bool within(const DiffStats &s, double max_abs, double rel_l2)
    {
        return s.max_abs <= max_abs && s.rel_l2 <= rel_l2;
    }

} // namespace testutils
