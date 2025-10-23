// @file TestTpOutputProjectionExecutor.cpp
#include "../src/tensors/TpOutputProjectionExecutor.h"
#include "../src/backends/InferenceBackend.h"
#include "../src/backends/PrefillBackend.h"
#include "../src/AdaptiveMatmul.h"
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstddef>

using namespace llaminar;

static bool baseline_mm(const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K)
{
    // naive reference
    for (std::size_t i = 0; i < M; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for (std::size_t k = 0; k < K; ++k)
                acc += (double)A[i * K + k] * (double)B[k * N + j];
            C[i * N + j] = (float)acc;
        }
    }
    return true;
}

static double rel_l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    return std::sqrt(num / (den + 1e-12));
}

int main()
{
    struct Case
    {
        std::size_t M, N, K;
        int tp;
    } cases[] = {
        {32, 64, 48, 3}, // uneven column
        {33, 50, 41, 4}, // uneven row+col potential
    };
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    bool all_ok = true;
    for (const auto &cs : cases)
    {
        std::vector<float> A(cs.M * cs.K), B(cs.K * cs.N);
        for (auto &v : A)
            v = dist(rng);
        for (auto &v : B)
            v = dist(rng);
        std::vector<float> full(cs.M * cs.N);
        baseline_mm(A.data(), B.data(), full.data(), cs.M, cs.N, cs.K);

        // Column split reconstruction
        std::vector<TPOutputLocalResult> col_parts;
        for (int r = 0; r < cs.tp; ++r)
        {
            TPOutputExecConfig cfg;
            cfg.tp_size = cs.tp;
            cfg.tp_rank = r;
            cfg.row_split = false;
            TPOutputProjectionExecutor exec(baseline_mm, cfg, cs.M, cs.N, cs.K);
            col_parts.push_back(exec.run(A.data(), B.data()));
        }
        std::vector<float> recon_col(cs.M * cs.N, 0.f);
        TPOutputProjectionExecutor::reconstruct_columns(col_parts, recon_col.data(), cs.M, cs.N);
        double relc = rel_l2(recon_col, full);
        if (relc > 1e-6)
        {
            std::cerr << "Column split mismatch rel_l2=" << relc << " for case M=" << cs.M << " N=" << cs.N << " K=" << cs.K << "\n";
            all_ok = false;
        }

        // Row split reconstruction
        std::vector<TPOutputLocalResult> row_parts;
        for (int r = 0; r < cs.tp; ++r)
        {
            TPOutputExecConfig cfg;
            cfg.tp_size = cs.tp;
            cfg.tp_rank = r;
            cfg.row_split = true;
            TPOutputProjectionExecutor exec(baseline_mm, cfg, cs.M, cs.N, cs.K);
            row_parts.push_back(exec.run(A.data(), B.data()));
        }
        std::vector<float> recon_row(cs.M * cs.N, 0.f);
        TPOutputProjectionExecutor::reconstruct_rows(row_parts, recon_row.data(), cs.M, cs.N);
        double relr = rel_l2(recon_row, full);
        if (relr > 1e-6)
        {
            std::cerr << "Row split mismatch rel_l2=" << relr << " for case M=" << cs.M << " N=" << cs.N << " K=" << cs.K << "\n";
            all_ok = false;
        }
    }
    if (!all_ok)
    {
        std::cerr << "TP output projection executor tests FAILED" << std::endl;
        return 1;
    }
    std::cout << "TP output projection executor tests PASSED" << std::endl;
    return 0;
}
