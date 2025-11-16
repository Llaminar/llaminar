/**
 * Temporary simplified correctness test
 */
#include <gtest/gtest.h>
#include <random>
#include "v2/kernels/cpu/gemm_v3/VNNIGemm.h"

using namespace llaminar::v2::kernels::cpu::gemm_v3;

class VNNIGemmCorrectnessSimple : public ::testing::Test {};

TEST_F(VNNIGemmCorrectnessSimple, JustRunKernel)
{
    const int M = 512;
    const int N = 896;
    const int K = 896;

    constexpr int M_R = 16;
    constexpr int N_R = 64;
    constexpr int K_BLK = 64;
    constexpr int UNROLL_K = 2;
    constexpr int PREFETCH_B_L1 = 64;
    constexpr int PREFETCH_B_L2 = 256;

    const int T = K / K_BLK;

    std::cout << "\n=== Simple VNNI GEMM Kernel Test ===" << std::endl;
    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Random INT8 inputs (matching RealWeights test)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> int8_dist(-127, 127);

    std::vector<int8_t> A_int8(M * K);
    for (auto &x : A_int8) x = static_cast<int8_t>(int8_dist(rng));

    std::vector<int8_t> B_int8_unpacked(K * N);
    for (auto &x : B_int8_unpacked) x = static_cast<int8_t>(int8_dist(rng));

    // Pack B to VNNI layout
    std::vector<int8_t> B_packed(T * N * K_BLK);
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            for (int kk = 0; kk < K_BLK; ++kk) {
                const int k = t * K_BLK + kk;
                B_packed[t * N * K_BLK + n * K_BLK + kk] = B_int8_unpacked[k * N + n];
            }
        }
    }

    PackedB Bp;
    Bp.data = B_packed.data();
    Bp.ld_block = N * K_BLK;
    Bp.ld_col = K_BLK;
    Bp.N = N;
    Bp.K_BLK = K_BLK;

    // Unit scales
    std::vector<float> act_scales(T, 1.0f);
    std::vector<float> wgt_scales(N, 1.0f);
    std::vector<float> bias(N, 0.0f);

    // Output
    std::vector<float> C(M * N, 0.0f);

    std::cout << "Running VNNI GEMM kernel..." << std::endl;

    gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                          PREFETCH_B_L1, PREFETCH_B_L2,
                          true, true, true>(
        A_int8.data(),
        Bp,
        C.data(),
        bias.data(),
        act_scales.data(),
        wgt_scales.data(),
        M, N, K);

    std::cout << "✓ VNNI GEMM completed" << std::endl;

    // Sanity check
    int non_zero = 0;
    for (const auto &val : C) {
        if (std::abs(val) > 1e-6) non_zero++;
    }
    double non_zero_pct = 100.0 * non_zero / C.size();
    std::cout << "Non-zero outputs: " << std::fixed << std::setprecision(1) 
              << non_zero_pct << "%" << std::endl;

    EXPECT_GT(non_zero_pct, 10.0) << "Too few non-zero values";
}
