#include "VNNIGemm.h"
#include <vector>
#include <random>
#include <iostream>

// Simple example of how you might prepare B_packed and call the kernel.

int main()
{
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 64;

    constexpr int M_R = 16;
    constexpr int N_R = 16;
    constexpr int K_BLK = 64;

    // Random input data
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> bias(N, 0.0f);
    std::vector<float> act_scales(K / K_BLK, 1.0f);
    std::vector<float> wgt_scales(N, 1.0f);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto &x : A)
        x = static_cast<int8_t>(dist(rng));
    for (auto &x : B)
        x = static_cast<int8_t>(dist(rng));

    // Pack full B for all K blocks (here only 1 block since K=K_BLK)
    const int T = K / K_BLK;
    const int nr = N; // full N panel

    int ld_block_B = 0, ld_col_B = 0;
    std::vector<int8_t> B_packed_storage(T * nr * K_BLK);

    for (int t = 0; t < T; ++t)
    {
        const int k0 = t * K_BLK;
        int8_t *panel = B_packed_storage.data() + t * nr * K_BLK;
        pack_B_panel_vnni<K_BLK>(
            B.data(), K, N,
            k0,
            0, nr,
            panel,
            ld_block_B, ld_col_B);
    }

    PackedB Bp;
    Bp.data = B_packed_storage.data();
    Bp.ld_block = ld_block_B;
    Bp.ld_col = ld_col_B;
    Bp.N = N;
    Bp.K_BLK = K_BLK;

    // Call GEMM kernel
    gemm_int8_vnni_kernel<
        M_R, N_R, K_BLK,
        2,    // UNROLL_K
        64,   // PREFETCH_B_L1
        256,  // PREFETCH_B_L2
        true, // ACCUM_INT32
        true, // USE_L2_PREFETCH
        true  // USE_VNNI
        >(
        A.data(),
        Bp,
        C.data(),
        bias.data(),
        act_scales.data(),
        wgt_scales.data(),
        M, N, K);

    std::cout << "C[0] = " << C[0] << "\n";
    return 0;
}