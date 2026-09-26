/**
 * @file Test__GroupedDp4aLaunchGeometry.cpp
 * @brief Device-free totality proof for register-bounded grouped DP4A ownership.
 *
 * The production launch/index authority must cover each matrix element once,
 * even when one logical tile becomes multiple CTAs or the last tile is partial.
 * Arithmetic equivalence remains a separate captured all-codebook GPU gate.
 */
#include <gtest/gtest.h>
#include "kernels/cuda/gemm/GroupedDp4aLaunchGeometry.h"
#include <algorithm>
#include <vector>

namespace
{
/** @brief Exhaust row extents and column tails for one compiled launch shape. */
template <int Rows, int Columns, int CPT>
void verifyGeometry()
{
    using G = llaminar2::cuda::GroupedDp4aLaunchGeometry<Rows, Columns, CPT>;
    static_assert(G::threads_per_block <= 256 && G::threads_per_block >= 32);
    static_assert(G::rows_per_lane <= 16 && G::rows_per_lane >= 2);
    static_assert(G::rows_per_block <= Rows);
    if constexpr (G::row_groups == 1)
    {
        // The ignored thread argument must be immaterial even before compiler
        // launch-range analysis. CUDA can then retain a uniform row address
        // instead of expanding every unrolled row's address into lane registers.
        static_assert(G::rowBase(7, 0) == 7 * G::rows_per_block);
        static_assert(G::rowBase(7, 1023) == 7 * G::rows_per_block);
    }
    for (int m = 1; m <= 129; ++m)
    {
        for (int n : {1, Columns - 1, Columns, Columns + 1})
        {
            std::vector<unsigned char> owners(m * n, 0);
            for (int bz = 0; bz * G::rows_per_block < m; ++bz)
                for (int bx = 0; bx * G::columns_per_block < n; ++bx)
                    for (int thread = 0; thread < G::threads_per_block; ++thread)
                        for (int r = 0; r < G::rows_per_lane; ++r)
                            for (int c = 0; c < CPT; ++c)
                            {
                                const int row = G::rowBase(bz, thread) + r;
                                const int col = G::columnBase(bx, thread) + c;
                                if (row < m && col < n)
                                    ++owners[row * n + col];
                            }
            ASSERT_TRUE(std::all_of(owners.begin(), owners.end(),
                                   [](unsigned char count) { return count == 1; }))
                << "rows=" << Rows << " tile=" << Columns << " cpt=" << CPT
                << " M=" << m << " N=" << n;
        }
    }
}

/** @brief Exercise the complete column/column-per-thread dispatch catalog. */
template <int Rows>
void verifyRowPolicy()
{
    verifyGeometry<Rows, 32, 1>();
    verifyGeometry<Rows, 64, 1>();
    verifyGeometry<Rows, 64, 2>();
    verifyGeometry<Rows, 128, 1>();
    verifyGeometry<Rows, 128, 2>();
    verifyGeometry<Rows, 256, 2>();
    verifyGeometry<Rows, 256, 4>();
    verifyGeometry<Rows, 512, 4>();
}
} // namespace

TEST(GroupedDp4aLaunchGeometry, EveryRowPolicyOwnsEveryOutputExactlyOnce)
{
    verifyRowPolicy<2>();
    verifyRowPolicy<4>();
    verifyRowPolicy<8>();
    verifyRowPolicy<16>();
    verifyRowPolicy<32>();
    verifyRowPolicy<64>();
}
