/**
 * @file Test__Q8ResidualAllFormatsCPU.cpp
 * @brief Explicit CPU integration sweep of original-format TwoTerm rows,
 * threads and tails.
 *
 * Source weights and captured-stage policy remain production-owned. The M1 row
 * loop exists only as an independent diagnostic oracle. ISA aliases start new
 * processes because the canonical runtime ISA policy is immutable after first
 * use.
 */
#include "../../utils/Q8ResidualAllFormatCPUHarness.h"
#include "utils/MPIBootstrap.h"

/** @test All positive worker teams execute every format on a tiny complete
 * source. */
TEST(Q8ResidualAllFormatsCPU, NativeRowsThreadsAndLargeSmallReuse) {
  const int budget = std::max(
      1, llaminar2::MPIBootstrap::detectCPUTopology().cores_per_socket);
  std::vector<int> workers(budget);
  std::iota(workers.begin(), workers.end(), 1);
  std::vector<int> representative{1, std::min(budget, 3), budget};
  std::sort(representative.begin(), representative.end());
  representative.erase(
      std::unique(representative.begin(), representative.end()),
      representative.end());
  for (const auto &format : llaminar2::test::quantizedVerifierFormats()) {
    const std::array tinyLives{3, 0, 1, 3};
    llaminar2::test::q8_all_formats::proveCPUThreads(format, 3, 5, 256, workers,
                                                     false, tinyLives);
    ASSERT_FALSE(::testing::Test::HasFailure());
    for (int rows : {1, 9, 15, 33, 64, 65}) {
      const std::array single{1};
      const std::span<const int> teams =
          rows == 65 ? std::span<const int>(representative)
                     : std::span<const int>(single);
      llaminar2::test::q8_all_formats::proveCPUThreads(format, rows, 65, 512,
                                                       teams);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
  }
}
/** @test Large physical work uses bounded distinct-row oracles and retains
 * poisoned suffixes. */
TEST(Q8ResidualAllFormatsCPU, NativeLargeTiledRows) {
  const std::array team{1};
  for (const auto &format : llaminar2::test::quantizedVerifierFormats()) {
    for (int rows : {512, 1024, 2048}) {
      const std::array lives{rows, 1, 0, std::min(rows, 15), rows};
      llaminar2::test::q8_all_formats::proveCPUThreads(format, rows, 65, 256,
                                                       team, true, lives);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
    // A separate untiled (seven distinct authored rows) large bank prevents the
    // one-row tiling specialization from being mistaken for general M support.
    const std::array lives{2048, 512, 1024, 1, 0, 2048};
    llaminar2::test::q8_all_formats::proveCPUThreads(format, 2048, 5, 256, team,
                                                     false, lives);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
