/**
 * @file Test__Q8ResidualGroupedGeometry.cpp
 * @brief Device-free totality, partial-tile and launch identity regressions.
 *
 * These tests exercise the actual immutable production geometry authority.
 * They prove dispatch and physical coverage, not device arithmetic or speed.
 */
#include <gtest/gtest.h>
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include <array>
#include <climits>
#include <cstdint>
using namespace llaminar2;
/** @brief Verify all supported rows and neighboring column crossover geometries. */
TEST(Q8ResidualGroupedGeometry, TotalRowAndColumnBoundaries) {
 for(auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm})
  for(int rows=1;rows<=2049;++rows)
   for(int cols:{1,31,63,64,65,127,128,129,2559,2560,2561,10239,10240,10241}) {
    const auto g=q8ResidualGroupedGeometry(backend,rows,cols);
    const int tile=backend==Q8ResidualGroupedBackend::CUDA?64:32;
    const bool grouped=rows>=tile&&cols>=64;
    EXPECT_EQ(g.strategy,grouped?Q8ResidualGroupedStrategy::CooperativeMatrix:Q8ResidualGroupedStrategy::RowVector);
    EXPECT_EQ(g.threads,grouped?256:128);
    EXPECT_EQ(g.rowTile,grouped?tile:(rows==1?1:4));
    EXPECT_EQ(g.columnTile,grouped?64:128);
    const auto rowTiles=(std::uint64_t(rows)+g.rowTile-1)/g.rowTile;
    const auto colTiles=(std::uint64_t(cols)+g.columnTile-1)/g.columnTile;
    EXPECT_EQ(g.blocks,rowTiles*colTiles);
    EXPECT_GE(rowTiles*g.rowTile,std::uint64_t(rows));
    EXPECT_LT((rowTiles-1)*g.rowTile,std::uint64_t(rows));
    EXPECT_GE(colTiles*g.columnTile,std::uint64_t(cols));
    EXPECT_LT((colTiles-1)*g.columnTile,std::uint64_t(cols));
   }
}
/** @brief Physical matrix identity excludes mutable live-row observations. */
TEST(Q8ResidualGroupedGeometry, StablePartialAndEmptyReplayIdentity) {
 for(auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm}) {
  for(int columns:{2560,10240}) {
   const auto admitted=q8ResidualGroupedGeometry(backend,2048,columns);
   for(int live:{2048,1025,513,33,2,1,0,2048}) {
    EXPECT_EQ(admitted,q8ResidualGroupedGeometry(backend,2048,columns));
    const auto liveTiles=(std::uint64_t(live)+admitted.rowTile-1)/admitted.rowTile;
    EXPECT_LE(liveTiles,(std::uint64_t(2048)+admitted.rowTile-1)/admitted.rowTile);
    if(!live){EXPECT_EQ(liveTiles,0u);}
   }
  }
 }
}
/** @brief Reject invalid inputs and retain 64-bit grid arithmetic before admission checks. */
TEST(Q8ResidualGroupedGeometry, InvalidInputsAndWideCapacity) {
 for(auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm}) {
  EXPECT_THROW(q8ResidualGroupedGeometry(backend,0,64),std::invalid_argument);
  EXPECT_THROW(q8ResidualGroupedGeometry(backend,64,0),std::invalid_argument);
  EXPECT_THROW(q8ResidualGroupedGeometry(backend,-1,64),std::invalid_argument);
  EXPECT_THROW(q8ResidualGroupedGeometry(backend,64,-1),std::invalid_argument);
  const auto g=q8ResidualGroupedGeometry(backend,INT_MAX,INT_MAX);
  EXPECT_GT(g.blocks,std::uint64_t(UINT_MAX));
 }
 EXPECT_THROW(q8ResidualGroupedGeometry(static_cast<Q8ResidualGroupedBackend>(-1),64,64),std::invalid_argument);
}
