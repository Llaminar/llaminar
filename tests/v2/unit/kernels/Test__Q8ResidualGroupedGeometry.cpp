/**
 * @file Test__Q8ResidualGroupedGeometry.cpp
 * @brief Device-free totality, partial-tile and launch identity regressions.
 *
 * These tests exercise the actual immutable production geometry authority.
 * They prove dispatch and physical coverage, not device arithmetic or speed.
 */
#include <gtest/gtest.h>
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include "kernels/common/Q8ResidualShortKGeometry.h"
#include "tensors/NativeVnniFormatInfo.h"
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

/**
 * @test Every original source selects its exact row/column crossover and tail grid.
 *
 * IQ1_M needs fewer simultaneous CUDA accumulators. Its narrower tile must
 * change the launch grid as well as the kernel, including ragged columns and
 * M1024/M2048; otherwise a spill fix could leave valid outputs unwritten.
 */
TEST(Q8ResidualGroupedGeometry, OriginalFormatOwnsCooperativeTileAndGrid)
{
    for(const auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm})
    for(const auto &source:native_vnni_formats::kAllSourceFormats)
    for(const int rows:{1,31,32,33,63,64,65,511,512,513,1023,1024,1025,2047,2048,2049})
    for(const int columns:{1,31,32,33,63,64,65,127,128,129})
    {
        const auto codebook=canonicalDeviceVnniCodebookId(*source.metadata);
        SCOPED_TRACE(::testing::Message()<<source.quant_type<<" rows="<<rows<<" columns="<<columns);
        const auto geometry=q8ResidualGroupedGeometry(backend,rows,columns,codebook);
        const bool narrow=backend==Q8ResidualGroupedBackend::CUDA &&
            codebook==native_vnni_formats::IQ1_M.codebook_id;
        // These are the independently expected launch-family crossover rules,
        // not a second call to the production cooperative-column helper.
        const int matrixRows=backend==Q8ResidualGroupedBackend::CUDA ? 64 : 32;
        const int matrixColumns=narrow ? 32 : 64;
        const bool matrix=rows>=matrixRows && columns>=matrixColumns;
        EXPECT_EQ(geometry.strategy,matrix ? Q8ResidualGroupedStrategy::CooperativeMatrix : Q8ResidualGroupedStrategy::RowVector);
        EXPECT_EQ(geometry.threads,matrix ? 256 : 128);
        EXPECT_EQ(geometry.rowTile,matrix ? matrixRows : rows==1 ? 1 : 4);
        EXPECT_EQ(geometry.columnTile,matrix ? matrixColumns : 128);
        if(!narrow)
            EXPECT_EQ(geometry,q8ResidualGroupedGeometry(backend,rows,columns));
        const auto coveredColumns=((std::uint64_t(columns)+geometry.columnTile-1)/geometry.columnTile)*geometry.columnTile;
        EXPECT_GE(coveredColumns,std::uint64_t(columns));
        EXPECT_LT(coveredColumns,std::uint64_t(columns)+geometry.columnTile);
        const auto rowTiles=(std::uint64_t(rows)+geometry.rowTile-1)/geometry.rowTile;
        EXPECT_EQ(geometry.blocks,rowTiles*(coveredColumns/geometry.columnTile));
        EXPECT_GE(rowTiles*geometry.rowTile,std::uint64_t(rows));
        EXPECT_LT(rowTiles*geometry.rowTile,std::uint64_t(rows)+geometry.rowTile);
    }
    // Below N32 or M64 IQ1_M stays row-vector; crossing either threshold
    // independently must not drop a partial row or a second output tile.
    EXPECT_EQ(q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::CUDA,64,31,
        native_vnni_formats::IQ1_M.codebook_id).blocks,16u);
    EXPECT_EQ(q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::CUDA,63,32,
        native_vnni_formats::IQ1_M.codebook_id).blocks,16u);
    EXPECT_EQ(q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::CUDA,64,32,
        native_vnni_formats::IQ1_M.codebook_id).blocks,1u);
    EXPECT_EQ(q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::CUDA,65,33,
        native_vnni_formats::IQ1_M.codebook_id).blocks,4u);
    EXPECT_EQ(q8ResidualGroupedGeometry(Q8ResidualGroupedBackend::CUDA,64,65,
        native_vnni_formats::IQ1_M.codebook_id).blocks,3u);
}

/** @brief Every bounded K specialization owns complete/tail output coverage and finite CTA/LDS extents. */
TEST(Q8ResidualShortKGeometry, EveryContributionGroupAndAspectBoundary) {
 for(auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm})
  for(int k=64;k<=512;k+=32) {
   EXPECT_FALSE(q8ResidualShortKGeometry(backend,1,4*k-1,k,19));
   for(int n:{4*k,4*k+1,4*k+31,4*k+32,4*k+33,4*k+63,4*k+64,4*k+65,INT_MAX}) {
    const auto g=q8ResidualShortKGeometry(backend,1,n,k,19);
    ASSERT_TRUE(g);
    EXPECT_EQ(g->groups,k/32); EXPECT_EQ(g->columns,32);
    EXPECT_EQ(g->threads,(k/32)*32); EXPECT_LE(g->threads,512);
    EXPECT_EQ(g->shared_bytes,std::size_t(k/32)*32*2*sizeof(float));
    EXPECT_LE(g->shared_bytes,4096u);
    EXPECT_GE(g->blocks*32,std::uint64_t(n));
    EXPECT_LT((g->blocks-1)*32,std::uint64_t(n));
   }
  }
}
/** @brief Ordered short-K never enters grouped/other-format/long-K or narrow-N operations. */
TEST(Q8ResidualShortKGeometry, UnsupportedGeometryRemainsExplicitlyInapplicable) {
 for(auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm}) {
  for(int m:{2,3,4,64,65,512,2048}) EXPECT_FALSE(q8ResidualShortKGeometry(backend,m,10240,320,19));
  for(int k:{32,544,1024,8192}) EXPECT_FALSE(q8ResidualShortKGeometry(backend,1,10240,k,19));
  for(int n:{1,31,32,33,63,64,65}) EXPECT_FALSE(q8ResidualShortKGeometry(backend,1,n,320,19));
  for(const auto &format:native_vnni_formats::kAllSourceFormats) {
   if(format.metadata->codebook_id!=19)
    EXPECT_FALSE(q8ResidualShortKGeometry(backend,1,10240,320,format.metadata->codebook_id));
  }
 }
}
/** @brief Exhaust all retained M1..4096 and K32-domain boundaries without live-state probes. */
TEST(Q8ResidualShortKGeometry, AllRowsAndShortKDomainAreExact) {
 EXPECT_EQ(kQ8ResidualShortKColumns,32);
 for(auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm})
  for(int m=1;m<=4096;++m)
   for(int k=32;k<=1024;k+=32)
    for(int n:{4*k-1,4*k,4*k+1}) {
     const auto g=q8ResidualShortKGeometry(backend,m,n,k,19);
     const bool eligible=m==1&&k>=64&&k<=512&&n>=4*k;
     ASSERT_EQ(bool(g),eligible) << "M=" << m << " N=" << n << " K=" << k;
     if(g) {
      EXPECT_EQ(g->groups,k/32);
      EXPECT_EQ(g->threads,(k/32)*32);
      EXPECT_EQ(g->shared_bytes,std::size_t(k/32)*32*2*sizeof(float));
      EXPECT_EQ(g->blocks,(std::uint64_t(n)+31)/32);
     }
    }
}
/** @brief Every physical codebook and representative non-short K has one explicit family. */
TEST(Q8ResidualShortKGeometry, EveryPhysicalCodebookAndExtremeExtent) {
 for(auto backend:{Q8ResidualGroupedBackend::CUDA,Q8ResidualGroupedBackend::ROCm}) {
  for(int code=0;code<=255;++code)
   for(int m:{1,2,64,4096})
    for(int k:{32,64,320,512,544,8192}) {
     const auto g=q8ResidualShortKGeometry(backend,m,4*k,k,static_cast<std::uint8_t>(code));
     EXPECT_EQ(bool(g),m==1&&code==19&&k>=64&&k<=512);
    }
  EXPECT_FALSE(q8ResidualShortKGeometry(backend,1,INT_MAX,INT_MAX-31,19));
  EXPECT_FALSE(q8ResidualShortKGeometry(backend,INT_MAX,INT_MAX,320,19));
 }
}
/** @brief Reject malformed physical geometry before any production submission. */
TEST(Q8ResidualShortKGeometry, InvalidInputIsFatal) {
 const auto b=Q8ResidualGroupedBackend::ROCm;
 EXPECT_THROW(q8ResidualShortKGeometry(b,0,10240,320,19),std::invalid_argument);
 EXPECT_THROW(q8ResidualShortKGeometry(b,1,0,320,19),std::invalid_argument);
 EXPECT_THROW(q8ResidualShortKGeometry(b,1,10240,0,19),std::invalid_argument);
 EXPECT_THROW(q8ResidualShortKGeometry(b,1,10240,321,19),std::invalid_argument);
 EXPECT_THROW(q8ResidualShortKGeometry(static_cast<Q8ResidualGroupedBackend>(-1),1,10240,320,19),std::invalid_argument);
}
