/**
 * @file Q8ResidualGroupedGeometry.h
 * @brief Immutable geometry for two-term native-Q8 grouped projection.
 *
 * The retained physical matrix determines the launch identity. The kernel reads
 * the canonical device live count, never a host mirror. Cooperative matrix
 * tiles become useful only when the admitted geometry contains one complete
 * row and column tile; smaller matrices retain the native vector specialization.
 * No model-specific dimensions, environment dispatch or extra scratch is used.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
namespace llaminar2 {
/** @brief Native arithmetic engine; gfx906 uses grouped integer dots. */
enum class Q8ResidualGroupedBackend { CUDA, ROCm };
/** @brief Explicit compiled launch family, independent of mutable live state. */
enum class Q8ResidualGroupedStrategy { RowVector, CooperativeMatrix };
/** @brief Complete physical tile geometry; every field is derivable from captured operands. */
struct Q8ResidualGroupedGeometry {
 Q8ResidualGroupedStrategy strategy;
 int rowTile,columnTile,threads;
 std::uint64_t blocks;
 bool operator==(const Q8ResidualGroupedGeometry&)const=default;
};
/**
 * @brief Bound per-thread accumulators for the original correction program.
 * @param backend Compiled native arithmetic engine.
 * @param codebook Actual immutable execution format embedded in capture.
 * @return Cooperative output-column extent shared by admission and the kernel.
 *
 * IQ1_M retains four signed subgroup corrections for each of two terms.
 * Its CUDA tile uses half as many output accumulators per thread so the
 * unchanged dual-dot/correction tree fits the two-CTA register envelope.
 * Other formats, including the existing Q8 fast specialization, retain64.
 */
[[nodiscard]]
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline constexpr int q8ResidualCooperativeColumns(
 Q8ResidualGroupedBackend backend,std::uint8_t codebook) noexcept {
 return backend==Q8ResidualGroupedBackend::CUDA&&codebook==17?32:64;
}
/**
 * @brief Resolve a backend-native tile without querying a device or reading counts.
 * @param backend Compiled participant arithmetic engine.
 * @param rows Admitted physical rows, not a host-observed live count.
 * @param columns Prepared native output width, including valid partial final columns.
 * @param codebook Immutable execution format; default19 preserves the original Q8 policy.
 * @return Exact immutable launch family and bounded grid.
 * @throws std::invalid_argument For invalid geometry or backend.
 */
inline Q8ResidualGroupedGeometry q8ResidualGroupedGeometry(
 Q8ResidualGroupedBackend backend,int rows,int columns,std::uint8_t codebook=19) {
 if(rows<=0||columns<=0||(backend!=Q8ResidualGroupedBackend::CUDA&&backend!=Q8ResidualGroupedBackend::ROCm))
  throw std::invalid_argument("Two-term grouped projection needs positive geometry and an explicit backend");
 const int matrixRows=backend==Q8ResidualGroupedBackend::CUDA?64:32;
 const int matrixColumns=q8ResidualCooperativeColumns(backend,codebook);
 const bool matrix=rows>=matrixRows&&columns>=matrixColumns;
 const int rowTile=matrix?matrixRows:(rows==1?1:4),columnTile=matrix?matrixColumns:128;
 return {matrix?Q8ResidualGroupedStrategy::CooperativeMatrix:Q8ResidualGroupedStrategy::RowVector,
  rowTile,columnTile,matrix?256:128,
  ((static_cast<std::uint64_t>(rows)+rowTile-1)/rowTile)*
  ((static_cast<std::uint64_t>(columns)+columnTile-1)/columnTile)};
}
}
