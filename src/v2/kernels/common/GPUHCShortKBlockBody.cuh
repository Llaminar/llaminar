/** @file GPUHCShortKBlockBody.cuh
 * @brief Original signed-Q8 K32 primary/residual HC contribution shared by standalone/cooperative entries.
 * No contribution, scale rounding or independent ascending-K fold is replaced.
 */
#pragma once
#include "Q8ResidualCorrection.h"
#include "DeviceNativeQuantisedContributionContract.h"
#if defined(__HIPCC__)
#include "kernels/rocm/common/ROCmSignedDot4.h"
#endif
namespace llaminar2::hc_short_k_native {
/** @brief Exact original signed four-byte dot. @param a First packed operand. @param b Original native weight. @param c Integer chain. @return Exact signed-dot accumulator. */
__device__ __forceinline__ int dot4(int a,int b,int c) {
#if defined(__HIPCC__)
 return ::llaminar2::rocm_integer_contract::signedDot4(a,b,c);
#else
 return __dp4a(a,b,c);
#endif
}
/** @brief Original generic-address acquisition, retained for every existing caller. */
struct OriginalWeightReader {
 /** @brief Read one original aligned weight operand.
  * @tparam T Existing int4 payload or half metadata word.
  * @param pointer Original prepared bytes. @return Identical original operand.
  */
 template<class T> __device__ __forceinline__ T read(const void* pointer) const {
  return *reinterpret_cast<const T*>(pointer);
 }
};
/** @brief Original complete K32 contribution.
 * @tparam Reader Weight-address policy; original generic acquisition by default.
 * @param a Original native source/banks. @param group Original K32 index.
 * @param column Original output column. @param p Primary term. @param q Residual term.
 * @param reader Acquisition only; integer chain and FP commit remain canonical.
 */
template<class Reader=OriginalWeightReader>
__device__ __forceinline__ void contribution(const GPUQ8ResidualCorrectionArgs& a,unsigned group,
 std::uint64_t column,float& p,float& q,Reader reader={}) {

  const auto linear=static_cast<std::uint64_t>(group)*a.weights.n+column;
  const auto*payload=a.weights.payload+linear*32;
  const int4 low=reader.template read<int4>(payload),high=reader.template read<int4>(payload+16);
  const int weights[8]{low.x,low.y,low.z,low.w,high.x,high.y,high.z,high.w};
  const auto bits=reader.template read<unsigned short>(static_cast<const unsigned short*>(a.weights.scales)+linear);
  const float weightScale=__half2float(__ushort_as_half(bits));
  const auto*p0=reinterpret_cast<const int*>(a.primary+group*32);
  const auto*p1=reinterpret_cast<const int*>(a.residual+group*32);
  int dot0=0,dot1=0;
#pragma unroll
  for(int word=0;word<8;++word) {
   dot0=hc_short_k_native::dot4(p0[word],weights[word],dot0);
   dot1=hc_short_k_native::dot4(p1[word],weights[word],dot1);
  }
  p=device_native_quantised_contract::singleScaleBlock(dot0,weightScale,a.primaryScales[group]);
  q=device_native_quantised_contract::singleScaleBlock(dot1,weightScale,a.residualScales[group]);

}
}
