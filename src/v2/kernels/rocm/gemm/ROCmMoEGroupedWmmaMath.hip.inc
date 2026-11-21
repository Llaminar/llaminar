/**
 * @file ROCmMoEGroupedWmmaMath.hip.inc
 * @brief Shared exact contribution and statically indexed row helpers.
 *
 * Included inside each native shard's detail namespace. These helpers own no
 * state or kernels; per-translation-unit weight lookup tables remain separate.
 */
using I32x4 = int __attribute__((ext_vector_type(4)));

/** @brief Commit one K32 integer result through the common source-exact FP32 authority.
 * @tparam FMT Physical decoder. @param acc Prior partition result.
 * @param low First half dot. @param high Second half or complete single-scale dot.
 * @param weight Decoded weight scales. @param aux Auxiliary scale/minimum word.
 * @param minima Dual minimum words. @param qh IQ1_M signs. @param scale_a Activation scale.
 * @param sums Exact activation sums, with unused components zero.
 * @return One canonical rounded block contribution accumulated in order. */
template<NativeQuantisedGemmFormat FMT>
__device__ __forceinline__ float wideCommit(float acc, int low, int high,
    ROCmNativeQuantisedDualWeightScales weight, uint16_t aux, uint32_t minima,
    uint16_t qh, float scale_a, I32x4 sums)
{
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    if constexpr (Traits::is_dual_scale_asym)
    {
        const uint16_t ml = minima, mh = minima >> 16;
        return rocm_native_quantised_commit_corrected_dual_scale_block(acc,low,high,
            weight.low,weight.high,sums[0],sums[1],
            __half2float(*reinterpret_cast<const __half *>(&ml)),
            __half2float(*reinterpret_cast<const __half *>(&mh)),scale_a);
    }
    else if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
        return rocm_native_quantised_commit_iq1_m_block(acc,low,high,
            sums[0],sums[1],sums[2],sums[3],
            qh & 0x0008 ? -0.125f : 0.125f,qh & 0x0080 ? -0.125f : 0.125f,
            qh & 0x0800 ? -0.125f : 0.125f,qh & 0x8000 ? -0.125f : 0.125f,
            weight.low,weight.high,scale_a);
    else if constexpr (Traits::is_dual_scale)
        return rocm_native_quantised_commit_dual_scale_block(acc,low,high,weight.low,weight.high,scale_a);
    else if constexpr (Traits::is_asymmetric)
        return rocm_native_quantised_commit_corrected_single_scale_block(acc,high,weight.low,sums[0],
            __half2float(*reinterpret_cast<const __half *>(&aux)),scale_a);
    else return rocm_native_quantised_commit_single_scale_block(acc,high,weight.low,scale_a);
}

/** @brief Visit live rows with compile-time accumulator indices.
 * @tparam Row Next static row. @tparam Count Row-tile extent. @tparam Visitor Scalar row operation.
 * @param live Actual rows in this tile. @param visit Operation borrowing one static index. */
template<int Row, int Count, class Visitor>
__device__ __forceinline__ void wideVisitRows(int live, const Visitor &visit)
{
    if constexpr (Row < Count)
        if (Row < live)
        {
            visit(std::integral_constant<int,Row>{});
            wideVisitRows<Row+1,Count>(live,visit);
        }
}
