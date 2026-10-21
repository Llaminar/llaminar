/**
 * @file MTPShiftedCacheOrigin.h
 * @brief One model-owned shifted-cache coordinate contract for host and device consumers.
 *
 * Published QFN predictors retain a visible token-zero/zero-hidden bootstrap.
 * Other families retain their existing successor-pair coordinates. This policy
 * is frozen from model identity, never user selected or inferred from counters.
 * Pure host/device arithmetic derives counts and packing from canonical state;
 * it stores neither a count mirror nor an independent allocation authority.
 */
#pragma once
#include <cstdint>
#include <climits>
#include <limits>
#include <stdexcept>
#include <optional>
#include <span>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_ORIGIN_HD __host__ __device__
#else
#define LLAMINAR_MTP_ORIGIN_HD
#endif
namespace llaminar2 {
/** @brief Immutable cache coordinate convention, authenticated by the model family. */
enum class MTPShiftedCacheOrigin : std::uint8_t { SuccessorPairsOnly, VisibleZeroHiddenTokenZero };
/** @return Whether the encoded policy is recognized. */
LLAMINAR_MTP_ORIGIN_HD constexpr bool validMTPShiftedCacheOrigin(MTPShiftedCacheOrigin origin) noexcept
{ return origin==MTPShiftedCacheOrigin::SuccessorPairsOnly || origin==MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero; }
/**
 * @brief Authenticate an archived or captured convention before any payload read/write.
 * @throws std::invalid_argument For an unknown tag or a convention from another model.
 */
inline void requireMTPShiftedCacheOrigin(MTPShiftedCacheOrigin expected,MTPShiftedCacheOrigin actual)
{
    if(!validMTPShiftedCacheOrigin(expected) || !validMTPShiftedCacheOrigin(actual) || expected!=actual)
        throw std::invalid_argument("Shifted cache origin differs from authenticated model ownership");
}
/** @return Logical prefix offset for one learned cache depth, or -1 for an impossible policy/depth. */
LLAMINAR_MTP_ORIGIN_HD constexpr int mtpShiftedPrefixOffset(MTPShiftedCacheOrigin origin,int depth=0) noexcept
{
    if(!validMTPShiftedCacheOrigin(origin) || depth<0 || depth==INT_MAX)return -1;
    if(origin==MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero)return depth==0?0:-1;
    return depth+1;
}
/** @return Exact shifted count for one nonnegative committed main prefix, or -1 for invalid input. */
LLAMINAR_MTP_ORIGIN_HD constexpr int mtpShiftedPrefixRows(int main_rows,MTPShiftedCacheOrigin origin,int depth=0) noexcept
{
    const int offset=mtpShiftedPrefixOffset(origin,depth);
    return main_rows<0 || offset<0 ? -1 : (main_rows>offset?main_rows-offset:0);
}
/** @brief Exact immutable main-block projection into one cache's ordinal coordinates. */
struct MTPShiftedBlockRange { int first,count,end; };
/**
 * @brief Project a complete logical prefix block with checked host arithmetic.
 * @param first Main block start; zero is a valid empty-prefix origin.
 * @param count Exact live main rows, never allocation capacity.
 * @param origin Authenticated cache origin, distinct from learned depth.
 * @param depth Learned cache ordinal; QFN has only depth0.
 * @return Exact contiguous shifted range, preserving the visible QFN seed.
 * @throws std::invalid_argument For unknown tags, negative ranges or unsupported learned depth.
 * @throws std::overflow_error For an unrepresentable main endpoint.
 */
inline MTPShiftedBlockRange mtpShiftedBlockRange(int first,int count,MTPShiftedCacheOrigin origin,int depth=0)
{
    if(first<0 || count<0 || mtpShiftedPrefixOffset(origin,depth)<0)
        throw std::invalid_argument("Shifted block requires a valid model-owned origin/depth and nonnegative live range");
    if(first>std::numeric_limits<int>::max()-count)throw std::overflow_error("Shifted main block endpoint overflows");
    const int begin=mtpShiftedPrefixRows(first,origin,depth),end=mtpShiftedPrefixRows(first+count,origin,depth);
    return {begin,end-begin,end};
}
/**
 * @brief Validate the host-owned compact CPU QFN prefill boundary before graph mutation.
 * @param origin Authenticated model convention; ordinary families preserve their existing admission.
 * @param first Absolute main prefix position before this compact segment.
 * @param rows Physical and live rows, which must be identical for the initial R1 CPU composer.
 * @param requests Exact request count; QFN currently admits one compact request.
 * @param lengths Optional canonical request lengths, not a device-count mirror.
 * @param positions Optional admitted absolute positions, which must be contiguous from first.
 * @throws std::invalid_argument For padding, holes, inconsistent coordinates or unrepresentable endpoints.
 */
inline void requireMTPShiftedHostPrefill(MTPShiftedCacheOrigin origin,int first,int rows,int requests,
    std::optional<std::span<const int>> lengths={},std::optional<std::span<const int>> positions={})
{
    if(!validMTPShiftedCacheOrigin(origin))throw std::invalid_argument("Unknown shifted-prefill model origin");
    if(origin==MTPShiftedCacheOrigin::SuccessorPairsOnly)return;
    if(rows<=0 || first<0 || first>std::numeric_limits<int>::max()-rows || requests!=1 ||
       (lengths && (lengths->size()!=1 || lengths->front()!=rows)))
        throw std::invalid_argument("QFN CPU shifted prefill requires exact compact R1 live rows and valid prefix coordinates");
    if(positions) {
        if(positions->size()!=static_cast<size_t>(rows))
            throw std::invalid_argument("QFN CPU shifted prefill requires one exact compact position row");
        for(int row=0;row<rows;++row)
            if((*positions)[static_cast<size_t>(row)]!=first+row)
                throw std::invalid_argument("QFN CPU shifted prefill positions differ from canonical prefix coordinates");
    }
}
/**
 * @brief Exact pre-append cache frontier for host-owned accepted-row publication.
 * @param position Absolute main condition position; no allocation capacity is involved.
 * @param resident Number of already resident condition rows at this verifier boundary.
 * @param origin Immutable model-owned coordinate convention.
 * @return max(0,position-offset+resident), preserving ordinary position-zero behavior.
 * @throws std::invalid_argument For invalid inputs; std::overflow_error for an unrepresentable endpoint.
 */
inline int mtpShiftedAppendBoundaryRows(int position,int resident,MTPShiftedCacheOrigin origin)
{
    const int offset=mtpShiftedPrefixOffset(origin);
    if(position<0 || resident<0 || offset<0)
        throw std::invalid_argument("Shifted append boundary requires valid model origin and nonnegative live coordinates");
    if(position>std::numeric_limits<int>::max()-resident)
        throw std::overflow_error("Shifted absolute append position overflows");
    const int64_t endpoint=int64_t(position)-offset+resident;
    if(endpoint>std::numeric_limits<int>::max())throw std::overflow_error("Shifted append frontier overflows");
    return endpoint>0?static_cast<int>(endpoint):0;
}
/**
 * @brief Resolve only a real input-hidden row from canonical device cache progress.
 * @return Row in the current segment, or -1 when a seed/prior row is needed or progress is malformed.
 * A QFN seed/prior is packed only by ShiftedPrefillTransaction; this selector
 * cannot alias it to an unrelated current hidden row.
 */
LLAMINAR_MTP_ORIGIN_HD constexpr int mtpShiftedInputHiddenRow(
    int main_rows,int shifted_rows,int segment_rows,MTPShiftedCacheOrigin origin) noexcept
{
    const int offset=mtpShiftedPrefixOffset(origin);
    if(offset<0 || segment_rows<=0 || main_rows<segment_rows || shifted_rows<0)return -1;
    const int64_t row=int64_t(shifted_rows)-(main_rows-segment_rows)+offset-1;
    return row>=0 && row<segment_rows ? static_cast<int>(row) : -1;
}
/** @brief Checked producer plan derived only from canonical main/shifted device counters. */
struct MTPShiftedPrefillPlan {
    bool valid=false,bootstrap=false,prior=false;
    int append_rows=0,token_offset=0;
    /** @return Input hidden row; -1 means exact archived prior, -2 means the mandatory zero4H seed. */
    LLAMINAR_MTP_ORIGIN_HD constexpr int hiddenRow(int output_row)const noexcept
    {
        if(output_row<0 || output_row>=append_rows)return -3;
        if(output_row==0 && bootstrap)return -2;
        if(output_row==0 && prior)return -1;
        return output_row+token_offset-1;
    }
};
/**
 * @brief Derive exact initial/continuation packing without observing a host count shadow.
 * @param main_rows Canonical main count after this segment's forward.
 * @param shifted_rows Existing canonical shifted cache count.
 * @param segment_rows Exact live rows in this segment.
 * @param origin Frozen model convention embedded in capture identity.
 * @return Valid packing, or invalid state to trap/reject before writes. Counted GPU callers invoke this on device.
 */
LLAMINAR_MTP_ORIGIN_HD constexpr MTPShiftedPrefillPlan mtpShiftedPrefillPlan(
    int main_rows,int shifted_rows,int segment_rows,MTPShiftedCacheOrigin origin) noexcept
{
    if(!validMTPShiftedCacheOrigin(origin) || segment_rows<=0 || main_rows<segment_rows || shifted_rows<0)return {};
    const int begin=main_rows-segment_rows;
    if(origin==MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero) {
        if(shifted_rows==begin)return {true,begin==0,begin>0,segment_rows,0};
        if(begin<INT_MAX && shifted_rows==begin+1)
            return {true,false,false,segment_rows-1,1};
        return {};
    }
    if(shifted_rows==begin)return {true,false,false,segment_rows-1,1};
    if(begin>0 && shifted_rows==begin-1)return {true,false,true,segment_rows,0};
    return {};
}
} // namespace llaminar2
#undef LLAMINAR_MTP_ORIGIN_HD
