/** @file LocalHotRecordedResponse.cuh
 * @brief Captured immutable response trace producer for production L1 kernel replay.
 *
 * The native bytes are authored fixture inputs copied from the same immutable
 * provider rows. The pool interval is a recorded replay coordinate, explicitly
 * not evidence that any L2 transfer, grant, service lookup or Rank cohort ran.
 */
#pragma once
#include "LocalHotRecordedResponse.h"
namespace llaminar2::test::local_hot_recorded {
namespace q=qwen4exp;
namespace t=q::local_hot_transport;
/** @brief Typed root subrange from the admitted frame, without another owner or allocation. */
template<class T> __device__ T* at(std::byte* root,std::size_t offset)
{return reinterpret_cast<T*>(root+offset);}
/** @brief Publish genuine original admission, or a recorded compact native response after actual Probe. */
static __global__ void recordedInput(Action action,Args a)
{
    if(action==Action::ForwardOriginalAdmission) {
        if(threadIdx.x==0)t::publishSourceAdmission({a.table,a.source_generation,a.original_admission,a.original_invocation,a.admission});
        return;
    }
    auto* control=at<q::adaptive_lookup::FrameControl>(a.frame,a.offsets.control);
    auto* identity=at<q::local_hot::FrameIdentity>(a.frame,a.offsets.identity);
    auto* misses=at<std::uint64_t>(a.frame,a.offsets.miss_keys);
    if(threadIdx.x==0) {
        const std::uint64_t received=sizeof(t::SourceAdmissionHeader);
        t::prepareRequest({&identity->query,&control->original_ids,&control->source_rows,a.admission,&received,
            a.consumer,identity->intent,a.request,a.lane,a.key_bytes});
    }
    __syncthreads();
    if(!control->source_rows)return;
    auto* native=at<std::byte>(a.frame,a.offsets.response);
    for(std::uint64_t i=threadIdx.x;i<control->source_rows*a.row_bytes;i+=blockDim.x) {
        const auto row=misses[i/a.row_bytes];unsigned found=a.recorded_count;
        for(unsigned j=0;j<a.recorded_count;++j)if(a.recorded_keys[j]==row){found=j;break;}
        if(found<a.recorded_count)native[i]=a.recorded_rows[std::uint64_t(found)*a.row_bytes+i%a.row_bytes];
    }
    __syncthreads();
    if(threadIdx.x==0) {
        auto* header=at<t::ResultHeader>(a.frame,a.offsets.result_header);
        auto* payload=at<std::uint64_t>(a.frame,a.offsets.received_payload_bytes);
        // Recorded response coordinate, not a minted actual pool authority.
        const q::pooled_native::p::Interval recorded{a.table,a.admission->source_admission,identity->query.sequence};
        t::prepareResult(*a.request,a.consumer,*a.admission,recorded,*header,*payload);
        *at<std::uint64_t>(a.frame,a.offsets.received_header_bytes)=sizeof(t::ResultHeader);
    }
}
}
