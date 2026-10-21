/**
 * @file QwenSparseReplayGPU.cuh
 * @brief Shared fixed-shape CUDA/HIP publication from immutable pre-append authority.
 *
 * One thread writes replay. Independent row threads read canonical source
 * metadata and publish each row once, so grouped launch geometry introduces
 * no races or partition-dependent reductions. Invalid device admission fails
 * before a dependent native cache append can execute successfully.
 */
#pragma once
#include "QwenSparseReplay.h"
namespace llaminar2::qwen4exp::QWEN_REPLAY_GPU_NAMESPACE
{
/** @brief Fatal device validation, with no host observation or repair. */
__device__ inline void require(bool valid)
{
    if(!valid) {
#if defined(__HIPCC__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
}
/** @brief Publish exact live horizons and stable ring origin for all admitted query rows. */
__global__ __launch_bounds__(256) void prepareReplay(QwenSparsePrepareReplay p)
{
    const int prefix=*p.sequence.cached_tokens,live=p.live_rows ? *p.live_rows : p.row_capacity,cursor=*p.sequence.cursor,capacity=p.sequence.capacity;
    const int firstPosition=p.position_ids ? p.position_ids[0] : p.first_position;
    require(cursor>=0 && cursor<capacity && prefix>=0 && prefix<=capacity && live>=0 && live<=p.row_capacity &&
        static_cast<long long>(prefix)+live<=capacity && firstPosition>=prefix &&
        static_cast<long long>(firstPosition)+live<=0x7fffffffLL);
    const int origin=cursor>=prefix ? cursor-prefix : cursor-prefix+capacity;
    const int row=int(blockIdx.x)*256+int(threadIdx.x);
    if(row==0)*p.replay={prefix,live,firstPosition-prefix};
    if(row>=p.row_capacity)return;
    if(row>=live) { p.row_parameters[row]={};return; }
    require(!p.position_ids || p.position_ids[row]==firstPosition+row);
    p.row_parameters[row]={prefix+row+1,capacity,firstPosition+row,0,origin,capacity};
}
} // namespace llaminar2::qwen4exp::QWEN_REPLAY_GPU_NAMESPACE
