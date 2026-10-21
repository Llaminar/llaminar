/**
 * @file Qwen4ExpBranchConcatGPU.cuh
 * @brief One exact Dense0 layout kernel shared by CUDA and HIP.
 *
 * One CTA handles a contiguous 256-word tile in one branch half-row. All
 * branches borrow that row's live authority and integer loads/stores preserve
 * FP32 payloads exactly. No workspace, reductions or synchronization is needed.
 * The native kernel belongs to its compiler's translation unit so CUDA and HIP
 * registration stubs can coexist in one inference library without symbol aliasing.
 */
#pragma once
#include "Qwen4ExpBranchConcat.h"
namespace llaminar2::qwen4exp {
namespace {
/** @brief Copy one independently addressed live row tile; trap invalid counts. */
__global__ void branchConcatKernel(BranchConcatArgs a) {
    const int live=a.rows.activeRows();
    const int column=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    if(column>=a.hidden)return;
    const unsigned half=blockIdx.z,branch=half/2;
    for(std::size_t row=blockIdx.y;row<std::size_t(live);row+=gridDim.y){
        const auto *source=(half&1)?a.hiddenBranches+(row*4+branch)*a.hidden+column:
            a.embedding+row*a.hidden+column;
        reinterpret_cast<std::uint32_t*>(a.output)[(row*8+half)*a.hidden+column]=
            *reinterpret_cast<const std::uint32_t*>(source);
    }
}
} // namespace
}
