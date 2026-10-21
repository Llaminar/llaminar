/**
 * @file Qwen4ExpStateIOGPU.cuh
 * @brief Shared CUDA/HIP live-tail checkpoint and native fill kernels.
 *
 * Each word has one writer. The original source/destination intervals must be
 * disjoint or identical, making capture/restore exact without a byte-order or
 * arithmetic dependency on grouped rows or thread-block partitioning.
 */
#pragma once
#include "Qwen4ExpStateIO.h"
namespace llaminar2::qwen4exp::QWEN_STATE_GPU_NAMESPACE
{
/** @brief Copy exactly live native words and publish one compact count header. */
__global__ __launch_bounds__(256) void transfer(QwenStateTransfer p)
{
    const int count=p.rowSource==QwenStateRowSource::Fixed ? p.fixedRows :
        p.rowSource==QwenStateRowSource::CanonicalModuloFour ? *p.count%4 : *p.count;
    if(count<0 || count>p.capacityRows) {
#if defined(__HIPCC__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
    const auto word=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(word==0 && p.publishedRows)*p.publishedRows=count;
    if(word==0 && p.copiedBytes)*p.copiedBytes=std::uint64_t(count)*p.rowBytes;
    if(word==0 && p.aggregateCopiedBytes.bound() &&
       !addNativeCounter64(p.aggregateCopiedBytes,std::uint64_t(count)*p.rowBytes)) {
#if defined(__HIPCC__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
    if(word<std::size_t(count)*(p.rowBytes/4))
        static_cast<std::uint32_t *>(p.destination)[word]=static_cast<const std::uint32_t *>(p.source)[word];
}
/** @brief Reset each declared native word once, including exact EOS-token bits. */
__global__ __launch_bounds__(256) void fill(QwenStateFill p)
{
    const auto word=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(word<p.words)static_cast<std::uint32_t *>(p.destination)[word]=p.bits;
}
}
