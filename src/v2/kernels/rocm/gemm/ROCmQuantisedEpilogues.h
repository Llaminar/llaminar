/**
 * @file ROCmQuantisedEpilogues.h
 * @brief Canonical count-free and count-owned FP32 projection epilogue ABI.
 *
 * The retained ordinary ABI executes a compile-time count-free specialization.
 * Explicit counted callers copy one immutable row descriptor into kernel args;
 * its device owner supplies the live prefix on the exact execution stream.
 * Inactive rows are rejected before output reads, bias reads or output writes.
 * Both specializations retain the original multiply-then-bias arithmetic.
 */
#pragma once
#include "kernels/common/DeviceRowRange.h"
extern "C" {
/**
 * @brief Apply the original full-active FP32 epilogue.
 * @param output Borrowed row-major persistent FP32 destination [M,N].
 * @param bias Optional borrowed FP32 bias [N].
 * @param M Positive retained physical row count.
 * @param N Positive output width; M*N must fit the native INT32 index program.
 * @param alpha Original output multiplier, evaluated before optional bias.
 * @param device Exact owning ROCm device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @return Original native submission result, with no alternate arithmetic route.
 * @throws std::invalid_argument For invalid immutable geometry or stream metadata.
 */
bool rocmQuantGemm_applyFp32Epilogue(float* output,const float* bias,int M,int N,
    float alpha,int device,void* stream);
/**
 * @brief Apply the same arithmetic to device-count-owned live rows only.
 * @param output Borrowed row-major persistent FP32 destination [M,N].
 * @param bias Optional borrowed FP32 bias [N].
 * @param M Positive retained physical row count, never a downloaded live count.
 * @param N Positive output width; M*N must fit the native INT32 index program.
 * @param alpha Original output multiplier, evaluated before optional bias.
 * @param device Exact owning ROCm device ordinal.
 * @param stream Exact non-null, non-legacy, non-per-thread-default stream.
 * @param rows Optional immutable descriptor with exactly M physical rows.
 * @return Native submission result. Null/count-free rows retain ordinary specialization.
 * @throws std::invalid_argument For invalid immutable geometry, stream or row metadata.
 */
bool rocmQuantGemm_applyFp32Epilogue_rows(float* output,const float* bias,int M,int N,
    float alpha,int device,void* stream,const llaminar2::DeviceRowRange* rows);
}
