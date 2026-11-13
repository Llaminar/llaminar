/**
 * @file HC103OriginalGPU.h
 * @brief Test-only frozen original HC activation/read/publication native bridge.
 *
 * The fixture owns exact prepared source descriptors, banks and explicit stream.
 * No serving implementation may include or invoke this independent control.
 */
#pragma once
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "kernels/common/Q8ResidualCorrection.h"
namespace llaminar2::test103 {
#ifdef HAVE_CUDA
/** @brief Original CUDA divide-by-four SiLU; explicit-stream capture only. */
void cudaOriginalDown(qwen4exp::ResidualDownArgs,void*);
/** @brief Original CUDA ordered residual read. */
void cudaOriginalRead(qwen4exp::ResidualReadArgs,void*);
/** @brief Original CUDA two-term Q8 publication into independent retained banks. */
void cudaOriginalPublish(GPUQ8ResidualCorrectionArgs,void*);
#endif
#ifdef HAVE_ROCM
/** @brief Original HIP divide-by-four SiLU; explicit-stream capture only. */
void rocmOriginalDown(qwen4exp::ResidualDownArgs,void*);
/** @brief Original HIP ordered residual read. */
void rocmOriginalRead(qwen4exp::ResidualReadArgs,void*);
/** @brief Original HIP two-term Q8 publication into independent retained banks. */
void rocmOriginalPublish(GPUQ8ResidualCorrectionArgs,void*);
#endif
}
