/**
 * @file ROCmGatedNormSingleQ8.h
 * @brief Borrowed native operands for prepared-GEMM-owned gated normalization.
 *
 * The caller owns all original tensor/workspace leases and validates their
 * complete spans. One128-thread CTA per head retains the original norm tree,
 * full FP32 output and canonical SingleQ8 bytes/scales. This closed scalar
 * program does not admit counted rows or publish a new durable buffer owner.
 */
#pragma once
#include "kernels/common/GatedRMSNormProjection.h"
#include <cstdint>
namespace llaminar2::rocm
{
/** @brief Exact copied kernel ABI, with no pointer indirection or allocation. */
struct GatedNormSingleQ8Input
{
    const float* input;
    const float* gate;
    const float* gamma;
    float* output;
    int columns;
    GatedRMSNormProjectionParameters parameters;
    std::int8_t* quant;
    float* scales;
};
/**
 * @brief Submit the admitted original scalar normalization+SingleQ8 producer.
 * @param input Original FP32 tensors and existing separate byte/scale workspace banks.
 * @param device Exact execution participant owning every pointer.
 * @param stream Exact non-default stream, without substitution.
 * @return Native successful submission; invalid immutable geometry throws before enqueue.
 */
bool launchGatedNormSingleQ8(const GatedNormSingleQ8Input& input,int device,void* stream);
} // namespace llaminar2::rocm
