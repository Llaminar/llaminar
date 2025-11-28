/**
 * @file CpuAttentionKernelT.cpp
 * @brief Explicit instantiations of CpuAttentionKernelT template
 *
 * Explicit instantiations reduce compilation time and binary size.
 * All precision variants (FP32, BF16, FP16, INT32) compiled once here.
 *
 * @author David Sanftenberg
 */

#include "CpuAttentionKernelT.h"

namespace llaminar2
{
    // Explicit instantiations for all supported tensor types
    // Compiled once, reused across translation units

    template class CpuAttentionKernelT<FP32Tensor>;
    template class CpuAttentionKernelT<BF16Tensor>;
    template class CpuAttentionKernelT<FP16Tensor>;
    template class CpuAttentionKernelT<Q8_1Tensor>;

} // namespace llaminar2
