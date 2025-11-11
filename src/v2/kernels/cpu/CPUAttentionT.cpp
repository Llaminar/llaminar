/**
 * @file CPUAttentionT.cpp
 * @brief Explicit instantiations of CPUAttentionT template
 *
 * Explicit instantiations reduce compilation time and binary size.
 * All precision variants (FP32, BF16, FP16, INT32) compiled once here.
 *
 * @author David Sanftenberg
 */

#include "CPUAttentionT.h"

namespace llaminar2
{
    // Explicit instantiations for all supported tensor types
    // Compiled once, reused across translation units

    template class CPUAttentionT<FP32Tensor>;
    template class CPUAttentionT<BF16Tensor>;
    template class CPUAttentionT<FP16Tensor>;
    template class CPUAttentionT<INT32Tensor>;
    template class CPUAttentionT<Q8_0Tensor>;

} // namespace llaminar2
