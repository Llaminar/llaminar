#include "CPUFlashAttentionKernelT.h"

namespace llaminar2
{
    template class CPUFlashAttentionKernelT<ActivationPrecision::FP32>;
    template class CPUFlashAttentionKernelT<ActivationPrecision::BF16>;
    template class CPUFlashAttentionKernelT<ActivationPrecision::FP16>;
}
