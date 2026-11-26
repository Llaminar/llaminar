/**
 * @file CUDARetainedHCFFNSuffixShard1.cu
 * @brief Sharded 1D Gate/Up and Down suffix instantiations (IQ codebooks + IQ4XSSourceScale).
 */
#include "CUDARetainedHCFFNSuffix.cuh"

namespace llaminar2::cuda::moe
{
    const void* retainedQFNM1SuffixGateUpShard1(uint8_t codebook) noexcept
    {
#if CUDART_VERSION >= 12080
        using namespace retained_suffix_detail;
        switch (codebook)
        {
        case 4: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<4>);
        case 11: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<11>);
        case 12: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<12>);
        case 13: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<13>);
        case 14: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<14>);
        case 15: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<15>);
        case 16: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<16>);
        case 17: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<17>);
        case kNativeQuantisedIQ4XSSourceScaleCodebook:
            return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<kNativeQuantisedIQ4XSSourceScaleCodebook>);
        case kRetainedSuffixDynamicCodebook:
            return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<kRetainedSuffixDynamicCodebook>);
        default: return nullptr;
        }
#else
        (void)codebook;
        return nullptr;
#endif
    }

    const void* retainedQFNM1SuffixDownShard1(uint8_t codebook) noexcept
    {
#if CUDART_VERSION >= 12080
        using namespace retained_suffix_detail;
        switch (codebook)
        {
        case 4: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<4>);
        case 11: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<11>);
        case 12: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<12>);
        case 13: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<13>);
        case 14: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<14>);
        case 15: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<15>);
        case 16: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<16>);
        case 17: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<17>);
        case kNativeQuantisedIQ4XSSourceScaleCodebook:
            return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<kNativeQuantisedIQ4XSSourceScaleCodebook>);
        case kRetainedSuffixDynamicCodebook:
            return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<kRetainedSuffixDynamicCodebook>);
        default: return nullptr;
        }
#else
        (void)codebook;
        return nullptr;
#endif
    }
} // namespace llaminar2::cuda::moe
