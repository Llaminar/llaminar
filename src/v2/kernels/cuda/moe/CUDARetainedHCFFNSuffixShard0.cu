/**
 * @file CUDARetainedHCFFNSuffixShard0.cu
 * @brief Sharded 1D Gate/Up and Down suffix instantiations (standard/legacy/K-quant + ExpandedInt8Min).
 */
#include "CUDARetainedHCFFNSuffix.cuh"

namespace llaminar2::cuda::moe
{
    const void* retainedQFNM1SuffixGateUpShard0(uint8_t codebook) noexcept
    {
#if CUDART_VERSION >= 12080
        using namespace retained_suffix_detail;
        switch (codebook)
        {
        case 0: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<0>);
        case 5: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<5>);
        case 6: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<6>);
        case 7: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<7>);
        case 8: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<8>);
        case 9: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<9>);
        case 10: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<10>);
        case 19: return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<19>);
        case kNativeQuantisedExpandedInt8MinCodebook:
            return reinterpret_cast<const void*>(retainedQFNM1SuffixGateUp<kNativeQuantisedExpandedInt8MinCodebook>);
        default: return nullptr;
        }
#else
        (void)codebook;
        return nullptr;
#endif
    }

    const void* retainedQFNM1SuffixDownShard0(uint8_t codebook) noexcept
    {
#if CUDART_VERSION >= 12080
        using namespace retained_suffix_detail;
        switch (codebook)
        {
        case 0: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<0>);
        case 5: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<5>);
        case 6: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<6>);
        case 7: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<7>);
        case 8: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<8>);
        case 9: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<9>);
        case 10: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<10>);
        case 19: return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<19>);
        case kNativeQuantisedExpandedInt8MinCodebook:
            return reinterpret_cast<const void*>(retainedQFNM1SuffixDown<kNativeQuantisedExpandedInt8MinCodebook>);
        default: return nullptr;
        }
#else
        (void)codebook;
        return nullptr;
#endif
    }
} // namespace llaminar2::cuda::moe
