// CpuFeatures.cpp - CPU instruction set detection implementation
#include "utils/CpuFeatures.h"
#include "Logger.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace llaminar
{

    const CpuFeatures &CpuFeatures::instance()
    {
        static CpuFeatures instance;
        return instance;
    }

    CpuFeatures::CpuFeatures()
        : avx512f_(false), avx512_bf16_(false), avx512_fp16_(false), avx512_vnni_(false), f16c_(false), amx_bf16_(false)
    {
        detect_features();
    }

    // Helper to execute CPUID instruction - cross-platform
    static inline void cpuid_impl(unsigned int leaf, unsigned int subleaf,
                                  unsigned int *eax, unsigned int *ebx,
                                  unsigned int *ecx, unsigned int *edx)
    {
#ifdef _MSC_VER
        int cpuInfo[4];
        __cpuidex(cpuInfo, leaf, subleaf);
        *eax = cpuInfo[0];
        *ebx = cpuInfo[1];
        *ecx = cpuInfo[2];
        *edx = cpuInfo[3];
#elif defined(__x86_64__) || defined(__i386__)
        // Use inline assembly for GCC/Clang on x86/x86_64
        __asm__ __volatile__(
            "cpuid"
            : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "a"(leaf), "c"(subleaf));
#else
        // Unsupported architecture - return zeros
        *eax = *ebx = *ecx = *edx = 0;
#endif
    }

    void CpuFeatures::detect_features()
    {
        unsigned int eax, ebx, ecx, edx;

        // Get max supported CPUID function
        cpuid_impl(0, 0, &eax, &ebx, &ecx, &edx);
        unsigned int max_level = eax;

        if (max_level == 0)
        {
            LOG_WARN("CPUID instruction not supported or unavailable");
            return;
        }

        if (max_level >= 1)
        {
            // Function 1: Feature flags
            cpuid_impl(1, 0, &eax, &ebx, &ecx, &edx);
            f16c_ = (ecx & (1 << 29)) != 0; // ECX bit 29
        }

        if (max_level >= 7)
        {
            // Function 7, sub-leaf 0: Extended features
            cpuid_impl(7, 0, &eax, &ebx, &ecx, &edx);
            avx512f_ = (ebx & (1 << 16)) != 0;     // EBX bit 16
            avx512_vnni_ = (ecx & (1 << 11)) != 0; // ECX bit 11
            amx_bf16_ = (edx & (1 << 22)) != 0;    // EDX bit 22 (AMX-BF16)

            // Function 7, sub-leaf 1: More extended features
            cpuid_impl(7, 1, &eax, &ebx, &ecx, &edx);
            avx512_bf16_ = (eax & (1 << 5)) != 0;  // EAX bit 5
            avx512_fp16_ = (eax & (1 << 23)) != 0; // EAX bit 23
        }

        // Log detected features at INFO level (important for debugging perf issues)
        LOG_INFO("CPU Features: "
                 << "AVX512F=" << (avx512f_ ? "YES" : "NO")
                 << " AVX512_BF16=" << (avx512_bf16_ ? "YES" : "NO")
                 << " AMX_BF16=" << (amx_bf16_ ? "YES" : "NO")
                 << " AVX512_FP16=" << (avx512_fp16_ ? "YES" : "NO")
                 << " AVX512_VNNI=" << (avx512_vnni_ ? "YES" : "NO")
                 << " F16C=" << (f16c_ ? "YES" : "NO"));

        // Warn if AVX512 is present but BF16 is not (common on Cascade Lake)
        if (avx512f_ && !avx512_bf16_)
        {
            LOG_WARN("CPU has AVX512 but not AVX512_BF16 - BF16 GEMM will use slower emulation or FP32 fallback");
        }
    }

    std::string CpuFeatures::summary() const
    {
        std::ostringstream oss;
        oss << "AVX512F=" << (avx512f_ ? "YES" : "NO")
            << ", AVX512_BF16=" << (avx512_bf16_ ? "YES" : "NO")
            << ", AMX_BF16=" << (amx_bf16_ ? "YES" : "NO")
            << ", AVX512_FP16=" << (avx512_fp16_ ? "YES" : "NO")
            << ", AVX512_VNNI=" << (avx512_vnni_ ? "YES" : "NO")
            << ", F16C=" << (f16c_ ? "YES" : "NO");
        return oss.str();
    }

} // namespace llaminar
