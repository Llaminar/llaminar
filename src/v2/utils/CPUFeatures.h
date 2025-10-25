#pragma once

#include <cstdint>

/**
 * @file CPUFeatures.h
 * @brief CPU SIMD feature detection for v2 architecture
 * @author David Sanftenberg
 *
 * Detects AVX512, AVX2, AVX, SSE4.1 at runtime via CPUID.
 * Used to select optimal SIMD decode paths for quantized tensors.
 */

namespace llaminar2
{

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

    /**
     * @brief Execute CPUID instruction
     * @param eax Input EAX value
     * @param ecx Input ECX value
     * @param[out] out Array of 4 uint32_t to receive [EAX, EBX, ECX, EDX]
     */
    inline void cpuid(uint32_t eax, uint32_t ecx, uint32_t out[4])
    {
#if defined(_MSC_VER)
        __cpuidex(reinterpret_cast<int *>(out), eax, ecx);
#elif defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__(
            "cpuid"
            : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
            : "a"(eax), "c"(ecx));
#else
        out[0] = out[1] = out[2] = out[3] = 0;
#endif
    }

    /**
     * @brief Check if CPU supports AVX512 Foundation (AVX512F)
     */
    inline bool cpu_supports_avx512()
    {
        uint32_t regs[4];
        cpuid(1, 0, regs);
        bool osxsave = (regs[2] & (1 << 27)) != 0;
        if (!osxsave)
            return false;

        // Check OS supports AVX512 (ZMM state)
        uint32_t xcr0;
#if defined(_MSC_VER)
        xcr0 = static_cast<uint32_t>(_xgetbv(0));
#elif defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#else
        return false;
#endif
        constexpr uint32_t AVX512_MASK = 0xE6; // XMM, YMM, opmask, ZMM_Hi256, Hi16_ZMM
        if ((xcr0 & AVX512_MASK) != AVX512_MASK)
            return false;

        // Check CPUID for AVX512F
        cpuid(7, 0, regs);
        return (regs[1] & (1 << 16)) != 0; // EBX bit 16 = AVX512F
    }

    /**
     * @brief Check if CPU supports AVX2
     */
    inline bool cpu_supports_avx2()
    {
        uint32_t regs[4];
        cpuid(1, 0, regs);
        bool osxsave = (regs[2] & (1 << 27)) != 0;
        if (!osxsave)
            return false;

        // Check OS supports AVX (YMM state)
        uint32_t xcr0;
#if defined(_MSC_VER)
        xcr0 = static_cast<uint32_t>(_xgetbv(0));
#elif defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#else
        return false;
#endif
        constexpr uint32_t AVX_MASK = 0x06; // XMM and YMM
        if ((xcr0 & AVX_MASK) != AVX_MASK)
            return false;

        // Check CPUID for AVX2
        cpuid(7, 0, regs);
        return (regs[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
    }

    /**
     * @brief Check if CPU supports AVX
     */
    inline bool cpu_supports_avx()
    {
        uint32_t regs[4];
        cpuid(1, 0, regs);
        bool osxsave = (regs[2] & (1 << 27)) != 0;
        bool avx = (regs[2] & (1 << 28)) != 0;
        if (!osxsave || !avx)
            return false;

        // Check OS supports AVX
        uint32_t xcr0;
#if defined(_MSC_VER)
        xcr0 = static_cast<uint32_t>(_xgetbv(0));
#elif defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#else
        return false;
#endif
        constexpr uint32_t AVX_MASK = 0x06;
        return (xcr0 & AVX_MASK) == AVX_MASK;
    }

    /**
     * @brief Check if CPU supports SSE4.1
     */
    inline bool cpu_supports_sse41()
    {
        uint32_t regs[4];
        cpuid(1, 0, regs);
        return (regs[2] & (1 << 19)) != 0; // ECX bit 19 = SSE4.1
    }

    /**
     * @brief Detect CPU vendor
     * @return "GenuineIntel" for Intel, "AuthenticAMD" for AMD, or other vendor string
     */
    inline const char *cpu_vendor()
    {
        static char vendor[13] = {0};
        static bool detected = false;

        if (!detected)
        {
            uint32_t regs[4];
            cpuid(0, 0, regs);
            // EBX, EDX, ECX contain 12-character vendor string
            *reinterpret_cast<uint32_t *>(vendor + 0) = regs[1]; // EBX
            *reinterpret_cast<uint32_t *>(vendor + 4) = regs[3]; // EDX
            *reinterpret_cast<uint32_t *>(vendor + 8) = regs[2]; // ECX
            vendor[12] = '\0';
            detected = true;
        }

        return vendor;
    }

    /**
     * @brief Check if CPU is Intel
     */
    inline bool cpu_is_intel()
    {
        const char *vendor = cpu_vendor();
        return vendor[0] == 'G' && vendor[1] == 'e' && vendor[2] == 'n' &&
               vendor[3] == 'u' && vendor[4] == 'i' && vendor[5] == 'n' &&
               vendor[6] == 'e' && vendor[7] == 'I' && vendor[8] == 'n' &&
               vendor[9] == 't' && vendor[10] == 'e' && vendor[11] == 'l';
    }

#else
    // Non-x86 platforms: no SIMD support, no vendor detection
    inline bool cpu_supports_avx512() { return false; }
    inline bool cpu_supports_avx2() { return false; }
    inline bool cpu_supports_avx() { return false; }
    inline bool cpu_supports_sse41() { return false; }
    inline const char *cpu_vendor() { return "Unknown"; }
    inline bool cpu_is_intel() { return false; }
#endif

} // namespace llaminar2
