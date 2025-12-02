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

    namespace detail
    {
        /**
         * @brief Internal: Perform actual AVX512 detection (called once)
         */
        inline bool detect_avx512()
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
         * @brief Internal: Perform actual AVX2 detection (called once)
         */
        inline bool detect_avx2()
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
         * @brief Internal: Perform actual AVX detection (called once)
         */
        inline bool detect_avx()
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
    } // namespace detail

    /**
     * @brief Check if CPU supports AVX512 Foundation (AVX512F)
     * @note Result is cached on first call - no cpuid overhead on subsequent calls
     */
    inline bool cpu_supports_avx512()
    {
        static const bool result = detail::detect_avx512();
        return result;
    }

    /**
     * @brief Check if CPU supports AVX2
     * @note Result is cached on first call - no cpuid overhead on subsequent calls
     */
    inline bool cpu_supports_avx2()
    {
        static const bool result = detail::detect_avx2();
        return result;
    }

    /**
     * @brief Check if CPU supports AVX
     * @note Result is cached on first call - no cpuid overhead on subsequent calls
     */
    inline bool cpu_supports_avx()
    {
        static const bool result = detail::detect_avx();
        return result;
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
     * @brief Check if CPU supports AVX512-FP16
     * @note AVX512-FP16 is supported on Intel Sapphire Rapids (4th gen Xeon) and later
     */
    inline bool cpu_supports_avx512_fp16()
    {
        if (!cpu_supports_avx512())
            return false;

        uint32_t regs[4];
        cpuid(7, 0, regs);
        return (regs[3] & (1 << 23)) != 0; // EDX bit 23 = AVX512_FP16
    }

    /**
     * @brief Check if CPU supports AVX512-BF16
     * @note AVX512-BF16 is supported on Intel Cooper Lake (3rd gen Xeon) and later
     */
    inline bool cpu_supports_avx512_bf16()
    {
        if (!cpu_supports_avx512())
            return false;

        uint32_t regs[4];
        cpuid(7, 1, regs);
        return (regs[0] & (1 << 5)) != 0; // EAX bit 5 = AVX512_BF16
    }

    /**
     * @brief Check if CPU supports AVX512-VNNI
     * @note AVX512-VNNI is supported on Intel Ice Lake (2nd gen Xeon) and later
     */
    inline bool cpu_supports_avx512_vnni()
    {
        if (!cpu_supports_avx512())
            return false;

        uint32_t regs[4];
        cpuid(7, 0, regs);
        return (regs[2] & (1 << 11)) != 0; // ECX bit 11 = AVX512_VNNI
    }

    /**
     * @brief Check if CPU supports AMX-BF16
     * @note AMX-BF16 is supported on Intel Sapphire Rapids (4th gen Xeon) and later
     */
    inline bool cpu_supports_amx_bf16()
    {
        uint32_t regs[4];
        cpuid(7, 0, regs);
        return (regs[3] & (1 << 22)) != 0; // EDX bit 22 = AMX_BF16
    }

    /**
     * @brief Check if CPU supports AMX-INT8
     * @note AMX-INT8 is supported on Intel Sapphire Rapids (4th gen Xeon) and later
     */
    inline bool cpu_supports_amx_int8()
    {
        uint32_t regs[4];
        cpuid(7, 0, regs);
        return (regs[3] & (1 << 25)) != 0; // EDX bit 25 = AMX_INT8
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

    /**
     * @brief Get L2 cache size in bytes (per core)
     * @return L2 cache size in bytes, or 0 if unknown
     *
     * Uses CPUID leaf 0x04 (Intel Deterministic Cache Parameters)
     * Iterates through cache levels to find L2 cache.
     *
     * For Xeon Gold 6238R: Returns 1048576 (1MB per core)
     * Note: Total L2 per socket = 28MB (28 cores × 1MB)
     */
    inline uint32_t cpu_l2_cache_size()
    {
        static uint32_t cached_size = 0;
        static bool detected = false;

        if (detected)
            return cached_size;

        // CPUID leaf 0x04: Deterministic Cache Parameters (Intel)
        // Iterate through cache levels (ECX = subleaf index)
        for (uint32_t subleaf = 0; subleaf < 16; ++subleaf)
        {
            uint32_t regs[4];
            cpuid(0x04, subleaf, regs);

            // EAX[4:0] = Cache Type (1=Data, 2=Instruction, 3=Unified)
            uint32_t cache_type = regs[0] & 0x1F;
            if (cache_type == 0)
                break; // No more cache levels

            // EAX[7:5] = Cache Level (1=L1, 2=L2, 3=L3)
            uint32_t cache_level = (regs[0] >> 5) & 0x7;

            // Look for L2 cache (level 2, unified or data)
            if (cache_level == 2 && (cache_type == 1 || cache_type == 3))
            {
                // Cache size = (Ways + 1) × (Partitions + 1) × (Line Size + 1) × (Sets + 1)
                uint32_t ways = ((regs[1] >> 22) & 0x3FF) + 1;       // EBX[31:22]
                uint32_t partitions = ((regs[1] >> 12) & 0x3FF) + 1; // EBX[21:12]
                uint32_t line_size = (regs[1] & 0xFFF) + 1;          // EBX[11:0]
                uint32_t sets = regs[2] + 1;                         // ECX

                cached_size = ways * partitions * line_size * sets;
                detected = true;
                return cached_size;
            }
        }

        // Fallback: couldn't detect, assume conservative 256KB
        cached_size = 256 * 1024;
        detected = true;
        return cached_size;
    }

    /**
     * @brief Get L3 cache size in bytes (shared across all cores)
     * @return L3 cache size in bytes, or 0 if unknown
     *
     * For Xeon Gold 6238R: Returns ~39MB (shared L3)
     */
    inline uint32_t cpu_l3_cache_size()
    {
        static uint32_t cached_size = 0;
        static bool detected = false;

        if (detected)
            return cached_size;

        // CPUID leaf 0x04: Deterministic Cache Parameters (Intel)
        for (uint32_t subleaf = 0; subleaf < 16; ++subleaf)
        {
            uint32_t regs[4];
            cpuid(0x04, subleaf, regs);

            uint32_t cache_type = regs[0] & 0x1F;
            if (cache_type == 0)
                break;

            uint32_t cache_level = (regs[0] >> 5) & 0x7;

            // Look for L3 cache (level 3, unified)
            if (cache_level == 3 && (cache_type == 1 || cache_type == 3))
            {
                uint32_t ways = ((regs[1] >> 22) & 0x3FF) + 1;
                uint32_t partitions = ((regs[1] >> 12) & 0x3FF) + 1;
                uint32_t line_size = (regs[1] & 0xFFF) + 1;
                uint32_t sets = regs[2] + 1;

                cached_size = ways * partitions * line_size * sets;
                detected = true;
                return cached_size;
            }
        }

        // Fallback: couldn't detect, assume conservative 8MB
        cached_size = 8 * 1024 * 1024;
        detected = true;
        return cached_size;
    }

    /**
     * @brief Get total L2 cache per socket (L2 per core × num cores)
     * @return Total L2 cache size in bytes
     *
     * For Xeon Gold 6238R with 28 cores: Returns 28MB
     * This is the actual working set available for NC blocking.
     */
    inline uint32_t cpu_l2_cache_total()
    {
        static uint32_t cached_size = 0;
        static bool detected = false;

        if (detected)
            return cached_size;

        // Get number of logical processors (hyperthreaded cores)
        uint32_t regs[4];
        cpuid(0x01, 0, regs);
        uint32_t logical_processors = (regs[1] >> 16) & 0xFF; // EBX[23:16]

        // For Intel with HT: divide by 2 to get physical cores
        // For simplicity, use logical_processors as upper bound
        uint32_t l2_per_core = cpu_l2_cache_size();

        // Conservative estimate: assume half of logical processors are physical cores
        // (covers HT case without over-estimating)
        uint32_t estimated_physical_cores = logical_processors / 2;
        if (estimated_physical_cores == 0)
            estimated_physical_cores = 1;

        cached_size = l2_per_core * estimated_physical_cores;
        detected = true;
        return cached_size;
    }

#else
    // Non-x86 platforms: no SIMD support, no vendor detection
    inline bool cpu_supports_avx512() { return false; }
    inline bool cpu_supports_avx2() { return false; }
    inline bool cpu_supports_avx() { return false; }
    inline bool cpu_supports_sse41() { return false; }
    inline bool cpu_supports_avx512_fp16() { return false; }
    inline bool cpu_supports_avx512_bf16() { return false; }
    inline bool cpu_supports_avx512_vnni() { return false; }
    inline bool cpu_supports_amx_bf16() { return false; }
    inline bool cpu_supports_amx_int8() { return false; }
    inline const char *cpu_vendor() { return "Unknown"; }
    inline bool cpu_is_intel() { return false; }
    inline uint32_t cpu_l2_cache_size() { return 256 * 1024; }       // Conservative 256KB
    inline uint32_t cpu_l3_cache_size() { return 8 * 1024 * 1024; }  // Conservative 8MB
    inline uint32_t cpu_l2_cache_total() { return 8 * 1024 * 1024; } // Conservative 8MB
#endif

} // namespace llaminar2
