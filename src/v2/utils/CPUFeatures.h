#pragma once

#include <cstdint>
#include <cstdlib> // std::getenv for LLAMINAR_ISA_LEVEL
#include <cstdio>
#include <algorithm> // For std::min, std::max in AttentionCacheConfig

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
         * @brief Internal: Get maximum supported basic CPUID leaf
         */
        inline uint32_t max_basic_cpuid_leaf()
        {
            uint32_t regs[4];
            cpuid(0, 0, regs);
            return regs[0]; // EAX = max basic leaf
        }

        /**
         * @brief Internal: Get maximum supported extended CPUID leaf
         */
        inline uint32_t max_extended_cpuid_leaf()
        {
            uint32_t regs[4];
            cpuid(0x80000000, 0, regs);
            return regs[0]; // EAX = max extended leaf
        }

        /**
         * @brief Internal: Detect cache size for a given cache level using
         *        Intel leaf 0x04 or AMD extended leaf 0x8000001D.
         *
         * Both leaves use the same register format:
         *   EAX[4:0]  = Cache Type (1=Data, 2=Instruction, 3=Unified, 0=Null)
         *   EAX[7:5]  = Cache Level (1=L1, 2=L2, 3=L3)
         *   EBX[11:0] = Line Size - 1
         *   EBX[21:12]= Partitions - 1
         *   EBX[31:22]= Associativity (Ways) - 1
         *   ECX       = Number of Sets - 1
         *   EAX[31:26]= Max cores in package - 1 (Intel leaf 0x04 only)
         *
         * @param target_level Cache level to find (1=L1, 2=L2, 3=L3)
         * @param data_only If true, match only Data caches (type 1); if false,
         *        match Data or Unified (types 1 and 3)
         * @param[out] out_size Cache size in bytes
         * @param[out] out_max_cores_in_pkg If non-null and Intel leaf 0x04 was used,
         *        set to EAX[31:26]+1 (max addressable core IDs in package)
         * @return true if cache level was found, false if not detected
         */
        inline bool detect_cache_by_level(int target_level, bool data_only,
                                          uint32_t &out_size,
                                          uint32_t *out_max_cores_in_pkg = nullptr)
        {
            // Ordered: try Intel 0x04 first (more fields), then AMD 0x8000001D
            struct LeafInfo
            {
                uint32_t leaf;
                uint32_t min_supported;
                bool is_extended;
            };
            const LeafInfo leaves[] = {
                {0x04, 4, false},              // Intel Deterministic Cache Parameters
                {0x8000001D, 0x8000001D, true} // AMD Cache Topology
            };

            uint32_t max_basic = max_basic_cpuid_leaf();
            uint32_t max_ext = max_extended_cpuid_leaf();

            for (const auto &li : leaves)
            {
                // Check if this leaf is supported
                if (li.is_extended)
                {
                    if (max_ext < li.min_supported)
                        continue;
                }
                else
                {
                    if (max_basic < li.min_supported)
                        continue;
                }

                for (uint32_t subleaf = 0; subleaf < 16; ++subleaf)
                {
                    uint32_t regs[4];
                    cpuid(li.leaf, subleaf, regs);

                    uint32_t cache_type = regs[0] & 0x1F;
                    if (cache_type == 0)
                        break; // No more cache levels at this leaf

                    uint32_t cache_level = (regs[0] >> 5) & 0x7;

                    bool type_match = data_only
                                          ? (cache_type == 1)
                                          : (cache_type == 1 || cache_type == 3);

                    if (cache_level == static_cast<uint32_t>(target_level) && type_match)
                    {
                        uint32_t ways = ((regs[1] >> 22) & 0x3FF) + 1;
                        uint32_t partitions = ((regs[1] >> 12) & 0x3FF) + 1;
                        uint32_t line_size = (regs[1] & 0xFFF) + 1;
                        uint32_t sets = regs[2] + 1;

                        out_size = ways * partitions * line_size * sets;

                        // EAX[31:26] = max cores in package (Intel leaf 0x04 only)
                        if (out_max_cores_in_pkg && !li.is_extended)
                            *out_max_cores_in_pkg = ((regs[0] >> 26) & 0x3F) + 1;

                        return true;
                    }
                }
            }

            return false; // Not detected by any leaf
        }

        /**
         * @brief Internal: Get physical core count per package
         *
         * Tries, in order:
         *   1. Intel CPUID leaf 0x04 EAX[31:26]+1 (from any cache subleaf)
         *   2. AMD CPUID leaf 0x80000008 ECX[7:0]+1
         *   3. CPUID leaf 0x01 EBX[23:16] / 2 (fallback, assumes SMT-2)
         */
        inline uint32_t detect_physical_cores_per_package()
        {
            uint32_t regs[4];

            // Method 1: Intel leaf 0x04 — EAX[31:26]+1 from any valid subleaf
            if (max_basic_cpuid_leaf() >= 4)
            {
                cpuid(0x04, 0, regs);
                uint32_t cache_type = regs[0] & 0x1F;
                if (cache_type != 0)
                {
                    uint32_t cores = ((regs[0] >> 26) & 0x3F) + 1;
                    if (cores > 0)
                        return cores;
                }
            }

            // Method 2: AMD leaf 0x80000008 — ECX[7:0]+1
            if (max_extended_cpuid_leaf() >= 0x80000008)
            {
                cpuid(0x80000008, 0, regs);
                uint32_t cores = (regs[2] & 0xFF) + 1;
                if (cores > 1) // Sanity check: 0 means "use leaf 1"
                    return cores;
            }

            // Method 3: Fallback — CPUID.1 EBX[23:16] / 2
            cpuid(0x01, 0, regs);
            uint32_t logical = (regs[1] >> 16) & 0xFF;
            uint32_t physical = logical / 2;
            return (physical > 0) ? physical : 1;
        }

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

    // =========================================================================
    // ISA Level: runtime dispatch for refactored _scalar/_avx2/_avx512 variants
    // =========================================================================

    /**
     * @brief ISA level for runtime SIMD dispatch
     *
     * Resolved once at startup from hardware detection, overridable via
     * LLAMINAR_ISA_LEVEL=scalar|avx2|avx512 for testing.
     */
    enum class ISALevel : uint8_t
    {
        Scalar = 0,
        AVX2 = 1,
        AVX512 = 2
    };

    // Forward declarations for activeISALevel()
    inline bool cpu_supports_avx2();
    inline bool cpu_supports_avx512();

    /**
     * @brief Get the active ISA level (cached singleton)
     *
     * Resolution order:
     *   1. LLAMINAR_ISA_LEVEL env var (scalar / avx2 / avx512)
     *   2. Hardware detection via CPUID
     *
     * @note Result is cached on first call — zero overhead on subsequent calls.
     *       The env var is read with std::getenv to avoid a circular dependency
     *       on DebugEnv.h (which includes CPUFeatures.h).
     */
    inline ISALevel activeISALevel()
    {
        static const ISALevel level = []
        {
            const char *env = std::getenv("LLAMINAR_ISA_LEVEL");
            if (env)
            {
                // Case-insensitive compare without <strings.h>
                auto eq = [](const char *a, const char *b) -> bool
                {
                    for (; *a && *b; ++a, ++b)
                        if ((*a | 0x20) != (*b | 0x20))
                            return false;
                    return *a == *b;
                };
                if (eq(env, "scalar"))
                    return ISALevel::Scalar;
                if (eq(env, "avx2"))
                    return ISALevel::AVX2;
                if (eq(env, "avx512"))
                    return ISALevel::AVX512;
            }
            if (cpu_supports_avx512())
                return ISALevel::AVX512;
            if (cpu_supports_avx2())
                return ISALevel::AVX2;
            return ISALevel::Scalar;
        }();
        return level;
    }

// ---------------------------------------------------------------------------
// ISA_DISPATCH macros — runtime dispatch to _scalar / _avx2 / _avx512 variants
//
// Usage:
//   ISA_DISPATCH_VOID(fused_fp32_residual_add, out, a, b, n);
//   float result; ISA_DISPATCH_RET(result, activation_row_max_abs, row, len);
//   double val = ISA_DISPATCH_RETVAL(compute_sumsq, data, count);
//
// The _avx2 and _avx512 variants MUST be compiled (possibly as unreachable
// stubs) regardless of the host ISA so the switch is always complete.
// ---------------------------------------------------------------------------

/// Dispatch a void-returning function: name##_scalar / _avx2 / _avx512
#define ISA_DISPATCH_VOID(name, ...)           \
    do                                         \
    {                                          \
        switch (::llaminar2::activeISALevel()) \
        {                                      \
        case ::llaminar2::ISALevel::AVX512:    \
            name##_avx512(__VA_ARGS__);        \
            break;                             \
        case ::llaminar2::ISALevel::AVX2:      \
            name##_avx2(__VA_ARGS__);          \
            break;                             \
        default:                               \
            name##_scalar(__VA_ARGS__);        \
            break;                             \
        }                                      \
    } while (0)

/// Dispatch and assign: lhs = name##_{scalar|avx2|avx512}(...)
#define ISA_DISPATCH_RET(lhs, name, ...)       \
    do                                         \
    {                                          \
        switch (::llaminar2::activeISALevel()) \
        {                                      \
        case ::llaminar2::ISALevel::AVX512:    \
            lhs = name##_avx512(__VA_ARGS__);  \
            break;                             \
        case ::llaminar2::ISALevel::AVX2:      \
            lhs = name##_avx2(__VA_ARGS__);    \
            break;                             \
        default:                               \
            lhs = name##_scalar(__VA_ARGS__);  \
            break;                             \
        }                                      \
    } while (0)

/// Dispatch and return the value directly (for use inside a return statement
/// context or initializer).  Expands to an immediately-invoked lambda.
#define ISA_DISPATCH_RETVAL(name, ...) \
    [&]() -> decltype(name##_scalar(__VA_ARGS__)) {              \
        switch (::llaminar2::activeISALevel())                   \
        {                                                        \
        case ::llaminar2::ISALevel::AVX512:                      \
            return name##_avx512(__VA_ARGS__);                   \
        case ::llaminar2::ISALevel::AVX2:                        \
            return name##_avx2(__VA_ARGS__);                     \
        default:                                                 \
            return name##_scalar(__VA_ARGS__);                   \
        } }()

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
     * @note Result is cached on first call - no cpuid overhead on subsequent calls
     */
    inline bool cpu_supports_sse41()
    {
        static const bool result = []
        {
            uint32_t regs[4];
            cpuid(1, 0, regs);
            return (regs[2] & (1 << 19)) != 0; // ECX bit 19 = SSE4.1
        }();
        return result;
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
     * @note AMX-BF16 is supported on Intel Sapphire Rapids (4th gen Xeon) and later.
     *       Checks both CPUID feature bit AND OS support via XCR0 bits 17-18.
     */
    inline bool cpu_supports_amx_bf16()
    {
        static const bool result = []
        {
            // Check OSXSAVE first
            uint32_t regs[4];
            cpuid(1, 0, regs);
            bool osxsave = (regs[2] & (1 << 27)) != 0;
            if (!osxsave)
                return false;

            // Check XCR0 for AMX tile state (bits 17=XTILECFG, 18=XTILEDATA)
            uint32_t xcr0_lo;
            [[maybe_unused]] uint32_t xcr0_hi;
#if defined(_MSC_VER)
            uint64_t xcr0_full = _xgetbv(0);
            xcr0_lo = static_cast<uint32_t>(xcr0_full);
            xcr0_hi = static_cast<uint32_t>(xcr0_full >> 32);
#elif defined(__GNUC__) || defined(__clang__)
            __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
#else
            return false;
#endif
            // AMX requires bits 17 (XTILECFG) and 18 (XTILEDATA)
            constexpr uint32_t AMX_MASK = (1u << 17) | (1u << 18);
            if ((xcr0_lo & AMX_MASK) != AMX_MASK)
                return false;

            // Check CPUID for AMX-BF16
            cpuid(7, 0, regs);
            return (regs[3] & (1 << 22)) != 0; // EDX bit 22 = AMX_BF16
        }();
        return result;
    }

    /**
     * @brief Check if CPU supports AMX-INT8
     * @note AMX-INT8 is supported on Intel Sapphire Rapids (4th gen Xeon) and later.
     *       Checks both CPUID feature bit AND OS support via XCR0 bits 17-18.
     */
    inline bool cpu_supports_amx_int8()
    {
        static const bool result = []
        {
            // Check OSXSAVE first
            uint32_t regs[4];
            cpuid(1, 0, regs);
            bool osxsave = (regs[2] & (1 << 27)) != 0;
            if (!osxsave)
                return false;

            // Check XCR0 for AMX tile state (bits 17=XTILECFG, 18=XTILEDATA)
            uint32_t xcr0_lo;
            [[maybe_unused]] uint32_t xcr0_hi;
#if defined(_MSC_VER)
            uint64_t xcr0_full = _xgetbv(0);
            xcr0_lo = static_cast<uint32_t>(xcr0_full);
            xcr0_hi = static_cast<uint32_t>(xcr0_full >> 32);
#elif defined(__GNUC__) || defined(__clang__)
            __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
#else
            return false;
#endif
            constexpr uint32_t AMX_MASK = (1u << 17) | (1u << 18);
            if ((xcr0_lo & AMX_MASK) != AMX_MASK)
                return false;

            // Check CPUID for AMX-INT8
            cpuid(7, 0, regs);
            return (regs[3] & (1 << 25)) != 0; // EDX bit 25 = AMX_INT8
        }();
        return result;
    }

    /**
     * @brief Detect CPU vendor
     * @return "GenuineIntel" for Intel, "AuthenticAMD" for AMD, or other vendor string
     * @note Thread-safe via C++11 static local initialization guarantee
     */
    inline const char *cpu_vendor()
    {
        struct VendorString
        {
            char str[13];
            VendorString()
            {
                uint32_t regs[4];
                cpuid(0, 0, regs);
                // EBX, EDX, ECX contain 12-character vendor string
                *reinterpret_cast<uint32_t *>(str + 0) = regs[1]; // EBX
                *reinterpret_cast<uint32_t *>(str + 4) = regs[3]; // EDX
                *reinterpret_cast<uint32_t *>(str + 8) = regs[2]; // ECX
                str[12] = '\0';
            }
        };
        static const VendorString vendor;
        return vendor.str;
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
     * @brief Check if CPU is AMD
     */
    inline bool cpu_is_amd()
    {
        const char *vendor = cpu_vendor();
        return vendor[0] == 'A' && vendor[1] == 'u' && vendor[2] == 't' &&
               vendor[3] == 'h' && vendor[4] == 'e' && vendor[5] == 'n' &&
               vendor[6] == 't' && vendor[7] == 'i' && vendor[8] == 'c' &&
               vendor[9] == 'A' && vendor[10] == 'M' && vendor[11] == 'D';
    }

    /**
     * @brief Get L1 data cache size in bytes (per core)
     * @return L1 data cache size in bytes, or 32KB if unknown
     *
     * Uses CPUID leaf 0x04 (Intel) or 0x8000001D (AMD) for detection.
     *
     * Typical values:
     *   - Intel Xeon: 32KB per core
     *   - AMD EPYC: 32KB per core (Zen 3/4), 48KB per core (Zen 5)
     */
    inline uint32_t cpu_l1_cache_size()
    {
        static const uint32_t cached_size = []
        {
            uint32_t size = 0;
            if (detail::detect_cache_by_level(1, /*data_only=*/true, size))
                return size;
            return static_cast<uint32_t>(32 * 1024); // Fallback: 32KB
        }();
        return cached_size;
    }

    /**
     * @brief Get L2 cache size in bytes (per core)
     * @return L2 cache size in bytes, or 256KB if unknown
     *
     * Uses CPUID leaf 0x04 (Intel) or 0x8000001D (AMD) for detection.
     *
     * Typical values:
     *   - Intel Xeon Gold 6238R: 1MB per core
     *   - AMD EPYC 9004 (Zen 4): 1MB per core
     *   - AMD EPYC 9005 (Zen 5): 1MB per core
     */
    inline uint32_t cpu_l2_cache_size()
    {
        static const uint32_t cached_size = []
        {
            uint32_t size = 0;
            if (detail::detect_cache_by_level(2, /*data_only=*/false, size))
                return size;
            return static_cast<uint32_t>(256 * 1024); // Fallback: 256KB
        }();
        return cached_size;
    }

    /**
     * @brief Get L3 cache size in bytes (shared across all cores)
     * @return L3 cache size in bytes, or 8MB if unknown
     *
     * Uses CPUID leaf 0x04 (Intel) or 0x8000001D (AMD) for detection.
     *
     * Typical values:
     *   - Intel Xeon Gold 6238R: ~39MB (shared)
     *   - AMD EPYC 9004 (Zen 4): 32-384MB depending on SKU
     *
     * Note: AMD Zen reports L3 per-CCD (e.g., 32MB per CCD), not total.
     * The value returned is the L3 slice visible to this core's topology.
     */
    inline uint32_t cpu_l3_cache_size()
    {
        static const uint32_t cached_size = []
        {
            uint32_t size = 0;
            if (detail::detect_cache_by_level(3, /*data_only=*/false, size))
                return size;
            return static_cast<uint32_t>(8 * 1024 * 1024); // Fallback: 8MB
        }();
        return cached_size;
    }

    /**
     * @brief Get total L2 cache per socket (L2 per core × physical cores)
     * @return Total L2 cache size in bytes
     *
     * Uses vendor-appropriate CPUID leaves for core count:
     *   - Intel: leaf 0x04 EAX[31:26]+1
     *   - AMD:   leaf 0x80000008 ECX[7:0]+1
     *   - Fallback: leaf 0x01 EBX[23:16] / 2
     *
     * For Xeon Gold 6238R with 28 cores: Returns 28MB
     * For AMD EPYC 9654 with 96 cores: Returns 96MB
     */
    inline uint32_t cpu_l2_cache_total()
    {
        static const uint32_t cached_size = []
        {
            uint32_t l2_per_core = cpu_l2_cache_size();
            uint32_t physical_cores = detail::detect_physical_cores_per_package();
            return l2_per_core * physical_cores;
        }();
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
    inline bool cpu_is_amd() { return false; }
    inline uint32_t cpu_l1_cache_size() { return 32 * 1024; }        // Conservative 32KB
    inline uint32_t cpu_l2_cache_size() { return 256 * 1024; }       // Conservative 256KB
    inline uint32_t cpu_l3_cache_size() { return 8 * 1024 * 1024; }  // Conservative 8MB
    inline uint32_t cpu_l2_cache_total() { return 8 * 1024 * 1024; } // Conservative 8MB
#endif

    /**
     * @brief Cache hierarchy information for tiling/batching decisions
     *
     * Provides detected cache sizes and computes optimal batch sizes
     * for operations like Wo projection that benefit from batching
     * to amortize weight matrix loads.
     *
     * Usage:
     *   CacheInfo cache;
     *   int wo_batch = cache.optimal_wo_batch_size(d_model);
     */
    struct CacheInfo
    {
        uint32_t l1_size;    ///< L1 data cache size in bytes (per core)
        uint32_t l2_size;    ///< L2 cache size in bytes (per core)
        uint32_t l3_size;    ///< L3 cache size in bytes (shared)
        uint32_t l2_total;   ///< Total L2 across all cores
        uint32_t cache_line; ///< Cache line size in bytes (typically 64)

        /**
         * @brief Construct CacheInfo with detected values
         */
        CacheInfo()
            : l1_size(cpu_l1_cache_size()), l2_size(cpu_l2_cache_size()), l3_size(cpu_l3_cache_size()), l2_total(cpu_l2_cache_total()), cache_line(64) // Standard x86 cache line
        {
        }

        /**
         * @brief Compute optimal batch size for Wo projection batching
         *
         * The goal is to accumulate multiple context vectors before doing
         * the Wo GEMM, so we load Wo weights once per batch instead of
         * once per query. This converts GEMV (m=1) to GEMM (m=batch).
         *
         * Constraints:
         *   1. Context batch buffer must fit in L2 to avoid evicting Wo
         *   2. Individual context vectors should fit in L1 during attention
         *   3. Batch size should be power of 2 for alignment (optional)
         *
         * Memory model:
         *   - Context buffer: batch_size × d_model × 4 bytes (FP32)
         *   - Wo matrix: d_model × d_model × elem_size (streamed from L3/DRAM)
         *   - We want context buffer to stay in L2 while Wo streams through
         *
         * @param d_model Model hidden dimension (e.g., 3584 for Qwen 7B)
         * @param target_cache_fraction Fraction of L2 to use (default 0.25)
         * @param min_batch Minimum batch size (default 2)
         * @param max_batch Maximum batch size (default 16)
         * @return Optimal batch size for Wo projection
         */
        int optimal_wo_batch_size(
            int d_model,
            float target_cache_fraction = 0.25f,
            int min_batch = 2,
            int max_batch = 16) const
        {
            // Context buffer per query: d_model × 4 bytes (FP32)
            size_t context_per_query = static_cast<size_t>(d_model) * 4;

            // Target budget: fraction of L2 for context buffer
            // We use 25% of L2 by default, leaving 75% for:
            //   - Wo weight tiles during GEMM
            //   - Output buffer writes
            //   - Other working data
            size_t budget = static_cast<size_t>(l2_size * target_cache_fraction);

            // How many queries fit in budget?
            int batch = static_cast<int>(budget / context_per_query);

            // Round down to power of 2 for cleaner loop bounds
            if (batch >= 16)
                batch = 16;
            else if (batch >= 8)
                batch = 8;
            else if (batch >= 4)
                batch = 4;
            else if (batch >= 2)
                batch = 2;

            // Clamp to [min_batch, max_batch]
            if (batch < min_batch)
                batch = min_batch;
            if (batch > max_batch)
                batch = max_batch;

            return batch;
        }

        /**
         * @brief Compute optimal KV tile size for attention
         *
         * Determines how many KV positions to process before updating
         * softmax state, balancing register pressure vs. memory traffic.
         *
         * @param head_dim Attention head dimension (e.g., 128)
         * @return Optimal KV tile size
         */
        int optimal_kv_tile_size(int head_dim) const
        {
            // K/V per position: head_dim × 36 bytes (Q8_1 blocks)
            size_t kv_per_pos = static_cast<size_t>(head_dim / 32) * 36 * 2;

            // Target: fit N KV positions in L1 with room for Q and scratch
            // Q takes ~144B per head, scratch ~512B
            size_t available = l1_size - 1024;

            int tile = static_cast<int>(available / kv_per_pos);

            // Clamp to reasonable range [4, 16]
            if (tile < 4)
                tile = 4;
            if (tile > 16)
                tile = 16;

            // Round down to power of 2 for simpler loop bounds
            if (tile >= 16)
                return 16;
            if (tile >= 8)
                return 8;
            return 4;
        }

        /**
         * @brief Check if a buffer fits entirely in L1
         */
        bool fits_l1(size_t bytes) const { return bytes <= l1_size; }

        /**
         * @brief Check if a buffer fits entirely in L2
         */
        bool fits_l2(size_t bytes) const { return bytes <= l2_size; }

        /**
         * @brief Check if a buffer fits entirely in L3
         */
        bool fits_l3(size_t bytes) const { return bytes <= l3_size; }

        /**
         * @brief Get human-readable cache info string
         */
        const char *summary() const
        {
            static char buf[256];
            snprintf(buf, sizeof(buf),
                     "L1=%uKB L2=%uKB L3=%uMB (L2_total=%uMB)",
                     l1_size / 1024,
                     l2_size / 1024,
                     l3_size / (1024 * 1024),
                     l2_total / (1024 * 1024));
            return buf;
        }

        /**
         * @brief Print detailed cache analysis for a model configuration
         * @param d_model Model hidden dimension
         * @param head_dim Attention head dimension
         * @param ostream Output stream (default: stdout)
         */
        void print_analysis(int d_model, int head_dim, FILE *out = stdout) const
        {
            size_t context_bytes = static_cast<size_t>(d_model) * 4;
            int wo_batch = optimal_wo_batch_size(d_model);
            int kv_tile = optimal_kv_tile_size(head_dim);

            size_t batch_buffer = context_bytes * wo_batch;
            float l2_usage_pct = 100.0f * batch_buffer / l2_size;

            fprintf(out, "╔════════════════════════════════════════════════════════════╗\n");
            fprintf(out, "║           CPU Cache Analysis for Wo Batching               ║\n");
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Cache Hierarchy:                                           ║\n");
            fprintf(out, "║   L1 Data:  %6u KB                                       ║\n", l1_size / 1024);
            fprintf(out, "║   L2:       %6u KB                                       ║\n", l2_size / 1024);
            fprintf(out, "║   L3:       %6u MB                                       ║\n", l3_size / (1024 * 1024));
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Model Config:                                              ║\n");
            fprintf(out, "║   d_model:  %6d                                          ║\n", d_model);
            fprintf(out, "║   head_dim: %6d                                          ║\n", head_dim);
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Optimal Tile Sizes:                                        ║\n");
            fprintf(out, "║   Wo batch:     %2d queries (%.1f%% L2 for context buffer)  ║\n", wo_batch, l2_usage_pct);
            fprintf(out, "║   KV tile:      %2d positions                              ║\n", kv_tile);
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Memory Footprints:                                         ║\n");
            fprintf(out, "║   Context/query: %6zu KB                                  ║\n", context_bytes / 1024);
            fprintf(out, "║   Batch buffer:  %6zu KB (%d queries)                     ║\n", batch_buffer / 1024, wo_batch);
            fprintf(out, "║   Wo matrix:     ~%4zu MB (streaming)                      ║\n", (size_t)d_model * d_model * 4 / (1024 * 1024));
            fprintf(out, "╚════════════════════════════════════════════════════════════╝\n");
        }
    };

    /**
     * @brief Get cached CacheInfo singleton
     * @return Reference to CacheInfo with detected cache sizes
     */
    inline const CacheInfo &cache_info()
    {
        static const CacheInfo info;
        return info;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ATTENTION KERNEL CACHE-AWARE CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // These utilities derive attention kernel heuristics from detected cache
    // sizes rather than hardcoded thresholds, improving portability across
    // different CPU microarchitectures (Zen 4, Golden Cove, Sapphire Rapids, etc.)
    //
    // Design Philosophy:
    //   - WorkSizeClass (SMALL/LARGE/XL) determined by KV cache footprint vs L2/L3
    //   - Prefetch distance scaled by cache line count at target level
    //   - FA2 tile width based on L1 capacity for score accumulation
    //
    // Memory Layout for Q8_1 attention (per KV position):
    //   K: (head_dim / 32) blocks × 36 bytes = head_dim * 36 / 32 bytes
    //   V: (head_dim / 32) blocks × 36 bytes = head_dim * 36 / 32 bytes
    //   Total per KV position per head: head_dim * 72 / 32 = head_dim * 2.25 bytes
    //
    // For 128-dim heads: 288 bytes per KV position per head
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Work size classification for attention kernels
     *
     * Maps cache behavior to compile-time kernel specialization:
     *   SMALL - KV fits comfortably in L2, optimize for low latency
     *   LARGE - KV spills to L3, optimize for bandwidth
     *   XL    - KV exceeds L3, streaming/prefetch critical
     */
    enum class AttentionWorkSize : uint8_t
    {
        SMALL = 0, ///< KV footprint < 50% L2 - keep hot in L2, short prefetch
        LARGE = 1, ///< KV footprint 50-200% L2 - use L3, medium prefetch
        XL = 2,    ///< KV footprint > 200% L2 - streaming, long prefetch to L3
    };

    /**
     * @brief Prefetch configuration derived from cache geometry
     */
    struct PrefetchConfig
    {
        int distance;    ///< Number of KV positions to prefetch ahead
        int cache_level; ///< Target cache level (0=L1/prefetcht0, 1=L2/prefetcht1, 2=L3/prefetcht2)

        bool operator==(const PrefetchConfig &o) const
        {
            return distance == o.distance && cache_level == o.cache_level;
        }
    };

    /**
     * @brief Cache-aware configuration for attention kernels
     *
     * Computes optimal parameters based on detected CPU cache hierarchy and
     * the actual working set size for a given attention configuration.
     *
     * Usage:
     *   AttentionCacheConfig cfg(head_dim, num_kv_heads, kv_seq_len);
     *   auto work_size = cfg.work_size();
     *   auto prefetch = cfg.prefetch_config();
     *   bool use_kv8 = cfg.prefer_kv8_tile();
     */
    struct AttentionCacheConfig
    {
        // Input parameters
        int head_dim;
        int num_kv_heads;
        int kv_seq_len;

        // Cached cache sizes (from singleton)
        uint32_t l1_size;
        uint32_t l2_size;
        uint32_t l3_size;

        /**
         * @brief Construct with attention dimensions
         * @param head_dim_ Dimension per head (typically 64-128)
         * @param num_kv_heads_ Number of KV heads (for GQA)
         * @param kv_seq_len_ Current KV cache sequence length
         */
        AttentionCacheConfig(int head_dim_, int num_kv_heads_, int kv_seq_len_)
            : head_dim(head_dim_), num_kv_heads(num_kv_heads_), kv_seq_len(kv_seq_len_)
        {
            const auto &ci = cache_info();
            l1_size = ci.l1_size;
            l2_size = ci.l2_size;
            l3_size = ci.l3_size;
        }

        /**
         * @brief Compute KV cache footprint per head in bytes
         *
         * Q8_1 format: each position stores (head_dim/32) blocks of 36 bytes each
         * for both K and V.
         */
        size_t kv_footprint_per_head() const
        {
            // K + V, each with ceil(head_dim/32) blocks of 36 bytes
            size_t blocks_per_pos = static_cast<size_t>((head_dim + 31) / 32);
            size_t bytes_per_pos = blocks_per_pos * 36 * 2; // K + V
            return bytes_per_pos * kv_seq_len;
        }

        /**
         * @brief Compute total KV cache footprint across all heads
         */
        size_t kv_footprint_total() const
        {
            return kv_footprint_per_head() * num_kv_heads;
        }

        /**
         * @brief Determine work size class based on KV footprint vs cache hierarchy
         *
         * Thresholds derived from cache sizes:
         *   SMALL: Total KV fits in 50% of L2 → all heads' KV stays L2-resident
         *   LARGE: Total KV fits in L3 but not L2 → spills to L3
         *   XL:    Total KV exceeds L3 → streaming from DRAM
         *
         * We use per-head footprint for SMALL threshold since GQA processes
         * one KV head at a time across multiple Q heads.
         */
        AttentionWorkSize work_size() const
        {
            size_t per_head = kv_footprint_per_head();

            // SMALL: per-head KV fits in L2 with room for Q, scratch, output
            // Target: 50% of L2 to leave room for other working data
            size_t l2_target = l2_size / 2;
            if (per_head <= l2_target)
            {
                return AttentionWorkSize::SMALL;
            }

            // LARGE: per-head KV fits in L3 slice
            // Assume we get ~1/8 of L3 per thread on typical multi-core
            size_t l3_per_thread = l3_size / 8;
            if (per_head <= l3_per_thread)
            {
                return AttentionWorkSize::LARGE;
            }

            // XL: streaming from DRAM
            return AttentionWorkSize::XL;
        }

        /**
         * @brief Compute optimal prefetch configuration
         *
         * Prefetch distance is scaled by:
         *   - Target cache's line capacity (larger cache → longer distance)
         *   - Memory latency hiding (L1=~4cy, L2=~12cy, L3=~40cy, DRAM=~100cy)
         *   - KV position stride (larger head_dim → smaller distance in positions)
         *
         * Returns prefetch distance in KV positions and target cache level.
         */
        PrefetchConfig prefetch_config() const
        {
            AttentionWorkSize ws = work_size();
            size_t bytes_per_kv_pos = static_cast<size_t>((head_dim + 31) / 32) * 36;

            PrefetchConfig cfg;

            switch (ws)
            {
            case AttentionWorkSize::SMALL:
                // L1 prefetch: short distance, keep data close
                // Target: prefetch 2-4 cache lines worth
                cfg.cache_level = 0; // prefetcht0
                cfg.distance = std::max(2, std::min(8,
                                                    static_cast<int>((4 * 64) / bytes_per_kv_pos)));
                break;

            case AttentionWorkSize::LARGE:
                // L2 prefetch: medium distance for L2→L1 promotion
                // Target: prefetch enough to hide L2 latency (~12 cycles)
                cfg.cache_level = 1; // prefetcht1
                cfg.distance = std::max(8, std::min(32,
                                                    static_cast<int>((16 * 64) / bytes_per_kv_pos)));
                break;

            case AttentionWorkSize::XL:
                // L3/DRAM prefetch: long distance for memory latency hiding
                // Target: prefetch enough to hide DRAM latency (~100+ cycles)
                cfg.cache_level = 2; // prefetcht2
                cfg.distance = std::max(32, std::min(128,
                                                     static_cast<int>((64 * 64) / bytes_per_kv_pos)));
                break;
            }

            return cfg;
        }

        /**
         * @brief Determine if KV8 tile width is preferred over KV4
         *
         * KV8 processes 8 positions at once, doubling register pressure but
         * halving loop overhead. Beneficial when:
         *   - KV fits in L1/L2 (low latency memory)
         *   - Loop overhead dominates (shorter sequences)
         *
         * KV4 preferred when:
         *   - Memory bandwidth limited (streaming from L3/DRAM)
         *   - Very long sequences where extra register pressure hurts
         *
         * Derived from cache: use KV8 if per-head KV fits in L2.
         */
        bool prefer_kv8_tile() const
        {
            // KV8 doubles the working set per iteration
            // Only use if per-head KV fits comfortably in L2
            size_t per_head = kv_footprint_per_head();
            return per_head <= l2_size;
        }

        /**
         * @brief Get human-readable configuration summary
         */
        void print_config(FILE *out = stdout) const
        {
            auto ws = work_size();
            auto pf = prefetch_config();

            const char *ws_names[] = {"SMALL", "LARGE", "XL"};
            const char *pf_names[] = {"L1 (prefetcht0)", "L2 (prefetcht1)", "L3 (prefetcht2)"};

            fprintf(out, "╔════════════════════════════════════════════════════════════╗\n");
            fprintf(out, "║      Cache-Aware Attention Configuration                   ║\n");
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Input Parameters:                                          ║\n");
            fprintf(out, "║   head_dim:     %4d                                        ║\n", head_dim);
            fprintf(out, "║   num_kv_heads: %4d                                        ║\n", num_kv_heads);
            fprintf(out, "║   kv_seq_len:   %4d                                        ║\n", kv_seq_len);
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Memory Footprint:                                          ║\n");
            fprintf(out, "║   Per head:  %8zu KB                                     ║\n", kv_footprint_per_head() / 1024);
            fprintf(out, "║   Total:     %8zu KB                                     ║\n", kv_footprint_total() / 1024);
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Cache Hierarchy:                                           ║\n");
            fprintf(out, "║   L1: %4u KB  L2: %4u KB  L3: %4u MB                     ║\n",
                    l1_size / 1024, l2_size / 1024, l3_size / (1024 * 1024));
            fprintf(out, "╠════════════════════════════════════════════════════════════╣\n");
            fprintf(out, "║ Derived Configuration:                                     ║\n");
            fprintf(out, "║   WorkSize:   %-6s (per-head vs L2/L3)                   ║\n", ws_names[static_cast<int>(ws)]);
            fprintf(out, "║   Prefetch:   %3d positions → %-20s  ║\n", pf.distance, pf_names[pf.cache_level]);
            fprintf(out, "║   FA2 Tile:   KV%d (KV8 %s)                             ║\n",
                    prefer_kv8_tile() ? 8 : 4,
                    prefer_kv8_tile() ? "fits L2" : "spills L2");
            fprintf(out, "╚════════════════════════════════════════════════════════════╝\n");
        }
    };

} // namespace llaminar2
