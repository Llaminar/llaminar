/**
 * @file CPUNativeVNNIDecodePolicyGenerated.inc
 * @brief Development placeholder for the sealed CPU NativeVNNI M=1 selector.
 *
 * The production trainer replaces this file atomically after development
 * cross-validation and untouched certification both pass.  Keeping the policy
 * ABI available in a development build lets explicit perf candidates compile,
 * but the missing certification marker makes ordinary `Auto` dispatch fail
 * closed.  This file intentionally contains no heuristic or serial fallback.
 */
#pragma once

#include <cstdint>

#define LLAMINAR_CPU_NVNNI_DECODE_POLICY_ABI 1
#define LLAMINAR_CPU_NVNNI_DECODE_POLICY_CERTIFIED 0

namespace llaminar2::cpu::native_vnni::generated
{
    /** Physical OpenMP N-chunk ownership selected for one M=1 launch. */
    enum class CPUNativeVNNIDecodePolicy : uint8_t
    {
        Nbc1 = 0,
        Nbc2 = 1,
        Nbc4 = 2,
        Nbc8 = 3,
        Nbc16 = 4,
    };

    /** Maximum vector ISA compiled into the policy-owning binary. */
    enum class CPUNativeVNNIDecodeBuildISA : uint8_t
    {
        AVX2 = 0,
        AVX512 = 1,
    };

    /** ISA selected by runtime dispatch for the current launch. */
    enum class CPUNativeVNNIDecodeRuntimeISA : uint8_t
    {
        AVX2 = 0,
        AVX512 = 1,
    };

    /**
     * @brief Resolve one sealed CPU M=1 schedule.
     *
     * The development placeholder owns no policy rows and therefore always
     * returns false.  A generated production include must provide total generic
     * coverage plus exact overlays for the same signature.
     */
    inline bool selectCPUNativeVNNIDecodeGeneratedPolicy(
        CPUNativeVNNIDecodeBuildISA,
        CPUNativeVNNIDecodeRuntimeISA,
        int,
        uint8_t,
        int,
        int,
        bool,
        CPUNativeVNNIDecodePolicy &)
    {
        return false;
    }
} // namespace llaminar2::cpu::native_vnni::generated
