/**
 * @file CPUNativeQuantisedExecutionExactGenerated.inc
 * @brief Emitter-owned additive exact observations, preserving installed generics.
 *
 * The initial empty artifact admits no measured entries. The canonical exact
 * collector replaces this file only with authenticated matching observations.
 */
#pragma once

namespace llaminar2::cpu::native_quantised::generated
{
    /**
     * @brief Look up one complete measured execution identity.
     * @param key Complete runtime ABI v1 scalar identity.
     * @param choice Written only for an exact measured hit.
     * @return False retains the existing complete selector unchanged.
     */
    inline bool selectCPUExecutionExactPolicy(
        const CPUExecutionExactKey &key, CPUExecutionExactChoice &choice) noexcept
    {
        (void)key;
        (void)choice;
        return false;
    }
}
