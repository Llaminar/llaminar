/**
 * @file ScopedGDNProjectionArithmetic.h
 * @brief Isolate a declared GDN arithmetic policy in device-free fixtures.
 *
 * Native codebook tests select their policy explicitly so a changed production
 * default cannot turn them into different arithmetic tests. Scope exit restores
 * the previous policy even when an assertion or source admission throws.
 */
#pragma once
#include "utils/DebugEnv.h"

namespace llaminar2::test
{
/** @brief Noncopyable policy owner for one serial test scope. */
class ScopedGDNProjectionArithmetic final
{
public:
    /** @brief Save the current policy and install the fixture's explicit mode. */
    explicit ScopedGDNProjectionArithmetic(GDNProjectionArithmetic selected)
        : previous_(debugEnv().gemm.gdn_projection_arithmetic)
    {
        mutableDebugEnv().gemm.gdn_projection_arithmetic = selected;
    }
    /** @brief Restore the exact prior policy at scope exit. */
    ~ScopedGDNProjectionArithmetic()
    {
        mutableDebugEnv().gemm.gdn_projection_arithmetic = previous_;
    }
    ScopedGDNProjectionArithmetic(const ScopedGDNProjectionArithmetic &) = delete;
    ScopedGDNProjectionArithmetic &operator=(const ScopedGDNProjectionArithmetic &) = delete;
private:
    GDNProjectionArithmetic previous_;
};
}
