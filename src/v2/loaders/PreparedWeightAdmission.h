/**
 * @file PreparedWeightAdmission.h
 * @brief Typed admission policy for model-owned prepared weight residency.
 *
 * Weight allocation, graph construction, and memory preflight must agree on
 * whether a runner creates a new prepared weight set or adopts a complete set
 * certified by the model-context reuse lifecycle.  Keeping that decision in
 * one enum prevents graph factories from inferring reuse from cache contents
 * or telemetry and prevents memory planning from charging retained weights a
 * second time.
 */

#pragma once

namespace llaminar2
{
    /**
     * @brief Whether graph setup allocates or adopts its complete weight set.
     *
     * `ReuseCertifiedCompleteSet` may only be selected by an upstream typed
     * reuse contract after exact plan, ownership, and lifecycle validation.
     * Consumers must fail on a missing prepared value; they may not fall back
     * to source loading because that would conceal a broken reuse certificate.
     */
    enum class PreparedWeightAdmission
    {
        /** The caller must materialize and prepare the complete planned set. */
        AllocateCompleteSet,
        /** The exact complete model-owned prepared set is already resident. */
        ReuseCertifiedCompleteSet,
    };
}
