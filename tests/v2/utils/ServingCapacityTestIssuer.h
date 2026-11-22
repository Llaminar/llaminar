/**
 * @file ServingCapacityTestIssuer.h
 * @brief Isolated device-free test authority for ResolvedServingCapacity.
 *
 * Production admission MUST use resolveServingCapacity (authenticated). This
 * issuer is the ONLY friend with mint access besides the executor's own
 * device-free binding, so fixtures can keep valid test geometry (e.g. R=64)
 * without a public resolved-capacity mint.
 */
#pragma once

#include "execution/config/ServingCapacity.h"

namespace llaminar2
{
    struct ServingCapacityTestIssuer
    {
        static ResolvedServingCapacity fromComponents(int resident_slots,
                                                      int response_leases,
                                                      int queue_capacity)
        {
            return ResolvedServingCapacity::fromComponentsChecked(
                resident_slots, response_leases, queue_capacity);
        }
    };
} // namespace llaminar2
