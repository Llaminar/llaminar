/**
 * @file CPUExecutionTestGeometry.h
 * @brief Explicit device-free CPU observation used by metadata fixtures.
 *
 * A synthetic fixture must not borrow the test host's CPUID. Real-kernel tests
 * use CPUExecutionGeometry::local() instead. Tests probing missing observations
 * deliberately leave the production geometry's zero defaults untouched.
 */
#pragma once
#include "backends/CPUExecutionGeometry.h"
namespace llaminar2::test
{
    inline constexpr CPUExecutionGeometry kSyntheticCPUExecutionGeometry{
        .cache = {1u << 20, 32u << 20, 16, 16}, .maximum_native_row_tile = 4};
}
