/**
 * @file SIMDKQuantHelpers.h
 * @brief Compatibility shim exposing SIMD quantization/decode helpers.
 *
 * Historical test files included "tensors/SIMDKQuantHelpers.h" which was
 * consolidated into SIMDHelpers.h during refactoring. This header restores
 * the include path and forwards all functionality.
 *
 * No new symbols are introduced here; tests relying on decode/quantize
 * routines (e.g. decode_q4_0_to_q8_0_avx2) already obtain them from
 * SIMDHelpers.h. Keeping this thin prevents divergence.
 */
#pragma once
#include "SIMDHelpers.h"

// Optional future extension point: legacy aliases could be added here if
// older code expected different names. Currently unnecessary.
