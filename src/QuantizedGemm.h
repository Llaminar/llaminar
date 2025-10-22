/**
 * @file QuantizedGemm.h
 * @brief Legacy compatibility header - use ITensorGemm.h instead.
 *
 * This file provides backward compatibility by aliasing IQuantizedGemm to ITensorGemm.
 * New code should include ITensorGemm.h directly.
 *
 * DEPRECATED: This interface has been renamed to ITensorGemm to reflect its broader
 * applicability to all tensor types (FP32, BF16, quantized), not just quantized tensors.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "ITensorGemm.h"

namespace llaminar
{
    // Backward compatibility alias
    using IQuantizedGemm = ITensorGemm;

} // namespace llaminar
