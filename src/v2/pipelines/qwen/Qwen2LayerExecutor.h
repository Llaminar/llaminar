/**
 * @file Qwen2LayerExecutor.h
 * @brief Backwards compatibility header - includes Qwen2Graph.h
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated This file exists for backwards compatibility only.
 * All functionality has been merged into Qwen2Graph.h.
 * The Qwen2LayerExecutor alias is provided for existing code.
 *
 * Migration Guide:
 * - Replace `#include "Qwen2LayerExecutor.h"` with `#include "Qwen2Graph.h"`
 * - Replace `Qwen2LayerExecutor` with `Qwen2Graph`
 * - Replace `Qwen2ExecutorConfig` with `Qwen2GraphConfig`
 */

#pragma once

// Include the new unified header
#include "Qwen2Graph.h"

// The following aliases are defined in Qwen2Graph.h for backward compatibility:
// - using Qwen2LayerExecutor = Qwen2Graph;
// - using Qwen2ExecutorConfig = Qwen2GraphConfig;
// - struct Qwen2LayerWeights (kept in Qwen2Graph.h)
// - struct Qwen2ActivationBuffers (kept in Qwen2Graph.h)
