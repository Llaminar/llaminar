/**
 * @file Qwen2ModelExecutor.h
 * @brief Backwards compatibility header - includes Qwen2Graph.h
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated This file exists for backwards compatibility only.
 * All functionality has been merged into Qwen2Graph.h.
 * The Qwen2ModelExecutor alias is provided for existing code.
 *
 * Migration Guide:
 * - Replace `#include "Qwen2ModelExecutor.h"` with `#include "Qwen2Graph.h"`
 * - Replace `Qwen2ModelExecutor` with `Qwen2Graph`
 * - Replace `Qwen2ModelExecutorConfig` with `Qwen2GraphConfig`
 */

#pragma once

// Include the new unified header
#include "Qwen2Graph.h"

// The following aliases are defined in Qwen2Graph.h for backward compatibility:
// - using Qwen2ModelExecutor = Qwen2Graph;
// - using Qwen2ModelExecutorConfig = Qwen2GraphConfig;
// - struct Qwen2ModelWeights (kept in Qwen2Graph.h)
// - struct Qwen2ModelBuffers (kept in Qwen2Graph.h)
