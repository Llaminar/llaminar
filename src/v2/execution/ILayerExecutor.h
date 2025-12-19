/**
 * @file ILayerExecutor.h
 * @brief Backwards compatibility header - includes IGraphExecutor.h
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated Use IGraphExecutor.h directly. This header exists for
 * backwards compatibility and will be removed in a future version.
 *
 * ILayerExecutor has been renamed to IGraphExecutor to better reflect
 * its purpose - it executes compute graphs, not specifically layers.
 */

#pragma once

#include "IGraphExecutor.h"

// All types are aliased in IGraphExecutor.h:
// - ILayerExecutor = IGraphExecutor
// - LayerExecutorConfig = GraphExecutorConfig
// - LayerExecutorStats = GraphExecutorStats
