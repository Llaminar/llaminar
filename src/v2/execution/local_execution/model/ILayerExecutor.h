/**
 * @file ILayerExecutor.h
 * @brief Backwards compatibility header - includes IDeviceGraphExecutor.h
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated Use IDeviceGraphExecutor.h directly. This header exists for
 * backwards compatibility and will be removed in a future version.
 *
 * ILayerExecutor has been renamed to IGraphExecutor to better reflect
 * its purpose - it executes compute graphs, not specifically layers.
 */

#pragma once

#include "../graph/IGraphExecutor.h"

// All types are aliased in IDeviceGraphExecutor.h:
// - ILayerExecutor = IGraphExecutor
// - LayerExecutorConfig = GraphExecutorConfig
// - LayerExecutorStats = GraphExecutorStats
