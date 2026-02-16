/**
 * @file LayerExecutor.h
 * @brief Backwards compatibility header - includes DeviceGraphExecutor.h
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated Use DeviceGraphExecutor.h directly. This header exists for
 * backwards compatibility and will be removed in a future version.
 *
 * LayerExecutor has been renamed to DeviceGraphExecutor to better reflect
 * its purpose - it executes compute graphs, not specifically layers.
 */

#pragma once

#include "../graph/DeviceGraphExecutor.h"

// All types are aliased in DeviceGraphExecutor.h:
// - LayerExecutor = DeviceGraphExecutor
// - ComputeGraph and ComputeNode are defined in DeviceGraphExecutor.h
