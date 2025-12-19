/**
 * @file LayerExecutor.h
 * @brief Backwards compatibility header - includes GraphExecutor.h
 * @author David Sanftenberg
 * @date December 2025
 *
 * @deprecated Use GraphExecutor.h directly. This header exists for
 * backwards compatibility and will be removed in a future version.
 *
 * LayerExecutor has been renamed to GraphExecutor to better reflect
 * its purpose - it executes compute graphs, not specifically layers.
 */

#pragma once

#include "GraphExecutor.h"

// All types are aliased in GraphExecutor.h:
// - LayerExecutor = GraphExecutor
// - ComputeGraph and ComputeNode are defined in GraphExecutor.h
