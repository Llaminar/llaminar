/**
 * @file HttpServiceThreadAffinity.h
 * @brief HTTP spelling of the shared independent-control placement authority.
 *
 * Listener/HTTP children and dedicated GPU-control children reuse one scoped
 * OpenMP-partition implementation. No second affinity algorithm or CPU mask
 * authority is retained here. The scope restores the inference caller exactly.
 */
#pragma once
#include "utils/ScopedOpenMPPartitionAffinity.h"

namespace llaminar2
{
    /** @brief HTTP listener uses the shared independent-control construction scope. */
    using HttpServiceThreadAffinity = ScopedOpenMPPartitionAffinity;
}
