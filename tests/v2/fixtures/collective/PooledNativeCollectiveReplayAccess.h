/**
 * @file PooledNativeCollectiveReplayAccess.h
 * @brief Closed construction-only native identity input for device-free control replay.
 *
 * Recorded pointers do not constitute native readiness, received data, terminal
 * completion or an executable communicator. Tests must not use this fixture to
 * authorize production graph materialization or claim native transfer coverage.
 */
#pragma once
#include "collective/NativeCollectiveCompletion.h"

namespace llaminar2::test {
/** @brief Supply immutable recorded identity to actual construction/control laws only. */
class PooledNativeCollectiveReplayAccess final {
public:
    /** @return Recorded immutable pointer pair, without reading it or manufacturing a receipt. */
    static NativeCompletionAuthority recorded(ncclLlaminarCompletionAuthority original)
    {
        return NativeCompletionAuthority(original);
    }
};
} // namespace llaminar2::test
