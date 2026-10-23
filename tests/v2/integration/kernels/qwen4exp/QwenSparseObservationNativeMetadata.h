/**
 * @file QwenSparseObservationNativeMetadata.h
 * @brief Native graph identity and static resources for fixture-only observation events.
 *
 * Adapters inspect already recorded nodes outside event intervals. No profiler
 * attachment or inference state readback is needed for these native attributes.
 */
#pragma once
#include <cstddef>
#include <string>
namespace llaminar2::qwen4exp::QSA_SUITE::observation115
{
/** @brief Exact captured function identity and its admitted 256-thread resources. */
struct NativeKernelMetadata final
{
    void* identity=nullptr; ///< Original native function-or-kernel handle, never a guessed stub.
    std::string name; ///< Original native symbol resolved from that captured handle.
    int registers=0; ///< Native function register attribute; interpretation remains backend-specific.
    std::size_t shared_bytes=0; ///< Static shared-memory bytes reported for this function.
    std::size_t local_bytes=0; ///< Per-thread local/private storage, required to be zero.
    int max_threads=0; ///< Native maximum admitted threads per CTA.
    int resident_block_limit=0; ///< Resource-based resident-block limit for 256 threads; theoretical, never achieved occupancy.
};
}
