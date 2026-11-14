/**
 * @file ROCmRouterQ8Prefill.h
 * @brief Native gfx1100 router tiles retaining the ordered serial Q8 contract.
 *
 * Selection consumes immutable hardware metadata. The diagnostic scope changes
 * graph recording only: replay embeds the selected native function, geometry
 * and borrowed banks. It never reads a host policy while executing on device.
 */
#pragma once
#include "kernels/common/DeviceRowPartition.h"
#include "ROCmRouterQ8Policy.h"
#include <cstdint>

namespace llaminar2::rocm
{
/** @brief Thread-local, nestable recording policy; does not alter retained graphs. */
class ScopedRouterQ8PrefillMode final
{
public:
    /** @brief Install a typed recording choice, restoring its predecessor on exit.
     * @param mode Candidate to record on this device worker thread. */
    explicit ScopedRouterQ8PrefillMode(RouterQ8PrefillMode mode);
    /** @brief Restore the enclosing recording policy. */
    ~ScopedRouterQ8PrefillMode();
    ScopedRouterQ8PrefillMode(const ScopedRouterQ8PrefillMode &) = delete;
    ScopedRouterQ8PrefillMode &operator=(const ScopedRouterQ8PrefillMode &) = delete;
private:
    RouterQ8PrefillMode previous_;
};

/** @brief Resolve the native entry from the admitted device and captured shape.
 * @param device Owning visible HIP ordinal. @param shape Original geometry and ownership.
 * @return Immutable engine and tile selection bound to this device and shape.
 * @throws std::invalid_argument For a forced unsupported architecture or geometry. */
RouterQ8PrefillSelection selectRouterQ8Prefill(int device, RouterQ8PrefillShape shape);

/** @brief Borrowed immutable bindings; device live rows remain the sole row authority. */
struct RouterQ8PrefillView
{
    const std::int8_t *hidden_codes = nullptr;
    const float *hidden_scales = nullptr;
    const std::int8_t *gate_codes = nullptr;
    const float *gate_scales = nullptr;
    float *logits = nullptr;
    int capacity = 0;
    int width = 0;
    int experts = 0;
    int device = -1;
    const int *live_rows = nullptr;
    DeviceRowPartition partition;
    RouterQ8Basis basis = RouterQ8Basis::Original;
};

/** @brief Enqueue one exact WMMA projection without allocation or synchronization.
 * @param view Frozen operands, ownership and device identity.
 * @param selection Resolved native engine; must match the submitted view.
 * @param stream Exact non-null stream owned by view.device.
 * @return False for invalid bindings or a native submission error. */
bool routerQ8Prefill(const RouterQ8PrefillView &view, const RouterQ8PrefillSelection &selection, void *stream);
}
