/**
 * @file ROCmDensePrefillPolicyEvidence.h
 * @brief Cold read-only evidence of the installed production dense-prefill table.
 *
 * Tuning must authenticate the actual linked selector's table, including rows
 * inherited from earlier generations. This view borrows the same compiled
 * authority without copying, selecting, allocating or querying a GPU. Keeping
 * this cold interface separate avoids rebuilding device shards for tooling.
 */
#pragma once
#include "ROCmDenseProductionPrefillOverlayTypes.h"
#include <span>

namespace llaminar2::rocm
{
/**
 * @brief Borrow the exact immutable table used by the linked production selector.
 * @return Process-lifetime read-only entries, including legacy and other-ISA rows.
 * Cold tuning admission authenticates its frozen base through this authority.
 * The view cannot change table entries or the worker's selection policy.
 */
[[nodiscard]] std::span<const generated::ROCmDensePrefillOverlayEntry> densePrefillInstalledEntries() noexcept;
}
