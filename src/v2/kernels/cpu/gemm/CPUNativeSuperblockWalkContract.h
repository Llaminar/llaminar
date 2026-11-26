/**
 * @file CPUNativeSuperblockWalkContract.h
 * @brief Host admission for bounded, strictly progressing superblock walks.
 */
#pragma once
#include <climits>
#include <cstddef>
#include "utils/Assertions.h"
namespace llaminar2::cpu::native_quantised {
/** @brief Reject ranges whose clipped superblock end cannot advance an int cursor.
 * @param first Inclusive partition start.
 * @param last Exclusive partition end.
 * @param blocks Physical row extent.
 * @param phase Wire superblock phase.
 */
inline void requireSuperblockWalk(int first, int last, std::size_t blocks, std::size_t phase = 0) noexcept {
    if (!(blocks <= INT_MAX && phase < 8 && first >= 0 && first <= last &&
          static_cast<std::size_t>(last) <= blocks))
        LLAMINAR_UNREACHABLE("Invalid CPU superblock walk range");
}
}
