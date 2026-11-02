/**
 * @file RowSelectPolicy.h
 * @brief Immutable row-copy semantics shared by CUDA/HIP and graph stages.
 *
 * Copy only owned live columns while respecting each tensor's physical stride.
 * An absent device-owned accepted row preserves the previously committed value;
 * ordinary row selection retains its original clamp-to-first semantics.
 */
#pragma once

namespace llaminar2
{
    /** @brief Meaning of a negative device-owned row index. */
    enum class NegativeRowSelectionPolicy { ClampToFirst, PreserveDestination };

    /** @brief Capture-time physical strides and fixed/external row ownership. */
    struct RowsSelectPolicy
    {
        NegativeRowSelectionPolicy negative_rows = NegativeRowSelectionPolicy::ClampToFirst;
        int source_stride = 0; ///< Zero retains the tightly packed live-column stride.
        int destination_stride = 0; ///< Zero retains the tightly packed live-column stride.
        int fixed_first_row = -1; ///< Nonnegative only for an immutable fixed row range.
    };

    /**
     * @brief Resolve the original row-selection rule without accessing storage.
     * @param row Original selected row from fixed or device-owned metadata.
     * @param source_rows Validated positive source row capacity.
     * @param policy Immutable negative-row semantics.
     * @return Selected source row, or -1 when output must remain unchanged.
     */
#if defined(__CUDACC__) || defined(__HIPCC__)
    __host__ __device__
#endif
    constexpr int selectedRowForCopy(int row, int source_rows, NegativeRowSelectionPolicy policy) noexcept
    {
        if (row < 0)
            return policy == NegativeRowSelectionPolicy::PreserveDestination ? -1 : 0;
        return row >= source_rows ? source_rows - 1 : row;
    }
}
