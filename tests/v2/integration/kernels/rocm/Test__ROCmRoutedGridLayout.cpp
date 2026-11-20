/**
 * @file Test__ROCmRoutedGridLayout.cpp
 * @brief Verify the production lookup layout borrows only retired HC storage.
 */
#include "kernels/rocm/gemm/ROCmRoutedGridDonation.h"
#include "tensors/IQQuantTables.h"
#include "tensors/NativeQuantisedFormatInfo.h"

#include <iostream>
#include <iterator>

int main() {
    using namespace llaminar2;
    using namespace llaminar2::rocm;
    using namespace native_quantised_formats;
    constexpr unsigned prefix_bytes = 32768;
    constexpr unsigned hidden_bytes = 3200;
    constexpr unsigned role_bytes = 96;
    for (unsigned mask = 0; mask < 8; ++mask) {
        const auto iq2 = mask & 1 ? std::size(iq2s_grid) : 0;
        const auto small = mask & 2 ? std::size(iq3s_grid) : 0;
        const auto extra = mask & 4 ? std::size(iq3xxs_grid) : 0;
        const auto iq2_bytes = iq2 * sizeof(iq2s_grid[0]);
        const auto iq3_bytes = small * sizeof(iq3s_grid[0]) + extra * sizeof(iq3xxs_grid[0]);
        // Vary the donated prefix while the existing hidden, panel and record
        // extents remain live. Test every byte across each exact fit threshold.
        for (std::size_t prefix = 0; prefix <= prefix_bytes; ++prefix) {
            constexpr auto reserved = hidden_bytes + 2048 + role_bytes;
            const auto formats = (iq2 ? 1u << IQ2_S.codebook_id : 0) |
                                 (small ? 1u << IQ3_S.codebook_id : 0) |
                                 (extra ? 1u << IQ3_XXS.codebook_id : 0) |
                                 (1u << IQ2_XS.codebook_id);
            const auto layout = inspectRoutedGridDonation(formats, prefix, hidden_bytes, 2048 + role_bytes);
            const auto fit_iq3 = prefix >= reserved + iq3_bytes;
            const auto fit_iq2 = prefix >= reserved + iq2_bytes + (fit_iq3 ? iq3_bytes : 0);
            const auto iq3_offset = layout.iq2_entries * sizeof(iq2s_grid[0]);
            const auto extra_offset = iq3_offset + layout.iq3_small_entries * sizeof(iq3s_grid[0]);
            const auto end = extra_offset + layout.iq3_extra_small_entries * sizeof(iq3xxs_grid[0]);
            if (layout.iq2_entries != (fit_iq2 ? iq2 : 0) ||
                layout.iq3_small_entries != (fit_iq3 ? small : 0) ||
                layout.iq3_extra_small_entries != (fit_iq3 ? extra : 0) ||
                iq3_offset % alignof(std::uint64_t) || extra_offset % alignof(std::uint32_t) ||
                (end && reserved + end > prefix)) {
                std::cerr << "Incorrect routed grid layout for mask " << mask
                          << ", admitted prefix bytes " << prefix << '\n';
                return 1;
            }
        }
    }
}
