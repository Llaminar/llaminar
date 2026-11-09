/**
 * @file RoPEAngleTestStorage.h
 * @brief Test-owned angle planes for direct calls into the CPU RoPE primitives.
 *
 * Production RoPE borrows its planes from the admitted `cpu_rope_angles`
 * workspace that RoPEStage binds. Device-free primitive tests have no stage,
 * so this fixture owns one plane-aligned byte buffer of the same layout and
 * carves views from it exactly as the kernel does. Construct it outside the
 * parallel region under test: every team member must share one buffer.
 */
#pragma once

#include "kernels/cpu/primitives/RoPEAngleRows.h"

#include <cstddef>
#include <span>
#include <vector>

namespace llaminar2::test
{
    /** @brief One plane-aligned angle buffer sized for a maximum geometry. */
    class RoPEAngleTestStorage final
    {
    public:
        /**
         * @brief Own storage for up to `rows` rows of `values_per_row` pairs.
         * @param rows Maximum rows any carved view will hold.
         * @param values_per_row Maximum rotary pairs any carved view will hold.
         */
        RoPEAngleTestStorage(int rows, int values_per_row)
            : bytes_(primitives::rope_angle_layout(rows, values_per_row).bytes +
                     primitives::kRoPEAnglePlaneAlignment)
        {
        }

        /** @return A view for one invocation, carved like the kernel's borrow. */
        primitives::RoPEAngleRows view(int rows, int values_per_row)
        {
            return primitives::carve_rope_angle_rows(aligned(), rows, values_per_row);
        }

        /** @return The complete plane-aligned buffer. */
        std::span<std::byte> aligned()
        {
            auto *base = bytes_.data();
            const auto misalignment = reinterpret_cast<std::uintptr_t>(base) % primitives::kRoPEAnglePlaneAlignment;
            const std::size_t skip = misalignment ? primitives::kRoPEAnglePlaneAlignment - misalignment : 0;
            return {base + skip, bytes_.size() - skip};
        }

    private:
        std::vector<std::byte> bytes_; ///< Over-allocated by one alignment step.
    };
}
