/**
 * @file CollectiveMemoryEstimator.h
 * @brief Exact persistent-memory BOM for rank-local tensor-parallel transport.
 *
 * LocalTP retains two independent allocations on every participant: the
 * backend's largest transport buffer and an FP16 scratch buffer used by
 * transport paths whose arithmetic representation differs from the graph's
 * activation precision. Capacity admission and RankOrchestrator consume this
 * same pure bill so neither can silently omit or reconstruct those owners.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Persistent allocations owned by one LocalTP participant. */
    struct LocalTPCollectiveMemoryBOM
    {
        /** Largest backend transport payload, including the fixed margin. */
        std::size_t backend_temp_bytes = 0;
        /** Logical FP16 scratch elements, including the fixed margin. */
        std::size_t fp16_scratch_elements = 0;
        /** Physical FP16 scratch bytes allocated on each participant. */
        std::size_t fp16_scratch_bytes = 0;

        /** @return Total persistent bytes retained on one participant. */
        [[nodiscard]] std::size_t perDeviceBytes() const noexcept
        {
            return backend_temp_bytes + fp16_scratch_bytes;
        }
    };

    /** @brief Canonical sizing authority for LocalTP collective resources. */
    class CollectiveMemoryEstimator final
    {
    public:
        /**
         * @brief Price the exact reservation later installed by LocalTP.
         *
         * The backend buffer covers FP32 because graph stages may explicitly
         * request FP32 reduction even when resident activations use a narrower
         * format. FP16 scratch is a separate allocation and therefore remains
         * additive. Both use the production ten-percent geometry margin.
         *
         * @param max_seq_len Maximum context rows admitted by the runner.
         * @param hidden_size Model hidden width.
         * @return Per-participant persistent collective BOM.
         * @throws std::invalid_argument for non-positive geometry.
         * @throws std::overflow_error when byte arithmetic exceeds size_t.
         */
        [[nodiscard]] static LocalTPCollectiveMemoryBOM localTP(
            int max_seq_len,
            int hidden_size)
        {
            if (max_seq_len <= 0 || hidden_size <= 0)
            {
                throw std::invalid_argument(
                    "LocalTP collective memory requires positive sequence and hidden geometry");
            }

            const auto checkedMultiply = [](
                std::size_t left,
                std::size_t right,
                const char *what)
            {
                if (left != 0 &&
                    right > std::numeric_limits<std::size_t>::max() / left)
                {
                    throw std::overflow_error(what);
                }
                return left * right;
            };
            const auto withMargin = [&](std::size_t value, const char *what)
            {
                return checkedMultiply(value, 11u, what) / 10u;
            };

            const std::size_t elements = checkedMultiply(
                static_cast<std::size_t>(max_seq_len),
                static_cast<std::size_t>(hidden_size),
                "LocalTP collective element count overflows size_t");
            const std::size_t backend_bytes = withMargin(
                checkedMultiply(
                    elements,
                    sizeof(float),
                    "LocalTP backend transport bytes overflow size_t"),
                "LocalTP backend transport margin overflows size_t");
            const std::size_t fp16_elements = withMargin(
                elements,
                "LocalTP FP16 scratch elements overflow size_t");
            const std::size_t fp16_bytes = checkedMultiply(
                fp16_elements,
                sizeof(std::uint16_t),
                "LocalTP FP16 scratch bytes overflow size_t");
            if (fp16_bytes >
                std::numeric_limits<std::size_t>::max() - backend_bytes)
            {
                throw std::overflow_error(
                    "LocalTP collective per-device total overflows size_t");
            }
            return {
                .backend_temp_bytes = backend_bytes,
                .fp16_scratch_elements = fp16_elements,
                .fp16_scratch_bytes = fp16_bytes,
            };
        }
    };
} // namespace llaminar2
