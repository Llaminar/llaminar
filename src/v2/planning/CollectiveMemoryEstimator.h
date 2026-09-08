/**
 * @file CollectiveMemoryEstimator.h
 * @brief Exact persistent-memory BOM for rank-local tensor-parallel transport.
 *
 * LocalTP publishes a logical maximum payload to every collective backend and
 * retains an FP16 scratch allocation on homogeneous GPU participants whose
 * transport representation differs from the graph's activation precision.
 * The payload capacity is not itself a physical allocation: NCCL and RCCL use
 * it only as a setup contract, while heterogeneous backends use it to size
 * separately-accounted bridge resources. Capacity admission and
 * RankOrchestrator consume this same pure bill so logical protocol capacity
 * cannot be mistaken for persistent device memory.
 */

#pragma once

#include "config/CollectiveBackendType.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Persistent allocations owned by one LocalTP participant. */
    struct LocalTPCollectiveMemoryBOM
    {
        /** Largest logical backend payload, including the geometry margin. */
        std::size_t backend_payload_capacity_bytes = 0;
        /** Logical FP16 scratch elements, including the fixed margin. */
        std::size_t fp16_scratch_elements = 0;
        /** Physical FP16 scratch bytes allocated on each participant. */
        std::size_t fp16_scratch_bytes = 0;
        /** Persistent graph-boundary control word on each native GPU. */
        std::size_t graph_capture_boundary_bytes = 0;

        /** @return Total persistent bytes retained on one participant. */
        [[nodiscard]] std::size_t perDeviceBytes() const noexcept
        {
            return fp16_scratch_bytes + graph_capture_boundary_bytes;
        }
    };

    /** @brief Canonical sizing authority for LocalTP collective resources. */
    class CollectiveMemoryEstimator final
    {
    public:
        /** Guard retained after the maximum logical FP16 payload. */
        static constexpr std::size_t kFP16ScratchGuardBytes = 4096u;

        /**
         * @brief Convert a logical FP16 element capacity to allocation bytes.
         * @param element_count Maximum number of FP16 transport elements.
         * @return Exact bytes allocated by LocalTP on one GPU participant.
         * @throws std::overflow_error when the allocation does not fit size_t.
         *
         * This function is deliberately shared by admission and allocation;
         * the guard is physical memory and must never live as a runtime-only
         * magic number.
         */
        [[nodiscard]] static std::size_t fp16ScratchAllocationBytes(
            std::size_t element_count)
        {
            constexpr std::size_t element_bytes = sizeof(std::uint16_t);
            if (element_count >
                (std::numeric_limits<std::size_t>::max() -
                 kFP16ScratchGuardBytes) /
                    element_bytes)
            {
                throw std::overflow_error(
                    "LocalTP FP16 scratch allocation overflows size_t");
            }
            return element_count * element_bytes +
                   kFP16ScratchGuardBytes;
        }

        /**
         * @brief Price the exact reservation later installed by LocalTP.
         *
         * The logical backend capacity covers FP32 because graph stages may
         * explicitly request FP32 reduction even when resident activations use
         * a narrower format. It is protocol geometry, not an allocation. FP16
         * scratch is the exact persistent allocation and includes its guard.
         * Both logical capacities use the production ten-percent margin.
         *
         * @param max_seq_len Maximum context rows admitted by the runner.
         * @param hidden_size Model hidden width.
         * @param backend Concrete backend installed for this LocalTP group.
         * @return Per-participant persistent collective BOM.
         * @throws std::invalid_argument for non-positive geometry.
         * @throws std::overflow_error when byte arithmetic exceeds size_t.
         */
        [[nodiscard]] static LocalTPCollectiveMemoryBOM localTP(
            int max_seq_len,
            int hidden_size,
            CollectiveBackendType backend)
        {
            if (max_seq_len <= 0 || hidden_size <= 0)
            {
                throw std::invalid_argument(
                    "LocalTP collective memory requires positive sequence and hidden geometry");
            }
            if (backend == CollectiveBackendType::AUTO)
            {
                throw std::invalid_argument(
                    "LocalTP collective memory requires a resolved backend");
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
            const bool owns_native_gpu_scratch =
                backend == CollectiveBackendType::NCCL ||
                backend == CollectiveBackendType::RCCL;
            const std::size_t fp16_bytes = owns_native_gpu_scratch
                                               ? fp16ScratchAllocationBytes(
                                                     fp16_elements)
                                               : 0u;
            return {
                .backend_payload_capacity_bytes = backend_bytes,
                .fp16_scratch_elements = fp16_elements,
                .fp16_scratch_bytes = fp16_bytes,
                .graph_capture_boundary_bytes =
                    owns_native_gpu_scratch ? sizeof(std::int32_t) : 0u,
            };
        }
    };
} // namespace llaminar2
