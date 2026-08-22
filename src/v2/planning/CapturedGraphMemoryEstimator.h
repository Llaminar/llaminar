/**
 * @file CapturedGraphMemoryEstimator.h
 * @brief Admission envelope for opaque CUDA/HIP graph-executable storage.
 *
 * Native graph drivers retain internal node and executable metadata outside
 * Llaminar's tensor/workspace arenas. The bytes are not allocator-addressable,
 * but they are physically resident and scale with model layers and retained
 * identities. This estimator makes that otherwise invisible storage an
 * explicit capacity term before expert placement fills the device.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** Base driver allocation observed for one complete executable identity. */
    inline constexpr std::size_t kCapturedGraphBaseBytesPerExecutable =
        2ULL * 1024ULL * 1024ULL;

    /**
     * Per-layer envelope covering captured nodes and backend executable state.
     *
     * Production CUDA/ROCm graph families contain several kernel/memcpy/event
     * nodes per transformer layer. Half a MiB per layer bounds the measured
     * complete Qwen35 MoE graph cost while remaining proportional for smaller
     * and larger model graphs.
     */
    inline constexpr std::size_t kCapturedGraphBytesPerModelLayer =
        512ULL * 1024ULL;

    /**
     * @brief Estimate persistent native-driver bytes for retained executables.
     * @param model_layer_count Number of raw model graph layers.
     * @param executable_count Simultaneously retained graph identities.
     * @return Checked conservative byte envelope.
     * @throws std::invalid_argument for non-positive model geometry.
     * @throws std::overflow_error when the result exceeds size_t.
     */
    [[nodiscard]] inline std::size_t
    estimateCapturedGraphExecutableBytes(
        int model_layer_count,
        std::size_t executable_count)
    {
        if (model_layer_count <= 0)
        {
            throw std::invalid_argument(
                "Captured graph memory estimation requires positive model layers");
        }
        if (executable_count == 0u)
            return 0u;
        const std::size_t layers =
            static_cast<std::size_t>(model_layer_count);
        if (layers >
            (std::numeric_limits<std::size_t>::max() -
             kCapturedGraphBaseBytesPerExecutable) /
                kCapturedGraphBytesPerModelLayer)
        {
            throw std::overflow_error(
                "Captured graph per-executable memory overflows size_t");
        }
        const std::size_t per_executable =
            kCapturedGraphBaseBytesPerExecutable +
            layers * kCapturedGraphBytesPerModelLayer;
        if (executable_count >
            std::numeric_limits<std::size_t>::max() / per_executable)
        {
            throw std::overflow_error(
                "Captured graph retained-family memory overflows size_t");
        }
        return executable_count * per_executable;
    }
} // namespace llaminar2
