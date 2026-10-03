/**
 * @file SharedExpertColumnGateKernels.cpp
 * @brief Exact-stream dispatch and checked extents for shared-output column gating.
 *
 * Existing CUDA/HIP gate kernels supply the arithmetic. This interface only
 * authenticates disjoint borrowed ranges and selects the declared backend;
 * it cannot allocate, transfer, synchronize or select a different precision.
 */
#include "SharedExpertColumnGateKernels.h"
#include <array>
#include <cstdint>
#include <limits>

#ifdef HAVE_CUDA
extern "C" bool cudaMoE_shared_expert_column_gate_add(const float *, const float *, float *,
    const float *, float *, int, int, int, const int *, int, void *);
#endif
#ifdef HAVE_ROCM
extern "C" bool hipMoE_shared_expert_column_gate_add(const float *, const float *, float *,
    const float *, float *, int, int, int, const int *, int, void *);
#endif

namespace llaminar2
{
    bool gateSharedExpertColumnsFP32(DeviceId device, const SharedExpertColumnGate &p, void *stream)
    {
        if (!device.is_gpu() || !stream || p.rows <= 0 || p.model_columns <= 0 ||
            p.local_columns <= 0 || p.local_columns > p.model_columns) return false;
        constexpr auto limit = std::numeric_limits<std::uintptr_t>::max();
        if (std::size_t(p.rows) > limit / sizeof(float) / p.model_columns) return false;
        const auto input_bytes = std::size_t(p.rows) * p.model_columns * sizeof(float);
        const auto output_bytes = std::size_t(p.rows) * p.local_columns * sizeof(float);
        const std::array pointers{reinterpret_cast<std::uintptr_t>(p.input), reinterpret_cast<std::uintptr_t>(p.gate),
            reinterpret_cast<std::uintptr_t>(p.shared), reinterpret_cast<std::uintptr_t>(p.routed),
            reinterpret_cast<std::uintptr_t>(p.combined)};
        const std::array sizes{input_bytes, std::size_t(p.model_columns) * sizeof(float), output_bytes, output_bytes, output_bytes};
        for (std::size_t i = 0; i < pointers.size(); ++i)
        {
            // Both backends use float4 only for a four-column-aligned stride.
            // Authenticate the corresponding base alignment as well as extent.
            const auto columns = i < 2 ? p.model_columns : p.local_columns;
            const auto alignment = columns % 4 == 0 ? 4 * alignof(float) : alignof(float);
            if (!pointers[i] || pointers[i] % alignment || pointers[i] > limit - sizes[i]) return false;
            for (std::size_t j = 0; j < i; ++j)
                if (pointers[i] < pointers[j] + sizes[j] && pointers[j] < pointers[i] + sizes[i]) return false;
        }
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cudaMoE_shared_expert_column_gate_add(p.input, p.gate, p.shared,
            p.routed, p.combined, p.rows, p.model_columns, p.local_columns, p.active_rows, device.ordinal, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return hipMoE_shared_expert_column_gate_add(p.input, p.gate, p.shared,
            p.routed, p.combined, p.rows, p.model_columns, p.local_columns, p.active_rows, device.ordinal, stream);
#endif
        return false;
    }
}
