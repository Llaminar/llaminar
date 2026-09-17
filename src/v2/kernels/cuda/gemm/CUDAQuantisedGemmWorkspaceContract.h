/**
 * @file CUDAQuantisedGemmWorkspaceContract.h
 * @brief Shared CUDA quantized projection scratch contract before weight allocation.
 *
 * Prepared engines and source-backed planning samples call the same named-buffer
 * composition. Geometry is independent of weight addresses. Native prefill
 * dispatch remains device-specific, so callers query on the actual device's
 * worker; this is not a device-free estimate for an unobserved remote GPU.
 */
#pragma once
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include <cstdint>
#include <optional>
#include <span>

namespace llaminar2::cuda::quantized_gemm_workspace
{
    /** @brief Execution bytes and arithmetic provenance are distinct after expert migration. */
    struct NativeCodebooks final
    {
        uint8_t execution;
        uint8_t arithmetic;
    };

    /**
     * @brief Declare exact runtime scratch without constructing a prepared engine.
     * @param m Maximum execution rows for this serial family.
     * @param n Exact output width.
     * @param k Exact input width.
     * @param device_ordinal Actual local CUDA device selecting the launch policy.
     * @param native Native codebooks, absent only for the non-native INT8 adapter.
     * @return Named buffers consumed unchanged by runtime workspace binding.
     */
    WorkspaceRequirements projectionRequirements(int m, int n, int k, int device_ordinal,
        std::optional<NativeCodebooks> native);

    /** @brief Add disjoint verifier partials for simultaneously executing projections. */
    void appendFusedProjectionRequirements(WorkspaceRequirements &requirements, int m,
        std::span<const int> projection_columns, int k);
}
