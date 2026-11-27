/**
 * @file MoEExpertGPUWeightPlan.h
 * @brief Initial expert pool planning on the loader's canonical prepared layout.
 *
 * The service and its device-free allocation contract enter the same planner.
 * Coded metadata belongs to the original operand, not a local expert estimate;
 * the loader declaration remains the sole representation decision.
 */
#pragma once

#include "loaders/BoundedGPUExpertSource.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"

namespace llaminar2
{
    /**
     * @brief Plan one initial expert operand without materializing its source.
     * @param orchestrator Existing owner of the admitted device pool.
     * @param device Exact target backend and ordinal.
     * @param name Stable expert/projection slot identity.
     * @param source Original matrix supplying format, shape and superblock phase.
     * @param raw_bytes Original encoded extent used by the bounded upload ring.
     * @param layout Physical representation admitted for this pool.
     * @throws std::invalid_argument If the operand has no native quantized format.
     */
    inline void planMoEExpertGPUWeight(LoadOrchestrator &orchestrator, DeviceId device,
        const std::string &name, const TensorBase &source, size_t raw_bytes,
        GPUPreparedWeightPoolLayout layout)
    {
        const auto *format = gpuWeightSourceFormat(source);
        if (!format)
            throw std::invalid_argument("GPU expert pool plan requires a native quantized source");
        const auto physical = gpuPreparedWeightPoolAllocation(*format, layout);
        orchestrator.planWeight(device.ordinal, name,
            static_cast<int>(source.rows()), static_cast<int>(source.cols()),
            physical.payload_bytes_per_block, physical.has_mins, physical.has_emins,
            nativeQuantisedScalePlaneWidth(canonicalDeviceVnniCodebookId(*format)),
            raw_bytes, {0}, gpuPreparedWeightCodedDeclaration(source, layout, device));
    }
}
