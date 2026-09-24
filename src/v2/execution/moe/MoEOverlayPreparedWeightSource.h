/**
 * @file MoEOverlayPreparedWeightSource.h
 * @brief Typed immutable source view for live ExpertOverlay migration.
 *
 * A migration must read the exact prepared engine retained by the published
 * participant residency bank. CPU and GPU engines expose different physical
 * layouts, but both retain the original GGUF format identity. This resolver
 * validates those facts once and returns one unambiguous source authority; it
 * never recreates weights from a loader or manufactures a host mirror.
 */

#pragma once

#include "ExpertWeightFormat.h"
#include "GPUExpertTransfer.h"
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"

#include <memory>
#include <string>

namespace llaminar2
{
    class ITensorGemm;

    /** @brief Physical prepared representation owned by the retained engine. */
    enum class MoEOverlayPreparedWeightSourceKind
    {
        CpuNativeVnni,
        GpuSeparatedNativeVnni,
        CpuContiguousFloating,
        GpuContiguousFloating,
    };

    /**
     * @brief One validated projection source pinned by shared engine ownership.
     *
     * Exactly one quantized or floating descriptor is active according to
     * `kind`. Every pointer is borrowed from `engine`; retaining this value
     * therefore pins both the physical slot and its bytes.
     */
    struct MoEOverlayPreparedWeightSource
    {
        MoEOverlayPreparedWeightSourceKind kind =
            MoEOverlayPreparedWeightSourceKind::CpuNativeVnni;
        DeviceId device = DeviceId::invalid();
        ExpertWeightFormat format;
        std::shared_ptr<ITensorGemm> engine;
        const cpu::native_vnni::CPUNativeVNNIPackedWeights *cpu_packed =
            nullptr;
        GpuExpertPackedDescriptor gpu_packed;
        ContiguousFloatingPointWeightDescriptor floating;

        /** @return Whether ownership, device, provenance, and layout agree. */
        [[nodiscard]] bool valid(std::string *error = nullptr) const noexcept;
    };

    /**
     * @brief Resolve the single authoritative migration view of a live engine.
     * @param engine Shared prepared engine retained from an immutable RCU bank.
     * @param device Exact participant device that owns @p engine.
     * @param output Receives a lifetime-pinned CPU or GPU source view.
     * @param error Optional exact validation diagnostic.
     * @return True only for a complete NativeVNNI or contiguous floating source.
     */
    bool resolveMoEOverlayPreparedWeightSource(
        std::shared_ptr<ITensorGemm> engine,
        DeviceId device,
        MoEOverlayPreparedWeightSource &output,
        std::string *error = nullptr) noexcept;
} // namespace llaminar2
