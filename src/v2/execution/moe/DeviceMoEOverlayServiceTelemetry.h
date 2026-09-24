/**
 * @file DeviceMoEOverlayServiceTelemetry.h
 * @brief Fixed-width device-local and mapped service-timing records for MoE.
 *
 * Dynamic ExpertOverlay policy needs a measured cost for executing one routed
 * activation on every participant and model-layer equivalence class.  GPU
 * inference accumulates those observations in device-local memory so ordinary
 * layer execution never writes a PCIe/UPI mapped page.  A finite maintenance
 * graph later copies the cumulative cells into one participant-owned mapped
 * publication and performs a system-release generation store.  Host evidence
 * composition may inspect that immutable snapshot, but it cannot express an
 * owner map, movement command, or placement epoch through this ABI.
 */
#pragma once

#include "DeviceMoERuntimeABI.h"
#include "MoEOverlayActivationEpochABI.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_SERVICE_HD __host__ __device__
#else
#define LLAMINAR_MOE_SERVICE_HD
#endif

namespace llaminar2
{
    struct DeviceMoELayerRuntime;

    /**
     * @brief Optional semantic hint for one routed-expert service sample.
     *
     * Auto detects the phase from device histogram deltas. Explicit values are
     * used by graph families whose semantic identity is immutable at capture.
     */
    enum class MoEOverlayServicePhaseHint : std::uint32_t
    {
        Auto = 0u,
        Decode = 1u,
        Prefill = 2u,
        GroupedVerifier = 3u,
    };

    /**
     * @brief Translate an authenticated inference graph role to service phase.
     *
     * The third service plane means routed work inside an MTP transaction. On
     * ordinary model layers that is grouped-verifier execution; on retained
     * predictor layers it is draft execution. The per-layer production
     * topology keeps those costs distinct without adding an ambiguous global
     * "decode" alias or asking histogram deltas to infer graph identity.
     */
    [[nodiscard]] LLAMINAR_MOE_SERVICE_HD constexpr
    MoEOverlayServicePhaseHint
    moeOverlayServicePhaseHintForGraphRole(
        MoEOverlayInferenceGraphRole role) noexcept
    {
        switch (role)
        {
        case MoEOverlayInferenceGraphRole::MainPrefill:
            return MoEOverlayServicePhaseHint::Prefill;
        case MoEOverlayInferenceGraphRole::MainDecode:
            return MoEOverlayServicePhaseHint::Decode;
        case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
        case MoEOverlayInferenceGraphRole::MTPDraft:
            return MoEOverlayServicePhaseHint::GroupedVerifier;
        case MoEOverlayInferenceGraphRole::None:
            return MoEOverlayServicePhaseHint::Auto;
        }
        return MoEOverlayServicePhaseHint::Auto;
    }

    /** Binary identity of a device-local service sample (`MOST`). */
    inline constexpr std::uint32_t kDeviceMoEOverlayServiceTelemetryMagic =
        0x54534f4du;

    /** Version shared by host, CUDA, and ROCm telemetry code. */
    inline constexpr std::uint32_t kDeviceMoEOverlayServiceTelemetryVersion =
        1u;

    /** Number of decode, prefill, and grouped-verifier timing planes. */
    inline constexpr std::size_t kDeviceMoEOverlayServicePhaseCount =
        moe_runtime_abi::kHistogramSourceCount;

    /**
     * @brief Cumulative service evidence for one layer and semantic phase.
     *
     * All values are written by device kernels.  Totals use saturating atomic
     * addition; @ref overflowed is sticky and makes certification fail rather
     * than allowing wrapped evidence to look economical.  A snapshot copies
     * the complete cell only after the inference terminal that produced it has
     * completed, so no host lock or inference-stream synchronization is needed.
     */
    struct alignas(32) DeviceMoEOverlayServiceTelemetryCell
    {
        std::uint64_t total_nanoseconds = 0u;
        std::uint64_t activation_count = 0u;
        std::uint64_t sample_count = 0u;
        std::uint64_t dropped_samples = 0u;
        std::uint32_t overflowed = 0u;
        std::uint32_t reserved[7] = {};
    };

    /**
     * @brief One graph-local timing cursor surrounding a routed-expert stage.
     *
     * A graph workspace owns this record at a stable address.  Retained graph
     * families execute serially against that workspace, so the start timestamp
     * cannot be overwritten by another replay.  Per-phase cumulative route
     * baselines let the finish kernel identify whether a shape-one transaction
     * was decode or a one-row prefill without consulting mutable host state.
     */
    struct alignas(64) DeviceMoEOverlayServiceTelemetrySample
    {
        std::uint32_t magic = kDeviceMoEOverlayServiceTelemetryMagic;
        std::uint32_t version =
            kDeviceMoEOverlayServiceTelemetryVersion;
        std::uint32_t armed = 0u;
        std::uint32_t reserved0 = 0u;
        std::uint64_t begin_tick = 0u;
        std::uint64_t previous_local_activations
            [kDeviceMoEOverlayServicePhaseCount] = {};
        std::uint64_t reserved[2] = {};
    };

    /**
     * @brief Exact device-local capability needed to observe one MoE layer.
     *
     * This is deliberately narrower than an @c IMoERuntimeTable pointer. A
     * retained sparse endpoint gets its executable expert set from an immutable
     * residency bank and must not gain a second placement authority merely to
     * report service time. The binding lends only the route-count source, this
     * layer's cumulative destination cells, and its serial timing cursor.
     * Every pointer is model-lifetime storage owned by the canonical runtime
     * table; consumers neither allocate nor publish through this view.
     */
    struct DeviceMoEOverlayServiceTelemetryBinding
    {
        /** Device-local route counters used to price executed activations. */
        DeviceMoELayerRuntime *runtime_layer = nullptr;
        /** First decode/prefill/grouped-verifier accumulator for this layer. */
        DeviceMoEOverlayServiceTelemetryCell *layer_telemetry = nullptr;
        /** Serial graph-family timing cursor for this layer. */
        DeviceMoEOverlayServiceTelemetrySample *sample = nullptr;

        /** @return Whether telemetry is deliberately absent. */
        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return !runtime_layer && !layer_telemetry && !sample;
        }

        /** @return Whether all three device-local observation addresses exist. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return runtime_layer && layer_telemetry && sample;
        }

        /** @return Whether two bindings name the same model-lifetime storage. */
        [[nodiscard]] constexpr bool operator==(
            const DeviceMoEOverlayServiceTelemetryBinding &other) const
            noexcept
        {
            return runtime_layer == other.runtime_layer &&
                   layer_telemetry == other.layer_telemetry &&
                   sample == other.sample;
        }
    };

    /**
     * @brief Release-published header for one participant's mapped snapshot.
     *
     * The publisher writes every following cell, performs a system fence, and
     * publishes an odd @ref generation while copying, then system-release
     * publishes the next even generation. Readers accept only one unchanged,
     * nonzero even generation before and after copying the cells.
     */
    struct alignas(64) MoEOverlayDeviceServiceTelemetryPublicationHeader
    {
        std::uint32_t magic = kDeviceMoEOverlayServiceTelemetryMagic;
        std::uint32_t version =
            kDeviceMoEOverlayServiceTelemetryVersion;
        std::int32_t participant_id = -1;
        std::uint32_t layer_count = 0u;
        std::uint32_t phase_count = static_cast<std::uint32_t>(
            kDeviceMoEOverlayServicePhaseCount);
        std::uint32_t reserved0 = 0u;
        std::uint64_t generation = 0u;
        /** Layer cursors carrying the expected magic/version at publication. */
        std::uint32_t valid_sample_count = 0u;
        /** Layer cursors still armed after the joined inference terminal. */
        std::uint32_t armed_sample_count = 0u;
        /** Layer cursors whose begin marker has executed at least once. */
        std::uint64_t begun_sample_count = 0u;
        std::uint64_t reserved[2] = {};
    };

    /** @return Number of cells in one complete participant snapshot. */
    [[nodiscard]] constexpr std::size_t
    deviceMoEOverlayServiceTelemetryCellCount(
        std::size_t layer_count) noexcept
    {
        return layer_count * kDeviceMoEOverlayServicePhaseCount;
    }

    /** @return Raw bytes in one complete participant publication. */
    [[nodiscard]] constexpr std::size_t
    deviceMoEOverlayServiceTelemetryPublicationBytes(
        std::size_t layer_count) noexcept
    {
        return sizeof(MoEOverlayDeviceServiceTelemetryPublicationHeader) +
               deviceMoEOverlayServiceTelemetryCellCount(layer_count) *
                   sizeof(DeviceMoEOverlayServiceTelemetryCell);
    }

    /** @return First cell immediately following a mapped publication header. */
    [[nodiscard]] inline DeviceMoEOverlayServiceTelemetryCell *
    deviceMoEOverlayServiceTelemetryCells(
        MoEOverlayDeviceServiceTelemetryPublicationHeader *publication) noexcept
    {
        return publication
                   ? reinterpret_cast<DeviceMoEOverlayServiceTelemetryCell *>(
                         publication + 1)
                   : nullptr;
    }

    /** @copydoc deviceMoEOverlayServiceTelemetryCells */
    [[nodiscard]] inline const DeviceMoEOverlayServiceTelemetryCell *
    deviceMoEOverlayServiceTelemetryCells(
        const MoEOverlayDeviceServiceTelemetryPublicationHeader
            *publication) noexcept
    {
        return publication
                   ? reinterpret_cast<
                         const DeviceMoEOverlayServiceTelemetryCell *>(
                         publication + 1)
                   : nullptr;
    }

    static_assert(std::is_trivially_copyable_v<
                  DeviceMoEOverlayServiceTelemetryCell>);
    static_assert(std::is_trivially_copyable_v<
                  DeviceMoEOverlayServiceTelemetrySample>);
    static_assert(std::is_trivially_copyable_v<
                  DeviceMoEOverlayServiceTelemetryBinding>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceServiceTelemetryPublicationHeader>);
    static_assert(sizeof(DeviceMoEOverlayServiceTelemetryCell) == 64u);
    static_assert(sizeof(DeviceMoEOverlayServiceTelemetrySample) == 64u);
    static_assert(
        sizeof(MoEOverlayDeviceServiceTelemetryPublicationHeader) == 64u);
} // namespace llaminar2

#undef LLAMINAR_MOE_SERVICE_HD
