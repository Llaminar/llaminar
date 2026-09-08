/**
 * @file DeviceMoEOverlayServiceTelemetryDevice.inl
 * @brief Shared CUDA/HIP kernels for device-local MoE service evidence.
 *
 * The including backend defines `LLAMINAR_MOE_SERVICE_NOW_TICK()` and
 * `LLAMINAR_MOE_SERVICE_TICKS_TO_NS(delta, clock_rate_khz)`.  CUDA uses its
 * global nanosecond timer; HIP uses `wall_clock64()` plus the runtime-reported
 * constant wall-clock rate.  The remainder of the lifecycle and arithmetic is
 * byte-identical across backends.
 */

#include "execution/moe/DeviceMoEOverlayServiceTelemetry.h"

#include <cstddef>
#include <cstdint>

#if !defined(LLAMINAR_MOE_SERVICE_NOW_TICK) || \
    !defined(LLAMINAR_MOE_SERVICE_TICKS_TO_NS)
#error "Backend must define the ExpertOverlay telemetry clock contract"
#endif

namespace llaminar2::moe_overlay_service_device
{
    /** Fixed block width for bounded 256-expert route-count reduction. */
    inline constexpr std::uint32_t kTelemetryThreads = 128u;

    /** @return One inline local-histogram plane from the shared runtime ABI. */
    __device__ __forceinline__ const std::uint64_t *inlineLocalHistogram(
        const std::byte *runtime,
        std::uint32_t phase) noexcept
    {
        std::size_t offset = moe_runtime_abi::kDecodeLocalHistogramOffset;
        if (phase == 1u)
            offset = moe_runtime_abi::kPrefillLocalHistogramOffset;
        else if (phase == 2u)
            offset = moe_runtime_abi::kGroupedVerifierLocalHistogramOffset;
        return reinterpret_cast<const std::uint64_t *>(runtime + offset);
    }

    /**
     * @return The active device-local histogram plane, including async banks.
     *
     * A partially installed external bank is fatal device state. Returning no
     * data would make a Dynamic controller wait forever and is therefore not a
     * safe fallback.
     */
    __device__ __forceinline__ const std::uint64_t *localHistogram(
        const std::byte *runtime,
        std::uint32_t phase) noexcept
    {
        using Bank = moe_runtime_abi::DeviceMoERuntimeHistogramBank;
        auto *const banks = *reinterpret_cast<Bank *const *>(
            runtime + moe_runtime_abi::kRuntimeHistogramBanksOffset);
        auto *const active = *reinterpret_cast<const std::uint32_t *const *>(
            runtime + moe_runtime_abi::kRuntimeHistogramActiveBankOffset);
        if (banks || active)
        {
            if (!banks || !active || phase >= kDeviceMoEOverlayServicePhaseCount)
            {
#if defined(__CUDA_ARCH__)
                asm("trap;");
#else
                __builtin_trap();
#endif
            }
            /*
             * The device scalar is a complete writer state, not a bare bank
             * ordinal.  Its quarantine bit deliberately remains visible to
             * inference so route writers can discard calibration-tail rows.
             * Telemetry is observation-only: it must read the physical bank
             * selected by that state even while row admission is closed.
             */
            const std::uint32_t writer_state = *active;
            if (!moe_runtime_abi::validHistogramWriterState(writer_state))
            {
#if defined(__CUDA_ARCH__)
                asm("trap;");
#else
                __builtin_trap();
#endif
            }
            const std::uint32_t bank =
                moe_runtime_abi::histogramWriterBank(writer_state);
            return banks[bank].local[phase];
        }
        return inlineLocalHistogram(runtime, phase);
    }

    /** @return The device expert-count scratch pointer from the runtime ABI. */
    __device__ __forceinline__ const std::int32_t *expertCounts(
        const std::byte *runtime) noexcept
    {
        return *reinterpret_cast<const std::int32_t *const *>(
            runtime + moe_runtime_abi::kExpertCountsOffset);
    }

    /**
     * @brief Saturating atomic add with sticky overflow evidence.
     * @return True when no overflow occurred.
     */
    __device__ __forceinline__ bool saturatingAtomicAdd(
        std::uint64_t *destination,
        std::uint64_t increment) noexcept
    {
        auto *const address = reinterpret_cast<unsigned long long *>(destination);
        unsigned long long observed = *address;
        while (true)
        {
            const unsigned long long maximum = ~0ull;
            const bool overflow = increment > maximum - observed;
            const unsigned long long desired = overflow
                                                   ? maximum
                                                   : observed + increment;
            const unsigned long long prior = atomicCAS(
                address, observed, desired);
            if (prior == observed)
                return !overflow;
            observed = prior;
        }
    }

    /** Record an exact start timestamp on the inference producer stream. */
    static __global__ void beginServiceTelemetryKernel(
        DeviceMoEOverlayServiceTelemetrySample *sample)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u || !sample)
            return;
        sample->magic = kDeviceMoEOverlayServiceTelemetryMagic;
        sample->version = kDeviceMoEOverlayServiceTelemetryVersion;
        sample->begin_tick = LLAMINAR_MOE_SERVICE_NOW_TICK();
        sample->armed = 1u;
    }

    /**
     * Deterministically reduce locally executed routes and publish one sample.
     * The runtime pointer is byte-addressed so this lightweight translation unit
     * need not include host tensor/SIMD definitions from MoERuntimeTable.h.
     */
    static __global__ void finishServiceTelemetryKernel(
        const void *runtime_layer,
        DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
        DeviceMoEOverlayServiceTelemetrySample *sample,
        std::uint32_t num_experts,
        std::uint32_t phase_hint,
        const MoEOverlayInferenceGraphRole *runtime_graph_role,
        std::uint64_t clock_rate_khz)
    {
        __shared__ std::uint64_t phase_totals
            [kDeviceMoEOverlayServicePhaseCount][kTelemetryThreads];
        __shared__ std::uint64_t grouped_count_totals[kTelemetryThreads];

        const std::uint32_t lane = threadIdx.x;
        if (!runtime_layer || !layer_telemetry || !sample ||
            blockIdx.x != 0u || blockDim.x != kTelemetryThreads ||
            num_experts == 0u ||
            num_experts > moe_runtime_abi::kHistogramMaxExperts)
        {
            return;
        }
        const auto *const runtime = static_cast<const std::byte *>(runtime_layer);
        std::uint64_t grouped_count = 0u;
        for (std::uint32_t phase = 0u;
             phase < kDeviceMoEOverlayServicePhaseCount;
             ++phase)
        {
            phase_totals[phase][lane] = 0u;
        }
        const std::int32_t *const counts = expertCounts(runtime);
        for (std::uint32_t expert = lane;
             expert < num_experts;
             expert += kTelemetryThreads)
        {
            for (std::uint32_t phase = 0u;
                 phase < kDeviceMoEOverlayServicePhaseCount;
                 ++phase)
            {
                phase_totals[phase][lane] +=
                    localHistogram(runtime, phase)[expert];
            }
            if (counts && counts[expert] > 0)
                grouped_count += static_cast<std::uint64_t>(counts[expert]);
        }
        grouped_count_totals[lane] = grouped_count;
        __syncthreads();

        for (std::uint32_t stride = kTelemetryThreads / 2u;
             stride != 0u;
             stride >>= 1u)
        {
            if (lane < stride)
            {
                for (std::uint32_t phase = 0u;
                     phase < kDeviceMoEOverlayServicePhaseCount;
                     ++phase)
                {
                    phase_totals[phase][lane] +=
                        phase_totals[phase][lane + stride];
                }
                grouped_count_totals[lane] +=
                    grouped_count_totals[lane + stride];
            }
            __syncthreads();
        }

        if (lane != 0u)
            return;
        if (sample->magic != kDeviceMoEOverlayServiceTelemetryMagic ||
            sample->version != kDeviceMoEOverlayServiceTelemetryVersion ||
            sample->armed != 1u)
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &layer_telemetry[0].dropped_samples),
                1ull);
            return;
        }
        sample->armed = 0u;

        std::uint64_t deltas[kDeviceMoEOverlayServicePhaseCount] = {};
        std::uint32_t changed_phases = 0u;
        std::uint32_t detected_phase = 0u;
        for (std::uint32_t phase = 0u;
             phase < kDeviceMoEOverlayServicePhaseCount;
             ++phase)
        {
            const std::uint64_t current = phase_totals[phase][0];
            const std::uint64_t previous =
                sample->previous_local_activations[phase];
            deltas[phase] = current >= previous
                                ? current - previous
                                : current;
            sample->previous_local_activations[phase] = current;
            if (deltas[phase] != 0u)
            {
                ++changed_phases;
                detected_phase = phase;
            }
        }

        if (runtime_graph_role)
        {
            phase_hint = static_cast<std::uint32_t>(
                moeOverlayServicePhaseHintForGraphRole(
                    *runtime_graph_role));
        }

        std::uint32_t phase = detected_phase;
        if (phase_hint >= 1u && phase_hint <= 3u)
            phase = phase_hint - 1u;
        else if (changed_phases != 1u)
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &layer_telemetry[0].dropped_samples),
                1ull);
            return;
        }

        std::uint64_t activations = deltas[phase];
        if (activations == 0u)
            activations = grouped_count_totals[0];
        const std::uint64_t end_tick = LLAMINAR_MOE_SERVICE_NOW_TICK();
        if (activations == 0u || end_tick <= sample->begin_tick)
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &layer_telemetry[phase].dropped_samples),
                1ull);
            return;
        }
        const std::uint64_t elapsed_ns =
            LLAMINAR_MOE_SERVICE_TICKS_TO_NS(
                end_tick - sample->begin_tick,
                clock_rate_khz);
        if (elapsed_ns == 0u)
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &layer_telemetry[phase].dropped_samples),
                1ull);
            return;
        }

        auto &cell = layer_telemetry[phase];
        const bool duration_ok = saturatingAtomicAdd(
            &cell.total_nanoseconds, elapsed_ns);
        const bool activations_ok = saturatingAtomicAdd(
            &cell.activation_count, activations);
        const bool samples_ok = saturatingAtomicAdd(
            &cell.sample_count, 1u);
        if (!duration_ok || !activations_ok || !samples_ok)
            atomicExch(&cell.overflowed, 1u);
    }

    /** Copy cumulative device-local totals into one mapped participant record. */
    static __global__ void publishServiceTelemetryKernel(
        const DeviceMoEOverlayServiceTelemetryCell *telemetry,
        const DeviceMoEOverlayServiceTelemetrySample *samples,
        std::uint32_t layer_count,
        std::int32_t participant_id,
        MoEOverlayDeviceServiceTelemetryPublicationHeader *publication)
    {
        if (!telemetry || !samples || !publication || layer_count == 0u ||
            layer_count > 256u || participant_id < 0)
        {
            return;
        }
        const std::size_t cell_count =
            static_cast<std::size_t>(layer_count) *
            kDeviceMoEOverlayServicePhaseCount;
        auto *const destination =
            reinterpret_cast<DeviceMoEOverlayServiceTelemetryCell *>(
                publication + 1);

        /*
         * The mapped publication is a seqlock. Mark it odd before any cell
         * changes so a host reader can never accept a mixture of consecutive
         * snapshots, even if diagnostics inspect it while this graph runs.
         */
        __shared__ std::uint64_t completed_generation;
        if (threadIdx.x == 0u)
        {
            const std::uint64_t previous = publication->generation;
            completed_generation = (previous & 1u) != 0u
                                       ? previous + 1u
                                       : previous + 2u;
            publication->generation = completed_generation - 1u;
            __threadfence_system();
        }
        __syncthreads();
        for (std::size_t cell = threadIdx.x;
             cell < cell_count;
             cell += blockDim.x)
        {
            destination[cell] = telemetry[cell];
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            std::uint32_t valid_sample_count = 0u;
            std::uint32_t armed_sample_count = 0u;
            std::uint64_t begun_sample_count = 0u;
            for (std::uint32_t layer = 0u; layer < layer_count; ++layer)
            {
                const auto &sample = samples[layer];
                valid_sample_count +=
                    sample.magic == kDeviceMoEOverlayServiceTelemetryMagic &&
                            sample.version ==
                                kDeviceMoEOverlayServiceTelemetryVersion
                        ? 1u
                        : 0u;
                armed_sample_count += sample.armed == 1u ? 1u : 0u;
                begun_sample_count += sample.begin_tick != 0u ? 1u : 0u;
            }
            publication->magic = kDeviceMoEOverlayServiceTelemetryMagic;
            publication->version =
                kDeviceMoEOverlayServiceTelemetryVersion;
            publication->participant_id = participant_id;
            publication->layer_count = layer_count;
            publication->phase_count = static_cast<std::uint32_t>(
                kDeviceMoEOverlayServicePhaseCount);
            publication->valid_sample_count = valid_sample_count;
            publication->armed_sample_count = armed_sample_count;
            publication->begun_sample_count = begun_sample_count;
            __threadfence_system();
            publication->generation = completed_generation;
        }
    }
} // namespace llaminar2::moe_overlay_service_device

#undef LLAMINAR_MOE_SERVICE_NOW_TICK
#undef LLAMINAR_MOE_SERVICE_TICKS_TO_NS
