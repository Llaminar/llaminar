/**
 * @file PlanningGPUClockBuffer.h
 * @brief PMA-owned graph-stable device clock slots for bounded startup observations.
 *
 * Device stamps are instrumentation, not inference state. Only the final small
 * timestamp bank is copied through TransferEngine after the measured graph;
 * original worker/stream ordering owns visibility and retirement.
 */
#pragma once
#include "backends/DeviceId.h"
#include "backends/ExplicitGPUStream.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <cstdint>
#include <memory>
#include <span>

namespace llaminar2
{
    class IWorkerGPUContext;
    /** @brief Typed original clock storage; implementations never compare clocks across devices. */
    class PlanningGPUClockBuffer
    {
    public:
        virtual ~PlanningGPUClockBuffer() = default;
        /** @return Immutable count of uint64 timestamp slots. */
        virtual size_t slotCount() const noexcept = 0;
        /** @return Positive admitted fixed clock frequency, never a shader/DVFS clock. */
        virtual std::uint64_t ticksPerSecond() const noexcept = 0;
        /** @brief Capture a timestamp write to the exact slot on the original non-null stream. */
        virtual void record(size_t slot, ExplicitGPUStream stream) = 0;
        /** @brief Enqueue the complete compact timestamp bank after the timed graph on its exact stream. */
        virtual void enqueueReadback(ExplicitGPUStream stream) = 0;
        /** @return Final host ticks only after readback completion; premature observations fail. */
        virtual std::span<const std::uint64_t> ticks() const = 0;
    };

    /** @brief Native fixed-clock slots and their exact PMA-owned host/device allocations. */
    class PlanningNativeGPUClockBuffer final : public PlanningGPUClockBuffer
    {
    public:
        /** @brief Claim all backing before materializing exact-worker/stream clock buffers. */
        PlanningNativeGPUClockBuffer(IWorkerGPUContext &worker, DeviceId device, ExplicitGPUStream stream,
            size_t slots, const std::shared_ptr<PhysicalMemoryAuthority> &memory);
        /** @brief Release backend buffers before their original physical allocation claims. */
        ~PlanningNativeGPUClockBuffer() override;
        /** @return Page-rounded host backing contributed before original PMA admission. */
        static size_t hostAllocationBytes(size_t slots);
        /** @return Exact positive device bank extent; overflow is rejected before allocation. */
        static size_t deviceAllocationBytes(size_t slots);
        size_t slotCount() const noexcept override;
        std::uint64_t ticksPerSecond() const noexcept override;
        void record(size_t slot, ExplicitGPUStream stream) override;
        void enqueueReadback(ExplicitGPUStream stream) override;
        std::span<const std::uint64_t> ticks() const override;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
